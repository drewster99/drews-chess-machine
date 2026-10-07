import Foundation

/// Plays one Lichess game, from its game stream to a finished status
/// (plan §6, §8, §12.4, §20).
///
/// **Stream handling.** The session holds the game's NDJSON stream. The
/// first line of every (re)connection is `gameFull`, a complete resync.
/// Only a finished status ends the game; a stream that ends, stalls or
/// fails for any other reason is reopened (plan §6, lichess-bot #1184).
///
/// **Position.** `LichessBotPositionTracker` rebuilds the position from the
/// server's move list; a list that fails to replay is a divergence, answered
/// by reopening the stream for a fresh `gameFull` — never by moving from a
/// position DCM can't vouch for (E5).
///
/// **Moving, and actor reentrancy.** While the session awaits inference or
/// a POST, other calls can run on the actor — above all a held move's own
/// task, which posts while the stream keeps being read. So before every POST
/// attempt, the first and each retry, it re-checks that it is still our turn
/// at the ply it decided for, in the position it decided from (its FEN: a
/// takeback and a different reply can return to the same ply in a new
/// position), and that nothing has been posted for this ply. The ply is
/// recorded as posted before the POST goes out, so a second decision for the
/// same ply can never be sent. A POST already waiting inside the request gate
/// goes out without a further check; so while one may be pending, a takeback
/// (the only way the position at its ply can legitimately change) is not
/// accepted.
///
/// **Requests.** Every call goes through the account-wide gate via the API.
/// A 429 means waiting out the cooldown (the gate does that) and posting
/// again only if it is still our turn at the same ply, in the same position
/// (§5.4). A 400 on a
/// move means state disagreement: reopen the stream, never resend blindly
/// (§5.5).
actor LichessBotGameSession {

    let gameID: String
    private let ourAccountID: String
    private let api: any LichessBotGameAPI
    private let observer: any LichessBotGameObserver
    private let time: any LichessBotTimeSource
    private let settingsProvider: @Sendable () async -> LichessBotSettings
    /// The newest model generation already built. Read while our clock runs,
    /// so it never builds one.
    private let latestMoveSource: @Sendable () async -> any LichessBotMoveSource
    private let onTurnStatus: @Sendable (String, LichessBotTurnStatus) async -> Void
    /// The generation playing this game: the one it started with, or, for a
    /// live-trainer game with mid-game refresh on, the newest snapshot since.
    /// Chat reports this one.
    private var playingMoveSource: any LichessBotMoveSource

    private var gameFull: LichessBotGameFull?
    private var ourColor: LichessBotColorName?
    private var tracker: LichessBotPositionTracker?
    private var lastPostedPly: Int?
    private var readings: [LichessBotValueReading] = []
    private var takebacksAccepted = 0
    private var greeted = false
    private var farewellSent = false
    private var commandBudget = LichessBotChatCommandBudget()
    /// Our clock as of the latest `gameState`, for the command budget's
    /// low-clock rule.
    private var ourClockMilliseconds: Int?
    private var finished = false
    /// This game is one DCM can't play; it is never tracked.
    private var isUnplayable = false
    /// Lichess accepted our abort or resignation of the unplayable game.
    /// Set only after success: a failure leaves it false, so the next
    /// `gameFull` (after the stream reopens) tries again.
    private var unplayableResolved = false
    private var pendingClaim: Task<Void, Never>?
    /// A chat-command reply waiting to be sent.
    private struct PendingCommandReply: Sendable {
        let command: LichessBotChatCommand
        let username: String
        let room: LichessBotChatRoom
        let texts: [String]
    }
    /// Command replies not yet sent, oldest first.
    private var pendingCommandReplies: [PendingCommandReply] = []
    /// Sends `pendingCommandReplies` in order, off the stream loop: each
    /// reply is a request through the shared gate, and awaiting it in the
    /// loop would hold up reading the next line, and so our next move.
    private var commandReplySender: Task<Void, Never>?
    /// The operator's move pacing for this game (plan §14.3c).
    private let pacing: @Sendable () async -> LichessBotMovePacingSnapshot
    /// A decided move being held or delayed. It posts from its own task so
    /// the stream keeps being read (an opponent's resignation, a flag, a
    /// takeback) while it waits.
    private var pacedMove: Task<Void, Never>?
    private var pacedMovePly: Int?
    /// The position the held move was decided in (its FEN). A takeback and
    /// a different reply can return to the same ply in a new position; the
    /// held move is then dropped rather than played there.
    private var pacedMovePosition: String?
    /// The held move has been released and its POST may be in flight: it is
    /// no longer dropped when the position moves (its own echo moves it).
    private var pacedMoveReleased = false
    /// When `ourClockMilliseconds` was received, to run it down locally.
    private var ourClockReceivedAt: Duration?
    private var lastMoveRoundTripMilliseconds: Double?
    private var keepAlivesSeen = false
    private var loggedDrawDisagreementPly: Int?
    /// A local draw rule seen while Lichess said "started", held until the
    /// next `gameState`: Lichess sends a game-ending move with status
    /// `started` and the result a moment later in a second message, so the
    /// two only disagree if the game is *still* live in the next one.
    private var pendingDrawDisagreement: String?
    /// Resyncs in a row without the game advancing; each waits longer.
    private var consecutiveResyncs = 0
    /// Of those, the ones caused by a disagreement with Lichess; enough of
    /// them and the session stops moving.
    private var consecutiveDisagreements = 0
    private var highestPlySeen = 0
    /// Rejected move POSTs per ply (plan §6.1 B, §5.5).
    private var rejectionsByPly: [Int: Int] = [:]
    /// The move this client tried to post at each ply, normalized UCI.
    private var attemptedPosts: [Int: String] = [:]
    /// Moves arriving after the first `gameFull` of this session are
    /// checked against `attemptedPosts`; the first one may carry moves this
    /// client made before a relaunch.
    private var checksForeignMoves = false
    private var movingStopped = false
    private static let maximumRejectionsPerPly = 3
    private static let maximumConsecutiveResyncs = 8
    private static let resyncBackoff = LichessBotBackoff(initial: .seconds(1), multiplier: 2, cap: .seconds(30))
    private let stallTimeout = SyncBox<Duration?>(nil)
    /// Set from outside the stream loop to make it drop its connection and
    /// reopen it for a fresh `gameFull`; counted like a thrown
    /// `ResyncNeeded`.
    private let resyncRequest = SyncBox<ResyncNeeded?>(nil)

    /// Raised inside the stream loop to drop the connection and reopen it
    /// for a fresh `gameFull`.
    private struct ResyncNeeded: Error {
        let reason: String
        /// Lichess and this client disagree about the game. False when
        /// Lichess itself is failing (5xx, network): that is waited out
        /// with backoff, never counted toward giving up on the game.
        let isDisagreement: Bool
    }

    init(
        gameID: String,
        ourAccountID: String,
        api: any LichessBotGameAPI,
        moveSource: any LichessBotMoveSource,
        latestMoveSource: @escaping @Sendable () async -> any LichessBotMoveSource,
        settingsProvider: @escaping @Sendable () async -> LichessBotSettings,
        observer: any LichessBotGameObserver,
        time: any LichessBotTimeSource,
        onTurnStatus: @escaping @Sendable (String, LichessBotTurnStatus) async -> Void,
        pacing: @escaping @Sendable () async -> LichessBotMovePacingSnapshot = { LichessBotMovePacingSnapshot() },
        carryover: LichessBotGameSessionCarryover
    ) {
        self.gameID = gameID
        self.ourAccountID = ourAccountID
        self.api = api
        self.playingMoveSource = moveSource
        self.latestMoveSource = latestMoveSource
        self.settingsProvider = settingsProvider
        self.observer = observer
        self.time = time
        self.onTurnStatus = onTurnStatus
        self.pacing = pacing
        self.greeted = carryover.greeted
        self.farewellSent = carryover.farewellSent
        self.takebacksAccepted = carryover.takebacksAccepted
        self.readings = carryover.readings
        self.commandBudget = LichessBotChatCommandBudget(repliesSent: carryover.commandRepliesQueued)
    }

    var isFinished: Bool {
        finished
    }

    /// Drop the current game-stream connection and reopen it. Used when the
    /// event stream reports `gameFinish`: Lichess may send no final
    /// `gameState` on a resignation (plan E22), and the `gameFull` of a
    /// reopened stream carries the finished status. Takes effect at the
    /// stream watchdog's next check.
    nonisolated func requestResync(reason: String) {
        requestResync(ResyncNeeded(reason: reason, isDisagreement: false))
    }

    /// Queue a resync for the stream loop's next watchdog check, counted like
    /// a thrown `ResyncNeeded`. A pending disagreement is never downgraded by
    /// a later request that is not one.
    private nonisolated func requestResync(_ resync: ResyncNeeded) {
        resyncRequest.modify { pending in
            let disagreement = resync.isDisagreement || pending?.isDisagreement == true
            pending = ResyncNeeded(reason: resync.reason, isDisagreement: disagreement)
        }
    }

    /// For a followed lineage, whether `latest`'s file is later work than
    /// `playing`'s on the same branch. A higher generation ID alone is not:
    /// after the playing file is deleted and the source is switched away
    /// and back, the lineage's newest remaining file — an older one — is
    /// built as a new generation. Without a lineage (the live trainer) the
    /// generation ID is the order.
    nonisolated static func isLaterInItsLineage(_ latest: LichessBotGenerationInfo, than playing: LichessBotGenerationInfo) -> Bool {
        switch (latest.lineage, playing.lineage) {
        case (nil, nil):
            return true
        case let (latestLineage?, playingLineage?):
            return latestLineage.rank.isOnTheSameBranch(as: playingLineage.rank) && latestLineage.rank.isAbove(playingLineage.rank)
        case (.some, nil), (nil, .some):
            return false
        }
    }

    // MARK: - Running

    /// Play until the game finishes, the task is cancelled, or the gate
    /// closes (going offline). Returns normally in every case; everything
    /// notable is reported to the observer.
    func run() async {
        var attempt = 0
        while !finished && !Task.isCancelled {
            let settings = await settingsProvider()
            do {
                let chunks = try await api.openGameStream(gameID: gameID)
                await observer.gameEvent(gameID: gameID, .streamOpened(attempt: attempt))
                updateStallTimeout(settings: settings.connection)
                let stallBox = stallTimeout
                let resyncBox = resyncRequest
                let items = LichessBotStreamReader.items(
                    from: chunks,
                    time: time,
                    stallTimeout: { resyncBox.value != nil ? .zero : stallBox.value },
                    checkInterval: .seconds(1)
                )
                for try await item in items {
                    // Settings are read fresh for every line, so a change
                    // in Settings reaches games already in progress; one
                    // line's handling uses one consistent copy.
                    let current = await settingsProvider()
                    updateStallTimeout(settings: current.connection)
                    try await handle(item, settings: current)
                    if finished { break }
                }
                if !finished {
                    await observer.gameEvent(gameID: gameID, .streamEnded(reason: "server closed the stream"))
                }
                attempt = 0
            } catch let resync as ResyncNeeded {
                attempt = 0
                // A request queued meanwhile is satisfied by this reopen; left
                // pending, it would force a second, counted reopen at once.
                let queued = resyncRequest.mutate { request -> ResyncNeeded? in
                    defer { request = nil }
                    return request
                }
                let merged = ResyncNeeded(reason: resync.reason, isDisagreement: resync.isDisagreement || queued?.isDisagreement == true)
                guard await continueAfterResync(merged) else { break }
                continue
            } catch LichessBotStreamError.stalled(let silence) {
                if let requested = resyncRequest.mutate({ request -> ResyncNeeded? in
                    defer { request = nil }
                    return request
                }) {
                    attempt = 0
                    guard await continueAfterResync(requested) else { break }
                    continue
                }
                await observer.gameEvent(gameID: gameID, .streamEnded(reason: LichessBotStreamError.stalled(silence: silence).localizedDescription))
            } catch is CancellationError {
                break
            } catch LichessBotGateError.closed(let reason) {
                await observer.gameEvent(gameID: gameID, .streamEnded(reason: "request gate closed: \(reason)"))
                break
            } catch let error as LichessBotAPIError {
                await observer.gameEvent(gameID: gameID, .streamEnded(reason: error.localizedDescription))
                if case .unauthorized = error {
                    await observer.gameEvent(gameID: gameID, .tokenRejected(error.localizedDescription))
                    break
                }
                if case .http(404, _) = error {
                    // The game no longer exists as a playable game; the
                    // export API settles how it ended (plan §10.2).
                    break
                }
            } catch {
                await observer.gameEvent(gameID: gameID, .streamEnded(reason: String(describing: error)))
            }
            if finished || Task.isCancelled { break }
            let backoff = LichessBotBackoff(
                initial: .seconds(settings.connection.reconnectInitialSeconds),
                multiplier: 2,
                cap: .seconds(settings.connection.reconnectCapSeconds)
            )
            do {
                try await time.sleep(for: backoff.delay(attempt: attempt, unitRandom: Double.random(in: 0...1)))
            } catch {
                break
            }
            attempt += 1
        }
        pendingClaim?.cancel()
        pacedMove?.cancel()
        await stopCommandReplies(reason: "the game session stopped")
        await onTurnStatus(gameID, LichessBotTurnStatus(awaitingOurMove: false, ourClock: nil))
    }

    /// Count a resync (thrown in the loop or requested from elsewhere),
    /// report it, and back off on repeats; false when the loop should end
    /// (too many disagreements in a row, or cancelled while backing off).
    private func continueAfterResync(_ resync: ResyncNeeded) async -> Bool {
        await observer.gameEvent(gameID: gameID, .streamEnded(reason: "resync: \(resync.reason)"))
        consecutiveResyncs += 1
        if resync.isDisagreement {
            consecutiveDisagreements += 1
        }
        if consecutiveDisagreements >= Self.maximumConsecutiveResyncs {
            await stopMoving("\(consecutiveDisagreements) resyncs in a row without the game advancing (last: \(resync.reason))")
            return false
        }
        // The first resync is immediate; repeats back off, so a
        // disagreement that persists can never turn into a request storm.
        if consecutiveResyncs > 1 {
            do {
                try await time.sleep(for: Self.resyncBackoff.delay(attempt: consecutiveResyncs - 2, unitRandom: Double.random(in: 0...1)))
            } catch {
                return false
            }
        }
        return true
    }

    /// Resign now (the operator's per-game or Resign-all button).
    func resignNow() async throws {
        guard !finished else { return }
        try await api.resign(gameID: gameID)
        await observer.gameEvent(gameID: gameID, .action("resigned (operator)"))
    }

    // MARK: - Stream items

    private func handle(_ item: LichessBotStreamItem, settings: LichessBotSettings) async throws {
        switch item {
        case .line(let data):
            await observer.gameEvent(gameID: gameID, .streamLine(data, receivedAt: Date()))
            try await handleLine(data, settings: settings)
        case .keepAlive:
            await observer.gameEvent(gameID: gameID, .keepAlive(receivedAt: Date()))
            if !keepAlivesSeen {
                keepAlivesSeen = true
                updateStallTimeout(settings: settings.connection)
            }
        case .oversizeLineDiscarded(let byteCount):
            await observer.gameEvent(gameID: gameID, .anomaly("discarded an oversize stream line of \(byteCount) bytes"))
        case .truncatedAtEnd(let byteCount):
            await observer.gameEvent(gameID: gameID, .anomaly("stream ended mid-line with \(byteCount) bytes pending"))
        }
    }

    /// Stall limit for this game's stream: the event-stream limit once
    /// keep-alives have been seen on it, otherwise the longer resync bound
    /// (whether game streams carry keep-alives is a live-verification item).
    private func updateStallTimeout(settings: LichessBotConnectionSettings) {
        stallTimeout.value = keepAlivesSeen
            ? .seconds(settings.eventStreamStallTimeoutSeconds)
            : .seconds(settings.gameStreamResyncSeconds)
    }

    private func handleLine(_ data: Data, settings: LichessBotSettings) async throws {
        let line: LichessBotGameStreamLine
        do {
            line = try LichessBotGameStreamLine.decode(data)
        } catch {
            await observer.gameEvent(gameID: gameID, .anomaly("undecodable game-stream line: \(error)"))
            return
        }
        switch line {
        case .gameFull(let full):
            try await handleGameFull(full, settings: settings)
        case .gameState(let state):
            try await handleState(state, settings: settings)
        case .chatLine(let chat):
            await observer.gameEvent(gameID: gameID, .chat(chat))
            await answerCommand(in: chat)
        case .opponentGone(let gone):
            await handleOpponentGone(gone, settings: settings)
        case .unknown(let type):
            await observer.gameEvent(gameID: gameID, .anomaly("unknown game-stream line type \(type)"))
        }
    }

    private func handleGameFull(_ full: LichessBotGameFull, settings: LichessBotSettings) async throws {
        gameFull = full
        let color: LichessBotColorName
        if full.white.id == ourAccountID {
            color = .white
        } else if full.black.id == ourAccountID {
            color = .black
        } else {
            await observer.gameEvent(gameID: gameID, .anomaly("neither player is \(ourAccountID); not our game"))
            finished = true
            return
        }
        ourColor = color
        await observer.gameEvent(gameID: gameID, .gameInfo(full, ourColor: color))

        if tracker == nil {
            if let reason = unplayableReason(full) {
                try await handleUnplayable(reason: reason, full: full)
            } else {
                do {
                    tracker = try LichessBotPositionTracker(initialFen: full.initialFen)
                } catch {
                    try await handleUnplayable(reason: error.localizedDescription, full: full)
                }
            }
        }
        // Always process the embedded state: it may already carry a finished
        // status, including for a game handled as unplayable above.
        try await handleState(full.state, settings: settings)
        checksForeignMoves = true
    }

    /// Why DCM can't play this game at all, or nil (plan E20).
    private func unplayableReason(_ full: LichessBotGameFull) -> String? {
        if full.variant.key.known != .standard {
            return "variant \(full.variant.key)"
        }
        if !LichessBotPositionTracker.isStandardStart(full.initialFen) {
            return "non-standard start position"
        }
        if full.clock == nil {
            return "no clock (correspondence or unlimited)"
        }
        return nil
    }

    /// A game the policy would never accept arrived anyway (accepted on
    /// lichess.org, or left over from before a restart): abort while that
    /// is still allowed, otherwise resign (plan E20).
    private func handleUnplayable(reason: String, full: LichessBotGameFull) async throws {
        guard !unplayableResolved else { return }
        isUnplayable = true
        guard full.state.status.isLive == true else { return }
        await observer.gameEvent(gameID: gameID, .anomaly("unplayable game (\(reason))"))
        if full.state.moveTokens.count < 2 {
            do {
                try await api.abort(gameID: gameID)
                unplayableResolved = true
                await observer.gameEvent(gameID: gameID, .action("aborted unplayable game"))
                return
            } catch let error as LichessBotAPIError where Self.isRefusal(error) {
                // Too late to abort (the opponent moved meanwhile): resign.
                // Any other failure propagates: the stream reopens with
                // backoff and the next gameFull tries the abort again.
                await observer.gameEvent(gameID: gameID, .anomaly("abort refused (\(error.localizedDescription)); resigning instead"))
            }
        }
        try await api.resign(gameID: gameID)
        unplayableResolved = true
        await observer.gameEvent(gameID: gameID, .action("resigned unplayable game"))
    }

    private func handleState(_ state: LichessBotGameState, settings: LichessBotSettings) async throws {
        if let tracker {
            let sync: LichessBotPositionSync
            do {
                sync = try tracker.sync(to: state.moveTokens)
            } catch {
                await observer.gameEvent(gameID: gameID, .anomaly("divergence: \(error.localizedDescription)"))
                throw ResyncNeeded(reason: "move list did not replay", isDisagreement: true)
            }
            if sync != .unchanged {
                await observer.gameEvent(gameID: gameID, .positionSynced(sync, ply: tracker.ply))
            }
            let previouslySeenPlies = highestPlySeen
            if tracker.ply > highestPlySeen {
                highestPlySeen = tracker.ply
                consecutiveResyncs = 0
                consecutiveDisagreements = 0
            }
            // Only plies this session has never seen are checked: earlier
            // ones were checked when they arrived, or predate this session
            // (a game resumed after a relaunch), and a tracker rebuilt after
            // a resync replays them all again.
            if checksForeignMoves, let ourColor, tracker.ply > previouslySeenPlies {
                let ourPieceColor: PieceColor = ourColor == .white ? .white : .black
                for ply in previouslySeenPlies..<tracker.ply where Self.mover(atPly: ply) == ourPieceColor {
                    let played = tracker.moves[ply].uci
                    if attemptedPosts[ply] != played {
                        await stopMoving("move \(played) at ply \(ply) is on our side but this client did not send it; another client may be playing this account")
                        // Not a return: this same state may carry the finished
                        // status, which must still end the session.
                        break
                    }
                }
            }
            if case .rebuilt = sync {
                // A takeback went back to `tracker.ply`: readings for plies
                // no longer on the board stop counting toward a streak. The
                // journal fold applies the same rule
                // (`LichessBotGameSessionCarryover.fold`), so a resumed
                // session's streaks match this one's.
                readings.removeAll { $0.ply >= tracker.ply }
                if let posted = lastPostedPly, posted >= tracker.ply {
                    // It removed a ply we had moved at (E30).
                    lastPostedPly = nil
                }
            }
        }

        // A finished status ends the game whether or not DCM could track
        // its position — including an unplayable game it just aborted.
        switch state.status.isLive {
        case .some(false):
            await finish(status: state.status, winner: state.winner, settings: settings)
            return
        case .none:
            await observer.gameEvent(gameID: gameID, .anomaly("unknown game status \(state.status); not moving until it resolves"))
            return
        case .some(true):
            break
        }

        if let pending = pendingDrawDisagreement {
            pendingDrawDisagreement = nil
            await observer.gameEvent(gameID: gameID, .anomaly(pending))
        }

        guard let tracker, let ourColor else {
            if isUnplayable { return }
            await observer.gameEvent(gameID: gameID, .anomaly("gameState before gameFull"))
            return
        }
        guard !movingStopped else { return }

        if let condition = tracker.engine.drawCondition, loggedDrawDisagreementPly != tracker.ply {
            // Lichess ends bot games on these rules itself (plan E10, E11).
            // Report it only if the next state is still live; see
            // `pendingDrawDisagreement`.
            loggedDrawDisagreementPly = tracker.ply
            pendingDrawDisagreement = "local draw rule \(condition.rawValue) applies at ply \(tracker.ply) (\(FENParser.fen(from: tracker.engine.state))) but Lichess kept the game live"
        }

        let opponentColor: LichessBotColorName = ourColor == .white ? .black : .white
        // A released held move may still be waiting in the request gate,
        // where nothing re-checks its position before it is sent. A takeback
        // is the only way the position at its ply can legitimately change,
        // so the proposal is left unanswered until that move is resolved; a
        // later state answers it if it still stands.
        if state.isProposingTakeback(opponentColor),
           !pacedMoveReleased,
           LichessBotPlayPolicy.shouldAcceptTakeback(acceptedSoFar: takebacksAccepted, settings: settings.play) {
            do {
                try await api.respondToTakeback(gameID: gameID, accept: true)
                takebacksAccepted += 1
                await observer.gameEvent(gameID: gameID, .takebackAccepted)
                return
            } catch let error where Self.endsTheStreamLoop(error) {
                throw error
            } catch {
                // Accepting is optional. The proposal is still pending, so a
                // later state can try again with the allowance unspent; play
                // on meanwhile, since our clock may be running.
                await observer.gameEvent(gameID: gameID, .anomaly("accepting the takeback failed: \(error.localizedDescription); playing on"))
            }
        }

        let ourPieceColor: PieceColor = ourColor == .white ? .white : .black
        let ourTurn = tracker.sideToMove == ourPieceColor && tracker.engine.result == nil
        ourClockMilliseconds = state.remaining(for: ourColor).value
        ourClockReceivedAt = time.now()
        if let heldPly = pacedMovePly, !pacedMoveReleased,
           heldPly != tracker.ply || pacedMovePosition != FENParser.fen(from: tracker.engine.state) {
            await dropPacedMove(reason: "the position changed")
        }
        // After any drop, so a dropped held move is not reported as still
        // holding while its replacement is decided.
        await reportTurnStatus()
        if ourTurn && lastPostedPly != tracker.ply && pacedMovePly != tracker.ply {
            try await playMove(state: state, ourColor: ourColor, opponentColor: opponentColor, settings: settings)
        } else if !ourTurn && state.isOfferingDraw(opponentColor), let last = readings.last,
                  LichessBotPlayPolicy.shouldAcceptDraw(current: last, settings: settings.play) {
            // An offer made while it is not our turn can't ride on our next
            // move yet; answer it directly. Accepting is optional, so a
            // failure is journaled and the stream kept.
            do {
                try await api.respondToDraw(gameID: gameID, accept: true)
                await observer.gameEvent(gameID: gameID, .action("accepted draw offer"))
            } catch let error where Self.endsTheStreamLoop(error) {
                throw error
            } catch {
                await observer.gameEvent(gameID: gameID, .anomaly("accepting the draw offer failed: \(error.localizedDescription)"))
            }
        }

        if !greeted {
            greeted = true
            if tracker.ply <= 1 {
                await sendChat(settings.chat.greetingEnabled ? settings.chat.greetingTemplate : nil, origin: .greeting, settings: settings)
            }
        }
    }

    // MARK: - Moving

    private func playMove(state: LichessBotGameState, ourColor: LichessBotColorName, opponentColor: LichessBotColorName, settings: LichessBotSettings) async throws {
        guard let tracker else { return }
        let ply = tracker.ply
        let engine = tracker.engine
        // The position this decision is for: every later check compares it,
        // since a takeback and a different reply can return to the same ply.
        let position = FENParser.fen(from: engine.state)
        let request = LichessBotMoveRequest(
            state: engine.state,
            history: engine.recentStates,
            legalMoves: engine.currentLegalMoves,
            ply: ply
        )

        // Only a live-trainer or followed-lineage game follows new
        // generations (follow-lineage plan §3.8), and only to one already
        // built: this runs on our clock, so it must never start or wait on a
        // network build. It stays with its source and, for a followed
        // lineage, with the lineage it started on (a source re-pointed at
        // another lineage never reaches a game in progress), and only moves
        // forward.
        let refreshingSource = settings.model.source
        if settings.model.midGameRefresh,
           refreshingSource == .liveTrainer || refreshingSource == .followLineage,
           playingMoveSource.info.sourceKind == refreshingSource {
            let latest = await latestMoveSource()
            if latest.info.sourceKind == refreshingSource,
               latest.info.lineage?.followed == playingMoveSource.info.lineage?.followed,
               latest.info.generationID > playingMoveSource.info.generationID,
               Self.isLaterInItsLineage(latest.info, than: playingMoveSource.info) {
                playingMoveSource = latest
            }
        }
        let source = playingMoveSource

        let decision = try await source.decide(request, schedule: settings.play.samplingSchedule)
        guard stillOurMove(at: ply, in: position) else {
            await observer.gameEvent(gameID: gameID, .anomaly("discarded a decision for ply \(ply): the position moved on while deciding"))
            return
        }
        let reading = LichessBotValueReading(ply: ply, decision: decision)
        // One reading per ply: a ply decided again (after a resync or a
        // rejected move) replaces its earlier reading, so a streak counts our
        // moves, not our decisions.
        readings.removeAll { $0.ply >= ply }
        readings.append(reading)
        await observer.gameEvent(gameID: gameID, .moveDecided(ply: ply, decision: decision, generation: source.info))

        if LichessBotPlayPolicy.shouldResign(readings: readings, settings: settings.play) {
            lastPostedPly = ply
            do {
                try await api.resign(gameID: gameID)
                await observer.gameEvent(gameID: gameID, .action("resigned (value head: p_loss \(decision.loss))"))
                return
            } catch let error where Self.endsTheStreamLoop(error) {
                lastPostedPly = nil
                throw error
            } catch {
                // Not resigned: this ply is still ours to play, on our clock,
                // so play the decided move; the next reading can resign again.
                lastPostedPly = nil
                await observer.gameEvent(gameID: gameID, .anomaly("resigning failed: \(error.localizedDescription); playing the move instead"))
            }
        }

        if settings.play.minimumThinkMilliseconds > 0 {
            try await time.sleep(for: .milliseconds(settings.play.minimumThinkMilliseconds))
            guard stillOurMove(at: ply, in: position) else { return }
        }

        let opponentOffering = state.isOfferingDraw(opponentColor)
        let acceptDraw = opponentOffering && LichessBotPlayPolicy.shouldAcceptDraw(current: reading, settings: settings.play)
        let offerDraw = !opponentOffering && LichessBotPlayPolicy.shouldOfferDraw(readings: readings, settings: settings.play)

        // The operator's delay or hold applies only after DCM's first move
        // (each side's first ply is below this), so our own wait can never
        // abort a game.
        if ply > 1, await pacing().isActive {
            guard stillOurMove(at: ply, in: position) else { return }
            await schedulePacedMove(decision: decision, ply: ply, position: position, offeringDraw: acceptDraw || offerDraw, settings: settings)
            return
        }
        // The awaits above can let another state for this ply in; post only
        // if it is still ours to play.
        guard stillOurMove(at: ply, in: position), pacedMovePly != ply else { return }
        try await post(decision: decision, ply: ply, position: position, offeringDraw: acceptDraw || offerDraw, settings: settings)
    }

    // MARK: - Operator pacing (plan §14.3c)

    private func schedulePacedMove(decision: LichessBotMoveDecision, ply: Int, position: String, offeringDraw: Bool, settings: LichessBotSettings) async {
        pacedMovePly = ply
        pacedMovePosition = position
        pacedMoveReleased = false
        await observer.gameEvent(gameID: gameID, .moveHeld(ply: ply, uci: decision.uci, san: decision.san))
        let started = time.now()
        pacedMove = Task { [weak self] in
            await self?.runPacedMove(decision: decision, ply: ply, position: position, offeringDraw: offeringDraw, settings: settings, started: started)
        }
        await reportTurnStatus()
    }

    /// Forget a held move that no longer applies, and say why.
    private func dropPacedMove(reason: String) async {
        guard let ply = pacedMovePly else { return }
        pacedMove?.cancel()
        pacedMove = nil
        pacedMovePly = nil
        pacedMovePosition = nil
        pacedMoveReleased = false
        await observer.gameEvent(gameID: gameID, .moveReleased(ply: ply, reason: "dropped: \(reason)"))
    }

    private func runPacedMove(decision: LichessBotMoveDecision, ply: Int, position: String, offeringDraw: Bool, settings: LichessBotSettings, started: Duration) async {
        // Cancelled (the game ended, the session stopped, or the move was
        // dropped): whoever cancelled has cleared the state. A task cancelled
        // while its release check was suspended must not touch the held-move
        // state either: it may already belong to a newer held move, possibly
        // for the same ply.
        guard let reason = await pacedMoveReleaseReason(ply: ply, started: started),
              !Task.isCancelled, pacedMovePly == ply else {
            return
        }
        // `pacedMovePly` stays set until `post` has marked the ply posted,
        // so no state arriving during the awaits below can start a second
        // decision and POST for this ply.
        pacedMoveReleased = true
        await observer.gameEvent(gameID: gameID, .moveReleased(ply: ply, reason: reason))
        await reportTurnStatus()
        guard pacedMovePly == ply, stillOurMove(at: ply, in: position) else {
            clearPacedMove(ply: ply)
            await reportTurnStatus()
            await resyncIfHeldPlyWasLeftUndecided(ply: ply)
            return
        }
        do {
            try await post(decision: decision, ply: ply, position: position, offeringDraw: offeringDraw, settings: settings)
            clearPacedMove(ply: ply)
            await resyncIfHeldPlyWasLeftUndecided(ply: ply)
            return
        } catch is CancellationError {
            // The game ended or the session stopped; nothing more to post.
        } catch let resync as ResyncNeeded {
            requestResync(resync)
        } catch let error as LichessBotAPIError {
            if case .unauthorized = error {
                await observer.gameEvent(gameID: gameID, .tokenRejected(error.localizedDescription))
            } else {
                await observer.gameEvent(gameID: gameID, .anomaly("posting the held move at ply \(ply) failed: \(error.localizedDescription)"))
                requestResync(ResyncNeeded(reason: "held move post failed", isDisagreement: false))
            }
        } catch {
            await observer.gameEvent(gameID: gameID, .anomaly("posting the held move at ply \(ply) failed: \(error.localizedDescription)"))
            requestResync(ResyncNeeded(reason: "held move post failed", isDisagreement: false))
        }
        clearPacedMove(ply: ply)
    }

    /// A held move for `ply` was not posted because the position changed
    /// under it at the same ply. The stream loop skipped deciding for that
    /// state (the held move occupied the ply) and may receive no further line
    /// while our clock runs, so reopen the stream: its `gameFull` decides
    /// afresh.
    private func resyncIfHeldPlyWasLeftUndecided(ply: Int) async {
        guard !movingStopped, stillOurMove(at: ply) else { return }
        await observer.gameEvent(gameID: gameID, .anomaly("the held move for ply \(ply) no longer fits the position; reopening the stream to decide again"))
        requestResync(ResyncNeeded(reason: "held move's position changed at ply \(ply)", isDisagreement: false))
    }

    private func clearPacedMove(ply: Int) {
        guard pacedMovePly == ply else { return }
        pacedMove = nil
        pacedMovePly = nil
        pacedMovePosition = nil
        pacedMoveReleased = false
    }

    /// Wait until the held move should go out, and say why; nil when
    /// cancelled. The clock floor overrides both the delay and the hold.
    private func pacedMoveReleaseReason(ply: Int, started: Duration) async -> String? {
        while true {
            guard stillOurMove(at: ply) else { return "the position moved on" }
            let now = time.now()
            if let remaining = ourClockNow()?.value {
                let floor = LichessBotMovePacingSnapshot.clockFloorMilliseconds(lastMoveRoundTripMilliseconds: lastMoveRoundTripMilliseconds)
                if remaining <= floor {
                    return "our clock reached \(floor / 1000) s, so it was played automatically"
                }
            }
            let current = await pacing()
            if current.releaseRequested {
                return "the operator played it"
            }
            if !current.holds && now - started >= .seconds(current.delaySeconds) {
                return current.delaySeconds > 0 ? "the \(current.delaySeconds) s delay elapsed" : "pacing was turned off"
            }
            do {
                try await time.sleep(for: .milliseconds(250))
            } catch {
                return nil
            }
        }
    }

    private func stillOurMove(at ply: Int) -> Bool {
        guard !finished, let tracker, let ourColor else { return false }
        let ourPieceColor: PieceColor = ourColor == .white ? .white : .black
        return tracker.ply == ply && tracker.sideToMove == ourPieceColor && lastPostedPly != ply
    }

    /// `stillOurMove(at:)`, and the position is still `position` (a FEN): a
    /// takeback and a different reply can return to the same ply in a new
    /// position, where a move decided before is wrong or illegal.
    private func stillOurMove(at ply: Int, in position: String) -> Bool {
        guard stillOurMove(at: ply), let tracker else { return false }
        return FENParser.fen(from: tracker.engine.state) == position
    }

    /// Our clock, run down locally from the latest `gameState`. It runs down
    /// whether or not it is our turn, so it means something only while we
    /// are to move — the only time the manager and the clock floor use it.
    private func ourClockNow() -> LichessBotMilliseconds? {
        guard let clock = ourClockMilliseconds, let receivedAt = ourClockReceivedAt else { return nil }
        return LichessBotMilliseconds(clock - Int(LichessBotBackoff.seconds(time.now() - receivedAt) * 1000))
    }

    /// Tell the session manager whether this game waits on our move now, and
    /// our clock. A move held for the operator does not count as waiting:
    /// nothing is posted until it is released, and counting it would hold
    /// back every housekeeping request in the gate for the whole hold.
    private func reportTurnStatus() async {
        var awaiting = false
        if !finished, !movingStopped, let tracker, let ourColor {
            let ourPieceColor: PieceColor = ourColor == .white ? .white : .black
            let holdingOurMove = pacedMovePly == tracker.ply && !pacedMoveReleased
            awaiting = tracker.sideToMove == ourPieceColor
                && tracker.engine.result == nil
                && lastPostedPly != tracker.ply
                && !holdingOurMove
        }
        await onTurnStatus(gameID, LichessBotTurnStatus(awaitingOurMove: awaiting, ourClock: ourClockNow()))
    }

    /// Post the move. The ply is marked posted first, so no second POST for
    /// it can start while this one is in flight.
    private func post(decision: LichessBotMoveDecision, ply: Int, position: String, offeringDraw: Bool, settings: LichessBotSettings) async throws {
        let retries = LichessBotBackoff(initial: .seconds(1), multiplier: 2, cap: .seconds(4))
        var transientFailures = 0
        while true {
            lastPostedPly = ply
            attemptedPosts[ply] = decision.uci
            let started = time.now()
            do {
                try await api.makeMove(gameID: gameID, uci: decision.uci, offeringDraw: offeringDraw)
                let elapsed = LichessBotBackoff.seconds(time.now() - started) * 1000
                lastMoveRoundTripMilliseconds = elapsed
                await observer.gameEvent(gameID: gameID, .movePosted(ply: ply, uci: decision.uci, offeringDraw: offeringDraw, milliseconds: elapsed))
                await onTurnStatus(gameID, LichessBotTurnStatus(awaitingOurMove: false, ourClock: nil))
                return
            } catch LichessBotGateError.rateLimited {
                // The gate now holds every request for the cooldown. Post
                // again only if it is still our move at this ply (§5.4).
                lastPostedPly = nil
                await observer.gameEvent(gameID: gameID, .moveRejected(ply: ply, uci: decision.uci, error: "rate limited; waiting for the cooldown"))
                guard stillOurMove(at: ply, in: position) else { return }
            } catch let error as LichessBotAPIError where Self.isTokenRejection(error) {
                lastPostedPly = nil
                await observer.gameEvent(gameID: gameID, .moveRejected(ply: ply, uci: decision.uci, error: error.localizedDescription))
                throw error
            } catch let error as LichessBotAPIError where Self.isRefusal(error) {
                // A 4xx: Lichess refused the move. (A 5xx is Lichess failing,
                // not refusing; it is retried below like a network error.)
                lastPostedPly = nil
                await observer.gameEvent(gameID: gameID, .moveRejected(ply: ply, uci: decision.uci, error: error.localizedDescription))
                let rejections = rejectionsByPly[ply, default: 0] + 1
                rejectionsByPly[ply] = rejections
                if rejections >= Self.maximumRejectionsPerPly {
                    // Moves that keep being refused mean the game is not in
                    // the state this client believes; never keep racing.
                    await stopMoving("\(rejections) moves rejected at ply \(ply) (last: \(error.localizedDescription))")
                    return
                }
                // A 4xx means Lichess disagrees about the position (or the
                // game just ended). Reopen for a fresh gameFull (§5.5, E21).
                throw ResyncNeeded(reason: "move rejected: \(error.localizedDescription)", isDisagreement: true)
            } catch let error as LichessBotAPIError where !Self.isServerFailure(error) {
                // The request could not be built or its answer read: a fault
                // in this client, not a refusal by Lichess, and sending the
                // same move again can't fix it.
                lastPostedPly = nil
                await observer.gameEvent(gameID: gameID, .moveRejected(ply: ply, uci: decision.uci, error: error.localizedDescription))
                await stopMoving("the move at ply \(ply) could not be sent: \(error.localizedDescription)")
                return
            } catch let error as LichessBotGateError {
                throw error
            } catch is CancellationError {
                throw CancellationError()
            } catch {
                lastPostedPly = nil
                await observer.gameEvent(gameID: gameID, .moveRejected(ply: ply, uci: decision.uci, error: String(describing: error)))
                guard transientFailures < LichessBotGameSession.maximumTransientPostRetries, stillOurMove(at: ply, in: position) else {
                    throw ResyncNeeded(reason: "move post kept failing: \(error)", isDisagreement: false)
                }
                try await time.sleep(for: retries.delay(attempt: transientFailures, unitRandom: Double.random(in: 0...1)))
                transientFailures += 1
                guard stillOurMove(at: ply, in: position) else { return }
            }
        }
    }

    private static let maximumTransientPostRetries = 3

    /// A 5xx: Lichess failed to process the request, which says nothing
    /// about whether the move was acceptable.
    private static func isServerFailure(_ error: LichessBotAPIError) -> Bool {
        if case .http(let status, _) = error {
            return (500..<600).contains(status)
        }
        return false
    }

    /// Lichess answered and refused the request (a client error other than a
    /// token rejection), as opposed to failing to answer it or the gate
    /// holding it back. The API maps a rejected token to `.unauthorized` and
    /// the gate turns a 429 into its own error, so neither lands here.
    private static func isRefusal(_ error: LichessBotAPIError) -> Bool {
        if case .http(let status, _) = error {
            return (400..<500).contains(status)
        }
        return false
    }

    private static func isTokenRejection(_ error: LichessBotAPIError) -> Bool {
        if case .unauthorized = error {
            return true
        }
        return false
    }

    /// Errors that must end the stream loop rather than be journaled while
    /// play continues: cancellation, the gate closing, a rejected token.
    private static func endsTheStreamLoop(_ error: Error) -> Bool {
        if error is CancellationError { return true }
        if let gateError = error as? LichessBotGateError, case .closed = gateError { return true }
        if let apiError = error as? LichessBotAPIError, isTokenRejection(apiError) { return true }
        return false
    }

    /// Who moves at `ply` in a game from the standard start.
    private static func mover(atPly ply: Int) -> PieceColor {
        ply % 2 == 0 ? .white : .black
    }

    /// Stop moving in this game for good and report why. The game is left
    /// to the operator (and the controller takes the bot offline).
    private func stopMoving(_ reason: String) async {
        guard !movingStopped else { return }
        movingStopped = true
        await onTurnStatus(gameID, LichessBotTurnStatus(awaitingOurMove: false, ourClock: nil))
        await observer.gameEvent(gameID: gameID, .stoppedMoving(reason: reason))
    }

    // MARK: - Opponent gone

    private func handleOpponentGone(_ gone: LichessBotOpponentGone, settings: LichessBotSettings) async {
        pendingClaim?.cancel()
        pendingClaim = nil
        guard gone.gone else {
            await observer.gameEvent(gameID: gameID, .action("opponent returned; pending claim cancelled"))
            return
        }
        guard settings.play.claimWhenOpponentGone, let seconds = gone.claimWinInSeconds else { return }
        await observer.gameEvent(gameID: gameID, .action("opponent gone; claim scheduled in \(seconds)s"))
        let time = self.time
        pendingClaim = Task { [weak self] in
            do {
                try await time.sleep(for: .seconds(seconds) + LichessBotGameSession.claimGrace)
            } catch {
                return
            }
            await self?.claimAfterOpponentLeft()
        }
    }

    private static let claimGrace: Duration = .seconds(1)

    /// Claim victory; if Lichess refuses it, claim a draw once. Lichess
    /// checks that the opponent is really gone, so a stale `opponentGone`
    /// (plan §12.4) is refused rather than acted on. Only a refusal leads to
    /// the draw claim: a claim that got no answer says nothing about the win.
    private func claimAfterOpponentLeft() async {
        // `pendingClaim` stays set while the claim is in flight, so the
        // opponent returning or the game ending still cancels it.
        guard !finished, !Task.isCancelled else { return }
        do {
            try await api.claimVictory(gameID: gameID)
            await observer.gameEvent(gameID: gameID, .action("claimed victory (opponent gone)"))
        } catch let error as LichessBotAPIError where Self.isRefusal(error) {
            guard !finished, !Task.isCancelled else {
                await observer.gameEvent(gameID: gameID, .action("victory claim refused: \(error.localizedDescription); the claim was cancelled meanwhile, so no draw is claimed"))
                return
            }
            await observer.gameEvent(gameID: gameID, .action("victory claim refused: \(error.localizedDescription); claiming a draw"))
            do {
                try await api.claimDraw(gameID: gameID)
                await observer.gameEvent(gameID: gameID, .action("claimed draw (opponent gone)"))
            } catch {
                if let apiError = error as? LichessBotAPIError, Self.isTokenRejection(apiError) {
                    await observer.gameEvent(gameID: gameID, .tokenRejected(apiError.localizedDescription))
                }
                await observer.gameEvent(gameID: gameID, .action("draw claim refused: \(error.localizedDescription)"))
            }
        } catch {
            if Task.isCancelled {
                // The opponent returned, the game ended or the session
                // stopped while the claim was queued or in flight; the
                // cancellation surfaces from the gate or from URLSession.
                await observer.gameEvent(gameID: gameID, .action("victory claim cancelled: \(error.localizedDescription)"))
                return
            }
            if let apiError = error as? LichessBotAPIError, Self.isTokenRejection(apiError) {
                await observer.gameEvent(gameID: gameID, .tokenRejected(apiError.localizedDescription))
            }
            // Lichess did not answer the claim: nothing says the victory is
            // refused, and a draw claim would give a won game away. A new
            // `opponentGone` schedules a new claim; if the opponent stays
            // away, their clock ends the game.
            await observer.gameEvent(gameID: gameID, .anomaly("victory claim failed: \(error.localizedDescription); not claiming a draw"))
        }
    }

    // MARK: - Finishing and chat

    private func finish(status: LichessBotOpenValue<LichessBotGameStatusName>, winner: LichessBotOpenValue<LichessBotColorName>?, settings: LichessBotSettings) async {
        guard !finished else { return }
        finished = true
        pendingDrawDisagreement = nil
        pacedMove?.cancel()
        pacedMove = nil
        pacedMovePly = nil
        pacedMovePosition = nil
        pacedMoveReleased = false
        pendingClaim?.cancel()
        pendingClaim = nil
        await stopCommandReplies(reason: "the game ended")
        await observer.gameEvent(gameID: gameID, .finished(status: status, winner: winner, localDrawCondition: tracker?.engine.drawCondition))
        await onTurnStatus(gameID, LichessBotTurnStatus(awaitingOurMove: false, ourClock: nil))
        if status.known != .aborted && status.known != .noStart && !farewellSent {
            farewellSent = true
            await sendChat(settings.chat.goodbyeEnabled ? settings.chat.goodbyeTemplate : nil, origin: .goodbye, settings: settings)
        }
    }

    /// Answer a chat command (plan §12.5a) in the room it came from, within
    /// the per-game budget. Skips and failures are journaled, never retried.
    private func answerCommand(in line: LichessBotChatLine) async {
        guard let command = LichessBotChatCommands.parse(line, ourAccountID: ourAccountID) else { return }
        guard let room = line.room.known else {
            await observer.gameEvent(gameID: gameID, .anomaly("!\(command.rawValue) from \(line.username) in unknown room \(line.room.raw); not answered"))
            return
        }
        // Before the budget, so a skip here doesn't spend it.
        guard let ourUsername = ourDisplayName() else {
            await observer.gameEvent(gameID: gameID, .anomaly("!\(command.rawValue) from \(line.username) not answered: no display name for our account in gameFull"))
            return
        }
        switch commandBudget.decide(now: time.now(), ourClockMilliseconds: ourClockMilliseconds) {
        case .skip(let reason):
            await observer.gameEvent(gameID: gameID, .action("not answering !\(command.rawValue) from \(line.username): \(reason)"))
            return
        case .reply:
            await observer.gameEvent(gameID: gameID, .commandReplyQueued(command: command, username: line.username, room: room))
        }
        let info = playingMoveSource.info
        let context = LichessBotChatCommandContext(
            ourUsername: ourUsername,
            modelID: info.modelID,
            trainingStep: info.trainingStep,
            build: BuildInfo.buildNumber,
            hardware: HardwareInfo.current
        )
        let replies: [String]
        do {
            replies = try LichessBotChatCommands.replies(to: command, context: context)
        } catch {
            await observer.gameEvent(gameID: gameID, .anomaly("!\(command.rawValue) not answered: \(error.localizedDescription)"))
            return
        }
        pendingCommandReplies.append(PendingCommandReply(command: command, username: line.username, room: room, texts: replies))
        if commandReplySender == nil {
            commandReplySender = Task { [weak self] in
                await self?.sendPendingCommandReplies()
            }
        }
    }

    /// Send queued command replies in order until none are left or the
    /// sender is cancelled.
    private func sendPendingCommandReplies() async {
        while !Task.isCancelled, !pendingCommandReplies.isEmpty {
            let reply = pendingCommandReplies.removeFirst()
            await sendCommandReply(reply)
        }
        // A cancelled sender was cleared by whoever cancelled it; only one
        // that ran out of work clears itself.
        if !Task.isCancelled {
            commandReplySender = nil
        }
    }

    private func sendCommandReply(_ reply: PendingCommandReply) async {
        for text in reply.texts {
            do {
                try await api.chat(gameID: gameID, room: reply.room, text: text)
            } catch {
                if Task.isCancelled {
                    await observer.gameEvent(gameID: gameID, .action("reply to !\(reply.command.rawValue) from \(reply.username) cut short: \(error.localizedDescription)"))
                } else {
                    await observer.gameEvent(gameID: gameID, .anomaly("reply to !\(reply.command.rawValue) failed: \(error.localizedDescription)"))
                }
                return
            }
            await observer.gameEvent(gameID: gameID, .chatSent(room: reply.room, text: text, origin: .commandReply))
        }
        await observer.gameEvent(gameID: gameID, .action("replied to !\(reply.command.rawValue) from \(reply.username) (\(reply.room.rawValue))"))
    }

    /// Stop sending command replies (the game or the session ended); any
    /// not yet sent are journaled as unanswered, with `reason`.
    private func stopCommandReplies(reason: String) async {
        commandReplySender?.cancel()
        commandReplySender = nil
        let unsent = pendingCommandReplies
        pendingCommandReplies.removeAll()
        for reply in unsent {
            await observer.gameEvent(gameID: gameID, .action("not answering !\(reply.command.rawValue) from \(reply.username): \(reason)"))
        }
    }

    /// Our display name in this game. Lichess always names an account's
    /// player, so nil means no `gameFull` has identified us.
    private func ourDisplayName() -> String? {
        guard let full = gameFull, let ourColor else { return nil }
        return (ourColor == .white ? full.white : full.black).name
    }

    /// The opponent as Lichess shows them: their username, or, for Lichess's
    /// built-in AI (no account, so no name), the label Lichess itself uses.
    /// Nil when neither is known.
    private func opponentDisplayName() -> String? {
        guard let full = gameFull, let ourColor else { return nil }
        let opponent = ourColor == .white ? full.black : full.white
        if let name = opponent.name { return name }
        if let level = opponent.aiLevel { return "Stockfish level \(level)" }
        return nil
    }

    private func sendChat(_ template: String?, origin: LichessBotChatOrigin, settings: LichessBotSettings) async {
        guard let template else { return }
        let info = playingMoveSource.info
        var values = [
            "modelID": info.modelID,
            "source": info.sourceKind.rawValue,
            "build": "\(BuildInfo.buildNumber)",
        ]
        // A template that names the opponent needs a name for them; one that
        // doesn't is never held back by a missing name.
        if template.contains("{opponent}") {
            guard let opponent = opponentDisplayName() else {
                await observer.gameEvent(gameID: gameID, .anomaly("chat message skipped: it names the opponent, and gameFull gives no name for them"))
                return
            }
            values["opponent"] = opponent
        }
        guard let text = LichessBotChat.message(from: template, values: values) else {
            await observer.gameEvent(gameID: gameID, .anomaly("chat message skipped: longer than \(LichessBotChat.maximumLength) characters"))
            return
        }
        do {
            try await api.chat(gameID: gameID, room: settings.chat.room, text: text)
            await observer.gameEvent(gameID: gameID, .chatSent(room: settings.chat.room, text: text, origin: origin))
        } catch {
            await observer.gameEvent(gameID: gameID, .anomaly("chat failed: \(error.localizedDescription)"))
        }
    }
}
