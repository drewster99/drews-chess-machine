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
/// a POST, other calls can run on the actor. So after every await it
/// re-checks that the position is still the one it decided for, that it is
/// still our turn, and that nothing has been posted for this ply. The ply is
/// recorded as posted before the POST goes out, so a second decision for
/// the same ply can never be sent.
///
/// **Requests.** Every call goes through the account-wide gate via the API.
/// A 429 means waiting out the cooldown (the gate does that) and posting
/// again only if it is still our turn at the same ply (§5.4). A 400 on a
/// move means state disagreement: reopen the stream, never resend blindly
/// (§5.5).
actor LichessBotGameSession {

    let gameID: String
    private let ourAccountID: String
    private let api: any LichessBotGameAPI
    private let observer: any LichessBotGameObserver
    private let time: any LichessBotTimeSource
    private let settingsProvider: @Sendable () async -> LichessBotSettings
    private let latestMoveSource: @Sendable () async -> (any LichessBotMoveSource)?
    private let onTurnStatus: @Sendable (String, LichessBotTurnStatus) async -> Void
    private let pinnedMoveSource: any LichessBotMoveSource

    private var gameFull: LichessBotGameFull?
    private var ourColor: LichessBotColorName?
    private var tracker: LichessBotPositionTracker?
    private var lastPostedPly: Int?
    private var readings: [LichessBotValueReading] = []
    private var takebacksAccepted = 0
    private var greeted = false
    private var finished = false
    private var unplayableHandled = false
    private var pendingClaim: Task<Void, Never>?
    private var keepAlivesSeen = false
    private var loggedDrawDisagreementPly: Int?
    /// Resyncs in a row without the game advancing; each waits longer.
    private var consecutiveResyncs = 0
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
    /// Set from outside the actor to make the stream loop drop its
    /// connection and reopen it for a fresh `gameFull`.
    private let resyncRequest = SyncBox<String?>(nil)

    /// Raised inside the stream loop to drop the connection and reopen it
    /// for a fresh `gameFull`.
    private struct ResyncNeeded: Error {
        let reason: String
    }

    init(
        gameID: String,
        ourAccountID: String,
        api: any LichessBotGameAPI,
        moveSource: any LichessBotMoveSource,
        latestMoveSource: @escaping @Sendable () async -> (any LichessBotMoveSource)?,
        settingsProvider: @escaping @Sendable () async -> LichessBotSettings,
        observer: any LichessBotGameObserver,
        time: any LichessBotTimeSource,
        onTurnStatus: @escaping @Sendable (String, LichessBotTurnStatus) async -> Void
    ) {
        self.gameID = gameID
        self.ourAccountID = ourAccountID
        self.api = api
        self.pinnedMoveSource = moveSource
        self.latestMoveSource = latestMoveSource
        self.settingsProvider = settingsProvider
        self.observer = observer
        self.time = time
        self.onTurnStatus = onTurnStatus
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
        resyncRequest.value = reason
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
                    try await handle(item, settings: settings)
                    if finished { break }
                }
                if !finished {
                    await observer.gameEvent(gameID: gameID, .streamEnded(reason: "server closed the stream"))
                }
                attempt = 0
            } catch let resync as ResyncNeeded {
                await observer.gameEvent(gameID: gameID, .streamEnded(reason: "resync: \(resync.reason)"))
                attempt = 0
                consecutiveResyncs += 1
                if consecutiveResyncs >= Self.maximumConsecutiveResyncs {
                    await stopMoving("\(consecutiveResyncs) resyncs in a row without the game advancing (last: \(resync.reason))")
                    break
                }
                // The first resync is immediate; repeats back off, so a
                // disagreement that persists can never turn into a request
                // storm.
                if consecutiveResyncs > 1 {
                    do {
                        try await time.sleep(for: Self.resyncBackoff.delay(attempt: consecutiveResyncs - 2, unitRandom: Double.random(in: 0...1)))
                    } catch {
                        break
                    }
                }
                continue
            } catch LichessBotStreamError.stalled(let silence) {
                if let reason = resyncRequest.mutate({ request -> String? in
                    defer { request = nil }
                    return request
                }) {
                    await observer.gameEvent(gameID: gameID, .streamEnded(reason: "resync: \(reason)"))
                    attempt = 0
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
        await onTurnStatus(gameID, LichessBotTurnStatus(awaitingOurMove: false, ourClock: nil))
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
        guard !unplayableHandled else { return }
        unplayableHandled = true
        guard full.state.status.isLive == true else { return }
        let plies = full.state.moveTokens.count
        await observer.gameEvent(gameID: gameID, .anomaly("unplayable game (\(reason))"))
        if plies < 2 {
            do {
                try await api.abort(gameID: gameID)
                await observer.gameEvent(gameID: gameID, .action("aborted unplayable game"))
                return
            } catch let error as LichessBotAPIError {
                if case .unauthorized = error { throw error }
                // Too late to abort (the opponent moved meanwhile): resign.
                await observer.gameEvent(gameID: gameID, .anomaly("abort refused (\(error.localizedDescription)); resigning instead"))
            }
            try await api.resign(gameID: gameID)
            await observer.gameEvent(gameID: gameID, .action("resigned unplayable game"))
        } else {
            try await api.resign(gameID: gameID)
            await observer.gameEvent(gameID: gameID, .action("resigned unplayable game"))
        }
    }

    private func handleState(_ state: LichessBotGameState, settings: LichessBotSettings) async throws {
        if let tracker {
            let sync: LichessBotPositionSync
            do {
                sync = try tracker.sync(to: state.moveTokens)
            } catch {
                await observer.gameEvent(gameID: gameID, .anomaly("divergence: \(error.localizedDescription)"))
                throw ResyncNeeded(reason: "move list did not replay")
            }
            if sync != .unchanged {
                await observer.gameEvent(gameID: gameID, .positionSynced(sync, ply: tracker.ply))
            }
            if tracker.ply > highestPlySeen {
                highestPlySeen = tracker.ply
                consecutiveResyncs = 0
            }
            if checksForeignMoves, case .extended(let fromPly, let toPly) = sync, let ourColor {
                let ourPieceColor: PieceColor = ourColor == .white ? .white : .black
                for ply in fromPly..<toPly where Self.mover(atPly: ply) == ourPieceColor {
                    let played = tracker.moves[ply].uci
                    if attemptedPosts[ply] != played {
                        await stopMoving("move \(played) at ply \(ply) is on our side but this client did not send it; another client may be playing this account")
                        return
                    }
                }
            }
            if case .rebuilt = sync, let posted = lastPostedPly, posted >= tracker.ply {
                // A takeback removed a ply we had moved at (E30).
                lastPostedPly = nil
                readings.removeAll { $0.ply >= tracker.ply }
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

        guard let tracker, let ourColor else {
            if unplayableHandled { return }
            await observer.gameEvent(gameID: gameID, .anomaly("gameState before gameFull"))
            return
        }
        guard !movingStopped else { return }

        if let condition = tracker.engine.drawCondition, loggedDrawDisagreementPly != tracker.ply {
            // Lichess ends bot games on these rules itself, so a live game
            // here means the two rule sets disagree (plan E10, E11).
            loggedDrawDisagreementPly = tracker.ply
            await observer.gameEvent(gameID: gameID, .anomaly("local draw rule \(condition.rawValue) applies at ply \(tracker.ply) (\(FENParser.fen(from: tracker.engine.state))) but Lichess reports the game as \(state.status)"))
        }

        let opponentColor: LichessBotColorName = ourColor == .white ? .black : .white
        if state.isProposingTakeback(opponentColor) && LichessBotPlayPolicy.shouldAcceptTakeback(acceptedSoFar: takebacksAccepted, settings: settings.play) {
            takebacksAccepted += 1
            try await api.respondToTakeback(gameID: gameID, accept: true)
            await observer.gameEvent(gameID: gameID, .action("accepted takeback"))
            return
        }

        let ourPieceColor: PieceColor = ourColor == .white ? .white : .black
        let ourTurn = tracker.sideToMove == ourPieceColor && tracker.engine.result == nil
        await onTurnStatus(gameID, LichessBotTurnStatus(awaitingOurMove: ourTurn && lastPostedPly != tracker.ply, ourClock: state.remaining(for: ourColor)))

        if ourTurn && lastPostedPly != tracker.ply {
            try await playMove(state: state, ourColor: ourColor, opponentColor: opponentColor, settings: settings)
        } else if !ourTurn && state.isOfferingDraw(opponentColor), let last = readings.last,
                  LichessBotPlayPolicy.shouldAcceptDraw(current: last, settings: settings.play) {
            // An offer made while it is not our turn can't ride on our next
            // move yet; answer it directly.
            try await api.respondToDraw(gameID: gameID, accept: true)
            await observer.gameEvent(gameID: gameID, .action("accepted draw offer"))
        }

        if !greeted {
            greeted = true
            if tracker.ply <= 1 {
                await sendChat(settings.chat.greetingEnabled ? settings.chat.greetingTemplate : nil, settings: settings)
            }
        }
    }

    // MARK: - Moving

    private func playMove(state: LichessBotGameState, ourColor: LichessBotColorName, opponentColor: LichessBotColorName, settings: LichessBotSettings) async throws {
        guard let tracker else { return }
        let ply = tracker.ply
        let engine = tracker.engine
        let request = LichessBotMoveRequest(
            state: engine.state,
            history: engine.recentStates,
            legalMoves: engine.currentLegalMoves,
            ply: ply
        )

        var source = pinnedMoveSource
        if settings.model.midGameRefresh {
            if let latest = await latestMoveSource() {
                source = latest
            } else {
                await observer.gameEvent(gameID: gameID, .anomaly("mid-game refresh: no current model; continuing with generation \(pinnedMoveSource.info.generationID)"))
            }
        }

        let decision = try await source.decide(request, schedule: settings.play.samplingSchedule)
        guard stillOurMove(at: ply) else {
            await observer.gameEvent(gameID: gameID, .anomaly("discarded a decision for ply \(ply): the position moved on while deciding"))
            return
        }
        let reading = LichessBotValueReading(ply: ply, win: decision.win, draw: decision.draw, loss: decision.loss)
        readings.append(reading)
        await observer.gameEvent(gameID: gameID, .moveDecided(ply: ply, decision: decision, generation: source.info))

        if LichessBotPlayPolicy.shouldResign(readings: readings, settings: settings.play) {
            lastPostedPly = ply
            do {
                try await api.resign(gameID: gameID)
            } catch {
                // Not resigned: this ply is still ours to play.
                lastPostedPly = nil
                throw error
            }
            await observer.gameEvent(gameID: gameID, .action("resigned (value head: p_loss \(decision.loss))"))
            return
        }

        if settings.play.minimumThinkMilliseconds > 0 {
            try await time.sleep(for: .milliseconds(settings.play.minimumThinkMilliseconds))
            guard stillOurMove(at: ply) else { return }
        }

        let opponentOffering = state.isOfferingDraw(opponentColor)
        let acceptDraw = opponentOffering && LichessBotPlayPolicy.shouldAcceptDraw(current: reading, settings: settings.play)
        let offerDraw = !opponentOffering && LichessBotPlayPolicy.shouldOfferDraw(readings: readings, settings: settings.play)
        try await post(decision: decision, ply: ply, offeringDraw: acceptDraw || offerDraw, settings: settings)
    }

    private func stillOurMove(at ply: Int) -> Bool {
        guard !finished, let tracker, let ourColor else { return false }
        let ourPieceColor: PieceColor = ourColor == .white ? .white : .black
        return tracker.ply == ply && tracker.sideToMove == ourPieceColor && lastPostedPly != ply
    }

    /// Post the move. The ply is marked posted first, so no second POST for
    /// it can start while this one is in flight.
    private func post(decision: LichessBotMoveDecision, ply: Int, offeringDraw: Bool, settings: LichessBotSettings) async throws {
        let retries = LichessBotBackoff(initial: .seconds(1), multiplier: 2, cap: .seconds(4))
        var transientFailures = 0
        while true {
            lastPostedPly = ply
            attemptedPosts[ply] = decision.uci
            let started = time.now()
            do {
                try await api.makeMove(gameID: gameID, uci: decision.uci, offeringDraw: offeringDraw)
                let elapsed = LichessBotBackoff.seconds(time.now() - started) * 1000
                await observer.gameEvent(gameID: gameID, .movePosted(ply: ply, uci: decision.uci, offeringDraw: offeringDraw, milliseconds: elapsed))
                await onTurnStatus(gameID, LichessBotTurnStatus(awaitingOurMove: false, ourClock: nil))
                return
            } catch LichessBotGateError.rateLimited {
                // The gate now holds every request for the cooldown. Post
                // again only if it is still our move at this ply (§5.4).
                lastPostedPly = nil
                await observer.gameEvent(gameID: gameID, .moveRejected(ply: ply, uci: decision.uci, error: "rate limited; waiting for the cooldown"))
                guard stillOurMove(at: ply) else { return }
            } catch let error as LichessBotAPIError {
                lastPostedPly = nil
                await observer.gameEvent(gameID: gameID, .moveRejected(ply: ply, uci: decision.uci, error: error.localizedDescription))
                if case .unauthorized = error { throw error }
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
                throw ResyncNeeded(reason: "move rejected: \(error.localizedDescription)")
            } catch let error as LichessBotGateError {
                throw error
            } catch is CancellationError {
                throw CancellationError()
            } catch {
                lastPostedPly = nil
                await observer.gameEvent(gameID: gameID, .moveRejected(ply: ply, uci: decision.uci, error: String(describing: error)))
                guard transientFailures < LichessBotGameSession.maximumTransientPostRetries, stillOurMove(at: ply) else {
                    throw ResyncNeeded(reason: "move post kept failing: \(error)")
                }
                try await time.sleep(for: retries.delay(attempt: transientFailures, unitRandom: Double.random(in: 0...1)))
                transientFailures += 1
                guard stillOurMove(at: ply) else { return }
            }
        }
    }

    private static let maximumTransientPostRetries = 3

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

    /// Claim victory; if Lichess refuses, claim a draw once. Lichess checks
    /// that the opponent is really gone, so a stale `opponentGone` (plan
    /// §12.4) is refused rather than acted on.
    private func claimAfterOpponentLeft() async {
        pendingClaim = nil
        guard !finished else { return }
        do {
            try await api.claimVictory(gameID: gameID)
            await observer.gameEvent(gameID: gameID, .action("claimed victory (opponent gone)"))
        } catch {
            await observer.gameEvent(gameID: gameID, .action("victory claim refused: \(error.localizedDescription); claiming a draw"))
            do {
                try await api.claimDraw(gameID: gameID)
                await observer.gameEvent(gameID: gameID, .action("claimed draw (opponent gone)"))
            } catch {
                await observer.gameEvent(gameID: gameID, .action("draw claim refused: \(error.localizedDescription)"))
            }
        }
    }

    // MARK: - Finishing and chat

    private func finish(status: LichessBotOpenValue<LichessBotGameStatusName>, winner: LichessBotOpenValue<LichessBotColorName>?, settings: LichessBotSettings) async {
        guard !finished else { return }
        finished = true
        pendingClaim?.cancel()
        pendingClaim = nil
        await observer.gameEvent(gameID: gameID, .finished(status: status, winner: winner, localDrawCondition: tracker?.engine.drawCondition))
        await onTurnStatus(gameID, LichessBotTurnStatus(awaitingOurMove: false, ourClock: nil))
        if status.known != .aborted && status.known != .noStart {
            await sendChat(settings.chat.goodbyeEnabled ? settings.chat.goodbyeTemplate : nil, settings: settings)
        }
    }

    private func sendChat(_ template: String?, settings: LichessBotSettings) async {
        guard let template else { return }
        let opponent: String
        if let full = gameFull, let ourColor {
            opponent = (ourColor == .white ? full.black.name : full.white.name) ?? "opponent"
        } else {
            opponent = "opponent"
        }
        let values = [
            "modelID": pinnedMoveSource.info.modelID,
            "source": pinnedMoveSource.info.sourceKind.rawValue,
            "build": "\(BuildInfo.buildNumber)",
            "opponent": opponent,
        ]
        guard let text = LichessBotChat.message(from: template, values: values) else {
            await observer.gameEvent(gameID: gameID, .anomaly("chat message skipped: longer than \(LichessBotChat.maximumLength) characters"))
            return
        }
        do {
            try await api.chat(gameID: gameID, room: settings.chat.room, text: text)
            await observer.gameEvent(gameID: gameID, .action("chat sent"))
        } catch {
            await observer.gameEvent(gameID: gameID, .anomaly("chat failed: \(error.localizedDescription)"))
        }
    }
}
