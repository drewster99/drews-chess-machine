import Foundation

/// The account-level Lichess calls: the event stream and challenge
/// responses. `LichessBotAPIClient` is the real implementation.
protocol LichessBotAccountAPI: Sendable {
    func openEventStream() async throws -> LichessBotChunkStream
    func acceptChallenge(id: String) async throws
    func declineChallenge(id: String, reason: LichessBotDeclineReason) async throws
}

extension LichessBotAPIClient: LichessBotAccountAPI {}

/// What the session manager reports, for the protocol log and the UI.
enum LichessBotManagerEvent: Sendable {
    case eventStreamOpened(attempt: Int)
    case eventStreamEnded(reason: String)
    /// The event stream keeps being closed right after opening: another
    /// client is using the same token. The manager stops (plan §6.1).
    case takeoverSuspected(consecutiveShortStreams: Int)
    case challengeDecision(challengeID: String, challengerID: String, decision: LichessBotChallengeDecision)
    case challengeResponseFailed(challengeID: String, error: String)
    case gameSessionStarted(gameID: String, generation: LichessBotGenerationInfo)
    case gameSessionEnded(gameID: String)
    case anomaly(String)
    /// An event-stream line exactly as received (plan §14.3a transcript).
    case eventStreamLine(Data, receivedAt: Date)
    /// Keep-alive gap statistics for the last minute of the event stream.
    case eventStreamGaps(LichessBotStreamGapSummary)
    /// One silence on the event stream longer than twice Lichess's
    /// keep-alive interval (still short of the stall limit).
    case eventStreamLongGap(seconds: Double)
    /// An outgoing challenge (plan §7.1) was accepted, declined or canceled.
    case outgoingChallengeResolved(challengeID: String, outcome: LichessBotOutgoingChallengeOutcome)
    /// "Play one game": its game started, and the manager has stopped
    /// accepting new games.
    case oneGameStarted(gameID: String)
    /// The manager's `run()` returned; `reason` says why.
    /// Lichess rejected the token (401/403) on an account request.
    case tokenRejected(String)
    case stopped(reason: String)
}

enum LichessBotOutgoingChallengeOutcome: Sendable, Equatable {
    case accepted(gameID: String)
    case declined(reason: String?)
    case canceled
}

/// Gaps between bytes on a stream over one window.
struct LichessBotStreamGapSummary: Sendable, Equatable, Codable {
    let windowSeconds: Double
    let gapCount: Int
    let meanSeconds: Double
    let maximumSeconds: Double
}

/// Accumulates the silences between arrivals on a stream and emits a summary
/// once per window (plan §14.3a: event-stream keep-alives are logged as
/// statistics, not one line each).
struct LichessBotStreamGapTracker: Sendable {
    let window: Duration
    let longGap: Duration
    private var lastArrival: Duration?
    private var windowStart: Duration?
    private var gaps: [Double] = []

    init(window: Duration, longGap: Duration) {
        self.window = window
        self.longGap = longGap
    }

    enum Report: Sendable, Equatable {
        case longGap(seconds: Double)
        case summary(LichessBotStreamGapSummary)
    }

    /// Note an arrival at `now`; returns anything worth logging.
    mutating func arrival(at now: Duration) -> [Report] {
        var reports: [Report] = []
        if let lastArrival {
            let gap = now - lastArrival
            gaps.append(LichessBotBackoff.seconds(gap))
            if gap > longGap {
                reports.append(.longGap(seconds: LichessBotBackoff.seconds(gap)))
            }
        }
        lastArrival = now
        let start = windowStart ?? now
        windowStart = start
        if now - start >= window {
            if let maximum = gaps.max() {
                reports.append(.summary(LichessBotStreamGapSummary(
                    windowSeconds: LichessBotBackoff.seconds(now - start),
                    gapCount: gaps.count,
                    meanSeconds: gaps.reduce(0, +) / Double(gaps.count),
                    maximumSeconds: maximum
                )))
            }
            gaps.removeAll()
            windowStart = now
        }
        return reports
    }

    /// A new connection: the silence across a reconnect is not a gap.
    mutating func reset() {
        lastArrival = nil
        windowStart = nil
        gaps.removeAll()
    }
}

/// Runs the account's event stream: answers challenges through
/// `LichessBotChallengePolicy`, starts and resumes one
/// `LichessBotGameSession` per game, and feeds turn status into the request
/// gate (plan §5.2, §6, §7, E18).
///
/// On (re)connect Lichess replays every current challenge and game, so
/// resuming after a dropped connection or an app relaunch needs nothing
/// special: each `gameStart` for a game without a session starts one.
actor LichessBotSessionManager {

    private let accountAPI: any LichessBotAccountAPI
    private let gameAPI: any LichessBotGameAPI
    private let gate: LichessBotRequestGate
    private let slots: LichessBotModelSlots
    private let ourAccountID: String
    private let time: any LichessBotTimeSource
    private let settingsProvider: @Sendable () async -> LichessBotSettings
    private let gameObserver: any LichessBotGameObserver
    private let onEvent: @Sendable (LichessBotManagerEvent) -> Void

    /// Online (true) or Draining (false) — the operator's choice, or "one
    /// game" once its game started.
    private var acceptingNewGames = true
    /// A 429 hold is in force (plan §5.4). Kept apart from
    /// `acceptingNewGames` so the hold ending can never undo a drain.
    private var rateLimitHold = false
    private var sessions: [String: LichessBotGameSession] = [:]
    private var sessionTasks: [String: Task<Void, Never>] = [:]
    private var opponentByGame: [String: String] = [:]
    private var turnStatus: [String: LichessBotTurnStatus] = [:]
    private var challengeResponseTimes: [Duration] = []
    private var dayStart = Calendar.current.startOfDay(for: Date())
    private var gamesToday = 0
    private var gamesTodayByOpponent: [String: Int] = [:]
    private var countedGames: Set<String> = []
    private var pendingOutgoingChallengeID: String?
    /// Challenges accepted whose `gameStart` hasn't arrived yet. They count
    /// against capacity, so two quick challenges can't both be accepted into
    /// one free slot. An accepted challenge's game has the challenge's id.
    private var acceptedAwaitingStart: [String: (opponentID: String, acceptedAt: Duration)] = [:]
    /// Declines and cancels seen for challenges that weren't (yet) known as
    /// ours: the answer to a challenge can arrive before its POST returns.
    private var recentChallengeOutcomes: [(id: String, outcome: LichessBotOutgoingChallengeOutcome)] = []
    private static let acceptedStartTimeout: Duration = .seconds(60)
    private static let recentOutcomeLimit = 64
    private var oneGameMode = false
    private var gapTracker = LichessBotStreamGapTracker(
        window: .seconds(60),
        longGap: .seconds(2 * LichessBotLimits.eventStreamKeepAliveSeconds)
    )

    init(
        accountAPI: any LichessBotAccountAPI,
        gameAPI: any LichessBotGameAPI,
        gate: LichessBotRequestGate,
        slots: LichessBotModelSlots,
        ourAccountID: String,
        time: any LichessBotTimeSource,
        settingsProvider: @escaping @Sendable () async -> LichessBotSettings,
        gameObserver: any LichessBotGameObserver,
        onEvent: @escaping @Sendable (LichessBotManagerEvent) -> Void
    ) {
        self.accountAPI = accountAPI
        self.gameAPI = gameAPI
        self.gate = gate
        self.slots = slots
        self.ourAccountID = ourAccountID
        self.time = time
        self.settingsProvider = settingsProvider
        self.gameObserver = gameObserver
        self.onEvent = onEvent
    }

    // MARK: - Controls

    /// Online (true) or Draining (false). Draining declines new challenges
    /// with `later`; games in progress continue.
    /// Hold (or release) new games after a 429. Independent of draining.
    func setRateLimitHold(_ held: Bool) {
        rateLimitHold = held
    }

    /// Whether a new challenge could be accepted right now.
    var isAcceptingNewGames: Bool {
        acceptingNewGames && !rateLimitHold
    }

    /// Accepted challenges whose game hasn't started yet (expired ones
    /// dropped). The controller counts them as games in progress, so a
    /// drain or quit waits for them.
    func acceptedAwaitingStartIDs() -> Set<String> {
        let now = time.now()
        acceptedAwaitingStart = acceptedAwaitingStart.filter { now - $0.value.acceptedAt < Self.acceptedStartTimeout }
        return Set(acceptedAwaitingStart.keys)
    }

    func setAcceptingNewGames(_ accepting: Bool) {
        acceptingNewGames = accepting
    }

    var activeGameCount: Int {
        sessions.count
    }

    var activeGameIDs: [String] {
        Array(sessions.keys)
    }

    /// "Play one game" (plan §7.1): accept at most one game at a time, and
    /// stop accepting once a game starts.
    func setOneGameMode(_ on: Bool) {
        oneGameMode = on
    }

    var isOneGameMode: Bool {
        oneGameMode
    }

    /// Note a challenge DCM just sent, so its acceptance, decline or cancel
    /// on the event stream is reported.
    func noteOutgoingChallenge(id: String) {
        if sessions[id] != nil {
            onEvent(.outgoingChallengeResolved(challengeID: id, outcome: .accepted(gameID: id)))
            return
        }
        if let known = recentChallengeOutcomes.last(where: { $0.id == id }) {
            onEvent(.outgoingChallengeResolved(challengeID: id, outcome: known.outcome))
            return
        }
        pendingOutgoingChallengeID = id
    }

    /// Ask every game session to reopen its stream (after the Mac wakes:
    /// connections held across sleep are usually dead; plan E34).
    func resyncAllSessions(reason: String) {
        for session in sessions.values {
            session.requestResync(reason: reason)
        }
    }

    var outgoingChallengeID: String? {
        pendingOutgoingChallengeID
    }

    /// Forget the pending outgoing challenge (it expired, or Lichess no
    /// longer knows it).
    func clearOutgoingChallenge() {
        pendingOutgoingChallengeID = nil
    }

    /// Games started today against `opponentID` (for the Challenge sheet's
    /// bot-pair check).
    func gamesToday(against opponentID: String) -> Int {
        rollDayIfNeeded()
        return gamesTodayByOpponent[opponentID] ?? 0
    }

    /// Seed today's counts from the record store after a relaunch, so the
    /// daily limits (plan §7) survive a restart.
    func seedDailyCounts(gameIDs: [String], opponentByGame: [String: String]) {
        rollDayIfNeeded()
        for gameID in gameIDs where !countedGames.contains(gameID) {
            countedGames.insert(gameID)
            gamesToday += 1
            if let opponent = opponentByGame[gameID] {
                gamesTodayByOpponent[opponent, default: 0] += 1
            }
        }
    }

    /// Resign every game in progress (the operator's Resign all).
    func resignAll() async {
        for (gameID, session) in sessions {
            do {
                try await session.resignNow()
            } catch {
                onEvent(.anomaly("resign failed for \(gameID): \(error.localizedDescription)"))
            }
        }
    }

    // MARK: - Running

    /// Hold the event stream until cancelled, the gate closes, or a
    /// takeover is detected. Reconnects with backoff otherwise.
    func run() async {
        var attempt = 0
        var shortStreams = 0
        let stopReason: String
        runLoop: while true {
            if Task.isCancelled {
                stopReason = "cancelled"
                break
            }
            let settings = await settingsProvider()
            var openedAt: Duration?
            var closedByServer = false
            do {
                let chunks = try await accountAPI.openEventStream()
                openedAt = time.now()
                onEvent(.eventStreamOpened(attempt: attempt))
                let items = LichessBotStreamReader.items(
                    from: chunks,
                    time: time,
                    stallTimeout: { .seconds(settings.connection.eventStreamStallTimeoutSeconds) },
                    checkInterval: .seconds(1)
                )
                gapTracker.reset()
                for try await item in items {
                    for report in gapTracker.arrival(at: time.now()) {
                        switch report {
                        case .longGap(let seconds):
                            onEvent(.eventStreamLongGap(seconds: seconds))
                        case .summary(let summary):
                            onEvent(.eventStreamGaps(summary))
                        }
                    }
                    switch item {
                    case .line(let data):
                        onEvent(.eventStreamLine(data, receivedAt: Date()))
                        await handleEventLine(data)
                    case .keepAlive:
                        break
                    case .oversizeLineDiscarded(let byteCount):
                        onEvent(.anomaly("event stream: discarded an oversize line of \(byteCount) bytes"))
                    case .truncatedAtEnd(let byteCount):
                        onEvent(.anomaly("event stream ended mid-line with \(byteCount) bytes pending"))
                    }
                }
                closedByServer = true
                onEvent(.eventStreamEnded(reason: "server closed the stream"))
            } catch is CancellationError {
                stopReason = "cancelled"
                break runLoop
            } catch LichessBotGateError.closed(let reason) {
                stopReason = "request gate closed: \(reason)"
                break runLoop
            } catch let error as LichessBotAPIError {
                onEvent(.eventStreamEnded(reason: error.localizedDescription))
                if case .unauthorized = error {
                    stopReason = error.localizedDescription
                    break runLoop
                }
            } catch {
                onEvent(.eventStreamEnded(reason: String(describing: error)))
            }

            // Takeover: Lichess allows one event stream per token and closes
            // the older one when another client opens its own, so the
            // signature is a stream that opened and was then *closed by the
            // server* right away, again and again. Failed connection
            // attempts, stalls and transport errors don't count — they are
            // network or server problems, handled by the backoff.
            if let openedAt {
                let lived = time.now() - openedAt
                if lived >= Self.healthyStreamDuration {
                    attempt = 0
                }
                if lived >= Self.takeoverWindow {
                    shortStreams = 0
                } else if closedByServer {
                    shortStreams += 1
                    if shortStreams >= Self.takeoverThreshold {
                        onEvent(.takeoverSuspected(consecutiveShortStreams: shortStreams))
                        stopReason = "another client appears to be using this token"
                        break runLoop
                    }
                }
            }

            let backoff = LichessBotBackoff(
                initial: .seconds(settings.connection.reconnectInitialSeconds),
                multiplier: 2,
                cap: .seconds(settings.connection.reconnectCapSeconds)
            )
            do {
                try await time.sleep(for: backoff.delay(attempt: attempt, unitRandom: Double.random(in: 0...1)))
            } catch {
                stopReason = "cancelled"
                break runLoop
            }
            attempt += 1
        }
        onEvent(.stopped(reason: stopReason))
    }

    /// Cancel every game session (going offline without draining: the games
    /// are abandoned and reconciled later).
    func abandonAllSessions() {
        for task in sessionTasks.values {
            task.cancel()
        }
    }

    static let takeoverWindow: Duration = .seconds(30)
    /// A stream that stayed up this long was healthy: the next reconnect
    /// starts the backoff over (plan §6).
    static let healthyStreamDuration: Duration = .seconds(300)
    static let takeoverThreshold = 3

    // MARK: - Events

    private func handleEventLine(_ data: Data) async {
        let event: LichessBotEvent
        do {
            event = try LichessBotEvent.decode(data)
        } catch {
            onEvent(.anomaly("undecodable event line: \(error)"))
            return
        }
        switch event {
        case .challenge(let challenge, let compat):
            await handleChallenge(challenge, compat: compat)
        case .gameStart(let info):
            if info.gameId == pendingOutgoingChallengeID {
                // An accepted challenge's game has the challenge's id.
                pendingOutgoingChallengeID = nil
                onEvent(.outgoingChallengeResolved(challengeID: info.gameId, outcome: .accepted(gameID: info.gameId)))
            }
            await startSessionIfNeeded(info)
        case .gameFinish(let info):
            countGame(info)
            // The game stream may never deliver the final state (plan E22).
            sessions[info.gameId]?.requestResync(reason: "gameFinish on the event stream")
        case .challengeDeclined(let reference):
            resolveChallenge(reference.id, outcome: .declined(reason: reference.declineReason ?? reference.declineReasonKey))
        case .challengeCanceled(let reference):
            acceptedAwaitingStart[reference.id] = nil
            resolveChallenge(reference.id, outcome: .canceled)
        case .unknown(let type):
            onEvent(.anomaly("unknown event type \(type)"))
        }
    }

    /// Report the answer to our outgoing challenge, or remember it in case
    /// the challenge is noted as ours a moment later.
    private func resolveChallenge(_ id: String, outcome: LichessBotOutgoingChallengeOutcome) {
        if id == pendingOutgoingChallengeID {
            pendingOutgoingChallengeID = nil
            onEvent(.outgoingChallengeResolved(challengeID: id, outcome: outcome))
            return
        }
        recentChallengeOutcomes.append((id, outcome))
        if recentChallengeOutcomes.count > Self.recentOutcomeLimit {
            recentChallengeOutcomes.removeFirst(recentChallengeOutcomes.count - Self.recentOutcomeLimit)
        }
    }

    private func handleChallenge(_ challenge: LichessBotChallenge, compat: LichessBotCompat?) async {
        let settings = await settingsProvider()
        rollDayIfNeeded()
        let now = time.now()
        challengeResponseTimes.removeAll { now - $0 > .seconds(60) }
        acceptedAwaitingStart = acceptedAwaitingStart.filter { now - $0.value.acceptedAt < Self.acceptedStartTimeout }
        let modelReady = await slots.sourceAvailable(for: settings.model)
        var byOpponent: [String: Int] = [:]
        for opponent in opponentByGame.values {
            byOpponent[opponent, default: 0] += 1
        }
        for pending in acceptedAwaitingStart.values {
            byOpponent[pending.opponentID, default: 0] += 1
        }
        let context = LichessBotChallengeContext(
            acceptingNewGames: acceptingNewGames && !rateLimitHold,
            modelReady: modelReady,
            activeGames: sessions.count + acceptedAwaitingStart.count,
            activeGamesByOpponent: byOpponent,
            gamesToday: gamesToday,
            gamesTodayByOpponent: gamesTodayByOpponent,
            challengeResponsesInLastMinute: challengeResponseTimes.count
        )
        var challengeSettings = settings.challenge
        if oneGameMode {
            challengeSettings.maxConcurrentGames = 1
            challengeSettings.gamesReservedForHumans = 0
        }
        let decision = LichessBotChallengePolicy.decide(challenge, compat: compat, settings: challengeSettings, context: context)
        onEvent(.challengeDecision(challengeID: challenge.id, challengerID: challenge.challenger.id, decision: decision))

        switch decision {
        case .ignore:
            return
        case .accept:
            // Make sure a generation is built before accepting, so the first
            // move never waits on a network build (E15).
            // Hold the slot across the model build: another challenge
            // handled meanwhile must see it as taken.
            acceptedAwaitingStart[challenge.id] = (challenge.challenger.id, time.now())
            do {
                _ = try await slots.ready(for: settings.model)
            } catch {
                acceptedAwaitingStart[challenge.id] = nil
                onEvent(.anomaly("model not ready, declining \(challenge.id): \(error.localizedDescription)"))
                await decline(challenge, reason: .later)
                return
            }
            await accept(challenge)
        case .decline(let reason, _):
            await decline(challenge, reason: reason)
        }
    }

    private func accept(_ challenge: LichessBotChallenge) async {
        challengeResponseTimes.append(time.now())
        do {
            try await accountAPI.acceptChallenge(id: challenge.id)
            if acceptedAwaitingStart[challenge.id] != nil {
                acceptedAwaitingStart[challenge.id] = (challenge.challenger.id, time.now())
            }
        } catch {
            acceptedAwaitingStart[challenge.id] = nil
            reportResponseFailure(challenge, error)
        }
    }

    private func decline(_ challenge: LichessBotChallenge, reason: LichessBotDeclineReason) async {
        challengeResponseTimes.append(time.now())
        do {
            try await accountAPI.declineChallenge(id: challenge.id, reason: reason)
        } catch {
            reportResponseFailure(challenge, error)
        }
    }

    private func reportResponseFailure(_ challenge: LichessBotChallenge, _ error: Error) {
        onEvent(.challengeResponseFailed(challengeID: challenge.id, error: error.localizedDescription))
        if let apiError = error as? LichessBotAPIError, case .unauthorized = apiError {
            onEvent(.tokenRejected(apiError.localizedDescription))
        }
    }

    private func startSessionIfNeeded(_ info: LichessBotGameEventInfo) async {
        let gameID = info.gameId
        acceptedAwaitingStart[gameID] = nil
        guard sessions[gameID] == nil else { return }
        let settings = await settingsProvider()
        let generation: LichessBotModelGeneration
        do {
            generation = try await slots.ready(for: settings.model)
        } catch {
            // A game is running and there is no model to play it: the
            // session can't move. Say so loudly; the game will be lost on
            // time or aborted by Lichess, and reconciled afterwards.
            onEvent(.anomaly("game \(gameID) started but no model is available: \(error.localizedDescription)"))
            return
        }
        if let opponent = info.opponent?.id {
            opponentByGame[gameID] = opponent
        }
        countGame(info)

        let slots = self.slots
        let settingsProvider = self.settingsProvider
        let onEvent = self.onEvent
        let session = LichessBotGameSession(
            gameID: gameID,
            ourAccountID: ourAccountID,
            api: gameAPI,
            moveSource: generation,
            latestMoveSource: {
                let settings = await settingsProvider()
                do {
                    return try await slots.ready(for: settings.model)
                } catch {
                    onEvent(.anomaly("game \(gameID): no current model for mid-game refresh: \(error.localizedDescription)"))
                    return nil
                }
            },
            settingsProvider: settingsProvider,
            observer: gameObserver,
            time: time,
            onTurnStatus: { [weak self] gameID, status in
                await self?.updateTurnStatus(gameID: gameID, status: status)
            }
        )
        sessions[gameID] = session
        onEvent(.gameSessionStarted(gameID: gameID, generation: generation.info))
        if oneGameMode {
            oneGameMode = false
            acceptingNewGames = false
            onEvent(.oneGameStarted(gameID: gameID))
        }
        sessionTasks[gameID] = Task { [weak self] in
            await session.run()
            await self?.sessionEnded(gameID: gameID)
        }
    }

    private func sessionEnded(gameID: String) async {
        sessions[gameID] = nil
        sessionTasks[gameID] = nil
        opponentByGame[gameID] = nil
        turnStatus[gameID] = nil
        await applyTurnStatusToGate()
        onEvent(.gameSessionEnded(gameID: gameID))
    }

    private func countGame(_ info: LichessBotGameEventInfo) {
        rollDayIfNeeded()
        guard !countedGames.contains(info.gameId) else { return }
        countedGames.insert(info.gameId)
        gamesToday += 1
        if let opponent = info.opponent?.id {
            gamesTodayByOpponent[opponent, default: 0] += 1
        }
    }

    /// Daily limits use the local calendar day (plan E50).
    private func rollDayIfNeeded() {
        let today = Calendar.current.startOfDay(for: Date())
        guard today != dayStart else { return }
        dayStart = today
        gamesToday = 0
        gamesTodayByOpponent = [:]
        countedGames = countedGames.filter { sessions[$0] != nil }
    }

    // MARK: - Gate eligibility

    private func updateTurnStatus(gameID: String, status: LichessBotTurnStatus) async {
        turnStatus[gameID] = status
        await applyTurnStatusToGate()
    }

    private func applyTurnStatusToGate() async {
        let settings = await settingsProvider()
        let awaiting = turnStatus.values.filter(\.awaitingOurMove)
        let lowClock = awaiting.contains { status in
            guard let clock = status.ourClock else { return false }
            return clock.value < settings.connection.lowClockThresholdMilliseconds
        }
        await gate.setGamesAwaitingOurMove(awaiting.count)
        await gate.setLowClockUrgency(lowClock)
    }
}
