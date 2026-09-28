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
    /// The manager's `run()` returned; `reason` says why.
    case stopped(reason: String)
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

    private var acceptingNewGames = true
    private var sessions: [String: LichessBotGameSession] = [:]
    private var sessionTasks: [String: Task<Void, Never>] = [:]
    private var opponentByGame: [String: String] = [:]
    private var turnStatus: [String: LichessBotTurnStatus] = [:]
    private var challengeResponseTimes: [Duration] = []
    private var dayStart = Calendar.current.startOfDay(for: Date())
    private var gamesToday = 0
    private var gamesTodayByOpponent: [String: Int] = [:]
    private var countedGames: Set<String> = []

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
    func setAcceptingNewGames(_ accepting: Bool) {
        acceptingNewGames = accepting
    }

    var activeGameCount: Int {
        sessions.count
    }

    var activeGameIDs: [String] {
        Array(sessions.keys)
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
                for try await item in items {
                    if case .line(let data) = item {
                        await handleEventLine(data)
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
            await startSessionIfNeeded(info)
        case .gameFinish(let info):
            countGame(info)
            // The game stream may never deliver the final state (plan E22).
            sessions[info.gameId]?.requestResync(reason: "gameFinish on the event stream")
        case .challengeCanceled, .challengeDeclined:
            break
        case .unknown(let type):
            onEvent(.anomaly("unknown event type \(type)"))
        }
    }

    private func handleChallenge(_ challenge: LichessBotChallenge, compat: LichessBotCompat?) async {
        let settings = await settingsProvider()
        rollDayIfNeeded()
        let now = time.now()
        challengeResponseTimes.removeAll { now - $0 > .seconds(60) }
        let modelReady = await slots.sourceAvailable(for: settings.model)
        var byOpponent: [String: Int] = [:]
        for opponent in opponentByGame.values {
            byOpponent[opponent, default: 0] += 1
        }
        let context = LichessBotChallengeContext(
            acceptingNewGames: acceptingNewGames,
            modelReady: modelReady,
            activeGames: sessions.count,
            activeGamesByOpponent: byOpponent,
            gamesToday: gamesToday,
            gamesTodayByOpponent: gamesTodayByOpponent,
            challengeResponsesInLastMinute: challengeResponseTimes.count
        )
        let decision = LichessBotChallengePolicy.decide(challenge, compat: compat, settings: settings.challenge, context: context)
        onEvent(.challengeDecision(challengeID: challenge.id, challengerID: challenge.challenger.id, decision: decision))

        switch decision {
        case .ignore:
            return
        case .accept:
            // Make sure a generation is built before accepting, so the first
            // move never waits on a network build (E15).
            do {
                _ = try await slots.ready(for: settings.model)
            } catch {
                onEvent(.anomaly("model not ready, declining \(challenge.id): \(error.localizedDescription)"))
                await respond(to: challenge, accept: false, reason: .later)
                return
            }
            await respond(to: challenge, accept: true, reason: nil)
        case .decline(let reason, _):
            await respond(to: challenge, accept: false, reason: reason)
        }
    }

    private func respond(to challenge: LichessBotChallenge, accept: Bool, reason: LichessBotDeclineReason?) async {
        challengeResponseTimes.append(time.now())
        do {
            if accept {
                try await accountAPI.acceptChallenge(id: challenge.id)
            } else {
                try await accountAPI.declineChallenge(id: challenge.id, reason: reason ?? .generic)
            }
        } catch {
            onEvent(.challengeResponseFailed(challengeID: challenge.id, error: error.localizedDescription))
        }
    }

    private func startSessionIfNeeded(_ info: LichessBotGameEventInfo) async {
        let gameID = info.gameId
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
