import AppKit
import Foundation

/// Everything the bot's runtime reports, funneled onto the main actor in
/// order. Game sessions, the manager, the gate, the reconciler and the API
/// client all run off the main actor; they `yield` here without waiting, so
/// a busy main thread can never delay a move (plan §15: no global hub for
/// game traffic).
enum LichessBotControllerEvent: Sendable {
    case game(String, LichessBotGameEvent)
    case request(LichessBotRequestRecord)
    case manager(LichessBotManagerEvent)
    case gate(LichessBotGateEvent)
    case reconciler(LichessBotReconcilerEvent)
    case journalWriteFailed(gameID: String, error: String)
    case gameFinished(gameID: String)
}

/// The game observer the sessions report to: it forwards to the controller's
/// event stream without blocking.
struct LichessBotControllerFeed: LichessBotGameObserver {
    let continuation: AsyncStream<LichessBotControllerEvent>.Continuation

    func gameEvent(gameID: String, _ event: LichessBotGameEvent) async {
        continuation.yield(.game(gameID, event))
    }
}

/// Owns the Lichess bot's lifecycle and publishes its state to the UI (plan
/// §13, §14, §15). One per app, created at app level and kept for the app's
/// lifetime: closing the window never stops the bot.
///
/// **States.** Offline → Connecting → Online. From Online: Draining (no new
/// games; running games finish, then Offline), or Error (token, takeover,
/// breaker, an unexpected stop), which stays until the operator acts.
///
/// **Runtime.** Going online builds a fresh runtime — instance lock,
/// request gate, API client, model slots, journal writer, record store,
/// reconciler and session manager — and going offline tears it down. The
/// token is read from the Keychain once, when going online.
@MainActor
@Observable
final class LichessBotController {

    enum ConnectionState: Equatable {
        case offline
        case connecting
        case online
        case draining
        case error(String)

        var label: String {
            switch self {
            case .offline: return "Offline"
            case .connecting: return "Connecting"
            case .online: return "Online"
            case .draining: return "Draining"
            case .error: return "Error"
            }
        }
    }

    enum TokenState: Equatable {
        case unknown
        case none
        case checking
        case saved(LichessBotTokenInfo)
        case error(String)
    }

    /// Why the "Finishing games" sheet is up (plan §13).
    enum FinishingPurpose: Equatable {
        case goOffline
        case quit
    }

    struct PendingChallenge: Equatable {
        let id: String
        let username: String
        let sentAt: Date
    }

    struct Alarm: Identifiable, Equatable {
        let id: Int
        let at: Date
        let text: String
    }

    // MARK: - Published state

    private(set) var connection: ConnectionState = .offline
    private(set) var settings: LichessBotSettings
    private(set) var settingsError: String?
    private(set) var tokenState: TokenState = .unknown
    private(set) var account: LichessBotAccount?
    private(set) var gateSnapshot: LichessBotRequestGate.Snapshot?
    private(set) var generation: LichessBotGenerationInfo?
    /// Games in progress plus finished games kept for a while, oldest first.
    private(set) var games: [LichessBotLiveGame] = []
    private(set) var activeGameIDs: Set<String> = []
    private(set) var pendingChallenge: PendingChallenge?
    private(set) var lastChallengeOutcome: String?
    private(set) var oneGameRequested = false
    private(set) var alarms: [Alarm] = []
    private(set) var unreconciledGameIDs: [String] = []
    private(set) var finishing: FinishingPurpose?
    private(set) var index: LichessBotIndex.File?
    private(set) var onlineBots: [LichessBotUserSummary] = []
    /// A drain has finished its games and is filing their records before
    /// going offline.
    private(set) var isFilingRecords = false
    /// Games in the grid/list the operator has dismissed.
    var focusedGameID: String?
    var showsGrid = false

    // MARK: - Configuration

    let dataDirectory: LichessBotDataDirectory
    private let defaults: UserDefaults
    private let tokenStore = LichessBotTokenStore()
    private let fileQueue = LichessBotFileQueue()
    private let modelProvider: any LichessBotModelProvider
    let protocolLog: LichessBotProtocolLog
    private var nextAlarmID = 0

    // MARK: - Runtime

    private struct Runtime {
        let client: LichessBotAPIClient
        let manager: LichessBotSessionManager
        let slots: LichessBotModelSlots
        let reconciler: LichessBotReconciler
        let recordStore: LichessBotRecordStore
        let journal: LichessBotJournalWriter
        let lock: LichessBotInstanceLock
        let continuation: AsyncStream<LichessBotControllerEvent>.Continuation
        var tasks: [Task<Void, Never>]
        let sleepActivity: NSObjectProtocol
        let wakeObserver: NSObjectProtocol
    }

    private var runtime: Runtime?
    /// Bumped each time a runtime starts or stops. Work tied to one runtime
    /// (its event consumer, its poll loop) stops acting as soon as this
    /// moves on, so nothing queued by a torn-down runtime can change state.
    private var runtimeGeneration = 0
    /// Accepted challenges whose games haven't started (from the manager).
    /// A drain or quit waits for them like games in progress.
    private(set) var acceptedAwaitingStartIDs: Set<String> = []
    private var quitReplyPending = false
    /// The account-wide request gate (plan §5). One for the controller's
    /// lifetime — not one per runtime — so a 429 cooldown and the breaker's
    /// history survive going offline and back, and offline account calls
    /// share the same single flight.
    private let gate: LichessBotRequestGate
    /// Where gate events go while a runtime is up.
    private let gateEventSink = SyncBox<AsyncStream<LichessBotControllerEvent>.Continuation?>(nil)
    /// After a 429: new games are declined until this time, then accepting
    /// resumes (plan §5.4, `postRateLimitDrainMinutes`).
    private var rateLimitHoldUntil: Date?
    /// The operator (or "Play one game") asked to stop taking games: the bot
    /// goes offline once they finish. A rate-limit hold alone does not.
    private var drainRequested = false

    init(modelProvider: any LichessBotModelProvider, defaults: UserDefaults = .standard, dataDirectory: LichessBotDataDirectory = .standard) {
        self.modelProvider = modelProvider
        self.defaults = defaults
        self.dataDirectory = dataDirectory
        let fileQueue = self.fileQueue
        self.protocolLog = LichessBotProtocolLog(directory: dataDirectory, fileQueue: fileQueue) { error in
            SessionLogger.shared.log("[ALARM] LICHESS-BOT protocol log write failed: \(error.localizedDescription)")
        }
        let loadedSettings: LichessBotSettings
        do {
            loadedSettings = try LichessBotSettingsStore.load(from: defaults)
        } catch {
            loadedSettings = LichessBotSettings()
            settingsError = error.localizedDescription
        }
        settings = loadedSettings
        let log = protocolLog
        let sink = gateEventSink
        gate = LichessBotRequestGate(
            time: LichessBotSystemTimeSource(),
            breakerWindow: .seconds(loadedSettings.connection.rateLimitBreakerWindowMinutes * 60)
        ) { event in
            log.record(LichessBotController.kind(of: event), LichessBotController.describe(event))
            sink.value?.yield(.gate(event))
        }
    }

    /// The configured account id. Lichess ids are lowercase.
    private var accountID: String {
        settings.connection.expectedAccountID.lowercased()
    }

    /// An error's text, safe to show and log (plan §10.3).
    private static func safeDescription(_ error: Error) -> String {
        LichessBotRedaction.redact(error.localizedDescription)
    }

    // MARK: - Derived

    var isRunning: Bool {
        runtime != nil
    }

    /// Games in progress, or accepted and about to start.
    var hasGamesInPlay: Bool {
        !activeGameIDs.isEmpty || !acceptedAwaitingStartIDs.isEmpty
    }

    var gamesInProgress: [LichessBotLiveGame] {
        games.filter { activeGameIDs.contains($0.id) }
    }

    /// The game the single-game view shows: the operator's pick if it still
    /// exists, otherwise the first-started game in progress, otherwise the
    /// most recent game (plan §14.3a).
    var displayedGame: LichessBotLiveGame? {
        if let focusedGameID, let game = games.first(where: { $0.id == focusedGameID }) {
            return game
        }
        return gamesInProgress.first ?? games.last
    }

    var hasChallengeScope: Bool {
        if case .saved(let info) = tokenState {
            return info.scopeList.contains("challenge:write")
        }
        return false
    }

    /// Our record against `opponentID` from the finished-games index.
    func headToHead(against opponentID: String?) -> (wins: Int, draws: Int, losses: Int) {
        guard let opponentID, let rows = index?.rows else { return (0, 0, 0) }
        var record = (wins: 0, draws: 0, losses: 0)
        for row in rows where row.opponentID == opponentID {
            switch row.ourScore {
            case 1: record.wins += 1
            case 0.5: record.draws += 1
            case 0: record.losses += 1
            default: break
            }
        }
        return record
    }

    // MARK: - Settings

    /// Apply new settings. Invalid settings are rejected whole (plan §12.1).
    func updateSettings(_ newSettings: LichessBotSettings) throws {
        try LichessBotSettingsStore.save(newSettings, to: defaults)
        apply(newSettings)
    }

    /// The operator's explicit reset after unreadable saved settings.
    func resetSettings() throws {
        try LichessBotSettingsStore.reset(in: defaults)
        apply(LichessBotSettings())
    }

    /// Make saved settings the ones in force: the UI's copy, the running
    /// bot's copy, and the request gate's breaker window.
    private func apply(_ newSettings: LichessBotSettings) {
        settings = newSettings
        settingsError = nil
        settingsBox?.value = newSettings
        let gate = self.gate
        let window = Duration.seconds(newSettings.connection.rateLimitBreakerWindowMinutes * 60)
        Task {
            await gate.setBreakerWindow(window)
        }
    }

    // MARK: - Token and account (plan §12.2)

    /// Read the Keychain and, if a token is stored, verify it with Lichess.
    func refreshTokenState() async {
        let token: String?
        do {
            token = try await readToken()
        } catch {
            tokenState = .error(Self.safeDescription(error))
            return
        }
        guard let token else {
            tokenState = .none
            return
        }
        await verify(token: token, saveOnSuccess: false)
    }

    /// Check a pasted token and, if it belongs to the configured account and
    /// carries `bot:play`, store it in the Keychain.
    func submitToken(_ rawToken: String) async {
        let token = rawToken.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !token.isEmpty else {
            tokenState = .error("The token is empty")
            return
        }
        await verify(token: token, saveOnSuccess: true)
    }

    func removeToken() async {
        let store = tokenStore
        let accountID = self.accountID
        do {
            try await fileQueue.run {
                try store.delete(account: accountID)
            }
            tokenState = .none
            account = nil
            protocolLog.record(.account, "token removed")
        } catch {
            tokenState = .error(Self.safeDescription(error))
        }
    }

    /// The stored token, read off the main actor (Keychain calls block).
    private func readToken() async throws -> String? {
        let store = tokenStore
        let accountID = self.accountID
        return try await fileQueue.run {
            try store.read(account: accountID)
        }
    }

    /// Whether the guarded BOT upgrade may be offered: a verified token for
    /// an account that is not yet a BOT and has played no games.
    var canUpgradeToBot: Bool {
        guard case .saved = tokenState, let account, !account.isBot, let count = account.count else { return false }
        return count.all == 0
    }

    /// `POST /api/bot/account/upgrade` — irreversible. Only the guarded
    /// Settings flow calls this, after the operator typed the confirmation.
    func upgradeToBot() async {
        guard canUpgradeToBot else {
            raiseAlarm("BOT upgrade refused: the account must have played no games and not already be a BOT")
            return
        }
        do {
            let client = try await accountClient()
            try await client.upgradeToBot()
            protocolLog.record(.account, "account upgraded to BOT")
            account = try await client.account()
        } catch {
            raiseAlarm("BOT upgrade failed: \(Self.safeDescription(error))")
        }
    }

    private func verify(token: String, saveOnSuccess: Bool) async {
        tokenState = .checking
        let expected = accountID
        do {
            let client = try makeClient(token: token)
            guard let info = try await client.testToken() else {
                tokenState = .error("Lichess reports this token as invalid or revoked")
                return
            }
            guard info.userId == expected else {
                tokenState = .error("This token belongs to \(info.userId), not \(expected)")
                return
            }
            guard info.scopeList.contains("bot:play") else {
                tokenState = .error("This token lacks the bot:play scope (it has: \(info.scopes.isEmpty ? "none" : info.scopes))")
                return
            }
            if let expires = info.expires, Date(timeIntervalSince1970: Double(expires) / 1000) <= Date() {
                tokenState = .error("This token has expired")
                return
            }
            if saveOnSuccess {
                let store = tokenStore
                try await fileQueue.run {
                    try store.save(token: token, account: expected)
                }
                protocolLog.record(.account, "token saved", fields: ["scopes": info.scopes])
            }
            account = try await client.account()
            tokenState = .saved(info)
        } catch {
            tokenState = .error(Self.safeDescription(error))
        }
    }

    /// A client for account calls outside the game loop: the runtime's own
    /// while online, otherwise one for the stored token. Either way it goes
    /// through the controller's one request gate.
    private func accountClient() async throws -> LichessBotAPIClient {
        if let runtime {
            return runtime.client
        }
        guard let token = try await readToken() else {
            throw LichessBotControllerError.noToken
        }
        return try makeClient(token: token)
    }

    private func makeClient(token: String) throws -> LichessBotAPIClient {
        let log = protocolLog
        return LichessBotAPIClient(
            baseURL: try LichessBotAPIClient.lichessBaseURL(),
            token: token,
            transport: LichessBotURLSessionTransport(),
            gate: gate,
            onRequest: { record in log.record(.request, Self.describe(record)) }
        )
    }

    // MARK: - Going online and offline (plan §13)

    /// Go online. With `oneGame`, take one game and then go offline (plan
    /// §7.1).
    func goOnline(oneGame: Bool = false) async {
        switch connection {
        case .offline, .error:
            break
        case .connecting, .online, .draining:
            return
        }
        if let settingsError {
            connection = .error("Settings: \(settingsError)")
            return
        }
        connection = .connecting
        oneGameRequested = oneGame
        drainRequested = false
        do {
            try await startRuntime(oneGame: oneGame)
            // Something may have stopped the new runtime while it started.
            guard connection == .connecting, runtime != nil else { return }
            // A post-429 hold outlives going offline and back (plan §5.4).
            connection = rateLimitHoldUntil == nil ? .online : .draining
            protocolLog.record(.lifecycle, oneGame ? "online (one game)" : "online")
            SessionLogger.shared.log("[LICHESS-BOT] online\(oneGame ? " (one game)" : "") as \(accountID)")
        } catch {
            tearDownRuntime()
            oneGameRequested = false
            let text = Self.safeDescription(error)
            connection = .error(text)
            protocolLog.record(.lifecycle, "going online failed: \(text)")
            SessionLogger.shared.log("[ALARM] LICHESS-BOT going online failed: \(text)")
        }
    }

    /// Stop accepting new games; go offline once running games finish.
    func drain() async {
        guard let runtime, connection == .online || connection == .draining else { return }
        drainRequested = true
        let manager = runtime.manager
        await manager.setAcceptingNewGames(false)
        guard self.runtime?.manager === manager, connection == .online || connection == .draining else { return }
        connection = .draining
        protocolLog.record(.lifecycle, "draining")
        SessionLogger.shared.log("[LICHESS-BOT] draining (\(activeGameIDs.count) game(s) in progress)")
        finishIfDrained()
    }

    /// Go offline. With games in progress this drains and shows the
    /// "Finishing games" sheet instead (plan §13).
    func goOffline() async {
        guard runtime != nil else { return }
        if !hasGamesInPlay {
            stopRuntime(reason: "operator went offline")
            return
        }
        finishing = .goOffline
        LichessBotWindowLauncher.openWindow(controller: self)
        await drain()
    }

    /// Resign every game in progress (the sheet's **Abort**, and Resign all).
    func resignAll() async {
        guard let runtime else { return }
        protocolLog.record(.lifecycle, "resign all (\(activeGameIDs.count) game(s))")
        await runtime.manager.resignAll()
    }

    /// The sheet's **Quit now** / abandon: stop at once. Games are
    /// abandoned; the opponents can claim after the timeout, and launch
    /// recovery reconciles them.
    func abandonAndStop() {
        protocolLog.record(.lifecycle, "abandoning \(activeGameIDs.count) game(s)")
        stopRuntime(reason: "operator abandoned running games")
        completeQuitIfPending(quit: true)
    }

    /// The sheet's **Cancel**: close it; the bot stays Draining.
    func cancelFinishing() {
        let purpose = finishing
        finishing = nil
        if purpose == .quit {
            completeQuitIfPending(quit: false)
        }
    }

    // MARK: - Quit (plan §13)

    /// `applicationShouldTerminate`: quit at once when the bot holds no
    /// games; otherwise drain, show the sheet, and quit when the games end.
    func applicationShouldTerminate() -> NSApplication.TerminateReply {
        guard runtime != nil else { return .terminateNow }
        if !hasGamesInPlay {
            stopRuntime(reason: "app quit")
            return .terminateNow
        }
        quitReplyPending = true
        finishing = .quit
        // The "Finishing games" sheet lives in the bot window: bring it up,
        // or quitting would seem to do nothing.
        LichessBotWindowLauncher.openWindow(controller: self)
        Task {
            await drain()
        }
        return .terminateLater
    }

    private func completeQuitIfPending(quit: Bool) {
        guard quitReplyPending else { return }
        quitReplyPending = false
        NSApp.reply(toApplicationShouldTerminate: quit)
    }

    // MARK: - Challenges (plan §7.1)

    func refreshOnlineBots() async {
        do {
            let client = try await accountClient()
            onlineBots = try await client.onlineBots(count: 100).sorted { $0.username.lowercased() < $1.username.lowercased() }
        } catch {
            raiseAlarm("Loading online bots failed: \(Self.safeDescription(error))")
        }
    }

    func lookUpUser(_ username: String) async throws -> LichessBotUserSummary {
        let client = try await accountClient()
        return try await client.user(username: username)
    }

    /// Send a challenge. The bot must be online (the game arrives on its
    /// event stream); only one outgoing challenge may be pending.
    func sendChallenge(to username: String, request: LichessBotOutgoingChallenge) async throws {
        guard let runtime, connection == .online else {
            throw LichessBotControllerError.notOnline
        }
        guard pendingChallenge == nil else {
            throw LichessBotControllerError.challengeAlreadyPending
        }
        guard hasChallengeScope else {
            throw LichessBotControllerError.missingChallengeScope
        }
        let manager = runtime.manager
        let client = runtime.client
        let target = username.lowercased()
        let today = await manager.gamesToday(against: target)
        if today >= settings.challenge.botPairDailyStop {
            throw LichessBotControllerError.dailyLimitReached(opponent: username, games: today)
        }
        let created = try await client.challenge(username: username, request: request)
        guard self.runtime?.manager === manager else {
            // The bot went offline while the challenge was being sent:
            // withdraw it, or an acceptance would start an abandoned game.
            withdraw(challengeID: created.id, client: client)
            throw LichessBotControllerError.notOnline
        }
        pendingChallenge = PendingChallenge(id: created.id, username: username, sentAt: Date())
        lastChallengeOutcome = nil
        // May resolve at once, if the answer already arrived.
        await manager.noteOutgoingChallenge(id: created.id)
        protocolLog.record(.challenge, "challenge sent to \(username)", fields: ["id": created.id, "rated": "\(request.rated)", "clock": "\(request.clockLimitSeconds)+\(request.clockIncrementSeconds)", "color": request.color.rawValue])
    }

    func cancelChallenge() async {
        guard let runtime, let pendingChallenge else { return }
        do {
            try await runtime.client.cancelChallenge(id: pendingChallenge.id)
            await runtime.manager.clearOutgoingChallenge()
            if self.pendingChallenge?.id == pendingChallenge.id {
                self.pendingChallenge = nil
                lastChallengeOutcome = "\(pendingChallenge.username): canceled"
            }
        } catch LichessBotAPIError.http(let status, let message) where status == 400 || status == 404 {
            // Lichess no longer knows the challenge: it expired or was
            // answered. Nothing is pending any more.
            await runtime.manager.clearOutgoingChallenge()
            self.pendingChallenge = nil
            lastChallengeOutcome = "\(pendingChallenge.username): no longer pending (\(message ?? "HTTP \(status)"))"
        } catch {
            raiseAlarm("Cancelling the challenge to \(pendingChallenge.username) failed: \(Self.safeDescription(error))")
        }
    }

    /// Withdraw a challenge on Lichess without waiting; a failure is logged.
    private func withdraw(challengeID: String, client: LichessBotAPIClient) {
        let log = protocolLog
        Task {
            do {
                try await client.cancelChallenge(id: challengeID)
                log.record(.challenge, "withdrew challenge \(challengeID) on going offline")
            } catch {
                log.record(.anomaly, "withdrawing challenge \(challengeID) failed: \(LichessBotRedaction.redact(error.localizedDescription))")
            }
        }
    }

    // MARK: - Grid housekeeping

    func dismissFinishedGames() {
        games.removeAll { $0.isFinished && !activeGameIDs.contains($0.id) }
        if let focusedGameID, !games.contains(where: { $0.id == focusedGameID }) {
            self.focusedGameID = nil
        }
    }

    func dismissGame(_ gameID: String) {
        guard !activeGameIDs.contains(gameID) else { return }
        games.removeAll { $0.id == gameID }
    }

    func dismissAlarms() {
        alarms.removeAll()
    }

    // MARK: - Runtime construction

    private func startRuntime(oneGame: Bool) async throws {
        let settings = self.settings
        let accountID = self.accountID
        guard let token = try await readToken() else {
            throw LichessBotControllerError.noToken
        }
        let directory = dataDirectory
        let lock = try await fileQueue.run {
            try directory.createDirectories()
            return try LichessBotInstanceLock.acquire(at: directory.lockURL, holder: .current)
        }
        do {
            try await startRuntime(oneGame: oneGame, settings: settings, accountID: accountID, token: token, lock: lock)
        } catch {
            gateEventSink.value = nil
            fileQueue.enqueue { lock.release() }
            throw error
        }
    }

    private func startRuntime(oneGame: Bool, settings: LichessBotSettings, accountID: String, token: String, lock: LichessBotInstanceLock) async throws {
        let (stream, continuation) = AsyncStream<LichessBotControllerEvent>.makeStream()
        let time = LichessBotSystemTimeSource()
        let gate = self.gate
        runtimeGeneration += 1
        let generation = runtimeGeneration
        if case .closed(let reason) = await gate.snapshot().phase {
            // Going online is the operator's explicit action after a
            // breaker trip (plan §5.4).
            protocolLog.record(.lifecycle, "reopening the request gate (closed: \(reason))")
            await gate.reopen()
        }
        gateEventSink.value = continuation
        let client = LichessBotAPIClient(
            baseURL: try LichessBotAPIClient.lichessBaseURL(),
            token: token,
            transport: LichessBotURLSessionTransport(),
            gate: gate,
            onRequest: { record in continuation.yield(.request(record)) }
        )

        // Verify the token and the account before holding any stream.
        guard let info = try await client.testToken() else {
            throw LichessBotControllerError.tokenInvalid
        }
        guard info.userId == accountID else {
            throw LichessBotControllerError.tokenForWrongAccount(info.userId)
        }
        let account = try await client.account()
        guard account.isBot else {
            throw LichessBotControllerError.notABot
        }
        self.account = account
        tokenState = .saved(info)

        let settingsBox = SyncBox(settings)
        self.settingsBox = settingsBox
        let settingsProvider: @Sendable () async -> LichessBotSettings = { settingsBox.value }
        let slots = LichessBotModelSlots(provider: modelProvider, time: time) { line in
            SessionLogger.shared.log(line)
        }
        let recordStore = LichessBotRecordStore(directory: dataDirectory, fileQueue: fileQueue, ourAccountID: accountID)
        let journal = LichessBotJournalWriter(
            directory: dataDirectory,
            fileQueue: fileQueue,
            onWriteFailure: { gameID, error in continuation.yield(.journalWriteFailed(gameID: gameID, error: error.localizedDescription)) },
            onGameFinished: { gameID in continuation.yield(.gameFinished(gameID: gameID)) }
        )
        let manager = LichessBotSessionManager(
            accountAPI: client,
            gameAPI: client,
            gate: gate,
            slots: slots,
            ourAccountID: accountID,
            time: time,
            settingsProvider: settingsProvider,
            gameObserver: LichessBotGameObserverFanOut(observers: [journal, LichessBotControllerFeed(continuation: continuation)]),
            onEvent: { event in continuation.yield(.manager(event)) }
        )
        let reconciler = LichessBotReconciler(
            api: client,
            store: recordStore,
            time: time,
            settingsProvider: settingsProvider,
            isGameActive: { gameID in await manager.activeGameIDs.contains(gameID) },
            onEvent: { event in continuation.yield(.reconciler(event)) }
        )
        if oneGame {
            await manager.setOneGameMode(true)
        }
        if rateLimitHoldUntil != nil {
            await manager.setRateLimitHold(true)
        }
        await seedDailyCounts(manager: manager, store: recordStore)

        let sleepActivity = ProcessInfo.processInfo.beginActivity(
            // `.userInitiated` alone already includes idle-sleep prevention;
            // the variant that allows idle sleep is its own option.
            options: settings.connection.preventSleepWhileOnline ? [.userInitiated] : [.userInitiatedAllowingIdleSystemSleep],
            reason: "Lichess bot online"
        )
        // After a wake, connections held across sleep are usually dead:
        // reopen every game stream at once instead of waiting for each
        // watchdog (plan E34).
        let wakeObserver = NSWorkspace.shared.notificationCenter.addObserver(
            forName: NSWorkspace.didWakeNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            Task { @MainActor in
                await self?.handleWake()
            }
        }

        var tasks: [Task<Void, Never>] = []
        tasks.append(Task { [weak self] in
            for await event in stream {
                // Events still queued when this runtime was torn down are
                // dropped: they describe a runtime that no longer exists.
                guard let self, self.runtimeGeneration == generation else { break }
                self.handle(event)
            }
        })
        tasks.append(Task {
            await manager.run()
        })
        tasks.append(Task {
            await reconciler.run()
        })
        tasks.append(Task { [weak self] in
            await self?.pollLoop(slots: slots, manager: manager, generation: generation)
        })
        tasks.append(Task { [weak self] in
            // Launch recovery (plan §10.2): after the event stream has had
            // time to replay `gameStart`s, every journal left in InProgress/
            // is handed to the reconciler, which leaves live games to their
            // sessions and finalizes the rest.
            do {
                try await Task.sleep(for: .seconds(30))
            } catch {
                return
            }
            await self?.recoverLeftoverJournals(store: recordStore, reconciler: reconciler)
        })

        runtime = Runtime(
            client: client, manager: manager, slots: slots, reconciler: reconciler,
            recordStore: recordStore, journal: journal, lock: lock, continuation: continuation,
            tasks: tasks, sleepActivity: sleepActivity, wakeObserver: wakeObserver
        )
    }

    /// Seed today's game counts from the records and leftover journals, so
    /// the daily limits (including the bot-pair stop short of Lichess's cap)
    /// survive going offline, relaunching or crashing (plan §7).
    private func seedDailyCounts(manager: LichessBotSessionManager, store: LichessBotRecordStore) async {
        do {
            let file = try await store.loadIndex()
            index = file
            let today = file.rows.filter { Calendar.current.isDateInToday($0.createdAt) }
            var opponents: [String: String] = [:]
            for row in today {
                if let opponentID = row.opponentID {
                    opponents[row.gameID] = opponentID
                }
            }
            let leftovers = try await store.inProgressJournalCreationDates()
                .filter { Calendar.current.isDateInToday($0.value) }
                .map(\.key)
            await manager.seedDailyCounts(gameIDs: today.map(\.gameID) + leftovers, opponentByGame: opponents)
        } catch {
            raiseAlarm("Seeding today's game counts failed: \(Self.safeDescription(error)); daily limits count from now")
        }
    }

    private func handleWake() async {
        guard let manager = runtime?.manager else { return }
        protocolLog.record(.lifecycle, "system woke; reopening game streams")
        await manager.resyncAllSessions(reason: "the Mac woke from sleep")
    }

    /// The settings the running bot reads, updated whenever settings change.
    private var settingsBox: SyncBox<LichessBotSettings>?

    private func stopRuntime(reason: String) {
        guard runtime != nil else { return }
        protocolLog.record(.lifecycle, "offline: \(reason)")
        SessionLogger.shared.log("[LICHESS-BOT] offline: \(reason)")
        tearDownRuntime(reason: reason)
        connection = .offline
        finishing = nil
    }

    /// Stop because something went wrong; stay in Error until the operator
    /// acts (plan §13).
    private func failRuntime(_ reason: String) {
        guard runtime != nil else { return }
        raiseAlarm("Bot stopped: \(reason)")
        protocolLog.record(.lifecycle, "error: \(reason)")
        tearDownRuntime(reason: reason)
        connection = .error(LichessBotRedaction.redact(reason))
        finishing = nil
        completeQuitIfPending(quit: true)
    }

    private func tearDownRuntime(reason: String = "offline") {
        guard let runtime else { return }
        self.runtime = nil
        runtimeGeneration += 1
        settingsBox = nil
        gateEventSink.value = nil
        if let pendingChallenge {
            withdraw(challengeID: pendingChallenge.id, client: runtime.client)
        }
        pendingChallenge = nil
        oneGameRequested = false
        drainRequested = false
        acceptedAwaitingStartIDs = []
        let manager = runtime.manager
        for task in runtime.tasks {
            task.cancel()
        }
        Task {
            await manager.abandonAllSessions()
        }
        runtime.continuation.finish()
        NSWorkspace.shared.notificationCenter.removeObserver(runtime.wakeObserver)
        ProcessInfo.processInfo.endActivity(runtime.sleepActivity)
        let lock = runtime.lock
        fileQueue.enqueue { lock.release() }
        for game in games where activeGameIDs.contains(game.id) {
            game.markSessionEnded(reason)
        }
        activeGameIDs.removeAll()
        gateSnapshot = nil
        generation = nil
    }

    private func pollLoop(slots: LichessBotModelSlots, manager: LichessBotSessionManager, generation runtimeGenerationAtStart: Int) async {
        var lastRefresh = Date.distantPast
        // Every await can outlive this runtime; nothing is written after it.
        func current() -> Bool { runtimeGeneration == runtimeGenerationAtStart }
        while !Task.isCancelled {
            let snapshot = await gate.snapshot()
            guard current() else { return }
            gateSnapshot = snapshot
            let info = await slots.current?.info
            guard current() else { return }
            generation = info
            let accepted = await manager.acceptedAwaitingStartIDs()
            guard current() else { return }
            if accepted != acceptedAwaitingStartIDs {
                acceptedAwaitingStartIDs = accepted
                finishIfDrained()
            }
            if let until = rateLimitHoldUntil, Date() >= until {
                await endRateLimitHold()
                guard current() else { return }
            }
            if Date().timeIntervalSince(lastRefresh) >= 15 {
                lastRefresh = Date()
                do {
                    try await slots.refreshIfDue(for: settings.model)
                } catch {
                    guard current() else { return }
                    raiseAlarm("Model refresh failed: \(Self.safeDescription(error))")
                }
                guard current() else { return }
            }
            pruneFinishedGames()
            do {
                try await Task.sleep(for: .seconds(1))
            } catch {
                return
            }
        }
    }

    private func recoverLeftoverJournals(store: LichessBotRecordStore, reconciler: LichessBotReconciler) async {
        do {
            let leftovers = try await store.inProgressGameIDs().filter { !activeGameIDs.contains($0) }
            for gameID in leftovers {
                await reconciler.enqueue(gameID: gameID)
            }
            if !leftovers.isEmpty {
                protocolLog.record(.game, "launch recovery: \(leftovers.count) leftover journal(s) queued for reconciliation", fields: ["games": leftovers.joined(separator: ",")])
            }
        } catch {
            raiseAlarm("Launch recovery could not list leftover journals: \(error.localizedDescription)")
        }
    }

    /// Load the finished-games index (for head-to-head records). Works
    /// offline too: the records are on disk.
    func refreshIndex() async {
        let store = runtime?.recordStore
            ?? LichessBotRecordStore(directory: dataDirectory, fileQueue: fileQueue, ourAccountID: accountID)
        do {
            index = try await store.loadIndex()
        } catch {
            raiseAlarm("Loading the games index failed: \(error.localizedDescription)")
        }
    }

    private func pruneFinishedGames() {
        let retention = TimeInterval(settings.display.finishedGameRetentionMinutes * 60)
        let now = Date()
        games.removeAll { game in
            guard let finishedAt = game.finishedAt, !activeGameIDs.contains(game.id) else { return false }
            return now.timeIntervalSince(finishedAt) >= retention && game.id != focusedGameID
        }
    }

    // MARK: - Event handling (main actor)

    private func handle(_ event: LichessBotControllerEvent) {
        switch event {
        case .game(let gameID, let gameEvent):
            liveGame(gameID).apply(gameEvent)
            switch gameEvent {
            case .stoppedMoving(let reason):
                // Plan §6.1 B: never race another client; go offline.
                failRuntime("game \(gameID): \(reason)")
            case .tokenRejected(let detail):
                failRuntime("Lichess rejected the token: \(detail)")
            default:
                break
            }
        case .request(let record):
            protocolLog.record(.request, Self.describe(record), gameID: record.gameID)
            if let gameID = record.gameID {
                liveGame(gameID).applyRequest(record)
                if let journal = runtime?.journal {
                    Task {
                        await journal.recordRequest(record)
                    }
                }
            }
        case .manager(let managerEvent):
            handle(managerEvent)
        case .gate(let gateEvent):
            handle(gateEvent)
        case .reconciler(let reconcilerEvent):
            handle(reconcilerEvent)
        case .journalWriteFailed(let gameID, let error):
            raiseAlarm("Journal write failed for game \(gameID): \(error)")
        case .gameFinished(let gameID):
            if let reconciler = runtime?.reconciler {
                // A short delay lets any last request (a goodbye chat) land
                // in the journal before it is filed.
                Task {
                    await reconciler.enqueue(gameID: gameID, after: .seconds(5))
                }
            }
        }
    }

    private func handle(_ event: LichessBotManagerEvent) {
        switch event {
        case .eventStreamOpened(let attempt):
            protocolLog.record(.stream, "event stream opened", fields: ["attempt": "\(attempt)"])
        case .eventStreamEnded(let reason):
            protocolLog.record(.stream, "event stream ended: \(reason)")
        case .takeoverSuspected(let count):
            // The manager stops itself next; its `.stopped` puts the bot in
            // Error.
            raiseAlarm("Another client appears to be using this token (\(count) immediate closes)")
        case .tokenRejected(let detail):
            failRuntime("Lichess rejected the token: \(detail)")
        case .challengeDecision(let challengeID, let challengerID, let decision):
            protocolLog.record(.challenge, "\(challengerID): \(Self.describe(decision))", fields: ["challenge": challengeID])
        case .challengeResponseFailed(let challengeID, let error):
            protocolLog.record(.anomaly, "challenge response failed: \(error)", fields: ["challenge": challengeID])
        case .gameSessionStarted(let gameID, let generation):
            if let pending = pendingChallenge, pending.id == gameID {
                // The accepted challenge's game can start before the
                // manager was told about the challenge.
                lastChallengeOutcome = "\(pending.username): accepted"
                pendingChallenge = nil
                focusedGameID = gameID
                if let manager = runtime?.manager {
                    Task { await manager.clearOutgoingChallenge() }
                }
            }
            activeGameIDs.insert(gameID)
            _ = liveGame(gameID)
            self.generation = generation
            protocolLog.record(.game, "game started", gameID: gameID, fields: ["model": generation.modelID, "generation": "\(generation.generationID)"])
            SessionLogger.shared.log("[LICHESS-BOT] game \(gameID) started with \(generation.sourceKind.rawValue) \(generation.modelID)")
        case .gameSessionEnded(let gameID):
            activeGameIDs.remove(gameID)
            if let game = games.first(where: { $0.id == gameID }) {
                game.markSessionEnded("the game session ended")
            }
            protocolLog.record(.game, "game session ended", gameID: gameID)
            finishIfDrained()
        case .anomaly(let text):
            protocolLog.record(.anomaly, text)
        case .eventStreamLine(let data, let receivedAt):
            protocolLog.record(.stream, String(decoding: data, as: UTF8.self), fields: ["stream": "event"], at: receivedAt)
        case .eventStreamGaps(let summary):
            protocolLog.record(.stream, "event-stream keep-alive gaps", fields: [
                "window_s": String(format: "%.1f", summary.windowSeconds),
                "count": "\(summary.gapCount)",
                "mean_s": String(format: "%.2f", summary.meanSeconds),
                "max_s": String(format: "%.2f", summary.maximumSeconds),
            ])
        case .eventStreamLongGap(let seconds):
            protocolLog.record(.stream, String(format: "event-stream silence of %.1f s", seconds))
        case .outgoingChallengeResolved(let challengeID, let outcome):
            let text: String
            switch outcome {
            case .accepted(let gameID):
                text = "accepted; game \(gameID)"
                focusedGameID = gameID
            case .declined(let reason):
                text = "declined" + (reason.map { ": \($0)" } ?? "")
            case .canceled:
                text = "canceled"
            }
            if let pendingChallenge {
                lastChallengeOutcome = "\(pendingChallenge.username): \(text)"
            } else {
                lastChallengeOutcome = "challenge \(challengeID): \(text)"
            }
            pendingChallenge = nil
            protocolLog.record(.challenge, "outgoing challenge \(text)", fields: ["challenge": challengeID])
        case .oneGameStarted(let gameID):
            drainRequested = true
            connection = .draining
            protocolLog.record(.lifecycle, "one game started (\(gameID)); draining")
        case .stopped(let reason):
            // The manager stopped on its own: the gate closed, the token was
            // rejected, or a takeover.
            failRuntime(reason)
        }
    }

    /// Gate events are already in the protocol log (the gate's own hook
    /// writes them); this reacts to the ones that change the bot's state.
    private func handle(_ event: LichessBotGateEvent) {
        switch event {
        case .rateLimited(let cooldown, let label, _, _):
            let minutes = settings.connection.postRateLimitDrainMinutes
            raiseAlarm("Rate limited by Lichess on \(label); cooling down \(cooldown) and taking no new games for \(minutes) min")
            Task {
                await startRateLimitHold(minutes: minutes)
            }
        case .breakerTripped:
            failRuntime("a second 429 within the breaker window closed the request gate")
        default:
            break
        }
    }

    /// After a 429: decline new games for a while; games in progress go on
    /// (plan §5.4). Accepting resumes when the hold ends, unless the
    /// operator drained meanwhile.
    private func startRateLimitHold(minutes: Int) async {
        rateLimitHoldUntil = Date().addingTimeInterval(TimeInterval(minutes * 60))
        guard let manager = runtime?.manager else { return }
        await manager.setRateLimitHold(true)
        guard runtime?.manager === manager else { return }
        if connection == .online {
            connection = .draining
        }
    }

    /// The hold is over. The manager accepts again only if nothing else
    /// (a drain, "one game") stopped it; the state follows the manager.
    private func endRateLimitHold() async {
        rateLimitHoldUntil = nil
        guard let manager = runtime?.manager else { return }
        await manager.setRateLimitHold(false)
        let accepting = await manager.isAcceptingNewGames
        guard runtime?.manager === manager, connection == .draining, accepting, !drainRequested else { return }
        connection = .online
        protocolLog.record(.lifecycle, "rate-limit hold over; accepting games again")
    }

    private func handle(_ event: LichessBotReconcilerEvent) {
        switch event {
        case .finalized(let finalized):
            runtime?.journal.markFinalized(gameID: finalized.record.gameID)
            unreconciledGameIDs.removeAll { $0 == finalized.record.gameID }
            SessionLogger.shared.log(LichessBotRecordStore.summaryLine(finalized.record))
            protocolLog.record(.game, "finalized (\(finalized.record.reconciliation.outcome.rawValue))", gameID: finalized.record.gameID, fields: ["mismatches": finalized.record.reconciliation.mismatches.joined(separator: "; ")])
            Task {
                await refreshIndex()
            }
        case .waiting(let gameID, let reason, let retryIn):
            protocolLog.record(.game, "reconciliation waiting: \(reason); retry in \(retryIn)", gameID: gameID)
        case .unreconciled(let gameID, let reason):
            if !unreconciledGameIDs.contains(gameID) {
                unreconciledGameIDs.append(gameID)
            }
            raiseAlarm("Game \(gameID) is unreconciled: \(reason)")
        case .finalizeFailed(let gameID, let error):
            raiseAlarm("Filing game \(gameID) failed: \(error)")
        case .stopped(let reason):
            protocolLog.record(.lifecycle, "reconciler stopped: \(reason)")
        }
    }

    /// Once draining leaves no games: when quitting, stop at once (launch
    /// recovery files anything unfinished); otherwise first give the
    /// reconciler a bounded time to file the finished games' records, then
    /// go offline.
    private func finishIfDrained() {
        guard connection == .draining, drainRequested, !hasGamesInPlay else { return }
        if finishing == .quit {
            stopRuntime(reason: "drained for quit")
            completeQuitIfPending(quit: true)
            return
        }
        guard !isFilingRecords else { return }
        isFilingRecords = true
        Task {
            await fileRecordsThenStop()
        }
    }

    private func fileRecordsThenStop() async {
        defer { isFilingRecords = false }
        let started = Date()
        while let reconciler = runtime?.reconciler, Date().timeIntervalSince(started) < Self.filingTimeLimitSeconds {
            // The finished game's enqueue happens on its own task: allow a
            // moment for it to land before trusting an empty queue.
            let due = await reconciler.dueGameIDs
            if due.isEmpty && Date().timeIntervalSince(started) >= 3 {
                break
            }
            do {
                try await Task.sleep(for: .seconds(1))
            } catch {
                return
            }
        }
        guard runtime != nil, connection == .draining, !hasGamesInPlay else { return }
        if let reconciler = runtime?.reconciler {
            let unfiled = await reconciler.queuedGameIDs
            if !unfiled.isEmpty {
                protocolLog.record(.game, "going offline with \(unfiled.count) game(s) not yet filed; launch recovery will file them", fields: ["games": unfiled.joined(separator: ",")])
            }
        }
        stopRuntime(reason: oneGameRequested ? "one game finished" : "drained")
        // A quit asked for while records were being filed is answered now.
        completeQuitIfPending(quit: true)
    }

    /// Longest a drain waits for records to be filed before going offline.
    private static let filingTimeLimitSeconds: TimeInterval = 120

    private func liveGame(_ gameID: String) -> LichessBotLiveGame {
        if let existing = games.first(where: { $0.id == gameID }) {
            return existing
        }
        let game = LichessBotLiveGame(id: gameID, startedAt: Date(), ourAccountID: accountID)
        games.append(game)
        return game
    }

    private func raiseAlarm(_ rawText: String) {
        let text = LichessBotRedaction.redact(rawText)
        alarms.append(Alarm(id: nextAlarmID, at: Date(), text: text))
        nextAlarmID += 1
        protocolLog.record(.anomaly, text)
        SessionLogger.shared.log("[ALARM] LICHESS-BOT \(text)")
    }

    // MARK: - Descriptions for the protocol log

    nonisolated static func describe(_ record: LichessBotRequestRecord) -> String {
        var text = "\(record.method) \(record.path) (\(record.label))"
        if let status = record.status {
            text += " \(status)"
        }
        if let roundTrip = record.roundTripMilliseconds {
            text += String(format: " %.0f ms", roundTrip)
        }
        text += String(format: " queued %.0f ms", record.queuedMilliseconds)
        if let networkProtocol = record.networkProtocol {
            text += " \(networkProtocol)"
        }
        if !record.formFields.isEmpty {
            text += " " + record.formFields.sorted { $0.key < $1.key }.map { "\($0.key)=\($0.value)" }.joined(separator: "&")
        }
        if let message = record.errorMessage {
            text += " error: \(message)"
        }
        if let failure = record.failure {
            text += " failed: \(failure)"
        }
        return text
    }

    nonisolated static func describe(_ event: LichessBotGateEvent) -> String {
        switch event {
        case .requestStarted(let id, let priority, let label):
            return "gate: start #\(id) \(priority.label) \(label)"
        case .requestFinished(let id, let priority, let label, let status, let latency, let retryAfter):
            return "gate: done #\(id) \(priority.label) \(label) \(status) \(latency)" + (retryAfter.map { " Retry-After=\($0)" } ?? "")
        case .requestFailed(let id, let priority, let label, let error, let latency):
            return "gate: failed #\(id) \(priority.label) \(label) after \(latency): \(error)"
        case .rateLimited(let cooldown, let label, let priority, let counts):
            let breakdown = counts.sorted { $0.key.label < $1.key.label }.map { "\($0.key.label)=\($0.value)" }.joined(separator: ",")
            return "gate: 429 on \(priority.label) \(label); cooldown \(cooldown); last minute \(breakdown)"
        case .cooldownEnded:
            return "gate: cooldown ended"
        case .breakerTripped(let count, let window):
            return "gate: breaker tripped (\(count) 429s within \(window))"
        case .closed(let reason):
            return "gate: closed (\(reason))"
        case .reopened:
            return "gate: reopened"
        }
    }

    nonisolated static func kind(of event: LichessBotGateEvent) -> LichessBotProtocolEventKind {
        switch event {
        case .rateLimited, .cooldownEnded: return .rateLimit
        case .breakerTripped: return .breaker
        case .closed, .reopened: return .lifecycle
        case .requestStarted, .requestFinished, .requestFailed: return .request
        }
    }

    nonisolated static func describe(_ decision: LichessBotChallengeDecision) -> String {
        switch decision {
        case .accept:
            return "accept"
        case .decline(let reason, let rule):
            return "decline (\(reason.rawValue)): \(rule)"
        case .ignore(let rule):
            return "ignore: \(rule)"
        }
    }
}

enum LichessBotControllerError: LocalizedError, Equatable {
    case noToken
    case tokenInvalid
    case tokenForWrongAccount(String)
    case notABot
    case notOnline
    case challengeAlreadyPending
    case missingChallengeScope
    case dailyLimitReached(opponent: String, games: Int)

    var errorDescription: String? {
        switch self {
        case .noToken:
            return "No Lichess token is saved. Add one in Settings ▸ Account."
        case .tokenInvalid:
            return "Lichess reports the saved token as invalid or revoked. Replace it in Settings ▸ Account."
        case .tokenForWrongAccount(let userID):
            return "The saved token belongs to \(userID), not the configured account"
        case .notABot:
            return "The account is not a BOT account yet. Upgrade it in Settings ▸ Account."
        case .notOnline:
            return "The bot must be online to send a challenge"
        case .challengeAlreadyPending:
            return "A challenge is already waiting for an answer"
        case .missingChallengeScope:
            return "The token lacks the challenge:write scope needed to send challenges"
        case .dailyLimitReached(let opponent, let games):
            return "Already \(games) games against \(opponent) today (the bot-pair daily stop)"
        }
    }
}
