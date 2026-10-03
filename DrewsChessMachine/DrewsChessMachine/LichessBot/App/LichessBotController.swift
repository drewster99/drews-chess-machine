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

/// Where the controller gets its connection to Lichess and the stored
/// token. The app uses `live`; tests substitute a scripted transport and a
/// fixed token so a whole runtime runs without the network or the Keychain.
struct LichessBotControllerServices: Sendable {
    let makeTransport: @Sendable () -> any LichessBotTransport
    /// The stored token for a Lichess account id, or nil if none is stored.
    /// Called on the file queue (Keychain calls block).
    let readToken: @Sendable (_ accountID: String) throws -> String?

    static let live = LichessBotControllerServices(
        makeTransport: { LichessBotURLSessionTransport() },
        readToken: { accountID in try LichessBotTokenStore().read(account: accountID) }
    )
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

    /// Who sent an outgoing challenge. A decline is answered differently
    /// for each: only a challenge matchmaking picked may be resent as casual
    /// automatically (`LichessBotMatchmakingSettings.fallBackToCasual`); the
    /// operator's own challenges only ever get the manual offer, and the
    /// automatic resend itself is never resent, so a decline can't start a
    /// loop.
    enum ChallengeOrigin: Equatable {
        /// The operator: the Challenge sheet, the challenge queue, or the
        /// manual Resend as Casual.
        case manual
        /// A matchmaking pass, filling slots the way `fillMode` does, picked
        /// `opponent` from the online-bots list. Both are kept so a casual
        /// resend is checked against the same slot rule and candidate rules
        /// as the pass that picked them.
        case matchmaking(fillMode: LichessBotMatchmakingSettings.FillMode, opponent: LichessBotUserSummary)
        /// Matchmaking's one automatic casual resend of a declined rated
        /// challenge.
        case matchmakingCasualResend
    }

    struct PendingChallenge: Equatable {
        let id: String
        let username: String
        let sentAt: Date
        /// What was asked for, so a decline can offer the same challenge
        /// adjusted.
        let request: LichessBotOutgoingChallenge
        let origin: ChallengeOrigin
    }

    /// A rated challenge the player declined with Lichess's `casual` reason
    /// ("please send me a casual challenge instead"): the same challenge,
    /// unrated, ready to send.
    struct CasualResendOffer: Equatable {
        let username: String
        let request: LichessBotOutgoingChallenge
    }

    /// An opponent's public profile, fetched once per app session (plan
    /// §14.3b).
    enum OpponentProfile: Equatable {
        case loading
        /// `crosstableError` is set when only the head-to-head failed.
        case loaded(LichessBotUserSummary, crosstable: LichessBotCrosstable?, crosstableError: String?)
        case failed(String)
    }

    /// One alarm text. Raising the same text again updates `lastAt` and
    /// `repeatCount` instead of adding a row, so a condition that keeps
    /// failing can't bury the others.
    struct Alarm: Identifiable, Equatable {
        let id: Int
        let firstAt: Date
        var lastAt: Date
        let text: String
        var repeatCount: Int
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
    /// Outgoing challenges still unanswered, oldest first (plan §7.1: games
    /// from them run in parallel, within the concurrent-game limit).
    private(set) var pendingChallenges: [PendingChallenge] = []
    /// Challenge POSTs under way, counted against the concurrent-game limit.
    /// (The per-opponent limit counts them in the manager, as reservations.)
    private var challengeSendsInFlight = 0
    /// Games started today per opponent id, mirrored from the manager by the
    /// poll loop (the Challenge sheet warns past the daily per-opponent limit).
    private(set) var gamesTodayByOpponent: [String: Int] = [:]
    private(set) var lastChallengeOutcome: String?
    /// Set when the latest outgoing challenge was declined as "send me a
    /// casual challenge instead"; cleared by the next challenge sent.
    private(set) var casualResendOffer: CasualResendOffer?
    private(set) var oneGameRequested = false
    private(set) var alarms: [Alarm] = []
    /// Alarms dropped from the front of the list to keep it bounded; the
    /// session log keeps every one.
    private(set) var droppedAlarmCount = 0
    private static let alarmLimit = 200
    private(set) var unreconciledGameIDs: [String] = []
    private(set) var finishing: FinishingPurpose?
    private(set) var index: LichessBotIndex.File? {
        didSet {
            recordsByOpponent = LichessBotRecordSummary.byOpponent(rows: index?.rows ?? [])
            pastOpponents = LichessBotRecordSummary.pastOpponents(rows: index?.rows ?? [])
            // Reported whenever the set of undecodable records changes (and so
            // at least once per launch): they are left out of every count.
            if let index, !index.unreadableRecords.isEmpty, index.unreadableRecords != oldValue?.unreadableRecords {
                raiseAlarm("\(index.unreadableRecords.count) game record(s) don't decode and are left out of the index: \(index.unreadableRecords.map(\.path).joined(separator: ", "))")
            }
        }
    }
    /// Everyone DCM has played, most recent first (the Challenge sheet's
    /// History tab), rebuilt with the games index.
    private(set) var pastOpponents: [LichessBotPastOpponent] = []
    /// DCM's results against each opponent (lowercased id), rebuilt when the
    /// games index loads, so tables can show them without scanning records.
    private(set) var recordsByOpponent: [String: LichessBotResultTally] = [:]
    private(set) var onlineBots: [LichessBotUserSummary] = []
    /// When `onlineBots` was last fetched successfully.
    private(set) var onlineBotsFetchedAt: Date?
    /// Favorites and bot limit times (plan §7.2); nil until loaded.
    private(set) var playerNotes: LichessBotPlayerNotes?
    /// Every outgoing challenge attempt of the rolling day and how it
    /// ended: the single source of the challenge-credit and outcome counts
    /// on the Overview. Nil until loaded; persisted to
    /// `challenge-outcomes.json`.
    private(set) var challengeOutcomeLog: LichessBotChallengeOutcomeLog?
    /// Online and playing flags by lowercased user id, from
    /// `/api/users/status`: the single store every Challenge-sheet tab reads.
    /// An id Lichess didn't answer for (unknown or closed) has no entry.
    private(set) var playerStatuses: [String: LichessBotUserStatus] = [:]
    /// When each id's status was last asked for, answered or not.
    private var playerStatusFetchedAt: [String: Date] = [:]
    /// How old a status may get before a visible list asks again.
    static let playerStatusMaximumAge: TimeInterval = 120
    /// Opponent profiles by lowercased user id (plan §14.3b). Not
    /// persisted: each app session fetches fresh.
    private(set) var opponentProfiles: [String: OpponentProfile] = [:]
    /// A drain has finished its games and is filing their records before
    /// going offline.
    private(set) var isFilingRecords = false
    /// Outgoing challenges an automatic withdrawal was attempted for. One
    /// attempt per challenge: a failure raises an alarm once and leaves the
    /// Cancel button to the operator, rather than retrying (and alarming)
    /// on every poll.
    private var autoWithdrawAttemptedChallengeIDs: Set<String> = []
    /// The operator's challenges still to send (plan §7.3 A): the single
    /// source the Overview lists and the queue pump sends from. In memory
    /// only; going offline or quitting clears it.
    private(set) var challengeQueue = LichessBotChallengeQueue()
    /// A queue pump is running; at most one does, so entries go out one at
    /// a time.
    private var challengeQueuePumpRunning = false
    /// Matchmaking's send rate (plan §7.3 B). Kept across runtimes: Lichess
    /// counts challenges per account, not per connection.
    private(set) var matchmakingRateLimiter = LichessBotMatchmakingRateLimiter()
    /// A matchmaking pass, or matchmaking's automatic casual resend, is
    /// deciding or sending.
    private var matchmakingPassRunning = false
    /// Lowercased ids of bots whose automatic casual resend is waiting for a
    /// running matchmaking pass to finish. Counted as engaged by the
    /// candidate rules, so that pass can't pick the bot meanwhile (it has no
    /// decline cool-down yet, and no challenge pending).
    private var casualResendWaitingOpponentIDs: Set<String> = []
    /// The Overview's Fill Open Slots is filling slots.
    private(set) var isFillingOpenSlots = false
    /// What the latest matchmaking pass did, for the Overview.
    private(set) var matchmakingStatus: String?
    /// When the next automatic pass may run: after a send, the spacing;
    /// after a pass that found nobody, a longer wait.
    private var nextAutomaticMatchmakingPassAt = Date.distantPast
    /// The last pass outcome written to the protocol log by an automatic
    /// pass, so an unchanged "nobody fits" isn't logged every retry.
    private var lastLoggedAutomaticMatchmakingOutcome: String?
    /// When the online-bots list was last asked for, answered or not, so a
    /// failing fetch isn't retried every poll.
    private var onlineBotsRequestedAt: Date?
    /// The online-bots fetch under way, if any — the one record that a fetch
    /// is in flight. Everything that wants the list (the Challenge sheet,
    /// the poll loop's matchmaking refresh, a matchmaking pass) joins this
    /// fetch rather than starting a second one, and a matchmaking pass that
    /// finds it under way waits for it. A pass that skipped a fetch in
    /// flight would decide without a list: the first pass after going Online
    /// starts on the same poll as the first fetch, so it would always find
    /// none. Not tied to a runtime: the list outlives going offline, and the
    /// sheet fetches while offline too.
    private var onlineBotsRefresh: Task<Void, Never>?
    /// The Live tab's single-game choice; nil means "First in progress".
    /// Remembered across launches (a specific game shows again only while it
    /// is listed).
    var focusedGameID: String? {
        didSet {
            defaults.set(focusedGameID, forKey: Self.focusedGameIDKey)
            if focusedGameID != nil {
                // The operator's pick outranks following; a hold would only
                // move a view nobody is following.
                cancelAutoFollowHold()
            } else if oldValue != nil {
                // Back to "First in progress": follow the first live game
                // now, not whatever was followed before the pick.
                cancelAutoFollowHold()
                autoFollowedGameID = firstLiveGame?.id
            }
        }
    }
    /// The game the single view follows while the operator hasn't picked
    /// one: the first-started game in progress, kept on screen for
    /// `finishedGameHoldDuration` after it ends before moving to the next
    /// live game, so the view doesn't jump the instant a game finishes.
    private(set) var autoFollowedGameID: String?
    /// The hold on a finished followed game: which game, and a token that
    /// identifies this hold, so a stale hold's timer never moves the view.
    private var autoFollowHold: (gameID: String, token: UUID, task: Task<Void, Never>)?
    /// How long the single view stays on a followed game after it ends.
    static let finishedGameHold: Duration = .seconds(8)
    private let finishedGameHoldDuration: Duration
    /// The clock the live grid's order and tile phases are computed at.
    /// It advances only when a finished game's position hold or result
    /// highlight ends, so each change is one observable step the grid
    /// animates, rather than a continuously ticking value.
    private(set) var gridClock = Date()
    /// The pending wake-up that next advances `gridClock`.
    private var gridClockTask: Task<Void, Never>?
    /// The Live tab's Single / Grid choice, remembered across launches.
    var showsGrid = false {
        didSet { defaults.set(showsGrid, forKey: Self.showsGridKey) }
    }
    /// The Settings tab the operator last picked, remembered across
    /// launches as a convenience. The Settings view records only the
    /// operator's own picks here; the tab it selects by itself (Account,
    /// when the token check finds no token) is not remembered.
    var rememberedSettingsTab: LichessBotSettingsTab = .games {
        didSet { defaults.set(rememberedSettingsTab.rawValue, forKey: Self.rememberedSettingsTabKey) }
    }
    private static let focusedGameIDKey = "lichessBot.live.focusedGameID"
    private static let showsGridKey = "lichessBot.live.showsGrid"
    private static let rememberedSettingsTabKey = "lichessBot.settings.tab"

    // MARK: - Configuration

    let dataDirectory: LichessBotDataDirectory
    private let defaults: UserDefaults
    private let tokenStore = LichessBotTokenStore()
    /// Everything on disk except game journals and filing: index, protocol
    /// log, player notes, Keychain, the instance lock.
    private let fileQueue = LichessBotFileQueue()
    /// Game journals and filing. Every game-stream line waits for its journal
    /// append, so this queue carries nothing else. One for the controller's
    /// whole life, shared by every runtime: a torn-down journal writer and a
    /// new one may both still be appending to the same live game's journal.
    private let journalQueue = LichessBotFileQueue(label: "drewschess.lichessbot.journals", qos: .userInitiated)
    private let modelProvider: any LichessBotModelProvider
    private let services: LichessBotControllerServices
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
    /// Accepted challenges whose games haven't started, or whose game
    /// session is still being set up (from the manager). They hold slots,
    /// and a drain or quit waits for them like games in progress.
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

    init(
        modelProvider: any LichessBotModelProvider,
        defaults: UserDefaults = .standard,
        dataDirectory: LichessBotDataDirectory = .standard,
        services: LichessBotControllerServices = .live,
        finishedGameHold: Duration = LichessBotController.finishedGameHold,
        postGameChatFetchDelays: [Duration] = LichessBotController.postGameChatFetchDelays
    ) {
        self.modelProvider = modelProvider
        self.defaults = defaults
        self.dataDirectory = dataDirectory
        self.services = services
        self.finishedGameHoldDuration = finishedGameHold
        self.postGameChatFetchSchedule = postGameChatFetchDelays
        let fileQueue = self.fileQueue
        self.protocolLog = LichessBotProtocolLog(directory: dataDirectory, fileQueue: fileQueue) { error in
            SessionLogger.shared.log("[ALARM] LICHESS-BOT protocol log write failed: \(error.localizedDescription)")
        }
        let loadedSettings: LichessBotSettings
        do {
            let loaded = try LichessBotSettingsStore.loadReporting(from: defaults)
            loadedSettings = loaded.settings
            if !loaded.filledFromDefaults.isEmpty {
                SessionLogger.shared.log("[LICHESS-BOT] settings: saved settings predate \(loaded.filledFromDefaults.count) field(s); using today's defaults for: \(loaded.filledFromDefaults.joined(separator: ", "))")
            }
            if !loaded.ignoredSavedKeys.isEmpty {
                SessionLogger.shared.log("[LICHESS-BOT] settings: ignoring saved field(s) that no longer exist: \(loaded.ignoredSavedKeys.joined(separator: ", "))")
            }
        } catch {
            loadedSettings = LichessBotSettings()
            settingsError = error.localizedDescription
        }
        settings = loadedSettings
        showsGrid = defaults.bool(forKey: Self.showsGridKey)
        focusedGameID = defaults.string(forKey: Self.focusedGameIDKey)
        if let savedTab = defaults.string(forKey: Self.rememberedSettingsTabKey) {
            if let tab = LichessBotSettingsTab(rawValue: savedTab) {
                rememberedSettingsTab = tab
            } else {
                SessionLogger.shared.log("[LICHESS-BOT] settings: ignoring the remembered Settings tab \"\(savedTab)\", which no longer exists; opening on the default tab")
            }
        }
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

    /// Our account id, for views that key data by it (a crosstable).
    var botAccountID: String {
        accountID
    }

    /// An error's text, safe to show and log (plan §10.3).
    private nonisolated static func safeDescription(_ error: Error) -> String {
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
    /// exists, otherwise the followed game (the first-started game in
    /// progress, held on screen for a while after it ends), otherwise the
    /// first game in progress, otherwise the most recent game (plan §14.3a).
    var displayedGame: LichessBotLiveGame? {
        if let focusedGameID, let game = listedGame(focusedGameID) {
            return game
        }
        if let autoFollowedGameID, let game = listedGame(autoFollowedGameID) {
            return game
        }
        return gamesInProgress.first ?? games.last
    }

    /// The first-started game still being played. A game whose final state
    /// has arrived is finished even while its session winds down.
    private var firstLiveGame: LichessBotLiveGame? {
        gamesInProgress.first { !$0.isFinished }
    }

    /// Keep the followed game current. With none (or one no longer listed),
    /// follow the first live game. When the followed game has finished,
    /// start one hold; when it ends, the view moves on to the next live
    /// game. Called after every change to the games list or a game's state.
    private func updateAutoFollow() {
        refreshGridClock()
        guard let followedID = autoFollowedGameID, let followed = listedGame(followedID) else {
            cancelAutoFollowHold()
            let next = firstLiveGame?.id
            if autoFollowedGameID != next {
                autoFollowedGameID = next
            }
            return
        }
        guard followed.isFinished, autoFollowHold?.gameID != followedID else { return }
        cancelAutoFollowHold()
        if let focusedGameID, listedGame(focusedGameID) != nil {
            // The operator's pick is on screen: nothing to hold.
            autoFollowedGameID = firstLiveGame?.id
            return
        }
        let token = UUID()
        let hold = finishedGameHoldDuration
        let task = Task { [weak self] in
            do {
                try await Task.sleep(for: hold)
            } catch {
                return
            }
            self?.endAutoFollowHold(token: token)
        }
        autoFollowHold = (followedID, token, task)
    }

    /// The hold with this token is over: follow the next live game, unless
    /// the hold was replaced or canceled meanwhile.
    private func endAutoFollowHold(token: UUID) {
        guard let hold = autoFollowHold, hold.token == token else { return }
        autoFollowHold = nil
        guard autoFollowedGameID == hold.gameID else { return }
        autoFollowedGameID = firstLiveGame?.id
        // The finished game may have been kept only for the hold.
        pruneFinishedGames()
    }

    private func cancelAutoFollowHold() {
        autoFollowHold?.task.cancel()
        autoFollowHold = nil
    }

    /// The live grid's order, from `LichessBotGridOrdering` at `gridClock`.
    /// Used by the grid only; the single view's follow logic is separate.
    var gridOrderedGames: [LichessBotLiveGame] {
        let entries = games.map { LichessBotGridOrdering.Entry(id: $0.id, startedAt: $0.startedAt, finishedAt: $0.finishedAt) }
        return LichessBotGridOrdering.orderedIndices(entries, now: gridClock).map { games[$0] }
    }

    /// Advance `gridClock` if a position hold or result highlight has ended
    /// since it was last set, and schedule the next advance for the moment
    /// the next one ends. Rescheduling on every games change keeps exactly
    /// one wake-up pending, and none when no finished game is within its
    /// windows, so nothing polls.
    private func refreshGridClock() {
        gridClockTask?.cancel()
        gridClockTask = nil
        let now = Date()
        let finishTimes = games.compactMap(\.finishedAt)
        if LichessBotGridOrdering.hasTransition(finishTimes: finishTimes, since: gridClock, through: now) {
            gridClock = now
        }
        guard let next = LichessBotGridOrdering.nextTransition(finishTimes: finishTimes, after: now) else { return }
        let delay = Duration.milliseconds(Int64((next.timeIntervalSince(now) * 1000).rounded(.up)))
        gridClockTask = Task { [weak self] in
            do {
                try await Task.sleep(for: delay)
            } catch {
                // Canceled: a newer schedule replaced this one.
                return
            }
            self?.refreshGridClock()
        }
    }

    var hasChallengeScope: Bool {
        if case .saved(let info) = tokenState {
            return info.scopeList.contains("challenge:write")
        }
        return false
    }

    /// Our record against a player with a Lichess account, from the
    /// finished-games index; a player DCM hasn't played has an empty record.
    func headToHead(against opponentID: String) -> (wins: Int, draws: Int, losses: Int) {
        guard let tally = recordsByOpponent[opponentID.lowercased()] else { return (0, 0, 0) }
        return (tally.wins, tally.draws, tally.losses)
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
        let accountChanged = newSettings.connection.expectedAccountID.lowercased() != accountID
        settings = newSettings
        settingsError = nil
        settingsBox?.value = newSettings
        let gate = self.gate
        let window = Duration.seconds(newSettings.connection.rateLimitBreakerWindowMinutes * 60)
        Task {
            await gate.setBreakerWindow(window)
        }
        if accountChanged {
            // The token status and account shown belong to the previous id.
            tokenState = .unknown
            account = nil
            Task { await refreshTokenState() }
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
        let read = services.readToken
        let accountID = self.accountID
        return try await fileQueue.run {
            try read(accountID)
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
            transport: services.makeTransport(),
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
        guard !isShutDown else {
            connection = .error("The bot has shut down")
            SessionLogger.shared.log("[ALARM] LICHESS-BOT going online refused: the bot has shut down")
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
            // A shutdown while the runtime started found no runtime to stop.
            if isShutDown {
                tearDownRuntime(reason: "the bot shut down while going online")
                oneGameRequested = false
                connection = .error("The bot has shut down")
                SessionLogger.shared.log("[ALARM] LICHESS-BOT shut down while going online; the new runtime was stopped")
                return
            }
            // Something may have stopped the new runtime while it started.
            guard connection == .connecting, runtime != nil else { return }
            // A post-429 hold outlives going offline and back (plan §5.4).
            connection = rateLimitHoldUntil == nil ? .online : .draining
            protocolLog.record(.lifecycle, oneGame ? "online (one game)" : "online")
            SessionLogger.shared.log("[LICHESS-BOT] online\(oneGame ? " (one game)" : "") as \(accountID)")
            for failure in HardwareInfo.current.readFailures {
                SessionLogger.shared.log("[LICHESS-BOT] hardware fact unavailable for chat replies: \(failure)")
            }
        } catch {
            tearDownRuntime()
            oneGameRequested = false
            let text = Self.safeDescription(error)
            connection = .error(text)
            protocolLog.record(.lifecycle, "going online failed: \(text)")
            SessionLogger.shared.log("[ALARM] LICHESS-BOT going online failed: \(text)")
        }
    }

    // MARK: - Post-game chat

    /// When to fetch a finished game's chat. Lichess closes the game stream
    /// right after the final state, so chat after that (an opponent's "gg")
    /// arrives only through `GET /api/bot/game/{id}/chat`.
    static let postGameChatFetchDelays: [Duration] = [.seconds(60), .seconds(300)]
    /// This controller's fetch delays: `postGameChatFetchDelays`, or a
    /// shorter schedule a test passes in.
    private let postGameChatFetchSchedule: [Duration]

    /// Finished games whose filing waits for their last post-game chat
    /// fetch, so everything fetched reaches the record (a filed journal
    /// takes no more lines). A drain (including going offline) files them
    /// at once instead; a quit or a failure stops the reconciler with the
    /// runtime, so their journals are left to `recoverLeftoverJournals` the
    /// next time the bot goes online.
    private var gamesAwaitingPostGameChat: Set<String> = []

    /// Finished games handed to filing before their chat fetches ran (a
    /// drain doesn't wait for them). Their remaining fetches are skipped,
    /// and say so: lines fetched for a filed game could not be kept.
    private var gamesFiledBeforeTheirChatFetches: Set<String> = []

    // MARK: - Games left from the last run

    /// Games whose journal an earlier run left in `InProgress/` (it stopped
    /// with them running, or before filing them), found at launch while the
    /// bot is offline. Any still live on Lichess runs on DCM's clock until
    /// the bot goes online and resumes it; the status chip says so.
    private(set) var leftoverGamesFromLastRun: [String] = []
    private var leftoverJournalsChecked = false

    /// Look once, at launch, for journals an earlier run left, and say so
    /// (status chip, alarm, session log). Called from the main window's
    /// status chip, which exists only in a GUI run: the controller itself is
    /// also created for command-line runs, which must not report this.
    func noteLeftoverJournalsAtLaunch() async {
        guard !leftoverJournalsChecked else { return }
        leftoverJournalsChecked = true
        guard runtime == nil else { return }
        let store = LichessBotRecordStore(directory: dataDirectory, journalQueue: journalQueue, indexQueue: fileQueue, ourAccountID: accountID)
        let leftovers: [String]
        do {
            leftovers = try await store.inProgressGameIDs()
        } catch {
            raiseAlarm("Checking for games left from the last run failed: \(Self.safeDescription(error))")
            return
        }
        guard runtime == nil, !leftovers.isEmpty else { return }
        leftoverGamesFromLastRun = leftovers
        raiseAlarm("The last run left \(leftovers.count) game(s) not filed (\(leftovers.joined(separator: ", "))). Any still live on Lichess is running on DCM's clock: go online to resume it.")
    }

    /// Whether this game's filing waits for its post-game chat: the
    /// reconciler leaves it alone until `fileAfterPostGameChat` hands it over.
    private func isAwaitingPostGameChat(_ gameID: String) -> Bool {
        gamesAwaitingPostGameChat.contains(gameID)
    }

    private func schedulePostGameChatFetches(_ gameID: String) {
        let generationAtStart = runtimeGeneration
        let lastIndex = postGameChatFetchSchedule.count - 1
        for (index, delay) in postGameChatFetchSchedule.enumerated() {
            Task {
                do {
                    try await Task.sleep(for: delay)
                } catch {
                    return
                }
                guard runtimeGeneration == generationAtStart else { return }
                await fetchPostGameChat(gameID)
                if index == lastIndex {
                    fileAfterPostGameChat(gameID)
                }
            }
        }
    }

    /// Queue a game for filing now that its last post-game chat fetch is
    /// done (whether it found anything or failed).
    private func fileAfterPostGameChat(_ gameID: String) {
        guard gamesAwaitingPostGameChat.remove(gameID) != nil, let reconciler = runtime?.reconciler else { return }
        Task {
            await reconciler.enqueue(gameID: gameID)
        }
    }

    private func fetchPostGameChat(_ gameID: String) async {
        if gamesFiledBeforeTheirChatFetches.contains(gameID) {
            protocolLog.record(.game, "post-game chat not fetched: the game was filed without waiting for it while the bot went offline", gameID: gameID)
            return
        }
        guard let runtime, let game = games.first(where: { $0.id == gameID }) else {
            protocolLog.record(.game, "post-game chat not fetched: the game is no longer listed or the bot is offline", gameID: gameID)
            return
        }
        do {
            let fetched = try await runtime.client.gameChat(gameID: gameID)
            for line in game.unseenChatLines(in: fetched) {
                let event = LichessBotGameEvent.chatFetched(username: line.user, text: line.text)
                game.apply(event)
                await runtime.journal.gameEvent(gameID: gameID, event)
            }
        } catch {
            protocolLog.record(.anomaly, "post-game chat fetch failed: \(Self.safeDescription(error))", gameID: gameID)
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
        // Games waiting for their post-game chat fetch, or for a filing
        // delay, are filed now, so going offline doesn't wait for them.
        let reconciler = runtime.reconciler
        let waiting = gamesAwaitingPostGameChat
        gamesAwaitingPostGameChat = []
        gamesFiledBeforeTheirChatFetches.formUnion(waiting)
        for gameID in waiting {
            await reconciler.enqueue(gameID: gameID)
        }
        await reconciler.expediteUnattempted()
        protocolLog.record(.lifecycle, "draining")
        SessionLogger.shared.log("[LICHESS-BOT] draining (\(activeGameIDs.count) game(s) in progress)")
        finishIfDrained()
    }

    /// Go offline. With games in progress this drains and shows the
    /// "Finishing games" sheet instead (plan §13).
    func goOffline() async {
        guard let runtime else { return }
        let manager = runtime.manager
        // The operator's queued picks go with the pending challenges (plan
        // §7.3 A); a drain alone would keep them.
        clearChallengeQueue(reason: "going offline")
        if hasGamesInPlay || !gamesAwaitingPostGameChat.isEmpty {
            if hasGamesInPlay {
                finishing = .goOffline
                LichessBotWindowLauncher.openWindow(controller: self)
            }
            // Draining stops new games and leaves Online, so a challenge
            // still being sent withdraws itself. Finished games still waiting
            // to be filed are filed by the drain (the reconciler stops with
            // the runtime), within the filing time limit.
            await drain()
            // Then withdraw our unanswered challenges: one accepted while the
            // bot finishes would start a game nobody plays. If the drain
            // already stopped the runtime, teardown withdrew them.
            for pending in pendingChallenges where self.runtime?.manager === manager {
                await cancelChallenge(id: pending.id)
            }
            return
        }
        // Stopping the runtime withdraws our unanswered challenges.
        stopRuntime(reason: "operator went offline")
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
        clearChallengeQueue(reason: "app quit")
        if !hasGamesInPlay {
            // Reply once the queued journal and protocol-log writes (the
            // "app quit" line among them) have reached their files.
            quitReplyPending = true
            stopRuntime(reason: "app quit")
            completeQuitIfPending(quit: true)
            return .terminateLater
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
        guard quit else {
            NSApp.reply(toApplicationShouldTerminate: false)
            return
        }
        // Let journal and protocol-log writes already queued reach the files
        // before the process exits; anything later is refused and logged
        // rather than racing the exit.
        Task { @MainActor in
            await shutdown(reason: "app quit")
            NSApp.reply(toApplicationShouldTerminate: true)
        }
    }

    // MARK: - Shutdown

    /// The shutdown under way or done; every caller awaits the same one.
    private var shutdownTask: Task<Void, Never>?

    /// The controller has shut down (or is shutting down) and does no more
    /// file work.
    var isShutDown: Bool {
        shutdownTask != nil
    }

    /// Stop the bot for good and wait until nothing more reaches its data
    /// folder: stop the runtime if one is up, cancel the controller's own
    /// timers, then close the journal queue and the general file queue.
    /// Closing a queue first runs everything already enqueued on it — the
    /// stopped runtime's lock release and any journal lines already queued,
    /// every protocol event recorded so far, including this shutdown's own —
    /// and then refuses all later work, writing each refusal to the session
    /// log.
    ///
    /// Work started before the shutdown may still finish after it: a request
    /// in flight (an opponent-profile fetch, a challenge withdrawal the
    /// teardown started) still completes, and its gate events and request
    /// record are protocol-log appends. Those are refused, so once this
    /// returns the folder can be deleted without an append recreating
    /// `Protocol/` in the middle of the removal (which makes the removal
    /// fail). Abandoned game sessions wind down the same way: a journal line
    /// they produce after the close is refused. The request gate is left
    /// open on purpose: closing it would also stop the withdrawals of our
    /// unanswered challenges, and a challenge left standing can be accepted
    /// into a game nobody plays.
    ///
    /// The app calls this when it quits; tests call it before deleting a
    /// controller's temporary data folder. A controller that has shut down
    /// does not go online again. Idempotent: a second call waits for the
    /// first to finish.
    func shutdown(reason: String) async {
        if let shutdownTask {
            await shutdownTask.value
            return
        }
        let task = Task { @MainActor in
            await performShutdown(reason: reason)
        }
        shutdownTask = task
        await task.value
    }

    private func performShutdown(reason: String) async {
        protocolLog.record(.lifecycle, "shutting down: \(reason)")
        stopRuntime(reason: reason)
        cancelAutoFollowHold()
        gridClockTask?.cancel()
        gridClockTask = nil
        // The journal queue first: a stopped runtime's lock release and any
        // journal lines it already queued are on it.
        await journalQueue.close(reason: "the bot shut down: \(reason)")
        await fileQueue.close(reason: "the bot shut down: \(reason)")
        SessionLogger.shared.log("[LICHESS-BOT] shut down: \(reason)")
    }

    // MARK: - Challenges (plan §7.1)

    /// Fetch the online-bots list, or join the fetch already under way;
    /// returns when that fetch has finished, whether or not it succeeded
    /// (a failure raises an alarm and leaves the previous list).
    func refreshOnlineBots() async {
        await onlineBotsRefreshTask().value
    }

    /// The online-bots fetch under way, started now if none is. Never a
    /// second fetch while one is in flight.
    ///
    /// A fetch still in flight when the controller shuts down is left to
    /// finish, like an opponent-profile fetch: its request record (and any
    /// failure alarm's log line) reaches only closed queues, which refuse
    /// it. Cancelling it instead would only turn it into a failure alarm.
    @discardableResult
    private func onlineBotsRefreshTask() -> Task<Void, Never> {
        if let onlineBotsRefresh {
            return onlineBotsRefresh
        }
        let task = Task {
            await fetchOnlineBots()
            // Stored below before this body can run (both are on the main
            // actor, and nothing between them suspends), and nothing else
            // replaces it while it runs, so this clears this fetch's own entry.
            onlineBotsRefresh = nil
        }
        onlineBotsRefresh = task
        return task
    }

    private func fetchOnlineBots() async {
        onlineBotsRequestedAt = Date()
        do {
            let client = try await accountClient()
            onlineBots = try await client.onlineBots(count: LichessBotLimits.onlineBotsMaximum)
            onlineBotsFetchedAt = Date()
        } catch {
            raiseAlarm("Loading online bots failed: \(Self.safeDescription(error))")
        }
    }

    // MARK: - Player notes (plan §7.2)

    /// Load favorites and bot limit times from disk. A missing file is an
    /// empty set of notes; an unreadable one raises an alarm and leaves the
    /// notes unloaded (so nothing overwrites the file).
    func loadPlayerNotes() async {
        let url = dataDirectory.playerNotesURL
        do {
            var notes = try await fileQueue.run {
                try LichessBotPlayerNotes.load(from: url)
            }
            notes.pruneExpiredLimits(now: Date())
            playerNotes = notes
            // A refusal recorded before the notes loaded stays: the later of
            // the two ends wins.
            botLimitUntil.merge(notes.botLimitUntil) { max($0, $1) }
        } catch {
            raiseAlarm("Loading favorites failed (\(url.lastPathComponent)): \(error.localizedDescription)")
        }
    }

    // MARK: - Challenge outcomes

    /// Load the outgoing-challenge outcome log. A missing file is an empty
    /// log; an unreadable one raises an alarm and leaves the log unloaded,
    /// so nothing overwrites the file.
    func loadChallengeOutcomes() async {
        let url = dataDirectory.challengeOutcomesURL
        do {
            var log = try await fileQueue.run {
                try LichessBotChallengeOutcomeLog.load(from: url)
            }
            log.prune(now: Date())
            challengeOutcomeLog = log
        } catch {
            raiseAlarm("Loading challenge outcomes failed (\(url.lastPathComponent)): \(error.localizedDescription)")
        }
    }

    /// Apply `change` to the outcome log, prune it, and save it. Saves are
    /// enqueued on the serial file queue from the main actor, so they land
    /// in the order the changes were made.
    private func updateChallengeOutcomeLog(_ what: String, _ change: (inout LichessBotChallengeOutcomeLog) -> Void) {
        guard var log = challengeOutcomeLog else {
            protocolLog.record(.anomaly, "challenge outcome log isn't loaded; not recorded: \(what)")
            return
        }
        let now = Date()
        change(&log)
        log.prune(now: now)
        challengeOutcomeLog = log
        let summary = log.summary(now: now)
        protocolLog.record(.challenge, "challenge outcome: \(what)", fields: [
            "credits_day": "\(summary.creditsLastDay)/\(LichessBotChallengeCredits.perDay)",
            "credits_minute": "\(summary.creditsLastMinute)/\(LichessBotChallengeCredits.perMinute)",
        ])
        let url = dataDirectory.challengeOutcomesURL
        let snapshot = log
        fileQueue.enqueue("save \(url.lastPathComponent) after: \(what)") {
            do {
                try snapshot.save(to: url)
            } catch {
                let text = "Saving challenge outcomes failed (\(url.lastPathComponent)): \(error.localizedDescription)"
                Task { @MainActor [weak self] in
                    self?.raiseAlarm(text)
                }
            }
        }
    }

    /// Resolve a created challenge's record, if `resolve` would change it.
    private func resolveChallengeOutcome(challengeID: String, _ outcome: LichessBotChallengeOutcome) {
        guard let log = challengeOutcomeLog,
              log.canResolve(challengeID: challengeID, outcome: outcome) else { return }
        updateChallengeOutcomeLog("\(challengeID) \(Self.describe(outcome))") { log in
            log.resolve(challengeID: challengeID, outcome: outcome, at: Date())
        }
    }

    nonisolated static func describe(_ outcome: LichessBotChallengeOutcome) -> String {
        switch outcome {
        case .accepted:
            return "accepted"
        case .declined(let reason):
            return "declined (\(reason.keyText))"
        case .canceled:
            return "canceled"
        case .offline:
            return "offline"
        case .refused(let refusal):
            return "refused: \(refusal.kind.label), HTTP \(refusal.httpStatus)\(refusal.text.map { ": \($0)" } ?? "")"
        }
    }

    /// Who `username` is for credit costs, from their title.
    nonisolated static func challengeOpponentKind(title: String?) -> LichessBotChallengeOpponentKind {
        title == "BOT" ? .bot : .human
    }

    func toggleFavorite(_ userID: String) {
        guard var notes = playerNotes else {
            raiseAlarm("Favorites aren't loaded; not changing them")
            return
        }
        notes.toggleFavorite(userID)
        playerNotes = notes
        savePlayerNotes(notes)
    }

    /// Each bot's bot-vs-bot daily limit end, from Lichess's own refusals,
    /// by lowercased id: what the running bot consults (matchmaking, the
    /// challenge queue, the Challenge sheet). Loaded from the player notes,
    /// where refusals are also saved, but kept here even when the notes
    /// couldn't be loaded, so a refused bot is never asked again too early.
    private(set) var botLimitUntil: [String: Date] = [:]

    /// When `userID`'s bot-vs-bot limit ends, if that is still ahead of `now`.
    func botLimitEnds(_ userID: String, now: Date) -> Date? {
        guard let until = botLimitUntil[userID.lowercased()], until > now else { return nil }
        return until
    }

    /// The player notes as the bot lists and matchmaking read them: with the
    /// live bot limits. Unloaded notes (an unreadable file) are known to
    /// hold no favorites or cool-downs — what an absent notes value told
    /// those readers before — while the limits still count.
    var playerNotesWithLiveBotLimits: LichessBotPlayerNotes {
        var notes: LichessBotPlayerNotes
        if let playerNotes {
            notes = playerNotes
        } else {
            notes = LichessBotPlayerNotes()
        }
        notes.botLimitUntil = botLimitUntil
        return notes
    }

    private func recordBotLimit(_ refusal: LichessBotBotLimitRefusal.Parsed) {
        protocolLog.record(.challenge, "\(refusal.userID) is at its bot-game limit (\(refusal.gamesPlayed)) until \(refusal.until.formatted(date: .abbreviated, time: .standard))")
        botLimitUntil[refusal.userID] = refusal.until
        guard var notes = playerNotes else {
            protocolLog.record(.anomaly, "player notes aren't loaded: \(refusal.userID)'s bot-game limit is kept for this launch only")
            return
        }
        notes.botLimitUntil[refusal.userID] = refusal.until
        playerNotes = notes
        savePlayerNotes(notes)
    }

    private func savePlayerNotes(_ notes: LichessBotPlayerNotes) {
        let url = dataDirectory.playerNotesURL
        let queue = fileQueue
        Task {
            do {
                try await queue.run {
                    try notes.save(to: url)
                }
            } catch {
                raiseAlarm("Saving favorites failed (\(url.lastPathComponent)): \(error.localizedDescription)")
            }
        }
    }

    /// Online and playing flags for every favorite (the Favorites tab shows
    /// both, and the online list reports neither for players missing from
    /// it), at housekeeping priority.
    func refreshFavoriteStatuses() async {
        guard let notes = playerNotes else { return }
        do {
            try await refreshPlayerStatuses(notes.favoriteIDs)
        } catch {
            raiseAlarm("Checking favorites' online status failed: \(Self.safeDescription(error))")
        }
    }

    /// Fetch statuses for those of `ids` not asked for within
    /// `playerStatusMaximumAge`, in batches at housekeeping priority. An id
    /// Lichess leaves out loses any older status, so a closed account
    /// doesn't keep showing online.
    func refreshPlayerStatuses(_ ids: [String]) async throws {
        let now = Date()
        let stale = Set(ids.map { $0.lowercased() }).filter { id in
            guard let fetchedAt = playerStatusFetchedAt[id] else { return true }
            return now.timeIntervalSince(fetchedAt) >= Self.playerStatusMaximumAge
        }
        guard !stale.isEmpty else { return }
        let fetched = try await userStatuses(Array(stale))
        let fetchedAt = Date()
        for id in stale {
            playerStatuses[id] = fetched[id]
            playerStatusFetchedAt[id] = fetchedAt
        }
    }

    // MARK: - Opponent profiles (plan §14.3b)

    /// Start fetching an opponent's profile and Lichess head-to-head, once
    /// per app session, in the background at housekeeping priority (the gate
    /// never starts it while a move is due). `retry` refetches after a
    /// failure; nothing retries on its own.
    func loadOpponentProfile(_ username: String, retry: Bool = false) {
        let id = username.lowercased()
        guard id != accountID else { return }
        if let existing = opponentProfiles[id] {
            guard retry, case .failed = existing else { return }
        }
        opponentProfiles[id] = .loading
        Task {
            await fetchOpponentProfile(username: username, id: id)
        }
    }

    private func fetchOpponentProfile(username: String, id: String) async {
        do {
            let client = try await accountClient()
            let user = try await client.user(username: username)
            var crosstable: LichessBotCrosstable?
            var crosstableError: String?
            do {
                crosstable = try await client.crosstable(accountID, id)
            } catch {
                crosstableError = Self.safeDescription(error)
            }
            opponentProfiles[id] = .loaded(user, crosstable: crosstable, crosstableError: crosstableError)
        } catch {
            opponentProfiles[id] = .failed(Self.safeDescription(error))
        }
    }

    /// DCM's games against bots started in the 24 hours before `now`:
    /// filed games plus live ones not yet filed; nil until the games index
    /// is loaded. Lichess limits this to `LichessBotLimits.botGamesPerDay`.
    func botGamesInLastDay(now: Date) -> Int? {
        guard let rows = index?.rows else { return nil }
        let since = now.addingTimeInterval(-24 * 3600)
        let filedIDs = Set(rows.map(\.gameID))
        let live = games.filter { game in
            !filedIDs.contains(game.id) && game.opponent?.title == "BOT" && game.startedAt >= since
        }.count
        return LichessBotRecordSummary.botGames(rows: rows, since: since) + live
    }

    /// A speed's leaderboard, fetched when asked for and kept for
    /// `leaderboardMaximumAge` (Lichess itself recomputes it every few
    /// minutes).
    private(set) var leaderboards: [LichessBotSpeed: (fetchedAt: Date, users: [LichessBotLeaderboardUser])] = [:]
    private static let leaderboardMaximumAge: TimeInterval = 300

    func refreshLeaderboard(_ speed: LichessBotSpeed, force: Bool = false) async throws {
        if !force, let cached = leaderboards[speed], Date().timeIntervalSince(cached.fetchedAt) < Self.leaderboardMaximumAge {
            return
        }
        let client = try await accountClient()
        leaderboards[speed] = (Date(), try await client.leaderboard(speed: speed, count: LichessBotLimits.leaderboardMaximum))
    }

    /// The highest-rated online humans from Lichess's undocumented
    /// `/player/online`, refetched only after `onlinePlayersMaximumAge`
    /// (matching how often Lichess refreshes it).
    private(set) var onlinePlayers: (fetchedAt: Date, users: [LichessBotUserSummary])?
    private static let onlinePlayersMaximumAge: TimeInterval = 120

    func refreshOnlinePlayers() async throws {
        if let onlinePlayers, Date().timeIntervalSince(onlinePlayers.fetchedAt) < Self.onlinePlayersMaximumAge {
            return
        }
        let client = try await accountClient()
        onlinePlayers = (Date(), try await client.onlinePlayers())
    }

    /// Online and playing flags for players, by lowercased id, in batches of
    /// at most `LichessBotLimits.userStatusMaximumIDs` (housekeeping
    /// priority; Lichess calls this endpoint cheap).
    func userStatuses(_ ids: [String]) async throws -> [String: LichessBotUserStatus] {
        let client = try await accountClient()
        var statuses: [String: LichessBotUserStatus] = [:]
        var start = 0
        while start < ids.count {
            let batch = Array(ids[start..<min(start + LichessBotLimits.userStatusMaximumIDs, ids.count)])
            for status in try await client.usersStatus(ids: batch) {
                statuses[status.id.lowercased()] = status
            }
            start += LichessBotLimits.userStatusMaximumIDs
        }
        return statuses
    }

    func autocompleteUsers(_ term: String) async throws -> [LichessBotLightUser] {
        let client = try await accountClient()
        return try await client.autocompleteUsers(term: term)
    }

    /// Opponents from DCM's own records, most recent game first, one row
    /// each (the Username tab before anything is typed). Nil until the
    /// games index loads.
    var recentOpponents: [LichessBotLightUser]? {
        guard let rows = index?.rows else { return nil }
        var seen: Set<String> = []
        var result: [LichessBotLightUser] = []
        for row in rows.sorted(by: { $0.createdAt > $1.createdAt }) {
            guard let id = row.opponentID, let name = row.opponentName, seen.insert(id).inserted else { continue }
            result.append(LichessBotLightUser(id: id, name: name, title: row.opponentTitle, online: nil))
        }
        return result
    }

    func lookUpUser(_ username: String) async throws -> LichessBotUserSummary {
        let client = try await accountClient()
        do {
            return try await client.user(username: username)
        } catch LichessBotAPIError.http(404, _) {
            // Lichess answers an unknown username with its HTML "not found"
            // page; say what it means instead.
            throw LichessBotControllerError.noSuchPlayer(username)
        }
    }

    /// Send the operator's challenge. The bot must be online (the game
    /// arrives on its event stream). Several may be pending, within the
    /// concurrent-game limit and the per-opponent limit.
    func sendChallenge(to username: String, request: LichessBotOutgoingChallenge) async throws {
        try await sendChallenge(to: username, request: request, origin: .manual)
    }

    /// Send a challenge on behalf of `origin`, which the pending challenge
    /// keeps so its decline is answered the way its sender calls for. Every
    /// send — the operator's, the queue's and matchmaking's — goes through
    /// here, so every check applies to all of them alike.
    private func sendChallenge(to username: String, request: LichessBotOutgoingChallenge, origin: ChallengeOrigin) async throws {
        guard let runtime, connection == .online else {
            throw LichessBotControllerError.notOnline
        }
        let limit = settings.challenge.maxConcurrentGames
        let committed = committedGameSlots
        guard committed < limit else {
            throw LichessBotControllerError.concurrentGameLimit(limit: limit, committed: committed)
        }
        // Held for the POST, so a second send started meanwhile counts it.
        challengeSendsInFlight += 1
        defer { challengeSendsInFlight -= 1 }
        guard hasChallengeScope else {
            throw LichessBotControllerError.missingChallengeScope
        }
        let manager = runtime.manager
        let client = runtime.client
        let opponentID = username.lowercased()
        let perOpponentLimit = settings.challenge.maxSimultaneousGamesPerOpponent
        // Held in the manager for the POST, so an incoming challenge from the
        // same player, or a second send to them, counts this one.
        let reservation = await manager.reserveOutgoingChallenge(against: opponentID, perOpponentLimit: perOpponentLimit)
        guard reservation.reserved else {
            throw LichessBotControllerError.perOpponentGameLimit(username: username, limit: perOpponentLimit, committed: reservation.committedBefore)
        }
        let created: LichessBotChallenge
        do {
            guard self.runtime?.manager === manager else {
                throw LichessBotControllerError.notOnline
            }
            // Lichess keeps a challenge acceptable for hours after the
            // challenged player comes back, so a challenge to someone offline
            // could be accepted while DCM is gone. Only online players are
            // challenged.
            let statuses = try await client.usersStatus(ids: [username])
            guard let status = statuses.first(where: { $0.id.lowercased() == opponentID }) else {
                throw LichessBotControllerError.noSuchPlayer(username)
            }
            guard status.online == true else {
                let offlineKind = Self.challengeOpponentKind(title: status.title)
                updateChallengeOutcomeLog("\(opponentID) offline") { log in
                    log.recordNotCreated(opponentID: opponentID, kind: offlineKind, outcome: .offline, at: Date())
                }
                throw LichessBotControllerError.opponentOffline(status.name)
            }
            // The status check was awaited, so the bot may have gone
            // offline meanwhile.
            guard self.runtime?.manager === manager else {
                throw LichessBotControllerError.notOnline
            }
            let opponentKind = Self.challengeOpponentKind(title: status.title)
            do {
                created = try await client.challenge(username: username, request: request)
            } catch {
                // A bot at its bot-vs-bot daily limit is refused with the
                // exact time it frees up (plan §7.2); remember it for the list.
                if case LichessBotAPIError.http(_, let message?) = error, let refusal = LichessBotBotLimitRefusal.parse(message) {
                    recordBotLimit(refusal)
                }
                if let refusal = LichessBotChallengeRefusal.classify(postError: error) {
                    let outcome = LichessBotChallengeOutcome.refused(refusal)
                    let charged = LichessBotChallengeOutcomeLog.creditCost(notCreated: outcome, kind: opponentKind)
                    updateChallengeOutcomeLog("\(opponentID) \(Self.describe(outcome)); counted \(charged) credits (worst case)") { log in
                        log.recordNotCreated(opponentID: opponentID, kind: opponentKind, outcome: outcome, at: Date())
                    }
                } else {
                    protocolLog.record(.anomaly, "challenge to \(opponentID) failed without an answer from Lichess; outcome not recorded: \(Self.safeDescription(error))")
                }
                throw error
            }
            updateChallengeOutcomeLog("\(opponentID) challenge \(created.id) created; counted \(LichessBotChallengeCredits.cost(for: opponentKind)) credits (worst case: Lichess charges nothing if they follow DCM)") { log in
                log.recordCreated(challengeID: created.id, opponentID: opponentID, kind: opponentKind, at: Date())
            }
            guard self.runtime?.manager === manager, connection == .online else {
                // The bot went offline, or began going offline, while the
                // challenge was being sent: withdraw it, or an acceptance
                // would start an abandoned game.
                withdraw(challengeID: created.id, client: client)
                resolveChallengeOutcome(challengeID: created.id, .canceled)
                throw LichessBotControllerError.notOnline
            }
        } catch {
            await manager.releaseOutgoingChallengeReservation(against: opponentID)
            throw error
        }
        pendingChallenges.append(PendingChallenge(id: created.id, username: username, sentAt: Date(), request: request, origin: origin))
        casualResendOffer = nil
        loadOpponentProfile(username)
        lastChallengeOutcome = nil
        // Turns the reservation into a pending challenge; may resolve at
        // once, if the answer already arrived.
        await manager.noteSentChallenge(id: created.id, opponentID: opponentID)
        protocolLog.record(.challenge, "challenge sent to \(username)", fields: ["id": created.id, "rated": "\(request.rated)", "clock": "\(request.clockLimitSeconds)+\(request.clockIncrementSeconds)", "color": request.color.rawValue])
    }

    /// Send the declined rated challenge again as casual, as the player
    /// asked. A failure is reported in the outcome line and the offer stays.
    func resendAsCasual() async {
        guard let offer = casualResendOffer else { return }
        do {
            try await sendChallenge(to: offer.username, request: offer.request)
        } catch {
            lastChallengeOutcome = "\(offer.username): resending as casual failed: \(Self.safeDescription(error))"
        }
    }

    func cancelChallenge(id: String) async {
        guard let runtime, let pending = pendingChallenges.first(where: { $0.id == id }) else { return }
        do {
            try await runtime.client.cancelChallenge(id: id)
            await runtime.manager.clearOutgoingChallenge(id: id)
            if pendingChallenges.contains(where: { $0.id == id }) {
                pendingChallenges.removeAll { $0.id == id }
                lastChallengeOutcome = "\(pending.username): canceled"
            }
            resolveChallengeOutcome(challengeID: id, .canceled)
            scheduleChallengeQueuePump()
        } catch LichessBotAPIError.http(let status, let message) where status == 400 || status == 404 {
            // Lichess no longer knows the challenge: it expired or was
            // answered. It is no longer pending.
            await runtime.manager.clearOutgoingChallenge(id: id)
            pendingChallenges.removeAll { $0.id == id }
            lastChallengeOutcome = "\(pending.username): no longer pending (\(message ?? "HTTP \(status)"))"
            resolveChallengeOutcome(challengeID: id, .canceled)
            scheduleChallengeQueuePump()
        } catch {
            raiseAlarm("Cancelling the challenge to \(pending.username) failed: \(Self.safeDescription(error))")
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

    // MARK: - Slots

    /// Everything holding one of the concurrent-game slots: games in
    /// progress, accepted challenges whose game is still starting, our
    /// challenges waiting for an answer, and sends under way. The one count
    /// every send decision uses — a single send, the queue and matchmaking.
    var committedGameSlots: Int {
        activeGameIDs.union(acceptedAwaitingStartIDs).count + pendingChallenges.count + challengeSendsInFlight
    }

    /// Slots a send may still take under the concurrent-game limit.
    var freeChallengeSlots: Int {
        settings.challenge.maxConcurrentGames - committedGameSlots
    }

    /// Why no challenge may be sent now, from the controller's own state and
    /// the latest gate snapshot, or nil when one may.
    private var outgoingSendBlockedReason: String? {
        guard runtime != nil else { return "the bot is offline" }
        if rateLimitHoldUntil != nil {
            return "rate-limit hold after a 429"
        }
        switch connection {
        case .online:
            break
        case .draining:
            return "draining"
        case .offline, .connecting, .error:
            return "the bot is not Online"
        }
        if !hasChallengeScope {
            return "the token lacks challenge:write"
        }
        return Self.gateBlockedReason(gateSnapshot?.phase)
    }

    /// Why the request gate keeps a challenge from going out, or nil when it
    /// is open. No snapshot yet means the runtime hasn't polled it; the
    /// gate's own queueing still applies.
    private static func gateBlockedReason(_ phase: LichessBotRequestGate.Snapshot.Phase?) -> String? {
        switch phase {
        case .none, .open:
            return nil
        case .coolingDown:
            return "Lichess rate limit: requests paused"
        case .closed(let reason):
            return "the request gate is closed (\(reason))"
        }
    }

    // MARK: - Challenge queue (plan §7.3 A)

    /// Why the queue's waiting entries wait, for the Overview; nil when
    /// nothing waits.
    var challengeQueueWaitReason: String? {
        switch challengeQueue.nextStep(sendingBlockedReason: outgoingSendBlockedReason, freeSlots: freeChallengeSlots) {
        case .idle:
            return nil
        case .wait(let reason):
            return reason
        case .send:
            return "sending next"
        }
    }

    /// Add players to the challenge queue, all with one clock, color and
    /// rated setting, and start sending. A player already queued, or with a
    /// challenge waiting for an answer, isn't added again. The bot must be
    /// Online.
    @discardableResult
    func enqueueChallenges(to players: [LichessBotChallengeQueue.Player], request: LichessBotOutgoingChallenge) throws -> LichessBotChallengeQueue.AddResult {
        guard runtime != nil, connection == .online else {
            throw LichessBotControllerError.notOnline
        }
        guard hasChallengeScope else {
            throw LichessBotControllerError.missingChallengeScope
        }
        let pendingIDs = Set(pendingChallenges.map { $0.username.lowercased() })
        let result = challengeQueue.add(players, request: request, pendingUserIDs: pendingIDs)
        let fields = ["rated": "\(request.rated)", "clock": request.clockText, "color": request.color.rawValue]
        for username in result.added {
            protocolLog.record(.challenge, "challenge queue: added \(username)", fields: fields)
        }
        for username in result.alreadyQueued {
            protocolLog.record(.challenge, "challenge queue: \(username) not added: already queued")
        }
        for username in result.alreadyPending {
            protocolLog.record(.challenge, "challenge queue: \(username) not added: a challenge to them is waiting for an answer")
        }
        var summary = "Queued \(result.added.count) challenge(s)"
        let duplicates = result.alreadyQueued + result.alreadyPending
        if !duplicates.isEmpty {
            summary += "; not added (already queued or challenged): \(duplicates.joined(separator: ", "))"
        }
        lastChallengeOutcome = summary
        scheduleChallengeQueuePump()
        return result
    }

    /// The Overview's per-entry Cancel. An entry already being sent still
    /// goes out; its challenge then shows as pending, with its own Cancel.
    func cancelQueuedChallenge(_ id: UUID) {
        guard let entry = challengeQueue.entries.first(where: { $0.id == id }) else { return }
        challengeQueue.remove(id)
        let whileSending = entry.status == .sending ? " while its challenge was being sent" : ""
        protocolLog.record(.challenge, "challenge queue: removed \(entry.username)\(whileSending)")
    }

    /// The Overview's Clear Queue, and going offline or quitting.
    func clearChallengeQueue(reason: String = "cleared by the operator") {
        guard !challengeQueue.isEmpty else { return }
        protocolLog.record(.challenge, "challenge queue: \(challengeQueue.entries.count) entr(ies) removed: \(reason)", fields: ["players": challengeQueue.entries.map(\.username).joined(separator: ",")])
        challengeQueue.removeAll()
    }

    /// Start the queue pump if an entry could be sent now. Deciding that
    /// needs no request, so the poll loop calls this every second, and every
    /// event that frees a slot calls it at once.
    private func scheduleChallengeQueuePump() {
        guard !challengeQueuePumpRunning,
              case .send = challengeQueue.nextStep(sendingBlockedReason: outgoingSendBlockedReason, freeSlots: freeChallengeSlots) else { return }
        challengeQueuePumpRunning = true
        let generation = runtimeGeneration
        Task {
            await pumpChallengeQueue(generation: generation)
        }
    }

    /// Send queued entries one at a time while slots are free. Each send is
    /// `sendChallenge`, so every check of a single send applies unchanged:
    /// online status, the concurrent-game and per-opponent limits, the bot
    /// limit refusal. Stops when nothing can be sent, when a 429 or a state
    /// change stops a send (that entry waits again in its place), or when
    /// the runtime it started under is gone (teardown cleared the queue).
    private func pumpChallengeQueue(generation generationAtStart: Int) async {
        // Every await can outlive this runtime; nothing is written after it.
        func current() -> Bool { runtimeGeneration == generationAtStart }
        defer {
            // Teardown clears the flag itself; a stale pump must not clear
            // the flag a newer runtime's pump set.
            if current() { challengeQueuePumpRunning = false }
        }
        while current() {
            // The gate's live phase, not the poll loop's copy: a 429 a
            // moment ago must stop the queue before the hold is in place.
            let phase = await gate.snapshot().phase
            guard current() else { return }
            let blocked = outgoingSendBlockedReason ?? Self.gateBlockedReason(phase)
            guard case .send(let entry) = challengeQueue.nextStep(sendingBlockedReason: blocked, freeSlots: freeChallengeSlots) else { return }
            if let until = botLimitEnds(entry.userID, now: Date()) {
                let reason = "at its bot-game limit until \(until.formatted(date: .omitted, time: .shortened))"
                challengeQueue.skip(entry.id, reason: reason)
                protocolLog.record(.challenge, "challenge queue: skipped \(entry.username): \(reason)")
                continue
            }
            challengeQueue.markSending(entry.id)
            let outcome: LichessBotChallengeQueue.SendOutcome
            do {
                try await sendChallenge(to: entry.username, request: entry.request)
                outcome = .sent
            } catch {
                outcome = Self.queueOutcome(for: error)
            }
            guard current() else { return }
            challengeQueue.record(outcome, for: entry.id)
            switch outcome {
            case .sent:
                protocolLog.record(.challenge, "challenge queue: sent \(entry.username)")
            case .skipped(let reason):
                protocolLog.record(.challenge, "challenge queue: skipped \(entry.username): \(reason)")
            case .dropped(let reason):
                lastChallengeOutcome = "\(entry.username): not sent: \(reason)"
                protocolLog.record(.challenge, "challenge queue: dropped \(entry.username): \(reason)")
            case .stopped(let reason):
                protocolLog.record(.challenge, "challenge queue: stopped at \(entry.username), which waits again: \(reason)")
                return
            }
        }
    }

    /// How a failed queued send affects its entry (plan §7.3 A): reasons
    /// tied to the player skip it; a 429, a closed gate or a change of
    /// state stops the queue with the entry kept; anything else Lichess
    /// refused drops it.
    nonisolated static func queueOutcome(for error: Error) -> LichessBotChallengeQueue.SendOutcome {
        let text = safeDescription(error)
        if let controllerError = error as? LichessBotControllerError {
            switch controllerError {
            case .perOpponentGameLimit:
                return .skipped(reason: "already playing or challenging them (the per-opponent limit)")
            case .opponentOffline:
                return .skipped(reason: "offline")
            case .notOnline, .concurrentGameLimit, .missingChallengeScope, .noToken, .tokenInvalid, .tokenForWrongAccount, .notABot:
                return .stopped(reason: text)
            case .noSuchPlayer, .chatNotSendable:
                return .dropped(reason: text)
            }
        }
        if error is LichessBotGateError || error is CancellationError {
            return .stopped(reason: text)
        }
        if case LichessBotAPIError.http(_, let message?) = error, let refusal = LichessBotBotLimitRefusal.parse(message) {
            return .skipped(reason: "at its bot-game limit until \(refusal.until.formatted(date: .omitted, time: .shortened))")
        }
        return .dropped(reason: text)
    }

    // MARK: - Matchmaking (plan §7.3 B)

    enum MatchmakingPassOutcome: Equatable {
        case sent(username: String)
        /// Matchmaking may not send at all now.
        case blocked(reason: String)
        /// Every slot matchmaking may use is taken.
        case noOpenSlot
        case rateLimited(LichessBotMatchmakingRateLimiter.Decision)
        case noCandidate(String)
        case failed(String)
        /// A 429, the gate closing, or the runtime going away stopped the
        /// send.
        case stopped(String)

        var text: String {
            switch self {
            case .sent(let username):
                return "sent a challenge to \(username)"
            case .blocked(let reason):
                return "paused: \(reason)"
            case .noOpenSlot:
                return "every matchmaking slot is taken"
            case .rateLimited(.spacing(let until)):
                return "next send not before \(until.formatted(date: .omitted, time: .standard)) (spacing)"
            case .rateLimited(.hourlyCap(let until, let count)):
                return "\(count) challenges in the last hour, the cap; next at \(until.formatted(date: .omitted, time: .shortened))"
            case .rateLimited(.allowed):
                return "allowed"
            case .noCandidate(let detail):
                return detail
            case .failed(let detail):
                return "failed: \(detail)"
            case .stopped(let detail):
                return "stopped: \(detail)"
            }
        }
    }

    private static let secondsPerHour: TimeInterval = 3600

    /// The Overview's Fill Open Slots: one pass that fills every open slot
    /// (outside those reserved for humans) with matchmaking's criteria, even
    /// when matchmaking is off. Sends go one at a time with matchmaking's
    /// spacing, and the per-hour cap still holds.
    func fillOpenSlots() async {
        guard !isFillingOpenSlots, runtime != nil else { return }
        let generationAtStart = runtimeGeneration
        func current() -> Bool { runtimeGeneration == generationAtStart }
        isFillingOpenSlots = true
        defer {
            if current() { isFillingOpenSlots = false }
        }
        protocolLog.record(.challenge, "matchmaking: Fill Open Slots")
        var sentCount = 0
        while current() {
            // An automatic pass may be deciding or sending; let it finish.
            while matchmakingPassRunning {
                do {
                    try await Task.sleep(for: .milliseconds(250))
                } catch {
                    return
                }
                guard current() else { return }
            }
            let outcome = await runMatchmakingPass(fillMode: .everyFreeSlot)
            guard current() else { return }
            switch outcome {
            case .sent:
                sentCount += 1
                continue
            case .rateLimited(.spacing(let until)):
                let delay = until.timeIntervalSinceNow
                if delay > 0 {
                    do {
                        try await Task.sleep(for: .seconds(delay))
                    } catch {
                        return
                    }
                }
                continue
            default:
                let summary = "Fill Open Slots sent \(sentCount); \(outcome.text)"
                matchmakingStatus = "\(Date().formatted(date: .omitted, time: .standard)) \(summary)"
                protocolLog.record(.challenge, "matchmaking: \(summary)")
                return
            }
        }
    }

    /// Start an automatic pass when matchmaking is on and one is due. The
    /// common "every slot is taken" case is decided here without a request
    /// or a log line.
    private func startAutomaticMatchmakingPassIfDue() {
        let matchmaking = settings.matchmaking
        guard matchmaking.enabled, !matchmakingPassRunning, !isFillingOpenSlots,
              runtime != nil, connection == .online, rateLimitHoldUntil == nil, !oneGameRequested,
              !challengeQueue.hasEntriesToSend, Date() >= nextAutomaticMatchmakingPassAt else { return }
        let slots = LichessBotMatchmaking.openSlots(
            maxConcurrentGames: settings.challenge.maxConcurrentGames,
            gamesReservedForHumans: settings.challenge.gamesReservedForHumans,
            fillMode: matchmaking.fillMode,
            committed: committedGameSlots
        )
        guard slots > 0 else {
            // Assigned only on a change: every assignment redraws the
            // Overview, and this runs every poll.
            let text = MatchmakingPassOutcome.noOpenSlot.text
            if matchmakingStatus != text {
                matchmakingStatus = text
            }
            return
        }
        // Held until the pass reports, so the next poll can't start another.
        nextAutomaticMatchmakingPassAt = .distantFuture
        let generation = runtimeGeneration
        Task {
            let outcome = await runMatchmakingPass(fillMode: matchmaking.fillMode)
            // Teardown reset the schedule for the next runtime.
            guard runtimeGeneration == generation else { return }
            nextAutomaticMatchmakingPassAt = Self.nextAutomaticPass(after: outcome, now: Date())
            let text = outcome.text
            matchmakingStatus = "\(Date().formatted(date: .omitted, time: .standard)) \(text)"
            if case .sent = outcome {
                lastLoggedAutomaticMatchmakingOutcome = nil
            } else if text != lastLoggedAutomaticMatchmakingOutcome {
                lastLoggedAutomaticMatchmakingOutcome = text
                protocolLog.record(.challenge, "matchmaking: \(text)")
            }
        }
    }

    /// When the next automatic pass may run after `outcome`.
    nonisolated static func nextAutomaticPass(after outcome: MatchmakingPassOutcome, now: Date) -> Date {
        switch outcome {
        case .sent, .noOpenSlot:
            // The rate limiter decides the spacing on the next pass.
            return now
        case .rateLimited(.spacing(let until)), .rateLimited(.hourlyCap(let until, _)):
            return until
        case .rateLimited(.allowed):
            return now
        case .blocked, .noCandidate, .failed, .stopped:
            return now.addingTimeInterval(LichessBotMatchmaking.retryAfterUnproductivePass)
        }
    }

    /// One matchmaking pass: at most one challenge, sent only if every
    /// condition holds before the send (plan §7.3 B). The conditions are
    /// checked again after the online-bots refresh, since that awaits.
    private func runMatchmakingPass(fillMode: LichessBotMatchmakingSettings.FillMode) async -> MatchmakingPassOutcome {
        guard !matchmakingPassRunning else { return .blocked(reason: "another matchmaking pass is running") }
        matchmakingPassRunning = true
        let generationAtStart = runtimeGeneration
        func current() -> Bool { runtimeGeneration == generationAtStart }
        defer {
            if current() { matchmakingPassRunning = false }
        }
        if let outcome = await matchmakingPrecheck(fillMode: fillMode) {
            return outcome
        }
        guard current() else { return .stopped("the bot went offline") }
        // A fetch already under way (the poll loop's, the Challenge sheet's)
        // is waited for, not skipped: deciding before it lands would, right
        // after going Online, find no list at all.
        if onlineBotsRefresh != nil || onlineBotsNeedMatchmakingRefresh(now: Date()) {
            await refreshOnlineBots()
            guard current() else { return .stopped("the bot went offline") }
            if let outcome = await matchmakingPrecheck(fillMode: fillMode) {
                return outcome
            }
            guard current() else { return .stopped("the bot went offline") }
        }
        guard onlineBotsFetchedAt != nil else {
            // No fetch has succeeded and none is under way: the last one
            // failed (its alarm has the error) and the retry interval hasn't
            // passed.
            guard let requestedAt = onlineBotsRequestedAt else {
                return .failed("the online-bots list has never been fetched")
            }
            let retryAt = requestedAt.addingTimeInterval(LichessBotMatchmaking.onlineBotsRetryInterval)
            return .failed("the last online-bots fetch failed (see Alarms); next try at \(retryAt.formatted(date: .omitted, time: .standard))")
        }
        let matchmaking = settings.matchmaking
        let now = Date()
        var generator = SystemRandomNumberGenerator()
        let result = LichessBotMatchmaking.pick(from: onlineBots, settings: matchmaking, ourPerfs: account?.perfs, context: matchmakingContext(now: now), using: &generator)
        let pick: LichessBotMatchmaking.Pick
        switch result {
        case .noTimeControl:
            return .failed("no time control is chosen in Settings ▸ Matchmaking")
        case .noCandidate(let clock, let bounds, let listed, let exclusions):
            return .noCandidate("no bot fits for \(clock.rawValue): \(listed) listed, rating window \(bounds.description(speed: clock.speed)); \(LichessBotMatchmaking.describe(exclusions))")
        case .picked(let picked):
            pick = picked
        }
        let speed = pick.clock.speed
        let rating = pick.rating
        protocolLog.record(.challenge, "matchmaking pick: \(pick.bot.username) (\(speed.rawValue) \(rating)) at \(pick.clock.rawValue), uniformly from \(pick.candidateCount) candidate(s)\(pick.fromFavorites ? ", favorites first" : ""); rating window \(pick.bounds.description(speed: speed)); excluded: \(LichessBotMatchmaking.describe(pick.exclusions))")
        let request = pick.clock.challenge(rated: matchmaking.rated, color: .random)
        let outcome = await sendMatchmakingChallenge(
            to: pick.bot.username, request: request, origin: .matchmaking(fillMode: fillMode, opponent: pick.bot))
        switch outcome {
        case .sent:
            protocolLog.record(.challenge, "matchmaking sent a challenge to \(pick.bot.username)", fields: ["clock": request.clockText, "rated": "\(request.rated)", "window": pick.bounds.description(speed: speed)])
            SessionLogger.shared.log("[LICHESS-BOT] matchmaking challenge sent to \(pick.bot.username) (\(speed.rawValue) \(rating)) at \(request.clockText) \(request.rated ? "rated" : "casual"); rating window \(pick.bounds.description(speed: speed))")
        default:
            protocolLog.record(.challenge, "matchmaking send to \(pick.bot.username) \(outcome.text)")
        }
        return outcome
    }

    /// Send one matchmaking challenge and note the attempt in the rate
    /// limiter: the one send path for a pass's pick and for the automatic
    /// casual resend, so both count against the hourly cap and the spacing
    /// the same way.
    private func sendMatchmakingChallenge(to username: String, request: LichessBotOutgoingChallenge, origin: ChallengeOrigin) async -> MatchmakingPassOutcome {
        let attemptAt = Date()
        let outcome: MatchmakingPassOutcome
        let reachedLichess: Bool
        do {
            try await sendChallenge(to: username, request: request, origin: origin)
            outcome = .sent(username: username)
            reachedLichess = true
        } catch {
            let text = Self.safeDescription(error)
            // Errors raised before the challenge is posted (offline, a limit,
            // the gate) cost no challenge; everything else reached Lichess.
            reachedLichess = !(error is LichessBotControllerError) && !(error is LichessBotGateError) && !(error is CancellationError)
            if error is LichessBotGateError || error is CancellationError {
                outcome = .stopped(text)
            } else if case LichessBotAPIError.http(_, let message?) = error, let refusal = LichessBotBotLimitRefusal.parse(message) {
                // `sendChallenge` recorded the limit; say plainly what it is.
                outcome = .failed("\(username) is at Lichess's bot-vs-bot daily limit until \(refusal.until.formatted(date: .omitted, time: .shortened))")
            } else {
                outcome = .failed("\(username): \(text)")
            }
        }
        matchmakingRateLimiter.recordAttempt(at: attemptAt, reachedLichess: reachedLichess)
        return outcome
    }

    /// Whether a matchmaking send may happen now, and why not; nil when it
    /// may. Reads the gate's live phase.
    private func matchmakingPrecheck(fillMode: LichessBotMatchmakingSettings.FillMode) async -> MatchmakingPassOutcome? {
        if let outcome = await matchmakingSendConditionsBlock(fillMode: fillMode) {
            return outcome
        }
        let decision = matchmakingRateLimiter.decision(now: Date(), perHourCap: settings.matchmaking.maxChallengesPerHour, minimumSpacing: LichessBotMatchmaking.minimumSendSpacing)
        guard decision == .allowed else { return .rateLimited(decision) }
        return nil
    }

    /// The part of `matchmakingPrecheck` before the send rate: the pass
    /// conditions (Online, no 429 hold, the gate open, the challenge scope,
    /// no Play One Game, the queue empty, under Lichess's daily bot-game
    /// limit) and an open slot under `fillMode`. Nil when both hold. Reads
    /// the gate's live phase.
    private func matchmakingSendConditionsBlock(fillMode: LichessBotMatchmakingSettings.FillMode) async -> MatchmakingPassOutcome? {
        let phase = await gate.snapshot().phase
        let conditions = LichessBotMatchmaking.PassConditions(
            isOnline: runtime != nil && connection == .online,
            rateLimitHoldActive: rateLimitHoldUntil != nil,
            gateOpen: phase == .open,
            hasChallengeScope: hasChallengeScope,
            playOneGameActive: oneGameRequested,
            queueHasEntriesToSend: challengeQueue.hasEntriesToSend,
            botGamesInLastDay: botGamesInLastDay(now: Date())
        )
        if let reason = LichessBotMatchmaking.passBlockedReason(conditions) {
            return .blocked(reason: reason)
        }
        let slots = LichessBotMatchmaking.openSlots(
            maxConcurrentGames: settings.challenge.maxConcurrentGames,
            gamesReservedForHumans: settings.challenge.gamesReservedForHumans,
            fillMode: fillMode,
            committed: committedGameSlots
        )
        guard slots > 0 else { return .noOpenSlot }
        return nil
    }

    /// What the candidate rules consult, from the controller's state now.
    private func matchmakingContext(now: Date) -> LichessBotMatchmaking.CandidateContext {
        var engaged = challengeQueue.activeUserIDs.union(casualResendWaitingOpponentIDs)
        for pending in pendingChallenges {
            engaged.insert(pending.username.lowercased())
        }
        for game in gamesInProgress {
            if let opponentID = game.opponent?.id {
                engaged.insert(opponentID.lowercased())
            }
        }
        return LichessBotMatchmaking.CandidateContext(
            ourAccountID: accountID,
            blockedUserIDs: Set(settings.challenge.blockedUserIDs.map { $0.lowercased() }),
            engagedUserIDs: engaged,
            notes: playerNotesWithLiveBotLimits,
            gamesTodayByOpponent: gamesTodayByOpponent,
            maxGamesPerOpponentPerDay: settings.challenge.maxGamesPerOpponentPerDay,
            now: now
        )
    }

    /// No fetch is under way, the online-bots list is older than
    /// matchmaking's refresh interval (or missing), and no fetch was tried
    /// within the retry interval.
    private func onlineBotsNeedMatchmakingRefresh(now: Date) -> Bool {
        guard onlineBotsRefresh == nil else { return false }
        if let fetchedAt = onlineBotsFetchedAt, now.timeIntervalSince(fetchedAt) < LichessBotMatchmaking.onlineBotsRefreshInterval {
            return false
        }
        if let requestedAt = onlineBotsRequestedAt, now.timeIntervalSince(requestedAt) < LichessBotMatchmaking.onlineBotsRetryInterval {
            return false
        }
        return true
    }

    /// While matchmaking is on and the bot is Online, keep the online-bots
    /// list on matchmaking's own cadence (plan §7.3 B). The fetch is
    /// recorded as under way before this returns, so neither the next poll
    /// nor a pass started meanwhile starts a second one; the pass waits for
    /// it.
    private func refreshOnlineBotsForMatchmakingIfDue() {
        guard settings.matchmaking.enabled, runtime != nil, connection == .online,
              onlineBotsNeedMatchmakingRefresh(now: Date()) else { return }
        onlineBotsRefreshTask()
    }

    /// A player declined one of DCM's challenges: matchmaking leaves them
    /// alone for the configured cool-down (plan §7.3 B). Recorded for every
    /// decline, including a `casual` decline of a rated challenge — except
    /// one that matchmaking resends as casual on its own
    /// (`fallBackToCasual`): that resend's answer decides, and a resend that
    /// is not sent records it then.
    private func recordDeclineCooldown(_ username: String) {
        let hours = settings.matchmaking.declineCooldownHours
        guard hours > 0 else { return }
        let userID = username.lowercased()
        guard var notes = playerNotes else {
            protocolLog.record(.anomaly, "\(userID) declined, but player notes aren't loaded; no decline cool-down recorded")
            return
        }
        let until = Date().addingTimeInterval(TimeInterval(hours) * Self.secondsPerHour)
        notes.recordDeclineCooldown(userID, until: until)
        playerNotes = notes
        savePlayerNotes(notes)
        protocolLog.record(.challenge, "\(userID) declined; matchmaking leaves them alone until \(until.formatted(date: .abbreviated, time: .standard))")
    }

    // MARK: - Matchmaking's casual resend (fallBackToCasual)

    /// A rated matchmaking challenge declined with Lichess's `casual`
    /// reason, to be sent again unrated.
    private struct MatchmakingCasualResend {
        let username: String
        /// Lowercased.
        let opponentID: String
        /// The bot as the pass that picked it saw it, for the candidate rules.
        let opponent: LichessBotUserSummary
        /// The fill mode of the pass that picked it, for the slot rule.
        let fillMode: LichessBotMatchmakingSettings.FillMode
        /// The declined challenge with `rated` false: the same clock and the
        /// same color asked for.
        let request: LichessBotOutgoingChallenge
    }

    /// The automatic casual resend that declining `pending` with
    /// `reasonKey` calls for, or nil when it calls for none: the setting is
    /// off, the reason isn't exactly `casual`, the challenge wasn't rated,
    /// or matchmaking didn't pick it. The operator's own challenges keep the
    /// manual offer, and the resend itself (unrated, and of its own origin)
    /// never qualifies, so at most one resend follows a rated challenge.
    private func matchmakingCasualResend(for pending: PendingChallenge, reasonKey: String?) -> MatchmakingCasualResend? {
        guard settings.matchmaking.fallBackToCasual,
              reasonKey == LichessBotDeclineReason.casual.rawValue,
              pending.request.rated,
              case .matchmaking(let fillMode, let opponent) = pending.origin else { return nil }
        var request = pending.request
        request.rated = false
        return MatchmakingCasualResend(
            username: pending.username, opponentID: pending.username.lowercased(),
            opponent: opponent, fillMode: fillMode, request: request)
    }

    /// Start the casual resend. Called from the decline's handler, which
    /// records no cool-down for it: the resend's own answer decides that,
    /// and a resend that can't be sent records it (see
    /// `fallBackFromMatchmakingCasualResend`).
    private func startMatchmakingCasualResend(_ resend: MatchmakingCasualResend) {
        // Until the resend holds the pass flag, a pass already running could
        // pick this bot: it has no cool-down and nothing pending now.
        casualResendWaitingOpponentIDs.insert(resend.opponentID)
        protocolLog.record(.challenge, "matchmaking: \(resend.username) declined rated (casual); resending as casual", fields: ["clock": resend.request.clockText, "color": resend.request.color.rawValue])
        SessionLogger.shared.log("[LICHESS-BOT] matchmaking: \(resend.username) declined rated (casual); resending as casual")
        let generation = runtimeGeneration
        Task {
            await sendMatchmakingCasualResend(resend, generation: generation)
        }
    }

    /// Send the casual resend as a matchmaking send: it waits for a running
    /// pass, holds the pass flag (so no pass or Fill Open Slots sends
    /// meanwhile, and the hourly cap can't be overrun by two sends deciding
    /// at once), checks what a pass checks for this bot, and sends through
    /// `sendMatchmakingChallenge`, which notes the attempt in the rate
    /// limiter. Anything that keeps it from being sent falls back to what
    /// the decline gets with the setting off.
    private func sendMatchmakingCasualResend(_ resend: MatchmakingCasualResend, generation generationAtStart: Int) async {
        // Every await can outlive this runtime.
        func current() -> Bool { runtimeGeneration == generationAtStart }
        while matchmakingPassRunning, current() {
            do {
                try await Task.sleep(for: .milliseconds(250))
            } catch {
                fallBackFromMatchmakingCasualResend(resend, reason: "cancelled while waiting for a matchmaking pass to finish", runtimeIsCurrent: current())
                return
            }
        }
        guard current() else {
            fallBackFromMatchmakingCasualResend(resend, reason: "the bot went offline", runtimeIsCurrent: false)
            return
        }
        casualResendWaitingOpponentIDs.remove(resend.opponentID)
        matchmakingPassRunning = true
        defer {
            // Teardown clears the flag itself; a stale resend must not clear
            // the flag a newer runtime's pass set.
            if current() { matchmakingPassRunning = false }
        }
        if let reason = await matchmakingCasualResendBlockedReason(resend) {
            fallBackFromMatchmakingCasualResend(resend, reason: reason, runtimeIsCurrent: current())
            return
        }
        guard current() else {
            fallBackFromMatchmakingCasualResend(resend, reason: "the bot went offline", runtimeIsCurrent: false)
            return
        }
        let outcome = await sendMatchmakingChallenge(to: resend.username, request: resend.request, origin: .matchmakingCasualResend)
        guard case .sent = outcome else {
            fallBackFromMatchmakingCasualResend(resend, reason: outcome.text, runtimeIsCurrent: current())
            return
        }
        lastChallengeOutcome = "\(resend.username): declined rated (casual); resent as casual"
        protocolLog.record(.challenge, "matchmaking resent a challenge to \(resend.username) as casual", fields: ["clock": resend.request.clockText, "rated": "\(resend.request.rated)", "color": resend.request.color.rawValue])
        SessionLogger.shared.log("[LICHESS-BOT] matchmaking challenge resent to \(resend.username) at \(resend.request.clockText) casual, as they asked")
    }

    /// Why the casual resend may not be sent now, or nil when it may: every
    /// check a pass makes before its send, applied to this bot — the pass
    /// conditions (Online, no 429 hold, the gate open, the challenge scope,
    /// no Play One Game, the queue empty, Lichess's daily bot-game limit),
    /// an open slot under the picking pass's fill mode, the hourly cap, and
    /// the candidate rules (the per-opponent daily limit, blocks, the bot
    /// limit, already engaged, the rating window) — except the spacing
    /// between sends. The spacing paces how fast matchmaking fills free
    /// slots and keeps a run of failures from looping quickly; this resend
    /// answers the bot's own request, once per rated challenge, so waiting
    /// out the spacing would only delay it. It still counts toward the
    /// spacing of the next pass, and toward the hourly cap.
    private func matchmakingCasualResendBlockedReason(_ resend: MatchmakingCasualResend) async -> String? {
        if let outcome = await matchmakingSendConditionsBlock(fillMode: resend.fillMode) {
            return outcome.text
        }
        let now = Date()
        let decision = matchmakingRateLimiter.decision(now: now, perHourCap: settings.matchmaking.maxChallengesPerHour, minimumSpacing: LichessBotMatchmaking.minimumSendSpacing)
        switch decision {
        case .hourlyCap:
            return MatchmakingPassOutcome.rateLimited(decision).text
        case .spacing, .allowed:
            break
        }
        let speed = LichessBotSpeed.forClock(limitSeconds: resend.request.clockLimitSeconds, incrementSeconds: resend.request.clockIncrementSeconds)
        let bounds = LichessBotMatchmaking.ratingBounds(settings: settings.matchmaking, ourPerfs: account?.perfs, speed: speed)
        if let exclusion = LichessBotMatchmaking.exclusion(of: resend.opponent, speed: speed, bounds: bounds, context: matchmakingContext(now: now)) {
            return "\(resend.username) is no longer a candidate: \(exclusion.rawValue)"
        }
        return nil
    }

    /// The casual resend was not sent: answer the decline the way it is
    /// answered with the setting off — the decline cool-down and the manual
    /// Resend as Casual offer — and say why in the outcome line and both
    /// logs. Applied even when the runtime has gone, as the decline's own
    /// handler would have applied it.
    private func fallBackFromMatchmakingCasualResend(_ resend: MatchmakingCasualResend, reason: String, runtimeIsCurrent: Bool) {
        if runtimeIsCurrent {
            // Otherwise teardown already emptied the set for the next runtime.
            casualResendWaitingOpponentIDs.remove(resend.opponentID)
        }
        protocolLog.record(.challenge, "matchmaking: casual resend to \(resend.username) not sent: \(reason)")
        SessionLogger.shared.log("[LICHESS-BOT] matchmaking: casual resend to \(resend.username) not sent: \(reason); recording the decline cool-down and offering Resend as Casual")
        recordDeclineCooldown(resend.username)
        casualResendOffer = CasualResendOffer(username: resend.username, request: resend.request)
        lastChallengeOutcome = "\(resend.username): resending as casual failed: \(reason)"
    }

    // MARK: - Grid housekeeping

    func dismissFinishedGames() {
        games.removeAll { $0.isFinished && !activeGameIDs.contains($0.id) }
        if let focusedGameID, !games.contains(where: { $0.id == focusedGameID }) {
            self.focusedGameID = nil
        }
        updateAutoFollow()
    }

    func dismissGame(_ gameID: String) {
        guard !activeGameIDs.contains(gameID) else { return }
        games.removeAll { $0.id == gameID }
        updateAutoFollow()
    }

    func dismissAlarms() {
        alarms.removeAll()
        droppedAlarmCount = 0
    }

    // MARK: - Runtime construction

    private func startRuntime(oneGame: Bool) async throws {
        let settings = self.settings
        let accountID = self.accountID
        guard let token = try await readToken() else {
            throw LichessBotControllerError.noToken
        }
        let directory = dataDirectory
        // On the journal queue, where the previous runtime's release runs,
        // so going straight back online never finds our own lock still held.
        let lock = try await journalQueue.run {
            try directory.createDirectories()
            return try LichessBotInstanceLock.acquire(at: directory.lockURL, holder: .current)
        }
        do {
            try await startRuntime(oneGame: oneGame, settings: settings, accountID: accountID, token: token, lock: lock)
        } catch {
            gateEventSink.value = nil
            journalQueue.enqueue("release the instance lock after a failed start") { lock.release() }
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
            transport: services.makeTransport(),
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
        // Going online resumes (or files) whatever the last run left.
        leftoverGamesFromLastRun = []

        let settingsBox = SyncBox(settings)
        self.settingsBox = settingsBox
        let settingsProvider: @Sendable () async -> LichessBotSettings = { settingsBox.value }
        let slots = LichessBotModelSlots(provider: modelProvider, time: time) { line in
            SessionLogger.shared.log(line)
        }
        let recordStore = LichessBotRecordStore(directory: dataDirectory, journalQueue: journalQueue, indexQueue: fileQueue, ourAccountID: accountID)
        let journal = LichessBotJournalWriter(
            directory: dataDirectory,
            fileQueue: journalQueue,
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
            onEvent: { event in continuation.yield(.manager(event)) },
            pacingProvider: { [weak self] gameID in
                // No controller means the app is quitting: no pacing.
                await self?.pacingSnapshot(for: gameID) ?? LichessBotMovePacingSnapshot()
            },
            journalReader: { gameID in
                try await recordStore.resumedJournal(gameID: gameID)
            }
        )
        let reconciler = LichessBotReconciler(
            api: client,
            store: recordStore,
            time: time,
            settingsProvider: settingsProvider,
            // `self` is held strongly: a weak reference would need a made-up
            // answer for a missing controller. Teardown breaks the cycle — it
            // drops the runtime and cancels the reconciler's task, which
            // releases this closure.
            hasLiveFilingOwner: { [self] gameID in
                if await manager.hasSession(forGameID: gameID) {
                    return true
                }
                return await isAwaitingPostGameChat(gameID)
            },
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
    /// Also tells the manager which games have a journal left from before
    /// this runtime: their `gameStart`s (replayed by Lichess on connect)
    /// resume them rather than start them.
    private func seedDailyCounts(manager: LichessBotSessionManager, store: LichessBotRecordStore) async {
        let leftovers: [String: Date]
        do {
            leftovers = try await store.inProgressJournalCreationDates()
        } catch {
            raiseAlarm("Listing the journals left from before going online failed: \(Self.safeDescription(error)); games still live on Lichess show as new games and daily limits count from now")
            return
        }
        await manager.setResumableGames(leftovers)
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
            let todaysLeftovers = leftovers.filter { Calendar.current.isDateInToday($0.value) }.map(\.key)
            let earlierLeftovers = Set(leftovers.filter { !Calendar.current.isDateInToday($0.value) }.map(\.key))
            await manager.seedDailyCounts(gameIDs: today.map(\.gameID) + todaysLeftovers, opponentByGame: opponents, earlierDayGameIDs: earlierLeftovers)
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
        for pending in pendingChallenges {
            withdraw(challengeID: pending.id, client: runtime.client)
            resolveChallengeOutcome(challengeID: pending.id, .canceled)
        }
        pendingChallenges = []
        clearChallengeQueue(reason: "offline: \(reason)")
        // Work tied to the old runtime checks its generation and leaves
        // these alone; they are reset here for the next runtime.
        challengeQueuePumpRunning = false
        matchmakingPassRunning = false
        casualResendWaitingOpponentIDs = []
        isFillingOpenSlots = false
        nextAutomaticMatchmakingPassAt = .distantPast
        lastLoggedAutomaticMatchmakingOutcome = nil
        matchmakingStatus = nil
        oneGameRequested = false
        drainRequested = false
        acceptedAwaitingStartIDs = []
        gamesTodayByOpponent = [:]
        if !gamesAwaitingPostGameChat.isEmpty {
            protocolLog.record(.game, "offline with \(gamesAwaitingPostGameChat.count) finished game(s) not yet filed; recovery files them when the bot next goes online", fields: ["games": gamesAwaitingPostGameChat.sorted().joined(separator: ",")])
        }
        // The ids belong to this runtime's reconciler; a later drain must not
        // enqueue games that recovery has since filed.
        gamesAwaitingPostGameChat = []
        gamesFiledBeforeTheirChatFetches = []
        // Likewise the reconciler's view of what is unreconciled: the next
        // runtime's recovery derives it afresh from the journals left.
        unreconciledGameIDs = []
        isFilingRecords = false
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
        // On the journal queue, so the lock is held until every journal
        // append this runtime already queued has run. A queue already closed
        // by `shutdown` refuses it; the lock's deinit releases it then.
        journalQueue.enqueue("release the instance lock (\(reason))") { lock.release() }
        for game in games where activeGameIDs.contains(game.id) {
            game.markSessionEnded(reason)
        }
        activeGameIDs.removeAll()
        updateAutoFollow()
        gateSnapshot = nil
        generation = nil
    }

    private func pollLoop(slots: LichessBotModelSlots, manager: LichessBotSessionManager, generation runtimeGenerationAtStart: Int) async {
        var nextModelRefreshAt = Date.distantPast
        var consecutiveModelRefreshFailures = 0
        var modelSettingsAtLastFailure: LichessBotModelSettings?
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
            let todays = await manager.todaysGamesByOpponent()
            guard current() else { return }
            if todays != gamesTodayByOpponent {
                gamesTodayByOpponent = todays
            }
            if let until = rateLimitHoldUntil, Date() >= until {
                await endRateLimitHold()
                guard current() else { return }
            }
            let timeout = settings.challenge.outgoingChallengeTimeoutSeconds
            if timeout > 0 {
                let now = Date()
                let expired = pendingChallenges.filter {
                    !autoWithdrawAttemptedChallengeIDs.contains($0.id) && now.timeIntervalSince($0.sentAt) >= TimeInterval(timeout)
                }
                for pending in expired {
                    autoWithdrawAttemptedChallengeIDs.insert(pending.id)
                    protocolLog.record(.challenge, "withdrawing unanswered challenge to \(pending.username) after \(timeout) s")
                    await cancelChallenge(id: pending.id)
                    guard current() else { return }
                }
            }
            // A settings change is a new attempt, not a continued failure.
            if let failed = modelSettingsAtLastFailure, failed != settings.model {
                consecutiveModelRefreshFailures = 0
                modelSettingsAtLastFailure = nil
                nextModelRefreshAt = .distantPast
            }
            if Date() >= nextModelRefreshAt {
                do {
                    try await slots.refreshIfDue(for: settings.model)
                    guard current() else { return }
                    consecutiveModelRefreshFailures = 0
                    modelSettingsAtLastFailure = nil
                    nextModelRefreshAt = Date().addingTimeInterval(Self.modelRefreshInterval)
                } catch {
                    guard current() else { return }
                    // Each failure rebuilds a network; back off so a missing
                    // or broken source isn't retried (and alarmed) every poll.
                    let delay = LichessBotBackoff.seconds(Self.modelRefreshBackoff.delay(attempt: consecutiveModelRefreshFailures, unitRandom: Double.random(in: 0...1)))
                    consecutiveModelRefreshFailures += 1
                    modelSettingsAtLastFailure = settings.model
                    nextModelRefreshAt = Date().addingTimeInterval(delay)
                    protocolLog.record(.anomaly, String(format: "model refresh retry in %.0f s", delay))
                    raiseAlarm("Model refresh failed: \(Self.safeDescription(error))")
                }
            }
            pruneFinishedGames()
            // Each starts its own task when due, so a send or a fetch never
            // holds up this loop.
            scheduleChallengeQueuePump()
            startAutomaticMatchmakingPassIfDue()
            refreshOnlineBotsForMatchmakingIfDue()
            do {
                try await Task.sleep(for: .seconds(1))
            } catch {
                return
            }
        }
    }

    /// How often a working model source is checked for a newer generation.
    private static let modelRefreshInterval: TimeInterval = 15
    /// Retry spacing after a failed model refresh.
    private static let modelRefreshBackoff = LichessBotBackoff(initial: .seconds(30), multiplier: 2, cap: .seconds(900))

    private func recoverLeftoverJournals(store: LichessBotRecordStore, reconciler: LichessBotReconciler) async {
        do {
            // Every leftover goes to the reconciler, which alone decides:
            // it leaves a game to its live owner (a session playing or
            // starting it, or a post-game chat wait) and recognizes one
            // already filed.
            let leftovers = try await store.inProgressGameIDs()
            for gameID in leftovers {
                await reconciler.enqueue(gameID: gameID)
            }
            if !leftovers.isEmpty {
                protocolLog.record(.game, "launch recovery: \(leftovers.count) leftover journal(s) handed to the reconciler", fields: ["games": leftovers.joined(separator: ",")])
            }
        } catch {
            raiseAlarm("Launch recovery could not list leftover journals: \(error.localizedDescription)")
        }
    }

    /// Load the finished-games index (for head-to-head records). Works
    /// offline too: the records are on disk.
    func refreshIndex() async {
        let store = runtime?.recordStore
            ?? LichessBotRecordStore(directory: dataDirectory, journalQueue: journalQueue, indexQueue: fileQueue, ourAccountID: accountID)
        do {
            index = try await store.loadIndex()
        } catch {
            raiseAlarm("Loading the games index failed: \(error.localizedDescription)")
        }
    }

    private func pruneFinishedGames() {
        let retention = TimeInterval(settings.display.finishedGameRetentionMinutes * 60)
        let now = Date()
        // The followed game stays while its hold keeps it on screen.
        let heldGameID = autoFollowHold?.gameID
        let countBefore = games.count
        games.removeAll { game in
            guard let finishedAt = game.finishedAt, !activeGameIDs.contains(game.id) else { return false }
            return now.timeIntervalSince(finishedAt) >= retention && game.id != focusedGameID && game.id != heldGameID
        }
        if games.count != countBefore {
            updateAutoFollow()
        }
    }

    // MARK: - Event handling (main actor)

    private func handle(_ event: LichessBotControllerEvent) {
        switch event {
        case .game(let gameID, let gameEvent):
            // Only a session start lists a game: a late event for a game that
            // was dismissed or pruned must not bring it back (the journal
            // records every event regardless).
            listedGame(gameID)?.apply(gameEvent)
            // A final state may have just finished the followed game.
            updateAutoFollow()
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
                listedGame(gameID)?.applyRequest(record)
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
                if drainRequested || finishing != nil {
                    // Going offline or quitting: file once the last
                    // request (the goodbye) has landed.
                    gamesFiledBeforeTheirChatFetches.insert(gameID)
                    Task {
                        await reconciler.enqueue(gameID: gameID, after: .seconds(5))
                    }
                } else {
                    // File after the last post-game chat fetch, so the
                    // record includes everything said after the game.
                    gamesAwaitingPostGameChat.insert(gameID)
                }
            }
            schedulePostGameChatFetches(gameID)
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
        case .challengeArrived(let challengeID, let challengerID, let challengerTitle):
            // Settings are read here, on arrival, so a changed tone applies
            // to the very next challenge.
            let soundName = LichessBotChallengeAlert.soundName(
                challengerID: challengerID,
                challengerTitle: challengerTitle,
                ourAccountID: accountID,
                alerts: settings.alerts
            )
            if let soundName {
                do {
                    try LichessBotSystemSounds.play(named: soundName)
                } catch {
                    protocolLog.record(.anomaly, "challenge alert failed: \(error.localizedDescription)", fields: ["challenge": challengeID])
                }
            }
        case .challengeDecision(let challengeID, let challengerID, let decision):
            protocolLog.record(.challenge, "\(challengerID): \(Self.describe(decision))", fields: ["challenge": challengeID])
            if decision == .accept {
                // Fetch before the game starts, so it never competes
                // with a move (plan §14.3b).
                loadOpponentProfile(challengerID)
            }
        case .challengeResponseFailed(let challengeID, let error):
            protocolLog.record(.anomaly, "challenge response failed: \(error)", fields: ["challenge": challengeID])
        case .gameSessionStarted(let gameID, let generation, let origin):
            if let pending = pendingChallenges.first(where: { $0.id == gameID }) {
                // The accepted challenge's game can start before the
                // manager was told about the challenge.
                lastChallengeOutcome = "\(pending.username): accepted"
                pendingChallenges.removeAll { $0.id == gameID }
                resolveChallengeOutcome(challengeID: gameID, .accepted)
                if let manager = runtime?.manager {
                    Task { await manager.clearOutgoingChallenge(id: gameID) }
                }
            }
            activeGameIDs.insert(gameID)
            // Counted from here as a game in progress, not as starting.
            acceptedAwaitingStartIDs.remove(gameID)
            listStartedGame(gameID, origin: origin)
            updateAutoFollow()
            self.generation = generation
            let modelFields = ["model": generation.modelID, "generation": "\(generation.generationID)"]
            let model = "\(generation.sourceKind.rawValue) \(generation.modelID)"
            switch origin {
            case .new:
                protocolLog.record(.game, "game started", gameID: gameID, fields: modelFields)
                SessionLogger.shared.log("[LICHESS-BOT] game \(gameID) started with \(model)")
            case .resumed(let journal):
                let since = journal.firstJournaledAt.formatted(date: .abbreviated, time: .standard)
                protocolLog.record(.game, "game resumed", gameID: gameID, fields: modelFields.merging(["journal_since": since, "journal_entries": "\(journal.items.count)"]) { current, _ in current })
                SessionLogger.shared.log("[LICHESS-BOT] game \(gameID) resumed (journal since \(since), \(journal.items.count) entries) with \(model)")
            case .resumedWithUnreadableJournal(let journalStartedAt, let reason):
                let since = journalStartedAt.formatted(date: .abbreviated, time: .standard)
                protocolLog.record(.game, "game resumed", gameID: gameID, fields: modelFields.merging(["journal_since": since, "history": "unavailable: \(reason)"]) { current, _ in current })
                SessionLogger.shared.log("[LICHESS-BOT] game \(gameID) resumed (journal since \(since)) with \(model)")
                raiseAlarm("Game \(gameID) was resumed without its history: \(reason). Takebacks, command replies, greeting and goodbye are off for it, since there is no telling what was already done.")
            }
        case .gameSessionEnded(let gameID):
            activeGameIDs.remove(gameID)
            if let game = games.first(where: { $0.id == gameID }) {
                game.markSessionEnded("the game session ended")
            }
            protocolLog.record(.game, "game session ended", gameID: gameID)
            updateAutoFollow()
            finishIfDrained()
            scheduleChallengeQueuePump()
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
                resolveChallengeOutcome(challengeID: challengeID, .accepted)
                // The game holds its slot from here, though its session is
                // still being set up; without this, the slot would look free
                // until the next poll mirrors the manager's starting games.
                if !activeGameIDs.contains(gameID) {
                    acceptedAwaitingStartIDs.insert(gameID)
                }
            case .declined(let reason, let reasonKey):
                text = "declined" + (reason.map { ": \($0)" } ?? "")
                resolveChallengeOutcome(challengeID: challengeID, .declined(LichessBotDeclineReasonRecord(reasonKey: reasonKey)))
                if let pending = pendingChallenges.first(where: { $0.id == challengeID }),
                   let resend = matchmakingCasualResend(for: pending, reasonKey: reasonKey) {
                    // No cool-down and no manual offer yet: the resend's
                    // outcome decides both.
                    startMatchmakingCasualResend(resend)
                } else {
                    if let pending = pendingChallenges.first(where: { $0.id == challengeID }) {
                        recordDeclineCooldown(pending.username)
                    }
                    if reasonKey == LichessBotDeclineReason.casual.rawValue,
                       let pending = pendingChallenges.first(where: { $0.id == challengeID }),
                       pending.request.rated {
                        var casual = pending.request
                        casual.rated = false
                        casualResendOffer = CasualResendOffer(username: pending.username, request: casual)
                    }
                }
            case .canceled:
                text = "canceled"
                resolveChallengeOutcome(challengeID: challengeID, .canceled)
            }
            if let pending = pendingChallenges.first(where: { $0.id == challengeID }) {
                lastChallengeOutcome = "\(pending.username): \(text)"
            } else {
                lastChallengeOutcome = "challenge \(challengeID): \(text)"
            }
            pendingChallenges.removeAll { $0.id == challengeID }
            protocolLog.record(.challenge, "outgoing challenge \(text)", fields: ["challenge": challengeID])
            scheduleChallengeQueuePump()
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
        case .closed(let reason):
            // Any other close (a failed cooldown timer) stops the bot just as
            // visibly. After a breaker trip the runtime is already gone and
            // this does nothing.
            failRuntime("request gate closed: \(reason)")
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

    /// Refresh the Overview's account (ratings, game counts) once a game is
    /// filed. Housekeeping priority, so the gate never starts it while any
    /// game awaits our move; a failure is logged and the previous values
    /// stay on screen.
    private func refreshAccountAfterGame() async {
        do {
            let client = try await accountClient()
            account = try await client.account()
        } catch {
            protocolLog.record(.account, "account refresh after a game failed: \(Self.safeDescription(error))")
        }
    }

    private func handle(_ event: LichessBotReconcilerEvent) {
        switch event {
        case .finalized(let finalized):
            runtime?.journal.markFinalized(gameID: finalized.record.gameID)
            unreconciledGameIDs.removeAll { $0 == finalized.record.gameID }
            if let failure = finalized.indexUpdateFailure {
                raiseAlarm("Game \(finalized.record.gameID) is filed, but updating the games index failed: \(failure). The index is rebuilt on its next load.")
            }
            SessionLogger.shared.log(LichessBotRecordStore.summaryLine(finalized.record))
            protocolLog.record(.game, "finalized (\(finalized.record.reconciliation.outcome.rawValue))", gameID: finalized.record.gameID, fields: ["mismatches": finalized.record.reconciliation.mismatches.joined(separator: "; ")])
            Task {
                await refreshIndex()
                await refreshAccountAfterGame()
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
        case .quarantined(let gameID, let reason):
            if !unreconciledGameIDs.contains(gameID) {
                unreconciledGameIDs.append(gameID)
            }
            raiseAlarm("Game \(gameID) can't be filed and won't be retried until the bot next goes online: \(reason).")
        case .alreadyFiled(let gameID):
            runtime?.journal.markFinalized(gameID: gameID)
            unreconciledGameIDs.removeAll { $0 == gameID }
            protocolLog.record(.game, "already filed; nothing to do", gameID: gameID)
        case .nothingToFile(let gameID, let reason):
            unreconciledGameIDs.removeAll { $0 == gameID }
            raiseAlarm("Game \(gameID) was handed to filing, but \(reason); nothing was filed.")
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
        let generationAtStart = runtimeGeneration
        // Every await can outlive this runtime; nothing is written after it.
        func current() -> Bool { runtimeGeneration == generationAtStart }
        defer {
            // Teardown clears the flag itself; a stale filing must not clear
            // the flag a newer runtime's drain set.
            if current() { isFilingRecords = false }
        }
        let started = Date()
        while current(), let reconciler = runtime?.reconciler, Date().timeIntervalSince(started) < Self.filingTimeLimitSeconds {
            // The finished game's enqueue happens on its own task: allow a
            // moment for it to land before trusting an empty queue.
            let due = await reconciler.dueGameIDs
            guard current() else { return }
            if due.isEmpty && Date().timeIntervalSince(started) >= Self.filingSettleSeconds {
                break
            }
            do {
                try await Task.sleep(for: .seconds(1))
            } catch {
                return
            }
        }
        guard current(), runtime != nil, connection == .draining, !hasGamesInPlay else { return }
        if let reconciler = runtime?.reconciler {
            let unfiled = await reconciler.queuedGameIDs
            guard current() else { return }
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
    /// How long an empty filing queue must wait before it is trusted, since
    /// a finished game's enqueue lands on its own task.
    private static let filingSettleSeconds: TimeInterval = 3

    /// A game's operator move pacing (plan §14.3c). A game not in the list
    /// has no controls on screen, so no pacing.
    private func pacingSnapshot(for gameID: String) -> LichessBotMovePacingSnapshot {
        games.first { $0.id == gameID }?.pacingSnapshot ?? LichessBotMovePacingSnapshot()
    }

    /// Send the operator's own chat message in a live game (plan §14.3c).
    /// It is labeled "operator chat" in the transcript and journal, apart
    /// from DCM's automatic messages.
    func sendOperatorChat(gameID: String, room: LichessBotChatRoom, text: String) async throws {
        guard let runtime else {
            throw LichessBotControllerError.notOnline
        }
        let message = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if let problem = LichessBotOperatorChat.problem(with: message) {
            throw LichessBotControllerError.chatNotSendable(problem)
        }
        try await runtime.client.chat(gameID: gameID, room: room, text: message, label: "operator chat")
        let sent = LichessBotGameEvent.chatSent(room: room, text: message, origin: .operator)
        listedGame(gameID)?.apply(sent)
        await runtime.journal.gameEvent(gameID: gameID, sent)
    }

    /// The listed game with this id, if any.
    private func listedGame(_ gameID: String) -> LichessBotLiveGame? {
        games.first { $0.id == gameID }
    }

    /// List a game whose session just started: a new game, or one listed
    /// from before going offline, which is followed again. The same object
    /// is kept wherever it still exists — listed, or dismissed or pruned
    /// but still held by a pop-out window — so that window keeps updating.
    /// A resumed game keeps the time its journal was started; the list
    /// stays in start order (oldest first), since resumed games can come
    /// back in any order.
    private func listStartedGame(_ gameID: String, origin: LichessBotSessionOrigin) {
        if let existing = listedGame(gameID) {
            existing.resumeFollowing()
            return
        }
        let game: LichessBotLiveGame
        if let retained = retainedLiveGames[gameID]?.game {
            // It already holds the game's history from this launch.
            game = retained
            game.resumeFollowing()
        } else {
            switch origin {
            case .new:
                game = LichessBotLiveGame(id: gameID, startedAt: Date(), ourAccountID: accountID)
            case .resumed(let journal):
                game = LichessBotLiveGame(id: gameID, startedAt: journal.firstJournaledAt, ourAccountID: accountID)
                // Before it is listed, so no view redraws entry by entry.
                game.replay(journal)
            case .resumedWithUnreadableJournal(let journalStartedAt, let reason):
                game = LichessBotLiveGame(id: gameID, startedAt: journalStartedAt, ourAccountID: accountID)
                game.apply(.anomaly("resumed without its earlier history: \(reason)"))
            }
            retainedLiveGames = retainedLiveGames.filter { $0.value.game != nil }
            retainedLiveGames[gameID] = WeakLiveGame(game: game)
        }
        let position = games.firstIndex { $0.startedAt > game.startedAt } ?? games.endIndex
        games.insert(game, at: position)
    }

    /// A live game, held weakly: a game window (or the list) keeps it alive.
    private struct WeakLiveGame {
        weak var game: LichessBotLiveGame?
    }

    /// Every live game this launch created, held weakly, by id: a game
    /// dismissed or pruned while a pop-out window still shows it is listed
    /// again as that same object when its session resumes.
    @ObservationIgnored private var retainedLiveGames: [String: WeakLiveGame] = [:]

    private func raiseAlarm(_ rawText: String) {
        let text = LichessBotRedaction.redact(rawText)
        let now = Date()
        if let index = alarms.lastIndex(where: { $0.text == text }) {
            var repeated = alarms.remove(at: index)
            repeated.lastAt = now
            repeated.repeatCount += 1
            // Last-raised order, so the card's newest-first list shows it on top.
            alarms.append(repeated)
        } else {
            alarms.append(Alarm(id: nextAlarmID, firstAt: now, lastAt: now, text: text, repeatCount: 1))
            nextAlarmID += 1
            if alarms.count > Self.alarmLimit {
                droppedAlarmCount += alarms.count - Self.alarmLimit
                alarms.removeFirst(alarms.count - Self.alarmLimit)
            }
        }
        // Every raise is logged, repeats included.
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
    case concurrentGameLimit(limit: Int, committed: Int)
    case perOpponentGameLimit(username: String, limit: Int, committed: Int)
    case noSuchPlayer(String)
    case opponentOffline(String)
    case missingChallengeScope
    case chatNotSendable(String)

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
            return "The bot must be online to do that"
        case .concurrentGameLimit(let limit, let committed):
            return "Games in progress, accepted, and challenges waiting already total \(committed); the concurrent-game limit is \(limit)"
        case .perOpponentGameLimit(let username, let limit, let committed):
            return "Games in progress, accepted, and challenges waiting with \(username) already total \(committed); the per-opponent limit is \(limit)"
        case .missingChallengeScope:
            return "The token lacks the challenge:write scope needed to send challenges"
        case .noSuchPlayer(let username):
            return "No Lichess player is named “\(username)”"
        case .opponentOffline(let username):
            return "\(username) is offline. Lichess would keep the challenge open for hours, so it isn't sent."
        case .chatNotSendable(let problem):
            return "Chat not sent: \(problem)"
        }
    }
}
