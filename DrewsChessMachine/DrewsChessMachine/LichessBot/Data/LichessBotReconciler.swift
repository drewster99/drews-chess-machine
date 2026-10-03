import Foundation

/// The export call the reconciler makes. `LichessBotAPIClient` is the real
/// implementation.
protocol LichessBotExportAPI: Sendable {
    func exportGame(gameID: String) async throws -> Data
}

extension LichessBotAPIClient: LichessBotExportAPI {}

enum LichessBotReconcilerEvent: Sendable {
    case finalized(LichessBotFinalizedGame)
    /// The export isn't usable yet; another attempt is scheduled.
    case waiting(gameID: String, reason: String, retryIn: Duration)
    /// The bounded retry window ran out. The journal stays in `InProgress/`
    /// and the game shows as unreconciled; attempts continue at a slow
    /// cadence (plan §10.2, E23).
    case unreconciled(gameID: String, reason: String)
    /// Writing the record failed.
    case finalizeFailed(gameID: String, error: String)
    /// The game can't be filed as things stand: its journal or export can't
    /// make a record, a record filed for it doesn't decode, or Lichess has no
    /// export for a game the journal doesn't show as aborted. Attempts stop.
    /// A journal still in `InProgress/` stays there, so the recovery when the
    /// bot next goes online tries once more (an update may have fixed the
    /// cause).
    case quarantined(gameID: String, reason: String)
    /// The game was already filed (by another path, or before this enqueue
    /// landed): nothing was done, and nothing needs doing.
    case alreadyFiled(gameID: String)
    /// The game has no journal in `InProgress/` and no record: there is
    /// nothing to file and nothing any later attempt could file. Dropped
    /// from the queue.
    case nothingToFile(gameID: String, reason: String)
    case stopped(reason: String)
}

/// Turns finished games into records by reconciling each journal against
/// Lichess's export (plan §10.2).
///
/// Exports are housekeeping traffic: one at a time, at least
/// `exportMinimumSpacingSeconds` apart, and through the request gate like
/// every other call. Right after a game ends the export can still say
/// `started` (E23), so a live export is retried with backoff for a bounded
/// window before the game is marked unreconciled and retried slowly from
/// then on. A failure no retry can fix (a journal or export that can't make
/// a record) quarantines the game instead: one alarm, and no more attempts
/// until the bot next goes online.
///
/// **One decider.** The reconciler is the only place that decides whether a
/// game is filed now. A game with a live filing owner — a session playing
/// it or still starting, or the controller still collecting its post-game
/// chat — is dropped from the queue, and is enqueued again when that owner
/// is done: a session that saw its game finish hands it over through the
/// journal's finish (the controller files it after the post-game chat
/// fetches, or at once in a drain), and a session that ended without seeing
/// a finish (its game stream answered 404) is enqueued by the controller
/// when the manager reports the session's end. The owner is checked before the export and again just before
/// filing, since the export round trip can outlast a session starting or a
/// finish being recorded. Filing is idempotent: before any export, the files
/// say whether the game still has a journal to file; an already-filed game
/// is reported and dropped, so a game handed over twice (launch recovery
/// listing it just before another path filed it) is never exported again or
/// quarantined for a journal that is already filed.
///
/// Launch recovery hands every leftover `InProgress/` journal over: a game
/// still live is left to its session (or comes back as a live export), while
/// a game that ended while the app was down is finalized from its journal
/// plus the export.
actor LichessBotReconciler {
    private enum ItemState {
        /// Inside its retry window.
        case retrying
        /// Past its retry window; retried on a slow cadence.
        case unreconciled
        /// Can't succeed as things stand; not retried until the bot next goes
        /// online.
        case quarantined
    }

    private struct Item {
        let gameID: String
        let firstAttemptAt: Duration
        var attempts: Int
        var dueAt: Duration
        var state: ItemState
    }

    /// What a retry does once the live-export window is over.
    private enum WindowExpiry {
        case markUnreconciled
        case quarantine
    }

    private let api: any LichessBotExportAPI
    private let store: LichessBotRecordStore
    private let time: any LichessBotTimeSource
    private let settingsProvider: @Sendable () async -> LichessBotSettings
    /// Whether something else still owns the game's filing: a session
    /// playing it (or starting one), or the controller collecting its
    /// post-game chat. The game is enqueued again when that owner is done
    /// (see "One decider" above).
    private let hasLiveFilingOwner: @Sendable (String) async -> Bool
    private let onEvent: @Sendable (LichessBotReconcilerEvent) -> Void

    private var items: [String: Item] = [:]
    private var lastFetchAt: Duration?

    /// How long a still-live export is retried before the game is marked
    /// unreconciled.
    static let liveExportRetryWindow: Duration = .seconds(600)
    static let unreconciledRetryInterval: Duration = .seconds(1800)
    static let retryBackoff = LichessBotBackoff(initial: .seconds(5), multiplier: 2, cap: .seconds(60))
    private static let pollInterval: Duration = .seconds(1)

    init(
        api: any LichessBotExportAPI,
        store: LichessBotRecordStore,
        time: any LichessBotTimeSource,
        settingsProvider: @escaping @Sendable () async -> LichessBotSettings,
        hasLiveFilingOwner: @escaping @Sendable (String) async -> Bool,
        onEvent: @escaping @Sendable (LichessBotReconcilerEvent) -> Void
    ) {
        self.api = api
        self.store = store
        self.time = time
        self.settingsProvider = settingsProvider
        self.hasLiveFilingOwner = hasLiveFilingOwner
        self.onEvent = onEvent
    }

    /// Queue a game for reconciliation, `delay` from now. Enqueueing a game
    /// already queued (say, one left unreconciled at launch whose session
    /// has now ended) starts it over with a fresh retry window, due no
    /// later than it already was; a quarantined game is tried once more.
    func enqueue(gameID: String, after delay: Duration = .zero) {
        let due = time.now() + delay
        let dueAt: Duration
        if let existing = items[gameID], existing.state != .quarantined {
            dueAt = min(existing.dueAt, due)
        } else {
            dueAt = due
        }
        items[gameID] = Item(gameID: gameID, firstAttemptAt: dueAt, attempts: 0, dueAt: dueAt, state: .retrying)
    }

    /// Bring forward games still waiting for their first attempt (going
    /// offline shouldn't wait out a filing delay). Games already retrying
    /// or unreconciled keep their schedule and retry window.
    func expediteUnattempted() {
        let now = time.now()
        for (gameID, item) in items where item.attempts == 0 && item.state == .retrying {
            items[gameID]?.dueAt = min(item.dueAt, now)
        }
    }

    var queuedGameIDs: [String] {
        items.keys.sorted()
    }

    /// Queued games still inside their retry window: what a drain waits for
    /// (unreconciled games retry on a slow cadence and aren't waited on).
    var dueGameIDs: [String] {
        items.values.filter { $0.state == .retrying }.map(\.gameID).sorted()
    }

    var unreconciledGameIDs: [String] {
        items.values.filter { $0.state == .unreconciled }.map(\.gameID).sorted()
    }

    /// Games that can't be filed as things stand and aren't retried this
    /// launch.
    var quarantinedGameIDs: [String] {
        items.values.filter { $0.state == .quarantined }.map(\.gameID).sorted()
    }

    /// Work the queue until cancelled or the gate closes.
    func run() async {
        while !Task.isCancelled {
            guard let next = items.values.filter({ $0.state != .quarantined }).min(by: { $0.dueAt < $1.dueAt }) else {
                if !(await sleep(Self.pollInterval)) { break }
                continue
            }
            let settings = await settingsProvider()
            let now = time.now()
            var wait = next.dueAt - now
            if let lastFetchAt {
                wait = max(wait, lastFetchAt + .seconds(settings.connection.exportMinimumSpacingSeconds) - now)
            }
            if wait > .zero {
                if !(await sleep(min(wait, Self.pollInterval))) { break }
                continue
            }
            if let stopReason = await attempt(next.gameID) {
                onEvent(.stopped(reason: stopReason))
                return
            }
        }
        onEvent(.stopped(reason: "cancelled"))
    }

    /// False when cancelled.
    private func sleep(_ duration: Duration) async -> Bool {
        do {
            try await time.sleep(for: duration)
            return true
        } catch {
            return false
        }
    }

    /// One export attempt. Returns a reason when the reconciler must stop.
    private func attempt(_ gameID: String) async -> String? {
        if await hasLiveFilingOwner(gameID) {
            items[gameID] = nil
            return nil
        }
        let filingState: LichessBotFilingState
        do {
            filingState = try await store.filingState(gameID: gameID)
        } catch {
            if Self.isPermanentFilingFailure(error) {
                quarantine(gameID, reason: "checking the game's files failed: \(error.localizedDescription)")
            } else {
                onEvent(.finalizeFailed(gameID: gameID, error: "checking the game's files: \(error.localizedDescription)"))
                markUnreconciled(gameID, reason: "checking the game's files failed")
            }
            return nil
        }
        switch filingState {
        case .journalInProgress:
            break
        case .filed:
            items[gameID] = nil
            onEvent(.alreadyFiled(gameID: gameID))
            return nil
        case .filedButUnreadable(let path, let error):
            quarantine(gameID, reason: "a record filed for it doesn't decode (\(path): \(error))")
            return nil
        case .missing:
            items[gameID] = nil
            onEvent(.nothingToFile(gameID: gameID, reason: "it has no journal in InProgress/ and no filed record"))
            return nil
        }
        lastFetchAt = time.now()
        items[gameID]?.attempts += 1
        do {
            let export = try LichessBotGameExport.decode(try await api.exportGame(gameID: gameID))
            if export.status.isLive == true {
                retry(gameID, reason: "export still says \(export.status.raw)", whenWindowEnds: .markUnreconciled)
                return nil
            }
            await finalize(gameID, export: export, exportUnavailableReason: nil)
        } catch LichessBotGateError.closed(let reason) {
            return "request gate closed: \(reason)"
        } catch LichessBotGateError.rateLimited {
            // The gate holds every request for the cooldown; try again as
            // soon as it lets requests through.
            items[gameID]?.dueAt = time.now()
        } catch let error as LichessBotAPIError {
            if case .unauthorized = error {
                return error.localizedDescription
            }
            if case .http(404, _) = error {
                await handleMissingExport(gameID)
            } else {
                retry(gameID, reason: error.localizedDescription, whenWindowEnds: .markUnreconciled)
            }
        } catch is CancellationError {
            return "cancelled"
        } catch {
            retry(gameID, reason: "export unusable: \(error)", whenWindowEnds: .markUnreconciled)
        }
        return nil
    }

    /// Failures no retry can fix: the journal, the records and the export
    /// are what they are. Disk-full, permission and other write errors are
    /// not here: they stay on the slow retry.
    private static func isPermanentFilingFailure(_ error: Error) -> Bool {
        switch error {
        case is LichessBotRecordError, is LichessBotJSONLinesError, is DecodingError, is EncodingError:
            return true
        case let error as CocoaError:
            return error.code == .fileReadNoSuchFile
        default:
            return false
        }
    }

    /// Lichess keeps no export for some aborted games. If the journal says
    /// the game was aborted, the journal alone is the record; otherwise a
    /// missing export is retried through the live-export window (it may be
    /// transient) and then quarantined rather than retried forever.
    private func handleMissingExport(_ gameID: String) async {
        let journalStatus: String?
        do {
            journalStatus = try await store.journalFinishedStatus(gameID: gameID)
        } catch {
            if Self.isPermanentFilingFailure(error) {
                quarantine(gameID, reason: "export not found and the journal is unreadable: \(error.localizedDescription)")
            } else {
                onEvent(.finalizeFailed(gameID: gameID, error: "reading the journal: \(error.localizedDescription)"))
                markUnreconciled(gameID, reason: "export not found and the journal is unreadable")
            }
            return
        }
        switch journalStatus.flatMap(LichessBotGameStatusName.init(rawValue:)) {
        case .aborted, .noStart:
            await finalize(gameID, export: nil, exportUnavailableReason: "Lichess has no export for this aborted game")
        default:
            // Most likely an aborted game whose finish the journal missed
            // (Lichess deletes those).
            retry(gameID, reason: "export not found", whenWindowEnds: .quarantine)
        }
    }

    private func finalize(_ gameID: String, export: LichessBotGameExport?, exportUnavailableReason: String?) async {
        // The export (or the 404 path's journal read) can outlast a session
        // starting for the game or the controller recording its finish: a
        // game that gained an owner meanwhile is left to it.
        if await hasLiveFilingOwner(gameID) {
            items[gameID] = nil
            return
        }
        do {
            let finalized = try await store.finalize(gameID: gameID, export: export, exportUnavailableReason: exportUnavailableReason)
            items[gameID] = nil
            onEvent(.finalized(finalized))
        } catch {
            if Self.isPermanentFilingFailure(error) {
                quarantine(gameID, reason: "filing failed: \(error.localizedDescription)")
            } else {
                onEvent(.finalizeFailed(gameID: gameID, error: error.localizedDescription))
                markUnreconciled(gameID, reason: "finalize failed: \(error.localizedDescription)")
            }
        }
    }

    private func retry(_ gameID: String, reason: String, whenWindowEnds expiry: WindowExpiry) {
        guard var item = items[gameID] else { return }
        let now = time.now()
        let windowOver = item.state == .unreconciled || now - item.firstAttemptAt >= Self.liveExportRetryWindow
        if windowOver && expiry == .quarantine {
            quarantine(gameID, reason: reason)
            return
        }
        if item.state == .unreconciled {
            item.dueAt = now + Self.unreconciledRetryInterval
            items[gameID] = item
            onEvent(.waiting(gameID: gameID, reason: reason, retryIn: Self.unreconciledRetryInterval))
            return
        }
        if windowOver {
            markUnreconciled(gameID, reason: reason)
            return
        }
        let delay = Self.retryBackoff.delay(attempt: max(0, item.attempts - 1), unitRandom: Double.random(in: 0...1))
        item.dueAt = now + delay
        items[gameID] = item
        onEvent(.waiting(gameID: gameID, reason: reason, retryIn: delay))
    }

    private func markUnreconciled(_ gameID: String, reason: String) {
        guard var item = items[gameID] else { return }
        item.state = .unreconciled
        item.dueAt = time.now() + Self.unreconciledRetryInterval
        items[gameID] = item
        onEvent(.unreconciled(gameID: gameID, reason: reason))
    }

    private func quarantine(_ gameID: String, reason: String) {
        guard var item = items[gameID], item.state != .quarantined else { return }
        item.state = .quarantined
        items[gameID] = item
        onEvent(.quarantined(gameID: gameID, reason: reason))
    }
}
