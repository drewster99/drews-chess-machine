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
/// then on. A game that is still being played by a session is dropped from
/// the queue; its session's end enqueues it again.
///
/// Launch recovery enqueues every leftover `InProgress/` journal: a game
/// still live comes back as a live export (and its `gameStart` restarts
/// its session), while a game that ended while the app was down is
/// finalized from its journal plus the export.
actor LichessBotReconciler {
    private struct Item {
        let gameID: String
        let firstAttemptAt: Duration
        var attempts: Int
        var dueAt: Duration
        var unreconciled: Bool
    }

    private let api: any LichessBotExportAPI
    private let store: LichessBotRecordStore
    private let time: any LichessBotTimeSource
    private let settingsProvider: @Sendable () async -> LichessBotSettings
    private let isGameActive: @Sendable (String) async -> Bool
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
        isGameActive: @escaping @Sendable (String) async -> Bool,
        onEvent: @escaping @Sendable (LichessBotReconcilerEvent) -> Void
    ) {
        self.api = api
        self.store = store
        self.time = time
        self.settingsProvider = settingsProvider
        self.isGameActive = isGameActive
        self.onEvent = onEvent
    }

    /// Queue a game for reconciliation, `delay` from now. Enqueueing a game
    /// already queued (say, one left unreconciled at launch whose session
    /// has now ended) starts it over with a fresh retry window, due no
    /// later than it already was.
    func enqueue(gameID: String, after delay: Duration = .zero) {
        let due = time.now() + delay
        let dueAt = items[gameID].map { min($0.dueAt, due) } ?? due
        items[gameID] = Item(gameID: gameID, firstAttemptAt: dueAt, attempts: 0, dueAt: dueAt, unreconciled: false)
    }

    var queuedGameIDs: [String] {
        items.keys.sorted()
    }

    /// Queued games still inside their retry window: what a drain waits for
    /// (unreconciled games retry on a slow cadence and aren't waited on).
    var dueGameIDs: [String] {
        items.values.filter { !$0.unreconciled }.map(\.gameID).sorted()
    }

    var unreconciledGameIDs: [String] {
        items.values.filter(\.unreconciled).map(\.gameID).sorted()
    }

    /// Work the queue until cancelled or the gate closes.
    func run() async {
        while !Task.isCancelled {
            guard let next = items.values.min(by: { $0.dueAt < $1.dueAt }) else {
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
        if await isGameActive(gameID) {
            items[gameID] = nil
            return nil
        }
        lastFetchAt = time.now()
        items[gameID]?.attempts += 1
        do {
            let export = try LichessBotGameExport.decode(try await api.exportGame(gameID: gameID))
            if export.status.isLive == true {
                retry(gameID, reason: "export still says \(export.status.raw)")
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
                retry(gameID, reason: error.localizedDescription)
            }
        } catch is CancellationError {
            return "cancelled"
        } catch {
            retry(gameID, reason: "export unusable: \(error)")
        }
        return nil
    }

    /// Lichess keeps no export for some aborted games. If the journal says
    /// the game was aborted, the journal alone is the record; otherwise a
    /// missing export is unexpected and retried.
    private func handleMissingExport(_ gameID: String) async {
        let journalStatus: String?
        do {
            journalStatus = try await store.journalFinishedStatus(gameID: gameID)
        } catch {
            onEvent(.finalizeFailed(gameID: gameID, error: "reading the journal: \(error.localizedDescription)"))
            markUnreconciled(gameID, reason: "export not found and the journal is unreadable")
            return
        }
        switch journalStatus.flatMap(LichessBotGameStatusName.init(rawValue:)) {
        case .aborted, .noStart:
            await finalize(gameID, export: nil, exportUnavailableReason: "Lichess has no export for this aborted game")
        default:
            retry(gameID, reason: "export not found")
        }
    }

    private func finalize(_ gameID: String, export: LichessBotGameExport?, exportUnavailableReason: String?) async {
        do {
            let finalized = try await store.finalize(gameID: gameID, export: export, exportUnavailableReason: exportUnavailableReason)
            items[gameID] = nil
            onEvent(.finalized(finalized))
        } catch {
            onEvent(.finalizeFailed(gameID: gameID, error: error.localizedDescription))
            markUnreconciled(gameID, reason: "finalize failed: \(error.localizedDescription)")
        }
    }

    private func retry(_ gameID: String, reason: String) {
        guard var item = items[gameID] else { return }
        let now = time.now()
        if item.unreconciled {
            item.dueAt = now + Self.unreconciledRetryInterval
            items[gameID] = item
            onEvent(.waiting(gameID: gameID, reason: reason, retryIn: Self.unreconciledRetryInterval))
            return
        }
        if now - item.firstAttemptAt >= Self.liveExportRetryWindow {
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
        item.unreconciled = true
        item.dueAt = time.now() + Self.unreconciledRetryInterval
        items[gameID] = item
        onEvent(.unreconciled(gameID: gameID, reason: reason))
    }
}
