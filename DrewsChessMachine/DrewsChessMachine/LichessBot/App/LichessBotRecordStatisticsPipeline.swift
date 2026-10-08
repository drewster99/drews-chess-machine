import Foundation

/// What the Record card shows for its statistics.
enum LichessBotRecordStatisticsState: Sendable, Equatable {
    /// No snapshot yet (the games index has not loaded).
    case loading
    case ready(LichessBotRecordStatistics)
    /// The latest computation failed; the text says why.
    case failed(String)
}

extension LichessBotRecordStatisticsState {
    /// The snapshot as a list of zero or one item with a constant identity,
    /// so a view can show it with `ForEach` (no `if` in the body) and keep
    /// its children's state across recomputes.
    var readyItems: [LichessBotReadyStatistics] {
        guard case .ready(let statistics) = self else { return [] }
        return [LichessBotReadyStatistics(statistics: statistics)]
    }

    var isLoading: Bool {
        self == .loading
    }

    /// The failure text, or nil when not failed.
    var failureText: String? {
        guard case .failed(let text) = self else { return nil }
        return text
    }
}

/// A ready snapshot, identified by a constant: there is only ever one, and
/// a new snapshot must not look like a new view to SwiftUI.
struct LichessBotReadyStatistics: Identifiable {
    let statistics: LichessBotRecordStatistics
    var id: Int { 0 }
}

/// Why a statistics computation ran, for the session log.
enum LichessBotRecordStatisticsReason: String, Sendable {
    /// The games index changed (launch, a filed game, a rebuild).
    case index
    /// A period boundary passed, or the clock or time zone changed.
    case clock
    /// The games' resolved origins changed (a challenge-log fact arrived).
    case origins
    /// The Model filter changed (`rememberedModel`).
    case model
}

/// Computes the Record card's statistics off the main actor and keeps the
/// latest result (`LICHESS_BOT_RECORD_STATS_PLAN.md` §4.3). Owned by the
/// controller, which feeds it every new games index and shuts it down with
/// itself.
///
/// **Where the work runs.** `LichessBotRecordStatistics.compute` is
/// synchronous CPU work over every index row — milliseconds today, longer
/// at thousands of games — so it never runs on the main actor or on the
/// cooperative pool (where it would starve other tasks). It runs on
/// `queue`, a serial `DispatchQueue` behind `LichessBotFileQueue.run`'s
/// continuation, separate from the controller's file queue so an index
/// rebuild never delays a recompute and the reverse.
///
/// **Ordering.** Each request takes the next number from `requestCount`;
/// only the outcome of the latest request is applied. An older outcome that
/// finishes late — a result or an error — is dropped, so the card never
/// steps back to older numbers. Requests are recomputed from the full rows
/// rather than updated incrementally: one code path and one definition per
/// statistic, bounded by the 10,000-row scale test.
///
/// **When it recomputes.** On every index change; when a period boundary
/// passes (`clockTick`, driven by a 60-second loop in the window) or the
/// time zone differs from the snapshot's; and at once on the system's
/// time-zone and clock-change notifications, because a clock set backwards
/// leaves `now` before `validUntil` and the tick would never fire for it. A
/// failed state is retried by the next index or time change, not by every
/// tick (which would log the same failure every minute).
///
/// It also remembers the panel's tab, period and filter in the controller's
/// defaults (like the Settings tab): a value that no longer exists is
/// logged and ignored. Not `@AppStorage`, which would read the test host's
/// real defaults in render tests.
@MainActor
@Observable
final class LichessBotRecordStatisticsPipeline {

    typealias Compute = @Sendable (_ rows: [LichessBotGameSummary], _ origins: [String: LichessBotGameOriginCategory]?, _ now: Date, _ calendar: Calendar) throws -> LichessBotRecordStatistics

    private(set) var state: LichessBotRecordStatisticsState = .loading

    var rememberedPane: LichessBotRecordPane = .timeControls {
        didSet { defaults.set(rememberedPane.rawValue, forKey: Self.paneKey) }
    }
    var rememberedPeriod: LichessBotStatsPeriod = .allTime {
        didSet { defaults.set(rememberedPeriod.rawValue, forKey: Self.periodKey) }
    }
    var rememberedFilter: LichessBotStatsFilter = .all {
        didSet { defaults.set(rememberedFilter.rawValue, forKey: Self.filterKey) }
    }
    /// The Model filter. Unlike the period and the Rated / Casual filter it
    /// is not precomputed for every value (a run has dozens of models): the
    /// statistics are recomputed from the selected model's games when it
    /// changes.
    var rememberedModel: LichessBotStatsModelSelection = .all {
        didSet {
            guard rememberedModel != oldValue else { return }
            do {
                defaults.set(try JSONEncoder().encode(rememberedModel), forKey: Self.modelKey)
            } catch {
                SessionLogger.shared.log("[LICHESS-BOT] record card: could not remember the model filter: \(error.localizedDescription)")
            }
            schedule(reason: .model, now: Date())
        }
    }
    /// Every model some game is attributed to, for the Model filter's menu;
    /// from all games, whatever the selection. Empty until the first result.
    private(set) var modelChoices: [LichessBotStatsModelChoice] = []

    static let paneKey = "lichessBot.overview.record.pane"
    static let periodKey = "lichessBot.overview.record.period"
    static let filterKey = "lichessBot.overview.record.filter"
    static let modelKey = "lichessBot.overview.record.model"

    @ObservationIgnored private let defaults: UserDefaults
    @ObservationIgnored private let queue: LichessBotFileQueue
    @ObservationIgnored private let compute: Compute
    @ObservationIgnored private let calendar: @MainActor () -> Calendar
    /// The latest index rows; nil until the first index arrives.
    @ObservationIgnored private var rows: [LichessBotGameSummary]?
    /// Each game's origin category from the controller's resolver; nil
    /// until it first reports, and then there is no origin breakdown.
    @ObservationIgnored private var origins: [String: LichessBotGameOriginCategory]?
    @ObservationIgnored private(set) var requestCount = 0
    @ObservationIgnored private var isShutDown = false
    @ObservationIgnored private var observers: [NSObjectProtocol] = []
    /// The latest request's task, so tests can wait for it.
    @ObservationIgnored private(set) var latestComputation: Task<Void, Never>?
    /// Outcomes applied (results and errors), for tests that check a late
    /// outcome was dropped.
    @ObservationIgnored private(set) var appliedOutcomeCount = 0
    /// An index change whose `record stats (index)` log line has not been
    /// written yet; the next applied result writes it.
    @ObservationIgnored private(set) var indexLogPending = false

    /// - Parameters:
    ///   - compute: the statistics function; tests substitute one that
    ///     blocks or throws.
    ///   - calendar: the calendar periods are computed in, read on each
    ///     request so a time-zone change is picked up.
    init(
        defaults: UserDefaults,
        queue: LichessBotFileQueue = LichessBotFileQueue(label: "drewschess.lichessbot.statistics", qos: .utility),
        compute: @escaping Compute = { rows, origins, now, calendar in try LichessBotRecordStatistics.compute(rows: rows, origins: origins, now: now, calendar: calendar) },
        calendar: @escaping @MainActor () -> Calendar = { Calendar.current }
    ) {
        self.defaults = defaults
        self.queue = queue
        self.compute = compute
        self.calendar = calendar
        if let saved = defaults.string(forKey: Self.paneKey) {
            if let pane = LichessBotRecordPane(rawValue: saved) {
                rememberedPane = pane
            } else {
                SessionLogger.shared.log("[LICHESS-BOT] record card: ignoring the remembered tab \"\(saved)\", which no longer exists")
            }
        }
        if let saved = defaults.string(forKey: Self.periodKey) {
            if let period = LichessBotStatsPeriod(rawValue: saved) {
                rememberedPeriod = period
            } else {
                SessionLogger.shared.log("[LICHESS-BOT] record card: ignoring the remembered period \"\(saved)\", which no longer exists")
            }
        }
        if let saved = defaults.string(forKey: Self.filterKey) {
            if let filter = LichessBotStatsFilter(rawValue: saved) {
                rememberedFilter = filter
            } else {
                SessionLogger.shared.log("[LICHESS-BOT] record card: ignoring the remembered filter \"\(saved)\", which no longer exists")
            }
        }
        if let saved = defaults.data(forKey: Self.modelKey) {
            do {
                rememberedModel = try JSONDecoder().decode(LichessBotStatsModelSelection.self, from: saved)
            } catch {
                SessionLogger.shared.log("[LICHESS-BOT] record card: ignoring the remembered model filter, which does not decode: \(error.localizedDescription)")
            }
        }
    }

    /// Start following the system's time-zone and clock changes. Separate
    /// from `init` so a controller made only for a test of something else
    /// registers nothing.
    func observeSystemTimeChanges() {
        guard observers.isEmpty, !isShutDown else { return }
        for name in [Notification.Name.NSSystemTimeZoneDidChange, .NSSystemClockDidChange] {
            observers.append(NotificationCenter.default.addObserver(forName: name, object: nil, queue: .main) { [weak self] _ in
                Task { @MainActor in
                    self?.systemTimeChanged()
                }
            })
        }
    }

    /// The controller's origin resolver (`originsByGameID`) changed: keep
    /// each game's category for the origin breakdown (§11 D1), and
    /// recompute when there are rows to compute from.
    func originsChanged(_ categories: [String: LichessBotGameOriginCategory]) {
        guard categories != origins else { return }
        origins = categories
        guard rows != nil else { return }
        schedule(reason: .origins, now: Date())
    }

    /// The games index changed: recompute from its rows.
    func indexChanged(rows: [LichessBotGameSummary]) {
        self.rows = rows
        indexLogPending = true
        schedule(reason: .index, now: Date())
    }

    /// The games index changed, and with it the origins resolved for its
    /// games: take both, then recompute once. A new game gets an origin, so
    /// reporting the two separately would compute the new origins over the
    /// old rows first, only to throw that result away.
    func indexChanged(rows: [LichessBotGameSummary], origins categories: [String: LichessBotGameOriginCategory]) {
        origins = categories
        indexChanged(rows: rows)
    }

    /// Recompute if a period boundary has passed since the snapshot, or the
    /// time zone differs from the snapshot's. A failed state is left for the
    /// next index or time change.
    func clockTick(now: Date) {
        guard case .ready(let statistics) = state else { return }
        if now >= statistics.validUntil || calendar().timeZone.identifier != statistics.timeZoneIdentifier {
            schedule(reason: .clock, now: now)
        }
    }

    /// The Record card's clock (`LichessBotRecordStatisticsClock`): a
    /// `clockTick` at once — a period boundary may have passed while the
    /// window was closed — then every `tickInterval`. Returns when the task
    /// is canceled (the window closed). `Task.sleep` runs on the continuous
    /// clock, which keeps counting while the Mac sleeps, so a wake is
    /// caught at the next tick.
    func runClock(tickInterval: Duration) async {
        clockTick(now: Date())
        while true {
            do {
                try await Task.sleep(for: tickInterval)
            } catch {
                // Canceled: the window closed. `Task.sleep` throws nothing
                // else.
                return
            }
            clockTick(now: Date())
        }
    }

    /// The system's time zone or clock changed: recompute unconditionally
    /// (a clock set backwards leaves `now` before `validUntil`). Whether a
    /// running process's `TimeZone.current` follows a System Settings change
    /// unaided is not something this code relies on: the cached zone is
    /// reset first.
    private func systemTimeChanged() {
        NSTimeZone.resetSystemTimeZone()
        guard rows != nil else { return }
        schedule(reason: .clock, now: Date())
    }

    /// Compute from the latest rows on the statistics queue; apply the
    /// outcome only if no newer request was made meanwhile.
    func schedule(reason: LichessBotRecordStatisticsReason, now: Date) {
        guard !isShutDown, let rows else { return }
        requestCount += 1
        let request = requestCount
        let calendar = calendar()
        let origins = self.origins
        let compute = self.compute
        let queue = self.queue
        let model = rememberedModel
        latestComputation = Task { @MainActor [weak self] in
            let outcome: Result<(LichessBotRecordStatistics, Double, [LichessBotStatsModelChoice]), Error>
            do {
                let computed = try await queue.run {
                    let started = DispatchTime.now().uptimeNanoseconds
                    let statistics = try compute(rows.filter { model.includes($0) }, origins, now, calendar)
                    let choices = LichessBotStatsModelChoice.choices(from: rows)
                    let milliseconds = Double(DispatchTime.now().uptimeNanoseconds - started) / 1_000_000
                    return (statistics, milliseconds, choices)
                }
                outcome = .success(computed)
            } catch {
                outcome = .failure(error)
            }
            self?.apply(outcome, request: request, reason: reason, model: model)
        }
    }

    private func apply(_ outcome: Result<(LichessBotRecordStatistics, Double, [LichessBotStatsModelChoice]), Error>, request: Int, reason: LichessBotRecordStatisticsReason, model: LichessBotStatsModelSelection) {
        guard request == requestCount else { return }
        appliedOutcomeCount += 1
        switch outcome {
        case .success(let (statistics, milliseconds, choices)):
            state = .ready(statistics)
            modelChoices = choices
            // A remembered model no game is attributed to any more (its
            // records were cleared): back to every model, which recomputes.
            if case .model(let key) = rememberedModel, !choices.contains(where: { $0.key == key }) {
                SessionLogger.shared.log("[LICHESS-BOT] record card: the model filter's model has no games; showing every model")
                rememberedModel = .all
            }
            // One line per index change, not per clock tick: the numbers a
            // tick changes are period boundaries, already in the snapshot.
            // Keyed on the pending flag rather than this request's reason:
            // a clock request made while an index request was in flight
            // supersedes it but computes from the same new rows, and the
            // index change must still get its line.
            if indexLogPending {
                indexLogPending = false
                // The numbers are the selected model's when a model is
                // selected; the line says which.
                SessionLogger.shared.log(LichessBotRecordStatsLogLine.text(statistics, reason: LichessBotRecordStatisticsReason.index.rawValue, milliseconds: milliseconds)
                    + model.logSuffix)
            }
        case .failure(let error):
            if isShutDown, error is LichessBotFileQueueError {
                // The queue closed under a request made just before the
                // shutdown: nothing to show any more.
                return
            }
            let text = error.localizedDescription
            state = .failed(text)
            SessionLogger.shared.log("[LICHESS-BOT] record stats failed (\(reason.rawValue)): \(text)")
        }
    }

    /// Stop for good: no new requests, the system observers removed, and
    /// the queue closed once the work already on it has run.
    func shutdown(reason: String) async {
        stopScheduling()
        await queue.close(reason: "the bot shut down: \(reason)")
    }

    /// Accept no more requests and drop the system observers, without
    /// waiting for anything. Idempotent.
    func stopScheduling() {
        isShutDown = true
        for observer in observers {
            NotificationCenter.default.removeObserver(observer)
        }
        observers = []
    }
}
