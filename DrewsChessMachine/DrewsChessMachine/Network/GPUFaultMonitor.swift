import Foundation
import OSLog
import os

/// Reads macOS's own GPU fault messages for this process and records each
/// one in `GPUFaultLedger` — the detection layer that sees what the app's
/// checked submissions can't (GPU fault forensics plan, A3).
///
/// When the GPU hangs or resets, Metal logs one message per affected
/// command buffer, in the process that owned it:
///
///     (Metal) Execution of the command buffer was aborted due to an error
///     during execution. Caused GPU Hang Error (00000003:kIOGPUCommandBufferCallbackErrorHang)
///     (Metal) … Discarded (victim of GPU error/recovery) (00000005:kIOGPUCommandBufferCallbackErrorInnocentVictim)
///
/// That includes the command buffers MPSGraph creates when it splits a
/// submission, which the app can't reach (`GPUSubmission`'s doc). On
/// 2026-10-09 three such resets reached no session log at all; these
/// messages were the only record, and the system log keeps error entries for
/// hours, not days.
///
/// The monitor polls `OSLogStore(scope: .currentProcessIdentifier)` — this
/// process's own entries, readable without an entitlement — every
/// `pollIntervalSeconds` on a utility queue. A query takes about two seconds on a
/// busy machine (measured beside three training processes), so it is never
/// run on a training thread; `checkNow()` runs one on the monitor's queue and
/// is awaited only before actions that make state durable (saves,
/// promotions), where two seconds don't matter.
///
/// Only messages naming a command-buffer callback error count
/// (`isFaultMessage`); each new one is logged once as `[GPU-SYSLOG]` and
/// recorded in the ledger with macOS's log time. If the store can't be
/// opened or queried, that is logged once and reported through
/// `availability`; the run continues on the other layers, and every record
/// of the run says the monitor was unavailable, so "no faults" is never read
/// from a run whose monitor wasn't working.
///
/// Runs and pauses. A run's `GPUFaultWatch` acts on every ledger fault
/// recorded after the run began, so the monitor must never record, after a
/// start, a fault that belongs to no run. Every `start` therefore sets the
/// window the run counts from — `lookBack` before the start, so a fault
/// during the model or session load just before it still counts — and never
/// reads further back than that window or the last read, whichever is later.
/// A fault logged earlier but returned by a query (the polls' overlap) is
/// logged as not counted and never recorded. Between GUI runs the monitor is
/// paused: one closing poll an interval after the pause records a fault from
/// the run's last seconds (logged after its last poll), and then nothing is
/// read until the next start. Without that window, a resume would read every
/// hour of the pause in one query and record a fault from between runs as
/// the new run's, stopping it at once.
final class GPUFaultMonitor: @unchecked Sendable {
    static let shared = GPUFaultMonitor(ledger: .shared)

    /// How often the background poll reads the log, in seconds.
    static let pollIntervalSeconds = 10

    /// How far each poll re-reads before the previous one's end: an entry
    /// logged just before a query ran may not have been visible to it.
    /// `seen` keeps a re-read entry from being recorded twice.
    static let readOverlapSeconds: TimeInterval = 5

    /// How long `seen` remembers an entry before the newest read position.
    /// Longer than `readOverlapSeconds`, so no entry a query can return
    /// again has been forgotten.
    static let seenHorizonSeconds: TimeInterval = 60

    /// The text every counted message contains: Metal's name for a command
    /// buffer's failure callback (`…Hang`, `…InnocentVictim`, `…PageFault`,
    /// `…Timeout`, …).
    static let faultMarker = "kIOGPUCommandBufferCallbackError"

    enum Availability: Sendable, Equatable {
        /// `start` hasn't run.
        case notStarted
        case available
        /// The store couldn't be opened or queried; the reason, verbatim.
        case unavailable(String)

        /// `available`, `unavailable: …` or `not started`, for `[RUN]`
        /// and `results.json`.
        var label: String {
            switch self {
            case .notStarted: return "not started"
            case .available: return "available"
            case .unavailable(let reason): return "unavailable: \(reason)"
            }
        }
    }

    /// One of macOS's log entries for this process: its log time and
    /// composed message. Also the key `seen` dedupes on.
    struct LogEntry: Hashable, Sendable {
        let date: Date
        let message: String
    }

    /// What a start or resume reads and counts.
    struct StartWindow: Equatable, Sendable {
        /// Where the next poll reads from (less `readOverlapSeconds`).
        let readFrom: Date
        /// Entries logged before this are not the starting run's: a poll
        /// that returns one logs it as not counted and doesn't record it.
        let countFrom: Date
        /// The span of a pause no poll reads (from the last read to
        /// `countFrom`, give or take the polls' overlap), when the pause
        /// outlasted the look-back window; nil otherwise.
        let unread: DateInterval?
    }

    /// Whether a log message is a GPU command-buffer fault. Pure; the
    /// incidents' captured messages are its test fixtures.
    static func isFaultMessage(_ message: String) -> Bool {
        message.contains(faultMarker)
    }

    /// The window a start at `now` reads and counts, given where the last
    /// poll left off (nil before the first start). Pure.
    ///
    /// The run counts faults logged from `lookBack` before `now`. Reading
    /// starts at the later of that and the last read: a short pause is read
    /// in full (all of it is inside the look-back window), a long one is not
    /// read before the window.
    static func startWindow(lastReadFrom: Date?, now: Date, lookBack: TimeInterval) -> StartWindow {
        let countFrom = now.addingTimeInterval(-lookBack)
        guard let lastReadFrom, lastReadFrom < countFrom else {
            return StartWindow(readFrom: lastReadFrom ?? countFrom, countFrom: countFrom, unread: nil)
        }
        return StartWindow(readFrom: countFrom, countFrom: countFrom,
                           unread: DateInterval(start: lastReadFrom, end: countFrom))
    }

    /// A fault entry a poll hasn't reported before, and whether it counts
    /// for the current run.
    struct PolledEntry: Equatable, Sendable {
        let entry: LogEntry
        /// Logged at or after the run's `countFrom`: logged and recorded.
        /// Otherwise logged as not counted, never recorded.
        let counts: Bool
    }

    /// The fault entries among `entries` not in `seen`, in the order given,
    /// each with whether it counts (logged at or after `countFrom`). Adds
    /// them to `seen`, so neither kind is reported twice. Pure.
    static func newFaultEntries(_ entries: [LogEntry], countFrom: Date, seen: inout Set<LogEntry>) -> [PolledEntry] {
        var found: [PolledEntry] = []
        for entry in entries where isFaultMessage(entry.message) {
            guard seen.insert(entry).inserted else { continue }
            found.append(PolledEntry(entry: entry, counts: entry.date >= countFrom))
        }
        return found
    }

    private let ledger: GPUFaultLedger
    /// Opens the log on the first start (and on every start until it opens).
    private let openLogSource: @Sendable () throws -> any GPUFaultLogSource
    /// How long after a pause its closing poll runs.
    private let closingPollDelaySeconds: TimeInterval
    /// Every poll and all mutable state below run on this queue.
    private let queue = DispatchQueue(label: "drewschessmachine.gpu-fault-monitor", qos: .utility)
    private let availabilityBox = SyncBox<Availability>(.notStarted)
    /// Poll durations since the last `takePollDurations()`, for the `[MEM]`
    /// line (so the monitor's own cost is visible in a long-lived process).
    private let pollDurationsMs = SyncBox<[Double]>([])

    // Queue-confined state.
    private var logSource: (any GPUFaultLogSource)?
    private var timer: DispatchSourceTimer?
    /// Read entries from here on (less `readOverlapSeconds`; `seen` drops
    /// repeats). Nil before the first start.
    private var readFrom: Date?
    /// The current run's window start (`StartWindow.countFrom`). Nil before
    /// the first start and once a pause's closing poll has run: with no run
    /// to count for, a poll reads nothing.
    private var countFrom: Date?
    /// Entries already reported (counted or not).
    private var seen = Set<LogEntry>()
    /// Queries that failed since the last success.
    private var consecutiveFailures = 0
    /// Advanced by every start and pause; a pause's closing poll runs only
    /// if neither has happened since it was scheduled.
    private var startPauseGeneration = 0

    /// `openLogSource` and `closingPollDelaySeconds` are for tests; production
    /// reads this process's `OSLogStore` and closes one poll interval after
    /// a pause.
    init(ledger: GPUFaultLedger,
         openLogSource: @escaping @Sendable () throws -> any GPUFaultLogSource = { try ProcessLogStoreFaultSource() },
         closingPollDelaySeconds: TimeInterval = TimeInterval(GPUFaultMonitor.pollIntervalSeconds)) {
        self.ledger = ledger
        self.openLogSource = openLogSource
        self.closingPollDelaySeconds = closingPollDelaySeconds
    }

    var availability: Availability {
        availabilityBox.value
    }

    /// A query that fails this many times in a row marks the monitor
    /// unavailable (logged); it keeps retrying at every poll and marks
    /// itself available again on the next success.
    static let failuresBeforeUnavailable = 3

    /// Opens the store (on the monitor's queue, so a caller on the main actor
    /// never waits) and starts or resumes the background poll, counting
    /// faults logged from `lookBack` before now (`startWindow`), so a fault
    /// just before the run started is still seen and one from before that is
    /// not counted. Does nothing while already polling.
    func start(lookBack: TimeInterval = 60) {
        queue.async { [weak self] in
            self?.startOnQueue(lookBack: lookBack)
        }
    }

    /// Stops the background poll until the next `start()` — between GUI
    /// runs, when no training is left to protect, so the ≈2 s query every
    /// 10 s isn't spent for nothing. One closing poll runs
    /// `closingPollDelaySeconds` later (unless a start comes first; its polls
    /// cover the same entries), so a fault macOS logged in the run's last
    /// seconds, after its last poll, is still logged and recorded. After it,
    /// `checkNow()` reads nothing until the next start.
    func pause() {
        queue.async { [weak self] in
            self?.pauseOnQueue()
        }
    }

    /// Runs one poll now, on the monitor's queue, and returns once it has
    /// recorded whatever it found. The barrier before a save or a promotion:
    /// afterwards the ledger holds every fault macOS had logged for this
    /// process since the run's window began. Does nothing (beyond returning)
    /// when the monitor isn't available, or is paused with its closing poll
    /// done.
    func checkNow() async {
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            queue.async {
                self.pollOnQueue()
                continuation.resume()
            }
        }
    }

    /// The poll durations recorded since the last call, in milliseconds.
    func takePollDurations() -> [Double] {
        pollDurationsMs.mutate { durations -> [Double] in
            let taken = durations
            durations = []
            return taken
        }
    }

    // MARK: - Private (on `queue`)

    /// `start`'s work, on `queue`.
    private func startOnQueue(lookBack: TimeInterval) {
        // Already polling: the run in progress keeps its window.
        guard timer == nil else { return }
        startPauseGeneration += 1
        let window = Self.startWindow(lastReadFrom: readFrom, now: Date(), lookBack: lookBack)
        readFrom = window.readFrom
        countFrom = window.countFrom
        if logSource == nil {
            do {
                logSource = try openLogSource()
                availabilityBox.value = .available
                SessionLogger.shared.log(
                    "[GPU-SYSLOG] monitoring this process's GPU fault messages every \(Self.pollIntervalSeconds) s")
            } catch {
                markUnavailable("could not open the log store: \(error.localizedDescription)")
                return
            }
        } else {
            let unread = window.unread.map {
                "; faults logged from \(Self.timestamp($0.start)) to \(Self.timestamp($0.end)), "
                    + "while paused, are not read"
            } ?? ""
            SessionLogger.shared.log(
                "[GPU-SYSLOG] monitoring resumed: faults logged from \(Self.timestamp(window.countFrom)) on count\(unread)")
        }
        let timer = DispatchSource.makeTimerSource(queue: queue)
        let interval = DispatchTimeInterval.seconds(Self.pollIntervalSeconds)
        timer.schedule(deadline: .now() + interval, repeating: interval, leeway: .seconds(1))
        timer.setEventHandler { [weak self] in
            self?.pollOnQueue()
        }
        timer.resume()
        self.timer = timer
    }

    /// `pause`'s work, on `queue`.
    private func pauseOnQueue() {
        guard let timer else { return }
        timer.cancel()
        self.timer = nil
        startPauseGeneration += 1
        let generation = startPauseGeneration
        queue.asyncAfter(deadline: .now() + closingPollDelaySeconds) { [weak self] in
            self?.closingPollOnQueue(generation: generation)
        }
    }

    /// A pause's closing poll, on `queue`: skipped when a start or another
    /// pause came after it was scheduled.
    private func closingPollOnQueue(generation: Int) {
        guard startPauseGeneration == generation else { return }
        pollOnQueue()
        countFrom = nil
    }

    private func pollOnQueue() {
        guard let logSource, let readFrom, let countFrom else { return }
        let started = Date()
        do {
            let entries = try logSource.faultEntries(from: readFrom.addingTimeInterval(-Self.readOverlapSeconds))
            var newest = readFrom
            for polled in Self.newFaultEntries(entries, countFrom: countFrom, seen: &seen) {
                let entry = polled.entry
                if polled.counts {
                    SessionLogger.shared.log("[GPU-SYSLOG] \(Self.timestamp(entry.date)) \(entry.message)")
                    ledger.record(.systemLog(message: entry.message), at: entry.date)
                } else {
                    SessionLogger.shared.log("[GPU-SYSLOG] \(Self.timestamp(entry.date)) \(entry.message) "
                        + "(not counted: logged before this run's window, \(Self.timestamp(countFrom)))")
                }
                if entry.date > newest { newest = entry.date }
            }
            let nextReadFrom = max(newest, started.addingTimeInterval(-Self.readOverlapSeconds))
            self.readFrom = nextReadFrom
            // Forget entries no query can return again.
            let horizon = nextReadFrom.addingTimeInterval(-Self.seenHorizonSeconds)
            seen = seen.filter { $0.date >= horizon }
            if consecutiveFailures > 0, case .unavailable = availabilityBox.value {
                availabilityBox.value = .available
                SessionLogger.shared.log("[GPU-SYSLOG] available again after \(consecutiveFailures) failed queries")
            }
            consecutiveFailures = 0
        } catch {
            consecutiveFailures += 1
            if consecutiveFailures == Self.failuresBeforeUnavailable {
                markUnavailable("\(consecutiveFailures) queries in a row failed, the last with: "
                    + "\(error.localizedDescription); retrying at every poll")
            }
        }
        let elapsedMs = Date().timeIntervalSince(started) * 1000
        pollDurationsMs.modify { $0.append(elapsedMs) }
    }

    private func markUnavailable(_ reason: String) {
        availabilityBox.value = .unavailable(reason)
        SessionLogger.shared.log("[GPU-SYSLOG] unavailable: \(reason); until it recovers, GPU faults are detected "
            + "only by the checked submissions and the numeric guards")
    }

    private static func timestamp(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "yyyy-MM-dd HH:mm:ss.SSS"
        return formatter.string(from: date)
    }
}

/// Where `GPUFaultMonitor` reads macOS's log entries: this process's
/// `OSLogStore` in production, a scripted list in tests. Used only on the
/// monitor's queue.
protocol GPUFaultLogSource {
    /// The entries logged at or after `date` whose message may name a
    /// command-buffer fault (the monitor applies `isFaultMessage` itself),
    /// oldest first.
    func faultEntries(from date: Date) throws -> [GPUFaultMonitor.LogEntry]
}

/// This process's own system-log entries, readable without an entitlement.
struct ProcessLogStoreFaultSource: GPUFaultLogSource {
    private let store: OSLogStore

    init() throws {
        store = try OSLogStore(scope: .currentProcessIdentifier)
    }

    func faultEntries(from date: Date) throws -> [GPUFaultMonitor.LogEntry] {
        let position = store.position(date: date)
        let predicate = NSPredicate(format: "composedMessage CONTAINS %@", GPUFaultMonitor.faultMarker)
        var found: [GPUFaultMonitor.LogEntry] = []
        for case let entry as OSLogEntryLog in try store.getEntries(at: position, matching: predicate) {
            found.append(GPUFaultMonitor.LogEntry(date: entry.date, message: entry.composedMessage))
        }
        return found
    }
}
