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
final class GPUFaultMonitor: @unchecked Sendable {
    static let shared = GPUFaultMonitor(ledger: .shared)

    /// How often the background poll reads the log, in seconds.
    static let pollIntervalSeconds = 10

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

    /// Whether a log message is a GPU command-buffer fault. Pure; the
    /// incidents' captured messages are its test fixtures.
    static func isFaultMessage(_ message: String) -> Bool {
        message.contains(faultMarker)
    }

    private let ledger: GPUFaultLedger
    /// Every poll and all mutable state below run on this queue.
    private let queue = DispatchQueue(label: "drewschessmachine.gpu-fault-monitor", qos: .utility)
    private let availabilityBox = SyncBox<Availability>(.notStarted)
    /// Poll durations since the last `takePollDurations()`, for the `[MEM]`
    /// line (so the monitor's own cost is visible in a long-lived process).
    private let pollDurationsMs = SyncBox<[Double]>([])

    // Queue-confined state.
    private var store: OSLogStore?
    private var timer: DispatchSourceTimer?
    /// Read entries from here on (with a margin; `seen` drops repeats).
    private var readFrom = Date()
    /// Entries already recorded: log time and message.
    private var seen = Set<SeenEntry>()
    /// Queries that failed since the last success.
    private var consecutiveFailures = 0

    private struct SeenEntry: Hashable {
        let date: Date
        let message: String
    }

    init(ledger: GPUFaultLedger) {
        self.ledger = ledger
    }

    var availability: Availability {
        availabilityBox.value
    }

    /// A query that fails this many times in a row marks the monitor
    /// unavailable (logged); it keeps retrying at every poll and marks
    /// itself available again on the next success.
    static let failuresBeforeUnavailable = 3

    /// Opens the store (on the monitor's queue, so a caller on the main actor
    /// never waits) and starts or resumes the background poll. Idempotent.
    /// Reads from `lookBack` before now on first start, so a fault just before
    /// the run started is still seen.
    func start(lookBack: TimeInterval = 60) {
        queue.async { [weak self] in
            self?.startOnQueue(lookBack: lookBack)
        }
    }

    /// `start`'s work, on `queue`.
    private func startOnQueue(lookBack: TimeInterval) {
        if self.store == nil {
            self.readFrom = Date().addingTimeInterval(-lookBack)
            do {
                self.store = try OSLogStore(scope: .currentProcessIdentifier)
                self.availabilityBox.value = .available
                SessionLogger.shared.log(
                    "[GPU-SYSLOG] monitoring this process's GPU fault messages every \(Self.pollIntervalSeconds) s")
            } catch {
                self.markUnavailable("could not open the log store: \(error.localizedDescription)")
                return
            }
        }
        guard self.timer == nil else { return }
        let timer = DispatchSource.makeTimerSource(queue: self.queue)
        let interval = DispatchTimeInterval.seconds(Self.pollIntervalSeconds)
        timer.schedule(deadline: .now() + interval, repeating: interval, leeway: .seconds(1))
        timer.setEventHandler { [weak self] in
            self?.pollOnQueue()
        }
        timer.resume()
        self.timer = timer
    }

    /// Stops the background poll until the next `start()` — between GUI
    /// runs, when no training is left to protect, so the ≈2 s query every
    /// 10 s isn't spent for nothing. `checkNow()` still works.
    func pause() {
        queue.async {
            self.timer?.cancel()
            self.timer = nil
        }
    }

    /// Runs one poll now, on the monitor's queue, and returns once it has
    /// recorded whatever it found. The barrier before a save or a promotion:
    /// afterwards the ledger holds every fault macOS had logged for this
    /// process. Does nothing (beyond returning) when the monitor isn't
    /// available.
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

    private func pollOnQueue() {
        guard let store else { return }
        let started = Date()
        // Overlap the previous read by a few seconds: an entry logged just
        // before the last query ran may not have been visible to it.
        let from = readFrom.addingTimeInterval(-5)
        do {
            let position = store.position(date: from)
            let predicate = NSPredicate(format: "composedMessage CONTAINS %@", Self.faultMarker)
            let entries = try store.getEntries(at: position, matching: predicate)
            var newest = readFrom
            for case let entry as OSLogEntryLog in entries {
                let message = entry.composedMessage
                guard Self.isFaultMessage(message) else { continue }
                let key = SeenEntry(date: entry.date, message: message)
                guard seen.insert(key).inserted else { continue }
                SessionLogger.shared.log("[GPU-SYSLOG] \(Self.timestamp(entry.date)) \(message)")
                ledger.record(.systemLog(message: message), at: entry.date)
                if entry.date > newest { newest = entry.date }
            }
            readFrom = max(newest, started.addingTimeInterval(-5))
            // Forget entries older than the overlap window: they can't be
            // returned again.
            let horizon = readFrom.addingTimeInterval(-60)
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
