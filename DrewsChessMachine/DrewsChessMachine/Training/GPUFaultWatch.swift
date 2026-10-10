import Foundation

/// One training run's view of `GPUFaultLedger`: the faults recorded since the
/// run began, and the barrier every training path runs before it makes state
/// durable or authoritative (GPU fault forensics plan, A3 and A5).
///
/// The rule it serves: a GPU fault while the trainer is active means the
/// weights, optimizer state or RNG state may be wrong in ways nothing can
/// locate — a GPU reset discards command buffers the app can't see, and the
/// system-log layer reports them seconds late. So a run stops training at the
/// first fault it sees, and before every save, promotion or pointer update it
/// runs `barrier()`: a fresh read of macOS's fault messages, so a fault that
/// happened moments earlier is not missed and the last save a run writes is
/// always one written before any fault.
struct GPUFaultWatch: Sendable {
    let ledger: GPUFaultLedger
    let monitor: GPUFaultMonitor
    /// The ledger's newest sequence when the run began; faults after it are
    /// this run's.
    let baselineSequence: Int

    /// Starts the process's fault and memory monitors (once per process) and
    /// begins watching from the ledger's current end.
    static func startForRun() -> GPUFaultWatch {
        GPUFaultMonitor.shared.start()
        MemoryStatusMonitor.shared.start()
        return GPUFaultWatch(ledger: .shared, monitor: .shared)
    }

    init(ledger: GPUFaultLedger, monitor: GPUFaultMonitor) {
        self.ledger = ledger
        self.monitor = monitor
        self.baselineSequence = ledger.latestSequence
    }

    /// The faults recorded since the run began, oldest first. In memory; no
    /// log read.
    var faultsSinceStart: [GPUFaultLedger.Fault] {
        ledger.faults(after: baselineSequence)
    }

    /// The first fault since the run began, if any.
    var firstFault: GPUFaultLedger.Fault? {
        faultsSinceStart.first
    }

    /// Reads macOS's fault messages now (≈2 s on a busy machine) and returns
    /// the first fault since the run began, if any. Await it before a save,
    /// a promotion or a pointer update, and refuse the action on a fault.
    func barrier() async -> GPUFaultLedger.Fault? {
        await monitor.checkNow()
        return firstFault
    }

    /// What `results.json` records under `gpu_faults`.
    func report(stoppedAtTrainerStep: Int?, crashDumps: [String]) -> GPUFaultReport {
        GPUFaultReport(
            monitor: monitor.availability.label,
            faults: faultsSinceStart.map(GPUFaultReport.Fault.init),
            stoppedAtTrainerStep: stoppedAtTrainerStep,
            crashDumps: crashDumps)
    }
}

/// `results.json`'s `gpu_faults`: whether the system-log monitor worked, every
/// fault the run saw, where training stopped for one, and the crash dumps
/// written. Present on every run that started a `GPUFaultWatch`, so a run
/// with no faults says so explicitly (with the monitor's availability).
struct GPUFaultReport: Encodable, Sendable, Equatable {
    struct Fault: Encodable, Sendable, Equatable {
        let sequence: Int
        /// ISO 8601 with milliseconds.
        let time: String
        let source: String
        let detail: String

        init(_ fault: GPUFaultLedger.Fault) {
            sequence = fault.sequence
            time = Self.iso8601(fault.time)
            switch fault.source {
            case .submission(let stage, let detail):
                source = "submission: \(stage)"
                self.detail = detail
            case .systemLog(let message):
                source = "system log"
                self.detail = message
            }
        }

        private static func iso8601(_ date: Date) -> String {
            let formatter = ISO8601DateFormatter()
            formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
            return formatter.string(from: date)
        }
    }

    /// `available`, `unavailable: <reason>` or `not started`.
    let monitor: String
    let faults: [Fault]
    /// The trainer step at which training stopped for a fault; nil when it
    /// didn't.
    let stoppedAtTrainerStep: Int?
    /// Crash-dump folders written by the run.
    let crashDumps: [String]

    enum CodingKeys: String, CodingKey {
        case monitor, faults
        case stoppedAtTrainerStep = "stopped_at_trainer_step"
        case crashDumps = "crash_dumps"
    }
}
