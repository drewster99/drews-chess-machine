import Foundation
import os

/// How much work one GPU submission covers, for its command-buffer label,
/// its `[GPU-ERR]` / `[GPU-SLOW]` lines and the flight recorder.
///
/// Why the "newly compiled shape" case exists: batched inference and the
/// value baseline compile one `MPSGraphExecutable` per batch size, on first
/// use of that size. Arena ticks split their games between two networks in
/// sizes that change every tick, so an arena runs many first-time shapes
/// where self-play runs the same few over and over. The three GPU hangs of
/// 2026-10-09 all came from the run that plays arenas, each during one; a
/// recurrence recorded with this case tells whether the hung work was a
/// shape's first run.
enum GPUWorkSize: Sendable, Equatable, Encodable {
    /// The work isn't per position (weight load / export, state reads).
    case notPerPosition
    /// A forward or training pass over this many positions, through a
    /// graph or executable that has run at this size before.
    case positions(Int)
    /// A forward pass over this many positions, the first run of the
    /// executable compiled for this size.
    case positionsOnNewlyCompiledShape(Int)

    /// `n=37`, `n=37 new-shape`, or `n/a`.
    var label: String {
        switch self {
        case .notPerPosition: return "n/a"
        case .positions(let count): return "n=\(count)"
        case .positionsOnNewlyCompiledShape(let count): return "n=\(count) new-shape"
        }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(label)
    }
}

/// One GPU submission as the flight recorder saw it.
struct GPUSubmissionRecord: Sendable, Equatable, Encodable {
    enum Outcome: Sendable, Equatable, Encodable {
        /// Encoded; no completion reported yet.
        case running
        /// The completion handler reported success (the buffers' statuses
        /// are judged by `GPUSubmission.verify`, which marks a failure).
        case completed
        case failed(String)

        var label: String {
            switch self {
            case .running: return "running"
            case .completed: return "completed"
            case .failed(let detail): return "failed (\(detail))"
            }
        }

        func encode(to encoder: Encoder) throws {
            var container = encoder.singleValueContainer()
            try container.encode(label)
        }
    }

    /// 1, 2, 3, … in the order the submissions began encoding.
    let sequence: UInt64
    let stage: GPUStage
    /// The command queue's label: which network submitted the work (each
    /// network has its own queue, labelled by its owner, e.g. "champion
    /// (self-play)", "startrealTraining arena champion").
    let queue: String
    let work: GPUWorkSize
    /// When the encode began (MPS may commit split buffers during it).
    let startedAt: Date
    /// When the completion handler ran, or when `verify` gave up waiting for
    /// it; nil while running.
    var finishedAt: Date?
    var outcome: Outcome
    /// First buffer's GPU start to last buffer's GPU end, once verified.
    var gpuMilliseconds: Double?

    /// True when the submission was running at some moment in `window`.
    func overlaps(_ window: ClosedRange<Date>) -> Bool {
        startedAt <= window.upperBound && (finishedAt.map { $0 >= window.lowerBound } ?? true)
    }

    private enum CodingKeys: String, CodingKey {
        case sequence, stage, queue, work
        case startedAt = "started_at"
        case finishedAt = "finished_at"
        case outcome
        case gpuMilliseconds = "gpu_ms"
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(sequence, forKey: .sequence)
        try container.encode(stage.rawValue, forKey: .stage)
        try container.encode(queue, forKey: .queue)
        try container.encode(work, forKey: .work)
        try container.encode(GPUFlightRecorder.timestamp(startedAt), forKey: .startedAt)
        try container.encode(finishedAt.map(GPUFlightRecorder.timestamp), forKey: .finishedAt)
        try container.encode(outcome, forKey: .outcome)
        try container.encode(gpuMilliseconds, forKey: .gpuMilliseconds)
    }
}

/// The last couple of minutes of this process's GPU submissions — which
/// network, which stage, how many positions, when each began and finished —
/// so a GPU fault can be matched to the work that was on the GPU when it
/// happened (GPU fault forensics plan, owner request 2026-10-10).
///
/// Why: macOS's fault report names the guilty process but not the work, and
/// by the time the app learns of a fault (a failed check, or the system-log
/// monitor up to one poll later) every submission involved has finished with
/// an error. A list of what is in flight *now* would be empty; this keeps
/// recent history and answers "what was running at time T".
///
/// On each fault the ledger records, `logAccount(for:)` writes
/// `[GPU-INFLIGHT]` lines: every submission running in the
/// `accountWindowSeconds` before the fault (a hung submission is one that
/// started long before and finished at the reset, or never), and a
/// per-network / per-stage summary of the `summaryWindowSeconds` before it.
/// Faults within `coalesceSeconds` of the last accounted one (one reset is
/// several system-log entries and a failed check) get one line pointing back.
/// Crash dumps carry the whole history (`gpu-submissions.json`).
///
/// Cost: two lock sections and two `Date()` reads per submission; the
/// history is a ring bounded by `retentionSeconds` and `capacity`.
final class GPUFlightRecorder: @unchecked Sendable {
    static let shared = GPUFlightRecorder()

    /// Finished submissions older than this are dropped.
    static let retentionSeconds: TimeInterval = 120
    /// Hard bound on the history; the oldest records go first, running or
    /// not (a submission that never finishes must not grow it forever).
    static let capacity = 20_000
    /// Submissions running in this span before a fault are listed one by one.
    static let accountWindowSeconds: TimeInterval = 2
    /// Most submissions listed one by one per fault; the rest are counted.
    static let accountListLimit = 64
    /// The span before a fault the per-network summary covers.
    static let summaryWindowSeconds: TimeInterval = 10
    /// A fault this close to the last accounted one is not accounted again.
    static let coalesceSeconds: TimeInterval = 5

    private struct State {
        /// Records in sequence order; `records[head...]` are live.
        var records: [GPUSubmissionRecord] = []
        var head = 0
        var nextSequence: UInt64 = 1
        /// Records evicted by `capacity` while still running.
        var evictedWhileRunning = 0
        var lastAccounted: (sequence: Int, time: Date)?
    }

    private let state = OSAllocatedUnfairLock<State>(initialState: State())

    /// A recorder of its own, for tests; production code uses `shared`.
    init() {}

    /// Records a submission whose encode is about to begin; returns its
    /// sequence number for `finish`.
    func begin(stage: GPUStage, queue: String, work: GPUWorkSize, at time: Date = Date()) -> UInt64 {
        state.withLock { state in
            let sequence = state.nextSequence
            state.nextSequence += 1
            state.records.append(GPUSubmissionRecord(
                sequence: sequence, stage: stage, queue: queue, work: work,
                startedAt: time, finishedAt: nil, outcome: .running, gpuMilliseconds: nil))
            Self.prune(&state, now: time)
            return sequence
        }
    }

    /// Marks a submission finished. A later call refines an earlier one:
    /// the completion handler reports first, `verify` adds the GPU time and,
    /// on a failure, its account. The first finish time is kept.
    func finish(_ sequence: UInt64, outcome: GPUSubmissionRecord.Outcome, gpuMilliseconds: Double? = nil,
                at time: Date = Date()) {
        state.withLock { state in
            guard let first = state.records[state.head...].first?.sequence, sequence >= first else { return }
            let index = state.head + Int(sequence - first)
            guard index < state.records.count, state.records[index].sequence == sequence else { return }
            if state.records[index].finishedAt == nil {
                state.records[index].finishedAt = time
            }
            // A failure stays a failure: the handler's success can't
            // overwrite what `verify` found in the buffers.
            let alreadyFailed: Bool
            if case .failed = state.records[index].outcome { alreadyFailed = true } else { alreadyFailed = false }
            if !alreadyFailed {
                state.records[index].outcome = outcome
            }
            if let gpuMilliseconds {
                state.records[index].gpuMilliseconds = gpuMilliseconds
            }
        }
    }

    /// Every retained record, oldest first.
    func snapshot() -> [GPUSubmissionRecord] {
        state.withLock { Array($0.records[$0.head...]) }
    }

    /// How many records `capacity` evicted while they were still running.
    var evictedWhileRunning: Int {
        state.withLock { $0.evictedWhileRunning }
    }

    /// Logs the `[GPU-INFLIGHT]` account of `fault` (see the type doc).
    func logAccount(for fault: GPUFaultLedger.Fault) {
        for line in account(for: fault) {
            SessionLogger.shared.log(line)
        }
    }

    /// The `[GPU-INFLIGHT]` lines for `fault`; marks it accounted.
    func account(for fault: GPUFaultLedger.Fault) -> [String] {
        let (records, previous): ([GPUSubmissionRecord], (sequence: Int, time: Date)?) = state.withLock { state in
            let previous = state.lastAccounted
            if let previous, abs(fault.time.timeIntervalSince(previous.time)) < Self.coalesceSeconds {
                return ([], previous)
            }
            state.lastAccounted = (fault.sequence, fault.time)
            return (Array(state.records[state.head...]), nil)
        }
        let when = Self.timestamp(fault.time)
        if let previous {
            return ["[GPU-INFLIGHT] fault #\(fault.sequence) at \(when): within \(Int(Self.coalesceSeconds)) s of "
                    + "fault #\(previous.sequence) (\(Self.timestamp(previous.time))), see its account"]
        }
        let window = fault.time.addingTimeInterval(-Self.accountWindowSeconds)...fault.time
        let running = records.filter { $0.overlaps(window) }
        var lines = ["[GPU-INFLIGHT] fault #\(fault.sequence) at \(when): \(running.count) submission(s) running in the "
                     + "\(Self.format(Self.accountWindowSeconds)) s before it (\(records.count) recorded since "
                     + "\(records.first.map { Self.timestamp($0.startedAt) } ?? "none"))"]
        for record in running.prefix(Self.accountListLimit) {
            lines.append("[GPU-INFLIGHT]   " + Self.line(for: record, relativeTo: fault.time))
        }
        if running.count > Self.accountListLimit {
            lines.append("[GPU-INFLIGHT]   … \(running.count - Self.accountListLimit) more (crash dump has all)")
        }
        let summaryWindow = fault.time.addingTimeInterval(-Self.summaryWindowSeconds)...fault.time
        lines += Self.summaryLines(records.filter { $0.overlaps(summaryWindow) })
        return lines
    }

    // MARK: - Private

    private static func prune(_ state: inout State, now: Date) {
        let horizon = now.addingTimeInterval(-retentionSeconds)
        while state.head < state.records.count {
            let record = state.records[state.head]
            let overCapacity = state.records.count - state.head > capacity
            let oldAndDone = record.finishedAt.map { $0 < horizon } ?? false
            guard overCapacity || oldAndDone else { break }
            if record.finishedAt == nil { state.evictedWhileRunning += 1 }
            state.head += 1
        }
        // Compact once the dead prefix is large, so the array doesn't grow
        // without bound and the copy is amortized.
        if state.head > 4_096, state.head * 2 > state.records.count {
            state.records.removeFirst(state.head)
            state.head = 0
        }
    }

    /// `seq=… stage=… queue="…" n=… started=-1234ms ran=56ms outcome=…`;
    /// times relative to the fault.
    private static func line(for record: GPUSubmissionRecord, relativeTo time: Date) -> String {
        let started = Int((record.startedAt.timeIntervalSince(time) * 1000).rounded())
        let ran: String
        if let finished = record.finishedAt {
            ran = "ran=\(Int((finished.timeIntervalSince(record.startedAt) * 1000).rounded()))ms"
        } else {
            ran = "still-running"
        }
        let gpu = record.gpuMilliseconds.map { " gpuMs=\(String(format: "%.0f", $0))" } ?? ""
        return "seq=\(record.sequence) stage=\(record.stage.rawValue) queue=\"\(record.queue)\" \(record.work.label) "
            + "started=\(started)ms \(ran)\(gpu) outcome=\(record.outcome.label)"
    }

    /// One line per (queue, stage) over `records`: count, sizes, new shapes,
    /// longest run.
    private static func summaryLines(_ records: [GPUSubmissionRecord]) -> [String] {
        struct Key: Hashable { let queue: String; let stage: GPUStage }
        let groups = Dictionary(grouping: records) { Key(queue: $0.queue, stage: $0.stage) }
        let header = "[GPU-INFLIGHT] last \(format(summaryWindowSeconds)) s by network and stage:"
        let body = groups.keys.sorted { ($0.queue, $0.stage.rawValue) < ($1.queue, $1.stage.rawValue) }.map { key in
            guard let group = groups[key] else { return "" }
            var sizes: [Int] = []
            var newShapes = 0
            for record in group {
                switch record.work {
                case .notPerPosition: break
                case .positions(let count): sizes.append(count)
                case .positionsOnNewlyCompiledShape(let count): sizes.append(count); newShapes += 1
                }
            }
            let distinct = Array(Set(sizes)).sorted()
            let sizeText = distinct.isEmpty ? "" : " sizes=[\(distinct.prefix(24).map { String($0) }.joined(separator: ","))"
                + (distinct.count > 24 ? ",…" : "") + "]"
            let longest = group.compactMap { record in
                record.finishedAt.map { $0.timeIntervalSince(record.startedAt) * 1000 }
            }.max()
            let running = group.filter { $0.finishedAt == nil }.count
            return "[GPU-INFLIGHT]   queue=\"\(key.queue)\" stage=\(key.stage.rawValue) count=\(group.count)\(sizeText)"
                + " newShapes=\(newShapes) longestMs=\(longest.map { String(format: "%.0f", $0) } ?? "n/a")"
                + (running > 0 ? " stillRunning=\(running)" : "")
        }
        return [header] + body
    }

    private static func format(_ seconds: TimeInterval) -> String {
        String(format: "%g", seconds)
    }

    /// Local wall-clock time with milliseconds, matching the session log.
    static func timestamp(_ date: Date) -> String {
        timestampFormatter.string(from: date)
    }

    private static let timestampFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "yyyy-MM-dd HH:mm:ss.SSS"
        return formatter
    }()
}
