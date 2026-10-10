import Foundation
import Metal
import MetalPerformanceShaders
import MetalPerformanceShadersGraph
import os

/// One GPU submission — an MPSGraph or MPSGraphExecutable encode into a
/// command buffer this type creates — and the one place that decides whether
/// the GPU actually did the work.
///
/// Why this exists (GPU fault forensics plan, Part A; the 2026-10-09 hangs):
/// `waitUntilCompleted` returns whether or not the GPU finished the work. A
/// GPU hang in one process makes macOS reset the GPU and discard every other
/// in-flight command buffer ("victim of GPU error/recovery"), in every process
/// sharing it; a discarded buffer's outputs are whatever the memory held. The
/// app used to read those outputs as if they were real: a NaN gradient in one
/// run and a 10.6 million gradient norm in another, with nothing in either
/// session log.
///
/// No single public signal covers a submission completely:
///
/// - MPSGraph splits large work across several command buffers
///   (`commitAndContinue`). The buffer created here is the first; after a
///   split `MPSCommandBuffer.commandBuffer` names the newest (last). The
///   buffers in between are not reachable through any public API (call audit,
///   `GPU_CALL_AUDIT_2026-10-09.md`, verified by probe).
/// - The execution descriptor's completion handler reports an `NSError` for
///   the execution as a whole. Whether that error covers a middle buffer, or
///   a hang / victim at all, could not be verified without forcing a fault.
///
/// So this checks everything reachable — the first buffer's and the last
/// buffer's status and error, and the completion handler's error — and the
/// process-wide `GPUFaultMonitor` (which reads macOS's own fault messages for
/// this process) catches what none of them reports.
///
/// Usage: create, encode with `graphExecutionDescriptor` or
/// `executableExecutionDescriptor` (each wires the completion handler; use
/// exactly one, once), `commit()`, then `verify()` — immediately, or
/// later for work that overlaps the next submission (the value baseline).
/// Inside a lock section that must not throw, call `waitUntilCompleted()`
/// there and `verify()` after the unlock.
/// A failure logs one `[GPU-ERR]` line and throws
/// `ChessNetworkError.gpuCommandFailed` naming the stage.
///
/// Thread safety: one owner drives create → encode → commit → wait in order;
/// the completion handler runs on a Metal thread and only writes
/// `handlerOutcome` (lock-guarded) and signals `handlerFired`.
final class GPUSubmission: @unchecked Sendable {
    /// What the work was, for logs and errors.
    let stage: GPUStage
    /// The `MPSCommandBuffer` to encode into. After `commit()`, its
    /// `commandBuffer` is the last buffer of the submission.
    let commandBuffer: MPSCommandBuffer
    /// The buffer this type created: the submission's first.
    private let firstBuffer: MTLCommandBuffer
    /// The completion handler's report, once it has run.
    private let handlerOutcome = OSAllocatedUnfairLock<HandlerOutcome>(initialState: .pending)
    /// Signalled once by the completion handler.
    private let handlerFired = DispatchSemaphore(value: 0)
    /// Guards the one-descriptor rule and the commit / wait order.
    private let lifecycle = OSAllocatedUnfairLock<Lifecycle>(initialState: Lifecycle())

    /// How long `verify` waits for the completion handler after the
    /// last buffer has completed. MPS calls it from the last buffer's
    /// completion, so it normally lands within milliseconds; not hearing
    /// from it at all means the submission's outcome is unknown, which is a
    /// failure.
    static let completionHandlerGrace: DispatchTimeInterval = .seconds(10)

    /// A submission whose first or last command buffer ran this long on the
    /// GPU logs `[GPU-SLOW]`. macOS's GPU watchdog ends a buffer that runs too
    /// long with a hang error and resets the GPU, and it acts per buffer —
    /// so the test is per buffer, not the submission's span (which includes
    /// the CPU encode time between MPS's split buffers). The slow line names
    /// the stage that came closest.
    static let slowBufferMs: Double = 5_000

    enum HandlerOutcome: Sendable, Equatable {
        case pending
        case succeeded
        case failed(GPUErrorDescription)
    }

    private struct Lifecycle {
        var descriptorHandedOut = false
        var committed = false
        var verified = false
    }

    /// Creates the submission's first command buffer on `queue`, with
    /// encoder execution status reporting on (it covers this first buffer
    /// only: MPS-created continuation buffers don't inherit it).
    init(queue: MTLCommandQueue, stage: GPUStage) throws {
        let descriptor = MTLCommandBufferDescriptor()
        descriptor.errorOptions = .encoderExecutionStatus
        guard let buffer = queue.makeCommandBuffer(descriptor: descriptor) else {
            // No command buffer is a GPU failure like any other: logged,
            // recorded, and thrown as the one GPU-failure error.
            let detail = "the command queue returned no command buffer"
            SessionLogger.shared.log("[GPU-ERR] stage=\(stage.rawValue) \(detail)")
            GPUFaultLedger.shared.record(.submission(stage: stage.rawValue, detail: detail))
            throw ChessNetworkError.gpuCommandFailed(stage: stage.rawValue, status: .notEnqueued, error: detail)
        }
        buffer.label = stage.rawValue
        self.stage = stage
        self.firstBuffer = buffer
        self.commandBuffer = MPSCommandBuffer(commandBuffer: buffer)
    }

    /// An `MPSGraph.encode` descriptor whose completion handler reports to
    /// this submission. Hand out one descriptor per submission.
    var graphExecutionDescriptor: MPSGraphExecutionDescriptor {
        claimDescriptor()
        let descriptor = MPSGraphExecutionDescriptor()
        descriptor.completionHandler = { [handlerOutcome, handlerFired] _, error in
            Self.record(error, in: handlerOutcome, signalling: handlerFired)
        }
        return descriptor
    }

    /// An `MPSGraphExecutable.encode` descriptor whose completion handler
    /// reports to this submission. Hand out one descriptor per submission.
    var executableExecutionDescriptor: MPSGraphExecutableExecutionDescriptor {
        claimDescriptor()
        let descriptor = MPSGraphExecutableExecutionDescriptor()
        descriptor.completionHandler = { [handlerOutcome, handlerFired] _, error in
            Self.record(error, in: handlerOutcome, signalling: handlerFired)
        }
        return descriptor
    }

    /// Commits the submission's current (last) buffer.
    func commit() {
        lifecycle.withLock { state in
            precondition(state.descriptorHandedOut, "GPUSubmission(\(stage.rawValue)): commit before any encode")
            precondition(!state.committed, "GPUSubmission(\(stage.rawValue)): committed twice")
            state.committed = true
        }
        commandBuffer.commit()
    }

    /// Waits until the submission's last command buffer (and so every
    /// earlier one) has finished, without judging the outcome. Never throws,
    /// so it can sit inside a lock section that must reach its unlock
    /// (`ChessNetwork.weightAccessLock` around the training step); call
    /// `verify()` after the lock is released.
    func waitUntilCompleted() {
        lifecycle.withLock { state in
            precondition(state.committed, "GPUSubmission(\(stage.rawValue)): wait before commit")
        }
        commandBuffer.commandBuffer.waitUntilCompleted()
        // The first buffer was committed before the last (by MPS's split, or
        // it is the last); waiting on it too costs nothing and makes its
        // status final before `verify` reads it.
        firstBuffer.waitUntilCompleted()
    }

    /// Throws unless every reachable signal says the submission completed:
    /// first and last buffer `.completed`, and the completion handler
    /// reporting success. Waits for the buffers first if they haven't
    /// finished. Returns the submission's GPU time in milliseconds (first
    /// buffer's start to last buffer's end; 0 when the GPU reported none).
    @discardableResult
    func verify() throws -> Double {
        lifecycle.withLock { state in
            precondition(state.committed, "GPUSubmission(\(stage.rawValue)): verify before commit")
            precondition(!state.verified, "GPUSubmission(\(stage.rawValue)): verified twice")
            state.verified = true
        }
        waitUntilCompleted()
        let lastBuffer = commandBuffer.commandBuffer
        let handler: HandlerOutcome
        if handlerFired.wait(timeout: .now() + Self.completionHandlerGrace) == .timedOut {
            handler = .pending
        } else {
            handler = handlerOutcome.withLock { $0 }
        }
        let report = GPUSubmissionReport(
            stage: stage,
            firstStatus: firstBuffer.status,
            firstError: firstBuffer.error.map(GPUErrorDescription.init),
            lastStatus: lastBuffer.status,
            lastError: lastBuffer.error.map(GPUErrorDescription.init),
            splitIntoSeveralBuffers: lastBuffer !== firstBuffer,
            handler: handler,
            gpuMilliseconds: Self.milliseconds(from: firstBuffer.gpuStartTime, to: lastBuffer.gpuEndTime),
            firstBufferMilliseconds: Self.milliseconds(from: firstBuffer.gpuStartTime, to: firstBuffer.gpuEndTime),
            lastBufferMilliseconds: Self.milliseconds(from: lastBuffer.gpuStartTime, to: lastBuffer.gpuEndTime)
        )
        if let failure = report.failure {
            SessionLogger.shared.log(report.logLine)
            GPUFaultLedger.shared.record(.submission(stage: stage.rawValue, detail: failure.detail))
            throw ChessNetworkError.gpuCommandFailed(stage: stage.rawValue, status: failure.status, error: failure.detail)
        }
        if let slowLine = report.slowLine(thresholdMs: Self.slowBufferMs) {
            SessionLogger.shared.log(slowLine)
        }
        return report.gpuMilliseconds ?? 0
    }

    /// Encodes `executable` with `inputs` into a new submission on `queue`,
    /// commits, waits and verifies — the checked replacement for
    /// `MPSGraphExecutable.run`, which reports no GPU failure.
    static func runExecutable(
        _ executable: MPSGraphExecutable,
        on queue: MTLCommandQueue,
        inputs: [MPSGraphTensorData],
        results: [MPSGraphTensorData]?,
        stage: GPUStage
    ) throws -> [MPSGraphTensorData] {
        let submission = try GPUSubmission(queue: queue, stage: stage)
        let outputs = executable.encode(
            to: submission.commandBuffer,
            inputs: inputs,
            results: results,
            executionDescriptor: submission.executableExecutionDescriptor
        )
        submission.commit()
        try submission.verify()
        return outputs
    }

    /// Encodes `graph` for `feeds` into a new submission on `queue`, commits,
    /// waits and verifies — the checked replacement for the synchronous
    /// `MPSGraph.run(with:feeds:targetTensors:targetOperations:)`, which is
    /// encode + commit + wait inside and reports no GPU failure.
    static func runGraph(
        _ graph: MPSGraph,
        on queue: MTLCommandQueue,
        feeds: [MPSGraphTensor: MPSGraphTensorData],
        targetTensors: [MPSGraphTensor],
        targetOperations: [MPSGraphOperation]?,
        stage: GPUStage
    ) throws -> [MPSGraphTensor: MPSGraphTensorData] {
        let submission = try GPUSubmission(queue: queue, stage: stage)
        let outputs = graph.encode(
            to: submission.commandBuffer,
            feeds: feeds,
            targetTensors: targetTensors,
            targetOperations: targetOperations,
            executionDescriptor: submission.graphExecutionDescriptor
        )
        submission.commit()
        try submission.verify()
        return outputs
    }

    // MARK: - Private

    private func claimDescriptor() {
        lifecycle.withLock { state in
            precondition(!state.descriptorHandedOut, "GPUSubmission(\(stage.rawValue)): one execution descriptor per submission")
            state.descriptorHandedOut = true
        }
    }

    private static func record(
        _ error: Error?,
        in outcome: OSAllocatedUnfairLock<HandlerOutcome>,
        signalling fired: DispatchSemaphore
    ) {
        let reported: HandlerOutcome = error.map { .failed(GPUErrorDescription($0)) } ?? .succeeded
        let first = outcome.withLock { state -> Bool in
            guard state == .pending else { return false }
            state = reported
            return true
        }
        // MPS calls the handler once per execution; a second call would be a
        // framework change worth seeing, but it must not over-signal.
        if first {
            fired.signal()
        } else {
            SessionLogger.shared.log("[GPU-ERR] completion handler called again: \(reported)")
        }
    }

    /// Milliseconds between two GPU timestamps; nil when either is 0 (a
    /// buffer that never ran on the GPU — discarded before it was scheduled,
    /// or failed) or they are out of order.
    private static func milliseconds(from start: CFTimeInterval, to end: CFTimeInterval) -> Double? {
        guard start > 0, end >= start else { return nil }
        return (end - start) * 1000
    }
}

/// Every kind of GPU work the app submits — the `stage` a `[GPU-ERR]`
/// line, a `[GPU-SLOW]` line and a `gpuCommandFailed` error name.
enum GPUStage: String, Sendable, CaseIterable {
    case trainingStep = "training step"
    case workingWeightSync = "working-weight sync"
    case valueBaseline = "value baseline"
    case klProbe = "KL probe"
    case dropoutAdvance = "dropout RNG advance"
    case weightLoad = "weight load"
    case weightExport = "weight export"
    case batchedInference = "batched inference"
    case singleInference = "single-position inference"
    case valueDistribution = "value distribution"
    case analysisTaps = "analysis taps"
    case bnBatchStats = "BN batch statistics"
    case bnRunningStatsLoad = "BN running-statistics load"
    case velocityRead = "velocity read"
    case velocityWrite = "velocity write"
    case masterRead = "fp32 master read"
    case masterWrite = "fp32 master write"
    case trainableVelocityRead = "trainable velocity read"
    case layerHealthRead = "layer-health read"
    case syncMastersFromWorking = "master sync from working weights"
    case dropoutStateWrite = "dropout state write"
    case dropoutStateRead = "dropout state read"
    case dropoutStateDerive = "dropout state derivation"
}

/// An `NSError` from Metal or MPSGraph, kept as the facts a log line and a
/// crash dump need: domain, code, description and, for a command buffer
/// with encoder execution status on, each encoder's state.
struct GPUErrorDescription: Sendable, Equatable, CustomStringConvertible {
    let domain: String
    let code: Int
    let message: String
    /// `label:state` per encoder, from `MTLCommandBufferEncoderInfoErrorKey`;
    /// empty when the buffer had no encoder info.
    let encoderStates: [String]

    init(domain: String, code: Int, message: String, encoderStates: [String]) {
        self.domain = domain
        self.code = code
        self.message = message
        self.encoderStates = encoderStates
    }

    init(_ error: Error) {
        let nsError = error as NSError
        let infos = nsError.userInfo[MTLCommandBufferEncoderInfoErrorKey] as? [MTLCommandBufferEncoderInfo] ?? []
        self.init(
            domain: nsError.domain,
            code: nsError.code,
            message: nsError.localizedDescription,
            encoderStates: infos.map { "\($0.label):\(Self.name(of: $0.errorState))" }
        )
    }

    var description: String {
        let encoders = encoderStates.isEmpty ? "" : " encoders=[\(encoderStates.joined(separator: ","))]"
        return "\(domain)#\(code) \"\(message)\"\(encoders)"
    }

    private static func name(of state: MTLCommandEncoderErrorState) -> String {
        switch state {
        case .unknown: return "unknown"
        case .completed: return "completed"
        case .affected: return "affected"
        case .pending: return "pending"
        case .faulted: return "faulted"
        @unknown default: return "state\(state.rawValue)"
        }
    }
}

/// Everything `GPUSubmission.verify` read about one submission, and
/// the one rule that decides whether it failed. Pure, so the rule is tested
/// without a GPU fault.
struct GPUSubmissionReport: Sendable, Equatable {
    let stage: GPUStage
    let firstStatus: MTLCommandBufferStatus
    let firstError: GPUErrorDescription?
    let lastStatus: MTLCommandBufferStatus
    let lastError: GPUErrorDescription?
    let splitIntoSeveralBuffers: Bool
    let handler: GPUSubmission.HandlerOutcome
    /// First buffer's GPU start to last buffer's GPU end (includes the CPU
    /// encode time between split buffers).
    let gpuMilliseconds: Double?
    /// The first and last buffers' own GPU times (equal when not split).
    let firstBufferMilliseconds: Double?
    let lastBufferMilliseconds: Double?

    struct Failure: Sendable, Equatable {
        /// The status the error reports: the first non-completed of the
        /// last and first buffers, else `.completed` (only the handler
        /// failed).
        let status: MTLCommandBufferStatus
        /// Every signal's account, for the error text and the log.
        let detail: String
    }

    /// Non-nil when any reachable signal says the work did not complete:
    /// either buffer not `.completed` or carrying an error, the handler
    /// reporting an error, or the handler never reporting.
    var failure: Failure? {
        let buffersCompleted = Self.bufferCompleted(firstStatus) && Self.bufferCompleted(lastStatus)
        let handlerSucceeded = handler == .succeeded
        guard !buffersCompleted || !handlerSucceeded else { return nil }
        let status: MTLCommandBufferStatus
        if lastStatus != .completed {
            status = lastStatus
        } else if firstStatus != .completed {
            status = firstStatus
        } else {
            status = .completed
        }
        return Failure(status: status, detail: signals)
    }

    /// `[GPU-SLOW]` when the first or last buffer's own GPU time reaches
    /// `thresholdMs`; nil otherwise.
    func slowLine(thresholdMs: Double) -> String? {
        let longest = [firstBufferMilliseconds, lastBufferMilliseconds].compactMap { $0 }.max()
        guard let longest, longest >= thresholdMs else { return nil }
        func ms(_ value: Double?) -> String { value.map { String(format: "%.0f", $0) } ?? "n/a" }
        return "[GPU-SLOW] stage=\(stage.rawValue) firstMs=\(ms(firstBufferMilliseconds))"
            + " lastMs=\(splitIntoSeveralBuffers ? ms(lastBufferMilliseconds) : "same") spanMs=\(ms(gpuMilliseconds))"
    }

    /// One `[GPU-ERR]` line with every signal.
    var logLine: String {
        "[GPU-ERR] stage=\(stage.rawValue) \(signals)"
    }

    private var signals: String {
        var parts = [
            "first=\(Self.name(of: firstStatus))" + (firstError.map { " (\($0))" } ?? ""),
            "last=\(splitIntoSeveralBuffers ? Self.name(of: lastStatus) : "same")"
                + (splitIntoSeveralBuffers ? (lastError.map { " (\($0))" } ?? "") : ""),
        ]
        switch handler {
        case .pending:
            parts.append("handler=did-not-report")
        case .succeeded:
            parts.append("handler=ok")
        case .failed(let error):
            parts.append("handler=error (\(error))")
        }
        if let gpuMilliseconds {
            parts.append("gpuMs=\(String(format: "%.0f", gpuMilliseconds))")
        }
        return parts.joined(separator: " ")
    }

    /// The one rule for a single command buffer: it did its work exactly
    /// when its status is `.completed` (Metal sets an error only on
    /// `.error`). `ChessNetwork.requireCompleted` is the throwing form.
    static func bufferCompleted(_ status: MTLCommandBufferStatus) -> Bool {
        status == .completed
    }

    static func name(of status: MTLCommandBufferStatus) -> String {
        switch status {
        case .notEnqueued: return "notEnqueued"
        case .enqueued: return "enqueued"
        case .committed: return "committed"
        case .scheduled: return "scheduled"
        case .completed: return "completed"
        case .error: return "error"
        @unknown default: return "status\(status.rawValue)"
        }
    }
}
