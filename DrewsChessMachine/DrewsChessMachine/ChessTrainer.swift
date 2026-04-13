import Darwin
import Foundation
import Metal
import MetalPerformanceShadersGraph

// MARK: - Errors

enum ChessTrainerError: LocalizedError {
    case lossOutputMissing
    case gradientMissing(String)

    var errorDescription: String? {
        switch self {
        case .lossOutputMissing:
            return "Training step ran but loss tensor was not in the result map"
        case .gradientMissing(let name):
            return "Gradient missing for variable: \(name)"
        }
    }
}

// MARK: - Training Step Timing

/// Per-step timing breakdown. All values in milliseconds.
struct TrainStepTiming: Sendable {
    /// CPU work to synthesize random inputs and pack them into MPSGraphTensorData.
    let dataPrepMs: Double
    /// GPU graph.run() — forward + backward + SGD weight updates.
    let gpuRunMs: Double
    /// CPU work to read the loss scalar back from the tensor result.
    let readbackMs: Double
    /// Total wall-clock time for the whole step.
    let totalMs: Double
    /// Loss value reported by the graph (lets us spot NaNs / explosions).
    let loss: Float
}

// MARK: - Sweep Result

/// Either a measured row or a row we refused to run because it would
/// have blown past the device's working-set or single-buffer caps.
/// Once a sweep skips one batch size, every larger size is also skipped
/// (memory only grows from there) so the table still has one entry per
/// requested batch size — the skipped ones just carry our estimates
/// instead of timings.
enum SweepRow: Sendable {
    case completed(SweepResult)
    case skipped(SkippedRow)

    var batchSize: Int {
        switch self {
        case .completed(let r): return r.batchSize
        case .skipped(let r): return r.batchSize
        }
    }
}

/// A batch size we declined to actually run because our footprint estimate
/// would exceed the device caps. Carries the estimate so the caller can
/// show *why* it was skipped.
struct SkippedRow: Sendable {
    let batchSize: Int
    /// Estimated total working-set bytes for one training step at this batch size.
    let estimatedBytes: UInt64
    /// Estimated size in bytes of the largest single MTLBuffer we'd allocate.
    let largestBufferBytes: UInt64
    /// Which cap we tripped (or both).
    let exceededWorkingSet: Bool
    let exceededBufferLength: Bool
}

/// Snapshot of the Metal device's memory caps. Captured once at sweep
/// start so the UI can show "here's the ceiling and how close we are".
struct DeviceMemoryCaps: Sendable {
    let recommendedMaxWorkingSet: UInt64
    let currentAllocated: UInt64
    let maxBufferLength: UInt64
}

/// One row of a batch-size sweep — what we measured at one fixed batch size.
struct SweepResult: Sendable {
    let batchSize: Int
    /// Wall-clock time of the very first step at this batch size. Includes
    /// MPSGraph kernel compilation; useful to see when the JIT recompiles.
    let warmupMs: Double
    /// Number of post-warmup steps timed.
    let steps: Int
    /// Wall-clock seconds for those `steps` steps.
    let elapsedSec: Double
    /// Mean per-step total wall time across the timed steps.
    let avgStepMs: Double
    /// Mean GPU run time (subset of avgStepMs) across the timed steps.
    let avgGpuMs: Double
    /// Effective per-second training throughput. The headline number — this
    /// is what the user actually wants to compare across batch sizes.
    let positionsPerSec: Double
    /// Last loss value at this batch size, for sanity checking.
    let lastLoss: Float
    /// Peak `phys_footprint` (process-wide resident memory, including
    /// everything Metal pulled into the unified-memory pool) sampled
    /// across this row's run. Sampled by the UI heartbeat ~10× per second
    /// while the row is in flight, plus once at row start and once at
    /// row end so even very fast rows get at least two readings. This is
    /// what we feed into the linear fit that predicts subsequent rows.
    let peakResidentBytes: UInt64
}

// MARK: - Continuous Training Stats

/// Aggregated stats over a continuous training run. Updated after every step.
struct TrainingRunStats: Sendable {
    var steps: Int = 0
    var totalGpuMs: Double = 0
    var totalStepMs: Double = 0
    var minStepMs: Double = .infinity
    var maxStepMs: Double = 0
    var lastTiming: TrainStepTiming?
    var startTime: CFAbsoluteTime = CFAbsoluteTimeGetCurrent()

    mutating func record(_ t: TrainStepTiming) {
        steps += 1
        totalGpuMs += t.gpuRunMs
        totalStepMs += t.totalMs
        if t.totalMs < minStepMs { minStepMs = t.totalMs }
        if t.totalMs > maxStepMs { maxStepMs = t.totalMs }
        lastTiming = t
    }

    var avgStepMs: Double { steps > 0 ? totalStepMs / Double(steps) : 0 }
    var avgGpuMs: Double { steps > 0 ? totalGpuMs / Double(steps) : 0 }
    var stepsPerSecond: Double {
        let elapsed = CFAbsoluteTimeGetCurrent() - startTime
        return elapsed > 0 ? Double(steps) / elapsed : 0
    }

    /// Projected wall time for one "epoch" of 250 batches, based on average step time.
    var projectedSecPer250Steps: Double { avgStepMs * 250 / 1000 }
}

// MARK: - Chess Trainer

/// Builds a separate training-mode copy of the chess network and runs
/// benchmark training steps against it. The trainer owns its own
/// ChessNetwork instance (with `bnMode = .training`), distinct from the
/// inference network used by Play Game / Forward Pass — that way the
/// inference network keeps its frozen-stats BN for fast play, while the
/// trainer measures realistic training-step costs through batch-stats BN
/// and the full backward graph.
///
/// Repeated trainStep() calls actually update the trainer's internal
/// weights via SGD (this is how we verified the training pipeline is
/// mechanically correct: random data, random labels, but loss still drops).
///
/// Marked @unchecked Sendable for the same reason as ChessNetwork — Metal
/// objects aren't Sendable but access is serialized externally (UI gates
/// training and inference to never overlap).
final class ChessTrainer: @unchecked Sendable {

    // MARK: Configuration

    let learningRate: Float

    // MARK: Graph Tensors

    private(set) var network: ChessNetwork
    private var movePlayedPlaceholder: MPSGraphTensor   // [batch] int32
    private var zPlaceholder: MPSGraphTensor            // [batch, 1] float
    private var totalLoss: MPSGraphTensor               // scalar
    private var assignOps: [MPSGraphOperation]

    // MARK: Init

    init(learningRate: Float = 1e-3) throws {
        self.learningRate = learningRate
        let net = try ChessNetwork(bnMode: .training)
        self.network = net
        let built = try Self.buildTrainingOps(network: net, learningRate: learningRate)
        self.movePlayedPlaceholder = built.movePlayed
        self.zPlaceholder = built.z
        self.totalLoss = built.totalLoss
        self.assignOps = built.assignOps
    }

    /// Tear down the current training-mode network and build a fresh one.
    /// Used at the start of a sweep so each run starts from random weights
    /// rather than whatever the previous run left behind. Throws if the
    /// underlying ChessNetwork init fails (Metal/device problems) or if
    /// gradient lookup fails for any trainable variable.
    func resetNetwork() throws {
        let net = try ChessNetwork(bnMode: .training)
        self.network = net
        let built = try Self.buildTrainingOps(network: net, learningRate: learningRate)
        self.movePlayedPlaceholder = built.movePlayed
        self.zPlaceholder = built.z
        self.totalLoss = built.totalLoss
        self.assignOps = built.assignOps
    }

    /// Build the training subgraph (loss + gradients + SGD assigns) on top
    /// of the given network's forward graph. Returns the placeholders, loss
    /// tensor, and assign ops the caller needs to run a training step.
    /// Throws `ChessTrainerError.gradientMissing` if any trainable variable
    /// fails gradient lookup — that would mean the autodiff couldn't reach
    /// it from the loss, which is a network-construction bug we want to
    /// surface immediately rather than silently train without it.
    private static func buildTrainingOps(
        network: ChessNetwork,
        learningRate: Float
    ) throws -> (
        movePlayed: MPSGraphTensor,
        z: MPSGraphTensor,
        totalLoss: MPSGraphTensor,
        assignOps: [MPSGraphOperation]
    ) {
        let graph = network.graph
        let dtype = ChessNetwork.dataType

        // --- Placeholders for training targets ---

        let movePlayed = graph.placeholder(
            shape: [-1],
            dataType: .int32,
            name: "move_played"
        )
        let z = graph.placeholder(
            shape: [-1, 1],
            dataType: dtype,
            name: "z_outcome"
        )

        // --- Policy loss: L = mean( z * -log_softmax(logits)[a*] ) ---
        //
        // Standard outcome-weighted cross entropy. logSoftMax gives us
        // numerically stable log probabilities. We one-hot the played move
        // and sum across the move axis to pull out p(a*). Multiplying by z
        // applies the outcome weighting from chess-engine-design.md:
        //   z=+1 → push p(a*) up, z=-1 → push it down, z=0 → no contribution.

        // Compute log_softmax manually: softMax then log. (MPSGraph has
        // softMax + logarithm but no fused logSoftMax wrapper.)
        let softmax = graph.softMax(
            with: network.policyOutput,
            axis: 1,
            name: "policy_softmax"
        )
        let logSoftmax = graph.logarithm(with: softmax, name: "policy_log_softmax")

        let oneHot = graph.oneHot(
            withIndicesTensor: movePlayed,
            depth: 4096,
            axis: 1,
            dataType: dtype,
            onValue: 1.0,
            offValue: 0.0,
            name: "move_onehot"
        )

        let ceProduct = graph.multiplication(oneHot, logSoftmax, name: "ce_product")
        let logProbAtMove = graph.reductionSum(
            with: ceProduct,
            axis: 1,
            name: "log_prob_at_move"
        )
        // logProbAtMove shape: [batch, 1] (axis-1 reduction keeps the dim)
        let negLogProb = graph.negative(with: logProbAtMove, name: "neg_log_prob")
        let weightedCE = graph.multiplication(z, negLogProb, name: "z_weighted_ce")
        let policyLoss = graph.mean(of: weightedCE, axes: [0, 1], name: "policy_loss")

        // --- Value loss: L = mean( (z - v)^2 ) ---

        let diff = graph.subtraction(z, network.valueOutput, name: "value_diff")
        let sq = graph.square(with: diff, name: "value_sq")
        let valueLoss = graph.mean(of: sq, axes: [0, 1], name: "value_loss")

        // --- Total loss ---

        let totalLossTensor = graph.addition(valueLoss, policyLoss, name: "total_loss")

        // --- Gradients w.r.t. trainable variables ---

        let grads = graph.gradients(
            of: totalLossTensor,
            with: network.trainableVariables,
            name: "gradients"
        )

        // --- SGD updates: v_new = v - lr * grad(v); assign back to v ---

        let lrTensor = graph.constant(Double(learningRate), dataType: dtype)
        var ops: [MPSGraphOperation] = []
        ops.reserveCapacity(network.trainableVariables.count)
        for (i, variable) in network.trainableVariables.enumerated() {
            guard let grad = grads[variable] else {
                // If autodiff didn't produce a gradient for a trainable
                // variable, the network is mis-wired (the loss can't reach
                // this variable through the graph). Surface it instead of
                // silently training without it — a "shouldn't happen" that
                // happens silently is the worst kind of bug.
                throw ChessTrainerError.gradientMissing(
                    variable.operation.name.isEmpty ? "trainable[\(i)]" : variable.operation.name
                )
            }
            let scaled = graph.multiplication(lrTensor, grad, name: nil)
            let updated = graph.subtraction(variable, scaled, name: nil)
            let assignOp = graph.assign(variable, tensor: updated, name: nil)
            ops.append(assignOp)
        }

        return (movePlayed, z, totalLossTensor, ops)
    }

    // MARK: - Training Step

    /// Run a single training step on a batch of randomly synthesized data.
    /// Returns timing breakdown and the loss scalar. Repeated calls update
    /// this trainer's internal network weights via SGD — that's how we
    /// verified the training pipeline is mechanically correct (random data
    /// + random labels + monotonically decreasing loss). The trainer's
    /// internal network is **not** the inference network, so these updates
    /// don't affect Play Game or Forward Pass.
    func trainStep(batchSize: Int) throws -> TrainStepTiming {
        let totalStart = CFAbsoluteTimeGetCurrent()

        // --- Data prep: synthesize random boards, moves, outcomes ---

        let prepStart = CFAbsoluteTimeGetCurrent()
        let floatsPerBoard = ChessNetwork.inputPlanes * ChessNetwork.boardSize * ChessNetwork.boardSize
        let totalBoardFloats = batchSize * floatsPerBoard

        var boardFloats = [Float](repeating: 0, count: totalBoardFloats)
        Self.fillRandomFloats(&boardFloats)

        var moveIndices = [Int32](repeating: 0, count: batchSize)
        // Random move indices in [0, 4096). One per batch row.
        for i in 0..<batchSize {
            moveIndices[i] = Int32.random(in: 0..<Int32(ChessNetwork.policySize))
        }

        var zValues = [Float](repeating: 0, count: batchSize)
        // Random outcomes from {-1, 0, +1} so the loss includes all three
        // signed regimes (push up, push down, no contribution).
        for i in 0..<batchSize {
            zValues[i] = Float(Int.random(in: 0..<3) - 1)
        }

        let boardData = MPSGraphTensorData(
            device: network.graphDevice,
            data: ChessNetwork.makeWeightData(boardFloats),
            shape: [
                NSNumber(value: batchSize),
                NSNumber(value: ChessNetwork.inputPlanes),
                NSNumber(value: ChessNetwork.boardSize),
                NSNumber(value: ChessNetwork.boardSize)
            ],
            dataType: ChessNetwork.dataType
        )

        let moveData = moveIndices.withUnsafeBufferPointer { buf -> MPSGraphTensorData in
            let bytes = Data(buffer: buf)
            return MPSGraphTensorData(
                device: network.graphDevice,
                data: bytes,
                shape: [NSNumber(value: batchSize)],
                dataType: .int32
            )
        }

        let zData = MPSGraphTensorData(
            device: network.graphDevice,
            data: ChessNetwork.makeWeightData(zValues),
            shape: [NSNumber(value: batchSize), 1],
            dataType: ChessNetwork.dataType
        )
        let prepMs = (CFAbsoluteTimeGetCurrent() - prepStart) * 1000

        // --- GPU run: forward + backward + SGD update in one graph execution ---

        let gpuStart = CFAbsoluteTimeGetCurrent()
        let results = network.graph.run(
            with: network.commandQueue,
            feeds: [
                network.inputPlaceholder: boardData,
                movePlayedPlaceholder: moveData,
                zPlaceholder: zData
            ],
            targetTensors: [totalLoss],
            targetOperations: assignOps
        )
        let gpuMs = (CFAbsoluteTimeGetCurrent() - gpuStart) * 1000

        // --- Read loss scalar back ---

        let readbackStart = CFAbsoluteTimeGetCurrent()
        guard let lossData = results[totalLoss] else {
            throw ChessTrainerError.lossOutputMissing
        }
        let lossBuf = ChessNetwork.readFloats(from: lossData, count: 1)
        let readbackMs = (CFAbsoluteTimeGetCurrent() - readbackStart) * 1000

        let totalMs = (CFAbsoluteTimeGetCurrent() - totalStart) * 1000

        return TrainStepTiming(
            dataPrepMs: prepMs,
            gpuRunMs: gpuMs,
            readbackMs: readbackMs,
            totalMs: totalMs,
            loss: lossBuf[0]
        )
    }

    // MARK: - Batch Size Sweep

    /// Run a batch-size sweep. For each size in `sizes`:
    ///   1. Run one warmup step (which pays MPSGraph kernel-compile cost
    ///      the first time a new batch shape is seen — measured separately
    ///      so it doesn't pollute the throughput number).
    ///   2. Loop trainStep until `targetSecondsPerSize` elapsed (or a step
    ///      cap is hit, whichever comes first).
    ///   3. Compute average per-step time and positions/sec from the timed
    ///      window only.
    ///
    /// `progress` is called from the worker thread before each step so the
    /// UI can show "currently sweeping batch=X, step Y, elapsed Z". Pass
    /// `cancelled` from the UI to stop a sweep early — checked between steps.
    ///
    /// The trainer's network is **not** reset by this method. Callers that
    /// want fresh weights should call `resetNetwork()` first. Loss across a
    /// long sweep will drift downward as SGD overfits the random inputs;
    /// that's harmless for timing purposes.
    func runSweep(
        sizes: [Int],
        targetSecondsPerSize: Double,
        maxStepsPerSize: Int = 10_000,
        cancelled: @Sendable () -> Bool = { false },
        progress: @Sendable (Int, Int, Double) -> Void = { _, _, _ in },
        recordPeakSampleNow: @Sendable () -> Void = {},
        consumeRowPeak: @Sendable () -> UInt64 = { 0 },
        onRowCompleted: @Sendable (SweepRow) -> Void = { _ in }
    ) throws -> [SweepRow] {
        var results: [SweepRow] = []
        results.reserveCapacity(sizes.count)

        // Read device caps once. They're fixed for the lifetime of the
        // process so it's safe to cache for the whole sweep.
        let device = network.metalDevice
        let workingSetCap = device.recommendedMaxWorkingSetSize
        let bufferCap = UInt64(device.maxBufferLength)
        // Skip threshold: 75% of the smaller of the two caps. The "lesser"
        // bit is deliberately conservative — on this hardware
        // maxBufferLength is well under recommendedMaxWorkingSetSize, so
        // capping the *total* estimate against the smaller of the two
        // gives a safety margin even though the comparison mixes
        // different things (total vs. single-buffer). Better to skip a
        // borderline batch than to take down the machine.
        let safetyFraction = 0.75
        let estimateThreshold = UInt64(Double(min(workingSetCap, bufferCap)) * safetyFraction)
        // Once we cross either threshold, every larger batch size will too —
        // latch this so we stop trying instead of crashing the machine.
        var skipFromHere = false
        // Empirically observed (batch, currentAllocatedSize) pairs from rows
        // we've already run. We fit a line through these to predict the
        // next batch's working-set footprint instead of guessing from the
        // network architecture — the architectural estimate was wildly
        // pessimistic compared to what MPSGraph actually allocates.
        var allocSamples: [(batch: Int, bytes: UInt64)] = []

        for batchSize in sizes {
            if cancelled() { break }

            // Largest single MTLBuffer we'll ask Metal for. Exact, not
            // estimated: the trainer literally uploads a [batch, 128, 8, 8]
            // float32 activation tensor and that's the biggest buffer in
            // the graph (beats the [batch, 4096] policy tensors and the
            // [batch, 18, 8, 8] input).
            let largestBufferBytes = Self.largestBufferBytes(forBatchSize: batchSize)
            // Working-set prediction comes from a least-squares fit over
            // the rows we've already run. Returns nil before we have any
            // data to fit, in which case we don't skip on this criterion.
            let predictedBytes = Self.predictAllocatedBytes(
                forBatchSize: batchSize,
                from: allocSamples
            )

            let exceedsBuffer = largestBufferBytes > bufferCap
            let exceedsWorkingSet: Bool
            if let predictedBytes {
                exceedsWorkingSet = predictedBytes > estimateThreshold
            } else {
                exceedsWorkingSet = false
            }
            if exceedsWorkingSet || exceedsBuffer {
                skipFromHere = true
            }

            if skipFromHere {
                let skipped = SkippedRow(
                    batchSize: batchSize,
                    estimatedBytes: predictedBytes ?? 0,
                    largestBufferBytes: largestBufferBytes,
                    exceededWorkingSet: exceedsWorkingSet,
                    exceededBufferLength: exceedsBuffer
                )
                let row = SweepRow.skipped(skipped)
                results.append(row)
                onRowCompleted(row)
                continue
            }

            // Drop a peak sample right before warmup so even rows that
            // finish between heartbeats get a baseline reading.
            recordPeakSampleNow()

            // Warmup: first call at this batch size pays whatever per-shape
            // compile cost MPSGraph charges. Time it but don't count it
            // toward the throughput number.
            let warmup = try trainStep(batchSize: batchSize)
            if cancelled() { break }
            recordPeakSampleNow()

            var timedSteps = 0
            var totalStepMs: Double = 0
            var totalGpuMs: Double = 0
            var lastLoss: Float = warmup.loss
            let runStart = CFAbsoluteTimeGetCurrent()

            while !cancelled() && timedSteps < maxStepsPerSize {
                let elapsed = CFAbsoluteTimeGetCurrent() - runStart
                if elapsed >= targetSecondsPerSize { break }
                progress(batchSize, timedSteps, elapsed)

                let timing = try trainStep(batchSize: batchSize)
                timedSteps += 1
                totalStepMs += timing.totalMs
                totalGpuMs += timing.gpuRunMs
                lastLoss = timing.loss
            }

            let elapsed = CFAbsoluteTimeGetCurrent() - runStart
            let avgStepMs = timedSteps > 0 ? totalStepMs / Double(timedSteps) : 0
            let avgGpuMs = timedSteps > 0 ? totalGpuMs / Double(timedSteps) : 0
            let positions = timedSteps * batchSize
            let positionsPerSec = elapsed > 0 ? Double(positions) / elapsed : 0

            // Final sample before we read the peak — captures whatever
            // resident memory the just-finished steps left behind.
            recordPeakSampleNow()
            let peakResident = consumeRowPeak()
            // Feed the empirical linear fit that predicts the next row's
            // footprint. The fit only sees what we actually observed, no
            // architectural fudge factors.
            allocSamples.append((batch: batchSize, bytes: peakResident))

            let row = SweepRow.completed(
                SweepResult(
                    batchSize: batchSize,
                    warmupMs: warmup.totalMs,
                    steps: timedSteps,
                    elapsedSec: elapsed,
                    avgStepMs: avgStepMs,
                    avgGpuMs: avgGpuMs,
                    positionsPerSec: positionsPerSec,
                    lastLoss: lastLoss,
                    peakResidentBytes: peakResident
                )
            )
            results.append(row)
            // Fire after the row is complete so the UI can show partial
            // results as the sweep advances rather than waiting for the
            // whole sweep to finish.
            onRowCompleted(row)
        }

        return results
    }

    // MARK: - Footprint Helpers

    /// Exact size of the largest single MTLBuffer the trainer requests at
    /// this batch size — one [batch, 128, 8, 8] float32 activation tensor.
    /// That's larger than the [batch, 4096] policy tensors and the
    /// [batch, 18, 8, 8] input, so it's the buffer that would first hit
    /// `maxBufferLength`. This is an architectural fact, not a guess.
    static func largestBufferBytes(forBatchSize batchSize: Int) -> UInt64 {
        let floatBytes = MemoryLayout<Float>.size
        let spatial = ChessNetwork.boardSize * ChessNetwork.boardSize
        let channels = ChessNetwork.channels
        return UInt64(channels * spatial * floatBytes) * UInt64(batchSize)
    }

    /// Predict `currentAllocatedSize` for `batchSize` from the
    /// (batch, allocated) pairs already observed during this sweep.
    /// Returns nil before we have any samples to fit.
    ///
    /// With a single sample we draw a line from the origin through it
    /// (slope-only). With two or more samples we use ordinary least
    /// squares on (batch, bytes), which automatically captures both the
    /// per-sample slope and any fixed overhead. No fudge factors — what
    /// MPSGraph actually allocated is what we extrapolate from.
    static func predictAllocatedBytes(
        forBatchSize batchSize: Int,
        from samples: [(batch: Int, bytes: UInt64)]
    ) -> UInt64? {
        if samples.isEmpty { return nil }

        let target = Double(batchSize)
        if samples.count == 1 {
            let only = samples[0]
            let perSample = Double(only.bytes) / Double(only.batch)
            return UInt64(max(0, perSample * target))
        }

        let n = Double(samples.count)
        var sumX = 0.0
        var sumY = 0.0
        var sumXY = 0.0
        var sumXX = 0.0
        for s in samples {
            let x = Double(s.batch)
            let y = Double(s.bytes)
            sumX += x
            sumY += y
            sumXY += x * y
            sumXX += x * x
        }
        let denom = n * sumXX - sumX * sumX
        // denom is zero only if all sample batch sizes are identical —
        // which can't happen here since the sweep monotonically increases
        // batch size — but fall back to the slope-from-origin rule rather
        // than dividing by zero.
        guard denom != 0 else {
            let perSample = sumY / sumX
            return UInt64(max(0, perSample * target))
        }
        let slope = (n * sumXY - sumX * sumY) / denom
        let intercept = (sumY - slope * sumX) / n
        let predicted = slope * target + intercept
        return UInt64(max(0, predicted))
    }

    /// Read the process-wide `phys_footprint` from `task_info`. On Apple
    /// Silicon's unified memory architecture this captures everything the
    /// process is holding onto — CPU buffers and Metal-managed GPU memory
    /// alike — so it's a strictly better high-water-mark proxy than
    /// `MTLDevice.currentAllocatedSize`, which only sees memory that's
    /// still live at the moment you query it. Returns 0 on failure rather
    /// than throwing — the caller is sampling on a hot path and a missed
    /// reading is recoverable, while throwing would force exception
    /// handling around every UI tick.
    static func currentPhysFootprintBytes() -> UInt64 {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size
        )
        let kr = withUnsafeMutablePointer(to: &info) { infoPtr in
            infoPtr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { intPtr in
                task_info(
                    mach_task_self_,
                    task_flavor_t(TASK_VM_INFO),
                    intPtr,
                    &count
                )
            }
        }
        guard kr == KERN_SUCCESS else { return 0 }
        return UInt64(info.phys_footprint)
    }

    /// Snapshot the device's memory caps right now. Read once at the start
    /// of a sweep so the UI header has a stable reference point.
    func deviceMemoryCaps() -> DeviceMemoryCaps {
        let device = network.metalDevice
        return DeviceMemoryCaps(
            recommendedMaxWorkingSet: device.recommendedMaxWorkingSetSize,
            currentAllocated: UInt64(device.currentAllocatedSize),
            maxBufferLength: UInt64(device.maxBufferLength)
        )
    }

    // MARK: - Random Fill

    /// Fill a float buffer with pseudo-random values in [0, 1) using a fast
    /// inline LCG. Avoids the cost of arc4random_buf + conversion for the
    /// ~1.15M floats per batch we need. Quality doesn't matter — we only need
    /// non-zero, non-uniform values to exercise the same compute paths real
    /// data would.
    private static func fillRandomFloats(_ buffer: inout [Float]) {
        var rng: UInt64 = UInt64.random(in: 0...UInt64.max) | 1
        let scale: Float = Float(1.0 / 4294967296.0)
        buffer.withUnsafeMutableBufferPointer { buf in
            guard let base = buf.baseAddress else { return }
            for i in 0..<buf.count {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                let high = UInt32(truncatingIfNeeded: rng >> 32)
                base[i] = Float(high) * scale
            }
        }
    }
}
