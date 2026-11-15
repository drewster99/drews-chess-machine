//
//  GraphAudit.swift
//  DrewsChessMachine
//
//  The fp64 graph audit (`documentation/plans-active/FP64_GRAPH_AUDIT_PLAN.md`):
//  every forward operation of the app's own network graph — inference or
//  training mode, at a real batch size — recomputed in float64 on the CPU from
//  the operand values MPSGraph held, and, in training graphs, every tensor's
//  gradient recomputed from its consumers' output gradients.
//
//  How a case runs:
//  1. Build `ChessNetwork` for the case's architecture with an audit recording
//     (`InferenceGraphRecorder(purpose: .audit)`), which keeps every op's kind,
//     operand tensors and output tensor. Load the model file's weights (or
//     draw fresh ones for a preset).
//  2. Training mode: add a loss L = Σ R_p·policy + Σ R_l·valueLogits +
//     Σ R_v·value with fixed random R (the network's three outputs, every path
//     to the heads), and ask MPSGraph for ∂L/∂t for every recorded tensor t.
//  3. Run the graph in chunks of consecutive ops (GPU memory), each run
//     returning the chunk's operands, outputs and their gradients.
//  4. Per op: the forward result from its operands in float64 vs MPSGraph's.
//     Per tensor whose gradient-carrying consumers are all in the chunk: the
//     sum of the consumers' float64 vector-Jacobian products (each from
//     MPSGraph's own gradient of that consumer's output) vs MPSGraph's
//     gradient of the tensor.
//
//  What it cannot see: tensors that also feed an op built on the raw graph
//  (dropout's mask multiply, the loss itself) get gradient the recording
//  doesn't account for; those are reported as such, not compared. Extra
//  target tensors add consumers to the graph, and MPSGraph's kernel choice
//  can depend on consumers (the fp32 conv defect did), so the audit sees the
//  graph with its readbacks — a finding is confirmed with a standalone repro.
//

import Foundation
import Metal
import MetalPerformanceShadersGraph

enum GraphAudit {

    /// One graph to audit.
    struct Case: Codable, Equatable {
        enum Mode: String, Codable { case training, inference }

        let label: String
        /// A model file whose architecture and weights to use…
        let model: String?
        /// …or a preset name / architecture file, with freshly drawn weights.
        let architecture: String?
        /// Overrides the architecture's compute data type (with `policyTail`).
        let computeDataType: ComputeDataType?
        let policyTail: PolicyTailPrecisionSetting?
        let mode: Mode
        let batch: Int
        /// Weight-init seed for an `architecture` case, and the loss weights' seed.
        let seed: UInt64?

        enum CodingKeys: String, CodingKey {
            case label, model, architecture, mode, batch, seed
            case computeDataType = "compute_data_type"
            case policyTail = "policy_tail_precision"
        }
    }

    struct Config: Codable {
        let cases: [Case]
        /// Byte budget for one chunk's CPU-side float64 tensors (default 1 GiB).
        let chunkBytes: Int?
        /// Process footprint at which a case stops (default 12 GiB).
        let memoryLimitGiB: Double?

        enum CodingKeys: String, CodingKey {
            case cases
            case chunkBytes = "chunk_bytes"
            case memoryLimitGiB = "memory_limit_gib"
        }
    }

    enum AuditError: Error, LocalizedError {
        case caseNeedsOneSource(String)
        case recordingFailed(String)
        case tooFewWeights(have: Int, need: Int)
        case missingResult(String)
        case unsupportedDataType(String)
        case footprintLimit(stage: String, bytes: UInt64, limit: UInt64)
        /// A per-chunk rebuild recorded different ops than the plan's build.
        case rebuildMismatch(chunk: Int)
        case footprintUnreadable(Int32)

        var errorDescription: String? {
            switch self {
            case let .footprintLimit(stage, bytes, limit):
                return "process footprint \(bytes >> 20) MB at \(stage) is above the limit \(limit >> 20) MB; "
                    + "case stopped (lower the batch or chunk_bytes)"
            case .rebuildMismatch(let chunk):
                return "the graph rebuilt for chunk \(chunk) recorded different operations than the planning build"
            case .footprintUnreadable(let kr):
                return "could not read the process footprint (kern_return \(kr))"
            case .caseNeedsOneSource(let label):
                return "case '\(label)' needs exactly one of model / architecture"
            case .recordingFailed(let detail):
                return "audit recording failed: \(detail)"
            case .tooFewWeights(let have, let need):
                return "model file holds \(have) tensors; the network needs \(need)"
            case .missingResult(let id):
                return "MPSGraph returned no value for \(id)"
            case .unsupportedDataType(let detail):
                return "unsupported tensor data type \(detail)"
            }
        }
    }

    /// One line of output (JSON object).
    typealias Record = [String: Any]

    // MARK: - Case setup

    static func architectureAndWeights(for auditCase: Case) throws -> (NetworkArchitecture, [[Float]]?) {
        var arch: NetworkArchitecture
        var weights: [[Float]]?
        switch (auditCase.model, auditCase.architecture) {
        case (let model?, nil):
            let file = try CheckpointManager.loadModelFile(at: URL(fileURLWithPath: (model as NSString).expandingTildeInPath))
            arch = file.architecture
            weights = file.weights
        case (nil, let name?):
            arch = try ArchitecturePresetStore.resolve(nameOrPath: name).named.architecture
            weights = nil
        default:
            throw AuditError.caseNeedsOneSource(auditCase.label)
        }
        if let dtype = auditCase.computeDataType {
            if dtype != arch.computeDataType || auditCase.policyTail != nil {
                arch = try arch.withComputeDataType(dtype, tail: auditCase.policyTail)
            }
        }
        try arch.validate()
        return (arch, weights)
    }

    // MARK: - Run one case

    /// Audits one case, handing each output record to `emit` as it is made.
    /// Memory bounds for a run (the first full-size audit run exhausted the
    /// machine's memory and hung it).
    struct Limits {
        /// Budget for one chunk's CPU-side float64 tensors.
        let chunkBytes: Int
        /// The process footprint at which a case stops.
        let footprintBytes: UInt64
    }

    static func run(_ auditCase: Case, limits: Limits, emit: (Record) throws -> Void,
                    progress: (String) -> Void) throws {
        func requireFootprintBelowLimit(_ stage: String) throws {
            let now = try footprintBytes()
            if now > limits.footprintBytes {
                throw AuditError.footprintLimit(stage: stage, bytes: now, limit: limits.footprintBytes)
            }
        }
        let (arch, fileWeights) = try architectureAndWeights(for: auditCase)
        let seed = auditCase.seed ?? 1
        let n = auditCase.batch

        // Everything one GPU-side pass needs. Built afresh for every chunk:
        // MPSGraph caches a compiled executable, with its buffers, for each
        // distinct set of targets, and the chunks all ask for different ones —
        // one graph across chunks grew the process by ~10 GB per chunk at
        // batch 5376. Dropping the graph after its chunk frees its cache.
        struct Built {
            let network: ChessNetwork
            let recorder: InferenceGraphRecorder
            let operations: [GraphOperation]
            let feeds: [MPSGraphTensor: MPSGraphTensorData]
            /// ∂L/∂t by recorded id (training graphs).
            let gradientOf: [String: MPSGraphTensor]
            /// Recorded ids the audit loss reads directly.
            let lossConsumed: Set<String>
        }
        func build() throws -> Built {
            let recorder = InferenceGraphRecorder(purpose: .audit)
            let bnMode: BNMode = auditCase.mode == .training ? .training : .inference
            let network = try ChessNetwork(
                arch: arch, bnMode: bnMode,
                initialization: fileWeights == nil ? .seeded(initSeed: seed) : .overwrittenByLoad,
                graphRecorder: recorder)
            if let failure = recorder.recordingFailure {
                throw AuditError.recordingFailed(failure.localizedDescription)
            }
            if let fileWeights {
                let need = network.trainableVariables.count + network.bnRunningStatsVariables.count
                guard fileWeights.count >= need else {
                    throw AuditError.tooFewWeights(have: fileWeights.count, need: need)
                }
                let prefix = Array(fileWeights.prefix(need))
                try runBlocking { try await network.loadWeights(prefix) }
            }
            network.commandQueue.label = "GraphAudit"
            let graph = network.graph
            var feeds: [MPSGraphTensor: MPSGraphTensorData] = [:]
            feeds[network.inputPlaceholder] = try boardFeed(network: network, batch: n)
            if let rate = network.dropoutRateFeedPlaceholder, let zero = network.dropoutRateZeroTensorData {
                feeds[rate] = zero
            }
            let operations = recorder.recordedOperations
            var producerKind: [String: GraphOperationKind] = [:]
            for op in operations { producerKind[op.id] = op.kind }
            func isConstant(_ id: String) -> Bool {
                if case .constant = producerKind[id] { return true }
                return false
            }
            var gradientOf: [String: MPSGraphTensor] = [:]
            var lossConsumed: Set<String> = []
            if auditCase.mode == .training {
                let outputs: [(MPSGraphTensor, Int)] = [(network.policyOutput, 0), (network.valueLogits, 1),
                                                         (network.valueOutput, 2)]
                var loss: MPSGraphTensor?
                for (output, index) in outputs {
                    let width = output.shape?.last?.intValue ?? 1
                    let weightsPlaceholder = graph.placeholder(shape: [-1, NSNumber(value: width)], dataType: .float32,
                                                               name: "audit_loss_weights_\(index)")
                    feeds[weightsPlaceholder] = try lossWeightsFeed(network: network, batch: n, width: width,
                                                                    seed: seed &+ UInt64(index) &+ 1)
                    let wide = output.dataType == .float32 ? output : graph.cast(output, to: .float32, name: nil)
                    let term = graph.reductionSum(with: graph.multiplication(wide, weightsPlaceholder, name: nil),
                                                  axes: [0, 1], name: nil)
                    loss = loss.map { graph.addition($0, term, name: nil) } ?? term
                    for id in recordedIDs(of: output, recorder: recorder, operations: operations) {
                        lossConsumed.insert(id)
                    }
                }
                guard let loss else { preconditionFailure("three loss terms were built") }
                // MPSGraph refuses (asserts) a gradient for a tensor the loss does
                // not depend on — the BN running-stat EMA chain, constants — so ask
                // only for the loss's ancestors, found through every op's inputs
                // (raw training-only ops included).
                let ancestors = ancestorTensors(of: [loss])
                var wanted: [String] = []
                var wantedTensors: [MPSGraphTensor] = []
                var seen = Set<String>()
                for op in operations {
                    if case .constant = op.kind { continue }
                    for id in [op.id] + op.kind.operands where !isConstant(id) && seen.insert(id).inserted {
                        guard let t = recorder.tensor(forID: id), ancestors.contains(ObjectIdentifier(t)),
                              t.dataType == .float32 || t.dataType == .float16 || t.dataType == .bFloat16 else { continue }
                        wanted.append(id)
                        wantedTensors.append(t)
                    }
                }
                let gradients = graph.gradients(of: loss, with: wantedTensors, name: "audit_gradients")
                for (id, t) in zip(wanted, wantedTensors) {
                    if let g = gradients[t] { gradientOf[id] = g }
                }
            }
            return Built(network: network, recorder: recorder, operations: operations, feeds: feeds,
                         gradientOf: gradientOf, lossConsumed: lossConsumed)
        }

        // Plan from a first build, then release it.
        let operations: [GraphOperation]
        let planGradientIDs: Set<String>
        let lossConsumed: Set<String>
        var elementCounts: [String: Int] = [:]
        var tensorNames: [String: String] = [:]
        do {
            let plan = try autoreleasepool { try build() }
            operations = plan.operations
            planGradientIDs = Set(plan.gradientOf.keys)
            lossConsumed = plan.lossConsumed
            for op in operations {
                for id in [op.id] + op.kind.operands where elementCounts[id] == nil {
                    let t = plan.recorder.tensor(forID: id)
                    elementCounts[id] = t?.shape.map { $0.reduce(1) { $0 * ($1.intValue < 0 ? n : $1.intValue) } }
                        ?? n * 128 * 64
                    tensorNames[id] = t?.operation.name ?? ""
                }
            }
        }
        var producer: [String: Int] = [:]
        var consumers: [String: [Int]] = [:]
        for (index, op) in operations.enumerated() {
            producer[op.id] = index
            for operand in op.kind.operands { consumers[operand, default: []].append(index) }
        }
        func isConstant(_ id: String) -> Bool {
            guard let p = producer[id] else { return false }
            if case .constant = operations[p].kind { return true }
            return false
        }
        // A tensor's gradient-carrying consumers: recorded ops whose output has a gradient.
        func gradientConsumers(_ id: String) -> [Int] {
            (consumers[id] ?? []).filter { planGradientIDs.contains(operations[$0].id) }
        }

        progress("\(auditCase.label): \(operations.count) ops, batch \(n), \(auditCase.mode.rawValue), "
                 + "\(planGradientIDs.count) gradient tensors")

        // Chunks of consecutive ops. CPU-side cost of one tensor: its float64
        // values, and for a tensor with a gradient also MPSGraph's gradient,
        // the reference sum and the magnitude sum (8 bytes each per element).
        let chunks = makeChunks(operations: operations, budget: limits.chunkBytes, cost: { id in
            (elementCounts[id] ?? n * 128 * 64) * MemoryLayout<Double>.size * (planGradientIDs.contains(id) ? 4 : 1)
        }, gradientConsumers: gradientConsumers)
        var opRecords = 0
        var gradientRecords = 0
        for (chunkIndex, range) in chunks.enumerated() {
            // Each chunk in its own autorelease pool: MPSGraph hands its results
            // back autoreleased, and this runs on a thread whose pool never
            // drains (a CLI pre-flight), so without it every chunk's GPU buffers
            // stay alive until exit (measured: +10 GB per chunk at batch 5376).
            try autoreleasepool {
                let started = DispatchTime.now()
                let built = try build()
                guard built.operations == operations, Set(built.gradientOf.keys) == planGradientIDs else {
                    throw AuditError.rebuildMismatch(chunk: chunkIndex + 1)
                }
                let graph = built.network.graph
                let network = built.network
                let feeds = built.feeds
                let gradientOf = built.gradientOf
                func tensor(_ id: String) throws -> MPSGraphTensor {
                    guard let t = built.recorder.tensor(forID: id) else { throw AuditError.missingResult(id) }
                    return t
                }
                var valueIDs: [String] = []
                var seen = Set<String>()
                for index in range {
                    let op = operations[index]
                    if case .constant = op.kind { continue }
                    for id in [op.id] + op.kind.operands where seen.insert(id).inserted { valueIDs.append(id) }
                }
                var gradientIDs: [String] = []
                for id in valueIDs where gradientOf[id] != nil { gradientIDs.append(id) }

                var targets: [MPSGraphTensor] = []
                var targetIndex: [ObjectIdentifier: Int] = [:]
                func addTarget(_ t: MPSGraphTensor) {
                    if targetIndex[ObjectIdentifier(t)] == nil {
                        targetIndex[ObjectIdentifier(t)] = targets.count
                        targets.append(t)
                    }
                }
                for id in valueIDs { addTarget(try tensor(id)) }
                for id in gradientIDs {
                    guard let g = gradientOf[id] else { continue }
                    addTarget(g)
                }
                // The audit is the only user of this network and runs outside Swift
                // concurrency (a CLI pre-flight), so it submits and waits here.
                var results = try GPUSubmission.runGraph(graph, on: network.commandQueue, feeds: feeds,
                                                         targetTensors: targets, targetOperations: nil,
                                                         stage: .graphAudit, work: .positions(n))
                try requireFootprintBelowLimit("chunk \(chunkIndex + 1) GPU results")
                var values: [String: Tensor64] = [:]
                // Float operands and outputs; a boolean or integer operand (the
                // dropout keep-mask) is read exactly and has no float type.
                var valueTypes: [String: GraphDataType] = [:]
                for id in valueIDs {
                    let t = try tensor(id)
                    guard let data = results[t] else { throw AuditError.missingResult(id) }
                    values[id] = try readTensor(data)
                    valueTypes[id] = GraphDataType(data.dataType)
                }
                for id in valueIDs { results.removeValue(forKey: try tensor(id)) }
                var mpsGradients: [String: Tensor64] = [:]
                var gradientTypes: [String: GraphDataType] = [:]
                for id in gradientIDs {
                    guard let g = gradientOf[id], let data = results[g] else { throw AuditError.missingResult("∂L/∂\(id)") }
                    mpsGradients[id] = try readTensor(data)
                    gradientTypes[id] = try graphDataType(data.dataType, "∂L/∂\(id)")
                    results.removeValue(forKey: g)
                }
                results.removeAll()
                try requireFootprintBelowLimit("chunk \(chunkIndex + 1) readback")

                // Forward checks, and each op's VJP contributions.
                var contributions: [String: Tensor64] = [:]
                // Σ over consumers of the magnitude of the terms each contribution
                // sums: the scale a cancelling gradient is judged on.
                var magnitudes: [String: Tensor64] = [:]
                var vjpFailed: Set<String> = []
                for index in range {
                    let op = operations[index]
                    if case .constant = op.kind { continue }
                    let operands = try op.kind.operands.map { id -> Tensor64 in
                        guard let v = values[id] else { throw AuditError.missingResult(id) }
                        return v
                    }
                    guard let actual = values[op.id], let outType = valueTypes[op.id] else {
                        throw AuditError.missingResult(op.id)
                    }
                    var record: Record = baseRecord(auditCase, arch: arch)
                    record["event"] = "op"
                    record["op_index"] = index
                    record["op_id"] = op.id
                    record["label"] = op.label ?? ""
                    record["kind"] = kindName(op.kind)
                    record["dtype"] = outType.rawValue
                    record["shape"] = actual.shape
                    record["operand_dtypes"] = op.kind.operands.map { valueTypes[$0]?.rawValue ?? "?" }
                    do {
                        let reference = try forward(op.kind, operands: operands, outputShape: actual.shape,
                                                    outputType: outType)
                        var forwardRecord = comparisonRecord(GraphAuditReference.compare(
                            actual: actual.values, reference: reference.values, dataType: outType))
                        let magnitude = try forwardMagnitude(op.kind, operands: operands, reference: reference,
                                                             outputShape: actual.shape)
                        forwardRecord["scale_rel_err"] = scaleRelativeError(actual: actual.values,
                                                                            reference: reference.values,
                                                                            magnitude: magnitude.values)
                        record["forward"] = forwardRecord
                    } catch {
                        record["forward_error"] = error.localizedDescription
                    }
                    if let dY = mpsGradients[op.id] {
                        do {
                            let grads = try vjp(op.kind, operands: operands, dY: dY)
                            let mags = try vjpMagnitude(op.kind, operands: operands, dY: dY)
                            for (j, operandID) in op.kind.operands.enumerated() {
                                guard j < grads.count, let grad = grads[j], !isConstant(operandID) else { continue }
                                guard j < mags.count, let mag = mags[j] else {
                                    throw GraphAuditReferenceError.unsupported("no magnitude for \(operandID)")
                                }
                                try accumulate(grad, into: &contributions, id: operandID)
                                try accumulate(mag, into: &magnitudes, id: operandID)
                            }
                        } catch {
                            record["backward_error"] = error.localizedDescription
                            for operandID in op.kind.operands { vjpFailed.insert(operandID) }
                        }
                    }
                    try emit(record)
                    opRecords += 1
                    try requireFootprintBelowLimit("op \(index) (\(kindName(op.kind)))")
                }

                // Gradient checks: tensors whose gradient-carrying consumers all ran in this chunk.
                for id in gradientIDs {
                    let carrying = gradientConsumers(id)
                    guard !carrying.isEmpty, carrying.allSatisfy({ range.contains($0) }) else { continue }
                    guard let mps = mpsGradients[id], let type = gradientTypes[id] else { continue }
                    var record: Record = baseRecord(auditCase, arch: arch)
                    record["event"] = "gradient"
                    record["tensor_id"] = id
                    record["tensor_name"] = tensorNames[id] ?? ""
                    record["producer_op_index"] = producer[id] ?? -1
                    record["producer_kind"] = producer[id].map { kindName(operations[$0].kind) } ?? "leaf"
                    record["consumer_op_indices"] = carrying
                    record["consumer_kinds"] = carrying.map { kindName(operations[$0].kind) }
                    record["dtype"] = type.rawValue
                    record["shape"] = mps.shape
                    if lossConsumed.contains(id) {
                        record["skipped"] = "also read by the audit loss"
                    } else if vjpFailed.contains(id) {
                        record["skipped"] = "a consumer's reference gradient failed"
                    } else if let reference = contributions[id], reference.shape == mps.shape,
                              let magnitude = magnitudes[id] {
                        var gradientRecord = comparisonRecord(GraphAuditReference.compare(
                            actual: mps.values, reference: reference.values, dataType: type))
                        gradientRecord["scale_rel_err"] = scaleRelativeError(actual: mps.values,
                                                                             reference: reference.values,
                                                                             magnitude: magnitude.values)
                        record["gradient"] = gradientRecord
                    } else {
                        record["skipped"] = "no reference contribution with the gradient's shape"
                    }
                    try emit(record)
                    gradientRecords += 1
                }
                let seconds = Double(DispatchTime.now().uptimeNanoseconds - started.uptimeNanoseconds) / 1e9
                let footprint = try footprintBytes()
                progress("\(auditCase.label): chunk \(chunkIndex + 1)/\(chunks.count) ops \(range.lowerBound)-"
                         + "\(range.upperBound - 1) (\(String(format: "%.1f", seconds)) s, footprint \(footprint >> 20) MB)")
            }
        }
        var summary = baseRecord(auditCase, arch: arch)
        summary["event"] = "case_done"
        summary["op_records"] = opRecords
        summary["gradient_records"] = gradientRecords
        summary["chunks"] = chunks.count
        try emit(summary)
    }

    /// Every tensor `roots` are computed from, through each op's inputs.
    static func ancestorTensors(of roots: [MPSGraphTensor]) -> Set<ObjectIdentifier> {
        var found = Set<ObjectIdentifier>()
        var visitedOps = Set<ObjectIdentifier>()
        var stack = roots
        while let t = stack.popLast() {
            guard found.insert(ObjectIdentifier(t)).inserted else { continue }
            let op = t.operation
            guard visitedOps.insert(ObjectIdentifier(op)).inserted else { continue }
            stack.append(contentsOf: op.inputTensors)
        }
        return found
    }

    // MARK: - Chunking

    /// Consecutive op ranges whose CPU-side tensors fit `budget` bytes. A
    /// chunk is extended so a tensor's gradient-carrying consumers close by
    /// (within 16 ops) fall in the same chunk — a BatchNorm's mean, variance and
    /// normalize, which share their input — but never past twice the budget.
    /// `cost(id)` is what one tensor costs the chunk once (each id is counted
    /// once per chunk).
    static func makeChunks(operations: [GraphOperation], budget: Int, cost: (String) -> Int,
                           gradientConsumers: (String) -> [Int]) -> [Range<Int>] {
        func opCost(_ index: Int, counted: inout Set<String>) -> Int {
            let op = operations[index]
            if case .constant = op.kind { return 0 }
            var total = 0
            for id in [op.id] + op.kind.operands where counted.insert(id).inserted { total += cost(id) }
            return total
        }
        var chunks: [Range<Int>] = []
        var start = 0
        while start < operations.count {
            var end = start
            var bytes = 0
            var counted = Set<String>()
            while end < operations.count {
                var trial = counted
                let add = opCost(end, counted: &trial)
                if end > start, bytes + add > budget { break }
                bytes += add
                counted = trial
                end += 1
            }
            var extended = true
            while extended {
                extended = false
                for index in start..<end {
                    for id in operations[index].kind.operands {
                        for c in gradientConsumers(id) where c >= end && c < end + 16 {
                            var trial = counted
                            var add = 0
                            for k in end...c { add += opCost(k, counted: &trial) }
                            guard bytes + add <= 2 * budget else { continue }
                            bytes += add
                            counted = trial
                            end = c + 1
                            extended = true
                        }
                    }
                }
            }
            chunks.append(start..<end)
            start = end
        }
        return chunks
    }

    /// The process's physical footprint (`phys_footprint`, which includes its
    /// GPU allocations), in bytes.
    static func footprintBytes() throws -> UInt64 {
        switch ChessTrainer.readPhysFootprintBytes() {
        case .success(let bytes): return bytes
        case .failure(let error): throw AuditError.footprintUnreadable(error.kernReturn)
        }
    }

    // MARK: - References per op kind

    static func forward(_ kind: GraphOperationKind, operands x: [Tensor64], outputShape: [Int],
                        outputType: GraphDataType) throws -> Tensor64 {
        typealias R = GraphAuditReference
        switch kind {
        case .cast:
            return Tensor64(shape: x[0].shape, values: x[0].values.map { R.rounded($0, to: outputType) })
        case .convolution2D(_, _, let sy, let sx, let dy, let dx, let groups, let pt, let pb, let pl, let pr):
            guard groups == 1 else { throw GraphAuditReferenceError.unsupported("grouped convolution") }
            return try R.convolution(x[0], weights: x[1], geometry: .init(
                strideY: sy, strideX: sx, dilationY: dy, dilationX: dx,
                paddingTop: pt, paddingBottom: pb, paddingLeft: pl, paddingRight: pr))
        case .normalize(_, _, _, _, _, let epsilon):
            return R.normalize(x[0], mean: x[1], variance: x[2], gamma: x[3], beta: x[4], epsilon: Double(epsilon))
        case .mean(_, let axes):
            return R.reduceMean(x[0], axes: axes)
        case .variance(_, let axes):
            return R.reduceVariance(x[0], axes: axes)
        case .reductionSum(_, let axis):
            return R.reduceSum(x[0], axis: axis)
        case .reshape:
            return Tensor64(shape: outputShape, values: x[0].values)
        case .matrixMultiplication:
            return try R.matmul(x[0], x[1])
        case .addition: return try R.binary(.add, x[0], x[1])
        case .subtraction: return try R.binary(.subtract, x[0], x[1])
        case .multiplication: return try R.binary(.multiply, x[0], x[1])
        case .division: return try R.binary(.divide, x[0], x[1])
        case .tanh: return R.unary(.tanh, x[0])
        case .sigmoid: return R.unary(.sigmoid, x[0])
        case .erf: return R.unary(.erf, x[0])
        case .reLU: return R.unary(.relu, x[0])
        case .leakyReLU(_, let alpha): return R.unary(.leakyReLU(alpha), x[0])
        case .softMax(_, let axis): return R.softmax(x[0], axis: axis)
        case .concat(_, let dimension): return try R.concat(x, dimension: dimension)
        case .slice(_, let dimension, let start, let length):
            return R.slice(x[0], dimension: dimension, start: start, length: length)
        case .constant:
            throw GraphAuditReferenceError.unsupported("constant")
        }
    }

    /// Each operand's gradient given the output's (nil where the op has none).
    static func vjp(_ kind: GraphOperationKind, operands x: [Tensor64], dY: Tensor64) throws -> [Tensor64?] {
        typealias R = GraphAuditReference
        switch kind {
        case .cast:
            return [dY]
        case .convolution2D(_, _, let sy, let sx, let dy, let dx, let groups, let pt, let pb, let pl, let pr):
            guard groups == 1 else { throw GraphAuditReferenceError.unsupported("grouped convolution") }
            let g = try R.convolutionGradients(x[0], weights: x[1], geometry: .init(
                strideY: sy, strideX: sx, dilationY: dy, dilationX: dx,
                paddingTop: pt, paddingBottom: pb, paddingLeft: pl, paddingRight: pr), dY: dY)
            return [g.dx, g.dWeights]
        case .normalize(_, _, _, _, _, let epsilon):
            let g = R.normalizeGradients(x[0], mean: x[1], variance: x[2], gamma: x[3], beta: x[4],
                                         epsilon: Double(epsilon), dY: dY)
            return [g.dx, g.dMean, g.dVariance, g.dGamma, g.dBeta]
        case .mean(_, let axes):
            return [R.meanGradient(x[0], axes: axes, dY: dY)]
        case .variance(_, let axes):
            return [R.varianceGradient(x[0], axes: axes, dY: dY)]
        case .reductionSum:
            return [R.broadcast(dY, to: x[0].shape)]
        case .reshape:
            return [Tensor64(shape: x[0].shape, values: dY.values)]
        case .matrixMultiplication:
            let g = try R.matmulGradients(x[0], x[1], dY: dY)
            return [g.0, g.1]
        case .addition: let g = try R.binaryGradients(.add, x[0], x[1], dY: dY); return [g.0, g.1]
        case .subtraction: let g = try R.binaryGradients(.subtract, x[0], x[1], dY: dY); return [g.0, g.1]
        case .multiplication: let g = try R.binaryGradients(.multiply, x[0], x[1], dY: dY); return [g.0, g.1]
        case .division: let g = try R.binaryGradients(.divide, x[0], x[1], dY: dY); return [g.0, g.1]
        case .tanh: return [R.unaryGradient(.tanh, x[0], dY: dY)]
        case .sigmoid: return [R.unaryGradient(.sigmoid, x[0], dY: dY)]
        case .erf: return [R.unaryGradient(.erf, x[0], dY: dY)]
        case .reLU: return [R.unaryGradient(.relu, x[0], dY: dY)]
        case .leakyReLU(_, let alpha): return [R.unaryGradient(.leakyReLU(alpha), x[0], dY: dY)]
        case .softMax(_, let axis): return [R.softmaxGradient(x[0], axis: axis, dY: dY)]
        case .concat(_, let dimension): return R.concatGradients(x, dimension: dimension, dY: dY)
        case .slice(_, let dimension, let start, _):
            return [R.sliceGradient(inputShape: x[0].shape, dimension: dimension, start: start, dY: dY)]
        case .constant:
            return []
        }
    }

    // MARK: - Magnitudes (the scale of the terms a result sums)

    /// ‖actual − reference‖ / ‖magnitude‖: the error relative to the size of
    /// the terms the result was summed from, not to the (possibly cancelled)
    /// result. 0 when the magnitude is 0 (then the reference is 0 too).
    static func scaleRelativeError(actual: [Double], reference: [Double], magnitude: [Double]) -> Double {
        var diff2 = 0.0, mag2 = 0.0
        for k in 0..<reference.count where actual[k].isFinite {
            let d = actual[k] - reference[k]
            diff2 += d * d
            mag2 += magnitude[k] * magnitude[k]
        }
        return mag2 > 0 ? (diff2 / mag2).squareRoot() : 0
    }

    private static func absolute(_ t: Tensor64) -> Tensor64 {
        Tensor64(shape: t.shape, values: t.values.map { abs($0) })
    }

    private static func accumulate(_ t: Tensor64, into sums: inout [String: Tensor64], id: String) throws {
        if var sum = sums[id] {
            guard sum.shape == t.shape else { throw GraphAuditReferenceError.shapeMismatch("contribution to \(id)") }
            for k in 0..<sum.count { sum.values[k] += t.values[k] }
            sums[id] = sum
        } else {
            sums[id] = t
        }
    }

    /// The forward result computed from the absolute values of the terms it
    /// sums: abs operands for the (multi)linear ops; |result| for elementwise
    /// nonlinear ones; |x − mean|·|γ|/s + |β| for normalize.
    static func forwardMagnitude(_ kind: GraphOperationKind, operands x: [Tensor64], reference: Tensor64,
                                 outputShape: [Int]) throws -> Tensor64 {
        switch kind {
        case .convolution2D, .matrixMultiplication, .mean, .reductionSum, .addition, .subtraction,
             .multiplication, .division:
            return absolute(try forward(kind, operands: x.map(absolute), outputShape: outputShape,
                                        outputType: .float32))
        case .normalize(_, _, _, _, _, let epsilon):
            let centered = try GraphAuditReference.binary(.subtract, x[0], x[1])
            let scaled = GraphAuditReference.normalize(
                absolute(centered), mean: Tensor64(zeros: [1]), variance: x[2], gamma: absolute(x[3]),
                beta: absolute(x[4]), epsilon: Double(epsilon))
            return scaled
        default:
            return absolute(reference)
        }
    }

    /// Each operand's gradient computed from the absolute values of the terms
    /// it sums, for the scale its cancellation is judged on.
    static func vjpMagnitude(_ kind: GraphOperationKind, operands x: [Tensor64], dY: Tensor64) throws -> [Tensor64?] {
        let absDY = absolute(dY)
        switch kind {
        case .convolution2D, .matrixMultiplication, .mean, .reductionSum, .addition, .subtraction,
             .multiplication, .division, .concat, .slice, .reshape, .cast:
            return try vjp(kind, operands: x.map(absolute), dY: absDY).map { $0.map(absolute) }
        case .tanh, .sigmoid, .erf, .reLU, .leakyReLU:
            return try vjp(kind, operands: x, dY: absDY).map { $0.map(absolute) }
        case .softMax(_, let axis):
            // y·(|dY| + Σ|dY|·y): both terms of y·(dY − Σ dY·y) in magnitude.
            let y = GraphAuditReference.softmax(x[0], axis: axis)
            let negated = GraphAuditReference.softmaxGradient(x[0], axis: axis,
                                                              dY: Tensor64(shape: dY.shape, values: absDY.values.map { -$0 }))
            var out = y
            // negated = y·(−|dY| + S), S = Σ|dY|·y, so y·(|dY| + S) = 2·y·|dY| + negated.
            for k in 0..<y.count { out.values[k] = 2 * y.values[k] * absDY.values[k] + negated.values[k] }
            return [absolute(out)]
        case .variance(_, let axes):
            let n = Double(axes.reduce(1) { $0 * x[0].shape[$1] })
            let meanB = GraphAuditReference.broadcast(GraphAuditReference.reduceMean(x[0], axes: axes), to: x[0].shape)
            let g = GraphAuditReference.broadcast(absDY, to: x[0].shape)
            var out = x[0]
            for k in 0..<out.count { out.values[k] = g.values[k] * 2 * abs(x[0].values[k] - meanB.values[k]) / n }
            return [out]
        case .normalize(_, _, _, _, _, let epsilon):
            let xs = x[0], mean = x[1], variance = x[2], gamma = x[3], beta = x[4]
            var dx = Tensor64(zeros: xs.shape), dMean = Tensor64(zeros: mean.shape)
            var dVar = Tensor64(zeros: variance.shape), dGamma = Tensor64(zeros: gamma.shape)
            var dBeta = Tensor64(zeros: beta.shape)
            GraphAuditReference.walk(xs.shape, [mean.shape, variance.shape, gamma.shape, beta.shape]) { k, i in
                let vePlus = variance.values[i[1]] + Double(epsilon)
                let s = vePlus.squareRoot()
                let centered = abs(xs.values[k] - mean.values[i[0]])
                let g = absDY.values[k], gammaK = abs(gamma.values[i[2]])
                dx.values[k] = g * gammaK / s
                dMean.values[i[0]] += g * gammaK / s
                dVar.values[i[1]] += g * gammaK * centered * 0.5 / (vePlus * s)
                dGamma.values[i[2]] += g * centered / s
                dBeta.values[i[3]] += g
            }
            return [dx, dMean, dVar, dGamma, dBeta]
        case .constant:
            return []
        }
    }

    static func kindName(_ kind: GraphOperationKind) -> String {
        switch kind {
        case .cast: return "cast"
        case .convolution2D: return "convolution2D"
        case .normalize: return "normalize"
        case .mean: return "mean"
        case .variance: return "variance"
        case .reductionSum: return "reductionSum"
        case .reshape: return "reshape"
        case .matrixMultiplication: return "matrixMultiplication"
        case .addition: return "addition"
        case .subtraction: return "subtraction"
        case .multiplication: return "multiplication"
        case .division: return "division"
        case .tanh: return "tanh"
        case .sigmoid: return "sigmoid"
        case .erf: return "erf"
        case .reLU: return "reLU"
        case .leakyReLU: return "leakyReLU"
        case .softMax: return "softMax"
        case .concat: return "concat"
        case .slice: return "slice"
        case .constant: return "constant"
        }
    }

    // MARK: - Feeds and readback

    /// The board input: the bundled puzzle positions, cycled to `batch`.
    private static func boardFeed(network: ChessNetwork, batch: Int) throws -> MPSGraphTensorData {
        let probes = LichessProbeData.modelFileTestSets.flatMap(\.probes)
        let encoding = network.arch.inputEncoding
        let perPosition = BoardEncoder.tensorLength(for: encoding)
        let encoded = TacticalProbeRunner.encodedBoards(probes, encoding: encoding)
        let positions = encoded.count / perPosition
        var input = [Float](repeating: 0, count: batch * perPosition)
        for b in 0..<batch {
            let source = (b % positions) * perPosition
            for k in 0..<perPosition { input[b * perPosition + k] = encoded[source + k] }
        }
        let planes = perPosition / 64
        return floatFeed(network: network, values: input, shape: [batch, planes, 8, 8])
    }

    /// Uniform(−1, 1) loss weights from a fixed seed.
    private static func lossWeightsFeed(network: ChessNetwork, batch: Int, width: Int, seed: UInt64) throws
        -> MPSGraphTensorData {
        var generator = DCMRandom(seed: seed)
        let values = (0..<(batch * width)).map { _ in Float.random(in: -1...1, using: &generator) }
        return floatFeed(network: network, values: values, shape: [batch, width])
    }

    private static func floatFeed(network: ChessNetwork, values: [Float], shape: [Int]) -> MPSGraphTensorData {
        let data = values.withUnsafeBufferPointer { Data(buffer: $0) }
        return MPSGraphTensorData(device: network.graphDevice, data: data,
                                  shape: shape.map { NSNumber(value: $0) }, dataType: .float32)
    }

    private static func graphDataType(_ type: MPSDataType, _ what: String) throws -> GraphDataType {
        guard let t = GraphDataType(type) else {
            throw AuditError.unsupportedDataType("\(type.rawValue) at \(what)")
        }
        return t
    }

    /// A tensor's values as float64, from fp32, fp16 or bf16 storage.
    static func readTensor(_ data: MPSGraphTensorData) throws -> Tensor64 {
        let shape = data.shape.map(\.intValue)
        let count = shape.reduce(1, *)
        let array = data.mpsndarray()
        switch data.dataType {
        case .float32:
            var buffer = [Float](repeating: 0, count: count)
            buffer.withUnsafeMutableBytes { raw in
                if let base = raw.baseAddress { array.readBytes(base, strideBytes: nil) }
            }
            return Tensor64(shape: shape, values: buffer.map(Double.init))
        case .float16:
            var buffer = [Float16](repeating: 0, count: count)
            buffer.withUnsafeMutableBytes { raw in
                if let base = raw.baseAddress { array.readBytes(base, strideBytes: nil) }
            }
            return Tensor64(shape: shape, values: buffer.map { Double(Float($0)) })
        case .bFloat16:
            var buffer = [UInt16](repeating: 0, count: count)
            buffer.withUnsafeMutableBytes { raw in
                if let base = raw.baseAddress { array.readBytes(base, strideBytes: nil) }
            }
            return Tensor64(shape: shape, values: buffer.map { Double(Float(bitPattern: UInt32($0) << 16)) })
        case .bool, .uInt8:
            var buffer = [UInt8](repeating: 0, count: count)
            buffer.withUnsafeMutableBytes { raw in
                if let base = raw.baseAddress { array.readBytes(base, strideBytes: nil) }
            }
            return Tensor64(shape: shape, values: buffer.map(Double.init))
        case .int32:
            var buffer = [Int32](repeating: 0, count: count)
            buffer.withUnsafeMutableBytes { raw in
                if let base = raw.baseAddress { array.readBytes(base, strideBytes: nil) }
            }
            return Tensor64(shape: shape, values: buffer.map(Double.init))
        default:
            throw AuditError.unsupportedDataType("\(data.dataType.rawValue)")
        }
    }

    // MARK: - Records

    private static func baseRecord(_ auditCase: Case, arch: NetworkArchitecture) -> Record {
        [
            "case": auditCase.label,
            "mode": auditCase.mode.rawValue,
            "batch": auditCase.batch,
            "compute_data_type": arch.computeDataType.rawValue,
            "policy_tail_precision": arch.policyTailPrecision.rawValue,
        ]
    }

    private static func comparisonRecord(_ c: GraphAuditReference.Comparison) -> Record {
        var r: Record = [
            "max_abs_err": c.maxAbsoluteError, "ref_norm": c.referenceNorm, "ref_max_abs": c.referenceMaxAbs,
            "non_finite": c.nonFiniteActual, "count": c.count,
        ]
        if let e = c.relativeError { r["rel_err"] = e }
        if let f = c.roundingFloor { r["rounding_floor"] = f }
        return r
    }

    /// The recorded ids that are `tensor` (an output the audit loss reads).
    private static func recordedIDs(of tensor: MPSGraphTensor, recorder: InferenceGraphRecorder,
                                    operations: [GraphOperation]) -> [String] {
        operations.compactMap { op in
            recorder.tensor(forID: op.id).map { ObjectIdentifier($0) == ObjectIdentifier(tensor) } == true ? op.id : nil
        }
    }
}
