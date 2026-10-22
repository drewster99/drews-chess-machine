import CoreML
import Foundation

/// How closely a Core ML model exported from a network (`InferenceGraphExport`
/// + `scripts/dcm_coreml_export.py`) reproduces that network's MPSGraph
/// forward pass on the same positions, at one Core ML batch size.
///
/// The check an export has to pass before anything plays with it, and the one
/// `--coreml-parity` and `CoreMLExportParityTests` share. Differences are
/// measured on the raw outputs (every policy logit, the value scalar, the W/D/L
/// probabilities) and on the move each side ranks first among the legal moves —
/// the quantity play depends on.
struct CoreMLParityReport: Encodable, Sendable {
    /// The Core ML batch size the predictions ran at.
    let batchSize: Int
    let positions: Int
    let computeUnits: String
    /// Largest |MPSGraph − Core ML| over every policy logit of every position.
    let policyMaxAbsDiff: Float
    let policyMeanAbsDiff: Float
    /// The same, over the logits of legal moves only.
    let legalPolicyMaxAbsDiff: Float
    let valueMaxAbsDiff: Float
    let valueProbsMaxAbsDiff: Float
    /// Positions where both rank the same legal move first.
    let topMoveAgreements: Int
    /// Positions with a non-finite Core ML output.
    let nonFiniteCoreMLPositions: Int
    /// Positions with a non-finite MPSGraph output (a broken model, not an
    /// export fault).
    let nonFiniteReferencePositions: Int
    /// Positions with no legal move (checkmate, stalemate): no top move to
    /// compare, so not in `topMoveAgreements`.
    let positionsWithoutLegalMoves: Int

    enum CodingKeys: String, CodingKey {
        case positions
        case batchSize = "batch_size"
        case computeUnits = "compute_units"
        case policyMaxAbsDiff = "policy_max_abs_diff"
        case policyMeanAbsDiff = "policy_mean_abs_diff"
        case legalPolicyMaxAbsDiff = "legal_policy_max_abs_diff"
        case valueMaxAbsDiff = "value_max_abs_diff"
        case valueProbsMaxAbsDiff = "value_probs_max_abs_diff"
        case topMoveAgreements = "top_move_agreements"
        case nonFiniteCoreMLPositions = "non_finite_coreml_positions"
        case nonFiniteReferencePositions = "non_finite_reference_positions"
        case positionsWithoutLegalMoves = "positions_without_legal_moves"
    }

    /// Whether every raw-output difference (over the positions where both
    /// sides are finite) is at most `tolerance` and nothing on either side was
    /// non-finite.
    func isWithin(tolerance: Float) -> Bool {
        nonFiniteCoreMLPositions == 0 && nonFiniteReferencePositions == 0
            && policyMaxAbsDiff <= tolerance && valueMaxAbsDiff <= tolerance && valueProbsMaxAbsDiff <= tolerance
    }
}

enum CoreMLParity {
    /// The Core ML model's input and output names, as the exporter writes them
    /// (`InferenceGraph`'s input id and output names).
    static let inputName = InferenceGraphRecorder.inputID

    enum ParityError: LocalizedError {
        case missingOutput(String)
        case outputSize(name: String, have: Int, want: Int)
        case networkOutputSize(have: Int, want: Int)
        case noBatchSizes
        case unsupportedInputShape(String)
        /// The compiled model was left on disk; `failure` is the error the
        /// comparison itself ended with, when it ended with one, so a cleanup
        /// problem never hides why the comparison failed.
        case compiledModelRemains(path: String, reason: String, failure: String?)

        var errorDescription: String? {
            switch self {
            case .missingOutput(let name):
                return "the Core ML model has no output '\(name)'"
            case .outputSize(let name, let have, let want):
                return "Core ML output '\(name)' holds \(have) values; expected \(want)"
            case .networkOutputSize(let have, let want):
                return "the network returned \(have) values; expected \(want)"
            case .noBatchSizes:
                return "the Core ML model's input names no batch size this check can use (it needs batch size 1)"
            case .unsupportedInputShape(let detail):
                return "the Core ML model's input shape is not one this check reads: \(detail)"
            case .compiledModelRemains(let path, let reason, let failure):
                let removal = "the compiled model at \(path) could not be removed: \(reason)"
                return failure.map { "\($0) (and \(removal))" } ?? removal
            }
        }
    }

    enum ComputeUnits: String, CaseIterable, Sendable {
        case cpu
        case cpuAndGPU = "cpu-gpu"
        case cpuAndNeuralEngine = "cpu-ne"
        case all

        var mlComputeUnits: MLComputeUnits {
            switch self {
            case .cpu: return .cpuOnly
            case .cpuAndGPU: return .cpuAndGPU
            case .cpuAndNeuralEngine: return .cpuAndNeuralEngine
            case .all: return .all
            }
        }
    }

    struct Outputs: Sendable {
        var policy: [Float]
        var value: [Float]
        var valueProbabilities: [Float]
    }

    /// Runs `body` with the Core ML model at `url` loaded. An `.mlpackage` is
    /// compiled first, and the compiled copy (Core ML's temporary, the
    /// caller's to remove) is removed afterwards by identity; an `.mlmodelc`
    /// is loaded as it is.
    static func withModel<T: Sendable>(at url: URL, computeUnits: ComputeUnits,
                                       _ body: (MLModel) async throws -> T) async throws -> T {
        try await withCompiledModel(at: url, computeUnits: computeUnits) { model, _ in try await body(model) }
    }

    /// `withModel`, also handing `body` the compiled model's folder (for what
    /// reads it again, such as `MLComputePlan`); valid only inside `body`.
    static func withCompiledModel<T: Sendable>(at url: URL, computeUnits: ComputeUnits,
                                               _ body: (MLModel, URL) async throws -> T) async throws -> T {
        let configuration = MLModelConfiguration()
        configuration.computeUnits = computeUnits.mlComputeUnits
        guard url.pathExtension != "mlmodelc" else {
            return try await body(try await MLModel.load(contentsOf: url, configuration: configuration), url)
        }
        let compiled = try await MLModel.compileModel(at: url)
        // The compiled folder is this call's own: its identity is taken before
        // anything else can fail, so it is removed (by identity) on every path.
        let identity: FileSafety.FileIdentity
        do {
            guard let item = try FileSafety.existingItem(at: compiled) else {
                throw ParityError.compiledModelRemains(path: compiled.path,
                                                       reason: "Core ML returned a path with nothing at it",
                                                       failure: nil)
            }
            identity = item.identity
        } catch let error as ParityError {
            throw error
        } catch {
            throw ParityError.compiledModelRemains(path: compiled.path, reason: error.localizedDescription, failure: nil)
        }
        let result: Result<T, Error>
        do {
            result = .success(try await body(try await MLModel.load(contentsOf: compiled, configuration: configuration),
                                             compiled))
        } catch {
            result = .failure(error)
        }
        do {
            try FileSafety.removeOwnedItem(at: compiled, identity: identity)
        } catch {
            let failure: String?
            if case .failure(let bodyError) = result { failure = bodyError.localizedDescription } else { failure = nil }
            throw ParityError.compiledModelRemains(path: compiled.path, reason: error.localizedDescription,
                                                   failure: failure)
        }
        return try result.get()
    }

    /// The batch sizes the model's input accepts (its enumerated shapes), or
    /// its one fixed batch size; batch size 1 must be among them.
    static func batchSizes(of model: MLModel) throws -> [Int] {
        guard let constraint = model.modelDescription.inputDescriptionsByName[inputName]?.multiArrayConstraint else {
            throw ParityError.noBatchSizes
        }
        let shapeConstraint = constraint.shapeConstraint
        // A range-flexible input would be tested at its default shape only,
        // which is not what the comparison claims to check; the exporter writes
        // enumerated shapes, so anything else is refused, never guessed at.
        guard shapeConstraint.type != .range else {
            throw ParityError.unsupportedInputShape("a range of shapes (the exporter writes enumerated batch sizes)")
        }
        let shapes = shapeConstraint.type == .enumerated ? shapeConstraint.enumeratedShapes : [constraint.shape]
        var sizes: [Int] = []
        for shape in shapes {
            guard shape.count == 4 else {
                throw ParityError.unsupportedInputShape("\(shape.map(\.intValue)) is not [batch, planes, 8, 8]")
            }
            sizes.append(shape[0].intValue)
        }
        let sorted = Array(Set(sizes)).sorted()
        guard sorted.contains(1) else { throw ParityError.noBatchSizes }
        return sorted
    }

    /// Core ML's outputs for `count` encoded boards, `batchSize` per prediction;
    /// a remainder smaller than `batchSize` runs one board at a time.
    static func predict(model: MLModel, boards: [Float], count: Int, planes: Int,
                        valueClasses: Int, batchSize: Int) async throws -> Outputs {
        let boardLength = planes * ChessNetwork.boardSize * ChessNetwork.boardSize
        precondition(boards.count == count * boardLength, "boards hold count encoded positions")
        var outputs = Outputs(policy: [], value: [], valueProbabilities: [])
        outputs.policy.reserveCapacity(count * ChessNetwork.policySize)
        var start = 0
        while start < count {
            let size = count - start >= batchSize ? batchSize : 1
            let slice = Array(boards[(start * boardLength)..<((start + size) * boardLength)])
            let input = MLMultiArray(MLShapedArray<Float>(scalars: slice, shape: [size, planes, 8, 8]))
            let features = try MLDictionaryFeatureProvider(dictionary: [inputName: MLFeatureValue(multiArray: input)])
            let result = try await model.prediction(from: features)
            outputs.policy += try values(result, InferenceGraphOutputName.policy.rawValue, count: size * ChessNetwork.policySize)
            outputs.value += try values(result, InferenceGraphOutputName.value.rawValue, count: size)
            outputs.valueProbabilities += try values(
                result, InferenceGraphOutputName.valueProbabilities.rawValue, count: size * valueClasses)
            start += size
        }
        return outputs
    }

    private static func values(_ provider: MLFeatureProvider, _ name: String, count: Int) throws -> [Float] {
        guard let array = provider.featureValue(for: name)?.multiArrayValue else {
            throw ParityError.missingOutput(name)
        }
        let scalars = MLShapedArray<Float>(converting: array).scalars
        guard scalars.count == count else { throw ParityError.outputSize(name: name, have: scalars.count, want: count) }
        return scalars
    }

    /// The network's outputs for the same boards, through the batched forward
    /// pass every inference consumer uses.
    static func networkOutputs(network: ChessNetwork, boards: [Float], count: Int) async throws -> Outputs {
        let box = SyncBox(Outputs(policy: [], value: [], valueProbabilities: []))
        try await network.evaluateBatched(batchBoards: boards, count: count) { policy, value, probabilities in
            box.value = Outputs(policy: Array(policy), value: Array(value), valueProbabilities: Array(probabilities))
        }
        let outputs = box.value
        guard outputs.policy.count == count * ChessNetwork.policySize else {
            throw ParityError.networkOutputSize(have: outputs.policy.count, want: count * ChessNetwork.policySize)
        }
        return outputs
    }

    /// Compares `model` with `network` on `states`, once per batch size the
    /// model accepts.
    static func compare(network: ChessNetwork, model: MLModel, states: [GameState],
                        computeUnits: ComputeUnits) async throws -> [CoreMLParityReport] {
        try await compare(network: network, model: model, states: states, computeUnits: computeUnits,
                          batchSizes: try batchSizes(of: model))
    }

    /// Compares `model` with `network` on `states` at each of `batchSizes`
    /// (sizes the model accepts; `--coreml-benchmark` checks exactly the sizes
    /// it times).
    static func compare(network: ChessNetwork, model: MLModel, states: [GameState],
                        computeUnits: ComputeUnits, batchSizes: [Int]) async throws -> [CoreMLParityReport] {
        let arch = network.arch
        let boards = await onWorkQueue {
            var boards: [Float] = []
            boards.reserveCapacity(states.count * BoardEncoder.tensorLength(for: arch.inputEncoding))
            for state in states { boards += BoardEncoder.encode(state, encoding: arch.inputEncoding) }
            return boards
        }
        let reference = try await networkOutputs(network: network, boards: boards, count: states.count)
        var reports: [CoreMLParityReport] = []
        for batchSize in batchSizes {
            let candidate = try await predict(model: model, boards: boards, count: states.count,
                                              planes: arch.inputPlanes, valueClasses: arch.valueHeadClasses,
                                              batchSize: batchSize)
            reports.append(await onWorkQueue {
                report(states: states, reference: reference, candidate: candidate,
                       valueClasses: arch.valueHeadClasses, batchSize: batchSize, computeUnits: computeUnits)
            })
        }
        return reports
    }

    /// Runs CPU-bound work (encoding, move generation over thousands of
    /// positions) on a GCD queue, off the cooperative pool.
    private static func onWorkQueue<T: Sendable>(_ work: @escaping @Sendable () -> T) async -> T {
        await withCheckedContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async { continuation.resume(returning: work()) }
        }
    }

    static func report(states: [GameState], reference: Outputs, candidate: Outputs, valueClasses: Int,
                       batchSize: Int, computeUnits: ComputeUnits) -> CoreMLParityReport {
        let size = ChessNetwork.policySize
        var policyMax: Float = 0, policySum: Double = 0, policyCount = 0
        var legalMax: Float = 0, valueMax: Float = 0, probabilitiesMax: Float = 0
        var agreements = 0, nonFinite = 0, nonFiniteReference = 0, withoutLegalMoves = 0
        for (index, state) in states.enumerated() {
            let ref = reference.policy[(index * size)..<((index + 1) * size)]
            let cand = candidate.policy[(index * size)..<((index + 1) * size)]
            let refProbabilities = reference.valueProbabilities[(index * valueClasses)..<((index + 1) * valueClasses)]
            let candProbabilities = candidate.valueProbabilities[(index * valueClasses)..<((index + 1) * valueClasses)]
            let referenceFinite = ref.allSatisfy(\.isFinite) && reference.value[index].isFinite
                && refProbabilities.allSatisfy(\.isFinite)
            let candidateFinite = cand.allSatisfy(\.isFinite) && candidate.value[index].isFinite
                && candProbabilities.allSatisfy(\.isFinite)
            if !referenceFinite { nonFiniteReference += 1 }
            if !candidateFinite { nonFinite += 1 }
            guard referenceFinite, candidateFinite else { continue }
            for (r, c) in zip(ref, cand) {
                let d = abs(r - c)
                policyMax = max(policyMax, d)
                policySum += Double(d)
                policyCount += 1
            }
            valueMax = max(valueMax, abs(reference.value[index] - candidate.value[index]))
            for (r, c) in zip(refProbabilities, candProbabilities) { probabilitiesMax = max(probabilitiesMax, abs(r - c)) }

            let legalIndices = MoveGenerator.legalMoves(for: state)
                .map { PolicyEncoding.policyIndex($0, currentPlayer: state.currentPlayer) }
            guard !legalIndices.isEmpty else {
                withoutLegalMoves += 1
                continue
            }
            var refBest: Int?, candBest: Int?
            for i in legalIndices {
                legalMax = max(legalMax, abs(ref[ref.startIndex + i] - cand[cand.startIndex + i]))
                if refBest.map({ ref[ref.startIndex + i] > ref[ref.startIndex + $0] }) ?? true { refBest = i }
                if candBest.map({ cand[cand.startIndex + i] > cand[cand.startIndex + $0] }) ?? true { candBest = i }
            }
            if refBest == candBest { agreements += 1 }
        }
        return CoreMLParityReport(
            batchSize: batchSize, positions: states.count, computeUnits: computeUnits.rawValue,
            policyMaxAbsDiff: policyMax, policyMeanAbsDiff: policyCount == 0 ? 0 : Float(policySum / Double(policyCount)),
            legalPolicyMaxAbsDiff: legalMax, valueMaxAbsDiff: valueMax, valueProbsMaxAbsDiff: probabilitiesMax,
            topMoveAgreements: agreements, nonFiniteCoreMLPositions: nonFinite,
            nonFiniteReferencePositions: nonFiniteReference, positionsWithoutLegalMoves: withoutLegalMoves)
    }
}
