import CoreML
import Foundation

/// How closely a Core ML model exported from a network (`InferenceGraphExport`
/// + `scripts/dcm_coreml_export.py`) reproduces that network's MPSGraph
/// forward pass on the same positions.
///
/// The check an export has to pass before anything plays with it, and the one
/// `--coreml-parity` and `CoreMLExportParityTests` share. Differences are
/// measured on the raw outputs (every policy logit, the value scalar, the W/D/L
/// probabilities) and on the move each side would rank first among the legal
/// moves — the quantity play depends on.
struct CoreMLParityReport: Encodable, Sendable {
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
    /// Positions with a non-finite Core ML output; their differences are not in
    /// the maxima above.
    let nonFiniteCoreMLPositions: Int
    /// Positions with a non-finite MPSGraph output (a broken model, not an
    /// export fault); also left out of the maxima.
    let nonFiniteReferencePositions: Int
    /// Positions with no legal move (checkmate, stalemate): there is no top
    /// move to compare, so they are not in `topMoveAgreements`.
    let positionsWithoutLegalMoves: Int

    enum CodingKeys: String, CodingKey {
        case positions
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

    /// Whether every raw-output difference is at most `tolerance` and nothing
    /// on either side was non-finite.
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

        var errorDescription: String? {
            switch self {
            case .missingOutput(let name):
                return "the Core ML model has no output '\(name)'"
            case .outputSize(let name, let have, let want):
                return "Core ML output '\(name)' holds \(have) values; expected \(want)"
            case .networkOutputSize(let have, let want):
                return "the network returned \(have) values; expected \(want)"
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

    /// Loads an `.mlpackage` (compiled first; the compiled copy is Core ML's
    /// temporary) or an already compiled `.mlmodelc`.
    static func loadModel(at url: URL, computeUnits: ComputeUnits) async throws -> MLModel {
        let compiled = url.pathExtension == "mlmodelc" ? url : try await MLModel.compileModel(at: url)
        let configuration = MLModelConfiguration()
        configuration.computeUnits = computeUnits.mlComputeUnits
        return try await MLModel.load(contentsOf: compiled, configuration: configuration)
    }

    /// Core ML's outputs for `count` encoded boards, one prediction each.
    static func predict(model: MLModel, boards: [Float], count: Int, planes: Int,
                        valueClasses: Int) async throws -> Outputs {
        let boardLength = planes * ChessNetwork.boardSize * ChessNetwork.boardSize
        precondition(boards.count == count * boardLength, "boards hold count encoded positions")
        var outputs = Outputs(policy: [], value: [], valueProbabilities: [])
        outputs.policy.reserveCapacity(count * ChessNetwork.policySize)
        for index in 0..<count {
            let board = Array(boards[(index * boardLength)..<((index + 1) * boardLength)])
            let input = MLMultiArray(MLShapedArray<Float>(scalars: board, shape: [1, planes, 8, 8]))
            let features = try MLDictionaryFeatureProvider(dictionary: [inputName: MLFeatureValue(multiArray: input)])
            let result = try await model.prediction(from: features)
            outputs.policy += try values(result, InferenceGraphOutputName.policy.rawValue, count: ChessNetwork.policySize)
            outputs.value += try values(result, InferenceGraphOutputName.value.rawValue, count: 1)
            outputs.valueProbabilities += try values(
                result, InferenceGraphOutputName.valueProbabilities.rawValue, count: valueClasses)
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

    /// Compares `model` with `network` on `states`.
    static func compare(network: ChessNetwork, model: MLModel, states: [GameState],
                        computeUnits: ComputeUnits) async throws -> CoreMLParityReport {
        let arch = network.arch
        let boards = await onWorkQueue {
            var boards: [Float] = []
            boards.reserveCapacity(states.count * BoardEncoder.tensorLength(for: arch.inputEncoding))
            for state in states { boards += BoardEncoder.encode(state, encoding: arch.inputEncoding) }
            return boards
        }
        let reference = try await networkOutputs(network: network, boards: boards, count: states.count)
        let candidate = try await predict(model: model, boards: boards, count: states.count,
                                          planes: arch.inputPlanes, valueClasses: arch.valueHeadClasses)
        return await onWorkQueue {
            report(states: states, reference: reference, candidate: candidate,
                   valueClasses: arch.valueHeadClasses, computeUnits: computeUnits)
        }
    }

    /// Runs CPU-bound work (encoding, move generation over thousands of
    /// positions) on a GCD queue, off the cooperative pool.
    private static func onWorkQueue<T: Sendable>(_ work: @escaping @Sendable () -> T) async -> T {
        await withCheckedContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async { continuation.resume(returning: work()) }
        }
    }

    static func report(states: [GameState], reference: Outputs, candidate: Outputs, valueClasses: Int,
                       computeUnits: ComputeUnits) -> CoreMLParityReport {
        let size = ChessNetwork.policySize
        var policyMax: Float = 0, policySum: Double = 0, policyCount = 0
        var legalMax: Float = 0, valueMax: Float = 0, probabilitiesMax: Float = 0
        var agreements = 0, nonFinite = 0, nonFiniteReference = 0, withoutLegalMoves = 0
        for (index, state) in states.enumerated() {
            let ref = reference.policy[(index * size)..<((index + 1) * size)]
            let cand = candidate.policy[(index * size)..<((index + 1) * size)]
            let refProbabilities = reference.valueProbabilities[(index * valueClasses)..<((index + 1) * valueClasses)]
            let candProbabilities = candidate.valueProbabilities[(index * valueClasses)..<((index + 1) * valueClasses)]
            guard ref.allSatisfy(\.isFinite), reference.value[index].isFinite,
                  refProbabilities.allSatisfy(\.isFinite) else {
                nonFiniteReference += 1
                continue
            }
            guard cand.allSatisfy(\.isFinite), candidate.value[index].isFinite,
                  candProbabilities.allSatisfy(\.isFinite) else {
                nonFinite += 1
                continue
            }
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
            positions: states.count, computeUnits: computeUnits.rawValue,
            policyMaxAbsDiff: policyMax, policyMeanAbsDiff: policyCount == 0 ? 0 : Float(policySum / Double(policyCount)),
            legalPolicyMaxAbsDiff: legalMax, valueMaxAbsDiff: valueMax, valueProbsMaxAbsDiff: probabilitiesMax,
            topMoveAgreements: agreements, nonFiniteCoreMLPositions: nonFinite,
            nonFiniteReferencePositions: nonFiniteReference, positionsWithoutLegalMoves: withoutLegalMoves)
    }
}
