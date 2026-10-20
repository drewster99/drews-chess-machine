import Foundation
import MetalPerformanceShadersGraph

/// The MPSGraph calls `ChessNetwork`'s builders make for the forward pass, each
/// built into `graph` exactly as a direct call would build it and, when a
/// `recorder` is attached, also recorded as an `InferenceGraph` operation.
///
/// Every method has the signature of the MPSGraph method it wraps and returns
/// that method's tensor, so the graph a builder produces is the same with or
/// without a recorder (`InferenceGraphRecorderTests` checks the outputs are
/// bit-identical). Production networks pass a nil recorder: one optional check
/// per op at build time, nothing at run time.
///
/// The inference forward pass must go through here; nothing else needs to.
/// Ops this type does not wrap (random draws, comparisons, `minimum`,
/// `assign`) are built on `graph` directly, and only the training-only parts
/// of a training-mode graph use them: dropout masks, BN running-stat EMA
/// assigns, the dropout RNG state. The training-only arithmetic around them
/// goes through here or not, indifferently — a recorder is refused on a
/// training-mode build (`ChessNetwork.init`), so none of it can reach a
/// recording. The weight-load placeholders and assigns and the analysis-tap
/// readback casts, built after the forward pass, use the raw graph too.
///
/// A forward op built with `graph` directly instead of through here is caught
/// when the recording is finished: the next recorded op that reads its output
/// names a tensor no recorded op produced, and `finish` throws naming it.
struct ForwardGraphOps {
    let graph: MPSGraph
    let recorder: InferenceGraphRecorder?

    func placeholder(shape: [NSNumber]?, dataType: MPSDataType, name: String?) -> MPSGraphTensor {
        let t = graph.placeholder(shape: shape, dataType: dataType, name: name)
        recorder?.recordInput(t)
        return t
    }

    func variable(with data: Data, shape: [NSNumber], dataType: MPSDataType, name: String?) -> MPSGraphTensor {
        let t = graph.variable(with: data, shape: shape, dataType: dataType, name: name)
        recorder?.recordWeight(t)
        return t
    }

    func constant(_ scalar: Double, dataType: MPSDataType) -> MPSGraphTensor {
        let t = graph.constant(scalar, dataType: dataType)
        recorder?.record(t, label: nil, .constant(values: [scalar], shape: InferenceGraphRecorder.shape(of: t)))
        return t
    }

    func constant(_ scalar: Double, shape: [NSNumber], dataType: MPSDataType) -> MPSGraphTensor {
        let t = graph.constant(scalar, shape: shape, dataType: dataType)
        let count = shape.reduce(1) { $0 * $1.intValue }
        recorder?.record(t, label: nil, .constant(values: [Double](repeating: scalar, count: count),
                                                  shape: shape.map(\.intValue)))
        return t
    }

    func constant(_ data: Data, shape: [NSNumber], dataType: MPSDataType) -> MPSGraphTensor {
        let t = graph.constant(data, shape: shape, dataType: dataType)
        if let recorder {
            recorder.record(t, label: nil, .constant(
                values: ChessNetwork.decodeWeightData(data, dataType: dataType).map(Double.init),
                shape: shape.map(\.intValue)))
        }
        return t
    }

    func cast(_ tensor: MPSGraphTensor, to type: MPSDataType, name: String?) -> MPSGraphTensor {
        let t = graph.cast(tensor, to: type, name: name)
        if let recorder { recorder.record(t, label: name, .cast(input: recorder.operand(tensor, consumer: name))) }
        return t
    }

    func convolution2D(_ source: MPSGraphTensor, weights: MPSGraphTensor,
                       descriptor: MPSGraphConvolution2DOpDescriptor, name: String?) -> MPSGraphTensor {
        let t = graph.convolution2D(source, weights: weights, descriptor: descriptor, name: name)
        if let recorder {
            recorder.recordConvolution(t, label: name, source: source, weights: weights, descriptor: descriptor)
        }
        return t
    }

    func normalize(_ tensor: MPSGraphTensor, mean: MPSGraphTensor, variance: MPSGraphTensor,
                   gamma: MPSGraphTensor?, beta: MPSGraphTensor?, epsilon: Float, name: String?) -> MPSGraphTensor {
        let t = graph.normalize(tensor, mean: mean, variance: variance, gamma: gamma, beta: beta,
                                epsilon: epsilon, name: name)
        if let recorder {
            // Every normalization the network builds has both affine terms;
            // a nil one would need its own operation form.
            guard let gamma, let beta else {
                recorder.fail(.unsupported("normalize '\(name ?? "")' without gamma and beta"))
                return t
            }
            recorder.record(t, label: name, .normalize(
                input: recorder.operand(tensor, consumer: name), mean: recorder.operand(mean, consumer: name),
                variance: recorder.operand(variance, consumer: name), gamma: recorder.operand(gamma, consumer: name),
                beta: recorder.operand(beta, consumer: name), epsilon: epsilon))
        }
        return t
    }

    func mean(of tensor: MPSGraphTensor, axes: [NSNumber], name: String?) -> MPSGraphTensor {
        let t = graph.mean(of: tensor, axes: axes, name: name)
        if let recorder {
            recorder.record(t, label: name, .mean(input: recorder.operand(tensor, consumer: name), axes: axes.map(\.intValue)))
        }
        return t
    }

    func variance(of tensor: MPSGraphTensor, axes: [NSNumber], name: String?) -> MPSGraphTensor {
        let t = graph.variance(of: tensor, axes: axes, name: name)
        if let recorder {
            recorder.record(t, label: name, .variance(input: recorder.operand(tensor, consumer: name),
                                                      axes: axes.map(\.intValue)))
        }
        return t
    }

    func reductionSum(with tensor: MPSGraphTensor, axis: Int, name: String?) -> MPSGraphTensor {
        let t = graph.reductionSum(with: tensor, axis: axis, name: name)
        if let recorder {
            recorder.record(t, label: name, .reductionSum(input: recorder.operand(tensor, consumer: name), axis: axis))
        }
        return t
    }

    func reshape(_ tensor: MPSGraphTensor, shape: [NSNumber], name: String?) -> MPSGraphTensor {
        let t = graph.reshape(tensor, shape: shape, name: name)
        if let recorder {
            recorder.record(t, label: name, .reshape(input: recorder.operand(tensor, consumer: name),
                                                     shape: shape.map(\.intValue)))
        }
        return t
    }

    func matrixMultiplication(primary: MPSGraphTensor, secondary: MPSGraphTensor, name: String?) -> MPSGraphTensor {
        let t = graph.matrixMultiplication(primary: primary, secondary: secondary, name: name)
        if let recorder {
            recorder.record(t, label: name, .matrixMultiplication(
                primary: recorder.operand(primary, consumer: name), secondary: recorder.operand(secondary, consumer: name)))
        }
        return t
    }

    func addition(_ primary: MPSGraphTensor, _ secondary: MPSGraphTensor, name: String?) -> MPSGraphTensor {
        let t = graph.addition(primary, secondary, name: name)
        if let recorder {
            recorder.record(t, label: name, .addition(
                primary: recorder.operand(primary, consumer: name), secondary: recorder.operand(secondary, consumer: name)))
        }
        return t
    }

    func subtraction(_ primary: MPSGraphTensor, _ secondary: MPSGraphTensor, name: String?) -> MPSGraphTensor {
        let t = graph.subtraction(primary, secondary, name: name)
        if let recorder {
            recorder.record(t, label: name, .subtraction(
                primary: recorder.operand(primary, consumer: name), secondary: recorder.operand(secondary, consumer: name)))
        }
        return t
    }

    func multiplication(_ primary: MPSGraphTensor, _ secondary: MPSGraphTensor, name: String?) -> MPSGraphTensor {
        let t = graph.multiplication(primary, secondary, name: name)
        if let recorder {
            recorder.record(t, label: name, .multiplication(
                primary: recorder.operand(primary, consumer: name), secondary: recorder.operand(secondary, consumer: name)))
        }
        return t
    }

    func division(_ primary: MPSGraphTensor, _ secondary: MPSGraphTensor, name: String?) -> MPSGraphTensor {
        let t = graph.division(primary, secondary, name: name)
        if let recorder {
            recorder.record(t, label: name, .division(
                primary: recorder.operand(primary, consumer: name), secondary: recorder.operand(secondary, consumer: name)))
        }
        return t
    }

    func tanh(with tensor: MPSGraphTensor, name: String?) -> MPSGraphTensor {
        let t = graph.tanh(with: tensor, name: name)
        if let recorder { recorder.record(t, label: name, .tanh(input: recorder.operand(tensor, consumer: name))) }
        return t
    }

    func sigmoid(with tensor: MPSGraphTensor, name: String?) -> MPSGraphTensor {
        let t = graph.sigmoid(with: tensor, name: name)
        if let recorder { recorder.record(t, label: name, .sigmoid(input: recorder.operand(tensor, consumer: name))) }
        return t
    }

    func erf(with tensor: MPSGraphTensor, name: String?) -> MPSGraphTensor {
        let t = graph.erf(with: tensor, name: name)
        if let recorder { recorder.record(t, label: name, .erf(input: recorder.operand(tensor, consumer: name))) }
        return t
    }

    func reLU(with tensor: MPSGraphTensor, name: String?) -> MPSGraphTensor {
        let t = graph.reLU(with: tensor, name: name)
        if let recorder { recorder.record(t, label: name, .reLU(input: recorder.operand(tensor, consumer: name))) }
        return t
    }

    func leakyReLU(with tensor: MPSGraphTensor, alpha: Double, name: String?) -> MPSGraphTensor {
        let t = graph.leakyReLU(with: tensor, alpha: alpha, name: name)
        if let recorder {
            recorder.record(t, label: name, .leakyReLU(input: recorder.operand(tensor, consumer: name), alpha: alpha))
        }
        return t
    }

    func softMax(with tensor: MPSGraphTensor, axis: Int, name: String?) -> MPSGraphTensor {
        let t = graph.softMax(with: tensor, axis: axis, name: name)
        if let recorder { recorder.record(t, label: name, .softMax(input: recorder.operand(tensor, consumer: name), axis: axis)) }
        return t
    }

    func concatTensors(_ tensors: [MPSGraphTensor], dimension: Int, name: String?) -> MPSGraphTensor {
        let t = graph.concatTensors(tensors, dimension: dimension, name: name)
        if let recorder {
            recorder.record(t, label: name, .concat(inputs: tensors.map { recorder.operand($0, consumer: name) },
                                                    dimension: dimension))
        }
        return t
    }

    func sliceTensor(_ tensor: MPSGraphTensor, dimension: Int, start: Int, length: Int, name: String?) -> MPSGraphTensor {
        let t = graph.sliceTensor(tensor, dimension: dimension, start: start, length: length, name: name)
        if let recorder {
            recorder.record(t, label: name, .slice(input: recorder.operand(tensor, consumer: name),
                                                   dimension: dimension, start: start, length: length))
        }
        return t
    }
}

/// Collects the operations `ForwardGraphOps` records while one `ChessNetwork`
/// is built, then (`finish`) checks the recording is complete and names its
/// weights by the plan.
///
/// Used only on the thread building the graph, during `ChessNetwork.init`, and
/// read once afterwards; never shared (the same contract as
/// `AnalysisTapRecorder`).
///
/// Tensors are tracked by object identity. The recorder keeps a strong
/// reference to every tensor it has seen, so no identity can be reused by a
/// later tensor while the recording is open.
final class InferenceGraphRecorder {
    enum RecordingError: LocalizedError, Equatable {
        /// A recorded op read a tensor that no recorded op produced and that is
        /// neither the graph input nor a recorded weight — a forward op built
        /// with the raw MPSGraph instead of `ForwardGraphOps`.
        case unrecordedTensor(consumer: String)
        case unsupported(String)
        case noInput
        case multipleInputs
        /// A recorded weight variable that is not one of the network's plan
        /// tensors, or a plan tensor that was never recorded.
        case weightNotInPlan(variable: String)
        case planTensorNotRecorded(name: String)
        /// A plan tensor the forward pass never reads.
        case weightUnused(name: String)
        case outputNotRecorded(InferenceGraphOutputName)

        var errorDescription: String? {
            switch self {
            case .unrecordedTensor(let consumer):
                return "inference graph recording: '\(consumer)' reads a tensor no recorded operation produced; "
                    + "a forward-pass op was built with the MPSGraph directly instead of through ForwardGraphOps"
            case .unsupported(let detail):
                return "inference graph recording: unsupported \(detail)"
            case .noInput:
                return "inference graph recording: no graph input was recorded"
            case .multipleInputs:
                return "inference graph recording: more than one graph input was recorded"
            case .weightNotInPlan(let variable):
                return "inference graph recording: weight variable '\(variable)' is not a plan tensor"
            case .planTensorNotRecorded(let name):
                return "inference graph recording: plan tensor '\(name)' was not recorded"
            case .weightUnused(let name):
                return "inference graph recording: plan tensor '\(name)' is never read by the forward pass"
            case .outputNotRecorded(let output):
                return "inference graph recording: output '\(output.rawValue)' was not produced by a recorded operation"
            }
        }
    }

    private enum TensorRole {
        case input
        case weight
        case operation
    }

    private var idsByTensor: [ObjectIdentifier: (id: String, role: TensorRole)] = [:]
    private var retained: [MPSGraphTensor] = []
    private var inputTensor: MPSGraphTensor?
    private var inputCount = 0
    private var weightTensors: [(id: String, tensor: MPSGraphTensor)] = []
    private var operations: [GraphOperation] = []
    private var failure: RecordingError?

    /// The id of the board input in every recording.
    static let inputID = "board"

    init() {}

    /// A tensor's static shape, `[]` when MPSGraph gives none (a scalar constant).
    static func shape(of tensor: MPSGraphTensor) -> [Int] {
        tensor.shape?.map(\.intValue) ?? []
    }

    /// Keeps the first failure; `finish` throws it. Recording methods are called
    /// from non-throwing builder code, so a failure is held until then.
    func fail(_ error: RecordingError) {
        if failure == nil { failure = error }
    }

    func recordInput(_ tensor: MPSGraphTensor) {
        inputCount += 1
        inputTensor = tensor
        register(tensor, id: Self.inputID, role: .input)
    }

    func recordWeight(_ tensor: MPSGraphTensor) {
        let id = "w\(weightTensors.count)"
        weightTensors.append((id: id, tensor: tensor))
        register(tensor, id: id, role: .weight)
    }

    /// The id of `tensor` as an operand of the op `consumer` builds; records a
    /// failure (and returns a placeholder id) when it was never recorded.
    func operand(_ tensor: MPSGraphTensor, consumer: String?) -> String {
        if let known = idsByTensor[ObjectIdentifier(tensor)] { return known.id }
        fail(.unrecordedTensor(consumer: consumer ?? "an unnamed operation"))
        return "unrecorded"
    }

    func record(_ tensor: MPSGraphTensor, label: String?, _ kind: GraphOperationKind) {
        guard let dataType = GraphDataType(tensor.dataType) else {
            fail(.unsupported("data type \(tensor.dataType.rawValue) at '\(label ?? "")'"))
            return
        }
        let id = "t\(operations.count)"
        register(tensor, id: id, role: .operation)
        operations.append(GraphOperation(id: id, label: label, dataType: dataType, kind: kind))
    }

    func recordConvolution(_ tensor: MPSGraphTensor, label: String?, source: MPSGraphTensor,
                           weights: MPSGraphTensor, descriptor d: MPSGraphConvolution2DOpDescriptor) {
        guard d.dataLayout == .NCHW, d.weightsLayout == .OIHW, d.paddingStyle == .explicit else {
            fail(.unsupported("convolution layout or padding style at '\(label ?? "")'; recordings support NCHW / OIHW / explicit"))
            return
        }
        record(tensor, label: label, .convolution2D(
            input: operand(source, consumer: label), weights: operand(weights, consumer: label),
            strideY: d.strideInY, strideX: d.strideInX, dilationY: d.dilationRateInY, dilationX: d.dilationRateInX,
            groups: d.groups, paddingTop: d.paddingTop, paddingBottom: d.paddingBottom,
            paddingLeft: d.paddingLeft, paddingRight: d.paddingRight))
    }

    private func register(_ tensor: MPSGraphTensor, id: String, role: TensorRole) {
        retained.append(tensor)
        idsByTensor[ObjectIdentifier(tensor)] = (id: id, role: role)
    }

    /// The finished recording, or the reason it is incomplete.
    ///
    /// - Parameters:
    ///   - persistentVariables: the network's trainables then BN running stats —
    ///     the order `weightTensorPlan()` describes, which names each weight.
    ///   - plan: `weightTensorPlan()` of the architecture built.
    ///   - outputs: the tensors the inference targets read.
    func finish(persistentVariables: [MPSGraphTensor], plan: [WeightTensorSpec],
                outputs: [(InferenceGraphOutputName, MPSGraphTensor)]) throws -> InferenceGraph {
        if let failure { throw failure }
        guard let inputTensor else { throw RecordingError.noInput }
        guard inputCount == 1 else { throw RecordingError.multipleInputs }
        guard let inputType = GraphDataType(inputTensor.dataType) else {
            throw RecordingError.unsupported("input data type \(inputTensor.dataType.rawValue)")
        }

        // Name each recorded weight by its plan position; the builder already
        // checked (validateAgainstPlan) that the variables match the plan.
        var planIndexByTensor: [ObjectIdentifier: Int] = [:]
        for (index, variable) in persistentVariables.enumerated() {
            planIndexByTensor[ObjectIdentifier(variable)] = index
        }
        var weightIDByPlanIndex: [Int: String] = [:]
        for weight in weightTensors {
            guard let index = planIndexByTensor[ObjectIdentifier(weight.tensor)], index < plan.count else {
                throw RecordingError.weightNotInPlan(variable: weight.tensor.operation.name)
            }
            weightIDByPlanIndex[index] = weight.id
        }
        var read = Set<String>()
        for operation in operations { read.formUnion(operation.kind.operands) }

        var weights: [GraphWeight] = []
        var offset = 0
        for (index, spec) in plan.enumerated() {
            guard let id = weightIDByPlanIndex[index] else {
                throw RecordingError.planTensorNotRecorded(name: spec.name)
            }
            guard read.contains(id) else { throw RecordingError.weightUnused(name: spec.name) }
            let shape = Self.shape(of: persistentVariables[index])
            weights.append(GraphWeight(id: id, name: spec.name, shape: shape, byteOffset: offset,
                                       elementCount: spec.elementCount))
            offset += spec.elementCount * MemoryLayout<Float>.size
        }

        var graphOutputs: [GraphOutput] = []
        for (name, tensor) in outputs {
            guard let known = idsByTensor[ObjectIdentifier(tensor)], known.role == .operation else {
                throw RecordingError.outputNotRecorded(name)
            }
            graphOutputs.append(GraphOutput(name: name, id: known.id))
        }
        return InferenceGraph(
            formatVersion: InferenceGraph.currentFormatVersion,
            input: GraphInput(id: Self.inputID, shape: Self.shape(of: inputTensor), dataType: inputType),
            weights: weights, operations: operations, outputs: graphOutputs)
    }
}
