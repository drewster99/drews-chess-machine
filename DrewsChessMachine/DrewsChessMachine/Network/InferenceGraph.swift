import Foundation
import MetalPerformanceShadersGraph

/// A network's inference forward pass written out as the primitive operations
/// `ChessNetwork` builds into its MPSGraph, in build order — the graph an
/// outside toolchain (the Core ML exporter, `scripts/dcm_coreml_export.py`)
/// reproduces op for op.
///
/// WHY A RECORDING AND NOT A SECOND BUILDER: the network's topology is defined
/// once, by `ChessNetwork`'s graph builders. Any other description of it — a
/// PyTorch module, a hand-written MIL program — is a second implementation that
/// drifts the first time an architecture option is added on one side only. So
/// the description is captured from the builders themselves: every forward op
/// they emit goes through `ForwardGraphOps`, which builds it into the MPSGraph
/// and, when an `InferenceGraphRecorder` is attached, appends the same op here.
/// The exporter then only translates each primitive to its Core ML equivalent;
/// it holds no knowledge of blocks, heads or architecture options.
///
/// Tensor ids: the graph input, every weight and every operation's output each
/// have one id, unique across the graph. Operands name those ids. Weights are
/// stored separately (`weights.bin`, see `InferenceGraphExport`) in the layout
/// the MPSGraph variables hold — the exporter never needs the safetensors file
/// layout, the value-head recentering, or any other load-time rule.
struct InferenceGraph: Codable, Equatable, Sendable {
    /// Bumped whenever an operation kind, field or meaning changes, so an
    /// exporter written for one version refuses another instead of misreading it.
    static let currentFormatVersion = 2

    let formatVersion: Int
    let input: GraphInput
    let weights: [GraphWeight]
    let operations: [GraphOperation]
    let outputs: [GraphOutput]

    enum CodingKeys: String, CodingKey {
        case formatVersion = "format_version"
        case input, weights, operations, outputs
    }
}

/// The graph's one input: the encoded boards, `[batch, planes, 8, 8]`, with the
/// batch dimension `-1` (any size).
struct GraphInput: Codable, Equatable, Sendable {
    let id: String
    let shape: [Int]
    let dataType: GraphDataType

    enum CodingKeys: String, CodingKey {
        case id, shape
        case dataType = "data_type"
    }
}

/// One persistent tensor the graph reads: its plan name (`weightTensorPlan`),
/// the shape the MPSGraph variable is declared with (BN parameters
/// `[1, C, 1, 1]`, fully connected weights `[in, out]`, conv weights OIHW), the
/// data type the variable holds, and where its values sit in the weights file
/// (float32, already rounded to `dataType` — the values the variable holds).
struct GraphWeight: Codable, Equatable, Sendable {
    let id: String
    let name: String
    let shape: [Int]
    let dataType: GraphDataType
    let byteOffset: Int
    let elementCount: Int

    enum CodingKeys: String, CodingKey {
        case id, name, shape
        case dataType = "data_type"
        case byteOffset = "byte_offset"
        case elementCount = "element_count"
    }
}

/// One operation: the id of the tensor it produces, the MPSGraph op name it was
/// built under (nil where the builder gave none; for reading the graph, not for
/// identity), the data type of its result in the MPSGraph (which is what tells
/// the exporter where the network computes in reduced precision and where in
/// float32), and what it computes.
struct GraphOperation: Codable, Equatable, Sendable {
    let id: String
    let label: String?
    let dataType: GraphDataType
    let kind: GraphOperationKind

    enum CodingKeys: String, CodingKey {
        case id, label, kind
        case dataType = "data_type"
    }
}

/// The tensors an inference consumer reads — the same three
/// `ChessNetwork.inferenceTargets` reads back.
struct GraphOutput: Codable, Equatable, Sendable {
    let name: InferenceGraphOutputName
    let id: String
}

enum InferenceGraphOutputName: String, Codable, Sendable, CaseIterable {
    /// Raw policy logits, `[batch, 4864]`, `PolicyEncoding` order.
    case policy
    /// The derived scalar value `p_win − p_loss` (`scalar_tanh`: `tanh` of the
    /// logit), `[batch, 1]`.
    case value
    /// The W/D/L softmax, `[batch, 3]` (`scalar_tanh`: the scalar), the
    /// distribution `ChessNetwork.valueProbs` exposes.
    case valueProbabilities = "value_probs"
}

/// The element types a forward graph uses. Any other `MPSDataType` reaching a
/// recorded op is a recording failure, not a silent mapping.
enum GraphDataType: String, Codable, Sendable {
    case float32
    case float16
    case bfloat16

    init?(_ mps: MPSDataType) {
        switch mps {
        case .float32: self = .float32
        case .float16: self = .float16
        case .bFloat16: self = .bfloat16
        default: return nil
        }
    }
}

/// The primitive operations `ForwardGraphOps` records — each one MPSGraph call
/// with the arguments that determine its result. Operands are tensor ids.
///
/// Semantics are MPSGraph's: elementwise binary ops broadcast NumPy-style;
/// `mean`, `variance` and `reductionSum` keep the reduced axes as size 1;
/// `variance` is the population variance; `normalize` is
/// `(x − mean) / sqrt(variance + epsilon) · gamma + beta`; `convolution2D` is
/// NCHW input with OIHW weights; `cast` converts to the operation's own
/// `dataType`; `constant` holds `values` in row-major order (`shape` empty for
/// a scalar).
enum GraphOperationKind: Codable, Equatable, Sendable {
    case cast(input: String)
    case convolution2D(input: String, weights: String,
                       strideY: Int, strideX: Int, dilationY: Int, dilationX: Int, groups: Int,
                       paddingTop: Int, paddingBottom: Int, paddingLeft: Int, paddingRight: Int)
    case normalize(input: String, mean: String, variance: String, gamma: String, beta: String, epsilon: Float)
    case mean(input: String, axes: [Int])
    case variance(input: String, axes: [Int])
    case reductionSum(input: String, axis: Int)
    case reshape(input: String, shape: [Int])
    case matrixMultiplication(primary: String, secondary: String)
    case addition(primary: String, secondary: String)
    case subtraction(primary: String, secondary: String)
    case multiplication(primary: String, secondary: String)
    case division(primary: String, secondary: String)
    case tanh(input: String)
    case sigmoid(input: String)
    case erf(input: String)
    case reLU(input: String)
    case leakyReLU(input: String, alpha: Double)
    case softMax(input: String, axis: Int)
    case concat(inputs: [String], dimension: Int)
    case slice(input: String, dimension: Int, start: Int, length: Int)
    case constant(values: [Double], shape: [Int])

    /// Every tensor id this operation reads, in argument order.
    var operands: [String] {
        switch self {
        case .cast(let input), .mean(let input, _), .variance(let input, _), .reductionSum(let input, _),
             .reshape(let input, _), .tanh(let input), .sigmoid(let input), .erf(let input), .reLU(let input),
             .leakyReLU(let input, _), .softMax(let input, _), .slice(let input, _, _, _):
            return [input]
        case .convolution2D(let input, let weights, _, _, _, _, _, _, _, _, _):
            return [input, weights]
        case .normalize(let input, let mean, let variance, let gamma, let beta, _):
            return [input, mean, variance, gamma, beta]
        case .matrixMultiplication(let primary, let secondary), .addition(let primary, let secondary),
             .subtraction(let primary, let secondary), .multiplication(let primary, let secondary),
             .division(let primary, let secondary):
            return [primary, secondary]
        case .concat(let inputs, _):
            return inputs
        case .constant:
            return []
        }
    }
}
