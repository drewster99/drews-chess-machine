//
//  WeightInitialization.swift
//  DrewsChessMachine
//
//  Seeded, per-tensor weight initialization (determinism plan, Part B1).
//

import Foundation
import MetalPerformanceShadersGraph

/// How a network's random-init tensors get their first values.
///
/// - `seeded`: every random tensor draws from its own stream, seeded from the
///   model's init seed and the tensor's safetensors name
///   (`WeightInitScheme.tensorSeed`). The same seed gives bit-identical
///   weights on every machine, and two architectures that share a tensor
///   (same name, same shape) get the same values for it.
/// - `overwrittenByLoad`: the network's variables are replaced by
///   `loadWeights` before first use (a resumed trainer, a model loaded from
///   disk, an inference mirror that receives snapshots). Its variables are
///   built zero-filled and no random work is done; the network refuses to
///   evaluate, export or train until a load has happened, so a missing load
///   is an error rather than silent garbage.
///
/// Deterministic constants (BN γ = 1 / β = 0, running mean 0 / var 1, biases,
/// ReZero α, the W/D/L draw prior, zero-β halves) are the same in both modes.
enum WeightInitialization: Sendable, Equatable {
    case seeded(initSeed: UInt64)
    case overwrittenByLoad

    /// A `seeded` initialization whose seed is drawn from the system
    /// generator — "unseeded" in the plan's sense: the seed is chosen for
    /// you, but there is still exactly one seeded code path, and the drawn
    /// seed is readable (`initSeed`) so a caller can log and record it.
    static func drawnSeed() -> WeightInitialization {
        .seeded(initSeed: drawnInitSeed())
    }

    /// An init seed drawn from the system generator.
    static func drawnInitSeed() -> UInt64 {
        RunRandomSeed.systemDrawnSeed()
    }

    /// The init seed, or nil when the weights come from a load.
    var initSeed: UInt64? {
        switch self {
        case let .seeded(initSeed): return initSeed
        case .overwrittenByLoad: return nil
        }
    }
}

/// The published weight-initialization scheme: everything that decides the
/// value of a freshly drawn weight from `(init seed, tensor name)`.
///
/// `dcm-init-1` pins all of: the per-tensor seed derivation
/// (`DCMRandomStreams.childSeed` over the stream name `init/<tensor name>`);
/// the generator (`DCMRandom`); the normal transform
/// (`DCMRandom.nextStandardNormalPair`, i.e. `DCMNormalMath`); the per-role
/// distributions (He-normal `√(2 / fanIn)` and Glorot-normal
/// `√(2 / (fanIn + fanOut))`, fans read from the tensor's own shape); the fill
/// order — row-major in the tensor's on-disk (PyTorch state_dict) layout, pairs
/// filling consecutive elements, a trailing unpaired normal discarded; and the
/// rule that values are drawn in fp32 and narrowed to the storage dtype only
/// afterwards. A published scheme never changes: any change to any of these
/// gets a new identifier.
enum WeightInitScheme {
    static let current = "dcm-init-1"

    /// The two random roles a tensor can have.
    enum Distribution: Sendable, Equatable {
        /// He-normal, for every conv and every FC whose output feeds a
        /// rectifier-family activation (or a linear readout).
        case heNormal
        /// Glorot-normal, for the SE FC2, whose output feeds a sigmoid gate.
        case glorotNormal
    }

    /// The stream name a tensor's initial values are drawn from.
    static func streamName(tensorName: String) -> String {
        "init/" + tensorName
    }

    /// The seed of `tensorName`'s initial values under `initSeed`.
    static func tensorSeed(initSeed: UInt64, tensorName: String) -> UInt64 {
        DCMRandomStreams.childSeed(parent: initSeed, name: streamName(tensorName: tensorName))
    }

    /// `count` standard normals from a generator seeded with `seed`: each
    /// pair fills two consecutive elements; an odd count drops the pair's
    /// second value.
    static func standardNormals(seed: UInt64, count: Int) -> [Float] {
        var generator = DCMRandom(seed: seed)
        var values = [Float](repeating: 0, count: count)
        var index = 0
        while index < count {
            let (first, second) = generator.nextStandardNormalPair()
            values[index] = first
            if index + 1 < count { values[index + 1] = second }
            index += 2
        }
        return values
    }

    /// `(fanIn, fanOut)` of a conv or FC tensor, read from its native shape:
    /// conv OIHW `[out, in, kH, kW]` → `(in·kH·kW, out·kH·kW)`; FC `[in, out]`
    /// → `(in, out)`.
    static func fans(of spec: WeightTensorSpec) throws -> (fanIn: Int, fanOut: Int) {
        switch spec.kind {
        case .conv:
            guard spec.shape.count == 4 else { throw WeightInitError.unexpectedShape(name: spec.name, shape: spec.shape) }
            let area = spec.shape[2] * spec.shape[3]
            return (spec.shape[1] * area, spec.shape[0] * area)
        case .linear:
            guard spec.shape.count == 2 else { throw WeightInitError.unexpectedShape(name: spec.name, shape: spec.shape) }
            return (spec.shape[0], spec.shape[1])
        case .bias, .bnAffine, .bnRunningStat, .scalar:
            throw WeightInitError.notRandomlyInitialized(name: spec.name, kind: spec.kind)
        }
    }

    /// The standard deviation `distribution` gives `spec`, computed in fp32.
    static func standardDeviation(of spec: WeightTensorSpec, distribution: Distribution) throws -> Float {
        let (fanIn, fanOut) = try fans(of: spec)
        switch distribution {
        case .heNormal:
            return (2.0 / Float(fanIn)).squareRoot()
        case .glorotNormal:
            return (2.0 / Float(fanIn + fanOut)).squareRoot()
        }
    }

    /// `spec`'s initial fp32 values under `initSeed`, in the engine's native
    /// layout (the order `ChessNetwork` variables and `loadWeights` use):
    /// drawn row-major in the on-disk layout, scaled, then converted with the
    /// same transform the safetensors loader uses.
    static func nativeValues(initSeed: UInt64, spec: WeightTensorSpec, distribution: Distribution) throws -> [Float] {
        let stored = try storedValues(initSeed: initSeed, spec: spec, distribution: distribution)
        return SafetensorsModelIO.fromTorchLayout(kind: spec.kind, nativeShape: spec.shape, torchData: stored)
    }

    /// `spec`'s initial fp32 values under `initSeed` in the ON-DISK (PyTorch)
    /// layout the scheme draws in — what `--derive-model`, which rewrites
    /// on-disk tensors, writes for a re-draw. `nativeValues` is this, converted.
    static func storedValues(initSeed: UInt64, spec: WeightTensorSpec, distribution: Distribution) throws -> [Float] {
        let std = try standardDeviation(of: spec, distribution: distribution)
        var stored = standardNormals(seed: tensorSeed(initSeed: initSeed, tensorName: spec.name), count: spec.elementCount)
        for index in stored.indices { stored[index] = std * stored[index] }
        return stored
    }

    /// The initial SE FC2 bias of a block of `group`: the γ half (all of it
    /// for `attenuate_only`, the first `C` elements for `scale_and_bias`) at
    /// the group's `seGammaBiasInit`, the β half at zero. `[expand]` in both
    /// layouts. The single source shared by the graph builder and
    /// `--derive-model`.
    static func seFC2BiasValues(group: BlockGroup) throws -> [Float] {
        switch group.seStyle {
        case .none:
            throw WeightInitError.seBiasOnSELessGroup
        case .attenuateOnly:
            return [Float](repeating: group.seGammaBiasInit, count: group.channels)
        case .scaleAndBias:
            return [Float](repeating: group.seGammaBiasInit, count: group.channels)
                + [Float](repeating: 0, count: group.channels)
        }
    }

    /// The identity-like skip projection `[outC, inC, 1, 1]` (OIHW, which is
    /// both the native and the on-disk layout of a conv): 1 from input channel
    /// `i` to output channel `i` for every `i < min(inC, outC)`, 0 elsewhere.
    static func identityLikeProjectionValues(outChannels: Int, inChannels: Int) -> [Float] {
        var values = [Float](repeating: 0, count: outChannels * inChannels)
        for channel in 0..<min(outChannels, inChannels) {
            values[channel * inChannels + channel] = 1
        }
        return values
    }

    /// The native initial values of a block's SE FC2 weight: a Glorot draw
    /// for the whole `[reduced, expand]` matrix, then — for a zero-β group —
    /// its β columns zeroed, so the γ half is identical under either β init.
    /// The single source shared by the graph builder and `--derive-model`.
    static func seFC2NativeValues(initSeed: UInt64, spec: WeightTensorSpec, group: BlockGroup) throws -> [Float] {
        var values = try nativeValues(initSeed: initSeed, spec: spec, distribution: .glorotNormal)
        switch group.seBetaInit {
        case .glorot:
            break
        case .zero:
            guard group.seStyle == .scaleAndBias, spec.shape.count == 2 else {
                throw WeightInitError.zeroBetaOnUnsupportedTensor(name: spec.name)
            }
            for betaRange in SEScaleAndBiasBetaHalf.nativeWeightRanges(reducedChannels: spec.shape[0], channels: group.channels) {
                for index in betaRange { values[index] = 0 }
            }
        }
        return values
    }
}

enum WeightInitError: Error, Equatable, LocalizedError {
    /// The graph builder asked for a tensor the architecture's plan does not
    /// name — the builder and `weightTensorPlan()` have drifted apart.
    case tensorNotInPlan(name: String)
    /// The builder's shape for a tensor differs from the plan's.
    case shapeDisagreesWithPlan(name: String, builder: [Int], plan: [Int])
    /// The same tensor was initialized twice in one build.
    case initializedTwice(name: String)
    /// After the build, these plan tensors were never initialized.
    case planTensorsNotInitialized(names: [String])
    /// A tensor whose kind has no random role.
    case notRandomlyInitialized(name: String, kind: WeightKind)
    /// A conv or FC tensor whose shape has the wrong rank.
    case unexpectedShape(name: String, shape: [Int])
    /// Zero-β applied to a tensor that is not a scale-and-bias SE FC2.
    case zeroBetaOnUnsupportedTensor(name: String)
    /// An SE FC2 bias asked for on a group with no SE block.
    case seBiasOnSELessGroup
    /// An identity-like projection asked for on a tensor that is not a 1×1 conv.
    case identityLikeOnUnsupportedTensor(name: String, shape: [Int])

    var errorDescription: String? {
        switch self {
        case let .tensorNotInPlan(name):
            return "Weight init: the graph builder initialized '\(name)', which the architecture's tensor plan does not name."
        case let .shapeDisagreesWithPlan(name, builder, plan):
            return "Weight init: '\(name)' is built with shape \(builder) but the tensor plan says \(plan)."
        case let .initializedTwice(name):
            return "Weight init: '\(name)' was initialized twice in one network build."
        case let .planTensorsNotInitialized(names):
            return "Weight init: the network build never initialized \(names.joined(separator: ", "))."
        case let .notRandomlyInitialized(name, kind):
            return "Weight init: '\(name)' is a \(kind.rawValue) tensor, which has no random initialization."
        case let .unexpectedShape(name, shape):
            return "Weight init: '\(name)' has shape \(shape), which is not a conv (OIHW) or FC ([in, out]) shape."
        case let .zeroBetaOnUnsupportedTensor(name):
            return "Weight init: zero-β applies only to a scale-and-bias SE FC2, not '\(name)'."
        case .seBiasOnSELessGroup:
            return "Weight init: an SE FC2 bias was requested for a block group with no SE block."
        case let .identityLikeOnUnsupportedTensor(name, shape):
            return "Weight init: an identity-like init applies only to a 1×1 conv, not '\(name)' with shape \(shape)."
        }
    }
}

/// How a conv or FC weight got its first values (a draw, or an init option's
/// constant) — the record of a graph build's per-tensor choice
/// (`TensorInitializer.randomTensorRoles`).
enum RandomTensorRole: String, Sendable, Equatable {
    case heNormal = "he_normal"
    case glorotNormal = "glorot_normal"
    /// A scale-and-bias SE FC2 whose β half is zeroed after the Glorot draw.
    case glorotNormalZeroBeta = "glorot_normal_zero_beta"
    /// A width-transition skip projection under `skip_projection_init:
    /// identity_like` — a constant, not a draw.
    case identityLike = "identity_like"
    /// A head's final layer under `policy_head_final_init` /
    /// `value_head_final_init: zero` — a constant, not a draw.
    case zero

    init(_ distribution: WeightInitScheme.Distribution) {
        switch distribution {
        case .heNormal: self = .heNormal
        case .glorotNormal: self = .glorotNormal
        }
    }

    /// Whether the tensor's values come from the init seed (a draw) rather
    /// than an init option's constant. The analyzers' init reference
    /// (`AnalysisInitReference`) reads it to tell which tensors it knows
    /// exactly without the model's own seed, so a new role must decide here.
    var isDrawnFromInitSeed: Bool {
        switch self {
        case .heNormal, .glorotNormal, .glorotNormalZeroBeta: return true
        case .identityLike, .zero: return false
        }
    }
}

/// Hands the graph builder each random tensor's initial data during one
/// `ChessNetwork` build, by the tensor's plan name, and checks the build
/// against the architecture's tensor plan: every name requested must be in
/// the plan with the same shape, none twice, and at the end every conv and
/// FC weight in the plan must have been requested. That makes the builder's
/// names provably the safetensors names the per-tensor seeds are keyed on.
///
/// Used only on the thread building the network; not shared.
final class TensorInitializer {
    let initialization: WeightInitialization
    private let planByName: [String: WeightTensorSpec]
    private var initializedNames: Set<String> = []
    /// How each conv and FC weight was initialized (its draw, or an init
    /// option's constant), by plan name (`RandomTensorRole`). Every conv and
    /// FC weight of a completed build is here; biases, BN, ReZero α and the
    /// value prior are not. Recorded by `--derive-model --graft-to` for every
    /// conv and FC weight it initializes.
    private(set) var randomTensorRoles: [String: RandomTensorRole] = [:]

    init(initialization: WeightInitialization, architecture: NetworkArchitecture) {
        self.initialization = initialization
        var byName: [String: WeightTensorSpec] = [:]
        for spec in architecture.weightTensorPlan() { byName[spec.name] = spec }
        self.planByName = byName
    }

    /// The plan entry for `name`, checked against the builder's shape and
    /// marked initialized.
    private func claim(_ name: String, nativeShape: [Int]) throws -> WeightTensorSpec {
        guard let spec = planByName[name] else { throw WeightInitError.tensorNotInPlan(name: name) }
        guard spec.shape == nativeShape else {
            throw WeightInitError.shapeDisagreesWithPlan(name: name, builder: nativeShape, plan: spec.shape)
        }
        guard initializedNames.insert(name).inserted else { throw WeightInitError.initializedTwice(name: name) }
        return spec
    }

    /// The initial data of a conv or FC weight, in `dataType`.
    func weightData(_ name: String, nativeShape: [Int], distribution: WeightInitScheme.Distribution,
                    dataType: MPSDataType) throws -> Data {
        let spec = try claim(name, nativeShape: nativeShape)
        randomTensorRoles[name] = RandomTensorRole(distribution)
        switch initialization {
        case let .seeded(initSeed):
            let values = try WeightInitScheme.nativeValues(initSeed: initSeed, spec: spec, distribution: distribution)
            return ChessNetwork.makeWeightData(values, dataType: dataType)
        case .overwrittenByLoad:
            return ChessNetwork.zerosData(count: spec.elementCount, dataType: dataType)
        }
    }

    /// The initial data of a block's SE FC2 weight (Glorot, zero-β aware).
    func seFC2Data(_ name: String, nativeShape: [Int], group: BlockGroup, dataType: MPSDataType) throws -> Data {
        let spec = try claim(name, nativeShape: nativeShape)
        randomTensorRoles[name] = group.seBetaInit == .zero ? .glorotNormalZeroBeta : .glorotNormal
        switch initialization {
        case let .seeded(initSeed):
            let values = try WeightInitScheme.seFC2NativeValues(initSeed: initSeed, spec: spec, group: group)
            return ChessNetwork.makeWeightData(values, dataType: dataType)
        case .overwrittenByLoad:
            return ChessNetwork.zerosData(count: spec.elementCount, dataType: dataType)
        }
    }

    /// The initial data of a width-transition skip projection under the
    /// group's `skipProjectionInit`: a He draw, or the identity-like constant.
    func skipProjectionData(_ name: String, nativeShape: [Int], initialization projectionInit: SkipProjectionInit,
                            dataType: MPSDataType) throws -> Data {
        switch projectionInit {
        case .he:
            return try weightData(name, nativeShape: nativeShape, distribution: .heNormal, dataType: dataType)
        case .identityLike:
            let spec = try claim(name, nativeShape: nativeShape)
            guard spec.kind == .conv, spec.shape.count == 4, spec.shape[2] == 1, spec.shape[3] == 1 else {
                throw WeightInitError.identityLikeOnUnsupportedTensor(name: name, shape: spec.shape)
            }
            randomTensorRoles[name] = .identityLike
            switch initialization {
            case .seeded:
                return ChessNetwork.makeWeightData(
                    WeightInitScheme.identityLikeProjectionValues(outChannels: spec.shape[0], inChannels: spec.shape[1]),
                    dataType: dataType)
            case .overwrittenByLoad:
                return ChessNetwork.zerosData(count: spec.elementCount, dataType: dataType)
            }
        }
    }

    /// The initial data of a head's final projection under `finalInit`: a He
    /// draw, or exact zeros.
    func headFinalData(_ name: String, nativeShape: [Int], initialization finalInit: HeadFinalInit,
                       dataType: MPSDataType) throws -> Data {
        switch finalInit {
        case .he:
            return try weightData(name, nativeShape: nativeShape, distribution: .heNormal, dataType: dataType)
        case .zero:
            let spec = try claim(name, nativeShape: nativeShape)
            randomTensorRoles[name] = .zero
            return ChessNetwork.zerosData(count: spec.elementCount, dataType: dataType)
        }
    }

    /// Throws unless every conv and FC weight in the plan was initialized.
    func verifyEveryRandomTensorInitialized() throws {
        let expected = planByName.values.filter { $0.kind == .conv || $0.kind == .linear }.map(\.name)
        let missing = expected.filter { !initializedNames.contains($0) }.sorted()
        guard missing.isEmpty else { throw WeightInitError.planTensorsNotInitialized(names: missing) }
    }
}
