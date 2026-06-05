//
//  NetworkArchitecture.swift
//  DrewsChessMachine
//
//  Single source of truth for a network's *shape*. This value type is the
//  runtime, per-model replacement for the compile-time `static let` arch
//  constants that currently live on `ChessNetwork` (channels, numBlocks,
//  tower kernel, SE shape, value-head dims, compute dtype). It owns the
//  derived quantities those constants feed — `parameterCount`, the
//  `archHash`, the human `summary`, and — critically — the ordered
//  `weightTensorPlan` that names and shapes every persistent tensor in the
//  exact order `ChessNetwork.exportWeights()` / `loadWeights()` use.
//
//  Phase 1 (this file): the value type + its derivations + presets, with NO
//  wiring into `ChessNetwork` yet. `ChessNetwork` keeps building from its
//  statics; the parity between this struct's derivations and the live build
//  is asserted by tests (param counts and arch hashes reproduce the
//  documented values exactly) and, in later phases, by the bit-exact
//  forward-pass save verification.
//
//  Scope note: only the v4 (pre-activation) block style is modeled here.
//  The v3 (post-activation) historical style — needed to *load* the 8- and
//  16-block legacy models — is reintroduced in the v3/v4 phase alongside the
//  v3 builder, where its tensor plan and parameter formula can be verified
//  against the real historical weights rather than guessed.
//

import Foundation

// MARK: - Component enums

/// Residual-block topology. Only v4 is currently buildable; `.v3PostActivation`
/// is added in the v3/v4 phase (it changes block ordering, removes ReZero, and
/// uses a different value head, so it can't be modeled accurately until the
/// historical builder is reintroduced).
enum BlockStyle: String, Sendable, Codable, Hashable {
    /// Pre-activation ResNet-v2: `BN→ReLU→conv→BN→ReLU→conv→[SE]`, clean
    /// identity skip with a trainable per-block ReZero scalar
    /// (`out = input + α·F(input)`, α init `1/√numBlocks`).
    case v4PreActivation
}

/// Squeeze-and-Excitation channel-attention variant inside each residual block.
enum SEStyle: String, Sendable, Codable, Hashable {
    /// No SE module.
    case none
    /// Standard SE gate: FC2 emits `channels`, applied as `sigmoid(z)·x`.
    case attenuateOnly
    /// Scale-and-bias SE: FC2 emits `2·channels` (γ‖β), applied as
    /// `sigmoid(γ)·x + β`. The current v4 default; non-standard vs stock SE,
    /// hence the `se_scalebias` flag in the exported tensor names.
    case scaleAndBias
}

/// GPU compute precision. NOT a storage property — weights are always Float32
/// on disk; this only selects the MPSGraph compute dtype (and, when bf16, the
/// trainer's fp32-master mixed-precision update path). Recorded in model
/// metadata as informational ("trained-as").
enum ComputeDataType: String, Sendable, Codable, Hashable {
    case float32
    case bFloat16
}

// MARK: - Weight tensor plan

/// What a persistent tensor *is*, so the safetensors writer can apply the
/// right PyTorch-orientation transform at the export boundary (conv stays
/// OIHW; Linear weights transpose `[in,out]→[out,in]`; biases reshape to 1-D).
/// The native shapes recorded in `WeightTensorSpec.shape` are the engine's own
/// layout; orientation transforms are a writer concern, not a plan concern.
enum WeightKind: String, Sendable, Hashable {
    case conv            // [outC, inC, kH, kW] (OIHW)
    case linear          // [in, out] (MPSGraph matmul orientation)
    case bias            // [1, N, ...] / [1, N] — element count N
    case bnAffine        // BN gamma / beta — [channels]
    case bnRunningStat   // BN running_mean / running_var — [channels]
    case scalar          // ReZero alpha — [1]
}

/// One persistent tensor's identity: its PyTorch-ready name, native shape, and
/// kind. The ordered list of these (`weightTensorPlan`) is the single source of
/// truth shared by the builder, the analyzer, the safetensors writer, and the
/// loader.
struct WeightTensorSpec: Sendable, Equatable {
    let name: String
    let shape: [Int]
    let kind: WeightKind
    var elementCount: Int { shape.reduce(1, *) }
}

// MARK: - Errors

enum NetworkArchitectureError: Error, CustomStringConvertible, Equatable {
    case kernelMustBeOdd(Int)
    case nonPositive(field: String, value: Int)
    case channelsNotDivisibleByReduction(channels: Int, reduction: Int)
    case fixedFieldChanged(field: String, expected: Int, got: Int)

    var description: String {
        switch self {
        case .kernelMustBeOdd(let k):
            return "towerConvKernelSize must be odd for symmetric same-padding (got \(k))"
        case .nonPositive(let field, let value):
            return "\(field) must be positive (got \(value))"
        case .channelsNotDivisibleByReduction(let c, let r):
            return "channels (\(c)) must be divisible by seReductionRatio (\(r))"
        case .fixedFieldChanged(let field, let expected, let got):
            return "\(field) is fixed by the engine at \(expected) (got \(got)); changing it requires new encoders"
        }
    }
}

// MARK: - NetworkArchitecture

/// Immutable description of one network's architecture. Construct via the
/// memberwise init or a `preset`; call `validate()` before building.
struct NetworkArchitecture: Sendable, Codable, Hashable {

    // Variable knobs ------------------------------------------------------
    var channels: Int
    var numBlocks: Int
    var towerConvKernelSize: Int
    var blockStyle: BlockStyle
    var se: SEStyle
    var seReductionRatio: Int
    var valueHeadConvChannels: Int
    var valueHeadHiddenUnits: Int
    var computeDataType: ComputeDataType

    // Fixed-by-engine -----------------------------------------------------
    // Pinned by BoardEncoder / PolicyEncoding / the WDL head. Carried here for
    // hashing + serialization completeness; `validate()` enforces the pins.
    var inputPlanes: Int = 30
    var boardSize: Int = 8
    var policyChannels: Int = 76
    var valueHeadClasses: Int = 3

    static let fixedInputPlanes = 30
    static let fixedBoardSize = 8
    static let fixedPolicyChannels = 76
    static let fixedValueHeadClasses = 3

    init(
        channels: Int,
        numBlocks: Int,
        towerConvKernelSize: Int,
        blockStyle: BlockStyle,
        se: SEStyle,
        seReductionRatio: Int,
        valueHeadConvChannels: Int,
        valueHeadHiddenUnits: Int,
        computeDataType: ComputeDataType
    ) {
        self.channels = channels
        self.numBlocks = numBlocks
        self.towerConvKernelSize = towerConvKernelSize
        self.blockStyle = blockStyle
        self.se = se
        self.seReductionRatio = seReductionRatio
        self.valueHeadConvChannels = valueHeadConvChannels
        self.valueHeadHiddenUnits = valueHeadHiddenUnits
        self.computeDataType = computeDataType
    }

    // Derived shape scalars ----------------------------------------------

    var policySize: Int { policyChannels * boardSize * boardSize }

    /// Topology version, mixed into `archHash`. v4 = pre-activation tower.
    var architectureVersion: Int {
        switch blockStyle {
        case .v4PreActivation: return 4
        }
    }

    // MARK: Validation

    func validate() throws {
        guard towerConvKernelSize % 2 == 1 else {
            throw NetworkArchitectureError.kernelMustBeOdd(towerConvKernelSize)
        }
        guard channels > 0 else {
            throw NetworkArchitectureError.nonPositive(field: "channels", value: channels)
        }
        guard numBlocks > 0 else {
            throw NetworkArchitectureError.nonPositive(field: "numBlocks", value: numBlocks)
        }
        guard towerConvKernelSize > 0 else {
            throw NetworkArchitectureError.nonPositive(field: "towerConvKernelSize", value: towerConvKernelSize)
        }
        guard valueHeadConvChannels > 0 else {
            throw NetworkArchitectureError.nonPositive(field: "valueHeadConvChannels", value: valueHeadConvChannels)
        }
        guard valueHeadHiddenUnits > 0 else {
            throw NetworkArchitectureError.nonPositive(field: "valueHeadHiddenUnits", value: valueHeadHiddenUnits)
        }
        if se != .none {
            guard seReductionRatio > 0 else {
                throw NetworkArchitectureError.nonPositive(field: "seReductionRatio", value: seReductionRatio)
            }
            guard channels % seReductionRatio == 0 else {
                throw NetworkArchitectureError.channelsNotDivisibleByReduction(channels: channels, reduction: seReductionRatio)
            }
        }
        try checkFixed("inputPlanes", inputPlanes, Self.fixedInputPlanes)
        try checkFixed("boardSize", boardSize, Self.fixedBoardSize)
        try checkFixed("policyChannels", policyChannels, Self.fixedPolicyChannels)
        try checkFixed("valueHeadClasses", valueHeadClasses, Self.fixedValueHeadClasses)
    }

    private func checkFixed(_ field: String, _ got: Int, _ expected: Int) throws {
        guard got == expected else {
            throw NetworkArchitectureError.fixedFieldChanged(field: field, expected: expected, got: got)
        }
    }

    // MARK: Parameter count

    /// Total persistent-tensor element count (trainable weights + BN running
    /// mean/var). Mirrors the layer shapes built by `ChessNetwork`; equals the
    /// summed element counts of `weightTensorPlan` (asserted in tests).
    var parameterCount: Int {
        let c = channels
        let seReduced = se == .none ? 0 : c / seReductionRatio
        let convArea = towerConvKernelSize * towerConvKernelSize

        let convPerBlock = 2 * (c * c * convArea)
        let bnPerBlock = 2 * (4 * c)            // bn1 + bn2, each γ/β/mean/var
        let seParams: Int
        switch se {
        case .none:
            seParams = 0
        case .attenuateOnly:
            seParams = (c * seReduced + seReduced) + (seReduced * c + c)
        case .scaleAndBias:
            seParams = (c * seReduced + seReduced) + (seReduced * 2 * c + 2 * c)
        }
        let resScale: Int
        switch blockStyle {
        case .v4PreActivation: resScale = 1
        }
        let perBlock = convPerBlock + bnPerBlock + seParams + resScale

        let stem = (inputPlanes * c * convArea) + (4 * c)
        let towerFinalBN = 4 * c
        let policy = (c * c) + (4 * c) + (c * policyChannels + policyChannels)
        let valueFlatten = boardSize * boardSize * valueHeadConvChannels
        let valueConvBN = (c * valueHeadConvChannels) + (4 * valueHeadConvChannels)
        let valueFC1 = (valueFlatten * valueHeadHiddenUnits) + valueHeadHiddenUnits
        let valueFC2 = (valueHeadHiddenUnits * valueHeadClasses) + valueHeadClasses
        let value = valueConvBN + valueFC1 + valueFC2

        return (numBlocks * perBlock) + stem + towerFinalBN + policy + value
    }

    // MARK: Arch hash

    /// FNV-1a over the shape scalars + topology version. Byte-for-byte the same
    /// formula as the legacy `ModelCheckpointFile.currentArchHash`, so the four
    /// documented historical hashes reproduce exactly. `architectureVersion`
    /// was added to the mix at v4; v3 (when reintroduced) omits it.
    var archHash: UInt32 {
        var h: UInt32 = 0x811C_9DC5 // FNV-1a offset basis
        func mix(_ value: Int) {
            let u32 = UInt32(truncatingIfNeeded: value)
            var v = u32.littleEndian
            withUnsafeBytes(of: &v) { raw in
                for byte in raw {
                    h ^= UInt32(byte)
                    h = h &* 0x0100_0193
                }
            }
        }
        mix(channels)
        mix(numBlocks)
        mix(inputPlanes)
        mix(boardSize)
        mix(policySize)
        mix(valueHeadClasses)
        // Topology version only entered the hash at v4; v3 hashes 6 scalars.
        if architectureVersion >= 4 {
            mix(architectureVersion)
        }
        return h
    }

    /// `archHash` formatted as the project's `0x........` lowercase hex tag.
    var archHashHex: String {
        "0x" + String(format: "%08x", archHash)
    }

    // MARK: Summary

    var architectureSummary: String {
        let k = towerConvKernelSize
        let seDesc: String
        switch se {
        case .none: seDesc = "no-SE"
        case .attenuateOnly: seDesc = "SE÷\(seReductionRatio)"
        case .scaleAndBias: seDesc = "SE±÷\(seReductionRatio)"
        }
        let blockDesc: String
        switch blockStyle {
        case .v4PreActivation: blockDesc = "ReZero"
        }
        return "v\(architectureVersion) · stem \(inputPlanes)→\(channels) (\(k)×\(k))"
            + " · \(numBlocks)×[\(k)×\(k) conv×2, \(seDesc), \(blockDesc)]"
            + " · policy→\(policyChannels) (\(policySize))"
            + " · value→\(valueHeadConvChannels)→FC\(valueHeadHiddenUnits)→WDL(\(valueHeadClasses))"
            + " · \(computeDataType.rawValue) · \(parameterCount.formatted(.number)) params"
    }

    // MARK: Weight tensor plan

    /// Ordered (name, shape, kind) for every persistent tensor, in the exact
    /// order `ChessNetwork.exportWeights()` emits and `loadWeights()` expects:
    /// **all trainables in build order, then all BN running stats in build
    /// order**. Names are PyTorch-ready module paths. This is the single source
    /// of truth the safetensors writer/loader and the analyzer build against.
    func weightTensorPlan() -> [WeightTensorSpec] {
        let c = channels
        let k = towerConvKernelSize
        let seReduced = se == .none ? 0 : c / seReductionRatio

        var trainables: [WeightTensorSpec] = []
        var running: [WeightTensorSpec] = []

        func bn(_ prefix: String, _ ch: Int) {
            trainables.append(.init(name: "\(prefix).weight", shape: [ch], kind: .bnAffine))
            trainables.append(.init(name: "\(prefix).bias", shape: [ch], kind: .bnAffine))
            running.append(.init(name: "\(prefix).running_mean", shape: [ch], kind: .bnRunningStat))
            running.append(.init(name: "\(prefix).running_var", shape: [ch], kind: .bnRunningStat))
        }

        // Stem: conv → BN (no ReLU in v4).
        trainables.append(.init(name: "stem.conv.weight", shape: [c, inputPlanes, k, k], kind: .conv))
        bn("stem.bn", c)

        // Tower: pre-activation residual blocks.
        for i in 0..<numBlocks {
            let p = "blocks.\(i)"
            bn("\(p).bn1", c)
            trainables.append(.init(name: "\(p).conv1.weight", shape: [c, c, k, k], kind: .conv))
            bn("\(p).bn2", c)
            trainables.append(.init(name: "\(p).conv2.weight", shape: [c, c, k, k], kind: .conv))
            switch se {
            case .none:
                break
            case .attenuateOnly:
                trainables.append(.init(name: "\(p).se_attenuate.fc1.weight", shape: [c, seReduced], kind: .linear))
                trainables.append(.init(name: "\(p).se_attenuate.fc1.bias", shape: [seReduced], kind: .bias))
                trainables.append(.init(name: "\(p).se_attenuate.fc2.weight", shape: [seReduced, c], kind: .linear))
                trainables.append(.init(name: "\(p).se_attenuate.fc2.bias", shape: [c], kind: .bias))
            case .scaleAndBias:
                trainables.append(.init(name: "\(p).se_scalebias.fc1.weight", shape: [c, seReduced], kind: .linear))
                trainables.append(.init(name: "\(p).se_scalebias.fc1.bias", shape: [seReduced], kind: .bias))
                trainables.append(.init(name: "\(p).se_scalebias.fc2.weight", shape: [seReduced, 2 * c], kind: .linear))
                trainables.append(.init(name: "\(p).se_scalebias.fc2.bias", shape: [2 * c], kind: .bias))
            }
            switch blockStyle {
            case .v4PreActivation:
                trainables.append(.init(name: "\(p).rezero_alpha", shape: [1], kind: .scalar))
            }
        }

        // Tower-end normalization.
        bn("tower_final_bn", c)

        // Policy head: 1×1 conv → BN → ReLU → 1×1 conv (+bias).
        trainables.append(.init(name: "policy.pre_conv.weight", shape: [c, c, 1, 1], kind: .conv))
        bn("policy.pre_bn", c)
        trainables.append(.init(name: "policy.conv.weight", shape: [policyChannels, c, 1, 1], kind: .conv))
        trainables.append(.init(name: "policy.conv.bias", shape: [1, policyChannels, 1, 1], kind: .bias))

        // Value head: 1×1 conv → BN → ReLU → flatten → FC → ReLU → FC(WDL).
        trainables.append(.init(name: "value.conv.weight", shape: [valueHeadConvChannels, c, 1, 1], kind: .conv))
        bn("value.bn", valueHeadConvChannels)
        let flat = boardSize * boardSize * valueHeadConvChannels
        trainables.append(.init(name: "value.fc1.weight", shape: [flat, valueHeadHiddenUnits], kind: .linear))
        trainables.append(.init(name: "value.fc1.bias", shape: [1, valueHeadHiddenUnits], kind: .bias))
        trainables.append(.init(name: "value.wdl_fc2.weight", shape: [valueHeadHiddenUnits, valueHeadClasses], kind: .linear))
        trainables.append(.init(name: "value.wdl_fc2.bias", shape: [1, valueHeadClasses], kind: .bias))

        return trainables + running
    }
}

// MARK: - Presets

extension NetworkArchitecture {
    /// Named architectures. v4 presets only for now; the v3 historical presets
    /// (8-block, 16-block) arrive with the v3 builder in the v3/v4 phase.
    enum Preset: String, Sendable, CaseIterable {
        case v4_5block_7x7     // current champion architecture
        case v4_12block_3x3    // bf16 3×3 baseline ("Session A")
        case v4_8block_3x3     // the proposed re-run of the 8-block tower in v4

        static let current = Preset.v4_5block_7x7
    }

    static func preset(_ p: Preset) -> NetworkArchitecture {
        switch p {
        case .v4_5block_7x7:
            return NetworkArchitecture(
                channels: 128, numBlocks: 5, towerConvKernelSize: 7,
                blockStyle: .v4PreActivation, se: .scaleAndBias, seReductionRatio: 4,
                valueHeadConvChannels: 16, valueHeadHiddenUnits: 128,
                computeDataType: .bFloat16
            )
        case .v4_12block_3x3:
            return NetworkArchitecture(
                channels: 128, numBlocks: 12, towerConvKernelSize: 3,
                blockStyle: .v4PreActivation, se: .scaleAndBias, seReductionRatio: 4,
                valueHeadConvChannels: 16, valueHeadHiddenUnits: 128,
                computeDataType: .bFloat16
            )
        case .v4_8block_3x3:
            return NetworkArchitecture(
                channels: 128, numBlocks: 8, towerConvKernelSize: 3,
                blockStyle: .v4PreActivation, se: .scaleAndBias, seReductionRatio: 4,
                valueHeadConvChannels: 16, valueHeadHiddenUnits: 128,
                computeDataType: .bFloat16
            )
        }
    }

    /// The architecture the current build's `ChessNetwork` statics describe.
    static var current: NetworkArchitecture { preset(.current) }
}
