//
//  NetworkArchitecture.swift
//  DrewsChessMachine
//
//  Single source of truth for a network's *shape* — the runtime, per-model
//  replacement for the compile-time arch constants that historically lived on
//  `ChessNetwork`. This value type is **purely topological**: it owns the
//  decomposed configurable axes (§5a of RUNTIME_ARCHITECTURE_CONFIG_PLAN.md), the
//  derived quantities they feed (`inputPlanes`, `valueHeadClasses`,
//  `parameterCount`, `architectureSummary`), and the ordered `weightTensorPlan`
//  that names + shapes every persistent tensor in the exact order
//  `ChessNetwork.exportWeights()` / `loadWeights()` use.
//
//  Design rules baked in here:
//  - **No silent defaults.** The memberwise init has zero defaulted fields — every
//    configurable parameter is passed explicitly. The only place concrete values
//    live is a `Preset`.
//  - **Flat schema.** Style + its numeric param are separate flat fields
//    (`blockSeStyle` + `blockSeReductionRatio`); `validate()` enforces consistency.
//  - **Naming.** camelCase Swift property ↔ lower_snake_case JSON key (explicit
//    `CodingKeys`); enum rawValues are snake/lowercase tokens. Canonical JSON
//    (sortedKeys, done at the storage boundary) makes field declaration order
//    irrelevant to identity.
//  - **Identity = the value itself** (`Equatable`/`Hashable`). There is no config
//    hash; `arch_hash` survives only as a legacy-`.dcmmodel` lookup (see `Preset`).
//  - **`label` lives OUTSIDE this struct** (in the surrounding model/preset
//    metadata), so the topology stays the sole identity.
//

import Foundation

// MARK: - Component enums (snake_case rawValues = the JSON tokens)

/// The set of input feature planes `BoardEncoder` produces. Single source of truth
/// shared by `BoardEncoder` (writes them), `ChessNetwork` (stem input depth), and
/// `ReplayBuffer` (per-position stride). Adding a case + a `BoardEncoder` branch is
/// the ONLY way to introduce an encoding — the type system then forbids defining a
/// config the encoder can't produce.
enum InputEncoding: String, Codable, CaseIterable, Sendable, Hashable {
    /// 20 planes: pieces / castling / EP / 50-move / 2 repetition-count planes —
    /// the original encoding, with NO temporal-repetition history.
    case basic20
    /// 30 planes: `basic20` (planes 0–19) + 10 temporal-repetition history planes.
    case basic30
    /// 200 planes: the 20-plane `basic20` block stacked 10× for plies N, N-1,
    /// … N-9 — every frame rendered from the ply-N mover's perspective. No
    /// marker planes; absent (pre-game-start) frames stay all-zero. History
    /// frames carry no temporal-repetition block (each is plain basic20).
    case full10ply200

    /// Ordered plane-group spec. `description` renders from this and a unit test
    /// asserts the encoder fills exactly these ranges (no doc/impl drift).
    var planeGroups: [PlaneGroup] {
        let base: [PlaneGroup] = [
            PlaneGroup(0...5,   "my pieces: pawn, knight, bishop, rook, queen, king"),
            PlaneGroup(6...11,  "opponent's pieces: same order"),
            PlaneGroup(12...13, "my castling: kingside, queenside"),
            PlaneGroup(14...15, "opponent's castling: kingside, queenside"),
            PlaneGroup(16...16, "en passant target square"),
            PlaneGroup(17...17, "halfmove / 50-move clock, min(clock,99)/99"),
            PlaneGroup(18...18, "repetition: current position seen >=1x before"),
            PlaneGroup(19...19, "repetition: seen >=2x before (3-fold threshold)"),
        ]
        switch self {
        case .basic20:
            return base
        case .basic30:
            return base + [
                PlaneGroup(20...29, "temporal-repetition history: plane 20+i = position i+1 plies ago is a strict duplicate")
            ]
        case .full10ply200:
            // 10 stacked basic20 frames (current + 9 prior), all from the
            // ply-N mover's perspective. Frame 0 = ply N; frame f = ply N-f.
            var groups: [PlaneGroup] = []
            for f in 0..<historyFrameCount {
                let off = f * planesPerFrame
                let label = f == 0 ? "ply N" : "ply N-\(f)"
                for g in base {
                    groups.append(PlaneGroup(
                        (g.range.lowerBound + off)...(g.range.upperBound + off),
                        "[\(label)] \(g.meaning)"))
                }
            }
            return groups
        }
    }

    /// Number of 8x8 planes — derived from `planeGroups`, never duplicated.
    var planeCount: Int { (planeGroups.last?.range.upperBound ?? -1) + 1 }

    /// Number of stacked position frames (current + history). 1 for single-
    /// frame encodings; 10 for `full10ply200`.
    var historyFrameCount: Int {
        switch self {
        case .basic20, .basic30: return 1
        case .full10ply200: return 10
        }
    }

    /// Planes per stacked frame. History encodings stack the 20-plane basic20
    /// block; single-frame encodings report their whole plane count.
    /// Invariant (asserted in tests): `historyFrameCount * planesPerFrame == planeCount`.
    var planesPerFrame: Int {
        switch self {
        case .basic20: return 20
        case .basic30: return 30
        case .full10ply200: return 20
        }
    }

    /// Human-readable table, rendered from `planeGroups` (single source of
    /// truth). History-stacking encodings repeat one frame's groups many times,
    /// so they get a one-line structural summary instead of the full table.
    var planeDescription: String {
        if historyFrameCount > 1 {
            // History-stacking encoding. Enumerating all `planeCount` groups
            // would repeat one frame's table `historyFrameCount`× (an 80+ line
            // wall), so instead list the per-frame plane ranges (the "ply
            // ranges") and the shared basic20 sub-structure once.
            var lines = ["\(rawValue) — \(planeCount) planes: \(historyFrameCount) stacked "
                + "\(planesPerFrame)-plane basic20 frames, each from the ply-N mover's "
                + "perspective; absent (pre-game) frames are zero."]
            lines.append("  frames (each a \(planesPerFrame)-plane basic20 block):")
            for f in 0..<historyFrameCount {
                let lo = f * planesPerFrame, hi = lo + planesPerFrame - 1
                let ply = f == 0 ? "ply N (current)" : "ply N-\(f)"
                lines.append("    [\(lo)-\(hi)] \(ply)")
            }
            lines.append("  within each frame:")
            for g in InputEncoding.basic20.planeGroups {
                let lo = g.range.lowerBound, hi = g.range.upperBound
                let label = lo == hi ? "\(lo)" : "\(lo)-\(hi)"
                lines.append("    [\(label)] \(g.meaning)")
            }
            return lines.joined(separator: "\n")
        }
        var lines = ["\(rawValue) — \(planeCount) planes:"]
        for g in planeGroups {
            let lo = g.range.lowerBound, hi = g.range.upperBound
            let label = lo == hi ? "\(lo)" : "\(lo)-\(hi)"
            lines.append("  [\(label)] \(g.meaning)")
        }
        return lines.joined(separator: "\n")
    }
}

/// One contiguous range of input planes + what it means. Drives both the rendered
/// description and the encoder-correctness test.
struct PlaneGroup: Sendable, Hashable {
    let range: ClosedRange<Int>
    let meaning: String
    init(_ range: ClosedRange<Int>, _ meaning: String) {
        self.range = range
        self.meaning = meaning
    }
}

/// The single network-wide hidden-activation function (block main path, SE FC1,
/// tower-end, both heads). Verified across all of git history: every architecture
/// used ReLU at every hidden site, so `.relu` reproduces all historical nets. The
/// SE gate (`sigmoid`) and the value output (`tanh`/`softmax`) are structural and
/// NOT governed by this.
enum ActivationFunction: String, Codable, CaseIterable, Sendable, Hashable {
    case relu
    case silu
    case gelu
}

/// Residual-block activation placement. Bundles the correlated choices: `pre` =
/// pre-activation (BN→act→conv…), stem ReLU OFF, tower-end BN ON; `post` =
/// post-activation (conv→BN→act…), stem ReLU ON, tower-end BN OFF.
enum BlockActivationStyle: String, Codable, CaseIterable, Sendable, Hashable {
    case pre
    case post
}

/// How the residual branch merges with the skip.
enum BlockSkipMerge: String, Codable, CaseIterable, Sendable, Hashable {
    /// `out = input + alpha*F(input)` — clean identity highway (v4).
    case cleanAdd = "clean_add"
    /// `out = activation(input + F(input))` — activation-gated sum (v3 was the ReLU case).
    case activationGated = "activation_gated"
}

/// Squeeze-and-Excitation channel-attention variant inside each residual block.
enum SEStyle: String, Codable, CaseIterable, Sendable, Hashable {
    case none
    /// FC2 emits `channels`, applied as `sigmoid(z)*x`.
    case attenuateOnly = "attenuate_only"
    /// FC2 emits `2*channels` (gamma||beta), applied as `sigmoid(gamma)*x + beta`.
    case scaleAndBias = "scale_and_bias"
}

/// Policy-head topology. All three emit 4864 raw logits in the current
/// `PolicyEncoding` (76x64); masking + softmax happen CPU-side downstream.
enum PolicyHeadStyle: String, Codable, CaseIterable, Sendable, Hashable {
    /// Single 1x1 conv channels->76 (+bias) -> reshape. Ignores `policyPreConvChannels`.
    case simpleConv = "simple_conv"
    /// 1x1 conv channels->K -> BN -> act -> 1x1 conv K->76 (+bias) -> reshape.
    case intermediateConv = "intermediate_conv"
    /// 1x1 conv channels->K -> BN -> act -> flatten(K*64) -> FC(K*64->4864) (+bias).
    case fcBottleneck = "fc_bottleneck"

    /// Human-readable summary shown beside the picker in the Build screen,
    /// mirroring `InputEncoding.planeDescription`. Every style emits the same
    /// 4864 raw logits (76 channels × 64 squares); they differ only in how the
    /// tower output is projected down to them. `K` = policy pre-conv channels.
    var styleDescription: String {
        switch self {
        case .simpleConv:
            return "simple_conv — one 1×1 conv (channels → 76) → 4864 logits. "
                + "Fully convolutional, fewest parameters; ignores K."
        case .intermediateConv:
            return "intermediate_conv — 1×1 conv (channels → K) → BN → activation "
                + "→ 1×1 conv (K → 76) → 4864 logits. An added conv layer of width K "
                + "before the projection; still fully convolutional."
        case .fcBottleneck:
            return "fc_bottleneck — 1×1 conv (channels → K) → BN → activation → "
                + "flatten(K×64) → fully-connected (K×64 → 4864). A dense final "
                + "projection — the most parameters (FC = K×64×4864 weights)."
        }
    }
}

/// Value-head topology. Determines output count + activation + the training loss.
enum ValueHeadStyle: String, Codable, CaseIterable, Sendable, Hashable {
    /// 1 logit -> tanh; trained with MSE vs game result z in {-1,0,+1}.
    case scalarTanh = "scalar_tanh"
    /// 3 logits -> softmax (W/D/L); trained with categorical cross-entropy.
    case wdlSoftmax = "wdl_softmax"
}

/// GPU compute precision. NOT a storage property — weights are always Float32 on
/// disk; this selects the MPSGraph compute dtype (and the trainer's fp32-master
/// mixed-precision path when bf16). Honored as configured — no hardware gate
/// (bf16 works everywhere on supported OS, only faster on M5+; see plan §9).
enum ComputeDataType: String, Codable, CaseIterable, Sendable, Hashable {
    case float32
    case bFloat16 = "bfloat16"
}

// MARK: - Weight tensor plan

/// What a persistent tensor *is*, so the safetensors writer can apply the right
/// PyTorch-orientation transform at the export boundary (conv stays OIHW; Linear
/// transposes `[in,out]->[out,in]`; biases reshape to 1-D).
enum WeightKind: String, Sendable, Hashable {
    case conv            // [outC, inC, kH, kW] (OIHW)
    case linear          // [in, out] (MPSGraph matmul orientation)
    case bias            // element count N
    case bnAffine        // BN gamma / beta — [channels]
    case bnRunningStat   // BN running_mean / running_var — [channels]
    case scalar          // ReZero alpha — [1]
}

/// One persistent tensor's identity: PyTorch-ready name, native shape, and kind.
/// The ordered list (`weightTensorPlan`) is the single source of truth shared by
/// builder, analyzer, safetensors writer, and loader.
struct WeightTensorSpec: Sendable, Equatable {
    let name: String
    let shape: [Int]
    let kind: WeightKind
    var elementCount: Int { shape.reduce(1, *) }
}

// MARK: - Errors

enum NetworkArchitectureError: Error, CustomStringConvertible, Equatable {
    case kernelMustBeOdd(field: String, value: Int)
    case nonPositive(field: String, value: Int)
    case channelsNotDivisibleByReduction(channels: Int, reduction: Int)
    case valueConvChannelsExceedChannels(conv: Int, channels: Int)

    var description: String {
        switch self {
        case .kernelMustBeOdd(let field, let value):
            return "\(field) must be odd for symmetric same-padding (got \(value))"
        case .nonPositive(let field, let value):
            return "\(field) must be positive (got \(value))"
        case .channelsNotDivisibleByReduction(let c, let r):
            return "channels (\(c)) must be divisible by blockSeReductionRatio (\(r))"
        case .valueConvChannelsExceedChannels(let conv, let channels):
            return "valueHeadConvChannels (\(conv)) cannot exceed channels (\(channels))"
        }
    }
}

// MARK: - NetworkArchitecture

/// Immutable, purely-topological description of one network's architecture.
/// Construct via the memberwise init (all fields required) or a `Preset`; call
/// `validate()` before building.
struct NetworkArchitecture: Sendable, Codable, Hashable {

    // Input ---------------------------------------------------------------
    var inputEncoding: InputEncoding

    // Tower ---------------------------------------------------------------
    var channels: Int
    var numBlocks: Int
    var stemConvKernelSize: Int
    var activationFunction: ActivationFunction

    // Block ---------------------------------------------------------------
    var blockActivationStyle: BlockActivationStyle
    var blockSkipMerge: BlockSkipMerge
    var blockUseRezero: Bool
    var rezeroAlphaInit: Float           // consumed only when blockUseRezero
    var blockConv1KernelSize: Int
    var blockConv2KernelSize: Int
    var blockSeStyle: SEStyle
    var blockSeReductionRatio: Int       // consumed only when blockSeStyle != none

    // Policy head ---------------------------------------------------------
    var policyHeadStyle: PolicyHeadStyle
    var policyPreConvChannels: Int       // K for intermediate_conv / fc_bottleneck

    // Value head ----------------------------------------------------------
    var valueHeadStyle: ValueHeadStyle
    var valueHeadConvChannels: Int
    var valueHeadHiddenUnits: Int

    // Precision -----------------------------------------------------------
    var computeDataType: ComputeDataType

    // Fixed-by-engine (not stored, not in init) ---------------------------
    static let boardSize = 8
    static let policyChannels = 76
    static var policySize: Int { policyChannels * boardSize * boardSize }   // 4864

    /// All-required memberwise init — NO defaults (no silent fallbacks).
    init(
        inputEncoding: InputEncoding,
        channels: Int,
        numBlocks: Int,
        stemConvKernelSize: Int,
        activationFunction: ActivationFunction,
        blockActivationStyle: BlockActivationStyle,
        blockSkipMerge: BlockSkipMerge,
        blockUseRezero: Bool,
        rezeroAlphaInit: Float,
        blockConv1KernelSize: Int,
        blockConv2KernelSize: Int,
        blockSeStyle: SEStyle,
        blockSeReductionRatio: Int,
        policyHeadStyle: PolicyHeadStyle,
        policyPreConvChannels: Int,
        valueHeadStyle: ValueHeadStyle,
        valueHeadConvChannels: Int,
        valueHeadHiddenUnits: Int,
        computeDataType: ComputeDataType
    ) {
        self.inputEncoding = inputEncoding
        self.channels = channels
        self.numBlocks = numBlocks
        self.stemConvKernelSize = stemConvKernelSize
        self.activationFunction = activationFunction
        self.blockActivationStyle = blockActivationStyle
        self.blockSkipMerge = blockSkipMerge
        self.blockUseRezero = blockUseRezero
        self.rezeroAlphaInit = rezeroAlphaInit
        self.blockConv1KernelSize = blockConv1KernelSize
        self.blockConv2KernelSize = blockConv2KernelSize
        self.blockSeStyle = blockSeStyle
        self.blockSeReductionRatio = blockSeReductionRatio
        self.policyHeadStyle = policyHeadStyle
        self.policyPreConvChannels = policyPreConvChannels
        self.valueHeadStyle = valueHeadStyle
        self.valueHeadConvChannels = valueHeadConvChannels
        self.valueHeadHiddenUnits = valueHeadHiddenUnits
        self.computeDataType = computeDataType
    }

    // MARK: Codable — explicit lower_snake_case keys

    enum CodingKeys: String, CodingKey {
        case inputEncoding = "input_encoding"
        case channels
        case numBlocks = "num_blocks"
        case stemConvKernelSize = "stem_conv_kernel_size"
        case activationFunction = "activation_function"
        case blockActivationStyle = "block_activation_style"
        case blockSkipMerge = "block_skip_merge"
        case blockUseRezero = "block_use_rezero"
        case rezeroAlphaInit = "rezero_alpha_init"
        case blockConv1KernelSize = "block_conv1_kernel_size"
        case blockConv2KernelSize = "block_conv2_kernel_size"
        case blockSeStyle = "block_se_style"
        case blockSeReductionRatio = "block_se_reduction_ratio"
        case policyHeadStyle = "policy_head_style"
        case policyPreConvChannels = "policy_pre_conv_channels"
        case valueHeadStyle = "value_head_style"
        case valueHeadConvChannels = "value_head_conv_channels"
        case valueHeadHiddenUnits = "value_head_hidden_units"
        case computeDataType = "compute_data_type"
    }

    // MARK: Derived shape scalars

    var inputPlanes: Int { inputEncoding.planeCount }
    var boardSize: Int { Self.boardSize }
    var policyChannels: Int { Self.policyChannels }
    var policySize: Int { Self.policySize }
    var valueHeadClasses: Int { valueHeadStyle == .wdlSoftmax ? 3 : 1 }
    /// Tower-end BN exists only for pre-activation (post-act blocks end in an activation).
    var hasTowerEndBN: Bool { blockActivationStyle == .pre }
    /// Stem ReLU exists only for post-activation (pre-act defers it to block 0).
    var hasStemActivation: Bool { blockActivationStyle == .post }
    /// Human v-number for display only (no role in identity / hashing).
    var architectureVersionLabel: Int { blockActivationStyle == .pre ? 4 : 3 }

    // MARK: Validation (structural only — memory budget is a build-time, device-aware check)

    func validate() throws {
        try requireOdd("stemConvKernelSize", stemConvKernelSize)
        try requireOdd("blockConv1KernelSize", blockConv1KernelSize)
        try requireOdd("blockConv2KernelSize", blockConv2KernelSize)
        try requirePositive("channels", channels)
        try requirePositive("numBlocks", numBlocks)
        try requirePositive("policyPreConvChannels", policyPreConvChannels)
        try requirePositive("valueHeadConvChannels", valueHeadConvChannels)
        try requirePositive("valueHeadHiddenUnits", valueHeadHiddenUnits)
        // Unconditional: the block computes channels / ratio regardless of SE style.
        try requirePositive("blockSeReductionRatio", blockSeReductionRatio)
        if blockSeStyle != .none {
            guard channels % blockSeReductionRatio == 0 else {
                throw NetworkArchitectureError.channelsNotDivisibleByReduction(
                    channels: channels, reduction: blockSeReductionRatio)
            }
        }
        guard valueHeadConvChannels <= channels else {
            throw NetworkArchitectureError.valueConvChannelsExceedChannels(
                conv: valueHeadConvChannels, channels: channels)
        }
    }

    private func requireOdd(_ field: String, _ v: Int) throws {
        guard v > 0 else { throw NetworkArchitectureError.nonPositive(field: field, value: v) }
        guard v % 2 == 1 else { throw NetworkArchitectureError.kernelMustBeOdd(field: field, value: v) }
    }
    private func requirePositive(_ field: String, _ v: Int) throws {
        guard v > 0 else { throw NetworkArchitectureError.nonPositive(field: field, value: v) }
    }

    // MARK: Parameter count (verified against all four documented presets)

    /// Total persistent-tensor element count (trainable weights + BN running
    /// mean/var). Equals the summed element counts of `weightTensorPlan` (asserted
    /// in tests). Branches on every axis that changes a shape.
    var parameterCount: Int {
        let c = channels
        let seReduced = blockSeStyle == .none ? 0 : c / blockSeReductionRatio

        // Stem: conv (bias-free) + BN.
        let stem = (inputPlanes * c * stemConvKernelSize * stemConvKernelSize) + 4 * c

        // Per block: two convs (bias-free) + two BNs + SE + optional rezero.
        let conv1 = c * c * blockConv1KernelSize * blockConv1KernelSize
        let conv2 = c * c * blockConv2KernelSize * blockConv2KernelSize
        let bns = 2 * (4 * c)
        let se: Int
        switch blockSeStyle {
        case .none:         se = 0
        case .attenuateOnly: se = (c * seReduced + seReduced) + (seReduced * c + c)
        case .scaleAndBias:  se = (c * seReduced + seReduced) + (seReduced * 2 * c + 2 * c)
        }
        let rezero = blockUseRezero ? 1 : 0
        let perBlock = conv1 + conv2 + bns + se + rezero

        let towerEndBN = hasTowerEndBN ? 4 * c : 0

        // Policy head.
        let pK = policyPreConvChannels
        let policy: Int
        switch policyHeadStyle {
        case .simpleConv:
            policy = (c * policyChannels) + policyChannels
        case .intermediateConv:
            policy = (c * pK) + 4 * pK + (pK * policyChannels) + policyChannels
        case .fcBottleneck:
            let flat = pK * boardSize * boardSize
            policy = (c * pK) + 4 * pK + (flat * policySize) + policySize
        }

        // Value head.
        let cv = valueHeadConvChannels
        let h = valueHeadHiddenUnits
        let flatV = boardSize * boardSize * cv
        let value = (c * cv) + 4 * cv + (flatV * h + h) + (h * valueHeadClasses + valueHeadClasses)

        return stem + numBlocks * perBlock + towerEndBN + policy + value
    }

    // MARK: Summary (human-readable, computed from the config)

    /// Compact one-glance label for the title bar, e.g. "v3 · 8-block 3×3 · 128ch · 2,483,667 params".
    var shortLabel: String {
        let k1 = blockConv1KernelSize, k2 = blockConv2KernelSize
        let kDesc = k1 == k2 ? "\(k1)×\(k1)" : "\(k1)×\(k1),\(k2)×\(k2)"
        return "v\(architectureVersionLabel) · \(numBlocks)-block \(kDesc) · \(channels)ch · \(parameterCount.formatted(.number)) params"
    }

    var architectureSummary: String {
        let seDesc: String
        switch blockSeStyle {
        case .none: seDesc = "no-SE"
        case .attenuateOnly: seDesc = "SE/\(blockSeReductionRatio)"
        case .scaleAndBias: seDesc = "SE+/\(blockSeReductionRatio)"
        }
        let k1 = blockConv1KernelSize, k2 = blockConv2KernelSize
        let kDesc = k1 == k2 ? "\(k1)x\(k1)" : "\(k1)x\(k1),\(k2)x\(k2)"
        let rezeroDesc = blockUseRezero ? "ReZero" : "no-ReZero"
        let valueDesc = valueHeadStyle == .wdlSoftmax
            ? "WDL(\(valueHeadConvChannels)->FC\(valueHeadHiddenUnits))"
            : "tanh(\(valueHeadConvChannels)->FC\(valueHeadHiddenUnits))"
        return "v\(architectureVersionLabel) \(blockActivationStyle.rawValue)"
            + " . in \(inputEncoding.rawValue)(\(inputPlanes)) -> stem \(channels) (\(stemConvKernelSize)x\(stemConvKernelSize))"
            + " . \(numBlocks)x[\(kDesc) conv, \(seDesc), \(blockSkipMerge.rawValue), \(rezeroDesc)]"
            + " . act \(activationFunction.rawValue)"
            + " . policy \(policyHeadStyle.rawValue)(\(policySize))"
            + " . value \(valueDesc)"
            + " . \(computeDataType.rawValue) . \(parameterCount.formatted(.number)) params"
    }

    // MARK: Weight tensor plan

    /// Ordered (name, shape, kind) for every persistent tensor in the exact order
    /// `ChessNetwork.exportWeights()` emits and `loadWeights()` expects: **all
    /// trainables in build order, then all BN running stats in build order**. Names
    /// are PyTorch-ready module paths. Branches on every topology axis.
    func weightTensorPlan() -> [WeightTensorSpec] {
        let c = channels
        let seReduced = blockSeStyle == .none ? 0 : c / blockSeReductionRatio

        var trainables: [WeightTensorSpec] = []
        var running: [WeightTensorSpec] = []

        func bn(_ prefix: String, _ ch: Int) {
            trainables.append(.init(name: "\(prefix).weight", shape: [ch], kind: .bnAffine))
            trainables.append(.init(name: "\(prefix).bias", shape: [ch], kind: .bnAffine))
            running.append(.init(name: "\(prefix).running_mean", shape: [ch], kind: .bnRunningStat))
            running.append(.init(name: "\(prefix).running_var", shape: [ch], kind: .bnRunningStat))
        }
        func se(_ prefix: String) {
            switch blockSeStyle {
            case .none:
                break
            case .attenuateOnly:
                trainables.append(.init(name: "\(prefix).se_attenuate.fc1.weight", shape: [c, seReduced], kind: .linear))
                trainables.append(.init(name: "\(prefix).se_attenuate.fc1.bias", shape: [seReduced], kind: .bias))
                trainables.append(.init(name: "\(prefix).se_attenuate.fc2.weight", shape: [seReduced, c], kind: .linear))
                trainables.append(.init(name: "\(prefix).se_attenuate.fc2.bias", shape: [c], kind: .bias))
            case .scaleAndBias:
                trainables.append(.init(name: "\(prefix).se_scalebias.fc1.weight", shape: [c, seReduced], kind: .linear))
                trainables.append(.init(name: "\(prefix).se_scalebias.fc1.bias", shape: [seReduced], kind: .bias))
                trainables.append(.init(name: "\(prefix).se_scalebias.fc2.weight", shape: [seReduced, 2 * c], kind: .linear))
                trainables.append(.init(name: "\(prefix).se_scalebias.fc2.bias", shape: [2 * c], kind: .bias))
            }
        }

        // Stem: conv -> BN.
        trainables.append(.init(name: "stem.conv.weight", shape: [c, inputPlanes, stemConvKernelSize, stemConvKernelSize], kind: .conv))
        bn("stem.bn", c)

        // Tower.
        for i in 0..<numBlocks {
            let p = "blocks.\(i)"
            let conv1 = WeightTensorSpec(name: "\(p).conv1.weight", shape: [c, c, blockConv1KernelSize, blockConv1KernelSize], kind: .conv)
            let conv2 = WeightTensorSpec(name: "\(p).conv2.weight", shape: [c, c, blockConv2KernelSize, blockConv2KernelSize], kind: .conv)
            switch blockActivationStyle {
            case .pre:
                // BN1 -> act -> conv1 -> BN2 -> act -> conv2 -> SE -> [rezero]
                bn("\(p).bn1", c)
                trainables.append(conv1)
                bn("\(p).bn2", c)
                trainables.append(conv2)
                se(p)
            case .post:
                // conv1 -> BN1 -> act -> conv2 -> BN2 -> SE -> (act on merged sum)
                trainables.append(conv1)
                bn("\(p).bn1", c)
                trainables.append(conv2)
                bn("\(p).bn2", c)
                se(p)
            }
            if blockUseRezero {
                trainables.append(.init(name: "\(p).rezero_alpha", shape: [1], kind: .scalar))
            }
        }

        // Tower-end normalization (pre-activation only).
        if hasTowerEndBN { bn("tower_final_bn", c) }

        // Policy head.
        let pK = policyPreConvChannels
        switch policyHeadStyle {
        case .simpleConv:
            trainables.append(.init(name: "policy.conv.weight", shape: [policyChannels, c, 1, 1], kind: .conv))
            trainables.append(.init(name: "policy.conv.bias", shape: [1, policyChannels, 1, 1], kind: .bias))
        case .intermediateConv:
            trainables.append(.init(name: "policy.pre_conv.weight", shape: [pK, c, 1, 1], kind: .conv))
            bn("policy.pre_bn", pK)
            trainables.append(.init(name: "policy.conv.weight", shape: [policyChannels, pK, 1, 1], kind: .conv))
            trainables.append(.init(name: "policy.conv.bias", shape: [1, policyChannels, 1, 1], kind: .bias))
        case .fcBottleneck:
            trainables.append(.init(name: "policy.pre_conv.weight", shape: [pK, c, 1, 1], kind: .conv))
            bn("policy.pre_bn", pK)
            let flat = pK * boardSize * boardSize
            trainables.append(.init(name: "policy.fc.weight", shape: [flat, policySize], kind: .linear))
            trainables.append(.init(name: "policy.fc.bias", shape: [1, policySize], kind: .bias))
        }

        // Value head.
        let cv = valueHeadConvChannels
        let h = valueHeadHiddenUnits
        trainables.append(.init(name: "value.conv.weight", shape: [cv, c, 1, 1], kind: .conv))
        bn("value.bn", cv)
        let flatV = boardSize * boardSize * cv
        trainables.append(.init(name: "value.fc1.weight", shape: [flatV, h], kind: .linear))
        trainables.append(.init(name: "value.fc1.bias", shape: [1, h], kind: .bias))
        let fc2Name = valueHeadStyle == .wdlSoftmax ? "value.wdl_fc2" : "value.scalar_fc2"
        trainables.append(.init(name: "\(fc2Name).weight", shape: [h, valueHeadClasses], kind: .linear))
        trainables.append(.init(name: "\(fc2Name).bias", shape: [1, valueHeadClasses], kind: .bias))

        return trainables + running
    }
}

// MARK: - Presets

extension NetworkArchitecture {
    /// Named built-in architectures. Compiled-in and immutable (never written to the
    /// Presets folder — that's user-saved only). The historical presets are also the
    /// targets of the legacy `.dcmmodel` hash table (`legacyDcmmodelArchHashes`).
    enum Preset: String, Sendable, CaseIterable {
        case v3_8block_3x3        // 0x13ba0b55, 2,483,667 params (Ko63 / IWkd / sMe9)
        case v3_16block_3x3       // 0x5347c53d, 4,934,867 params
        case v4_12block_3x3       // 0xbad32ced, 3,898,139 params (WcRm)
        case v4_5block_7x7        // 0xdf23a86c, 8,445,748 params (current)
        case v4_8block_3x3        // 2,664,087 params (proposed re-run)

        static let current = Preset.v4_5block_7x7
    }

    static func preset(_ p: Preset) -> NetworkArchitecture {
        switch p {
        case .v3_8block_3x3:
            return NetworkArchitecture(
                inputEncoding: .basic30, channels: 128, numBlocks: 8, stemConvKernelSize: 3,
                activationFunction: .relu, blockActivationStyle: .post,
                blockSkipMerge: .activationGated, blockUseRezero: false, rezeroAlphaInit: 1,
                blockConv1KernelSize: 3, blockConv2KernelSize: 3,
                blockSeStyle: .attenuateOnly, blockSeReductionRatio: 4,
                policyHeadStyle: .simpleConv, policyPreConvChannels: 128,
                valueHeadStyle: .wdlSoftmax, valueHeadConvChannels: 1, valueHeadHiddenUnits: 64,
                computeDataType: .float32
            )
        case .v3_16block_3x3:
            return NetworkArchitecture(
                inputEncoding: .basic30, channels: 128, numBlocks: 16, stemConvKernelSize: 3,
                activationFunction: .relu, blockActivationStyle: .post,
                blockSkipMerge: .activationGated, blockUseRezero: false, rezeroAlphaInit: 1,
                blockConv1KernelSize: 3, blockConv2KernelSize: 3,
                blockSeStyle: .attenuateOnly, blockSeReductionRatio: 4,
                policyHeadStyle: .intermediateConv, policyPreConvChannels: 128,
                valueHeadStyle: .wdlSoftmax, valueHeadConvChannels: 1, valueHeadHiddenUnits: 64,
                computeDataType: .float32
            )
        case .v4_12block_3x3:
            return NetworkArchitecture(
                inputEncoding: .basic30, channels: 128, numBlocks: 12, stemConvKernelSize: 3,
                activationFunction: .relu, blockActivationStyle: .pre,
                blockSkipMerge: .cleanAdd, blockUseRezero: true,
                rezeroAlphaInit: 1.0 / Float(12).squareRoot(),
                blockConv1KernelSize: 3, blockConv2KernelSize: 3,
                blockSeStyle: .scaleAndBias, blockSeReductionRatio: 4,
                policyHeadStyle: .intermediateConv, policyPreConvChannels: 128,
                valueHeadStyle: .wdlSoftmax, valueHeadConvChannels: 16, valueHeadHiddenUnits: 128,
                computeDataType: .bFloat16
            )
        case .v4_5block_7x7:
            return NetworkArchitecture(
                inputEncoding: .basic30, channels: 128, numBlocks: 5, stemConvKernelSize: 7,
                activationFunction: .relu, blockActivationStyle: .pre,
                blockSkipMerge: .cleanAdd, blockUseRezero: true,
                rezeroAlphaInit: 1.0 / Float(5).squareRoot(),
                blockConv1KernelSize: 7, blockConv2KernelSize: 7,
                blockSeStyle: .scaleAndBias, blockSeReductionRatio: 4,
                policyHeadStyle: .intermediateConv, policyPreConvChannels: 128,
                valueHeadStyle: .wdlSoftmax, valueHeadConvChannels: 16, valueHeadHiddenUnits: 128,
                computeDataType: .bFloat16
            )
        case .v4_8block_3x3:
            return NetworkArchitecture(
                inputEncoding: .basic30, channels: 128, numBlocks: 8, stemConvKernelSize: 3,
                activationFunction: .relu, blockActivationStyle: .pre,
                blockSkipMerge: .cleanAdd, blockUseRezero: true,
                rezeroAlphaInit: 1.0 / Float(8).squareRoot(),
                blockConv1KernelSize: 3, blockConv2KernelSize: 3,
                blockSeStyle: .scaleAndBias, blockSeReductionRatio: 4,
                policyHeadStyle: .intermediateConv, policyPreConvChannels: 128,
                valueHeadStyle: .wdlSoftmax, valueHeadConvChannels: 16, valueHeadHiddenUnits: 128,
                computeDataType: .bFloat16
            )
        }
    }

    /// The architecture the current build defaults to.
    static var current: NetworkArchitecture { preset(.current) }

    /// Legacy `.dcmmodel` archHash (the old 7-scalar FNV value stored at byte
    /// offset 12) -> the historical preset to rebuild. The ONLY backward-compat
    /// shim; used by the legacy reader (Phase F). Bidirectional via `legacyArchHash(for:)`.
    static let legacyDcmmodelArchHashes: [UInt32: Preset] = [
        0x13ba_0b55: .v3_8block_3x3,
        0x5347_c53d: .v3_16block_3x3,
        0xbad3_2ced: .v4_12block_3x3,
        0xdf23_a86c: .v4_5block_7x7,
    ]

    /// Reverse lookup: the legacy stored hash for a preset, if it has one.
    static func legacyArchHash(for preset: Preset) -> UInt32? {
        legacyDcmmodelArchHashes.first(where: { $0.value == preset })?.key
    }
}
