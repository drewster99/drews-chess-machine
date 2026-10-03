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
    /// 24 planes: `basic20` (planes 0–19) + the four temporal-repetition
    /// planes that can fire — the position 4, 6, 8 and 10 plies ago is a
    /// strict duplicate (`possibleRepetitionPlyDistances`), in that order.
    /// It is `basic30` without planes 20, 21, 22, 24, 26 and 28: an odd ply
    /// distance puts the other side to move, and a position cannot recur
    /// after two plies (each side has moved a piece), so those six planes are
    /// zero in every legal game and carry nothing (GitHub issue #10).
    case basic24
    /// 200 planes: the 20-plane `basic20` block stacked 10× for plies N, N-1,
    /// … N-9 — every frame rendered from the ply-N mover's perspective. No
    /// marker planes; absent (pre-game-start) frames stay all-zero. History
    /// frames carry no temporal-repetition block (each is plain basic20).
    case full10ply200
    /// 210 planes: `full10ply200`'s 200 planes + the 10 `basic30` temporal-
    /// repetition planes (planes 20–29 there) appended at 200–209, describing
    /// the CURRENT position only. History frames carry no reps. The appended
    /// block is NOT part of the stacked frames — it is a non-stacked tail
    /// (`tailPlaneCount == 10`), reproduced bit-for-bit from `basic30`'s
    /// `recentRepetitionMask`.
    case full10Ply10Reps210

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
        case .basic24:
            return base + InputEncoding.possibleRepetitionPlyDistances.enumerated().map { offset, distance in
                PlaneGroup((20 + offset)...(20 + offset),
                           "temporal repetition: position \(distance) plies ago is a strict duplicate")
            }
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
        case .full10Ply10Reps210:
            // full10ply200's 200 planes (10 stacked basic20 frames) + the 10
            // basic30 temporal-repetition planes appended at 200–209. Reuses
            // full10ply200's group layout verbatim so the two never drift.
            return InputEncoding.full10ply200.planeGroups + [
                PlaneGroup(200...209, "temporal-repetition history: plane 200+i = position i+1 plies ago is a strict duplicate")
            ]
        }
    }

    /// The ply distances, within the ten-ply temporal-repetition window, at
    /// which the current position can be a strict duplicate of an earlier
    /// one: even (the same side to move) and at least four (after two plies
    /// each side has moved a piece, so the position cannot have recurred).
    /// `basic24` keeps exactly these planes, in this order; the single source
    /// for its layout, its encoder branch and its channel names.
    static let possibleRepetitionPlyDistances = [4, 6, 8, 10]

    /// Number of 8x8 planes — derived from `planeGroups`, never duplicated.
    var planeCount: Int { (planeGroups.last?.range.upperBound ?? -1) + 1 }

    /// Number of stacked position frames (current + history). 1 for single-
    /// frame encodings; 10 for `full10ply200`.
    var historyFrameCount: Int {
        switch self {
        case .basic20, .basic30, .basic24: return 1
        case .full10ply200, .full10Ply10Reps210: return 10
        }
    }

    /// Planes per stacked frame. History encodings stack the 20-plane basic20
    /// block; single-frame encodings report their whole plane count.
    /// Invariant (asserted in tests): `historyFrameCount * planesPerFrame + tailPlaneCount == planeCount`.
    var planesPerFrame: Int {
        switch self {
        case .basic20: return 20
        case .basic30: return 30
        case .basic24: return 24
        case .full10ply200, .full10Ply10Reps210: return 20
        }
    }

    /// Planes appended after the stacked frames that are NOT part of any frame
    /// (e.g. whole-position repetition planes describing only the current ply).
    /// `0` for every encoding whose planes are exactly
    /// `historyFrameCount × planesPerFrame`. The replay buffer stores only the
    /// stacked frames; a non-zero tail is produced by the consumer at sample
    /// time (see `ReplayBuffer.appendRepetitionTail`) and at inference time
    /// from the live `GameState`.
    /// Invariant (asserted in tests): `historyFrameCount × planesPerFrame + tailPlaneCount == planeCount`.
    var tailPlaneCount: Int {
        switch self {
        case .basic20, .basic30, .basic24, .full10ply200: return 0
        case .full10Ply10Reps210: return 10
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
            // Non-stacked tail planes (e.g. appended repetition block), if any.
            // Empty for full10ply200, so its description is unchanged.
            let stackedPlanes = historyFrameCount * planesPerFrame
            let tailGroups = planeGroups.filter { $0.range.lowerBound >= stackedPlanes }
            if !tailGroups.isEmpty {
                lines.append("  appended (not stacked):")
                for g in tailGroups {
                    let lo = g.range.lowerBound, hi = g.range.upperBound
                    let label = lo == hi ? "\(lo)" : "\(lo)-\(hi)"
                    lines.append("    [\(label)] \(g.meaning)")
                }
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

/// Hidden-activation function. Chosen per block group (`BlockGroup.activationFunction`:
/// block main path, `activation_gated` merge; `BlockGroup.seActivation`: SE FC1) and
/// once at the tower level
/// (`NetworkArchitecture.activationFunction`: stem activation, tower-end activation,
/// policy head, value head conv and FC1). Verified across all of git history: every
/// architecture before SiLU/GELU were added used ReLU at every hidden site, so `.relu`
/// reproduces all historical nets. The SE gate (`sigmoid`) and the value output
/// (`tanh` for `scalar_tanh`, `softmax` for `wdl_softmax`) are structural and NOT
/// governed by this.
enum ActivationFunction: String, Codable, CaseIterable, Sendable, Hashable {
    case relu
    case silu
    case gelu
    /// `x` for `x ≥ 0`, `leakyReLUNegativeSlope · x` below. Keeps a small
    /// gradient where ReLU's is exactly zero, so a unit pushed negative for
    /// every input (a dead ReLU unit, as measured in the SE bottlenecks) can
    /// still recover.
    case leakyRelu = "leaky_relu"

    /// The negative-side slope of `leaky_relu`: a fixed constant, not an
    /// architecture field, so `leaky_relu` means the same function in every
    /// model. 0.01 is the conventional value (PyTorch's default).
    static let leakyReLUNegativeSlope: Double = 0.01
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

/// Optional normalization applied to the block's *final output* — after the
/// skip merge, just before the block returns. Orthogonal to `BlockSkipMerge`:
/// it composes with either merge mode. Its purpose is to re-center the residual
/// stream every block so the un-recentered clean-add highway (v4) cannot
/// accumulate a drifting mean. LayerNorm is chosen over BatchNorm here precisely
/// because it has **no train/eval statistics gap** — it recomputes its stats
/// per-forward, identically at train and inference — so it kills the
/// running-stat drift that degraded v4 inference without reintroducing the same
/// failure mode. (`v5` = `v4` + `.layerNorm`.) See ROADMAP / rezero notes.
enum BlockOutputNorm: String, Codable, CaseIterable, Sendable, Hashable {
    /// No output normalization — the block returns the merge result directly (v3/v4).
    case none
    /// `out = LayerNorm(merge)` — channel-wise LayerNorm over the C dimension at
    /// each board square (ConvNeXt convention), with per-channel learnable γ/β.
    case layerNorm = "layer_norm"
}

/// Source tensor for the optional "feature skip" — a single long concat skip that
/// hands a routed consumer *direct*, un-mixed access to early features alongside the
/// deep tower output (the DenseNet feature-reuse idea distilled to one skip). `.none`
/// is the global off switch: with `.none` the whole feature is absent and the network
/// builds byte-identical to a net that never had the axis. Future sources (raw input
/// planes, a mid-tower tap) slot in here.
enum FeatureSkipSource: String, Codable, CaseIterable, Sendable, Hashable {
    /// Feature skip disabled.
    case none
    /// The post-stem-BN tensor `x0` (normalized, 8×8, compute dtype).
    case stemOutput = "stem_output"
}

/// How a routed destination consumes the feature-skip source.
enum FeatureSkipFusion: String, Codable, CaseIterable, Sendable, Hashable {
    /// The destination reads `concat([dest_input, source])` directly; its own first
    /// conv (width `towerC + sourceC`) absorbs the projection. No extra tensors.
    case concatDirect = "concat_direct"
    /// One shared `ReLU(BN(Conv1×1(concat → towerC)))` node feeds the routed
    /// destinations at fixed width `towerC`. Adds a conv + BN. Supported for the
    /// policy/value heads; `validate()` rejects it only in combination with the
    /// final-block destination (the compressed node has no meaning as a block input).
    case compressConvBNReLU = "compress_conv_bn_relu"
}

/// Squeeze-and-Excitation channel-attention variant inside each residual block.
enum SEStyle: String, Codable, CaseIterable, Sendable, Hashable {
    case none
    /// FC2 emits `channels`, applied as `sigmoid(z)*x`.
    case attenuateOnly = "attenuate_only"
    /// FC2 emits `2*channels` (gamma||beta), applied as `sigmoid(gamma)*x + beta`.
    case scaleAndBias = "scale_and_bias"
}

/// How the β (additive) half of a `scaleAndBias` SE block's FC2 is
/// initialized when a network is built with random weights. FC2 emits
/// `2·channels` values per position: the γ half (columns `0..<C` of the
/// `[in, out]` weight, and bias `0..<C`) feeds `sigmoid` as the per-channel
/// scale; the β half (columns `C..<2C`, bias `C..<2C`) is added linearly.
/// Only meaningful for `SEStyle.scaleAndBias`; `validate()` rejects any value
/// other than `.glorot` on groups with another SE style.
enum SEBetaInit: String, Codable, CaseIterable, Sendable, Hashable {
    /// The β weight half is Glorot-normal like the γ half (the only behavior
    /// before this setting existed). Each block therefore starts by adding an
    /// input-dependent random per-channel offset to its residual branch.
    case glorot
    /// The β weight half and β bias are exactly zero at build, so at step 0 a
    /// block's SE computes `sigmoid(γ)·x` and adds nothing. β still receives
    /// gradient (its input — the FC1 activation — is nonzero), so it can learn
    /// an offset if one helps. The γ half keeps its Glorot init.
    case zero
}

/// Where the β half of a `scale_and_bias` SE FC2 lives, in each layout the
/// code handles. Single source of truth for the graph builder's zero-β init,
/// the `--derive-model` tensor rewrite, and the tests, so "which elements are
/// β" is never re-derived by hand. `C` = the block's channels, `r` = its SE
/// reduced width (`C / se_reduction_ratio`).
enum SEScaleAndBiasBetaHalf {
    /// Native engine layout of the FC2 weight, `[in, out] = [r, 2C]`,
    /// row-major: the β half is columns `C..<2C` of every row, i.e. one
    /// contiguous range per row.
    static func nativeWeightRanges(reducedChannels r: Int, channels c: Int) -> [Range<Int>] {
        (0..<r).map { row in (row * 2 * c + c)..<(row * 2 * c + 2 * c) }
    }

    /// PyTorch / on-disk layout of the FC2 weight, `[out, in] = [2C, r]`,
    /// row-major: the β half is output rows `C..<2C`, one contiguous range.
    static func torchWeightRange(reducedChannels r: Int, channels c: Int) -> Range<Int> {
        (c * r)..<(2 * c * r)
    }

    /// The FC2 bias, `[2C]` in both layouts: the β half is `C..<2C`.
    static func biasRange(channels c: Int) -> Range<Int> {
        c..<(2 * c)
    }
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
    /// IEEE 754 half. Same 10-bit mantissa precision as bf16's 7-bit but a
    /// far narrower exponent range than either bf16 or fp32 (max ≈ 65504,
    /// min normal ≈ 6.1e-5). ANE-native, so inference may run faster than
    /// bf16; training carries no loss scaling here, so small gradients can
    /// underflow to zero in the fp16 forward/backward even though the
    /// optimizer keeps fp32 masters (see `ChessTrainer`'s optimizer).
    case float16
}

// MARK: - BlockGroup

/// One run of identical residual blocks: a fully-specified block recipe (flat
/// fields, per the project's flat-schema rule) plus a `count`. The tower is an
/// ordered `[BlockGroup]`; EVERY block-configurable element lives here, so a
/// tower of count-1 groups can make every block different
/// (ARCHITECTURE_EXPANSION_PLAN.md Feature 2).
///
/// Width (`channels`) is per-group (WRN-style staircase). Where consecutive
/// expanded blocks differ in width, the engine inserts a 1×1 skip projection
/// on that block — a per-square linear remap, zero spatial mixing — and the
/// branch's conv1 carries the `inC → outC` step. Spatial shape is immutable
/// (8×8 everywhere; per-conv stride was considered and dropped 2026-06-12 —
/// decision record in the plan).
struct BlockGroup: Codable, Hashable, Sendable {
    /// How many consecutive blocks this recipe produces (>= 1).
    var count: Int
    /// The blocks' output width (their conv1 maps the incoming width here).
    var channels: Int
    var conv1KernelSize: Int
    var conv2KernelSize: Int
    var seStyle: SEStyle
    var seReductionRatio: Int            // consumed only when seStyle != none
    var useRezero: Bool
    /// The value every block's trainable ReZero scalar α starts at. Consumed
    /// only when `useRezero`. Zero is legal and is the published ReZero init
    /// (Bachlechner et al. 2020): every residual branch starts switched off,
    /// the identity path carries the signal, and α learns from step 1 (see
    /// `ChessNetwork.residualBlock`). A positive init such as `1/√N` starts
    /// every branch contributing instead.
    var rezeroAlphaInit: Float
    /// The asymptote `C` of the forward soft bound `C·tanh(α/C)` the trained α
    /// is applied through: the effective branch scale can approach ±C but
    /// never exceed it. Consumed only when `useRezero`. Before this field
    /// existed C was derived from the init (`rezeroAlphaInit ×
    /// NetworkArchitecture.rezeroTanhCeilingMultiple`), which made a zero
    /// init impossible — C = 0 divides by zero in the forward. An explicit
    /// cap decouples the two, so a zero-init group can still carry any
    /// positive cap. Decoding is format-version gated (`ArchitectureFormat`):
    /// files before format v6 resolve a missing value to that old derivation
    /// (`legacyRezeroAlphaCap`); v6+ files must state it.
    var rezeroAlphaCap: Float
    /// Hidden activation on this group's block main path (and the merge
    /// when `skipMerge == .activationGated`). The SE FC1 has its own
    /// `seActivation`.
    var activationFunction: ActivationFunction
    var activationStyle: BlockActivationStyle
    var skipMerge: BlockSkipMerge
    /// Per-group scale on the global live `DropoutRate`:
    /// effective rate = clamp(rate × multiplier, 0, 0.95). Baked into the
    /// graph as a constant composed with the live rate variable.
    var dropoutMultiplier: Float
    /// Optional normalization on the block's final output (after the skip
    /// merge). Optional-typed so models/sessions saved before this field
    /// existed decode it as `nil`; `nil` and `.none` both mean "no output
    /// norm". Read through `resolvedOutputNorm`, never the raw Optional.
    var outputNorm: BlockOutputNorm?
    /// Init of the β half of this group's `scale_and_bias` SE FC2 (see
    /// `SEBetaInit`). Changes only the random-weights build, never a tensor
    /// shape. Decoding is format-version gated (`ArchitectureFormat`): files
    /// before format v4 resolve a missing value to `.glorot` (the only
    /// behavior that existed then); v4+ files must state it. The in-code
    /// value is `.glorot` for the same reason — every group built before this
    /// field existed was Glorot — and callers opting into `.zero` set it
    /// explicitly.
    var seBetaInit: SEBetaInit
    /// Hidden activation on this group's SE excitation FC1 (the pooled
    /// `C → C/r` bottleneck), independent of the main path's
    /// `activationFunction` (GitHub issue #2). It exists because the ReLU
    /// bottleneck is the one place in the network where units were measured
    /// dying (up to 41% of one block's FC1 units); FC1 runs on pooled
    /// `[batch, C/r]` vectors, so a leaky ReLU there costs essentially
    /// nothing, while leaky ReLU on every conv costs several percent of
    /// training speed. No activation has parameters, so this never changes a
    /// tensor. Meaningful only when `seStyle != .none`; `validate()` requires
    /// it to equal `activationFunction` on an SE-less group so two
    /// architectures that build the same graph are equal. Decoding is
    /// format-version gated (`ArchitectureFormat`): files before format v5
    /// resolve a missing value to the group's `activationFunction` (what the
    /// SE FC1 used before the field existed); v5+ files must state it.
    var seActivation: ActivationFunction

    /// `outputNorm` with the legacy-`nil` case folded into `.none`, so callers
    /// never branch on the Optional. This is the value the builder,
    /// `weightTensorPlan`, and `parameterCount` all read.
    var resolvedOutputNorm: BlockOutputNorm { outputNorm ?? .none }

    /// The ReZero soft-bound asymptote `C` in the forward's `C·tanh(α/C)`:
    /// the group's explicit `rezeroAlphaCap`. Meaningful only when
    /// `useRezero`. The one formula the graph builder, the group summary, the
    /// Build screen's diagram, the numerics audit and layer health all read —
    /// none of them derives C from the init on its own.
    var rezeroTanhCeiling: Double {
        Double(rezeroAlphaCap)
    }

    /// The cap a group had before `rezeroAlphaCap` existed: the init times
    /// `NetworkArchitecture.rezeroTanhCeilingMultiple`. The single source of
    /// that rule, used by the format decoder for files older than v6 and by
    /// the memberwise inits that predate the field. With the multiple at 1.0
    /// the product is exactly `alphaInit` (a Float widened to Double, scaled by
    /// 1, narrowed back), so a legacy-resolved group is bit-identical to what
    /// the engine computed from the init before.
    static func legacyRezeroAlphaCap(forAlphaInit alphaInit: Float) -> Float {
        Float(Double(alphaInit) * NetworkArchitecture.rezeroTanhCeilingMultiple)
    }

    /// Set the group's main-path activation. On a group without an SE block
    /// `seActivation` moves with it: there it is dead configuration (no FC1
    /// to apply it to) and `validate()` requires the two equal, so two
    /// architectures that build the same graph stay equal. On a group with an
    /// SE block `seActivation` is left alone — it changes only when set on its
    /// own. The single rule the Build-New-Model screen and
    /// `--derive-model --set-activation` both apply, so the same edit made
    /// either way yields the same architecture.
    mutating func setActivationFunction(_ activation: ActivationFunction) {
        activationFunction = activation
        if seStyle == .none {
            seActivation = activation
        }
    }

    enum CodingKeys: String, CodingKey {
        case count
        case channels
        case conv1KernelSize = "conv1_kernel_size"
        case conv2KernelSize = "conv2_kernel_size"
        case seStyle = "se_style"
        case seReductionRatio = "se_reduction_ratio"
        case useRezero = "use_rezero"
        case rezeroAlphaInit = "rezero_alpha_init"
        case rezeroAlphaCap = "rezero_alpha_cap"
        case activationFunction = "activation_function"
        case activationStyle = "activation_style"
        case skipMerge = "skip_merge"
        case dropoutMultiplier = "dropout_multiplier"
        case outputNorm = "output_norm"
        case seBetaInit = "se_beta_init"
        case seActivation = "se_activation"
    }

    /// Full memberwise init (spelled out because the custom `Codable` below
    /// suppresses the synthesized one): every field, the ReZero cap and the
    /// SE FC1 activation included. `outputNorm` and `seBetaInit` keep the
    /// defaults the synthesized init had: both are the behavior every group
    /// had before the field existed. The overloads below omit the cap (and
    /// optionally the SE activation) and spell the arrangements that existed
    /// before those fields did.
    init(
        count: Int,
        channels: Int,
        conv1KernelSize: Int,
        conv2KernelSize: Int,
        seStyle: SEStyle,
        seReductionRatio: Int,
        useRezero: Bool,
        rezeroAlphaInit: Float,
        rezeroAlphaCap: Float,
        activationFunction: ActivationFunction,
        activationStyle: BlockActivationStyle,
        skipMerge: BlockSkipMerge,
        dropoutMultiplier: Float,
        outputNorm: BlockOutputNorm? = nil,
        seBetaInit: SEBetaInit = .glorot,
        seActivation: ActivationFunction
    ) {
        self.count = count
        self.channels = channels
        self.conv1KernelSize = conv1KernelSize
        self.conv2KernelSize = conv2KernelSize
        self.seStyle = seStyle
        self.seReductionRatio = seReductionRatio
        self.useRezero = useRezero
        self.rezeroAlphaInit = rezeroAlphaInit
        self.rezeroAlphaCap = rezeroAlphaCap
        self.activationFunction = activationFunction
        self.activationStyle = activationStyle
        self.skipMerge = skipMerge
        self.dropoutMultiplier = dropoutMultiplier
        self.outputNorm = outputNorm
        self.seBetaInit = seBetaInit
        self.seActivation = seActivation
    }

    /// A group whose ReZero cap is derived from its init
    /// (`legacyRezeroAlphaCap`: `rezeroAlphaInit ×
    /// NetworkArchitecture.rezeroTanhCeilingMultiple`) — the only arrangement
    /// that existed before `rezeroAlphaCap` did, so every recipe written
    /// before it (code presets, tests) keeps its meaning and builds the same
    /// graph. `seActivation` is required here; the overload below, which also
    /// omits it, is the "SE FC1 shares the group's activation" spelling. A
    /// zero-init group needs the full init: its derived cap would be zero,
    /// which `validate()` rejects.
    init(
        count: Int,
        channels: Int,
        conv1KernelSize: Int,
        conv2KernelSize: Int,
        seStyle: SEStyle,
        seReductionRatio: Int,
        useRezero: Bool,
        rezeroAlphaInit: Float,
        activationFunction: ActivationFunction,
        activationStyle: BlockActivationStyle,
        skipMerge: BlockSkipMerge,
        dropoutMultiplier: Float,
        outputNorm: BlockOutputNorm? = nil,
        seBetaInit: SEBetaInit = .glorot,
        seActivation: ActivationFunction
    ) {
        self.init(
            count: count,
            channels: channels,
            conv1KernelSize: conv1KernelSize,
            conv2KernelSize: conv2KernelSize,
            seStyle: seStyle,
            seReductionRatio: seReductionRatio,
            useRezero: useRezero,
            rezeroAlphaInit: rezeroAlphaInit,
            rezeroAlphaCap: Self.legacyRezeroAlphaCap(forAlphaInit: rezeroAlphaInit),
            activationFunction: activationFunction,
            activationStyle: activationStyle,
            skipMerge: skipMerge,
            dropoutMultiplier: dropoutMultiplier,
            outputNorm: outputNorm,
            seBetaInit: seBetaInit,
            seActivation: seActivation)
    }

    /// A group whose SE FC1 uses the group's own `activationFunction` and
    /// whose ReZero cap is derived from its init — the only arrangement that
    /// existed before `seActivation` and `rezeroAlphaCap` did, so every recipe
    /// written before them (code presets, tests) keeps its meaning.
    init(
        count: Int,
        channels: Int,
        conv1KernelSize: Int,
        conv2KernelSize: Int,
        seStyle: SEStyle,
        seReductionRatio: Int,
        useRezero: Bool,
        rezeroAlphaInit: Float,
        activationFunction: ActivationFunction,
        activationStyle: BlockActivationStyle,
        skipMerge: BlockSkipMerge,
        dropoutMultiplier: Float,
        outputNorm: BlockOutputNorm? = nil,
        seBetaInit: SEBetaInit = .glorot
    ) {
        self.init(
            count: count,
            channels: channels,
            conv1KernelSize: conv1KernelSize,
            conv2KernelSize: conv2KernelSize,
            seStyle: seStyle,
            seReductionRatio: seReductionRatio,
            useRezero: useRezero,
            rezeroAlphaInit: rezeroAlphaInit,
            activationFunction: activationFunction,
            activationStyle: activationStyle,
            skipMerge: skipMerge,
            dropoutMultiplier: dropoutMultiplier,
            outputNorm: outputNorm,
            seBetaInit: seBetaInit,
            seActivation: activationFunction)
    }

    /// Decodes under the format attached to the decoder (strict current
    /// version when none is attached — see `ArchitectureFormat`).
    init(from decoder: Decoder) throws {
        try self.init(from: decoder, format: ArchitectureFormat.DecodeFormat.from(decoder))
    }

    /// Decodes one group from a file of `format.formatVersion`. `se_beta_init`
    /// is required from `ArchitectureFormat.seBetaInitRequiredFromVersion`
    /// (older files resolve it to `.glorot`), `se_activation` from
    /// `ArchitectureFormat.seActivationRequiredFromVersion` (older files
    /// resolve it to this group's `activation_function`), and
    /// `rezero_alpha_cap` from `ArchitectureFormat.rezeroAlphaCapRequiredFromVersion`
    /// (older files resolve it to `legacyRezeroAlphaCap` of this group's
    /// `rezero_alpha_init`). Every resolution is recorded on `format`'s log.
    init(from decoder: Decoder, format: ArchitectureFormat.DecodeFormat) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        count = try c.decode(Int.self, forKey: .count)
        channels = try c.decode(Int.self, forKey: .channels)
        conv1KernelSize = try c.decode(Int.self, forKey: .conv1KernelSize)
        conv2KernelSize = try c.decode(Int.self, forKey: .conv2KernelSize)
        seStyle = try c.decode(SEStyle.self, forKey: .seStyle)
        seReductionRatio = try c.decode(Int.self, forKey: .seReductionRatio)
        useRezero = try c.decode(Bool.self, forKey: .useRezero)
        rezeroAlphaInit = try c.decode(Float.self, forKey: .rezeroAlphaInit)
        activationFunction = try c.decode(ActivationFunction.self, forKey: .activationFunction)
        activationStyle = try c.decode(BlockActivationStyle.self, forKey: .activationStyle)
        skipMerge = try c.decode(BlockSkipMerge.self, forKey: .skipMerge)
        dropoutMultiplier = try c.decode(Float.self, forKey: .dropoutMultiplier)
        outputNorm = try c.decodeIfPresent(BlockOutputNorm.self, forKey: .outputNorm)
        if let stated = try c.decodeIfPresent(SEBetaInit.self, forKey: .seBetaInit) {
            seBetaInit = stated
        } else if format.allowsMissingSEBetaInit {
            seBetaInit = .glorot
            format.legacyLog.record(
                "\(ArchitectureFormat.location(of: decoder)).\(CodingKeys.seBetaInit.rawValue) := \(SEBetaInit.glorot.rawValue)")
        } else {
            throw ArchitectureFormat.FormatError.missingRequiredField(
                field: CodingKeys.seBetaInit.rawValue,
                location: ArchitectureFormat.location(of: decoder),
                formatVersion: format.formatVersion,
                source: format.source)
        }
        if let stated = try c.decodeIfPresent(ActivationFunction.self, forKey: .seActivation) {
            seActivation = stated
        } else if format.allowsMissingSEActivation {
            seActivation = activationFunction
            format.legacyLog.record(
                "\(ArchitectureFormat.location(of: decoder)).\(CodingKeys.seActivation.rawValue) := \(activationFunction.rawValue) "
                    + "(the group's \(CodingKeys.activationFunction.rawValue))")
        } else {
            throw ArchitectureFormat.FormatError.missingRequiredField(
                field: CodingKeys.seActivation.rawValue,
                location: ArchitectureFormat.location(of: decoder),
                formatVersion: format.formatVersion,
                source: format.source)
        }
        if let stated = try c.decodeIfPresent(Float.self, forKey: .rezeroAlphaCap) {
            rezeroAlphaCap = stated
        } else if format.allowsMissingRezeroAlphaCap {
            rezeroAlphaCap = Self.legacyRezeroAlphaCap(forAlphaInit: rezeroAlphaInit)
            format.legacyLog.record(
                "\(ArchitectureFormat.location(of: decoder)).\(CodingKeys.rezeroAlphaCap.rawValue) := \(rezeroAlphaCap) "
                    + "(the group's \(CodingKeys.rezeroAlphaInit.rawValue) × \(NetworkArchitecture.rezeroTanhCeilingMultiple))")
        } else {
            throw ArchitectureFormat.FormatError.missingRequiredField(
                field: CodingKeys.rezeroAlphaCap.rawValue,
                location: ArchitectureFormat.location(of: decoder),
                formatVersion: format.formatVersion,
                source: format.source)
        }
    }

    /// Writes every field. `output_norm` keeps its pre-existing
    /// write-only-when-set form so older fields encode byte-identically;
    /// `se_beta_init`, `se_activation` and `rezero_alpha_cap` are ALWAYS
    /// written (even when they equal their legacy resolution, and on groups
    /// without ReZero), so a current-version file is self-describing.
    func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(count, forKey: .count)
        try c.encode(channels, forKey: .channels)
        try c.encode(conv1KernelSize, forKey: .conv1KernelSize)
        try c.encode(conv2KernelSize, forKey: .conv2KernelSize)
        try c.encode(seStyle, forKey: .seStyle)
        try c.encode(seReductionRatio, forKey: .seReductionRatio)
        try c.encode(useRezero, forKey: .useRezero)
        try c.encode(rezeroAlphaInit, forKey: .rezeroAlphaInit)
        try c.encode(rezeroAlphaCap, forKey: .rezeroAlphaCap)
        try c.encode(activationFunction, forKey: .activationFunction)
        try c.encode(activationStyle, forKey: .activationStyle)
        try c.encode(skipMerge, forKey: .skipMerge)
        try c.encode(dropoutMultiplier, forKey: .dropoutMultiplier)
        try c.encodeIfPresent(outputNorm, forKey: .outputNorm)
        try c.encode(seBetaInit, forKey: .seBetaInit)
        try c.encode(seActivation, forKey: .seActivation)
    }
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

    /// `shape` with size-1 axes removed — the comparable form when checking a
    /// plan entry against a tensor from somewhere else.
    ///
    /// The plan records LOGICAL shapes (torch/state_dict convention: a BN gamma
    /// is `[C]`), while the MPSGraph builder declares the same tensor in
    /// broadcast-ready form (`[1, C, 1, 1]`) so it can be applied across NCHW
    /// without a reshape. Both describe the same values in the same order;
    /// only the degenerate axes differ. Comparing raw shapes would therefore
    /// reject every correctly-built network, while comparing element counts
    /// alone is too weak — it cannot tell `[in, out]` from `[out, in]`.
    /// Squeezing is the useful middle: tolerant of broadcast axes, still
    /// sensitive to a transposed or re-factored tensor.
    var squeezedShape: [Int] { Self.squeeze(shape) }

    /// Drop size-1 axes. A tensor that is all-ones squeezes to `[]`, which is
    /// fine as long as both sides of a comparison are squeezed the same way.
    static func squeeze(_ shape: [Int]) -> [Int] { shape.filter { $0 != 1 } }
}

// MARK: - Errors

enum NetworkArchitectureError: Error, CustomStringConvertible, Equatable {
    case kernelMustBeOdd(field: String, value: Int)
    case nonPositive(field: String, value: Int)
    case channelsNotDivisibleByReduction(channels: Int, reduction: Int)
    case valueConvChannelsExceedChannels(conv: Int, channels: Int)
    /// A Float field that must be finite and >= 0 (e.g. a dropout
    /// multiplier). Carries the Float directly — never coerced to Int,
    /// which would trap on the NaN/infinite values this case exists to
    /// reject.
    case mustBeFiniteNonNegative(field: String, value: Float)
    /// A Float field that must be finite and strictly > 0. Distinct from
    /// `mustBeFiniteNonNegative` because zero is itself the failure mode here:
    /// a ReZero cap `rezeroAlphaCap == 0` makes the forward `C · tanh(α / C)`
    /// divide by zero (`0/0 → NaN` at α = 0, `±∞ · 0` otherwise) and poisons
    /// the whole tower. (Before the cap was its own field it was derived from
    /// `rezeroAlphaInit`, so this guarded the init instead and a zero init was
    /// impossible; the init is now only required to be finite and >= 0.)
    /// Carries the Float directly (never coerced to Int) so it can report the
    /// NaN/infinite values it also rejects.
    case mustBeFinitePositive(field: String, value: Float)
    /// `se_beta_init` other than `glorot` on a group whose SE style has no
    /// β half (only `scale_and_bias` does).
    case seBetaInitRequiresScaleAndBias(group: Int, seStyle: SEStyle, seBetaInit: SEBetaInit)
    /// `se_activation` differing from the group's `activation_function` on a
    /// group with no SE block, where it would be dead configuration.
    case seActivationRequiresSE(group: Int, seActivation: ActivationFunction, activationFunction: ActivationFunction)
    /// Feature skip is enabled (`source != .none`) but no destination is routed.
    case featureSkipNoDestination
    /// A feature-skip combination that is config-carried but unsupported —
    /// currently only `compressConvBNReLU` fusion together with the
    /// `toFinalBlock` destination. Every other feature-skip option (head
    /// fusion in either mode, concat-direct to the final block) is fully built.
    case featureSkipUnsupported(option: String)

    var description: String {
        switch self {
        case .featureSkipNoDestination:
            return "featureSkipSource is enabled but no destination is routed (set at least one of featureSkipToPolicyHead / featureSkipToValueHead)"
        case .featureSkipUnsupported(let option):
            return "feature-skip combination '\(option)' is not supported"
        case .mustBeFiniteNonNegative(let field, let value):
            return "\(field) must be finite and >= 0 (got \(value))"
        case .mustBeFinitePositive(let field, let value):
            return "\(field) must be finite and > 0 (got \(value))"
        case .seBetaInitRequiresScaleAndBias(let group, let seStyle, let seBetaInit):
            return "blockGroups[\(group)].seBetaInit is '\(seBetaInit.rawValue)' but its se_style is "
                + "'\(seStyle.rawValue)'; only '\(SEStyle.scaleAndBias.rawValue)' has a β half, so every "
                + "other SE style requires se_beta_init '\(SEBetaInit.glorot.rawValue)'"
        case .seActivationRequiresSE(let group, let seActivation, let activationFunction):
            return "blockGroups[\(group)].seActivation is '\(seActivation.rawValue)' but the group has no SE block "
                + "(se_style '\(SEStyle.none.rawValue)'); an SE-less group's se_activation must equal its "
                + "activation_function ('\(activationFunction.rawValue)')"
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
    /// Ordered block groups, input → output (>= 1 group; validated). The
    /// stem outputs the FIRST group's width; the heads read the LAST's.
    var blockGroups: [BlockGroup]
    var stemConvKernelSize: Int
    /// Hidden activation for the tower-LEVEL sites (stem activation when
    /// post-act, tower-end activation, both heads). Block main paths use
    /// their group's own `activationFunction`.
    var activationFunction: ActivationFunction

    // Policy head ---------------------------------------------------------
    var policyHeadStyle: PolicyHeadStyle
    var policyPreConvChannels: Int       // K for intermediate_conv / fc_bottleneck

    // Value head ----------------------------------------------------------
    var valueHeadStyle: ValueHeadStyle
    var valueHeadConvChannels: Int
    var valueHeadHiddenUnits: Int

    // Precision -----------------------------------------------------------
    var computeDataType: ComputeDataType

    // Feature skip (optional long concat skip) ----------------------------
    /// Source tensor for the feature skip; `.none` = disabled (default).
    var featureSkipSource: FeatureSkipSource
    /// How routed destinations consume the source.
    var featureSkipFusion: FeatureSkipFusion
    /// Route the skip into the policy head's input.
    var featureSkipToPolicyHead: Bool
    /// Route the skip into the value head's input.
    var featureSkipToValueHead: Bool
    /// Route the skip into the final tower block's input. (Phase 2 — `validate()`
    /// currently rejects this; carried so the axis is complete.)
    var featureSkipToFinalBlock: Bool

    // Fixed-by-engine (not stored, not in init) ---------------------------
    static let boardSize = 8
    static let policyChannels = 76
    static var policySize: Int { policyChannels * boardSize * boardSize }   // 4864

    /// All-required memberwise init — NO defaults (no silent fallbacks).
    init(
        inputEncoding: InputEncoding,
        blockGroups: [BlockGroup],
        stemConvKernelSize: Int,
        activationFunction: ActivationFunction,
        policyHeadStyle: PolicyHeadStyle,
        policyPreConvChannels: Int,
        valueHeadStyle: ValueHeadStyle,
        valueHeadConvChannels: Int,
        valueHeadHiddenUnits: Int,
        computeDataType: ComputeDataType,
        featureSkipSource: FeatureSkipSource,
        featureSkipFusion: FeatureSkipFusion,
        featureSkipToPolicyHead: Bool,
        featureSkipToValueHead: Bool,
        featureSkipToFinalBlock: Bool
    ) {
        self.inputEncoding = inputEncoding
        self.blockGroups = blockGroups
        self.stemConvKernelSize = stemConvKernelSize
        self.activationFunction = activationFunction
        self.policyHeadStyle = policyHeadStyle
        self.policyPreConvChannels = policyPreConvChannels
        self.valueHeadStyle = valueHeadStyle
        self.valueHeadConvChannels = valueHeadConvChannels
        self.valueHeadHiddenUnits = valueHeadHiddenUnits
        self.computeDataType = computeDataType
        self.featureSkipSource = featureSkipSource
        self.featureSkipFusion = featureSkipFusion
        self.featureSkipToPolicyHead = featureSkipToPolicyHead
        self.featureSkipToValueHead = featureSkipToValueHead
        self.featureSkipToFinalBlock = featureSkipToFinalBlock
    }

    /// Convenience for the (common) uniform tower: one group carrying every
    /// block field, count = `numBlocks`. All-required — no defaults.
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
        computeDataType: ComputeDataType,
        blockOutputNorm: BlockOutputNorm? = nil
    ) {
        self.init(
            inputEncoding: inputEncoding,
            blockGroups: [BlockGroup(
                count: numBlocks,
                channels: channels,
                conv1KernelSize: blockConv1KernelSize,
                conv2KernelSize: blockConv2KernelSize,
                seStyle: blockSeStyle,
                seReductionRatio: blockSeReductionRatio,
                useRezero: blockUseRezero,
                rezeroAlphaInit: rezeroAlphaInit,
                // Every historical tower derived its ReZero cap from the init.
                // A tower with an explicit cap (e.g. a zero init) sets
                // `rezeroAlphaCap` on the returned value's groups.
                rezeroAlphaCap: BlockGroup.legacyRezeroAlphaCap(forAlphaInit: rezeroAlphaInit),
                activationFunction: activationFunction,
                activationStyle: blockActivationStyle,
                skipMerge: blockSkipMerge,
                dropoutMultiplier: 1,
                outputNorm: blockOutputNorm,
                // The uniform convenience init describes the historical
                // single-recipe towers, all of which were Glorot-β. A zero-β
                // tower sets `seBetaInit` on the returned value's groups.
                seBetaInit: .glorot,
                // Every historical tower's SE FC1 used the tower's single
                // activation. A tower with a different SE FC1 activation sets
                // `seActivation` on the returned value's groups.
                seActivation: activationFunction
            )],
            stemConvKernelSize: stemConvKernelSize,
            activationFunction: activationFunction,
            policyHeadStyle: policyHeadStyle,
            policyPreConvChannels: policyPreConvChannels,
            valueHeadStyle: valueHeadStyle,
            valueHeadConvChannels: valueHeadConvChannels,
            valueHeadHiddenUnits: valueHeadHiddenUnits,
            computeDataType: computeDataType,
            // Uniform towers default to feature-skip OFF; presets that enable it
            // mutate the returned value's `featureSkip*` fields.
            featureSkipSource: .none,
            featureSkipFusion: .concatDirect,
            featureSkipToPolicyHead: false,
            featureSkipToValueHead: false,
            featureSkipToFinalBlock: false
        )
    }

    // MARK: Codable — explicit lower_snake_case keys
    //
    // Encode writes ONLY `block_groups` for the tower. Decode reads both
    // forms forever: `block_groups` when present, otherwise the legacy
    // uniform keys (`channels`, `num_blocks`, `block_*`) expand to a single
    // group — every existing safetensors/session file loads unchanged.
    // `dropout_multiplier` for legacy saves is 1 (that IS the legacy
    // semantic: the global rate applied unscaled).
    //
    // Fields added after format v3 are version-gated per block group (see
    // `ArchitectureFormat` and `BlockGroup.init(from:format:)`): the carrier's
    // format version decides whether an absent field resolves to its legacy
    // value or is a hard error.

    enum CodingKeys: String, CodingKey {
        case inputEncoding = "input_encoding"
        case blockGroups = "block_groups"
        case stemConvKernelSize = "stem_conv_kernel_size"
        case activationFunction = "activation_function"
        case policyHeadStyle = "policy_head_style"
        case policyPreConvChannels = "policy_pre_conv_channels"
        case valueHeadStyle = "value_head_style"
        case valueHeadConvChannels = "value_head_conv_channels"
        case valueHeadHiddenUnits = "value_head_hidden_units"
        case computeDataType = "compute_data_type"
        case featureSkipSource = "feature_skip_source"
        case featureSkipFusion = "feature_skip_fusion"
        case featureSkipToPolicyHead = "feature_skip_to_policy_head"
        case featureSkipToValueHead = "feature_skip_to_value_head"
        case featureSkipToFinalBlock = "feature_skip_to_final_block"
        // Legacy uniform-tower keys — decode-only, never written.
        case legacyChannels = "channels"
        case legacyNumBlocks = "num_blocks"
        case legacyBlockActivationStyle = "block_activation_style"
        case legacyBlockSkipMerge = "block_skip_merge"
        case legacyBlockUseRezero = "block_use_rezero"
        case legacyRezeroAlphaInit = "rezero_alpha_init"
        case legacyBlockConv1KernelSize = "block_conv1_kernel_size"
        case legacyBlockConv2KernelSize = "block_conv2_kernel_size"
        case legacyBlockSeStyle = "block_se_style"
        case legacyBlockSeReductionRatio = "block_se_reduction_ratio"
    }

    /// Decodes under the format attached to the decoder (strict current
    /// version when none is attached — see `ArchitectureFormat`).
    init(from decoder: Decoder) throws {
        try self.init(from: decoder, format: ArchitectureFormat.DecodeFormat.from(decoder))
    }

    /// Decodes an architecture from a file of `format.formatVersion`, threading
    /// the format into every block group (see `BlockGroup.init(from:format:)`).
    init(from decoder: Decoder, format: ArchitectureFormat.DecodeFormat) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        inputEncoding = try c.decode(InputEncoding.self, forKey: .inputEncoding)
        stemConvKernelSize = try c.decode(Int.self, forKey: .stemConvKernelSize)
        activationFunction = try c.decode(ActivationFunction.self, forKey: .activationFunction)
        policyHeadStyle = try c.decode(PolicyHeadStyle.self, forKey: .policyHeadStyle)
        policyPreConvChannels = try c.decode(Int.self, forKey: .policyPreConvChannels)
        valueHeadStyle = try c.decode(ValueHeadStyle.self, forKey: .valueHeadStyle)
        valueHeadConvChannels = try c.decode(Int.self, forKey: .valueHeadConvChannels)
        valueHeadHiddenUnits = try c.decode(Int.self, forKey: .valueHeadHiddenUnits)
        computeDataType = try c.decode(ComputeDataType.self, forKey: .computeDataType)
        // Feature skip: optional + defaulted so every pre-feature-skip file decodes
        // to a fully-off (byte-identical) configuration.
        featureSkipSource = try c.decodeIfPresent(FeatureSkipSource.self, forKey: .featureSkipSource) ?? .none
        featureSkipFusion = try c.decodeIfPresent(FeatureSkipFusion.self, forKey: .featureSkipFusion) ?? .concatDirect
        featureSkipToPolicyHead = try c.decodeIfPresent(Bool.self, forKey: .featureSkipToPolicyHead) ?? false
        featureSkipToValueHead = try c.decodeIfPresent(Bool.self, forKey: .featureSkipToValueHead) ?? false
        featureSkipToFinalBlock = try c.decodeIfPresent(Bool.self, forKey: .featureSkipToFinalBlock) ?? false
        if c.contains(.blockGroups) {
            var groupsContainer = try c.nestedUnkeyedContainer(forKey: .blockGroups)
            var groups: [BlockGroup] = []
            while !groupsContainer.isAtEnd {
                groups.append(try BlockGroup(from: try groupsContainer.superDecoder(), format: format))
            }
            // An empty array is structurally invalid: the stem/head/summary
            // accessors `preconditionFailure` on no groups, and they are read
            // (SessionManifest.extract, SafetensorsModelIO load) before
            // `validate()` runs — so a malformed `"block_groups": []` must be
            // rejected here as a thrown decode error, not allowed to crash the
            // process later.
            guard !groups.isEmpty else {
                throw DecodingError.dataCorruptedError(
                    forKey: .blockGroups, in: c,
                    debugDescription: "block_groups must contain at least one group")
            }
            blockGroups = groups
        } else {
            let legacyAlphaInit = try c.decode(Float.self, forKey: .legacyRezeroAlphaInit)
            let legacyAlphaCap = BlockGroup.legacyRezeroAlphaCap(forAlphaInit: legacyAlphaInit)
            blockGroups = [BlockGroup(
                count: try c.decode(Int.self, forKey: .legacyNumBlocks),
                channels: try c.decode(Int.self, forKey: .legacyChannels),
                conv1KernelSize: try c.decode(Int.self, forKey: .legacyBlockConv1KernelSize),
                conv2KernelSize: try c.decode(Int.self, forKey: .legacyBlockConv2KernelSize),
                seStyle: try c.decode(SEStyle.self, forKey: .legacyBlockSeStyle),
                seReductionRatio: try c.decode(Int.self, forKey: .legacyBlockSeReductionRatio),
                useRezero: try c.decode(Bool.self, forKey: .legacyBlockUseRezero),
                rezeroAlphaInit: legacyAlphaInit,
                rezeroAlphaCap: legacyAlphaCap,
                activationFunction: activationFunction,
                activationStyle: try c.decode(BlockActivationStyle.self, forKey: .legacyBlockActivationStyle),
                skipMerge: try c.decode(BlockSkipMerge.self, forKey: .legacyBlockSkipMerge),
                dropoutMultiplier: 1,
                seBetaInit: .glorot,
                seActivation: activationFunction
            )]
            // The uniform-tower keys predate block groups, so no writer of any
            // version that has `se_beta_init`, `se_activation` or
            // `rezero_alpha_cap` emits them: this form is legacy by
            // construction, whatever version the carrier states.
            format.legacyLog.record(
                "legacy uniform-tower keys: block_groups[0].\(BlockGroup.CodingKeys.seBetaInit.rawValue) := \(SEBetaInit.glorot.rawValue), "
                    + "block_groups[0].\(BlockGroup.CodingKeys.seActivation.rawValue) := \(activationFunction.rawValue), "
                    + "block_groups[0].\(BlockGroup.CodingKeys.rezeroAlphaCap.rawValue) := \(legacyAlphaCap)")
        }
    }

    func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(inputEncoding, forKey: .inputEncoding)
        try c.encode(blockGroups, forKey: .blockGroups)
        try c.encode(stemConvKernelSize, forKey: .stemConvKernelSize)
        try c.encode(activationFunction, forKey: .activationFunction)
        try c.encode(policyHeadStyle, forKey: .policyHeadStyle)
        try c.encode(policyPreConvChannels, forKey: .policyPreConvChannels)
        try c.encode(valueHeadStyle, forKey: .valueHeadStyle)
        try c.encode(valueHeadConvChannels, forKey: .valueHeadConvChannels)
        try c.encode(valueHeadHiddenUnits, forKey: .valueHeadHiddenUnits)
        try c.encode(computeDataType, forKey: .computeDataType)
        try c.encode(featureSkipSource, forKey: .featureSkipSource)
        try c.encode(featureSkipFusion, forKey: .featureSkipFusion)
        try c.encode(featureSkipToPolicyHead, forKey: .featureSkipToPolicyHead)
        try c.encode(featureSkipToValueHead, forKey: .featureSkipToValueHead)
        try c.encode(featureSkipToFinalBlock, forKey: .featureSkipToFinalBlock)
    }

    // MARK: Derived shape scalars

    var inputPlanes: Int { inputEncoding.planeCount }
    var boardSize: Int { Self.boardSize }
    var policyChannels: Int { Self.policyChannels }
    var policySize: Int { Self.policySize }
    var valueHeadClasses: Int { valueHeadStyle == .wdlSoftmax ? 3 : 1 }

    // MARK: Feature skip — the single width formula shared by builder, plan, paramCount

    /// Whether the feature skip is active.
    var featureSkipEnabled: Bool { featureSkipSource != .none }

    /// The width of the source tensor being skipped. Stem output = the stem's width.
    var featureSkipSourceChannels: Int {
        switch featureSkipSource {
        case .none:       return 0
        case .stemOutput: return stemOutputChannels
        }
    }

    /// Effective input width for a head, accounting for a routed `concatDirect` skip.
    /// In `concatDirect` a routed head reads `concat([tower_out, source])`, so its
    /// first conv widens by the source width; unrouted heads (and any future compress
    /// mode, which feeds a fixed-`towerC` node) stay at `towerOutputChannels`. This is
    /// the ONE formula the graph builder, `weightTensorPlan`, and `parameterCount` all
    /// consume so the three never drift.
    func headInputChannels(routed: Bool) -> Int {
        let widen = featureSkipEnabled && featureSkipFusion == .concatDirect && routed
        return towerOutputChannels + (widen ? featureSkipSourceChannels : 0)
    }

    var policyHeadInputChannels: Int { headInputChannels(routed: featureSkipToPolicyHead) }
    var valueHeadInputChannels: Int { headInputChannels(routed: featureSkipToValueHead) }

    /// True when a shared compress-fusion node (`feature_skip.conv` + BN) must be
    /// built: `compressConvBNReLU` with at least one head routed. The node compresses
    /// `concat([tower_out, source])` back to `towerOutputChannels`, and the routed
    /// heads read it (so their input width stays `towerOutputChannels` — `headInputChannels`
    /// widens only for `concatDirect`).
    var featureSkipUsesCompressNode: Bool {
        featureSkipEnabled
            && featureSkipFusion == .compressConvBNReLU
            && (featureSkipToPolicyHead || featureSkipToValueHead)
    }

    /// The compress node's 1×1 conv input width = tower output + source.
    var featureSkipCompressInputChannels: Int {
        towerOutputChannels + featureSkipSourceChannels
    }

    /// Extra input channels concatenated onto an expanded block's input by a routed
    /// `concatDirect` skip-to-final-block. Non-zero ONLY for the last expanded block,
    /// and only under `concatDirect` (compress is head-only). The block's existing
    /// width-transition machinery (conv1 `inC`, pre-act BN1 size, and the 1×1 skip
    /// projection that appears when `inC != outC`) absorbs the widening — so the SAME
    /// `inC` must be threaded by the builder, `parameterCount`, `weightTensorPlan`, and
    /// the analyzer's `blockSpec`, or those four desync on the final block.
    func blockSkipExtraInputChannels(blockIndex: Int) -> Int {
        guard featureSkipEnabled,
              featureSkipFusion == .concatDirect,
              featureSkipToFinalBlock,
              blockIndex == numBlocks - 1 else { return 0 }
        return featureSkipSourceChannels
    }

    /// The tower flattened to one element per block (each returned group has
    /// `count == 1`). The ENGINE'S ONLY VIEW of the tower: graph builders,
    /// `weightTensorPlan`, `parameterCount`, and the analyzer walk this —
    /// groups are an authoring/persistence structure, never an engine concept.
    var expandedBlocks: [BlockGroup] {
        blockGroups.flatMap { group -> [BlockGroup] in
            var single = group
            single.count = 1
            return Array(repeating: single, count: group.count)
        }
    }

    /// Total block count across all groups (derived; no stored copy).
    var numBlocks: Int { blockGroups.reduce(0) { $0 + $1.count } }

    /// The stem's output width = the first group's channels.
    var stemOutputChannels: Int {
        guard let first = blockGroups.first else {
            preconditionFailure("NetworkArchitecture.blockGroups is empty (validate() rejects this)")
        }
        return first.channels
    }

    /// The tower's output width = the last group's channels. What the heads
    /// and the tower-end BN read. (There is deliberately NO uniform
    /// `channels` accessor — mixed towers have no single width, so every
    /// consumer must choose stem-side or head-side explicitly.)
    var towerOutputChannels: Int {
        guard let last = blockGroups.last else {
            preconditionFailure("NetworkArchitecture.blockGroups is empty (validate() rejects this)")
        }
        return last.channels
    }

    /// The widest block in the tower — sizes worst-case activation buffers
    /// and tensor-size guards.
    var maxBlockChannels: Int {
        guard let widest = blockGroups.map(\.channels).max() else {
            preconditionFailure("NetworkArchitecture.blockGroups is empty (validate() rejects this)")
        }
        return widest
    }

    /// The widest single activation tensor's channel count — the buffer-footprint
    /// worst case. Normally `maxBlockChannels`, but an enabled feature skip materializes
    /// a `concat([tower_out, source])` tensor (`towerOutputChannels + source`) that can
    /// exceed it, so fold that in. (`maxBlockChannels` stays the pure-tower quantity
    /// used by the legacy arch hash and per-block guards; this is the activation-size one.)
    var maxActivationChannels: Int {
        let base = maxBlockChannels
        return featureSkipEnabled ? max(base, towerOutputChannels + featureSkipSourceChannels) : base
    }

    /// Tower-end BN exists only when the LAST block is pre-activation (a
    /// pre-act tail ends un-normalized/un-activated; a post-act tail is
    /// already conditioned).
    var hasTowerEndBN: Bool {
        guard let last = blockGroups.last else {
            preconditionFailure("NetworkArchitecture.blockGroups is empty (validate() rejects this)")
        }
        return last.activationStyle == .pre
    }
    /// Stem ReLU exists only when the FIRST block is post-activation (a
    /// pre-act first block defers the first nonlinearity to its own BN→act).
    var hasStemActivation: Bool {
        guard let first = blockGroups.first else {
            preconditionFailure("NetworkArchitecture.blockGroups is empty (validate() rejects this)")
        }
        return first.activationStyle == .post
    }
    /// Human v-number for display only (no role in identity / hashing).
    /// Mixed-style towers report the FIRST group's lineage.
    var architectureVersionLabel: Int {
        guard let first = blockGroups.first else {
            preconditionFailure("NetworkArchitecture.blockGroups is empty (validate() rejects this)")
        }
        // Output normalization on any block is the v5-era addition (re-centered
        // clean-add highway); it post-dates the pre-vs-post v3/v4 split, so it
        // takes precedence. v3 (post-act) and v4 (pre-act) keep their labels
        // because neither carries an output norm.
        if blockGroups.contains(where: { $0.resolvedOutputNorm != .none }) { return 5 }
        return first.activationStyle == .pre ? 4 : 3
    }

    // MARK: Validation (structural only — memory budget is a build-time, device-aware check)

    func validate() throws {
        try requireOdd("stemConvKernelSize", stemConvKernelSize)
        try requirePositive("blockGroups.count", blockGroups.count)
        for (gi, g) in blockGroups.enumerated() {
            try requirePositive("blockGroups[\(gi)].count", g.count)
            try requirePositive("blockGroups[\(gi)].channels", g.channels)
            try requireOdd("blockGroups[\(gi)].conv1KernelSize", g.conv1KernelSize)
            try requireOdd("blockGroups[\(gi)].conv2KernelSize", g.conv2KernelSize)
            // Unconditional: the block computes channels / ratio regardless of SE style.
            try requirePositive("blockGroups[\(gi)].seReductionRatio", g.seReductionRatio)
            if g.seStyle != .none {
                guard g.channels % g.seReductionRatio == 0 else {
                    throw NetworkArchitectureError.channelsNotDivisibleByReduction(
                        channels: g.channels, reduction: g.seReductionRatio)
                }
            }
            guard g.dropoutMultiplier >= 0, g.dropoutMultiplier.isFinite else {
                throw NetworkArchitectureError.mustBeFiniteNonNegative(
                    field: "blockGroups[\(gi)].dropoutMultiplier", value: g.dropoutMultiplier)
            }
            if g.seStyle != .scaleAndBias, g.seBetaInit != .glorot {
                throw NetworkArchitectureError.seBetaInitRequiresScaleAndBias(
                    group: gi, seStyle: g.seStyle, seBetaInit: g.seBetaInit)
            }
            // An SE-less group has no FC1, so its se_activation is dead
            // configuration. Pinning it to the group's activation keeps one
            // value per graph: otherwise two SE-less architectures that build
            // the identical network would compare (and hash) unequal.
            if g.seStyle == .none, g.seActivation != g.activationFunction {
                throw NetworkArchitectureError.seActivationRequiresSE(
                    group: gi, seActivation: g.seActivation, activationFunction: g.activationFunction)
            }
            // ReZero. The cap C feeds a division in the forward `C · tanh(α / C)`:
            // a zero (or NaN/infinite) cap produces a NaN that propagates
            // through the entire tower, so it must be finite and > 0. The init
            // is the starting value of α: finite and >= 0, where 0 is the
            // published ReZero init (branch off at step 0, learning from step
            // 1). An init above the cap is allowed — the forward then simply
            // starts saturated at C·tanh(α₀/C) < α₀, which is a legitimate (if
            // unusual) choice, and trained raw α routinely exceeds C anyway.
            // The Build-New-Model fields are unvalidated TextFields, so a user
            // clearing one or typing 0 reaches here; guard at the single
            // chokepoint both the UI build path and JSON decode pass through.
            //
            // Neither value is constrained on a group without ReZero, matching
            // how the init has always been treated there: no tensor or graph
            // node reads them, and the Build screen hides both fields, so a
            // group switched off keeps whatever the user had without failing
            // validation for a value they cannot see.
            if g.useRezero {
                guard g.rezeroAlphaCap.isFinite, g.rezeroAlphaCap > 0 else {
                    throw NetworkArchitectureError.mustBeFinitePositive(
                        field: "blockGroups[\(gi)].rezeroAlphaCap", value: g.rezeroAlphaCap)
                }
                guard g.rezeroAlphaInit.isFinite, g.rezeroAlphaInit >= 0 else {
                    throw NetworkArchitectureError.mustBeFiniteNonNegative(
                        field: "blockGroups[\(gi)].rezeroAlphaInit", value: g.rezeroAlphaInit)
                }
            }
        }
        try requirePositive("policyPreConvChannels", policyPreConvChannels)
        try requirePositive("valueHeadConvChannels", valueHeadConvChannels)
        try requirePositive("valueHeadHiddenUnits", valueHeadHiddenUnits)
        guard valueHeadConvChannels <= towerOutputChannels else {
            throw NetworkArchitectureError.valueConvChannelsExceedChannels(
                conv: valueHeadConvChannels, channels: towerOutputChannels)
        }
        if featureSkipEnabled {
            guard featureSkipToPolicyHead || featureSkipToValueHead || featureSkipToFinalBlock else {
                throw NetworkArchitectureError.featureSkipNoDestination
            }
            // Compress builds a single tower-width head-fusion node; that tensor has no
            // meaning as a block input, so it cannot combine with the final-block
            // destination (which concatenates the raw source onto a block's input).
            if featureSkipFusion == .compressConvBNReLU, featureSkipToFinalBlock {
                throw NetworkArchitectureError.featureSkipUnsupported(option: "compress_conv_bn_relu + to_final_block")
            }
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
    /// in tests). Walks `expandedBlocks`, threading the incoming width — block
    /// `i`'s conv1 maps `inC → outC`, BN1 is sized `inC`, everything after runs
    /// at `outC`, and a width transition adds the 1×1 skip projection.
    var parameterCount: Int {
        let c0 = stemOutputChannels

        // Stem: conv (bias-free) + BN.
        let stem = (inputPlanes * c0 * stemConvKernelSize * stemConvKernelSize) + 4 * c0

        // Tower. `inCEff` folds in the final-block feature skip (`+ source` on the
        // last block's input under a routed concatDirect skip); it equals `inC`
        // everywhere else, so non-finalBlock configs are unchanged.
        var tower = 0
        var inC = c0
        for (i, spec) in expandedBlocks.enumerated() {
            let inCEff = inC + blockSkipExtraInputChannels(blockIndex: i)
            let outC = spec.channels
            let conv1 = outC * inCEff * spec.conv1KernelSize * spec.conv1KernelSize
            let conv2 = outC * outC * spec.conv2KernelSize * spec.conv2KernelSize
            // BN1 normalizes the block input (pre-act) or conv1 output
            // (post-act) — sized inCEff vs outC accordingly; BN2 is always outC.
            let bn1 = 4 * (spec.activationStyle == .pre ? inCEff : outC)
            let bn2 = 4 * outC
            let seReduced = spec.seStyle == .none ? 0 : outC / spec.seReductionRatio
            let se: Int
            switch spec.seStyle {
            case .none:          se = 0
            case .attenuateOnly: se = (outC * seReduced + seReduced) + (seReduced * outC + outC)
            case .scaleAndBias:  se = (outC * seReduced + seReduced) + (seReduced * 2 * outC + 2 * outC)
            }
            let rezero = spec.useRezero ? 1 : 0
            let proj = inCEff != outC ? inCEff * outC : 0
            // Optional output LayerNorm: per-channel γ + β (no running stats).
            let outNorm = spec.resolvedOutputNorm == .layerNorm ? 2 * outC : 0
            tower += conv1 + conv2 + bn1 + bn2 + se + rezero + proj + outNorm
            inC = outC
        }

        let cT = towerOutputChannels
        let towerEndBN = hasTowerEndBN ? 4 * cT : 0

        // Heads. The FIRST conv of each head reads the effective input width — wider
        // by the feature-skip source under a routed `concatDirect` skip, else `cT`.
        let cP = policyHeadInputChannels
        let cVin = valueHeadInputChannels

        // Policy head.
        let pK = policyPreConvChannels
        let policy: Int
        switch policyHeadStyle {
        case .simpleConv:
            policy = (cP * policyChannels) + policyChannels
        case .intermediateConv:
            policy = (cP * pK) + 4 * pK + (pK * policyChannels) + policyChannels
        case .fcBottleneck:
            let flat = pK * boardSize * boardSize
            policy = (cP * pK) + 4 * pK + (flat * policySize) + policySize
        }

        // Value head.
        let cv = valueHeadConvChannels
        let h = valueHeadHiddenUnits
        let flatV = boardSize * boardSize * cv
        let value = (cVin * cv) + 4 * cv + (flatV * h + h) + (h * valueHeadClasses + valueHeadClasses)

        // Compress fusion node (head-only): 1×1 conv (towerC+source → towerC) + BN.
        let compressNode = featureSkipUsesCompressNode
            ? (featureSkipCompressInputChannels * cT + 4 * cT)
            : 0

        return stem + tower + towerEndBN + compressNode + policy + value
    }

    // MARK: Summary (human-readable, computed from the config)

    /// Compact one-glance label for the title bar, e.g. "v3 · 8-block 3×3 · 128ch · 2,483,667 params".
    /// Multi-group towers render the kernel mix as "mixed" and the width as a
    /// stem→tower range when the widths differ.
    var shortLabel: String {
        let kDesc: String
        if blockGroups.count == 1, let g = blockGroups.first {
            kDesc = g.conv1KernelSize == g.conv2KernelSize
                ? "\(g.conv1KernelSize)×\(g.conv1KernelSize)"
                : "\(g.conv1KernelSize)×\(g.conv1KernelSize),\(g.conv2KernelSize)×\(g.conv2KernelSize)"
        } else {
            kDesc = "mixed"
        }
        let chDesc = stemOutputChannels == towerOutputChannels
            ? "\(towerOutputChannels)ch"
            : "\(stemOutputChannels)→\(towerOutputChannels)ch"
        return "v\(architectureVersionLabel) · \(numBlocks)-block \(kDesc) · \(chDesc) · \(parameterCount.formatted(.number)) params"
    }

    /// Fully-explicit tower description — every attribute of every group is
    /// rendered, with NO silent defaults (a reader never needs to know a
    /// default to read a summary; user direction 2026-06-12). `->` separates
    /// groups; skip projections are implied by adjacent width changes in the
    /// expansion, never written. The golden-string tests pin this exact form.
    var architectureSummary: String {
        let groupsDesc = blockGroups.map { Self.groupSummary($0) }.joined(separator: " -> ")
        let valueDesc = valueHeadStyle == .wdlSoftmax
            ? "WDL(\(valueHeadConvChannels)->FC\(valueHeadHiddenUnits))"
            : "tanh(\(valueHeadConvChannels)->FC\(valueHeadHiddenUnits))"
        // Rendered ONLY when enabled, so off (the default) keeps every preset's golden
        // string byte-identical to the pre-feature-skip form.
        let skipDesc: String
        if featureSkipEnabled {
            var dests: [String] = []
            if featureSkipToPolicyHead { dests.append("policy") }
            if featureSkipToValueHead { dests.append("value") }
            if featureSkipToFinalBlock { dests.append("finalBlock") }
            skipDesc = " . skip \(featureSkipSource.rawValue)->[\(dests.joined(separator: ","))]/\(featureSkipFusion.rawValue)"
        } else {
            skipDesc = ""
        }
        return "v\(architectureVersionLabel)"
            + " . in \(inputEncoding.rawValue)(\(inputPlanes)) -> stem \(stemOutputChannels) (\(stemConvKernelSize)x\(stemConvKernelSize))"
            + " . \(groupsDesc)"
            + " . act \(activationFunction.rawValue)"
            + " . policy \(policyHeadStyle.rawValue)(\(policySize))"
            + " . value \(valueDesc)"
            + skipDesc
            + " . \(computeDataType.rawValue) . \(parameterCount.formatted(.number)) params"
    }

    /// Multiplier on the per-block ReZero init `α₀` that gave the asymptotic
    /// ceiling `C = rezeroTanhCeilingMultiple · α₀` of the soft-bound
    /// `C·tanh(α/C)` in the forward before the cap became its own per-group
    /// field (`BlockGroup.rezeroAlphaCap`, format v6; see
    /// `ChessNetwork.residualBlock`, `documentation/rezero-alpha-clamp.md`).
    /// It now defines only that legacy rule (`BlockGroup.legacyRezeroAlphaCap`):
    /// how a file older than v6 resolves its missing cap, and what the
    /// memberwise inits that predate the field set. It must never change —
    /// every legacy file's cap, and therefore its graph and identity, is
    /// computed from it.
    ///
    /// With multiplier 1.0, `C = α₀ = 1/√N`, so effective α saturates at the
    /// variance-preserving value (Σα² ≈ 1 across N blocks). An absolute `C = 1.0`
    /// with `α₀ = 1/√N` was tried on the 5-block 7×7 tower and failed: effective α
    /// saturated ~0.95 across all blocks (Σα² ≈ 4.5), the residual-stream mean
    /// still exploded (bn1Mean 43→1384 over 1k steps), and the run broke ~step
    /// 5800 — the hard-clamp failure, delayed.
    static let rezeroTanhCeilingMultiple: Double = 1.0

    /// One group's explicit rendering, e.g.
    /// `5x[7x7+7x7 @128, SE+/4, relu/pre, clean_add, ReZero(0.447·tanh≤0.447), drop*1]`
    /// — see `rezeroDescription` for the ReZero clause.
    static func groupSummary(_ g: BlockGroup) -> String {
        let seDesc: String
        switch g.seStyle {
        case .none: seDesc = "no-SE"
        case .attenuateOnly: seDesc = "SE/\(g.seReductionRatio)"
        case .scaleAndBias: seDesc = "SE+/\(g.seReductionRatio)"
        }
        // Rendered only for a zero-β group, so every Glorot-β summary (all
        // architectures that predate the setting) is byte-identical.
        let seBetaDesc = g.seBetaInit == .zero ? " β0" : ""
        let seActivationDesc = seActivationMarker(g)
        let rezeroDesc = rezeroDescription(g)
        // Only render the output-norm clause when present, so v3/v4 group
        // summaries (and their golden-string tests) are byte-identical.
        let outNormDesc = g.resolvedOutputNorm == .none ? "" : ", out:\(g.resolvedOutputNorm.rawValue)"
        return "\(g.count)x[\(g.conv1KernelSize)x\(g.conv1KernelSize)+\(g.conv2KernelSize)x\(g.conv2KernelSize)"
            + " @\(g.channels), \(seDesc)\(seBetaDesc)\(seActivationDesc), \(g.activationFunction.rawValue)/\(g.activationStyle.rawValue)"
            + ", \(g.skipMerge.rawValue), \(rezeroDesc)\(outNormDesc)"
            + ", drop*\(String(format: "%g", g.dropoutMultiplier))]"
    }

    /// The ReZero clause of a group's rendering: `ReZero(<α₀>·tanh≤<cap>)`,
    /// e.g. `ReZero(0.447·tanh≤0.447)` for a legacy 1/√5 group or
    /// `ReZero(0·tanh≤1)` for a zero-init group with cap 1, or `no-ReZero`.
    /// The `tanh≤` value is the forward soft-bound asymptote
    /// (`BlockGroup.rezeroTanhCeiling`). Always both numbers, so a group whose
    /// cap equals its init (every architecture that predates the explicit
    /// cap) renders byte-identically to the pre-field form. Shared by
    /// `groupSummary` and the Build screen's diagram so the two never drift.
    static func rezeroDescription(_ g: BlockGroup) -> String {
        guard g.useRezero else { return "no-ReZero" }
        return "ReZero(\(String(format: "%.3g", g.rezeroAlphaInit))·tanh≤\(String(format: "%.3g", g.rezeroTanhCeiling)))"
    }

    /// The SE-activation clause of a group's rendering, e.g.
    /// ` (fc1 leaky_relu)`: present only when the group has an SE block whose
    /// FC1 activation differs from the group's main-path activation. Every
    /// architecture that predates `seActivation` has them equal, so its
    /// summary is byte-identical to the pre-field form. Shared by
    /// `groupSummary` and the Build screen's diagram so the two never drift.
    static func seActivationMarker(_ g: BlockGroup) -> String {
        guard g.seStyle != .none, g.seActivation != g.activationFunction else { return "" }
        return " (fc1 \(g.seActivation.rawValue))"
    }

    // MARK: Weight tensor plan

    /// Ordered (name, shape, kind) for every persistent tensor in the exact order
    /// `ChessNetwork.exportWeights()` emits and `loadWeights()` expects: **all
    /// trainables in build order, then all BN running stats in build order**. Names
    /// are PyTorch-ready module paths. Branches on every topology axis.
    func weightTensorPlan() -> [WeightTensorSpec] {
        var trainables: [WeightTensorSpec] = []
        var running: [WeightTensorSpec] = []

        func bn(_ prefix: String, _ ch: Int) {
            trainables.append(.init(name: "\(prefix).weight", shape: [ch], kind: .bnAffine))
            trainables.append(.init(name: "\(prefix).bias", shape: [ch], kind: .bnAffine))
            running.append(.init(name: "\(prefix).running_mean", shape: [ch], kind: .bnRunningStat))
            running.append(.init(name: "\(prefix).running_var", shape: [ch], kind: .bnRunningStat))
        }
        func se(_ prefix: String, _ spec: BlockGroup) {
            let outC = spec.channels
            let seReduced = spec.seStyle == .none ? 0 : outC / spec.seReductionRatio
            switch spec.seStyle {
            case .none:
                break
            case .attenuateOnly:
                trainables.append(.init(name: "\(prefix).se_attenuate.fc1.weight", shape: [outC, seReduced], kind: .linear))
                trainables.append(.init(name: "\(prefix).se_attenuate.fc1.bias", shape: [seReduced], kind: .bias))
                trainables.append(.init(name: "\(prefix).se_attenuate.fc2.weight", shape: [seReduced, outC], kind: .linear))
                trainables.append(.init(name: "\(prefix).se_attenuate.fc2.bias", shape: [outC], kind: .bias))
            case .scaleAndBias:
                trainables.append(.init(name: "\(prefix).se_scalebias.fc1.weight", shape: [outC, seReduced], kind: .linear))
                trainables.append(.init(name: "\(prefix).se_scalebias.fc1.bias", shape: [seReduced], kind: .bias))
                trainables.append(.init(name: "\(prefix).se_scalebias.fc2.weight", shape: [seReduced, 2 * outC], kind: .linear))
                trainables.append(.init(name: "\(prefix).se_scalebias.fc2.bias", shape: [2 * outC], kind: .bias))
            }
        }

        // Stem: conv -> BN.
        let c0 = stemOutputChannels
        trainables.append(.init(name: "stem.conv.weight", shape: [c0, inputPlanes, stemConvKernelSize, stemConvKernelSize], kind: .conv))
        bn("stem.bn", c0)

        // Tower: thread the incoming width through the expanded blocks. The
        // per-block tensor order mirrors `ChessNetwork.residualBlock`'s
        // trainables append order EXACTLY (the builder is the other half of
        // this contract): pre = bn1, conv1, bn2, conv2, SE, [rezero]; post =
        // conv1, bn1, conv2, bn2, SE, [rezero]; the skip projection — present
        // only on width transitions — appends LAST within its block. Uniform
        // towers therefore keep today's exact layout.
        var inC = c0
        for (i, spec) in expandedBlocks.enumerated() {
            // `inCEff` folds in the final-block feature skip (+ source on the last
            // block under a routed concatDirect skip); == inC everywhere else, so the
            // conv1/bn1/skip-proj shapes are unchanged for non-finalBlock configs.
            let inCEff = inC + blockSkipExtraInputChannels(blockIndex: i)
            let p = "blocks.\(i)"
            let outC = spec.channels
            let conv1 = WeightTensorSpec(name: "\(p).conv1.weight", shape: [outC, inCEff, spec.conv1KernelSize, spec.conv1KernelSize], kind: .conv)
            let conv2 = WeightTensorSpec(name: "\(p).conv2.weight", shape: [outC, outC, spec.conv2KernelSize, spec.conv2KernelSize], kind: .conv)
            switch spec.activationStyle {
            case .pre:
                // BN1 -> act -> conv1 -> BN2 -> act -> conv2 -> SE -> [rezero]
                bn("\(p).bn1", inCEff)
                trainables.append(conv1)
                bn("\(p).bn2", outC)
                trainables.append(conv2)
                se(p, spec)
            case .post:
                // conv1 -> BN1 -> act -> conv2 -> BN2 -> SE -> (act on merged sum)
                trainables.append(conv1)
                bn("\(p).bn1", outC)
                trainables.append(conv2)
                bn("\(p).bn2", outC)
                se(p, spec)
            }
            if spec.useRezero {
                trainables.append(.init(name: "\(p).rezero_alpha", shape: [1], kind: .scalar))
            }
            if inCEff != outC {
                trainables.append(.init(name: "\(p).skip_proj.weight", shape: [outC, inCEff, 1, 1], kind: .conv))
            }
            // Output LayerNorm γ/β append LAST within the block — the builder
            // applies the norm to the merged output AFTER the skip projection,
            // so this mirrors `ChessNetwork.residualBlock`'s append order. No
            // running stats (LayerNorm computes its stats per-forward).
            if spec.resolvedOutputNorm == .layerNorm {
                trainables.append(.init(name: "\(p).res_ln.weight", shape: [outC], kind: .bnAffine))
                trainables.append(.init(name: "\(p).res_ln.bias", shape: [outC], kind: .bnAffine))
            }
            inC = outC
        }

        // Tower-end normalization (pre-activation tail only).
        if hasTowerEndBN { bn("tower_final_bn", towerOutputChannels) }

        // Compress fusion node (head-only): inserted between the tower-end BN and the
        // heads, mirroring the graph builder's append position so the index-aligned
        // export/load contract holds. concatDirect adds no entry here.
        if featureSkipUsesCompressNode {
            trainables.append(.init(
                name: "feature_skip.conv.weight",
                shape: [towerOutputChannels, featureSkipCompressInputChannels, 1, 1], kind: .conv))
            bn("feature_skip.bn", towerOutputChannels)
        }

        // Heads read the tower-output width — wider by the feature-skip source when a
        // routed `concatDirect` skip feeds that head (FIRST conv only). Same formula
        // as `parameterCount` and the graph builder.
        let cP = policyHeadInputChannels
        let cVin = valueHeadInputChannels

        // Policy head.
        let pK = policyPreConvChannels
        switch policyHeadStyle {
        case .simpleConv:
            trainables.append(.init(name: "policy.conv.weight", shape: [policyChannels, cP, 1, 1], kind: .conv))
            trainables.append(.init(name: "policy.conv.bias", shape: [1, policyChannels, 1, 1], kind: .bias))
        case .intermediateConv:
            trainables.append(.init(name: "policy.pre_conv.weight", shape: [pK, cP, 1, 1], kind: .conv))
            bn("policy.pre_bn", pK)
            trainables.append(.init(name: "policy.conv.weight", shape: [policyChannels, pK, 1, 1], kind: .conv))
            trainables.append(.init(name: "policy.conv.bias", shape: [1, policyChannels, 1, 1], kind: .bias))
        case .fcBottleneck:
            trainables.append(.init(name: "policy.pre_conv.weight", shape: [pK, cP, 1, 1], kind: .conv))
            bn("policy.pre_bn", pK)
            let flat = pK * boardSize * boardSize
            trainables.append(.init(name: "policy.fc.weight", shape: [flat, policySize], kind: .linear))
            trainables.append(.init(name: "policy.fc.bias", shape: [1, policySize], kind: .bias))
        }

        // Value head.
        let cv = valueHeadConvChannels
        let h = valueHeadHiddenUnits
        trainables.append(.init(name: "value.conv.weight", shape: [cv, cVin, 1, 1], kind: .conv))
        bn("value.bn", cv)
        let flatV = boardSize * boardSize * cv
        trainables.append(.init(name: "value.fc1.weight", shape: [flatV, h], kind: .linear))
        trainables.append(.init(name: "value.fc1.bias", shape: [1, h], kind: .bias))
        let fc2Name = valueHeadStyle == .wdlSoftmax ? "value.wdl_fc2" : "value.scalar_fc2"
        trainables.append(.init(name: "\(fc2Name).weight", shape: [h, valueHeadClasses], kind: .linear))
        trainables.append(.init(name: "\(fc2Name).bias", shape: [1, valueHeadClasses], kind: .bias))

        return trainables + running
    }

    /// The trainable prefix of `weightTensorPlan()` (everything but the BN
    /// running statistics), in trainable order — the order the optimizer's
    /// per-trainable velocity tensors follow.
    func trainableTensorPlan() -> [WeightTensorSpec] {
        weightTensorPlan().filter { $0.kind != .bnRunningStat }
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
        case v4_4block_3x3_fp32   // fp32 4-block 3x3 — beta-stack stable-precision run (2026-06-14)
        case v4_5block_7x7_fusion // current preset + feature skip (stem -> both heads, concat_direct)
        case v5_5block_7x7_lnout  // v4_5block_7x7 recipe + LayerNorm on each block's output
        case nt8y_3x3stem         // exactly nt8y (3× @32 15×15 fat-conv, LN-out) but a 3×3 stem (nt8y's was 5×5)
        case nt8y_15x15stem       // exactly nt8y but a 15×15 stem (matches the block conv kernel; nt8y's was 5×5)

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
        case .v4_4block_3x3_fp32:
            // Normal v4 3x3 recipe (matches v4_8block_3x3) but 4 blocks and
            // FLOAT32 — a stable-precision run while the Xcode/macOS 27 beta
            // bf16 training stomp is unresolved.
            return NetworkArchitecture(
                inputEncoding: .basic30, channels: 128, numBlocks: 4, stemConvKernelSize: 3,
                activationFunction: .relu, blockActivationStyle: .pre,
                blockSkipMerge: .cleanAdd, blockUseRezero: true,
                rezeroAlphaInit: 1.0 / Float(4).squareRoot(),
                blockConv1KernelSize: 3, blockConv2KernelSize: 3,
                blockSeStyle: .scaleAndBias, blockSeReductionRatio: 4,
                policyHeadStyle: .intermediateConv, policyPreConvChannels: 128,
                valueHeadStyle: .wdlSoftmax, valueHeadConvChannels: 16, valueHeadHiddenUnits: 128,
                computeDataType: .float32
            )
        case .v4_5block_7x7_fusion:
            // The current preset with the feature skip enabled: stem output routed
            // into BOTH heads via concat_direct. Authored by mutating the base preset
            // so it tracks any future change to v4_5block_7x7's recipe.
            var a = NetworkArchitecture.preset(.v4_5block_7x7)
            a.featureSkipSource = .stemOutput
            a.featureSkipFusion = .concatDirect
            a.featureSkipToPolicyHead = true
            a.featureSkipToValueHead = true
            a.featureSkipToFinalBlock = false
            return a
        case .v5_5block_7x7_lnout:
            // The current v4 preset, byte-for-byte, plus a channel-wise LayerNorm
            // on each block's output (clean-add highway re-centered every block,
            // ReZero retained). Authored by mutating the base preset so it tracks
            // any future change to v4_5block_7x7's recipe. Adds 2·128 params/block
            // (γ/β) over v4 — 8,445,748 → 8,447,028.
            var a = NetworkArchitecture.preset(.v4_5block_7x7)
            a.blockGroups = a.blockGroups.map { group in
                var g = group
                g.outputNorm = .layerNorm
                return g
            }
            return a
        case .nt8y_3x3stem:
            // Exactly the nt8y architecture (GUI-built 20260701, 1,533,930 params:
            // 3× @32 15×15+15×15 fat-conv, SE scale+bias/4, ReLU/pre, ReZero cap
            // 1/3, LayerNorm-out, policy intermediate_conv pre-conv 512, value WDL
            // 16→FC64, bf16) — but with a 3×3 stem instead of nt8y's 5×5. Only the
            // stem kernel changes; every other field is copied from nt8y's embedded
            // architecture JSON. LayerNorm-out isn't expressible in the flat
            // initializer, so it's patched onto the block group afterward (same
            // technique as v5_5block_7x7_lnout).
            var a = NetworkArchitecture(
                inputEncoding: .basic30, channels: 32, numBlocks: 3, stemConvKernelSize: 3,
                activationFunction: .relu, blockActivationStyle: .pre,
                blockSkipMerge: .cleanAdd, blockUseRezero: true, rezeroAlphaInit: 1.0 / Float(3),
                blockConv1KernelSize: 15, blockConv2KernelSize: 15,
                blockSeStyle: .scaleAndBias, blockSeReductionRatio: 4,
                policyHeadStyle: .intermediateConv, policyPreConvChannels: 512,
                valueHeadStyle: .wdlSoftmax, valueHeadConvChannels: 16, valueHeadHiddenUnits: 64,
                computeDataType: .bFloat16
            )
            a.blockGroups = a.blockGroups.map { group in
                var g = group
                g.outputNorm = .layerNorm
                return g
            }
            return a
        case .nt8y_15x15stem:
            // Exactly the nt8y architecture (as in nt8y_3x3stem) but with a 15×15
            // stem — matching the block conv kernel — instead of nt8y's 5×5. Only
            // the stem kernel changes; every other field is copied verbatim.
            var a = NetworkArchitecture(
                inputEncoding: .basic30, channels: 32, numBlocks: 3, stemConvKernelSize: 15,
                activationFunction: .relu, blockActivationStyle: .pre,
                blockSkipMerge: .cleanAdd, blockUseRezero: true, rezeroAlphaInit: 1.0 / Float(3),
                blockConv1KernelSize: 15, blockConv2KernelSize: 15,
                blockSeStyle: .scaleAndBias, blockSeReductionRatio: 4,
                policyHeadStyle: .intermediateConv, policyPreConvChannels: 512,
                valueHeadStyle: .wdlSoftmax, valueHeadConvChannels: 16, valueHeadHiddenUnits: 64,
                computeDataType: .bFloat16
            )
            a.blockGroups = a.blockGroups.map { group in
                var g = group
                g.outputNorm = .layerNorm
                return g
            }
            return a
        }
    }

    /// The architecture the current build defaults to.
    static var current: NetworkArchitecture { preset(.current) }

    /// What a newly built model starts from (the Build-New-Model screen and
    /// the session's Build Network): the current preset with the `basic24`
    /// input, which drops the six repetition planes that can never fire.
    /// Presets keep their own encoding, so every existing model and every
    /// preset still describes exactly what it was trained with; only new
    /// builds move to the smaller input.
    static var newModelDefault: NetworkArchitecture {
        var architecture = current
        architecture.inputEncoding = .basic24
        return architecture
    }

    /// Legacy `.dcmmodel` archHash (the old FNV value stored at byte offset 12:
    /// six shape scalars for v3 files, plus the architecture version for v4)
    /// -> the historical preset to rebuild. The ONLY backward-compat
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
