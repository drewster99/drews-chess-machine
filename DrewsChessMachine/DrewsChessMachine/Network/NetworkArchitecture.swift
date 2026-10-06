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

/// Hidden-activation function, or the `does_not_apply` marker. Chosen per block
/// group (`BlockGroup.activationFunction`: block main path, `activation_gated`
/// merge; `BlockGroup.seActivation`: SE FC1) and once per architecture-level site
/// (`NetworkArchitecture.stemActivation`, `towerEndActivation`,
/// `featureSkipActivation`, `policyHeadActivation`, `valueHeadConvActivation`,
/// `valueHeadFC1HiddenActivation`; see `ArchitectureActivationSite`). Verified
/// across all of git history: every architecture before SiLU/GELU were added used
/// ReLU at every hidden site, so `.relu` reproduces all historical nets. The SE
/// gate (`sigmoid`) and the value output (`tanh` for `scalar_tanh`, `softmax` for
/// `wdl_softmax`) are structural and NOT governed by this.
///
/// `does_not_apply` is not a function. It marks an architecture-level site the
/// model's topology does not have (a pre-activation tower's stem, a simple_conv
/// policy head's pre-block, …), and never means identity or linear. It is legal
/// only in the six architecture-level site fields, and there exactly when the
/// site does not exist (`NetworkArchitecture.activationSiteMismatch`); a block
/// group's fields never accept it. Every list a person or a CLI chooses a
/// function from is `functions`, never `allCases`.
enum ActivationFunction: String, Codable, CaseIterable, Sendable, Hashable {
    case relu
    case silu
    case gelu
    /// `x` for `x ≥ 0`, `leakyReLUNegativeSlope · x` below. Keeps a small
    /// gradient where ReLU's is exactly zero, so a unit pushed negative for
    /// every input (a dead ReLU unit, as measured in the SE bottlenecks) can
    /// still recover.
    case leakyRelu = "leaky_relu"
    /// The marker an architecture-level site field holds when the topology
    /// lacks the site. Not a function (see the type's doc).
    case doesNotApply = "does_not_apply"

    /// Every case that names a function: the list every picker, every derive
    /// value syntax and every per-function test iterates. `allCases` minus
    /// `does_not_apply`, so a function added later joins it automatically
    /// while the marker never does.
    static let functions: [ActivationFunction] = allCases.filter { $0 != .doesNotApply }

    /// `functions` as the `relu, silu, …` text error messages offer.
    static var functionList: String {
        functions.map(\.rawValue).joined(separator: ", ")
    }

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

// MARK: - Init-neutral options (determinism plan B2, decision D-4)
//
// "Init-neutral" = the last layer of a path that is ADDED into a residual
// stream or an output, initialized so the path contributes nothing (or a
// known prior) at step 0 while still receiving gradient. Each option below
// changes only the values a fresh build gives a tensor, never a tensor's
// shape, and has a standard value equal to what every model was built with
// before the option existed. Paths where zero would destroy the signal (the
// post-merge LayerNorm, the stem, the feature-skip fusion conv, a zeroed skip
// projection) are deliberately not offered.

/// The init of the LAST BatchNorm γ of a block's residual branch.
enum BranchOutputInit: String, Codable, CaseIterable, Sendable, Hashable {
    /// γ = 1, like every BN (the behavior before this option existed).
    case standard
    /// γ = 0 on the BN that follows the branch's last conv (the "zero-init
    /// last BN γ" of Goyal et al.): the branch adds exactly nothing at step 0,
    /// so the block starts as its skip path; γ still receives gradient (its
    /// normalized input is nonzero), and every conv keeps its random init.
    /// Only a post-activation block has a BN after its last conv, so
    /// `validate()` refuses it on a pre-activation group.
    case zeroLastBNGamma = "zero_last_bn_gamma"
}

/// The init of a width-transition skip projection (the bias-free 1×1 conv a
/// block gets where its input and output widths differ).
enum SkipProjectionInit: String, Codable, CaseIterable, Sendable, Hashable {
    /// He-normal, like every conv (the behavior before this option existed).
    case he
    /// A partial identity: weight 1 from input channel `i` to output channel
    /// `i` for every `i` both widths have, 0 everywhere else — the projection
    /// starts by passing the shared channels straight through and zero-filling
    /// the new ones. It is the block's identity path, so it is never zero.
    case identityLike = "identity_like"
}

/// The init of a head's final projection (the layer that produces the
/// head's logits).
enum HeadFinalInit: String, Codable, CaseIterable, Sendable, Hashable {
    /// He-normal (the behavior before this option existed).
    case he
    /// Exactly zero: the head's output is its bias at step 0 — uniform policy
    /// logits, or the value head's W/D/L prior. The final layer still gets
    /// gradient (its input is nonzero); the earlier head layers start moving
    /// one step later.
    case zero
}

/// One init-neutral option of one architecture, named for the Build screen's
/// highlight, the summary and the tests' "differs from standard" set. Block
/// options carry their 0-based group index.
enum InitOptionField: Hashable, Sendable {
    case seGammaBiasInit(group: Int)
    case branchOutputInit(group: Int)
    case skipProjectionInit(group: Int)
    case policyHeadFinalInit
    case valueHeadFinalInit
    case valueHeadDrawPrior

    /// The architecture JSON key.
    var jsonKey: String {
        switch self {
        case .seGammaBiasInit: return BlockGroup.CodingKeys.seGammaBiasInit.rawValue
        case .branchOutputInit: return BlockGroup.CodingKeys.branchOutputInit.rawValue
        case .skipProjectionInit: return BlockGroup.CodingKeys.skipProjectionInit.rawValue
        case .policyHeadFinalInit: return NetworkArchitecture.CodingKeys.policyHeadFinalInit.rawValue
        case .valueHeadFinalInit: return NetworkArchitecture.CodingKeys.valueHeadFinalInit.rawValue
        case .valueHeadDrawPrior: return NetworkArchitecture.CodingKeys.valueHeadDrawPrior.rawValue
        }
    }

    /// What a non-standard value of this option changes at step 0 — the
    /// Build screen's tooltip.
    var stepZeroEffect: String {
        switch self {
        case .seGammaBiasInit:
            return "The SE gate bias starts at this level: at step 0 every SE gate is about sigmoid(bias) "
                + "(0 → 0.5 halves the branch; a large bias passes it nearly unchanged). It still learns."
        case .branchOutputInit:
            return "The last BN γ of each branch starts at 0: the branch adds nothing at step 0, "
                + "so each block starts as its skip path. γ still learns; every conv keeps its random init."
        case .skipProjectionInit:
            return "The width-transition 1×1 projection starts as a partial identity: shared channels pass "
                + "straight through, new channels start at zero. It still learns."
        case .policyHeadFinalInit:
            return "The final policy layer starts at zero: the policy is uniform over every move at step 0. "
                + "The layer still learns."
        case .valueHeadFinalInit:
            return "The final value layer starts at zero: at step 0 the value head outputs its W/D/L prior "
                + "for every position. The layer still learns."
        case .valueHeadDrawPrior:
            return "The value head's initial draw probability: its output bias is set so the W/D/L softmax "
                + "of zero logits is (½(1−p), p, ½(1−p))."
        }
    }
}

/// One architecture-level activation site: an activation the graph builds
/// outside the block groups, whose function the architecture states in its own
/// field. In graph build order. Each site exists only for some topologies
/// (`NetworkArchitecture.hasActivationSite`); where it does not, its field
/// holds `does_not_apply`.
enum ArchitectureActivationSite: CaseIterable, Hashable, Sendable {
    /// `stem_act`: after the stem BN, when the first block group is
    /// post-activation.
    case stem
    /// `tower_final_act`: after the tower-end BN, when the last block group
    /// is pre-activation.
    case towerEnd
    /// `feature_skip_act`: after the compress fusion node's BN, when that
    /// node is built.
    case featureSkipFusion
    /// `policy_pre_act`: the policy pre-block's activation
    /// (`intermediate_conv`, `fc_bottleneck`).
    case policyHead
    /// `value_act`: after the value head's conv BN. Always exists.
    case valueHeadConv
    /// `value_fc1_act`: the value head's FC1 hidden layer. Always exists.
    case valueHeadFC1Hidden

    /// The architecture JSON key, read from `NetworkArchitecture.CodingKeys`
    /// (the single source of every key).
    var codingKey: NetworkArchitecture.CodingKeys {
        switch self {
        case .stem: return .stemActivation
        case .towerEnd: return .towerEndActivation
        case .featureSkipFusion: return .featureSkipActivation
        case .policyHead: return .policyHeadActivation
        case .valueHeadConv: return .valueHeadConvActivation
        case .valueHeadFC1Hidden: return .valueHeadFC1HiddenActivation
        }
    }

    var jsonKey: String { codingKey.rawValue }

    /// The site's name in `architectureSummary`'s per-site activation clause.
    var summaryLabel: String {
        switch self {
        case .stem: return "stem"
        case .towerEnd: return "tower_end"
        case .featureSkipFusion: return "fusion"
        case .policyHead: return "policy"
        case .valueHeadConv: return "value_conv"
        case .valueHeadFC1Hidden: return "value_fc1_hidden"
        }
    }

    /// The Build New Model screen's picker label.
    var displayName: String {
        switch self {
        case .stem: return "Stem activation"
        case .towerEnd: return "Tower-end activation"
        case .featureSkipFusion: return "Fusion activation"
        case .policyHead: return "Pre-block activation"
        case .valueHeadConv: return "Conv activation"
        case .valueHeadFC1Hidden: return "FC1 hidden activation"
        }
    }

    /// What the site's activation is applied to — the Build screen's help
    /// text for a site that exists.
    var siteDescription: String {
        switch self {
        case .stem:
            return "Applied to the stem BN's output (a post-activation first block group only)."
        case .towerEnd:
            return "Applied to the tower-end BN's output, which every head reads (a pre-activation last block group only)."
        case .featureSkipFusion:
            return "Applied to the compress fusion node's BN output (compress_conv_bn_relu routed to a head only)."
        case .policyHead:
            return "Applied to the policy pre-block's BN output (intermediate_conv and fc_bottleneck only)."
        case .valueHeadConv:
            return "Applied to the value head's conv BN output."
        case .valueHeadFC1Hidden:
            return "Applied to the value head's FC1 hidden layer (value_head_hidden_units wide)."
        }
    }

    /// Why the site is missing when the topology lacks it — for errors, the
    /// legacy log and the Build screen's help text. The two value sites exist
    /// in every model, so theirs only says so.
    var absentReason: String {
        switch self {
        case .stem:
            return "the first block group is pre-activation, so the stem has no activation"
        case .towerEnd:
            return "the last block group is post-activation, so the tower has no tower-end activation"
        case .featureSkipFusion:
            return "no compress fusion node is built (that needs a feature-skip source, "
                + "\(FeatureSkipFusion.compressConvBNReLU.rawValue) fusion and a routed head)"
        case .policyHead:
            return "the policy head is \(PolicyHeadStyle.simpleConv.rawValue), so it has no pre-block"
        case .valueHeadConv, .valueHeadFC1Hidden:
            return "never: every value head has this layer"
        }
    }
}

/// An architecture-level site whose activation field disagrees with whether
/// the site exists: `does_not_apply` at a site the topology has, or a function
/// at one it lacks. Carries the one message every error built from it uses
/// (`NetworkArchitectureError.activationSiteMismatch`,
/// `ArchitectureFormat.FormatError.activationSiteMismatch`).
struct ActivationSiteMismatch: Equatable, Sendable, CustomStringConvertible {
    let site: ArchitectureActivationSite
    let value: ActivationFunction
    let siteExists: Bool
    /// Why the site exists or not in this architecture
    /// (`NetworkArchitecture.activationSiteReason`).
    let reason: String

    var description: String {
        if siteExists {
            return "\(site.jsonKey) is '\(value.rawValue)', but \(reason): "
                + "choose one of \(ActivationFunction.functionList)"
        }
        return "\(site.jsonKey) is '\(value.rawValue)', but \(reason): "
            + "it must be '\(ActivationFunction.doesNotApply.rawValue)'"
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
    /// The group's SE style. Removing the SE block (`.none`) removes its FC1,
    /// so `seActivation` becomes `does_not_apply` in the same assignment —
    /// every path that edits the style (the Build screen, derive, tests) gets
    /// the one consistent value without a separate step. Adding an SE block
    /// leaves `seActivation` as it is: `does_not_apply` when the group had no
    /// SE block, which `validate()` names until a function is chosen, so an
    /// FC1 activation is never invented. (Assignments in the initializers do
    /// not run this observer.)
    var seStyle: SEStyle {
        didSet {
            if seStyle == .none {
                seActivation = .doesNotApply
            }
        }
    }
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
    /// tensor. A group without an SE block has no FC1, so its value is
    /// `does_not_apply` there and only there — the rule of the
    /// architecture-level sites (`ArchitectureActivationSite`), applied to the
    /// group (`validate()`, decode; owner decision OD-13). Decoding is
    /// format-version gated (`ArchitectureFormat`): files before format v5
    /// resolve a missing value to the group's `activationFunction` (what the
    /// SE FC1 used before the field existed) on an SE group and to
    /// `does_not_apply` on an SE-less one; v5+ files must state it; and a file
    /// before v10 whose SE-less group states the group's own activation (the
    /// value `validate()` forced there then, never applied) resolves it to
    /// `does_not_apply`.
    var seActivation: ActivationFunction
    /// The value every element of the γ half of this group's SE FC2 bias
    /// starts at (both SE styles; `attenuate_only`'s whole FC2 bias is its γ).
    /// The standard value, 0, gives every SE gate `sigmoid(0) = 0.5` at init;
    /// a positive level starts the SE nearly transparent (the neutral value,
    /// `neutralSEGammaBiasInit`, gives 0.9). A constant, never a draw. On an
    /// SE-less group it has no tensor, so `validate()` requires the standard
    /// value there. Format-version gated (`ArchitectureFormat`): files older
    /// than v8 resolve a missing value to the standard one.
    var seGammaBiasInit: Float
    /// The init of the last BN γ of this group's residual branches (see
    /// `BranchOutputInit`). Format-version gated like `seGammaBiasInit`.
    var branchOutputInit: BranchOutputInit
    /// The init of this group's width-transition skip projection (see
    /// `SkipProjectionInit`); meaningful only where the group's first block
    /// changes width, and `validate()` requires the standard value elsewhere.
    /// Format-version gated like `seGammaBiasInit`.
    var skipProjectionInit: SkipProjectionInit

    /// The SE γ-bias level every group had before `seGammaBiasInit` existed.
    static let standardSEGammaBiasInit: Float = 0
    /// The "near-identity SE" level the Neutral set uses: `ln 9`, so the gate
    /// starts at exactly `sigmoid(ln 9) = 0.9` and the SE passes its branch
    /// almost unchanged at step 0.
    static let neutralSEGammaBiasInit = Float(log(9.0))

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

    /// Set the group's main-path activation. `seActivation` is never touched:
    /// on a group with an SE block it changes only when set on its own, and
    /// on an SE-less group it is `does_not_apply` whatever the main path uses.
    /// The single rule the Build-New-Model screen and
    /// `--derive-model --set-activation` both apply, so the same edit made
    /// either way yields the same architecture.
    ///
    /// `does_not_apply` is a defect here, not an input: a group's main path
    /// exists whenever the group does. It cannot throw (its caller is the
    /// non-throwing `BlockGroupDraft.activationFunction` setter), and every
    /// caller passes a function — the group picker lists
    /// `ActivationFunction.functions`, and
    /// `NetworkArchitecture.setMainActivationEverywhere` refuses the marker
    /// before it calls this.
    mutating func setActivationFunction(_ activation: ActivationFunction) {
        precondition(activation != .doesNotApply,
                     "BlockGroup.setActivationFunction: 'does_not_apply' is not an activation function")
        activationFunction = activation
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
        case seGammaBiasInit = "se_gamma_bias_init"
        case branchOutputInit = "branch_output_init"
        case skipProjectionInit = "skip_projection_init"
    }

    /// Full memberwise init (spelled out because the custom `Codable` below
    /// suppresses the synthesized one): every field, the ReZero cap and the
    /// SE FC1 activation included. `outputNorm` and `seBetaInit` keep the
    /// defaults the synthesized init had: both are the behavior every group
    /// had before the field existed. The init-neutral options are required
    /// here with no default. The overloads below omit the cap (and optionally
    /// the SE activation) and spell the arrangements that existed before
    /// those fields did.
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
        seActivation: ActivationFunction,
        seGammaBiasInit: Float,
        branchOutputInit: BranchOutputInit,
        skipProjectionInit: SkipProjectionInit
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
        self.seGammaBiasInit = seGammaBiasInit
        self.branchOutputInit = branchOutputInit
        self.skipProjectionInit = skipProjectionInit
    }

    /// A group whose ReZero cap is derived from its init
    /// (`legacyRezeroAlphaCap`: `rezeroAlphaInit ×
    /// NetworkArchitecture.rezeroTanhCeilingMultiple`) — the only arrangement
    /// that existed before `rezeroAlphaCap` did, so every recipe written
    /// before it (code presets, tests) keeps its meaning and builds the same
    /// graph. `seActivation` is required here; the overload below, which also
    /// omits it, is the "SE FC1 shares the group's activation" spelling. A
    /// zero-init group needs the full init: its derived cap would be zero,
    /// which `validate()` rejects. Those recipes also predate the init-neutral
    /// options, so this spelling states the standard value of each; a group
    /// with a non-standard one sets it on the returned value.
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
            seActivation: seActivation,
            seGammaBiasInit: Self.standardSEGammaBiasInit,
            branchOutputInit: .standard,
            skipProjectionInit: .he)
    }

    /// A group whose SE FC1 uses the group's own `activationFunction` (and
    /// whose `seActivation` is `does_not_apply` when it has no SE block) and
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
            seActivation: seStyle == .none ? .doesNotApply : activationFunction)
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
    /// `rezero_alpha_init`), and the init-neutral options (`se_gamma_bias_init`,
    /// `branch_output_init`, `skip_projection_init`) from
    /// `ArchitectureFormat.initOptionsRequiredFromVersion` (older files resolve
    /// each to its standard value). Every resolution is recorded on `format`'s
    /// log.
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
        guard activationFunction != .doesNotApply else {
            throw ArchitectureFormat.FormatError.doesNotApplyAtAnAlwaysPresentSite(
                field: CodingKeys.activationFunction.rawValue,
                location: ArchitectureFormat.location(of: decoder),
                formatVersion: format.formatVersion,
                source: format.source)
        }
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
            seActivation = try Self.decodedSEActivation(
                stated: stated, seStyle: seStyle, activationFunction: activationFunction,
                decoder: decoder, format: format)
        } else if format.allowsMissingSEActivation {
            if seStyle == .none {
                seActivation = .doesNotApply
                format.legacyLog.record(
                    "\(ArchitectureFormat.location(of: decoder)).\(CodingKeys.seActivation.rawValue) := "
                        + "\(ActivationFunction.doesNotApply.rawValue) (\(Self.seLessReason))")
            } else {
                seActivation = activationFunction
                format.legacyLog.record(
                    "\(ArchitectureFormat.location(of: decoder)).\(CodingKeys.seActivation.rawValue) := \(activationFunction.rawValue) "
                        + "(the group's \(CodingKeys.activationFunction.rawValue))")
            }
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
        seGammaBiasInit = try ArchitectureFormat.decodeInitOption(
            Float.self, key: CodingKeys.seGammaBiasInit, in: c, decoder: decoder, format: format,
            legacyByConstruction: false,
            standard: Self.standardSEGammaBiasInit, rendered: { "\($0)" })
        branchOutputInit = try ArchitectureFormat.decodeInitOption(
            BranchOutputInit.self, key: CodingKeys.branchOutputInit, in: c, decoder: decoder, format: format,
            legacyByConstruction: false,
            standard: .standard, rendered: \.rawValue)
        skipProjectionInit = try ArchitectureFormat.decodeInitOption(
            SkipProjectionInit.self, key: CodingKeys.skipProjectionInit, in: c, decoder: decoder, format: format,
            legacyByConstruction: false,
            standard: .he, rendered: \.rawValue)
    }

    /// Why an SE-less group's `seActivation` is `does_not_apply` — for the
    /// legacy log and errors.
    static let seLessReason = "the group has no SE block, so it has no SE FC1"

    /// The second half of every `se_activation` mismatch message: why the
    /// value is wrong for a group of `seStyle`, and what it must be.
    static func seActivationRule(seStyle: SEStyle) -> String {
        if seStyle == .none {
            return "\(seLessReason) (se_style '\(SEStyle.none.rawValue)'): it must be "
                + "'\(ActivationFunction.doesNotApply.rawValue)'"
        }
        return "the group has an SE block (se_style '\(seStyle.rawValue)'): choose one of \(ActivationFunction.functionList)"
    }

    /// A stated `se_activation`, checked against the group's SE style: a
    /// function on a group with an SE block, `does_not_apply` on one without.
    /// A file older than v10 whose SE-less group states the group's own
    /// activation — the value `validate()` required there before the field
    /// could say `does_not_apply`, never applied by any graph — resolves to
    /// `does_not_apply` (logged). Any other disagreement is refused, never
    /// repaired.
    private static func decodedSEActivation(
        stated: ActivationFunction,
        seStyle: SEStyle,
        activationFunction: ActivationFunction,
        decoder: Decoder,
        format: ArchitectureFormat.DecodeFormat
    ) throws -> ActivationFunction {
        let hasSE = seStyle != .none
        if (stated != .doesNotApply) == hasSE {
            return stated
        }
        if !hasSE, format.allowsLegacySELessSEActivation, stated == activationFunction {
            format.legacyLog.record(
                "\(ArchitectureFormat.location(of: decoder)).\(CodingKeys.seActivation.rawValue) := "
                    + "\(ActivationFunction.doesNotApply.rawValue) (\(seLessReason); the file's value "
                    + "\(stated.rawValue) was its activation_function)")
            return .doesNotApply
        }
        throw ArchitectureFormat.FormatError.seActivationMismatch(
            seStyle: seStyle,
            seActivation: stated,
            location: ArchitectureFormat.location(of: decoder),
            formatVersion: format.formatVersion,
            source: format.source)
    }

    /// Writes every field. `output_norm` keeps its pre-existing
    /// write-only-when-set form so older fields encode byte-identically;
    /// `se_beta_init`, `se_activation`, `rezero_alpha_cap` and the
    /// init-neutral options are ALWAYS written (even when they equal their
    /// legacy resolution, and on groups without ReZero or SE), so a
    /// current-version file is self-describing.
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
        try c.encode(seGammaBiasInit, forKey: .seGammaBiasInit)
        try c.encode(branchOutputInit, forKey: .branchOutputInit)
        try c.encode(skipProjectionInit, forKey: .skipProjectionInit)
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
    /// A Float field that must be finite (any sign), e.g. the SE γ-bias init.
    case mustBeFinite(field: String, value: Float)
    /// An init-neutral option set to a non-standard value where its layer
    /// does not exist (or, for the draw prior, outside its valid range).
    /// `reason` says which.
    case initOptionWithoutItsLayer(field: String, value: String, reason: String)
    /// `se_beta_init` other than `glorot` on a group whose SE style has no
    /// β half (only `scale_and_bias` does).
    case seBetaInitRequiresScaleAndBias(group: Int, seStyle: SEStyle, seBetaInit: SEBetaInit)
    /// A group's `se_activation` disagrees with whether it has an SE block:
    /// a function on an SE-less group (no FC1 to apply it to), or
    /// `does_not_apply` on a group with one.
    case seActivationMismatch(group: Int, seStyle: SEStyle, seActivation: ActivationFunction)
    /// Feature skip is enabled (`source != .none`) but no destination is routed.
    case featureSkipNoDestination
    /// A feature-skip combination that is config-carried but unsupported —
    /// currently only `compressConvBNReLU` fusion together with the
    /// `toFinalBlock` destination. Every other feature-skip option (head
    /// fusion in either mode, concat-direct to the final block) is fully built.
    case featureSkipUnsupported(option: String)
    /// A count derived from the architecture — the total block count or the
    /// parameter count (`quantity` says which) — does not fit in an `Int`.
    /// There is no cap on block count, channels or kernel size; this is the
    /// one size an architecture cannot have, because nothing could count it.
    case arithmeticOverflow(quantity: String)
    /// The architecture's training state does not fit in this Mac's physical
    /// memory (`ModelSizeGuidance`). Not part of `validate()`, which is
    /// device-independent: the Build New Model screen, `--new-model`, the
    /// GUI build and a graft refuse with it before building.
    case trainingStateExceedsPhysicalMemory(parameterCount: Int, trainingStateBytes: Double, physicalMemoryBytes: UInt64)
    /// An architecture-level site's activation field disagrees with whether
    /// the topology has the site (`NetworkArchitecture.activationSiteMismatch`).
    case activationSiteMismatch(ActivationSiteMismatch)
    /// `does_not_apply` passed where a function is required: a setter that
    /// applies one activation to every existing site. `context` names it.
    case notAnActivationFunction(context: String)
    /// `does_not_apply` in a block group's `activationFunction` or
    /// `seActivation`, sites that exist whenever their group does.
    case doesNotApplyAtAnAlwaysPresentSite(field: String)

    var description: String {
        switch self {
        case .activationSiteMismatch(let mismatch):
            return mismatch.description
        case .notAnActivationFunction(let context):
            return "\(context): '\(ActivationFunction.doesNotApply.rawValue)' is not an activation function; "
                + "it marks a site the topology lacks (choose one of \(ActivationFunction.functionList))"
        case .doesNotApplyAtAnAlwaysPresentSite(let field):
            return "\(field) is '\(ActivationFunction.doesNotApply.rawValue)', but that site exists whenever "
                + "its block group does: choose one of \(ActivationFunction.functionList)"
        case .arithmeticOverflow(let quantity):
            return "\(quantity) overflows Int: the architecture is too large to represent"
        case .trainingStateExceedsPhysicalMemory(let parameterCount, let trainingStateBytes, let physicalMemoryBytes):
            return "cannot be trained on this Mac: the training state of \(parameterCount.formatted(.number)) parameters "
                + "(\(ModelSizeGuidance.trainingBytesPerParameter) bytes each: fp32 working weights, master weights, "
                + "momentum velocity and gradient) needs \(ModelSizeGuidance.gigabytesText(trainingStateBytes)), "
                + "more than this Mac's \(ModelSizeGuidance.gigabytesText(Double(physicalMemoryBytes))) of physical memory"
        case .featureSkipNoDestination:
            return "featureSkipSource is enabled but no destination is routed (set at least one of featureSkipToPolicyHead / featureSkipToValueHead)"
        case .featureSkipUnsupported(let option):
            return "feature-skip combination '\(option)' is not supported"
        case .mustBeFiniteNonNegative(let field, let value):
            return "\(field) must be finite and >= 0 (got \(value))"
        case .mustBeFinitePositive(let field, let value):
            return "\(field) must be finite and > 0 (got \(value))"
        case .mustBeFinite(let field, let value):
            return "\(field) must be finite (got \(value))"
        case .initOptionWithoutItsLayer(let field, let value, let reason):
            return "\(field) is '\(value)', but \(reason)"
        case .seBetaInitRequiresScaleAndBias(let group, let seStyle, let seBetaInit):
            return "blockGroups[\(group)].seBetaInit is '\(seBetaInit.rawValue)' but its se_style is "
                + "'\(seStyle.rawValue)'; only '\(SEStyle.scaleAndBias.rawValue)' has a β half, so every "
                + "other SE style requires se_beta_init '\(SEBetaInit.glorot.rawValue)'"
        case .seActivationMismatch(let group, let seStyle, let seActivation):
            return "blockGroups[\(group)].seActivation is '\(seActivation.rawValue)', but "
                + BlockGroup.seActivationRule(seStyle: seStyle)
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

    // Architecture-level activation sites (format v9) ------------------------
    // One field per site (`ArchitectureActivationSite`). Each holds a function
    // exactly when the topology has the site and `does_not_apply` exactly when
    // it does not (`activationSiteMismatch`; checked on decode, in
    // `validate()` and by every setter). Block main paths and SE FC1s use
    // their group's own fields.
    /// `stem_act`, after the stem BN. Exists when the first block group is
    /// post-activation (`hasStemActivation`).
    var stemActivation: ActivationFunction
    /// `tower_final_act`, after the tower-end BN that every head reads.
    /// Exists when the last block group is pre-activation (`hasTowerEndBN`).
    var towerEndActivation: ActivationFunction
    /// `feature_skip_act`, after the compress fusion node's BN. Exists when
    /// that node is built (`featureSkipUsesCompressNode`).
    var featureSkipActivation: ActivationFunction

    // Policy head ---------------------------------------------------------
    var policyHeadStyle: PolicyHeadStyle
    var policyPreConvChannels: Int       // K for intermediate_conv / fc_bottleneck
    /// `policy_pre_act`, the policy pre-block's activation. Exists when
    /// `policyHeadStyle` is `intermediate_conv` or `fc_bottleneck`.
    var policyHeadActivation: ActivationFunction

    // Value head ----------------------------------------------------------
    var valueHeadStyle: ValueHeadStyle
    var valueHeadConvChannels: Int
    var valueHeadHiddenUnits: Int
    /// `value_act`, after the value head's conv BN. Every model has it.
    var valueHeadConvActivation: ActivationFunction
    /// `value_fc1_act`, the value head's FC1 hidden layer (whose width is
    /// `valueHeadHiddenUnits`). Every model has it.
    var valueHeadFC1HiddenActivation: ActivationFunction

    // Head init (init-neutral options, format v8) ---------------------------
    /// Init of the policy head's final projection (`policy.conv.weight`, or
    /// `policy.fc.weight` for `fc_bottleneck`). See `HeadFinalInit`.
    var policyHeadFinalInit: HeadFinalInit
    /// Init of the value head's final FC (`value.wdl_fc2.weight` /
    /// `value.scalar_fc2.weight`). See `HeadFinalInit`.
    var valueHeadFinalInit: HeadFinalInit
    /// The W/D/L head's initial draw probability `p` in (0, 1): its final
    /// bias starts at `wdlBiasPrior(drawProbability: p)`, so the softmax of
    /// zero logits is `(½(1−p), p, ½(1−p))`. The standard value
    /// (`standardValueHeadDrawPrior`) is the `ln 6` bias every W/D/L model was
    /// built with. A claim about the data rather than a neutral choice, so the
    /// Neutral set never changes it. A scalar value head has no draw class:
    /// `validate()` requires the standard value there.
    var valueHeadDrawPrior: Float

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

    /// All-required memberwise init — NO defaults (no silent fallbacks). Like
    /// every other field, the six site activations are not checked here;
    /// `validate()` checks them against the topology.
    init(
        inputEncoding: InputEncoding,
        blockGroups: [BlockGroup],
        stemConvKernelSize: Int,
        stemActivation: ActivationFunction,
        towerEndActivation: ActivationFunction,
        featureSkipActivation: ActivationFunction,
        policyHeadStyle: PolicyHeadStyle,
        policyPreConvChannels: Int,
        policyHeadActivation: ActivationFunction,
        valueHeadStyle: ValueHeadStyle,
        valueHeadConvChannels: Int,
        valueHeadHiddenUnits: Int,
        valueHeadConvActivation: ActivationFunction,
        valueHeadFC1HiddenActivation: ActivationFunction,
        policyHeadFinalInit: HeadFinalInit,
        valueHeadFinalInit: HeadFinalInit,
        valueHeadDrawPrior: Float,
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
        self.stemActivation = stemActivation
        self.towerEndActivation = towerEndActivation
        self.featureSkipActivation = featureSkipActivation
        self.policyHeadStyle = policyHeadStyle
        self.policyPreConvChannels = policyPreConvChannels
        self.policyHeadActivation = policyHeadActivation
        self.valueHeadStyle = valueHeadStyle
        self.valueHeadConvChannels = valueHeadConvChannels
        self.valueHeadHiddenUnits = valueHeadHiddenUnits
        self.valueHeadConvActivation = valueHeadConvActivation
        self.valueHeadFC1HiddenActivation = valueHeadFC1HiddenActivation
        self.policyHeadFinalInit = policyHeadFinalInit
        self.valueHeadFinalInit = valueHeadFinalInit
        self.valueHeadDrawPrior = valueHeadDrawPrior
        self.computeDataType = computeDataType
        self.featureSkipSource = featureSkipSource
        self.featureSkipFusion = featureSkipFusion
        self.featureSkipToPolicyHead = featureSkipToPolicyHead
        self.featureSkipToValueHead = featureSkipToValueHead
        self.featureSkipToFinalBlock = featureSkipToFinalBlock
    }

    /// Convenience for the (common) uniform tower: one group carrying every
    /// block field, count = `numBlocks`. All-required — no defaults.
    ///
    /// `activationFunction` is the one activation a historical
    /// single-recipe tower used at every hidden site: the group's main path,
    /// its SE FC1, and every architecture-level site its topology has. Every
    /// site it lacks is `does_not_apply` (`clearActivationSitesTheTopologyLacks`,
    /// so the existence rule is not restated here).
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
                // activation (an SE-less tower has no FC1). A tower with a
                // different SE FC1 activation sets `seActivation` on the
                // returned value's groups.
                seActivation: blockSeStyle == .none ? .doesNotApply : activationFunction,
                // Every historical tower predates the init-neutral options,
                // so it states their standard values; a tower with a
                // non-standard one sets it on the returned value.
                seGammaBiasInit: BlockGroup.standardSEGammaBiasInit,
                branchOutputInit: .standard,
                skipProjectionInit: .he
            )],
            stemConvKernelSize: stemConvKernelSize,
            stemActivation: activationFunction,
            towerEndActivation: activationFunction,
            featureSkipActivation: activationFunction,
            policyHeadStyle: policyHeadStyle,
            policyPreConvChannels: policyPreConvChannels,
            policyHeadActivation: activationFunction,
            valueHeadStyle: valueHeadStyle,
            valueHeadConvChannels: valueHeadConvChannels,
            valueHeadHiddenUnits: valueHeadHiddenUnits,
            valueHeadConvActivation: activationFunction,
            valueHeadFC1HiddenActivation: activationFunction,
            policyHeadFinalInit: .he,
            valueHeadFinalInit: .he,
            valueHeadDrawPrior: Self.standardValueHeadDrawPrior,
            computeDataType: computeDataType,
            // Uniform towers default to feature-skip OFF; presets that enable it
            // mutate the returned value's `featureSkip*` fields.
            featureSkipSource: .none,
            featureSkipFusion: .concatDirect,
            featureSkipToPolicyHead: false,
            featureSkipToValueHead: false,
            featureSkipToFinalBlock: false
        )
        clearActivationSitesTheTopologyLacks()
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
    //
    // The six architecture-level site activations (format v9) replace the
    // single top-level `activation_function`, which is now decode-only
    // (`legacyActivationFunction`): a file older than v9, or in the
    // uniform-tower form, resolves each unstated site from it (an existing
    // site) or to `does_not_apply` (a site the topology lacks); a v9+
    // block-groups file must state all six and must not state it.

    enum CodingKeys: String, CodingKey {
        case inputEncoding = "input_encoding"
        case blockGroups = "block_groups"
        case stemConvKernelSize = "stem_conv_kernel_size"
        case stemActivation = "stem_activation"
        case towerEndActivation = "tower_end_activation"
        case featureSkipActivation = "feature_skip_activation"
        case policyHeadStyle = "policy_head_style"
        case policyPreConvChannels = "policy_pre_conv_channels"
        case policyHeadActivation = "policy_head_activation"
        case valueHeadStyle = "value_head_style"
        case valueHeadConvChannels = "value_head_conv_channels"
        case valueHeadHiddenUnits = "value_head_hidden_units"
        case valueHeadConvActivation = "value_head_conv_activation"
        case valueHeadFC1HiddenActivation = "value_head_fc1_hidden_activation"
        case policyHeadFinalInit = "policy_head_final_init"
        case valueHeadFinalInit = "value_head_final_init"
        case valueHeadDrawPrior = "value_head_draw_prior"
        case computeDataType = "compute_data_type"
        case featureSkipSource = "feature_skip_source"
        case featureSkipFusion = "feature_skip_fusion"
        case featureSkipToPolicyHead = "feature_skip_to_policy_head"
        case featureSkipToValueHead = "feature_skip_to_value_head"
        case featureSkipToFinalBlock = "feature_skip_to_final_block"
        // The tower-wide activation before format v9 — decode-only, never
        // written; refused in a v9+ block-groups file.
        case legacyActivationFunction = "activation_function"
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
    ///
    /// The six site activations decode in two passes, because which sites
    /// exist depends on the topology (groups, policy style, feature skip) and
    /// the existence rule (`hasActivationSite`) is an instance method usable
    /// only once every stored property is set. Pass one takes each stated
    /// value, or — in a file allowed to omit it — provisionally the file's own
    /// top-level `activation_function`, marking the site resolved
    /// (`ArchitectureFormat.decodeSiteActivation`). Pass two, on the complete
    /// value, turns each resolved site the topology lacks into
    /// `does_not_apply`, records every resolution on the legacy log, and then
    /// refuses any site whose value disagrees with its existence. Nothing is
    /// ever repaired: a mismatch can only come from a stated value (or a
    /// legacy `activation_function` of `does_not_apply`), and it is an error.
    init(from decoder: Decoder, format: ArchitectureFormat.DecodeFormat) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        inputEncoding = try c.decode(InputEncoding.self, forKey: .inputEncoding)
        stemConvKernelSize = try c.decode(Int.self, forKey: .stemConvKernelSize)
        policyHeadStyle = try c.decode(PolicyHeadStyle.self, forKey: .policyHeadStyle)
        policyPreConvChannels = try c.decode(Int.self, forKey: .policyPreConvChannels)
        valueHeadStyle = try c.decode(ValueHeadStyle.self, forKey: .valueHeadStyle)
        valueHeadConvChannels = try c.decode(Int.self, forKey: .valueHeadConvChannels)
        valueHeadHiddenUnits = try c.decode(Int.self, forKey: .valueHeadHiddenUnits)
        // The uniform-tower keys predate block groups, so a file in that form
        // predates the head init options too, whatever version its carrier
        // states (see the uniform-tower branch below).
        let isUniformTowerForm = !c.contains(.blockGroups)
        // A v9+ block-groups file states the six site keys instead; a stale
        // `activation_function` there (an old preset hand-bumped to v9) would
        // otherwise be silently ignored by a reader who thinks it still sets
        // the heads. The uniform-tower form is legacy by construction.
        if !isUniformTowerForm, !format.allowsMissingSiteActivations, c.contains(.legacyActivationFunction) {
            throw ArchitectureFormat.FormatError.retiredField(
                field: CodingKeys.legacyActivationFunction.rawValue,
                location: ArchitectureFormat.location(of: decoder),
                formatVersion: format.formatVersion,
                source: format.source,
                replacedBy: ArchitectureActivationSite.allCases.map(\.jsonKey))
        }
        // Optional here: a file that states all six site keys needs no
        // tower-wide value (the committed tests re-stamp current encodes as
        // older versions that way). The uniform-tower form requires it below,
        // because its group's own fields come from it.
        let legacyTowerActivation = try c.decodeIfPresent(ActivationFunction.self, forKey: .legacyActivationFunction)
        let siteKeys = ArchitectureActivationSite.allCases.map(\.codingKey)
        func decodeSite(_ site: ArchitectureActivationSite) throws -> ArchitectureFormat.DecodedSiteActivation {
            try ArchitectureFormat.decodeSiteActivation(
                key: site.codingKey, in: c, decoder: decoder, format: format,
                legacyByConstruction: isUniformTowerForm,
                legacyTowerActivation: legacyTowerActivation,
                allSiteKeys: siteKeys)
        }
        let decodedStem = try decodeSite(.stem)
        let decodedTowerEnd = try decodeSite(.towerEnd)
        let decodedFeatureSkip = try decodeSite(.featureSkipFusion)
        let decodedPolicyHead = try decodeSite(.policyHead)
        let decodedValueHeadConv = try decodeSite(.valueHeadConv)
        let decodedValueHeadFC1Hidden = try decodeSite(.valueHeadFC1Hidden)
        stemActivation = decodedStem.value
        towerEndActivation = decodedTowerEnd.value
        featureSkipActivation = decodedFeatureSkip.value
        policyHeadActivation = decodedPolicyHead.value
        valueHeadConvActivation = decodedValueHeadConv.value
        valueHeadFC1HiddenActivation = decodedValueHeadFC1Hidden.value
        let resolvedSites: [ArchitectureActivationSite] = [
            (ArchitectureActivationSite.stem, decodedStem),
            (.towerEnd, decodedTowerEnd),
            (.featureSkipFusion, decodedFeatureSkip),
            (.policyHead, decodedPolicyHead),
            (.valueHeadConv, decodedValueHeadConv),
            (.valueHeadFC1Hidden, decodedValueHeadFC1Hidden),
        ].filter { $0.1.wasResolvedFromLegacy }.map(\.0)
        policyHeadFinalInit = try ArchitectureFormat.decodeInitOption(
            HeadFinalInit.self, key: CodingKeys.policyHeadFinalInit, in: c, decoder: decoder, format: format,
            legacyByConstruction: isUniformTowerForm,
            standard: .he, rendered: \.rawValue)
        valueHeadFinalInit = try ArchitectureFormat.decodeInitOption(
            HeadFinalInit.self, key: CodingKeys.valueHeadFinalInit, in: c, decoder: decoder, format: format,
            legacyByConstruction: isUniformTowerForm,
            standard: .he, rendered: \.rawValue)
        valueHeadDrawPrior = try ArchitectureFormat.decodeInitOption(
            Float.self, key: CodingKeys.valueHeadDrawPrior, in: c, decoder: decoder, format: format,
            legacyByConstruction: isUniformTowerForm,
            standard: Self.standardValueHeadDrawPrior, rendered: { "\($0)" })
        computeDataType = try c.decode(ComputeDataType.self, forKey: .computeDataType)
        // Feature skip: optional + defaulted so every pre-feature-skip file decodes
        // to a fully-off (byte-identical) configuration.
        featureSkipSource = try c.decodeIfPresent(FeatureSkipSource.self, forKey: .featureSkipSource) ?? .none
        featureSkipFusion = try c.decodeIfPresent(FeatureSkipFusion.self, forKey: .featureSkipFusion) ?? .concatDirect
        featureSkipToPolicyHead = try c.decodeIfPresent(Bool.self, forKey: .featureSkipToPolicyHead) ?? false
        featureSkipToValueHead = try c.decodeIfPresent(Bool.self, forKey: .featureSkipToValueHead) ?? false
        featureSkipToFinalBlock = try c.decodeIfPresent(Bool.self, forKey: .featureSkipToFinalBlock) ?? false
        if !isUniformTowerForm {
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
            let towerActivation = try c.decode(ActivationFunction.self, forKey: .legacyActivationFunction)
            guard towerActivation != .doesNotApply else {
                throw ArchitectureFormat.FormatError.doesNotApplyAtAnAlwaysPresentSite(
                    field: CodingKeys.legacyActivationFunction.rawValue,
                    location: ArchitectureFormat.location(of: decoder),
                    formatVersion: format.formatVersion,
                    source: format.source)
            }
            let legacyAlphaInit = try c.decode(Float.self, forKey: .legacyRezeroAlphaInit)
            let legacyAlphaCap = BlockGroup.legacyRezeroAlphaCap(forAlphaInit: legacyAlphaInit)
            let legacySEStyle = try c.decode(SEStyle.self, forKey: .legacyBlockSeStyle)
            blockGroups = [BlockGroup(
                count: try c.decode(Int.self, forKey: .legacyNumBlocks),
                channels: try c.decode(Int.self, forKey: .legacyChannels),
                conv1KernelSize: try c.decode(Int.self, forKey: .legacyBlockConv1KernelSize),
                conv2KernelSize: try c.decode(Int.self, forKey: .legacyBlockConv2KernelSize),
                seStyle: legacySEStyle,
                seReductionRatio: try c.decode(Int.self, forKey: .legacyBlockSeReductionRatio),
                useRezero: try c.decode(Bool.self, forKey: .legacyBlockUseRezero),
                rezeroAlphaInit: legacyAlphaInit,
                rezeroAlphaCap: legacyAlphaCap,
                activationFunction: towerActivation,
                activationStyle: try c.decode(BlockActivationStyle.self, forKey: .legacyBlockActivationStyle),
                skipMerge: try c.decode(BlockSkipMerge.self, forKey: .legacyBlockSkipMerge),
                dropoutMultiplier: 1,
                seBetaInit: .glorot,
                seActivation: legacySEStyle == .none ? .doesNotApply : towerActivation,
                seGammaBiasInit: BlockGroup.standardSEGammaBiasInit,
                branchOutputInit: .standard,
                skipProjectionInit: .he
            )]
        }

        // Pass two (`self` is complete): a resolved site the topology lacks
        // becomes `does_not_apply`; each resolution is described for the log.
        var siteResolutions: [String] = []
        for site in resolvedSites {
            if hasActivationSite(site) {
                siteResolutions.append(
                    "\(site.jsonKey) := \(activation(at: site).rawValue) "
                        + "(the file's \(CodingKeys.legacyActivationFunction.rawValue))")
            } else {
                setStoredActivation(.doesNotApply, at: site)
                siteResolutions.append(
                    "\(site.jsonKey) := \(ActivationFunction.doesNotApply.rawValue) (\(site.absentReason))")
            }
        }
        if isUniformTowerForm {
            // The uniform-tower keys predate block groups, so no writer of any
            // version that has `se_beta_init`, `se_activation`,
            // `rezero_alpha_cap` or the site activations emits them: this form
            // is legacy by construction, whatever version the carrier states.
            let group = blockGroups[0]
            let groupResolutions = [
                "block_groups[0].\(BlockGroup.CodingKeys.seBetaInit.rawValue) := \(SEBetaInit.glorot.rawValue)",
                "block_groups[0].\(BlockGroup.CodingKeys.seActivation.rawValue) := \(group.seActivation.rawValue)",
                "block_groups[0].\(BlockGroup.CodingKeys.rezeroAlphaCap.rawValue) := \(group.rezeroAlphaCap)",
                "block_groups[0].\(BlockGroup.CodingKeys.seGammaBiasInit.rawValue) := \(BlockGroup.standardSEGammaBiasInit)",
                "block_groups[0].\(BlockGroup.CodingKeys.branchOutputInit.rawValue) := \(BranchOutputInit.standard.rawValue)",
                "block_groups[0].\(BlockGroup.CodingKeys.skipProjectionInit.rawValue) := \(SkipProjectionInit.he.rawValue)",
            ]
            format.legacyLog.record(
                "legacy uniform-tower keys: " + (groupResolutions + siteResolutions).joined(separator: ", "))
        } else {
            let prefix = decoder.codingPath.isEmpty ? "" : "\(ArchitectureFormat.location(of: decoder))."
            for resolution in siteResolutions {
                format.legacyLog.record(prefix + resolution)
            }
        }
        if let mismatch = activationSiteMismatch {
            throw ArchitectureFormat.FormatError.activationSiteMismatch(
                mismatch,
                location: ArchitectureFormat.location(of: decoder),
                formatVersion: format.formatVersion,
                source: format.source)
        }
    }

    func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(inputEncoding, forKey: .inputEncoding)
        try c.encode(blockGroups, forKey: .blockGroups)
        try c.encode(stemConvKernelSize, forKey: .stemConvKernelSize)
        // All six, `does_not_apply` included, so a v9 file says which sites
        // its topology lacks; the retired `activation_function` is never
        // written.
        try c.encode(stemActivation, forKey: .stemActivation)
        try c.encode(towerEndActivation, forKey: .towerEndActivation)
        try c.encode(featureSkipActivation, forKey: .featureSkipActivation)
        try c.encode(policyHeadStyle, forKey: .policyHeadStyle)
        try c.encode(policyPreConvChannels, forKey: .policyPreConvChannels)
        try c.encode(policyHeadActivation, forKey: .policyHeadActivation)
        try c.encode(valueHeadStyle, forKey: .valueHeadStyle)
        try c.encode(valueHeadConvChannels, forKey: .valueHeadConvChannels)
        try c.encode(valueHeadHiddenUnits, forKey: .valueHeadHiddenUnits)
        try c.encode(valueHeadConvActivation, forKey: .valueHeadConvActivation)
        try c.encode(valueHeadFC1HiddenActivation, forKey: .valueHeadFC1HiddenActivation)
        try c.encode(policyHeadFinalInit, forKey: .policyHeadFinalInit)
        try c.encode(valueHeadFinalInit, forKey: .valueHeadFinalInit)
        try c.encode(valueHeadDrawPrior, forKey: .valueHeadDrawPrior)
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
        blockIndex == numBlocks - 1 ? finalBlockSkipExtraInputChannels : 0
    }

    /// `blockSkipExtraInputChannels(blockIndex:)` of the last expanded block —
    /// the one block the final-block feature skip widens — without counting
    /// the blocks, for the per-group formulas (`parameterCountBreakdown`,
    /// `groupsWithSkipProjection`) that never expand the tower.
    var finalBlockSkipExtraInputChannels: Int {
        guard featureSkipEnabled,
              featureSkipFusion == .concatDirect,
              featureSkipToFinalBlock else { return 0 }
        return featureSkipSourceChannels
    }

    /// The tower flattened to one element per block (each returned group has
    /// `count == 1`). The ENGINE'S ONLY VIEW of the tower: graph builders,
    /// `weightTensorPlan`, and the analyzer walk this — groups are an
    /// authoring/persistence structure, never an engine concept.
    ///
    /// Requires a validated tower shape (`validateTowerShape()`, which
    /// `validate()` runs first): a negative count traps here, and a huge one
    /// allocates the whole expansion. Nothing that reads an architecture the
    /// user is still typing may call it before the shape is known to be
    /// good; the per-group formulas exist so the Build screen never has to.
    var expandedBlocks: [BlockGroup] {
        blockGroups.flatMap { group -> [BlockGroup] in
            var single = group
            single.count = 1
            return Array(repeating: single, count: group.count)
        }
    }

    /// Total block count across all groups (derived; no stored copy).
    /// Requires a validated tower shape: a total that overflows `Int` is a
    /// defect here, reported by `validateTowerShape()`.
    var numBlocks: Int {
        do {
            return try checkedTotalBlockCount()
        } catch {
            preconditionFailure("NetworkArchitecture.numBlocks: \(error) (validateTowerShape() rejects this)")
        }
    }

    /// The sum of every group's block count, or `arithmeticOverflow` when it
    /// does not fit in an `Int`.
    func checkedTotalBlockCount() throws -> Int {
        try blockGroups.reduce(0) { total, group in
            try Self.checkedSum([total, group.count], quantity: "the total block count")
        }
    }

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

    /// Everything an architecture must satisfy on any machine. Never walks
    /// the tower block by block: the shape is checked first
    /// (`validateTowerShape()`), and the one size check — that the parameter
    /// count fits in an `Int` — runs group by group
    /// (`checkedParameterCountBreakdown()`), so the Build New Model screen
    /// can validate whatever the user has typed. Whether the model fits this
    /// Mac is a separate, build-time question (`ModelSizeGuidance`).
    ///
    /// Activations: a block group's `activationFunction` and `seActivation`
    /// must be functions (their sites exist whenever the group does), and
    /// each architecture-level site field must be a function exactly when
    /// the topology has the site and `does_not_apply` exactly when it does
    /// not (`activationSiteMismatch`). The site check runs after the
    /// init-option and feature-skip checks, so an architecture with one of
    /// those older, more specific errors reports that error first.
    func validate() throws {
        try requireOdd("stemConvKernelSize", stemConvKernelSize)
        try validateTowerShape()
        for (gi, g) in blockGroups.enumerated() {
            guard g.activationFunction != .doesNotApply else {
                throw NetworkArchitectureError.doesNotApplyAtAnAlwaysPresentSite(
                    field: "blockGroups[\(gi)].activationFunction")
            }
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
            // An SE-less group has no FC1, so its se_activation is
            // `does_not_apply` there and only there (OD-13): one value per
            // graph, so two architectures that build the identical network
            // compare (and hash) equal, and an SE group always names its
            // FC1's function.
            if (g.seActivation != .doesNotApply) != (g.seStyle != .none) {
                throw NetworkArchitectureError.seActivationMismatch(
                    group: gi, seStyle: g.seStyle, seActivation: g.seActivation)
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
        try validateInitOptions()
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
        if let mismatch = activationSiteMismatch {
            throw NetworkArchitectureError.activationSiteMismatch(mismatch)
        }
        // Last: every width, kernel and count it multiplies is now known to
        // be positive, so the only way it can fail is an overflow.
        _ = try checkedParameterCountBreakdown()
    }

    /// The tower's shape, checked without walking it: at least one group,
    /// every group's block count positive, and a total block count that
    /// fits in an `Int`. Everything that walks the tower block by block
    /// (`expandedBlocks`, `numBlocks`, `blockRange(ofGroup:)`,
    /// `skipProjectionBlockIndices`, `weightTensorPlan`, the graph builders)
    /// requires it; `validate()` runs it first.
    func validateTowerShape() throws {
        try requirePositive("blockGroups.count", blockGroups.count)
        for (gi, g) in blockGroups.enumerated() {
            try requirePositive("blockGroups[\(gi)].count", g.count)
        }
        _ = try checkedTotalBlockCount()
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
    /// in tests). Requires a validated architecture: `validate()` computes the
    /// same breakdown with overflow checking and refuses one that overflows.
    var parameterCount: Int { parameterCountBreakdown.total }

    /// `checkedParameterCountBreakdown()` for a validated architecture.
    var parameterCountBreakdown: ParameterCountBreakdown {
        do {
            return try checkedParameterCountBreakdown()
        } catch {
            preconditionFailure("NetworkArchitecture.parameterCountBreakdown: \(error) (validate() rejects this)")
        }
    }

    /// Persistent-tensor element counts by section of the network, the
    /// sections `weightTensorPlan` names: `stem.*`, the `blocks.*` of each
    /// group, `tower_final_bn.*`, `feature_skip.*`, `policy.*` and `value.*`.
    struct ParameterCountBreakdown: Equatable, Sendable {
        let stem: Int
        /// One entry per block group, in tower order.
        let perGroup: [Int]
        let towerEndBN: Int
        /// The compress-fusion node; zero unless `featureSkipUsesCompressNode`.
        let featureSkip: Int
        let policy: Int
        let value: Int
        let total: Int
    }

    /// The parameter count section by section, computed group by group with
    /// every product and sum overflow-checked — the one formula behind
    /// `parameterCount`, `validate()`'s size check and the architecture
    /// diagram's per-segment counts. A group of `count` identical blocks
    /// costs its first block (input = the previous group's width), then
    /// `count − 1` blocks at its own width, with the final-block feature
    /// skip widening only the tower's last block; so a tower of any depth is
    /// counted without expanding it.
    ///
    /// Requires every width, kernel size, count and SE ratio positive and
    /// each group's channels divisible by its SE ratio — the checks
    /// `validate()` makes before calling it. Throws `arithmeticOverflow` when
    /// a count does not fit in an `Int`.
    func checkedParameterCountBreakdown() throws -> ParameterCountBreakdown {
        let quantity = "the parameter count"
        func product(_ factors: Int...) throws -> Int { try Self.checkedProduct(factors, quantity: quantity) }
        func sum(_ terms: Int...) throws -> Int { try Self.checkedSum(terms, quantity: quantity) }

        /// One block of `spec` reading `inCEff` channels: block `i`'s conv1
        /// maps `inC → outC`, BN1 is sized by the block input (pre-act) or the
        /// conv1 output (post-act), everything after runs at `outC`, and a
        /// width transition adds the 1×1 skip projection.
        func blockCount(inputChannels inCEff: Int, spec: BlockGroup) throws -> Int {
            let outC = spec.channels
            let conv1 = try product(outC, inCEff, spec.conv1KernelSize, spec.conv1KernelSize)
            let conv2 = try product(outC, outC, spec.conv2KernelSize, spec.conv2KernelSize)
            let bn1 = try product(4, spec.activationStyle == .pre ? inCEff : outC)
            let bn2 = try product(4, outC)
            let seReduced = spec.seStyle == .none ? 0 : outC / spec.seReductionRatio
            let se: Int
            switch spec.seStyle {
            case .none:
                se = 0
            case .attenuateOnly:
                se = try sum(product(outC, seReduced), seReduced, product(seReduced, outC), outC)
            case .scaleAndBias:
                se = try sum(product(outC, seReduced), seReduced, product(seReduced, 2, outC), product(2, outC))
            }
            let rezero = spec.useRezero ? 1 : 0
            let proj = try inCEff != outC ? product(inCEff, outC) : 0
            // Optional output LayerNorm: per-channel γ + β (no running stats).
            let outNorm = try spec.resolvedOutputNorm == .layerNorm ? product(2, outC) : 0
            return try sum(conv1, conv2, bn1, bn2, se, rezero, proj, outNorm)
        }

        let c0 = stemOutputChannels
        // Stem: conv (bias-free) + BN.
        let stem = try sum(product(inputPlanes, c0, stemConvKernelSize, stemConvKernelSize), product(4, c0))

        // Tower. The final-block feature skip (`+ source` on the last block's
        // input under a routed concatDirect skip) widens only the last block
        // of the last group.
        let finalExtra = finalBlockSkipExtraInputChannels
        var perGroup: [Int] = []
        perGroup.reserveCapacity(blockGroups.count)
        var inC = c0
        for (index, group) in blockGroups.enumerated() {
            let isLastGroup = index == blockGroups.count - 1
            let outC = group.channels
            let firstInput = try sum(inC, isLastGroup && group.count == 1 ? finalExtra : 0)
            var groupTotal = try blockCount(inputChannels: firstInput, spec: group)
            if group.count >= 2 {
                let interior = try product(group.count - 2, blockCount(inputChannels: outC, spec: group))
                let lastInput = try sum(outC, isLastGroup ? finalExtra : 0)
                groupTotal = try sum(groupTotal, interior, blockCount(inputChannels: lastInput, spec: group))
            }
            perGroup.append(groupTotal)
            inC = outC
        }

        let cT = towerOutputChannels
        let towerEndBN = try hasTowerEndBN ? product(4, cT) : 0

        // Heads. The FIRST conv of each head reads the effective input width — wider
        // by the feature-skip source under a routed `concatDirect` skip, else `cT`
        // (`headInputChannels(routed:)`, restated here with overflow checking).
        let routedWidening = featureSkipEnabled && featureSkipFusion == .concatDirect ? featureSkipSourceChannels : 0
        let cP = try sum(cT, featureSkipToPolicyHead ? routedWidening : 0)
        let cVin = try sum(cT, featureSkipToValueHead ? routedWidening : 0)

        // Policy head.
        let pK = policyPreConvChannels
        let policy: Int
        switch policyHeadStyle {
        case .simpleConv:
            policy = try sum(product(cP, policyChannels), policyChannels)
        case .intermediateConv:
            policy = try sum(product(cP, pK), product(4, pK), product(pK, policyChannels), policyChannels)
        case .fcBottleneck:
            let flat = try product(pK, boardSize, boardSize)
            policy = try sum(product(cP, pK), product(4, pK), product(flat, policySize), policySize)
        }

        // Value head.
        let cv = valueHeadConvChannels
        let h = valueHeadHiddenUnits
        let flatV = try product(boardSize, boardSize, cv)
        let value = try sum(product(cVin, cv), product(4, cv), product(flatV, h), h,
                            product(h, valueHeadClasses), valueHeadClasses)

        // Compress fusion node (head-only): 1×1 conv (towerC+source → towerC) + BN.
        let featureSkip = try featureSkipUsesCompressNode
            ? sum(product(sum(cT, featureSkipSourceChannels), cT), product(4, cT))
            : 0

        let total = try sum(stem, Self.checkedSum(perGroup, quantity: quantity), towerEndBN, featureSkip, policy, value)
        return ParameterCountBreakdown(
            stem: stem, perGroup: perGroup, towerEndBN: towerEndBN,
            featureSkip: featureSkip, policy: policy, value: value, total: total
        )
    }

    /// `terms` added up, or `arithmeticOverflow(quantity:)`.
    static func checkedSum(_ terms: [Int], quantity: String) throws -> Int {
        var total = 0
        for term in terms {
            let (next, overflow) = total.addingReportingOverflow(term)
            guard !overflow else { throw NetworkArchitectureError.arithmeticOverflow(quantity: quantity) }
            total = next
        }
        return total
    }

    /// `factors` multiplied, or `arithmeticOverflow(quantity:)`.
    static func checkedProduct(_ factors: [Int], quantity: String) throws -> Int {
        var total = 1
        for factor in factors {
            let (next, overflow) = total.multipliedReportingOverflow(by: factor)
            guard !overflow else { throw NetworkArchitectureError.arithmeticOverflow(quantity: quantity) }
            total = next
        }
        return total
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
            + siteActivationClause
            + " . policy \(policyHeadStyle.rawValue)(\(policySize))"
            + " . value \(valueDesc)"
            + headInitMarker
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
            + ", drop*\(String(format: "%g", g.dropoutMultiplier))\(groupInitMarker(g))]"
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

// MARK: - Architecture-level activation sites (format v9)

extension NetworkArchitecture {

    /// Whether the topology has `site`: the single existence rule every
    /// consumer reads (decode, `validate()`, the setters, the Build screen,
    /// the summary). Built from the same properties the graph builder
    /// branches on, and never expands the tower, so the Build screen can ask
    /// it about any draft (one with at least one group).
    func hasActivationSite(_ site: ArchitectureActivationSite) -> Bool {
        switch site {
        case .stem:
            return hasStemActivation
        case .towerEnd:
            return hasTowerEndBN
        case .featureSkipFusion:
            return featureSkipUsesCompressNode
        case .policyHead:
            switch policyHeadStyle {
            case .simpleConv: return false
            case .intermediateConv, .fcBottleneck: return true
            }
        case .valueHeadConv, .valueHeadFC1Hidden:
            return true
        }
    }

    /// The value `site`'s field holds. The six stored properties stay the
    /// single source of truth; this only selects one.
    func activation(at site: ArchitectureActivationSite) -> ActivationFunction {
        switch site {
        case .stem: return stemActivation
        case .towerEnd: return towerEndActivation
        case .featureSkipFusion: return featureSkipActivation
        case .policyHead: return policyHeadActivation
        case .valueHeadConv: return valueHeadConvActivation
        case .valueHeadFC1Hidden: return valueHeadFC1HiddenActivation
        }
    }

    /// Why `site` exists in this architecture, or why it does not — the
    /// reason a mismatch error gives.
    func activationSiteReason(_ site: ArchitectureActivationSite) -> String {
        guard hasActivationSite(site) else { return site.absentReason }
        switch site {
        case .stem:
            return "the first block group is post-activation, so the stem has an activation"
        case .towerEnd:
            return "the last block group is pre-activation, so the tower ends in a BN and an activation"
        case .featureSkipFusion:
            return "a compress fusion node is built (\(FeatureSkipFusion.compressConvBNReLU.rawValue) routed to a head)"
        case .policyHead:
            return "the policy head has a pre-block (\(policyHeadStyle.rawValue))"
        case .valueHeadConv:
            return "every value head has a conv activation"
        case .valueHeadFC1Hidden:
            return "every value head has an FC1 hidden layer"
        }
    }

    /// The first site, in build order, whose field disagrees with whether
    /// the topology has it: `does_not_apply` at an existing site, or a
    /// function at an absent one. Nil when every site agrees. The one rule
    /// decode, `validate()` and the Build screen all apply.
    var activationSiteMismatch: ActivationSiteMismatch? {
        for site in ArchitectureActivationSite.allCases {
            let value = activation(at: site)
            let exists = hasActivationSite(site)
            if (value != .doesNotApply) != exists {
                return ActivationSiteMismatch(
                    site: site, value: value, siteExists: exists, reason: activationSiteReason(site))
            }
        }
        return nil
    }

    /// Writes one site's stored field, unchecked. Every public path goes
    /// through a checking setter; decode uses this for its second pass.
    private mutating func setStoredActivation(_ value: ActivationFunction, at site: ArchitectureActivationSite) {
        switch site {
        case .stem: stemActivation = value
        case .towerEnd: towerEndActivation = value
        case .featureSkipFusion: featureSkipActivation = value
        case .policyHead: policyHeadActivation = value
        case .valueHeadConv: valueHeadConvActivation = value
        case .valueHeadFC1Hidden: valueHeadFC1HiddenActivation = value
        }
    }

    /// Sets one site's activation. Refuses a value that disagrees with the
    /// site's existence — a function at a site the topology lacks, or
    /// `does_not_apply` at one it has — leaving the architecture unchanged.
    mutating func setActivation(_ value: ActivationFunction, at site: ArchitectureActivationSite) throws {
        let exists = hasActivationSite(site)
        guard (value != .doesNotApply) == exists else {
            throw NetworkArchitectureError.activationSiteMismatch(ActivationSiteMismatch(
                site: site, value: value, siteExists: exists, reason: activationSiteReason(site)))
        }
        setStoredActivation(value, at: site)
    }

    /// Sets `value` at every architecture-level site the topology has and
    /// never touches a site it lacks (those stay `does_not_apply`). Refuses
    /// `does_not_apply` before changing anything.
    mutating func setActivationAtEveryExistingSite(_ value: ActivationFunction) throws {
        guard value != .doesNotApply else {
            throw NetworkArchitectureError.notAnActivationFunction(context: "the activation for every existing site")
        }
        for site in ArchitectureActivationSite.allCases where hasActivationSite(site) {
            setStoredActivation(value, at: site)
        }
    }

    /// The `--set-activation` rule, shared with the Build screen's "Use for
    /// every activation" so an edit made either way gives the same
    /// architecture: `value` at every existing architecture-level site and
    /// on every block group's main path (`BlockGroup.setActivationFunction`,
    /// which never touches `seActivation`: an SE group keeps its FC1's
    /// function and an SE-less group's stays `does_not_apply`). Refuses
    /// `does_not_apply` before changing anything.
    ///
    /// It equals the uniform convenience init built with `value` only on an
    /// SE-less tower; with SE blocks that also needs each group's
    /// `seActivation` set (`--set-se-activation`).
    mutating func setMainActivationEverywhere(_ value: ActivationFunction) throws {
        guard value != .doesNotApply else {
            throw NetworkArchitectureError.notAnActivationFunction(context: "the main activation everywhere")
        }
        try setActivationAtEveryExistingSite(value)
        for index in blockGroups.indices {
            blockGroups[index].setActivationFunction(value)
        }
    }

    /// Sets every site the topology lacks to `does_not_apply` and leaves
    /// every existing site alone. It never fills a site that exists: a site
    /// that has just appeared keeps whatever it holds — `does_not_apply` if
    /// it was cleared when it last disappeared — and `validate()` names it
    /// until a function is chosen. For code that has just changed the
    /// topology (the Build screen, tests that flip a style). Decode and
    /// `validate()` never call it: a stored mismatch is an error, never
    /// repaired.
    mutating func clearActivationSitesTheTopologyLacks() {
        for site in ArchitectureActivationSite.allCases where !hasActivationSite(site) {
            setStoredActivation(.doesNotApply, at: site)
        }
    }

    /// The summary's activation clause. ` . act X` when every site the
    /// topology has uses one activation `X` — the form every architecture
    /// had before the sites were split, so their summaries are unchanged —
    /// otherwise ` . act ` and each existing site's `<label> <fn>` in build
    /// order. `does_not_apply` never appears: absent sites are not listed.
    var siteActivationClause: String {
        let present = ArchitectureActivationSite.allCases.filter { hasActivationSite($0) }
        let activations = present.map { activation(at: $0) }
        if let first = activations.first, activations.allSatisfy({ $0 == first }) {
            return " . act \(first.rawValue)"
        }
        return " . act " + zip(present, activations)
            .map { "\($0.summaryLabel) \($1.rawValue)" }
            .joined(separator: ", ")
    }
}

// MARK: - Init-neutral options

extension NetworkArchitecture {

    /// The draw probability every W/D/L model was built with before
    /// `valueHeadDrawPrior` existed: bias `[0, ln 6, 0]`, softmax
    /// `(0.125, 0.75, 0.125)`.
    static let standardValueHeadDrawPrior: Float = 0.75

    /// The W/D/L head's final bias for an initial draw probability `p`:
    /// `[0, ln(2p / (1 − p)), 0]` (slot order `[win, draw, loss]`), whose
    /// softmax is `(½(1−p), p, ½(1−p))` and whose derived scalar
    /// `p_win − p_loss` is 0. The single source of the prior: the graph
    /// builder, `--derive-model` and the tests all read it. Computed in Double
    /// and narrowed once, so the standard prior is bit-identical to the `ln 6`
    /// literal it replaced.
    static func wdlBiasPrior(drawProbability p: Float) -> [Float] {
        let probability = Double(p)
        return [0, Float(log(2 * probability / (1 - probability))), 0]
    }

    /// The expanded-block index range of block group `group`. Requires a
    /// validated tower shape (`validateTowerShape()`).
    func blockRange(ofGroup group: Int) -> Range<Int> {
        let first = blockGroups[..<group].reduce(0) { $0 + $1.count }
        return first..<(first + blockGroups[group].count)
    }

    /// The expanded blocks that carry a width-transition skip projection —
    /// exactly the blocks `weightTensorPlan()` gives a `skip_proj.weight`
    /// (input width, feature skip included, differs from the block's width).
    /// Walks the expanded tower, so it requires a validated tower shape; the
    /// per-group question is `groupsWithSkipProjection`, which does not.
    var skipProjectionBlockIndices: [Int] {
        var indices: [Int] = []
        var inC = stemOutputChannels
        for (index, block) in expandedBlocks.enumerated() {
            if inC + blockSkipExtraInputChannels(blockIndex: index) != block.channels {
                indices.append(index)
            }
            inC = block.channels
        }
        return indices
    }

    /// The block groups with at least one skip projection — where a group's
    /// `skipProjectionInit` takes effect — worked out group by group, never
    /// by expanding the tower, and defined for any tower the user can type:
    /// the Build New Model screen reads it on every redraw, and
    /// `validateInitOptions()` runs before the parameter count is known to
    /// fit.
    ///
    /// The same rule as `skipProjectionBlockIndices` (which it equals on a
    /// valid tower; pinned in tests): a group's first block reads the
    /// previous group's width (the stem's, for the first group), every later
    /// block reads its own group's width, and the final-block feature skip
    /// widens only the tower's last block. A group whose count is not
    /// positive has no blocks, so it has no projection and passes the
    /// incoming width through unchanged.
    var groupsWithSkipProjection: Set<Int> {
        guard let lastGroupWithBlocks = blockGroups.lastIndex(where: { $0.count > 0 }) else { return [] }
        let finalExtra = finalBlockSkipExtraInputChannels
        var groups: Set<Int> = []
        var inC = stemOutputChannels
        for (index, group) in blockGroups.enumerated() where group.count > 0 {
            let holdsFinalBlock = index == lastGroupWithBlocks
            let firstInput = inC + (holdsFinalBlock && group.count == 1 ? finalExtra : 0)
            let lastBlockWidened = holdsFinalBlock && group.count >= 2 && finalExtra != 0
            if firstInput != group.channels || lastBlockWidened {
                groups.insert(index)
            }
            inC = group.channels
        }
        return groups
    }

    /// Whether any block of group `group` has a skip projection — where the
    /// group's `skipProjectionInit` takes effect (`groupsWithSkipProjection`).
    func groupHasSkipProjection(_ group: Int) -> Bool {
        groupsWithSkipProjection.contains(group)
    }

    /// The Neutral init set (the Build screen's "Neutral init" button and
    /// `--derive-model --set-neutral-init`, so the two can never differ):
    /// every option that has a layer to act on starts that path as a no-op —
    /// a near-identity SE gate, a zero last BN γ where a BN follows the
    /// branch's last conv (post-activation), an identity-like skip projection
    /// where one exists, and zero head finals. Options without their layer
    /// keep the standard value (`validate()` requires it there). The draw
    /// prior is deliberately left alone: it is a claim about the training
    /// data, and with a zero value-head final it IS the initial value output,
    /// so it stays an explicit per-experiment choice.
    func withNeutralInit() -> NetworkArchitecture {
        var edited = self
        let projectedGroups = groupsWithSkipProjection
        for index in edited.blockGroups.indices {
            let group = edited.blockGroups[index]
            edited.blockGroups[index].seGammaBiasInit = group.seStyle == .none
                ? BlockGroup.standardSEGammaBiasInit
                : BlockGroup.neutralSEGammaBiasInit
            edited.blockGroups[index].branchOutputInit = group.activationStyle == .post ? .zeroLastBNGamma : .standard
            edited.blockGroups[index].skipProjectionInit = projectedGroups.contains(index) ? .identityLike : .he
        }
        edited.policyHeadFinalInit = .zero
        edited.valueHeadFinalInit = .zero
        return edited
    }

    /// The Standard init set (the Build screen's "Standard init" button):
    /// every option, the draw prior included, at the value every model was
    /// built with before the options existed.
    func withStandardInit() -> NetworkArchitecture {
        var edited = self
        for index in edited.blockGroups.indices {
            edited.blockGroups[index].seGammaBiasInit = BlockGroup.standardSEGammaBiasInit
            edited.blockGroups[index].branchOutputInit = .standard
            edited.blockGroups[index].skipProjectionInit = .he
        }
        edited.policyHeadFinalInit = .he
        edited.valueHeadFinalInit = .he
        edited.valueHeadDrawPrior = Self.standardValueHeadDrawPrior
        return edited
    }

    /// Every option whose value differs from the Standard set, groups in
    /// order (SE γ bias, branch output, skip projection per group), then the
    /// heads. Always compared against the standard value, never against an
    /// earlier edit, so a reopened model or preset still shows what is
    /// non-standard.
    var nonStandardInitOptions: [InitOptionField] {
        var fields: [InitOptionField] = []
        for (index, group) in blockGroups.enumerated() {
            if group.seGammaBiasInit != BlockGroup.standardSEGammaBiasInit { fields.append(.seGammaBiasInit(group: index)) }
            if group.branchOutputInit != .standard { fields.append(.branchOutputInit(group: index)) }
            if group.skipProjectionInit != .he { fields.append(.skipProjectionInit(group: index)) }
        }
        if policyHeadFinalInit != .he { fields.append(.policyHeadFinalInit) }
        if valueHeadFinalInit != .he { fields.append(.valueHeadFinalInit) }
        if valueHeadDrawPrior != Self.standardValueHeadDrawPrior { fields.append(.valueHeadDrawPrior) }
        return fields
    }

    /// The init clause of a group's rendering, e.g. ` init:γb2.2,bnγ0,proj-id`:
    /// present only when the group has a non-standard option, so every
    /// architecture that predates the options renders byte-identically.
    /// Shared by `groupSummary` and the Build screen's diagram.
    static func groupInitMarker(_ g: BlockGroup) -> String {
        var parts: [String] = []
        if g.seGammaBiasInit != BlockGroup.standardSEGammaBiasInit {
            parts.append("γb\(String(format: "%.3g", g.seGammaBiasInit))")
        }
        if g.branchOutputInit == .zeroLastBNGamma { parts.append("bnγ0") }
        if g.skipProjectionInit == .identityLike { parts.append("proj-id") }
        return parts.isEmpty ? "" : " init:" + parts.joined(separator: ",")
    }

    /// The head-init clause of the summary, e.g. ` . init: policy0, value0,
    /// draw 0.6`: present only when a head option is non-standard.
    var headInitMarker: String {
        var parts: [String] = []
        if policyHeadFinalInit == .zero { parts.append("policy0") }
        if valueHeadFinalInit == .zero { parts.append("value0") }
        if valueHeadDrawPrior != Self.standardValueHeadDrawPrior {
            parts.append("draw \(String(format: "%.3g", valueHeadDrawPrior))")
        }
        return parts.isEmpty ? "" : " . init: " + parts.joined(separator: ", ")
    }

    /// The init-neutral checks `validate()` runs: each option needs its layer
    /// (or must hold its standard value without one), the SE γ bias must be
    /// finite, and the draw prior must be a probability strictly inside
    /// (0, 1) on a W/D/L head and the standard value on a scalar one.
    func validateInitOptions() throws {
        let projectedGroups = groupsWithSkipProjection
        for (index, group) in blockGroups.enumerated() {
            guard group.seGammaBiasInit.isFinite else {
                throw NetworkArchitectureError.mustBeFinite(
                    field: "blockGroups[\(index)].seGammaBiasInit", value: group.seGammaBiasInit)
            }
            if group.seStyle == .none, group.seGammaBiasInit != BlockGroup.standardSEGammaBiasInit {
                throw NetworkArchitectureError.initOptionWithoutItsLayer(
                    field: "blockGroups[\(index)].\(BlockGroup.CodingKeys.seGammaBiasInit.rawValue)",
                    value: "\(group.seGammaBiasInit)",
                    reason: "the group has no SE block, so it must be \(BlockGroup.standardSEGammaBiasInit)")
            }
            if group.activationStyle == .pre, group.branchOutputInit != .standard {
                throw NetworkArchitectureError.initOptionWithoutItsLayer(
                    field: "blockGroups[\(index)].\(BlockGroup.CodingKeys.branchOutputInit.rawValue)",
                    value: group.branchOutputInit.rawValue,
                    reason: "a pre-activation block has no BN after its last conv")
            }
            if !projectedGroups.contains(index), group.skipProjectionInit != .he {
                throw NetworkArchitectureError.initOptionWithoutItsLayer(
                    field: "blockGroups[\(index)].\(BlockGroup.CodingKeys.skipProjectionInit.rawValue)",
                    value: group.skipProjectionInit.rawValue,
                    reason: "no block of the group changes width, so it has no skip projection")
            }
        }
        let priorField = CodingKeys.valueHeadDrawPrior.rawValue
        switch valueHeadStyle {
        case .wdlSoftmax:
            guard valueHeadDrawPrior.isFinite, valueHeadDrawPrior > 0, valueHeadDrawPrior < 1 else {
                throw NetworkArchitectureError.initOptionWithoutItsLayer(
                    field: priorField, value: "\(valueHeadDrawPrior)",
                    reason: "a draw probability must lie strictly between 0 and 1")
            }
        case .scalarTanh:
            guard valueHeadDrawPrior == Self.standardValueHeadDrawPrior else {
                throw NetworkArchitectureError.initOptionWithoutItsLayer(
                    field: priorField, value: "\(valueHeadDrawPrior)",
                    reason: "a scalar value head has no draw class, so it must be \(Self.standardValueHeadDrawPrior)")
            }
        }
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
