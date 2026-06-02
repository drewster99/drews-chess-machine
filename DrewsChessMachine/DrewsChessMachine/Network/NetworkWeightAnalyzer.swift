import Foundation

// MARK: - Whole-Network Weight Analyzer
//
// Snapshot-time diagnostic for the entire network's weight tensors.
// Answers: "is everything doing work, is any layer collapsed, is the
// stem looking at the planes it should be, is the policy head's
// output capacity spread across the 76 channels or concentrated in a
// few, are BN channels alive, are SE modules doing real attention?"
//
// Per-variable stats: count, L1/L2 norm, mean|w|, min/max, mean,
// stdev, p10/p50/p90 percentiles, ratio to He-init L2 reference,
// drift-from-init L2 (for deterministic-init variables only).
//
// Per-section aggregates: stem / tower blocks / policy / value
// each get totalElementCount, totalL2Norm, totalInitL2Norm,
// totalL2RatioToInit. Lets you eyeball "block 5 is unusually quiet
// compared to its neighbors" at a glance.
//
// Per-conv per-output-channel L2 for every conv weight tensor.
// Surfaces dead output channels mid-tower (1152 weights per output
// channel in tower convs; ratio to He-init reference tells you which
// channels are effectively zeroed out).
//
// Per-BN dead-channel summary for every BN layer with >1 channel.
// Counts channels with |gamma| < threshold (channels effectively
// zeroed out by BN, since output ≈ beta when gamma ≈ 0).
//
// Per-SE-module baseline attention gate distribution. The scale-and-bias
// SE module's bias-only gate is `sigmoid(gammas_bias)` — the gammas
// (scale) half of the 2·channels-wide FC2 bias, a 128-element vector in
// [0, 1]. Distribution + counts of channels suppressed (<0.1) /
// pass-through (>0.9) tell whether SE is doing real attention vs.
// degenerate. (The betas bias half is a linear offset, not a gate.)
//
// Pure analysis — single `exportWeights()` call + CPU stat-crunching.
// Takes a `ChessNetwork` (the inner type with `trainableVariables`),
// so works against both the champion's network and the trainer's
// network without translation.

enum NetworkWeightAnalyzer {

    // MARK: - Section + fanIn helpers

    /// Canonical section ordering in the JSON output. Block indices
    /// match the code's 0-based numbering — `ChessNetwork` builds
    /// residual blocks via `for i in 0..<ChessNetwork.numBlocks`, so
    /// variable names are `block0_conv1_weights` through
    /// `block<numBlocks-1>_conv1_weights` and the section names mirror
    /// that. Unknown variables fall through to `"other"` so nothing is
    /// silently dropped.
    static let sectionOrder: [String] =
        ["stem"]
        + (0..<ChessNetwork.numBlocks).map { "block_\($0)" }
        + ["tower_final", "policy", "value", "other"]

    /// 0-based section bucket for a variable. Drives both the
    /// per-section summaries and the text-summary grouping.
    static func section(forVariableNamed name: String) -> String {
        if name.hasPrefix("stem_") { return "stem" }
        if name.hasPrefix("tower_final_") { return "tower_final" }
        if name.hasPrefix("policy_") { return "policy" }
        if name.hasPrefix("value_") { return "value" }
        if name.hasPrefix("block") {
            let afterPrefix = name.dropFirst("block".count)
            var digits = ""
            for c in afterPrefix {
                if c.isNumber { digits.append(c) } else { break }
            }
            if !digits.isEmpty { return "block_\(digits)" }
        }
        return "other"
    }

    /// He-init fan-in for a variable, or `nil` for tensors without a
    /// He-init reference (biases, BN gamma/beta, BN running stats).
    /// Mirrors the shapes set in `ChessNetwork`'s graph construction;
    /// if those shapes change the analyzer needs updating in lockstep.
    static func fanIn(forVariableNamed name: String) -> Int? {
        let convArea = ChessNetwork.towerConvKernelSize * ChessNetwork.towerConvKernelSize
        switch name {
        case "stem_conv_weights":      return 30 * convArea
        case "policy_pre_conv_weights": return 128 * 1 * 1
        case "policy_conv_weights":    return 128 * 1 * 1
        case "value_conv_weights":  return ChessNetwork.channels   // 1×1 conv: inC = channels
        case "value_fc1_weights":   return ChessNetwork.boardSize * ChessNetwork.boardSize * ChessNetwork.valueHeadConvChannels  // FC [flatten, hidden], fan_in = flatten
        case "value_fc2_weights":   return ChessNetwork.valueHeadHiddenUnits  // FC [hidden, classes], fan_in = hidden
        default: break
        }
        if name.hasSuffix("_conv1_weights") || name.hasSuffix("_conv2_weights") {
            return 128 * convArea
        }
        if name.hasSuffix("_se_fc1_weights") { return 128 }
        // se_fc2 is Glorot-init, handled by `expectedInitL2` before `fanIn`
        // is ever consulted — so no He fan-in entry here.
        return nil
    }

    /// He-init L2 reference for a tensor of `n` elements drawn from
    /// `N(0, sqrt(2/fanIn))`. Expected L2 = `sqrt(n) · std`.
    static func heInitL2(elementCount n: Int, fanIn: Int) -> Double {
        guard n > 0, fanIn > 0 else { return 0 }
        let std = sqrt(2.0 / Double(fanIn))
        return sqrt(Double(n)) * std
    }

    /// Glorot (Xavier) init L2 reference: `sqrt(n) · sqrt(2/(fanIn+fanOut))`.
    static func glorotInitL2(elementCount n: Int, fanIn: Int, fanOut: Int) -> Double {
        guard n > 0, fanIn + fanOut > 0 else { return 0 }
        let std = sqrt(2.0 / Double(fanIn + fanOut))
        return sqrt(Double(n)) * std
    }

    /// Expected initial L2 norm for a weight tensor, or `nil` for tensors
    /// with no random-init reference. The SE FC2 weight uses Glorot (it
    /// feeds the sigmoid gate — see `ChessNetwork.glorotInitDataFCInOut`),
    /// so it gets the Glorot reference; everything else uses He.
    static func expectedInitL2(forVariableNamed name: String, elementCount n: Int) -> Double? {
        if name.hasSuffix("_se_fc2_weights") {
            // `[in, out]` = [channels/r, 2·channels] = [32, 256].
            return glorotInitL2(elementCount: n, fanIn: 32, fanOut: 256)
        }
        return fanIn(forVariableNamed: name).map { heInitL2(elementCount: n, fanIn: $0) }
    }

    /// Deterministic initial value for variables that don't use He-init.
    /// Returns `nil` for He-init weights (their init was random, so
    /// "drift from init" isn't a meaningful single number). Otherwise
    /// returns the per-element initial values as a `[Double]` of the
    /// same length as the variable's element count. Used to compute
    /// `WeightStats.driftFromInit`.
    static func deterministicInit(
        forVariableNamed name: String,
        elementCount n: Int
    ) -> [Double]? {
        guard n > 0 else { return nil }
        // Special case: value_fc2_bias initializes to [0, ln 6, 0]
        // — see ChessNetwork.valueHead for the rationale (draw-heavy
        // prior of a fresh self-play buffer).
        if name == "value_fc2_bias" && n == 3 {
            return [0.0, log(6.0), 0.0]
        }
        // Per-block ReZero branch scalar α initializes to 1/√numBlocks
        // — see ChessNetwork.residualBlock. Its drift-from-init reference
        // is that value, so the analyzer reports how far each block has
        // grown or shrunk its residual contribution.
        if name.hasSuffix("_res_scale") {
            return Array(repeating: 1.0 / Double(ChessNetwork.numBlocks).squareRoot(), count: n)
        }
        // BN gamma initializes to ones, var initializes to ones —
        // see ChessNetwork.batchNorm.
        if name.hasSuffix("_gamma") || name.hasSuffix("_running_var") {
            return Array(repeating: 1.0, count: n)
        }
        // BN beta, running mean, and FC/conv biases all init to zero.
        if name.hasSuffix("_beta")
            || name.hasSuffix("_running_mean")
            || name.hasSuffix("_bias") {
            return Array(repeating: 0.0, count: n)
        }
        // Weight tensors (conv + FC) use He-init — no deterministic
        // reference. Return nil so the drift field stays nil.
        return nil
    }

    /// Threshold for counting BN channels as "dead." Channels where
    /// `|gamma|` is below this contribute essentially nothing to the
    /// output of the BN layer (post-BN value ≈ beta independent of
    /// input). 0.1 is conservative — production-trained channels
    /// usually have gamma in [0.5, 2.0] range.
    static let bnDeadGammaThreshold: Double = 0.1

    /// Thresholds for SE baseline-gate channel classification.
    /// A channel whose `sigmoid(SE_FC2_bias[c])` is below
    /// `seSuppressedGateThreshold` is approximately zeroed at the
    /// network's "no-input" baseline. Above `sePassThroughGateThreshold`,
    /// it's essentially un-attenuated. Between → SE is contributing
    /// real attention to that channel.
    static let seSuppressedGateThreshold: Double = 0.1
    static let sePassThroughGateThreshold: Double = 0.9

    /// Underused / overused thresholds for per-output-channel ratios
    /// inside conv-detail summaries. A channel with `currentL2 /
    /// initL2 < 0.25` is "weak" — gradients haven't been pushing on
    /// it. `< 0.05` is "dead." `> 2.0` is "overactive."
    static let convDeadRatio: Double = 0.05
    static let convWeakRatio: Double = 0.25

    /// Human-readable per-input-plane label. Order matches
    /// `BoardEncoder.encode`. Used in the stem detail.
    static let inputPlaneLabels: [String] = [
        "mover_pawn", "mover_knight", "mover_bishop",
        "mover_rook", "mover_queen", "mover_king",
        "opp_pawn", "opp_knight", "opp_bishop",
        "opp_rook", "opp_queen", "opp_king",
        "my_kingside_castle", "my_queenside_castle",
        "opp_kingside_castle", "opp_queenside_castle",
        "en_passant", "halfmove_clock",
        "rep_>=1", "rep_>=2",
        "rep_1_ply_ago", "rep_2_plies_ago", "rep_3_plies_ago",
        "rep_4_plies_ago", "rep_5_plies_ago", "rep_6_plies_ago",
        "rep_7_plies_ago", "rep_8_plies_ago", "rep_9_plies_ago",
        "rep_10_plies_ago"
    ]

    // MARK: - Result struct

    struct Result: Codable, Sendable {

        struct WeightStats: Codable, Sendable {
            let name: String
            let elementCount: Int
            let l1Norm: Double
            let l2Norm: Double
            let meanAbs: Double
            let min: Double
            let max: Double
            let mean: Double
            let stdev: Double
            let percentiles: [Double]
            let initL2Norm: Double?
            let l2NormRatioToInit: Double?
            /// L2 norm of `current - initial` for variables with a
            /// deterministic init (biases → init 0, BN gamma → init 1,
            /// BN beta / running_mean → init 0, BN running_var → init 1,
            /// value_fc2_bias → init [0, ln 6, 0]). `nil` for He-init
            /// weight tensors where "drift from init" isn't meaningful.
            /// Directly answers "has this parameter moved from where
            /// the architecture put it?".
            let driftFromInit: Double?
        }

        struct SectionSummary: Codable, Sendable {
            let sectionName: String
            let totalElementCount: Int
            let totalL2Norm: Double
            let totalInitL2Norm: Double?
            let totalL2RatioToInit: Double?
            let variables: [WeightStats]
        }

        struct StemInputChannelDetail: Codable, Sendable {
            let perInputChannelL2: [Double]
            let planeLabels: [String]
            let initPerInputChannelL2: Double
        }

        /// Per-output-channel L2 norm for one conv weight tensor.
        /// Computed for every conv variable in the network so dead
        /// output channels can be spotted regardless of which layer
        /// they're in.
        struct ConvOutputChannelDetail: Codable, Sendable {
            let variableName: String
            /// Length = outC. Indexed by output channel.
            let perOutputChannelL2: [Double]
            /// Per-channel He-init reference. Each output channel of
            /// the conv has `inC × kH × kW` weights drawn from
            /// `N(0, sqrt(2/fanIn))`, so the init L2 is
            /// `sqrt(inC·kH·kW) · sqrt(2/fanIn) = sqrt(2)` for the
            /// usual case where `fanIn = inC·kH·kW`. Reported here so
            /// the JSON consumer doesn't have to recompute it.
            let initPerOutputChannelL2: Double
        }

        /// Dead-channel summary for one BN layer. Genuinely single-channel
        /// BN layers are skipped (a one-element "distribution" has no
        /// shape); every multi-channel BN — including the widened
        /// `value_bn` — is summarized.
        struct BNLayerDetail: Codable, Sendable {
            /// Layer name without the `_gamma`/`_beta` suffix (e.g.
            /// "stem_bn", "block_3_bn2").
            let layerName: String
            let gammaVariableName: String
            let betaVariableName: String
            let channelCount: Int
            /// Count of channels where `|gamma| < deadThreshold`.
            let deadChannelCount: Int
            /// Threshold used (echoed for the JSON reader's benefit).
            let deadThreshold: Double
            /// Percentiles of `|gamma|` across the layer's channels.
            let gammaPercentilesAbs: [Double]
            /// Percentiles of raw `beta` values across the layer's
            /// channels. Useful for spotting BN layers whose beta has
            /// drifted significantly.
            let betaPercentiles: [Double]
        }

        /// SE baseline-attention gate distribution for one residual
        /// block. The gate at "zero input" is `sigmoid(gammas_bias)` —
        /// the scale half of the scale-and-bias FC2 bias — a
        /// 128-element vector in `[0, 1]`. This struct summarizes
        /// its distribution.
        struct SEAttentionDetail: Codable, Sendable {
            let blockName: String
            let seBiasVariableName: String
            let channelCount: Int
            let baselineGateMin: Double
            let baselineGateMean: Double
            let baselineGateMax: Double
            let baselineGatePercentiles: [Double]
            /// Channels whose baseline gate is below
            /// `seSuppressedGateThreshold` — essentially silenced at
            /// the network's "no-input" baseline.
            let channelsSuppressedCount: Int
            let suppressedThreshold: Double
            /// Channels whose baseline gate is above
            /// `sePassThroughGateThreshold` — essentially un-attenuated.
            let channelsPassThroughCount: Int
            let passThroughThreshold: Double
        }

        let producedAtISO8601: String
        let modelLabel: String
        let totalParamCount: Int
        let sections: [SectionSummary]
        let stemInputChannelDetail: StemInputChannelDetail?
        /// Per-output-channel L2 detail for every conv weight tensor
        /// in the network (stem, block_X_conv1, block_X_conv2,
        /// policy_pre_conv, policy_conv, value_conv). Ordered as they
        /// appear in graph build order.
        let convOutputChannelDetails: [ConvOutputChannelDetail]
        /// One entry per multi-channel BN layer (only genuinely
        /// single-channel BN layers are excluded).
        let bnLayerDetails: [BNLayerDetail]
        /// One entry per residual block's SE module — `numBlocks`
        /// entries (block_0 .. block_<numBlocks-1>) in build order.
        let seAttentionDetails: [SEAttentionDetail]
    }

    // MARK: - Entry point

    /// Run the analyzer against `network` (a `ChessNetwork` — works
    /// against both the champion's wrapped network and the trainer's
    /// network without a wrapper hop). `modelLabel` is opaque metadata
    /// the caller chooses and is round-tripped into the result header.
    static func run(
        network: ChessNetwork,
        modelLabel: String
    ) async throws -> Result {
        let weights = try await network.exportWeights()
        let allVariables = network.trainableVariables + network.bnRunningStatsVariables

        guard weights.count == allVariables.count else {
            throw NetworkWeightAnalyzerError.weightCountMismatch(
                expected: allVariables.count,
                got: weights.count
            )
        }

        // Pair (name, values) in build order for downstream lookups.
        var perSection: [String: [(name: String, values: [Float])]] = [:]
        var allByName: [String: [Float]] = [:]
        for (i, variable) in allVariables.enumerated() {
            let name = variable.operation.name
            let sec = section(forVariableNamed: name)
            perSection[sec, default: []].append((name, weights[i]))
            allByName[name] = weights[i]
        }

        // Per-section summaries.
        var sections: [Result.SectionSummary] = []
        for sec in sectionOrder {
            guard let vars = perSection[sec] else { continue }
            sections.append(makeSectionSummary(sectionName: sec, variables: vars))
        }

        // Stem per-input-channel detail.
        let stemInputDetail: Result.StemInputChannelDetail? = {
            guard let stemVars = perSection["stem"],
                  let stem = stemVars.first(where: { $0.name == "stem_conv_weights" }) else {
                return nil
            }
            return makeStemInputChannelDetail(stemConvValues: stem.values)
        }()

        // Per-output-channel L2 for every conv weight tensor — stem,
        // block convs, policy, value. Walk allVariables in original
        // order so the output list matches graph build order.
        var convDetails: [Result.ConvOutputChannelDetail] = []
        for variable in allVariables {
            let name = variable.operation.name
            guard let values = allByName[name] else { continue }
            guard let shape = convShape(forVariableNamed: name) else { continue }
            if let detail = makeConvOutputChannelDetail(
                variableName: name,
                values: values,
                outC: shape.outC,
                inC: shape.inC,
                kH: shape.kH,
                kW: shape.kW
            ) {
                convDetails.append(detail)
            }
        }

        // BN layer details — find every `*_gamma` variable, pair with
        // its `*_beta` sibling, build the summary. Skip genuinely
        // single-channel BN layers, where the percentile distribution
        // has no shape (the widened `value_bn` no longer falls here).
        var bnDetails: [Result.BNLayerDetail] = []
        for variable in allVariables {
            let name = variable.operation.name
            guard name.hasSuffix("_gamma") else { continue }
            // Skip BN running_var (which also ends in _var, but
            // doesn't match _gamma). Defensive check, just in case.
            guard let gammaValues = allByName[name] else { continue }
            let betaName = name.replacingOccurrences(of: "_gamma", with: "_beta")
            guard let betaValues = allByName[betaName] else { continue }
            guard gammaValues.count > 1 else { continue } // skip single-channel BN
            let layerName = String(name.dropLast("_gamma".count))
            bnDetails.append(makeBNLayerDetail(
                layerName: layerName,
                gammaVariableName: name,
                gammaValues: gammaValues,
                betaVariableName: betaName,
                betaValues: betaValues
            ))
        }

        // SE attention detail — one per block, walking blocks in order
        // (matching ChessNetwork's `for i in 0..<ChessNetwork.numBlocks`
        // loop). The SE FC2 bias for a block is named "blockN_se_fc2_bias".
        var seDetails: [Result.SEAttentionDetail] = []
        for blockIndex in 0..<ChessNetwork.numBlocks {
            let seBiasName = "block\(blockIndex)_se_fc2_bias"
            guard let seBiasValues = allByName[seBiasName] else { continue }
            seDetails.append(makeSEAttentionDetail(
                blockName: "block_\(blockIndex)",
                seBiasVariableName: seBiasName,
                seBiasValues: seBiasValues
            ))
        }

        let totalParamCount = sections.reduce(0) { $0 + $1.totalElementCount }
        let iso = ISO8601DateFormatter()
        iso.formatOptions = [.withInternetDateTime]
        return Result(
            producedAtISO8601: iso.string(from: Date()),
            modelLabel: modelLabel,
            totalParamCount: totalParamCount,
            sections: sections,
            stemInputChannelDetail: stemInputDetail,
            convOutputChannelDetails: convDetails,
            bnLayerDetails: bnDetails,
            seAttentionDetails: seDetails
        )
    }

    // MARK: - Section assembly

    static let percentileLabels: [Int] = [10, 50, 90]

    private static func makeSectionSummary(
        sectionName: String,
        variables: [(name: String, values: [Float])]
    ) -> Result.SectionSummary {
        var perVarStats: [Result.WeightStats] = []
        perVarStats.reserveCapacity(variables.count)
        var totalElements = 0
        var heInitSumSq: Double = 0
        var heCurrentSumSq: Double = 0
        var totalSumSq: Double = 0
        var anyHeInit = false

        for (name, values) in variables {
            let stats = makeWeightStats(name: name, values: values)
            perVarStats.append(stats)
            totalElements += stats.elementCount
            totalSumSq += stats.l2Norm * stats.l2Norm
            if let initL2 = stats.initL2Norm {
                anyHeInit = true
                heInitSumSq += initL2 * initL2
                heCurrentSumSq += stats.l2Norm * stats.l2Norm
            }
        }

        let totalL2 = sqrt(totalSumSq)
        let totalInitL2: Double? = anyHeInit ? sqrt(heInitSumSq) : nil
        let totalRatio: Double? = anyHeInit && heInitSumSq > 0
            ? sqrt(heCurrentSumSq) / sqrt(heInitSumSq)
            : nil

        return Result.SectionSummary(
            sectionName: sectionName,
            totalElementCount: totalElements,
            totalL2Norm: totalL2,
            totalInitL2Norm: totalInitL2,
            totalL2RatioToInit: totalRatio,
            variables: perVarStats
        )
    }

    // MARK: - Per-variable stats

    private static func makeWeightStats(
        name: String,
        values: [Float]
    ) -> Result.WeightStats {
        let n = values.count
        guard n > 0 else {
            return Result.WeightStats(
                name: name, elementCount: 0,
                l1Norm: 0, l2Norm: 0, meanAbs: 0,
                min: 0, max: 0, mean: 0, stdev: 0,
                percentiles: Array(repeating: 0, count: percentileLabels.count),
                initL2Norm: nil, l2NormRatioToInit: nil, driftFromInit: nil
            )
        }

        var sum: Double = 0
        var sumAbs: Double = 0
        var sumSq: Double = 0
        var vMin: Double = Double.infinity
        var vMax: Double = -Double.infinity
        for v in values {
            let d = Double(v)
            sum += d
            sumAbs += abs(d)
            sumSq += d * d
            if d < vMin { vMin = d }
            if d > vMax { vMax = d }
        }
        let dN = Double(n)
        let mean = sum / dN
        let variance = max(0.0, (sumSq / dN) - (mean * mean))
        let stdev = sqrt(variance)
        let l2 = sqrt(sumSq)
        let meanAbs = sumAbs / dN

        let sorted = values.map { Double($0) }.sorted()
        let percentiles = percentileLabels.map { p in
            percentile(p: Double(p), sortedAscending: sorted)
        }

        let initL2 = expectedInitL2(forVariableNamed: name, elementCount: n)
        let ratio = initL2.map { $0 > 0 ? l2 / $0 : 0 }

        // Drift from init — only meaningful for variables with
        // deterministic initial values.
        let drift: Double? = {
            guard let initial = deterministicInit(forVariableNamed: name, elementCount: n) else {
                return nil
            }
            var driftSq: Double = 0
            for i in 0..<n {
                let d = Double(values[i]) - initial[i]
                driftSq += d * d
            }
            return sqrt(driftSq)
        }()

        return Result.WeightStats(
            name: name,
            elementCount: n,
            l1Norm: sumAbs,
            l2Norm: l2,
            meanAbs: meanAbs,
            min: vMin,
            max: vMax,
            mean: mean,
            stdev: stdev,
            percentiles: percentiles,
            initL2Norm: initL2,
            l2NormRatioToInit: ratio,
            driftFromInit: drift
        )
    }

    // MARK: - Detail builders

    /// Shape (OIHW) of a conv weight tensor, looked up by name.
    /// Returns `nil` for non-conv variables. Used by the per-output-
    /// channel detail walker so it can iterate every conv tensor
    /// without needing to inspect the MPSGraphTensor shape directly.
    private static func convShape(
        forVariableNamed name: String
    ) -> (outC: Int, inC: Int, kH: Int, kW: Int)? {
        let k = ChessNetwork.towerConvKernelSize
        switch name {
        case "stem_conv_weights":       return (128, 30, k, k)
        case "policy_pre_conv_weights": return (128, 128, 1, 1)
        case "policy_conv_weights":     return (76, 128, 1, 1)
        case "value_conv_weights":      return (ChessNetwork.valueHeadConvChannels, 128, 1, 1)
        default: break
        }
        if name.hasSuffix("_conv1_weights") || name.hasSuffix("_conv2_weights") {
            return (128, 128, k, k)
        }
        return nil
    }

    private static func makeStemInputChannelDetail(stemConvValues: [Float]) -> Result.StemInputChannelDetail? {
        let outC = 128, inC = 30
        let kH = ChessNetwork.towerConvKernelSize, kW = ChessNetwork.towerConvKernelSize
        let expected = outC * inC * kH * kW
        guard stemConvValues.count == expected else { return nil }

        var perInputSumSq = [Double](repeating: 0, count: inC)
        let strideO = inC * kH * kW
        let strideI = kH * kW
        for o in 0..<outC {
            for i in 0..<inC {
                let base = o * strideO + i * strideI
                for hw in 0..<(kH * kW) {
                    let v = Double(stemConvValues[base + hw])
                    perInputSumSq[i] += v * v
                }
            }
        }
        let perInputL2 = perInputSumSq.map { sqrt($0) }
        let initPerInputL2 = heInitL2(
            elementCount: outC * kH * kW,
            fanIn: inC * kH * kW
        )
        return Result.StemInputChannelDetail(
            perInputChannelL2: perInputL2,
            planeLabels: Array(inputPlaneLabels.prefix(inC)),
            initPerInputChannelL2: initPerInputL2
        )
    }

    private static func makeConvOutputChannelDetail(
        variableName: String,
        values: [Float],
        outC: Int,
        inC: Int,
        kH: Int,
        kW: Int
    ) -> Result.ConvOutputChannelDetail? {
        let expected = outC * inC * kH * kW
        guard values.count == expected else { return nil }
        let perOutSize = inC * kH * kW   // weights per output channel
        var perOutputSumSq = [Double](repeating: 0, count: outC)
        // OIHW: data[o * perOutSize + idx] for idx in 0..<perOutSize
        for o in 0..<outC {
            let base = o * perOutSize
            for k in 0..<perOutSize {
                let v = Double(values[base + k])
                perOutputSumSq[o] += v * v
            }
        }
        let perOutputL2 = perOutputSumSq.map { sqrt($0) }
        // Each output channel has `perOutSize` weights drawn from
        // N(0, sqrt(2/fanIn)) where fanIn = perOutSize (for stem/
        // tower convs) or inC (for 1x1 policy/value convs). In the
        // OIHW conv case fanIn always equals perOutSize anyway.
        let initPerOutputL2 = heInitL2(elementCount: perOutSize, fanIn: perOutSize)
        return Result.ConvOutputChannelDetail(
            variableName: variableName,
            perOutputChannelL2: perOutputL2,
            initPerOutputChannelL2: initPerOutputL2
        )
    }

    private static func makeBNLayerDetail(
        layerName: String,
        gammaVariableName: String,
        gammaValues: [Float],
        betaVariableName: String,
        betaValues: [Float]
    ) -> Result.BNLayerDetail {
        let gammaAbs = gammaValues.map { abs(Double($0)) }
        let beta = betaValues.map { Double($0) }
        let sortedGammaAbs = gammaAbs.sorted()
        let sortedBeta = beta.sorted()
        let gammaPct = percentileLabels.map {
            percentile(p: Double($0), sortedAscending: sortedGammaAbs)
        }
        let betaPct = percentileLabels.map {
            percentile(p: Double($0), sortedAscending: sortedBeta)
        }
        var deadCount = 0
        for g in gammaAbs where g < bnDeadGammaThreshold { deadCount += 1 }
        return Result.BNLayerDetail(
            layerName: layerName,
            gammaVariableName: gammaVariableName,
            betaVariableName: betaVariableName,
            channelCount: gammaValues.count,
            deadChannelCount: deadCount,
            deadThreshold: bnDeadGammaThreshold,
            gammaPercentilesAbs: gammaPct,
            betaPercentiles: betaPct
        )
    }

    private static func makeSEAttentionDetail(
        blockName: String,
        seBiasVariableName: String,
        seBiasValues: [Float]
    ) -> Result.SEAttentionDetail {
        // The scale-and-bias SE FC2 bias is `2·channels` wide: the first
        // `channels` entries are the `gammas` (scale) half that feeds the
        // sigmoid gate; the rest are the `betas` (linear bias) half. The
        // baseline gate is `sigmoid(gammas_bias)`, so slice the first half.
        let half = seBiasValues.count / 2
        let gammaBias = half > 0 ? Array(seBiasValues.prefix(half)) : seBiasValues
        // sigmoid(x) = 1 / (1 + exp(-x))
        let gates: [Double] = gammaBias.map { v in
            let d = Double(v)
            return 1.0 / (1.0 + exp(-d))
        }
        let n = gates.count
        let sum = gates.reduce(0, +)
        let mean = n > 0 ? sum / Double(n) : 0
        let gMin = gates.min() ?? 0
        let gMax = gates.max() ?? 0
        let sortedGates = gates.sorted()
        let pct = percentileLabels.map {
            percentile(p: Double($0), sortedAscending: sortedGates)
        }
        var suppressed = 0
        var passThrough = 0
        for g in gates {
            if g < seSuppressedGateThreshold { suppressed += 1 }
            if g > sePassThroughGateThreshold { passThrough += 1 }
        }
        return Result.SEAttentionDetail(
            blockName: blockName,
            seBiasVariableName: seBiasVariableName,
            channelCount: n,
            baselineGateMin: gMin,
            baselineGateMean: mean,
            baselineGateMax: gMax,
            baselineGatePercentiles: pct,
            channelsSuppressedCount: suppressed,
            suppressedThreshold: seSuppressedGateThreshold,
            channelsPassThroughCount: passThrough,
            passThroughThreshold: sePassThroughGateThreshold
        )
    }

    // MARK: - Numeric helpers

    /// Linear-interpolation percentile over ascending-sorted input.
    static func percentile(p: Double, sortedAscending: [Double]) -> Double {
        guard !sortedAscending.isEmpty else { return 0 }
        if sortedAscending.count == 1 { return sortedAscending[0] }
        let pos = (p / 100.0) * Double(sortedAscending.count - 1)
        let lo = Int(floor(pos))
        let hi = Int(ceil(pos))
        if lo == hi { return sortedAscending[lo] }
        let w = pos - Double(lo)
        return sortedAscending[lo] * (1.0 - w) + sortedAscending[hi] * w
    }
}

// MARK: - Error type

enum NetworkWeightAnalyzerError: LocalizedError {
    case weightCountMismatch(expected: Int, got: Int)

    var errorDescription: String? {
        switch self {
        case .weightCountMismatch(let expected, let got):
            return "NetworkWeightAnalyzer: exportWeights() returned \(got) tensors, expected \(expected)"
        }
    }
}

// MARK: - Text summary

extension NetworkWeightAnalyzer.Result {

    /// Multi-line digest of the result. JSON has the full data; this
    /// is the glanceable form for the session log and the NSAlert.
    func textSummary() -> String {
        var out = ""

        out += "Network weight analysis (model: \(modelLabel))\n"
        out += "  produced:   \(producedAtISO8601)\n"
        out += "  total params: \(formatInt(totalParamCount))\n\n"

        // Section-level summary table.
        out += "Per-section summary:\n"
        out += "  section         params      L2          init_L2     ratio\n"
        for s in sections {
            let initStr = s.totalInitL2Norm.map { String(format: "%9.3f", $0) } ?? "    --   "
            let ratioStr = s.totalL2RatioToInit.map { String(format: "%6.3f", $0) } ?? "  --  "
            out += String(
                format: "  %@  %@  %@  %@  %@\n",
                s.sectionName.padded(14),
                formatInt(s.totalElementCount).leftPadded(toLength: 10),
                String(format: "%9.3f", s.totalL2Norm).leftPadded(toLength: 10),
                initStr.leftPadded(toLength: 10),
                ratioStr.leftPadded(toLength: 7)
            )
        }
        out += "\n"

        // Per-variable detail per section.
        for s in sections {
            out += "Section: \(s.sectionName) (\(formatInt(s.totalElementCount)) params)\n"
            out += "  variable                              count      L2       init_L2  ratio   drift   mean|w|   min       max\n"
            for v in s.variables {
                let initStr = v.initL2Norm.map { String(format: "%7.3f", $0) } ?? "  -- "
                let ratioStr = v.l2NormRatioToInit.map { String(format: "%6.3f", $0) } ?? "  --  "
                let driftStr = v.driftFromInit.map { String(format: "%6.3f", $0) } ?? "  --  "
                out += String(
                    format: "  %@  %@  %@  %@  %@  %@  %@  %@  %@\n",
                    v.name.padded(38),
                    formatInt(v.elementCount).leftPadded(toLength: 7),
                    String(format: "%8.3f", v.l2Norm).leftPadded(toLength: 8),
                    initStr.leftPadded(toLength: 8),
                    ratioStr.leftPadded(toLength: 7),
                    driftStr.leftPadded(toLength: 7),
                    String(format: "%7.4f", v.meanAbs).leftPadded(toLength: 8),
                    String(format: "%+7.3f", v.min).leftPadded(toLength: 9),
                    String(format: "%+7.3f", v.max).leftPadded(toLength: 9)
                )
            }
            out += "\n"
        }

        // Stem per-input-channel detail.
        if let stem = stemInputChannelDetail {
            out += "Stem per-input-channel L2 (init ref \(String(format: "%.3f", stem.initPerInputChannelL2)) per plane):\n"
            for (i, l2) in stem.perInputChannelL2.enumerated() {
                let label = i < stem.planeLabels.count ? stem.planeLabels[i] : "plane_\(i)"
                let ratio = stem.initPerInputChannelL2 > 0 ? l2 / stem.initPerInputChannelL2 : 0
                out += String(
                    format: "  %@ %@   L2=%6.3f   ratio=%6.3f\n",
                    String(format: "%2d", i),
                    label.padded(26),
                    l2,
                    ratio
                )
            }
            out += "\n"
        }

        // Per-conv per-output-channel summary: dead/weak channel
        // counts per conv tensor instead of dumping every channel's
        // L2 (which would be 24+ × 128 = 3000+ rows for the tower).
        // JSON has the full per-channel arrays.
        if !convOutputChannelDetails.isEmpty {
            out += "Per-conv output-channel health (dead/weak counts vs init ref):\n"
            out += "  variable                              channels  dead(<\(String(format: "%.2f", NetworkWeightAnalyzer.convDeadRatio)))  weak(<\(String(format: "%.2f", NetworkWeightAnalyzer.convWeakRatio)))  p10ratio  p50ratio  p90ratio  initL2\n"
            for c in convOutputChannelDetails {
                let initL2 = c.initPerOutputChannelL2
                let ratios = c.perOutputChannelL2.map { initL2 > 0 ? $0 / initL2 : 0 }
                let dead = ratios.filter { $0 < NetworkWeightAnalyzer.convDeadRatio }.count
                let weak = ratios.filter { $0 < NetworkWeightAnalyzer.convWeakRatio }.count
                let sortedRatios = ratios.sorted()
                let p10 = NetworkWeightAnalyzer.percentile(p: 10, sortedAscending: sortedRatios)
                let p50 = NetworkWeightAnalyzer.percentile(p: 50, sortedAscending: sortedRatios)
                let p90 = NetworkWeightAnalyzer.percentile(p: 90, sortedAscending: sortedRatios)
                out += String(
                    format: "  %@  %@  %@  %@  %@  %@  %@  %@\n",
                    c.variableName.padded(38),
                    formatInt(c.perOutputChannelL2.count).leftPadded(toLength: 8),
                    formatInt(dead).leftPadded(toLength: 7),
                    formatInt(weak).leftPadded(toLength: 7),
                    String(format: "%6.3f", p10).leftPadded(toLength: 8),
                    String(format: "%6.3f", p50).leftPadded(toLength: 8),
                    String(format: "%6.3f", p90).leftPadded(toLength: 8),
                    String(format: "%6.3f", initL2).leftPadded(toLength: 8)
                )
            }
            out += "\n"
        }

        // BN layer dead-channel summary.
        if !bnLayerDetails.isEmpty {
            out += "BN layer dead-channel summary (dead = |gamma| < \(String(format: "%.2f", bnLayerDetails.first?.deadThreshold ?? 0))):\n"
            out += "  layer                  ch   dead   |gamma|p10  |gamma|p50  |gamma|p90  beta p10    beta p50    beta p90\n"
            for b in bnLayerDetails {
                out += String(
                    format: "  %@  %@  %@  %@  %@  %@  %@  %@  %@\n",
                    b.layerName.padded(20),
                    formatInt(b.channelCount).leftPadded(toLength: 4),
                    formatInt(b.deadChannelCount).leftPadded(toLength: 5),
                    String(format: "%6.3f", b.gammaPercentilesAbs[safe: 0] ?? 0).leftPadded(toLength: 10),
                    String(format: "%6.3f", b.gammaPercentilesAbs[safe: 1] ?? 0).leftPadded(toLength: 10),
                    String(format: "%6.3f", b.gammaPercentilesAbs[safe: 2] ?? 0).leftPadded(toLength: 10),
                    String(format: "%+6.3f", b.betaPercentiles[safe: 0] ?? 0).leftPadded(toLength: 10),
                    String(format: "%+6.3f", b.betaPercentiles[safe: 1] ?? 0).leftPadded(toLength: 10),
                    String(format: "%+6.3f", b.betaPercentiles[safe: 2] ?? 0).leftPadded(toLength: 10)
                )
            }
            out += "\n"
        }

        // SE attention baseline gate distribution per block.
        if !seAttentionDetails.isEmpty {
            let supT = seAttentionDetails.first?.suppressedThreshold ?? 0
            let passT = seAttentionDetails.first?.passThroughThreshold ?? 0
            out += "SE attention baseline gates (sigmoid of SE_FC2 bias, channelwise):\n"
            out += "  block      ch    gate_min  gate_mean  gate_max  gate_p10  gate_p50  gate_p90  suppr(<\(String(format: "%.2f", supT)))  pass(>\(String(format: "%.2f", passT)))\n"
            for s in seAttentionDetails {
                out += String(
                    format: "  %@  %@  %@  %@  %@  %@  %@  %@  %@  %@\n",
                    s.blockName.padded(8),
                    formatInt(s.channelCount).leftPadded(toLength: 4),
                    String(format: "%6.3f", s.baselineGateMin).leftPadded(toLength: 8),
                    String(format: "%6.3f", s.baselineGateMean).leftPadded(toLength: 9),
                    String(format: "%6.3f", s.baselineGateMax).leftPadded(toLength: 8),
                    String(format: "%6.3f", s.baselineGatePercentiles[safe: 0] ?? 0).leftPadded(toLength: 8),
                    String(format: "%6.3f", s.baselineGatePercentiles[safe: 1] ?? 0).leftPadded(toLength: 8),
                    String(format: "%6.3f", s.baselineGatePercentiles[safe: 2] ?? 0).leftPadded(toLength: 8),
                    formatInt(s.channelsSuppressedCount).leftPadded(toLength: 7),
                    formatInt(s.channelsPassThroughCount).leftPadded(toLength: 6)
                )
            }
        }

        return out
    }

    private func formatInt(_ n: Int) -> String {
        let f = NumberFormatter()
        f.numberStyle = .decimal
        f.usesGroupingSeparator = true
        return f.string(from: NSNumber(value: n)) ?? "\(n)"
    }
}

// MARK: - String padding helpers

private extension String {
    func padded(_ length: Int) -> String {
        if count >= length { return self }
        return self + String(repeating: " ", count: length - count)
    }
    func leftPadded(toLength length: Int) -> String {
        if count >= length { return self }
        return String(repeating: " ", count: length - count) + self
    }
}

private extension Array {
    /// Safe subscript that returns `nil` for out-of-range indices.
    /// Used in the text-summary formatter to defensively pull
    /// percentile values without crashing if the array length is
    /// somehow off (shouldn't happen, but the formatter is best-effort
    /// and shouldn't take down the analyzer's log output).
    subscript(safe index: Int) -> Element? {
        return (0..<count).contains(index) ? self[index] : nil
    }
}
