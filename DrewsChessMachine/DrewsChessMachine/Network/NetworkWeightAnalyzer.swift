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
// stdev, p10/p50/p90 percentiles, the tensor's initial L2 norm and the
// ratio to it, and drift from init where the initial values are known
// exactly. BN running statistics get no init figures (their start depends
// on how the model was built).
//
// Every "init" figure comes from the snapshot's init reference
// (`AnalysisInitReference`): the architecture built by the network builder
// itself, under the model's own init seed when it is recorded. So a model
// built with another final init, draw prior, branch or skip init, or SE bias
// init is compared with its own starting point, and no init rule is copied
// here. A figure that includes a same-distribution draw rather than the
// model's own initial values is flagged (`initExact` false; "~" in the text
// summary).
//
// Per-section aggregates: stem / tower blocks / policy / value
// each get totalElementCount (every element, BN running statistics
// included), and totalL2Norm and totalInitL2Norm over their trainable
// tensors only (BN running statistics, which are not weights, are reported
// apart as runningStatsL2Norm), and totalL2RatioToInit. Lets you
// eyeball "block 5 is unusually quiet compared to its neighbors" at a
// glance.
//
// Per-conv per-output-channel L2 for every conv weight tensor, each
// channel against its own initial L2. Surfaces dead output channels
// mid-tower.
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
// Pure analysis over an `AnalyzedNetworkSnapshot` (one export, shared with
// the other analyses of the same request) + CPU stat-crunching; works for
// the champion and the trainer alike.

enum NetworkWeightAnalyzer {

    // MARK: - Section + fanIn helpers

    /// Canonical section ordering in the JSON output. Block indices
    /// match the code's 0-based numbering — `ChessNetwork` builds
    /// residual blocks via `for i in 0..<arch.numBlocks`, so
    /// variable names are `block0_conv1_weights` through
    /// `block<numBlocks-1>_conv1_weights` and the section names mirror
    /// that. Unknown variables fall through to `"other"` so nothing is
    /// silently dropped.
    static func sectionOrder(numBlocks: Int) -> [String] {
        ["stem"]
        + (0..<numBlocks).map { "block_\($0)" }
        + ["tower_final", "feature_skip", "policy", "value", "other"]
    }

    /// 0-based section bucket for a variable. Drives both the
    /// per-section summaries and the text-summary grouping.
    static func section(forVariableNamed name: String) -> String {
        if name.hasPrefix("stem_") { return "stem" }
        if name.hasPrefix("tower_final_") { return "tower_final" }
        if name.hasPrefix("feature_skip_") { return "feature_skip" }
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

    /// The expanded block spec + its INPUT width for a `block<i>_*` variable
    /// name, or `nil` for non-block names. Per-block fan-ins must come from
    /// each block's OWN kernels and widths — with block groups the uniform
    /// assumption is wrong, not just stale (this exact bug class shipped
    /// once, ≤ build 1566, as a stale kernel area).
    static func blockSpec(
        forVariableNamed name: String, arch: NetworkArchitecture
    ) -> (spec: BlockGroup, inChannels: Int)? {
        guard name.hasPrefix("block") else { return nil }
        var digits = ""
        for ch in name.dropFirst("block".count) {
            if ch.isNumber { digits.append(ch) } else { break }
        }
        guard let i = Int(digits) else { return nil }
        let expanded = arch.expandedBlocks
        guard i < expanded.count else { return nil }
        // Fold in the final-block feature skip (+ source on the last block under a
        // routed concatDirect skip) so the analyzer's conv1/skip_proj fan-in and shape
        // match the builder/plan; zero for every other block.
        let baseInC = i == 0 ? arch.stemOutputChannels : expanded[i - 1].channels
        let inC = baseInC + arch.blockSkipExtraInputChannels(blockIndex: i)
        return (expanded[i], inC)
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
            /// A BN running statistic: no init figures (`AnalysisInitReference`).
            let isRunningStatistic: Bool
            /// The tensor's L2 norm at init, from the init reference: exact
            /// when `initExact`, else a draw of the same distribution; nil
            /// for a BN running statistic.
            let initL2Norm: Double?
            /// `l2Norm / initL2Norm`; nil when the tensor started at zero or
            /// is a BN running statistic.
            let l2NormRatioToInit: Double?
            /// Whether the initial values are known exactly (the model's own
            /// init seed, or a tensor the build does not draw from the seed);
            /// false for a BN running statistic.
            let initExact: Bool
            /// L2 norm of `current - initial` when `initExact`; nil
            /// otherwise. Directly answers "has this parameter moved from
            /// where the architecture put it?".
            let driftFromInit: Double?
        }

        struct SectionSummary: Codable, Sendable {
            let sectionName: String
            /// Every element of the section, BN running statistics included
            /// (the project's `parameterCount` counts them too).
            let totalElementCount: Int
            /// Over the section's trainable tensors only.
            let totalL2Norm: Double
            /// Over the section's BN running statistics, which are not
            /// weights; nil for a section with none.
            let runningStatsL2Norm: Double?
            /// The trainable tensors' L2 norm at init (init reference).
            let totalInitL2Norm: Double
            /// `totalL2Norm / totalInitL2Norm`; nil when everything started
            /// at zero.
            let totalL2RatioToInit: Double?
            /// Whether every trainable's initial values are known exactly;
            /// false means `totalInitL2Norm` and its ratio include
            /// same-distribution draws.
            let initExact: Bool
            let variables: [WeightStats]
        }

        struct StemInputChannelDetail: Codable, Sendable {
            let perInputChannelL2: [Double]
            let planeLabels: [String]
            /// Each input plane's L2 at init (init reference).
            let initPerInputChannelL2: [Double]
            /// Whether `initPerInputChannelL2` is the model's own start (else
            /// a draw of the same distribution).
            let initExact: Bool
        }

        /// Per-output-channel L2 norm for one conv weight tensor.
        /// Computed for every conv variable in the network so dead
        /// output channels can be spotted regardless of which layer
        /// they're in.
        struct ConvOutputChannelDetail: Codable, Sendable {
            let variableName: String
            /// Length = outC. Indexed by output channel.
            let perOutputChannelL2: [Double]
            /// Each output channel's L2 at init (init reference): about √2
            /// for a He-normal conv, 0 for a zero-initialized head, 1 or 0
            /// per channel for an identity-like skip projection.
            let initPerOutputChannelL2: [Double]
            /// Whether `initPerOutputChannelL2` is the model's own start
            /// (else a draw of the same distribution).
            let initExact: Bool
        }

        /// Dead-channel summary for one BN layer. Genuinely single-channel
        /// BN layers are skipped (a one-element "distribution" has no
        /// shape); every multi-channel BN — including the widened
        /// `value_bn` — is summarized.
        struct BNLayerDetail: Codable, Sendable {
            /// Layer name without the `_gamma`/`_beta` suffix (e.g.
            /// "stem_bn", "block0_bn2").
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

        /// Whose weights these are and from when (`analyzedWeights`), plus
        /// the session's context. Stamped on by `SessionController` at
        /// export time; nil (key omitted) for analyzer-only callers.
        var exportMetadata: AnalysisExportMetadata? = nil
        /// Every element, BN running statistics included (the project's
        /// `parameterCount` definition).
        let totalParamCount: Int
        let sections: [SectionSummary]
        let stemInputChannelDetail: StemInputChannelDetail?
        /// Per-output-channel L2 detail for every conv weight tensor
        /// in the network (stem, block_X_conv1, block_X_conv2,
        /// policy_pre_conv, policy_conv, value_conv). Ordered as they
        /// appear in graph build order.
        let convOutputChannelDetails: [ConvOutputChannelDetail]
        /// One entry per multi-channel normalization layer with a γ/β pair
        /// (every `*_gamma` variable): the BN layers and the residual
        /// LayerNorms (`res_ln`). Only genuinely single-channel layers are
        /// excluded.
        let bnLayerDetails: [BNLayerDetail]
        /// One entry per residual block's SE module — `numBlocks`
        /// entries (block_0 .. block_<numBlocks-1>) in build order.
        let seAttentionDetails: [SEAttentionDetail]
    }

    // MARK: - Entry point

    /// `run(snapshot:modelLabel:)` on a GCD queue: it scans and sorts every
    /// weight, long synchronous work that must not hold a cooperative thread.
    static func runOffPool(snapshot: AnalyzedNetworkSnapshot, modelLabel: String) async throws -> Result {
        try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .utility).async {
                continuation.resume(with: Swift.Result(catching: { try run(snapshot: snapshot, modelLabel: modelLabel) }))
            }
        }
    }

    /// Run the analyzer on `snapshot`. `modelLabel` is opaque metadata the
    /// caller chooses and is round-tripped into the result header.
    static func run(snapshot: AnalyzedNetworkSnapshot, modelLabel: String) throws -> Result {
        let arch = snapshot.architecture

        // Pair (name, values, start) in build order for downstream lookups.
        var perSection: [String: [VariableValues]] = [:]
        var allByName: [String: [Float]] = [:]
        var trainableStartByName: [String: (initial: [Float], exact: Bool)] = [:]
        for (i, name) in snapshot.names.enumerated() {
            let start = try snapshot.start(ofVariableAt: i)
            if case .trainable(let initial, let exact) = start {
                trainableStartByName[name] = (initial, exact)
            }
            perSection[section(forVariableNamed: name), default: []].append(
                VariableValues(name: name, values: snapshot.weights[i], start: start))
            allByName[name] = snapshot.weights[i]
        }

        // Per-section summaries.
        var sections: [Result.SectionSummary] = []
        for sec in sectionOrder(numBlocks: arch.numBlocks) {
            guard let vars = perSection[sec] else { continue }
            sections.append(makeSectionSummary(sectionName: sec, variables: vars))
        }

        // Stem per-input-channel detail.
        let stemInputDetail: Result.StemInputChannelDetail? = {
            guard let values = allByName["stem_conv_weights"],
                  let start = trainableStartByName["stem_conv_weights"] else { return nil }
            return makeStemInputChannelDetail(stemConvValues: values, initialValues: start.initial,
                                              initExact: start.exact, arch: arch)
        }()

        // Per-output-channel L2 for every conv weight tensor — stem,
        // block convs, policy, value. Walk the names in export order so
        // the output list matches graph build order.
        var convDetails: [Result.ConvOutputChannelDetail] = []
        for name in snapshot.names {
            guard let values = allByName[name], let start = trainableStartByName[name] else { continue }
            guard let shape = convShape(forVariableNamed: name, arch: arch) else { continue }
            if let detail = makeConvOutputChannelDetail(
                variableName: name,
                values: values,
                initialValues: start.initial,
                initExact: start.exact,
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
        for name in snapshot.names {
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
        // (matching ChessNetwork's `for i in 0..<arch.numBlocks`
        // loop). The SE FC2 bias for a block is named "blockN_se_fc2_bias".
        var seDetails: [Result.SEAttentionDetail] = []
        for blockIndex in 0..<arch.numBlocks {
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

    /// One variable's current values and where it started.
    private struct VariableValues {
        let name: String
        let values: [Float]
        let start: AnalysisInitReference.Start
    }

    private static func sumOfSquares(_ values: [Float]) -> Double {
        values.reduce(0.0) { $0 + Double($1) * Double($1) }
    }

    private static func makeSectionSummary(
        sectionName: String,
        variables: [VariableValues]
    ) -> Result.SectionSummary {
        var perVarStats: [Result.WeightStats] = []
        perVarStats.reserveCapacity(variables.count)
        var totalElements = 0
        var trainableSumSq: Double = 0
        var trainableInitSumSq: Double = 0
        var runningSumSq: Double = 0
        var anyRunningStatistic = false
        var allTrainablesExact = true

        for variable in variables {
            let stats = makeWeightStats(variable)
            perVarStats.append(stats)
            totalElements += stats.elementCount
            switch variable.start {
            case .runningStatistic:
                anyRunningStatistic = true
                runningSumSq += stats.l2Norm * stats.l2Norm
            case .trainable(let initial, let exact):
                trainableSumSq += stats.l2Norm * stats.l2Norm
                trainableInitSumSq += sumOfSquares(initial)
                allTrainablesExact = allTrainablesExact && exact
            }
        }

        let totalL2 = sqrt(trainableSumSq)
        let totalInitL2 = sqrt(trainableInitSumSq)
        return Result.SectionSummary(
            sectionName: sectionName,
            totalElementCount: totalElements,
            totalL2Norm: totalL2,
            runningStatsL2Norm: anyRunningStatistic ? sqrt(runningSumSq) : nil,
            totalInitL2Norm: totalInitL2,
            totalL2RatioToInit: totalInitL2 > 0 ? totalL2 / totalInitL2 : nil,
            initExact: allTrainablesExact,
            variables: perVarStats
        )
    }

    // MARK: - Per-variable stats

    private static func makeWeightStats(_ variable: VariableValues) -> Result.WeightStats {
        let name = variable.name
        let values = variable.values
        let n = values.count
        let initial: [Float]?
        let initExact: Bool
        switch variable.start {
        case .trainable(let startValues, let exact):
            initial = startValues
            initExact = exact
        case .runningStatistic:
            initial = nil
            initExact = false
        }
        let isRunningStatistic = initial == nil
        let initL2: Double? = initial.map { sqrt(sumOfSquares($0)) }
        guard n > 0 else {
            return Result.WeightStats(
                name: name, elementCount: 0,
                l1Norm: 0, l2Norm: 0, meanAbs: 0,
                min: 0, max: 0, mean: 0, stdev: 0,
                percentiles: Array(repeating: 0, count: percentileLabels.count),
                isRunningStatistic: isRunningStatistic, initL2Norm: initL2, l2NormRatioToInit: nil,
                initExact: initExact, driftFromInit: initExact ? 0 : nil
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

        let ratio: Double? = initL2.flatMap { $0 > 0 ? l2 / $0 : nil }

        // Drift from init — only where the initial values are known exactly.
        let drift: Double? = {
            guard initExact, let initial else { return nil }
            var driftSq: Double = 0
            for i in 0..<n {
                let d = Double(values[i]) - Double(initial[i])
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
            isRunningStatistic: isRunningStatistic,
            initL2Norm: initL2,
            l2NormRatioToInit: ratio,
            initExact: initExact,
            driftFromInit: drift
        )
    }

    // MARK: - Detail builders

    /// Shape (OIHW) of a conv weight tensor, looked up by name.
    /// Returns `nil` for non-conv variables. Used by the per-output-
    /// channel detail walker so it can iterate every conv tensor
    /// without needing to inspect the MPSGraphTensor shape directly.
    private static func convShape(
        forVariableNamed name: String,
        arch: NetworkArchitecture
    ) -> (outC: Int, inC: Int, kH: Int, kW: Int)? {
        let c0 = arch.stemOutputChannels
        // The policy/value FIRST convs read the head INPUT width (widened by a
        // routed concatDirect feature skip; == towerOutputChannels when off).
        switch name {
        case "stem_conv_weights":       return (c0, arch.inputPlanes, arch.stemConvKernelSize, arch.stemConvKernelSize)
        case "policy_pre_conv_weights": return (arch.policyPreConvChannels, arch.policyHeadInputChannels, 1, 1)
        case "policy_conv_weights":
            return arch.policyHeadStyle == .simpleConv
                ? (ChessNetwork.policyChannels, arch.policyHeadInputChannels, 1, 1)
                : (ChessNetwork.policyChannels, arch.policyPreConvChannels, 1, 1)
        case "value_conv_weights":      return (arch.valueHeadConvChannels, arch.valueHeadInputChannels, 1, 1)
        case "feature_skip_conv_weights": return (arch.towerOutputChannels, arch.featureSkipCompressInputChannels, 1, 1)
        default: break
        }
        if let (spec, inC) = blockSpec(forVariableNamed: name, arch: arch) {
            if name.hasSuffix("_conv1_weights") {
                return (spec.channels, inC, spec.conv1KernelSize, spec.conv1KernelSize)
            }
            if name.hasSuffix("_conv2_weights") {
                return (spec.channels, spec.channels, spec.conv2KernelSize, spec.conv2KernelSize)
            }
            if name.hasSuffix("_skip_proj_weights") {
                return (spec.channels, inC, 1, 1)
            }
        }
        return nil
    }

    private static func makeStemInputChannelDetail(
        stemConvValues: [Float],
        initialValues: [Float],
        initExact: Bool,
        arch: NetworkArchitecture
    ) -> Result.StemInputChannelDetail? {
        let outC = arch.stemOutputChannels, inC = arch.inputPlanes
        let kH = arch.stemConvKernelSize, kW = arch.stemConvKernelSize
        let expected = outC * inC * kH * kW
        guard stemConvValues.count == expected, initialValues.count == expected else { return nil }
        return Result.StemInputChannelDetail(
            perInputChannelL2: perInputPlaneL2(stemConvValues, outC: outC, inC: inC, kH: kH, kW: kW),
            planeLabels: Array(arch.inputEncoding.analyzerPlaneLabels.prefix(inC)),
            initPerInputChannelL2: perInputPlaneL2(initialValues, outC: outC, inC: inC, kH: kH, kW: kW),
            initExact: initExact
        )
    }

    /// Each input plane's L2 norm over an OIHW conv tensor.
    private static func perInputPlaneL2(_ values: [Float], outC: Int, inC: Int, kH: Int, kW: Int) -> [Double] {
        var perInputSumSq = [Double](repeating: 0, count: inC)
        let strideO = inC * kH * kW
        let strideI = kH * kW
        for o in 0..<outC {
            for i in 0..<inC {
                let base = o * strideO + i * strideI
                for hw in 0..<(kH * kW) {
                    let v = Double(values[base + hw])
                    perInputSumSq[i] += v * v
                }
            }
        }
        return perInputSumSq.map { sqrt($0) }
    }

    private static func makeConvOutputChannelDetail(
        variableName: String,
        values: [Float],
        initialValues: [Float],
        initExact: Bool,
        outC: Int,
        inC: Int,
        kH: Int,
        kW: Int
    ) -> Result.ConvOutputChannelDetail? {
        let expected = outC * inC * kH * kW
        guard values.count == expected, initialValues.count == expected else { return nil }
        return Result.ConvOutputChannelDetail(
            variableName: variableName,
            perOutputChannelL2: perOutputChannelL2(values, outC: outC, perOutSize: inC * kH * kW),
            initPerOutputChannelL2: perOutputChannelL2(initialValues, outC: outC, perOutSize: inC * kH * kW),
            initExact: initExact
        )
    }

    /// Each output channel's L2 norm over an OIHW conv tensor:
    /// `data[o * perOutSize + k]`.
    private static func perOutputChannelL2(_ values: [Float], outC: Int, perOutSize: Int) -> [Double] {
        var perOutputSumSq = [Double](repeating: 0, count: outC)
        for o in 0..<outC {
            let base = o * perOutSize
            for k in 0..<perOutSize {
                let v = Double(values[base + k])
                perOutputSumSq[o] += v * v
            }
        }
        return perOutputSumSq.map { sqrt($0) }
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
        out += "Per-section summary (L2 over trainable tensors; running stats apart):\n"
        out += "  (~ = init figure includes draws of the same distribution, not the model's own initial values; "
            + "-- = none: started at zero, or a BN running statistic, whose start depends on how the model was built)\n"
        out += "  section         params      L2          init_L2     ratio\n"
        for s in sections {
            let initStr = String(format: "%9.3f", s.totalInitL2Norm) + (s.initExact ? "" : "~")
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
                let initStr = v.initL2Norm.map { String(format: "%7.3f", $0) + (v.initExact ? "" : "~") } ?? "  --  "
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
            out += "Stem per-input-channel L2 (each plane against its own L2 at init):\n"
            for (i, l2) in stem.perInputChannelL2.enumerated() {
                let label = i < stem.planeLabels.count ? stem.planeLabels[i] : "plane_\(i)"
                let initL2 = i < stem.initPerInputChannelL2.count ? stem.initPerInputChannelL2[i] : 0
                let ratioStr = initL2 > 0 ? String(format: "%6.3f", l2 / initL2) : "  --  "
                out += String(
                    format: "  %@ %@   L2=%6.3f   init=%@   ratio=%@\n",
                    String(format: "%2d", i),
                    label.padded(26),
                    l2,
                    String(format: "%6.3f", initL2) + (stem.initExact ? "" : "~"),
                    ratioStr
                )
            }
            out += "\n"
        }

        // Per-conv per-output-channel summary: dead/weak channel
        // counts per conv tensor instead of dumping every channel's
        // L2 (which would be 24+ × 128 = 3000+ rows for the tower).
        // JSON has the full per-channel arrays.
        if !convOutputChannelDetails.isEmpty {
            out += "Per-conv output-channel health (dead/weak counts vs each channel's L2 at init; channels that started at zero are left out):\n"
            out += "  variable                              channels  dead(<\(String(format: "%.2f", NetworkWeightAnalyzer.convDeadRatio)))  weak(<\(String(format: "%.2f", NetworkWeightAnalyzer.convWeakRatio)))  p10ratio  p50ratio  p90ratio  initL2\n"
            for c in convOutputChannelDetails {
                let ratios = zip(c.perOutputChannelL2, c.initPerOutputChannelL2).compactMap { current, initial in
                    initial > 0 ? current / initial : nil
                }
                let initialMean = c.initPerOutputChannelL2.isEmpty ? 0
                    : c.initPerOutputChannelL2.reduce(0, +) / Double(c.initPerOutputChannelL2.count)
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
                    (String(format: "%6.3f", initialMean) + (c.initExact ? "" : "~")).leftPadded(toLength: 8)
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
