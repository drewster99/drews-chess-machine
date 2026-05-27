import Foundation

// MARK: - Whole-Network Weight Analyzer
//
// Snapshot-time diagnostic for the entire network's weight tensors,
// not just the value head. Answers "is everything doing work, is
// any layer collapsed, is the stem looking at the planes it should
// be looking at, and is the policy head's output capacity spread
// across the 76 channels or concentrated in a few?"
//
// Per-variable: same stats family `ValueHeadAnalyzer` reports
// (count, L1/L2, mean|w|, min/max, p10/p50/p90, ratio to He-init L2).
//
// Per-section: aggregated L2 + element count for stem / each tower
// block / policy / value, plus the section's overall L2 ratio to its
// combined He-init reference. Lets you eyeball "block 5 is unusually
// quiet compared to its neighbors" at a glance.
//
// Special details:
//   - Stem per-input-channel L2 over the 30 input planes (with the
//     human-readable plane labels from BoardEncoder). Tells you
//     whether the network has learned to attend to, e.g., the
//     repetition-history planes vs ignored them.
//   - Policy-conv per-output-channel L2 over the 76 policy channels.
//     Tells you whether some move-type channels (e.g. long-distance
//     queen slides) carry essentially zero weight magnitude.
//
// Pure analysis — single `exportWeights()` call + CPU stat-crunching.

enum NetworkWeightAnalyzer {

    // MARK: - Section + fanIn helpers

    /// Canonical section ordering used in the JSON and the text
    /// summary. Unknown variables (none expected from the current
    /// architecture, but defended against) land in `"other"` so
    /// nothing is silently dropped.
    static let sectionOrder: [String] =
        ["stem"]
        + (1...8).map { "block_\($0)" }
        + ["policy", "value", "other"]

    /// 0-based section bucket for a variable. Drives both the
    /// per-section summaries and the text-summary grouping.
    static func section(forVariableNamed name: String) -> String {
        if name.hasPrefix("stem_") { return "stem" }
        if name.hasPrefix("policy_") { return "policy" }
        if name.hasPrefix("value_") { return "value" }
        if name.hasPrefix("block") {
            // Names look like "block1_conv1_weights", "block7_se_fc2_bias", etc.
            // Pull the digit(s) between "block" and the first underscore.
            let afterPrefix = name.dropFirst("block".count)
            var digits = ""
            for c in afterPrefix {
                if c.isNumber { digits.append(c) } else { break }
            }
            if !digits.isEmpty { return "block_\(digits)" }
        }
        return "other"
    }

    /// He-init fan-in for the variable name, or `nil` for tensors
    /// that don't have a meaningful He-init reference (biases,
    /// BN gamma/beta, BN running stats). The values here mirror the
    /// shapes set in `ChessNetwork`'s graph construction; if those
    /// shapes change the analyzer needs updating in lockstep.
    static func fanIn(forVariableNamed name: String) -> Int? {
        switch name {
        case "stem_conv_weights":  return 30 * 3 * 3
        case "policy_conv_weights": return 128 * 1 * 1
        case "value_conv_weights": return 128
        case "value_fc1_weights":  return 64
        case "value_fc2_weights":  return 64
        default: break
        }
        if name.hasSuffix("_conv1_weights") || name.hasSuffix("_conv2_weights") {
            return 128 * 3 * 3
        }
        if name.hasSuffix("_se_fc1_weights") { return 128 }
        if name.hasSuffix("_se_fc2_weights") { return 32 }
        return nil
    }

    /// He-init L2 reference for a tensor with `n` elements drawn from
    /// `N(0, sqrt(2/fanIn))`. Expected L2 = sqrt(n) · std. Used as
    /// the denominator of the `currentL2 / initL2` ratio.
    static func heInitL2(elementCount n: Int, fanIn: Int) -> Double {
        guard n > 0, fanIn > 0 else { return 0 }
        let std = sqrt(2.0 / Double(fanIn))
        return sqrt(Double(n)) * std
    }

    /// Human-readable label per input plane. Order matches
    /// `BoardEncoder.encode`'s plane layout. Kept here rather than
    /// re-derived because the analyzer is the only place that needs
    /// per-plane names for display.
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
        }

        struct SectionSummary: Codable, Sendable {
            /// Canonical section name from `sectionOrder` (e.g. "stem",
            /// "block_3", "policy", "value", "other").
            let sectionName: String
            /// Sum of element counts across all variables in the section.
            let totalElementCount: Int
            /// Combined L2 norm `sqrt(Σ w²)` over all variables. Bias /
            /// BN parameters contribute too; their magnitudes are
            /// typically small enough that they don't move this number.
            let totalL2Norm: Double
            /// Combined He-init reference L2 — sum-of-squares of each
            /// He-initialized variable's init L2, square-rooted. Only
            /// counts variables with a He-init reference (biases and
            /// BN params are excluded). `nil` if the section has no
            /// He-initialized variables.
            let totalInitL2Norm: Double?
            /// `totalL2Norm / totalInitL2Norm` restricted to the
            /// He-initialized variables — both numerator and denominator
            /// are restricted to that subset so the ratio is meaningful.
            let totalL2RatioToInit: Double?
            /// Per-variable stats inside the section, in graph build
            /// order (which matches the order `exportWeights()` returns
            /// them in).
            let variables: [WeightStats]
        }

        struct StemInputChannelDetail: Codable, Sendable {
            /// Per-input-channel L2 norm of `stem_conv_weights`. The
            /// stem conv is [outC=128, inC=30, kH=3, kW=3] in OIHW
            /// layout; for input plane `i`, this sums squares of
            /// `weights[o, i, h, w]` for all `o ∈ 0..<128`, `h, w ∈ 0..<3`
            /// — 128 × 9 = 1152 weights per input plane. A near-zero
            /// per-plane L2 means the network has learned to ignore
            /// that input plane.
            let perInputChannelL2: [Double]
            /// Plane labels matching `perInputChannelL2[i]`. Same
            /// length as the array above; mirrors `inputPlaneLabels`.
            let planeLabels: [String]
            /// He-init reference per-plane L2 — same `heInitL2`
            /// calculation applied to 1152 weights with fan_in = 270
            /// (30·3·3). Lets the reader compute "this plane is at X%
            /// of init" per plane.
            let initPerInputChannelL2: Double
        }

        struct PolicyOutputChannelDetail: Codable, Sendable {
            /// Per-output-channel L2 norm of `policy_conv_weights`.
            /// Policy conv is [outC=76, inC=128, kH=1, kW=1]; for
            /// output channel `c`, this sums squares of
            /// `weights[c, i, 0, 0]` for all `i ∈ 0..<128` — 128
            /// weights per output channel. A near-zero per-channel
            /// L2 means the network can never assign meaningful
            /// probability to that move-type channel regardless of
            /// position.
            let perOutputChannelL2: [Double]
            /// He-init reference per-output-channel L2 — same
            /// `heInitL2` calculation applied to 128 weights with
            /// fan_in = 128.
            let initPerOutputChannelL2: Double
        }

        let producedAtISO8601: String
        let modelLabel: String
        let totalParamCount: Int
        let sections: [SectionSummary]
        let stemDetail: StemInputChannelDetail?
        let policyDetail: PolicyOutputChannelDetail?
    }

    // MARK: - Entry point

    /// Run the analyzer against `network`. Single `exportWeights()`
    /// call; the rest is CPU stat-crunching.
    static func run(
        network: ChessMPSNetwork,
        modelLabel: String
    ) async throws -> Result {
        let weights = try await network.exportWeights()
        let allVariables = network.network.trainableVariables
            + network.network.bnRunningStatsVariables

        guard weights.count == allVariables.count else {
            throw NetworkWeightAnalyzerError.weightCountMismatch(
                expected: allVariables.count,
                got: weights.count
            )
        }

        // Group (name, values) pairs by section in the order they
        // appear in `allVariables` so per-section variable lists
        // preserve graph build order.
        var perSection: [String: [(name: String, values: [Float])]] = [:]
        for (i, variable) in allVariables.enumerated() {
            let name = variable.operation.name
            let sec = section(forVariableNamed: name)
            perSection[sec, default: []].append((name, weights[i]))
        }

        // Build per-section summaries in canonical order, skipping
        // sections that don't exist in this particular network.
        var sections: [Result.SectionSummary] = []
        for sec in sectionOrder {
            guard let vars = perSection[sec] else { continue }
            sections.append(makeSectionSummary(sectionName: sec, variables: vars))
        }

        // Stem detail.
        let stemDetail: Result.StemInputChannelDetail? = {
            guard let stemVars = perSection["stem"],
                  let stemConv = stemVars.first(where: { $0.name == "stem_conv_weights" }) else {
                return nil
            }
            return makeStemDetail(stemConvValues: stemConv.values)
        }()

        // Policy detail.
        let policyDetail: Result.PolicyOutputChannelDetail? = {
            guard let policyVars = perSection["policy"],
                  let policyConv = policyVars.first(where: { $0.name == "policy_conv_weights" }) else {
                return nil
            }
            return makePolicyDetail(policyConvValues: policyConv.values)
        }()

        let totalParamCount = sections.reduce(0) { $0 + $1.totalElementCount }

        let iso = ISO8601DateFormatter()
        iso.formatOptions = [.withInternetDateTime]
        return Result(
            producedAtISO8601: iso.string(from: Date()),
            modelLabel: modelLabel,
            totalParamCount: totalParamCount,
            sections: sections,
            stemDetail: stemDetail,
            policyDetail: policyDetail
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
        var totalSumSq: Double = 0
        // Section-level "init L2" is restricted to He-initialized
        // variables — biases / BN params don't have a comparable init
        // reference. Both numerator and denominator of the ratio use
        // the same subset.
        var heInitSumSq: Double = 0
        var heCurrentSumSq: Double = 0
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
                initL2Norm: nil, l2NormRatioToInit: nil
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

        let initL2 = fanIn(forVariableNamed: name).map {
            heInitL2(elementCount: n, fanIn: $0)
        }
        let ratio = initL2.map { $0 > 0 ? l2 / $0 : 0 }

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
            l2NormRatioToInit: ratio
        )
    }

    // MARK: - Detail builders

    private static func makeStemDetail(stemConvValues: [Float]) -> Result.StemInputChannelDetail? {
        let outC = 128, inC = 30, kH = 3, kW = 3
        let expected = outC * inC * kH * kW
        guard stemConvValues.count == expected else { return nil }

        var perInputSumSq = [Double](repeating: 0, count: inC)
        // OIHW layout: weights[o, i, h, w] = data[((o * inC + i) * kH + h) * kW + w]
        // = data[o * inC * kH * kW + i * kH * kW + h * kW + w]
        let strideO = inC * kH * kW   // 30 * 9 = 270
        let strideI = kH * kW         // 9
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
            fanIn: inC * kH * kW   // stem conv fan-in = 270
        )
        return Result.StemInputChannelDetail(
            perInputChannelL2: perInputL2,
            planeLabels: Array(inputPlaneLabels.prefix(inC)),
            initPerInputChannelL2: initPerInputL2
        )
    }

    private static func makePolicyDetail(policyConvValues: [Float]) -> Result.PolicyOutputChannelDetail? {
        let outC = 76, inC = 128, kH = 1, kW = 1
        let expected = outC * inC * kH * kW
        guard policyConvValues.count == expected else { return nil }

        var perOutputSumSq = [Double](repeating: 0, count: outC)
        // OIHW layout with kH=kW=1: weights[o, i] = data[o * inC + i]
        for o in 0..<outC {
            let base = o * inC
            for i in 0..<inC {
                let v = Double(policyConvValues[base + i])
                perOutputSumSq[o] += v * v
            }
        }
        let perOutputL2 = perOutputSumSq.map { sqrt($0) }
        let initPerOutputL2 = heInitL2(elementCount: inC, fanIn: inC)
        return Result.PolicyOutputChannelDetail(
            perOutputChannelL2: perOutputL2,
            initPerOutputChannelL2: initPerOutputL2
        )
    }

    // MARK: - Numeric helpers

    /// Linear-interpolation percentile over ascending-sorted input.
    /// Same convention `ValueHeadAnalyzer.percentile` uses.
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

    /// Multi-line digest. Mirrors the layout used by
    /// `ReplayBufferAnalyzer.Result` and `ValueHeadAnalyzer.Result`.
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

        // Per-variable detail under each section.
        for s in sections {
            out += "Section: \(s.sectionName) (\(formatInt(s.totalElementCount)) params)\n"
            out += "  variable                              count      L2       init_L2  ratio   mean|w|   min       max\n"
            for v in s.variables {
                let initStr = v.initL2Norm.map { String(format: "%7.3f", $0) } ?? "  -- "
                let ratioStr = v.l2NormRatioToInit.map { String(format: "%6.3f", $0) } ?? "  --  "
                out += String(
                    format: "  %@  %@  %@  %@  %@  %@  %@  %@\n",
                    v.name.padded(38),
                    formatInt(v.elementCount).leftPadded(toLength: 7),
                    String(format: "%8.3f", v.l2Norm).leftPadded(toLength: 8),
                    initStr.leftPadded(toLength: 8),
                    ratioStr.leftPadded(toLength: 7),
                    String(format: "%7.4f", v.meanAbs).leftPadded(toLength: 8),
                    String(format: "%+7.3f", v.min).leftPadded(toLength: 9),
                    String(format: "%+7.3f", v.max).leftPadded(toLength: 9)
                )
            }
            out += "\n"
        }

        // Stem per-input-channel detail.
        if let stem = stemDetail {
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

        // Policy per-output-channel detail — too many channels (76)
        // to list inline; summarize as top-5 and bottom-5 by L2.
        if let policy = policyDetail {
            out += "Policy per-output-channel L2 (init ref \(String(format: "%.3f", policy.initPerOutputChannelL2)) per channel):\n"
            let indexed = policy.perOutputChannelL2.enumerated().map { ($0, $1) }
            let sortedDesc = indexed.sorted { $0.1 > $1.1 }
            let topN = 5
            let botN = 5
            out += "  Top \(topN) by L2:\n"
            for (idx, l2) in sortedDesc.prefix(topN) {
                let ratio = policy.initPerOutputChannelL2 > 0 ? l2 / policy.initPerOutputChannelL2 : 0
                out += String(
                    format: "    chan %@  L2=%6.3f  ratio=%6.3f\n",
                    String(format: "%2d", idx),
                    l2,
                    ratio
                )
            }
            out += "  Bottom \(botN) by L2:\n"
            for (idx, l2) in sortedDesc.suffix(botN).reversed() {
                let ratio = policy.initPerOutputChannelL2 > 0 ? l2 / policy.initPerOutputChannelL2 : 0
                out += String(
                    format: "    chan %@  L2=%6.3f  ratio=%6.3f\n",
                    String(format: "%2d", idx),
                    l2,
                    ratio
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
    /// Right-pad to `length` with spaces (left-aligned column).
    func padded(_ length: Int) -> String {
        if count >= length { return self }
        return self + String(repeating: " ", count: length - count)
    }
    /// Left-pad to `length` with spaces (right-aligned column).
    func leftPadded(toLength length: Int) -> String {
        if count >= length { return self }
        return String(repeating: " ", count: length - count) + self
    }
}
