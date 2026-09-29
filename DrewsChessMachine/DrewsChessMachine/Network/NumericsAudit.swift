import Foundation

// MARK: - Numerics audit
//
// Measures how well a network's numbers fit each compute format (fp32, bf16,
// fp16) — the Phase 0 gauge of the head numerics plan
// (documentation/plans-active/HEAD_NUMERICS_PLAN.md).
//
// Static checks read only the weights, so they run on any checkpoint:
// per-tensor fitness for every format, plus the known hot spots — the shared
// offset the heads' final layers grow (softmax cannot see it, so nothing in
// the loss removes it, and a large one is rounded into ties), BatchNorm
// running statistics whose mean dwarfs their spread (cancellation), the
// ReZero bound saturating in a narrow format, and fp32 masters against their
// working copy.
//
// Dynamic checks run the real MPSGraph network, built once per format from
// the same weights, over a fixed position set, and compare the heads and
// every analysis tap against the fp32 build (`NumericsAudit+Dynamic.swift`).

enum NumericsAudit {

    // MARK: - Thresholds

    /// Verdict thresholds. Starting points from the bf16 head-offset survey
    /// (documentation/research/bf16-head-offset/); adjust as evidence grows.
    enum Threshold {
        /// Value-head cross-entropy increase against fp32, in nats.
        static let valueCEDeltaDegraded = 0.005
        static let valueCEDeltaBad = 0.03
        /// Fraction of positions with two or more W/D/L logits exactly equal.
        static let valueTieFractionDegraded = 0.10
        /// Mean KL(fp32 ‖ format) over the legal-move softmax, in nats.
        static let policyKLDegraded = 1e-3
        static let policyKLBad = 1e-2
        /// Fraction of positions whose two best legal logits are exactly equal.
        static let policyTop2TieDegraded = 0.10
        static let policyTop2TieBad = 0.30
        /// A head's final-layer mean-row norm relative to its per-output
        /// residual norm, as a multiple of what random init gives.
        static let sharedOffsetRatioToInitDegraded = 4.0
        static let sharedOffsetRatioToInitBad = 10.0
        /// |mean| / spread of a normalization input (or a BN running mean
        /// against its running standard deviation). Large means the mean is
        /// subtracted from values that were rounded at the mean's scale.
        static let offsetToSpreadDegraded = 16.0
        static let offsetToSpreadBad = 64.0
        /// Rounding error (RMS) relative to a tensor's own standard deviation.
        static let roundingToSpreadDegraded = 0.01
        static let roundingToSpreadBad = 0.05
        /// Fraction of nonzero values below the format's normal range, and
        /// the fraction that round to zero.
        static let underflowFractionDegraded = 0.01
        static let flushToZeroFractionBad = 0.01
        /// How many times the largest value fits under the format's maximum.
        static let overflowHeadroomDegraded = 16.0
        /// Relative RMS error of an analysis tap against the fp32 build.
        static let tapRelativeErrorDegraded = 0.05
        static let tapRelativeErrorBad = 0.2
    }

    // MARK: - Result types

    struct TensorFitness: Codable, Sendable, Equatable {
        let format: NumericFormat
        /// `maxFinite / maxAbs`; nil for an all-zero tensor.
        let overflowHeadroom: Double?
        /// Some value exceeds the format's largest finite value.
        let overflows: Bool
        /// Fraction of nonzero values below the format's normal range.
        let underflowFraction: Double
        /// Fraction of nonzero values that round to zero.
        let flushToZeroFraction: Double
        /// The format's step at the median |value|, over the standard
        /// deviation; nil for a constant tensor.
        let stepToSpread: Double?
        /// RMS rounding error over the standard deviation; nil for a
        /// constant tensor.
        let roundingRMSToSpread: Double?
        let verdict: NumericsVerdict
    }

    struct TensorReport: Codable, Sendable {
        let name: String
        let elementCount: Int
        let mean: Double
        let stdev: Double
        let maxAbs: Double
        let medianAbs: Double
        let fitness: [TensorFitness]
    }

    /// A head's final layer: how much of it is a component shared by every
    /// output, which softmax ignores and bf16 rounds into ties.
    struct SharedOffsetReport: Codable, Sendable {
        let weightName: String
        let outputCount: Int
        /// Norm of the per-input mean across outputs.
        let meanRowNorm: Double
        /// Median over outputs of the norm of that output's weights minus
        /// the mean row.
        let residualNormMedian: Double
        let ratio: Double
        /// What random init gives for `ratio`.
        let initExpectedRatio: Double
        let ratioToInitExpectation: Double
        let biasName: String
        let biasMean: Double
        let biasInitMean: Double
        let verdict: NumericsVerdict
    }

    struct BatchNormStatsReport: Codable, Sendable {
        let layerName: String
        let channelCount: Int
        /// Largest |running mean| / √running variance over channels.
        let maxMeanToStd: Double
        let maxMeanToStdChannel: Int
        let medianMeanToStd: Double
        let minVariance: Double
        /// Channels whose running variance is below each format's normal
        /// range (keyed by format).
        let varianceBelowNormalCount: [String: Int]
        let verdict: NumericsVerdict
    }

    struct ReZeroReport: Codable, Sendable {
        let variableName: String
        let alpha: Double
        let ceiling: Double
        let tanhValue: Double
        /// Whether tanh(α/C) rounds to exactly 1 in each format (keyed by
        /// format). Reported, not flagged: the ceiling is by design.
        let saturatesInFormat: [String: Bool]
    }

    struct MasterDivergence: Codable, Sendable {
        let name: String
        let maxAbsDifference: Double
        /// The largest difference over the bf16 step at that value's size.
        let maxDifferenceInBF16Steps: Double
    }

    struct StaticResult: Codable, Sendable {
        let tensors: [TensorReport]
        let valueHeadOffset: SharedOffsetReport?
        let policyHeadOffset: SharedOffsetReport?
        let batchNormStats: [BatchNormStatsReport]
        let reZero: [ReZeroReport]
        /// The tensors with the largest master/working divergence; nil when
        /// no masters were read.
        let masterDivergence: [MasterDivergence]?
        /// Why masters weren't compared, when they weren't.
        let mastersNote: String?
    }

    struct Finding: Codable, Sendable {
        let area: String
        let subject: String
        let format: NumericFormat?
        let verdict: NumericsVerdict
        let detail: String
    }

    struct Result: Codable, Sendable {
        let producedAtISO8601: String
        let modelLabel: String
        let modelID: String?
        let trainingStep: Int?
        let computeDataType: String
        var exportMetadata: AnalysisExportMetadata? = nil
        let staticChecks: StaticResult
        let dynamicChecks: DynamicResult?
        /// Why the dynamic checks didn't run, when they didn't.
        let dynamicSkippedReason: String?
        let findings: [Finding]
        let overallVerdict: NumericsVerdict
    }

    // MARK: - Entry point

    /// Audit `weights` (the layout `exportWeights()` produces; `names` are the
    /// matching graph variable names). `positions` nil skips the dynamic
    /// checks and `dynamicSkippedReason` says why.
    static func run(
        names: [String],
        weights: [[Float]],
        arch: NetworkArchitecture,
        masters: [[Float]]?,
        mastersNote: String?,
        positions: PositionSet?,
        dynamicSkippedReason: String?,
        modelLabel: String,
        modelID: String?,
        trainingStep: Int?
    ) async throws -> Result {
        let staticResult = try runStatic(names: names, weights: weights, arch: arch, masters: masters, mastersNote: mastersNote)
        var dynamicResult: DynamicResult?
        if let positions {
            dynamicResult = try await runDynamic(weights: weights, arch: arch, positions: positions)
        }
        let findings = collectFindings(staticResult: staticResult, dynamicResult: dynamicResult)
        let overall = findings.map(\.verdict).max() ?? .fine
        let iso = ISO8601DateFormatter()
        iso.formatOptions = [.withInternetDateTime]
        return Result(
            producedAtISO8601: iso.string(from: Date()),
            modelLabel: modelLabel,
            modelID: modelID,
            trainingStep: trainingStep,
            computeDataType: arch.computeDataType.rawValue,
            staticChecks: staticResult,
            dynamicChecks: dynamicResult,
            dynamicSkippedReason: positions == nil ? dynamicSkippedReason : nil,
            findings: findings,
            overallVerdict: overall
        )
    }

    // MARK: - Static checks

    /// Weights-only checks. Pure: no GPU, no files.
    static func runStatic(
        names: [String],
        weights: [[Float]],
        arch: NetworkArchitecture,
        masters: [[Float]]?,
        mastersNote: String?
    ) throws -> StaticResult {
        guard names.count == weights.count else {
            throw NumericsAuditError.weightCountMismatch(names: names.count, weights: weights.count)
        }
        var byName: [String: [Float]] = [:]
        for (name, values) in zip(names, weights) {
            byName[name] = values
        }

        let tensors = zip(names, weights).map { tensorReport(name: $0.0, values: $0.1) }

        let valueOffset: SharedOffsetReport?
        if arch.valueHeadStyle == .wdlSoftmax,
           let w = byName["value_wdl_fc2_weights"], let b = byName["value_wdl_fc2_bias"] {
            valueOffset = sharedOffset(
                weightName: "value_wdl_fc2_weights", weights: w, layout: .inputMajor(outputs: b.count),
                biasName: "value_wdl_fc2_bias", bias: b,
                biasInitMean: initMean(of: "value_wdl_fc2_bias", count: b.count, arch: arch)
            )
        } else {
            valueOffset = nil
        }

        let policyOffset: SharedOffsetReport?
        if let w = byName["policy_conv_weights"], let b = byName["policy_conv_bias"] {
            policyOffset = sharedOffset(
                weightName: "policy_conv_weights", weights: w, layout: .outputMajor(outputs: b.count),
                biasName: "policy_conv_bias", bias: b,
                biasInitMean: initMean(of: "policy_conv_bias", count: b.count, arch: arch)
            )
        } else if let w = byName["policy_fc_weights"], let b = byName["policy_fc_bias"] {
            policyOffset = sharedOffset(
                weightName: "policy_fc_weights", weights: w, layout: .inputMajor(outputs: b.count),
                biasName: "policy_fc_bias", bias: b,
                biasInitMean: initMean(of: "policy_fc_bias", count: b.count, arch: arch)
            )
        } else {
            policyOffset = nil
        }

        var bnReports: [BatchNormStatsReport] = []
        for name in names where name.hasSuffix("_running_mean") {
            let layer = String(name.dropLast("_running_mean".count))
            guard let mean = byName[name], let variance = byName["\(layer)_running_var"] else { continue }
            bnReports.append(batchNormStats(layerName: layer, mean: mean, variance: variance))
        }

        var reZero: [ReZeroReport] = []
        for name in names where name.hasSuffix("_res_scale") {
            guard let values = byName[name], let alpha = values.first,
                  let (spec, _) = NetworkWeightAnalyzer.blockSpec(forVariableNamed: name, arch: arch) else { continue }
            let ceiling = Double(spec.rezeroAlphaInit) * NetworkArchitecture.rezeroTanhCeilingMultiple
            reZero.append(reZeroReport(name: name, alpha: Double(alpha), ceiling: ceiling))
        }

        var divergence: [MasterDivergence]?
        if let masters {
            guard masters.count == weights.count else {
                throw NumericsAuditError.masterCountMismatch(masters: masters.count, weights: weights.count)
            }
            divergence = masterDivergence(names: names, working: weights, masters: masters)
        }

        return StaticResult(
            tensors: tensors,
            valueHeadOffset: valueOffset,
            policyHeadOffset: policyOffset,
            batchNormStats: bnReports,
            reZero: reZero,
            masterDivergence: divergence,
            mastersNote: masters == nil ? mastersNote : nil
        )
    }

    /// The mean of a tensor's deterministic init, or zero for one without.
    private static func initMean(of name: String, count: Int, arch: NetworkArchitecture) -> Double {
        guard let values = NetworkWeightAnalyzer.deterministicInit(forVariableNamed: name, elementCount: count, arch: arch),
              !values.isEmpty else {
            return 0
        }
        return values.reduce(0, +) / Double(values.count)
    }

    static func tensorReport(name: String, values: [Float]) -> TensorReport {
        let n = values.count
        guard n > 0 else {
            return TensorReport(name: name, elementCount: 0, mean: 0, stdev: 0, maxAbs: 0, medianAbs: 0, fitness: [])
        }
        var sum = 0.0
        var maxAbs = 0.0
        for v in values {
            let d = Double(v)
            sum += d
            maxAbs = max(maxAbs, abs(d))
        }
        let mean = sum / Double(n)
        var sq = 0.0
        for v in values {
            let d = Double(v) - mean
            sq += d * d
        }
        let stdev = (sq / Double(n)).squareRoot()
        let sortedAbs = values.map { abs(Double($0)) }.sorted()
        let medianAbs = NetworkWeightAnalyzer.percentile(p: 50, sortedAscending: sortedAbs)
        let spreadMatters = !multiplicativeSuffixes.contains { name.hasSuffix($0) }
        let fitness = NumericFormat.allCases.map { format in
            tensorFitness(values: values, format: format, maxAbs: maxAbs, medianAbs: medianAbs, stdev: stdev, spreadMatters: spreadMatters)
        }
        return TensorReport(name: name, elementCount: n, mean: mean, stdev: stdev, maxAbs: maxAbs, medianAbs: medianAbs, fitness: fitness)
    }

    /// Tensors used as multipliers (normalization scales and variances, the
    /// ReZero scalar): their precision matters relative to their size, not to
    /// their spread across channels, so rounding-against-spread is reported
    /// for them but doesn't set a verdict.
    static let multiplicativeSuffixes = ["_gamma", "_running_var", "_res_scale"]

    static func tensorFitness(values: [Float], format: NumericFormat, maxAbs: Double, medianAbs: Double, stdev: Double, spreadMatters: Bool) -> TensorFitness {
        var nonzero = 0
        var underflow = 0
        var flushed = 0
        var roundingSq = 0.0
        var overflows = false
        for v in values {
            let rounded = format.round(v)
            if !rounded.isFinite && v.isFinite {
                overflows = true
            }
            if v != 0 {
                nonzero += 1
                if Double(abs(v)) < format.minNormal { underflow += 1 }
                if rounded == 0 { flushed += 1 }
            }
            if rounded.isFinite {
                let e = Double(rounded) - Double(v)
                roundingSq += e * e
            }
        }
        let n = Double(values.count)
        let underflowFraction = nonzero > 0 ? Double(underflow) / Double(nonzero) : 0
        let flushFraction = nonzero > 0 ? Double(flushed) / Double(nonzero) : 0
        let headroom: Double? = maxAbs > 0 ? format.maxFinite / maxAbs : nil
        let spreadBased = stdev > 0
        let stepToSpread: Double? = spreadBased ? format.step(atMagnitude: medianAbs) / stdev : nil
        let roundingToSpread: Double? = spreadBased ? (roundingSq / n).squareRoot() / stdev : nil

        var verdict = NumericsVerdict.fine
        if overflows || flushFraction >= Threshold.flushToZeroFractionBad {
            verdict = .bad
        } else if spreadMatters, let r = roundingToSpread, r >= Threshold.roundingToSpreadBad {
            verdict = .bad
        } else if spreadMatters, let r = roundingToSpread, r >= Threshold.roundingToSpreadDegraded {
            verdict = .degraded
        } else if underflowFraction >= Threshold.underflowFractionDegraded {
            verdict = .degraded
        } else if let h = headroom, h < Threshold.overflowHeadroomDegraded {
            verdict = .degraded
        }
        return TensorFitness(
            format: format,
            overflowHeadroom: headroom,
            overflows: overflows,
            underflowFraction: underflowFraction,
            flushToZeroFraction: flushFraction,
            stepToSpread: stepToSpread,
            roundingRMSToSpread: roundingToSpread,
            verdict: verdict
        )
    }

    /// How a final layer's weights are laid out, flat and row-major.
    enum FinalLayerLayout {
        /// `[inputs, outputs]` (an FC weight: value fc2, policy fc).
        case inputMajor(outputs: Int)
        /// `[outputs, inputs]` (a 1×1 conv's OIHW weight: policy conv).
        case outputMajor(outputs: Int)
    }

    static func sharedOffset(
        weightName: String,
        weights: [Float],
        layout: FinalLayerLayout,
        biasName: String,
        bias: [Float],
        biasInitMean: Double
    ) -> SharedOffsetReport? {
        let outputs: Int
        let outputMajor: Bool
        switch layout {
        case .inputMajor(let o): outputs = o; outputMajor = false
        case .outputMajor(let o): outputs = o; outputMajor = true
        }
        guard outputs > 1, weights.count % outputs == 0 else { return nil }
        let inputs = weights.count / outputs
        func w(_ input: Int, _ output: Int) -> Double {
            Double(outputMajor ? weights[output * inputs + input] : weights[input * outputs + output])
        }
        var meanRow = [Double](repeating: 0, count: inputs)
        for i in 0..<inputs {
            var s = 0.0
            for o in 0..<outputs { s += w(i, o) }
            meanRow[i] = s / Double(outputs)
        }
        let meanRowNorm = meanRow.reduce(0) { $0 + $1 * $1 }.squareRoot()
        var residuals: [Double] = []
        residuals.reserveCapacity(outputs)
        for o in 0..<outputs {
            var s = 0.0
            for i in 0..<inputs {
                let d = w(i, o) - meanRow[i]
                s += d * d
            }
            residuals.append(s.squareRoot())
        }
        let residualMedian = NetworkWeightAnalyzer.percentile(p: 50, sortedAscending: residuals.sorted())
        let ratio = residualMedian > 0 ? meanRowNorm / residualMedian : 0
        // With independent init across outputs the mean over them has
        // variance σ²/outputs and each residual σ²(outputs−1)/outputs.
        let initExpected = 1 / Double(outputs - 1).squareRoot()
        let ratioToInit = ratio / initExpected
        let biasMean = bias.isEmpty ? 0 : bias.reduce(0.0) { $0 + Double($1) } / Double(bias.count)
        let verdict: NumericsVerdict = ratioToInit >= Threshold.sharedOffsetRatioToInitBad ? .bad
            : (ratioToInit >= Threshold.sharedOffsetRatioToInitDegraded ? .degraded : .fine)
        return SharedOffsetReport(
            weightName: weightName,
            outputCount: outputs,
            meanRowNorm: meanRowNorm,
            residualNormMedian: residualMedian,
            ratio: ratio,
            initExpectedRatio: initExpected,
            ratioToInitExpectation: ratioToInit,
            biasName: biasName,
            biasMean: biasMean,
            biasInitMean: biasInitMean,
            verdict: verdict
        )
    }

    static func batchNormStats(layerName: String, mean: [Float], variance: [Float]) -> BatchNormStatsReport {
        let n = min(mean.count, variance.count)
        var ratios: [Double] = []
        ratios.reserveCapacity(n)
        var maxRatio = 0.0
        var maxChannel = 0
        var minVariance = Double.infinity
        var belowNormal: [String: Int] = [:]
        for format in NumericFormat.allCases {
            belowNormal[format.rawValue] = 0
        }
        for c in 0..<n {
            let v = Double(variance[c])
            minVariance = min(minVariance, v)
            for format in NumericFormat.allCases where v < format.minNormal {
                belowNormal[format.rawValue, default: 0] += 1
            }
            // A zero or negative running variance has no spread to compare
            // against; it is reported through `minVariance` instead.
            guard v > 0 else { continue }
            let r = abs(Double(mean[c])) / v.squareRoot()
            ratios.append(r)
            if r > maxRatio {
                maxRatio = r
                maxChannel = c
            }
        }
        let median = NetworkWeightAnalyzer.percentile(p: 50, sortedAscending: ratios.sorted())
        let verdict: NumericsVerdict = maxRatio >= Threshold.offsetToSpreadBad ? .bad
            : (maxRatio >= Threshold.offsetToSpreadDegraded ? .degraded : .fine)
        return BatchNormStatsReport(
            layerName: layerName,
            channelCount: n,
            maxMeanToStd: maxRatio,
            maxMeanToStdChannel: maxChannel,
            medianMeanToStd: median,
            minVariance: n > 0 ? minVariance : 0,
            varianceBelowNormalCount: belowNormal,
            verdict: verdict
        )
    }

    static func reZeroReport(name: String, alpha: Double, ceiling: Double) -> ReZeroReport {
        let t = ceiling > 0 ? tanh(alpha / ceiling) : 0
        var saturates: [String: Bool] = [:]
        for format in NumericFormat.allCases {
            saturates[format.rawValue] = abs(format.round(Float(t))) == 1
        }
        return ReZeroReport(variableName: name, alpha: alpha, ceiling: ceiling, tanhValue: t, saturatesInFormat: saturates)
    }

    /// Tensors with the largest master/working divergence, most first.
    static let masterDivergenceReportCount = 10

    static func masterDivergence(names: [String], working: [[Float]], masters: [[Float]]) -> [MasterDivergence] {
        var out: [MasterDivergence] = []
        for (i, name) in names.enumerated() {
            let w = working[i]
            let m = masters[i]
            guard w.count == m.count else { continue }
            var maxDiff = 0.0
            var maxSteps = 0.0
            for j in 0..<w.count {
                let d = abs(Double(m[j]) - Double(w[j]))
                maxDiff = max(maxDiff, d)
                let step = NumericFormat.bf16.step(atMagnitude: abs(Double(m[j])))
                maxSteps = max(maxSteps, d / step)
            }
            out.append(MasterDivergence(name: name, maxAbsDifference: maxDiff, maxDifferenceInBF16Steps: maxSteps))
        }
        return Array(out.sorted { $0.maxDifferenceInBF16Steps > $1.maxDifferenceInBF16Steps }.prefix(masterDivergenceReportCount))
    }

    // MARK: - Findings

    static func collectFindings(staticResult: StaticResult, dynamicResult: DynamicResult?) -> [Finding] {
        var findings: [Finding] = []
        for tensor in staticResult.tensors {
            for fit in tensor.fitness where fit.verdict != .fine {
                findings.append(Finding(
                    area: "weights", subject: tensor.name, format: fit.format, verdict: fit.verdict,
                    detail: "headroom \(fmt(fit.overflowHeadroom)), underflow \(fmt(fit.underflowFraction)), flush-to-zero \(fmt(fit.flushToZeroFraction)), rounding/spread \(fmt(fit.roundingRMSToSpread))"
                ))
            }
        }
        for report in [staticResult.valueHeadOffset, staticResult.policyHeadOffset].compactMap({ $0 }) where report.verdict != .fine {
            findings.append(Finding(
                area: "head shared offset", subject: report.weightName, format: nil, verdict: report.verdict,
                detail: "mean-row ratio \(fmt(report.ratio)) is \(fmt(report.ratioToInitExpectation))x init; bias mean \(fmt(report.biasMean)) (init \(fmt(report.biasInitMean)))"
            ))
        }
        for bn in staticResult.batchNormStats where bn.verdict != .fine {
            findings.append(Finding(
                area: "batch-norm stats", subject: bn.layerName, format: nil, verdict: bn.verdict,
                detail: "max |mean|/std \(fmt(bn.maxMeanToStd)) at channel \(bn.maxMeanToStdChannel)"
            ))
        }
        if let dynamicResult {
            findings.append(contentsOf: dynamicFindings(dynamicResult))
        }
        return findings.sorted { $0.verdict > $1.verdict }
    }

    static func fmt(_ value: Double?) -> String {
        guard let value else { return "-" }
        if value == 0 { return "0" }
        let magnitude = abs(value)
        if magnitude >= 1e4 || magnitude < 1e-3 {
            return String(format: "%.3e", value)
        }
        return String(format: "%.4g", value)
    }
}

enum NumericsAuditError: LocalizedError {
    case weightCountMismatch(names: Int, weights: Int)
    case masterCountMismatch(masters: Int, weights: Int)
    case tapMissing(String)
    case tapShapeMismatch(name: String, detail: String)
    case positionSetEmpty
    case recordReplayFailed(String)
    case formatBuildFailed(format: NumericFormat, error: String)

    var errorDescription: String? {
        switch self {
        case .weightCountMismatch(let names, let weights):
            return "numerics audit: \(names) variable names for \(weights) weight tensors"
        case .masterCountMismatch(let masters, let weights):
            return "numerics audit: \(masters) master tensors for \(weights) working tensors"
        case .tapMissing(let name):
            return "numerics audit: analysis tap \(name) is missing"
        case .tapShapeMismatch(let name, let detail):
            return "numerics audit: analysis tap \(name) has an unexpected shape (\(detail))"
        case .positionSetEmpty:
            return "numerics audit: the position set is empty"
        case .recordReplayFailed(let detail):
            return "numerics audit: \(detail)"
        case .formatBuildFailed(let format, let error):
            return "numerics audit: could not build the \(format.rawValue) network: \(error)"
        }
    }
}
