import Foundation

// MARK: - Dynamic checks
//
// The same weights are built as one audit network per format (fp32, bf16,
// fp16; `ChessNetwork(analysisTaps:)`), and the position set is run through
// each. The fp32 build is the reference: the heads are compared against it
// (value ties, cross-entropy against game results, W/D/L error; policy KL,
// top-move changes and ties, the legal-logit level and spread), and every
// analysis tap gets per-format range, headroom, channel offset-to-spread, and
// error against the fp32 values.

extension NumericsAudit {

    /// Positions per forward pass.
    static let dynamicBatchSize = 64
    /// Percentiles reported for per-position distributions.
    static let distributionPercentiles: [Double] = [5, 25, 50, 75, 95]

    // MARK: Result types

    struct ValueHeadFormatReport: Codable, Sendable {
        let format: NumericFormat
        let nonFiniteFraction: Double
        /// Positions with two or more W/D/L logits exactly equal.
        let tieFraction: Double
        /// Cross-entropy against the game result, over positions with one.
        let crossEntropyMean: Double?
        let crossEntropyDeltaVsFP32: Double?
        /// Mean |Δ(p_win − p_loss)| against fp32.
        let meanAbsDeltaV: Double?
        let argmaxChangedFraction: Double?
        /// The per-position mean of the W/D/L logits (the shared offset), at
        /// `distributionPercentiles`.
        let sharedLogitPercentiles: [Double]
        /// W/D/L at the start position, as this network outputs it.
        let startPositionWDL: [Double]
        let verdict: NumericsVerdict
    }

    struct PolicyHeadFormatReport: Codable, Sendable {
        let format: NumericFormat
        let nonFiniteFraction: Double
        /// Mean KL(fp32 ‖ format) over the legal-move softmax.
        let klMean: Double?
        let top1ChangedFraction: Double?
        /// Positions whose two best legal logits are exactly equal.
        let top2TieFraction: Double
        /// Per-position mean legal logit (its level), at `distributionPercentiles`.
        let legalMeanPercentiles: [Double]
        /// Median per-position standard deviation of the legal logits.
        let legalSpreadMedian: Double
        /// Median per-position mean over all policy logits.
        let allMoveMeanMedian: Double
        let verdict: NumericsVerdict
    }

    struct TapFormatReport: Codable, Sendable {
        let format: NumericFormat
        let maxAbs: Double
        let minNonzeroAbs: Double?
        let nonFiniteFraction: Double
        /// Per-channel |mean| / std, largest and median over channels.
        let maxChannelOffsetToSpread: Double?
        let medianChannelOffsetToSpread: Double?
        /// This format's largest finite value over the fp32 build's largest
        /// |value| at this tap.
        let headroomToFormatMax: Double?
        /// RMS error against the fp32 build over the fp32 values' std.
        let relativeRMSErrorVsFP32: Double?
        let verdict: NumericsVerdict
    }

    struct TapReport: Codable, Sendable {
        let name: String
        /// The tap's shape for one position.
        let perPositionShape: [Int]
        let perFormat: [TapFormatReport]
    }

    struct DynamicResult: Codable, Sendable {
        let positions: PositionSetSummary
        let formatsBuilt: [NumericFormat]
        /// Formats whose network couldn't be built, with the error.
        let formatBuildErrors: [String: String]
        let valueHead: [ValueHeadFormatReport]?
        /// Why there is no value-head report, when there isn't one.
        let valueHeadNote: String?
        let policyHead: [PolicyHeadFormatReport]
        let taps: [TapReport]
    }

    // MARK: Run

    static func runDynamic(
        weights: [[Float]],
        arch: NetworkArchitecture,
        positions: PositionSet,
        policyTailPrecision: ChessNetwork.PolicyTailPrecision
    ) async throws -> DynamicResult {
        guard !positions.positions.isEmpty else { throw NumericsAuditError.positionSetEmpty }

        var networks: [(format: NumericFormat, network: ChessNetwork)] = []
        var buildErrors: [String: String] = [:]
        for format in NumericFormat.allCases {
            var formatArch = arch
            formatArch.computeDataType = format.computeDataType
            do {
                let network = try await buildAuditNetwork(arch: formatArch, policyTailPrecision: policyTailPrecision)
                try await network.loadWeights(weights)
                networks.append((format, network))
            } catch {
                // Without the fp32 reference nothing can be compared.
                if format == .fp32 {
                    throw NumericsAuditError.formatBuildFailed(format: format, error: "\(error)")
                }
                buildErrors[format.rawValue] = "\(error)"
            }
        }

        let accumulator = DynamicAccumulator(
            formats: networks.map(\.format),
            valueClasses: arch.valueHeadClasses
        )
        let all = positions.positions
        let planeCount = arch.inputPlanes * ChessNetwork.boardSize * ChessNetwork.boardSize
        var start = 0
        while start < all.count {
            let batch = Array(all[start..<min(start + dynamicBatchSize, all.count)])
            var boards: [Float] = []
            boards.reserveCapacity(batch.count * planeCount)
            for position in batch { boards.append(contentsOf: position.board) }
            var outputs: [NumericFormat: [ChessNetwork.AnalysisTapValues]] = [:]
            for (format, network) in networks {
                outputs[format] = try await network.evaluateAnalysisTaps(boards: boards, count: batch.count)
            }
            let isFirstBatch = start == 0
            try await accumulate(accumulator: accumulator, batch: batch, outputs: outputs, isFirstBatch: isFirstBatch)
            start += batch.count
        }

        return try accumulator.result(positions: positions.summary, formatsBuilt: networks.map(\.format), buildErrors: buildErrors)
    }

    /// Build an audit network off the cooperative pool (graph construction
    /// is long synchronous work). `policyTailPrecision` is moot for the fp32
    /// build, where the whole graph is fp32 either way.
    private static func buildAuditNetwork(
        arch: NetworkArchitecture, policyTailPrecision: ChessNetwork.PolicyTailPrecision
    ) async throws -> ChessNetwork {
        try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    continuation.resume(returning: try ChessNetwork(
                        arch: arch, bnMode: .inference, initialization: .overwrittenByLoad,
                        policyTailPrecision: policyTailPrecision, analysisTaps: true))
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    /// The per-batch statistics pass is plain CPU work over every tap, so it
    /// runs on a serial queue rather than a cooperative thread.
    private static let accumulationQueue = DispatchQueue(label: "drewschess.numericsaudit.accumulate", qos: .userInitiated)

    private static func accumulate(
        accumulator: DynamicAccumulator,
        batch: [AuditPosition],
        outputs: [NumericFormat: [ChessNetwork.AnalysisTapValues]],
        isFirstBatch: Bool
    ) async throws {
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            accumulationQueue.async {
                do {
                    try accumulator.add(batch: batch, outputs: outputs, isFirstBatch: isFirstBatch)
                    continuation.resume()
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    static func dynamicFindings(_ result: DynamicResult) -> [Finding] {
        var findings: [Finding] = []
        for (format, error) in result.formatBuildErrors.sorted(by: { $0.key < $1.key }) {
            findings.append(Finding(area: "network build", subject: format, format: NumericFormat(rawValue: format), verdict: .bad, detail: error))
        }
        if let valueHead = result.valueHead {
            for report in valueHead where report.verdict != .fine {
                findings.append(Finding(
                    area: "value head", subject: "W/D/L", format: report.format, verdict: report.verdict,
                    detail: "ties \(fmt(report.tieFraction)), CE delta \(fmt(report.crossEntropyDeltaVsFP32)), mean |dv| \(fmt(report.meanAbsDeltaV)), shared logit median \(fmt(report.sharedLogitPercentiles.count > 2 ? report.sharedLogitPercentiles[2] : nil))"
                ))
            }
        }
        for report in result.policyHead where report.verdict != .fine {
            findings.append(Finding(
                area: "policy head", subject: "legal moves", format: report.format, verdict: report.verdict,
                detail: "KL \(fmt(report.klMean)), top-2 ties \(fmt(report.top2TieFraction)), top-1 changed \(fmt(report.top1ChangedFraction)), legal level median \(fmt(report.legalMeanPercentiles.count > 2 ? report.legalMeanPercentiles[2] : nil))"
            ))
        }
        for tap in result.taps {
            for report in tap.perFormat where report.verdict != .fine {
                findings.append(Finding(
                    area: "activations", subject: tap.name, format: report.format, verdict: report.verdict,
                    detail: "max |x| \(fmt(report.maxAbs)), headroom \(fmt(report.headroomToFormatMax)), max channel |mean|/std \(fmt(report.maxChannelOffsetToSpread)), error vs fp32 \(fmt(report.relativeRMSErrorVsFP32)), non-finite \(fmt(report.nonFiniteFraction))"
                ))
            }
        }
        return findings
    }
}

// MARK: - Accumulation

/// Running statistics across batches. Used strictly sequentially: each batch
/// is added on `NumericsAudit.accumulationQueue` and awaited before the next,
/// and `result` runs after the last — never concurrently.
private final class DynamicAccumulator: @unchecked Sendable {

    struct TapStats {
        var perPositionShape: [Int] = []
        var channels = 1
        var inner = 1
        var channelSum: [Double] = []
        var channelSumSq: [Double] = []
        var channelCount: [Int] = []
        var maxAbs = 0.0
        var minNonzeroAbs = Double.infinity
        var nonFinite = 0
        var total = 0
        var sum = 0.0
        var sumSq = 0.0
        var errorSq = 0.0
        var errorCount = 0
    }

    struct ValueStats {
        var positions = 0
        var nonFinite = 0
        var ties = 0
        var ceSum = 0.0
        var ceCount = 0
        var dvSum = 0.0
        var dvCount = 0
        var argmaxChanged = 0
        var argmaxCount = 0
        var sharedLogits: [Double] = []
        var startWDL: [Double] = []
    }

    struct PolicyStats {
        var positions = 0
        var nonFinite = 0
        var klSum = 0.0
        var klCount = 0
        var top1Changed = 0
        var top1Count = 0
        var top2Ties = 0
        var top2Count = 0
        var legalMeans: [Double] = []
        var legalSpreads: [Double] = []
        var allMoveMeans: [Double] = []
    }

    let formats: [NumericFormat]
    let valueClasses: Int
    private var tapOrder: [String] = []
    private var taps: [String: [NumericFormat: TapStats]] = [:]
    private var value: [NumericFormat: ValueStats] = [:]
    private var policy: [NumericFormat: PolicyStats] = [:]

    init(formats: [NumericFormat], valueClasses: Int) {
        self.formats = formats
        self.valueClasses = valueClasses
        for format in formats {
            value[format] = ValueStats()
            policy[format] = PolicyStats()
        }
    }

    func add(batch: [NumericsAudit.AuditPosition], outputs: [NumericFormat: [ChessNetwork.AnalysisTapValues]], isFirstBatch: Bool) throws {
        guard let reference = outputs[.fp32] else { throw NumericsAuditError.tapMissing("fp32 reference outputs") }
        let count = batch.count
        if tapOrder.isEmpty {
            tapOrder = reference.map(\.name)
        }
        var referenceByName: [String: ChessNetwork.AnalysisTapValues] = [:]
        for tap in reference { referenceByName[tap.name] = tap }

        for format in formats {
            guard let formatOutputs = outputs[format] else { throw NumericsAuditError.tapMissing("\(format.rawValue) outputs") }
            var byName: [String: ChessNetwork.AnalysisTapValues] = [:]
            for tap in formatOutputs { byName[tap.name] = tap }

            for name in tapOrder {
                guard let tap = byName[name], let ref = referenceByName[name] else { throw NumericsAuditError.tapMissing(name) }
                guard tap.shape.first == count, tap.values.count == ref.values.count else {
                    throw NumericsAuditError.tapShapeMismatch(name: name, detail: "shape \(tap.shape) for \(count) positions")
                }
                // Accumulators start empty on a tap's first batch.
                accumulateTap(&taps[name, default: [:]][format, default: TapStats()], tap: tap, reference: ref)
            }

            guard let policyTap = byName["policy_logits"], let policyRef = referenceByName["policy_logits"] else {
                throw NumericsAuditError.tapMissing("policy_logits")
            }
            guard let valueTap = byName["value_logits"], let valueRef = referenceByName["value_logits"],
                  let probsTap = byName["value_probs"] else {
                throw NumericsAuditError.tapMissing("value_logits / value_probs")
            }
            let policyWidth = policyTap.values.count / count
            let valueWidth = valueTap.values.count / count
            for (row, position) in batch.enumerated() {
                accumulatePolicy(format: format, row: row, width: policyWidth, logits: policyTap.values, reference: policyRef.values, legal: position.legalPolicyIndices)
                if valueWidth == 3 {
                    accumulateValue(format: format, row: row, logits: valueTap.values, reference: valueRef.values, target: position.valueTarget)
                }
            }
            if isFirstBatch, var stats = value[format] {
                let probsWidth = probsTap.values.count / count
                stats.startWDL = probsTap.values[0..<probsWidth].map { Double($0) }
                value[format] = stats
            }
        }
    }

    private func accumulateTap(_ stats: inout TapStats, tap: ChessNetwork.AnalysisTapValues, reference: ChessNetwork.AnalysisTapValues) {
        let shape = tap.shape
        if stats.channelSum.isEmpty {
            stats.perPositionShape = Array(shape.dropFirst())
            stats.channels = shape.count >= 2 ? shape[1] : 1
            stats.inner = shape.count > 2 ? shape[2...].reduce(1, *) : 1
            stats.channelSum = [Double](repeating: 0, count: stats.channels)
            stats.channelSumSq = [Double](repeating: 0, count: stats.channels)
            stats.channelCount = [Int](repeating: 0, count: stats.channels)
        }
        let channels = stats.channels
        let inner = stats.inner
        let perPosition = channels * inner
        tap.values.withUnsafeBufferPointer { values in
            reference.values.withUnsafeBufferPointer { ref in
                for index in 0..<values.count {
                    let v = values[index]
                    stats.total += 1
                    guard v.isFinite else {
                        stats.nonFinite += 1
                        continue
                    }
                    let d = Double(v)
                    let magnitude = abs(d)
                    stats.maxAbs = max(stats.maxAbs, magnitude)
                    if magnitude > 0 { stats.minNonzeroAbs = min(stats.minNonzeroAbs, magnitude) }
                    stats.sum += d
                    stats.sumSq += d * d
                    let channel = (index % perPosition) / inner
                    stats.channelSum[channel] += d
                    stats.channelSumSq[channel] += d * d
                    stats.channelCount[channel] += 1
                    let r = ref[index]
                    if r.isFinite {
                        let e = d - Double(r)
                        stats.errorSq += e * e
                        stats.errorCount += 1
                    }
                }
            }
        }
    }

    private func accumulatePolicy(format: NumericFormat, row: Int, width: Int, logits: [Float], reference: [Float], legal: [Int]) {
        guard var stats = policy[format] else { return }
        stats.positions += 1
        let base = row * width
        var allSum = 0.0
        var finite = true
        for i in 0..<width {
            let v = logits[base + i]
            if !v.isFinite { finite = false }
            allSum += Double(v)
        }
        guard finite, !legal.isEmpty else {
            if !finite { stats.nonFinite += 1 }
            policy[format] = stats
            return
        }
        stats.allMoveMeans.append(allSum / Double(width))
        let own = legal.map { Double(logits[base + $0]) }
        let ref = legal.map { Double(reference[base + $0]) }
        let mean = own.reduce(0, +) / Double(own.count)
        let variance = own.reduce(0) { $0 + ($1 - mean) * ($1 - mean) } / Double(own.count)
        stats.legalMeans.append(mean)
        stats.legalSpreads.append(variance.squareRoot())

        if own.count >= 2 {
            let sorted = own.sorted(by: >)
            stats.top2Count += 1
            if sorted[0] == sorted[1] { stats.top2Ties += 1 }
        }
        if ref.allSatisfy({ $0.isFinite }) {
            let pRef = Self.softmax(ref)
            let pOwn = Self.softmax(own)
            var kl = 0.0
            for i in 0..<pRef.count where pRef[i] > 0 {
                kl += pRef[i] * (log(pRef[i]) - log(max(pOwn[i], Double.leastNormalMagnitude)))
            }
            stats.klSum += kl
            stats.klCount += 1
            stats.top1Count += 1
            if Self.argmax(ref) != Self.argmax(own) { stats.top1Changed += 1 }
        }
        policy[format] = stats
    }

    private func accumulateValue(format: NumericFormat, row: Int, logits: [Float], reference: [Float], target: Int?) {
        guard var stats = value[format] else { return }
        stats.positions += 1
        let own = (0..<3).map { Double(logits[row * 3 + $0]) }
        let ref = (0..<3).map { Double(reference[row * 3 + $0]) }
        guard own.allSatisfy({ $0.isFinite }) else {
            stats.nonFinite += 1
            value[format] = stats
            return
        }
        if own[0] == own[1] || own[0] == own[2] || own[1] == own[2] { stats.ties += 1 }
        stats.sharedLogits.append(own.reduce(0, +) / 3)
        let p = Self.softmax(own)
        if let target {
            stats.ceSum += -log(max(p[target], Double.leastNormalMagnitude))
            stats.ceCount += 1
        }
        if ref.allSatisfy({ $0.isFinite }) {
            let pRef = Self.softmax(ref)
            stats.dvSum += abs((p[0] - p[2]) - (pRef[0] - pRef[2]))
            stats.dvCount += 1
            stats.argmaxCount += 1
            if Self.argmax(own) != Self.argmax(ref) { stats.argmaxChanged += 1 }
        }
        value[format] = stats
    }

    private static func softmax(_ logits: [Double]) -> [Double] {
        guard let top = logits.max() else { return [] }
        let exps = logits.map { exp($0 - top) }
        let total = exps.reduce(0, +)
        return exps.map { $0 / total }
    }

    /// First index of the largest value.
    private static func argmax(_ values: [Double]) -> Int {
        var best = 0
        for i in values.indices where values[i] > values[best] { best = i }
        return best
    }

    private static func percentiles(_ values: [Double]) -> [Double] {
        let sorted = values.sorted()
        return NumericsAudit.distributionPercentiles.map { NetworkWeightAnalyzer.percentile(p: $0, sortedAscending: sorted) }
    }

    /// A measurement that exists and reaches `threshold`. A missing one (no
    /// positions or values to measure) never trips a verdict.
    private static func atLeast(_ value: Double?, _ threshold: Double) -> Bool {
        guard let value else { return false }
        return value >= threshold
    }

    /// A measurement that exists and falls below `threshold`.
    private static func below(_ value: Double?, _ threshold: Double) -> Bool {
        guard let value else { return false }
        return value < threshold
    }

    private static func fraction(_ part: Int, _ whole: Int) -> Double {
        whole > 0 ? Double(part) / Double(whole) : 0
    }

    func result(positions: NumericsAudit.PositionSetSummary, formatsBuilt: [NumericFormat], buildErrors: [String: String]) throws -> NumericsAudit.DynamicResult {
        typealias T = NumericsAudit.Threshold

        let referenceValue = value[.fp32]
        let referenceCE: Double? = referenceValue.flatMap { $0.ceCount > 0 ? $0.ceSum / Double($0.ceCount) : nil }

        var valueReports: [NumericsAudit.ValueHeadFormatReport]?
        var valueNote: String?
        if valueClasses == 3 {
            valueReports = formats.compactMap { format in
                guard let s = value[format] else { return nil }
                let ce: Double? = s.ceCount > 0 ? s.ceSum / Double(s.ceCount) : nil
                let ceDelta: Double? = ce.flatMap { own in referenceCE.map { own - $0 } }
                let ties = Self.fraction(s.ties, s.positions)
                let nonFinite = Self.fraction(s.nonFinite, s.positions)
                var verdict = NumericsVerdict.fine
                if nonFinite > 0 || Self.atLeast(ceDelta, T.valueCEDeltaBad) {
                    verdict = .bad
                } else if ties >= T.valueTieFractionDegraded || Self.atLeast(ceDelta, T.valueCEDeltaDegraded) {
                    verdict = .degraded
                }
                return NumericsAudit.ValueHeadFormatReport(
                    format: format,
                    nonFiniteFraction: nonFinite,
                    tieFraction: ties,
                    crossEntropyMean: ce,
                    crossEntropyDeltaVsFP32: ceDelta,
                    meanAbsDeltaV: s.dvCount > 0 ? s.dvSum / Double(s.dvCount) : nil,
                    argmaxChangedFraction: s.argmaxCount > 0 ? Self.fraction(s.argmaxChanged, s.argmaxCount) : nil,
                    sharedLogitPercentiles: Self.percentiles(s.sharedLogits),
                    startPositionWDL: s.startWDL,
                    verdict: verdict
                )
            }
        } else {
            valueNote = "the value head has \(valueClasses) output(s), not W/D/L; its checks don't apply"
        }

        let policyReports: [NumericsAudit.PolicyHeadFormatReport] = formats.compactMap { format in
            guard let s = policy[format] else { return nil }
            let kl: Double? = s.klCount > 0 ? s.klSum / Double(s.klCount) : nil
            let ties = Self.fraction(s.top2Ties, s.top2Count)
            let nonFinite = Self.fraction(s.nonFinite, s.positions)
            var verdict = NumericsVerdict.fine
            if nonFinite > 0 || Self.atLeast(kl, T.policyKLBad) || ties >= T.policyTop2TieBad {
                verdict = .bad
            } else if Self.atLeast(kl, T.policyKLDegraded) || ties >= T.policyTop2TieDegraded {
                verdict = .degraded
            }
            return NumericsAudit.PolicyHeadFormatReport(
                format: format,
                nonFiniteFraction: nonFinite,
                klMean: kl,
                top1ChangedFraction: s.top1Count > 0 ? Self.fraction(s.top1Changed, s.top1Count) : nil,
                top2TieFraction: ties,
                legalMeanPercentiles: Self.percentiles(s.legalMeans),
                legalSpreadMedian: NetworkWeightAnalyzer.percentile(p: 50, sortedAscending: s.legalSpreads.sorted()),
                allMoveMeanMedian: NetworkWeightAnalyzer.percentile(p: 50, sortedAscending: s.allMoveMeans.sorted()),
                verdict: verdict
            )
        }

        var tapReports: [NumericsAudit.TapReport] = []
        for name in tapOrder {
            guard let perFormat = taps[name], let reference = perFormat[.fp32] else { throw NumericsAuditError.tapMissing(name) }
            let referenceCount = reference.total - reference.nonFinite
            let referenceStd: Double? = referenceCount > 0
                ? max(0, reference.sumSq / Double(referenceCount) - pow(reference.sum / Double(referenceCount), 2)).squareRoot()
                : nil
            let isNormalizationInput = name.hasSuffix("_input")
            let reports: [NumericsAudit.TapFormatReport] = formats.compactMap { format in
                guard let s = perFormat[format] else { return nil }
                var ratios: [Double] = []
                for c in 0..<s.channelSum.count where s.channelCount[c] > 0 {
                    let n = Double(s.channelCount[c])
                    let mean = s.channelSum[c] / n
                    let std = max(0, s.channelSumSq[c] / n - mean * mean).squareRoot()
                    if std > 0 { ratios.append(abs(mean) / std) }
                }
                let sortedRatios = ratios.sorted()
                let maxRatio = sortedRatios.last
                let medianRatio: Double? = sortedRatios.isEmpty ? nil : NetworkWeightAnalyzer.percentile(p: 50, sortedAscending: sortedRatios)
                let headroom: Double? = reference.maxAbs > 0 ? format.maxFinite / reference.maxAbs : nil
                let relativeError: Double? = format == .fp32 ? nil : referenceStd.flatMap { std in
                    std > 0 && s.errorCount > 0 ? (s.errorSq / Double(s.errorCount)).squareRoot() / std : nil
                }
                let nonFinite = Self.fraction(s.nonFinite, s.total)
                var verdict = NumericsVerdict.fine
                let checksOffset = format != .fp32 && isNormalizationInput
                if nonFinite > 0 || Self.below(headroom, 1) || Self.atLeast(relativeError, T.tapRelativeErrorBad)
                    || (checksOffset && Self.atLeast(maxRatio, T.offsetToSpreadBad)) {
                    verdict = .bad
                } else if Self.below(headroom, T.overflowHeadroomDegraded) || Self.atLeast(relativeError, T.tapRelativeErrorDegraded)
                    || (checksOffset && Self.atLeast(maxRatio, T.offsetToSpreadDegraded)) {
                    verdict = .degraded
                }
                return NumericsAudit.TapFormatReport(
                    format: format,
                    maxAbs: s.maxAbs,
                    minNonzeroAbs: s.minNonzeroAbs.isFinite ? s.minNonzeroAbs : nil,
                    nonFiniteFraction: nonFinite,
                    maxChannelOffsetToSpread: maxRatio,
                    medianChannelOffsetToSpread: medianRatio,
                    headroomToFormatMax: headroom,
                    relativeRMSErrorVsFP32: relativeError,
                    verdict: verdict
                )
            }
            tapReports.append(NumericsAudit.TapReport(name: name, perPositionShape: reference.perPositionShape, perFormat: reports))
        }

        return NumericsAudit.DynamicResult(
            positions: positions,
            formatsBuilt: formatsBuilt,
            formatBuildErrors: buildErrors,
            valueHead: valueReports,
            valueHeadNote: valueNote,
            policyHead: policyReports,
            taps: tapReports
        )
    }
}
