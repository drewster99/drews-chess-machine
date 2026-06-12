import Foundation

// MARK: - Value Head Analyzer
//
// Snapshot-time diagnostic for the network's value-head weights.
// Answers the question: "is the value head riding the prior, or is it
// actually conditioning on its input?"
//
// The value head's structure (see `ChessNetwork.valueHead`):
//
//   1×1 conv (channels → valueHeadConvChannels) — `value_conv_weights`
//   BatchNorm γ, β            — `value_bn_gamma|beta`  (valueHeadConvChannels each)
//   FC flatten → hidden       — `value_fc1_weights`
//   FC flatten → hidden bias  — `value_fc1_bias`
//   FC hidden → 3 (W/D/L)     — `value_fc2_weights`
//   FC hidden → 3 bias        — `value_fc2_bias`              3 floats
//
// (flatten = boardSize² × valueHeadConvChannels; hidden =
// valueHeadHiddenUnits — see `ChessNetwork.valueHead`.)
//
// The two highest-signal reads are `value_fc2_weights` (output-layer
// magnitudes — if these are near zero the head is bias-only) and
// `value_fc2_bias` (initialized to `[0, ln 6, 0]` for a draw-heavy
// 0.125 / 0.75 / 0.125 prior; current value plus its softmax shows
// how much the bias-only prediction has shifted toward the empirical
// W/D/L distribution).
//
// The BN running stats (`value_bn_running_mean|var`) are also pulled
// in even though they're not "weights" per se — they're learned
// statistics that affect inference behavior, and a sanity check on
// their magnitudes is cheap.
//
// Pure analysis — no Metal/GPU work besides the single
// `network.exportWeights()` call that's already in the codebase for
// session checkpoint serialization.

enum ValueHeadAnalyzer {

    // MARK: - He-init reference scales

    /// Reference He-init L2 norm for a weight tensor, used as the
    /// denominator of the `currentL2 / initL2` ratio reported per
    /// variable. Computed as `sqrt(N) · std`, where `std = sqrt(2 / fanIn)`
    /// is the per-element He-init standard deviation and `N` is the
    /// tensor's element count. Mirrors `ChessNetwork.heInitDataConvOIHW`
    /// / `heInitDataFCInOut`. A ratio near 1 means the tensor's
    /// magnitude is close to its initialization scale (weight decay
    /// hasn't pulled it down much); a ratio near 0 means the tensor
    /// has collapsed toward zero.
    private static func heInitL2(elementCount n: Int, fanIn: Int) -> Double {
        guard n > 0, fanIn > 0 else { return 0 }
        let std = sqrt(2.0 / Double(fanIn))
        return sqrt(Double(n)) * std
    }

    /// Per-variable fan-in for the He-init reference. Returns `nil`
    /// for tensors that don't have a meaningful "init L2" — BN
    /// gamma/beta (init constants, not He-init), BN running stats
    /// (statistics, not weights), and the FC biases (init to zero).
    private static func fanIn(forVariableNamed name: String, arch: NetworkArchitecture) -> Int? {
        switch name {
        case "value_conv_weights":   return arch.towerOutputChannels  // 1×1 conv: inC = tower output
        case "value_fc1_weights":    return ChessNetwork.boardSize * ChessNetwork.boardSize * arch.valueHeadConvChannels  // FC [flatten, hidden], fan_in = flatten
        case "value_fc2_weights":    return arch.valueHeadHiddenUnits  // FC [hidden, classes], fan_in = hidden
        default:                     return nil         // bn/bias: no He-init reference
        }
    }

    /// Initial value of `value_fc2_bias` as set in
    /// `ChessNetwork.valueHead`: `[0, ln 6, 0]`, which softmaxes to
    /// `[0.125, 0.75, 0.125]` — the empirically draw-heavy prior of
    /// a fresh self-play buffer. Reported so the JSON consumer can
    /// see the delta between current and initial bias side by side.
    static let valueFC2BiasInitial: [Double] = [0.0, log(6.0), 0.0]

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
            /// `l2Norm / heInitL2` when the variable has a He-init
            /// reference; `nil` for tensors that don't (BN γ/β,
            /// running stats, FC biases). A value near 1.0 means the
            /// tensor is still at roughly its init scale; near 0
            /// means weight decay has pulled it close to zero.
            let l2NormRatioToInit: Double?
            /// Reference He-init L2 norm — kept alongside the ratio
            /// so downstream readers don't have to recompute it.
            let initL2Norm: Double?
        }

        struct FC2BiasDetail: Codable, Sendable {
            /// Current 3-element bias values, in slot order
            /// `[win, draw, loss]`.
            let current: [Double]
            /// Softmax of `current` — the bias-only prediction the
            /// network would produce if every input multiplied to
            /// zero. Compare against the empirical buffer W/D/L
            /// distribution to see whether the bias is tracking it.
            let currentSoftmax: [Double]
            /// Initial value `[0, ln 6, 0]`, included so the
            /// per-slot delta is computable without consulting the
            /// network code.
            let initial: [Double]
            /// Initial softmax `[0.125, 0.75, 0.125]`, same rationale.
            let initialSoftmax: [Double]
            /// Per-slot delta `current[i] - initial[i]`.
            let delta: [Double]
        }

        struct FC2WeightsDetail: Codable, Sendable {
            /// Per-output-column L2 norm of `value_fc2_weights`.
            /// Indexed `[win, draw, loss]`. The `value_fc2_weights`
            /// tensor has shape `[64, 3]` (in × out); for column `c`
            /// we sum the squares of `weights[i, c]` for i in 0..<64.
            /// A near-zero norm for, say, the `draw` column would
            /// say the network puts no input-dependent information
            /// into its draw prediction (it's all bias).
            let columnL2Norms: [Double]
            /// Reference He-init per-column L2 norm — same
            /// He-init reference applied to a 64-element column.
            let initColumnL2Norm: Double
        }

        let producedAtISO8601: String
        let modelLabel: String

        /// Cross-cutting training-progress context (step count, elapsed
        /// time, build/git provenance). Stamped on by `SessionController`
        /// at export time — the analyzer leaves it `nil` and a `nil`
        /// optional omits its key, so analyzer-only callers and tests
        /// produce JSON unchanged from before this field existed.
        var exportMetadata: AnalysisExportMetadata? = nil
        let weightStats: [WeightStats]
        let fc2Bias: FC2BiasDetail?
        let fc2Weights: FC2WeightsDetail?
    }

    // MARK: - Entry point

    /// Run the analyzer against `network`. Calls `exportWeights()`,
    /// filters for `value_*` variables by name, computes stats, and
    /// returns a single `Result`.
    ///
    /// `network` must be the inference or training network of interest;
    /// the analyzer doesn't care which but the JSON's `modelLabel`
    /// should reflect the caller's choice so downstream readers know
    /// what was analyzed.
    static func run(
        network: ChessMPSNetwork,
        modelLabel: String
    ) async throws -> Result {
        // exportWeights returns one [Float] per variable, in the order
        // `trainableVariables + bnRunningStatsVariables`. Match each
        // chunk to its variable's name via the same join. The
        // variable lists live on the underlying `ChessNetwork`, not
        // the `ChessMPSNetwork` wrapper, hence the `.network.` hop.
        let weights = try await network.exportWeights()
        let arch = network.network.arch
        let allVariables = network.network.trainableVariables
            + network.network.bnRunningStatsVariables

        guard weights.count == allVariables.count else {
            throw ValueHeadAnalyzerError.weightCountMismatch(
                expected: allVariables.count,
                got: weights.count
            )
        }

        var stats: [Result.WeightStats] = []
        var fc2BiasDetail: Result.FC2BiasDetail?
        var fc2WeightsDetail: Result.FC2WeightsDetail?

        for (i, variable) in allVariables.enumerated() {
            let name = variable.operation.name
            guard name.hasPrefix("value_") else { continue }
            let values = weights[i]

            stats.append(makeStats(name: name, values: values, arch: arch))

            if name == "value_fc2_bias" {
                fc2BiasDetail = makeFC2BiasDetail(values: values)
            } else if name == "value_fc2_weights" {
                fc2WeightsDetail = makeFC2WeightsDetail(values: values, arch: arch)
            }
        }

        let iso = ISO8601DateFormatter()
        iso.formatOptions = [.withInternetDateTime]

        return Result(
            producedAtISO8601: iso.string(from: Date()),
            modelLabel: modelLabel,
            weightStats: stats,
            fc2Bias: fc2BiasDetail,
            fc2Weights: fc2WeightsDetail
        )
    }

    // MARK: - Per-variable stats

    /// Percentiles (in 0..100) reported per variable in
    /// `Result.WeightStats.percentiles`. The labels must match what's
    /// written into the JSON; they're a static constant so JSON
    /// consumers and the text-summary formatter both stay in sync.
    static let percentileLabels: [Int] = [10, 50, 90]

    private static func makeStats(
        name: String,
        values: [Float],
        arch: NetworkArchitecture
    ) -> Result.WeightStats {
        let n = values.count
        guard n > 0 else {
            return Result.WeightStats(
                name: name,
                elementCount: 0,
                l1Norm: 0, l2Norm: 0, meanAbs: 0,
                min: 0, max: 0, mean: 0, stdev: 0,
                percentiles: Array(repeating: 0, count: percentileLabels.count),
                l2NormRatioToInit: nil,
                initL2Norm: nil
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
        let l1 = sumAbs
        let l2 = sqrt(sumSq)
        let meanAbs = sumAbs / dN

        // Percentiles: sort ascending and linearly interpolate.
        let sorted = values.map { Double($0) }.sorted()
        let percentiles = percentileLabels.map { p in
            percentile(p: Double(p), sortedAscending: sorted)
        }

        let initL2 = fanIn(forVariableNamed: name, arch: arch).map {
            heInitL2(elementCount: n, fanIn: $0)
        }
        let ratio = initL2.map { $0 > 0 ? l2 / $0 : 0 }

        return Result.WeightStats(
            name: name,
            elementCount: n,
            l1Norm: l1,
            l2Norm: l2,
            meanAbs: meanAbs,
            min: vMin,
            max: vMax,
            mean: mean,
            stdev: stdev,
            percentiles: percentiles,
            l2NormRatioToInit: ratio,
            initL2Norm: initL2
        )
    }

    private static func makeFC2BiasDetail(values: [Float]) -> Result.FC2BiasDetail? {
        guard values.count == 3 else { return nil }
        let current = values.map { Double($0) }
        let initial = valueFC2BiasInitial
        let currentSoftmax = softmax(current)
        let initialSoftmax = softmax(initial)
        let delta = zip(current, initial).map { $0 - $1 }
        return Result.FC2BiasDetail(
            current: current,
            currentSoftmax: currentSoftmax,
            initial: initial,
            initialSoftmax: initialSoftmax,
            delta: delta
        )
    }

    private static func makeFC2WeightsDetail(values: [Float], arch: NetworkArchitecture) -> Result.FC2WeightsDetail? {
        // The `value_fc2_weights` tensor has shape [hidden, 3] (in × out)
        // and is stored row-major (every 3 consecutive floats are the
        // weights from one input neuron to W/D/L). To get the L2 norm
        // of the `[c]` output column, sum squares of `values[i*outDim + c]`
        // for i in 0..<hidden. Dims come from `ChessNetwork` so this tracks
        // the value-head shape — they're structural facts, not tunables.
        let outDim = arch.valueHeadClasses
        let inDim = arch.valueHeadHiddenUnits
        guard values.count == inDim * outDim else { return nil }
        var columnSumSq = [Double](repeating: 0, count: outDim)
        for i in 0..<inDim {
            for c in 0..<outDim {
                let v = Double(values[i * outDim + c])
                columnSumSq[c] += v * v
            }
        }
        let columnL2 = columnSumSq.map { sqrt($0) }
        let initColL2 = heInitL2(elementCount: inDim, fanIn: inDim)
        return Result.FC2WeightsDetail(
            columnL2Norms: columnL2,
            initColumnL2Norm: initColL2
        )
    }

    // MARK: - Numeric helpers

    /// Linear-interpolation percentile for `p` in 0..100 over
    /// ascending-sorted `sortedAscending`. Empty input returns 0.
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

    /// Numerically stable softmax over a small array of Doubles.
    /// Subtracts the max before exponentiating so very large logits
    /// don't overflow. The value-head FC2 bias is 3 elements so this
    /// could be done naively; using the stable form anyway costs
    /// nothing and avoids a footgun if the function is reused.
    static func softmax(_ x: [Double]) -> [Double] {
        guard !x.isEmpty else { return [] }
        let m = x.max() ?? 0
        let exps = x.map { exp($0 - m) }
        let sum = exps.reduce(0, +)
        guard sum > 0 else { return Array(repeating: 1.0 / Double(x.count), count: x.count) }
        return exps.map { $0 / sum }
    }
}

// MARK: - Error type

enum ValueHeadAnalyzerError: LocalizedError {
    case weightCountMismatch(expected: Int, got: Int)

    var errorDescription: String? {
        switch self {
        case .weightCountMismatch(let expected, let got):
            return "ValueHeadAnalyzer: exportWeights() returned \(got) tensors, expected \(expected) (trainables + BN running stats)"
        }
    }
}

// MARK: - Text summary

extension ValueHeadAnalyzer.Result {

    /// Multi-line human-readable digest of the result, formatted for
    /// the session log and a CLI/NSAlert preview. Mirrors the
    /// `ReplayBufferAnalyzer.Result.textSummary()` style.
    func textSummary() -> String {
        var out = ""

        let pctLabels = ValueHeadAnalyzer.percentileLabels
        let pctHeader = pctLabels.map { "p\($0)" }.joined(separator: "/")

        out += "Value head weight analysis (model: \(modelLabel))\n"
        out += "  produced: \(producedAtISO8601)\n\n"

        // Per-variable stats table.
        out += "Per-variable stats:\n"
        out += String(format: "  %@  %@  %@  %@  %@  %@  %@  %@  %@\n",
                      "name".padding(toLength: 26, withPad: " ", startingAt: 0),
                      "count".padded(7),
                      "L2".padded(9),
                      "init_L2".padded(8),
                      "ratio".padded(7),
                      "meanAbs".padded(8),
                      "min".padded(9),
                      "max".padded(9),
                      pctHeader.padded(20))
        for s in weightStats {
            let ratioStr = s.l2NormRatioToInit.map { String(format: "%6.3f", $0) } ?? "  --  "
            let initL2Str = s.initL2Norm.map { String(format: "%6.3f", $0) } ?? "  --  "
            let pctStr = s.percentiles
                .map { String(format: "%+6.3f", $0) }
                .joined(separator: "/")
            out += String(format: "  %@  %@  %@  %@  %@  %@  %@  %@  %@\n",
                          s.name.padding(toLength: 26, withPad: " ", startingAt: 0),
                          String(s.elementCount).padded(7),
                          String(format: "%7.3f", s.l2Norm).padded(9),
                          initL2Str.padded(8),
                          ratioStr.padded(7),
                          String(format: "%6.4f", s.meanAbs).padded(8),
                          String(format: "%+7.3f", s.min).padded(9),
                          String(format: "%+7.3f", s.max).padded(9),
                          pctStr.padded(20))
        }
        out += "\n"

        // FC2 bias detail.
        if let bias = fc2Bias {
            out += "value_fc2_bias — bias-only prediction:\n"
            out += String(format: "  current values        : %@\n",
                          bias.current.map { String(format: "%+7.4f", $0) }.joined(separator: ", "))
            out += String(format: "  current softmax (WDL) : %@\n",
                          bias.currentSoftmax.map { String(format: "%.4f", $0) }.joined(separator: ", "))
            out += String(format: "  initial values        : %@\n",
                          bias.initial.map { String(format: "%+7.4f", $0) }.joined(separator: ", "))
            out += String(format: "  initial softmax (WDL) : %@\n",
                          bias.initialSoftmax.map { String(format: "%.4f", $0) }.joined(separator: ", "))
            out += String(format: "  delta from initial    : %@\n",
                          bias.delta.map { String(format: "%+7.4f", $0) }.joined(separator: ", "))
            out += "\n"
        }

        // FC2 weights per-output-column.
        if let fc2w = fc2Weights {
            out += "value_fc2_weights — per-output-column L2 norms (vs init_L2 ≈ \(String(format: "%.3f", fc2w.initColumnL2Norm))):\n"
            let names = ["win", "draw", "loss"]
            for (i, n) in names.enumerated() where i < fc2w.columnL2Norms.count {
                let v = fc2w.columnL2Norms[i]
                let ratio = fc2w.initColumnL2Norm > 0 ? v / fc2w.initColumnL2Norm : 0
                out += String(format: "  %@: L2=%6.3f  ratio=%6.3f\n", n.padded(5), v, ratio)
            }
        }

        return out
    }
}

// MARK: - Small string padding helper

private extension String {
    /// Right-pad the receiver to `length` characters with spaces.
    /// Used by the text-summary table formatter.
    func padded(_ length: Int) -> String {
        if count >= length { return self }
        return self + String(repeating: " ", count: length - count)
    }
}
