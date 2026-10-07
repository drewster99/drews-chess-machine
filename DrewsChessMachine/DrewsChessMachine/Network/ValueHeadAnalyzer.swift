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
//   FC hidden → 3 (W/D/L)     — `value_wdl_fc2_weights`
//   FC hidden → 3 bias        — `value_wdl_fc2_bias`              3 floats
//
// (flatten = boardSize² × valueHeadConvChannels; hidden =
// valueHeadHiddenUnits — see `ChessNetwork.valueHead`.)
//
// The two highest-signal reads are `value_wdl_fc2_weights` (output-layer
// magnitudes — if these are near zero the head is bias-only) and
// `value_wdl_fc2_bias` (initialized to
// `NetworkArchitecture.wdlBiasPrior(drawProbability:)` of the model's
// `value_head_draw_prior`; `[0, ln 6, 0]` for the standard 0.75). The
// bias's softmax is what the head would predict if fc2's input were zero —
// not what it predicts: the hidden activations are not zero-mean, so fc2's
// weights add a per-class offset of their own. The head's real W/D/L
// prediction is `pW` / `pD` / `pL` on the `[STATS]` line.
//
// Every "init" figure comes from the snapshot's init reference
// (`AnalysisInitReference`): the architecture built by the network builder
// itself, under the model's own init seed when it is recorded — so a model
// built with another final init or draw prior is compared with its own
// starting point, and no init rule is copied here.
//
// The BN running stats (`value_bn_running_mean|var`) are also pulled
// in even though they're not "weights" per se — they're learned
// statistics that affect inference behavior, and a sanity check on
// their magnitudes is cheap. They are reported without init figures:
// their start depends on how the model was built (`AnalysisInitReference`).
// An init figure that includes a same-distribution draw rather than the
// model's own initial values is flagged (`initExact` false; "~" in the
// text summary).
//
// Pure analysis over an `AnalyzedNetworkSnapshot` (one export, shared with
// the other analyses of the same request).

enum ValueHeadAnalyzer {

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
            /// `l2Norm / initL2Norm`; nil when the tensor started at zero or is
            /// a BN running statistic. Near 1.0, the tensor is still at roughly
            /// its init scale; near 0, weight decay has pulled it close to zero.
            let l2NormRatioToInit: Double?
            /// The tensor's L2 norm at init (init reference): exact when
            /// `initExact`, else a draw of the same distribution; nil for a BN
            /// running statistic.
            let initL2Norm: Double?
            /// Whether the initial values are known exactly; false for a BN
            /// running statistic.
            let initExact: Bool
        }

        struct FC2BiasDetail: Codable, Sendable {
            /// Current 3-element bias values, in slot order
            /// `[win, draw, loss]`.
            let current: [Double]
            /// Softmax of `current` alone: what the head would predict if
            /// fc2's input were zero. NOT the head's prediction — the hidden
            /// activations are not zero-mean, so fc2's weights add their own
            /// per-class offset; `pW` / `pD` / `pL` on the `[STATS]` line is
            /// the prediction.
            let biasOnlySoftmax: [Double]
            /// Whether the bias's start is known exactly. The builder never
            /// draws it from the seed (it is the draw prior's bias), so it is.
            let initExact: Bool
            /// The bias this model started with (init reference: the draw
            /// prior's bias as the model stores it), so the per-slot delta is
            /// computable without the network code; nil unless `initExact`,
            /// since a delta from a draw means nothing.
            let initial: [Double]?
            /// Softmax of `initial` alone, read the same way as
            /// `biasOnlySoftmax`; nil unless `initExact`.
            let initialBiasOnlySoftmax: [Double]?
            /// Per-slot delta `current[i] - initial[i]`; nil unless `initExact`.
            let delta: [Double]?
        }

        struct FC2WeightsDetail: Codable, Sendable {
            /// Per-output-column L2 norm of `value_wdl_fc2_weights`.
            /// Indexed `[win, draw, loss]`. The `value_wdl_fc2_weights`
            /// tensor has shape `[valueHeadHiddenUnits, 3]` (in × out);
            /// for column `c` we sum the squares of `weights[i, c]` over
            /// every input `i`.
            /// A near-zero norm for, say, the `draw` column would
            /// say the network puts no input-dependent information
            /// into its draw prediction (it's all bias).
            let columnL2Norms: [Double]
            /// Each column's L2 norm at init (init reference): about √2 for
            /// a He-normal fc2, 0 for a zero init (no ratio is then
            /// meaningful).
            let initColumnL2Norms: [Double]
            /// Whether `initColumnL2Norms` is the model's own start (else a
            /// draw of the same distribution).
            let initExact: Bool
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

    /// `run(snapshot:modelLabel:)` on a GCD queue: it scans and sorts every
    /// value-head tensor, synchronous work that must hold neither the main
    /// actor nor a cooperative thread.
    static func runOffPool(snapshot: AnalyzedNetworkSnapshot, modelLabel: String) async throws -> Result {
        try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .utility).async {
                continuation.resume(with: Swift.Result(catching: { try run(snapshot: snapshot, modelLabel: modelLabel) }))
            }
        }
    }

    /// Run the analyzer on `snapshot`: the `value_*` variables, by name, with
    /// their init figures from the snapshot's init reference. `modelLabel` is
    /// opaque metadata the caller chooses (which network was analyzed) and is
    /// round-tripped into the result header.
    static func run(snapshot: AnalyzedNetworkSnapshot, modelLabel: String) throws -> Result {
        let arch = snapshot.architecture

        var stats: [Result.WeightStats] = []
        var fc2BiasDetail: Result.FC2BiasDetail?
        var fc2WeightsDetail: Result.FC2WeightsDetail?

        for (i, name) in snapshot.names.enumerated() where name.hasPrefix("value_") {
            let values = snapshot.weights[i]
            let start = try snapshot.start(ofVariableAt: i)
            stats.append(makeStats(name: name, values: values, start: start))
            guard case .trainable(let initial, let exact) = start else { continue }

            // The W/D/L details describe the three-class head's fc2 (its
            // variables are named for the head style; see
            // `ChessNetwork.valueHead`).
            if name == "value_wdl_fc2_bias" {
                fc2BiasDetail = makeFC2BiasDetail(values: values, initial: initial, initExact: exact)
            } else if name == "value_wdl_fc2_weights" {
                fc2WeightsDetail = makeFC2WeightsDetail(values: values, initial: initial, initExact: exact, arch: arch)
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
        start: AnalysisInitReference.Start
    ) -> Result.WeightStats {
        let initial: [Float]?
        let initExact: Bool
        switch start {
        case .trainable(let startValues, let exact):
            initial = startValues
            initExact = exact
        case .runningStatistic:
            initial = nil
            initExact = false
        }
        let isRunningStatistic = initial == nil
        let initL2: Double? = initial.map { startValues in
            sqrt(startValues.reduce(0.0) { $0 + Double($1) * Double($1) })
        }
        let n = values.count
        guard n > 0 else {
            return Result.WeightStats(
                name: name,
                elementCount: 0,
                l1Norm: 0, l2Norm: 0, meanAbs: 0,
                min: 0, max: 0, mean: 0, stdev: 0,
                percentiles: Array(repeating: 0, count: percentileLabels.count),
                isRunningStatistic: isRunningStatistic,
                l2NormRatioToInit: nil,
                initL2Norm: initL2,
                initExact: initExact
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

        // A tensor that started at zero, or a BN running statistic, has no
        // ratio to its start.
        let ratio: Double? = initL2.flatMap { $0 > 0 ? l2 / $0 : nil }

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
            isRunningStatistic: isRunningStatistic,
            l2NormRatioToInit: ratio,
            initL2Norm: initL2,
            initExact: initExact
        )
    }

    private static func makeFC2BiasDetail(values: [Float], initial initialValues: [Float], initExact: Bool) -> Result.FC2BiasDetail? {
        guard values.count == 3, initialValues.count == 3 else { return nil }
        let current = values.map { Double($0) }
        let initial: [Double]? = initExact ? initialValues.map { Double($0) } : nil
        return Result.FC2BiasDetail(
            current: current,
            biasOnlySoftmax: softmax(current),
            initExact: initExact,
            initial: initial,
            initialBiasOnlySoftmax: initial.map(softmax),
            delta: initial.map { start in zip(current, start).map { $0 - $1 } }
        )
    }

    private static func makeFC2WeightsDetail(
        values: [Float], initial: [Float], initExact: Bool, arch: NetworkArchitecture
    ) -> Result.FC2WeightsDetail? {
        // The `value_wdl_fc2_weights` tensor has shape [hidden, 3] (in × out)
        // and is stored row-major (every 3 consecutive floats are the
        // weights from one input neuron to W/D/L). To get the L2 norm
        // of the `[c]` output column, sum squares of `values[i*outDim + c]`
        // for i in 0..<hidden. Dims come from `ChessNetwork` so this tracks
        // the value-head shape — they're structural facts, not tunables.
        let outDim = arch.valueHeadClasses
        let inDim = arch.valueHeadHiddenUnits
        guard values.count == inDim * outDim, initial.count == values.count else { return nil }
        return Result.FC2WeightsDetail(
            columnL2Norms: columnL2Norms(values, inDim: inDim, outDim: outDim),
            initColumnL2Norms: columnL2Norms(initial, inDim: inDim, outDim: outDim),
            initExact: initExact
        )
    }

    /// Each output column's L2 norm of an `[inDim, outDim]` row-major tensor.
    private static func columnL2Norms(_ values: [Float], inDim: Int, outDim: Int) -> [Double] {
        var columnSumSq = [Double](repeating: 0, count: outDim)
        for i in 0..<inDim {
            for c in 0..<outDim {
                let v = Double(values[i * outDim + c])
                columnSumSq[c] += v * v
            }
        }
        return columnSumSq.map { sqrt($0) }
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
        out += "  (~ = init figure includes draws of the same distribution, not the model's own initial values; "
            + "-- = none: started at zero, or a BN running statistic, whose start depends on how the model was built)\n"
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
            let initL2Str = s.initL2Norm.map { String(format: "%6.3f", $0) + (s.initExact ? "" : "~") } ?? "  --  "
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
            out += "value_fc2_bias (the head's prediction is pW/pD/pL on [STATS], not these softmaxes):\n"
            out += String(format: "  current values             : %@\n",
                          bias.current.map { String(format: "%+7.4f", $0) }.joined(separator: ", "))
            out += String(format: "  softmax of bias alone (WDL): %@\n",
                          bias.biasOnlySoftmax.map { String(format: "%.4f", $0) }.joined(separator: ", "))
            if let initial = bias.initial, let initialSoftmax = bias.initialBiasOnlySoftmax, let delta = bias.delta {
                out += String(format: "  initial values             : %@\n",
                              initial.map { String(format: "%+7.4f", $0) }.joined(separator: ", "))
                out += String(format: "  initial bias softmax (WDL) : %@\n",
                              initialSoftmax.map { String(format: "%.4f", $0) }.joined(separator: ", "))
                out += String(format: "  delta from initial         : %@\n",
                              delta.map { String(format: "%+7.4f", $0) }.joined(separator: ", "))
            } else {
                out += "  initial values             : not known exactly (no delta)\n"
            }
            out += "\n"
        }

        // FC2 weights per-output-column.
        if let fc2w = fc2Weights {
            out += "value_fc2_weights — per-output-column L2 norms (each against its own L2 at init):\n"
            let names = ["win", "draw", "loss"]
            for (i, n) in names.enumerated() where i < fc2w.columnL2Norms.count && i < fc2w.initColumnL2Norms.count {
                let v = fc2w.columnL2Norms[i]
                let initial = fc2w.initColumnL2Norms[i]
                let ratioStr = initial > 0 ? String(format: "%6.3f", v / initial) : "  --  "
                out += String(format: "  %@: L2=%6.3f  init=%@  ratio=%@\n", n.padded(5), v,
                              String(format: "%6.3f", initial) + (fc2w.initExact ? "" : "~"), ratioStr)
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
