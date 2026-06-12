import SwiftUI
import SwiftUIFastCharts

/// Two stacked compact line charts plotting the OVERALL 200-puzzle
/// summary across every recorded tick of the Lichess probe watcher.
/// Rendered on the path-based `FastLineChart` (not SwiftUI Charts) so
/// the whole-run trajectory stays cheap to draw even at many thousands
/// of samples; both share a `FastChartGroup` so a hover on one shows
/// the crosshair on both.
///
/// Top: mean per-probe `−log p_bookmove` in nats — the cross-entropy
/// of the bookmove evaluated against all 200 puzzles. Lower is
/// better. This is the same loss the trainer minimizes on its replay
/// buffer, computed on these held-out positions, so it's the closest
/// thing the project has to a "test loss" trajectory.
///
/// Bottom: MLE puzzle-Elo fitted by Bradley-Terry on the
/// per-puzzle (rating, correct) outcomes. Higher is better. Only
/// finite-valued samples are plotted; ticks where every puzzle was
/// wrong (−∞) or every puzzle was right (+∞) are dropped from the
/// series so the y-axis doesn't blow up. The set's rating range is
/// 800–1800, so values pin near 800 until the network is solving
/// at least one puzzle.
///
/// X axis is the trainer step (`ChessTrainer.completedTrainSteps`) at
/// which each tick was recorded — the same index the rest of the app's
/// telemetry uses, so the trajectory lines up with steps/positions
/// elsewhere. If any sample lacks a step (e.g. a manual "Probe now"
/// taken before training started, or the steps aren't monotonic) the
/// chart falls back to plotting against tick index and labels the axis
/// accordingly.
///
/// When a comparison snapshot is loaded, a dashed horizontal
/// reference line at the cmp's value is overlaid on each chart so
/// the live trajectory's distance from cmp reads at a glance.
struct LichessProbeOverallTrendChart: View {
    @Bindable var history: LichessProbeHistory
    /// Optional comparison snapshot's overall values, drawn as
    /// dashed horizontal reference lines. nil = no comparison
    /// loaded (don't draw the reference).
    let cmpNll: Double?
    let cmpElo: Double?

    /// Shared hover state so the crosshair on the NLL chart and the
    /// Elo chart move together. Owned here; both `FastLineChart`s read it.
    @State private var chartGroup = FastChartGroup()

    /// EMA overlay state. The probe ticks every ~17s and the raw per-tick
    /// 200-puzzle (soon 1000-puzzle) series is sampling-noisy, so an
    /// exponential moving average makes the *trend* readable — the whole point
    /// when eyeballing the effect of a training-dynamics change (LR/momentum
    /// cycling). On by default; `emaSpan` is the window in ticks.
    @State private var emaEnabled = true
    @State private var emaSpan = 25

    /// X-axis positions (one per sample, parallel to the input array)
    /// plus the axis label. Uses the per-tick trainer step when every
    /// sample carries a non-nil, non-decreasing step; otherwise falls
    /// back to tick index (0-based). Pure + static so it can be unit
    /// tested without a SwiftUI host.
    static func xPositions(
        for samples: [LichessProbeHistory.OverallTickSample]
    ) -> (xs: [Double], label: String) {
        let steps = samples.map(\.trainingStep)
        let allPresent = steps.allSatisfy { $0 != nil }
        var nonDecreasing = true
        if allPresent {
            var prev = Int.min
            for case let s? in steps {
                if s < prev { nonDecreasing = false; break }
                prev = s
            }
        }
        if allPresent && nonDecreasing {
            return (steps.map { Double($0 ?? 0) }, "trainer step")
        }
        return ((0..<samples.count).map(Double.init), "tick #")
    }

    var body: some View {
        let samples = history.overallSeries
        if samples.isEmpty {
            placeholder
        } else {
            let xs = Self.xPositions(for: samples).xs
            VStack(alignment: .leading, spacing: 4) {
                emaControls
                nllChart(samples: samples, xs: xs)
                eloChart(samples: samples, xs: xs)
            }
        }
    }

    @ViewBuilder
    private var placeholder: some View {
        Text("OVERALL trend — waiting for first tick "
            + "(periodic Lichess probe; click Probe now for an immediate sample)")
            .font(.system(.caption, design: .monospaced))
            .foregroundStyle(.secondary)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.vertical, 8)
    }

    /// Toggle + span control for the EMA overlay (drawn above both charts).
    @ViewBuilder
    private var emaControls: some View {
        HStack(spacing: 12) {
            Toggle("EMA overlay", isOn: $emaEnabled)
                .toggleStyle(.checkbox)
            if emaEnabled {
                Stepper("span \(emaSpan)", value: $emaSpan, in: 3...200)
                    .fixedSize()
            }
        }
        .font(.system(.caption, design: .monospaced))
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    /// Exponential moving average over `ys` with the given span. Thin
    /// delegate to `FastChartMath.ema` (the single implementation shared
    /// with the combined Training-vs-Eval-Loss window); kept as a static
    /// here so existing call sites and tests referencing
    /// `LichessProbeOverallTrendChart.ema` stay valid.
    static func ema(_ ys: [Double], span: Int) -> [Double] {
        FastChartMath.ema(ys, span: span)
    }

    /// Uniform-over-30-legals reference value (≈ ln 30) for the NLL chart.
    private static let uniformNLL = log(30.0)

    /// NLL chart. Single view type, so a plain function (not a
    /// `@ViewBuilder`) — lets the reference-line / domain prep use
    /// ordinary statements before the one returned `FastLineChart`.
    private func nllChart(
        samples: [LichessProbeHistory.OverallTickSample],
        xs: [Double]
    ) -> some View {
        let points = zip(xs, samples).map { CGPoint(x: $0.0, y: $0.1.meanNegLogProb) }
        var yValues = samples.map(\.meanNegLogProb) + [Self.uniformNLL]
        var refs: [FastChartReferenceLine] = [
            FastChartReferenceLine(
                id: "uniform", y: Self.uniformNLL, label: "uniform ~3.4",
                color: Color.gray.opacity(0.45), lineWidth: 1, dashed: true
            )
        ]
        if let cmp = cmpNll {
            yValues.append(cmp)
            refs.append(FastChartReferenceLine(
                id: "cmp", y: cmp, label: "cmp",
                color: Color.orange.opacity(0.7), lineWidth: 1, dashed: true
            ))
        }
        // When the EMA overlay is on, fade the raw series into a faint "noise
        // cloud" (low opacity, thin) and draw the EMA bold on top so the trend
        // reads clearly — otherwise the dense full-opacity raw line drowns the
        // overlay. With the overlay off, the raw series is the only line, so it
        // stays at normal weight.
        var series = [FastChartSeries(
            id: "nll",
            color: emaEnabled ? Color.blue.opacity(0.28) : .blue,
            lineWidth: emaEnabled ? 1.0 : 1.5,
            data: .points(points)
        )]
        if emaEnabled {
            let ey = Self.ema(points.map { Double($0.y) }, span: emaSpan)
            yValues.append(contentsOf: ey)
            let emaPts = points.indices.map { CGPoint(x: points[$0].x, y: CGFloat(ey[$0])) }
            series.append(FastChartSeries(id: "nll-ema", color: .indigo, lineWidth: 2.5, data: .points(emaPts)))
        }
        return FastLineChart(
            title: "OVERALL NLL (nats, lower = better)",
            group: chartGroup,
            xDomain: Self.paddedXDomain(xs),
            yDomain: Self.paddedYDomain(yValues),
            series: series,
            referenceLines: refs,
            showXAxisLabels: true,
            yLabelFormatter: { String(format: "%.2f", $0) },
            xLabelFormatter: FastChartFormatters.compact,
            headerValue: { ctx in
                overallHeaderValue(
                    ctx: ctx, samples: samples, xs: xs,
                    value: { $0.meanNegLogProb }, cmp: cmpNll,
                    fmt: { String(format: "%.3f", $0) }
                )
            }
        )
        .frame(height: 110)
    }

    @ViewBuilder
    private func eloChart(
        samples: [LichessProbeHistory.OverallTickSample],
        xs: [Double]
    ) -> some View {
        // Only finite-valued Elo samples are plotted (ticks where every
        // puzzle was wrong / right map to ±∞). If none are finite, show a
        // clarifying caption instead of an empty chart.
        let points = zip(xs, samples).compactMap { pair -> CGPoint? in
            pair.1.puzzleElo.isFinite ? CGPoint(x: pair.0, y: pair.1.puzzleElo) : nil
        }
        if points.isEmpty {
            VStack(alignment: .leading, spacing: 1) {
                Text("OVERALL puzzle-Elo (higher = better, 800–1800 set range)")
                    .font(.system(.caption, design: .monospaced).weight(.semibold))
                Text("All samples at floor (every puzzle wrong) — Elo MLE is unbounded below.")
                    .font(.system(.caption2, design: .monospaced))
                    .foregroundStyle(.secondary)
                    .frame(height: 60, alignment: .center)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        } else {
            eloLineChart(points: points, samples: samples, xs: xs)
        }
    }

    private func eloLineChart(
        points: [CGPoint],
        samples: [LichessProbeHistory.OverallTickSample],
        xs: [Double]
    ) -> some View {
        var yValues = samples.map(\.puzzleElo).filter(\.isFinite)
        var refs: [FastChartReferenceLine] = []
        if let cmp = cmpElo, cmp.isFinite {
            yValues.append(cmp)
            refs.append(FastChartReferenceLine(
                id: "cmp", y: cmp, label: "cmp",
                color: Color.orange.opacity(0.7), lineWidth: 1, dashed: true
            ))
        }
        // Same treatment as the NLL chart: fade the raw series when the EMA
        // overlay is on so the bold trend line reads over the noise cloud.
        var series = [FastChartSeries(
            id: "elo",
            color: emaEnabled ? Color.green.opacity(0.28) : .green,
            lineWidth: emaEnabled ? 1.0 : 1.5,
            data: .points(points)
        )]
        if emaEnabled {
            let ey = Self.ema(points.map { Double($0.y) }, span: emaSpan)
            yValues.append(contentsOf: ey)
            let emaPts = points.indices.map { CGPoint(x: points[$0].x, y: CGFloat(ey[$0])) }
            series.append(FastChartSeries(id: "elo-ema", color: .teal, lineWidth: 2.5, data: .points(emaPts)))
        }
        return FastLineChart(
            title: "OVERALL puzzle-Elo (higher = better, 800–1800 set range)",
            group: chartGroup,
            xDomain: Self.paddedXDomain(xs),
            yDomain: Self.paddedYDomain(yValues),
            series: series,
            referenceLines: refs,
            showXAxisLabels: true,
            yLabelFormatter: { String(format: "%.0f", $0) },
            xLabelFormatter: FastChartFormatters.compact,
            headerValue: { ctx in
                overallHeaderValue(
                    ctx: ctx, samples: samples, xs: xs,
                    value: { $0.puzzleElo }, cmp: cmpElo,
                    fmt: Self.formatElo
                )
            }
        )
        .frame(height: 110)
    }

    // MARK: - Header + domain helpers

    /// Right-aligned header value for either OVERALL chart: the hovered
    /// sample's value when a crosshair is active, else the latest ("live")
    /// value, with the comparison value appended when a cmp snapshot is
    /// loaded. `value` selects NLL or Elo off a sample; `fmt` renders it.
    private func overallHeaderValue(
        ctx: FastChartHoverContext,
        samples: [LichessProbeHistory.OverallTickSample],
        xs: [Double],
        value: (LichessProbeHistory.OverallTickSample) -> Double,
        cmp: Double?,
        fmt: (Double) -> String
    ) -> AttributedString {
        // The hovered/live sample's trainer step leads the value so the
        // reader always knows *where* in the run the number came from.
        // Ticks recorded before a trainer existed have no step — those
        // fall back to the bare value.
        let main: String
        if let hx = ctx.hoveredX, let i = Self.nearestIndex(hoveredX: hx, xs: xs) {
            let sample = samples[i]
            if let step = sample.trainingStep {
                main = "step \(step.formatted())  " + fmt(value(sample))
            } else {
                main = fmt(value(sample))
            }
        } else if let last = samples.last {
            if let step = last.trainingStep {
                main = "live @ \(step.formatted())  " + fmt(value(last))
            } else {
                main = "live " + fmt(value(last))
            }
        } else {
            main = "--"
        }
        if let cmp { return AttributedString(main + "  cmp " + fmt(cmp)) }
        return AttributedString(main)
    }

    /// Index of the sample whose x-position is closest to `hoveredX`.
    /// `xs` is non-decreasing (see `xPositions`); linear scan is fine at
    /// these series sizes.
    static func nearestIndex(hoveredX: Double, xs: [Double]) -> Int? {
        guard !xs.isEmpty else { return nil }
        var best = 0
        var bestDist = abs(xs[0] - hoveredX)
        for i in 1..<xs.count {
            let d = abs(xs[i] - hoveredX)
            if d < bestDist { bestDist = d; best = i }
        }
        return best
    }

    /// X domain spanning the whole series. `xs` is non-decreasing, so
    /// `first`/`last` are the min/max. Degenerate (single sample) pads to
    /// a unit width so the `ClosedRange` is valid.
    static func paddedXDomain(_ xs: [Double]) -> ClosedRange<Double> {
        guard let lo = xs.first, let hi = xs.last else { return 0...1 }
        if hi <= lo { return lo...(lo + 1) }
        return lo...hi
    }

    /// Y domain fit to the data (plus any reference values passed in)
    /// with 8% padding, so the trajectory fills the panel instead of
    /// being compressed against a 0-anchored auto-axis.
    static func paddedYDomain(_ values: [Double]) -> ClosedRange<Double> {
        let finite = values.filter(\.isFinite)
        guard let lo = finite.min(), let hi = finite.max() else { return 0...1 }
        if hi <= lo { return (lo - 0.5)...(hi + 0.5) }
        let pad = (hi - lo) * 0.08
        return (lo - pad)...(hi + pad)
    }

    /// Mirror of `LichessProbeDetailView.formatPuzzleElo` without
    /// the "pElo" prefix — used in the chart header's value chip.
    private static func formatElo(_ elo: Double) -> String {
        if elo.isNaN { return "—" }
        if elo == -.infinity { return "<floor" }
        if elo == .infinity { return ">ceil" }
        return String(format: "%.0f", elo)
    }
}
