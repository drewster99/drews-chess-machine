import SwiftUI
import SwiftUIFastCharts

/// Overlays the trainer's **total training loss** (`pLoss + vLoss`)
/// against the **held-out eval loss** — the wide (~4,435-puzzle)
/// Lichess set's bookmove cross-entropy (`meanNegLogProb`) — on a
/// single plot with two independent, auto-scaling Y axes:
/// training loss on the leading (left) axis, eval loss on the trailing
/// (right) axis. The shared X axis is the trainer step
/// (`ChessTrainer.completedTrainSteps`), the same index the Lichess
/// probe records against, so the two trajectories line up by training
/// progress rather than wall-clock time (pauses/arenas don't distort
/// the comparison).
///
/// Interpretation note: the eval line is *pure* policy cross-entropy on
/// held-out positions, while the training line is *outcome-weighted*
/// policy CE + value CE (`pLoss` can go negative). They measure
/// related-but-different things — hence the independent axes — so read
/// the *trends*: both falling = healthy; training falling while eval
/// flattens or rises = overfitting / plateau.
///
/// Rendering mirrors the Lichess overall-trend charts: a faint raw
/// "noise cloud" with a bold EMA overlay on top (toggleable), reusing
/// `FastChartMath.ema`.
struct CombinedLossChartView: View {
    @Bindable var coordinator: ChartCoordinator
    @Bindable var evalHistory: LichessProbeHistory

    @State private var emaEnabled = true
    @State private var emaSpan = 25
    @State private var chartGroup = FastChartGroup()

    /// Color the training (leading-axis) elements share.
    private let trainColor = Color.blue
    /// Color the eval (trailing-axis) elements share.
    private let evalColor = Color.orange

    var body: some View {
        // Establish the @Observable dependency on the training ring's
        // append counter so the chart re-reads the ring on each new
        // sample. `evalHistory.overallSeries` is read below and tracks
        // itself.
        let _ = coordinator.trainingChartNextId
        let trainPts = trainingPoints()
        let evalPts = evalPoints()

        VStack(alignment: .leading, spacing: 6) {
            controls
            if trainPts.isEmpty && evalPts.isEmpty {
                placeholder
            } else {
                chart(trainPts: trainPts, evalPts: evalPts)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
            caption
        }
        .padding(10)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
    }

    // MARK: - Subviews

    @ViewBuilder
    private var controls: some View {
        HStack(spacing: 12) {
            Toggle("EMA overlay", isOn: $emaEnabled)
                .toggleStyle(.checkbox)
            if emaEnabled {
                Stepper("span \(emaSpan)", value: $emaSpan, in: 3...200)
                    .fixedSize()
            }
            Spacer(minLength: 0)
        }
        .font(.system(.caption, design: .monospaced))
    }

    @ViewBuilder
    private var placeholder: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Waiting for data")
                .font(.system(.caption, design: .monospaced).weight(.semibold))
            Text("The training-loss line appears once Play-and-Train is running; "
                + "the eval line appears after the first wide Lichess probe tick.")
                .font(.system(.caption2, design: .monospaced))
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .center)
    }

    @ViewBuilder
    private var caption: some View {
        Text("Left axis: training total loss (pLoss + vLoss, outcome-weighted). "
            + "Right axis: eval NLL (wide Lichess set bookmove cross-entropy). "
            + "Shared X: trainer step.")
            .font(.system(size: 9, design: .monospaced))
            .foregroundStyle(.secondary)
            .fixedSize(horizontal: false, vertical: true)
    }

    private func chart(trainPts: [CGPoint], evalPts: [CGPoint]) -> some View {
        // Bound the plotted point count so an always-open window over a
        // multi-day run doesn't rebuild and stroke ~100k points on the
        // main actor every ~1s. `strideCap` preserves endpoints, so the
        // trend, the live right edge, and the X range are unchanged —
        // this is purely a render-cost guard, mirroring the decimated
        // frame the main chart grid uses. The header below still reads
        // the *full* arrays' last value for an exact live readout.
        let trainPlot = Self.strideCap(trainPts, max: Self.maxPlotPoints)
        let evalPlot = Self.strideCap(evalPts, max: Self.maxPlotPoints)

        let xs = trainPlot.map { Double($0.x) } + evalPlot.map { Double($0.x) }

        var trainYs = trainPlot.map { Double($0.y) }
        var evalYs = evalPlot.map { Double($0.y) }

        var series: [FastChartSeries] = []
        if !trainPlot.isEmpty {
            series.append(FastChartSeries(
                id: "train total loss (left)",
                color: emaEnabled ? trainColor.opacity(0.28) : trainColor,
                lineWidth: emaEnabled ? 1.0 : 1.6,
                data: .points(trainPlot),
                yAxis: .primary
            ))
            if emaEnabled {
                let ey = FastChartMath.ema(trainPlot.map { Double($0.y) }, span: emaSpan)
                trainYs.append(contentsOf: ey)
                let pts = trainPlot.indices.map { CGPoint(x: trainPlot[$0].x, y: CGFloat(ey[$0])) }
                series.append(FastChartSeries(
                    id: "train EMA", color: .indigo, lineWidth: 2.5,
                    data: .points(pts), yAxis: .primary
                ))
            }
        }
        if !evalPlot.isEmpty {
            series.append(FastChartSeries(
                id: "eval NLL (right)",
                color: emaEnabled ? evalColor.opacity(0.30) : evalColor,
                lineWidth: emaEnabled ? 1.0 : 1.6,
                data: .points(evalPlot),
                yAxis: .secondary
            ))
            if emaEnabled {
                let ey = FastChartMath.ema(evalPlot.map { Double($0.y) }, span: emaSpan)
                evalYs.append(contentsOf: ey)
                let pts = evalPlot.indices.map { CGPoint(x: evalPlot[$0].x, y: CGFloat(ey[$0])) }
                series.append(FastChartSeries(
                    id: "eval EMA", color: .red, lineWidth: 2.5,
                    data: .points(pts), yAxis: .secondary
                ))
            }
        }

        let legendItems = [
            FastChartLegendItem(label: "train total loss (left)", color: trainColor),
            FastChartLegendItem(label: "eval NLL (right)", color: evalColor)
        ]

        return FastLineChart(
            title: "Training vs Eval Loss (X = trainer step)",
            group: chartGroup,
            xDomain: Self.paddedXDomain(xs),
            yDomain: Self.paddedYDomain(trainYs),
            secondaryYDomain: evalPlot.isEmpty ? nil : Self.paddedYDomain(evalYs),
            secondaryYLabelFormatter: { String(format: "%.2f", $0) },
            secondaryYLabelColor: evalColor,
            series: series,
            yLabelCount: 5,
            showXAxisLabels: true,
            yLabelFormatter: { String(format: "%.2f", $0) },
            xLabelFormatter: FastChartFormatters.compact,
            legend: .custom(legendItems),
            headerValue: { _ in
                var parts: [String] = []
                if let t = trainPts.last { parts.append(String(format: "train %.3f", Double(t.y))) }
                if let e = evalPts.last { parts.append(String(format: "eval %.3f", Double(e.y))) }
                return AttributedString(parts.joined(separator: "   "))
            }
        )
    }

    // MARK: - Data

    private func trainingPoints() -> [CGPoint] {
        let ring = coordinator.trainingRing
        var pts: [CGPoint] = []
        pts.reserveCapacity(ring.count)
        // Clamp x to be non-decreasing: live steps already are, but a
        // back-filled prefix in the degraded (no-probe-anchor) case
        // could otherwise dip a hair below its live neighbor and break
        // `FastChartSeries`'s non-decreasing-x invariant.
        var lastX = -Double.greatestFiniteMagnitude
        for i in 0..<ring.count {
            let s = ring[i]
            guard let step = s.trainingStep,
                  let loss = s.rollingTotalLoss, loss.isFinite
            else { continue }
            let x = max(lastX, Double(step))
            lastX = x
            pts.append(CGPoint(x: x, y: loss))
        }
        return pts
    }

    private func evalPoints() -> [CGPoint] {
        var pts: [CGPoint] = []
        pts.reserveCapacity(evalHistory.overallSeries.count)
        var lastX = -Double.greatestFiniteMagnitude
        for s in evalHistory.overallSeries {
            guard let step = s.trainingStep, s.meanNegLogProb.isFinite else { continue }
            let x = max(lastX, Double(step))
            lastX = x
            pts.append(CGPoint(x: x, y: s.meanNegLogProb))
        }
        return pts
    }

    // MARK: - Render-cost guard

    /// Upper bound on points handed to a single series. Comfortably above
    /// the horizontal pixel count of any window, so the strided curve is
    /// visually indistinguishable from the full one, while keeping the
    /// per-tick rebuild + stroke cost flat regardless of run length.
    private static let maxPlotPoints = 6000

    /// Uniformly down-sample `pts` to at most `cap` points, always keeping
    /// the first and last so the X range and the live right edge are
    /// preserved. Returns the input unchanged when already within `cap`.
    /// Output is a subsequence, so the non-decreasing-X invariant
    /// `FastChartSeries` requires is preserved.
    static func strideCap(_ pts: [CGPoint], max cap: Int) -> [CGPoint] {
        guard cap > 2, pts.count > cap else { return pts }
        let step = Double(pts.count - 1) / Double(cap - 1)
        var out: [CGPoint] = []
        out.reserveCapacity(cap)
        var lastIdx = -1
        for i in 0..<cap {
            let idx = min(pts.count - 1, Int((Double(i) * step).rounded()))
            if idx != lastIdx {
                out.append(pts[idx])
                lastIdx = idx
            }
        }
        if let last = pts.last, out.last != last {
            out.append(last)
        }
        return out
    }

    // MARK: - Domain helpers

    /// X domain spanning both series. Points are non-decreasing in x, so
    /// the extremes are the overall min/max. Degenerate spans pad to a
    /// unit so the `ClosedRange` is valid.
    static func paddedXDomain(_ xs: [Double]) -> ClosedRange<Double> {
        let finite = xs.filter(\.isFinite)
        guard let lo = finite.min(), let hi = finite.max() else { return 0...1 }
        if hi <= lo { return lo...(lo + 1) }
        return lo...hi
    }

    /// Y domain fit to the data with 8% padding so the trajectory fills
    /// the panel instead of being compressed.
    static func paddedYDomain(_ values: [Double]) -> ClosedRange<Double> {
        let finite = values.filter(\.isFinite)
        guard let lo = finite.min(), let hi = finite.max() else { return 0...1 }
        if hi <= lo { return (lo - 0.5)...(hi + 0.5) }
        let pad = (hi - lo) * 0.08
        return (lo - pad)...(hi + pad)
    }
}
