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

    /// Manually-chosen visible x-window (trainer-step range). `nil`
    /// means "auto-fit the full run" — the chart follows the live right
    /// edge as new steps arrive. Once the user pans or zooms, this is
    /// set and the view stops following live until "Reset zoom" clears
    /// it back to `nil`. Always stored already clamped to the data's
    /// full extent (see `clampVisible`).
    @State private var visibleXRange: ClosedRange<Double>?

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
            if visibleXRange != nil {
                Button("Reset zoom") { visibleXRange = nil }
                    .buttonStyle(.link)
            } else {
                Text("drag to pan · pinch to zoom")
                    .foregroundStyle(.secondary)
            }
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

        let trainRawYs = trainPlot.map { Double($0.y) }
        let evalRawYs = evalPlot.map { Double($0.y) }
        let trainEMAYs = emaEnabled ? FastChartMath.ema(trainRawYs, span: emaSpan) : []
        let evalEMAYs = emaEnabled ? FastChartMath.ema(evalRawYs, span: emaSpan) : []

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
                let pts = trainPlot.indices.map { CGPoint(x: trainPlot[$0].x, y: CGFloat(trainEMAYs[$0])) }
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
                let pts = evalPlot.indices.map { CGPoint(x: evalPlot[$0].x, y: CGFloat(evalEMAYs[$0])) }
                series.append(FastChartSeries(
                    id: "eval EMA", color: .red, lineWidth: 2.5,
                    data: .points(pts), yAxis: .secondary
                ))
            }
        }

        // Auto-scale each Y axis to the *smoothed* (EMA) trajectory plus
        // padding when the overlay is on, rather than the raw min/max. A
        // single warm-up/transient raw sample (e.g. the rolling-loss
        // accumulator still filling at the oldest end of the ring) would
        // otherwise stretch the axis and waste most of the panel. Raw
        // points outside the EMA range still draw — the Canvas clips them
        // to the plot's top/bottom edge. With the EMA off there is no
        // smoothed curve to scale to, so fall back to the raw extents.
        let trainDomainYs = emaEnabled ? trainEMAYs : trainRawYs
        let evalDomainYs = emaEnabled ? evalEMAYs : evalRawYs

        // The full run's x-extent, then the actually-visible window. When
        // the user has panned/zoomed (`visibleXRange != nil`) we honor
        // that window; otherwise we fit the whole run and track the live
        // right edge. Either way the Y axes rescale to *only* the points
        // inside the visible window so zooming into a flat-looking tail
        // expands it to fill the panel.
        let fullXDomain = Self.paddedXDomain(xs)
        let activeXDomain = Self.clampVisible(visibleXRange, to: fullXDomain)
        let trainYDomainYs = Self.windowedYs(plot: trainPlot, ys: trainDomainYs, xDomain: activeXDomain)
        let evalYDomainYs = Self.windowedYs(plot: evalPlot, ys: evalDomainYs, xDomain: activeXDomain)

        let legendItems = [
            FastChartLegendItem(label: "train total loss (left)", color: trainColor),
            FastChartLegendItem(label: "eval NLL (right)", color: evalColor)
        ]

        return FastLineChart(
            title: "Training vs Eval Loss (X = trainer step)",
            group: chartGroup,
            xDomain: activeXDomain,
            yDomain: Self.paddedYDomain(trainYDomainYs),
            secondaryYDomain: evalPlot.isEmpty ? nil : Self.paddedYDomain(evalYDomainYs),
            secondaryYLabelFormatter: { String(format: "%.2f", $0) },
            secondaryYLabelColor: evalColor,
            series: series,
            yLabelCount: 5,
            showXAxisLabels: true,
            yLabelFormatter: { String(format: "%.2f", $0) },
            xLabelFormatter: FastChartFormatters.compact,
            legend: .custom(legendItems),
            headerValue: { ctx in
                // While a crosshair is active, read the sample under it on
                // each series; otherwise show the latest ("live") values.
                var parts: [String] = []
                if let hx = ctx.hoveredX {
                    if let i = Self.nearestIndex(hoveredX: hx, points: trainPts) {
                        parts.append(String(format: "train %.3f", Double(trainPts[i].y)))
                    }
                    if let j = Self.nearestIndex(hoveredX: hx, points: evalPts) {
                        parts.append(String(format: "eval %.3f", Double(evalPts[j].y)))
                    }
                } else {
                    if let t = trainPts.last { parts.append(String(format: "train %.3f", Double(t.y))) }
                    if let e = evalPts.last { parts.append(String(format: "eval %.3f", Double(e.y))) }
                }
                return AttributedString(parts.joined(separator: "   "))
            },
            onInteractiveXDomainChange: { requested in
                // A window that (after clamping) spans essentially the whole
                // run means "no zoom" — drop back to auto-fit so the chart
                // resumes following the live right edge instead of freezing
                // at the current full extent. This makes panning a
                // fully-zoomed-out chart a no-op and makes pinch-back-to-full
                // restore live tracking without needing the Reset button.
                let clamped = Self.clampVisible(requested, to: fullXDomain)
                let fullSpan = fullXDomain.upperBound - fullXDomain.lowerBound
                let clampedSpan = clamped.upperBound - clamped.lowerBound
                visibleXRange = (fullSpan > 0 && clampedSpan >= fullSpan * 0.999) ? nil : clamped
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

    /// Y domain fit to the data with a small fractional padding so the
    /// trajectory fills the panel instead of being pinned to the edges.
    static func paddedYDomain(_ values: [Double]) -> ClosedRange<Double> {
        let finite = values.filter(\.isFinite)
        guard let lo = finite.min(), let hi = finite.max() else { return 0...1 }
        if hi <= lo { return (lo - 0.5)...(hi + 0.5) }
        let pad = (hi - lo) * 0.10
        return (lo - pad)...(hi + pad)
    }

    // MARK: - Pan / zoom helpers

    /// Clamp a requested visible x-window to the run's full extent so a
    /// pan/zoom can never scroll past the data or zoom out beyond the
    /// whole run. A `nil` request means "auto-fit" and returns the full
    /// extent unchanged. The window's span is preserved where possible
    /// and slid back inside the bounds when it would overhang an edge;
    /// a span wider than the full extent collapses to the full extent.
    /// A tiny lower bound on the span keeps the most extreme zoom-in from
    /// producing a degenerate range.
    static func clampVisible(
        _ requested: ClosedRange<Double>?,
        to full: ClosedRange<Double>
    ) -> ClosedRange<Double> {
        guard let requested else { return full }
        let fullSpan = full.upperBound - full.lowerBound
        guard fullSpan > 0 else { return full }
        var span = min(requested.upperBound - requested.lowerBound, fullSpan)
        span = max(span, fullSpan * 0.002)
        var lo = requested.lowerBound
        if lo < full.lowerBound { lo = full.lowerBound }
        if lo + span > full.upperBound { lo = full.upperBound - span }
        return lo...(lo + span)
    }

    /// The y-values of `plot` whose x falls inside `xDomain`, used to
    /// rescale the Y axis to what's currently visible. `plot` and `ys`
    /// are parallel (same index = same point). Returns the full `ys`
    /// when the window happens to bracket no plotted point, so the axis
    /// never collapses to an empty range.
    static func windowedYs(
        plot: [CGPoint],
        ys: [Double],
        xDomain: ClosedRange<Double>
    ) -> [Double] {
        guard plot.count == ys.count, !plot.isEmpty else { return ys }
        var out: [Double] = []
        out.reserveCapacity(plot.count)
        for i in plot.indices {
            let x = Double(plot[i].x)
            if x >= xDomain.lowerBound && x <= xDomain.upperBound { out.append(ys[i]) }
        }
        return out.isEmpty ? ys : out
    }

    /// Index of the point whose x is closest to `hoveredX`. Points are
    /// non-decreasing in x; a linear scan is fine at these series sizes.
    static func nearestIndex(hoveredX: Double, points: [CGPoint]) -> Int? {
        guard !points.isEmpty else { return nil }
        var best = 0
        var bestDist = abs(Double(points[0].x) - hoveredX)
        for i in 1..<points.count {
            let d = abs(Double(points[i].x) - hoveredX)
            if d < bestDist { bestDist = d; best = i }
        }
        return best
    }
}
