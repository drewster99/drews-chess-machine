import SwiftUI

/// A line chart designed to handle 10k+ visible points and frequent
/// data appends without the per-update stalls that SwiftUI `Charts`
/// hits at scale. The rasterized layer is a single Metal-backed
/// texture produced by a `Canvas` + `.drawingGroup()`; hover
/// crosshair and axis labels are siblings above it so neither
/// invalidates the rasterized layer.
///
/// One struct, reused N times per app. Shared horizontal hover
/// crosshair across instances is wired through `FastChartGroup`.
/// Scroll/zoom is bound-driven: caller updates `xDomain` and the
/// chart re-renders for the new visible window. Y range is caller-
/// supplied — the chart does not auto-scale. Decimation is opt-in
/// upstream via `FastChartDecimator`.
public struct FastLineChart: View {
    private let title: String
    private let titleHelp: AttributedString?
    private let group: FastChartGroup
    private let xDomain: ClosedRange<Double>
    private let yDomain: ClosedRange<Double>
    /// Independent domain for series tagged `yAxis: .secondary`. When
    /// non-nil the chart reserves a trailing label column and draws a
    /// second set of Y labels on the right; when nil the chart is a
    /// plain single-axis chart exactly as before.
    private let secondaryYDomain: ClosedRange<Double>?
    private let secondaryYTickValues: [Double]?
    private let secondaryYLabelFormatter: @Sendable (Double) -> String
    /// Tint for the trailing-axis labels. When nil they render in the
    /// same secondary style as the primary labels; pass a color to
    /// visually bind the right axis to the series it scales.
    private let secondaryYLabelColor: Color?
    private let series: [FastChartSeries]
    private let referenceLines: [FastChartReferenceLine]
    private let yLabelCount: Int
    private let xLabelCount: Int
    private let yTickValues: [Double]?
    private let xTickValues: [Double]?
    private let showXAxisLabels: Bool
    private let yLabelFormatter: @Sendable (Double) -> String
    private let xLabelFormatter: @Sendable (Double) -> String
    private let legend: FastChartLegend
    private let headerValue: ((FastChartHoverContext) -> AttributedString)?

    /// Local state for the title's "click-to-explain" popover.
    /// Owned here so each `FastLineChart` instance gets its own
    /// independent popover toggle; callers stay stateless.
    @State private var showingTitleHelp: Bool = false

    public init(
        title: String,
        titleHelp: AttributedString? = nil,
        group: FastChartGroup,
        xDomain: ClosedRange<Double>,
        yDomain: ClosedRange<Double>,
        secondaryYDomain: ClosedRange<Double>? = nil,
        secondaryYTickValues: [Double]? = nil,
        secondaryYLabelFormatter: @escaping @Sendable (Double) -> String = FastChartFormatters.compact,
        secondaryYLabelColor: Color? = nil,
        series: [FastChartSeries],
        referenceLines: [FastChartReferenceLine] = [],
        yLabelCount: Int = 4,
        xLabelCount: Int = 3,
        yTickValues: [Double]? = nil,
        xTickValues: [Double]? = nil,
        showXAxisLabels: Bool = false,
        yLabelFormatter: @escaping @Sendable (Double) -> String = FastChartFormatters.compact,
        xLabelFormatter: @escaping @Sendable (Double) -> String = FastChartFormatters.elapsedTime,
        legend: FastChartLegend = .auto,
        headerValue: ((FastChartHoverContext) -> AttributedString)? = nil
    ) {
        self.title = title
        self.titleHelp = titleHelp
        self.group = group
        self.xDomain = xDomain
        self.yDomain = yDomain
        self.secondaryYDomain = secondaryYDomain
        self.secondaryYTickValues = secondaryYTickValues
        self.secondaryYLabelFormatter = secondaryYLabelFormatter
        self.secondaryYLabelColor = secondaryYLabelColor
        self.series = series
        self.referenceLines = referenceLines
        self.yLabelCount = yLabelCount
        self.xLabelCount = xLabelCount
        self.yTickValues = yTickValues
        self.xTickValues = xTickValues
        self.showXAxisLabels = showXAxisLabels
        self.yLabelFormatter = yLabelFormatter
        self.xLabelFormatter = xLabelFormatter
        self.legend = legend
        self.headerValue = headerValue
    }

    /// Tick positions actually rendered for the Y axis. Explicit
    /// `yTickValues` win when provided (clipped to `yDomain`).
    /// Otherwise we fall back to "nice number" ticks so generic
    /// numeric data — entropy, gradient norms, memory, etc. — gets
    /// labels at round values like `0, 20, 40` instead of
    /// `0, 18, 36.1, 54.1`. Gridlines + axis labels read from the
    /// same array so they always align.
    private var yTicks: [Double] {
        ChartAxisLayout.resolvedTicks(
            explicit: yTickValues,
            domain: yDomain,
            fallback: { ChartAxisLayout.niceTicks(domain: $0, approxCount: yLabelCount) }
        )
    }

    /// Tick positions for the X axis. Default is evenly-spaced
    /// rather than nice-number, because elapsed-time domains are
    /// already chosen from `ChartZoom`'s nice stops (15m/30m/1h/…)
    /// and the X formatter renders them as `mm:ss` / `h:mm:ss`.
    /// Generic nice-number ticks at raw-seconds boundaries would
    /// instead read as `27:46:40` and friends.
    private var xTicks: [Double] {
        ChartAxisLayout.resolvedTicks(
            explicit: xTickValues,
            domain: xDomain,
            fallback: { ChartAxisLayout.evenlySpacedTicks(domain: $0, count: xLabelCount) }
        )
    }

    /// Tick positions for the trailing (secondary) Y axis. Empty when
    /// no `secondaryYDomain` is set. Mirrors the primary `yTicks`
    /// resolution so the right labels read as round numbers too.
    private var secondaryYTicks: [Double] {
        guard let secondaryYDomain else { return [] }
        return ChartAxisLayout.resolvedTicks(
            explicit: secondaryYTickValues,
            domain: secondaryYDomain,
            fallback: { ChartAxisLayout.niceTicks(domain: $0, approxCount: yLabelCount) }
        )
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: 1) {
            headerRow
            plotArea
            legendRow
        }
    }

    // MARK: - Header

    // The header row always renders so that tile heights line up
    // across a grid of `FastLineChart`s — even when a particular
    // tile has neither title nor header value, an empty caption2-
    // tall row is reserved so the plot area starts at the same Y
    // offset as its neighbors.
    //
    // When `titleHelp` is provided, the title text becomes a small
    // button; clicking it opens a popover with the description.
    // Without `titleHelp` the title is a plain Text — no button
    // chrome, no cursor change — so chart tiles that didn't bother
    // to write a description still look the same as before.
    private var headerRow: some View {
        HStack(spacing: 4) {
            titleLabel
            Spacer(minLength: 4)
            if let headerValue {
                let ctx = FastChartHoverContext(
                    hoveredX: group.hoveredX,
                    xDomain: xDomain,
                    yDomain: yDomain
                )
                Text(headerValue(ctx))
                    .font(.caption2)
                    .monospacedDigit()
                    .lineLimit(1)
            }
        }
    }

    @ViewBuilder
    private var titleLabel: some View {
        if let titleHelp {
            Button(action: { showingTitleHelp = true }) {
                Text(title)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }
            .buttonStyle(.plain)
            .help("Click for description")
            .popover(isPresented: $showingTitleHelp, arrowEdge: .top) {
                // `.fixedSize(horizontal: false, vertical: true)` is
                // load-bearing: without it the popover container takes
                // the Text's *unconstrained* natural single-line
                // intrinsic width and then truncates with an ellipsis
                // when the popover's host clamps the result. Pinning
                // horizontal to the proposed width (320) and letting
                // vertical grow gives a wrapped multi-line block.
                Text(titleHelp)
                    .font(.body)
                    .textSelection(.enabled)
                    .fixedSize(horizontal: false, vertical: true)
                    .frame(width: 360, alignment: .leading)
                    .padding(10)
            }
        } else {
            Text(title)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .lineLimit(1)
        }
    }

    // MARK: - Plot area

    private var plotArea: some View {
        let yTicksResolved = yTicks
        let xTicksResolved = xTicks
        let secondaryYTicksResolved = secondaryYTicks
        return GeometryReader { geo in
            let layout = ChartAxisLayout(
                totalSize: geo.size,
                yLabelWidth: yAxisLabelColumnWidth,
                xLabelHeight: showXAxisLabels ? xAxisLabelRowHeight : 0,
                rightLabelWidth: secondaryYDomain != nil ? yAxisLabelColumnWidth : 0
            )
            ZStack(alignment: .topLeading) {
                FastChartCanvasContent(
                    series: series,
                    referenceLines: referenceLines,
                    xDomain: xDomain,
                    yDomain: yDomain,
                    secondaryYDomain: secondaryYDomain,
                    yTicks: yTicksResolved,
                    xTicks: xTicksResolved,
                    gridlineColor: Color.gray.opacity(0.18),
                    backgroundColor: Color(nsColor: .windowBackgroundColor)
                )
                .frame(width: layout.plotRect.width, height: layout.plotRect.height)
                .offset(x: layout.plotRect.origin.x, y: layout.plotRect.origin.y)

                yAxisLabels(layout: layout, ticks: yTicksResolved)
                if let secondaryYDomain {
                    secondaryYAxisLabels(
                        layout: layout,
                        ticks: secondaryYTicksResolved,
                        domain: secondaryYDomain
                    )
                }
                if showXAxisLabels {
                    xAxisLabels(layout: layout, ticks: xTicksResolved)
                }
                referenceLineLabels(layout: layout)
                hoverHitArea(layout: layout)
                crosshairOverlay(layout: layout)
            }
        }
    }

    /// Trailing-axis labels for the secondary domain. Mirrors
    /// `yAxisLabels` but positions each label inside the right-hand
    /// reserved column and maps tick values through a transform built
    /// from `domain` so they line up with the `.secondary` series.
    private func secondaryYAxisLabels(
        layout: ChartAxisLayout,
        ticks: [Double],
        domain: ClosedRange<Double>
    ) -> some View {
        let plot = layout.plotRect
        let transform = ChartCoordTransform(
            xDomain: xDomain,
            yDomain: domain,
            rect: plot
        )
        return ZStack(alignment: .topLeading) {
            ForEach(ticks.indices, id: \.self) { i in
                let y = ticks[i]
                Text(secondaryYLabelFormatter(y))
                    .font(.system(size: 7))
                    .monospacedDigit()
                    .foregroundStyle(secondaryYLabelColor ?? .secondary)
                    .lineLimit(1)
                    .frame(width: layout.rightLabelWidth - 2, alignment: .leading)
                    .position(
                        x: plot.maxX + layout.rightLabelWidth / 2,
                        y: transform.viewY(y)
                    )
            }
        }
        .frame(width: layout.totalSize.width, height: layout.totalSize.height, alignment: .topLeading)
        .allowsHitTesting(false)
    }

    private func yAxisLabels(layout: ChartAxisLayout, ticks: [Double]) -> some View {
        let plot = layout.plotRect
        let transform = ChartCoordTransform(
            xDomain: xDomain,
            yDomain: yDomain,
            rect: plot
        )
        return ZStack(alignment: .topLeading) {
            ForEach(ticks.indices, id: \.self) { i in
                let y = ticks[i]
                Text(yLabelFormatter(y))
                    .font(.system(size: 7))
                    .monospacedDigit()
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .frame(width: layout.yLabelWidth - 2, alignment: .trailing)
                    .position(
                        x: layout.yLabelWidth / 2,
                        y: transform.viewY(y)
                    )
            }
        }
        .frame(width: layout.totalSize.width, height: layout.totalSize.height, alignment: .topLeading)
        .allowsHitTesting(false)
    }

    private func xAxisLabels(layout: ChartAxisLayout, ticks: [Double]) -> some View {
        let plot = layout.plotRect
        let transform = ChartCoordTransform(
            xDomain: xDomain,
            yDomain: yDomain,
            rect: plot
        )
        return ZStack(alignment: .topLeading) {
            ForEach(ticks.indices, id: \.self) { i in
                let x = ticks[i]
                Text(xLabelFormatter(x))
                    .font(.system(size: 7))
                    .monospacedDigit()
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .position(
                        x: transform.viewX(x),
                        y: plot.maxY + layout.xLabelHeight / 2
                    )
            }
        }
        .frame(width: layout.totalSize.width, height: layout.totalSize.height, alignment: .topLeading)
        .allowsHitTesting(false)
    }

    private func referenceLineLabels(layout: ChartAxisLayout) -> some View {
        let plot = layout.plotRect
        let transform = ChartCoordTransform(
            xDomain: xDomain,
            yDomain: yDomain,
            rect: plot
        )
        return ZStack(alignment: .topLeading) {
            ForEach(referenceLines) { ref in
                if let label = ref.label, ref.y.isFinite {
                    let vy = transform.viewY(ref.y)
                    if vy >= plot.minY && vy <= plot.maxY {
                        Text(label)
                            .font(.system(size: 7))
                            .foregroundStyle(ref.color)
                            .lineLimit(1)
                            .padding(.horizontal, 2)
                            .position(
                                x: plot.maxX - 14,
                                y: max(plot.minY + 4, vy - 5)
                            )
                    }
                }
            }
        }
        .frame(width: layout.totalSize.width, height: layout.totalSize.height, alignment: .topLeading)
        .allowsHitTesting(false)
    }

    // The hover Rectangle fills the entire GeometryReader (no
    // `.offset`) so the hover coordinates `pt` arrive in the same
    // layout space the rest of the ZStack draws in — `.offset` is a
    // rendering-only transform and would not move `.local` hover
    // coordinates with it, leaving the crosshair shifted right by
    // `plot.origin.x` (the Y-label column width). We filter to the
    // plot rect in code instead.
    private func hoverHitArea(layout: ChartAxisLayout) -> some View {
        let plot = layout.plotRect
        let span = xDomain.upperBound - xDomain.lowerBound
        let xLower = xDomain.lowerBound
        return Rectangle()
            .fill(Color.clear)
            .contentShape(Rectangle())
            .onContinuousHover { phase in
                switch phase {
                case .active(let pt):
                    guard plot.width > 0 else { return }
                    let xInPlot = pt.x - plot.origin.x
                    if xInPlot < 0 || xInPlot > plot.width {
                        if group.hoveredX != nil { group.hoveredX = nil }
                        return
                    }
                    let normalized = Double(xInPlot / plot.width)
                    let dataX = xLower + normalized * span
                    if dataX < xDomain.lowerBound || dataX > xDomain.upperBound {
                        if group.hoveredX != nil { group.hoveredX = nil }
                        return
                    }
                    if group.hoveredX != dataX {
                        group.hoveredX = dataX
                    }
                case .ended:
                    if group.hoveredX != nil { group.hoveredX = nil }
                }
            }
    }

    @ViewBuilder
    private func crosshairOverlay(layout: ChartAxisLayout) -> some View {
        if let hx = group.hoveredX,
           hx >= xDomain.lowerBound && hx <= xDomain.upperBound {
            let plot = layout.plotRect
            let transform = ChartCoordTransform(
                xDomain: xDomain,
                yDomain: yDomain,
                rect: plot
            )
            let vx = transform.viewX(hx)
            Path { p in
                p.move(to: CGPoint(x: vx, y: plot.minY))
                p.addLine(to: CGPoint(x: vx, y: plot.maxY))
            }
            .stroke(Color.gray.opacity(0.5), lineWidth: 1)
            .allowsHitTesting(false)
        }
    }

    // MARK: - Legend

    @ViewBuilder
    private var legendRow: some View {
        if let items = resolvedLegendItems() {
            HStack(spacing: 8) {
                ForEach(items) { item in
                    HStack(spacing: 3) {
                        Rectangle()
                            .fill(item.color)
                            .frame(width: 8, height: 2)
                        Text(item.label)
                            .font(.system(size: 8))
                            .foregroundStyle(.secondary)
                    }
                }
                Spacer(minLength: 0)
            }
            .padding(.leading, yAxisLabelColumnWidth)
        }
    }

    private func resolvedLegendItems() -> [FastChartLegendItem]? {
        switch legend {
        case .auto:
            return series.count > 1
                ? series.map { FastChartLegendItem(label: $0.id, color: $0.color) }
                : nil
        case .off:
            return nil
        case .custom(let items):
            return items
        }
    }

    // MARK: - Layout constants

    private var yAxisLabelColumnWidth: CGFloat { 26 }
    private var xAxisLabelRowHeight: CGFloat { 10 }
}
