import SwiftUI

/// The data-only Canvas content. Holds only the inputs that affect
/// the rasterized path/band layer — series, reference lines, domain,
/// gridline geometry — so SwiftUI's view-graph diffing skips
/// re-evaluation when only hover state changes on the parent. The
/// hover crosshair lives in a separate sibling overlay so the
/// rasterized texture below is not invalidated by mouse motion.
///
/// Wrapped in `.drawingGroup()` so the Canvas commits to a single
/// Metal-backed offscreen texture. The texture is rebuilt only when
/// this View's inputs actually change (data append, zoom, scroll,
/// size change). Text labels are NOT drawn here — they live as live
/// SwiftUI `Text` views in a sibling overlay above this layer so
/// they stay crisp and don't force re-rasterization on hover.
struct FastChartCanvasContent: View {
    let series: [FastChartSeries]
    let referenceLines: [FastChartReferenceLine]
    let xDomain: ClosedRange<Double>
    let yDomain: ClosedRange<Double>
    /// Y positions where gridlines are drawn. Must match the
    /// labels drawn outside this view.
    let yTicks: [Double]
    /// X positions where gridlines are drawn.
    let xTicks: [Double]
    let gridlineColor: Color
    let backgroundColor: Color

    var body: some View {
        Canvas(rendersAsynchronously: false) { context, size in
            draw(context: &context, size: size)
        }
        .drawingGroup()
    }

    private func draw(context: inout GraphicsContext, size: CGSize) {
        let rect = CGRect(origin: .zero, size: size)

        context.fill(Path(rect), with: .color(backgroundColor))

        let transform = ChartCoordTransform(
            xDomain: xDomain,
            yDomain: yDomain,
            rect: rect
        )

        drawGridlines(context: &context, transform: transform, rect: rect)
        drawReferenceLines(context: &context, transform: transform, rect: rect)
        drawSeries(context: &context, transform: transform)
    }

    private func drawGridlines(
        context: inout GraphicsContext,
        transform: ChartCoordTransform,
        rect: CGRect
    ) {
        var path = Path()
        for y in yTicks {
            let vy = transform.viewY(y)
            path.move(to: CGPoint(x: rect.minX, y: vy))
            path.addLine(to: CGPoint(x: rect.maxX, y: vy))
        }
        for x in xTicks {
            let vx = transform.viewX(x)
            path.move(to: CGPoint(x: vx, y: rect.minY))
            path.addLine(to: CGPoint(x: vx, y: rect.maxY))
        }
        context.stroke(
            path,
            with: .color(gridlineColor),
            style: StrokeStyle(lineWidth: 0.5)
        )
    }

    private func drawReferenceLines(
        context: inout GraphicsContext,
        transform: ChartCoordTransform,
        rect: CGRect
    ) {
        for ref in referenceLines {
            guard ref.y.isFinite else { continue }
            let vy = transform.viewY(ref.y)
            guard vy >= rect.minY && vy <= rect.maxY else { continue }
            var path = Path()
            path.move(to: CGPoint(x: rect.minX, y: vy))
            path.addLine(to: CGPoint(x: rect.maxX, y: vy))
            let style: StrokeStyle = ref.dashed
                ? StrokeStyle(lineWidth: ref.lineWidth, dash: [3, 3])
                : StrokeStyle(lineWidth: ref.lineWidth)
            context.stroke(path, with: .color(ref.color), style: style)
        }
    }

    private func drawSeries(
        context: inout GraphicsContext,
        transform: ChartCoordTransform
    ) {
        for s in series {
            switch s.data {
            case .points(let pts):
                let visible = ChartPathBuilder.visibleRange(in: pts, xDomain: xDomain)
                guard !visible.isEmpty else { continue }
                let path = ChartPathBuilder.strokePath(
                    points: pts,
                    visible: visible,
                    interpolation: s.interpolation,
                    transform: transform
                )
                context.stroke(
                    path,
                    with: .color(s.color),
                    style: StrokeStyle(lineWidth: s.lineWidth, lineJoin: .round)
                )
            case .buckets(let bs):
                let visible = ChartPathBuilder.visibleRange(in: bs, xDomain: xDomain)
                guard !visible.isEmpty else { continue }
                if s.showMinMaxBand {
                    let band = ChartPathBuilder.bandPath(
                        buckets: bs,
                        visible: visible,
                        transform: transform
                    )
                    context.stroke(
                        band,
                        with: .color(s.color.opacity(0.25)),
                        style: StrokeStyle(lineWidth: max(0.5, s.lineWidth * 0.75))
                    )
                }
                let path = ChartPathBuilder.strokePath(
                    buckets: bs,
                    visible: visible,
                    interpolation: s.interpolation,
                    transform: transform
                )
                context.stroke(
                    path,
                    with: .color(s.color),
                    style: StrokeStyle(lineWidth: s.lineWidth, lineJoin: .round)
                )
            }
        }
    }
}
