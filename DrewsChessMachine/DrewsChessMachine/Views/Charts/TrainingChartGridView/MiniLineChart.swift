import SwiftUI
import SwiftUIFastCharts

/// Generic single-series mini chart used for pwNorm, gNorm, ||v||,
/// and the value-head row metrics. Optional dashed horizontal
/// reference line + label.
struct MiniLineChart: View {
    let title: String
    let buckets: [TrainingBucket]
    let rangeAccessor: (TrainingBucket) -> ChartBucketRange?
    let unit: String
    let color: Color
    let group: FastChartGroup
    let xDomain: ClosedRange<Double>
    let bucketWidthSec: Double
    var wholeNumber: Bool = false
    /// Optional dashed horizontal reference line value (e.g. the
    /// `gradClipMaxNorm` cap on the gNorm tile).
    var referenceLine: Double? = nil
    /// Short label rendered at the right edge of the reference line
    /// (e.g. `"clip 10"`). Ignored when `referenceLine` is nil.
    var referenceLineLabel: String? = nil
    /// Color for the reference line. Defaults to a subdued red.
    var referenceLineColor: Color = Color.red.opacity(0.55)
    /// Brief description shown in a popover when the user clicks the
    /// chart's title. Varies per call site — pwNorm / gNorm / ||v|| /
    /// vLoss / vMean / vAbs each pass their own description through.
    var titleHelp: AttributedString? = nil

    var body: some View {
        let yRange = observedYRange()
        return FastLineChart(
            title: title,
            titleHelp: titleHelp,
            group: group,
            xDomain: xDomain,
            yDomain: yRange,
            series: [
                FastChartSeries(
                    id: title,
                    color: color,
                    lineWidth: 1.5,
                    data: .buckets(buckets.enumerated().map { (i, b) in
                        let r = rangeAccessor(b)
                        return FastChartBucket(
                            id: i,
                            x: b.elapsedSec,
                            yMin: r?.min ?? .nan,
                            yMax: r?.max ?? .nan
                        )
                    })
                )
            ],
            referenceLines: referenceLine.map { ref in
                [FastChartReferenceLine(
                    id: "ref",
                    y: ref,
                    label: referenceLineLabel,
                    color: referenceLineColor,
                    lineWidth: 1,
                    dashed: true
                )]
            } ?? [],
            headerValue: { ctx in headerString(at: ctx.hoveredX) }
        )
        .frame(height: 75)
        .chartCard()
    }

    private func observedYRange() -> ClosedRange<Double> {
        let maxes = buckets.compactMap { rangeAccessor($0)?.max }
        let mins = buckets.compactMap { rangeAccessor($0)?.min }
        let observedMin = mins.min() ?? 0
        let observedMax = maxes.max() ?? 1
        var lo = observedMin
        var hi = observedMax
        // Reference line, when present, pulls the range out to
        // include it (e.g. gNorm's clip ceiling, vMean's 0-line).
        if let ref = referenceLine {
            lo = Swift.min(lo, ref)
            hi = Swift.max(hi, ref)
        }
        let span = hi - lo
        if span <= 0 {
            return (lo - 0.5)...(hi + 0.5)
        }
        // Sign-aware padding: pin the floor at 0 for non-negative
        // series (vAbs, vLoss, gNorm, pwNorm, ||v||) so the axis
        // doesn't render a band of meaningless negative ticks below
        // the data. Mirror image for purely non-positive series.
        // Series that genuinely cross 0 pad on both sides.
        if lo >= 0 {
            return 0...(hi * 1.1)
        } else if hi <= 0 {
            return (lo * 1.1)...0
        } else {
            let pad = span * 0.05
            return (lo - pad)...(hi + pad)
        }
    }

    private func headerString(at hoveredX: Double?) -> AttributedString {
        let unitSuffix = unit.isEmpty ? "" : " \(unit)"
        if let t = hoveredX {
            if let v = nearest(at: t).flatMap({ rangeAccessor($0)?.max }) {
                let valueStr = wholeNumber ? String(Int(v))
                    : FastChartFormatters.compact(v)
                return AttributedString("\(valueStr)\(unitSuffix)")
            }
            return AttributedString("— no data")
        }
        if let v = buckets.last.flatMap({ rangeAccessor($0)?.max }) {
            let valueStr = wholeNumber ? String(Int(v))
                : FastChartFormatters.compact(v)
            return AttributedString("\(valueStr)\(unitSuffix)")
        }
        return AttributedString("--")
    }

    private func nearest(at t: Double) -> TrainingBucket? {
        TrainingChartGridView.nearestTrainingBucket(
            at: t,
            in: buckets,
            tolerance: Swift.max(
                TrainingChartGridView.hoverMatchToleranceSec,
                bucketWidthSec * 1.5
            )
        )
    }
}
