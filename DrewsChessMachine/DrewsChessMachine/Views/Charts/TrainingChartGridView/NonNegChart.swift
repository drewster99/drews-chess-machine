import Charts
import SwiftUI

/// Above-uniform policy count chart — legal vs illegal counts on
/// a fixed `0...policySize` Y axis.
struct NonNegChart: View {
    let buckets: [TrainingBucket]
    @Binding var hoveredSec: Double?
    @Binding var scrollX: Double
    let context: TrainingChartGridView.Context

    var body: some View {
        let legalReadout = TrainingChartGridView.hoverReadoutTraining(
            hoveredSec: hoveredSec,
            buckets: buckets,
            accessor: { $0.policyNonNegCount },
            bucketWidthSec: context.bucketWidthSec
        )
        let policyMax = ChessNetwork.policySize
        let headerText: String
        switch legalReadout {
        case .notHovering:
            let lastLegal = buckets.last?.policyNonNegCount?.max
            let lastIllegal = buckets.last?.policyNonNegIllegalCount?.max
            let legalStr = lastLegal.map { String(Int($0)) } ?? "--"
            let illegalStr = lastIllegal.map { String(Int($0)) } ?? "--"
            headerText = "legal \(legalStr) • illegal \(illegalStr)"
        case .hoveringNoData(let t):
            headerText = "t=\(TrainingChartGridView.formatElapsedAxis(t)) — no data"
        case .hoveringWithData(let t, let v):
            // Pull the matching illegal value from the same hovered bucket.
            let tolerance = Swift.max(
                TrainingChartGridView.hoverMatchToleranceSec,
                context.bucketWidthSec * 1.5
            )
            let illegalAtHover = TrainingChartGridView.nearestTrainingBucket(
                at: t, in: buckets, tolerance: tolerance
            )?.policyNonNegIllegalCount?.max
            let illegalStr = illegalAtHover.map { String(Int($0)) } ?? "--"
            headerText = "t=\(TrainingChartGridView.formatElapsedAxis(t))  legal \(Int(v)) • illegal \(illegalStr)"
        }
        return VStack(alignment: .leading, spacing: 1) {
            ChartTileHeader(title: "Above-uniform policy count", value: headerText)
            Chart {
                ForEach(buckets) { b in
                    LineMark(
                        x: .value("Time", b.elapsedSec),
                        y: .value("Count", b.policyNonNegCount?.max ?? .nan),
                        series: .value("Series", "Legal")
                    )
                    .foregroundStyle(by: .value("Series", "Legal"))
                    LineMark(
                        x: .value("Time", b.elapsedSec),
                        y: .value("Count", b.policyNonNegIllegalCount?.max ?? .nan),
                        series: .value("Series", "Illegal")
                    )
                    .foregroundStyle(by: .value("Series", "Illegal"))
                }
                if let t = hoveredSec {
                    RuleMark(x: .value("Time", t))
                        .foregroundStyle(Color.gray.opacity(0.5))
                        .lineStyle(StrokeStyle(lineWidth: 1))
                }
                if case .hoveringWithData(let t, let v) = legalReadout {
                    PointMark(x: .value("Time", t), y: .value("Count", v))
                        .foregroundStyle(.mint)
                        .symbolSize(40)
                }
            }
            .chartForegroundStyleScale([
                "Legal": Color.mint,
                "Illegal": Color.red
            ])
            .chartLegend(position: .bottom, alignment: .leading, spacing: 4)
            .chartYScale(domain: 0...Double(policyMax))
            .modifier(StandardTimeSeriesChartModifiers(
                context: context,
                scrollX: $scrollX,
                hoveredSec: $hoveredSec
            ))
        }
        .frame(height: 75)
        .chartCard()
    }
}
