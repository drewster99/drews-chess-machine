import Charts
import SwiftUI

/// Value-head row tile: the W/D/L softmax batch means (`pW` green,
/// `pD` gray, `pL` red), which sum to ≈ 1 and live on `[0, 1]`.
///
/// This is the direct readout of the value head's behaviour after
/// the WDL switch. A fresh head starts at `pD ≈ 0.75` (the
/// `[0, ln 6, 0]` bias init) with `pW ≈ pL ≈ 0.125`; healthy
/// training pulls `pD` *down* toward the buffer's true draw rate as
/// the head learns to call decisive games, with `pW`/`pL` rising and
/// staying roughly symmetric. The failure mode the WDL representation
/// was adopted to escape is `pD → 1` (`pW ≈ pL ≈ 0` — "everything is
/// a draw") — `TrainingAlarmController` raises on it; this tile is
/// where you watch it coming.
struct WDLProbabilityChart: View {
    let buckets: [TrainingBucket]
    @Binding var hoveredSec: Double?
    @Binding var scrollX: Double
    let context: TrainingChartGridView.Context

    var body: some View {
        let wReadout = TrainingChartGridView.hoverReadoutTraining(
            hoveredSec: hoveredSec,
            buckets: buckets,
            accessor: { $0.valueProbWin },
            bucketWidthSec: context.bucketWidthSec
        )
        let dReadout = TrainingChartGridView.hoverReadoutTraining(
            hoveredSec: hoveredSec,
            buckets: buckets,
            accessor: { $0.valueProbDraw },
            bucketWidthSec: context.bucketWidthSec
        )
        let lReadout = TrainingChartGridView.hoverReadoutTraining(
            hoveredSec: hoveredSec,
            buckets: buckets,
            accessor: { $0.valueProbLoss },
            bucketWidthSec: context.bucketWidthSec
        )
        func value(
            for readout: TrainingChartGridView.HoverReadout,
            lastBucketValue: Double?
        ) -> String {
            switch readout {
            case .notHovering:
                return lastBucketValue.map { String(format: "%.3f", $0) } ?? "--"
            case .hoveringNoData:
                return "--"
            case .hoveringWithData(_, let v):
                return String(format: "%.3f", v)
            }
        }
        let wStr = value(for: wReadout, lastBucketValue: buckets.last?.valueProbWin?.max)
        let dStr = value(for: dReadout, lastBucketValue: buckets.last?.valueProbDraw?.max)
        let lStr = value(for: lReadout, lastBucketValue: buckets.last?.valueProbLoss?.max)
        let headerText: String
        if wStr == "--" && dStr == "--" && lStr == "--" {
            switch dReadout {
            case .hoveringNoData, .hoveringWithData:
                headerText = "— no data"
            case .notHovering:
                headerText = "--"
            }
        } else {
            headerText = "W \(wStr) / D \(dStr) / L \(lStr)"
        }
        return VStack(alignment: .leading, spacing: 1) {
            ChartTileHeader(title: "value W/D/L probabilities", value: headerText)
            Chart {
                ForEach(buckets) { b in
                    LineMark(
                        x: .value("Time", b.elapsedSec),
                        y: .value("p", b.valueProbWin?.max ?? .nan)
                    )
                    .foregroundStyle(by: .value("Series", "pW"))
                }
                ForEach(buckets) { b in
                    LineMark(
                        x: .value("Time", b.elapsedSec),
                        y: .value("p", b.valueProbDraw?.max ?? .nan)
                    )
                    .foregroundStyle(by: .value("Series", "pD"))
                }
                ForEach(buckets) { b in
                    LineMark(
                        x: .value("Time", b.elapsedSec),
                        y: .value("p", b.valueProbLoss?.max ?? .nan)
                    )
                    .foregroundStyle(by: .value("Series", "pL"))
                }
                // 0.75 reference: the W/D/L head's draw-bias init. A
                // pD trace sitting persistently above this is the
                // regression-toward-collapse signal.
                RuleMark(y: .value("init pD", 0.75))
                    .foregroundStyle(Color.gray.opacity(0.4))
                    .lineStyle(StrokeStyle(lineWidth: 0.5, dash: [2, 2]))
                if let t = hoveredSec {
                    RuleMark(x: .value("Time", t))
                        .foregroundStyle(Color.gray.opacity(0.5))
                        .lineStyle(StrokeStyle(lineWidth: 1))
                }
                if case .hoveringWithData(let t, let v) = wReadout {
                    PointMark(x: .value("Time", t), y: .value("p", v))
                        .foregroundStyle(Color.green)
                        .symbolSize(40)
                }
                if case .hoveringWithData(let t, let v) = dReadout {
                    PointMark(x: .value("Time", t), y: .value("p", v))
                        .foregroundStyle(Color.gray)
                        .symbolSize(40)
                }
                if case .hoveringWithData(let t, let v) = lReadout {
                    PointMark(x: .value("Time", t), y: .value("p", v))
                        .foregroundStyle(Color.red)
                        .symbolSize(40)
                }
            }
            .chartYScale(domain: 0...1)
            .chartForegroundStyleScale([
                "pW": Color.green,
                "pD": Color.gray,
                "pL": Color.red
            ])
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
