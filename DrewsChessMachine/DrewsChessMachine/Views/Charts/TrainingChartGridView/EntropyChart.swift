import Charts
import SwiftUI

/// Policy entropy chart with an extra `pEntLegal` series.
struct EntropyChart: View {
    let buckets: [TrainingBucket]
    @Binding var hoveredSec: Double?
    @Binding var scrollX: Double
    let context: TrainingChartGridView.Context

    nonisolated static let maxEntropy = log(Double(ChessNetwork.policySize))
    /// Reference uniform-distribution entropy for the legal-only
    /// renormalized softmax. A typical chess position has ~30 legal
    /// moves, so log(30) ≈ 3.40 nats is a reasonable "fully diffuse
    /// over legal" baseline.
    nonisolated static let maxLegalEntropy = log(30.0)

    var body: some View {
        let readout = TrainingChartGridView.hoverReadoutTraining(
            hoveredSec: hoveredSec,
            buckets: buckets,
            accessor: { $0.policyEntropy },
            bucketWidthSec: context.bucketWidthSec
        )
        let latestPEnt = buckets.last?.policyEntropy?.max
        let latestPEntLegal = buckets.last?.legalEntropy?.max
        let headerText = entropyHeader(
            readout: readout,
            latestPEnt: latestPEnt,
            latestPEntLegal: latestPEntLegal
        )

        return VStack(alignment: .leading, spacing: 1) {
            ChartTileHeader(title: "Policy entropy", value: headerText)
            Chart {
                ForEach(buckets) { b in
                    LineMark(
                        x: .value("Time", b.elapsedSec),
                        y: .value("Entropy", b.policyEntropy?.max ?? .nan)
                    )
                    .foregroundStyle(by: .value("Series", "pEnt"))
                }
                ForEach(buckets) { b in
                    LineMark(
                        x: .value("Time", b.elapsedSec),
                        y: .value("Entropy", b.legalEntropy?.max ?? .nan)
                    )
                    .foregroundStyle(by: .value("Series", "pEntLegal"))
                }
                if let t = hoveredSec {
                    RuleMark(x: .value("Time", t))
                        .foregroundStyle(Color.gray.opacity(0.5))
                        .lineStyle(StrokeStyle(lineWidth: 1))
                }
                if case .hoveringWithData(let t, let v) = readout {
                    PointMark(x: .value("Time", t), y: .value("Entropy", v))
                        .foregroundStyle(.purple)
                        .symbolSize(40)
                }
            }
            .chartForegroundStyleScale([
                "pEnt": Color.purple,
                "pEntLegal": Color.green
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

    private func entropyHeader(
        readout: TrainingChartGridView.HoverReadout,
        latestPEnt: Double?,
        latestPEntLegal: Double?
    ) -> String {
        func formatEntropy(_ v: Double?) -> String {
            guard let v else { return "--" }
            return String(format: "%.3f", v)
        }
        switch readout {
        case .notHovering:
            let pEntStr: String
            if let v = latestPEnt {
                pEntStr = String(format: "%.3f (%.1f%%)", v, v / Self.maxEntropy * 100)
            } else {
                pEntStr = "--"
            }
            let pEntLegalStr: String
            if let v = latestPEntLegal {
                pEntLegalStr = String(format: "%.3f (%.1f%%)", v, v / Self.maxLegalEntropy * 100)
            } else {
                pEntLegalStr = "--"
            }
            return "pEnt \(pEntStr) / pEntLegal \(pEntLegalStr)"
        case .hoveringNoData(let t):
            return "t=\(TrainingChartGridView.formatElapsedAxis(t)) — no data"
        case .hoveringWithData(let t, let v):
            let pEntStr = String(format: "%.3f (%.1f%%)", v, v / Self.maxEntropy * 100)
            return "t=\(TrainingChartGridView.formatElapsedAxis(t)) pEnt \(pEntStr) / pEntLegal \(formatEntropy(latestPEntLegal))"
        }
    }
}
