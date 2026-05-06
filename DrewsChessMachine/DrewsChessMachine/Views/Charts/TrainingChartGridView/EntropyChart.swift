import Charts
import SwiftUI

/// Legal-only policy-entropy chart — Shannon entropy in nats over the
/// legal-renormalized softmax at the probed position.
///
/// The full-head `pEnt` (entropy over the entire 4864-cell policy
/// head) was removed because it conflates "is the policy concentrating
/// on a single legal move" with "is policy mass leaking onto illegal
/// moves." `pEntLegal` answers the first question on its own.
struct EntropyChart: View {
    let buckets: [TrainingBucket]
    @Binding var hoveredSec: Double?
    @Binding var scrollX: Double
    let context: TrainingChartGridView.Context

    /// Reference uniform-distribution entropy for the legal-only
    /// renormalized softmax. A typical chess position has ~30 legal
    /// moves, so log(30) ≈ 3.40 nats is a reasonable "fully diffuse
    /// over legal" baseline; the header reports the current value as
    /// a percentage of this reference so 100 % means uniform-over-legal.
    nonisolated static let maxLegalEntropy = log(30.0)

    var body: some View {
        let readout = TrainingChartGridView.hoverReadoutTraining(
            hoveredSec: hoveredSec,
            buckets: buckets,
            accessor: { $0.legalEntropy },
            bucketWidthSec: context.bucketWidthSec
        )
        let latest = buckets.last?.legalEntropy?.max
        let headerText = entropyHeader(readout: readout, latest: latest)

        return VStack(alignment: .leading, spacing: 1) {
            ChartTileHeader(title: "Policy entropy", value: headerText)
            Chart {
                ForEach(buckets) { b in
                    LineMark(
                        x: .value("Time", b.elapsedSec),
                        y: .value("Entropy", b.legalEntropy?.max ?? .nan)
                    )
                    .foregroundStyle(.green)
                }
                if let t = hoveredSec {
                    RuleMark(x: .value("Time", t))
                        .foregroundStyle(Color.gray.opacity(0.5))
                        .lineStyle(StrokeStyle(lineWidth: 1))
                }
                if case .hoveringWithData(let t, let v) = readout {
                    PointMark(x: .value("Time", t), y: .value("Entropy", v))
                        .foregroundStyle(.green)
                        .symbolSize(40)
                }
            }
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
        latest: Double?
    ) -> String {
        func format(_ v: Double) -> String {
            String(format: "%.3f (%.1f%%)", v, v / Self.maxLegalEntropy * 100)
        }
        switch readout {
        case .notHovering:
            if let v = latest {
                return "pEntLegal \(format(v))"
            } else {
                return "--"
            }
        case .hoveringNoData:
            return "— no data"
        case .hoveringWithData(_, let v):
            return "pEntLegal \(format(v))"
        }
    }
}
