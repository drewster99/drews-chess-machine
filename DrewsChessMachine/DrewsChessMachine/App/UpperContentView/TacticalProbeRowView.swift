import SwiftUI

/// One row of the Tactical Probe Monitor: verdict pill, short probe
/// description, the move the network currently picks (ACTUAL), the
/// move we expected (EXPECTED), then three metric columns (Prob,
/// Rank, Entropy %) each showing a value + delta arrow + spark, then
/// the value head's W/D/L softmax.
///
/// Tick-direction color for the metric values: green if the current
/// value > previous, red if < previous, neutral gray otherwise (no
/// prior tick, or equal to the previous). The same color is reused
/// for the spark stroke, so the row's color story is visually
/// unified. Note that for `rank` this means "rank numerically went
/// up" not "rank improved" — rank going from 5 to 1 is a numerical
/// drop and renders red even though it's a tactical improvement; the
/// user reads the direction literally and applies the semantic
/// interpretation themselves.
///
/// The row itself is tappable — clicking anywhere on it opens
/// `TacticalProbeBoardPopover` showing the position rendered with
/// the network's top-5 legal moves as arrows.
struct TacticalProbeRowView: View {
    let probeName: String
    let current: TacticalProbeHistory.Entry
    let previous: TacticalProbeHistory.Entry?
    let probSeries: [Float]
    let rankSeries: [Float]
    let entropyPctSeries: [Float]

    /// Click-toggled popover showing the rendered board for this
    /// probe's position, with the top-K legal moves drawn as arrows.
    /// Local @State so each row's popover toggle is independent.
    @State private var isShowingBoardPopover = false

    var body: some View {
        HStack(spacing: 8) {
            verdictPill
                .frame(width: 100, alignment: .center)
            Text(current.result.probe.shortDescription)
                .font(.system(.body))
                .lineLimit(1)
                .truncationMode(.tail)
                .frame(width: 130, alignment: .leading)

            actualMoveCell
                .frame(width: 80, alignment: .leading)
            expectedMoveCell
                .frame(width: 80, alignment: .leading)

            metricCell(
                value: current.result.expectedProb,
                previous: previous?.result.expectedProb,
                format: "%.3f",
                series: probSeries,
                valueWidth: 56
            )
            metricCell(
                value: current.result.expectedRank.map(Float.init) ?? Float.nan,
                previous: previous?.result.expectedRank.map(Float.init),
                format: "%g",
                series: rankSeries,
                valueWidth: 38
            )
            metricCell(
                value: entropyPercent(current.result),
                previous: previous.map { entropyPercent($0.result) },
                format: "%.0f%%",
                series: entropyPctSeries,
                valueWidth: 56
            )
            wdlCell
                .frame(width: 120, alignment: .trailing)
        }
        .padding(.vertical, 2)
        .contentShape(Rectangle())
        .onTapGesture {
            isShowingBoardPopover.toggle()
        }
        .popover(isPresented: $isShowingBoardPopover, arrowEdge: .leading) {
            TacticalProbeBoardPopover(result: current.result)
        }
    }

    // MARK: Verdict pill

    @ViewBuilder
    private var verdictPill: some View {
        let v = current.result.verdict
        Text(verdictLabel(v))
            .font(.system(.caption2, design: .monospaced).weight(.semibold))
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(
                RoundedRectangle(cornerRadius: 4)
                    .fill(verdictColor(v).opacity(0.18))
            )
            .overlay(
                RoundedRectangle(cornerRadius: 4)
                    .stroke(verdictColor(v).opacity(0.55), lineWidth: 1)
            )
            .foregroundStyle(verdictColor(v))
    }

    // MARK: Actual / expected move cells

    /// The network's current top-mass move. Green when it equals an
    /// acceptable (correct) move — i.e. the network is currently
    /// picking right — primary text otherwise. Hyphen if no top move
    /// is available (e.g. the probe errored out).
    @ViewBuilder
    private var actualMoveCell: some View {
        if let top = current.result.topMoves.first {
            let isCorrect = current.result.probe.acceptable.contains(top.move)
            Text(top.move.notation)
                .font(.system(.body, design: .monospaced))
                .foregroundStyle(isCorrect ? Color.green : Color.primary)
                .lineLimit(1)
                .truncationMode(.tail)
        } else {
            Text("—")
                .font(.system(.body, design: .monospaced))
                .foregroundStyle(.secondary)
        }
    }

    /// The fixture's correct move (or first of several if the fixture
    /// has multiple acceptable). Shown in the secondary color since
    /// it's static reference info — what we want the network to play
    /// — not a live metric.
    @ViewBuilder
    private var expectedMoveCell: some View {
        Text(expectedNotation)
            .font(.system(.body, design: .monospaced))
            .foregroundStyle(.secondary)
            .lineLimit(1)
            .truncationMode(.tail)
    }

    private var expectedNotation: String {
        if let first = current.result.probe.acceptable.sorted(by: { $0.notation < $1.notation }).first {
            let count = current.result.probe.acceptable.count
            return count > 1 ? "\(first.notation)+\(count - 1)" : first.notation
        }
        return "—"
    }

    // MARK: WDL cell — three small probabilities

    @ViewBuilder
    private var wdlCell: some View {
        let wdl = current.result.valueWDL
        Text(String(format: "%.2f/%.2f/%.2f", wdl.win, wdl.draw, wdl.loss))
            .font(.system(.caption, design: .monospaced))
            .monospacedDigit()
            .foregroundStyle(.secondary)
    }

    // MARK: Metric cell — value + delta + spark

    @ViewBuilder
    private func metricCell(
        value: Float,
        previous: Float?,
        format: String,
        series: [Float],
        valueWidth: CGFloat
    ) -> some View {
        let direction = deltaDirection(current: value, previous: previous)
        let color = colorForDirection(direction)
        HStack(spacing: 4) {
            Text(value.isFinite ? String(format: format, value) : "—")
                .font(.system(.body, design: .monospaced))
                .monospacedDigit()
                .foregroundStyle(color)
                .frame(width: valueWidth, alignment: .trailing)
            arrow(for: direction)
                .frame(width: 10)
            TacticalProbeSparkView(values: series, stroke: color)
                .frame(width: 80)
        }
    }

    @ViewBuilder
    private func arrow(for direction: Direction) -> some View {
        switch direction {
        case .up:
            Image(systemName: "arrow.up")
                .font(.caption2.weight(.semibold))
                .foregroundStyle(Color.green)
        case .down:
            Image(systemName: "arrow.down")
                .font(.caption2.weight(.semibold))
                .foregroundStyle(Color.red)
        case .neutral:
            Image(systemName: "minus")
                .font(.caption2.weight(.regular))
                .foregroundStyle(Color.secondary)
        }
    }

    // MARK: Direction helpers

    private enum Direction { case up, down, neutral }

    private func deltaDirection(current: Float, previous: Float?) -> Direction {
        guard current.isFinite else { return .neutral }
        guard let prev = previous, prev.isFinite else { return .neutral }
        if current > prev { return .up }
        if current < prev { return .down }
        return .neutral
    }

    private func colorForDirection(_ d: Direction) -> Color {
        switch d {
        case .up: return .green
        case .down: return .red
        case .neutral: return .primary
        }
    }

    /// Entropy as a percentage of the legal-uniform ceiling. 100% =
    /// uniform on legals (no preference); 0% = collapsed onto a single
    /// move. Returns 0 for the degenerate `uniformLegalEntropy == 0`
    /// case (one-legal-move position — entropy is by definition 0).
    private func entropyPercent(_ r: ProbeResult) -> Float {
        let denom = r.uniformLegalEntropy
        guard denom > 1e-6 else { return 0 }
        return r.legalEntropyNats / denom * 100
    }

    // MARK: Verdict color / label

    private func verdictColor(_ v: ProbeVerdict) -> Color {
        switch v {
        case .correctAndConfident: return .green
        case .correctButFlat: return .yellow
        case .correctInTop5: return .orange
        case .wrong: return .red
        case .error: return .gray
        }
    }

    private func verdictLabel(_ v: ProbeVerdict) -> String {
        switch v {
        case .correctAndConfident: return "TOP·CONF"
        case .correctButFlat: return "TOP·FLAT"
        case .correctInTop5: return "TOP·5"
        case .wrong: return "WRONG"
        case .error: return "ERROR"
        }
    }
}
