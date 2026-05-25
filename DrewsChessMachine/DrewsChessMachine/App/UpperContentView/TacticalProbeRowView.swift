import SwiftUI

/// One row of the Tactical Probe Monitor: verdict pill on the left,
/// then three metric columns (Prob, Rank, Entropy) each showing a
/// value + delta arrow + spark, then the value head's W/D/L softmax.
///
/// Tick-direction color: green if the current value > previous,
/// red if < previous, neutral gray otherwise (no prior tick, or equal
/// to the previous). The same color is reused for the spark stroke,
/// so the row's color story is visually unified. Note that for `rank`
/// this means "rank numerically went up" not "rank improved" — rank
/// going from 5 to 1 is a numerical drop and renders red even though
/// it's a tactical improvement; the user reads the direction
/// literally and applies the semantic interpretation themselves.
struct TacticalProbeRowView: View {
    let probeName: String
    let current: TacticalProbeHistory.Entry
    let previous: TacticalProbeHistory.Entry?
    let probSeries: [Float]
    let rankSeries: [Float]
    let entropySeries: [Float]

    /// Click-toggled popover showing the rendered board for this
    /// probe's position, with the top-K legal moves drawn as arrows.
    /// Local @State so each row's popover toggle is independent.
    @State private var isShowingBoardPopover = false

    var body: some View {
        HStack(spacing: 8) {
            verdictPill
                .frame(width: 100, alignment: .center)
            Text(probeName)
                .font(.system(.body))
                .lineLimit(1)
                .truncationMode(.tail)
                .frame(minWidth: 180, idealWidth: 240, maxWidth: .infinity, alignment: .leading)

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
                value: current.result.legalEntropyNats,
                previous: previous?.result.legalEntropyNats,
                format: "%.2f",
                series: entropySeries,
                valueWidth: 56
            )
            wdlCell
                .frame(width: 120, alignment: .trailing)
        }
        .padding(.vertical, 2)
        // Make the entire row tappable (not just the parts with text
        // / arrows / sparks) so clicking near the row's empty padding
        // also triggers the popover.
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
