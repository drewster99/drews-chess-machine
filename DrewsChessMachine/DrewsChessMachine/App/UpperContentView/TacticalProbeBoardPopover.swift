import SwiftUI

/// Popover content shown when the user clicks a row in the Tactical
/// Probe Monitor. Renders the probe's position on a board, draws the
/// top-K legal moves from the network's current policy as arrows, and
/// lists the same top-K with their probabilities + an "EXPECTED"
/// marker so the user can see at a glance whether the correct move is
/// in the top picks and how confident the network is.
///
/// All data comes from the latest tick's `ProbeResult`. The position
/// (`probe.state`) is fixture-static; the top-K and probabilities
/// are live and refresh every 15s while the parent monitor is open.
struct TacticalProbeBoardPopover: View {
    let result: ProbeResult

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            header
            board
                .frame(width: 320, height: 320)
            Divider()
            details
        }
        .padding(16)
        .frame(width: 360)
    }

    // MARK: Header

    @ViewBuilder
    private var header: some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            Text(result.probe.shortDescription)
                .font(.system(.headline))
                .lineLimit(2)
                .multilineTextAlignment(.leading)
            Spacer()
            Text(verdictLabel(result.verdict))
                .font(.system(.caption2, design: .monospaced).weight(.semibold))
                .padding(.horizontal, 6)
                .padding(.vertical, 2)
                .background(
                    RoundedRectangle(cornerRadius: 4)
                        .fill(verdictColor(result.verdict).opacity(0.18))
                )
                .overlay(
                    RoundedRectangle(cornerRadius: 4)
                        .stroke(verdictColor(result.verdict).opacity(0.55), lineWidth: 1)
                )
                .foregroundStyle(verdictColor(result.verdict))
        }
    }

    // MARK: Board with top-K arrows

    @ViewBuilder
    private var board: some View {
        ChessBoardView(
            pieces: result.probe.state.board,
            overlay: .topMoves(topMoveVisualizations())
        )
    }

    /// Convert the result's top-5 entries into the
    /// `MoveVisualization` shape `ChessBoardView` overlays expect.
    /// All entries are legal (the row's series is built from the
    /// legal-masked + renormalized policy, see `buildProbeResult` in
    /// `SessionController+TacticalProbe.swift`), so `isLegal` is
    /// always true here.
    private func topMoveVisualizations() -> [MoveVisualization] {
        let board = result.probe.state.board
        return result.topMoves.map { entry in
            let move = entry.move
            let piece = board[move.fromRow * 8 + move.fromCol]?.assetName
            return MoveVisualization(
                fromRow: move.fromRow,
                fromCol: move.fromCol,
                toRow: move.toRow,
                toCol: move.toCol,
                probability: entry.prob,
                piece: piece,
                isLegal: true,
                promotion: move.promotion
            )
        }
    }

    // MARK: Details

    @ViewBuilder
    private var details: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 6) {
                Text("Expected:")
                    .font(.system(.caption, design: .monospaced).weight(.semibold))
                    .foregroundStyle(.secondary)
                Text(expectedMovesText)
                    .font(.system(.caption, design: .monospaced))
            }
            Text("Top 5 legal moves (network's view):")
                .font(.system(.caption, design: .monospaced).weight(.semibold))
                .foregroundStyle(.secondary)
            VStack(alignment: .leading, spacing: 2) {
                ForEach(Array(result.topMoves.enumerated()), id: \.offset) { _, entry in
                    HStack(spacing: 8) {
                        Text(entry.move.notation)
                            .font(.system(.caption, design: .monospaced))
                            .frame(width: 100, alignment: .leading)
                            .foregroundStyle(isExpected(entry.move) ? Color.green : Color.primary)
                        Text(String(format: "%.3f", entry.prob))
                            .font(.system(.caption, design: .monospaced))
                            .frame(width: 60, alignment: .trailing)
                        if isExpected(entry.move) {
                            Text("← EXPECTED")
                                .font(.system(.caption2, design: .monospaced).weight(.semibold))
                                .foregroundStyle(Color.green)
                        }
                    }
                }
            }
            Divider()
            HStack(spacing: 12) {
                detailScalar(label: "Rank", text: rankText)
                detailScalar(label: "Prob", text: String(format: "%.3f", result.expectedProb))
                detailScalar(label: "Entropy", text: String(format: "%.2f / %.2f", result.legalEntropyNats, result.uniformLegalEntropy))
            }
            HStack(spacing: 12) {
                detailScalar(label: "Legals", text: "\(result.legalCount)")
                detailScalar(label: "IllM", text: String(format: "%.3f", result.illegalMass))
                detailScalar(
                    label: "W/D/L",
                    text: String(format: "%.2f/%.2f/%.2f",
                                 result.valueWDL.win,
                                 result.valueWDL.draw,
                                 result.valueWDL.loss)
                )
            }
        }
    }

    @ViewBuilder
    private func detailScalar(label: String, text: String) -> some View {
        HStack(spacing: 4) {
            Text("\(label):")
                .font(.system(.caption2, design: .monospaced).weight(.semibold))
                .foregroundStyle(.secondary)
            Text(text)
                .font(.system(.caption, design: .monospaced))
        }
    }

    // MARK: Helpers

    private var expectedMovesText: String {
        let sorted = result.probe.acceptable.sorted(by: { $0.notation < $1.notation })
        return sorted.map { $0.notation }.joined(separator: ", ")
    }

    private var rankText: String {
        if let r = result.expectedRank {
            return "\(r) / \(result.legalCount)"
        }
        return "—"
    }

    private func isExpected(_ move: ChessMove) -> Bool {
        result.probe.acceptable.contains(move)
    }

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
