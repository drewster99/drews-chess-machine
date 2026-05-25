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

    /// Four-state cycle on board taps: all arrows → no arrows →
    /// just the expected move → just the other (non-expected) moves
    /// → back to all. Lets the user dial down visual noise on probes
    /// where the top-5 fan-out is busy, and isolate the expected vs
    /// network-actual contrast.
    @State private var arrowDisplayMode: ArrowDisplayMode = .all

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            header
            board
                .frame(width: 320, height: 320)
                .contentShape(Rectangle())
                .onTapGesture {
                    arrowDisplayMode = arrowDisplayMode.next()
                }
            arrowModeLabel
            Divider()
            details
        }
        .padding(16)
        .frame(width: 360)
    }

    @ViewBuilder
    private var arrowModeLabel: some View {
        HStack(spacing: 4) {
            Text("Arrows:")
                .foregroundStyle(.secondary)
            Text(arrowDisplayMode.label)
            Spacer()
            Text("tap board to cycle")
                .foregroundStyle(.secondary.opacity(0.6))
        }
        .font(.system(.caption2, design: .monospaced))
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

    /// Build the visualizations for the current display mode.
    ///
    /// Source set: the network's top-5 legal moves + the expected
    /// move(s) from the fixture if not already in the top 5. Each
    /// arrow gets a color encoding what it represents:
    ///
    ///   • Expected move, rank 1                → green
    ///   • Expected move, rank 2-5              → yellow-green
    ///   • Expected move, NOT in top 5          → red
    ///   • Network's #1 move, NOT expected      → blue
    ///   • Other top-5 moves, not expected      → tan
    ///
    /// `ChessBoardView` uses the per-move color when present (we set
    /// it here) and otherwise falls back to its default rank-based
    /// hue gradient. Drawing order matters: arrows later in the array
    /// render UNDER arrows earlier (the overlay loop iterates
    /// reversed), so the synthetic expected-not-in-top-5 arrow is
    /// placed FIRST so it renders on top of the top-5 arrows.
    private func topMoveVisualizations() -> [MoveVisualization] {
        let board = result.probe.state.board
        let acceptable = result.probe.acceptable
        let topMoves = result.topMoves

        // Detect whether any acceptable move is in the top-5 by
        // identity (ChessMove is Equatable / Hashable). If none are,
        // we'll synthesize an arrow for the fixture-canonical
        // acceptable move so the user can see "where the answer
        // would have been."
        let topMoveSet = Set(topMoves.map { $0.move })
        let acceptableInTop5 = acceptable.contains(where: { topMoveSet.contains($0) })

        var arrows: [MoveVisualization] = []

        // Synthetic expected arrow (drawn ON TOP — first in the
        // array — so it isn't hidden under top-5 arrows).
        if !acceptableInTop5,
           let expected = acceptable.sorted(by: { $0.notation < $1.notation }).first {
            let piece = board[expected.fromRow * 8 + expected.fromCol]?.assetName
            arrows.append(
                MoveVisualization(
                    fromRow: expected.fromRow,
                    fromCol: expected.fromCol,
                    toRow: expected.toRow,
                    toCol: expected.toCol,
                    probability: result.expectedProb,
                    piece: piece,
                    isLegal: true,
                    promotion: expected.promotion,
                    color: Self.colorExpectedMissed
                )
            )
        }

        // Top-5 arrows in their original order. Color computed per
        // the rules above.
        for (rank, entry) in topMoves.enumerated() {
            let move = entry.move
            let piece = board[move.fromRow * 8 + move.fromCol]?.assetName
            let isExpected = acceptable.contains(move)
            let isTopOne = rank == 0
            let color: Color
            if isExpected {
                color = isTopOne ? Self.colorExpectedRank1 : Self.colorExpectedInTop5
            } else {
                color = isTopOne ? Self.colorTop1NotExpected : Self.colorOtherTop5
            }
            arrows.append(
                MoveVisualization(
                    fromRow: move.fromRow,
                    fromCol: move.fromCol,
                    toRow: move.toRow,
                    toCol: move.toCol,
                    probability: entry.prob,
                    piece: piece,
                    isLegal: true,
                    promotion: move.promotion,
                    color: color
                )
            )
        }

        // Filter by display mode. `isExpected` is encoded by color
        // — we check the source move set rather than re-derive from
        // color so the filter logic remains decoupled from the
        // palette.
        switch arrowDisplayMode {
        case .all:
            return arrows
        case .none:
            return []
        case .expectedOnly:
            return arrows.filter { arrow in
                let move = ChessMove(
                    fromRow: arrow.fromRow,
                    fromCol: arrow.fromCol,
                    toRow: arrow.toRow,
                    toCol: arrow.toCol,
                    promotion: arrow.promotion
                )
                return acceptable.contains(move)
            }
        case .othersOnly:
            return arrows.filter { arrow in
                let move = ChessMove(
                    fromRow: arrow.fromRow,
                    fromCol: arrow.fromCol,
                    toRow: arrow.toRow,
                    toCol: arrow.toCol,
                    promotion: arrow.promotion
                )
                return !acceptable.contains(move)
            }
        }
    }

    // MARK: - Arrow palette

    private static let colorExpectedRank1   = Color.green
    private static let colorExpectedInTop5  = Color(red: 0.62, green: 0.88, blue: 0.24)  // yellow-green / chartreuse
    private static let colorExpectedMissed  = Color.red
    private static let colorTop1NotExpected = Color.blue
    private static let colorOtherTop5       = Color(red: 0.82, green: 0.71, blue: 0.55)  // tan

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

// MARK: - Arrow display mode

/// Four-state cycle for the board's arrow overlay, advanced by
/// tapping the board. Lets the user dial down visual noise on
/// probes where the top-5 fan-out clutters the position.
enum ArrowDisplayMode: Int, CaseIterable, Sendable {
    /// All five top-policy arrows + the synthetic expected arrow
    /// when the fixture's correct move isn't in the top 5.
    case all = 0
    /// No arrows; clean board for studying piece placement.
    case none = 1
    /// Only the expected move's arrow (synthetic or one of the
    /// top-5 colored arrows). Isolates "where the right answer is."
    case expectedOnly = 2
    /// Everything except the expected move. Isolates "what the
    /// network is actually picking instead."
    case othersOnly = 3

    func next() -> ArrowDisplayMode {
        ArrowDisplayMode(rawValue: (rawValue + 1) % ArrowDisplayMode.allCases.count) ?? .all
    }

    var label: String {
        switch self {
        case .all: return "all"
        case .none: return "none"
        case .expectedOnly: return "expected only"
        case .othersOnly: return "others only"
        }
    }
}
