import SwiftUI

/// A compact card for one game in the live grid (plan §14.3a): small board,
/// players with clocks, W/D/L, and a result badge once finished. Clicking it
/// opens the game in its own window; the grid keeps no selection. A
/// finished game's tile is grayed as a whole, so live games stand out.
struct LichessBotGameTileView: View {
    let controller: LichessBotController
    let game: LichessBotLiveGame
    let onOpen: () -> Void
    let onDismiss: () -> Void

    var body: some View {
        let phase = LichessBotGridOrdering.phase(finishedAt: game.finishedAt, now: controller.gridClock)
        ZStack {
            Text("Waiting for \(game.id)…")
                .foregroundStyle(.secondary)
                .frame(maxWidth: .infinity, minHeight: 120)
                .shown(game.ourColor == nil)
            ForEach(game.ourColor.map { [$0] } ?? [], id: \.self) { ourColor in
                LichessBotGameTileContent(controller: controller, game: game, ourColor: ourColor, phase: phase, onOpen: onOpen, onDismiss: onDismiss)
            }
        }
    }
}

/// A tile's content once DCM's color is known.
struct LichessBotGameTileContent: View {
    let controller: LichessBotController
    let game: LichessBotLiveGame
    let ourColor: PieceColor
    let phase: LichessBotGridOrdering.Phase
    let onOpen: () -> Void
    let onDismiss: () -> Void

    var body: some View {
        let clocksRun = !game.isFinished
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                PieceColorDisc(color: ourColor == .white ? .black : .white)
                LichessBotFavoriteStar(controller: controller, userID: game.opponent?.id)
                Text(game.opponent?.name ?? game.id)
                    .font(.callout.weight(.semibold))
                    .lineLimit(1)
                Text(game.opponent?.rating.map { String(format: "%4d", $0) } ?? "")
                    .font(.system(.caption, design: .monospaced))
                    .foregroundStyle(.secondary)
                Spacer()
                Text(resultText)
                    .font(.system(.callout, design: .monospaced).weight(.bold))
                    .foregroundStyle(resultColor)
                    .shown(game.isFinished)
                Button(action: onDismiss) {
                    Image(systemName: "xmark.circle.fill")
                        .foregroundStyle(.secondary)
                }
                .buttonStyle(.plain)
                .help("Remove from the grid")
                .shown(game.isFinished)
            }
            // Board, clocks and bar dim together for a finished game; the
            // header's result stays at full strength.
            VStack(alignment: .leading, spacing: 4) {
                LichessBotBoardView(game: game, plyCount: game.plies.count)
                // Each side's label above its clock, and the ply count in
                // two lines between them: a tile is too narrow to keep a
                // clock beside its label without the time wrapping.
                HStack(alignment: .lastTextBaseline) {
                    LichessBotTileClockColumn(
                        label: "DCM",
                        pieceColor: ourColor,
                        alignment: .leading,
                        milliseconds: ourColor == .white ? game.whiteClockMilliseconds : game.blackClockMilliseconds,
                        receivedAt: game.clocksReceivedAt,
                        isRunning: clocksRun && game.sideToMove == ourColor,
                        isOurs: true
                    )
                    Spacer(minLength: 4)
                    VStack(spacing: 0) {
                        Text("\(game.plies.count)")
                        Text("plies")
                    }
                    .font(.system(.caption, design: .monospaced))
                    .foregroundStyle(.secondary)
                    Spacer(minLength: 4)
                    LichessBotTileClockColumn(
                        label: "Opp.",
                        pieceColor: ourColor == .white ? .black : .white,
                        alignment: .trailing,
                        milliseconds: ourColor == .white ? game.blackClockMilliseconds : game.whiteClockMilliseconds,
                        receivedAt: game.clocksReceivedAt,
                        isRunning: clocksRun && game.sideToMove != ourColor,
                        isOurs: false
                    )
                }
                WLDBar(wins: Double(game.latestDecision?.win ?? 0), draws: Double(game.latestDecision?.draw ?? 0), losses: Double(game.latestDecision?.loss ?? 0))
                    .frame(height: 6)
            }
            .saturation(game.isFinished ? 0 : 1)
            .opacity(game.isFinished ? 0.55 : 1)
        }
        .padding(8)
        .background(
            RoundedRectangle(cornerRadius: 8)
                .fill(backgroundColor)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .strokeBorder(borderColor, lineWidth: phase == .finished ? 1 : 2)
        )
        // The phase changes when the game ends and again when its result
        // highlight ends; both fade rather than snap.
        .animation(.default, value: phase)
        .contentShape(Rectangle())
        .onTapGesture(perform: onOpen)
        .help("Open this game in its own window")
        .accessibilityElement(children: .combine)
        .accessibilityAddTraits(.isButton)
        .accessibilityAction(.default, onOpen)
    }

    private enum Result {
        case won, lost, drew, noResult
    }

    /// DCM's result.
    private var result: Result {
        switch game.ourScore {
        case .some(1): return .won
        case .some(0): return .lost
        case .some: return .drew
        case .none: return .noResult
        }
    }

    private var resultText: String {
        switch result {
        case .won: return "DCM won"
        case .lost: return "DCM lost"
        case .drew: return "Draw"
        case .noResult: return game.status
        }
    }

    /// The result's tint while a just-finished game is highlighted: green
    /// for a win, red for a loss, gray for a draw or no result.
    private var resultHighlightColor: Color {
        switch result {
        case .won: return .green
        case .lost: return .red
        case .drew, .noResult: return .gray
        }
    }

    private var backgroundColor: Color {
        switch phase {
        case .live: return .clear
        case .justFinished: return resultHighlightColor.opacity(0.22)
        case .finished: return Color.gray.opacity(0.18)
        }
    }

    private var borderColor: Color {
        switch phase {
        case .live: return Color.green.opacity(0.6)
        case .justFinished: return resultHighlightColor.opacity(0.8)
        case .finished: return Color.gray.opacity(0.3)
        }
    }

    private var resultColor: Color {
        switch result {
        case .won: return .green
        case .lost: return .red
        case .drew, .noResult: return .secondary
        }
    }
}

/// One side's clock in a tile's footer: its color disc and label on the
/// first line, the clock on the second.
struct LichessBotTileClockColumn: View {
    let label: String
    let pieceColor: PieceColor
    /// Leading for DCM's column, trailing for the opponent's, so the two
    /// clocks sit at the tile's edges.
    let alignment: HorizontalAlignment
    let milliseconds: Int?
    let receivedAt: Date?
    let isRunning: Bool
    let isOurs: Bool

    var body: some View {
        VStack(alignment: alignment, spacing: 0) {
            // Inset like the time within its running highlight, so label and
            // time line up and the highlight stays inside the column.
            HStack(spacing: 4) {
                PieceColorDisc(color: pieceColor, diameter: 10)
                Text(label)
                    .font(.caption2.weight(.semibold))
                    .foregroundStyle(.secondary)
            }
            .padding(.horizontal, LichessBotClockView.horizontalPadding)
            LichessBotClockView(milliseconds: milliseconds, receivedAt: receivedAt, isRunning: isRunning, isOurs: isOurs)
        }
    }
}
