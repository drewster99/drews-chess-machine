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
        ZStack {
            Text("Waiting for \(game.id)…")
                .foregroundStyle(.secondary)
                .frame(maxWidth: .infinity, minHeight: 120)
                .shown(game.ourColor == nil)
            ForEach(game.ourColor.map { [$0] } ?? [], id: \.self) { ourColor in
                LichessBotGameTileContent(controller: controller, game: game, ourColor: ourColor, onOpen: onOpen, onDismiss: onDismiss)
            }
        }
    }
}

/// A tile's content once DCM's color is known.
struct LichessBotGameTileContent: View {
    let controller: LichessBotController
    let game: LichessBotLiveGame
    let ourColor: PieceColor
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
                HStack(alignment: .firstTextBaseline) {
                    PieceColorDisc(color: ourColor, diameter: 10)
                    Text("DCM")
                        .font(.caption2.weight(.semibold))
                        .foregroundStyle(.secondary)
                    LichessBotClockView(
                        milliseconds: ourColor == .white ? game.whiteClockMilliseconds : game.blackClockMilliseconds,
                        receivedAt: game.clocksReceivedAt,
                        isRunning: clocksRun && game.sideToMove == ourColor,
                        isOurs: true
                    )
                    Spacer()
                    Text("\(game.plies.count) plies")
                        .font(.system(.caption, design: .monospaced))
                        .foregroundStyle(.secondary)
                    Spacer()
                    PieceColorDisc(color: ourColor == .white ? .black : .white, diameter: 10)
                    Text("Opp.")
                        .font(.caption2.weight(.semibold))
                        .foregroundStyle(.secondary)
                    LichessBotClockView(
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
                .fill(game.isFinished ? Color.gray.opacity(0.18) : Color.clear)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .strokeBorder(game.isFinished ? Color.gray.opacity(0.3) : Color.green.opacity(0.6), lineWidth: game.isFinished ? 1 : 2)
        )
        .contentShape(Rectangle())
        .onTapGesture(perform: onOpen)
        .help("Open this game in its own window")
        .accessibilityElement(children: .combine)
        .accessibilityAddTraits(.isButton)
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

    private var resultColor: Color {
        switch result {
        case .won: return .green
        case .lost: return .red
        case .drew, .noResult: return .secondary
        }
    }
}
