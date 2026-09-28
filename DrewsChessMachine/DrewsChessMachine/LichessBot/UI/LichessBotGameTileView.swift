import SwiftUI

/// A compact card for one game in the live grid (plan §14.3a): small board,
/// players with clocks, W/D/L, and a result badge once finished. Clicking it
/// focuses the game.
struct LichessBotGameTileView: View {
    let game: LichessBotLiveGame
    let isFocused: Bool
    let onFocus: () -> Void
    let onDismiss: () -> Void

    var body: some View {
        ZStack {
            Text("Waiting for \(game.id)…")
                .foregroundStyle(.secondary)
                .frame(maxWidth: .infinity, minHeight: 120)
                .shown(game.ourColor == nil)
            ForEach(game.ourColor.map { [$0] } ?? [], id: \.self) { ourColor in
                LichessBotGameTileContent(game: game, ourColor: ourColor, isFocused: isFocused, onFocus: onFocus, onDismiss: onDismiss)
            }
        }
    }
}

/// A tile's content once DCM's color is known.
struct LichessBotGameTileContent: View {
    let game: LichessBotLiveGame
    let ourColor: PieceColor
    let isFocused: Bool
    let onFocus: () -> Void
    let onDismiss: () -> Void

    var body: some View {
        let clocksRun = !game.isFinished
        VStack(alignment: .leading, spacing: 4) {
            HStack {
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
            LichessBotBoardView(game: game, plyCount: game.plies.count)
                .opacity(game.isFinished ? 0.75 : 1)
            HStack {
                LichessBotClockView(
                    milliseconds: ourColor == .white ? game.whiteClockMilliseconds : game.blackClockMilliseconds,
                    receivedAt: game.clocksReceivedAt,
                    isRunning: clocksRun && game.sideToMove == ourColor,
                    isOurs: true
                )
                Spacer()
                Text("ply \(game.plies.count)")
                    .font(.system(.caption, design: .monospaced))
                    .foregroundStyle(.secondary)
                Spacer()
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
        .padding(8)
        .background(
            RoundedRectangle(cornerRadius: 8)
                .strokeBorder(isFocused ? Color.accentColor : Color.gray.opacity(0.3), lineWidth: isFocused ? 2 : 1)
        )
        .contentShape(Rectangle())
        .onTapGesture(perform: onFocus)
        .accessibilityElement(children: .combine)
        .accessibilityAddTraits(.isButton)
    }

    /// DCM's result: W, D or L.
    private var resultText: String {
        switch game.winner {
        case "white": return ourColor == .white ? "W" : "L"
        case "black": return ourColor == .black ? "W" : "L"
        default: return game.status == "aborted" ? "–" : "D"
        }
    }

    private var resultColor: Color {
        switch resultText {
        case "W": return .green
        case "L": return .red
        default: return .secondary
        }
    }
}
