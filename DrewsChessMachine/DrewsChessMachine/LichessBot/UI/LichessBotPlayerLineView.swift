import SwiftUI

/// A player's name, title and rating, with a clock. For the opponent it
/// also shows DCM's head-to-head record against them (plan §14.3a).
struct LichessBotPlayerLineView: View {
    let controller: LichessBotController
    let player: LichessBotLiveGame.Player?
    /// The side this player has.
    let pieceColor: PieceColor
    let isOurs: Bool
    let headToHead: (wins: Int, draws: Int, losses: Int)?
    let clockMilliseconds: Int?
    let clockReceivedAt: Date?
    let clockRunning: Bool
    /// Material in the position on the board (the browsed ply, not
    /// necessarily the live one).
    let material: MaterialCount

    var body: some View {
        HStack(spacing: 8) {
            PieceColorDisc(color: pieceColor, diameter: 14)
            Text(pieceColor == .white ? "White" : "Black")
                .font(.callout)
                .foregroundStyle(.secondary)
                .frame(width: 44, alignment: .leading)
            Text(player?.title ?? "")
                .font(.callout.weight(.semibold))
                .foregroundStyle(.orange)
                .shown(player?.title != nil)
            // The opponent can be starred; our own line and Lichess's AI
            // (no account) can't.
            LichessBotFavoriteStar(controller: controller, userID: isOurs ? nil : player?.id)
            Text(player?.name ?? "—")
                .font(.body.weight(.medium))
                .lineLimit(1)
            Text(player?.rating.map { String(format: "%4d", $0) } ?? "")
                .font(.system(.callout, design: .monospaced))
                .foregroundStyle(.secondary)
            Text(headToHeadText)
                .font(.system(.caption, design: .monospaced))
                .foregroundStyle(.secondary)
                .help("DCM's record against this opponent: wins–draws–losses")
                .shown(headToHead != nil)
            Spacer(minLength: 8)
            LichessBotMaterialText(points: material.points(for: pieceColor), lead: material.lead(for: pieceColor))
            LichessBotClockView(milliseconds: clockMilliseconds, receivedAt: clockReceivedAt, isRunning: clockRunning, isOurs: isOurs)
        }
    }

    private var headToHeadText: String {
        guard let headToHead else { return "" }
        return "vs: \(headToHead.wins)–\(headToHead.draws)–\(headToHead.losses)"
    }
}

/// A side's material in points, with its lead when ahead ("39 +3"). The
/// lead column keeps its width when empty so the two players' lines align.
struct LichessBotMaterialText: View {
    let points: Int
    let lead: Int

    var body: some View {
        HStack(spacing: 4) {
            Text(String(format: "%2d", points))
                .foregroundStyle(.secondary)
            Text(lead > 0 ? String(format: "+%d", lead) : "")
                .foregroundStyle(.green)
                .frame(width: 30, alignment: .leading)
        }
        .font(.system(.callout, design: .monospaced))
        .help("Material in points: pawn 1, knight 3, bishop 3, rook 5, queen 9")
    }
}
