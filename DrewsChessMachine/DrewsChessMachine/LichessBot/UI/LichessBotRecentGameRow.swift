import SwiftUI

/// One row of the recent games list (a `GridRow` of its grid).
struct LichessBotRecentGameRow: View {
    let controller: LichessBotController
    let row: LichessBotGameSummary

    var body: some View {
        GridRow {
            LichessBotResultChip(ourScore: row.ourScore)
            PieceColorDisc(color: row.ourColor == .white ? .white : .black, diameter: 10)
                .help(row.ourColor == .white ? "DCM played White" : "DCM played Black")
            LichessBotGameOriginGlyph(display: controller.originsByGameID[row.gameID])
                .font(LichessBotStatsStyle.noteFont)
            HStack(spacing: 4) {
                LichessBotFavoriteStar(controller: controller, userID: row.opponentID)
                Text(row.opponentName ?? "?")
                    .lineLimit(1)
            }
            LichessBotNoteText(text: Self.kindText(row))
            Text(row.opponentRating.map { "\($0)" } ?? "")
                .font(LichessBotStatsStyle.numberFont)
                .gridColumnAlignment(.trailing)
            LichessBotNoteText(text: "\(row.speed)\(row.rated ? " · rated" : "")")
            LichessBotNoteText(text: row.createdAt.formatted(.relative(presentation: .named)))
        }
        .font(LichessBotStatsStyle.rowFont)
    }

    private static func kindText(_ row: LichessBotGameSummary) -> String {
        switch row.opponentKind {
        case .bot: return "bot"
        case .human: return row.opponentTitle ?? "human"
        case .lichessAI: return "Lichess AI"
        }
    }
}
