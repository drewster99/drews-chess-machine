import SwiftUI

/// One opponent's row of the most-played table (a `GridRow`).
struct LichessBotOpponentRecordRowView: View {
    let row: LichessBotOpponentRecordRow
    let countWidth: Int

    var body: some View {
        GridRow {
            LichessBotRowLabel(text: row.name)
            Text(Self.kindText(row.kind))
                .font(LichessBotStatsStyle.noteFont)
                .foregroundStyle(LichessBotStatsStyle.neutral)
            Text("\(row.tally.games)")
            LichessBotTallyText(tally: row.tally, countWidth: countWidth)
            Text(LichessBotStatsFormat.score(row.tally.score))
            Text(row.lastPlayedAt.formatted(date: .numeric, time: .omitted))
        }
        .font(LichessBotStatsStyle.numberFont)
        .lineLimit(1)
        .fixedSize()
    }

    private static func kindText(_ kind: LichessBotOpponentKind) -> String {
        switch kind {
        case .bot: return "bot"
        case .human: return "human"
        case .lichessAI: return "Lichess AI"
        }
    }
}
