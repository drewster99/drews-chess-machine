import SwiftUI

/// One opponent's row of the most-played table (a `GridRow`).
struct LichessBotOpponentRecordRowView: View {
    let controller: LichessBotController
    let row: LichessBotOpponentRecordRow
    let countWidth: Int

    var body: some View {
        GridRow {
            Button {
                LichessBotOpponentWindowLauncher.open(opponentID: row.id, name: row.name, controller: controller)
            } label: {
                LichessBotRowLabel(text: row.name)
            }
            .buttonStyle(.link)
            .gridColumnAlignment(.leading)
            .help("Open \(row.name)'s breakdown")
            Text(Self.kindText(row.kind))
                .font(LichessBotStatsStyle.noteFont)
                .foregroundStyle(LichessBotStatsStyle.neutral)
            Text("\(row.tally.games)")
            LichessBotTallyText(tally: row.tally, countWidth: countWidth)
            Text(LichessBotStatsFormat.score(row.tally.score))
            Text(LichessBotStatsFormat.ratingChange(row.ratingChange))
                .help(LichessBotStatsFormat.ratingChangeHelp(row.ratingChange))
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
