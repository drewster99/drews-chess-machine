import SwiftUI

/// The Challenge Log's table (challenge-log plan §3.9): one row per
/// challenge, or per send that created none, already filtered and sorted.
/// Every cell's text and glyph come from `LichessBotChallengeLogStyle`.
struct LichessBotChallengeLogTable: View {
    let controller: LichessBotController
    let rows: [LichessBotChallengeLogRow]
    @Binding var sortOrder: [KeyPathComparator<LichessBotChallengeLogRow>]

    var body: some View {
        Table(rows, sortOrder: $sortOrder) {
            TableColumn("When", value: \.at) { row in
                Text(row.at.formatted(date: .abbreviated, time: .standard))
                    .font(.system(.callout, design: .monospaced))
            }
            .width(min: 200, ideal: 220)
            TableColumn("Direction") { row in
                Text(LichessBotChallengeLogStyle.directionLabel(row.direction))
                    .font(.system(.callout, design: .monospaced))
                    .foregroundStyle(row.direction == nil ? Color.secondary : Color.primary)
                    .help(LichessBotChallengeLogStyle.directionText(row.direction))
                    .accessibilityLabel(LichessBotChallengeLogStyle.directionText(row.direction))
            }
            .width(60)
            TableColumn("Opponent", value: \.opponentSortKey) { row in
                LichessBotChallengeLogOpponentCell(controller: controller, opponent: row.opponent)
            }
            .width(min: 120, ideal: 170)
            TableColumn("Rating", value: \.ratingSortKey) { row in
                Text(LichessBotChallengeLogStyle.ratingText(row.opponent))
                    .font(.system(.callout, design: .monospaced))
            }
            .width(min: 44, ideal: 50)
            TableColumn("Terms") { row in
                Text(LichessBotChallengeLogStyle.termsText(row.terms))
                    .foregroundStyle(row.terms == nil ? Color.secondary : Color.primary)
            }
            .width(min: 110, ideal: 140)
            TableColumn("Sender / Decision") { row in
                Text(LichessBotChallengeLogStyle.initiativeText(row.initiative))
                    .help(LichessBotChallengeLogStyle.initiativeHelp(row.initiative))
            }
            .width(min: 120, ideal: 200)
            TableColumn("State") { row in
                Text(LichessBotChallengeLogStyle.stateText(row.state))
                    .help(LichessBotChallengeLogStyle.stateHelp(row))
            }
            .width(min: 130, ideal: 210)
            TableColumn("Game") { row in
                LichessBotChallengeLogGameCell(gameID: row.gameID)
            }
            .width(min: 70, ideal: 90)
            TableColumn("Credits") { row in
                Text(LichessBotChallengeLogStyle.creditsText(row.credits))
                    .font(.system(.callout, design: .monospaced))
                    .help(LichessBotChallengeLogStyle.creditsHelp(row.credits))
            }
            .width(50)
            TableColumn("Source") { row in
                Text(LichessBotChallengeLogStyle.sourceText(row.source))
                    .foregroundStyle(row.isReconstructed ? Color.secondary : Color.primary)
                    .help(LichessBotChallengeLogStyle.sourceHelp(row.source))
            }
            .width(min: 90, ideal: 150)
        }
    }
}
