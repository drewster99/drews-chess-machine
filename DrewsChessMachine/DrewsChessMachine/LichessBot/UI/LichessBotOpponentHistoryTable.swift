import SwiftUI

/// Every game DCM has played one opponent, in a sortable table: result,
/// when, DCM's color, DCM's and their starting rating, Lichess's rating
/// change, the model that played, the end and length, and a link to the game
/// on Lichess. Rows are built off `body` into state when the games change, so
/// the table never attributes models during layout.
struct LichessBotOpponentHistoryTable: View {
    let controller: LichessBotController
    let games: [LichessBotGameSummary]

    @State private var rows: [LichessBotOpponentGameRow] = []
    @State private var sortOrder = [KeyPathComparator(\LichessBotOpponentGameRow.createdAt, order: .reverse)]

    var body: some View {
        GroupBox("Games vs them (\(games.count))") {
            VStack(alignment: .leading, spacing: 6) {
                Text("No games filed against this opponent yet.")
                    .foregroundStyle(.secondary)
                    .shown(games.isEmpty)
                Table(rows, sortOrder: $sortOrder) {
                    TableColumn("") { row in
                        LichessBotResultChip(ourScore: row.summary.ourScore)
                    }
                    .width(26)
                    TableColumn("When", value: \LichessBotOpponentGameRow.createdAt) { row in
                        Text(row.summary.createdAt.formatted(date: .abbreviated, time: .shortened))
                            .font(.system(.callout, design: .monospaced))
                    }
                    .width(min: 130, ideal: 150)
                    TableColumn("Color") { row in
                        PieceColorDisc(color: row.summary.ourColor == .white ? .white : .black, diameter: 10)
                    }
                    .width(40)
                    // Grouped: a column builder block takes at most ten
                    // columns. The key paths name their root, which a group
                    // can't infer.
                    Group {
                        TableColumn("DCM", value: \LichessBotOpponentGameRow.ourRatingSortKey) { row in
                            Text(row.summary.ourRatingBefore.map { "\($0)" } ?? "")
                                .font(.system(.callout, design: .monospaced))
                        }
                        .width(min: 48, ideal: 58)
                        TableColumn("Them", value: \LichessBotOpponentGameRow.opponentRatingSortKey) { row in
                            Text(row.summary.opponentRating.map { "\($0)" } ?? "")
                                .font(.system(.callout, design: .monospaced))
                        }
                        .width(min: 48, ideal: 58)
                        TableColumn("Rating Δ", value: \LichessBotOpponentGameRow.ratingChangeSortKey) { row in
                            LichessBotGameRatingChangeText(change: row.ratingChange)
                                .font(LichessBotStatsStyle.numberFont)
                                .frame(maxWidth: .infinity, alignment: .trailing)
                        }
                        .width(min: 55, ideal: 65)
                    }
                    TableColumn("Model", value: \.modelLabel) { row in
                        Text(row.modelLabel)
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                    }
                    .width(min: 120, ideal: 200)
                    TableColumn("Speed", value: \.speed) { row in
                        Text(row.summary.speed + (row.summary.rated ? " · rated" : ""))
                    }
                    .width(min: 90, ideal: 110)
                    TableColumn("End", value: \.status) { row in
                        Text(row.summary.status)
                            .foregroundStyle(.secondary)
                    }
                    .width(min: 70, ideal: 90)
                    TableColumn("Plies", value: \.plies) { row in
                        Text("\(row.summary.plies)")
                            .font(.system(.callout, design: .monospaced))
                    }
                    .width(50)
                    TableColumn("Game") { row in
                        Button(row.summary.gameID) {
                            LichessBotLinks.openGame(row.summary.gameID)
                        }
                        .buttonStyle(.link)
                        .help("Open on lichess.org")
                    }
                    .width(min: 80, ideal: 90)
                }
                .frame(minHeight: 260)
                .shown(!games.isEmpty)
            }
            .padding(6)
        }
        .onChange(of: games, initial: true) {
            Task { @MainActor in rebuildRows() }
        }
        .onChange(of: sortOrder) {
            Task { @MainActor in rows.sort(using: sortOrder) }
        }
    }

    private func rebuildRows() {
        let labels = Dictionary(uniqueKeysWithValues: LichessBotStatsModelChoice.choices(from: games).map { ($0.key, "\($0.modelID) · \($0.checkpointLabel)") })
        rows = games.map { LichessBotOpponentGameRow(summary: $0, modelLabels: labels) }.sorted(using: sortOrder)
    }
}

/// A game-history row with its sort keys and the model that played it,
/// resolved once at build time.
struct LichessBotOpponentGameRow: Identifiable {
    let summary: LichessBotGameSummary
    let modelLabel: String
    let ratingChange: LichessBotGameRatingChange

    var id: String { summary.gameID }
    var createdAt: Date { summary.createdAt }
    var speed: String { summary.speed }
    var status: String { summary.status }
    var plies: Int { summary.plies }
    /// A game with no starting rating sorts below every one that has one.
    var ourRatingSortKey: Int { summary.ourRatingBefore ?? Int.min }
    var opponentRatingSortKey: Int { summary.opponentRating ?? Int.min }
    var ratingChangeSortKey: Int {
        switch ratingChange {
        case .change(let diff): return diff
        default: return Int.min
        }
    }

    init(summary: LichessBotGameSummary, modelLabels: [LichessBotModelKey: String]) {
        self.summary = summary
        ratingChange = LichessBotGameRatingChange(summary)
        if let key = LichessBotModelAttribution(summary.facts?.moves).model?.key {
            modelLabel = modelLabels[key] ?? "unknown"
        } else {
            modelLabel = "—"
        }
    }
}
