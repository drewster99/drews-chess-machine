import SwiftUI

/// DCM's record today, this week and in total, from the saved game records:
/// games, W–D–L, score, and splits by opponent kind and by color.
struct LichessBotRecordCard: View {
    let controller: LichessBotController
    /// The card's content height, set by dragging its bottom edge and
    /// remembered across launches (a viewing preference, not a bot setting).
    @AppStorage("lichessBot.overview.recordCardHeight") private var contentHeight: Double = 240

    /// Short enough to keep the card compact, never so short the record
    /// table's rows are cut off.
    private static let contentHeightRange: ClosedRange<Double> = 160...1600

    var body: some View {
        GroupBox("Record") {
            VStack(spacing: 4) {
                // Applied here rather than passed in, so dragging the handle
                // doesn't recompute the record on every frame.
                LichessBotRecordCardContent(controller: controller)
                    .frame(height: min(max(contentHeight, Self.contentHeightRange.lowerBound), Self.contentHeightRange.upperBound), alignment: .top)
                LichessBotHeightResizeHandle(height: $contentHeight, range: Self.contentHeightRange)
            }
        }
    }
}

/// The record table beside the recent games; the recent games list scrolls
/// within whatever height the card gives it.
struct LichessBotRecordCardContent: View {
    let controller: LichessBotController

    var body: some View {
        ZStack(alignment: .leading) {
            Text("Loading the game records…")
                .foregroundStyle(.secondary)
                .shown(controller.index == nil)
            ForEach(controller.index.map { [$0] } ?? [], id: \.recordCount) { index in
                // Re-evaluated each minute, so "Today" and "This week" roll
                // over at midnight and the relative times stay current.
                TimelineView(.everyMinute) { context in
                    HStack(alignment: .top, spacing: 28) {
                        LichessBotRecordGrid(result: Result {
                            try LichessBotRecordSummary.compute(rows: index.rows, now: context.date, calendar: .current)
                        })
                        .fixedSize()
                        Divider()
                        LichessBotRecentGamesList(controller: controller, rows: Array(index.rows.sorted { $0.createdAt > $1.createdAt }.prefix(LichessBotRecentGamesList.maximumRows)))
                    }
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

/// The record table, or why it can't be shown.
struct LichessBotRecordGrid: View {
    let result: Result<LichessBotRecordSummary.Records, Error>

    var body: some View {
        switch result {
        case .failure(let error):
            Text(error.localizedDescription)
                .foregroundStyle(.red)
        case .success(let records):
            let countWidth = Self.countWidth(records)
            Grid(alignment: .trailing, horizontalSpacing: 22, verticalSpacing: 6) {
                GridRow {
                    Text("")
                    Text("Games")
                    Text("W–D–L")
                    Text("Score")
                    Text("vs bots")
                    Text("vs humans")
                    Text("as White")
                    Text("as Black")
                }
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .fixedSize()
                ForEach(LichessBotRecordSummary.Period.allCases, id: \.self) { period in
                    let record = records[period]
                    GridRow {
                        Text(period.rawValue)
                            .font(.callout.weight(.medium))
                            .gridColumnAlignment(.leading)
                        Text("\(record.all.games)")
                        LichessBotTallyText(tally: record.all, countWidth: countWidth)
                        Text(record.all.score.map { String(format: "%.1f%%", 100 * $0) } ?? "–")
                        LichessBotTallyText(tally: record.versusBots, countWidth: countWidth)
                        LichessBotTallyText(tally: record.versusHumans, countWidth: countWidth)
                        LichessBotTallyText(tally: record.asWhite, countWidth: countWidth)
                        LichessBotTallyText(tally: record.asBlack, countWidth: countWidth)
                    }
                    .font(.system(.callout, design: .monospaced))
                    .lineLimit(1)
                    .fixedSize()
                }
            }
        }
    }

    /// Digits in the largest win, draw or loss count anywhere in the table,
    /// so every count pads to the same width and the dashes line up down
    /// each column.
    static func countWidth(_ records: LichessBotRecordSummary.Records) -> Int {
        let largest = LichessBotRecordSummary.Period.allCases
            .map { records[$0] }
            .flatMap { [$0.all, $0.versusBots, $0.versusHumans, $0.asWhite, $0.asBlack] }
            .flatMap { [$0.wins, $0.draws, $0.losses] }
            .reduce(0, max)
        return String(largest).count
    }
}

/// A W–D–L tally: counts in the primary color, each left-padded to
/// `countWidth` digits (spaces are digit-wide in the monospaced font the
/// grid sets), and faint dashes between them.
struct LichessBotTallyText: View {
    let tally: LichessBotResultTally
    let countWidth: Int

    var body: some View {
        let dash = Text("–").foregroundStyle(.tertiary)
        Text("\(padded(tally.wins))\(dash)\(padded(tally.draws))\(dash)\(padded(tally.losses))")
    }

    private func padded(_ count: Int) -> String {
        String(repeating: " ", count: max(0, countWidth - String(count).count)) + String(count)
    }
}

/// DCM's most recent filed games, newest first: result chip, color, how the
/// game began, the opponent and their kind and rating, the game's speed, and
/// when. It
/// scrolls within the card's height; "More…" opens every game in a
/// sortable window.
struct LichessBotRecentGamesList: View {
    /// A bound on what the card lists; the window lists every game.
    static let maximumRows = 200
    let controller: LichessBotController
    let rows: [LichessBotGameSummary]

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text("Recent games")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)
                Spacer()
                Button("More…") {
                    LichessBotAllGamesWindowController.open(controller: controller)
                }
                .font(.caption)
            }
            Text("No games filed yet")
                .foregroundStyle(.secondary)
                .shown(rows.isEmpty)
            ScrollView {
                Grid(alignment: .leading, horizontalSpacing: 10, verticalSpacing: 4) {
                    ForEach(rows, id: \.gameID) { row in
                        GridRow {
                            LichessBotResultChip(ourScore: row.ourScore)
                            PieceColorDisc(color: row.ourColor == .white ? .white : .black, diameter: 10)
                                .help(row.ourColor == .white ? "DCM played White" : "DCM played Black")
                            LichessBotGameOriginGlyph(display: controller.originsByGameID[row.gameID])
                                .font(.caption)
                            HStack(spacing: 4) {
                                LichessBotFavoriteStar(controller: controller, userID: row.opponentID)
                                Text(row.opponentName ?? "?")
                                    .lineLimit(1)
                            }
                            Text(Self.kindText(row))
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            Text(row.opponentRating.map { "\($0)" } ?? "")
                                .font(.system(.callout, design: .monospaced))
                                .gridColumnAlignment(.trailing)
                            Text("\(row.speed)\(row.rated ? " · rated" : "")")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            Text(row.createdAt.formatted(.relative(presentation: .named)))
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                        .font(.callout)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
    }

    private static func kindText(_ row: LichessBotGameSummary) -> String {
        switch row.opponentKind {
        case .bot: return "bot"
        case .human: return row.opponentTitle ?? "human"
        case .lichessAI: return "Lichess AI"
        }
    }
}

/// W / D / L for DCM, colored; "–" for a game without a result.
struct LichessBotResultChip: View {
    let ourScore: Double?

    var body: some View {
        Text(letter)
            .font(.system(.caption, design: .monospaced).weight(.bold))
            .foregroundStyle(.white)
            .frame(width: 20, height: 18)
            .background(RoundedRectangle(cornerRadius: 4).fill(color))
            .help(helpText)
    }

    private var letter: String {
        switch ourScore {
        case .some(1): return "W"
        case .some(0): return "L"
        case .some: return "D"
        case .none: return "–"
        }
    }

    private var color: Color {
        switch ourScore {
        case .some(1): return .green
        case .some(0): return .red
        case .some: return .gray
        case .none: return .secondary.opacity(0.4)
        }
    }

    private var helpText: String {
        switch ourScore {
        case .some(1): return "DCM won"
        case .some(0): return "DCM lost"
        case .some: return "Draw"
        case .none: return "No result (aborted)"
        }
    }
}
