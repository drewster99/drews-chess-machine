import Foundation

/// Each number column's width in characters in the period table
/// (`LichessBotRecordPeriodGrid`, whose font is monospaced), taken over
/// every table it is given — the All, Rated and Casual filters' — so
/// switching filters changes the numbers and never the columns.
struct LichessBotRecordPeriodColumnWidths: Sendable, Equatable {
    let games: Int
    /// Digits in the largest win, draw or loss count, so every count pads to
    /// one width and the dashes line up down each column.
    let count: Int
    let score: Int
    let performance: Int
    let opponentAverage: Int

    init(tables: [LichessBotPeriodValues<LichessBotPeriodStatistics>]) {
        let rows = tables.flatMap { table in LichessBotStatsPeriod.allCases.map { table[$0] } }
        func widest(_ text: (LichessBotPeriodStatistics) -> String) -> Int {
            rows.map { text($0).count }.max() ?? 0
        }
        games = widest { "\($0.record.all.scored)" }
        count = LichessBotStatsFormat.countWidth(rows.flatMap { row -> [LichessBotResultTally] in
            let record = row.record
            return [record.all, record.versusBots, record.versusHumans, record.asWhite, record.asBlack]
        })
        score = widest { LichessBotStatsFormat.score($0.record.all.score) }
        performance = widest { LichessBotStatsFormat.estimate($0.performance) }
        opponentAverage = widest { LichessBotStatsFormat.average($0.opponentAverage) }
    }
}
