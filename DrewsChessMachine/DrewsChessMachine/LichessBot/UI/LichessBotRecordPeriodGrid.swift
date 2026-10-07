import SwiftUI

/// Item 1's table (§3.3): one row per period, last hour to all time —
/// scored games, W–D–L, score, performance rating, average opponent rating,
/// rating change, and the opponent-kind and color splits.
struct LichessBotRecordPeriodGrid: View {
    let rows: LichessBotPeriodValues<LichessBotPeriodStatistics>

    static let titles = ["", "Games", "W–D–L", "Score", "Perf", "Opp avg", "Rating ±", "vs bots", "vs humans", "as White", "as Black"]

    var body: some View {
        let countWidth = Self.countWidth(rows)
        Grid(alignment: .trailing, horizontalSpacing: LichessBotStatsStyle.columnSpacing, verticalSpacing: LichessBotStatsStyle.rowSpacing) {
            LichessBotStatsHeaderRow(titles: Self.titles)
            ForEach(LichessBotStatsPeriod.allCases, id: \.self) { period in
                LichessBotRecordPeriodRow(period: period, row: rows[period], countWidth: countWidth)
            }
        }
    }

    /// Digits in the largest win, draw or loss count anywhere in the table,
    /// so every count pads to one width and the dashes line up down each
    /// column.
    static func countWidth(_ rows: LichessBotPeriodValues<LichessBotPeriodStatistics>) -> Int {
        LichessBotStatsFormat.countWidth(LichessBotStatsPeriod.allCases.flatMap { period -> [LichessBotResultTally] in
            let record = rows[period].record
            return [record.all, record.versusBots, record.versusHumans, record.asWhite, record.asBlack]
        })
    }
}
