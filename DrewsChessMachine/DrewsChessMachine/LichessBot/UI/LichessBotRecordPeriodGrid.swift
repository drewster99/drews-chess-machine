import SwiftUI

/// Item 1's table (§3.3): one row per period, last hour to all time —
/// scored games, W–D–L, score, performance rating, average opponent rating,
/// and the opponent-kind and color splits. No rating change: each speed is
/// its own Lichess rating pool, and a sum across pools is not a change of
/// any rating DCM had; the Time controls pane shows each pool's change.
struct LichessBotRecordPeriodGrid: View {
    let rows: LichessBotPeriodValues<LichessBotPeriodStatistics>
    let widths: LichessBotRecordPeriodColumnWidths

    static let titles = ["", "Games", "W–D–L", "Score", "Perf", "Opp avg", "vs bots", "vs humans", "as White", "as Black"]

    /// Columns sized to `rows` alone.
    init(rows: LichessBotPeriodValues<LichessBotPeriodStatistics>) {
        self.init(rows: rows, widths: LichessBotRecordPeriodColumnWidths(tables: [rows]))
    }

    /// Columns sized by `widths`, so the table keeps its shape when another
    /// filter's rows replace these.
    init(rows: LichessBotPeriodValues<LichessBotPeriodStatistics>, widths: LichessBotRecordPeriodColumnWidths) {
        self.rows = rows
        self.widths = widths
    }

    var body: some View {
        Grid(alignment: .trailing, horizontalSpacing: LichessBotStatsStyle.columnSpacing, verticalSpacing: LichessBotStatsStyle.rowSpacing) {
            LichessBotStatsHeaderRow(titles: Self.titles)
            ForEach(LichessBotStatsPeriod.allCases, id: \.self) { period in
                LichessBotRecordPeriodRow(period: period, row: rows[period], widths: widths)
            }
        }
    }
}
