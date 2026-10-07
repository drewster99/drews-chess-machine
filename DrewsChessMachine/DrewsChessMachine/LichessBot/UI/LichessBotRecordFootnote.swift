import SwiftUI

/// The line under the period table (§5.1): games not counted, rated games
/// without a rating change, and games without move data — each part only
/// when non-zero. It always takes two lines' height, said or not, so the
/// panel below does not jump as games arrive or the filter changes (Casual
/// has no rated games, so no rating-change note).
struct LichessBotRecordFootnote: View {
    let statistics: LichessBotRecordStatistics.FilterStatistics

    var body: some View {
        Text(Self.text(statistics))
            .font(LichessBotStatsStyle.noteFont)
            .foregroundStyle(LichessBotStatsStyle.neutral)
            .lineLimit(2, reservesSpace: true)
    }

    static func text(_ statistics: LichessBotRecordStatistics.FilterStatistics) -> String {
        var parts: [String] = []
        if statistics.notCounted > 0 {
            parts.append("\(statistics.notCounted) game\(statistics.notCounted == 1 ? "" : "s") not counted (aborted or never started)")
        }
        if statistics.ratedWithoutRatingChange > 0 {
            parts.append("* \(statistics.ratedWithoutRatingChange) rated game\(statistics.ratedWithoutRatingChange == 1 ? "" : "s") without a rating change from Lichess")
        }
        if statistics.rowsWithoutMoveData > 0 {
            parts.append("\(statistics.rowsWithoutMoveData) game\(statistics.rowsWithoutMoveData == 1 ? "" : "s") without move data")
        }
        return parts.joined(separator: " · ")
    }
}
