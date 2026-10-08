import SwiftUI

/// Item 6 (`LICHESS_BOT_RECORD_STATS_PLAN.md` §3.6): the network judging
/// itself from its own per-move W / D / L. The calibration table leads,
/// because the held-win counts below mostly measure how optimistic the
/// value head is, and read correctly only in its light.
struct LichessBotSelfAssessmentPane: View {
    let assessment: LichessBotSelfAssessmentStatistics

    var body: some View {
        VStack(alignment: .leading, spacing: LichessBotStatsStyle.sectionSpacing) {
            ScrollView(.horizontal) {
                LichessBotCalibrationTable(rows: assessment.calibration)
            }
            HStack(alignment: .top, spacing: 24) {
                LichessBotReliabilityChart(buckets: assessment.reliability)
                LichessBotSelfAssessmentFigures(assessment: assessment)
            }
            // One above the other: side by side they are wider than the
            // narrowest window.
            LichessBotHeldGamesList(
                title: "Recent blown wins",
                help: "Games where the value head held a win (≥ 80% on two consecutive DCM moves) and DCM then drew or lost: where the network was sure it was winning and was wrong.",
                games: assessment.heldWins.recentTurned)
            LichessBotHeldGamesList(
                title: "Recent saves",
                help: "Games where the value head held a loss (≥ 80% on two consecutive DCM moves) and DCM then drew or won.",
                games: assessment.heldLosses.recentTurned)
            LichessBotPaneEmptyNote(text: "\(assessment.gamesWithoutMoveData) scored game\(assessment.gamesWithoutMoveData == 1 ? "" : "s") without move data are left out of this pane")
                .shown(assessment.gamesWithoutMoveData > 0)
        }
    }
}
