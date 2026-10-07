import SwiftUI

/// The held-result and decisive-ply figures beside the reliability chart
/// (§3.6, OD-14, OD-15).
struct LichessBotSelfAssessmentFigures: View {
    let assessment: LichessBotSelfAssessmentStatistics

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(LichessBotStatsFormat.held(turned: assessment.heldWins.turned, held: assessment.heldWins.held, verb: "blown", noun: "held wins"))
                .help("A held win: the value head gave a win ≥ 80% on two consecutive DCM moves. Blown: the game was then drawn or lost.")
            Text(LichessBotStatsFormat.held(turned: assessment.heldLosses.turned, held: assessment.heldLosses.held, verb: "saved", noun: "held losses"))
                .help("A held loss: a loss ≥ 80% on two consecutive DCM moves. Saved: the game was then drawn or won.")
            Text(LichessBotStatsFormat.decisive(assessment.decisiveWins, result: "Won"))
                .help("The earliest DCM move from which the win probability stayed ≥ 80% through DCM's last decision")
            Text(LichessBotStatsFormat.decisive(assessment.decisiveLosses, result: "Lost"))
                .help("The earliest DCM move from which the loss probability stayed ≥ 80% through DCM's last decision")
        }
        .font(LichessBotStatsStyle.figureFont)
        .fixedSize(horizontal: false, vertical: true)
    }
}
