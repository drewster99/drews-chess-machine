import SwiftUI

/// The held-result and decisive-ply figures beside the reliability chart
/// (§3.6, OD-14, OD-15).
struct LichessBotSelfAssessmentFigures: View {
    let assessment: LichessBotSelfAssessmentStatistics

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(LichessBotStatsFormat.held(turned: assessment.heldWins.turned, held: assessment.heldWins.held, verb: "blown", noun: "held wins"))
                .help("A held win: the value head gave a win ≥ 80% on two consecutive DCM moves (two, so a one-move spike before a recapture does not count). Blown: the game was then drawn or lost. Many blown wins mean the value head is too optimistic.")
            Text(LichessBotStatsFormat.held(turned: assessment.heldLosses.turned, held: assessment.heldLosses.held, verb: "saved", noun: "held losses"))
                .help("A held loss: the value head gave a loss ≥ 80% on two consecutive DCM moves. Saved: the game was then drawn or won.")
            Text(LichessBotStatsFormat.decisive(assessment.decisiveWins, result: "Won"))
                .help("Won games: the earliest DCM move from which the value head's win probability stayed ≥ 80% through DCM's last decision, as a ply. Plies count both sides' moves. \"plies before the end\": the average distance from that ply to the game's last ply. n: games that settled; never: games whose probability for their result was below 80% at DCM's last decision; no data: games with no recorded readings.")
            Text(LichessBotStatsFormat.decisive(assessment.decisiveLosses, result: "Lost"))
                .help("Lost games: the earliest DCM move from which the value head's loss probability stayed ≥ 80% through DCM's last decision, as a ply. Plies count both sides' moves. \"plies before the end\": the average distance from that ply to the game's last ply. n: games that settled; never: games whose probability for their result was below 80% at DCM's last decision; no data: games with no recorded readings.")
        }
        .font(LichessBotStatsStyle.figureFont)
        .fixedSize(horizontal: false, vertical: true)
    }
}
