import SwiftUI

/// The first-pass panes (§9 P4–P8), each shown only when selected.
struct LichessBotFirstPassPanes: View {
    let account: LichessBotAccount?
    let statistics: LichessBotRecordStatistics
    let filter: LichessBotStatsFilter
    let period: LichessBotStatsPeriod
    let pane: LichessBotRecordPane

    var body: some View {
        let breakdowns = statistics[filter].byPeriod[period]
        ZStack(alignment: .topLeading) {
            // Time controls also lists speeds the account is rated in, so
            // it shows even in an empty period.
            ScrollView(.horizontal) {
                LichessBotTimeControlTable(statistics: statistics, filter: filter, period: period, account: account)
            }
            .shown(pane == .timeControls)
            LichessBotModelsPane(models: breakdowns.models)
                .shown(pane == .models)
            LichessBotSelfAssessmentPane(assessment: breakdowns.selfAssessment)
                .shown(pane == .selfAssessment)
            ScrollView(.horizontal) {
                LichessBotEndingsTable(endings: breakdowns.endings)
            }
            .shown(pane == .endings)
            LichessBotOpponentStrengthPane(strength: breakdowns.opponentStrength)
                .shown(pane == .opponentStrength)
        }
    }
}
