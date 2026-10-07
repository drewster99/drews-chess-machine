import SwiftUI

/// Item 4's pane: the per-model table, a note for decisions that name no
/// recorded model, and the progression chart when some run has at least
/// two points.
struct LichessBotModelsPane: View {
    let models: LichessBotModelStatistics

    var body: some View {
        VStack(alignment: .leading, spacing: LichessBotStatsStyle.sectionSpacing) {
            LichessBotModelRecordTable(models: models)
            LichessBotPaneEmptyNote(text: "\(models.decisionsWithoutGeneration) of DCM's decisions name no model the record lists; they are left out of every model's count")
                .shown(models.decisionsWithoutGeneration > 0)
            LichessBotModelProgressChart(points: models.progression)
                .shown(!models.progression.isEmpty)
        }
    }
}
