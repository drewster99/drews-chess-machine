import SwiftUI

/// The panel's pickers on one row: the pane, the period, the Rated /
/// Casual / All filter (OD-3, OD-7), the Opponents filter (bots / humans)
/// and the Model filter, all menus. All five are the pipeline's remembered
/// selections; the filters combine, and apply to the period table as well
/// as the panes.
struct LichessBotRecordPanelHeader: View {
    @Bindable var pipeline: LichessBotRecordStatisticsPipeline

    var body: some View {
        HStack(spacing: 16) {
            LichessBotRecordPanePicker(pane: $pipeline.rememberedPane)
            Picker("Period", selection: $pipeline.rememberedPeriod) {
                ForEach(LichessBotStatsPeriod.allCases, id: \.self) { period in
                    Text(period.label).tag(period)
                }
            }
            .fixedSize()
            Picker("Games", selection: $pipeline.rememberedFilter) {
                ForEach(LichessBotStatsFilter.allCases, id: \.self) { filter in
                    Text(filter.label).tag(filter)
                }
            }
            .fixedSize()
            LichessBotOpponentKindFilterPicker(filter: $pipeline.rememberedOpponents)
            LichessBotRecordModelPicker(pipeline: pipeline)
        }
        .pickerStyle(.menu)
    }
}
