import SwiftUI

/// The panel's pickers: the pane, the period and the Rated / Casual / All
/// filter (OD-3, OD-7). All three are the pipeline's remembered selections;
/// the filter applies to the period table as well as the panes.
///
/// The pane is a menu, not a segmented control: eleven panes (the first
/// pass and §11's later ones) do not fit one segmented row at the
/// narrowest window, and two segmented rows bound to one selection would
/// leave one row with a selection it has no tag for, which SwiftUI reports
/// as an invalid selection at run time. The menu groups the first pass and
/// the later panes in two sections.
struct LichessBotRecordPanelHeader: View {
    @Bindable var pipeline: LichessBotRecordStatisticsPipeline

    var body: some View {
        HStack(spacing: 16) {
            Picker("Show", selection: $pipeline.rememberedPane) {
                Section {
                    ForEach(LichessBotRecordPane.firstPass) { pane in
                        Text(pane.label).tag(pane)
                    }
                }
                Section {
                    ForEach(LichessBotRecordPane.later) { pane in
                        Text(pane.label).tag(pane)
                    }
                }
            }
            .fixedSize()
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
        }
        .pickerStyle(.menu)
    }
}
