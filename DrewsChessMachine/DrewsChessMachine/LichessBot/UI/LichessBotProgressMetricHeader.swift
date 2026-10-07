import SwiftUI

/// The progression chart's title and its score / performance picker.
struct LichessBotProgressMetricHeader: View {
    @Binding var metric: LichessBotProgressMetric

    var body: some View {
        HStack {
            Text("Progression")
                .font(LichessBotStatsStyle.sectionFont)
            Picker("Plot", selection: $metric) {
                ForEach(LichessBotProgressMetric.allCases) { metric in
                    Text(metric.label).tag(metric)
                }
            }
            .pickerStyle(.menu)
            .fixedSize()
        }
    }
}
