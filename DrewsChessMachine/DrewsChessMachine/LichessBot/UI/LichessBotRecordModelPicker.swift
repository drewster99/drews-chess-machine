import SwiftUI

/// The Record card's Model filter: every model, or one model's games
/// (`LichessBotStatsModelSelection`). The menu lists the models that played
/// in the selected period and Games filter, with those games' counts, so it
/// matches the panes (most recently played first). A selected model with no
/// games there keeps its own item, saying so, so the menu always holds the
/// selection.
struct LichessBotRecordModelPicker: View {
    @Bindable var pipeline: LichessBotRecordStatisticsPipeline

    var body: some View {
        let choices = pipeline.modelMenu.choices(filter: pipeline.rememberedFilter, period: pipeline.rememberedPeriod)
        Picker("Model", selection: $pipeline.rememberedModel) {
            Text("All models").tag(LichessBotStatsModelSelection.all)
            Divider()
            ForEach(choices) { choice in
                Text(choice.menuLabel).tag(LichessBotStatsModelSelection.model(choice.key))
            }
            ForEach(unlistedSelection(choices), id: \.key) { item in
                Text(item.label).tag(LichessBotStatsModelSelection.model(item.key))
            }
        }
        .frame(maxWidth: LichessBotStatsStyle.modelPickerMaximumWidth)
        .help("Count only the games a model played: the model that chose most of DCM's moves in the game (a mid-game switch counts for the majority model). Applies to every pane and the period table. The menu lists the models that played in the selected period and Games filter.")
    }

    /// The selected model when this period's list lacks it: zero or one item,
    /// named from every game when it played earlier.
    private func unlistedSelection(_ choices: [LichessBotStatsModelChoice]) -> [(key: LichessBotModelKey, label: String)] {
        guard case .model(let key) = pipeline.rememberedModel,
              !choices.contains(where: { $0.key == key }) else { return [] }
        guard let earlier = pipeline.modelMenu.everyModel.first(where: { $0.key == key }) else {
            return [(key, "Remembered model (loading)")]
        }
        return [(key, "\(earlier.modelID) · \(earlier.checkpointLabel) (no games in this period)")]
    }
}
