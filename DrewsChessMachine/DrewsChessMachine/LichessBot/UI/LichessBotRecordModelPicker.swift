import SwiftUI

/// The Record card's Model filter: every model, or one model's games
/// (`LichessBotStatsModelSelection`), most recently played first. A
/// remembered model the first result hasn't listed yet keeps its own item,
/// so the menu always holds the selection.
struct LichessBotRecordModelPicker: View {
    @Bindable var pipeline: LichessBotRecordStatisticsPipeline

    var body: some View {
        Picker("Model", selection: $pipeline.rememberedModel) {
            Text("All models").tag(LichessBotStatsModelSelection.all)
            Divider()
            ForEach(pipeline.modelChoices) { choice in
                Text(choice.menuLabel).tag(LichessBotStatsModelSelection.model(choice.key))
            }
            ForEach(unlistedSelection, id: \.self) { key in
                Text("Remembered model (loading)").tag(LichessBotStatsModelSelection.model(key))
            }
        }
        .frame(maxWidth: 360)
        .help("Count only the games a model played: the model that chose most of DCM's moves in the game (a mid-game switch counts for the majority model). Applies to every pane and the period table.")
    }

    /// The selected model when the choices don't list it (yet): zero or one
    /// key.
    private var unlistedSelection: [LichessBotModelKey] {
        guard case .model(let key) = pipeline.rememberedModel,
              !pipeline.modelChoices.contains(where: { $0.key == key }) else { return [] }
        return [key]
    }
}
