import SwiftUI

/// Main panel — two fixed-width columns (game stats + training
/// stats) so a mode that shows one never changes size when a
/// mode that shows both (real-training) is active. Each column
/// is gated independently: whichever is relevant for the current
/// mode is rendered, the other is simply omitted. In real-training
/// mode both are shown side-by-side. The columns themselves are
/// supplied by the caller as `@ViewBuilder` closures, so this
/// shell stays decoupled from their state-heavy initializers.
struct MainTextPanel<SelfPlay: View, Training: View>: View {
    let isGameMode: Bool
    let isTrainingMode: Bool
    let isCandidateTestActive: Bool
    /// Pre-rendered inference result text used in both Candidate-test
    /// mode (replaces the self-play column) and pure forward-pass
    /// mode (renders alone). `nil` when there is no inference data.
    let inferenceResultText: String?
    /// Most-recent training error message, surfaced in red below the
    /// columns. `nil` when training has not raised an error.
    let trainingError: String?
    @ViewBuilder var selfPlayColumn: () -> SelfPlay
    @ViewBuilder var trainingColumn: () -> Training

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 8) {
                HStack(alignment: .top, spacing: 16) {
                    // Candidate test mode replaces the game-stats
                    // column with the inference-result column so
                    // the user sees what the network thinks of the
                    // probe position alongside the running training
                    // stats. Same min-width as the game column so
                    // the overall text panel doesn't reflow when
                    // toggling between Game run and Candidate test.
                    if isGameMode && !isCandidateTestActive {
                        selfPlayColumn()
                    }
                    if isCandidateTestActive, let result = inferenceResultText {
                        Text(result)
                            .frame(minWidth: 330, alignment: .topLeading)
                    }
                    if isTrainingMode {
                        trainingColumn()
                    }
                    if !isGameMode, !isTrainingMode, let result = inferenceResultText {
                        Text(result)
                    }
                }

                if let trainingError {
                    Text(trainingError).foregroundStyle(.red)
                }
            }
            .font(.system(.body, design: .monospaced))
            .textSelection(.enabled)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }
}
