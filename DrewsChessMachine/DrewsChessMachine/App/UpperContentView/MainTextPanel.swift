import SwiftUI

/// Main panel — two fixed-width columns (game stats + training
/// stats) so a mode that shows one never changes size when a
/// mode that shows both (real-training) is active. Each column
/// is gated independently: whichever is relevant for the current
/// mode is rendered, the other is simply omitted. In real-training
/// mode both are shown side-by-side. The columns themselves are
/// supplied by the caller as `@ViewBuilder` closures, so this
/// shell stays decoupled from their state-heavy initializers.
///
/// `topColumn2` sits above whatever column-2 body the current mode
/// emits (selfPlayColumn / inference text / nothing). This is the
/// always-visible top slot the View > Show Emit Window Stats
/// toggle uses — it must render regardless of `isCandidateTestActive`
/// or `inferenceResultText` so the operator can see the panel even
/// in multi-worker training mode (where the per-mode body of column
/// 2 might be empty).
struct MainTextPanel<TopColumn2: View, SelfPlay: View, Training: View>: View {
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
    @ViewBuilder var topColumn2: () -> TopColumn2
    @ViewBuilder var selfPlayColumn: () -> SelfPlay
    @ViewBuilder var trainingColumn: () -> Training

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 8) {
                HStack(alignment: .top, spacing: 16) {
                    // Column 2 — wrapped in a VStack so the always-on
                    // `topColumn2` content sits above whichever
                    // mode-specific body the current state selects
                    // (self-play stats, candidate-test inference text,
                    // or nothing). When `topColumn2` returns
                    // `EmptyView`, SwiftUI elides the slot and the
                    // VStack collapses to its body's natural size.
                    VStack(alignment: .leading, spacing: 8) {
                        topColumn2()
                        if isGameMode && !isCandidateTestActive {
                            selfPlayColumn()
                        }
                        if isCandidateTestActive, let result = inferenceResultText {
                            Text(result)
                                .frame(minWidth: 330, alignment: .topLeading)
                        }
                        // Pure forward-pass mode (neither game nor
                        // training): inference text floats here too,
                        // matching the original placement before
                        // column 2 was wrapped in a VStack.
                        if !isGameMode, !isTrainingMode, let result = inferenceResultText {
                            Text(result)
                        }
                    }
                    if isTrainingMode {
                        trainingColumn()
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
