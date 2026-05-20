import SwiftUI

/// Left-hand stats panel under the board. In Play-and-Train mode
/// the parent supplies the pre-built `(header, body)` column from
/// `playAndTrainStatsText`; otherwise the parent supplies a
/// pre-rendered `gameSnapshot.statsText` string and the panel
/// renders just that. The Concurrency Stepper used to live inside
/// this column, but the column itself is hidden whenever
/// `isCandidateTestActive` is true (which is forced on for
/// `selfPlayConcurrency > 1`), so the Stepper became unreachable as
/// soon as the user bumped concurrency above 1. The Stepper now
/// lives in its own always-visible row at the top of
/// `UpperContentView` (`ConcurrencyStepperRow`).
struct SelfPlayStatsColumn: View {
    let realTrainingColumn: (header: String, body: String)?
    let fallbackText: String
    let colorize: (String) -> AttributedString

    var body: some View {
        Group {
            if let column = realTrainingColumn {
                VStack(alignment: .leading, spacing: 0) {
                    Text(column.header)
                    Text(colorize(column.body))
                }
                .frame(minWidth: 330, alignment: .topLeading)
            } else {
                Text(fallbackText)
                    .frame(minWidth: 330, alignment: .topLeading)
            }
        }
    }
}

/// Always-visible Concurrency control row. Rendered above the
/// board+text panel whenever `realTraining` is true so the user
/// can change `selfPlayConcurrency` regardless of which board mode
/// (game-run, candidate-test, progress-rate) the session happens
/// to be in. The Stepper writes through to
/// `trainingParams.selfPlayConcurrency` directly via `@Bindable`; the
/// `didSet` on that property propagates to the workers' shared
/// box on the next reconcile tick.
struct ConcurrencyStepperRow: View {
    @Bindable var trainingParams: TrainingParameters
    let maxWorkers: Int

    var body: some View {
        HStack(spacing: 6) {
            Text("Concurrency:")
            Text("\(trainingParams.selfPlayConcurrency)")
                .monospacedDigit()
                .frame(minWidth: 24, alignment: .trailing)
            Stepper(
                "Concurrency",
                value: $trainingParams.selfPlayConcurrency,
                in: 1...maxWorkers
            )
            .labelsHidden()
        }
        .font(.system(.body, design: .monospaced))
    }
}
