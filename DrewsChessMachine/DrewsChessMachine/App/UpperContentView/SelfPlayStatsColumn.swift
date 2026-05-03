import SwiftUI

/// Left-hand stats panel under the board. In Play-and-Train mode
/// the parent supplies the pre-built `(header, body)` column from
/// `playAndTrainStatsText`; otherwise the parent supplies a
/// pre-rendered `gameSnapshot.statsText` string and the panel
/// renders just that. The Concurrency Stepper writes through
/// `trainingParams.selfPlayWorkers` directly via the `@Bindable`
/// projection.
struct SelfPlayStatsColumn: View {
    let realTrainingColumn: (header: String, body: String)?
    let fallbackText: String
    @Bindable var trainingParams: TrainingParameters
    let maxWorkers: Int
    let colorize: (String) -> AttributedString

    var body: some View {
        Group {
            if let column = realTrainingColumn {
                // Split layout: header Text, then the Concurrency
                // control row with the live N Stepper, then the body
                // Text. Zero spacing so the three pieces read as a
                // single continuous block. The HStack's leading "  "
                // mirrors the body's two-space label indent, and the
                // minWidth on the value Text keeps the Stepper from
                // jittering horizontally when the count changes width
                // (1 ↔ 16).
                VStack(alignment: .leading, spacing: 0) {
                    Text(column.header)
                    HStack(spacing: 6) {
                        Text("  Concurrency:")
                        Text("\(trainingParams.selfPlayWorkers)")
                            .monospacedDigit()
                            .frame(minWidth: 24, alignment: .trailing)
                        Stepper(
                            "Concurrency",
                            value: $trainingParams.selfPlayWorkers,
                            in: 1...maxWorkers
                        )
                        .labelsHidden()
                    }
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
