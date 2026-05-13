import SwiftUI

/// Wrapper around `TrainingStatsColumn` that owns the read-only
/// replay-ratio status row. Pulled out of `UpperContentView` so it
/// is its own SwiftUI identity boundary — its body re-evaluates
/// only when one of its declared inputs changes, instead of every
/// time the parent view re-renders.
///
/// Below the policy/optimizer stats block, this view also renders
/// the self-play stats section (Games / Moves / Avg-move / Results)
/// whenever a non-nil `selfPlayStats` is supplied. The left-hand
/// `SelfPlayStatsColumn` is hidden when `isCandidateTestActive` is
/// true (forced on for `selfPlayWorkers > 1`), so without this
/// in-column duplicate the self-play counters disappear from the
/// UI for every typical multi-worker run. Showing them here as a
/// trailing section keeps them always visible and parses naturally
/// with the metric-colorizer.
struct UpperTrainingStatsColumn: View {
    let header: String
    let bodyText: AttributedString
    let realTraining: Bool
    let replayRatioSnapshot: ReplayRatioController.RatioSnapshot?
    let replayRatioTarget: Double
    let replayRatioAutoAdjust: Bool
    /// Optional secondary section appended below the policy stats —
    /// the same `(header, body)` pair `SelfPlayStatsColumn` would
    /// render on the left. `nil` outside a Play-and-Train session.
    let selfPlayStats: (header: String, body: AttributedString)?

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            TrainingStatsColumn(
                header: header,
                bodyText: bodyText,
                realTraining: realTraining
            ) {
                // Read-only replay-ratio display. Editable controls
                // (Step Delay / SP Delay / Auto toggle / Target Ratio
                // stepper) moved to the Training Settings popover's
                // Replay tab. The user explicitly asked to keep this
                // view-only readout on the main screen so the live
                // ratio remains glanceable without opening the popover.
                HStack(spacing: 6) {
                    Text("  Replay Ratio:")
                    if let snap = replayRatioSnapshot {
                        Text(String(format: "%.2f", snap.currentRatio))
                            .monospacedDigit()
                            .frame(minWidth: 40, alignment: .trailing)
                            .foregroundStyle(
                                abs(snap.currentRatio - snap.targetRatio) < 0.3
                                ? Color.primary : Color.red
                            )
                    } else {
                        Text("--")
                            .monospacedDigit()
                            .frame(minWidth: 40, alignment: .trailing)
                    }
                    Text("target:")
                        .foregroundStyle(.secondary)
                    Text(String(format: "%.2f", replayRatioTarget))
                        .monospacedDigit()
                        .frame(minWidth: 32, alignment: .trailing)
                    if replayRatioAutoAdjust {
                        Text("(auto)")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
            }
            if let selfPlayStats {
                Text("")
                Text(selfPlayStats.header)
                Text(selfPlayStats.body)
            }
        }
        .frame(minWidth: 260, alignment: .topLeading)
    }
}
