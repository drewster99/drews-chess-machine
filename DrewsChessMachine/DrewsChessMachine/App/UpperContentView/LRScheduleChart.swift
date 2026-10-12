import Charts
import SwiftUI
import TrainingParametersMacroSupport

/// The LR schedule being edited, drawn the way the trainer would be fed it
/// (LR_KEYFRAMES_PLAN.md, owner decision K4): the keyframes' peak and trough
/// as lines with the keyframes marked, the fed LR as a min–max band per
/// bucket (so the oscillation's extremes show even when a pixel spans many
/// steps), the practical trough floor (O-5) as a dashed rule, and the fed
/// momentum below. Sampling runs off the main thread and is abandoned when
/// the draft changes.
struct LRScheduleChart: View {
    let input: LRScheduleChartInput?
    @State private var sample: LRScheduleSample?

    private nonisolated static let bucketCount = 400

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            LRScheduleLRPlot(sample: sample, keyframes: input?.cycle.lrKeyframes.keyframes ?? [])
                .frame(height: 160)
            LRScheduleMomentumPlot(sample: sample)
                .frame(height: 60)
            Text(input == nil ? "Fix the keyframes to see the schedule." : "")
                .font(.system(size: 11))
                .foregroundStyle(.secondary)
        }
        .task(id: input) {
            guard let input else {
                sample = nil
                return
            }
            sample = await Self.sampleOffMain(input)
        }
    }

    /// Samples on a utility queue (synchronous work never runs inside the
    /// task), resuming the task with the result; a cancelled task stops the
    /// sampling at its next bucket.
    private static func sampleOffMain(_ input: LRScheduleChartInput) async -> LRScheduleSample? {
        let cancelled = SyncBox(false)
        return await withTaskCancellationHandler {
            await withCheckedContinuation { (continuation: CheckedContinuation<LRScheduleSample?, Never>) in
                DispatchQueue.global(qos: .userInitiated).async {
                    continuation.resume(returning: input.sample(bucketCount: bucketCount, isCancelled: { cancelled.value }))
                }
            }
        } onCancel: {
            cancelled.mutate { $0 = true }
        }
    }
}

/// The LR panel of `LRScheduleChart`.
private struct LRScheduleLRPlot: View {
    let sample: LRScheduleSample?
    let keyframes: [LRKeyframe]

    var body: some View {
        Chart {
            ForEach(sample?.buckets ?? [], id: \.firstStep) { bucket in
                AreaMark(x: .value("Step", bucket.firstStep),
                         yStart: .value("LR", max(bucket.lrMin, 1e-9)),
                         yEnd: .value("LR", max(bucket.lrMax, 1e-9)))
                    .foregroundStyle(Color.accentColor.opacity(0.35))
            }
            ForEach(sample?.buckets ?? [], id: \.firstStep) { bucket in
                LineMark(x: .value("Step", bucket.firstStep), y: .value("Peak", bucket.lrPeak ?? .nan),
                         series: .value("Bound", "peak"))
                    .foregroundStyle(Color.primary.opacity(0.6))
                LineMark(x: .value("Step", bucket.firstStep), y: .value("Trough", bucket.lrTrough ?? .nan),
                         series: .value("Bound", "trough"))
                    .foregroundStyle(Color.primary.opacity(0.6))
            }
            ForEach(keyframes, id: \.step) { k in
                PointMark(x: .value("Step", k.step), y: .value("Peak", k.peak))
                PointMark(x: .value("Step", k.step), y: .value("Trough", k.trough))
            }
            RuleMark(y: .value("Floor", LRMomentumCycleLogFormat.practicalTroughFloor))
                .lineStyle(StrokeStyle(lineWidth: 1, dash: [4, 3]))
                .foregroundStyle(Color.secondary)
        }
        .chartYScale(type: .log)
        .chartYAxisLabel("LR (fed)")
        .help("Band: the LR fed across each bucket of steps (min to max). Lines: the keyframes' peak and trough. Dashed: \(LRMomentumCycleLogFormat.practicalTroughFloor), below which the bf16 working weights barely move.")
    }
}

/// The momentum panel of `LRScheduleChart`.
private struct LRScheduleMomentumPlot: View {
    let sample: LRScheduleSample?

    var body: some View {
        Chart {
            ForEach(sample?.buckets ?? [], id: \.firstStep) { bucket in
                AreaMark(x: .value("Step", bucket.firstStep),
                         yStart: .value("Momentum", bucket.momentumMin),
                         yEnd: .value("Momentum", bucket.momentumMax))
                    .foregroundStyle(Color.secondary.opacity(0.4))
            }
        }
        .chartYAxisLabel("μ (fed)")
    }
}
