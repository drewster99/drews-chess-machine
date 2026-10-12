import Foundation
import TrainingParametersMacroSupport

/// What `LRScheduleChart` draws: the draft cycle being edited and the other
/// values the fed LR depends on. Equatable, so the chart re-samples only when
/// something it shows changes.
struct LRScheduleChartInput: Equatable, Sendable {
    let cycle: LRMomentumCycle
    let lrWarmupSteps: Int
    let staticLearningRate: Float
    let staticMomentum: Float
    let batchSize: Int
    let sqrtBatchScaling: Bool

    /// The steps the chart spans: two LR-cycle periods past the last
    /// keyframe, and at least 50,000.
    var stepRange: ClosedRange<Int> {
        let last = cycle.lrKeyframes.keyframes.last?.step ?? 0
        return 0...max(50_000, last + 2 * max(1, cycle.lrPeriodSteps))
    }

    /// The schedule over `stepRange` in `bucketCount` buckets, every step
    /// evaluated through `LRMomentumCycleReadout.values` (what the trainer is
    /// fed). `isCancelled` lets a superseded draft abandon its sampling.
    func sample(bucketCount: Int, isCancelled: () -> Bool) -> LRScheduleSample? {
        LRScheduleSample.sample(steps: stepRange, bucketCount: bucketCount, isCancelled: isCancelled) { step in
            let fed = LRMomentumCycleReadout.values(
                completedTrainSteps: step, lrWarmupSteps: lrWarmupSteps, cycle: cycle,
                staticLearningRate: staticLearningRate, staticMomentum: staticMomentum, batchSize: batchSize,
                sqrtBatchScaling: sqrtBatchScaling, sqrtScaleBaseBatchSize: ChessTrainer.sqrtScaleBaseBatchSize)
            let values = cycle.values(completedTrainSteps: step, lrWarmupSteps: lrWarmupSteps)
            return .init(learningRate: Double(fed.learningRate), momentum: Double(fed.momentum),
                         lrPeak: values.lrPeak, lrTrough: values.lrTrough)
        }
    }
}
