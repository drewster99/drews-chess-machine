import Foundation

/// Conformance shared by every sample type that can flow through
/// `ChartSampleRing` and the chart decimator. Single requirement is
/// the X-axis projection so `firstIndex(elapsedSecAtLeast:projection:)`
/// can locate the visible-window bounds without knowing the concrete
/// element type.
protocol ChartSample: Sendable {
    /// Seconds since session start. Used as the X coordinate by
    /// every time-series chart and as the binary-search key by the
    /// decimator.
    var elapsedSec: Double { get }
}

/// Min/max envelope for a single numeric series within one
/// decimated bucket. `nil` everywhere a bucket has no samples that
/// reported a value for the underlying field.
struct ChartBucketRange: Sendable, Equatable {
    let min: Double
    let max: Double
}

/// One decimation bucket of training samples. Carries a time anchor
/// plus a min/max envelope per numeric series and the most-recent
/// observed value per categorical series, so each chart subview can
/// render one or two marks per bucket without walking the full
/// underlying ring.
///
/// A new instance is emitted for every bucket that contains at
/// least one sample; empty buckets are skipped entirely. The `id`
/// doubles as the bucket index inside the visible window — stable
/// across chart redraws within a single decimation pass, which is
/// what SwiftUI's `ForEach` keying needs.
struct TrainingBucket: Sendable, Equatable, Identifiable {
    let id: Int
    let elapsedSec: Double

    let policyLoss: ChartBucketRange?
    let valueLoss: ChartBucketRange?
    /// W/D/L value-head metrics (post-WDL switch). `valueMean` /
    /// `valueAbsMean` are the mean / mean-abs of the derived scalar
    /// `v = p_win − p_loss`; `valueProbWin/Draw/Loss` are the batch
    /// means of the W/D/L softmax (sum ≈ 1). Charted on the new
    /// value-head row of `TrainingChartGridView`.
    let valueMean: ChartBucketRange?
    let valueAbsMean: ChartBucketRange?
    let valueProbWin: ChartBucketRange?
    let valueProbDraw: ChartBucketRange?
    let valueProbLoss: ChartBucketRange?
    let policyEntropy: ChartBucketRange?
    let policyNonNegCount: ChartBucketRange?
    let policyNonNegIllegalCount: ChartBucketRange?
    let gradNorm: ChartBucketRange?
    let velocityNorm: ChartBucketRange?
    let policyHeadWeightNorm: ChartBucketRange?
    let replayRatio: ChartBucketRange?
    let policyLossWin: ChartBucketRange?
    let policyLossLoss: ChartBucketRange?
    let legalEntropy: ChartBucketRange?
    let legalMass: ChartBucketRange?
    let cpuPercent: ChartBucketRange?
    let gpuBusyPercent: ChartBucketRange?
    let appMemoryGB: ChartBucketRange?
    let gpuMemoryGB: ChartBucketRange?

    /// Last observed `lowPowerMode` flag in this bucket. Charts
    /// render this as a step trace, so the bucket's last-observed
    /// value is the right semantics — a brief mid-bucket toggle
    /// would otherwise look like the bucket's overall state.
    let lowPowerMode: Bool?
    /// Last observed thermal state in this bucket. Same step-trace
    /// semantics as `lowPowerMode`.
    let thermalState: ProcessInfo.ThermalState?
}

/// One decimation bucket of progress-rate samples — separate type
/// from `TrainingBucket` because the underlying sample schema is
/// different (cumulative move counters + rolling rates) and the
/// progress-rate chart is rendered both inside `TrainingChartGridView`
/// (the small sparkline tile) and standalone in `ContentView` (the
/// large session-overview chart).
struct ProgressRateBucket: Sendable, Equatable, Identifiable {
    let id: Int
    let elapsedSec: Double

    let combinedMovesPerHour: ChartBucketRange?
    let selfPlayMovesPerHour: ChartBucketRange?
    let trainingMovesPerHour: ChartBucketRange?
}

/// One decimation pass's complete output. Held in `LowerContentView`'s
/// `@State` and recomputed every time a sample appends, the user
/// scrolls or zooms, or the chart cell's pixel width changes. The
/// per-chart subviews inside the chart grid and the big progress
/// chart in `ContentView` all read from this single value — that is
/// the "same data structure throughout" rule from the design note.
///
/// `Equatable` so SwiftUI's view-diff can skip re-rendering when a
/// recomputation produces an identical frame (e.g. a heartbeat tick
/// that didn't cross a bucket boundary).
struct DecimatedChartFrame: Sendable, Equatable {
    let trainingBuckets: [TrainingBucket]
    let progressRateBuckets: [ProgressRateBucket]
    /// Visible window in elapsed-seconds. Charts use this both for
    /// `chartXVisibleDomain(length:)` (via `upperBound - lowerBound`)
    /// and for "is this hover time inside the data range" gating.
    let visibleDomain: ClosedRange<Double>
    /// Maximum elapsed time observed in the underlying training ring
    /// at decimation time. Used for the chart's full X-scale domain
    /// (`0...max(lastElapsed, visibleDomainSec)`) and for the
    /// in-progress arena's "now" marker.
    let lastTrainingElapsedSec: Double?
    /// Same for the progress-rate ring. Tracked separately because
    /// the two rings can have slightly different last-elapsed
    /// timestamps (the heartbeat appends them sequentially in the
    /// same tick).
    let lastProgressRateElapsedSec: Double?

    static let empty = DecimatedChartFrame(
        trainingBuckets: [],
        progressRateBuckets: [],
        visibleDomain: 0...0,
        lastTrainingElapsedSec: nil,
        lastProgressRateElapsedSec: nil
    )
}

// MARK: - Decimator

/// Pure functions that walk a chart sample array inside a visible
/// window and reduce it to a fixed-budget bucket array.
enum ChartDecimator {

    /// Floor on the bucket budget.
    static let minimumBucketCount: Int = 32
    /// Ceiling on the bucket budget.
    static let maximumBucketCount: Int = 1500

    /// Compute a fresh `DecimatedChartFrame` from the supplied sample slices.
    static func decimate(
        trainingSamples: [TrainingChartSample],
        progressRateSamples: [ProgressRateSample],
        visibleStart: Double,
        visibleEnd: Double,
        lastT: Double?,
        lastP: Double?,
        trainingBucketBudget: Int,
        progressRateBucketBudget: Int
    ) -> DecimatedChartFrame {
        let domain = visibleStart...visibleEnd

        let trainingBuckets = decimateTraining(
            samples: trainingSamples,
            visibleStart: visibleStart,
            visibleEnd: visibleEnd,
            bucketBudget: trainingBucketBudget
        )
        let progressBuckets = decimateProgressRate(
            samples: progressRateSamples,
            visibleStart: visibleStart,
            visibleEnd: visibleEnd,
            bucketBudget: progressRateBucketBudget
        )

        return DecimatedChartFrame(
            trainingBuckets: trainingBuckets,
            progressRateBuckets: progressBuckets,
            visibleDomain: domain,
            lastTrainingElapsedSec: lastT,
            lastProgressRateElapsedSec: lastP
        )
    }

    // MARK: - Training

    static func decimateTraining(
        samples: [TrainingChartSample],
        visibleStart: Double,
        visibleEnd: Double,
        bucketBudget: Int
    ) -> [TrainingBucket] {
        let bucketCount = clampBucketCount(bucketBudget)
        guard bucketCount > 0, visibleEnd > visibleStart, !samples.isEmpty else { return [] }

        let bucketWidth = (visibleEnd - visibleStart) / Double(bucketCount)
        var builders = Array(
            repeating: TrainingBucketBuilder(),
            count: bucketCount
        )

        for sample in samples {
            let raw = (sample.elapsedSec - visibleStart) / bucketWidth
            let idx = max(0, min(bucketCount - 1, Int(raw)))
            builders[idx].absorb(sample)
        }

        var out: [TrainingBucket] = []
        out.reserveCapacity(bucketCount)
        for i in 0..<bucketCount where builders[i].hasAnyData {
            out.append(builders[i].finalize(id: i))
        }
        return out
    }

    // MARK: - Progress rate

    static func decimateProgressRate(
        samples: [ProgressRateSample],
        visibleStart: Double,
        visibleEnd: Double,
        bucketBudget: Int
    ) -> [ProgressRateBucket] {
        let bucketCount = clampBucketCount(bucketBudget)
        guard bucketCount > 0, visibleEnd > visibleStart, !samples.isEmpty else { return [] }

        let bucketWidth = (visibleEnd - visibleStart) / Double(bucketCount)
        var builders = Array(
            repeating: ProgressRateBucketBuilder(),
            count: bucketCount
        )

        for sample in samples {
            let raw = (sample.elapsedSec - visibleStart) / bucketWidth
            let idx = max(0, min(bucketCount - 1, Int(raw)))
            builders[idx].absorb(sample)
        }

        var out: [ProgressRateBucket] = []
        out.reserveCapacity(bucketCount)
        for i in 0..<bucketCount where builders[i].hasAnyData {
            out.append(builders[i].finalize(id: i))
        }
        return out
    }

    // MARK: - Helpers

    private static func clampBucketCount(_ requested: Int) -> Int {
        if requested < minimumBucketCount { return minimumBucketCount }
        if requested > maximumBucketCount { return maximumBucketCount }
        return requested
    }
}

// MARK: - Mutable accumulators (file-private builder structs)

/// Tracks min/max for a single numeric series across the samples
/// absorbed into one bucket. `nil` until the first non-nil value
/// arrives; absorbing further values widens the envelope.
private struct NumericAccumulator {
    var range: ChartBucketRange?

    mutating func absorb(_ value: Double?) {
        guard let v = value else { return }
        if let r = range {
            range = ChartBucketRange(min: Swift.min(r.min, v), max: Swift.max(r.max, v))
        } else {
            range = ChartBucketRange(min: v, max: v)
        }
    }
}

private struct TrainingBucketBuilder {
    var hasAnyData: Bool = false
    var lastElapsedSec: Double = 0

    var policyLoss = NumericAccumulator()
    var valueLoss = NumericAccumulator()
    var valueMean = NumericAccumulator()
    var valueAbsMean = NumericAccumulator()
    var valueProbWin = NumericAccumulator()
    var valueProbDraw = NumericAccumulator()
    var valueProbLoss = NumericAccumulator()
    var policyEntropy = NumericAccumulator()
    var policyNonNegCount = NumericAccumulator()
    var policyNonNegIllegalCount = NumericAccumulator()
    var gradNorm = NumericAccumulator()
    var velocityNorm = NumericAccumulator()
    var policyHeadWeightNorm = NumericAccumulator()
    var replayRatio = NumericAccumulator()
    var policyLossWin = NumericAccumulator()
    var policyLossLoss = NumericAccumulator()
    var legalEntropy = NumericAccumulator()
    var legalMass = NumericAccumulator()
    var cpuPercent = NumericAccumulator()
    var gpuBusyPercent = NumericAccumulator()
    var appMemoryGB = NumericAccumulator()
    var gpuMemoryGB = NumericAccumulator()

    var lowPowerMode: Bool? = nil
    var thermalState: ProcessInfo.ThermalState? = nil

    mutating func absorb(_ s: TrainingChartSample) {
        hasAnyData = true
        lastElapsedSec = s.elapsedSec

        policyLoss.absorb(s.rollingPolicyLoss)
        valueLoss.absorb(s.rollingValueLoss)
        valueMean.absorb(s.rollingValueMean)
        valueAbsMean.absorb(s.rollingValueAbsMean)
        valueProbWin.absorb(s.rollingValueProbWin)
        valueProbDraw.absorb(s.rollingValueProbDraw)
        valueProbLoss.absorb(s.rollingValueProbLoss)
        policyEntropy.absorb(s.rollingPolicyEntropy)
        policyNonNegCount.absorb(s.rollingPolicyNonNegCount)
        policyNonNegIllegalCount.absorb(s.rollingPolicyNonNegIllegalCount)
        gradNorm.absorb(s.rollingGradNorm)
        velocityNorm.absorb(s.rollingVelocityNorm)
        policyHeadWeightNorm.absorb(s.rollingPolicyHeadWeightNorm)
        replayRatio.absorb(s.replayRatio)
        policyLossWin.absorb(s.rollingPolicyLossWin)
        policyLossLoss.absorb(s.rollingPolicyLossLoss)
        legalEntropy.absorb(s.rollingLegalEntropy)
        legalMass.absorb(s.rollingLegalMass)
        cpuPercent.absorb(s.cpuPercent)
        gpuBusyPercent.absorb(s.gpuBusyPercent)
        appMemoryGB.absorb(s.appMemoryGB)
        gpuMemoryGB.absorb(s.gpuMemoryGB)

        if let v = s.lowPowerMode { lowPowerMode = v }
        if let v = s.thermalState { thermalState = v }
    }

    func finalize(id: Int) -> TrainingBucket {
        TrainingBucket(
            id: id,
            elapsedSec: lastElapsedSec,
            policyLoss: policyLoss.range,
            valueLoss: valueLoss.range,
            valueMean: valueMean.range,
            valueAbsMean: valueAbsMean.range,
            valueProbWin: valueProbWin.range,
            valueProbDraw: valueProbDraw.range,
            valueProbLoss: valueProbLoss.range,
            policyEntropy: policyEntropy.range,
            policyNonNegCount: policyNonNegCount.range,
            policyNonNegIllegalCount: policyNonNegIllegalCount.range,
            gradNorm: gradNorm.range,
            velocityNorm: velocityNorm.range,
            policyHeadWeightNorm: policyHeadWeightNorm.range,
            replayRatio: replayRatio.range,
            policyLossWin: policyLossWin.range,
            policyLossLoss: policyLossLoss.range,
            legalEntropy: legalEntropy.range,
            legalMass: legalMass.range,
            cpuPercent: cpuPercent.range,
            gpuBusyPercent: gpuBusyPercent.range,
            appMemoryGB: appMemoryGB.range,
            gpuMemoryGB: gpuMemoryGB.range,
            lowPowerMode: lowPowerMode,
            thermalState: thermalState
        )
    }
}

// MARK: - ChartSample conformances

extension TrainingChartSample: ChartSample {}
extension ProgressRateSample: ChartSample {}

private struct ProgressRateBucketBuilder {
    var hasAnyData: Bool = false
    var lastElapsedSec: Double = 0

    var combined = NumericAccumulator()
    var selfPlay = NumericAccumulator()
    var training = NumericAccumulator()

    mutating func absorb(_ s: ProgressRateSample) {
        hasAnyData = true
        lastElapsedSec = s.elapsedSec
        combined.absorb(s.combinedMovesPerHour)
        selfPlay.absorb(s.selfPlayMovesPerHour)
        training.absorb(s.trainingMovesPerHour)
    }

    func finalize(id: Int) -> ProgressRateBucket {
        ProgressRateBucket(
            id: id,
            elapsedSec: lastElapsedSec,
            combinedMovesPerHour: combined.range,
            selfPlayMovesPerHour: selfPlay.range,
            trainingMovesPerHour: training.range
        )
    }
}
