import Foundation
import TrainingParametersMacroSupport

/// The LR/momentum cycle with its long-run envelope given as keyframes
/// (documentation/plans-active/LR_KEYFRAMES_PLAN.md).
///
/// The in-cycle shape is the previous cycle's: a cosine fraction from the
/// cycle step (trainer step − warmup), geometric between this step's trough
/// and peak for the LR, and — when momentum follows — the same fraction,
/// inverted, linear between the momentum at the LR trough and at the LR peak.
/// What changed is where the bounds come from: `lrKeyframes`, read at the
/// trainer step, instead of one start → end decay over one horizon in cycle
/// steps. The phase deliberately stays on the cycle step, so the cycle still
/// starts after warmup and an existing run's peaks and troughs don't move.
struct LRKeyframeCycle: Sendable, Equatable {
    var lrEnabled: Bool
    var lrPeriodSteps: Int
    var lrCount: Int
    var lrInvert: Bool
    var lrKeyframes: LRKeyframeSchedule

    var momentumEnabled: Bool
    var momentumPeriodSteps: Int
    var momentumCount: Int
    var momentumMin: Double
    var momentumMax: Double
    var momentumInvert: Bool
    /// When on, momentum is driven by the LR cycle's phase (inverted) between
    /// the keyframes' momentum pair, instead of the independent momentum cycle.
    var momentumFollowsLRCycle: Bool

    /// Everything the schedule says about one trainer step. `lrPeak` /
    /// `lrTrough` are the bounds the LR swings between there, for the log
    /// lines; `nil` exactly when `learningRate` is.
    struct Values: Sendable, Equatable {
        let learningRate: Double?
        let momentum: Double?
        let lrPeak: Double?
        let lrTrough: Double?
    }

    /// The single source of truth for the schedule at trainer step
    /// `completedTrainSteps`.
    ///
    /// The gates and the guard are the previous cycle's, in the same order:
    /// LR only when cycling is on with a positive period (else `nil`, the
    /// caller's static LR); momentum only when its channel is on; following
    /// momentum only when the LR channel produced a phase and the bounds are
    /// ordered (`high >= low`, which a validated schedule always is — the
    /// guard stays so an unvalidated value can never run the cycle backwards).
    func values(completedTrainSteps: Int, lrWarmupSteps: Int) -> Values {
        let cycleStep = LRMomentumCycle.cycleStep(completedTrainSteps: completedTrainSteps, lrWarmupSteps: lrWarmupSteps)
        let bounds = lrKeyframes.bounds(atTrainerStep: completedTrainSteps)

        var learningRate: Double? = nil
        var lrPeak: Double? = nil
        var lrTrough: Double? = nil
        var lrFraction: Double? = nil
        if lrEnabled, lrPeriodSteps > 0 {
            let frac = LRMomentumCycle.cycleFraction(step: cycleStep, period: lrPeriodSteps, count: lrCount, invert: lrInvert)
            learningRate = bounds.trough * pow(bounds.peak / bounds.trough, frac)
            lrPeak = bounds.peak
            lrTrough = bounds.trough
            lrFraction = frac
        }

        var momentum: Double? = nil
        if momentumEnabled {
            if momentumFollowsLRCycle {
                let low = bounds.momentumAtPeak
                let high = bounds.momentumAtTrough
                if let lrFraction, high >= low {
                    momentum = high - (high - low) * lrFraction
                }
            } else if momentumPeriodSteps > 0, momentumMax >= momentumMin {
                let frac = LRMomentumCycle.cycleFraction(step: cycleStep, period: momentumPeriodSteps,
                                                         count: momentumCount, invert: momentumInvert)
                momentum = momentumMin + (momentumMax - momentumMin) * frac
            }
        }
        return Values(learningRate: learningRate, momentum: momentum, lrPeak: lrPeak, lrTrough: lrTrough)
    }
}

// MARK: - Converting the previous single-decay schedule

/// The previous schedule's settings, as plain values: the cycle, its decay
/// envelope and the momentum-follow bounds, plus the warmup length the decay
/// was offset by. The one input to the converter, whatever carried the old
/// schedule (a trainer file, a session, a lineage snapshot, `parameters.json`
/// or `UserDefaults`).
struct LegacyLRSchedule: Sendable, Equatable {
    var lrEnabled: Bool
    var lrPeriodSteps: Int
    var lrCount: Int
    var lrMin: Double
    var lrMax: Double
    var lrInvert: Bool
    var momentumEnabled: Bool
    var momentumPeriodSteps: Int
    var momentumCount: Int
    var momentumMin: Double
    var momentumMax: Double
    var momentumInvert: Bool
    var lrPeakEnd: Double
    var lrTroughEnd: Double
    /// In cycle steps (trainer step − warmup); 0 = no decay.
    var decayHorizonSteps: Int
    var momentumFollowsLRCycle: Bool
    var momentumFollowStartLow: Double
    var momentumFollowStartHigh: Double
    var momentumFollowEndLow: Double
    var momentumFollowEndHigh: Double
    var lrWarmupSteps: Int
}

/// A legacy schedule the keyframes cannot express. The previous code ran
/// static momentum on exactly the steps where the follow bounds were
/// inverted; when that is only part of the run, no keyframe schedule (whose
/// bounds are always ordered) reproduces it. Refused rather than approximated
/// (owner decision Q-4).
struct LegacyLRScheduleConversionError: Error, Equatable, CustomStringConvertible {
    let description: String
}

extension LRKeyframeCycle {
    /// The previous schedule, converted so that every step feeds bit-identical
    /// LR and momentum (owner decision Q-4; Validation V-2).
    ///
    /// With warmup `w`, horizon `H` (cycle steps), the keyframes are
    /// `(0, start)` `(w, start)` `(w + H, end)` — the keyframe at `w` holds the
    /// start values through warmup, where the old decay fraction was 0 —
    /// collapsing to `(0, start)` `(H, end)` when `w == 0` (no duplicate step)
    /// and to `(0, start)` alone when `H == 0`.
    ///
    /// Settings the previous code treated as "no cycle" convert to what it
    /// did. A cycle whose LR pair was unusable (a bound not above 0, a peak
    /// below its trough at the start or the end) ran the static LR, and its
    /// follow momentum (which needs the LR phase) ran static too: it becomes
    /// `lrEnabled == false`. Follow bounds inverted at every step ran static
    /// momentum: they become `momentumEnabled == false`. Values that are
    /// inert after that, and invalid, are replaced by `inertSubstitute`'s so
    /// the result validates; inert values that are valid are kept as they
    /// are, so re-enabling a disabled cycle later gives back its schedule.
    ///
    /// `notes` says what was converted and why, for the caller's log line.
    static func converting(_ legacy: LegacyLRSchedule) throws -> (cycle: LRKeyframeCycle, notes: [String]) {
        var notes: [String] = []
        let w = max(0, legacy.lrWarmupSteps)
        let h = max(0, legacy.decayHorizonSteps)

        func lrPairValid(peak: Double, trough: Double) -> Bool {
            peak.isFinite && trough.isFinite && trough > 0 && peak >= trough
                && peak <= LRKeyframeSchedule.maximumLearningRate
        }
        func momentumValue(_ value: Double) -> Bool {
            value.isFinite && LRKeyframeSchedule.momentumRange.contains(value)
        }
        let startLRValid = lrPairValid(peak: legacy.lrMax, trough: legacy.lrMin)
        let endLRValid = h == 0 || lrPairValid(peak: legacy.lrPeakEnd, trough: legacy.lrTroughEnd)
        let lrValid = startLRValid && endLRValid

        let followValuesInRange = [legacy.momentumFollowStartLow, legacy.momentumFollowStartHigh]
            .allSatisfy(momentumValue)
            && (h == 0 || [legacy.momentumFollowEndLow, legacy.momentumFollowEndHigh].allSatisfy(momentumValue))
        let startOrdered = legacy.momentumFollowStartHigh >= legacy.momentumFollowStartLow
        let endOrdered = h == 0 || legacy.momentumFollowEndHigh >= legacy.momentumFollowEndLow
        let followValid = followValuesInRange && startOrdered && endOrdered

        var lrEnabled = legacy.lrEnabled
        var momentumEnabled = legacy.momentumEnabled

        if lrEnabled && legacy.lrPeriodSteps > 0 && !lrValid {
            // The old code returned nil here at every step: static LR, and no
            // phase for following momentum to read.
            lrEnabled = false
            notes.append("the LR cycle was inactive (unusable bounds: start \(legacy.lrMax)/\(legacy.lrMin)"
                         + (h > 0 ? ", end \(legacy.lrPeakEnd)/\(legacy.lrTroughEnd)" : "")
                         + "); converted as disabled")
        }
        let followDrivesMomentum = momentumEnabled && legacy.momentumFollowsLRCycle
            && lrEnabled && legacy.lrPeriodSteps > 0
        if followDrivesMomentum && !followValid {
            if followValuesInRange && !startOrdered && !endOrdered {
                // Inverted at the start and at the end; the bounds move
                // linearly between, so inverted at every step: the old code
                // fed the static momentum throughout.
                momentumEnabled = false
                notes.append("momentum following was inactive (bounds inverted at every step); converted as momentum cycling disabled")
            } else if followValuesInRange {
                throw LegacyLRScheduleConversionError(
                    description: "the momentum-follow bounds are inverted on only part of the run "
                        + "(start low \(legacy.momentumFollowStartLow) high \(legacy.momentumFollowStartHigh), "
                        + "end low \(legacy.momentumFollowEndLow) high \(legacy.momentumFollowEndHigh)); "
                        + "the previous code fed static momentum on those steps only, which no keyframe schedule can express")
            } else {
                throw LegacyLRScheduleConversionError(
                    description: "a momentum-follow bound is outside \(LRKeyframeSchedule.momentumRange)")
            }
        }

        let substitute = inertSubstitute
        let startPeak = lrValid ? legacy.lrMax : substitute.peak
        let startTrough = lrValid ? legacy.lrMin : substitute.trough
        let endPeak = lrValid ? legacy.lrPeakEnd : substitute.peak
        let endTrough = lrValid ? legacy.lrTroughEnd : substitute.trough
        let startLow = followValid ? legacy.momentumFollowStartLow : substitute.momentumAtPeak
        let startHigh = followValid ? legacy.momentumFollowStartHigh : substitute.momentumAtTrough
        let endLow = followValid ? legacy.momentumFollowEndLow : substitute.momentumAtPeak
        let endHigh = followValid ? legacy.momentumFollowEndHigh : substitute.momentumAtTrough
        if !lrValid {
            notes.append("inert LR bounds replaced by \(substitute.peak)/\(substitute.trough)")
        }
        if !followValid {
            notes.append("inert momentum-follow bounds replaced by \(substitute.momentumAtPeak)/\(substitute.momentumAtTrough)")
        }

        let start = { (step: Int) in
            LRKeyframe(step: step, peak: startPeak, trough: startTrough, momentumAtPeak: startLow, momentumAtTrough: startHigh)
        }
        let keyframes: [LRKeyframe]
        if h == 0 {
            keyframes = [start(0)]
        } else {
            let end = LRKeyframe(step: w + h, peak: endPeak, trough: endTrough, momentumAtPeak: endLow, momentumAtTrough: endHigh)
            keyframes = w == 0 ? [start(0), end] : [start(0), start(w), end]
        }
        let schedule = try LRKeyframeSchedule(keyframes: keyframes).validated()

        let cycle = LRKeyframeCycle(
            lrEnabled: lrEnabled, lrPeriodSteps: legacy.lrPeriodSteps, lrCount: legacy.lrCount,
            lrInvert: legacy.lrInvert, lrKeyframes: schedule,
            momentumEnabled: momentumEnabled, momentumPeriodSteps: legacy.momentumPeriodSteps,
            momentumCount: legacy.momentumCount, momentumMin: legacy.momentumMin, momentumMax: legacy.momentumMax,
            momentumInvert: legacy.momentumInvert, momentumFollowsLRCycle: legacy.momentumFollowsLRCycle
        )
        return (cycle, notes)
    }

    /// The values that stand in for invalid ones the converted schedule never
    /// reads (a disabled channel's numbers): the previous defaults' start
    /// values, which validate.
    static let inertSubstitute = LRKeyframe(step: 0, peak: 0.1, trough: 0.001, momentumAtPeak: 0.85, momentumAtTrough: 0.95)
}

// MARK: - Sampling the schedule for display

/// The schedule over a range of trainer steps, bucketed for a chart or a
/// table: per bucket the bounds at its first step and the minimum, maximum
/// and last value of the fed LR and momentum across its steps.
///
/// Min / max per bucket is what keeps the oscillation visible when one pixel
/// spans hundreds of steps: a plain every-Nth sample aliases the cosine and
/// can draw a flat line through a schedule that swings tenfold. The values
/// come from a caller-supplied function of the step — in the app, the fed
/// values (`LRMomentumCycleReadout`), so the picture is what the trainer is
/// fed, warmup and √batch included, with no formula of its own.
struct LRScheduleSample: Sendable, Equatable {
    struct Point: Sendable, Equatable {
        let learningRate: Double
        let momentum: Double
        let lrPeak: Double?
        let lrTrough: Double?
    }

    struct Bucket: Sendable, Equatable {
        let firstStep: Int
        let lastStep: Int
        let lrPeak: Double?
        let lrTrough: Double?
        let lrMin: Double
        let lrMax: Double
        let lrLast: Double
        let momentumMin: Double
        let momentumMax: Double
        let momentumLast: Double
    }

    let buckets: [Bucket]

    /// Samples every step in `range` (no step is skipped, so no extreme is
    /// missed) into at most `bucketCount` buckets of equal width (the last may
    /// be shorter). `isCancelled` is polled once per bucket, so an editor can
    /// abandon a stale sampling when the draft changes.
    static func sample(steps range: ClosedRange<Int>, bucketCount: Int,
                       isCancelled: () -> Bool = { false },
                       valueAt: (Int) -> Point) -> LRScheduleSample? {
        precondition(bucketCount > 0, "LRScheduleSample needs at least one bucket")
        let total = range.upperBound - range.lowerBound + 1
        let width = max(1, (total + bucketCount - 1) / bucketCount)
        var buckets: [Bucket] = []
        buckets.reserveCapacity(min(bucketCount, total))
        var first = range.lowerBound
        while first <= range.upperBound {
            if isCancelled() { return nil }
            let last = min(range.upperBound, first + width - 1)
            let head = valueAt(first)
            var lrMin = head.learningRate, lrMax = head.learningRate, lrLast = head.learningRate
            var momMin = head.momentum, momMax = head.momentum, momLast = head.momentum
            if last > first {
                for step in (first + 1)...last {
                    let p = valueAt(step)
                    lrMin = min(lrMin, p.learningRate)
                    lrMax = max(lrMax, p.learningRate)
                    lrLast = p.learningRate
                    momMin = min(momMin, p.momentum)
                    momMax = max(momMax, p.momentum)
                    momLast = p.momentum
                }
            }
            buckets.append(Bucket(firstStep: first, lastStep: last, lrPeak: head.lrPeak, lrTrough: head.lrTrough,
                                  lrMin: lrMin, lrMax: lrMax, lrLast: lrLast,
                                  momentumMin: momMin, momentumMax: momMax, momentumLast: momLast))
            first = last + 1
        }
        return LRScheduleSample(buckets: buckets)
    }
}
