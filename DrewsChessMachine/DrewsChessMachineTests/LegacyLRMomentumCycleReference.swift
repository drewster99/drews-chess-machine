//
//  LegacyLRMomentumCycleReference.swift
//  DrewsChessMachineTests
//
//  The LR/momentum cycle as it was before LR keyframes (commit babcf3ca),
//  copied verbatim and renamed, so the conversion of an old schedule can be
//  checked bit for bit against the code that used to feed the trainer
//  (documentation/plans-active/LR_KEYFRAMES_PLAN.md, owner decision Q-4,
//  Validation V-2). Never edit its arithmetic: it is the reference.
//

import Foundation
@testable import DrewsChessMachine

struct LegacyLRMomentumCycleReference: Sendable, Equatable {

    // MARK: LR cycle (geometric interpolation between absolute endpoints)

    var lrEnabled: Bool
    /// Full up-then-down period, in optimizer steps.
    var lrPeriodSteps: Int
    /// Number of cycles before freezing at the boundary. 0 = unbounded.
    var lrCount: Int
    /// Absolute LR at the cycle's trough (the period boundaries when
    /// `lrInvert == false`) at the start of the decay horizon — the trough's
    /// start value. Must be > 0 for geometric interpolation.
    var lrMin: Double
    /// Absolute LR at the cycle's peak (the period midpoint when
    /// `lrInvert == false`) at the start of the decay horizon — the peak's
    /// start value.
    var lrMax: Double
    /// Flip the waveform so the cycle starts at `lrMax` instead of `lrMin`.
    var lrInvert: Bool

    // MARK: Momentum cycle (linear interpolation between absolute endpoints)

    var momentumEnabled: Bool
    var momentumPeriodSteps: Int
    var momentumCount: Int
    var momentumMin: Double
    var momentumMax: Double
    var momentumInvert: Bool

    // MARK: Decay envelope + momentum-follows-LR

    /// The decay envelope and the momentum-follow mode. Deliberately left
    /// out of this struct's `Codable` form (see `CodingKeys`): session files
    /// written before the envelope existed carry an `lrMomentumCycle` object
    /// without it, and adding a required key would make those sessions fail
    /// to decode outright. The envelope is persisted as its own Optional
    /// field on `SessionCheckpointState` instead, and resume re-attaches it.
    /// Its default is the no-decay, no-follow configuration, under which this
    /// struct behaves exactly as it did before the envelope existed.
    var envelope: LegacyLRMomentumCycleEnvelopeReference = .noDecay


    /// The all-off configuration: the trainer's value before a session pushes
    /// the configured cycle onto it. Both channels are off, so the endpoint
    /// values are inert. They are a fixed, sane set (the unit tests build on
    /// them) and deliberately do not track the parameter defaults, which live
    /// only on the `@TrainingParameter` declarations.
    static let disabledReference = LegacyLRMomentumCycleReference(
        lrEnabled: false,
        lrPeriodSteps: 2000,
        lrCount: 0,
        lrMin: 0.001,
        lrMax: 0.03,
        lrInvert: false,
        momentumEnabled: false,
        momentumPeriodSteps: 2000,
        momentumCount: 0,
        momentumMin: 0.85,
        momentumMax: 0.95,
        momentumInvert: true
    )

    /// Cosine up-then-down position in `[0, 1]` for cycle step `step`, honoring the
    /// `count` completion freeze (phase clamped to 0 after `count` cycles)
    /// and the `invert` waveform flip. This is the shared shape both the LR
    /// and momentum channels map onto their own endpoints.
    static func cycleFraction(step: Int, period: Int, count: Int, invert: Bool) -> Double {
        // A non-positive period has no meaningful cycle; report the boundary
        // value (frac 0, flipped by `invert`) so callers degrade gracefully.
        guard period > 0 else { return invert ? 1.0 : 0.0 }
        let clampedStep = max(0, step)
        let effectiveStep: Int
        if count > 0, clampedStep / period >= count {
            // Completed `count` cycles → freeze at the boundary (phase 0).
            effectiveStep = 0
        } else {
            effectiveStep = clampedStep % period
        }
        let phase = Double(effectiveStep) / Double(period)
        let frac = 0.5 * (1.0 - cos(2.0 * Double.pi * phase))
        return invert ? (1.0 - frac) : frac
    }

    /// The cycle's own step for a given trainer global step. The cycle starts
    /// when LR warmup ends, so every step inside warmup maps to cycle step 0
    /// (the warmup ramp then targets the cycle's starting value) and every
    /// later step is shifted back by the warmup length. This is the single
    /// place that relates the global step to the cycle's phase; every LR and
    /// momentum reader (the SGD feed, the status readouts, the log lines) goes
    /// through it so they can never disagree about where in the cycle a step
    /// sits.
    static func cycleStep(completedTrainSteps: Int, lrWarmupSteps: Int) -> Int {
        max(0, completedTrainSteps - max(0, lrWarmupSteps))
    }

    /// Cycled learning rate at trainer global step `completedTrainSteps`,
    /// with the cycle offset to begin after `lrWarmupSteps`. `nil` under the
    /// same conditions as `learningRate(forStep:)`. The warmup multiplier is
    /// NOT applied here — callers compose it (and √batch) on top.
    func learningRate(completedTrainSteps: Int, lrWarmupSteps: Int) -> Double? {
        learningRate(forStep: Self.cycleStep(completedTrainSteps: completedTrainSteps, lrWarmupSteps: lrWarmupSteps))
    }

    /// Cycled momentum at trainer global step `completedTrainSteps`, offset
    /// by the same warmup length as the LR channel so the two stay in phase.
    /// During warmup this is the cycle's step-0 momentum.
    func momentum(completedTrainSteps: Int, lrWarmupSteps: Int) -> Double? {
        momentum(forStep: Self.cycleStep(completedTrainSteps: completedTrainSteps, lrWarmupSteps: lrWarmupSteps))
    }

    /// Effective learning rate at cycle step `step` (already offset past
    /// warmup — see `cycleStep`), or `nil` when LR cycling is
    /// inactive or misconfigured — the caller then falls back to the static
    /// base learning rate. Geometric (log-space) interpolation requires
    /// `lrMin > 0` and `lrMax >= lrMin`; otherwise `nil` is returned.
    func learningRate(forStep step: Int) -> Double? {
        values(forCycleStep: step).learningRate
    }

    /// Effective Polyak momentum at cycle step `step` (already offset past
    /// warmup — see `cycleStep`), or `nil` when momentum cycling
    /// is inactive or misconfigured — the caller then falls back to the static
    /// momentum coefficient. Linear interpolation between the endpoints.
    ///
    /// Requires `momentumMax >= momentumMin`: an inverted endpoint pair would
    /// silently run the schedule backwards, which is exactly what the explicit
    /// `momentumInvert` flag exists to express. Rather than honor an accidental
    /// reversal, fall back to the static coefficient (mirroring the LR channel's
    /// `lrMax >= lrMin` guard).
    ///
    /// When `envelope.momentumFollowsLRCycle` is on, the separate momentum
    /// cycle's period, count, endpoints and invert flag are ignored and the
    /// channel is driven by the LR cycle instead — see `values(forCycleStep:)`.
    func momentum(forStep step: Int) -> Double? {
        values(forCycleStep: step).momentum
    }

    /// Everything the schedule says about one cycle step. `lrPeak` /
    /// `lrTrough` are the (possibly decayed) envelope bounds the LR is
    /// swinging between at that step; they exist for the log lines, so a
    /// reader can see where in the decay the run sits, and are `nil`
    /// exactly when `learningRate` is.
    struct Values: Sendable, Equatable {
        let learningRate: Double?
        let momentum: Double?
        let lrPeak: Double?
        let lrTrough: Double?
    }

    /// Position along the decay horizon in `[0, 1]`: the fraction of the
    /// horizon elapsed at cycle step `step`, held at 1 once the horizon is
    /// passed so the envelope stays at its end values and cycling continues
    /// between them. A non-positive horizon means "no decay", which pins the
    /// envelope at its start values forever — the pre-envelope behavior.
    static func decayFraction(step: Int, horizonSteps: Int) -> Double {
        guard horizonSteps > 0 else { return 0.0 }
        return Double(min(max(0, step), horizonSteps)) / Double(horizonSteps)
    }

    /// The single source of truth for the schedule: learning rate and
    /// momentum (plus the LR envelope bounds) at cycle step `step`, which is
    /// already offset past warmup (see `cycleStep`). Every other accessor
    /// reads through here so the SGD feed, the status readouts and the logs
    /// can never disagree.
    ///
    /// LR: the cycle's cosine fraction interpolates geometrically between
    /// this step's trough and peak. Each bound decays geometrically from its
    /// start value (`lrMin` for the trough, `lrMax` for the peak) to its end
    /// value in the envelope across the decay horizon — log-space, for the
    /// same scale-invariance reason the in-cycle interpolation is log-space.
    /// `nil` when LR cycling is off or any bound pair is unusable (a bound
    /// not > 0, or a peak below its trough), so the caller falls back to the
    /// static base LR rather than running a nonsensical schedule.
    ///
    /// Momentum, following the LR cycle: the same cosine fraction, inverted,
    /// so momentum is at its low bound exactly where LR is at its peak and at
    /// its high bound where LR bottoms out — Smith's inverse coupling without
    /// having to keep two periods and two invert flags in agreement by hand.
    /// Its bounds move linearly (momentum's useful band is narrow) across the
    /// same horizon. Following requires an active LR cycle: without one there
    /// is no phase to follow, so the channel reports `nil` (static momentum).
    ///
    /// Momentum, not following: the independent momentum cycle, unchanged.
    func values(forCycleStep step: Int) -> Values {
        let decay = Self.decayFraction(step: step, horizonSteps: envelope.decayHorizonSteps)

        var learningRate: Double? = nil
        var lrPeak: Double? = nil
        var lrTrough: Double? = nil
        var lrFraction: Double? = nil
        let decayEndsUsable = envelope.decayHorizonSteps <= 0
            || (envelope.lrPeakEnd > 0 && envelope.lrTroughEnd > 0 && envelope.lrPeakEnd >= envelope.lrTroughEnd)
        if lrEnabled, lrPeriodSteps > 0, lrMin > 0, lrMax >= lrMin, decayEndsUsable {
            let peak = lrMax * pow(envelope.lrPeakEnd / lrMax, decay)
            let trough = lrMin * pow(envelope.lrTroughEnd / lrMin, decay)
            let frac = Self.cycleFraction(step: step, period: lrPeriodSteps, count: lrCount, invert: lrInvert)
            learningRate = trough * pow(peak / trough, frac)
            lrPeak = peak
            lrTrough = trough
            lrFraction = frac
        }

        var momentum: Double? = nil
        if momentumEnabled {
            if envelope.momentumFollowsLRCycle {
                let low = envelope.momentumFollowStartLow
                    + (envelope.momentumFollowEndLow - envelope.momentumFollowStartLow) * decay
                let high = envelope.momentumFollowStartHigh
                    + (envelope.momentumFollowEndHigh - envelope.momentumFollowStartHigh) * decay
                if let lrFraction, high >= low {
                    momentum = high - (high - low) * lrFraction
                }
            } else if momentumPeriodSteps > 0, momentumMax >= momentumMin {
                let frac = Self.cycleFraction(step: step, period: momentumPeriodSteps, count: momentumCount, invert: momentumInvert)
                momentum = momentumMin + (momentumMax - momentumMin) * frac
            }
        }

        return Values(learningRate: learningRate, momentum: momentum, lrPeak: lrPeak, lrTrough: lrTrough)
    }

    /// `values(forCycleStep:)` at trainer global step `completedTrainSteps`,
    /// with the warmup offset applied.
    func values(completedTrainSteps: Int, lrWarmupSteps: Int) -> Values {
        values(forCycleStep: Self.cycleStep(completedTrainSteps: completedTrainSteps, lrWarmupSteps: lrWarmupSteps))
    }

    /// True when either channel is actively cycling — used to decide whether
    /// to emit the effective values into `[STATS]`.
    var isActive: Bool { lrEnabled || momentumEnabled }
}


struct LegacyLRMomentumCycleEnvelopeReference: Sendable, Equatable {
    /// LR peak at and after the end of the decay horizon.
    var lrPeakEnd: Double
    /// LR trough at and after the end of the decay horizon.
    var lrTroughEnd: Double
    /// Length of the decay, in cycle steps (i.e. counted from the end of
    /// warmup). Zero disables the decay: the envelope stays at its start
    /// values forever.
    var decayHorizonSteps: Int
    /// When on, momentum is driven by the LR cycle's phase (inverted) instead
    /// of the independent momentum cycle.
    var momentumFollowsLRCycle: Bool
    /// Follow-mode momentum at the LR peak, at the start of the horizon.
    var momentumFollowStartLow: Double
    /// Follow-mode momentum at the LR trough, at the start of the horizon.
    var momentumFollowStartHigh: Double
    /// Follow-mode momentum at the LR peak, at and after the horizon's end.
    var momentumFollowEndLow: Double
    /// Follow-mode momentum at the LR trough, at and after the horizon's end.
    var momentumFollowEndHigh: Double

    /// No decay and no momentum following: the configuration under which
    /// `LRMomentumCycle` computes exactly what it did before the envelope
    /// existed. The end values are inert while the horizon is zero and
    /// following is off; they mirror the parameter defaults so a
    /// freshly-enabled-but-unedited envelope is sane.
    static let noDecay = LegacyLRMomentumCycleEnvelopeReference(
        lrPeakEnd: 1.0e-4,
        lrTroughEnd: 1.0e-6,
        decayHorizonSteps: 0,
        momentumFollowsLRCycle: false,
        momentumFollowStartLow: 0.85,
        momentumFollowStartHigh: 0.95,
        momentumFollowEndLow: 0.90,
        momentumFollowEndHigh: 0.95
    )
}

