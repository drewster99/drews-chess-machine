import Foundation

/// Repeating cyclical schedule for the optimizer's learning rate and Polyak
/// momentum, driven purely by the trainer's global step number.
///
/// This is the implementation of section 3 ("Cyclical LR + inverse-coupled
/// momentum") of `TRAINING_DYNAMICS_PLAN.md` — a Leslie-Smith-style super-
/// convergence lever (arXiv 1708.07120 / 1803.09820) adapted to open-ended
/// self-play, where there is no epoch and no known total training length.
///
/// Design properties, all load-bearing:
///
/// - **Phase is a pure function of the global step, offset by warmup.** The
///   cycle begins only once LR warmup finishes: `cycleStep(completedTrainSteps:
///   lrWarmupSteps:)` maps the trainer's global step onto the cycle's own step
///   axis, and `phase = (cycleStep mod period) / period`. During warmup the
///   cycle is held at its step-0 value, so the warmup ramp climbs to exactly
///   the cycle's starting value and hands off without a discontinuity. Had
///   the cycle started with warmup instead, an inverted cycle (which starts at
///   its maximum) would already be descending while warmup was still ramping
///   up, and warmup would never reach the cycle's peak. Both channels use the
///   same offset so an inverse-coupled LR/momentum pair stays in phase. The
///   schedule still carries *no state of its own*: the trainer's
///   `completedTrainSteps` is persisted, so on resume the phase continues
///   with no discontinuity.
///
/// - **Repeating cycles, not 1cycle.** 1cycle needs a known total length to
///   place its single cycle + annihilation tail; self-play has neither. These
///   are repeating cycles used as a plateau-breaking lever.
///
/// - **Cosine up-then-down.** `frac = 0.5·(1 − cos(2π·phase))` sits at `min`
///   at the period boundaries and reaches `max` at the midpoint, smoothly
///   (no triangular corner). The `invert` flag flips the waveform (`1 − frac`),
///   which is how momentum is made inverse to LR — high LR ↔ low momentum,
///   per Smith — when run at an equal period.
///
/// - **LR interpolates geometrically, momentum linearly.** LR's effect is
///   ~scale-invariant, so log-space interpolation spends equal time per
///   multiplicative octave (`lrMin · (lrMax/lrMin)^frac`). Momentum's useful
///   band (~0.85–0.95) is modest, so a plain linear lerp is fine.
///
/// - **Absolute endpoints.** The endpoints are absolute LR / momentum values,
///   not multipliers over a base. The caller composes the cycled LR with the
///   existing warmup × √batch multipliers (`effectiveLR = cycledBaseLR ·
///   warmupMul · √batchMul`); enabling LR cycling therefore overrides the
///   static base-LR schedule.
///
/// - **`count` completion freezes at the cycle boundary.** With `count == 0`
///   the cycle repeats forever (the open-ended default). With `count != 0`,
///   once `step / period ≥ count` the phase is clamped to 0 (the boundary),
///   which — respecting `invert` — leaves LR at `lrMin` and momentum at
///   `momentumMax`, i.e. the low-LR / high-momentum converged regime. The
///   count is measured on the cycle's own step axis, i.e. from the end of
///   warmup, so warmup never consumes any of the requested cycles.
///
/// - **Decaying envelope.** The peak and trough the LR swings between each
///   decay geometrically from a start value (`lrMax` / `lrMin`) to an end
///   value over a horizon measured on the cycle's own step axis, then hold at
///   the end values while cycling continues. Open-ended self-play still has
///   no known end, but a long run wants its LR to come down over time the way
///   a finite schedule would; the envelope provides that without giving up
///   the plateau-breaking swings. A zero horizon disables the decay.
///
/// - **Momentum can follow the LR cycle.** Rather than keeping a separate
///   momentum period and invert flag in agreement with the LR cycle by hand,
///   the follow mode reads the LR cycle's own phase and inverts it, so the
///   inverse coupling holds by construction — including after a period edit.
///
/// The struct is `Sendable` + `Equatable` and the math is pure, so it is
/// carried lock-free across the trainer's main/off-main boundary (in a
/// `SyncBox`) and is exercised directly by `LRMomentumCycleTests`.
struct LRMomentumCycle: Sendable, Equatable, Codable {

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
    var envelope: LRMomentumCycleEnvelope = .noDecay

    enum CodingKeys: String, CodingKey {
        case lrEnabled, lrPeriodSteps, lrCount, lrMin, lrMax, lrInvert
        case momentumEnabled, momentumPeriodSteps, momentumCount, momentumMin, momentumMax, momentumInvert
    }

    /// The all-off configuration: the trainer's value before a session pushes
    /// the configured cycle onto it. Both channels are off, so the endpoint
    /// values are inert. They are a fixed, sane set (the unit tests build on
    /// them) and deliberately do not track the parameter defaults, which live
    /// only on the `@TrainingParameter` declarations.
    static let disabled = LRMomentumCycle(
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

@MainActor
extension TrainingParameters {
    /// The current LR/momentum cycle configuration, read live from the
    /// singleton. Delegates to the snapshot so there is exactly one mapping
    /// from parameters to `LRMomentumCycle`; reading it touches every stored
    /// parameter, which is what lets `ControlSideEffectsProbe` observe it with
    /// a single `.onChange`.
    var lrMomentumCycle: LRMomentumCycle {
        snapshot().lrMomentumCycle
    }

    /// The decay envelope + momentum-follow configuration, read live from the
    /// singleton. Also persisted on its own in session files (see
    /// `LRMomentumCycle.envelope` for why it is not part of the cycle's
    /// encoded form).
    var lrMomentumCycleEnvelope: LRMomentumCycleEnvelope {
        snapshot().lrMomentumCycleEnvelope
    }
}

extension TrainingParametersSnapshot {
    /// The LR/momentum cycle these parameters describe, envelope included —
    /// the single mapping from the `lr_cycle_*` / `momentum_cycle_*` /
    /// `momentum_follow*` parameters to the struct the trainer consumes.
    /// Pushed onto a trainer via `TrainerHyperparameters.apply(to:)`.
    var lrMomentumCycle: LRMomentumCycle {
        LRMomentumCycle(
            lrEnabled: lrCycleEnabled,
            lrPeriodSteps: lrCyclePeriodSteps,
            lrCount: lrCycleCount,
            lrMin: lrCycleMin,
            lrMax: lrCycleMax,
            lrInvert: lrCycleInvert,
            momentumEnabled: momentumCycleEnabled,
            momentumPeriodSteps: momentumCyclePeriodSteps,
            momentumCount: momentumCycleCount,
            momentumMin: momentumCycleMin,
            momentumMax: momentumCycleMax,
            momentumInvert: momentumCycleInvert,
            envelope: lrMomentumCycleEnvelope
        )
    }

    /// These parameters with the schedule a trainer runs under written in:
    /// `lr_warmup_steps` and every `lr_cycle_*` / `momentum_cycle_*` /
    /// `momentum_follow*` key, taken from `schedule` — the exact inverse of
    /// `lrMomentumCycle` / `lrMomentumCycleEnvelope` above, so
    /// `adoptingSchedule(s).lrMomentumCycle == s.lrMomentumCycle` for every
    /// schedule. Every other key is unchanged.
    ///
    /// Why: an exact resume trains under the checkpoint's own schedule
    /// (`TrainerHyperparameters.adoptingSchedule`), whatever the run was
    /// configured with. The lineage record's parameter snapshot must say the
    /// same, or a file resumed under a parameters file with another cycle
    /// period records that period while the weights were trained at the
    /// checkpoint's. Composing the record from this one
    /// value is what keeps the snapshot and the file's flat `trainer_*` keys
    /// equal by construction.
    ///
    /// The values are written as they are, **never validated or clamped**
    /// against today's declared ranges (`TrainingParametersSnapshot.replacing`):
    /// the checkpoint's value is what ran, even when a range has narrowed
    /// since it was saved.
    func adoptingSchedule(_ schedule: TrainerScheduleState) -> TrainingParametersSnapshot {
        let cycle = schedule.lrMomentumCycle
        let envelope = cycle.envelope
        return self
            .replacing(LRWarmupSteps.self, with: schedule.lrWarmupSteps)
            .replacing(LRCycleEnabled.self, with: cycle.lrEnabled)
            .replacing(LRCyclePeriodSteps.self, with: cycle.lrPeriodSteps)
            .replacing(LRCycleCount.self, with: cycle.lrCount)
            .replacing(LRCycleMin.self, with: cycle.lrMin)
            .replacing(LRCycleMax.self, with: cycle.lrMax)
            .replacing(LRCycleInvert.self, with: cycle.lrInvert)
            .replacing(MomentumCycleEnabled.self, with: cycle.momentumEnabled)
            .replacing(MomentumCyclePeriodSteps.self, with: cycle.momentumPeriodSteps)
            .replacing(MomentumCycleCount.self, with: cycle.momentumCount)
            .replacing(MomentumCycleMin.self, with: cycle.momentumMin)
            .replacing(MomentumCycleMax.self, with: cycle.momentumMax)
            .replacing(MomentumCycleInvert.self, with: cycle.momentumInvert)
            .replacing(LRCyclePeakEnd.self, with: envelope.lrPeakEnd)
            .replacing(LRCycleTroughEnd.self, with: envelope.lrTroughEnd)
            .replacing(LRCycleDecayHorizonSteps.self, with: envelope.decayHorizonSteps)
            .replacing(MomentumFollowsLRCycle.self, with: envelope.momentumFollowsLRCycle)
            .replacing(MomentumFollowStartLow.self, with: envelope.momentumFollowStartLow)
            .replacing(MomentumFollowStartHigh.self, with: envelope.momentumFollowStartHigh)
            .replacing(MomentumFollowEndLow.self, with: envelope.momentumFollowEndLow)
            .replacing(MomentumFollowEndHigh.self, with: envelope.momentumFollowEndHigh)
    }

    /// The decay envelope + momentum-follow configuration these parameters
    /// describe.
    var lrMomentumCycleEnvelope: LRMomentumCycleEnvelope {
        LRMomentumCycleEnvelope(
            lrPeakEnd: lrCyclePeakEnd,
            lrTroughEnd: lrCycleTroughEnd,
            decayHorizonSteps: lrCycleDecayHorizonSteps,
            momentumFollowsLRCycle: momentumFollowsLRCycle,
            momentumFollowStartLow: momentumFollowStartLow,
            momentumFollowStartHigh: momentumFollowStartHigh,
            momentumFollowEndLow: momentumFollowEndLow,
            momentumFollowEndHigh: momentumFollowEndHigh
        )
    }
}

/// The part of the LR/momentum schedule that changes over the long run: where
/// the LR cycle's peak and trough decay to, over how many cycle steps, and
/// whether (and between which bounds) momentum follows the LR cycle. See
/// `LRMomentumCycle.values(forCycleStep:)` for the math.
struct LRMomentumCycleEnvelope: Sendable, Equatable, Codable {
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
    static let noDecay = LRMomentumCycleEnvelope(
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

/// Log-line renderings of the schedule, shared by every tag that reports it
/// (`[PARAM]`, `[RESUME-PARAM]`, `[STATS]`) so the same configuration always
/// reads the same way wherever it is grepped.
enum LRMomentumCycleLogFormat {
    /// The decay envelope and momentum-follow configuration, e.g. for the
    /// `lr_momentum_cycle` audit lines.
    static func envelopeDescription(_ envelope: LRMomentumCycleEnvelope) -> String {
        let decayPart = envelope.decayHorizonSteps > 0
            ? String(
                format: "decay=[peak->%.2e trough->%.2e over %dst]",
                envelope.lrPeakEnd, envelope.lrTroughEnd, envelope.decayHorizonSteps
            )
            : "decay=off"
        let followPart = envelope.momentumFollowsLRCycle
            ? String(
                format: "momFollow=[low %.3f->%.3f high %.3f->%.3f]",
                envelope.momentumFollowStartLow, envelope.momentumFollowEndLow,
                envelope.momentumFollowStartHigh, envelope.momentumFollowEndHigh
            )
            : "momFollow=off"
        return "\(decayPart) \(followPart)"
    }

    /// The whole schedule configuration — both channels plus the envelope —
    /// e.g. `lr=[trough 1.00e-03,peak 1.00e-01]^20000st cnt=0 inv=true
    /// mom=off decay=… momFollow=…`. A disabled channel renders as `lr=off` /
    /// `mom=off`, meaning the trainer uses the static value. Shared by the
    /// `[PARAM] lr_momentum_cycle` audit line and the CLI runners' startup
    /// cycle lines.
    static func cycleDescription(_ cycle: LRMomentumCycle) -> String {
        let lrPart = cycle.lrEnabled
            ? "lr=[trough \(String(format: "%.2e", cycle.lrMin)),peak \(String(format: "%.2e", cycle.lrMax))]^\(cycle.lrPeriodSteps)st cnt=\(cycle.lrCount) inv=\(cycle.lrInvert)"
            : "lr=off"
        let momPart = cycle.momentumEnabled
            ? "mom=[\(String(format: "%.2f", cycle.momentumMin)),\(String(format: "%.2f", cycle.momentumMax))]^\(cycle.momentumPeriodSteps)st cnt=\(cycle.momentumCount) inv=\(cycle.momentumInvert)"
            : "mom=off"
        return "\(lrPart) \(momPart) \(envelopeDescription(cycle.envelope))"
    }

    /// The current LR envelope bounds for the `[STATS]` `lr=` field, e.g.
    /// `[pk=…,tr=…]`. Empty when the LR cycle is not driving the LR.
    static func envelopeBounds(_ values: LRMomentumCycle.Values) -> String {
        guard let peak = values.lrPeak, let trough = values.lrTrough else { return "" }
        return String(format: "[pk=%.1e,tr=%.1e]", peak, trough)
    }
}
