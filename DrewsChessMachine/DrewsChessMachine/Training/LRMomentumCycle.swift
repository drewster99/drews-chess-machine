import Foundation
import TrainingParametersMacroSupport

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
///   multiplicative octave (`trough · (peak/trough)^frac`). Momentum's useful
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
///   which — respecting `invert` — leaves LR at the trough and momentum at
///   `momentumMax`, i.e. the low-LR / high-momentum converged regime. The
///   count is measured on the cycle's own step axis, i.e. from the end of
///   warmup, so warmup never consumes any of the requested cycles.
///
/// - **Keyframed envelope.** The peak and trough the LR swings between come
///   from `lrKeyframes` (`LRKeyframeSchedule`), read at the **trainer** step:
///   geometric between keyframes, held after the last. A long run wants its
///   LR to come down over time the way a finite schedule would, often in
///   stages (a fast early decay, then a slow one, then a floor); keyframes
///   express that without giving up the plateau-breaking swings. They replaced
///   a single start → end decay over one horizon
///   (documentation/plans-active/LR_KEYFRAMES_PLAN.md); an old schedule
///   converts to keyframes exactly (`converting(_:)`). The phase stays on the
///   cycle step, so the cycle still starts after warmup.
///
/// - **Momentum can follow the LR cycle.** Rather than keeping a separate
///   momentum period and invert flag in agreement with the LR cycle by hand,
///   the follow mode reads the LR cycle's own phase and inverts it, so the
///   inverse coupling holds by construction — including after a period edit.
///   Its bounds (momentum at the LR peak and at the LR trough) ride on the
///   same keyframes, linear between them: one timeline, not two.
///
/// The struct is `Sendable` + `Equatable` and the math is pure, so it is
/// carried lock-free across the trainer's main/off-main boundary (in a
/// `SyncBox`) and is exercised directly by `LRMomentumCycleTests`.
struct LRMomentumCycle: Sendable, Equatable {

    // MARK: LR cycle

    var lrEnabled: Bool
    /// Full up-then-down period, in optimizer steps.
    var lrPeriodSteps: Int
    /// Number of cycles before freezing at the boundary. 0 = unbounded.
    var lrCount: Int
    /// Flip the waveform so the cycle starts at its peak instead of its trough.
    var lrInvert: Bool
    /// The peak and trough over the run (and the follow-momentum bounds).
    var lrKeyframes: LRKeyframeSchedule

    // MARK: Momentum cycle (linear interpolation between absolute endpoints)

    var momentumEnabled: Bool
    var momentumPeriodSteps: Int
    var momentumCount: Int
    var momentumMin: Double
    var momentumMax: Double
    var momentumInvert: Bool
    /// When on, momentum is driven by the LR cycle's phase (inverted), between
    /// the keyframes' momentum at the LR peak and at the LR trough, instead of
    /// by the independent momentum cycle above.
    var momentumFollowsLRCycle: Bool

    /// The all-off configuration: the trainer's value before a session pushes
    /// the configured cycle onto it (and the behavior fingerprint's).
    /// Both channels are off, so every other value is inert; they are a fixed,
    /// sane set and deliberately do not track the parameter defaults, which
    /// live only on the `@TrainingParameter` declarations.
    static let disabled = LRMomentumCycle(
        lrEnabled: false,
        lrPeriodSteps: 2000,
        lrCount: 0,
        lrInvert: false,
        lrKeyframes: LRKeyframeSchedule(keyframes: [
            LRKeyframe(step: 0, peak: 0.03, trough: 0.001, momentumAtPeak: 0.85, momentumAtTrough: 0.95),
        ]),
        momentumEnabled: false,
        momentumPeriodSteps: 2000,
        momentumCount: 0,
        momentumMin: 0.85,
        momentumMax: 0.95,
        momentumInvert: true,
        momentumFollowsLRCycle: false
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

    /// Everything the schedule says about one trainer step. `lrPeak` /
    /// `lrTrough` are the bounds the LR swings between there; they exist for
    /// the log lines, so a reader can see where in the schedule the run sits,
    /// and are `nil` exactly when `learningRate` is.
    struct Values: Sendable, Equatable {
        let learningRate: Double?
        let momentum: Double?
        let lrPeak: Double?
        let lrTrough: Double?
    }

    /// The single source of truth for the schedule at trainer step
    /// `completedTrainSteps`. Every other reader goes through here so the SGD
    /// feed, the status readouts and the logs can never disagree.
    ///
    /// LR: the cycle's cosine fraction (from the cycle step: trainer step −
    /// warmup) interpolates geometrically between this step's trough and peak
    /// (from the keyframes, at the trainer step). `nil` when LR cycling is off
    /// or the period is not positive, so the caller uses the static base LR.
    ///
    /// Momentum, following the LR cycle: the same fraction, inverted, so
    /// momentum is at its low bound exactly where LR peaks and at its high
    /// bound where LR bottoms out. Following needs an active LR cycle: without
    /// one there is no phase to follow, so the channel reports `nil` (static
    /// momentum). The `high >= low` guard is unreachable for a validated
    /// schedule and stays so an unvalidated one can never run backwards.
    ///
    /// Momentum, not following: the independent momentum cycle.
    ///
    /// The gates, their order and the arithmetic are the previous
    /// single-decay cycle's; a converted old schedule feeds bit-identical
    /// values (LRKeyframeScheduleTests).
    func values(completedTrainSteps: Int, lrWarmupSteps: Int) -> Values {
        let cycleStep = Self.cycleStep(completedTrainSteps: completedTrainSteps, lrWarmupSteps: lrWarmupSteps)
        let bounds = lrKeyframes.bounds(atTrainerStep: completedTrainSteps)

        var learningRate: Double? = nil
        var lrPeak: Double? = nil
        var lrTrough: Double? = nil
        var lrFraction: Double? = nil
        if lrEnabled, lrPeriodSteps > 0 {
            let frac = Self.cycleFraction(step: cycleStep, period: lrPeriodSteps, count: lrCount, invert: lrInvert)
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
                let frac = Self.cycleFraction(step: cycleStep, period: momentumPeriodSteps,
                                              count: momentumCount, invert: momentumInvert)
                momentum = momentumMin + (momentumMax - momentumMin) * frac
            }
        }
        return Values(learningRate: learningRate, momentum: momentum, lrPeak: lrPeak, lrTrough: lrTrough)
    }

    /// True when either channel is actively cycling — used to decide whether
    /// to emit the effective values into `[STATS]`.
    var isActive: Bool { lrEnabled || momentumEnabled }

    /// The schedule in behavior form: the fields that can change a value the
    /// trainer is fed. Two cycles with equal behavior forms feed the same LR
    /// and momentum at every step, whatever their inert fields hold (a
    /// disabled channel's numbers, the independent momentum cycle while
    /// momentum follows the LR). An exact resume compares these (owner
    /// decision Q-5), so an inert difference never costs a resume its
    /// exactness.
    var behaviorForm: LRMomentumCycle {
        var form = LRMomentumCycle.disabled
        let lrActive = lrEnabled && lrPeriodSteps > 0
        if lrActive {
            form.lrEnabled = true
            form.lrPeriodSteps = lrPeriodSteps
            form.lrCount = lrCount
            form.lrInvert = lrInvert
            if momentumEnabled && momentumFollowsLRCycle {
                form.lrKeyframes = lrKeyframes
            } else {
                // The keyframes' momentum pairs are read only by following
                // momentum; otherwise they are inert and must not count.
                let inert = LRMomentumCycle.disabled.lrKeyframes.keyframes[0]
                form.lrKeyframes = LRKeyframeSchedule(keyframes: lrKeyframes.keyframes.map {
                    LRKeyframe(step: $0.step, peak: $0.peak, trough: $0.trough,
                               momentumAtPeak: inert.momentumAtPeak, momentumAtTrough: inert.momentumAtTrough)
                })
            }
        }
        if momentumEnabled {
            if momentumFollowsLRCycle {
                // Following reads the LR phase and the keyframes' momentum pair.
                if lrActive {
                    form.momentumEnabled = true
                    form.momentumFollowsLRCycle = true
                }
            } else if momentumPeriodSteps > 0, momentumMax >= momentumMin {
                form.momentumEnabled = true
                form.momentumPeriodSteps = momentumPeriodSteps
                form.momentumCount = momentumCount
                form.momentumMin = momentumMin
                form.momentumMax = momentumMax
                form.momentumInvert = momentumInvert
            }
        }
        return form
    }
}

// MARK: - Persisted form

/// The one encoded form of a schedule: trainer files (`trainer_lr_schedule`),
/// session files (`lrSchedule`) and anything else that stores one. Versioned
/// by `schedule_format` so a later change can be told apart; a decode
/// validates the keyframes, so no stored schedule bypasses the validator.
extension LRMomentumCycle: Codable {
    static let currentScheduleFormat = 1

    enum CodingKeys: String, CodingKey {
        case scheduleFormat = "schedule_format"
        case lrEnabled = "lr_enabled"
        case lrPeriodSteps = "lr_period_steps"
        case lrCount = "lr_count"
        case lrInvert = "lr_invert"
        case lrKeyframes = "lr_keyframes"
        case momentumEnabled = "momentum_enabled"
        case momentumPeriodSteps = "momentum_period_steps"
        case momentumCount = "momentum_count"
        case momentumMin = "momentum_min"
        case momentumMax = "momentum_max"
        case momentumInvert = "momentum_invert"
        case momentumFollowsLRCycle = "momentum_follows_lr_cycle"
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let format = try c.decode(Int.self, forKey: .scheduleFormat)
        guard format == Self.currentScheduleFormat else {
            throw DecodingError.dataCorruptedError(
                forKey: .scheduleFormat, in: c,
                debugDescription: "LR schedule format \(format) is not one this build reads (\(Self.currentScheduleFormat))")
        }
        let keyframes = try c.decode(LRKeyframeSchedule.self, forKey: .lrKeyframes)
        do {
            try keyframes.validated()
        } catch {
            throw DecodingError.dataCorruptedError(forKey: .lrKeyframes, in: c, debugDescription: "\(error)")
        }
        self.init(
            lrEnabled: try c.decode(Bool.self, forKey: .lrEnabled),
            lrPeriodSteps: try c.decode(Int.self, forKey: .lrPeriodSteps),
            lrCount: try c.decode(Int.self, forKey: .lrCount),
            lrInvert: try c.decode(Bool.self, forKey: .lrInvert),
            lrKeyframes: keyframes,
            momentumEnabled: try c.decode(Bool.self, forKey: .momentumEnabled),
            momentumPeriodSteps: try c.decode(Int.self, forKey: .momentumPeriodSteps),
            momentumCount: try c.decode(Int.self, forKey: .momentumCount),
            momentumMin: try c.decode(Double.self, forKey: .momentumMin),
            momentumMax: try c.decode(Double.self, forKey: .momentumMax),
            momentumInvert: try c.decode(Bool.self, forKey: .momentumInvert),
            momentumFollowsLRCycle: try c.decode(Bool.self, forKey: .momentumFollowsLRCycle)
        )
    }

    func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(Self.currentScheduleFormat, forKey: .scheduleFormat)
        try c.encode(lrEnabled, forKey: .lrEnabled)
        try c.encode(lrPeriodSteps, forKey: .lrPeriodSteps)
        try c.encode(lrCount, forKey: .lrCount)
        try c.encode(lrInvert, forKey: .lrInvert)
        try c.encode(lrKeyframes, forKey: .lrKeyframes)
        try c.encode(momentumEnabled, forKey: .momentumEnabled)
        try c.encode(momentumPeriodSteps, forKey: .momentumPeriodSteps)
        try c.encode(momentumCount, forKey: .momentumCount)
        try c.encode(momentumMin, forKey: .momentumMin)
        try c.encode(momentumMax, forKey: .momentumMax)
        try c.encode(momentumInvert, forKey: .momentumInvert)
        try c.encode(momentumFollowsLRCycle, forKey: .momentumFollowsLRCycle)
    }
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
}

extension TrainingParametersSnapshot {
    /// The LR/momentum cycle these parameters describe — the single mapping
    /// from the `lr_cycle_*` / `momentum_cycle_*` / `momentum_follows_lr_cycle`
    /// parameters to the struct the trainer consumes. Pushed onto a trainer
    /// via `TrainerHyperparameters.apply(to:)`.
    var lrMomentumCycle: LRMomentumCycle {
        LRMomentumCycle(
            lrEnabled: lrCycleEnabled,
            lrPeriodSteps: lrCyclePeriodSteps,
            lrCount: lrCycleCount,
            lrInvert: lrCycleInvert,
            lrKeyframes: lrCycleKeyframes,
            momentumEnabled: momentumCycleEnabled,
            momentumPeriodSteps: momentumCyclePeriodSteps,
            momentumCount: momentumCycleCount,
            momentumMin: momentumCycleMin,
            momentumMax: momentumCycleMax,
            momentumInvert: momentumCycleInvert,
            momentumFollowsLRCycle: momentumFollowsLRCycle
        )
    }

    /// The ids `adoptingSchedule(_:)` writes: the warmup length and every
    /// cycle key. The schedule backstop of a trainer file
    /// (`SafetensorsModelIO.checkScheduleAgrees`) compares exactly these.
    static let scheduleKeyIDs: [String] = [
        LRWarmupSteps.id,
        LRCycleEnabled.id, LRCyclePeriodSteps.id, LRCycleCount.id, LRCycleInvert.id, LRCycleKeyframes.id,
        MomentumCycleEnabled.id, MomentumCyclePeriodSteps.id, MomentumCycleCount.id, MomentumCycleMin.id,
        MomentumCycleMax.id, MomentumCycleInvert.id, MomentumFollowsLRCycle.id,
    ]

    /// These parameters with the schedule a trainer runs under written in:
    /// `lr_warmup_steps` and every cycle key, taken from `schedule` — the
    /// exact inverse of `lrMomentumCycle`, so
    /// `adoptingSchedule(s).lrMomentumCycle == s.lrMomentumCycle` for every
    /// schedule. Every other key is unchanged.
    ///
    /// Why: the lineage record's parameter snapshot must state the schedule
    /// the weights were trained under, and composing the record from this one
    /// value keeps the snapshot and the file's `trainer_*` keys equal by
    /// construction.
    ///
    /// The values are written as they are, **never validated or clamped**
    /// against today's declared ranges (`TrainingParametersSnapshot.replacing`):
    /// the schedule's values are what ran, even when a range has narrowed
    /// since.
    func adoptingSchedule(_ schedule: TrainerScheduleState) -> TrainingParametersSnapshot {
        let cycle = schedule.lrMomentumCycle
        return self
            .replacing(LRWarmupSteps.self, with: schedule.lrWarmupSteps)
            .replacing(LRCycleEnabled.self, with: cycle.lrEnabled)
            .replacing(LRCyclePeriodSteps.self, with: cycle.lrPeriodSteps)
            .replacing(LRCycleCount.self, with: cycle.lrCount)
            .replacing(LRCycleInvert.self, with: cycle.lrInvert)
            .replacing(LRCycleKeyframes.self, with: cycle.lrKeyframes)
            .replacing(MomentumCycleEnabled.self, with: cycle.momentumEnabled)
            .replacing(MomentumCyclePeriodSteps.self, with: cycle.momentumPeriodSteps)
            .replacing(MomentumCycleCount.self, with: cycle.momentumCount)
            .replacing(MomentumCycleMin.self, with: cycle.momentumMin)
            .replacing(MomentumCycleMax.self, with: cycle.momentumMax)
            .replacing(MomentumCycleInvert.self, with: cycle.momentumInvert)
            .replacing(MomentumFollowsLRCycle.self, with: cycle.momentumFollowsLRCycle)
    }
}

/// Log-line renderings of the schedule, shared by every tag that reports it
/// (`[PARAM]`, `[RESUME-PARAM]`, `[STATS]`, the CLI cycle lines) so the same
/// configuration always reads the same way wherever it is grepped.
enum LRMomentumCycleLogFormat {
    /// The keyframes, e.g. `kf=[0:1.00e+00/1.00e-03 m.85/.95, 1000:…]`
    /// (step: peak/trough, then the follow-momentum pair).
    static func keyframesDescription(_ schedule: LRKeyframeSchedule) -> String {
        let frames = schedule.keyframes.map { k in
            String(format: "%d:%.2e/%.2e m%.3f/%.3f", k.step, k.peak, k.trough, k.momentumAtPeak, k.momentumAtTrough)
        }
        return "kf=[\(frames.joined(separator: ", "))]"
    }

    /// The whole schedule configuration — both channels plus the keyframes —
    /// e.g. `lr=^10000st cnt=0 inv=true mom=off momFollow=on kf=[…]`. A
    /// disabled channel renders as `lr=off` / `mom=off`, meaning the trainer
    /// uses the static value. Shared by the `[PARAM] lr_momentum_cycle` audit
    /// line and the CLI runners' startup cycle lines.
    static func cycleDescription(_ cycle: LRMomentumCycle) -> String {
        let lrPart = cycle.lrEnabled
            ? "lr=^\(cycle.lrPeriodSteps)st cnt=\(cycle.lrCount) inv=\(cycle.lrInvert)"
            : "lr=off"
        let momPart = cycle.momentumEnabled
            ? "mom=[\(String(format: "%.2f", cycle.momentumMin)),\(String(format: "%.2f", cycle.momentumMax))]^\(cycle.momentumPeriodSteps)st cnt=\(cycle.momentumCount) inv=\(cycle.momentumInvert)"
            : "mom=off"
        let followPart = cycle.momentumFollowsLRCycle ? "momFollow=on" : "momFollow=off"
        return "\(lrPart) \(momPart) \(followPart) \(keyframesDescription(cycle.lrKeyframes))"
    }

    /// The practical floor for a trough: below it the bf16 working weights
    /// barely move (measured 2026-10-12: at an LR of ≈ 0.0007–0.0013 half the
    /// bf16 weights change per 1,000 steps, and motion scales with the LR;
    /// documentation/plans-active/LR_KEYFRAMES_PLAN.md, owner decision O-5).
    static let practicalTroughFloor = 1.0e-4

    /// A `[PARAM]` warning when an active LR cycle's keyframes put a trough
    /// below `practicalTroughFloor`; nil otherwise. A warning, never a
    /// refusal (O-5).
    static func troughFloorWarning(_ cycle: LRMomentumCycle) -> String? {
        guard cycle.lrEnabled else { return nil }
        let low = cycle.lrKeyframes.keyframes.filter { $0.trough < practicalTroughFloor }
        guard !low.isEmpty else { return nil }
        let steps = low.map { "\($0.step):\(String(format: "%.1e", $0.trough))" }.joined(separator: ", ")
        return "[PARAM] WARNING lr_cycle_keyframes: trough below \(practicalTroughFloor) at step(s) \(steps); "
            + "the bf16 working weights barely move at such an LR (LR_KEYFRAMES_PLAN.md)"
    }

    /// The current LR envelope bounds for the `[STATS]` `lr=` field, e.g.
    /// `[pk=…,tr=…]`. Empty when the LR cycle is not driving the LR.
    static func envelopeBounds(_ values: LRMomentumCycle.Values) -> String {
        guard let peak = values.lrPeak, let trough = values.lrTrough else { return "" }
        return String(format: "[pk=%.1e,tr=%.1e]", peak, trough)
    }
}

/// The learning rate and momentum a trainer is fed at a clock — the one
/// implementation of that host-side arithmetic (review N6, owner decision
/// O-18). `ChessTrainer.buildFeeds` writes its result into the step's feeds,
/// the status-bar readouts (`effectiveLearningRate` / `effectiveMomentum`)
/// show it, and a lineage record's `configuration.schedule_at_save` stores
/// it, so the three can never disagree. Before it, the readouts were a
/// hand-kept mirror of the feed code.
///
/// The cycle is evaluated once for both channels, as the feed always did.
/// The LR is the cycle's value (or the static LR when no cycle is active),
/// times the √batch scale when that rule is on, times the warmup ramp; the
/// arithmetic and its order are the feed's, in `Float`.
enum LRMomentumCycleReadout {
    struct Fed: Sendable, Equatable {
        /// The cycle's own step (warmup offset applied).
        let cycleStep: Int
        let learningRate: Float
        let momentum: Float
    }

    static func values(completedTrainSteps: Int, lrWarmupSteps: Int, cycle: LRMomentumCycle,
                       staticLearningRate: Float, staticMomentum: Float, batchSize: Int,
                       sqrtBatchScaling: Bool, sqrtScaleBaseBatchSize: Int) -> Fed {
        let warmupMul: Float
        if lrWarmupSteps > 0 {
            warmupMul = Float(min(1.0, Double(completedTrainSteps) / Double(lrWarmupSteps)))
        } else {
            warmupMul = 1.0
        }
        let cycleValues = cycle.values(completedTrainSteps: completedTrainSteps, lrWarmupSteps: lrWarmupSteps)
        let baseLR: Float = cycleValues.learningRate.map { Float($0) } ?? staticLearningRate
        var lr: Float
        if sqrtBatchScaling {
            let sqrtBatchScale: Float = Float(sqrt(Double(batchSize) / Double(sqrtScaleBaseBatchSize)))
            lr = baseLR * sqrtBatchScale
        } else {
            lr = baseLR
        }
        lr *= warmupMul
        let momentum: Float = cycleValues.momentum.map { Float($0) } ?? staticMomentum
        return Fed(cycleStep: LRMomentumCycle.cycleStep(completedTrainSteps: completedTrainSteps, lrWarmupSteps: lrWarmupSteps),
                   learningRate: lr, momentum: momentum)
    }

    /// `configuration.schedule_at_save` for a record whose parameters are
    /// `inForce` (its schedule already adopted) at trainer clock
    /// `completedTrainSteps`: the values the trainer is fed there, computed
    /// from the record's own inputs.
    static func scheduleAtSave(inForce p: TrainingParametersSnapshot,
                               completedTrainSteps: Int) -> LineageRecord.ScheduleAtSave {
        let fed = values(completedTrainSteps: completedTrainSteps, lrWarmupSteps: p.lrWarmupSteps,
                         cycle: p.lrMomentumCycle, staticLearningRate: Float(p.learningRate),
                         staticMomentum: Float(p.momentumCoeff), batchSize: p.trainingBatchSize,
                         sqrtBatchScaling: p.sqrtBatchScalingLR,
                         sqrtScaleBaseBatchSize: ChessTrainer.sqrtScaleBaseBatchSize)
        return LineageRecord.ScheduleAtSave(cycleStep: fed.cycleStep, learningRateFed: Double(fed.learningRate),
                                            momentumFed: Double(fed.momentum))
    }
}
