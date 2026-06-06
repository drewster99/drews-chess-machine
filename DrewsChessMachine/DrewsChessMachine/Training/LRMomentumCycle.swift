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
/// - **Phase is a pure function of the global step.** `phase = (step mod
///   period) / period ∈ [0,1)`, so the schedule carries *no state of its own*.
///   Stop/resume is automatic: the trainer's `completedTrainSteps` is already
///   persisted, so on resume the phase continues with no discontinuity.
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
///   `momentumMax`, i.e. the low-LR / high-momentum converged regime.
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
    /// Absolute LR at the period boundaries (the cycle's low point when
    /// `lrInvert == false`). Must be > 0 for geometric interpolation.
    var lrMin: Double
    /// Absolute LR at the period midpoint (when `lrInvert == false`).
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

    /// The all-off configuration. Endpoint defaults mirror the parameter
    /// defaults so a freshly-enabled-but-unedited cycle is sane.
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

    /// Cosine up-then-down position in `[0, 1]` for `step`, honoring the
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

    /// Effective learning rate for `step`, or `nil` when LR cycling is
    /// inactive or misconfigured — the caller then falls back to the static
    /// base learning rate. Geometric (log-space) interpolation requires
    /// `lrMin > 0` and `lrMax >= lrMin`; otherwise `nil` is returned.
    func learningRate(forStep step: Int) -> Double? {
        guard lrEnabled, lrPeriodSteps > 0, lrMin > 0, lrMax >= lrMin else { return nil }
        let frac = Self.cycleFraction(step: step, period: lrPeriodSteps, count: lrCount, invert: lrInvert)
        return lrMin * pow(lrMax / lrMin, frac)
    }

    /// Effective Polyak momentum for `step`, or `nil` when momentum cycling
    /// is inactive — the caller then falls back to the static momentum
    /// coefficient. Linear interpolation between the endpoints.
    func momentum(forStep step: Int) -> Double? {
        guard momentumEnabled, momentumPeriodSteps > 0 else { return nil }
        let frac = Self.cycleFraction(step: step, period: momentumPeriodSteps, count: momentumCount, invert: momentumInvert)
        return momentumMin + (momentumMax - momentumMin) * frac
    }

    /// True when either channel is actively cycling — used to decide whether
    /// to emit the effective values into `[STATS]`.
    var isActive: Bool { lrEnabled || momentumEnabled }
}

@MainActor
extension TrainingParameters {
    /// The current LR/momentum cycle configuration, read live from the
    /// singleton's stored parameter values. Pushed onto the running trainer
    /// at session start (`SessionController.makeTrainer`), on each cycling
    /// edit (`ControlSideEffectsProbe`), and on resume.
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
            momentumInvert: momentumCycleInvert
        )
    }
}
