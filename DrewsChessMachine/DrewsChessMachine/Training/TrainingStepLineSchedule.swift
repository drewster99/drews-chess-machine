//
//  TrainingStepLineSchedule.swift
//  DrewsChessMachine
//
//  The single source of "is a step line due": when the GUI writes `[STATS]`,
//  corpus replay `[REPLAY]` and train-vs-UCI `[VS-UCI]`, and at which trainer
//  steps the CLI runners save.
//
//  Why it is keyed on the trainer step. The step line used to be written
//  every 50 steps of the *segment's* own count, while the trainer computes
//  its diagnostic reductions (policy entropy, value W/D/L, played-move
//  probability) on multiples of its *cumulative* step. In a first segment
//  the two counters are equal; after an exact resume at a trainer step that
//  is not a multiple of the diagnostics interval (arm C resumed at 513) no
//  logged step was ever a diagnostic step, and every line printed `--`.
//  Keying the lines on the trainer step makes a resumed segment log at the
//  trainer steps the uninterrupted run logs at, and `ChessTrainer` forces
//  its diagnostics on every fixed line step (`isDiagnosticsStep`), so every
//  line after the first carries them for any `batch_stats_interval`.
//
//  The cadence: dense at the start (every 50 trainer steps through trainer
//  step 1000), then a line at every trainer-step multiple of 1000, plus a
//  time-based line — the first diagnostics step at least
//  `step_line_interval_sec` after the previous line — so a long run's log
//  stays readable without carrying a 72 KB `[BATCH-STATS]` line every 50
//  steps. A segment's first step always gets a line. Any line restarts the
//  interval, and the time rule applies in the dense phase too (it adds lines
//  there only when 50 steps take longer than the interval).
//
//  Nothing here decides an alarm evaluation: alarms run on their own
//  50-step tick. The step line is logging only. It never touches trainer,
//  optimizer, buffer or RNG state, and which steps compute diagnostics
//  depends on the trainer step alone, so a resumed run and the
//  uninterrupted run run the same graph on every step.
//

import Foundation

/// When a step line is due, and at which trainer steps runs save. A pure
/// value: callers supply the trainer step and a monotonic elapsed time, so
/// the tests drive both directly.
struct TrainingStepLineSchedule: Sendable, Equatable {

    /// The trainer-step stride of the dense lines at the start of training.
    static let denseLineIntervalSteps = 50
    /// The last trainer step of the dense phase.
    static let denseLinesThroughTrainerStep = 1_000
    /// The trainer-step stride of the fixed lines after the dense phase, and
    /// the save interval of the CLI runners (autosaves and enumerated
    /// checkpoints) — one constant, so lines and saves cannot drift apart.
    static let checkpointIntervalSteps = 1_000

    /// Why a line is due.
    enum Reason: String, Sendable, Equatable {
        /// The first step this schedule observed — a segment's first step.
        /// It may lack diagnostics: the trainer computes them only on its
        /// own cadence.
        case segmentStart
        /// A fixed line step was reached (or, for a polling caller, passed).
        case fixedStep
        /// The time interval since the previous line has elapsed and this
        /// step carries diagnostics.
        case interval
    }

    /// Whether `trainerStep` is a fixed line step: above 0, and either a
    /// multiple of 50 through trainer step 1000 or a multiple of 1000.
    static func isFixedLineStep(trainerStep: Int) -> Bool {
        guard trainerStep > 0 else { return false }
        if trainerStep <= denseLinesThroughTrainerStep && trainerStep % denseLineIntervalSteps == 0 {
            return true
        }
        return trainerStep % checkpointIntervalSteps == 0
    }

    /// Whether `trainerStep` is a save step: above 0 and a multiple of 1000.
    /// Every checkpoint step is a fixed line step.
    static func isCheckpointStep(trainerStep: Int) -> Bool {
        trainerStep > 0 && trainerStep % checkpointIntervalSteps == 0
    }

    /// The smallest fixed line step above `trainerStep`.
    static func firstFixedLineStep(after trainerStep: Int) -> Int {
        let next = max(trainerStep, 0) + 1
        if next <= denseLinesThroughTrainerStep {
            return roundedUp(next, toMultipleOf: denseLineIntervalSteps)
        }
        return roundedUp(next, toMultipleOf: checkpointIntervalSteps)
    }

    /// The smallest checkpoint (save) step above `trainerStep` — what a run's
    /// start line names as its next save. (`firstFixedLineStep` is not: it is
    /// 50 on a fresh run.)
    static func firstCheckpointStep(after trainerStep: Int) -> Int {
        roundedUp(max(trainerStep, 0) + 1, toMultipleOf: checkpointIntervalSteps)
    }

    private static func roundedUp(_ value: Int, toMultipleOf stride: Int) -> Int {
        ((value + stride - 1) / stride) * stride
    }

    /// A `ContinuousClock` duration in seconds, for `lineDue`'s `elapsedSec`.
    static func seconds(_ duration: Duration) -> Double {
        let parts = duration.components
        return Double(parts.seconds) + Double(parts.attoseconds) / 1e18
    }

    /// The one-line description of the cadence each CLI runner logs at start.
    static func cadenceDescription(intervalSec: Double, startTrainerStep: Int) -> String {
        "step lines: every \(denseLineIntervalSteps) trainer steps through \(denseLinesThroughTrainerStep), "
            + "every \(checkpointIntervalSteps), and the first diagnostics step ≥ "
            + String(format: "%g", intervalSec) + " s after the previous line; saves at trainer-step multiples of "
            + "\(checkpointIntervalSteps) (next: \(firstCheckpointStep(after: startTrainerStep)))"
    }

    /// The last trainer step observed, nil before the first call.
    private(set) var lastObservedTrainerStep: Int?
    /// The elapsed time of the last line, nil before the first line.
    private(set) var lastLineElapsedSec: Double?

    init() {}

    /// Whether a line is due at `trainerStep`, observed `elapsedSec` seconds
    /// (monotonic, since the caller's start) into the run, and why. Rules,
    /// in order: the first call is `.segmentStart`; a trainer step below the
    /// last one observed (the GUI's promotion rewinds the trainer clock)
    /// moves the step baseline there and is not itself a fixed line; a fixed
    /// line step in `(lastObserved, trainerStep]` is `.fixedStep` (the CLI
    /// observes every step; the GUI polls, so for it this is the first poll
    /// at or past the step); otherwise a step that carries diagnostics at
    /// least `intervalSec` after the previous line is `.interval`. Every line
    /// restarts the interval.
    mutating func lineDue(trainerStep: Int, elapsedSec: Double, carriesDiagnostics: Bool,
                          intervalSec: Double) -> Reason? {
        precondition(intervalSec > 0, "step_line_interval_sec must be positive (got \(intervalSec))")
        let reason = reasonForLine(trainerStep: trainerStep, elapsedSec: elapsedSec,
                                   carriesDiagnostics: carriesDiagnostics, intervalSec: intervalSec)
        lastObservedTrainerStep = trainerStep
        if reason != nil {
            lastLineElapsedSec = elapsedSec
        }
        return reason
    }

    private func reasonForLine(trainerStep: Int, elapsedSec: Double, carriesDiagnostics: Bool,
                               intervalSec: Double) -> Reason? {
        guard let lastObserved = lastObservedTrainerStep, let lastLine = lastLineElapsedSec else {
            return .segmentStart
        }
        // A rewind (a step below the last observed) never satisfies this, so
        // it is not a fixed line; `lineDue` then moves the baseline to it, and
        // fixed steps re-crossed after it get their lines again.
        if trainerStep >= Self.firstFixedLineStep(after: lastObserved) {
            return .fixedStep
        }
        if carriesDiagnostics && elapsedSec - lastLine >= intervalSec {
            return .interval
        }
        return nil
    }
}
