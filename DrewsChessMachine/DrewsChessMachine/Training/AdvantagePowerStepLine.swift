//
//  AdvantagePowerStepLine.swift
//  DrewsChessMachine
//
//  The policy-gradient advantage's power, as the step lines report it:
//  `advRMS=<mean>/<min>/<max> advFloored=<k>/<N>` over the trainer steps since
//  the previous `[REPLAY]` / `[VS-UCI]` / `[STATS]` line.
//
//  Why it is logged. The policy loss divides the raw advantage
//  `A = z − vBaseline` by `sqrt(max(E[A²], floor) + ε)`. Above the floor that
//  is a pure RMS normalization: every batch's REINFORCE weights have unit
//  power, whatever the baseline's quality. Below it the divisor stops
//  shrinking, so the policy gradient scales down with the batch's advantage
//  power — a batch whose value baseline already predicts the outcomes well
//  (or a draw-heavy batch with a confident draw baseline) pushes the policy
//  less. Which regime a run trains in, and how often it crosses over, is
//  invisible from `pLoss` / `gNorm` alone, and the diagnostic-step `adv=`
//  block samples one batch in many. Every step's pre-floor mean square is
//  already computed on the GPU for the normalization; reading it back with
//  the other per-step scalars costs one fp32 scalar and no extra pass.
//
//  Why the window lives with the trainer (like `TrainingDynamicsLog`) rather
//  than in each path: the GUI's `[STATS]` line runs in a different task from
//  the step loop, so a path-side accumulator would need its own shared box
//  there and a second code path on the CLI. One trainer-held window records
//  on the trainer's queue after each real-data step and is drained by every
//  path's step-line code, so the three lines compare field for field.
//

import Foundation
import os

/// What one step line reports about the advantage power of the steps since
/// the previous line. Shared by `[REPLAY]`, `[VS-UCI]`, `[STATS]` and the
/// `results.json` rows written at the same ticks.
struct AdvantagePowerStepLineReading: Sendable, Equatable {
    /// Steps the reading covers.
    let steps: Int
    /// Mean, min and max over those steps of the per-step pre-floor RMS,
    /// `sqrt(E[A²])`; nil when `steps` is 0.
    let rmsMean: Double?
    let rmsMin: Double?
    let rmsMax: Double?
    /// Steps whose pre-floor mean square was below
    /// `ChessTrainer.advantagePowerFloor`, i.e. steps whose normalization
    /// divided by the floor instead of the batch's own RMS.
    let flooredSteps: Int

    /// ` advRMS=<mean>/<min>/<max> advFloored=<k>/<N>`; `advRMS=none` for a
    /// window with no step (a GUI line before the first step, or right after
    /// a rewind).
    var logFields: String {
        let rmsText: String
        if let rmsMean, let rmsMin, let rmsMax {
            rmsText = String(format: "%.4f/%.4f/%.4f", rmsMean, rmsMin, rmsMax)
        } else {
            rmsText = "none"
        }
        return " advRMS=\(rmsText) advFloored=\(flooredSteps)/\(steps)"
    }
}

enum AdvantagePowerStepLineError: Error, CustomStringConvertible, LocalizedError {
    /// A step recorded at or below one already pending: the trainer clock
    /// moved back without `rewind(toTrainerStep:)`.
    case nonIncreasingStep(last: Int, got: Int)

    var description: String {
        switch self {
        case .nonIncreasingStep(let last, let got):
            return "advantage-power window: trainer step \(got) recorded after step \(last) without a rewind"
        }
    }

    var errorDescription: String? { description }
}

/// The pending per-step advantage mean squares since the previous step line,
/// keyed by trainer step so a rewind can drop exactly the steps of the
/// discarded trajectory. Pure value type; `AdvantagePowerStepLog` holds the
/// trainer's one instance under a lock.
struct AdvantagePowerStepLineWindow: Sendable, Equatable {
    /// The floor a step's mean square is compared against.
    let floor: Float
    /// Pending entries, strictly increasing in trainer step.
    private(set) var pendingTrainerSteps: [Int] = []
    private(set) var pendingMeanSquares: [Float] = []

    init(floor: Float) {
        self.floor = floor
    }

    /// Record trainer step `trainerStep`'s pre-floor advantage mean square.
    /// Steps must increase; a clock that moved back must `rewind` first.
    mutating func record(trainerStep: Int, meanSquare: Float) throws {
        if let last = pendingTrainerSteps.last, trainerStep <= last {
            throw AdvantagePowerStepLineError.nonIncreasingStep(last: last, got: trainerStep)
        }
        pendingTrainerSteps.append(trainerStep)
        pendingMeanSquares.append(meanSquare)
    }

    /// The trainer clock was set to `trainerStep` (a promotion's rewind to the
    /// arena start, a resume's restored clock, a fresh network's 0): pending
    /// steps above it belong to the discarded trajectory and are dropped
    /// unreported, as the gradient-cap window drops them with the restored
    /// gradient-norm history. Steps at or below it were trained on the
    /// weights kept and stay for the next line.
    mutating func rewind(toTrainerStep trainerStep: Int) {
        guard let keep = pendingTrainerSteps.lastIndex(where: { $0 <= trainerStep }) else {
            pendingTrainerSteps.removeAll()
            pendingMeanSquares.removeAll()
            return
        }
        pendingTrainerSteps.removeSubrange((keep + 1)...)
        pendingMeanSquares.removeSubrange((keep + 1)...)
    }

    /// The reading for the pending steps at or below `trainerStep`, which
    /// leave the window; steps above it (a GUI step that landed after the
    /// line read its clock) stay for the next line.
    mutating func take(throughTrainerStep trainerStep: Int) -> AdvantagePowerStepLineReading {
        let count = pendingTrainerSteps.firstIndex(where: { $0 > trainerStep }) ?? pendingTrainerSteps.count
        var sum = 0.0
        var minimum: Double?
        var maximum: Double?
        var floored = 0
        for meanSquare in pendingMeanSquares[..<count] {
            let rms = Double(meanSquare).squareRoot()
            sum += rms
            minimum = minimum.map { Swift.min($0, rms) } ?? rms
            maximum = maximum.map { Swift.max($0, rms) } ?? rms
            if meanSquare < floor { floored += 1 }
        }
        pendingTrainerSteps.removeFirst(count)
        pendingMeanSquares.removeFirst(count)
        return AdvantagePowerStepLineReading(
            steps: count,
            rmsMean: count > 0 ? sum / Double(count) : nil,
            rmsMin: minimum,
            rmsMax: maximum,
            flooredSteps: floored)
    }
}

/// The trainer's advantage-power window under a lock: the trainer records on
/// its execution queue after every real-data step and rewinds it with every
/// clock change; each path's step-line code takes a reading from wherever it
/// runs. A pure observer: nothing here feeds back into training.
final class AdvantagePowerStepLog: @unchecked Sendable {
    private let lock: OSAllocatedUnfairLock<AdvantagePowerStepLineWindow>

    init(floor: Float) {
        lock = OSAllocatedUnfairLock(initialState: AdvantagePowerStepLineWindow(floor: floor))
    }

    func record(trainerStep: Int, meanSquare: Float) throws {
        try lock.withLock { try $0.record(trainerStep: trainerStep, meanSquare: meanSquare) }
    }

    func rewind(toTrainerStep trainerStep: Int) {
        lock.withLock { $0.rewind(toTrainerStep: trainerStep) }
    }

    func take(throughTrainerStep trainerStep: Int) -> AdvantagePowerStepLineReading {
        lock.withLock { $0.take(throughTrainerStep: trainerStep) }
    }
}
