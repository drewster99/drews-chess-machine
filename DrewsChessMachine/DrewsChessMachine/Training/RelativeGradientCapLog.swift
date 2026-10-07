//
//  RelativeGradientCapLog.swift
//  DrewsChessMachine
//
//  The relative gradient cap's log text, shared by the GUI, corpus replay
//  and train-vs-UCI so the three paths' lines compare field for field. The
//  trainer itself logs none of it: each path formats from the step's
//  `TrainStepTiming.gradientCap` and its own history summary, like the other
//  per-step lines.
//

import Foundation

enum RelativeGradientCapLogFormat {
    /// `%g` text for a configuration number (k, floor, hard max): `3`, `0.5`,
    /// `15`.
    static func number(_ value: Double) -> String { String(format: "%g", value) }

    /// One `[GRAD-CLIP]` line for a step whose pre-clip norm exceeded the fed
    /// cap (`applied=true`, any mode — hard-max clips included), or, in mode
    /// `logOnly`, the rule's cap (`applied=false`); nil for any other step.
    /// Every clip is logged (owner decision OD-13) because the step lines
    /// are every 50 steps and a clip between them would otherwise be
    /// invisible.
    static func eventLine(trainerStep: Int, preClipNorm: Float, decision: GradientCapDecision, learningRate: Double) -> String? {
        let applied = decision.clipped(preClipNorm: preClipNorm)
        let logOnlyEvent = decision.mode == .logOnly && decision.wouldClip(preClipNorm: preClipNorm)
        guard applied || logOnlyEvent else { return nil }
        let medianText = decision.referenceMedian.map { String(format: "%.4f", $0) } ?? "none"
        let ratioText: String
        if let median = decision.referenceMedian, median > 0 {
            ratioText = String(format: "%.2f", Double(preClipNorm) / median)
        } else {
            ratioText = "none"
        }
        let kText = decision.mode == .off ? "none" : number(decision.multiple)
        let floorText = decision.mode == .off ? "none" : number(decision.floor)
        return "[GRAD-CLIP] trainerStep=\(trainerStep)"
            + String(format: " preNorm=%.4f cap=%.4f decided=%.4f", preClipNorm, decision.fedCap, decision.decidedCap)
            + " binding=\(decision.binding.rawValue) applied=\(applied) mode=\(decision.mode.token)"
            + " median=\(medianText) n=\(decision.referenceCount) k=\(kText) floor=\(floorText)"
            + " hardMax=\(number(Double(decision.hardMax))) ratio=\(ratioText)"
            + String(format: " lr=%.3g", learningRate)
    }

    /// `[GRAD-CLIP] config …`, logged at each run start (and, in the GUI,
    /// at each Play-and-Train start). A floor at or above the hard max is
    /// allowed (OD-14) but makes the relative term inert; the line says so.
    static func configLine(_ configuration: RelativeGradientCapConfiguration, hardMax: Float) -> String {
        var line = "[GRAD-CLIP] config mode=\(configuration.mode.token) k=\(number(configuration.multiple))"
            + " N=\(configuration.windowSteps) W=\(configuration.minimumHistorySteps)"
            + " floor=\(number(configuration.floor)) hardMax=\(number(Double(hardMax)))"
        if configuration.mode != .off && configuration.floor >= Double(hardMax) {
            line += " relative_cap_inert=floor_at_or_above_hard_max"
        }
        return line
    }

    /// ` gNormMax=<max pre-clip norm since the previous step line> clips=<n>
    /// gCap=<this step's fed cap>`, appended to `[REPLAY]`, `[VS-UCI]` and
    /// `[STATS]`.
    static func stepLineFields(summary: (maxPreClipNorm: Float?, clipped: Int, steps: Int), fedCap: Float) -> String {
        let maxText = summary.maxPreClipNorm.map { String(format: "%.4f", $0) } ?? "none"
        return " gNormMax=\(maxText) clips=\(summary.clipped)" + String(format: " gCap=%.4f", fedCap)
    }
}

/// What one step line reports about the steps since the previous line: the
/// largest pre-clip norm, how many were clipped, and the fed cap of the
/// line's own step. Shared by `[REPLAY]`, `[VS-UCI]` and `[STATS]` and by the
/// `results.json` rows written at the same ticks.
struct GradientCapStepLineReading: Sendable, Equatable {
    let maxPreClipNorm: Float?
    let clipped: Int
    let steps: Int
    let fedCap: Float

    /// ` gNormMax=… clips=… gCap=…`.
    var logFields: String {
        RelativeGradientCapLogFormat.stepLineFields(
            summary: (maxPreClipNorm: maxPreClipNorm, clipped: clipped, steps: steps), fedCap: fedCap)
    }
}

/// The trainer steps a run's step lines have covered, so each line
/// summarizes exactly the steps since the previous one — the blind spot of
/// every-50-step lines (E-0020: the precursor clip landed on an unlogged
/// step) closes because an unlogged spike shows in the next line's
/// `gNormMax=`.
struct GradientCapStepLineWindow: Sendable {
    /// The trainer step the previous line (or the run start) covered.
    private(set) var lastCoveredTrainerStep: Int

    init(startTrainerStep: Int) {
        lastCoveredTrainerStep = startTrainerStep
    }

    /// The reading for the steps after the previous line through
    /// `trainerStep`, from the trainer's `history`; advances the window.
    mutating func take(history: GradientNormHistory, throughTrainerStep trainerStep: Int,
                       fedCap: Float) -> GradientCapStepLineReading {
        let lower = lastCoveredTrainerStep + 1
        let summary = lower <= trainerStep
            ? history.summary(trainerSteps: lower...trainerStep)
            : (maxPreClipNorm: nil, clipped: 0, steps: 0)
        lastCoveredTrainerStep = max(lastCoveredTrainerStep, trainerStep)
        return GradientCapStepLineReading(maxPreClipNorm: summary.maxPreClipNorm, clipped: summary.clipped,
                                          steps: summary.steps, fedCap: fedCap)
    }
}
