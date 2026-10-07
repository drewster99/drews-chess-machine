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
