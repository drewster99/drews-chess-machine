import Foundation

/// Lock-protected holder for the live training-step delay in
/// milliseconds. The training worker reads this at the bottom of
/// every step to decide how long to pause before looping; the
/// Stepper in the Play and Train row writes through it whenever
/// the user nudges the value. Decoupled from `@State
/// `trainingParams.trainingStepDelayMs` so the worker task doesn't have to hop back
/// to the main actor to read a single Int per step. Clamped at the
/// bottom to 0 — negative delays are meaningless.
final class TrainingStepDelayBox: @unchecked Sendable {
    private let queue = DispatchQueue(label: "drewschess.trainingstepdelaybox.serial")
    private var _ms: Int

    init(initial: Int) {
        _ms = max(0, initial)
    }

    var milliseconds: Int {
        queue.sync { _ms }
    }

    func set(_ value: Int) {
        queue.async { [weak self] in self?._ms = max(0, value) }
    }
}
