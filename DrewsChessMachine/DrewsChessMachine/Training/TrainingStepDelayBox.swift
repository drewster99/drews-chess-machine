import Foundation

/// Lock-protected holder for the live training-step delay in
/// milliseconds. The training worker reads this at the bottom of
/// every step to decide how long to pause before looping; the
/// Stepper in the Play and Train row writes through it whenever
/// the user nudges the value. Decoupled from
/// `trainingParams.trainingStepDelayMs` so the worker task doesn't
/// have to hop back to the main actor to read a single Int per
/// step. Clamped at the bottom to 0 — negative delays are
/// meaningless.
///
/// Backed by `SyncBox<Int>` (an `OSAllocatedUnfairLock`); reads and
/// writes are sub-microsecond.
final class TrainingStepDelayBox: @unchecked Sendable {
    private let _ms: SyncBox<Int>

    init(initial: Int) {
        _ms = SyncBox<Int>(max(0, initial))
    }

    var milliseconds: Int {
        _ms.value
    }

    func set(_ value: Int) {
        _ms.value = max(0, value)
    }
}
