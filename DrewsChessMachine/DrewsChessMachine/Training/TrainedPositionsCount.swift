import Foundation

/// Positions a trainer has trained on — the sum, over its steps, of the batch
/// size each step trained at — counted on one step axis from one start (a
/// GUI Play-and-Train start, a train-vs-UCI run) on, on top of what was
/// recorded for the steps before it.
///
/// **Why this exists.** session.json's `trainingPositionsSeen`, the status
/// bar's "Positions trained", the progress-rate chart and the Lichess probe
/// export each multiplied the whole step count by the batch size in force at
/// the moment. A run continued, resumed or branched at another batch size
/// restated every earlier step at the new one (10 steps at 4096 then 10 at
/// 1024 read as 20 × 1024). A start's batch size never changes until the next
/// start (`RunStartParameterCapture`), so the count is kept per start: what
/// was recorded for the steps before it, plus this start's steps at this
/// start's batch. Steps before a start that no record describes leave the
/// count unrecorded (`nil`) — never modeled from a batch size they may not
/// have trained at.
struct TrainedPositionsCount: Sendable, Equatable {
    /// The step count, on this count's axis, when the start began.
    let stepsAtStart: Int
    /// Positions trained over the steps before `stepsAtStart`; nil when any
    /// of those steps is not recorded.
    let positionsBeforeStart: Int?
    /// The batch size every step from `stepsAtStart` on trains at.
    let batchSize: Int

    enum CountError: Error, LocalizedError, Equatable {
        /// Asked for a step count below the start's: steps that came before
        /// this count began, which it does not describe.
        case stepsBeforeStart(steps: Int, stepsAtStart: Int)

        var errorDescription: String? {
            switch self {
            case .stepsBeforeStart(let steps, let stepsAtStart):
                return "trained positions asked for at step \(steps), before the count began at step \(stepsAtStart)"
            }
        }
    }

    /// Positions trained by the time the axis reads `steps`; nil when the
    /// steps before the start are unrecorded.
    func positions(atSteps steps: Int) throws -> Int? {
        guard steps >= stepsAtStart else {
            throw CountError.stepsBeforeStart(steps: steps, stepsAtStart: stepsAtStart)
        }
        return positionsBeforeStart.map { $0 + (steps - stepsAtStart) * batchSize }
    }
}
