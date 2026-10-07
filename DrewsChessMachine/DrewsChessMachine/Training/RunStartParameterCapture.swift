import Foundation

/// The training parameters a GUI Play-and-Train run reads once, at its start,
/// and trains under until the next start: the training batch size, the replay
/// buffer's pre-train fill, and the replay buffer's capacity.
///
/// **Why this exists.** These three keys are not live-tunable: the trainer
/// steps at the batch size captured at the start, the training worker waits
/// for the pre-train fill captured at the start, and the buffer keeps the
/// capacity it was built with (a Continue reuses it). But the settings popover
/// still writes them to `TrainingParameters.shared` during a run, so they take
/// effect at the next start. Every save used to snapshot the singleton, so a
/// run whose batch-size field was edited mid-run wrote files that recorded
/// the edited batch size, not the one it stepped at — and the √batch
/// learning-rate scale recomputed from such a file was wrong by the square
/// root of their ratio. The same stale value reached `session.json`,
/// the status-bar learning-rate readouts, the arena record and the stats
/// sample.
///
/// One capture per start (`SessionController.beginRunStartCapture(buffer:)`)
/// is the single source for those three values while the run's state is
/// described: every in-run reader takes it, and a saved record's parameter
/// snapshot is `inForce(over:)` the singleton's — the singleton's other keys
/// with these three replaced by what the run actually uses. There is no
/// fallback to the singleton: a reader that finds no capture inside a run is
/// a bug and fails.
///
/// The capacity is read from the run's buffer itself (`ReplayBuffer.capacity`),
/// the measured value, which on a Continue is the reused buffer's.
struct RunStartParameterCapture: Sendable, Equatable {
    let trainingBatchSize: Int
    let replayBufferMinPositionsBeforeTraining: Int
    let replayBufferCapacity: Int

    /// The ids of the captured keys.
    static let capturedKeyIDs: Set<String> = [
        TrainingBatchSize.id,
        ReplayBufferMinPositionsBeforeTraining.id,
        ReplayBufferCapacity.id,
    ]

    /// When an edit of a captured key takes effect — the one wording every
    /// caption and `[PARAM]` line about it uses.
    static let nextStartPhrase = "at the next Play-and-Train start"

    /// The caption the settings popover shows beside a captured field.
    static let appliesAtNextStartCaption = "Applies \(nextStartPhrase)"

    /// The caption under the popover's Replay-buffer group, which holds two
    /// captured fields.
    static let replayBufferCaption = "Capacity and pre-train fill apply \(nextStartPhrase)"

    /// `snapshot` with the three captured keys replaced by the values this
    /// run uses — the parameters in force. Every other key is `snapshot`'s.
    /// The captured values are written as they are: they were valid when the
    /// run started, and they are what it trains under.
    func inForce(over snapshot: TrainingParametersSnapshot) -> TrainingParametersSnapshot {
        snapshot
            .replacing(TrainingBatchSize.self, with: trainingBatchSize)
            .replacing(ReplayBufferMinPositionsBeforeTraining.self, with: replayBufferMinPositionsBeforeTraining)
            .replacing(ReplayBufferCapacity.self, with: replayBufferCapacity)
    }

    /// The `[PARAM]` line for a settings edit of the captured key `id` while
    /// a run holds this capture: the edit is saved, and takes effect at the
    /// next start; this run keeps `inForceValue`. Keys are named by their
    /// parameter id, as the run-start capture line and `[RESUME-DIFF]` name
    /// them, so one grep finds every line about a key.
    static func deferredEditLogLine(id: String, old: Int, new: Int, inForceValue: Int) -> String {
        "[PARAM] \(id): \(old) -> \(new) (applies \(nextStartPhrase); this run keeps \(inForceValue))"
    }
}
