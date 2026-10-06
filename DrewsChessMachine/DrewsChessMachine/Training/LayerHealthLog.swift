import Foundation

/// The `[LAYER-HEALTH]` log lines, shared by every training path — the GUI
/// session's `[STATS]` ticker and session saves, corpus replay, and
/// train-vs-UCI — so the format and the failure handling cannot drift
/// between them. The analysis itself is `LayerHealth`; this file only reads
/// (through the trainer's own queue), schedules, and renders.
///
/// Two line shapes, both grep-able by the tag:
///  - live: `[LAYER-HEALTH] live trainerStep=<n> <compact fields>` — one
///    line at the training path's stats cadence, from the BN state and
///    ReZero α only.
///  - checkpoint: `[LAYER-HEALTH] checkpoint <context> step=<n>
///    [trainerStep=<n>] <compact fields>` followed by the detailed block,
///    every line carrying the tag — from the full trainer state a save has
///    already exported (weights + optimizer velocity), so no extra GPU read.
///
/// A failed health pass never stops training or a save: it is an observer.
/// The failure is logged under the same tag so its absence is never silent.
enum LayerHealthLog {

    static let tag = "[LAYER-HEALTH]"

    // MARK: - Live tier

    /// The outcome of a live health read: the lines to log, and — when the
    /// read and the summary succeeded — the summary together with the
    /// trainer's completed-step clock read on the trainer queue with the
    /// tensors, so a consumer (the training-health monitor) can pair the
    /// summary with exactly the steps it describes. Both are nil when the
    /// read failed; the failure is then the one line in `lines`.
    struct LiveOutcome: Sendable {
        let lines: [String]
        let summary: LayerHealthSummary?
        let trainerStep: Int?
    }

    /// Read the trainer's BN state and ReZero α between steps, summarize,
    /// and render the live line (or the line reporting why it failed). The
    /// summary runs on a GCD queue: a SiLU or GELU site's parked count is a
    /// numerical integral per channel, too long for the cooperative pool.
    static func live(trainer: ChessTrainer) async -> LiveOutcome {
        do {
            let state = try await trainer.readLayerHealthLiveState()
            let arch = trainer.arch
            let tensors = state.tensors
            let summary = try await runOffPool {
                try LayerHealth.summarizeLiveState(arch: arch, tensors: tensors)
            }
            return LiveOutcome(
                lines: [liveLine(summary: summary, trainerStep: state.completedTrainSteps)],
                summary: summary,
                trainerStep: state.completedTrainSteps)
        } catch {
            return LiveOutcome(
                lines: ["\(tag) live read failed: \(error.localizedDescription)"],
                summary: nil,
                trainerStep: nil)
        }
    }

    static func liveLine(summary: LayerHealthSummary, trainerStep: Int) -> String {
        "\(tag) live trainerStep=\(trainerStep) \(summary.compactLine())"
    }

    // MARK: - Checkpoint tier

    /// The outcome of a checkpoint health pass: the lines to log, and the
    /// summary when the pass succeeded (for a results recorder).
    struct CheckpointOutcome: Sendable {
        let lines: [String]
        let summary: LayerHealthSummary?
    }

    /// Summarize a full trainer-state export (`TrainerResumeSnapshot
    /// .trainerWeights`: plan tensors, then one velocity per trainable) and
    /// render the checkpoint block. The scan covers every weight and
    /// velocity value, so it runs on a GCD queue rather than the cooperative
    /// pool. `context` names the save (e.g. `replay-autosave`), `step` is the
    /// caller's own step count and `trainerStep` the trainer's completed-step
    /// clock when the caller has it.
    static func checkpoint(
        arch: NetworkArchitecture,
        trainerWeights: [[Float]],
        context: String,
        step: Int,
        trainerStep: Int?
    ) async -> CheckpointOutcome {
        let headline = checkpointHeadline(context: context, step: step, trainerStep: trainerStep)
        do {
            let summary = try await summarizeOffPool(arch: arch, trainerWeights: trainerWeights)
            return CheckpointOutcome(lines: checkpointLines(summary: summary, headline: headline), summary: summary)
        } catch {
            return CheckpointOutcome(lines: ["\(headline) failed: \(error.localizedDescription)"], summary: nil)
        }
    }

    static func checkpointLines(summary: LayerHealthSummary, headline: String) -> [String] {
        ["\(headline) \(summary.compactLine())"] + summary.detailedLines().map { "\(tag)   \($0)" }
    }

    static func checkpointHeadline(context: String, step: Int, trainerStep: Int?) -> String {
        let trainerStepField = trainerStep.map { " trainerStep=\($0)" } ?? ""
        return "\(tag) checkpoint \(context) step=\(step)\(trainerStepField)"
    }

    private static func summarizeOffPool(
        arch: NetworkArchitecture,
        trainerWeights: [[Float]]
    ) async throws -> LayerHealthSummary {
        try await runOffPool {
            try LayerHealth.summarizeTrainerState(arch: arch, trainerWeights: trainerWeights)
        }
    }

    /// Run a synchronous summary on a GCD queue and resume with its result,
    /// so the cooperative thread keeps making progress.
    private static func runOffPool(
        _ work: @escaping @Sendable () throws -> LayerHealthSummary
    ) async throws -> LayerHealthSummary {
        try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .utility).async {
                do {
                    continuation.resume(returning: try work())
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }
}
