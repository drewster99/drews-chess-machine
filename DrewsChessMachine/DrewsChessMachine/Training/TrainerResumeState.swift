import Foundation

/// Everything besides the tensors that decides what the optimizer does on the
/// next step: the trainer's completed-step clock, and the warmup length and
/// LR/momentum cycle (decay envelope included) that clock is read against.
///
/// **Why this exists.** "Resume" in this project means training continues
/// exactly as if it had never stopped. The LR warmup multiplier, the cycle's
/// phase and the decay envelope are all pure functions of
/// `ChessTrainer.completedTrainSteps` (see `LRMomentumCycle`), so a resumed
/// trainer whose clock restarts at zero replays warmup and restarts the cycle
/// and the decay from their beginnings — which is what the corpus-replay and
/// train-vs-UCI runners did (`phaseOrigin=segment-step-0`): every segment
/// began with a fresh clock, zero optimizer velocity, and a short
/// "momentum-refill" warmup. Persisting this state next to the weights, and
/// restoring it through one function, is what makes a resume exact.
///
/// It travels in the safetensors `__metadata__` of every trainer-state file
/// (the GUI session's trainer file and the CLI runners' rolling and
/// enumerated checkpoints) under the `trainer_*` keys, always together with
/// the optimizer velocity tensors and the fp32 master weights — a file that
/// carries one without the others cannot be resumed exactly, so
/// `SafetensorsModelIO` refuses to write or read that combination.
///
/// The clock here is cumulative across every exact resume of the lineage,
/// and it is what a resume restores — never `ModelCheckpointMetadata.trainingStep`.
/// From format v11 a trainer-state file's `training_step` states the same
/// value (the writer and the reader refuse a file where they differ); the
/// segment's own step is the lineage record's `segment_local_step`. Files
/// written before v11 by the CLI runners state their *segment-local* step as
/// `training_step` (read by `ModelFileStepReading`), which is why the clock
/// was never taken from it.
struct TrainerScheduleState: Sendable, Equatable {
    /// The trainer's completed SGD steps — the clock warmup, the cycle phase
    /// and the decay envelope are evaluated against.
    var completedTrainSteps: Int
    /// LR warmup length, in trainer steps.
    var lrWarmupSteps: Int
    /// The LR/momentum cycle, with its decay envelope attached.
    var lrMomentumCycle: LRMomentumCycle

    /// The schedule a trainer is running right now. Callers must have paused
    /// training so the clock cannot advance between this read and the weight
    /// export it accompanies.
    init(currentlyRunningOn trainer: ChessTrainer) {
        completedTrainSteps = trainer.completedTrainSteps
        lrWarmupSteps = trainer.lrWarmupSteps
        lrMomentumCycle = trainer.lrMomentumCycle
    }

    init(completedTrainSteps: Int, lrWarmupSteps: Int, lrMomentumCycle: LRMomentumCycle) {
        self.completedTrainSteps = completedTrainSteps
        self.lrWarmupSteps = lrWarmupSteps
        self.lrMomentumCycle = lrMomentumCycle
    }

    /// The schedule a GUI session resume restores, with the `[RESUME-PARAM]`
    /// lines that account for it.
    ///
    /// The clock comes from the trainer file's `trainer_completed_steps` — the
    /// trainer's own counter, read under the same pause as its weights. The
    /// session's `trainingSteps` is the stats box's count, which is not the
    /// trainer's clock after "New Session, keep trainer" (the box restarts at
    /// zero while the trainer carries on). A trainer file written before the
    /// clock was persisted has no such key; the session's step count is then
    /// used, which is what every resume did before and is the trainer's clock
    /// for any session that never kept a trainer across "New Session".
    ///
    /// The warmup length and cycle are the session's own (from session.json,
    /// already resolved onto the trainer by the `[RESUME-PARAM]` block). The
    /// trainer file carries them too; a disagreement is logged, not resolved,
    /// because session.json is the GUI's source for parameters.
    static func forSessionResume(
        trainerFileSchedule: TrainerScheduleState?,
        sessionTrainingSteps: Int,
        sessionWarmupSteps: Int,
        sessionCycle: LRMomentumCycle
    ) -> (schedule: TrainerScheduleState, logLines: [String]) {
        var lines: [String] = []
        let clock: Int
        if let fileSchedule = trainerFileSchedule {
            clock = fileSchedule.completedTrainSteps
            lines.append("trainer_completed_steps: \(clock) (from trainer file; session step count \(sessionTrainingSteps))")
            if fileSchedule.lrWarmupSteps != sessionWarmupSteps {
                lines.append("WARNING trainer file lr_warmup_steps \(fileSchedule.lrWarmupSteps) differs from the session's \(sessionWarmupSteps); using the session's")
            }
            if fileSchedule.lrMomentumCycle != sessionCycle {
                lines.append(
                    "WARNING trainer file lr_momentum_cycle [\(LRMomentumCycleLogFormat.cycleDescription(fileSchedule.lrMomentumCycle))] "
                        + "differs from the session's [\(LRMomentumCycleLogFormat.cycleDescription(sessionCycle))]; using the session's"
                )
            }
        } else {
            clock = sessionTrainingSteps
            lines.append("trainer_completed_steps: saved=nil applied=\(clock) (trainer file predates the persisted trainer clock; using the session step count)")
        }
        let schedule = TrainerScheduleState(
            completedTrainSteps: clock,
            lrWarmupSteps: sessionWarmupSteps,
            lrMomentumCycle: sessionCycle
        )
        lines.append("trainer_schedule_origin: trainerStep \(clock) cycleStep \(schedule.cycleStep) (exact resume)")
        return (schedule, lines)
    }

    /// The cycle's own step at the saved clock (warmup offset applied), for
    /// the resume log lines.
    var cycleStep: Int {
        LRMomentumCycle.cycleStep(completedTrainSteps: completedTrainSteps, lrWarmupSteps: lrWarmupSteps)
    }

    // MARK: - Safetensors metadata

    enum MetadataError: Error, CustomStringConvertible, LocalizedError {
        case missingKey(String)
        case malformedValue(key: String, value: String)
        case undecodableJSON(key: String, detail: String)

        var description: String {
            switch self {
            case .missingKey(let key):
                return "trainer schedule metadata is incomplete: '\(key)' is missing"
            case .malformedValue(let key, let value):
                return "trainer schedule metadata '\(key)' has malformed value '\(value)'"
            case .undecodableJSON(let key, let detail):
                return "trainer schedule metadata '\(key)' failed to decode: \(detail)"
            }
        }

        var errorDescription: String? { description }
    }

    /// The `__metadata__` keys this state is written under.
    enum MetadataKey {
        static let completedTrainSteps = "trainer_completed_steps"
        static let lrWarmupSteps = "trainer_lr_warmup_steps"
        static let lrMomentumCycle = "trainer_lr_momentum_cycle"
        static let lrMomentumCycleEnvelope = "trainer_lr_momentum_cycle_envelope"
        static let all: [String] = [completedTrainSteps, lrWarmupSteps, lrMomentumCycle, lrMomentumCycleEnvelope]
    }

    /// The metadata entries for this state. The cycle and its envelope are
    /// encoded as separate JSON objects because `LRMomentumCycle`'s `Codable`
    /// form deliberately excludes the envelope (see `LRMomentumCycle.envelope`).
    /// `JSONEncoder` writes doubles in shortest round-trip form, so every
    /// endpoint decodes back bit-exactly.
    func metadataEntries() throws -> [String: String] {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let cycleJSON = try encoder.encode(lrMomentumCycle)
        let envelopeJSON = try encoder.encode(lrMomentumCycle.envelope)
        return [
            MetadataKey.completedTrainSteps: String(completedTrainSteps),
            MetadataKey.lrWarmupSteps: String(lrWarmupSteps),
            MetadataKey.lrMomentumCycle: String(decoding: cycleJSON, as: UTF8.self),
            MetadataKey.lrMomentumCycleEnvelope: String(decoding: envelopeJSON, as: UTF8.self),
        ]
    }

    /// The state stored in `metadata`, or nil when the file carries none of
    /// the `trainer_*` keys (a plain model file, or a trainer file written
    /// before exact resume existed). A file carrying only some of them, or a
    /// malformed value, throws: half a schedule is not a resumable one.
    static func decode(fromMetadata metadata: [String: String]) throws -> TrainerScheduleState? {
        guard MetadataKey.all.contains(where: { metadata[$0] != nil }) else { return nil }
        func required(_ key: String) throws -> String {
            guard let value = metadata[key] else { throw MetadataError.missingKey(key) }
            return value
        }
        func integer(_ key: String) throws -> Int {
            let text = try required(key)
            guard let value = Int(text), value >= 0 else {
                throw MetadataError.malformedValue(key: key, value: text)
            }
            return value
        }
        func json<T: Decodable>(_ type: T.Type, _ key: String) throws -> T {
            let text = try required(key)
            do {
                return try JSONDecoder().decode(T.self, from: Data(text.utf8))
            } catch {
                throw MetadataError.undecodableJSON(key: key, detail: error.localizedDescription)
            }
        }
        var cycle = try json(LRMomentumCycle.self, MetadataKey.lrMomentumCycle)
        cycle.envelope = try json(LRMomentumCycleEnvelope.self, MetadataKey.lrMomentumCycleEnvelope)
        return TrainerScheduleState(
            completedTrainSteps: try integer(MetadataKey.completedTrainSteps),
            lrWarmupSteps: try integer(MetadataKey.lrWarmupSteps),
            lrMomentumCycle: cycle
        )
    }
}

/// A trainer's complete resumable state: base weights (the fp32 masters under
/// mixed precision) followed by the optimizer velocity — the layout
/// `ChessTrainer.exportTrainerWeights()` produces — plus the schedule and the
/// dropout RNG state.
struct TrainerResumeSnapshot: Sendable {
    var trainerWeights: [[Float]]
    var schedule: TrainerScheduleState
    var dropoutRNG: DropoutRNGResumeState
    /// The relative gradient cap's per-step history
    /// (`trainer_grad_norm_history`), or `.notInCheckpoint` for a source
    /// written before it existed.
    var gradNormHistory: GradNormHistoryResumeState

    init(trainerWeights: [[Float]], schedule: TrainerScheduleState,
         dropoutRNG: DropoutRNGResumeState, gradNormHistory: GradNormHistoryResumeState) {
        self.trainerWeights = trainerWeights
        self.schedule = schedule
        self.dropoutRNG = dropoutRNG
        self.gradNormHistory = gradNormHistory
    }

    /// A snapshot whose source carries no gradient-norm history — a source
    /// that predates the relative cap. Every production path that has one
    /// passes it through the four-argument init.
    init(trainerWeights: [[Float]], schedule: TrainerScheduleState, dropoutRNG: DropoutRNGResumeState) {
        self.init(trainerWeights: trainerWeights, schedule: schedule, dropoutRNG: dropoutRNG,
                  gradNormHistory: .notInCheckpoint)
    }
}

/// Where a resume's dropout Philox state comes from.
enum DropoutRNGResumeState: Sendable, Equatable {
    /// Read from the saving trainer; restoring it continues that run's mask
    /// sequence exactly.
    case philox(DropoutPhiloxState)
    /// The source carries no dropout RNG state — every checkpoint file written
    /// before the lineage record stored it (`rng.dropout_philox_state`). The
    /// resumed run's masks continue from its own dropout seed instead, which
    /// is not the saved run's sequence once dropout is on.
    case notInCheckpoint

    /// The state a checkpoint file's lineage record carries, or
    /// `.notInCheckpoint` for a file without one (written before lineage)
    /// or whose record holds none.
    init(lineage: LineageRecord.Presence?) {
        if let state = lineage?.record?.rng.dropoutPhiloxState {
            self = .philox(state)
        } else {
            self = .notInCheckpoint
        }
    }

    /// The Philox state to record with a save of this snapshot, nil when it
    /// has none.
    var philoxState: DropoutPhiloxState? {
        switch self {
        case .philox(let state):
            return state
        case .notInCheckpoint:
            return nil
        }
    }
}

enum TrainerResumeError: Error, CustomStringConvertible, LocalizedError {
    /// The checkpoint cannot be resumed exactly; `missing` names what it lacks.
    case notExactlyResumable(file: String, missing: [String])

    var description: String {
        switch self {
        case .notExactlyResumable(let file, let missing):
            return "\(file) cannot be resumed exactly: it lacks \(missing.joined(separator: " and ")). "
                + "It was written by a build that did not save exact trainer state; continue from it as a new "
                + "branch (without the exact-resume flag) instead."
        }
    }

    var errorDescription: String? { description }
}

extension TrainerResumeSnapshot {
    /// The exact-resume state a checkpoint file carries, or a
    /// `TrainerResumeError` naming what it lacks. Shared by every runner that
    /// resumes from a single trainer-state file.
    init(checkpoint file: ModelCheckpointFile, fileName: String) throws {
        var missing: [String] = []
        if !file.includesOptimizerVelocity {
            missing.append("optimizer velocity tensors (opt.*.velocity)")
        }
        if file.metadata.trainerSchedule == nil {
            missing.append("trainer schedule metadata (\(TrainerScheduleState.MetadataKey.all.joined(separator: ", ")))")
        }
        guard missing.isEmpty, let schedule = file.metadata.trainerSchedule else {
            throw TrainerResumeError.notExactlyResumable(file: fileName, missing: missing)
        }
        self.init(trainerWeights: file.weights, schedule: schedule,
                  dropoutRNG: DropoutRNGResumeState(lineage: file.safetensorsProvenance?.lineage),
                  gradNormHistory: GradNormHistoryResumeState(file.metadata.trainerGradNormHistory))
    }
}

extension GradNormHistoryResumeState {
    /// `.restored` when a file carries a history, `.notInCheckpoint`
    /// otherwise.
    init(_ history: GradientNormHistory?) {
        if let history {
            self = .restored(history)
        } else {
            self = .notInCheckpoint
        }
    }
}

extension ChessTrainer {
    /// Snapshot everything an exact resume needs. Caller MUST have paused
    /// training (same contract as `exportTrainerWeights()`), which is also
    /// what keeps the clock read here consistent with the exported weights.
    func exportResumeSnapshot() async throws -> TrainerResumeSnapshot {
        let weights = try await exportTrainerWeights()
        let dropoutState = try await captureDropoutState()
        let history = try await exportGradNormHistory()
        let schedule = TrainerScheduleState(currentlyRunningOn: self)
        // Training is paused, so the history ends at the clock the schedule
        // read; a mismatch is a bug, refused here rather than saved.
        try history.checkEnds(atTrainerClock: schedule.completedTrainSteps)
        return TrainerResumeSnapshot(
            trainerWeights: weights,
            schedule: schedule,
            dropoutRNG: .philox(dropoutState),
            gradNormHistory: .restored(history)
        )
    }

    /// Restore a trainer to `snapshot` exactly: weights and fp32 masters,
    /// optimizer velocity, the completed-step clock, the warmup length, the
    /// LR/momentum cycle and — when the snapshot carries it — the dropout
    /// Philox state. The single restore path for GUI session resume,
    /// corpus-replay `--resume-exact` and train-vs-UCI `--resume-exact`.
    ///
    /// Warmup is not re-run: the warmup multiplier is
    /// `min(1, completedTrainSteps / lrWarmupSteps)`, so restoring the clock
    /// restores the multiplier at exactly the value the saved run had reached.
    /// Caller MUST have paused training.
    func restoreExactly(from snapshot: TrainerResumeSnapshot) async throws {
        try await loadTrainerWeights(snapshot.trainerWeights)
        lrWarmupSteps = snapshot.schedule.lrWarmupSteps
        lrMomentumCycle = snapshot.schedule.lrMomentumCycle
        completedTrainSteps = snapshot.schedule.completedTrainSteps
        switch snapshot.dropoutRNG {
        case .philox(let state):
            try await restoreDropoutState(state)
            SessionLogger.shared.log("[RESUME] rng: dropout=restored")
        case .notInCheckpoint:
            SessionLogger.shared.log(
                "[RESUME] rng: dropout=not restored (the checkpoint predates saved dropout RNG state; "
                + "masks continue from this run's own dropout seed)"
            )
        }
        switch snapshot.gradNormHistory {
        case .restored(let history):
            // Throws unless the history ends at the clock just restored: a
            // file whose history and clock disagree is internally
            // inconsistent and is not resumed from.
            try await restoreGradNormHistory(history)
            SessionLogger.shared.log(
                "[RESUME] grad-norm history: restored entries=\(history.count) "
                + "last_trainer_step=\(history.lastTrainerStep.map(String.init) ?? "none")"
            )
        case .notInCheckpoint:
            try await restoreGradNormHistory(GradientNormHistory())
            SessionLogger.shared.log(
                "[RESUME] grad-norm history: not in checkpoint (relative cap mode=\(relativeGradientCap.modeToken); "
                + "the relative cap warms up from an empty history)"
            )
        }
    }
}

/// How a CLI runner's trainer came to hold its weights, for the startup
/// schedule line.
enum TrainerLaunchKind: Sendable, Equatable {
    /// Random initialization.
    case fresh
    /// `--start-model` without `--resume-exact`: the file's weights, a fresh
    /// clock and zero velocity.
    case newBranch(fromModelID: String)
    /// `--resume-exact`: the file's complete trainer state.
    case exactResume(ofModelID: String)
}

extension LRMomentumCycleLogFormat {
    /// Where the schedule continues from, e.g. `origin=trainerStep 41000
    /// cycleStep 40000 lr=… mom=… (exact resume of 20260929-1-AbCd)`. The
    /// trainer step is the clock warmup, the cycle phase and the decay
    /// envelope are all read against; the LR and momentum are the values the
    /// next SGD step will use before √batch scaling.
    static func scheduleOrigin(of trainer: ChessTrainer, launch: TrainerLaunchKind) -> String {
        let steps = trainer.completedTrainSteps
        let cycleStep = LRMomentumCycle.cycleStep(completedTrainSteps: steps, lrWarmupSteps: trainer.lrWarmupSteps)
        let values = trainer.lrMomentumCycleValues(completedSteps: steps)
        let lrText = values.learningRate.map { String(format: "%.4e", $0) } ?? "static"
        let momentumText = values.momentum.map { String(format: "%.4f", $0) } ?? "static"
        let how: String
        switch launch {
        case .fresh:
            how = "fresh trainer"
        case .newBranch(let modelID):
            how = "new branch from \(modelID); its clock, warmup and velocity are not continued"
        case .exactResume(let modelID):
            how = "exact resume of \(modelID)"
        }
        return "origin=trainerStep \(steps) cycleStep \(cycleStep) cycleLR=\(lrText) cycleMom=\(momentumText) (\(how))"
    }
}

extension TrainerHyperparameters {
    /// These hyperparameters with the schedule fields replaced by a resumed
    /// checkpoint's. The CLI runners adopt a schedule through
    /// `ReplayParams.adoptingSchedule`, which rebuilds everything — these
    /// hyperparameters and the lineage snapshot — from one adopted parameter
    /// snapshot; the trainer configuration that yields equals this one.
    func adoptingSchedule(_ schedule: TrainerScheduleState) -> TrainerHyperparameters {
        var adopted = self
        adopted.lrWarmupSteps = schedule.lrWarmupSteps
        adopted.lrMomentumCycle = schedule.lrMomentumCycle
        return adopted
    }

    /// One line per schedule field whose configured value (from
    /// `--parameters` / the app's settings) differs from the value the
    /// resumed checkpoint trained under. An exact resume restores the
    /// checkpoint's value; the lines say so.
    func scheduleDifferences(from schedule: TrainerScheduleState) -> [String] {
        var lines: [String] = []
        if lrWarmupSteps != schedule.lrWarmupSteps {
            lines.append("lr_warmup_steps: configured \(lrWarmupSteps) but the checkpoint trained under \(schedule.lrWarmupSteps); restoring the checkpoint's value")
        }
        if lrMomentumCycle != schedule.lrMomentumCycle {
            lines.append(
                "lr_momentum_cycle: configured [\(LRMomentumCycleLogFormat.cycleDescription(lrMomentumCycle))] "
                    + "but the checkpoint trained under [\(LRMomentumCycleLogFormat.cycleDescription(schedule.lrMomentumCycle))]; "
                    + "restoring the checkpoint's value"
            )
        }
        return lines
    }
}
