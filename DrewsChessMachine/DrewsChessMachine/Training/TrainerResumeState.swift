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
    /// and the keyframes are evaluated against.
    var completedTrainSteps: Int
    /// LR warmup length, in trainer steps.
    var lrWarmupSteps: Int
    /// The LR/momentum cycle, keyframes included.
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
        case bothScheduleForms(legacyKeys: [String])
        case unconvertibleLegacySchedule(detail: String)

        var description: String {
            switch self {
            case .missingKey(let key):
                return "trainer schedule metadata is incomplete: '\(key)' is missing"
            case .malformedValue(let key, let value):
                return "trainer schedule metadata '\(key)' has malformed value '\(value)'"
            case .undecodableJSON(let key, let detail):
                return "trainer schedule metadata '\(key)' failed to decode: \(detail)"
            case .bothScheduleForms(let legacyKeys):
                return "trainer schedule metadata carries '\(MetadataKey.lrSchedule)' and the previous form's "
                    + "\(legacyKeys.joined(separator: ", ")); a file has one or the other"
            case .unconvertibleLegacySchedule(let detail):
                return "trainer schedule metadata (previous form) cannot be converted to LR keyframes: \(detail)"
            }
        }

        var errorDescription: String? { description }
    }

    /// The `__metadata__` keys this state is written under.
    enum MetadataKey {
        static let completedTrainSteps = "trainer_completed_steps"
        static let lrWarmupSteps = "trainer_lr_warmup_steps"
        /// The schedule (`LRMomentumCycle`'s persisted form, keyframes and
        /// all) — every file written since LR keyframes.
        static let lrSchedule = "trainer_lr_schedule"
        /// The keys this build writes.
        static let current: [String] = [completedTrainSteps, lrWarmupSteps, lrSchedule]

        /// The previous schedule's two keys (the cycle, and its decay
        /// envelope as a separate object). Read and converted; never written.
        static let legacyLRMomentumCycle = "trainer_lr_momentum_cycle"
        static let legacyLRMomentumCycleEnvelope = "trainer_lr_momentum_cycle_envelope"
        static let legacy: [String] = [legacyLRMomentumCycle, legacyLRMomentumCycleEnvelope]

        /// Every key that marks a file as carrying trainer state, in either
        /// form — what `ModelDerivation.requirePlainModelSource` refuses.
        static let all: [String] = current + legacy
    }

    /// The metadata entries for this state. `JSONEncoder` writes doubles in
    /// shortest round-trip form, so every keyframe value decodes back
    /// bit-exactly.
    func metadataEntries() throws -> [String: String] {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let scheduleJSON = try encoder.encode(lrMomentumCycle)
        return [
            MetadataKey.completedTrainSteps: String(completedTrainSteps),
            MetadataKey.lrWarmupSteps: String(lrWarmupSteps),
            MetadataKey.lrSchedule: String(decoding: scheduleJSON, as: UTF8.self),
        ]
    }

    /// The state stored in `metadata`, or nil when the file carries none of
    /// the `trainer_*` keys (a plain model file, or a trainer file written
    /// before exact resume existed). A file carrying only some of them, or a
    /// malformed value, throws: half a schedule is not a resumable one.
    ///
    /// A file written before LR keyframes carries the previous schedule's two
    /// keys instead of `trainer_lr_schedule`; they are converted
    /// (`LRMomentumCycle.converting`), so the resumed run feeds exactly what
    /// the saved one would have (owner decision Q-4). `conversionNotes` says
    /// what was converted, for the resume log. A file with both forms is
    /// refused: nothing ever writes one.
    static func decode(fromMetadata metadata: [String: String]) throws -> TrainerScheduleState? {
        try decodeReportingConversion(fromMetadata: metadata)?.state
    }

    static func decodeReportingConversion(
        fromMetadata metadata: [String: String]
    ) throws -> (state: TrainerScheduleState, conversionNotes: [String])? {
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
                throw MetadataError.undecodableJSON(key: key, detail: "\(error)")
            }
        }
        let completed = try integer(MetadataKey.completedTrainSteps)
        let warmup = try integer(MetadataKey.lrWarmupSteps)
        let hasCurrent = metadata[MetadataKey.lrSchedule] != nil
        let legacyPresent = MetadataKey.legacy.filter { metadata[$0] != nil }
        if hasCurrent {
            guard legacyPresent.isEmpty else { throw MetadataError.bothScheduleForms(legacyKeys: legacyPresent) }
            let cycle = try json(LRMomentumCycle.self, MetadataKey.lrSchedule)
            return (TrainerScheduleState(completedTrainSteps: completed, lrWarmupSteps: warmup, lrMomentumCycle: cycle), [])
        }
        let legacyCycle = try json(LegacyLRMomentumCycleCoding.self, MetadataKey.legacyLRMomentumCycle)
        let legacyEnvelope = try json(LegacyLRMomentumCycleEnvelopeCoding.self, MetadataKey.legacyLRMomentumCycleEnvelope)
        let converted: (cycle: LRMomentumCycle, notes: [String])
        do {
            converted = try LRMomentumCycle.converting(
                LegacyLRSchedule(cycle: legacyCycle, envelope: legacyEnvelope, lrWarmupSteps: warmup))
        } catch let error as LegacyLRScheduleConversionError {
            throw MetadataError.unconvertibleLegacySchedule(detail: error.description)
        }
        return (TrainerScheduleState(completedTrainSteps: completed, lrWarmupSteps: warmup, lrMomentumCycle: converted.cycle),
                converted.notes + [LRMomentumCycleLogFormat.keyframesDescription(converted.cycle.lrKeyframes)])
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

    /// Which schedule a restore leaves on the trainer.
    enum ScheduleOnRestore: Sendable, Equatable {
        /// The snapshot's warmup length and cycle (the GUI: the session's
        /// schedule is the configured one, already resolved onto the snapshot).
        case snapshots
        /// The schedule the trainer was configured with before the restore —
        /// the CLI `--resume-exact` rule (owner decision O-4): the configured
        /// schedule wins; a behavior difference from the checkpoint's is the
        /// `schedule` resume gap, decided before this call.
        case keepConfigured
    }

    /// Restore a trainer to `snapshot` exactly: weights and fp32 masters,
    /// optimizer velocity, the completed-step clock, the schedule as
    /// `schedule` says, and — when the snapshot carries it — the dropout
    /// Philox state. The single restore path for GUI session resume,
    /// corpus-replay `--resume-exact` and train-vs-UCI `--resume-exact`.
    ///
    /// Warmup is not re-run: the warmup multiplier is
    /// `min(1, completedTrainSteps / lrWarmupSteps)`, so restoring the clock
    /// restores the multiplier at exactly the value the saved run had reached.
    /// Caller MUST have paused training.
    func restoreExactly(from snapshot: TrainerResumeSnapshot, schedule: ScheduleOnRestore) async throws {
        try await loadTrainerWeights(snapshot.trainerWeights)
        switch schedule {
        case .snapshots:
            lrWarmupSteps = snapshot.schedule.lrWarmupSteps
            lrMomentumCycle = snapshot.schedule.lrMomentumCycle
        case .keepConfigured:
            break
        }
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

/// The trainer state a GUI arena captures at its start, under the training
/// pause, so a promotion can rewind the trainer to exactly it: the working
/// weights the candidate was snapshotted from, the optimizer velocity that
/// built them, the clock, the dropout Philox state and the relative cap's
/// gradient-norm history — everything indexed by or accumulated along that
/// trajectory.
struct ArenaStartTrainerState: Sendable {
    let weights: [[Float]]
    let velocity: [[Float]]
    let completedSteps: Int
    let dropoutState: DropoutPhiloxState
    let gradNormHistory: GradientNormHistory
}

extension ChessTrainer {
    /// Capture the arena-start state. Caller MUST have paused training, which
    /// is what keeps the clock read here consistent with the exports; the
    /// history must end at that clock, or the capture is refused.
    func captureArenaStartState() async throws -> ArenaStartTrainerState {
        let weights = try await network.exportWeights()
        let velocity = try await exportVelocitySnapshot()
        let completedSteps = completedTrainSteps
        let dropoutState = try await captureDropoutState()
        let gradNormHistory = try await exportGradNormHistory()
        try gradNormHistory.checkEnds(atTrainerClock: completedSteps)
        return ArenaStartTrainerState(weights: weights, velocity: velocity, completedSteps: completedSteps,
                                      dropoutState: dropoutState, gradNormHistory: gradNormHistory)
    }

    /// A promotion's rewind: the promoted candidate's `weights` (the
    /// arena-start weights) with the fp32 masters re-seeded from them, and
    /// the velocity, clock, dropout state and gradient-norm history captured
    /// with them at the arena start. The clock is set before the history,
    /// which must end at it. Caller MUST have paused training.
    ///
    /// Why the velocity is the snapshot's and not reset: it is the momentum
    /// that built the validated candidate, so it belongs to the candidate's
    /// weight surface, while the trainer's current velocity was built on the
    /// post-arena surface just discarded. Why the clock rewinds: warmup, the
    /// LR/momentum cycle and the decay envelope are functions of it, and a
    /// clock ahead of the weights would drive them with the wrong LR.
    func rewindToArenaStart(_ state: ArenaStartTrainerState, promotedWeights weights: [[Float]]) async throws {
        // Open the replacement window; the trainer's new identity, stamped by
        // the caller afterwards, closes it.
        noteWeightsReplaced()
        try await network.loadWeights(weights)
        try await syncMastersFromWorking()
        try await loadVelocitySnapshot(state.velocity)
        completedTrainSteps = state.completedSteps
        try await restoreDropoutState(state.dropoutState)
        try await restoreGradNormHistory(state.gradNormHistory)
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
    /// One line per schedule field whose configured value (from
    /// `--parameters` / the app's settings) changes what the trainer is fed
    /// relative to the resumed checkpoint's, compared in behavior form
    /// (`LRMomentumCycle.behaviorForm`, owner decision Q-5): an inert
    /// difference (a disabled channel's numbers, the independent momentum
    /// cycle while momentum follows the LR) is not a difference. Empty when
    /// the two feed the same values at every step from `clock` on.
    ///
    /// The warmup length matters while either run is still warming up at
    /// `clock`, or whenever a cycle is active (it offsets the cycle's phase).
    ///
    /// An exact CLI resume runs the **configured** schedule (owner decision
    /// O-4); a non-empty result is the `schedule` resume gap.
    func scheduleDifferences(from schedule: TrainerScheduleState) -> [String] {
        var lines: [String] = []
        let configured = lrMomentumCycle.behaviorForm
        let checkpoint = schedule.lrMomentumCycle.behaviorForm
        let anyCycleActive = configured.isActive || checkpoint.isActive
        let stillWarmingUp = schedule.completedTrainSteps < max(lrWarmupSteps, schedule.lrWarmupSteps)
        if lrWarmupSteps != schedule.lrWarmupSteps, anyCycleActive || stillWarmingUp {
            lines.append("lr_warmup_steps checkpoint=\(schedule.lrWarmupSteps) configured=\(lrWarmupSteps)")
        }
        if configured != checkpoint {
            lines.append(
                "lr_momentum_cycle checkpoint=[\(LRMomentumCycleLogFormat.cycleDescription(schedule.lrMomentumCycle))] "
                    + "configured=[\(LRMomentumCycleLogFormat.cycleDescription(lrMomentumCycle))]"
            )
        }
        return lines
    }
}
