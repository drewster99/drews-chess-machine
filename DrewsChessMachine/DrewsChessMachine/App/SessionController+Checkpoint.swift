import SwiftUI
import Darwin

/// `SessionController`'s checkpoint persistence — split out of
/// `SessionController.swift` to keep that file navigable. Holds the manual /
/// periodic Save Session paths, Save Champion as Model, the model/session
/// load paths, the `SessionCheckpointState` snapshot builder, and the
/// chart-data restore on resume. All state these touch is `var` (internal) on
/// `SessionController`, so this extension has full access. (`handleSaveSessionPeriodic`
/// is `internal` rather than `private` because the heartbeat's `periodicSaveTick`
/// in `SessionController.swift` calls it across files.)
extension SessionController {

    // MARK: - Checkpoint save / load + snapshot / resume

    /// Manual "Save Champion as Model" — writes a standalone
    /// `.dcmmodel` containing the current champion's weights.
    /// If Play-and-Train is active, pauses self-play worker 0
    /// briefly so the export doesn't race with in-flight
    /// inference calls on the shared champion graph, then
    /// resumes. Uses `pauseAndWait(timeoutMs:)` so a
    /// mid-save session end can't deadlock the save task.
    func handleSaveChampionAsModel() {
        // Belt-and-suspenders guards — menu disable is the primary
        // gate but these cover keyboard-shortcut / URL-scheme
        // invocations under a race.
        if checkpoint?.checkpointSaveInFlight == true {
            onRefuseMenuAction("A save is already in progress. Wait for it to finish.")
            return
        }
        if isArenaRunning {
            onRefuseMenuAction("Can't save the champion while the arena is running. Wait for it to finish.")
            return
        }
        if isBusyProvider() && !realTraining {
            onRefuseMenuAction("Another operation is in progress. Wait for it to finish, then try again.")
            return
        }
        guard let champion = network else {
            onRefuseMenuAction("Build or load a model first.")
            return
        }
        guard let championID = champion.identifier?.description else {
            onRefuseMenuAction("The champion has no model ID, so it cannot be saved as a model file.")
            return
        }
        // Snapshot the active self-play gate up front. If there
        // is no active session, we can safely export directly —
        // nobody is racing against us.
        let gate = activeSelfPlayGate
        checkpoint?.checkpointSaveInFlight = true
        checkpoint?.setCheckpointStatus("Saving champion…", kind: .progress)
        checkpoint?.startSlowSaveWatchdog(label: "champion save")

        Task {
            // Pause worker 0 if a session is running. Bail with a
            // user-visible error on timeout (indicates the session
            // has already ended or the worker is stuck — either way
            // we shouldn't spin forever).
            if let gate {
                let acquired = await gate.pauseAndWait(timeoutMs: Self.saveGateTimeoutMs)
                if !acquired {
                    checkpoint?.cancelSlowSaveWatchdog()
                    checkpoint?.checkpointSaveInFlight = false
                    checkpoint?.setCheckpointStatus("Save aborted: could not pause self-play (timeout)", kind: .error)
                    return
                }
            }

            var championWeights: [[Float]] = []
            var exportError: Error?
            do {
                championWeights = try await Task.detached(priority: .userInitiated) {
                    try await champion.exportWeights()
                }.value
            } catch {
                exportError = error
            }
            // Where the exported weights came from, read in the same pause:
            // the file's lineage record and training step describe these
            // weights, not whatever the champion holds later.
            let exportedOrigin = championOrigin
            gate?.resume()

            if let exportError {
                checkpoint?.cancelSlowSaveWatchdog()
                checkpoint?.checkpointSaveInFlight = false
                checkpoint?.setCheckpointStatus("Save failed (export): \(exportError.localizedDescription)", kind: .error)
                SessionLogger.shared.log("[CHECKPOINT] Save champion export failed: \(exportError.localizedDescription)")
                return
            }

            let saveDate = Date()
            let createdAtUnix = Int64(saveDate.timeIntervalSince1970)
            let championArch = champion.network.arch
            let lineage: LineageRecord
            let metadata: ModelCheckpointMetadata
            do {
                lineage = try Self.championFileLineageRecord(origin: exportedOrigin, at: saveDate)
                metadata = ModelCheckpointMetadata(
                    creator: "manual",
                    trainingStep: try Self.championFileTrainingStep(origin: exportedOrigin),
                    parentModelID: "",
                    notes: "Manual Save Champion export"
                )
            } catch {
                checkpoint?.cancelSlowSaveWatchdog()
                checkpoint?.checkpointSaveInFlight = false
                checkpoint?.setCheckpointStatus("Save failed (lineage): \(error.localizedDescription)", kind: .error)
                SessionLogger.shared.log("[CHECKPOINT] Save champion failed building its lineage record: \(error.localizedDescription)")
                return
            }

            let outcome: Result<URL, Error> = await Task.detached(priority: .userInitiated) {
                do {
                    let url = try await CheckpointManager.saveModel(
                        weights: championWeights,
                        modelID: championID,
                        createdAtUnix: createdAtUnix,
                        metadata: metadata,
                        architecture: championArch,
                        lineage: lineage,
                        testSetEvaluator: ModelTestSetEvaluator.modelFiles,
                        trigger: "manual"
                    )
                    return .success(url)
                } catch {
                    return .failure(error)
                }
            }.value
            checkpoint?.cancelSlowSaveWatchdog()
            checkpoint?.checkpointSaveInFlight = false
            switch outcome {
            case .success(let url):
                checkpoint?.setCheckpointStatus("Saved \(url.lastPathComponent)", kind: .success)
                SessionLogger.shared.log("[CHECKPOINT] Saved champion: \(url.lastPathComponent)")
            case .failure(let error):
                checkpoint?.setCheckpointStatus("Save failed: \(error.localizedDescription)", kind: .error)
                SessionLogger.shared.log("[CHECKPOINT] Save champion failed: \(error.localizedDescription)")
            }
        }
    }

    /// Upper bound on how long a save path will wait for a
    /// worker to acknowledge a pause request. Has to cover one
    /// in-flight self-play game or training step — a comfortable
    /// margin above the worst-case game length at typical self-play
    /// rates. On timeout the save bails with a user-visible error
    /// rather than blocking forever.
    nonisolated static let saveGateTimeoutMs: Int = 15_000

    /// Manual "Save Session" — writes a full `.dcmsession` with
    /// champion and trainer model files plus `session.json`, and the replay
    /// buffer when `includeReplayBuffer` (the Save Session sheet's choice).
    /// Requires an active Play-and-Train session and an available
    /// trainer. Briefly pauses both self-play worker 0 and the
    /// training gate to snapshot the two networks' weights.
    func handleSaveSessionManual(includeReplayBuffer: Bool) {
        // Belt-and-suspenders guards — menu disable is the primary
        // gate but these cover keyboard-shortcut / URL-scheme
        // invocations under a race, and a state change while the Save
        // Session sheet was open.
        if let reason = manualSaveSessionRefusal() {
            onRefuseMenuAction(reason)
            return
        }
        guard let champion = network,
              let trainer,
              let selfPlayGate = activeSelfPlayGate,
              let trainingGate = activeTrainingGate else {
            onRefuseMenuAction(Self.noTrainingSessionToSave)
            return
        }
        saveSessionInternal(
            champion: champion,
            trainer: trainer,
            selfPlayGate: selfPlayGate,
            trainingGate: trainingGate,
            trigger: .manual,
            includeReplayBuffer: includeReplayBuffer
        )
    }

    private static let noTrainingSessionToSave = "No active training session to save. Start Play and Train first."

    /// Why a manual Save Session cannot run now, or nil when it can: checked
    /// before the Save Session sheet opens and again when it saves.
    func manualSaveSessionRefusal() -> String? {
        if checkpoint?.checkpointSaveInFlight == true {
            return "A save is already in progress. Wait for it to finish."
        }
        if isArenaRunning {
            return "Can't save the session while the arena is running. Wait for it to finish."
        }
        guard realTraining, network != nil, trainer != nil,
              activeSelfPlayGate != nil, activeTrainingGate != nil else {
            return Self.noTrainingSessionToSave
        }
        return nil
    }

    /// The Save Session sheet's request: the checkbox starts from
    /// `session_save_include_replay_buffer`, and the size is the live
    /// buffer's on disk (stored positions × bytes per position).
    func saveSessionSheetRequest() -> SaveSessionSheetRequest {
        let sizeText: String
        if let buffer = replayBuffer {
            sizeText = "≈ " + BinaryByteCount.text(buffer.count * buffer.bytesPerPosition)
        } else {
            sizeText = "no buffer"
        }
        return SaveSessionSheetRequest(
            initialIncludeReplayBuffer: TrainingParameters.shared.sessionSaveIncludeReplayBuffer,
            replayBufferSizeText: sizeText)
    }

    /// SIGUSR2 entry point — "checkpoint now, then shut down." Writes a full
    /// `.dcmsession` through the same path as the manual save (gate dance,
    /// weight export, resume-pointer record) tagged `.signalSave`, and on
    /// SUCCESS exits the process so the next launch auto-resumes. On FAILURE it
    /// logs loudly and STAYS RUNNING — never trade the live session for a failed
    /// checkpoint. Also stays running (refuses) when there's no active session,
    /// a save is already in flight, or an arena is running.
    func handleSaveSessionFromSignal() {
        SessionLogger.shared.log("[SIGUSR2] save-and-shutdown requested")
        if checkpoint?.checkpointSaveInFlight == true {
            SessionLogger.shared.log("[SIGUSR2] a save is already in progress; ignoring (staying running)")
            return
        }
        if isArenaRunning {
            SessionLogger.shared.log("[SIGUSR2] arena running; cannot checkpoint now — staying running")
            return
        }
        guard realTraining,
              let champion = network,
              let trainer,
              let selfPlayGate = activeSelfPlayGate,
              let trainingGate = activeTrainingGate else {
            SessionLogger.shared.log("[SIGUSR2] no active training session to save — staying running")
            return
        }
        saveSessionInternal(
            champion: champion,
            trainer: trainer,
            selfPlayGate: selfPlayGate,
            trainingGate: trainingGate,
            trigger: .signalSave,
            includeReplayBuffer: TrainingParameters.shared.sessionSaveIncludeReplayBuffer,
            onComplete: { success in
                if success {
                    SessionLogger.shared.log("[SIGUSR2] session checkpoint written — shutting down")
                    // Graceful path (not a hard-kill): flush the log tail so the
                    // deliberate shutdown is traceable, then exit. The
                    // `.dcmsession` is already on disk (CheckpointManager writes
                    // synchronously); `shutdown()` is `queue.sync` so it FIFOs
                    // behind the pending log writes without deadlocking.
                    SessionLogger.shared.shutdown()
                    Darwin._exit(0)
                } else {
                    SessionLogger.shared.log("[SIGUSR2] session checkpoint FAILED — staying running (not shutting down)")
                }
            }
        )
    }

    /// Fired by `PeriodicSaveController` when its 4-hour deadline
    /// elapses (after any arena-deferral has resolved). Behaves
    /// exactly like the manual save path but tagged `.periodic` so
    /// the filename, status-line, and log line distinguish the two
    /// triggers. The controller has already decided we should fire;
    /// any remaining guard failures here (no session, save already
    /// in flight) just make the periodic attempt a no-op — the next
    /// tick of the controller will re-fire since `noteSuccessfulSave`
    /// is never called.
    @MainActor
    func handleSaveSessionPeriodic() {
        // Guard against an arena starting in the tiny race window
        // between the controller's decide() and this call.
        if isArenaRunning {
            return
        }
        if checkpoint?.checkpointSaveInFlight == true {
            SessionLogger.shared.log("[CHECKPOINT] Periodic save skipped — another save is in flight")
            return
        }
        guard realTraining,
              let champion = network,
              let trainer,
              let selfPlayGate = activeSelfPlayGate,
              let trainingGate = activeTrainingGate else {
            // Should not happen if the controller is armed correctly,
            // but disarm ourselves and bail to be safe.
            periodicSaveController?.disarm()
            return
        }
        periodicSaveInFlight = true
        saveSessionInternal(
            champion: champion,
            trainer: trainer,
            selfPlayGate: selfPlayGate,
            trainingGate: trainingGate,
            trigger: .periodic,
            includeReplayBuffer: TrainingParameters.shared.sessionSaveIncludeReplayBuffer
        )
    }

    /// Shared save-session internal used by the manual save button,
    /// the periodic autosave, and the post-"Promote Trainee Now"
    /// autosave (`SessionController+ManualPromote.swift`). Handles the
    /// gate dance, exports both networks, builds the session state on
    /// the main actor, and fires off the actual write to a detached
    /// task. The *arena's* post-promotion autosave uses its own inline
    /// code path (in the arena coordinator) instead, because it re-uses
    /// weights already snapshotted under the arena's own pause and so
    /// does not need to dance the gates again here. (`internal` rather
    /// than `private` so the manual-promote path in another extension
    /// file can reach it.) `includeReplayBuffer` decides whether the save
    /// writes the replay buffer (determinism plan D-8): the Save Session
    /// sheet's choice for a manual save, `session_save_include_replay_buffer`
    /// for every other trigger. `sessionsDirectory` is the canonical
    /// `Sessions/` folder in the app; tests pass a temporary folder, and
    /// the retention sweep a save starts looks only in the folder it wrote
    /// to.
    func saveSessionInternal(
        champion: ChessMPSNetwork,
        trainer: ChessTrainer,
        selfPlayGate: WorkerPauseGate,
        trainingGate: WorkerPauseGate,
        trigger: SessionSaveTrigger,
        includeReplayBuffer: Bool,
        sessionsDirectory: URL = CheckpointPaths.sessionsDir,
        onComplete: (@MainActor @Sendable (Bool) -> Void)? = nil
    ) {
        let diskTag = trigger.diskTag
        let uiSuffix = trigger.uiSuffix
        // Both files are written under these IDs and the trainer file names
        // the champion as its parent; a network without an identity is a
        // failed save, never a file under a placeholder ID.
        guard let championID = champion.identifier?.description,
              let trainerID = trainer.identifier?.description else {
            checkpoint?.checkpointSaveInFlight = false
            periodicSaveInFlight = false
            let message = "Save failed: \(RunCutError.noModelID.localizedDescription)"
            checkpoint?.setCheckpointStatus("\(message)\(uiSuffix)", kind: .error)
            SessionLogger.shared.log("[CHECKPOINT] Save session (\(diskTag)) failed: \(message)")
            onComplete?(false)
            return
        }
        checkpoint?.checkpointSaveInFlight = true
        checkpoint?.setCheckpointStatus("Saving session\(uiSuffix)…", kind: .progress)
        checkpoint?.startSlowSaveWatchdog(label: "session save\(uiSuffix)")

        // Capture the replay buffer handle so the detached write path can
        // serialize it alongside the two network files — `ReplayBuffer`
        // is `@unchecked Sendable` and serializes access via its own
        // lock, so the buffer can be written from a background task
        // while self-play workers (which only append) are paused.
        let bufferForSave = includeReplayBuffer ? replayBuffer : nil

        Task {
            // Fire `onComplete(success)` exactly once on every exit path
            // (all the error `return`s and the normal end) via `defer`, so a
            // caller like the SIGUSR2 save-and-shutdown handler can act on the
            // outcome. `didSucceed` flips true only in the success branch
            // below; the defer reads its final value at scope exit. nil
            // onComplete (manual / periodic / promotion callers) ⇒ no-op.
            var didSucceed = false
            defer { onComplete?(didSucceed) }

            // Helper to clear both in-flight flags consistently on
            // every early-return path below. The periodic flag is
            // only meaningful when `trigger == .periodic`, but it's
            // cheap to always clear so we don't have to repeat the
            // branch on every error exit. Cancels the slow-save
            // watchdog too so a fast-failure path doesn't leave a
            // stale "Saving… (still running)" amber line behind.
            @MainActor func clearInFlight() {
                checkpoint?.cancelSlowSaveWatchdog()
                checkpoint?.checkpointSaveInFlight = false
                periodicSaveInFlight = false
            }

            // One consistent cut. Self-play is paused for the champion
            // export and stays paused while training is paused for the
            // trainer export, and the run's record — the replay buffer's
            // sampler, the next game serial, the fed counts — is read
            // under both pauses, so a resume that continues those streams
            // continues them from the instant the trainer state was taken.
            // Releasing self-play earlier let new games take serials and
            // feed the buffer before the record read them. The session
            // state — step and game counts included — is built under both
            // pauses too, after the record. Training is released once both
            // are built; self-play once the
            // replay buffer is written when the save includes it (the
            // written buffer must be the one the record describes), right
            // after the record otherwise. Bounded waits, so a session end
            // mid-save cannot leave the save waiting on exited workers.
            let selfPlayAcquired = await selfPlayGate.pauseAndWait(timeoutMs: Self.saveGateTimeoutMs)
            guard selfPlayAcquired else {
                clearInFlight()
                checkpoint?.setCheckpointStatus("Save aborted: could not pause self-play (timeout)", kind: .error)
                SessionLogger.shared.log("[CHECKPOINT] Save session aborted at self-play pause timeout")
                return
            }
            // Every exit from here on releases self-play through this
            // hold, which resumes the gate once whichever path gets there
            // first.
            let selfPlayHold = WorkerPauseGateHold(selfPlayGate)
            defer { selfPlayHold.release() }
            if let dropped = activeSelfPlayPauseDrops?.value {
                SessionLogger.shared.log("[CHECKPOINT] dropped \(dropped.games) in-flight games (\(dropped.plies) plies) at the save's self-play pause; the save holds none of them")
            }
            var championWeights: [[Float]] = []
            var championError: Error?
            do {
                championWeights = try await Task.detached(priority: .userInitiated) {
                    try await champion.exportWeights()
                }.value
            } catch {
                championError = error
            }
            // The champion file's record and step describe the exported
            // weights, so their origin is read in the same pause.
            let exportedChampionOrigin = championOrigin

            if let championError {
                clearInFlight()
                checkpoint?.setCheckpointStatus("Save failed (champion export): \(championError.localizedDescription)", kind: .error)
                SessionLogger.shared.log("[CHECKPOINT] Save session failed at champion export: \(championError.localizedDescription)")
                return
            }

            let trainingAcquired = await trainingGate.pauseAndWait(timeoutMs: Self.saveGateTimeoutMs)
            guard trainingAcquired else {
                clearInFlight()
                checkpoint?.setCheckpointStatus("Save aborted: could not pause training (timeout)", kind: .error)
                SessionLogger.shared.log("[CHECKPOINT] Save session aborted at training pause timeout")
                return
            }
            // The save's configuration cut, in this main-actor turn under the
            // training pause (gap 5, review X4): the record's parameters and
            // journals, and the trainer file's schedule, all describe this
            // instant, whatever the popover commits during the awaits below.
            let configurationCut: GuiConfigurationCut
            do {
                configurationCut = try takeConfigurationCut(trainer: trainer)
            } catch {
                trainingGate.resume()
                clearInFlight()
                checkpoint?.setCheckpointStatus("Save failed (configuration cut): \(error.localizedDescription)", kind: .error)
                SessionLogger.shared.log("[CHECKPOINT] Save session failed at the configuration cut: \(error.localizedDescription)")
                return
            }
            // The complete resumable trainer state: trainables + BN (fp32
            // masters under mixed precision), momentum velocity, and the
            // completed-step clock + schedule read under the same pause.
            let trainerExport: Result<TrainerResumeSnapshot, Error>
            do {
                trainerExport = .success(try await Task.detached(priority: .userInitiated) {
                    try await trainer.exportResumeSnapshot()
                }.value)
            } catch {
                trainerExport = .failure(error)
            }
            let trainerSnapshot: TrainerResumeSnapshot
            switch trainerExport {
            case .success(let snapshot) where snapshot.schedule.completedTrainSteps != configurationCut.schedule.completedTrainSteps:
                // Training is paused, so the clock cannot move between the
                // cut and the export; a difference is a bug, never written.
                let error = LineageSegmentError.configurationCutClockMoved(
                    cut: configurationCut.schedule.completedTrainSteps, exported: snapshot.schedule.completedTrainSteps)
                trainingGate.resume()
                clearInFlight()
                checkpoint?.setCheckpointStatus("Save failed: \(error.localizedDescription)", kind: .error)
                SessionLogger.shared.log("[CHECKPOINT] Save session failed: \(error.localizedDescription)")
                return
            case .success(let snapshot):
                // The exported weights, velocity and dropout state, with the
                // schedule the cut read: the file's flat `trainer_*` keys
                // then come from the same value as the record's.
                trainerSnapshot = TrainerResumeSnapshot(trainerWeights: snapshot.trainerWeights,
                                                        schedule: configurationCut.schedule,
                                                        dropoutRNG: snapshot.dropoutRNG,
                                                        gradNormHistory: snapshot.gradNormHistory)
            case .failure(let trainerError):
                trainingGate.resume()
                clearInFlight()
                checkpoint?.setCheckpointStatus("Save failed (trainer export): \(trainerError.localizedDescription)", kind: .error)
                SessionLogger.shared.log("[CHECKPOINT] Save session failed at trainer export: \(trainerError.localizedDescription)")
                return
            }
            let trainerWeights = trainerSnapshot.trainerWeights
            // The training-health stamp of exactly the exported state, under
            // the same training pause (D2).
            let checkpointHealth = makeTrainingHealthCheckpoint()
            // The run's lineage at this save, for the session's trainer
            // file and session.json, read under both pauses; the champion
            // file's own record and training step come from where its
            // weights came from.
            let saveDate = Date()
            let lineageResult: Result<(run: LineageRecord, champion: LineageRecord, championMetadata: ModelCheckpointMetadata), Error>
            do {
                let run = try lineageRecordForSave(
                    at: saveDate, cut: configurationCut,
                    trainerCompletedSteps: trainerSnapshot.schedule.completedTrainSteps,
                    dropoutPhiloxState: trainerSnapshot.dropoutRNG.philoxState,
                    dropoutStreamState: try await trainer.dropoutStreamState())
                let champion = try Self.championFileLineageRecord(origin: exportedChampionOrigin, at: saveDate)
                let championMetadata = ModelCheckpointMetadata(
                    creator: diskTag,
                    trainingStep: try Self.championFileTrainingStep(origin: exportedChampionOrigin),
                    parentModelID: "",
                    notes: "Session checkpoint (\(diskTag))"
                )
                lineageResult = .success((run, champion, championMetadata))
            } catch {
                lineageResult = .failure(error)
            }
            // The session state, the trainer file's training step and the
            // chart companion files, still under both pauses: the run's
            // counters are published from the live boxes and the state is
            // built from them in one main-actor stretch, so session.json's
            // step and game counts are the ones the trainer state, the
            // record and the buffer describe.
            let sessionStateResult: Result<(state: SessionCheckpointState, trainingStep: Int, chartSnapshot: ChartCoordinatorSnapshot?), Error>
            do {
                let trainingStep = try publishRunCountersAtCut()
                let sessionState = try buildCurrentSessionState(
                    championID: championID,
                    trainerID: trainerID,
                    arenaClock: .live,
                    includeReplayBuffer: includeReplayBuffer
                )
                // `buildSnapshot()` returns nil when chart collection is off
                // or both rings are empty, in which case the save skips the
                // chart-companion files entirely.
                sessionStateResult = .success((sessionState, trainingStep, chartCoordinator?.buildSnapshot()))
            } catch {
                sessionStateResult = .failure(error)
            }
            trainingGate.resume()
            if bufferForSave == nil {
                selfPlayHold.release()
            }
            let lineage: LineageRecord
            let championLineage: LineageRecord
            let championMetadata: ModelCheckpointMetadata
            switch lineageResult {
            case .success(let records):
                lineage = records.run
                championLineage = records.champion
                championMetadata = records.championMetadata
            case .failure(let error):
                clearInFlight()
                checkpoint?.setCheckpointStatus("Save failed (lineage): \(error.localizedDescription)", kind: .error)
                SessionLogger.shared.log("[CHECKPOINT] Save session failed building its lineage record: \(error.localizedDescription)")
                return
            }
            let sessionState: SessionCheckpointState
            let trainingStep: Int
            let chartSnapshotForSave: ChartCoordinatorSnapshot?
            switch sessionStateResult {
            case .success(let cut):
                sessionState = cut.state
                trainingStep = cut.trainingStep
                chartSnapshotForSave = cut.chartSnapshot
            case .failure(let error):
                clearInFlight()
                checkpoint?.setCheckpointStatus("Save failed (session state): \(error.localizedDescription)", kind: .error)
                SessionLogger.shared.log("[CHECKPOINT] Save session failed building its session state: \(error.localizedDescription)")
                return
            }

            // Final write + verify on a detached task so UI stays
            // responsive during the scratch-network build (sub-second).
            // The trainer file states its own snapshot's clock (format v11:
            // `training_step` is the trainer step, and on a trainer-state
            // file it must equal the schedule's clock); the writer reads one
            // source. The stats box's count at the cut is the same number
            // after a fresh start or a session resume
            // (`SessionSaveConsistentCutTests`), but not after "New Session,
            // keep trainer", whose box counts the session from 0 — another
            // reason the file never states the box's count.
            let trainerMetadata = ModelCheckpointMetadata.trainerFile(
                creator: diskTag,
                trainingStep: trainerSnapshot.schedule.completedTrainSteps,
                parentModelID: championID,
                notes: "Trainer lineage at session checkpoint (\(diskTag))",
                schedule: trainerSnapshot.schedule,
                gradNormHistory: trainerSnapshot.gradNormHistory.history
            )
            let now = Int64(Date().timeIntervalSince1970)
            // Champion and trainer share a topology; the trainer was built to
            // the champion's arch, so it's the authoritative source.
            let sessionArch = trainer.arch
            let outcome: Result<URL, Error> = await Task.detached(priority: .userInitiated) {
                [bufferForSave, chartSnapshotForSave] in
                do {
                    let url = try await CheckpointManager.saveSession(
                        championWeights: championWeights,
                        championID: championID,
                        championMetadata: championMetadata,
                        championCreatedAtUnix: now,
                        trainerWeights: trainerWeights,
                        trainerID: trainerID,
                        trainerMetadata: trainerMetadata,
                        trainerCreatedAtUnix: now,
                        state: sessionState,
                        lineage: lineage,
                        championLineage: championLineage,
                        architecture: sessionArch,
                        testSetEvaluator: ModelTestSetEvaluator.modelFiles,
                        replayBuffer: bufferForSave,
                        chartSnapshot: chartSnapshotForSave,
                        trigger: diskTag,
                        sessionsDirectory: sessionsDirectory,
                        onReplayBufferWritten: { selfPlayHold.release() }
                    )
                    return .success(url)
                } catch {
                    return .failure(error)
                }
            }.value

            clearInFlight()
            switch outcome {
            case .success(let url):
                didSucceed = true
                checkpoint?.setCheckpointStatus("Saved \(url.lastPathComponent)\(uiSuffix)", kind: .success)
                SessionLogger.shared.log(
                    "[CHECKPOINT] Saved session (\(diskTag)): \(url.lastPathComponent) build=\(BuildInfo.buildNumber) git=\(BuildInfo.gitHash)"
                        + Self.savedReplayBufferLogFields(writtenBuffer: bufferForSave)
                )
                checkpoint?.recordLastSessionPointer(
                    directoryURL: url,
                    sessionID: sessionState.sessionID,
                    trigger: diskTag
                )
                periodicSaveController?.noteSuccessfulSave(at: Date())
                checkpoint?.lastSavedAt = Date()
                checkpoint?.lastResumedAt = nil
                Self.logSavedTrainerLayerHealth(
                    arch: sessionArch,
                    trainerWeights: trainerWeights,
                    context: "session-\(diskTag)",
                    step: trainingStep,
                    trainerStep: trainerSnapshot.schedule.completedTrainSteps,
                    recorder: cliRecorder,
                    trainingHealth: checkpointHealth)
                // Periodic and Promote Trainee Now saves are in the
                // automatic-save retention pool; manual and SIGUSR2 saves
                // are not, and the helper decides that from the disk tag.
                scheduleAutomaticSaveRetentionSweep(afterSaving: url, diskTag: diskTag, in: sessionsDirectory)
            case .failure(let error):
                checkpoint?.setCheckpointStatus("Save failed: \(error.localizedDescription)", kind: .error)
                SessionLogger.shared.log("[CHECKPOINT] Save session (\(diskTag)) failed: \(error.localizedDescription)")
            }
        }
    }

    /// Log the full `[LAYER-HEALTH]` checkpoint block for a trainer state a
    /// session save just wrote, and record it on the `--output` recorder
    /// when there is one. Runs detached at utility priority from the
    /// already-exported weights (no extra GPU read, no gate pause); the scan
    /// itself hops to a GCD queue inside `LayerHealthLog.checkpoint`. Never
    /// throws — a failed pass logs its own `[LAYER-HEALTH]` failure line.
    nonisolated static func logSavedTrainerLayerHealth(
        arch: NetworkArchitecture,
        trainerWeights: [[Float]],
        context: String,
        step: Int,
        trainerStep: Int,
        recorder: CliTrainingRecorder?,
        trainingHealth: GuiTrainingHealthCheckpoint?
    ) {
        Task.detached(priority: .utility) {
            let health = await LayerHealthLog.checkpoint(
                arch: arch, trainerWeights: trainerWeights, context: context,
                step: step, trainerStep: trainerStep)
            for line in health.lines {
                SessionLogger.shared.log(line)
            }
            if let recorder, let summary = health.summary {
                recorder.appendLayerHealth(CliTrainingRecorder.LayerHealthRecord(
                    step: step, trainerStep: trainerStep, context: context, summary: summary))
            }
            // The training-health checkpoint evaluation of the same pass
            // (rules 1, 2, 3, 8), judged under the settings in force when
            // the state was exported; the stop decision is made on the main
            // actor when it arrives (R2). A failed pass is no observation.
            if let trainingHealth, let summary = health.summary {
                let evaluation = trainingHealth.monitor.evaluateCheckpoint(
                    stamp: trainingHealth.stamp, layerHealth: LayerHealthDigest(summary: summary),
                    digestTrainerStep: trainerStep, config: trainingHealth.config,
                    log: GuiTrainingHealthWorker.logSink)
                if let recorder, let evaluation {
                    recorder.appendAlarmEvents(evaluation.events)
                }
                trainingHealth.deliver(trainingHealth.monitor)
            }
        }
    }

    /// Start the automatic-save retention sweep
    /// (`CheckpointPaths.pruneAutomaticSaves`) after a successful
    /// session save to `url` tagged `diskTag` — but only when saves with
    /// that tag are in the retention pool (`AutomaticSaveKind`): periodic
    /// autosaves and promotion saves. Manual and SIGUSR2 saves never start
    /// a sweep, which keeps a deliberate save from deleting anything.
    ///
    /// Every save path that can write a pool member calls this exactly
    /// once on success, and nothing else calls it: `saveSessionInternal`
    /// (periodic, Promote Trainee Now — and manual and SIGUSR2, which this
    /// filters out) and the arena's inline post-promotion save. Deciding
    /// from the disk tag rather than from each caller's own condition is
    /// what keeps the "which saves start a sweep" rule identical to the
    /// "which saves are in the pool" rule.
    ///
    /// Whether a pool member's save actually starts a sweep is then
    /// `CheckpointPaths.automaticSavePruningDecision` — the build's kill
    /// switch (`automaticSavePruningForcedOff`), the
    /// `automaticSavePruningEnabled` setting and the
    /// `maxPeriodicAutosavesKept` cap — and nothing else. When it says
    /// no, exactly one `[PRUNE] skipped` line names the reason and both
    /// settings, so a save that deleted nothing is never silent about why.
    ///
    /// The setting and the cap are read live here, on the main actor, so
    /// a change made mid-session applies to the next automatic save. The
    /// sweep itself is file-system work, so it runs detached at utility
    /// priority; it reports everything through the session log and does
    /// not throw, so nothing is lost by not awaiting it.
    func scheduleAutomaticSaveRetentionSweep(afterSaving url: URL, diskTag: String, in sessionsDirectory: URL) {
        guard CheckpointPaths.AutomaticSaveKind(diskTag: diskTag) != nil else { return }
        let current = currentAutomaticSavePruningDecision()
        guard case .prune(let keep) = current.decision else {
            SessionLogger.shared.log(
                "[PRUNE] skipped after \(url.lastPathComponent): \(current.logDescription)"
            )
            return
        }
        Task.detached(priority: .utility) {
            CheckpointPaths.pruneAutomaticSaves(keeping: keep, protecting: url, in: sessionsDirectory)
        }
    }

    /// One `[PRUNE]` line stating whether automatic-save pruning would run
    /// for this session's automatic saves, and why, with both settings.
    /// Logged when Play-and-Train starts — after a resumed session's saved
    /// values have been restored — so a run's log shows the effective state
    /// before its first automatic save. Uses the same decision as
    /// `scheduleAutomaticSaveRetentionSweep`, so the two cannot disagree.
    func logAutomaticSavePruningState() {
        SessionLogger.shared.log(
            "[PRUNE] automatic-save pruning at Play-and-Train start: \(currentAutomaticSavePruningDecision().logDescription)"
        )
    }

    /// The pruning decision for the current live settings, and its log
    /// text. The one place the build's kill switch, the
    /// `automaticSavePruningEnabled` setting and the
    /// `maxPeriodicAutosavesKept` cap are read together.
    private func currentAutomaticSavePruningDecision() -> (
        decision: CheckpointPaths.AutomaticSavePruningDecision,
        logDescription: String
    ) {
        let settingEnabled = TrainingParameters.shared.automaticSavePruningEnabled
        let cap = TrainingParameters.shared.maxPeriodicAutosavesKept
        let decision = CheckpointPaths.automaticSavePruningDecision(
            forcedOff: CheckpointPaths.automaticSavePruningForcedOff,
            settingEnabled: settingEnabled,
            cap: cap
        )
        return (decision, decision.logDescription(settingEnabled: settingEnabled, cap: cap))
    }

    /// Load a standalone `.dcmmodel` into the current champion
    /// network. Triggered from the Load Model file importer. The
    /// network must exist (loading into a built network preserves
    /// the existing graph compilation; we don't rebuild).
    func handleLoadModelPickResult(_ result: Result<[URL], Error>) {
        switch result {
        case .failure(let error):
            checkpoint?.setCheckpointStatus("Load cancelled: \(error.localizedDescription)", kind: .error)
        case .success(let urls):
            guard let url = urls.first else { return }
            loadModelFrom(url: url)
        }
    }

    func loadModelFrom(url: URL) {
        // In-function guards (belt-and-suspenders with menu disable).
        if isBuildingOrBusyProvider() {
            onRefuseMenuAction(busyReasonProvider())
            return
        }
        Task {
            _ = await performLoadModel(url: url)
        }
    }

    /// Awaitable core of `loadModelFrom(url:)` — read + decode the model
    /// file, build (or rebuild) the champion at its architecture, and apply
    /// the weights. Returns `true` on success. Exposed separately so the
    /// `--train --start-model` launch sequence can sequence "load champion,
    /// THEN start Play-and-Train" without polling UI state.
    @discardableResult
    func performLoadModel(url: URL) async -> Bool {
        checkpoint?.checkpointSaveInFlight = true
        checkpoint?.setCheckpointStatus("Loading \(url.lastPathComponent)…", kind: .progress)
        return await withCheckedContinuation { continuation in
            performLoadModelBody(url: url, completion: { continuation.resume(returning: $0) })
        }
    }

    private func performLoadModelBody(url: URL, completion: @escaping @MainActor (Bool) -> Void) {
        Task {
            // 1. Read + decode the file first (CPU) to learn its architecture.
            //    The security scope is held across the read; loadWeights below
            //    works off the in-memory decode and needs no file access.
            let readResult: Result<ModelCheckpointFile, Error> = await Task.detached(priority: .userInitiated) {
                let scopeAccessed = url.startAccessingSecurityScopedResource()
                defer {
                    if scopeAccessed {
                        url.stopAccessingSecurityScopedResource()
                    }
                }
                do {
                    return .success(try CheckpointManager.loadModelFile(at: url))
                } catch {
                    return .failure(error)
                }
            }.value
            guard case .success(let file) = readResult else {
                checkpoint?.checkpointSaveInFlight = false
                if case .failure(let error) = readResult {
                    checkpoint?.setCheckpointStatus("Load failed: \(error.localizedDescription)", kind: .error)
                    SessionLogger.shared.log("[CHECKPOINT] Load model failed: \(error.localizedDescription)")
                }
                completion(false)
                return
            }

            // 2. Build (or rebuild) the champion at the file's architecture so a
            //    non-default / historical model gets a matching graph. The random
            //    init only satisfies graph compilation; weights overwrite it next.
            let championResult = await self.ensureChampionBuilt(arch: file.architecture)
            guard case .success(let champion) = championResult else {
                checkpoint?.checkpointSaveInFlight = false
                if case .failure(let error) = championResult {
                    checkpoint?.setCheckpointStatus("Build failed: \(error.localizedDescription)", kind: .error)
                    SessionLogger.shared.log("[CHECKPOINT] Load model auto-build failed: \(error.localizedDescription)")
                }
                completion(false)
                return
            }

            // 3. Apply weights.
            // `adoptLoadedChampionOrigin` closes this; a failed load leaves it
            // open, since the champion's weights are then not known to match
            // its recorded origin.
            noteChampionWeightsReplaced()
            let applyResult: Result<Void, Error> = await Task.detached(priority: .userInitiated) {
                do {
                    try await champion.loadWeights(file.networkWeights)
                    return .success(())
                } catch {
                    return .failure(error)
                }
            }.value
            checkpoint?.checkpointSaveInFlight = false
            switch applyResult {
            case .success:
                champion.identifier = ModelID(value: file.modelID)
                // A branch from this champion records the file as its parent.
                adoptLoadedChampionOrigin(file)
                networkStatus = "Loaded model \(file.modelID)\nFrom: \(url.lastPathComponent)"
                checkpoint?.setCheckpointStatus("Loaded \(file.modelID)", kind: .success)
                SessionLogger.shared.log("[CHECKPOINT] Loaded model: \(url.lastPathComponent) → \(file.modelID)")
                SessionLogger.shared.logArchitecture(
                    event: "loaded model \(url.lastPathComponent) → \(file.modelID)",
                    arch: file.architecture
                )
                onClearInferenceResult()
                // Flag champion-replaced for the post-Stop Start dialog's
                // "Continue" annotation. Cleared as soon as a new training
                // segment starts.
                if replayBuffer != nil {
                    championLoadedSinceLastTrainingSegment = true
                }
                completion(true)
            case .failure(let error):
                checkpoint?.setCheckpointStatus("Load failed: \(error.localizedDescription)", kind: .error)
                SessionLogger.shared.log("[CHECKPOINT] Load model failed: \(error.localizedDescription)")
                completion(false)
            }
        }
    }

    /// Load a `.dcmsession` directory. Parses everything, loads
    /// champion weights immediately into the live champion
    /// network, and stores the session state + trainer weights
    /// in `pendingLoadedSession` so the next Play-and-Train start
    /// resumes from them.
    func handleLoadSessionPickResult(_ result: Result<[URL], Error>) {
        switch result {
        case .failure(let error):
            checkpoint?.setCheckpointStatus("Load cancelled: \(error.localizedDescription)", kind: .error)
        case .success(let urls):
            guard let url = urls.first else { return }
            loadSessionFrom(url: url)
        }
    }

    /// `forceFloat32`: when true, the loaded session's `computeDataType` is
    /// overridden to `.float32` for the champion — and therefore for the trainer
    /// and every inference/arena network, which all derive their arch from
    /// `network.network.arch`. Used to load a bf16-trained session and continue
    /// training in fp32 (lossless bf16→fp32 widen on weight load), to dodge the
    /// Xcode 27 / macOS 27 beta bf16 training stomp. Driven by the auto-resume
    /// sheet's "Load as float32" checkbox.
    func loadSessionFrom(url: URL, startAfterLoad: Bool = false, forceFloat32: Bool = false) {
        loadSessionFrom(url: url, startAfterLoad: startAfterLoad, forceFloat32: forceFloat32, acceptedReplacements: [])
    }

    /// Resume the load the user reviewed, replacing the listed saved settings
    /// with the current ones.
    func acceptSessionSettingsReview(_ review: SessionSettingsReview) {
        sessionSettingsReview = nil
        let ids = Set(review.findings.map(\.id))
        SessionLogger.shared.log(
            "[RESUME] user accepted replacements for \(ids.sorted().joined(separator: ", ")) in \(review.sessionURL.lastPathComponent)"
        )
        loadSessionFrom(
            url: review.sessionURL,
            startAfterLoad: review.startAfterLoad,
            forceFloat32: review.forceFloat32,
            acceptedReplacements: ids
        )
    }

    /// The user chose not to resume a session with unusable saved settings.
    func declineSessionSettingsReview(_ review: SessionSettingsReview) {
        sessionSettingsReview = nil
        SessionLogger.shared.log("[RESUME] user declined to resume \(review.sessionURL.lastPathComponent) (unusable saved settings)")
        checkpoint?.setCheckpointStatus("Not resumed: \(review.sessionURL.lastPathComponent) has unusable saved settings", kind: .error)
    }

    /// `acceptedReplacements`: ids of saved settings the user has already
    /// agreed to replace (`acceptSessionSettingsReview`). Any other unusable
    /// saved setting stops the load for review before anything is built.
    private func loadSessionFrom(
        url: URL,
        startAfterLoad: Bool,
        forceFloat32: Bool,
        acceptedReplacements: Set<String>
    ) {
        // In-function guards (belt-and-suspenders with menu disable).
        if isBuildingOrBusyProvider() {
            onRefuseMenuAction(busyReasonProvider())
            return
        }

        checkpoint?.checkpointSaveInFlight = true
        checkpoint?.setCheckpointStatus("Loading session \(url.lastPathComponent)…", kind: .progress)

        Task {
            // 1. Decode the session first (CPU only, no graph) so we know the
            //    architecture it was saved with before building anything.
            let loadResult: Result<LoadedSession, Error> = await Task.detached(priority: .userInitiated) {
                let scopeAccessed = url.startAccessingSecurityScopedResource()
                defer {
                    if scopeAccessed {
                        url.stopAccessingSecurityScopedResource()
                    }
                }
                do {
                    return .success(try CheckpointManager.loadSession(at: url))
                } catch {
                    return .failure(error)
                }
            }.value
            guard case .success(let loaded) = loadResult else {
                checkpoint?.checkpointSaveInFlight = false
                if case .failure(let error) = loadResult {
                    checkpoint?.setCheckpointStatus("Load failed: \(error.localizedDescription)", kind: .error)
                    SessionLogger.shared.log("[CHECKPOINT] Load session decode failed: \(error.localizedDescription)")
                }
                if startAfterLoad { onResumeFinished() }
                return
            }

            // 1a. A session another path wrote (train-vs-UCI) has no self-play
            //     or arena run to continue here; refuse it by name.
            if let refusal = loaded.state.guiLoadRefusal {
                checkpoint?.checkpointSaveInFlight = false
                checkpoint?.setCheckpointStatus("Load refused: \(refusal)", kind: .error)
                SessionLogger.shared.log("[CHECKPOINT] Load session refused (\(url.lastPathComponent)): \(refusal)")
                if startAfterLoad { onResumeFinished() }
                return
            }

            // 1b. Saved settings that cannot be used as found stop the load
            //     until the user reviews them; nothing is replaced silently.
            let findings = loaded.state.invalidSavedSettings(current: TrainingParameters.shared.snapshot())
            let unaccepted = findings.filter { !acceptedReplacements.contains($0.id) }
            if !unaccepted.isEmpty {
                checkpoint?.checkpointSaveInFlight = false
                for finding in unaccepted {
                    SessionLogger.shared.log(
                        "[RESUME] ERROR \(url.lastPathComponent): saved \(finding.id) = \(finding.found) cannot be used "
                            + "(\(finding.problem)); offered replacement: \(finding.replacement)"
                    )
                }
                checkpoint?.setCheckpointStatus(
                    "Session \(url.lastPathComponent) has \(unaccepted.count) unusable saved setting(s) — review to resume",
                    kind: .error
                )
                sessionSettingsReview = SessionSettingsReview(
                    sessionURL: url,
                    startAfterLoad: startAfterLoad,
                    forceFloat32: forceFloat32,
                    findings: findings
                )
                if startAfterLoad { onResumeFinished() }
                return
            }

            // 2. Build (or rebuild) the champion at the SESSION's architecture —
            //    not the current build default — so non-default / historical
            //    sessions get a matching graph. The random init only satisfies
            //    graph compilation; the weights are overwritten next.
            var championArch = loaded.championFile.architecture
            if forceFloat32 && championArch.computeDataType != .float32 {
                SessionLogger.shared.log(
                    "[RESUME] forceFloat32: overriding saved computeDataType "
                    + "\(championArch.computeDataType.rawValue) -> float32 (policy tail "
                    + "\(championArch.policyTailPrecision.rawValue) -> \(PolicyTailPrecisionSetting.doesNotApply.rawValue)) "
                    + "for champion + trainer + all inference nets"
                )
                do {
                    championArch = try championArch.withComputeDataType(.float32, tail: nil)
                } catch {
                    checkpoint?.checkpointSaveInFlight = false
                    checkpoint?.setCheckpointStatus("Load failed: \(error.localizedDescription)", kind: .error)
                    SessionLogger.shared.log("[CHECKPOINT] Load session forceFloat32 failed: \(error.localizedDescription)")
                    if startAfterLoad { onResumeFinished() }
                    return
                }
            }
            let championResult = await self.ensureChampionBuilt(arch: championArch)
            guard case .success(let champion) = championResult else {
                checkpoint?.checkpointSaveInFlight = false
                if case .failure(let error) = championResult {
                    checkpoint?.setCheckpointStatus("Build failed: \(error.localizedDescription)", kind: .error)
                    SessionLogger.shared.log("[CHECKPOINT] Load session auto-build failed: \(error.localizedDescription)")
                }
                if startAfterLoad { onResumeFinished() }
                return
            }

            // 3. Apply champion weights; trainer weights are held for the next
            //    startRealTraining via `pendingLoadedSession`.
            // `adoptLoadedChampionOrigin` closes this; a failed load leaves it
            // open, since the champion's weights are then not known to match
            // its recorded origin.
            noteChampionWeightsReplaced()
            let applyResult: Result<Void, Error> = await Task.detached(priority: .userInitiated) {
                do {
                    try await champion.loadWeights(loaded.championFile.weights)
                    return .success(())
                } catch {
                    return .failure(error)
                }
            }.value
            checkpoint?.checkpointSaveInFlight = false
            switch applyResult {
            case .success:
                champion.identifier = ModelID(value: loaded.championFile.modelID)
                // A branch from this champion records its file as the parent.
                adoptLoadedChampionOrigin(loaded.championFile)
                pendingLoadedSession = loaded
                pendingLoadedSessionAcceptedReplacements = Set(findings.map(\.id))
                networkStatus = """
                    Loaded session \(loaded.state.sessionID)
                    Champion: \(loaded.championFile.modelID)
                    Trainer: \(loaded.trainerFile.modelID)
                    Steps: \(loaded.state.trainingSteps) / Games: \(loaded.state.selfPlayGames)
                    Click Play and Train to resume.
                    """
                checkpoint?.lastSavedAt = nil
                checkpoint?.lastResumedAt = Date()
                checkpoint?.setCheckpointStatus("Loaded session \(loaded.state.sessionID) — click Play and Train to resume", kind: .success)
                let savedBuild = loaded.state.buildNumber.map(String.init) ?? "?"
                let savedGit = loaded.state.buildGitHash ?? "?"
                let bufStr: String
                if let stored = loaded.state.replayBufferStoredCount,
                   let cap = loaded.state.replayBufferCapacity {
                    bufStr = " replay=\(stored)/\(cap)"
                } else {
                    bufStr = " replay=none"
                }
                SessionLogger.shared.log("[CHECKPOINT] Loaded session: \(url.lastPathComponent) savedBuild=\(savedBuild) savedGit=\(savedGit)\(bufStr)")
                SessionLogger.shared.logArchitecture(
                    event: "resumed session \(loaded.state.sessionID) (champion \(loaded.championFile.modelID))",
                    arch: loaded.championFile.architecture
                )
                onClearInferenceResult()
                if startAfterLoad {
                    // Auto-resume path (from the launch-time sheet or
                    // the File menu "Resume training from autosave"
                    // command). Chain straight into Play-and-Train so
                    // the user's single click results in the session
                    // both loaded AND running.
                    SessionLogger.shared.log("[CHECKPOINT] Auto-resume: starting Play-and-Train on loaded session")
                    startRealTraining()
                }
            case .failure(let error):
                checkpoint?.setCheckpointStatus("Load failed: \(error.localizedDescription)", kind: .error)
                SessionLogger.shared.log("[CHECKPOINT] Load session failed: \(error.localizedDescription)")
            }
            if startAfterLoad {
                onResumeFinished()
            }
        }
    }

    // startRealTraining(mode:) / stopRealTraining() moved to SessionController+Training.swift.

    // runArenaParallel(...) / logArenaResult(...) / cleanupArenaState(...) + the
    // arena statics moved to SessionController+Arena.swift.

    // MARK: - Session-state snapshot (Stage 4m)

    nonisolated static let trainerLearningRateDefault: Float = 1e-3
    nonisolated static let entropyRegularizationCoeffDefault: Float = 1e-3

    /// Where the arena clock stands for a session save.
    enum ArenaClockAtSave {
        /// The trigger box's clock as it reads now. Saves outside an arena:
        /// manual, periodic and SIGUSR2 saves are refused while one runs.
        case live
        /// The arena that triggered this save has just finished. The
        /// post-promotion save is written inside the arena, before the
        /// trigger box records that it ended, so the box still reads the time
        /// since the arena before it.
        case arenaJustFinished
    }

    /// The replay-buffer fields of a `[CHECKPOINT] Saved session` line:
    /// `buffer=included` with the buffer's fill when the save wrote it
    /// (`writtenBuffer`), `buffer=omitted` when it did not (D-8).
    nonisolated static func savedReplayBufferLogFields(writtenBuffer: ReplayBuffer?) -> String {
        guard let snap = writtenBuffer?.stateSnapshot() else { return " buffer=omitted" }
        return " buffer=included replay=\(snap.storedCount)/\(snap.capacity)"
    }

    /// Why a save or a promotion could not take its cut of the run.
    enum RunCutError: LocalizedError {
        case noModelID
        case noLiveCounters

        var errorDescription: String? {
            switch self {
            case .noModelID:
                return "the champion or the trainer has no model ID"
            case .noLiveCounters:
                return "the run has no training stats box or no self-play stats box"
            }
        }
    }

    /// Publish one training-stats-box snapshot as the run's displayed
    /// training counters. Every field comes from the same snapshot, so the
    /// step count, the last step's timing and the rolling losses always
    /// describe one instant, whichever path publishes them (the heartbeat,
    /// an arena start, a save's cut).
    func publishTrainingStats(_ snapshot: TrainingLiveStatsBox.Snapshot) {
        trainingStats = snapshot.stats
        lastTrainStep = snapshot.lastTiming
        realRollingPolicyLoss = snapshot.rollingPolicyLoss
        realRollingValueLoss = snapshot.rollingValueLoss
    }

    /// Publish the run's counters — `trainingStats` and `parallelStats` —
    /// as the live boxes hold them now, and return the run's step count.
    ///
    /// Those two properties are the heartbeat's mirror of the boxes, so
    /// between ticks they trail the workers. A save builds session.json
    /// from them (`buildCurrentSessionState`, and the training-segment
    /// close it performs), and takes the trainer file's training step from
    /// them; read as the heartbeat left them, those counts described an
    /// earlier instant than the trainer state, the lineage record and the
    /// replay buffer the same save took under its pauses, and a resume
    /// restored the run's counters from that earlier instant while its
    /// trainer clock and fed totals came from the cut. Called with
    /// self-play and training both paused, this makes every count the save
    /// writes describe the cut.
    ///
    /// Both boxes exist for the whole of a Play-and-Train run, which is the
    /// only time anything saves or promotes, so a missing one is an error
    /// rather than a count to leave as it was.
    func publishRunCountersAtCut() throws -> Int {
        guard let trainingBox, let parallelWorkerStatsBox else {
            throw RunCutError.noLiveCounters
        }
        let training = trainingBox.snapshot()
        publishTrainingStats(training)
        parallelStats = parallelWorkerStatsBox.snapshot()
        return training.stats.steps
    }

    /// Build the Codable snapshot of the current session state (counters,
    /// hyperparameters, arena history, replay-buffer footprint, build info).
    /// Called at save time, on the main actor, by both the manual/periodic
    /// save path and `runArenaParallel`'s post-promotion save — each with
    /// self-play and training paused and the run's counters just published
    /// (`publishRunCountersAtCut`), so the snapshot describes the save's
    /// cut. Closes the active training segment at save time (and re-opens a
    /// fresh one if training is still in progress) so the on-disk cumulative
    /// wall-time totals stay correct across mid-training saves.
    /// `arenaClock` says which arena clock the save records;
    /// `includeReplayBuffer` whether the save writes the replay buffer, which
    /// `hasReplayBuffer` and the buffer counters then describe.
    ///
    /// `batchSize` and `replayBufferMinPositionsBeforeTraining` come from the
    /// run's start-time capture (`RunStartParameterCapture`), the values the
    /// run trains under; `trainingPositionsSeen` from its positions count
    /// (`RunTrainedPositions`), each step at the batch it trained at, nil
    /// when steps before the run are unrecorded. A save describes a run, so
    /// a missing capture or count throws, before any side effect. The other
    /// settings are recorded from `TrainingParameters` as before: a GUI
    /// resume restores its settings from them.
    @MainActor
    func buildCurrentSessionState(
        championID: String,
        trainerID: String,
        arenaClock: ArenaClockAtSave,
        includeReplayBuffer: Bool
    ) throws -> SessionCheckpointState {
        let params = TrainingParameters.shared
        let runCapture = try requiredRunStartCapture(for: "the session state of this save")
        let trainedPositions = try trainedPositionsForSessionState(atSessionSteps: trainingStats?.steps ?? 0)
        let wasTraining = realTraining
        checkpoint?.closeActiveTrainingSegment(reason: "save")
        // A suspended run (health alarm or divergence) closed its segment
        // when it parked, so the parked idle is not training wall time; a
        // save then must not reopen it.
        if wasTraining && trainingSuspension == nil && checkpoint?.activeSegmentStart == nil {
            checkpoint?.beginActiveTrainingSegment()
        }
        let now = Date()
        let secondsSinceLastArena: Double?
        switch arenaClock {
        case .live:
            secondsSinceLastArena = arenaTriggerBox?.secondsSinceLastArena(now: now)
        case .arenaJustFinished:
            secondsSinceLastArena = 0
        }
        let sessionStart = checkpoint?.currentSessionStart ?? (parallelStats?.sessionStart ?? now)
        let elapsedSec = max(0, now.timeIntervalSince(sessionStart))
        let snap = parallelStats
        let trainingSnap = trainingStats
        let history = tournamentHistory.map { record in
            ArenaHistoryEntryCodable(
                finishedAtStep: record.finishedAtStep,
                candidateWins: record.candidateWins,
                championWins: record.championWins,
                draws: record.draws,
                score: record.score,
                promoted: record.promoted,
                promotedID: record.promotedID?.description,
                durationSec: record.durationSec,
                gamesPlayed: record.gamesPlayed,
                promotionKind: record.promotionKind?.rawValue,
                candidateWinsAsWhite: record.candidateWinsAsWhite,
                candidateWinsAsBlack: record.candidateWinsAsBlack,
                candidateLossesAsWhite: record.candidateLossesAsWhite,
                candidateLossesAsBlack: record.candidateLossesAsBlack,
                candidateDrawsAsWhite: record.candidateDrawsAsWhite,
                candidateDrawsAsBlack: record.candidateDrawsAsBlack,
                finishedAtUnix: record.finishedAt.map { Int64($0.timeIntervalSince1970) },
                candidateID: record.candidateID?.description,
                championID: record.championID?.description,
                extendedSummary: record.extendedSummary,
                promotionCriterion: record.promotionCriterion?.logToken,
                sprt: record.sprtVerdict.map(ArenaSPRTVerdictCodable.init),
                sprtGamesFinishedAtDecision: record.sprtGamesFinishedAtDecision
            )
        }
        let lr = trainer?.learningRate ?? Self.trainerLearningRateDefault
        let entropyCoeff = trainer?.entropyRegularizationCoeff ?? Self.entropyRegularizationCoeffDefault
        let drawPen = trainer?.drawPenalty ?? Float(params.drawPenalty)
        let bufferSnap = includeReplayBuffer ? replayBuffer?.stateSnapshot() : nil
        // Architecture metadata must reflect the ACTUAL built network, not the
        // ChessNetwork static defaults (which only describe the current preset).
        // Without this, a non-default session (e.g. a rebuilt v3 8-block) saved
        // the wrong arch and the resume prompt showed v4/5-block.
        let resolvedArch = network?.network.arch ?? trainer?.arch ?? .current
        let segments: [SessionCheckpointState.TrainingSegment]? =
            (checkpoint?.completedTrainingSegments.isEmpty ?? true)
            ? nil
            : checkpoint?.completedTrainingSegments
        return SessionCheckpointState(
            formatVersion: SessionCheckpointState.currentFormatVersion,
            sessionID: checkpoint?.currentSessionID ?? "unknown-session",
            savedAtUnix: Int64(now.timeIntervalSince1970),
            sessionStartUnix: Int64(sessionStart.timeIntervalSince1970),
            elapsedTrainingSec: elapsedSec,
            trainingSteps: trainingSnap?.steps ?? 0,
            selfPlayGames: snap?.selfPlayGames ?? 0,
            selfPlayMoves: snap?.selfPlayPositions ?? 0,
            trainingPositionsSeen: trainedPositions,
            batchSize: runCapture.trainingBatchSize,
            learningRate: lr,
            entropyRegularizationCoeff: entropyCoeff,
            drawPenalty: drawPen,
            promoteThreshold: params.arenaPromoteThreshold,
            arenaGames: params.arenaGamesPerTournament,
            arenaConcurrency: params.arenaConcurrency,
            selfPlayTau: TauConfigCodable(samplingScheduleBox?.selfPlay ?? buildSelfPlaySchedule()),
            arenaTau: TauConfigCodable(samplingScheduleBox?.arena ?? buildArenaSchedule()),
            selfPlayWorkerCount: params.selfPlayConcurrency,
            gradClipMaxNorm: Float(params.gradClipMaxNorm),
            weightDecayCoeff: Float(params.weightDecay),
            dropoutRate: trainer?.dropoutRate ?? Float(params.dropoutRate),
            policyLossWeight: Float(params.policyLossWeight),
            valueLossWeight: Float(params.valueLossWeight),
            momentumCoeff: Float(params.momentumCoeff),
            illegalMassPenaltyWeight: Float(params.illegalMassWeight),
            policyLabelSmoothingEpsilon: Float(params.policyLabelSmoothingEpsilon),
            policyLabelSmoothingMode: params.policyLabelSmoothingMode.logToken,
            policyLabelSmoothingPerMove: Float(params.policyLabelSmoothingPerMove),
            policyLabelSmoothingPerMoveCap: Float(params.policyLabelSmoothingPerMoveCap),
            valueLabelSmoothingEpsilon: Float(params.valueLabelSmoothingEpsilon),
            replayRatioTarget: params.replayRatioTarget,
            replayRatioAutoAdjust: params.replayRatioAutoAdjust,
            stepDelayMs: params.trainingStepDelayMs,
            selfPlayDelayMs: params.selfPlayDelayMs,
            lastAutoComputedDelayMs: try savedAutoComputedDelayMs(),
            // Schema-expansion fields (close the autotrain reproducibility gap
            // — these previously lived only in @AppStorage / @State and so
            // silently picked up the user's current preference on resume rather
            // than the session's saved value). All Optional for back-compat.
            lrWarmupSteps: params.lrWarmupSteps,
            sqrtBatchScalingForLR: params.sqrtBatchScalingLR,
            signedAdvantageComplementCE: params.signedAdvantageComplementCE,
            replayBufferMinPositionsBeforeTraining: runCapture.replayBufferMinPositionsBeforeTraining,
            arenaAutoIntervalSec: params.arenaAutoIntervalSec,
            candidateProbeIntervalSec: params.candidateProbeIntervalSec,
            legalMassCollapseThreshold: params.legalMassCollapseThreshold,
            legalMassCollapseGraceSeconds: params.legalMassCollapseGraceSeconds,
            legalMassCollapseNoImprovementProbes: params.legalMassCollapseNoImprovementProbes,
            batchStatsInterval: params.batchStatsInterval,
            klProbeInterval: params.klProbeInterval,
            relativeGradClipMode: params.relativeGradClipMode,
            relativeGradClipMultiple: params.relativeGradClipMultiple,
            relativeGradClipWindowSteps: params.relativeGradClipWindowSteps,
            relativeGradClipMinHistorySteps: params.relativeGradClipMinHistorySteps,
            relativeGradClipFloor: params.relativeGradClipFloor,
            stepLineIntervalSec: params.stepLineIntervalSec,
            periodicAutosaveIntervalSec: params.periodicAutosaveIntervalSec,
            maxPeriodicAutosavesKept: params.maxPeriodicAutosavesKept,
            automaticSavePruningEnabled: params.automaticSavePruningEnabled,
            sessionSaveIncludeReplayBuffer: params.sessionSaveIncludeReplayBuffer,
            arenaPromotionCriterion: params.arenaPromotionCriterion.logToken,
            arenaSPRTElo0: params.arenaSPRTElo0,
            arenaSPRTElo1: params.arenaSPRTElo1,
            arenaSPRTAlpha: params.arenaSPRTAlpha,
            arenaSPRTBeta: params.arenaSPRTBeta,
            arenaSPRTMinGames: params.arenaSPRTMinGames,
            arenaSPRTMaxGames: params.arenaSPRTMaxGames,
            recordingCorpusID: activeRecordingCorpusID,
            recordSelfPlayGames: params.recordSelfPlayGames,
            lrMomentumCycle: params.lrMomentumCycle,
            lrMomentumCycleEnvelope: params.lrMomentumCycleEnvelope,
            maxPliesFromAnyOneGame: params.maxPliesFromAnyOneGame,
            targetSampledGameLengthPlies: params.targetSampledGameLengthPlies,
            maxDrawPercentPerBatch: params.maxDrawPercentPerBatch,
            replayBufferStratifyByMaterial: params.replayBufferStratifyByMaterial,
            selfPlayDrawKeepFraction: params.selfPlayDrawKeepFraction,
            maxPliesPerGame: params.selfPlayMaxPliesPerGame,
            drawWatchPDrawThreshold: params.drawWatchPDrawThreshold,
            drawWatchTerminateGames: params.drawWatchTerminateGames,
            drawWatchStreakLength: params.drawWatchStreakLength,
            emittedGames: snap?.emittedGames,
            emittedPositions: snap?.emittedPositions,
            whiteCheckmates: snap?.whiteCheckmates,
            blackCheckmates: snap?.blackCheckmates,
            stalemates: snap?.stalemates,
            fiftyMoveDraws: snap?.fiftyMoveDraws,
            threefoldRepetitionDraws: snap?.threefoldRepetitionDraws,
            insufficientMaterialDraws: snap?.insufficientMaterialDraws,
            maxPliesDropped: snap?.maxPliesDropped,
            totalGameWallMs: snap?.totalGameWallMs,
            emittedWhiteCheckmates: snap?.emittedWhiteCheckmates,
            emittedBlackCheckmates: snap?.emittedBlackCheckmates,
            emittedStalemates: snap?.emittedStalemates,
            emittedFiftyMoveDraws: snap?.emittedFiftyMoveDraws,
            emittedThreefoldRepetitionDraws: snap?.emittedThreefoldRepetitionDraws,
            emittedInsufficientMaterialDraws: snap?.emittedInsufficientMaterialDraws,
            buildNumber: BuildInfo.buildNumber,
            buildGitHash: BuildInfo.gitHash,
            buildGitBranch: BuildInfo.gitBranch,
            buildDate: BuildInfo.buildDate,
            buildTimestamp: BuildInfo.buildTimestamp,
            buildGitDirty: BuildInfo.gitDirty,
            hasReplayBuffer: bufferSnap != nil,
            replayBufferStoredCount: bufferSnap?.storedCount,
            replayBufferCapacity: bufferSnap?.capacity,
            replayBufferTotalPositionsAdded: bufferSnap?.totalPositionsAdded,
            championID: championID,
            trainerID: trainerID,
            arenaHistory: history
        )
        .withTrainingHealthSettings(
            enabled: params.trainingHealthAlarmsEnabled,
            checkIntervalSteps: params.trainingHealthCheckIntervalSteps,
            learningGraceSteps: params.trainingHealthLearningGraceSteps,
            actions: TrainingHealthActions { params[keyPath: TrainingParameters.trainingHealthActionKeyPath(for: $0)] })
        .withTrainingSegments(segments)
        .withArchitecture(ArchitectureMetadata(describing: resolvedArch))
        .withProbeHistories(
            lichess: lichessProbeHistory.makeSnapshot(),
            wideLichess: lichessProbeWideHistory.makeSnapshot(),
            tactical: tacticalProbeHistory.makeSnapshot()
        )
        .withArenaClock(secondsSinceLastArena: secondsSinceLastArena)
        .withRunObservability(diversityWindow: selfPlayDiversityTracker?.windowSequences(),
                              alarmStreaks: trainingAlarm?.streaks)
        .withLegalMassCollapseDetector(legalMassCollapseDetector?.snapshot(now: Date()))
    }

    // MARK: - Session resume (Stage 4k)

    /// Resume helper — reads `training_chart.json` / `progress_rate_chart.json`
    /// from the previously-loaded session and seeds `chartCoordinator` with the
    /// restored trajectory (plus the inline `arenaChartEvents` / `legalMassMaxAllTime`
    /// from `pendingLoadedSession.state`). Decode failures log and skip so a
    /// corrupt chart file never blocks the rest of the session-resume flow.
    /// Honors "View > Collect Chart Data" being off.
    func seedChartCoordinatorFromLoadedSession(
        chartURLs: (training: URL, progressRate: URL)
    ) {
        guard let chartCoordinator, chartCoordinator.collectionEnabled else {
            SessionLogger.shared.log(
                "[CHECKPOINT] Skipping chart-data restore — collection is disabled in View > Collect Chart Data"
            )
            return
        }
        let trainingSamples: [TrainingChartSample]
        let progressSamples: [ProgressRateSample]
        do {
            trainingSamples = try readChartFile(
                [TrainingChartSample].self, from: chartURLs.training
            )
            progressSamples = try readChartFile(
                [ProgressRateSample].self, from: chartURLs.progressRate
            )
        } catch {
            SessionLogger.shared.log(
                "[CHECKPOINT] Chart-data restore skipped — decode failed: \(error.localizedDescription)"
            )
            return
        }
        let arenaEvents = pendingLoadedSession?.state.arenaChartEvents ?? []
        let legalMassMax = pendingLoadedSession?.state.legalMassMaxAllTime ?? 0
        let lastTrainElapsed = trainingSamples.last?.elapsedSec ?? 0
        let lastProgressElapsed = progressSamples.last?.elapsedSec ?? 0
        let lastElapsed = max(lastTrainElapsed, lastProgressElapsed)
        let snapshot = ChartCoordinatorSnapshot(
            trainingSamples: trainingSamples,
            progressRateSamples: progressSamples,
            arenaChartEvents: arenaEvents,
            legalMassMaxAllTime: legalMassMax,
            lastElapsedSec: lastElapsed
        )
        chartCoordinator.seedFromRestoredSession(snapshot)
        SessionLogger.shared.log(
            "[CHECKPOINT] Restored chart data: \(trainingSamples.count) training samples, \(progressSamples.count) progress-rate samples, \(arenaEvents.count) arena events"
        )
    }

}
