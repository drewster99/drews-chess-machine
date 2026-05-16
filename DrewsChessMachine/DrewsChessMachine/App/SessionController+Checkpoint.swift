import SwiftUI

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
        let championID = champion.identifier?.description ?? "unknown"
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
            gate?.resume()

            if let exportError {
                checkpoint?.cancelSlowSaveWatchdog()
                checkpoint?.checkpointSaveInFlight = false
                checkpoint?.setCheckpointStatus("Save failed (export): \(exportError.localizedDescription)", kind: .error)
                SessionLogger.shared.log("[CHECKPOINT] Save champion export failed: \(exportError.localizedDescription)")
                return
            }

            let metadata = ModelCheckpointMetadata(
                creator: "manual",
                trainingStep: trainingStats?.steps,
                parentModelID: "",
                notes: "Manual Save Champion export"
            )
            let createdAtUnix = Int64(Date().timeIntervalSince1970)

            let outcome: Result<URL, Error> = await Task.detached(priority: .userInitiated) {
                do {
                    let url = try await CheckpointManager.saveModel(
                        weights: championWeights,
                        modelID: championID,
                        createdAtUnix: createdAtUnix,
                        metadata: metadata,
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
    /// in-flight self-play game or training step, so 15 s is a
    /// comfortable margin above the worst-case game length at
    /// typical self-play rates. On timeout the save bails with
    /// a user-visible error rather than blocking forever.
    nonisolated static let saveGateTimeoutMs: Int = 15_000

    /// Manual "Save Session" — writes a full `.dcmsession` with
    /// champion and trainer model files plus `session.json`.
    /// Requires an active Play-and-Train session and an available
    /// trainer. Briefly pauses both self-play worker 0 and the
    /// training gate to snapshot the two networks' weights.
    func handleSaveSessionManual() {
        // Belt-and-suspenders guards — menu disable is the primary
        // gate but these cover keyboard-shortcut / URL-scheme
        // invocations under a race.
        if checkpoint?.checkpointSaveInFlight == true {
            onRefuseMenuAction("A save is already in progress. Wait for it to finish.")
            return
        }
        if isArenaRunning {
            onRefuseMenuAction("Can't save the session while the arena is running. Wait for it to finish.")
            return
        }
        guard realTraining,
              let champion = network,
              let trainer,
              let selfPlayGate = activeSelfPlayGate,
              let trainingGate = activeTrainingGate else {
            onRefuseMenuAction("No active training session to save. Start Play and Train first.")
            return
        }
        saveSessionInternal(
            champion: champion,
            trainer: trainer,
            selfPlayGate: selfPlayGate,
            trainingGate: trainingGate,
            trigger: .manual
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
            trigger: .periodic
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
    /// file can reach it.)
    func saveSessionInternal(
        champion: ChessMPSNetwork,
        trainer: ChessTrainer,
        selfPlayGate: WorkerPauseGate,
        trainingGate: WorkerPauseGate,
        trigger: SessionSaveTrigger
    ) {
        let championID = champion.identifier?.description ?? "unknown"
        let trainerID = trainer.identifier?.description ?? "unknown"
        let diskTag = trigger.diskTag
        let uiSuffix = trigger.uiSuffix
        checkpoint?.checkpointSaveInFlight = true
        checkpoint?.setCheckpointStatus("Saving session\(uiSuffix)…", kind: .progress)
        checkpoint?.startSlowSaveWatchdog(label: "session save\(uiSuffix)")

        // Build the state snapshot on the main actor before
        // jumping to detached work. Capture the replay buffer handle
        // here too so the detached write path can serialize it
        // alongside the two network files — `ReplayBuffer` is
        // `@unchecked Sendable` and serializes access via its own
        // lock, so the buffer can be written from a background task
        // while self-play workers (which only append) are paused.
        let sessionState = buildCurrentSessionState(
            championID: championID,
            trainerID: trainerID
        )
        let trainingStep = trainingStats?.steps ?? 0
        let bufferForSave = replayBuffer
        // Snapshot the chart-coordinator state on the main actor
        // BEFORE jumping to detached work — the rings are
        // `@MainActor`-isolated so the array copies have to happen
        // here. `buildSnapshot()` returns nil when collection is off
        // or both rings are empty, in which case the save path skips
        // writing chart-companion files entirely (matching the
        // existing `bufferForSave == nil` skip).
        let chartSnapshotForSave = chartCoordinator?.buildSnapshot()

        Task {
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

            // Pause self-play briefly so the champion export is
            // race-free, snapshot weights, then resume. Uses the
            // bounded variant so a session end mid-save doesn't
            // spin forever waiting for workers that have exited.
            let selfPlayAcquired = await selfPlayGate.pauseAndWait(timeoutMs: Self.saveGateTimeoutMs)
            guard selfPlayAcquired else {
                clearInFlight()
                checkpoint?.setCheckpointStatus("Save aborted: could not pause self-play (timeout)", kind: .error)
                SessionLogger.shared.log("[CHECKPOINT] Save session aborted at self-play pause timeout")
                return
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
            selfPlayGate.resume()

            if let championError {
                clearInFlight()
                checkpoint?.setCheckpointStatus("Save failed (champion export): \(championError.localizedDescription)", kind: .error)
                SessionLogger.shared.log("[CHECKPOINT] Save session failed at champion export: \(championError.localizedDescription)")
                return
            }

            // Pause training briefly to snapshot trainer weights.
            let trainingAcquired = await trainingGate.pauseAndWait(timeoutMs: Self.saveGateTimeoutMs)
            guard trainingAcquired else {
                clearInFlight()
                checkpoint?.setCheckpointStatus("Save aborted: could not pause training (timeout)", kind: .error)
                SessionLogger.shared.log("[CHECKPOINT] Save session aborted at training pause timeout")
                return
            }
            var trainerWeights: [[Float]] = []
            var trainerError: Error?
            do {
                // exportTrainerWeights bundles trainables + bn +
                // momentum velocity. Caller is responsible for
                // pausing both gates, which we did above.
                trainerWeights = try await Task.detached(priority: .userInitiated) {
                    try await trainer.exportTrainerWeights()
                }.value
            } catch {
                trainerError = error
            }
            trainingGate.resume()

            if let trainerError {
                clearInFlight()
                checkpoint?.setCheckpointStatus("Save failed (trainer export): \(trainerError.localizedDescription)", kind: .error)
                SessionLogger.shared.log("[CHECKPOINT] Save session failed at trainer export: \(trainerError.localizedDescription)")
                return
            }

            // Final write + verify on a detached task so UI stays
            // responsive during the ~150 ms scratch-network build.
            let championMetadata = ModelCheckpointMetadata(
                creator: diskTag,
                trainingStep: trainingStep,
                parentModelID: "",
                notes: "Session checkpoint (\(diskTag))"
            )
            let trainerMetadata = ModelCheckpointMetadata(
                creator: diskTag,
                trainingStep: trainingStep,
                parentModelID: championID,
                notes: "Trainer lineage at session checkpoint (\(diskTag))"
            )
            let now = Int64(Date().timeIntervalSince1970)
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
                        replayBuffer: bufferForSave,
                        chartSnapshot: chartSnapshotForSave,
                        trigger: diskTag
                    )
                    return .success(url)
                } catch {
                    return .failure(error)
                }
            }.value

            clearInFlight()
            switch outcome {
            case .success(let url):
                checkpoint?.setCheckpointStatus("Saved \(url.lastPathComponent)\(uiSuffix)", kind: .success)
                let bufStr: String
                if let snap = bufferForSave?.stateSnapshot() {
                    bufStr = " replay=\(snap.storedCount)/\(snap.capacity)"
                } else {
                    bufStr = ""
                }
                SessionLogger.shared.log(
                    "[CHECKPOINT] Saved session (\(diskTag)): \(url.lastPathComponent) build=\(BuildInfo.buildNumber) git=\(BuildInfo.gitHash)\(bufStr)"
                )
                checkpoint?.recordLastSessionPointer(
                    directoryURL: url,
                    sessionID: sessionState.sessionID,
                    trigger: diskTag
                )
                periodicSaveController?.noteSuccessfulSave(at: Date())
                checkpoint?.lastSavedAt = Date()
                checkpoint?.lastResumedAt = nil
            case .failure(let error):
                checkpoint?.setCheckpointStatus("Save failed: \(error.localizedDescription)", kind: .error)
                SessionLogger.shared.log("[CHECKPOINT] Save session (\(diskTag)) failed: \(error.localizedDescription)")
            }
        }
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

        checkpoint?.checkpointSaveInFlight = true
        checkpoint?.setCheckpointStatus("Loading \(url.lastPathComponent)…", kind: .progress)

        Task {
            // Auto-build the champion shell if it doesn't exist yet.
            // The weights are about to be overwritten, so the random
            // init is only satisfying graph compilation — no reason
            // to require the user to press Build first.
            let championResult = await self.ensureChampionBuilt()
            switch championResult {
            case .failure(let error):
                checkpoint?.checkpointSaveInFlight = false
                checkpoint?.setCheckpointStatus("Build failed: \(error.localizedDescription)", kind: .error)
                SessionLogger.shared.log("[CHECKPOINT] Load model auto-build failed: \(error.localizedDescription)")
                return
            case .success(let champion):
                // Keep the security scope open across the entire
                // detached read+load so files picked from outside the
                // sandbox (Downloads, AirDrop, external volumes) stay
                // accessible until the work finishes. Start/stop must
                // happen inside the detached closure to bracket the
                // actual I/O.
                let outcome: Result<ModelCheckpointFile, Error> = await Task.detached(priority: .userInitiated) {
                    let scopeAccessed = url.startAccessingSecurityScopedResource()
                    defer {
                        if scopeAccessed {
                            url.stopAccessingSecurityScopedResource()
                        }
                    }
                    do {
                        let file = try CheckpointManager.loadModelFile(at: url)
                        try await champion.loadWeights(file.weights)
                        return .success(file)
                    } catch {
                        return .failure(error)
                    }
                }.value
                checkpoint?.checkpointSaveInFlight = false
                switch outcome {
                case .success(let file):
                    champion.identifier = ModelID(value: file.modelID)
                    networkStatus = "Loaded model \(file.modelID)\nFrom: \(url.lastPathComponent)"
                    checkpoint?.setCheckpointStatus("Loaded \(file.modelID)", kind: .success)
                    SessionLogger.shared.log("[CHECKPOINT] Loaded model: \(url.lastPathComponent) → \(file.modelID)")
                    onClearInferenceResult()
                    // Flag champion-replaced for the post-Stop Start
                    // dialog's "Continue" annotation. Cleared as
                    // soon as a new training segment starts.
                    if replayBuffer != nil {
                        championLoadedSinceLastTrainingSegment = true
                    }
                case .failure(let error):
                    checkpoint?.setCheckpointStatus("Load failed: \(error.localizedDescription)", kind: .error)
                    SessionLogger.shared.log("[CHECKPOINT] Load model failed: \(error.localizedDescription)")
                }
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

    func loadSessionFrom(url: URL, startAfterLoad: Bool = false) {
        // In-function guards (belt-and-suspenders with menu disable).
        if isBuildingOrBusyProvider() {
            onRefuseMenuAction(busyReasonProvider())
            return
        }

        checkpoint?.checkpointSaveInFlight = true
        checkpoint?.setCheckpointStatus("Loading session \(url.lastPathComponent)…", kind: .progress)

        Task {
            // Auto-build the champion shell if it doesn't exist yet.
            // The weights are about to be overwritten, so the random
            // init is only satisfying graph compilation — no reason
            // to require the user to press Build first.
            let championResult = await self.ensureChampionBuilt()
            guard case .success(let champion) = championResult else {
                checkpoint?.checkpointSaveInFlight = false
                if case .failure(let error) = championResult {
                    checkpoint?.setCheckpointStatus("Build failed: \(error.localizedDescription)", kind: .error)
                    SessionLogger.shared.log("[CHECKPOINT] Load session auto-build failed: \(error.localizedDescription)")
                }
                return
            }
            let outcome: Result<LoadedSession, Error> = await Task.detached(priority: .userInitiated) {
                let scopeAccessed = url.startAccessingSecurityScopedResource()
                defer {
                    if scopeAccessed {
                        url.stopAccessingSecurityScopedResource()
                    }
                }
                do {
                    let loaded = try CheckpointManager.loadSession(at: url)
                    // Apply champion weights immediately; trainer
                    // weights are held for the next startRealTraining.
                    try await champion.loadWeights(loaded.championFile.weights)
                    return .success(loaded)
                } catch {
                    return .failure(error)
                }
            }.value
            checkpoint?.checkpointSaveInFlight = false
            switch outcome {
            case .success(let loaded):
                champion.identifier = ModelID(value: loaded.championFile.modelID)
                pendingLoadedSession = loaded
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

    nonisolated static let trainerLearningRateDefault: Float = 5e-5
    nonisolated static let entropyRegularizationCoeffDefault: Float = 1e-3

    /// Build the Codable snapshot of the current session state (counters,
    /// hyperparameters, arena history, replay-buffer footprint, build info).
    /// Called at save time with the live state read off the main actor by both
    /// the manual/periodic save path and `runArenaParallel`'s post-promotion
    /// save. Closes the active training segment at save time (and re-opens a
    /// fresh one if training is still in progress) so the on-disk cumulative
    /// wall-time totals stay correct across mid-training saves.
    @MainActor
    func buildCurrentSessionState(
        championID: String,
        trainerID: String
    ) -> SessionCheckpointState {
        let params = TrainingParameters.shared
        let wasTraining = realTraining
        checkpoint?.closeActiveTrainingSegment(reason: "save")
        if wasTraining && checkpoint?.activeSegmentStart == nil {
            checkpoint?.beginActiveTrainingSegment()
        }
        let now = Date()
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
                championID: record.championID?.description
            )
        }
        let lr = trainer?.learningRate ?? Self.trainerLearningRateDefault
        let entropyCoeff = trainer?.entropyRegularizationCoeff ?? Self.entropyRegularizationCoeffDefault
        let drawPen = trainer?.drawPenalty ?? Float(params.drawPenalty)
        let bufferSnap = replayBuffer?.stateSnapshot()
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
            trainingPositionsSeen: (trainingSnap?.steps ?? 0) * params.trainingBatchSize,
            batchSize: params.trainingBatchSize,
            learningRate: lr,
            entropyRegularizationCoeff: entropyCoeff,
            drawPenalty: drawPen,
            promoteThreshold: params.arenaPromoteThreshold,
            arenaGames: params.arenaGamesPerTournament,
            arenaConcurrency: params.arenaConcurrency,
            selfPlayTau: TauConfigCodable(samplingScheduleBox?.selfPlay ?? buildSelfPlaySchedule()),
            arenaTau: TauConfigCodable(samplingScheduleBox?.arena ?? buildArenaSchedule()),
            selfPlayWorkerCount: params.selfPlayWorkers,
            gradClipMaxNorm: Float(params.gradClipMaxNorm),
            weightDecayCoeff: Float(params.weightDecay),
            policyLossWeight: Float(params.policyLossWeight),
            valueLossWeight: Float(params.valueLossWeight),
            momentumCoeff: Float(params.momentumCoeff),
            illegalMassPenaltyWeight: Float(params.illegalMassWeight),
            policyLabelSmoothingEpsilon: Float(params.policyLabelSmoothingEpsilon),
            valueLabelSmoothingEpsilon: Float(params.valueLabelSmoothingEpsilon),
            replayRatioTarget: params.replayRatioTarget,
            replayRatioAutoAdjust: params.replayRatioAutoAdjust,
            stepDelayMs: params.trainingStepDelayMs,
            selfPlayDelayMs: params.selfPlayDelayMs,
            lastAutoComputedDelayMs: lastAutoComputedDelayMs,
            // Schema-expansion fields (close the autotrain reproducibility gap
            // — these previously lived only in @AppStorage / @State and so
            // silently picked up the user's current preference on resume rather
            // than the session's saved value). All Optional for back-compat.
            lrWarmupSteps: params.lrWarmupSteps,
            sqrtBatchScalingForLR: params.sqrtBatchScalingLR,
            replayBufferMinPositionsBeforeTraining: params.replayBufferMinPositionsBeforeTraining,
            arenaAutoIntervalSec: params.arenaAutoIntervalSec,
            candidateProbeIntervalSec: params.candidateProbeIntervalSec,
            legalMassCollapseThreshold: params.legalMassCollapseThreshold,
            legalMassCollapseGraceSeconds: params.legalMassCollapseGraceSeconds,
            legalMassCollapseNoImprovementProbes: params.legalMassCollapseNoImprovementProbes,
            batchStatsInterval: params.batchStatsInterval,
            maxPliesFromAnyOneGame: params.maxPliesFromAnyOneGame,
            targetSampledGameLengthPlies: params.targetSampledGameLengthPlies,
            maxDrawPercentPerBatch: params.maxDrawPercentPerBatch,
            selfPlayDrawKeepFraction: params.selfPlayDrawKeepFraction,
            maxPliesPerGame: params.maxPliesPerGame,
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
        ).withTrainingSegments(segments)
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
