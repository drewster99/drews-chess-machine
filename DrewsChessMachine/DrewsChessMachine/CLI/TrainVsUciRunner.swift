import Foundation
import Darwin
import os

/// Cross-thread one-shot "please stop" flag for the train-vs-UCI loop.
/// The SIGINT `DispatchSource` handler flips it; the training loop reads
/// it once per step. Mirrors `CorpusReplayRunner`'s abort flag.
private final class TrainVsUciAbortFlag: @unchecked Sendable {
    private let state = OSAllocatedUnfairLock(initialState: false)
    func request() { state.withLock { $0 = true } }
    var isRequested: Bool { state.withLock { $0 } }
}

/// One opponent kind parsed from a `--train-vs-uci` spec.
struct TrainVsUciOpponentSpec: Sendable {
    /// Engine executable path.
    let command: String
    /// Number of concurrent instances of this engine.
    let count: Int
    /// The `go` limit sent every move, e.g. `"nodes 1"`, `"depth 4"`.
    let goLimit: String
    /// `setoption` pairs applied at handshake (e.g. `UCI_Elo=1400`).
    let options: [UCIArbiter.Option]
    /// Aggregation label (executable basename, e.g. `"stockfish"`).
    let kind: String
}

/// Configuration for one `--train-vs-uci` run.
struct TrainVsUciConfig: Sendable {
    var opponents: [TrainVsUciOpponentSpec]
    var stepLimit: Int?
    var timeLimitSec: Double?
    var startModelPath: String?
    /// Continue the `--start-model`'s training exactly: restore its complete
    /// trainer state (fp32 masters, optimizer velocity, the completed-step
    /// clock, warmup length and LR/momentum cycle — see
    /// `TrainerScheduleState`), so warmup does not re-run and the cycle phase
    /// and decay continue where they stopped. The replay buffer cannot be
    /// restored — its games were played live and are not persisted — so it
    /// refills from new games before training resumes. Without this flag,
    /// `--start-model` starts a new branch (fresh clock, zero velocity).
    var resumeExact: Bool
    var presetName: String?
    /// Explicit destination for the rolling trainer-model file; nil derives
    /// `<start-model stem>-vsuci-latest.safetensors` next to `--start-model`,
    /// or `<runModelID>-vsuci-latest.safetensors` in the Models directory. A
    /// file already there is replaced only under
    /// `TrainerOutputFileGuard.checkRollingOutput`'s rule (see
    /// `CorpusReplayConfig.outModelPath`).
    var outModelPath: String?
    /// `--overwrite-out-model` — see `CorpusReplayConfig.overwriteOutModel`.
    var overwriteOutModel: Bool
    /// `--enumerate-checkpoints`: also write `<stem>-vsuci-step<N>` copies,
    /// never over a file this run did not write (see
    /// `CorpusReplayConfig.enumerateCheckpoints`).
    var enumerateCheckpoints: Bool
    /// Max total half-moves before a game is dropped without flush.
    var maxPliesPerGame: Int
    /// How often (in trainer steps) to refresh the play network's weights
    /// from the live trainer. Small = closer to truly-live play.
    var evalSyncEverySteps: Int
    var runModelID: String
    /// Destination for the run's `results.json` (`--output`, checked before
    /// the run by `CliResultsOutput.preflight`), or nil for no JSON.
    var output: CliResultsOutput?
}

enum TrainVsUciError: LocalizedError {
    case noOpponents
    case startModelTooSmall(have: Int, need: Int)
    case noGamesProduced

    var errorDescription: String? {
        switch self {
        case .noOpponents:
            return "--train-vs-uci requires at least one opponent engine"
        case let .startModelTooSmall(have, need):
            return "--start-model has \(have) weight tensors but the network needs at least \(need)"
        case .noGamesProduced:
            return "no games were produced (all opponent engines failed to start or produce moves)"
        }
    }
}

/// Headless trainer that plays the live trainer network against a pool of
/// external UCI engines and trains on the resulting games — the live
/// analog of `--replay-corpus`. Invoked from the `--train-vs-uci`
/// CLI pre-flight handler.
enum TrainVsUciRunner {

    struct Result: Sendable {
        var steps: Int
        var gamesCompleted: Int
    }

    private static func emit(_ message: String) {
        SessionLogger.shared.log(message)
        print(message)
    }

    /// Run to completion and exit the process. Never returns.
    static func runAndExit(config: TrainVsUciConfig, params: ReplayParams) -> Never {
        SessionLogger.shared.start()
        emit("[VS-UCI] starting train-vs-UCI over \(config.opponents.count) opponent kind(s)")

        // Ctrl-C: first press requests a clean abort (finish the step, save,
        // exit); second press force-quits. Same pattern as CorpusReplayRunner.
        let abort = TrainVsUciAbortFlag()
        signal(SIGINT, SIG_IGN)
        let sigSource = DispatchSource.makeSignalSource(signal: SIGINT, queue: .global())
        sigSource.setEventHandler {
            if abort.isRequested {
                signal(SIGINT, SIG_DFL)
                raise(SIGINT)
                return
            }
            abort.request()
            emit("[VS-UCI] SIGINT received — finishing current step, saving, then exiting (Ctrl-C again to force-quit)")
        }
        sigSource.resume()

        let result: Result
        do {
            result = try withExtendedLifetime(sigSource) {
                try syncWait {
                    try await runTraining(config: config, params: params, abort: abort)
                }
            }
        } catch {
            FileHandle.standardError.write(Data("train-vs-uci: failed: \(error.localizedDescription)\n".utf8))
            SessionLogger.shared.log("[VS-UCI] failed: \(error.localizedDescription)")
            SessionLogger.shared.shutdown()
            Darwin.exit(33)
        }
        emit("[VS-UCI] done: steps=\(result.steps) gamesCompleted=\(result.gamesCompleted)")
        SessionLogger.shared.shutdown()
        Darwin.exit(0)
    }

    // MARK: - The run

    private static func runTraining(config: TrainVsUciConfig, params p: ReplayParams, abort: TrainVsUciAbortFlag) async throws -> Result {
        // `--output` support. Only allocated when a destination was given, so a
        // run without `--output` carries no per-step recording cost at all.
        let recorder: CliTrainingRecorder? = config.output == nil ? nil : {
            let r = CliTrainingRecorder()
            r.setSessionID(config.runModelID)
            r.setRunKind(.trainVsUci)
            return r
        }()
        let runStart = CFAbsoluteTimeGetCurrent()
        guard !config.opponents.isEmpty else { throw TrainVsUciError.noOpponents }

        // Resolve architecture from --start-model (embeds its own arch) or a
        // fresh preset / the current default.
        let arch: NetworkArchitecture
        let startModelFile: ModelCheckpointFile?
        let parentModelID: String
        // Set only for `--resume-exact` (see `TrainVsUciConfig.resumeExact`).
        var resumeSnapshot: TrainerResumeSnapshot? = nil
        if let sm = config.startModelPath {
            let url = URL(fileURLWithPath: (sm as NSString).expandingTildeInPath)
            let file = try CheckpointManager.loadModelFile(at: url)
            startModelFile = file
            parentModelID = file.modelID
            arch = file.architecture
            emit("[VS-UCI] start-model: \(url.lastPathComponent) modelID=\(file.modelID) encoding=\(arch.inputEncoding.rawValue)")
            if config.resumeExact {
                resumeSnapshot = try TrainerResumeSnapshot(checkpoint: file, fileName: url.lastPathComponent)
                let precisionDecision = PolicyTailPrecisionResume.exactResumeDecision(
                    saved: file.metadata.trainerPolicyTailPrecision,
                    running: ChessNetwork.PolicyTailPrecision.process
                )
                emit(precisionDecision.logLine)
                if let refusal = precisionDecision.refusal {
                    FileHandle.standardError.write(Data("error: \(refusal)\n".utf8))
                    Darwin.exit(2)
                }
            }
        } else {
            startModelFile = nil
            parentModelID = ""
            if let pn = config.presetName {
                guard let preset = NetworkArchitecture.Preset(rawValue: pn) else {
                    let names = NetworkArchitecture.Preset.allCases.map(\.rawValue).joined(separator: ", ")
                    FileHandle.standardError.write(Data("error: unknown --preset '\(pn)'. Available: \(names)\n".utf8))
                    Darwin.exit(2)
                }
                arch = NetworkArchitecture.preset(preset)
                emit("[VS-UCI] fresh net from preset: \(pn)")
            } else {
                arch = NetworkArchitecture.current
            }
        }

        emit("[VS-UCI-ARCH] (\(startModelFile == nil ? "default preset" : "start-model")) \(arch.architectureSummary)")

        // Rolling trainer-model output file (mirrors CorpusReplayRunner),
        // checked before any network is built or engine launched: a file
        // already there is replaced only when it is the rolling file of the
        // model line this run continues (or with --overwrite-out-model), and
        // an enumerated stem must not already hold step files this run could
        // reach.
        let outModelURL: URL = {
            if let explicit = config.outModelPath {
                let url = URL(fileURLWithPath: (explicit as NSString).expandingTildeInPath)
                return url.pathExtension.lowercased() == "safetensors" ? url : url.appendingPathExtension("safetensors")
            }
            if let sm = config.startModelPath {
                let smURL = URL(fileURLWithPath: (sm as NSString).expandingTildeInPath)
                let stem = smURL.deletingPathExtension().lastPathComponent
                return smURL.deletingLastPathComponent().appendingPathComponent("\(stem)-vsuci-latest.safetensors")
            }
            return CheckpointPaths.modelsDir.appendingPathComponent("\(config.runModelID)-vsuci-latest.safetensors")
        }()
        // Every save lands on a multiple of this (plus the final save).
        let autosaveEvery = 1000
        let rollingPlan = try TrainerOutputFileGuard.checkRollingOutput(
            outModelURL: outModelURL,
            startModelURL: config.startModelPath.map { URL(fileURLWithPath: ($0 as NSString).expandingTildeInPath) },
            startModel: startModelFile.map {
                TrainerModelFileIdentity(modelID: $0.modelID, trainingStep: $0.metadata.trainingStep)
            },
            overwriteAuthorized: config.overwriteOutModel)
        let rollingWriter = RollingTrainerModelWriter(url: outModelURL, plan: rollingPlan)
        emit("[VS-UCI] trainer-model output: \(outModelURL.path) (\(rollingPlan.logDescription))")
        let enumeratedWriter: EnumeratedCheckpointWriter?
        if config.enumerateCheckpoints {
            let naming = EnumeratedCheckpointNaming(rollingOutputURL: outModelURL, runTag: EnumeratedCheckpointNaming.trainVsUciRunTag)
            try TrainerOutputFileGuard.requireNoReachableEnumeratedCheckpoints(naming: naming, stepLimit: config.stepLimit)
            enumeratedWriter = EnumeratedCheckpointWriter(naming: naming)
            emit("[VS-UCI] enumerated checkpoints: \(naming.url(step: autosaveEvery).path) and siblings (never overwritten)")
        } else {
            enumeratedWriter = nil
        }

        emit("[VS-UCI] building play network + trainer (encoding=\(arch.inputEncoding.rawValue))")
        // `evalNet` is the network the driver plays with. It is kept ~live by
        // syncing its weights from the trainer every `evalSyncEverySteps`
        // steps (see the loop). A separate instance from the trainer's graph
        // avoids concurrent eval/train GPU access to one network and the
        // ChessNetwork/ChessMPSNetwork type mismatch (trainer.network is a
        // ChessNetwork; the driver + ActiveGame need a ChessMPSNetwork).
        let evalNet = try ChessMPSNetwork(.randomWeights, arch: arch)
        // Configured through `TrainerHyperparameters` — the same path the GUI
        // session and corpus replay use — so every trainer-level parameter
        // lands, including the LR/momentum cycle, dropout and the stats /
        // KL-probe intervals.
        // An exact resume trains under the checkpoint's own schedule — see
        // `[REPLAY-RESUME]` in CorpusReplayRunner.
        var resumedHyperparameters = p.trainer
        if let resumeSnapshot {
            for line in p.trainer.scheduleDifferences(from: resumeSnapshot.schedule) {
                emit("[VS-UCI-RESUME] WARNING \(line)")
            }
            resumedHyperparameters = p.trainer.adoptingSchedule(resumeSnapshot.schedule)
        }
        let hp = resumedHyperparameters
        let trainer = try ChessTrainer(
            dropoutStream: RunMasterSeed.systemDrawn(context: "train-vs-uci").generator(.dropout),
            hyperparameters: hp, arch: arch
        )
        emit(ChessNetwork.PolicyTailPrecision.processLogLine)
        // Field for field with `[REPLAY-HPARAMS]` so the two CLI paths can be
        // diffed directly.
        emit(String(
            format: "[VS-UCI-HPARAMS] lr=%.6g batch=%ld wd=%.4g momentum=%.3g gradClip=%.3g entropyBonus=%.4g drawPenalty=%.4g policyW=%.3g valueW=%.3g illegalW=%.4g ",
            Double(hp.learningRate), p.trainingBatchSize, Double(hp.weightDecayC), Double(hp.momentumCoeff), Double(hp.gradClipMaxNorm),
            Double(hp.entropyRegularizationCoeff), Double(hp.drawPenalty), Double(hp.policyLossWeight), Double(hp.valueLossWeight), Double(hp.illegalMassPenaltyWeight)
        )
            + hp.policyLabelSmoothingLogFields
            + String(
                format: " vLabelSmooth=%.4g dropout=%.4g lrWarmup=%ld bufCap=%ld",
                Double(hp.valueLabelSmoothingEpsilon), Double(hp.dropoutRate),
                hp.lrWarmupSteps, p.replayBufferCapacity
            )
            + " complementCE=\(hp.useSignedAdvantageComplementCE ? "on" : "off")"
            + " sqrtBatchLR=\(hp.sqrtBatchScalingForLR ? "on" : "off")"
            + " batchStats=\(hp.batchStatsInterval) klProbe=\(hp.klProbeInterval)")
        let buffer = ReplayBuffer(capacity: p.replayBufferCapacity, inputEncoding: evalNet.inputEncoding)

        // Number of base tensors (trainables + BN running stats) — the prefix
        // both the trainer's masters/working copy and evalNet's inference net
        // are seeded from and re-synced with.
        let baseCount = evalNet.network.trainableVariables.count + evalNet.network.bnRunningStatsVariables.count

        if let file = startModelFile {
            guard file.weights.count >= baseCount else {
                throw TrainVsUciError.startModelTooSmall(have: file.weights.count, need: baseCount)
            }
            try await evalNet.network.loadWeights(file.networkWeights)
            if let resumeSnapshot {
                try await trainer.restoreExactly(from: resumeSnapshot)
                emit("[VS-UCI] start-model trainer state restored exactly (fp32 masters, velocity, trainerStep=\(trainer.completedTrainSteps)) + play net (base tensors=\(baseCount))")
            } else {
                try await trainer.loadBaseWeightsResetVelocity(file.networkWeights)
                emit("[VS-UCI] start-model weights loaded into trainer + play net (velocity zeroed; new branch) (base tensors=\(baseCount))")
            }
        } else {
            // Fresh run: seed evalNet from the trainer so both start identical.
            let base = Array((try await trainer.network.exportWeights()).prefix(baseCount))
            try await evalNet.network.loadWeights(base)
        }

        // The resolved LR/momentum schedule, once, with the trainer step it
        // continues from — see `[REPLAY-CYCLE]`.
        let launch: TrainerLaunchKind = resumeSnapshot != nil
            ? .exactResume(ofModelID: parentModelID)
            : (startModelFile != nil ? .newBranch(fromModelID: parentModelID) : .fresh)
        emit("[VS-UCI-CYCLE] \(LRMomentumCycleLogFormat.cycleDescription(trainer.lrMomentumCycle)) "
            + LRMomentumCycleLogFormat.scheduleOrigin(of: trainer, launch: launch))

        // Build the opponent pool: one UCIArbiter per instance.
        var opponents: [TrainVsUciDriver.Opponent] = []
        for spec in config.opponents {
            for k in 1...max(1, spec.count) {
                let label = "\(spec.kind)#\(k)"
                let arbiterConfig = UCIArbiter.Configuration(
                    command: URL(fileURLWithPath: (spec.command as NSString).expandingTildeInPath),
                    options: spec.options,
                    goLimit: spec.goLimit,
                    label: label
                )
                opponents.append(TrainVsUciDriver.Opponent(
                    arbiter: UCIArbiter(configuration: arbiterConfig),
                    kind: spec.kind,
                    instanceLabel: label))
            }
        }
        emit("[VS-UCI] opponent pool: " + config.opponents.map { "\($0.kind)×\($0.count) [go=\($0.goLimit)]" }.joined(separator: ", "))

        let driver = TrainVsUciDriver(
            network: evalNet,
            buffer: buffer,
            opponents: opponents,
            // DCM side plays deterministic best-move, exactly like the `--uci`
            // engine's default (`Temperature=0` → `.argmax`). Game variety must
            // come from start positions, not temperature (see `.argmax` doc).
            schedule: .argmax,
            maxPliesPerGame: config.maxPliesPerGame)

        // Consecutive-failure tracking per kind of save — see
        // `TrainerSaveFailureStreak` and `CorpusReplayRunner.reportSaveFailure`.
        var rollingSaveFailures = TrainerSaveFailureStreak(what: "trainer-model save")
        var enumeratedSaveFailures = TrainerSaveFailureStreak(what: "enumerated checkpoint save")
        func saveTrainerModel(step: Int, reason: String) async throws {
            let encoded: Data
            do {
                // Complete trainer state, resumable with `--resume-exact`;
                // `training_step` stays segment-local as in CorpusReplayRunner.
                // The SGD loop awaits each step, so none is in flight here.
                let snapshot = try await trainer.exportResumeSnapshot()
                let weights = snapshot.trainerWeights
                let metadata = ModelCheckpointMetadata.trainerFile(
                    creator: "train-vs-uci",
                    trainingStep: step,
                    parentModelID: parentModelID,
                    notes: "train-vs-uci \(reason) @ step \(step)",
                    schedule: snapshot.schedule,
                    policyTailPrecision: trainer.policyTailPrecision)
                encoded = try SafetensorsModelIO.encode(
                    modelID: config.runModelID,
                    createdAtUnix: Int64(Date().timeIntervalSince1970),
                    metadata: metadata,
                    weights: weights,
                    architecture: arch,
                    includesVelocity: true,
                    resumeMetadata: [:])
                try FileManager.default.createDirectory(
                    at: outModelURL.deletingLastPathComponent(), withIntermediateDirectories: true)
                try rollingWriter.write(encoded)
                rollingSaveFailures.recordSuccess()
                emit("[VS-UCI] saved trainer model (\(reason)) step=\(step) trainerStep=\(snapshot.schedule.completedTrainSteps) -> \(outModelURL.lastPathComponent)")
                // Full layer health of the state just written — see
                // CorpusReplayRunner's save. Never throws.
                let health = await LayerHealthLog.checkpoint(
                    arch: arch, trainerWeights: weights, context: "vsuci-\(reason)",
                    step: step, trainerStep: snapshot.schedule.completedTrainSteps)
                for line in health.lines { emit(line) }
                if let summary = health.summary {
                    recorder?.appendLayerHealth(CliTrainingRecorder.LayerHealthRecord(
                        step: step, trainerStep: snapshot.schedule.completedTrainSteps,
                        context: "vsuci-\(reason)", summary: summary))
                }
            } catch let ownershipRefusal as FileSafetyError where ownershipRefusal.isOwnershipRefusal {
                // The rolling path no longer holds the file this run owns —
                // halt rather than overwrite something this run did not write.
                throw ownershipRefusal
            } catch {
                try CorpusReplayRunner.reportSaveFailure(error, step: step, what: "trainer-model save (\(reason))")
                try rollingSaveFailures.recordFailure(step: step)
                return
            }
            if let enumeratedWriter {
                do {
                    let written = try enumeratedWriter.write(encoded, step: step)
                    enumeratedSaveFailures.recordSuccess()
                    let note = written.outcome == .replacedThisRunsEarlierSave
                        ? " (replaced this run's own earlier save of step \(step))"
                        : ""
                    emit("[VS-UCI] enumerated checkpoint -> \(written.url.lastPathComponent)\(note)")
                } catch let collision as TrainerOutputFileError {
                    throw collision
                } catch let ownershipRefusal as FileSafetyError where ownershipRefusal.isOwnershipRefusal {
                    throw ownershipRefusal
                } catch {
                    try CorpusReplayRunner.reportSaveFailure(error, step: step, what: "enumerated checkpoint")
                    try enumeratedSaveFailures.recordFailure(step: step)
                }
            }
        }

        /// Refresh the play network's weights from the live trainer.
        ///
        /// Runs on the training task while the driver task is calling
        /// `evalNet.evaluateBatched` concurrently — but this is race-free:
        /// `ChessNetwork` funnels BOTH `loadWeights` and `evaluateBatched`
        /// (via `enqueue`) through its single serial `executionQueue`
        /// (`drewschess.chessnetwork.serial`), so they strictly serialize and
        /// any in-flight eval sees a complete pre- or post-sync weight set,
        /// never a torn one.
        func syncEvalNet() async throws {
            let base = Array((try await trainer.network.exportWeights()).prefix(baseCount))
            try await evalNet.network.loadWeights(base)
        }

        let batchSize = max(1, p.trainingBatchSize)
        let minPrefill = max(batchSize, p.replayBufferMinPositionsBeforeTraining)
        let syncEvery = max(1, config.evalSyncEverySteps)
        emit("[VS-UCI] batchSize=\(batchSize) minPrefill=\(minPrefill) evalSyncEvery=\(syncEvery) stepLimit=\(config.stepLimit.map(String.init) ?? "none") timeLimit=\(config.timeLimitSec.map { String(format: "%.0fs", $0) } ?? "none")")

        // Start the game producer. `driverDone` flips when driver.run()
        // returns on its own — in practice only when every engine failed to
        // launch/handshake (a healthy producer runs until cancelled) — so the
        // prefill loop below can fail fast instead of sitting out its full
        // deadline against a producer that has already given up.
        let driverDone = SyncBox<Bool>(false)
        let driverTask = Task { await driver.run(); driverDone.value = true }

        // Periodic [VS-UCI-STATS] block: per-kind summary lines, then a
        // per-instance breakdown, with rates over the window since the
        // previous emit. Runs until teardown cancels it.
        let statsIntervalSec: Double = 12
        let statsTask = Task {
            var previous: [TrainVsUciDriver.SlotStats] = []
            var lastEmit = Date()
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(statsIntervalSec))
                if Task.isCancelled { break }
                let now = Date()
                let snapshot = driver.statsSnapshot()
                for line in TrainVsUciStatsFormatter.lines(
                    current: snapshot,
                    previous: previous,
                    intervalSec: now.timeIntervalSince(lastEmit)
                ) {
                    emit(line)
                }
                previous = snapshot
                lastEmit = now
            }
        }

        func dg(_ v: Float, _ digits: Int) -> String { v.isFinite ? String(format: "%.\(digits)f", v) : "--" }

        let startWall = Date()
        func elapsed() -> Double { Date().timeIntervalSince(startWall) }
        func overTime() -> Bool { if let tl = config.timeLimitSec { return elapsed() >= tl }; return false }

        // Everything from here to teardown runs with the producer live. Any
        // throw must cancel + await the driver first — `runAndExit`'s
        // `Darwin.exit` would otherwise bypass the driver's engine shutdown
        // and orphan the external UCI engine subprocesses.
        var step = 0
        var aborted = false
        // Which condition actually ended the loop, for the recorded termination
        // reason. Inferring it from the configured limits is wrong when both a
        // step limit and a time limit are set.
        var timedOut = false
        do {
            // Wait for the producer to prefill the buffer. Games are produced
            // by actually playing the engines, so this takes as long as it
            // takes (deep-search opponents, a large minPrefill) — there is no
            // arbitrary time deadline. The only bail is a terminal condition:
            // the driver task exited, which happens only when every engine
            // failed to launch/handshake, so no games will ever be produced.
            while buffer.count < minPrefill {
                if abort.isRequested { break }
                if overTime() { timedOut = true; break }
                if driverDone.value { throw TrainVsUciError.noGamesProduced }
                try await Task.sleep(for: .milliseconds(100))
            }
            emit("[VS-UCI] prefilled: bufCount=\(buffer.count)")

            // Step-locked SGD loop. Games are produced concurrently by the driver.
            let logEvery = 50
            while true {
                if abort.isRequested { aborted = true; emit("[VS-UCI] abort requested — stopping at step \(step)"); break }
                if let sl = config.stepLimit, step >= sl { break }
                if overTime() { timedOut = true; emit("[VS-UCI] time limit reached at step \(step)"); break }

                // The buffer is filled asynchronously; if the producer transiently
                // falls behind, wait rather than stopping.
                if buffer.count < batchSize {
                    try await Task.sleep(for: .milliseconds(50)); continue
                }
                guard let timing = try await trainer.trainStep(replayBuffer: buffer, batchSize: batchSize) else {
                    try await Task.sleep(for: .milliseconds(50)); continue
                }
                step += 1

                // Keep the play network ~live.
                if step % syncEvery == 0 { try await syncEvalNet() }

                if step == 1 || step % logEvery == 0 {
                    // Pin LR, momentum and the cycle values to one step-count
                    // observation so the three agree with each other.
                    let observedSteps = trainer.completedTrainSteps
                    let liveLR = trainer.effectiveLearningRate(forBatchSize: batchSize, completedSteps: observedSteps)
                    let liveMomentum = trainer.effectiveMomentum(completedSteps: observedSteps)
                    let cycleValues = trainer.lrMomentumCycleValues(completedSteps: observedSteps)
                    let line = "[VS-UCI] step=\(step)"
                        + String(format: " loss=%.4f pLoss=%.4f vLoss=%.4f", timing.loss, timing.policyLoss, timing.valueLoss)
                        + " pEnt=\(dg(timing.policyEntropy, 3)) playedP=\(dg(timing.playedMoveProb, 3))"
                        + " pLogitMean=\(dg(timing.policyLogitMean, 4)) vLogitMean=\(dg(timing.valueLogitMean, 4))"
                        + String(format: " gNorm=%.3f lr=%.3g ms=%.1f", timing.gradGlobalNorm, liveLR, timing.totalMs)
                        + " buf=\(buffer.count)"
                        + String(format: " mom=%.4f", liveMomentum)
                        + (cycleValues.learningRate != nil ? " lrCyc" + LRMomentumCycleLogFormat.envelopeBounds(cycleValues) : "")
                        + " trainerStep=\(observedSteps)"
                    emit(line)
                    // Live layer health at the same cadence (BN state +
                    // ReZero α, read on the trainer's queue between steps).
                    for healthLine in await LayerHealthLog.liveLines(trainer: trainer) {
                        emit(healthLine)
                    }
                    // Same cadence as the log line, so results.json and the log
                    // describe the same ticks.
                    // One snapshot, reused: `statsSnapshot()` is a lock-guarded
                    // COW array read, but taking it once keeps every derived
                    // field describing the same instant.
                    let slots = driver.statsSnapshot()
                    recorder?.appendStats(CliTrainingRecorder.StatsLine(
                        elapsedSec: CFAbsoluteTimeGetCurrent() - runStart,
                        steps: step,
                        // Plies appended to the replay buffer. `pliesPlayed`
                        // accrues only in the driver's `finishGame`, the same
                        // path that flushes the game, so aborted and
                        // cap-dropped games contribute nothing.
                        positionsFed: slots.reduce(0) { $0 + $1.pliesPlayed },
                        bufferCount: buffer.count,
                        bufferCapacity: p.replayBufferCapacity,
                        policyLoss: Double(timing.policyLoss),
                        valueLoss: Double(timing.valueLoss),
                        policyEntropy: timing.hasDiagnostics ? Double(timing.policyEntropy) : nil,
                        policyIllegalMassPenalty: Double(timing.illegalMassPenalty),
                        gradGlobalNorm: Double(timing.gradGlobalNorm),
                        playedMoveProb: timing.hasDiagnostics ? Double(timing.playedMoveProb) : nil,
                        valueMean: timing.hasDiagnostics ? Double(timing.valueMean) : nil,
                        valueAbsMean: timing.hasDiagnostics ? Double(timing.valueAbsMean) : nil,
                        valueProbWin: timing.hasDiagnostics ? Double(timing.valueProbWin) : nil,
                        valueProbDraw: timing.hasDiagnostics ? Double(timing.valueProbDraw) : nil,
                        valueProbLoss: timing.hasDiagnostics ? Double(timing.valueProbLoss) : nil,
                        // Diagnostic-step fields (these, and the entropy, played-move,
                        // value and W/D/L fields above): nil (not measured) on a step
                        // that skipped the diagnostic reductions.
                        policyLogitMean: timing.hasDiagnostics ? Double(timing.policyLogitMean) : nil,
                        valueLogitMean: timing.hasDiagnostics ? Double(timing.valueLogitMean) : nil,
                        batchSize: batchSize,
                        // Static base plus the cycle evaluated at this step —
                        // see CorpusReplayRunner.
                        trainerHyperparameters: hp,
                        cycleValues: cycleValues,
                        buildNumber: BuildInfo.buildNumber,
                        trainerID: config.runModelID,
                        // `positionsProduced` nil: the driver counts dropped
                        // GAMES (`capDropped`), never their plies, so the
                        // produced total is genuinely unmeasured here.
                        // `positions_trained` therefore falls back to the fed
                        // count — a lower bound, not the self-play
                        // raw-produced convention. `run_kind` disambiguates.
                        positionsProduced: nil,
                        // Games this run completed. Recorded under the
                        // self-play-shaped `self_play_games` key; `run_kind`
                        // at top level says what that means here.
                        gamesPlayed: slots.reduce(0) { $0 + $1.gamesCompleted },
                        pliesCapDropped: slots.reduce(0) { $0 + $1.capDropped },
                        maxPliesPerGame: config.maxPliesPerGame,
                        // nil: this path never reads the replay-ratio target,
                        // so emitting it would be a fresh false claim rather
                        // than a recovered one.
                        replayRatioTarget: nil
                    ))
                }
                if step % autosaveEvery == 0 {
                    try await saveTrainerModel(step: step, reason: "autosave")
                }
            }
        } catch {
            // Tear the producer down (shuts every engine down) before the
            // error propagates to runAndExit's Darwin.exit.
            statsTask.cancel()
            driverTask.cancel()
            _ = await driverTask.value
            throw error
        }

        // Stop the producer and wait for it to shut down its engines cleanly.
        statsTask.cancel()
        driverTask.cancel()
        _ = await driverTask.value

        // Sync one last time so the saved model reflects the final weights,
        // then final save.
        try await saveTrainerModel(step: step, reason: aborted ? "abort" : "final")

        // `results.json` last, after the final model save — a run that dies
        // saving weights should not also claim a clean results record.
        if let recorder, let output = config.output {
            // Reported from the cause the loop actually exited on, not inferred
            // from which limits were configured: a run with BOTH a step limit
            // and a time limit can exit on either, and inferring would mislabel
            // a timeout as `stepLimitReached`.
            recorder.setTerminationReason(
                aborted ? .manualStop : (timedOut ? .timerExpired : .stepLimitReached)
            )
            let counts = recorder.countsSnapshot()
            // Logged, not thrown — see CorpusReplayRunner: the trainer model is
            // already saved, so a failed results write must not fail the run.
            do {
                let written = try recorder.write(
                    to: output,
                    totalTrainingSeconds: CFAbsoluteTimeGetCurrent() - runStart
                )
                emit("[VS-UCI] wrote results: \(written.path) (stats=\(counts.stats))")
            } catch {
                emit("[VS-UCI] results write FAILED for \(output.url.path): \(error.localizedDescription)")
            }
        }

        let totalGames = driver.statsSnapshot().reduce(0) { $0 + $1.gamesCompleted }
        return Result(steps: step, gamesCompleted: totalGames)
    }

    // MARK: - async→sync bridge (mirrors CorpusReplayRunner.syncWait)

    private final class SyncBoxRef<T>: @unchecked Sendable {
        var success: T?
        var failure: Error?
    }

    private static func syncWait<T>(_ work: @Sendable @escaping () async throws -> T) throws -> T {
        let box = SyncBoxRef<T>()
        let semaphore = DispatchSemaphore(value: 0)
        Task.detached(priority: .userInitiated) {
            do { box.success = try await work() }
            catch { box.failure = error }
            semaphore.signal()
        }
        semaphore.wait()
        if let error = box.failure { throw error }
        guard let success = box.success else {
            preconditionFailure("TrainVsUciRunner.syncWait: result box carried neither success nor failure")
        }
        return success
    }
}
