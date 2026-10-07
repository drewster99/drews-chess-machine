import Foundation
import Darwin
import os

/// Cross-thread one-shot "please stop" flag for the train-vs-UCI loop.
/// The SIGINT `DispatchSource` handler flips it; the training loop reads
/// it once per step. Mirrors `CorpusReplayRunner`'s abort flag.
final class TrainVsUciAbortFlag: @unchecked Sendable {
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
    /// A model file or a `.dcmsession` folder (`TrainVsUciSession.StartSource`).
    var startModelPath: String?
    /// Continue the `--start-model`'s training exactly: restore its complete
    /// trainer state (fp32 masters, optimizer velocity, the completed-step
    /// clock, warmup length and LR/momentum cycle — see
    /// `TrainerScheduleState`) and the run's random streams, so warmup does
    /// not re-run and the cycle phase and decay continue where they stopped.
    /// A session folder saved with `--save-replay-buffer` also restores the
    /// replay buffer; any other start refills it from new games before
    /// training resumes (`NOT EXACT: buffer`). Without this flag,
    /// `--start-model` starts a new branch (fresh clock, zero velocity).
    var resumeExact: Bool
    /// Resume gaps `--resume-exact` may proceed without (`--accept-inexact`,
    /// determinism plan D-7). A start without a saved replay buffer needs
    /// `buffer` named for an exact resume to proceed.
    var acceptInexact: Set<ResumeGap>
    var presetName: String?
    /// Where the run's session folders are written (`--out-session-dir`; the
    /// app's `Sessions/` folder by default). Every save is a new folder.
    var sessionDirectory: URL
    /// `--save-replay-buffer`: session saves include `replay_buffer.bin`, so an
    /// exact resume from them restores the buffer (plan D-8). Off by default:
    /// a save is then far smaller and a resume refills from new games.
    var saveReplayBuffer: Bool
    /// `--enumerate-checkpoints`: also write `<stem>-vsuci-step<N>` trainer
    /// files every 1000 steps and at the end, never over a file this run did
    /// not write (see `CorpusReplayConfig.enumerateCheckpoints`).
    var enumerateCheckpoints: Bool
    /// `--checkpoint-stem`: the path stem of those step files
    /// (`TrainVsUciSession.enumeratedNamingBase`), nil for the default.
    var checkpointStem: String?
    /// Max total half-moves before a game is dropped without flush.
    var maxPliesPerGame: Int
    /// How often (in trainer steps) to refresh the play network's weights
    /// from the live trainer. Small = closer to truly-live play.
    var evalSyncEverySteps: Int
    var runModelID: String
    /// Destination for the run's `results.json` (`--output`, checked before
    /// the run by `CliResultsOutput.preflight`), or nil for no JSON.
    var output: CliResultsOutput?
    /// The run's master seed (`RunRandomSeed.resolve`, at launch): the
    /// replay buffer's `sampler` stream and each game's `vsuci.game.<serial>`
    /// stream derive from it.
    var runRandomSeed: RunRandomSeed
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
        /// The training-health alarm whose stop ended the run (honoured by
        /// the loop, final save succeeded), or nil; `runAndExit` then exits
        /// with `CorpusReplayRunner.trainingHealthStopExitStatus`.
        var healthStop: TrainingHealthEvent?
    }

    private static func emit(_ message: String) {
        SessionLogger.shared.log(message)
        print(message)
    }

    /// Run to completion and exit the process. Never returns. Exit status 0
    /// on success, 2 when the run is refused before it starts
    /// (`CLIRunRefusal`), 33 on any other failure, 35 when a training-health
    /// alarm stopped the run after a successful final save.
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

        // Synchronous pre-flight: hash the engine executables here, on this
        // thread, before the async run starts (`TrainVsUciOpponentExecutableDigests`).
        let executableDigests: TrainVsUciOpponentExecutableDigests
        do {
            executableDigests = try TrainVsUciOpponentExecutableDigests(hashingExecutablesOf: config)
        } catch {
            FileHandle.standardError.write(Data("train-vs-uci: failed: \(error.localizedDescription)\n".utf8))
            SessionLogger.shared.log("[VS-UCI] failed: \(error.localizedDescription)")
            SessionLogger.shared.shutdown()
            Darwin.exit(33)
        }

        let result: Result
        do {
            result = try withExtendedLifetime(sigSource) {
                try syncWait {
                    try await runTraining(config: config, params: params, executableDigests: executableDigests,
                                          abort: abort)
                }
            }
        } catch let refusal as CLIRunRefusal {
            // Refused before any engine or network started: a usage
            // problem, status 2, with the log drained before the exit.
            FileHandle.standardError.write(Data("error: \(refusal.message)\n".utf8))
            SessionLogger.shared.log("[VS-UCI] refused: \(refusal.message)")
            SessionLogger.shared.shutdown()
            Darwin.exit(2)
        } catch {
            FileHandle.standardError.write(Data("train-vs-uci: failed: \(error.localizedDescription)\n".utf8))
            SessionLogger.shared.log("[VS-UCI] failed: \(error.localizedDescription)")
            SessionLogger.shared.shutdown()
            Darwin.exit(33)
        }
        emit("[VS-UCI] done: steps=\(result.steps) gamesCompleted=\(result.gamesCompleted)")
        if let healthStop = result.healthStop {
            let status = CorpusReplayRunner.trainingHealthStopExitStatus
            emit("[VS-UCI] stopped by training health alarm \(healthStop.rule.rawValue) (\(healthStop.severity.rawValue)) "
                + "at trainerStep=\(healthStop.trainerStep); exit status \(status)")
            SessionLogger.shared.shutdown()
            Darwin.exit(status)
        }
        SessionLogger.shared.shutdown()
        Darwin.exit(0)
    }

    // MARK: - The run

    /// The whole train-vs-UCI run. Internal (not private) only so tests can
    /// run its launch checks in-process; production enters through
    /// `runAndExit`.
    /// `executableDigests` is the pre-flight's hash of the opponents'
    /// executables (`runAndExit`).
    static func runTraining(config: TrainVsUciConfig, params configuredParams: ReplayParams,
                            executableDigests: TrainVsUciOpponentExecutableDigests,
                            abort: TrainVsUciAbortFlag) async throws -> Result {
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
        // The run's master seed: inherited from the checkpoint on an exact
        // resume that records the run's streams (see CorpusReplayRunner).
        var runSeed = config.runRandomSeed
        var resumedStreams: LineageRecord.RunStreams? = nil
        var resumeGaps: [ResumeGap] = []
        // What `--start-model` named: a model file, or a session folder whose
        // trainer file is the start (and whose replay buffer, when saved, an
        // exact resume restores).
        let startSource: TrainVsUciSession.StartSource? = try config.startModelPath.map {
            try TrainVsUciSession.startSource(path: $0)
        }
        var startSession: LoadedSession? = nil
        // The parameters the run trains under: the configured ones, with the
        // checkpoint's own schedule adopted on an exact resume (see
        // `ReplayParams.adoptingSchedule`).
        let p: ReplayParams
        if let startSource {
            let file: ModelCheckpointFile
            let fileName: String
            switch startSource {
            case .modelFile(let url):
                file = try CheckpointManager.loadModelFile(at: url)
                fileName = url.lastPathComponent
                emit("[VS-UCI] start-model: \(url.lastPathComponent) modelID=\(file.modelID) encoding=\(file.architecture.inputEncoding.rawValue)")
            case .session(let url):
                let loaded = try CheckpointManager.loadSession(at: url)
                startSession = loaded
                file = loaded.trainerFile
                fileName = "\(url.lastPathComponent)/\(SessionCheckpointLayout.trainerFilename)"
                emit("[VS-UCI] start-model: session \(url.lastPathComponent) trainer modelID=\(file.modelID) "
                    + "encoding=\(file.architecture.inputEncoding.rawValue) "
                    + "replayBuffer=\(loaded.replayBufferURL == nil ? "not saved" : "saved")")
            }
            startModelFile = file
            parentModelID = file.modelID
            arch = file.architecture
            if config.resumeExact {
                // Only a session saved with its buffer restores it; every
                // other exact resume refills from new games (decision D-8).
                if startSession?.replayBufferURL == nil {
                    resumeGaps.append(.buffer)
                }
                let snapshot = try TrainerResumeSnapshot(checkpoint: file, fileName: fileName)
                resumeSnapshot = snapshot
                // An exact resume trains under the checkpoint's own schedule
                // — see `[REPLAY-RESUME]` in CorpusReplayRunner.
                for line in configuredParams.trainer.scheduleDifferences(from: snapshot.schedule) {
                    emit("[VS-UCI-RESUME] WARNING \(line)")
                }
                p = try configuredParams.adoptingSchedule(snapshot.schedule)
                emit(PolicyTailPrecisionResume.exactResumeLogLine(
                    saved: file.metadata.trainerPolicyTailPrecision, running: ChessNetwork.PolicyTailPrecision.process))
                resumeGaps += PolicyTailPrecisionResume.gaps(
                    saved: file.metadata.trainerPolicyTailPrecision, running: ChessNetwork.PolicyTailPrecision.process)
                resumeGaps += ResumeGap.dropoutGaps(restoring: snapshot.dropoutRNG)
                // A history-less checkpoint is a gap only when this run clips
                // with the relative cap.
                resumeGaps += ResumeGap.gradNormHistoryGaps(
                    restoring: snapshot.gradNormHistory, runningMode: p.trainer.relativeGradientCap.mode)
                if let parentRecord = file.lineageParent.lineage.record {
                    if let streams = parentRecord.rng.streams, streams.nextGameSerial != nil {
                        do {
                            runSeed = try RunRandomSeed.inherited(
                                from: streams,
                                configuredSeed: config.runRandomSeed.configuredSeed,
                                commandLineSeed: config.runRandomSeed.origin == .commandLine ? config.runRandomSeed.masterSeed : nil)
                        } catch {
                            throw CLIRunRefusal(message: "--resume-exact: \(error.localizedDescription)")
                        }
                        resumedStreams = streams
                        // Each opponent instance's game index sets the
                        // trainer's colour; it continues only into the same
                        // opponent pool.
                        let instanceCount = config.opponents.reduce(0) { $0 + max(1, $1.count) }
                        if TrainVsUciDriver.continuedGameIndices(saved: streams.opponentGameIndices,
                                                                 instanceCount: instanceCount) == nil {
                            emit("[RESUME] opponents: the checkpoint records "
                                + (streams.opponentGameIndices.map { "\($0.count) opponent game indices" } ?? "no opponent game indices")
                                + ", this run has \(instanceCount) opponent instances — each starts at game 0")
                            resumeGaps.append(.serials)
                        }
                    } else {
                        resumeGaps += [.rngSampler, .serials]
                    }
                    if let parentParameters = parentRecord.parameters {
                        for line in try ParameterDifference.exactResumeLogLines(parent: parentParameters, inForce: p.parameters) {
                            emit(line)
                        }
                    } else {
                        resumeGaps.append(.params)
                    }
                    let environment = ResumeGap.environmentGaps(
                        writtenBy: parentRecord, runningBuild: try .current, runningDevice: .current,
                        runningFingerprint: try await BehaviorFingerprint.compute(
                            for: .init(arch: arch, policyTailPrecision: ChessNetwork.PolicyTailPrecision.process)))
                    for line in environment.logLines { emit(line) }
                    resumeGaps += environment.gaps
                } else {
                    resumeGaps += [.rngSampler, .serials, .params]
                }
                let exactness = ResumeExactness.resume(of: file.lineageParent, gaps: resumeGaps)
                emit(exactness.logLine)
                if let refusal = exactness.refusal(accepting: config.acceptInexact) {
                    throw CLIRunRefusal(message: refusal)
                }
            } else {
                p = configuredParams
            }
        } else {
            p = configuredParams
            startModelFile = nil
            parentModelID = ""
            if let pn = config.presetName {
                guard let preset = NetworkArchitecture.Preset(rawValue: pn) else {
                    let names = NetworkArchitecture.Preset.allCases.map(\.rawValue).joined(separator: ", ")
                    throw CLIRunRefusal(message: "unknown --preset '\(pn)'. Available: \(names)")
                }
                arch = NetworkArchitecture.preset(preset)
                emit("[VS-UCI] fresh net from preset: \(pn)")
            } else {
                arch = NetworkArchitecture.current
            }
        }

        for line in runSeed.parameterNotes { emit(line) }
        recorder?.setRunRandomSeed(runSeed)
        recorder?.setSamplingConstraints(p.samplingConstraints, batchSize: p.trainingBatchSize)
        emit("[VS-UCI-ARCH] (\(startModelFile == nil ? "default preset" : "start-model")) \(arch.architectureSummary)")

        // Session saves, checked before any network is built or engine
        // launched: the folder must be usable and a save's name must fit the
        // staging rename. Each save is a new `.dcmsession` folder written by
        // the GUI's session writer (see `TrainVsUciSession`).
        try CheckpointPaths.ensureDirectory(config.sessionDirectory)
        try FileSafety.requireStageableDestination(config.sessionDirectory.appendingPathComponent(
            CheckpointPaths.makeSessionDirectoryName(
                sessionID: config.runModelID, trigger: TrainVsUciSession.SaveKind.periodic.diskTag)))
        let periodicSessionIntervalSec = p.parameters.periodicAutosaveIntervalSec
        emit(TrainVsUciSession.launchLine(
            directory: config.sessionDirectory,
            periodicIntervalSec: periodicSessionIntervalSec,
            includesReplayBuffer: config.saveReplayBuffer))

        // The trainer step this segment starts from (the checkpoint's clock
        // on an exact resume, 0 otherwise), known before the trainer is built
        // and checked against it once it is — see CorpusReplayRunner.
        // Step-enumerated checkpoints (the per-step record probe loops and
        // dashboards read) land on its trainer-step multiples of
        // `TrainingStepLineSchedule.checkpointIntervalSteps`, plus the final
        // step, named by the trainer step.
        let segmentStartTrainerStep = resumeSnapshot?.schedule.completedTrainSteps ?? 0
        let enumeratedWriter: EnumeratedCheckpointWriter?
        if config.enumerateCheckpoints {
            let naming = EnumeratedCheckpointNaming(
                rollingOutputURL: try TrainVsUciSession.enumeratedNamingBase(
                    checkpointStem: config.checkpointStem, startSource: startSource,
                    runModelID: config.runModelID),
                runTag: EnumeratedCheckpointNaming.trainVsUciRunTag)
            try TrainerOutputFileGuard.requireNoReachableEnumeratedCheckpoints(
                naming: naming, segmentStartTrainerStep: segmentStartTrainerStep, stepLimit: config.stepLimit)
            enumeratedWriter = EnumeratedCheckpointWriter(naming: naming)
            let firstSave = TrainingStepLineSchedule.firstCheckpointStep(after: segmentStartTrainerStep)
            emit("[VS-UCI] enumerated checkpoints: \(naming.url(trainerStep: firstSave).path) and siblings (never overwritten)")
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
        // Its weights are always replaced before play: from --start-model, or
        // from the trainer's fresh initialization (see below).
        let evalNet = try ChessMPSNetwork(.overwrittenByLoad, arch: arch)
        // Configured through `TrainerHyperparameters` — the same path the GUI
        // session and corpus replay use — so every trainer-level parameter
        // lands, including the LR/momentum cycle, dropout and the stats /
        // KL-probe intervals. On an exact resume `p` already carries the
        // checkpoint's own schedule (adopted where the snapshot was read).
        let hp = p.trainer
        // A start model's weights replace the trainer's; a fresh run starts
        // from an init seed derived from the run seed, so `--seed`
        // reproduces its initialization.
        //
        // This segment's lineage follows from the same choice — see
        // CorpusReplayRunner. A vs-UCI resume starts from a fresh buffer, so
        // it is not exact in that either.
        let trainerInitialization: WeightInitialization
        let lineageStart: LineageTracker.Start
        if let file = startModelFile {
            trainerInitialization = .overwrittenByLoad
            lineageStart = resumeSnapshot != nil
                ? .resume(parent: file.lineageParent, gaps: resumeGaps, legacyTotals: nil)
                : .branch(parent: file.lineageParent)
        } else {
            let initSeed = runSeed.streams.freshModelInitSeed
            emit("[VS-UCI] fresh trainer init_seed=\(initSeed) init_scheme=\(WeightInitScheme.current) "
                + "(from run seed \(runSeed.masterSeed))")
            trainerInitialization = .seeded(initSeed: initSeed)
            lineageStart = .fresh(initialization: ModelInitRecord(initSeed: initSeed, scheme: WeightInitScheme.current))
        }
        let trainer = try ChessTrainer(
            dropoutStream: runSeed.streams.generator(.dropout),
            hyperparameters: hp, arch: arch, initialization: trainerInitialization
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
            + " batchStats=\(hp.batchStatsInterval) klProbe=\(hp.klProbeInterval)"
            + " stepLineSec=" + String(format: "%g", p.parameters.stepLineIntervalSec)
            + p.samplingConstraints.logFields(batchSize: p.trainingBatchSize)
            + " relClip=\(hp.relativeGradientCap.compactDescription)")
        emit(RelativeGradientCapLogFormat.configLine(hp.relativeGradientCap, hardMax: hp.gradClipMaxNorm))
        let buffer = ReplayBuffer(
            capacity: p.replayBufferCapacity,
            inputEncoding: evalNet.inputEncoding,
            sampler: runSeed.streams.generator(.sampler))
        buffer.setSamplingConstraints(p.samplingConstraints)
        // An exact resume from a session saved with its buffer continues from
        // that buffer. A restore that fails stops the run: an exact resume
        // that silently started from an empty buffer would not be one.
        if config.resumeExact, let startSession, let bufferURL = startSession.replayBufferURL {
            try await Task.detached(priority: .userInitiated) { [buffer] in
                try buffer.restore(from: bufferURL)
            }.value
            try CheckpointManager.verifyReplayBufferMatchesSession(buffer: buffer, state: startSession.state)
            let restored = buffer.stateSnapshot()
            emit("[RESUME] replay buffer restored: stored=\(restored.storedCount)/\(restored.capacity) "
                + "totalAdded=\(restored.totalPositionsAdded)")
        }
        if let resumedStreams {
            buffer.restoreSamplerState(resumedStreams.samplerState)
            emit("[RESUME] rng: sampler=restored")
        }

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
                if let resumedStreams {
                    try await trainer.restoreDropoutStreamState(resumedStreams.dropoutStreamState)
                    emit("[RESUME] rng: dropout stream=restored")
                }
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

        // The trainer's move selection, handed to the driver and recorded
        // from this one value (plan B1).
        let trainerMoveSelection: SamplingSchedule = .argmax
        try SegmentStartTrainerStepMismatch.require(
            preflight: segmentStartTrainerStep, trainerClock: trainer.completedTrainSteps)
        emit("[VS-UCI] " + TrainingStepLineSchedule.cadenceDescription(
            intervalSec: p.parameters.stepLineIntervalSec, startTrainerStep: segmentStartTrainerStep))

        let lineageTracker = try LineageTracker(
            start: lineageStart, pathKind: .vsuci, argv: CommandLine.arguments,
            startedAt: Date(), segmentStartTrainerStep: trainer.completedTrainSteps)
        try lineageTracker.configureSegment(LineageTracker.SegmentConfiguration(
            policyTailPrecision: trainer.policyTailPrecision,
            budget: LineageRecord.Budget(trainingStepLimit: config.stepLimit, trainingTimeLimitSec: config.timeLimitSec,
                                         epochLimit: nil),
            vsuci: try TrainVsUciSession.lineageGeneration(config: config, executableDigests: executableDigests,
                                                           trainerMoveSelection: trainerMoveSelection),
            selfPlayDirichlet: nil,
            startValueHeadRecentered: .recorded(try LineageTracker.startValueHeadRecentered(of: startModelFile))))
        lineageTracker.noteRunSeed(runSeed, atTrainerStep: trainer.completedTrainSteps)
        emit(RunProvenanceLine.line(
            record: try lineageTracker.startRecord(
                at: Date(), trainerCompletedSteps: trainer.completedTrainSteps, parameters: p.lineageParameters,
                inputs: lineageTracker.saveInputs(
                    scheduleAtSave: LRMomentumCycleReadout.scheduleAtSave(
                        inForce: p.parameters, completedTrainSteps: trainer.completedTrainSteps),
                    replayRatioAtSave: nil, healthAlarms: nil)),
            seed: runSeed))
        // session.json's positions trained: this run's steps at its batch,
        // on top of what a record says about the steps before it.
        let trainedPositions = TrainVsUciSession.trainedPositionsCount(
            startTrainerSteps: trainer.completedTrainSteps, startSession: startSession?.state,
            batchSize: p.trainingBatchSize)
        // Training-health alarms: one monitor for the run, its settings read
        // once from the run-start snapshot; logs `[HEALTH] config` beside
        // the `[RUN]` line.
        let trainingHealth = try CliTrainingHealth(
            parameters: p.parameters, arch: arch, path: "vsuci", pathTag: "[VS-UCI]",
            recorder: recorder, emit: { Self.emit($0) })

        // Build the opponent pool: one UCIArbiter per instance.
        var opponents: [TrainVsUciDriver.Opponent] = []
        // Which `config.opponents` spec each instance runs, for recording the
        // pool's engine identity.
        var opponentSpecIndices: [Int] = []
        for (specIndex, spec) in config.opponents.enumerated() {
            for k in 1...max(1, spec.count) {
                opponentSpecIndices.append(specIndex)
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

        // Game serials continue the saved run's on an exact resume, so later
        // games draw from streams that run never used.
        let gameSerials = GameSerialCounter(firstSerial: resumedStreams?.nextGameSerial ?? 0)
        // Each instance's next game continues the saved run's index, so the
        // trainer's colour alternation carries on. The games in progress at
        // the save are not in the checkpoint: each instance starts a new
        // game at that index, with a new serial.
        let startingGameIndices: [Int]
        if let saved = TrainVsUciDriver.continuedGameIndices(saved: resumedStreams?.opponentGameIndices,
                                                             instanceCount: opponents.count) {
            startingGameIndices = saved
            emit("[RESUME] opponents: the \(saved.count) games in progress at the save are not continued; "
                + "new games start at indices \(saved)")
        } else {
            startingGameIndices = Array(repeating: 0, count: opponents.count)
        }
        let driver = TrainVsUciDriver(
            network: evalNet,
            buffer: buffer,
            opponents: opponents,
            // DCM side plays deterministic best-move, exactly like the `--uci`
            // engine's default (`Temperature=0` → `.argmax`). Game variety must
            // come from start positions, not temperature (see `.argmax` doc).
            schedule: trainerMoveSelection,
            maxPliesPerGame: config.maxPliesPerGame,
            randomStreams: runSeed.streams,
            gameSerials: gameSerials,
            startingGameIndices: startingGameIndices)

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

        /// The complete trainer state at one save: weights + velocity, the
        /// trainer-file metadata, and the lineage record with the run's random
        /// streams. Resumable with `--resume-exact`; `training_step` is the
        /// trainer step (format v11) and the segment step is the record's
        /// `segment_local_step`, as in CorpusReplayRunner. The SGD loop awaits
        /// each step, so none is in flight when one is taken.
        struct TrainerSave {
            let snapshot: TrainerResumeSnapshot
            let metadata: ModelCheckpointMetadata
            let lineage: LineageRecord
            let savedAt: Date
            /// The training-health stamp of exactly this exported state
            /// (D2), taken right after the export.
            let healthStamp: TrainingHealthStamp
        }
        func trainerSave(step: Int, reason: String) async throws -> TrainerSave {
            let snapshot = try await trainer.exportResumeSnapshot()
            let healthStamp = trainingHealth.observationStamp()
            let streams = runSeed.runStreams(
                samplerState: buffer.samplerState(),
                dropoutStreamState: try await trainer.dropoutStreamState(),
                nextGameSerial: gameSerials.nextSerial, arenasStarted: nil,
                opponentGameIndices: driver.currentGameIndices())
            let metadata = ModelCheckpointMetadata.trainerFile(
                creator: ModelCheckpointMetadata.trainVsUciCreator,
                trainingStep: snapshot.schedule.completedTrainSteps,
                parentModelID: parentModelID,
                notes: "train-vs-uci \(reason) @ trainer step \(snapshot.schedule.completedTrainSteps) (segment step \(step))",
                schedule: snapshot.schedule,
                policyTailPrecision: trainer.policyTailPrecision,
                gradNormHistory: snapshot.gradNormHistory.history)
            // Games and plies the driver flushed into the buffer.
            let slots = driver.statsSnapshot()
            // Each pool's engine identity, from its first instance that has
            // completed a handshake (unrecorded until one has).
            for (instance, opponent) in opponents.enumerated() {
                if let identity = await opponent.arbiter.completedHandshakeIdentity() {
                    lineageTracker.noteEngineIdentity(
                        opponentIndex: opponentSpecIndices[instance],
                        LineageRecord.VsUciGeneration.EngineIdentity(idName: identity.idName, idAuthor: identity.idAuthor))
                }
            }
            let saveDate = Date()
            let lineage = try lineageTracker.record(
                at: saveDate,
                trainerCompletedSteps: snapshot.schedule.completedTrainSteps,
                segmentLocalStep: step,
                segmentGames: slots.reduce(0) { $0 + $1.gamesCompleted },
                segmentPositions: slots.reduce(0) { $0 + $1.pliesPlayed },
                corpus: nil,
                parameters: p.lineageParameters,
                rng: LineageRecord.RNG(
                    dropoutPhiloxState: snapshot.dropoutRNG.philoxState, streams: streams,
                    behaviorFingerprint: try await BehaviorFingerprint.compute(
                        for: .init(arch: arch, policyTailPrecision: trainer.policyTailPrecision))),
                inputs: lineageTracker.saveInputs(
                    scheduleAtSave: LRMomentumCycleReadout.scheduleAtSave(
                        inForce: p.parameters, completedTrainSteps: snapshot.schedule.completedTrainSteps),
                    replayRatioAtSave: nil,
                    // The run's one monitor is the segment's (a CLI segment
                    // is one process), so its summary is the segment's.
                    healthAlarms: trainingHealth.monitor.segmentSummary()))
            return TrainerSave(
                snapshot: snapshot, metadata: metadata, lineage: lineage, savedAt: saveDate, healthStamp: healthStamp)
        }

        /// Full layer health of a state just written — see CorpusReplayRunner's
        /// save. Never throws.
        func logLayerHealth(_ save: TrainerSave, step: Int, context: String) async {
            let health = await LayerHealthLog.checkpoint(
                arch: arch, trainerWeights: save.snapshot.trainerWeights, context: context,
                step: step, trainerStep: save.snapshot.schedule.completedTrainSteps)
            for line in health.lines { emit(line) }
            if let summary = health.summary {
                recorder?.appendLayerHealth(CliTrainingRecorder.LayerHealthRecord(
                    step: step, trainerStep: save.snapshot.schedule.completedTrainSteps,
                    context: context, summary: summary))
            }
            // The checkpoint evaluation (rules 1, 2, 3, 8) of the same pass.
            trainingHealth.evaluateCheckpoint(
                stamp: save.healthStamp, summary: health.summary,
                trainerStep: save.snapshot.schedule.completedTrainSteps)
        }

        // Consecutive-failure tracking per kind of save — see
        // `TrainerSaveFailureStreak` and `CorpusReplayRunner.reportSaveFailure`.
        var sessionSaveFailures = TrainerSaveFailureStreak(what: "session save")
        var enumeratedSaveFailures = TrainerSaveFailureStreak(what: "enumerated checkpoint save")
        // The periodic session cadence counts from the run start and from
        // each successful session save.
        var lastSessionSave = Date()

        /// Write one session folder through the GUI's session writer
        /// (`CheckpointManager.saveSession`): champion = the play network,
        /// synced from the trainer at this save; trainer = the complete
        /// trainer state; `session.json` = the run's settings and counters;
        /// the replay buffer only with `--save-replay-buffer`. A new folder
        /// every time. A failure is a warning the first time and stops the
        /// run when it repeats; disk full stops it at once.
        func saveSession(step: Int, kind: TrainVsUciSession.SaveKind) async throws {
            do {
                try await syncEvalNet()
                let save = try await trainerSave(step: step, reason: kind.diskTag)
                let championWeights = Array(save.snapshot.trainerWeights.prefix(baseCount))
                let createdAt = Int64(save.savedAt.timeIntervalSince1970)
                let bufferForSave: ReplayBuffer? = config.saveReplayBuffer ? buffer : nil
                let state = TrainVsUciSession.sessionState(
                    sessionID: config.runModelID,
                    savedAt: save.savedAt,
                    runStart: Date(timeIntervalSinceReferenceDate: runStart),
                    trainerCompletedSteps: save.snapshot.schedule.completedTrainSteps,
                    trainedPositions: try trainedPositions.positions(atSteps: save.snapshot.schedule.completedTrainSteps),
                    parameters: p.parameters,
                    hyperparameters: hp,
                    arch: arch,
                    bufferSnapshot: bufferForSave?.stateSnapshot(),
                    maxPliesPerGame: config.maxPliesPerGame)
                let url = try await CheckpointManager.saveSession(
                    championWeights: championWeights,
                    championID: config.runModelID,
                    // The play network synced from the trainer at this save:
                    // the same trainer step as the trainer file.
                    championMetadata: ModelCheckpointMetadata(
                        creator: ModelCheckpointMetadata.trainVsUciCreator,
                        trainingStep: save.snapshot.schedule.completedTrainSteps,
                        parentModelID: parentModelID,
                        notes: "train-vs-uci session (\(kind.diskTag)): play network synced from the trainer @ trainer "
                            + "step \(save.snapshot.schedule.completedTrainSteps) (segment step \(step))"),
                    championCreatedAtUnix: createdAt,
                    trainerWeights: save.snapshot.trainerWeights,
                    trainerID: config.runModelID,
                    trainerMetadata: save.metadata,
                    trainerCreatedAtUnix: createdAt,
                    state: state,
                    lineage: save.lineage,
                    // The play network was just synced from the trainer, so
                    // it holds exactly the weights the run's record
                    // describes; a model file carries no trainer state.
                    championLineage: try save.lineage.withoutTrainerState(),
                    architecture: arch,
                    replayBuffer: bufferForSave,
                    chartSnapshot: nil,
                    trigger: kind.diskTag,
                    at: save.savedAt,
                    sessionsDirectory: config.sessionDirectory)
                sessionSaveFailures.recordSuccess()
                lastSessionSave = Date()
                recorder?.recordSave(of: save.lineage, savedAt: SessionCheckpointLayout.trainerURL(in: url), log: emit)
                emit("[CHECKPOINT] Saved session (\(kind.diskTag)): \(url.lastPathComponent) "
                    + "step=\(step) trainerStep=\(save.snapshot.schedule.completedTrainSteps) "
                    + "build=\(BuildInfo.buildNumber) git=\(BuildInfo.gitHash)"
                    + SessionController.savedReplayBufferLogFields(writtenBuffer: bufferForSave))
                await logLayerHealth(save, step: step, context: "vsuci-session-\(kind.rawValue)")
            } catch {
                try CorpusReplayRunner.reportSaveFailure(error, step: step, what: "session save (\(kind.diskTag))")
                try sessionSaveFailures.recordFailure(step: step)
            }
        }

        /// Write the step-enumerated checkpoint of the trainer's current state
        /// (the trainer file alone, named by its trainer step) when
        /// `--enumerate-checkpoints` is on and the segment trained at least
        /// one step. `step` is the segment step, for the record and logs.
        func writeEnumeratedCheckpoint(step: Int, reason: String) async throws {
            guard let enumeratedWriter else { return }
            guard TrainerOutputFileGuard.enumeratedCopyIsWritten(
                trainerStep: trainer.completedTrainSteps, segmentStartTrainerStep: segmentStartTrainerStep) else {
                emit("[VS-UCI] enumerated checkpoint not written: the segment trained no step "
                    + "(trainerStep=\(trainer.completedTrainSteps), the start state)")
                return
            }
            do {
                let save = try await trainerSave(step: step, reason: reason)
                let encoded = try SafetensorsModelIO.encode(
                    modelID: config.runModelID,
                    createdAtUnix: Int64(save.savedAt.timeIntervalSince1970),
                    metadata: save.metadata,
                    weights: save.snapshot.trainerWeights,
                    architecture: arch,
                    includesVelocity: true,
                    lineage: save.lineage)
                let savedTrainerStep = save.snapshot.schedule.completedTrainSteps
                let written = try enumeratedWriter.write(encoded, trainerStep: savedTrainerStep)
                enumeratedSaveFailures.recordSuccess()
                let note = written.outcome == .replacedThisRunsEarlierSave
                    ? " (replaced this run's own earlier save of trainer step \(savedTrainerStep))"
                    : ""
                emit("[VS-UCI] enumerated checkpoint -> \(written.url.lastPathComponent)\(note)")
                await logLayerHealth(save, step: step, context: "vsuci-\(reason)")
            } catch let collision as TrainerOutputFileError {
                throw collision
            } catch let ownershipRefusal as FileSafetyError where ownershipRefusal.isOwnershipRefusal {
                throw ownershipRefusal
            } catch {
                try CorpusReplayRunner.reportSaveFailure(error, step: step, what: "enumerated checkpoint")
                try enumeratedSaveFailures.recordFailure(step: step)
            }
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
        // The training-health stop the loop honoured (R3), or nil.
        var healthStop: TrainingHealthEvent? = nil
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
            // Step lines follow the shared trainer-step schedule
            // (`TrainingStepLineSchedule`); its time rule reads a monotonic
            // clock started here.
            var stepLines = TrainingStepLineSchedule()
            // Each step line's `gNormMax=` / `clips=` cover the steps since
            // the previous line, starting from the clock the run resumed at.
            var gradientCapWindow = GradientCapStepLineWindow(startTrainerStep: trainer.completedTrainSteps)
            let lineClock = ContinuousClock()
            let lineClockStart = lineClock.now
            while true {
                if abort.isRequested { aborted = true; emit("[VS-UCI] abort requested — stopping at step \(step)"); break }
                // A training-health alarm whose action stops the run: stop
                // before another step (see CorpusReplayRunner).
                if let stopLine = trainingHealth.stopLine(segmentStep: step) {
                    healthStop = trainingHealth.requestedStop
                    emit(stopLine)
                    break
                }
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
                lineageTracker.recordTrainingStep(totalMs: timing.totalMs)
                step += 1

                // Keep the play network ~live.
                if step % syncEvery == 0 { try await syncEvalNet() }

                // One trainer-step observation for this step: the line's
                // cadence, its LR / momentum / cycle values, and the save
                // point all read it.
                let observedSteps = trainer.completedTrainSteps
                // Training health: record the step, and on a live-evaluation
                // step (every 50 trainer steps) take its stamp before any read.
                trainingHealth.record(timing, trainerStep: observedSteps)
                // Every clip (and every would-be clip in log-only mode) is one
                // `[GRAD-CLIP]` line — see CorpusReplayRunner.
                if let clipLine = RelativeGradientCapLogFormat.eventLine(
                    trainerStep: observedSteps, preClipNorm: timing.gradGlobalNorm, decision: timing.gradientCap,
                    learningRate: Double(trainer.effectiveLearningRate(forBatchSize: batchSize, completedSteps: observedSteps - 1))) {
                    emit(clipLine)
                }
                let healthStamp = trainingHealth.liveEvaluationStamp(trainerStep: observedSteps)
                // The step line's live read, when the line falls on this
                // step: it also serves this step's live evaluation.
                var lineLiveRead: LayerHealthLog.LiveOutcome? = nil
                if stepLines.lineDue(trainerStep: observedSteps,
                                     elapsedSec: TrainingStepLineSchedule.seconds(lineClock.now - lineClockStart),
                                     carriesDiagnostics: timing.hasDiagnostics,
                                     intervalSec: p.parameters.stepLineIntervalSec) != nil {
                    let liveLR = trainer.effectiveLearningRate(forBatchSize: batchSize, completedSteps: observedSteps)
                    let liveMomentum = trainer.effectiveMomentum(completedSteps: observedSteps)
                    let cycleValues = trainer.lrMomentumCycleValues(completedSteps: observedSteps)
                    let gradientCapReading = gradientCapWindow.take(
                        history: try await trainer.exportGradNormHistory(), throughTrainerStep: observedSteps,
                        fedCap: timing.gradientCap.fedCap)
                    let line = "[VS-UCI] step=\(step)"
                        + String(format: " loss=%.4f pLoss=%.4f vLoss=%.4f", timing.loss, timing.policyLoss, timing.valueLoss)
                        + " pEnt=\(dg(timing.policyEntropy, 3)) playedP=\(dg(timing.playedMoveProb, 3))"
                        + " pLogitMean=\(dg(timing.policyLogitMean, 4)) vLogitMean=\(dg(timing.valueLogitMean, 4))"
                        + String(format: " gNorm=%.3f lr=%.3g ms=%.1f", timing.gradGlobalNorm, liveLR, timing.totalMs)
                        + " buf=\(buffer.count)"
                        + String(format: " mom=%.4f", liveMomentum)
                        + (cycleValues.learningRate != nil ? " lrCyc" + LRMomentumCycleLogFormat.envelopeBounds(cycleValues) : "")
                        + gradientCapReading.logFields
                        + " trainerStep=\(observedSteps)"
                    emit(line)
                    // This step's own batch statistics ride the step line — to the session
                    // log only, as the trainer wrote them before (a 72 KB JSON line
                    // does not belong on the console).
                    if let batchStatsLine = BatchStatsLogLine.line(summary: trainer.lastBatchStatsSummary,
                                                                   ofTrainerStep: observedSteps) {
                        SessionLogger.shared.log(batchStatsLine)
                    }
                    // Live layer health at the same cadence (BN state +
                    // ReZero α, read on the trainer's queue between steps).
                    let liveRead = await LayerHealthLog.live(trainer: trainer)
                    for healthLine in liveRead.lines {
                        emit(healthLine)
                    }
                    lineLiveRead = liveRead
                    // Same cadence as the log line, so results.json and the log
                    // describe the same ticks.
                    // One snapshot, reused: `statsSnapshot()` is a lock-guarded
                    // COW array read, but taking it once keeps every derived
                    // field describing the same instant.
                    let slots = driver.statsSnapshot()
                    var statsRow = CliTrainingRecorder.StatsLine(
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
                        replayRatioTarget: nil,
                        lineageTotals: lineageTracker.totals(
                            trainerCompletedSteps: observedSteps,
                            segmentGames: slots.reduce(0) { $0 + $1.gamesCompleted })
                    )
                    statsRow.recordGradientCap(gradientCapReading, configuration: hp.relativeGradientCap)
                    recorder?.appendStats(statsRow)
                }
                // The live training-health evaluation, every 50 trainer
                // steps: after the step-line block, before the save block
                // (R0).
                if let healthStamp {
                    await trainingHealth.evaluateLive(
                        stamp: healthStamp, sharedRead: lineLiveRead, trainer: trainer, trainerStep: observedSteps,
                        learningRate: trainer.effectiveLearningRate(forBatchSize: batchSize, completedSteps: observedSteps),
                        momentum: trainer.effectiveMomentum(completedSteps: observedSteps))
                }
                // At every trainer-step multiple of 1000, after the step line
                // (so the line and its live readout precede the save).
                if TrainingStepLineSchedule.isCheckpointStep(trainerStep: observedSteps) {
                    try await writeEnumeratedCheckpoint(step: step, reason: "autosave")
                }
                if TrainVsUciSession.periodicSaveIsDue(now: Date(), lastSave: lastSessionSave,
                                                       intervalSec: periodicSessionIntervalSec) {
                    try await saveSession(step: step, kind: .periodic)
                }
                // Rule 3's value-FC1 velocity at most 1,000 trainer steps
                // apart (D6): covered by the enumerated checkpoints' passes
                // with `--enumerate-checkpoints`, a dedicated read otherwise.
                await trainingHealth.valueFC1ReadIfDue(trainer: trainer, trainerStep: observedSteps)
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

        // The final session (its save syncs the play network first), then the
        // final step's enumerated checkpoint.
        let finalKind: TrainVsUciSession.SaveKind
        if aborted {
            finalKind = .abort
        } else if healthStop != nil {
            finalKind = .healthStop
        } else {
            finalKind = .final
        }
        // The last partial window is judged before the final save's pass
        // (R0); a stop it or the final pass requests changes nothing now.
        await trainingHealth.evaluateBeforeFinalSave(trainer: trainer, batchSize: batchSize)
        try await saveSession(step: step, kind: finalKind)
        // The final trainer step's enumerated copy, unless the autosave at a
        // checkpoint step already wrote it (a segment that trained no step
        // writes none: `writeEnumeratedCheckpoint` logs that).
        if !TrainingStepLineSchedule.isCheckpointStep(trainerStep: trainer.completedTrainSteps) {
            try await writeEnumeratedCheckpoint(step: step, reason: finalKind.rawValue)
        }
        // The session folder is the end state: the run fails without it,
        // even when the step-enumerated copy above was written.
        try sessionSaveFailures.requireLastSaveSucceeded(step: step, reason: finalKind.diskTag)
        // The final `[HEALTH] check` line, after the final saves' checkpoint
        // evaluations.
        trainingHealth.finish()

        // `results.json` last, after the final model save — a run that dies
        // saving weights should not also claim a clean results record.
        if let recorder, let output = config.output {
            // Reported from the cause the loop actually exited on, not inferred
            // from which limits were configured: a run with BOTH a step limit
            // and a time limit can exit on either, and inferring would mislabel
            // a timeout as `stepLimitReached`.
            let reason: CliTrainingRecorder.TerminationReason
            if aborted {
                reason = .manualStop
            } else if healthStop != nil {
                reason = .trainingHealthAlarm
            } else if timedOut {
                reason = .timerExpired
            } else {
                reason = .stepLimitReached
            }
            recorder.setTerminationReason(reason)
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
        return Result(steps: step, gamesCompleted: totalGames, healthStop: healthStop)
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
