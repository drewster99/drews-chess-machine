import Foundation
import CryptoKit

/// How `--train-vs-uci` saves and resumes: as `.dcmsession` folders written by
/// the same `CheckpointManager.saveSession` the GUI uses (exclusive staging,
/// bit-exact model verification, forward-pass round trip, `session.json`
/// round trip, replay-buffer round trip, F_FULLFSYNC, publish without
/// overwrite). Every save is a new folder; nothing is ever replaced.
///
/// A train-vs-UCI session holds:
/// - `trainer.safetensors` — the complete trainer state (fp32 masters,
///   optimizer velocity, schedule, lineage record with the run's random
///   streams), exactly what the run's rolling model file held before session
///   folders replaced it;
/// - `champion.safetensors` — the play network's weights, synced from the
///   trainer at the save, so the two describe the same instant;
/// - `session.json` — the run's settings and counters (the GUI's session
///   schema; the self-play and arena fields record the configured
///   parameters, which this path does not use), carrying the same lineage
///   record, whose `path_kind` is `vsuci`;
/// - `replay_buffer.bin` — only with `--save-replay-buffer` (plan D-8).
///
/// Resume: `--start-model <folder> --resume-exact` restores the trainer state,
/// the run's streams and, when the folder has one, the replay buffer — the
/// one thing a resume from a model file can never restore. The GUI refuses a
/// train-vs-UCI session (`SessionCheckpointState.guiLoadRefusal`): its state
/// has no self-play or arena run to continue.
enum TrainVsUciSession {

    /// Why a session save was written; its tag ends the folder name, so a
    /// train-vs-UCI save is never mistaken for a GUI save (and never matches
    /// the automatic-save retention pool, which counts only GUI `periodic` and
    /// `promote` folders).
    enum SaveKind: String, CaseIterable, Sendable {
        /// The `periodic_autosave_interval_sec` cadence elapsed.
        case periodic
        /// The run reached its step or time limit.
        case final
        /// The run stopped on Ctrl-C.
        case abort

        var diskTag: String { "vsuci-\(rawValue)" }
    }

    /// What `--start-model` names for a train-vs-UCI run.
    enum StartSource: Equatable, Sendable {
        /// A model file (a corpus-replay or step-enumerated checkpoint, a GUI
        /// trainer file, a fresh `--new-model`). An exact resume from one
        /// cannot restore a replay buffer.
        case modelFile(URL)
        /// A `.dcmsession` folder (its `session.json` is there).
        case session(URL)
    }

    enum StartSourceError: LocalizedError, Equatable {
        case missing(path: String)
        case folderWithoutSessionJSON(path: String)
        case notAFileOrFolder(path: String, kind: String)

        var errorDescription: String? {
            switch self {
            case .missing(let path):
                return "--start-model \(path) does not exist"
            case .folderWithoutSessionJSON(let path):
                return "--start-model \(path) is a folder without \(SessionCheckpointLayout.stateFilename) — "
                    + "name a .dcmsession folder or a model file"
            case .notAFileOrFolder(let path, let kind):
                return "--start-model \(path) is a \(kind), not a model file or a session folder"
            }
        }
    }

    /// Classify `--start-model`. A symbolic link is followed (the start is
    /// only read, never written), so a link to a session or model works; a
    /// path that resolves to anything else is refused by name.
    static func startSource(path: String) throws -> StartSource {
        let url = URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory) else {
            throw StartSourceError.missing(path: url.path)
        }
        if isDirectory.boolValue {
            let stateURL = SessionCheckpointLayout.stateURL(in: url)
            var stateIsDirectory: ObjCBool = false
            guard FileManager.default.fileExists(atPath: stateURL.path, isDirectory: &stateIsDirectory),
                  !stateIsDirectory.boolValue else {
                throw StartSourceError.folderWithoutSessionJSON(path: url.path)
            }
            return .session(url)
        }
        // `attributesOfItem` does not follow a final symbolic link, so it is
        // asked about what the path resolves to; the path given is what is
        // returned, so step files are named next to it.
        let attributes = try FileManager.default.attributesOfItem(atPath: url.resolvingSymlinksInPath().path)
        guard let type = attributes[.type] as? FileAttributeType else {
            throw StartSourceError.notAFileOrFolder(path: url.path, kind: "item of unknown type")
        }
        guard type == .typeRegular else {
            throw StartSourceError.notAFileOrFolder(path: url.path, kind: type.rawValue)
        }
        return .modelFile(url)
    }

    /// The `session.json` state of one train-vs-UCI save. Settings come from
    /// the run's parameter snapshot — the same values the lineage record's
    /// parameter snapshot carries — and the trainer-level values the run
    /// actually trained with (`hyperparameters`), so a field means the same
    /// thing it means in a GUI session. This path plays no self-play and runs
    /// no arenas: its self-play counters are zero and its arena history is
    /// empty.
    ///
    /// `bufferSnapshot` is nil for a save without the replay buffer;
    /// `CheckpointManager.saveSession` replaces the buffer counters with the
    /// snapshot the buffer write itself took.
    ///
    /// `maxPliesPerGame` is the run's own game cap (`--max-plies`,
    /// `TrainVsUciConfig.maxPliesPerGame`), which every game against the
    /// engines was played to. The self-play cap in the parameter snapshot is
    /// a setting this path never reads.
    ///
    /// `trainedPositions` is the run's `trainingPositionsSeen` over
    /// `trainerCompletedSteps` (the lifetime trainer clock): its positions
    /// count (`trainedPositionsCount(startTrainerSteps:startSession:batchSize:)`)
    /// at that clock, nil when steps before the run are unrecorded.
    static func sessionState(
        sessionID: String,
        savedAt: Date,
        runStart: Date,
        trainerCompletedSteps: Int,
        trainedPositions: Int?,
        parameters p: TrainingParametersSnapshot,
        hyperparameters hp: TrainerHyperparameters,
        arch: NetworkArchitecture,
        bufferSnapshot: ReplayBuffer.StateSnapshot?,
        maxPliesPerGame: Int
    ) -> SessionCheckpointState {
        SessionCheckpointState(
            formatVersion: SessionCheckpointState.currentFormatVersion,
            sessionID: sessionID,
            savedAtUnix: Int64(savedAt.timeIntervalSince1970),
            sessionStartUnix: Int64(runStart.timeIntervalSince1970),
            elapsedTrainingSec: max(0, savedAt.timeIntervalSince(runStart)),
            trainingSteps: trainerCompletedSteps,
            selfPlayGames: 0,
            selfPlayMoves: 0,
            trainingPositionsSeen: trainedPositions,
            batchSize: p.trainingBatchSize,
            learningRate: hp.learningRate,
            entropyRegularizationCoeff: hp.entropyRegularizationCoeff,
            drawPenalty: hp.drawPenalty,
            promoteThreshold: p.arenaPromoteThreshold,
            arenaGames: p.arenaGamesPerTournament,
            arenaConcurrency: p.arenaConcurrency,
            selfPlayTau: TauConfigCodable(SamplingSchedule(
                startTau: Float(p.selfPlayStartTau),
                decayPerPly: Float(p.selfPlayTauDecayPerPly),
                floorTau: Float(p.selfPlayTargetTau))),
            arenaTau: TauConfigCodable(SamplingSchedule(
                startTau: Float(p.arenaStartTau),
                decayPerPly: Float(p.arenaTauDecayPerPly),
                floorTau: Float(p.arenaTargetTau))),
            selfPlayWorkerCount: p.selfPlayConcurrency,
            gradClipMaxNorm: hp.gradClipMaxNorm,
            weightDecayCoeff: hp.weightDecayC,
            dropoutRate: hp.dropoutRate,
            policyLossWeight: hp.policyLossWeight,
            valueLossWeight: hp.valueLossWeight,
            momentumCoeff: hp.momentumCoeff,
            illegalMassPenaltyWeight: hp.illegalMassPenaltyWeight,
            policyLabelSmoothingEpsilon: hp.policyLabelSmoothingEpsilon,
            policyLabelSmoothingMode: hp.policyLabelSmoothingMode.logToken,
            policyLabelSmoothingPerMove: hp.policyLabelSmoothingPerMove,
            policyLabelSmoothingPerMoveCap: hp.policyLabelSmoothingPerMoveCap,
            valueLabelSmoothingEpsilon: hp.valueLabelSmoothingEpsilon,
            replayRatioTarget: p.replayRatioTarget,
            replayRatioAutoAdjust: p.replayRatioAutoAdjust,
            stepDelayMs: p.trainingStepDelayMs,
            selfPlayDelayMs: p.selfPlayDelayMs,
            lrWarmupSteps: hp.lrWarmupSteps,
            sqrtBatchScalingForLR: hp.sqrtBatchScalingForLR,
            signedAdvantageComplementCE: hp.useSignedAdvantageComplementCE,
            replayBufferMinPositionsBeforeTraining: p.replayBufferMinPositionsBeforeTraining,
            arenaAutoIntervalSec: p.arenaAutoIntervalSec,
            candidateProbeIntervalSec: p.candidateProbeIntervalSec,
            legalMassCollapseThreshold: p.legalMassCollapseThreshold,
            legalMassCollapseGraceSeconds: p.legalMassCollapseGraceSeconds,
            legalMassCollapseNoImprovementProbes: p.legalMassCollapseNoImprovementProbes,
            batchStatsInterval: hp.batchStatsInterval,
            klProbeInterval: hp.klProbeInterval,
            periodicAutosaveIntervalSec: p.periodicAutosaveIntervalSec,
            maxPeriodicAutosavesKept: p.maxPeriodicAutosavesKept,
            automaticSavePruningEnabled: p.automaticSavePruningEnabled,
            sessionSaveIncludeReplayBuffer: p.sessionSaveIncludeReplayBuffer,
            arenaPromotionCriterion: p.arenaPromotionCriterion.logToken,
            arenaSPRTElo0: p.arenaSPRTElo0,
            arenaSPRTElo1: p.arenaSPRTElo1,
            arenaSPRTAlpha: p.arenaSPRTAlpha,
            arenaSPRTBeta: p.arenaSPRTBeta,
            arenaSPRTMinGames: p.arenaSPRTMinGames,
            arenaSPRTMaxGames: p.arenaSPRTMaxGames,
            recordSelfPlayGames: p.recordSelfPlayGames,
            lrMomentumCycle: hp.lrMomentumCycle,
            lrMomentumCycleEnvelope: hp.lrMomentumCycle.envelope,
            maxPliesFromAnyOneGame: p.maxPliesFromAnyOneGame,
            targetSampledGameLengthPlies: p.targetSampledGameLengthPlies,
            maxDrawPercentPerBatch: p.maxDrawPercentPerBatch,
            replayBufferStratifyByMaterial: p.replayBufferStratifyByMaterial,
            selfPlayDrawKeepFraction: p.selfPlayDrawKeepFraction,
            maxPliesPerGame: maxPliesPerGame,
            drawWatchPDrawThreshold: p.drawWatchPDrawThreshold,
            drawWatchTerminateGames: p.drawWatchTerminateGames,
            drawWatchStreakLength: p.drawWatchStreakLength,
            buildNumber: BuildInfo.buildNumber,
            buildGitHash: BuildInfo.gitHash,
            buildGitBranch: BuildInfo.gitBranch,
            buildDate: BuildInfo.buildDate,
            buildTimestamp: BuildInfo.buildTimestamp,
            buildGitDirty: BuildInfo.gitDirty,
            hasReplayBuffer: bufferSnapshot != nil,
            replayBufferStoredCount: bufferSnapshot?.storedCount,
            replayBufferCapacity: bufferSnapshot?.capacity,
            replayBufferTotalPositionsAdded: bufferSnapshot?.totalPositionsAdded,
            championID: sessionID,
            trainerID: sessionID,
            arenaHistory: []
        )
        .withArchitecture(ArchitectureMetadata(describing: arch))
    }

    /// The positions count of a run whose trainer starts at clock
    /// `startTrainerSteps` and trains at `batchSize` (`TrainedPositionsCount`,
    /// on the trainer-clock axis session.json's `trainingSteps` uses here).
    /// The steps before the run are counted from what records them:
    /// - none (a fresh trainer, or a new branch, whose clock starts at 0): 0;
    /// - a session folder whose session.json covers exactly those steps
    ///   (its step count is the clock the run starts from): the positions it
    ///   recorded, nil if it recorded none;
    /// - anything else (a model file, or a GUI session counted from a "New
    ///   Session, keep trainer" start): unrecorded — never this run's batch
    ///   size times the earlier steps, which may have trained at another.
    static func trainedPositionsCount(startTrainerSteps: Int, startSession: SessionCheckpointState?,
                                      batchSize: Int) -> TrainedPositionsCount {
        let before: Int?
        if startTrainerSteps == 0 {
            before = 0
        } else if let startSession, startSession.trainingSteps == startTrainerSteps {
            before = startSession.trainingPositionsSeen
        } else {
            before = nil
        }
        return TrainedPositionsCount(stepsAtStart: startTrainerSteps, positionsBeforeStart: before, batchSize: batchSize)
    }

    /// The run's game generation as its lineage configuration records it
    /// (plan B1, gap 11): the ply cap, the eval-sync cadence, the trainer's
    /// move selection, and each opponent pool with the SHA-256 of the
    /// executable its command names (read once, here) and its options
    /// (values redacted like argv). Engine identities start unrecorded and
    /// are noted after a handshake (`LineageTracker.noteEngineIdentity`).
    static func lineageGeneration(config: TrainVsUciConfig,
                                  trainerMoveSelection: SamplingSchedule) throws -> LineageRecord.VsUciGeneration {
        let opponents = try config.opponents.map { spec -> LineageRecord.VsUciGeneration.Opponent in
            let executable = URL(fileURLWithPath: (spec.command as NSString).expandingTildeInPath)
            let bytes = try Data(contentsOf: executable)
            let digest = SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()
            return LineageRecord.VsUciGeneration.Opponent(
                command: spec.command,
                executableSHA256: digest,
                count: spec.count,
                goLimit: spec.goLimit,
                options: spec.options.map { option in
                    LineageRecord.VsUciGeneration.Option(
                        name: option.name,
                        value: LineageRecord.isSecretOptionName(option.name) ? LineageRecord.redactedValue : option.value)
                },
                identity: .unrecorded)
        }
        return LineageRecord.VsUciGeneration(
            maxPliesPerGame: config.maxPliesPerGame, evalSyncEverySteps: config.evalSyncEverySteps,
            trainerMoveSelection: LineageRecord.MoveSelection(trainerMoveSelection), opponents: opponents)
    }

    /// Lower-cased extensions that make a `--checkpoint-stem` name a model
    /// file or a session folder rather than a stem.
    static let modelOrSessionExtensions: Set<String> = ["safetensors", "dcmmodel", "dcmsession"]

    enum CheckpointStemError: LocalizedError, Equatable {
        case namedLikeAnEnumeratedCheckpoint(stem: String, step: Int)
        case hasExtension(stem: String)

        var errorDescription: String? {
            switch self {
            case .namedLikeAnEnumeratedCheckpoint(let stem, let step):
                return "--checkpoint-stem \(stem) is itself named like the step-\(step) checkpoint of another stem; "
                    + "choose a stem that is not a checkpoint name"
            case .hasExtension(let stem):
                return "--checkpoint-stem \(stem) names a model file or session; give the stem without that extension "
                    + "(step files are <stem>-vsuci-step<N>.safetensors)"
            }
        }
    }

    /// The base `EnumeratedCheckpointNaming` derives the step-file names
    /// from: `<dir>/<stem>-vsuci-latest.safetensors`, whose `-vsuci-latest`
    /// marker each step name replaces with `-vsuci-step<N>`. The base itself
    /// is never written. The stem is `--checkpoint-stem` when given;
    /// otherwise the start model file's own stem next to it (the names runs
    /// have always produced), or the run's model ID in `Models/` for a fresh
    /// run or a session start.
    ///
    /// A stem is a name, and names may hold dots (`sf100-lr0.5`); a stem is
    /// refused as naming a file only when its extension is a model or
    /// session one (`modelOrSessionExtensions`, any letter case).
    static func enumeratedNamingBase(
        checkpointStem: String?, startSource: StartSource?, runModelID: String
    ) throws -> URL {
        let marker = "-\(EnumeratedCheckpointNaming.trainVsUciRunTag)-latest.safetensors"
        if let checkpointStem {
            let url = URL(fileURLWithPath: (checkpointStem as NSString).expandingTildeInPath)
            guard !Self.modelOrSessionExtensions.contains(url.pathExtension.lowercased()) else {
                throw CheckpointStemError.hasExtension(stem: url.path)
            }
            if let step = EnumeratedCheckpointNaming.step(
                ofEnumeratedFileNameUnderAnyStem: url.lastPathComponent + ".safetensors") {
                throw CheckpointStemError.namedLikeAnEnumeratedCheckpoint(stem: url.path, step: step)
            }
            return url.deletingLastPathComponent().appendingPathComponent(url.lastPathComponent + marker)
        }
        if case .modelFile(let url)? = startSource {
            let stem = url.deletingPathExtension().lastPathComponent
            return url.deletingLastPathComponent().appendingPathComponent(stem + marker)
        }
        return CheckpointPaths.modelsDir.appendingPathComponent(runModelID + marker)
    }

    /// The `[VS-UCI] session saves: …` launch line: where sessions go, when,
    /// and whether they carry the replay buffer — so a run's crash exposure
    /// and disk cost are stated up front.
    static func launchLine(directory: URL, periodicIntervalSec: Double, includesReplayBuffer: Bool) -> String {
        let cadence = String(format: "every %.0fs (periodic_autosave_interval_sec) and at the end", periodicIntervalSec)
        let buffer = includesReplayBuffer
            ? "buffer=included (--save-replay-buffer)"
            : "buffer=omitted (pass --save-replay-buffer to include it; a resume then refills from new games, NOT EXACT: buffer)"
        return "[VS-UCI] session saves: \(directory.path) \(cadence); \(buffer)"
    }

    /// Whether a periodic session save is due: at least `intervalSec` has
    /// passed since the last successful session save (or the run start).
    /// Pure, so the cadence is unit-testable.
    static func periodicSaveIsDue(now: Date, lastSave: Date, intervalSec: Double) -> Bool {
        now.timeIntervalSince(lastSave) >= intervalSec
    }
}

extension SessionCheckpointState {
    /// Why the GUI must not load this session, or nil when it may. A
    /// train-vs-UCI session (lineage `path_kind` `vsuci`) records a run with
    /// no self-play or arena state for the GUI to continue; it resumes only
    /// under `--train-vs-uci --start-model <folder>`.
    var guiLoadRefusal: String? {
        guard lineage?.invocation.pathKind == .vsuci else { return nil }
        return "this session was written by --train-vs-uci; resume it with "
            + "--train-vs-uci … --start-model <this folder> --resume-exact"
    }
}
