import Foundation

/// GUI side of the file lineage (`LineageRecord`, `LineageTracker`).
///
/// A GUI lineage segment spans one trainer's continuous training in this
/// process. It begins at Play-and-Train start (fresh, a branch from the
/// champion's source file, or a session's run continued), continues across
/// Stop / Continue on the same trainer, and every session save and model
/// save takes its record from it.
///
/// Games and positions come from the self-play stats box's emitted
/// counters (what entered the replay buffer). That box is replaced or torn
/// down across Stop and new-session starts, so what each box counted is
/// folded into `lineageFedCarry` before it goes, and a new box's counts at
/// the segment's (re)start become the baseline: the segment's fed totals are
/// `carry + (current − baseline)`, one number however many boxes the segment
/// lived through.
extension SessionController {

    /// Fed counts this lineage segment has accumulated across stats boxes.
    struct LineageFedCarry: Equatable, Sendable {
        var games: Int = 0
        var positions: Int = 0
        /// The live box's emitted counts when the segment (re)started
        /// counting on it; nil when no box is counting for the segment.
        var baselineGames: Int?
        var baselinePositions: Int?
    }

    /// Where the champion's current weights came from.
    enum ChampionOrigin: Sendable {
        /// Built in this process from fresh weights drawn under
        /// `initialization`.
        case built(initialization: ModelInitRecord)
        /// Loaded from a file (or promoted with a run's record): the lineage
        /// parent a run from the champion branches from, and how its weights
        /// reached this process (B9).
        case file(LineageTracker.ParentFile, startWeights: ChampionStartWeights)
    }

    /// How a file champion's weights reached this process — what a run that
    /// branches from it records as `start_value_head_recentered` (B9). The
    /// champion's start weights came from its load, not from anything at the
    /// segment's start.
    enum ChampionStartWeights: Sendable {
        /// Loaded (Load Model, Load Session) with this value-head centering.
        case loaded(ValueHeadCentering)
        /// Promoted from the trainer: never read from a file.
        case notLoaded
    }

    enum LineageSegmentError: LocalizedError {
        case noSegment(String)
        case noRunSeed
        case noChampionOrigin
        /// A Play-and-Train start reached its lineage segment without the
        /// inputs its start records (run seed kind, replay-ratio start).
        case noSegmentStartInputs
        /// The trainer's clock moved between a save's configuration cut and
        /// its export, which training's pause should make impossible.
        case configurationCutClockMoved(cut: Int, exported: Int)
        /// The Dirichlet noise self-play uses is not configured.
        case noSelfPlayDirichlet

        var errorDescription: String? {
            switch self {
            case .noSegment(let what):
                return "No lineage segment is running for \(what)."
            case .noRunSeed:
                return "The run's master seed was not resolved before its lineage segment began."
            case .noChampionOrigin:
                return "The champion's weights have no recorded origin (built here or loaded from a file)."
            case .noSegmentStartInputs:
                return "The Play-and-Train start's seed kind or replay-ratio start was not set before its lineage segment began."
            case .configurationCutClockMoved(let cut, let exported):
                return "The trainer's clock moved between the save's configuration cut (\(cut)) and its export (\(exported))."
            case .noSelfPlayDirichlet:
                return "Self-play's Dirichlet noise is not configured, so the segment cannot record it."
            }
        }
    }

    /// The `training_step` a champion model file states: the trainer step
    /// the champion's weights were taken at, from where they came — the
    /// step their source file (or promotion) stated, nil when it stated
    /// none, and 0 for weights built in this process. Never the trainer's
    /// step: the champion holds weights from an earlier point than the
    /// trainer whenever they differ, and a model file that claims the
    /// trainer's step would be read as trained when its weights are fresh.
    static func championFileTrainingStep(origin: ChampionOrigin?) throws -> Int? {
        switch origin {
        case .built: return 0
        case .file(let source, _): return source.trainerCompletedSteps
        case nil: throw LineageSegmentError.noChampionOrigin
        }
    }

    /// A run from the champion's weights: fresh when the champion was built
    /// in this process, a branch from the file it was loaded from otherwise.
    private func championLineageStart() throws -> LineageTracker.Start {
        switch championOrigin {
        case .built(let initialization): return .fresh(initialization: initialization)
        case .file(let source, _): return .branch(parent: source)
        case nil: throw LineageSegmentError.noChampionOrigin
        }
    }

    /// The lineage part of a Play-and-Train start, run on the main actor
    /// once the trainer holds its starting state: stamp the trainer's ID for
    /// `mode`, record the run's behavior fingerprint, begin (or continue) the
    /// lineage segment, and — only when all of that succeeded — consume the
    /// pending loaded session, which the running session now owns.
    ///
    /// Trainer ID by mode, matching how the start set the trainer's weights:
    /// - `.freshOrFromLoadedSession`: the loaded session's trainer file's ID
    ///   when resuming one, else a new generation off the champion;
    /// - `.newSessionResetTrainerFromChampion`: a new generation off the
    ///   champion (the trainer was just forked from it);
    /// - `.continueAfterStop`, `.newSessionKeepTrainer`: kept — its weights
    ///   were not touched, so the lineage is continuous.
    ///
    /// A failure leaves things as the next start can use them. A loaded
    /// session stays pending, so the next start redoes the whole resume from
    /// it (consuming it here would have left the restored trainer to a
    /// "Continue" that re-records the run as unrecorded history). A kept
    /// trainer's segment is put back as it was, with the run-start
    /// parameter capture, positions count and replay buffer it was trained
    /// with, since a failed `[RUN]`
    /// record must not replace or reset it. A trainer this start already
    /// reset leaves no segment behind: the old one no longer describes it.
    func beginRunLineage(mode: TrainingStartMode, trainer: ChessTrainer, championIdentifier: ModelID?,
                         continuedRunStreams: LineageRecord.RunStreams?, replayBufferRestored: Bool,
                         fingerprintResult: Result<BehaviorFingerprint.Record, Error>) -> Result<LineageTracker, Error> {
        let previousTracker = lineageTracker
        let previousCarry = lineageFedCarry
        let previousExactness = checkpoint?.runResumeExactness
        do {
            switch mode {
            case .continueAfterStop, .newSessionKeepTrainer:
                break
            case .newSessionResetTrainerFromChampion:
                trainer.identifier = ModelIDMinter.mintTrainerGeneration(from: try Self.requiredChampionID(championIdentifier))
            case .freshOrFromLoadedSession:
                if let resumed = pendingLoadedSession {
                    // A GUI resume never refuses (determinism plan D-1);
                    // what it could not restore — the policy-tail precision
                    // included — is named on the one `[RESUME]` line
                    // `beginLineageSegment` logs.
                    trainer.identifier = ModelID(value: resumed.trainerFile.modelID)
                } else {
                    trainer.identifier = ModelIDMinter.mintTrainerGeneration(from: try Self.requiredChampionID(championIdentifier))
                }
            }
            let fingerprint = try fingerprintResult.get()
            SessionLogger.shared.log("[RUN] behavior fingerprint recipe=\(fingerprint.recipe) sha256=\(fingerprint.sha256)")
            runBehaviorFingerprint = fingerprint
            try beginLineageSegment(mode: mode, trainer: trainer, championIdentifier: championIdentifier,
                                    resumed: pendingLoadedSession,
                                    continuedRunStreams: continuedRunStreams,
                                    replayBufferRestored: replayBufferRestored,
                                    behaviorFingerprint: fingerprint)
            guard let tracker = lineageTracker else {
                throw LineageSegmentError.noSegment("Play and Train")
            }
            // The trainer holds this start's starting state now — the clock
            // the segment just read — which the positions count needs to
            // tell whether the session's steps are the trainer's clock.
            anchorRunTrainedPositions(atTrainerStep: trainer.completedTrainSteps)
            // Consume the pending load — from here on, the running session
            // owns the restored state.
            pendingLoadedSession = nil
            pendingLoadedSessionAcceptedReplacements = []
            // A training segment has started, so clear the "champion
            // replaced since last training" flag (the Start dialog's
            // annotation is resolved).
            championLoadedSinceLastTrainingSegment = false
            return .success(tracker)
        } catch {
            switch mode {
            case .continueAfterStop, .newSessionKeepTrainer:
                lineageTracker = previousTracker
                lineageFedCarry = previousCarry
                checkpoint?.runResumeExactness = previousExactness
                restoreRunStartCaptureAfterFailedStart()
            case .freshOrFromLoadedSession, .newSessionResetTrainerFromChampion:
                lineageTracker = nil
                lineageFedCarry = LineageFedCarry()
                checkpoint?.runResumeExactness = nil
            }
            return .failure(error)
        }
    }

    /// The champion's ID, which a trainer generation is minted from; a
    /// champion without one is an error, never a freshly minted stand-in.
    private static func requiredChampionID(_ identifier: ModelID?) throws -> ModelID {
        guard let identifier else {
            throw LineageTracker.TrackerError.noModelID(what: "the champion")
        }
        return identifier
    }

    /// Begin (or continue) the lineage segment for a Play-and-Train start.
    /// Call once the trainer holds its starting state and the run's stats
    /// box exists; `resumed` is the loaded session being resumed, if any,
    /// with what the resume restored: `continuedRunStreams` (the saved run's
    /// streams it continues, nil when it drew a new seed) and
    /// `replayBufferRestored` (`guiResumeGaps`). Both are ignored when
    /// `resumed` is nil.
    func beginLineageSegment(mode: TrainingStartMode, trainer: ChessTrainer, championIdentifier: ModelID?,
                             resumed: LoadedSession?,
                             continuedRunStreams: LineageRecord.RunStreams?, replayBufferRestored: Bool,
                             behaviorFingerprint: BehaviorFingerprint.Record) throws {
        let counts = parallelWorkerStatsBox?.snapshot()
        let start: LineageTracker.Start?
        // Whether the weights a new segment starts from were value-head
        // recentered at load (B9); unused when the segment continues.
        let startValueHeadRecentered: LineageRecord.Recorded<Bool>
        switch mode {
        case .continueAfterStop, .newSessionKeepTrainer:
            if lineageTracker != nil {
                // The same trainer trains on: its segment continues.
                start = nil
            } else {
                // A kept trainer whose history this process never tracked
                // (a model was loaded since): its weights continue, with the
                // history before them unrecorded.
                SessionLogger.shared.log("[LINEAGE] continuing a trainer with no tracked lineage: a new run begins, earlier history unrecorded")
                // No file states this trainer's history: none is carried,
                // and `continues_unrecorded_history` says the run's earlier
                // history is unrecorded.
                start = .resume(
                    parent: try LineageTracker.ParentFile.untrackedTrainer(
                        identifier: trainer.identifier, completedSteps: trainer.completedTrainSteps),
                    gaps: [.rngSampler, .serials, .buffer, .clocks],
                    legacyTotals: nil)
            }
            // A continued segment keeps its value; a kept trainer with no
            // tracked lineage was loaded in a way this process cannot state.
            startValueHeadRecentered = .unrecorded
        case .newSessionResetTrainerFromChampion:
            start = try championLineageStart()
            startValueHeadRecentered = .recorded(try championStartWeightsRecentered())
        case .freshOrFromLoadedSession:
            if let resumed {
                let gaps = Self.guiResumeGaps(
                    resumed: resumed, continuedRunStreams: continuedRunStreams,
                    replayBufferRestored: replayBufferRestored,
                    runningPolicyTailPrecision: trainer.policyTailPrecision,
                    runningBuild: try .current, runningDevice: .current, runningFingerprint: behaviorFingerprint)
                    // A history-less trainer file is a gap only when this run
                    // clips with the relative cap.
                    + ResumeGap.gradNormHistoryGaps(
                        restoring: GradNormHistoryResumeState(resumed.trainerFile.metadata.trainerGradNormHistory),
                        runningMode: try trainer.relativeGradientCap.validated().mode)
                let exactness = ResumeExactness.resume(of: resumed.trainerFile.lineageParent, gaps: gaps)
                SessionLogger.shared.log(exactness.logLine)
                checkpoint?.runResumeExactness = exactness
                // A session written before lineage still recorded its
                // elapsed time; that is its one usable total.
                let legacyTotals: LineageTracker.LegacySessionTotals?
                if resumed.state.lineage == nil {
                    legacyTotals = LineageTracker.LegacySessionTotals(wallSec: resumed.state.elapsedTrainingSec)
                } else {
                    legacyTotals = nil
                }
                start = .resume(parent: resumed.trainerFile.lineageParent,
                                gaps: gaps, legacyTotals: legacyTotals)
                startValueHeadRecentered = .recorded(try LineageTracker.startValueHeadRecentered(
                    resumed.trainerFile.valueHeadCentering, file: resumed.trainerFile.modelID))
            } else {
                start = try championLineageStart()
                startValueHeadRecentered = .recorded(try championStartWeightsRecentered())
            }
        }
        guard let seed = runRandomSeed else { throw LineageSegmentError.noRunSeed }
        let tracker: LineageTracker
        if let start {
            if case .resume = start {} else {
                checkpoint?.runResumeExactness = nil
            }
            tracker = try LineageTracker(
                start: start, pathKind: .gui, argv: CommandLine.arguments,
                startedAt: Date(), segmentStartTrainerStep: trainer.completedTrainSteps)
            try noteSegmentStart(on: tracker, isNewSegment: true, trainer: trainer, championIdentifier: championIdentifier,
                                 startValueHeadRecentered: startValueHeadRecentered, seed: seed)
            lineageTracker = tracker
            lineageFedCarry = LineageFedCarry()
        } else {
            guard let kept = lineageTracker else { throw LineageSegmentError.noSegment("Play and Train") }
            tracker = kept
        }
        // A kept segment's journals get this start's entries; a start that
        // then fails takes them back out, since the start never happened.
        let journalsBeforeStart = tracker.checkpointJournals()
        do {
            if start == nil {
                try noteSegmentStart(on: tracker, isNewSegment: false, trainer: trainer,
                                     championIdentifier: championIdentifier,
                                     startValueHeadRecentered: startValueHeadRecentered, seed: seed)
            }
            // The one [RUN] line of this Play-and-Train start: the segment as
            // it stands now (a continued segment reports its totals so far).
            let runRecord = try lineageRecordForSave(at: Date(), cut: try takeConfigurationCut(trainer: trainer),
                                                     trainerCompletedSteps: trainer.completedTrainSteps,
                                                     dropoutPhiloxState: nil, dropoutStreamState: nil)
            SessionLogger.shared.log(RunProvenanceLine.line(record: runRecord, seed: seed))
        } catch {
            tracker.restoreJournals(journalsBeforeStart)
            throw error
        }
        installParameterChangeJournal(tracker: tracker, trainer: trainer)
        lineageFedCarry.baselineGames = counts?.emittedGames
        lineageFedCarry.baselinePositions = counts?.emittedPositions
    }

    /// Record a champion just loaded from `file` as the champion's origin,
    /// with the value-head centering its weights were decoded with (B9), and
    /// journal the change into a running segment (B5, review X10: a model
    /// loaded between Stop and Continue changes the data-generating champion
    /// of the segment a Continue keeps). A decoded file always carries its
    /// centering decision; one without it leaves no origin, logged, so a
    /// later branch or save from the champion fails loudly.
    func adoptLoadedChampionOrigin(_ file: ModelCheckpointFile) {
        guard let centering = file.valueHeadCentering else {
            SessionLogger.shared.log("[LINEAGE] loaded champion \(file.modelID) has no value-head centering decision: its origin is cleared")
            championOrigin = nil
            return
        }
        championOrigin = .file(file.lineageParent, startWeights: .loaded(centering))
        guard let trainer else { return }
        do {
            try noteChampionChange(trainerStep: trainer.completedTrainSteps, trigger: .loadedModel)
        } catch {
            SessionLogger.shared.log("[LINEAGE] the loaded champion could not be journalled: \(error.localizedDescription)")
        }
    }

    /// Whether the champion a new run starts from (fresh or branched) was
    /// value-head recentered when it was loaded (B9): a built champion and a
    /// promoted one were never read from a file.
    private func championStartWeightsRecentered() throws -> Bool {
        switch championOrigin {
        case .built: return false
        case .file(let source, let startWeights):
            switch startWeights {
            case .loaded(let centering):
                return try LineageTracker.startValueHeadRecentered(centering, file: source.modelID)
            case .notLoaded:
                return false
            }
        case nil: throw LineageSegmentError.noChampionOrigin
        }
    }

    /// What a Play-and-Train start contributes to its lineage segment, noted
    /// once the segment's tracker is selected and the trainer holds its
    /// starting state (its clock is the restored one), before the `[RUN]`
    /// record (gaps 9b, 3.5; B5, B6):
    /// - a new segment: its fixed configuration, the run's seed, the
    ///   replay-ratio start and the `segment_start` champion;
    /// - a kept segment (Continue after Stop, "New Session, keep trainer"):
    ///   the replay-ratio start, a seed entry when the start resolved a new
    ///   seed (keep-trainer; a Continue keeps the run's), and a
    ///   `parameter_changes` entry per captured key whose value this start
    ///   captured differently, at the start clock — no step is in flight at
    ///   a start, so the next step uses it.
    func noteSegmentStart(on tracker: LineageTracker, isNewSegment: Bool, trainer: ChessTrainer,
                          championIdentifier: ModelID?,
                          startValueHeadRecentered: LineageRecord.Recorded<Bool>, seed: RunRandomSeed) throws {
        guard let seedKind = runSeedStartKind, let ratioStart = replayRatioStart else {
            throw LineageSegmentError.noSegmentStartInputs
        }
        let clock = trainer.completedTrainSteps
        let now = Int64(Date().timeIntervalSince1970)
        if isNewSegment {
            guard let dirichlet = SamplingSchedule.selfPlay.dirichletNoise else {
                throw LineageSegmentError.noSelfPlayDirichlet
            }
            // The budget `--train` enforces; none for an interactive run.
            let budget: LineageRecord.Budget = autoTrainOnLaunch
                ? LineageRecord.Budget(trainingStepLimit: cliConfig?.trainingStepLimit,
                                       trainingTimeLimitSec: cliConfig?.trainingTimeLimitSec, epochLimit: nil)
                : .none
            try tracker.configureSegment(LineageTracker.SegmentConfiguration(
                policyTailPrecision: trainer.policyTailPrecision, budget: budget, vsuci: nil,
                selfPlayDirichlet: LineageRecord.Dirichlet(dirichlet),
                startValueHeadRecentered: startValueHeadRecentered))
            tracker.noteRunSeed(seed, atTrainerStep: clock)
            // The champion this start generates games with: the identifier
            // the start was given (the one its trainer ID is minted from).
            let championID = try Self.requiredChampionID(championIdentifier)
            tracker.noteChampionChange(LineageRecord.ChampionChange(
                trainerStep: clock, recordedUnix: now, championModelID: championID.description,
                championContentSHA256: try championContentSHA256(), trigger: .segmentStart))
        } else {
            if seedKind == .resolvedOrInherited {
                tracker.noteRunSeed(seed, atTrainerStep: clock)
            }
            if let previous = runStartStateReplacedByLatestStart?.capture, let current = runStartCapture {
                let pairs: [(String, Int, Int)] = [
                    (TrainingBatchSize.id, previous.trainingBatchSize, current.trainingBatchSize),
                    (ReplayBufferMinPositionsBeforeTraining.id, previous.replayBufferMinPositionsBeforeTraining,
                     current.replayBufferMinPositionsBeforeTraining),
                    (ReplayBufferCapacity.id, previous.replayBufferCapacity, current.replayBufferCapacity),
                ]
                for (id, old, new) in pairs where old != new {
                    tracker.journalParameterChange(LineageRecord.ParameterChange(
                        committedAtTrainerStep: clock, recordedUnix: now, id: id, old: .int(old), new: .int(new),
                        restampedFrom: nil))
                }
            }
        }
        tracker.noteReplayRatioStart(LineageRecord.ReplayRatio.Start(
            trainerStep: clock, autoAdjust: ratioStart.autoAdjust,
            initialTrainingStepDelayMs: ratioStart.delayMs, initialDelaySource: ratioStart.source))
    }

    /// The live champion's file content hash for a champion-change entry:
    /// the file it was loaded from, null for weights built or promoted here.
    func championContentSHA256() throws -> String? {
        switch championOrigin {
        case .built: return nil
        case .file(let source, _): return source.contentSHA256
        case nil: throw LineageSegmentError.noChampionOrigin
        }
    }

    /// Journal a change of the data-generating champion into the running
    /// segment (B5), at `trainerStep` — the clock the champion's games start
    /// feeding at. Called after the change's own records are built, so a
    /// champion's origin record never lists itself. With no segment there
    /// is nothing to journal into: the next segment's `segment_start` entry
    /// names the champion.
    func noteChampionChange(trainerStep: Int, trigger: LineageRecord.ChampionChange.Trigger) throws {
        guard let tracker = lineageTracker else { return }
        guard let championID = network?.identifier else {
            throw LineageTracker.TrackerError.noModelID(what: "the champion")
        }
        tracker.noteChampionChange(LineageRecord.ChampionChange(
            trainerStep: trainerStep, recordedUnix: Int64(Date().timeIntervalSince1970),
            championModelID: championID.description, championContentSHA256: try championContentSHA256(),
            trigger: trigger))
    }

    /// Point the settings' run-change observer at `tracker`'s journal (gap
    /// 4), reading the clock from `trainer`. Both are held weakly, so an
    /// observer left behind cannot keep a replaced trainer or segment alive;
    /// a later segment's start replaces it.
    func installParameterChangeJournal(tracker: LineageTracker, trainer: ChessTrainer) {
        TrainingParameters.runChangeObserver.value = { [weak tracker, weak trainer] id, old, new in
            guard let tracker, let trainer else { return }
            tracker.journalParameterChange(LineageRecord.ParameterChange(
                committedAtTrainerStep: trainer.completedTrainSteps, recordedUnix: Int64(Date().timeIntervalSince1970),
                id: id, old: old, new: new, restampedFrom: nil))
        }
    }

    /// What one GUI save (or promotion record, or `[RUN]` line) describes,
    /// taken in one main-actor turn with no suspension (review X4): the
    /// parameters in force, the schedule the trainer is running, and the
    /// journal positions and derived values at that instant. The popover's
    /// commits and their pushes onto the trainer run on the main actor, so
    /// none can fall inside the cut; one that lands during a save's later
    /// awaits is in neither the record's snapshot nor its journal — it is
    /// the next save's journal entry. The trainer file is written with the
    /// cut's schedule, so its flat `trainer_*` keys, the record's schedule
    /// keys and the journal describe one moment.
    struct GuiConfigurationCut: Sendable {
        let inForce: TrainingParametersSnapshot
        let schedule: TrainerScheduleState
        let saveInputs: LineageTracker.SaveInputs

        /// The record's parameters: the in-force snapshot with the cut's
        /// schedule adopted, without the seed settings.
        func lineageParameters() throws -> LineageRecord.Parameters {
            try LineageRecord.Parameters(values: inForce.adoptingSchedule(schedule).lineageValues())
        }
    }

    /// Take the configuration cut of the running segment. Callers hold the
    /// training pause (or call it where no step can run), so the trainer's
    /// clock read here is the one the save exports.
    func takeConfigurationCut(trainer: ChessTrainer) throws -> GuiConfigurationCut {
        guard let tracker = lineageTracker else {
            throw LineageSegmentError.noSegment("a configuration cut")
        }
        let inForce = try requiredRunStartCapture(for: "a configuration cut")
            .inForce(over: TrainingParameters.shared.snapshot())
        let schedule = TrainerScheduleState(currentlyRunningOn: trainer)
        let atSave = replayRatioSnapshot.map {
            LineageRecord.ReplayRatio.AtSave(autoAdjust: $0.autoAdjust, targetRatio: $0.targetRatio,
                                             currentRatio: $0.currentRatio, trainingStepDelayMs: $0.computedDelayMs,
                                             selfPlayDelayMs: $0.computedSelfPlayDelayMs)
        }
        return GuiConfigurationCut(
            inForce: inForce,
            schedule: schedule,
            saveInputs: tracker.saveInputs(
                scheduleAtSave: LRMomentumCycleReadout.scheduleAtSave(
                    inForce: inForce.adoptingSchedule(schedule), completedTrainSteps: schedule.completedTrainSteps),
                replayRatioAtSave: atSave,
                healthAlarms: tracker.healthAlarms(withLive: trainingHealthMonitor?.segmentSummary())))
    }

    /// The running segment's lineage now, for `results.json`: its record as
    /// of this moment (no trainer snapshot behind it, so no dropout state,
    /// and no saved file) and the per-row totals it implies.
    ///
    /// The record's clock is the one its configuration cut read, never a
    /// clock the caller read earlier: training runs while the results ticker
    /// calls this, and a settings change journalled between an earlier read
    /// and the cut would sit above that clock, which `LineageRecord`'s
    /// invariants refuse. Edits commit on the main actor, so none lands
    /// between the cut and the record built here.
    func lineageForResults() throws -> (totals: LineageTracker.Totals, record: LineageRecord) {
        guard let trainer else { throw LineageSegmentError.noSegment("the results record (no trainer)") }
        let cut = try takeConfigurationCut(trainer: trainer)
        let record = try lineageRecordForSave(at: Date(), cut: cut,
                                              trainerCompletedSteps: cut.schedule.completedTrainSteps,
                                              dropoutPhiloxState: nil, dropoutStreamState: nil)
        return (LineageTracker.Totals(of: record), record)
    }

    /// What a GUI resume of `resumed` does not restore (determinism plan C3,
    /// D-1: a GUI resume is state-exact at most, so it is reported, never
    /// refused). The periodic-save clock needs nothing: the save being
    /// resumed reset it. The arena clock is restored when the session
    /// recorded it. A session whose trainer file predates lineage also lacks
    /// `lineage`, which `ResumeExactness.resume(of:gaps:)` adds.
    ///
    /// The run's streams and the replay buffer are judged by what the
    /// resume actually restored, which the caller knows and passes in — not
    /// by what the session file contains: `continuedRunStreams` is the
    /// streams the run continues (nil when it drew a new seed, because the
    /// file has none, `--seed` named another seed, or they were named under
    /// another derivation), and `replayBufferRestored` whether the buffer
    /// file was restored into the run's buffer (false when the session has
    /// none or its restore failed).
    static func guiResumeGaps(resumed: LoadedSession,
                              continuedRunStreams: LineageRecord.RunStreams?,
                              replayBufferRestored: Bool,
                              runningPolicyTailPrecision: ChessNetwork.PolicyTailPrecision,
                              runningBuild: LineageRecord.Build,
                              runningDevice: LineageRecord.Device,
                              runningFingerprint: BehaviorFingerprint.Record) -> [ResumeGap] {
        let lineage = resumed.trainerFile.safetensorsProvenance?.lineage
        var gaps: [ResumeGap] = []
        if resumed.state.arenaSecondsSinceLastArena == nil {
            gaps.append(.clocks)
        }
        gaps += ResumeGap.dropoutGaps(restoring: DropoutRNGResumeState(lineage: lineage))
        gaps += PolicyTailPrecisionResume.gaps(
            saved: resumed.trainerFile.metadata.trainerPolicyTailPrecision, running: runningPolicyTailPrecision)
        if !replayBufferRestored {
            gaps.append(.buffer)
        }
        if continuedRunStreams == nil {
            gaps += [.rngSampler, .serials]
        }
        if let record = lineage?.record {
            if record.parameters == nil { gaps.append(.params) }
            let environment = ResumeGap.environmentGaps(writtenBy: record, runningBuild: runningBuild,
                                                        runningDevice: runningDevice, runningFingerprint: runningFingerprint)
            for line in environment.logLines { SessionLogger.shared.log(line) }
            gaps += environment.gaps
        }
        return gaps
    }

    /// The run streams a resumed session continues: its trainer file's
    /// record's, when they include the self-play serial and the arena count
    /// a GUI run needs; nil otherwise (a session saved before the record
    /// carried them).
    static func resumableRunStreams(of resumed: LoadedSession) -> LineageRecord.RunStreams? {
        guard let streams = resumed.trainerFile.safetensorsProvenance?.lineage.record?.rng.streams,
              streams.nextGameSerial != nil, streams.arenasStarted != nil else { return nil }
        return streams
    }

    /// The saved run's seed for a GUI resume, or nil — logged — when it
    /// cannot be continued (a `--seed` naming another seed, or streams named
    /// under another derivation); the resume then runs on a newly resolved
    /// seed, passes no continued streams to `beginLineageSegment`, and
    /// reports `rng_sampler` / `serials` NOT EXACT.
    static func inheritedRunSeed(streams: LineageRecord.RunStreams, commandLineSeed: UInt64?) -> RunRandomSeed? {
        do {
            return try RunRandomSeed.inherited(
                from: streams, configuredSeed: TrainingParameters.shared.randomSeed, commandLineSeed: commandLineSeed)
        } catch {
            SessionLogger.shared.log("[RESUME] rng: the saved run's seed is not continued: \(error.localizedDescription)")
            return nil
        }
    }

    /// Fold what the live stats box counted for the segment into the carry,
    /// before the box is torn down or replaced.
    func foldLineageFedCounts() {
        guard let counts = parallelWorkerStatsBox?.snapshot(),
              let baselineGames = lineageFedCarry.baselineGames,
              let baselinePositions = lineageFedCarry.baselinePositions else { return }
        lineageFedCarry.games += counts.emittedGames - baselineGames
        lineageFedCarry.positions += counts.emittedPositions - baselinePositions
        lineageFedCarry.baselineGames = nil
        lineageFedCarry.baselinePositions = nil
    }

    /// Reset the self-play game stats for a new champion (both promotion
    /// paths), banking what the lineage segment counted on the box in the
    /// same lock acquisition as the reset, so no game recorded around the
    /// reset is lost from the segment's fed totals, and counting on from
    /// zero. A nil baseline means no segment is counting on this box, so the
    /// carry is left as it is — not a fallback: there is nothing to bank.
    func resetSelfPlayGameStatsForNewChampion() {
        guard let box = parallelWorkerStatsBox else { return }
        let discarded = box.resetGameStatsReturningEmittedCounts()
        guard let baselineGames = lineageFedCarry.baselineGames,
              let baselinePositions = lineageFedCarry.baselinePositions else { return }
        lineageFedCarry.games += discarded.games - baselineGames
        lineageFedCarry.positions += discarded.positions - baselinePositions
        lineageFedCarry.baselineGames = 0
        lineageFedCarry.baselinePositions = 0
    }

    /// Capture the parameters this Play-and-Train start trains under — the
    /// batch size and pre-train fill from the settings, the capacity from
    /// the run's buffer (on a Continue, the reused one) — as the single
    /// source every reader of them uses until the next start (see
    /// `RunStartParameterCapture`). Called at every start, right after the
    /// buffer is chosen and before the lineage segment begins, whose `[RUN]`
    /// record describes the parameters in force. The positions count begins
    /// with it (`beginRunTrainedPositions`), at the capture's batch.
    ///
    /// Call it before the start installs `buffer` as `replayBuffer`: the
    /// capture, positions count and buffer it replaces are kept together,
    /// so a start that fails before its lineage segment begins puts all
    /// three back (`restoreRunStartCaptureAfterFailedStart`) — a new buffer
    /// left behind with the old capture would make session.json's buffer
    /// counters disagree with the record's capacity.
    @discardableResult
    func beginRunStartCapture(buffer: ReplayBuffer) -> RunStartParameterCapture {
        let params = TrainingParameters.shared
        let capture = RunStartParameterCapture(
            trainingBatchSize: params.trainingBatchSize,
            replayBufferMinPositionsBeforeTraining: params.replayBufferMinPositionsBeforeTraining,
            replayBufferCapacity: buffer.capacity)
        runStartStateReplacedByLatestStart = ReplacedRunStartState(
            capture: runStartCapture, trainedPositions: runTrainedPositions, replayBuffer: replayBuffer)
        runStartCapture = capture
        runTrainedPositions = beginRunTrainedPositions(batchSize: capture.trainingBatchSize)
        SessionLogger.shared.log(
            "[PARAM] run-start capture: \(TrainingBatchSize.id)=\(capture.trainingBatchSize) "
                + "\(ReplayBufferMinPositionsBeforeTraining.id)=\(capture.replayBufferMinPositionsBeforeTraining) "
                + "\(ReplayBufferCapacity.id)=\(capture.replayBufferCapacity) (in force until the next Play-and-Train start)")
        return capture
    }

    /// Put back the capture, positions count and replay buffer the latest
    /// start replaced, for a start that ends without beginning (or
    /// replacing) its lineage segment: the segment left in place — and any
    /// save of it before the next start — was trained under that capture,
    /// counted by that count, and holds that buffer, not the failed start's.
    func restoreRunStartCaptureAfterFailedStart() {
        guard let replaced = runStartStateReplacedByLatestStart else {
            SessionLogger.shared.log("[PARAM] error: a failed Play-and-Train start found no replaced run-start state to put back")
            return
        }
        runStartCapture = replaced.capture
        runTrainedPositions = replaced.trainedPositions
        replayBuffer = replaced.replayBuffer
        runStartStateReplacedByLatestStart = nil
    }

    /// The running (or last stopped) run's start-time capture. Asked for
    /// only where a run's state is being described; none there is a bug,
    /// reported as the missing segment it implies — never answered from the
    /// settings, which may hold an edit this run does not use.
    func requiredRunStartCapture(for what: String) throws -> RunStartParameterCapture {
        guard let runStartCapture else {
            throw LineageSegmentError.noSegment("\(what) (no run-start parameter capture)")
        }
        return runStartCapture
    }

    /// The record for a save of the running segment's state, with the
    /// trainer clock `trainerCompletedSteps` the saved trainer state carries
    /// and the dropout state and dropout-stream position captured with it
    /// (both nil when the save has no trainer snapshot). A save with a
    /// trainer snapshot also records the run's streams: its seed, the replay
    /// buffer's sampler, the next self-play game serial and the arena count.
    /// Its parameters, journals and derived values come from `cut`, the
    /// configuration cut the save (or promotion, or `[RUN]` line) took — never
    /// from the settings as they stand when the record is built.
    func lineageRecordForSave(at date: Date, cut: GuiConfigurationCut, trainerCompletedSteps: Int,
                              dropoutPhiloxState: DropoutPhiloxState?,
                              dropoutStreamState: DCMRandom?) throws -> LineageRecord {
        guard let tracker = lineageTracker else {
            throw LineageSegmentError.noSegment("this save")
        }
        var games = lineageFedCarry.games
        var positions = lineageFedCarry.positions
        if let counts = parallelWorkerStatsBox?.snapshot(),
           let baselineGames = lineageFedCarry.baselineGames,
           let baselinePositions = lineageFedCarry.baselinePositions {
            games += counts.emittedGames - baselineGames
            positions += counts.emittedPositions - baselinePositions
        }
        guard let segmentStart = tracker.segmentStartTrainerStep else {
            throw LineageSegmentError.noSegment("a save with a trainer clock")
        }
        return try tracker.record(
            at: date,
            trainerCompletedSteps: trainerCompletedSteps,
            segmentLocalStep: trainerCompletedSteps - segmentStart,
            segmentGames: games,
            segmentPositions: positions,
            corpus: nil,
            parameters: try cut.lineageParameters(),
            rng: LineageRecord.RNG(dropoutPhiloxState: dropoutPhiloxState,
                                   streams: try runStreamsForSave(dropoutStreamState: dropoutStreamState),
                                   behaviorFingerprint: try behaviorFingerprintForSave(dropoutStreamState: dropoutStreamState)),
            inputs: cut.saveInputs)
    }

    /// The behavior fingerprint for a save that carries trainer state (the
    /// run's, computed at its start); nil for one that does not.
    private func behaviorFingerprintForSave(dropoutStreamState: DCMRandom?) throws -> BehaviorFingerprint.Record? {
        guard dropoutStreamState != nil else { return nil }
        guard let fingerprint = runBehaviorFingerprint else {
            throw LineageSegmentError.noSegment("a save with trainer state before the run's behavior fingerprint")
        }
        return fingerprint
    }

    /// The run's streams for a save that carries trainer state; nil for one
    /// that does not (`dropoutStreamState` nil). A save with trainer state
    /// happens only inside a run, which always has its seed, game serials
    /// and buffer — their absence is an error, not an unrecorded value.
    private func runStreamsForSave(dropoutStreamState: DCMRandom?) throws -> LineageRecord.RunStreams? {
        guard let dropoutStreamState else { return nil }
        guard let runSeed = runRandomSeed, let serials = selfPlayGameSerials, let buffer = replayBuffer else {
            throw LineageSegmentError.noSegment("a save with trainer state outside a run")
        }
        return runSeed.runStreams(
            samplerState: buffer.samplerState(),
            dropoutStreamState: dropoutStreamState,
            nextGameSerial: serials.nextSerial,
            arenasStarted: arenasStartedThisRun,
            opponentGameIndices: nil)
    }

    /// The lineage record of a champion model file — Save Champion, and the
    /// champion file of every session save — from where the champion's
    /// weights came, never from the trainer's run at the save: the champion
    /// holds weights from an earlier point (its build, its load, the last
    /// promotion) whenever training has moved on, so the run's record would
    /// claim steps, games and time those weights never saw.
    ///
    /// - built here: a fresh mint record of its initialization;
    /// - from a file or a promotion whose record exists: that record as it
    ///   is (the weights are exactly the ones it describes, so a save does
    ///   not start a new run), without trainer state, which a model file
    ///   does not carry;
    /// - from a file written before lineage: an untrained copy of it, whose
    ///   totals stay unrecorded;
    /// - no recorded origin: an error, never a guess.
    static func championFileLineageRecord(origin: ChampionOrigin?, at date: Date) throws -> LineageRecord {
        switch origin {
        case .built(let initialization):
            return try LineageTracker.mintRecord(pathKind: .gui, argv: CommandLine.arguments,
                                                 initialization: initialization, at: date)
        case .file(let source, _):
            switch source.lineage {
            case .recorded(let record):
                return try record.withoutTrainerState()
            case .unrecorded:
                // An untrained copy changes no architecture (gap 1c).
                return try LineageTracker.untrainedCopyRecord(source: source, derivation: nil, sourceArchitecture: nil,
                                                              pathKind: .gui, argv: CommandLine.arguments, at: date)
            }
        case nil:
            throw LineageSegmentError.noChampionOrigin
        }
    }

    /// Record where a promoted champion's weights came from — the trainer's
    /// weights at `trainerCompletedSteps`, described by the run's `record`
    /// at that point — as the champion's origin, used by every later
    /// champion file and by a run that branches from the champion. Shared
    /// by arena promotion and Promote Trainee Now.
    ///
    /// The record is kept without trainer state (a champion file carries
    /// none, and a branch draws fresh random state). When the record could
    /// not be built, or the champion has no model ID, the origin is cleared
    /// and logged: a later save or branch then fails loudly instead of
    /// describing the promoted weights by the previous champion's origin or
    /// as unrecorded history.
    func recordPromotedChampionOrigin(championID: ModelID?, trainerCompletedSteps: Int,
                                      record: Result<LineageRecord, Error>) {
        guard let championID else {
            SessionLogger.shared.log("[LINEAGE] promoted champion has no model ID: its origin is cleared, so a later save or branch from it fails")
            championOrigin = nil
            return
        }
        switch record {
        case .success(let record):
            do {
                let champion = try record.withoutTrainerState()
                championOrigin = .file(LineageTracker.ParentFile(
                    modelID: championID.description, contentSHA256: nil,
                    trainerCompletedSteps: trainerCompletedSteps, lineage: .recorded(champion),
                    derivationHistory: champion.derivationHistory), startWeights: .notLoaded)
            } catch {
                SessionLogger.shared.log("[LINEAGE] promoted champion's lineage could not be recorded (\(error.localizedDescription)): its origin is cleared, so a later save or branch from it fails")
                championOrigin = nil
            }
        case .failure(let error):
            SessionLogger.shared.log("[LINEAGE] promoted champion's lineage could not be recorded (\(error.localizedDescription)): its origin is cleared, so a later save or branch from it fails")
            championOrigin = nil
        }
    }
}
