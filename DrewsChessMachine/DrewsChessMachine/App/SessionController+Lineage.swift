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
        /// parent a run from the champion branches from.
        case file(LineageTracker.ParentFile)
    }

    enum LineageSegmentError: LocalizedError {
        case noSegment(String)
        case noRunSeed
        case noChampionOrigin

        var errorDescription: String? {
            switch self {
            case .noSegment(let what):
                return "No lineage segment is running for \(what)."
            case .noRunSeed:
                return "The run's master seed was not resolved before its lineage segment began."
            case .noChampionOrigin:
                return "The champion's weights have no recorded origin (built here or loaded from a file)."
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
        case .file(let source): return source.trainerCompletedSteps
        case nil: throw LineageSegmentError.noChampionOrigin
        }
    }

    /// A run from the champion's weights: fresh when the champion was built
    /// in this process, a branch from the file it was loaded from otherwise.
    private func championLineageStart() throws -> LineageTracker.Start {
        switch championOrigin {
        case .built(let initialization): return .fresh(initialization: initialization)
        case .file(let source): return .branch(parent: source)
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
    /// trainer's segment is put back as it was, since a failed `[RUN]`
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
            try beginLineageSegment(mode: mode, trainer: trainer, resumed: pendingLoadedSession,
                                    continuedRunStreams: continuedRunStreams,
                                    replayBufferRestored: replayBufferRestored,
                                    behaviorFingerprint: fingerprint)
            guard let tracker = lineageTracker else {
                throw LineageSegmentError.noSegment("Play and Train")
            }
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
    func beginLineageSegment(mode: TrainingStartMode, trainer: ChessTrainer, resumed: LoadedSession?,
                             continuedRunStreams: LineageRecord.RunStreams?, replayBufferRestored: Bool,
                             behaviorFingerprint: BehaviorFingerprint.Record) throws {
        let counts = parallelWorkerStatsBox?.snapshot()
        let start: LineageTracker.Start?
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
        case .newSessionResetTrainerFromChampion:
            start = try championLineageStart()
        case .freshOrFromLoadedSession:
            if let resumed {
                let gaps = Self.guiResumeGaps(
                    resumed: resumed, continuedRunStreams: continuedRunStreams,
                    replayBufferRestored: replayBufferRestored,
                    runningPolicyTailPrecision: trainer.policyTailPrecision,
                    runningBuild: .current, runningDevice: .current, runningFingerprint: behaviorFingerprint)
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
            } else {
                start = try championLineageStart()
            }
        }
        if let start {
            if case .resume = start {} else {
                checkpoint?.runResumeExactness = nil
            }
            lineageTracker = try LineageTracker(
                start: start, pathKind: .gui, argv: CommandLine.arguments,
                startedAt: Date(), segmentStartTrainerStep: trainer.completedTrainSteps)
            lineageFedCarry = LineageFedCarry()
        }
        lineageFedCarry.baselineGames = counts?.emittedGames
        lineageFedCarry.baselinePositions = counts?.emittedPositions
        // The one [RUN] line of this Play-and-Train start: the segment as it
        // stands now (a continued segment reports its totals so far).
        guard let seed = runRandomSeed else { throw LineageSegmentError.noRunSeed }
        SessionLogger.shared.log(RunProvenanceLine.line(
            record: try lineageRecordForSave(at: Date(), trainerCompletedSteps: trainer.completedTrainSteps,
                                             dropoutPhiloxState: nil, dropoutStreamState: nil),
            seed: seed))
    }

    /// The running segment's lineage now, for `results.json`: its record as
    /// of this moment (no trainer snapshot behind it, so no dropout state,
    /// and no saved file) and the per-row totals it implies.
    func lineageForResults(trainerCompletedSteps: Int) throws -> (totals: LineageTracker.Totals, record: LineageRecord) {
        let record = try lineageRecordForSave(at: Date(), trainerCompletedSteps: trainerCompletedSteps,
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

    /// The record for a save of the running segment's state, with the
    /// trainer clock `trainerCompletedSteps` the saved trainer state carries
    /// and the dropout state and dropout-stream position captured with it
    /// (both nil when the save has no trainer snapshot). A save with a
    /// trainer snapshot also records the run's streams: its seed, the replay
    /// buffer's sampler, the next self-play game serial and the arena count.
    func lineageRecordForSave(at date: Date, trainerCompletedSteps: Int,
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
            parameters: try LineageRecord.Parameters(values: TrainingParameters.shared.snapshot().rawValueMap()),
            rng: LineageRecord.RNG(dropoutPhiloxState: dropoutPhiloxState,
                                   streams: try runStreamsForSave(dropoutStreamState: dropoutStreamState),
                                   behaviorFingerprint: try behaviorFingerprintForSave(dropoutStreamState: dropoutStreamState)))
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
        case .file(let source):
            switch source.lineage {
            case .recorded(let record):
                return record.withoutTrainerState()
            case .unrecorded:
                return LineageTracker.untrainedCopyRecord(source: source, derivation: nil, pathKind: .gui,
                                                          argv: CommandLine.arguments, at: date)
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
            let champion = record.withoutTrainerState()
            championOrigin = .file(LineageTracker.ParentFile(
                modelID: championID.description, contentSHA256: nil,
                trainerCompletedSteps: trainerCompletedSteps, lineage: .recorded(champion),
                derivationHistory: champion.derivationHistory))
        case .failure(let error):
            SessionLogger.shared.log("[LINEAGE] promoted champion's lineage could not be recorded (\(error.localizedDescription)): its origin is cleared, so a later save or branch from it fails")
            championOrigin = nil
        }
    }
}
