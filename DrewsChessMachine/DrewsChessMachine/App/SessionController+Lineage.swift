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

    enum LineageSegmentError: LocalizedError {
        case noSegment(String)
        case noRunSeed

        var errorDescription: String? {
            switch self {
            case .noSegment(let what):
                return "No lineage segment is running for \(what)."
            case .noRunSeed:
                return "The run's master seed was not resolved before its lineage segment began."
            }
        }
    }

    /// Where the champion's weights came from, as a parent for a branch:
    /// nil when the champion was built fresh in this process.
    private var championLineageStart: LineageTracker.Start {
        if let source = championLineageSource {
            return .branch(parent: source)
        }
        return .fresh
    }

    /// Begin (or continue) the lineage segment for a Play-and-Train start.
    /// Call once the trainer holds its starting state and the run's stats
    /// box exists; `resumed` is the loaded session being resumed, if any.
    func beginLineageSegment(mode: TrainingStartMode, trainer: ChessTrainer, resumed: LoadedSession?) throws {
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
                start = .resume(
                    parent: LineageTracker.ParentFile(
                        modelID: trainer.identifier?.description ?? "unknown",
                        contentSHA256: nil,
                        trainerCompletedSteps: trainer.completedTrainSteps,
                        lineage: .unrecorded(formatVersion: ArchitectureFormat.currentVersion),
                        // No file states this trainer's history: none is
                        // carried, and `continues_unrecorded_history` says
                        // the run's earlier history is unrecorded.
                        derivationHistory: []),
                    gaps: [.rngSampler, .serials, .buffer, .clocks],
                    legacyTotals: nil)
            }
        case .newSessionResetTrainerFromChampion:
            start = championLineageStart
        case .freshOrFromLoadedSession:
            if let resumed {
                let gaps = Self.guiResumeGaps(
                    resumed: resumed, runningPolicyTailPrecision: trainer.policyTailPrecision,
                    runningBuild: .current, runningDevice: .current)
                SessionLogger.shared.log(ResumeExactness.resume(of: resumed.trainerFile.lineageParent, gaps: gaps).logLine)
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
                start = championLineageStart
            }
        }
        if let start {
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
    static func guiResumeGaps(resumed: LoadedSession,
                              runningPolicyTailPrecision: ChessNetwork.PolicyTailPrecision,
                              runningBuild: LineageRecord.Build,
                              runningDevice: LineageRecord.Device) -> [ResumeGap] {
        let lineage = resumed.trainerFile.safetensorsProvenance?.lineage
        var gaps: [ResumeGap] = []
        if resumed.state.arenaSecondsSinceLastArena == nil {
            gaps.append(.clocks)
        }
        gaps += ResumeGap.dropoutGaps(restoring: DropoutRNGResumeState(lineage: lineage))
        gaps += PolicyTailPrecisionResume.gaps(
            saved: resumed.trainerFile.metadata.trainerPolicyTailPrecision, running: runningPolicyTailPrecision)
        if resumed.replayBufferURL == nil {
            gaps.append(.buffer)
        }
        if resumableRunStreams(of: resumed) == nil {
            gaps += [.rngSampler, .serials]
        }
        if let record = lineage?.record {
            if record.parameters == nil { gaps.append(.params) }
            gaps += ResumeGap.environmentGaps(writtenBy: record, runningBuild: runningBuild, runningDevice: runningDevice)
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
    /// seed and reports `rng_sampler` / `serials` NOT EXACT.
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

    /// Start counting on the live stats box from its current counts (after
    /// `foldLineageFedCounts()` banked what it counted before a reset).
    func rebaselineLineageFedCounts() {
        let counts = parallelWorkerStatsBox?.snapshot()
        lineageFedCarry.baselineGames = counts?.emittedGames
        lineageFedCarry.baselinePositions = counts?.emittedPositions
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
            rng: LineageRecord.RNG(dropoutPhiloxState: dropoutPhiloxState, streams: try runStreamsForSave(dropoutStreamState: dropoutStreamState)))
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
            arenasStarted: arenasStartedThisRun)
    }

    /// The record for a model-only save of the champion (Save Champion):
    /// the running segment's state when one exists; otherwise the
    /// champion's own origin — a fresh mint, or an untrained copy of the
    /// file it was loaded from.
    func lineageRecordForChampionSave(at date: Date) throws -> LineageRecord {
        if lineageTracker != nil, let trainer {
            // A champion-only save carries no trainer state, so no dropout
            // state either.
            return try lineageRecordForSave(at: date, trainerCompletedSteps: trainer.completedTrainSteps,
                                            dropoutPhiloxState: nil, dropoutStreamState: nil)
        }
        if let source = championLineageSource {
            return LineageTracker.untrainedCopyRecord(source: source, derivation: nil, pathKind: .gui,
                                                      argv: CommandLine.arguments, at: date)
        }
        return try LineageTracker.mintRecord(pathKind: .gui, argv: CommandLine.arguments, at: date)
    }
}
