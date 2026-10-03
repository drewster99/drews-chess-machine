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
                    notExactItems: LineageTracker.NotExactItem.guiResume + [LineageTracker.NotExactItem.buffer],
                    legacyTotals: nil)
            }
        case .newSessionResetTrainerFromChampion:
            start = championLineageStart
        case .freshOrFromLoadedSession:
            if let resumed {
                var gaps = LineageTracker.NotExactItem.resumeGaps(
                    LineageTracker.NotExactItem.guiResume,
                    restoring: DropoutRNGResumeState(lineage: resumed.trainerFile.safetensorsProvenance?.lineage))
                if resumed.replayBufferURL == nil {
                    gaps.append(LineageTracker.NotExactItem.buffer)
                }
                // A session written before lineage still recorded its
                // elapsed time; that is its one usable total.
                let legacyTotals: LineageTracker.LegacySessionTotals?
                if resumed.state.lineage == nil {
                    legacyTotals = LineageTracker.LegacySessionTotals(wallSec: resumed.state.elapsedTrainingSec)
                } else {
                    legacyTotals = nil
                }
                start = .resume(parent: resumed.trainerFile.lineageParent,
                                notExactItems: gaps, legacyTotals: legacyTotals)
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
                                             dropoutPhiloxState: nil),
            seed: seed))
    }

    /// The running segment's lineage now, for `results.json`: its record as
    /// of this moment (no trainer snapshot behind it, so no dropout state,
    /// and no saved file) and the per-row totals it implies.
    func lineageForResults(trainerCompletedSteps: Int) throws -> (totals: LineageTracker.Totals, record: LineageRecord) {
        let record = try lineageRecordForSave(at: Date(), trainerCompletedSteps: trainerCompletedSteps,
                                              dropoutPhiloxState: nil)
        return (LineageTracker.Totals(of: record), record)
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
    /// and the dropout state captured with it (nil when the save has no
    /// trainer snapshot).
    func lineageRecordForSave(at date: Date, trainerCompletedSteps: Int,
                              dropoutPhiloxState: DropoutPhiloxState?) throws -> LineageRecord {
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
            dropoutPhiloxState: dropoutPhiloxState)
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
                                            dropoutPhiloxState: nil)
        }
        if let source = championLineageSource {
            return LineageTracker.untrainedCopyRecord(source: source, derivation: nil, pathKind: .gui,
                                                      argv: CommandLine.arguments, at: date)
        }
        return try LineageTracker.mintRecord(pathKind: .gui, argv: CommandLine.arguments, at: date)
    }
}
