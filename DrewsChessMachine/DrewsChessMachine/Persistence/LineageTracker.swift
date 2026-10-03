//
//  LineageTracker.swift
//  DrewsChessMachine
//
//  The one builder of `LineageRecord`s. Each training path (corpus replay,
//  train-vs-UCI, the GUI session) owns one tracker per segment — a segment is
//  one process's continuous training of one trainer — and asks it for a
//  record at every save. The tracker holds the run identity and the totals
//  the segment started from, and accumulates what the segment itself adds:
//  measured trainer-step time. Games, positions and the trainer clock are
//  owned by the path's own counters, which the caller hands in at record
//  time, so there is one source for each number.
//
//  Totals continue across segments: a segment that continues a run starts
//  from the parent record's totals, so a chain of exact resumes reports the
//  same totals an uninterrupted run would (to the measurement), and the
//  dashboards no longer need hand-entered bases.
//

import Foundation

/// Builds the `LineageRecord` for each save of one segment.
final class LineageTracker: @unchecked Sendable {

    /// The file a segment trains from, as identified by what the file says.
    struct ParentFile: Equatable, Sendable {
        let modelID: String
        let contentSHA256: String?
        let trainerCompletedSteps: Int?
        let lineage: LineageRecord.Presence
        /// The derivations the file's weights went through
        /// (`derivationHistory(lineage:metadata:)`), which a child carries
        /// on verbatim.
        let derivationHistory: [ModelDerivation.DerivationRecord]

        /// A file's derivation history: its lineage record's when it has
        /// one; for a file written before lineage, the `derivation_history`
        /// key it states (`--derive-model` wrote it), or none when it states
        /// none. A key that is present but unreadable is an error, never
        /// read as "not derived".
        static func derivationHistory(lineage: LineageRecord.Presence,
                                      metadata: [String: String]) throws -> [ModelDerivation.DerivationRecord] {
            switch lineage {
            case .recorded(let record):
                return record.derivationHistory
            case .unrecorded:
                return try ModelDerivation.decodeHistory(metadata[ModelDerivation.derivationHistoryKey])
            }
        }

        /// The parent as recorded in a child's record.
        var recordParent: LineageRecord.Parent {
            LineageRecord.Parent(
                modelID: modelID,
                contentSHA256: contentSHA256,
                trainerCompletedSteps: trainerCompletedSteps,
                lineageRunID: lineage.record?.run.lineageRunID,
                segmentID: lineage.record?.run.segmentID
            )
        }
    }

    /// The total a continued run had reached before this segment that a GUI
    /// session saved before session files carried a lineage did record: its
    /// elapsed session time. Its game and position counters restart at
    /// every promotion, so they are not run totals and are not used; games,
    /// positions and measured trainer-step time stay unrecorded.
    struct LegacySessionTotals: Equatable, Sendable {
        let wallSec: Double
    }

    /// How the segment begins.
    enum Start: Sendable {
        /// Weights initialized by this run, drawn under `initialization`.
        case fresh(initialization: ModelInitRecord)
        /// Train from `parent`'s weights with a fresh trainer clock: a new
        /// run that records its parent.
        case branch(parent: ParentFile)
        /// Continue `parent`'s run by exact resume. `gaps` names whatever the
        /// resume could not restore (empty for a complete restore; `lineage`
        /// is added here when the parent carries no record).
        /// `legacyTotals` supplies the totals of a GUI session written before
        /// lineage existed; it must be nil when the parent carries a lineage.
        case resume(parent: ParentFile, gaps: [ResumeGap], legacyTotals: LegacySessionTotals?)
    }

    enum TrackerError: Error, CustomStringConvertible {
        case legacyTotalsWithRecordedLineage(parentModelID: String)
        case negativeSegmentCount(what: String, value: Int)

        var description: String {
            switch self {
            case .legacyTotalsWithRecordedLineage(let id):
                return "lineage: parent \(id) carries a lineage record, so legacy session totals must not be supplied"
            case .negativeSegmentCount(let what, let value):
                return "lineage: segment \(what) is negative (\(value))"
            }
        }
    }

    let pathKind: LineageRecord.PathKind
    let argv: [String]
    private let run: (lineageRunID: String, segmentIndex: Int, segmentID: String,
                      start: LineageRecord.SegmentStart, exactResume: Bool,
                      notExactItems: [String], continuesUnrecordedHistory: Bool)
    private let parent: LineageRecord.Parent?
    private let segments: [LineageRecord.SegmentSummary]
    /// How the run's starting weights were drawn, recorded in every record
    /// (`rng.init_seed` / `init_scheme`); nil for a run that started from a
    /// file's weights.
    private let initialization: ModelInitRecord?
    /// Carried verbatim into every record of the segment (determinism plan
    /// B4): the parent's history on a branch or resume, none on a fresh run.
    private let derivationHistory: [ModelDerivation.DerivationRecord]
    private let startedAt: Date
    /// The trainer clock when the segment began; nil for a writer with no
    /// trainer.
    let segmentStartTrainerStep: Int?
    /// Totals before this segment; nil where no predecessor recorded one.
    private let baseGames: Int?
    private let basePositions: Int?
    private let baseTrainStepSec: Double?
    private let baseWallSec: Double?
    /// Measured trainer-step seconds this segment.
    private let segmentTrainStepSec = SyncBox<Double>(0)

    /// The segment index a run records, decided by how it starts: an exact
    /// resume of a run that carries a lineage record continues that run as
    /// its next segment; anything else — a fresh run, a branch, or a resume
    /// of a file written before lineage existed — begins a run, at segment 0.
    /// The one rule for it: the tracker below and the step-checkpoint names
    /// (`EnumeratedCheckpointNaming`) both take it from here, so a segment's
    /// step files always carry the index its records do.
    static func segmentIndex(exactResumeOf parent: ParentFile?) -> Int {
        guard let parent, case .recorded(let record) = parent.lineage else { return 0 }
        return record.run.segmentIndex + 1
    }

    /// Start a segment. `segmentStartTrainerStep` is the trainer clock when
    /// the segment begins (after any exact restore), or nil when the writer
    /// has no trainer.
    init(start: Start,
         pathKind: LineageRecord.PathKind,
         argv: [String],
         startedAt: Date,
         segmentStartTrainerStep: Int?) throws {
        self.pathKind = pathKind
        self.argv = LineageRecord.redactedArguments(argv)
        self.startedAt = startedAt
        self.segmentStartTrainerStep = segmentStartTrainerStep
        let segmentID = UUID().uuidString
        switch start {
        case .fresh(let freshInitialization):
            run = (UUID().uuidString, Self.segmentIndex(exactResumeOf: nil), segmentID, .fresh, false, [], false)
            initialization = freshInitialization
            parent = nil
            segments = []
            derivationHistory = []
            baseGames = 0; basePositions = 0; baseTrainStepSec = 0; baseWallSec = 0
        case .branch(let file):
            run = (UUID().uuidString, Self.segmentIndex(exactResumeOf: nil), segmentID, .branch, false, [], false)
            initialization = nil
            parent = file.recordParent
            segments = []
            derivationHistory = file.derivationHistory
            baseGames = 0; basePositions = 0; baseTrainStepSec = 0; baseWallSec = 0
        case .resume(let file, let gaps, let legacyTotals):
            parent = file.recordParent
            derivationHistory = file.derivationHistory
            let exactness = ResumeExactness.resume(of: file, gaps: gaps)
            // The same run continues, drawn from the same starting weights.
            initialization = file.lineage.record?.rng.initialization
            switch file.lineage {
            case .recorded(let record):
                guard legacyTotals == nil else {
                    throw TrackerError.legacyTotalsWithRecordedLineage(parentModelID: file.modelID)
                }
                run = (record.run.lineageRunID, Self.segmentIndex(exactResumeOf: file), segmentID, .resume,
                       exactness.isExact, exactness.tokens, record.run.continuesUnrecordedHistory)
                segments = record.segments + [LineageRecord.SegmentSummary(of: record)]
                baseGames = record.fed.cumGames
                basePositions = record.fed.cumPositions
                baseTrainStepSec = record.time.cumTrainStepSec
                baseWallSec = record.time.cumWallSec
            case .unrecorded:
                // The parent predates lineage: this run starts here, and the
                // history before it is unrecorded except what a legacy GUI
                // session counted itself.
                run = (UUID().uuidString, Self.segmentIndex(exactResumeOf: file), segmentID, .resume, false, exactness.tokens, true)
                segments = []
                baseGames = nil
                basePositions = nil
                baseTrainStepSec = nil
                baseWallSec = legacyTotals.map(\.wallSec)
            }
        }
    }

    /// Add one measured trainer step.
    func recordTrainingStep(totalMs: Double) {
        segmentTrainStepSec.modify { $0 += totalMs / 1000 }
    }

    /// The record for a save at `date`.
    ///
    /// - Parameters:
    ///   - trainerCompletedSteps: the trainer clock (`trainer_completed_steps`)
    ///     of a trainer-state file, or the trainer step a model file's
    ///     weights were taken at; nil when the writer has no trainer.
    ///   - segmentLocalStep: steps this segment trained.
    ///   - segmentGames, segmentPositions: what this segment fed into its
    ///     replay buffer (not counting a resume's reconstruction refeed of
    ///     games the parent had already fed).
    ///   - corpus: the corpus position, for corpus replay only.
    ///   - parameters: the training parameters in force.
    ///   - rng: the run's random state captured in the same consistent cut as
    ///     the saved trainer state (its dropout Philox state and stream
    ///     positions), or `.withoutRunStreams` for a save with no run behind
    ///     it.
    func record(at date: Date,
                trainerCompletedSteps: Int?,
                segmentLocalStep: Int,
                segmentGames: Int,
                segmentPositions: Int,
                corpus: LineageRecord.CorpusPosition?,
                parameters: LineageRecord.Parameters?,
                rng: LineageRecord.RNG) throws -> LineageRecord {
        guard segmentGames >= 0 else { throw TrackerError.negativeSegmentCount(what: "games", value: segmentGames) }
        guard segmentPositions >= 0 else { throw TrackerError.negativeSegmentCount(what: "positions", value: segmentPositions) }
        guard segmentLocalStep >= 0 else { throw TrackerError.negativeSegmentCount(what: "steps", value: segmentLocalStep) }
        let stepSec = segmentTrainStepSec.value
        let wallSec = max(0, date.timeIntervalSince(startedAt))
        return LineageRecord(
            schema: LineageRecord.currentSchema,
            run: LineageRecord.Run(
                lineageRunID: run.lineageRunID,
                segmentIndex: run.segmentIndex,
                segmentID: run.segmentID,
                segmentStartedUnix: Int64(startedAt.timeIntervalSince1970),
                start: run.start,
                exactResume: run.exactResume,
                notExactItems: run.notExactItems,
                continuesUnrecordedHistory: run.continuesUnrecordedHistory,
                recordedUnix: Int64(date.timeIntervalSince1970)
            ),
            parent: parent,
            steps: LineageRecord.Steps(
                cumTrainerStep: trainerCompletedSteps,
                segmentStartTrainerStep: segmentStartTrainerStep,
                segmentLocalStep: segmentLocalStep
            ),
            fed: LineageRecord.Fed(
                cumGames: baseGames.map { $0 + segmentGames },
                cumPositions: basePositions.map { $0 + segmentPositions },
                segmentGames: segmentGames,
                segmentPositions: segmentPositions,
                corpus: corpus
            ),
            time: LineageRecord.Time(
                cumTrainStepSec: baseTrainStepSec.map { $0 + stepSec },
                cumWallSec: baseWallSec.map { $0 + wallSec },
                segmentTrainStepSec: stepSec,
                segmentWallSec: wallSec
            ),
            parameters: parameters,
            build: .current,
            invocation: LineageRecord.Invocation(argv: argv, pathKind: pathKind),
            device: .current,
            rng: rng.withInitialization(initialization),
            segments: segments,
            derivationHistory: derivationHistory
        )
    }

    /// The record of the segment as it starts, before it trains or feeds
    /// anything: what the `[RUN]` line reports.
    func startRecord(at date: Date, trainerCompletedSteps: Int?,
                     parameters: LineageRecord.Parameters?) throws -> LineageRecord {
        try record(at: date, trainerCompletedSteps: trainerCompletedSteps, segmentLocalStep: 0,
                   segmentGames: 0, segmentPositions: 0, corpus: nil, parameters: parameters,
                   rng: .withoutRunStreams(dropoutPhiloxState: nil))
    }

    /// The run's totals now, for a `results.json` row: the trainer clock,
    /// measured trainer-step time and games fed, each continuing the
    /// totals the segment started from (nil where no predecessor recorded
    /// one).
    struct Totals: Equatable, Sendable {
        let cumTrainerStep: Int?
        let cumTrainStepSec: Double?
        let cumGames: Int?

        init(cumTrainerStep: Int?, cumTrainStepSec: Double?, cumGames: Int?) {
            self.cumTrainerStep = cumTrainerStep
            self.cumTrainStepSec = cumTrainStepSec
            self.cumGames = cumGames
        }

        /// The totals a record states.
        init(of record: LineageRecord) {
            self.init(cumTrainerStep: record.steps.cumTrainerStep,
                      cumTrainStepSec: record.time.cumTrainStepSec,
                      cumGames: record.fed.cumGames)
        }
    }

    /// `Totals` with `segmentGames` fed by this segment so far.
    func totals(trainerCompletedSteps: Int?, segmentGames: Int) -> Totals {
        Totals(cumTrainerStep: trainerCompletedSteps,
               cumTrainStepSec: baseTrainStepSec.map { $0 + segmentTrainStepSec.value },
               cumGames: baseGames.map { $0 + segmentGames })
    }

    // MARK: Records outside training

    /// The record of a model minted by this process with fresh weights drawn
    /// under `initialization` (`--new-model`, Build Network), never trained.
    static func mintRecord(pathKind: LineageRecord.PathKind, argv: [String], initialization: ModelInitRecord,
                           at date: Date) throws -> LineageRecord {
        let tracker = try LineageTracker(start: .fresh(initialization: initialization), pathKind: pathKind, argv: argv,
                                         startedAt: date, segmentStartTrainerStep: nil)
        return try tracker.record(at: date, trainerCompletedSteps: 0, segmentLocalStep: 0,
                                  segmentGames: 0, segmentPositions: 0, corpus: nil, parameters: nil,
                                  rng: .withoutRunStreams(dropoutPhiloxState: nil))
    }

    /// The record of a model made from `source` without training —
    /// `--derive-model`'s output, or a GUI save of a loaded model before any
    /// training: a new run whose weights carry the source's history, so the
    /// totals continue from the source's record (or stay unrecorded when the
    /// source predates lineage, step total included). `derivation` is the derive step that made
    /// the copy, appended to the source's derivation history; nil for a copy
    /// no derivation made.
    static func untrainedCopyRecord(source: ParentFile, derivation: ModelDerivation.DerivationRecord?,
                                    pathKind: LineageRecord.PathKind, argv: [String], at date: Date) -> LineageRecord {
        let sourceRecord = source.lineage.record
        var derivationHistory = source.derivationHistory
        if let derivation {
            derivationHistory.append(derivation)
        }
        // Only the source's own record states the line's step total. A
        // source written before lineage states a trainer clock (or the step
        // its weights were taken at), but a resumed segment of that era
        // restarted its clock, so that number can be segment-local: it is
        // kept as the parent's stated step (`parent.trainer_completed_steps`)
        // and the total stays unrecorded. (A legacy trainer-state resume
        // differs on purpose: it continues that very clock, so the clock is
        // its total.)
        let sourceStepTotal = sourceRecord?.steps.cumTrainerStep
        return LineageRecord(
            schema: LineageRecord.currentSchema,
            run: LineageRecord.Run(
                lineageRunID: UUID().uuidString,
                segmentIndex: 0,
                segmentID: UUID().uuidString,
                segmentStartedUnix: Int64(date.timeIntervalSince1970),
                start: .derive,
                exactResume: false,
                notExactItems: [],
                continuesUnrecordedHistory: sourceRecord == nil,
                recordedUnix: Int64(date.timeIntervalSince1970)
            ),
            parent: source.recordParent,
            steps: LineageRecord.Steps(
                cumTrainerStep: sourceStepTotal,
                segmentStartTrainerStep: nil,
                segmentLocalStep: 0
            ),
            fed: LineageRecord.Fed(
                cumGames: sourceRecord?.fed.cumGames,
                cumPositions: sourceRecord?.fed.cumPositions,
                segmentGames: 0,
                segmentPositions: 0,
                corpus: nil
            ),
            time: LineageRecord.Time(
                cumTrainStepSec: sourceRecord?.time.cumTrainStepSec,
                cumWallSec: sourceRecord?.time.cumWallSec,
                segmentTrainStepSec: 0,
                segmentWallSec: 0
            ),
            parameters: sourceRecord?.parameters,
            build: .current,
            invocation: LineageRecord.Invocation(argv: LineageRecord.redactedArguments(argv), pathKind: pathKind),
            device: .current,
            // The copy has no trainer behind it: there is no dropout state
            // to continue.
            rng: .withoutRunStreams(dropoutPhiloxState: nil),
            segments: [],
            derivationHistory: derivationHistory
        )
    }
}
