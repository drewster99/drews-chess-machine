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
        /// Weights initialized by this run.
        case fresh
        /// Train from `parent`'s weights with a fresh trainer clock: a new
        /// run that records its parent.
        case branch(parent: ParentFile)
        /// Continue `parent`'s run by exact resume. `notExactItems` names
        /// whatever the resume could not restore (empty for a complete
        /// restore). `legacyTotals` supplies the totals of a GUI session
        /// written before lineage existed; it must be nil when the parent
        /// carries a lineage.
        case resume(parent: ParentFile, notExactItems: [String], legacyTotals: LegacySessionTotals?)
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
        case .fresh:
            run = (UUID().uuidString, 0, segmentID, .fresh, false, [], false)
            parent = nil
            segments = []
            baseGames = 0; basePositions = 0; baseTrainStepSec = 0; baseWallSec = 0
        case .branch(let file):
            run = (UUID().uuidString, 0, segmentID, .branch, false, [], false)
            parent = file.recordParent
            segments = []
            baseGames = 0; basePositions = 0; baseTrainStepSec = 0; baseWallSec = 0
        case .resume(let file, let notExactItems, let legacyTotals):
            parent = file.recordParent
            switch file.lineage {
            case .recorded(let record):
                guard legacyTotals == nil else {
                    throw TrackerError.legacyTotalsWithRecordedLineage(parentModelID: file.modelID)
                }
                run = (record.run.lineageRunID, record.run.segmentIndex + 1, segmentID, .resume,
                       notExactItems.isEmpty, notExactItems, record.run.continuesUnrecordedHistory)
                segments = record.segments + [LineageRecord.SegmentSummary(of: record)]
                baseGames = record.fed.cumGames
                basePositions = record.fed.cumPositions
                baseTrainStepSec = record.time.cumTrainStepSec
                baseWallSec = record.time.cumWallSec
            case .unrecorded:
                // The parent predates lineage: this run starts here, and the
                // history before it is unrecorded except what a legacy GUI
                // session counted itself.
                let items = notExactItems.contains(Self.lineageNotExactItem)
                    ? notExactItems : notExactItems + [Self.lineageNotExactItem]
                run = (UUID().uuidString, 0, segmentID, .resume, false, items, true)
                segments = []
                baseGames = nil
                basePositions = nil
                baseTrainStepSec = nil
                baseWallSec = legacyTotals.map(\.wallSec)
            }
        }
    }

    /// The `not_exact_items` token for a resume whose parent carried no
    /// lineage (determinism plan C3 `lineage`).
    static let lineageNotExactItem = "lineage"

    /// `not_exact_items` tokens (determinism plan C3) for the state a resume
    /// does not yet restore. Provisional: the plan's `ResumeGap` /
    /// `ResumeExactness` (phase P9) becomes their single source and decides
    /// them per checkpoint; until then each path lists what it is known not
    /// to restore.
    enum NotExactItem {
        /// The replay-buffer sampler draws from the system generator.
        static let rngSampler = "rng_sampler"
        /// Dropout masks are reseeded per process.
        static let dropoutState = "dropout_state"
        /// Corpus replay's feed phase restarts at the resume.
        static let feedCarry = "feed_carry"
        /// The replay buffer was not saved with the checkpoint.
        static let buffer = "buffer"

        static let replayResume = [rngSampler, dropoutState, feedCarry]
        static let vsUciResume = [rngSampler, dropoutState, buffer]
        static let guiResume = [rngSampler, dropoutState]
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
    func record(at date: Date,
                trainerCompletedSteps: Int?,
                segmentLocalStep: Int,
                segmentGames: Int,
                segmentPositions: Int,
                corpus: LineageRecord.CorpusPosition?,
                parameters: LineageRecord.Parameters?) throws -> LineageRecord {
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
            rng: .unseeded,
            segments: segments
        )
    }

    // MARK: Records outside training

    /// The record of a model minted by this process with fresh weights
    /// (`--new-model`, Build Network), never trained.
    static func mintRecord(pathKind: LineageRecord.PathKind, argv: [String], at date: Date) throws -> LineageRecord {
        let tracker = try LineageTracker(start: .fresh, pathKind: pathKind, argv: argv,
                                         startedAt: date, segmentStartTrainerStep: nil)
        return try tracker.record(at: date, trainerCompletedSteps: 0, segmentLocalStep: 0,
                                  segmentGames: 0, segmentPositions: 0, corpus: nil, parameters: nil)
    }

    /// The record of a model made from `source` without training —
    /// `--derive-model`'s output, or a GUI save of a loaded model before any
    /// training: a new run whose weights carry the source's history, so the
    /// totals continue from the source's record (or stay unrecorded when the
    /// source predates lineage).
    static func untrainedCopyRecord(source: ParentFile, pathKind: LineageRecord.PathKind, argv: [String], at date: Date) -> LineageRecord {
        let sourceRecord = source.lineage.record
        // The source's own record is the total when it has one; a source
        // written before lineage still states its trainer clock (or the
        // step its weights were taken at), which is that total too.
        let sourceStepTotal: Int?
        if let sourceRecord {
            sourceStepTotal = sourceRecord.steps.cumTrainerStep
        } else {
            sourceStepTotal = source.trainerCompletedSteps
        }
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
            rng: .unseeded,
            segments: []
        )
    }
}
