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
//  Schema 3: the tracker also holds the segment's configuration (set once by
//  the path that owns the segment: `configureSegment`), the seeds it trained
//  under (`noteRunSeed`), and — on the GUI, where a segment outlives Stop,
//  Continue and "New Session, keep trainer" — the journals of what changed
//  during it: committed settings changes, champion changes and replay-ratio
//  starts. A record takes the journals up to a `SaveInputs` cut, so a save
//  whose own awaits let a later edit land never records that edit; the next
//  save does. The ancestry a branch or derive carries is built here too, so
//  every path records it the same way.
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

        /// A trainer this process holds but never tracked (a model was
        /// loaded since its history began) as the parent of a run that
        /// keeps training it: no file states it, so it has no content hash,
        /// an unrecorded lineage and no derivation history. Its model ID is
        /// written into every later record, so a trainer without one is
        /// refused rather than recorded under a placeholder.
        static func untrackedTrainer(identifier: ModelID?, completedSteps: Int) throws -> ParentFile {
            guard let identifier else {
                throw TrackerError.noModelID(what: "the kept trainer")
            }
            return ParentFile(
                modelID: identifier.description,
                contentSHA256: nil,
                trainerCompletedSteps: completedSteps,
                lineage: .unrecorded(formatVersion: ArchitectureFormat.currentVersion),
                derivationHistory: [])
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

    enum TrackerError: Error, CustomStringConvertible, LocalizedError {
        case legacyTotalsWithRecordedLineage(parentModelID: String)
        case negativeSegmentCount(what: String, value: Int)
        case noModelID(what: String)
        case segmentNotConfigured
        case segmentConfiguredTwice
        case noRunSeedNoted
        case noSegmentStartChampion
        case composedSnapshotHasSeedSettings(ids: [String])
        case cutBeyondJournal(what: String, cut: Int, count: Int)

        var description: String {
            switch self {
            case .legacyTotalsWithRecordedLineage(let id):
                return "lineage: parent \(id) carries a lineage record, so legacy session totals must not be supplied"
            case .negativeSegmentCount(let what, let value):
                return "lineage: segment \(what) is negative (\(value))"
            case .noModelID(let what):
                return "lineage: \(what) has no model ID"
            case .segmentNotConfigured:
                return "lineage: a record with training behind it needs the segment's configuration (configureSegment)"
            case .segmentConfiguredTwice:
                return "lineage: a segment's configuration is set once, when the segment begins"
            case .noRunSeedNoted:
                return "lineage: a record with training behind it needs the run's seed (noteRunSeed)"
            case .noSegmentStartChampion:
                return "lineage: a gui segment's record needs its segment_start champion (noteChampionChange)"
            case .composedSnapshotHasSeedSettings(let ids):
                return "lineage: a composed parameter snapshot omits the seed settings (\(ids.joined(separator: ", "))); "
                    + "the run's seed is in rng.streams and run_seeds"
            case .cutBeyondJournal(let what, let cut, let count):
                return "lineage: a save cut of \(cut) \(what) entries exceeds the journal's \(count)"
            }
        }

        var errorDescription: String? { description }
    }

    /// What a segment's configuration holds that does not change while the
    /// segment runs, set once when it begins (`configureSegment`).
    struct SegmentConfiguration: Sendable, Equatable {
        /// The trainer architecture's policy tail (format v12: an
        /// architecture field; every path passes `trainer.arch`'s).
        let policyTailPrecision: PolicyTailPrecisionSetting
        let budget: LineageRecord.Budget
        /// Train-vs-UCI only.
        let vsuci: LineageRecord.VsUciGeneration?
        /// GUI only: the self-play Dirichlet noise its games use.
        let selfPlayDirichlet: LineageRecord.Dirichlet?
        /// Whether the start weights were value-head recentered at load.
        let startValueHeadRecentered: LineageRecord.Recorded<Bool>
    }

    /// The journal positions and per-save values one save records — taken
    /// in the same instant as the save's parameter snapshot (the GUI's
    /// configuration cut), so the snapshot, the journal it is undone with and
    /// the derived values describe one moment.
    struct SaveInputs: Sendable, Equatable {
        let parameterChangeCount: Int
        let championChangeCount: Int
        /// The fed LR/momentum at the save (`LRMomentumCycleReadout`); nil for
        /// a record with no trainer clock.
        let scheduleAtSave: LineageRecord.ScheduleAtSave?
        /// GUI only: the replay-ratio controller's state at the save.
        let replayRatioAtSave: LineageRecord.ReplayRatio.AtSave?
        /// The merged training-health alarm summary at the save; nil while
        /// no monitor reports to this segment.
        let healthAlarms: TrainingHealthSegmentSummary?
    }

    let pathKind: LineageRecord.PathKind
    let argv: [String]
    private let run: (lineageRunID: String, segmentIndex: Int, segmentID: String,
                      start: LineageRecord.SegmentStart, exactResume: Bool,
                      notExactItems: [String], continuesUnrecordedHistory: Bool)
    private let parent: LineageRecord.Parent?
    private let segments: [LineageRecord.SegmentSummary]
    /// The runs these weights descend from (plan S2's table).
    let ancestry: LineageRecord.Ancestry
    /// How the run's starting weights were drawn, recorded in every record
    /// (`rng.init_seed` / `init_scheme`); nil for a run that started from a
    /// file's weights.
    private let initialization: ModelInitRecord?

    /// `initialization`, for the analyzers' init reference
    /// (`AnalysisInitReference`): the seed the run's weights descend from.
    var startingInitialization: ModelInitRecord? { initialization }
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

    /// The segment's journals (lock: the box's own; never held while calling
    /// out). Appended from any thread — the settings observer, the arena,
    /// the start path — and read by `record`.
    fileprivate struct Journals: Sendable {
        var configuration: SegmentConfiguration?
        var runSeeds: [LineageRecord.RunSeedEntry] = []
        var parameterChanges: [LineageRecord.ParameterChange] = []
        var championChanges: [LineageRecord.ChampionChange] = []
        var replayRatioStarts: [LineageRecord.ReplayRatio.Start] = []
        /// The summary of the segment's ended health monitors, merged; empty
        /// until the first one ends.
        var endedMonitorsHealthAlarms: TrainingHealthSegmentSummary = .empty
    }
    private let journals = SyncBox<Journals>(Journals())

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
            ancestry = .fresh
            baseGames = 0; basePositions = 0; baseTrainStepSec = 0; baseWallSec = 0
        case .branch(let file):
            run = (UUID().uuidString, Self.segmentIndex(exactResumeOf: nil), segmentID, .branch, false, [], false)
            initialization = nil
            parent = file.recordParent
            segments = []
            derivationHistory = file.derivationHistory
            ancestry = try Self.ancestry(leaving: file, by: .branch, architectureAtDeparture: nil)
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
                ancestry = record.ancestry
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
                ancestry = .unrecordedHistory
                baseGames = nil
                basePositions = nil
                baseTrainStepSec = nil
                baseWallSec = legacyTotals.map(\.wallSec)
            }
        }
    }

    /// The ancestry of a run that leaves `file`'s run by `leftBy` (plan S2):
    /// the file's own ancestry plus its run; for a file with no lineage, an
    /// unrecorded history and no runs.
    static func ancestry(leaving file: ParentFile, by leftBy: LineageRecord.AncestorRun.LeftBy,
                         architectureAtDeparture: LineageRecord.AncestorRun.ArchitectureAtDeparture?) throws
        -> LineageRecord.Ancestry {
        guard let record = file.lineage.record else { return .unrecordedHistory }
        let left = LineageRecord.AncestorRun(
            lineageRunID: record.run.lineageRunID,
            leftBy: leftBy,
            leftAt: file.recordParent,
            totalsAtDeparture: LineageRecord.AncestorRun.TotalsAtDeparture(of: record),
            architectureAtDeparture: architectureAtDeparture,
            initialization: try LineageRecord.AncestorRun.initialization(of: record),
            segments: record.segments + [LineageRecord.SegmentSummary(of: record)])
        return LineageRecord.Ancestry(historyBeforeOldestRun: record.ancestry.historyBeforeOldestRun,
                                      runs: record.ancestry.runs + [left])
    }

    /// Add one measured trainer step.
    func recordTrainingStep(totalMs: Double) {
        segmentTrainStepSec.modify { $0 += totalMs / 1000 }
    }

    // MARK: Segment configuration and journals

    /// Set the segment's fixed configuration, once, when it begins.
    func configureSegment(_ configuration: SegmentConfiguration) throws {
        let alreadySet = journals.mutate { journals -> Bool in
            if journals.configuration != nil { return true }
            journals.configuration = configuration
            return false
        }
        if alreadySet { throw TrackerError.segmentConfiguredTwice }
    }

    /// Whether the segment's configuration has been set.
    var isConfigured: Bool { journals.value.configuration != nil }

    /// Note a seed the run trains under from trainer step `trainerStep`
    /// (gap 9b): once when the segment begins, and again whenever a start
    /// that keeps the segment resolves a new one ("New Session, keep
    /// trainer").
    func noteRunSeed(_ seed: RunRandomSeed, atTrainerStep trainerStep: Int) {
        let entry = LineageRecord.RunSeedEntry(
            fromTrainerStep: trainerStep, masterSeed: seed.masterSeed, seedOrigin: seed.recordedOrigin,
            streamDerivation: DCMRandomStreams.derivationVersion)
        journals.modify { $0.runSeeds.append(entry) }
    }

    /// Record the engine identity of train-vs-UCI opponent pool
    /// `opponentIndex` (its index in the run's opponent specs) from a
    /// completed handshake. The pool's first completed handshake wins; a
    /// later one is not compared (an engine that changed identity mid-run
    /// would be a different engine at the same path, which the executable
    /// hash recorded at the start does not cover either).
    func noteEngineIdentity(opponentIndex: Int, _ identity: LineageRecord.VsUciGeneration.EngineIdentity) {
        journals.modify { journals in
            guard let configuration = journals.configuration, let vsuci = configuration.vsuci,
                  vsuci.opponents.indices.contains(opponentIndex),
                  case .unrecorded = vsuci.opponents[opponentIndex].identity else { return }
            var opponents = vsuci.opponents
            let old = opponents[opponentIndex]
            opponents[opponentIndex] = LineageRecord.VsUciGeneration.Opponent(
                command: old.command, executableSHA256: old.executableSHA256, count: old.count,
                goLimit: old.goLimit, options: old.options, identity: .recorded(identity))
            journals.configuration = SegmentConfiguration(
                policyTailPrecision: configuration.policyTailPrecision, budget: configuration.budget,
                vsuci: LineageRecord.VsUciGeneration(
                    maxPliesPerGame: vsuci.maxPliesPerGame, evalSyncEverySteps: vsuci.evalSyncEverySteps,
                    trainerMoveSelection: vsuci.trainerMoveSelection, opponents: opponents),
                selfPlayDirichlet: configuration.selfPlayDirichlet,
                startValueHeadRecentered: configuration.startValueHeadRecentered)
        }
    }

    /// Journal a committed settings change (gap 4).
    func journalParameterChange(_ change: LineageRecord.ParameterChange) {
        journals.modify { $0.parameterChanges.append(change) }
    }

    /// Re-stamp every journalled change committed after `arenaStartStep` to
    /// that step, keeping its original step in `restamped_from`: an arena
    /// promotion rewound the trainer to `arenaStartStep`, and the rewound
    /// trainer uses those settings from the next step on. Commit order is
    /// kept, so the last edit of a key still wins.
    func restampParameterChanges(after arenaStartStep: Int) {
        journals.modify { journals in
            journals.parameterChanges = journals.parameterChanges.map { change in
                guard change.committedAtTrainerStep > arenaStartStep else { return change }
                return LineageRecord.ParameterChange(
                    committedAtTrainerStep: arenaStartStep, recordedUnix: change.recordedUnix, id: change.id,
                    old: change.old, new: change.new,
                    restampedFrom: change.restampedFrom ?? change.committedAtTrainerStep)
            }
        }
    }

    /// Journal a change of the data-generating champion (B5).
    func noteChampionChange(_ change: LineageRecord.ChampionChange) {
        journals.modify { $0.championChanges.append(change) }
    }

    /// Journal a Play-and-Train start's replay-ratio controller inputs (B6).
    func noteReplayRatioStart(_ start: LineageRecord.ReplayRatio.Start) {
        journals.modify { $0.replayRatioStarts.append(start) }
    }

    /// Fold an ended health monitor's summary into the segment's stored one
    /// (`configuration.health_alarms`, alarm plan OD-10). A GUI segment spans
    /// several monitors — every start makes one, and a Continue or
    /// keep-trainer start stays in the segment — so the monitor a start
    /// replaces is folded in here, once, at the replacement. A new
    /// segment's tracker starts from `.empty`, so nothing carries across a
    /// segment boundary.
    func mergeEndedHealthMonitor(_ summary: TrainingHealthSegmentSummary) {
        journals.modify { journals in
            journals.endedMonitorsHealthAlarms = journals.endedMonitorsHealthAlarms.merging(summary)
        }
    }

    /// The segment's summary for a save: the stored one with `live` (the
    /// running monitor's summary) merged in, without storing it — the live
    /// monitor is folded in only when it ends. Nil when there is no live
    /// monitor to describe the segment's monitoring, so the record says
    /// unrecorded rather than stating a summary no monitor produced.
    func healthAlarms(withLive live: TrainingHealthSegmentSummary?) -> TrainingHealthSegmentSummary? {
        guard let live else { return nil }
        return journals.value.endedMonitorsHealthAlarms.merging(live)
    }

    /// The journal positions now, with this save's derived values: what a
    /// save records when it takes no cut of its own (the CLI paths, whose
    /// journals never change during a save, and any record composed in one
    /// main-actor turn).
    func saveInputs(scheduleAtSave: LineageRecord.ScheduleAtSave?,
                    replayRatioAtSave: LineageRecord.ReplayRatio.AtSave?,
                    healthAlarms: TrainingHealthSegmentSummary?) -> SaveInputs {
        let journals = journals.value
        return SaveInputs(parameterChangeCount: journals.parameterChanges.count,
                          championChangeCount: journals.championChanges.count,
                          scheduleAtSave: scheduleAtSave, replayRatioAtSave: replayRatioAtSave,
                          healthAlarms: healthAlarms)
    }

    /// The journals as they stand, to put back if a start that added to them
    /// fails (`restoreJournals`).
    struct JournalCheckpoint: Sendable {
        fileprivate let journals: Journals
    }

    func checkpointJournals() -> JournalCheckpoint {
        JournalCheckpoint(journals: journals.value)
    }

    func restoreJournals(_ checkpoint: JournalCheckpoint) {
        journals.value = checkpoint.journals
    }

    /// The journalled settings changes, for tests and the undo derivation.
    var parameterChanges: [LineageRecord.ParameterChange] { journals.value.parameterChanges }
    var championChanges: [LineageRecord.ChampionChange] { journals.value.championChanges }
    var runSeeds: [LineageRecord.RunSeedEntry] { journals.value.runSeeds }

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
    ///   - inputs: the save's journal cut and derived values (`SaveInputs`).
    func record(at date: Date,
                trainerCompletedSteps: Int?,
                segmentLocalStep: Int,
                segmentGames: Int,
                segmentPositions: Int,
                corpus: LineageRecord.CorpusPosition?,
                parameters: LineageRecord.Parameters?,
                rng: LineageRecord.RNG,
                inputs: SaveInputs) throws -> LineageRecord {
        guard segmentGames >= 0 else { throw TrackerError.negativeSegmentCount(what: "games", value: segmentGames) }
        guard segmentPositions >= 0 else { throw TrackerError.negativeSegmentCount(what: "positions", value: segmentPositions) }
        guard segmentLocalStep >= 0 else { throw TrackerError.negativeSegmentCount(what: "steps", value: segmentLocalStep) }
        let stepSec = segmentTrainStepSec.value
        let wallSec = max(0, date.timeIntervalSince(startedAt))
        let configuration: LineageRecord.RecordedIfTrained<LineageRecord.TrainingConfiguration>
        let runSeeds: LineageRecord.RecordedIfTrained<[LineageRecord.RunSeedEntry]>
        if let parameters {
            let seedIDs = try Self.seedSettingIDs(in: parameters)
            guard seedIDs.isEmpty else { throw TrackerError.composedSnapshotHasSeedSettings(ids: seedIDs) }
            let journals = journals.value
            guard let fixed = journals.configuration else { throw TrackerError.segmentNotConfigured }
            guard !journals.runSeeds.isEmpty else { throw TrackerError.noRunSeedNoted }
            guard inputs.parameterChangeCount <= journals.parameterChanges.count else {
                throw TrackerError.cutBeyondJournal(what: "parameter-change", cut: inputs.parameterChangeCount,
                                                    count: journals.parameterChanges.count)
            }
            guard inputs.championChangeCount <= journals.championChanges.count else {
                throw TrackerError.cutBeyondJournal(what: "champion-change", cut: inputs.championChangeCount,
                                                    count: journals.championChanges.count)
            }
            let championChanges = Array(journals.championChanges.prefix(inputs.championChangeCount))
            if pathKind == .gui, championChanges.first?.trigger != .segmentStart {
                throw TrackerError.noSegmentStartChampion
            }
            configuration = .recorded(try LineageRecord.TrainingConfiguration(
                pathKind: pathKind,
                policyTailPrecision: fixed.policyTailPrecision.rawValue,
                budget: fixed.budget,
                parameterChanges: Array(journals.parameterChanges.prefix(inputs.parameterChangeCount)),
                championChanges: championChanges,
                vsuci: fixed.vsuci,
                selfPlayDirichlet: fixed.selfPlayDirichlet,
                startValueHeadRecentered: fixed.startValueHeadRecentered,
                scheduleAtSave: inputs.scheduleAtSave,
                replayRatio: pathKind == .gui
                    ? LineageRecord.ReplayRatio(starts: journals.replayRatioStarts, atSave: inputs.replayRatioAtSave)
                    : nil,
                healthAlarms: inputs.healthAlarms.map { .recorded($0) } ?? .unrecorded))
            runSeeds = .recorded(journals.runSeeds)
        } else {
            configuration = .notTrained
            runSeeds = .notTrained
        }
        return try LineageRecord(
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
            configuration: configuration,
            runSeeds: runSeeds,
            build: try .current,
            invocation: LineageRecord.Invocation(argv: argv, pathKind: pathKind),
            device: .current,
            rng: rng.withInitialization(initialization),
            segments: segments,
            ancestry: ancestry,
            derivationHistory: derivationHistory
        )
    }

    /// The seed-setting ids a snapshot holds (gap 9): a composed snapshot
    /// holds none, since the run's actual seed is in `rng.streams` and
    /// `run_seeds`. Carried snapshots are never checked.
    static func seedSettingIDs(in parameters: LineageRecord.Parameters) throws -> [String] {
        let values = try ParameterValue.parametersObject(fromJSON: Data(parameters.snapshotJSON.utf8))
        return LineageRecord.Parameters.excludedParameterIDs.filter { values[$0] != nil }.sorted()
    }

    /// The record of the segment as it starts, before it trains or feeds
    /// anything: what the `[RUN]` line reports.
    func startRecord(at date: Date, trainerCompletedSteps: Int?,
                     parameters: LineageRecord.Parameters?, inputs: SaveInputs) throws -> LineageRecord {
        try record(at: date, trainerCompletedSteps: trainerCompletedSteps, segmentLocalStep: 0,
                   segmentGames: 0, segmentPositions: 0, corpus: nil, parameters: parameters,
                   rng: .withoutRunStreams(dropoutPhiloxState: nil), inputs: inputs)
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
                                  rng: .withoutRunStreams(dropoutPhiloxState: nil),
                                  inputs: tracker.saveInputs(scheduleAtSave: nil, replayRatioAtSave: nil, healthAlarms: nil))
    }

    /// The record of a model made from `source` without training —
    /// `--derive-model`'s output, or a GUI save of a loaded model before any
    /// training: a new run whose weights carry the source's history, so the
    /// totals continue from the source's record (or stay unrecorded when the
    /// source predates lineage, step total included). `derivation` is the derive step that made
    /// the copy, appended to the source's derivation history; nil for a copy
    /// no derivation made. `sourceArchitecture` is the source file's
    /// architecture text and format version when the copy changes the
    /// architecture (gap 1c), nil when it keeps it.
    ///
    /// The copy carries the source's parameters, configuration and run
    /// seeds verbatim (a carried configuration keeps its own `path_kind`),
    /// and its ancestry gains the source's run, left by `derive`.
    static func untrainedCopyRecord(source: ParentFile, derivation: ModelDerivation.DerivationRecord?,
                                    sourceArchitecture: LineageRecord.AncestorRun.ArchitectureAtDeparture?,
                                    pathKind: LineageRecord.PathKind, argv: [String], at date: Date) throws -> LineageRecord {
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
        return try LineageRecord(
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
            configuration: sourceRecord?.configuration ?? .notTrained,
            runSeeds: sourceRecord?.runSeeds ?? .notTrained,
            build: try .current,
            invocation: LineageRecord.Invocation(argv: LineageRecord.redactedArguments(argv), pathKind: pathKind),
            device: .current,
            // The copy has no trainer behind it: there is no dropout state
            // to continue.
            rng: .withoutRunStreams(dropoutPhiloxState: nil),
            segments: [],
            ancestry: try ancestry(leaving: source, by: .derive, architectureAtDeparture: sourceArchitecture),
            derivationHistory: derivationHistory
        )
    }
}
