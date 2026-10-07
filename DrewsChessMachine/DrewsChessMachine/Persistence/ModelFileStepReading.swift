//
//  ModelFileStepReading.swift
//  DrewsChessMachine
//
//  What a model file's `training_step` means, read one way everywhere.
//
//  From architecture format v11 every writer states the trainer step — the
//  overall step the weights were taken at — as `training_step`; on a
//  trainer-state file it is the same value as `trainer_completed_steps` and
//  the lineage record's `cum_trainer_step` (one value, three places, refused
//  by the writer and by the reader when they differ). The writing segment's
//  own step count is the sidecar: the lineage record's `segment_local_step`,
//  mirrored flat as `lineage_segment_local_step`.
//
//  Before v11 the value is whatever its writer wrote. Corpus replay and
//  train-vs-UCI wrote the segment's own step (the dashboards added each
//  segment's `cumstep_base` to it); the GUI wrote its cumulative trainer step
//  (or, on a champion file, the trainer clock its source stated). Nothing on
//  disk is rewritten: an older file is read by its writer — the file's
//  `creator`, never the lineage record's `path_kind`, which on a GUI champion
//  file names where the weights came from, not who wrote the file — and every
//  load of one that states a step is flagged with one `[ARCH] legacy file`
//  entry. A writer outside the known sets keeps the reading every reader used
//  before v11 (`trainer_completed_steps`, else `training_step`) and is
//  flagged too, rather than refused: it is a readable file.
//
//  Python mirror: `scripts/dcm_lineage.py` `step_reading`, with the same
//  table and constants (pinned against this file by
//  `documentation/dashboards/tests/test_lineage.py`).
//

import Foundation

/// Which rule a model file's `training_step` was read under.
enum TrainingStepBasis: String, Sendable, Equatable, CaseIterable {
    /// Format v11 or later: `training_step` is the trainer step.
    case trainerStep = "trainer_step"
    /// Before v11, written by corpus replay or train-vs-UCI: `training_step`
    /// is the writing segment's own step.
    case legacySegmentStep = "legacy_segment_step"
    /// Before v11, written by the GUI: `training_step` is its trainer step.
    case legacyGUITrainerStep = "legacy_gui_trainer_step"
    /// Before v11, any other writer (or none): read as every reader read it
    /// before v11, and flagged.
    case legacyUnknownWriter = "legacy_unknown_writer"
}

/// What one model file says about its step, under its own format and
/// writer.
struct ModelFileStepReading: Sendable, Equatable {
    let basis: TrainingStepBasis
    /// The header's `training_step` as written; nil when the file states
    /// none (a fresh build, a derive, a graft).
    let statedTrainingStep: Int?
    /// The trainer step the weights were taken at; nil when the file never
    /// recorded one (never reconstructed).
    let trainerStep: Int?
    /// The steps the writing segment had trained; nil when unknown.
    let segmentStep: Int?
    /// The `[ARCH] legacy file` entry for a file before v11 that states a
    /// step; nil otherwise.
    let legacyResolution: String?

    /// The trainer step where one is known, else the stated step: what
    /// catalogs order and display by, and what a copy of the file records as
    /// its parent's step (`LineageTracker.ParentFile.trainerCompletedSteps`,
    /// "the parent's stated step").
    var trainerStepOrStatedStep: Int? { trainerStep ?? statedTrainingStep }

    /// The reading of a file's facts — the one rule set. `recordSteps` is
    /// asked only when the reading needs the lineage record (a segment step
    /// from it, or the trainer step of a pre-v11 corpus-replay /
    /// train-vs-UCI file with no trainer clock), so a header reader that
    /// does not otherwise decode the record does so only then. Throws when a
    /// v11 trainer-state file's `training_step` is not its trainer clock.
    static func reading(
        formatVersion: Int,
        creator: String,
        statedTrainingStep: Int?,
        trainerCompletedSteps: Int?,
        recordSteps: () throws -> LineageRecord.Steps?,
        source: String
    ) throws -> ModelFileStepReading {
        if formatVersion >= ArchitectureFormat.trainingStepIsTrainerStepFromVersion {
            if let clock = trainerCompletedSteps, statedTrainingStep != clock {
                throw SafetensorsModelIO.IOError.decodedTrainingStepDisagreesWithTrainerClock(
                    source: source, formatVersion: formatVersion,
                    trainingStep: statedTrainingStep, trainerClock: clock)
            }
            guard let stated = statedTrainingStep else {
                return ModelFileStepReading(basis: .trainerStep, statedTrainingStep: nil,
                                            trainerStep: trainerCompletedSteps, segmentStep: nil,
                                            legacyResolution: nil)
            }
            return ModelFileStepReading(basis: .trainerStep, statedTrainingStep: stated, trainerStep: stated,
                                        segmentStep: try recordSteps()?.segmentLocalStep, legacyResolution: nil)
        }
        let basis: TrainingStepBasis
        if ModelCheckpointMetadata.segmentStepCreators.contains(creator) {
            basis = .legacySegmentStep
        } else if ModelCheckpointMetadata.guiCreators.contains(creator) {
            basis = .legacyGUITrainerStep
        } else {
            basis = .legacyUnknownWriter
        }
        guard let stated = statedTrainingStep else {
            // A file that states no step has none to misread; never flagged.
            return ModelFileStepReading(basis: basis, statedTrainingStep: nil, trainerStep: trainerCompletedSteps,
                                        segmentStep: nil, legacyResolution: nil)
        }
        let before = "before format v\(ArchitectureFormat.trainingStepIsTrainerStepFromVersion)"
        switch basis {
        case .legacySegmentStep:
            let trainer: Int?
            let origin: String
            if let clock = trainerCompletedSteps {
                trainer = clock
                origin = "trainer step \(clock) (\(TrainerScheduleState.MetadataKey.completedTrainSteps))"
            } else if let cumulative = try recordSteps()?.cumTrainerStep {
                trainer = cumulative
                origin = "trainer step \(cumulative) (lineage cum_trainer_step)"
            } else {
                trainer = nil
                origin = "trainer step not recorded"
            }
            let writer = creator == ModelCheckpointMetadata.corpusReplayCreator ? "corpus replay" : "train-vs-UCI"
            return ModelFileStepReading(
                basis: basis, statedTrainingStep: stated, trainerStep: trainer, segmentStep: stated,
                legacyResolution: "training_step \(stated) is the writing segment's step (\(writer) \(before)); \(origin)")
        case .legacyGUITrainerStep:
            return ModelFileStepReading(
                basis: basis, statedTrainingStep: stated, trainerStep: trainerCompletedSteps ?? stated,
                segmentStep: try recordSteps()?.segmentLocalStep,
                legacyResolution: "training_step \(stated) is the GUI's trainer step (written by '\(creator)' \(before))")
        case .trainerStep:
            preconditionFailure("a file before format v11 is never read under the v11 basis")
        case .legacyUnknownWriter:
            return ModelFileStepReading(
                basis: .legacyUnknownWriter, statedTrainingStep: stated, trainerStep: trainerCompletedSteps ?? stated,
                segmentStep: try recordSteps()?.segmentLocalStep,
                legacyResolution: "training_step \(stated) was stated by writer '\(creator)', read as the trainer "
                    + "step as \(before)")
        }
    }
}

extension SafetensorsModelIO {

    /// The step reading of a safetensors header, without decoding weights —
    /// for catalogs, guards and header-only readers. The lineage record is
    /// decoded only when the reading needs it.
    static func trainingStepReading(fromMetadata md: [String: String], source: String) throws -> ModelFileStepReading {
        let version = try ArchitectureFormat.safetensorsFormatVersion(
            metadataValue: md[Key.formatVersion], source: source)
        return try ModelFileStepReading.reading(
            formatVersion: version,
            creator: md[Key.creator] ?? "",
            statedTrainingStep: try trainingStep(fromMetadata: md, source: source),
            trainerCompletedSteps: try TrainerScheduleState.decode(fromMetadata: md)?.completedTrainSteps,
            recordSteps: { try lineage(fromMetadata: md, formatVersion: version).record?.steps },
            source: source)
    }
}
