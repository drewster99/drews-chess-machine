//
//  LineageConfigurationInputs.swift
//  DrewsChessMachine
//
//  The one place each lineage-configuration input is derived from what a
//  path already holds, so corpus replay, train-vs-UCI and the GUI compute it
//  the same way: the snapshot a record composes, whether a segment's start
//  weights were value-head recentered, and corpus replay's resolved budget.
//

import Foundation

extension TrainingParametersSnapshot {
    /// The parameter values a lineage record composes: every parameter except
    /// the seed settings (`LineageRecord.Parameters.excludedParameterIDs`,
    /// gap 9), which the run's actual seed in `rng.streams` / `run_seeds`
    /// supersedes.
    func lineageValues() -> [String: ParameterValue] {
        let excluded = Set(LineageRecord.Parameters.excludedParameterIDs)
        return rawValueMap().filter { !excluded.contains($0.key) }
    }
}

extension LineageTracker {
    enum StartWeightsError: Error, CustomStringConvertible, LocalizedError {
        case centeringNotDecided(file: String)
        case keptAsStored(file: String)

        var description: String {
            switch self {
            case .centeringNotDecided(let file):
                return "lineage: \(file) was decoded without a value-head centering decision"
            case .keptAsStored(let file):
                return "lineage: \(file) was decoded as stored, for analysis only; it never starts training"
            }
        }

        var errorDescription: String? { description }
    }

    /// Whether weights decoded with `centering` were recentered at load (B9):
    /// the start weights then differ from the bytes the parent's content hash
    /// names. A file decoded as stored is analysis-only and never trains, so
    /// it is an error here, as is a file with no centering decision.
    static func startValueHeadRecentered(_ centering: ValueHeadCentering?, file: String) throws -> Bool {
        switch centering {
        case .recentered: return true
        case .alreadyCentered, .notApplicable: return false
        case .keptAsStored: throw StartWeightsError.keptAsStored(file: file)
        case nil: throw StartWeightsError.centeringNotDecided(file: file)
        }
    }

    /// A CLI run's start weights: its `--start-model`'s centering, or false
    /// for a fresh run, whose weights were drawn (review P3-2).
    static func startValueHeadRecentered(of startModel: ModelCheckpointFile?) throws -> Bool {
        guard let startModel else { return false }
        return try startValueHeadRecentered(startModel.valueHeadCentering, file: startModel.modelID)
    }
}

extension CorpusReplayConfig {
    /// The limits corpus replay enforces, resolved once for both the run and
    /// its records: an explicit step limit, else `--epochs`, else one pass
    /// when neither is given. Replay has no wall-clock limit (O-21).
    var resolvedBudget: LineageRecord.Budget {
        LineageRecord.Budget(trainingStepLimit: stepLimit, trainingTimeLimitSec: nil,
                             epochLimit: epochs ?? (stepLimit == nil ? 1 : nil))
    }
}
