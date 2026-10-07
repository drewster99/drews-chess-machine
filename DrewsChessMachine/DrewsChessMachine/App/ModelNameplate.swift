//
//  ModelNameplate.swift
//  DrewsChessMachine
//
//  What the title bar and About popover say about the champion beyond its
//  topology: the name and preset it was made under (`ModelNaming`, from its
//  lineage) and the format of the file its weights were read from
//  (MODEL_NAMING_PLAN.md, owner decision D3).
//
//  Derived from `SessionController.championOrigin`, the one record of where
//  the champion's weights came from, whenever it changes — never set on its
//  own, so it can't describe weights the champion no longer holds.
//

import Foundation

/// The container a model file's weights were decoded from.
enum ModelFileFormat: Sendable, Equatable {
    /// A safetensors file at this architecture format version
    /// (`dcm_format_version`).
    case safetensors(architectureFormat: Int)
    /// A legacy `.dcmmodel` file at this binary format version.
    case legacyDCMModel(formatVersion: UInt32)
}

struct ModelNameplate: Sendable, Equatable {
    /// Where the champion's current weights came from.
    enum WeightsSource: Sendable, Equatable {
        /// Read from a file in this format (Load Model, Load Session,
        /// resume).
        case file(ModelFileFormat)
        /// Made in this process — a build, or a promotion of the trainer's
        /// weights — and never read from a file: shown as the format this
        /// build writes, which every save of them uses.
        case thisProcess
    }

    let naming: LineageRecord.Recorded<ModelNaming>
    let weightsSource: WeightsSource

    init(naming: LineageRecord.Recorded<ModelNaming>, weightsSource: WeightsSource) {
        self.naming = naming
        self.weightsSource = weightsSource
    }

    /// The champion's nameplate. A file or promotion origin names the model
    /// by its record (unrecorded for a file written before lineage).
    init(championOrigin: SessionController.ChampionOrigin) {
        switch championOrigin {
        case .built(_, let naming):
            self.init(naming: .recorded(naming), weightsSource: .thisProcess)
        case .file(let source, let startWeights):
            let naming = source.lineage.record?.modelNaming ?? .unrecorded
            switch startWeights {
            case .loaded(_, let fileFormat):
                self.init(naming: naming, weightsSource: .file(fileFormat))
            case .notLoaded:
                self.init(naming: naming, weightsSource: .thisProcess)
            }
        }
    }

    /// "format v12", or "legacy .dcmmodel v2".
    var formatText: String {
        switch weightsSource {
        case .file(.safetensors(let version)):
            return "format v\(version)"
        case .file(.legacyDCMModel(let version)):
            return "legacy .dcmmodel v\(version)"
        case .thisProcess:
            return "format v\(ArchitectureFormat.currentVersion)"
        }
    }

    /// The title bar's text before the topology: the naming's parts that
    /// are stated, then the format — "my-net · preset v4_5block_7x7
    /// (edited) · format v12".
    var headerText: String {
        [ModelNaming.headerText(naming), formatText].compactMap { $0 }.joined(separator: " · ")
    }

    /// The About popover's File format row.
    var formatRowText: String {
        let writes = "saves write v\(ArchitectureFormat.currentVersion)"
        switch weightsSource {
        case .file:
            return "\(formatText) — the file the weights were loaded from (\(writes))"
        case .thisProcess:
            return "\(formatText) — weights made in this process, not loaded from a file (\(writes))"
        }
    }
}
