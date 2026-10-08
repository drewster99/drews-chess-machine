import Foundation

/// How a model's weights were trained: the training path of a segment that
/// trained them. Named for what the path feeds the trainer, the way the
/// pickers show it (`MODEL_TRAINING_METHOD_PLAN.md`).
enum ModelTrainingMethod: String, Codable, Sendable, CaseIterable {
    /// GUI Play-and-Train (and `--train`): the champion's own games.
    case selfPlay = "self_play"
    /// `--replay-corpus`: imported games.
    case corpusReplay = "corpus_replay"
    /// `--train-vs-uci`: games against external UCI engines.
    case uciPlay = "uci_play"

    /// The method a segment of `pathKind` trains by; nil for the paths that
    /// write weights without training them (`--derive-model`,
    /// `--new-model`).
    init?(pathKind: LineageRecord.PathKind) {
        switch pathKind {
        case .gui: self = .selfPlay
        case .replay: self = .corpusReplay
        case .vsuci: self = .uciPlay
        case .derive, .newModel: return nil
        }
    }

    /// The method a file written before lineage records was trained by, from
    /// the `creator` that wrote it: only the CLI training paths say. The GUI
    /// writers (`manual`, `periodic`, `promote`, `sigusr2`) also saved built
    /// and loaded models, so their files say nothing about training.
    init?(creator: String) {
        switch creator {
        case ModelCheckpointMetadata.corpusReplayCreator: self = .corpusReplay
        case ModelCheckpointMetadata.trainVsUciCreator: self = .uciPlay
        default: return nil
        }
    }

    var displayName: String {
        switch self {
        case .selfPlay: return "self-play"
        case .corpusReplay: return "corpus replay"
        case .uciPlay: return "UCI play"
        }
    }
}

/// Every method a model's weights went through, oldest first, consecutive
/// repeats collapsed (owner decision 2026-10-08: `corpus replay → self-play`).
/// Empty when nothing records how the weights were trained, or they never
/// were; the pickers then show no method ("if we know"). History before the
/// oldest run a record covers is not marked: it would sit on nearly every
/// older file.
///
/// Derived in one place from a model file's lineage record or, without one,
/// its `creator`; a Lichess game record keeps the history of the weights it
/// played (`LichessBotGenerationInfo.trainingHistory`).
struct ModelTrainingHistory: Codable, Equatable, Sendable {
    let methods: [ModelTrainingMethod]

    enum CodingKeys: String, CodingKey {
        case methods
    }

    init(methods: [ModelTrainingMethod]) {
        var collapsed: [ModelTrainingMethod] = []
        for method in methods where collapsed.last != method {
            collapsed.append(method)
        }
        self.methods = collapsed
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.init(methods: try c.decode([ModelTrainingMethod].self, forKey: .methods))
    }

    /// Nothing known.
    static let unknown = ModelTrainingHistory(methods: [])

    /// "corpus replay → self-play"; nil when nothing is known.
    var displayText: String? {
        methods.isEmpty ? nil : methods.map(\.displayName).joined(separator: " → ")
    }

    /// For a log line: the methods' raw values oldest first, joined by
    /// ">" ("corpus_replay>self_play"), or "unrecorded".
    var logText: String {
        methods.isEmpty ? "unrecorded" : methods.map(\.rawValue).joined(separator: ">")
    }

    /// The history a lineage record states: each ancestor run's segments,
    /// then the record's earlier segments, then its own
    /// (`LineageRecord.SegmentSummary(of:)`), oldest first.
    init(record: LineageRecord) {
        let segments = record.ancestry.runs.flatMap(\.segments)
            + record.segments
            + [LineageRecord.SegmentSummary(of: record)]
        self.init(methods: segments.compactMap(Self.method(of:)))
    }

    /// The history of a file with lineage `lineage` written by `creator`: its
    /// record's, or for a file before lineage records, its creator's method.
    init(lineage: LineageRecord.Presence, creator: String?) {
        switch lineage {
        case .recorded(let record):
            self.init(record: record)
        case .unrecorded:
            self.init(methods: creator.flatMap(ModelTrainingMethod.init(creator:)).map { [$0] } ?? [])
        }
    }

    /// The method one segment trained by. Its configuration's path when
    /// recorded (none when it states the segment did not train); for a
    /// segment of a record before schema 3, which has no configuration, the
    /// path that wrote it when it records a parameter snapshot (it trained).
    /// Otherwise unknown.
    static func method(of segment: LineageRecord.SegmentSummary) -> ModelTrainingMethod? {
        switch segment.configuration {
        case .recorded(let configuration):
            return configuration.flatMap { ModelTrainingMethod(pathKind: $0.pathKind) }
        case .unrecorded:
            guard case .recorded(let parameters) = segment.parameters, parameters != nil,
                  case .recorded(let pathKind) = segment.pathKind else { return nil }
            return ModelTrainingMethod(pathKind: pathKind)
        }
    }
}

extension ModelCheckpointFile {
    /// How this file's weights were trained: its lineage record's history,
    /// or for a file before lineage records (a legacy `.dcmmodel` among
    /// them), its creator's method.
    var trainingHistory: ModelTrainingHistory {
        ModelTrainingHistory(lineage: lineageParent.lineage, creator: metadata.creator.isEmpty ? nil : metadata.creator)
    }
}

extension ModelTrainingHistory {
    /// The history in a safetensors model file's header: nil when the file
    /// doesn't exist (a legacy `.dcmmodel` session has none at the
    /// safetensors path) or its header or lineage record doesn't read, the
    /// reason logged under `logTag`.
    static func ofModelFile(at url: URL, logTag: String) -> ModelTrainingHistory? {
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        do {
            let metadata = try ModelFileCatalog.headerMetadata(at: url)
            let lineage = try SafetensorsModelIO.readParentFile(fromMetadata: metadata, source: url.lastPathComponent).lineage
            return ModelTrainingHistory(lineage: lineage, creator: metadata[SafetensorsModelIO.Key.creator].flatMap { $0.isEmpty ? nil : $0 })
        } catch {
            SessionLogger.shared.log("\(logTag) training method of \(url.deletingLastPathComponent().lastPathComponent)/\(url.lastPathComponent) not read: \(error)")
            return nil
        }
    }
}
