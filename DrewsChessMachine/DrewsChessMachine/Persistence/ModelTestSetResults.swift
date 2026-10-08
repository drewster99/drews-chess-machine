import Foundation

/// The puzzle test-set results of the weights in one model file, written into
/// its safetensors `__metadata__` under `metadataKey` (test-set results plan,
/// `documentation/plans-active/TEST_SET_RESULTS_METADATA_PLAN.md`).
///
/// Every writer evaluates the exact weights it is about to write
/// (`ModelTestSetEvaluator`) and passes the outcome to
/// `SafetensorsModelIO.encode`, which requires it, so no file is written
/// without one. An evaluation that fails never fails the save: the file
/// records `.failed` with the reason instead.
///
/// The value is one JSON object (sorted keys, snake_case):
///
/// ```json
/// {"schema": 1, "status": "evaluated", "evaluated_at_unix": …, "build": …,
///  "policy_tail_precision": "…", "sets": [{"id": "lichess-wide", …}]}
/// {"schema": 1, "status": "failed", "reason": "…"}
/// ```
///
/// Percentages are not stored; readers derive them from the counts.
enum ModelTestSetResultsField: Equatable, Sendable {
    case evaluated(ModelTestSetResults)
    case failed(reason: String)

    static let metadataKey = "dcm_test_set_results"
    static let schema = 1

    /// The metadata value: the JSON text.
    func metadataValue() throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        return String(decoding: try encoder.encode(Envelope(self)), as: UTF8.self)
    }

    /// What a file's `__metadata__` says. Never throws: the results are
    /// informational, so an unreadable value is reported as such and the
    /// model still loads.
    static func reading(fromMetadata metadata: [String: String]) -> ModelTestSetResultsInFile {
        guard let raw = metadata[metadataKey] else { return .notRecorded }
        do {
            return .recorded(try JSONDecoder().decode(Envelope.self, from: Data(raw.utf8)).field)
        } catch let error as Envelope.ReadError {
            return .unreadable(error.description)
        } catch {
            return .unreadable("\(metadataKey) does not decode: \(error)")
        }
    }

    /// One-line description for the session log.
    var logSummary: String {
        switch self {
        case .failed(let reason):
            return "failed: \(reason)"
        case .evaluated(let results):
            return results.sets.map { set in
                "\(set.id) n=\(set.positions) top1=\(set.top1Correct) top5=\(set.top5Correct) "
                    + "nll=\(String(format: "%.4f", set.nll)) pElo=\(set.pElo.logText)"
            }.joined(separator: "; ")
        }
    }

    /// The JSON object, status and schema included.
    private struct Envelope: Codable {
        let field: ModelTestSetResultsField

        init(_ field: ModelTestSetResultsField) {
            self.field = field
        }

        enum Status: String, Codable {
            case evaluated
            case failed
        }

        enum CodingKeys: String, CodingKey {
            case schema
            case status
            case reason
            case evaluatedAtUnix = "evaluated_at_unix"
            case build
            case policyTailPrecision = "policy_tail_precision"
            case sets
        }

        struct ReadError: Error, CustomStringConvertible {
            let description: String
        }

        init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            let schema = try container.decode(Int.self, forKey: .schema)
            guard schema == ModelTestSetResultsField.schema else {
                throw ReadError(description: "\(ModelTestSetResultsField.metadataKey) schema \(schema) is not one this build reads (\(ModelTestSetResultsField.schema))")
            }
            switch try container.decode(Status.self, forKey: .status) {
            case .failed:
                field = .failed(reason: try container.decode(String.self, forKey: .reason))
            case .evaluated:
                field = .evaluated(ModelTestSetResults(
                    evaluatedAtUnix: try container.decode(Int64.self, forKey: .evaluatedAtUnix),
                    build: try container.decode(Int.self, forKey: .build),
                    policyTailPrecision: try container.decode(String.self, forKey: .policyTailPrecision),
                    sets: try container.decode([ModelTestSetResults.SetResult].self, forKey: .sets)
                ))
            }
        }

        func encode(to encoder: Encoder) throws {
            var container = encoder.container(keyedBy: CodingKeys.self)
            try container.encode(ModelTestSetResultsField.schema, forKey: .schema)
            switch field {
            case .failed(let reason):
                try container.encode(Status.failed, forKey: .status)
                try container.encode(reason, forKey: .reason)
            case .evaluated(let results):
                try container.encode(Status.evaluated, forKey: .status)
                try container.encode(results.evaluatedAtUnix, forKey: .evaluatedAtUnix)
                try container.encode(results.build, forKey: .build)
                try container.encode(results.policyTailPrecision, forKey: .policyTailPrecision)
                try container.encode(results.sets, forKey: .sets)
            }
        }
    }
}

/// A file's test-set results as read: absent (older files), unreadable
/// (reported, never fatal), or what the writer recorded.
enum ModelTestSetResultsInFile: Equatable, Sendable {
    case notRecorded
    case unreadable(String)
    case recorded(ModelTestSetResultsField)
}

/// The results of one evaluation: every test set, in
/// `LichessProbeData.modelFileTestSets` order.
struct ModelTestSetResults: Equatable, Sendable {
    let evaluatedAtUnix: Int64
    /// The app build that evaluated (results can shift with the probe code).
    let build: Int
    /// The policy tail the evaluating network was built with (the file's
    /// own, `NetworkArchitecture.policyTailPrecision`).
    let policyTailPrecision: String
    let sets: [SetResult]

    /// The set with the most positions (ties: the first), the one a picker
    /// summarizes. Nil only for a record with no sets.
    var largestSet: SetResult? {
        sets.reduce(nil) { best, set in
            guard let best else { return set }
            return set.positions > best.positions ? set : best
        }
    }

    struct SetResult: Codable, Equatable, Sendable {
        let id: String
        let title: String
        let description: String
        /// `ProbeTestSet.fingerprintSHA256`: results are comparable only
        /// between equal fingerprints.
        let fingerprintSHA256: String
        let positions: Int
        /// Positions whose right move is the network's top legal move.
        let top1Correct: Int
        /// Positions whose right move is among its top five legal moves
        /// (includes `top1Correct`).
        let top5Correct: Int
        /// Mean legal-masked probability on the right move.
        let avgCorrectProbability: Double
        /// Mean 1-based rank of the right move among the legal moves.
        let avgCorrectRank: Double
        /// Mean `−log(p)` of the right move, nats (`ProbeBookmoveNLL`).
        let nll: Double
        let pElo: PuzzleElo
        /// Per theme, in `ProbeCategory.allCases` order; themes the set lacks
        /// are absent.
        let themes: [ThemeResult]

        var top1Fraction: Double { positions > 0 ? Double(top1Correct) / Double(positions) : 0 }
        var top5Fraction: Double { positions > 0 ? Double(top5Correct) / Double(positions) : 0 }

        enum CodingKeys: String, CodingKey {
            case id, title, description
            case fingerprintSHA256 = "fingerprint_sha256"
            case positions
            case top1Correct = "top1_correct"
            case top5Correct = "top5_correct"
            case avgCorrectProbability = "avg_correct_probability"
            case avgCorrectRank = "avg_correct_rank"
            case nll
            case pElo = "pelo"
            case pEloBound = "pelo_bound"
            case themes
        }

        init(id: String, title: String, description: String, fingerprintSHA256: String, positions: Int,
             top1Correct: Int, top5Correct: Int, avgCorrectProbability: Double, avgCorrectRank: Double,
             nll: Double, pElo: PuzzleElo, themes: [ThemeResult]) {
            self.id = id
            self.title = title
            self.description = description
            self.fingerprintSHA256 = fingerprintSHA256
            self.positions = positions
            self.top1Correct = top1Correct
            self.top5Correct = top5Correct
            self.avgCorrectProbability = avgCorrectProbability
            self.avgCorrectRank = avgCorrectRank
            self.nll = nll
            self.pElo = pElo
            self.themes = themes
        }

        init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            id = try container.decode(String.self, forKey: .id)
            title = try container.decode(String.self, forKey: .title)
            description = try container.decode(String.self, forKey: .description)
            fingerprintSHA256 = try container.decode(String.self, forKey: .fingerprintSHA256)
            positions = try container.decode(Int.self, forKey: .positions)
            top1Correct = try container.decode(Int.self, forKey: .top1Correct)
            top5Correct = try container.decode(Int.self, forKey: .top5Correct)
            avgCorrectProbability = try container.decode(Double.self, forKey: .avgCorrectProbability)
            avgCorrectRank = try container.decode(Double.self, forKey: .avgCorrectRank)
            nll = try container.decode(Double.self, forKey: .nll)
            let estimate = try container.decodeNil(forKey: .pElo) ? nil : try container.decode(Double.self, forKey: .pElo)
            let bound = try container.decodeNil(forKey: .pEloBound) ? nil : try container.decode(PuzzleElo.Bound.self, forKey: .pEloBound)
            switch (estimate, bound) {
            case (let value?, nil): pElo = .estimate(value)
            case (nil, .allCorrect?): pElo = .allCorrect
            case (nil, .allWrong?): pElo = .allWrong
            default:
                throw DecodingError.dataCorruptedError(forKey: .pElo, in: container,
                    debugDescription: "exactly one of pelo and pelo_bound must be set")
            }
            themes = try container.decode([ThemeResult].self, forKey: .themes)
        }

        func encode(to encoder: Encoder) throws {
            var container = encoder.container(keyedBy: CodingKeys.self)
            try container.encode(id, forKey: .id)
            try container.encode(title, forKey: .title)
            try container.encode(description, forKey: .description)
            try container.encode(fingerprintSHA256, forKey: .fingerprintSHA256)
            try container.encode(positions, forKey: .positions)
            try container.encode(top1Correct, forKey: .top1Correct)
            try container.encode(top5Correct, forKey: .top5Correct)
            try container.encode(avgCorrectProbability, forKey: .avgCorrectProbability)
            try container.encode(avgCorrectRank, forKey: .avgCorrectRank)
            try container.encode(nll, forKey: .nll)
            switch pElo {
            case .estimate(let value):
                try container.encode(value, forKey: .pElo)
                try container.encodeNil(forKey: .pEloBound)
            case .allCorrect:
                try container.encodeNil(forKey: .pElo)
                try container.encode(PuzzleElo.Bound.allCorrect, forKey: .pEloBound)
            case .allWrong:
                try container.encodeNil(forKey: .pElo)
                try container.encode(PuzzleElo.Bound.allWrong, forKey: .pEloBound)
            }
            try container.encode(themes, forKey: .themes)
        }
    }

    /// The puzzle Elo fit (`LichessProbeHistory.mlePuzzleElo`). Every
    /// answer right or every answer wrong has no finite estimate; JSON has
    /// no infinity, so those are recorded as bounds.
    enum PuzzleElo: Equatable, Sendable {
        case estimate(Double)
        case allCorrect
        case allWrong

        enum Bound: String, Codable {
            case allCorrect = "all_correct"
            case allWrong = "all_wrong"
        }

        var logText: String {
            switch self {
            case .estimate(let value): return String(format: "%.1f", value)
            case .allCorrect: return "above every puzzle (all correct)"
            case .allWrong: return "below every puzzle (all wrong)"
            }
        }
    }

    struct ThemeResult: Codable, Equatable, Sendable {
        /// The Lichess theme id (`ProbeCategory.lichessThemeID`).
        let id: String
        let title: String
        /// Positions in this theme whose right move is the top legal move.
        let correct: Int
        let total: Int
    }
}

/// Evaluates a model's weights for a writer that has no evaluator of its own
/// (`ModelDerivation.derive`, `ModelGraft.graft`, which do no GPU work
/// themselves): the caller injects one, usually `ModelTestSetEvaluator`
/// through a synchronous bridge.
typealias ModelTestSetEvaluation = (_ weights: [[Float]], _ architecture: NetworkArchitecture) throws -> ModelTestSetResultsField

extension ModelTestSetResultsField {
    /// `metadata` plus the results key, for a writer that assembles the
    /// tensors and metadata itself. The weights are evaluated as a load would
    /// read them: the file is encoded without the key and decoded through
    /// the normal loader (value-head handling included), in memory.
    static func metadata(_ metadata: [String: String], addingResultsFor tensors: [SafetensorsTensor],
                         evaluation: ModelTestSetEvaluation) throws -> [String: String] {
        var withoutResults = metadata
        withoutResults[metadataKey] = nil
        let provisional = try SafetensorsFile.encode(tensors: tensors, metadata: withoutResults)
        let decoded = try SafetensorsModelIO.decode(provisional)
        var withResults = withoutResults
        withResults[metadataKey] = try evaluation(decoded.file.weights, decoded.architecture).metadataValue()
        return withResults
    }
}
