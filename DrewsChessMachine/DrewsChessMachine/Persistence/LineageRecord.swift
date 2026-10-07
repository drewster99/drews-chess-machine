//
//  LineageRecord.swift
//  DrewsChessMachine
//
//  Where a file's weights came from, carried inside the file itself.
//
//  Every model file written at architecture format v7 or later carries one
//  `LineageRecord` as a single JSON value under `__metadata__["dcm_lineage"]`
//  (and a `.dcmsession`'s `session.json` embeds the same record). It answers
//  the questions the run-tracking dashboards used to reconstruct by hand from
//  logs and hand-entered registry bases: which run and segment wrote this
//  file, what it was trained from, how many trainer steps, games and
//  positions the weights have seen in total, how much measured training time
//  went into them, under which parameters, build and machine.
//
//  Totals are continuous across sessions. A segment that continues a run by
//  exact resume starts its totals from the parent file's record; a branch
//  (`--start-model` without `--resume-exact`, or a GUI fork from a champion)
//  starts a new run whose totals restart with the trainer's fresh clock and
//  records the parent it forked from. A total that a predecessor never
//  recorded — a resume of a file written before lineage existed — is
//  written as JSON `null` and stays `null` for the rest of that run: a total
//  is never guessed or reconstructed, because a modeled number in a measured
//  column still looks plottable.
//
//  Strictness: every key is required on decode. An optional value is encoded
//  as an explicit `null`, so a record missing a key is a writer bug that
//  fails loudly rather than reading as "unrecorded".
//
//  Schema 3 (hyperparameter recording plan, Part S) makes one file state how
//  its weights were trained across every segment and run behind them, not
//  only the segment that wrote it: each earlier segment's summary carries its
//  parameters, configuration, corpus, argv and seeds; `ancestry` carries the
//  runs a branch or derive left; `configuration` holds what no parameter
//  states (policy-tail precision, budget, game generation, live edits, the
//  champions that generated the data); `run_seeds` keeps the actual seeds
//  where `rng.streams` is dropped. Schema-2 records still decode (owner
//  decision O-23): the facts they never stored read back as explicitly
//  unrecorded (`Recorded.unrecorded`), never filled in, and every write —
//  including a champion file or derive of a schema-2 model — is schema 3.
//  The in-memory record always has the schema-3 shape; only decoding knows
//  the schema a file was written at.
//
//  The JSON value is the single source of truth. The flat `__metadata__`
//  mirror keys (`lineage_run_id`, `cum_trainer_step`, …) are derived from it
//  at write time for grep and header-scanning tools and are never read back.
//

import Foundation
import CryptoKit

/// Provenance of a model file's weights — see the file comment.
struct LineageRecord: Codable, Equatable, Sendable {
    /// Schema of the record itself, independent of the architecture format.
    /// Schema 2 records the run's random streams (`rng.streams`) and the
    /// corpus feed phase and shard identities (`fed.corpus`) an exact resume
    /// continues from. Schema 3 adds the segment configuration, run seeds,
    /// ancestry, per-segment summaries of every earlier segment's
    /// configuration, the corpus identity of mixed corpora and the build's
    /// diff hash and toolchain (see the file comment).
    static let currentSchema = 3
    /// The oldest schema this build reads (owner decision O-23). Schema 1
    /// stays refused.
    static let oldestDecodableSchema = 2

    let schema: Int
    let run: Run
    /// The file this segment's weights came from; nil only when the weights
    /// were initialized by this run (a fresh build or mint).
    let parent: Parent?
    let steps: Steps
    let fed: Fed
    let time: Time
    /// The training parameters in force when the file's weights were last
    /// trained; nil when none were recorded — a fresh mint, or a derive of a
    /// source that did not record them.
    let parameters: Parameters?
    let build: Build
    let invocation: Invocation
    let device: Device
    let rng: RNG
    /// Earlier segments of this run, oldest first; the segment that wrote
    /// this file is described by `run`, `steps`, `fed` and `time`.
    let segments: [SegmentSummary]
    /// Every `--derive-model` step the weights went through, oldest first,
    /// carried verbatim by every later save of the line — training saves
    /// continue a history, they never append to it; only a derive does.
    /// Empty when the line records no derivation; a line that continues a
    /// history earlier files did not record says so with
    /// `run.continuesUnrecordedHistory`.
    let derivationHistory: [ModelDerivation.DerivationRecord]
    /// How the segment that last trained these weights trained beyond its
    /// parameter snapshot; `.notTrained` exactly when `parameters` is nil,
    /// unrecorded when carried from a schema-2 record.
    let configuration: RecordedIfTrained<TrainingConfiguration>
    /// The seeds the run trained under, oldest first (gap 9b); `.notTrained`
    /// exactly when `parameters` is nil.
    let runSeeds: RecordedIfTrained<[RunSeedEntry]>
    /// The earlier runs these weights descend from (gaps 1b, 1c, B3).
    let ancestry: Ancestry

    // MARK: Run

    /// How the segment that wrote the file began.
    enum SegmentStart: String, Codable, Sendable {
        /// Weights initialized by this run.
        case fresh
        /// Trained from another file's weights with a fresh trainer clock:
        /// a new run.
        case branch
        /// Continued another file's run by exact resume (trainer clock,
        /// optimizer state and totals carried on).
        case resume
        /// A new model made from another file without training: a
        /// `--derive-model` output, or a GUI save of a loaded model before
        /// any training.
        case derive
    }

    struct Run: Codable, Equatable, Sendable {
        /// Minted when a run starts (fresh, branch, derive, or a resume of a
        /// file without a lineage); inherited by every exact resume.
        let lineageRunID: String
        /// 0 for the segment that started the run, +1 per exact resume.
        let segmentIndex: Int
        /// Minted per process (per segment).
        let segmentID: String
        let segmentStartedUnix: Int64
        let start: SegmentStart
        /// Whether this segment restored its parent's complete training
        /// state; false lists what it did not restore in `notExactItems`.
        let exactResume: Bool
        let notExactItems: [String]
        /// True when the run continues weights whose earlier history was
        /// never recorded (an exact resume of a file written before lineage
        /// existed): totals the predecessor did not carry are `null`.
        let continuesUnrecordedHistory: Bool
        /// When this record was taken (the save time).
        let recordedUnix: Int64

        enum CodingKeys: String, CodingKey {
            case lineageRunID = "lineage_run_id"
            case segmentIndex = "segment_index"
            case segmentID = "segment_id"
            case segmentStartedUnix = "segment_started_unix"
            case start
            case exactResume = "exact_resume"
            case notExactItems = "not_exact_items"
            case continuesUnrecordedHistory = "continues_unrecorded_history"
            case recordedUnix = "recorded_unix"
        }
    }

    // MARK: Parent

    struct Parent: Codable, Equatable, Sendable {
        let modelID: String
        /// The parent file's `content_sha256` — what identifies the exact
        /// file even when model IDs collide. Null when the parent weights
        /// were never written to a file (an in-memory champion).
        let contentSHA256: String?
        /// The parent trainer's step clock, when the parent recorded one.
        let trainerCompletedSteps: Int?
        /// The parent's run and segment, when the parent carried a lineage.
        let lineageRunID: String?
        let segmentID: String?

        enum CodingKeys: String, CodingKey {
            case modelID = "model_id"
            case contentSHA256 = "content_sha256"
            case trainerCompletedSteps = "trainer_completed_steps"
            case lineageRunID = "lineage_run_id"
            case segmentID = "segment_id"
        }

        init(modelID: String, contentSHA256: String?, trainerCompletedSteps: Int?, lineageRunID: String?, segmentID: String?) {
            self.modelID = modelID
            self.contentSHA256 = contentSHA256
            self.trainerCompletedSteps = trainerCompletedSteps
            self.lineageRunID = lineageRunID
            self.segmentID = segmentID
        }

        init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            modelID = try c.decode(String.self, forKey: .modelID)
            contentSHA256 = try c.decode(String?.self, forKey: .contentSHA256)
            trainerCompletedSteps = try c.decode(Int?.self, forKey: .trainerCompletedSteps)
            lineageRunID = try c.decode(String?.self, forKey: .lineageRunID)
            segmentID = try c.decode(String?.self, forKey: .segmentID)
        }

        func encode(to encoder: Encoder) throws {
            var c = encoder.container(keyedBy: CodingKeys.self)
            try c.encode(modelID, forKey: .modelID)
            try c.encode(contentSHA256, forKey: .contentSHA256)
            try c.encode(trainerCompletedSteps, forKey: .trainerCompletedSteps)
            try c.encode(lineageRunID, forKey: .lineageRunID)
            try c.encode(segmentID, forKey: .segmentID)
        }
    }

    // MARK: Steps

    struct Steps: Codable, Equatable, Sendable {
        /// This run's total trainer steps (B4, owner decision O-15): a
        /// branch restarts the totals with its fresh trainer clock, a derive
        /// or graft continues its source's. The steps behind the weights
        /// across runs add the `totals_at_departure` of each ancestor left by
        /// a branch (`scripts/dcm_lineage.py` `weights_totals`). On a
        /// trainer-state file it equals `trainer_completed_steps` (the writer
        /// refuses otherwise). Null only when no predecessor recorded it.
        let cumTrainerStep: Int?
        /// The trainer clock when this segment began — what the dashboards
        /// used to enter by hand as `cumstep_base`.
        let segmentStartTrainerStep: Int?
        /// Steps this segment trained — the sidecar to the file's
        /// `training_step`, which from format v11 is the trainer step (on CLI
        /// files before v11 `training_step` held this value). Mirrored flat as
        /// `lineage_segment_local_step`.
        let segmentLocalStep: Int

        enum CodingKeys: String, CodingKey {
            case cumTrainerStep = "cum_trainer_step"
            case segmentStartTrainerStep = "segment_start_trainer_step"
            case segmentLocalStep = "segment_local_step"
        }

        init(cumTrainerStep: Int?, segmentStartTrainerStep: Int?, segmentLocalStep: Int) {
            self.cumTrainerStep = cumTrainerStep
            self.segmentStartTrainerStep = segmentStartTrainerStep
            self.segmentLocalStep = segmentLocalStep
        }

        init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            cumTrainerStep = try c.decode(Int?.self, forKey: .cumTrainerStep)
            segmentStartTrainerStep = try c.decode(Int?.self, forKey: .segmentStartTrainerStep)
            segmentLocalStep = try c.decode(Int.self, forKey: .segmentLocalStep)
        }

        func encode(to encoder: Encoder) throws {
            var c = encoder.container(keyedBy: CodingKeys.self)
            try c.encode(cumTrainerStep, forKey: .cumTrainerStep)
            try c.encode(segmentStartTrainerStep, forKey: .segmentStartTrainerStep)
            try c.encode(segmentLocalStep, forKey: .segmentLocalStep)
        }
    }

    // MARK: Fed

    /// Games and positions that entered the replay buffer (corpus games fed,
    /// or self-play / vs-UCI games recorded) — the device-independent compute
    /// axis.
    struct Fed: Codable, Equatable, Sendable {
        let cumGames: Int?
        let cumPositions: Int?
        let segmentGames: Int
        let segmentPositions: Int
        /// Corpus-replay position, the resume point of `--resume-exact`;
        /// nil on every other path.
        let corpus: CorpusPosition?

        enum CodingKeys: String, CodingKey {
            case cumGames = "cum_games"
            case cumPositions = "cum_positions"
            case segmentGames = "segment_games"
            case segmentPositions = "segment_positions"
            case corpus
        }

        init(cumGames: Int?, cumPositions: Int?, segmentGames: Int, segmentPositions: Int, corpus: CorpusPosition?) {
            self.cumGames = cumGames
            self.cumPositions = cumPositions
            self.segmentGames = segmentGames
            self.segmentPositions = segmentPositions
            self.corpus = corpus
        }

        init(from decoder: Decoder) throws {
            try self.init(from: decoder, schema: LineageRecord.currentSchema)
        }

        init(from decoder: Decoder, schema: Int) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            cumGames = try c.decode(Int?.self, forKey: .cumGames)
            cumPositions = try c.decode(Int?.self, forKey: .cumPositions)
            segmentGames = try c.decode(Int.self, forKey: .segmentGames)
            segmentPositions = try c.decode(Int.self, forKey: .segmentPositions)
            if try c.decodeNil(forKey: .corpus) {
                corpus = nil
            } else {
                corpus = try CorpusPosition(from: try c.superDecoder(forKey: .corpus), schema: schema)
            }
        }

        func encode(to encoder: Encoder) throws {
            var c = encoder.container(keyedBy: CodingKeys.self)
            try c.encode(cumGames, forKey: .cumGames)
            try c.encode(cumPositions, forKey: .cumPositions)
            try c.encode(segmentGames, forKey: .segmentGames)
            try c.encode(segmentPositions, forKey: .segmentPositions)
            try c.encode(corpus, forKey: .corpus)
        }
    }

    /// Which corpora a replay fed, in feed order (gap 7).
    enum CorpusIdentity: Codable, Equatable, Sendable {
        /// Every corpus, with its sealed-shard count: their counts sum to
        /// `shard_sha256`'s length.
        case listed([CorpusEntry])
        /// Only the first corpus is known: a position carried from a schema-2
        /// record, which named no other.
        case firstOnly(id: String, path: String)

        struct CorpusEntry: Codable, Equatable, Sendable {
            let corpusID: String
            let corpusPath: String
            let shardCount: Int

            enum CodingKeys: String, CodingKey {
                case corpusID = "corpus_id"
                case corpusPath = "corpus_path"
                case shardCount = "shard_count"
            }
        }

        private struct FirstOnly: Codable, Equatable {
            let corpusID: String
            let corpusPath: String

            enum CodingKeys: String, CodingKey {
                case corpusID = "corpus_id"
                case corpusPath = "corpus_path"
            }
        }

        private enum CodingKeys: String, CodingKey {
            case listed
            case firstOnly = "first_only"
        }

        init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            switch (c.contains(.listed), c.contains(.firstOnly)) {
            case (true, false):
                let entries = try c.decode([CorpusEntry].self, forKey: .listed)
                guard !entries.isEmpty else {
                    throw DecodingError.dataCorruptedError(forKey: .listed, in: c, debugDescription: "an empty corpus list")
                }
                self = .listed(entries)
            case (false, true):
                let first = try c.decode(FirstOnly.self, forKey: .firstOnly)
                self = .firstOnly(id: first.corpusID, path: first.corpusPath)
            default:
                throw DecodingError.dataCorrupted(.init(
                    codingPath: c.codingPath, debugDescription: "corpus_identity holds exactly one of listed / first_only"))
            }
        }

        func encode(to encoder: Encoder) throws {
            var c = encoder.container(keyedBy: CodingKeys.self)
            switch self {
            case .listed(let entries):
                try c.encode(entries, forKey: .listed)
            case .firstOnly(let id, let path):
                try c.encode(FirstOnly(corpusID: id, corpusPath: path), forKey: .firstOnly)
            }
        }

        /// The first corpus fed, which a resume checks against.
        var firstCorpusID: String {
            switch self {
            case .listed(let entries): return entries[0].corpusID
            case .firstOnly(let id, _): return id
            }
        }

        var firstCorpusPath: String {
            switch self {
            case .listed(let entries): return entries[0].corpusPath
            case .firstOnly(_, let path): return path
            }
        }
    }

    /// A point in a corpus feed: the epoch and the within-epoch index of the
    /// next game.
    struct FeedPoint: Codable, Equatable, Sendable {
        let epoch: Int
        let nextGameIndex: Int

        enum CodingKeys: String, CodingKey {
            case epoch
            case nextGameIndex = "next_game_index"
        }
    }

    /// Where a corpus-replay run stood in its corpus at the save.
    struct CorpusPosition: Codable, Equatable, Sendable {
        /// The corpora fed, in feed order.
        let corpusIdentity: CorpusIdentity
        /// Where this segment's feed began (after `--start-shard` /
        /// `--start-game-index` / a resume); unrecorded only on a position
        /// carried from a schema-2 record.
        let segmentStart: Recorded<FeedPoint>
        let epoch: Int
        /// Within-epoch index of the next game the run would have fed.
        let nextGameIndex: Int
        let shard: Int
        /// Plies resident in the replay buffer at the save.
        let populatedPlies: Int
        let bufferCapacity: Int
        /// Positions fed beyond the feed target of the next untrained step
        /// (`positions fed − (feed base + step × feedPerStep)`; negative
        /// while that step's feed is still owed). Whole games overshoot their
        /// target, so this phase is state: a resume continues it, so the
        /// games fed before every later step are the uninterrupted run's.
        let feedAheadPositions: Int
        /// Positions the run feeds per trainer step.
        let feedPerStep: Int
        /// Each sealed shard's trailer SHA-256 (hex), in feed order — what a
        /// resume checks the corpus against before refeeding any of it.
        let shardSHA256: [String]

        enum CodingKeys: String, CodingKey {
            case corpusID = "corpus_id"
            case corpusPath = "corpus_path"
            case corpusIdentity = "corpus_identity"
            case segmentStart = "segment_start"
            case epoch
            case nextGameIndex = "next_game_index"
            case shard
            case populatedPlies = "populated_plies"
            case bufferCapacity = "buffer_capacity"
            case feedAheadPositions = "feed_ahead_positions"
            case feedPerStep = "feed_per_step"
            case shardSHA256 = "shard_sha256"
        }

        init(corpusIdentity: CorpusIdentity, segmentStart: Recorded<FeedPoint>, epoch: Int, nextGameIndex: Int,
             shard: Int, populatedPlies: Int, bufferCapacity: Int, feedAheadPositions: Int, feedPerStep: Int,
             shardSHA256: [String]) throws {
            self.corpusIdentity = corpusIdentity
            self.segmentStart = segmentStart
            self.epoch = epoch
            self.nextGameIndex = nextGameIndex
            self.shard = shard
            self.populatedPlies = populatedPlies
            self.bufferCapacity = bufferCapacity
            self.feedAheadPositions = feedAheadPositions
            self.feedPerStep = feedPerStep
            self.shardSHA256 = shardSHA256
            try checkShardCounts()
        }

        init(from decoder: Decoder) throws {
            try self.init(from: decoder, schema: LineageRecord.currentSchema)
        }

        init(from decoder: Decoder, schema: Int) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            if schema >= 3 {
                try LineageRecord.refuseKeys([.corpusID, .corpusPath], in: c,
                                             reason: "a schema-\(schema) corpus position names its corpora in corpus_identity")
                corpusIdentity = try c.decode(CorpusIdentity.self, forKey: .corpusIdentity)
                segmentStart = try c.decode(Recorded<FeedPoint>.self, forKey: .segmentStart)
            } else {
                try LineageRecord.refuseSchemaThreeKeys([.corpusIdentity, .segmentStart], in: c, schema: schema)
                corpusIdentity = .firstOnly(id: try c.decode(String.self, forKey: .corpusID),
                                            path: try c.decode(String.self, forKey: .corpusPath))
                segmentStart = .unrecorded
            }
            epoch = try c.decode(Int.self, forKey: .epoch)
            nextGameIndex = try c.decode(Int.self, forKey: .nextGameIndex)
            shard = try c.decode(Int.self, forKey: .shard)
            populatedPlies = try c.decode(Int.self, forKey: .populatedPlies)
            bufferCapacity = try c.decode(Int.self, forKey: .bufferCapacity)
            feedAheadPositions = try c.decode(Int.self, forKey: .feedAheadPositions)
            feedPerStep = try c.decode(Int.self, forKey: .feedPerStep)
            shardSHA256 = try c.decode([String].self, forKey: .shardSHA256)
            do {
                try checkShardCounts()
            } catch {
                throw DecodingError.dataCorruptedError(forKey: .corpusIdentity, in: c, debugDescription: "\(error)")
            }
        }

        func encode(to encoder: Encoder) throws {
            try checkShardCounts()
            var c = encoder.container(keyedBy: CodingKeys.self)
            try c.encode(corpusIdentity, forKey: .corpusIdentity)
            try c.encode(segmentStart, forKey: .segmentStart)
            try c.encode(epoch, forKey: .epoch)
            try c.encode(nextGameIndex, forKey: .nextGameIndex)
            try c.encode(shard, forKey: .shard)
            try c.encode(populatedPlies, forKey: .populatedPlies)
            try c.encode(bufferCapacity, forKey: .bufferCapacity)
            try c.encode(feedAheadPositions, forKey: .feedAheadPositions)
            try c.encode(feedPerStep, forKey: .feedPerStep)
            try c.encode(shardSHA256, forKey: .shardSHA256)
        }

        /// A listed identity's shard counts sum to the shard hashes.
        private func checkShardCounts() throws {
            guard case .listed(let entries) = corpusIdentity else { return }
            let total = entries.reduce(0) { $0 + $1.shardCount }
            guard total == shardSHA256.count else {
                throw SchemaError.invariant(
                    "corpus_identity lists \(total) shards but shard_sha256 holds \(shardSHA256.count)")
            }
        }
    }

    // MARK: Time

    struct Time: Codable, Equatable, Sendable {
        /// Sum of measured trainer-step durations behind these weights, in
        /// seconds — training time with idle, pauses and sleep excluded by
        /// construction.
        let cumTrainStepSec: Double?
        /// Sum of each segment's wall time from its start to its last save.
        let cumWallSec: Double?
        let segmentTrainStepSec: Double
        let segmentWallSec: Double

        enum CodingKeys: String, CodingKey {
            case cumTrainStepSec = "cum_train_step_sec"
            case cumWallSec = "cum_wall_sec"
            case segmentTrainStepSec = "segment_train_step_sec"
            case segmentWallSec = "segment_wall_sec"
        }

        init(cumTrainStepSec: Double?, cumWallSec: Double?, segmentTrainStepSec: Double, segmentWallSec: Double) {
            self.cumTrainStepSec = cumTrainStepSec
            self.cumWallSec = cumWallSec
            self.segmentTrainStepSec = segmentTrainStepSec
            self.segmentWallSec = segmentWallSec
        }

        init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            cumTrainStepSec = try c.decode(Double?.self, forKey: .cumTrainStepSec)
            cumWallSec = try c.decode(Double?.self, forKey: .cumWallSec)
            segmentTrainStepSec = try c.decode(Double.self, forKey: .segmentTrainStepSec)
            segmentWallSec = try c.decode(Double.self, forKey: .segmentWallSec)
        }

        func encode(to encoder: Encoder) throws {
            var c = encoder.container(keyedBy: CodingKeys.self)
            try c.encode(cumTrainStepSec, forKey: .cumTrainStepSec)
            try c.encode(cumWallSec, forKey: .cumWallSec)
            try c.encode(segmentTrainStepSec, forKey: .segmentTrainStepSec)
            try c.encode(segmentWallSec, forKey: .segmentWallSec)
        }
    }

    // MARK: Parameters

    /// The full training-parameter set, as the flat snake_case JSON object
    /// `--show-default-parameters` prints (sorted keys, compact), kept as
    /// that exact text so it round-trips byte for byte.
    struct Parameters: Codable, Equatable, Sendable {
        let snapshotJSON: String
        /// SHA-256 (hex) of `snapshotJSON`'s UTF-8 bytes.
        let sha256: String

        enum CodingKeys: String, CodingKey {
            case snapshotJSON = "snapshot_json"
            case sha256
        }

        /// The seed *settings* (gap 9). A snapshot this build composes omits
        /// them (`TrainingParametersSnapshot.lineageValues()`; the tracker
        /// refuses one that holds them): the run's actual seed is in
        /// `rng.streams` and `run_seeds`, and a setting `--seed` or a resume
        /// overrode only contradicted it. A carried (schema-2) snapshot keeps
        /// them, exactly as written.
        static let excludedParameterIDs = [RandomSeedModeParameter.id, RandomSeed.id]

        /// Snapshot of `values` (parameter id → value).
        init(values: [String: ParameterValue]) throws {
            var dictionary: [String: Any] = [:]
            for (id, value) in values {
                switch value {
                case .bool(let x): dictionary[id] = x
                case .int(let x): dictionary[id] = x
                case .double(let x): dictionary[id] = x
                // A decimal string, as parameters.json writes it, so a seed
                // above 2^53 survives JSON number parsing.
                case .uint64(let x): dictionary[id] = String(x)
                }
            }
            let data = try JSONSerialization.data(withJSONObject: dictionary, options: [.sortedKeys])
            let text = String(decoding: data, as: UTF8.self)
            self.snapshotJSON = text
            self.sha256 = Self.hash(text)
        }

        init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            snapshotJSON = try c.decode(String.self, forKey: .snapshotJSON)
            sha256 = try c.decode(String.self, forKey: .sha256)
            let computed = Self.hash(snapshotJSON)
            guard computed == sha256 else {
                throw DecodingError.dataCorruptedError(
                    forKey: .sha256, in: c,
                    debugDescription: "parameters sha256 \(sha256) does not match its snapshot (\(computed))")
            }
        }

        static func hash(_ text: String) -> String {
            SHA256.hash(data: Data(text.utf8)).map { String(format: "%02x", $0) }.joined()
        }
    }

    // MARK: Build, invocation, device

    struct Build: Codable, Equatable, Sendable {
        let buildNumber: Int
        let gitHash: String
        let gitBranch: String
        /// Whether the compiled project differed from `gitHash`. From build
        /// generation P1 on, its scope is `DrewsChessMachine/` without the
        /// generated files; a schema-2 record written before then says true
        /// for every build.
        let gitDirty: Bool
        /// SHA-256 of the built `DrewsChessMachine/` tree's git id
        /// (`BuildInfo.gitDiffSHA256`): recorded `null` exactly when the
        /// build was clean; unrecorded on a schema-2 build.
        let gitDiffSHA256: Recorded<String?>
        /// Xcode's, the SDK's build versions and the build configuration
        /// (`BuildInfo`); unrecorded on a schema-2 build.
        let xcodeBuild: Recorded<String>
        let sdkBuild: Recorded<String>
        let configuration: Recorded<String>

        enum CodingKeys: String, CodingKey, CaseIterable {
            case buildNumber = "build_number"
            case gitHash = "git_hash"
            case gitBranch = "git_branch"
            case gitDirty = "git_dirty"
            case gitDiffSHA256 = "git_diff_sha256"
            case xcodeBuild = "xcode_build"
            case sdkBuild = "sdk_build"
            case configuration
        }

        /// The keys schema 3 added.
        static let schemaThreeKeys: [CodingKeys] = [.gitDiffSHA256, .xcodeBuild, .sdkBuild, .configuration]

        init(buildNumber: Int, gitHash: String, gitBranch: String, gitDirty: Bool, gitDiffSHA256: Recorded<String?>,
             xcodeBuild: Recorded<String>, sdkBuild: Recorded<String>, configuration: Recorded<String>) throws {
            self.buildNumber = buildNumber
            self.gitHash = gitHash
            self.gitBranch = gitBranch
            self.gitDirty = gitDirty
            self.gitDiffSHA256 = gitDiffSHA256
            self.xcodeBuild = xcodeBuild
            self.sdkBuild = sdkBuild
            self.configuration = configuration
            try checkDiffHash()
        }

        init(from decoder: Decoder) throws {
            try self.init(from: decoder, schema: LineageRecord.currentSchema)
        }

        init(from decoder: Decoder, schema: Int) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            buildNumber = try c.decode(Int.self, forKey: .buildNumber)
            gitHash = try c.decode(String.self, forKey: .gitHash)
            gitBranch = try c.decode(String.self, forKey: .gitBranch)
            gitDirty = try c.decode(Bool.self, forKey: .gitDirty)
            if schema >= 3 {
                gitDiffSHA256 = try c.decode(Recorded<String?>.self, forKey: .gitDiffSHA256)
                xcodeBuild = try c.decode(Recorded<String>.self, forKey: .xcodeBuild)
                sdkBuild = try c.decode(Recorded<String>.self, forKey: .sdkBuild)
                configuration = try c.decode(Recorded<String>.self, forKey: .configuration)
            } else {
                try LineageRecord.refuseSchemaThreeKeys(Self.schemaThreeKeys, in: c, schema: schema)
                gitDiffSHA256 = .unrecorded
                xcodeBuild = .unrecorded
                sdkBuild = .unrecorded
                configuration = .unrecorded
            }
            do {
                try checkDiffHash()
            } catch {
                throw DecodingError.dataCorruptedError(forKey: .gitDiffSHA256, in: c, debugDescription: "\(error)")
            }
        }

        func encode(to encoder: Encoder) throws {
            try checkDiffHash()
            var c = encoder.container(keyedBy: CodingKeys.self)
            try c.encode(buildNumber, forKey: .buildNumber)
            try c.encode(gitHash, forKey: .gitHash)
            try c.encode(gitBranch, forKey: .gitBranch)
            try c.encode(gitDirty, forKey: .gitDirty)
            try c.encode(gitDiffSHA256, forKey: .gitDiffSHA256)
            try c.encode(xcodeBuild, forKey: .xcodeBuild)
            try c.encode(sdkBuild, forKey: .sdkBuild)
            try c.encode(configuration, forKey: .configuration)
        }

        /// A recorded diff hash is null exactly when the build was clean.
        private func checkDiffHash() throws {
            guard case .recorded(let hash) = gitDiffSHA256 else { return }
            guard (hash == nil) == !gitDirty else {
                throw SchemaError.invariant("build git_diff_sha256 is null exactly when git_dirty is false")
            }
        }

        /// The running build.
        static var current: Build {
            get throws {
                try Build(buildNumber: BuildInfo.buildNumber, gitHash: BuildInfo.gitHash,
                          gitBranch: BuildInfo.gitBranch, gitDirty: BuildInfo.gitDirty,
                          gitDiffSHA256: .recorded(BuildInfo.gitDiffSHA256),
                          xcodeBuild: .recorded(BuildInfo.xcodeBuild), sdkBuild: .recorded(BuildInfo.sdkBuild),
                          configuration: .recorded(BuildInfo.configuration))
            }
        }
    }

    /// Which code path wrote the file.
    enum PathKind: String, Codable, Sendable {
        case gui
        case replay
        case vsuci
        case derive
        case newModel = "new_model"
    }

    struct Invocation: Codable, Equatable, Sendable {
        /// The process's arguments, with secret-bearing values redacted
        /// (`LineageRecord.redactedArguments`).
        let argv: [String]
        let pathKind: PathKind

        enum CodingKeys: String, CodingKey {
            case argv
            case pathKind = "path_kind"
        }
    }

    struct Device: Codable, Equatable, Sendable {
        let hardwareModel: String?
        let cpu: String?
        let isVirtualMachine: Bool?
        let osVersion: String
        let gpu: String?

        enum CodingKeys: String, CodingKey {
            case hardwareModel = "hw_model"
            case cpu = "chip"
            case isVirtualMachine = "is_vm"
            case osVersion = "os_version"
            case gpu = "gpu_name"
        }

        init(hardwareModel: String?, cpu: String?, isVirtualMachine: Bool?, osVersion: String, gpu: String?) {
            self.hardwareModel = hardwareModel
            self.cpu = cpu
            self.isVirtualMachine = isVirtualMachine
            self.osVersion = osVersion
            self.gpu = gpu
        }

        init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            hardwareModel = try c.decode(String?.self, forKey: .hardwareModel)
            cpu = try c.decode(String?.self, forKey: .cpu)
            isVirtualMachine = try c.decode(Bool?.self, forKey: .isVirtualMachine)
            osVersion = try c.decode(String.self, forKey: .osVersion)
            gpu = try c.decode(String?.self, forKey: .gpu)
        }

        func encode(to encoder: Encoder) throws {
            var c = encoder.container(keyedBy: CodingKeys.self)
            try c.encode(hardwareModel, forKey: .hardwareModel)
            try c.encode(cpu, forKey: .cpu)
            try c.encode(isVirtualMachine, forKey: .isVirtualMachine)
            try c.encode(osVersion, forKey: .osVersion)
            try c.encode(gpu, forKey: .gpu)
        }

        /// This Mac. A fact the system does not report is recorded as null
        /// (see `HardwareInfo`), never guessed.
        static var current: Device {
            let hardware = HardwareInfo.current
            return Device(
                hardwareModel: hardware.hardwareModel,
                cpu: hardware.cpuBrand,
                isVirtualMachine: hardware.isVirtualMachine,
                osVersion: ProcessInfo.processInfo.operatingSystemVersionString,
                gpu: hardware.gpuModel
            )
        }
    }

    // MARK: RNG

    /// How the run's randomness was seeded, and the random state a resume
    /// needs to continue the run's draws where they left off.
    struct RNG: Codable, Equatable, Sendable {
        /// The training graph's dropout Philox state when the saved trainer
        /// state was captured — the state the next step's masks start from.
        /// Present on every record written with a trainer snapshot (it is
        /// part of the same consistent cut as the weights and the clock);
        /// null for a record with no trainer state behind it (a fresh mint,
        /// a derived or untrained copy). A resume restores it, so the
        /// resumed run's masks continue the saved run's sequence.
        let dropoutPhiloxState: DropoutPhiloxState?
        /// The run's master seed and every stream position a resume
        /// continues; null for a record with no training run behind it.
        let streams: RunStreams?
        /// The init seed and scheme this run drew its starting weights with
        /// (`init_seed` / `init_scheme`, determinism plan B1.1): set on a run
        /// that started from fresh weights — a mint, or a training run with
        /// no start model — and carried by every later record of the run,
        /// across exact resumes. Null for a run that started from another
        /// file's weights (a branch, a derive or an untrained copy), whose
        /// own record says how they were drawn, and for a run continuing a
        /// file written before this field.
        let initialization: ModelInitRecord?
        /// The writing process's behavior fingerprint for the saved trainer's
        /// numerics (`BehaviorFingerprint`): a resume under another build or
        /// OS whose fingerprint matches it is not a `build` / `os` gap. Null
        /// for a record with no trainer state behind it.
        let behaviorFingerprint: BehaviorFingerprint.Record?

        enum CodingKeys: String, CodingKey {
            case dropoutPhiloxState = "dropout_philox_state"
            case streams
            case initSeed = "init_seed"
            case initScheme = "init_scheme"
            case behaviorFingerprint = "behavior_fingerprint"
        }

        /// The random state a save captures, with the fingerprint of the
        /// process that captured it. `initialization` is the run's, which
        /// `LineageTracker` fills in (`withInitialization`).
        init(dropoutPhiloxState: DropoutPhiloxState?, streams: RunStreams?,
             behaviorFingerprint: BehaviorFingerprint.Record?) {
            self.init(dropoutPhiloxState: dropoutPhiloxState, streams: streams, initialization: nil,
                      behaviorFingerprint: behaviorFingerprint)
        }

        private init(dropoutPhiloxState: DropoutPhiloxState?, streams: RunStreams?,
                     initialization: ModelInitRecord?, behaviorFingerprint: BehaviorFingerprint.Record?) {
            self.dropoutPhiloxState = dropoutPhiloxState
            self.streams = streams
            self.initialization = initialization
            self.behaviorFingerprint = behaviorFingerprint
        }

        /// This state with the run's init seed and scheme.
        func withInitialization(_ initialization: ModelInitRecord?) -> RNG {
            RNG(dropoutPhiloxState: dropoutPhiloxState, streams: streams, initialization: initialization,
                behaviorFingerprint: behaviorFingerprint)
        }

        init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            dropoutPhiloxState = try c.decode(DropoutPhiloxState?.self, forKey: .dropoutPhiloxState)
            streams = try c.decode(RunStreams?.self, forKey: .streams)
            behaviorFingerprint = try c.decode(BehaviorFingerprint.Record?.self, forKey: .behaviorFingerprint)
            let seedText = try c.decode(String?.self, forKey: .initSeed)
            let scheme = try c.decode(String?.self, forKey: .initScheme)
            switch (seedText, scheme) {
            case (nil, nil):
                initialization = nil
            case let (seedText?, scheme?):
                guard let seed = UInt64(strictDecimal: seedText) else {
                    throw DecodingError.dataCorruptedError(
                        forKey: .initSeed, in: c, debugDescription: "init_seed '\(seedText)' is not a decimal UInt64")
                }
                guard !scheme.isEmpty else {
                    throw DecodingError.dataCorruptedError(forKey: .initScheme, in: c, debugDescription: "init_scheme is empty")
                }
                initialization = ModelInitRecord(initSeed: seed, scheme: scheme)
            case (.some, nil):
                throw DecodingError.dataCorruptedError(forKey: .initScheme, in: c, debugDescription: "init_seed without init_scheme")
            case (nil, .some):
                throw DecodingError.dataCorruptedError(forKey: .initSeed, in: c, debugDescription: "init_scheme without init_seed")
            }
        }

        func encode(to encoder: Encoder) throws {
            var c = encoder.container(keyedBy: CodingKeys.self)
            try c.encode(dropoutPhiloxState, forKey: .dropoutPhiloxState)
            try c.encode(streams, forKey: .streams)
            // A decimal string: a UInt64 above 2^53 does not survive a JSON
            // number in every reader.
            try c.encode(initialization.map { String($0.initSeed) }, forKey: .initSeed)
            try c.encode(initialization?.scheme, forKey: .initScheme)
            try c.encode(behaviorFingerprint, forKey: .behaviorFingerprint)
        }

        /// A record with no training run behind it (a mint, a derive, a
        /// model-only save before any training), with the dropout state of
        /// the trainer snapshot it was written with (nil when there is none).
        static func withoutRunStreams(dropoutPhiloxState: DropoutPhiloxState?) -> RNG {
            RNG(dropoutPhiloxState: dropoutPhiloxState, streams: nil, behaviorFingerprint: nil)
        }
    }

    /// The run's named random streams at the save (determinism plan A3): the
    /// master seed every stream derives from, and the position of each
    /// stream that lives across the run's steps. A resume that restores them
    /// continues every draw the uninterrupted run would have made.
    struct RunStreams: Codable, Equatable, Sendable {
        /// Whether the master seed was configured (`random_seed_mode` =
        /// seeded, or `--seed`) or drawn at run start. Either way it is the
        /// run's seed; a drawn seed replays the run when passed to `--seed`.
        enum SeedOrigin: String, Codable, Sendable {
            case configured
            /// `--seed` named it (schema 3; a schema-2 record folded it into
            /// `configured`).
            case commandLine = "command_line"
            case drawn
        }

        let masterSeed: UInt64
        let seedOrigin: SeedOrigin
        /// `DCMRandomStreams.derivationVersion` the streams were named under.
        let streamDerivation: String
        /// The replay buffer's `sampler` stream: what the next minibatch
        /// draws from.
        let samplerState: DCMRandom
        /// The trainer's `dropout` stream: what the next dropout reseed (a
        /// training-graph rebuild) draws from.
        let dropoutStreamState: DCMRandom
        /// The serial the next self-play (GUI) or train-vs-UCI game takes,
        /// naming its stream; null on corpus replay, whose recorded games
        /// draw nothing.
        let nextGameSerial: Int?
        /// Arenas the run has started, naming the next arena's game streams;
        /// null outside the GUI.
        let arenasStarted: Int?
        /// Train-vs-UCI: each opponent instance's game index (in the run's
        /// opponent order) — the game in progress at the save, whose index
        /// sets which colour the trainer plays. A resume starts each
        /// instance's next game at this index, so the colour alternation
        /// continues. Null outside train-vs-UCI.
        let opponentGameIndices: [Int]?

        /// The largest game serial, arena count or opponent game index a
        /// record may carry: far above anything a run reaches, and far
        /// enough below `Int.max` that continuing from it cannot overflow.
        /// A larger or negative counter is a corrupt or hand-edited record,
        /// refused when it is decoded — a negative serial would trap the
        /// serial counter, and one at `Int.max` overflows its next
        /// increment.
        static let maximumRecordedCounter = Int(UInt32.max)

        enum CodingKeys: String, CodingKey {
            case masterSeed = "master_seed"
            case seedOrigin = "seed_origin"
            case streamDerivation = "stream_derivation"
            case samplerState = "sampler_state"
            case dropoutStreamState = "dropout_stream_state"
            case nextGameSerial = "next_game_serial"
            case arenasStarted = "arenas_started"
            case opponentGameIndices = "opponent_game_indices"
        }

        init(masterSeed: UInt64, seedOrigin: SeedOrigin, streamDerivation: String, samplerState: DCMRandom,
             dropoutStreamState: DCMRandom, nextGameSerial: Int?, arenasStarted: Int?, opponentGameIndices: [Int]?) {
            self.masterSeed = masterSeed
            self.seedOrigin = seedOrigin
            self.streamDerivation = streamDerivation
            self.samplerState = samplerState
            self.dropoutStreamState = dropoutStreamState
            self.nextGameSerial = nextGameSerial
            self.arenasStarted = arenasStarted
            self.opponentGameIndices = opponentGameIndices
        }

        init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            // A decimal string, so seeds above 2^53 survive JSON parsing;
            // digits only, exactly as `String(masterSeed)` writes it.
            let seedText = try c.decode(String.self, forKey: .masterSeed)
            guard let seed = UInt64(strictDecimal: seedText) else {
                throw DecodingError.dataCorruptedError(
                    forKey: .masterSeed, in: c, debugDescription: "master_seed \"\(seedText)\" is not a decimal UInt64")
            }
            masterSeed = seed
            seedOrigin = try c.decode(SeedOrigin.self, forKey: .seedOrigin)
            streamDerivation = try c.decode(String.self, forKey: .streamDerivation)
            samplerState = try c.decode(DCMRandom.self, forKey: .samplerState)
            dropoutStreamState = try c.decode(DCMRandom.self, forKey: .dropoutStreamState)
            let counterRange = 0...Self.maximumRecordedCounter
            func counter(_ value: Int, _ key: CodingKeys) throws -> Int {
                guard counterRange.contains(value) else {
                    throw DecodingError.dataCorruptedError(
                        forKey: key, in: c,
                        debugDescription: "\(key.stringValue) \(value) is outside 0…\(Self.maximumRecordedCounter)")
                }
                return value
            }
            nextGameSerial = try c.decode(Int?.self, forKey: .nextGameSerial).map { try counter($0, .nextGameSerial) }
            arenasStarted = try c.decode(Int?.self, forKey: .arenasStarted).map { try counter($0, .arenasStarted) }
            opponentGameIndices = try c.decode([Int]?.self, forKey: .opponentGameIndices)
                .map { indices in try indices.map { try counter($0, .opponentGameIndices) } }
        }

        func encode(to encoder: Encoder) throws {
            var c = encoder.container(keyedBy: CodingKeys.self)
            try c.encode(String(masterSeed), forKey: .masterSeed)
            try c.encode(seedOrigin, forKey: .seedOrigin)
            try c.encode(streamDerivation, forKey: .streamDerivation)
            try c.encode(samplerState, forKey: .samplerState)
            try c.encode(dropoutStreamState, forKey: .dropoutStreamState)
            try c.encode(nextGameSerial, forKey: .nextGameSerial)
            try c.encode(arenasStarted, forKey: .arenasStarted)
            try c.encode(opponentGameIndices, forKey: .opponentGameIndices)
        }
    }

    // MARK: Segment history

    /// One earlier segment of the run.
    struct SegmentSummary: Codable, Equatable, Sendable {
        let segmentIndex: Int
        let segmentID: String
        let start: SegmentStart
        let startedUnix: Int64
        /// When that segment's last file (the one the next segment resumed
        /// from) was recorded.
        let recordedUnix: Int64
        let startTrainerStep: Int?
        let endTrainerStep: Int?
        let segmentLocalStep: Int
        let segmentGames: Int
        let segmentPositions: Int
        let segmentTrainStepSec: Double
        let segmentWallSec: Double
        let exactResume: Bool
        let build: Build
        let device: Device
        /// That segment's configuration: `recorded(nil)` for a segment with
        /// no training behind it; unrecorded for a summary of, or carried
        /// inside, a record that predates schema 3.
        let configuration: Recorded<TrainingConfiguration?>
        /// That segment's parameter snapshot (its own record's).
        let parameters: Recorded<Parameters?>
        /// The corpora it fed (`recorded(nil)` when it was not a replay).
        let corpusIdentity: Recorded<CorpusIdentity?>
        /// Where its corpus feed began (`recorded(nil)` when not a replay).
        let segmentStartCorpus: Recorded<FeedPoint?>
        let pathKind: Recorded<PathKind>
        let argv: Recorded<[String]>
        let runSeeds: Recorded<[RunSeedEntry]?>

        enum CodingKeys: String, CodingKey {
            case configuration
            case parameters
            case corpusIdentity = "corpus_identity"
            case segmentStartCorpus = "segment_start_corpus"
            case pathKind = "path_kind"
            case argv
            case runSeeds = "run_seeds"
            case segmentIndex = "segment_index"
            case segmentID = "segment_id"
            case start
            case startedUnix = "started_unix"
            case recordedUnix = "recorded_unix"
            case startTrainerStep = "start_trainer_step"
            case endTrainerStep = "end_trainer_step"
            case segmentLocalStep = "segment_local_step"
            case segmentGames = "segment_games"
            case segmentPositions = "segment_positions"
            case segmentTrainStepSec = "segment_train_step_sec"
            case segmentWallSec = "segment_wall_sec"
            case exactResume = "exact_resume"
            case build
            case device
        }

        /// The summary of `record`'s own segment. Every value comes from the
        /// record; a fact it does not hold (a schema-2 record's configuration
        /// or the start of a carried corpus feed) stays unrecorded.
        init(of record: LineageRecord) {
            segmentIndex = record.run.segmentIndex
            segmentID = record.run.segmentID
            start = record.run.start
            startedUnix = record.run.segmentStartedUnix
            recordedUnix = record.run.recordedUnix
            startTrainerStep = record.steps.segmentStartTrainerStep
            endTrainerStep = record.steps.cumTrainerStep
            segmentLocalStep = record.steps.segmentLocalStep
            segmentGames = record.fed.segmentGames
            segmentPositions = record.fed.segmentPositions
            segmentTrainStepSec = record.time.segmentTrainStepSec
            segmentWallSec = record.time.segmentWallSec
            exactResume = record.run.exactResume
            build = record.build
            device = record.device
            switch record.configuration {
            case .notTrained: configuration = .recorded(nil)
            case .unrecorded: configuration = .unrecorded
            case .recorded(let value): configuration = .recorded(value)
            }
            parameters = .recorded(record.parameters)
            if let corpus = record.fed.corpus {
                corpusIdentity = .recorded(corpus.corpusIdentity)
                switch corpus.segmentStart {
                case .recorded(let point): segmentStartCorpus = .recorded(point)
                case .unrecorded: segmentStartCorpus = .unrecorded
                }
            } else {
                corpusIdentity = .recorded(nil)
                segmentStartCorpus = .recorded(nil)
            }
            pathKind = .recorded(record.invocation.pathKind)
            argv = .recorded(record.invocation.argv)
            switch record.runSeeds {
            case .notTrained: runSeeds = .recorded(nil)
            case .unrecorded: runSeeds = .unrecorded
            case .recorded(let seeds): runSeeds = .recorded(seeds)
            }
        }

        /// The keys schema 3 added.
        static let schemaThreeKeys: [CodingKeys] = [
            .configuration, .parameters, .corpusIdentity, .segmentStartCorpus, .pathKind, .argv, .runSeeds,
        ]

        init(from decoder: Decoder) throws {
            try self.init(from: decoder, schema: LineageRecord.currentSchema)
        }

        /// A summary inside a record written at `schema`: at schema 2 the
        /// schema-3 keys are absent and read back as unrecorded.
        init(from decoder: Decoder, schema: Int) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            segmentIndex = try c.decode(Int.self, forKey: .segmentIndex)
            segmentID = try c.decode(String.self, forKey: .segmentID)
            start = try c.decode(SegmentStart.self, forKey: .start)
            startedUnix = try c.decode(Int64.self, forKey: .startedUnix)
            recordedUnix = try c.decode(Int64.self, forKey: .recordedUnix)
            startTrainerStep = try c.decode(Int?.self, forKey: .startTrainerStep)
            endTrainerStep = try c.decode(Int?.self, forKey: .endTrainerStep)
            segmentLocalStep = try c.decode(Int.self, forKey: .segmentLocalStep)
            segmentGames = try c.decode(Int.self, forKey: .segmentGames)
            segmentPositions = try c.decode(Int.self, forKey: .segmentPositions)
            segmentTrainStepSec = try c.decode(Double.self, forKey: .segmentTrainStepSec)
            segmentWallSec = try c.decode(Double.self, forKey: .segmentWallSec)
            exactResume = try c.decode(Bool.self, forKey: .exactResume)
            build = try Build(from: try c.superDecoder(forKey: .build), schema: schema)
            device = try c.decode(Device.self, forKey: .device)
            if schema >= 3 {
                configuration = try c.decode(Recorded<TrainingConfiguration?>.self, forKey: .configuration)
                parameters = try c.decode(Recorded<Parameters?>.self, forKey: .parameters)
                corpusIdentity = try c.decode(Recorded<CorpusIdentity?>.self, forKey: .corpusIdentity)
                segmentStartCorpus = try c.decode(Recorded<FeedPoint?>.self, forKey: .segmentStartCorpus)
                pathKind = try c.decode(Recorded<PathKind>.self, forKey: .pathKind)
                argv = try c.decode(Recorded<[String]>.self, forKey: .argv)
                runSeeds = try c.decode(Recorded<[RunSeedEntry]?>.self, forKey: .runSeeds)
            } else {
                try LineageRecord.refuseSchemaThreeKeys(Self.schemaThreeKeys, in: c, schema: schema)
                configuration = .unrecorded
                parameters = .unrecorded
                corpusIdentity = .unrecorded
                segmentStartCorpus = .unrecorded
                pathKind = .unrecorded
                argv = .unrecorded
                runSeeds = .unrecorded
            }
        }

        func encode(to encoder: Encoder) throws {
            var c = encoder.container(keyedBy: CodingKeys.self)
            try c.encode(segmentIndex, forKey: .segmentIndex)
            try c.encode(segmentID, forKey: .segmentID)
            try c.encode(start, forKey: .start)
            try c.encode(startedUnix, forKey: .startedUnix)
            try c.encode(recordedUnix, forKey: .recordedUnix)
            try c.encode(startTrainerStep, forKey: .startTrainerStep)
            try c.encode(endTrainerStep, forKey: .endTrainerStep)
            try c.encode(segmentLocalStep, forKey: .segmentLocalStep)
            try c.encode(segmentGames, forKey: .segmentGames)
            try c.encode(segmentPositions, forKey: .segmentPositions)
            try c.encode(segmentTrainStepSec, forKey: .segmentTrainStepSec)
            try c.encode(segmentWallSec, forKey: .segmentWallSec)
            try c.encode(exactResume, forKey: .exactResume)
            try c.encode(build, forKey: .build)
            try c.encode(device, forKey: .device)
            try c.encode(configuration, forKey: .configuration)
            try c.encode(parameters, forKey: .parameters)
            try c.encode(corpusIdentity, forKey: .corpusIdentity)
            try c.encode(segmentStartCorpus, forKey: .segmentStartCorpus)
            try c.encode(pathKind, forKey: .pathKind)
            try c.encode(argv, forKey: .argv)
            try c.encode(runSeeds, forKey: .runSeeds)
        }
    }

    // MARK: Top-level coding

    enum CodingKeys: String, CodingKey {
        case schema, run, parent, steps, fed, time, parameters, build, invocation, device, rng, segments
        case derivationHistory = "derivation_history"
        case configuration
        case runSeeds = "run_seeds"
        case ancestry
    }

    /// A record this build composes: always at the current schema, with
    /// every invariant checked.
    init(run: Run, parent: Parent?, steps: Steps, fed: Fed, time: Time,
         parameters: Parameters?, configuration: RecordedIfTrained<TrainingConfiguration>,
         runSeeds: RecordedIfTrained<[RunSeedEntry]>, build: Build, invocation: Invocation, device: Device,
         rng: RNG, segments: [SegmentSummary], ancestry: Ancestry,
         derivationHistory: [ModelDerivation.DerivationRecord]) throws {
        self.schema = Self.currentSchema
        self.run = run
        self.parent = parent
        self.steps = steps
        self.fed = fed
        self.time = time
        self.parameters = parameters
        self.configuration = configuration
        self.runSeeds = runSeeds
        self.build = build
        self.invocation = invocation
        self.device = device
        self.rng = rng
        self.segments = segments
        self.ancestry = ancestry
        self.derivationHistory = derivationHistory
        try checkInvariants()
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        schema = try c.decode(Int.self, forKey: .schema)
        guard (Self.oldestDecodableSchema...Self.currentSchema).contains(schema) else {
            throw DecodingError.dataCorruptedError(
                forKey: .schema, in: c,
                debugDescription: "lineage schema \(schema) is not a supported schema "
                    + "(\(Self.oldestDecodableSchema)...\(Self.currentSchema))")
        }
        run = try c.decode(Run.self, forKey: .run)
        parent = try c.decode(Parent?.self, forKey: .parent)
        steps = try c.decode(Steps.self, forKey: .steps)
        fed = try Fed(from: try c.superDecoder(forKey: .fed), schema: schema)
        time = try c.decode(Time.self, forKey: .time)
        parameters = try c.decode(Parameters?.self, forKey: .parameters)
        build = try Build(from: try c.superDecoder(forKey: .build), schema: schema)
        invocation = try c.decode(Invocation.self, forKey: .invocation)
        device = try c.decode(Device.self, forKey: .device)
        rng = try c.decode(RNG.self, forKey: .rng)
        var list = try c.nestedUnkeyedContainer(forKey: .segments)
        var summaries: [SegmentSummary] = []
        while !list.isAtEnd {
            summaries.append(try SegmentSummary(from: try list.superDecoder(), schema: schema))
        }
        segments = summaries
        derivationHistory = try c.decode([ModelDerivation.DerivationRecord].self, forKey: .derivationHistory)
        if schema >= 3 {
            configuration = try c.decode(RecordedIfTrained<TrainingConfiguration>.self, forKey: .configuration)
            runSeeds = try c.decode(RecordedIfTrained<[RunSeedEntry]>.self, forKey: .runSeeds)
            ancestry = try c.decode(Ancestry.self, forKey: .ancestry)
        } else {
            try Self.refuseSchemaThreeKeys([.configuration, .runSeeds, .ancestry], in: c, schema: schema)
            if rng.streams?.seedOrigin == .commandLine {
                throw DecodingError.dataCorruptedError(
                    forKey: .rng, in: c, debugDescription: "seed_origin command_line is a schema-3 value")
            }
            // What a schema-2 record states, in the schema-3 shape: a record
            // with training behind it has a configuration schema 2 never
            // stored (unrecorded) and the seed its streams name, from no
            // known step; one without has neither. Schema 2 kept no
            // ancestry, so the chain before this run is unrecorded unless
            // the run began fresh.
            configuration = parameters == nil ? .notTrained : .unrecorded
            if parameters == nil {
                runSeeds = .notTrained
            } else if let streams = rng.streams {
                runSeeds = .recorded([RunSeedEntry(convertedFrom: streams)])
            } else {
                runSeeds = .unrecorded
            }
            ancestry = Ancestry(historyBeforeOldestRun: AncestorRun.historyBefore(
                                    runFirstStartedAs: summaries.first?.start ?? run.start),
                                runs: [])
        }
        do {
            try checkInvariants()
        } catch {
            throw DecodingError.dataCorruptedError(forKey: .schema, in: c, debugDescription: "\(error)")
        }
    }

    func encode(to encoder: Encoder) throws {
        // A record is only ever written at the current schema: a decoded
        // older record is converted (`withoutTrainerState`, a tracker, a
        // copy) before it is written, never re-emitted as it was.
        guard schema == Self.currentSchema else {
            throw SchemaError.invariant("a schema-\(schema) record is written only after conversion to schema \(Self.currentSchema)")
        }
        try checkInvariants()
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(schema, forKey: .schema)
        try c.encode(run, forKey: .run)
        try c.encode(parent, forKey: .parent)
        try c.encode(steps, forKey: .steps)
        try c.encode(fed, forKey: .fed)
        try c.encode(time, forKey: .time)
        try c.encode(parameters, forKey: .parameters)
        try c.encode(build, forKey: .build)
        try c.encode(invocation, forKey: .invocation)
        try c.encode(device, forKey: .device)
        try c.encode(rng, forKey: .rng)
        try c.encode(segments, forKey: .segments)
        try c.encode(derivationHistory, forKey: .derivationHistory)
        try c.encode(configuration, forKey: .configuration)
        try c.encode(runSeeds, forKey: .runSeeds)
        try c.encode(ancestry, forKey: .ancestry)
    }

    /// The record-level invariants (plan S2), checked on encode and decode.
    func checkInvariants() throws {
        guard (parameters == nil) == configuration.isNotTrained, (parameters == nil) == runSeeds.isNotTrained else {
            throw SchemaError.invariant("parameters, configuration and run_seeds are null together or not at all")
        }
        if let seeds = runSeeds.value {
            guard !seeds.isEmpty else {
                throw SchemaError.invariant("a recorded run_seeds lists at least one seed")
            }
            if seeds.contains(where: { $0.fromTrainerStep == nil }) {
                guard seeds.count == 1 else {
                    throw SchemaError.invariant("a run_seeds entry without a step is the only entry (converted from schema 2)")
                }
            }
            let steps = seeds.compactMap(\.fromTrainerStep)
            guard steps == steps.sorted() else {
                throw SchemaError.invariant("run_seeds entries are in trainer-step order")
            }
            if seeds.count > 1 {
                guard configuration.value?.pathKind == .gui else {
                    throw SchemaError.invariant("only a gui segment changes its seed within the segment")
                }
            }
            if let streams = rng.streams, let last = seeds.last {
                guard last.masterSeed == streams.masterSeed, last.seedOrigin == streams.seedOrigin,
                      last.streamDerivation == streams.streamDerivation else {
                    throw SchemaError.invariant("the last run_seeds entry is the seed rng.streams names")
                }
            }
        }
        if let configuration = configuration.value, let clock = steps.cumTrainerStep {
            if let late = configuration.parameterChanges.first(where: { $0.committedAtTrainerStep > clock }) {
                throw SchemaError.invariant(
                    "parameter change \(late.id) at trainer step \(late.committedAtTrainerStep) is after the record's clock \(clock)")
            }
        }
        for summary in segments {
            guard let configuration = summary.configuration.value ?? nil, let end = summary.endTrainerStep else { continue }
            if let late = configuration.parameterChanges.first(where: { $0.committedAtTrainerStep > end }) {
                throw SchemaError.invariant(
                    "segment \(summary.segmentIndex)'s parameter change \(late.id) is after its end step \(end)")
            }
        }
    }

    /// Refuse any of `keys` in a container decoded at a schema before 3.
    static func refuseSchemaThreeKeys<K: CodingKey>(_ keys: [K], in c: KeyedDecodingContainer<K>, schema: Int) throws {
        try refuseKeys(keys, in: c, reason: "a schema-\(schema) record does not carry this schema-3 key")
    }

    static func refuseKeys<K: CodingKey>(_ keys: [K], in c: KeyedDecodingContainer<K>, reason: String) throws {
        if let present = keys.first(where: { c.contains($0) }) {
            throw DecodingError.dataCorruptedError(forKey: present, in: c, debugDescription: reason)
        }
    }

    // MARK: Safetensors carriage

    /// `__metadata__` key of the record's JSON.
    static let metadataKey = "dcm_lineage"

    /// Flat mirror keys, derived at write time and never read back.
    /// `derivation_history` is written only when the history is non-empty,
    /// the convention files before lineage used for it.
    enum MirrorKey {
        static let lineageRunID = "lineage_run_id"
        static let segmentIndex = "lineage_segment_index"
        static let cumTrainerStep = "cum_trainer_step"
        static let cumGamesFed = "cum_games_fed"
        static let cumTrainStepSec = "cum_train_step_sec"
        static let gitDirty = "git_dirty"
        /// The segment step (`steps.segment_local_step`), so a person reading
        /// a header sees both numbers — `training_step` (the trainer step)
        /// and this — without opening the JSON record.
        static let segmentLocalStep = "lineage_segment_local_step"
        static let derivationHistory = ModelDerivation.derivationHistoryKey
        static let all = [lineageRunID, segmentIndex, cumTrainerStep, cumGamesFed, cumTrainStepSec, gitDirty,
                          segmentLocalStep, derivationHistory]
        /// Written for a total no predecessor recorded.
        static let unrecorded = "unrecorded"
    }

    /// This record for a model file that holds the same weights but no
    /// trainer state (a champion file): the run's dropout Philox state,
    /// stream positions and behavior fingerprint describe a trainer
    /// snapshot that file does not carry, so they are dropped; the run's
    /// init seed and scheme describe the weights and stay. A run started
    /// from such a file branches with fresh random state, never resumes
    /// this one's.
    ///
    /// The result is always at the current schema: a champion file of a
    /// loaded schema-2 model carries that record's facts, with what schema 2
    /// never stored unrecorded (see the file comment). The configuration,
    /// run seeds, segments and ancestry stay: they describe the weights.
    func withoutTrainerState() throws -> LineageRecord {
        try LineageRecord(
            run: run,
            parent: parent,
            steps: steps,
            fed: fed,
            time: time,
            parameters: parameters,
            configuration: configuration,
            runSeeds: runSeeds,
            build: build,
            invocation: invocation,
            device: device,
            rng: RNG.withoutRunStreams(dropoutPhiloxState: nil).withInitialization(rng.initialization),
            segments: segments,
            ancestry: ancestry,
            derivationHistory: derivationHistory)
    }

    /// The record as the compact, sorted-key JSON text stored in a file.
    func jsonText() throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        return String(decoding: try encoder.encode(self), as: UTF8.self)
    }

    /// Decode a record from the JSON text a file stores.
    static func decode(jsonText: String) throws -> LineageRecord {
        try JSONDecoder().decode(LineageRecord.self, from: Data(jsonText.utf8))
    }

    /// The `__metadata__` entries for this record: the JSON value plus its
    /// flat mirrors.
    func metadataEntries() throws -> [String: String] {
        var entries: [String: String] = [
            Self.metadataKey: try jsonText(),
            MirrorKey.lineageRunID: run.lineageRunID,
            MirrorKey.segmentIndex: String(run.segmentIndex),
            MirrorKey.cumTrainerStep: steps.cumTrainerStep.map(String.init) ?? MirrorKey.unrecorded,
            MirrorKey.cumGamesFed: fed.cumGames.map(String.init) ?? MirrorKey.unrecorded,
            MirrorKey.cumTrainStepSec: time.cumTrainStepSec.map { String($0) } ?? MirrorKey.unrecorded,
            MirrorKey.gitDirty: build.gitDirty ? "true" : "false",
            MirrorKey.segmentLocalStep: String(steps.segmentLocalStep),
        ]
        if !derivationHistory.isEmpty {
            entries[MirrorKey.derivationHistory] = try Self.derivationHistoryJSON(derivationHistory)
        }
        return entries
    }

    /// A derivation history as the compact, sorted-key JSON array a file
    /// stores (the same text `--derive-model` has always written).
    static func derivationHistoryJSON(_ history: [ModelDerivation.DerivationRecord]) throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        return String(decoding: try encoder.encode(history), as: UTF8.self)
    }

    /// What a file says about its lineage.
    enum Presence: Equatable, Sendable {
        case recorded(LineageRecord)
        /// The file was written at a format version before lineage existed.
        case unrecorded(formatVersion: Int)

        var record: LineageRecord? {
            if case .recorded(let record) = self { return record }
            return nil
        }
    }

    // MARK: Argument redaction

    /// `arguments` with the value of every secret-bearing option replaced by
    /// `<redacted>`. An option is secret-bearing when its name contains one
    /// of `secretOptionMarkers`; both `--name value` and `--name=value` forms
    /// are covered.
    static func redactedArguments(_ arguments: [String]) -> [String] {
        var out: [String] = []
        out.reserveCapacity(arguments.count)
        var redactNext = false
        for argument in arguments {
            if redactNext {
                out.append(redactedValue)
                redactNext = false
                continue
            }
            guard argument.hasPrefix("-") else {
                out.append(argument)
                continue
            }
            let name: Substring
            let inlineValue: Bool
            if let equals = argument.firstIndex(of: "=") {
                name = argument[..<equals]
                inlineValue = true
            } else {
                name = argument[...]
                inlineValue = false
            }
            guard isSecretOptionName(String(name)) else {
                out.append(argument)
                continue
            }
            if inlineValue {
                out.append(String(name) + "=" + redactedValue)
            } else {
                out.append(argument)
                redactNext = true
            }
        }
        return out
    }

    static let secretOptionMarkers = ["token", "secret", "password", "apikey", "api-key"]

    /// Whether an option named `name` carries a secret value (argv options
    /// and train-vs-UCI engine options alike).
    static func isSecretOptionName(_ name: String) -> Bool {
        let lowered = name.lowercased()
        return secretOptionMarkers.contains { lowered.contains($0) }
    }
    static let redactedValue = "<redacted>"
}
