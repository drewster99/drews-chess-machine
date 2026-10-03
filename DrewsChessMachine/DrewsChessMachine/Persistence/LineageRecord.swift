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
    /// continues from.
    static let currentSchema = 2

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
        /// Total trainer steps behind these weights. On a trainer-state file
        /// it equals `trainer_completed_steps` (the writer refuses
        /// otherwise). Null only when no predecessor recorded it.
        let cumTrainerStep: Int?
        /// The trainer clock when this segment began — what the dashboards
        /// used to enter by hand as `cumstep_base`.
        let segmentStartTrainerStep: Int?
        /// Steps this segment trained (the CLI files' `training_step`).
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
            let c = try decoder.container(keyedBy: CodingKeys.self)
            cumGames = try c.decode(Int?.self, forKey: .cumGames)
            cumPositions = try c.decode(Int?.self, forKey: .cumPositions)
            segmentGames = try c.decode(Int.self, forKey: .segmentGames)
            segmentPositions = try c.decode(Int.self, forKey: .segmentPositions)
            corpus = try c.decode(CorpusPosition?.self, forKey: .corpus)
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

    /// Where a corpus-replay run stood in its corpus at the save.
    struct CorpusPosition: Codable, Equatable, Sendable {
        let corpusID: String
        let corpusPath: String
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
            case epoch
            case nextGameIndex = "next_game_index"
            case shard
            case populatedPlies = "populated_plies"
            case bufferCapacity = "buffer_capacity"
            case feedAheadPositions = "feed_ahead_positions"
            case feedPerStep = "feed_per_step"
            case shardSHA256 = "shard_sha256"
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
        let gitDirty: Bool

        enum CodingKeys: String, CodingKey {
            case buildNumber = "build_number"
            case gitHash = "git_hash"
            case gitBranch = "git_branch"
            case gitDirty = "git_dirty"
        }

        /// The running build.
        static var current: Build {
            Build(buildNumber: BuildInfo.buildNumber, gitHash: BuildInfo.gitHash,
                  gitBranch: BuildInfo.gitBranch, gitDirty: BuildInfo.gitDirty)
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

        enum CodingKeys: String, CodingKey {
            case dropoutPhiloxState = "dropout_philox_state"
            case streams
        }

        init(dropoutPhiloxState: DropoutPhiloxState?, streams: RunStreams?) {
            self.dropoutPhiloxState = dropoutPhiloxState
            self.streams = streams
        }

        init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            dropoutPhiloxState = try c.decode(DropoutPhiloxState?.self, forKey: .dropoutPhiloxState)
            streams = try c.decode(RunStreams?.self, forKey: .streams)
        }

        func encode(to encoder: Encoder) throws {
            var c = encoder.container(keyedBy: CodingKeys.self)
            try c.encode(dropoutPhiloxState, forKey: .dropoutPhiloxState)
            try c.encode(streams, forKey: .streams)
        }

        /// A record with no training run behind it (a mint, a derive, a
        /// model-only save before any training), with the dropout state of
        /// the trainer snapshot it was written with (nil when there is none).
        static func withoutRunStreams(dropoutPhiloxState: DropoutPhiloxState?) -> RNG {
            RNG(dropoutPhiloxState: dropoutPhiloxState, streams: nil)
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

        enum CodingKeys: String, CodingKey {
            case masterSeed = "master_seed"
            case seedOrigin = "seed_origin"
            case streamDerivation = "stream_derivation"
            case samplerState = "sampler_state"
            case dropoutStreamState = "dropout_stream_state"
            case nextGameSerial = "next_game_serial"
            case arenasStarted = "arenas_started"
        }

        init(masterSeed: UInt64, seedOrigin: SeedOrigin, streamDerivation: String, samplerState: DCMRandom,
             dropoutStreamState: DCMRandom, nextGameSerial: Int?, arenasStarted: Int?) {
            self.masterSeed = masterSeed
            self.seedOrigin = seedOrigin
            self.streamDerivation = streamDerivation
            self.samplerState = samplerState
            self.dropoutStreamState = dropoutStreamState
            self.nextGameSerial = nextGameSerial
            self.arenasStarted = arenasStarted
        }

        init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            // A decimal string, so seeds above 2^53 survive JSON parsing.
            let seedText = try c.decode(String.self, forKey: .masterSeed)
            guard let seed = UInt64(seedText) else {
                throw DecodingError.dataCorruptedError(
                    forKey: .masterSeed, in: c, debugDescription: "master_seed \"\(seedText)\" is not a decimal UInt64")
            }
            masterSeed = seed
            seedOrigin = try c.decode(SeedOrigin.self, forKey: .seedOrigin)
            streamDerivation = try c.decode(String.self, forKey: .streamDerivation)
            samplerState = try c.decode(DCMRandom.self, forKey: .samplerState)
            dropoutStreamState = try c.decode(DCMRandom.self, forKey: .dropoutStreamState)
            nextGameSerial = try c.decode(Int?.self, forKey: .nextGameSerial)
            arenasStarted = try c.decode(Int?.self, forKey: .arenasStarted)
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

        enum CodingKeys: String, CodingKey {
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
        }

        init(from decoder: Decoder) throws {
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
            build = try c.decode(Build.self, forKey: .build)
            device = try c.decode(Device.self, forKey: .device)
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
        }
    }

    // MARK: Top-level coding

    enum CodingKeys: String, CodingKey {
        case schema, run, parent, steps, fed, time, parameters, build, invocation, device, rng, segments
        case derivationHistory = "derivation_history"
    }

    init(schema: Int, run: Run, parent: Parent?, steps: Steps, fed: Fed, time: Time,
         parameters: Parameters?, build: Build, invocation: Invocation, device: Device,
         rng: RNG, segments: [SegmentSummary], derivationHistory: [ModelDerivation.DerivationRecord]) {
        self.schema = schema
        self.run = run
        self.parent = parent
        self.steps = steps
        self.fed = fed
        self.time = time
        self.parameters = parameters
        self.build = build
        self.invocation = invocation
        self.device = device
        self.rng = rng
        self.segments = segments
        self.derivationHistory = derivationHistory
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        schema = try c.decode(Int.self, forKey: .schema)
        guard schema == Self.currentSchema else {
            throw DecodingError.dataCorruptedError(
                forKey: .schema, in: c,
                debugDescription: "lineage schema \(schema) is not the supported schema \(Self.currentSchema)")
        }
        run = try c.decode(Run.self, forKey: .run)
        parent = try c.decode(Parent?.self, forKey: .parent)
        steps = try c.decode(Steps.self, forKey: .steps)
        fed = try c.decode(Fed.self, forKey: .fed)
        time = try c.decode(Time.self, forKey: .time)
        parameters = try c.decode(Parameters?.self, forKey: .parameters)
        build = try c.decode(Build.self, forKey: .build)
        invocation = try c.decode(Invocation.self, forKey: .invocation)
        device = try c.decode(Device.self, forKey: .device)
        rng = try c.decode(RNG.self, forKey: .rng)
        segments = try c.decode([SegmentSummary].self, forKey: .segments)
        derivationHistory = try c.decode([ModelDerivation.DerivationRecord].self, forKey: .derivationHistory)
    }

    func encode(to encoder: Encoder) throws {
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
        static let derivationHistory = ModelDerivation.derivationHistoryKey
        static let all = [lineageRunID, segmentIndex, cumTrainerStep, cumGamesFed, cumTrainStepSec, gitDirty, derivationHistory]
        /// Written for a total no predecessor recorded.
        static let unrecorded = "unrecorded"
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
            let lowered = name.lowercased()
            guard secretOptionMarkers.contains(where: { lowered.contains($0) }) else {
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
    static let redactedValue = "<redacted>"
}
