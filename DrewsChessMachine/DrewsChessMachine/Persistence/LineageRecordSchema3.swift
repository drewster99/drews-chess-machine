//
//  LineageRecordSchema3.swift
//  DrewsChessMachine
//
//  The parts of a `LineageRecord` that schema 3 added (hyperparameter
//  recording plan, Part S): the segment's training configuration, the seeds
//  a run trained under, the runs its weights descend from, and the
//  `Recorded` wrapper that says "this record cannot know" without inventing
//  a value.
//
//  Why a wrapper rather than `null`. Schema 3 is written by every new save,
//  including a champion file of a loaded schema-2 model, a derive of one, and
//  the summary of a schema-2 parent inside a resumed run. Those records carry
//  facts schema 2 never stored (the earlier segment's configuration, a
//  build's diff hash, the corpus position a segment started from). Writing
//  `null` would read as "none" where the truth is "not recorded"; inventing a
//  value would put a model in a measured column. So a value that can come
//  from a record that predates it is a `Recorded<T>`: recorded (and then
//  possibly a legitimate `null` inside), or explicitly unrecorded. Nothing in
//  a schema-3 record is ever filled in.
//
//  Strictness follows the rest of the record: every key is required at
//  schema 3, and a schema-2 record must not carry any of them (no schema-2
//  writer produced them).
//

import Foundation

extension LineageRecord {

    // MARK: Recorded

    /// A value the record either states or explicitly does not know. JSON:
    /// `{"recorded": true, "value": …}` / `{"recorded": false}`. Decoding is
    /// strict: a missing `recorded`, `recorded: true` without `value`,
    /// `recorded: false` with a `value`, or any other key is an error.
    enum Recorded<Value: Codable & Equatable & Sendable>: Codable, Equatable, Sendable {
        case recorded(Value)
        case unrecorded

        private enum CodingKeys: String, CodingKey, CaseIterable {
            case recorded
            case value
        }

        /// Any key, so unexpected ones can be named and refused.
        private struct AnyKey: CodingKey {
            let stringValue: String
            var intValue: Int? { nil }
            init(stringValue: String) { self.stringValue = stringValue }
            init?(intValue: Int) { nil }
        }

        var value: Value? {
            if case .recorded(let value) = self { return value }
            return nil
        }

        init(from decoder: Decoder) throws {
            let all = try decoder.container(keyedBy: AnyKey.self)
            let allowed = Set(CodingKeys.allCases.map(\.stringValue))
            if let extra = all.allKeys.first(where: { !allowed.contains($0.stringValue) }) {
                throw DecodingError.dataCorruptedError(
                    forKey: extra, in: all, debugDescription: "unexpected key '\(extra.stringValue)' in a recorded value")
            }
            let c = try decoder.container(keyedBy: CodingKeys.self)
            let isRecorded = try c.decode(Bool.self, forKey: .recorded)
            if isRecorded {
                guard c.contains(.value) else {
                    throw DecodingError.keyNotFound(
                        CodingKeys.value,
                        .init(codingPath: c.codingPath, debugDescription: "recorded: true without a value"))
                }
                self = .recorded(try c.decode(Value.self, forKey: .value))
            } else {
                guard !c.contains(.value) else {
                    throw DecodingError.dataCorruptedError(
                        forKey: .value, in: c, debugDescription: "recorded: false with a value")
                }
                self = .unrecorded
            }
        }

        func encode(to encoder: Encoder) throws {
            var c = encoder.container(keyedBy: CodingKeys.self)
            switch self {
            case .recorded(let value):
                try c.encode(true, forKey: .recorded)
                try c.encode(value, forKey: .value)
            case .unrecorded:
                try c.encode(false, forKey: .recorded)
            }
        }
    }

    // A `Recorded<T?>`'s JSON `null` inside `value` round-trips through
    // `Optional`'s own `Codable`: `encode(nil)` writes `null`, and
    // `decode(T?.self)` of a present `null` is nil.

    /// A record-level fact that exists only when training stands behind the
    /// weights: JSON `null` (no training — a mint, or an untrained copy of
    /// one), `{"recorded": false}` (training happened, but the record it was
    /// carried from predates the field), or `{"recorded": true, "value": …}`.
    /// `configuration` and `run_seeds` use it.
    enum RecordedIfTrained<Value: Codable & Equatable & Sendable>: Codable, Equatable, Sendable {
        case notTrained
        case unrecorded
        case recorded(Value)

        var value: Value? {
            if case .recorded(let value) = self { return value }
            return nil
        }

        var isNotTrained: Bool {
            if case .notTrained = self { return true }
            return false
        }

        init(from decoder: Decoder) throws {
            let single = try decoder.singleValueContainer()
            if single.decodeNil() {
                self = .notTrained
                return
            }
            switch try Recorded<Value>(from: decoder) {
            case .recorded(let value): self = .recorded(value)
            case .unrecorded: self = .unrecorded
            }
        }

        func encode(to encoder: Encoder) throws {
            switch self {
            case .notTrained:
                var single = encoder.singleValueContainer()
                try single.encodeNil()
            case .unrecorded:
                try Recorded<Value>.unrecorded.encode(to: encoder)
            case .recorded(let value):
                try Recorded<Value>.recorded(value).encode(to: encoder)
            }
        }
    }

    // MARK: Run seeds (gap 9b)

    /// One seed a run trained under, from the trainer step it took over.
    struct RunSeedEntry: Codable, Equatable, Sendable {
        /// The trainer clock when this seed took over; null only in an
        /// entry converted from a schema-2 record, which recorded the seed in
        /// force at its save but not since when.
        let fromTrainerStep: Int?
        let masterSeed: UInt64
        let seedOrigin: RunStreams.SeedOrigin
        let streamDerivation: String

        enum CodingKeys: String, CodingKey {
            case fromTrainerStep = "from_trainer_step"
            case masterSeed = "master_seed"
            case seedOrigin = "seed_origin"
            case streamDerivation = "stream_derivation"
        }

        init(fromTrainerStep: Int?, masterSeed: UInt64, seedOrigin: RunStreams.SeedOrigin, streamDerivation: String) {
            self.fromTrainerStep = fromTrainerStep
            self.masterSeed = masterSeed
            self.seedOrigin = seedOrigin
            self.streamDerivation = streamDerivation
        }

        init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            fromTrainerStep = try c.decode(Int?.self, forKey: .fromTrainerStep)
            let seedText = try c.decode(String.self, forKey: .masterSeed)
            guard let seed = UInt64(strictDecimal: seedText) else {
                throw DecodingError.dataCorruptedError(
                    forKey: .masterSeed, in: c, debugDescription: "master_seed \"\(seedText)\" is not a decimal UInt64")
            }
            masterSeed = seed
            seedOrigin = try c.decode(RunStreams.SeedOrigin.self, forKey: .seedOrigin)
            streamDerivation = try c.decode(String.self, forKey: .streamDerivation)
        }

        func encode(to encoder: Encoder) throws {
            var c = encoder.container(keyedBy: CodingKeys.self)
            try c.encode(fromTrainerStep, forKey: .fromTrainerStep)
            // A decimal string, as `rng.streams.master_seed` is written.
            try c.encode(String(masterSeed), forKey: .masterSeed)
            try c.encode(seedOrigin, forKey: .seedOrigin)
            try c.encode(streamDerivation, forKey: .streamDerivation)
        }

        /// The seed a schema-2 record's streams state, from no known step.
        init(convertedFrom streams: RunStreams) {
            self.init(fromTrainerStep: nil, masterSeed: streams.masterSeed, seedOrigin: streams.seedOrigin,
                      streamDerivation: streams.streamDerivation)
        }
    }

    // MARK: Training configuration (schema 3)

    /// How a segment trained beyond its parameter snapshot: what a path does
    /// that no parameter states (its policy-tail precision, budget, game
    /// generation, data-generating champions, live edits). One per segment;
    /// `null` on a record with no training behind it.
    struct TrainingConfiguration: Codable, Equatable, Sendable {
        /// The path whose segment composed this configuration — the key every
        /// path-dependent invariant reads (never the record's own
        /// `invocation.path_kind`, which a derive or copy replaces).
        let pathKind: PathKind
        /// The policy-tail precision the trainer ran, as its token
        /// (`PolicyTailPrecisionSetting`): from format v12 the trainer
        /// architecture's (`does_not_apply` on fp32); before it the
        /// process-wide launch flag's value, recorded whatever the dtype.
        let policyTailPrecision: String
        let budget: Budget
        /// Every committed settings change during the segment, in commit
        /// order (GUI); always empty on the CLI paths, whose parameters are
        /// one immutable snapshot.
        let parameterChanges: [ParameterChange]
        /// The champions whose games fed the segment (GUI); empty elsewhere.
        let championChanges: [ChampionChange]
        /// Train-vs-UCI game generation; null on every other path.
        let vsuci: VsUciGeneration?
        /// Self-play Dirichlet root noise (GUI); null on every other path.
        let selfPlayDirichlet: Dirichlet?
        /// Whether the weights the segment began from were value-head
        /// recentered at load; taken once when the segment began.
        let startValueHeadRecentered: Recorded<Bool>
        /// The learning rate and momentum the optimizer is fed at the saved
        /// clock, derived from this record's own inputs and never read back;
        /// null for a record with no trainer clock.
        let scheduleAtSave: ScheduleAtSave?
        /// GUI replay-ratio controller state; null on every other path.
        let replayRatio: ReplayRatio?
        /// The training-health alarm summary of the segment
        /// (`TrainingHealthSegmentSummary`, alarm plan OD-10). Unrecorded
        /// while no path runs the health monitor; recorded once one does.
        let healthAlarms: Recorded<TrainingHealthSegmentSummary>

        enum CodingKeys: String, CodingKey {
            case pathKind = "path_kind"
            case policyTailPrecision = "policy_tail_precision"
            case budget
            case parameterChanges = "parameter_changes"
            case championChanges = "champion_changes"
            case vsuci
            case selfPlayDirichlet = "self_play_dirichlet"
            case startValueHeadRecentered = "start_value_head_recentered"
            case scheduleAtSave = "schedule_at_save"
            case replayRatio = "replay_ratio"
            case healthAlarms = "health_alarms"
        }

        init(pathKind: PathKind, policyTailPrecision: String, budget: Budget, parameterChanges: [ParameterChange],
             championChanges: [ChampionChange], vsuci: VsUciGeneration?, selfPlayDirichlet: Dirichlet?,
             startValueHeadRecentered: Recorded<Bool>, scheduleAtSave: ScheduleAtSave?, replayRatio: ReplayRatio?,
             healthAlarms: Recorded<TrainingHealthSegmentSummary>) throws {
            self.pathKind = pathKind
            self.policyTailPrecision = policyTailPrecision
            self.budget = budget
            self.parameterChanges = parameterChanges
            self.championChanges = championChanges
            self.vsuci = vsuci
            self.selfPlayDirichlet = selfPlayDirichlet
            self.startValueHeadRecentered = startValueHeadRecentered
            self.scheduleAtSave = scheduleAtSave
            self.replayRatio = replayRatio
            self.healthAlarms = healthAlarms
            try validate()
        }

        init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            pathKind = try c.decode(PathKind.self, forKey: .pathKind)
            policyTailPrecision = try c.decode(String.self, forKey: .policyTailPrecision)
            budget = try c.decode(Budget.self, forKey: .budget)
            parameterChanges = try c.decode([ParameterChange].self, forKey: .parameterChanges)
            championChanges = try c.decode([ChampionChange].self, forKey: .championChanges)
            vsuci = try c.decode(VsUciGeneration?.self, forKey: .vsuci)
            selfPlayDirichlet = try c.decode(Dirichlet?.self, forKey: .selfPlayDirichlet)
            startValueHeadRecentered = try c.decode(Recorded<Bool>.self, forKey: .startValueHeadRecentered)
            scheduleAtSave = try c.decode(ScheduleAtSave?.self, forKey: .scheduleAtSave)
            replayRatio = try c.decode(ReplayRatio?.self, forKey: .replayRatio)
            healthAlarms = try c.decode(Recorded<TrainingHealthSegmentSummary>.self, forKey: .healthAlarms)
            do {
                try validate()
            } catch {
                throw DecodingError.dataCorruptedError(forKey: .pathKind, in: c, debugDescription: "\(error)")
            }
        }

        func encode(to encoder: Encoder) throws {
            try validate()
            var c = encoder.container(keyedBy: CodingKeys.self)
            try c.encode(pathKind, forKey: .pathKind)
            try c.encode(policyTailPrecision, forKey: .policyTailPrecision)
            try c.encode(budget, forKey: .budget)
            try c.encode(parameterChanges, forKey: .parameterChanges)
            try c.encode(championChanges, forKey: .championChanges)
            try c.encode(vsuci, forKey: .vsuci)
            try c.encode(selfPlayDirichlet, forKey: .selfPlayDirichlet)
            try c.encode(startValueHeadRecentered, forKey: .startValueHeadRecentered)
            try c.encode(scheduleAtSave, forKey: .scheduleAtSave)
            try c.encode(replayRatio, forKey: .replayRatio)
            try c.encode(healthAlarms, forKey: .healthAlarms)
        }

        /// The invariants keyed to `pathKind` (plan S2, review N4).
        func validate() throws {
            switch pathKind {
            case .derive, .newModel:
                throw SchemaError.invalidConfiguration("path_kind \(pathKind.rawValue) composes no configuration")
            case .gui, .replay, .vsuci:
                break
            }
            guard PolicyTailPrecisionSetting(rawValue: policyTailPrecision) != nil else {
                throw SchemaError.invalidConfiguration("policy_tail_precision '\(policyTailPrecision)' is not a known precision")
            }
            guard (vsuci != nil) == (pathKind == .vsuci) else {
                throw SchemaError.invalidConfiguration("vsuci is present exactly on a vsuci configuration")
            }
            guard (selfPlayDirichlet != nil) == (pathKind == .gui), (replayRatio != nil) == (pathKind == .gui) else {
                throw SchemaError.invalidConfiguration("self_play_dirichlet and replay_ratio are present exactly on a gui configuration")
            }
            if pathKind == .gui {
                guard championChanges.first?.trigger == .segmentStart else {
                    throw SchemaError.invalidConfiguration("a gui configuration's champion_changes begins with its segment_start champion")
                }
            } else {
                guard championChanges.isEmpty, parameterChanges.isEmpty else {
                    throw SchemaError.invalidConfiguration("champion_changes and parameter_changes are empty outside the gui")
                }
            }
        }
    }

    /// The run budget a path enforces (resolved limits; `null` = none).
    struct Budget: Codable, Equatable, Sendable {
        let trainingStepLimit: Int?
        let trainingTimeLimitSec: Double?
        let epochLimit: Int?

        enum CodingKeys: String, CodingKey {
            case trainingStepLimit = "training_step_limit"
            case trainingTimeLimitSec = "training_time_limit_sec"
            case epochLimit = "epoch_limit"
        }

        init(trainingStepLimit: Int?, trainingTimeLimitSec: Double?, epochLimit: Int?) {
            self.trainingStepLimit = trainingStepLimit
            self.trainingTimeLimitSec = trainingTimeLimitSec
            self.epochLimit = epochLimit
        }

        init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            trainingStepLimit = try c.decode(Int?.self, forKey: .trainingStepLimit)
            trainingTimeLimitSec = try c.decode(Double?.self, forKey: .trainingTimeLimitSec)
            epochLimit = try c.decode(Int?.self, forKey: .epochLimit)
        }

        func encode(to encoder: Encoder) throws {
            var c = encoder.container(keyedBy: CodingKeys.self)
            try c.encode(trainingStepLimit, forKey: .trainingStepLimit)
            try c.encode(trainingTimeLimitSec, forKey: .trainingTimeLimitSec)
            try c.encode(epochLimit, forKey: .epochLimit)
        }

        /// No limit at all: the interactive GUI.
        static let none = Budget(trainingStepLimit: nil, trainingTimeLimitSec: nil, epochLimit: nil)
    }

    /// One committed settings change (gap 4).
    struct ParameterChange: Codable, Equatable, Sendable {
        /// The trainer clock when the assignment committed — when the setting
        /// changed, not the first step that used it: a trainer-level key is
        /// first used at this step + 1, or + 2 when the step in progress had
        /// already built its feeds; sampling keys at each game's next start;
        /// captured keys at the next Play-and-Train start.
        let committedAtTrainerStep: Int
        let recordedUnix: Int64
        let id: String
        let old: ParameterValue
        let new: ParameterValue
        /// The original commit step when an arena promotion rewound the clock
        /// past the entry and it was re-stamped to the arena-start step; null
        /// otherwise.
        let restampedFrom: Int?

        enum CodingKeys: String, CodingKey {
            case committedAtTrainerStep = "committed_at_trainer_step"
            case recordedUnix = "recorded_unix"
            case id, old, new
            case restampedFrom = "restamped_from"
        }

        init(committedAtTrainerStep: Int, recordedUnix: Int64, id: String, old: ParameterValue, new: ParameterValue,
             restampedFrom: Int?) {
            self.committedAtTrainerStep = committedAtTrainerStep
            self.recordedUnix = recordedUnix
            self.id = id
            self.old = old
            self.new = new
            self.restampedFrom = restampedFrom
        }

        init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            committedAtTrainerStep = try c.decode(Int.self, forKey: .committedAtTrainerStep)
            recordedUnix = try c.decode(Int64.self, forKey: .recordedUnix)
            id = try c.decode(String.self, forKey: .id)
            let oldRaw = try c.decode(ParameterValue.self, forKey: .old)
            let newRaw = try c.decode(ParameterValue.self, forKey: .new)
            // JSON does not keep a whole `Double`'s kind (`180.0` is written
            // `180` and reads back as `.int`), so a known key's values are
            // read as its declared type and re-encoded canonically — the
            // record decodes equal to the one written. A key this build
            // does not know keeps the value as the file states it.
            if let key = TrainingParameters.keysByID[id] {
                old = try Self.canonical(key, oldRaw)
                new = try Self.canonical(key, newRaw)
            } else {
                old = oldRaw
                new = newRaw
            }
            restampedFrom = try c.decode(Int?.self, forKey: .restampedFrom)
        }

        private static func canonical<K: TrainingParameterKey>(_ key: K.Type, _ raw: ParameterValue) throws -> ParameterValue {
            K.encode(try K.decode(raw))
        }

        func encode(to encoder: Encoder) throws {
            var c = encoder.container(keyedBy: CodingKeys.self)
            try c.encode(committedAtTrainerStep, forKey: .committedAtTrainerStep)
            try c.encode(recordedUnix, forKey: .recordedUnix)
            try c.encode(id, forKey: .id)
            try c.encode(old, forKey: .old)
            try c.encode(new, forKey: .new)
            try c.encode(restampedFrom, forKey: .restampedFrom)
        }
    }

    /// A change of the champion whose games fed the segment (B5).
    struct ChampionChange: Codable, Equatable, Sendable {
        enum Trigger: String, Codable, Sendable {
            case segmentStart = "segment_start"
            case arena
            case manual
            case loadedModel = "loaded_model"
        }

        /// The trainer clock the champion's games start feeding at; for a
        /// promotion, the clock the promoted weights carry.
        let trainerStep: Int
        let recordedUnix: Int64
        let championModelID: String
        /// The champion file's `content_sha256`; null for weights never read
        /// from a file (built or promoted).
        let championContentSHA256: String?
        let trigger: Trigger

        enum CodingKeys: String, CodingKey {
            case trainerStep = "trainer_step"
            case recordedUnix = "recorded_unix"
            case championModelID = "champion_model_id"
            case championContentSHA256 = "champion_content_sha256"
            case trigger
        }

        init(trainerStep: Int, recordedUnix: Int64, championModelID: String, championContentSHA256: String?, trigger: Trigger) {
            self.trainerStep = trainerStep
            self.recordedUnix = recordedUnix
            self.championModelID = championModelID
            self.championContentSHA256 = championContentSHA256
            self.trigger = trigger
        }

        init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            trainerStep = try c.decode(Int.self, forKey: .trainerStep)
            recordedUnix = try c.decode(Int64.self, forKey: .recordedUnix)
            championModelID = try c.decode(String.self, forKey: .championModelID)
            championContentSHA256 = try c.decode(String?.self, forKey: .championContentSHA256)
            trigger = try c.decode(Trigger.self, forKey: .trigger)
        }

        func encode(to encoder: Encoder) throws {
            var c = encoder.container(keyedBy: CodingKeys.self)
            try c.encode(trainerStep, forKey: .trainerStep)
            try c.encode(recordedUnix, forKey: .recordedUnix)
            try c.encode(championModelID, forKey: .championModelID)
            try c.encode(championContentSHA256, forKey: .championContentSHA256)
            try c.encode(trigger, forKey: .trigger)
        }
    }

    /// Dirichlet root noise (`DirichletNoiseConfig`), its `Float`s written as
    /// the exact `Double` they hold.
    struct Dirichlet: Codable, Equatable, Sendable {
        let alpha: Double
        let epsilon: Double
        let plyLimit: Int

        enum CodingKeys: String, CodingKey {
            case alpha, epsilon
            case plyLimit = "ply_limit"
        }

        init(_ config: DirichletNoiseConfig) {
            alpha = Double(config.alpha)
            epsilon = Double(config.epsilon)
            plyLimit = config.plyLimit
        }
    }

    /// A move-selection schedule (`SamplingSchedule`).
    struct MoveSelection: Codable, Equatable, Sendable {
        let startTau: Double
        let decayPerPly: Double
        let floorTau: Double
        let dirichlet: Dirichlet?

        enum CodingKeys: String, CodingKey {
            case startTau = "start_tau"
            case decayPerPly = "decay_per_ply"
            case floorTau = "floor_tau"
            case dirichlet
        }

        init(_ schedule: SamplingSchedule) {
            startTau = Double(schedule.startTau)
            decayPerPly = Double(schedule.decayPerPly)
            floorTau = Double(schedule.floorTau)
            dirichlet = schedule.dirichletNoise.map(Dirichlet.init)
        }

        init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            startTau = try c.decode(Double.self, forKey: .startTau)
            decayPerPly = try c.decode(Double.self, forKey: .decayPerPly)
            floorTau = try c.decode(Double.self, forKey: .floorTau)
            dirichlet = try c.decode(Dirichlet?.self, forKey: .dirichlet)
        }

        func encode(to encoder: Encoder) throws {
            var c = encoder.container(keyedBy: CodingKeys.self)
            try c.encode(startTau, forKey: .startTau)
            try c.encode(decayPerPly, forKey: .decayPerPly)
            try c.encode(floorTau, forKey: .floorTau)
            try c.encode(dirichlet, forKey: .dirichlet)
        }
    }

    /// Train-vs-UCI game generation (B1, gap 11).
    struct VsUciGeneration: Codable, Equatable, Sendable {
        let maxPliesPerGame: Int
        let evalSyncEverySteps: Int
        let trainerMoveSelection: MoveSelection
        let opponents: [Opponent]

        enum CodingKeys: String, CodingKey {
            case maxPliesPerGame = "max_plies_per_game"
            case evalSyncEverySteps = "eval_sync_every_steps"
            case trainerMoveSelection = "trainer_move_selection"
            case opponents
        }

        struct Opponent: Codable, Equatable, Sendable {
            let command: String
            /// SHA-256 of the executable the command names, read once at run
            /// start.
            let executableSHA256: String
            let count: Int
            let goLimit: String
            /// `setoption` pairs, values redacted like argv.
            let options: [Option]
            /// The engine's `id` lines from the pool's first completed
            /// handshake; unrecorded before any instance completed one.
            let identity: Recorded<EngineIdentity>

            enum CodingKeys: String, CodingKey {
                case command
                case executableSHA256 = "executable_sha256"
                case count
                case goLimit = "go_limit"
                case options, identity
            }
        }

        struct Option: Codable, Equatable, Sendable {
            let name: String
            let value: String
        }

        struct EngineIdentity: Codable, Equatable, Sendable {
            let idName: String?
            let idAuthor: String?

            enum CodingKeys: String, CodingKey {
                case idName = "id_name"
                case idAuthor = "id_author"
            }

            init(idName: String?, idAuthor: String?) {
                self.idName = idName
                self.idAuthor = idAuthor
            }

            init(from decoder: Decoder) throws {
                let c = try decoder.container(keyedBy: CodingKeys.self)
                idName = try c.decode(String?.self, forKey: .idName)
                idAuthor = try c.decode(String?.self, forKey: .idAuthor)
            }

            func encode(to encoder: Encoder) throws {
                var c = encoder.container(keyedBy: CodingKeys.self)
                try c.encode(idName, forKey: .idName)
                try c.encode(idAuthor, forKey: .idAuthor)
            }
        }
    }

    /// The fed learning rate and momentum at the saved clock (gap 12),
    /// computed by `LRMomentumCycleReadout.values` — the one function the
    /// optimizer's feeds and the status-bar readouts both use — from the
    /// record's own parameters and clock.
    struct ScheduleAtSave: Codable, Equatable, Sendable {
        let cycleStep: Int
        let learningRateFed: Double
        let momentumFed: Double

        enum CodingKeys: String, CodingKey {
            case cycleStep = "cycle_step"
            case learningRateFed = "learning_rate_fed"
            case momentumFed = "momentum_fed"
        }
    }

    /// GUI replay-ratio controller state (B6).
    struct ReplayRatio: Codable, Equatable, Sendable {
        let starts: [Start]
        /// One heartbeat `RatioSnapshot` at the save; null when none exists
        /// (no heartbeat since the start, or training stopped).
        let atSave: AtSave?

        enum CodingKeys: String, CodingKey {
            case starts
            case atSave = "at_save"
        }

        init(starts: [Start], atSave: AtSave?) {
            self.starts = starts
            self.atSave = atSave
        }

        init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            starts = try c.decode([Start].self, forKey: .starts)
            atSave = try c.decode(AtSave?.self, forKey: .atSave)
        }

        func encode(to encoder: Encoder) throws {
            var c = encoder.container(keyedBy: CodingKeys.self)
            try c.encode(starts, forKey: .starts)
            try c.encode(atSave, forKey: .atSave)
        }

        enum InitialDelaySource: String, Codable, Sendable {
            case lastAutoComputedDelayMs = "last_auto_computed_delay_ms"
            case trainingStepDelayMs = "training_step_delay_ms"
        }

        struct Start: Codable, Equatable, Sendable {
            let trainerStep: Int
            let autoAdjust: Bool
            let initialTrainingStepDelayMs: Int
            let initialDelaySource: InitialDelaySource

            enum CodingKeys: String, CodingKey {
                case trainerStep = "trainer_step"
                case autoAdjust = "auto_adjust"
                case initialTrainingStepDelayMs = "initial_training_step_delay_ms"
                case initialDelaySource = "initial_delay_source"
            }
        }

        struct AtSave: Codable, Equatable, Sendable {
            let autoAdjust: Bool
            let targetRatio: Double
            let currentRatio: Double
            let trainingStepDelayMs: Int
            let selfPlayDelayMs: Int

            enum CodingKeys: String, CodingKey {
                case autoAdjust = "auto_adjust"
                case targetRatio = "target_ratio"
                case currentRatio = "current_ratio"
                case trainingStepDelayMs = "training_step_delay_ms"
                case selfPlayDelayMs = "self_play_delay_ms"
            }
        }
    }

    // MARK: Ancestry (gaps 1b, 1c, B3, B4)

    /// Every earlier run the weights descend from, oldest first, and whether
    /// anything came before the oldest.
    struct Ancestry: Codable, Equatable, Sendable {
        enum HistoryBeforeOldestRun: String, Codable, Sendable {
            /// The oldest run began fresh: the chain is complete.
            case none
            /// The oldest run began from weights no file in the chain records.
            case unrecorded
        }

        let historyBeforeOldestRun: HistoryBeforeOldestRun
        let runs: [AncestorRun]

        enum CodingKeys: String, CodingKey {
            case historyBeforeOldestRun = "history_before_oldest_run"
            case runs
        }

        static let fresh = Ancestry(historyBeforeOldestRun: .none, runs: [])
        static let unrecordedHistory = Ancestry(historyBeforeOldestRun: .unrecorded, runs: [])
    }

    /// One earlier run, as the next run left it.
    struct AncestorRun: Codable, Equatable, Sendable {
        enum LeftBy: String, Codable, Sendable {
            case branch
            case derive
        }

        /// A source file's architecture, kept as the text and format version
        /// it was read with (gap 1c).
        struct ArchitectureAtDeparture: Codable, Equatable, Sendable {
            let formatVersion: String
            let architectureJSON: String

            enum CodingKeys: String, CodingKey {
                case formatVersion = "format_version"
                case architectureJSON = "architecture_json"
            }

            enum ArchitectureError: Error, CustomStringConvertible, LocalizedError {
                case noArchitectureText(source: String)

                var description: String {
                    switch self {
                    case .noArchitectureText(let source):
                        return "lineage: \(source) has no architecture metadata to record as the architecture its weights trained under"
                    }
                }

                var errorDescription: String? { description }
            }

            /// The source's architecture as it stated it (its `architecture`
            /// metadata text byte for byte, with its format version), when a
            /// copy changes the architecture; nil when it keeps it. The text
            /// is never re-encoded: a legacy source keeps its own format, and
            /// its legacy resolution stays the reader's job.
            static func ifChanged(from source: NetworkArchitecture, to target: NetworkArchitecture,
                                  sourceMetadata: [String: String], sourceFormatVersion: Int,
                                  sourceName: String) throws -> ArchitectureAtDeparture? {
                guard source != target else { return nil }
                guard let text = sourceMetadata[SafetensorsModelIO.Key.architecture] else {
                    throw ArchitectureError.noArchitectureText(source: sourceName)
                }
                return ArchitectureAtDeparture(formatVersion: String(sourceFormatVersion), architectureJSON: text)
            }
        }

        struct TotalsAtDeparture: Codable, Equatable, Sendable {
            let cumTrainerStep: Int?
            let cumGames: Int?
            let cumPositions: Int?
            let cumTrainStepSec: Double?
            let cumWallSec: Double?

            enum CodingKeys: String, CodingKey {
                case cumTrainerStep = "cum_trainer_step"
                case cumGames = "cum_games"
                case cumPositions = "cum_positions"
                case cumTrainStepSec = "cum_train_step_sec"
                case cumWallSec = "cum_wall_sec"
            }

            init(of record: LineageRecord) {
                cumTrainerStep = record.steps.cumTrainerStep
                cumGames = record.fed.cumGames
                cumPositions = record.fed.cumPositions
                cumTrainStepSec = record.time.cumTrainStepSec
                cumWallSec = record.time.cumWallSec
            }

            init(from decoder: Decoder) throws {
                let c = try decoder.container(keyedBy: CodingKeys.self)
                cumTrainerStep = try c.decode(Int?.self, forKey: .cumTrainerStep)
                cumGames = try c.decode(Int?.self, forKey: .cumGames)
                cumPositions = try c.decode(Int?.self, forKey: .cumPositions)
                cumTrainStepSec = try c.decode(Double?.self, forKey: .cumTrainStepSec)
                cumWallSec = try c.decode(Double?.self, forKey: .cumWallSec)
            }

            func encode(to encoder: Encoder) throws {
                var c = encoder.container(keyedBy: CodingKeys.self)
                try c.encode(cumTrainerStep, forKey: .cumTrainerStep)
                try c.encode(cumGames, forKey: .cumGames)
                try c.encode(cumPositions, forKey: .cumPositions)
                try c.encode(cumTrainStepSec, forKey: .cumTrainStepSec)
                try c.encode(cumWallSec, forKey: .cumWallSec)
            }
        }

        /// How the ancestor run drew its starting weights, JSON
        /// `{"init_seed": "…", "init_scheme": "…"}`.
        struct Initialization: Codable, Equatable, Sendable {
            let initSeed: UInt64
            let initScheme: String

            enum CodingKeys: String, CodingKey {
                case initSeed = "init_seed"
                case initScheme = "init_scheme"
            }

            init(_ record: ModelInitRecord) {
                initSeed = record.initSeed
                initScheme = record.scheme
            }

            init(from decoder: Decoder) throws {
                let c = try decoder.container(keyedBy: CodingKeys.self)
                let text = try c.decode(String.self, forKey: .initSeed)
                guard let seed = UInt64(strictDecimal: text) else {
                    throw DecodingError.dataCorruptedError(forKey: .initSeed, in: c,
                                                           debugDescription: "init_seed '\(text)' is not a decimal UInt64")
                }
                initSeed = seed
                initScheme = try c.decode(String.self, forKey: .initScheme)
            }

            func encode(to encoder: Encoder) throws {
                var c = encoder.container(keyedBy: CodingKeys.self)
                try c.encode(String(initSeed), forKey: .initSeed)
                try c.encode(initScheme, forKey: .initScheme)
            }
        }

        let lineageRunID: String
        let leftBy: LeftBy
        let leftAt: Parent
        let totalsAtDeparture: TotalsAtDeparture
        let architectureAtDeparture: ArchitectureAtDeparture?
        /// `recorded(value)` for a run that drew its weights; `recorded(nil)`
        /// for one that started from a file's weights (that file's history
        /// is the previous entry, or unrecorded); unrecorded for one that
        /// continued weights whose start this process cannot know.
        let initialization: Recorded<Initialization?>
        let segments: [SegmentSummary]

        enum CodingKeys: String, CodingKey {
            case lineageRunID = "lineage_run_id"
            case leftBy = "left_by"
            case leftAt = "left_at"
            case totalsAtDeparture = "totals_at_departure"
            case architectureAtDeparture = "architecture_at_departure"
            case initialization
            case segments
        }

        init(lineageRunID: String, leftBy: LeftBy, leftAt: Parent, totalsAtDeparture: TotalsAtDeparture,
             architectureAtDeparture: ArchitectureAtDeparture?, initialization: Recorded<Initialization?>,
             segments: [SegmentSummary]) {
            self.lineageRunID = lineageRunID
            self.leftBy = leftBy
            self.leftAt = leftAt
            self.totalsAtDeparture = totalsAtDeparture
            self.architectureAtDeparture = architectureAtDeparture
            self.initialization = initialization
            self.segments = segments
        }

        init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            lineageRunID = try c.decode(String.self, forKey: .lineageRunID)
            leftBy = try c.decode(LeftBy.self, forKey: .leftBy)
            leftAt = try c.decode(Parent.self, forKey: .leftAt)
            totalsAtDeparture = try c.decode(TotalsAtDeparture.self, forKey: .totalsAtDeparture)
            architectureAtDeparture = try c.decode(ArchitectureAtDeparture?.self, forKey: .architectureAtDeparture)
            initialization = try c.decode(Recorded<Initialization?>.self, forKey: .initialization)
            var list = try c.nestedUnkeyedContainer(forKey: .segments)
            var decoded: [SegmentSummary] = []
            while !list.isAtEnd {
                decoded.append(try SegmentSummary(from: try list.superDecoder(), schema: LineageRecord.currentSchema))
            }
            segments = decoded
            if leftBy == .branch, architectureAtDeparture != nil {
                throw DecodingError.dataCorruptedError(forKey: .architectureAtDeparture, in: c,
                                                       debugDescription: "a branch never changes the architecture")
            }
        }

        func encode(to encoder: Encoder) throws {
            var c = encoder.container(keyedBy: CodingKeys.self)
            try c.encode(lineageRunID, forKey: .lineageRunID)
            try c.encode(leftBy, forKey: .leftBy)
            try c.encode(leftAt, forKey: .leftAt)
            try c.encode(totalsAtDeparture, forKey: .totalsAtDeparture)
            try c.encode(architectureAtDeparture, forKey: .architectureAtDeparture)
            try c.encode(initialization, forKey: .initialization)
            try c.encode(segments, forKey: .segments)
        }

        /// The ancestor `record`'s initialization as it states it, read with
        /// how that run began (plan S2): its first segment's start decides
        /// whether a null `rng.initialization` means "started from a file"
        /// or "continued weights whose start is unknown".
        static func initialization(of record: LineageRecord) throws -> Recorded<Initialization?> {
            if let initialization = record.rng.initialization {
                return .recorded(Initialization(initialization))
            }
            let firstStart = record.segments.first?.start ?? record.run.start
            switch firstStart {
            case .branch, .derive:
                return .recorded(nil)
            case .resume:
                guard record.run.continuesUnrecordedHistory else {
                    throw SchemaError.invalidAncestor(
                        "run \(record.run.lineageRunID) began by resume, records no initialization and does not say its history is unrecorded")
                }
                return .unrecorded
            case .fresh:
                throw SchemaError.invalidAncestor(
                    "run \(record.run.lineageRunID) began fresh but records no initialization")
            }
        }

        /// How much history precedes a run with no recorded ancestry,
        /// decided by how its first segment began (`segments.first?.start ??
        /// run.start`): only a fresh run is a complete chain.
        static func historyBefore(runFirstStartedAs firstStart: SegmentStart) -> Ancestry.HistoryBeforeOldestRun {
            firstStart == .fresh ? .none : .unrecorded
        }
    }

    // MARK: Errors

    enum SchemaError: Error, CustomStringConvertible, LocalizedError {
        case invalidConfiguration(String)
        case invalidAncestor(String)
        case invariant(String)

        var description: String {
            switch self {
            case .invalidConfiguration(let detail): return "lineage configuration: \(detail)"
            case .invalidAncestor(let detail): return "lineage ancestry: \(detail)"
            case .invariant(let detail): return "lineage record: \(detail)"
            }
        }

        var errorDescription: String? { description }
    }
}
