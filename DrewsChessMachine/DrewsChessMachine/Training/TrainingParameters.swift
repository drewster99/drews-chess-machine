import Foundation
import Observation
import TrainingParametersMacroSupport

// MARK: - ParameterType

public enum ParameterType: String, Codable, Sendable {
    case bool
    case int
    case double
}

// MARK: - ParameterValue

public enum ParameterValue: Codable, Equatable, Sendable {
    case bool(Bool)
    case int(Int)
    case double(Double)

    public var type: ParameterType {
        switch self {
        case .bool: .bool
        case .int: .int
        case .double: .double
        }
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.singleValueContainer()

        if let x = try? c.decode(Bool.self) {
            self = .bool(x)
        } else if let x = try? c.decode(Int.self) {
            self = .int(x)
        } else if let x = try? c.decode(Double.self) {
            self = .double(x)
        } else {
            throw DecodingError.typeMismatch(
                ParameterValue.self,
                .init(codingPath: decoder.codingPath, debugDescription: "Unsupported parameter value")
            )
        }
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.singleValueContainer()

        switch self {
        case .bool(let x): try c.encode(x)
        case .int(let x): try c.encode(x)
        case .double(let x): try c.encode(x)
        }
    }
}

// MARK: - NumericRange

public struct NumericRange<T: Codable & Comparable & Sendable>: Codable, Sendable {
    public var min: T
    public var max: T

    public init(min: T, max: T) {
        self.min = min
        self.max = max
    }

    public func contains(_ value: T) -> Bool {
        value >= min && value <= max
    }
}

// MARK: - TrainingParameterDefinition

public struct TrainingParameterDefinition: Sendable {
    public let id: String
    public let name: String
    public let description: String
    public let type: ParameterType
    public let defaultValue: ParameterValue
    public let intRange: NumericRange<Int>?
    public let doubleRange: NumericRange<Double>?
    public let category: String
    public let liveTunable: Bool

    public init(
        id: String,
        name: String,
        description: String,
        type: ParameterType,
        defaultValue: ParameterValue,
        intRange: NumericRange<Int>? = nil,
        doubleRange: NumericRange<Double>? = nil,
        category: String,
        liveTunable: Bool
    ) {
        self.id = id
        self.name = name
        self.description = description
        self.type = type
        self.defaultValue = defaultValue
        self.intRange = intRange
        self.doubleRange = doubleRange
        self.category = category
        self.liveTunable = liveTunable
    }

    public func validate(_ value: ParameterValue) throws {
        switch (type, value) {
        case (.bool, .bool):
            return

        case (.int, .int(let x)):
            if let intRange, !intRange.contains(x) {
                throw TrainingConfigError.outOfRange(id: id, value: "\(x)")
            }

        case (.double, .double(let x)):
            if let doubleRange, !doubleRange.contains(x) {
                throw TrainingConfigError.outOfRange(id: id, value: "\(x)")
            }

        case (.double, .int(let x)):
            // Tolerate JSON ints for double-typed parameters.
            let asDouble = Double(x)
            if let doubleRange, !doubleRange.contains(asDouble) {
                throw TrainingConfigError.outOfRange(id: id, value: "\(asDouble)")
            }

        default:
            throw TrainingConfigError.wrongType(id: id)
        }
    }
}

// MARK: - TrainingConfigError

public enum TrainingConfigError: Error, CustomStringConvertible {
    case unknownParameter(id: String)
    case wrongType(id: String)
    case outOfRange(id: String, value: String)

    public var description: String {
        switch self {
        case .unknownParameter(let id):
            "Unknown parameter '\(id)'"
        case .wrongType(let id):
            "Wrong value type for parameter '\(id)'"
        case .outOfRange(let id, let value):
            "Value \(value) is out of range for parameter '\(id)'"
        }
    }
}

// MARK: - TrainingParameterKey

public protocol TrainingParameterKey: Sendable {
    associatedtype Value: Sendable
    static var id: String { get }
    static var definition: TrainingParameterDefinition { get }
    static func encode(_ value: Value) -> ParameterValue
    static func decode(_ value: ParameterValue) throws -> Value
}

// MARK: - 39 parameter keys (macro-driven)

@TrainingParameter(
    name: "Entropy Bonus",
    description: "Entropy regularization coefficient. Higher keeps the policy diverse longer; too high stalls learning.",
    default: 0.0025,
    range: 0.0...0.1,
    category: "Optimizer",
    liveTunable: true
)
public enum EntropyBonus: TrainingParameterKey {}

@TrainingParameter(
    name: "Illegal Mass Penalty Weight",
    description: "Weight for the penalty term that pushes probability mass off illegal moves. Start at 1.0; increase if illegal mass leaks.",
    default: 1.0,
    range: 0.0...100.0,
    category: "Optimizer",
    liveTunable: true
)
public enum IllegalMassWeight: TrainingParameterKey {}

@TrainingParameter(
    name: "Policy Label Smoothing ε",
    description: "Label-smoothing coefficient on the policy CE target. ε=0 = one-hot played-move target (original behavior). ε=0.1 = (1−ε) at played + ε spread uniformly over legal moves; caps per-position concentration at p(played)=1−ε, structurally prevents delta-policy collapse. Range [0, 0.9].",
    default: 0.1,
    range: 0.0...0.9,
    category: "Optimizer",
    liveTunable: true
)
public enum PolicyLabelSmoothingEpsilon: TrainingParameterKey {}

@TrainingParameter(
    name: "Value Label Smoothing ε",
    description: "Label-smoothing coefficient on the value-head W/D/L cross-entropy target. The target is built in-graph as (1−ε)·one_hot(1−z) + ε·(1/3), where 1−z maps the play-time outcome z ∈ {+1,0,−1} to the [win,draw,loss] slot. ε=0 = hard one-hot on the game result. ε>0 gives the value CE a reachable finite-logit equilibrium instead of ±∞, the same way Policy Label Smoothing does for the policy CE. Range [0, 0.5].",
    default: 0.013,
    range: 0.0...0.5,
    category: "Optimizer",
    liveTunable: true
)
public enum ValueLabelSmoothingEpsilon: TrainingParameterKey {}

@TrainingParameter(
    name: "Gradient Clip Max Norm",
    description: "Global L2 norm cap for gradient clipping. Above this, gradients are scaled down before the SGD step.",
    default: 30.0,
    range: 0.1...1000.0,
    category: "Optimizer",
    liveTunable: true
)
public enum GradClipMaxNorm: TrainingParameterKey {}

@TrainingParameter(
    name: "Weight Decay",
    description: "L2 weight decay coefficient. Couples with batch size and the number of update steps per epoch.",
    default: 0.0001,
    range: 0.0...0.1,
    category: "Optimizer",
    liveTunable: true
)
public enum WeightDecay: TrainingParameterKey {}

@TrainingParameter(
    name: "Policy Loss Weight",
    description: "Per-component weighting on the POLICY-LOSS TENSOR inside total_loss = valueLossWeight · valueLoss + policyLossWeight · policyLoss − entropyCoeff · policyEntropy. Pairs with valueLossWeight. Higher values shift shared-trunk gradients toward policy fitting and away from the value head's W/D/L cross-entropy: at policyLossWeight = valueLossWeight = 1 the trunk is pulled equally by both heads (AlphaZero canonical); at policyLossWeight=5+ the policy head dominates trunk shaping and the value head trails. NOT a multiplier on policy logits — that's a common misreading. Without MCTS-quality policy targets (this engine has none), values above ~3 amplify policy-target noise faster than the value head can supply a useful baseline.",
    default: 1.0,
    range: 0.0...20.0,
    category: "Optimizer",
    id: "policy_loss_weight",
    liveTunable: true
)
public enum PolicyLossWeight: TrainingParameterKey {}

@TrainingParameter(
    name: "Value Loss Weight",
    description: "Per-component weighting on the VALUE-LOSS TENSOR inside total_loss = valueLossWeight · valueLoss + policyLossWeight · policyLoss − entropyCoeff · policyEntropy. Mirrors policyLossWeight: at 1.0 the value head's W/D/L categorical-cross-entropy term feeds the trunk at its natural magnitude (AlphaZero canonical); raising it makes the trunk prioritize value fitting over policy fitting. Lc0/KataGo expose the same knob as `value_loss_weight`. The two weights only matter relative to each other plus the entropy term — scaling both by the same factor is equivalent to scaling the learning rate. NOTE: post-2026-05-12 the value loss is CE-scale (~[0, ln 3] at convergence), not the old MSE scale (~0.1–0.4), so the value term's contribution is a bit larger at the same weight — consider a value lower than 1.0 for early WDL runs.",
    default: 1.0,
    range: 0.0...20.0,
    category: "Optimizer",
    id: "value_loss_weight",
    liveTunable: true
)
public enum ValueLossWeight: TrainingParameterKey {}

@TrainingParameter(
    name: "Learning Rate",
    description: "SGD-with-momentum optimizer learning rate. Lower is slower but more stable. Pairs with sqrt_batch_scaling_lr.",
    default: 5.0e-4,
    range: 1.0e-7...1.0,
    category: "Optimizer",
    liveTunable: true
)
public enum LearningRate: TrainingParameterKey {}

@TrainingParameter(
    name: "Momentum Coefficient",
    description: "Polyak momentum μ for SGD. 0.0 disables momentum (pure SGD); higher μ accumulates more gradient history. The optimizer uses decoupled weight decay (AdamW-style), so μ and Weight Decay tune independently — raising μ no longer amplifies decay. Effective step size in correlated-gradient regimes still scales ~1/(1−μ), so a high μ paired with the existing LR can be too aggressive — pair μ jumps with a proportional LR drop. Start low (≤0.5) and watch legalMass / pEntLegal before raising further.",
    default: 0.65,
    range: 0.0...0.99,
    category: "Optimizer",
    liveTunable: true
)
public enum MomentumCoeff: TrainingParameterKey {}

@TrainingParameter(
    name: "Sqrt-Batch Scaling LR",
    description: "When true, scales the effective learning rate by sqrt(batch / referenceBatch). Standard practice when scaling SGD-with-momentum by batch size.",
    default: true,
    category: "Optimizer",
    liveTunable: true
)
public enum SqrtBatchScalingLR: TrainingParameterKey {}

@TrainingParameter(
    name: "LR Warmup Steps",
    description: "Number of training steps over which the learning rate linearly ramps from zero to its target.",
    default: 500,
    range: 0...100000,
    category: "Optimizer",
    liveTunable: true
)
public enum LRWarmupSteps: TrainingParameterKey {}

@TrainingParameter(
    name: "Draw Penalty",
    description: "Outcome value applied to drawn games (in [-1, +1]). Push negative to discourage learned drawing behavior.",
    default: 0.0,
    range: -1.0...1.0,
    category: "Optimizer",
    liveTunable: true
)
public enum DrawPenalty: TrainingParameterKey {}

@TrainingParameter(
    name: "Self-Play Start Tau",
    description: "Initial sampling temperature for self-play games at game-total ply 0 (the starting position). Decays toward target by self_play_tau_decay_per_ply each game-total ply (i.e. each half-move from either side advances the schedule).",
    default: 1.0,
    range: 0.05...5.0,
    category: "Self-Play Sampling",
    liveTunable: true
)
public enum SelfPlayStartTau: TrainingParameterKey {}

@TrainingParameter(
    name: "Self-Play Target Tau",
    description: "Floor sampling temperature for self-play games — start_tau decays toward this value.",
    default: 0.5,
    range: 0.05...5.0,
    category: "Self-Play Sampling",
    liveTunable: true
)
public enum SelfPlayTargetTau: TrainingParameterKey {}

@TrainingParameter(
    name: "Self-Play Tau Decay Per Ply",
    description: "Decay applied to tau on every game-total ply (each half-move from either side), moving start_tau toward target_tau during a self-play game. tau(ply) = max(target_tau, start_tau − decay·ply).",
    default: 0.007,
    range: 0.0...1.0,
    category: "Self-Play Sampling",
    liveTunable: true
)
public enum SelfPlayTauDecayPerPly: TrainingParameterKey {}

@TrainingParameter(
    name: "Self-Play Draw Keep Fraction",
    description: "Fraction of self-play games ending in a draw that are pushed into the replay buffer at game end. 1.0 = keep every drawn game (legacy behavior, no filtering); 0.0 = discard every drawn game (only decisive games train). Decisive (checkmate) games are always kept regardless of this knob. The replay-ratio controller targets the EMITTED positions/sec rate, so dropping draws automatically slows training to keep the cons/prod ratio at target.",
    default: 1.0,
    range: 0.0...1.0,
    category: "Self-Play Sampling",
    liveTunable: true
)
public enum SelfPlayDrawKeepFraction: TrainingParameterKey {}

@TrainingParameter(
    name: "Self-Play Max Plies Per Game",
    description: "Self-play games are auto-terminated when they reach this many plies. Acts as a safety net against games that fail to terminate via the 50-move rule or 3-fold repetition. Terminated games are NOT emitted — they're counted as 'dropped' in the Played stats and never reach the replay buffer. Applies to self-play only — arena games are not affected.",
    default: 150,
    range: 25...500,
    category: "Self-Play Sampling",
    liveTunable: true
)
public enum SelfPlayMaxPliesPerGame: TrainingParameterKey {}

@TrainingParameter(
    name: "Arena Start Tau",
    description: "Initial sampling temperature for arena games. Tighter than self-play to improve W/L/D signal.",
    default: 0.6,
    range: 0.05...5.0,
    category: "Arena",
    liveTunable: true
)
public enum ArenaStartTau: TrainingParameterKey {}

@TrainingParameter(
    name: "Arena Target Tau",
    description: "Floor sampling temperature for arena games.",
    default: 0.2,
    range: 0.05...5.0,
    category: "Arena",
    liveTunable: true
)
public enum ArenaTargetTau: TrainingParameterKey {}

@TrainingParameter(
    name: "Arena Tau Decay Per Ply",
    description: "Decay applied to tau on every game-total ply (each half-move from either side), moving arena start_tau toward target_tau.",
    default: 0.02,
    range: 0.0...1.0,
    category: "Arena",
    liveTunable: true
)
public enum ArenaTauDecayPerPly: TrainingParameterKey {}

@TrainingParameter(
    name: "Replay Ratio Target",
    description: "Target ratio of consumed (training) positions to produced (self-play) positions. ReplayRatioController auto-adjusts step delay to track this.",
    default: 1.0,
    range: 0.01...100.0,
    category: "Replay Buffer",
    liveTunable: true
)
public enum ReplayRatioTarget: TrainingParameterKey {}

@TrainingParameter(
    name: "Replay Ratio Auto Adjust",
    description: "Whether ReplayRatioController auto-tunes the trainer step delay to track Replay Ratio Target.",
    default: false,
    category: "Replay Buffer",
    liveTunable: true
)
public enum ReplayRatioAutoAdjust: TrainingParameterKey {}

@TrainingParameter(
    name: "Self-Play Concurrency",
    description: "Parallel self-play game count. More = faster replay-buffer fill but more GPU contention.",
    default: 4000,
    range: 1...8192,
    category: "Training Window",
    liveTunable: true
)
public enum SelfPlayConcurrency: TrainingParameterKey {}

@TrainingParameter(
    name: "Training Step Delay (ms)",
    description: "Delay between trainer SGD steps in milliseconds. Auto-adjusted by ReplayRatioController when auto-adjust is on.",
    default: 0,
    range: 0...10000,
    category: "Training Window",
    liveTunable: true
)
public enum TrainingStepDelayMs: TrainingParameterKey {}

@TrainingParameter(
    name: "Self-Play Delay (ms)",
    description: "Per-game-per-worker delay between self-play games in milliseconds. Used only when replay-ratio auto-adjust is OFF; auto-adjust on lets the controller manage it.",
    default: 0,
    range: 0...10000,
    category: "Training Window",
    liveTunable: true
)
public enum SelfPlayDelayMs: TrainingParameterKey {}

@TrainingParameter(
    name: "Training Batch Size",
    description: "SGD minibatch size. Couples with learning_rate (scaled via sqrt_batch_scaling_lr) and weight_decay.",
    default: 4096,
    range: 32...65536,
    category: "Training Window",
    liveTunable: false
)
public enum TrainingBatchSize: TrainingParameterKey {}

@TrainingParameter(
    name: "Replay Buffer Capacity",
    description: "Maximum number of positions retained in the FIFO replay buffer.",
    default: 1500000,
    range: 1000...10000000,
    category: "Replay Buffer",
    liveTunable: false
)
public enum ReplayBufferCapacity: TrainingParameterKey {}

@TrainingParameter(
    name: "Replay Buffer Min Positions Before Training",
    description: "Number of self-play positions accumulated before the trainer starts pulling minibatches.",
    default: 500000,
    range: 0...10000000,
    category: "Replay Buffer",
    liveTunable: false
)
public enum ReplayBufferMinPositionsBeforeTraining: TrainingParameterKey {}

@TrainingParameter(
    name: "Max Plies From Any 1 Game",
    description: "Cap on how many plies may be drawn from any single self-play game within one training batch. Decorrelates the minibatch by forcing it to span many distinct games rather than letting one long marathon dominate. Default 10. Range capped at 400 — with typical batch sizes (e.g. 4096) the cap is always active.",
    default: 10,
    range: 1...400,
    category: "Replay Buffer",
    liveTunable: true
)
public enum MaxPliesFromAnyOneGame: TrainingParameterKey {}

@TrainingParameter(
    name: "Target Sampled Game Length (plies)",
    description: "When > 0, the batch sampler exponentially down-weights positions from long games so the position-weighted mean game length of the sampled batch approaches this value (in plies). 0 disables the length tilt. Intended to be set below the buffer's natural mean to de-weight shuffle-draw marathons.",
    default: 124,
    range: 0...10000,
    category: "Replay Buffer",
    liveTunable: true
)
public enum TargetSampledGameLengthPlies: TrainingParameterKey {}

@TrainingParameter(
    name: "Max Draws Per Batch (%)",
    description: "Ceiling on the percentage of positions in a training batch that come from drawn games. If the buffer holds fewer drawn positions than the cap allows, the batch simply contains fewer (no padding); freed slots go to positions from decisive games. 100 disables the cap.",
    default: 75,
    range: 0...100,
    category: "Replay Buffer",
    liveTunable: true
)
public enum MaxDrawPercentPerBatch: TrainingParameterKey {}

@TrainingParameter(
    name: "Arena Promote Threshold",
    description: "Minimum candidate score (in [0, 1]) required to promote the candidate over the champion.",
    default: 0.53,
    range: 0.5...1.0,
    category: "Arena",
    liveTunable: false
)
public enum ArenaPromoteThreshold: TrainingParameterKey {}

@TrainingParameter(
    name: "Arena Games Per Tournament",
    description: "Number of candidate-vs-champion games per arena run. Higher = tighter Wilson confidence interval.",
    default: 400,
    range: 4...10000,
    category: "Arena",
    liveTunable: false
)
public enum ArenaGamesPerTournament: TrainingParameterKey {}

@TrainingParameter(
    name: "Arena Auto Interval (sec)",
    description: "Automatic arena interval in seconds; the play-and-train loop schedules a new arena every N seconds.",
    default: 900.0,
    range: 60.0...86400.0,
    category: "Arena",
    liveTunable: true
)
public enum ArenaAutoIntervalSec: TrainingParameterKey {}

@TrainingParameter(
    name: "Candidate Probe Interval (sec)",
    description: "Interval between candidate forward-pass probes for collapse-detection telemetry.",
    default: 15.0,
    range: 1.0...3600.0,
    category: "Collapse Detection",
    liveTunable: false
)
public enum CandidateProbeIntervalSec: TrainingParameterKey {}

@TrainingParameter(
    name: "Legal-Mass Collapse Threshold",
    description: "If illegal_mass_sum stays at or above this for the no-improvement window, the run early-bails.",
    default: 0.99,
    range: 0.5...1.0,
    category: "Collapse Detection",
    liveTunable: false
)
public enum LegalMassCollapseThreshold: TrainingParameterKey {}

@TrainingParameter(
    name: "Legal-Mass Collapse Grace (sec)",
    description: "Post-training-start grace window during which legal-mass collapse early-bail is suppressed.",
    default: 600.0,
    range: 0.0...86400.0,
    category: "Collapse Detection",
    liveTunable: false
)
public enum LegalMassCollapseGraceSeconds: TrainingParameterKey {}

@TrainingParameter(
    name: "Legal-Mass Collapse No-Improvement Probes",
    description: "Number of consecutive collapsed probes (after grace) before the run early-bails.",
    default: 8,
    range: 1...1000,
    category: "Collapse Detection",
    liveTunable: false
)
public enum LegalMassCollapseNoImprovementProbes: TrainingParameterKey {}

@TrainingParameter(
    name: "Arena Concurrency",
    description: "Number of concurrent arena games. Higher = faster arena throughput at cost of GPU contention.",
    default: 400,
    range: 1...4096,
    category: "Arena",
    liveTunable: true
)
public enum ArenaConcurrency: TrainingParameterKey {}

@TrainingParameter(
    name: "Batch Stats Interval",
    description: "Compute and emit [BATCH-STATS] every N training batches. 0 disables. Cost is ~1ms per evaluated batch; default 10 keeps log volume manageable.",
    default: 10,
    range: 0...10000,
    category: "Observability",
    liveTunable: true
)
public enum BatchStatsInterval: TrainingParameterKey {}

// MARK: - TrainingParametersSnapshot

public struct TrainingParametersSnapshot: Sendable {
    fileprivate let values: [String: ParameterValue]

    public func value<K: TrainingParameterKey>(for key: K.Type) -> K.Value {
        let raw = values[K.id] ?? K.definition.defaultValue
        do {
            return try K.decode(raw)
        } catch {
            // Stored value validated on insert; should never happen. If the
            // baked-in default itself doesn't round-trip, that's a programmer
            // error in the parameter declaration — surface it loudly rather
            // than crash with an opaque `try!`.
            do {
                return try K.decode(K.definition.defaultValue)
            } catch {
                preconditionFailure("default value for \(K.id) does not round-trip through decode: \(error)")
            }
        }
    }

    public subscript<K: TrainingParameterKey>(_ key: K.Type) -> K.Value {
        value(for: key)
    }

    /// Untyped {id: ParameterValue} view of the snapshot. Used for diffing
    /// before/after when applying a JSON override, and for save/load
    /// that needs to iterate by id rather than by typed key.
    public func rawValueMap() -> [String: ParameterValue] {
        values
    }
}

// Typed accessors on the snapshot — keep parallel with the stored properties on TrainingParameters.
public extension TrainingParametersSnapshot {
    var entropyBonus: Double { value(for: EntropyBonus.self) }
    var gradClipMaxNorm: Double { value(for: GradClipMaxNorm.self) }
    var weightDecay: Double { value(for: WeightDecay.self) }
    var policyLossWeight: Double { value(for: PolicyLossWeight.self) }
    var valueLossWeight: Double { value(for: ValueLossWeight.self) }
    var learningRate: Double { value(for: LearningRate.self) }
    var momentumCoeff: Double { value(for: MomentumCoeff.self) }
    var sqrtBatchScalingLR: Bool { value(for: SqrtBatchScalingLR.self) }
    var lrWarmupSteps: Int { value(for: LRWarmupSteps.self) }
    var drawPenalty: Double { value(for: DrawPenalty.self) }
    var selfPlayStartTau: Double { value(for: SelfPlayStartTau.self) }
    var selfPlayTargetTau: Double { value(for: SelfPlayTargetTau.self) }
    var selfPlayTauDecayPerPly: Double { value(for: SelfPlayTauDecayPerPly.self) }
    var selfPlayDrawKeepFraction: Double { value(for: SelfPlayDrawKeepFraction.self) }
    var selfPlayMaxPliesPerGame: Int { value(for: SelfPlayMaxPliesPerGame.self) }
    var arenaStartTau: Double { value(for: ArenaStartTau.self) }
    var arenaTargetTau: Double { value(for: ArenaTargetTau.self) }
    var arenaTauDecayPerPly: Double { value(for: ArenaTauDecayPerPly.self) }
    var replayRatioTarget: Double { value(for: ReplayRatioTarget.self) }
    var replayRatioAutoAdjust: Bool { value(for: ReplayRatioAutoAdjust.self) }
    var selfPlayConcurrency: Int { value(for: SelfPlayConcurrency.self) }
    var trainingStepDelayMs: Int { value(for: TrainingStepDelayMs.self) }
    var selfPlayDelayMs: Int { value(for: SelfPlayDelayMs.self) }
    var trainingBatchSize: Int { value(for: TrainingBatchSize.self) }
    var replayBufferCapacity: Int { value(for: ReplayBufferCapacity.self) }
    var replayBufferMinPositionsBeforeTraining: Int { value(for: ReplayBufferMinPositionsBeforeTraining.self) }
    var maxPliesFromAnyOneGame: Int { value(for: MaxPliesFromAnyOneGame.self) }
    var targetSampledGameLengthPlies: Int { value(for: TargetSampledGameLengthPlies.self) }
    var maxDrawPercentPerBatch: Int { value(for: MaxDrawPercentPerBatch.self) }
    var arenaPromoteThreshold: Double { value(for: ArenaPromoteThreshold.self) }
    var arenaGamesPerTournament: Int { value(for: ArenaGamesPerTournament.self) }
    var arenaAutoIntervalSec: Double { value(for: ArenaAutoIntervalSec.self) }
    var candidateProbeIntervalSec: Double { value(for: CandidateProbeIntervalSec.self) }
    var legalMassCollapseThreshold: Double { value(for: LegalMassCollapseThreshold.self) }
    var legalMassCollapseGraceSeconds: Double { value(for: LegalMassCollapseGraceSeconds.self) }
    var legalMassCollapseNoImprovementProbes: Int { value(for: LegalMassCollapseNoImprovementProbes.self) }
    var arenaConcurrency: Int { value(for: ArenaConcurrency.self) }
    var batchStatsInterval: Int { value(for: BatchStatsInterval.self) }
}

// MARK: - TrainingParameters singleton

@MainActor
@Observable
public final class TrainingParameters {
    public static let shared = TrainingParameters()

    // Stored properties — one per parameter. didSet persists to UserDefaults.
    // @Observable instruments these for SwiftUI re-renders.
    public var entropyBonus: Double { didSet { Self.persist(EntropyBonus.self, value: entropyBonus) } }
    public var illegalMassWeight: Double { didSet { Self.persist(IllegalMassWeight.self, value: illegalMassWeight) } }
    public var policyLabelSmoothingEpsilon: Double { didSet { Self.persist(PolicyLabelSmoothingEpsilon.self, value: policyLabelSmoothingEpsilon) } }
    public var valueLabelSmoothingEpsilon: Double { didSet { Self.persist(ValueLabelSmoothingEpsilon.self, value: valueLabelSmoothingEpsilon) } }
    public var gradClipMaxNorm: Double { didSet { Self.persist(GradClipMaxNorm.self, value: gradClipMaxNorm) } }
    public var weightDecay: Double { didSet { Self.persist(WeightDecay.self, value: weightDecay) } }
    public var policyLossWeight: Double { didSet { Self.persist(PolicyLossWeight.self, value: policyLossWeight) } }
    public var valueLossWeight: Double { didSet { Self.persist(ValueLossWeight.self, value: valueLossWeight) } }
    public var learningRate: Double { didSet { Self.persist(LearningRate.self, value: learningRate) } }
    public var momentumCoeff: Double { didSet { Self.persist(MomentumCoeff.self, value: momentumCoeff) } }
    public var sqrtBatchScalingLR: Bool { didSet { Self.persist(SqrtBatchScalingLR.self, value: sqrtBatchScalingLR) } }
    public var lrWarmupSteps: Int { didSet { Self.persist(LRWarmupSteps.self, value: lrWarmupSteps) } }
    public var drawPenalty: Double { didSet { Self.persist(DrawPenalty.self, value: drawPenalty) } }
    public var selfPlayStartTau: Double { didSet { Self.persist(SelfPlayStartTau.self, value: selfPlayStartTau) } }
    public var selfPlayTargetTau: Double { didSet { Self.persist(SelfPlayTargetTau.self, value: selfPlayTargetTau) } }
    public var selfPlayTauDecayPerPly: Double { didSet { Self.persist(SelfPlayTauDecayPerPly.self, value: selfPlayTauDecayPerPly) } }
    public var selfPlayDrawKeepFraction: Double { didSet { Self.persist(SelfPlayDrawKeepFraction.self, value: selfPlayDrawKeepFraction) } }
    public var selfPlayMaxPliesPerGame: Int { didSet { Self.persist(SelfPlayMaxPliesPerGame.self, value: selfPlayMaxPliesPerGame) } }
    public var arenaStartTau: Double { didSet { Self.persist(ArenaStartTau.self, value: arenaStartTau) } }
    public var arenaTargetTau: Double { didSet { Self.persist(ArenaTargetTau.self, value: arenaTargetTau) } }
    public var arenaTauDecayPerPly: Double { didSet { Self.persist(ArenaTauDecayPerPly.self, value: arenaTauDecayPerPly) } }
    public var replayRatioTarget: Double { didSet { Self.persist(ReplayRatioTarget.self, value: replayRatioTarget) } }
    public var replayRatioAutoAdjust: Bool { didSet { Self.persist(ReplayRatioAutoAdjust.self, value: replayRatioAutoAdjust) } }
    public var selfPlayConcurrency: Int { didSet { Self.persist(SelfPlayConcurrency.self, value: selfPlayConcurrency) } }
    public var trainingStepDelayMs: Int { didSet { Self.persist(TrainingStepDelayMs.self, value: trainingStepDelayMs) } }
    public var selfPlayDelayMs: Int { didSet { Self.persist(SelfPlayDelayMs.self, value: selfPlayDelayMs) } }
    public var trainingBatchSize: Int { didSet { Self.persist(TrainingBatchSize.self, value: trainingBatchSize) } }
    public var replayBufferCapacity: Int { didSet { Self.persist(ReplayBufferCapacity.self, value: replayBufferCapacity) } }
    public var replayBufferMinPositionsBeforeTraining: Int { didSet { Self.persist(ReplayBufferMinPositionsBeforeTraining.self, value: replayBufferMinPositionsBeforeTraining) } }
    public var maxPliesFromAnyOneGame: Int { didSet { Self.persist(MaxPliesFromAnyOneGame.self, value: maxPliesFromAnyOneGame) } }
    public var targetSampledGameLengthPlies: Int { didSet { Self.persist(TargetSampledGameLengthPlies.self, value: targetSampledGameLengthPlies) } }
    public var maxDrawPercentPerBatch: Int { didSet { Self.persist(MaxDrawPercentPerBatch.self, value: maxDrawPercentPerBatch) } }
    public var arenaPromoteThreshold: Double { didSet { Self.persist(ArenaPromoteThreshold.self, value: arenaPromoteThreshold) } }
    public var arenaGamesPerTournament: Int { didSet { Self.persist(ArenaGamesPerTournament.self, value: arenaGamesPerTournament) } }
    public var arenaAutoIntervalSec: Double { didSet { Self.persist(ArenaAutoIntervalSec.self, value: arenaAutoIntervalSec) } }
    public var candidateProbeIntervalSec: Double { didSet { Self.persist(CandidateProbeIntervalSec.self, value: candidateProbeIntervalSec) } }
    public var legalMassCollapseThreshold: Double { didSet { Self.persist(LegalMassCollapseThreshold.self, value: legalMassCollapseThreshold) } }
    public var legalMassCollapseGraceSeconds: Double { didSet { Self.persist(LegalMassCollapseGraceSeconds.self, value: legalMassCollapseGraceSeconds) } }
    public var legalMassCollapseNoImprovementProbes: Int { didSet { Self.persist(LegalMassCollapseNoImprovementProbes.self, value: legalMassCollapseNoImprovementProbes) } }
    public var arenaConcurrency: Int { didSet { Self.persist(ArenaConcurrency.self, value: arenaConcurrency) } }
    public var batchStatsInterval: Int { didSet { Self.persist(BatchStatsInterval.self, value: batchStatsInterval) } }

    private init() {
        // Read each value from UserDefaults (or definition default if absent / invalid).
        // didSet does not fire on initial assignment in init — which is what we want.
        self.entropyBonus = Self.read(EntropyBonus.self)
        self.illegalMassWeight = Self.read(IllegalMassWeight.self)
        self.policyLabelSmoothingEpsilon = Self.read(PolicyLabelSmoothingEpsilon.self)
        self.valueLabelSmoothingEpsilon = Self.read(ValueLabelSmoothingEpsilon.self)
        self.gradClipMaxNorm = Self.read(GradClipMaxNorm.self)
        self.weightDecay = Self.read(WeightDecay.self)
        self.policyLossWeight = Self.read(PolicyLossWeight.self)
        self.valueLossWeight = Self.read(ValueLossWeight.self)
        self.learningRate = Self.read(LearningRate.self)
        self.momentumCoeff = Self.read(MomentumCoeff.self)
        self.sqrtBatchScalingLR = Self.read(SqrtBatchScalingLR.self)
        self.lrWarmupSteps = Self.read(LRWarmupSteps.self)
        self.drawPenalty = Self.read(DrawPenalty.self)
        self.selfPlayStartTau = Self.read(SelfPlayStartTau.self)
        self.selfPlayTargetTau = Self.read(SelfPlayTargetTau.self)
        self.selfPlayTauDecayPerPly = Self.read(SelfPlayTauDecayPerPly.self)
        self.selfPlayDrawKeepFraction = Self.read(SelfPlayDrawKeepFraction.self)
        self.selfPlayMaxPliesPerGame = Self.read(SelfPlayMaxPliesPerGame.self)
        self.arenaStartTau = Self.read(ArenaStartTau.self)
        self.arenaTargetTau = Self.read(ArenaTargetTau.self)
        self.arenaTauDecayPerPly = Self.read(ArenaTauDecayPerPly.self)
        self.replayRatioTarget = Self.read(ReplayRatioTarget.self)
        self.replayRatioAutoAdjust = Self.read(ReplayRatioAutoAdjust.self)
        self.selfPlayConcurrency = Self.read(SelfPlayConcurrency.self)
        self.trainingStepDelayMs = Self.read(TrainingStepDelayMs.self)
        self.selfPlayDelayMs = Self.read(SelfPlayDelayMs.self)
        self.trainingBatchSize = Self.read(TrainingBatchSize.self)
        self.replayBufferCapacity = Self.read(ReplayBufferCapacity.self)
        self.replayBufferMinPositionsBeforeTraining = Self.read(ReplayBufferMinPositionsBeforeTraining.self)
        self.maxPliesFromAnyOneGame = Self.read(MaxPliesFromAnyOneGame.self)
        self.targetSampledGameLengthPlies = Self.read(TargetSampledGameLengthPlies.self)
        self.maxDrawPercentPerBatch = Self.read(MaxDrawPercentPerBatch.self)
        self.arenaPromoteThreshold = Self.read(ArenaPromoteThreshold.self)
        self.arenaGamesPerTournament = Self.read(ArenaGamesPerTournament.self)
        self.arenaAutoIntervalSec = Self.read(ArenaAutoIntervalSec.self)
        self.candidateProbeIntervalSec = Self.read(CandidateProbeIntervalSec.self)
        self.legalMassCollapseThreshold = Self.read(LegalMassCollapseThreshold.self)
        self.legalMassCollapseGraceSeconds = Self.read(LegalMassCollapseGraceSeconds.self)
        self.legalMassCollapseNoImprovementProbes = Self.read(LegalMassCollapseNoImprovementProbes.self)
        self.arenaConcurrency = Self.read(ArenaConcurrency.self)
        self.batchStatsInterval = Self.read(BatchStatsInterval.self)
    }

    // MARK: Snapshot

    public func snapshot() -> TrainingParametersSnapshot {
        TrainingParametersSnapshot(values: collectValues())
    }

    private func collectValues() -> [String: ParameterValue] {
        var v: [String: ParameterValue] = [:]
        v[EntropyBonus.id] = EntropyBonus.encode(entropyBonus)
        v[IllegalMassWeight.id] = IllegalMassWeight.encode(illegalMassWeight)
        v[PolicyLabelSmoothingEpsilon.id] = PolicyLabelSmoothingEpsilon.encode(policyLabelSmoothingEpsilon)
        v[ValueLabelSmoothingEpsilon.id] = ValueLabelSmoothingEpsilon.encode(valueLabelSmoothingEpsilon)
        v[GradClipMaxNorm.id] = GradClipMaxNorm.encode(gradClipMaxNorm)
        v[WeightDecay.id] = WeightDecay.encode(weightDecay)
        v[PolicyLossWeight.id] = PolicyLossWeight.encode(policyLossWeight)
        v[ValueLossWeight.id] = ValueLossWeight.encode(valueLossWeight)
        v[LearningRate.id] = LearningRate.encode(learningRate)
        v[MomentumCoeff.id] = MomentumCoeff.encode(momentumCoeff)
        v[SqrtBatchScalingLR.id] = SqrtBatchScalingLR.encode(sqrtBatchScalingLR)
        v[LRWarmupSteps.id] = LRWarmupSteps.encode(lrWarmupSteps)
        v[DrawPenalty.id] = DrawPenalty.encode(drawPenalty)
        v[SelfPlayStartTau.id] = SelfPlayStartTau.encode(selfPlayStartTau)
        v[SelfPlayTargetTau.id] = SelfPlayTargetTau.encode(selfPlayTargetTau)
        v[SelfPlayTauDecayPerPly.id] = SelfPlayTauDecayPerPly.encode(selfPlayTauDecayPerPly)
        v[SelfPlayDrawKeepFraction.id] = SelfPlayDrawKeepFraction.encode(selfPlayDrawKeepFraction)
        v[SelfPlayMaxPliesPerGame.id] = SelfPlayMaxPliesPerGame.encode(selfPlayMaxPliesPerGame)
        v[ArenaStartTau.id] = ArenaStartTau.encode(arenaStartTau)
        v[ArenaTargetTau.id] = ArenaTargetTau.encode(arenaTargetTau)
        v[ArenaTauDecayPerPly.id] = ArenaTauDecayPerPly.encode(arenaTauDecayPerPly)
        v[ReplayRatioTarget.id] = ReplayRatioTarget.encode(replayRatioTarget)
        v[ReplayRatioAutoAdjust.id] = ReplayRatioAutoAdjust.encode(replayRatioAutoAdjust)
        v[SelfPlayConcurrency.id] = SelfPlayConcurrency.encode(selfPlayConcurrency)
        v[TrainingStepDelayMs.id] = TrainingStepDelayMs.encode(trainingStepDelayMs)
        v[SelfPlayDelayMs.id] = SelfPlayDelayMs.encode(selfPlayDelayMs)
        v[TrainingBatchSize.id] = TrainingBatchSize.encode(trainingBatchSize)
        v[ReplayBufferCapacity.id] = ReplayBufferCapacity.encode(replayBufferCapacity)
        v[ReplayBufferMinPositionsBeforeTraining.id] = ReplayBufferMinPositionsBeforeTraining.encode(replayBufferMinPositionsBeforeTraining)
        v[MaxPliesFromAnyOneGame.id] = MaxPliesFromAnyOneGame.encode(maxPliesFromAnyOneGame)
        v[TargetSampledGameLengthPlies.id] = TargetSampledGameLengthPlies.encode(targetSampledGameLengthPlies)
        v[MaxDrawPercentPerBatch.id] = MaxDrawPercentPerBatch.encode(maxDrawPercentPerBatch)
        v[ArenaPromoteThreshold.id] = ArenaPromoteThreshold.encode(arenaPromoteThreshold)
        v[ArenaGamesPerTournament.id] = ArenaGamesPerTournament.encode(arenaGamesPerTournament)
        v[ArenaAutoIntervalSec.id] = ArenaAutoIntervalSec.encode(arenaAutoIntervalSec)
        v[CandidateProbeIntervalSec.id] = CandidateProbeIntervalSec.encode(candidateProbeIntervalSec)
        v[LegalMassCollapseThreshold.id] = LegalMassCollapseThreshold.encode(legalMassCollapseThreshold)
        v[LegalMassCollapseGraceSeconds.id] = LegalMassCollapseGraceSeconds.encode(legalMassCollapseGraceSeconds)
        v[LegalMassCollapseNoImprovementProbes.id] = LegalMassCollapseNoImprovementProbes.encode(legalMassCollapseNoImprovementProbes)
        v[ArenaConcurrency.id] = ArenaConcurrency.encode(arenaConcurrency)
        v[BatchStatsInterval.id] = BatchStatsInterval.encode(batchStatsInterval)
        return v
    }

    // MARK: Apply (from JSON load, from CLI)

    /// Applies a value map (e.g. from a parsed parameters.json) to the singleton.
    /// Each value goes through the typed setter so validation runs and SwiftUI sees the mutation.
    /// Unknown ids: throw `unknownParameter`. Out-of-range / wrong-type: throw the corresponding error.
    public func apply(_ values: [String: ParameterValue]) throws {
        for (id, raw) in values {
            try applyOne(id: id, raw: raw)
        }
    }

    private func applyOne(id: String, raw: ParameterValue) throws {
        switch id {
        case EntropyBonus.id:
            try EntropyBonus.definition.validate(raw); entropyBonus = try EntropyBonus.decode(raw)
        case IllegalMassWeight.id:
            try IllegalMassWeight.definition.validate(raw); illegalMassWeight = try IllegalMassWeight.decode(raw)
        case PolicyLabelSmoothingEpsilon.id:
            try PolicyLabelSmoothingEpsilon.definition.validate(raw); policyLabelSmoothingEpsilon = try PolicyLabelSmoothingEpsilon.decode(raw)
        case ValueLabelSmoothingEpsilon.id:
            try ValueLabelSmoothingEpsilon.definition.validate(raw); valueLabelSmoothingEpsilon = try ValueLabelSmoothingEpsilon.decode(raw)
        case GradClipMaxNorm.id:
            try GradClipMaxNorm.definition.validate(raw); gradClipMaxNorm = try GradClipMaxNorm.decode(raw)
        case WeightDecay.id:
            try WeightDecay.definition.validate(raw); weightDecay = try WeightDecay.decode(raw)
        case PolicyLossWeight.id:
            try PolicyLossWeight.definition.validate(raw); policyLossWeight = try PolicyLossWeight.decode(raw)
        case ValueLossWeight.id:
            try ValueLossWeight.definition.validate(raw); valueLossWeight = try ValueLossWeight.decode(raw)
        case LearningRate.id:
            try LearningRate.definition.validate(raw); learningRate = try LearningRate.decode(raw)
        case MomentumCoeff.id:
            try MomentumCoeff.definition.validate(raw); momentumCoeff = try MomentumCoeff.decode(raw)
        case SqrtBatchScalingLR.id:
            try SqrtBatchScalingLR.definition.validate(raw); sqrtBatchScalingLR = try SqrtBatchScalingLR.decode(raw)
        case LRWarmupSteps.id:
            try LRWarmupSteps.definition.validate(raw); lrWarmupSteps = try LRWarmupSteps.decode(raw)
        case DrawPenalty.id:
            try DrawPenalty.definition.validate(raw); drawPenalty = try DrawPenalty.decode(raw)
        case SelfPlayStartTau.id:
            try SelfPlayStartTau.definition.validate(raw); selfPlayStartTau = try SelfPlayStartTau.decode(raw)
        case SelfPlayTargetTau.id:
            try SelfPlayTargetTau.definition.validate(raw); selfPlayTargetTau = try SelfPlayTargetTau.decode(raw)
        case SelfPlayTauDecayPerPly.id:
            try SelfPlayTauDecayPerPly.definition.validate(raw); selfPlayTauDecayPerPly = try SelfPlayTauDecayPerPly.decode(raw)
        case SelfPlayDrawKeepFraction.id:
            try SelfPlayDrawKeepFraction.definition.validate(raw); selfPlayDrawKeepFraction = try SelfPlayDrawKeepFraction.decode(raw)
        case SelfPlayMaxPliesPerGame.id:
            try SelfPlayMaxPliesPerGame.definition.validate(raw); selfPlayMaxPliesPerGame = try SelfPlayMaxPliesPerGame.decode(raw)
        case ArenaStartTau.id:
            try ArenaStartTau.definition.validate(raw); arenaStartTau = try ArenaStartTau.decode(raw)
        case ArenaTargetTau.id:
            try ArenaTargetTau.definition.validate(raw); arenaTargetTau = try ArenaTargetTau.decode(raw)
        case ArenaTauDecayPerPly.id:
            try ArenaTauDecayPerPly.definition.validate(raw); arenaTauDecayPerPly = try ArenaTauDecayPerPly.decode(raw)
        case ReplayRatioTarget.id:
            try ReplayRatioTarget.definition.validate(raw); replayRatioTarget = try ReplayRatioTarget.decode(raw)
        case ReplayRatioAutoAdjust.id:
            try ReplayRatioAutoAdjust.definition.validate(raw); replayRatioAutoAdjust = try ReplayRatioAutoAdjust.decode(raw)
        case SelfPlayConcurrency.id:
            try SelfPlayConcurrency.definition.validate(raw); selfPlayConcurrency = try SelfPlayConcurrency.decode(raw)
        case TrainingStepDelayMs.id:
            try TrainingStepDelayMs.definition.validate(raw); trainingStepDelayMs = try TrainingStepDelayMs.decode(raw)
        case SelfPlayDelayMs.id:
            try SelfPlayDelayMs.definition.validate(raw); selfPlayDelayMs = try SelfPlayDelayMs.decode(raw)
        case TrainingBatchSize.id:
            try TrainingBatchSize.definition.validate(raw); trainingBatchSize = try TrainingBatchSize.decode(raw)
        case ReplayBufferCapacity.id:
            try ReplayBufferCapacity.definition.validate(raw); replayBufferCapacity = try ReplayBufferCapacity.decode(raw)
        case ReplayBufferMinPositionsBeforeTraining.id:
            try ReplayBufferMinPositionsBeforeTraining.definition.validate(raw); replayBufferMinPositionsBeforeTraining = try ReplayBufferMinPositionsBeforeTraining.decode(raw)
        case MaxPliesFromAnyOneGame.id:
            try MaxPliesFromAnyOneGame.definition.validate(raw); maxPliesFromAnyOneGame = try MaxPliesFromAnyOneGame.decode(raw)
        case TargetSampledGameLengthPlies.id:
            try TargetSampledGameLengthPlies.definition.validate(raw); targetSampledGameLengthPlies = try TargetSampledGameLengthPlies.decode(raw)
        case MaxDrawPercentPerBatch.id:
            try MaxDrawPercentPerBatch.definition.validate(raw); maxDrawPercentPerBatch = try MaxDrawPercentPerBatch.decode(raw)
        case ArenaPromoteThreshold.id:
            try ArenaPromoteThreshold.definition.validate(raw); arenaPromoteThreshold = try ArenaPromoteThreshold.decode(raw)
        case ArenaGamesPerTournament.id:
            try ArenaGamesPerTournament.definition.validate(raw); arenaGamesPerTournament = try ArenaGamesPerTournament.decode(raw)
        case ArenaAutoIntervalSec.id:
            try ArenaAutoIntervalSec.definition.validate(raw); arenaAutoIntervalSec = try ArenaAutoIntervalSec.decode(raw)
        case CandidateProbeIntervalSec.id:
            try CandidateProbeIntervalSec.definition.validate(raw); candidateProbeIntervalSec = try CandidateProbeIntervalSec.decode(raw)
        case LegalMassCollapseThreshold.id:
            try LegalMassCollapseThreshold.definition.validate(raw); legalMassCollapseThreshold = try LegalMassCollapseThreshold.decode(raw)
        case LegalMassCollapseGraceSeconds.id:
            try LegalMassCollapseGraceSeconds.definition.validate(raw); legalMassCollapseGraceSeconds = try LegalMassCollapseGraceSeconds.decode(raw)
        case LegalMassCollapseNoImprovementProbes.id:
            try LegalMassCollapseNoImprovementProbes.definition.validate(raw); legalMassCollapseNoImprovementProbes = try LegalMassCollapseNoImprovementProbes.decode(raw)
        case ArenaConcurrency.id:
            try ArenaConcurrency.definition.validate(raw); arenaConcurrency = try ArenaConcurrency.decode(raw)
        case BatchStatsInterval.id:
            try BatchStatsInterval.definition.validate(raw); batchStatsInterval = try BatchStatsInterval.decode(raw)
        default:
            throw TrainingConfigError.unknownParameter(id: id)
        }
    }

    // MARK: Persistence (per-key UserDefaults)

    private nonisolated static func read<K: TrainingParameterKey>(_ key: K.Type) -> K.Value {
        let defaults = UserDefaults.standard

        if let object = defaults.object(forKey: K.id) {
            switch K.definition.type {
            case .bool:
                if let b = object as? Bool {
                    let raw: ParameterValue = .bool(b)
                    if (try? K.definition.validate(raw)) != nil,
                       let decoded = try? K.decode(raw) {
                        return decoded
                    }
                }
            case .int:
                if let n = object as? NSNumber {
                    let raw: ParameterValue = .int(n.intValue)
                    if (try? K.definition.validate(raw)) != nil,
                       let decoded = try? K.decode(raw) {
                        return decoded
                    }
                }
            case .double:
                if let n = object as? NSNumber {
                    let raw: ParameterValue = .double(n.doubleValue)
                    if (try? K.definition.validate(raw)) != nil,
                       let decoded = try? K.decode(raw) {
                        return decoded
                    }
                }
            }
        }
        // Fall back to definition default. If the baked-in default doesn't
        // round-trip, that's a programmer error — surface it rather than
        // crash with an opaque `try!`.
        do {
            return try K.decode(K.definition.defaultValue)
        } catch {
            preconditionFailure("default value for \(K.id) does not round-trip through decode: \(error)")
        }
    }

    private nonisolated static func persist<K: TrainingParameterKey>(_ key: K.Type, value: K.Value) {
        let raw = K.encode(value)
        if (try? K.definition.validate(raw)) == nil {
            // Validation failed — programmer or CLI wrote a bad value through the typed setter.
            // Per the design this is fatal: typed setters should only be reached after validation.
            // The applyOne path validates explicitly before assignment; UI controls clamp.
            // If this fires in practice, the ContentView control is missing a clamp.
            assertionFailure("TrainingParameters.persist: validation failed for \(K.id) value=\(value)")
            return
        }
        let defaults = UserDefaults.standard
        switch raw {
        case .bool(let x): defaults.set(x, forKey: K.id)
        case .int(let x): defaults.set(x, forKey: K.id)
        case .double(let x): defaults.set(x, forKey: K.id)
        }
    }

    // MARK: Registry

    /// All keys, in declaration order. Used by save/load and by the `--show-default-parameters` CLI flag.
    public nonisolated static let allKeys: [any TrainingParameterKey.Type] = [
        EntropyBonus.self,
        IllegalMassWeight.self,
        PolicyLabelSmoothingEpsilon.self,
        ValueLabelSmoothingEpsilon.self,
        GradClipMaxNorm.self,
        WeightDecay.self,
        PolicyLossWeight.self,
        ValueLossWeight.self,
        LearningRate.self,
        MomentumCoeff.self,
        SqrtBatchScalingLR.self,
        LRWarmupSteps.self,
        DrawPenalty.self,
        SelfPlayStartTau.self,
        SelfPlayTargetTau.self,
        SelfPlayTauDecayPerPly.self,
        SelfPlayDrawKeepFraction.self,
        SelfPlayMaxPliesPerGame.self,
        ArenaStartTau.self,
        ArenaTargetTau.self,
        ArenaTauDecayPerPly.self,
        ReplayRatioTarget.self,
        ReplayRatioAutoAdjust.self,
        SelfPlayConcurrency.self,
        TrainingStepDelayMs.self,
        SelfPlayDelayMs.self,
        TrainingBatchSize.self,
        ReplayBufferCapacity.self,
        ReplayBufferMinPositionsBeforeTraining.self,
        MaxPliesFromAnyOneGame.self,
        TargetSampledGameLengthPlies.self,
        MaxDrawPercentPerBatch.self,
        ArenaPromoteThreshold.self,
        ArenaGamesPerTournament.self,
        ArenaAutoIntervalSec.self,
        CandidateProbeIntervalSec.self,
        LegalMassCollapseThreshold.self,
        LegalMassCollapseGraceSeconds.self,
        LegalMassCollapseNoImprovementProbes.self,
        ArenaConcurrency.self,
        BatchStatsInterval.self
    ]

    public nonisolated static var allDefinitions: [TrainingParameterDefinition] {
        allKeys.map { $0.definition }
    }

    // MARK: Defaults emit (used by --show-default-parameters and --create-parameters-file)

    /// Emit a flat `{snake_case_key: jsonValue}` JSON object representing definition defaults.
    /// Pretty-printed and sorted-key for stable diffs. Used by `--show-default-parameters`
    /// and `--create-parameters-file`. Synchronous, never touches the singleton.
    public nonisolated static func defaultsJSON() throws -> Data {
        var dict: [String: Any] = [:]
        for key in allKeys {
            let def = key.definition
            switch def.defaultValue {
            case .bool(let x): dict[def.id] = x
            case .int(let x): dict[def.id] = x
            case .double(let x): dict[def.id] = x
            }
        }
        return try JSONSerialization.data(
            withJSONObject: dict,
            options: [.prettyPrinted, .sortedKeys]
        )
    }

    /// Per-parameter description lines for stderr in `--show-default-parameters`.
    public nonisolated static func defaultsDescriptionLines() -> [String] {
        allDefinitions.map { def in
            let rangeText: String
            switch def.type {
            case .bool:
                rangeText = "Bool"
            case .int:
                if let r = def.intRange {
                    rangeText = "Int, range \(r.min)..\(r.max)"
                } else {
                    rangeText = "Int"
                }
            case .double:
                if let r = def.doubleRange {
                    rangeText = "Double, range \(r.min)..\(r.max)"
                } else {
                    rangeText = "Double"
                }
            }
            return "\(def.id): \(def.description) (\(rangeText))"
        }
    }

    /// Categorized markdown for `--create-parameters-file` to write next to `parameters.json`.
    public nonisolated static func defaultsMarkdown() -> String {
        var out = "# DrewsChessMachine training parameters\n\n"
        out += "Generated by `DrewsChessMachine --create-parameters-file`. Edit values in `parameters.json`; this file is reference only.\n\n"

        // Group by category, preserving the order in `allKeys`.
        var seenCategories: [String] = []
        var byCategory: [String: [TrainingParameterDefinition]] = [:]
        for def in allDefinitions {
            if byCategory[def.category] == nil {
                seenCategories.append(def.category)
                byCategory[def.category] = []
            }
            byCategory[def.category]?.append(def)
        }

        for category in seenCategories {
            out += "## \(category)\n\n"
            for def in byCategory[category] ?? [] {
                out += "### \(def.id)\n\n"
                out += "\(def.description)\n\n"
                let typeText: String
                let rangeText: String
                let defaultText: String
                switch def.type {
                case .bool:
                    typeText = "Bool"; rangeText = "—"
                    if case .bool(let x) = def.defaultValue { defaultText = "\(x)" } else { defaultText = "?" }
                case .int:
                    typeText = "Int"
                    rangeText = def.intRange.map { "\($0.min)..\($0.max)" } ?? "—"
                    if case .int(let x) = def.defaultValue { defaultText = "\(x)" } else { defaultText = "?" }
                case .double:
                    typeText = "Double"
                    rangeText = def.doubleRange.map { "\($0.min)..\($0.max)" } ?? "—"
                    if case .double(let x) = def.defaultValue { defaultText = "\(x)" } else { defaultText = "?" }
                }
                out += "**Type:** \(typeText) · **Range:** \(rangeText) · **Default:** \(defaultText)"
                if def.liveTunable {
                    out += " · **Live-tunable** (mid-session UI changes propagate to the running trainer)"
                }
                out += "\n\n"
            }
        }

        return out
    }

    // MARK: Pretty JSON load / save (current values, not defaults)

    public func save(to url: URL) throws {
        let snap = collectValues()
        var dict: [String: Any] = [:]
        for (id, raw) in snap {
            switch raw {
            case .bool(let x): dict[id] = x
            case .int(let x): dict[id] = x
            case .double(let x): dict[id] = x
            }
        }
        let data = try JSONSerialization.data(
            withJSONObject: dict,
            options: [.prettyPrinted, .sortedKeys]
        )
        try data.write(to: url, options: [.atomic])
    }

    public func load(from url: URL) throws {
        let data = try Data(contentsOf: url)
        guard let dict = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw TrainingConfigError.wrongType(id: "<root>")
        }
        var values: [String: ParameterValue] = [:]
        for (id, anyValue) in dict {
            if let b = anyValue as? Bool {
                values[id] = .bool(b)
            } else if let n = anyValue as? NSNumber {
                // Distinguish Int from Double via objCType — ".d" / ".f" are floats.
                let typeChar = String(cString: n.objCType)
                if typeChar == "d" || typeChar == "f" {
                    values[id] = .double(n.doubleValue)
                } else {
                    values[id] = .int(n.intValue)
                }
            } else {
                throw TrainingConfigError.wrongType(id: id)
            }
        }
        try apply(values)
    }
}
