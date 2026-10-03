import Foundation
import Observation
import TrainingParametersMacroSupport

// MARK: - ParameterType

public enum ParameterType: String, Codable, Sendable {
    case bool
    case int
    case double
    /// Full-range unsigned 64-bit (a random seed). Written as a decimal
    /// string everywhere it is serialized (see `ParameterValue.jsonValue`).
    case uint64
}

// MARK: - ParameterValue

public enum ParameterValue: Codable, Equatable, Sendable {
    case bool(Bool)
    case int(Int)
    case double(Double)
    case uint64(UInt64)

    public var type: ParameterType {
        switch self {
        case .bool: .bool
        case .int: .int
        case .double: .double
        case .uint64: .uint64
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
        } else if let text = try? c.decode(String.self) {
            // The only string-encoded kind: a full-range UInt64 written as a
            // decimal string so no reader rounds it through a Double.
            guard let x = UInt64(strictDecimal: text) else {
                throw DecodingError.dataCorruptedError(
                    in: c, debugDescription: "\"\(text)\" is not a decimal UInt64 parameter value"
                )
            }
            self = .uint64(x)
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
        case .uint64(let x): try c.encode(String(x))
        }
    }

    /// The value as a `JSONSerialization` object: the number or Bool itself,
    /// and a UInt64 as its decimal string (JSON numbers are doubles in many
    /// readers, which would silently drop the low bits of a seed above 2^53).
    /// The one encoding every JSON writer of parameters uses.
    public var jsonValue: Any {
        switch self {
        case .bool(let x): x
        case .int(let x): x
        case .double(let x): x
        case .uint64(let x): String(x)
        }
    }

    /// Parse one value of a parameters JSON object (`JSONSerialization`
    /// output) — the one reader both the `--parameters` loader and the
    /// settings "load" path use. `NSNumber` bridging is treacherous (`as? Bool`
    /// succeeds for any number, so `1` would read as `true`), so the kind is
    /// taken from the number's `objCType`: true/false are char-typed
    /// ("c"/"B"), JSON doubles are "d"/"f", everything else is an integer. A
    /// string is accepted only as a decimal UInt64, digits only
    /// (`UInt64(strictDecimal:)`). Whether the kind matches
    /// the key is checked later, by the definition's `validate`.
    public init(jsonValue: Any, id: String) throws {
        if let n = jsonValue as? NSNumber {
            switch String(cString: n.objCType) {
            case "c", "B": self = .bool(n.boolValue)
            case "d", "f": self = .double(n.doubleValue)
            // An unsigned 64-bit number: `intValue` would wrap one above
            // `Int.max` to a negative value, so that one is read as unsigned.
            case "Q":
                let unsigned = n.uint64Value
                self = unsigned <= UInt64(Int.max) ? .int(Int(unsigned)) : .uint64(unsigned)
            default: self = .int(n.intValue)
            }
        } else if let text = jsonValue as? String {
            guard let x = UInt64(strictDecimal: text) else { throw TrainingConfigError.wrongType(id: id) }
            self = .uint64(x)
        } else {
            throw TrainingConfigError.wrongType(id: id)
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
    public let uint64Range: NumericRange<UInt64>?
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
        uint64Range: NumericRange<UInt64>? = nil,
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
        self.uint64Range = uint64Range
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

        case (.uint64, .uint64(let x)):
            if let uint64Range, !uint64Range.contains(x) {
                throw TrainingConfigError.outOfRange(id: id, value: "\(x)")
            }

        case (.uint64, .int(let x)) where x >= 0:
            // Tolerate a non-negative JSON integer for a UInt64 parameter
            // (exact up to `Int.max`); the app itself writes a decimal string.
            if let uint64Range, !uint64Range.contains(UInt64(x)) {
                throw TrainingConfigError.outOfRange(id: id, value: "\(x)")
            }

        default:
            throw TrainingConfigError.wrongType(id: id)
        }
    }
}

// MARK: - TrainingConfigError

/// `LocalizedError` so `error.localizedDescription` — what every CLI error
/// path prints — carries the real message ("Value 0.02 is out of range for
/// parameter 'self_play_target_tau'") instead of Foundation's generic
/// "(DrewsChessMachine.TrainingConfigError error 2.)".
public enum TrainingConfigError: Error, CustomStringConvertible, LocalizedError {
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

    public var errorDescription: String? { description }
}

// MARK: - TrainingParameterKey

public protocol TrainingParameterKey: Sendable {
    associatedtype Value: Sendable & Equatable
    static var id: String { get }
    static var definition: TrainingParameterDefinition { get }
    static func encode(_ value: Value) -> ParameterValue
    static func decode(_ value: ParameterValue) throws -> Value
    /// What a session resume applies when the saved checkpoint carries no
    /// value for this key — declared per key through `@TrainingParameter`'s
    /// `absentValue:` and applied by `TrainingParameterResolution` (the one
    /// resolver every resume path uses).
    static var absentValue: TrainingParameterAbsence<Value> { get }
}

// MARK: - Declared-range validation (the one validator)
//
// Every writer of a training parameter checks the value against the range
// declared in its `@TrainingParameter` — the single source of truth — through
// these helpers: the `--parameters` loader (`applyOne`), the UserDefaults load
// (`read`), the singleton's setters (`commitAssignment`) and the settings
// popovers (`parsedInDeclaredRange` / `snappedToDeclaredRange`). Before they
// existed the popovers restated each range as literals, and several had
// drifted from the declarations (the τ fields accepted values below the
// declared minimum), so the UI accepted values the loader rejects. The one
// intentional exception is session resume, which restores a session's own
// saved values even outside today's range — see
// `TrainingParameters.restoreFromSession`.

public extension TrainingParameterKey {
    /// Throw `TrainingConfigError` unless `value` has the declared type and
    /// lies inside the declared range.
    static func validateAgainstDeclaration(_ value: Value) throws {
        try definition.validate(encode(value))
    }

    /// True when `value` passes `validateAgainstDeclaration(_:)`.
    static func isWithinDeclaration(_ value: Value) -> Bool {
        do {
            try validateAgainstDeclaration(value)
            return true
        } catch {
            return false
        }
    }

    /// The default declared in the key's `@TrainingParameter`, as the key's
    /// typed value. For controls (an edit field's placeholder, a stepper's
    /// value when its text does not parse) that must show the real default
    /// rather than restate it as a literal — those copies drift every time the
    /// declared default changes. A default that does not decode as its own
    /// key's type is a programmer error in the declaration, so it traps.
    static var declaredDefault: Value {
        do {
            return try decode(definition.defaultValue)
        } catch {
            preconditionFailure("default value for \(id) does not round-trip through decode: \(error)")
        }
    }
}

public extension TrainingParameterKey where Value == Double {
    /// Parse an edit field: the finite number it holds if that number is
    /// inside the declared range, else nil (the caller flags the field).
    static func parsedInDeclaredRange(_ text: String) -> Double? {
        guard let value = Double(text.trimmingCharacters(in: .whitespaces)),
              value.isFinite,
              isWithinDeclaration(value) else { return nil }
        return value
    }

    /// The declared range as a `ClosedRange`, for controls (a `Stepper`'s
    /// bounds) that must agree with the validator rather than restate the
    /// range as literals. Only for keys declared with a range.
    static var declaredClosedRange: ClosedRange<Double> {
        guard let range = definition.doubleRange else {
            preconditionFailure("\(id) is declared without a range")
        }
        return range.min...range.max
    }

    /// `value` pulled inside the declared range. Only for stepper-style
    /// controls whose arrows can overshoot an end; typed values go through
    /// `parsedInDeclaredRange(_:)` and are rejected, not clamped.
    static func snappedToDeclaredRange(_ value: Double) -> Double {
        guard let range = definition.doubleRange else { return value }
        return Swift.min(range.max, Swift.max(range.min, value))
    }
}

public extension TrainingParameterKey where Value == Int {
    /// Parse an edit field: the integer it holds if that integer is inside
    /// the declared range, else nil (the caller flags the field).
    static func parsedInDeclaredRange(_ text: String) -> Int? {
        guard let value = Int(text.trimmingCharacters(in: .whitespaces)),
              isWithinDeclaration(value) else { return nil }
        return value
    }

    /// The declared range as a `ClosedRange`, for controls (a `Stepper`'s
    /// bounds) that must agree with the validator rather than restate the
    /// range as literals. Only for keys declared with a range.
    static var declaredClosedRange: ClosedRange<Int> {
        guard let range = definition.intRange else {
            preconditionFailure("\(id) is declared without a range")
        }
        return range.min...range.max
    }

    /// `value` pulled inside the declared range. Only for stepper-style
    /// controls; typed values go through `parsedInDeclaredRange(_:)`.
    static func snappedToDeclaredRange(_ value: Int) -> Int {
        guard let range = definition.intRange else { return value }
        return Swift.min(range.max, Swift.max(range.min, value))
    }
}

public extension TrainingParameterKey where Value == UInt64 {
    /// Parse an edit field: the decimal UInt64 it holds (digits only, after
    /// trimming surrounding spaces) if that value is inside the declared
    /// range, else nil (the caller flags the field).
    static func parsedInDeclaredRange(_ text: String) -> UInt64? {
        guard let value = UInt64(strictDecimal: text.trimmingCharacters(in: .whitespaces)),
              isWithinDeclaration(value) else { return nil }
        return value
    }
}

// MARK: - Parameter keys (macro-driven)

@TrainingParameter(
    name: "Entropy Bonus",
    description: "Entropy regularization coefficient. Higher keeps the policy diverse longer; too high stalls learning.",
    default: 0.0,
    range: 0.0...0.1,
    category: "Optimizer",
    liveTunable: true,
    absentValue: .refuseExact
)
public enum EntropyBonus: TrainingParameterKey {}

@TrainingParameter(
    name: "Illegal Mass Penalty Weight",
    description: "Weight for the penalty term that pushes probability mass off illegal moves. Start at 1.0; increase if illegal mass leaks.",
    default: 1.0,
    range: 0.0...100.0,
    category: "Optimizer",
    liveTunable: true,
    absentValue: .preFeature(0.0)
)
public enum IllegalMassWeight: TrainingParameterKey {}

@TrainingParameter(
    name: "Policy Label Smoothing ε",
    description: "Label-smoothing coefficient on the policy CE target. ε=0 = one-hot played-move target (original behavior). ε=0.1 = (1−ε) at played + ε spread uniformly over legal moves; caps per-position concentration at p(played)=1−ε, structurally prevents delta-policy collapse. Range [0, 0.9].",
    default: 0.1,
    range: 0.0...0.9,
    category: "Optimizer",
    liveTunable: true,
    absentValue: .preFeature(0.0)
)
public enum PolicyLabelSmoothingEpsilon: TrainingParameterKey {}

// MARK: Policy label smoothing mode (fixed total vs per move)
//
// `Policy Label Smoothing ε` is read only in fixed-total mode; the per-move
// mass δ and its cap are read only in per-move mode. All three are fed to the
// training graph every step (like ε), so all are live-tunable and a mode
// switch takes effect on the next step. See `PolicyLabelSmoothingMode` and
// `documentation/plans-active/POLICY_LABEL_SMOOTHING_EXPERIMENTS.md`.

@TrainingParameter(
    name: "Policy Label Smoothing Mode",
    description: "How the policy CE target spreads its label-smoothing mass over the legal moves: 0 = fixed total (Policy Label Smoothing ε is the total, split evenly over all legal moves, so the mass per alternative shrinks as the number of legal moves grows), 1 = per move (every non-played legal move gets Policy Label Smoothing Per Move δ, so the total grows with the number of legal moves, capped at Policy Label Smoothing Per Move Cap and shared equally above it). Fixed total ignores δ and the cap; per move ignores ε.",
    default: 0,
    range: 0...1,
    category: "Optimizer",
    id: "policy_label_smoothing_mode",
    liveTunable: true,
    absentValue: .preFeature(0)
)
public enum PolicyLabelSmoothingModeParameter: TrainingParameterKey {}

@TrainingParameter(
    name: "Policy Label Smoothing Per Move δ",
    description: "Per-move label-smoothing mass, used only when Policy Label Smoothing Mode = 1 (per move). Every non-played legal move gets δ and the played move gets the rest: target = (1 − δ·(n−1))·one_hot(played) + δ·(other legal moves), n = number of legal moves, so the trained gap between the played move and each alternative is nearly independent of n. The total δ·(n−1) is capped by Policy Label Smoothing Per Move Cap. A position with one legal move gets an exact one-hot. δ=0 = one-hot. Range [0, 0.05].",
    default: 0.0033,
    range: 0.0...0.05,
    category: "Optimizer",
    liveTunable: true,
    absentValue: .currentSetting
)
public enum PolicyLabelSmoothingPerMove: TrainingParameterKey {}

@TrainingParameter(
    name: "Policy Label Smoothing Per Move Cap",
    description: "Cap on the total per-move smoothing mass δ·(n−1), used only when Policy Label Smoothing Mode = 1 (per move). Wide positions (up to ~218 legal moves) would otherwise give the played move little or no target mass. Above the cap the capped total is shared equally over the non-played legal moves. In the complement (negative-advantage) target the played move gets the same per-alternative mass, min(δ, cap/(n−1)), so it never gets more than any other legal move. Same ceiling as Policy Label Smoothing ε. Range [0, 0.9].",
    default: 0.5,
    range: 0.0...0.9,
    category: "Optimizer",
    liveTunable: true,
    absentValue: .currentSetting
)
public enum PolicyLabelSmoothingPerMoveCap: TrainingParameterKey {}

@TrainingParameter(
    name: "Value Label Smoothing ε",
    description: "Label-smoothing coefficient on the value-head W/D/L cross-entropy target. The target is built in-graph as (1−ε)·one_hot(1−z) + ε·(1/3), where 1−z maps the play-time outcome z ∈ {+1,0,−1} to the [win,draw,loss] slot. ε=0 = hard one-hot on the game result. ε>0 gives the value CE a reachable finite-logit equilibrium instead of ±∞, the same way Policy Label Smoothing does for the policy CE. Range [0, 0.5].",
    default: 0.013,
    range: 0.0...0.5,
    category: "Optimizer",
    liveTunable: true,
    absentValue: .preFeature(0.0)
)
public enum ValueLabelSmoothingEpsilon: TrainingParameterKey {}

@TrainingParameter(
    name: "Gradient Clip Max Norm",
    description: "Global L2 norm cap for gradient clipping. Above this, gradients are scaled down before the SGD step.",
    default: 15.0,
    range: 0.1...1000.0,
    category: "Optimizer",
    liveTunable: true,
    absentValue: .refuseExact
)
public enum GradClipMaxNorm: TrainingParameterKey {}

@TrainingParameter(
    name: "Weight Decay",
    description: "L2 weight decay coefficient. Couples with batch size and the number of update steps per epoch.",
    default: 0.0003,
    range: 0.0...0.1,
    category: "Optimizer",
    liveTunable: true,
    absentValue: .refuseExact
)
public enum WeightDecay: TrainingParameterKey {}

@TrainingParameter(
    name: "Dropout Rate",
    description: "Probability that a residual-branch CHANNEL is dropped (spatial dropout at the WRN slot inside every block, between the conv2-side activation and conv2; inverted scaling so train-time expectations match the dropout-free inference graphs). 0 disables: the graph nodes are exact identities at measurably zero cost. NOTE this is the DROP probability (PyTorch/Keras convention) — the 2014 paper and older tutorials quote the complement (retention p), so their p=0.5-0.8 equals 0.2-0.5 here. Conv-channel dropout typically wants far less than FC-era defaults; with continuous fresh self-play data there is little overfitting to fight, so treat nonzero values as an experiment, not a default.",
    default: 0.0,
    range: 0.0...0.95,
    category: "Optimizer",
    liveTunable: true,
    absentValue: .preFeature(0.0)
)
public enum DropoutRate: TrainingParameterKey {}

@TrainingParameter(
    name: "Policy Loss Weight",
    description: "Per-component weighting on the POLICY-LOSS TENSOR inside total_loss = valueLossWeight · valueLoss + policyLossWeight · policyLoss − entropyCoeff · policyEntropy. Pairs with valueLossWeight. Higher values shift shared-trunk gradients toward policy fitting and away from the value head's W/D/L cross-entropy: at policyLossWeight = valueLossWeight = 1 the trunk is pulled equally by both heads (AlphaZero canonical); at policyLossWeight=5+ the policy head dominates trunk shaping and the value head trails. NOT a multiplier on policy logits — that's a common misreading. Without MCTS-quality policy targets (this engine has none), values above ~3 amplify policy-target noise faster than the value head can supply a useful baseline.",
    default: 1.0,
    range: 0.0...20.0,
    category: "Optimizer",
    id: "policy_loss_weight",
    liveTunable: true,
    absentValue: .refuseExact
)
public enum PolicyLossWeight: TrainingParameterKey {}

@TrainingParameter(
    name: "Value Loss Weight",
    description: "Per-component weighting on the VALUE-LOSS TENSOR inside total_loss = valueLossWeight · valueLoss + policyLossWeight · policyLoss − entropyCoeff · policyEntropy. Mirrors policyLossWeight: at 1.0 the value head's W/D/L categorical-cross-entropy term feeds the trunk at its natural magnitude (AlphaZero canonical); raising it makes the trunk prioritize value fitting over policy fitting. Lc0/KataGo expose the same knob as `value_loss_weight`. The two weights only matter relative to each other plus the entropy term — scaling both by the same factor is equivalent to scaling the learning rate. NOTE: post-2026-05-12 the value loss is CE-scale (~[0, ln 3] at convergence), not the old MSE scale (~0.1–0.4), so the value term's contribution is a bit larger at the same weight — consider a value lower than 1.0 for early WDL runs.",
    default: 1.0,
    range: 0.0...20.0,
    category: "Optimizer",
    id: "value_loss_weight",
    liveTunable: true,
    absentValue: .refuseExact
)
public enum ValueLossWeight: TrainingParameterKey {}

@TrainingParameter(
    name: "Learning Rate",
    description: "SGD-with-momentum optimizer learning rate. Lower is slower but more stable. Pairs with sqrt_batch_scaling_lr. Note: under the bf16 weight path, updates below the bf16 weight ULP (~0.8% of a weight's magnitude) round away, so LRs much below ~1e-3 are largely no-ops.",
    default: 1.0e-3,
    range: 1.0e-7...1.0,
    category: "Optimizer",
    liveTunable: true,
    absentValue: .refuseExact
)
public enum LearningRate: TrainingParameterKey {}

@TrainingParameter(
    name: "Momentum Coefficient",
    description: "Polyak momentum μ for SGD. 0.0 disables momentum (pure SGD); higher μ accumulates more gradient history. The optimizer uses decoupled weight decay (AdamW-style), so μ and Weight Decay tune independently — raising μ no longer amplifies decay. Effective step size in correlated-gradient regimes still scales ~1/(1−μ), so a high μ paired with the existing LR can be too aggressive — pair μ jumps with a proportional LR drop. Start low (≤0.5) and watch legalMass / pEntLegal before raising further.",
    default: 0.9,
    range: 0.0...0.99,
    category: "Optimizer",
    liveTunable: true,
    absentValue: .preFeature(0.0)
)
public enum MomentumCoeff: TrainingParameterKey {}

@TrainingParameter(
    name: "Sqrt-Batch Scaling LR",
    description: "When true, scales the effective learning rate by sqrt(batch / referenceBatch). Standard practice when scaling SGD-with-momentum by batch size.",
    default: true,
    category: "Optimizer",
    liveTunable: true,
    absentValue: .refuseExact
)
public enum SqrtBatchScalingLR: TrainingParameterKey {}

@TrainingParameter(
    name: "Signed-Advantage Complement CE",
    description: "When true, the policy gradient runs two cross-entropies — positive-advantage samples teach via standard smoothed CE on the played move, negative-advantage samples teach via a complementary smoothed CE that pushes mass off the played move toward the OTHER legal moves. Each contribution is bounded below by zero so the total policy loss stays bounded below by zero on both signs. When false, only positive-advantage samples teach the policy (legacy clamp-on regime) and negative samples contribute zero gradient. Note: complement-target entropy is structurally higher than positive-target entropy (the (1−ε) main mass spreads over (|legal|−1) cells vs 1 cell), so the per-position negative-branch loss magnitudes will look larger in [STATS] — that's the target geometry, not a divergence signal.",
    default: true,
    category: "Optimizer",
    liveTunable: true,
    absentValue: .preFeature(false)
)
public enum SignedAdvantageComplementCE: TrainingParameterKey {}

@TrainingParameter(
    name: "LR Warmup Steps",
    description: "Number of training steps over which the learning rate linearly ramps from zero to its target.",
    default: 1000,
    range: 0...100000,
    category: "Optimizer",
    liveTunable: true,
    absentValue: .refuseExact
)
public enum LRWarmupSteps: TrainingParameterKey {}

@TrainingParameter(
    name: "Draw Penalty",
    description: "Contempt factor. When greater than 0, each drawn game's training outcome is rewritten from 0 to −drawPenalty before it reaches the trainer, so draws train as partial losses and the network is discouraged from drawing. 0 (default) leaves draws neutral; 1.0 trains draws as full losses. Negative values have no effect.",
    default: 0.0,
    range: 0.0...1.0,
    category: "Optimizer",
    liveTunable: true,
    absentValue: .refuseExact
)
public enum DrawPenalty: TrainingParameterKey {}

@TrainingParameter(
    name: "Self-Play Start Tau",
    description: "Initial sampling temperature for self-play games at game-total ply 0 (the starting position). Decays toward target by self_play_tau_decay_per_ply each game-total ply (i.e. each half-move from either side advances the schedule).",
    default: 0.2,
    range: 0.01...5.0,
    category: "Self-Play Sampling",
    liveTunable: true,
    absentValue: .refuseExact
)
public enum SelfPlayStartTau: TrainingParameterKey {}

@TrainingParameter(
    name: "Self-Play Target Tau",
    description: "Floor sampling temperature for self-play games — start_tau decays toward this value.",
    default: 0.02,
    range: 0.01...5.0,
    category: "Self-Play Sampling",
    liveTunable: true,
    absentValue: .refuseExact
)
public enum SelfPlayTargetTau: TrainingParameterKey {}

@TrainingParameter(
    name: "Self-Play Tau Decay Per Ply",
    description: "Decay applied to tau on every game-total ply (each half-move from either side), moving start_tau toward target_tau during a self-play game. tau(ply) = max(target_tau, start_tau − decay·ply).",
    default: 0.02,
    range: 0.0...1.0,
    category: "Self-Play Sampling",
    liveTunable: true,
    absentValue: .refuseExact
)
public enum SelfPlayTauDecayPerPly: TrainingParameterKey {}

@TrainingParameter(
    name: "Self-Play Draw Keep Fraction",
    description: "Fraction of self-play games ending in a draw that are pushed into the replay buffer at game end. 1.0 = keep every drawn game (legacy behavior, no filtering); 0.0 = discard every drawn game (only decisive games train). Decisive (checkmate) games are always kept regardless of this knob. The replay-ratio controller targets the EMITTED positions/sec rate, so dropping draws automatically slows training to keep the cons/prod ratio at target.",
    default: 1.0,
    range: 0.0...1.0,
    category: "Self-Play Sampling",
    liveTunable: true,
    absentValue: .refuseExact
)
public enum SelfPlayDrawKeepFraction: TrainingParameterKey {}

@TrainingParameter(
    name: "Self-Play Max Plies Per Game",
    description: "Self-play games are auto-terminated when they reach this many plies. Acts as a safety net against games that fail to terminate via the 50-move rule or 3-fold repetition. Terminated games are NOT emitted — they're counted as 'dropped' in the Played stats and never reach the replay buffer. Applies to self-play only — arena games are not affected.",
    default: 450,
    range: 25...500,
    category: "Self-Play Sampling",
    liveTunable: true,
    absentValue: .refuseExact
)
public enum SelfPlayMaxPliesPerGame: TrainingParameterKey {}

@TrainingParameter(
    name: "Draw-Watch pDraw Threshold",
    description: "Per-ply pDraw value (W/D/L softmax draw slot) a self-play position must clear to count toward the draw-watch streak. When N consecutive plies in the same game clear this threshold (N from 'Draw-Watch Streak Length', default 8), the game is flagged on the Draw-watch chart tile. With the 'Terminate flagged games' toggle off (default) flagging is purely observational; with it on, the game is dropped immediately on flag fire. Lowering this catches more games (and earlier); raising it tightens the precision-toward-draw calibration.",
    default: 0.985,
    range: 0.5...1.0,
    category: "Self-Play Sampling",
    liveTunable: true,
    absentValue: .refuseExact
)
public enum DrawWatchPDrawThreshold: TrainingParameterKey {}

@TrainingParameter(
    name: "Terminate Draw-Watched Games",
    description: "When ON: self-play games whose N-ply pDraw streak completes are dropped on the spot — same drop path as ply-cap-terminated games (no flush to the replay buffer, counted as 'dropped' in the [STATS] outcomes). Saves the GPU/throughput cost of playing out games the network has already decided are drawn. When OFF (default): the draw-watch is purely observational; games play to natural termination. Toggling this OFF mid-session lets the calibration metric on the Draw-watch chart tile resume showing flag→draw precision (a meaningless metric when termination is engaged because every flagged game is forced to a draw).",
    default: false,
    category: "Self-Play Sampling",
    liveTunable: true,
    absentValue: .refuseExact
)
public enum DrawWatchTerminateGames: TrainingParameterKey {}

@TrainingParameter(
    name: "Draw-Watch Streak Length",
    description: "Number of consecutive plies a self-play position must clear the pDraw threshold to count as a draw-watch flag. Lowering this fires flags earlier (catches more games, with looser confidence in their draw nature); raising it requires the network to be sustained-confident for longer (fewer flags but tighter calibration). Default 8 is a reasonable balance — long enough to filter out one-off pDraw spikes, short enough to catch the network's confident-draw signal before games drag on. Lives alongside the threshold + terminate toggle in Self-Play Sampling; all three are re-read by the driver on every tick.",
    default: 8,
    range: 2...32,
    category: "Self-Play Sampling",
    liveTunable: true,
    absentValue: .refuseExact
)
public enum DrawWatchStreakLength: TrainingParameterKey {}

@TrainingParameter(
    name: "Arena Start Tau",
    description: "Initial sampling temperature for arena games. Decays toward Arena Target Tau by Arena Tau Decay Per Ply each game-total ply.",
    default: 0.2,
    range: 0.01...5.0,
    category: "Arena",
    liveTunable: true,
    absentValue: .currentSetting
)
public enum ArenaStartTau: TrainingParameterKey {}

@TrainingParameter(
    name: "Arena Target Tau",
    description: "Floor sampling temperature for arena games.",
    default: 0.02,
    range: 0.01...5.0,
    category: "Arena",
    liveTunable: true,
    absentValue: .currentSetting
)
public enum ArenaTargetTau: TrainingParameterKey {}

@TrainingParameter(
    name: "Arena Tau Decay Per Ply",
    description: "Decay applied to tau on every game-total ply (each half-move from either side), moving arena start_tau toward target_tau.",
    default: 0.02,
    range: 0.0...1.0,
    category: "Arena",
    liveTunable: true,
    absentValue: .currentSetting
)
public enum ArenaTauDecayPerPly: TrainingParameterKey {}

@TrainingParameter(
    name: "Replay Ratio Target",
    description: "Target ratio of consumed (training) positions to produced (self-play) positions. ReplayRatioController auto-adjusts step delay to track this.",
    default: 0.48,
    range: 0.01...100.0,
    category: "Replay Buffer",
    liveTunable: true,
    absentValue: .currentSetting
)
public enum ReplayRatioTarget: TrainingParameterKey {}

@TrainingParameter(
    name: "Replay Ratio Auto Adjust",
    description: "Whether ReplayRatioController auto-tunes the trainer step delay to track Replay Ratio Target.",
    default: false,
    category: "Replay Buffer",
    liveTunable: true,
    absentValue: .currentSetting
)
public enum ReplayRatioAutoAdjust: TrainingParameterKey {}

@TrainingParameter(
    name: "Record Self-Play Games",
    description: "Record completed (post-draw-filter) self-play games to a reusable game corpus under Corpora/. Read once at run start (not live-tunable).",
    default: false,
    category: "Self-Play Sampling",
    liveTunable: false,
    absentValue: .currentSetting
)
public enum RecordSelfPlayGames: TrainingParameterKey {}

@TrainingParameter(
    name: "Self-Play Concurrency",
    description: "Parallel self-play game count. More = faster replay-buffer fill but more GPU contention.",
    default: 180,
    range: 1...8192,
    category: "Training Window",
    liveTunable: true,
    absentValue: .currentSetting
)
public enum SelfPlayConcurrency: TrainingParameterKey {}

@TrainingParameter(
    name: "Training Step Delay (ms)",
    description: "Delay between trainer SGD steps in milliseconds. Auto-adjusted by ReplayRatioController when auto-adjust is on.",
    default: 0,
    range: 0...3000,
    category: "Training Window",
    liveTunable: true,
    absentValue: .currentSetting
)
public enum TrainingStepDelayMs: TrainingParameterKey {}

@TrainingParameter(
    name: "Self-Play Delay (ms)",
    description: "Per-game-per-worker delay between self-play games in milliseconds. Used only when replay-ratio auto-adjust is OFF; auto-adjust on lets the controller manage it.",
    default: 0,
    range: 0...3000,
    category: "Training Window",
    liveTunable: true,
    absentValue: .currentSetting
)
public enum SelfPlayDelayMs: TrainingParameterKey {}

@TrainingParameter(
    name: "Training Batch Size",
    description: "SGD minibatch size. Couples with learning_rate (scaled via sqrt_batch_scaling_lr) and weight_decay.",
    default: 4096,
    range: 32...65536,
    category: "Training Window",
    liveTunable: false,
    absentValue: .refuseExact
)
public enum TrainingBatchSize: TrainingParameterKey {}

@TrainingParameter(
    name: "Replay Buffer Capacity",
    description: "Maximum number of positions retained in the FIFO replay buffer.",
    default: 1000000,
    range: 1000...10000000,
    category: "Replay Buffer",
    liveTunable: false,
    absentValue: .refuseExact
)
public enum ReplayBufferCapacity: TrainingParameterKey {}

@TrainingParameter(
    name: "Replay Buffer Min Positions Before Training",
    description: "Number of self-play positions accumulated before the trainer starts pulling minibatches.",
    default: 500000,
    range: 0...10000000,
    category: "Replay Buffer",
    liveTunable: false,
    absentValue: .currentSetting
)
public enum ReplayBufferMinPositionsBeforeTraining: TrainingParameterKey {}

@TrainingParameter(
    name: "Max Plies From Any 1 Game",
    description: "Cap on how many plies may be drawn from any single game (self-play, corpus or engine game) within one training batch. Decorrelates the minibatch by forcing it to span many distinct games rather than letting one long marathon dominate. At the default (10), the cap is essentially always active for long games at typical batch sizes (e.g. 4096); near the range max (400), the cap rarely binds.",
    default: 10,
    range: 1...400,
    category: "Replay Buffer",
    liveTunable: true,
    absentValue: .declaredRangeMaximum
)
public enum MaxPliesFromAnyOneGame: TrainingParameterKey {}

@TrainingParameter(
    name: "Target Sampled Game Length (plies)",
    description: "When > 0, the batch sampler exponentially down-weights positions from long games so the position-weighted mean game length of the sampled batch approaches this value (in plies). 0 disables the length tilt, and any value at or above the buffer's natural mean game length leaves the batch effectively untilted. To de-weight shuffle-draw marathons, set it below the buffer's natural mean.",
    default: 999,
    range: 0...10000,
    category: "Replay Buffer",
    liveTunable: true,
    absentValue: .refuseExact
)
public enum TargetSampledGameLengthPlies: TrainingParameterKey {}

@TrainingParameter(
    name: "Max Draws Per Batch (%)",
    description: "Ceiling on the percentage of positions in a training batch that come from drawn games. If the buffer holds fewer drawn positions than the cap allows, the batch simply contains fewer (no padding); freed slots go to positions from decisive games. 100 disables the cap.",
    default: 100,
    range: 0...100,
    category: "Replay Buffer",
    liveTunable: true,
    absentValue: .preFeature(100)
)
public enum MaxDrawPercentPerBatch: TrainingParameterKey {}

@TrainingParameter(
    name: "Stratify Training Batches By Game Phase",
    description: "Stratify training minibatches by game phase. When ON, each batch is drawn with roughly equal weight from four game-phase buckets defined by NON-PAWN piece count: 0–4 (deep endgame), 5–8 (late endgame), 9–14 (middlegame), 15–22 (full piece set). This compensates for the replay buffer's natural skew toward late-endgame positions, where the trainer otherwise sees ~2× as many endgame as middlegame samples. The per-batch draw-percent cap and per-game K cap do NOT apply while this is on (V1 limitation) — the UI grays those controls out with an inline banner while stratification is on. Bucket distribution converges to balanced as the buffer fills; the popover's per-batch mini-chart shows the realized mix vs the buffer's natural mix.",
    default: false,
    category: "Replay Buffer",
    liveTunable: true,
    absentValue: .preFeature(false)
)
public enum ReplayBufferStratifyByMaterial: TrainingParameterKey {}

@TrainingParameter(
    name: "Arena Promote Threshold",
    description: "Minimum candidate score (in [0, 1]) required to promote the candidate over the champion.",
    default: 0.53,
    range: 0.5...1.0,
    category: "Arena",
    liveTunable: false,
    absentValue: .currentSetting
)
public enum ArenaPromoteThreshold: TrainingParameterKey {}

@TrainingParameter(
    name: "Arena Games Per Tournament",
    description: "Number of candidate-vs-champion games per arena run. Higher = tighter Wilson confidence interval.",
    default: 400,
    range: 4...10000,
    category: "Arena",
    liveTunable: false,
    absentValue: .currentSetting
)
public enum ArenaGamesPerTournament: TrainingParameterKey {}

@TrainingParameter(
    name: "Arena Auto Interval (sec)",
    description: "Automatic arena interval in seconds; the play-and-train loop schedules a new arena every N seconds.",
    default: 900.0,
    range: 60.0...86400.0,
    category: "Arena",
    liveTunable: true,
    absentValue: .currentSetting
)
public enum ArenaAutoIntervalSec: TrainingParameterKey {}

@TrainingParameter(
    name: "Candidate Probe Interval (sec)",
    description: "Interval between candidate forward-pass probes for collapse-detection telemetry.",
    default: 15.0,
    range: 1.0...3600.0,
    category: "Collapse Detection",
    liveTunable: false,
    absentValue: .currentSetting
)
public enum CandidateProbeIntervalSec: TrainingParameterKey {}

@TrainingParameter(
    name: "Legal-Mass Collapse Threshold",
    description: "If illegal_mass_sum stays at or above this for the no-improvement window, the run early-bails.",
    default: 0.99,
    range: 0.5...1.0,
    category: "Collapse Detection",
    liveTunable: false,
    absentValue: .currentSetting
)
public enum LegalMassCollapseThreshold: TrainingParameterKey {}

@TrainingParameter(
    name: "Legal-Mass Collapse Grace (sec)",
    description: "Post-training-start grace window during which legal-mass collapse early-bail is suppressed.",
    default: 600.0,
    range: 0.0...86400.0,
    category: "Collapse Detection",
    liveTunable: false,
    absentValue: .currentSetting
)
public enum LegalMassCollapseGraceSeconds: TrainingParameterKey {}

@TrainingParameter(
    name: "Legal-Mass Collapse No-Improvement Probes",
    description: "Number of consecutive collapsed probes (after grace) before the run early-bails.",
    default: 8,
    range: 1...1000,
    category: "Collapse Detection",
    liveTunable: false,
    absentValue: .currentSetting
)
public enum LegalMassCollapseNoImprovementProbes: TrainingParameterKey {}

@TrainingParameter(
    name: "Arena Concurrency",
    description: "Number of concurrent arena games. Higher = faster arena throughput at cost of GPU contention.",
    default: 400,
    range: 1...1024,
    category: "Arena",
    liveTunable: true,
    absentValue: .currentSetting
)
public enum ArenaConcurrency: TrainingParameterKey {}

// MARK: Arena promotion criterion (score threshold vs SPRT)
//
// The SPRT knobs below are inert while the criterion is Score Threshold, and
// `Arena Games Per Tournament` is inert while it is SPRT: a sequential test
// decides its own sample size, and capping it at a fixed count destroys the
// calibration the test exists to provide (measured: ~0.6% accept where β
// promises 95%). See `documentation/arena-sprt.md`.
//
// All are `liveTunable: false`. The likelihood ratio is only meaningful
// against hypotheses fixed for the whole test, so these are snapshotted at
// arena start and a mid-arena edit takes effect on the next arena.

@TrainingParameter(
    name: "Arena Promotion Criterion",
    description: "Which rule decides promotion: 0 = score threshold (candidate score over a fixed game count must clear Arena Promote Threshold), 1 = SPRT (sequential test of elo0 vs elo1 at error rates alpha/beta, running until the evidence crosses a bound). SPRT ignores Arena Games Per Tournament and Arena Promote Threshold.",
    default: 1,
    range: 0...1,
    category: "Arena",
    id: "arena_promotion_criterion",
    liveTunable: false,
    absentValue: .preFeature(0)
)
public enum ArenaPromotionCriterionParameter: TrainingParameterKey {}

@TrainingParameter(
    name: "Arena SPRT elo0 (H0)",
    description: "SPRT null hypothesis, in Elo: the candidate is exactly this many Elo stronger than the champion. Usually 0 -- 'no improvement'. Must be below elo1.",
    default: 0.0,
    range: -50.0...50.0,
    category: "Arena",
    liveTunable: false,
    absentValue: .currentSetting
)
public enum ArenaSPRTElo0: TrainingParameterKey {}

@TrainingParameter(
    name: "Arena SPRT elo1 (H1)",
    description: "SPRT alternative hypothesis, in Elo: the smallest improvement the test is asked to detect. Smaller values need far more games (simulated at an 0.85 draw rate: ~720 median games at 10 Elo, ~190 at 20, ~60 at 35).",
    default: 10.0,
    range: -50.0...50.0,
    category: "Arena",
    liveTunable: false,
    absentValue: .currentSetting
)
public enum ArenaSPRTElo1: TrainingParameterKey {}

@TrainingParameter(
    name: "Arena SPRT alpha",
    description: "SPRT type I error rate: long-run probability of promoting a candidate that is only elo0 strong. Lower = stricter promotion, more games per decision.",
    default: 0.05,
    range: 0.001...0.5,
    category: "Arena",
    liveTunable: false,
    absentValue: .currentSetting
)
public enum ArenaSPRTAlpha: TrainingParameterKey {}

@TrainingParameter(
    name: "Arena SPRT beta",
    description: "SPRT type II error rate: long-run probability of rejecting a candidate that genuinely is elo1 strong. Lower = fewer missed improvements, more games per decision. alpha + beta must be below 1.",
    default: 0.05,
    range: 0.001...0.5,
    category: "Arena",
    liveTunable: false,
    absentValue: .currentSetting
)
public enum ArenaSPRTBeta: TrainingParameterKey {}

@TrainingParameter(
    name: "Arena SPRT Min Games",
    description: "Games that must complete before the SPRT may fire at all. Guards against an early streak crossing a boundary on almost no evidence.",
    default: 32,
    range: 2...10000,
    category: "Arena",
    liveTunable: false,
    absentValue: .currentSetting
)
public enum ArenaSPRTMinGames: TrainingParameterKey {}

@TrainingParameter(
    name: "Arena SPRT Max Games",
    description: "Runaway guard for the SPRT; 0 means unbounded. Reaching it with the evidence still between the bounds is INCONCLUSIVE and never promotes. Set it far above the expected decision point -- hitting it routinely means elo0 and elo1 are too close together, not that the candidate is bad.",
    default: 20000,
    range: 0...1000000,
    category: "Arena",
    liveTunable: false,
    absentValue: .currentSetting
)
public enum ArenaSPRTMaxGames: TrainingParameterKey {}


@TrainingParameter(
    name: "Batch Stats Interval",
    description: "Compute and emit [BATCH-STATS] every N training batches. 0 disables. Cost is ~1ms per evaluated batch; default 10 keeps log volume manageable.",
    default: 10,
    range: 0...10000,
    category: "Observability",
    liveTunable: true,
    absentValue: .currentSetting
)
public enum BatchStatsInterval: TrainingParameterKey {}

@TrainingParameter(
    name: "KL Probe Interval",
    description: "Measure KL(policy before the SGD step || policy after) on the training minibatch every N steps, and chart it with its across-batch spread. 0 disables. This is the only metric that shows how far a step moves the policy in FUNCTION space -- gNorm measures the step in parameter space, and the two diverge: a large gradient across a flat region barely moves the distribution, a small one across a sharp region can move it a lot. Costs one extra forward pass on probe steps only (roughly 8-11% of a training step at batch 4096, so ~1% at interval 10). The probe holds the dropout RNG steady across both of its forward passes, so the measurement isolates the weight update at any dropout rate.",
    default: 100,
    range: 0...10000,
    category: "Observability",
    liveTunable: true,
    absentValue: .currentSetting
)
public enum KLProbeInterval: TrainingParameterKey {}

// MARK: - LR / Momentum cycling (TRAINING_DYNAMICS_PLAN.md §3)
//
// Two independent repeating cycles — one for the learning rate (geometric
// interpolation between absolute endpoints), one for Polyak momentum (linear).
// The phase is a pure function of the trainer's global step, offset so the
// cycle begins when LR warmup ends, so resume is seamless. See `LRMomentumCycle.swift` for the math. Inverse coupling (high
// LR ↔ low momentum) is recovered either by enabling the momentum cycle with
// `momentum_cycle_invert = true` at an equal period, or — without having to
// keep the two in agreement by hand — by `momentum_follows_lr_cycle`, which
// drives momentum from the LR cycle's own phase. The LR cycle's peak and
// trough (and the follow-mode momentum bounds) additionally decay over
// `lr_cycle_decay_horizon_steps`, so a long open-ended run anneals.

@TrainingParameter(
    name: "LR Cycle Enabled",
    description: "Enable the repeating learning-rate cycle. When on, the base LR each step is set by the cycle (geometric interpolation between LR Cycle Min and Max over LR Cycle Period Steps) instead of the static Learning Rate, then composed with the √batch multiplier. The cycle begins when LR warmup ends; during warmup the LR ramps linearly up to the cycle's starting value. Overrides the static base-LR schedule while enabled.",
    default: true,
    category: "LR/Momentum Cycling",
    id: "lr_cycle_enabled",
    liveTunable: true,
    absentValue: .preFeature(false)
)
public enum LRCycleEnabled: TrainingParameterKey {}

@TrainingParameter(
    name: "LR Cycle Period (steps)",
    description: "Full up-then-down period of the LR cycle, in optimizer steps. A sensible default is 2–8× the replay-buffer turnover (bufferCapacity / batchSize), the self-play analog of an epoch.",
    default: 20000,
    range: 1...10000000,
    category: "LR/Momentum Cycling",
    id: "lr_cycle_period_steps",
    liveTunable: true,
    absentValue: .currentSetting
)
public enum LRCyclePeriodSteps: TrainingParameterKey {}

@TrainingParameter(
    name: "LR Cycle Count",
    description: "Number of LR cycles (counted from the end of warmup) to run before freezing at the cycle boundary (LR Cycle Min, or Max when inverted). 0 = unbounded (repeat forever), the default for open-ended self-play.",
    default: 0,
    range: 0...1000000,
    category: "LR/Momentum Cycling",
    id: "lr_cycle_count",
    liveTunable: true,
    absentValue: .currentSetting
)
public enum LRCycleCount: TrainingParameterKey {}

@TrainingParameter(
    name: "LR Cycle Min",
    description: "Absolute learning rate at the LR cycle's trough (the period boundaries when not inverted) at the start of the decay horizon — the trough's start value. The trough decays geometrically from here to LR Cycle Trough End over LR Cycle Decay Horizon. Must be > 0 — geometric interpolation is undefined at zero, and LR Cycle Max must be ≥ this value or the cycle is ignored.",
    default: 0.001,
    range: 1.0e-7...1.0,
    category: "LR/Momentum Cycling",
    id: "lr_cycle_min",
    liveTunable: true,
    absentValue: .currentSetting
)
public enum LRCycleMin: TrainingParameterKey {}

@TrainingParameter(
    name: "LR Cycle Max",
    description: "Absolute learning rate at the LR cycle's peak (the period midpoint when not inverted) at the start of the decay horizon — the peak's start value. The peak decays geometrically from here to LR Cycle Peak End over LR Cycle Decay Horizon. Must be ≥ LR Cycle Min.",
    default: 0.1,
    range: 1.0e-7...1.0,
    category: "LR/Momentum Cycling",
    id: "lr_cycle_max",
    liveTunable: true,
    absentValue: .currentSetting
)
public enum LRCycleMax: TrainingParameterKey {}

@TrainingParameter(
    name: "LR Cycle Invert",
    description: "Flip the LR waveform so the cycle starts at its peak and dips to its trough at the midpoint. Default ON, so each cycle opens with a high-LR burst and anneals down into the trough before the next one.",
    default: true,
    category: "LR/Momentum Cycling",
    id: "lr_cycle_invert",
    liveTunable: true,
    absentValue: .currentSetting
)
public enum LRCycleInvert: TrainingParameterKey {}

@TrainingParameter(
    name: "Momentum Cycle Enabled",
    description: "Enable the repeating Polyak-momentum cycle. When on, the momentum coefficient each step is set by the cycle (linear interpolation between Momentum Cycle Min and Max) instead of the static Momentum Coefficient. Like the LR cycle, it begins when LR warmup ends and holds its starting value during warmup.",
    default: true,
    category: "LR/Momentum Cycling",
    id: "momentum_cycle_enabled",
    liveTunable: true,
    absentValue: .preFeature(false)
)
public enum MomentumCycleEnabled: TrainingParameterKey {}

@TrainingParameter(
    name: "Momentum Cycle Period (steps)",
    description: "Full up-then-down period of the momentum cycle, in optimizer steps. Set equal to the LR Cycle Period (with Momentum Cycle Invert on) for Smith-style inverse coupling — high LR paired with low momentum.",
    default: 1000,
    range: 1...10000000,
    category: "LR/Momentum Cycling",
    id: "momentum_cycle_period_steps",
    liveTunable: true,
    absentValue: .currentSetting
)
public enum MomentumCyclePeriodSteps: TrainingParameterKey {}

@TrainingParameter(
    name: "Momentum Cycle Count",
    description: "Number of momentum cycles (counted from the end of warmup) before freezing at the cycle boundary. 0 = unbounded (the default).",
    default: 0,
    range: 0...1000000,
    category: "LR/Momentum Cycling",
    id: "momentum_cycle_count",
    liveTunable: true,
    absentValue: .currentSetting
)
public enum MomentumCycleCount: TrainingParameterKey {}

@TrainingParameter(
    name: "Momentum Cycle Min",
    description: "Polyak momentum at the cycle's low point (the period midpoint when inverted, where LR peaks). Smith's recommendation is ~0.85.",
    default: 0.75,
    range: 0.0...0.99,
    category: "LR/Momentum Cycling",
    id: "momentum_cycle_min",
    liveTunable: true,
    absentValue: .currentSetting
)
public enum MomentumCycleMin: TrainingParameterKey {}

@TrainingParameter(
    name: "Momentum Cycle Max",
    description: "Polyak momentum at the cycle's high point (the period boundaries when inverted, where LR bottoms). Smith's recommendation is ~0.95.",
    default: 0.9,
    range: 0.0...0.99,
    category: "LR/Momentum Cycling",
    id: "momentum_cycle_max",
    liveTunable: true,
    absentValue: .currentSetting
)
public enum MomentumCycleMax: TrainingParameterKey {}

@TrainingParameter(
    name: "Momentum Cycle Invert",
    description: "Flip the momentum waveform so it starts at Momentum Cycle Max and dips to Min at the midpoint. Default OFF. Turned on at a period equal to the LR Cycle Period, this makes momentum the inverse of LR (high LR ↔ low momentum), Smith's super-convergence coupling.",
    default: false,
    category: "LR/Momentum Cycling",
    id: "momentum_cycle_invert",
    liveTunable: true,
    absentValue: .currentSetting
)
public enum MomentumCycleInvert: TrainingParameterKey {}

@TrainingParameter(
    name: "LR Cycle Peak End",
    description: "LR cycle peak at the end of the decay horizon, and held there afterwards. The peak decays geometrically from LR Cycle Max to this value across LR Cycle Decay Horizon. Must be > 0 and ≥ LR Cycle Trough End. Ignored when the horizon is 0.",
    default: 1.0e-4,
    range: 1.0e-7...1.0,
    category: "LR/Momentum Cycling",
    id: "lr_cycle_peak_end",
    liveTunable: true,
    absentValue: .currentSetting
)
public enum LRCyclePeakEnd: TrainingParameterKey {}

@TrainingParameter(
    name: "LR Cycle Trough End",
    description: "LR cycle trough at the end of the decay horizon, and held there afterwards. The trough decays geometrically from LR Cycle Min to this value across LR Cycle Decay Horizon. Must be > 0 and ≤ LR Cycle Peak End. Ignored when the horizon is 0.",
    default: 1.0e-6,
    range: 1.0e-7...1.0,
    category: "LR/Momentum Cycling",
    id: "lr_cycle_trough_end",
    liveTunable: true,
    absentValue: .currentSetting
)
public enum LRCycleTroughEnd: TrainingParameterKey {}

@TrainingParameter(
    name: "LR Cycle Decay Horizon (steps)",
    description: "Number of cycle steps (counted from the end of LR warmup) over which the LR cycle's peak and trough decay from their start values (LR Cycle Max / Min) to their end values, and over which follow-mode momentum bounds drift from start to end. After the horizon the envelope holds at the end values and cycling continues. 0 = no decay (the envelope stays at its start values).",
    default: 1000000,
    range: 0...1000000000,
    category: "LR/Momentum Cycling",
    id: "lr_cycle_decay_horizon_steps",
    liveTunable: true,
    absentValue: .preFeature(0)
)
public enum LRCycleDecayHorizonSteps: TrainingParameterKey {}

@TrainingParameter(
    name: "Momentum Follows LR Cycle",
    description: "When on (and momentum cycling is enabled), momentum uses the LR cycle's period and phase, inverted: lowest at the LR peak, highest at the LR trough. Its low and high bounds move linearly from the Follow Start values to the Follow End values over LR Cycle Decay Horizon. The separate momentum cycle's period, count, min, max and invert settings are then ignored. Requires the LR cycle to be enabled; otherwise the static momentum coefficient applies.",
    default: true,
    category: "LR/Momentum Cycling",
    id: "momentum_follows_lr_cycle",
    liveTunable: true,
    absentValue: .preFeature(false)
)
public enum MomentumFollowsLRCycle: TrainingParameterKey {}

@TrainingParameter(
    name: "Momentum Follow Start Low",
    description: "Follow-mode momentum at the LR peak, at the start of the decay horizon. Must be ≤ Momentum Follow Start High.",
    default: 0.85,
    range: 0.0...0.99,
    category: "LR/Momentum Cycling",
    id: "momentum_follow_start_low",
    liveTunable: true,
    absentValue: .currentSetting
)
public enum MomentumFollowStartLow: TrainingParameterKey {}

@TrainingParameter(
    name: "Momentum Follow Start High",
    description: "Follow-mode momentum at the LR trough, at the start of the decay horizon.",
    default: 0.95,
    range: 0.0...0.99,
    category: "LR/Momentum Cycling",
    id: "momentum_follow_start_high",
    liveTunable: true,
    absentValue: .currentSetting
)
public enum MomentumFollowStartHigh: TrainingParameterKey {}

@TrainingParameter(
    name: "Momentum Follow End Low",
    description: "Follow-mode momentum at the LR peak, at and after the end of the decay horizon. Must be ≤ Momentum Follow End High.",
    default: 0.90,
    range: 0.0...0.99,
    category: "LR/Momentum Cycling",
    id: "momentum_follow_end_low",
    liveTunable: true,
    absentValue: .currentSetting
)
public enum MomentumFollowEndLow: TrainingParameterKey {}

@TrainingParameter(
    name: "Momentum Follow End High",
    description: "Follow-mode momentum at the LR trough, at and after the end of the decay horizon.",
    default: 0.95,
    range: 0.0...0.99,
    category: "LR/Momentum Cycling",
    id: "momentum_follow_end_high",
    liveTunable: true,
    absentValue: .currentSetting
)
public enum MomentumFollowEndHigh: TrainingParameterKey {}

// MARK: - Sessions / autosave policy

@TrainingParameter(
    name: "Periodic Autosave Interval (sec)",
    description: "Cadence of the periodic full-session autosave while Play-and-Train is active, in seconds. The default 21600 = 6 hours. Read live: the heartbeat reconciles a mid-session change against the running PeriodicSaveController and re-anchors the next-save deadline, so a shorter interval takes effect without restarting Play-and-Train. Does not affect manual saves or post-promotion autosaves.",
    default: 21600.0,
    range: 60.0...604800.0,
    category: "Sessions",
    liveTunable: true,
    absentValue: .currentSetting
)
public enum PeriodicAutosaveIntervalSec: TrainingParameterKey {}

@TrainingParameter(
    name: "Max Periodic Autosaves Kept",
    description: "Retention cap on the number of periodic and post-promotion autosaves (`-periodic.dcmsession` and `-promote.dcmsession`, which includes Promote Trainee Now saves) kept on disk, counted as one pool across every session in the Sessions folder. After each successful periodic or post-promotion save, the pool is ranked newest first by the timestamp in the folder name and every autosave beyond this count is deleted. The save just written and the current resume target are never deleted, nor is any folder whose name and session.json do not agree on a minted session ID. Default 3. 0 = unlimited (no pruning). Manual (`-manual`) and signal (`-sigusr2`) saves are never pruned by this knob. Applies only when Automatic Save Pruning Enabled is on (and the build does not force pruning off); otherwise nothing is pruned whatever this cap says.",
    default: 3,
    range: 0...10000,
    category: "Sessions",
    liveTunable: true,
    absentValue: .currentSetting
)
public enum MaxPeriodicAutosavesKept: TrainingParameterKey {}

@TrainingParameter(
    name: "Automatic Save Pruning Enabled",
    description: "Turns on the automatic-save retention pool: after each successful periodic or post-promotion save, delete the oldest `-periodic.dcmsession` and `-promote.dcmsession` folders, across every session, beyond Max Periodic Autosaves Kept (with that knob's protections — the save just written, the resume target, and unverified folders are never deleted). Off by default: no automatic save is ever deleted. NOTE: the current build forces pruning off regardless of this setting (a code-level kill switch, `CheckpointPaths.automaticSavePruningForcedOff`), so turning this on has no effect until a build lifts that switch; every periodic or post-promotion save logs a `[PRUNE] skipped` line naming the reason.",
    default: false,
    category: "Sessions",
    id: "automatic_save_pruning_enabled",
    liveTunable: true,
    absentValue: .currentSetting
)
public enum AutomaticSavePruningEnabled: TrainingParameterKey {}

@TrainingParameter(
    name: "Session Save Include Replay Buffer",
    description: "Whether automatic session saves write the replay buffer (`replay_buffer.bin`, several GB at the usual capacities) into the session folder: the GUI's periodic, post-promotion and SIGUSR2 saves. A manual File > Save Session asks each time, starting from this setting. Off by default: a save holds the weights, optimizer state and run state, and a resume from it refills the buffer from new games before training continues, reported as `[RESUME] NOT EXACT: buffer`. On: the buffer is saved and a resume restores it. Read at each save.",
    default: false,
    category: "Sessions",
    id: "session_save_include_replay_buffer",
    liveTunable: true,
    absentValue: .currentSetting
)
public enum SessionSaveIncludeReplayBuffer: TrainingParameterKey {}

// MARK: Reproducibility (determinism plan, Part A3.2)
//
// One master seed per run; every random stream (replay-buffer draws, each
// self-play / arena / train-vs-UCI game, probe subsets) derives from it by
// name. Not live-tunable: the seed is fixed for the life of a run. See
// `RunRandomSeed` (the one resolver) and `RandomSeedMode`.

@TrainingParameter(
    name: "Random Seed Mode",
    description: "Where a run's master seed comes from: 0 = unseeded (a seed is drawn at run start), 1 = seeded (Random Seed is used). Either way the run uses the seeded random streams and logs its seed on the [RUN] line, so an unseeded run can be replayed by giving its logged seed back (seeded mode, or --seed on the command line, which overrides both settings).",
    default: 0,
    range: 0...1,
    category: "Reproducibility",
    id: "random_seed_mode",
    liveTunable: false,
    absentValue: .preFeature(0)
)
public enum RandomSeedModeParameter: TrainingParameterKey {}

@TrainingParameter(
    name: "Random Seed",
    description: "The master seed used when Random Seed Mode = 1 (seeded); ignored, and logged as ignored, when unseeded. A whole number from 0 to 18446744073709551615, written in parameters.json as a decimal string (JSON numbers are doubles in many readers, which would drop the low bits of a large seed). Every random stream of the run — replay-buffer draws, each game's moves, probe subsets — derives from it by name, so the same seed on the same build, device and data reproduces the same draws.",
    default: UInt64(0),
    range: 0...UInt64.max,
    category: "Reproducibility",
    id: "random_seed",
    liveTunable: false,
    absentValue: .refuseExact
)
public enum RandomSeed: TrainingParameterKey {}

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

    /// Every parameter at its declared default with `overrides` applied —
    /// a snapshot that does not depend on anyone's saved settings, for
    /// computations that must be reproducible (the resume-equivalence
    /// harness). Each override is validated against its declaration; an
    /// unknown id or out-of-range value throws.
    public nonisolated static func declaredDefaults(
        overriding overrides: [String: ParameterValue]
    ) throws -> TrainingParametersSnapshot {
        try TrainingParameters.validate(overrides)
        var values = Dictionary(uniqueKeysWithValues: TrainingParameters.allDefinitions.map { ($0.id, $0.defaultValue) })
        for (id, value) in overrides { values[id] = value }
        return TrainingParametersSnapshot(values: values)
    }
}

// Typed accessors on the snapshot — keep parallel with the stored properties on TrainingParameters.
public extension TrainingParametersSnapshot {
    var entropyBonus: Double { value(for: EntropyBonus.self) }
    var illegalMassWeight: Double { value(for: IllegalMassWeight.self) }
    var policyLabelSmoothingEpsilon: Double { value(for: PolicyLabelSmoothingEpsilon.self) }
    var policyLabelSmoothingMode: PolicyLabelSmoothingMode {
        PolicyLabelSmoothingMode(persistedRawValue: value(for: PolicyLabelSmoothingModeParameter.self))
    }
    var policyLabelSmoothingPerMove: Double { value(for: PolicyLabelSmoothingPerMove.self) }
    var policyLabelSmoothingPerMoveCap: Double { value(for: PolicyLabelSmoothingPerMoveCap.self) }
    var valueLabelSmoothingEpsilon: Double { value(for: ValueLabelSmoothingEpsilon.self) }
    var gradClipMaxNorm: Double { value(for: GradClipMaxNorm.self) }
    var weightDecay: Double { value(for: WeightDecay.self) }
    var dropoutRate: Double { value(for: DropoutRate.self) }
    var policyLossWeight: Double { value(for: PolicyLossWeight.self) }
    var valueLossWeight: Double { value(for: ValueLossWeight.self) }
    var learningRate: Double { value(for: LearningRate.self) }
    var momentumCoeff: Double { value(for: MomentumCoeff.self) }
    var sqrtBatchScalingLR: Bool { value(for: SqrtBatchScalingLR.self) }
    var signedAdvantageComplementCE: Bool { value(for: SignedAdvantageComplementCE.self) }
    var lrWarmupSteps: Int { value(for: LRWarmupSteps.self) }
    var drawPenalty: Double { value(for: DrawPenalty.self) }
    var selfPlayStartTau: Double { value(for: SelfPlayStartTau.self) }
    var selfPlayTargetTau: Double { value(for: SelfPlayTargetTau.self) }
    var selfPlayTauDecayPerPly: Double { value(for: SelfPlayTauDecayPerPly.self) }
    var selfPlayDrawKeepFraction: Double { value(for: SelfPlayDrawKeepFraction.self) }
    var selfPlayMaxPliesPerGame: Int { value(for: SelfPlayMaxPliesPerGame.self) }
    var recordSelfPlayGames: Bool { value(for: RecordSelfPlayGames.self) }
    var drawWatchPDrawThreshold: Double { value(for: DrawWatchPDrawThreshold.self) }
    var drawWatchTerminateGames: Bool { value(for: DrawWatchTerminateGames.self) }
    var drawWatchStreakLength: Int { value(for: DrawWatchStreakLength.self) }
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
    var replayBufferStratifyByMaterial: Bool { value(for: ReplayBufferStratifyByMaterial.self) }
    var arenaPromoteThreshold: Double { value(for: ArenaPromoteThreshold.self) }
    var arenaGamesPerTournament: Int { value(for: ArenaGamesPerTournament.self) }
    var arenaAutoIntervalSec: Double { value(for: ArenaAutoIntervalSec.self) }
    var candidateProbeIntervalSec: Double { value(for: CandidateProbeIntervalSec.self) }
    var legalMassCollapseThreshold: Double { value(for: LegalMassCollapseThreshold.self) }
    var legalMassCollapseGraceSeconds: Double { value(for: LegalMassCollapseGraceSeconds.self) }
    var legalMassCollapseNoImprovementProbes: Int { value(for: LegalMassCollapseNoImprovementProbes.self) }
    var arenaConcurrency: Int { value(for: ArenaConcurrency.self) }
    var arenaPromotionCriterion: ArenaPromotionCriterion {
        ArenaPromotionCriterion(persistedRawValue: value(for: ArenaPromotionCriterionParameter.self))
    }
    var arenaSPRTElo0: Double { value(for: ArenaSPRTElo0.self) }
    var arenaSPRTElo1: Double { value(for: ArenaSPRTElo1.self) }
    var arenaSPRTAlpha: Double { value(for: ArenaSPRTAlpha.self) }
    var arenaSPRTBeta: Double { value(for: ArenaSPRTBeta.self) }
    var arenaSPRTMinGames: Int { value(for: ArenaSPRTMinGames.self) }
    var arenaSPRTMaxGames: Int { value(for: ArenaSPRTMaxGames.self) }
    var batchStatsInterval: Int { value(for: BatchStatsInterval.self) }
    var klProbeInterval: Int { value(for: KLProbeInterval.self) }
    var lrCycleEnabled: Bool { value(for: LRCycleEnabled.self) }
    var lrCyclePeriodSteps: Int { value(for: LRCyclePeriodSteps.self) }
    var lrCycleCount: Int { value(for: LRCycleCount.self) }
    var lrCycleMin: Double { value(for: LRCycleMin.self) }
    var lrCycleMax: Double { value(for: LRCycleMax.self) }
    var lrCycleInvert: Bool { value(for: LRCycleInvert.self) }
    var momentumCycleEnabled: Bool { value(for: MomentumCycleEnabled.self) }
    var momentumCyclePeriodSteps: Int { value(for: MomentumCyclePeriodSteps.self) }
    var momentumCycleCount: Int { value(for: MomentumCycleCount.self) }
    var momentumCycleMin: Double { value(for: MomentumCycleMin.self) }
    var momentumCycleMax: Double { value(for: MomentumCycleMax.self) }
    var momentumCycleInvert: Bool { value(for: MomentumCycleInvert.self) }
    var lrCyclePeakEnd: Double { value(for: LRCyclePeakEnd.self) }
    var lrCycleTroughEnd: Double { value(for: LRCycleTroughEnd.self) }
    var lrCycleDecayHorizonSteps: Int { value(for: LRCycleDecayHorizonSteps.self) }
    var momentumFollowsLRCycle: Bool { value(for: MomentumFollowsLRCycle.self) }
    var momentumFollowStartLow: Double { value(for: MomentumFollowStartLow.self) }
    var momentumFollowStartHigh: Double { value(for: MomentumFollowStartHigh.self) }
    var momentumFollowEndLow: Double { value(for: MomentumFollowEndLow.self) }
    var momentumFollowEndHigh: Double { value(for: MomentumFollowEndHigh.self) }
    var periodicAutosaveIntervalSec: Double { value(for: PeriodicAutosaveIntervalSec.self) }
    var maxPeriodicAutosavesKept: Int { value(for: MaxPeriodicAutosavesKept.self) }
    var automaticSavePruningEnabled: Bool { value(for: AutomaticSavePruningEnabled.self) }
    var sessionSaveIncludeReplayBuffer: Bool { value(for: SessionSaveIncludeReplayBuffer.self) }
    var randomSeedMode: RandomSeedMode {
        RandomSeedMode(persistedRawValue: value(for: RandomSeedModeParameter.self))
    }
    var randomSeed: UInt64 { value(for: RandomSeed.self) }

}


// `arenaSPRTConfig()` returns `ArenaSPRT.SPRTConfig`, which is internal, so it
// cannot sit in the public accessor extension above.
extension TrainingParametersSnapshot {

    /// Builds the validated SPRT configuration these parameters describe.
    ///
    /// Per-field ranges are enforced by the parameter definitions, but the
    /// cross-field constraints (`elo1 > elo0`, `alpha + beta < 1`,
    /// `minGames <= maxGames`) are relationships the registry cannot express,
    /// and a `parameters.json` or CLI override can set each field to a legal
    /// value while making the pair meaningless. `ArenaSPRT.SPRTConfig`'s
    /// throwing init is the single place those are checked, so this rethrows
    /// rather than repairing the combination: an arena that cannot form a
    /// valid test must say so, not quietly run a different test.
    func arenaSPRTConfig() throws -> ArenaSPRT.SPRTConfig {
        try ArenaSPRT.SPRTConfig(
            elo0: arenaSPRTElo0,
            elo1: arenaSPRTElo1,
            alpha: arenaSPRTAlpha,
            beta: arenaSPRTBeta,
            minGames: arenaSPRTMinGames,
            maxGames: arenaSPRTMaxGames
        )
    }
}

// MARK: - TrainingParameters singleton

@MainActor
@Observable
public final class TrainingParameters {
    public static let shared = TrainingParameters()

    // Stored properties — one per parameter. didSet persists to UserDefaults.
    // @Observable instruments these for SwiftUI re-renders.
    public var entropyBonus: Double { didSet { if !Self.commitAssignment(EntropyBonus.self, value: entropyBonus) { entropyBonus = oldValue } } }
    public var illegalMassWeight: Double { didSet { if !Self.commitAssignment(IllegalMassWeight.self, value: illegalMassWeight) { illegalMassWeight = oldValue } } }
    public var policyLabelSmoothingEpsilon: Double { didSet { if !Self.commitAssignment(PolicyLabelSmoothingEpsilon.self, value: policyLabelSmoothingEpsilon) { policyLabelSmoothingEpsilon = oldValue } } }
    /// Stored as the enum rather than its raw value so no use site ever sees
    /// the persisted integer; the `didSet` unwraps it at the persistence
    /// boundary, which is the only place the raw form is meaningful.
    public var policyLabelSmoothingMode: PolicyLabelSmoothingMode {
        didSet {
            if !Self.commitAssignment(PolicyLabelSmoothingModeParameter.self, value: policyLabelSmoothingMode.rawValue) {
                policyLabelSmoothingMode = oldValue
            }
        }
    }
    public var policyLabelSmoothingPerMove: Double { didSet { if !Self.commitAssignment(PolicyLabelSmoothingPerMove.self, value: policyLabelSmoothingPerMove) { policyLabelSmoothingPerMove = oldValue } } }
    public var policyLabelSmoothingPerMoveCap: Double { didSet { if !Self.commitAssignment(PolicyLabelSmoothingPerMoveCap.self, value: policyLabelSmoothingPerMoveCap) { policyLabelSmoothingPerMoveCap = oldValue } } }
    public var valueLabelSmoothingEpsilon: Double { didSet { if !Self.commitAssignment(ValueLabelSmoothingEpsilon.self, value: valueLabelSmoothingEpsilon) { valueLabelSmoothingEpsilon = oldValue } } }
    public var gradClipMaxNorm: Double { didSet { if !Self.commitAssignment(GradClipMaxNorm.self, value: gradClipMaxNorm) { gradClipMaxNorm = oldValue } } }
    public var weightDecay: Double { didSet { if !Self.commitAssignment(WeightDecay.self, value: weightDecay) { weightDecay = oldValue } } }
    public var dropoutRate: Double { didSet { if !Self.commitAssignment(DropoutRate.self, value: dropoutRate) { dropoutRate = oldValue } } }
    public var policyLossWeight: Double { didSet { if !Self.commitAssignment(PolicyLossWeight.self, value: policyLossWeight) { policyLossWeight = oldValue } } }
    public var valueLossWeight: Double { didSet { if !Self.commitAssignment(ValueLossWeight.self, value: valueLossWeight) { valueLossWeight = oldValue } } }
    public var learningRate: Double { didSet { if !Self.commitAssignment(LearningRate.self, value: learningRate) { learningRate = oldValue } } }
    public var momentumCoeff: Double { didSet { if !Self.commitAssignment(MomentumCoeff.self, value: momentumCoeff) { momentumCoeff = oldValue } } }
    public var sqrtBatchScalingLR: Bool { didSet { if !Self.commitAssignment(SqrtBatchScalingLR.self, value: sqrtBatchScalingLR) { sqrtBatchScalingLR = oldValue } } }
    public var signedAdvantageComplementCE: Bool { didSet { if !Self.commitAssignment(SignedAdvantageComplementCE.self, value: signedAdvantageComplementCE) { signedAdvantageComplementCE = oldValue } } }
    public var lrWarmupSteps: Int { didSet { if !Self.commitAssignment(LRWarmupSteps.self, value: lrWarmupSteps) { lrWarmupSteps = oldValue } } }
    public var drawPenalty: Double { didSet { if !Self.commitAssignment(DrawPenalty.self, value: drawPenalty) { drawPenalty = oldValue } } }
    public var selfPlayStartTau: Double { didSet { if !Self.commitAssignment(SelfPlayStartTau.self, value: selfPlayStartTau) { selfPlayStartTau = oldValue } } }
    public var selfPlayTargetTau: Double { didSet { if !Self.commitAssignment(SelfPlayTargetTau.self, value: selfPlayTargetTau) { selfPlayTargetTau = oldValue } } }
    public var selfPlayTauDecayPerPly: Double { didSet { if !Self.commitAssignment(SelfPlayTauDecayPerPly.self, value: selfPlayTauDecayPerPly) { selfPlayTauDecayPerPly = oldValue } } }
    public var selfPlayDrawKeepFraction: Double { didSet { if !Self.commitAssignment(SelfPlayDrawKeepFraction.self, value: selfPlayDrawKeepFraction) { selfPlayDrawKeepFraction = oldValue } } }
    public var selfPlayMaxPliesPerGame: Int { didSet { if !Self.commitAssignment(SelfPlayMaxPliesPerGame.self, value: selfPlayMaxPliesPerGame) { selfPlayMaxPliesPerGame = oldValue } } }
    public var recordSelfPlayGames: Bool { didSet { if !Self.commitAssignment(RecordSelfPlayGames.self, value: recordSelfPlayGames) { recordSelfPlayGames = oldValue } } }
    public var drawWatchPDrawThreshold: Double { didSet { if !Self.commitAssignment(DrawWatchPDrawThreshold.self, value: drawWatchPDrawThreshold) { drawWatchPDrawThreshold = oldValue } } }
    public var drawWatchTerminateGames: Bool { didSet { if !Self.commitAssignment(DrawWatchTerminateGames.self, value: drawWatchTerminateGames) { drawWatchTerminateGames = oldValue } } }
    public var drawWatchStreakLength: Int { didSet { if !Self.commitAssignment(DrawWatchStreakLength.self, value: drawWatchStreakLength) { drawWatchStreakLength = oldValue } } }
    public var arenaStartTau: Double { didSet { if !Self.commitAssignment(ArenaStartTau.self, value: arenaStartTau) { arenaStartTau = oldValue } } }
    public var arenaTargetTau: Double { didSet { if !Self.commitAssignment(ArenaTargetTau.self, value: arenaTargetTau) { arenaTargetTau = oldValue } } }
    public var arenaTauDecayPerPly: Double { didSet { if !Self.commitAssignment(ArenaTauDecayPerPly.self, value: arenaTauDecayPerPly) { arenaTauDecayPerPly = oldValue } } }
    public var replayRatioTarget: Double { didSet { if !Self.commitAssignment(ReplayRatioTarget.self, value: replayRatioTarget) { replayRatioTarget = oldValue } } }
    public var replayRatioAutoAdjust: Bool { didSet { if !Self.commitAssignment(ReplayRatioAutoAdjust.self, value: replayRatioAutoAdjust) { replayRatioAutoAdjust = oldValue } } }
    public var selfPlayConcurrency: Int { didSet { if !Self.commitAssignment(SelfPlayConcurrency.self, value: selfPlayConcurrency) { selfPlayConcurrency = oldValue } } }
    public var trainingStepDelayMs: Int { didSet { if !Self.commitAssignment(TrainingStepDelayMs.self, value: trainingStepDelayMs) { trainingStepDelayMs = oldValue } } }
    public var selfPlayDelayMs: Int { didSet { if !Self.commitAssignment(SelfPlayDelayMs.self, value: selfPlayDelayMs) { selfPlayDelayMs = oldValue } } }
    public var trainingBatchSize: Int { didSet { if !Self.commitAssignment(TrainingBatchSize.self, value: trainingBatchSize) { trainingBatchSize = oldValue } } }
    public var replayBufferCapacity: Int { didSet { if !Self.commitAssignment(ReplayBufferCapacity.self, value: replayBufferCapacity) { replayBufferCapacity = oldValue } } }
    public var replayBufferMinPositionsBeforeTraining: Int { didSet { if !Self.commitAssignment(ReplayBufferMinPositionsBeforeTraining.self, value: replayBufferMinPositionsBeforeTraining) { replayBufferMinPositionsBeforeTraining = oldValue } } }
    public var maxPliesFromAnyOneGame: Int { didSet { if !Self.commitAssignment(MaxPliesFromAnyOneGame.self, value: maxPliesFromAnyOneGame) { maxPliesFromAnyOneGame = oldValue } } }
    public var targetSampledGameLengthPlies: Int { didSet { if !Self.commitAssignment(TargetSampledGameLengthPlies.self, value: targetSampledGameLengthPlies) { targetSampledGameLengthPlies = oldValue } } }
    public var maxDrawPercentPerBatch: Int { didSet { if !Self.commitAssignment(MaxDrawPercentPerBatch.self, value: maxDrawPercentPerBatch) { maxDrawPercentPerBatch = oldValue } } }
    public var replayBufferStratifyByMaterial: Bool { didSet { if !Self.commitAssignment(ReplayBufferStratifyByMaterial.self, value: replayBufferStratifyByMaterial) { replayBufferStratifyByMaterial = oldValue } } }
    public var arenaPromoteThreshold: Double { didSet { if !Self.commitAssignment(ArenaPromoteThreshold.self, value: arenaPromoteThreshold) { arenaPromoteThreshold = oldValue } } }
    public var arenaGamesPerTournament: Int { didSet { if !Self.commitAssignment(ArenaGamesPerTournament.self, value: arenaGamesPerTournament) { arenaGamesPerTournament = oldValue } } }
    public var arenaAutoIntervalSec: Double { didSet { if !Self.commitAssignment(ArenaAutoIntervalSec.self, value: arenaAutoIntervalSec) { arenaAutoIntervalSec = oldValue } } }
    public var candidateProbeIntervalSec: Double { didSet { if !Self.commitAssignment(CandidateProbeIntervalSec.self, value: candidateProbeIntervalSec) { candidateProbeIntervalSec = oldValue } } }
    public var legalMassCollapseThreshold: Double { didSet { if !Self.commitAssignment(LegalMassCollapseThreshold.self, value: legalMassCollapseThreshold) { legalMassCollapseThreshold = oldValue } } }
    public var legalMassCollapseGraceSeconds: Double { didSet { if !Self.commitAssignment(LegalMassCollapseGraceSeconds.self, value: legalMassCollapseGraceSeconds) { legalMassCollapseGraceSeconds = oldValue } } }
    public var legalMassCollapseNoImprovementProbes: Int { didSet { if !Self.commitAssignment(LegalMassCollapseNoImprovementProbes.self, value: legalMassCollapseNoImprovementProbes) { legalMassCollapseNoImprovementProbes = oldValue } } }
    public var arenaConcurrency: Int { didSet { if !Self.commitAssignment(ArenaConcurrency.self, value: arenaConcurrency) { arenaConcurrency = oldValue } } }
    /// Stored as the enum rather than its raw value so no use site ever sees
    /// the persisted integer; the `didSet` unwraps it at the persistence
    /// boundary, which is the only place the raw form is meaningful.
    public var arenaPromotionCriterion: ArenaPromotionCriterion {
        didSet {
            if !Self.commitAssignment(ArenaPromotionCriterionParameter.self, value: arenaPromotionCriterion.rawValue) {
                arenaPromotionCriterion = oldValue
            }
        }
    }
    public var arenaSPRTElo0: Double { didSet { if !Self.commitAssignment(ArenaSPRTElo0.self, value: arenaSPRTElo0) { arenaSPRTElo0 = oldValue } } }
    public var arenaSPRTElo1: Double { didSet { if !Self.commitAssignment(ArenaSPRTElo1.self, value: arenaSPRTElo1) { arenaSPRTElo1 = oldValue } } }
    public var arenaSPRTAlpha: Double { didSet { if !Self.commitAssignment(ArenaSPRTAlpha.self, value: arenaSPRTAlpha) { arenaSPRTAlpha = oldValue } } }
    public var arenaSPRTBeta: Double { didSet { if !Self.commitAssignment(ArenaSPRTBeta.self, value: arenaSPRTBeta) { arenaSPRTBeta = oldValue } } }
    public var arenaSPRTMinGames: Int { didSet { if !Self.commitAssignment(ArenaSPRTMinGames.self, value: arenaSPRTMinGames) { arenaSPRTMinGames = oldValue } } }
    public var arenaSPRTMaxGames: Int { didSet { if !Self.commitAssignment(ArenaSPRTMaxGames.self, value: arenaSPRTMaxGames) { arenaSPRTMaxGames = oldValue } } }
    public var batchStatsInterval: Int { didSet { if !Self.commitAssignment(BatchStatsInterval.self, value: batchStatsInterval) { batchStatsInterval = oldValue } } }
    public var klProbeInterval: Int { didSet { if !Self.commitAssignment(KLProbeInterval.self, value: klProbeInterval) { klProbeInterval = oldValue } } }
    public var lrCycleEnabled: Bool { didSet { if !Self.commitAssignment(LRCycleEnabled.self, value: lrCycleEnabled) { lrCycleEnabled = oldValue } } }
    public var lrCyclePeriodSteps: Int { didSet { if !Self.commitAssignment(LRCyclePeriodSteps.self, value: lrCyclePeriodSteps) { lrCyclePeriodSteps = oldValue } } }
    public var lrCycleCount: Int { didSet { if !Self.commitAssignment(LRCycleCount.self, value: lrCycleCount) { lrCycleCount = oldValue } } }
    public var lrCycleMin: Double { didSet { if !Self.commitAssignment(LRCycleMin.self, value: lrCycleMin) { lrCycleMin = oldValue } } }
    public var lrCycleMax: Double { didSet { if !Self.commitAssignment(LRCycleMax.self, value: lrCycleMax) { lrCycleMax = oldValue } } }
    public var lrCycleInvert: Bool { didSet { if !Self.commitAssignment(LRCycleInvert.self, value: lrCycleInvert) { lrCycleInvert = oldValue } } }
    public var momentumCycleEnabled: Bool { didSet { if !Self.commitAssignment(MomentumCycleEnabled.self, value: momentumCycleEnabled) { momentumCycleEnabled = oldValue } } }
    public var momentumCyclePeriodSteps: Int { didSet { if !Self.commitAssignment(MomentumCyclePeriodSteps.self, value: momentumCyclePeriodSteps) { momentumCyclePeriodSteps = oldValue } } }
    public var momentumCycleCount: Int { didSet { if !Self.commitAssignment(MomentumCycleCount.self, value: momentumCycleCount) { momentumCycleCount = oldValue } } }
    public var momentumCycleMin: Double { didSet { if !Self.commitAssignment(MomentumCycleMin.self, value: momentumCycleMin) { momentumCycleMin = oldValue } } }
    public var momentumCycleMax: Double { didSet { if !Self.commitAssignment(MomentumCycleMax.self, value: momentumCycleMax) { momentumCycleMax = oldValue } } }
    public var momentumCycleInvert: Bool { didSet { if !Self.commitAssignment(MomentumCycleInvert.self, value: momentumCycleInvert) { momentumCycleInvert = oldValue } } }
    public var lrCyclePeakEnd: Double { didSet { if !Self.commitAssignment(LRCyclePeakEnd.self, value: lrCyclePeakEnd) { lrCyclePeakEnd = oldValue } } }
    public var lrCycleTroughEnd: Double { didSet { if !Self.commitAssignment(LRCycleTroughEnd.self, value: lrCycleTroughEnd) { lrCycleTroughEnd = oldValue } } }
    public var lrCycleDecayHorizonSteps: Int { didSet { if !Self.commitAssignment(LRCycleDecayHorizonSteps.self, value: lrCycleDecayHorizonSteps) { lrCycleDecayHorizonSteps = oldValue } } }
    public var momentumFollowsLRCycle: Bool { didSet { if !Self.commitAssignment(MomentumFollowsLRCycle.self, value: momentumFollowsLRCycle) { momentumFollowsLRCycle = oldValue } } }
    public var momentumFollowStartLow: Double { didSet { if !Self.commitAssignment(MomentumFollowStartLow.self, value: momentumFollowStartLow) { momentumFollowStartLow = oldValue } } }
    public var momentumFollowStartHigh: Double { didSet { if !Self.commitAssignment(MomentumFollowStartHigh.self, value: momentumFollowStartHigh) { momentumFollowStartHigh = oldValue } } }
    public var momentumFollowEndLow: Double { didSet { if !Self.commitAssignment(MomentumFollowEndLow.self, value: momentumFollowEndLow) { momentumFollowEndLow = oldValue } } }
    public var momentumFollowEndHigh: Double { didSet { if !Self.commitAssignment(MomentumFollowEndHigh.self, value: momentumFollowEndHigh) { momentumFollowEndHigh = oldValue } } }
    public var periodicAutosaveIntervalSec: Double { didSet { if !Self.commitAssignment(PeriodicAutosaveIntervalSec.self, value: periodicAutosaveIntervalSec) { periodicAutosaveIntervalSec = oldValue } } }
    public var maxPeriodicAutosavesKept: Int { didSet { if !Self.commitAssignment(MaxPeriodicAutosavesKept.self, value: maxPeriodicAutosavesKept) { maxPeriodicAutosavesKept = oldValue } } }
    public var automaticSavePruningEnabled: Bool { didSet { if !Self.commitAssignment(AutomaticSavePruningEnabled.self, value: automaticSavePruningEnabled) { automaticSavePruningEnabled = oldValue } } }
    public var sessionSaveIncludeReplayBuffer: Bool { didSet { if !Self.commitAssignment(SessionSaveIncludeReplayBuffer.self, value: sessionSaveIncludeReplayBuffer) { sessionSaveIncludeReplayBuffer = oldValue } } }
    /// Stored as the enum; the raw value appears only at the persistence
    /// boundary (see `RandomSeedMode`).
    public var randomSeedMode: RandomSeedMode {
        didSet {
            if !Self.commitAssignment(RandomSeedModeParameter.self, value: randomSeedMode.rawValue) {
                randomSeedMode = oldValue
            }
        }
    }
    public var randomSeed: UInt64 { didSet { if !Self.commitAssignment(RandomSeed.self, value: randomSeed) { randomSeed = oldValue } } }

    /// Stored preferences found unusable at launch (wrong type or outside the
    /// declared range). The app starts on each one's declared default (a
    /// `--parameters` file or a resumed session may then set its own); the
    /// settings list shows them until the user resets each one
    /// (`resetInvalidStoredSetting(id:)`), which repairs only the stored
    /// entry and leaves the value in effect alone.
    public private(set) var invalidStoredSettings: [InvalidStoredSetting]

    private init() {
        // Read each value from UserDefaults (or definition default if absent / invalid).
        // didSet does not fire on initial assignment in init — which is what we want.
        self.entropyBonus = Self.read(EntropyBonus.self)
        self.illegalMassWeight = Self.read(IllegalMassWeight.self)
        self.policyLabelSmoothingEpsilon = Self.read(PolicyLabelSmoothingEpsilon.self)
        self.policyLabelSmoothingMode = PolicyLabelSmoothingMode(
            persistedRawValue: Self.read(PolicyLabelSmoothingModeParameter.self)
        )
        self.policyLabelSmoothingPerMove = Self.read(PolicyLabelSmoothingPerMove.self)
        self.policyLabelSmoothingPerMoveCap = Self.read(PolicyLabelSmoothingPerMoveCap.self)
        self.valueLabelSmoothingEpsilon = Self.read(ValueLabelSmoothingEpsilon.self)
        self.gradClipMaxNorm = Self.read(GradClipMaxNorm.self)
        self.weightDecay = Self.read(WeightDecay.self)
        self.dropoutRate = Self.read(DropoutRate.self)
        self.policyLossWeight = Self.read(PolicyLossWeight.self)
        self.valueLossWeight = Self.read(ValueLossWeight.self)
        self.learningRate = Self.read(LearningRate.self)
        self.momentumCoeff = Self.read(MomentumCoeff.self)
        self.sqrtBatchScalingLR = Self.read(SqrtBatchScalingLR.self)
        self.signedAdvantageComplementCE = Self.read(SignedAdvantageComplementCE.self)
        self.lrWarmupSteps = Self.read(LRWarmupSteps.self)
        self.drawPenalty = Self.read(DrawPenalty.self)
        self.selfPlayStartTau = Self.read(SelfPlayStartTau.self)
        self.selfPlayTargetTau = Self.read(SelfPlayTargetTau.self)
        self.selfPlayTauDecayPerPly = Self.read(SelfPlayTauDecayPerPly.self)
        self.selfPlayDrawKeepFraction = Self.read(SelfPlayDrawKeepFraction.self)
        self.selfPlayMaxPliesPerGame = Self.read(SelfPlayMaxPliesPerGame.self)
        self.recordSelfPlayGames = Self.read(RecordSelfPlayGames.self)
        self.drawWatchPDrawThreshold = Self.read(DrawWatchPDrawThreshold.self)
        self.drawWatchTerminateGames = Self.read(DrawWatchTerminateGames.self)
        self.drawWatchStreakLength = Self.read(DrawWatchStreakLength.self)
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
        self.replayBufferStratifyByMaterial = Self.read(ReplayBufferStratifyByMaterial.self)
        self.arenaPromoteThreshold = Self.read(ArenaPromoteThreshold.self)
        self.arenaGamesPerTournament = Self.read(ArenaGamesPerTournament.self)
        self.arenaAutoIntervalSec = Self.read(ArenaAutoIntervalSec.self)
        self.candidateProbeIntervalSec = Self.read(CandidateProbeIntervalSec.self)
        self.legalMassCollapseThreshold = Self.read(LegalMassCollapseThreshold.self)
        self.legalMassCollapseGraceSeconds = Self.read(LegalMassCollapseGraceSeconds.self)
        self.legalMassCollapseNoImprovementProbes = Self.read(LegalMassCollapseNoImprovementProbes.self)
        self.arenaConcurrency = Self.read(ArenaConcurrency.self)
        self.arenaPromotionCriterion = ArenaPromotionCriterion(
            persistedRawValue: Self.read(ArenaPromotionCriterionParameter.self)
        )
        self.arenaSPRTElo0 = Self.read(ArenaSPRTElo0.self)
        self.arenaSPRTElo1 = Self.read(ArenaSPRTElo1.self)
        self.arenaSPRTAlpha = Self.read(ArenaSPRTAlpha.self)
        self.arenaSPRTBeta = Self.read(ArenaSPRTBeta.self)
        self.arenaSPRTMinGames = Self.read(ArenaSPRTMinGames.self)
        self.arenaSPRTMaxGames = Self.read(ArenaSPRTMaxGames.self)
        self.batchStatsInterval = Self.read(BatchStatsInterval.self)
        self.klProbeInterval = Self.read(KLProbeInterval.self)
        self.lrCycleEnabled = Self.read(LRCycleEnabled.self)
        self.lrCyclePeriodSteps = Self.read(LRCyclePeriodSteps.self)
        self.lrCycleCount = Self.read(LRCycleCount.self)
        self.lrCycleMin = Self.read(LRCycleMin.self)
        self.lrCycleMax = Self.read(LRCycleMax.self)
        self.lrCycleInvert = Self.read(LRCycleInvert.self)
        self.momentumCycleEnabled = Self.read(MomentumCycleEnabled.self)
        self.momentumCyclePeriodSteps = Self.read(MomentumCyclePeriodSteps.self)
        self.momentumCycleCount = Self.read(MomentumCycleCount.self)
        self.momentumCycleMin = Self.read(MomentumCycleMin.self)
        self.momentumCycleMax = Self.read(MomentumCycleMax.self)
        self.momentumCycleInvert = Self.read(MomentumCycleInvert.self)
        self.lrCyclePeakEnd = Self.read(LRCyclePeakEnd.self)
        self.lrCycleTroughEnd = Self.read(LRCycleTroughEnd.self)
        self.lrCycleDecayHorizonSteps = Self.read(LRCycleDecayHorizonSteps.self)
        self.momentumFollowsLRCycle = Self.read(MomentumFollowsLRCycle.self)
        self.momentumFollowStartLow = Self.read(MomentumFollowStartLow.self)
        self.momentumFollowStartHigh = Self.read(MomentumFollowStartHigh.self)
        self.momentumFollowEndLow = Self.read(MomentumFollowEndLow.self)
        self.momentumFollowEndHigh = Self.read(MomentumFollowEndHigh.self)
        self.periodicAutosaveIntervalSec = Self.read(PeriodicAutosaveIntervalSec.self)
        self.maxPeriodicAutosavesKept = Self.read(MaxPeriodicAutosavesKept.self)
        self.automaticSavePruningEnabled = Self.read(AutomaticSavePruningEnabled.self)
        self.sessionSaveIncludeReplayBuffer = Self.read(SessionSaveIncludeReplayBuffer.self)
        self.randomSeedMode = RandomSeedMode(persistedRawValue: Self.read(RandomSeedModeParameter.self))
        self.randomSeed = Self.read(RandomSeed.self)
        self.invalidStoredSettings = Self.invalidStoredValuesFound.value.values.sorted { $0.id < $1.id }
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
        v[PolicyLabelSmoothingModeParameter.id] = PolicyLabelSmoothingModeParameter.encode(policyLabelSmoothingMode.rawValue)
        v[PolicyLabelSmoothingPerMove.id] = PolicyLabelSmoothingPerMove.encode(policyLabelSmoothingPerMove)
        v[PolicyLabelSmoothingPerMoveCap.id] = PolicyLabelSmoothingPerMoveCap.encode(policyLabelSmoothingPerMoveCap)
        v[ValueLabelSmoothingEpsilon.id] = ValueLabelSmoothingEpsilon.encode(valueLabelSmoothingEpsilon)
        v[GradClipMaxNorm.id] = GradClipMaxNorm.encode(gradClipMaxNorm)
        v[WeightDecay.id] = WeightDecay.encode(weightDecay)
        v[DropoutRate.id] = DropoutRate.encode(dropoutRate)
        v[PolicyLossWeight.id] = PolicyLossWeight.encode(policyLossWeight)
        v[ValueLossWeight.id] = ValueLossWeight.encode(valueLossWeight)
        v[LearningRate.id] = LearningRate.encode(learningRate)
        v[MomentumCoeff.id] = MomentumCoeff.encode(momentumCoeff)
        v[SqrtBatchScalingLR.id] = SqrtBatchScalingLR.encode(sqrtBatchScalingLR)
        v[SignedAdvantageComplementCE.id] = SignedAdvantageComplementCE.encode(signedAdvantageComplementCE)
        v[LRWarmupSteps.id] = LRWarmupSteps.encode(lrWarmupSteps)
        v[DrawPenalty.id] = DrawPenalty.encode(drawPenalty)
        v[SelfPlayStartTau.id] = SelfPlayStartTau.encode(selfPlayStartTau)
        v[SelfPlayTargetTau.id] = SelfPlayTargetTau.encode(selfPlayTargetTau)
        v[SelfPlayTauDecayPerPly.id] = SelfPlayTauDecayPerPly.encode(selfPlayTauDecayPerPly)
        v[SelfPlayDrawKeepFraction.id] = SelfPlayDrawKeepFraction.encode(selfPlayDrawKeepFraction)
        v[SelfPlayMaxPliesPerGame.id] = SelfPlayMaxPliesPerGame.encode(selfPlayMaxPliesPerGame)
        v[RecordSelfPlayGames.id] = RecordSelfPlayGames.encode(recordSelfPlayGames)
        v[DrawWatchPDrawThreshold.id] = DrawWatchPDrawThreshold.encode(drawWatchPDrawThreshold)
        v[DrawWatchTerminateGames.id] = DrawWatchTerminateGames.encode(drawWatchTerminateGames)
        v[DrawWatchStreakLength.id] = DrawWatchStreakLength.encode(drawWatchStreakLength)
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
        v[ReplayBufferStratifyByMaterial.id] = ReplayBufferStratifyByMaterial.encode(replayBufferStratifyByMaterial)
        v[ArenaPromoteThreshold.id] = ArenaPromoteThreshold.encode(arenaPromoteThreshold)
        v[ArenaGamesPerTournament.id] = ArenaGamesPerTournament.encode(arenaGamesPerTournament)
        v[ArenaAutoIntervalSec.id] = ArenaAutoIntervalSec.encode(arenaAutoIntervalSec)
        v[CandidateProbeIntervalSec.id] = CandidateProbeIntervalSec.encode(candidateProbeIntervalSec)
        v[LegalMassCollapseThreshold.id] = LegalMassCollapseThreshold.encode(legalMassCollapseThreshold)
        v[LegalMassCollapseGraceSeconds.id] = LegalMassCollapseGraceSeconds.encode(legalMassCollapseGraceSeconds)
        v[LegalMassCollapseNoImprovementProbes.id] = LegalMassCollapseNoImprovementProbes.encode(legalMassCollapseNoImprovementProbes)
        v[ArenaConcurrency.id] = ArenaConcurrency.encode(arenaConcurrency)
        v[ArenaPromotionCriterionParameter.id] = ArenaPromotionCriterionParameter.encode(arenaPromotionCriterion.rawValue)
        v[ArenaSPRTElo0.id] = ArenaSPRTElo0.encode(arenaSPRTElo0)
        v[ArenaSPRTElo1.id] = ArenaSPRTElo1.encode(arenaSPRTElo1)
        v[ArenaSPRTAlpha.id] = ArenaSPRTAlpha.encode(arenaSPRTAlpha)
        v[ArenaSPRTBeta.id] = ArenaSPRTBeta.encode(arenaSPRTBeta)
        v[ArenaSPRTMinGames.id] = ArenaSPRTMinGames.encode(arenaSPRTMinGames)
        v[ArenaSPRTMaxGames.id] = ArenaSPRTMaxGames.encode(arenaSPRTMaxGames)
        v[BatchStatsInterval.id] = BatchStatsInterval.encode(batchStatsInterval)
        v[KLProbeInterval.id] = KLProbeInterval.encode(klProbeInterval)
        v[LRCycleEnabled.id] = LRCycleEnabled.encode(lrCycleEnabled)
        v[LRCyclePeriodSteps.id] = LRCyclePeriodSteps.encode(lrCyclePeriodSteps)
        v[LRCycleCount.id] = LRCycleCount.encode(lrCycleCount)
        v[LRCycleMin.id] = LRCycleMin.encode(lrCycleMin)
        v[LRCycleMax.id] = LRCycleMax.encode(lrCycleMax)
        v[LRCycleInvert.id] = LRCycleInvert.encode(lrCycleInvert)
        v[MomentumCycleEnabled.id] = MomentumCycleEnabled.encode(momentumCycleEnabled)
        v[MomentumCyclePeriodSteps.id] = MomentumCyclePeriodSteps.encode(momentumCyclePeriodSteps)
        v[MomentumCycleCount.id] = MomentumCycleCount.encode(momentumCycleCount)
        v[MomentumCycleMin.id] = MomentumCycleMin.encode(momentumCycleMin)
        v[MomentumCycleMax.id] = MomentumCycleMax.encode(momentumCycleMax)
        v[MomentumCycleInvert.id] = MomentumCycleInvert.encode(momentumCycleInvert)
        v[LRCyclePeakEnd.id] = LRCyclePeakEnd.encode(lrCyclePeakEnd)
        v[LRCycleTroughEnd.id] = LRCycleTroughEnd.encode(lrCycleTroughEnd)
        v[LRCycleDecayHorizonSteps.id] = LRCycleDecayHorizonSteps.encode(lrCycleDecayHorizonSteps)
        v[MomentumFollowsLRCycle.id] = MomentumFollowsLRCycle.encode(momentumFollowsLRCycle)
        v[MomentumFollowStartLow.id] = MomentumFollowStartLow.encode(momentumFollowStartLow)
        v[MomentumFollowStartHigh.id] = MomentumFollowStartHigh.encode(momentumFollowStartHigh)
        v[MomentumFollowEndLow.id] = MomentumFollowEndLow.encode(momentumFollowEndLow)
        v[MomentumFollowEndHigh.id] = MomentumFollowEndHigh.encode(momentumFollowEndHigh)
        v[PeriodicAutosaveIntervalSec.id] = PeriodicAutosaveIntervalSec.encode(periodicAutosaveIntervalSec)
        v[MaxPeriodicAutosavesKept.id] = MaxPeriodicAutosavesKept.encode(maxPeriodicAutosavesKept)
        v[AutomaticSavePruningEnabled.id] = AutomaticSavePruningEnabled.encode(automaticSavePruningEnabled)
        v[SessionSaveIncludeReplayBuffer.id] = SessionSaveIncludeReplayBuffer.encode(sessionSaveIncludeReplayBuffer)
        v[RandomSeedModeParameter.id] = RandomSeedModeParameter.encode(randomSeedMode.rawValue)
        v[RandomSeed.id] = RandomSeed.encode(randomSeed)
        return v
    }

    // MARK: Apply (from JSON load, from CLI)

    /// Applies a value map (e.g. from a parsed parameters.json) to the singleton.
    /// Each value goes through the typed setter so validation runs and SwiftUI sees the mutation.
    /// Unknown ids: throw `unknownParameter`. Out-of-range / wrong-type: throw the corresponding error.
    ///
    /// All-or-nothing: the whole map is validated before anything is
    /// assigned, so a rejected file leaves the singleton exactly as it was
    /// instead of half-applied in dictionary order.
    public func apply(_ values: [String: ParameterValue]) throws {
        try Self.validate(values)
        for (id, raw) in values {
            try applyOne(id: id, raw: raw)
        }
    }

    /// Validate a value map against the declared parameters without applying
    /// it. Ids are checked in sorted order so the reported error is the same
    /// on every run.
    public nonisolated static func validate(_ values: [String: ParameterValue]) throws {
        let definitionsByID = Dictionary(uniqueKeysWithValues: allKeys.map { ($0.id, $0.definition) })
        for id in values.keys.sorted() {
            guard let definition = definitionsByID[id], let raw = values[id] else {
                throw TrainingConfigError.unknownParameter(id: id)
            }
            try definition.validate(raw)
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
        case PolicyLabelSmoothingModeParameter.id:
            try PolicyLabelSmoothingModeParameter.definition.validate(raw)
            policyLabelSmoothingMode = PolicyLabelSmoothingMode(
                persistedRawValue: try PolicyLabelSmoothingModeParameter.decode(raw)
            )
        case PolicyLabelSmoothingPerMove.id:
            try PolicyLabelSmoothingPerMove.definition.validate(raw); policyLabelSmoothingPerMove = try PolicyLabelSmoothingPerMove.decode(raw)
        case PolicyLabelSmoothingPerMoveCap.id:
            try PolicyLabelSmoothingPerMoveCap.definition.validate(raw); policyLabelSmoothingPerMoveCap = try PolicyLabelSmoothingPerMoveCap.decode(raw)
        case ValueLabelSmoothingEpsilon.id:
            try ValueLabelSmoothingEpsilon.definition.validate(raw); valueLabelSmoothingEpsilon = try ValueLabelSmoothingEpsilon.decode(raw)
        case GradClipMaxNorm.id:
            try GradClipMaxNorm.definition.validate(raw); gradClipMaxNorm = try GradClipMaxNorm.decode(raw)
        case WeightDecay.id:
            try WeightDecay.definition.validate(raw); weightDecay = try WeightDecay.decode(raw)
        case DropoutRate.id:
            try DropoutRate.definition.validate(raw); dropoutRate = try DropoutRate.decode(raw)
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
        case SignedAdvantageComplementCE.id:
            try SignedAdvantageComplementCE.definition.validate(raw); signedAdvantageComplementCE = try SignedAdvantageComplementCE.decode(raw)
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
        case RecordSelfPlayGames.id:
            try RecordSelfPlayGames.definition.validate(raw); recordSelfPlayGames = try RecordSelfPlayGames.decode(raw)
        case DrawWatchPDrawThreshold.id:
            try DrawWatchPDrawThreshold.definition.validate(raw); drawWatchPDrawThreshold = try DrawWatchPDrawThreshold.decode(raw)
        case DrawWatchTerminateGames.id:
            try DrawWatchTerminateGames.definition.validate(raw); drawWatchTerminateGames = try DrawWatchTerminateGames.decode(raw)
        case DrawWatchStreakLength.id:
            try DrawWatchStreakLength.definition.validate(raw); drawWatchStreakLength = try DrawWatchStreakLength.decode(raw)
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
        case ReplayBufferStratifyByMaterial.id:
            try ReplayBufferStratifyByMaterial.definition.validate(raw); replayBufferStratifyByMaterial = try ReplayBufferStratifyByMaterial.decode(raw)
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
        case ArenaPromotionCriterionParameter.id:
            try ArenaPromotionCriterionParameter.definition.validate(raw)
            arenaPromotionCriterion = ArenaPromotionCriterion(
                persistedRawValue: try ArenaPromotionCriterionParameter.decode(raw)
            )
        case ArenaSPRTElo0.id:
            try ArenaSPRTElo0.definition.validate(raw); arenaSPRTElo0 = try ArenaSPRTElo0.decode(raw)
        case ArenaSPRTElo1.id:
            try ArenaSPRTElo1.definition.validate(raw); arenaSPRTElo1 = try ArenaSPRTElo1.decode(raw)
        case ArenaSPRTAlpha.id:
            try ArenaSPRTAlpha.definition.validate(raw); arenaSPRTAlpha = try ArenaSPRTAlpha.decode(raw)
        case ArenaSPRTBeta.id:
            try ArenaSPRTBeta.definition.validate(raw); arenaSPRTBeta = try ArenaSPRTBeta.decode(raw)
        case ArenaSPRTMinGames.id:
            try ArenaSPRTMinGames.definition.validate(raw); arenaSPRTMinGames = try ArenaSPRTMinGames.decode(raw)
        case ArenaSPRTMaxGames.id:
            try ArenaSPRTMaxGames.definition.validate(raw); arenaSPRTMaxGames = try ArenaSPRTMaxGames.decode(raw)
        case BatchStatsInterval.id:
            try BatchStatsInterval.definition.validate(raw); batchStatsInterval = try BatchStatsInterval.decode(raw)
        case KLProbeInterval.id:
            try KLProbeInterval.definition.validate(raw); klProbeInterval = try KLProbeInterval.decode(raw)
        case LRCycleEnabled.id:
            try LRCycleEnabled.definition.validate(raw); lrCycleEnabled = try LRCycleEnabled.decode(raw)
        case LRCyclePeriodSteps.id:
            try LRCyclePeriodSteps.definition.validate(raw); lrCyclePeriodSteps = try LRCyclePeriodSteps.decode(raw)
        case LRCycleCount.id:
            try LRCycleCount.definition.validate(raw); lrCycleCount = try LRCycleCount.decode(raw)
        case LRCycleMin.id:
            try LRCycleMin.definition.validate(raw); lrCycleMin = try LRCycleMin.decode(raw)
        case LRCycleMax.id:
            try LRCycleMax.definition.validate(raw); lrCycleMax = try LRCycleMax.decode(raw)
        case LRCycleInvert.id:
            try LRCycleInvert.definition.validate(raw); lrCycleInvert = try LRCycleInvert.decode(raw)
        case MomentumCycleEnabled.id:
            try MomentumCycleEnabled.definition.validate(raw); momentumCycleEnabled = try MomentumCycleEnabled.decode(raw)
        case MomentumCyclePeriodSteps.id:
            try MomentumCyclePeriodSteps.definition.validate(raw); momentumCyclePeriodSteps = try MomentumCyclePeriodSteps.decode(raw)
        case MomentumCycleCount.id:
            try MomentumCycleCount.definition.validate(raw); momentumCycleCount = try MomentumCycleCount.decode(raw)
        case MomentumCycleMin.id:
            try MomentumCycleMin.definition.validate(raw); momentumCycleMin = try MomentumCycleMin.decode(raw)
        case MomentumCycleMax.id:
            try MomentumCycleMax.definition.validate(raw); momentumCycleMax = try MomentumCycleMax.decode(raw)
        case MomentumCycleInvert.id:
            try MomentumCycleInvert.definition.validate(raw); momentumCycleInvert = try MomentumCycleInvert.decode(raw)
        case LRCyclePeakEnd.id:
            try LRCyclePeakEnd.definition.validate(raw); lrCyclePeakEnd = try LRCyclePeakEnd.decode(raw)
        case LRCycleTroughEnd.id:
            try LRCycleTroughEnd.definition.validate(raw); lrCycleTroughEnd = try LRCycleTroughEnd.decode(raw)
        case LRCycleDecayHorizonSteps.id:
            try LRCycleDecayHorizonSteps.definition.validate(raw); lrCycleDecayHorizonSteps = try LRCycleDecayHorizonSteps.decode(raw)
        case MomentumFollowsLRCycle.id:
            try MomentumFollowsLRCycle.definition.validate(raw); momentumFollowsLRCycle = try MomentumFollowsLRCycle.decode(raw)
        case MomentumFollowStartLow.id:
            try MomentumFollowStartLow.definition.validate(raw); momentumFollowStartLow = try MomentumFollowStartLow.decode(raw)
        case MomentumFollowStartHigh.id:
            try MomentumFollowStartHigh.definition.validate(raw); momentumFollowStartHigh = try MomentumFollowStartHigh.decode(raw)
        case MomentumFollowEndLow.id:
            try MomentumFollowEndLow.definition.validate(raw); momentumFollowEndLow = try MomentumFollowEndLow.decode(raw)
        case MomentumFollowEndHigh.id:
            try MomentumFollowEndHigh.definition.validate(raw); momentumFollowEndHigh = try MomentumFollowEndHigh.decode(raw)
        case PeriodicAutosaveIntervalSec.id:
            try PeriodicAutosaveIntervalSec.definition.validate(raw); periodicAutosaveIntervalSec = try PeriodicAutosaveIntervalSec.decode(raw)
        case MaxPeriodicAutosavesKept.id:
            try MaxPeriodicAutosavesKept.definition.validate(raw); maxPeriodicAutosavesKept = try MaxPeriodicAutosavesKept.decode(raw)
        case AutomaticSavePruningEnabled.id:
            try AutomaticSavePruningEnabled.definition.validate(raw); automaticSavePruningEnabled = try AutomaticSavePruningEnabled.decode(raw)
        case SessionSaveIncludeReplayBuffer.id:
            try SessionSaveIncludeReplayBuffer.definition.validate(raw); sessionSaveIncludeReplayBuffer = try SessionSaveIncludeReplayBuffer.decode(raw)
        case RandomSeedModeParameter.id:
            try RandomSeedModeParameter.definition.validate(raw)
            randomSeedMode = RandomSeedMode(persistedRawValue: try RandomSeedModeParameter.decode(raw))
        case RandomSeed.id:
            try RandomSeed.definition.validate(raw); randomSeed = try RandomSeed.decode(raw)
        default:
            throw TrainingConfigError.unknownParameter(id: id)
        }
    }

    // MARK: Persistence (per-key UserDefaults)

    /// Current persisted value for `key`, readable from **any** thread.
    ///
    /// The singleton is `@MainActor`, which makes it unreachable from the
    /// pre-flight CLI paths — and, worse, actively dangerous from any thread
    /// the main actor might need: `SweepCLI`/`ProbeModelCLI`-style `syncWait`
    /// helpers block the calling thread on a semaphore while a detached task
    /// runs, so an `await TrainingParameters.shared.…` from inside one can
    /// never complete and deadlocks the process.
    ///
    /// This goes through the same validated `UserDefaults` path `init` uses,
    /// with no actor hop. Note it reads what is *persisted*: a transient
    /// `--parameters` run-override applied under `suppressPersistence` will
    /// not be visible here.
    public nonisolated static func persistedValue<K: TrainingParameterKey>(_ key: K.Type) -> K.Value {
        read(key)
    }

    /// The value `init` and `persistedValue(_:)` use for `key`: the stored
    /// value when it is usable, else the declared default. An unusable stored
    /// value is never silently replaced: it is recorded (see
    /// `invalidStoredSettings`), logged once, and left in `UserDefaults`
    /// untouched until the user resets it.
    private nonisolated static func read<K: TrainingParameterKey>(_ key: K.Type) -> K.Value {
        switch inspectStored(key, in: .standard) {
        case .absent:
            return K.declaredDefault
        case .valid(let value):
            return value
        case .invalid(let finding):
            recordInvalidStoredValue(finding)
            return K.declaredDefault
        }
    }

    /// What `defaults` holds for one parameter.
    enum StoredInspection<Value: Sendable>: Sendable {
        case absent
        case valid(Value)
        case invalid(InvalidStoredSetting)
    }

    /// Classify the stored entry for `key` in `defaults`: absent, usable, or
    /// unusable with the reason. Pure apart from reading `defaults`, so tests
    /// run it against a private suite.
    nonisolated static func inspectStored<K: TrainingParameterKey>(
        _ key: K.Type,
        in defaults: UserDefaults
    ) -> StoredInspection<K.Value> {
        guard let object = defaults.object(forKey: K.id) else { return .absent }
        let definition = K.definition
        let replacement = definition.defaultValue.displayText
        func invalid(_ found: String, _ problem: String) -> StoredInspection<K.Value> {
            .invalid(InvalidStoredSetting(
                id: K.id, name: definition.name, found: found, problem: problem, replacement: replacement
            ))
        }
        // A stored number is classified by its own kind — true/false, whole
        // number or real — through `ParameterValue(jsonValue:id:)`, the same
        // reader a parameters file goes through, never coerced into the
        // key's kind: `NSNumber` bridging would read 7.9 as the whole number
        // 7, `true` as 1, and 1 as `true`, each silently running a setting
        // nobody stored. A whole number for a real-valued key is accepted,
        // as it is in a parameters file.
        func storedKind(_ value: ParameterValue) -> String {
            switch value {
            case .bool: return "a true/false value"
            case .int, .uint64: return "a whole number"
            case .double: return "a real number"
            }
        }
        let raw: ParameterValue
        switch definition.type {
        case .bool, .int, .double:
            let expected: String
            switch definition.type {
            case .bool: expected = "not a true/false value"
            case .int: expected = "not a whole number"
            case .double, .uint64: expected = "not a number"
            }
            guard let number = object as? NSNumber else {
                return invalid("\(object)", "stored as \(type(of: object)), \(expected)")
            }
            let value: ParameterValue
            do {
                value = try ParameterValue(jsonValue: number, id: K.id)
            } catch {
                return invalid("\(object)", "\(error)")
            }
            switch (definition.type, value) {
            case (.bool, .bool), (.int, .int), (.double, .double), (.double, .int):
                raw = value
            default:
                return invalid("\(object)", "stored as \(storedKind(value)), \(expected)")
            }
        case .uint64:
            guard let text = object as? String else {
                return invalid("\(object)", "stored as \(type(of: object)), not a decimal string")
            }
            guard let n = UInt64(strictDecimal: text) else {
                return invalid(text, "not a whole number from 0 to \(UInt64.max) written in digits only")
            }
            raw = .uint64(n)
        }
        do {
            try definition.validate(raw)
        } catch {
            return invalid(raw.displayText, "\(error)")
        }
        do {
            return .valid(try K.decode(raw))
        } catch {
            return invalid(raw.displayText, "\(error)")
        }
    }

    /// Unusable stored values found by `read`, keyed by parameter id. Read
    /// can run before the singleton exists (the CLI's `persistedValue`) and
    /// more than once per key, so findings are collected here, deduplicated,
    /// and logged the first time each is seen.
    private nonisolated static let invalidStoredValuesFound = SyncBox<[String: InvalidStoredSetting]>([:])

    private nonisolated static func recordInvalidStoredValue(_ finding: InvalidStoredSetting) {
        let isNew = invalidStoredValuesFound.mutate { found -> Bool in
            guard found[finding.id] == nil else { return false }
            found[finding.id] = finding
            return true
        }
        guard isNew else { return }
        let line = "[PARAM-INVALID] \(finding.id): stored value \(finding.found) cannot be used (\(finding.problem)); "
            + "using \(finding.replacement) until it is reset — the stored value is left as found"
        SessionLogger.shared.log(line)
        FileHandle.standardError.write(Data((line + "\n").utf8))
    }

    /// Reset one unusable stored value (from `invalidStoredSettings`): the
    /// user's explicit choice from the settings list. It repairs the stored
    /// entry only (`repairStoredValue`) and never assigns the live value: by
    /// the time the user clicks, this run may be on a `--parameters` value or
    /// a resumed session's value, and a reset of what is saved must not
    /// replace what is running. A stored entry that has meanwhile been
    /// replaced by a usable value is left alone.
    public func resetInvalidStoredSetting(id: String) throws {
        guard invalidStoredSettings.contains(where: { $0.id == id }) else {
            throw TrainingConfigError.unknownParameter(id: id)
        }
        guard let key = Self.allKeys.first(where: { $0.id == id }) else {
            throw TrainingConfigError.unknownParameter(id: id)
        }
        let repair = Self.repairStoredValue(key, in: .standard)
        invalidStoredSettings.removeAll { $0.id == id }
        Self.invalidStoredValuesFound.modify { $0[id] = nil }
        switch repair {
        case .replacedWithDeclaredDefault(let raw):
            SessionLogger.shared.log(
                "[PARAM-INVALID] \(id): stored value reset to \(raw.displayText) by the user (the current run's value is unchanged)"
            )
        case .alreadyUsable:
            SessionLogger.shared.log(
                "[PARAM-INVALID] \(id): stored value was already replaced by a usable one; nothing reset"
            )
        }
    }

    /// What `repairStoredValue` did with one stored entry.
    enum StoredValueRepair: Equatable, Sendable {
        /// The entry was still unusable; it now holds the declared default.
        case replacedWithDeclaredDefault(ParameterValue)
        /// The entry was not unusable any more (a valid value replaced it
        /// since launch, or it was removed), so it was left as it is.
        case alreadyUsable
    }

    /// Replace `key`'s stored entry in `defaults` with the declared default
    /// if — and only if — it is still unusable (`inspectStored`). Touches
    /// nothing but that one entry: no live value, no other key.
    nonisolated static func repairStoredValue<K: TrainingParameterKey>(
        _ key: K.Type,
        in defaults: UserDefaults
    ) -> StoredValueRepair {
        guard case .invalid = inspectStored(K.self, in: defaults) else { return .alreadyUsable }
        let raw = K.definition.defaultValue
        store(raw, forKey: K.id, in: defaults)
        return .replacedWithDeclaredDefault(raw)
    }

    /// Write `raw` under `id` in the stored form `inspectStored` reads back:
    /// the one place a parameter's `UserDefaults` entry is written.
    nonisolated static func store(_ raw: ParameterValue, forKey id: String, in defaults: UserDefaults) {
        switch raw {
        case .bool(let x): defaults.set(x, forKey: id)
        case .int(let x): defaults.set(x, forKey: id)
        case .double(let x): defaults.set(x, forKey: id)
        // A decimal string, like every other serialization of a UInt64
        // parameter: a property-list integer is signed 64-bit.
        case .uint64(let x): defaults.set(String(x), forKey: id)
        }
    }

    /// When true, the `didSet` persisters skip writing to `UserDefaults`.
    /// Set only around a transient run-override apply (`--parameters` /
    /// "Load Parameters…") so a per-run hyperparameter override does NOT
    /// mutate the user's saved defaults — and, critically, does not leak
    /// across processes (a headless `--train` arm and the GUI share the
    /// same UserDefaults domain; without this, an experiment arm's value
    /// silently became the next launch's default — the 2026-06-12
    /// dropout=0.7 leak). Toggled and read only synchronously on the main
    /// thread inside `applyCliConfigOverrides` (apply → didSet → persist is
    /// one synchronous main-actor sequence), so the unchecked storage is
    /// race-free in practice.
    nonisolated(unsafe) static var suppressPersistence = false

    /// True only for the duration of one `restoreFromSession` assignment of a
    /// value outside the declared range. Set and read synchronously on the
    /// main actor inside that call (assign → didSet → commit is one
    /// synchronous sequence), like `suppressPersistence`.
    nonisolated(unsafe) private static var admittingSessionValueOutsideDeclaredRange = false

    /// Session resume's write: restore a resumed `.dcmsession`'s own saved
    /// value, even when it lies outside the range declared today.
    ///
    /// **The one intentional exception to single-path validation.** Every
    /// other writer is held to the declared range. A resume is different:
    /// resuming means continuing the saved run exactly, and that run trained
    /// under its saved values. A range that was narrower when the session was
    /// saved (or a value the drifted popovers accepted before they enforced
    /// the declarations) must not turn a resume into a run under different
    /// hyperparameters. Substituting the current value — what resume used to
    /// do — was exactly that.
    ///
    /// An in-range value is assigned normally (validated and persisted, as
    /// every resume write always has been). An out-of-range value is logged
    /// as a `[RESUME-PARAM] WARNING`, held in memory for this run, and never
    /// persisted to `UserDefaults`: it is the session's value, not an app
    /// setting, and the next launch's validated load would reject it. A later
    /// in-range edit replaces and persists normally; a later session save
    /// carries the restored value forward.
    func restoreFromSession<K: TrainingParameterKey>(
        _ key: K.Type,
        _ value: K.Value,
        into keyPath: ReferenceWritableKeyPath<TrainingParameters, K.Value>
    ) {
        if K.isWithinDeclaration(value) {
            self[keyPath: keyPath] = value
            return
        }
        let definition = K.definition
        let rangeText = definition.doubleRange.map { "\($0.min)...\($0.max)" }
            ?? definition.intRange.map { "\($0.min)...\($0.max)" }
            ?? "(\(definition.type))"
        SessionLogger.shared.log(
            "[RESUME-PARAM] WARNING \(K.id): saved value \(value) is outside the current declared range \(rangeText); "
                + "restored anyway (a resume runs on the session's own values), held for this run only and not saved to app settings"
        )
        Self.admittingSessionValueOutsideDeclaredRange = true
        defer { Self.admittingSessionValueOutsideDeclaredRange = false }
        self[keyPath: keyPath] = value
    }

    /// Restore a `Double` parameter whose session copy is stored as `Float`.
    ///
    /// Sessions keep the trainer's hyperparameters as `Float` (what the graph
    /// is fed). Widening one with `Double(_:)` keeps the float's binary value,
    /// so a saved 0.1 comes back as 0.10000000149011612 — which then persists
    /// to `UserDefaults`, shows up in `parameters.json`, and can land a value
    /// typed at a declared bound just outside it. The session value is the
    /// number that was set, so it is widened through
    /// `doubleFromSavedFloat(_:)` instead.
    func restoreFromSession<K: TrainingParameterKey>(
        _ key: K.Type,
        savedFloat: Float,
        into keyPath: ReferenceWritableKeyPath<TrainingParameters, Double>
    ) where K.Value == Double {
        restoreFromSession(K.self, Self.doubleFromSavedFloat(savedFloat), into: keyPath)
    }

    /// Session resume's write for a value that belongs to the resumed run but
    /// not to the user's settings: a parameter's declared pre-feature value,
    /// applied because the session predates the parameter. Validated like
    /// every assignment, but never written to `UserDefaults` — the session
    /// factually trained without the feature, while the user's saved setting
    /// (say, dropout 0.7) still governs the next fresh run. A later edit of
    /// the field persists normally; a later session save carries the held
    /// value forward.
    func holdForThisRun(_ assign: () -> Void) {
        let previous = Self.suppressPersistence
        Self.suppressPersistence = true
        defer { Self.suppressPersistence = previous }
        assign()
    }

    /// `holdForThisRun(_:)` for one key path.
    func holdForThisRun<K: TrainingParameterKey>(
        _ key: K.Type,
        _ value: K.Value,
        into keyPath: ReferenceWritableKeyPath<TrainingParameters, K.Value>
    ) {
        holdForThisRun { self[keyPath: keyPath] = value }
    }

    /// The `Double` whose shortest decimal text is the same as `saved`'s: the
    /// value that was typed or computed before it was narrowed to `Float`.
    /// `Float.description` is the shortest text that reads back as the same
    /// `Float` (including `nan` / `inf`), and every such text parses as a
    /// `Double`, so a failure here is a toolchain defect, not bad data.
    nonisolated static func doubleFromSavedFloat(_ saved: Float) -> Double {
        guard let value = Double(saved.description) else {
            preconditionFailure("Float.description of \(saved) did not parse as a Double")
        }
        return value
    }

    /// The singleton's setter hook: validate the newly assigned value against
    /// the declared range and, if it passes, persist it to `UserDefaults`
    /// (unless `suppressPersistence`). Returns false for an out-of-range
    /// value, in which case the calling `didSet` restores the previous value.
    ///
    /// This is the backstop behind every other validation site. Writers that
    /// can show an error — the popovers, the `--parameters` loader — check
    /// first with the same `validateAgainstDeclaration`, so a
    /// rejection here means a code path skipped that check; it is logged
    /// loudly rather than trapped, so a stray write can neither crash the app
    /// nor leave the singleton holding a value that disagrees with what is
    /// persisted. (Before this, the rejection was an `assertionFailure` —
    /// a crash in Debug, and in Release the in-memory value silently stayed
    /// out of range while `UserDefaults` kept the old one, so a session ran on
    /// a value the next launch would not see.) Validation runs even when
    /// persistence is suppressed, so a transient CLI override is held to the
    /// same range.
    private nonisolated static func commitAssignment<K: TrainingParameterKey>(_ key: K.Type, value: K.Value) -> Bool {
        let raw = K.encode(value)
        do {
            try K.definition.validate(raw)
        } catch {
            if admittingSessionValueOutsideDeclaredRange {
                // `restoreFromSession` is holding a resumed session's own
                // out-of-range value in memory for this run. Never persisted:
                // it is the session's value, not an app setting, and the
                // UserDefaults load would reject it on the next launch anyway.
                return true
            }
            let message = "[PARAM-REJECTED] \(error.localizedDescription); assignment reverted to the previous value"
            SessionLogger.shared.log(message)
            FileHandle.standardError.write(Data((message + "\n").utf8))
            return false
        }
        if suppressPersistence { return true }
        store(raw, forKey: K.id, in: .standard)
        return true
    }

    // MARK: Registry

    /// All keys, in declaration order. Used by save/load and by the `--show-default-parameters` CLI flag.
    public nonisolated static let allKeys: [any TrainingParameterKey.Type] = [
        EntropyBonus.self,
        IllegalMassWeight.self,
        PolicyLabelSmoothingEpsilon.self,
        PolicyLabelSmoothingModeParameter.self,
        PolicyLabelSmoothingPerMove.self,
        PolicyLabelSmoothingPerMoveCap.self,
        ValueLabelSmoothingEpsilon.self,
        GradClipMaxNorm.self,
        WeightDecay.self,
        DropoutRate.self,
        PolicyLossWeight.self,
        ValueLossWeight.self,
        LearningRate.self,
        MomentumCoeff.self,
        SqrtBatchScalingLR.self,
        SignedAdvantageComplementCE.self,
        LRWarmupSteps.self,
        DrawPenalty.self,
        SelfPlayStartTau.self,
        SelfPlayTargetTau.self,
        SelfPlayTauDecayPerPly.self,
        SelfPlayDrawKeepFraction.self,
        SelfPlayMaxPliesPerGame.self,
        RecordSelfPlayGames.self,
        DrawWatchPDrawThreshold.self,
        DrawWatchTerminateGames.self,
        DrawWatchStreakLength.self,
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
        ReplayBufferStratifyByMaterial.self,
        ArenaPromoteThreshold.self,
        ArenaGamesPerTournament.self,
        ArenaAutoIntervalSec.self,
        CandidateProbeIntervalSec.self,
        LegalMassCollapseThreshold.self,
        LegalMassCollapseGraceSeconds.self,
        LegalMassCollapseNoImprovementProbes.self,
        ArenaConcurrency.self,
        ArenaPromotionCriterionParameter.self,
        ArenaSPRTElo0.self,
        ArenaSPRTElo1.self,
        ArenaSPRTAlpha.self,
        ArenaSPRTBeta.self,
        ArenaSPRTMinGames.self,
        ArenaSPRTMaxGames.self,
        BatchStatsInterval.self,
        KLProbeInterval.self,
        LRCycleEnabled.self,
        LRCyclePeriodSteps.self,
        LRCycleCount.self,
        LRCycleMin.self,
        LRCycleMax.self,
        LRCycleInvert.self,
        MomentumCycleEnabled.self,
        MomentumCyclePeriodSteps.self,
        MomentumCycleCount.self,
        MomentumCycleMin.self,
        MomentumCycleMax.self,
        MomentumCycleInvert.self,
        LRCyclePeakEnd.self,
        LRCycleTroughEnd.self,
        LRCycleDecayHorizonSteps.self,
        MomentumFollowsLRCycle.self,
        MomentumFollowStartLow.self,
        MomentumFollowStartHigh.self,
        MomentumFollowEndLow.self,
        MomentumFollowEndHigh.self,
        PeriodicAutosaveIntervalSec.self,
        MaxPeriodicAutosavesKept.self,
        AutomaticSavePruningEnabled.self,
        SessionSaveIncludeReplayBuffer.self,
        RandomSeedModeParameter.self,
        RandomSeed.self
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
            dict[def.id] = def.defaultValue.jsonValue
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
            case .uint64:
                if let r = def.uint64Range {
                    rangeText = "UInt64 as a decimal string, range \(r.min)..\(r.max)"
                } else {
                    rangeText = "UInt64 as a decimal string"
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
                case .uint64:
                    typeText = "UInt64 (written as a decimal string)"
                    rangeText = def.uint64Range.map { "\($0.min)..\($0.max)" } ?? "—"
                    if case .uint64(let x) = def.defaultValue { defaultText = "\"\(x)\"" } else { defaultText = "?" }
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
            dict[id] = raw.jsonValue
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
            values[id] = try ParameterValue(jsonValue: anyValue, id: id)
        }
        try apply(values)
    }
}
