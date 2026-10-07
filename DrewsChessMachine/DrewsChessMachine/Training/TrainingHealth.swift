import Foundation

// MARK: - Training health: the pure core
//
// Every training path (GUI Play-and-Train, `--replay-corpus`,
// `--train-vs-uci`) and the offline `--replay-health-log` judge a run's
// health through the one evaluator in this file, so the nine rules and
// their thresholds have exactly one home. Nothing here touches the GPU, the
// trainer, the replay buffer, a file or the log: the inputs are values the
// paths already compute (`TrainStepTiming`, `LayerHealthSummary`), the
// output is a list of events that `TrainingHealthLog` renders. The
// lock-protected wrapper that collects per-step records and serializes
// evaluations is `TrainingHealthMonitor`.
//
// Design and evidence: `documentation/plans-active/TRAINING_HEALTH_ALARMS_PLAN.md`
// (Part R for the rules, Part D for the types). The thresholds were measured
// against three incident runs and the healthy baselines listed there; they
// are declared constants (owner decision OD-14), never parameters.

/// One training-health rule. The raw value is the stable id used in log
/// lines, results.json and parameter ids. Declaration order is rule order:
/// events, stop decisions and summaries are always reported in it.
enum TrainingHealthRule: String, CaseIterable, Codable, Sendable {
    case nonFinite = "non_finite"
    case deadChannels = "dead_channels"
    case valueFC1ZeroVelocity = "value_fc1_zero_velocity"
    case illegalMass = "illegal_mass"
    case gradientCollapse = "gradient_collapse"
    case lossSpike = "loss_spike"
    case policyOffsetDrift = "policy_offset_drift"
    case batchNormRunningVarianceRunaway = "bn_running_variance_runaway"
    case gradientSpike = "gradient_spike"

    /// Position in rule order (the declaration order of `allCases`).
    var ruleOrder: Int {
        switch self {
        case .nonFinite: return 0
        case .deadChannels: return 1
        case .valueFC1ZeroVelocity: return 2
        case .illegalMass: return 3
        case .gradientCollapse: return 4
        case .lossSpike: return 5
        case .policyOffsetDrift: return 6
        case .batchNormRunningVarianceRunaway: return 7
        case .gradientSpike: return 8
        }
    }

    /// Whether the rule ever raises at critical. Rules 6–9 are warning
    /// only, so `stop_on_critical` can never stop on them.
    var hasCriticalLevel: Bool {
        switch self {
        case .nonFinite, .deadChannels, .valueFC1ZeroVelocity, .illegalMass, .gradientCollapse:
            return true
        case .lossSpike, .policyOffsetDrift, .batchNormRunningVarianceRunaway, .gradientSpike:
            return false
        }
    }

    /// Whether the rule ever raises at warning. Rules 1, 4 and 5 are
    /// critical only.
    var hasWarningLevel: Bool {
        switch self {
        case .nonFinite, .illegalMass, .gradientCollapse:
            return false
        case .deadChannels, .valueFC1ZeroVelocity, .lossSpike, .policyOffsetDrift,
             .batchNormRunningVarianceRunaway, .gradientSpike:
            return true
        }
    }

    /// How long the raise condition must hold (Part R's "Sustain" column).
    var raiseSustain: TrainingHealthSustain {
        switch self {
        case .illegalMass, .gradientCollapse, .policyOffsetDrift:
            return TrainingHealthThresholds.windowRuleSustain
        case .nonFinite, .deadChannels, .valueFC1ZeroVelocity, .lossSpike,
             .batchNormRunningVarianceRunaway, .gradientSpike:
            return .immediate
        }
    }

    /// How long the clear condition must hold, or nil when the rule never
    /// clears (non-finite weights do not heal). Clear has its own sustain,
    /// separate from the raise's (hysteresis, Part R0).
    var clearSustain: TrainingHealthSustain? {
        switch self {
        case .nonFinite:
            return nil
        case .illegalMass, .gradientCollapse, .policyOffsetDrift:
            return TrainingHealthThresholds.windowRuleSustain
        case .deadChannels, .batchNormRunningVarianceRunaway:
            return TrainingHealthThresholds.layerHealthRuleClearSustain
        case .valueFC1ZeroVelocity, .lossSpike, .gradientSpike:
            return .immediate
        }
    }
}

/// What a rule does besides logging when it is active. The raw value is
/// the persisted parameter value (`training_health_action_<rule>`); JSON
/// writes the name. Public because the `TrainingParameters` singleton (a
/// public class) stores one per rule as this enum.
public enum TrainingHealthAction: Int, CaseIterable, Codable, Sendable {
    /// Log, record and show only — the default for every rule.
    case log = 0
    /// Additionally stop the run while the rule is active at critical.
    case stopOnCritical = 1
    /// Additionally stop the run while the rule is active at any severity.
    case stopOnAny = 2

    /// The name written in log lines and JSON.
    var name: String {
        switch self {
        case .log: return "log"
        case .stopOnCritical: return "stop_on_critical"
        case .stopOnAny: return "stop_on_any"
        }
    }

    init(name: String) throws {
        guard let action = Self.allCases.first(where: { $0.name == name }) else {
            throw TrainingHealthError.unknownActionName(name)
        }
        self = action
    }

    /// Closed range of raw values this enum covers, for pinning against the
    /// action parameters' declared range.
    static var parameterRawValueRange: ClosedRange<Int> {
        let raws = allCases.map(\.rawValue)
        guard let low = raws.min(), let high = raws.max() else {
            preconditionFailure("TrainingHealthAction must have at least one case")
        }
        return low...high
    }

    /// Converts a persisted raw value. Every path that reaches this has
    /// checked the value against the parameter's declared range (pinned to
    /// `parameterRawValueRange` by test), so an unrepresentable value is a
    /// programmer error and traps rather than silently picking an action.
    init(persistedRawValue raw: Int) {
        guard let action = TrainingHealthAction(rawValue: raw) else {
            preconditionFailure(
                "training_health_action raw value \(raw) has no TrainingHealthAction case; "
                + "the parameters' declared range and \(TrainingHealthAction.self) have drifted apart")
        }
        self = action
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        self = try TrainingHealthAction(name: try container.decode(String.self))
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(name)
    }
}

/// `TrainingAlarm.Severity` is the one severity type; the health events
/// encode it by its raw value (`"warning"` / `"critical"`).
extension TrainingAlarm.Severity: Codable {}

extension TrainingAlarm.Severity {
    /// warning < critical. A critical alarm never de-escalates.
    var healthRank: Int {
        switch self {
        case .warning: return 0
        case .critical: return 1
        }
    }
}

/// One action per rule, total by construction: every rule has a stored
/// field, so a lookup can never miss.
struct TrainingHealthActions: Sendable, Equatable, Encodable {
    var nonFinite: TrainingHealthAction
    var deadChannels: TrainingHealthAction
    var valueFC1ZeroVelocity: TrainingHealthAction
    var illegalMass: TrainingHealthAction
    var gradientCollapse: TrainingHealthAction
    var lossSpike: TrainingHealthAction
    var policyOffsetDrift: TrainingHealthAction
    var batchNormRunningVarianceRunaway: TrainingHealthAction
    var gradientSpike: TrainingHealthAction

    /// Build every rule's action from one function of the rule.
    init(_ actionForRule: (TrainingHealthRule) -> TrainingHealthAction) {
        nonFinite = actionForRule(.nonFinite)
        deadChannels = actionForRule(.deadChannels)
        valueFC1ZeroVelocity = actionForRule(.valueFC1ZeroVelocity)
        illegalMass = actionForRule(.illegalMass)
        gradientCollapse = actionForRule(.gradientCollapse)
        lossSpike = actionForRule(.lossSpike)
        policyOffsetDrift = actionForRule(.policyOffsetDrift)
        batchNormRunningVarianceRunaway = actionForRule(.batchNormRunningVarianceRunaway)
        gradientSpike = actionForRule(.gradientSpike)
    }

    subscript(rule: TrainingHealthRule) -> TrainingHealthAction {
        get {
            switch rule {
            case .nonFinite: return nonFinite
            case .deadChannels: return deadChannels
            case .valueFC1ZeroVelocity: return valueFC1ZeroVelocity
            case .illegalMass: return illegalMass
            case .gradientCollapse: return gradientCollapse
            case .lossSpike: return lossSpike
            case .policyOffsetDrift: return policyOffsetDrift
            case .batchNormRunningVarianceRunaway: return batchNormRunningVarianceRunaway
            case .gradientSpike: return gradientSpike
            }
        }
        set {
            switch rule {
            case .nonFinite: nonFinite = newValue
            case .deadChannels: deadChannels = newValue
            case .valueFC1ZeroVelocity: valueFC1ZeroVelocity = newValue
            case .illegalMass: illegalMass = newValue
            case .gradientCollapse: gradientCollapse = newValue
            case .lossSpike: lossSpike = newValue
            case .policyOffsetDrift: policyOffsetDrift = newValue
            case .batchNormRunningVarianceRunaway: batchNormRunningVarianceRunaway = newValue
            case .gradientSpike: gradientSpike = newValue
            }
        }
    }

    /// `{rule id: action name}`, the shape results.json's `alarm_config`
    /// carries.
    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: RuleKey.self)
        for rule in TrainingHealthRule.allCases {
            try container.encode(self[rule], forKey: RuleKey(rule))
        }
    }

    private struct RuleKey: CodingKey {
        let stringValue: String
        var intValue: Int? { nil }
        init(_ rule: TrainingHealthRule) { stringValue = rule.rawValue }
        init?(stringValue: String) { self.stringValue = stringValue }
        init?(intValue: Int) { nil }
    }
}

/// The resolved settings one evaluation runs under. P2 adds the resolution
/// from a `TrainingParametersSnapshot`, in one place for every path; until
/// then it is built from its fields. Encoded as results.json's
/// `alarm_config`.
struct TrainingHealthConfig: Sendable, Equatable, Encodable {
    let enabled: Bool
    /// Cadence of the `[HEALTH] check` line, the `active` reminders and the
    /// `worsen` rate limit, in trainer steps.
    let checkIntervalSteps: Int
    /// Added to `lrWarmupSteps` for rule 4's not-learned form.
    let learningGraceSteps: Int
    let lrWarmupSteps: Int
    /// The run's base momentum μ, recorded for rule 3's reading (exact-zero
    /// velocity lags death by a μ-dependent number of steps); never a
    /// condition.
    let momentumCoefficient: Double
    let actions: TrainingHealthActions

    /// Refuses values no evaluation can run under (a check interval below 1
    /// would divide by zero; negative step counts mean nothing).
    init(
        enabled: Bool,
        checkIntervalSteps: Int,
        learningGraceSteps: Int,
        lrWarmupSteps: Int,
        momentumCoefficient: Double,
        actions: TrainingHealthActions
    ) throws {
        guard checkIntervalSteps >= 1 else {
            throw TrainingHealthError.invalidConfig("checkIntervalSteps must be at least 1, got \(checkIntervalSteps)")
        }
        guard learningGraceSteps >= 0 else {
            throw TrainingHealthError.invalidConfig("learningGraceSteps must not be negative, got \(learningGraceSteps)")
        }
        guard lrWarmupSteps >= 0 else {
            throw TrainingHealthError.invalidConfig("lrWarmupSteps must not be negative, got \(lrWarmupSteps)")
        }
        guard momentumCoefficient.isFinite else {
            throw TrainingHealthError.invalidConfig("momentumCoefficient must be finite, got \(momentumCoefficient)")
        }
        self.enabled = enabled
        self.checkIntervalSteps = checkIntervalSteps
        self.learningGraceSteps = learningGraceSteps
        self.lrWarmupSteps = lrWarmupSteps
        self.momentumCoefficient = momentumCoefficient
        self.actions = actions
    }

    /// The one resolution from the parameters, shared by every path: the
    /// command-line runners resolve it once from their run-start snapshot,
    /// the GUI at every evaluation from `TrainingParameters.shared
    /// .snapshot()`, so a live edit applies at the next evaluation. The
    /// declared ranges already exclude every value the memberwise
    /// initializer refuses, so a throw here means a snapshot bypassed
    /// validation — surfaced, never defaulted.
    init(_ snapshot: TrainingParametersSnapshot) throws {
        try self.init(
            enabled: snapshot.trainingHealthAlarmsEnabled,
            checkIntervalSteps: snapshot.trainingHealthCheckIntervalSteps,
            learningGraceSteps: snapshot.trainingHealthLearningGraceSteps,
            lrWarmupSteps: snapshot.lrWarmupSteps,
            momentumCoefficient: snapshot.momentumCoeff,
            actions: TrainingHealthActions { snapshot.trainingHealthAction(for: $0) })
    }

    /// Trainer step from which rule 4's not-learned form applies.
    var learningGateTrainerStep: Int { lrWarmupSteps + learningGraceSteps }

    enum CodingKeys: String, CodingKey {
        case enabled
        case checkIntervalSteps = "check_interval_steps"
        case learningGraceSteps = "learning_grace_steps"
        case lrWarmupSteps = "lr_warmup_steps"
        case momentumCoefficient = "momentum_coefficient"
        case actions
    }
}

/// How long a condition must hold: on `evaluations` consecutive
/// evaluations that have data, spanning at least `spanSteps` trainer steps
/// (first to last). Measuring both keeps a rule independent of the
/// evaluation cadence (25 steps, 50 steps, one 60 s interval).
struct TrainingHealthSustain: Sendable, Equatable {
    let evaluations: Int
    let spanSteps: Int

    static let immediate = TrainingHealthSustain(evaluations: 1, spanSteps: 0)
}

/// The thresholds — the one declaration (owner decisions OD-1, OD-14).
/// Every number here was measured against the incident and healthy runs in
/// the plan's Evidence section; change one only with a new owner decision.
enum TrainingHealthThresholds {
    /// Sustain of the window rules (4, 5, 7), for both raise and clear.
    static let windowRuleSustain = TrainingHealthSustain(evaluations: 2, spanSteps: 50)

    /// Clear sustain of the layer-health rules 2 and 8: two evaluations that
    /// span at least one trainer step. The span is what makes it two observed
    /// states rather than two reads of one: at every save step the live
    /// evaluation and the save's checkpoint evaluation run at the same trainer
    /// step and read the same batch-norm γ, β and running statistics, so
    /// without it a single recovered state would clear the alarm and the
    /// next damaged window would raise it again (raise → clear → raise). A
    /// span rather than "count only steps above the streak's last" because
    /// it is the same measure every other sustain uses (`isSustained`), and
    /// with non-decreasing steps (the freshness check) the two are
    /// equivalent: two evaluations spanning ≥ 1 step are exactly two
    /// distinct trainer steps. One step is enough — any SGD step changes the
    /// weights — so on the 50-step cadence a clear still comes at the live
    /// evaluation after the first recovered one.
    static let layerHealthRuleClearSustain = TrainingHealthSustain(evaluations: 2, spanSteps: 1)

    /// Live evaluations run every this many trainer steps on every path, on
    /// overall trainer-step multiples, independent of when the paths write
    /// their step lines (owner decision 2026-10-06).
    static let liveEvaluationIntervalSteps = 50

    // Rule 2 — dead_channels. Counts parked channels (`BatchNormPassThrough`)
    // over every BN site an activation consumes — for relu / leaky_relu
    // exactly the dead channels, for silu / gelu the activation-aware
    // equivalent (owner decision 2026-10-06). Warning on any parked channel
    // (absolute form, OD-16); critical at this fraction of all such
    // channels, or of any one site's channels.
    static let deadChannelsCriticalOverallFraction: Double = 0.05
    static let deadChannelsCriticalSiteFraction: Double = 0.2

    // Rule 3 — value_fc1_zero_velocity (fraction of value FC1 units whose
    // weight velocity is exactly zero).
    static let valueFC1ZeroVelocityWarningFraction: Double = 0.05
    static let valueFC1ZeroVelocityCriticalFraction: Double = 0.5
    static let valueFC1ZeroVelocityClearFraction: Double = 0.025
    /// Rule 3 has data only once this process has trained this many steps
    /// (when the state was read): velocity that has not accumulated reads as
    /// exactly zero.
    static let valueFC1MinimumStepsTrainedByThisProcess = 200
    /// Two consecutive rule-3 observations are never more than this many
    /// trainer steps apart on any path (OD-15, D6).
    static let valueFC1CheckIntervalSteps = 1000

    // Rule 4 — illegal_mass (window median of the illegal-mass penalty).
    // Two regression arms (either raises; owner decision 2026-10-06 keeps
    // both): (i) the absolute arm caught arm C (minimum 0.236, then 0.9971
    // and 0.9458); (ii) the relative arm caught B-silu (minimum 0.0023, then
    // 0.7988 and 0.7817), which (i) misses because it never reached 0.8.
    /// Arm (i): the run's running minimum of window medians has been below
    /// this …
    static let illegalMassRegressionLearnedBelow: Double = 0.5
    /// … and the window median is now at or above this.
    static let illegalMassRegressionRaise: Double = 0.8
    /// Arm (ii): the window median is at or above this floor …
    static let illegalMassRelativeRegressionFloor: Double = 0.3
    /// … and at or above this multiple of the running minimum.
    static let illegalMassRelativeRegressionMultiple: Double = 10
    /// Not-learned form, past the learning gate.
    static let illegalMassNotLearnedRaise: Double = 0.5
    /// Clear below this: half of arm (ii)'s floor, so a cleared alarm sits
    /// clearly below the lowest level that raises it (the same 2:1
    /// hysteresis as gradient_collapse).
    static let illegalMassClear: Double = 0.15

    // Rule 5 — gradient_collapse (window median of the pre-clip gradient norm).
    static let gradientCollapseRaise: Double = 0.1
    static let gradientCollapseClear: Double = 0.2

    // Rule 6 — loss_spike, against the median loss of the reference span.
    static let lossSpikeMedianRatioRaise: Double = 1.5
    static let lossSpikeMaxRatioRaise: Double = 3.0
    static let lossSpikeMedianRatioClear: Double = 1.2

    // Rule 9 — gradient_spike: the window's largest pre-clip gradient norm
    // against the median gradient norm of the reference span (owner
    // decision 2026-10-06). Healthy runs' largest logged ratio: 1.40.
    static let gradientSpikeMaxRatioRaise: Double = 5.0
    static let gradientSpikeMaxRatioClear: Double = 2.5

    /// Rules 6 and 9: the reference is the records in this many trainer
    /// steps before the window …
    static let spikeReferenceLookbackSteps = 1000
    /// … and has data only when they span at least this many trainer steps
    /// (first to last) …
    static let spikeReferenceMinimumSpanSteps = 200
    /// … and number at least this many.
    static let spikeReferenceMinimumRecords = 5
    /// Entries kept in the reference history (one per recorded step in-app,
    /// so it always covers the look-back).
    static let spikeReferenceHistoryCapacity = 1024

    // Rule 7 — policy_offset_drift (window median of |policy logit mean|).
    static let policyOffsetDriftRaise: Double = 3.0
    static let policyOffsetDriftClear: Double = 2.0

    // Rule 8 — bn_running_variance_runaway (largest BN running-variance
    // max/median).
    static let batchNormRunningVarianceRunawayRaise: Double = 1000
    static let batchNormRunningVarianceRunawayClear: Double = 300

    /// Records the pending window holds before it drops the oldest (and
    /// counts them, `truncated=` on the `[HEALTH] check` line). Guards
    /// against evaluations that never come.
    static let pendingWindowCapacity = 65_536
}

/// When a path runs a live evaluation: every
/// `TrainingHealthThresholds.liveEvaluationIntervalSteps` trainer steps, on
/// overall trainer-step multiples, so a resumed run evaluates on the steps
/// the uninterrupted run would have, and independent of the step-line
/// cadence (owner decision 2026-10-06). The paths wire it in P2 / P3.
enum TrainingHealthCadence {
    static func isLiveEvaluationStep(trainerStep: Int) -> Bool {
        trainerStep > 0 && trainerStep % TrainingHealthThresholds.liveEvaluationIntervalSteps == 0
    }
}

/// Whether rule 3 (`value_fc1_zero_velocity`) means anything for this
/// model. Exact-zero weight velocity marks a dead ReLU unit; with a leaky
/// or smooth activation the gradient is almost never exactly zero, so the
/// rule would never fire and would mean nothing.
enum TrainingHealthValueFC1Applicability: Sendable, Equatable {
    /// The value FC1 layer is ReLU: the rule runs and D6 schedules reads.
    case applies
    /// The value FC1 layer uses another function: never raised, no reads.
    case doesNotApply(activation: ActivationFunction)
    /// Offline replay only: the logs never name the layer's activation
    /// (no checkpoint table), so the rule has no data.
    case unknownFromLog

    /// From the activation `LayerHealth.valueFC1Layer(for:)` reports.
    init(valueFC1Activation: ActivationFunction) {
        self = valueFC1Activation == .relu ? .applies : .doesNotApply(activation: valueFC1Activation)
    }
}

// MARK: - Inputs

/// One SGD step, as the monitor records it. Lean fields are valid on every
/// step in-app; `policyLogitMean` only on diagnostic steps (the trainer
/// computes it only then). nil means "not measured" — offline log rows can
/// lack a lean field too (`[VS-UCI]` rows carry no `pIllM`) — never 0.
struct TrainingHealthStepRecord: Sendable, Equatable {
    let trainerStep: Int
    let loss: Float?
    let illegalMassPenalty: Float?
    let gradGlobalNorm: Float?
    /// The step's own `trainStep` wall time, summed into `train_ms=`.
    let totalMs: Double?
    /// Present on a diagnostic step; may be non-finite (rule 1 reads that).
    let policyLogitMean: Float?

    init(timing: TrainStepTiming, trainerStep: Int) {
        self.trainerStep = trainerStep
        self.loss = timing.loss
        self.illegalMassPenalty = timing.illegalMassPenalty
        self.gradGlobalNorm = timing.gradGlobalNorm
        self.totalMs = timing.totalMs
        self.policyLogitMean = timing.hasDiagnostics ? timing.policyLogitMean : nil
    }

    init(
        trainerStep: Int,
        loss: Float?,
        illegalMassPenalty: Float?,
        gradGlobalNorm: Float?,
        totalMs: Double?,
        policyLogitMean: Float?
    ) {
        self.trainerStep = trainerStep
        self.loss = loss
        self.illegalMassPenalty = illegalMassPenalty
        self.gradGlobalNorm = gradGlobalNorm
        self.totalMs = totalMs
        self.policyLogitMean = policyLogitMean
    }
}

/// The layer-health facts the rules read, built in one place from a
/// `LayerHealthSummary` (in-app) or from parsed `[LAYER-HEALTH]` lines
/// (offline replay). A nil field means the source did not carry it, so the
/// rules that read it have no data — never "healthy".
struct LayerHealthDigest: Sendable, Equatable {
    /// Which read produced the digest. It decides which rules the digest is
    /// meant to feed, so a rule it never feeds is not counted as "no data"
    /// on the `[HEALTH] check` line (only a source that should have carried
    /// a rule's input and did not is).
    enum Tier: String, Sendable {
        /// The live read: BN state and ReZero α (rules 1, 2, 8).
        case live
        /// A save's full-tensor pass (rules 1, 2, 3, 8).
        case checkpoint
        /// The dedicated value-FC1 velocity read (D6): rule 3 only.
        case valueFC1Read = "value-fc1"

        /// Whether a digest from this read is meant to carry `rule`'s input.
        /// The window rules (4–7, 9) come from step records, never from a
        /// digest.
        func feeds(_ rule: TrainingHealthRule) -> Bool {
            switch rule {
            case .nonFinite, .deadChannels, .batchNormRunningVarianceRunaway:
                return self != .valueFC1Read
            case .valueFC1ZeroVelocity:
                return self != .live
            case .illegalMass, .gradientCollapse, .lossSpike, .policyOffsetDrift, .gradientSpike:
                return false
            }
        }
    }

    /// One BN site an activation consumes, with its parked channels
    /// (`BatchNormPassThrough`; for relu / leaky_relu exactly the dead
    /// channels). Either count is nil when unknown (offline: a site's
    /// channel count comes only from a checkpoint table or a `parkedBy`
    /// entry); the per-site arm of rule 2 then skips that site, so offline it
    /// is a lower bound.
    struct SiteDeadChannels: Sendable, Equatable {
        let site: String
        let parkedChannelCount: Int?
        let channelCount: Int?
    }

    /// Rule 2's input: parked channels over every site an activation
    /// consumes, whatever the function (owner decision 2026-10-06; for
    /// relu / leaky_relu sites parked is dead). "Modeled" is a site or
    /// channel with a pass-through model (`BatchNormPassThrough.isModeled`) —
    /// every BN output an activation consumes — not `LayerHealthSummary`'s
    /// relu / leaky_relu-only "classified" sites.
    struct DeadChannels: Sendable, Equatable {
        /// Sites an activation consumes; 0 means none, so rule 2 does not
        /// apply. nil when unknown (offline: a log line written before the
        /// parked counts, in a run with no checkpoint table).
        let modeledSiteCount: Int?
        /// Every channel of those sites: the overall critical arm's
        /// denominator (OD-18). nil when unknown, and then the overall arm has
        /// no data — it is never computed over a partial count, which would
        /// overstate the fraction.
        let modeledChannelCount: Int?
        let parkedChannelCount: Int
        /// The sites whose counts are known (in-app: every such site;
        /// offline: the sites the line names).
        let sites: [SiteDeadChannels]
        /// False when only relu / leaky_relu sites were counted — an offline
        /// log written before the parked counts existed, whose silu / gelu
        /// sites were never checked. `parkedChannelCount` is then a lower
        /// bound (over a full `modeledChannelCount`, so the overall fraction
        /// is one too).
        let coversEveryActivation: Bool
    }

    struct RunningVarianceRunaway: Sendable, Equatable {
        let maxOverMedian: Double
        let site: String
    }

    struct ValueFC1Velocity: Sendable, Equatable {
        let zeroVelocityUnitCount: Int
        let unitCount: Int
    }

    let tier: Tier
    let deadChannels: DeadChannels?
    let nonFiniteValueCount: Int?
    let runningVariance: RunningVarianceRunaway?
    let valueFC1: ValueFC1Velocity?

    init(
        tier: Tier,
        deadChannels: DeadChannels?,
        nonFiniteValueCount: Int?,
        runningVariance: RunningVarianceRunaway?,
        valueFC1: ValueFC1Velocity?
    ) {
        self.tier = tier
        self.deadChannels = deadChannels
        self.nonFiniteValueCount = nonFiniteValueCount
        self.runningVariance = runningVariance
        self.valueFC1 = valueFC1
    }

    /// The in-app source: a live summary (BN state only) is the live tier, a
    /// full-tensor summary the checkpoint tier. Rule 2 reads the
    /// activation-aware parked counts of every site an activation consumes.
    init(summary: LayerHealthSummary) {
        switch summary.scope {
        case .batchNormStateOnly: tier = .live
        case .allTensors: tier = .checkpoint
        }
        let sites = summary.passThroughSites
        deadChannels = DeadChannels(
            modeledSiteCount: sites.count,
            modeledChannelCount: summary.passThroughChannelCount,
            parkedChannelCount: summary.parkedChannelCount,
            sites: sites.map {
                SiteDeadChannels(site: $0.site, parkedChannelCount: $0.parkedChannelCount, channelCount: $0.channelCount)
            },
            coversEveryActivation: true)
        nonFiniteValueCount = summary.nonFiniteValueCount
        if let site = summary.worstRunningVarianceSite, let ratio = site.runningVarianceMaxOverMedian {
            runningVariance = RunningVarianceRunaway(maxOverMedian: ratio, site: site.site)
        } else {
            runningVariance = nil
        }
        valueFC1 = summary.valueFC1.map {
            ValueFC1Velocity(zeroVelocityUnitCount: $0.zeroVelocityUnitCount, unitCount: $0.unitCount)
        }
    }

    /// A digest carrying only a value-FC1 velocity reading (the dedicated
    /// read, D6). It feeds rule 3 alone, by design, so the other rules are
    /// neither applied nor counted as "no data" on it.
    static func valueFC1Only(_ valueFC1: ValueFC1Velocity) -> LayerHealthDigest {
        LayerHealthDigest(
            tier: .valueFC1Read, deadChannels: nil, nonFiniteValueCount: nil,
            runningVariance: nil, valueFC1: valueFC1)
    }
}

/// Identifies the monitor state an observation was read against, taken
/// before its data was read (D2). An evaluation whose stamp's generation is
/// not the monitor's current one describes weights a trainer-clock rewind
/// discarded, and is ignored.
struct TrainingHealthStamp: Sendable, Equatable {
    let runID: UUID
    let generation: Int
    /// Rule 3's gate: steps this process trained (less any rewound span)
    /// when the state was read — never the trainer clock.
    let stepsTrainedByThisProcess: Int
    let lastRecordedTrainerStep: Int?
}

/// One step's entry in the reference history the spike rules (6 and 9)
/// compare a window against.
struct TrainingHealthReferenceEntry: Sendable, Equatable {
    let trainerStep: Int
    let loss: Float?
    let gradGlobalNorm: Float?
}

/// The median of one field over the records in the look-back span before a
/// window.
struct TrainingHealthReference: Sendable, Equatable {
    let median: Double
    let recordCount: Int
    let spanSteps: Int

    /// The reference of `values` (trainer step, value) in the look-back span
    /// before `windowStart`: nil unless at least
    /// `spikeReferenceMinimumRecords` finite values span at least
    /// `spikeReferenceMinimumSpanSteps` trainer steps (first to last) — a
    /// span, not a record count, so per-step history and sparse offline rows
    /// share one definition.
    static func make(
        _ values: [(trainerStep: Int, value: Float?)],
        windowStart: Int
    ) -> TrainingHealthReference? {
        let lowerBound = windowStart - TrainingHealthThresholds.spikeReferenceLookbackSteps
        var steps: [Int] = []
        var finite: [Double] = []
        for entry in values where entry.trainerStep >= lowerBound && entry.trainerStep < windowStart {
            guard let value = entry.value, value.isFinite else { continue }
            steps.append(entry.trainerStep)
            finite.append(Double(value))
        }
        guard finite.count >= TrainingHealthThresholds.spikeReferenceMinimumRecords,
              let lo = steps.min(), let hi = steps.max(),
              hi - lo >= TrainingHealthThresholds.spikeReferenceMinimumSpanSteps,
              let median = TrainingHealthWindowStatistics.median(finite) else {
            return nil
        }
        return TrainingHealthReference(median: median, recordCount: finite.count, spanSteps: hi - lo)
    }
}

/// The window statistics of Part R0, computed by the monitor; the evaluator
/// never sees raw records.
struct TrainingHealthWindowStatistics: Sendable, Equatable {
    let recordCount: Int
    let firstTrainerStep: Int?
    let lastTrainerStep: Int?
    let lossMedian: Double?
    let lossMax: Double?
    let illegalMassMedian: Double?
    let gradientNormMedian: Double?
    let gradientNormMax: Double?
    let diagnosticRecordCount: Int
    /// Median of |policyLogitMean| over the diagnostic records with a
    /// finite value.
    let policyLogitMeanAbsMedian: Double?
    /// Non-finite values among the window's measured fields (rule 1).
    let nonFiniteValueCount: Int
    /// Rule 6's reference: median loss before the window.
    let lossReference: TrainingHealthReference?
    /// Rule 9's reference: median gradient norm before the window.
    let gradientNormReference: TrainingHealthReference?

    /// Statistics of `records` (any order) against the reference history
    /// (entries recorded before the window).
    static func make(
        records: [TrainingHealthStepRecord],
        history: [TrainingHealthReferenceEntry]
    ) -> TrainingHealthWindowStatistics {
        var losses: [Double] = []
        var illegal: [Double] = []
        var gradients: [Double] = []
        var offsets: [Double] = []
        var diagnostic = 0
        var nonFinite = 0
        var first: Int?
        var last: Int?
        func take(_ value: Float?, into list: inout [Double]) {
            guard let value else { return }
            if value.isFinite {
                list.append(Double(value))
            } else {
                nonFinite += 1
            }
        }
        for record in records {
            first = min(first ?? record.trainerStep, record.trainerStep)
            last = max(last ?? record.trainerStep, record.trainerStep)
            take(record.loss, into: &losses)
            take(record.illegalMassPenalty, into: &illegal)
            take(record.gradGlobalNorm, into: &gradients)
            if let offset = record.policyLogitMean {
                diagnostic += 1
                if offset.isFinite {
                    offsets.append(abs(Double(offset)))
                } else {
                    nonFinite += 1
                }
            }
        }
        var lossReference: TrainingHealthReference?
        var gradientReference: TrainingHealthReference?
        if let first {
            lossReference = TrainingHealthReference.make(
                history.map { ($0.trainerStep, $0.loss) }, windowStart: first)
            gradientReference = TrainingHealthReference.make(
                history.map { ($0.trainerStep, $0.gradGlobalNorm) }, windowStart: first)
        }
        return TrainingHealthWindowStatistics(
            recordCount: records.count,
            firstTrainerStep: first,
            lastTrainerStep: last,
            lossMedian: median(losses),
            lossMax: losses.max(),
            illegalMassMedian: median(illegal),
            gradientNormMedian: median(gradients),
            gradientNormMax: gradients.max(),
            diagnosticRecordCount: diagnostic,
            policyLogitMeanAbsMedian: median(offsets),
            nonFiniteValueCount: nonFinite,
            lossReference: lossReference,
            gradientNormReference: gradientReference)
    }

    /// Median; the mean of the two middle values for an even count. nil
    /// for no values.
    static func median(_ values: [Double]) -> Double? {
        guard !values.isEmpty else { return nil }
        let sorted = values.sorted()
        let mid = sorted.count / 2
        return sorted.count % 2 == 1 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2
    }
}

/// One evaluation's input. A live observation carries a window (possibly
/// empty) and, when the live read succeeded, the live digest; a checkpoint
/// observation carries only its digest and never consumes the window.
struct TrainingHealthObservation: Sendable {
    let trainerStep: Int
    /// nil for a checkpoint-only evaluation.
    let window: TrainingHealthWindowStatistics?
    let layerHealth: LayerHealthDigest?
    let stamp: TrainingHealthStamp
    /// Recorded on event lines, never a condition.
    let effectiveLearningRate: Double?
    /// Recorded on event lines (`mom=`), never a condition; nil for a
    /// checkpoint evaluation.
    let effectiveMomentum: Double?

    var isLive: Bool { window != nil }
}

// MARK: - Outputs

struct TrainingHealthEvent: Encodable, Sendable, Equatable {
    enum Kind: String, Codable, Sendable, CaseIterable {
        case raise, escalate, worsen, active, clear, stop

        /// Position in the deterministic event order (after rule order).
        var kindOrder: Int {
            switch self {
            case .raise: return 0
            case .escalate: return 1
            case .worsen: return 2
            case .active: return 3
            case .clear: return 4
            case .stop: return 5
            }
        }
    }

    let kind: Kind
    let rule: TrainingHealthRule
    let severity: TrainingAlarm.Severity
    let trainerStep: Int
    /// Trainer step the alarm was raised.
    let since: Int?
    /// The measured value, rendered (`dead=5/1040`).
    let value: String
    /// The crossed threshold, rendered (`site>=0.2`); empty when none was
    /// crossed (worsen, active, clear, stop).
    let threshold: String
    /// Extra `key=value` fields (`sites=value.bn(5/16)`, `was=5`); may be
    /// empty.
    let detail: String
    let action: TrainingHealthAction
    let learningRate: Double?
    let momentum: Double?

    enum CodingKeys: String, CodingKey {
        case kind, rule, severity
        case trainerStep = "trainer_step"
        case since, value, threshold, detail, action
        case learningRate = "learning_rate"
        case momentum
    }
}

struct TrainingHealthActiveAlarm: Sendable, Equatable {
    let rule: TrainingHealthRule
    let severity: TrainingAlarm.Severity
    /// Trainer step it was raised.
    let since: Int
    /// Latest measured value, rendered.
    let value: String
    let detail: String
    let action: TrainingHealthAction
}

struct TrainingHealthEvaluation: Sendable {
    /// Deterministic order: rule order, then kind.
    let events: [TrainingHealthEvent]
    /// The first qualifying active alarm (R2), once per evaluator.
    let stopRequest: TrainingHealthEvent?
    let active: [TrainingHealthActiveAlarm]
    /// Rules whose inputs had no data in this observation although its
    /// source is meant to carry them (state held). A rule the source never
    /// carries is not listed (`TrainingHealthEvaluator.observation(_:feeds:)`).
    let noDataRules: [TrainingHealthRule]
    /// Rules whose observation was older than the newest one they applied
    /// (not applied to them; counted `stale=`).
    let staleRules: [TrainingHealthRule]
    /// Rules found not to apply for the first time in this evaluator, with
    /// the reason (logged once).
    let newlyNotApplicable: [(rule: TrainingHealthRule, reason: String)]
    /// True on the first live evaluation at or after each multiple of the
    /// check interval: the `[HEALTH] check` line is due (reminders are in
    /// `events`).
    let checkDue: Bool
    /// The window this evaluation judged (nil for a checkpoint evaluation).
    let window: TrainingHealthWindowStatistics?
}

/// Who turns an active alarm into a stop (R2). The decision itself is always
/// `TrainingHealthStopPolicy.firstQualifying`; this says where it runs.
enum TrainingHealthStopDecision: Sendable, Equatable {
    /// At the end of every evaluation, with that evaluation's config: the
    /// command-line paths and the offline replay, whose actions never
    /// change during a run. The evaluation's `stopRequest` (and its `stop`
    /// event line) is the stop.
    case byEvaluator
    /// By the caller after each evaluation is delivered, with the actions in
    /// force at that moment: the GUI, whose actions are live-tunable and
    /// whose detached checkpoint passes can finish after an action changed
    /// (R2, R3). The evaluator then never requests a stop nor writes a
    /// `stop` line; the caller writes it when it acts.
    case byCaller
}

/// R2: one pure decision, shared by every path.
enum TrainingHealthStopPolicy {
    /// The first active alarm, in rule order, whose severity qualifies under
    /// its rule's action — or nil.
    static func firstQualifying(
        active: [TrainingHealthActiveAlarm],
        actions: TrainingHealthActions
    ) -> TrainingHealthActiveAlarm? {
        active
            .sorted { $0.rule.ruleOrder < $1.rule.ruleOrder }
            .first { qualifies(severity: $0.severity, action: actions[$0.rule]) }
    }

    static func qualifies(severity: TrainingAlarm.Severity, action: TrainingHealthAction) -> Bool {
        switch action {
        case .log: return false
        case .stopOnCritical: return severity == .critical
        case .stopOnAny: return true
        }
    }
}

// MARK: - The evaluator

/// The nine rules as a value type: per-rule sustain counters, hysteresis,
/// the active set and the stop flag. The monitor runs every transition on a
/// copy and commits it only if no trainer-clock rewind intervened (D2).
struct TrainingHealthEvaluator: Sendable {

    struct Streak: Sendable, Equatable {
        var count: Int
        let firstTrainerStep: Int
    }

    struct ActiveState: Sendable, Equatable {
        var severity: TrainingAlarm.Severity
        let since: Int
        var value: String
        var detail: String
        var action: TrainingHealthAction
        /// Largest measured count seen while active (rules that count).
        var peakCount: Int?
    }

    struct RuleState: Sendable, Equatable {
        var active: ActiveState?
        var raiseStreak: Streak?
        var clearStreak: Streak?
        /// Newest trainer step whose observation this rule applied (D2's
        /// freshness across tiers).
        var newestAppliedTrainerStep: Int?
        /// Check-interval bucket of the last `worsen` line (rate limit).
        var lastWorsenBucket: Int?
        var notApplicableReported = false
    }

    struct State: Sendable, Equatable {
        /// One state per rule, indexed by `TrainingHealthRule.ruleOrder`
        /// (total by construction).
        var rules: [RuleState]
        /// Rule 4's regression form: the running minimum of window medians.
        var illegalMassRunningMinimum: Double?
        /// Check-interval bucket of the last `[HEALTH] check` (nil before the
        /// first live evaluation).
        var lastCheckBucket: Int?
        var stopRequested = false
    }

    let valueFC1Applicability: TrainingHealthValueFC1Applicability
    let stopDecision: TrainingHealthStopDecision
    private(set) var state: State

    /// An evaluator that decides stops itself (`.byEvaluator`): the
    /// command-line paths and the offline replay, whose actions never change
    /// during a run.
    init(valueFC1Applicability: TrainingHealthValueFC1Applicability) {
        self.init(valueFC1Applicability: valueFC1Applicability, stopDecision: .byEvaluator)
    }

    init(valueFC1Applicability: TrainingHealthValueFC1Applicability, stopDecision: TrainingHealthStopDecision) {
        self.valueFC1Applicability = valueFC1Applicability
        self.stopDecision = stopDecision
        state = State(
            rules: Array(repeating: RuleState(), count: TrainingHealthRule.allCases.count),
            illegalMassRunningMinimum: nil, lastCheckBucket: nil)
    }

    /// The active alarms, in rule order.
    var activeAlarms: [TrainingHealthActiveAlarm] {
        TrainingHealthRule.allCases.compactMap { rule in
            guard let active = state.rules[rule.ruleOrder].active else { return nil }
            return TrainingHealthActiveAlarm(
                rule: rule, severity: active.severity, since: active.since,
                value: active.value, detail: active.detail, action: active.action)
        }
    }

    var stopRequested: Bool { state.stopRequested }

    /// Newest trainer step any rule applied, for log lines.
    var newestAppliedTrainerStep: Int? {
        state.rules.compactMap(\.newestAppliedTrainerStep).max()
    }

    /// R0's evaluation-side reset after a trainer-clock rewind: pending
    /// sustain progress, rule 4's running minimum and every rule's freshness
    /// step describe weights that no longer exist. Active alarms stay; they
    /// clear only by recovery.
    mutating func resetForTrainerClockRewind() {
        for rule in TrainingHealthRule.allCases {
            state.rules[rule.ruleOrder].raiseStreak = nil
            state.rules[rule.ruleOrder].clearStreak = nil
            state.rules[rule.ruleOrder].newestAppliedTrainerStep = nil
        }
        state.illegalMassRunningMinimum = nil
    }

    // MARK: Evaluate

    mutating func evaluate(
        _ observation: TrainingHealthObservation,
        config: TrainingHealthConfig
    ) -> TrainingHealthEvaluation {
        guard config.enabled else {
            return TrainingHealthEvaluation(
                events: [], stopRequest: nil, active: activeAlarms, noDataRules: [],
                staleRules: [], newlyNotApplicable: [], checkDue: false, window: observation.window)
        }
        let step = observation.trainerStep
        let bucket = step / config.checkIntervalSteps

        var checkDue = false
        if let window = observation.window {
            let startBucket = state.lastCheckBucket ?? ((window.firstTrainerStep ?? step) - 1) / config.checkIntervalSteps
            if bucket > startBucket {
                checkDue = true
                state.lastCheckBucket = bucket
            } else {
                state.lastCheckBucket = startBucket
            }
        }

        var events: [TrainingHealthEvent] = []
        var noData: [TrainingHealthRule] = []
        var stale: [TrainingHealthRule] = []
        var newlyNotApplicable: [(rule: TrainingHealthRule, reason: String)] = []
        var raisedOrEscalated: Set<TrainingHealthRule> = []

        for rule in TrainingHealthRule.allCases {
            let assessment = assess(rule, observation: observation, config: config)
            let outcome = apply(
                assessment, to: rule, observation: observation, config: config, bucket: bucket)
            events.append(contentsOf: outcome.events)
            switch outcome.status {
            case .applied: break
            case .noData:
                // R0 counts "no data" so that a silent rule is visible. A
                // rule this observation's source never carries (the window
                // rules on a checkpoint pass, rule 3 on a live read, all but
                // rule 3 on a dedicated value-FC1 read) would be counted on
                // every healthy evaluation, and a broken input path would
                // look exactly like normal operation.
                if Self.observation(observation, feeds: rule) { noData.append(rule) }
            case .stale: stale.append(rule)
            case .notApplicable(let reason, let first):
                if first { newlyNotApplicable.append((rule, reason)) }
            }
            if outcome.events.contains(where: { $0.kind == .raise || $0.kind == .escalate }) {
                raisedOrEscalated.insert(rule)
            }
        }

        // Rule 4's running minimum includes this window after it was judged.
        if let median = observation.window?.illegalMassMedian {
            state.illegalMassRunningMinimum = min(state.illegalMassRunningMinimum ?? median, median)
        }

        if checkDue {
            for rule in TrainingHealthRule.allCases where !raisedOrEscalated.contains(rule) {
                guard let active = state.rules[rule.ruleOrder].active else { continue }
                events.append(TrainingHealthEvent(
                    kind: .active, rule: rule, severity: active.severity, trainerStep: step,
                    since: active.since, value: active.value, threshold: "", detail: active.detail,
                    action: active.action, learningRate: observation.effectiveLearningRate,
                    momentum: observation.effectiveMomentum))
            }
        }

        var stopRequest: TrainingHealthEvent?
        if stopDecision == .byEvaluator,
           !state.stopRequested,
           let qualifying = TrainingHealthStopPolicy.firstQualifying(active: activeAlarms, actions: config.actions) {
            let event = TrainingHealthEvent(
                kind: .stop, rule: qualifying.rule, severity: qualifying.severity, trainerStep: step,
                since: qualifying.since, value: qualifying.value, threshold: "", detail: "",
                action: config.actions[qualifying.rule], learningRate: observation.effectiveLearningRate,
                momentum: observation.effectiveMomentum)
            events.append(event)
            stopRequest = event
            state.stopRequested = true
        }

        events.sort {
            ($0.rule.ruleOrder, $0.kind.kindOrder) < ($1.rule.ruleOrder, $1.kind.kindOrder)
        }
        return TrainingHealthEvaluation(
            events: events, stopRequest: stopRequest, active: activeAlarms, noDataRules: noData,
            staleRules: stale, newlyNotApplicable: newlyNotApplicable, checkDue: checkDue,
            window: observation.window)
    }

    /// Whether `observation`'s source is meant to carry `rule`'s input. A
    /// live evaluation carries the window (rules 1, 4–7, 9) and the live read
    /// (rules 1, 2, 8) — whether or not that read succeeded, since a failed
    /// read is exactly the no-data case the line must show; a checkpoint
    /// evaluation carries only its digest, whose tier says which rules it
    /// feeds.
    static func observation(_ observation: TrainingHealthObservation, feeds rule: TrainingHealthRule) -> Bool {
        if observation.isLive {
            return LayerHealthDigest.Tier.live.feeds(rule) || isWindowRule(rule)
        }
        guard let tier = observation.layerHealth?.tier else { return false }
        return tier.feeds(rule)
    }

    /// Rules whose input is the step window.
    static func isWindowRule(_ rule: TrainingHealthRule) -> Bool {
        switch rule {
        case .illegalMass, .gradientCollapse, .lossSpike, .policyOffsetDrift, .gradientSpike:
            return true
        case .nonFinite, .deadChannels, .valueFC1ZeroVelocity, .batchNormRunningVarianceRunaway:
            return false
        }
    }

    // MARK: Assessment (one rule, one observation)

    /// What one observation says about one rule, before sustain.
    enum Assessment: Sendable {
        case noData
        case notApplicable(reason: String)
        /// The raise condition holds at `severity`.
        case raise(severity: TrainingAlarm.Severity, value: String, threshold: String, detail: String, count: Int?)
        /// Between the clear and raise thresholds.
        case hold(value: String, detail: String)
        /// The clear condition holds.
        case clear(value: String, detail: String)
    }

    func assess(
        _ rule: TrainingHealthRule,
        observation: TrainingHealthObservation,
        config: TrainingHealthConfig
    ) -> Assessment {
        switch rule {
        case .nonFinite: return assessNonFinite(observation)
        case .deadChannels: return assessDeadChannels(observation.layerHealth?.deadChannels)
        case .valueFC1ZeroVelocity: return assessValueFC1(observation)
        case .illegalMass: return assessIllegalMass(observation, config: config)
        case .gradientCollapse: return assessGradientCollapse(observation.window)
        case .lossSpike: return assessLossSpike(observation.window)
        case .policyOffsetDrift: return assessPolicyOffset(observation.window)
        case .batchNormRunningVarianceRunaway: return assessRunningVariance(observation.layerHealth)
        case .gradientSpike: return assessGradientSpike(observation.window)
        }
    }

    private func assessGradientSpike(_ window: TrainingHealthWindowStatistics?) -> Assessment {
        guard let window, let maximum = window.gradientNormMax,
              let reference = window.gradientNormReference, reference.median > 0 else {
            return .noData
        }
        let ratio = maximum / reference.median
        let value = "max/ref=\(Self.fixed(ratio, 2))"
        let detail = "max=\(Self.fixed(maximum, 4)) ref=\(Self.fixed(reference.median, 4))"
        if ratio >= TrainingHealthThresholds.gradientSpikeMaxRatioRaise {
            return .raise(
                severity: .warning, value: value,
                threshold: "max>=\(Self.plain(TrainingHealthThresholds.gradientSpikeMaxRatioRaise))xref",
                detail: detail, count: nil)
        }
        if ratio < TrainingHealthThresholds.gradientSpikeMaxRatioClear {
            return .clear(value: value, detail: detail)
        }
        return .hold(value: value, detail: detail)
    }

    private func assessNonFinite(_ observation: TrainingHealthObservation) -> Assessment {
        let digestCount = observation.layerHealth?.nonFiniteValueCount
        let windowCount: Int? = observation.window.flatMap { $0.recordCount > 0 ? $0.nonFiniteValueCount : nil }
        guard digestCount != nil || windowCount != nil else { return .noData }
        let total = (digestCount ?? 0) + (windowCount ?? 0)
        var parts: [String] = []
        if let digestCount { parts.append("nonFinite=\(digestCount)") }
        if let windowCount { parts.append("stepValuesNonFinite=\(windowCount)") }
        let value = parts.joined(separator: " ")
        if total > 0 {
            return .raise(severity: .critical, value: value, threshold: "nonFinite>0", detail: "", count: total)
        }
        return .hold(value: value, detail: "")
    }

    private func assessDeadChannels(_ dead: LayerHealthDigest.DeadChannels?) -> Assessment {
        guard let dead else { return .noData }
        guard dead.modeledSiteCount != 0 else {
            return .notApplicable(reason: "no BN site feeds an activation")
        }
        let denominator = dead.modeledChannelCount.map(String.init) ?? TrainingHealthLog.notMeasured
        let value = "dead=\(dead.parkedChannelCount)/\(denominator)"
        let coverage = dead.coversEveryActivation ? "" : " coverage=relu_leaky_relu_only"
        // A clear (or a pending one) still names the sites — none now — so a
        // reminder of an alarm still active during its clear sustain says why
        // it is quiet, and the check line's `dead_channels_sites=` is never
        // empty (OD-22).
        guard dead.parkedChannelCount > 0 else { return .clear(value: value, detail: "sites=none\(coverage)") }

        // Every site with a parked channel, named with its counts (owner
        // decision 2026-10-06): largest known fraction first, then sites
        // whose channel count is unknown, by count.
        let affected = dead.sites.compactMap { site -> (site: String, dead: Int, channels: Int?)? in
            guard let count = site.parkedChannelCount, count > 0 else { return nil }
            return (site.site, count, site.channelCount)
        }
        func fraction(_ entry: (site: String, dead: Int, channels: Int?)) -> Double? {
            guard let channels = entry.channels, channels > 0 else { return nil }
            return Double(entry.dead) / Double(channels)
        }
        let ordered = affected.enumerated().sorted { lhs, rhs in
            switch (fraction(lhs.element), fraction(rhs.element)) {
            case let (l?, r?): return l != r ? l > r : lhs.offset < rhs.offset
            case (.some, .none): return true
            case (.none, .some): return false
            case (.none, .none):
                return lhs.element.dead != rhs.element.dead ? lhs.element.dead > rhs.element.dead : lhs.offset < rhs.offset
            }
        }.map(\.element)
        let siteList = ordered.isEmpty
            ? "unknown"
            : ordered.map { "\($0.site)(\($0.dead)/\($0.channels.map(String.init) ?? "?"))" }.joined(separator: ",")
        let detail = "sites=\(siteList)\(coverage)"
        let largestSiteFraction = ordered.compactMap(fraction).max()

        // The overall arm judges only when the denominator is every modeled
        // channel; with it unknown the arm has no data (the warning and the
        // per-site arm still judge).
        if let modeled = dead.modeledChannelCount, modeled > 0,
           Double(dead.parkedChannelCount) / Double(modeled) >= TrainingHealthThresholds.deadChannelsCriticalOverallFraction {
            return .raise(
                severity: .critical, value: value,
                threshold: "overall>=\(Self.plain(TrainingHealthThresholds.deadChannelsCriticalOverallFraction))",
                detail: detail, count: dead.parkedChannelCount)
        }
        if let largestSiteFraction, largestSiteFraction >= TrainingHealthThresholds.deadChannelsCriticalSiteFraction {
            return .raise(
                severity: .critical, value: value,
                threshold: "site>=\(Self.plain(TrainingHealthThresholds.deadChannelsCriticalSiteFraction))",
                detail: detail, count: dead.parkedChannelCount)
        }
        return .raise(severity: .warning, value: value, threshold: "dead>0", detail: detail, count: dead.parkedChannelCount)
    }

    private func assessValueFC1(_ observation: TrainingHealthObservation) -> Assessment {
        switch valueFC1Applicability {
        case .doesNotApply(let activation):
            return .notApplicable(reason: "value FC1 activation is \(activation.rawValue), not relu")
        case .unknownFromLog:
            return .noData
        case .applies:
            break
        }
        guard let velocity = observation.layerHealth?.valueFC1, velocity.unitCount > 0 else { return .noData }
        guard observation.stamp.stepsTrainedByThisProcess >= TrainingHealthThresholds.valueFC1MinimumStepsTrainedByThisProcess else {
            return .noData
        }
        let fraction = Double(velocity.zeroVelocityUnitCount) / Double(velocity.unitCount)
        let value = "zero=\(velocity.zeroVelocityUnitCount)/\(velocity.unitCount)"
        if fraction >= TrainingHealthThresholds.valueFC1ZeroVelocityCriticalFraction {
            return .raise(
                severity: .critical, value: value,
                threshold: "zero>=\(Self.plain(TrainingHealthThresholds.valueFC1ZeroVelocityCriticalFraction))",
                detail: "", count: velocity.zeroVelocityUnitCount)
        }
        if fraction >= TrainingHealthThresholds.valueFC1ZeroVelocityWarningFraction {
            return .raise(
                severity: .warning, value: value,
                threshold: "zero>=\(Self.plain(TrainingHealthThresholds.valueFC1ZeroVelocityWarningFraction))",
                detail: "", count: velocity.zeroVelocityUnitCount)
        }
        if fraction < TrainingHealthThresholds.valueFC1ZeroVelocityClearFraction {
            return .clear(value: value, detail: "")
        }
        return .hold(value: value, detail: "")
    }

    private func assessIllegalMass(_ observation: TrainingHealthObservation, config: TrainingHealthConfig) -> Assessment {
        guard let median = observation.window?.illegalMassMedian else { return .noData }
        let value = "median=\(Self.fixed(median, 4))"
        if let minimum = state.illegalMassRunningMinimum {
            if minimum < TrainingHealthThresholds.illegalMassRegressionLearnedBelow,
               median >= TrainingHealthThresholds.illegalMassRegressionRaise {
                return .raise(
                    severity: .critical, value: value,
                    threshold: "regression>=\(Self.plain(TrainingHealthThresholds.illegalMassRegressionRaise))",
                    detail: "min=\(Self.fixed(minimum, 4))", count: nil)
            }
            if median >= TrainingHealthThresholds.illegalMassRelativeRegressionFloor,
               median >= TrainingHealthThresholds.illegalMassRelativeRegressionMultiple * minimum {
                return .raise(
                    severity: .critical, value: value,
                    threshold: "regression>=\(Self.plain(TrainingHealthThresholds.illegalMassRelativeRegressionFloor))"
                        + "&>=\(Self.plain(TrainingHealthThresholds.illegalMassRelativeRegressionMultiple))xmin",
                    detail: "min=\(Self.fixed(minimum, 4))", count: nil)
            }
        }
        if observation.trainerStep >= config.learningGateTrainerStep,
           median >= TrainingHealthThresholds.illegalMassNotLearnedRaise {
            return .raise(
                severity: .critical, value: value,
                threshold: "notLearned>=\(Self.plain(TrainingHealthThresholds.illegalMassNotLearnedRaise))",
                detail: "gate=\(config.learningGateTrainerStep)", count: nil)
        }
        if median < TrainingHealthThresholds.illegalMassClear {
            return .clear(value: value, detail: "")
        }
        return .hold(value: value, detail: "")
    }

    private func assessGradientCollapse(_ window: TrainingHealthWindowStatistics?) -> Assessment {
        guard let median = window?.gradientNormMedian else { return .noData }
        let value = "median=\(Self.fixed(median, 4))"
        if median < TrainingHealthThresholds.gradientCollapseRaise {
            return .raise(
                severity: .critical, value: value,
                threshold: "median<\(Self.plain(TrainingHealthThresholds.gradientCollapseRaise))",
                detail: "", count: nil)
        }
        if median >= TrainingHealthThresholds.gradientCollapseClear {
            return .clear(value: value, detail: "")
        }
        return .hold(value: value, detail: "")
    }

    private func assessLossSpike(_ window: TrainingHealthWindowStatistics?) -> Assessment {
        guard let window, let median = window.lossMedian, let maximum = window.lossMax,
              let reference = window.lossReference, reference.median > 0 else {
            return .noData
        }
        let medianRatio = median / reference.median
        let maxRatio = maximum / reference.median
        let value = "median/ref=\(Self.fixed(medianRatio, 2))"
        let detail = "max/ref=\(Self.fixed(maxRatio, 2)) ref=\(Self.fixed(reference.median, 4))"
        if medianRatio >= TrainingHealthThresholds.lossSpikeMedianRatioRaise {
            return .raise(
                severity: .warning, value: value,
                threshold: "median>=\(Self.plain(TrainingHealthThresholds.lossSpikeMedianRatioRaise))xref",
                detail: detail, count: nil)
        }
        if maxRatio >= TrainingHealthThresholds.lossSpikeMaxRatioRaise {
            return .raise(
                severity: .warning, value: value,
                threshold: "max>=\(Self.plain(TrainingHealthThresholds.lossSpikeMaxRatioRaise))xref",
                detail: detail, count: nil)
        }
        if medianRatio < TrainingHealthThresholds.lossSpikeMedianRatioClear {
            return .clear(value: value, detail: detail)
        }
        return .hold(value: value, detail: detail)
    }

    private func assessPolicyOffset(_ window: TrainingHealthWindowStatistics?) -> Assessment {
        guard let median = window?.policyLogitMeanAbsMedian else { return .noData }
        let value = "medianAbs=\(Self.fixed(median, 4))"
        if median >= TrainingHealthThresholds.policyOffsetDriftRaise {
            return .raise(
                severity: .warning, value: value,
                threshold: "medianAbs>=\(Self.plain(TrainingHealthThresholds.policyOffsetDriftRaise))",
                detail: "", count: nil)
        }
        if median < TrainingHealthThresholds.policyOffsetDriftClear {
            return .clear(value: value, detail: "")
        }
        return .hold(value: value, detail: "")
    }

    private func assessRunningVariance(_ digest: LayerHealthDigest?) -> Assessment {
        guard let runaway = digest?.runningVariance else { return .noData }
        let value = "rvMaxOverMedian=\(Self.fixed(runaway.maxOverMedian, 1))"
        let detail = "site=\(runaway.site)"
        if runaway.maxOverMedian >= TrainingHealthThresholds.batchNormRunningVarianceRunawayRaise {
            return .raise(
                severity: .warning, value: value,
                threshold: ">=\(Self.plain(TrainingHealthThresholds.batchNormRunningVarianceRunawayRaise))",
                detail: detail, count: nil)
        }
        if runaway.maxOverMedian < TrainingHealthThresholds.batchNormRunningVarianceRunawayClear {
            return .clear(value: value, detail: detail)
        }
        return .hold(value: value, detail: detail)
    }

    // MARK: Transition (sustain, hysteresis, escalation, worsen)

    /// An assessment that has data: what the transition acts on.
    private enum Measurement {
        case raise(severity: TrainingAlarm.Severity, value: String, threshold: String, detail: String, count: Int?)
        case hold(value: String, detail: String)
        case clear(value: String, detail: String)
    }

    private enum ApplyStatus {
        case applied
        case noData
        case stale
        case notApplicable(reason: String, firstReport: Bool)
    }

    private struct ApplyOutcome {
        let status: ApplyStatus
        let events: [TrainingHealthEvent]
    }

    private mutating func apply(
        _ assessment: Assessment,
        to rule: TrainingHealthRule,
        observation: TrainingHealthObservation,
        config: TrainingHealthConfig,
        bucket: Int
    ) -> ApplyOutcome {
        var ruleState = state.rules[rule.ruleOrder]
        defer { state.rules[rule.ruleOrder] = ruleState }
        let step = observation.trainerStep
        let action = config.actions[rule]

        let measurement: Measurement
        switch assessment {
        case .noData:
            return ApplyOutcome(status: .noData, events: [])
        case .notApplicable(let reason):
            let first = !ruleState.notApplicableReported
            ruleState.notApplicableReported = true
            return ApplyOutcome(status: .notApplicable(reason: reason, firstReport: first), events: [])
        case .raise(let severity, let value, let threshold, let detail, let count):
            measurement = .raise(severity: severity, value: value, threshold: threshold, detail: detail, count: count)
        case .hold(let value, let detail):
            measurement = .hold(value: value, detail: detail)
        case .clear(let value, let detail):
            measurement = .clear(value: value, detail: detail)
        }
        if let newest = ruleState.newestAppliedTrainerStep, step < newest {
            return ApplyOutcome(status: .stale, events: [])
        }
        ruleState.newestAppliedTrainerStep = step

        func event(
            _ kind: TrainingHealthEvent.Kind, severity: TrainingAlarm.Severity, since: Int?,
            value: String, threshold: String, detail: String
        ) -> TrainingHealthEvent {
            TrainingHealthEvent(
                kind: kind, rule: rule, severity: severity, trainerStep: step, since: since,
                value: value, threshold: threshold, detail: detail, action: action,
                learningRate: observation.effectiveLearningRate, momentum: observation.effectiveMomentum)
        }

        var events: [TrainingHealthEvent] = []
        switch measurement {
        case .raise(let severity, let value, let threshold, let detail, let count):
            ruleState.clearStreak = nil
            if var active = ruleState.active {
                active.value = value
                active.detail = detail
                active.action = action
                if severity.healthRank > active.severity.healthRank {
                    active.severity = severity
                    active.peakCount = max(active.peakCount ?? 0, count ?? 0)
                    events.append(event(.escalate, severity: severity, since: active.since,
                                        value: value, threshold: threshold, detail: detail))
                } else if let count, let peak = active.peakCount, count > peak {
                    if ruleState.lastWorsenBucket != bucket {
                        ruleState.lastWorsenBucket = bucket
                        events.append(event(.worsen, severity: active.severity, since: active.since,
                                            value: value, threshold: "",
                                            detail: detail.isEmpty ? "was=\(peak)" : "was=\(peak) \(detail)"))
                    }
                    active.peakCount = count
                } else if let count, active.peakCount == nil {
                    active.peakCount = count
                }
                ruleState.active = active
            } else {
                var streak = ruleState.raiseStreak ?? Streak(count: 0, firstTrainerStep: step)
                streak.count += 1
                if Self.isSustained(streak, by: rule.raiseSustain, at: step) {
                    ruleState.raiseStreak = nil
                    ruleState.active = ActiveState(
                        severity: severity, since: step, value: value, detail: detail,
                        action: action, peakCount: count)
                    events.append(event(.raise, severity: severity, since: step,
                                        value: value, threshold: threshold, detail: detail))
                } else {
                    ruleState.raiseStreak = streak
                }
            }

        case .hold(let value, let detail):
            ruleState.raiseStreak = nil
            ruleState.clearStreak = nil
            if var active = ruleState.active {
                active.value = value
                active.detail = detail
                active.action = action
                ruleState.active = active
            }

        case .clear(let value, let detail):
            ruleState.raiseStreak = nil
            guard var active = ruleState.active else {
                ruleState.clearStreak = nil
                break
            }
            active.value = value
            active.detail = detail
            active.action = action
            guard let clearSustain = rule.clearSustain else {
                ruleState.active = active
                break
            }
            var streak = ruleState.clearStreak ?? Streak(count: 0, firstTrainerStep: step)
            streak.count += 1
            if Self.isSustained(streak, by: clearSustain, at: step) {
                ruleState.clearStreak = nil
                ruleState.active = nil
                events.append(event(.clear, severity: active.severity, since: active.since,
                                    value: value, threshold: "", detail: detail))
            } else {
                ruleState.clearStreak = streak
                ruleState.active = active
            }
        }
        return ApplyOutcome(status: .applied, events: events)
    }

    static func isSustained(_ streak: Streak, by sustain: TrainingHealthSustain, at step: Int) -> Bool {
        streak.count >= sustain.evaluations && step - streak.firstTrainerStep >= sustain.spanSteps
    }

    // MARK: Rendering helpers

    /// A threshold constant as written in a line (`0.05`, `1000`, `1.5`).
    static func plain(_ value: Double) -> String {
        if value == value.rounded(), abs(value) < 1e15 {
            return String(Int(value))
        }
        return String(format: "%g", value)
    }

    static func fixed(_ value: Double, _ digits: Int) -> String {
        String(format: "%.\(digits)f", value)
    }
}

/// The per-segment summary HPARAM_RECORDING_PLAN P4 stores as
/// `configuration.health_alarms` (OD-10): bounded, one entry per rule that
/// raised, never per event.
struct TrainingHealthSegmentSummary: Codable, Sendable, Equatable {
    struct Raised: Codable, Sendable, Equatable {
        let rule: TrainingHealthRule
        let firstTrainerStep: Int
        let highestSeverity: TrainingAlarm.Severity
        let raiseCount: Int

        enum CodingKeys: String, CodingKey {
            case rule
            case firstTrainerStep = "first_trainer_step"
            case highestSeverity = "highest_severity"
            case raiseCount = "raise_count"
        }
    }

    /// Committed evaluations (live and checkpoint; stale ones excluded; 0
    /// when alarms are disabled).
    let evaluations: Int
    /// One entry per rule that raised, in rule order.
    let raised: [Raised]
}

enum TrainingHealthError: LocalizedError, Equatable {
    case unknownActionName(String)
    case invalidConfig(String)

    var errorDescription: String? {
        switch self {
        case .unknownActionName(let name):
            return "training health: unknown action \"\(name)\" (expected \(TrainingHealthAction.allCases.map(\.name).joined(separator: ", ")))"
        case .invalidConfig(let reason):
            return "training health: invalid config: \(reason)"
        }
    }
}
