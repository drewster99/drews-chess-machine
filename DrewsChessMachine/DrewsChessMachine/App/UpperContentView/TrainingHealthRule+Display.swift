import Foundation

/// The user-facing names and one-line meanings of the training-health rules
/// and actions, shared by the alarm list and the Health tab so the two never
/// name a rule differently. The rule ids (`rawValue`) stay the log and JSON
/// contract; these strings are display only.
extension TrainingHealthRule {

    var displayName: String {
        switch self {
        case .nonFinite: return "Non-finite values"
        case .deadChannels: return "Dead channels"
        case .valueFC1ZeroVelocity: return "Value FC1 zero velocity"
        case .illegalMass: return "Illegal-move mass"
        case .gradientCollapse: return "Gradient collapse"
        case .lossSpike: return "Loss spike"
        case .policyOffsetDrift: return "Policy offset drift"
        case .batchNormRunningVarianceRunaway: return "BN running variance"
        case .gradientSpike: return "Gradient spike"
        case .divergence: return "Divergence"
        case .valueSaturation: return "Value saturation"
        case .valueDrawSaturation: return "Value draw saturation"
        case .legalMassStall: return "Legal-mass stall"
        case .batchNormRunningVarianceJump: return "BN running-variance jump"
        }
    }

    /// What the rule watches, in one line (thresholds are
    /// `TrainingHealthThresholds`).
    var meaning: String {
        switch self {
        case .nonFinite: return "NaN or infinity in BN state, ReZero α, a checkpoint tensor or a step's loss"
        case .deadChannels: return "Parked BN channels: any → warning; 5% overall or 20% of a site → critical"
        case .valueFC1ZeroVelocity: return "Value FC1 units with zero velocity: 5% → warning, 50% → critical (ReLU only)"
        case .illegalMass: return "Illegal-move mass regressing after it was learned, or not learned past the grace"
        case .gradientCollapse: return "Window median gradient norm below 0.1"
        case .lossSpike: return "Window loss at 1.5× (median) or 3× (max) the previous 1000 steps"
        case .policyOffsetDrift: return "Window median |policy logit mean| at or above 3"
        case .batchNormRunningVarianceRunaway: return "Largest BN running-variance max/median at or above 1000"
        case .gradientSpike: return "Window's largest gradient norm at 5× the previous 1000 steps' median"
        case .divergence: return "Policy entropy < 1.0 with gNorm > 50 (warning); entropy < 0.5 or gNorm > 500 (critical)"
        case .valueSaturation: return "Value head mean |p_win − p_loss| at 0.97 (warning) or 0.995 (critical)"
        case .valueDrawSaturation: return "Value head mean p_draw at 0.92 (warning) or 0.97 (critical); fresh is 0.75"
        case .legalMassStall: return "Illegal mass above the legal-mass threshold, not improving, for the probe count"
        case .batchNormRunningVarianceJump:
            return "A BN channel's running variance rose 10× within 1,000 steps to ≥ 10× its site median (critical ≥ 100×), or the count of such channels doubled (+5)"
        }
    }
}

extension TrainingHealthAction {

    var displayName: String {
        switch self {
        case .log: return "Log"
        case .stopOnCritical: return "Stop on critical"
        case .stopOnAny: return "Stop on any"
        }
    }
}
