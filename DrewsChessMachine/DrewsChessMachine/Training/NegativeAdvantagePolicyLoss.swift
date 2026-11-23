import Foundation
import TrainingParametersMacroSupport

/// What a negative-advantage sample trains the policy with — the negative
/// branch of the signed-advantage policy loss
/// `max(0, A)·CE(positive target) + max(0, −A)·negativeBranch`.
///
/// **Why three forms.**
/// - `off`: negative-advantage samples contribute no policy gradient; only
///   positive-advantage samples teach the policy (the clamp regime every run
///   trained under before the complement CE existed).
/// - `complementCE`: cross-entropy against the complement target — the
///   `(1 − ε)` main mass spread uniformly over the other legal moves, the
///   smoothing share over all legal moves, so the played move keeps a floor.
///   Its gradient pulls every other legal move toward the *same* target mass,
///   whatever the policy currently thinks of it: a sample saying "this move
///   was bad" also says "every other legal move is equally good", flattening
///   the policy over the alternatives.
/// - `unlikelihood`: says only the first half. It minimizes
///   `−log Σ_{legal j ≠ played} p_j` — minus the log of the probability the
///   policy gives the other legal moves — over the same centered, unmasked
///   softmax the cross-entropies use. Its logit gradient is `+p_a` on the
///   played move, `+p_j` on each illegal cell (pushed down, as on the
///   positive branch), and `−p_j·(1/S − 1)` on each other legal move,
///   `S = Σ_{legal j ≠ played} p_j`: each alternative is raised in proportion
///   to its current probability, which is exactly the cross-entropy gradient
///   against the current policy renormalized over the other legal moves, so
///   the ranking among the alternatives is left as the policy has it. No
///   label smoothing and no floor on the played move: the loss is bounded
///   below by zero, and its pull on the played move (`p_a` times the weight)
///   fades as that move's probability falls instead of holding it at a floor
///   — no constant drive like the unsmoothed `−A·log p` that diverged.
///
/// **Why it is live-tunable.** Both the complement CE and the unlikelihood
/// are built into every training graph, and a fed scalar
/// (`graphSelectorValue`) picks one per step and gates the branch off, so a
/// change takes effect on the next step with no graph rebuild.
///
/// **Why it is stored as an `Int`.** As for `PolicyLabelSmoothingMode`: the
/// raw integer is confined to the persistence boundary
/// (`negative_advantage_policy_loss` in `parameters.json` and
/// `UserDefaults`); session files and logs carry `logToken`.
/// `parameterRawValueRange` is pinned to the parameter's declared range by
/// `NegativeAdvantagePolicyLossTests`.
///
/// It replaced the Bool `signed_advantage_complement_ce`; every old form is
/// converted on read by `LegacySignedAdvantageComplementCE`.
public enum NegativeAdvantagePolicyLoss: Int, CaseIterable, Sendable, Identifiable {
    case off = 0
    /// Cross-entropy against `HeadLossGraph.PolicyTargets.complement`.
    case complementCE = 1
    /// `HeadLossGraph.policyUnlikelihoodPerPosition`.
    case unlikelihood = 2

    public var id: Int { rawValue }

    /// Label for the settings picker.
    public var displayName: String {
        switch self {
        case .off: return "Off"
        case .complementCE: return "Complement CE"
        case .unlikelihood: return "Unlikelihood"
        }
    }

    /// Short, stable token for logs, `results.json` and session files —
    /// deliberately not `displayName`, which is free to change for
    /// readability.
    public var logToken: String {
        switch self {
        case .off: return "off"
        case .complementCE: return "complement_ce"
        case .unlikelihood: return "unlikelihood"
        }
    }

    /// The value fed to the training graph's selector placeholder
    /// (`negative_advantage_policy_loss_selector`). The graph reads it as two
    /// thresholds (`HeadLossGraph.negativeAdvantagePolicyLossPerPosition`):
    /// above 0.5 the negative branch is on, above 1.5 it is the
    /// unlikelihood rather than the complement CE.
    public var graphSelectorValue: Float {
        switch self {
        case .off: return 0
        case .complementCE: return 1
        case .unlikelihood: return 2
        }
    }

    /// Closed range of raw values this enum covers, for pinning against the
    /// `negative_advantage_policy_loss` parameter definition.
    public static var parameterRawValueRange: ClosedRange<Int> {
        let raws = allCases.map(\.rawValue)
        guard let low = raws.min(), let high = raws.max() else {
            preconditionFailure("NegativeAdvantagePolicyLoss must have at least one case")
        }
        return low...high
    }

    /// Converts a persisted raw value.
    ///
    /// Every path that can reach this receives a valid raw value (the
    /// parameter's declared range is checked first and pinned to
    /// `parameterRawValueRange` by test), so an unrepresentable value is a
    /// programmer error and traps rather than training a loss the user did
    /// not choose.
    public init(persistedRawValue raw: Int) {
        guard let form = NegativeAdvantagePolicyLoss(rawValue: raw) else {
            preconditionFailure(
                "negative_advantage_policy_loss raw value \(raw) has no NegativeAdvantagePolicyLoss case; "
                + "the parameter's declared range and \(NegativeAdvantagePolicyLoss.self) have drifted apart"
            )
        }
        self = form
    }

    /// The form whose `logToken` is `token`, or nil for a token no case
    /// spells. Used by session resume, which reads the token form.
    public init?(logToken token: String) {
        guard let form = NegativeAdvantagePolicyLoss.allCases.first(where: { $0.logToken == token }) else {
            return nil
        }
        self = form
    }
}

// MARK: - The Bool it replaced

/// `signed_advantage_complement_ce`, the Bool this parameter replaced, and
/// its one conversion: `true` (complement CE on) is `complementCE`, `false`
/// (negative samples clamped out) is `off`. Every reader of an old form goes
/// through here — parameter JSON (`parameters.json`, settings files, lineage
/// snapshots: `ParameterValue.parametersObject`), the GUI's stored settings
/// (`TrainingParameters.convertLegacySignedAdvantageComplementCESetting`) and
/// session files (`SessionCheckpointState.convertLegacySignedAdvantageComplementCE`).
/// Nothing writes the old key.
enum LegacySignedAdvantageComplementCE {
    /// The removed parameter id (also the GUI's stored-settings key).
    static let id = "signed_advantage_complement_ce"

    /// The form the old flag trained.
    static func form(complementCEEnabled: Bool) -> NegativeAdvantagePolicyLoss {
        complementCEEnabled ? .complementCE : .off
    }

    /// `values` with the old key replaced by `negative_advantage_policy_loss`
    /// at the converted value; unchanged when the old key is absent. Refused
    /// when both keys are present (two sources for one setting) or the old
    /// value is not a Bool. `note` describes a conversion, for the caller's
    /// log line.
    static func converting(
        _ values: [String: ParameterValue]
    ) throws -> (values: [String: ParameterValue], note: String?) {
        guard let old = values[id] else { return (values, nil) }
        guard values[NegativeAdvantagePolicyLossParameter.id] == nil else {
            throw TrainingConfigError.invalidValue(
                id: NegativeAdvantagePolicyLossParameter.id,
                reason: "given together with the previous key \(id); use one or the other")
        }
        guard case .bool(let enabled) = old else {
            throw TrainingConfigError.wrongType(id: id)
        }
        let converted = form(complementCEEnabled: enabled)
        var out = values
        out[id] = nil
        out[NegativeAdvantagePolicyLossParameter.id] = .int(converted.rawValue)
        return (out, "\(id)=\(enabled) -> \(NegativeAdvantagePolicyLossParameter.id)=\(converted.logToken)")
    }
}
