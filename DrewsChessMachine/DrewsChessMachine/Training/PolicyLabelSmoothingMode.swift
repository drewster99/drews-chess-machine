import Foundation

/// How the policy cross-entropy target spreads its label-smoothing mass over
/// a position's legal moves.
///
/// **Why two modes.** The original form, `fixedTotal`, gives every position
/// the same total smoothing mass ε, split evenly over all |legal| moves. The
/// mass each *alternative* receives — and therefore the equilibrium logit gap
/// between the played move and each alternative — then depends on how many
/// legal moves the position has: the gap is `ln((1 − ε + ε/n)/(ε/n))`,
/// which shrinks as n falls, so with only two legal moves it is roughly half
/// what it is in a typical middlegame position. The net is trained to be
/// least decisive in forced and narrow positions, which are often the ones
/// where one move is clearly right. `perMove` instead gives every non-played
/// legal move the same mass δ (additive / Lidstone smoothing, a per-move
/// pseudo-count), so the trained gap ≈ ln((1 − δ(n − 1))/δ) is nearly
/// independent of n. Its total `δ·(n − 1)` grows with the branching factor,
/// so it is capped (`policy_label_smoothing_per_move_cap`); above the cap the
/// capped total is shared equally over the alternatives. See
/// `documentation/plans-active/POLICY_LABEL_SMOOTHING_EXPERIMENTS.md` and
/// `HeadLossGraph.policyTargets(graph:movePlayed:legalMask:labelSmoothing:policySize:)`.
///
/// **Why it is stored as an `Int`.** `ParameterType` has only
/// `bool`/`int`/`double`; this follows `ArenaPromotionCriterion`, the existing
/// enum-valued parameter: the raw integer is confined to the persistence
/// boundary (`policy_label_smoothing_mode` in `parameters.json` and
/// `UserDefaults`), and every use site reads this type. Session files carry
/// `logToken` instead, so they stay readable and survive any renumbering.
///
/// `parameterRawValueRange` is the contract between this enum and the
/// parameter's declared range — two separate declarations, because the macro
/// needs a literal range — pinned together by `PolicyLabelSmoothingModeTests`.
public enum PolicyLabelSmoothingMode: Int, CaseIterable, Sendable, Identifiable {
    /// Total mass ε spread uniformly over all legal moves (the played move
    /// included): `(1 − ε)·oneHot(played) + ε·legalMask/|legal|`.
    case fixedTotal = 0
    /// Mass δ on every non-played legal move, total capped:
    /// `(1 − total)·oneHot(played) + (total/(n − 1))·(legalMask − oneHot(played))`
    /// with `total = min(δ·(n − 1), cap)`.
    case perMove = 1

    public var id: Int { rawValue }

    /// Label for the settings picker.
    public var displayName: String {
        switch self {
        case .fixedTotal: return "Fixed total ε"
        case .perMove: return "Per move δ"
        }
    }

    /// Short, stable token for logs, `results.json` and session files —
    /// deliberately not `displayName`, which is free to change for
    /// readability. Matches the spelling of the parameter's documented values.
    public var logToken: String {
        switch self {
        case .fixedTotal: return "fixed_total"
        case .perMove: return "per_move"
        }
    }

    /// The value fed to the training graph's mode selector placeholder:
    /// 1 selects the per-move targets, 0 the fixed-total targets.
    public var graphSelectorValue: Float {
        switch self {
        case .fixedTotal: return 0
        case .perMove: return 1
        }
    }

    /// Closed range of raw values this enum covers, for pinning against the
    /// `policy_label_smoothing_mode` parameter definition.
    public static var parameterRawValueRange: ClosedRange<Int> {
        let raws = allCases.map(\.rawValue)
        guard let low = raws.min(), let high = raws.max() else {
            preconditionFailure("PolicyLabelSmoothingMode must have at least one case")
        }
        return low...high
    }

    /// Converts a persisted raw value.
    ///
    /// Every path that can reach this — `TrainingParameters.read`,
    /// `applyOne`, and the snapshot accessor — validates against the
    /// parameter's declared range first, and that range is pinned to
    /// `parameterRawValueRange` by test. An unrepresentable value here is
    /// therefore a programmer error in one of those validators, not bad user
    /// input, so it traps rather than quietly training under a target form
    /// the user did not choose.
    public init(persistedRawValue raw: Int) {
        guard let mode = PolicyLabelSmoothingMode(rawValue: raw) else {
            preconditionFailure(
                "policy_label_smoothing_mode raw value \(raw) has no PolicyLabelSmoothingMode case; "
                + "the parameter's declared range and \(PolicyLabelSmoothingMode.self) have drifted apart"
            )
        }
        self = mode
    }

    /// The mode whose `logToken` is `token`, or nil for a token no case
    /// spells. Used by session resume, which reads the token form.
    public init?(logToken token: String) {
        guard let mode = PolicyLabelSmoothingMode.allCases.first(where: { $0.logToken == token }) else {
            return nil
        }
        self = mode
    }

    /// The policy-smoothing fields shared by every hyperparameter log line
    /// (`[REPLAY-HPARAMS]`, `[VS-UCI-HPARAMS]`, the GUI `[STATS]` line), so
    /// the three paths spell them identically and can be diffed directly.
    /// All four values are printed whatever the mode — the inactive ones are
    /// inert, but a log that omitted them could not show what a mode switch
    /// would have resumed with. `pLabelSmooth=` keeps the spelling it has
    /// always had.
    public static func logFields(
        mode: PolicyLabelSmoothingMode,
        epsilon: Float,
        perMove: Float,
        perMoveCap: Float
    ) -> String {
        String(format: "pLabelSmooth=%.4g", Double(epsilon))
            + " pLabelSmoothMode=\(mode.logToken)"
            + String(format: " pLabelSmoothPerMove=%.4g pLabelSmoothPerMoveCap=%.4g", Double(perMove), Double(perMoveCap))
    }
}
