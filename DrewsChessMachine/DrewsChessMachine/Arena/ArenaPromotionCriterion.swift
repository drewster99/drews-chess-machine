import Foundation

/// Which statistical rule decides whether an arena's candidate replaces the
/// champion.
///
/// The two criteria answer different questions. `scoreThreshold` asks "over a
/// sample size fixed in advance, did the candidate's raw score clear a flat
/// cutoff?" — cheap, predictable in duration, and with no stated error rate:
/// a candidate that is exactly as strong as the champion still promotes
/// whenever noise happens to land above the cutoff. `sprt` asks "is there
/// enough accumulated evidence to prefer `elo1` over `elo0` at error rates
/// α and β?" and keeps playing until there is, which costs an unpredictable
/// number of games but bounds the false-promote rate by construction.
///
/// Stored as an `Int` because `ParameterType` has only `bool`/`int`/`double`
/// — extending the parameter system with a persistence-visible enum case
/// would thread through the macro, `Codable`, the `.dcmsession` config block
/// and the parameter display for the benefit of exactly two cases. The raw
/// integer is confined to the persistence boundary: every use site reads
/// this type instead, so no `switch` in the arena is stringly- or
/// integer-typed.
///
/// `parameterRawValueRange` is the contract between this enum and the
/// `arena_promotion_criterion` parameter's declared range. They are separate
/// declarations (the macro needs a literal range), so a test pins them
/// together — adding a third criterion without widening the parameter range
/// would otherwise make the new case unreachable through persistence, and
/// silently fall back to `scoreThreshold` on every load.
public enum ArenaPromotionCriterion: Int, CaseIterable, Sendable, Identifiable {
    /// Raw score over a fixed game count against `arena_promote_threshold`.
    case scoreThreshold = 0
    /// Sequential probability ratio test; see `ArenaSPRT`.
    case sprt = 1

    public var id: Int { rawValue }

    /// Label for the settings picker and the `crit=` log field.
    public var displayName: String {
        switch self {
        case .scoreThreshold: return "Score Threshold"
        case .sprt: return "SPRT"
        }
    }

    /// Short, stable token for logs and persisted records — deliberately not
    /// `displayName`, which is free to change for readability.
    public var logToken: String {
        switch self {
        case .scoreThreshold: return "score"
        case .sprt: return "sprt"
        }
    }

    /// Closed range of raw values this enum covers, for pinning against the
    /// `arena_promotion_criterion` parameter definition.
    public static var parameterRawValueRange: ClosedRange<Int> {
        let raws = allCases.map(\.rawValue)
        guard let low = raws.min(), let high = raws.max() else {
            preconditionFailure("ArenaPromotionCriterion must have at least one case")
        }
        return low...high
    }

    /// Converts a persisted raw value.
    ///
    /// Every path that can reach this — `TrainingParameters.read`,
    /// `applyOne`, and the UI picker — validates against the parameter's
    /// declared range first, and that range is pinned to
    /// `parameterRawValueRange` by test. An unrepresentable value here is
    /// therefore a programmer error in one of those validators, not bad user
    /// input, so it traps rather than quietly selecting a criterion the user
    /// did not choose.
    public init(persistedRawValue raw: Int) {
        guard let criterion = ArenaPromotionCriterion(rawValue: raw) else {
            preconditionFailure(
                "arena_promotion_criterion raw value \(raw) has no ArenaPromotionCriterion case; "
                + "the parameter's declared range and \(ArenaPromotionCriterion.self) have drifted apart"
            )
        }
        self = criterion
    }
}
