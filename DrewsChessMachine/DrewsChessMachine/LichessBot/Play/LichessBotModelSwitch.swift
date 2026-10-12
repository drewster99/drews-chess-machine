import Foundation

/// The bot's most recent model switch (issue #20,
/// documentation/plans-active/LICHESS_BOT_MODEL_IN_USE_PLAN.md): the
/// generation new games now start with, the one it replaced, when, and why.
/// Recorded by `LichessBotModelSlots.publish`, the one place the current
/// generation changes, so the Online panel and Settings ▸ Model read the
/// same fact the session log states.
struct LichessBotModelSwitch: Sendable, Equatable {
    /// When the switch happened: the generation's own `snapshotAt`, the
    /// moment it was published.
    var at: Date { to.snapshotAt }
    /// The generation new games start with from `at` on.
    let to: LichessBotGenerationInfo
    /// The generation it replaced; nil for the first generation since the
    /// bot went online.
    let from: LichessBotGenerationInfo?
    /// The build reason ("going online", "newer file of the followed
    /// lineage", …), as the ready line logs it.
    let reason: String
}
