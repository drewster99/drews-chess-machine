import Foundation

/// Which code path initiated a session save. Used to pick the
/// on-disk filename tag, the UI status-line suffix, and the log
/// prefix so every save-success line is unambiguous when grepping
/// through a long session log.
///
/// Kept outside of `ContentView` so it is also visible to whatever
/// caller the periodic autosave path eventually lives in. Deliberately
/// excludes `.postPromotion` — that save runs in an inline detached
/// task in the arena coordinator and does not go through the shared
/// `saveSessionInternal` helper, so it has its own display strings
/// hard-coded there.
enum SessionSaveTrigger: Sendable {
    /// User explicitly invoked File > Save Session (or the
    /// equivalent menu command).
    case manual
    /// Fired by `PeriodicSaveController` when its 4-hour deadline
    /// elapsed. Arena-conflicts are already resolved by the
    /// controller before we get here.
    case periodic

    /// Short tag written into the `.dcmsession` filename.
    /// Matches the `trigger:` string the existing `CheckpointManager`
    /// API already expects.
    var diskTag: String {
        switch self {
        case .manual: "manual"
        case .periodic: "periodic"
        }
    }

    /// Suffix appended to the user-visible status line.
    /// Manual saves intentionally show no suffix — the user just
    /// clicked Save, they don't need a reminder — while periodic
    /// saves are tagged so autosaves don't look like a surprise
    /// save happened out of nowhere.
    var uiSuffix: String {
        switch self {
        case .manual: ""
        case .periodic: " (periodic)"
        }
    }
}
