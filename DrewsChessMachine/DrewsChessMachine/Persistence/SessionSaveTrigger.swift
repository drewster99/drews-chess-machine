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
/// hard-coded there. It does share the promotion disk tag
/// (`promotionDiskTag`), which is what puts both kinds of promotion
/// save in the automatic-save retention pool
/// (`CheckpointPaths.AutomaticSaveKind`).
///
/// `CaseIterable` so the tests can walk every trigger and pin which of
/// them start the retention sweep; a new case then fails to compile in
/// that test's exhaustive switch until its retention behavior is
/// decided.
enum SessionSaveTrigger: Sendable, CaseIterable {
    /// User explicitly invoked File > Save Session (or the
    /// equivalent menu command).
    case manual
    /// Fired by `PeriodicSaveController` when its configured deadline
    /// elapsed. Arena-conflicts are already resolved by the
    /// controller before we get here.
    case periodic
    /// Fired right after a user-initiated "Promote Trainee Now"
    /// (`Train ▸ Promote Trainee Now`) so the just-promoted champion
    /// + trainer pair is captured to disk the same way an arena
    /// promotion is. Distinct from the arena's own post-promotion
    /// autosave (which is inline in the arena coordinator and reuses
    /// arena-snapshot weights) — this one goes through the shared
    /// `saveSessionInternal` gate dance because there is no arena
    /// snapshot to reuse, just the live trainer weights we just
    /// copied into the champion.
    case manualPromote
    /// Fired by SIGUSR2 (`EarlyStopCoordinator`) — a "checkpoint now and shut
    /// down" request, typically before deploying a new build. Goes through the
    /// shared `saveSessionInternal` gate dance like a manual save; tagged
    /// distinctly so the filename + log make clear the save came from the
    /// signal path (and the process then exited on success).
    case signalSave

    /// Disk tag of every promotion save: the arena's inline
    /// post-promotion save writes it directly, and `manualPromote`
    /// returns it from `diskTag`, so the two are grep-identical on disk
    /// (`…-promote.dcmsession`) and the retention sweep treats them as
    /// one kind. Declared once here so the two writers cannot drift
    /// apart from each other or from the sweep.
    static let promotionDiskTag = "promote"

    /// Short tag written into the `.dcmsession` filename.
    /// Matches the `trigger:` string the existing `CheckpointManager`
    /// API already expects. `manualPromote` reuses the promotion tag
    /// (`promotionDiskTag`) so its filename is grep-identical to an
    /// arena post-promotion save.
    var diskTag: String {
        switch self {
        case .manual: "manual"
        case .periodic: "periodic"
        case .manualPromote: Self.promotionDiskTag
        case .signalSave: "sigusr2"
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
        case .manualPromote: " (post-promotion)"
        case .signalSave: " (SIGUSR2)"
        }
    }
}
