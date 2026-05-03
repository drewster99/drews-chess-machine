import Foundation

/// One completed arena tournament, positioned on the chart grid's
/// shared elapsed-time X axis. The `startElapsedSec` / `endElapsedSec`
/// pair lets us render arenas as duration bands rather than point
/// events, so the reader can see WHEN training was paused for
/// arena play and for HOW LONG.
struct ArenaChartEvent: Identifiable, Sendable, Equatable {
    let id: Int
    /// Session-elapsed seconds when the arena began.
    let startElapsedSec: Double
    /// Session-elapsed seconds when the arena ended (may extend
    /// past the visible chart window for very recent arenas).
    let endElapsedSec: Double
    /// Candidate score in `[0, 1]` — fraction of games the
    /// candidate won (draws count 0.5).
    let score: Double
    /// Whether the candidate was promoted to champion. Drives the
    /// bar color (green) vs kept-champion (gray) and the promotion
    /// marker.
    let promoted: Bool
}
