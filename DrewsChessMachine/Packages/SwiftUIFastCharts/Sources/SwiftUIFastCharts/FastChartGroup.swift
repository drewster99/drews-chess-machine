import Foundation
import Observation

/// Minimal shared-state object that synchronizes the hover crosshair
/// across every `FastLineChart` instance that holds a reference to
/// the same group. Held as a class so multiple charts share identity;
/// `@Observable` so SwiftUI re-renders each chart that reads
/// `hoveredX` (only those — fine-grained tracking).
///
/// X-domain (visible scroll window) is intentionally NOT on the
/// group. Pass it as a per-chart parameter to `FastLineChart` so the
/// caller decides whether 15 charts share one domain or each one
/// drives its own. DCM's `ChartCoordinator` owns the domain and
/// passes the same `ClosedRange<Double>` into every grid chart.
///
/// Persistence (saving zoom/scroll/hover state to disk) is also the
/// caller's responsibility — the group is intentionally dumb.
@MainActor
@Observable
public final class FastChartGroup {
    /// Elapsed-X position currently under the cursor on any chart
    /// participating in this group, or nil when no chart has the
    /// cursor. Each chart's hover-hit overlay writes here; each
    /// chart's crosshair overlay reads here.
    public var hoveredX: Double?

    public init(hoveredX: Double? = nil) {
        self.hoveredX = hoveredX
    }
}
