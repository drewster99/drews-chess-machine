import Foundation
import Observation

/// Per-probe ring buffer of timestamped `ProbeResult` entries that the
/// Tactical Probe Monitor window observes. `record(_:)` is called once
/// per watcher tick with the full 7-probe snapshot; UI rows reach in
/// with `latest(_:)` / `latestPair(_:)` for current-value + delta and
/// with `sparkSeries(_:metric:)` for the small line chart next to each
/// value.
///
/// @MainActor so SwiftUI reads/writes happen on the same actor. The
/// store is mutated only by the watcher (which is also main-actor
/// isolated) and read by the view tree — no cross-actor synchronization
/// concerns.
@Observable
@MainActor
final class TacticalProbeHistory {
    /// One timestamped entry per recorded probe sample. `Sendable`
    /// because the underlying `ProbeResult` is `Sendable`.
    struct Entry: Sendable {
        let timestamp: Date
        let result: ProbeResult
    }

    /// Which scalar field of `ProbeResult` to extract for a spark
    /// series. Kept as an enum (rather than a key path) so the view
    /// can iterate the cases for the per-row spark column.
    enum SparkMetric: String, CaseIterable, Sendable {
        case expectedProb
        case expectedRank
        /// Entropy expressed as percentage of the legal-uniform
        /// ceiling: `legalEntropyNats / uniformLegalEntropy * 100`.
        /// Easier to read at a glance than raw nats (a 3.0-nats value
        /// is "high" when uniform is 3.4 but "collapsed" when uniform
        /// is 8.5).
        case entropyPercent
    }

    /// Per-probe-name time series, newest at the end. The dictionary is
    /// keyed by `ProbeResult.probe.name` so a probe-fixture rename would
    /// reset that series rather than corrupting it.
    private(set) var entries: [String: [Entry]] = [:]

    /// Cap per series so a long-running monitor session doesn't grow
    /// unboundedly. 120 entries × 15-second cadence ≈ 30 minutes of
    /// history — enough to read short-term drift in the spark line
    /// without filling the panel with stale data.
    let maxEntriesPerProbe: Int

    init(maxEntriesPerProbe: Int = 120) {
        self.maxEntriesPerProbe = maxEntriesPerProbe
    }

    /// Append one tick's worth of probe results, one per series. Trims
    /// each series to the cap. O(probes × 1) amortized; the trim is
    /// O(1) amortized because we drop a single oldest entry per tick
    /// once the cap is reached.
    func record(_ results: [ProbeResult]) {
        let now = Date()
        for r in results {
            var series = entries[r.probe.name] ?? []
            series.append(Entry(timestamp: now, result: r))
            if series.count > maxEntriesPerProbe {
                series.removeFirst(series.count - maxEntriesPerProbe)
            }
            entries[r.probe.name] = series
        }
    }

    /// Most recent entry for one probe (nil before the first tick lands
    /// for that probe).
    func latest(_ name: String) -> Entry? {
        entries[name]?.last
    }

    /// Latest entry plus the immediately-prior entry — the row uses
    /// this pair to color the value cell (up vs down vs first-tick).
    /// Returns nil if the series doesn't exist yet; `previous` is nil
    /// on the first tick after `record(_:)`.
    func latestPair(_ name: String) -> (current: Entry, previous: Entry?)? {
        guard let series = entries[name], let last = series.last else { return nil }
        let prior: Entry? = series.count >= 2 ? series[series.count - 2] : nil
        return (last, prior)
    }

    /// Extract a series of one metric in timestamp order for the spark
    /// chart. `expectedRank` is converted to `Float` (1-indexed); when
    /// the expected move was not in the legal set (a fixture bug, would
    /// surface as `expectedRank == nil`), the entry is dropped from the
    /// returned series so the spark line doesn't have a synthetic
    /// fill-value spike.
    func sparkSeries(_ name: String, metric: SparkMetric) -> [Float] {
        guard let series = entries[name] else { return [] }
        var out: [Float] = []
        out.reserveCapacity(series.count)
        for entry in series {
            switch metric {
            case .expectedProb:
                out.append(entry.result.expectedProb)
            case .expectedRank:
                if let r = entry.result.expectedRank { out.append(Float(r)) }
            case .entropyPercent:
                let denom = entry.result.uniformLegalEntropy
                if denom > 1e-6 {
                    out.append(entry.result.legalEntropyNats / denom * 100)
                } else {
                    out.append(0)
                }
            }
        }
        return out
    }

    /// Clear all series. Wired to the future "Reset" button in the
    /// monitor window header (not yet exposed in UI) so the user can
    /// drop accumulated history without closing/reopening the window.
    func clearAll() {
        entries.removeAll()
    }
}
