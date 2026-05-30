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
    /// unboundedly. `maxEntriesPerProbe` × the watcher's effective
    /// tick interval (`TacticalProbeWatcher.triggerEverySteps` × the
    /// trainer's average step latency, ~0.4 s at default cadence)
    /// determines the visible history window — enough to read
    /// short-term drift in the spark line without filling the panel
    /// with stale data.
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

    /// Aggregate "Tactical" score for the upper status bar: sum of
    /// `expectedRank` across the LATEST entry of each probe, minus
    /// the count of probes that contributed a valid rank. 0 = every
    /// probe got its expected move ranked #1 (perfect). Higher =
    /// worse (each "off-by-one" rank adds 1 to the score).
    ///
    /// Probes with `verdict == .error` (forward pass failed) or
    /// `expectedRank == nil` (fixture bug: no acceptable move was
    /// legal in the position) are excluded from BOTH the sum and the
    /// minus-count, so they don't poison the metric. Returns `nil`
    /// when no probe has produced a valid latest result yet — the
    /// status-bar cell renders "—" in that case.
    ///
    /// Reads from the LATEST per-probe entry only (not averaged
    /// across the history ring) — the user wants a snapshot of "how
    /// is the champion doing right now" rather than a smoothed
    /// trend.
    var tacticalRankSumMinusCount: Int? {
        var sum = 0
        var count = 0
        for series in entries.values {
            guard let last = series.last,
                  let rank = last.result.expectedRank else { continue }
            sum += rank
            count += 1
        }
        return count > 0 ? sum - count : nil
    }

    /// Aggregate "Tactical prob" companion to `tacticalRankSumMinusCount`:
    /// arithmetic mean of `expectedProb` across the LATEST entry of each
    /// probe series. `expectedProb` is the legal-masked renormalized
    /// probability mass the network puts on that probe's acceptable
    /// move(s), so the per-probe value is in `[0, 1]` and the mean is
    /// too. 1.0 (rendered as `100.0000%`) = every probe puts all of its
    /// legal mass on the right move; 0.0 = nothing right. Returns `nil`
    /// when no probe has produced a latest entry yet (status-bar cell
    /// renders "—" in that case).
    ///
    /// Unlike `tacticalRankSumMinusCount`, errored probes (verdict
    /// `.error` or fixture-bug nil rank) are NOT excluded — they
    /// contribute their `expectedProb` (which is 0 in the error case,
    /// per `buildProbeResult` / the `.error`-verdict fallback path) and
    /// count toward the denominator. The reason: an errored probe is a
    /// real performance failure (forward pass didn't surface a useful
    /// distribution) and should drag the score down, not silently
    /// disappear from the average.
    var tacticalAvgExpectedProb: Double? {
        var sum: Double = 0
        var count = 0
        for series in entries.values {
            guard let last = series.last else { continue }
            sum += Double(last.result.expectedProb)
            count += 1
        }
        return count > 0 ? sum / Double(count) : nil
    }
}
