import Foundation
import Observation

/// Per-theme rolling history of the Lichess 200-puzzle probe set.
///
/// Aggregated, not per-probe: each watcher tick runs all 200 puzzles
/// through the network, then folds the 200 `ProbeResult`s into eight
/// per-theme aggregates (25 puzzles per theme), and appends one
/// `Entry` per theme to that theme's series. This is the cheap version
/// of the 9-probe `TacticalProbeHistory` — the monitor needs to surface
/// "how is the network doing at <fork|pin|mateIn1|…>?" rather than 200
/// individual rows.
///
/// @MainActor + @Observable so SwiftUI re-renders the monitor view when
/// `record(_:)` lands a new tick. The watcher is also main-actor isolated.
@Observable
@MainActor
final class LichessProbeHistory {

    /// Aggregate for one theme bucket at one tick.
    struct Aggregate: Sendable {
        let theme: ProbeCategory
        /// Total probes in the theme bucket (currently always 25).
        let total: Int
        /// Probes where the expected move is the model's argmax —
        /// verdicts `.correctAndConfident` and `.correctButFlat`. This
        /// is the primary score the monitor's "correct/total" cell
        /// reports.
        let argmaxCorrect: Int
        /// Probes where the expected move is somewhere in the top-5 —
        /// includes `argmaxCorrect`. Used for a secondary "near-misses
        /// also counted" view in the trend line.
        let top5Correct: Int
        /// Probes whose forward pass failed (network missing,
        /// transient Metal error). Excluded from both numerators —
        /// shown only as an at-a-glance counter so a tick with N>0
        /// errors stands out.
        let errored: Int
        /// Sum of `expectedProb` (legal-masked mass on the bookmove)
        /// across all probes in this tick. Continuous-valued so the
        /// average can sit anywhere in `[0, 1]` and read as a smooth
        /// trend, not a 25-step quantized one. Errored probes have
        /// `expectedProb = 0` by construction and contribute 0 to the
        /// sum but DO count in `total` — they correctly drag the avg
        /// down rather than being silently excluded.
        let sumExpectedProb: Float
        /// Sum of `expectedRank` (1-indexed) across probes that
        /// produced a valid (non-nil) rank. Errored probes and any
        /// fixture where no acceptable move was legal are excluded
        /// from BOTH the sum and `countWithRank` so the avg reads as
        /// "of the probes that gave a meaningful number, what was the
        /// typical rank?"
        let sumExpectedRank: Int
        /// Denominator for `avgExpectedRank` — number of probes whose
        /// `expectedRank` was non-nil.
        let countWithRank: Int

        /// `argmaxCorrect / total` in `[0, 1]`, or 0 when `total == 0`.
        var argmaxCorrectFraction: Float {
            total > 0 ? Float(argmaxCorrect) / Float(total) : 0
        }
        /// `top5Correct / total` in `[0, 1]`.
        var top5CorrectFraction: Float {
            total > 0 ? Float(top5Correct) / Float(total) : 0
        }
        /// Mean legal-masked mass on the bookmove across the 25 probes.
        /// Continuous in `[0, 1]`. Same intuition as the `PROB` column
        /// of the small-set monitor, averaged.
        var avgExpectedProb: Float {
            total > 0 ? sumExpectedProb / Float(total) : 0
        }
        /// Mean rank of the bookmove among legal moves, restricted to
        /// probes that gave a valid rank. `nil` when every probe in
        /// this tick errored or had nil rank (transient — would only
        /// happen if the network is entirely unbuilt).
        var avgExpectedRank: Float? {
            guard countWithRank > 0 else { return nil }
            return Float(sumExpectedRank) / Float(countWithRank)
        }
    }

    /// One timestamped tick: 8 aggregates, one per theme bucket.
    struct Entry: Sendable {
        let timestamp: Date
        let aggregate: Aggregate
    }

    /// Per-theme series, newest at the end. Keyed by `ProbeCategory`
    /// rather than by raw string so a future rename of one of the
    /// `lichess*` cases reaches every site that compiles.
    private(set) var entries: [ProbeCategory: [Entry]] = [:]

    /// Full per-puzzle results from the most recent tick. 200 entries
    /// once the first tick lands, empty otherwise. The detail window
    /// reads this directly; the JSON exporter writes a snapshot of
    /// this array to disk. Replaced wholesale on each tick so the
    /// "latest" semantic is unambiguous.
    private(set) var latestPerPuzzleResults: [ProbeResult] = []

    /// Timestamp of `latestPerPuzzleResults`. nil before the first
    /// tick.
    private(set) var latestTickTimestamp: Date?

    /// ModelID-style label of the network the latest tick ran against.
    /// Surfaced in the detail window header and the JSON export so the
    /// user can tell which weight snapshot produced which numbers.
    private(set) var latestTickModelLabel: String?

    /// Cap per series so a long-running monitor doesn't grow without
    /// bound. 120 ticks × 30 min/tick = 60 hours of visible history —
    /// matches the 9-probe monitor's window (120 × 10 min = 20 hours)
    /// scaled up by the slower cadence.
    let maxEntriesPerTheme: Int

    init(maxEntriesPerTheme: Int = 120) {
        self.maxEntriesPerTheme = maxEntriesPerTheme
    }

    /// Append one tick's eight aggregates (one per theme) and replace
    /// the latest per-puzzle snapshot in one shot. Aggregate series are
    /// trimmed to the cap; the per-puzzle snapshot is unconditional —
    /// callers see the freshest 200 results.
    func record(
        _ aggregates: [Aggregate],
        allResults: [ProbeResult],
        modelLabel: String?
    ) {
        let now = Date()
        for agg in aggregates {
            var series = entries[agg.theme] ?? []
            series.append(Entry(timestamp: now, aggregate: agg))
            if series.count > maxEntriesPerTheme {
                series.removeFirst(series.count - maxEntriesPerTheme)
            }
            entries[agg.theme] = series
        }
        latestPerPuzzleResults = allResults
        latestTickTimestamp = now
        latestTickModelLabel = modelLabel
    }

    /// Most recent aggregate for one theme, or nil before the first
    /// tick lands for that theme.
    func latest(_ theme: ProbeCategory) -> Entry? {
        entries[theme]?.last
    }

    /// Latest + immediately-prior pair. Used by the row view to color
    /// the value cell (up/down/first-tick).
    func latestPair(_ theme: ProbeCategory) -> (current: Entry, previous: Entry?)? {
        guard let series = entries[theme], let last = series.last else {
            return nil
        }
        let prior: Entry? = series.count >= 2 ? series[series.count - 2] : nil
        return (last, prior)
    }

    /// Spark series of `argmaxCorrectFraction` over time for one theme.
    /// Returned as `[Float]` in the same shape the existing
    /// `TacticalProbeSparkView` expects so the row view can reuse it
    /// directly.
    func argmaxFractionSeries(_ theme: ProbeCategory) -> [Float] {
        guard let series = entries[theme] else { return [] }
        return series.map(\.aggregate.argmaxCorrectFraction)
    }

    /// Spark series of `top5CorrectFraction` over time for one theme.
    func top5FractionSeries(_ theme: ProbeCategory) -> [Float] {
        guard let series = entries[theme] else { return [] }
        return series.map(\.aggregate.top5CorrectFraction)
    }

    /// Drop all series. Wired to the monitor's "Clear history" button.
    /// Clears both the per-theme aggregates and the latest per-puzzle
    /// snapshot — leaving the snapshot around would falsely imply a
    /// "current" tick exists.
    func clearAll() {
        entries.removeAll()
        latestPerPuzzleResults = []
        latestTickTimestamp = nil
        latestTickModelLabel = nil
    }

    /// Sum of `argmaxCorrect` across all themes' latest entries, or
    /// nil if no theme has ticked yet. Cheap aggregate for a
    /// status-bar "Lichess" cell.
    var totalArgmaxCorrect: Int? {
        var hasAny = false
        var sum = 0
        for series in entries.values {
            if let last = series.last {
                sum += last.aggregate.argmaxCorrect
                hasAny = true
            }
        }
        return hasAny ? sum : nil
    }

    /// Companion of `totalArgmaxCorrect` — sum of `total` across all
    /// themes' latest entries. Together they give "X / Y" for the
    /// status-bar cell (e.g. "47 / 200").
    var totalLatestProbes: Int? {
        var hasAny = false
        var sum = 0
        for series in entries.values {
            if let last = series.last {
                sum += last.aggregate.total
                hasAny = true
            }
        }
        return hasAny ? sum : nil
    }

    /// Number of ticks recorded. Max-of-counts across themes so a
    /// transient single-theme record failure doesn't undercount.
    var tickCount: Int {
        entries.values.map(\.count).max() ?? 0
    }

    /// Fold a tick's worth of `ProbeResult`s into per-theme aggregates.
    /// Shared between the periodic `LichessProbeWatcher` and the manual
    /// "Run Lichess Probe" handler in `SessionController+LichessProbe`
    /// so both produce identical bookkeeping and write identically
    /// shaped rows into history. Pure function — exposed as `static`
    /// so callers don't need a history instance to fold.
    nonisolated static func aggregates(from results: [ProbeResult]) -> [Aggregate] {
        var byCategory: [ProbeCategory: [ProbeResult]] = [:]
        for r in results {
            byCategory[r.probe.category, default: []].append(r)
        }
        var out: [Aggregate] = []
        out.reserveCapacity(byCategory.count)
        for (cat, perThemeResults) in byCategory {
            var argmaxCorrect = 0
            var top5Correct = 0
            var errored = 0
            var sumProb: Float = 0
            var sumRank = 0
            var countRank = 0
            for r in perThemeResults {
                sumProb += r.expectedProb
                if let rank = r.expectedRank {
                    sumRank += rank
                    countRank += 1
                }
                switch r.verdict {
                case .correctAndConfident, .correctButFlat:
                    argmaxCorrect += 1
                    top5Correct += 1
                case .correctInTop5:
                    top5Correct += 1
                case .wrong:
                    break
                case .error:
                    errored += 1
                }
            }
            out.append(Aggregate(
                theme: cat,
                total: perThemeResults.count,
                argmaxCorrect: argmaxCorrect,
                top5Correct: top5Correct,
                errored: errored,
                sumExpectedProb: sumProb,
                sumExpectedRank: sumRank,
                countWithRank: countRank
            ))
        }
        return out
    }
}
