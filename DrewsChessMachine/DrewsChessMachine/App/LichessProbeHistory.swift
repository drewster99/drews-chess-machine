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
        /// Sum of `−log(max(expectedProb, ε))` across all probes in
        /// this tick, in nats. Floor `ε = 1e-8` keeps an `expectedProb
        /// = 0` (errored probe / illegal bookmove from the network's
        /// POV) from producing `+∞` — those probes contribute
        /// `−log(1e-8) ≈ 18.4` instead, a heavy but finite penalty.
        /// Averaging this over `total` is the legal-masked
        /// cross-entropy of the bookmove — the same loss the trainer
        /// minimizes, evaluated on these 200 held-out positions. Use
        /// `Double` because logs are exponentially-scaled values that
        /// can accumulate small fractional differences across 200
        /// probes that `Float` would round away.
        let sumNegLogProb: Double

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
        /// Mean per-probe `−log(p_bookmove)` in nats — the
        /// legal-masked cross-entropy of the bookmove. Lower is
        /// better. Reference points: 0 = perfect (prob 1 on
        /// bookmove); ≈ log(30) ≈ 3.4 = uniform over ~30 legal
        /// moves; the `1e-8` floor caps any single probe's
        /// contribution at ≈ 18.4 nats so a few errored probes can't
        /// dominate the mean.
        var meanNegLogProb: Double {
            total > 0 ? sumNegLogProb / Double(total) : 0
        }
    }


    /// Maximum-likelihood engine rating fitted by Bradley-Terry on
    /// per-puzzle (rating, correct) pairs — i.e. "what single Elo
    /// best explains the observed solve pattern given Lichess's
    /// per-puzzle Glicko-2 ratings?" Model:
    ///
    ///   P(solve puzzle of rating R | engine = E) = 1 / (1 + 10^((R − E)/400))
    ///
    /// The log-likelihood is concave in E with derivative
    /// `(ln 10 / 400) · Σ_i (correct_i − p_i(E))`, monotone-decreasing
    /// in E (because each `p_i` is increasing in E). Bisection on
    /// `Σ p_i = Σ correct_i` finds the zero of the derivative — 60
    /// iterations over `E ∈ [−1000, 4000]` shrinks the bracket to
    /// `5e-15`, far below display precision.
    ///
    /// Edge cases: all-wrong gives `−.infinity` (no finite MLE — the
    /// likelihood is monotone in E, peaking at `E → −∞`); all-correct
    /// gives `+.infinity`. Callers should clamp / sentinel for
    /// display. Empty input returns `.nan` — a "no data" signal.
    nonisolated static func mlePuzzleElo(pairs: [(rating: Int, correct: Bool)]) -> Double {
        guard !pairs.isEmpty else { return .nan }
        let target = Double(pairs.reduce(0) { $0 + ($1.correct ? 1 : 0) })
        if target == 0 { return -.infinity }
        if target == Double(pairs.count) { return .infinity }

        var lo: Double = -1000
        var hi: Double = 4000
        for _ in 0..<60 {
            let mid = (lo + hi) / 2
            // `expSpread` is unbounded for very negative `R - mid`, so
            // compute as `1 / (1 + 10^x)` directly — `Foundation.pow`
            // on Double handles the full range without overflow at
            // these magnitudes (|R - E| ≤ 5000 → |x| ≤ 12.5).
            let sumP = pairs.reduce(0.0) { running, pair in
                running + 1.0 / (1.0 + pow(10.0, (Double(pair.rating) - mid) / 400.0))
            }
            if sumP < target {
                lo = mid
            } else {
                hi = mid
            }
        }
        return (lo + hi) / 2
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

    /// One overall-summary sample per tick — the OVERALL row's
    /// `meanNegLogProb` and MLE puzzle-Elo, evaluated against the
    /// full 200-puzzle batch at that tick. Populated by `record(...)`
    /// alongside the per-theme `entries` so the chart can plot the
    /// across-the-board test-loss trajectory without re-folding eight
    /// per-theme series at every render. Trimmed in lockstep with
    /// `entries` to `maxEntriesPerTheme`.
    struct OverallTickSample: Sendable {
        let timestamp: Date
        let meanNegLogProb: Double
        /// MLE puzzle-Elo for this tick. Stored as Double for the same
        /// reasons as the live cell formatter: NaN if the tick had no
        /// rated puzzles, ±∞ for all-wrong / all-correct edges. Chart
        /// renderers should filter to `.isFinite` before plotting.
        let puzzleElo: Double
    }
    private(set) var overallSeries: [OverallTickSample] = []

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

    /// Trainer step count (`ChessTrainer.completedTrainSteps`) at the
    /// moment the latest tick was recorded. Captured here — rather than
    /// read live at export time — so the number in an exported snapshot
    /// is the step that actually produced the probed weights, even if the
    /// user clicks "Export latest…" long after the tick landed. nil when
    /// no trainer existed at tick time (e.g. probing a freshly built
    /// champion before Play-and-Train has started).
    private(set) var latestTickTrainingStep: Int?

    /// Total positions the trainer has consumed at tick time
    /// (`completedTrainSteps × trainingBatchSize`). Matches the
    /// "Positions trained" status bar cell. nil iff
    /// `latestTickTrainingStep` is nil.
    private(set) var latestTickPositionsTrained: Int?

    /// Cumulative active training wall-time in seconds at tick time
    /// (sum of `TrainingSegment.durationSec` plus the in-progress
    /// segment if any). Matches the "Active training time" status
    /// bar cell. nil when no checkpoint controller was available at
    /// tick time.
    private(set) var latestTickActiveTrainingSec: Double?

    /// Count of arena tournaments in `SessionController.tournamentHistory`
    /// at tick time — matches the "Arenas" status bar cell. nil when
    /// no session was available at tick time.
    private(set) var latestTickArenaCount: Int?

    /// Count of arena tournaments where the candidate was promoted
    /// (`TournamentRecord.promoted == true`) at tick time — matches
    /// the "Promotions" status bar cell. nil when no session was
    /// available at tick time.
    private(set) var latestTickPromotionCount: Int?

    /// Cap per series so a long-running monitor doesn't grow without
    /// bound. The watcher is now step-triggered (default 400 trainer
    /// steps per tick), so 120 ticks = 120 × 400 = 48,000 steps of
    /// visible history — roughly 5 hours at a sustained ~2.5 steps/s.
    /// Unlike the old time-based cadence the window stretches or
    /// shrinks with the actual step rate rather than being a fixed span
    /// of wall-clock time.
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
        modelLabel: String?,
        trainingStep: Int?,
        positionsTrained: Int?,
        activeTrainingSec: Double?,
        arenaCount: Int?,
        promotionCount: Int?
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
        latestTickTrainingStep = trainingStep
        latestTickPositionsTrained = positionsTrained
        latestTickActiveTrainingSec = activeTrainingSec
        latestTickArenaCount = arenaCount
        latestTickPromotionCount = promotionCount

        // Append the overall (200-puzzle) summary sample. Folded from the
        // per-theme aggregates we just recorded; pElo needs per-puzzle
        // ratings which only `allResults` carries, so it's computed here
        // rather than recoverable later.
        let overall = LichessProbeOverallSummary(folding: aggregates)
        let pairs: [(rating: Int, correct: Bool)] = allResults.compactMap {
            guard let meta = LichessProbeData.metadata[$0.probe.name] else { return nil }
            let isArgmaxCorrect = $0.verdict == .correctAndConfident
                || $0.verdict == .correctButFlat
            return (rating: meta.rating, correct: isArgmaxCorrect)
        }
        let elo = Self.mlePuzzleElo(pairs: pairs)
        overallSeries.append(OverallTickSample(
            timestamp: now,
            meanNegLogProb: overall.meanNegLogProb,
            puzzleElo: elo
        ))
        if overallSeries.count > maxEntriesPerTheme {
            overallSeries.removeFirst(overallSeries.count - maxEntriesPerTheme)
        }
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
        overallSeries.removeAll()
        latestPerPuzzleResults = []
        latestTickTimestamp = nil
        latestTickModelLabel = nil
        latestTickTrainingStep = nil
        latestTickPositionsTrained = nil
        latestTickActiveTrainingSec = nil
        latestTickArenaCount = nil
        latestTickPromotionCount = nil
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
            var sumNegLog: Double = 0
            for r in perThemeResults {
                sumProb += r.expectedProb
                if let rank = r.expectedRank {
                    sumRank += rank
                    countRank += 1
                }
                // 1e-8 floor: an errored probe / strict-illegal bookmove
                // would otherwise contribute +∞. Floor matches the one in
                // `LichessProbeComparison`'s analogous accumulator so live
                // and snapshot NLL stay byte-identical at equal inputs.
                sumNegLog += -log(Double(max(r.expectedProb, 1e-8)))
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
                countWithRank: countRank,
                sumNegLogProb: sumNegLog
            ))
        }
        return out
    }
}
