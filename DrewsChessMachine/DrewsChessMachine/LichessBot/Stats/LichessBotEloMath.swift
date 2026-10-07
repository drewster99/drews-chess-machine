import Foundation

/// A rating estimated from game results (`LICHESS_BOT_RECORD_STATS_PLAN.md`
/// §3.4). A perfect or a zero score has no finite maximum-likelihood
/// rating — the likelihood keeps rising without bound — so those are
/// reported as bounds rather than as a number that looks exact.
enum LichessBotRatingEstimate: Sendable, Equatable {
    /// No game with a rating to estimate from.
    case none
    /// The maximum-likelihood rating: the R whose expected total score
    /// equals the actual one.
    case estimate(Double)
    /// Every game was won. The value is the rating that half a point less
    /// would give (the half-point convention), shown as "≥".
    case atLeast(Double)
    /// Every game was lost. The value is the rating that half a point more
    /// would give, shown as "≤".
    case atMost(Double)
}

/// A 95% interval for a score fraction, both ends in 0…1.
struct LichessBotScoreInterval: Sendable, Equatable {
    let lower: Double
    let upper: Double
}

/// Elo arithmetic shared by every statistic that compares results with
/// ratings: the expected score, the maximum-likelihood performance rating,
/// the offset at which DCM scores 50%, and the score interval. One solver
/// serves the last two because they are the same equation (§3.4, §3.7).
enum LichessBotEloMath {

    /// The Elo logistic: the expected score of a player rated `rating`
    /// against one rated `opponent`. 400 points is 10:1 odds.
    static func expectedScore(rating: Double, opponent: Double) -> Double {
        1 / (1 + pow(10, (opponent - rating) / 400))
    }

    /// The performance rating over games against `opponents` (their ratings)
    /// in which DCM scored `score` points in total (a win 1, a draw ½).
    ///
    /// Why maximum likelihood rather than the linear "algorithm of 400"
    /// (`avg + 400·(W − L)/n`): the linear form can fall after a win against
    /// a much weaker opponent, and it does not agree with the expected
    /// scores the Opponent strength pane compares results with (OD-4).
    static func performanceRating(opponents: [Int], score: Double) -> LichessBotRatingEstimate {
        solve(points: opponents.map(Double.init), score: score)
    }

    /// The rating gap Δ at which DCM's results equal Elo's expectation:
    /// the Δ solving `Σ 1 / (1 + 10^((dᵢ − Δ) / 400)) = S`, where `dᵢ` is
    /// each opponent's rating minus DCM's at the game's start. DCM scores
    /// 50% against opponents rated Δ above its own rating; an accurate
    /// rating gives Δ = 0. The equation is the performance rating's with
    /// the gaps in place of the ratings, so it shares the solver and its
    /// edge cases.
    static func ratingOffset(gaps: [Int], score: Double) -> LichessBotRatingEstimate {
        solve(points: gaps.map(Double.init), score: score)
    }

    /// The bisection stops once the bracket is narrower than this; ratings
    /// are shown as integers.
    static let solverTolerance = 0.01

    /// Solve `Σᵢ E(R, pointsᵢ) = score` for R.
    ///
    /// The sum is strictly increasing in R, so the root is unique. The
    /// bracket comes from the data, never a fixed margin: with `p = S / n`
    /// and `L = 400·log₁₀(p / (1 − p))`, `n·E(R, max) ≤ Σ E(R, pᵢ) ≤
    /// n·E(R, min)`, so the root lies in `[min + L, max + L]`. A fixed
    /// margin such as ±2000 misses the root once n is large (at a
    /// half-point score, past about 50,000 games). When every point is equal
    /// the bracket has zero width and `R₀ + L` is the exact root.
    ///
    /// A perfect score (S = n) or a zero score has no finite root; those
    /// solve with half a point less or more instead (the half-point
    /// convention) and come back as `.atLeast` / `.atMost`.
    ///
    /// Ratings are integers and repeat (a bot meets the same opponents), so
    /// the sum runs over distinct values weighted by their counts: the same
    /// root, at a cost per bisection step of the distinct values rather than
    /// of the games.
    private static func solve(points: [Double], score: Double) -> LichessBotRatingEstimate {
        let n = Double(points.count)
        guard n > 0, let low = points.min(), let high = points.max() else {
            return .none
        }
        var counts: [Double: Double] = [:]
        for point in points {
            counts[point, default: 0] += 1
        }
        // Sorted so the sum runs in one order every time: the same games
        // give the same last bits, whatever the dictionary's order.
        let weighted = counts.map { (value: $0.key, count: $0.value) }.sorted { $0.value < $1.value }
        if score >= n {
            return .atLeast(root(weighted, games: n, low: low, high: high, score: n - 0.5))
        }
        if score <= 0 {
            return .atMost(root(weighted, games: n, low: low, high: high, score: 0.5))
        }
        return .estimate(root(weighted, games: n, low: low, high: high, score: score))
    }

    /// The root for `0 < score < n`, by bisection inside the data-derived
    /// bracket (see `solve`).
    private static func root(_ points: [(value: Double, count: Double)], games n: Double, low: Double, high: Double, score: Double) -> Double {
        let p = score / n
        let offset = 400 * log10(p / (1 - p))
        var lower = low + offset
        var upper = high + offset
        if upper - lower == 0 {
            return lower
        }
        while upper - lower >= solverTolerance {
            let middle = (lower + upper) / 2
            var expected = 0.0
            for point in points {
                expected += point.count * expectedScore(rating: middle, opponent: point.value)
            }
            if expected < score {
                lower = middle
            } else {
                upper = middle
            }
        }
        return (lower + upper) / 2
    }

    /// The z of a two-sided 95% interval.
    static let z95 = 1.96

    /// A 95% interval for DCM's score fraction over `games` scored games in
    /// which it scored `score` points (a win 1, a draw ½).
    ///
    /// The Wilson score interval on `p̂ = score / games` (R-1): center
    /// `(p̂ + z²/2n) / (1 + z²/n)`, half-width `z / (1 + z²/n) ·
    /// √(p̂(1 − p̂)/n + z²/4n²)`, z = 1.96. Why not the normal
    /// approximation `p̂ ± 1.96·sd/√n`: at 2–5 games every result is often
    /// the same, the sample deviation is then 0, and the interval collapses
    /// to ±0 — a claim of certainty from almost no data. Wilson's interval
    /// stays wide at small n and never leaves 0…1.
    ///
    /// Draws count as half points in `p̂`. The interval uses the binomial
    /// variance `p(1 − p)`; a game's score lies in 0…1, so its variance is
    /// at most that (`E[s²] ≤ E[s]`), and with draws the interval is
    /// conservative (a little wider than it needs to be), never too narrow.
    /// Nil with no games.
    static func scoreInterval(score: Double, games: Int) -> LichessBotScoreInterval? {
        guard games > 0 else { return nil }
        let n = Double(games)
        let p = min(1, max(0, score / n))
        let z2 = z95 * z95
        let denominator = 1 + z2 / n
        let center = (p + z2 / (2 * n)) / denominator
        let half = z95 / denominator * (p * (1 - p) / n + z2 / (4 * n * n)).squareRoot()
        return LichessBotScoreInterval(lower: max(0, center - half), upper: min(1, center + half))
    }
}
