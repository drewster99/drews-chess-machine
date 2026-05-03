import Foundation

/// Statistics helpers for converting raw arena W/D/L tallies into the
/// AlphaZero-style score in `[0, 1]`, an Elo differential, and a 95%
/// confidence interval on both. The candidate-perspective convention
/// is used throughout — wins/losses/draws are tallied from the point
/// of view of whoever "playerA" is in the `TournamentDriver`, which
/// at the arena call site is always the candidate / trainer network
/// being evaluated against the champion.
///
/// Conventions:
///   - `score` = `(W + 0.5·D) / N` in `[0, 1]`. 0.5 = even.
///   - `elo`   = `400 · log₁₀(p / (1 − p))`. Positive when
///               candidate outperforms champion. Undefined at p ∈ {0, 1},
///               reported as `nil` in those degenerate endpoints.
///   - CI uses the empirical variance over the three outcome buckets
///     (rather than a Bernoulli variance on p alone) so draws reduce
///     the SE correctly. Wald 95% interval: `p ± 1.96 · SE`.
enum ArenaEloStats {

    /// AlphaZero-style score. Returns 0 on an empty sample so callers
    /// that need a scalar can avoid a nil check; use `games > 0`
    /// explicitly if you need to distinguish "no data" from "zero
    /// score".
    static func score(wins: Int, draws: Int, losses: Int) -> Double {
        let n = wins + draws + losses
        guard n > 0 else { return 0 }
        return (Double(wins) + 0.5 * Double(draws)) / Double(n)
    }

    /// Elo differential implied by a score in the open interval
    /// `(0, 1)`. Returns `nil` for `p ≤ 0` or `p ≥ 1` (both
    /// mathematically ±∞) so callers can render those degenerate
    /// endpoints as "—" rather than a garbage float.
    static func elo(fromScore p: Double) -> Double? {
        guard p > 0, p < 1 else { return nil }
        return 400.0 * log10(p / (1.0 - p))
    }

    /// Full CI95 summary for a W/D/L sample. `elo`, `eloLo`, `eloHi`
    /// each become `nil` if their corresponding score value lands on
    /// or outside `(0, 1)`. `score`, `scoreLo`, `scoreHi` are always
    /// finite; the score bounds are clamped into `[0, 1]` so the
    /// displayed interval never strays outside the unit interval even
    /// for small N.
    struct Summary {
        let games: Int
        let score: Double
        let scoreLo: Double
        let scoreHi: Double
        let elo: Double?
        let eloLo: Double?
        let eloHi: Double?
    }

    /// Wald 95% confidence interval using the empirical variance over
    /// the three outcome buckets (win=1, draw=0.5, loss=0). With
    /// fewer than two games the SE collapses to 0 and the interval
    /// degenerates to a point at the score — still well-defined; the
    /// caller can gate display on `games >= 2` if a point interval
    /// isn't useful.
    static func summary(wins: Int, draws: Int, losses: Int) -> Summary {
        let n = wins + draws + losses
        guard n > 0 else {
            return Summary(games: 0, score: 0, scoreLo: 0, scoreHi: 0,
                           elo: nil, eloLo: nil, eloHi: nil)
        }
        let w = Double(wins), d = Double(draws), l = Double(losses)
        let p = (w + 0.5 * d) / Double(n)

        // Per-game-value empirical variance: each observation is
        // drawn from {1, 0.5, 0} with counts (W, D, L), so the
        // variance is the weighted squared deviation from the mean.
        let variance = (
            w * pow(1.0 - p, 2)
            + d * pow(0.5 - p, 2)
            + l * pow(0.0 - p, 2)
        ) / Double(n)

        let se = (n >= 2) ? sqrt(variance / Double(n)) : 0.0
        let z = 1.96

        let rawLo = p - z * se
        let rawHi = p + z * se
        let scoreLo = min(max(rawLo, 0), 1)
        let scoreHi = min(max(rawHi, 0), 1)

        return Summary(
            games: n,
            score: p,
            scoreLo: scoreLo,
            scoreHi: scoreHi,
            elo: elo(fromScore: p),
            eloLo: elo(fromScore: scoreLo),
            eloHi: elo(fromScore: scoreHi)
        )
    }

    /// Format an Elo value as a signed integer string (e.g. `"+28"`,
    /// `"-11"`). Returns `"—"` for `nil`.
    static func formatEloSigned(_ elo: Double?) -> String {
        guard let e = elo else { return "—" }
        let rounded = Int(e.rounded())
        if rounded >= 0 { return "+\(rounded)" }
        return "\(rounded)"
    }

    /// Pretty-printed Elo-with-CI like `"+28 [+13, +34]"`. Endpoints
    /// that are mathematically undefined (score CI hit 0 or 1)
    /// render as `"—"`.
    static func formatEloWithCI(_ s: Summary) -> String {
        let mid = formatEloSigned(s.elo)
        let lo = formatEloSigned(s.eloLo)
        let hi = formatEloSigned(s.eloHi)
        return "\(mid) [\(lo), \(hi)]"
    }

    /// Pretty-printed score-with-CI like `"51.2% [48.1%, 54.3%]"`.
    /// Score endpoints are always finite so no "—" case.
    static func formatScorePercentWithCI(_ s: Summary) -> String {
        let mid = String(format: "%.1f%%", s.score * 100)
        let lo = String(format: "%.1f%%", s.scoreLo * 100)
        let hi = String(format: "%.1f%%", s.scoreHi * 100)
        return "\(mid) [\(lo), \(hi)]"
    }
}
