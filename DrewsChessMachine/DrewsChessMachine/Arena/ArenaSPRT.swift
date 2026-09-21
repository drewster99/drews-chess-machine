import Foundation

/// Sequential Probability Ratio Test for arena promotion.
///
/// The score-threshold criterion asks "did the candidate score at least X over
/// a fixed N games?". That fixes the sample size in advance, so a candidate
/// that is obviously better still plays every game, and a candidate that is
/// marginally better promotes or not depending on where the noise landed
/// relative to a flat cutoff.
///
/// SPRT instead tests two hypotheses about the true strength difference —
/// `H₀: elo = elo0` against `H₁: elo = elo1` — and stops as soon as the
/// accumulated evidence favours one decisively. The stopping boundaries are
/// chosen so the long-run false-promote rate is `alpha` and the long-run
/// false-reject rate is `beta`.
///
/// **The test must be open-ended.** Truncating it at a fixed game count
/// destroys the calibration: simulated at `elo0 = 0, elo1 = 10` with a 400-game
/// cap, the test accepts about 0.6% of the time when `H₁` is true, against the
/// 95% its `beta` promises. `maxGames` exists only as a runaway guard, and
/// reaching it is an inconclusive result, never a promotion.
///
/// This type is deliberately free of actors, UI and I/O so the statistics can
/// be tested directly — including by simulating synthetic tournaments, which
/// needs no engine and no GPU. See `documentation/arena-sprt.md`.
enum ArenaSPRT {

    // MARK: - Configuration

    /// Why a `SPRTConfig` could not be formed. Each case is a combination that
    /// makes the likelihood ratio meaningless rather than merely unusual, so
    /// they are rejected at construction rather than producing a test that
    /// silently never decides.
    enum ConfigError: Error, Equatable, CustomStringConvertible {
        case hypothesesNotOrdered(elo0: Double, elo1: Double)
        case errorRateOutOfRange(name: String, value: Double)
        case errorRatesSumTooLarge(alpha: Double, beta: Double)
        case minGamesTooSmall(Int)
        case maxGamesBelowMinGames(minGames: Int, maxGames: Int)

        var description: String {
            switch self {
            case let .hypothesesNotOrdered(elo0, elo1):
                return "SPRT requires elo1 > elo0; got elo0=\(elo0), elo1=\(elo1)."
            case let .errorRateOutOfRange(name, value):
                return "SPRT \(name) must be in (0, 1); got \(value)."
            case let .errorRatesSumTooLarge(alpha, beta):
                return "SPRT alpha + beta must be < 1; got \(alpha) + \(beta)."
            case let .minGamesTooSmall(value):
                return "SPRT minGames must be at least 2; got \(value)."
            case let .maxGamesBelowMinGames(minGames, maxGames):
                return "SPRT maxGames (\(maxGames)) must be 0 (unbounded) or >= minGames (\(minGames))."
            }
        }
    }

    /// A validated SPRT configuration. Snapshot this at tournament start: the
    /// likelihood ratio is only meaningful against fixed hypotheses, so
    /// changing `elo0` or `elo1` mid-test invalidates every game played so far.
    struct SPRTConfig: Equatable, Sendable {
        /// Null hypothesis: the candidate is this many Elo stronger than the
        /// champion. Usually 0 — "no improvement".
        let elo0: Double
        /// Alternative hypothesis: the candidate is this many Elo stronger.
        /// The smallest improvement the test is being asked to detect.
        let elo1: Double
        /// Probability of promoting a candidate that is only `elo0` strong.
        let alpha: Double
        /// Probability of rejecting a candidate that is genuinely `elo1` strong.
        let beta: Double
        /// Games that must be played before the test may fire at all. Guards
        /// against an early streak crossing a boundary on almost no evidence.
        let minGames: Int
        /// Runaway guard. `0` means unbounded. Reaching it is inconclusive.
        let maxGames: Int

        init(
            elo0: Double,
            elo1: Double,
            alpha: Double,
            beta: Double,
            minGames: Int,
            maxGames: Int
        ) throws {
            guard elo1 > elo0 else {
                throw ConfigError.hypothesesNotOrdered(elo0: elo0, elo1: elo1)
            }
            guard alpha > 0, alpha < 1 else {
                throw ConfigError.errorRateOutOfRange(name: "alpha", value: alpha)
            }
            guard beta > 0, beta < 1 else {
                throw ConfigError.errorRateOutOfRange(name: "beta", value: beta)
            }
            guard alpha + beta < 1 else {
                throw ConfigError.errorRatesSumTooLarge(alpha: alpha, beta: beta)
            }
            guard minGames >= 2 else {
                throw ConfigError.minGamesTooSmall(minGames)
            }
            guard maxGames == 0 || maxGames >= minGames else {
                throw ConfigError.maxGamesBelowMinGames(minGames: minGames, maxGames: maxGames)
            }
            self.elo0 = elo0
            self.elo1 = elo1
            self.alpha = alpha
            self.beta = beta
            self.minGames = minGames
            self.maxGames = maxGames
        }

        /// Log-likelihood-ratio boundaries implied by `alpha` and `beta`.
        /// Crossing `upper` accepts `H₁` (promote); crossing `lower` rejects it.
        var bounds: (lower: Double, upper: Double) {
            ArenaSPRT.bounds(alpha: alpha, beta: beta)
        }
    }

    // MARK: - Decision

    /// The verdict after some number of games.
    ///
    /// `inconclusive` is distinct from `reject`: a rejection means the evidence
    /// actively favours `H₀`, while inconclusive means the runaway guard was
    /// reached with the evidence still ambiguous. Neither promotes, but they
    /// mean different things — an inconclusive run usually says `elo0` and
    /// `elo1` are too close together for the sample the guard allows.
    enum Decision: String, Equatable, Sendable {
        case accept
        case reject
        case continueTesting
        case inconclusive

        /// Only an accepted test promotes.
        var promotes: Bool { self == .accept }

        /// True when the tournament should stop spawning new games.
        var isFinal: Bool { self != .continueTesting }
    }

    // MARK: - Statistics

    /// Expected score in `[0, 1]` implied by an Elo difference, under the
    /// standard logistic model.
    static func expectedScore(forElo elo: Double) -> Double {
        1.0 / (1.0 + pow(10.0, -elo / 400.0))
    }

    /// Wald boundaries on the log-likelihood ratio.
    ///
    /// `lower = log(beta / (1 - alpha))`, `upper = log((1 - beta) / alpha)`.
    /// At `alpha = beta = 0.05` these are approximately `[-2.94, +2.94]`.
    static func bounds(alpha: Double, beta: Double) -> (lower: Double, upper: Double) {
        (log(beta / (1.0 - alpha)), log((1.0 - beta) / alpha))
    }

    /// Generalised SPRT log-likelihood ratio for a W/D/L tally.
    ///
    /// Each game contributes a score in `{0, 0.5, 1}`. Testing the mean of that
    /// variable against the two hypothesised means gives
    ///
    ///     LLR = N · (mu1 − mu0) · (xbar − (mu0 + mu1)/2) / variance
    ///
    /// where `variance` is the empirical per-game score variance over the three
    /// outcome buckets — the same quantity `ArenaEloStats` uses for its
    /// confidence interval, which is what makes draws reduce the evidence per
    /// game rather than being treated as half a win.
    ///
    /// Returns `nil` when the ratio is undefined: fewer than two games, or a
    /// zero-variance tally (every game the same result), where the normalised
    /// form has no denominator. Callers treat `nil` as "keep testing".
    static func logLikelihoodRatio(
        wins: Int,
        draws: Int,
        losses: Int,
        config: SPRTConfig
    ) -> Double? {
        let n = wins + draws + losses
        guard n >= 2 else { return nil }

        let w = Double(wins), d = Double(draws), l = Double(losses)
        let count = Double(n)
        let xbar = (w + 0.5 * d) / count
        let variance = (
            w * pow(1.0 - xbar, 2)
            + d * pow(0.5 - xbar, 2)
            + l * pow(0.0 - xbar, 2)
        ) / count
        guard variance > 0 else { return nil }

        let mu0 = expectedScore(forElo: config.elo0)
        let mu1 = expectedScore(forElo: config.elo1)
        return count * (mu1 - mu0) * (xbar - (mu0 + mu1) / 2.0) / variance
    }

    // MARK: - Latched verdict

    /// The decision a tournament actually stopped on, together with the tally
    /// it was made from.
    ///
    /// This is a distinct type from `Decision` because *when* the test stopped
    /// is part of the result. A tournament runs `concurrency` games at once,
    /// so when the log-likelihood ratio crosses a bound there are still up to
    /// `K − 1` games in flight; they finish, and they are reported. The tally
    /// here is the one at the crossing, not the one at the end.
    struct Verdict: Equatable, Sendable {
        /// Always a final decision — `.accept`, `.reject` or `.inconclusive`.
        /// `.continueTesting` never reaches a verdict.
        let decision: Decision
        /// The ratio at the crossing. `nil` only in the degenerate tallies
        /// `logLikelihoodRatio` declines to score, which means this verdict
        /// came from the runaway guard.
        let llr: Double?
        let wins: Int
        let draws: Int
        let losses: Int
        /// The hypotheses this was decided against. Carried so a persisted
        /// record is interpretable later without also knowing what the
        /// parameters happened to be on the day.
        let config: SPRTConfig

        /// Games completed when the test stopped. Less than the tournament's
        /// final `gamesPlayed` whenever `concurrency > 1`.
        var gamesAtDecision: Int { wins + draws + losses }

        /// Only an accepted test promotes.
        var promotes: Bool { decision.promotes }
    }

    /// Tracks a running sequential test and latches the first final verdict.
    ///
    /// **Why latch rather than re-decide at the end.** Recomputing `decide`
    /// on the tournament's final tally looks equivalent and is not. The extra
    /// games are a variable number of observations admitted *because* the test
    /// already stopped, which is exactly the optional-stopping bias the
    /// calibration assumes away; and a re-decision can land on
    /// `.continueTesting`, for which there is no action once the tournament is
    /// over. The stopping rule is the thing that carries the error-rate
    /// guarantee, so the answer is fixed at the moment it fires.
    ///
    /// Holds no tally of its own — the driver already owns one, and a second
    /// copy is a second source of truth. Callers pass the tally *including*
    /// the game just completed.
    ///
    /// Value type, mutated only by the single task that owns the tournament
    /// loop, so it needs no lock.
    struct Monitor: Sendable {
        let config: SPRTConfig
        private(set) var verdict: Verdict?

        init(config: SPRTConfig) {
            self.config = config
        }

        /// True while the tournament should keep spawning games.
        var shouldKeepPlaying: Bool { verdict == nil }

        /// Feed one completed game's updated tally. Ignored once a verdict is
        /// latched, so the drain of in-flight games cannot overwrite it.
        mutating func observeCompletedGame(wins: Int, draws: Int, losses: Int) {
            guard verdict == nil else { return }
            let decision = ArenaSPRT.decide(wins: wins, draws: draws, losses: losses, config: config)
            guard decision.isFinal else { return }
            verdict = Verdict(
                decision: decision,
                llr: ArenaSPRT.logLikelihoodRatio(wins: wins, draws: draws, losses: losses, config: config),
                wins: wins,
                draws: draws,
                losses: losses,
                config: config
            )
        }
    }

    /// Verdict for a W/D/L tally under `config`.
    ///
    /// Order matters: the minimum-games floor is checked before the boundaries
    /// so an early streak cannot decide the test, and the runaway guard is
    /// checked last so a tally that both crosses a boundary and reaches the
    /// guard on the same game is decided on the evidence rather than declared
    /// inconclusive.
    static func decide(
        wins: Int,
        draws: Int,
        losses: Int,
        config: SPRTConfig
    ) -> Decision {
        let n = wins + draws + losses
        guard n >= config.minGames else {
            return .continueTesting
        }
        if let llr = logLikelihoodRatio(wins: wins, draws: draws, losses: losses, config: config) {
            let (lower, upper) = config.bounds
            if llr >= upper { return .accept }
            if llr <= lower { return .reject }
        }
        if config.maxGames > 0, n >= config.maxGames {
            return .inconclusive
        }
        return .continueTesting
    }
}
