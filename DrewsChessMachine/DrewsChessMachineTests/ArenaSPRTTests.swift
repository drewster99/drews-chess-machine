//
//  ArenaSPRTTests.swift
//  DrewsChessMachineTests
//
//  Correctness tests for the Sequential Probability Ratio Test used as an
//  alternative arena promotion criterion:
//   - config validation (ordered hypotheses, probability error rates)
//   - Wald boundaries `log(β/(1−α))` and `log((1−β)/α)`
//   - the generalised (score-based) log-likelihood ratio
//   - decision ordering: minimum-games floor, then evidence, then the
//     runaway guard
//   - calibration by simulation
//
//  The calibration tests at the bottom are the ones that actually prove the
//  implementation. Boundaries and plumbing are easy to verify by inspection;
//  a mis-specified likelihood silently mis-calibrates the error rates while
//  appearing to work, and would promote noise into the champion slot.
//

import XCTest
@testable import DrewsChessMachine

final class ArenaSPRTTests: XCTestCase {

    private func makeConfig(
        elo0: Double = 0,
        elo1: Double = 10,
        alpha: Double = 0.05,
        beta: Double = 0.05,
        minGames: Int = 32,
        maxGames: Int = 20000
    ) throws -> ArenaSPRT.SPRTConfig {
        try ArenaSPRT.SPRTConfig(
            elo0: elo0, elo1: elo1, alpha: alpha, beta: beta,
            minGames: minGames, maxGames: maxGames
        )
    }

    // MARK: - Configuration validation

    func testHypothesesMustBeOrdered() {
        XCTAssertThrowsError(try makeConfig(elo0: 10, elo1: 10))
        XCTAssertThrowsError(try makeConfig(elo0: 20, elo1: 10))
        XCTAssertNoThrow(try makeConfig(elo0: 0, elo1: 10))
    }

    func testErrorRatesMustBeProbabilities() {
        for bad in [0.0, 1.0, -0.1, 1.5] {
            XCTAssertThrowsError(try makeConfig(alpha: bad), "alpha \(bad)")
            XCTAssertThrowsError(try makeConfig(beta: bad), "beta \(bad)")
        }
    }

    func testErrorRatesMustSumBelowOne() {
        XCTAssertThrowsError(try makeConfig(alpha: 0.6, beta: 0.6))
    }

    func testMinGamesMustBeAtLeastTwo() {
        XCTAssertThrowsError(try makeConfig(minGames: 1))
        XCTAssertNoThrow(try makeConfig(minGames: 2))
    }

    func testMaxGamesMustNotUndercutMinGames() {
        XCTAssertThrowsError(try makeConfig(minGames: 100, maxGames: 50))
    }

    func testZeroMaxGamesMeansUnbounded() throws {
        let config = try makeConfig(minGames: 100, maxGames: 0)
        XCTAssertEqual(config.maxGames, 0)
        // 50,000 games with the score sitting at the midpoint of the two
        // hypotheses (x̄ = 0.5072 against μ0 = 0.5000 and μ1 = 0.5144, LLR
        // ≈ +0.09): the evidence genuinely has not decided, and with the
        // guard disabled the test must keep going rather than time out
        // into `.inconclusive` on sample size alone.
        //
        // The tally has to be chosen this way. An *even* record is not the
        // ambiguous case it looks like — at N = 50,000, x̄ = 0.5 is
        // overwhelming evidence AGAINST "the candidate is 10 Elo better"
        // (LLR ≈ −103, some 35× past the rejection bound), and this test
        // originally asserted `.continueTesting` on exactly that record.
        let decision = ArenaSPRT.decide(wins: 5360, draws: 40000, losses: 4640, config: config)
        XCTAssertEqual(decision, .continueTesting)
    }

    // MARK: - Boundaries

    func testBoundsAtFivePercentAreSymmetric() {
        let (lower, upper) = ArenaSPRT.bounds(alpha: 0.05, beta: 0.05)
        XCTAssertEqual(upper, 2.9444, accuracy: 0.001)
        XCTAssertEqual(lower, -2.9444, accuracy: 0.001)
        XCTAssertEqual(lower + upper, 0, accuracy: 1e-12)
    }

    func testBoundsAreAsymmetricWhenErrorRatesDiffer() {
        let (lower, upper) = ArenaSPRT.bounds(alpha: 0.01, beta: 0.10)
        XCTAssertGreaterThan(
            upper, abs(lower),
            "a stricter alpha should demand more evidence before accepting"
        )
    }

    // MARK: - Expected score

    func testZeroEloIsAnEvenScore() {
        XCTAssertEqual(ArenaSPRT.expectedScore(forElo: 0), 0.5, accuracy: 1e-12)
    }

    func testEloAndScoreRoundTripThroughArenaEloStats() {
        for elo in [-100.0, -10, 0, 10, 100] {
            let score = ArenaSPRT.expectedScore(forElo: elo)
            let back = ArenaEloStats.elo(fromScore: score)
            XCTAssertNotNil(back)
            XCTAssertEqual(back ?? .nan, elo, accuracy: 1e-9)
        }
    }

    // MARK: - Likelihood ratio

    func testUndefinedBelowTwoGames() throws {
        let config = try makeConfig()
        XCTAssertNil(ArenaSPRT.logLikelihoodRatio(wins: 1, draws: 0, losses: 0, config: config))
        XCTAssertNil(ArenaSPRT.logLikelihoodRatio(wins: 0, draws: 0, losses: 0, config: config))
    }

    func testUndefinedWhenVarianceIsZero() throws {
        let config = try makeConfig()
        // Every game identical: no spread, so the normalised form has no denominator.
        XCTAssertNil(ArenaSPRT.logLikelihoodRatio(wins: 50, draws: 0, losses: 0, config: config))
        XCTAssertNil(ArenaSPRT.logLikelihoodRatio(wins: 0, draws: 50, losses: 0, config: config))
    }

    func testWinningRecordsPushTheRatioUp() throws {
        let config = try makeConfig()
        let llr = ArenaSPRT.logLikelihoodRatio(wins: 120, draws: 700, losses: 80, config: config)
        XCTAssertNotNil(llr)
        XCTAssertGreaterThan(llr ?? 0, 0)
    }

    func testLosingRecordsPushTheRatioDown() throws {
        let config = try makeConfig()
        let llr = ArenaSPRT.logLikelihoodRatio(wins: 80, draws: 700, losses: 120, config: config)
        XCTAssertNotNil(llr)
        XCTAssertLessThan(llr ?? 0, 0)
    }

    func testDrawsConcentrateEvidenceRelativeToDecisiveGames() throws {
        let config = try makeConfig()
        // Same score (0.55), same N, but one tally is draw-heavy. Lower
        // variance means the same edge counts as stronger evidence.
        let decisive = ArenaSPRT.logLikelihoodRatio(wins: 110, draws: 0, losses: 90, config: config)
        let drawHeavy = ArenaSPRT.logLikelihoodRatio(wins: 60, draws: 100, losses: 40, config: config)
        XCTAssertNotNil(decisive)
        XCTAssertNotNil(drawHeavy)
        XCTAssertGreaterThan(drawHeavy ?? 0, decisive ?? 0)
    }

    // MARK: - Decisions

    func testBelowMinimumGamesNeverDecides() throws {
        let config = try makeConfig(minGames: 100)
        // A record that would otherwise cross the upper boundary immediately.
        let decision = ArenaSPRT.decide(wins: 90, draws: 0, losses: 1, config: config)
        XCTAssertEqual(decision, .continueTesting)
        XCTAssertFalse(decision.isFinal)
    }

    func testLopsidedWinsAccept() throws {
        let config = try makeConfig(minGames: 32)
        let decision = ArenaSPRT.decide(wins: 400, draws: 200, losses: 100, config: config)
        XCTAssertEqual(decision, .accept)
        XCTAssertTrue(decision.promotes)
        XCTAssertTrue(decision.isFinal)
    }

    func testLopsidedLossesReject() throws {
        let config = try makeConfig(minGames: 32)
        let decision = ArenaSPRT.decide(wins: 100, draws: 200, losses: 400, config: config)
        XCTAssertEqual(decision, .reject)
        XCTAssertFalse(decision.promotes)
        XCTAssertTrue(decision.isFinal)
    }

    func testEvenRecordKeepsTesting() throws {
        let config = try makeConfig(minGames: 32, maxGames: 100000)
        let decision = ArenaSPRT.decide(wins: 150, draws: 700, losses: 150, config: config)
        XCTAssertEqual(
            decision, .continueTesting,
            "sample size alone must not manufacture a decision"
        )
    }

    func testRunawayGuardYieldsInconclusiveNotRejection() throws {
        let config = try makeConfig(minGames: 32, maxGames: 1000)
        let decision = ArenaSPRT.decide(wins: 150, draws: 700, losses: 150, config: config)
        XCTAssertEqual(decision, .inconclusive)
        XCTAssertFalse(decision.promotes, "inconclusive must never promote")
    }

    func testEvidenceBeatsTheGuardOnTheSameGame() throws {
        // A tally that both crosses the boundary and reaches maxGames should be
        // decided on the evidence, not declared inconclusive.
        let config = try makeConfig(minGames: 32, maxGames: 700)
        let decision = ArenaSPRT.decide(wins: 400, draws: 200, losses: 100, config: config)
        XCTAssertEqual(decision, .accept)
    }

    // MARK: - Calibration by simulation

    /// Drive the decision function over synthetic tournaments at a known true
    /// Elo. No games are played and no engine is involved: a true Elo becomes
    /// W/D/L probabilities at a fixed draw rate, outcomes are sampled, and the
    /// decision is evaluated game by game exactly as the driver will evaluate
    /// it.
    private func simulate(
        trueElo: Double,
        drawRate: Double,
        config: ArenaSPRT.SPRTConfig,
        trials: Int,
        seed: UInt64
    ) -> (accept: Double, reject: Double, inconclusive: Double) {
        var generator = SplitMix64(seed: seed)
        let mean = ArenaSPRT.expectedScore(forElo: trueElo)
        let pWin = max(0.0, mean - drawRate / 2.0)
        let pDraw = drawRate
        var accepts = 0, rejects = 0, inconclusives = 0
        let cap = config.maxGames > 0 ? config.maxGames : 100_000

        for _ in 0..<trials {
            var wins = 0, draws = 0, losses = 0
            for _ in 0..<cap {
                let roll = Double.random(in: 0..<1, using: &generator)
                if roll < pWin { wins += 1 }
                else if roll < pWin + pDraw { draws += 1 }
                else { losses += 1 }

                let decision = ArenaSPRT.decide(
                    wins: wins, draws: draws, losses: losses, config: config
                )
                if decision == .accept { accepts += 1; break }
                if decision == .reject { rejects += 1; break }
                if decision == .inconclusive { inconclusives += 1; break }
            }
        }
        let total = Double(trials)
        return (Double(accepts) / total, Double(rejects) / total, Double(inconclusives) / total)
    }

    func testFalsePromoteRateApproximatesAlpha() throws {
        // H0 true: the candidate is exactly as strong as the champion, so the
        // accept rate is the type I error the config promises.
        let config = try makeConfig(
            elo0: 0, elo1: 10, alpha: 0.05, beta: 0.05, minGames: 32, maxGames: 20000
        )
        let result = simulate(
            trueElo: 0, drawRate: 0.85, config: config, trials: 300, seed: 0x00C0FFEE
        )
        XCTAssertLessThan(
            result.accept, 0.13,
            "false-promote rate \(result.accept) should sit near alpha = 0.05"
        )
        XCTAssertGreaterThan(result.reject, 0.80)
    }

    func testTruePromoteRateApproximatesOneMinusBeta() throws {
        // H1 true: the candidate really is elo1 stronger, so the accept rate is
        // the power, 1 − beta.
        let config = try makeConfig(
            elo0: 0, elo1: 10, alpha: 0.05, beta: 0.05, minGames: 32, maxGames: 20000
        )
        let result = simulate(
            trueElo: 10, drawRate: 0.85, config: config, trials: 300, seed: 0x0000BEEF
        )
        XCTAssertGreaterThan(
            result.accept, 0.82,
            "power \(result.accept) should sit near 1 - beta = 0.95"
        )
    }

    func testClearlyWorseCandidateIsRejected() throws {
        let config = try makeConfig(elo0: 0, elo1: 10, minGames: 32, maxGames: 20000)
        let result = simulate(
            trueElo: -30, drawRate: 0.85, config: config, trials: 150, seed: 0x0000D00D
        )
        XCTAssertGreaterThan(result.reject, 0.93)
        XCTAssertLessThan(result.accept, 0.03)
    }
}

/// Deterministic generator so the calibration tests are reproducible across
/// runs and machines. SplitMix64.
private struct SplitMix64: RandomNumberGenerator {
    private var state: UInt64

    init(seed: UInt64) { self.state = seed }

    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}

// MARK: - Latched verdict

/// Tests for `ArenaSPRT.Monitor`, the piece that turns a per-game stream of
/// tallies into the single verdict a tournament stopped on.
///
/// This is deliberately separate from `TickTournamentDriverTests`: the driver
/// tests run real networks with random weights, so *which* way a sequential
/// test decides there is a coin flip, and a test that asserted on it would be
/// flaky by construction. The stopping and latching rules are pure, so they
/// are driven here with synthetic outcomes instead — including the one
/// property that only appears at `concurrency > 1`, where the tally at the
/// crossing and the tally at the end are different tallies.
final class ArenaSPRTMonitorTests: XCTestCase {

    private enum GameOutcome {
        case win, draw, loss
    }

    /// A monitor plus the running tally that feeds it, in candidate
    /// perspective.
    ///
    /// The tally has to live here rather than inside `Monitor` (which keeps
    /// none, by design — the driver already owns one) and it has to persist
    /// across calls, because the drain test's whole point is that the second
    /// batch of games continues the first rather than starting over.
    private struct SyntheticTournament {
        private(set) var monitor: ArenaSPRT.Monitor
        private(set) var wins = 0
        private(set) var draws = 0
        private(set) var losses = 0

        init(config: ArenaSPRT.SPRTConfig) {
            self.monitor = ArenaSPRT.Monitor(config: config)
        }

        var gamesPlayed: Int { wins + draws + losses }
        var verdict: ArenaSPRT.Verdict? { monitor.verdict }
        var shouldKeepPlaying: Bool { monitor.shouldKeepPlaying }

        /// Feeds outcomes one at a time, exactly as the driver's game-end
        /// pass does.
        mutating func play(_ outcomes: [GameOutcome]) {
            for outcome in outcomes {
                switch outcome {
                case .win: wins += 1
                case .draw: draws += 1
                case .loss: losses += 1
                }
                monitor.observeCompletedGame(wins: wins, draws: draws, losses: losses)
            }
        }
    }

    private func makeConfig(
        elo0: Double = 0,
        elo1: Double = 10,
        alpha: Double = 0.05,
        beta: Double = 0.05,
        minGames: Int = 32,
        maxGames: Int = 20000
    ) throws -> ArenaSPRT.SPRTConfig {
        try ArenaSPRT.SPRTConfig(
            elo0: elo0, elo1: elo1, alpha: alpha, beta: beta,
            minGames: minGames, maxGames: maxGames
        )
    }

    func testUndecidedMonitorKeepsPlaying() throws {
        var tournament = SyntheticTournament(config: try makeConfig())
        XCTAssertTrue(tournament.shouldKeepPlaying)
        XCTAssertNil(tournament.verdict)

        // Well short of minGames, so nothing can fire regardless of record.
        tournament.play([.loss] + Array(repeating: .win, count: 9))
        XCTAssertTrue(tournament.shouldKeepPlaying)
        XCTAssertNil(tournament.verdict)
    }

    func testLatchesAcceptAndStopsPlaying() throws {
        var tournament = SyntheticTournament(config: try makeConfig(minGames: 32))
        // Note the leading loss. A *perfect* record has zero empirical score
        // variance, which is the one tally the GSPRT declines to score at all
        // (see `testPerfectRecordNeverDecidesUntilTheGuardFires`), so a
        // fixture of pure wins would never latch anything.
        tournament.play([.loss] + Array(repeating: .win, count: 39))

        let verdict = try XCTUnwrap(tournament.verdict)
        XCTAssertEqual(verdict.decision, .accept)
        XCTAssertTrue(verdict.promotes)
        XCTAssertFalse(tournament.shouldKeepPlaying)
        // Fires on the first game that clears the floor, not later.
        XCTAssertEqual(verdict.gamesAtDecision, 32)
        XCTAssertEqual(verdict.wins, 31)
        XCTAssertEqual(verdict.losses, 1)
    }

    func testLatchesRejectAndDoesNotPromote() throws {
        var tournament = SyntheticTournament(config: try makeConfig(minGames: 32))
        tournament.play([.win] + Array(repeating: .loss, count: 39))

        let verdict = try XCTUnwrap(tournament.verdict)
        XCTAssertEqual(verdict.decision, .reject)
        XCTAssertFalse(verdict.promotes)
        XCTAssertFalse(tournament.shouldKeepPlaying)
        XCTAssertEqual(verdict.gamesAtDecision, 32)
    }

    /// The whole reason `Monitor` exists. When the ratio crosses a bound there
    /// are still up to `K − 1` games in flight; they finish and they are
    /// tallied, but they must not move the answer. Here the drain is lopsided
    /// enough that `decide` on the *final* tally disagrees with the latched
    /// verdict — which is exactly the disagreement the driver must resolve in
    /// favour of the latch.
    func testVerdictSurvivesADrainThatWouldFlipIt() throws {
        let config = try makeConfig(minGames: 32)
        var tournament = SyntheticTournament(config: config)

        tournament.play([.loss] + Array(repeating: .win, count: 39))
        let verdict = try XCTUnwrap(tournament.verdict)
        XCTAssertEqual(verdict.decision, .accept)

        // Now drain a long run of losses, as if K had been very large.
        tournament.play(Array(repeating: .loss, count: 400))

        // The latch is untouched...
        let after = try XCTUnwrap(tournament.verdict)
        XCTAssertEqual(after, verdict, "a latched verdict must not be revised by drained games")

        // ...and it genuinely disagrees with the final record, which is the
        // situation this test exists to pin.
        let onFinalTally = ArenaSPRT.decide(
            wins: tournament.wins, draws: tournament.draws, losses: tournament.losses, config: config
        )
        XCTAssertEqual(onFinalTally, .reject)
        XCTAssertNotEqual(onFinalTally, after.decision)
        XCTAssertEqual(after.gamesAtDecision, 32)
        XCTAssertEqual(tournament.gamesPlayed, 440)
        XCTAssertLessThan(
            after.gamesAtDecision, tournament.gamesPlayed,
            "the drain must leave more games played than the verdict was decided on"
        )
    }

    /// The runaway guard is evaluated per completed game, so it fires on a
    /// tally of exactly `maxGames`.
    func testRunawayGuardLatchesInconclusiveAtMaxGames() throws {
        // elo1 = 1 puts the hypotheses close enough together that 100 games
        // cannot separate them, and alternating results hold the score at 0.5.
        let config = try makeConfig(elo1: 1, minGames: 32, maxGames: 100)
        var tournament = SyntheticTournament(config: config)
        tournament.play((0..<100).map { $0 % 2 == 0 ? .win : .loss })

        let verdict = try XCTUnwrap(tournament.verdict)
        XCTAssertEqual(verdict.decision, .inconclusive)
        XCTAssertFalse(verdict.promotes, "an inconclusive test is not evidence of improvement")
        XCTAssertEqual(verdict.gamesAtDecision, 100)
        XCTAssertEqual(verdict.wins, 50)
        XCTAssertEqual(verdict.losses, 50)
    }

    /// **Pins a live question, not a blessed behaviour.** The GSPRT's
    /// denominator is the *empirical* score variance, which is exactly zero
    /// when every game had the same result. `logLikelihoodRatio` returns nil
    /// there, so `decide` never reaches the boundary check — and a candidate
    /// that wins every single game therefore never crosses the accept bound.
    /// It runs to `maxGames` and latches `.inconclusive`, which does not
    /// promote.
    ///
    /// That is backwards on its face: a flawless record is the most promotable
    /// one there is. It is also self-correcting the instant a single game
    /// differs — 31W/1L already accepts at LLR ≈ 7.0 — so it can only bite a
    /// candidate that stays perfect for an entire run. This test records the
    /// current behaviour so that changing it is a deliberate change; see
    /// `documentation/arena-sprt.md`, "Zero-variance records".
    func testPerfectRecordNeverDecidesUntilTheGuardFires() throws {
        let config = try makeConfig(minGames: 32, maxGames: 64)
        var tournament = SyntheticTournament(config: config)

        tournament.play(Array(repeating: .win, count: 63))
        XCTAssertNil(tournament.verdict, "zero variance means the ratio is never scored")

        tournament.play([.win])
        let verdict = try XCTUnwrap(tournament.verdict)
        XCTAssertEqual(verdict.decision, .inconclusive)
        XCTAssertNil(verdict.llr, "the guard fired without a scoreable ratio")
        XCTAssertFalse(verdict.promotes, "a 64-0 candidate does not promote under the current statistic")
    }

    /// The same degeneracy in the other direction. This one matters more in
    /// practice: the engine's regime is draw-heavy, so an all-draw opening
    /// stretch is far likelier than an all-win one. It resolves as soon as any
    /// decisive game lands.
    func testAllDrawRecordIsUnscoreableUntilOneDecisiveGame() throws {
        let config = try makeConfig(minGames: 32, maxGames: 64)
        var tournament = SyntheticTournament(config: config)

        tournament.play(Array(repeating: .draw, count: 63))
        XCTAssertNil(tournament.verdict)

        tournament.play([.loss])
        let verdict = try XCTUnwrap(tournament.verdict)
        XCTAssertNotNil(verdict.llr, "a single decisive game restores a non-zero variance")
    }

    /// The verdict carries its own hypotheses so a persisted record stays
    /// interpretable without also knowing what the parameters were that day.
    func testVerdictCarriesTheConfigItWasDecidedUnder() throws {
        let config = try makeConfig(elo0: -5, elo1: 25, alpha: 0.01, beta: 0.2, minGames: 32)
        var tournament = SyntheticTournament(config: config)
        tournament.play([.loss] + Array(repeating: .win, count: 39))

        let verdict = try XCTUnwrap(tournament.verdict)
        XCTAssertEqual(verdict.decision, .accept)
        XCTAssertEqual(verdict.config, config)
    }

    /// `llr` is the ratio at the crossing, so it must agree with recomputing
    /// the ratio from the verdict's own tally — not from the final one.
    func testVerdictLLRMatchesItsOwnTally() throws {
        let config = try makeConfig(minGames: 32)
        var tournament = SyntheticTournament(config: config)
        tournament.play([.loss] + Array(repeating: .win, count: 39))

        let verdict = try XCTUnwrap(tournament.verdict)
        let recomputed = try XCTUnwrap(ArenaSPRT.logLikelihoodRatio(
            wins: verdict.wins, draws: verdict.draws, losses: verdict.losses, config: config
        ))
        XCTAssertEqual(try XCTUnwrap(verdict.llr), recomputed, accuracy: 1e-12)
        XCTAssertGreaterThanOrEqual(recomputed, config.bounds.upper)
    }
}
