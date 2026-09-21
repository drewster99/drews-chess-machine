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
        // A large ambiguous tally must not be declared inconclusive.
        let decision = ArenaSPRT.decide(wins: 5000, draws: 40000, losses: 5000, config: config)
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
