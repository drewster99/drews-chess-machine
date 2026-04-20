//
//  ArenaEloStatsTests.swift
//  DrewsChessMachineTests
//
//  Correctness tests for the arena statistics helpers:
//   - score formula `(W + 0.5·D) / N`
//   - Elo formula `400 · log10(p/(1-p))`, including degenerate p∈{0,1}
//   - 95% Wald CI using the empirical variance over the three outcome
//     buckets (draws reduce SE correctly)
//   - the string formatters used by the status bar and arena log
//
//  Bugs in this code would silently mis-label arena Elo differentials
//  and CI bands, which feed into promotion decisions and the user-
//  facing status bar. Worth pinning with unit tests even though the
//  math is small.
//

import XCTest
@testable import DrewsChessMachine

final class ArenaEloStatsTests: XCTestCase {

    // MARK: - score(wins:draws:losses:)

    func testScoreAllDraws() {
        XCTAssertEqual(ArenaEloStats.score(wins: 0, draws: 100, losses: 0), 0.5, accuracy: 1e-9)
    }

    func testScoreAllWins() {
        XCTAssertEqual(ArenaEloStats.score(wins: 50, draws: 0, losses: 0), 1.0, accuracy: 1e-9)
    }

    func testScoreAllLosses() {
        XCTAssertEqual(ArenaEloStats.score(wins: 0, draws: 0, losses: 50), 0.0, accuracy: 1e-9)
    }

    func testScoreBalanced() {
        // 30W, 40D, 30L → (30 + 20) / 100 = 0.5
        XCTAssertEqual(ArenaEloStats.score(wins: 30, draws: 40, losses: 30), 0.5, accuracy: 1e-9)
    }

    func testScoreWorkedExample() {
        // 120W, 45D, 35L → (120 + 22.5) / 200 = 0.7125
        XCTAssertEqual(ArenaEloStats.score(wins: 120, draws: 45, losses: 35), 0.7125, accuracy: 1e-9)
    }

    func testScoreEmptySampleIsZero() {
        // Documented: "returns 0 on an empty sample so callers that
        // need a scalar can avoid a nil check".
        XCTAssertEqual(ArenaEloStats.score(wins: 0, draws: 0, losses: 0), 0.0, accuracy: 1e-9)
    }

    // MARK: - elo(fromScore:)

    func testEloAtHalfIsZero() {
        XCTAssertEqual(ArenaEloStats.elo(fromScore: 0.5)!, 0.0, accuracy: 1e-9)
    }

    func testEloSignsByScore() {
        // > 0.5 → positive (outperforming), < 0.5 → negative.
        XCTAssertGreaterThan(ArenaEloStats.elo(fromScore: 0.60)!, 0)
        XCTAssertLessThan(ArenaEloStats.elo(fromScore: 0.40)!, 0)
    }

    func testEloReferenceValues() {
        // Canonical references: 0.75 ≈ +191, 0.90 ≈ +382, 0.55 ≈ +35
        XCTAssertEqual(ArenaEloStats.elo(fromScore: 0.55)!, 34.9, accuracy: 0.15)
        XCTAssertEqual(ArenaEloStats.elo(fromScore: 0.75)!, 190.8, accuracy: 0.15)
        XCTAssertEqual(ArenaEloStats.elo(fromScore: 0.90)!, 381.7, accuracy: 0.15)
    }

    func testEloDegenerateEndpoints() {
        // p = 0 and p = 1 are mathematically ±∞; return nil so the
        // display can render "—" rather than a garbage float.
        XCTAssertNil(ArenaEloStats.elo(fromScore: 0.0))
        XCTAssertNil(ArenaEloStats.elo(fromScore: 1.0))
        XCTAssertNil(ArenaEloStats.elo(fromScore: -0.1))
        XCTAssertNil(ArenaEloStats.elo(fromScore: 1.1))
    }

    func testEloMirrorSymmetry() {
        // Mirror: elo(p) == -elo(1-p). Fundamental symmetry; guards
        // against accidentally squaring or dropping a sign.
        for p in stride(from: 0.05, through: 0.95, by: 0.05) {
            let a = ArenaEloStats.elo(fromScore: p)!
            let b = ArenaEloStats.elo(fromScore: 1 - p)!
            XCTAssertEqual(a, -b, accuracy: 1e-9, "elo mirror broke at p=\(p)")
        }
    }

    // MARK: - summary(wins:draws:losses:) — variance, SE, CI

    func testSummaryAllDrawsZeroVariance() {
        // Pure-draw sample: all observations equal 0.5, so variance
        // and SE are exactly 0. CI collapses to a point at 0.5.
        let s = ArenaEloStats.summary(wins: 0, draws: 100, losses: 0)
        XCTAssertEqual(s.score, 0.5, accuracy: 1e-12)
        XCTAssertEqual(s.scoreLo, 0.5, accuracy: 1e-12)
        XCTAssertEqual(s.scoreHi, 0.5, accuracy: 1e-12)
        // At p exactly 0.5, Elo is 0 and CI endpoints are also 0.
        XCTAssertEqual(s.elo!, 0, accuracy: 1e-9)
        XCTAssertEqual(s.eloLo!, 0, accuracy: 1e-9)
        XCTAssertEqual(s.eloHi!, 0, accuracy: 1e-9)
    }

    func testSummaryBalancedDecisiveCI() {
        // 50W, 0D, 50L → score 0.5, variance = 0.25 (pure Bernoulli).
        // SE = sqrt(0.25 / 100) = 0.05. CI = 0.5 ± 0.098.
        let s = ArenaEloStats.summary(wins: 50, draws: 0, losses: 50)
        XCTAssertEqual(s.score, 0.5, accuracy: 1e-12)
        XCTAssertEqual(s.scoreLo, 0.402, accuracy: 0.002)
        XCTAssertEqual(s.scoreHi, 0.598, accuracy: 0.002)
    }

    func testSummaryDrawsReduceSE() {
        // Same score (0.5), but with draws SE should be strictly
        // smaller because draw-heavy samples have smaller variance
        // than the Bernoulli maximum.
        let allDecisive = ArenaEloStats.summary(wins: 50, draws: 0, losses: 50)
        let drawHeavy   = ArenaEloStats.summary(wins: 20, draws: 60, losses: 20)
        let seDecisive = (allDecisive.scoreHi - allDecisive.scoreLo) / 2
        let seDrawHeavy = (drawHeavy.scoreHi - drawHeavy.scoreLo) / 2
        XCTAssertGreaterThan(seDecisive, seDrawHeavy,
            "draws should tighten the CI — empirical variance depends on outcome distribution, not Bernoulli(p)")
    }

    func testSummaryClampsBoundsToUnitInterval() {
        // Extreme score with few games: Wald CI can run outside
        // [0, 1]. Must be clamped so the displayed interval is
        // well-formed.
        let s = ArenaEloStats.summary(wins: 10, draws: 0, losses: 0)
        XCTAssertGreaterThanOrEqual(s.scoreLo, 0)
        XCTAssertLessThanOrEqual(s.scoreHi, 1)
    }

    func testSummaryZeroGames() {
        // No data — every field safe-defaulted to 0 / nil so
        // downstream formatters don't crash.
        let s = ArenaEloStats.summary(wins: 0, draws: 0, losses: 0)
        XCTAssertEqual(s.games, 0)
        XCTAssertEqual(s.score, 0)
        XCTAssertNil(s.elo)
        XCTAssertNil(s.eloLo)
        XCTAssertNil(s.eloHi)
    }

    func testSummaryOneGame() {
        // n=1 → SE forced to 0 (per implementation: "With fewer
        // than two games the SE collapses to 0"). CI is a point.
        let win = ArenaEloStats.summary(wins: 1, draws: 0, losses: 0)
        XCTAssertEqual(win.score, 1.0)
        XCTAssertEqual(win.scoreLo, 1.0)
        XCTAssertEqual(win.scoreHi, 1.0)
        // p = 1 → elo is nil for both the mid and the (equal) lo/hi.
        XCTAssertNil(win.elo)
        XCTAssertNil(win.eloLo)
        XCTAssertNil(win.eloHi)
    }

    func testSummaryDegenerateClampProducesNilEloEndpoint() {
        // 0W / 0D / 200L → score 0 → elo is nil AT the midpoint,
        // and eloLo is nil. eloHi comes from the clamped upper
        // bound which is still 0 → also nil.
        let s = ArenaEloStats.summary(wins: 0, draws: 0, losses: 200)
        XCTAssertNil(s.elo)
        XCTAssertNil(s.eloLo)
        XCTAssertNil(s.eloHi)
    }

    func testSummaryWorkedExampleFromUserSpec() {
        // User-provided example: 1000 games, 312W / 401D / 287L →
        // score 0.5125, Elo diff ~ +8.7. CI half-width from the
        // variance formula in the ticket.
        let s = ArenaEloStats.summary(wins: 312, draws: 401, losses: 287)
        XCTAssertEqual(s.score, 0.5125, accuracy: 1e-4)
        XCTAssertEqual(s.elo!, 8.7, accuracy: 0.3)
        // CI should straddle zero in this case (not statistically
        // convincing) — matches the user's "Verdict: not
        // statistically convincing" sample line.
        XCTAssertLessThan(s.eloLo!, 0)
        XCTAssertGreaterThan(s.eloHi!, 0)
    }

    // MARK: - Formatters

    func testFormatEloSignedPositive() {
        XCTAssertEqual(ArenaEloStats.formatEloSigned(28), "+28")
        XCTAssertEqual(ArenaEloStats.formatEloSigned(0), "+0")
        XCTAssertEqual(ArenaEloStats.formatEloSigned(27.6), "+28")   // rounds
    }

    func testFormatEloSignedNegative() {
        XCTAssertEqual(ArenaEloStats.formatEloSigned(-28), "-28")
        XCTAssertEqual(ArenaEloStats.formatEloSigned(-27.6), "-28")
    }

    func testFormatEloSignedNil() {
        XCTAssertEqual(ArenaEloStats.formatEloSigned(nil), "—")
    }

    func testFormatEloWithCI() {
        // Construct a summary we control — 60W / 20D / 20L, 100 games.
        let s = ArenaEloStats.summary(wins: 60, draws: 20, losses: 20)
        let str = ArenaEloStats.formatEloWithCI(s)
        // Should match shape "+NNN [+NN, +NNN]" with integers.
        XCTAssertTrue(str.contains("["))
        XCTAssertTrue(str.contains("]"))
        XCTAssertTrue(str.contains(","))
        // Signed integers only — no decimals.
        XCTAssertFalse(str.contains("."))
    }

    func testFormatEloWithCIDegenerateReadsDash() {
        // A 1W / 0D / 0L sample has summary.elo == nil. The
        // formatter should render "—" for that midpoint.
        let s = ArenaEloStats.summary(wins: 1, draws: 0, losses: 0)
        let str = ArenaEloStats.formatEloWithCI(s)
        XCTAssertTrue(str.contains("—"))
    }

    func testFormatScorePercentWithCI() {
        let s = ArenaEloStats.summary(wins: 60, draws: 20, losses: 20)
        let str = ArenaEloStats.formatScorePercentWithCI(s)
        XCTAssertTrue(str.contains("70.0%"))
        XCTAssertTrue(str.hasSuffix("]"))
    }
}
