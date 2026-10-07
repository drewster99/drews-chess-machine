import XCTest
@testable import DrewsChessMachine

/// The Elo arithmetic behind Perf, the 50% point and the score interval
/// (`LICHESS_BOT_RECORD_STATS_PLAN.md` §3.4, §3.7, R-1).
final class LichessBotEloMathTests: XCTestCase {

    private func value(_ estimate: LichessBotRatingEstimate, file: StaticString = #filePath, line: UInt = #line) -> Double? {
        switch estimate {
        case .estimate(let value), .atLeast(let value), .atMost(let value):
            return value
        case .none:
            XCTFail("expected a value, got none", file: file, line: line)
            return nil
        }
    }

    /// The closed form for one opponent rating: `R₀ + 400·log₁₀(p/(1−p))`.
    private func closedForm(_ rating: Double, _ p: Double) -> Double {
        rating + 400 * log10(p / (1 - p))
    }

    func testExpectedScoreIsSymmetricAndFourHundredPointsIsTenToOne() {
        XCTAssertEqual(LichessBotEloMath.expectedScore(rating: 1500, opponent: 1500), 0.5, accuracy: 1e-12)
        let higher = LichessBotEloMath.expectedScore(rating: 1900, opponent: 1500)
        let lower = LichessBotEloMath.expectedScore(rating: 1500, opponent: 1900)
        XCTAssertEqual(higher + lower, 1, accuracy: 1e-12)
        XCTAssertEqual(higher / lower, 10, accuracy: 1e-9)
        XCTAssertEqual(higher, 10.0 / 11.0, accuracy: 1e-12)
    }

    func testSingleOpponentMatchesTheClosedForm() throws {
        // Four games against 1600 at p = 0.25, 0.5, 0.75.
        for (score, p) in [(1.0, 0.25), (2.0, 0.5), (3.0, 0.75)] {
            guard case .estimate(let r) = LichessBotEloMath.performanceRating(opponents: [1600, 1600, 1600, 1600], score: score) else {
                return XCTFail("expected an estimate at p = \(p)")
            }
            XCTAssertEqual(r, closedForm(1600, p), accuracy: 1e-9, "p = \(p)")
        }
    }

    func testMixedOpponentsSolveTheMaximumLikelihoodEquation() throws {
        let opponents = [1400, 1550, 1700, 1900, 2100]
        let score = 2.5
        guard case .estimate(let r) = LichessBotEloMath.performanceRating(opponents: opponents, score: score) else {
            return XCTFail("expected an estimate")
        }
        // Σ E(R, Rᵢ) = S at the root, within the solver's tolerance.
        let expected = opponents.reduce(0.0) { $0 + LichessBotEloMath.expectedScore(rating: r, opponent: Double($1)) }
        XCTAssertEqual(expected, score, accuracy: 1e-4)
        // Hand-computed root for this set (bisection in Python to 1e-9).
        XCTAssertEqual(r, 1723.773, accuracy: 0.01)
        // A half-point score against a symmetric field is its mean.
        guard case .estimate(let symmetric) = LichessBotEloMath.performanceRating(opponents: [1400, 1600], score: 1) else {
            return XCTFail("expected an estimate")
        }
        XCTAssertEqual(symmetric, 1500, accuracy: LichessBotEloMath.solverTolerance)
    }

    func testPerfectAndZeroScoresAreBoundsByTheHalfPointConvention() throws {
        // All wins against 1600 ×4: ≥ the rating 3.5/4 would give.
        XCTAssertEqual(LichessBotEloMath.performanceRating(opponents: [1600, 1600, 1600, 1600], score: 4), .atLeast(closedForm(1600, 3.5 / 4)))
        XCTAssertEqual(LichessBotEloMath.performanceRating(opponents: [1600, 1600, 1600, 1600], score: 0), .atMost(closedForm(1600, 0.5 / 4)))
        // One win: ≥ the opponent's rating; one loss: ≤ it; one draw: = it.
        XCTAssertEqual(LichessBotEloMath.performanceRating(opponents: [1700], score: 1), .atLeast(1700))
        XCTAssertEqual(LichessBotEloMath.performanceRating(opponents: [1700], score: 0), .atMost(1700))
        XCTAssertEqual(LichessBotEloMath.performanceRating(opponents: [1700], score: 0.5), .estimate(1700))
    }

    func testNoGamesIsNone() {
        XCTAssertEqual(LichessBotEloMath.performanceRating(opponents: [], score: 0), .none)
        XCTAssertEqual(LichessBotEloMath.ratingOffset(gaps: [], score: 0), .none)
        XCTAssertNil(LichessBotEloMath.scoreInterval(score: 0, games: 0))
    }

    func testEveryOpponentAtOneRatingIsExact() {
        // Zero-width bracket: the root is R₀ + L exactly.
        let opponents = Array(repeating: 1823, count: 7)
        XCTAssertEqual(LichessBotEloMath.performanceRating(opponents: opponents, score: 5), .estimate(closedForm(1823, 5.0 / 7)))
    }

    func testTheDataDerivedBracketHoldsTheRootAtExtremes() throws {
        // n = 100,000 at a half-point score, ratings spread wide: a fixed
        // ±2000 bracket would miss; the data-derived one cannot.
        var opponents: [Int] = []
        for index in 0..<100_000 {
            opponents.append(index % 2 == 0 ? 400 : 3200)
        }
        guard case .estimate(let r) = LichessBotEloMath.performanceRating(opponents: opponents, score: 50_000) else {
            return XCTFail("expected an estimate")
        }
        XCTAssertEqual(r, 1800, accuracy: LichessBotEloMath.solverTolerance)
        // Nearly every game won against a narrow field.
        guard case .estimate(let high) = LichessBotEloMath.performanceRating(opponents: Array(repeating: 1500, count: 100_000), score: 99_999.5) else {
            return XCTFail("expected an estimate")
        }
        XCTAssertEqual(high, closedForm(1500, 99_999.5 / 100_000), accuracy: 1e-6)
        // Extreme ratings.
        guard case .estimate(let extreme) = LichessBotEloMath.performanceRating(opponents: [-500, 4000], score: 1) else {
            return XCTFail("expected an estimate")
        }
        XCTAssertEqual(extreme, 1750, accuracy: LichessBotEloMath.solverTolerance)
    }

    func testPerformanceRisesWithTheScore() throws {
        let opponents = [1400, 1500, 1650, 1800, 1950, 2000]
        var previous = -Double.infinity
        for halfPoints in 1..<12 {
            let r = try XCTUnwrap(value(LichessBotEloMath.performanceRating(opponents: opponents, score: Double(halfPoints) / 2)))
            XCTAssertGreaterThan(r, previous)
            previous = r
        }
    }

    func testRatingOffsetIsZeroWhenResultsEqualExpectations() throws {
        // Two games: gap +200 (expected E) and gap −200 (expected 1 − E);
        // a total score of 1 is exactly the sum of expectations.
        let offset = try XCTUnwrap(value(LichessBotEloMath.ratingOffset(gaps: [200, -200], score: 1)))
        XCTAssertEqual(offset, 0, accuracy: LichessBotEloMath.solverTolerance)
        // Scoring 50% against +100 opponents means the 50% point is +100.
        let plus = try XCTUnwrap(value(LichessBotEloMath.ratingOffset(gaps: [100, 100], score: 1)))
        XCTAssertEqual(plus, 100, accuracy: 1e-9)
    }

    func testWilsonIntervalStaysHonestAtSmallN() throws {
        // All wins in two games: not "100% ± 0".
        let twoWins = try XCTUnwrap(LichessBotEloMath.scoreInterval(score: 2, games: 2))
        XCTAssertEqual(twoWins.upper, 1, accuracy: 1e-12)
        XCTAssertLessThan(twoWins.lower, 0.4)
        XCTAssertGreaterThan(twoWins.lower, 0.3)
        // Hand-computed Wilson bounds for 7/10.
        let seven = try XCTUnwrap(LichessBotEloMath.scoreInterval(score: 7, games: 10))
        XCTAssertEqual(seven.lower, 0.3968, accuracy: 1e-4)
        XCTAssertEqual(seven.upper, 0.8922, accuracy: 1e-4)
        // Draws count as half points: 5 draws in 10 games = 0.5.
        let draws = try XCTUnwrap(LichessBotEloMath.scoreInterval(score: 5, games: 10))
        XCTAssertEqual((draws.lower + draws.upper) / 2, 0.5, accuracy: 1e-12)
        // Narrows as n grows; always inside 0…1.
        let large = try XCTUnwrap(LichessBotEloMath.scoreInterval(score: 500, games: 1000))
        XCTAssertLessThan(large.upper - large.lower, draws.upper - draws.lower)
        let none = try XCTUnwrap(LichessBotEloMath.scoreInterval(score: 0, games: 3))
        XCTAssertEqual(none.lower, 0, accuracy: 1e-12)
        XCTAssertGreaterThan(none.upper, 0.4)
    }
}
