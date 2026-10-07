import XCTest
@testable import DrewsChessMachine

/// The sequential test sees the arena's games in the order they started, so
/// short games finishing first cannot decide it (`ArenaSPRTStartOrderFeed`).
final class ArenaSPRTStartOrderFeedTests: XCTestCase {

    private typealias Feed = ArenaSPRTStartOrderFeed

    private func config() throws -> ArenaSPRT.SPRTConfig {
        try ArenaSPRT.SPRTConfig(elo0: 0, elo1: 10, alpha: 0.05, beta: 0.05, minGames: 32, maxGames: 20000)
    }

    func testResultsAreHeldUntilEveryEarlierGameHasFinished() {
        var feed = Feed()
        XCTAssertEqual(feed.record(.draw, gameIndex: 2), [])
        XCTAssertEqual(feed.record(.loss, gameIndex: 1), [])
        XCTAssertEqual(feed.heldCount, 2)
        XCTAssertEqual(feed.record(.win, gameIndex: 0), [
            Feed.Tally(wins: 1, draws: 0, losses: 0),
            Feed.Tally(wins: 1, draws: 0, losses: 1),
            Feed.Tally(wins: 1, draws: 1, losses: 1),
        ])
        XCTAssertEqual(feed.heldCount, 0)
        XCTAssertEqual(feed.nextGameIndex, 3)
        XCTAssertEqual(feed.record(.draw, gameIndex: 4), [], "game 3 is still being played")
        XCTAssertEqual(feed.record(.win, gameIndex: 3), [
            Feed.Tally(wins: 2, draws: 1, losses: 1),
            Feed.Tally(wins: 2, draws: 2, losses: 1),
        ])
    }

    func testInOrderFinishesAreReleasedOneAtATime() {
        var feed = Feed()
        XCTAssertEqual(feed.record(.win, gameIndex: 0), [Feed.Tally(wins: 1, draws: 0, losses: 0)])
        XCTAssertEqual(feed.record(.draw, gameIndex: 1), [Feed.Tally(wins: 1, draws: 1, losses: 0)])
    }

    /// A strong candidate's arena where a game's length follows its result,
    /// as on 2026-10-07: draws end first, then losses, then wins. In start
    /// order every ten games are six wins, three draws and one loss (score
    /// 0.75). Fed in finishing order, as the driver did, the test sees only
    /// draws and losses first and rejects; fed through the start-order
    /// feed, it decides on an unbiased prefix and accepts.
    func testShortDrawsFinishingFirstNoLongerDecideTheTest() throws {
        let gameCount = 400
        func outcome(_ index: Int) -> Feed.Outcome {
            switch index % 10 {
            case 0..<6: return .win
            case 6..<9: return .draw
            default: return .loss
            }
        }
        func length(_ outcome: Feed.Outcome) -> Int {
            switch outcome {
            case .draw: return 0
            case .loss: return 1
            case .win: return 2
            }
        }
        // All 400 in flight at once; they finish shortest first, start order
        // within a length.
        let finishingOrder = (0..<gameCount).sorted { lhs, rhs in
            let (l, r) = (length(outcome(lhs)), length(outcome(rhs)))
            return l != r ? l < r : lhs < rhs
        }

        // What the driver did: the running tally in finishing order.
        var finishingOrderMonitor = ArenaSPRT.Monitor(config: try config())
        var wins = 0, draws = 0, losses = 0
        for index in finishingOrder {
            switch outcome(index) {
            case .win: wins += 1
            case .draw: draws += 1
            case .loss: losses += 1
            }
            finishingOrderMonitor.observeCompletedGame(wins: wins, draws: draws, losses: losses)
        }
        let biased = try XCTUnwrap(finishingOrderMonitor.verdict)
        XCTAssertEqual(biased.decision, .reject, "the bug this guards against")
        XCTAssertEqual(biased.wins, 0)

        // The fix: the same finishing order through the start-order feed.
        var monitor = ArenaSPRT.Monitor(config: try config())
        var feed = Feed()
        for index in finishingOrder {
            for tally in feed.record(outcome(index), gameIndex: index) {
                monitor.observeCompletedGame(wins: tally.wins, draws: tally.draws, losses: tally.losses)
            }
        }
        let verdict = try XCTUnwrap(monitor.verdict)
        XCTAssertEqual(verdict.decision, .accept)
        // The sample is exactly the first `gamesAtDecision` games in start
        // order.
        let prefix = (0..<verdict.gamesAtDecision).map(outcome)
        XCTAssertEqual(verdict.wins, prefix.filter { $0 == .win }.count)
        XCTAssertEqual(verdict.draws, prefix.filter { $0 == .draw }.count)
        XCTAssertEqual(verdict.losses, prefix.filter { $0 == .loss }.count)
    }
}
