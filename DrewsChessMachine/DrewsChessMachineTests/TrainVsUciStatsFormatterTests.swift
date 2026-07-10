import XCTest
@testable import DrewsChessMachine

/// Pins the `[VS-UCI-STATS]` block format: per-kind summary lines first
/// (aggregating that kind's instances), then one per-instance line each;
/// cumulative games/plies/W-L-D plus windowed g/s & p/s rates.
final class TrainVsUciStatsFormatterTests: XCTestCase {

    private func slot(
        kind: String, label: String,
        games: Int = 0, plies: Int = 0,
        wins: Int = 0, losses: Int = 0, draws: Int = 0,
        aborted: Int = 0, capDropped: Int = 0
    ) -> TrainVsUciDriver.SlotStats {
        var s = TrainVsUciDriver.SlotStats(kind: kind, instanceLabel: label)
        s.gamesCompleted = games
        s.pliesPlayed = plies
        s.trainerWins = wins
        s.trainerLosses = losses
        s.draws = draws
        s.aborted = aborted
        s.capDropped = capDropped
        return s
    }

    func testKindSummariesPrecedeInstanceLines() {
        let current = [
            slot(kind: "stockfish", label: "stockfish#1", games: 4, plies: 200, wins: 1, losses: 2, draws: 1),
            slot(kind: "stockfish", label: "stockfish#2", games: 6, plies: 300, wins: 2, losses: 3, draws: 1),
            slot(kind: "sloppy", label: "sloppy#1", games: 10, plies: 400, wins: 7, losses: 2, draws: 1),
        ]
        let lines = TrainVsUciStatsFormatter.lines(current: current, previous: [], intervalSec: 10)

        XCTAssertEqual(lines.count, 5, "2 kind summaries + 3 instance lines")
        // Kind order preserves first appearance; summaries come first.
        XCTAssertTrue(lines[0].contains("stockfish n=2:"), "line 0: \(lines[0])")
        XCTAssertTrue(lines[1].contains("sloppy n=1:"), "line 1: \(lines[1])")
        XCTAssertTrue(lines[2].contains("stockfish#1:"))
        XCTAssertTrue(lines[3].contains("stockfish#2:"))
        XCTAssertTrue(lines[4].contains("sloppy#1:"))
    }

    func testKindAggregationSumsInstances() {
        let current = [
            slot(kind: "stockfish", label: "stockfish#1", games: 4, plies: 200, wins: 1, losses: 2, draws: 1),
            slot(kind: "stockfish", label: "stockfish#2", games: 6, plies: 300, wins: 2, losses: 3, draws: 1),
        ]
        let lines = TrainVsUciStatsFormatter.lines(current: current, previous: [], intervalSec: 10)
        XCTAssertTrue(lines[0].contains("games=10"), "aggregate games: \(lines[0])")
        XCTAssertTrue(lines[0].contains("plies=500"), "aggregate plies: \(lines[0])")
        XCTAssertTrue(lines[0].contains("W-L-D=3-5-2"), "aggregate W-L-D: \(lines[0])")
    }

    func testRatesUseWindowDeltaAgainstPrevious() {
        let previous = [slot(kind: "sloppy", label: "sloppy#1", games: 10, plies: 400)]
        let current = [slot(kind: "sloppy", label: "sloppy#1", games: 16, plies: 700)]
        // Δgames=6, Δplies=300 over 12s → 0.50 g/s, 25.00 p/s.
        let lines = TrainVsUciStatsFormatter.lines(current: current, previous: previous, intervalSec: 12)
        XCTAssertTrue(lines[0].contains("g/s=0.50"), "kind rate: \(lines[0])")
        XCTAssertTrue(lines[0].contains("p/s=25.00"), "kind rate: \(lines[0])")
        XCTAssertTrue(lines[1].contains("g/s=0.50"), "instance rate: \(lines[1])")
        XCTAssertTrue(lines[1].contains("p/s=25.00"), "instance rate: \(lines[1])")
        // Cumulative counts still report the totals, not the delta.
        XCTAssertTrue(lines[1].contains("games=16"))
        XCTAssertTrue(lines[1].contains("plies=700"))
    }

    func testAbortAndCapDropAppearOnlyWhenNonzero() {
        let healthy = [slot(kind: "sloppy", label: "sloppy#1", games: 2, plies: 50)]
        let healthyLines = TrainVsUciStatsFormatter.lines(current: healthy, previous: [], intervalSec: 10)
        XCTAssertFalse(healthyLines[0].contains("aborted="))
        XCTAssertFalse(healthyLines[1].contains("capDropped="))

        let troubled = [slot(kind: "sloppy", label: "sloppy#1", games: 2, plies: 50, aborted: 3, capDropped: 1)]
        let troubledLines = TrainVsUciStatsFormatter.lines(current: troubled, previous: [], intervalSec: 10)
        XCTAssertTrue(troubledLines[0].contains("aborted=3"))
        XCTAssertTrue(troubledLines[0].contains("capDropped=1"))
        XCTAssertTrue(troubledLines[1].contains("aborted=3"))
        XCTAssertTrue(troubledLines[1].contains("capDropped=1"))
    }

    func testEmptyStatsProduceNoLines() {
        XCTAssertEqual(TrainVsUciStatsFormatter.lines(current: [], previous: [], intervalSec: 10), [])
    }
}
