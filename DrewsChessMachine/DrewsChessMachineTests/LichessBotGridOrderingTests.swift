import XCTest
@testable import DrewsChessMachine

/// The live grid's ordering, position hold and result-highlight phases.
final class LichessBotGridOrderingTests: XCTestCase {
    private let base = Date(timeIntervalSince1970: 1_800_000_000)
    private let hold = LichessBotGridOrdering.positionHold
    private let highlight = LichessBotGridOrdering.resultHighlight

    private func entry(_ id: String, started: TimeInterval, finished: TimeInterval?) -> LichessBotGridOrdering.Entry {
        LichessBotGridOrdering.Entry(id: id, startedAt: base.addingTimeInterval(started), finishedAt: finished.map { base.addingTimeInterval($0) })
    }

    private func orderedIDs(_ entries: [LichessBotGridOrdering.Entry], at seconds: TimeInterval) -> [String] {
        LichessBotGridOrdering.orderedIndices(entries, now: base.addingTimeInterval(seconds)).map { entries[$0].id }
    }

    func testTheHoldIsShorterThanTheHighlight() {
        XCTAssertLessThan(hold, highlight)
    }

    func testLiveByStartThenFinishedMostRecentFirst() {
        let entries = [
            entry("f-old", started: 0, finished: 100),
            entry("live-late", started: 50, finished: nil),
            entry("f-new", started: 1, finished: 200),
            entry("live-early", started: 10, finished: nil),
        ]
        XCTAssertEqual(orderedIDs(entries, at: 1000), ["live-early", "live-late", "f-new", "f-old"])
    }

    func testAJustFinishedGameKeepsItsLivePositionUntilTheHoldEnds() {
        let entries = [
            entry("a", started: 0, finished: 100),
            entry("b", started: 10, finished: nil),
            entry("old", started: -50, finished: 20),
        ]
        XCTAssertEqual(orderedIDs(entries, at: 100 + hold - 0.001), ["a", "b", "old"])
        XCTAssertEqual(orderedIDs(entries, at: 100 + hold), ["b", "a", "old"])
    }

    func testAClockOlderThanTheFinishStillHoldsTheGame() {
        let entries = [entry("a", started: 0, finished: 100), entry("b", started: 10, finished: nil)]
        XCTAssertEqual(orderedIDs(entries, at: 50), ["a", "b"])
    }

    func testTiesOrderByID() {
        let entries = [
            entry("z", started: 0, finished: nil),
            entry("y", started: 0, finished: nil),
            entry("q", started: 0, finished: 5),
            entry("p", started: 0, finished: 5),
        ]
        XCTAssertEqual(orderedIDs(entries, at: 1000), ["y", "z", "p", "q"])
    }

    func testPhases() {
        let finished = base
        XCTAssertEqual(LichessBotGridOrdering.phase(finishedAt: nil, now: base), .live)
        XCTAssertEqual(LichessBotGridOrdering.phase(finishedAt: finished, now: base.addingTimeInterval(-1)), .justFinished)
        XCTAssertEqual(LichessBotGridOrdering.phase(finishedAt: finished, now: base), .justFinished)
        XCTAssertEqual(LichessBotGridOrdering.phase(finishedAt: finished, now: base.addingTimeInterval(highlight - 0.001)), .justFinished)
        XCTAssertEqual(LichessBotGridOrdering.phase(finishedAt: finished, now: base.addingTimeInterval(highlight)), .finished)
    }

    func testNextTransition() {
        let finishTimes = [base, base.addingTimeInterval(1)]
        XCTAssertEqual(LichessBotGridOrdering.nextTransition(finishTimes: finishTimes, after: base), base.addingTimeInterval(hold))
        XCTAssertEqual(LichessBotGridOrdering.nextTransition(finishTimes: finishTimes, after: base.addingTimeInterval(hold)), base.addingTimeInterval(1 + hold))
        XCTAssertEqual(LichessBotGridOrdering.nextTransition(finishTimes: finishTimes, after: base.addingTimeInterval(1 + hold)), base.addingTimeInterval(highlight))
        XCTAssertNil(LichessBotGridOrdering.nextTransition(finishTimes: finishTimes, after: base.addingTimeInterval(1 + highlight)))
        XCTAssertNil(LichessBotGridOrdering.nextTransition(finishTimes: [], after: base))
    }

    func testHasTransition() {
        let finishTimes = [base]
        XCTAssertFalse(LichessBotGridOrdering.hasTransition(finishTimes: finishTimes, since: base, through: base.addingTimeInterval(hold - 0.001)))
        XCTAssertTrue(LichessBotGridOrdering.hasTransition(finishTimes: finishTimes, since: base, through: base.addingTimeInterval(hold)))
        XCTAssertFalse(LichessBotGridOrdering.hasTransition(finishTimes: finishTimes, since: base.addingTimeInterval(hold), through: base.addingTimeInterval(highlight - 0.001)))
        XCTAssertTrue(LichessBotGridOrdering.hasTransition(finishTimes: finishTimes, since: base.addingTimeInterval(hold), through: base.addingTimeInterval(highlight)))
    }
}
