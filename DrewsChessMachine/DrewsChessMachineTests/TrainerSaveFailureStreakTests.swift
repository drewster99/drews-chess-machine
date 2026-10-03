import XCTest
@testable import DrewsChessMachine

/// The save-failure rule corpus replay and train-vs-UCI share
/// (`TrainerSaveFailureStreak`): one failure is a warning, a second in a row
/// halts the run, and a failed last save fails it.
final class TrainerSaveFailureStreakTests: XCTestCase {

    func testOneFailureIsToleratedAndTheNextInARowHalts() throws {
        var streak = TrainerSaveFailureStreak(what: "trainer-model save")
        XCTAssertNoThrow(try streak.recordFailure(step: 1000))
        XCTAssertThrowsError(try streak.recordFailure(step: 2000))
    }

    func testALastSaveSucceededWhenNoSaveHasFailedSinceTheLastSuccess() throws {
        var streak = TrainerSaveFailureStreak(what: "trainer-model save")
        XCTAssertNoThrow(try streak.requireLastSaveSucceeded(step: 0, reason: "final"))
        try streak.recordFailure(step: 1000)
        streak.recordSuccess()
        XCTAssertNoThrow(try streak.requireLastSaveSucceeded(step: 2000, reason: "final"),
                         "an earlier failure that a later save recovered from is not a lost end state")
    }

    func testALastSaveThatFailedFailsTheRun() throws {
        var streak = TrainerSaveFailureStreak(what: "session save")
        try streak.recordFailure(step: 1234)
        XCTAssertThrowsError(try streak.requireLastSaveSucceeded(step: 1234, reason: "vsuci-abort")) { error in
            XCTAssertEqual(error as? TrainerSaveFailureStreak.LastSaveFailedError,
                           .init(what: "session save", reason: "vsuci-abort", step: 1234))
        }
    }
}
