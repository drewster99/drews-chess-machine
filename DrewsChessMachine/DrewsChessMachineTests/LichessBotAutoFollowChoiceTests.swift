import XCTest
@testable import DrewsChessMachine

/// The Live tab's follow rule (`LichessBotAutoFollowChoice`): a game against
/// a human is preferred over bot games, and the followed game never flips
/// back and forth (owner request 2026-10-13).
final class LichessBotAutoFollowChoiceTests: XCTestCase {

    private typealias Choice = LichessBotAutoFollowChoice

    private func bot(_ id: String) -> Choice.Candidate {
        Choice.Candidate(id: id, isHuman: false)
    }

    private func human(_ id: String) -> Choice.Candidate {
        Choice.Candidate(id: id, isHuman: true)
    }

    func testInitialWithNoLiveGamesIsNil() {
        XCTAssertNil(Choice.initial(among: []))
    }

    func testInitialWithOnlyBotGamesIsTheFirstStarted() {
        XCTAssertEqual(Choice.initial(among: [bot("b1"), bot("b2")]), "b1")
    }

    func testInitialPrefersAHumanGameStartedAfterBotGames() {
        XCTAssertEqual(Choice.initial(among: [bot("b1"), bot("b2"), human("h1")]), "h1")
    }

    func testInitialWithTwoHumanGamesIsTheFirstStartedHumanGame() {
        XCTAssertEqual(Choice.initial(among: [bot("b1"), human("h1"), bot("b2"), human("h2")]), "h1")
    }

    /// A human game starting while a bot game is followed takes the view.
    func testAHumanGameTakesTheViewFromAFollowedBotGame() {
        XCTAssertEqual(Choice.next(following: bot("b1"), among: [bot("b1"), human("h1")]), "h1")
    }

    /// A game whose opponent isn't known yet counts as not human, so a
    /// known human game takes the view from it.
    func testAGameWithAnOpponentNotKnownYetGivesWayToAHumanGame() {
        let notKnownYet = Choice.Candidate(id: "u1", isHuman: false)
        XCTAssertEqual(Choice.next(following: notKnownYet, among: [notKnownYet, human("h1")]), "h1")
    }

    /// A followed human game is kept when a bot game starts, including one
    /// that started earlier.
    func testAFollowedHumanGameIsKeptOverBotGames() {
        XCTAssertEqual(Choice.next(following: human("h1"), among: [bot("b0"), human("h1"), bot("b2")]), "h1")
    }

    /// Two human games never trade places: the followed one stays, even
    /// when the other started first.
    func testAFollowedHumanGameIsKeptOverAnotherHumanGame() {
        XCTAssertEqual(Choice.next(following: human("h2"), among: [human("h1"), human("h2")]), "h2")
        XCTAssertEqual(Choice.next(following: human("h1"), among: [human("h1"), human("h2")]), "h1")
    }

    /// A new bot game never takes the view from a followed bot game.
    func testAFollowedBotGameIsKeptOverOtherBotGames() {
        XCTAssertEqual(Choice.next(following: bot("b2"), among: [bot("b1"), bot("b2"), bot("b3")]), "b2")
    }

    /// A followed bot game gives way to the first-started human game when
    /// several are live.
    func testAFollowedBotGameGivesWayToTheFirstStartedHumanGame() {
        XCTAssertEqual(Choice.next(following: bot("b1"), among: [bot("b1"), human("h1"), human("h2")]), "h1")
    }

    /// Once a human game is followed, applying the rule again keeps it: the
    /// view settles in one step and doesn't oscillate.
    func testTheRuleSettlesInOneStep() {
        let live = [bot("b1"), human("h1"), bot("b2"), human("h2")]
        let first = Choice.next(following: bot("b1"), among: live)
        XCTAssertEqual(first, "h1")
        XCTAssertEqual(Choice.next(following: human(first), among: live), first)
    }
}
