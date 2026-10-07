import XCTest
@testable import DrewsChessMachine

/// The bridged `navigationTitle` replaces the window's title, so it has to
/// lead with the window's name.
final class LichessBotWindowTitleTests: XCTestCase {
    func testTheTitleLeadsWithTheWindowName() {
        XCTAssertEqual(LichessBotWindowTitle.text(account: "DrewsChessMachine", connection: .online, gamesInProgress: 2),
                       "Lichess Bot — DrewsChessMachine — Online — 2 games in progress")
        XCTAssertEqual(LichessBotWindowTitle.text(account: "DrewsChessMachine", connection: .offline, gamesInProgress: 1),
                       "Lichess Bot — DrewsChessMachine — Offline — 1 game in progress")
    }
}
