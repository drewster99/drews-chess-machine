import XCTest
@testable import DrewsChessMachine

/// Browse-only stepping (plan §14.3a) and the Lichess live-game model that
/// feeds the live views.
@MainActor
final class GameBrowseAndLiveGameTests: XCTestCase {

    // MARK: - GameBrowseCursor

    func testCursorFollowsLiveUntilMoved() {
        let cursor = GameBrowseCursor()
        XCTAssertTrue(cursor.isLive)
        XCTAssertEqual(cursor.displayedPlyCount(totalPlies: 7), 7)
        XCTAssertEqual(cursor.displayedPlyCount(totalPlies: 8), 8, "live follows new moves")
    }

    func testSteppingBackHoldsPositionAsTheGameGrows() {
        var cursor = GameBrowseCursor()
        cursor.stepBack(totalPlies: 10)
        cursor.stepBack(totalPlies: 10)
        XCTAssertEqual(cursor.displayedPlyCount(totalPlies: 10), 8)
        XCTAssertEqual(cursor.displayedPlyCount(totalPlies: 12), 8, "new moves don't pull the view along")
        XCTAssertFalse(cursor.isLive)
    }

    func testSteppingForwardPastTheEndReturnsToLive() {
        var cursor = GameBrowseCursor()
        cursor.select(plyCount: 9, totalPlies: 10)
        cursor.stepForward(totalPlies: 10)
        XCTAssertTrue(cursor.isLive)
        cursor.stepForward(totalPlies: 10)
        XCTAssertTrue(cursor.isLive, "stepping forward while live does nothing")
    }

    func testStartAndLive() {
        var cursor = GameBrowseCursor()
        cursor.goToStart(totalPlies: 5)
        XCTAssertEqual(cursor.displayedPlyCount(totalPlies: 5), 0)
        cursor.stepBack(totalPlies: 5)
        XCTAssertEqual(cursor.displayedPlyCount(totalPlies: 5), 0, "cannot step before the start")
        cursor.goLive()
        XCTAssertTrue(cursor.isLive)
        cursor.goToStart(totalPlies: 0)
        XCTAssertTrue(cursor.isLive, "an empty game's start is its live position")
    }

    func testSelectingTheLatestMoveIsLive() {
        var cursor = GameBrowseCursor()
        cursor.select(plyCount: 6, totalPlies: 6)
        XCTAssertTrue(cursor.isLive)
        cursor.select(plyCount: 3, totalPlies: 6)
        XCTAssertEqual(cursor.viewedPlyCount, 3)
    }

    /// A takeback can shrink the game under a browsed position: the view
    /// shows the last position that still exists.
    func testShrinkingGameClampsTheView() {
        var cursor = GameBrowseCursor()
        cursor.select(plyCount: 9, totalPlies: 10)
        XCTAssertEqual(cursor.displayedPlyCount(totalPlies: 6), 6)
    }

    // MARK: - LichessBotLiveGame

    private func stateLine(_ tokens: [String], status: String = "started", extra: String = "") -> Data {
        Data(#"{"type":"gameState","moves":"\#(tokens.joined(separator: " "))","wtime":170000,"btime":165000,"winc":2000,"binc":2000,"status":"\#(status)"\#(extra)}"#.utf8)
    }

    private func gameFullLine(tokens: [String] = []) -> Data {
        Data(#"{"type":"gameFull","id":"g1","variant":{"key":"standard"},"clock":{"initial":180000,"increment":2000},"speed":"blitz","rated":false,"createdAt":1700000000000,"white":{"id":"drewschessmachine","name":"DrewsChessMachine","title":"BOT","rating":1500},"black":{"id":"alice","name":"Alice","rating":1600},"initialFen":"startpos","state":{"type":"gameState","moves":"\#(tokens.joined(separator: " "))","wtime":180000,"btime":180000,"winc":2000,"binc":2000,"status":"started"}}"#.utf8)
    }

    func testLiveGameReplaysPositionsForBrowsing() throws {
        let game = LichessBotLiveGame(id: "g1", startedAt: Date(), ourAccountID: "drewschessmachine")
        game.apply(.streamLine(gameFullLine(), receivedAt: Date()))
        game.apply(.streamLine(stateLine(["e2e4", "e7e5", "g1f3"]), receivedAt: Date()))
        XCTAssertEqual(game.plies.map(\.san), ["e4", "e5", "Nf3"])
        XCTAssertEqual(game.whiteClockMilliseconds, 170000)
        XCTAssertEqual(game.sideToMove, .black)
        let afterOne = game.state(afterPlies: 1)
        XCTAssertEqual(afterOne.currentPlayer, .black)
        XCTAssertNil(afterOne.board[6 * 8 + 4], "the e2 pawn has left e2 after 1. e4")
        XCTAssertNotNil(game.state(afterPlies: 0).board[6 * 8 + 4])
    }

    /// E1: king-to-rook castling tokens replay; E30: a takeback shrinks the
    /// list and is counted.
    func testCastlingTokenAndTakeback() {
        let game = LichessBotLiveGame(id: "g1", startedAt: Date(), ourAccountID: "drewschessmachine")
        let tokens = ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "e1h1"]
        game.apply(.streamLine(stateLine(tokens), receivedAt: Date()))
        XCTAssertEqual(game.plies.last?.san, "O-O")
        XCTAssertEqual(game.plies.last?.uciAsGiven, "e1h1")
        game.apply(.streamLine(stateLine(Array(tokens.prefix(5))), receivedAt: Date()))
        XCTAssertEqual(game.plies.count, 5)
        XCTAssertEqual(game.retractedPlyCount, 2)
    }

    func testFinishAndOffersAndTranscript() throws {
        let game = LichessBotLiveGame(id: "g1", startedAt: Date(), ourAccountID: "drewschessmachine")
        let full = gameFullLine()
        game.apply(.streamLine(full, receivedAt: Date()))
        game.apply(.gameInfo(try LichessBotGameFull.decodeForTest(full), ourColor: .white))
        game.apply(.streamLine(stateLine(["e2e4", "e7e5"], extra: #","bdraw":true"#), receivedAt: Date()))
        XCTAssertTrue(game.opponentOffersDraw)
        game.apply(.keepAlive(receivedAt: Date()))
        game.apply(.keepAlive(receivedAt: Date()))
        game.apply(.keepAlive(receivedAt: Date()))
        XCTAssertEqual(game.transcript.last?.title, "keep-alive")
        XCTAssertEqual(game.transcript.last?.repeatCount, 3, "consecutive keep-alives collapse into one entry")
        game.apply(.streamLine(stateLine(["e2e4", "e7e5"], status: "resign", extra: #","winner":"white""#), receivedAt: Date()))
        XCTAssertTrue(game.isFinished)
        XCTAssertEqual(game.winner, "white")
        XCTAssertEqual(game.opponent?.name, "Alice")
    }

    func testRequestsAppearInTheTranscript() {
        let game = LichessBotLiveGame(id: "g1", startedAt: Date(), ourAccountID: "drewschessmachine")
        game.applyRequest(LichessBotRequestRecord(
            startedAt: Date(), gameID: "g1", label: "move", method: "POST", path: "/api/bot/game/g1/move/e2e4",
            formFields: [:], status: 400, queuedMilliseconds: 2, roundTripMilliseconds: 55, networkProtocol: "h2",
            errorMessage: "Not your turn", failure: nil
        ))
        let entry = game.transcript.last
        XCTAssertEqual(entry?.direction, .outgoing)
        XCTAssertEqual(entry?.isProblem, true)
        XCTAssertTrue(entry?.detail.contains("Not your turn") == true)
    }
}

private extension LichessBotGameFull {
    static func decodeForTest(_ data: Data) throws -> LichessBotGameFull {
        guard case .gameFull(let full) = try LichessBotGameStreamLine.decode(data) else {
            throw CocoaError(.coderReadCorrupt)
        }
        return full
    }
}
