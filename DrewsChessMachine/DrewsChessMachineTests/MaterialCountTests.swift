import XCTest
@testable import DrewsChessMachine

/// `MaterialCount`: standard points per side (pawn 1, knight 3, bishop 3,
/// rook 5, queen 9, king 0) and the lead from each side's view.
final class MaterialCountTests: XCTestCase {

    func testStartingPositionIsThirtyNineEach() {
        let material = MaterialCount(GameState.starting)
        XCTAssertEqual(material, MaterialCount(white: 39, black: 39))
        XCTAssertEqual(material.lead(for: .white), 0)
        XCTAssertEqual(material.lead(for: .black), 0)
    }

    /// The final position of Lichess game qgwVLXEV: a bare white king
    /// against rook, rook, queen, two bishops, a knight and eight pawns.
    func testBareKingAgainstEverything() throws {
        let state = try FENParser.parse("4r1k1/p4p1p/2p4p/1p1r4/5p2/b5q1/p7/n4K1b w - - 0 32")
        let material = MaterialCount(state)
        XCTAssertEqual(material.points(for: .white), 0)
        XCTAssertEqual(material.points(for: .black), 36)
        XCTAssertEqual(material.lead(for: .black), 36)
        XCTAssertEqual(material.lead(for: .white), -36)
    }
}
