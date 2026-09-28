import XCTest
@testable import DrewsChessMachine

/// Plan §14.3b: decoding an opponent's profile and the Lichess crosstable,
/// and the head-to-head text.
final class LichessBotOpponentProfileTests: XCTestCase {

    /// The API docs' crosstable example.
    func testCrosstableDecodesAndReadsFromOurSide() throws {
        let crosstable = try JSONDecoder().decode(LichessBotCrosstable.self, from: Data(#"{"users":{"drnykterstein":753.5,"rebeccaharris":459.5},"nbGames":1213}"#.utf8))
        XCTAssertEqual(crosstable.nbGames, 1213)
        XCTAssertEqual(
            LichessBotOpponentCardContent.headToHeadText(crosstable, us: "drnykterstein", them: "rebeccaharris", error: nil),
            "Lichess head-to-head: DCM 753.5 – 459.5 over 1213 games"
        )
        let none = LichessBotCrosstable(users: [:], nbGames: 0)
        XCTAssertEqual(LichessBotOpponentCardContent.headToHeadText(none, us: "a", them: "b", error: nil), "No games against DCM on Lichess")
        XCTAssertEqual(LichessBotOpponentCardContent.headToHeadText(nil, us: "a", them: "b", error: "HTTP 500"), "Lichess head-to-head unavailable: HTTP 500")
    }

    /// A `GET /api/user` payload with the card's fields, and one without any
    /// of the optional ones.
    func testUserDecodesTheCardFields() throws {
        let full = #"""
        {"id":"maia1","username":"maia1","title":"BOT","verified":true,"patronColor":3,"flair":"nature.seedling","createdAt":1582579972726,"seenAt":1789845689010,
         "perfs":{"blitz":{"games":437509,"rating":1352,"rd":45,"prog":-11,"rank":812}},
         "count":{"all":1716000,"rated":1500000,"ai":0,"draw":1000,"drawH":900,"loss":2000,"lossH":1800,"win":3000,"winH":2900,"bookmark":0,"playing":2,"import":0,"me":0},
         "playTime":{"total":1353611440,"tv":3032114,"human":1084283226},
         "profile":{"bio":"Maia is a human-like neural network chess engine.","realName":"Maia Chess 1100","location":"Toronto","fideRating":1100}}
        """#
        let user = try JSONDecoder().decode(LichessBotUserSummary.self, from: Data(full.utf8))
        XCTAssertEqual(user.verified, true)
        XCTAssertEqual(user.patronColor, 3)
        XCTAssertEqual(user.rating("blitz")?.rank, 812)
        XCTAssertEqual(user.count?.winH, 2900)
        XCTAssertEqual(user.playTime?.total, 1_353_611_440)
        XCTAssertEqual(user.profile?.location, "Toronto")
        XCTAssertEqual(user.profile?.fideRating, 1100)
        XCTAssertEqual(user.bioFirstLine, "Maia is a human-like neural network chess engine.")

        let bare = try JSONDecoder().decode(LichessBotUserSummary.self, from: Data(#"{"id":"x","username":"X"}"#.utf8))
        XCTAssertNil(bare.count)
        XCTAssertNil(bare.profile)
        XCTAssertNil(bare.verified)
    }
}
