import XCTest
@testable import DrewsChessMachine

/// Plan §7.2: the bot-limit refusal parse, player notes persistence, and
/// the Challenge sheet's filtering and ordering.
final class LichessBotBotListTests: XCTestCase {

    // MARK: - Refusal parsing

    func testParsesTheObservedRefusal() throws {
        let text = "bernstein-4ply played 100 games against other bots today, please wait until 2026-09-29T06:57:07.895Z to challenge them."
        let parsed = try XCTUnwrap(LichessBotBotLimitRefusal.parse(text))
        XCTAssertEqual(parsed.userID, "bernstein-4ply")
        XCTAssertEqual(parsed.gamesPlayed, 100)
        XCTAssertEqual(parsed.until.timeIntervalSince1970, 1_790_665_027.895, accuracy: 0.001)
    }

    func testParsesWholeSecondsAndAnotherCountAndLowercasesTheID() throws {
        let parsed = try XCTUnwrap(LichessBotBotLimitRefusal.parse("SomeBot played 150 games against other bots today, please wait until 2026-09-29T06:57:07Z to challenge them."))
        XCTAssertEqual(parsed.userID, "somebot")
        XCTAssertEqual(parsed.gamesPlayed, 150)
    }

    func testOtherMessagesDoNotParse() {
        XCTAssertNil(LichessBotBotLimitRefusal.parse("Not your turn, or game already over"))
        XCTAssertNil(LichessBotBotLimitRefusal.parse("x played 100 games against other bots today, please wait until soon to challenge them."))
    }

    // MARK: - Notes

    func testNotesRoundTripAndMissingFileIsEmpty() throws {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent("LichessBotBotListTests-\(UUID().uuidString)", isDirectory: true)
        addTeardownBlock {
            do {
                try FileManager.default.removeItem(at: folder)
            } catch {
                XCTFail("cleanup failed: \(error)")
            }
        }
        let url = folder.appendingPathComponent("player-notes.json")
        XCTAssertEqual(try LichessBotPlayerNotes.load(from: url), LichessBotPlayerNotes())

        var notes = LichessBotPlayerNotes()
        notes.toggleFavorite("SomeBot")
        notes.toggleFavorite("other")
        notes.botLimitUntil["somebot"] = Date(timeIntervalSince1970: 2_000_000_000)
        try notes.save(to: url)
        let loaded = try LichessBotPlayerNotes.load(from: url)
        XCTAssertEqual(loaded.favoriteIDs, ["somebot", "other"])
        XCTAssertEqual(loaded.botLimitUntil["somebot"], Date(timeIntervalSince1970: 2_000_000_000))

        notes.toggleFavorite("SOMEBOT")
        XCTAssertEqual(notes.favoriteIDs, ["other"])
    }

    func testExpiredLimitsAreIgnoredAndPruned() {
        var notes = LichessBotPlayerNotes()
        let now = Date(timeIntervalSince1970: 1_000)
        notes.botLimitUntil = ["past": Date(timeIntervalSince1970: 999), "future": Date(timeIntervalSince1970: 1_001)]
        XCTAssertNil(notes.limitUntil("past", now: now))
        XCTAssertNotNil(notes.limitUntil("FUTURE", now: now))
        notes.pruneExpiredLimits(now: now)
        XCTAssertEqual(Array(notes.botLimitUntil.keys), ["future"])
    }

    // MARK: - Filtering and ordering

    private func bot(_ name: String, blitz: Int?, provisional: Bool = false, bio: String? = nil) throws -> LichessBotUserSummary {
        var perfs: [String: Any] = [:]
        if let blitz {
            var perf: [String: Any] = ["rating": blitz, "games": 50, "rd": 60, "prog": 0]
            if provisional { perf["prov"] = true }
            perfs["blitz"] = perf
        }
        var object: [String: Any] = ["id": name.lowercased(), "username": name, "title": "BOT", "perfs": perfs]
        if let bio { object["profile"] = ["bio": bio] }
        return try JSONDecoder().decode(LichessBotUserSummary.self, from: JSONSerialization.data(withJSONObject: object))
    }

    func testRatingRangeAppliesToTheSelectedSpeedAndHidesProvisional() throws {
        let rows = LichessBotBotList.onlineRows(
            bots: [try bot("Low", blitz: 1200), try bot("Mid", blitz: 1800), try bot("Prov", blitz: 1800, provisional: true), try bot("Unrated", blitz: nil)],
            notes: LichessBotPlayerNotes(),
            now: Date()
        )
        var filter = LichessBotBotListFilter(minimumRating: 1500, maximumRating: 2000, speed: .blitz)
        XCTAssertEqual(rows.filter(filter.matches).map(\.username), ["Mid", "Prov"])
        filter.hidesProvisional = true
        XCTAssertEqual(rows.filter(filter.matches).map(\.username), ["Mid"])
        filter = LichessBotBotListFilter(minimumRating: 1000, speed: .rapid)
        XCTAssertEqual(rows.filter(filter.matches).map(\.username), [], "no rapid ratings, so nothing matches a rapid bound")
    }

    func testSearchCoversNameAndBio() throws {
        let rows = LichessBotBotList.onlineRows(
            bots: [try bot("maia1", blitz: 1300, bio: "A human-like neural network"), try bot("stockbot", blitz: 2000)],
            notes: nil,
            now: Date()
        )
        XCTAssertEqual(rows.filter(LichessBotBotListFilter(search: "NEURAL").matches).map(\.username), ["maia1"])
        XCTAssertEqual(rows.filter(LichessBotBotListFilter(search: "stock").matches).map(\.username), ["stockbot"])
    }

    func testFavoritesSortFirstAndOfflineFavoritesAppearFromStatus() throws {
        var notes = LichessBotPlayerNotes()
        notes.toggleFavorite("zeta")
        notes.toggleFavorite("gone")
        let bots = [try bot("alpha", blitz: 1500), try bot("zeta", blitz: 1400)]
        let ordered = LichessBotBotList.ordered(
            LichessBotBotList.onlineRows(bots: bots, notes: notes, now: Date()),
            filter: LichessBotBotListFilter(),
            sortOrder: [KeyPathComparator(\LichessBotBotRow.usernameSortKey)]
        )
        XCTAssertEqual(ordered.map(\.username), ["zeta", "alpha"])

        let statuses = ["gone": LichessBotUserStatus(id: "gone", name: "Gone", title: "BOT", online: nil, playing: nil)]
        let favorites = LichessBotBotList.favoriteRows(notes: notes, bots: bots, statuses: statuses, now: Date())
        XCTAssertEqual(favorites.map(\.username), ["zeta", "Gone"])
        XCTAssertEqual(favorites.map(\.isOnline), [true, false])
    }

    // MARK: - Our own bot-game count

    private func row(_ id: String, kind: LichessBotOpponentKind, at date: Date) throws -> LichessBotGameSummary {
        let json = """
        {"gameID":"\(id)","createdAt":"\(date.formatted(Date.ISO8601FormatStyle(includingFractionalSeconds: true)))","speed":"blitz","rated":false,"ourColor":"white","opponentKind":"\(kind.rawValue)","status":"mate","plies":40,"modelIDs":[],"sourceKinds":[],"builds":[],"reconciliation":"matched","anomalyCount":0,"ourScore":1}
        """
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601withFractionalSeconds
        return try decoder.decode(LichessBotGameSummary.self, from: Data(json.utf8))
    }

    func testBotGamesCountsOnlyBotsSinceTheCutoff() throws {
        let rows = [
            try row("a", kind: .bot, at: Date(timeIntervalSince1970: 100)),
            try row("b", kind: .bot, at: Date(timeIntervalSince1970: 50)),
            try row("c", kind: .human, at: Date(timeIntervalSince1970: 100)),
        ]
        XCTAssertEqual(LichessBotRecordSummary.botGames(rows: rows, since: Date(timeIntervalSince1970: 60)), 1)
    }
}
