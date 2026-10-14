import XCTest
@testable import DrewsChessMachine

/// A refused (or otherwise not created) outgoing challenge names its
/// opponent as DCM knew them at the send, so its Challenge Log row shows the
/// name, BOT badge and rating; older lines fall back to the id and the kind.
final class LichessBotNotCreatedOpponentTests: XCTestCase {

    private typealias F = LichessBotChallengeLogFixtures

    private static let requestJSON = #"{"clockIncrementSeconds":2,"clockLimitSeconds":180,"color":"random","rated":true}"#

    /// A line as builds before 2026-10-14 wrote it: no `opponent` key.
    private static let oldLine = #"{"at":"2026-10-05T12:00:00.000Z","build":2400,"event":{"outgoingNotCreated":{"attemptID":"00000000-0000-0000-0000-000000000001","creditCost":1,"opponentID":"dala-700","opponentKind":{"bot":{}},"reason":{"refused":{"_0":{"httpStatus":400,"kind":"botDailyGameLimit","text":"dala-700 played 100 games against other bots today, please wait until 2026-10-14T07:26:24.461Z to challenge them."}}},"request":"# + requestJSON + #","sender":{"matchmaking":{"fillMode":"everyFreeSlot","trigger":"automaticPass"}}}},"gitHash":"0123abc","schemaVersion":1}"#

    private static let dala = LichessBotChallengeParty(id: "dala-700", name: "dala-700", title: "BOT", rating: 837, provisional: false)

    private func decode(_ line: String) throws -> LichessBotChallengeLogEntry {
        try LichessBotJSONLines.makeDecoder().decode(LichessBotChallengeLogEntry.self, from: Data(line.utf8))
    }

    private func notCreated(opponentKind: LichessBotChallengeOpponentKind?, opponent: LichessBotChallengeParty?) -> LichessBotChallengeLogEvent {
        .outgoingNotCreated(attemptID: F.attemptID(1), opponentID: "dala-700", sender: F.automaticPass, request: F.request,
                            opponentKind: opponentKind, reason: .refused(LichessBotChallengeRefusal(kind: .botDailyGameLimit, httpStatus: 400, text: nil)),
                            creditCost: 1, opponent: opponent)
    }

    private func onlyRow(_ event: LichessBotChallengeLogEvent) throws -> LichessBotChallengeLogRow {
        var ledger = LichessBotChallengeLedger(loadStatus: .complete)
        ledger.apply(contentsOf: [F.entry(event, at: 0)])
        let rows = LichessBotChallengeLogRow.rows(ledger: ledger, reconstruction: nil, pendingChallengeIDs: [])
        XCTAssertEqual(rows.count, 1)
        return try XCTUnwrap(rows.first)
    }

    func testAnOldLineDecodesWithNoOpponent() throws {
        let entry = try decode(Self.oldLine)
        guard case .outgoingNotCreated(_, let opponentID, _, _, let kind, _, _, let opponent) = entry.event else {
            return XCTFail("expected outgoingNotCreated")
        }
        XCTAssertEqual(opponentID, "dala-700")
        XCTAssertEqual(kind, .bot)
        XCTAssertNil(opponent)
    }

    func testANewLineRoundTripsItsOpponent() throws {
        let entry = F.entry(notCreated(opponentKind: .bot, opponent: Self.dala), at: 0)
        let data = try LichessBotJSONLines.encodeLine(entry)
        XCTAssertTrue(String(decoding: data, as: UTF8.self).contains(#""opponent":{"id":"dala-700","name":"dala-700","provisional":false,"rating":837,"title":"BOT"}"#))
        XCTAssertEqual(try LichessBotJSONLines.makeDecoder().decode(LichessBotChallengeLogEntry.self, from: data), entry)
    }

    func testANoOpponentEventWritesNoKey() throws {
        let data = try LichessBotJSONLines.encodeLine(F.entry(notCreated(opponentKind: .bot, opponent: nil), at: 0))
        XCTAssertFalse(String(decoding: data, as: UTF8.self).contains(#""opponent""#), "an older build's lines and this one's agree when nothing is known")
    }

    func testRowShowsTheRecordedOpponent() throws {
        let row = try onlyRow(notCreated(opponentKind: .bot, opponent: Self.dala))
        XCTAssertEqual(row.opponent, .player(.init(id: "dala-700", name: "dala-700", title: "BOT", rating: 837)))
        XCTAssertEqual(LichessBotChallengeLogStyle.ratingText(row.opponent), " 837")
    }

    func testOldRowFallsBackToTheIDWithTheKindBadge() throws {
        let bot = try onlyRow(try decode(Self.oldLine).event)
        XCTAssertEqual(bot.opponent, .player(.init(id: "dala-700", name: "dala-700", title: "BOT", rating: nil)))
        let human = try onlyRow(notCreated(opponentKind: .human, opponent: nil))
        XCTAssertEqual(human.opponent, .player(.init(id: "dala-700", name: "dala-700", title: nil, rating: nil)))
        let unknown = try onlyRow(notCreated(opponentKind: nil, opponent: nil))
        XCTAssertEqual(unknown.opponent, .player(.init(id: "dala-700", name: "dala-700", title: nil, rating: nil)))
    }

    func testPartyAtSendTakesTheRatingAtTheChallengesSpeed() throws {
        let status = try JSONDecoder().decode(LichessBotUserStatus.self, from: Data(#"{"id":"Dala-700","name":"dala-700","title":"BOT","online":true}"#.utf8))
        let summary = try JSONDecoder().decode(LichessBotUserSummary.self, from: Data(#"{"id":"dala-700","username":"dala-700","title":"BOT","perfs":{"blitz":{"rating":837,"games":500,"prov":false},"bullet":{"rating":900,"games":10,"prov":true}}}"#.utf8))
        XCTAssertEqual(LichessBotChallengeParty(sentTo: status, summary: summary, speed: .blitz), Self.dala)
        XCTAssertEqual(LichessBotChallengeParty(sentTo: status, summary: summary, speed: .bullet).rating, 900)
        XCTAssertEqual(LichessBotChallengeParty(sentTo: status, summary: nil, speed: .blitz),
                       LichessBotChallengeParty(id: "dala-700", name: "dala-700", title: "BOT", rating: nil, provisional: nil))
    }
}
