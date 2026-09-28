import XCTest
@testable import DrewsChessMachine

/// Regression (live, 2026-09-28): the goodbye DCM sends after a game ends
/// never showed in the Chat tab or the record. Lichess echoes our messages
/// on the game stream, but the session stops reading the stream when the
/// game finishes, so the goodbye's echo never arrives. Sent messages are
/// recorded from their successful POST and matched to echoes.
final class LichessBotSentChatTests: XCTestCase {

    private let botID = "drewschessmachine"

    private func chatLine(_ username: String, _ text: String, room: String = "player") -> Data {
        Data(#"{"type":"chatLine","room":"\#(room)","username":"\#(username)","text":"\#(text)"}"#.utf8)
    }

    @MainActor
    func testGoodbyeAfterTheStreamClosesIsShownAndEchoesAreNotDoubled() {
        let game = LichessBotLiveGame(id: "g1", startedAt: Date(), ourAccountID: botID)
        // The echo can arrive after the POST's reply…
        game.apply(.chatSent(room: .player, text: "Hi there", origin: .greeting))
        game.apply(.streamLine(chatLine("DrewsChessMachine", "Hi there"), receivedAt: Date()))
        // …or before it.
        game.apply(.streamLine(chatLine("DrewsChessMachine", "Commands: !name"), receivedAt: Date()))
        game.apply(.chatSent(room: .player, text: "Commands: !name", origin: .commandReply))
        game.apply(.streamLine(chatLine("alice", "gg"), receivedAt: Date()))
        // The goodbye: sent after the stream closed, so never echoed.
        game.apply(.chatSent(room: .player, text: "Thanks for the game, alice!", origin: .goodbye))

        XCTAssertEqual(game.chat.map(\.text), ["Hi there", "Commands: !name", "gg", "Thanks for the game, alice!"])
        XCTAssertEqual(game.chat.map(\.origin), [.greeting, .commandReply, nil, .goodbye])
        XCTAssertEqual(game.chat.map(\.echoed), [true, true, false, false])
    }

    func testRecordIncludesTheUnechoedGoodbyeOnce() throws {
        let start = Date(timeIntervalSince1970: 1_790_000_000)
        let full = #"{"type":"gameFull","id":"g1","variant":{"key":"standard","name":"Standard","short":"Std"},"speed":"blitz","perf":{"name":"Blitz"},"rated":false,"createdAt":1790000000000,"white":{"id":"drewschessmachine","name":"DrewsChessMachine","title":"BOT","rating":1500},"black":{"id":"alice","name":"Alice","rating":1500},"initialFen":"startpos","clock":{"initial":180000,"increment":2000},"state":{"type":"gameState","moves":"","wtime":180000,"btime":180000,"winc":2000,"binc":2000,"status":"started"}}"#
        let entries: [LichessBotJournalEntry] = [
            .init(at: start, event: .header(schemaVersion: LichessBotJournal.schemaVersion, gameID: "g1", build: 1, gitHash: "test", resumed: false)),
            .init(at: start, event: .streamLine(raw: full)),
            .init(at: start.addingTimeInterval(1), event: .chatSent(room: "player", text: "Hi there", origin: "greeting")),
            .init(at: start.addingTimeInterval(1.1), event: .streamLine(raw: String(decoding: chatLine("DrewsChessMachine", "Hi there"), as: UTF8.self))),
            .init(at: start.addingTimeInterval(20), event: .finished(status: "aborted", winner: nil, localDrawCondition: nil)),
            .init(at: start.addingTimeInterval(20.1), event: .chatSent(room: "player", text: "Thanks for the game, Alice!", origin: "goodbye")),
        ]
        let record = try LichessBotRecordBuilder.build(
            gameID: "g1",
            journal: .init(elements: entries, droppedTrailingByteCount: 0),
            export: nil,
            exportUnavailableReason: "no export in this test",
            ourAccountID: botID,
            checkedAt: Date()
        )
        XCTAssertEqual(record.chat.map(\.text), ["Hi there", "Thanks for the game, Alice!"])
        XCTAssertEqual(record.chat.map(\.username), ["DrewsChessMachine", "DrewsChessMachine"])
    }
}

/// After a game, Lichess has closed the game stream; the player-room chat
/// is fetched and only lines not already shown are added (by author and
/// text, one known line per fetched line).
final class LichessBotPostGameChatTests: XCTestCase {

    @MainActor
    func testOnlyUnseenFetchedLinesAreNew() {
        let game = LichessBotLiveGame(id: "g1", startedAt: Date(), ourAccountID: "drewschessmachine")
        game.apply(.chatSent(room: .player, text: "Hi", origin: .greeting))
        game.apply(.streamLine(Data(#"{"type":"chatLine","room":"player","username":"alice","text":"gl"}"#.utf8), receivedAt: Date()))
        game.apply(.streamLine(Data(#"{"type":"chatLine","room":"spectator","username":"watcher","text":"gl"}"#.utf8), receivedAt: Date()))
        game.apply(.chatSent(room: .player, text: "Thanks for the game, alice!", origin: .goodbye))
        let fetched = [
            LichessBotFetchedChatLine(text: "Hi", user: "DrewsChessMachine"),
            LichessBotFetchedChatLine(text: "gl", user: "alice"),
            LichessBotFetchedChatLine(text: "Thanks for the game, alice!", user: "DrewsChessMachine"),
            LichessBotFetchedChatLine(text: "gg", user: "alice"),
            LichessBotFetchedChatLine(text: "gl", user: "alice"),
        ]
        XCTAssertEqual(game.unseenChatLines(in: fetched), [
            LichessBotFetchedChatLine(text: "gg", user: "alice"),
            LichessBotFetchedChatLine(text: "gl", user: "alice"),
        ], "a repeated line counts once per occurrence")
        for line in game.unseenChatLines(in: fetched) {
            game.apply(.chatFetched(username: line.user, text: line.text))
        }
        XCTAssertEqual(game.unseenChatLines(in: fetched), [], "a second fetch adds nothing")
    }

    func testRecordIncludesFetchedChat() throws {
        let start = Date(timeIntervalSince1970: 1_790_000_000)
        let entries: [LichessBotJournalEntry] = [
            .init(at: start, event: .header(schemaVersion: LichessBotJournal.schemaVersion, gameID: "g1", build: 1, gitHash: "test", resumed: false)),
            .init(at: start, event: .streamLine(raw: #"{"type":"gameFull","id":"g1","variant":{"key":"standard","name":"Standard","short":"Std"},"speed":"blitz","perf":{"name":"Blitz"},"rated":false,"createdAt":1790000000000,"white":{"id":"drewschessmachine","name":"DrewsChessMachine","title":"BOT","rating":1500},"black":{"id":"alice","name":"Alice","rating":1500},"initialFen":"startpos","clock":{"initial":180000,"increment":2000},"state":{"type":"gameState","moves":"","wtime":180000,"btime":180000,"winc":2000,"binc":2000,"status":"started"}}"#)),
            .init(at: start.addingTimeInterval(20), event: .finished(status: "aborted", winner: nil, localDrawCondition: nil)),
            .init(at: start.addingTimeInterval(60), event: .chatFetched(username: "alice", text: "gg")),
        ]
        let record = try LichessBotRecordBuilder.build(
            gameID: "g1",
            journal: .init(elements: entries, droppedTrailingByteCount: 0),
            export: nil,
            exportUnavailableReason: "no export in this test",
            ourAccountID: "drewschessmachine",
            checkedAt: Date()
        )
        XCTAssertEqual(record.chat.map(\.text), ["gg"])
        XCTAssertEqual(record.chat.map(\.room), ["player"])
    }
}
