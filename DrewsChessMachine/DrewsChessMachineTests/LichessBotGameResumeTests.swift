import XCTest
@testable import DrewsChessMachine

/// The pure parts of resuming a game from its journal: the session's
/// carryover, the checks on a leftover journal, and the live view's replay.
@MainActor
final class LichessBotGameResumeTests: XCTestCase {

    private nonisolated static let botID = "drewschessmachine"
    private static let start = Date(timeIntervalSince1970: 1_759_000_000)

    private static let generation = LichessBotGenerationInfo(
        generationID: 1,
        sourceKind: .champion,
        modelID: "20260928-1-TEST",
        trainingStep: nil,
        snapshotAt: start,
        architectureSummary: "test",
        filePath: nil,
        fileSHA256: nil
    )

    private static func decision(_ uci: String, loss: Float = 0.3) -> LichessBotMoveDecision {
        LichessBotMoveDecision(
            uci: uci, san: uci, chosenProbability: 0.5, topMoves: [],
            win: 0.3, draw: 1 - 0.3 - loss, loss: loss, temperature: 0.5, legalMoveCount: 20, randomish: false,
            encodeMilliseconds: 0.1, inferenceMilliseconds: 2, sampleMilliseconds: 0.1
        )
    }

    private static func gameFullJSON(gameID: String = "g1", whiteID: String = "bob", blackID: String = botID, moves: String = "") -> String {
        #"{"type":"gameFull","id":"\#(gameID)","variant":{"key":"standard"},"clock":{"initial":180000,"increment":2000},"speed":"blitz","perf":{"name":"Blitz"},"rated":false,"createdAt":1759000000000,"white":{"id":"\#(whiteID)","name":"\#(whiteID)","rating":1500},"black":{"id":"\#(blackID)","name":"\#(blackID)","title":"BOT","rating":1500},"initialFen":"startpos","state":{"type":"gameState","moves":"\#(moves)","wtime":180000,"btime":180000,"winc":2000,"binc":2000,"status":"started"}}"#
    }

    private static func stateJSON(_ moves: String) -> String {
        #"{"type":"gameState","moves":"\#(moves)","wtime":179000,"btime":178000,"winc":2000,"binc":2000,"status":"started"}"#
    }

    private func entries(_ events: [LichessBotJournalEvent]) -> [LichessBotJournalEntry] {
        events.enumerated().map { index, event in
            LichessBotJournalEntry(at: Self.start.addingTimeInterval(TimeInterval(index)), event: event)
        }
    }

    private func header(gameID: String = "g1", schemaVersion: Int = LichessBotJournal.schemaVersion, resumed: Bool = false) -> LichessBotJournalEvent {
        .header(schemaVersion: schemaVersion, gameID: gameID, build: 1, gitHash: "test", resumed: resumed)
    }

    // MARK: - Carryover

    func testTheFoldCountsAllowancesAndKeepsOneReadingPerPly() {
        let folded = LichessBotGameSessionCarryover.fold(entries([
            header(),
            .moveDecided(ply: 1, decision: Self.decision("e7e5", loss: 0.4), generation: Self.generation),
            .moveDecided(ply: 3, decision: Self.decision("b8c6", loss: 0.5), generation: Self.generation),
            // Decided again at ply 3 (a resync): replaces the reading.
            .moveDecided(ply: 3, decision: Self.decision("g8f6", loss: 0.6), generation: Self.generation),
            .takebackAccepted,
            .commandReplyQueued(command: "help", username: "bob", room: "player"),
            .commandReplyQueued(command: "name", username: "bob", room: "spectator"),
            .chatSent(room: "player", text: "hi", origin: LichessBotChatOrigin.greeting.rawValue),
        ]))
        XCTAssertEqual(folded.readings.map(\.ply), [1, 3])
        XCTAssertEqual(folded.readings.last?.loss, 0.6)
        XCTAssertEqual(folded.takebacksAccepted, 1)
        XCTAssertEqual(folded.commandRepliesQueued, 2)
        XCTAssertTrue(folded.greeted)
        XCTAssertFalse(folded.farewellSent)
    }

    func testARebuiltPositionDropsTheReadingsItTookBack() {
        let folded = LichessBotGameSessionCarryover.fold(entries([
            header(),
            .moveDecided(ply: 1, decision: Self.decision("e7e5"), generation: Self.generation),
            .moveDecided(ply: 3, decision: Self.decision("b8c6"), generation: Self.generation),
            .positionSynced(kind: .rebuilt, fromPly: 4, toPly: 2),
            .chatSent(room: "player", text: "gg", origin: LichessBotChatOrigin.goodbye.rawValue),
        ]))
        XCTAssertEqual(folded.readings.map(\.ply), [1])
        XCTAssertTrue(folded.farewellSent)
        XCTAssertFalse(folded.greeted)
    }

    func testANewGameCarriesNothingOver() {
        XCTAssertEqual(LichessBotGameSessionCarryover.fold([]), .newGame)
    }

    // MARK: - Checks on a leftover journal

    private func make(_ events: [LichessBotJournalEvent], gameID: String = "g1") throws -> LichessBotResumedJournal {
        try LichessBotResumedJournal.make(gameID: gameID, journal: .init(elements: entries(events), droppedTrailingByteCount: 0), ourAccountID: Self.botID)
    }

    func testAJournalWithoutAHeaderIsRefused() {
        XCTAssertThrowsError(try make([.streamOpened(attempt: 0)])) { error in
            XCTAssertEqual(error as? LichessBotResumeError, .noHeader(gameID: "g1"))
        }
    }

    func testAJournalForAnotherGameIsRefused() {
        XCTAssertThrowsError(try make([header(gameID: "g2")])) { error in
            XCTAssertEqual(error as? LichessBotResumeError, .headerForAnotherGame(gameID: "g1", headerGameID: "g2"))
        }
    }

    func testAJournalFromANewerSchemaIsRefused() {
        XCTAssertThrowsError(try make([header(schemaVersion: LichessBotJournal.schemaVersion + 1)])) { error in
            XCTAssertEqual(error as? LichessBotResumeError, .newerSchema(gameID: "g1", schemaVersion: LichessBotJournal.schemaVersion + 1))
        }
    }

    func testAJournalOfAGameDCMDoesNotPlayIsRefused() {
        XCTAssertThrowsError(try make([header(), .streamLine(raw: Self.gameFullJSON(whiteID: "bob", blackID: "carol"))])) { error in
            XCTAssertEqual(error as? LichessBotResumeError, .notOurGame(gameID: "g1"))
        }
    }

    func testAJournalStartsWhenItsHeaderWasWritten() throws {
        let journal = try make([header(), .streamOpened(attempt: 0), .streamLine(raw: Self.gameFullJSON())])
        XCTAssertEqual(journal.firstJournaledAt, Self.start)
        XCTAssertEqual(journal.lastJournaledAt, Self.start.addingTimeInterval(2))
    }

    // MARK: - The live view's replay

    /// The same events, live and replayed from the journal, give the same
    /// game: moves, decisions, chat, anomalies and stream count.
    func testReplayingTheJournalRebuildsWhatTheLiveEventsShowed() throws {
        let gameFull = Self.gameFullJSON(moves: "e2e4")
        let fullLine = try LichessBotGameStreamLine.decode(Data(gameFull.utf8))
        guard case .gameFull(let full) = fullLine else {
            return XCTFail("fixture is not a gameFull")
        }
        let chatLine = #"{"type":"chatLine","room":"player","username":"bob","text":"hello there"}"#
        let decided = Self.decision("e7e5")
        let liveEvents: [LichessBotGameEvent] = [
            .streamOpened(attempt: 0),
            .streamLine(Data(gameFull.utf8), receivedAt: Self.start),
            .gameInfo(full, ourColor: .black),
            .moveDecided(ply: 1, decision: decided, generation: Self.generation),
            .chatSent(room: .player, text: "Hi!", origin: .greeting),
            .streamLine(Data(chatLine.utf8), receivedAt: Self.start),
            .streamLine(Data(Self.stateJSON("e2e4 e7e5").utf8), receivedAt: Self.start),
            .takebackAccepted,
            .anomaly("something odd"),
        ]
        let live = LichessBotLiveGame(id: "g1", startedAt: Self.start, ourAccountID: Self.botID)
        for event in liveEvents {
            live.apply(event)
        }

        var journalEvents: [LichessBotJournalEvent] = [header()]
        journalEvents += liveEvents.compactMap(LichessBotJournal.event(for:))
        let journal = try make(journalEvents)
        let replayed = LichessBotLiveGame(id: "g1", startedAt: journal.firstJournaledAt, ourAccountID: Self.botID)
        replayed.replay(journal)

        XCTAssertEqual(replayed.plies.map(\.uciAsGiven), live.plies.map(\.uciAsGiven))
        XCTAssertEqual(replayed.decisions, live.decisions)
        XCTAssertEqual(replayed.chat.map(\.text), live.chat.map(\.text))
        XCTAssertEqual(replayed.chat.map(\.origin), live.chat.map(\.origin))
        XCTAssertEqual(replayed.anomalies, live.anomalies)
        XCTAssertEqual(replayed.streamConnections, live.streamConnections)
        XCTAssertEqual(replayed.ourColor, live.ourColor)
        XCTAssertEqual(replayed.opponent, live.opponent)
        XCTAssertTrue(replayed.transcript.contains { $0.title == "accepted takeback" })
        // A post-game fetch after the resume finds nothing new in what the
        // journal already held.
        XCTAssertEqual(replayed.unseenChatLines(in: [LichessBotFetchedChatLine(text: "hello there", user: "bob"), LichessBotFetchedChatLine(text: "Hi!", user: Self.botID)]), [])
    }
}
