import XCTest
@testable import DrewsChessMachine

/// Which model generation a game record credits with each of DCM's moves
/// (`LICHESS_BOT_RECORD_STATS_PLAN.md` §4.5, P0).
///
/// Generation IDs restart at 1 every time the bot goes online
/// (`LichessBotModelSlots.prepare`), so a game resumed after a relaunch can
/// carry two different models under the same ID, one per session. The record
/// must keep both and credit each move to the one that actually decided it;
/// per-model statistics read these records.
final class LichessBotGenerationAttributionTests: XCTestCase {

    private let botID = "drewschessmachine"
    private let createdAt: Int64 = 1_759_500_000_000

    /// 22 legal plies from the start, no castling. DCM is White, so it
    /// decides the even plies.
    private static let game = [
        "e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6", "d2d3", "f8e7", "c2c3",
        "b7b5", "a4b3", "d7d6", "h2h3", "c8b7", "b1d2", "d8d7", "d2f1", "a8d8", "f1g3", "h7h6",
    ]

    private static let firstModelID = "20261001-1-AAAA"
    private static let secondModelID = "20261005-3-BBBB"

    private static func generation(id: Int, modelID: String, snapshotAt: Date) -> LichessBotGenerationInfo {
        LichessBotGenerationInfo(
            generationID: id,
            sourceKind: .champion,
            modelID: modelID,
            trainingStep: nil,
            snapshotAt: snapshotAt,
            architectureSummary: "test",
            filePath: nil,
            fileSHA256: nil
        )
    }

    /// The first going-online's generation 1.
    private static let firstSessionGeneration = generation(
        id: 1, modelID: firstModelID, snapshotAt: Date(timeIntervalSince1970: 1_759_500_000)
    )

    /// The second going-online's generation 1: a different model under the
    /// same ID, because numbering restarts on every going-online.
    private static let secondSessionGeneration = generation(
        id: 1, modelID: secondModelID, snapshotAt: Date(timeIntervalSince1970: 1_759_503_600)
    )

    private static func decision(_ uci: String) -> LichessBotMoveDecision {
        LichessBotMoveDecision(
            uci: uci, san: uci, chosenProbability: 0.5, topMoves: [],
            win: 0.3, draw: 0.4, loss: 0.3, temperature: 0.5, legalMoveCount: 20, randomish: false,
            encodeMilliseconds: 0.1, inferenceMilliseconds: 2, sampleMilliseconds: 0.1
        )
    }

    private static func stateJSON(_ tokens: [String], status: String = "started", winner: String? = nil) -> String {
        let winnerField = winner.map { #","winner":"\#($0)""# } ?? ""
        return #"{"type":"gameState","moves":"\#(tokens.joined(separator: " "))","wtime":170000,"btime":160000,"winc":2000,"binc":2000,"status":"\#(status)"\#(winnerField)}"#
    }

    private func gameFullJSON(tokens: [String]) -> String {
        #"{"type":"gameFull","id":"g1","variant":{"key":"standard"},"clock":{"initial":180000,"increment":2000},"speed":"blitz","perf":{"name":"Blitz"},"rated":false,"createdAt":\#(createdAt),"white":{"id":"\#(botID)","name":"DrewsChessMachine","title":"BOT","rating":1500},"black":{"id":"alice","name":"Alice","rating":1600},"initialFen":"startpos","state":\#(Self.stateJSON(tokens))}"#
    }

    /// One going-online's stretch of the game: the app build that ran it,
    /// the generation that decided DCM's moves in it, and the plies it saw
    /// arrive.
    private struct Session {
        let build: Int
        let generation: LichessBotGenerationInfo
        let plies: Range<Int>
    }

    /// The journal a game leaves when each session in turn appends to it, as
    /// `LichessBotJournal`'s writer does: a resumed session writes its own
    /// header (`resumed: true`), reopens the stream and receives a `gameFull`
    /// holding every move so far. DCM posts each of its moves after deciding
    /// it; the game ends in a resignation after the last session.
    private func journal(_ sessions: [Session]) -> [LichessBotJournalEntry] {
        var at = Date(timeIntervalSince1970: Double(createdAt) / 1000)
        func next() -> Date {
            at = at.addingTimeInterval(1)
            return at
        }
        var entries: [LichessBotJournalEntry] = []
        for (index, session) in sessions.enumerated() {
            entries.append(.init(at: next(), event: .header(schemaVersion: LichessBotJournal.schemaVersion, gameID: "g1", build: session.build, gitHash: "test", resumed: index > 0)))
            entries.append(.init(at: next(), event: .streamOpened(attempt: 0)))
            entries.append(.init(at: next(), event: .streamLine(raw: gameFullJSON(tokens: Array(Self.game.prefix(session.plies.lowerBound))))))
            for ply in session.plies {
                let token = Self.game[ply]
                if ply % 2 == 0 {
                    entries.append(.init(at: next(), event: .moveDecided(ply: ply, decision: Self.decision(token), generation: session.generation)))
                    entries.append(.init(at: next(), event: .movePosted(ply: ply, uci: token, offeringDraw: false, milliseconds: 40)))
                }
                entries.append(.init(at: next(), event: .streamLine(raw: Self.stateJSON(Array(Self.game.prefix(ply + 1))))))
            }
        }
        entries.append(.init(at: next(), event: .streamLine(raw: Self.stateJSON(Self.game, status: "resign", winner: "white"))))
        entries.append(.init(at: next(), event: .finished(status: "resign", winner: "white", localDrawCondition: nil)))
        return entries
    }

    private func record(from entries: [LichessBotJournalEntry]) throws -> LichessBotGameRecord {
        try LichessBotRecordBuilder.build(
            gameID: "g1",
            journal: .init(elements: entries, droppedTrailingByteCount: 0),
            export: nil,
            exportUnavailableReason: "test: no export",
            ourAccountID: botID,
            checkedAt: Date(timeIntervalSince1970: 1_759_510_000)
        )
    }

    /// Session 1 (model A, generation 1) decides plies 0–10; the app is
    /// relaunched and session 2 (model B, also generation 1) decides plies
    /// 12–20.
    private var resumedAcrossARelaunch: [LichessBotJournalEntry] {
        journal([
            Session(build: 100, generation: Self.firstSessionGeneration, plies: 0..<12),
            Session(build: 101, generation: Self.secondSessionGeneration, plies: 12..<Self.game.count),
        ])
    }

    // MARK: - Tests

    /// The regression: both sessions' models are in the record and in its
    /// index row, in order of first use.
    func testTwoSessionsWithTheSameGenerationIDKeepBothModels() throws {
        let record = try record(from: resumedAcrossARelaunch)

        // The fixture itself is a clean two-session game.
        XCTAssertEqual(record.builds, [100, 101])
        XCTAssertEqual(record.moves.map(\.uciAsGiven), Self.game)
        XCTAssertTrue(record.moves.allSatisfy { $0.san != nil })
        XCTAssertEqual(record.anomalies, [])
        XCTAssertEqual(record.retractions, [])
        XCTAssertEqual(record.moves.filter { $0.ours && $0.decision != nil }.count, 11)

        XCTAssertEqual(record.generations.map(\.modelID), [Self.firstModelID, Self.secondModelID])
        XCTAssertEqual(LichessBotGameSummary(record: record).modelIDs, [Self.firstModelID, Self.secondModelID])
    }

    /// Every one of DCM's moves names, by index, the generation that decided
    /// it, though both sessions number theirs 1. The opponent's moves name
    /// none.
    func testEachMoveNamesTheGenerationThatDecidedIt() throws {
        let record = try record(from: resumedAcrossARelaunch)
        XCTAssertEqual(record.generations, [Self.firstSessionGeneration, Self.secondSessionGeneration])
        for move in record.moves {
            if move.ours {
                XCTAssertNotNil(move.decision, "ply \(move.ply)")
                XCTAssertEqual(move.generationIndex, move.ply <= 10 ? 0 : 1, "ply \(move.ply)")
                XCTAssertEqual(move.generationID, 1, "ply \(move.ply)")
            } else {
                XCTAssertNil(move.generationIndex, "ply \(move.ply)")
                XCTAssertNil(move.generationID, "ply \(move.ply)")
            }
        }
        XCTAssertEqual(record.moves.filter { $0.generationIndex == 0 }.map(\.ply), [0, 2, 4, 6, 8, 10])
        XCTAssertEqual(record.moves.filter { $0.generationIndex == 1 }.map(\.ply), [12, 14, 16, 18, 20])
    }

    /// A generation is journaled with every move it decides; equal infos are
    /// one generation, even across a session boundary.
    func testOneGenerationJournaledManyTimesIsListedOnce() throws {
        let record = try record(from: journal([
            Session(build: 100, generation: Self.firstSessionGeneration, plies: 0..<12),
            Session(build: 100, generation: Self.firstSessionGeneration, plies: 12..<Self.game.count),
        ]))
        XCTAssertEqual(record.generations, [Self.firstSessionGeneration])
        XCTAssertEqual(LichessBotGameSummary(record: record).modelIDs, [Self.firstModelID])
        let ours = record.moves.filter(\.ours)
        XCTAssertEqual(ours.count, 11)
        XCTAssertTrue(ours.allSatisfy { $0.generationIndex == 0 && $0.generationID == 1 })
    }

    /// The same model loaded again after a relaunch is a new generation (it
    /// was built anew), and still one model in the index row.
    func testTheSameModelInTwoSessionsIsTwoGenerationsAndOneModelID() throws {
        let reloaded = Self.generation(
            id: 1, modelID: Self.firstModelID, snapshotAt: Self.secondSessionGeneration.snapshotAt
        )
        let record = try record(from: journal([
            Session(build: 100, generation: Self.firstSessionGeneration, plies: 0..<12),
            Session(build: 101, generation: reloaded, plies: 12..<Self.game.count),
        ]))
        XCTAssertEqual(record.generations, [Self.firstSessionGeneration, reloaded])
        XCTAssertEqual(LichessBotGameSummary(record: record).modelIDs, [Self.firstModelID])
        XCTAssertEqual(record.moves.filter { $0.generationIndex == 0 }.map(\.ply), [0, 2, 4, 6, 8, 10])
        XCTAssertEqual(record.moves.filter { $0.generationIndex == 1 }.map(\.ply), [12, 14, 16, 18, 20])
    }

    /// A record filed before `generationIndex` existed has no such key on its
    /// moves. It still decodes through the reader the index uses, with every
    /// other field intact, and each of its moves' `generationID` names
    /// exactly one listed generation.
    func testOldRecordWithoutGenerationIndexStillDecodes() throws {
        let built = try record(from: journal([
            Session(build: 100, generation: Self.firstSessionGeneration, plies: 0..<Self.game.count),
        ]))

        // Encoded as `LichessBotRecordStore.finalize` writes a record, then
        // stripped of the new key.
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601withFractionalSeconds
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        var object = try XCTUnwrap(JSONSerialization.jsonObject(with: encoder.encode(built)) as? [String: Any])
        let encodedMoves = try XCTUnwrap(object["moves"] as? [[String: Any]])
        var strippedCount = 0
        object["moves"] = encodedMoves.map { move in
            var move = move
            if move.removeValue(forKey: "generationIndex") != nil {
                strippedCount += 1
            }
            return move
        }
        XCTAssertEqual(strippedCount, 11, "every decided move carried the key before it was stripped")
        let oldRecordData = try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
        XCTAssertFalse(String(decoding: oldRecordData, as: UTF8.self).contains("generationIndex"))

        let folder = FileManager.default.temporaryDirectory.appendingPathComponent("LichessBotGenerationAttributionTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: false)
        defer {
            do {
                try FileManager.default.removeItem(at: folder)
            } catch {
                XCTFail("removing \(folder.path): \(error)")
            }
        }
        let url = folder.appendingPathComponent("g1.json")
        try oldRecordData.write(to: url, options: .withoutOverwriting)
        let decoded = try LichessBotIndex.readRecord(at: url)

        XCTAssertTrue(decoded.moves.allSatisfy { $0.generationIndex == nil })
        let expectedMoves = built.moves.map { move in
            LichessBotGameRecord.Move(
                ply: move.ply, color: move.color, san: move.san, uciAsGiven: move.uciAsGiven,
                whiteClockMilliseconds: move.whiteClockMilliseconds, blackClockMilliseconds: move.blackClockMilliseconds,
                receivedAt: move.receivedAt, ours: move.ours, decision: move.decision,
                generationID: move.generationID, generationIndex: nil,
                postMilliseconds: move.postMilliseconds, offeredDraw: move.offeredDraw
            )
        }
        XCTAssertEqual(decoded.moves, expectedMoves)
        XCTAssertEqual(decoded.generations, built.generations)
        XCTAssertEqual(decoded.outcome, built.outcome)
        for move in decoded.moves {
            guard let generationID = move.generationID else { continue }
            XCTAssertEqual(decoded.generations.filter { $0.generationID == generationID }.map(\.modelID), [Self.firstModelID], "ply \(move.ply)")
        }
    }
}
