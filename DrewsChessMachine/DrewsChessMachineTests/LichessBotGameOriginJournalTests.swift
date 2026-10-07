//
//  LichessBotGameOriginJournalTests.swift
//  DrewsChessMachineTests
//
//  A game's origin through its journal (challenge-log plan §3.5): the record
//  builder and the resumed journal keep the first determined origin (a
//  determined one replaces an earlier undetermined one; a second, different
//  determined one is an anomaly), an old journal without the case gives no
//  origin, the live view's replay sets it, the carryover fold ignores it, the
//  index row and the PGN carry it, and an index stored at the previous
//  schema version is rebuilt at the current one.
//

import XCTest
@testable import DrewsChessMachine

@MainActor
final class LichessBotGameOriginJournalTests: XCTestCase {

    private static let botID = "drewschessmachine"
    private static let createdAt: Int64 = 1_759_000_000_000

    private var tempRoot: URL!

    override func setUpWithError() throws {
        tempRoot = FileManager.default.temporaryDirectory
            .appendingPathComponent("LichessBotGameOriginJournalTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: tempRoot, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try FileManager.default.removeItem(at: tempRoot)
    }

    private static let gameFull = #"{"type":"gameFull","id":"g1","variant":{"key":"standard"},"clock":{"initial":180000,"increment":2000},"speed":"blitz","perf":{"name":"Blitz"},"rated":false,"createdAt":1759000000000,"white":{"id":"drewschessmachine","name":"drewschessmachine","title":"BOT","rating":1500},"black":{"id":"alice","name":"Alice","rating":1600},"initialFen":"startpos","state":{"type":"gameState","moves":"","wtime":180000,"btime":180000,"winc":2000,"binc":2000,"status":"started"}}"#

    private static let queueOrigin = LichessBotGameOrigin.outgoingChallengeAccepted(challengeID: "g1", sender: .challengeQueue)
    private static let sheetOrigin = LichessBotGameOrigin.outgoingChallengeAccepted(challengeID: "g1", sender: .challengeSheet)
    private static let undetermined = LichessBotGameOrigin.undetermined(source: LichessBotOpenValue(.friend), gap: .noChallengeRecord)

    private func journal(origins: [LichessBotGameOrigin], finished: Bool = true) -> [LichessBotJournalEntry] {
        var at = Date(timeIntervalSince1970: Double(Self.createdAt) / 1000)
        func next() -> Date {
            at = at.addingTimeInterval(1)
            return at
        }
        var entries: [LichessBotJournalEntry] = [
            .init(at: next(), event: .header(schemaVersion: LichessBotJournal.schemaVersion, gameID: "g1", build: 1, gitHash: "test", resumed: false)),
            .init(at: next(), event: .streamOpened(attempt: 0)),
            .init(at: next(), event: .streamLine(raw: Self.gameFull)),
        ]
        for origin in origins {
            entries.append(.init(at: next(), event: .gameOrigin(origin)))
        }
        if finished {
            entries.append(.init(at: next(), event: .finished(status: "resign", winner: "white", localDrawCondition: nil)))
        }
        return entries
    }

    private func record(origins: [LichessBotGameOrigin]) throws -> LichessBotGameRecord {
        try LichessBotRecordBuilder.build(
            gameID: "g1", journal: .init(elements: journal(origins: origins), droppedTrailingByteCount: 0),
            export: nil, exportUnavailableReason: "test", ourAccountID: Self.botID, checkedAt: Date()
        )
    }

    // MARK: - Record builder and resumed journal

    func testTheFirstDeterminedOriginIsKept() throws {
        XCTAssertEqual(try record(origins: [Self.queueOrigin]).origin, Self.queueOrigin)
        XCTAssertEqual(try record(origins: [Self.undetermined, Self.queueOrigin]).origin, Self.queueOrigin, "a determined origin replaces an undetermined one")
        XCTAssertEqual(try record(origins: [Self.queueOrigin, Self.undetermined]).origin, Self.queueOrigin)
        XCTAssertEqual(try record(origins: [Self.undetermined]).origin, Self.undetermined)
    }

    func testTwoDifferentDeterminedOriginsAreAnAnomalyAndTheFirstIsKept() throws {
        let record = try record(origins: [Self.queueOrigin, Self.sheetOrigin])
        XCTAssertEqual(record.origin, Self.queueOrigin)
        XCTAssertTrue(record.anomalies.contains { $0.text.contains("second origin, sheet") }, "\(record.anomalies)")
        XCTAssertFalse(try self.record(origins: [Self.queueOrigin, Self.queueOrigin]).anomalies.contains { $0.text.contains("origin") })
    }

    func testAnOldJournalWithoutTheCaseHasNoOrigin() throws {
        let record = try record(origins: [])
        XCTAssertNil(record.origin)
        XCTAssertNil(LichessBotGameSummary(record: record).origin)
        XCTAssertFalse(LichessBotPGNWriter.pgn(for: record).contains("DCMOrigin"))
    }

    func testTheResumedJournalHoldsTheRecordedOrigin() throws {
        let resumed = try LichessBotResumedJournal.make(
            gameID: "g1", journal: .init(elements: journal(origins: [Self.undetermined, Self.queueOrigin], finished: false), droppedTrailingByteCount: 0),
            ourAccountID: Self.botID)
        XCTAssertEqual(resumed.recordedOrigin, Self.queueOrigin)
        let none = try LichessBotResumedJournal.make(
            gameID: "g1", journal: .init(elements: journal(origins: [], finished: false), droppedTrailingByteCount: 0),
            ourAccountID: Self.botID)
        XCTAssertNil(none.recordedOrigin)
    }

    func testTheCarryoverFoldIsUnchangedByTheOrigin() {
        XCTAssertEqual(LichessBotGameSessionCarryover.fold(journal(origins: [Self.queueOrigin], finished: false)),
                       LichessBotGameSessionCarryover.fold(journal(origins: [], finished: false)))
    }

    func testTheLiveViewsReplaySetsTheOrigin() throws {
        let resumed = try LichessBotResumedJournal.make(
            gameID: "g1", journal: .init(elements: journal(origins: [Self.queueOrigin, Self.undetermined], finished: false), droppedTrailingByteCount: 0),
            ourAccountID: Self.botID)
        let game = LichessBotLiveGame(id: "g1", startedAt: Date(), ourAccountID: Self.botID)
        XCTAssertNil(game.origin)
        game.replay(resumed)
        XCTAssertEqual(game.origin, Self.queueOrigin)
    }

    // MARK: - Index and PGN

    func testANewRecordsOriginReachesItsRowAndItsPGN() throws {
        let record = try record(origins: [Self.queueOrigin])
        XCTAssertEqual(LichessBotGameSummary(record: record).origin, Self.queueOrigin)
        XCTAssertTrue(LichessBotPGNWriter.pgn(for: record).contains(#"[DCMOrigin "queue"]"#))
    }

    func testARecordWrittenBeforeOriginsDecodesWithNone() throws {
        let record = try record(origins: [Self.queueOrigin])
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        encoder.dateEncodingStrategy = .iso8601withFractionalSeconds
        var object = try XCTUnwrap(JSONSerialization.jsonObject(with: encoder.encode(record)) as? [String: Any])
        XCTAssertNotNil(object.removeValue(forKey: "origin"))
        let old = try JSONSerialization.data(withJSONObject: object)
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601withFractionalSeconds
        XCTAssertNil(try decoder.decode(LichessBotGameRecord.self, from: old).origin)
    }

    func testAnIndexStoredAtThePreviousSchemaIsRebuiltAtTheCurrentOne() async throws {
        let directory = LichessBotDataDirectory(root: tempRoot)
        try directory.createDirectories()
        let store = LichessBotRecordStore(directory: directory, fileQueue: LichessBotFileQueue(), ourAccountID: Self.botID)
        var data = Data()
        for entry in journal(origins: [Self.queueOrigin]) {
            data.append(try LichessBotJSONLines.encodeLine(entry))
        }
        try data.write(to: directory.inProgressJournalURL(gameID: "g1"))
        _ = try await store.finalize(gameID: "g1", export: nil, exportUnavailableReason: "test")
        let built = try await store.loadIndex()
        XCTAssertEqual(built.schemaVersion, LichessBotIndex.schemaVersion)
        XCTAssertEqual(built.rows.first?.origin, Self.queueOrigin)

        // Rewrite the stored index as the previous schema wrote it.
        var stored = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: directory.indexURL)) as? [String: Any])
        stored["schemaVersion"] = LichessBotIndex.schemaVersion - 1
        try JSONSerialization.data(withJSONObject: stored).write(to: directory.indexURL)
        let rebuilt = try await store.loadIndex()
        XCTAssertEqual(rebuilt.schemaVersion, LichessBotIndex.schemaVersion)
        XCTAssertEqual(rebuilt.rows.first?.origin, Self.queueOrigin)
        let onDisk = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: directory.indexURL)) as? [String: Any])
        XCTAssertEqual(onDisk["schemaVersion"] as? Int, LichessBotIndex.schemaVersion, "the rebuilt index is stored at the current schema")
    }
}
