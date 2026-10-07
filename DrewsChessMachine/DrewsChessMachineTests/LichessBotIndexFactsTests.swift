import XCTest
@testable import DrewsChessMachine

/// The games index carries each game's facts (`LICHESS_BOT_RECORD_STATS_PLAN.md`
/// §4.2): an index of the previous schema is rebuilt with them, the
/// incremental path equals a rebuild, and a row without them still decodes.
/// The schema is named only through `LichessBotIndex.schemaVersion`, so
/// another plan's bump does not break these tests.
final class LichessBotIndexFactsTests: XCTestCase {
    private typealias Fixtures = LichessBotStatsFixtures

    private var directory: LichessBotDataDirectory!

    override func setUpWithError() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("LichessBotIndexFactsTests-\(UUID().uuidString)", isDirectory: true)
        directory = LichessBotDataDirectory(root: root)
        addTeardownBlock {
            if FileManager.default.fileExists(atPath: root.path) {
                try FileManager.default.removeItem(at: root)
            }
        }
    }

    private let model = Fixtures.generationInfo(id: 1, modelID: "20261001-1-AAAA")

    /// A record of `game` with its own ID and start time.
    private func record(_ gameID: String, minutesAfterTheFixture minutes: Double, plies: Int, winner: String?) throws -> LichessBotGameRecord {
        let built = try Fixtures.record(ourColor: .white, plies: plies, status: winner == nil ? "draw" : "resign", winner: winner) { ply in
            .decided(win: Float(ply % 10) / 10, draw: 0.1, loss: 0.9 - Float(ply % 10) / 10, generation: self.model)
        }
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601withFractionalSeconds
        var object = try XCTUnwrap(JSONSerialization.jsonObject(with: encoder.encode(built)) as? [String: Any])
        object["gameID"] = gameID
        object["url"] = "https://lichess.org/\(gameID)"
        object["createdAt"] = built.createdAt.addingTimeInterval(minutes * 60).formatted(Date.ISO8601FormatStyle(includingFractionalSeconds: true))
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601withFractionalSeconds
        return try decoder.decode(LichessBotGameRecord.self, from: JSONSerialization.data(withJSONObject: object))
    }

    /// Write `record` where the record store files it, and return its URL.
    @discardableResult
    private func write(_ record: LichessBotGameRecord) throws -> URL {
        let folder = directory.gamesMonthDirectory(createdAt: record.createdAt)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let url = folder.appendingPathComponent("\(record.gameID).json")
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601withFractionalSeconds
        try encoder.encode(record).write(to: url)
        return url
    }

    func testAnIndexOfThePreviousSchemaIsRebuiltWithFacts() throws {
        let records = [
            try record("aaaa0001", minutesAfterTheFixture: 0, plies: 30, winner: "white"),
            try record("aaaa0002", minutesAfterTheFixture: 10, plies: 42, winner: nil),
            try record("aaaa0003", minutesAfterTheFixture: 20, plies: 21, winner: "black"),
        ]
        for record in records {
            try write(record)
        }
        // The previous schema's index, otherwise current (same records,
        // same signature, rows without facts): only its version is stale.
        let current = try LichessBotIndex.load(directory)
        let previous = LichessBotIndex.File(
            schemaVersion: LichessBotIndex.schemaVersion - 1,
            recordCount: current.recordCount,
            recordSignature: current.recordSignature,
            rows: [],
            unreadableRecords: []
        )
        try LichessBotIndex.write(previous, to: directory)

        let loaded = try LichessBotIndex.load(directory)
        XCTAssertEqual(loaded.schemaVersion, LichessBotIndex.schemaVersion)
        XCTAssertEqual(loaded.rows.count, 3)
        for record in records {
            let row = try XCTUnwrap(loaded.rows.first { $0.gameID == record.gameID })
            XCTAssertEqual(row.facts, LichessBotGameFacts(record: record), record.gameID)
            XCTAssertNotNil(row.facts?.moves)
        }
        // The rewritten file decodes to the same index, and a second load
        // returns it unchanged.
        XCTAssertEqual(try LichessBotIndex.load(directory), loaded)
    }

    func testIncrementalUpsertEqualsRebuildWithFacts() throws {
        try write(try record("bbbb0001", minutesAfterTheFixture: 0, plies: 30, winner: "white"))
        try write(try record("bbbb0002", minutesAfterTheFixture: 5, plies: 40, winner: "black"))
        _ = try LichessBotIndex.load(directory)
        let third = try record("bbbb0003", minutesAfterTheFixture: 9, plies: 42, winner: nil)
        let url = try write(third)
        let incremental = try LichessBotIndex.upsert(LichessBotGameSummary(record: third), recordURL: url, in: directory)
        let rebuilt = try LichessBotIndex.rebuild(directory, reason: "test")
        XCTAssertEqual(incremental, rebuilt)
        XCTAssertEqual(incremental.rows.first?.gameID, "bbbb0003", "newest first")
        XCTAssertEqual(incremental.rows.first?.facts, LichessBotGameFacts(record: third))
    }

    func testARowWithoutFactsDecodesAsNil() throws {
        let row = try Fixtures.row(id: "old", at: Date(timeIntervalSince1970: 1_759_500_000), score: 1)
        XCTAssertNil(row.facts)
        // And the statistics count it as "no move data".
        let statistics = try LichessBotRecordStatistics.compute(rows: [row], now: Date(timeIntervalSince1970: 1_759_503_600), calendar: Fixtures.utcCalendar)
        XCTAssertEqual(statistics[.all].rowsWithoutMoveData, 1)
        XCTAssertEqual(statistics[.all].byPeriod.allTime.selfAssessment.gamesWithoutMoveData, 1)
    }
}
