import XCTest
@testable import DrewsChessMachine

/// How the weights a game played were trained (`MODEL_TRAINING_METHOD_PLAN.md`)
/// reaches the game's facts and the Record card's Model menu and Models
/// pane; a game recorded before generations kept it reads the played file's
/// header (`LichessBotPlayedFileHistories`).
final class LichessBotTrainingMethodTests: XCTestCase {
    private typealias Fixtures = LichessBotStatsFixtures

    private static let sha = String(repeating: "bc", count: 32)
    private static let replay = ModelTrainingHistory(methods: [.corpusReplay])
    private static let replayThenSelfPlay = ModelTrainingHistory(methods: [.corpusReplay, .selfPlay])

    private func entry(modelID: String, history: ModelTrainingHistory) -> ModelFileEntry {
        ModelFileEntry(url: URL(fileURLWithPath: "/models/x.safetensors"), modelID: modelID, trainingStep: 1000, createdAt: nil,
                       architectureLabel: "a", fileModifiedAt: Date(timeIntervalSince1970: 1_759_000_000), trainingHistory: history)
    }

    private func record(_ info: LichessBotGenerationInfo) throws -> LichessBotGameRecord {
        try Fixtures.record(ourColor: .white, plies: 20, status: "resign", winner: "white") { _ in
            .decided(win: 0.5, draw: 0.2, loss: 0.3, generation: info)
        }
    }

    func testTheRecordedHistoryReachesTheFacts() throws {
        var info = Fixtures.generationInfo(id: 1, modelID: "20261008-5-mIMw", sha: Self.sha, step: 7000)
        info.trainingHistory = Self.replay
        let generation = try XCTUnwrap(LichessBotGameFacts(record: try record(info)).moves?.generations.first)
        XCTAssertEqual(generation.trainingHistory, Self.replay)
    }

    /// The recorded history wins; without one the played file's, read once
    /// per path, only while the file holds the generation's model.
    func testAPastGameReadsThePlayedFileOnlyWhileItHoldsTheSameModel() throws {
        var reads = 0
        var heldModelID = "20260714-1-h7vI"
        let playedFiles = LichessBotPlayedFileHistories { _ in
            reads += 1
            return self.entry(modelID: heldModelID, history: Self.replay)
        }
        let past = Fixtures.generationInfo(id: 1, modelID: "20260714-1-h7vI", sha: Self.sha, step: 270_000)
        XCTAssertNil(past.trainingHistory, "a record from before the field")
        XCTAssertEqual(playedFiles.history(of: past), Self.replay)
        XCTAssertEqual(playedFiles.history(of: past), Self.replay)
        XCTAssertEqual(reads, 1, "each path is read once")

        var recorded = past
        recorded.trainingHistory = Self.replayThenSelfPlay
        XCTAssertEqual(playedFiles.history(of: recorded), Self.replayThenSelfPlay, "the record's own history wins")

        heldModelID = "20261008-9-OTHR"
        let other = LichessBotPlayedFileHistories { _ in self.entry(modelID: heldModelID, history: Self.replay) }
        XCTAssertNil(other.history(of: past), "the file now holds another model")

        let gone = LichessBotPlayedFileHistories { url in throw CocoaError(.fileNoSuchFile, userInfo: [NSFilePathErrorKey: url.path]) }
        XCTAssertNil(gone.history(of: past))

        let snapshot = Fixtures.generationInfo(id: 2, modelID: "20261008-1-LIVE")
        XCTAssertNil(snapshot.filePath)
        XCTAssertNil(playedFiles.history(of: snapshot), "an in-memory source names no file")
    }

    func testTheIndexRowTakesThePlayedFilesHistory() throws {
        let info = Fixtures.generationInfo(id: 1, modelID: "20260714-1-h7vI", sha: Self.sha, step: 270_000)
        let playedFiles = LichessBotPlayedFileHistories { _ in self.entry(modelID: "20260714-1-h7vI", history: Self.replay) }
        let row = LichessBotGameSummary(record: try record(info), playedFiles: playedFiles)
        XCTAssertEqual(row.facts?.moves?.generations.first?.trainingHistory, Self.replay)
        XCTAssertNil(LichessBotGameSummary(record: try record(info)).facts?.moves?.generations.first?.trainingHistory,
                     "the record alone says nothing")
    }

    /// The Models pane and the Model menu share the label; it names the
    /// method when any of the key's games says.
    func testTheMenuAndModelsPaneNameTheMethod() throws {
        let now = Date(timeIntervalSince1970: 1_791_000_000)
        let known = Fixtures.generation("20261008-5-mIMw", source: .file, step: 7000, sha: Self.sha, trainingHistory: Self.replay, moves: 20)
        let unknown = Fixtures.generation("20261008-5-mIMw", source: .file, step: 7000, sha: Self.sha, moves: 20)
        let rows = [
            try Fixtures.row(id: "old", at: now.addingTimeInterval(-600), score: 1, facts: Fixtures.facts(generations: [unknown])),
            try Fixtures.row(id: "new", at: now, score: 0, facts: Fixtures.facts(generations: [known])),
        ]
        let choice = try XCTUnwrap(LichessBotStatsModelChoice.choices(from: rows).first)
        XCTAssertEqual(choice.menuLabel, "20261008-5-mIMw · step 7,000 · \(LichessBotModelSourceKind.file.displayName) · bcbcbcbc · corpus replay (2 games)")
        let statistics = try LichessBotRecordStatistics.compute(rows: rows, now: now, calendar: Fixtures.utcCalendar)
        let checkpoint = try XCTUnwrap(statistics[.all].byPeriod.allTime.models.groups.first?.checkpoints.first)
        XCTAssertEqual(checkpoint.label, "step 7,000 · \(LichessBotModelSourceKind.file.displayName) · bcbcbcbc · corpus replay")
    }

    func testAGenerationRecordedBeforeTheFieldDecodes() throws {
        var info = Fixtures.generationInfo(id: 1, modelID: "M", sha: Self.sha)
        info.trainingHistory = Self.replay
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601withFractionalSeconds
        var object = try XCTUnwrap(try JSONSerialization.jsonObject(with: encoder.encode(info)) as? [String: Any])
        XCTAssertNotNil(object.removeValue(forKey: "trainingHistory"))
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601withFractionalSeconds
        let old = try decoder.decode(LichessBotGenerationInfo.self, from: JSONSerialization.data(withJSONObject: object))
        XCTAssertNil(old.trainingHistory)
        XCTAssertEqual(try decoder.decode(LichessBotGenerationInfo.self, from: encoder.encode(info)).trainingHistory, Self.replay)
    }
}
