import XCTest
@testable import DrewsChessMachine

/// Reading test-set results back for the pickers (test-set results plan D6,
/// validation 7 and 9): the summary of the largest set and its text, the
/// model catalog's entries, and the session manifest's champion and trainer
/// summaries.
final class ModelTestSetSummaryTests: XCTestCase {

    private static func set(id: String, positions: Int, top1: Int = 2212, top5: Int = 3733,
                            pElo: ModelTestSetResults.PuzzleElo = .estimate(1630.4)) -> ModelTestSetResults.SetResult {
        ModelTestSetResults.SetResult(
            id: id, title: "Lichess puzzles, \(id)", description: "\(positions) puzzles.",
            fingerprintSHA256: String(repeating: "c", count: 64), positions: positions,
            top1Correct: top1, top5Correct: top5, avgCorrectProbability: 0.1923, avgCorrectRank: 3.667,
            nll: 2.071, pElo: pElo, themes: [])
    }

    private static let evaluated = ModelTestSetResultsField.evaluated(ModelTestSetResults(
        evaluatedAtUnix: 1_791_414_697, build: 2427, policyTailPrecision: .mixedFinalProjection,
        sets: [set(id: "200", positions: 200, top1: 99, top5: 170), set(id: "wide", positions: 4435)]))

    // MARK: - Summary

    func testTheSummaryIsTheLargestSet() {
        let summary = ModelTestSetSummary(.recorded(Self.evaluated))
        XCTAssertEqual(summary.largestSet?.id, "wide")
        XCTAssertEqual(summary.pEloText, "1630")
        XCTAssertEqual(summary.nllText, "2.071")
        XCTAssertEqual(summary.top1Text, "49.9%")
        XCTAssertEqual(summary.top5Text, "84.2%")
        XCTAssertEqual(summary.line, "Lichess puzzles, wide: pElo 1630 · NLL 2.071 · top-1 49.9% · top-5 84.2%")
        XCTAssertTrue(summary.help.contains("top-1 2212/4435"), summary.help)
        XCTAssertTrue(summary.help.contains("4435 puzzles."), summary.help)
    }

    func testBoundsAndMissingResultsAreSaidNotBlank() {
        XCTAssertEqual(ModelTestSetSummary.evaluated(Self.set(id: "w", positions: 10, pElo: .allCorrect)).pEloText, "all ✓")
        XCTAssertEqual(ModelTestSetSummary.evaluated(Self.set(id: "w", positions: 10, pElo: .allWrong)).pEloText, "all ✗")

        let cases: [(ModelTestSetResultsInFile, String)] = [
            (.notRecorded, "not recorded"),
            (.unreadable("bad json"), "unreadable: bad json"),
            (.recorded(.failed(reason: "3 of 200 positions errored")), "failed: 3 of 200 positions errored"),
        ]
        for (reading, phrase) in cases {
            let summary = ModelTestSetSummary(reading)
            XCTAssertNil(summary.largestSet)
            XCTAssertEqual([summary.pEloText, summary.nllText, summary.top1Text, summary.top5Text], ["—", "—", "—", "—"])
            XCTAssertTrue(summary.line.contains(phrase), summary.line)
            XCTAssertEqual(summary.help, summary.line)
        }
        let empty = ModelTestSetResultsField.evaluated(ModelTestSetResults(evaluatedAtUnix: 1, build: 1, policyTailPrecision: .mixedFinalProjection, sets: []))
        guard case .unreadable = ModelTestSetSummary(.recorded(empty)) else { return XCTFail("a record without sets is unreadable") }
    }

    // MARK: - Files

    private var tempDir: URL!

    override func setUpWithError() throws {
        tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("dcm-test-set-summary-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        if let tempDir {
            try FileManager.default.removeItem(at: tempDir)
        }
    }

    private func writeModel(named name: String, results: ModelTestSetResultsField?) throws -> URL {
        let arch = NetworkArchitecture.current
        let weights = arch.weightTensorPlan().enumerated().map { index, spec in
            (0..<spec.elementCount).map { Float(index * 5 + $0 % 7) * 0.01 }
        }
        let meta = ModelCheckpointMetadata(creator: "test", trainingStep: nil, parentModelID: "", notes: "summary")
        let lineage = try LineageRecord.forTests(trainerCompletedSteps: nil, corpus: nil)
        var data: Data
        if let results {
            data = try SafetensorsModelIO.encode(modelID: "20261007-8-SUMM", createdAtUnix: 1_790_000_000, metadata: meta,
                                                 weights: weights, architecture: arch, includesVelocity: false,
                                                 lineage: lineage, testSetResults: results)
        } else {
            // A file from before test-set results: the same file without the key.
            let written = try SafetensorsModelIO.encode(modelID: "20261007-8-SUMM", createdAtUnix: 1_790_000_000, metadata: meta,
                                                        weights: weights, architecture: arch, includesVelocity: false, lineage: lineage)
            let (tensors, metadata) = try SafetensorsFile.decode(written)
            var withoutKey = metadata
            withoutKey[ModelTestSetResultsField.metadataKey] = nil
            withoutKey[SafetensorsFile.contentHashKey] = nil
            data = try SafetensorsFile.encode(tensors: tensors, metadata: withoutKey)
        }
        let url = tempDir.appendingPathComponent(name)
        try data.write(to: url)
        return url
    }

    func testTheCatalogEntryCarriesTheSummaryAndHeaderAndFullReadsAgree() throws {
        let url = try writeModel(named: "with.safetensors", results: Self.evaluated)
        let entry = try ModelFileCatalog.entry(for: url)
        XCTAssertEqual(entry.testSets, ModelTestSetSummary(.recorded(Self.evaluated)))
        XCTAssertEqual(ModelTestSetSummary.ofModelFile(at: url), entry.testSets)
        let (_, fullMetadata) = try SafetensorsFile.decode(try Data(contentsOf: url))
        XCTAssertEqual(ModelTestSetResultsField.reading(fromMetadata: fullMetadata), .recorded(Self.evaluated))
        XCTAssertNoThrow(try CheckpointManager.loadModelFile(at: url))
    }

    func testAnOlderFileIsNotRecordedAndStillLoads() throws {
        let url = try writeModel(named: "older.safetensors", results: nil)
        XCTAssertEqual(try ModelFileCatalog.entry(for: url).testSets, .notRecorded)
        XCTAssertNoThrow(try CheckpointManager.loadModelFile(at: url))
        XCTAssertEqual(ModelTestSetSummary.ofModelFile(at: tempDir.appendingPathComponent("absent.safetensors")), .notRecorded)
    }

    func testAnUnreadableValueIsReportedAndTheModelStillLoads() throws {
        let url = try writeModel(named: "with.safetensors", results: Self.evaluated)
        let (tensors, metadata) = try SafetensorsFile.decode(try Data(contentsOf: url))
        var broken = metadata
        broken[ModelTestSetResultsField.metadataKey] = #"{"schema":1,"status":"evaluated"}"#
        broken[SafetensorsFile.contentHashKey] = nil
        let brokenURL = tempDir.appendingPathComponent("broken.safetensors")
        try SafetensorsFile.encode(tensors: tensors, metadata: broken).write(to: brokenURL)
        guard case .unreadable = try ModelFileCatalog.entry(for: brokenURL).testSets else {
            return XCTFail("expected unreadable")
        }
        XCTAssertNoThrow(try CheckpointManager.loadModelFile(at: brokenURL))
    }

    // MARK: - Session manifest

    func testTheSessionManifestCarriesBothFilesSummaries() async throws {
        let champion = try await ChessMPSNetwork(.randomWeights(initSeed: 12)).exportWeights()
        let velocity = NetworkArchitecture.current.weightTensorPlan().filter { $0.kind != .bnRunningStat }.map {
            [Float](repeating: 0.001, count: $0.elementCount)
        }
        let meta = ModelCheckpointMetadata(creator: "manual", trainingStep: 1, parentModelID: "", notes: "manifest")
        let state = try SessionCheckpointState.decode(Data("""
        {
          "formatVersion": \(SessionCheckpointState.currentFormatVersion),
          "lineage": \(LineageRecord.sessionTestFixtureJSON),
          "sessionID": "20261007-9-MNFS", "savedAtUnix": 1700000000,
          "sessionStartUnix": 1699996400, "elapsedTrainingSec": 3600,
          "trainingSteps": 12345, "selfPlayGames": 678, "selfPlayMoves": 45678,
          "trainingPositionsSeen": 12641280, "batchSize": 1024, "learningRate": 5.0e-5,
          "promoteThreshold": 0.53, "arenaGames": 400,
          "selfPlayTau": {"startTau": 1.0, "decayPerPly": 0.007, "floorTau": 0.5},
          "arenaTau": {"startTau": 0.6, "decayPerPly": 0.02, "floorTau": 0.2},
          "selfPlayWorkerCount": 4,
          "championID": "20261007-9-MNFS", "trainerID": "20261007-9-MNFT", "arenaHistory": []
        }
        """.utf8))
        let dir = try await CheckpointManager.saveSession(
            championWeights: champion, championID: "20261007-9-MNFS", championMetadata: meta, championCreatedAtUnix: 1_790_000_000,
            trainerWeights: champion + velocity, trainerID: "20261007-9-MNFT", trainerMetadata: meta, trainerCreatedAtUnix: 1_790_000_000,
            state: state, lineage: try LineageRecord.forTests(trainerCompletedSteps: nil, corpus: nil),
            championLineage: try LineageRecord.forTests(trainerCompletedSteps: nil, corpus: nil),
            testSetEvaluator: FixtureTestSetEvaluator(Self.evaluated), trigger: "unittest", sessionsDirectory: tempDir)

        let expected = ModelTestSetSummary(.recorded(Self.evaluated))
        let written = try JSONDecoder().decode(SessionManifest.self, from: Data(contentsOf: dir.appendingPathComponent("manifest.json")))
        XCTAssertEqual(written.championTestSets, expected)
        XCTAssertEqual(written.trainerTestSets, expected)
        let extracted = SessionManifest.extract(fromSessionFolder: dir)
        XCTAssertEqual(extracted.championTestSets, expected)
        XCTAssertEqual(extracted.trainerTestSets, expected)
    }

    /// A manifest written before these fields decodes, with no summaries.
    func testAnOlderManifestDecodesWithoutSummaries() throws {
        let manifest = SessionManifest.extract(fromSessionFolder: tempDir.appendingPathComponent("missing.dcmsession"))
        var json = try XCTUnwrap(JSONSerialization.jsonObject(with: JSONEncoder().encode(manifest)) as? [String: Any])
        json["championTestSets"] = nil
        json["trainerTestSets"] = nil
        let older = try JSONDecoder().decode(SessionManifest.self, from: JSONSerialization.data(withJSONObject: json))
        XCTAssertNil(older.championTestSets)
        XCTAssertNil(older.trainerTestSets)
    }
}
