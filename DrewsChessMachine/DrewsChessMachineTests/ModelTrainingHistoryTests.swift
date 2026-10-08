import XCTest
@testable import DrewsChessMachine

/// `ModelTrainingHistory` (`MODEL_TRAINING_METHOD_PLAN.md`): how a model's
/// weights were trained, oldest first, from its lineage record or its
/// creator, and where the pickers read it.
final class ModelTrainingHistoryTests: XCTestCase {

    private let start = Date(timeIntervalSince1970: 1_790_000_000)

    private func parameters() throws -> LineageRecord.Parameters {
        try LineageRecord.Parameters(values: ["learning_rate": .double(0.0005)])
    }

    /// A trained record of a segment begun by `start` on `pathKind`.
    private func trainedRecord(_ start: LineageTracker.Start, _ pathKind: LineageRecord.PathKind, clock: Int) throws -> LineageRecord {
        let tracker = try LineageTracker(start: start, pathKind: pathKind, argv: ["dcm"],
                                         startedAt: self.start, segmentStartTrainerStep: 0)
        try tracker.noteSegmentStartForTests(trainerStep: 0)
        return try tracker.record(at: self.start.addingTimeInterval(60), trainerCompletedSteps: clock, segmentLocalStep: clock,
                                  segmentGames: 0, segmentPositions: 0, corpus: nil, parameters: try parameters(),
                                  rng: .withoutRunStreams(dropoutPhiloxState: nil), inputs: tracker.testInputs)
    }

    private func parent(_ record: LineageRecord, modelID: String) -> LineageTracker.ParentFile {
        LineageTracker.ParentFile(modelID: modelID, contentSHA256: String(repeating: "c", count: 64),
                                  trainerCompletedSteps: record.steps.cumTrainerStep, lineage: .recorded(record),
                                  derivationHistory: [])
    }

    private static let fresh = LineageTracker.Start.fresh(initialization: .forTests, naming: .unnamedWithoutPreset)

    // MARK: - Methods and text

    func testConsecutiveRepeatsCollapseAndTheTextReadsOldestFirst() {
        let history = ModelTrainingHistory(methods: [.corpusReplay, .corpusReplay, .selfPlay, .selfPlay, .corpusReplay])
        XCTAssertEqual(history.methods, [.corpusReplay, .selfPlay, .corpusReplay])
        XCTAssertEqual(history.displayText, "corpus replay → self-play → corpus replay")
        XCTAssertEqual(history.logText, "corpus_replay>self_play>corpus_replay")
        XCTAssertNil(ModelTrainingHistory.unknown.displayText)
        XCTAssertEqual(ModelTrainingHistory.unknown.logText, "unrecorded")
    }

    func testDecodingCollapsesRepeatsToo() throws {
        let decoded = try JSONDecoder().decode(ModelTrainingHistory.self, from: Data(#"{"methods":["uci_play","uci_play","self_play"]}"#.utf8))
        XCTAssertEqual(decoded.methods, [.uciPlay, .selfPlay])
        XCTAssertEqual(try JSONDecoder().decode(ModelTrainingHistory.self, from: JSONEncoder().encode(decoded)), decoded)
    }

    func testEveryTrainingPathHasAMethodAndTheOthersNone() {
        XCTAssertEqual(ModelTrainingMethod(pathKind: .gui), .selfPlay)
        XCTAssertEqual(ModelTrainingMethod(pathKind: .replay), .corpusReplay)
        XCTAssertEqual(ModelTrainingMethod(pathKind: .vsuci), .uciPlay)
        XCTAssertNil(ModelTrainingMethod(pathKind: .derive))
        XCTAssertNil(ModelTrainingMethod(pathKind: .newModel))
        XCTAssertEqual(ModelTrainingMethod.allCases.map(\.displayName), ["self-play", "corpus replay", "UCI play"])
    }

    /// Without a record only the CLI training writers say how a file was
    /// trained; a GUI writer also saved built and loaded models.
    func testAFileBeforeLineageIsKnownOnlyFromACLITrainingCreator() {
        let unrecorded = LineageRecord.Presence.unrecorded(formatVersion: ArchitectureFormat.unversionedLegacyVersion)
        XCTAssertEqual(ModelTrainingHistory(lineage: unrecorded, creator: "replay").methods, [.corpusReplay])
        XCTAssertEqual(ModelTrainingHistory(lineage: unrecorded, creator: "train-vs-uci").methods, [.uciPlay])
        for creator in ["manual", "periodic", "promote", "sigusr2", "derive-model"] {
            XCTAssertEqual(ModelTrainingHistory(lineage: unrecorded, creator: creator), .unknown, creator)
        }
        XCTAssertEqual(ModelTrainingHistory(lineage: unrecorded, creator: nil), .unknown)
    }

    // MARK: - From a lineage record

    func testAFreshTrainedRunIsItsPath() throws {
        XCTAssertEqual(ModelTrainingHistory(record: try trainedRecord(Self.fresh, .replay, clock: 10)).methods, [.corpusReplay])
        XCTAssertEqual(ModelTrainingHistory(record: try trainedRecord(Self.fresh, .vsuci, clock: 10)).methods, [.uciPlay])
    }

    func testAnUntrainedRecordSaysNothing() throws {
        XCTAssertEqual(ModelTrainingHistory(record: try LineageRecord.forTests(trainerCompletedSteps: 0, corpus: nil)), .unknown)
        XCTAssertEqual(ModelTrainingHistory(record: LineageRecord.sessionTestFixture), .unknown, "a GUI mint record")
    }

    /// Branches keep the runs they left, oldest first; an exact resume on
    /// the same path adds no new method.
    func testBranchesChainTheRunsTheyLeftOldestFirst() throws {
        let replay = try trainedRecord(Self.fresh, .replay, clock: 40)
        let resumed = try trainedRecord(.resume(parent: parent(replay, modelID: "20261008-1-REPL"), gaps: [], legacyTotals: nil), .replay, clock: 50)
        XCTAssertEqual(ModelTrainingHistory(record: resumed).methods, [.corpusReplay])
        let vsuci = try trainedRecord(.branch(parent: parent(resumed, modelID: "20261008-1-REPL")), .vsuci, clock: 5)
        XCTAssertEqual(ModelTrainingHistory(record: vsuci).methods, [.corpusReplay, .uciPlay])
        let back = try trainedRecord(.branch(parent: parent(vsuci, modelID: "20261008-2-VSUC")), .replay, clock: 5)
        XCTAssertEqual(ModelTrainingHistory(record: back).methods, [.corpusReplay, .uciPlay, .corpusReplay])
        XCTAssertEqual(try LineageRecord.decode(jsonText: try back.jsonText()), back)
        XCTAssertEqual(ModelTrainingHistory(record: try LineageRecord.decode(jsonText: try back.jsonText())).methods,
                       [.corpusReplay, .uciPlay, .corpusReplay], "the history survives the file round trip")
    }

    /// A schema-2 record has no configuration: a trained one names the path
    /// that wrote it; its earlier segments' summaries name no path.
    func testASchemaTwoRecordNamesThePathThatWroteIt() throws {
        let single = try LineageRecord.decode(jsonText: LineageSchemaTwoFixtures.fatconv98ContStep6093)
        XCTAssertEqual(single.configuration, .unrecorded)
        XCTAssertEqual(ModelTrainingHistory(record: single).methods, [.corpusReplay])
        let second = try LineageRecord.decode(jsonText: LineageSchemaTwoFixtures.segResumeCheckSeg1Step1000)
        XCTAssertEqual(second.segments.first?.pathKind, .unrecorded)
        XCTAssertEqual(ModelTrainingHistory(record: second).methods, [.corpusReplay])
    }

    /// A segment summary whose configuration says it did not train adds
    /// nothing, whatever wrote it.
    func testAnUntrainedSegmentAddsNothing() throws {
        let untrained = try LineageRecord.forTests(trainerCompletedSteps: 0, corpus: nil)
        let summary = LineageRecord.SegmentSummary(of: untrained)
        XCTAssertEqual(summary.configuration, .recorded(nil))
        XCTAssertNil(ModelTrainingHistory.method(of: summary))
    }

    // MARK: - Tracker, catalog

    /// The GUI's trainer: the runs and segments behind it, then its own
    /// path, before it has saved anything.
    func testATrackerNamesItsPathAfterTheRunsBehindIt() throws {
        let replay = try trainedRecord(Self.fresh, .replay, clock: 40)
        let gui = try LineageTracker(start: .branch(parent: parent(replay, modelID: "20261008-1-REPL")), pathKind: .gui, argv: ["dcm"],
                                     startedAt: start, segmentStartTrainerStep: 0)
        XCTAssertEqual(gui.trainingHistory.methods, [.corpusReplay, .selfPlay])
        let fresh = try LineageTracker(start: Self.fresh, pathKind: .gui, argv: ["dcm"], startedAt: start, segmentStartTrainerStep: 0)
        XCTAssertEqual(fresh.trainingHistory.methods, [.selfPlay])
    }

    func testTheCatalogEntryAndAModelFileHeaderCarryTheHistory() throws {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent("ModelTrainingHistoryTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        addTeardownBlock {
            do {
                try FileManager.default.removeItem(at: folder)
            } catch {
                XCTFail("cleanup failed: \(error)")
            }
        }
        func write(_ name: String, creator: String) throws -> URL {
            let metadata = [
                "model_id": "20260701-1-OLDF", "created_at_unix": "1759000000", "training_step": "1000", "creator": creator,
                "architecture": String(decoding: try JSONEncoder().encode(NetworkArchitecture.current), as: UTF8.self),
            ]
            let url = folder.appendingPathComponent(name)
            try SafetensorsFile.encode(tensors: [SafetensorsTensor(name: "w", shape: [2], data: [1, 2])], metadata: metadata).write(to: url)
            return url
        }
        let replay = try write("old-replay.safetensors", creator: "replay")
        let manual = try write("old-manual.safetensors", creator: "manual")
        XCTAssertEqual(try ModelFileCatalog.entry(for: replay).trainingHistory?.methods, [.corpusReplay])
        XCTAssertEqual(try ModelFileCatalog.entry(for: manual).trainingHistory, .unknown)
        XCTAssertEqual(ModelTrainingHistory.ofModelFile(at: replay, logTag: "[TEST]")?.methods, [.corpusReplay])
        XCTAssertNil(ModelTrainingHistory.ofModelFile(at: folder.appendingPathComponent("missing.safetensors"), logTag: "[TEST]"))
    }
}
