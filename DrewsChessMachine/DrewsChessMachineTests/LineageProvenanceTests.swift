import XCTest
@testable import DrewsChessMachine

/// Determinism plan phase P10: `derivation_history` carried forward by every
/// later save (B4), the `[RUN]` startup provenance line (D4), and the lineage
/// a CLI run records in `results.json` (D5).
final class LineageProvenanceTests: XCTestCase {

    // MARK: - Fixtures

    private let arch = NetworkArchitecture.current
    private var directory: URL!

    override func setUpWithError() throws {
        directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("LineageProvenanceTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
    }

    override func tearDownWithError() throws {
        try FileManager.default.removeItem(at: directory)
    }

    private func write(_ data: Data, named name: String) throws -> URL {
        let url = directory.appendingPathComponent(name)
        try data.write(to: url, options: .withoutOverwriting)
        return url
    }

    private func baseWeights() -> [[Float]] {
        arch.weightTensorPlan().enumerated().map { i, spec in
            (0..<spec.elementCount).map { Float((i * 5 + $0) % 11) * 0.01 }
        }
    }

    private func trainerWeights() -> [[Float]] {
        baseWeights() + arch.trainableTensorPlan().map { [Float](repeating: 0.25, count: $0.elementCount) }
    }

    private func trainerFile(modelID: String, steps: Int, architecture: NetworkArchitecture, lineage: LineageRecord) throws -> Data {
        try SafetensorsModelIO.encode(
            modelID: modelID, createdAtUnix: 1_790_000_300,
            metadata: ModelCheckpointMetadata.trainerFile(
                creator: "replay", trainingStep: steps, parentModelID: "", notes: "provenance test",
                schedule: TrainerScheduleState(completedTrainSteps: steps, lrWarmupSteps: 3, lrMomentumCycle: .disabled),
                policyTailPrecision: .float32FromPreBatchNorm),
            weights: trainerWeights(), architecture: architecture, includesVelocity: true, lineage: lineage)
    }

    /// A fresh model file, then a `--derive-model` of it.
    private func derivedModel() throws -> ModelDerivation.Result {
        let mintDate = Date(timeIntervalSince1970: 1_790_000_000)
        let source = try SafetensorsModelIO.encode(
            modelID: "20261002-1-SRCB", createdAtUnix: 1_790_000_000,
            metadata: ModelCheckpointMetadata(creator: "test", trainingStep: 0, parentModelID: "", notes: ""),
            weights: baseWeights(), architecture: arch, includesVelocity: false,
            lineage: try LineageTracker.mintRecord(pathKind: .newModel, argv: ["dcm", "--new-model"], initialization: .forTests, at: mintDate))
        return try ModelDerivation.derive(
            sourceData: source, sourceName: "source.safetensors",
            operations: [SetRezeroAlphaCapDeriveOperation(value: 2, groupIndices: nil)],
            newModelID: "20261002-2-DRVB", createdAtUnix: 1_790_000_100, build: "test",
            invocationArguments: ["dcm", "--derive-model"])
    }

    // MARK: - B4: derivation_history carried forward

    func testDerivationHistorySurvivesTrainingSavesAndResumes() throws {
        let derived = try derivedModel()
        XCTAssertEqual(derived.history.count, 1)
        let derivedHeader = try SafetensorsFile.decode(derived.data).1

        // Train from the derived file (a branch) and save.
        let parent = try SafetensorsModelIO.readParentFile(at: try write(derived.data, named: "derived.safetensors"))
        XCTAssertEqual(parent.derivationHistory, derived.history)
        let start = Date(timeIntervalSince1970: 1_790_000_200)
        let branch = try LineageTracker(start: .branch(parent: parent), pathKind: .replay, argv: ["dcm"],
                                        startedAt: start, segmentStartTrainerStep: 0)
        let trainedRecord = try branch.record(at: start.addingTimeInterval(30), trainerCompletedSteps: 1, segmentLocalStep: 1,
                                              segmentGames: 1, segmentPositions: 60, corpus: nil, parameters: nil,
                                              rng: .withoutRunStreams(dropoutPhiloxState: nil), inputs: branch.testInputs)
        XCTAssertEqual(trainedRecord.derivationHistory, derived.history)
        let trained = try trainerFile(modelID: "20261002-3-TRNB", steps: 1,
                                      architecture: derived.targetArchitecture, lineage: trainedRecord)
        let trainedHeader = try SafetensorsFile.decode(trained).1
        XCTAssertEqual(trainedHeader[ModelDerivation.derivationHistoryKey], derivedHeader[ModelDerivation.derivationHistoryKey])
        XCTAssertNotNil(trainedHeader[ModelDerivation.derivationHistoryKey])

        // An exact resume of the trained file keeps it too, and so does the
        // decoded file's own view of itself as a parent.
        let trainedFile = try SafetensorsModelIO.decode(trained).file
        XCTAssertEqual(trainedFile.lineageParent.derivationHistory, derived.history)
        let resume = try LineageTracker(
            start: .resume(parent: trainedFile.lineageParent, gaps: [], legacyTotals: nil),
            pathKind: .replay, argv: ["dcm"], startedAt: start.addingTimeInterval(60), segmentStartTrainerStep: 1)
        let resumedRecord = try resume.record(at: start.addingTimeInterval(90), trainerCompletedSteps: 2, segmentLocalStep: 1,
                                              segmentGames: 1, segmentPositions: 60, corpus: nil, parameters: nil,
                                              rng: .withoutRunStreams(dropoutPhiloxState: nil), inputs: resume.testInputs)
        XCTAssertEqual(resumedRecord.derivationHistory, derived.history)

        // Deriving again from a model file of the trained weights (derive
        // takes model files, not trainer state) extends the same history.
        let trainedModel = try SafetensorsModelIO.encode(
            modelID: "20261002-3-TRNB", createdAtUnix: 1_790_000_300,
            metadata: ModelCheckpointMetadata(creator: "test", trainingStep: 1, parentModelID: "", notes: ""),
            weights: baseWeights(), architecture: derived.targetArchitecture, includesVelocity: false,
            lineage: trainedRecord)
        let second = try ModelDerivation.derive(
            sourceData: trainedModel, sourceName: "trained.safetensors",
            operations: [SetRezeroAlphaCapDeriveOperation(value: 3, groupIndices: nil)],
            newModelID: "20261002-4-DRV2", createdAtUnix: 1_790_000_400, build: "test",
            invocationArguments: ["dcm", "--derive-model"])
        XCTAssertEqual(second.history.map(\.modelID), ["20261002-2-DRVB", "20261002-4-DRV2"])
    }

    func testARunThatWasNeverDerivedHasAnEmptyHistoryAndNoMirror() throws {
        let record = try LineageRecord.forTests(trainerCompletedSteps: 3, corpus: nil)
        XCTAssertEqual(record.derivationHistory, [])
        let data = try trainerFile(modelID: "20261002-5-FRSH", steps: 3, architecture: arch, lineage: record)
        XCTAssertNil(try SafetensorsFile.decode(data).1[ModelDerivation.derivationHistoryKey])
    }

    func testAFileWrittenBeforeLineageStatesItsHistoryWithTheFlatKey() throws {
        let derived = try derivedModel()
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let json = String(decoding: try encoder.encode(derived.history), as: UTF8.self)
        XCTAssertEqual(
            try LineageTracker.ParentFile.derivationHistory(
                lineage: .unrecorded(formatVersion: 6), metadata: [ModelDerivation.derivationHistoryKey: json]),
            derived.history)
        XCTAssertEqual(
            try LineageTracker.ParentFile.derivationHistory(lineage: .unrecorded(formatVersion: 6), metadata: [:]), [])
        XCTAssertThrowsError(try LineageTracker.ParentFile.derivationHistory(
            lineage: .unrecorded(formatVersion: 6), metadata: [ModelDerivation.derivationHistoryKey: "not json"]))
    }

    func testARecordWithoutTheDerivationHistoryKeyFailsToDecode() throws {
        let record = try LineageRecord.forTests(trainerCompletedSteps: 4, corpus: nil)
        var object = try XCTUnwrap(try JSONSerialization.jsonObject(with: Data(try record.jsonText().utf8)) as? [String: Any])
        XCTAssertNotNil(object.removeValue(forKey: "derivation_history"))
        let stripped = String(decoding: try JSONSerialization.data(withJSONObject: object), as: UTF8.self)
        XCTAssertThrowsError(try LineageRecord.decode(jsonText: stripped))
    }

    // MARK: - D4: the [RUN] line

    private func resumedRecord() throws -> (record: LineageRecord, parentSHA: String) {
        let start = Date(timeIntervalSince1970: 1_790_001_000)
        let first = try LineageTracker(start: .fresh(initialization: .forTests), pathKind: .replay, argv: ["dcm"], startedAt: start, segmentStartTrainerStep: 0)
        first.recordTrainingStep(totalMs: 2500)
        let firstRecord = try first.record(at: start.addingTimeInterval(100), trainerCompletedSteps: 10, segmentLocalStep: 10,
                                           segmentGames: 7, segmentPositions: 400, corpus: nil, parameters: nil,
                                           rng: .withoutRunStreams(dropoutPhiloxState: nil), inputs: first.testInputs)
        let parentData = try trainerFile(modelID: "20261002-6-PRNT", steps: 10, architecture: arch, lineage: firstRecord)
        let parentURL = try write(parentData, named: "parent.safetensors")
        let parent = try SafetensorsModelIO.readParentFile(at: parentURL)
        let parentSHA = try XCTUnwrap(parent.contentSHA256)
        let resume = try LineageTracker(start: .resume(parent: parent, gaps: [], legacyTotals: nil),
                                        pathKind: .replay, argv: ["dcm", "--replay-corpus", "w3aA5b", "--seed", "42"],
                                        startedAt: start.addingTimeInterval(200), segmentStartTrainerStep: 10)
        try resume.noteSegmentStartForTests(trainerStep: 10)
        let parameters = try LineageRecord.Parameters(values: ["learning_rate": .double(0.0005)])
        let record = try resume.startRecord(at: start.addingTimeInterval(200), trainerCompletedSteps: 10, parameters: parameters,
                                            inputs: resume.testInputs)
        return (record, parentSHA)
    }

    func testRunLineCarriesEveryProvenanceField() throws {
        let (record, parentSHA) = try resumedRecord()
        let seed = RunRandomSeed.resolve(mode: .unseeded, configuredSeed: 0, commandLineSeed: 42, drawSeed: { 7 })
        let line = RunProvenanceLine.line(record: record, seed: seed)

        XCTAssertTrue(line.hasPrefix("[RUN] "), line)
        XCTAssertEqual(line.components(separatedBy: "[RUN]").count, 2, "one [RUN] line: \(line)")
        for fragment in [
            "path=replay",
            "run=\(record.run.lineageRunID)",
            "seg=1",
            "(exact resume of 20261002-6-PRNT sha=\(parentSHA.prefix(12)))",
            "build=\(BuildInfo.buildNumber)",
            "git=\(BuildInfo.gitHash)",
            "dirty=\(BuildInfo.gitDirty)",
            "device=",
            "vm=",
            "os=",
            "seed=42 mode=seeded(--seed) derivation=\(DCMRandomStreams.derivationVersion)",
            "params_sha=\(try XCTUnwrap(record.parameters?.sha256).prefix(12))",
            "policy_tail=\(ChessNetwork.PolicyTailPrecision.default.rawValue)",
            "git_diff=",
            "cum_step=10",
            "cum_games=7",
            "cum_train_sec=2.5",
            "argv=\"dcm --replay-corpus w3aA5b --seed 42\"",
        ] {
            XCTAssertTrue(line.contains(fragment), "missing \(fragment) in \(line)")
        }
    }

    func testRunLineSaysWhatIsUnrecordedOrAbsent() throws {
        let start = Date(timeIntervalSince1970: 1_790_002_000)
        let legacyParent = LineageTracker.ParentFile(modelID: "20260901-1-OLDP", contentSHA256: nil, trainerCompletedSteps: 41_000,
                                                     lineage: .unrecorded(formatVersion: 6), derivationHistory: [])
        let tracker = try LineageTracker(start: .resume(parent: legacyParent, gaps: [], legacyTotals: nil),
                                         pathKind: .vsuci, argv: ["dcm"], startedAt: start, segmentStartTrainerStep: 41_000)
        let record = try tracker.startRecord(at: start, trainerCompletedSteps: 41_000, parameters: nil,
                                             inputs: tracker.testInputs)
        let line = RunProvenanceLine.line(record: record, seed: nil)
        for fragment in ["path=vsuci", "resume of 20260901-1-OLDP sha=unrecorded", "not exact: lineage",
                         "seed=none", "params_sha=none", "policy_tail=none", "cum_step=41000", "cum_games=unrecorded", "cum_train_sec=unrecorded"] {
            XCTAssertTrue(line.contains(fragment), "missing \(fragment) in \(line)")
        }

        let uci = RunProvenanceLine.line(pathLabel: "uci", build: try .current, device: .current, argv: ["dcm", "--uci"])
        for fragment in ["[RUN] path=uci", "run=none", "build=\(BuildInfo.buildNumber)", "seed=none", "argv=\"dcm --uci\""] {
            XCTAssertTrue(uci.contains(fragment), "missing \(fragment) in \(uci)")
        }
        XCTAssertFalse(uci.contains("cum_step"), uci)
    }

    // MARK: - D5: results.json

    private func statsLine(cumulative: CliTrainingRecorder.LineageTotals) -> CliTrainingRecorder.StatsLine {
        CliTrainingRecorder.StatsLine(
            elapsedSec: 1, steps: 1, positionsFed: 4096, bufferCount: 4096, bufferCapacity: 8192,
            policyLoss: 2.3, valueLoss: 0.9, policyEntropy: nil,
            policyIllegalMassPenalty: 0.001, gradGlobalNorm: 4.8, playedMoveProb: nil,
            valueMean: nil, valueAbsMean: nil, valueProbWin: nil, valueProbDraw: nil, valueProbLoss: nil,
            policyLogitMean: nil, valueLogitMean: nil,
            batchSize: 4096, learningRate: 1e-3, gradClipMaxNorm: 30, weightDecayC: 5e-4, dropoutRate: 0,
            entropyRegularizationCoeff: 0, drawPenalty: 0, policyLossWeight: 1, valueLossWeight: 1,
            lrEffectiveBase: 1e-3, momentumEffective: 0.9, buildNumber: 1, trainerID: "20261002-7-TEST",
            positionsProduced: 4096, lineageTotals: cumulative
        )
    }

    func testResultsJSONCarriesTheRunsLineageAndPerRowTotals() throws {
        let (record, _) = try resumedRecord()
        let recorder = CliTrainingRecorder()
        recorder.appendStats(statsLine(cumulative: CliTrainingRecorder.LineageTotals(cumTrainerStep: 11, cumTrainStepSec: 3.0, cumGames: 9)))
        recorder.appendStats(statsLine(cumulative: CliTrainingRecorder.LineageTotals(cumTrainerStep: 12, cumTrainStepSec: nil, cumGames: nil)))
        recorder.setFinalLineage(record, checkpointSHA256: "abc123")
        let object = try XCTUnwrap(try JSONSerialization.jsonObject(with: try recorder.encodedJSONData(totalTrainingSeconds: 1)) as? [String: Any])

        let lineage = try XCTUnwrap(object["lineage"] as? [String: Any])
        XCTAssertEqual(lineage["checkpoint_sha256"] as? String, "abc123")
        XCTAssertEqual((lineage["run"] as? [String: Any])?["lineage_run_id"] as? String, record.run.lineageRunID)
        XCTAssertEqual((lineage["steps"] as? [String: Any])?["cum_trainer_step"] as? Int, 10)
        XCTAssertNotNil(lineage["fed"])
        XCTAssertNotNil(lineage["time"])
        XCTAssertNotNil(lineage["parameters"])
        // Schema 3: the configuration and seeds the run trained under travel
        // with the parameters they qualify.
        XCTAssertNotNil(lineage["configuration"])
        XCTAssertNotNil(lineage["run_seeds"])
        XCTAssertNil(lineage["segments"], "segments are left out of results.json")
        XCTAssertNil(lineage["derivation_history"], "derivation_history is left out of results.json")

        let rows = try XCTUnwrap(object["stats"] as? [[String: Any]])
        XCTAssertEqual(rows[0]["cum_trainer_step"] as? Int, 11)
        XCTAssertEqual(rows[0]["cum_train_step_sec"] as? Double, 3.0)
        XCTAssertEqual(rows[0]["cum_games"] as? Int, 9)
        // An unrecorded total is left out of the row, like every other
        // unmeasured value in results.json.
        XCTAssertEqual(rows[1]["cum_trainer_step"] as? Int, 12)
        XCTAssertNil(rows[1]["cum_games"])
        XCTAssertNil(rows[1]["cum_train_step_sec"])
    }

    func testResultsJSONOfARunThatRecordedNoLineageHasNone() throws {
        let recorder = CliTrainingRecorder()
        let object = try XCTUnwrap(try JSONSerialization.jsonObject(with: try recorder.encodedJSONData(totalTrainingSeconds: 1)) as? [String: Any])
        XCTAssertNil(object["lineage"])
    }
}
