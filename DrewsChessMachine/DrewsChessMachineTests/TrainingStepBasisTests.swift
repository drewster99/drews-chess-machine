import XCTest
@testable import DrewsChessMachine

/// What a model file's `training_step` means (`ModelFileStepReading`): from
/// format v11 the trainer step, with the segment step as the lineage
/// record's `segment_local_step` (and its flat mirror); before v11 what its
/// writer — the file's `creator` — wrote, flagged once per load. Fixtures are
/// written at the current format and re-stamped to older ones by rewriting
/// the header, the way files of those versions look on disk.
@MainActor
final class TrainingStepBasisTests: XCTestCase {

    private let arch = ResumeEquivalenceTests.architecture

    private var root: URL!

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory
            .appendingPathComponent("TrainingStepBasisTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        if let root, FileManager.default.fileExists(atPath: root.path) {
            try FileManager.default.removeItem(at: root)
        }
    }

    // MARK: - Fixtures

    private func baseWeights() -> [[Float]] {
        arch.weightTensorPlan().map { [Float](repeating: 0.25, count: $0.elementCount) }
    }

    private func trainerWeights() -> [[Float]] {
        baseWeights() + arch.trainableTensorPlan().map { [Float](repeating: -0.5, count: $0.elementCount) }
    }

    /// The record a resumed segment writes after `segmentStep` steps from
    /// trainer step `startTrainerStep` (path kind `replay`).
    private func resumedRecord(startTrainerStep: Int, segmentStep: Int) throws -> LineageRecord {
        let first = try LineageRecord.forTests(trainerCompletedSteps: startTrainerStep, corpus: nil)
        let parent = LineageTracker.ParentFile(
            modelID: "20261006-1-PRNT", contentSHA256: nil, trainerCompletedSteps: startTrainerStep,
            lineage: .recorded(first), derivationHistory: [])
        let start = Date(timeIntervalSince1970: 1_790_000_000)
        let tracker = try LineageTracker(
            start: .resume(parent: parent, gaps: [], legacyTotals: nil), pathKind: .replay,
            argv: ["DrewsChessMachine", "--test"], startedAt: start, segmentStartTrainerStep: startTrainerStep)
        return try tracker.record(
            at: start.addingTimeInterval(60), trainerCompletedSteps: startTrainerStep + segmentStep,
            segmentLocalStep: segmentStep, segmentGames: 0, segmentPositions: 0, corpus: nil, parameters: nil,
            rng: .withoutRunStreams(dropoutPhiloxState: nil))
    }

    private func schedule(_ steps: Int) -> TrainerScheduleState {
        TrainerScheduleState(completedTrainSteps: steps, lrWarmupSteps: 3, lrMomentumCycle: .disabled)
    }

    /// A trainer-state file at the current format, written as the runners
    /// write theirs: segment 1 of a run, resumed at trainer step 1000 and
    /// saved 513 steps later.
    private func v11TrainerFile(creator: String = ModelCheckpointMetadata.corpusReplayCreator) throws -> Data {
        try SafetensorsModelIO.encode(
            modelID: "20261006-2-TRNR", createdAtUnix: 1_790_000_000,
            metadata: ModelCheckpointMetadata.trainerFile(
                creator: creator, trainingStep: 1_513, parentModelID: "20261006-1-PRNT", notes: "basis test",
                schedule: schedule(1_513), policyTailPrecision: .float32FromPreBatchNorm),
            weights: trainerWeights(), architecture: arch, includesVelocity: true,
            lineage: try resumedRecord(startTrainerStep: 1_000, segmentStep: 513))
    }

    /// A plain model file (no trainer state) at the current format.
    private func plainFile(creator: String, trainingStep: Int?, record: LineageRecord) throws -> Data {
        try SafetensorsModelIO.encode(
            modelID: "20261006-3-PLAN", createdAtUnix: 1_790_000_000,
            metadata: ModelCheckpointMetadata(creator: creator, trainingStep: trainingStep, parentModelID: "",
                                              notes: "basis test"),
            weights: baseWeights(), architecture: arch, includesVelocity: false, lineage: record)
    }

    /// `data` with its header changed by `change` (the content hash is
    /// recomputed on re-encode).
    private func rewritten(_ data: Data, _ change: (inout [String: String]) -> Void) throws -> Data {
        let (tensors, md) = try SafetensorsFile.decode(data)
        var metadata = md.filter { $0.key != SafetensorsFile.contentHashKey }
        change(&metadata)
        return try SafetensorsFile.encode(tensors: tensors, metadata: metadata)
    }

    /// Remove the lineage record and its mirrors, as a file written before
    /// lineage (format < 7) has none.
    private func dropLineage(_ md: inout [String: String]) {
        let keys = Set([LineageRecord.metadataKey] + LineageRecord.MirrorKey.all)
        md = md.filter { !keys.contains($0.key) }
    }

    private func trainingStepEntries(_ format: ArchitectureFormat.DecodeFormat) -> [String] {
        format.legacyLog.resolutions.filter { $0.hasPrefix("training_step ") }
    }

    // MARK: - From format v11

    func testAFileFromV11StatesTheTrainerStepWithTheSegmentStepAsTheSidecar() throws {
        let data = try v11TrainerFile()
        let (_, md) = try SafetensorsFile.decode(data)
        XCTAssertEqual(md[SafetensorsModelIO.Key.formatVersion], String(ArchitectureFormat.currentVersion))
        XCTAssertEqual(md[SafetensorsModelIO.Key.trainingStep], "1513")
        XCTAssertEqual(md[TrainerScheduleState.MetadataKey.completedTrainSteps], "1513")
        XCTAssertEqual(md[LineageRecord.MirrorKey.segmentLocalStep], "513")
        let decoded = try SafetensorsModelIO.decode(data)
        let record = try XCTUnwrap(decoded.file.safetensorsProvenance?.lineage.record)
        XCTAssertEqual(record.steps.cumTrainerStep, 1_513)
        XCTAssertEqual(record.steps.segmentLocalStep, 513)
        let reading = decoded.file.trainingStepReading
        XCTAssertEqual(reading.basis, .trainerStep)
        XCTAssertEqual(reading.statedTrainingStep, 1_513)
        XCTAssertEqual(reading.trainerStep, 1_513)
        XCTAssertEqual(reading.segmentStep, 513)
        XCTAssertNil(reading.legacyResolution)
        XCTAssertEqual(try SafetensorsModelIO.trainingStepReading(fromMetadata: md, source: "v11"), reading)
    }

    func testEncodingRefusesATrainerFileWhoseStepIsNotItsClock() throws {
        for stated in [513, nil] as [Int?] {
            XCTAssertThrowsError(try SafetensorsModelIO.encode(
                modelID: "20261006-2-TRNR", createdAtUnix: 1_790_000_000,
                metadata: ModelCheckpointMetadata.trainerFile(
                    creator: ModelCheckpointMetadata.corpusReplayCreator, trainingStep: stated,
                    parentModelID: "", notes: "", schedule: schedule(1_513),
                    policyTailPrecision: .float32FromPreBatchNorm),
                weights: trainerWeights(), architecture: arch, includesVelocity: true,
                lineage: try resumedRecord(startTrainerStep: 1_000, segmentStep: 513))) { error in
                guard case .trainingStepDisagreesWithTrainerClock(let step, let clock)? =
                        error as? SafetensorsModelIO.IOError else {
                    return XCTFail("expected trainingStepDisagreesWithTrainerClock, got \(error)")
                }
                XCTAssertEqual(step, stated)
                XCTAssertEqual(clock, 1_513)
            }
        }
    }

    func testDecodingRefusesAV11TrainerFileWhoseStepIsNotItsClock() throws {
        for stated in ["513", nil] as [String?] {
            let data = try rewritten(try v11TrainerFile()) { md in md[SafetensorsModelIO.Key.trainingStep] = stated }
            XCTAssertThrowsError(try SafetensorsModelIO.decode(data, valueHead: .recenterUnlessMarked, source: "bad.safetensors")) {
                error in
                guard case .decodedTrainingStepDisagreesWithTrainerClock(let source, _, let step, let clock)? =
                        error as? SafetensorsModelIO.IOError else {
                    return XCTFail("expected decodedTrainingStepDisagreesWithTrainerClock, got \(error)")
                }
                XCTAssertEqual(source, "bad.safetensors")
                XCTAssertEqual(step, stated.flatMap { Int($0) })
                XCTAssertEqual(clock, 1_513)
            }
            let (_, md) = try SafetensorsFile.decode(data)
            XCTAssertThrowsError(try SafetensorsModelIO.trainingStepReading(fromMetadata: md, source: "bad.safetensors"))
        }
    }

    // MARK: - Before format v11

    func testAPreV11ReplayFileReadsItsStepAsTheSegmentStep() throws {
        let data = try rewritten(try v11TrainerFile()) { md in
            md[SafetensorsModelIO.Key.formatVersion] = "10"
            md[SafetensorsModelIO.Key.trainingStep] = "513"
            md.removeValue(forKey: LineageRecord.MirrorKey.segmentLocalStep)
        }
        let file = try SafetensorsModelIO.decode(data).file
        let reading = file.trainingStepReading
        XCTAssertEqual(reading.basis, .legacySegmentStep)
        XCTAssertEqual(reading.statedTrainingStep, 513)
        XCTAssertEqual(reading.trainerStep, 1_513, "the trainer clock")
        XCTAssertEqual(reading.segmentStep, 513)
        let note = try XCTUnwrap(reading.legacyResolution)
        XCTAssertTrue(note.contains("training_step 513 is the writing segment's step"), note)
        XCTAssertTrue(note.contains("trainer step 1513"), note)
        XCTAssertEqual(file.lineageParent.trainerCompletedSteps, 1_513)
    }

    func testAPreV11ReplayFileWithoutAScheduleHasNoTrainerStep() throws {
        let data = try rewritten(try plainFile(
            creator: ModelCheckpointMetadata.corpusReplayCreator, trainingStep: 41_000,
            record: try LineageRecord.forTests(trainerCompletedSteps: nil, corpus: nil))) { md in
            self.dropLineage(&md)
            md.removeValue(forKey: SafetensorsModelIO.Key.formatVersion)
        }
        let file = try SafetensorsModelIO.decode(data).file
        let reading = file.trainingStepReading
        XCTAssertEqual(reading.basis, .legacySegmentStep)
        XCTAssertNil(reading.trainerStep, "never reconstructed")
        XCTAssertEqual(reading.segmentStep, 41_000)
        XCTAssertEqual(reading.trainerStepOrStatedStep, 41_000)
        XCTAssertEqual(file.lineageParent.trainerCompletedSteps, 41_000, "the parent's stated step")
        XCTAssertTrue(try XCTUnwrap(reading.legacyResolution).contains("trainer step not recorded"))
    }

    func testAPreV11GUIFileReadsItsStepAsTheTrainerStep() throws {
        let data = try rewritten(try v11TrainerFile(creator: "manual")) { md in
            md[SafetensorsModelIO.Key.formatVersion] = "10"
        }
        let reading = try SafetensorsModelIO.decode(data).file.trainingStepReading
        XCTAssertEqual(reading.basis, .legacyGUITrainerStep)
        XCTAssertEqual(reading.trainerStep, 1_513)
        XCTAssertEqual(reading.segmentStep, 513, "the record's segment_local_step")
        XCTAssertTrue(try XCTUnwrap(reading.legacyResolution).contains("is the GUI's trainer step"))
    }

    func testAPreV11FileOfAnUnknownWriterKeepsItsStatedStep() throws {
        let data = try rewritten(try plainFile(
            creator: "test", trainingStep: 40, record: try LineageRecord.forTests(trainerCompletedSteps: nil, corpus: nil))) {
            md in
            self.dropLineage(&md)
            md[SafetensorsModelIO.Key.formatVersion] = "6"
        }
        let file = try SafetensorsModelIO.decode(data).file
        let reading = file.trainingStepReading
        XCTAssertEqual(reading.basis, .legacyUnknownWriter)
        XCTAssertEqual(reading.trainerStep, 40)
        XCTAssertEqual(reading.segmentStep, 40)
        XCTAssertEqual(file.lineageParent.trainerCompletedSteps, 40)
        XCTAssertTrue(try XCTUnwrap(reading.legacyResolution).contains("was stated by writer 'test'"))
    }

    func testTheCreatorNamesTheWriterNotTheRecordsPathKind() throws {
        // A GUI champion loaded from a corpus-replay file carries that file's
        // record (path kind `replay`), but the GUI wrote it, stating the
        // source's trainer clock.
        let record = try resumedRecord(startTrainerStep: 1_000, segmentStep: 513)
        XCTAssertEqual(record.invocation.pathKind, .replay)
        let data = try rewritten(try plainFile(creator: "manual", trainingStep: 1_513, record: record)) { md in
            md[SafetensorsModelIO.Key.formatVersion] = "8"
        }
        let reading = try SafetensorsModelIO.decode(data).file.trainingStepReading
        XCTAssertEqual(reading.basis, .legacyGUITrainerStep)
        XCTAssertEqual(reading.trainerStep, 1_513)
    }

    func testACopyOfASchedulelessPreV7ReplayFileStillReadsAsTrained() throws {
        let source = try rewritten(try plainFile(
            creator: ModelCheckpointMetadata.corpusReplayCreator, trainingStep: 41_000,
            record: try LineageRecord.forTests(trainerCompletedSteps: nil, corpus: nil))) { md in
            self.dropLineage(&md)
            md[SafetensorsModelIO.Key.formatVersion] = "3"
        }
        let sourceFile = try SafetensorsModelIO.decode(source).file
        let championStep = try SessionController.championFileTrainingStep(origin: .file(sourceFile.lineageParent))
        XCTAssertEqual(championStep, 41_000)
        // The GUI's champion file of those weights states that step, so a
        // tensor-rewriting derive of it refuses it as trained.
        let champion = try plainFile(creator: "manual", trainingStep: championStep,
                                     record: try LineageRecord.forTests(trainerCompletedSteps: nil, corpus: nil))
        let (_, md) = try SafetensorsFile.decode(champion)
        let lineage = try SafetensorsModelIO.lineage(fromMetadata: md, formatVersion: ArchitectureFormat.currentVersion)
        XCTAssertThrowsError(try ModelDerivation.requireUntrainedSource(
            md, lineage: lineage, derivationHistory: [], sourceName: "champion.safetensors")) { error in
            guard case .sourceIsTrained(_, let step)? = error as? ModelDerivation.DeriveError else {
                return XCTFail("expected sourceIsTrained, got \(error)")
            }
            XCTAssertEqual(step, 41_000)
        }
    }

    func testADcmmodelFileIsReadByItsCreatorAndFlaggedByItsLoader() async throws {
        let net = try ChessMPSNetwork(.randomWeights(initSeed: 6))
        let weights = try await net.network.exportWeights()
        let data = try ModelCheckpointFile(
            modelID: "20261006-4-DCMM", createdAtUnix: 1_790_000_000,
            metadata: ModelCheckpointMetadata(creator: "periodic", trainingStep: 777, parentModelID: "", notes: "dcmmodel"),
            weights: weights
        ).encode()
        let url = root.appendingPathComponent("legacy.dcmmodel")
        try data.write(to: url)
        let file = try CheckpointManager.loadModelFile(at: url)
        XCTAssertNil(file.architectureFormat, "a .dcmmodel has no DecodeFormat")
        XCTAssertEqual(file.trainingStepReading.basis, .legacyGUITrainerStep)
        XCTAssertEqual(file.trainingStepReading.trainerStep, 777)
        let line = try XCTUnwrap(CheckpointManager.legacyFileFactsLine(file, source: url.lastPathComponent))
        XCTAssertTrue(line.hasPrefix("[ARCH] legacy file (format v\(ArchitectureFormat.unversionedLegacyVersion)) legacy.dcmmodel: "), line)
        XCTAssertTrue(line.contains("training_step 777 is the GUI's trainer step"), line)
    }

    func testTheGUICreatorsAreTheSaveTriggerTags() {
        XCTAssertEqual(ModelCheckpointMetadata.guiCreators,
                       Set(SessionSaveTrigger.allCases.map(\.diskTag)).union([SessionSaveTrigger.promotionDiskTag]))
        XCTAssertEqual(ModelCheckpointMetadata.segmentStepCreators,
                       [ModelCheckpointMetadata.corpusReplayCreator, ModelCheckpointMetadata.trainVsUciCreator])
        XCTAssertTrue(ModelCheckpointMetadata.guiCreators.isDisjoint(with: ModelCheckpointMetadata.segmentStepCreators))
    }

    func testAnOlderFileIsFlaggedOnceAndAV11FileNotAtAll() throws {
        let current = try SafetensorsModelIO.decode(try v11TrainerFile())
        XCTAssertEqual(trainingStepEntries(current.architectureFormat), [])
        let older = try SafetensorsModelIO.decode(try rewritten(try v11TrainerFile()) { md in
            md[SafetensorsModelIO.Key.formatVersion] = "10"
            md[SafetensorsModelIO.Key.trainingStep] = "513"
        })
        XCTAssertEqual(trainingStepEntries(older.architectureFormat).count, 1)
        XCTAssertNotNil(older.architectureFormat.legacyLogLine)
    }

    func testTheRollingFileIdentityUsesTheReading() throws {
        let currentURL = root.appendingPathComponent("v11.safetensors")
        try v11TrainerFile().write(to: currentURL)
        XCTAssertEqual(try TrainerModelFileIdentity.read(from: currentURL),
                       TrainerModelFileIdentity(modelID: "20261006-2-TRNR", trainingStep: 1_513))
        let olderURL = root.appendingPathComponent("v8.safetensors")
        try rewritten(try v11TrainerFile()) { md in
            md[SafetensorsModelIO.Key.formatVersion] = "8"
            md[SafetensorsModelIO.Key.trainingStep] = "513"
        }.write(to: olderURL)
        XCTAssertEqual(try TrainerModelFileIdentity.read(from: olderURL),
                       TrainerModelFileIdentity(modelID: "20261006-2-TRNR", trainingStep: 1_513),
                       "a v8 corpus-replay file of the same line reads as its trainer step too")
        XCTAssertEqual(TrainerModelFileIdentity(file: try CheckpointManager.loadModelFile(at: olderURL)),
                       TrainerModelFileIdentity(modelID: "20261006-2-TRNR", trainingStep: 1_513))
    }

    func testAFileStatingNoStepIsNeverFlagged() throws {
        let data = try rewritten(try plainFile(
            creator: "new-model", trainingStep: nil, record: try LineageRecord.forTests(trainerCompletedSteps: nil, corpus: nil))) {
            md in
            md[SafetensorsModelIO.Key.formatVersion] = "10"
        }
        let decoded = try SafetensorsModelIO.decode(data)
        XCTAssertEqual(trainingStepEntries(decoded.architectureFormat), [])
        XCTAssertNil(decoded.file.trainingStepReading.statedTrainingStep)
        XCTAssertNil(decoded.file.trainingStepReading.legacyResolution)
    }
}
