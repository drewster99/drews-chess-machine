import XCTest
@testable import DrewsChessMachine

/// `--train-vs-uci` saves session folders through the GUI's session writer:
/// what `--start-model` may name, where step files go, when a periodic save
/// is due, that a save round-trips through `CheckpointManager.saveSession` /
/// `loadSession` as a train-vs-UCI session the GUI refuses, and that a full
/// disk inside a session save is recognized as one.
@MainActor
final class TrainVsUciSessionTests: XCTestCase {
    private var tempDir: URL!

    override func setUp() async throws {
        try await super.setUp()
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("dcm-vsuci-session-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        tempDir = dir
    }

    override func tearDown() async throws {
        if let tempDir {
            do { try FileManager.default.removeItem(at: tempDir) } catch { /* best-effort cleanup */ }
        }
        try await super.tearDown()
    }

    // MARK: - --start-model

    func testStartSourceTellsModelFilesFromSessionFolders() throws {
        let model = tempDir.appendingPathComponent("m.safetensors")
        try Data("x".utf8).write(to: model)
        XCTAssertEqual(try TrainVsUciSession.startSource(path: model.path), .modelFile(model))

        let session = tempDir.appendingPathComponent("s.dcmsession", isDirectory: true)
        try FileManager.default.createDirectory(at: session, withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: SessionCheckpointLayout.stateURL(in: session))
        guard case .session(let resolved) = try TrainVsUciSession.startSource(path: session.path) else {
            return XCTFail("a folder with session.json is a session start")
        }
        XCTAssertEqual(resolved.standardizedFileURL.path, session.standardizedFileURL.path)
    }

    func testStartSourceRefusesWhatIsNeitherAModelFileNorASession() throws {
        let missing = tempDir.appendingPathComponent("nothing-here")
        XCTAssertThrowsError(try TrainVsUciSession.startSource(path: missing.path)) { error in
            XCTAssertEqual(error as? TrainVsUciSession.StartSourceError, .missing(path: missing.path))
        }
        let bare = tempDir.appendingPathComponent("bare", isDirectory: true)
        try FileManager.default.createDirectory(at: bare, withIntermediateDirectories: true)
        XCTAssertThrowsError(try TrainVsUciSession.startSource(path: bare.path)) { error in
            XCTAssertEqual(error as? TrainVsUciSession.StartSourceError, .folderWithoutSessionJSON(path: bare.path))
        }
        let fifo = tempDir.appendingPathComponent("pipe")
        XCTAssertEqual(mkfifo(fifo.path, 0o600), 0)
        XCTAssertThrowsError(try TrainVsUciSession.startSource(path: fifo.path)) { error in
            guard case .notAFileOrFolder? = error as? TrainVsUciSession.StartSourceError else {
                return XCTFail("a FIFO is refused by kind, got \(error)")
            }
        }
    }

    /// The start is only read, so a symbolic link to a model file is
    /// followed — and kept as the path given, so step files are named next
    /// to it — while a link to anything that is not a model file or a
    /// session is still refused by what it points at.
    func testStartSourceFollowsASymbolicLinkToAModelFile() throws {
        let model = tempDir.appendingPathComponent("m.safetensors")
        try Data("x".utf8).write(to: model)
        let link = tempDir.appendingPathComponent("l.safetensors")
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: model)
        XCTAssertEqual(try TrainVsUciSession.startSource(path: link.path), .modelFile(link))

        let fifo = tempDir.appendingPathComponent("pipe")
        XCTAssertEqual(mkfifo(fifo.path, 0o600), 0)
        let fifoLink = tempDir.appendingPathComponent("pipe-link")
        try FileManager.default.createSymbolicLink(at: fifoLink, withDestinationURL: fifo)
        XCTAssertThrowsError(try TrainVsUciSession.startSource(path: fifoLink.path)) { error in
            guard case .notAFileOrFolder? = error as? TrainVsUciSession.StartSourceError else {
                return XCTFail("a link to a FIFO is refused by kind, got \(error)")
            }
        }
    }

    // MARK: - Step-file names

    func testStepFilesKeepTheNamesRunsHaveAlwaysProduced() throws {
        let start = tempDir.appendingPathComponent("champ.safetensors")
        let base = try TrainVsUciSession.enumeratedNamingBase(
            checkpointStem: nil, startSource: .modelFile(start), runModelID: "20261003-1-AbCd")
        let naming = EnumeratedCheckpointNaming(rollingOutputURL: base, runTag: EnumeratedCheckpointNaming.trainVsUciRunTag, segmentIndex: 0)
        XCTAssertEqual(naming.url(step: 1000), tempDir.appendingPathComponent("champ-vsuci-step1000.safetensors"))

        let fresh = try TrainVsUciSession.enumeratedNamingBase(
            checkpointStem: nil, startSource: nil, runModelID: "20261003-1-AbCd")
        XCTAssertEqual(
            EnumeratedCheckpointNaming(rollingOutputURL: fresh, runTag: EnumeratedCheckpointNaming.trainVsUciRunTag, segmentIndex: 0)
                .url(step: 2000),
            CheckpointPaths.modelsDir.appendingPathComponent("20261003-1-AbCd-vsuci-step2000.safetensors"))

        let explicit = try TrainVsUciSession.enumeratedNamingBase(
            checkpointStem: tempDir.appendingPathComponent("sf100-resume2").path,
            startSource: .session(tempDir), runModelID: "20261003-1-AbCd")
        XCTAssertEqual(
            EnumeratedCheckpointNaming(rollingOutputURL: explicit, runTag: EnumeratedCheckpointNaming.trainVsUciRunTag, segmentIndex: 0)
                .url(step: 3000),
            tempDir.appendingPathComponent("sf100-resume2-vsuci-step3000.safetensors"))
    }

    func testACheckpointStemMustNotBeAFileOrAStepName() {
        let withExtension = tempDir.appendingPathComponent("run.safetensors").path
        XCTAssertThrowsError(try TrainVsUciSession.enumeratedNamingBase(
            checkpointStem: withExtension, startSource: nil, runModelID: "id")) { error in
            XCTAssertEqual(error as? TrainVsUciSession.CheckpointStemError, .hasExtension(stem: withExtension))
        }
        let stepName = tempDir.appendingPathComponent("run-vsuci-step4000").path
        XCTAssertThrowsError(try TrainVsUciSession.enumeratedNamingBase(
            checkpointStem: stepName, startSource: nil, runModelID: "id")) { error in
            XCTAssertEqual(error as? TrainVsUciSession.CheckpointStemError,
                           .namedLikeAnEnumeratedCheckpoint(stem: stepName, step: 4000))
        }
    }

    /// A stem is a name, and names may hold dots (`sf100-lr0.5`); only a
    /// model or session extension says the stem names a file instead.
    func testACheckpointStemMayContainADot() throws {
        let stem = tempDir.appendingPathComponent("sf100-lr0.5").path
        let naming = EnumeratedCheckpointNaming(
            rollingOutputURL: try TrainVsUciSession.enumeratedNamingBase(
                checkpointStem: stem, startSource: nil, runModelID: "20261003-1-AbCd"),
            runTag: EnumeratedCheckpointNaming.trainVsUciRunTag, segmentIndex: 0)
        XCTAssertEqual(naming.url(step: 1000), tempDir.appendingPathComponent("sf100-lr0.5-vsuci-step1000.safetensors"))
        for named in ["run.DCMMODEL", "run.dcmsession", "run.SafeTensors"] {
            let path = tempDir.appendingPathComponent(named).path
            XCTAssertThrowsError(try TrainVsUciSession.enumeratedNamingBase(
                checkpointStem: path, startSource: nil, runModelID: "id")) { error in
                XCTAssertEqual(error as? TrainVsUciSession.CheckpointStemError, .hasExtension(stem: path))
            }
        }
    }

    // MARK: - Cadence and failures

    func testAPeriodicSaveIsDueOnceItsIntervalHasElapsed() {
        let t0 = Date(timeIntervalSince1970: 1_800_000_000)
        XCTAssertFalse(TrainVsUciSession.periodicSaveIsDue(now: t0.addingTimeInterval(59), lastSave: t0, intervalSec: 60))
        XCTAssertTrue(TrainVsUciSession.periodicSaveIsDue(now: t0.addingTimeInterval(60), lastSave: t0, intervalSec: 60))
        XCTAssertFalse(TrainVsUciSession.periodicSaveIsDue(now: t0.addingTimeInterval(-1), lastSave: t0, intervalSec: 60),
                       "a clock that stepped backwards is not a due save")
    }

    func testADiskFullInsideASessionSaveIsRecognized() {
        let enospc = NSError(domain: NSPOSIXErrorDomain, code: Int(ENOSPC))
        let url = tempDir.appendingPathComponent("x")
        XCTAssertTrue(CorpusReplayRunner.isOutOfSpace(CheckpointManagerError.writeFailed(url, enospc)))
        XCTAssertTrue(CorpusReplayRunner.isOutOfSpace(CheckpointManagerError.fsyncFailed(url, enospc)))
        XCTAssertTrue(CorpusReplayRunner.isOutOfSpace(
            CheckpointManagerError.writeFailed(url, ReplayBuffer.PersistenceError.writeFailed(enospc))))
        XCTAssertFalse(CorpusReplayRunner.isOutOfSpace(
            CheckpointManagerError.writeFailed(url, NSError(domain: NSPOSIXErrorDomain, code: Int(EACCES)))))
        XCTAssertFalse(CorpusReplayRunner.isOutOfSpace(CheckpointManagerError.targetAlreadyExists(url)))
    }

    // MARK: - The save itself

    func testASessionSaveRoundTripsAsATrainVsUciSessionTheGUIRefuses() async throws {
        let arch = ResumeEquivalenceTests.architecture
        let parameters = TrainingParameters.shared.snapshot()
        let hyperparameters = TrainerHyperparameters(parameters)
        let trainer = try ChessTrainer(
            dropoutStream: DCMRandom(seed: 7), hyperparameters: hyperparameters, arch: arch,
            initialization: .seeded(initSeed: 7))
        let snapshot = try await trainer.exportResumeSnapshot()
        let baseCount = trainer.network.trainableVariables.count + trainer.network.bnRunningStatsVariables.count
        let started = Date(timeIntervalSince1970: 1_800_000_000)
        let tracker = try LineageTracker(
            start: .fresh(initialization: .forTests), pathKind: .vsuci, argv: ["DrewsChessMachine", "--train-vs-uci"],
            startedAt: started, segmentStartTrainerStep: 0)
        let saved = started.addingTimeInterval(120)
        let lineage = try tracker.record(
            at: saved, trainerCompletedSteps: snapshot.schedule.completedTrainSteps,
            segmentLocalStep: 0, segmentGames: 3, segmentPositions: 150, corpus: nil,
            parameters: nil, rng: .withoutRunStreams(dropoutPhiloxState: snapshot.dropoutRNG.philoxState))
        let buffer = try ResumeEquivalenceTests.fixtureBuffer(sampler: DCMRandom(seed: 3))
        let state = TrainVsUciSession.sessionState(
            sessionID: "20261003-9-TeSt", savedAt: saved, runStart: started,
            trainerCompletedSteps: snapshot.schedule.completedTrainSteps,
            parameters: parameters, hyperparameters: hyperparameters, arch: arch,
            bufferSnapshot: buffer.stateSnapshot(), maxPliesPerGame: 400)
        let sessions = tempDir.appendingPathComponent("Sessions", isDirectory: true)
        let url = try await CheckpointManager.saveSession(
            championWeights: Array(snapshot.trainerWeights.prefix(baseCount)),
            championID: "20261003-9-TeSt",
            championMetadata: ModelCheckpointMetadata(creator: "train-vs-uci", trainingStep: 0, parentModelID: "", notes: "test"),
            championCreatedAtUnix: Int64(saved.timeIntervalSince1970),
            trainerWeights: snapshot.trainerWeights,
            trainerID: "20261003-9-TeSt",
            trainerMetadata: ModelCheckpointMetadata.trainerFile(
                creator: "train-vs-uci", trainingStep: 0, parentModelID: "", notes: "test",
                schedule: snapshot.schedule, policyTailPrecision: trainer.policyTailPrecision),
            trainerCreatedAtUnix: Int64(saved.timeIntervalSince1970),
            state: state, lineage: lineage, championLineage: lineage.withoutTrainerState(), architecture: arch,
            replayBuffer: buffer, chartSnapshot: nil,
            trigger: TrainVsUciSession.SaveKind.final.diskTag, at: saved, sessionsDirectory: sessions)

        XCTAssertTrue(url.lastPathComponent.hasSuffix("-20261003-9-TeSt-vsuci-final.dcmsession"), url.lastPathComponent)
        XCTAssertNil(CheckpointPaths.parseAutomaticSaveFolderName(url.lastPathComponent),
                     "a train-vs-UCI save is never in the GUI's automatic-save retention pool")
        guard case .session = try TrainVsUciSession.startSource(path: url.path) else {
            return XCTFail("the saved folder is a session start")
        }

        let loaded = try CheckpointManager.loadSession(at: url)
        XCTAssertEqual(loaded.state.lineage?.invocation.pathKind, .vsuci)
        XCTAssertNotNil(loaded.state.guiLoadRefusal, "the GUI refuses a train-vs-UCI session")
        XCTAssertEqual(loaded.state.selfPlayGames, 0)
        XCTAssertTrue(loaded.state.arenaHistory.isEmpty)
        XCTAssertEqual(loaded.state.batchSize, parameters.trainingBatchSize)
        XCTAssertEqual(loaded.state.hasReplayBuffer, true)
        XCTAssertNotNil(loaded.replayBufferURL)
        XCTAssertEqual(loaded.trainerFile.weights, snapshot.trainerWeights)
        XCTAssertEqual(loaded.championFile.weights, Array(snapshot.trainerWeights.prefix(baseCount)))

        // The restore path an exact resume takes.
        let restored = ReplayBuffer(capacity: 512, inputEncoding: arch.inputEncoding, sampler: DCMRandom(seed: 3))
        try restored.restore(from: XCTUnwrap(loaded.replayBufferURL))
        XCTAssertNoThrow(try CheckpointManager.verifyReplayBufferMatchesSession(buffer: restored, state: loaded.state))
        XCTAssertEqual(restored.stateSnapshot().totalPositionsAdded, buffer.stateSnapshot().totalPositionsAdded)
    }

    func testTheGUIStillLoadsItsOwnSessions() {
        let gui = LineageRecord.sessionTestFixture
        XCTAssertEqual(gui.invocation.pathKind, .gui)
        let state = TrainVsUciSession.sessionState(
            sessionID: "s", savedAt: Date(timeIntervalSince1970: 1_800_000_100),
            runStart: Date(timeIntervalSince1970: 1_800_000_000), trainerCompletedSteps: 0,
            parameters: TrainingParameters.shared.snapshot(),
            hyperparameters: TrainerHyperparameters(TrainingParameters.shared.snapshot()),
            arch: ResumeEquivalenceTests.architecture, bufferSnapshot: nil, maxPliesPerGame: 400)
        XCTAssertNil(state.withLineage(gui).guiLoadRefusal)
        XCTAssertNil(state.guiLoadRefusal, "a state with no lineage (a pre-lineage GUI session) is not refused")
    }
}
