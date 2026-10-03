import XCTest
@testable import DrewsChessMachine

/// Determinism plan phase P6: the `dcm_lineage` record every model file
/// carries from architecture format v7, its writers and readers, and the
/// continuity of steps, games, positions and training time across a chain
/// of sessions.
final class LineageRecordTests: XCTestCase {

    // MARK: - Fixtures

    private let arch = NetworkArchitecture.current

    private func baseWeights() -> [[Float]] {
        arch.weightTensorPlan().enumerated().map { i, spec in
            (0..<spec.elementCount).map { Float((i * 7 + $0) % 13) * 0.01 }
        }
    }

    private func trainerWeights() -> [[Float]] {
        baseWeights() + arch.trainableTensorPlan().map { [Float](repeating: -0.5, count: $0.elementCount) }
    }

    private func schedule(_ steps: Int) -> TrainerScheduleState {
        TrainerScheduleState(completedTrainSteps: steps, lrWarmupSteps: 3, lrMomentumCycle: .disabled)
    }

    private func parameters() throws -> LineageRecord.Parameters {
        try LineageRecord.Parameters(values: ["learning_rate": .double(0.0005), "training_batch_size": .int(4096), "replay_ratio_auto_adjust": .bool(false)])
    }

    /// Encode a trainer-state file carrying `lineage`, the way the CLI
    /// runners write their checkpoints.
    private func trainerFile(modelID: String, steps: Int, lineage: LineageRecord) throws -> Data {
        try SafetensorsModelIO.encode(
            modelID: modelID, createdAtUnix: 1_790_000_000,
            metadata: ModelCheckpointMetadata.trainerFile(
                creator: "replay", trainingStep: steps, parentModelID: "", notes: "lineage test",
                schedule: schedule(steps), policyTailPrecision: .float32FromPreBatchNorm),
            weights: trainerWeights(), architecture: arch, includesVelocity: true, lineage: lineage)
    }

    private func decodedParent(_ data: Data) throws -> (file: ModelCheckpointFile, parent: LineageTracker.ParentFile) {
        let file = try SafetensorsModelIO.decode(data).file
        return (file, file.lineageParent)
    }

    // MARK: - The record

    func testRecordJSONRoundTripsAndWritesUnrecordedTotalsAsExplicitNulls() throws {
        let tracker = try LineageTracker(start: .fresh(initialization: .forTests), pathKind: .replay, argv: ["dcm"],
                                         startedAt: Date(timeIntervalSince1970: 1_000), segmentStartTrainerStep: 0)
        tracker.recordTrainingStep(totalMs: 1500)
        let record = try tracker.record(at: Date(timeIntervalSince1970: 1_100), trainerCompletedSteps: 1,
                                        segmentLocalStep: 1, segmentGames: 3, segmentPositions: 200,
                                        corpus: nil, parameters: try parameters(), rng: .withoutRunStreams(dropoutPhiloxState: nil))
        let text = try record.jsonText()
        XCTAssertEqual(try LineageRecord.decode(jsonText: text), record)
        XCTAssertTrue(text.contains("\"lineage_run_id\""), text)
        XCTAssertTrue(text.contains("\"cum_train_step_sec\":1.5"), text)
        // An absent value is written as null, never omitted.
        XCTAssertTrue(text.contains("\"parent\":null"), text)
        XCTAssertTrue(text.contains("\"corpus\":null"), text)

        let unrecorded = try LineageTracker(
            start: .resume(parent: LineageTracker.ParentFile(modelID: "m", contentSHA256: nil, trainerCompletedSteps: 5,
                                                           lineage: .unrecorded(formatVersion: 6), derivationHistory: []),
                           gaps: [], legacyTotals: nil),
            pathKind: .replay, argv: ["dcm"], startedAt: Date(timeIntervalSince1970: 1_000), segmentStartTrainerStep: 5)
            .record(at: Date(timeIntervalSince1970: 1_010), trainerCompletedSteps: 6, segmentLocalStep: 1,
                    segmentGames: 1, segmentPositions: 60, corpus: nil, parameters: nil, rng: .withoutRunStreams(dropoutPhiloxState: nil))
        let unrecordedText = try unrecorded.jsonText()
        XCTAssertTrue(unrecordedText.contains("\"cum_games\":null"), unrecordedText)
        XCTAssertTrue(unrecordedText.contains("\"cum_train_step_sec\":null"), unrecordedText)
        XCTAssertEqual(try LineageRecord.decode(jsonText: unrecordedText), unrecorded)
    }

    func testRecordMissingAKeyFailsToDecode() throws {
        let record = try LineageRecord.forTests(trainerCompletedSteps: 4, corpus: nil)
        var object = try XCTUnwrap(try JSONSerialization.jsonObject(with: Data(try record.jsonText().utf8)) as? [String: Any])
        var fed = try XCTUnwrap(object["fed"] as? [String: Any])
        fed.removeValue(forKey: "cum_games")
        object["fed"] = fed
        let stripped = String(decoding: try JSONSerialization.data(withJSONObject: object), as: UTF8.self)
        XCTAssertThrowsError(try LineageRecord.decode(jsonText: stripped))
    }

    func testParametersHashMustMatchItsSnapshot() throws {
        let record = try LineageRecord.forTests(trainerCompletedSteps: 4, corpus: nil)
        let withParameters = LineageRecord(
            schema: record.schema, run: record.run, parent: record.parent, steps: record.steps, fed: record.fed,
            time: record.time, parameters: try parameters(), build: record.build, invocation: record.invocation,
            device: record.device, rng: record.rng, segments: record.segments,
            derivationHistory: record.derivationHistory)
        let text = try withParameters.jsonText()
        XCTAssertEqual(try LineageRecord.decode(jsonText: text), withParameters)
        let tampered = text.replacingOccurrences(of: "4096", with: "2048")
        XCTAssertThrowsError(try LineageRecord.decode(jsonText: tampered))
    }

    func testSecretArgumentsAreRedacted() {
        let argv = ["dcm", "--lichess-token", "lip_secret", "--api-key=abc", "--steps", "10", "--TOKEN=x"]
        XCTAssertEqual(LineageRecord.redactedArguments(argv),
                       ["dcm", "--lichess-token", "<redacted>", "--api-key=<redacted>", "--steps", "10", "--TOKEN=<redacted>"])
    }

    // MARK: - Safetensors carriage

    func testFileRoundTripCarriesTheRecordAndItsMirrors() throws {
        let record = try LineageRecord.forTests(trainerCompletedSteps: 12, corpus: nil)
        let data = try trainerFile(modelID: "20261002-1-LNGA", steps: 12, lineage: record)
        let (tensors, md) = try SafetensorsFile.decode(data)
        XCTAssertFalse(tensors.isEmpty)
        XCTAssertEqual(md[SafetensorsModelIO.Key.formatVersion], "8")
        XCTAssertEqual(md[LineageRecord.MirrorKey.lineageRunID], record.run.lineageRunID)
        XCTAssertEqual(md[LineageRecord.MirrorKey.cumTrainerStep], "12")
        XCTAssertEqual(md[LineageRecord.MirrorKey.segmentIndex], "0")

        let file = try SafetensorsModelIO.decode(data).file
        XCTAssertEqual(file.safetensorsProvenance?.lineage, .recorded(record))
        XCTAssertEqual(file.safetensorsProvenance?.contentSHA256, md[SafetensorsFile.contentHashKey])
        let parent = try SafetensorsModelIO.readParentFile(at: try write(data, named: "a.safetensors"))
        XCTAssertEqual(parent.lineage, .recorded(record))
        XCTAssertEqual(parent.trainerCompletedSteps, 12)
        XCTAssertEqual(parent.contentSHA256, md[SafetensorsFile.contentHashKey])
    }

    func testTrainerFileWhoseLineageDisagreesWithItsClockIsRefused() throws {
        let record = try LineageRecord.forTests(trainerCompletedSteps: 11, corpus: nil)
        XCTAssertThrowsError(try trainerFile(modelID: "20261002-1-LNGB", steps: 12, lineage: record)) { error in
            XCTAssertTrue(String(describing: error).contains("trainer_completed_steps"), String(describing: error))
        }
    }

    // MARK: - Dropout RNG state

    func testDropoutPhiloxStateTravelsThroughTheFileAndDecidesTheResumeGap() throws {
        let state = try DropoutPhiloxState(words: [1, 2, 3, 4, 5, 6, -7])
        let start = Date(timeIntervalSince1970: 1_000)
        let tracker = try LineageTracker(start: .fresh(initialization: .forTests), pathKind: .replay, argv: ["dcm"],
                                         startedAt: start, segmentStartTrainerStep: 0)
        let withState = try tracker.record(at: start.addingTimeInterval(5), trainerCompletedSteps: 12, segmentLocalStep: 12,
                                           segmentGames: 1, segmentPositions: 50, corpus: nil, parameters: nil,
                                           rng: .withoutRunStreams(dropoutPhiloxState: state))
        let withStateText = try withState.jsonText()
        XCTAssertTrue(withStateText.contains("\"dropout_philox_state\":[1,2,3,4,5,6,-7]"), withStateText)

        let file = try SafetensorsModelIO.decode(try trainerFile(modelID: "20261002-1-LNGR", steps: 12, lineage: withState)).file
        let snapshot = try TrainerResumeSnapshot(checkpoint: file, fileName: "a.safetensors")
        XCTAssertEqual(snapshot.dropoutRNG, .philox(state))
        let resumed = try LineageTracker(
            start: .resume(parent: file.lineageParent,
                           gaps: [.rngSampler, .feedCarry] + ResumeGap.dropoutGaps(restoring: snapshot.dropoutRNG),
                           legacyTotals: nil),
            pathKind: .replay, argv: ["dcm"], startedAt: start.addingTimeInterval(100), segmentStartTrainerStep: 12)
            .record(at: start.addingTimeInterval(110), trainerCompletedSteps: 13, segmentLocalStep: 1,
                    segmentGames: 0, segmentPositions: 0, corpus: nil, parameters: nil, rng: .withoutRunStreams(dropoutPhiloxState: state))
        XCTAssertFalse(resumed.run.notExactItems.contains(ResumeGap.dropoutState.token), "\(resumed.run.notExactItems)")

        // A record with no trainer snapshot behind it writes an explicit null,
        // and a resume from it names the gap.
        let withoutState = try LineageRecord.forTests(trainerCompletedSteps: 12, corpus: nil)
        let withoutStateText = try withoutState.jsonText()
        XCTAssertTrue(withoutStateText.contains("\"dropout_philox_state\":null"), withoutStateText)
        let bareFile = try SafetensorsModelIO.decode(try trainerFile(modelID: "20261002-1-LNGS", steps: 12, lineage: withoutState)).file
        let bareSnapshot = try TrainerResumeSnapshot(checkpoint: bareFile, fileName: "b.safetensors")
        XCTAssertEqual(bareSnapshot.dropoutRNG, .notInCheckpoint)
        XCTAssertEqual(ResumeGap.dropoutGaps(restoring: bareSnapshot.dropoutRNG), [.dropoutState])
        // A file written before lineage carries no state either.
        XCTAssertEqual(DropoutRNGResumeState(lineage: .unrecorded(formatVersion: 6)), .notInCheckpoint)
    }

    func testRecordWithoutTheDropoutStateKeyFailsToDecode() throws {
        let record = try LineageRecord.forTests(trainerCompletedSteps: 4, corpus: nil)
        var object = try XCTUnwrap(try JSONSerialization.jsonObject(with: Data(try record.jsonText().utf8)) as? [String: Any])
        var rng = try XCTUnwrap(object["rng"] as? [String: Any])
        rng.removeValue(forKey: "dropout_philox_state")
        object["rng"] = rng
        let stripped = String(decoding: try JSONSerialization.data(withJSONObject: object), as: UTF8.self)
        XCTAssertThrowsError(try LineageRecord.decode(jsonText: stripped))
    }

    func testCurrentFormatFileWithoutLineageIsRefused() throws {
        let record = try LineageRecord.forTests(trainerCompletedSteps: nil, corpus: nil)
        let data = try SafetensorsModelIO.encode(
            modelID: "20261002-1-LNGC", createdAtUnix: 1_790_000_000,
            metadata: ModelCheckpointMetadata(creator: "test", trainingStep: nil, parentModelID: "", notes: ""),
            weights: baseWeights(), architecture: arch, includesVelocity: false, lineage: record)
        let (tensors, md) = try SafetensorsFile.decode(data)
        var stripped = md
        stripped.removeValue(forKey: LineageRecord.metadataKey)
        stripped.removeValue(forKey: SafetensorsFile.contentHashKey)
        let rewritten = try SafetensorsFile.encode(tensors: tensors, metadata: stripped)
        XCTAssertThrowsError(try SafetensorsModelIO.decode(rewritten)) { error in
            XCTAssertTrue(String(describing: error).contains(LineageRecord.metadataKey), String(describing: error))
        }
    }

    func testOlderFormatFileLoadsWithItsLineageUnrecorded() throws {
        let record = try LineageRecord.forTests(trainerCompletedSteps: nil, corpus: nil)
        let data = try SafetensorsModelIO.encode(
            modelID: "20261002-1-LNGD", createdAtUnix: 1_790_000_000,
            metadata: ModelCheckpointMetadata(creator: "test", trainingStep: 40, parentModelID: "", notes: ""),
            weights: baseWeights(), architecture: arch, includesVelocity: false, lineage: record)
        let (tensors, md) = try SafetensorsFile.decode(data)
        var legacy = md.filter { !Set([LineageRecord.metadataKey, SafetensorsFile.contentHashKey] + LineageRecord.MirrorKey.all).contains($0.key) }
        legacy[SafetensorsModelIO.Key.formatVersion] = "6"
        let rewritten = try SafetensorsFile.encode(tensors: tensors, metadata: legacy)
        let file = try SafetensorsModelIO.decode(rewritten).file
        XCTAssertEqual(file.safetensorsProvenance?.lineage, .unrecorded(formatVersion: 6))
        XCTAssertEqual(file.lineageParent.trainerCompletedSteps, 40)
    }

    // MARK: - Continuity across sessions

    /// Three exact-resume segments of one run, each segment's file decoded and
    /// resumed by the next: the run identity is stable, the segment index
    /// counts up, every parent hash is the parent file's own, and the step,
    /// game, position and time totals equal the sums an uninterrupted run
    /// would report. A branch from the last file then starts a new run.
    func testExactResumeChainKeepsTotalsContinuousAndABranchStartsANewRun() throws {
        let t0 = Date(timeIntervalSince1970: 2_000_000)

        // Segment 0: fresh, 100 steps of 0.5 s each, 40 games / 2,600 plies.
        let s0 = try LineageTracker(start: .fresh(initialization: .forTests), pathKind: .replay, argv: ["dcm"], startedAt: t0, segmentStartTrainerStep: 0)
        for _ in 0..<100 { s0.recordTrainingStep(totalMs: 500) }
        let r0 = try s0.record(at: t0.addingTimeInterval(80), trainerCompletedSteps: 100, segmentLocalStep: 100,
                               segmentGames: 40, segmentPositions: 2_600, corpus: nil, parameters: try parameters(), rng: .withoutRunStreams(dropoutPhiloxState: nil))
        let file0 = try trainerFile(modelID: "20261002-1-SEG0", steps: 100, lineage: r0)
        let (decoded0, parent0) = try decodedParent(file0)

        // Segment 1: resumes file 0, 50 more steps of 0.25 s, 10 games / 700 plies.
        let t1 = t0.addingTimeInterval(1_000)
        let s1 = try LineageTracker(start: .resume(parent: parent0, gaps: [], legacyTotals: nil),
                                    pathKind: .replay, argv: ["dcm"], startedAt: t1, segmentStartTrainerStep: 100)
        for _ in 0..<50 { s1.recordTrainingStep(totalMs: 250) }
        let r1 = try s1.record(at: t1.addingTimeInterval(20), trainerCompletedSteps: 150, segmentLocalStep: 50,
                               segmentGames: 10, segmentPositions: 700, corpus: nil, parameters: try parameters(), rng: .withoutRunStreams(dropoutPhiloxState: nil))
        let file1 = try trainerFile(modelID: "20261002-2-SEG1", steps: 150, lineage: r1)
        let (_, parent1) = try decodedParent(file1)

        // Segment 2: resumes file 1, 30 steps of 1 s, 5 games / 300 plies.
        let t2 = t1.addingTimeInterval(1_000)
        let s2 = try LineageTracker(start: .resume(parent: parent1, gaps: [], legacyTotals: nil),
                                    pathKind: .replay, argv: ["dcm"], startedAt: t2, segmentStartTrainerStep: 150)
        for _ in 0..<30 { s2.recordTrainingStep(totalMs: 1_000) }
        let r2 = try s2.record(at: t2.addingTimeInterval(40), trainerCompletedSteps: 180, segmentLocalStep: 30,
                               segmentGames: 5, segmentPositions: 300, corpus: nil, parameters: try parameters(), rng: .withoutRunStreams(dropoutPhiloxState: nil))
        let file2 = try trainerFile(modelID: "20261002-3-SEG2", steps: 180, lineage: r2)
        let (_, parent2) = try decodedParent(file2)

        // One run, three segments.
        XCTAssertEqual(r1.run.lineageRunID, r0.run.lineageRunID)
        XCTAssertEqual(r2.run.lineageRunID, r0.run.lineageRunID)
        XCTAssertEqual([r0.run.segmentIndex, r1.run.segmentIndex, r2.run.segmentIndex], [0, 1, 2])
        XCTAssertEqual(Set([r0.run.segmentID, r1.run.segmentID, r2.run.segmentID]).count, 3)
        XCTAssertEqual(r1.run.start, .resume)
        XCTAssertTrue(r1.run.exactResume)

        // Each parent is identified by its file's own content hash.
        XCTAssertEqual(r1.parent?.contentSHA256, decoded0.safetensorsProvenance?.contentSHA256)
        XCTAssertEqual(r1.parent?.contentSHA256, parent0.contentSHA256)
        XCTAssertEqual(r2.parent?.contentSHA256, parent1.contentSHA256)
        XCTAssertEqual(r1.parent?.segmentID, r0.run.segmentID)
        XCTAssertEqual(r1.parent?.trainerCompletedSteps, 100)

        // Totals: exactly the uninterrupted sums.
        XCTAssertEqual(r2.steps.cumTrainerStep, 180)
        XCTAssertEqual(r2.steps.segmentStartTrainerStep, 150)
        XCTAssertEqual(r2.fed.cumGames, 55)
        XCTAssertEqual(r2.fed.cumPositions, 3_600)
        XCTAssertEqual(try XCTUnwrap(r2.time.cumTrainStepSec), 50 + 12.5 + 30, accuracy: 1e-9)
        XCTAssertEqual(try XCTUnwrap(r2.time.cumWallSec), 80 + 20 + 40, accuracy: 1e-9)
        // Monotone along the chain.
        XCTAssertLessThan(try XCTUnwrap(r0.fed.cumGames), try XCTUnwrap(r1.fed.cumGames))
        XCTAssertLessThan(try XCTUnwrap(r1.time.cumTrainStepSec), try XCTUnwrap(r2.time.cumTrainStepSec))

        // The newest file alone describes every earlier segment.
        XCTAssertEqual(r2.segments.map(\.segmentIndex), [0, 1])
        XCTAssertEqual(r2.segments.map(\.segmentID), [r0.run.segmentID, r1.run.segmentID])
        XCTAssertEqual(r2.segments[1].endTrainerStep, 150)
        XCTAssertEqual(r2.segments[0].segmentGames, 40)
        XCTAssertEqual(r2.segments[1].segmentTrainStepSec, 12.5, accuracy: 1e-9)

        // A branch from file 2 is a new run with a fresh clock and totals,
        // recording the file it came from.
        let t3 = t2.addingTimeInterval(1_000)
        let branch = try LineageTracker(start: .branch(parent: parent2), pathKind: .replay, argv: ["dcm"],
                                        startedAt: t3, segmentStartTrainerStep: 0)
        branch.recordTrainingStep(totalMs: 400)
        let rb = try branch.record(at: t3.addingTimeInterval(5), trainerCompletedSteps: 1, segmentLocalStep: 1,
                                   segmentGames: 2, segmentPositions: 90, corpus: nil, parameters: try parameters(), rng: .withoutRunStreams(dropoutPhiloxState: nil))
        XCTAssertNotEqual(rb.run.lineageRunID, r0.run.lineageRunID)
        XCTAssertEqual(rb.run.segmentIndex, 0)
        XCTAssertEqual(rb.run.start, .branch)
        XCTAssertEqual(rb.parent?.contentSHA256, parent2.contentSHA256)
        XCTAssertEqual(rb.parent?.lineageRunID, r0.run.lineageRunID)
        XCTAssertEqual(rb.fed.cumGames, 2)
        XCTAssertEqual(rb.steps.cumTrainerStep, 1)
        XCTAssertEqual(rb.segments, [])
    }

    /// Resuming a file written before lineage starts a new run that says so:
    /// the trainer clock is still known, totals the file never recorded stay
    /// null, and `lineage` is listed among what the resume did not restore.
    func testResumeOfALegacyFileLeavesItsUnrecordedTotalsNull() throws {
        let parent = LineageTracker.ParentFile(modelID: "20260901-1-OLDX", contentSHA256: "abc", trainerCompletedSteps: 41_000,
                                               lineage: .unrecorded(formatVersion: 6), derivationHistory: [])
        let tracker = try LineageTracker(
            start: .resume(parent: parent, gaps: [.rngSampler, .feedCarry], legacyTotals: nil),
            pathKind: .replay, argv: ["dcm"], startedAt: Date(timeIntervalSince1970: 10), segmentStartTrainerStep: 41_000)
        tracker.recordTrainingStep(totalMs: 2_000)
        let record = try tracker.record(at: Date(timeIntervalSince1970: 20), trainerCompletedSteps: 41_001,
                                        segmentLocalStep: 1, segmentGames: 3, segmentPositions: 190,
                                        corpus: nil, parameters: nil, rng: .withoutRunStreams(dropoutPhiloxState: nil))
        XCTAssertTrue(record.run.continuesUnrecordedHistory)
        XCTAssertFalse(record.run.exactResume)
        XCTAssertTrue(record.run.notExactItems.contains(ResumeGap.lineage.token))
        XCTAssertEqual(record.run.segmentIndex, 0)
        XCTAssertEqual(record.steps.cumTrainerStep, 41_001)
        XCTAssertNil(record.fed.cumGames)
        XCTAssertNil(record.fed.cumPositions)
        XCTAssertNil(record.time.cumTrainStepSec)
        XCTAssertNil(record.time.cumWallSec)
        XCTAssertEqual(record.fed.segmentGames, 3)
        XCTAssertEqual(record.time.segmentTrainStepSec, 2, accuracy: 1e-9)
        XCTAssertEqual(record.parent?.trainerCompletedSteps, 41_000)
    }

    /// A GUI session written before lineage recorded its elapsed time, which
    /// continues; its game counters restart at every promotion and are not
    /// used.
    func testLegacyGUISessionContinuesItsElapsedTimeOnly() throws {
        let parent = LineageTracker.ParentFile(modelID: "20260901-2-OLDS", contentSHA256: "def", trainerCompletedSteps: 900,
                                               lineage: .unrecorded(formatVersion: 6), derivationHistory: [])
        let tracker = try LineageTracker(
            start: .resume(parent: parent, gaps: [.rngSampler],
                           legacyTotals: LineageTracker.LegacySessionTotals(wallSec: 3_600)),
            pathKind: .gui, argv: ["dcm"], startedAt: Date(timeIntervalSince1970: 100), segmentStartTrainerStep: 900)
        let record = try tracker.record(at: Date(timeIntervalSince1970: 160), trainerCompletedSteps: 910,
                                        segmentLocalStep: 10, segmentGames: 4, segmentPositions: 250,
                                        corpus: nil, parameters: nil, rng: .withoutRunStreams(dropoutPhiloxState: nil))
        XCTAssertEqual(try XCTUnwrap(record.time.cumWallSec), 3_660, accuracy: 1e-9)
        XCTAssertNil(record.fed.cumGames)
        XCTAssertNil(record.time.cumTrainStepSec)
    }

    func testLegacyTotalsWithARecordedParentAreRefused() throws {
        let record = try LineageRecord.forTests(trainerCompletedSteps: 5, corpus: nil)
        let parent = LineageTracker.ParentFile(modelID: "m", contentSHA256: "x", trainerCompletedSteps: 5, lineage: .recorded(record),
                                               derivationHistory: record.derivationHistory)
        XCTAssertThrowsError(try LineageTracker(
            start: .resume(parent: parent, gaps: [], legacyTotals: LineageTracker.LegacySessionTotals(wallSec: 1)),
            pathKind: .gui, argv: [], startedAt: Date(), segmentStartTrainerStep: 5))
    }

    func testMintRecordIsAFreshUntrainedRun() throws {
        let record = try LineageTracker.mintRecord(pathKind: .newModel, argv: ["dcm", "--new-model"], initialization: .forTests, at: Date(timeIntervalSince1970: 50))
        XCTAssertEqual(record.run.start, .fresh)
        XCTAssertNil(record.parent)
        XCTAssertEqual(record.steps.cumTrainerStep, 0)
        XCTAssertEqual(record.fed.cumGames, 0)
        XCTAssertEqual(record.time.cumTrainStepSec, 0)
        XCTAssertNil(record.parameters)
        XCTAssertEqual(record.invocation.pathKind, .newModel)
    }

    // MARK: - Corpus replay resume point

    func testReplayResumePointComesFromTheLineageCorpusPosition() throws {
        let corpus = LineageRecord.CorpusPosition(corpusID: "corp-1", corpusPath: "/c", epoch: 2, nextGameIndex: 345,
                                                  shard: 3, populatedPlies: 9_000, bufferCapacity: 10_000,
                                                  feedAheadPositions: 0, feedPerStep: 1, shardSHA256: [])
        let record = try LineageRecord.forTests(trainerCompletedSteps: 7, corpus: corpus)
        let url = try write(try trainerFile(modelID: "20261002-1-RPLY", steps: 7, lineage: record), named: "r.safetensors")
        let point = try SafetensorsModelIO.replayResumePoint(at: url)
        XCTAssertEqual(point.corpusID, "corp-1")
        XCTAssertEqual(point.nextGameIndex, 345)
        XCTAssertEqual(point.epoch, 2)
        XCTAssertEqual(point.populatedPlies, 9_000)
        XCTAssertEqual(point.capacity, 10_000)
        XCTAssertEqual(point.builtByGit, record.build.gitHash)

        let noCorpus = try write(try trainerFile(modelID: "20261002-2-RPLY", steps: 7,
                                                 lineage: try LineageRecord.forTests(trainerCompletedSteps: 7, corpus: nil)),
                                 named: "n.safetensors")
        XCTAssertThrowsError(try SafetensorsModelIO.replayResumePoint(at: noCorpus))
    }

    func testReplayResumePointOfALegacyFileUsesItsReplayKeys() throws {
        let data = try trainerFile(modelID: "20261002-1-LEGR", steps: 7,
                                   lineage: try LineageRecord.forTests(trainerCompletedSteps: 7, corpus: nil))
        let (tensors, md) = try SafetensorsFile.decode(data)
        var legacy = md.filter { !Set([LineageRecord.metadataKey, SafetensorsFile.contentHashKey] + LineageRecord.MirrorKey.all).contains($0.key) }
        legacy[SafetensorsModelIO.Key.formatVersion] = "6"
        legacy["replay_corpus_id"] = "corp-legacy"
        legacy["replay_next_game_index"] = "21"
        legacy["replay_epoch"] = "1"
        let url = try write(try SafetensorsFile.encode(tensors: tensors, metadata: legacy), named: "l.safetensors")
        let point = try SafetensorsModelIO.replayResumePoint(at: url)
        XCTAssertEqual(point.corpusID, "corp-legacy")
        XCTAssertEqual(point.nextGameIndex, 21)
        XCTAssertEqual(point.epoch, 1)
    }

    /// A pre-lineage replay file whose `replay_*` keys are `keys`: the
    /// trainer file's tensors and metadata, without its lineage, at format 6.
    private func legacyReplayFile(_ keys: [String: String], named name: String) throws -> URL {
        let data = try trainerFile(modelID: "20261003-1-LEGK", steps: 7,
                                   lineage: try LineageRecord.forTests(trainerCompletedSteps: 7, corpus: nil))
        let (tensors, md) = try SafetensorsFile.decode(data)
        var legacy = md.filter { !Set([LineageRecord.metadataKey, SafetensorsFile.contentHashKey] + LineageRecord.MirrorKey.all).contains($0.key) }
        legacy[SafetensorsModelIO.Key.formatVersion] = "6"
        legacy.merge(keys) { _, new in new }
        return try write(try SafetensorsFile.encode(tensors: tensors, metadata: legacy), named: name)
    }

    /// A missing position is not game 0: resuming there would silently
    /// retrain the corpus from its start.
    func testLegacyResumePointWithoutNextGameIndexThrows() throws {
        let url = try legacyReplayFile(["replay_corpus_id": "corp-legacy", "replay_epoch": "1"], named: "no-next.safetensors")
        XCTAssertThrowsError(try SafetensorsModelIO.replayResumePoint(at: url)) { error in
            XCTAssertTrue(String(describing: error).contains("replay_next_game_index"), "\(error)")
        }
    }

    func testLegacyResumePointWithMalformedEpochThrows() throws {
        let url = try legacyReplayFile(["replay_corpus_id": "corp-legacy", "replay_next_game_index": "21",
                                        "replay_epoch": "abc"], named: "bad-epoch.safetensors")
        XCTAssertThrowsError(try SafetensorsModelIO.replayResumePoint(at: url)) { error in
            XCTAssertTrue(String(describing: error).contains("replay_epoch"), "\(error)")
            XCTAssertTrue(String(describing: error).contains("abc"), "\(error)")
        }
    }

    // MARK: - Derived models

    func testDerivedModelStartsANewRunCarryingTheSourceTotals() throws {
        let source = try SafetensorsModelIO.encode(
            modelID: "20261002-1-DSRC", createdAtUnix: 1_790_000_000,
            metadata: ModelCheckpointMetadata(creator: "test", trainingStep: nil, parentModelID: "", notes: ""),
            weights: baseWeights(), architecture: arch, includesVelocity: false,
            lineage: try LineageRecord.forTests(trainerCompletedSteps: nil, corpus: nil))
        let sourceHash = try SafetensorsFile.decode(source).metadata[SafetensorsFile.contentHashKey]
        let result = try ModelDerivation.derive(
            sourceData: source, sourceName: "src.safetensors",
            operations: [SetActivationDeriveOperation(value: .leakyRelu)],
            newModelID: "20261002-2-DDRV", createdAtUnix: 1_790_000_100, build: "test",
            invocationArguments: ["dcm", "--derive-model"])
        let derived = try SafetensorsModelIO.decode(result.data).file
        let lineage = try XCTUnwrap(derived.safetensorsProvenance?.lineage.record)
        XCTAssertEqual(lineage.run.start, .derive)
        XCTAssertEqual(lineage.invocation.pathKind, .derive)
        XCTAssertEqual(lineage.parent?.modelID, "20261002-1-DSRC")
        XCTAssertEqual(lineage.parent?.contentSHA256, sourceHash)
        XCTAssertEqual(lineage.steps.segmentLocalStep, 0)
    }

    // MARK: - session.json

    func testSessionStateAtTheCurrentVersionRequiresALineage() throws {
        let state = SessionCheckpointState(
            formatVersion: SessionCheckpointState.currentFormatVersion, sessionID: "s", savedAtUnix: 1, sessionStartUnix: 0,
            elapsedTrainingSec: 1, trainingSteps: 1, selfPlayGames: 1, selfPlayMoves: 1, trainingPositionsSeen: 1,
            batchSize: 1, learningRate: 0.001, promoteThreshold: 0.55, arenaGames: 2,
            selfPlayTau: TauConfigCodable(SamplingSchedule.selfPlay), arenaTau: TauConfigCodable(SamplingSchedule.arena),
            selfPlayWorkerCount: 1, championID: "c", trainerID: "t", arenaHistory: [])
        XCTAssertThrowsError(try state.encode())
        let withLineage = state.withLineage(LineageRecord.sessionTestFixture)
        let encoded = try withLineage.encode()
        XCTAssertEqual(try SessionCheckpointState.decode(encoded), withLineage)

        var object = try XCTUnwrap(try JSONSerialization.jsonObject(with: encoded) as? [String: Any])
        object.removeValue(forKey: "lineage")
        let stripped = try JSONSerialization.data(withJSONObject: object)
        XCTAssertThrowsError(try SessionCheckpointState.decode(stripped)) { error in
            guard case SessionCheckpointError.missingLineage = error else {
                return XCTFail("expected missingLineage, got \(error)")
            }
        }
        object["formatVersion"] = 1
        let legacy = try SessionCheckpointState.decode(try JSONSerialization.data(withJSONObject: object))
        XCTAssertNil(legacy.lineage)
    }

    // MARK: - Helpers

    private var temporaryDirectory: URL!

    override func setUpWithError() throws {
        temporaryDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent("LineageRecordTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: temporaryDirectory, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try FileManager.default.removeItem(at: temporaryDirectory)
    }

    private func write(_ data: Data, named name: String) throws -> URL {
        let url = temporaryDirectory.appendingPathComponent(name)
        try data.write(to: url)
        return url
    }
}
