//
//  ReplayResumeRecordedParametersTests.swift
//  DrewsChessMachineTests
//
//  A CLI exact resume trains under the checkpoint's own LR warmup and
//  LR/momentum cycle, whatever its `--parameters` say. These tests pin that
//  the resumed segment's lineage snapshot records that adopted schedule —
//  the one its file's flat `trainer_*` keys are written from — rather than
//  the configured one, and that the resume's `[RESUME-DIFF]` comparison
//  reports every other changed parameter without tripping over an adopted
//  schedule, a retired key or a range that narrowed since.
//
//  The end-to-end case runs the real replay loop on `ResumeEquivalenceTests`'
//  synthetic sealed corpus and start model.
//

import XCTest
@testable import DrewsChessMachine

@MainActor
final class ReplayResumeRecordedParametersTests: XCTestCase {

    private var tempDir: URL!

    override func setUp() async throws {
        try await super.setUp()
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("dcm-replay-resume-params-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        tempDir = dir
    }

    override func tearDown() async throws {
        if let tempDir {
            try FileManager.default.removeItem(at: tempDir)
        }
        try await super.tearDown()
    }

    /// The ids of the schedule keys `adoptingSchedule` writes.
    private static let scheduleKeyIDs: Set<String> = [
        LRWarmupSteps.id,
        LRCycleEnabled.id, LRCyclePeriodSteps.id, LRCycleCount.id, LRCycleMin.id, LRCycleMax.id, LRCycleInvert.id,
        MomentumCycleEnabled.id, MomentumCyclePeriodSteps.id, MomentumCycleCount.id, MomentumCycleMin.id,
        MomentumCycleMax.id, MomentumCycleInvert.id,
        LRCyclePeakEnd.id, LRCycleTroughEnd.id, LRCycleDecayHorizonSteps.id,
        MomentumFollowsLRCycle.id, MomentumFollowStartLow.id, MomentumFollowStartHigh.id,
        MomentumFollowEndLow.id, MomentumFollowEndHigh.id,
    ]

    /// Declared-default parameters sized for the synthetic corpus, with the
    /// given warmup and LR-cycle period.
    private func params(warmup: Int, cyclePeriod: Int) throws -> ReplayParams {
        try ReplayParams(TrainingParametersSnapshot.declaredDefaults(overriding: [
            TrainingBatchSize.id: .int(32),
            ReplayBufferCapacity.id: .int(2000),
            ReplayBufferMinPositionsBeforeTraining.id: .int(500),
            KLProbeInterval.id: .int(0),
            BatchStatsInterval.id: .int(0),
            LRWarmupSteps.id: .int(warmup),
            LRCycleEnabled.id: .bool(true),
            LRCyclePeriodSteps.id: .int(cyclePeriod),
        ]))
    }

    private func config(corpus: URL, stepLimit: Int, startModel: URL, resumeExact: Bool, out: URL) -> CorpusReplayConfig {
        CorpusReplayConfig(
            corpusDirectories: [corpus],
            stepLimit: stepLimit,
            epochs: nil,
            startModelPath: startModel.path,
            presetName: nil,
            startShard: nil,
            startGameIndex: nil,
            resumeExact: resumeExact,
            acceptInexact: [],
            outModelPath: out.path,
            overwriteOutModel: false,
            runModelID: "20261006-4-RSPR",
            output: nil,
            runRandomSeed: RunRandomSeed.resolve(
                mode: .seeded, configuredSeed: 0x5C4E, commandLineSeed: nil,
                drawSeed: { preconditionFailure("a seeded run never draws its seed") }))
    }

    /// The lineage snapshot of `record` as a parameter snapshot.
    private func snapshot(of parameters: LineageRecord.Parameters) throws -> TrainingParametersSnapshot {
        let values = try JSONDecoder().decode([String: ParameterValue].self, from: Data(parameters.snapshotJSON.utf8))
        return try TrainingParametersSnapshot.declaredDefaults(overriding: values)
    }

    /// `JSONSerialization` writes a `Double` with every significant digit
    /// and does not always read that spelling back to the same `Double`;
    /// the comparison reads with `JSONDecoder`, which does.
    func testASnapshotDoesNotDifferFromItsOwnRecord() throws {
        let snapshot = try TrainingParametersSnapshot.declaredDefaults(overriding: [
            WeightDecay.id: .double(0.0003), ValueLabelSmoothingEpsilon.id: .double(0.013),
        ])
        let parameters = try LineageRecord.Parameters(values: snapshot.rawValueMap())
        XCTAssertEqual(try snapshot.differences(fromLineage: parameters), [])
        let decoded = try JSONDecoder().decode([String: ParameterValue].self, from: Data(parameters.snapshotJSON.utf8))
        XCTAssertEqual(decoded[WeightDecay.id], .double(0.0003))
        XCTAssertEqual(decoded[ValueLabelSmoothingEpsilon.id], .double(0.013))
    }

    // MARK: - End to end (regression)

    func testAnExactResumeRecordsTheScheduleItTrainsUnder() async throws {
        let corpus = try ResumeEquivalenceTests.writeCorpus(in: tempDir)
        let start = try await ResumeEquivalenceTests.writeStartModel(in: tempDir)
        let first = tempDir.appendingPathComponent("first.safetensors")
        _ = try await CorpusReplayRunner.runReplay(
            config: config(corpus: corpus, stepLimit: 3, startModel: start, resumeExact: false, out: first),
            params: try params(warmup: 5, cyclePeriod: 40), abort: ReplayAbortFlag())

        let second = tempDir.appendingPathComponent("second.safetensors")
        _ = try await CorpusReplayRunner.runReplay(
            config: config(corpus: corpus, stepLimit: 3, startModel: first, resumeExact: true, out: second),
            params: try params(warmup: 7, cyclePeriod: 80), abort: ReplayAbortFlag())

        let file = try CheckpointManager.loadModelFile(at: second)
        let record = try XCTUnwrap(file.lineageParent.lineage.record, "the resumed segment's file has a lineage record")
        XCTAssertEqual(record.run.segmentIndex, 1, "the second run is an exact resume of the first")
        let recorded = try snapshot(of: try XCTUnwrap(record.parameters))
        XCTAssertEqual(recorded.lrWarmupSteps, 5, "the snapshot records the checkpoint's warmup, which the trainer ran")
        XCTAssertEqual(recorded.lrCyclePeriodSteps, 40, "the snapshot records the checkpoint's cycle period")
        let flat = try XCTUnwrap(file.metadata.trainerSchedule, "a trainer-state file carries its schedule")
        XCTAssertEqual(recorded.lrWarmupSteps, flat.lrWarmupSteps, "the snapshot agrees with the flat trainer_* keys")
        XCTAssertEqual(recorded.lrMomentumCycle, flat.lrMomentumCycle, "every cycle and envelope key agrees with the flat keys")
    }

    // MARK: - adoptingSchedule

    func testSnapshotAdoptingScheduleRoundTripsEveryField() throws {
        let base = try TrainingParametersSnapshot.declaredDefaults(overriding: [WeightDecay.id: .double(0.00025)])
        var rng = DCMRandom(seed: 20261006)
        for trial in 0..<200 {
            let envelope = LRMomentumCycleEnvelope(
                lrPeakEnd: Double.random(in: 1e-6...1e-2, using: &rng),
                lrTroughEnd: Double.random(in: 1e-8...1e-4, using: &rng),
                decayHorizonSteps: Int.random(in: 0...2_000_000, using: &rng),
                momentumFollowsLRCycle: Bool.random(using: &rng),
                momentumFollowStartLow: Double.random(in: 0...0.99, using: &rng),
                momentumFollowStartHigh: Double.random(in: 0...0.99, using: &rng),
                momentumFollowEndLow: Double.random(in: 0...0.99, using: &rng),
                momentumFollowEndHigh: Double.random(in: 0...0.99, using: &rng))
            let cycle = LRMomentumCycle(
                lrEnabled: Bool.random(using: &rng),
                lrPeriodSteps: Int.random(in: 1...10_000_000, using: &rng),
                lrCount: Int.random(in: 0...1000, using: &rng),
                lrMin: Double.random(in: 1e-7...1e-3, using: &rng),
                lrMax: Double.random(in: 1e-4...1e-1, using: &rng),
                lrInvert: Bool.random(using: &rng),
                momentumEnabled: Bool.random(using: &rng),
                momentumPeriodSteps: Int.random(in: 1...10_000_000, using: &rng),
                momentumCount: Int.random(in: 0...1000, using: &rng),
                momentumMin: Double.random(in: 0...0.9, using: &rng),
                momentumMax: Double.random(in: 0.5...0.999, using: &rng),
                momentumInvert: Bool.random(using: &rng),
                envelope: envelope)
            let schedule = TrainerScheduleState(
                completedTrainSteps: Int.random(in: 0...1_000_000, using: &rng),
                lrWarmupSteps: Int.random(in: 0...100_000, using: &rng),
                lrMomentumCycle: cycle)

            let adopted = base.adoptingSchedule(schedule)
            XCTAssertEqual(adopted.lrWarmupSteps, schedule.lrWarmupSteps, "trial \(trial): warmup")
            XCTAssertEqual(adopted.lrMomentumCycle, schedule.lrMomentumCycle, "trial \(trial): cycle and envelope")
            XCTAssertEqual(adopted.lrMomentumCycle.envelope, envelope, "trial \(trial): envelope")
            let baseValues = base.rawValueMap()
            let adoptedValues = adopted.rawValueMap()
            XCTAssertEqual(Set(adoptedValues.keys), Set(baseValues.keys), "trial \(trial): no key added or dropped")
            for (id, value) in baseValues where !Self.scheduleKeyIDs.contains(id) {
                XCTAssertEqual(adoptedValues[id], value, "trial \(trial): \(id) is not a schedule key and is unchanged")
            }
        }
    }

    func testAdoptingScheduleKeepsAnOutOfRangeCheckpointValue() throws {
        let base = try TrainingParametersSnapshot.declaredDefaults(overriding: [:])
        var cycle = base.lrMomentumCycle
        cycle.lrPeriodSteps = 0
        let schedule = TrainerScheduleState(completedTrainSteps: 10, lrWarmupSteps: 200_000, lrMomentumCycle: cycle)
        XCTAssertThrowsError(try LRCyclePeriodSteps.validateAgainstDeclaration(0), "the test value is outside today's range")
        XCTAssertThrowsError(try LRWarmupSteps.validateAgainstDeclaration(200_000), "the test value is outside today's range")

        let adopted = base.adoptingSchedule(schedule)
        XCTAssertEqual(adopted.lrCyclePeriodSteps, 0, "the checkpoint's value is what ran: never clamped")
        XCTAssertEqual(adopted.lrWarmupSteps, 200_000, "the checkpoint's value is what ran: never clamped")
        XCTAssertEqual(adopted.rawValueMap()[LRCyclePeriodSteps.id], .int(0))
    }

    // MARK: - differences(fromLineage:)

    func testDifferencesListsEveryChangedKeyAndNothingElse() throws {
        let parent = try LineageRecord.Parameters(values: TrainingParametersSnapshot.declaredDefaults(overriding: [
            WeightDecay.id: .double(0.0003),
            EntropyBonus.id: .double(0.01),
            DropoutRate.id: .double(0.1),
        ]).rawValueMap())
        let thisRun = try TrainingParametersSnapshot.declaredDefaults(overriding: [
            WeightDecay.id: .double(0.0005),
            DropoutRate.id: .double(0.1),
            TrainingBatchSize.id: .int(2048),
        ])
        let differences = try thisRun.differences(fromLineage: parent)
        let defaultEntropyBonus = try TrainingParametersSnapshot.declaredDefaults(overriding: [:]).entropyBonus
        XCTAssertEqual(differences.map(\.id), [EntropyBonus.id, TrainingBatchSize.id, WeightDecay.id].sorted())
        let weightDecay = try XCTUnwrap(differences.first { $0.id == WeightDecay.id })
        XCTAssertEqual(weightDecay.logLine, "[RESUME-DIFF] weight_decay: parent=0.0003 this_run=0.0005")
        XCTAssertEqual(weightDecay.presence, .both)
        let entropy = try XCTUnwrap(differences.first { $0.id == EntropyBonus.id })
        XCTAssertEqual(entropy.parentValue, "0.01")
        XCTAssertEqual(entropy.thisRunValue, "\(defaultEntropyBonus)")
        XCTAssertTrue(differences.allSatisfy { !$0.parentValueOutsideTodaysRange && !$0.isSeedSetting })
    }

    /// A whole-number `Double` may be written to JSON as an integer, and a
    /// parameters file may hold one; either way it is the same value.
    func testAJSONIntegerForADoubleParameterIsNotADifference() throws {
        var values = try TrainingParametersSnapshot.declaredDefaults(overriding: [:]).rawValueMap()
        values[ReplayRatioTarget.id] = .int(1)
        let parent = try LineageRecord.Parameters(values: values)
        let thisRun = try TrainingParametersSnapshot.declaredDefaults(overriding: [ReplayRatioTarget.id: .double(1.0)])
        XCTAssertEqual(try thisRun.differences(fromLineage: parent), [])
    }

    func testAnAdoptedScheduleIsNotADifference() throws {
        let parentSnapshot = try params(warmup: 5, cyclePeriod: 40).parameters
        // What a schema-3 writer records: every parameter but the seed
        // settings (plan O-9).
        let parent = try LineageRecord.Parameters(values: parentSnapshot.lineageValues())
        let schedule = TrainerScheduleState(completedTrainSteps: 3, lrWarmupSteps: parentSnapshot.lrWarmupSteps,
                                            lrMomentumCycle: parentSnapshot.lrMomentumCycle)
        let configured = try params(warmup: 7, cyclePeriod: 80)
        XCTAssertEqual(try configured.parameters.differences(fromLineage: parent).map(\.id).sorted(),
                       [LRCyclePeriodSteps.id, LRWarmupSteps.id].sorted(), "before adoption the schedule differs")
        let inForce = try configured.adoptingSchedule(schedule)
        XCTAssertEqual(try inForce.parameters.differences(fromLineage: parent), [], "the adopted schedule is the parent's")
        XCTAssertEqual(inForce.trainer, configured.trainer.adoptingSchedule(schedule),
                       "rebuilding from the adopted snapshot configures the trainer as adopting its schedule does")
        XCTAssertEqual(inForce.lineageParameters, parent, "the run records the snapshot the parent recorded")
    }

    func testAParentOnlyKeyIsReportedNotDropped() throws {
        var values = try TrainingParametersSnapshot.declaredDefaults(overriding: [:]).rawValueMap()
        values["retired_knob"] = .double(1.5)
        let parent = try LineageRecord.Parameters(values: values)
        let differences = try TrainingParametersSnapshot.declaredDefaults(overriding: [:]).differences(fromLineage: parent)
        XCTAssertEqual(differences.count, 1)
        let retired = try XCTUnwrap(differences.first)
        XCTAssertEqual(retired.id, "retired_knob")
        XCTAssertEqual(retired.presence, .parentOnly)
        XCTAssertEqual(retired.parentValue, "1.5")
        XCTAssertNil(retired.thisRunValue)
        XCTAssertTrue(retired.logLine.hasPrefix("[RESUME-DIFF] retired_knob: parent=1.5 this_run=absent (parent only"),
                      retired.logLine)
    }

    func testAKeyThisRunAddedIsReportedButSeedSettingsAreNot() throws {
        var values = try TrainingParametersSnapshot.declaredDefaults(overriding: [:]).rawValueMap()
        values[WeightDecay.id] = nil
        values[RandomSeed.id] = nil
        values[RandomSeedModeParameter.id] = nil
        let parent = try LineageRecord.Parameters(values: values)
        let differences = try TrainingParametersSnapshot.declaredDefaults(overriding: [:]).differences(fromLineage: parent)
        XCTAssertEqual(differences.map(\.id), [WeightDecay.id],
                       "a parent without the seed settings is not compared on them")
        XCTAssertEqual(differences.first?.presence, .thisRunOnly)
    }

    func testAChangedSeedSettingIsInformational() throws {
        let parent = try LineageRecord.Parameters(values: TrainingParametersSnapshot.declaredDefaults(overriding: [
            RandomSeed.id: .uint64(777),
        ]).rawValueMap())
        let differences = try TrainingParametersSnapshot.declaredDefaults(overriding: [:]).differences(fromLineage: parent)
        XCTAssertEqual(differences.map(\.id), [RandomSeed.id])
        XCTAssertEqual(differences.first?.isSeedSetting, true)
        XCTAssertEqual(differences.first?.logLine.contains("informational"), true)
    }

    func testAnOutOfRangeParentValueIsReportedNotThrown() throws {
        var values = try TrainingParametersSnapshot.declaredDefaults(overriding: [:]).rawValueMap()
        values[TrainingBatchSize.id] = .int(16)
        XCTAssertThrowsError(try TrainingBatchSize.validateAgainstDeclaration(16), "16 is below today's range")
        let parent = try LineageRecord.Parameters(values: values)
        let differences = try TrainingParametersSnapshot.declaredDefaults(overriding: [:]).differences(fromLineage: parent)
        XCTAssertEqual(differences.map(\.id), [TrainingBatchSize.id])
        let batch = try XCTUnwrap(differences.first)
        XCTAssertEqual(batch.parentValue, "16")
        XCTAssertTrue(batch.parentValueOutsideTodaysRange)
        XCTAssertTrue(batch.logLine.contains("out of today's range"), batch.logLine)
    }

    func testAParentValueOfTheWrongTypeThrows() throws {
        var values = try TrainingParametersSnapshot.declaredDefaults(overriding: [:]).rawValueMap()
        values[TrainingBatchSize.id] = .bool(true)
        let parent = try LineageRecord.Parameters(values: values)
        XCTAssertThrowsError(try TrainingParametersSnapshot.declaredDefaults(overriding: [:]).differences(fromLineage: parent))
        XCTAssertThrowsError(try ParameterDifference.exactResumeLogLines(
            parent: parent, inForce: TrainingParametersSnapshot.declaredDefaults(overriding: [:]))) { error in
            XCTAssertTrue(error is CLIRunRefusal, "a CLI resume refuses: \(error)")
        }
    }
}
