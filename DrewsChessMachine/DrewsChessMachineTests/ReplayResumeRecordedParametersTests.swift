//
//  ReplayResumeRecordedParametersTests.swift
//  DrewsChessMachineTests
//
//  A CLI exact resume trains under its configured LR warmup and LR/momentum
//  cycle (owner decision O-4, LR_KEYFRAMES_PLAN.md). A configured schedule
//  that feeds the same values as the checkpoint's (compared in behavior form,
//  owner decision Q-5) is exact; one that feeds different values is the
//  `schedule` resume gap, refused unless `--accept-inexact schedule`. These
//  tests pin that, that the resumed segment's lineage snapshot records the
//  configured schedule it trains under — the one its file's flat `trainer_*`
//  keys are written from — and that the resume's `[RESUME-DIFF]` comparison
//  reports every changed parameter without tripping over a retired key or a
//  range that narrowed since.
//
//  The end-to-end case runs the real replay loop on `ResumeEquivalenceTests`'
//  synthetic sealed corpus and start model.
//

import XCTest
import TrainingParametersMacroSupport
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
        LRCycleEnabled.id, LRCyclePeriodSteps.id, LRCycleCount.id, LRCycleInvert.id, LRCycleKeyframes.id,
        MomentumCycleEnabled.id, MomentumCyclePeriodSteps.id, MomentumCycleCount.id, MomentumCycleMin.id,
        MomentumCycleMax.id, MomentumCycleInvert.id, MomentumFollowsLRCycle.id,
    ]

    /// Declared-default parameters sized for the synthetic corpus, with the
    /// given warmup and LR-cycle period (plus `extra`). The declared defaults
    /// have momentum following the LR cycle, so the independent momentum
    /// cycle's keys are inert.
    private func params(warmup: Int, cyclePeriod: Int, extra: [String: ParameterValue] = [:]) throws -> ReplayParams {
        var overrides: [String: ParameterValue] = [
            TrainingBatchSize.id: .int(32),
            ReplayBufferCapacity.id: .int(2000),
            ReplayBufferMinPositionsBeforeTraining.id: .int(500),
            KLProbeInterval.id: .int(0),
            BatchStatsInterval.id: .int(0),
            LRWarmupSteps.id: .int(warmup),
            LRCycleEnabled.id: .bool(true),
            LRCyclePeriodSteps.id: .int(cyclePeriod),
        ]
        overrides.merge(extra) { _, new in new }
        return try ReplayParams(TrainingParametersSnapshot.declaredDefaults(overriding: overrides))
    }

    private func config(corpus: URL, stepLimit: Int, startModel: URL, resumeExact: Bool, out: URL,
                        acceptInexact: Set<ResumeGap> = []) -> CorpusReplayConfig {
        CorpusReplayConfig(
            corpusDirectories: [corpus],
            stepLimit: stepLimit,
            epochs: nil,
            startModelPath: startModel.path,
            presetName: nil,
            startShard: nil,
            startGameIndex: nil,
            resumeExact: resumeExact,
            acceptInexact: acceptInexact,
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

    /// Changed with LR keyframes (owner decision O-4): this pinned that the
    /// resumed segment records the checkpoint's adopted schedule; an exact
    /// resume now trains under — and records — the configured one, and a
    /// configured schedule that feeds different values is the `schedule`
    /// gap, refused until `--accept-inexact schedule` accepts it.
    func testAnExactResumeRecordsTheScheduleItTrainsUnder() async throws {
        let corpus = try ResumeEquivalenceTests.writeCorpus(in: tempDir)
        let start = try await ResumeEquivalenceTests.writeStartModel(in: tempDir)
        let first = tempDir.appendingPathComponent("first.safetensors")
        _ = try await CorpusReplayRunner.runReplay(
            config: config(corpus: corpus, stepLimit: 3, startModel: start, resumeExact: false, out: first),
            params: try params(warmup: 5, cyclePeriod: 40), abort: ReplayAbortFlag())

        // A different warmup (still warming up at trainer step 3) and cycle
        // period: different values fed, so `--resume-exact` alone refuses.
        let refusedOut = tempDir.appendingPathComponent("second-refused.safetensors")
        do {
            _ = try await CorpusReplayRunner.runReplay(
                config: config(corpus: corpus, stepLimit: 3, startModel: first, resumeExact: true, out: refusedOut),
                params: try params(warmup: 7, cyclePeriod: 80), abort: ReplayAbortFlag())
            XCTFail("a schedule that feeds different values is a gap --resume-exact refuses without --accept-inexact")
        } catch let refusal as CLIRunRefusal {
            XCTAssertTrue(refusal.message.contains("--accept-inexact \(ResumeGap.schedule.token)"), refusal.message)
        }
        XCTAssertFalse(FileManager.default.fileExists(atPath: refusedOut.path), "a refused resume writes nothing")

        let second = tempDir.appendingPathComponent("second.safetensors")
        _ = try await CorpusReplayRunner.runReplay(
            config: config(corpus: corpus, stepLimit: 3, startModel: first, resumeExact: true, out: second,
                           acceptInexact: [.schedule]),
            params: try params(warmup: 7, cyclePeriod: 80), abort: ReplayAbortFlag())

        let file = try CheckpointManager.loadModelFile(at: second)
        let record = try XCTUnwrap(file.lineageParent.lineage.record, "the resumed segment's file has a lineage record")
        XCTAssertEqual(record.run.segmentIndex, 1, "the second run is a resume of the first")
        XCTAssertEqual(record.run.notExactItems, [ResumeGap.schedule.token], "the schedule difference is its one gap")
        let recorded = try snapshot(of: try XCTUnwrap(record.parameters))
        XCTAssertEqual(recorded.lrWarmupSteps, 7, "the snapshot records the configured warmup, which the trainer ran")
        XCTAssertEqual(recorded.lrCyclePeriodSteps, 80, "the snapshot records the configured cycle period")
        let flat = try XCTUnwrap(file.metadata.trainerSchedule, "a trainer-state file carries its schedule")
        XCTAssertEqual(flat.lrWarmupSteps, 7, "the trainer ran the configured warmup")
        XCTAssertEqual(flat.lrMomentumCycle, try params(warmup: 7, cyclePeriod: 80).trainer.lrMomentumCycle,
                       "the trainer ran the configured cycle")
        XCTAssertEqual(flat.completedTrainSteps, 6, "the clock continued from the checkpoint's")
        XCTAssertEqual(recorded.lrWarmupSteps, flat.lrWarmupSteps, "the snapshot agrees with the flat trainer_* keys")
        XCTAssertEqual(recorded.lrMomentumCycle, flat.lrMomentumCycle, "every cycle and keyframe key agrees with the flat keys")
    }

    /// A configured schedule that differs from the checkpoint's only in inert
    /// keys (the independent momentum cycle, unused while momentum follows
    /// the LR cycle) feeds the same values: the resume is exact with no
    /// `schedule` gap (owner decision Q-5), and the run records and keeps
    /// its configured values.
    func testAnExactResumeUnderAnInertlyDifferentScheduleIsExact() async throws {
        let corpus = try ResumeEquivalenceTests.writeCorpus(in: tempDir)
        let start = try await ResumeEquivalenceTests.writeStartModel(in: tempDir)
        let first = tempDir.appendingPathComponent("first.safetensors")
        let firstParams = try params(warmup: 5, cyclePeriod: 40)
        XCTAssertTrue(firstParams.parameters.momentumFollowsLRCycle && firstParams.parameters.momentumCycleEnabled,
                      "the fixture relies on momentum following the LR cycle")
        _ = try await CorpusReplayRunner.runReplay(
            config: config(corpus: corpus, stepLimit: 3, startModel: start, resumeExact: false, out: first),
            params: firstParams, abort: ReplayAbortFlag())

        let inertPeriod = firstParams.parameters.momentumCyclePeriodSteps + 17
        let configured = try params(warmup: 5, cyclePeriod: 40, extra: [MomentumCyclePeriodSteps.id: .int(inertPeriod)])
        let second = tempDir.appendingPathComponent("second.safetensors")
        _ = try await CorpusReplayRunner.runReplay(
            config: config(corpus: corpus, stepLimit: 3, startModel: first, resumeExact: true, out: second),
            params: configured, abort: ReplayAbortFlag())

        let file = try CheckpointManager.loadModelFile(at: second)
        let record = try XCTUnwrap(file.lineageParent.lineage.record)
        XCTAssertEqual(record.run.segmentIndex, 1)
        XCTAssertEqual(record.run.notExactItems, [], "an inert difference is not a gap")
        let recorded = try snapshot(of: try XCTUnwrap(record.parameters))
        XCTAssertEqual(recorded.momentumCyclePeriodSteps, inertPeriod, "the configured value is recorded")
        let flat = try XCTUnwrap(file.metadata.trainerSchedule)
        XCTAssertEqual(flat.lrMomentumCycle, configured.trainer.lrMomentumCycle, "the trainer kept the configured cycle")
        XCTAssertEqual(flat.lrMomentumCycle.behaviorForm, firstParams.trainer.lrMomentumCycle.behaviorForm,
                       "and it feeds what the checkpoint's did")
    }

    // MARK: - adoptingSchedule

    /// Updated with LR keyframes: the envelope and its follow bounds are now
    /// the `lr_cycle_keyframes` key, so each trial draws a keyframe schedule.
    func testSnapshotAdoptingScheduleRoundTripsEveryField() throws {
        XCTAssertEqual(Set(TrainingParametersSnapshot.scheduleKeyIDs), Self.scheduleKeyIDs,
                       "the app's schedule key list is the one this test checks")
        let base = try TrainingParametersSnapshot.declaredDefaults(overriding: [WeightDecay.id: .double(0.00025)])
        var rng = DCMRandom(seed: 20261006)
        for trial in 0..<200 {
            var keyframes: [LRKeyframe] = []
            var step = 0
            for _ in 0..<Int.random(in: 1...6, using: &rng) {
                let trough = Double.random(in: 1e-8...1e-2, using: &rng)
                keyframes.append(LRKeyframe(
                    step: step,
                    peak: trough * Double.random(in: 1...1000, using: &rng),
                    trough: trough,
                    momentumAtPeak: Double.random(in: 0...0.99, using: &rng),
                    momentumAtTrough: Double.random(in: 0...0.99, using: &rng)))
                step += Int.random(in: 1...2_000_000, using: &rng)
            }
            let lrKeyframes = LRKeyframeSchedule(keyframes: keyframes)
            let cycle = LRMomentumCycle(
                lrEnabled: Bool.random(using: &rng),
                lrPeriodSteps: Int.random(in: 1...10_000_000, using: &rng),
                lrCount: Int.random(in: 0...1000, using: &rng),
                lrInvert: Bool.random(using: &rng),
                lrKeyframes: lrKeyframes,
                momentumEnabled: Bool.random(using: &rng),
                momentumPeriodSteps: Int.random(in: 1...10_000_000, using: &rng),
                momentumCount: Int.random(in: 0...1000, using: &rng),
                momentumMin: Double.random(in: 0...0.9, using: &rng),
                momentumMax: Double.random(in: 0.5...0.999, using: &rng),
                momentumInvert: Bool.random(using: &rng),
                momentumFollowsLRCycle: Bool.random(using: &rng))
            let schedule = TrainerScheduleState(
                completedTrainSteps: Int.random(in: 0...1_000_000, using: &rng),
                lrWarmupSteps: Int.random(in: 0...100_000, using: &rng),
                lrMomentumCycle: cycle)

            let adopted = base.adoptingSchedule(schedule)
            XCTAssertEqual(adopted.lrWarmupSteps, schedule.lrWarmupSteps, "trial \(trial): warmup")
            XCTAssertEqual(adopted.lrMomentumCycle, schedule.lrMomentumCycle, "trial \(trial): cycle and keyframes")
            XCTAssertEqual(adopted.lrCycleKeyframes, lrKeyframes, "trial \(trial): keyframes")
            XCTAssertEqual(adopted.rawValueMap()[LRCycleKeyframes.id], .lrKeyframes(lrKeyframes), "trial \(trial): keyframes key")
            let baseValues = base.rawValueMap()
            let adoptedValues = adopted.rawValueMap()
            XCTAssertEqual(Set(adoptedValues.keys), Set(baseValues.keys), "trial \(trial): no key added or dropped")
            for (id, value) in baseValues where !Self.scheduleKeyIDs.contains(id) {
                XCTAssertEqual(adoptedValues[id], value, "trial \(trial): \(id) is not a schedule key and is unchanged")
            }
        }
    }

    /// Unchanged by LR keyframes: `adoptingSchedule` still composes a lineage
    /// snapshot from the schedule a trainer ran, which is never clamped.
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

    /// Changed with LR keyframes (owner decisions O-4, Q-5): this pinned that
    /// an adopted checkpoint schedule is not a difference. Nothing adopts it
    /// any more: the configured schedule is what runs and is recorded, so a
    /// configured change is a `[RESUME-DIFF]` like any other and, when it
    /// changes what is fed, a `scheduleDifferences` line (the `schedule` gap);
    /// a change only in inert keys is a `[RESUME-DIFF]` but no gap.
    func testAnAdoptedScheduleIsNotADifference() throws {
        let parentParams = try params(warmup: 5, cyclePeriod: 40)
        let parentSnapshot = parentParams.parameters
        // What a schema-3 writer records: every parameter but the seed
        // settings (plan O-9).
        let parent = try LineageRecord.Parameters(values: parentSnapshot.lineageValues())
        let schedule = TrainerScheduleState(completedTrainSteps: 3, lrWarmupSteps: parentSnapshot.lrWarmupSteps,
                                            lrMomentumCycle: parentSnapshot.lrMomentumCycle)

        let configured = try params(warmup: 7, cyclePeriod: 80)
        XCTAssertEqual(try configured.parameters.differences(fromLineage: parent).map(\.id).sorted(),
                       [LRCyclePeriodSteps.id, LRWarmupSteps.id].sorted(), "the configured schedule differs, and is reported")
        XCTAssertEqual(configured.trainer.scheduleDifferences(from: schedule), [
            "lr_warmup_steps checkpoint=5 configured=7",
            "lr_momentum_cycle checkpoint=[\(LRMomentumCycleLogFormat.cycleDescription(parentSnapshot.lrMomentumCycle))] "
                + "configured=[\(LRMomentumCycleLogFormat.cycleDescription(configured.parameters.lrMomentumCycle))]",
        ], "a difference in what is fed is the schedule gap")
        XCTAssertEqual(configured.trainer.lrWarmupSteps, 7, "the trainer is configured with the configured warmup")
        XCTAssertEqual(configured.trainer.lrMomentumCycle, configured.parameters.lrMomentumCycle,
                       "the trainer is configured with the configured cycle")
        XCTAssertEqual(configured.lineageParameters, try LineageRecord.Parameters(values: configured.parameters.lineageValues()),
                       "the run records its configured snapshot")
        XCTAssertNotEqual(configured.lineageParameters, parent)

        let inertPeriod = parentSnapshot.momentumCyclePeriodSteps + 17
        let inert = try params(warmup: 5, cyclePeriod: 40, extra: [MomentumCyclePeriodSteps.id: .int(inertPeriod)])
        XCTAssertEqual(try inert.parameters.differences(fromLineage: parent).map(\.id), [MomentumCyclePeriodSteps.id],
                       "an inert change is still reported as a parameter difference")
        XCTAssertEqual(inert.trainer.scheduleDifferences(from: schedule), [],
                       "but it does not change what is fed, so it is no schedule gap")
        XCTAssertEqual(inert.trainer.lrMomentumCycle.momentumPeriodSteps, inertPeriod, "the configured value is kept")
        XCTAssertEqual(parentParams.trainer.scheduleDifferences(from: schedule), [], "an identical schedule is no gap")
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
