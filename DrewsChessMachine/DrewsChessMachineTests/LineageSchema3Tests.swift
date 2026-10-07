//
//  LineageSchema3Tests.swift
//  DrewsChessMachineTests
//
//  Lineage schema 3 (hyperparameter recording plan P4): what a record with
//  training behind it adds — the segment's configuration and journals, the
//  run's seeds, the ancestry and the build identity — how the tracker
//  refuses a record missing any of it, and how a schema-2 record (every
//  file written before P4) is still read, continued and converted.
//

import XCTest
@testable import DrewsChessMachine

final class LineageSchema3Tests: XCTestCase {

    /// The real schema-2 records (`LineageSchemaTwoFixtures`).
    static let schemaTwoReplayRecord = LineageSchemaTwoFixtures.fatconv98ContStep6093
    static let schemaTwoSecondSegment = LineageSchemaTwoFixtures.segResumeCheckSeg1Step1000

    private let start = Date(timeIntervalSince1970: 1_790_000_000)

    private func tracker(_ pathKind: LineageRecord.PathKind, at step: Int = 0) throws -> LineageTracker {
        try LineageTracker(start: .fresh(initialization: .forTests), pathKind: pathKind, argv: ["dcm"],
                           startedAt: start, segmentStartTrainerStep: step)
    }

    private func parameters() throws -> LineageRecord.Parameters {
        try LineageRecord.Parameters(values: ["learning_rate": .double(0.0005)])
    }

    private func record(_ tracker: LineageTracker, clock: Int, inputs: LineageTracker.SaveInputs? = nil) throws -> LineageRecord {
        try tracker.record(at: start.addingTimeInterval(60), trainerCompletedSteps: clock, segmentLocalStep: clock,
                           segmentGames: 0, segmentPositions: 0, corpus: nil, parameters: try parameters(),
                           rng: .withoutRunStreams(dropoutPhiloxState: nil), inputs: inputs ?? tracker.testInputs)
    }

    private func change(_ id: String, at step: Int, old: Double, new: Double) -> LineageRecord.ParameterChange {
        LineageRecord.ParameterChange(committedAtTrainerStep: step, recordedUnix: 1_790_000_100, id: id,
                                      old: .double(old), new: .double(new), restampedFrom: nil)
    }

    // MARK: - Schema 2 stays readable

    func testARealSchemaTwoRecordDecodesInTheSchemaThreeShape() throws {
        let record = try LineageRecord.decode(jsonText: Self.schemaTwoReplayRecord)
        XCTAssertEqual(record.schema, 2)
        XCTAssertNotNil(record.parameters)
        XCTAssertEqual(record.configuration, .unrecorded, "schema 2 never stored a configuration")
        guard case .recorded(let seeds) = record.runSeeds else {
            return XCTFail("the seed its streams name: \(record.runSeeds)")
        }
        XCTAssertEqual(seeds, [LineageRecord.RunSeedEntry(fromTrainerStep: nil, masterSeed: 18_019_510_007_828_584_227,
                                                          seedOrigin: .drawn, streamDerivation: "v1")])
        XCTAssertEqual(record.ancestry, .unrecordedHistory, "an inexact resume of unrecorded history")
        XCTAssertEqual(record.build.gitDiffSHA256, .unrecorded)
        XCTAssertEqual(record.build.xcodeBuild, .unrecorded)
        XCTAssertEqual(record.build.sdkBuild, .unrecorded)
        XCTAssertEqual(record.build.configuration, .unrecorded)
        let corpus = try XCTUnwrap(record.fed.corpus)
        XCTAssertEqual(corpus.corpusIdentity, .firstOnly(
            id: "20260624-192615-w3aA5b",
            path: "/Users/andrew/Library/Application Support/DrewsChessMachine/Corpora/20260624-192615-w3aA5b"))
        XCTAssertEqual(corpus.segmentStart, .unrecorded)
        XCTAssertEqual(corpus.shardSHA256.count, 46)
        XCTAssertEqual(record.steps.cumTrainerStep, 39_093)
    }

    /// A decoded schema-2 record is never written back as schema 2: it is
    /// converted first, and the conversion is a schema-3 record that
    /// round-trips.
    func testASchemaTwoRecordIsWrittenOnlyAfterConversion() throws {
        let record = try LineageRecord.decode(jsonText: Self.schemaTwoReplayRecord)
        XCTAssertThrowsError(try record.jsonText())
        let converted = try record.withoutTrainerState()
        XCTAssertEqual(converted.schema, LineageRecord.currentSchema)
        XCTAssertEqual(converted.configuration, .unrecorded)
        XCTAssertEqual(converted.runSeeds, record.runSeeds)
        XCTAssertNil(converted.rng.streams)
        let text = try converted.jsonText()
        XCTAssertTrue(text.contains("\"configuration\":{\"recorded\":false}"), text)
        XCTAssertTrue(text.contains("\"first_only\""), text)
        XCTAssertEqual(try LineageRecord.decode(jsonText: text), converted)
    }

    /// A schema-2 segment summary decodes with every field schema 3 added
    /// unrecorded, and a run whose first segment was a branch has history
    /// before it.
    func testARealSchemaTwoSegmentSummaryDecodes() throws {
        let record = try LineageRecord.decode(jsonText: Self.schemaTwoSecondSegment)
        XCTAssertEqual(record.schema, 2)
        XCTAssertEqual(record.run.segmentIndex, 1)
        XCTAssertEqual(record.segments.count, 1)
        let summary = record.segments[0]
        XCTAssertEqual(summary.start, .branch)
        XCTAssertEqual(summary.configuration, .unrecorded)
        XCTAssertEqual(summary.corpusIdentity, .unrecorded)
        XCTAssertEqual(summary.segmentStartCorpus, .unrecorded)
        XCTAssertEqual(summary.pathKind, .unrecorded)
        XCTAssertEqual(summary.argv, .unrecorded)
        XCTAssertEqual(summary.runSeeds, .unrecorded)
        XCTAssertEqual(record.ancestry, .unrecordedHistory, "the run began as a branch of a file this record does not describe")
        XCTAssertEqual(record.runSeeds.value?.map(\.masterSeed), [777])
        XCTAssertEqual(record.runSeeds.value?.map(\.seedOrigin), [.configured])
        let converted = try record.withoutTrainerState()
        XCTAssertEqual(try LineageRecord.decode(jsonText: try converted.jsonText()), converted)
    }

    func testASchemaTwoRecordCarryingASchemaThreeKeyIsRefused() throws {
        var object = try XCTUnwrap(try JSONSerialization.jsonObject(with: Data(Self.schemaTwoReplayRecord.utf8)) as? [String: Any])
        object["ancestry"] = ["history_before_oldest_run": "none", "runs": [Any]()]
        XCTAssertThrowsError(try LineageRecord.decode(jsonText: try Self.text(object)))

        object.removeValue(forKey: "ancestry")
        var fed = try XCTUnwrap(object["fed"] as? [String: Any])
        var corpus = try XCTUnwrap(fed["corpus"] as? [String: Any])
        corpus["segment_start"] = ["recorded": false]
        fed["corpus"] = corpus
        object["fed"] = fed
        XCTAssertThrowsError(try LineageRecord.decode(jsonText: try Self.text(object)))
    }

    func testASchemaThreeRecordWithoutItsConfigurationIsRefused() throws {
        let tracker = try tracker(.replay)
        try tracker.noteSegmentStartForTests(trainerStep: 0)
        var object = try XCTUnwrap(try JSONSerialization.jsonObject(
            with: Data(try record(tracker, clock: 3).jsonText().utf8)) as? [String: Any])
        object.removeValue(forKey: "configuration")
        XCTAssertThrowsError(try LineageRecord.decode(jsonText: try Self.text(object)))
    }

    /// Continuing a schema-2 file: the new segment's record is schema 3,
    /// keeps the file's run and ancestry, and summarizes the schema-2
    /// segment with what that schema never stored left unrecorded.
    func testAnExactResumeOfASchemaTwoFileWritesSchemaThree() throws {
        let parentRecord = try LineageRecord.decode(jsonText: Self.schemaTwoReplayRecord)
        let parent = LineageTracker.ParentFile(modelID: "20261005-1-1vVl", contentSHA256: String(repeating: "a", count: 64),
                                               trainerCompletedSteps: 39_093, lineage: .recorded(parentRecord),
                                               derivationHistory: parentRecord.derivationHistory)
        let resumed = try LineageTracker(start: .resume(parent: parent, gaps: [], legacyTotals: nil), pathKind: .replay,
                                         argv: ["dcm"], startedAt: start, segmentStartTrainerStep: 39_093)
        try resumed.noteSegmentStartForTests(trainerStep: 39_093)
        let next = try record(resumed, clock: 39_103)
        XCTAssertEqual(next.schema, LineageRecord.currentSchema)
        XCTAssertEqual(next.run.lineageRunID, parentRecord.run.lineageRunID)
        XCTAssertEqual(next.run.segmentIndex, 1)
        XCTAssertEqual(next.ancestry, .unrecordedHistory)
        XCTAssertEqual(next.segments.count, 1)
        XCTAssertEqual(next.segments[0].configuration, .unrecorded)
        XCTAssertEqual(next.segments[0].corpusIdentity, .recorded(parentRecord.fed.corpus?.corpusIdentity))
        XCTAssertEqual(next.segments[0].segmentStartCorpus, .unrecorded)
        guard case .recorded(let configuration) = next.configuration else {
            return XCTFail("the new segment records its configuration")
        }
        XCTAssertEqual(configuration.pathKind, .replay)
        XCTAssertEqual(try LineageRecord.decode(jsonText: try next.jsonText()), next)
    }

    // MARK: - What the tracker requires

    func testARecordWithTrainingBehindItNeedsItsSegmentConfigured() throws {
        let unconfigured = try tracker(.replay)
        XCTAssertThrowsError(try record(unconfigured, clock: 1)) { error in
            guard case LineageTracker.TrackerError.segmentNotConfigured = error else { return XCTFail("\(error)") }
        }

        let noSeed = try tracker(.replay)
        try noSeed.configureSegment(LineageTracker.SegmentConfiguration(
            policyTailPrecision: .default, budget: .none, vsuci: nil, selfPlayDirichlet: nil,
            startValueHeadRecentered: .recorded(false)))
        XCTAssertThrowsError(try record(noSeed, clock: 1)) { error in
            guard case LineageTracker.TrackerError.noRunSeedNoted = error else { return XCTFail("\(error)") }
        }
        XCTAssertThrowsError(try noSeed.configureSegment(LineageTracker.SegmentConfiguration(
            policyTailPrecision: .default, budget: .none, vsuci: nil, selfPlayDirichlet: nil,
            startValueHeadRecentered: .recorded(false)))) { error in
            guard case LineageTracker.TrackerError.segmentConfiguredTwice = error else { return XCTFail("\(error)") }
        }

        // A record with nothing trained behind it needs none of it.
        let untrained = try unconfigured.record(at: start, trainerCompletedSteps: 0, segmentLocalStep: 0, segmentGames: 0,
                                                segmentPositions: 0, corpus: nil, parameters: nil,
                                                rng: .withoutRunStreams(dropoutPhiloxState: nil),
                                                inputs: unconfigured.testInputs)
        XCTAssertEqual(untrained.configuration, .notTrained)
        XCTAssertEqual(untrained.runSeeds, .notTrained)
    }

    func testAGuiRecordNeedsItsSegmentStartChampion() throws {
        let gui = try tracker(.gui)
        try gui.configureSegment(LineageTracker.SegmentConfiguration(
            policyTailPrecision: .default, budget: .none, vsuci: nil,
            selfPlayDirichlet: LineageRecord.Dirichlet(.alphaZero), startValueHeadRecentered: .recorded(false)))
        gui.noteRunSeed(RunRandomSeed.resolve(mode: .seeded, configuredSeed: 7, commandLineSeed: nil, drawSeed: { 0 }),
                        atTrainerStep: 0)
        XCTAssertThrowsError(try record(gui, clock: 1)) { error in
            guard case LineageTracker.TrackerError.noSegmentStartChampion = error else { return XCTFail("\(error)") }
        }
    }

    /// The seed settings are superseded by the run's actual seed
    /// (`run_seeds`), so a composed snapshot never holds them (O-9).
    func testAComposedSnapshotWithTheSeedSettingsIsRefused() throws {
        let tracker = try tracker(.replay)
        try tracker.noteSegmentStartForTests(trainerStep: 0)
        let everything = try LineageRecord.Parameters(
            values: try TrainingParametersSnapshot.declaredDefaults(overriding: [:]).rawValueMap())
        XCTAssertThrowsError(try tracker.record(at: start, trainerCompletedSteps: 1, segmentLocalStep: 1, segmentGames: 0,
                                                segmentPositions: 0, corpus: nil, parameters: everything,
                                                rng: .withoutRunStreams(dropoutPhiloxState: nil),
                                                inputs: tracker.testInputs)) { error in
            guard case LineageTracker.TrackerError.composedSnapshotHasSeedSettings(let ids) = error else {
                return XCTFail("\(error)")
            }
            XCTAssertEqual(ids, LineageRecord.Parameters.excludedParameterIDs.sorted())
        }
        let composed = try LineageRecord.Parameters(
            values: try TrainingParametersSnapshot.declaredDefaults(overriding: [:]).lineageValues())
        XCTAssertNoThrow(try tracker.record(at: start, trainerCompletedSteps: 1, segmentLocalStep: 1, segmentGames: 0,
                                            segmentPositions: 0, corpus: nil, parameters: composed,
                                            rng: .withoutRunStreams(dropoutPhiloxState: nil),
                                            inputs: tracker.testInputs))
    }

    // MARK: - Journals

    /// A save records the journal as its cut found it: a change committed
    /// after the cut is the next save's.
    func testTheSaveInputsCutBoundsTheParameterChangeJournal() throws {
        let gui = try tracker(.gui)
        try gui.noteSegmentStartForTests(trainerStep: 0)
        let first = change("entropy_bonus", at: 3, old: 0.01, new: 0.02)
        gui.journalParameterChange(first)
        let cut = gui.testInputs
        gui.journalParameterChange(change("entropy_bonus", at: 4, old: 0.02, new: 0.03))
        let atCut = try record(gui, clock: 5, inputs: cut)
        XCTAssertEqual(atCut.configuration.value?.parameterChanges, [first])
        XCTAssertEqual(try record(gui, clock: 5).configuration.value?.parameterChanges.count, 2)
    }

    /// An arena promotion rewinds the trainer to the arena's start step, so
    /// a change committed during the arena takes effect from that step.
    func testARestampMovesLaterChangesToTheRewoundStep() throws {
        let gui = try tracker(.gui)
        try gui.noteSegmentStartForTests(trainerStep: 0)
        gui.journalParameterChange(change("entropy_bonus", at: 4, old: 0.01, new: 0.02))
        gui.journalParameterChange(change("entropy_bonus", at: 9, old: 0.02, new: 0.03))
        gui.restampParameterChanges(after: 6)
        let changes = try XCTUnwrap(try record(gui, clock: 6).configuration.value?.parameterChanges)
        XCTAssertEqual(changes.map(\.committedAtTrainerStep), [4, 6])
        XCTAssertEqual(changes.map(\.restampedFrom), [nil, 9])
    }

    func testAChangeAfterTheRecordsClockIsRefused() throws {
        let gui = try tracker(.gui)
        try gui.noteSegmentStartForTests(trainerStep: 0)
        gui.journalParameterChange(change("entropy_bonus", at: 9, old: 0.01, new: 0.02))
        XCTAssertThrowsError(try record(gui, clock: 5))
    }

    /// Only a GUI segment changes its seed mid-segment ("New Session, keep
    /// trainer"); the CLI paths resolve one seed per segment.
    func testOnlyAGuiSegmentRecordsMoreThanOneSeed() throws {
        let second = RunRandomSeed.resolve(mode: .seeded, configuredSeed: 99, commandLineSeed: nil, drawSeed: { 0 })
        let gui = try tracker(.gui)
        try gui.noteSegmentStartForTests(trainerStep: 0)
        gui.noteRunSeed(second, atTrainerStep: 10)
        let seeds = try XCTUnwrap(try record(gui, clock: 12).runSeeds.value)
        XCTAssertEqual(seeds.map(\.fromTrainerStep), [0, 10])
        XCTAssertEqual(seeds.map(\.masterSeed), [7, 99])

        let replay = try tracker(.replay)
        try replay.noteSegmentStartForTests(trainerStep: 0)
        replay.noteRunSeed(second, atTrainerStep: 10)
        XCTAssertThrowsError(try record(replay, clock: 12))
    }

    /// A start that fails takes its journal entries back out.
    func testRestoredJournalsDropAFailedStartsEntries() throws {
        let gui = try tracker(.gui)
        try gui.noteSegmentStartForTests(trainerStep: 0)
        let before = gui.checkpointJournals()
        gui.journalParameterChange(change("entropy_bonus", at: 0, old: 0.01, new: 0.02))
        gui.noteRunSeed(RunRandomSeed.resolve(mode: .seeded, configuredSeed: 3, commandLineSeed: nil, drawSeed: { 0 }),
                        atTrainerStep: 0)
        gui.restoreJournals(before)
        let restored = try record(gui, clock: 1)
        XCTAssertEqual(restored.configuration.value?.parameterChanges, [])
        XCTAssertEqual(restored.runSeeds.value?.count, 1)
    }

    // MARK: - Ancestry

    /// A branch starts a new run and keeps the run it left, with that run's
    /// totals at departure and how its weights were drawn.
    func testABranchKeepsTheRunItLeft() throws {
        let first = try tracker(.replay)
        try first.noteSegmentStartForTests(trainerStep: 0)
        first.recordTrainingStep(totalMs: 1_000)
        let left = try record(first, clock: 40)
        let parent = LineageTracker.ParentFile(modelID: "20261006-1-LEFT", contentSHA256: String(repeating: "b", count: 64),
                                               trainerCompletedSteps: 40, lineage: .recorded(left), derivationHistory: [])
        let branch = try LineageTracker(start: .branch(parent: parent), pathKind: .replay, argv: ["dcm"],
                                        startedAt: start.addingTimeInterval(100), segmentStartTrainerStep: 0)
        try branch.noteSegmentStartForTests(trainerStep: 0)
        let next = try record(branch, clock: 2)
        XCTAssertNotEqual(next.run.lineageRunID, left.run.lineageRunID)
        XCTAssertEqual(next.ancestry.historyBeforeOldestRun, LineageRecord.Ancestry.HistoryBeforeOldestRun.none, "the run it left began fresh")
        XCTAssertEqual(next.ancestry.runs.count, 1)
        let ancestor = next.ancestry.runs[0]
        XCTAssertEqual(ancestor.lineageRunID, left.run.lineageRunID)
        XCTAssertEqual(ancestor.leftBy, .branch)
        XCTAssertEqual(ancestor.totalsAtDeparture.cumTrainerStep, 40)
        XCTAssertEqual(ancestor.initialization, .recorded(LineageRecord.AncestorRun.Initialization(.forTests)))
        XCTAssertNil(ancestor.architectureAtDeparture)
        XCTAssertEqual(try LineageRecord.decode(jsonText: try next.jsonText()), next)
    }

    // MARK: - Corpus identity

    func testACorpusListMustCoverEveryShardHash() throws {
        XCTAssertThrowsError(try LineageRecord.CorpusPosition(
            corpusIdentity: .listed([.init(corpusID: "a", corpusPath: "/a", shardCount: 2),
                                     .init(corpusID: "b", corpusPath: "/b", shardCount: 1)]),
            segmentStart: .recorded(LineageRecord.FeedPoint(epoch: 0, nextGameIndex: 0)), epoch: 0, nextGameIndex: 0,
            shard: 0, populatedPlies: 0, bufferCapacity: 1, feedAheadPositions: 0, feedPerStep: 1,
            shardSHA256: ["00", "11"]))
        let mixed = try LineageRecord.CorpusPosition(
            corpusIdentity: .listed([.init(corpusID: "a", corpusPath: "/a", shardCount: 2),
                                     .init(corpusID: "b", corpusPath: "/b", shardCount: 1)]),
            segmentStart: .recorded(LineageRecord.FeedPoint(epoch: 0, nextGameIndex: 0)), epoch: 0, nextGameIndex: 0,
            shard: 0, populatedPlies: 0, bufferCapacity: 1, feedAheadPositions: 0, feedPerStep: 1,
            shardSHA256: ["00", "11", "22"])
        XCTAssertEqual(mixed.corpusIdentity.firstCorpusID, "a")
        let data = try JSONEncoder().encode(mixed)
        XCTAssertEqual(try JSONDecoder().decode(LineageRecord.CorpusPosition.self, from: data), mixed)
    }

    // MARK: - Recorded values

    func testARecordedValueDecodesStrictly() throws {
        typealias Flag = LineageRecord.Recorded<Bool>
        XCTAssertEqual(try JSONDecoder().decode(Flag.self, from: Data(#"{"recorded":true,"value":true}"#.utf8)), .recorded(true))
        XCTAssertEqual(try JSONDecoder().decode(Flag.self, from: Data(#"{"recorded":false}"#.utf8)), .unrecorded)
        for bad in [#"{"recorded":true}"#, #"{"recorded":false,"value":true}"#, #"{"value":true}"#,
                    #"{"recorded":true,"value":true,"note":1}"#] {
            XCTAssertThrowsError(try JSONDecoder().decode(Flag.self, from: Data(bad.utf8)), bad)
        }
    }

    // MARK: - Replay-ratio initial delay

    func testTheReplayRatioInitialDelayNamesItsSource() {
        XCTAssertEqual(ReplayRatioInitialDelay.resolve(autoAdjust: true, savedAutoDelayMs: 120, trainingStepDelayMs: 40),
                       ReplayRatioInitialDelay.Resolved(autoAdjust: true, delayMs: 120, source: .lastAutoComputedDelayMs))
        XCTAssertEqual(ReplayRatioInitialDelay.resolve(autoAdjust: true, savedAutoDelayMs: nil, trainingStepDelayMs: 40),
                       ReplayRatioInitialDelay.Resolved(autoAdjust: true, delayMs: 40, source: .trainingStepDelayMs))
        XCTAssertEqual(ReplayRatioInitialDelay.resolve(autoAdjust: false, savedAutoDelayMs: 120, trainingStepDelayMs: 40),
                       ReplayRatioInitialDelay.Resolved(autoAdjust: false, delayMs: 40, source: .trainingStepDelayMs))
    }

    // MARK: - Helpers

    private static func text(_ object: [String: Any]) throws -> String {
        String(decoding: try JSONSerialization.data(withJSONObject: object), as: UTF8.self)
    }
}
