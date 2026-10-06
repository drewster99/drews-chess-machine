//
//  ExactResumeCompletionTests.swift
//  DrewsChessMachineTests
//
//  Determinism plan P9: what an exact resume continues beyond the trainer
//  state — the corpus feed phase, the run's random streams, and the one
//  exactness decision every path reports (`ResumeExactness`).
//

import XCTest
@testable import DrewsChessMachine

final class ExactResumeCompletionTests: XCTestCase {

    // MARK: - Corpus feed phase

    /// Game lengths of a synthetic corpus: varied, so whole games overshoot
    /// their targets by different amounts.
    private let gameLengths: [Int] = (0..<4_000).map { 20 + ($0 * 37) % 113 }

    /// Feeds whole games until `target` positions; returns the advanced
    /// game cursor and fed count.
    private func feed(until target: Int, from cursor: Int, fed: Int) -> (cursor: Int, fed: Int) {
        var cursor = cursor
        var fed = fed
        while fed < target {
            fed += gameLengths[cursor]
            cursor += 1
        }
        return (cursor, fed)
    }

    /// Games fed before each of `steps` trainer steps, the way corpus replay
    /// feeds (prefill, then step j's target before step j trains).
    private func uninterrupted(steps: Int, prefill: Int, perStep: Int) -> [Int] {
        var (cursor, fed) = feed(until: prefill, from: 0, fed: 0)
        let phase = CorpusFeedPhase.starting(fedPositions: fed, perStep: perStep)
        var consumed: [Int] = []
        for step in 0..<steps {
            (cursor, fed) = feed(until: phase.target(step: step), from: cursor, fed: fed)
            consumed.append(cursor)
        }
        return consumed
    }

    /// N steps, a save, a resume that refeeds a window of the games before
    /// the save point and continues the saved phase, then M more steps: the
    /// games fed before every resumed step equal the uninterrupted run's.
    func testAResumedFeedConsumesTheUninterruptedRunsGamesBeforeEveryStep() {
        let prefill = 3_000, perStep = 250, n = 40, m = 60
        let expected = uninterrupted(steps: n + m, prefill: prefill, perStep: perStep)

        // Segment 1: N steps, then a save before step N trains.
        var (cursor, fed) = feed(until: prefill, from: 0, fed: 0)
        let phase1 = CorpusFeedPhase.starting(fedPositions: fed, perStep: perStep)
        for step in 0..<n {
            (cursor, fed) = feed(until: phase1.target(step: step), from: cursor, fed: fed)
        }
        let savedNextGame = cursor
        let savedFeedAhead = phase1.feedAhead(fedPositions: fed, step: n)

        // Segment 2: refeed a window ending at the save point (its start, and
        // so the refed count, differs from the original's history), then
        // continue the saved phase.
        let windowStart = savedNextGame - 25
        var refed = 0
        for g in windowStart..<savedNextGame { refed += gameLengths[g] }
        let phase2 = CorpusFeedPhase.continuing(reconstructedFedPositions: refed,
                                                savedFeedAheadPositions: savedFeedAhead, perStep: perStep)
        var resumedCursor = savedNextGame
        var resumedFed = refed
        var resumed: [Int] = []
        for step in 0..<m {
            (resumedCursor, resumedFed) = feed(until: phase2.target(step: step), from: resumedCursor, fed: resumedFed)
            resumed.append(resumedCursor)
        }
        XCTAssertEqual(resumed, Array(expected[n...]))
    }

    /// The pre-P9 resume restarted the cadence at the refeed: it owed no feed
    /// before its first step, so it trained that step on fewer games than the
    /// uninterrupted run — the gap the saved phase closes.
    func testRestartingTheCadenceAtTheRefeedFallsBehindTheUninterruptedRun() {
        let prefill = 3_000, perStep = 250, n = 40
        let expected = uninterrupted(steps: n + 1, prefill: prefill, perStep: perStep)
        var (cursor, fed) = feed(until: prefill, from: 0, fed: 0)
        let phase = CorpusFeedPhase.starting(fedPositions: fed, perStep: perStep)
        for step in 0..<n {
            (cursor, fed) = feed(until: phase.target(step: step), from: cursor, fed: fed)
        }
        let restarted = CorpusFeedPhase.starting(fedPositions: fed, perStep: perStep)
        let (afterFirst, _) = feed(until: restarted.target(step: 0), from: cursor, fed: fed)
        XCTAssertLessThan(afterFirst, expected[n])
    }

    // MARK: - ResumeExactness

    func testGapsAreDeduplicatedInDeclarationOrderAndLoggedOnce() {
        let exactness = ResumeExactness(gaps: [.os, .rngSampler, .buffer, .rngSampler])
        XCTAssertEqual(exactness.gaps, [.rngSampler, .buffer, .os])
        XCTAssertEqual(exactness.logLine, "[RESUME] NOT EXACT: rng_sampler, buffer, os")
        XCTAssertFalse(exactness.isExact)
        XCTAssertEqual(ResumeExactness(gaps: []).logLine, "[RESUME] EXACT")
    }

    func testAnExactResumeIsRefusedUntilEveryGapIsAccepted() {
        let exactness = ResumeExactness(gaps: [.buffer, .build])
        XCTAssertNotNil(exactness.refusal(accepting: []))
        let partial = exactness.refusal(accepting: [.buffer])
        XCTAssertNotNil(partial)
        XCTAssertTrue(partial?.contains("--accept-inexact build") ?? false, partial ?? "")
        XCTAssertNil(exactness.refusal(accepting: [.buffer, .build]))
        XCTAssertNil(ResumeExactness(gaps: []).refusal(accepting: []))
    }

    func testAcceptListParsesTokensAndRejectsUnknownOnes() throws {
        XCTAssertEqual(try ResumeGap.parseAcceptList("buffer,build, os"), [.buffer, .build, .os])
        XCTAssertThrowsError(try ResumeGap.parseAcceptList("buffer,nonsense")) { error in
            XCTAssertEqual(error as? ResumeExactnessError, .unknownAcceptToken("nonsense"))
        }
    }

    /// A changed build or OS is a gap when the checkpoint has no behavior
    /// fingerprint to compare (this fixture records none); see
    /// `BehaviorFingerprintTests` for the fingerprinted cases.
    func testAChangedBuildOrOSIsAGap() throws {
        let record = try LineageRecord.forTests(trainerCompletedSteps: 1, corpus: nil)
        let running = BehaviorFingerprint.Record(recipe: BehaviorFingerprint.recipe, sha256: "00")
        XCTAssertEqual(ResumeGap.environmentGaps(writtenBy: record, runningBuild: record.build, runningDevice: record.device,
                                                 runningFingerprint: running).gaps, [])
        let otherBuild = LineageRecord.Build(buildNumber: record.build.buildNumber + 1, gitHash: record.build.gitHash,
                                             gitBranch: record.build.gitBranch, gitDirty: record.build.gitDirty)
        XCTAssertEqual(ResumeGap.environmentGaps(writtenBy: record, runningBuild: otherBuild, runningDevice: record.device,
                                                 runningFingerprint: running).gaps, [.build])
        let otherOS = LineageRecord.Device(hardwareModel: record.device.hardwareModel, cpu: record.device.cpu,
                                           isVirtualMachine: record.device.isVirtualMachine,
                                           osVersion: record.device.osVersion + " (later)", gpu: record.device.gpu)
        XCTAssertEqual(ResumeGap.environmentGaps(writtenBy: record, runningBuild: record.build, runningDevice: otherOS,
                                                 runningFingerprint: running).gaps, [.os])
    }

    func testDropoutGapFollowsWhatTheResumeRestores() {
        XCTAssertEqual(ResumeGap.dropoutGaps(restoring: .notInCheckpoint), [.dropoutState])
    }

    // MARK: - Run streams

    private func streams(seed: UInt64, serial: Int?, arenas: Int?) -> LineageRecord.RunStreams {
        var sampler = DCMRandom(seed: 11)
        _ = sampler.next()
        return LineageRecord.RunStreams(
            masterSeed: seed, seedOrigin: .drawn, streamDerivation: DCMRandomStreams.derivationVersion,
            samplerState: sampler, dropoutStreamState: DCMRandom(seed: 12),
            nextGameSerial: serial, arenasStarted: arenas, opponentGameIndices: [5, 0, 2])
    }

    func testRunStreamsRoundTripIncludingASeedAboveTwoToThe53() throws {
        let original = streams(seed: 9_007_199_254_740_993, serial: 412, arenas: 7)
        let data = try JSONEncoder().encode(original)
        let text = String(decoding: data, as: UTF8.self)
        XCTAssertTrue(text.contains("\"master_seed\":\"9007199254740993\""), text)
        XCTAssertEqual(try JSONDecoder().decode(LineageRecord.RunStreams.self, from: data), original)
    }

    /// `streams(...)` encoded, with `key` replaced by `value` (a JSON value).
    private func streamsJSON(replacing key: String, with value: Any) throws -> Data {
        let data = try JSONEncoder().encode(streams(seed: 77, serial: 412, arenas: 7))
        guard var object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw CocoaError(.coderReadCorrupt)
        }
        object[key] = value
        return try JSONSerialization.data(withJSONObject: object)
    }

    /// A counter a resume continues from must be one a run could have
    /// reached: a negative serial traps the serial counter, and one at
    /// `Int.max` overflows the next increment. Each is refused when the
    /// record is decoded, like any other malformed lineage field.
    func testRunStreamsRefuseOutOfRangeCounters() throws {
        let cap = LineageRecord.RunStreams.maximumRecordedCounter
        let refused: [(String, Any)] = [
            ("next_game_serial", -1), ("next_game_serial", cap + 1), ("next_game_serial", Int.max),
            ("arenas_started", -1), ("arenas_started", cap + 1),
            ("opponent_game_indices", [0, -1]), ("opponent_game_indices", [cap + 1, 0]),
        ]
        for (key, value) in refused {
            XCTAssertThrowsError(
                try JSONDecoder().decode(LineageRecord.RunStreams.self, from: try streamsJSON(replacing: key, with: value)),
                "\(key) = \(value)")
        }
        let atCap = try JSONDecoder().decode(
            LineageRecord.RunStreams.self, from: try streamsJSON(replacing: "next_game_serial", with: cap))
        XCTAssertEqual(atCap.nextGameSerial, cap, "a counter at the cap is a counter a run can reach")
    }

    /// The master seed is the decimal text the writer wrote: digits only. A
    /// sign is not part of that text, so "+5" or "-0" is a corrupt record,
    /// not seed 5 or 0.
    func testRunStreamsRefuseASignedMasterSeed() throws {
        for text in ["+5", "-0", "+0", " 5"] {
            XCTAssertThrowsError(
                try JSONDecoder().decode(LineageRecord.RunStreams.self,
                                         from: try streamsJSON(replacing: "master_seed", with: text)),
                "master_seed \"\(text)\"")
        }
    }

    /// A fresh run records the init seed and scheme its weights were drawn
    /// with (`rng.init_seed` / `init_scheme`), every record of the run
    /// carries them, an exact resume keeps them, and a branch — weights from
    /// another file — records none.
    func testTheRunsInitSeedIsRecordedCarriedAndOnlyOnAFreshRun() throws {
        let initialization = ModelInitRecord(initSeed: 18_446_744_073_709_551_557, scheme: WeightInitScheme.current)
        let start = Date(timeIntervalSince1970: 1_790_000_000)
        let fresh = try LineageTracker(start: .fresh(initialization: initialization), pathKind: .replay, argv: ["dcm"],
                                       startedAt: start, segmentStartTrainerStep: 0)
        let freshRecord = try fresh.record(at: start, trainerCompletedSteps: 3, segmentLocalStep: 3, segmentGames: 0,
                                           segmentPositions: 0, corpus: nil, parameters: nil,
                                           rng: .withoutRunStreams(dropoutPhiloxState: nil))
        XCTAssertEqual(freshRecord.rng.initialization, initialization)
        let text = try freshRecord.jsonText()
        XCTAssertTrue(text.contains("\"init_seed\":\"18446744073709551557\""), text)
        XCTAssertTrue(text.contains("\"init_scheme\":\"\(WeightInitScheme.current)\""), text)
        XCTAssertEqual(try LineageRecord.decode(jsonText: text), freshRecord)

        let parent = LineageTracker.ParentFile(modelID: "20261003-3-PRNT", contentSHA256: "ef", trainerCompletedSteps: 3,
                                               lineage: .recorded(freshRecord), derivationHistory: [])
        let resumed = try LineageTracker(start: .resume(parent: parent, gaps: [], legacyTotals: nil), pathKind: .replay,
                                         argv: ["dcm"], startedAt: start, segmentStartTrainerStep: 3)
        XCTAssertEqual(try resumed.startRecord(at: start, trainerCompletedSteps: 3, parameters: nil).rng.initialization,
                       initialization)
        let branch = try LineageTracker(start: .branch(parent: parent), pathKind: .replay, argv: ["dcm"],
                                        startedAt: start, segmentStartTrainerStep: 0)
        XCTAssertNil(try branch.startRecord(at: start, trainerCompletedSteps: 0, parameters: nil).rng.initialization)
        let minted = try LineageTracker.mintRecord(pathKind: .newModel, argv: ["dcm"], initialization: initialization, at: start)
        XCTAssertEqual(minted.rng.initialization, initialization)

        // A seed without its scheme is refused.
        let unpaired = text.replacingOccurrences(of: "\"init_scheme\":\"\(WeightInitScheme.current)\"", with: "\"init_scheme\":null")
        XCTAssertNotEqual(unpaired, text)
        XCTAssertThrowsError(try LineageRecord.decode(jsonText: unpaired))
    }

    func testARecordAtAnEarlierSchemaIsRefused() throws {
        let record = try LineageRecord.forTests(trainerCompletedSteps: 1, corpus: nil)
        let text = try record.jsonText()
        let earlier = text.replacingOccurrences(of: "\"schema\":\(LineageRecord.currentSchema)", with: "\"schema\":1")
        XCTAssertNotEqual(earlier, text)
        XCTAssertThrowsError(try LineageRecord.decode(jsonText: earlier))
    }

    func testAnExactResumeKeepsTheRunsSeedAndRefusesAConflictingSeedFlag() throws {
        let saved = streams(seed: 77, serial: 3, arenas: 1)
        let inherited = try RunRandomSeed.inherited(from: saved, configuredSeed: 5, commandLineSeed: nil)
        XCTAssertEqual(inherited.masterSeed, 77)
        XCTAssertEqual(inherited.recordedOrigin, .drawn)
        XCTAssertEqual(inherited.effectiveMode, .unseeded)
        XCTAssertTrue(inherited.logLine.contains("seed=77 mode=resumed(drawn)"), inherited.logLine)
        XCTAssertEqual(try RunRandomSeed.inherited(from: saved, configuredSeed: 5, commandLineSeed: 77).masterSeed, 77)
        XCTAssertThrowsError(try RunRandomSeed.inherited(from: saved, configuredSeed: 5, commandLineSeed: 78)) { error in
            XCTAssertEqual(error as? RunRandomSeedError, .resumeSeedConflict(commandLine: 78, checkpoint: 77))
        }
    }

    func testStreamsNamedUnderAnotherDerivationCannotBeContinued() {
        let saved = LineageRecord.RunStreams(
            masterSeed: 1, seedOrigin: .configured, streamDerivation: DCMRandomStreams.derivationVersion + "-other",
            samplerState: DCMRandom(seed: 1), dropoutStreamState: DCMRandom(seed: 2),
            nextGameSerial: nil, arenasStarted: nil, opponentGameIndices: nil)
        XCTAssertThrowsError(try RunRandomSeed.inherited(from: saved, configuredSeed: 0, commandLineSeed: nil))
    }

    /// The streams a save records are the run's own: the seed's origin as
    /// the run's first segment got it, and the positions passed in.
    func testASavesRunStreamsCarryTheSeedAndThePositionsGiven() {
        let seed = RunRandomSeed.resolve(mode: .seeded, configuredSeed: 42, commandLineSeed: nil, drawSeed: { 0 })
        let sampler = DCMRandom(seed: 3)
        let dropout = DCMRandom(seed: 4)
        let recorded = seed.runStreams(samplerState: sampler, dropoutStreamState: dropout,
                                       nextGameSerial: 9, arenasStarted: 2, opponentGameIndices: [3, 4])
        XCTAssertEqual(recorded.masterSeed, 42)
        XCTAssertEqual(recorded.seedOrigin, .configured)
        XCTAssertEqual(recorded.samplerState, sampler)
        XCTAssertEqual(recorded.dropoutStreamState, dropout)
        XCTAssertEqual(recorded.nextGameSerial, 9)
        XCTAssertEqual(recorded.arenasStarted, 2)
        XCTAssertEqual(recorded.opponentGameIndices, [3, 4])
    }

    /// The GUI status bar notes a not-exact resume with the same gap list the
    /// log carries, and says nothing for an exact one.
    func testTheStatusBarNoteListsTheGapsOfANotExactResume() {
        XCTAssertNil(ResumeExactness(gaps: []).statusBarNote)
        XCTAssertEqual(ResumeExactness(gaps: [.clocks, .buffer]).statusBarNote, "resumed not exact: buffer, clocks")
    }

    /// Train-vs-UCI: a resume continues each opponent instance's game index
    /// (and with it the trainer's colour) only into the same pool size;
    /// otherwise there is nothing to continue and every instance starts at 0.
    func testOpponentGameIndicesContinueOnlyIntoTheSamePool() throws {
        XCTAssertEqual(TrainVsUciDriver.continuedGameIndices(saved: [7, 3], instanceCount: 2), [7, 3])
        XCTAssertNil(TrainVsUciDriver.continuedGameIndices(saved: [7, 3], instanceCount: 3))
        XCTAssertNil(TrainVsUciDriver.continuedGameIndices(saved: nil, instanceCount: 2))
        let original = streams(seed: 5, serial: 40, arenas: nil)
        let decoded = try JSONDecoder().decode(LineageRecord.RunStreams.self, from: JSONEncoder().encode(original))
        XCTAssertEqual(decoded.opponentGameIndices, [5, 0, 2])
    }

    // MARK: - GUI arena clock

    private func sessionState() -> SessionCheckpointState {
        SessionCheckpointState(
            formatVersion: SessionCheckpointState.currentFormatVersion,
            sessionID: "test-session", savedAtUnix: 1_700_000_000, sessionStartUnix: 1_699_999_000,
            elapsedTrainingSec: 1000, trainingSteps: 1234, selfPlayGames: 10, selfPlayMoves: 600,
            trainingPositionsSeen: 1234 * 4096, batchSize: 4096, learningRate: 5e-5,
            promoteThreshold: 0.55, arenaGames: 200,
            selfPlayTau: TauConfigCodable(SamplingSchedule.selfPlay),
            arenaTau: TauConfigCodable(SamplingSchedule.arena),
            selfPlayWorkerCount: 4, championID: "champ-id", trainerID: "train-id", arenaHistory: []
        ).withLineage(LineageRecord.sessionTestFixture)
    }

    func testTheArenaClockRoundTripsThroughSessionJSON() throws {
        let original = sessionState().withArenaClock(secondsSinceLastArena: 512.25)
        let decoded = try SessionCheckpointState.decode(try original.encode())
        XCTAssertEqual(decoded.arenaSecondsSinceLastArena, 512.25)
        XCTAssertEqual(decoded, original)
        let without = try SessionCheckpointState.decode(try sessionState().encode())
        XCTAssertNil(without.arenaSecondsSinceLastArena)
    }

    /// A box restored from a saved clock reports that clock (plus the time
    /// since the restore), so the next automatic arena comes due on the
    /// saved run's schedule.
    func testARestoredArenaTriggerBoxContinuesTheSavedClock() {
        let now = Date()
        let box = ArenaTriggerBox(startTime: now.addingTimeInterval(-300))
        XCTAssertEqual(box.secondsSinceLastArena(now: now), 300, accuracy: 1e-6)
        XCTAssertTrue(box.shouldAutoTrigger(interval: 299))
        XCTAssertFalse(box.shouldAutoTrigger(interval: 3_600))
    }

    /// The post-promotion save is written inside the arena, before the
    /// trigger box hears that the arena ended; it must record the arena as
    /// just finished, not the time since the arena before it — else a resume
    /// of every `-promote` save would run an arena at once.
    @MainActor
    func testAPostPromotionSaveRecordsTheArenaAsJustFinished() throws {
        let controller = SessionController()
        controller.arenaTriggerBox = ArenaTriggerBox(startTime: Date().addingTimeInterval(-900))
        controller.beginRunStartCapture(buffer: ReplayBuffer(capacity: 64, inputEncoding: .basic30, sampler: DCMRandom(seed: 3)))
        let live = try controller.buildCurrentSessionState(championID: "c", trainerID: "t", arenaClock: .live, includeReplayBuffer: false)
        let postPromotion = try controller.buildCurrentSessionState(championID: "c", trainerID: "t", arenaClock: .arenaJustFinished, includeReplayBuffer: false)
        XCTAssertEqual(live.arenaSecondsSinceLastArena ?? -1, 900, accuracy: 5)
        XCTAssertEqual(postPromotion.arenaSecondsSinceLastArena, 0)
    }

    // MARK: - Lineage of a resume

    func testAResumesRecordNamesItsGapsAndAnExactOneNamesNone() throws {
        let parentRecord = try LineageRecord.forTests(trainerCompletedSteps: 10, corpus: nil)
        let parent = LineageTracker.ParentFile(modelID: "20261003-1-PRNT", contentSHA256: "ab", trainerCompletedSteps: 10,
                                               lineage: .recorded(parentRecord), derivationHistory: [])
        let start = Date(timeIntervalSince1970: 1_790_000_000)
        let inexact = try LineageTracker(start: .resume(parent: parent, gaps: [.buffer, .os], legacyTotals: nil),
                                         pathKind: .vsuci, argv: ["dcm"], startedAt: start, segmentStartTrainerStep: 10)
        let inexactRecord = try inexact.record(at: start, trainerCompletedSteps: 10, segmentLocalStep: 0, segmentGames: 0,
                                               segmentPositions: 0, corpus: nil, parameters: nil,
                                               rng: .withoutRunStreams(dropoutPhiloxState: nil))
        XCTAssertFalse(inexactRecord.run.exactResume)
        XCTAssertEqual(inexactRecord.run.notExactItems, ["buffer", "os"])

        let exact = try LineageTracker(start: .resume(parent: parent, gaps: [], legacyTotals: nil),
                                       pathKind: .replay, argv: ["dcm"], startedAt: start, segmentStartTrainerStep: 10)
        let exactRecord = try exact.record(at: start, trainerCompletedSteps: 10, segmentLocalStep: 0, segmentGames: 0,
                                           segmentPositions: 0, corpus: nil, parameters: nil,
                                           rng: .withoutRunStreams(dropoutPhiloxState: nil))
        XCTAssertTrue(exactRecord.run.exactResume)
        XCTAssertEqual(exactRecord.run.notExactItems, [])
    }

    /// The `[RUN]` line reports the same verdict as the `[RESUME]` line: both
    /// come from the one decision, listed the same way.
    func testTheRunLineAndTheResumeLineListTheSameGaps() throws {
        let parentRecord = try LineageRecord.forTests(trainerCompletedSteps: 10, corpus: nil)
        let parent = LineageTracker.ParentFile(modelID: "20261003-2-PRNT", contentSHA256: "cd", trainerCompletedSteps: 10,
                                               lineage: .recorded(parentRecord), derivationHistory: [])
        let gaps: [ResumeGap] = [.os, .buffer, .rngSampler]
        let exactness = ResumeExactness.resume(of: parent, gaps: gaps)
        let tracker = try LineageTracker(start: .resume(parent: parent, gaps: gaps, legacyTotals: nil),
                                         pathKind: .vsuci, argv: ["dcm"], startedAt: Date(timeIntervalSince1970: 1_790_000_000),
                                         segmentStartTrainerStep: 10)
        let record = try tracker.startRecord(at: Date(timeIntervalSince1970: 1_790_000_000), trainerCompletedSteps: 10,
                                             parameters: nil)
        let runLine = RunProvenanceLine.line(record: record, seed: nil)
        XCTAssertEqual(exactness.logLine, "[RESUME] NOT EXACT: rng_sampler, buffer, os")
        XCTAssertTrue(runLine.contains("not exact: rng_sampler, buffer, os"), runLine)

        let legacyParent = LineageTracker.ParentFile(modelID: "20260901-1-OLDP", contentSHA256: nil, trainerCompletedSteps: 10,
                                                     lineage: .unrecorded(formatVersion: 6), derivationHistory: [])
        XCTAssertEqual(ResumeExactness.resume(of: legacyParent, gaps: [.buffer]).tokens, ["buffer", "lineage"])
    }
}
