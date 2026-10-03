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

    func testAChangedBuildOrOSIsAGap() throws {
        let record = try LineageRecord.forTests(trainerCompletedSteps: 1, corpus: nil)
        XCTAssertEqual(ResumeGap.environmentGaps(writtenBy: record, runningBuild: record.build, runningDevice: record.device), [])
        let otherBuild = LineageRecord.Build(buildNumber: record.build.buildNumber + 1, gitHash: record.build.gitHash,
                                             gitBranch: record.build.gitBranch, gitDirty: record.build.gitDirty)
        XCTAssertEqual(ResumeGap.environmentGaps(writtenBy: record, runningBuild: otherBuild, runningDevice: record.device), [.build])
        let otherOS = LineageRecord.Device(hardwareModel: record.device.hardwareModel, cpu: record.device.cpu,
                                           isVirtualMachine: record.device.isVirtualMachine,
                                           osVersion: record.device.osVersion + " (later)", gpu: record.device.gpu)
        XCTAssertEqual(ResumeGap.environmentGaps(writtenBy: record, runningBuild: record.build, runningDevice: otherOS), [.os])
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
            nextGameSerial: serial, arenasStarted: arenas)
    }

    func testRunStreamsRoundTripIncludingASeedAboveTwoToThe53() throws {
        let original = streams(seed: 9_007_199_254_740_993, serial: 412, arenas: 7)
        let data = try JSONEncoder().encode(original)
        let text = String(decoding: data, as: UTF8.self)
        XCTAssertTrue(text.contains("\"master_seed\":\"9007199254740993\""), text)
        XCTAssertEqual(try JSONDecoder().decode(LineageRecord.RunStreams.self, from: data), original)
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
            nextGameSerial: nil, arenasStarted: nil)
        XCTAssertThrowsError(try RunRandomSeed.inherited(from: saved, configuredSeed: 0, commandLineSeed: nil))
    }

    /// The streams a save records are the run's own: the seed's origin as
    /// the run's first segment got it, and the positions passed in.
    func testASavesRunStreamsCarryTheSeedAndThePositionsGiven() {
        let seed = RunRandomSeed.resolve(mode: .seeded, configuredSeed: 42, commandLineSeed: nil, drawSeed: { 0 })
        let sampler = DCMRandom(seed: 3)
        let dropout = DCMRandom(seed: 4)
        let recorded = seed.runStreams(samplerState: sampler, dropoutStreamState: dropout,
                                       nextGameSerial: 9, arenasStarted: 2)
        XCTAssertEqual(recorded.masterSeed, 42)
        XCTAssertEqual(recorded.seedOrigin, .configured)
        XCTAssertEqual(recorded.samplerState, sampler)
        XCTAssertEqual(recorded.dropoutStreamState, dropout)
        XCTAssertEqual(recorded.nextGameSerial, 9)
        XCTAssertEqual(recorded.arenasStarted, 2)
    }

    // MARK: - Lineage of a resume

    func testAResumesRecordNamesItsGapsAndAnExactOneNamesNone() throws {
        let parentRecord = try LineageRecord.forTests(trainerCompletedSteps: 10, corpus: nil)
        let parent = LineageTracker.ParentFile(modelID: "20261003-1-PRNT", contentSHA256: "ab", trainerCompletedSteps: 10,
                                               lineage: .recorded(parentRecord))
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
}
