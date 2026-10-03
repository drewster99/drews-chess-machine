//
//  GuiSessionResumeTests.swift
//  DrewsChessMachineTests
//
//  What a GUI session resume does to the live settings
//  (`SessionParameterResume.applyGuiSession`):
//  - it leaves the run-seed settings alone. A resume either continues the
//    saved run's seed with its streams or draws one for the run; neither
//    uses the configured seed, so restoring (or holding) the seed settings
//    only produced a false "random_seed NOT EXACT" line and silently
//    switched a seeded user to unseeded for the rest of the launch;
//  - it restores the run-throughput knobs (workers, the two delays, the
//    replay-ratio target and auto-adjust) through the same resolver as
//    every other parameter, one `[RESUME-DIFF]` line each.
//
//  Every test snapshots `TrainingParameters.shared` and restores it
//  afterwards with persistence suppressed, so the user's saved settings are
//  never written.
//

import XCTest
@testable import DrewsChessMachine

@MainActor
final class GuiSessionResumeTests: XCTestCase {

    private var savedParameterValues: [String: ParameterValue] = [:]
    private var logLines: [String] = []

    override func setUp() async throws {
        try await super.setUp()
        savedParameterValues = TrainingParameters.shared.snapshot().rawValueMap()
        TrainingParameters.suppressPersistence = true
        logLines = []
    }

    override func tearDown() async throws {
        TrainingParameters.suppressPersistence = true
        try TrainingParameters.shared.apply(savedParameterValues)
        TrainingParameters.suppressPersistence = false
        try await super.tearDown()
    }

    private func makeResume() -> SessionParameterResume {
        SessionParameterResume(parameters: TrainingParameters.shared, log: { [weak self] line in
            self?.logLines.append(line)
        })
    }

    /// A session state at the current format with the given extra fields.
    private func sessionState(extraFields: String = "") throws -> SessionCheckpointState {
        let json = """
        {
          "formatVersion": \(SessionCheckpointState.currentFormatVersion),
          "lineage": \(LineageRecord.sessionTestFixtureJSON),
          "sessionID": "20261003-1-RSUM", "savedAtUnix": 1790000000,
          "sessionStartUnix": 1789996400, "elapsedTrainingSec": 3600,
          "trainingSteps": 500, "selfPlayGames": 40, "selfPlayMoves": 3000,
          "trainingPositionsSeen": 16000, "batchSize": 32, "learningRate": 0.001,
          "promoteThreshold": 0.53, "arenaGames": 400,
          "selfPlayTau": {"startTau": 1.0, "decayPerPly": 0.007, "floorTau": 0.5},
          "arenaTau": {"startTau": 0.6, "decayPerPly": 0.02, "floorTau": 0.02},
          "selfPlayWorkerCount": 6,
          \(extraFields)
          "championID": "20261003-1-RSUM", "trainerID": "20261003-2-RSUM", "arenaHistory": []
        }
        """
        return try SessionCheckpointState.decode(Data(json.utf8))
    }

    private func diffLines(for id: String) -> [String] {
        logLines.filter { $0.hasPrefix("[RESUME-DIFF] \(id):") }
    }

    func testGuiResumeLeavesTheSeedSettingsAlone() throws {
        let p = TrainingParameters.shared
        p.randomSeedMode = .seeded
        p.randomSeed = 12_345
        let resume = makeResume()
        resume.applyGuiSession(try sessionState(), acceptedReplacements: [])
        XCTAssertEqual(p.randomSeedMode, .seeded)
        XCTAssertEqual(p.randomSeed, 12_345)
        XCTAssertFalse(resume.notExactParameterIDs.contains(RandomSeed.id))
        XCTAssertFalse(resume.notExactParameterIDs.contains(RandomSeedModeParameter.id))
        XCTAssertTrue(diffLines(for: RandomSeed.id).isEmpty)
        XCTAssertTrue(diffLines(for: RandomSeedModeParameter.id).isEmpty)
    }

    func testGuiResumeRestoresRunThroughputKnobsThroughTheResolver() throws {
        let p = TrainingParameters.shared
        p.selfPlayConcurrency = 3
        p.trainingStepDelayMs = 0
        p.selfPlayDelayMs = 0
        p.replayRatioTarget = 0.5
        p.replayRatioAutoAdjust = false
        let rs = try sessionState(extraFields: """
          "stepDelayMs": 25, "selfPlayDelayMs": 15, "replayRatioTarget": 0.7, "replayRatioAutoAdjust": true,
        """)
        makeResume().applyGuiSession(rs, acceptedReplacements: [])

        XCTAssertEqual(p.selfPlayConcurrency, 6)
        XCTAssertEqual(p.trainingStepDelayMs, 25)
        XCTAssertEqual(p.selfPlayDelayMs, 15)
        XCTAssertEqual(p.replayRatioTarget, 0.7)
        XCTAssertEqual(p.replayRatioAutoAdjust, true)
        for id in [SelfPlayConcurrency.id, TrainingStepDelayMs.id, SelfPlayDelayMs.id,
                   ReplayRatioTarget.id, ReplayRatioAutoAdjust.id] {
            XCTAssertEqual(diffLines(for: id).count, 1, "one [RESUME-DIFF] line for \(id): \(logLines)")
        }
    }
}
