//
//  RelativeGradientCapGuiTests.swift
//  DrewsChessMachineTests
//
//  The relative gradient cap on the GUI side (plan X5 session cases, Part K
//  steps 4 and 7): a session saves the five settings and a resume restores
//  each with one `[RESUME-DIFF]` line; a session written before the cap
//  resumes with the mode off (held for the run) and the other four at the
//  current settings; the settings popover writes N and W only as a valid
//  pair, so the settings never hold W > N.
//
//  Tests that touch `TrainingParameters.shared` snapshot it and restore it
//  with persistence suppressed, so the user's saved settings are never
//  written.
//

import XCTest
@testable import DrewsChessMachine

@MainActor
final class RelativeGradientCapGuiTests: XCTestCase {

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
        _ = try TrainingParameters.shared.releaseRunHolds()
        try TrainingParameters.shared.apply(savedParameterValues)
        TrainingParameters.suppressPersistence = false
        try await super.tearDown()
    }

    private static let ids = [
        "relative_grad_clip_mode", "relative_grad_clip_multiple", "relative_grad_clip_window_steps",
        "relative_grad_clip_min_history_steps", "relative_grad_clip_floor",
    ]

    private func sessionState(extraFields: String) throws -> SessionCheckpointState {
        let json = """
        {
          "formatVersion": \(SessionCheckpointState.currentFormatVersion),
          "lineage": \(LineageRecord.sessionTestFixtureJSON),
          "sessionID": "20261007-1-RCAP", "savedAtUnix": 1790000000,
          "sessionStartUnix": 1789996400, "elapsedTrainingSec": 3600,
          "trainingSteps": 500, "selfPlayGames": 40, "selfPlayMoves": 3000,
          "trainingPositionsSeen": 16000, "batchSize": 32, "learningRate": 0.001,
          "promoteThreshold": 0.53, "arenaGames": 400,
          "selfPlayTau": {"startTau": 1.0, "decayPerPly": 0.007, "floorTau": 0.5},
          "arenaTau": {"startTau": 0.6, "decayPerPly": 0.02, "floorTau": 0.02},
          "selfPlayWorkerCount": 6,
          \(extraFields)
          "championID": "20261007-1-RCAP", "trainerID": "20261007-2-RCAP", "arenaHistory": []
        }
        """
        return try SessionCheckpointState.decode(Data(json.utf8))
    }

    private func makeResume() -> SessionParameterResume {
        SessionParameterResume(parameters: TrainingParameters.shared, log: { [weak self] line in
            self?.logLines.append(line)
        })
    }

    func test_sessionSavesAndResumeRestoresTheFiveSettings() throws {
        var saved = try sessionState(extraFields: "")
        saved.relativeGradClipMode = 2
        saved.relativeGradClipMultiple = 4
        saved.relativeGradClipWindowSteps = 2000
        saved.relativeGradClipMinHistorySteps = 300
        saved.relativeGradClipFloor = 0.25
        let decoded = try SessionCheckpointState.decode(try saved.encode())
        XCTAssertEqual(decoded.relativeGradClipMode, 2)
        XCTAssertEqual(decoded.relativeGradClipWindowSteps, 2000)

        let p = TrainingParameters.shared
        p.relativeGradClipMode = 1
        p.relativeGradClipMultiple = 3
        p.relativeGradClipWindowSteps = 1000
        p.relativeGradClipMinHistorySteps = 100
        p.relativeGradClipFloor = 0.5
        makeResume().applyGuiSession(decoded, acceptedReplacements: [])
        XCTAssertEqual(p.relativeGradClipMode, 2)
        XCTAssertEqual(p.relativeGradClipMultiple, 4)
        XCTAssertEqual(p.relativeGradClipWindowSteps, 2000)
        XCTAssertEqual(p.relativeGradClipMinHistorySteps, 300)
        XCTAssertEqual(p.relativeGradClipFloor, 0.25)
        for id in Self.ids {
            XCTAssertEqual(logLines.filter { $0.hasPrefix("[RESUME-DIFF] \(id):") }.count, 1, id)
        }
    }

    func test_sessionWrittenBeforeTheCap_resumesWithTheModeOff() throws {
        let p = TrainingParameters.shared
        p.relativeGradClipMode = 2
        p.relativeGradClipMultiple = 4
        makeResume().applyGuiSession(try sessionState(extraFields: ""), acceptedReplacements: [])
        XCTAssertEqual(p.relativeGradClipMode, 0, "a pre-feature run had no relative cap")
        XCTAssertEqual(p.relativeGradClipMultiple, 4, "k is inert while off and keeps the current setting")
    }

    /// Review MAJOR 1: a session saved with W > N is reported on resume and
    /// keeps the current pair, never a crash on every later start.
    func test_sessionWithMinimumHistoryAboveWindow_isReportedAndKeepsTheCurrentPair() throws {
        var saved = try sessionState(extraFields: "")
        saved.relativeGradClipMode = 2
        saved.relativeGradClipWindowSteps = 500
        saved.relativeGradClipMinHistorySteps = 800
        let p = TrainingParameters.shared
        p.relativeGradClipWindowSteps = 1000
        p.relativeGradClipMinHistorySteps = 100
        makeResume().applyGuiSession(saved, acceptedReplacements: [])
        XCTAssertEqual(p.relativeGradClipWindowSteps, 1000)
        XCTAssertEqual(p.relativeGradClipMinHistorySteps, 100)
        XCTAssertEqual(logLines.filter { $0.contains("saved pair refused") }.count, 1, logLines.joined(separator: "\n"))
        XCTAssertNoThrow(try TrainerHyperparameters.validated(p.snapshot()))
    }

    func test_popoverRefusesMinimumHistoryAboveWindow_andWritesNeither() {
        let p = TrainingParameters.shared
        p.relativeGradClipWindowSteps = 1000
        p.relativeGradClipMinHistorySteps = 100
        let model = TrainingSettingsPopoverModel(selfPlayDelayMaxMs: 1000, stepDelayMaxMs: 1000, maxSelfPlayWorkers: 8)
        model.relativeGradClipWindowStepsText = "500"
        model.relativeGradClipMinHistoryStepsText = "600"
        model.save()
        XCTAssertTrue(model.relativeGradClipMinHistoryStepsError)
        XCTAssertEqual(p.relativeGradClipWindowSteps, 1000)
        XCTAssertEqual(p.relativeGradClipMinHistorySteps, 100)

        model.relativeGradClipWindowStepsText = "2000"
        model.relativeGradClipMinHistoryStepsText = "600"
        model.relativeGradClipModeValue = .clip
        model.save()
        XCTAssertFalse(model.relativeGradClipMinHistoryStepsError)
        XCTAssertEqual(p.relativeGradClipWindowSteps, 2000)
        XCTAssertEqual(p.relativeGradClipMinHistorySteps, 600)
        XCTAssertEqual(p.relativeGradClipMode, 2)
    }
}
