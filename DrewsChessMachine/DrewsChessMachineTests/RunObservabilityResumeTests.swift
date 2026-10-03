//
//  RunObservabilityResumeTests.swift
//  DrewsChessMachineTests
//
//  Determinism plan C1 #18 and #20: a session save carries the self-play
//  diversity window and the alarm streak counters, and a resume continues
//  them instead of starting empty.
//

import XCTest
@testable import DrewsChessMachine

@MainActor
final class RunObservabilityResumeTests: XCTestCase {

    private func games(_ count: Int) -> [[ChessMove]] {
        let opening: [ChessMove] = [
            ChessMove(fromRow: 6, fromCol: 4, toRow: 4, toCol: 4, promotion: nil),
            ChessMove(fromRow: 1, fromCol: 4, toRow: 3, toCol: 4, promotion: nil),
        ]
        let knightMoves: [ChessMove] = [
            ChessMove(fromRow: 7, fromCol: 6, toRow: 5, toCol: 5, promotion: nil),
            ChessMove(fromRow: 7, fromCol: 1, toRow: 5, toCol: 2, promotion: nil),
            ChessMove(fromRow: 6, fromCol: 3, toRow: 4, toCol: 3, promotion: nil),
        ]
        return (0..<count).map { opening + [knightMoves[$0 % knightMoves.count]] }
    }

    /// A restored window holds the same games, in the same order, and reports
    /// the same diversity as the tracker it was saved from — including after
    /// the window has wrapped.
    func testARestoredDiversityWindowMatchesTheSavedOne() {
        let saved = GameDiversityTracker(windowSize: 5)
        for game in games(8) { saved.recordGame(moves: game) }
        let window = saved.windowSequences()
        XCTAssertEqual(window.count, 5)

        let restored = GameDiversityTracker(windowSize: 5)
        restored.restore(windowSequences: window)
        XCTAssertEqual(restored.windowSequences(), window)
        let restoredSnapshot = restored.snapshot()
        let savedSnapshot = saved.snapshot()
        XCTAssertEqual(restoredSnapshot.gamesInWindow, savedSnapshot.gamesInWindow)
        XCTAssertEqual(restoredSnapshot.uniqueGames, savedSnapshot.uniqueGames)
        XCTAssertEqual(restoredSnapshot.avgDivergencePly, savedSnapshot.avgDivergencePly)
        XCTAssertEqual(restoredSnapshot.divergenceHistogram, savedSnapshot.divergenceHistogram)

        // Both keep evolving identically.
        let next = games(10)[9]
        saved.recordGame(moves: next)
        restored.recordGame(moves: next)
        XCTAssertEqual(restored.windowSequences(), saved.windowSequences())
    }

    func testAlarmStreaksRoundTripThroughTheControllerAndSessionJSON() throws {
        var streaks = TrainingAlarmController.Streaks()
        streaks.divergenceWarning = 2
        streaks.valueDrawCollapseCritical = 1
        streaks.valueAbsMeanSaturationRecovery = 7
        let controller = TrainingAlarmController()
        controller.restore(streaks: streaks)
        XCTAssertEqual(controller.streaks, streaks)

        let state = SessionCheckpointState(
            formatVersion: SessionCheckpointState.currentFormatVersion,
            sessionID: "obs", savedAtUnix: 1_790_000_000, sessionStartUnix: 1_789_999_000,
            elapsedTrainingSec: 10, trainingSteps: 1, selfPlayGames: 1, selfPlayMoves: 1,
            trainingPositionsSeen: 1, batchSize: 1, learningRate: 1e-3,
            promoteThreshold: 0.55, arenaGames: 10,
            selfPlayTau: TauConfigCodable(SamplingSchedule.selfPlay),
            arenaTau: TauConfigCodable(SamplingSchedule.arena),
            selfPlayWorkerCount: 1, championID: "c", trainerID: "t", arenaHistory: []
        ).withLineage(LineageRecord.sessionTestFixture)
            .withRunObservability(diversityWindow: [[1, 2, 3], [4]], alarmStreaks: streaks)
        let decoded = try SessionCheckpointState.decode(try state.encode())
        XCTAssertEqual(decoded.selfPlayDiversityWindow, [[1, 2, 3], [4]])
        XCTAssertEqual(decoded.trainingAlarmStreaks, streaks)
    }
}
