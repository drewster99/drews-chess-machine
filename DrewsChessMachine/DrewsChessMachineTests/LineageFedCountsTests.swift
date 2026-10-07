//
//  LineageFedCountsTests.swift
//  DrewsChessMachineTests
//
//  A GUI lineage segment's fed totals are `carry + (box count − baseline)`
//  over every stats box the segment counted on. A promotion zeroes the box's
//  game stats so the display shows the new champion's self-play; the
//  segment must bank what the box counted in the same lock acquisition as
//  the reset, or games are lost — and a reset with no banking at all (what
//  Promote Trainee Now did) leaves a resumed run's box below its baseline,
//  so every later save's record has a negative game count and fails.
//

import XCTest
@testable import DrewsChessMachine

@MainActor
final class LineageFedCountsTests: XCTestCase {

    private nonisolated static let flushed = FlushedGameStats(positions: GuiSaveHarness.positionsPerGame,
                                                   phaseByPly: .zero, phaseByMaterial: .zero)

    nonisolated func testResetReturningEmittedCountsLosesNoConcurrentGame() {
        let box = ParallelWorkerStatsBox(sessionStart: Date())
        let writers = 4
        let gamesPerWriter = 3000
        let writersDone = DispatchGroup()
        let banked = SyncBox<(games: Int, positions: Int)>((0, 0))
        let stopResetting = SyncBox<Bool>(false)
        let resetterDone = DispatchGroup()
        // The test thread waits on these; the same QoS avoids an inversion.
        let queue = DispatchQueue.global(qos: .userInteractive)
        let resets = SyncBox<Int>(0)
        queue.async(group: resetterDone) {
            while !stopResetting.value {
                let counts = box.resetGameStatsReturningEmittedCounts()
                banked.modify { $0 = ($0.games + counts.games, $0.positions + counts.positions) }
                resets.modify { $0 += 1 }
            }
        }
        for _ in 0..<writers {
            queue.async(group: writersDone) {
                // Write only once resets are running, so the two overlap.
                while resets.value == 0 {}
                for _ in 0..<gamesPerWriter {
                    box.recordEmittedGame(result: .stalemate, flushed: Self.flushed)
                }
            }
        }
        writersDone.wait()
        stopResetting.value = true
        resetterDone.wait()
        let remaining = box.snapshot()
        XCTAssertGreaterThan(resets.value, 1)
        XCTAssertEqual(banked.value.games + remaining.emittedGames, writers * gamesPerWriter)
        XCTAssertEqual(banked.value.positions + remaining.emittedPositions,
                       writers * gamesPerWriter * GuiSaveHarness.positionsPerGame)
    }

    func testANewChampionResetKeepsTheSegmentsFedTotalOnAResumedBox() throws {
        let harness = try GuiSaveHarness(resumedEmittedGames: 500)
        defer {
            do { try harness.removeSessionsDirectory() } catch { XCTFail("\(error)") }
        }
        for _ in 0..<10 { harness.box.recordEmittedGame(result: .stalemate, flushed: Self.flushed) }
        harness.controller.resetSelfPlayGameStatsForNewChampion()
        XCTAssertEqual(harness.box.snapshot().emittedGames, 0, "the display shows only the new champion's games")
        for _ in 0..<3 { harness.box.recordEmittedGame(result: .stalemate, flushed: Self.flushed) }

        let record = try harness.controller.lineageRecordForSave(
            at: Date(), cut: try harness.controller.takeConfigurationCut(trainer: harness.trainer),
            trainerCompletedSteps: harness.trainer.completedTrainSteps,
            dropoutPhiloxState: nil, dropoutStreamState: nil)
        XCTAssertEqual(record.fed.segmentGames, 13)
        XCTAssertEqual(record.fed.segmentPositions, 13 * GuiSaveHarness.positionsPerGame)
    }

    /// Promote Trainee Now on a resumed run: its own autosave (and every
    /// later save) must still build a record.
    func testPromoteTraineeNowOnAResumedRunStillSaves() async throws {
        let harness = try GuiSaveHarness(resumedEmittedGames: 1_000_000)
        defer {
            do { try harness.removeSessionsDirectory() } catch { XCTFail("\(error)") }
        }
        harness.startWorkers()
        while harness.serials.nextSerial < 20 {
            try await Task.sleep(for: .milliseconds(10))
        }
        harness.controller.promoteTrainerNow(sessionsDirectory: harness.sessionsDirectory)
        let saved = try await harness.waitForSavedSession(timeout: 30)
        harness.stopWorkers()

        let url = try XCTUnwrap(saved, "the promotion's autosave was not written")
        let loaded = try CheckpointManager.loadSession(at: url)
        let fed = try XCTUnwrap(loaded.trainerFile.safetensorsProvenance?.lineage.record?.fed)
        XCTAssertGreaterThanOrEqual(fed.segmentGames, 20)
        XCTAssertEqual(fed.cumGames, fed.segmentGames, "the parent record fed nothing, so the totals are the segment's")
        XCTAssertNoThrow(try harness.controller.lineageRecordForSave(
            at: Date(), cut: try harness.controller.takeConfigurationCut(trainer: harness.trainer),
            trainerCompletedSteps: harness.trainer.completedTrainSteps,
            dropoutPhiloxState: nil, dropoutStreamState: nil))
    }
}
