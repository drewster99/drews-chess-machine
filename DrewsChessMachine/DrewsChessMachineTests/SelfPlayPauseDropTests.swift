import XCTest
@testable import DrewsChessMachine

/// Determinism plan C1 #11: a pause drops the self-play games in progress,
/// and the driver records how many (and their plies) before it reports the
/// pause, so the session save that paused it can log what its save leaves
/// out.
final class SelfPlayPauseDropTests: XCTestCase {

    func testAPauseRecordsTheGamesItDroppedBeforeReportingThePause() async throws {
        let network = try ChessMPSNetwork(.randomWeights(initSeed: 5))
        let buffer = ReplayBuffer(capacity: 50_000, inputEncoding: network.inputEncoding, sampler: DCMRandom(seed: 5))
        let pauseGate = WorkerPauseGate()
        let pauseDrops = SyncBox<DroppedInFlightGames?>(nil)
        let driver = BatchedSelfPlayDriver(
            network: network,
            buffer: buffer,
            statsBox: ParallelWorkerStatsBox(),
            diversityTracker: GameDiversityTracker(),
            countBox: WorkerCountBox(initial: 3),
            pauseGate: pauseGate,
            gameWatcher: nil,
            scheduleBox: SamplingScheduleBox(selfPlay: .uniform, arena: .uniform),
            replayRatioController: nil,
            randomStreams: DCMRandomStreams(masterSeed: 5),
            gameSerials: GameSerialCounter(firstSerial: 0),
            pauseDrops: pauseDrops
        )
        XCTAssertNil(pauseDrops.value)
        let task = Task { await driver.run() }
        // Let the three games play some plies.
        try await Task.sleep(for: .milliseconds(1500))
        await pauseGate.pauseAndWait()
        let dropped = try XCTUnwrap(pauseDrops.value, "the pause must be recorded before pauseAndWait returns")
        XCTAssertEqual(dropped.games, 3)
        XCTAssertGreaterThanOrEqual(dropped.plies, 0)
        pauseGate.resume()
        task.cancel()
        await task.value
    }
}
