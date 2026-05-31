import XCTest
@testable import DrewsChessMachine

/// Integration smoke test for `BatchedSelfPlayDriver`. Spins up the
/// driver with a real `ChessMPSNetwork` and a small K, lets it run
/// briefly, and asserts the end-to-end wiring works: ticks happen,
/// games complete, positions land in the replay buffer, and the
/// shrink-to-zero / grow-from-zero paths don't crash.
///
/// Does NOT attempt byte-identical-vs-legacy determinism: Swift's
/// default `Float.random(in:)` isn't seedable without changing the
/// `MoveSampler` signature to take a custom RNG. Sampling math
/// equivalence is covered by `MoveSamplerTests.swift` (Phase 1);
/// per-game flush layout by `ActiveGameTests.swift` (Phase 3);
/// pointer-evaluate equivalence by
/// `ChessMPSNetworkPointerEvaluateTests.swift` (Phase 2). This file
/// validates only that all those pieces wire together in the driver.
///
/// Runs ~5–15 seconds wall-clock per test on a typical Apple Silicon
/// dev machine (dominated by `ChessMPSNetwork(.randomWeights)` build
/// time + a few seconds of self-play). Long-form distributional /
/// throughput regression checks belong in the manual smoke validation
/// (Phase 6b), not in XCTest.
final class BatchedSelfPlayDriverTests: XCTestCase {

    private static var sharedNetwork: ChessMPSNetwork = {
        do {
            return try ChessMPSNetwork(.randomWeights)
        } catch {
            fatalError("BatchedSelfPlayDriverTests: ChessMPSNetwork(.randomWeights) failed: \(error)")
        }
    }()

    /// Construct a driver wired to fresh test-scoped dependencies.
    private func makeDriver(
        initialK: Int,
        buffer: ReplayBuffer
    ) -> (driver: BatchedSelfPlayDriver, countBox: WorkerCountBox, pauseGate: WorkerPauseGate) {
        let countBox = WorkerCountBox(initial: initialK)
        let pauseGate = WorkerPauseGate()
        let scheduleBox = SamplingScheduleBox(selfPlay: .uniform, arena: .uniform)
        let statsBox = ParallelWorkerStatsBox()
        let diversityTracker = GameDiversityTracker()
        let driver = BatchedSelfPlayDriver(
            network: Self.sharedNetwork,
            buffer: buffer,
            statsBox: statsBox,
            diversityTracker: diversityTracker,
            countBox: countBox,
            pauseGate: pauseGate,
            gameWatcher: nil,
            scheduleBox: scheduleBox,
            replayRatioController: nil
        )
        return (driver, countBox, pauseGate)
    }

    // MARK: - Smoke: driver runs and produces positions

    func test_drivesK2_producesPositionsInReplayBuffer() async {
        let buffer = ReplayBuffer(capacity: 100_000)
        let (driver, _, _) = makeDriver(initialK: 2, buffer: buffer)
        let task = Task(priority: .high) {
            await driver.run()
        }
        // Poll until the first finished game's positions land in the buffer,
        // rather than sleeping a fixed window and hoping a full game completed
        // within it. The buffer fills only on game *completion*, and under
        // `.uniform` sampling a game is hundreds of random plies — so a full
        // game at K=2 takes a few seconds, and longer on a loaded machine,
        // which made a fixed 5s window flaky. Polling returns as soon as data
        // appears (usually within a few seconds) and only fails if no game
        // completes within a generous ~30s deadline.
        var waited = 0
        while buffer.count == 0 && waited < 300 {
            try? await Task.sleep(for: .milliseconds(100))
            waited += 1
        }
        task.cancel()
        // Let the cancel propagate through the loop's top-of-iteration check.
        try? await Task.sleep(for: .milliseconds(50))
        XCTAssertGreaterThan(
            buffer.count, 0,
            "Driver should have produced at least one finished game's worth of positions at K=2 within the deadline"
        )
    }

    // MARK: - Live K change paths

    func test_growFromZero_thenShrinkToZero_noCrash() async {
        let buffer = ReplayBuffer(capacity: 100_000)
        let (driver, countBox, _) = makeDriver(initialK: 0, buffer: buffer)
        let task = Task(priority: .high) {
            await driver.run()
        }
        // K=0 for 250 ms (idle ticks at 50 ms cadence).
        try? await Task.sleep(for: .milliseconds(250))
        // Grow to K=2 and let some plies fire.
        countBox.set(2)
        try? await Task.sleep(for: .seconds(2))
        // Shrink to K=0 (mid-game drops).
        countBox.set(0)
        try? await Task.sleep(for: .milliseconds(250))
        // Grow back to K=1 briefly.
        countBox.set(1)
        try? await Task.sleep(for: .seconds(1))
        task.cancel()
        try? await Task.sleep(for: .milliseconds(50))
    }

    // MARK: - Arena pause / resume

    func test_pauseAndResume_idleDuringPause() async {
        let buffer = ReplayBuffer(capacity: 100_000)
        let (driver, _, pauseGate) = makeDriver(initialK: 2, buffer: buffer)
        let task = Task(priority: .high) {
            await driver.run()
        }
        // Let it produce some positions.
        try? await Task.sleep(for: .seconds(1))

        // Pause and capture the count.
        await pauseGate.pauseAndWait()
        let pausedCount = buffer.count

        // Sleep through the pause; the driver should make no progress.
        try? await Task.sleep(for: .milliseconds(500))
        let afterPauseCount = buffer.count

        // Resume and let some more positions land.
        pauseGate.resume()
        try? await Task.sleep(for: .seconds(1))

        task.cancel()
        try? await Task.sleep(for: .milliseconds(50))

        XCTAssertEqual(
            pausedCount, afterPauseCount,
            "Buffer count must not change while pause is requested (paused=\(pausedCount), after=\(afterPauseCount))"
        )
        XCTAssertGreaterThanOrEqual(
            buffer.count, afterPauseCount,
            "Buffer should resume growth after pause is released"
        )
    }
}
