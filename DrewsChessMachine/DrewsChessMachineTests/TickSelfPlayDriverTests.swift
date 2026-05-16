import XCTest
@testable import DrewsChessMachine

/// Integration smoke test for `TickSelfPlayDriver`. Spins up the
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
final class TickSelfPlayDriverTests: XCTestCase {

    private static var sharedNetwork: ChessMPSNetwork = {
        do {
            return try ChessMPSNetwork(.randomWeights)
        } catch {
            fatalError("TickSelfPlayDriverTests: ChessMPSNetwork(.randomWeights) failed: \(error)")
        }
    }()

    /// Construct a driver wired to fresh test-scoped dependencies.
    private func makeDriver(
        initialK: Int,
        buffer: ReplayBuffer
    ) -> (driver: TickSelfPlayDriver, countBox: WorkerCountBox, pauseGate: WorkerPauseGate) {
        let countBox = WorkerCountBox(initial: initialK)
        let pauseGate = WorkerPauseGate()
        let scheduleBox = SamplingScheduleBox(selfPlay: .uniform, arena: .uniform)
        let statsBox = ParallelWorkerStatsBox()
        let diversityTracker = GameDiversityTracker()
        let driver = TickSelfPlayDriver(
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

    /// Drive the loop for `seconds` then cancel and await exit. The
    /// driver self-cancels on `Task.isCancelled` between ticks, so
    /// cancel + a brief `await` is the clean shutdown.
    private func runDriver(_ driver: TickSelfPlayDriver, forSeconds seconds: Double) async {
        let task = Task(priority: .high) {
            await driver.run()
        }
        try? await Task.sleep(for: .seconds(seconds))
        task.cancel()
        // Give the cancel a few ms to propagate through the loop's
        // top-of-iteration check.
        try? await Task.sleep(for: .milliseconds(50))
    }

    // MARK: - Smoke: driver runs and produces positions

    func test_drivesK2_producesPositionsInReplayBuffer() async {
        let buffer = ReplayBuffer(capacity: 100_000)
        let (driver, _, _) = makeDriver(initialK: 2, buffer: buffer)
        await runDriver(driver, forSeconds: 5.0)
        XCTAssertGreaterThan(
            buffer.count, 0,
            "Driver should have produced at least one finished game's worth of positions in 5s at K=2"
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
