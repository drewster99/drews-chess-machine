import XCTest
import Metal
import MetalPerformanceShadersGraph
@testable import DrewsChessMachine

/// Phase P4 of the determinism plan: the training graph's dropout RNG (a
/// Philox state variable MPSGraph advances once per step) is seeded from the
/// run's `dropout` stream, can be read back, and can be written back so a
/// resumed run continues the exact mask sequence.
///
/// The mask a step draws is a pure function of the Philox state at the start
/// of the step and the batch shape, so these tests compare state sequences:
/// equal states at the start of every step, with equal batch shapes, are equal
/// masks.
final class DropoutRNGStateTests: XCTestCase {

    /// One small group with a nonzero dropout multiplier, so the training graph
    /// actually draws dropout randomness and builds the per-step advance.
    private func archWithDropout() -> NetworkArchitecture {
        var arch = NetworkArchitecture.current
        arch.blockGroups = [
            BlockGroup(
                count: 2, channels: NetworkArchitecture.current.towerOutputChannels,
                conv1KernelSize: 3, conv2KernelSize: 3,
                seStyle: .none, seReductionRatio: 4,
                useRezero: true, rezeroAlphaInit: 0.5,
                activationFunction: .relu, activationStyle: .pre,
                skipMerge: .cleanAdd, dropoutMultiplier: 0.5
            )
        ]
        return arch
    }

    private func makeTrainer(dropoutSeed: UInt64) throws -> ChessTrainer {
        let trainer = try ChessTrainer(dropoutStream: DCMRandom(seed: dropoutSeed), arch: archWithDropout(), initialization: .seeded(initSeed: 1))
        trainer.dropoutRate = 0.2
        return trainer
    }

    /// Captured state after each of `steps` random-data training steps.
    private func stateSequence(of trainer: ChessTrainer, steps: Int) async throws -> [DropoutPhiloxState] {
        var states: [DropoutPhiloxState] = []
        for _ in 0..<steps {
            _ = try await trainer.trainStep(batchSize: 8)
            states.append(try await trainer.captureDropoutState())
        }
        return states
    }

    private func requireMetal() throws {
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("Metal not available") }
    }

    /// Two trainers seeded from the same `dropout` stream start from the same
    /// Philox state and stay in step; a different seed starts elsewhere.
    func testSameDropoutStreamSeedGivesTheSameStateSequence() async throws {
        try requireMetal()
        let first = try makeTrainer(dropoutSeed: 7)
        let second = try makeTrainer(dropoutSeed: 7)
        let other = try makeTrainer(dropoutSeed: 8)

        let firstStart = try await first.captureDropoutState()
        let secondStart = try await second.captureDropoutState()
        let otherStart = try await other.captureDropoutState()
        XCTAssertEqual(firstStart, secondStart, "same dropout stream seed must give the same initial Philox state")
        XCTAssertNotEqual(firstStart, otherStart, "a different dropout stream seed must give a different initial state")

        let firstSequence = try await stateSequence(of: first, steps: 3)
        let secondSequence = try await stateSequence(of: second, steps: 3)
        XCTAssertEqual(firstSequence, secondSequence, "same seed and same batch shapes must advance identically")
        XCTAssertEqual(Set(firstSequence.map(\.words)).count, firstSequence.count,
                       "every step must advance the state")
        XCTAssertFalse(firstSequence.contains(firstStart), "the first step must move the state off its seed")
    }

    /// Capture, run three steps, restore the captured state, run three more:
    /// the second run repeats the first state for state.
    func testRestoringACapturedStateRepeatsTheSequence() async throws {
        try requireMetal()
        let trainer = try makeTrainer(dropoutSeed: 11)
        _ = try await trainer.trainStep(batchSize: 8)
        let saved = try await trainer.captureDropoutState()
        let firstRun = try await stateSequence(of: trainer, steps: 3)
        try await trainer.restoreDropoutState(saved)
        let afterRestore = try await trainer.captureDropoutState()
        XCTAssertEqual(afterRestore, saved)
        let secondRun = try await stateSequence(of: trainer, steps: 3)
        XCTAssertEqual(firstRun, secondRun, "restoring a captured state must resume the same mask sequence")
    }

    /// The exact-resume snapshot carries the Philox state and `restoreExactly`
    /// puts it back, even into a trainer seeded differently.
    func testExactResumeSnapshotCarriesTheDropoutState() async throws {
        try requireMetal()
        let source = try makeTrainer(dropoutSeed: 21)
        _ = try await stateSequence(of: source, steps: 2)
        let snapshot = try await source.exportResumeSnapshot()
        guard case .philox(let carried) = snapshot.dropoutRNG else {
            return XCTFail("a live trainer's snapshot must carry its Philox state, got \(snapshot.dropoutRNG)")
        }
        let sourceNow = try await source.captureDropoutState()
        XCTAssertEqual(carried, sourceNow)

        let target = try makeTrainer(dropoutSeed: 99)
        try await target.restoreExactly(from: snapshot)
        let targetNow = try await target.captureDropoutState()
        XCTAssertEqual(targetNow, carried)
        let sourceNext = try await stateSequence(of: source, steps: 2)
        let targetNext = try await stateSequence(of: target, steps: 2)
        XCTAssertEqual(sourceNext, targetNext, "the resumed trainer must continue the source's mask sequence")
    }

    /// Probe isolation: running the KL probe on every step must not change the
    /// dropout sequence the training steps see.
    func testKLProbeDoesNotChangeTheDropoutSequence() async throws {
        try requireMetal()
        let withProbe = try makeTrainer(dropoutSeed: 31)
        withProbe.klProbeInterval = 1
        let withoutProbe = try makeTrainer(dropoutSeed: 31)
        withoutProbe.klProbeInterval = 0
        let probed = try await stateSequence(of: withProbe, steps: 4)
        let unprobed = try await stateSequence(of: withoutProbe, steps: 4)
        XCTAssertEqual(probed, unprobed, "the KL probe must not add, skip or reorder dropout advances")
    }

    /// The KL probe schedule is derived from the step index. On a fresh run it
    /// fires on the same steps the old per-process counter did: the first step,
    /// then every `interval` steps.
    func testKLProbeScheduleMatchesTheOldCounterOnAFreshRun() {
        for interval in [0, 1, 2, 3, 7] {
            var oldCounter = 0
            for completedBefore in 0..<40 {
                let oldFires = interval > 0 && oldCounter % interval == 0
                oldCounter += 1
                XCTAssertEqual(
                    ChessTrainer.isKLProbeStep(stepIndex: completedBefore, interval: interval), oldFires,
                    "interval \(interval), step \(completedBefore)"
                )
            }
        }
    }

    /// A Philox state is exactly seven 32-bit words; anything else is refused.
    func testPhiloxStateRefusesTheWrongWordCount() {
        XCTAssertThrowsError(try DropoutPhiloxState(words: [1, 2, 3]))
        XCTAssertThrowsError(try DropoutPhiloxState(words: Array(repeating: 0, count: 8)))
        XCTAssertNoThrow(try DropoutPhiloxState(words: Array(repeating: 0, count: 7)))
    }

    /// Canary for an OS change in MPSGraph's Philox layout: the state derived
    /// from a fixed seed must stay the same seven words. A failure after an OS
    /// update means dropout streams are not comparable across that update.
    func testDerivedStateForAFixedSeedIsStable() throws {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
            throw XCTSkip("Metal not available")
        }
        let first = try DropoutPhiloxState.derived(fromSeed: 0x5EED, device: device, commandQueue: queue)
        let second = try DropoutPhiloxState.derived(fromSeed: 0x5EED, device: device, commandQueue: queue)
        XCTAssertEqual(first, second)
        XCTAssertEqual(first.words, DropoutRNGStateTests.pinnedStateForSeed5EED,
                       "MPSGraph's Philox state for a fixed seed changed: \(first.words)")
    }

    /// Recorded on macOS 27 beta / Xcode 27.2 beta for `randomPhiloxStateTensor(withSeed: 0x5EED)`.
    static let pinnedStateForSeed5EED: [Int32] = [
        1, 11912374, -1985430587, 2110984159, 266232230, 67638122, 2073528346
    ]
}
