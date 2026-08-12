import XCTest
import Metal
import MetalPerformanceShadersGraph
@testable import DrewsChessMachine

/// Closes the gap left open by `DropoutGraphSemanticsTests` /
/// `DropoutMultiplierSemanticsTests`: those verify the
/// rate → ×multiplier → cap → keep/scale arithmetic on a hand-built parallel
/// subgraph, but NOT that the production builder attaches dropout only to the
/// training graph. A regression that leaked dropout into an inference graph —
/// or dropped it from training — would leave both of those suites green while
/// silently corrupting every champion / arena / probe evaluation. This suite
/// asserts the wiring directly against a real `ChessNetwork`.
final class DropoutGraphWiringTests: XCTestCase {

    /// A small tower whose single group requests a nonzero dropout multiplier,
    /// so dropout is genuinely in play when the graph is built in training mode.
    /// Width is the current preset's tower-output width so the value-head
    /// constraint (`valueHeadConvChannels <= towerOutputChannels`) is satisfied
    /// by construction regardless of the preset's value-head size; SE is off so
    /// the test doesn't couple to that width's reduction-ratio divisibility.
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

    /// A training-mode build carries the live dropout scaffolding (rate
    /// variable, RNG state, and the rate-load assign plumbing); an
    /// inference build of the SAME architecture carries none of it — so a
    /// champion / arena / probe inference pass can never apply dropout.
    func testDropoutScaffoldingIsTrainingGraphOnly() throws {
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal not available")
        }
        let arch = archWithDropout()

        let training = try ChessNetwork(arch: arch, bnMode: .training)
        XCTAssertNotNil(training.dropoutRateFeedPlaceholder,
                        "training graph must own a dropout rate placeholder")
        XCTAssertNotNil(training.dropoutRngStateVariable,
                        "training graph must own the dropout RNG state")
        XCTAssertNotNil(training.dropoutRateZeroTensorData,
                        "training graph must expose the zero-rate binding used by dropout-free consumers")
        XCTAssertNotNil(training.dropoutRateLiveNDArray,
                        "training graph must expose the live-rate buffer bound by the training step")
        XCTAssertNotNil(training.dropoutRateLiveTensorData,
                        "training graph must expose the live-rate binding")

        let inference = try ChessNetwork(arch: arch, bnMode: .inference)
        XCTAssertNil(inference.dropoutRateFeedPlaceholder,
                     "inference graph must NOT contain a dropout rate placeholder")
        XCTAssertNil(inference.dropoutRngStateVariable,
                     "inference graph must NOT contain the dropout RNG state")
        XCTAssertNil(inference.dropoutRateZeroTensorData,
                     "inference graph must NOT expose a zero-rate binding")
        XCTAssertNil(inference.dropoutRateLiveNDArray,
                     "inference graph must NOT expose a live-rate buffer")
        XCTAssertNil(inference.dropoutRateLiveTensorData,
                     "inference graph must NOT expose a live-rate binding")
    }

    /// Read the single fp32 scalar out of a `[1]` rate buffer.
    private func readRate(_ ndArray: MPSNDArray) -> Float {
        var value = Float.nan
        ndArray.readBytes(&value, strideBytes: nil)
        return value
    }

    /// The live rate and the zero rate must be SEPARATE buffers, and setting a
    /// rate must move only the live one.
    ///
    /// This is the half of the fed-placeholder design that
    /// `testValueBaselineIsDropoutFree` cannot see. That test asserts the
    /// baseline is invariant to the rate — which would also hold if dropout
    /// were disabled everywhere, e.g. if the training step bound the zero
    /// buffer by mistake, or if both bindings aliased one buffer. Either
    /// mistake silently turns off regularization while every other dropout
    /// suite stays green, so the distinctness is asserted directly.
    func testLiveRateBufferIsSeparateFromZeroAndTracksTheSetter() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal not available")
        }
        let trainer = try ChessTrainer(arch: archWithDropout())
        let live = try XCTUnwrap(trainer.network.dropoutRateLiveNDArray)
        let zero = try XCTUnwrap(trainer.network.dropoutRateZeroTensorData).mpsndarray()

        XCTAssertFalse(live === zero, "live and zero rate bindings must not alias one buffer")
        XCTAssertEqual(readRate(live), 0, accuracy: 1e-7, "live rate must start at 0")
        XCTAssertEqual(readRate(zero), 0, accuracy: 1e-7, "zero rate must start at 0")

        trainer.dropoutRate = 0.25
        await trainer.awaitPendingWork()

        XCTAssertEqual(readRate(live), 0.25, accuracy: 1e-6,
                       "setting the rate must write the live buffer the training step binds")
        XCTAssertEqual(readRate(zero), 0, accuracy: 1e-7,
                       "the zero buffer every dropout-free consumer binds must never be written")
    }

    /// A network reset must carry the configured dropout rate onto the new
    /// graph. The rate lives in a per-network buffer built holding 0 while
    /// `dropoutRate` (what every reader reports) survives the reset, so without
    /// an explicit re-apply a reset silently trains at 0 while claiming
    /// otherwise.
    func testDropoutRateSurvivesNetworkReset() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal not available")
        }
        let trainer = try ChessTrainer(arch: archWithDropout())
        trainer.dropoutRate = 0.3
        // No wait needed: the setter stores `_dropoutRate` synchronously, and
        // `resetNetwork()` is itself a FIFO barrier on the same queue whose last
        // act is re-writing the NEW network's buffer from that stored value.

        try await trainer.resetNetwork()

        XCTAssertEqual(trainer.dropoutRate, 0.3, accuracy: 1e-6,
                       "the reported rate must survive a reset")
        let live = try XCTUnwrap(trainer.network.dropoutRateLiveNDArray)
        XCTAssertEqual(readRate(live), 0.3, accuracy: 1e-6,
                       "the NEW network's live-rate buffer must carry the configured rate, "
                       + "otherwise the reset silently trains dropout-free")
    }

    /// The training step must bind the LIVE rate buffer — the one thing that
    /// makes dropout actually apply during training.
    ///
    /// This closes a hole every other dropout test misses. `testValueBaseline…`
    /// asserts the baseline is invariant to the rate; the buffer test above
    /// asserts the two buffers are distinct and that the setter moves the live
    /// one. **All of them still pass if `runPreparedStep` binds
    /// `dropoutRateZeroTensorData`**, because none of them observes what the
    /// training step binds. The result would be training silently running
    /// dropout-free while the whole dropout suite stays green — the exact
    /// failure the fed-placeholder design is supposed to make impossible.
    ///
    /// Asserted structurally rather than numerically: both `trainStep` entry
    /// points synthesize or randomly sample their minibatch, so a "weights
    /// diverge at rate 0 vs rate 0.9" comparison would differ because the DATA
    /// differed and pass for the wrong reason.
    func testTrainingStepBindsTheLiveDropoutRate() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal not available")
        }
        let trainer = try ChessTrainer(arch: archWithDropout())
        XCTAssertNil(trainer.lastStepBoundLiveDropoutRate,
                     "nothing should be recorded before the first step")

        trainer.dropoutRate = 0.5
        _ = try await trainer.trainStep(batchSize: 8)

        XCTAssertEqual(trainer.lastStepBoundLiveDropoutRate, true,
                       "the training step must bind the live rate buffer; binding the "
                       + "preallocated zero instead would train dropout-free with no "
                       + "other test noticing")
    }

    /// The value baseline v(s) feeding the advantage `(z − vBaseline)` must be a
    /// CLEAN estimate — computed with dropout off — regardless of the live
    /// training dropout rate.
    ///
    /// `valueBaselineExecutable` compiles against `valueOutputFP32`, which on a
    /// training-mode graph sits downstream of every block's channel-dropout node.
    /// So at rate > 0 the baseline was evaluated through a masked tower, making
    /// the advantage the training step consumes a function of that step's dropout
    /// draw rather than of the position. This asserts the baseline is invariant to
    /// the rate: same boards + same weights ⇒ same v(s) at rate 0 and rate 0.5.
    ///
    /// A rate-dependent baseline is invisible to every other dropout suite —
    /// they check the mask arithmetic and the training/inference wiring split, not
    /// which executables the mask reaches.
    func testValueBaselineIsDropoutFree() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal not available")
        }
        let arch = archWithDropout()
        let trainer = try ChessTrainer(arch: arch)
        // Fully serialize the baseline so the result buffer has settled before
        // the host reads it (the default path commits without waiting).
        trainer.network.blockingValueBaseline = true

        let count = 8
        let planeFloats = arch.inputPlanes * 8 * 8
        // Deterministic, non-constant input: a constant board would let a
        // degenerate tower produce rate-independent output for the wrong reason.
        var boards = [Float](repeating: 0, count: count * planeFloats)
        for i in boards.indices {
            boards[i] = Float((i &* 7919) % 13) / 13.0
        }

        let liveRate = try XCTUnwrap(trainer.network.dropoutRateLiveNDArray)
        func baseline(atRate rate: Float) async throws -> [Float] {
            trainer.dropoutRate = rate
            // The setter writes the live buffer on the TRAINER's execution
            // queue; the baseline below runs on the NETWORK's. Two independent
            // serial queues with no ordering between them, so drain the trainer
            // queue and then PIN the buffer's contents. Both halves matter: a
            // build that wrongly fed the live rate into the baseline could
            // otherwise be sampled while the buffer still read 0, compare two
            // rate-0 baselines, and pass while the regression walked free.
            await trainer.awaitPendingWork()
            XCTAssertEqual(readRate(liveRate), rate, accuracy: 1e-6,
                           "live rate must read \(rate) before the baseline is sampled, "
                           + "otherwise this comparison proves nothing about dropout")
            nonisolated(unsafe) var out: [Float] = []
            try await trainer.network.computeValueBaselineGPU(
                batchBoards: boards,
                count: count
            ) { td in
                out = ChessNetwork.readFloatsFP32(from: td, count: count)
            }
            return out
        }

        let clean = try await baseline(atRate: 0)
        let dropped = try await baseline(atRate: 0.5)

        XCTAssertEqual(clean.count, count, "baseline must yield one v(s) per position")
        XCTAssertEqual(dropped.count, count, "baseline must yield one v(s) per position")
        // Guard against a vacuous pass: if the tower emitted a constant, the
        // equality below would hold for reasons unrelated to dropout.
        XCTAssertGreaterThan(
            (clean.max() ?? 0) - (clean.min() ?? 0), 1e-6,
            "test input must produce varying v(s), otherwise the comparison is vacuous"
        )
        for i in 0..<count {
            XCTAssertEqual(
                clean[i], dropped[i], accuracy: 1e-6,
                "v(s) at position \(i) changed with the dropout rate "
                + "(rate 0: \(clean[i]), rate 0.5: \(dropped[i])) — the value "
                + "baseline is being computed through the dropout-masked tower"
            )
        }
    }
}
