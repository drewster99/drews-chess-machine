import XCTest
import Metal
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
        XCTAssertNotNil(training.dropoutRateVariable,
                        "training graph must own a dropout rate variable")
        XCTAssertNotNil(training.dropoutRngStateVariable,
                        "training graph must own the dropout RNG state")
        XCTAssertNotNil(training.dropoutRateLoadPlaceholder,
                        "training graph must expose the rate-load placeholder")
        XCTAssertNotNil(training.dropoutRateAssignOp,
                        "training graph must expose the rate assign op")

        let inference = try ChessNetwork(arch: arch, bnMode: .inference)
        XCTAssertNil(inference.dropoutRateVariable,
                     "inference graph must NOT contain a dropout rate variable")
        XCTAssertNil(inference.dropoutRngStateVariable,
                     "inference graph must NOT contain the dropout RNG state")
        XCTAssertNil(inference.dropoutRateLoadPlaceholder,
                     "inference graph must NOT expose a rate-load placeholder")
        XCTAssertNil(inference.dropoutRateAssignOp,
                     "inference graph must NOT expose a rate assign op")
    }
}
