//
//  PolicyTailPrecisionTests.swift
//  DrewsChessMachineTests
//
//  Pins the experimental policy-tail precision switch
//  (`ChessNetwork.PolicyTailPrecision`):
//
//  - `mixedFinalProjection` still emits fp32 head outputs, for every compute
//    dtype and policy head style, in inference and training BN modes.
//  - On an fp32 build the two settings build the same graph: identical
//    weights give bit-identical policy logits.
//  - On a bf16 build the setting really changes the graph (the policy
//    pre-block runs in bf16 instead of fp32), and the logits move only by
//    bf16 rounding.
//  - A trainer built with `mixedFinalProjection` takes a finite step.
//

import XCTest
import Metal
import MetalPerformanceShadersGraph
@testable import DrewsChessMachine

final class PolicyTailPrecisionTests: XCTestCase {

    private func requireMetal() throws {
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("Metal not available") }
    }

    private func arch(_ dtype: ComputeDataType, policy: PolicyHeadStyle? = nil) -> NetworkArchitecture {
        var a = NetworkArchitecture.current
        a.computeDataType = dtype
        if let policy { a.policyHeadStyle = policy }
        return a
    }

    /// The starting position's policy logits from `net`.
    private func startingPolicy(_ net: ChessNetwork) async throws -> [Float] {
        let board = BoardEncoder.encode(.starting, encoding: net.arch.inputEncoding)
        let policyBox = SyncBox<[Float]>([])
        try await net.evaluateWithValueDistribution(board: board) { policy, _ in
            policyBox.value = Array(policy)
        }
        XCTAssertEqual(policyBox.value.count, ChessNetwork.policySize)
        return policyBox.value
    }

    /// Policy logits of the same weights built with each tail precision.
    private func policiesForBothPrecisions(
        _ architecture: NetworkArchitecture
    ) async throws -> (shipped: [Float], mixed: [Float]) {
        let shippedNet = try ChessNetwork(arch: architecture, policyTailPrecision: .float32FromPreBatchNorm)
        let mixedNet = try ChessNetwork(arch: architecture, policyTailPrecision: .mixedFinalProjection)
        try await mixedNet.loadWeights(try await shippedNet.exportWeights())
        return (try await startingPolicy(shippedNet), try await startingPolicy(mixedNet))
    }

    func testMixedFinalProjectionKeepsHeadOutputsFP32() throws {
        try requireMetal()
        for dtype in [ComputeDataType.float32, .bFloat16, .float16] {
            for style in PolicyHeadStyle.allCases {
                for bnMode in [BNMode.inference, .training] {
                    let net = try ChessNetwork(
                        arch: arch(dtype, policy: style), bnMode: bnMode, policyTailPrecision: .mixedFinalProjection)
                    let label = "\(dtype) \(style.rawValue) \(bnMode)"
                    XCTAssertEqual(net.policyTailPrecision, .mixedFinalProjection, label)
                    XCTAssertEqual(net.policyOutput.dataType, .float32, "\(label): policyOutput")
                    XCTAssertEqual(net.valueLogits.dataType, .float32, "\(label): valueLogits")
                }
            }
        }
    }

    func testFP32BuildIsIdenticalUnderBothPrecisions() async throws {
        try requireMetal()
        for style in PolicyHeadStyle.allCases {
            let (shipped, mixed) = try await policiesForBothPrecisions(arch(.float32, policy: style))
            XCTAssertEqual(shipped, mixed, "\(style.rawValue): an fp32 build has no narrow dtype to leave")
        }
    }

    /// The dtype the policy pre-block's activation runs in, read from the
    /// audit tap: a non-fp32 tap is widened for readback by a cast whose
    /// input is the tapped tensor.
    private func policyPreActivationDataType(_ net: ChessNetwork) throws -> MPSDataType {
        let tap = try XCTUnwrap(net.analysisTapReadbacks.first { $0.name == "policy_pre_act" }, "policy_pre_act tap")
        if tap.tensor.dataType == .float32, tap.tensor.operation.name.hasPrefix("analysis_tap_") {
            return try XCTUnwrap(tap.tensor.operation.inputTensors.first, "readback cast input").dataType
        }
        return tap.tensor.dataType
    }

    func testBF16PreBlockRunsInBF16OnlyUnderMixed() throws {
        try requireMetal()
        for style in PolicyHeadStyle.allCases where style != .simpleConv {
            let shipped = try ChessNetwork(
                arch: arch(.bFloat16, policy: style), policyTailPrecision: .float32FromPreBatchNorm, analysisTaps: true)
            let mixed = try ChessNetwork(
                arch: arch(.bFloat16, policy: style), policyTailPrecision: .mixedFinalProjection, analysisTaps: true)
            XCTAssertEqual(try policyPreActivationDataType(shipped), .float32, style.rawValue)
            XCTAssertEqual(try policyPreActivationDataType(mixed), .bFloat16, style.rawValue)
        }
    }

    func testBF16LogitsMoveOnlyByRounding() async throws {
        try requireMetal()
        for style in PolicyHeadStyle.allCases {
            let (shipped, mixed) = try await policiesForBothPrecisions(arch(.bFloat16, policy: style))
            XCTAssertTrue(shipped.allSatisfy(\.isFinite) && mixed.allSatisfy(\.isFinite), style.rawValue)
            // bf16 keeps 8 significant bits (relative step 2^-8 ≈ 0.4%). A
            // handful of roundings through BN and one projection stays well
            // inside 5% of the logits' spread.
            let spread = Double(try XCTUnwrap(shipped.max()) - XCTUnwrap(shipped.min()))
            XCTAssertGreaterThan(spread, 0, style.rawValue)
            let maxDifference = try XCTUnwrap(zip(shipped, mixed).map { Double(abs($0 - $1)) }.max())
            XCTAssertLessThan(maxDifference, 0.05 * spread, "\(style.rawValue): beyond bf16 rounding")
        }
    }

    func testTrainerStepsWithMixedFinalProjection() async throws {
        try requireMetal()
        let trainer = try ChessTrainer(arch: arch(.bFloat16), policyTailPrecision: .mixedFinalProjection)
        XCTAssertEqual(trainer.network.policyTailPrecision, .mixedFinalProjection)
        let timing = try await trainer.trainStep(batchSize: 32)
        XCTAssertTrue(timing.loss.isFinite, "loss must be finite")
        XCTAssertTrue(timing.gradGlobalNorm.isFinite, "gradient norm must be finite")
    }
}
