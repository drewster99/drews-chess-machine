//
//  PolicyTailPrecisionTests.swift
//  DrewsChessMachineTests
//
//  Pins the policy-tail precision (`PolicyTailPrecisionSetting`, an
//  architecture field from format v12) in the graph:
//
//  - `mixedFinalProjection` still emits fp32 head outputs, for every compute
//    dtype and policy head style, in inference and training BN modes, and
//    the network reads its tail from its architecture.
//  - An fp32 architecture has one tail, `does_not_apply`, whichever bf16
//    tail it was converted from, and builds one graph.
//  - On a bf16 build the setting really changes the graph (the policy
//    pre-block runs in bf16 instead of fp32), and the logits move only by
//    bf16 rounding.
//  - A trainer built with `mixedFinalProjection` takes a finite step and
//    builds its network at its architecture's tail.
//

import XCTest
import Metal
import MetalPerformanceShadersGraph
@testable import DrewsChessMachine

final class PolicyTailPrecisionTests: XCTestCase {

    private func requireMetal() throws {
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("Metal not available") }
    }

    /// `.current` at `dtype` under `tail` (nil for fp32), with `policy` as its
    /// policy head style.
    private func arch(_ dtype: ComputeDataType, tail: PolicyTailPrecisionSetting?,
                      policy: PolicyHeadStyle? = nil) throws -> NetworkArchitecture {
        var a = try NetworkArchitecture.current.withComputeDataType(dtype, tail: tail)
        if let policy { a.policyHeadStyle = policy }
        a.clearActivationSitesTheTopologyLacks()
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

    /// Policy logits of the same weights built at each bf16 tail.
    private func bf16PoliciesForBothTails(
        policy: PolicyHeadStyle
    ) async throws -> (fromPreBatchNorm: [Float], mixed: [Float]) {
        let fromPreBatchNormNet = try ChessNetwork(
            arch: arch(.bFloat16, tail: .float32FromPreBatchNorm, policy: policy), initialization: .seeded(initSeed: 1))
        let mixedNet = try ChessNetwork(
            arch: arch(.bFloat16, tail: .mixedFinalProjection, policy: policy), initialization: .seeded(initSeed: 2))
        try await mixedNet.loadWeights(try await fromPreBatchNormNet.exportWeights())
        return (try await startingPolicy(fromPreBatchNormNet), try await startingPolicy(mixedNet))
    }

    func testMixedFinalProjectionKeepsHeadOutputsFP32() throws {
        try requireMetal()
        for dtype in [ComputeDataType.float32, .bFloat16, .float16] {
            let tail: PolicyTailPrecisionSetting? = dtype == .float32 ? nil : .mixedFinalProjection
            for style in PolicyHeadStyle.allCases {
                for bnMode in [BNMode.inference, .training] {
                    let net = try ChessNetwork(
                        arch: arch(dtype, tail: tail, policy: style), bnMode: bnMode, initialization: .seeded(initSeed: 1))
                    let label = "\(dtype) \(style.rawValue) \(bnMode)"
                    XCTAssertEqual(net.arch.policyTailPrecision, tail ?? .doesNotApply, label)
                    XCTAssertEqual(net.policyOutput.dataType, .float32, "\(label): policyOutput")
                    XCTAssertEqual(net.valueLogits.dataType, .float32, "\(label): valueLogits")
                }
            }
        }
    }

    /// Before format v12 the tail was a process setting and an fp32 network
    /// could be built "under" either value, to the same graph. Now an fp32
    /// architecture has exactly one tail: converting either bf16 tail to fp32
    /// gives the same architecture, and two builds of it with the same
    /// weights give bit-identical logits.
    func testFP32BuildIsIdenticalUnderBothPrecisions() async throws {
        try requireMetal()
        for style in PolicyHeadStyle.allCases {
            let fromPreBatchNorm = try arch(.bFloat16, tail: .float32FromPreBatchNorm, policy: style)
                .withComputeDataType(.float32, tail: nil)
            let fromMixed = try arch(.bFloat16, tail: .mixedFinalProjection, policy: style)
                .withComputeDataType(.float32, tail: nil)
            XCTAssertEqual(fromPreBatchNorm, fromMixed, style.rawValue)
            XCTAssertEqual(fromPreBatchNorm.policyTailPrecision, .doesNotApply, style.rawValue)
            let first = try ChessNetwork(arch: fromPreBatchNorm, initialization: .seeded(initSeed: 1))
            let second = try ChessNetwork(arch: fromMixed, initialization: .seeded(initSeed: 2))
            try await second.loadWeights(try await first.exportWeights())
            let firstPolicy = try await startingPolicy(first)
            let secondPolicy = try await startingPolicy(second)
            XCTAssertEqual(firstPolicy.map(\.bitPattern), secondPolicy.map(\.bitPattern),
                           "\(style.rawValue): an fp32 build has no narrow dtype to leave")
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
            let fromPreBatchNorm = try ChessNetwork(
                arch: arch(.bFloat16, tail: .float32FromPreBatchNorm, policy: style),
                initialization: .seeded(initSeed: 1), analysisTaps: true)
            let mixed = try ChessNetwork(
                arch: arch(.bFloat16, tail: .mixedFinalProjection, policy: style),
                initialization: .seeded(initSeed: 2), analysisTaps: true)
            XCTAssertEqual(try policyPreActivationDataType(fromPreBatchNorm), .float32, style.rawValue)
            XCTAssertEqual(try policyPreActivationDataType(mixed), .bFloat16, style.rawValue)
        }
    }

    func testBF16LogitsMoveOnlyByRounding() async throws {
        try requireMetal()
        for style in PolicyHeadStyle.allCases {
            let (fromPreBatchNorm, mixed) = try await bf16PoliciesForBothTails(policy: style)
            XCTAssertTrue(fromPreBatchNorm.allSatisfy(\.isFinite) && mixed.allSatisfy(\.isFinite), style.rawValue)
            // bf16 keeps 8 significant bits (relative step 2^-8 ≈ 0.4%). A
            // handful of roundings through BN and one projection stays well
            // inside 5% of the logits' spread.
            let spread = Double(try XCTUnwrap(fromPreBatchNorm.max()) - XCTUnwrap(fromPreBatchNorm.min()))
            XCTAssertGreaterThan(spread, 0, style.rawValue)
            let maxDifference = try XCTUnwrap(zip(fromPreBatchNorm, mixed).map { Double(abs($0 - $1)) }.max())
            XCTAssertLessThan(maxDifference, 0.05 * spread, "\(style.rawValue): beyond bf16 rounding")
        }
    }

    func testTrainerStepsWithMixedFinalProjection() async throws {
        try requireMetal()
        let trainer = try ChessTrainer(dropoutStream: DCMRandom(seed: 1),
                                       arch: arch(.bFloat16, tail: .mixedFinalProjection),
                                       initialization: .seeded(initSeed: 1))
        XCTAssertEqual(trainer.network.arch.policyTailPrecision, .mixedFinalProjection)
        let timing = try await trainer.trainStep(batchSize: 32)
        XCTAssertTrue(timing.loss.isFinite, "loss must be finite")
        XCTAssertTrue(timing.gradGlobalNorm.isFinite, "gradient norm must be finite")
    }

    /// The network and the trainer take the tail from the architecture: no
    /// process value, no init parameter.
    func testNetworkAndTrainerTakeTheTailFromTheArchitecture() throws {
        try requireMetal()
        let fromPreBatchNorm = try arch(.bFloat16, tail: .float32FromPreBatchNorm)
        let network = try ChessMPSNetwork(.randomWeights(initSeed: 1), arch: fromPreBatchNorm)
        XCTAssertEqual(network.network.arch.policyTailPrecision, .float32FromPreBatchNorm)
        let trainer = try ChessTrainer(dropoutStream: DCMRandom(seed: 1), arch: fromPreBatchNorm, initialization: .seeded(initSeed: 1))
        XCTAssertEqual(trainer.arch.policyTailPrecision, .float32FromPreBatchNorm)
        XCTAssertEqual(trainer.network.arch.policyTailPrecision, .float32FromPreBatchNorm)
    }
}
