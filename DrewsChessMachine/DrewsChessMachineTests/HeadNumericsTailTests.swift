//
//  HeadNumericsTailTests.swift
//  DrewsChessMachineTests
//
//  Pins the heads' fp32 tails and the trainer's centered head losses:
//
//  - Every head output (`policyOutput`, `valueLogits`, `valueProbs`,
//    `valueOutput`) is fp32 for every compute dtype, every policy head style,
//    and training-mode BN.
//  - A bf16 forward reads the value head back as fp32: the W/D/L softmax
//    sums to 1 at fp32 precision, which a bf16 readback cannot reach.
//  - In the real training graph, the policy's final bias and the W/D/L
//    head's final layer receive no gradient along the shared direction.
//  - The scalar-tanh value head is NOT centered (centering its single logit
//    would pin it at 0 and cut its gradient).
//  - The pre-centering mean-logit monitors are measured on a diagnostics
//    step.
//

import XCTest
import Metal
import MetalPerformanceShadersGraph
@testable import DrewsChessMachine

final class HeadNumericsTailTests: XCTestCase {

    private func requireMetal() throws {
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("Metal not available") }
    }

    private func arch(
        _ dtype: ComputeDataType,
        policy: PolicyHeadStyle? = nil,
        value: ValueHeadStyle? = nil
    ) -> NetworkArchitecture {
        var a = NetworkArchitecture.current
        a.computeDataType = dtype
        if let policy { a.policyHeadStyle = policy }
        if let value { a.valueHeadStyle = value }
        return a
    }

    private func assertHeadOutputsFP32(_ net: ChessNetwork, _ label: String,
                                       file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(net.policyOutput.dataType, .float32, "\(label): policyOutput", file: file, line: line)
        XCTAssertEqual(net.valueLogits.dataType, .float32, "\(label): valueLogits", file: file, line: line)
        XCTAssertEqual(net.valueProbs.dataType, .float32, "\(label): valueProbs", file: file, line: line)
        XCTAssertEqual(net.valueOutput.dataType, .float32, "\(label): valueOutput", file: file, line: line)
        XCTAssertEqual(ChessNetwork.headTailDataType, .float32)
    }

    // MARK: - Output dtypes

    func testHeadOutputsAreFP32ForEveryComputeDtypeAndPolicyStyle() throws {
        try requireMetal()
        for dtype in [ComputeDataType.float32, .bFloat16, .float16] {
            for style in PolicyHeadStyle.allCases {
                let net = try ChessNetwork(arch: arch(dtype, policy: style))
                assertHeadOutputsFP32(net, "\(dtype) \(style.rawValue)")
            }
        }
    }

    func testHeadOutputsAreFP32UnderTrainingMode() throws {
        try requireMetal()
        let scalar = try ChessNetwork(arch: arch(.bFloat16, value: .scalarTanh), bnMode: .training)
        assertHeadOutputsFP32(scalar, "bf16 scalar-tanh, training BN")
    }

    // MARK: - Readback

    func testBF16ForwardReadsTheValueHeadBackInFP32() async throws {
        try requireMetal()
        let net = try ChessNetwork(arch: arch(.bFloat16))
        let board = BoardEncoder.encode(.starting, encoding: net.arch.inputEncoding)
        let policyBox = SyncBox<[Float]>([])
        let wdlBox = SyncBox<[Float]>([])
        try await net.evaluateWithValueDistribution(board: board) { policy, wdl in
            policyBox.value = Array(policy)
            wdlBox.value = [wdl.win, wdl.draw, wdl.loss]
        }
        XCTAssertEqual(policyBox.value.count, ChessNetwork.policySize)
        XCTAssertTrue(policyBox.value.allSatisfy(\.isFinite))
        let wdl = wdlBox.value
        XCTAssertEqual(wdl.count, 3)
        // A bf16 softmax readback carries an 8-bit mantissa; its three
        // probabilities typically miss 1 by far more than this.
        let sum = wdl.reduce(0.0) { $0 + Double($1) }
        XCTAssertEqual(sum, 1, accuracy: 1e-5, "W/D/L must sum to 1 at fp32 precision")

        let valueBox = SyncBox<Float>(.nan)
        try await net.evaluate(board: board) { _, value in valueBox.value = value }
        XCTAssertEqual(Double(valueBox.value), Double(wdl[0]) - Double(wdl[2]), accuracy: 1e-5,
                       "the value scalar is p_win − p_loss of the same fp32 softmax")
    }

    // MARK: - Training graph

    /// Velocity after one step at μ = 0 is exactly the clipped gradient, so
    /// it exposes the gradient of every trainable. The trainer's weights are
    /// ordered like the safetensors tensor names, which gives the indices.
    private func velocities(
        after trainer: ChessTrainer, architecture: NetworkArchitecture
    ) async throws -> (weights: [[Float]], names: [String]) {
        let weights = try await trainer.exportTrainerWeights()
        let names = SafetensorsModelIO.tensorNames(for: architecture, includesVelocity: true)
        XCTAssertEqual(weights.count, names.count, "trainer export must match the tensor-name layout")
        return (weights, names)
    }

    private func tensor(_ name: String, in exported: (weights: [[Float]], names: [String])) throws -> [Float] {
        let i = try XCTUnwrap(exported.names.firstIndex(of: name), "\(name) missing")
        return exported.weights[i]
    }

    /// Assert Σ values ≈ 0 relative to Σ|values|, and that the values are
    /// not all zero (a vanished gradient would pass trivially).
    private func assertZeroSum(_ values: ArraySlice<Float>, _ label: String,
                               file: StaticString = #filePath, line: UInt = #line) {
        let sum = values.reduce(0.0) { $0 + Double($1) }
        let magnitude = values.reduce(0.0) { $0 + abs(Double($1)) }
        XCTAssertGreaterThan(magnitude, 0, "\(label): gradient must not vanish", file: file, line: line)
        XCTAssertEqual(sum, 0, accuracy: 1e-4 * magnitude, "\(label): shared-direction gradient", file: file, line: line)
    }

    func testTrainingGraphGivesTheHeadsSharedDirectionZeroGradient() async throws {
        try requireMetal()
        let architecture = arch(.float32)
        let trainer = try ChessTrainer(momentumCoeff: 0, lrWarmupSteps: 0, arch: architecture)
        let timing = try await trainer.trainStep(batchSize: 32)
        XCTAssertTrue(timing.hasDiagnostics, "the synthetic-data step always computes diagnostics")
        XCTAssertTrue(timing.policyLogitMean.isFinite, "policy mean logit must be measured")
        XCTAssertTrue(timing.valueLogitMean.isFinite, "value mean logit must be measured")

        let exported = try await velocities(after: trainer, architecture: architecture)
        let policyBiasName = architecture.policyHeadStyle == .fcBottleneck
            ? "opt.policy.fc.bias.velocity" : "opt.policy.conv.bias.velocity"
        let policyBias = try tensor(policyBiasName, in: exported)
        assertZeroSum(policyBias[...], "policy final bias")

        let classes = architecture.valueHeadClasses
        let valueBias = try tensor("opt.value.wdl_fc2.bias.velocity", in: exported)
        assertZeroSum(valueBias[...], "value fc2 bias")
        let valueWeight = try tensor("opt.value.wdl_fc2.weight.velocity", in: exported)
        var nonZeroRows = 0
        for r in 0..<architecture.valueHeadHiddenUnits {
            let row = valueWeight[(r * classes)..<((r + 1) * classes)]
            // A hidden unit that never fired has an all-zero row; skip it.
            guard row.contains(where: { $0 != 0 }) else { continue }
            nonZeroRows += 1
            assertZeroSum(row, "value fc2 weight row \(r)")
        }
        XCTAssertGreaterThan(nonZeroRows, 0, "some value fc2 row must receive gradient")
    }

    func testScalarTanhValueHeadIsNotCentered() async throws {
        try requireMetal()
        let architecture = arch(.float32, value: .scalarTanh)
        let trainer = try ChessTrainer(momentumCoeff: 0, lrWarmupSteps: 0, arch: architecture)
        _ = try await trainer.trainStep(batchSize: 32)
        let exported = try await velocities(after: trainer, architecture: architecture)
        // Centering a single logit subtracts it from itself: the loss would
        // no longer depend on the head's bias at all.
        let bias = try tensor("opt.value.scalar_fc2.bias.velocity", in: exported)
        XCTAssertEqual(bias.count, 1)
        XCTAssertNotEqual(bias[0], 0, "the scalar head's bias must receive gradient")
    }
}
