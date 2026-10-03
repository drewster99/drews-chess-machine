//
//  NetworkWeightLoadTests.swift
//  DrewsChessMachineTests
//
//  `ChessNetwork.loadWeights` encodes its assigns into a command buffer it
//  owns and checks that buffer's completion status before it marks the
//  network loaded, so a GPU failure during a load surfaces as an error
//  instead of leaving a zero-filled network marked as holding real weights.
//  The load must still write exactly the values it was given.
//

import Metal
import XCTest
@testable import DrewsChessMachine

final class NetworkWeightLoadTests: XCTestCase {

    private func requireMetal() throws {
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("Metal not available") }
    }

    private static func architecture(compute: ComputeDataType) -> NetworkArchitecture {
        NetworkArchitecture(
            inputEncoding: .basic30, channels: 16, numBlocks: 1, stemConvKernelSize: 3,
            activationFunction: .relu, blockActivationStyle: .pre,
            blockSkipMerge: .cleanAdd, blockUseRezero: true, rezeroAlphaInit: 0.5,
            blockConv1KernelSize: 3, blockConv2KernelSize: 3,
            blockSeStyle: .scaleAndBias, blockSeReductionRatio: 4,
            policyHeadStyle: .intermediateConv, policyPreConvChannels: 16,
            valueHeadStyle: .wdlSoftmax, valueHeadConvChannels: 4, valueHeadHiddenUnits: 16,
            computeDataType: compute
        )
    }

    private static func bits(_ tensors: [[Float]]) -> [[UInt32]] {
        tensors.map { $0.map(\.bitPattern) }
    }

    /// Arbitrary fp32 values (not an initializer's distribution), shaped
    /// like `template`, so every bit of every value is checked.
    private static func arbitraryValues(shapedLike template: [[Float]], seed: UInt64) -> [[Float]] {
        var random = DCMRandom(seed: seed)
        return template.map { tensor in
            tensor.map { _ in Float(Int64(bitPattern: random.next()) % 2_000_001) / 1_000_000 }
        }
    }

    func testLoadWeightsViaEncodeMatchesExport() async throws {
        try requireMetal()
        let arch = Self.architecture(compute: .float32)
        let shapes = try await ChessNetwork(arch: arch, initialization: .seeded(initSeed: 3)).exportWeights()
        let values = Self.arbitraryValues(shapedLike: shapes, seed: 11)

        let awaiting = try ChessNetwork(arch: arch, initialization: .overwrittenByLoad)
        try await awaiting.loadWeights(values)
        let exported = try await awaiting.exportWeights()
        XCTAssertEqual(Self.bits(exported), Self.bits(values))

        // A second load replaces the first, bit for bit.
        let second = Self.arbitraryValues(shapedLike: shapes, seed: 12)
        try await awaiting.loadWeights(second)
        let reexported = try await awaiting.exportWeights()
        XCTAssertEqual(Self.bits(reexported), Self.bits(second))
    }

    /// The status check a load clears its gate behind: only `.completed`
    /// passes; every other status throws `gpuCommandFailed` naming the stage.
    /// (A real GPU failure cannot be induced in a test; this pins the check
    /// the load runs on both of its command buffers.)
    func testOnlyACompletedCommandBufferPasses() throws {
        XCTAssertNoThrow(try ChessNetwork.requireCompleted(status: .completed, error: nil, stage: "weight load"))
        let failing: [MTLCommandBufferStatus] = [.error, .notEnqueued, .enqueued, .committed, .scheduled]
        for status in failing {
            do {
                try ChessNetwork.requireCompleted(status: status, error: nil, stage: ChessNetwork.weightLoadStage)
                XCTFail("status \(status.rawValue) must throw")
            } catch ChessNetworkError.gpuCommandFailed(let stage, let reported, let error) {
                XCTAssertEqual(stage, "weight load")
                XCTAssertEqual(reported, status)
                XCTAssertNil(error)
            }
        }
        let underlying = NSError(domain: "MTLCommandBufferErrorDomain", code: 2,
                                 userInfo: [NSLocalizedDescriptionKey: "Timeout"])
        do {
            try ChessNetwork.requireCompleted(status: .error, error: underlying, stage: ChessNetwork.weightLoadStage)
            XCTFail("an errored buffer must throw")
        } catch ChessNetworkError.gpuCommandFailed(_, _, let error) {
            XCTAssertEqual(error, "Timeout")
        }
    }

    func testReducedPrecisionLoadWritesTheGivenRepresentableValues() async throws {
        try requireMetal()
        let arch = Self.architecture(compute: .bFloat16)
        let source = try await ChessNetwork(arch: arch, initialization: .seeded(initSeed: 4)).exportWeights()
        let awaiting = try ChessNetwork(arch: arch, initialization: .overwrittenByLoad)
        try await awaiting.loadWeights(source)
        let exported = try await awaiting.exportWeights()
        XCTAssertEqual(Self.bits(exported), Self.bits(source))
    }
}
