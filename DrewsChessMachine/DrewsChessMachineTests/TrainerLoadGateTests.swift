//
//  TrainerLoadGateTests.swift
//  DrewsChessMachineTests
//
//  A trainer built `overwrittenByLoad` holds zero-filled variables until a
//  load gives it real weights. Every reader of its state must refuse until
//  then, the way `trainStep` and the network's own export already do: the
//  live layer-health read, and the fp32 master and velocity readbacks behind
//  `exportTrainerWeights` and `exportVelocitySnapshot`. Without the gate a
//  reduced-precision trainer that never received its load exports zero
//  masters and zero velocity, which a session save would then write to disk
//  as if they were trained state.
//

import Metal
import XCTest
@testable import DrewsChessMachine

final class TrainerLoadGateTests: XCTestCase {

    private func requireMetal() throws {
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("Metal not available") }
    }

    /// A small tower, so each trainer builds quickly.
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

    /// Base network state (trainables + BN running statistics) from a
    /// seeded network of `arch`: what a fork loads into a trainer.
    private static func seededBaseWeights(_ arch: NetworkArchitecture) async throws -> [[Float]] {
        try await ChessNetwork(arch: arch, initialization: .seeded(initSeed: 5)).exportWeights()
    }

    private static func trainerAwaitingLoad(_ arch: NetworkArchitecture) throws -> ChessTrainer {
        try ChessTrainer(dropoutStream: DCMRandom(seed: 1), arch: arch, initialization: .overwrittenByLoad)
    }

    func testTrainerBuiltForLoadRefusesLiveLayerHealthUntilLoaded() async throws {
        try requireMetal()
        let arch = Self.architecture(compute: .float32)
        let trainer = try Self.trainerAwaitingLoad(arch)
        do {
            _ = try await trainer.readLayerHealthLiveState()
            XCTFail("a trainer built for loaded weights must not report layer health before the load")
        } catch ChessNetworkError.weightsNotLoaded(let operation) {
            XCTAssertEqual(operation, "readLayerHealthLiveState")
        }
        try await trainer.loadBaseWeightsResetVelocity(try await Self.seededBaseWeights(arch))
        let state = try await trainer.readLayerHealthLiveState()
        XCTAssertFalse(state.tensors.isEmpty)
    }

    func testReducedPrecisionTrainerBuiltForLoadRefusesTrainerExportUntilLoaded() async throws {
        try requireMetal()
        let arch = Self.architecture(compute: .bFloat16)
        let trainer = try Self.trainerAwaitingLoad(arch)
        do {
            _ = try await trainer.exportTrainerWeights()
            XCTFail("a bf16 trainer built for loaded weights must not export masters and velocity before the load")
        } catch ChessNetworkError.weightsNotLoaded(let operation) {
            XCTAssertEqual(operation, "readMasterValues")
        }
        do {
            _ = try await trainer.readMasterValues()
            XCTFail("a bf16 trainer built for loaded weights must not read its masters before the load")
        } catch ChessNetworkError.weightsNotLoaded(let operation) {
            XCTAssertEqual(operation, "readMasterValues")
        }
        do {
            _ = try await trainer.exportVelocitySnapshot()
            XCTFail("a trainer built for loaded weights must not export velocity before the load")
        } catch ChessNetworkError.weightsNotLoaded(let operation) {
            XCTAssertEqual(operation, "readVelocityValues")
        }
        do {
            _ = try await trainer.readLayerHealthLiveState()
            XCTFail("a bf16 trainer built for loaded weights must not report layer health before the load")
        } catch ChessNetworkError.weightsNotLoaded(let operation) {
            XCTAssertEqual(operation, "readLayerHealthLiveState")
        }

        let base = try await Self.seededBaseWeights(arch)
        try await trainer.loadBaseWeightsResetVelocity(base)
        let exported = try await trainer.exportTrainerWeights()
        let velocity = try await trainer.exportVelocitySnapshot()
        XCTAssertEqual(exported.count, base.count + velocity.count)
        XCTAssertEqual(Array(exported.prefix(base.count)).map { $0.map(\.bitPattern) }, base.map { $0.map(\.bitPattern) },
                       "after the load, the exported masters are the loaded weights")
        _ = try await trainer.readLayerHealthLiveState()
    }

    func testFloat32TrainerBuiltForLoadRefusesVelocityExportUntilLoaded() async throws {
        try requireMetal()
        let arch = Self.architecture(compute: .float32)
        let trainer = try Self.trainerAwaitingLoad(arch)
        do {
            _ = try await trainer.exportVelocitySnapshot()
            XCTFail("a trainer built for loaded weights must not export velocity before the load")
        } catch ChessNetworkError.weightsNotLoaded(let operation) {
            XCTAssertEqual(operation, "readVelocityValues")
        }
        try await trainer.loadBaseWeightsResetVelocity(try await Self.seededBaseWeights(arch))
        let velocity = try await trainer.exportVelocitySnapshot()
        XCTAssertTrue(velocity.allSatisfy { $0.allSatisfy { $0 == 0 } }, "a fork starts from zero velocity")
    }
}
