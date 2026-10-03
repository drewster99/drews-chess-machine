//
//  WeightInitializationTests.swift
//  DrewsChessMachineTests
//
//  Seeded per-tensor initialization in real network builds (determinism plan
//  B1, phase P5): same seed → bit-identical weights, shared tensors match
//  across architectures, the builder's names are the plan's names, and a
//  network built to receive loaded weights refuses use until it has them.
//

import Metal
import MetalPerformanceShaders
import XCTest
@testable import DrewsChessMachine

final class WeightInitializationTests: XCTestCase {

    private func requireMetal() throws {
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("Metal not available") }
    }

    /// A small fp32 tower: pre-activation, scale-and-bias SE with zero β,
    /// a width transition (16 → 24, so a skip projection), the intermediate
    /// policy head and the W/D/L value head.
    private static func smallArchitecture(blocksInSecondGroup: Int = 1) -> NetworkArchitecture {
        var arch = NetworkArchitecture(
            inputEncoding: .basic30, channels: 16, numBlocks: 1, stemConvKernelSize: 3,
            activationFunction: .relu, blockActivationStyle: .pre,
            blockSkipMerge: .cleanAdd, blockUseRezero: true, rezeroAlphaInit: 0.5,
            blockConv1KernelSize: 3, blockConv2KernelSize: 3,
            blockSeStyle: .scaleAndBias, blockSeReductionRatio: 4,
            policyHeadStyle: .intermediateConv, policyPreConvChannels: 16,
            valueHeadStyle: .wdlSoftmax, valueHeadConvChannels: 4, valueHeadHiddenUnits: 16,
            computeDataType: .float32
        )
        arch.blockGroups[0].seBetaInit = .zero
        var second = arch.blockGroups[0]
        second.channels = 24
        second.count = blocksInSecondGroup
        second.seBetaInit = .glorot
        arch.blockGroups.append(second)
        return arch
    }

    private static func bits(_ tensors: [[Float]]) -> [[UInt32]] {
        tensors.map { $0.map(\.bitPattern) }
    }

    func testSameSeedBuildsBitIdenticalNetworks() async throws {
        try requireMetal()
        let arch = Self.smallArchitecture()
        let first = try ChessNetwork(arch: arch, initialization: .seeded(initSeed: 42))
        let second = try ChessNetwork(arch: arch, initialization: .seeded(initSeed: 42))
        let other = try ChessNetwork(arch: arch, initialization: .seeded(initSeed: 43))
        let firstWeights = try await first.exportWeights()
        let secondWeights = try await second.exportWeights()
        let otherWeights = try await other.exportWeights()
        XCTAssertEqual(Self.bits(firstWeights), Self.bits(secondWeights))
        XCTAssertNotEqual(Self.bits(firstWeights), Self.bits(otherWeights))
        XCTAssertEqual(first.initialization.initSeed, 42)
    }

    /// Every conv and FC weight the builder creates equals the scheme's value
    /// for that plan name — so the builder's names, shapes and roles are the
    /// plan's, which is what the per-tensor seeds are keyed on.
    func testBuilderValuesAreTheSchemeValuesByPlanName() async throws {
        try requireMetal()
        let arch = Self.smallArchitecture()
        let network = try ChessNetwork(arch: arch, initialization: .seeded(initSeed: 1234))
        let exported = try await network.exportWeights()
        let plan = arch.weightTensorPlan()
        XCTAssertEqual(exported.count, plan.count)
        var blockGroupOfBlock: [Int] = []
        for (groupIndex, group) in arch.blockGroups.enumerated() {
            blockGroupOfBlock.append(contentsOf: Array(repeating: groupIndex, count: group.count))
        }
        var checked = 0
        for (index, spec) in plan.enumerated() where spec.kind == .conv || spec.kind == .linear {
            let expected: [Float]
            if spec.name.hasSuffix(".fc2.weight"), spec.name.contains(".se_") {
                let blockIndex = try XCTUnwrap(Int(spec.name.split(separator: ".")[1]))
                expected = try WeightInitScheme.seFC2NativeValues(
                    initSeed: 1234, spec: spec, group: arch.blockGroups[blockGroupOfBlock[blockIndex]])
            } else {
                expected = try WeightInitScheme.nativeValues(initSeed: 1234, spec: spec, distribution: .heNormal)
            }
            XCTAssertEqual(exported[index].map(\.bitPattern), expected.map(\.bitPattern), spec.name)
            checked += 1
        }
        XCTAssertGreaterThan(checked, 10)
        // The zero-β group's SE FC2 β columns are exactly zero.
        let zeroBetaIndex = try XCTUnwrap(plan.firstIndex { $0.name == "blocks.0.se_scalebias.fc2.weight" })
        for range in SEScaleAndBiasBetaHalf.nativeWeightRanges(reducedChannels: 4, channels: 16) {
            XCTAssertTrue(range.allSatisfy { exported[zeroBetaIndex][$0] == 0 })
        }
    }

    /// Two architectures that share tensors (same names and shapes) get the
    /// same values for them under the same seed; adding blocks changes
    /// nothing that already existed.
    func testSharedTensorsMatchAcrossArchitectures() async throws {
        try requireMetal()
        let shallow = Self.smallArchitecture(blocksInSecondGroup: 1)
        let deep = Self.smallArchitecture(blocksInSecondGroup: 2)
        let shallowWeights = try await ChessNetwork(arch: shallow, initialization: .seeded(initSeed: 7)).exportWeights()
        let deepWeights = try await ChessNetwork(arch: deep, initialization: .seeded(initSeed: 7)).exportWeights()
        let shallowPlan = shallow.weightTensorPlan()
        var deepByName: [String: (spec: WeightTensorSpec, values: [Float])] = [:]
        for (spec, values) in zip(deep.weightTensorPlan(), deepWeights) { deepByName[spec.name] = (spec, values) }
        var shared = 0
        for (spec, values) in zip(shallowPlan, shallowWeights) where spec.kind == .conv || spec.kind == .linear {
            let deepEntry = try XCTUnwrap(deepByName[spec.name], "deep tower lacks \(spec.name)")
            XCTAssertEqual(deepEntry.spec.shape, spec.shape, spec.name)
            XCTAssertEqual(deepEntry.values.map(\.bitPattern), values.map(\.bitPattern), spec.name)
            shared += 1
        }
        XCTAssertGreaterThan(shared, 10)
    }

    /// A bf16 build holds the round-to-nearest-even of the fp32 draw.
    func testBFloat16BuildIsTheRoundedFP32Draw() async throws {
        try requireMetal()
        var bf16 = Self.smallArchitecture()
        bf16.computeDataType = .bFloat16
        let fp32 = Self.smallArchitecture()
        let bf16Weights = try await ChessNetwork(arch: bf16, initialization: .seeded(initSeed: 5)).exportWeights()
        let fp32Weights = try await ChessNetwork(arch: fp32, initialization: .seeded(initSeed: 5)).exportWeights()
        for (index, spec) in bf16.weightTensorPlan().enumerated() where spec.kind == .conv || spec.kind == .linear {
            let rounded = fp32Weights[index].map {
                Float(bitPattern: UInt32(ChessNetwork.float32ToBFloat16Bits($0)) << 16)
            }
            XCTAssertEqual(bf16Weights[index].map(\.bitPattern), rounded.map(\.bitPattern), spec.name)
        }
    }

    /// Every built-in preset builds seeded: the builder names every conv and
    /// FC weight by its plan name and covers the whole plan.
    func testEveryPresetBuildsSeeded() throws {
        try requireMetal()
        for preset in NetworkArchitecture.Preset.allCases {
            XCTAssertNoThrow(
                try ChessNetwork(arch: NetworkArchitecture.preset(preset), initialization: .seeded(initSeed: 11)),
                preset.rawValue)
        }
    }

    func testOverwrittenByLoadRefusesUseUntilLoaded() async throws {
        try requireMetal()
        let arch = Self.smallArchitecture()
        let source = try await ChessNetwork(arch: arch, initialization: .seeded(initSeed: 9)).exportWeights()
        let awaiting = try ChessNetwork(arch: arch, initialization: .overwrittenByLoad)
        XCTAssertNil(awaiting.initialization.initSeed)
        do {
            _ = try await awaiting.exportWeights()
            XCTFail("export before load must throw")
        } catch ChessNetworkError.weightsNotLoaded(let operation) {
            XCTAssertEqual(operation, "exportWeights")
        }
        let board = BoardEncoder.encode(.starting, encoding: arch.inputEncoding)
        do {
            try await awaiting.evaluate(board: board) { _, _ in }
            XCTFail("evaluate before load must throw")
        } catch ChessNetworkError.weightsNotLoaded(let operation) {
            XCTAssertEqual(operation, "evaluate")
        }
        try await awaiting.loadWeights(source)
        let reloaded = try await awaiting.exportWeights()
        XCTAssertEqual(Self.bits(reloaded), Self.bits(source))
        try await awaiting.evaluate(board: board) { _, _ in }
    }

    func testWeightsToBeLoadedMPSNetworkRefusesUntilLoaded() async throws {
        try requireMetal()
        let arch = Self.smallArchitecture()
        let fresh = try ChessMPSNetwork(.seededRandomWeights(initSeed: 21), arch: arch)
        let weights = try await fresh.network.exportWeights()
        let mirror = try ChessMPSNetwork(.weightsToBeLoaded, arch: arch)
        let board = BoardEncoder.encode(.starting, encoding: arch.inputEncoding)
        do {
            try await mirror.evaluate(board: board) { _, _ in }
            XCTFail("evaluate before load must throw")
        } catch ChessNetworkError.weightsNotLoaded {
        }
        try await mirror.network.loadWeights(weights)
        let mirrored = try await mirror.network.exportWeights()
        XCTAssertEqual(Self.bits(mirrored), Self.bits(weights))
    }

    /// A seeded mint is reproducible end to end: identical trainables, and
    /// BN running statistics calibrated on the same seeded warmup game.
    func testSeededMintIsReproducible() async throws {
        try requireMetal()
        let arch = Self.smallArchitecture()
        let first = try await ChessMPSNetwork(.seededRandomWeights(initSeed: 77), arch: arch).network.exportWeights()
        let second = try await ChessMPSNetwork(.seededRandomWeights(initSeed: 77), arch: arch).network.exportWeights()
        let trainableCount = arch.trainableTensorPlan().count
        XCTAssertEqual(Self.bits(Array(first.prefix(trainableCount))), Self.bits(Array(second.prefix(trainableCount))))
        for (a, b) in zip(first.suffix(from: trainableCount), second.suffix(from: trainableCount)) {
            for (x, y) in zip(a, b) {
                XCTAssertEqual(x, y, accuracy: max(abs(x), 1) * 1e-5)
            }
        }
    }

    func testWarmupBatchFollowsTheInitSeed() {
        let encoding = InputEncoding.basic30
        XCTAssertEqual(ChessMPSNetwork.warmupBatch(encoding: encoding, initSeed: 5),
                       ChessMPSNetwork.warmupBatch(encoding: encoding, initSeed: 5))
        XCTAssertNotEqual(ChessMPSNetwork.warmupBatch(encoding: encoding, initSeed: 5),
                          ChessMPSNetwork.warmupBatch(encoding: encoding, initSeed: 6))
    }

    func testTrainerBuiltForLoadRefusesTrainingUntilLoaded() async throws {
        try requireMetal()
        let arch = Self.smallArchitecture()
        let trainer = try ChessTrainer(arch: arch, initialization: .overwrittenByLoad)
        do {
            _ = try await trainer.trainStep(batchSize: 4)
            XCTFail("a trainer built for loaded weights must not train before the load")
        } catch ChessNetworkError.weightsNotLoaded(let operation) {
            XCTAssertEqual(operation, "trainStep")
        }
    }

    // MARK: TensorInitializer checks

    func testInitializerRefusesNamesAndShapesOutsideThePlan() throws {
        let arch = Self.smallArchitecture()
        let initializer = TensorInitializer(initialization: .seeded(initSeed: 1), architecture: arch)
        XCTAssertThrowsError(try initializer.weightData("blocks.9.conv1.weight", nativeShape: [16, 16, 3, 3],
                                                         distribution: .heNormal, dataType: .float32)) { error in
            XCTAssertEqual(error as? WeightInitError, .tensorNotInPlan(name: "blocks.9.conv1.weight"))
        }
        XCTAssertThrowsError(try initializer.weightData("stem.conv.weight", nativeShape: [16, 30, 5, 5],
                                                         distribution: .heNormal, dataType: .float32)) { error in
            XCTAssertEqual(error as? WeightInitError,
                           .shapeDisagreesWithPlan(name: "stem.conv.weight", builder: [16, 30, 5, 5], plan: [16, 30, 3, 3]))
        }
        _ = try initializer.weightData("stem.conv.weight", nativeShape: [16, 30, 3, 3], distribution: .heNormal, dataType: .float32)
        XCTAssertThrowsError(try initializer.weightData("stem.conv.weight", nativeShape: [16, 30, 3, 3],
                                                         distribution: .heNormal, dataType: .float32)) { error in
            XCTAssertEqual(error as? WeightInitError, .initializedTwice(name: "stem.conv.weight"))
        }
        XCTAssertThrowsError(try initializer.verifyEveryRandomTensorInitialized()) { error in
            guard case .planTensorsNotInitialized(let names)? = error as? WeightInitError else {
                return XCTFail("expected planTensorsNotInitialized, got \(error)")
            }
            XCTAssertTrue(names.contains("blocks.0.conv1.weight"))
            XCTAssertFalse(names.contains("stem.conv.weight"))
        }
    }

    /// `overwrittenByLoad` hands out zeros and does no random work.
    func testOverwrittenByLoadInitializerReturnsZeros() throws {
        let arch = Self.smallArchitecture()
        let initializer = TensorInitializer(initialization: .overwrittenByLoad, architecture: arch)
        let data = try initializer.weightData("stem.conv.weight", nativeShape: [16, 30, 3, 3],
                                              distribution: .heNormal, dataType: .float32)
        XCTAssertEqual(data, ChessNetwork.zerosData(count: 16 * 30 * 9, dataType: .float32))
    }
}
