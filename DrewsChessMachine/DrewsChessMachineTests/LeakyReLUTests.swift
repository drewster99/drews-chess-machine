//
//  LeakyReLUTests.swift
//  DrewsChessMachineTests
//
//  Pins the `leaky_relu` activation (issue #2) and the `--set-activation`
//  derive operation used to A/B it from one fresh net:
//
//  - Values: x for x ≥ 0, `leakyReLUNegativeSlope`·x below, in fp32, bf16 and
//    fp16, including inputs near each format's range limit; never non-finite.
//  - Gradient: 1 above zero, the slope below, in every format.
//  - Wiring: a network whose activation is `leaky_relu` computes a different,
//    finite forward pass than the same weights under ReLU, and trains a
//    finite step.
//  - `--set-activation`: changes the tower-level and every group's activation,
//    copies every tensor bit-exact, and refuses a request that changes nothing.
//

import XCTest
import Metal
import MetalPerformanceShadersGraph
@testable import DrewsChessMachine

final class LeakyReLUTests: XCTestCase {

    private func requireMetal() throws -> MTLDevice {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("Metal not available") }
        return device
    }

    private let slope = ActivationFunction.leakyReLUNegativeSlope

    /// Inputs: a dense sweep over [-8, 8], plus large values well inside each
    /// format's range (fp16's largest finite value is 65504).
    private let inputs: [Float] = stride(from: -8.0, through: 8.0, by: 0.25).map { Float($0) }
        + [-20, 20, -100, 100, -60000, 60000]

    /// Run `leaky_relu` and its gradient on `inputs` in `dtype`. Returns the
    /// inputs as rounded into `dtype` (the values the activation actually
    /// saw), the outputs, and d(sum of outputs)/d(input), all read back as
    /// fp32.
    private func evaluate(
        _ dtype: MPSDataType, device: MTLDevice
    ) throws -> (rounded: [Float], outputs: [Float], gradients: [Float]) {
        let graph = MPSGraph()
        let shape: [NSNumber] = [NSNumber(value: inputs.count)]
        let placeholder = graph.placeholder(shape: shape, dataType: .float32, name: "x")
        let x = dtype == .float32 ? placeholder : graph.cast(placeholder, to: dtype, name: "x_cast")
        let y = ChessNetwork.activation(graph, x, .leakyRelu, name: "leaky")
        let total = graph.reductionSum(with: y, axes: [0], name: "total")
        let gradient = try XCTUnwrap(graph.gradients(of: total, with: [x], name: "grad")[x], "gradient of x")
        let readbacks = [x, y, gradient].map { $0.dataType == .float32 ? $0 : graph.cast($0, to: .float32, name: nil) }

        let data = inputs.withUnsafeBufferPointer { Data(buffer: $0) }
        let feed = MPSGraphTensorData(
            device: MPSGraphDevice(mtlDevice: device), data: data, shape: shape, dataType: .float32)
        let results = graph.run(feeds: [placeholder: feed], targetTensors: readbacks, targetOperations: nil)
        func read(_ tensor: MPSGraphTensor) throws -> [Float] {
            let tensorData = try XCTUnwrap(results[tensor], "readback")
            var values = [Float](repeating: .nan, count: inputs.count)
            try values.withUnsafeMutableBytes { raw in
                tensorData.mpsndarray().readBytes(try XCTUnwrap(raw.baseAddress), strideBytes: nil)
            }
            return values
        }
        return (try read(readbacks[0]), try read(readbacks[1]), try read(readbacks[2]))
    }

    /// Relative tolerance: two rounding steps of the format (the slope and
    /// the product are each rounded).
    private func tolerance(_ dtype: MPSDataType) -> Float {
        switch dtype {
        case .bFloat16: return 1.0 / 128
        case .float16: return 1.0 / 512
        default: return 1e-6
        }
    }

    /// Relative tolerance on the backward pass's negative-side slope. fp16
    /// is looser than its rounding step: measured on the macOS 27 beta,
    /// MPSGraph's fp16 leaky-ReLU gradient uses a slope rounded more coarsely
    /// than fp16's own nearest value to the configured slope, a few percent
    /// low. The forward pass and the bf16 / fp32 gradients are within their
    /// rounding step. Production trains in bf16, so this is recorded rather
    /// than worked around; the fp16 bound is still tight enough to catch a
    /// wrong slope.
    private func negativeSlopeGradientTolerance(_ dtype: MPSDataType) -> Float {
        dtype == .float16 ? 0.03 : tolerance(dtype)
    }

    func testValuesAndGradientsInEveryFormat() throws {
        let device = try requireMetal()
        for dtype in [MPSDataType.float32, .bFloat16, .float16] {
            let (rounded, outputs, gradients) = try evaluate(dtype, device: device)
            let tol = tolerance(dtype)
            for i in inputs.indices {
                let r = rounded[i]
                let label = "\(dtype.rawValue) x=\(inputs[i])"
                XCTAssertTrue(outputs[i].isFinite, "\(label): output must be finite")
                XCTAssertTrue(gradients[i].isFinite, "\(label): gradient must be finite")
                let expected = r >= 0 ? r : Float(slope) * r
                XCTAssertEqual(outputs[i], expected, accuracy: max(abs(expected) * tol, 1e-7), label)
                if r > 0 {
                    XCTAssertEqual(gradients[i], 1, accuracy: tol, "\(label): gradient above zero")
                } else if r < 0 {
                    XCTAssertEqual(gradients[i], Float(slope), accuracy: Float(slope) * negativeSlopeGradientTolerance(dtype),
                                   "\(label): gradient below zero")
                }
            }
            // Reference points from issue #2.
            if let minusOne = inputs.firstIndex(of: -1) {
                XCTAssertEqual(outputs[minusOne], -Float(slope), accuracy: Float(slope) * tol, "\(dtype.rawValue) leaky(-1)")
            } else {
                XCTFail("the sweep must contain -1")
            }
        }
    }

    func testLeakyReLUChangesTheForwardPassAndTrains() async throws {
        _ = try requireMetal()
        var reluArch = NetworkArchitecture.current
        try reluArch.setActivationAtEveryExistingSite(.relu)
        for index in reluArch.blockGroups.indices { reluArch.blockGroups[index].activationFunction = .relu }
        var leakyArch = reluArch
        try leakyArch.setActivationAtEveryExistingSite(.leakyRelu)
        for index in leakyArch.blockGroups.indices { leakyArch.blockGroups[index].activationFunction = .leakyRelu }

        let reluNet = try ChessMPSNetwork(.randomWeights(initSeed: 1), arch: reluArch)
        let leakyNet = try ChessMPSNetwork(.randomWeights(initSeed: 2), arch: leakyArch)
        try await leakyNet.network.loadWeights(try await reluNet.network.exportWeights())
        let board = BoardEncoder.encode(.starting, encoding: reluNet.inputEncoding)
        func policy(_ net: ChessMPSNetwork) async throws -> [Float] {
            let box = SyncBox<[Float]>([])
            try await net.evaluate(board: board) { p, _ in box.value = Array(p) }
            return box.value
        }
        let reluPolicy = try await policy(reluNet)
        let leakyPolicy = try await policy(leakyNet)
        XCTAssertEqual(leakyPolicy.count, ChessNetwork.policySize)
        XCTAssertTrue(leakyPolicy.allSatisfy(\.isFinite), "leaky_relu logits must be finite")
        XCTAssertNotEqual(reluPolicy, leakyPolicy, "same weights: leaky_relu must change the forward pass")

        let trainer = try ChessTrainer(dropoutStream: DCMRandom(seed: 1), lrWarmupSteps: 0, arch: leakyArch, initialization: .seeded(initSeed: 1))
        let timing = try await trainer.trainStep(batchSize: 8)
        XCTAssertTrue(timing.policyLoss.isFinite, "leaky_relu policy loss must be finite")
        XCTAssertTrue(timing.valueLoss.isFinite, "leaky_relu value loss must be finite")
    }

    // MARK: - --set-activation

    private func encodedModel(_ arch: NetworkArchitecture) throws -> Data {
        let weights = arch.weightTensorPlan().enumerated().map { tensorIndex, spec in
            (0..<spec.elementCount).map { Float(tensorIndex * 7 + $0 % 13) * 0.125 + 0.0625 }
        }
        let meta = ModelCheckpointMetadata(creator: "test", trainingStep: nil, parentModelID: "", notes: "fixture")
        return try SafetensorsModelIO.encode(
            modelID: "20261001-1-SRCE", createdAtUnix: 1_790_000_000, metadata: meta, weights: weights,
            architecture: arch, includesVelocity: false,
            lineage: try LineageRecord.forTests(trainerCompletedSteps: meta.trainerSchedule.map(\.completedTrainSteps), corpus: nil))
    }

    private func derive(_ source: Data, to value: ActivationFunction) throws -> ModelDerivation.Result {
        try ModelDerivation.derive(
            sourceData: source, sourceName: "source.safetensors",
            operations: [SetActivationDeriveOperation(value: value)],
            newModelID: "20261001-2-DRV1", createdAtUnix: 1_790_000_100, build: "test", invocationArguments: ["test"])
    }

    func testSetActivationChangesEverySiteAndCopiesEveryTensor() throws {
        var source = NetworkArchitecture.current
        try source.setMainActivationEverywhere(.relu)
        for index in source.blockGroups.indices { source.blockGroups[index].activationFunction = .relu }
        let sourceData = try encodedModel(source)

        let result = try derive(sourceData, to: .leakyRelu)
        for site in ArchitectureActivationSite.allCases {
            XCTAssertEqual(result.targetArchitecture.activation(at: site), result.targetArchitecture.hasActivationSite(site) ? .leakyRelu : .doesNotApply, "\(site)")
        }
        XCTAssertTrue(result.targetArchitecture.blockGroups.allSatisfy { $0.activationFunction == .leakyRelu })
        var expected = source
        try expected.setMainActivationEverywhere(.leakyRelu)
        for index in expected.blockGroups.indices { expected.blockGroups[index].activationFunction = .leakyRelu }
        XCTAssertEqual(result.targetArchitecture, expected, "only the activation fields may change")
        XCTAssertTrue(result.rewrites.isEmpty, "activations have no parameters")

        let (sourceTensors, _) = try SafetensorsFile.decode(sourceData)
        let (derivedTensors, derivedMetadata) = try SafetensorsFile.decode(result.data)
        XCTAssertEqual(sourceTensors.map(\.name), derivedTensors.map(\.name))
        for (before, after) in zip(sourceTensors, derivedTensors) {
            XCTAssertEqual(before.shape, after.shape, before.name)
            XCTAssertEqual(before.data.map(\.bitPattern), after.data.map(\.bitPattern), "\(before.name) must be bit-exact")
        }
        XCTAssertEqual(derivedMetadata[SafetensorsModelIO.Key.parentModelID], "20261001-1-SRCE")
        XCTAssertEqual(result.record.operations.map(\.operation), ["set-activation"])
        XCTAssertEqual(result.record.operations.first?.arguments, ["value": "leaky_relu"])
    }

    func testSetActivationRefusesANoOp() throws {
        var source = NetworkArchitecture.current
        try source.setActivationAtEveryExistingSite(.leakyRelu)
        for index in source.blockGroups.indices { source.blockGroups[index].activationFunction = .leakyRelu }
        XCTAssertThrowsError(try derive(try encodedModel(source), to: .leakyRelu)) { error in
            guard case .operationNotApplicable? = error as? ModelDerivation.DeriveError else {
                return XCTFail("expected operationNotApplicable, got \(error)")
            }
        }
    }

    func testSetActivationRejectsAnUnknownValue() {
        XCTAssertThrowsError(try SetActivationDeriveOperation.kind.make("leaky", nil))
        XCTAssertNoThrow(try SetActivationDeriveOperation.kind.make("leaky_relu", nil))
    }
}
