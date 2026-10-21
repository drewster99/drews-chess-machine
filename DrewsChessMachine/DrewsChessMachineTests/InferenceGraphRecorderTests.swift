import XCTest
import Metal
import MetalPerformanceShadersGraph
@testable import DrewsChessMachine

/// The inference-graph recording (`ForwardGraphOps` / `InferenceGraphRecorder`
/// / `InferenceGraphExport`): complete for every architecture option, without
/// effect on the network built, and refused where it would be wrong.
final class InferenceGraphRecorderTests: XCTestCase {

    private func requireMetal() throws {
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("Metal not available") }
    }

    /// Protects: every architecture option's forward pass is recorded completely
    /// — each plan tensor read, every output produced by a recorded op.
    /// Fails if: a builder branch builds a forward op on the raw MPSGraph (the
    /// recorder sees an unrecorded operand), drops a weight, or the weight table
    /// stops following the plan.
    func testEveryArchitectureVariantRecordsACompleteGraph() throws {
        try requireMetal()
        for variant in try InferenceGraphTestArchitectures.all() {
            let graph: InferenceGraph
            do {
                graph = try InferenceGraphExport.record(architecture: variant.architecture)
            } catch {
                XCTFail("\(variant.name): \(error.localizedDescription)")
                continue
            }
            let plan = variant.architecture.weightTensorPlan()
            XCTAssertEqual(graph.weights.map(\.name), plan.map(\.name), variant.name)
            XCTAssertEqual(graph.weights.map(\.elementCount), plan.map(\.elementCount), variant.name)
            XCTAssertEqual(Set(graph.outputs.map(\.name)), Set(InferenceGraphOutputName.allCases), variant.name)
            XCTAssertEqual(graph.input.shape, [-1, variant.architecture.inputPlanes, 8, 8], variant.name)
        }
    }

    /// Protects: attaching a recorder leaves the built graph unchanged, so a
    /// recording describes the network production builds without one.
    /// Fails if: the recorder branch of a `ForwardGraphOps` wrapper builds an
    /// extra op or returns a different tensor than the one it built. (A wrapper
    /// that calls the wrong MPSGraph op changes both networks alike; the Core ML
    /// parity test catches that.)
    func testTheRecorderDoesNotChangeTheBuiltGraph() async throws {
        try requireMetal()
        let states = Array(LichessProbeData.largeSet.prefix(16)).map(\.state)
        for variant in try InferenceGraphTestArchitectures.all() {
            let arch = variant.architecture
            let plain = try ChessNetwork(arch: arch, initialization: .seeded(initSeed: 21))
            let recorded = try ChessNetwork(arch: arch, initialization: .seeded(initSeed: 21),
                                            graphRecorder: InferenceGraphRecorder())
            XCTAssertNotNil(recorded.recordedInferenceGraph, variant.name)
            var boards: [Float] = []
            for state in states { boards += BoardEncoder.encode(state, encoding: arch.inputEncoding) }
            let a = try await CoreMLParity.networkOutputs(network: plain, boards: boards, count: states.count)
            let b = try await CoreMLParity.networkOutputs(network: recorded, boards: boards, count: states.count)
            XCTAssertEqual(a.policy, b.policy, variant.name)
            XCTAssertEqual(a.value, b.value, variant.name)
            XCTAssertEqual(a.valueProbabilities, b.valueProbabilities, variant.name)
        }
    }

    /// Protects: only an inference-mode graph can be recorded.
    /// Fails if: the training-mode refusal is removed, so a recording would
    /// describe batch-statistics BN instead of the running-stat inference pass.
    func testRecordingATrainingModeNetworkIsRefused() throws {
        try requireMetal()
        let arch = try XCTUnwrap(InferenceGraphTestArchitectures.all().first).architecture
        XCTAssertThrowsError(try ChessNetwork(arch: arch, bnMode: .training, initialization: .overwrittenByLoad,
                                              graphRecorder: InferenceGraphRecorder())) { error in
            guard case ChessNetworkError.inferenceGraphRecordingOfTrainingGraph = error else {
                return XCTFail("unexpected error \(error)")
            }
        }
    }

    /// Protects: a forward op built on the raw MPSGraph, bypassing
    /// `ForwardGraphOps`, fails the recording instead of leaving a hole in it.
    /// Fails if: `operand` stops refusing a tensor no recorded op produced.
    func testAnOpBuiltOutsideForwardGraphOpsFailsTheRecording() throws {
        let graph = MPSGraph()
        let recorder = InferenceGraphRecorder()
        let ops = ForwardGraphOps(graph: graph, recorder: recorder)
        let input = ops.placeholder(shape: [-1, 4], dataType: .float32, name: "in")
        let bypassed = graph.reLU(with: input, name: "raw_relu")
        let output = ops.sigmoid(with: bypassed, name: "after_raw")
        XCTAssertThrowsError(try recorder.finish(persistentVariables: [], plan: [],
                                                 outputs: [(.policy, output)])) { error in
            XCTAssertEqual(error as? InferenceGraphRecorder.RecordingError,
                           .unrecordedTensor(consumer: "after_raw", producer: "raw_relu"))
        }
    }

    /// Protects: a plan tensor the forward pass never reads fails the recording.
    /// Fails if: `finish` stops checking that every weight is read — a dropped
    /// branch (e.g. an SE block not applied) would then export silently.
    func testAWeightTheForwardPassNeverReadsFailsTheRecording() throws {
        let graph = MPSGraph()
        let recorder = InferenceGraphRecorder()
        let ops = ForwardGraphOps(graph: graph, recorder: recorder)
        let input = ops.placeholder(shape: [-1, 4], dataType: .float32, name: "in")
        let unused = ops.variable(with: Data(count: 16), shape: [4], dataType: .float32, name: "unused")
        let output = ops.sigmoid(with: input, name: "out")
        let plan = [WeightTensorSpec(name: "head.bias", shape: [4], kind: .bias)]
        XCTAssertThrowsError(try recorder.finish(persistentVariables: [unused], plan: plan,
                                                 outputs: [(.policy, output)])) { error in
            XCTAssertEqual(error as? InferenceGraphRecorder.RecordingError, .weightUnused(name: "head.bias"))
        }
    }

    /// Protects: each recorded op carries the data type the MPSGraph computes it
    /// in, which is how the exporter keeps the policy tail in fp32.
    /// Fails if: the recorded data type stops following the built tensor's —
    /// e.g. a bf16 model's fp32_from_pre_bn tail recorded as bf16 (the exporter
    /// would then run it in fp16), or the tower recorded as fp32.
    func testRecordedDataTypesFollowTheBuiltPolicyTail() throws {
        try requireMetal()
        let fp32 = try XCTUnwrap(InferenceGraphTestArchitectures.all().first).architecture
        for (tail, preBNType) in [(PolicyTailPrecisionSetting.float32FromPreBatchNorm, GraphDataType.float32),
                                  (.mixedFinalProjection, .bfloat16)] {
            let arch = try fp32.withComputeDataType(.bFloat16, tail: tail)
            let graph = try InferenceGraphExport.record(architecture: arch)
            let byLabel = Dictionary(graph.operations.compactMap { op in op.label.map { ($0, op) } },
                                     uniquingKeysWith: { first, _ in first })
            XCTAssertEqual(byLabel["stem_conv"]?.dataType, .bfloat16, "\(tail)")
            XCTAssertEqual(byLabel["policy_pre_bn"]?.dataType, preBNType, "\(tail)")
            let policyID = try XCTUnwrap(graph.outputs.first { $0.name == .policy }?.id)
            XCTAssertEqual(graph.operations.first { $0.id == policyID }?.dataType, .float32, "\(tail)")
        }
    }

    /// Protects: graph.json reads back as the recording that was written.
    /// Fails if: an operation kind or field stops round-tripping through its
    /// JSON form (a renamed key, a lossy number).
    func testTheGraphFileRoundTrips() throws {
        try requireMetal()
        for variant in try InferenceGraphTestArchitectures.all() {
            let graph = try InferenceGraphExport.record(architecture: variant.architecture)
            let decoded = try JSONDecoder().decode(InferenceGraph.self, from: InferenceGraphExport.graphFileData(graph))
            XCTAssertEqual(decoded, graph, variant.name)
        }
    }

    /// Protects: weights.bin holds each plan tensor's float32 values at the
    /// offset graph.json gives, and refuses weights that do not fit the graph.
    /// Fails if: tensors are written out of order or at the wrong offset, or a
    /// wrong-length tensor or a missing tensor is written instead of refused
    /// (with the error that names it).
    func testTheWeightsFileLaysOutEveryTensorAtItsOffsetAndRefusesMismatches() throws {
        try requireMetal()
        let arch = try XCTUnwrap(InferenceGraphTestArchitectures.all().first).architecture
        let graph = try InferenceGraphExport.record(architecture: arch)
        let weights = graph.weights.enumerated().map { index, spec in
            (0..<spec.elementCount).map { Float(index) + Float($0) / 1024 }
        }
        let data = try InferenceGraphExport.weightsFileData(graph: graph, weights: weights)
        XCTAssertEqual(data.count, graph.weights.reduce(0) { $0 + $1.elementCount } * 4)
        for (spec, values) in zip(graph.weights, weights) {
            let stored = data.subdata(in: spec.byteOffset..<(spec.byteOffset + spec.elementCount * 4))
                .withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
            XCTAssertEqual(stored, values, spec.name)
        }

        XCTAssertThrowsError(try InferenceGraphExport.weightsFileData(graph: graph, weights: Array(weights.dropLast()))) { error in
            XCTAssertEqual(error as? InferenceGraphExport.ExportError,
                           .weightCountMismatch(have: weights.count - 1, want: graph.weights.count))
        }
        var short = weights
        short[0].removeLast()
        XCTAssertThrowsError(try InferenceGraphExport.weightsFileData(graph: graph, weights: short)) { error in
            XCTAssertEqual(error as? InferenceGraphExport.ExportError,
                           .weightSizeMismatch(name: graph.weights[0].name, have: short[0].count,
                                               want: graph.weights[0].elementCount))
        }
    }

    /// Protects: weights.bin holds the values a reduced-precision network's
    /// variables hold — rounded to bf16 as `loadWeights` rounds them — so a
    /// Core ML export computes with the same weights as the MPSGraph.
    /// Fails if: the exporter writes the file's unrounded float32 values, or
    /// stops recording the variable's data type.
    func testTheWeightsFileHoldsTheValuesAReducedPrecisionVariableHolds() throws {
        try requireMetal()
        let fp32 = try XCTUnwrap(InferenceGraphTestArchitectures.all().first).architecture
        let arch = try fp32.withComputeDataType(.bFloat16, tail: .mixedFinalProjection)
        let graph = try InferenceGraphExport.record(architecture: arch)
        XCTAssertTrue(graph.weights.allSatisfy { $0.dataType == .bfloat16 })
        // 1 + 2^-10 is between two bf16 values (8 significant bits): it rounds to 1.
        let weights = graph.weights.map { [Float](repeating: 1 + 1.0 / 1024, count: $0.elementCount) }
        let data = try InferenceGraphExport.weightsFileData(graph: graph, weights: weights)
        let stored = data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        XCTAssertTrue(stored.allSatisfy { $0 == 1 })
    }
}
