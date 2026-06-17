//
//  FP16ComputePathTests.swift
//  DrewsChessMachineTests
//
//  fp16 (`ComputeDataType.float16`) end-to-end COMPUTE coverage, complementing
//  the pure conversion/persistence guards in `FP16ConversionTests`.
//
//  The in-graph casts and the fp32-master / fp16-working / working-sync
//  gradient machinery are dtype-generic — the bf16 suite
//  (`ConvKernelExecutionPathNumericsTests`, `MacOS27NaNIsolationTests`) runs
//  that exact code with `.bFloat16`. These tests run the SAME real production
//  paths with `.float16` so the fp16-specific numerics are exercised, not just
//  inferred from the bf16 arm:
//   - a real fp16 network forward (in-graph fp32→fp16 input cast, fp16 tower,
//     fp16→fp32 output widen, fp16 value/WDL readback) produces finite,
//     well-formed outputs — PASSES (fp16 inference is sound);
//   - a real fp16 model survives the safetensors save (incl. its bit-exact
//     forward-verify gate) → load round-trip on disk — PASSES.
//
//  KNOWN-FAILING (intentional investigation markers, kept failing by design —
//  like the `MacOS27NaNIsolationTests` cells, and like them pinned out of the
//  default scheme run):
//   - a real fp16 `trainStep` does NOT stay finite. It diverges on the very
//     first step at BOTH batch 1 and batch 64 — worse than bf16, whose batch-1
//     cell is finite. The forward CE components (policy, value) come back
//     finite, but the aggregate loss and the gradient norm are NaN, so the
//     overflow is in the fp16 BACKWARD and/or an auxiliary loss term (entropy /
//     illegal-mass), not the fp32-accumulated CE reductions. fp16's exponent
//     range is far narrower than bf16/fp32; viable fp16 training would need
//     loss scaling and/or fp32 computation of the loss + aux terms. Until that
//     exists, fp16 is an INFERENCE-only precision and these two cells stand as
//     the tripwire that proves it.
//
//  All Metal-backed; each skips if Metal is unavailable.
//

import XCTest
import Metal
import MetalPerformanceShadersGraph
@testable import DrewsChessMachine

final class FP16ComputePathTests: XCTestCase {

    private func requireMetal() throws {
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("Metal not available") }
    }

    /// The standard production tower (current default arch) with only the
    /// compute precision swapped to fp16.
    private func fp16Arch() -> NetworkArchitecture {
        var a = NetworkArchitecture.current
        a.computeDataType = .float16
        return a
    }

    /// A batch of `count` copies of the starting position, encoded the way the
    /// engine feeds the network.
    private func startingBatch(count: Int) -> [Float] {
        let boardFloats = BoardEncoder.tensorLength(for: .basic30)
        var oneBoard = [Float](repeating: 0, count: boardFloats)
        oneBoard.withUnsafeMutableBufferPointer { buf in
            BoardEncoder.encode(GameState.starting, into: buf, encoding: .basic30)
        }
        var batch = [Float](repeating: 0, count: count * boardFloats)
        for i in 0..<count {
            for j in 0..<boardFloats { batch[i * boardFloats + j] = oneBoard[j] }
        }
        return batch
    }

    /// A real fp16 forward pass must produce finite, well-formed outputs:
    /// policy of the right length, a value scalar in [-1, 1], and a W/D/L
    /// softmax whose three slots sum to 1 per position. Exercises the in-graph
    /// input narrow + fp16 tower + fp16→fp32 output widen + the fp16
    /// value/WDL readback (`readFloats(into:)`, the branch added for fp16).
    func test_fp16ForwardProducesFiniteWellFormedOutputs() async throws {
        try requireMetal()
        let arch = fp16Arch()
        let net = try ChessMPSNetwork(.randomWeights, arch: arch)
        let count = 4
        let batch = startingBatch(count: count)
        let classes = arch.valueHeadClasses

        nonisolated(unsafe) var policy: [Float] = []
        nonisolated(unsafe) var value: [Float] = []
        nonisolated(unsafe) var wdl: [Float] = []
        try await net.evaluateBatched(batchBoards: batch, count: count) { policyBuf, valueBuf, wdlBuf in
            policy = Array(policyBuf)
            value = Array(valueBuf)
            wdl = Array(wdlBuf)
        }

        XCTAssertEqual(policy.count, count * ChessNetwork.policySize)
        XCTAssertTrue(policy.allSatisfy { $0.isFinite }, "fp16 policy has non-finite logits")

        XCTAssertEqual(value.count, count)
        for v in value {
            XCTAssertTrue(v.isFinite, "fp16 value non-finite: \(v)")
            // v = p_win - p_loss, a difference of two probabilities.
            XCTAssertGreaterThanOrEqual(v, -1.0001, "fp16 value below -1: \(v)")
            XCTAssertLessThanOrEqual(v, 1.0001, "fp16 value above +1: \(v)")
        }

        XCTAssertEqual(wdl.count, count * classes)
        XCTAssertTrue(wdl.allSatisfy { $0.isFinite }, "fp16 W/D/L has non-finite probs")
        for i in 0..<count {
            let sum = (0..<classes).reduce(Float(0)) { $0 + wdl[i * classes + $1] }
            // Softmax computed in fp16 then widened; allow fp16 rounding slack.
            XCTAssertEqual(sum, 1.0, accuracy: 0.02, "fp16 W/D/L row \(i) softmax sum off: \(sum)")
        }
    }

    /// A real fp16 `trainStep` run over a few steps at the given batch, on the
    /// fp16 mixed-precision path (fp32 master, fp16 working, fp32 gradient
    /// widen). Mirrors `MacOS27NaNIsolationTests.sweep`, split per-batch (as the
    /// bf16 cells are) so the result localizes whether the break is batch-size-
    /// dependent.
    ///
    /// KNOWN-FAILING by design (see file header): fp16 diverges to a NaN
    /// gradient on the first step at every batch. These cells assert the
    /// finiteness that fp16 training *should* have; they fail today and are
    /// kept failing as the tripwire for "fp16 is inference-only". When fp16
    /// training is made viable (loss scaling / fp32 aux terms), they flip green
    /// and document the fix.
    private func fp16TrainSweep(batch: Int, steps: Int,
                                file: StaticString = #filePath, line: UInt = #line) async throws {
        try requireMetal()
        let trainer = try ChessTrainer(lrWarmupSteps: 0, arch: fp16Arch(),
                                       executableOptimizationLevel: .level1)
        for s in 0..<steps {
            let t = try await trainer.trainStep(batchSize: batch)
            XCTAssertTrue(t.loss.isFinite,
                "[fp16 batch=\(batch)] loss non-finite at step \(s): \(t.loss)", file: file, line: line)
            XCTAssertTrue(t.gradGlobalNorm.isFinite,
                "[fp16 batch=\(batch)] gradNorm non-finite at step \(s): \(t.gradGlobalNorm)", file: file, line: line)
        }
    }

    func test_fp16TrainStep_batch1_steps4()  async throws { try await fp16TrainSweep(batch: 1, steps: 4) }
    func test_fp16TrainStep_batch64_steps4() async throws { try await fp16TrainSweep(batch: 64, steps: 4) }

    /// A real fp16 model must survive the on-disk safetensors round-trip,
    /// including the bit-exact forward-pass verify gate inside `saveModel`
    /// (which builds an fp16 network and compares) — so the fp16 weight
    /// encode/decode byte path is exercised against the live graph, and the
    /// embedded architecture reloads as fp16.
    func test_fp16ModelSafetensorsRoundTrips() async throws {
        try requireMetal()
        let arch = fp16Arch()
        let net = try ChessMPSNetwork(.randomWeights, arch: arch)
        let weights = try await net.network.exportWeights()
        XCTAssertEqual(weights.count, arch.weightTensorPlan().count)

        let meta = ModelCheckpointMetadata(
            creator: "manual", trainingStep: 7, parentModelID: "", notes: "fp16 roundtrip unit"
        )
        let url = try await CheckpointManager.saveModel(
            weights: weights,
            modelID: "unittest-fp16-roundtrip",
            createdAtUnix: 1_780_000_001,
            metadata: meta,
            architecture: arch,
            trigger: "unittest"
        )
        defer {
            do { try FileManager.default.removeItem(at: url) }
            catch { /* best-effort cleanup of the unit-test artifact */ }
        }

        let loaded = try CheckpointManager.loadModelFile(at: url)
        XCTAssertEqual(loaded.modelID, "unittest-fp16-roundtrip")
        XCTAssertEqual(loaded.architecture.computeDataType, .float16,
                       "embedded compute dtype must reload as fp16")
        XCTAssertEqual(loaded.architecture, arch, "full architecture must round-trip")
        XCTAssertEqual(loaded.weights.count, weights.count)
        for (a, b) in zip(weights, loaded.weights) {
            XCTAssertEqual(a.map(\.bitPattern), b.map(\.bitPattern),
                           "fp16 weight bytes must round-trip bit-exact")
        }
    }
}
