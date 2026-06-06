//
//  RuntimeArchReachTests.swift
//  DrewsChessMachineTests
//
//  End-to-end "reach" coverage for the two runtime-architecture axes that
//  are selectable in the Build screen but were, until recently, not wired
//  through the runtime: the basic20 input encoding (Phase C) and the
//  scalar_tanh value head (Phase D). Each could be built/selected yet would
//  crash on first eval (basic20: boardSizeMismatch) or mistrain (scalar_tanh:
//  the trainer only had the W/D/L categorical-CE loss). These guard against a
//  regression that re-introduces the "selectable but broken" trap.
//
//  GPU-backed (build + forward pass + one synthetic train step), so they skip
//  cleanly when Metal is unavailable.
//

import XCTest
import Metal
@testable import DrewsChessMachine

final class RuntimeArchReachTests: XCTestCase {

    private func requireMetal() throws {
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal not available")
        }
    }

    /// `.current` with the input encoding overridden to basic20.
    private func basic20Arch() -> NetworkArchitecture {
        var arch = NetworkArchitecture.current
        arch.inputEncoding = .basic20
        return arch
    }

    /// `.current` with the value head overridden to the scalar tanh style.
    private func scalarTanhArch() -> NetworkArchitecture {
        var arch = NetworkArchitecture.current
        arch.valueHeadStyle = .scalarTanh
        return arch
    }

    // MARK: - basic20 (Phase C: input-encoding reach)

    func testBasic20ArchitectureShape() throws {
        let arch = basic20Arch()
        XCTAssertEqual(arch.inputPlanes, 20, "basic20 must report 20 input planes")
        XCTAssertEqual(
            BoardEncoder.tensorLength(for: arch.inputEncoding), 20 * 8 * 8,
            "basic20 encoded board must be 1280 floats")
        // Distinct from the default so the test is meaningful.
        XCTAssertNotEqual(arch.inputEncoding, NetworkArchitecture.current.inputEncoding)
    }

    func testBasic20NetworkBuildsAndEvaluates() async throws {
        try requireMetal()
        let net = try ChessMPSNetwork(.randomWeights, arch: basic20Arch())
        XCTAssertEqual(net.inputEncoding, .basic20)

        // Encode at the network's own encoding — 1280 floats — and run a
        // forward pass. Before Phase C this threw boardSizeMismatch because
        // the encode hardcoded basic30 (1920) against a 20-plane stem.
        let board = BoardEncoder.encode(.starting, encoding: net.inputEncoding)
        XCTAssertEqual(board.count, 20 * 8 * 8)

        let policyBox = SyncBox<[Float]>([])
        let valueBox = SyncBox<Float>(0)
        try await net.evaluate(board: board) { policyBuf, v in
            policyBox.value = Array(policyBuf)
            valueBox.value = v
        }
        XCTAssertEqual(policyBox.value.count, ChessNetwork.policySize)
        XCTAssertTrue(valueBox.value.isFinite, "basic20 value output must be finite")
    }

    func testBasic20TrainerStepRunsAtCorrectStride() async throws {
        try requireMetal()
        // The trainer synthesizes its own random boards at arch.inputPlanes ×
        // 64; a basic20 trainer must stage/feed 1280-wide boards into its
        // 1280-wide input placeholder. A stride mismatch would crash here.
        let trainer = try ChessTrainer(lrWarmupSteps: 0, arch: basic20Arch())
        let timing = try await trainer.trainStep(batchSize: 8)
        XCTAssertTrue(timing.policyLoss.isFinite, "basic20 policy loss must be finite")
        XCTAssertTrue(timing.valueLoss.isFinite, "basic20 value loss must be finite")
    }

    func testBasic20ReplayBufferStride() {
        // The buffer stride convenience matches the basic20 encoder width.
        XCTAssertEqual(
            BoardEncoder.tensorLength(for: .basic20), 1280)
        XCTAssertEqual(
            ReplayBuffer.bytesPerPosition(floatsPerBoard: BoardEncoder.tensorLength(for: .basic20)),
            ReplayBuffer.bytesPerPosition(floatsPerBoard: 1280))
    }

    // MARK: - scalar_tanh (Phase D: trainer value-loss branch)

    func testScalarTanhArchitectureShape() throws {
        let arch = scalarTanhArch()
        XCTAssertEqual(arch.valueHeadClasses, 1, "scalar tanh head emits a single value")
        XCTAssertNotEqual(arch.valueHeadStyle, NetworkArchitecture.current.valueHeadStyle)
    }

    func testScalarTanhNetworkValueInTanhRange() async throws {
        try requireMetal()
        let net = try ChessMPSNetwork(.randomWeights, arch: scalarTanhArch())
        let board = BoardEncoder.encode(.starting, encoding: net.inputEncoding)
        let valueBox = SyncBox<Float>(2)
        try await net.evaluate(board: board) { _, v in valueBox.value = v }
        XCTAssertTrue(valueBox.value.isFinite, "scalar tanh value must be finite")
        XCTAssertLessThanOrEqual(
            abs(valueBox.value), 1.0 + 1e-3,
            "scalar tanh value must lie in [-1, 1]")
    }

    func testScalarTanhTrainerStepUsesMSEBranch() async throws {
        try requireMetal()
        // Before Phase D the trainer's only value loss was the W/D/L
        // categorical CE (oneHot depth 3, 3-column valueProbs slices), which
        // is malformed for a 1-logit tanh head. The MSE branch must run and
        // produce a finite loss.
        let trainer = try ChessTrainer(lrWarmupSteps: 0, arch: scalarTanhArch())
        let timing = try await trainer.trainStep(batchSize: 8)
        XCTAssertTrue(timing.valueLoss.isFinite, "scalar tanh value loss (MSE) must be finite")
        XCTAssertGreaterThanOrEqual(timing.valueLoss, 0, "MSE is non-negative")
        XCTAssertTrue(timing.policyLoss.isFinite)
    }
}
