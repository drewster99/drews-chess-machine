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

    /// A basic20-stride buffer must save and restore round-trip. The restore
    /// target's stride is fixed at construction, so the loader has to peek the
    /// file's recorded stride (`peekFloatsPerBoard`) rather than assume the
    /// default basic30 width — otherwise the save-verify scratch and the CLI
    /// restore reject every basic20 file with `incompatibleBoardSize`. Pure
    /// persistence (no Metal).
    func testBasic20BufferSaveRestoreRoundTripViaPeek() throws {
        let stride = BoardEncoder.tensorLength(for: .basic20)
        XCTAssertEqual(stride, 1280)
        let buffer = ReplayBuffer(capacity: 64, floatsPerBoard: stride)
        XCTAssertEqual(buffer.floatsPerBoard, stride)

        var board = [Float](repeating: 0, count: stride)
        for i in 0..<stride { board[i] = Float(i % 7) * 0.5 }
        var move: Int32 = 99
        var ply: UInt16 = 0
        var tau: Float = 0.8
        var hash: UInt64 = 0x1234_5678
        var mat: UInt8 = 24
        // append takes a per-position outcomes pointer; one position here.
        var outcome: Float = 1.0
        board.withUnsafeBufferPointer { boardsBuf in
            guard let base = boardsBuf.baseAddress else {
                XCTFail("board buffer baseAddress is nil"); return
            }
            withUnsafePointer(to: &move) { moveP in
            withUnsafePointer(to: &ply) { plyP in
            withUnsafePointer(to: &tau) { tauP in
            withUnsafePointer(to: &hash) { hashP in
            withUnsafePointer(to: &mat) { matP in
            withUnsafePointer(to: &outcome) { outcomeP in
                buffer.append(
                    boards: base,
                    policyIndices: moveP,
                    plyIndices: plyP,
                    samplingTaus: tauP,
                    stateHashes: hashP,
                    materialCounts: matP,
                    gameLength: 1,
                    workerId: 0,
                    intraWorkerGameIndex: 0,
                    outcomes: outcomeP,
                    count: 1
                )
            }}}}}}
        }

        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("bin")
        defer { try? FileManager.default.removeItem(at: url) }
        try buffer.write(to: url)

        // The fix: peek the file's stride and construct a matching buffer.
        XCTAssertEqual(try ReplayBuffer.peekFloatsPerBoard(at: url), stride)
        let cap = try ReplayBuffer.peekCapacity(at: url)
        let restored = ReplayBuffer(
            capacity: cap,
            floatsPerBoard: try ReplayBuffer.peekFloatsPerBoard(at: url))
        try restored.restore(from: url)
        XCTAssertEqual(restored.count, 1)
        XCTAssertEqual(restored.floatsPerBoard, stride)

        // Negative: a default-stride (basic30) buffer must reject the basic20
        // file — proving the peek is load-bearing, not cosmetic.
        let wrongStride = ReplayBuffer(capacity: cap)
        XCTAssertThrowsError(
            try wrongStride.restore(from: url),
            "a basic30-stride buffer must reject a basic20 file")
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

    func testScalarTanhEvaluateValueDistributionStaysInBounds() async throws {
        try requireMetal()
        // Regression: evaluateValueDistribution sizes its W/D/L probs scratch to
        // arch.valueHeadClasses, which is 1 for scalar_tanh. The pre-fix code
        // unconditionally returned slots [0]/[1]/[2], reading two floats off the
        // end of a one-element allocation. It is reachable via the tactical /
        // candidate probes, which call this exact method on a ChessMPSNetwork
        // with no value-head-style gate. The fix branches on valueHeadStyle and
        // projects the single tanh scalar v onto (win, draw, loss) preserving
        // win - loss = v. This asserts the projection is sane and matches the
        // derived scalar from the universal eval path.
        let net = try ChessMPSNetwork(.randomWeights, arch: scalarTanhArch())
        let board = BoardEncoder.encode(.starting, encoding: net.inputEncoding)

        let wdl = try await net.evaluateValueDistribution(board: board)
        XCTAssertTrue(wdl.win.isFinite && wdl.draw.isFinite && wdl.loss.isFinite,
                      "scalar_tanh W/D/L projection must be finite")
        XCTAssertEqual(wdl.draw, 0, "a scalar head carries no separable draw mass")
        XCTAssertGreaterThanOrEqual(wdl.win, 0, "win mass must be non-negative")
        XCTAssertGreaterThanOrEqual(wdl.loss, 0, "loss mass must be non-negative")
        XCTAssertLessThanOrEqual(abs(wdl.win - wdl.loss), 1.0 + 1e-3,
                                 "win - loss reconstructs v = tanh in [-1, 1]")

        // The same deterministic forward pass through the universal eval path
        // yields the derived scalar v; win - loss must reconstruct it.
        let vBox = SyncBox<Float>(2)
        try await net.evaluate(board: board) { _, v in vBox.value = v }
        XCTAssertEqual(wdl.win - wdl.loss, vBox.value, accuracy: 1e-4,
                       "win - loss must equal the derived scalar value v")
    }

    // MARK: - bf16 compute precision (per-arch precision reach)

    /// `.current` with the compute precision overridden to bf16, independent of
    /// whichever preset (fp32 or bf16) is active.
    private func bf16Arch() -> NetworkArchitecture {
        var arch = NetworkArchitecture.current
        arch.computeDataType = .bFloat16
        return arch
    }

    func testBF16ArchitectureShape() throws {
        XCTAssertEqual(bf16Arch().computeDataType, .bFloat16)
    }

    func testBF16NetworkBuildsAndEvaluatesFinite() async throws {
        try requireMetal()
        // The bf16 cast math is characterized unconditionally in
        // BF16CastEquivalenceTests, but its higher-level "a bf16-configured
        // network builds and runs a finite forward pass" checks skip unless
        // `.current` is bf16. Pin it explicitly so per-arch precision threading
        // (compute_data_type) is exercised regardless of the active preset.
        let net = try ChessMPSNetwork(.randomWeights, arch: bf16Arch())
        XCTAssertEqual(net.arch.computeDataType, .bFloat16)

        let board = BoardEncoder.encode(.starting, encoding: net.inputEncoding)
        let policyBox = SyncBox<[Float]>([])
        let valueBox = SyncBox<Float>(2)
        try await net.evaluate(board: board) { policyBuf, v in
            policyBox.value = Array(policyBuf)
            valueBox.value = v
        }
        XCTAssertEqual(policyBox.value.count, ChessNetwork.policySize)
        XCTAssertTrue(policyBox.value.allSatisfy { $0.isFinite },
                      "bf16 policy logits must all be finite")
        XCTAssertTrue(valueBox.value.isFinite, "bf16 value must be finite")
        XCTAssertLessThanOrEqual(abs(valueBox.value), 1.0 + 1e-3,
                                 "value (p_win - p_loss) stays in [-1, 1]")
    }

    func testBF16TrainerStepProducesFiniteLosses() async throws {
        try requireMetal()
        // A bf16 trainer must stage/feed bf16 through the master-weights path
        // and produce finite losses; a precision-threading bug (wrong dtype on
        // a buffer or placeholder) would surface as NaN/Inf or a crash here.
        let trainer = try ChessTrainer(lrWarmupSteps: 0, arch: bf16Arch())
        let timing = try await trainer.trainStep(batchSize: 8)
        XCTAssertTrue(timing.policyLoss.isFinite, "bf16 policy loss must be finite")
        XCTAssertTrue(timing.valueLoss.isFinite, "bf16 value loss must be finite")
    }

    // MARK: - Hidden activation (relu / silu / gelu)

    /// The activation picker in the Build screen offers silu and gelu beside
    /// relu. They route through one `ChessNetwork.activation` helper used at
    /// every hidden site, but the code path had no coverage and the active
    /// preset is relu — so silu/gelu had never actually executed. This proves
    /// they're genuinely wired (not silently relu) by holding the weights fixed
    /// and showing each activation yields a DIFFERENT, finite forward pass.
    func testSiluAndGeluChangeForwardPassVsReLU() async throws {
        try requireMetal()
        var reluArch = NetworkArchitecture.current
        reluArch.activationFunction = .relu

        // One random weight set, shared across all three activations. Loading
        // it over each net overwrites the build-time BN warmup stats too, so
        // the ONLY difference between the three forward passes is the
        // activation function itself.
        let reluNet = try ChessMPSNetwork(.randomWeights, arch: reluArch)
        let weights = try await reluNet.network.exportWeights()
        let board = BoardEncoder.encode(.starting, encoding: reluNet.inputEncoding)

        func policy(_ activation: ActivationFunction) async throws -> [Float] {
            var arch = reluArch
            arch.activationFunction = activation
            let net = try ChessMPSNetwork(.randomWeights, arch: arch)
            try await net.network.loadWeights(weights)
            let box = SyncBox<[Float]>([])
            try await net.evaluate(board: board) { p, _ in box.value = Array(p) }
            return box.value
        }

        let reluP = try await policy(.relu)
        let siluP = try await policy(.silu)
        let geluP = try await policy(.gelu)

        XCTAssertEqual(reluP.count, ChessNetwork.policySize)
        for (name, p) in [("relu", reluP), ("silu", siluP), ("gelu", geluP)] {
            XCTAssertTrue(p.allSatisfy { $0.isFinite }, "\(name) policy logits must be finite")
        }
        // Same weights → any difference is purely the activation taking effect.
        XCTAssertNotEqual(reluP, siluP, "silu must change the forward pass vs relu")
        XCTAssertNotEqual(reluP, geluP, "gelu must change the forward pass vs relu")
        XCTAssertNotEqual(siluP, geluP, "silu and gelu must differ from each other")
    }

    func testGeluTrainerStepProducesFiniteLosses() async throws {
        try requireMetal()
        // gelu's exact erf path is the most exotic activation op; exercise it in
        // the training graph (not just inference) to prove it builds and runs.
        var arch = NetworkArchitecture.current
        arch.activationFunction = .gelu
        let trainer = try ChessTrainer(lrWarmupSteps: 0, arch: arch)
        let timing = try await trainer.trainStep(batchSize: 8)
        XCTAssertTrue(timing.policyLoss.isFinite, "gelu policy loss must be finite")
        XCTAssertTrue(timing.valueLoss.isFinite, "gelu value loss must be finite")
    }
}
