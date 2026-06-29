import XCTest
@testable import DrewsChessMachine

/// Feature 2 (ARCHITECTURE_EXPANSION_PLAN.md): block groups — heterogeneous
/// towers with per-group widths. Covers the four Phase A contracts:
/// legacy-decode fallback, authored-structure round-trip, golden explicit
/// summaries, and the transition (width-change) tensor plan — plus GPU
/// builds proving the forward graph and the trainer's gradient graph
/// construct and run through a skip projection.
final class BlockGroupArchitectureTests: XCTestCase {

    /// A small mixed tower: one 7×7 block at 64ch, then three 3×3 blocks at
    /// 128ch — the width transition lands at expanded block index 1.
    private func mixedArch() -> NetworkArchitecture {
        var arch = NetworkArchitecture.current
        arch.blockGroups = [
            BlockGroup(
                count: 1, channels: 64,
                conv1KernelSize: 7, conv2KernelSize: 3,
                seStyle: .scaleAndBias, seReductionRatio: 4,
                useRezero: true, rezeroAlphaInit: 0.5,
                activationFunction: .relu, activationStyle: .pre,
                skipMerge: .cleanAdd, dropoutMultiplier: 1
            ),
            BlockGroup(
                count: 3, channels: 128,
                conv1KernelSize: 3, conv2KernelSize: 3,
                seStyle: .scaleAndBias, seReductionRatio: 4,
                useRezero: true, rezeroAlphaInit: 0.5,
                activationFunction: .relu, activationStyle: .pre,
                skipMerge: .cleanAdd, dropoutMultiplier: 0.5
            ),
        ]
        return arch
    }

    // MARK: Legacy decode fallback

    func testLegacyUniformKeysDecodeToSingleGroup() throws {
        // The exact key set every pre-block-groups safetensors/session file
        // embeds. Decode must expand it to one group and never reject it.
        let legacyJSON = """
        {
          "input_encoding": "basic30",
          "channels": 128,
          "num_blocks": 5,
          "stem_conv_kernel_size": 7,
          "activation_function": "relu",
          "block_activation_style": "pre",
          "block_skip_merge": "clean_add",
          "block_use_rezero": true,
          "rezero_alpha_init": 0.5,
          "block_conv1_kernel_size": 7,
          "block_conv2_kernel_size": 7,
          "block_se_style": "scale_and_bias",
          "block_se_reduction_ratio": 4,
          "policy_head_style": "intermediate_conv",
          "policy_pre_conv_channels": 128,
          "value_head_style": "wdl_softmax",
          "value_head_conv_channels": 16,
          "value_head_hidden_units": 128,
          "compute_data_type": "bfloat16"
        }
        """
        let decoded = try JSONDecoder().decode(
            NetworkArchitecture.self, from: Data(legacyJSON.utf8)
        )
        XCTAssertEqual(decoded.blockGroups.count, 1)
        let g = try XCTUnwrap(decoded.blockGroups.first)
        XCTAssertEqual(g.count, 5)
        XCTAssertEqual(g.channels, 128)
        XCTAssertEqual(g.conv1KernelSize, 7)
        XCTAssertEqual(g.conv2KernelSize, 7)
        XCTAssertEqual(g.seStyle, .scaleAndBias)
        XCTAssertEqual(g.seReductionRatio, 4)
        XCTAssertTrue(g.useRezero)
        XCTAssertEqual(g.rezeroAlphaInit, 0.5)
        XCTAssertEqual(g.activationFunction, .relu)
        XCTAssertEqual(g.activationStyle, .pre)
        XCTAssertEqual(g.skipMerge, .cleanAdd)
        XCTAssertEqual(g.dropoutMultiplier, 1, "legacy semantic: global rate applied unscaled")

        // The decoded value equals the same tower built via the uniform
        // convenience init — one source of truth for the expansion.
        let built = NetworkArchitecture(
            inputEncoding: .basic30, channels: 128, numBlocks: 5, stemConvKernelSize: 7,
            activationFunction: .relu, blockActivationStyle: .pre,
            blockSkipMerge: .cleanAdd, blockUseRezero: true, rezeroAlphaInit: 0.5,
            blockConv1KernelSize: 7, blockConv2KernelSize: 7,
            blockSeStyle: .scaleAndBias, blockSeReductionRatio: 4,
            policyHeadStyle: .intermediateConv, policyPreConvChannels: 128,
            valueHeadStyle: .wdlSoftmax, valueHeadConvChannels: 16, valueHeadHiddenUnits: 128,
            computeDataType: .bFloat16
        )
        XCTAssertEqual(decoded, built)
    }

    // MARK: Round-trip preserves authored structure

    func testRoundTripPreservesAuthoredGroupBoundaries() throws {
        // Two adjacent groups with IDENTICAL recipes must round-trip as two
        // groups — encode/decode never normalizes to the expansion.
        var arch = NetworkArchitecture.current
        let recipe = arch.blockGroups[0]
        var halfA = recipe; halfA.count = 2
        var halfB = recipe; halfB.count = 3
        arch.blockGroups = [halfA, halfB]
        try arch.validate()

        let enc = JSONEncoder()
        enc.outputFormatting = [.sortedKeys]
        let data = try enc.encode(arch)
        let decoded = try JSONDecoder().decode(NetworkArchitecture.self, from: data)
        XCTAssertEqual(decoded.blockGroups.count, 2, "authored boundaries preserved")
        XCTAssertEqual(decoded, arch)
        // Byte-stable re-encode (canonical identity property).
        XCTAssertEqual(try enc.encode(decoded), data)
        // But the EXPANSION matches the single-group tower's expansion.
        XCTAssertEqual(decoded.expandedBlocks, NetworkArchitecture.current.expandedBlocks)
    }

    func testExpandedBlocksNormalization() {
        let arch = mixedArch()
        let expanded = arch.expandedBlocks
        XCTAssertEqual(expanded.count, 4)
        XCTAssertTrue(expanded.allSatisfy { $0.count == 1 }, "one element ≡ one block")
        XCTAssertEqual(expanded.map(\.channels), [64, 128, 128, 128])
        XCTAssertEqual(arch.numBlocks, 4)
        XCTAssertEqual(arch.stemOutputChannels, 64)
        XCTAssertEqual(arch.towerOutputChannels, 128)
        XCTAssertEqual(arch.maxBlockChannels, 128)
    }

    // MARK: Golden summaries (explicit form — no silent defaults)

    func testGoldenSummaryUniformTower() {
        let params = 8_445_748.formatted(.number)
        XCTAssertEqual(
            NetworkArchitecture.current.architectureSummary,
            "v4 . in basic30(30) -> stem 128 (7x7)"
            + " . 5x[7x7+7x7 @128, SE+/4, relu/pre, clean_add, ReZero(0.447·tanh≤0.447), drop*1]"
            + " . act relu . policy intermediate_conv(4864)"
            + " . value WDL(16->FC128) . bfloat16 . \(params) params"
        )
    }

    func testGoldenSummaryMixedTower() {
        let arch = mixedArch()
        let params = arch.parameterCount.formatted(.number)
        XCTAssertEqual(
            arch.architectureSummary,
            "v4 . in basic30(30) -> stem 64 (7x7)"
            + " . 1x[7x7+3x3 @64, SE+/4, relu/pre, clean_add, ReZero(0.5·tanh≤0.5), drop*1]"
            + " -> 3x[3x3+3x3 @128, SE+/4, relu/pre, clean_add, ReZero(0.5·tanh≤0.5), drop*0.5]"
            + " . act relu . policy intermediate_conv(4864)"
            + " . value WDL(16->FC128) . bfloat16 . \(params) params"
        )
    }

    // MARK: Transition tensors in the weight plan

    func testMixedPlanCarriesTransitionTensors() throws {
        let arch = mixedArch()
        try arch.validate()
        let plan = arch.weightTensorPlan()
        let byName = Dictionary(uniqueKeysWithValues: plan.map { ($0.name, $0) })

        // Stem at the FIRST group's width.
        XCTAssertEqual(byName["stem.conv.weight"]?.shape, [64, 30, 7, 7])
        // Block 0: 64 → 64, no projection.
        XCTAssertEqual(byName["blocks.0.conv1.weight"]?.shape, [64, 64, 7, 7])
        XCTAssertNil(byName["blocks.0.skip_proj.weight"])
        // Block 1 is the transition: BN1 sized inC, conv1 carries 64 → 128,
        // and the skip projection appears — shaped [outC, inC, 1, 1].
        XCTAssertEqual(byName["blocks.1.bn1.weight"]?.shape, [64])
        XCTAssertEqual(byName["blocks.1.conv1.weight"]?.shape, [128, 64, 3, 3])
        XCTAssertEqual(byName["blocks.1.conv2.weight"]?.shape, [128, 128, 3, 3])
        XCTAssertEqual(byName["blocks.1.skip_proj.weight"]?.shape, [128, 64, 1, 1])
        // Exactly ONE projection in this tower; it appends LAST in its block
        // (immediately after blocks.1.rezero_alpha).
        let projNames = plan.map(\.name).filter { $0.hasSuffix("skip_proj.weight") }
        XCTAssertEqual(projNames, ["blocks.1.skip_proj.weight"])
        let names = plan.map(\.name)
        let alphaIdx = try XCTUnwrap(names.firstIndex(of: "blocks.1.rezero_alpha"))
        XCTAssertEqual(names[alphaIdx + 1], "blocks.1.skip_proj.weight")
        // Heads read the tower-output width.
        XCTAssertEqual(byName["value.conv.weight"]?.shape, [16, 128, 1, 1])
        XCTAssertEqual(byName["policy.pre_conv.weight"]?.shape, [128, 128, 1, 1])

        // Cross-checks: unique names; the parameter formula matches the plan.
        XCTAssertEqual(Set(names).count, names.count)
        XCTAssertEqual(plan.reduce(0) { $0 + $1.elementCount }, arch.parameterCount)
    }

    // MARK: v5 — output LayerNorm

    func testV5OutputLayerNormPlanAndParams() throws {
        let arch = NetworkArchitecture.preset(.v5_5block_7x7_lnout)
        try arch.validate()
        let plan = arch.weightTensorPlan()
        let names = plan.map(\.name)
        let byName = Dictionary(uniqueKeysWithValues: plan.map { ($0.name, $0) })

        // Each block grows res_ln.weight + res_ln.bias, each [channels].
        for b in 0..<5 {
            XCTAssertEqual(byName["blocks.\(b).res_ln.weight"]?.shape, [128])
            XCTAssertEqual(byName["blocks.\(b).res_ln.bias"]?.shape, [128])
            XCTAssertEqual(byName["blocks.\(b).res_ln.weight"]?.kind, .bnAffine)
            // LayerNorm keeps NO running stats.
            XCTAssertNil(byName["blocks.\(b).res_ln.running_mean"])
        }
        // res_ln appends LAST within the block: this is a uniform 128-wide tower
        // (no skip_proj), so γ/β follow rezero_alpha directly, in weight→bias order.
        let alphaIdx = try XCTUnwrap(names.firstIndex(of: "blocks.0.rezero_alpha"))
        XCTAssertEqual(names[alphaIdx + 1], "blocks.0.res_ln.weight")
        XCTAssertEqual(names[alphaIdx + 2], "blocks.0.res_ln.bias")

        // Unique names; param formula matches the plan; exact count = v4 + 5·2·128.
        XCTAssertEqual(Set(names).count, names.count)
        XCTAssertEqual(plan.reduce(0) { $0 + $1.elementCount }, arch.parameterCount)
        XCTAssertEqual(arch.parameterCount, 8_447_028)
        XCTAssertEqual(arch.parameterCount,
                       NetworkArchitecture.preset(.v4_5block_7x7).parameterCount + 5 * 2 * 128)

        // groupSummary surfaces the output norm; v4 (no output norm) does not.
        XCTAssertTrue(NetworkArchitecture.groupSummary(arch.blockGroups[0]).contains("out:layer_norm"))
        XCTAssertFalse(NetworkArchitecture.groupSummary(
            NetworkArchitecture.preset(.v4_5block_7x7).blockGroups[0]).contains("out:"))

        // Family label: output norm bumps to v5; v4 (no output norm) stays v4.
        XCTAssertEqual(arch.architectureVersionLabel, 5)
        XCTAssertTrue(arch.architectureSummary.hasPrefix("v5 "))
        XCTAssertEqual(NetworkArchitecture.preset(.v4_5block_7x7).architectureVersionLabel, 4)
    }

    // MARK: GPU — v5 LayerNorm-output tower builds, evaluates, exports

    func testV5OutputLayerNormBuildsAndEvaluates() async throws {
        let arch = NetworkArchitecture.preset(.v5_5block_7x7_lnout)
        try arch.validate()
        let net = try ChessMPSNetwork(.randomWeights, arch: arch)
        // exportWeights count == plan count confirms the GRAPH BUILDER appended
        // exactly the tensors weightTensorPlan lists — i.e. the res_ln γ/β are
        // wired in the builder in lockstep with the plan (the index-aligned
        // save/load contract). A forward pass confirms the LN graph (mean/var
        // over the channel axis + normalize) actually builds and is finite.
        let weights = try await net.network.exportWeights()
        XCTAssertEqual(weights.count, arch.weightTensorPlan().count)

        let board = BoardEncoder.encode(.starting, encoding: .basic30)
        try await net.evaluate(board: board) { policyBuf, value in
            XCTAssertEqual(policyBuf.count, arch.policySize)
            XCTAssertTrue(value.isFinite)
            XCTAssertTrue(policyBuf.allSatisfy { $0.isFinite })
        }
    }

    // MARK: GPU — mixed tower builds, evaluates, exports

    func testMixedArchBuildsAndEvaluates() async throws {
        let arch = mixedArch()
        try arch.validate()
        let net = try ChessMPSNetwork(.randomWeights, arch: arch)
        let weights = try await net.network.exportWeights()
        XCTAssertEqual(weights.count, arch.weightTensorPlan().count)

        let board = BoardEncoder.encode(.starting, encoding: .basic30)
        try await net.evaluate(board: board) { policyBuf, value in
            XCTAssertEqual(policyBuf.count, arch.policySize)
            XCTAssertTrue(value.isFinite)
        }
    }

    // MARK: GPU — gradient graph builds through the projection

    func testMixedArchTrainerGradientGraphBuildsAndForks() async throws {
        // ChessTrainer's init runs MPSGraph autodiff over the full training
        // graph; an unreachable-gradient regression at the projection (or the
        // per-width dropout mask shapes) hard-fails here, not at runtime.
        let arch = mixedArch()
        try arch.validate()
        let champion = try ChessMPSNetwork(.randomWeights, arch: arch)
        let championWeights = try await champion.network.exportWeights()

        let trainer = try ChessTrainer(arch: arch)
        XCTAssertEqual(trainer.arch, arch)
        try await trainer.network.loadWeights(championWeights)
    }

    // MARK: GPU — mixed tower saves with bit-exact verification, reloads

    func testMixedArchSavesVerifiesAndReloadsBitExact() async throws {
        // CheckpointManager.saveModel runs the embedded forward-pass
        // verification against an arch-matched scratch network — for a
        // mixed tower that is the strongest end-to-end proof: weight plan,
        // safetensors layout, loader shape validation, and the inference
        // graph (projection included) all have to agree bit-for-bit.
        let arch = mixedArch()
        try arch.validate()
        let champion = try ChessMPSNetwork(.randomWeights, arch: arch)
        let weights = try await champion.network.exportWeights()

        let url = try await CheckpointManager.saveModel(
            weights: weights, modelID: "unittest-blockgroups",
            createdAtUnix: 1_781_300_000,
            metadata: ModelCheckpointMetadata(
                creator: "manual", trainingStep: nil, parentModelID: "", notes: "mixed-groups"
            ),
            architecture: arch, trigger: "unittest"
        )
        defer {
            do { try FileManager.default.removeItem(at: url) }
            catch { XCTFail("cleanup failed: \(error)") }
        }

        let loaded = try CheckpointManager.loadModelFile(at: url)
        XCTAssertEqual(loaded.architecture, arch,
                       "embedded config round-trips the authored group structure")
        XCTAssertEqual(loaded.weights.count, weights.count)
        for (a, b) in zip(weights, loaded.weights) {
            XCTAssertEqual(a.map(\.bitPattern), b.map(\.bitPattern))
        }
    }
}
