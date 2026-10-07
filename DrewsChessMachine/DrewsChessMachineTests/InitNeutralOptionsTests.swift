//
//  InitNeutralOptionsTests.swift
//  DrewsChessMachineTests
//
//  The init-neutral / ablation options (determinism plan B2, B2.1, decision
//  D-4): `se_gamma_bias_init`, `branch_output_init`, `skip_projection_init`
//  per block group, and `policy_head_final_init`, `value_head_final_init`,
//  `value_head_draw_prior` on the heads. These pin:
//  - every existing architecture already has the standard value of every
//    option, and a file older than the format that introduced them decodes to
//    exactly that, while a current-format file without one is an error;
//  - the Neutral / Standard sets are one function each, Neutral then Standard
//    restores the standard architecture, and the "differs from standard" set
//    names exactly the fields that were changed;
//  - forbidden combinations are refused by `validate()` with a named error;
//  - each option does what it claims at step 0 in a real build (uniform
//    policy, value softmax equal to the prior, zeroed last BN γ, identity-like
//    skip projection, SE γ-bias level) while every other tensor is the
//    standard build's, and one training step moves every zeroed tensor;
//  - each `--derive-model --set-<option>` rewrites only that option's tensors,
//    and `--set-neutral-init` gives the Build screen's neutral architecture.
//

import Metal
import XCTest
@testable import DrewsChessMachine

final class InitNeutralOptionsTests: XCTestCase {

    private func requireMetal() throws {
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("Metal not available") }
    }

    /// A small fp32 post-activation tower with two groups — attenuate-only SE
    /// at width 16, then scale-and-bias SE at width 24 (so the second group's
    /// first block has a skip projection) — no ReZero, clean add, the W/D/L
    /// value head, and the requested policy head.
    static func architecture(policy: PolicyHeadStyle = .intermediateConv) -> NetworkArchitecture {
        var arch = NetworkArchitecture(
            inputEncoding: .basic30, channels: 16, numBlocks: 1, stemConvKernelSize: 3,
            activationFunction: .relu, blockActivationStyle: .post,
            blockSkipMerge: .cleanAdd, blockUseRezero: false, rezeroAlphaInit: 0.5,
            blockConv1KernelSize: 3, blockConv2KernelSize: 3,
            blockSeStyle: .attenuateOnly, blockSeReductionRatio: 4,
            policyHeadStyle: policy, policyPreConvChannels: 8,
            valueHeadStyle: .wdlSoftmax, valueHeadConvChannels: 4, valueHeadHiddenUnits: 16,
            computeDataType: .float32
        )
        var second = arch.blockGroups[0]
        second.channels = 24
        second.count = 2
        second.seStyle = .scaleAndBias
        arch.blockGroups.append(second)
        return arch
    }

    private static func bits(_ values: [Float]) -> [UInt32] { values.map(\.bitPattern) }

    private static func planIndex(_ arch: NetworkArchitecture, _ name: String) throws -> Int {
        try XCTUnwrap(arch.weightTensorPlan().firstIndex { $0.name == name }, "no tensor \(name)")
    }

    // MARK: - Standard values and the two sets

    func testEveryPresetAlreadyHasTheStandardInit() {
        for preset in NetworkArchitecture.Preset.allCases {
            let arch = NetworkArchitecture.preset(preset)
            XCTAssertEqual(arch.withStandardInit(), arch, "\(preset)")
            XCTAssertTrue(arch.nonStandardInitOptions.isEmpty, "\(preset)")
        }
        XCTAssertEqual(Self.architecture().withStandardInit(), Self.architecture())
    }

    func testNeutralThenStandardRestoresTheStandardArchitecture() throws {
        let standard = Self.architecture()
        let neutral = standard.withNeutralInit()
        XCTAssertNotEqual(neutral, standard)
        try neutral.validate()
        XCTAssertEqual(neutral.withStandardInit(), standard)
    }

    func testNeutralSetsTheNeutralValueWhereEachOptionApplies() {
        var arch = Self.architecture()
        arch.valueHeadDrawPrior = 0.6
        let neutral = arch.withNeutralInit()
        for group in neutral.blockGroups {
            XCTAssertEqual(group.seGammaBiasInit, BlockGroup.neutralSEGammaBiasInit)
            XCTAssertEqual(group.branchOutputInit, .zeroLastBNGamma)
        }
        // Only the second group starts with a width change, so only it has a
        // skip projection.
        XCTAssertEqual(neutral.blockGroups[0].skipProjectionInit, .he)
        XCTAssertEqual(neutral.blockGroups[1].skipProjectionInit, .identityLike)
        XCTAssertEqual(neutral.policyHeadFinalInit, .zero)
        XCTAssertEqual(neutral.valueHeadFinalInit, .zero)
        // The draw prior is a claim about the data, not a neutral choice:
        // the Neutral set leaves it alone.
        XCTAssertEqual(neutral.valueHeadDrawPrior, 0.6)
        // Standard resets it, like every other option.
        XCTAssertEqual(neutral.withStandardInit().valueHeadDrawPrior, NetworkArchitecture.standardValueHeadDrawPrior)
    }

    func testNeutralLeavesOptionsWithoutTheirLayerStandard() {
        var arch = Self.architecture()
        arch.blockGroups[0].seStyle = .none
        arch.blockGroups[0].activationStyle = .pre
        arch.clearActivationSitesTheTopologyLacks()
        let neutral = arch.withNeutralInit()
        XCTAssertEqual(neutral.blockGroups[0].seGammaBiasInit, BlockGroup.standardSEGammaBiasInit)
        XCTAssertEqual(neutral.blockGroups[0].branchOutputInit, .standard)
        XCTAssertNoThrow(try neutral.validate())
    }

    func testNonStandardSetNamesExactlyTheChangedFields() {
        var arch = Self.architecture()
        arch.blockGroups[1].branchOutputInit = .zeroLastBNGamma
        arch.valueHeadDrawPrior = 0.5
        XCTAssertEqual(arch.nonStandardInitOptions, [.branchOutputInit(group: 1), .valueHeadDrawPrior])
        XCTAssertEqual(Self.architecture().withNeutralInit().nonStandardInitOptions, [
            .seGammaBiasInit(group: 0), .branchOutputInit(group: 0),
            .seGammaBiasInit(group: 1), .branchOutputInit(group: 1), .skipProjectionInit(group: 1),
            .policyHeadFinalInit, .valueHeadFinalInit,
        ])
    }

    // MARK: - Format gating

    /// `arch` encoded as JSON with the named block-group and top-level keys
    /// removed.
    private func json(_ arch: NetworkArchitecture, removingGroupKeys groupKeys: [String], topLevelKeys: [String]) throws -> Data {
        let encoded = try JSONEncoder().encode(arch)
        var object = try XCTUnwrap(try JSONSerialization.jsonObject(with: encoded) as? [String: Any])
        var groups = try XCTUnwrap(object["block_groups"] as? [[String: Any]])
        for index in groups.indices { for key in groupKeys { groups[index].removeValue(forKey: key) } }
        object["block_groups"] = groups
        for key in topLevelKeys { object.removeValue(forKey: key) }
        return try JSONSerialization.data(withJSONObject: object)
    }

    private static let groupInitKeys = ["se_gamma_bias_init", "branch_output_init", "skip_projection_init"]
    private static let headInitKeys = ["policy_head_final_init", "value_head_final_init", "value_head_draw_prior"]

    func testAFileOlderThanTheInitOptionsDecodesToTheStandardValues() throws {
        let arch = Self.architecture()
        let data = try json(arch, removingGroupKeys: Self.groupInitKeys, topLevelKeys: Self.headInitKeys)
        let format = ArchitectureFormat.DecodeFormat(
            formatVersion: ArchitectureFormat.initOptionsRequiredFromVersion - 1, source: "old.json")
        let decoded = try ArchitectureFormat.makeDecoder(format: format).decode(NetworkArchitecture.self, from: data)
        XCTAssertEqual(decoded, arch)
        let line = try XCTUnwrap(format.legacyLogLine)
        for key in Self.groupInitKeys + Self.headInitKeys {
            XCTAssertTrue(line.contains(key), "legacy log should name \(key): \(line)")
        }
    }

    func testACurrentFileMissingAnInitOptionIsAnError() throws {
        let arch = Self.architecture()
        for key in Self.groupInitKeys {
            let data = try json(arch, removingGroupKeys: [key], topLevelKeys: [])
            let format = ArchitectureFormat.DecodeFormat(formatVersion: ArchitectureFormat.currentVersion, source: "new.json")
            XCTAssertThrowsError(try ArchitectureFormat.makeDecoder(format: format).decode(NetworkArchitecture.self, from: data)) { error in
                guard case let ArchitectureFormat.FormatError.missingRequiredField(field, _, _, _) = error else {
                    return XCTFail("expected missingRequiredField, got \(error)")
                }
                XCTAssertEqual(field, key)
            }
        }
        for key in Self.headInitKeys {
            let data = try json(arch, removingGroupKeys: [], topLevelKeys: [key])
            let format = ArchitectureFormat.DecodeFormat(formatVersion: ArchitectureFormat.currentVersion, source: "new.json")
            XCTAssertThrowsError(try ArchitectureFormat.makeDecoder(format: format).decode(NetworkArchitecture.self, from: data)) { error in
                guard case let ArchitectureFormat.FormatError.missingRequiredField(field, _, _, _) = error else {
                    return XCTFail("expected missingRequiredField, got \(error)")
                }
                XCTAssertEqual(field, key)
            }
        }
    }

    func testEveryOptionRoundTripsThroughJSON() throws {
        var arch = Self.architecture().withNeutralInit()
        arch.valueHeadDrawPrior = 0.4
        arch.blockGroups[0].seGammaBiasInit = -1.25
        let data = try JSONEncoder().encode(arch)
        XCTAssertEqual(try JSONDecoder().decode(NetworkArchitecture.self, from: data), arch)
    }

    // MARK: - Validation

    private func assertRejected(_ arch: NetworkArchitecture, mentioning text: String,
                                file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertThrowsError(try arch.validate(), file: file, line: line) { error in
            XCTAssertTrue(String(describing: error).contains(text), "\(error)", file: file, line: line)
        }
    }

    func testForbiddenCombinationsAreRejected() {
        var noSE = Self.architecture()
        noSE.blockGroups[0].seStyle = .none
        noSE.blockGroups[0].seGammaBiasInit = 1
        assertRejected(noSE, mentioning: "se_gamma_bias_init")

        var nanBias = Self.architecture()
        nanBias.blockGroups[0].seGammaBiasInit = .nan
        assertRejected(nanBias, mentioning: "seGammaBiasInit")

        var preAct = Self.architecture()
        preAct.blockGroups[0].activationStyle = .pre
        preAct.blockGroups[0].branchOutputInit = .zeroLastBNGamma
        assertRejected(preAct, mentioning: "branch_output_init")

        var noProjection = Self.architecture()
        noProjection.blockGroups[0].skipProjectionInit = .identityLike
        assertRejected(noProjection, mentioning: "skip_projection_init")

        for prior: Float in [0, 1, -0.1, 1.5, .nan, .infinity] {
            var arch = Self.architecture()
            arch.valueHeadDrawPrior = prior
            assertRejected(arch, mentioning: "value_head_draw_prior")
        }

        var scalarHead = Self.architecture()
        scalarHead.valueHeadStyle = .scalarTanh
        scalarHead.valueHeadDrawPrior = 0.5
        assertRejected(scalarHead, mentioning: "value_head_draw_prior")
    }

    func testTheWDLPriorBiasGivesThePriorAsTheSoftmax() {
        for prior: Float in [0.75, 0.5, 0.2, 0.9] {
            let bias = NetworkArchitecture.wdlBiasPrior(drawProbability: prior).map(Double.init)
            let exps = bias.map { exp($0) }
            let total = exps.reduce(0, +)
            XCTAssertEqual(exps[1] / total, Double(prior), accuracy: 1e-6)
            XCTAssertEqual(exps[0] / total, (1 - Double(prior)) / 2, accuracy: 1e-6)
            XCTAssertEqual(exps[2] / total, (1 - Double(prior)) / 2, accuracy: 1e-6)
        }
        // The standard prior is exactly the literal every model was built with.
        XCTAssertEqual(NetworkArchitecture.wdlBiasPrior(drawProbability: 0.75)[1].bitPattern,
                       Float(1.791759469228055).bitPattern)
    }

    func testSummaryShowsOnlyNonStandardOptions() {
        let standard = Self.architecture()
        XCTAssertFalse(standard.architectureSummary.contains("init:"))
        let neutral = standard.withNeutralInit()
        XCTAssertTrue(neutral.architectureSummary.contains("init:"), neutral.architectureSummary)
    }

    // MARK: - Step-0 behavior in a real build

    func testPolicyHeadFinalZeroGivesAUniformPolicy() async throws {
        try requireMetal()
        for style in PolicyHeadStyle.allCases {
            var arch = Self.architecture(policy: style)
            arch.policyHeadFinalInit = .zero
            let network = try ChessNetwork(arch: arch, initialization: .seeded(initSeed: 3))
            let board = BoardEncoder.encode(.starting, encoding: arch.inputEncoding)
            let logits = SyncBox<[Float]>([])
            try await network.evaluate(board: board) { policy, _ in logits.value = Array(policy) }
            let first = try XCTUnwrap(logits.value.first)
            XCTAssertEqual(logits.value.count, NetworkArchitecture.policySize)
            XCTAssertTrue(logits.value.allSatisfy { $0 == first }, "\(style): policy logits are not uniform")
        }
    }

    func testValueHeadFinalZeroOutputsThePrior() async throws {
        try requireMetal()
        var arch = Self.architecture()
        arch.valueHeadFinalInit = .zero
        arch.valueHeadDrawPrior = 0.6
        let network = try ChessNetwork(arch: arch, initialization: .seeded(initSeed: 4))
        let wdl = try await network.evaluateValueDistribution(
            board: BoardEncoder.encode(.starting, encoding: arch.inputEncoding))
        XCTAssertEqual(wdl.draw, 0.6, accuracy: 1e-6)
        XCTAssertEqual(wdl.win, 0.2, accuracy: 1e-6)
        XCTAssertEqual(wdl.loss, 0.2, accuracy: 1e-6)
    }

    func testNeutralBuildChangesOnlyTheOptionTensors() async throws {
        try requireMetal()
        let standard = Self.architecture()
        let neutral = standard.withNeutralInit()
        let standardWeights = try await ChessNetwork(arch: standard, initialization: .seeded(initSeed: 8)).exportWeights()
        let neutralWeights = try await ChessNetwork(arch: neutral, initialization: .seeded(initSeed: 8)).exportWeights()
        let plan = neutral.weightTensorPlan()
        let gamma = BlockGroup.neutralSEGammaBiasInit

        var expectedChanged: Set<String> = [
            "blocks.0.se_attenuate.fc2.bias", "blocks.1.se_scalebias.fc2.bias", "blocks.2.se_scalebias.fc2.bias",
            "blocks.0.bn2.weight", "blocks.1.bn2.weight", "blocks.2.bn2.weight",
            "blocks.1.skip_proj.weight", "policy.conv.weight", "value.wdl_fc2.weight",
        ]
        for (index, spec) in plan.enumerated() {
            let changed = Self.bits(standardWeights[index]) != Self.bits(neutralWeights[index])
            if expectedChanged.remove(spec.name) != nil {
                XCTAssertTrue(changed, "\(spec.name) should differ from the standard build")
            } else {
                XCTAssertFalse(changed, "\(spec.name) should be the standard build's, bit for bit")
            }
        }
        XCTAssertTrue(expectedChanged.isEmpty, "not in the plan: \(expectedChanged)")

        func tensor(_ name: String) throws -> [Float] { neutralWeights[try Self.planIndex(neutral, name)] }
        XCTAssertEqual(try tensor("blocks.0.se_attenuate.fc2.bias"), [Float](repeating: gamma, count: 16))
        XCTAssertEqual(try tensor("blocks.1.se_scalebias.fc2.bias"),
                       [Float](repeating: gamma, count: 24) + [Float](repeating: 0, count: 24))
        for block in 0..<3 {
            XCTAssertTrue(try tensor("blocks.\(block).bn2.weight").allSatisfy { $0 == 0 })
        }
        let projection = try tensor("blocks.1.skip_proj.weight")
        for output in 0..<24 {
            for input in 0..<16 {
                XCTAssertEqual(projection[output * 16 + input], output == input ? 1 : 0)
            }
        }
        XCTAssertTrue(try tensor("policy.conv.weight").allSatisfy { $0 == 0 })
        XCTAssertTrue(try tensor("value.wdl_fc2.weight").allSatisfy { $0 == 0 })
        XCTAssertEqual(try tensor("value.wdl_fc2.bias"), NetworkArchitecture.wdlBiasPrior(drawProbability: 0.75))
    }

    /// Zero head finals pass no gradient back into the trunk (∂L/∂features =
    /// Wᵀ·∂L/∂logits = 0), so step one moves only the head finals; the zero
    /// last-BN γs start moving on step two, once the heads are non-zero. The
    /// synthetic step never advances the warmup count, so warmup is off here.
    func testZeroedTensorsAllMoveWithinTwoTrainingSteps() async throws {
        try requireMetal()
        let neutral = Self.architecture().withNeutralInit()
        let trainer = try ChessTrainer(
            dropoutStream: DCMRandom(seed: 1), lrWarmupSteps: 0, arch: neutral, initialization: .seeded(initSeed: 2))
        let heads = ["policy.conv.weight", "value.wdl_fc2.weight"]
        let lastBNGammas = (0..<3).map { "blocks.\($0).bn2.weight" }

        _ = try await trainer.trainStep(batchSize: 8)
        let afterOne = try await trainer.exportTrainerWeights()
        for name in heads {
            XCTAssertTrue(afterOne[try Self.planIndex(neutral, name)].contains { $0 != 0 }, "\(name) is still all zero after one step")
        }
        for name in lastBNGammas {
            XCTAssertTrue(afterOne[try Self.planIndex(neutral, name)].allSatisfy { $0 == 0 }, "\(name) moved with zero heads in front of it")
        }

        _ = try await trainer.trainStep(batchSize: 8)
        let afterTwo = try await trainer.exportTrainerWeights()
        for name in lastBNGammas {
            XCTAssertTrue(afterTwo[try Self.planIndex(neutral, name)].contains { $0 != 0 }, "\(name) is still all zero after two steps")
        }
    }

    // MARK: - --derive-model

    private func encodedModel(_ arch: NetworkArchitecture) throws -> Data {
        let weights = arch.weightTensorPlan().enumerated().map { tensorIndex, spec in
            (0..<spec.elementCount).map { Float(tensorIndex * 7 + $0 % 13) * 0.125 + 0.0625 }
        }
        let meta = ModelCheckpointMetadata(creator: "test", trainingStep: nil, parentModelID: "", notes: "fixture")
        return try SafetensorsModelIO.encode(
            modelID: "20261003-1-INIT", createdAtUnix: 1_790_000_000, metadata: meta, weights: weights,
            architecture: arch, includesVelocity: false,
            lineage: try LineageRecord.forTests(trainerCompletedSteps: nil, corpus: nil))
    }

    private func derive(_ data: Data, _ operations: [any DeriveOperation]) throws -> ModelDerivation.Result {
        try ModelDerivation.derive(
            sourceData: data, sourceName: "source.safetensors", operations: operations,
            newModelID: "20261003-2-DRVD", createdAtUnix: 1_790_000_100, build: "test", invocationArguments: ["test"], renamedTo: nil)
    }

    /// The names of the tensors `derived` holds differently from `source`.
    private func changedTensorNames(source: Data, derived: Data) throws -> Set<String> {
        let (sourceTensors, _) = try SafetensorsFile.decode(source)
        let (derivedTensors, _) = try SafetensorsFile.decode(derived)
        XCTAssertEqual(sourceTensors.map(\.name), derivedTensors.map(\.name))
        var changed: Set<String> = []
        for (before, after) in zip(sourceTensors, derivedTensors) where Self.bits(before.data) != Self.bits(after.data) {
            changed.insert(before.name)
        }
        return changed
    }

    /// A graft onto a neutral-init target gives every tensor it initializes
    /// the target's option values (the fresh build's), and records the
    /// constant ones by their option rather than as a draw.
    func testAGraftOntoANeutralTargetInitializesWithTheTargetsOptions() throws {
        try requireMetal()
        var target = Self.architecture().withNeutralInit()
        target.blockGroups[1].count = 3
        try target.validate()
        let fresh = try GraftFreshTarget.build(architecture: target, initSeed: 5)
        // Dropping the source's projection and head finals leaves the
        // target's to be initialized, beside the new block 3.
        let map = try GraftMap.parse("blocks.1.skip_proj.weight=,policy.conv.weight=,value.wdl_fc2.weight=")
        let result = try ModelDerivation.graft(
            sourceData: try encodedModel(Self.architecture()), sourceName: "source.safetensors", fresh: fresh,
            targetLabel: "neutral target", targetPreset: nil, map: map, initSeedOrigin: "entered", newModelID: "20261003-3-GRFT",
            createdAtUnix: 1_790_000_200, build: "test", invocationArguments: ["test"], renamedTo: nil)

        let perTensor = try XCTUnwrap(try XCTUnwrap(result.record.operations.last).perTensorInit)
        XCTAssertEqual(perTensor["blocks.1.skip_proj.weight"], RandomTensorRole.identityLike.rawValue)
        XCTAssertEqual(perTensor["policy.conv.weight"], RandomTensorRole.zero.rawValue)
        XCTAssertEqual(perTensor["value.wdl_fc2.weight"], RandomTensorRole.zero.rawValue)
        XCTAssertEqual(perTensor["blocks.3.bn2.weight"], ModelDerivation.builderConstantInit)

        let (tensors, _) = try SafetensorsFile.decode(result.data)
        var byName: [String: [Float]] = [:]
        for tensor in tensors { byName[tensor.name] = tensor.data }
        let projection = try XCTUnwrap(byName["blocks.1.skip_proj.weight"])
        XCTAssertEqual(projection, WeightInitScheme.identityLikeProjectionValues(outChannels: 24, inChannels: 16))
        XCTAssertTrue(try XCTUnwrap(byName["policy.conv.weight"]).allSatisfy { $0 == 0 })
        XCTAssertTrue(try XCTUnwrap(byName["value.wdl_fc2.weight"]).allSatisfy { $0 == 0 })
        XCTAssertTrue(try XCTUnwrap(byName["blocks.3.bn2.weight"]).allSatisfy { $0 == 0 })
        XCTAssertEqual(Array(try XCTUnwrap(byName["blocks.3.se_scalebias.fc2.bias"]).prefix(24)),
                       [Float](repeating: BlockGroup.neutralSEGammaBiasInit, count: 24))
    }

    private func kind(_ flag: String) throws -> DeriveOperationKind {
        try XCTUnwrap(ModelDerivation.kind(forFlag: flag), "no derive operation \(flag)")
    }

    func testEachSetOptionRewritesOnlyItsTensors() throws {
        let source = try encodedModel(Self.architecture())
        let cases: [(flag: String, value: String, groups: [Int]?, tensors: Set<String>)] = [
            ("--set-se-gamma-bias-init", "2", [1],
             ["blocks.1.se_scalebias.fc2.bias", "blocks.2.se_scalebias.fc2.bias"]),
            ("--set-branch-output-init", "zero_last_bn_gamma", nil,
             ["blocks.0.bn2.weight", "blocks.1.bn2.weight", "blocks.2.bn2.weight"]),
            ("--set-skip-projection-init", "identity_like", nil, ["blocks.1.skip_proj.weight"]),
            ("--set-policy-head-final-init", "zero", nil, ["policy.conv.weight"]),
            ("--set-value-head-final-init", "zero", nil, ["value.wdl_fc2.weight"]),
            ("--set-value-head-draw-prior", "0.5", nil, ["value.wdl_fc2.bias"]),
        ]
        for entry in cases {
            let operation = try kind(entry.flag).make(entry.value, entry.groups)
            let result = try derive(source, [operation])
            XCTAssertEqual(try changedTensorNames(source: source, derived: result.data), entry.tensors, entry.flag)
            XCTAssertEqual(result.record.operations.last?.operation, try kind(entry.flag).name)
            XCTAssertNoThrow(try result.targetArchitecture.validate())
        }
    }

    func testDerivedOptionTensorsHoldTheBuildersValues() throws {
        let source = try encodedModel(Self.architecture())
        let neutral = try derive(source, [try kind("--set-neutral-init").make("all", nil)])
        let decoded = try SafetensorsModelIO.decode(neutral.data).file
        let arch = neutral.targetArchitecture
        func tensor(_ name: String) throws -> [Float] { decoded.weights[try Self.planIndex(arch, name)] }
        XCTAssertEqual(try tensor("blocks.0.se_attenuate.fc2.bias"),
                       [Float](repeating: BlockGroup.neutralSEGammaBiasInit, count: 16))
        XCTAssertTrue(try tensor("blocks.1.bn2.weight").allSatisfy { $0 == 0 })
        XCTAssertTrue(try tensor("policy.conv.weight").allSatisfy { $0 == 0 })
        let projection = try tensor("blocks.1.skip_proj.weight")
        XCTAssertEqual(projection[3 * 16 + 3], 1)
        XCTAssertEqual(projection[3 * 16 + 4], 0)
        // The SE β half and every untouched tensor are the source's.
        let sourceDecoded = try SafetensorsModelIO.decode(source).file
        let sourceBias = sourceDecoded.weights[try Self.planIndex(arch, "blocks.1.se_scalebias.fc2.bias")]
        XCTAssertEqual(Array(try tensor("blocks.1.se_scalebias.fc2.bias")[24..<48]), Array(sourceBias[24..<48]))
    }

    func testDerivedHeRedrawIsTheFreshBuildsDraw() async throws {
        try requireMetal()
        var zeroed = Self.architecture()
        zeroed.policyHeadFinalInit = .zero
        zeroed.blockGroups[1].skipProjectionInit = .identityLike
        let source = try encodedModel(zeroed)
        let policy = try XCTUnwrap(try kind("--set-policy-head-final-init").make("he", nil) as? any InitSeedableDeriveOperation)
        let projection = try XCTUnwrap(try kind("--set-skip-projection-init").make("he", nil) as? any InitSeedableDeriveOperation)
        XCTAssertTrue(policy.drawsWeights)
        let result = try derive(source, [policy.withInitSeed(31), projection.withInitSeed(31)])
        let derived = try SafetensorsModelIO.decode(result.data).file.weights
        let fresh = try await ChessNetwork(arch: Self.architecture(), initialization: .seeded(initSeed: 31)).exportWeights()
        for name in ["policy.conv.weight", "blocks.1.skip_proj.weight"] {
            let index = try Self.planIndex(result.targetArchitecture, name)
            XCTAssertEqual(Self.bits(derived[index]), Self.bits(fresh[index]), name)
        }
        XCTAssertEqual(result.record.operations.first?.arguments["init_seed"], "31")
    }

    @MainActor
    func testDeriveNeutralMatchesTheBuildScreenNeutral() throws {
        let standard = Self.architecture()
        let derived = try derive(try encodedModel(standard), [try kind("--set-neutral-init").make("all", nil)])
        XCTAssertEqual(derived.targetArchitecture, standard.withNeutralInit())

        let model = BuildNewModelModel(NamedArchitecture(label: "test", architecture: standard))
        model.applyNeutralInit()
        XCTAssertEqual(model.architecture, derived.targetArchitecture)
        model.applyStandardInit()
        XCTAssertEqual(model.architecture, standard)
    }

    func testDeriveRefusesAnOptionThatChangesNothing() throws {
        let source = try encodedModel(Self.architecture())
        XCTAssertThrowsError(try derive(source, [try kind("--set-policy-head-final-init").make("he", nil)]))
        XCTAssertThrowsError(try derive(source, [try kind("--set-skip-projection-init").make("identity_like", [0])]))
    }
}
