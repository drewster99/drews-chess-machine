//
//  SEActivationTests.swift
//  DrewsChessMachineTests
//
//  GitHub issue #2: the per-block-group `se_activation` — the activation after
//  the SE excitation FC1, independent of the group's main-path activation, so
//  leaky ReLU can go on the bottleneck where FC1 units were measured dying
//  without paying for leaky ReLU on every conv.
//
//  Pinned here:
//  - Format v5 gate: a v5 file must state the field; an older file without it
//    resolves to the group's own `activation_function` (the SE FC1's
//    activation before the field existed), so every existing model and preset
//    decodes to the identical architecture — same equality, hash, summary,
//    parameter count, legacy archHash and forward pass.
//  - Builder: FC1 uses `seActivation` and no other site does.
//  - Validation: an SE-less group's `seActivation` must equal its activation.
//  - `--set-se-activation`: changes only the field, copies every tensor
//    bit-exact; `--set-activation` leaves SE groups' FC1 activation alone.
//  - The Build screen's follow rule.
//
//  The build / forward / train-step cases are GPU-backed and skip without
//  Metal; the format, validation and derivation cases are pure.
//

import XCTest
import Metal
@testable import DrewsChessMachine

final class SEActivationTests: XCTestCase {

    // MARK: - Fixtures

    /// Small fp32 pre-act tower with two SE groups (so group selection and
    /// mixed settings are both exercisable), cheap to build. Group 0 has
    /// `group0Activation` on its main path, group 1 ReLU; the SE FC1
    /// activations are given separately.
    private static func twoGroupArchitecture(
        group0Activation: ActivationFunction = .relu,
        group0SE: ActivationFunction,
        group1SE: ActivationFunction
    ) -> NetworkArchitecture {
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
        var first = arch.blockGroups[0]
        first.activationFunction = group0Activation
        first.seActivation = group0SE
        var second = arch.blockGroups[0]
        second.count = 2
        second.seStyle = .attenuateOnly
        second.seActivation = group1SE
        arch.blockGroups = [first, second]
        return arch
    }

    private func requireMetal() throws {
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal not available")
        }
    }

    /// `json` (an encoded architecture, a preset, or anything containing
    /// `block_groups` at the top level or under `architecture`) with every
    /// `key` removed from every block group.
    private static func removingGroupKey(_ key: String, from json: Data) throws -> Data {
        func strip(_ groups: Any?) throws -> [[String: Any]] {
            let list = try XCTUnwrap(groups as? [[String: Any]], "no block_groups")
            return list.map { group in
                var g = group
                g.removeValue(forKey: key)
                return g
            }
        }
        var object = try XCTUnwrap(try JSONSerialization.jsonObject(with: json) as? [String: Any],
                                   "architecture JSON is not an object")
        if var wrapped = object["architecture"] as? [String: Any] {
            wrapped["block_groups"] = try strip(wrapped["block_groups"])
            object["architecture"] = wrapped
        } else {
            object["block_groups"] = try strip(object["block_groups"])
        }
        return try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
    }

    private func encodedModel(_ arch: NetworkArchitecture, modelID: String = "20261001-1-SRCE") throws -> Data {
        let weights = arch.weightTensorPlan().enumerated().map { tensorIndex, spec in
            (0..<spec.elementCount).map { Float(tensorIndex * 7 + $0 % 13) * 0.125 + 0.0625 }
        }
        let meta = ModelCheckpointMetadata(creator: "test", trainingStep: nil, parentModelID: "", notes: "fixture")
        return try SafetensorsModelIO.encode(
            modelID: modelID, createdAtUnix: 1_790_000_000, metadata: meta, weights: weights,
            architecture: arch, includesVelocity: false,
            lineage: try LineageRecord.forTests(trainerCompletedSteps: meta.trainerSchedule.map(\.completedTrainSteps), corpus: nil))
    }

    /// Re-encode `data` with `dcm_format_version` set to `version` (nil =
    /// removed) and, when `stripField`, `se_activation` removed from the
    /// embedded architecture.
    private func rewritingHeader(_ data: Data, version: String?, stripField: Bool) throws -> Data {
        let (tensors, decodedMetadata) = try SafetensorsFile.decode(data)
        var metadata = decodedMetadata
        metadata.removeValue(forKey: SafetensorsFile.contentHashKey)
        metadata["dcm_format_version"] = version
        if stripField {
            let archJSON = try XCTUnwrap(metadata["architecture"])
            metadata["architecture"] = String(
                decoding: try Self.removingGroupKey("se_activation", from: Data(archJSON.utf8)), as: UTF8.self)
        }
        return try SafetensorsFile.encode(tensors: tensors, metadata: metadata)
    }

    private func temporaryURL(_ name: String) -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("se-activation-\(UUID().uuidString)-\(name)")
        addTeardownBlock {
            do { try FileManager.default.removeItem(at: url) } catch {}
        }
        return url
    }

    // MARK: - Format version

    func testCurrentVersionRequiresTheField() {
        XCTAssertEqual(ArchitectureFormat.seActivationRequiredFromVersion, 5)
        XCTAssertGreaterThanOrEqual(ArchitectureFormat.currentVersion, ArchitectureFormat.seActivationRequiredFromVersion)
        XCTAssertGreaterThan(ArchitectureFormat.seActivationRequiredFromVersion, ArchitectureFormat.seBetaInitRequiredFromVersion)
        XCTAssertEqual(SafetensorsModelIO.formatVersion, String(ArchitectureFormat.currentVersion))
    }

    // MARK: - Existing architectures are unchanged

    /// Every built-in preset sets `seActivation` = its group activation, and a
    /// legacy (format v4, no `se_activation`) file of it decodes to exactly
    /// the preset: equal, same hash, same summary, same parameter count, same
    /// legacy `.dcmmodel` archHash. This is the identity-stability guarantee
    /// for every model saved before the field existed.
    func testExistingPresetsKeepTheirIdentity() throws {
        for preset in NetworkArchitecture.Preset.allCases {
            let arch = NetworkArchitecture.preset(preset)
            XCTAssertTrue(arch.blockGroups.allSatisfy { $0.seActivation == $0.activationFunction }, preset.rawValue)
            XCTAssertFalse(arch.architectureSummary.contains("(fc1 "), preset.rawValue)
            if let legacyHash = NetworkArchitecture.legacyArchHash(for: preset) {
                XCTAssertEqual(ModelCheckpointFile.archHash(for: arch), legacyHash, preset.rawValue)
            }
            let legacyJSON = try Self.removingGroupKey("se_activation", from: try JSONEncoder().encode(arch))
            for version in [3, 4] {
                let format = ArchitectureFormat.DecodeFormat(formatVersion: version, source: "\(preset.rawValue).legacy")
                let decoded = try ArchitectureFormat.makeDecoder(format: format)
                    .decode(NetworkArchitecture.self, from: legacyJSON)
                XCTAssertEqual(decoded, arch, "\(preset.rawValue) v\(version)")
                XCTAssertEqual(decoded.hashValue, arch.hashValue, "\(preset.rawValue) v\(version)")
                XCTAssertEqual(decoded.architectureSummary, arch.architectureSummary, "\(preset.rawValue) v\(version)")
                XCTAssertEqual(decoded.parameterCount, arch.parameterCount, "\(preset.rawValue) v\(version)")
                XCTAssertEqual(decoded.weightTensorPlan(), arch.weightTensorPlan(), "\(preset.rawValue) v\(version)")
                XCTAssertNotNil(format.legacyLogLine, "\(preset.rawValue) v\(version): the resolution must be logged")
            }
        }
    }

    /// A pre-v5 file of a group whose main path is NOT ReLU resolves its SE
    /// activation to that group's activation — what the FC1 used then — not
    /// to ReLU.
    func testLegacyFileResolvesToEachGroupsOwnActivation() throws {
        let arch = Self.twoGroupArchitecture(group0Activation: .silu, group0SE: .silu, group1SE: .relu)
        try arch.validate()
        for version in ["4", "3", nil] as [String?] {
            let legacy = try rewritingHeader(try encodedModel(arch), version: version, stripField: true)
            let decoded = try SafetensorsModelIO.decode(legacy, valueHead: .recenterUnlessMarked, source: "old.safetensors")
            XCTAssertEqual(decoded.architecture, arch, "version \(version ?? "none")")
            XCTAssertEqual(decoded.architecture.blockGroups.map(\.seActivation), [.silu, .relu])
            let line = try XCTUnwrap(decoded.architectureFormat.legacyLogLine)
            XCTAssertTrue(line.contains("old.safetensors"), line)
            XCTAssertTrue(line.contains("block_groups[0].se_activation := silu"), line)
            XCTAssertTrue(line.contains("block_groups[1].se_activation := relu"), line)
        }
    }

    /// The legacy uniform-tower keys (pre-block-groups files) resolve the
    /// field to the tower's activation and say so in the log.
    func testLegacyUniformTowerKeysResolveToTheTowerActivation() throws {
        let legacyJSON = """
        {
          "input_encoding": "basic30", "channels": 16, "num_blocks": 2, "stem_conv_kernel_size": 3,
          "activation_function": "gelu", "block_activation_style": "pre", "block_skip_merge": "clean_add",
          "block_use_rezero": true, "rezero_alpha_init": 0.5, "block_conv1_kernel_size": 3,
          "block_conv2_kernel_size": 3, "block_se_style": "scale_and_bias", "block_se_reduction_ratio": 4,
          "policy_head_style": "intermediate_conv", "policy_pre_conv_channels": 16,
          "value_head_style": "wdl_softmax", "value_head_conv_channels": 4, "value_head_hidden_units": 16,
          "compute_data_type": "float32"
        }
        """
        let format = ArchitectureFormat.DecodeFormat(formatVersion: ArchitectureFormat.currentVersion, source: "uniform.json")
        let decoded = try ArchitectureFormat.makeDecoder(format: format)
            .decode(NetworkArchitecture.self, from: Data(legacyJSON.utf8))
        XCTAssertEqual(decoded.blockGroups.map(\.seActivation), [.gelu])
        let line = try XCTUnwrap(format.legacyLogLine)
        XCTAssertTrue(line.contains("block_groups[0].se_activation := gelu"), line)
    }

    /// Same weights: the legacy-decoded architecture builds the identical
    /// graph as the in-code one — the forward passes match exactly.
    func testLegacyDecodedArchitectureBuildsTheIdenticalForwardPass() async throws {
        try requireMetal()
        let arch = Self.twoGroupArchitecture(group0Activation: .silu, group0SE: .silu, group1SE: .relu)
        let legacy = try rewritingHeader(try encodedModel(arch), version: "4", stripField: true)
        let decodedArch = try SafetensorsModelIO.decode(legacy, valueHead: .asStored, source: "old.safetensors").architecture
        let weights = try await ChessMPSNetwork(.randomWeights, arch: arch).network.exportWeights()
        let inCode = try await forward(arch, weights)
        let fromFile = try await forward(decodedArch, weights)
        XCTAssertEqual(inCode.map(\.bitPattern), fromFile.map(\.bitPattern))
    }

    // MARK: - Version gate

    func testNewFilesAlwaysCarryTheField() throws {
        let data = try encodedModel(Self.twoGroupArchitecture(group0SE: .relu, group1SE: .relu))
        let (_, metadata) = try SafetensorsFile.decode(data)
        XCTAssertEqual(metadata["dcm_format_version"], String(ArchitectureFormat.currentVersion))
        let archJSON = try XCTUnwrap(metadata["architecture"])
        XCTAssertEqual(archJSON.components(separatedBy: "\"se_activation\":\"relu\"").count - 1, 2,
                       "encoding must write se_activation on every group, even when it equals the group's activation")
    }

    func testV5FileMissingFieldIsRejectedNamingFieldAndFile() throws {
        let arch = Self.twoGroupArchitecture(group0SE: .leakyRelu, group1SE: .relu)
        let broken = try rewritingHeader(try encodedModel(arch), version: "5", stripField: true)
        XCTAssertThrowsError(try SafetensorsModelIO.decode(broken, valueHead: .recenterUnlessMarked, source: "new.safetensors")) { error in
            XCTAssertEqual(
                error as? ArchitectureFormat.FormatError,
                .missingRequiredField(field: "se_activation", location: "block_groups[0]",
                                      formatVersion: 5, source: "new.safetensors"))
            let message = String(describing: error)
            XCTAssertTrue(message.contains("se_activation") && message.contains("new.safetensors"), message)
        }
        // A decode handed no format at all is strict (current version).
        let archJSON = try Self.removingGroupKey("se_activation", from: try JSONEncoder().encode(arch))
        XCTAssertThrowsError(try JSONDecoder().decode(NetworkArchitecture.self, from: archJSON))
    }

    func testRoundTripPreservesEveryValue() throws {
        for value in ActivationFunction.allCases {
            let arch = Self.twoGroupArchitecture(group0SE: value, group1SE: .relu)
            try arch.validate()
            let decoded = try SafetensorsModelIO.decode(try encodedModel(arch))
            XCTAssertEqual(decoded.architecture, arch, value.rawValue)
            XCTAssertEqual(decoded.architecture.blockGroups.map(\.seActivation), [value, .relu])
            XCTAssertNil(decoded.architectureFormat.legacyLogLine)
        }
    }

    func testPresetFileGate() throws {
        let arch = Self.twoGroupArchitecture(group0SE: .leakyRelu, group1SE: .relu)
        let named = NamedArchitecture(label: "se activation test", architecture: arch)
        let encoded = try JSONEncoder().encode(named)

        let current = temporaryURL("current.json")
        try encoded.write(to: current)
        XCTAssertEqual(try ArchitecturePresetStore.loadFile(at: current), named)

        // Current marker, field missing -> rejected, naming field and file.
        let missing = temporaryURL("missing.json")
        try Self.removingGroupKey("se_activation", from: encoded).write(to: missing)
        XCTAssertThrowsError(try ArchitecturePresetStore.loadFile(at: missing)) { error in
            guard case .missingRequiredField(let field, _, let version, let source)? = error as? ArchitectureFormat.FormatError else {
                return XCTFail("expected missingRequiredField, got \(error)")
            }
            XCTAssertEqual(field, "se_activation")
            XCTAssertEqual(version, ArchitectureFormat.currentVersion)
            XCTAssertEqual(source, missing.lastPathComponent)
        }

        // A format v4 preset (saved before the field existed) -> the group's activation.
        var v4Object = try XCTUnwrap(try JSONSerialization.jsonObject(
            with: try Self.removingGroupKey("se_activation", from: encoded)) as? [String: Any])
        v4Object["format_version"] = 4
        let v4 = temporaryURL("v4.json")
        try JSONSerialization.data(withJSONObject: v4Object).write(to: v4)
        let loaded = try ArchitecturePresetStore.loadFile(at: v4)
        XCTAssertEqual(loaded.architecture.blockGroups.map(\.seActivation), [.relu, .relu])
    }

    func testArchitectureFileGate() throws {
        let arch = Self.twoGroupArchitecture(group0SE: .relu, group1SE: .gelu)
        let url = temporaryURL("architecture.json")
        try ArchitectureConfig.writeTemplate(arch, to: url)
        XCTAssertEqual(try ArchitectureConfig.load(from: url), arch)

        let missing = temporaryURL("architecture-missing.json")
        try Self.removingGroupKey("se_activation", from: try Data(contentsOf: url)).write(to: missing)
        XCTAssertThrowsError(try ArchitectureConfig.load(from: missing)) { error in
            guard case .missingRequiredField(let field, _, _, _)? = error as? ArchitectureFormat.FormatError else {
                return XCTFail("expected missingRequiredField, got \(error)")
            }
            XCTAssertEqual(field, "se_activation")
        }
    }

    // MARK: - Validation and summary

    func testSELessGroupMustMatchItsActivation() throws {
        var arch = Self.twoGroupArchitecture(group0SE: .relu, group1SE: .relu)
        arch.blockGroups[1].seStyle = .none
        XCTAssertNoThrow(try arch.validate())
        arch.blockGroups[1].seActivation = .leakyRelu
        XCTAssertThrowsError(try arch.validate()) { error in
            XCTAssertEqual(
                error as? NetworkArchitectureError,
                .seActivationRequiresSE(group: 1, seActivation: .leakyRelu, activationFunction: .relu))
        }
        // Any value is valid on a group that has an SE block.
        for value in ActivationFunction.allCases {
            var withSE = Self.twoGroupArchitecture(group0SE: value, group1SE: value)
            withSE.blockGroups[0].seStyle = .attenuateOnly
            XCTAssertNoThrow(try withSE.validate(), value.rawValue)
        }
    }

    func testSummaryMarksOnlyADifferingSEActivation() throws {
        let arch = Self.twoGroupArchitecture(group0SE: .leakyRelu, group1SE: .relu)
        try arch.validate()
        let summary = arch.architectureSummary
        XCTAssertTrue(summary.contains("1x[3x3+3x3 @16, SE+/4 (fc1 leaky_relu), relu/pre, "), summary)
        XCTAssertTrue(summary.contains("2x[3x3+3x3 @16, SE/4, relu/pre, "), summary)
        var zeroBeta = arch
        zeroBeta.blockGroups[0].seBetaInit = .zero
        XCTAssertTrue(zeroBeta.architectureSummary.contains("SE+/4 β0 (fc1 leaky_relu), "), zeroBeta.architectureSummary)

        // The SE activation changes no tensor, but it is part of the identity.
        let plain = Self.twoGroupArchitecture(group0SE: .relu, group1SE: .relu)
        XCTAssertEqual(arch.parameterCount, plain.parameterCount)
        XCTAssertEqual(arch.weightTensorPlan(), plain.weightTensorPlan())
        XCTAssertNotEqual(arch, plain, "se_activation is part of the architecture's identity")
    }

    // MARK: - Builder (GPU)

    /// Policy logits followed by the value scalar for the starting position.
    private func forward(_ arch: NetworkArchitecture, _ weights: [[Float]]) async throws -> [Float] {
        let net = try ChessMPSNetwork(.randomWeights, arch: arch)
        try await net.network.loadWeights(weights)
        let board = BoardEncoder.encode(.starting, encoding: arch.inputEncoding)
        let box = SyncBox<[Float]>([])
        try await net.evaluate(board: board) { policy, value in box.value = Array(policy) + [value] }
        return box.value
    }

    private func maxAbsDifference(_ a: [Float], _ b: [Float]) throws -> Float {
        XCTAssertEqual(a.count, b.count)
        return try XCTUnwrap(zip(a, b).map { abs($0 - $1) }.max())
    }

    /// One fixed weight set, three architectures: ReLU everywhere; ReLU main
    /// path with a leaky FC1; leaky everywhere. All three must differ, which
    /// shows FC1 follows `seActivation` (first pair) and the main path follows
    /// `activationFunction` independently of it (second pair).
    func testFC1UsesSEActivationIndependentlyOfTheMainPath() async throws {
        try requireMetal()
        let reluEverywhere = Self.twoGroupArchitecture(group0SE: .relu, group1SE: .relu)
        let leakyFC1Only = Self.twoGroupArchitecture(group0SE: .leakyRelu, group1SE: .leakyRelu)
        var leakyEverywhere = Self.twoGroupArchitecture(
            group0Activation: .leakyRelu, group0SE: .leakyRelu, group1SE: .leakyRelu)
        leakyEverywhere.activationFunction = .leakyRelu
        leakyEverywhere.blockGroups[1].activationFunction = .leakyRelu
        for arch in [reluEverywhere, leakyFC1Only, leakyEverywhere] { try arch.validate() }

        let weights = try await ChessMPSNetwork(.randomWeights, arch: reluEverywhere).network.exportWeights()
        let relu = try await forward(reluEverywhere, weights)
        let fc1 = try await forward(leakyFC1Only, weights)
        let everywhere = try await forward(leakyEverywhere, weights)
        for (name, output) in [("relu", relu), ("leaky fc1", fc1), ("leaky everywhere", everywhere)] {
            XCTAssertTrue(output.allSatisfy(\.isFinite), "\(name) forward must be finite")
        }
        XCTAssertGreaterThan(try maxAbsDifference(relu, fc1), 0, "a leaky FC1 must change the forward pass")
        XCTAssertGreaterThan(try maxAbsDifference(fc1, everywhere), 0,
                             "the main-path activation must still matter when only FC1 is leaky")
    }

    /// With every FC1 bias pushed far positive, every FC1 pre-activation is
    /// positive, where ReLU and leaky ReLU are both the identity. A
    /// leaky-FC1 net must then compute what the ReLU net computes — so
    /// `seActivation` reaches no site other than FC1. With the bias pushed
    /// far negative the two must differ (test sensitivity).
    func testSEActivationReachesNoOtherSite() async throws {
        try requireMetal()
        let reluArch = Self.twoGroupArchitecture(group0SE: .relu, group1SE: .relu)
        let leakyArch = Self.twoGroupArchitecture(group0SE: .leakyRelu, group1SE: .leakyRelu)
        var weights = try await ChessMPSNetwork(.randomWeights, arch: reluArch).network.exportWeights()
        let plan = reluArch.weightTensorPlan()
        var biased = 0
        for (index, spec) in plan.enumerated() where spec.name.hasSuffix(".fc1.bias") && spec.name.contains(".se_") {
            weights[index] = [Float](repeating: 20, count: spec.elementCount)
            biased += 1
        }
        XCTAssertEqual(biased, reluArch.numBlocks, "one SE FC1 bias per block")
        let relu = try await forward(reluArch, weights)
        let leaky = try await forward(leakyArch, weights)
        XCTAssertTrue(relu.allSatisfy(\.isFinite) && leaky.allSatisfy(\.isFinite))
        let scale = try max(1, XCTUnwrap(relu.map { abs($0) }.max()))
        XCTAssertLessThanOrEqual(try maxAbsDifference(relu, leaky), 1e-6 * scale,
                                 "with positive FC1 inputs the two nets must agree; any gap means se_activation leaked elsewhere")

        // Sensitivity: the same bias pushed far negative makes leaky and ReLU
        // FC1 genuinely differ, so the agreement above is not vacuous.
        for (index, spec) in plan.enumerated() where spec.name.hasSuffix(".fc1.bias") && spec.name.contains(".se_") {
            weights[index] = [Float](repeating: -20, count: spec.elementCount)
        }
        let reluNegative = try await forward(reluArch, weights)
        let leakyNegative = try await forward(leakyArch, weights)
        XCTAssertGreaterThan(try maxAbsDifference(reluNegative, leakyNegative), 0,
                             "with negative FC1 inputs a leaky FC1 must change the forward pass")
    }

    func testTrainingStepIsFiniteWithLeakyFC1() async throws {
        try requireMetal()
        let arch = Self.twoGroupArchitecture(group0SE: .leakyRelu, group1SE: .leakyRelu)
        let trainer = try ChessTrainer(learningRate: 1e-2, lrWarmupSteps: 0, arch: arch)
        let before = try await trainer.network.exportWeights()
        let timing = try await trainer.trainStep(batchSize: 16)
        XCTAssertTrue(timing.policyLoss.isFinite, "policy loss must be finite")
        XCTAssertTrue(timing.valueLoss.isFinite, "value loss must be finite")
        let after = try await trainer.network.exportWeights()
        let plan = arch.weightTensorPlan()
        for (index, spec) in plan.enumerated() where spec.name.contains(".fc1.weight") && spec.name.contains(".se_") {
            XCTAssertTrue(after[index].allSatisfy(\.isFinite), "\(spec.name) must stay finite")
            XCTAssertNotEqual(before[index], after[index], "\(spec.name) must be updated by the step")
        }
    }

    // MARK: - --derive-model

    private func derive(_ source: Data, _ operations: [any DeriveOperation]) throws -> ModelDerivation.Result {
        try ModelDerivation.derive(
            sourceData: source, sourceName: "source.safetensors", operations: operations,
            newModelID: "20261001-2-DRV1", createdAtUnix: 1_790_000_100, build: "test", invocationArguments: ["test"])
    }

    private func assertEveryTensorBitExact(_ sourceData: Data, _ derivedData: Data) throws {
        let (sourceTensors, _) = try SafetensorsFile.decode(sourceData)
        let (derivedTensors, _) = try SafetensorsFile.decode(derivedData)
        XCTAssertEqual(sourceTensors.map(\.name), derivedTensors.map(\.name))
        for (before, after) in zip(sourceTensors, derivedTensors) {
            XCTAssertEqual(before.shape, after.shape, before.name)
            XCTAssertEqual(before.data.map(\.bitPattern), after.data.map(\.bitPattern), "\(before.name) must be bit-exact")
        }
    }

    func testDeriveSetsOnlyTheFieldAndCopiesEveryTensor() throws {
        let source = Self.twoGroupArchitecture(group0SE: .relu, group1SE: .relu)
        let sourceData = try encodedModel(source)
        let result = try derive(sourceData, [SetSEActivationDeriveOperation(value: .leakyRelu, groupIndices: nil)])

        var expected = source
        for index in expected.blockGroups.indices { expected.blockGroups[index].seActivation = .leakyRelu }
        XCTAssertEqual(result.targetArchitecture, expected, "only se_activation may change")
        XCTAssertTrue(result.rewrites.isEmpty, "activations have no parameters")
        try assertEveryTensorBitExact(sourceData, result.data)

        XCTAssertEqual(result.record.operations.map(\.operation), ["set-se-activation"])
        XCTAssertEqual(result.record.operations[0].arguments, ["value": "leaky_relu", "groups": "all with SE"])
        XCTAssertEqual(result.record.operations[0].changedArchitectureFields, ["block_groups[].se_activation"])
        XCTAssertEqual(result.record.operations[0].rewrittenTensors, [])
        let reloaded = try SafetensorsModelIO.decode(result.data)
        XCTAssertEqual(reloaded.architecture, expected)
        XCTAssertNil(reloaded.architectureFormat.legacyLogLine)
    }

    func testDeriveSelectedGroupOnly() throws {
        let source = Self.twoGroupArchitecture(group0SE: .relu, group1SE: .relu)
        let sourceData = try encodedModel(source)
        let result = try derive(sourceData, [SetSEActivationDeriveOperation(value: .leakyRelu, groupIndices: [1])])
        XCTAssertEqual(result.targetArchitecture.blockGroups.map(\.seActivation), [.relu, .leakyRelu])
        XCTAssertEqual(result.record.operations[0].arguments["groups"], "1")
        try assertEveryTensorBitExact(sourceData, result.data)
    }

    /// A legacy (v4) source derives fine, and the derived file is a current-
    /// version file that states the field.
    func testDeriveFromALegacySource() throws {
        let source = Self.twoGroupArchitecture(group0SE: .relu, group1SE: .relu)
        let legacyData = try rewritingHeader(try encodedModel(source), version: "4", stripField: true)
        let result = try derive(legacyData, [SetSEActivationDeriveOperation(value: .leakyRelu, groupIndices: [0])])
        XCTAssertEqual(result.record.sourceFormatVersion, 4)
        XCTAssertEqual(result.targetArchitecture.blockGroups.map(\.seActivation), [.leakyRelu, .relu])
        let (_, metadata) = try SafetensorsFile.decode(result.data)
        XCTAssertEqual(metadata["dcm_format_version"], String(ArchitectureFormat.currentVersion))
        XCTAssertTrue(try XCTUnwrap(metadata["architecture"]).contains("\"se_activation\""))
    }

    func testDeriveRefusesInapplicableRequests() throws {
        let source = Self.twoGroupArchitecture(group0SE: .relu, group1SE: .relu)
        let sourceData = try encodedModel(source)
        for operation in [
            SetSEActivationDeriveOperation(value: .relu, groupIndices: nil),         // nothing to change
            SetSEActivationDeriveOperation(value: .leakyRelu, groupIndices: [2]),    // out of range
        ] {
            XCTAssertThrowsError(try derive(sourceData, [operation])) { error in
                guard case .operationNotApplicable? = error as? ModelDerivation.DeriveError else {
                    return XCTFail("expected operationNotApplicable, got \(error)")
                }
            }
        }
        var noSE = source
        noSE.blockGroups[1].seStyle = .none
        let noSEData = try encodedModel(noSE)
        XCTAssertThrowsError(try derive(noSEData, [SetSEActivationDeriveOperation(value: .leakyRelu, groupIndices: [1])])) { error in
            guard case .operationNotApplicable? = error as? ModelDerivation.DeriveError else {
                return XCTFail("expected operationNotApplicable, got \(error)")
            }
        }
        // Without --group, only the SE group changes.
        let allSE = try derive(noSEData, [SetSEActivationDeriveOperation(value: .leakyRelu, groupIndices: nil)])
        XCTAssertEqual(allSE.targetArchitecture.blockGroups.map(\.seActivation), [.leakyRelu, .relu])

        XCTAssertThrowsError(try SetSEActivationDeriveOperation.kind.make("leaky", nil))
        XCTAssertNoThrow(try SetSEActivationDeriveOperation.kind.make("leaky_relu", [0]))
    }

    /// `--set-activation` changes the main path only: an SE group keeps its
    /// FC1 activation, an SE-less group's follows (validation requires it).
    /// Both flags together give "leaky everywhere".
    func testSetActivationLeavesSEGroupsFC1Alone() throws {
        var source = Self.twoGroupArchitecture(group0SE: .relu, group1SE: .relu)
        source.blockGroups[1].seStyle = .none
        let sourceData = try encodedModel(source)

        let mainOnly = try derive(sourceData, [SetActivationDeriveOperation(value: .leakyRelu)])
        XCTAssertEqual(mainOnly.targetArchitecture.blockGroups.map(\.activationFunction), [.leakyRelu, .leakyRelu])
        XCTAssertEqual(mainOnly.targetArchitecture.blockGroups.map(\.seActivation), [.relu, .leakyRelu])
        try assertEveryTensorBitExact(sourceData, mainOnly.data)

        let everywhere = try derive(sourceData, [
            SetActivationDeriveOperation(value: .leakyRelu),
            SetSEActivationDeriveOperation(value: .leakyRelu, groupIndices: nil),
        ])
        XCTAssertEqual(everywhere.targetArchitecture.activationFunction, .leakyRelu)
        XCTAssertTrue(everywhere.targetArchitecture.blockGroups.allSatisfy {
            $0.activationFunction == .leakyRelu && $0.seActivation == .leakyRelu
        })
        try assertEveryTensorBitExact(sourceData, everywhere.data)
    }

    // MARK: - Build screen

    /// The Build screen applies the shared rule (`BlockGroup.setActivationFunction`):
    /// a group with an SE block keeps its SE activation when its main-path
    /// activation changes; an SE-less group's SE activation moves with it.
    @MainActor
    func testBuildScreenActivationEditAppliesTheSharedRule() {
        let model = BuildNewModelModel(NamedArchitecture(
            label: "test", architecture: Self.twoGroupArchitecture(group0SE: .relu, group1SE: .relu)))
        let first = model.blockGroupDrafts[0]
        let second = model.blockGroupDrafts[1]
        // A group with an SE block: the SE activation stays put.
        first.activationFunction = .silu
        XCTAssertEqual(first.group.activationFunction, .silu)
        XCTAssertEqual(first.group.seActivation, .relu)
        // Set on its own, it stays through later activation changes too.
        first.group.seActivation = .leakyRelu
        first.activationFunction = .relu
        XCTAssertEqual(first.group.activationFunction, .relu)
        XCTAssertEqual(first.group.seActivation, .leakyRelu)
        XCTAssertTrue(model.isValid)
        XCTAssertTrue(model.summary.contains("(fc1 leaky_relu)"), model.summary)
        // An SE-less group's SE activation moves with it, keeping it valid.
        second.group.seStyle = .none
        second.activationFunction = .gelu
        XCTAssertEqual(second.group.seActivation, .gelu)
        XCTAssertTrue(model.isValid, model.validationError ?? "")
        // The other group is untouched throughout.
        XCTAssertEqual(first.group.seActivation, .leakyRelu)
    }

    /// The same activation edit made on the Build screen and through
    /// `--derive-model --set-activation` yields the same architecture, for a
    /// group with an SE block and an SE-less one alike.
    @MainActor
    func testBuildScreenAndDeriveApplyTheSameActivationRule() throws {
        var source = Self.twoGroupArchitecture(group0SE: .relu, group1SE: .relu)
        source.blockGroups[1].seStyle = .none
        try source.validate()
        let derived = try SetActivationDeriveOperation(value: .leakyRelu).apply(to: source)

        let model = BuildNewModelModel(NamedArchitecture(label: "test", architecture: source))
        model.activationFunction = .leakyRelu
        for draft in model.blockGroupDrafts {
            draft.activationFunction = .leakyRelu
        }
        XCTAssertEqual(model.architecture, derived)
        XCTAssertEqual(derived.blockGroups[0].seActivation, .relu)
        XCTAssertEqual(derived.blockGroups[1].seActivation, .leakyRelu)
    }
}
