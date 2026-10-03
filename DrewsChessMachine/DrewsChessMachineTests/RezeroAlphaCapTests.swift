//
//  RezeroAlphaCapTests.swift
//  DrewsChessMachineTests
//
//  The per-block-group `rezero_alpha_cap` (architecture format v6): the
//  asymptote C of the forward ReZero soft bound `C·tanh(α/C)`, stored
//  explicitly instead of derived as `rezero_alpha_init ×
//  rezeroTanhCeilingMultiple`. Decoupling the two is what makes the ReZero
//  paper's zero init legal — a derived C = 0 divided by zero.
//
//  Pinned here:
//  - Format v6 gate: a v6 file must state the field; an older file (v5, v4,
//    v3, unversioned, legacy uniform-tower keys) without it resolves to
//    `rezero_alpha_init × rezeroTanhCeilingMultiple` — the C the engine
//    computed before — and says so in the legacy log line. Every existing
//    model and preset therefore decodes to the identical architecture: same
//    equality, hash, summary, parameter count, tensor plan, legacy archHash
//    and forward pass.
//  - Encoding always writes the field; an explicit cap round-trips.
//  - Validation: cap finite and > 0, init finite and >= 0 (zero legal), only
//    on groups with ReZero; an init above the cap is allowed.
//  - Summary shows the explicit cap.
//  - `--set-rezero-alpha-init` rewrites every affected α tensor to exactly
//    the value and sets the field; `--set-rezero-alpha-cap` sets only the
//    field; both refuse groups without ReZero.
//  - The Build screen's cap-follows-init rule and depth-mismatch warning.
//
//  The forward / train-step cases are GPU-backed and skip without Metal; the
//  format, validation and derivation cases are pure.
//

import XCTest
import Metal
@testable import DrewsChessMachine

final class RezeroAlphaCapTests: XCTestCase {

    // MARK: - Fixtures

    /// Small fp32 pre-act tower with two ReZero groups (inits `init0` and
    /// `init1`, caps derived the legacy way) so group selection and mixed
    /// settings are both exercisable, cheap to build.
    private static func twoGroupArchitecture(init0: Float = 0.5, init1: Float = 0.25) -> NetworkArchitecture {
        var arch = NetworkArchitecture(
            inputEncoding: .basic30, channels: 16, numBlocks: 1, stemConvKernelSize: 3,
            activationFunction: .relu, blockActivationStyle: .pre,
            blockSkipMerge: .cleanAdd, blockUseRezero: true, rezeroAlphaInit: init0,
            blockConv1KernelSize: 3, blockConv2KernelSize: 3,
            blockSeStyle: .scaleAndBias, blockSeReductionRatio: 4,
            policyHeadStyle: .intermediateConv, policyPreConvChannels: 16,
            valueHeadStyle: .wdlSoftmax, valueHeadConvChannels: 4, valueHeadHiddenUnits: 16,
            computeDataType: .float32
        )
        var second = arch.blockGroups[0]
        second.count = 2
        second.rezeroAlphaInit = init1
        second.rezeroAlphaCap = init1
        arch.blockGroups.append(second)
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

    private func encodedModel(_ arch: NetworkArchitecture, modelID: String = "20261002-1-SRCE") throws -> Data {
        let weights = arch.weightTensorPlan().enumerated().map { tensorIndex, spec in
            (0..<spec.elementCount).map { Float(tensorIndex * 7 + $0 % 13) * 0.125 + 0.0625 }
        }
        let meta = ModelCheckpointMetadata(creator: "test", trainingStep: nil, parentModelID: "", notes: "fixture")
        return try SafetensorsModelIO.encode(
            modelID: modelID, createdAtUnix: 1_790_000_000, metadata: meta, weights: weights,
            architecture: arch, includesVelocity: false)
    }

    /// Re-encode `data` with `dcm_format_version` set to `version` (nil =
    /// removed) and, when `stripCap`, `rezero_alpha_cap` removed from the
    /// embedded architecture.
    private func rewritingHeader(_ data: Data, version: String?, stripCap: Bool) throws -> Data {
        let (tensors, decodedMetadata) = try SafetensorsFile.decode(data)
        var metadata = decodedMetadata
        metadata.removeValue(forKey: SafetensorsFile.contentHashKey)
        metadata["dcm_format_version"] = version
        if stripCap {
            let archJSON = try XCTUnwrap(metadata["architecture"])
            metadata["architecture"] = String(
                decoding: try Self.removingGroupKey("rezero_alpha_cap", from: Data(archJSON.utf8)), as: UTF8.self)
        }
        return try SafetensorsFile.encode(tensors: tensors, metadata: metadata)
    }

    private func temporaryURL(_ name: String) -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("rezero-cap-\(UUID().uuidString)-\(name)")
        addTeardownBlock {
            do { try FileManager.default.removeItem(at: url) } catch {}
        }
        return url
    }

    // MARK: - Format version

    func testCurrentVersionRequiresTheCap() {
        XCTAssertEqual(ArchitectureFormat.rezeroAlphaCapRequiredFromVersion, 6)
        XCTAssertGreaterThanOrEqual(ArchitectureFormat.currentVersion, ArchitectureFormat.rezeroAlphaCapRequiredFromVersion)
        XCTAssertGreaterThan(ArchitectureFormat.rezeroAlphaCapRequiredFromVersion, ArchitectureFormat.seActivationRequiredFromVersion)
        XCTAssertEqual(SafetensorsModelIO.formatVersion, String(ArchitectureFormat.currentVersion))
    }

    // MARK: - Existing architectures keep their identity

    /// Every built-in preset's cap is its init (bit for bit), and a legacy
    /// file of it (format v5 / v4 / v3, no `rezero_alpha_cap`) decodes to
    /// exactly the preset: equal, same hash, same summary, same parameter
    /// count, same tensor plan, same legacy `.dcmmodel` archHash — with the
    /// resolution logged. This is the identity guarantee for every model
    /// saved before the field existed.
    func testExistingPresetsKeepTheirIdentity() throws {
        for preset in NetworkArchitecture.Preset.allCases {
            let arch = NetworkArchitecture.preset(preset)
            for group in arch.blockGroups {
                XCTAssertEqual(group.rezeroAlphaCap.bitPattern, group.rezeroAlphaInit.bitPattern, preset.rawValue)
                XCTAssertEqual(group.rezeroTanhCeiling.bitPattern,
                               (Double(group.rezeroAlphaInit) * NetworkArchitecture.rezeroTanhCeilingMultiple).bitPattern,
                               "\(preset.rawValue): the ceiling must equal the pre-field formula exactly")
            }
            if let legacyHash = NetworkArchitecture.legacyArchHash(for: preset) {
                XCTAssertEqual(ModelCheckpointFile.archHash(for: arch), legacyHash, preset.rawValue)
            }
            let legacyJSON = try Self.removingGroupKey("rezero_alpha_cap", from: try JSONEncoder().encode(arch))
            for version in [3, 4, 5] {
                let format = ArchitectureFormat.DecodeFormat(formatVersion: version, source: "\(preset.rawValue).legacy")
                let decoded = try ArchitectureFormat.makeDecoder(format: format)
                    .decode(NetworkArchitecture.self, from: legacyJSON)
                XCTAssertEqual(decoded, arch, "\(preset.rawValue) v\(version)")
                XCTAssertEqual(decoded.hashValue, arch.hashValue, "\(preset.rawValue) v\(version)")
                XCTAssertEqual(decoded.architectureSummary, arch.architectureSummary, "\(preset.rawValue) v\(version)")
                XCTAssertEqual(decoded.parameterCount, arch.parameterCount, "\(preset.rawValue) v\(version)")
                XCTAssertEqual(decoded.weightTensorPlan(), arch.weightTensorPlan(), "\(preset.rawValue) v\(version)")
                if let legacyHash = NetworkArchitecture.legacyArchHash(for: preset) {
                    XCTAssertEqual(ModelCheckpointFile.archHash(for: decoded), legacyHash, "\(preset.rawValue) v\(version)")
                }
                let line = try XCTUnwrap(format.legacyLogLine, "\(preset.rawValue) v\(version): the resolution must be logged")
                XCTAssertTrue(line.contains("block_groups[0].rezero_alpha_cap := "), line)
            }
        }
    }

    /// Legacy safetensors (v5, v4, v3, unversioned) resolve each group's cap
    /// to its own init and name every resolution in the log line.
    func testLegacyFileResolvesCapToEachGroupsInit() throws {
        let arch = Self.twoGroupArchitecture(init0: 0.5, init1: 0.25)
        try arch.validate()
        for version in ["5", "4", "3", nil] as [String?] {
            let legacy = try rewritingHeader(try encodedModel(arch), version: version, stripCap: true)
            let decoded = try SafetensorsModelIO.decode(legacy, valueHead: .recenterUnlessMarked, source: "old.safetensors")
            XCTAssertEqual(decoded.architecture, arch, "version \(version ?? "none")")
            XCTAssertEqual(decoded.architecture.blockGroups.map(\.rezeroAlphaCap), [0.5, 0.25])
            let line = try XCTUnwrap(decoded.architectureFormat.legacyLogLine)
            XCTAssertTrue(line.contains("old.safetensors"), line)
            XCTAssertTrue(line.contains("block_groups[0].rezero_alpha_cap := 0.5"), line)
            XCTAssertTrue(line.contains("block_groups[1].rezero_alpha_cap := 0.25"), line)
        }
    }

    /// The pre-block-groups uniform-tower keys resolve the cap to the init
    /// and say so, whatever version the carrier states.
    func testLegacyUniformTowerKeysResolveCapToTheInit() throws {
        let legacyJSON = """
        {
          "input_encoding": "basic30", "channels": 16, "num_blocks": 2, "stem_conv_kernel_size": 3,
          "activation_function": "relu", "block_activation_style": "pre", "block_skip_merge": "clean_add",
          "block_use_rezero": true, "rezero_alpha_init": 0.375, "block_conv1_kernel_size": 3,
          "block_conv2_kernel_size": 3, "block_se_style": "scale_and_bias", "block_se_reduction_ratio": 4,
          "policy_head_style": "intermediate_conv", "policy_pre_conv_channels": 16,
          "value_head_style": "wdl_softmax", "value_head_conv_channels": 4, "value_head_hidden_units": 16,
          "compute_data_type": "float32"
        }
        """
        let format = ArchitectureFormat.DecodeFormat(formatVersion: ArchitectureFormat.currentVersion, source: "uniform.json")
        let decoded = try ArchitectureFormat.makeDecoder(format: format)
            .decode(NetworkArchitecture.self, from: Data(legacyJSON.utf8))
        XCTAssertEqual(decoded.blockGroups.map(\.rezeroAlphaCap), [0.375])
        let line = try XCTUnwrap(format.legacyLogLine)
        XCTAssertTrue(line.contains("block_groups[0].rezero_alpha_cap := 0.375"), line)
    }

    /// Same weights: the legacy-decoded architecture builds the identical
    /// graph as the in-code one — the forward passes match bit for bit.
    func testLegacyDecodedArchitectureBuildsTheIdenticalForwardPass() async throws {
        try requireMetal()
        let arch = Self.twoGroupArchitecture()
        let legacy = try rewritingHeader(try encodedModel(arch), version: "5", stripCap: true)
        let decodedArch = try SafetensorsModelIO.decode(legacy, valueHead: .asStored, source: "old.safetensors").architecture
        let weights = try await ChessMPSNetwork(.randomWeights(initSeed: 1), arch: arch).network.exportWeights()
        let inCode = try await forward(arch, weights)
        let fromFile = try await forward(decodedArch, weights)
        XCTAssertEqual(inCode.map(\.bitPattern), fromFile.map(\.bitPattern))
    }

    // MARK: - Version gate

    func testNewFilesAlwaysCarryTheCap() throws {
        var arch = Self.twoGroupArchitecture()
        arch.blockGroups[1].useRezero = false
        let data = try encodedModel(arch)
        let (_, metadata) = try SafetensorsFile.decode(data)
        XCTAssertEqual(metadata["dcm_format_version"], String(ArchitectureFormat.currentVersion))
        let archJSON = try XCTUnwrap(metadata["architecture"])
        XCTAssertEqual(archJSON.components(separatedBy: "\"rezero_alpha_cap\"").count - 1, 2,
                       "encoding must write rezero_alpha_cap on every group, even at its legacy value and without ReZero")
    }

    func testV6FileMissingCapIsRejectedNamingFieldAndFile() throws {
        let arch = Self.twoGroupArchitecture()
        let broken = try rewritingHeader(try encodedModel(arch), version: "6", stripCap: true)
        XCTAssertThrowsError(try SafetensorsModelIO.decode(broken, valueHead: .recenterUnlessMarked, source: "new.safetensors")) { error in
            XCTAssertEqual(
                error as? ArchitectureFormat.FormatError,
                .missingRequiredField(field: "rezero_alpha_cap", location: "block_groups[0]",
                                      formatVersion: 6, source: "new.safetensors"))
            let message = String(describing: error)
            XCTAssertTrue(message.contains("rezero_alpha_cap") && message.contains("new.safetensors"), message)
        }
        // A decode handed no format at all is strict (current version).
        let archJSON = try Self.removingGroupKey("rezero_alpha_cap", from: try JSONEncoder().encode(arch))
        XCTAssertThrowsError(try JSONDecoder().decode(NetworkArchitecture.self, from: archJSON))
    }

    func testExplicitCapRoundTrips() throws {
        var arch = Self.twoGroupArchitecture()
        arch.blockGroups[0].rezeroAlphaInit = 0
        arch.blockGroups[0].rezeroAlphaCap = 1
        arch.blockGroups[1].rezeroAlphaCap = 0.75
        try arch.validate()
        let decoded = try SafetensorsModelIO.decode(try encodedModel(arch))
        XCTAssertEqual(decoded.architecture, arch)
        XCTAssertEqual(decoded.architecture.blockGroups.map(\.rezeroAlphaInit), [0, 0.25])
        XCTAssertEqual(decoded.architecture.blockGroups.map(\.rezeroAlphaCap), [1, 0.75])
        XCTAssertNil(decoded.architectureFormat.legacyLogLine)
    }

    func testPresetAndArchitectureFileGate() throws {
        var arch = Self.twoGroupArchitecture()
        arch.blockGroups[0].rezeroAlphaInit = 0
        arch.blockGroups[0].rezeroAlphaCap = 1
        let named = NamedArchitecture(label: "rezero cap test", architecture: arch)
        let encoded = try JSONEncoder().encode(named)

        let current = temporaryURL("current.json")
        try encoded.write(to: current)
        XCTAssertEqual(try ArchitecturePresetStore.loadFile(at: current), named)

        // Current marker, cap missing -> rejected, naming field and file.
        let missing = temporaryURL("missing.json")
        try Self.removingGroupKey("rezero_alpha_cap", from: encoded).write(to: missing)
        XCTAssertThrowsError(try ArchitecturePresetStore.loadFile(at: missing)) { error in
            guard case .missingRequiredField(let field, _, let version, let source)? = error as? ArchitectureFormat.FormatError else {
                return XCTFail("expected missingRequiredField, got \(error)")
            }
            XCTAssertEqual(field, "rezero_alpha_cap")
            XCTAssertEqual(version, ArchitectureFormat.currentVersion)
            XCTAssertEqual(source, missing.lastPathComponent)
        }

        // A format v5 preset (saved before the field existed) -> cap = init.
        func v5Preset(_ named: NamedArchitecture, _ fileName: String) throws -> URL {
            var object = try XCTUnwrap(try JSONSerialization.jsonObject(
                with: try Self.removingGroupKey("rezero_alpha_cap", from: try JSONEncoder().encode(named))) as? [String: Any])
            object["format_version"] = 5
            let url = temporaryURL(fileName)
            try JSONSerialization.data(withJSONObject: object).write(to: url)
            return url
        }
        let legacyNamed = NamedArchitecture(label: "legacy cap test", architecture: Self.twoGroupArchitecture())
        let loaded = try ArchitecturePresetStore.loadFile(at: try v5Preset(legacyNamed, "v5.json"))
        XCTAssertEqual(loaded, legacyNamed)
        XCTAssertEqual(loaded.architecture.blockGroups.map(\.rezeroAlphaCap), [0.5, 0.25])
        // A v5 file can never hold a zero init (v5 validation refused it), but
        // if one is hand-made the legacy rule resolves its cap to 0 — what a
        // v5 engine would have built — and validation, not decoding, refuses
        // it.
        let zeroInitV5 = try v5Preset(named, "v5-zero.json")
        XCTAssertThrowsError(try ArchitecturePresetStore.loadFile(at: zeroInitV5)) { error in
            guard case .invalid? = error as? ArchitecturePresetStore.StoreError else {
                return XCTFail("expected StoreError.invalid, got \(error)")
            }
        }

        // architecture.json carries the marker and the field.
        let url = temporaryURL("architecture.json")
        try ArchitectureConfig.writeTemplate(arch, to: url)
        XCTAssertEqual(try ArchitectureConfig.load(from: url), arch)
        let missingArch = temporaryURL("architecture-missing.json")
        try Self.removingGroupKey("rezero_alpha_cap", from: try Data(contentsOf: url)).write(to: missingArch)
        XCTAssertThrowsError(try ArchitectureConfig.load(from: missingArch)) { error in
            guard case .missingRequiredField(let field, _, _, _)? = error as? ArchitectureFormat.FormatError else {
                return XCTFail("expected missingRequiredField, got \(error)")
            }
            XCTAssertEqual(field, "rezero_alpha_cap")
        }
    }

    // MARK: - Validation and summary

    func testValidationAcceptsZeroInitWithAPositiveCap() throws {
        var arch = Self.twoGroupArchitecture()
        arch.blockGroups[0].rezeroAlphaInit = 0
        arch.blockGroups[0].rezeroAlphaCap = 1
        XCTAssertNoThrow(try arch.validate())
        // An init above the cap is allowed: the forward starts saturated.
        arch.blockGroups[1].rezeroAlphaInit = 2
        arch.blockGroups[1].rezeroAlphaCap = 0.5
        XCTAssertNoThrow(try arch.validate())
    }

    func testValidationRejectsABadCap() {
        for bad: Float in [0, -0.5, .nan, .infinity, -.infinity] {
            var arch = Self.twoGroupArchitecture()
            arch.blockGroups[0].rezeroAlphaInit = 0
            arch.blockGroups[0].rezeroAlphaCap = bad
            XCTAssertThrowsError(try arch.validate(), "cap \(bad)") { error in
                guard case .mustBeFinitePositive(let field, _)? = error as? NetworkArchitectureError else {
                    return XCTFail("expected mustBeFinitePositive, got \(error)")
                }
                XCTAssertEqual(field, "blockGroups[0].rezeroAlphaCap")
            }
        }
    }

    func testValidationRejectsABadInit() {
        for bad: Float in [-0.5, -1e-7, .nan, .infinity, -.infinity] {
            var arch = Self.twoGroupArchitecture()
            arch.blockGroups[1].rezeroAlphaInit = bad
            XCTAssertThrowsError(try arch.validate(), "init \(bad)") { error in
                guard case .mustBeFiniteNonNegative(let field, _)? = error as? NetworkArchitectureError else {
                    return XCTFail("expected mustBeFiniteNonNegative, got \(error)")
                }
                XCTAssertEqual(field, "blockGroups[1].rezeroAlphaInit")
            }
        }
    }

    /// Without ReZero nothing reads either value, so neither is checked —
    /// the same treatment the init has always had.
    func testValidationIgnoresBothValuesWithoutRezero() {
        var arch = Self.twoGroupArchitecture()
        arch.blockGroups[0].useRezero = false
        arch.blockGroups[0].rezeroAlphaInit = -1
        arch.blockGroups[0].rezeroAlphaCap = 0
        XCTAssertNoThrow(try arch.validate())
    }

    func testSummaryShowsTheExplicitCap() throws {
        var arch = Self.twoGroupArchitecture()
        let legacySummary = arch.architectureSummary
        XCTAssertTrue(legacySummary.contains("ReZero(0.5·tanh≤0.5)"), legacySummary)
        XCTAssertTrue(legacySummary.contains("ReZero(0.25·tanh≤0.25)"), legacySummary)

        arch.blockGroups[0].rezeroAlphaInit = 0
        arch.blockGroups[0].rezeroAlphaCap = 1
        try arch.validate()
        XCTAssertEqual(arch.blockGroups[0].rezeroTanhCeiling, 1)
        XCTAssertTrue(arch.architectureSummary.contains("ReZero(0·tanh≤1)"), arch.architectureSummary)
        XCTAssertEqual(NetworkArchitecture.rezeroDescription(arch.blockGroups[0]), "ReZero(0·tanh≤1)")

        // The cap changes no tensor, but it is part of the identity.
        var capOnly = Self.twoGroupArchitecture()
        capOnly.blockGroups[0].rezeroAlphaCap = 1
        let plain = Self.twoGroupArchitecture()
        XCTAssertEqual(capOnly.parameterCount, plain.parameterCount)
        XCTAssertEqual(capOnly.weightTensorPlan(), plain.weightTensorPlan())
        XCTAssertNotEqual(capOnly, plain, "rezero_alpha_cap is part of the architecture's identity")
        XCTAssertTrue(capOnly.architectureSummary.contains("ReZero(0.5·tanh≤1)"), capOnly.architectureSummary)
    }

    // MARK: - Builder (GPU)

    /// Policy logits followed by the value scalar for the starting position.
    private func forward(_ arch: NetworkArchitecture, _ weights: [[Float]]) async throws -> [Float] {
        let net = try ChessMPSNetwork(.randomWeights(initSeed: 2), arch: arch)
        try await net.network.loadWeights(weights)
        let board = BoardEncoder.encode(.starting, encoding: arch.inputEncoding)
        let box = SyncBox<[Float]>([])
        try await net.evaluate(board: board) { policy, value in box.value = Array(policy) + [value] }
        return box.value
    }

    /// The forward reads the explicit cap: one weight set, two caps, two
    /// different outputs.
    func testForwardReadsTheExplicitCap() async throws {
        try requireMetal()
        let legacyCap = Self.twoGroupArchitecture()
        var widerCap = legacyCap
        widerCap.blockGroups[0].rezeroAlphaCap = 2
        let weights = try await ChessMPSNetwork(.randomWeights(initSeed: 3), arch: legacyCap).network.exportWeights()
        let a = try await forward(legacyCap, weights)
        let b = try await forward(widerCap, weights)
        XCTAssertTrue(a.allSatisfy(\.isFinite) && b.allSatisfy(\.isFinite))
        XCTAssertNotEqual(a.map(\.bitPattern), b.map(\.bitPattern), "the cap must reach the forward pass")
    }

    /// A zero-init net builds with every α exactly 0, runs a finite forward
    /// and training step, and α leaves zero on the first step — the soft
    /// bound passes α's gradient through at α = 0.
    func testZeroInitTrainsAndAlphaMovesOnTheFirstStep() async throws {
        try requireMetal()
        var arch = Self.twoGroupArchitecture()
        for index in arch.blockGroups.indices {
            arch.blockGroups[index].rezeroAlphaInit = 0
            arch.blockGroups[index].rezeroAlphaCap = 1
        }
        try arch.validate()
        let trainer = try ChessTrainer(learningRate: 1e-2, lrWarmupSteps: 0, arch: arch)
        let plan = arch.weightTensorPlan()
        let alphaIndices = plan.indices.filter { plan[$0].name.hasSuffix(".rezero_alpha") }
        XCTAssertEqual(alphaIndices.count, arch.numBlocks)
        let before = try await trainer.network.exportWeights()
        for index in alphaIndices {
            XCTAssertEqual(before[index].map(\.bitPattern), [Float(0).bitPattern], "\(plan[index].name) must start at exactly 0")
        }
        let timing = try await trainer.trainStep(batchSize: 16)
        XCTAssertTrue(timing.policyLoss.isFinite, "policy loss must be finite")
        XCTAssertTrue(timing.valueLoss.isFinite, "value loss must be finite")
        let after = try await trainer.network.exportWeights()
        for index in alphaIndices {
            let alpha = try XCTUnwrap(after[index].first)
            XCTAssertTrue(alpha.isFinite, "\(plan[index].name) must stay finite")
            XCTAssertNotEqual(alpha, 0, "\(plan[index].name) must receive gradient on step 1")
        }
    }

    // MARK: - --derive-model

    private func derive(_ source: Data, _ operations: [any DeriveOperation]) throws -> ModelDerivation.Result {
        try ModelDerivation.derive(
            sourceData: source, sourceName: "source.safetensors", operations: operations,
            newModelID: "20261002-2-DRV1", createdAtUnix: 1_790_000_100, build: "test")
    }

    /// Every tensor of `derivedData` bit-exact to `sourceData` except the
    /// names in `rewritten`, which must all hold exactly `value`.
    private func assertTensors(_ sourceData: Data, _ derivedData: Data, rewritten: Set<String>, value: Float) throws {
        let (sourceTensors, _) = try SafetensorsFile.decode(sourceData)
        let (derivedTensors, _) = try SafetensorsFile.decode(derivedData)
        XCTAssertEqual(sourceTensors.map(\.name), derivedTensors.map(\.name))
        for (before, after) in zip(sourceTensors, derivedTensors) {
            XCTAssertEqual(before.shape, after.shape, before.name)
            if rewritten.contains(before.name) {
                XCTAssertEqual(after.data.map(\.bitPattern), [Float](repeating: value, count: before.data.count).map(\.bitPattern),
                               "\(before.name) must hold exactly \(value)")
            } else {
                XCTAssertEqual(before.data.map(\.bitPattern), after.data.map(\.bitPattern), "\(before.name) must be bit-exact")
            }
        }
    }

    /// The experiment's derivation on a legacy (v3, no cap) fresh-net-like
    /// source: every α tensor becomes exactly 0, the init field 0, the cap 1,
    /// everything else bit-exact, and the derived file is a v6 file stating
    /// the cap.
    func testDeriveZeroInitWithCapFromALegacySource() throws {
        let source = Self.twoGroupArchitecture()
        let sourceData = try rewritingHeader(try encodedModel(source), version: "3", stripCap: true)
        let result = try derive(sourceData, [
            SetRezeroAlphaInitDeriveOperation(value: 0, groupIndices: nil),
            SetRezeroAlphaCapDeriveOperation(value: 1, groupIndices: nil),
        ])
        var expected = source
        for index in expected.blockGroups.indices {
            expected.blockGroups[index].rezeroAlphaInit = 0
            expected.blockGroups[index].rezeroAlphaCap = 1
        }
        XCTAssertEqual(result.targetArchitecture, expected, "only the ReZero init and cap may change")
        XCTAssertEqual(result.record.sourceFormatVersion, 3)
        let alphaNames: Set<String> = Set((0..<source.numBlocks).map { "blocks.\($0).rezero_alpha" })
        try assertTensors(sourceData, result.data, rewritten: alphaNames, value: 0)

        XCTAssertEqual(result.record.operations.map(\.operation), ["set-rezero-alpha-init", "set-rezero-alpha-cap"])
        XCTAssertEqual(result.record.operations[0].arguments, ["value": "0.0", "groups": "all with ReZero"])
        XCTAssertEqual(result.record.operations[0].changedArchitectureFields, ["block_groups[].rezero_alpha_init"])
        XCTAssertEqual(Set(result.record.operations[0].rewrittenTensors), alphaNames)
        XCTAssertEqual(result.record.operations[1].changedArchitectureFields, ["block_groups[].rezero_alpha_cap"])
        XCTAssertEqual(result.record.operations[1].rewrittenTensors, [])

        let (_, metadata) = try SafetensorsFile.decode(result.data)
        XCTAssertEqual(metadata["dcm_format_version"], String(ArchitectureFormat.currentVersion))
        XCTAssertTrue(try XCTUnwrap(metadata["architecture"]).contains("\"rezero_alpha_cap\""))
        let reloaded = try SafetensorsModelIO.decode(result.data)
        XCTAssertEqual(reloaded.architecture, expected)
        XCTAssertNil(reloaded.architectureFormat.legacyLogLine)
    }

    /// `--set-rezero-alpha-init` alone keeps each group's (legacy) cap, which
    /// keeps a zero init valid; `--set-rezero-alpha-cap` alone rewrites no
    /// tensor.
    func testEachOperationAlone() throws {
        let source = Self.twoGroupArchitecture()
        let sourceData = try encodedModel(source)

        let initOnly = try derive(sourceData, [SetRezeroAlphaInitDeriveOperation(value: 0, groupIndices: nil)])
        XCTAssertEqual(initOnly.targetArchitecture.blockGroups.map(\.rezeroAlphaInit), [0, 0])
        XCTAssertEqual(initOnly.targetArchitecture.blockGroups.map(\.rezeroAlphaCap), [0.5, 0.25])
        try assertTensors(sourceData, initOnly.data,
                          rewritten: Set((0..<source.numBlocks).map { "blocks.\($0).rezero_alpha" }), value: 0)

        let capOnly = try derive(sourceData, [SetRezeroAlphaCapDeriveOperation(value: 1, groupIndices: nil)])
        XCTAssertEqual(capOnly.targetArchitecture.blockGroups.map(\.rezeroAlphaCap), [1, 1])
        XCTAssertEqual(capOnly.targetArchitecture.blockGroups.map(\.rezeroAlphaInit), [0.5, 0.25])
        XCTAssertTrue(capOnly.rewrites.isEmpty, "the cap has no parameters")
        try assertTensors(sourceData, capOnly.data, rewritten: [], value: 0)
    }

    /// `--group 1` touches only group 1's blocks (blocks 1 and 2).
    func testDeriveSelectedGroupOnly() throws {
        let source = Self.twoGroupArchitecture()
        let sourceData = try encodedModel(source)
        let result = try derive(sourceData, [SetRezeroAlphaInitDeriveOperation(value: 0.125, groupIndices: [1])])
        XCTAssertEqual(result.targetArchitecture.blockGroups.map(\.rezeroAlphaInit), [0.5, 0.125])
        XCTAssertEqual(result.record.operations[0].arguments["groups"], "1")
        try assertTensors(sourceData, result.data, rewritten: ["blocks.1.rezero_alpha", "blocks.2.rezero_alpha"], value: 0.125)
    }

    func testDeriveRefusesInapplicableRequests() throws {
        let source = Self.twoGroupArchitecture()
        let sourceData = try encodedModel(source)
        let notApplicable: [any DeriveOperation] = [
            SetRezeroAlphaCapDeriveOperation(value: 0.5, groupIndices: [0]),     // nothing to change
            SetRezeroAlphaInitDeriveOperation(value: 0.5, groupIndices: [0]),    // nothing to change
            SetRezeroAlphaInitDeriveOperation(value: 0, groupIndices: [2]),      // out of range
        ]
        for operation in notApplicable {
            XCTAssertThrowsError(try derive(sourceData, [operation])) { error in
                guard case .operationNotApplicable? = error as? ModelDerivation.DeriveError else {
                    return XCTFail("expected operationNotApplicable, got \(error)")
                }
            }
        }

        // A group without ReZero, named or not.
        var mixed = source
        mixed.blockGroups[1].useRezero = false
        let mixedData = try encodedModel(mixed)
        let namingANonRezeroGroup: [any DeriveOperation] = [
            SetRezeroAlphaInitDeriveOperation(value: 0, groupIndices: [1]),
            SetRezeroAlphaCapDeriveOperation(value: 1, groupIndices: [1]),
        ]
        for operation in namingANonRezeroGroup {
            XCTAssertThrowsError(try derive(mixedData, [operation])) { error in
                guard case .operationNotApplicable(_, let detail)? = error as? ModelDerivation.DeriveError else {
                    return XCTFail("expected operationNotApplicable, got \(error)")
                }
                XCTAssertTrue(detail.contains("use_rezero false"), detail)
            }
        }
        // Without --group, only the ReZero group changes.
        let rezeroOnly = try derive(mixedData, [SetRezeroAlphaInitDeriveOperation(value: 0, groupIndices: nil)])
        XCTAssertEqual(rezeroOnly.targetArchitecture.blockGroups.map(\.rezeroAlphaInit), [0, 0.25])
        try assertTensors(mixedData, rezeroOnly.data, rewritten: ["blocks.0.rezero_alpha"], value: 0)

        var noRezero = source
        noRezero.blockGroups[0].useRezero = false
        noRezero.blockGroups[1].useRezero = false
        XCTAssertThrowsError(try derive(try encodedModel(noRezero), [SetRezeroAlphaCapDeriveOperation(value: 1, groupIndices: nil)])) { error in
            guard case .operationNotApplicable? = error as? ModelDerivation.DeriveError else {
                return XCTFail("expected operationNotApplicable, got \(error)")
            }
        }

        // Values validation rejects.
        let invalidValues: [any DeriveOperation] = [
            SetRezeroAlphaCapDeriveOperation(value: 0, groupIndices: nil),
            SetRezeroAlphaInitDeriveOperation(value: -0.5, groupIndices: nil),
        ]
        for operation in invalidValues {
            XCTAssertThrowsError(try derive(sourceData, [operation])) { error in
                guard case .invalidTargetArchitecture? = error as? ModelDerivation.DeriveError else {
                    return XCTFail("expected invalidTargetArchitecture, got \(error)")
                }
            }
        }

        XCTAssertThrowsError(try SetRezeroAlphaInitDeriveOperation.kind.make("zero", nil))
        XCTAssertThrowsError(try SetRezeroAlphaCapDeriveOperation.kind.make("", nil))
        XCTAssertNoThrow(try SetRezeroAlphaInitDeriveOperation.kind.make("0", [0]))
        XCTAssertNoThrow(try SetRezeroAlphaCapDeriveOperation.kind.make("1.0", nil))
        XCTAssertNotNil(ModelDerivation.kind(forFlag: "--set-rezero-alpha-init"))
        XCTAssertNotNil(ModelDerivation.kind(forFlag: "--set-rezero-alpha-cap"))
    }

    // MARK: - Build screen

    /// On the Build screen the α init and the cap are independent: editing
    /// one never moves the other. Only the labelled recommendation buttons
    /// set both.
    @MainActor
    func testBuildScreenRezeroInitAndCapAreIndependent() {
        let model = BuildNewModelModel(NamedArchitecture(label: "test", architecture: Self.twoGroupArchitecture()))
        let first = model.blockGroupDrafts[0]
        let second = model.blockGroupDrafts[1]
        // A legacy-shaped group (cap == init): a new init leaves the cap.
        first.group.rezeroAlphaInit = 0.3
        XCTAssertEqual(first.group.rezeroAlphaInit, 0.3)
        XCTAssertEqual(first.group.rezeroAlphaCap, 0.5)
        // A zero init leaves it too, and the group stays valid.
        first.group.rezeroAlphaInit = 0
        XCTAssertEqual(first.group.rezeroAlphaCap, 0.5)
        XCTAssertTrue(model.isValid, model.validationError ?? "")
        // A new cap leaves the init.
        first.group.rezeroAlphaCap = 1
        XCTAssertEqual(first.group.rezeroAlphaInit, 0)
        // The recommendation buttons set both.
        model.applyRecommendedRezero(model.recommendedRezeroAlphaInit, to: first)
        XCTAssertEqual(first.group.rezeroAlphaInit, model.recommendedRezeroAlphaInit)
        XCTAssertEqual(first.group.rezeroAlphaCap, model.recommendedRezeroAlphaInit)
        // The other group is untouched throughout.
        XCTAssertEqual(second.group.rezeroAlphaInit, 0.25)
        XCTAssertEqual(second.group.rezeroAlphaCap, 0.25)
    }

    @MainActor
    func testBuildScreenDepthWarningIgnoresAZeroInit() {
        let model = BuildNewModelModel(NamedArchitecture(label: "test", architecture: Self.twoGroupArchitecture()))
        let first = model.blockGroupDrafts[0]
        // 3 blocks: the recommendations are 1/√3 and 1/3.
        model.applyRecommendedRezero(model.recommendedRezeroAlphaInit, to: first)
        XCTAssertFalse(model.rezeroDepthScaleMismatch(for: first.group))
        // Zero init with a depth-appropriate cap: no warning.
        first.group.rezeroAlphaInit = 0
        XCTAssertFalse(model.rezeroDepthScaleMismatch(for: first.group))
        // A cap that matches neither recommendation is flagged, zero init or not.
        first.group.rezeroAlphaCap = 1
        XCTAssertTrue(model.rezeroDepthScaleMismatch(for: first.group))
        // A stale non-zero init is flagged even under a good cap.
        model.applyRecommendedRezero(model.recommendedRezeroAlphaInit1OverN, to: first)
        XCTAssertFalse(model.rezeroDepthScaleMismatch(for: first.group))
        first.group.rezeroAlphaInit = 0.9
        XCTAssertTrue(model.rezeroDepthScaleMismatch(for: first.group))
    }

    /// The same α-init edit on the Build screen and through
    /// `--derive-model --set-rezero-alpha-init` yields the same architecture:
    /// the cap stays where it was on both paths.
    @MainActor
    func testBuildScreenAndDeriveTreatTheInitAlike() throws {
        let source = Self.twoGroupArchitecture()
        let derived = try SetRezeroAlphaInitDeriveOperation(value: 0.2, groupIndices: [0]).apply(to: source)
        let model = BuildNewModelModel(NamedArchitecture(label: "test", architecture: source))
        model.blockGroupDrafts[0].group.rezeroAlphaInit = 0.2
        XCTAssertEqual(model.architecture, derived)
        XCTAssertEqual(derived.blockGroups[0].rezeroAlphaCap, 0.5)
    }
}
