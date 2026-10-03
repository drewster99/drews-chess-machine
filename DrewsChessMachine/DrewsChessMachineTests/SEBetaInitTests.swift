//
//  SEBetaInitTests.swift
//  DrewsChessMachineTests
//
//  GitHub issue #7: the per-block-group `se_beta_init` option (zero-initialized
//  β path of a scale_and_bias SE FC2), the architecture format v4 gate that
//  makes the field required in new files while legacy files resolve it to
//  `glorot`, and `--derive-model`'s paired-copy transform.
//
//  The build / forward / train-step cases are GPU-backed and skip without
//  Metal; the format and derivation cases are pure.
//

import XCTest
import Metal
@testable import DrewsChessMachine

final class SEBetaInitTests: XCTestCase {

    // MARK: - Fixtures

    /// Small fp32 pre-act tower with two scale_and_bias groups (so group
    /// selection and mixed init are both exercisable), cheap to build.
    private static func twoGroupArchitecture(group0: SEBetaInit, group1: SEBetaInit) -> NetworkArchitecture {
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
        first.seBetaInit = group0
        var second = arch.blockGroups[0]
        second.count = 2
        second.seBetaInit = group1
        arch.blockGroups = [first, second]
        return arch
    }

    private func requireMetal() throws {
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal not available")
        }
    }

    /// Plan index of `name` in `arch`'s tensor plan.
    private func planIndex(_ name: String, _ arch: NetworkArchitecture) throws -> Int {
        try XCTUnwrap(arch.weightTensorPlan().firstIndex { $0.name == name }, "no tensor \(name)")
    }

    /// The (β weight, β bias, γ weight) values of block `block`'s SE FC2 in
    /// NATIVE layout (as `exportWeights` returns them).
    private func seHalves(
        _ weights: [[Float]], block: Int, arch: NetworkArchitecture
    ) throws -> (betaWeight: [Float], betaBias: [Float], gammaWeight: [Float]) {
        let group = arch.expandedBlocks[block]
        let channels = group.channels
        let reduced = channels / group.seReductionRatio
        let weight = weights[try planIndex("blocks.\(block).se_scalebias.fc2.weight", arch)]
        let bias = weights[try planIndex("blocks.\(block).se_scalebias.fc2.bias", arch)]
        XCTAssertEqual(weight.count, reduced * 2 * channels)
        XCTAssertEqual(bias.count, 2 * channels)
        var betaWeight: [Float] = []
        var betaIndices = Set<Int>()
        for range in SEScaleAndBiasBetaHalf.nativeWeightRanges(reducedChannels: reduced, channels: channels) {
            for index in range {
                betaWeight.append(weight[index])
                betaIndices.insert(index)
            }
        }
        let gammaWeight = weight.indices.filter { !betaIndices.contains($0) }.map { weight[$0] }
        let betaBias = Array(bias[SEScaleAndBiasBetaHalf.biasRange(channels: channels)])
        return (betaWeight, betaBias, gammaWeight)
    }

    // MARK: - Validation, summary, identity

    func testZeroBetaOnNonScaleAndBiasGroupIsRejected() {
        for style in [SEStyle.none, .attenuateOnly] {
            var arch = Self.twoGroupArchitecture(group0: .glorot, group1: .glorot)
            arch.blockGroups[1].seStyle = style
            arch.blockGroups[1].seBetaInit = .zero
            XCTAssertThrowsError(try arch.validate()) { error in
                XCTAssertEqual(
                    error as? NetworkArchitectureError,
                    .seBetaInitRequiresScaleAndBias(group: 1, seStyle: style, seBetaInit: .zero))
            }
            arch.blockGroups[1].seBetaInit = .glorot
            XCTAssertNoThrow(try arch.validate(), "glorot is valid for \(style.rawValue)")
        }
    }

    func testSummaryMarksOnlyZeroBetaGroups() throws {
        let mixed = Self.twoGroupArchitecture(group0: .zero, group1: .glorot)
        try mixed.validate()
        let summary = mixed.architectureSummary
        XCTAssertTrue(summary.contains("1x[3x3+3x3 @16, SE+/4 β0, "), summary)
        XCTAssertTrue(summary.contains("2x[3x3+3x3 @16, SE+/4, "), summary)
        // Zero-β changes init only: the parameter count and tensor plan match.
        let glorot = Self.twoGroupArchitecture(group0: .glorot, group1: .glorot)
        XCTAssertEqual(mixed.parameterCount, glorot.parameterCount)
        XCTAssertEqual(mixed.weightTensorPlan(), glorot.weightTensorPlan())
        XCTAssertNotEqual(mixed, glorot, "se_beta_init is part of the architecture's identity")
    }

    /// Existing (Glorot-β) architectures keep their summary, legacy hash, and
    /// identity: a legacy file of each preset (no `se_beta_init`) decodes to
    /// exactly the preset.
    func testExistingArchitecturesUnchanged() throws {
        for preset in NetworkArchitecture.Preset.allCases {
            let arch = NetworkArchitecture.preset(preset)
            XCTAssertTrue(arch.blockGroups.allSatisfy { $0.seBetaInit == .glorot }, preset.rawValue)
            XCTAssertFalse(arch.architectureSummary.contains("β0"), preset.rawValue)
            if let legacyHash = NetworkArchitecture.legacyArchHash(for: preset) {
                XCTAssertEqual(ModelCheckpointFile.archHash(for: arch), legacyHash, preset.rawValue)
            }
            let legacyJSON = try Self.removingSEBetaInit(from: try JSONEncoder().encode(arch))
            let format = ArchitectureFormat.DecodeFormat(formatVersion: 3, source: "\(preset.rawValue).legacy")
            let decoded = try ArchitectureFormat.makeDecoder(format: format)
                .decode(NetworkArchitecture.self, from: legacyJSON)
            XCTAssertEqual(decoded, arch, preset.rawValue)
            XCTAssertEqual(decoded.hashValue, arch.hashValue, preset.rawValue)
            XCTAssertEqual(decoded.architectureSummary, arch.architectureSummary, preset.rawValue)
            XCTAssertEqual(decoded.parameterCount, arch.parameterCount, preset.rawValue)
        }
    }

    // MARK: - Fresh-build init (GPU)

    func testFreshZeroBetaNetHasExactlyZeroBetaAndNonzeroGamma() async throws {
        try requireMetal()
        let arch = Self.twoGroupArchitecture(group0: .zero, group1: .zero)
        try arch.validate()
        let net = try ChessMPSNetwork(.randomWeights(initSeed: 1), arch: arch)
        let weights = try await net.network.exportWeights()
        for block in 0..<arch.numBlocks {
            let halves = try seHalves(weights, block: block, arch: arch)
            XCTAssertTrue(halves.betaWeight.allSatisfy { $0.bitPattern == 0 }, "block \(block) β weight must be +0")
            XCTAssertTrue(halves.betaBias.allSatisfy { $0.bitPattern == 0 }, "block \(block) β bias must be +0")
            XCTAssertTrue(halves.gammaWeight.contains { $0 != 0 }, "block \(block) γ weight must stay Glorot")
        }
    }

    func testMixedGroupsInitEachGroupByItsOwnSetting() async throws {
        try requireMetal()
        let arch = Self.twoGroupArchitecture(group0: .zero, group1: .glorot)
        try arch.validate()
        let net = try ChessMPSNetwork(.randomWeights(initSeed: 2), arch: arch)
        let weights = try await net.network.exportWeights()
        // Block 0 is group 0 (zero-β); blocks 1–2 are group 1 (Glorot-β).
        let zeroBlock = try seHalves(weights, block: 0, arch: arch)
        XCTAssertTrue(zeroBlock.betaWeight.allSatisfy { $0 == 0 })
        XCTAssertTrue(zeroBlock.gammaWeight.contains { $0 != 0 })
        for block in 1...2 {
            let glorotBlock = try seHalves(weights, block: block, arch: arch)
            XCTAssertTrue(glorotBlock.betaWeight.contains { $0 != 0 }, "block \(block) β must be Glorot")
            XCTAssertTrue(glorotBlock.gammaWeight.contains { $0 != 0 })
        }
    }

    /// At step 0 a zero-β block computes `sigmoid(γ)·x`. An `attenuate_only`
    /// SE computes exactly `sigmoid(FC2(s))·x`, so a zero-β net and an
    /// attenuate-only net carrying the zero-β net's γ half as its FC2 must
    /// produce the same forward pass. A Glorot-β net (β ≠ 0) must not.
    func testZeroBetaBlockEqualsSigmoidGammaTimesInputAtStepZero() async throws {
        try requireMetal()
        let board = BoardEncoder.encode(.starting, encoding: .basic30)

        func forward(_ arch: NetworkArchitecture, _ weights: [[Float]]) async throws -> [Float] {
            let net = try ChessMPSNetwork(.randomWeights(initSeed: 3), arch: arch)
            try await net.network.loadWeights(weights)
            let box = SyncBox<[Float]>([])
            try await net.evaluate(board: board) { policy, value in box.value = Array(policy) + [value] }
            return box.value
        }

        /// The scale_and_bias weights re-expressed for the attenuate_only twin:
        /// FC2 keeps only its γ half (native columns `0..<C`, bias `0..<C`).
        func attenuateTwin(_ arch: NetworkArchitecture, _ weights: [[Float]]) throws -> (NetworkArchitecture, [[Float]]) {
            var twin = arch
            for index in twin.blockGroups.indices {
                twin.blockGroups[index].seStyle = .attenuateOnly
                twin.blockGroups[index].seBetaInit = .glorot
            }
            let sourcePlan = arch.weightTensorPlan()
            let twinPlan = twin.weightTensorPlan()
            XCTAssertEqual(sourcePlan.count, twinPlan.count)
            var twinWeights: [[Float]] = []
            for (index, spec) in sourcePlan.enumerated() {
                let values = weights[index]
                if spec.name.hasSuffix("se_scalebias.fc2.weight") {
                    let reduced = spec.shape[0]
                    let channels = spec.shape[1] / 2
                    var gamma: [Float] = []
                    for row in 0..<reduced {
                        gamma.append(contentsOf: values[(row * 2 * channels)..<(row * 2 * channels + channels)])
                    }
                    twinWeights.append(gamma)
                } else if spec.name.hasSuffix("se_scalebias.fc2.bias") {
                    twinWeights.append(Array(values[0..<(spec.elementCount / 2)]))
                } else {
                    twinWeights.append(values)
                }
                XCTAssertEqual(twinWeights[index].count, twinPlan[index].elementCount, twinPlan[index].name)
            }
            return (twin, twinWeights)
        }

        let zeroArch = Self.twoGroupArchitecture(group0: .zero, group1: .zero)
        let zeroWeights = try await ChessMPSNetwork(.randomWeights(initSeed: 4), arch: zeroArch).network.exportWeights()
        let (twinArch, twinWeights) = try attenuateTwin(zeroArch, zeroWeights)
        let zeroOut = try await forward(zeroArch, zeroWeights)
        let twinOut = try await forward(twinArch, twinWeights)
        XCTAssertEqual(zeroOut.count, twinOut.count)
        let maxZeroDiff = try XCTUnwrap(zip(zeroOut, twinOut).map { abs($0 - $1) }.max())
        XCTAssertLessThan(maxZeroDiff, 1e-4, "zero-β forward must equal the sigmoid(γ)·x forward")

        let glorotArch = Self.twoGroupArchitecture(group0: .glorot, group1: .glorot)
        let glorotWeights = try await ChessMPSNetwork(.randomWeights(initSeed: 5), arch: glorotArch).network.exportWeights()
        let (glorotTwinArch, glorotTwinWeights) = try attenuateTwin(glorotArch, glorotWeights)
        let glorotOut = try await forward(glorotArch, glorotWeights)
        let glorotTwinOut = try await forward(glorotTwinArch, glorotTwinWeights)
        let maxGlorotDiff = try XCTUnwrap(zip(glorotOut, glorotTwinOut).map { abs($0 - $1) }.max())
        XCTAssertGreaterThan(maxGlorotDiff, 1e-3, "a Glorot β must change the forward (test sensitivity)")
    }

    func testOneTrainingStepMakesBetaWeightsNonzero() async throws {
        try requireMetal()
        let arch = Self.twoGroupArchitecture(group0: .zero, group1: .zero)
        let trainer = try ChessTrainer(dropoutStream: DCMRandom(seed: 1), learningRate: 1e-2, lrWarmupSteps: 0, arch: arch)
        let before = try await trainer.network.exportWeights()
        for block in 0..<arch.numBlocks {
            XCTAssertTrue(try seHalves(before, block: block, arch: arch).betaWeight.allSatisfy { $0 == 0 })
        }
        _ = try await trainer.trainStep(batchSize: 32)
        let after = try await trainer.network.exportWeights()
        for block in 0..<arch.numBlocks {
            let halves = try seHalves(after, block: block, arch: arch)
            XCTAssertTrue(halves.betaWeight.contains { $0 != 0 }, "block \(block) β weights must receive gradient")
            XCTAssertTrue(halves.betaWeight.allSatisfy { $0.isFinite })
        }
    }

    // MARK: - Format v4 gate (safetensors)

    /// `json` (an encoded architecture, or anything containing
    /// `block_groups`) with every `se_beta_init` key removed.
    private static func removingSEBetaInit(from json: Data) throws -> Data {
        func strip(_ groups: Any?) throws -> [[String: Any]] {
            let list = try XCTUnwrap(groups as? [[String: Any]], "no block_groups")
            return list.map { group in
                var g = group
                g.removeValue(forKey: "se_beta_init")
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

    private func encodedModel(_ arch: NetworkArchitecture, modelID: String = "20260930-1-SRCE") throws -> Data {
        let weights = arch.weightTensorPlan().enumerated().map { tensorIndex, spec in
            (0..<spec.elementCount).map { Float(tensorIndex * 7 + $0 % 13) * 0.125 + 0.0625 }
        }
        let meta = ModelCheckpointMetadata(creator: "test", trainingStep: nil, parentModelID: "", notes: "fixture")
        return try SafetensorsModelIO.encode(
            modelID: modelID, createdAtUnix: 1_790_000_000, metadata: meta, weights: weights,
            architecture: arch, includesVelocity: false)
    }

    /// Re-encode `data` with `dcm_format_version` set to `version` (nil =
    /// removed) and, when `stripField`, `se_beta_init` removed from the
    /// embedded architecture.
    private func rewritingHeader(_ data: Data, version: String?, stripField: Bool) throws -> Data {
        let (tensors, decodedMetadata) = try SafetensorsFile.decode(data)
        var metadata = decodedMetadata
        metadata.removeValue(forKey: SafetensorsFile.contentHashKey)
        metadata["dcm_format_version"] = version
        if stripField {
            let archJSON = try XCTUnwrap(metadata["architecture"])
            metadata["architecture"] = String(decoding: try Self.removingSEBetaInit(from: Data(archJSON.utf8)), as: UTF8.self)
        }
        return try SafetensorsFile.encode(tensors: tensors, metadata: metadata)
    }

    func testNewFilesAreStampedV4AndAlwaysCarryTheField() throws {
        let data = try encodedModel(Self.twoGroupArchitecture(group0: .glorot, group1: .glorot))
        let (_, metadata) = try SafetensorsFile.decode(data)
        XCTAssertEqual(metadata["dcm_format_version"], String(ArchitectureFormat.currentVersion))
        XCTAssertGreaterThanOrEqual(ArchitectureFormat.currentVersion, ArchitectureFormat.seBetaInitRequiredFromVersion)
        XCTAssertEqual(SafetensorsModelIO.formatVersion, String(ArchitectureFormat.currentVersion))
        let archJSON = try XCTUnwrap(metadata["architecture"])
        XCTAssertEqual(archJSON.components(separatedBy: "\"se_beta_init\":\"glorot\"").count - 1, 2,
                       "encoding must write se_beta_init on every group, even at its legacy value")
    }

    func testV3FileWithoutFieldDecodesAsGlorotAndLogsTheResolution() throws {
        let arch = Self.twoGroupArchitecture(group0: .glorot, group1: .glorot)
        for version in ["3", nil] as [String?] {
            let legacy = try rewritingHeader(try encodedModel(arch), version: version, stripField: true)
            let decoded = try SafetensorsModelIO.decode(legacy, valueHead: .recenterUnlessMarked, source: "old.safetensors")
            XCTAssertEqual(decoded.architecture, arch)
            XCTAssertEqual(decoded.architectureFormat.formatVersion, 3)
            let line = try XCTUnwrap(decoded.architectureFormat.legacyLogLine)
            XCTAssertTrue(line.hasPrefix("[ARCH] legacy file (format v3) old.safetensors: "), line)
            XCTAssertTrue(line.contains("block_groups[0].se_beta_init := glorot"), line)
            XCTAssertTrue(line.contains("block_groups[1].se_beta_init := glorot"), line)
        }
    }

    func testV4FileMissingFieldIsRejectedNamingFieldAndFile() throws {
        let arch = Self.twoGroupArchitecture(group0: .zero, group1: .glorot)
        let broken = try rewritingHeader(try encodedModel(arch), version: "4", stripField: true)
        XCTAssertThrowsError(try SafetensorsModelIO.decode(broken, valueHead: .recenterUnlessMarked, source: "new.safetensors")) { error in
            XCTAssertEqual(
                error as? ArchitectureFormat.FormatError,
                .missingRequiredField(field: "se_beta_init", location: "block_groups[0]",
                                      formatVersion: 4, source: "new.safetensors"))
            let message = String(describing: error)
            XCTAssertTrue(message.contains("se_beta_init") && message.contains("new.safetensors"), message)
        }
        // A decode that is handed no format at all is strict (current version).
        let archJSON = try Self.removingSEBetaInit(from: try JSONEncoder().encode(arch))
        XCTAssertThrowsError(try JSONDecoder().decode(NetworkArchitecture.self, from: archJSON))
    }

    func testV4RoundTripPreservesZeroAndGlorot() throws {
        let arch = Self.twoGroupArchitecture(group0: .zero, group1: .glorot)
        let decoded = try SafetensorsModelIO.decode(try encodedModel(arch))
        XCTAssertEqual(decoded.architecture, arch)
        XCTAssertEqual(decoded.architecture.blockGroups.map(\.seBetaInit), [.zero, .glorot])
        XCTAssertNil(decoded.architectureFormat.legacyLogLine)
    }

    func testFutureAndMalformedVersionsAreRejected() throws {
        let data = try encodedModel(Self.twoGroupArchitecture(group0: .glorot, group1: .glorot))
        let futureVersion = ArchitectureFormat.currentVersion + 1
        XCTAssertThrowsError(try SafetensorsModelIO.decode(
            try rewritingHeader(data, version: String(futureVersion), stripField: false), valueHead: .asStored, source: "f.safetensors")) { error in
            XCTAssertEqual(error as? ArchitectureFormat.FormatError,
                           .unsupportedFutureVersion(version: futureVersion, newestSupported: ArchitectureFormat.currentVersion, source: "f.safetensors"))
        }
        XCTAssertThrowsError(try SafetensorsModelIO.decode(
            try rewritingHeader(data, version: "four", stripField: false), valueHead: .asStored, source: "f.safetensors")) { error in
            XCTAssertEqual(error as? ArchitectureFormat.FormatError,
                           .unparseableVersion(value: "four", source: "f.safetensors"))
        }
    }

    // MARK: - Format v4 gate (presets, --architecture, architecture.json)

    private func temporaryURL(_ name: String) -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("se-beta-\(UUID().uuidString)-\(name)")
        addTeardownBlock {
            do { try FileManager.default.removeItem(at: url) } catch {}
        }
        return url
    }

    func testPresetRoundTripAndVersionGate() throws {
        let arch = Self.twoGroupArchitecture(group0: .zero, group1: .glorot)
        let named = NamedArchitecture(label: "zero-beta test", architecture: arch)
        let encoded = try JSONEncoder().encode(named)
        let object = try XCTUnwrap(try JSONSerialization.jsonObject(with: encoded) as? [String: Any])
        XCTAssertEqual(object["format_version"] as? Int, ArchitectureFormat.currentVersion, "presets carry a version marker")

        let current = temporaryURL("current.json")
        try encoded.write(to: current)
        XCTAssertEqual(try ArchitecturePresetStore.loadFile(at: current), named)
        XCTAssertEqual(try ArchitecturePresetStore.resolve(nameOrPath: current.path).named, named)

        // v4 marker, field missing -> rejected, naming field and file.
        let missing = temporaryURL("missing.json")
        try Self.removingSEBetaInit(from: encoded).write(to: missing)
        XCTAssertThrowsError(try ArchitecturePresetStore.loadFile(at: missing)) { error in
            guard case .missingRequiredField(let field, _, let version, let source)? = error as? ArchitectureFormat.FormatError else {
                return XCTFail("expected missingRequiredField, got \(error)")
            }
            XCTAssertEqual(field, "se_beta_init")
            XCTAssertEqual(version, ArchitectureFormat.currentVersion)
            XCTAssertEqual(source, missing.lastPathComponent)
        }

        // No marker (a preset saved before format v4), field missing -> legacy glorot.
        var legacyObject = try XCTUnwrap(try JSONSerialization.jsonObject(
            with: try Self.removingSEBetaInit(from: encoded)) as? [String: Any])
        legacyObject.removeValue(forKey: "format_version")
        let legacy = temporaryURL("legacy.json")
        try JSONSerialization.data(withJSONObject: legacyObject).write(to: legacy)
        let loaded = try ArchitecturePresetStore.loadFile(at: legacy)
        XCTAssertEqual(loaded.architecture.blockGroups.map(\.seBetaInit), [.glorot, .glorot])
    }

    func testArchitectureFileRoundTripAndVersionGate() throws {
        let arch = Self.twoGroupArchitecture(group0: .glorot, group1: .zero)
        let url = temporaryURL("architecture.json")
        try ArchitectureConfig.writeTemplate(arch, to: url)
        let written = try XCTUnwrap(try JSONSerialization.jsonObject(with: try Data(contentsOf: url)) as? [String: Any])
        XCTAssertEqual(written["format_version"] as? Int, ArchitectureFormat.currentVersion)
        XCTAssertEqual(try ArchitectureConfig.load(from: url), arch)

        let missing = temporaryURL("architecture-missing.json")
        try Self.removingSEBetaInit(from: try Data(contentsOf: url)).write(to: missing)
        XCTAssertThrowsError(try ArchitectureConfig.load(from: missing)) { error in
            XCTAssertNotNil(error as? ArchitectureFormat.FormatError, "\(error)")
        }
    }

    // MARK: - --derive-model

    private struct DerivedParts {
        let tensors: [String: SafetensorsTensor]
        let metadata: [String: String]
    }

    private func parts(_ data: Data) throws -> DerivedParts {
        let (tensors, metadata) = try SafetensorsFile.decode(data)
        var byName: [String: SafetensorsTensor] = [:]
        for tensor in tensors { byName[tensor.name] = tensor }
        return DerivedParts(tensors: byName, metadata: metadata)
    }

    private func derive(
        _ source: Data, _ operations: [any DeriveOperation], newModelID: String = "20260930-2-DRV1"
    ) throws -> ModelDerivation.Result {
        try ModelDerivation.derive(
            sourceData: source, sourceName: "source.safetensors", operations: operations,
            newModelID: newModelID, createdAtUnix: 1_790_000_100, build: "test")
    }

    /// Every tensor of `derived` is bit-identical to `source`'s except the β
    /// ranges of the SE FC2 of `changedBlocks`: the β weight rows must satisfy
    /// `betaWeight` and the β bias half `betaBias`. They are separate because a
    /// re-draw changes the two differently: `glorot` draws fresh β weights but
    /// zeroes the β bias (the graph builder's bias init).
    private func assertOnlyBetaChanged(
        source: DerivedParts, derived: DerivedParts, arch: NetworkArchitecture,
        changedBlocks: Set<Int>, betaWeight: ([Float]) -> Bool, betaBias: ([Float]) -> Bool
    ) throws {
        XCTAssertEqual(Set(source.tensors.keys), Set(derived.tensors.keys))
        for (name, before) in source.tensors {
            let after = try XCTUnwrap(derived.tensors[name])
            XCTAssertEqual(before.shape, after.shape, name)
            var betaRange: Range<Int>?
            var betaIsWeight: Bool?
            for block in changedBlocks {
                let group = arch.expandedBlocks[block]
                let reduced = group.channels / group.seReductionRatio
                if name == "blocks.\(block).se_scalebias.fc2.weight" {
                    betaRange = SEScaleAndBiasBetaHalf.torchWeightRange(reducedChannels: reduced, channels: group.channels)
                    betaIsWeight = true
                } else if name == "blocks.\(block).se_scalebias.fc2.bias" {
                    betaRange = SEScaleAndBiasBetaHalf.biasRange(channels: group.channels)
                    betaIsWeight = false
                }
            }
            for index in before.data.indices where betaRange?.contains(index) != true {
                XCTAssertEqual(before.data[index].bitPattern, after.data[index].bitPattern, "\(name)[\(index)] must be copied bit-exact")
            }
            if let betaRange, let betaIsWeight {
                let values = Array(after.data[betaRange])
                XCTAssertTrue(betaIsWeight ? betaWeight(values) : betaBias(values), "\(name) β half")
            }
        }
    }

    func testDeriveZeroBetaCopiesEverythingElseBitExactWithLineage() throws {
        let sourceArch = Self.twoGroupArchitecture(group0: .glorot, group1: .glorot)
        let sourceData = try encodedModel(sourceArch)
        let result = try derive(sourceData, [SetSEBetaInitDeriveOperation(value: .zero, groupIndices: nil)])

        XCTAssertEqual(result.targetArchitecture.blockGroups.map(\.seBetaInit), [.zero, .zero])
        let source = try parts(sourceData)
        let derived = try parts(result.data)
        try assertOnlyBetaChanged(
            source: source, derived: derived, arch: sourceArch, changedBlocks: [0, 1, 2],
            betaWeight: { $0.allSatisfy { $0.bitPattern == 0 } },
            betaBias: { $0.allSatisfy { $0.bitPattern == 0 } })

        // Lineage metadata.
        XCTAssertEqual(derived.metadata["model_id"], "20260930-2-DRV1")
        XCTAssertEqual(derived.metadata["parent_model_id"], "20260930-1-SRCE")
        XCTAssertEqual(derived.metadata["creator"], ModelDerivation.creator)
        XCTAssertEqual(derived.metadata["dcm_format_version"], String(ArchitectureFormat.currentVersion))
        XCTAssertEqual(derived.metadata[ValueHeadRecentering.metadataKey], source.metadata[ValueHeadRecentering.metadataKey])
        let history = try ModelDerivation.decodeHistory(derived.metadata[ModelDerivation.derivationHistoryKey])
        XCTAssertEqual(history.count, 1)
        let record = try XCTUnwrap(history.first)
        XCTAssertEqual(record, result.record)
        XCTAssertEqual(record.parentModelID, "20260930-1-SRCE")
        XCTAssertEqual(record.sourceFile, "source.safetensors")
        XCTAssertEqual(record.sourceSHA256.count, 64)
        XCTAssertEqual(record.operations.map(\.operation), ["set-se-beta-init"])
        XCTAssertEqual(record.operations[0].changedArchitectureFields, ["block_groups[].se_beta_init"])
        XCTAssertEqual(record.operations[0].rewrittenTensors.count, 6, "weight + bias for each of the three blocks")
        XCTAssertTrue(try XCTUnwrap(derived.metadata["notes"]).contains("set-se-beta-init"))

        // The derived file loads through the normal loader as the target arch.
        let reloaded = try SafetensorsModelIO.decode(result.data)
        XCTAssertEqual(reloaded.architecture, result.targetArchitecture)
        XCTAssertNil(reloaded.architectureFormat.legacyLogLine)
    }

    func testDeriveSelectedGroupAndChainedHistory() throws {
        let sourceArch = Self.twoGroupArchitecture(group0: .glorot, group1: .glorot)
        let sourceData = try encodedModel(sourceArch)
        let first = try derive(sourceData, [SetSEBetaInitDeriveOperation(value: .zero, groupIndices: [1])])
        XCTAssertEqual(first.targetArchitecture.blockGroups.map(\.seBetaInit), [.glorot, .zero])
        try assertOnlyBetaChanged(
            source: try parts(sourceData), derived: try parts(first.data), arch: sourceArch,
            changedBlocks: [1, 2],
            betaWeight: { $0.allSatisfy { $0 == 0 } },
            betaBias: { $0.allSatisfy { $0 == 0 } })

        // Derive again from the derived file: history grows, lineage chains.
        let second = try ModelDerivation.derive(
            sourceData: first.data, sourceName: "first.safetensors",
            operations: [SetSEBetaInitDeriveOperation(value: .glorot, groupIndices: [1])],
            newModelID: "20260930-3-DRV2", createdAtUnix: 1_790_000_200, build: "test")
        XCTAssertEqual(second.targetArchitecture.blockGroups.map(\.seBetaInit), [.glorot, .glorot])
        try assertOnlyBetaChanged(
            source: try parts(first.data), derived: try parts(second.data), arch: sourceArch,
            changedBlocks: [1, 2],
            betaWeight: { $0.contains { $0 != 0 } && $0.allSatisfy { $0.isFinite } },
            betaBias: { $0.allSatisfy { $0.bitPattern == 0 } })
        let history = try ModelDerivation.decodeHistory(try parts(second.data).metadata[ModelDerivation.derivationHistoryKey])
        XCTAssertEqual(history.map(\.modelID), ["20260930-2-DRV1", "20260930-3-DRV2"])
        XCTAssertEqual(history.map(\.parentModelID), ["20260930-1-SRCE", "20260930-2-DRV1"])
    }

    /// A test-only operation that widens a group — the kind of request the
    /// engine must refuse before writing anything.
    private struct WidenGroupOperation: DeriveOperation {
        var kindName: String { "test-widen" }
        var recordedArguments: [String: String] { [:] }
        func apply(to architecture: NetworkArchitecture) throws -> NetworkArchitecture {
            var edited = architecture
            edited.blockGroups[0].channels *= 2
            return edited
        }
        func tensorRewrites(source: NetworkArchitecture, target: NetworkArchitecture) throws -> [DeriveTensorRewrite] { [] }
    }

    func testDeriveRefusesShapeChangesAndInapplicableRequests() throws {
        let sourceArch = Self.twoGroupArchitecture(group0: .glorot, group1: .glorot)
        let sourceData = try encodedModel(sourceArch)

        XCTAssertThrowsError(try derive(sourceData, [WidenGroupOperation()])) { error in
            guard case .shapeChangingRequest? = error as? ModelDerivation.DeriveError else {
                return XCTFail("expected shapeChangingRequest, got \(error)")
            }
        }
        XCTAssertThrowsError(try derive(sourceData, [])) { error in
            XCTAssertEqual(error as? ModelDerivation.DeriveError, .noOperations)
        }
        // Nothing to change (already glorot), out-of-range group, non-scale+bias group.
        for operation in [
            SetSEBetaInitDeriveOperation(value: .glorot, groupIndices: nil),
            SetSEBetaInitDeriveOperation(value: .zero, groupIndices: [2]),
        ] {
            XCTAssertThrowsError(try derive(sourceData, [operation])) { error in
                guard case .operationNotApplicable? = error as? ModelDerivation.DeriveError else {
                    return XCTFail("expected operationNotApplicable, got \(error)")
                }
            }
        }
        var attenuateArch = sourceArch
        attenuateArch.blockGroups[0].seStyle = .attenuateOnly
        XCTAssertThrowsError(try derive(try encodedModel(attenuateArch),
                                        [SetSEBetaInitDeriveOperation(value: .zero, groupIndices: [0])])) { error in
            guard case .operationNotApplicable? = error as? ModelDerivation.DeriveError else {
                return XCTFail("expected operationNotApplicable, got \(error)")
            }
        }
        XCTAssertThrowsError(try SetSEBetaInitDeriveOperation.kind.make("half", nil))
    }

    func testDeriveRefusesTrainerStateSources() throws {
        let arch = Self.twoGroupArchitecture(group0: .glorot, group1: .glorot)
        let plan = arch.weightTensorPlan()
        let trainables = plan.filter { $0.kind != .bnRunningStat }
        let weights = plan.map { [Float](repeating: 0.5, count: $0.elementCount) }
            + trainables.map { [Float](repeating: 0.25, count: $0.elementCount) }
        let data = try SafetensorsModelIO.encode(
            modelID: "20260930-1-TRNR", createdAtUnix: 1_790_000_000,
            metadata: ModelCheckpointMetadata(creator: "test", trainingStep: 10, parentModelID: "", notes: ""),
            weights: weights, architecture: arch, includesVelocity: true)
        XCTAssertThrowsError(try derive(data, [SetSEBetaInitDeriveOperation(value: .zero, groupIndices: nil)])) { error in
            guard case .sourceHasOptimizerState? = error as? ModelDerivation.DeriveError else {
                return XCTFail("expected sourceHasOptimizerState, got \(error)")
            }
        }
    }

    func testOperationCatalogDrivesTheCLI() {
        XCTAssertEqual(ModelDerivation.operationKinds.map(\.flag), ["--set-se-beta-init", "--set-activation", "--set-se-activation", "--set-rezero-alpha-init", "--set-rezero-alpha-cap"])
        XCTAssertEqual(Set(ModelDerivation.operationKinds.map(\.name)).count, ModelDerivation.operationKinds.count)
        let help = DeriveModelCLI.helpText
        for kind in ModelDerivation.operationKinds {
            XCTAssertTrue(help.contains("\(kind.flag) \(kind.valueSyntax)"), help)
        }
    }
}
