//
//  ArchitectureActivationSiteTests.swift
//  DrewsChessMachineTests
//
//  Format v9: one activation per architecture-level site (stem, tower end,
//  feature-skip fusion, policy pre-block, value conv, value FC1 hidden),
//  replacing the single top-level `activation_function`. A site the topology
//  lacks holds `does_not_apply`, and only such a site may hold it; both
//  directions are checked on decode, in `validate()` and by every setter.
//  Older files resolve each existing site from their own
//  `activation_function` and each absent one to `does_not_apply`, so every
//  existing model builds the identical graph. Pure: no Metal.
//

import Foundation
import XCTest
@testable import DrewsChessMachine

final class ArchitectureActivationSiteTests: XCTestCase {

    // MARK: - Fixtures

    private static let siteKeys = ArchitectureActivationSite.allCases.map(\.jsonKey)

    /// A small uniform tower through the convenience init (every existing
    /// site `activation`, every absent one `does_not_apply`).
    static func tiny(
        style: BlockActivationStyle = .pre,
        activation: ActivationFunction = .relu,
        seStyle: SEStyle = .none,
        policy: PolicyHeadStyle = .intermediateConv
    ) -> NetworkArchitecture {
        NetworkArchitecture(
            inputEncoding: .basic30, channels: 16, numBlocks: 1, stemConvKernelSize: 3,
            activationFunction: activation, blockActivationStyle: style,
            blockSkipMerge: style == .pre ? .cleanAdd : .activationGated,
            blockUseRezero: false, rezeroAlphaInit: 1,
            blockConv1KernelSize: 3, blockConv2KernelSize: 3,
            blockSeStyle: seStyle, blockSeReductionRatio: 4,
            policyHeadStyle: policy, policyPreConvChannels: 8,
            valueHeadStyle: .wdlSoftmax, valueHeadConvChannels: 4, valueHeadHiddenUnits: 8,
            computeDataType: .float32)
    }

    /// Every site exists: a post-activation first group (stem) into a
    /// pre-activation last group (tower end), a compress fusion node routed
    /// to an `fc_bottleneck` policy head (fusion, policy), and the two value
    /// sites. Every site is ReLU.
    static func fullSiteFixture() -> NetworkArchitecture {
        var arch = tiny(policy: .fcBottleneck)
        var post = arch.blockGroups[0]
        post.activationStyle = .post
        post.skipMerge = .activationGated
        arch.blockGroups = [post, arch.blockGroups[0]]
        arch.featureSkipSource = .stemOutput
        arch.featureSkipFusion = .compressConvBNReLU
        arch.featureSkipToPolicyHead = true
        arch.featureSkipToValueHead = false
        arch.featureSkipToFinalBlock = false
        arch.stemActivation = .relu
        arch.featureSkipActivation = .relu
        return arch
    }

    private func encodedModel(_ arch: NetworkArchitecture) throws -> Data {
        let weights = arch.weightTensorPlan().enumerated().map { tensorIndex, spec in
            (0..<spec.elementCount).map { Float(tensorIndex * 7 + $0 % 13) * 0.125 + 0.0625 }
        }
        let meta = ModelCheckpointMetadata(creator: "test", trainingStep: nil, parentModelID: "", notes: "fixture")
        return try SafetensorsModelIO.encode(
            modelID: "20261005-1-SITE", createdAtUnix: 1_790_000_000, metadata: meta, weights: weights,
            architecture: arch, includesVelocity: false,
            lineage: try LineageRecord.forTests(trainerCompletedSteps: meta.trainerSchedule.map(\.completedTrainSteps), corpus: nil))
    }

    /// `arch` encoded and parsed back into a JSON object.
    private func object(_ arch: NetworkArchitecture) throws -> [String: Any] {
        try XCTUnwrap(JSONSerialization.jsonObject(with: try JSONEncoder().encode(arch)) as? [String: Any])
    }

    private func data(_ object: [String: Any]) throws -> Data {
        try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
    }

    /// The pre-v9 form of `arch`: the six site keys removed and a top-level
    /// `activation_function` stated.
    private func legacyObject(_ arch: NetworkArchitecture, activationFunction: String?) throws -> [String: Any] {
        var object = try object(arch)
        for key in Self.siteKeys { object.removeValue(forKey: key) }
        if let activationFunction { object["activation_function"] = activationFunction }
        return object
    }

    /// Decodes architecture JSON at `version` (nil: an undecorated
    /// `JSONDecoder`, which decodes strict-current).
    private func decode(
        _ object: [String: Any], version: Int?, source: String = "fixture.json"
    ) throws -> (architecture: NetworkArchitecture, format: ArchitectureFormat.DecodeFormat?) {
        let json = try data(object)
        guard let version else {
            return (try JSONDecoder().decode(NetworkArchitecture.self, from: json), nil)
        }
        let format = ArchitectureFormat.DecodeFormat(formatVersion: version, source: source)
        return (try ArchitectureFormat.makeDecoder(format: format).decode(NetworkArchitecture.self, from: json), format)
    }

    /// `data` re-encoded with `dcm_format_version` = `version` (nil =
    /// removed) and its architecture JSON replaced by `architecture`.
    private func rewriting(_ data: Data, version: String?, architecture: [String: Any]) throws -> Data {
        let (tensors, decodedMetadata) = try SafetensorsFile.decode(data)
        var metadata = decodedMetadata
        metadata.removeValue(forKey: SafetensorsFile.contentHashKey)
        metadata["dcm_format_version"] = version
        metadata["architecture"] = String(decoding: try self.data(architecture), as: UTF8.self)
        return try SafetensorsFile.encode(tensors: tensors, metadata: metadata)
    }

    private func temporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("activation-sites-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: false)
        addTeardownBlock {
            do { try FileManager.default.removeItem(at: url) } catch {}
        }
        return url
    }

    private func setting(_ arch: NetworkArchitecture, _ value: ActivationFunction,
                         at site: ArchitectureActivationSite) throws -> NetworkArchitecture {
        var edited = arch
        try edited.setActivation(value, at: site)
        return edited
    }

    // MARK: - does_not_apply itself

    func testDoesNotApplyIsNotAFunction() {
        XCTAssertEqual(ActivationFunction.doesNotApply.rawValue, "does_not_apply")
        XCTAssertEqual(ActivationFunction.functions, [.relu, .silu, .gelu, .leakyRelu])
    }

    /// `ActivationFunction` keeps `CaseIterable`, so `functions` — `allCases`
    /// without `does_not_apply` — is the list every picker and every derive
    /// value syntax offers, and none of them may offer the marker.
    @MainActor
    func testEveryActivationChoiceListIsTheFunctionsList() {
        XCTAssertFalse(ActivationFunction.functions.contains(.doesNotApply))
        XCTAssertEqual(ActivationFunction.functions, ActivationFunction.allCases.filter { $0 != .doesNotApply })
        XCTAssertEqual(ArchitectureSiteActivationPicker.functionChoices, ActivationFunction.functions)
        XCTAssertEqual(BuildNewModelView.groupActivationChoices, ActivationFunction.functions)
        XCTAssertEqual(BuildNewModelView.mainActivationChoices, ActivationFunction.functions)
        let syntax = ActivationFunction.functions.map(\.rawValue).joined(separator: "|")
        XCTAssertEqual(SetActivationDeriveOperation.kind.valueSyntax, syntax)
        XCTAssertEqual(SetSEActivationDeriveOperation.kind.valueSyntax, syntax)
    }

    // MARK: - Format gate

    func testCurrentVersionRequiresSiteActivations() {
        XCTAssertEqual(ArchitectureFormat.siteActivationsRequiredFromVersion, 9)
        XCTAssertGreaterThanOrEqual(ArchitectureFormat.currentVersion, 9)
        XCTAssertEqual(SafetensorsModelIO.formatVersion, String(ArchitectureFormat.currentVersion))
    }

    func testNewFilesWriteEverySiteAndNoTopLevelActivationFunction() throws {
        let (_, metadata) = try SafetensorsFile.decode(try encodedModel(.current))
        let archJSON = try XCTUnwrap(metadata["architecture"])
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(archJSON.utf8)) as? [String: Any])
        for key in Self.siteKeys {
            XCTAssertNotNil(object[key], "\(key) must be written")
        }
        XCTAssertEqual(object["stem_activation"] as? String, "does_not_apply")
        XCTAssertEqual(object["feature_skip_activation"] as? String, "does_not_apply")
        XCTAssertNil(object["activation_function"], "the retired top-level key must never be written")
    }

    func testRoundTripPreservesEachSiteIndependently() throws {
        let directory = try temporaryDirectory()
        var architectures = [NetworkArchitecture.current]
        for site in ArchitectureActivationSite.allCases {
            for function in ActivationFunction.functions {
                architectures.append(try setting(Self.fullSiteFixture(), function, at: site))
            }
        }
        for (index, arch) in architectures.enumerated() {
            try arch.validate()
            let decoded = try SafetensorsModelIO.decode(try encodedModel(arch), valueHead: .recenterUnlessMarked, source: "m.safetensors")
            XCTAssertEqual(decoded.architecture, arch, "safetensors \(index)")
            XCTAssertNil(decoded.architectureFormat.legacyLogLine, "safetensors \(index)")

            let presetFormat = ArchitectureFormat.DecodeFormat(formatVersion: ArchitectureFormat.currentVersion, source: "p.json")
            let preset = try ArchitectureFormat.makeDecoder(format: presetFormat).decode(
                NamedArchitecture.self, from: try JSONEncoder().encode(NamedArchitecture(label: "t", architecture: arch)))
            XCTAssertEqual(preset.architecture, arch, "preset \(index)")
            XCTAssertNil(presetFormat.legacyLogLine, "preset \(index)")

            let url = try ArchitectureConfig.writeTemplate(arch, to: directory.appendingPathComponent("a\(index).json"))
            XCTAssertEqual(try ArchitectureConfig.load(from: url), arch, "architecture.json \(index)")
            let configFormat = ArchitectureFormat.DecodeFormat(formatVersion: ArchitectureFormat.currentVersion, source: "a.json")
            _ = try ArchitectureFormat.makeDecoder(format: configFormat)
                .decode(VersionedArchitectureFile.self, from: try Data(contentsOf: url))
            XCTAssertNil(configFormat.legacyLogLine, "architecture.json \(index)")
        }
    }

    func testPreV9FileResolvesExistingSitesToItsActivationFunctionAndAbsentSitesToDoesNotApply() throws {
        var arch = Self.tiny(activation: .relu)
        try arch.setActivationAtEveryExistingSite(.gelu)
        let legacy = try legacyObject(arch, activationFunction: "gelu")
        let source = try encodedModel(arch)
        for version in ["8", "5", "3", nil] as [String?] {
            let file = try rewriting(source, version: version, architecture: legacy)
            let decoded = try SafetensorsModelIO.decode(file, valueHead: .recenterUnlessMarked, source: "old.safetensors")
            let a = decoded.architecture
            XCTAssertEqual(a, arch, "version \(version ?? "none")")
            XCTAssertEqual(a.blockGroups[0].activationFunction, .relu)
            for site in [ArchitectureActivationSite.towerEnd, .policyHead, .valueHeadConv, .valueHeadFC1Hidden] {
                XCTAssertEqual(a.activation(at: site), .gelu, "\(site)")
            }
            XCTAssertEqual(a.stemActivation, .doesNotApply)
            XCTAssertEqual(a.featureSkipActivation, .doesNotApply)
            let line = try XCTUnwrap(decoded.architectureFormat.legacyLogLine)
            XCTAssertTrue(line.contains("old.safetensors"), line)
            for key in ["tower_end_activation", "policy_head_activation", "value_head_conv_activation",
                        "value_head_fc1_hidden_activation"] {
                XCTAssertTrue(line.contains("\(key) := gelu (the file's activation_function)"), line)
            }
            XCTAssertTrue(line.contains("stem_activation := does_not_apply (\(ArchitectureActivationSite.stem.absentReason))"), line)
            XCTAssertTrue(line.contains(
                "feature_skip_activation := does_not_apply (\(ArchitectureActivationSite.featureSkipFusion.absentReason))"), line)
        }
    }

    func testPreV9PostActivationFileResolvesTheStem() throws {
        let arch = Self.tiny(style: .post, policy: .simpleConv)
        XCTAssertEqual(arch.stemActivation, .relu)
        let file = try rewriting(try encodedModel(arch), version: "3",
                                 architecture: try legacyObject(arch, activationFunction: "relu"))
        let decoded = try SafetensorsModelIO.decode(file, valueHead: .recenterUnlessMarked, source: "post.safetensors").architecture
        XCTAssertEqual(decoded, arch)
        XCTAssertEqual(decoded.stemActivation, .relu)
        XCTAssertEqual(decoded.valueHeadConvActivation, .relu)
        XCTAssertEqual(decoded.valueHeadFC1HiddenActivation, .relu)
        XCTAssertEqual(decoded.towerEndActivation, .doesNotApply)
        XCTAssertEqual(decoded.policyHeadActivation, .doesNotApply)
        XCTAssertEqual(decoded.featureSkipActivation, .doesNotApply)
        let sites = LayerHealth.batchNormSites(for: decoded)
        XCTAssertEqual(sites.first { $0.name == "stem.bn" }?.activation, .relu)
        XCTAssertEqual(sites.first { $0.name == "value.bn" }?.activation, .relu)
        XCTAssertNil(sites.first { $0.name == "tower_final_bn" })
        XCTAssertNil(sites.first { $0.name == "policy.pre_bn" })
        XCTAssertEqual(LayerHealth.valueFC1Layer(for: decoded).activation, .relu)
    }

    func testEveryPresetDecodesFromItsLegacyFormToItself() throws {
        for preset in NetworkArchitecture.Preset.allCases {
            let arch = NetworkArchitecture.preset(preset)
            for site in ArchitectureActivationSite.allCases {
                XCTAssertEqual(arch.activation(at: site), arch.hasActivationSite(site) ? .relu : .doesNotApply,
                               "\(preset.rawValue) \(site)")
            }
            let legacy = try legacyObject(arch, activationFunction: "relu")
            for version in [8, 3] {
                let (decoded, format) = try decode(legacy, version: version, source: "\(preset.rawValue).legacy")
                XCTAssertEqual(decoded, arch, "\(preset.rawValue) v\(version)")
                XCTAssertEqual(decoded.hashValue, arch.hashValue, "\(preset.rawValue) v\(version)")
                XCTAssertEqual(decoded.architectureSummary, arch.architectureSummary, "\(preset.rawValue) v\(version)")
                XCTAssertEqual(decoded.parameterCount, arch.parameterCount, "\(preset.rawValue) v\(version)")
                XCTAssertEqual(decoded.weightTensorPlan(), arch.weightTensorPlan(), "\(preset.rawValue) v\(version)")
                if let legacyHash = NetworkArchitecture.legacyArchHash(for: preset) {
                    XCTAssertEqual(ModelCheckpointFile.archHash(for: decoded), legacyHash, "\(preset.rawValue) v\(version)")
                }
                XCTAssertNotNil(format?.legacyLogLine, "\(preset.rawValue) v\(version)")
            }
        }
    }

    private static func uniformTowerJSON(activation: String, style: String = "pre", extra: String = "") throws -> [String: Any] {
        let json = """
        {
          "input_encoding": "basic30", "channels": 16, "num_blocks": 2, "stem_conv_kernel_size": 3,
          "activation_function": "\(activation)", "block_activation_style": "\(style)", "block_skip_merge": "clean_add",
          "block_use_rezero": false, "rezero_alpha_init": 1, "block_conv1_kernel_size": 3,
          "block_conv2_kernel_size": 3, "block_se_style": "none", "block_se_reduction_ratio": 4,
          "policy_head_style": "intermediate_conv", "policy_pre_conv_channels": 16,
          "value_head_style": "wdl_softmax", "value_head_conv_channels": 4, "value_head_hidden_units": 16,
          "compute_data_type": "float32"\(extra)
        }
        """
        return try XCTUnwrap(JSONSerialization.jsonObject(with: Data(json.utf8)) as? [String: Any])
    }

    func testLegacyUniformTowerKeysResolveEverySiteAtAnyStatedVersion() throws {
        for version in [nil, 8] as [Int?] {
            let (decoded, _) = try decode(try Self.uniformTowerJSON(activation: "silu"), version: version)
            XCTAssertEqual(decoded.blockGroups[0].activationFunction, .silu)
            for site in ArchitectureActivationSite.allCases {
                XCTAssertEqual(decoded.activation(at: site), decoded.hasActivationSite(site) ? .silu : .doesNotApply,
                               "\(site) at \(version.map(String.init) ?? "strict")")
            }
            XCTAssertNoThrow(try decoded.validate())
        }
        let format = ArchitectureFormat.DecodeFormat(formatVersion: 9, source: "uniform.json")
        _ = try ArchitectureFormat.makeDecoder(format: format)
            .decode(NetworkArchitecture.self, from: try data(try Self.uniformTowerJSON(activation: "relu")))
        let line = try XCTUnwrap(format.legacyLogLine)
        XCTAssertTrue(line.contains("legacy uniform-tower keys: "), line)
        XCTAssertTrue(line.contains("tower_end_activation := relu (the file's activation_function)"), line)
        XCTAssertTrue(line.contains("stem_activation := does_not_apply"), line)
    }

    func testV9FileMissingASiteActivationIsRejectedNamingFieldAndFile() throws {
        let arch = NetworkArchitecture.current
        let source = try encodedModel(arch)
        for key in Self.siteKeys {
            var object = try object(arch)
            object.removeValue(forKey: key)
            let file = try rewriting(source, version: "9", architecture: object)
            XCTAssertThrowsError(try SafetensorsModelIO.decode(file, valueHead: .recenterUnlessMarked, source: "new.safetensors")) { error in
                XCTAssertEqual(error as? ArchitectureFormat.FormatError,
                               .missingRequiredField(field: key, location: "the top level", formatVersion: 9, source: "new.safetensors"))
            }

            let preset: [String: Any] = ["format_version": 9, "label": "p", "architecture": object]
            let format = ArchitectureFormat.DecodeFormat(formatVersion: ArchitectureFormat.currentVersion, source: "p.json")
            XCTAssertThrowsError(try ArchitectureFormat.makeDecoder(format: format).decode(NamedArchitecture.self, from: try data(preset))) { error in
                XCTAssertEqual(error as? ArchitectureFormat.FormatError,
                               .missingRequiredField(field: key, location: "architecture", formatVersion: 9, source: "p.json"))
            }

            XCTAssertThrowsError(try decode(object, version: nil)) { error in
                XCTAssertEqual(error as? ArchitectureFormat.FormatError,
                               .missingRequiredField(field: key, location: "the top level",
                                                     formatVersion: ArchitectureFormat.currentVersion,
                                                     source: "architecture JSON"))
            }
        }
    }

    func testPreV9FileWithNeitherSitesNorActivationFunctionIsRejected() throws {
        let arch = NetworkArchitecture.current
        XCTAssertThrowsError(try decode(try legacyObject(arch, activationFunction: nil), version: 5, source: "old.json")) { error in
            XCTAssertEqual(error as? ArchitectureFormat.FormatError,
                           .legacyActivationFunctionMissing(unresolvedSites: Self.siteKeys, location: "the top level",
                                                            formatVersion: 5, source: "old.json"))
        }
        var partial = try legacyObject(arch, activationFunction: nil)
        partial["tower_end_activation"] = "relu"
        partial["stem_activation"] = "does_not_apply"
        XCTAssertThrowsError(try decode(partial, version: 5, source: "old.json")) { error in
            XCTAssertEqual(error as? ArchitectureFormat.FormatError,
                           .legacyActivationFunctionMissing(
                            unresolvedSites: ["feature_skip_activation", "policy_head_activation",
                                              "value_head_conv_activation", "value_head_fc1_hidden_activation"],
                            location: "the top level", formatVersion: 5, source: "old.json"))
        }
        // The re-stamped-fixture shape: all six stated, no activation_function.
        let (decoded, format) = try decode(try object(arch), version: 5)
        XCTAssertEqual(decoded, arch)
        XCTAssertNil(format?.legacyLogLine)
    }

    func testV9BlockGroupsFileStatingActivationFunctionIsRefused() throws {
        var object = try object(.current)
        object["activation_function"] = "relu"
        for version in [9, nil] as [Int?] {
            XCTAssertThrowsError(try decode(object, version: version, source: "bumped.json")) { error in
                XCTAssertEqual(error as? ArchitectureFormat.FormatError,
                               .retiredField(field: "activation_function", location: "the top level",
                                             formatVersion: version ?? ArchitectureFormat.currentVersion,
                                             source: version == nil ? "architecture JSON" : "bumped.json",
                                             replacedBy: Self.siteKeys))
            }
        }
        // Before v9 the same key is the legacy tower activation, and stated
        // site keys win over it.
        XCTAssertEqual(try decode(object, version: 8).architecture, .current)
    }

    func testUniformTowerFormIsNotTreatedAsRetired() throws {
        let (decoded, _) = try decode(try Self.uniformTowerJSON(activation: "relu"), version: nil)
        XCTAssertEqual(decoded.towerEndActivation, .relu)
    }

    func testAStatedSiteValueWinsInAPreV9File() throws {
        var object = try legacyObject(Self.tiny(), activationFunction: "gelu")
        object["policy_head_activation"] = "leaky_relu"
        let (decoded, format) = try decode(object, version: 5, source: "mixed.json")
        XCTAssertEqual(decoded.policyHeadActivation, .leakyRelu)
        XCTAssertEqual(decoded.towerEndActivation, .gelu)
        XCTAssertEqual(decoded.valueHeadConvActivation, .gelu)
        XCTAssertEqual(decoded.valueHeadFC1HiddenActivation, .gelu)
        XCTAssertEqual(decoded.stemActivation, .doesNotApply)
        XCTAssertEqual(decoded.featureSkipActivation, .doesNotApply)
        let line = try XCTUnwrap(format?.legacyLogLine)
        for key in Self.siteKeys where key != "policy_head_activation" {
            XCTAssertTrue(line.contains("\(key) := "), line)
        }
        XCTAssertFalse(line.contains("policy_head_activation"), line)
    }

    func testAV9FileWithSomeSiteKeysNamesTheFirstMissingOne() throws {
        var object = try object(.current)
        let removed = ["policy_head_activation", "value_head_conv_activation", "value_head_fc1_hidden_activation"]
        for key in removed { object.removeValue(forKey: key) }
        XCTAssertThrowsError(try decode(object, version: 9, source: "partial.json")) { error in
            XCTAssertEqual(error as? ArchitectureFormat.FormatError,
                           .missingRequiredField(field: "policy_head_activation", location: "the top level",
                                                 formatVersion: 9, source: "partial.json"))
        }
    }

    func testAnUnknownSiteActivationTokenIsADecodeError() throws {
        for key in Self.siteKeys {
            var current = try object(.current)
            current[key] = "swish"
            var legacy = try legacyObject(.current, activationFunction: "relu")
            legacy[key] = "swish"
            for (object, version) in [(current, 9), (legacy, 5)] {
                XCTAssertThrowsError(try decode(object, version: version)) { error in
                    guard case DecodingError.dataCorrupted(let context)? = error as? DecodingError else {
                        return XCTFail("\(key) v\(version): expected a DecodingError, got \(error)")
                    }
                    XCTAssertEqual(context.codingPath.last?.stringValue, key, "\(key) v\(version)")
                }
            }
        }
    }

    func testUniformTowerFormUsesStatedSiteKeysThenResolvesTheRest() throws {
        let object = try Self.uniformTowerJSON(activation: "gelu", extra: #", "value_head_fc1_hidden_activation": "leaky_relu""#)
        let (decoded, _) = try decode(object, version: nil)
        XCTAssertEqual(decoded.valueHeadFC1HiddenActivation, .leakyRelu)
        XCTAssertEqual(decoded.valueHeadConvActivation, .gelu)
        XCTAssertEqual(decoded.policyHeadActivation, .gelu)
        XCTAssertEqual(decoded.towerEndActivation, .gelu)
        XCTAssertEqual(decoded.stemActivation, .doesNotApply)
        XCTAssertEqual(decoded.featureSkipActivation, .doesNotApply)
    }

    /// Each mismatch case: an architecture JSON object whose `key` holds a
    /// value that disagrees with its site's existence, plus the mismatch
    /// `validate()` / decode must report.
    private func mismatchCases() throws -> [(base: NetworkArchitecture, object: [String: Any], mismatch: ActivationSiteMismatch)] {
        var cases: [(base: NetworkArchitecture, object: [String: Any], mismatch: ActivationSiteMismatch)] = []
        func add(_ arch: NetworkArchitecture, _ site: ArchitectureActivationSite, _ value: ActivationFunction) throws {
            var object = try object(arch)
            object[site.jsonKey] = value.rawValue
            cases.append((arch, object, ActivationSiteMismatch(
                site: site, value: value, siteExists: arch.hasActivationSite(site),
                reason: arch.activationSiteReason(site))))
        }
        // A function at a site the topology lacks.
        try add(Self.tiny(style: .pre), .stem, .relu)
        try add(Self.tiny(style: .post), .towerEnd, .relu)
        try add(Self.tiny(), .featureSkipFusion, .relu)
        try add(Self.tiny(policy: .simpleConv), .policyHead, .relu)
        // does_not_apply at a site that exists.
        for site in ArchitectureActivationSite.allCases {
            try add(Self.fullSiteFixture(), site, .doesNotApply)
        }
        return cases
    }

    func testASiteMismatchIsALoadErrorNamingFieldSiteAndFile() throws {
        let directory = try temporaryDirectory()
        for (index, entry) in try mismatchCases().enumerated() {
            let description = entry.mismatch.description
            XCTAssertTrue(description.hasPrefix("\(entry.mismatch.site.jsonKey) is '\(entry.mismatch.value.rawValue)', but "),
                          description)

            let model = try encodedModel(entry.base)
            let file = try rewriting(model, version: "9", architecture: entry.object)
            XCTAssertThrowsError(try SafetensorsModelIO.decode(file, valueHead: .recenterUnlessMarked, source: "bad.safetensors")) { error in
                XCTAssertEqual(error as? ArchitectureFormat.FormatError,
                               .activationSiteMismatch(entry.mismatch, location: "the top level", formatVersion: 9,
                                                       source: "bad.safetensors"), "case \(index)")
                XCTAssertTrue(String(describing: error).contains("bad.safetensors"), "case \(index)")
            }

            // The key stated in a v5 file is used, and checked, all the same.
            var legacy = entry.object
            legacy["activation_function"] = "relu"
            XCTAssertThrowsError(try decode(legacy, version: 5, source: "old.json")) { error in
                XCTAssertEqual(error as? ArchitectureFormat.FormatError,
                               .activationSiteMismatch(entry.mismatch, location: "the top level", formatVersion: 5,
                                                       source: "old.json"), "case \(index) v5")
            }

            let presetURL = directory.appendingPathComponent("preset\(index).json")
            let preset: [String: Any] = ["format_version": 9, "label": "p", "architecture": entry.object]
            try data(preset).write(to: presetURL)
            XCTAssertThrowsError(try ArchitecturePresetStore.loadFile(at: presetURL)) { error in
                XCTAssertEqual(error as? ArchitectureFormat.FormatError,
                               .activationSiteMismatch(entry.mismatch, location: "architecture", formatVersion: 9,
                                                       source: presetURL.lastPathComponent), "case \(index) preset")
            }

            let configURL = directory.appendingPathComponent("architecture\(index).json")
            var config = entry.object
            config["format_version"] = 9
            try data(config).write(to: configURL)
            XCTAssertThrowsError(try ArchitectureConfig.load(from: configURL)) { error in
                XCTAssertEqual(error as? ArchitectureFormat.FormatError,
                               .activationSiteMismatch(entry.mismatch, location: "the top level", formatVersion: 9,
                                                       source: configURL.lastPathComponent), "case \(index) config")
            }
        }
    }

    func testALegacyActivationFunctionOfDoesNotApplyIsRefused() throws {
        let legacy = try legacyObject(Self.tiny(), activationFunction: "does_not_apply")
        XCTAssertThrowsError(try decode(legacy, version: 5, source: "old.json")) { error in
            guard case .activationSiteMismatch(let mismatch, _, _, _)? = error as? ArchitectureFormat.FormatError else {
                return XCTFail("expected a site mismatch, got \(error)")
            }
            XCTAssertEqual(mismatch.value, .doesNotApply)
            XCTAssertTrue(mismatch.siteExists)
        }
        XCTAssertThrowsError(try decode(try Self.uniformTowerJSON(activation: "does_not_apply"), version: nil)) { error in
            XCTAssertEqual(error as? ArchitectureFormat.FormatError,
                           .doesNotApplyAtAnAlwaysPresentSite(field: "activation_function", location: "the top level",
                                                              formatVersion: ArchitectureFormat.currentVersion,
                                                              source: "architecture JSON"))
        }
    }

    /// A group's main path always exists, so `does_not_apply` is refused
    /// there; its SE FC1 exists exactly when it has an SE block (OD-13), so
    /// `se_activation` is `does_not_apply` on an SE-less group and a
    /// function on an SE group — anything else is refused.
    func testBlockGroupActivationFieldsFollowTheGroupsTopology() throws {
        for seStyle in [SEStyle.scaleAndBias, .none] {
            let arch = Self.tiny(seStyle: seStyle)
            XCTAssertEqual(arch.blockGroups[0].seActivation, seStyle == .none ? .doesNotApply : .relu)
            var object = try object(arch)
            var groups = try XCTUnwrap(object["block_groups"] as? [[String: Any]])
            groups[0]["activation_function"] = "does_not_apply"
            object["block_groups"] = groups
            for version in [9, 5] {
                XCTAssertThrowsError(try decode(object, version: version, source: "g.json")) { error in
                    XCTAssertEqual(error as? ArchitectureFormat.FormatError,
                                   .doesNotApplyAtAnAlwaysPresentSite(field: "activation_function", location: "block_groups[0]",
                                                                      formatVersion: version, source: "g.json"),
                                   "\(seStyle.rawValue) v\(version)")
                }
            }
            var mainPath = arch
            mainPath.blockGroups[0].activationFunction = .doesNotApply
            XCTAssertThrowsError(try mainPath.validate()) { error in
                XCTAssertEqual(error as? NetworkArchitectureError,
                               .doesNotApplyAtAnAlwaysPresentSite(field: "blockGroups[0].activationFunction"))
            }
        }
        // An SE group stating does_not_apply, at v9 and v5.
        var seObject = try object(Self.tiny(seStyle: .scaleAndBias))
        var seGroups = try XCTUnwrap(seObject["block_groups"] as? [[String: Any]])
        seGroups[0]["se_activation"] = "does_not_apply"
        seObject["block_groups"] = seGroups
        for version in [9, 5] {
            XCTAssertThrowsError(try decode(seObject, version: version, source: "g.json")) { error in
                XCTAssertEqual(error as? ArchitectureFormat.FormatError,
                               .seActivationMismatch(seStyle: .scaleAndBias, seActivation: .doesNotApply,
                                                     location: "block_groups[0]", formatVersion: version, source: "g.json"))
            }
        }
        // An SE-less group stating a function in a current-format file (from
        // v10; a v9 file's value resolves, see the next test).
        var lessObject = try object(Self.tiny(seStyle: .none))
        var lessGroups = try XCTUnwrap(lessObject["block_groups"] as? [[String: Any]])
        lessGroups[0]["se_activation"] = "relu"
        lessObject["block_groups"] = lessGroups
        let current = ArchitectureFormat.currentVersion
        XCTAssertGreaterThanOrEqual(current, ArchitectureFormat.seLessSEActivationDoesNotApplyFromVersion)
        XCTAssertThrowsError(try decode(lessObject, version: current, source: "g.json")) { error in
            XCTAssertEqual(error as? ArchitectureFormat.FormatError,
                           .seActivationMismatch(seStyle: .none, seActivation: .relu,
                                                 location: "block_groups[0]", formatVersion: current, source: "g.json"))
        }
        // In memory, both directions.
        var seGroupWithoutChoice = Self.tiny(seStyle: .scaleAndBias)
        seGroupWithoutChoice.blockGroups[0].seActivation = .doesNotApply
        XCTAssertThrowsError(try seGroupWithoutChoice.validate()) { error in
            XCTAssertEqual(error as? NetworkArchitectureError,
                           .seActivationMismatch(group: 0, seStyle: .scaleAndBias, seActivation: .doesNotApply))
        }
        var seLessWithFunction = Self.tiny(seStyle: .none)
        seLessWithFunction.blockGroups[0].seActivation = .leakyRelu
        XCTAssertThrowsError(try seLessWithFunction.validate()) { error in
            XCTAssertEqual(error as? NetworkArchitectureError,
                           .seActivationMismatch(group: 0, seStyle: .none, seActivation: .leakyRelu))
        }
    }

    /// Before v10 (v9 included: the files the first per-site build wrote) an
    /// SE-less group's se_activation had to equal the group's activation and
    /// was never applied: such a file resolves it to
    /// `does_not_apply` (logged), a value that disagreed is refused (the app
    /// refused it then too), and a file older than v5 that omits it resolves
    /// it to `does_not_apply` as well.
    func testPreV9SELessGroupResolvesItsSEActivationToDoesNotApply() throws {
        let arch = Self.tiny(seStyle: .none)
        var object = try object(arch)
        var groups = try XCTUnwrap(object["block_groups"] as? [[String: Any]])
        groups[0]["se_activation"] = "relu"
        object["block_groups"] = groups
        XCTAssertEqual(ArchitectureFormat.seLessSEActivationDoesNotApplyFromVersion, 10)
        for version in [9, 8, 5] {
            let (decoded, format) = try decode(object, version: version, source: "old.json")
            XCTAssertEqual(decoded, arch, "v\(version)")
            let line = try XCTUnwrap(format?.legacyLogLine)
            XCTAssertTrue(line.contains("block_groups[0].se_activation := does_not_apply"), line)
        }
        groups[0]["se_activation"] = "gelu"
        object["block_groups"] = groups
        XCTAssertThrowsError(try decode(object, version: 8, source: "old.json")) { error in
            XCTAssertEqual(error as? ArchitectureFormat.FormatError,
                           .seActivationMismatch(seStyle: .none, seActivation: .gelu,
                                                 location: "block_groups[0]", formatVersion: 8, source: "old.json"))
        }
        groups[0].removeValue(forKey: "se_activation")
        object["block_groups"] = groups
        let (omitted, omittedFormat) = try decode(object, version: 4, source: "old.json")
        XCTAssertEqual(omitted, arch)
        XCTAssertTrue(try XCTUnwrap(omittedFormat?.legacyLogLine).contains("block_groups[0].se_activation := does_not_apply"))
        // The legacy uniform-tower form with no SE block.
        let (uniform, _) = try decode(try Self.uniformTowerJSON(activation: "silu"), version: nil)
        XCTAssertEqual(uniform.blockGroups[0].seActivation, .doesNotApply)
    }

    /// Switching SE off sets the FC1 activation to `does_not_apply`;
    /// switching it on leaves `does_not_apply`, which validation names until a
    /// function is chosen.
    func testSEStyleChangesKeepTheFC1ActivationConsistent() throws {
        var arch = Self.tiny(seStyle: .scaleAndBias)
        arch.blockGroups[0].seActivation = .leakyRelu
        arch.blockGroups[0].seStyle = .attenuateOnly
        XCTAssertEqual(arch.blockGroups[0].seActivation, .leakyRelu, "SE to SE keeps the choice")
        arch.blockGroups[0].seStyle = .none
        XCTAssertEqual(arch.blockGroups[0].seActivation, .doesNotApply)
        XCTAssertNoThrow(try arch.validate())
        arch.blockGroups[0].seStyle = .scaleAndBias
        XCTAssertEqual(arch.blockGroups[0].seActivation, .doesNotApply, "an earlier choice is never restored")
        XCTAssertThrowsError(try arch.validate())
        arch.blockGroups[0].seActivation = .silu
        XCTAssertNoThrow(try arch.validate())
    }

    @MainActor
    func testBuildScreenAsksForAnSEActivationWhenSEIsSwitchedOn() throws {
        let model = BuildNewModelModel(NamedArchitecture(label: "t", architecture: Self.tiny(seStyle: .none)))
        let draft = model.blockGroupDrafts[0]
        let off = ArchitectureSiteActivationPicker.presentation(
            activation: draft.group.seActivation, siteExists: draft.group.seStyle != .none,
            siteDescription: "", absentReason: BlockGroup.seLessReason)
        XCTAssertFalse(off.isEnabled)
        draft.group.seStyle = .scaleAndBias
        XCTAssertEqual(draft.group.seActivation, .doesNotApply)
        let error = try XCTUnwrap(model.validationError)
        XCTAssertTrue(error.contains("blockGroups[0].seActivation"), error)
        XCTAssertNil(model.buildRequest)
        let on = ArchitectureSiteActivationPicker.presentation(
            activation: draft.group.seActivation, siteExists: draft.group.seStyle != .none,
            siteDescription: "", absentReason: BlockGroup.seLessReason)
        XCTAssertTrue(on.isEnabled)
        XCTAssertTrue(on.needsChoice)
        draft.group.seActivation = .leakyRelu
        XCTAssertTrue(model.isValid, model.validationError ?? "")
    }

    // MARK: - Semantics

    func testSiteExistenceFollowsTheTopology() {
        func sites(_ arch: NetworkArchitecture) -> Set<ArchitectureActivationSite> {
            Set(ArchitectureActivationSite.allCases.filter { arch.hasActivationSite($0) })
        }
        XCTAssertEqual(sites(Self.tiny(style: .pre)), [.towerEnd, .policyHead, .valueHeadConv, .valueHeadFC1Hidden])
        XCTAssertEqual(sites(Self.tiny(style: .post)), [.stem, .policyHead, .valueHeadConv, .valueHeadFC1Hidden])
        XCTAssertEqual(sites(Self.tiny(policy: .simpleConv)), [.towerEnd, .valueHeadConv, .valueHeadFC1Hidden])
        XCTAssertEqual(sites(Self.tiny(policy: .fcBottleneck)), [.towerEnd, .policyHead, .valueHeadConv, .valueHeadFC1Hidden])
        XCTAssertEqual(sites(Self.fullSiteFixture()), Set(ArchitectureActivationSite.allCases))

        // pre → post: neither the stem nor the tower end.
        var preToPost = Self.tiny(style: .pre)
        preToPost.blockGroups.append(Self.tiny(style: .post).blockGroups[0])
        XCTAssertEqual(sites(preToPost), [.policyHead, .valueHeadConv, .valueHeadFC1Hidden])

        // The fusion node needs compress fusion and a routed head.
        var concat = Self.fullSiteFixture()
        concat.featureSkipFusion = .concatDirect
        XCTAssertFalse(concat.hasActivationSite(.featureSkipFusion))
        var unrouted = Self.fullSiteFixture()
        unrouted.featureSkipToPolicyHead = false
        unrouted.featureSkipToFinalBlock = true
        XCTAssertFalse(unrouted.hasActivationSite(.featureSkipFusion))
        var valueRouted = unrouted
        valueRouted.featureSkipToValueHead = true
        XCTAssertTrue(valueRouted.hasActivationSite(.featureSkipFusion))
        var off = Self.fullSiteFixture()
        off.featureSkipSource = .none
        XCTAssertFalse(off.hasActivationSite(.featureSkipFusion))
    }

    func testValidateRefusesASiteMismatchInBothDirections() throws {
        for site in ArchitectureActivationSite.allCases {
            var dna = Self.fullSiteFixture()
            dna.setStoredActivationForTests(.doesNotApply, at: site)
            XCTAssertThrowsError(try dna.validate()) { error in
                XCTAssertEqual(error as? NetworkArchitectureError, .activationSiteMismatch(ActivationSiteMismatch(
                    site: site, value: .doesNotApply, siteExists: true, reason: dna.activationSiteReason(site))))
            }
        }
        let absent: [(NetworkArchitecture, ArchitectureActivationSite)] = [
            (Self.tiny(style: .pre), .stem), (Self.tiny(style: .post), .towerEnd),
            (Self.tiny(), .featureSkipFusion), (Self.tiny(policy: .simpleConv), .policyHead),
        ]
        for (base, site) in absent {
            var arch = base
            arch.setStoredActivationForTests(.leakyRelu, at: site)
            XCTAssertThrowsError(try arch.validate()) { error in
                XCTAssertEqual(error as? NetworkArchitectureError, .activationSiteMismatch(ActivationSiteMismatch(
                    site: site, value: .leakyRelu, siteExists: false, reason: site.absentReason)))
            }
        }
        // An older, more specific error is reported first.
        var both = Self.tiny(style: .pre)
        both.blockGroups[0].branchOutputInit = .zeroLastBNGamma
        both.stemActivation = .relu
        XCTAssertThrowsError(try both.validate()) { error in
            guard case .initOptionWithoutItsLayer(let field, _, _)? = error as? NetworkArchitectureError else {
                return XCTFail("expected the branch_output_init error first, got \(error)")
            }
            XCTAssertTrue(field.contains("branch_output_init"), field)
        }
    }

    func testSettersRefuseDoesNotApplyAndMismatches() {
        let original = Self.tiny(style: .pre)
        var arch = original
        XCTAssertThrowsError(try arch.setActivationAtEveryExistingSite(.doesNotApply)) { error in
            guard case .notAnActivationFunction? = error as? NetworkArchitectureError else {
                return XCTFail("expected notAnActivationFunction, got \(error)")
            }
        }
        XCTAssertEqual(arch, original)
        XCTAssertThrowsError(try arch.setMainActivationEverywhere(.doesNotApply)) { error in
            guard case .notAnActivationFunction? = error as? NetworkArchitectureError else {
                return XCTFail("expected notAnActivationFunction, got \(error)")
            }
        }
        XCTAssertEqual(arch, original)
        XCTAssertThrowsError(try arch.setActivation(.relu, at: .stem)) { error in
            XCTAssertEqual(error as? NetworkArchitectureError, .activationSiteMismatch(ActivationSiteMismatch(
                site: .stem, value: .relu, siteExists: false, reason: ArchitectureActivationSite.stem.absentReason)))
        }
        XCTAssertEqual(arch, original)
        XCTAssertThrowsError(try arch.setActivation(.doesNotApply, at: .valueHeadConv)) { error in
            guard case .activationSiteMismatch(let mismatch)? = error as? NetworkArchitectureError else {
                return XCTFail("expected activationSiteMismatch, got \(error)")
            }
            XCTAssertEqual(mismatch.site, .valueHeadConv)
            XCTAssertTrue(mismatch.siteExists)
        }
        XCTAssertEqual(arch, original)
    }

    func testSetActivationAtEveryExistingSiteLeavesAbsentSitesAlone() throws {
        var arch = NetworkArchitecture.current
        try arch.setActivationAtEveryExistingSite(.silu)
        for site in ArchitectureActivationSite.allCases {
            XCTAssertEqual(arch.activation(at: site), arch.hasActivationSite(site) ? .silu : .doesNotApply, "\(site)")
        }
        XCTAssertEqual(arch.stemActivation, .doesNotApply)
        XCTAssertEqual(arch.featureSkipActivation, .doesNotApply)
        XCTAssertTrue(arch.blockGroups.allSatisfy { $0.activationFunction == .relu }, "groups are not sites")
        try arch.validate()
    }

    func testClearActivationSitesTheTopologyLacksClearsOnlyAbsentSites() {
        var arch = NetworkArchitecture.current
        arch.policyHeadActivation = .leakyRelu
        arch.policyHeadStyle = .simpleConv
        var cleared = arch
        cleared.clearActivationSitesTheTopologyLacks()
        XCTAssertEqual(cleared.policyHeadActivation, .doesNotApply)
        var expected = arch
        expected.policyHeadActivation = .doesNotApply
        XCTAssertEqual(cleared, expected, "only the absent site changes")
        XCTAssertNoThrow(try cleared.validate())

        cleared.policyHeadStyle = .intermediateConv
        cleared.clearActivationSitesTheTopologyLacks()
        XCTAssertEqual(cleared.policyHeadActivation, .doesNotApply, "a site that appears is never filled")
        XCTAssertThrowsError(try cleared.validate()) { error in
            guard case .activationSiteMismatch(let mismatch)? = error as? NetworkArchitectureError else {
                return XCTFail("expected activationSiteMismatch, got \(error)")
            }
            XCTAssertEqual(mismatch.site, .policyHead)
        }
    }

    func testUniformConvenienceInitSetsExistingSitesAndDoesNotApplyElsewhere() throws {
        for style in BlockActivationStyle.allCases {
            for policy in PolicyHeadStyle.allCases {
                let arch = Self.tiny(style: style, activation: .gelu, policy: policy)
                XCTAssertEqual(arch.stemActivation, style == .post ? .gelu : .doesNotApply)
                XCTAssertEqual(arch.towerEndActivation, style == .pre ? .gelu : .doesNotApply)
                XCTAssertEqual(arch.policyHeadActivation, policy == .simpleConv ? .doesNotApply : .gelu)
                XCTAssertEqual(arch.featureSkipActivation, .doesNotApply)
                XCTAssertEqual(arch.valueHeadConvActivation, .gelu)
                XCTAssertEqual(arch.valueHeadFC1HiddenActivation, .gelu)
                XCTAssertNoThrow(try arch.validate(), "\(style.rawValue) \(policy.rawValue)")
            }
        }
    }

    func testSetMainActivationEverywhereMatchesTheConvenienceInitOnAnSELessTower() throws {
        for function in ActivationFunction.functions {
            var arch = Self.tiny(activation: .relu, seStyle: .none)
            try arch.setMainActivationEverywhere(function)
            XCTAssertEqual(arch, Self.tiny(activation: function, seStyle: .none), function.rawValue)
        }
    }

    func testSetMainActivationEverywhereKeepsEverySEGroupsFC1Activation() throws {
        func group(_ seStyle: SEStyle, main: ActivationFunction, se: ActivationFunction) -> BlockGroup {
            BlockGroup(
                count: 1, channels: 16, conv1KernelSize: 3, conv2KernelSize: 3,
                seStyle: seStyle, seReductionRatio: 4, useRezero: false, rezeroAlphaInit: 1, rezeroAlphaCap: 1,
                activationFunction: main, activationStyle: .pre, skipMerge: .cleanAdd, dropoutMultiplier: 1,
                seActivation: se, seGammaBiasInit: BlockGroup.standardSEGammaBiasInit,
                branchOutputInit: .standard, skipProjectionInit: .he)
        }
        func architecture(groups: [BlockGroup], sites: ActivationFunction) -> NetworkArchitecture {
            NetworkArchitecture(
                inputEncoding: .basic30, blockGroups: groups, stemConvKernelSize: 3,
                stemActivation: .doesNotApply, towerEndActivation: sites, featureSkipActivation: .doesNotApply,
                policyHeadStyle: .intermediateConv, policyPreConvChannels: 8, policyHeadActivation: sites,
                valueHeadStyle: .wdlSoftmax, valueHeadConvChannels: 4, valueHeadHiddenUnits: 8,
                valueHeadConvActivation: sites, valueHeadFC1HiddenActivation: sites,
                policyHeadFinalInit: .he, valueHeadFinalInit: .he,
                valueHeadDrawPrior: NetworkArchitecture.standardValueHeadDrawPrior,
                computeDataType: .float32, featureSkipSource: .none, featureSkipFusion: .concatDirect,
                featureSkipToPolicyHead: false, featureSkipToValueHead: false, featureSkipToFinalBlock: false)
        }
        var mixed = architecture(groups: [
            group(.scaleAndBias, main: .relu, se: .relu),
            group(.attenuateOnly, main: .relu, se: .leakyRelu),
            group(.none, main: .relu, se: .doesNotApply),
        ], sites: .relu)
        try mixed.validate()
        try mixed.setMainActivationEverywhere(.gelu)
        let expected = architecture(groups: [
            group(.scaleAndBias, main: .gelu, se: .relu),
            group(.attenuateOnly, main: .gelu, se: .leakyRelu),
            group(.none, main: .gelu, se: .doesNotApply),
        ], sites: .gelu)
        XCTAssertEqual(mixed, expected)
        try mixed.validate()
    }

    func testSetMainActivationEverywherePlusSEActivationEqualsTheConvenienceInitOnAnSETower() throws {
        var arch = Self.tiny(activation: .relu, seStyle: .scaleAndBias)
        try arch.setMainActivationEverywhere(.gelu)
        var withoutSE = arch
        XCTAssertNotEqual(withoutSE, Self.tiny(activation: .gelu, seStyle: .scaleAndBias))
        withoutSE.blockGroups[0].seActivation = .gelu
        XCTAssertEqual(withoutSE, Self.tiny(activation: .gelu, seStyle: .scaleAndBias),
                       "without the SE step the two differ only in blockGroups[0].seActivation")
        arch.blockGroups[0].seActivation = .gelu
        XCTAssertEqual(arch, Self.tiny(activation: .gelu, seStyle: .scaleAndBias))
    }

    func testSummaryKeepsTheUniformClause() {
        XCTAssertTrue(NetworkArchitecture.current.architectureSummary.contains(" . act relu . policy "),
                      NetworkArchitecture.current.architectureSummary)
        let gelu = Self.tiny(activation: .gelu)
        XCTAssertTrue(gelu.architectureSummary.contains(" . act gelu . policy "), gelu.architectureSummary)
        XCTAssertEqual(gelu.siteActivationClause, " . act gelu")
    }

    func testSummaryListsExistingSitesWhenTheyDiffer() throws {
        var arch = NetworkArchitecture.current
        try arch.setActivation(.leakyRelu, at: .policyHead)
        try arch.setActivation(.leakyRelu, at: .valueHeadConv)
        try arch.setActivation(.leakyRelu, at: .valueHeadFC1Hidden)
        XCTAssertEqual(arch.siteActivationClause,
                       " . act tower_end relu, policy leaky_relu, value_conv leaky_relu, value_fc1_hidden leaky_relu")
        XCTAssertTrue(arch.architectureSummary.contains(arch.siteActivationClause + " . policy "), arch.architectureSummary)
        var full = Self.fullSiteFixture()
        try full.setActivation(.silu, at: .featureSkipFusion)
        XCTAssertEqual(full.siteActivationClause,
                       " . act stem relu, tower_end relu, fusion silu, policy relu, value_conv relu, value_fc1_hidden relu")
        for candidate in [arch, full, .current, Self.tiny(style: .post, policy: .simpleConv)] {
            XCTAssertFalse(candidate.architectureSummary.contains("does_not_apply"), candidate.architectureSummary)
        }
    }

    // MARK: - Layer health

    private static let batchNormSiteName: [ArchitectureActivationSite: String] = [
        .stem: "stem.bn", .towerEnd: "tower_final_bn", .featureSkipFusion: "feature_skip.bn",
        .policyHead: "policy.pre_bn", .valueHeadConv: "value.bn",
    ]

    func testEachBatchNormSiteTagFollowsOnlyItsOwnField() throws {
        let base = Self.fullSiteFixture()
        let baseTags = Dictionary(uniqueKeysWithValues: LayerHealth.batchNormSites(for: base).map { ($0.name, $0.activation) })
        for (site, name) in Self.batchNormSiteName {
            XCTAssertEqual(baseTags[name] ?? nil, .relu, name)
            for function in [ActivationFunction.silu, .gelu, .leakyRelu] {
                let edited = try setting(base, function, at: site)
                let tags = Dictionary(uniqueKeysWithValues: LayerHealth.batchNormSites(for: edited).map { ($0.name, $0.activation) })
                XCTAssertEqual(Set(tags.keys), Set(baseTags.keys))
                for (otherName, tag) in tags {
                    XCTAssertEqual(tag, otherName == name ? function : baseTags[otherName] ?? nil,
                                   "\(site) = \(function.rawValue): \(otherName)")
                }
                XCTAssertEqual(LayerHealth.valueFC1Layer(for: edited).activation, .relu)
            }
        }
        let current = LayerHealth.batchNormSites(for: .current)
        XCTAssertNil(current.first { $0.name == "stem.bn" }?.activation)
        XCTAssertNil(current.first { $0.name == "feature_skip.bn" })
    }

    func testValueFC1LayerFollowsOnlyItsOwnField() throws {
        let base = Self.fullSiteFixture()
        let baseSites = LayerHealth.batchNormSites(for: base)
        for function in ActivationFunction.functions {
            let edited = try setting(base, function, at: .valueHeadFC1Hidden)
            XCTAssertEqual(LayerHealth.valueFC1Layer(for: edited).activation, function)
            XCTAssertEqual(LayerHealth.batchNormSites(for: edited), baseSites, function.rawValue)
        }
    }

    private func syntheticWeights(_ arch: NetworkArchitecture) -> [[Float]] {
        arch.weightTensorPlan().map { spec -> [Float] in
            let value: Float
            switch spec.kind {
            case .bnAffine: value = spec.name.hasSuffix(".weight") ? 1 : 0
            case .bnRunningStat: value = spec.name.hasSuffix(".running_var") ? 1 : 0
            case .scalar: value = 0.5
            case .conv, .linear, .bias: value = 0.01
            }
            return [Float](repeating: value, count: spec.elementCount)
        }
    }

    func testSmoothHeadSitesAreNotClassifiedWhileAReLUTowerEndIs() throws {
        var arch = Self.tiny(style: .pre)
        try arch.setActivation(.silu, at: .policyHead)
        try arch.setActivation(.gelu, at: .valueHeadConv)
        let summary = try LayerHealth.summarizePlanAligned(
            arch: arch, baseWeights: syntheticWeights(arch), velocity: .unavailable(reason: "test"))
        func site(_ name: String) throws -> LayerHealthSummary.BatchNormSiteHealth {
            try XCTUnwrap(summary.batchNormSites.first { $0.site == name }, name)
        }
        XCTAssertEqual(try site("tower_final_bn").classification, .classified)
        XCTAssertEqual(try site("tower_final_bn").activation, .relu)
        XCTAssertEqual(try site("policy.pre_bn").classification, .notApplicableSmoothActivation)
        XCTAssertEqual(try site("value.bn").classification, .notApplicableSmoothActivation)
        XCTAssertEqual(try site("blocks.0.bn1").classification, .classified)
    }

    // MARK: - Derive (--set-activation, --set-se-activation)

    private func derive(_ source: Data, _ operations: [any DeriveOperation]) throws -> ModelDerivation.Result {
        try ModelDerivation.derive(
            sourceData: source, sourceName: "source.safetensors", operations: operations,
            newModelID: "20261005-2-DRV1", createdAtUnix: 1_790_000_100, build: "test", invocationArguments: ["test"])
    }

    func testSetActivationSetsEveryExistingSiteAndEveryGroup() throws {
        let source = Self.fullSiteFixture()
        let result = try derive(try encodedModel(source), [SetActivationDeriveOperation(value: .leakyRelu)])
        let target = result.targetArchitecture
        for site in ArchitectureActivationSite.allCases {
            XCTAssertEqual(target.activation(at: site), .leakyRelu, "\(site)")
        }
        XCTAssertTrue(target.blockGroups.allSatisfy { $0.activationFunction == .leakyRelu })
        var expected = source
        try expected.setMainActivationEverywhere(.leakyRelu)
        XCTAssertEqual(target, expected)
        XCTAssertEqual(Set(SetActivationDeriveOperation.kind.changedArchitectureFields),
                       Set(Self.siteKeys + ["block_groups[].activation_function"]))

        // On a pre-activation tower the absent sites stay does_not_apply.
        let pre = try derive(try encodedModel(.current), [SetActivationDeriveOperation(value: .silu)]).targetArchitecture
        XCTAssertEqual(pre.stemActivation, .doesNotApply)
        XCTAssertEqual(pre.featureSkipActivation, .doesNotApply)
        XCTAssertEqual(pre.towerEndActivation, .silu)
    }

    func testSetActivationAndSetSEActivationRefuseDoesNotApply() {
        for kind in [SetActivationDeriveOperation.kind, SetSEActivationDeriveOperation.kind] {
            XCTAssertThrowsError(try kind.make("does_not_apply", nil)) { error in
                guard case .operationNotApplicable(let operation, let detail)? = error as? ModelDerivation.DeriveError else {
                    return XCTFail("\(kind.flag): expected operationNotApplicable, got \(error)")
                }
                XCTAssertEqual(operation, kind.name)
                XCTAssertTrue(detail.contains("'does_not_apply' is not an activation function"), detail)
            }
            XCTAssertThrowsError(try kind.make("swish", nil)) { error in
                guard case .operationNotApplicable(_, let detail)? = error as? ModelDerivation.DeriveError else {
                    return XCTFail("\(kind.flag): expected operationNotApplicable, got \(error)")
                }
                XCTAssertTrue(detail.contains("relu, silu, gelu, leaky_relu"), detail)
                XCTAssertFalse(detail.contains("does_not_apply"), detail)
            }
        }
    }

    func testDirectlyConstructedOperationsRefuseDoesNotApply() {
        let source = Self.tiny(seStyle: .scaleAndBias)
        let operations: [any DeriveOperation] = [
            SetActivationDeriveOperation(value: .doesNotApply),
            SetSEActivationDeriveOperation(value: .doesNotApply, groupIndices: nil),
            SetSiteActivationDeriveOperation(site: .valueHeadConv, value: .doesNotApply),
            SetSiteActivationDeriveOperation(site: .stem, value: .doesNotApply),
        ]
        for operation in operations {
            XCTAssertThrowsError(try operation.apply(to: source)) { error in
                guard case .operationNotApplicable(_, let detail)? = error as? ModelDerivation.DeriveError else {
                    return XCTFail("\(operation.kindName): expected operationNotApplicable, got \(error)")
                }
                XCTAssertTrue(detail.contains("'does_not_apply' is not an activation function"), detail)
            }
        }
        XCTAssertEqual(source, Self.tiny(seStyle: .scaleAndBias))
    }

    func testLegacySourceDerivesToAV9FileStatingEverySite() throws {
        let arch = Self.tiny()
        let legacy = try rewriting(try encodedModel(arch), version: "5",
                                   architecture: try legacyObject(arch, activationFunction: "relu"))
        let result = try derive(legacy, [SetActivationDeriveOperation(value: .gelu)])
        let (_, metadata) = try SafetensorsFile.decode(result.data)
        XCTAssertEqual(metadata["dcm_format_version"], String(ArchitectureFormat.currentVersion))
        let object = try XCTUnwrap(JSONSerialization.jsonObject(
            with: Data(try XCTUnwrap(metadata["architecture"]).utf8)) as? [String: Any])
        XCTAssertNil(object["activation_function"])
        XCTAssertEqual(object["stem_activation"] as? String, "does_not_apply")
        XCTAssertEqual(object["feature_skip_activation"] as? String, "does_not_apply")
        for key in ["tower_end_activation", "policy_head_activation", "value_head_conv_activation",
                    "value_head_fc1_hidden_activation"] {
            XCTAssertEqual(object[key] as? String, "gelu", key)
        }
    }

    // MARK: - Derive (per-site setters)

    private func assertEveryTensorBitExact(_ source: Data, _ derived: Data, _ context: String) throws {
        let (sourceTensors, _) = try SafetensorsFile.decode(source)
        let (derivedTensors, _) = try SafetensorsFile.decode(derived)
        XCTAssertEqual(sourceTensors.map(\.name), derivedTensors.map(\.name), context)
        for (before, after) in zip(sourceTensors, derivedTensors) {
            XCTAssertEqual(before.shape, after.shape, "\(context): \(before.name)")
            XCTAssertEqual(before.data.map(\.bitPattern), after.data.map(\.bitPattern), "\(context): \(before.name)")
        }
    }

    func testEachSiteSetterChangesOnlyItsFieldAndCopiesEveryTensorBitExact() throws {
        let source = Self.fullSiteFixture()
        let sourceData = try encodedModel(source)
        for site in ArchitectureActivationSite.allCases {
            let kind = try XCTUnwrap(ModelDerivation.kind(forFlag: "--\(SetSiteActivationDeriveOperation.name(for: site))"))
            XCTAssertEqual(kind.changedArchitectureFields, [site.jsonKey])
            let result = try derive(sourceData, [try kind.make("leaky_relu", nil)])
            XCTAssertEqual(result.targetArchitecture, try setting(source, .leakyRelu, at: site), "\(site)")
            XCTAssertTrue(result.rewrites.isEmpty, "\(site)")
            try assertEveryTensorBitExact(sourceData, result.data, "\(site)")
            XCTAssertEqual(result.record.operations.map(\.operation), [kind.name])
            XCTAssertEqual(result.record.operations.first?.arguments, ["value": "leaky_relu"])
            XCTAssertEqual(result.record.operations.first?.changedArchitectureFields, [site.jsonKey])
        }
    }

    func testSiteSetterRefusesASiteTheTopologyLacks() throws {
        let cases: [(NetworkArchitecture, ArchitectureActivationSite)] = [
            (Self.tiny(style: .pre), .stem), (Self.tiny(style: .post), .towerEnd),
            (Self.tiny(), .featureSkipFusion), (Self.tiny(policy: .simpleConv), .policyHead),
        ]
        for (arch, site) in cases {
            XCTAssertThrowsError(try SetSiteActivationDeriveOperation(site: site, value: .relu).apply(to: arch)) { error in
                guard case .operationNotApplicable(_, let detail)? = error as? ModelDerivation.DeriveError else {
                    return XCTFail("\(site): expected operationNotApplicable, got \(error)")
                }
                XCTAssertTrue(detail.contains(site.absentReason), detail)
            }
        }
    }

    func testSiteSetterRefusesDoesNotApply() {
        for kind in SetSiteActivationDeriveOperation.kinds {
            XCTAssertThrowsError(try kind.make("does_not_apply", nil)) { error in
                guard case .operationNotApplicable(let operation, let detail)? = error as? ModelDerivation.DeriveError else {
                    return XCTFail("\(kind.flag): expected operationNotApplicable, got \(error)")
                }
                XCTAssertEqual(operation, kind.name)
                XCTAssertTrue(detail.contains("'does_not_apply' is not an activation function"), detail)
            }
        }
    }

    func testSiteSetterRefusesANoOp() {
        XCTAssertThrowsError(try SetSiteActivationDeriveOperation(site: .valueHeadConv, value: .relu).apply(to: .current)) { error in
            guard case .operationNotApplicable(_, let detail)? = error as? ModelDerivation.DeriveError else {
                return XCTFail("expected operationNotApplicable, got \(error)")
            }
            XCTAssertTrue(detail.contains("already"), detail)
        }
    }

    func testSiteSetterRejectsAnUnknownValue() {
        for kind in SetSiteActivationDeriveOperation.kinds {
            XCTAssertThrowsError(try kind.make("swish", nil)) { error in
                guard case .operationNotApplicable(_, let detail)? = error as? ModelDerivation.DeriveError else {
                    return XCTFail("\(kind.flag): expected operationNotApplicable, got \(error)")
                }
                XCTAssertTrue(detail.contains("'swish'"), detail)
            }
            XCTAssertEqual(kind.valueSyntax, "relu|silu|gelu|leaky_relu")
        }
    }

    func testSiteSettersAreAllowedOnATrainedSource() throws {
        let arch = Self.tiny()
        let weights = arch.weightTensorPlan().enumerated().map { tensorIndex, spec in
            (0..<spec.elementCount).map { Float(tensorIndex * 5 + $0 % 11) * 0.125 + 0.0625 }
        }
        let meta = ModelCheckpointMetadata(creator: "test", trainingStep: 1200, parentModelID: "", notes: "fixture")
        let trained = try SafetensorsModelIO.encode(
            modelID: "20261005-1-TRND", createdAtUnix: 1_790_000_000, metadata: meta, weights: weights,
            architecture: arch, includesVelocity: false,
            lineage: try LineageRecord.forTests(trainerCompletedSteps: meta.trainerSchedule.map(\.completedTrainSteps), corpus: nil))
        let result = try derive(trained, [SetSiteActivationDeriveOperation(site: .valueHeadFC1Hidden, value: .leakyRelu)])
        XCTAssertEqual(result.targetArchitecture.valueHeadFC1HiddenActivation, .leakyRelu)
        try assertEveryTensorBitExact(trained, result.data, "trained source")
    }

    func testSetActivationThenASiteSetterGivesTheMixedArchitecture() throws {
        let sourceData = try encodedModel(.current)
        // The CLI builds the operations in catalog order, which puts
        // --set-activation before every site setter.
        let kinds = ModelDerivation.operationKinds.map(\.name)
        let siteIndex = try XCTUnwrap(kinds.firstIndex(of: "set-value-head-fc1-hidden-activation"))
        let activationIndex = try XCTUnwrap(kinds.firstIndex(of: "set-activation"))
        XCTAssertLessThan(activationIndex, siteIndex)
        let result = try derive(sourceData, [
            SetActivationDeriveOperation(value: .gelu),
            SetSiteActivationDeriveOperation(site: .valueHeadFC1Hidden, value: .leakyRelu),
        ])
        let target = result.targetArchitecture
        XCTAssertEqual(target.valueHeadFC1HiddenActivation, .leakyRelu)
        for site in [ArchitectureActivationSite.towerEnd, .policyHead, .valueHeadConv] {
            XCTAssertEqual(target.activation(at: site), .gelu, "\(site)")
        }
        XCTAssertTrue(target.blockGroups.allSatisfy { $0.activationFunction == .gelu })
        XCTAssertEqual(target.stemActivation, .doesNotApply)
    }

    // MARK: - Build screen

    @MainActor
    func testBuildScreenLoadsAndComposesEverySiteActivation() throws {
        var arch = NetworkArchitecture.current
        try arch.setActivation(.leakyRelu, at: .policyHead)
        try arch.setActivation(.leakyRelu, at: .valueHeadConv)
        try arch.setActivation(.silu, at: .valueHeadFC1Hidden)
        let model = BuildNewModelModel(NamedArchitecture(label: "t", architecture: arch))
        XCTAssertEqual(model.architecture, arch)
        XCTAssertEqual(model.policyHeadActivation, .leakyRelu)
        XCTAssertEqual(model.valueHeadFC1HiddenActivation, .silu)
        let loaded = BuildNewModelModel()
        loaded.load(NamedArchitecture(label: "t", architecture: arch))
        XCTAssertEqual(loaded.architecture, arch)
    }

    @MainActor
    func testBuildScreenSiteAvailabilityFollowsTheTopology() throws {
        let model = BuildNewModelModel()
        XCTAssertEqual(model.existingActivationSites, [.towerEnd, .policyHead, .valueHeadConv, .valueHeadFC1Hidden])
        XCTAssertFalse(model.siteExists(.stem))
        model.policyHeadStyle = .simpleConv
        XCTAssertFalse(model.siteExists(.policyHead))
        model.featureSkipSource = .stemOutput
        model.featureSkipFusion = .compressConvBNReLU
        model.featureSkipToValueHead = true
        XCTAssertTrue(model.siteExists(.featureSkipFusion))
        model.blockGroupDrafts[0].activationStyle = .post
        XCTAssertTrue(model.siteExists(.stem))
        XCTAssertFalse(model.siteExists(.towerEnd))
        XCTAssertEqual(model.existingActivationSites,
                       Set(ArchitectureActivationSite.allCases.filter { model.siteExists($0) }))
    }

    @MainActor
    func testBuildScreenClearsASiteTheTopologyRemoves() throws {
        let model = BuildNewModelModel()
        XCTAssertEqual(model.policyHeadActivation, .relu)
        model.policyHeadStyle = .simpleConv
        XCTAssertEqual(model.architecture.policyHeadActivation, .doesNotApply)
        XCTAssertEqual(model.policyHeadActivation, .doesNotApply)
        XCTAssertNoThrow(try model.architecture.validate())
    }

    @MainActor
    func testASiteThatAppearsMustBeChosen() throws {
        let model = BuildNewModelModel(NamedArchitecture(label: "t", architecture: Self.tiny(policy: .simpleConv)))
        XCTAssertTrue(model.isValid)
        model.policyHeadStyle = .intermediateConv
        let error = try XCTUnwrap(model.validationError)
        XCTAssertTrue(error.contains("policy_head_activation"), error)
        XCTAssertNil(model.buildRequest)
        XCTAssertFalse(model.isValid)
        model.policyHeadActivation = .leakyRelu
        XCTAssertTrue(model.isValid, model.validationError ?? "")
        XCTAssertEqual(model.architecture.policyHeadActivation, .leakyRelu)
    }

    /// The Build screen's "Use for every activation" and `--derive-model
    /// --set-activation` give the same architecture: one SE group and one
    /// SE-less group, post → pre (so the stem and the tower end both exist).
    @MainActor
    func testUseForEveryActivationMatchesDeriveSetActivation() throws {
        var source = Self.tiny(seStyle: .scaleAndBias)
        var post = Self.tiny(style: .post).blockGroups[0]
        post.activationStyle = .post
        source.blockGroups.insert(post, at: 0)
        source.stemActivation = .relu
        try source.validate()
        for function in ActivationFunction.functions where function != .relu {
            let model = BuildNewModelModel(NamedArchitecture(label: "t", architecture: source))
            try model.applyMainActivationEverywhere(function)
            let derived = try SetActivationDeriveOperation(value: function).apply(to: source)
            XCTAssertEqual(model.architecture, derived, function.rawValue)
            XCTAssertEqual(model.architecture.blockGroups[1].seActivation, .relu, "an SE group keeps its FC1 activation")
            XCTAssertEqual(model.architecture.blockGroups[0].seActivation, .doesNotApply,
                           "an SE-less group's stays does_not_apply")
            for site in ArchitectureActivationSite.allCases {
                XCTAssertEqual(model.storedActivation(at: site),
                               source.hasActivationSite(site) ? function : .doesNotApply, "\(function.rawValue) \(site)")
            }
            XCTAssertTrue(model.isValid, model.validationError ?? "")
        }
        let model = BuildNewModelModel(NamedArchitecture(label: "t", architecture: source))
        XCTAssertThrowsError(try model.applyMainActivationEverywhere(.doesNotApply))
        XCTAssertEqual(model.architecture, source, "a refused edit changes nothing")
    }

    /// Snapshots of every topology kind the load must reproduce.
    private static func snapshots() throws -> [NetworkArchitecture] {
        let preToPre = Self.tiny()
        var postToPreSimple = Self.tiny(policy: .simpleConv)
        var post = postToPreSimple.blockGroups[0]
        post.activationStyle = .post
        post.skipMerge = .activationGated
        postToPreSimple.blockGroups = [post, postToPreSimple.blockGroups[0]]
        postToPreSimple.stemActivation = .silu
        var compress = Self.fullSiteFixture()
        try compress.setActivation(.gelu, at: .featureSkipFusion)
        var allPost = Self.tiny(style: .post)
        try allPost.setActivation(.leakyRelu, at: .policyHead)
        try allPost.setActivation(.leakyRelu, at: .valueHeadConv)
        for arch in [preToPre, postToPreSimple, compress, allPost] { try arch.validate() }
        return [preToPre, postToPreSimple, compress, allPost]
    }

    @MainActor
    func testLoadingASnapshotReproducesItAcrossTopologies() throws {
        let snapshots = try Self.snapshots()
        for (startIndex, start) in snapshots.enumerated() {
            for (snapshotIndex, snapshot) in snapshots.enumerated() {
                let model = BuildNewModelModel(NamedArchitecture(label: "start", architecture: start))
                model.load(NamedArchitecture(label: "snapshot", architecture: snapshot))
                XCTAssertEqual(model.architecture, snapshot, "start \(startIndex) → snapshot \(snapshotIndex)")
                for site in ArchitectureActivationSite.allCases {
                    XCTAssertEqual(model.storedActivation(at: site), snapshot.activation(at: site),
                                   "start \(startIndex) → snapshot \(snapshotIndex): stored \(site)")
                }
            }
            XCTAssertEqual(BuildNewModelModel(NamedArchitecture(label: "init", architecture: start)).architecture, start,
                           "init \(startIndex)")
        }
        // The loaded drafts are wired to the model's site sync.
        let model = BuildNewModelModel(NamedArchitecture(label: "start", architecture: snapshots[0]))
        model.load(NamedArchitecture(label: "post first", architecture: snapshots[1]))
        XCTAssertEqual(model.stemActivation, .silu)
        model.blockGroupDrafts[0].activationStyle = .pre
        XCTAssertEqual(model.stemActivation, .doesNotApply)
    }

    @MainActor
    func testADisappearedChoiceIsNeverRestored() throws {
        // Policy pre-block.
        let policy = BuildNewModelModel()
        policy.policyHeadActivation = .leakyRelu
        policy.policyHeadStyle = .simpleConv
        policy.policyHeadStyle = .intermediateConv
        XCTAssertEqual(policy.policyHeadActivation, .doesNotApply)

        // Fusion node: routing removed, then restored.
        let fusion = BuildNewModelModel(NamedArchitecture(label: "t", architecture: Self.fullSiteFixture()))
        fusion.featureSkipActivation = .relu
        fusion.featureSkipToPolicyHead = false
        fusion.featureSkipToPolicyHead = true
        XCTAssertEqual(fusion.featureSkipActivation, .doesNotApply)

        // Stem, through a group's style.
        let stem = BuildNewModelModel()
        stem.blockGroupDrafts[0].activationStyle = .post
        stem.stemActivation = .relu
        stem.blockGroupDrafts[0].activationStyle = .pre
        stem.blockGroupDrafts[0].activationStyle = .post
        XCTAssertEqual(stem.stemActivation, .doesNotApply)

        // Tower end, through group moves: [pre a, post b] has no tower end.
        var preThenPost = Self.tiny()
        var postGroup = preThenPost.blockGroups[0]
        postGroup.activationStyle = .post
        postGroup.skipMerge = .activationGated
        preThenPost.blockGroups.append(postGroup)
        preThenPost.clearActivationSitesTheTopologyLacks()
        let moves = BuildNewModelModel(NamedArchitecture(label: "t", architecture: preThenPost))
        XCTAssertEqual(moves.towerEndActivation, .doesNotApply)
        let preDraft = moves.blockGroupDrafts[0]
        moves.moveGroup(preDraft, offset: 1)
        XCTAssertEqual(moves.towerEndActivation, .doesNotApply, "a site that appears asks for a choice")
        moves.towerEndActivation = .relu
        moves.moveGroup(preDraft, offset: -1)
        moves.moveGroup(preDraft, offset: 1)
        XCTAssertEqual(moves.towerEndActivation, .doesNotApply)

        // Tower end, through removing a group: [post b, pre a] has one.
        var postThenPre = Self.tiny()
        postThenPre.blockGroups.insert(postGroup, at: 0)
        postThenPre.stemActivation = .relu
        let removal = BuildNewModelModel(NamedArchitecture(label: "t", architecture: postThenPre))
        XCTAssertEqual(removal.towerEndActivation, .relu)
        removal.removeGroup(removal.blockGroupDrafts[1])
        XCTAssertEqual(removal.towerEndActivation, .doesNotApply)
        removal.appendCopyOfLastGroup()
        removal.blockGroupDrafts[1].activationStyle = .pre
        XCTAssertEqual(removal.towerEndActivation, .doesNotApply)
    }
}

extension NetworkArchitecture {
    /// Writes one site's field without the setters' existence check, so a
    /// test can build the mismatched architectures `validate()` must refuse.
    mutating func setStoredActivationForTests(_ value: ActivationFunction, at site: ArchitectureActivationSite) {
        switch site {
        case .stem: stemActivation = value
        case .towerEnd: towerEndActivation = value
        case .featureSkipFusion: featureSkipActivation = value
        case .policyHead: policyHeadActivation = value
        case .valueHeadConv: valueHeadConvActivation = value
        case .valueHeadFC1Hidden: valueHeadFC1HiddenActivation = value
        }
    }
}
