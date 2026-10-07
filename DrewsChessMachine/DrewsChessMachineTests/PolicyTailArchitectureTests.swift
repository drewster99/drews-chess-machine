//
//  PolicyTailArchitectureTests.swift
//  DrewsChessMachineTests
//
//  The policy tail precision as an architecture field, format v12
//  (POLICY_TAIL_ARCHITECTURE_PLAN.md PT-D1..PT-D5, Validation 2): the format
//  gate and the pre-v12 resolution from what a file records (PT-D3 rule 1),
//  the consistency rule with the compute dtype, `withComputeDataType`, the
//  header-only read, a session champion that records no tail (rule 4), the
//  derive operation, Build New Model, the `.dcmmodel` writer, the schema-3
//  lineage validator, the flat-key header edit (rule 3) and the behavior
//  fingerprint's header. None of these needs Metal.
//

import XCTest
@testable import DrewsChessMachine

final class PolicyTailArchitectureTests: XCTestCase {

    private static let tailKey = NetworkArchitecture.CodingKeys.policyTailPrecision.rawValue

    // MARK: - Fixtures

    /// `.current` (bf16) at `tail`.
    private func bf16(_ tail: PolicyTailPrecisionSetting) throws -> NetworkArchitecture {
        try NetworkArchitecture.current.withComputeDataType(.bFloat16, tail: tail)
    }

    private func fp32() throws -> NetworkArchitecture {
        try NetworkArchitecture.current.withComputeDataType(.float32, tail: nil)
    }

    private func weights(_ arch: NetworkArchitecture) -> [[Float]] {
        arch.weightTensorPlan().enumerated().map { tensorIndex, spec in
            (0..<spec.elementCount).map { Float(tensorIndex * 7 + $0 % 13) * 0.125 + 0.0625 }
        }
    }

    /// A plain model file of `arch` from the current writer, with `lineage`
    /// (an untrained record when nil).
    private func encodedModel(_ arch: NetworkArchitecture, lineage: LineageRecord? = nil) throws -> Data {
        let meta = ModelCheckpointMetadata(creator: "test", trainingStep: nil, parentModelID: "", notes: "fixture")
        return try SafetensorsModelIO.encode(
            modelID: "20261007-1-TAIL", createdAtUnix: 1_790_000_000, metadata: meta, weights: weights(arch),
            architecture: arch, includesVelocity: false,
            lineage: try lineage ?? LineageRecord.forTests(trainerCompletedSteps: nil, corpus: nil))
    }

    /// `data` as a file written before v12 would carry it: `dcm_format_version`
    /// = `version`, `policy_tail_precision` removed from the architecture JSON,
    /// and `trainer_policy_tail_precision` = `flatKey` when given.
    private func preV12(_ data: Data, version: String = "11", flatKey: String? = nil) throws -> Data {
        let (tensors, decodedMetadata) = try SafetensorsFile.decode(data)
        var metadata = decodedMetadata
        metadata.removeValue(forKey: SafetensorsFile.contentHashKey)
        metadata[SafetensorsModelIO.Key.formatVersion] = version
        let archText = try XCTUnwrap(metadata[SafetensorsModelIO.Key.architecture])
        var object = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(archText.utf8)) as? [String: Any])
        object.removeValue(forKey: Self.tailKey)
        metadata[SafetensorsModelIO.Key.architecture] = String(
            decoding: try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys]), as: UTF8.self)
        if let flatKey { metadata[SafetensorsModelIO.Key.trainerPolicyTailPrecision] = flatKey }
        return try SafetensorsFile.encode(tensors: tensors, metadata: metadata)
    }

    /// A trained replay record whose configuration states `tail`.
    private func trainedRecord(tail: PolicyTailPrecisionSetting) throws -> LineageRecord {
        let start = Date(timeIntervalSince1970: 1_790_000_000)
        let tracker = try LineageTracker(start: .fresh(initialization: .forTests), pathKind: .replay, argv: ["dcm"],
                                         startedAt: start, segmentStartTrainerStep: 0)
        try tracker.noteSegmentStartForTests(trainerStep: 0, policyTailPrecision: tail)
        return try tracker.record(at: start.addingTimeInterval(60), trainerCompletedSteps: 1, segmentLocalStep: 1,
                                  segmentGames: 0, segmentPositions: 0, corpus: nil,
                                  parameters: try LineageRecord.Parameters(values: ["learning_rate": .double(0.0005)]),
                                  rng: .withoutRunStreams(dropoutPhiloxState: nil), inputs: tracker.testInputs)
    }

    private func architectureObject(_ arch: NetworkArchitecture) throws -> [String: Any] {
        try XCTUnwrap(JSONSerialization.jsonObject(with: try JSONEncoder().encode(arch)) as? [String: Any])
    }

    private func decode(_ object: [String: Any], version: Int,
                        recorded: ArchitectureFormat.RecordedPolicyTailPrecision? = nil)
        throws -> (NetworkArchitecture, ArchitectureFormat.DecodeFormat) {
        let format = ArchitectureFormat.DecodeFormat(formatVersion: version, source: "fixture.json",
                                                     recordedPolicyTailPrecision: recorded)
        let data = try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
        return (try ArchitectureFormat.makeDecoder(format: format).decode(NetworkArchitecture.self, from: data), format)
    }

    // MARK: - Format v12

    func testFormatV12RequiresTheField() throws {
        XCTAssertEqual(ArchitectureFormat.currentVersion, 12)
        XCTAssertEqual(ArchitectureFormat.policyTailPrecisionRequiredFromVersion, 12)
        var object = try architectureObject(try bf16(.mixedFinalProjection))
        object.removeValue(forKey: Self.tailKey)
        XCTAssertThrowsError(try decode(object, version: 12)) { error in
            XCTAssertEqual(error as? ArchitectureFormat.FormatError,
                           .missingRequiredField(field: Self.tailKey, location: "the top level",
                                                 formatVersion: 12, source: "fixture.json"))
        }
        // A decode handed no format is strict current.
        let data = try JSONSerialization.data(withJSONObject: object)
        XCTAssertThrowsError(try JSONDecoder().decode(NetworkArchitecture.self, from: data))
    }

    func testEncodeAlwaysWritesTheField() throws {
        for arch in [try bf16(.float32FromPreBatchNorm), try bf16(.mixedFinalProjection), try fp32()] {
            let object = try architectureObject(arch)
            XCTAssertEqual(object[Self.tailKey] as? String, arch.policyTailPrecision.rawValue)
        }
    }

    func testAStatedValueIsReadAtAnyVersion() throws {
        let arch = try bf16(.float32FromPreBatchNorm)
        for version in [3, 8, 11, 12] {
            let (decoded, format) = try decode(try architectureObject(arch), version: version,
                                               recorded: .init(value: .mixedFinalProjection, recordedIn: "x"))
            XCTAssertEqual(decoded.policyTailPrecision, .float32FromPreBatchNorm, "v\(version)")
            XCTAssertFalse(format.legacyLog.resolutions.contains { $0.hasPrefix(Self.tailKey) }, "v\(version)")
            XCTAssertEqual(format.legacyLog.policyTailPrecisionOrigin, .stated)
        }
    }

    func testAPreV12ArchitectureResolvesTheRecordedTailElseMixed() throws {
        var object = try architectureObject(try bf16(.mixedFinalProjection))
        object.removeValue(forKey: Self.tailKey)
        for version in [3, 9, 11] {
            let (recordedTail, recordedFormat) = try decode(
                object, version: version, recorded: .init(value: .float32FromPreBatchNorm, recordedIn: "the test"))
            XCTAssertEqual(recordedTail.policyTailPrecision, .float32FromPreBatchNorm)
            XCTAssertEqual(recordedFormat.legacyLog.resolutions.filter { $0.hasPrefix(Self.tailKey) },
                           ["\(Self.tailKey) := fp32_from_pre_bn (recorded in the test)"])
            let (unrecorded, unrecordedFormat) = try decode(object, version: version)
            XCTAssertEqual(unrecorded.policyTailPrecision, .mixedFinalProjection)
            XCTAssertEqual(unrecordedFormat.legacyLog.resolutions.filter { $0.hasPrefix(Self.tailKey) },
                           ["\(Self.tailKey) := mixed_final_projection (not recorded)"])
            XCTAssertEqual(unrecordedFormat.legacyLog.policyTailPrecisionOrigin, .notRecorded)
        }
    }

    func testAPreV12FP32ArchitectureIsDoesNotApplyWhateverItRecords() throws {
        var object = try architectureObject(try fp32())
        object.removeValue(forKey: Self.tailKey)
        let (decoded, format) = try decode(object, version: 11, recorded: .init(value: .float32FromPreBatchNorm, recordedIn: "k"))
        XCTAssertEqual(decoded.policyTailPrecision, .doesNotApply)
        XCTAssertEqual(format.legacyLog.resolutions.filter { $0.hasPrefix(Self.tailKey) },
                       ["\(Self.tailKey) := does_not_apply (recorded fp32_from_pre_bn ignored: float32)"])
        XCTAssertNoThrow(try decoded.validate())
    }

    func testTheUniformTowerFormResolvesLikeAPreV12File() throws {
        let uniform = """
        {"input_encoding":"basic30","channels":32,"num_blocks":2,"stem_conv_kernel_size":3,
         "activation_function":"relu","block_activation_style":"pre","block_skip_merge":"clean_add",
         "block_use_rezero":true,"rezero_alpha_init":0.5,"block_conv1_kernel_size":3,"block_conv2_kernel_size":3,
         "block_se_style":"scale_and_bias","block_se_reduction_ratio":4,"policy_head_style":"intermediate_conv",
         "policy_pre_conv_channels":32,"value_head_style":"wdl_softmax","value_head_conv_channels":4,
         "value_head_hidden_units":16,"compute_data_type":"bfloat16"}
        """
        let format = ArchitectureFormat.DecodeFormat(formatVersion: ArchitectureFormat.currentVersion, source: "uniform.json")
        let decoded = try ArchitectureFormat.makeDecoder(format: format)
            .decode(NetworkArchitecture.self, from: Data(uniform.utf8))
        XCTAssertEqual(decoded.policyTailPrecision, .mixedFinalProjection)
        XCTAssertEqual(format.legacyLog.resolutions.filter { $0.hasPrefix(Self.tailKey) },
                       ["\(Self.tailKey) := mixed_final_projection (not recorded)"])
    }

    // MARK: - Consistency with the compute dtype

    func testValidateRefusesATailOnFP32AndDoesNotApplyOnBF16() throws {
        var fp32WithTail = try fp32()
        fp32WithTail.policyTailPrecision = .float32FromPreBatchNorm
        XCTAssertThrowsError(try fp32WithTail.validate()) { error in
            XCTAssertEqual(error as? NetworkArchitectureError,
                           .policyTailPrecisionMismatch(computeDataType: .float32, policyTailPrecision: .float32FromPreBatchNorm))
        }
        var bf16WithoutTail = try bf16(.mixedFinalProjection)
        bf16WithoutTail.policyTailPrecision = .doesNotApply
        XCTAssertThrowsError(try bf16WithoutTail.validate()) { error in
            XCTAssertEqual(error as? NetworkArchitectureError,
                           .policyTailPrecisionMismatch(computeDataType: .bFloat16, policyTailPrecision: .doesNotApply))
        }
        XCTAssertNoThrow(try fp32().validate())
        for tail in PolicyTailPrecisionSetting.reducedPrecisionCases {
            XCTAssertNoThrow(try bf16(tail).validate())
        }
    }

    func testWithComputeDataTypeKeepsTheTailConsistentBothWays() throws {
        let fromPreBatchNorm = try bf16(.float32FromPreBatchNorm)
        let toFP32 = try fromPreBatchNorm.withComputeDataType(.float32, tail: nil)
        XCTAssertEqual(toFP32.computeDataType, .float32)
        XCTAssertEqual(toFP32.policyTailPrecision, .doesNotApply)
        XCTAssertThrowsError(try fromPreBatchNorm.withComputeDataType(.float32, tail: .mixedFinalProjection))
        // From fp32 the caller must state the tail; none is implied.
        XCTAssertThrowsError(try toFP32.withComputeDataType(.float16, tail: nil)) { error in
            XCTAssertEqual(error as? NetworkArchitectureError, .policyTailPrecisionRequired(computeDataType: .float16))
        }
        XCTAssertThrowsError(try toFP32.withComputeDataType(.bFloat16, tail: .doesNotApply))
        let back = try toFP32.withComputeDataType(.bFloat16, tail: .float32FromPreBatchNorm)
        XCTAssertEqual(back, fromPreBatchNorm)
    }

    func testEveryPresetStatesItsDtypesHistoricalTail() throws {
        for preset in NetworkArchitecture.Preset.allCases {
            let arch = NetworkArchitecture.preset(preset)
            XCTAssertEqual(arch.policyTailPrecision, PolicyTailPrecisionSetting.historical(for: arch.computeDataType),
                           preset.rawValue)
            XCTAssertNoThrow(try arch.validate(), preset.rawValue)
        }
    }

    // MARK: - Model files

    /// A pre-v12 file recording `fp32_from_pre_bn` — in its flat key or its
    /// lineage configuration — loads as `fp32_from_pre_bn`; one recording
    /// nothing as `mixed_final_projection`; each with one `[ARCH] legacy file`
    /// entry for the tail.
    func testAPreV12FileLoadsAsTheTailItRecords() throws {
        let current = try encodedModel(try bf16(.mixedFinalProjection))
        let flat = try SafetensorsModelIO.decode(try preV12(current, flatKey: "fp32_from_pre_bn"))
        XCTAssertEqual(flat.architecture.policyTailPrecision, .float32FromPreBatchNorm)
        XCTAssertTrue(try XCTUnwrap(flat.architectureFormat.legacyLogLine)
            .contains("\(Self.tailKey) := fp32_from_pre_bn (recorded in \(SafetensorsModelIO.Key.trainerPolicyTailPrecision))"))

        let withRecord = try encodedModel(try bf16(.mixedFinalProjection), lineage: try trainedRecord(tail: .float32FromPreBatchNorm))
        let lineage = try SafetensorsModelIO.decode(try preV12(withRecord))
        XCTAssertEqual(lineage.architecture.policyTailPrecision, .float32FromPreBatchNorm)
        XCTAssertTrue(try XCTUnwrap(lineage.architectureFormat.legacyLogLine)
            .contains("\(Self.tailKey) := fp32_from_pre_bn (recorded in \(LineageRecord.metadataKey) configuration)"))

        let nothing = try SafetensorsModelIO.decode(try preV12(current))
        XCTAssertEqual(nothing.architecture.policyTailPrecision, .mixedFinalProjection)
        XCTAssertTrue(try XCTUnwrap(nothing.architectureFormat.legacyLogLine)
            .contains("\(Self.tailKey) := mixed_final_projection (not recorded)"))
    }

    func testAPreV12FP32FileRecordingATailLoadsAsDoesNotApply() throws {
        let file = try SafetensorsModelIO.decode(try preV12(try encodedModel(try fp32()), flatKey: "fp32_from_pre_bn"))
        XCTAssertEqual(file.architecture.policyTailPrecision, .doesNotApply)
        XCTAssertTrue(try XCTUnwrap(file.architectureFormat.legacyLogLine)
            .contains("\(Self.tailKey) := does_not_apply (recorded fp32_from_pre_bn ignored: float32)"))
    }

    func testAFlatKeyDisagreeingWithTheLineageIsRefused() throws {
        let withRecord = try encodedModel(try bf16(.mixedFinalProjection), lineage: try trainedRecord(tail: .mixedFinalProjection))
        XCTAssertThrowsError(try SafetensorsModelIO.decode(try preV12(withRecord, flatKey: "fp32_from_pre_bn"))) { error in
            guard case SafetensorsModelIO.IOError.policyTailDisagreesWithLineage = error else {
                return XCTFail("expected policyTailDisagreesWithLineage, got \(error)")
            }
        }
        XCTAssertThrowsError(try SafetensorsModelIO.decode(try preV12(try encodedModel(try bf16(.mixedFinalProjection)), flatKey: "fp16"))) { error in
            guard case SafetensorsModelIO.IOError.malformedRecordedPolicyTailPrecision = error else {
                return XCTFail("expected malformedRecordedPolicyTailPrecision, got \(error)")
            }
        }
    }

    /// A v12 file never carries the flat key; one that does holds a second
    /// copy of the architecture's value and is refused.
    func testAV12FileCarryingTheFlatKeyIsRefused() throws {
        let (tensors, decodedMetadata) = try SafetensorsFile.decode(try encodedModel(try bf16(.mixedFinalProjection)))
        var metadata = decodedMetadata
        metadata.removeValue(forKey: SafetensorsFile.contentHashKey)
        metadata[SafetensorsModelIO.Key.trainerPolicyTailPrecision] = "mixed_final_projection"
        XCTAssertThrowsError(try SafetensorsModelIO.decode(try SafetensorsFile.encode(tensors: tensors, metadata: metadata))) { error in
            guard case SafetensorsModelIO.IOError.retiredTrainerPolicyTailPrecisionKey = error else {
                return XCTFail("expected retiredTrainerPolicyTailPrecisionKey, got \(error)")
            }
        }
    }

    /// The model catalog's header-only read resolves the tail a full load
    /// builds.
    func testTheHeaderOnlyReadResolvesTheSameTailAsTheFullDecode() throws {
        let current = try encodedModel(try bf16(.mixedFinalProjection))
        for data in [try preV12(current, flatKey: "fp32_from_pre_bn"), try preV12(current), current] {
            let (_, metadata) = try SafetensorsFile.decode(data)
            let headerOnly = try SafetensorsModelIO.decodeArchitecture(fromMetadata: metadata, source: "h.safetensors")
            XCTAssertEqual(headerOnly.architecture.policyTailPrecision,
                           try SafetensorsModelIO.decode(data).architecture.policyTailPrecision)
        }
    }

    /// PT-D3 rule 3: the approved header edit adds only the flat key. The file
    /// keeps its version and every other key, loads as the stated tail, and
    /// its weights are bit-identical to before.
    func testAFlatKeyHeaderEditLoadsAsItsTailWithIdenticalWeights() throws {
        let before = try preV12(try encodedModel(try bf16(.mixedFinalProjection)), version: "6")
        let (tensors, metadata) = try SafetensorsFile.decode(before)
        var edited = metadata
        edited.removeValue(forKey: SafetensorsFile.contentHashKey)
        edited[SafetensorsModelIO.Key.trainerPolicyTailPrecision] = "fp32_from_pre_bn"
        let after = try SafetensorsFile.encode(tensors: tensors, metadata: edited)
        let (_, afterMetadata) = try SafetensorsFile.decode(after)
        XCTAssertEqual(afterMetadata[SafetensorsFile.contentHashKey], metadata[SafetensorsFile.contentHashKey],
                       "content_sha256 covers the tensors only")
        XCTAssertEqual(afterMetadata[SafetensorsModelIO.Key.formatVersion], "6")
        let beforeFile = try SafetensorsModelIO.decode(before, valueHead: .asStored)
        let afterFile = try SafetensorsModelIO.decode(after, valueHead: .asStored)
        XCTAssertEqual(beforeFile.architecture.policyTailPrecision, .mixedFinalProjection)
        XCTAssertEqual(afterFile.architecture.policyTailPrecision, .float32FromPreBatchNorm)
        XCTAssertEqual(beforeFile.file.weights.map { $0.map(\.bitPattern) }, afterFile.file.weights.map { $0.map(\.bitPattern) })
    }

    // MARK: - Session champion (PT-D3 rule 4)

    func testAnUnrecordedSessionChampionTakesItsTrainerFilesTail() throws {
        let current = try encodedModel(try bf16(.mixedFinalProjection))
        let trainer = try CheckpointManager.decodeAnyModelFile(try preV12(current, flatKey: "fp32_from_pre_bn"))
        XCTAssertEqual(trainer.architecture.policyTailPrecision, .float32FromPreBatchNorm)
        let champion = try CheckpointManager.decodeSessionChampionFile(
            try preV12(current), source: "s.dcmsession champion", trainerFile: trainer)
        XCTAssertEqual(champion.architecture.policyTailPrecision, .float32FromPreBatchNorm)
        XCTAssertEqual(champion.architecture, trainer.architecture)
        XCTAssertTrue(try XCTUnwrap(champion.architectureFormat?.legacyLogLine)
            .contains("\(Self.tailKey) := fp32_from_pre_bn (recorded in the same session's trainer.safetensors)"))

        // A champion recording its own tail keeps it; a trainer recording
        // none lends nothing.
        let recordedChampion = try CheckpointManager.decodeSessionChampionFile(
            try preV12(current, flatKey: "mixed_final_projection"), source: "c", trainerFile: trainer)
        XCTAssertEqual(recordedChampion.architecture.policyTailPrecision, .mixedFinalProjection)
        let unrecordedTrainer = try CheckpointManager.decodeAnyModelFile(try preV12(current))
        let lonely = try CheckpointManager.decodeSessionChampionFile(
            try preV12(current), source: "c", trainerFile: unrecordedTrainer)
        XCTAssertEqual(lonely.architecture.policyTailPrecision, .mixedFinalProjection)
    }

    // MARK: - Derive

    func testTheDeriveOperationRewritesNoTensorAndRunsOnATrainedSource() throws {
        // A trained plain model: a positive training step refuses every
        // tensor-rewriting operation.
        let arch = try bf16(.mixedFinalProjection)
        let meta = ModelCheckpointMetadata(creator: "test", trainingStep: 5000, parentModelID: "", notes: "trained")
        let source = try SafetensorsModelIO.encode(
            modelID: "20261007-2-TRND", createdAtUnix: 1_790_000_000, metadata: meta, weights: weights(arch),
            architecture: arch, includesVelocity: false,
            lineage: try LineageRecord.forTests(trainerCompletedSteps: nil, corpus: nil))
        let operation = try SetPolicyTailPrecisionDeriveOperation.kind.make("fp32_from_pre_bn", nil)
        let target = try operation.apply(to: arch)
        XCTAssertEqual(target.policyTailPrecision, .float32FromPreBatchNorm)
        XCTAssertTrue(try operation.tensorRewrites(source: arch, target: target).isEmpty)
        let result = try ModelDerivation.derive(
            sourceData: source, sourceName: "trained.safetensors", operations: [operation],
            newModelID: "20261007-3-DRVD", createdAtUnix: 1_790_000_100, build: "test", invocationArguments: ["dcm"])
        let derived = try SafetensorsModelIO.decode(result.data, valueHead: .asStored)
        XCTAssertEqual(derived.architecture.policyTailPrecision, .float32FromPreBatchNorm)
        let sourceWeights = try SafetensorsModelIO.decode(source, valueHead: .asStored).file.weights
        XCTAssertEqual(derived.file.weights.map { $0.map(\.bitPattern) }, sourceWeights.map { $0.map(\.bitPattern) })

        // The dtype rule is the architecture's validation, and a no-op is
        // refused like every derive.
        let doesNotApply = try SetPolicyTailPrecisionDeriveOperation.kind.make("does_not_apply", nil)
        XCTAssertThrowsError(try ModelDerivation.derive(
            sourceData: source, sourceName: "trained.safetensors", operations: [doesNotApply],
            newModelID: "20261007-4-DRVD", createdAtUnix: 1_790_000_100, build: "test", invocationArguments: ["dcm"])) { error in
            guard case .invalidTargetArchitecture? = error as? ModelDerivation.DeriveError else {
                return XCTFail("expected invalidTargetArchitecture, got \(error)")
            }
        }
        let noOp = try SetPolicyTailPrecisionDeriveOperation.kind.make("mixed_final_projection", nil)
        XCTAssertThrowsError(try noOp.apply(to: arch))
        XCTAssertThrowsError(try SetPolicyTailPrecisionDeriveOperation.kind.make("fp16", nil))
    }

    // MARK: - Build New Model

    @MainActor
    func testBuildNewModelRoundTripsTheTail() throws {
        let arch = try bf16(.float32FromPreBatchNorm)
        let model = BuildNewModelModel(NamedArchitecture(label: "t", architecture: arch))
        XCTAssertEqual(model.architecture, arch)
        XCTAssertEqual(model.reducedPrecisionPolicyTail, .float32FromPreBatchNorm)
        model.computeDataType = .float32
        XCTAssertEqual(model.architecture.policyTailPrecision, .doesNotApply)
        XCTAssertNoThrow(try model.architecture.validate())
        model.computeDataType = .bFloat16
        XCTAssertEqual(model.architecture, arch, "the picker's choice survives a round trip through fp32")
        model.load(NamedArchitecture(label: "f", architecture: try fp32()))
        XCTAssertEqual(model.architecture.policyTailPrecision, .doesNotApply)
        XCTAssertEqual(model.reducedPrecisionPolicyTail, .mixedFinalProjection)
    }

    @MainActor
    func testThePickerIsDisabledOnFP32() {
        let fp32Presentation = PolicyTailPrecisionPicker.presentation(computeDataType: .float32)
        XCTAssertFalse(fp32Presentation.isEnabled)
        XCTAssertEqual(fp32Presentation.entries, [.doesNotApply])
        for dtype in [ComputeDataType.bFloat16, .float16] {
            let presentation = PolicyTailPrecisionPicker.presentation(computeDataType: dtype)
            XCTAssertTrue(presentation.isEnabled)
            XCTAssertEqual(presentation.entries, PolicyTailPrecisionSetting.reducedPrecisionCases)
        }
    }

    // MARK: - Legacy .dcmmodel

    func testTheDcmmodelWriterRefusesATailItsPresetDoesNotHave() throws {
        let preset = NetworkArchitecture.current
        var otherTail = preset
        otherTail.policyTailPrecision = .float32FromPreBatchNorm
        let meta = ModelCheckpointMetadata(creator: "test", trainingStep: nil, parentModelID: "", notes: "")
        XCTAssertThrowsError(try ModelCheckpointFile(
            modelID: "20261007-5-DCMM", createdAtUnix: 1_790_000_000, metadata: meta,
            weights: weights(otherTail), architecture: otherTail).encode()) { error in
            guard case ModelCheckpointError.policyTailNotLegacyEncodable = error else {
                return XCTFail("expected policyTailNotLegacyEncodable, got \(error)")
            }
        }
        XCTAssertNoThrow(try ModelCheckpointFile(
            modelID: "20261007-6-DCMM", createdAtUnix: 1_790_000_000, metadata: meta,
            weights: weights(preset), architecture: preset).encode())
    }

    // MARK: - Lineage schema 3

    func testTheSchemaThreeValidatorAcceptsDoesNotApply() throws {
        for tail in PolicyTailPrecisionSetting.allCases {
            XCTAssertNoThrow(try LineageRecord.TrainingConfiguration(
                pathKind: .replay, policyTailPrecision: tail.rawValue, budget: .none, parameterChanges: [],
                championChanges: [], vsuci: nil, selfPlayDirichlet: nil, startValueHeadRecentered: .recorded(false),
                scheduleAtSave: nil, replayRatio: nil, healthAlarms: .unrecorded), tail.rawValue)
        }
        XCTAssertThrowsError(try LineageRecord.TrainingConfiguration(
            pathKind: .replay, policyTailPrecision: "fp16", budget: .none, parameterChanges: [],
            championChanges: [], vsuci: nil, selfPlayDirichlet: nil, startValueHeadRecentered: .recorded(false),
            scheduleAtSave: nil, replayRatio: nil, healthAlarms: .unrecorded))
    }

    // MARK: - Behavior fingerprint

    /// The fingerprint's hashed header reads the tail from the architecture
    /// in the same text it read the process value with, so a pre-v12 bf16
    /// file trained under `fp32_from_pre_bn` (its flat key) hashes the header
    /// the build that trained it hashed.
    func testAStoredPreV12FilesFingerprintHeaderIsUnchanged() throws {
        let file = try SafetensorsModelIO.decode(
            try preV12(try encodedModel(try bf16(.mixedFinalProjection)), flatKey: "fp32_from_pre_bn"))
        XCTAssertEqual(BehaviorFingerprint.header(for: .init(arch: file.architecture)),
                       "dcm-behavior-fingerprint recipe=3 policy_tail=fp32_from_pre_bn")
        XCTAssertEqual(BehaviorFingerprint.recipe, 3)
        XCTAssertEqual(BehaviorFingerprint.header(for: .init(arch: try fp32())),
                       "dcm-behavior-fingerprint recipe=3 policy_tail=does_not_apply")
    }
}
