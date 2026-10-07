//
//  InitSeedRecordingTests.swift
//  DrewsChessMachineTests
//
//  Where a model's init seed travels (determinism plan, phase P5): the
//  lineage record's `rng.init_seed` / `init_scheme` of a fresh mint (its only
//  home), the seeded `--derive-model` β re-draw, and the Build-New-Model seed
//  field.
//

import XCTest
@testable import DrewsChessMachine

final class InitSeedRecordingTests: XCTestCase {

    private static let architecture: NetworkArchitecture = {
        var arch = NetworkArchitecture(
            inputEncoding: .basic30, channels: 16, numBlocks: 2, stemConvKernelSize: 3,
            activationFunction: .relu, blockActivationStyle: .pre,
            blockSkipMerge: .cleanAdd, blockUseRezero: false, rezeroAlphaInit: 0.5,
            blockConv1KernelSize: 3, blockConv2KernelSize: 3,
            blockSeStyle: .scaleAndBias, blockSeReductionRatio: 4,
            policyHeadStyle: .intermediateConv, policyPreConvChannels: 16,
            valueHeadStyle: .wdlSoftmax, valueHeadConvChannels: 4, valueHeadHiddenUnits: 16,
            computeDataType: .float32
        )
        arch.blockGroups[0].seBetaInit = .zero
        return arch
    }()

    private func encoded(metadata: ModelCheckpointMetadata, lineage: LineageRecord? = nil) throws -> Data {
        let weights = Self.architecture.weightTensorPlan().enumerated().map { index, spec in
            (0..<spec.elementCount).map { Float(index * 5 + $0 % 11) * 0.25 + 0.125 }
        }
        return try SafetensorsModelIO.encode(
            modelID: "20261002-1-INIT", createdAtUnix: 1_790_000_000, metadata: metadata, weights: weights,
            architecture: Self.architecture, includesVelocity: false,
            lineage: try lineage ?? LineageRecord.forTests(trainerCompletedSteps: nil, corpus: nil))
    }

    private func mintLineage(_ initialization: ModelInitRecord) throws -> LineageRecord {
        try LineageTracker.mintRecord(pathKind: .newModel, argv: ["DrewsChessMachine", "--new-model"],
                                      initialization: initialization, naming: .unnamedWithoutPreset,
                                      at: Date(timeIntervalSince1970: 1_790_000_000))
    }

    private func recordedInitialization(_ data: Data) throws -> ModelInitRecord? {
        let decoded = try SafetensorsModelIO.decode(data, valueHead: .asStored)
        let lineage = try XCTUnwrap(decoded.file.safetensorsProvenance?.lineage.record)
        return lineage.rng.initialization
    }

    // MARK: File metadata

    /// A mint's init seed and scheme travel in the lineage record only; the
    /// header carries no second copy.
    func testInitRecordRoundTripsThroughTheLineageRecord() throws {
        let record = ModelInitRecord(initSeed: 0xFFFF_FFFF_FFFF_FFF1, scheme: WeightInitScheme.current)
        let data = try encoded(
            metadata: ModelCheckpointMetadata(creator: "new-model", trainingStep: nil, parentModelID: "", notes: "fresh"),
            lineage: try mintLineage(record))
        let (_, metadata) = try SafetensorsFile.decode(data)
        XCTAssertNil(metadata["init_seed"])
        XCTAssertNil(metadata["init_scheme"])
        let lineageText = try XCTUnwrap(metadata[LineageRecord.metadataKey])
        XCTAssertTrue(lineageText.contains("\"init_seed\":\"18446744073709551601\""), lineageText)
        XCTAssertTrue(lineageText.contains("\"init_scheme\":\"\(WeightInitScheme.current)\""), lineageText)
        XCTAssertEqual(try recordedInitialization(data), record)
    }

    /// A model made from another file's weights records no init seed of its
    /// own: the derive copies weights, it draws none.
    func testFileWithoutInitRecordDecodesToNil() throws {
        let source = try encoded(metadata: ModelCheckpointMetadata(
            creator: "test", trainingStep: nil, parentModelID: "", notes: "none"))
        let derived = try derive(source, [SetSEBetaInitDeriveOperation(value: .glorot, groupIndices: nil, initSeed: 4242)])
        XCTAssertNil(try recordedInitialization(derived.data))
    }

    func testMalformedInitRecordsAreRefused() throws {
        let base = try encoded(
            metadata: ModelCheckpointMetadata(creator: "test", trainingStep: nil, parentModelID: "", notes: "none"),
            lineage: try mintLineage(ModelInitRecord(initSeed: 12, scheme: WeightInitScheme.current)))
        let cases: [[String: String?]] = [
            ["init_seed": "12", "init_scheme": nil],
            ["init_seed": nil, "init_scheme": "dcm-init-1"],
            ["init_seed": "-1", "init_scheme": "dcm-init-1"],
            ["init_seed": "0x10", "init_scheme": "dcm-init-1"],
            ["init_seed": "12", "init_scheme": ""],
        ]
        for edits in cases {
            let (tensors, decodedMetadata) = try SafetensorsFile.decode(base)
            var metadata = decodedMetadata
            metadata.removeValue(forKey: SafetensorsFile.contentHashKey)
            let lineageText = try XCTUnwrap(metadata[LineageRecord.metadataKey])
            var lineage = try XCTUnwrap(
                try JSONSerialization.jsonObject(with: Data(lineageText.utf8)) as? [String: Any])
            var rng = try XCTUnwrap(lineage["rng"] as? [String: Any])
            for (key, value) in edits { rng[key] = value.map { $0 as Any } ?? NSNull() }
            lineage["rng"] = rng
            let edited = try JSONSerialization.data(withJSONObject: lineage, options: [.sortedKeys])
            metadata[LineageRecord.metadataKey] = String(decoding: edited, as: UTF8.self)
            let data = try SafetensorsFile.encode(tensors: tensors, metadata: metadata)
            XCTAssertThrowsError(try SafetensorsModelIO.decode(data, valueHead: .asStored), "\(edits)") { error in
                guard case SafetensorsModelIO.IOError.malformedLineage = error else {
                    return XCTFail("expected malformedLineage for \(edits), got \(error)")
                }
            }
        }
    }

    /// The positional `.dcmmodel` layout has no place for a lineage record —
    /// and so for an init seed — so the legacy writer refuses a file carrying
    /// one rather than dropping it.
    func testLegacyWriterRefusesAFileWithALineageRecord() throws {
        let data = try encoded(
            metadata: ModelCheckpointMetadata(creator: "new-model", trainingStep: nil, parentModelID: "", notes: "fresh"),
            lineage: try mintLineage(ModelInitRecord(initSeed: 1, scheme: WeightInitScheme.current)))
        let file = try SafetensorsModelIO.decode(data, valueHead: .asStored).file
        XCTAssertNotNil(file.safetensorsProvenance?.lineage.record)
        XCTAssertThrowsError(try file.encode()) { error in
            guard case ModelCheckpointError.encodingFailed(let message) = error, message.contains("lineage") else {
                return XCTFail("expected a lineage refusal, got \(error)")
            }
        }
    }

    // MARK: Derive

    private func derive(_ source: Data, _ operations: [any DeriveOperation]) throws -> ModelDerivation.Result {
        try ModelDerivation.derive(
            sourceData: source, sourceName: "source.safetensors", operations: operations,
            newModelID: "20261002-2-DRVE", createdAtUnix: 1_790_000_100, build: "test", invocationArguments: ["test"], renamedTo: nil)
    }

    /// A seeded `glorot` β re-draw writes exactly the β rows a fresh mint
    /// with that init seed has, records the seed and scheme, and repeats bit
    /// for bit.
    func testSeededBetaRedrawIsAFreshMintsBetaRows() throws {
        let source = try encoded(metadata: ModelCheckpointMetadata(
            creator: "test", trainingStep: nil, parentModelID: "", notes: "fixture"))
        let operation = SetSEBetaInitDeriveOperation(value: .glorot, groupIndices: nil, initSeed: 4242)
        let first = try derive(source, [operation])
        let second = try derive(source, [operation])
        // The files differ in their lineage (each records when it was
        // written); the tensors must not.
        let firstWeights = try SafetensorsModelIO.decode(first.data, valueHead: .asStored).file.weights
        let secondWeights = try SafetensorsModelIO.decode(second.data, valueHead: .asStored).file.weights
        XCTAssertEqual(firstWeights.map { $0.map(\.bitPattern) }, secondWeights.map { $0.map(\.bitPattern) })

        let arguments = try XCTUnwrap(first.record.operations.first?.arguments)
        XCTAssertEqual(arguments["init_seed"], "4242")
        XCTAssertEqual(arguments["init_scheme"], WeightInitScheme.current)

        let target = try SetSEBetaInitDeriveOperation(value: .glorot, groupIndices: nil, initSeed: 4242)
            .apply(to: Self.architecture)
        let decoded = try SafetensorsModelIO.decode(first.data, valueHead: .asStored)
        let plan = target.weightTensorPlan()
        for block in 0..<2 {
            let name = "blocks.\(block).se_scalebias.fc2.weight"
            let index = try XCTUnwrap(plan.firstIndex { $0.name == name })
            let mint = try WeightInitScheme.nativeValues(initSeed: 4242, spec: plan[index], distribution: .glorotNormal)
            for range in SEScaleAndBiasBetaHalf.nativeWeightRanges(reducedChannels: 4, channels: 16) {
                for element in range {
                    XCTAssertEqual(decoded.file.weights[index][element].bitPattern, mint[element].bitPattern,
                                   "\(name)[\(element)]")
                }
            }
        }
    }

    func testZeroBetaDeriveRecordsNoSeed() throws {
        var arch = Self.architecture
        arch.blockGroups[0].seBetaInit = .glorot
        let weights = arch.weightTensorPlan().map { [Float](repeating: 0.5, count: $0.elementCount) }
        let source = try SafetensorsModelIO.encode(
            modelID: "20261002-3-ZERO", createdAtUnix: 1_790_000_000,
            metadata: ModelCheckpointMetadata(creator: "test", trainingStep: nil, parentModelID: "", notes: "fixture"),
            weights: weights, architecture: arch, includesVelocity: false,
            lineage: try LineageRecord.forTests(trainerCompletedSteps: nil, corpus: nil))
        let result = try derive(source, [SetSEBetaInitDeriveOperation(value: .zero, groupIndices: nil, initSeed: 9)])
        let arguments = try XCTUnwrap(result.record.operations.first?.arguments)
        XCTAssertNil(arguments["init_seed"])
        XCTAssertNil(arguments["init_scheme"])
    }

    // MARK: Build New Model

    @MainActor
    func testBuildScreenInitSeedEntry() {
        let model = BuildNewModelModel(NamedArchitecture(label: "test", architecture: Self.architecture))
        XCTAssertEqual(model.initSeedEntry, .drawnAtBuild)
        XCTAssertEqual(model.buildRequest, BuildNewModelRequest(architecture: Self.architecture, enteredInitSeed: nil,
                                                                naming: .unnamedWithoutPreset))
        model.initSeedText = " 18446744073709551615 "
        XCTAssertEqual(model.initSeedEntry, .entered(UInt64.max))
        XCTAssertEqual(model.buildRequest?.enteredInitSeed, UInt64.max)
        for bad in ["-1", "+3", "1.5", "18446744073709551616", "seed"] {
            model.initSeedText = bad
            guard case .invalid = model.initSeedEntry else {
                return XCTFail("'\(bad)' must be refused")
            }
            XCTAssertNil(model.buildRequest, bad)
        }
    }
}
