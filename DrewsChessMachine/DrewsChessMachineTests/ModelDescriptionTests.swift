//
//  ModelDescriptionTests.swift
//  DrewsChessMachineTests
//
//  `ModelDescription` (`--describe-model`, `model.json`) and the file reads
//  the export and parity commands share with it: a description is the file a
//  tool loads — a session's champion with the tail its trainer file lends it,
//  a legacy `.dcmmodel` told apart by its first bytes — and the value-head
//  marker is judged only where it is used. None of these needs Metal.
//

import XCTest
@testable import DrewsChessMachine

final class ModelDescriptionTests: XCTestCase {

    private static let tailKey = NetworkArchitecture.CodingKeys.policyTailPrecision.rawValue

    private func weights(_ arch: NetworkArchitecture) -> [[Float]] {
        arch.weightTensorPlan().enumerated().map { tensorIndex, spec in
            (0..<spec.elementCount).map { Float(tensorIndex * 7 + $0 % 13) * 0.125 + 0.0625 }
        }
    }

    private func encodedModel(_ arch: NetworkArchitecture) throws -> Data {
        let meta = ModelCheckpointMetadata(creator: "test", trainingStep: nil, parentModelID: "", notes: "fixture")
        return try SafetensorsModelIO.encode(
            modelID: "20261021-1-DESC", createdAtUnix: 1_790_000_000, metadata: meta, weights: weights(arch),
            architecture: arch, includesVelocity: false,
            lineage: try LineageRecord.forTests(trainerCompletedSteps: nil, corpus: nil))
    }

    /// `data` with its header rewritten by `edit` (the content hash dropped, so
    /// the encoder writes the new one).
    private func rewritten(_ data: Data, _ edit: (inout [String: String]) throws -> Void) throws -> Data {
        let (tensors, decodedMetadata) = try SafetensorsFile.decode(data)
        var metadata = decodedMetadata
        metadata.removeValue(forKey: SafetensorsFile.contentHashKey)
        try edit(&metadata)
        return try SafetensorsFile.encode(tensors: tensors, metadata: metadata)
    }

    /// `data` as a format-v11 file: no `policy_tail_precision` in its
    /// architecture, the flat trainer key set to `flatKey` when given.
    private func preV12(_ data: Data, flatKey: String? = nil) throws -> Data {
        try rewritten(data) { metadata in
            metadata[SafetensorsModelIO.Key.formatVersion] = "11"
            let archText = try XCTUnwrap(metadata[SafetensorsModelIO.Key.architecture])
            var object = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(archText.utf8)) as? [String: Any])
            object.removeValue(forKey: Self.tailKey)
            metadata[SafetensorsModelIO.Key.architecture] = String(
                decoding: try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys]), as: UTF8.self)
            if let flatKey { metadata[SafetensorsModelIO.Key.trainerPolicyTailPrecision] = flatKey }
        }
    }

    private func temporaryDirectory(named name: String) throws -> URL {
        let parent = FileManager.default.temporaryDirectory
            .appendingPathComponent("model-description-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: false)
        addTeardownBlock {
            try FileManager.default.removeItem(at: parent)
        }
        let url = parent.appendingPathComponent(name, isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: false)
        return url
    }

    /// Protects: a tool handed a session champion's path (`--describe-model`,
    /// `--export-inference-graph`, `--coreml-parity`) reads it as the network
    /// the session load builds: a pre-v12 champion that records no tail takes
    /// its trainer file's (PT-D3 rule 4).
    /// Fails if: `sessionTrainerPolicyTail(forFileAt:)` stops reading the
    /// sibling trainer file, or the description or the bytes-based load drops
    /// the tail it is given — then `model.json` and the exported graph
    /// describe different networks, and both differ from the session load.
    func testAToolReadsASessionChampionWithItsTrainerFilesTail() throws {
        let bf16 = try NetworkArchitecture.current.withComputeDataType(.bFloat16, tail: .mixedFinalProjection)
        let current = try encodedModel(bf16)
        let trainerData = try preV12(current, flatKey: "fp32_from_pre_bn")
        let championData = try preV12(current)
        let session = try temporaryDirectory(named: "20261021-000000-20261021-1-DESC-manual.dcmsession")
        try trainerData.write(to: session.appendingPathComponent(SessionCheckpointLayout.trainerFilename))
        let championURL = session.appendingPathComponent(SessionCheckpointLayout.championFilename)
        try championData.write(to: championURL)

        let sessionLoad = try CheckpointManager.decodeSessionChampionFile(
            championData, source: "champion", trainerFile: try CheckpointManager.decodeAnyModelFile(trainerData))
        XCTAssertEqual(sessionLoad.architecture.policyTailPrecision, .float32FromPreBatchNorm)

        XCTAssertEqual(try ModelDescription.describe(fileAt: championURL).architecture, sessionLoad.architecture)
        let tail = try CheckpointManager.sessionTrainerPolicyTail(forFileAt: championURL)
        XCTAssertEqual(try ModelDescription.describe(bytes: championData, source: "champion",
                                                     sessionTrainerPolicyTail: tail).architecture,
                       sessionLoad.architecture)
        XCTAssertEqual(try CheckpointManager.loadModelFile(fromBytes: championData, source: "champion",
                                                           sessionTrainerPolicyTail: tail).architecture,
                       sessionLoad.architecture)
    }

    /// Protects: only a session's `champion.safetensors` borrows a tail.
    /// Fails if: the rule is applied by file name alone (a loose
    /// `champion.safetensors`) or to the trainer file itself.
    func testOnlyASessionChampionBorrowsATail() throws {
        XCTAssertNil(try CheckpointManager.sessionTrainerPolicyTail(
            forFileAt: URL(fileURLWithPath: "/no-such-folder/champion.safetensors")))
        XCTAssertNil(try CheckpointManager.sessionTrainerPolicyTail(
            forFileAt: URL(fileURLWithPath: "/no-such-folder/x.dcmsession/trainer.safetensors")))
    }

    /// Protects: a legacy `.dcmmodel` stays describable (and so exportable),
    /// told apart by its first bytes as every load tells it apart.
    /// Fails if: describe dispatches on the extension or refuses anything but
    /// safetensors — a `.dcmmodel` export, which a load accepts, would fail.
    func testALegacyDcmmodelIsDescribedFromItsFirstBytes() throws {
        let preset = NetworkArchitecture.current
        let meta = ModelCheckpointMetadata(creator: "legacy-writer", trainingStep: nil, parentModelID: "", notes: "")
        let data = try ModelCheckpointFile(
            modelID: "20261021-2-DCMM", createdAtUnix: 1_790_000_000, metadata: meta,
            weights: weights(preset), architecture: preset).encode()
        // A name that says nothing about the format: the bytes decide.
        let url = try temporaryDirectory(named: "legacy").appendingPathComponent("model.bin")
        try data.write(to: url)

        let description = try ModelDescription.describe(fileAt: url)
        XCTAssertEqual(description.modelID, "20261021-2-DCMM")
        XCTAssertEqual(description.creator, "legacy-writer")
        XCTAssertEqual(description.architecture, preset)
        XCTAssertEqual(description.formatVersion, ArchitectureFormat.unversionedLegacyVersion)
        XCTAssertEqual(description.lineage, .unrecorded(formatVersion: ArchitectureFormat.unversionedLegacyVersion))
        XCTAssertEqual(description.testSetResults, .notRecorded)
        let object = try description.jsonObject()
        XCTAssertEqual((object["lineage"] as? [String: Any])?["status"] as? String, "unrecorded")
        XCTAssertEqual((object["test_set_results"] as? [String: Any])?["status"] as? String, "not_recorded")
    }

    /// Protects: a damaged value-head centering marker is refused only by a
    /// decode that recenters; an `.asStored` read (the numerics audit, derive
    /// and graft sources) still opens the file, as before the header rules
    /// moved into `inspect`.
    /// Fails if: the marker is validated in the header inspection, for every
    /// decode, again.
    func testADamagedMarkerIsRefusedOnlyWhereTheValueHeadIsRecentered() throws {
        let damaged = try rewritten(try encodedModel(NetworkArchitecture.current)) { metadata in
            metadata[ValueHeadRecentering.metadataKey] = "maybe"
        }
        XCTAssertEqual(try SafetensorsModelIO.decode(damaged, valueHead: .asStored).file.valueHeadCentering,
                       .keptAsStored)
        XCTAssertThrowsError(try SafetensorsModelIO.decode(damaged, valueHead: .recenterUnlessMarked)) { error in
            guard case ValueHeadRecentering.RecenteringError.badMarker("maybe") = error else {
                return XCTFail("expected badMarker, got \(error)")
            }
        }
    }
}
