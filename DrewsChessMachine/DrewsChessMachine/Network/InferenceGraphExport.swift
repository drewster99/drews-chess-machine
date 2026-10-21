import Foundation
import MetalPerformanceShadersGraph

/// A model file as this build reads it, from its header alone: every rule a
/// load applies to a header (`SafetensorsModelIO.inspect`) — the architecture
/// after every format-version gate and legacy resolution, the tensors against
/// the plan, the trainer schedule, the lineage record, the step reading — plus
/// the puzzle test-set results as the app reads them. The one description
/// outside tools read, so no script re-derives the loader's rules
/// (`--describe-model`, `scripts/dcm_arch.py` / `dcm_lineage.py`, and
/// `model.json` in every `--export-inference-graph` folder).
///
/// Header-only, so a scan of thousands of files reads kilobytes per file: a
/// description refuses exactly what a load refuses except a damaged tensor
/// data region (content hash), which needs the data.
struct ModelDescription {
    let source: String
    let metadata: [String: String]
    let inspected: SafetensorsModelIO.InspectedHeader
    let testSetResults: ModelTestSetResultsInFile

    enum DescriptionError: LocalizedError {
        case notSafetensors(String)
        case duplicateTensor(String)
        case missingDecodedValue(String)

        var errorDescription: String? {
            switch self {
            case .notSafetensors(let name):
                return "\(name): a description reads .safetensors files"
            case .duplicateTensor(let name):
                return "the header names tensor '\(name)' twice"
            case .missingDecodedValue(let key):
                return "the header has no \(key), though it was decoded from it"
            }
        }
    }

    /// The description of the model file at `url`. A session's
    /// `champion.safetensors` takes the policy tail its trainer file lends it,
    /// as the session load does (`CheckpointManager.sessionTrainerPolicyTail`).
    static func describe(fileAt url: URL) throws -> ModelDescription {
        guard url.pathExtension == "safetensors" else { throw DescriptionError.notSafetensors(url.lastPathComponent) }
        var trainerTail: ArchitectureFormat.RecordedPolicyTailPrecision?
        let folder = url.deletingLastPathComponent()
        if url.lastPathComponent == SessionCheckpointLayout.championFilename,
           folder.pathExtension == "dcmsession" {
            let trainer = try describe(headerAt: folder.appendingPathComponent(SessionCheckpointLayout.trainerFilename),
                                       sessionTrainerPolicyTail: nil)
            trainerTail = CheckpointManager.sessionTrainerPolicyTail(
                trainerArchitecture: trainer.inspected.architecture,
                trainerTailOrigin: trainer.inspected.architectureFormat.legacyLog.policyTailPrecisionOrigin)
        }
        return try describe(headerAt: url, sessionTrainerPolicyTail: trainerTail)
    }

    private static func describe(
        headerAt url: URL, sessionTrainerPolicyTail: ArchitectureFormat.RecordedPolicyTailPrecision?
    ) throws -> ModelDescription {
        let header = try SafetensorsFile.readHeader(at: url)
        var shapes: [String: [Int]] = [:]
        for entry in header.entries {
            guard shapes.updateValue(entry.shape, forKey: entry.name) == nil else {
                throw DescriptionError.duplicateTensor(entry.name)
            }
        }
        let inspected = try SafetensorsModelIO.inspect(
            tensorShapes: shapes, metadata: header.metadata, source: url.lastPathComponent,
            sessionTrainerPolicyTail: sessionTrainerPolicyTail)
        return ModelDescription(source: url.lastPathComponent, metadata: header.metadata, inspected: inspected,
                                testSetResults: ModelTestSetResultsField.reading(fromMetadata: header.metadata))
    }

    /// The description as a JSON object. The lineage record and the test-set
    /// results are given as stored (the app has decoded and checked them), so
    /// a reader sees each schema's own field layout.
    func jsonObject() throws -> [String: Any] {
        let arch = inspected.architecture
        let architectureObject = try JSONSerialization.jsonObject(with: JSONEncoder().encode(arch))
        let blocksObject = try JSONSerialization.jsonObject(with: JSONEncoder().encode(arch.expandedBlocks))
        let reading = inspected.stepReading
        var object: [String: Any] = [
            "source": source,
            "model_id": inspected.modelID,
            "creator": inspected.metadata.creator,
            "format_version": inspected.architectureFormat.formatVersion,
            "legacy_resolutions": inspected.architectureFormat.legacyLog.resolutions,
            "architecture": architectureObject,
            "expanded_blocks": blocksObject,
            "input_planes": arch.inputPlanes,
            "policy_size": ChessNetwork.policySize,
            "value_head_classes": arch.valueHeadClasses,
            "parameter_count": arch.parameterCount,
            "weight_plan": arch.weightTensorPlan().map { ["name": $0.name, "shape": $0.shape, "kind": $0.kind.rawValue] },
            "includes_optimizer_velocity": inspected.hasVelocity,
            "step_reading": [
                "basis": reading.basis.rawValue,
                "stated_training_step": Self.json(reading.statedTrainingStep),
                "trainer_step": Self.json(reading.trainerStep),
                "segment_step": Self.json(reading.segmentStep),
                "legacy_note": Self.json(reading.legacyResolution),
            ] as [String: Any],
        ]
        switch inspected.provenance.lineage {
        case .unrecorded(let formatVersion):
            object["lineage"] = ["status": "unrecorded", "format_version": formatVersion] as [String: Any]
        case .recorded:
            object["lineage"] = ["status": "recorded",
                                 "record": try storedJSON(LineageRecord.metadataKey)] as [String: Any]
        }
        switch testSetResults {
        case .notRecorded:
            object["test_set_results"] = ["status": "not_recorded"]
        case .unreadable(let reason):
            object["test_set_results"] = ["status": "unreadable", "reason": reason]
        case .recorded(.failed(let reason)):
            object["test_set_results"] = ["status": "failed", "reason": reason]
        case .recorded(.evaluated):
            object["test_set_results"] = ["status": "evaluated",
                                          "record": try storedJSON(ModelTestSetResultsField.metadataKey)]
        }
        return object
    }

    /// The header value under `key` (one the inspection decoded) as JSON.
    private func storedJSON(_ key: String) throws -> Any {
        guard let text = metadata[key] else { throw DescriptionError.missingDecodedValue(key) }
        return try JSONSerialization.jsonObject(with: Data(text.utf8))
    }

    private static func json<T>(_ value: T?) -> Any { value.map { $0 as Any } ?? NSNull() }

    func encodedJSON(pretty: Bool) throws -> Data {
        var options: JSONSerialization.WritingOptions = [.sortedKeys, .withoutEscapingSlashes]
        if pretty { options.insert(.prettyPrinted) }
        return try JSONSerialization.data(withJSONObject: try jsonObject(), options: options)
    }
}

/// Writes a model's inference graph (`InferenceGraph`) and weights in the form
/// `scripts/dcm_coreml_export.py` turns into a Core ML model.
///
/// The folder holds three files, all new (an existing folder is refused):
/// - `graph.json` — the recorded forward pass;
/// - `weights.bin` — every plan tensor as little-endian float32, in plan
///   order, in the layout the MPSGraph variable holds and rounded to its data
///   type (offsets in `graph.json`);
/// - `model.json` — the `ModelDescription` of the source file.
///
/// The weights are the ones `loadWeights` gives the network
/// (`ModelCheckpointFile.networkWeights`: un-transposed from the file's layout,
/// the value head recentered as every load does, then rounded to each
/// variable's type as the load rounds them), so the exporter applies no load
/// rule of its own. The folder is written under a staging name and moved into
/// place whole, so an interrupted export never leaves a partial folder under
/// the final name.
enum InferenceGraphExport {
    static let graphFileName = "graph.json"
    static let weightsFileName = "weights.bin"
    static let modelFileName = "model.json"

    enum ExportError: LocalizedError, Equatable {
        case weightCountMismatch(have: Int, want: Int)
        case weightSizeMismatch(name: String, have: Int, want: Int)
        case weightOffsetMismatch(name: String, have: Int, want: Int)
        case notLittleEndian
        case recordingMissing
        case stagingRemains(path: String, write: String, cleanup: String)

        var errorDescription: String? {
            switch self {
            case .weightCountMismatch(let have, let want):
                return "the model file has \(have) network tensors; its architecture's plan has \(want)"
            case .weightSizeMismatch(let name, let have, let want):
                return "tensor \(name) holds \(have) values; the graph declares \(want)"
            case .weightOffsetMismatch(let name, let have, let want):
                return "tensor \(name) would be written at byte \(have); the graph places it at \(want)"
            case .notLittleEndian:
                return "weights.bin is little-endian float32; this machine is not little-endian"
            case .recordingMissing:
                return "the network was built with a recorder but carries no recording"
            case .stagingRemains(let path, let write, let cleanup):
                return "writing the export failed (\(write)) and its staging folder \(path) could not be removed: \(cleanup)"
            }
        }
    }

    /// The inference forward pass of `architecture`, recorded from a network
    /// built to it. Builds the MPSGraph (no GPU work, no weights drawn).
    static func record(architecture: NetworkArchitecture) throws -> InferenceGraph {
        let recorder = InferenceGraphRecorder()
        let network = try ChessNetwork(arch: architecture, bnMode: .inference,
                                       initialization: .overwrittenByLoad, graphRecorder: recorder)
        guard let graph = network.recordedInferenceGraph else { throw ExportError.recordingMissing }
        return graph
    }

    /// The bytes of `weights.bin`: `weights` (plan order) rounded to each
    /// weight's data type as `loadWeights` rounds them, as little-endian
    /// float32, each checked against the graph's element count and offset.
    static func weightsFileData(graph: InferenceGraph, weights: [[Float]]) throws -> Data {
        guard UInt32(1).littleEndian == 1 else { throw ExportError.notLittleEndian }
        guard weights.count == graph.weights.count else {
            throw ExportError.weightCountMismatch(have: weights.count, want: graph.weights.count)
        }
        var data = Data()
        data.reserveCapacity(graph.weights.reduce(0) { $0 + $1.elementCount } * MemoryLayout<Float>.size)
        for (spec, values) in zip(graph.weights, weights) {
            guard values.count == spec.elementCount else {
                throw ExportError.weightSizeMismatch(name: spec.name, have: values.count, want: spec.elementCount)
            }
            guard data.count == spec.byteOffset else {
                throw ExportError.weightOffsetMismatch(name: spec.name, have: data.count, want: spec.byteOffset)
            }
            let held = rounded(values, to: spec.dataType)
            held.withUnsafeBufferPointer { data.append($0) }
        }
        return data
    }

    /// `values` as a variable of `dataType` holds them: the conversion
    /// `loadWeights` makes (`makeWeightData`) and back.
    static func rounded(_ values: [Float], to dataType: GraphDataType) -> [Float] {
        let mps: MPSDataType
        switch dataType {
        case .float32: return values
        case .float16: mps = .float16
        case .bfloat16: mps = .bFloat16
        }
        return ChessNetwork.decodeWeightData(ChessNetwork.makeWeightData(values, dataType: mps), dataType: mps)
    }

    static func graphFileData(_ graph: InferenceGraph) throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        return try encoder.encode(graph)
    }

    /// Writes the three files into `folder`, which must not exist: staged in a
    /// new sibling folder, then moved into place without replacing anything. A
    /// failure removes the staging folder this call created (by identity).
    static func write(graph: InferenceGraph, weights: [[Float]], description: ModelDescription, to folder: URL) throws {
        let graphData = try graphFileData(graph)
        let weightsData = try weightsFileData(graph: graph, weights: weights)
        let modelData = try description.encodedJSON(pretty: true)
        try FileSafety.requireStageableDestination(folder)
        let staging = FileSafety.temporarySibling(of: folder)
        let stagingIdentity = try FileSafety.createNewDirectory(at: staging)
        do {
            _ = try FileSafety.writeNewFile(graphData, at: staging.appendingPathComponent(graphFileName))
            _ = try FileSafety.writeNewFile(weightsData, at: staging.appendingPathComponent(weightsFileName))
            _ = try FileSafety.writeNewFile(modelData, at: staging.appendingPathComponent(modelFileName))
            try FileSafety.renameWithoutReplacing(from: staging, to: folder)
        } catch {
            do {
                try FileSafety.removeOwnedItem(at: staging, identity: stagingIdentity)
            } catch let cleanupError {
                throw ExportError.stagingRemains(path: staging.path, write: error.localizedDescription,
                                                 cleanup: cleanupError.localizedDescription)
            }
            throw error
        }
    }
}
