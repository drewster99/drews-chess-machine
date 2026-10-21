import Foundation
import MetalPerformanceShadersGraph

/// A model file as this build reads it: for a safetensors file, every rule a
/// load applies to a header (`SafetensorsModelIO.inspect`) — the architecture
/// after every format-version gate and legacy resolution, the tensors against
/// the plan, the trainer schedule, the lineage record, the step reading — plus
/// the puzzle test-set results as the app reads them. The one description
/// outside tools read, so no script re-derives the loader's rules
/// (`--describe-model`, `scripts/dcm_arch.py` / `dcm_lineage.py`, and
/// `model.json` in every `--export-inference-graph` folder).
///
/// A safetensors file is described from its header alone, so a scan of
/// thousands of files reads kilobytes per file: a description refuses exactly
/// what a load refuses except a damaged tensor data region (content hash),
/// which needs the data. A legacy `.dcmmodel` has no header to read apart from
/// its weights, so it is described from a full decode (its architecture is a
/// compiled-in preset; it records no lineage and no test-set results). Files
/// are told apart by their first bytes, as every load tells them apart.
struct ModelDescription {
    let source: String
    let modelID: String
    let creator: String
    let architecture: NetworkArchitecture
    let formatVersion: Int
    let legacyResolutions: [String]
    let includesOptimizerVelocity: Bool
    let stepReading: ModelFileStepReading
    let lineage: LineageRecord.Presence
    let testSetResults: ModelTestSetResultsInFile
    /// The header's `__metadata__`, from which a recorded lineage record and
    /// test-set results are given as stored; empty for a `.dcmmodel`.
    let storedMetadata: [String: String]

    enum DescriptionError: LocalizedError {
        case missingDecodedValue(String)

        var errorDescription: String? {
            switch self {
            case .missingDecodedValue(let key):
                return "the header has no \(key), though it was decoded from it"
            }
        }
    }

    /// The description of the model file at `url`. A session's
    /// `champion.safetensors` takes the policy tail its trainer file lends it,
    /// as the session load does (`CheckpointManager.sessionTrainerPolicyTail(forFileAt:)`).
    static func describe(fileAt url: URL) throws -> ModelDescription {
        let source = url.lastPathComponent
        if try startsLikeLegacyModelFile(url) {
            return try describe(bytes: try CheckpointManager.readModelFileBytes(at: url), source: source,
                                sessionTrainerPolicyTail: nil)
        }
        let tail = try CheckpointManager.sessionTrainerPolicyTail(forFileAt: url)
        return try describe(header: try SafetensorsFile.readHeader(at: url), source: source,
                            sessionTrainerPolicyTail: tail)
    }

    /// The description of a whole model file's bytes: for a caller that loads
    /// these same bytes and must describe exactly the file it loaded.
    static func describe(
        bytes data: Data, source: String,
        sessionTrainerPolicyTail: ArchitectureFormat.RecordedPolicyTailPrecision?
    ) throws -> ModelDescription {
        if CheckpointManager.isLegacyModelFile(data) {
            let file = try CheckpointManager.decodeAnyModelFile(data, valueHead: .recenterUnlessMarked, source: source)
            return ModelDescription(
                source: source, modelID: file.modelID, creator: file.metadata.creator,
                architecture: file.architecture, formatVersion: ArchitectureFormat.unversionedLegacyVersion,
                legacyResolutions: file.trainingStepReading.legacyResolution.map { [$0] } ?? [],
                includesOptimizerVelocity: file.includesOptimizerVelocity, stepReading: file.trainingStepReading,
                lineage: .unrecorded(formatVersion: ArchitectureFormat.unversionedLegacyVersion),
                testSetResults: .notRecorded, storedMetadata: [:])
        }
        return try describe(header: try SafetensorsFile.header(of: data), source: source,
                            sessionTrainerPolicyTail: sessionTrainerPolicyTail)
    }

    private static func describe(
        header: SafetensorsFile.Header, source: String,
        sessionTrainerPolicyTail: ArchitectureFormat.RecordedPolicyTailPrecision?
    ) throws -> ModelDescription {
        let inspected = try SafetensorsModelIO.inspect(header: header, source: source,
                                                       sessionTrainerPolicyTail: sessionTrainerPolicyTail)
        return ModelDescription(
            source: source, modelID: inspected.modelID, creator: inspected.metadata.creator,
            architecture: inspected.architecture, formatVersion: inspected.architectureFormat.formatVersion,
            legacyResolutions: inspected.architectureFormat.legacyLog.resolutions,
            includesOptimizerVelocity: inspected.hasVelocity, stepReading: inspected.stepReading,
            lineage: inspected.provenance.lineage,
            testSetResults: ModelTestSetResultsField.reading(fromMetadata: header.metadata),
            storedMetadata: header.metadata)
    }

    /// Whether the file at `url` starts with the legacy `.dcmmodel` magic:
    /// only its first bytes are read.
    private static func startsLikeLegacyModelFile(_ url: URL) throws -> Bool {
        let handle: FileHandle
        do {
            handle = try FileHandle(forReadingFrom: url)
        } catch {
            throw CheckpointManagerError.readFailed(url, error)
        }
        defer {
            do { try handle.close() } catch {
                SessionLogger.shared.log("[DESCRIBE] closing \(url.lastPathComponent) after reading its first bytes failed: \(error)")
            }
        }
        let prefix: Data
        do {
            prefix = try handle.read(upToCount: ModelCheckpointFile.magic.count) ?? Data()
        } catch {
            throw CheckpointManagerError.readFailed(url, error)
        }
        return CheckpointManager.isLegacyModelFile(prefix)
    }

    /// The description as a JSON object. The lineage record and the test-set
    /// results are given as stored (the app has decoded and checked them), so
    /// a reader sees each schema's own field layout.
    func jsonObject() throws -> [String: Any] {
        let arch = architecture
        let architectureObject = try JSONSerialization.jsonObject(with: JSONEncoder().encode(arch))
        let blocksObject = try JSONSerialization.jsonObject(with: JSONEncoder().encode(arch.expandedBlocks))
        let reading = stepReading
        var object: [String: Any] = [
            "source": source,
            "model_id": modelID,
            "creator": creator,
            "format_version": formatVersion,
            "legacy_resolutions": legacyResolutions,
            "architecture": architectureObject,
            "expanded_blocks": blocksObject,
            "input_planes": arch.inputPlanes,
            "policy_size": ChessNetwork.policySize,
            "value_head_classes": arch.valueHeadClasses,
            "parameter_count": arch.parameterCount,
            "weight_plan": arch.weightTensorPlan().map { ["name": $0.name, "shape": $0.shape, "kind": $0.kind.rawValue] },
            "includes_optimizer_velocity": includesOptimizerVelocity,
            "step_reading": [
                "basis": reading.basis.rawValue,
                "stated_training_step": Self.json(reading.statedTrainingStep),
                "trainer_step": Self.json(reading.trainerStep),
                "segment_step": Self.json(reading.segmentStep),
                "legacy_note": Self.json(reading.legacyResolution),
            ] as [String: Any],
        ]
        switch lineage {
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
        guard let text = storedMetadata[key] else { throw DescriptionError.missingDecodedValue(key) }
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
