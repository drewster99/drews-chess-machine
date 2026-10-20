import Foundation

/// A model file as this build reads it: the architecture after every
/// format-version rule and legacy resolution the loader applies, the expanded
/// tower, and the weight plan. The one description outside tools read, so no
/// script re-derives the loader's rules (`--describe-model`, and `model.json` in
/// every `--export-inference-graph` folder).
struct ModelDescription: Encodable {
    let source: String
    let modelID: String
    /// The file's `dcm_format_version`; nil for a legacy `.dcmmodel`.
    let formatVersion: Int?
    /// What the loader filled in for fields the file's format predates, as the
    /// `[ARCH] legacy file` log line lists them.
    let legacyResolutions: [String]
    /// The resolved architecture in this build's current JSON form.
    let architecture: NetworkArchitecture
    /// One entry per residual block, in build order (`expandedBlocks`).
    let expandedBlocks: [BlockGroup]
    let inputPlanes: Int
    let policySize: Int
    let valueHeadClasses: Int
    let parameterCount: Int
    let weightPlan: [PlannedTensor]
    let includesOptimizerVelocity: Bool

    struct PlannedTensor: Encodable {
        let name: String
        let shape: [Int]
        let kind: String
    }

    enum CodingKeys: String, CodingKey {
        case source, architecture
        case modelID = "model_id"
        case formatVersion = "format_version"
        case legacyResolutions = "legacy_resolutions"
        case expandedBlocks = "expanded_blocks"
        case inputPlanes = "input_planes"
        case policySize = "policy_size"
        case valueHeadClasses = "value_head_classes"
        case parameterCount = "parameter_count"
        case weightPlan = "weight_plan"
        case includesOptimizerVelocity = "includes_optimizer_velocity"
    }

    init(file: ModelCheckpointFile, source: String) {
        let arch = file.architecture
        self.source = source
        modelID = file.modelID
        formatVersion = file.architectureFormat?.formatVersion
        legacyResolutions = file.architectureFormat?.legacyLog.resolutions ?? []
        architecture = arch
        expandedBlocks = arch.expandedBlocks
        inputPlanes = arch.inputPlanes
        policySize = ChessNetwork.policySize
        valueHeadClasses = arch.valueHeadClasses
        parameterCount = arch.parameterCount
        weightPlan = arch.weightTensorPlan().map { PlannedTensor(name: $0.name, shape: $0.shape, kind: $0.kind.rawValue) }
        includesOptimizerVelocity = file.includesOptimizerVelocity
    }

    func encodedJSON() throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        return try encoder.encode(self)
    }
}

/// Writes a model's inference graph (`InferenceGraph`) and weights in the form
/// `scripts/dcm_coreml_export.py` turns into a Core ML model.
///
/// The folder holds three files, all new (an existing folder is refused):
/// - `graph.json` — the recorded forward pass;
/// - `weights.bin` — every plan tensor as little-endian float32, in plan
///   order, in the layout the MPSGraph variable holds (offsets in `graph.json`);
/// - `model.json` — the `ModelDescription` of the source file.
///
/// The weights are the ones `loadWeights` would receive (`ModelCheckpointFile.
/// networkWeights`: un-transposed from the file's layout, the value head
/// recentered as every load does), so the exporter applies no load rule of
/// its own.
enum InferenceGraphExport {
    static let graphFileName = "graph.json"
    static let weightsFileName = "weights.bin"
    static let modelFileName = "model.json"

    enum ExportError: LocalizedError {
        case weightCountMismatch(have: Int, want: Int)
        case weightSizeMismatch(name: String, have: Int, want: Int)
        case notLittleEndian
        case recordingMissing
        case partialExportRemains(folder: String, write: String, cleanup: String)

        var errorDescription: String? {
            switch self {
            case .weightCountMismatch(let have, let want):
                return "the model file has \(have) network tensors; its architecture's plan has \(want)"
            case .weightSizeMismatch(let name, let have, let want):
                return "tensor \(name) holds \(have) values; the graph declares \(want)"
            case .notLittleEndian:
                return "weights.bin is little-endian float32; this machine is not little-endian"
            case .recordingMissing:
                return "the network was built with a recorder but carries no recording"
            case .partialExportRemains(let folder, let write, let cleanup):
                return "writing the export failed (\(write)) and its folder \(folder) could not be removed: \(cleanup)"
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

    /// The bytes of `weights.bin`: `weights` (plan order) as little-endian
    /// float32, each checked against the graph's declared element count.
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
            precondition(data.count == spec.byteOffset, "graph weight offsets follow plan order")
            values.withUnsafeBufferPointer { data.append(Data(buffer: $0)) }
        }
        return data
    }

    static func graphFileData(_ graph: InferenceGraph) throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        return try encoder.encode(graph)
    }

    /// Writes the three files into `folder`, which must not exist. A failed
    /// write removes the folder this call created (by identity, never by
    /// name), so a half-written export is never left looking complete.
    static func write(graph: InferenceGraph, weights: [[Float]], description: ModelDescription, to folder: URL) throws {
        let graphData = try graphFileData(graph)
        let weightsData = try weightsFileData(graph: graph, weights: weights)
        let modelData = try description.encodedJSON()
        let folderIdentity = try FileSafety.createNewDirectory(at: folder)
        do {
            _ = try FileSafety.writeNewFile(graphData, at: folder.appendingPathComponent(graphFileName))
            _ = try FileSafety.writeNewFile(weightsData, at: folder.appendingPathComponent(weightsFileName))
            _ = try FileSafety.writeNewFile(modelData, at: folder.appendingPathComponent(modelFileName))
        } catch {
            do {
                try FileSafety.removeOwnedItem(at: folder, identity: folderIdentity)
            } catch let cleanupError {
                throw ExportError.partialExportRemains(folder: folder.path, write: error.localizedDescription,
                                                       cleanup: cleanupError.localizedDescription)
            }
            throw error
        }
    }
}
