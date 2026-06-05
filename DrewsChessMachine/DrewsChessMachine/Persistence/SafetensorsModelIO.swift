//
//  SafetensorsModelIO.swift
//  DrewsChessMachine
//
//  Bridges the in-memory `ModelCheckpointFile` (flat `[[Float]]` weights +
//  metadata) to/from the native safetensors container. Tensor names come from
//  `NetworkArchitecture.weightTensorPlan()`; trainer optimizer state (velocity)
//  is appended as `opt.<trainableName>.velocity`. Model metadata rides in the
//  safetensors `__metadata__` string map, including the full architecture as a
//  JSON string so a loader can rebuild the matching graph.
//

import Foundation

enum SafetensorsModelIO {

    static let formatVersion = "3"

    enum IOError: Error, CustomStringConvertible {
        case tensorCountMismatch(weights: Int, names: Int)
        case missingTensor(String)
        case missingArchitecture
        case badArchitectureJSON(String)

        var description: String {
            switch self {
            case .tensorCountMismatch(let w, let n): return "safetensors model: \(w) weight arrays but \(n) names"
            case .missingTensor(let name): return "safetensors model: missing tensor '\(name)'"
            case .missingArchitecture: return "safetensors model: no architecture in __metadata__"
            case .badArchitectureJSON(let d): return "safetensors model: architecture JSON failed to decode (\(d))"
            }
        }
    }

    // Metadata keys
    private enum Key {
        static let formatVersion = "dcm_format_version"
        static let modelID = "model_id"
        static let createdAt = "created_at_unix"
        static let creator = "creator"
        static let trainingStep = "training_step"
        static let parentModelID = "parent_model_id"
        static let notes = "notes"
        static let archHash = "arch_hash"
        static let architecture = "architecture"
    }

    /// Ordered tensor names for `architecture`: the base plan, plus, for a
    /// trainer file, one `opt.<trainableName>.velocity` per trainable (in
    /// trainable order) appended after the base tensors.
    static func tensorNames(for architecture: NetworkArchitecture, includesVelocity: Bool) -> [String] {
        let plan = architecture.weightTensorPlan()
        var names = plan.map(\.name)
        if includesVelocity {
            let trainables = plan.filter { $0.kind != .bnRunningStat }
            names.append(contentsOf: trainables.map { "opt.\($0.name).velocity" })
        }
        return names
    }

    /// Encode a model file to safetensors bytes. `weights` order must match
    /// `tensorNames(for:includesVelocity:)`.
    static func encode(
        modelID: String,
        createdAtUnix: Int64,
        metadata: ModelCheckpointMetadata,
        weights: [[Float]],
        architecture: NetworkArchitecture,
        includesVelocity: Bool
    ) throws -> Data {
        let names = tensorNames(for: architecture, includesVelocity: includesVelocity)
        guard weights.count == names.count else {
            throw IOError.tensorCountMismatch(weights: weights.count, names: names.count)
        }
        let plan = architecture.weightTensorPlan()
        var tensors: [SafetensorsTensor] = []
        tensors.reserveCapacity(weights.count)
        for (i, w) in weights.enumerated() {
            // Base tensors carry their plan shape; appended velocity tensors are
            // 1-D (their float count) since they mirror flattened trainables.
            let shape = i < plan.count ? plan[i].shape : [w.count]
            tensors.append(SafetensorsTensor(name: names[i], shape: shape, data: w))
        }

        var md: [String: String] = [
            Key.formatVersion: formatVersion,
            Key.modelID: modelID,
            Key.createdAt: String(createdAtUnix),
            Key.creator: metadata.creator,
            Key.parentModelID: metadata.parentModelID,
            Key.notes: metadata.notes,
            Key.archHash: architecture.archHashHex,
        ]
        if let step = metadata.trainingStep { md[Key.trainingStep] = String(step) }
        let archData = try JSONEncoder().encode(architecture)
        md[Key.architecture] = String(decoding: archData, as: UTF8.self)

        return try SafetensorsFile.encode(tensors: tensors, metadata: md)
    }

    struct Decoded {
        let file: ModelCheckpointFile
        let architecture: NetworkArchitecture
        /// Velocity tensors present (trainer file) beyond the base plan.
        let hasVelocity: Bool
    }

    /// Decode safetensors bytes into a `ModelCheckpointFile`, ordering weights
    /// to match the embedded architecture's plan (+ trailing velocity tensors,
    /// in trainable order, if present).
    static func decode(_ data: Data) throws -> Decoded {
        let (tensors, md) = try SafetensorsFile.decode(data)
        var byName: [String: [Float]] = [:]
        byName.reserveCapacity(tensors.count)
        for t in tensors { byName[t.name] = t.data }

        guard let archJSON = md[Key.architecture] else { throw IOError.missingArchitecture }
        let architecture: NetworkArchitecture
        do {
            architecture = try JSONDecoder().decode(NetworkArchitecture.self, from: Data(archJSON.utf8))
        } catch {
            throw IOError.badArchitectureJSON(error.localizedDescription)
        }

        let hasVelocity = tensors.contains { $0.name.hasPrefix("opt.") && $0.name.hasSuffix(".velocity") }
        let names = tensorNames(for: architecture, includesVelocity: hasVelocity)

        var weights: [[Float]] = []
        weights.reserveCapacity(names.count)
        for name in names {
            guard let w = byName[name] else { throw IOError.missingTensor(name) }
            weights.append(w)
        }

        let metadata = ModelCheckpointMetadata(
            creator: md[Key.creator] ?? "",
            trainingStep: md[Key.trainingStep].flatMap { Int($0) },
            parentModelID: md[Key.parentModelID] ?? "",
            notes: md[Key.notes] ?? ""
        )
        let file = ModelCheckpointFile(
            modelID: md[Key.modelID] ?? "",
            createdAtUnix: md[Key.createdAt].flatMap { Int64($0) } ?? 0,
            metadata: metadata,
            weights: weights
        )
        return Decoded(file: file, architecture: architecture, hasVelocity: hasVelocity)
    }
}
