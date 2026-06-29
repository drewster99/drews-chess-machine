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
        case tensorShapeMismatch(name: String, expected: Int, got: Int)

        var description: String {
            switch self {
            case .tensorCountMismatch(let w, let n): return "safetensors model: \(w) weight arrays but \(n) names"
            case .missingTensor(let name): return "safetensors model: missing tensor '\(name)'"
            case .missingArchitecture: return "safetensors model: no architecture in __metadata__"
            case .badArchitectureJSON(let d): return "safetensors model: architecture JSON failed to decode (\(d))"
            case .tensorShapeMismatch(let name, let expected, let got):
                return "safetensors model: tensor '\(name)' has \(got) elements but the embedded architecture's plan expects \(expected)"
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
        includesVelocity: Bool,
        resumeMetadata: [String: String]? = nil
    ) throws -> Data {
        let names = tensorNames(for: architecture, includesVelocity: includesVelocity)
        guard weights.count == names.count else {
            throw IOError.tensorCountMismatch(weights: weights.count, names: names.count)
        }
        let plan = architecture.weightTensorPlan()
        var tensors: [SafetensorsTensor] = []
        tensors.reserveCapacity(weights.count)
        for (i, w) in weights.enumerated() {
            if i < plan.count {
                // Base model tensors: store in PyTorch state_dict layout so the
                // file is load_state_dict-ready (FC weights transposed to
                // [out,in], biases 1-D; conv OIHW + BN [C] already match).
                let (shape, data) = Self.toTorchLayout(kind: plan[i].kind, nativeShape: plan[i].shape, data: w)
                tensors.append(SafetensorsTensor(name: names[i], shape: shape, data: data))
            } else {
                // Optimizer velocity (trainer file): DCM-internal optimizer state,
                // not part of a torch state_dict — stored 1-D in native order.
                tensors.append(SafetensorsTensor(name: names[i], shape: [w.count], data: w))
            }
        }

        var md: [String: String] = [
            Key.formatVersion: formatVersion,
            Key.modelID: modelID,
            Key.createdAt: String(createdAtUnix),
            Key.creator: metadata.creator,
            Key.parentModelID: metadata.parentModelID,
            Key.notes: metadata.notes,
        ]
        if let step = metadata.trainingStep { md[Key.trainingStep] = String(step) }
        let archData = try JSONEncoder().encode(architecture)
        md[Key.architecture] = String(decoding: archData, as: UTF8.self)

        // Optional resume provenance (e.g. corpus-replay `replay_*` / `built_by_*`
        // keys). Caller-namespaced strings written verbatim into `__metadata__`;
        // current decode ignores unknown keys, so this is purely additive. Never
        // overwrite a reserved key (the caller prefixes avoid collision anyway).
        if let rm = resumeMetadata {
            let reserved: Set<String> = [
                Key.formatVersion, Key.modelID, Key.createdAt, Key.creator,
                Key.trainingStep, Key.parentModelID, Key.notes, Key.architecture,
            ]
            for (k, v) in rm where !reserved.contains(k) { md[k] = v }
        }

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

        // Identity is the embedded architecture itself (no arch_hash); integrity
        // is content_sha256 (verified in SafetensorsFile). A hand-edited config
        // surfaces as a weight-shape mismatch against the plan below.
        let hasVelocity = tensors.contains { $0.name.hasPrefix("opt.") && $0.name.hasSuffix(".velocity") }
        let names = tensorNames(for: architecture, includesVelocity: hasVelocity)
        let plan = architecture.weightTensorPlan()

        var weights: [[Float]] = []
        weights.reserveCapacity(names.count)
        for (i, name) in names.enumerated() {
            guard let torchData = byName[name] else { throw IOError.missingTensor(name) }
            if i < plan.count {
                // The file's per-tensor element count (validated against its own
                // header shape in `SafetensorsFile.decode`) must also agree with
                // the embedded architecture's plan — they are two independent shape
                // sources and `fromTorchLayout` indexes by the plan's dims. A
                // mismatch (hand-edited config, buggy external writer) would
                // otherwise run `transpose2D` off the end of `torchData`; surface
                // it as a clean error so the embedded arch stays the single source
                // of truth for what shapes we accept.
                guard torchData.count == plan[i].elementCount else {
                    throw IOError.tensorShapeMismatch(name: name, expected: plan[i].elementCount, got: torchData.count)
                }
                // Reverse the PyTorch layout back to the engine's native flat order.
                weights.append(Self.fromTorchLayout(kind: plan[i].kind, nativeShape: plan[i].shape, torchData: torchData))
            } else {
                weights.append(torchData) // velocity: stored native flat
            }
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
            weights: weights,
            architecture: architecture
        )
        return Decoded(file: file, architecture: architecture, hasVelocity: hasVelocity)
    }

    // MARK: - Resume provenance

    /// Resume provenance read back from a checkpoint's `__metadata__` — the
    /// `replay_*` / `built_by_*` keys a corpus-replay run writes (Phase 1).
    /// Present only on checkpoints saved by `CorpusReplayRunner`.
    struct ReplayResumeMetadata: Sendable {
        var corpusID: String
        var corpusPath: String
        var nextGameIndex: Int
        var epoch: Int
        var populatedPlies: Int
        var capacity: Int
        var builtByBuild: Int?
        var builtByGit: String?
    }

    /// Header-only read of the resume metadata: parses just the safetensors JSON
    /// header (8-byte little-endian length prefix + that many JSON bytes), with
    /// no tensor decode — cheap even on a multi-MB checkpoint. Returns nil if the
    /// file isn't a resumable replay checkpoint (no `replay_corpus_id` key) or
    /// the header can't be read.
    static func readResumeMetadata(at url: URL) -> ReplayResumeMetadata? {
        guard let handle = try? FileHandle(forReadingFrom: url) else { return nil }
        defer { try? handle.close() }
        guard let lenData = try? handle.read(upToCount: 8), lenData.count == 8 else { return nil }
        var n: UInt64 = 0
        for (k, b) in lenData.enumerated() { n |= UInt64(b) << (8 * k) }
        guard n > 0, n < 64_000_000,
              let jsonData = try? handle.read(upToCount: Int(n)), jsonData.count == Int(n),
              let obj = try? JSONSerialization.jsonObject(with: jsonData) as? [String: Any],
              let md = obj["__metadata__"] as? [String: String],
              let corpusID = md["replay_corpus_id"] else { return nil }
        func intVal(_ key: String) -> Int? { md[key].flatMap { Int($0) } }
        return ReplayResumeMetadata(
            corpusID: corpusID,
            corpusPath: md["replay_corpus_path"] ?? "",
            nextGameIndex: intVal("replay_next_game_index") ?? 0,
            epoch: intVal("replay_epoch") ?? 0,
            populatedPlies: intVal("replay_populated_plies") ?? 0,
            capacity: intVal("replay_capacity") ?? 0,
            builtByBuild: intVal("built_by_build"),
            builtByGit: md["built_by_git"]
        )
    }

    // MARK: - PyTorch layout transforms

    /// Native engine layout -> PyTorch state_dict layout (for the on-disk file).
    /// Only Linear weights need a data transpose; biases reshape to 1-D; conv
    /// (OIHW) and BN params ([C]) already match torch.
    private static func toTorchLayout(kind: WeightKind, nativeShape: [Int], data: [Float]) -> (shape: [Int], data: [Float]) {
        switch kind {
        case .linear:
            // native [in, out] -> torch [out, in]
            let inDim = nativeShape[0]
            let outDim = nativeShape[1]
            return ([outDim, inDim], transpose2D(data, rows: inDim, cols: outDim))
        case .bias:
            return ([data.count], data)            // [1,N,1,1] / [1,N] -> [N]
        case .conv, .bnAffine, .bnRunningStat, .scalar:
            return (nativeShape, data)
        }
    }

    /// PyTorch layout -> native engine flat order (for loading into the graph).
    private static func fromTorchLayout(kind: WeightKind, nativeShape: [Int], torchData: [Float]) -> [Float] {
        switch kind {
        case .linear:
            // torch [out, in] -> native [in, out]
            let inDim = nativeShape[0]
            let outDim = nativeShape[1]
            return transpose2D(torchData, rows: outDim, cols: inDim)
        case .conv, .bias, .bnAffine, .bnRunningStat, .scalar:
            return torchData                       // count preserved; flat load is shape-agnostic
        }
    }

    /// Transpose a row-major `rows × cols` matrix (flat) to its `cols × rows`
    /// transpose (flat): out[c*rows + r] = flat[r*cols + c].
    private static func transpose2D(_ flat: [Float], rows: Int, cols: Int) -> [Float] {
        var out = [Float](repeating: 0, count: rows * cols)
        for r in 0..<rows {
            let base = r * cols
            for c in 0..<cols {
                out[c * rows + r] = flat[base + c]
            }
        }
        return out
    }
}
