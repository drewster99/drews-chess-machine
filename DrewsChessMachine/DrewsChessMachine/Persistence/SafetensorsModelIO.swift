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

    /// `dcm_format_version` stamped on every write — the architecture format
    /// version (`ArchitectureFormat.currentVersion`), which gates how the
    /// embedded architecture JSON is decoded on load.
    static let formatVersion = String(ArchitectureFormat.currentVersion)

    enum IOError: Error, CustomStringConvertible {
        case tensorCountMismatch(weights: Int, names: Int)
        case missingTensor(String)
        case missingArchitecture
        case badArchitectureJSON(String)
        case tensorShapeMismatch(name: String, expected: Int, got: Int)
        /// The stored DIMENSIONS disagree with the plan even though the element
        /// count may match — e.g. a `.linear` written `[in, out]` instead of
        /// `[out, in]`, which `fromTorchLayout` would transpose with swapped
        /// dims and silently scramble.
        case tensorDimsMismatch(name: String, expected: [Int], got: [Int])
        /// Trainer schedule metadata (`trainer_*`) without optimizer velocity
        /// tensors, on write or read.
        case trainerScheduleWithoutVelocity
        /// `trainer_policy_tail_precision` holds a value no precision spells.
        case malformedTrainerPolicyTailPrecision(String)
        /// A file at a format version that requires `dcm_lineage` has none.
        case missingLineage(formatVersion: Int)
        /// `dcm_lineage` is present but does not decode.
        case malformedLineage(String)
        /// A header without `model_id`.
        case missingModelID(source: String)
        /// `training_step` is present but not an integer.
        case malformedTrainingStep(String, source: String)
        /// A trainer-state file's lineage total disagrees with its trainer
        /// clock (`trainer_completed_steps`), on write.
        case lineageStepDisagreesWithTrainerClock(lineageStep: Int?, trainerClock: Int)

        var description: String {
            switch self {
            case .tensorCountMismatch(let w, let n): return "safetensors model: \(w) weight arrays but \(n) names"
            case .missingTensor(let name): return "safetensors model: missing tensor '\(name)'"
            case .missingArchitecture: return "safetensors model: no architecture in __metadata__"
            case .badArchitectureJSON(let d): return "safetensors model: architecture JSON failed to decode (\(d))"
            case .tensorShapeMismatch(let name, let expected, let got):
                return "safetensors model: tensor '\(name)' has \(got) elements but the embedded architecture's plan expects \(expected)"
            case .tensorDimsMismatch(let name, let expected, let got):
                return "safetensors model: tensor '\(name)' is stored with shape \(got) but the embedded "
                    + "architecture's plan expects \(expected) — same element count is not enough, the "
                    + "dimensions decide how the tensor is un-transposed on load"
            case .trainerScheduleWithoutVelocity:
                return "safetensors model: trainer schedule metadata (trainer_*) without optimizer velocity "
                    + "tensors — exact-resume state must travel with the velocity it was captured alongside"
            case .malformedTrainerPolicyTailPrecision(let raw):
                let allowed = ChessNetwork.PolicyTailPrecision.allCases.map(\.rawValue).joined(separator: ", ")
                return "safetensors model: trainer_policy_tail_precision is '\(raw)', expected one of \(allowed)"
            case .missingLineage(let version):
                return "safetensors model: format version \(version) requires a \(LineageRecord.metadataKey) record "
                    + "in __metadata__ (required from version \(ArchitectureFormat.lineageRequiredFromVersion)), and this file has none"
            case .malformedLineage(let detail):
                return "safetensors model: \(LineageRecord.metadataKey) does not decode (\(detail))"
            case .missingModelID(let source):
                return "safetensors model \(source): no model_id in __metadata__"
            case .malformedTrainingStep(let raw, let source):
                return "safetensors model \(source): training_step '\(raw)' is not an integer"
            case .lineageStepDisagreesWithTrainerClock(let lineageStep, let trainerClock):
                return "safetensors model: lineage cum_trainer_step \(lineageStep.map(String.init) ?? "null") must equal the "
                    + "trainer-state file's trainer_completed_steps \(trainerClock)"
            }
        }
    }

    // Metadata keys
    enum Key {
        static let formatVersion = "dcm_format_version"
        static let modelID = "model_id"
        static let createdAt = "created_at_unix"
        static let creator = "creator"
        static let trainingStep = "training_step"
        static let parentModelID = "parent_model_id"
        static let notes = "notes"
        static let architecture = "architecture"
        static let trainerPolicyTailPrecision = "trainer_policy_tail_precision"
    }

    /// Ordered tensor names for `architecture`: the base plan, plus, for a
    /// trainer file, one `opt.<trainableName>.velocity` per trainable (in
    /// trainable order) appended after the base tensors.
    static func tensorNames(for architecture: NetworkArchitecture, includesVelocity: Bool) -> [String] {
        var names = architecture.weightTensorPlan().map(\.name)
        if includesVelocity {
            names.append(contentsOf: architecture.trainableTensorPlan().map { velocityTensorName(forTrainable: $0.name) })
        }
        return names
    }

    /// The persisted name of a trainable's optimizer velocity tensor.
    static func velocityTensorName(forTrainable trainableName: String) -> String {
        "opt.\(trainableName).velocity"
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
        lineage: LineageRecord
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
        // Exact-resume state travels only with the optimizer velocity it was
        // captured alongside; one without the other is not resumable.
        if let schedule = metadata.trainerSchedule {
            guard includesVelocity else { throw IOError.trainerScheduleWithoutVelocity }
            // The lineage total of a trainer-state file IS its trainer clock;
            // two values that could disagree would not be one source.
            guard lineage.steps.cumTrainerStep == schedule.completedTrainSteps else {
                throw IOError.lineageStepDisagreesWithTrainerClock(
                    lineageStep: lineage.steps.cumTrainerStep, trainerClock: schedule.completedTrainSteps)
            }
            for (key, value) in try schedule.metadataEntries() { md[key] = value }
        }
        if let precision = metadata.trainerPolicyTailPrecision {
            md[Key.trainerPolicyTailPrecision] = precision.rawValue
        }
        // Every save marks the value head centered, so a file is recentered
        // at most once in its life and every new file round-trips bit-exactly
        // (see `ValueHeadRecentering`).
        md[ValueHeadRecentering.metadataKey] = ValueHeadRecentering.metadataValue
        let archData = try JSONEncoder().encode(architecture)
        md[Key.architecture] = String(decoding: archData, as: UTF8.self)

        // The file's lineage: the record's JSON plus its derived flat mirrors.
        for (key, value) in try lineage.metadataEntries() { md[key] = value }

        return try SafetensorsFile.encode(tensors: tensors, metadata: md)
    }

    struct Decoded {
        let file: ModelCheckpointFile
        let architecture: NetworkArchitecture
        /// Velocity tensors present (trainer file) beyond the base plan.
        let hasVelocity: Bool
        /// The format the embedded architecture was decoded under, carrying
        /// any legacy resolutions made (see `ArchitectureFormat`). Loaders
        /// that know the file's name call `logLegacyResolutions()` once.
        let architectureFormat: ArchitectureFormat.DecodeFormat
    }

    /// Name used in errors and log lines when the caller decodes bytes it
    /// did not read from a named file.
    static let unnamedSource = "safetensors data"

    /// Decode safetensors bytes into a `ModelCheckpointFile`, ordering weights
    /// to match the embedded architecture's plan (+ trailing velocity tensors,
    /// in trainable order, if present).
    static func decode(_ data: Data) throws -> Decoded {
        try decode(data, valueHead: .recenterUnlessMarked, source: unnamedSource)
    }

    /// `decode(_:)`, choosing whether the value head is recentered; only
    /// analysis asks for `.asStored`.
    static func decode(_ data: Data, valueHead: ValueHeadDecoding) throws -> Decoded {
        try decode(data, valueHead: valueHead, source: unnamedSource)
    }

    /// Decode the architecture embedded in a safetensors `__metadata__` map
    /// under the file's own `dcm_format_version`. Shared by the full decode
    /// and the header-only readers (model catalog, `--derive-model`) so every
    /// reader applies the same version gate.
    static func decodeArchitecture(
        fromMetadata md: [String: String],
        source: String
    ) throws -> (architecture: NetworkArchitecture, format: ArchitectureFormat.DecodeFormat) {
        guard let archJSON = md[Key.architecture] else { throw IOError.missingArchitecture }
        let version = try ArchitectureFormat.safetensorsFormatVersion(
            metadataValue: md[Key.formatVersion], source: source)
        let format = ArchitectureFormat.DecodeFormat(formatVersion: version, source: source)
        do {
            let architecture = try ArchitectureFormat.makeDecoder(format: format)
                .decode(NetworkArchitecture.self, from: Data(archJSON.utf8))
            return (architecture, format)
        } catch let formatError as ArchitectureFormat.FormatError {
            // Already names the field and the file; keep it intact.
            throw formatError
        } catch {
            throw IOError.badArchitectureJSON(String(describing: error))
        }
    }

    /// `decode(_:valueHead:)` for bytes read from `source` (a file name used in
    /// format-version errors and the legacy-resolution log line).
    static func decode(_ data: Data, valueHead: ValueHeadDecoding, source: String) throws -> Decoded {
        let (tensors, md) = try SafetensorsFile.decode(data)
        // Keep each tensor's stored SHAPE, not just its data: the dimensions are
        // what decide whether `fromTorchLayout` un-transposes correctly, and
        // discarding them here is what left the load guard unable to see a
        // transposed tensor (see the dims check below).
        var byName: [String: SafetensorsTensor] = [:]
        byName.reserveCapacity(tensors.count)
        for t in tensors { byName[t.name] = t }

        let (architecture, architectureFormat) = try decodeArchitecture(fromMetadata: md, source: source)

        // Identity is the embedded architecture itself (no arch_hash); integrity
        // is content_sha256 (verified in SafetensorsFile). A hand-edited config
        // surfaces as a weight-shape mismatch against the plan below.
        let hasVelocity = tensors.contains { $0.name.hasPrefix("opt.") && $0.name.hasSuffix(".velocity") }
        let names = tensorNames(for: architecture, includesVelocity: hasVelocity)
        let plan = architecture.weightTensorPlan()

        // Pass 1 — element counts, across EVERY plan position before any
        // dimension is judged.
        //
        // The file's per-tensor element count (validated against its own header
        // shape in `SafetensorsFile.decode`) must also agree with the embedded
        // architecture's plan — they are two independent shape sources and
        // `fromTorchLayout` indexes by the plan's dims. A mismatch (hand-edited
        // config, buggy external writer) would otherwise run `transpose2D` off
        // the end of the data; surface it as a clean error so the embedded arch
        // stays the single source of truth for what shapes we accept.
        //
        // Deliberately a separate pass rather than folded into the loop below: a
        // truncated or overlong tensor is a grosser failure than a mis-declared
        // shape, so it should be reported wherever it sits. Interleaved, a
        // dimension complaint about position 0 would mask a truncated tensor at
        // position 90 and send the reader chasing the wrong thing.
        for (i, name) in names.enumerated() where i < plan.count {
            guard let tensor = byName[name] else { throw IOError.missingTensor(name) }
            guard tensor.data.count == plan[i].elementCount else {
                throw IOError.tensorShapeMismatch(
                    name: name, expected: plan[i].elementCount, got: tensor.data.count
                )
            }
        }

        var weights: [[Float]] = []
        weights.reserveCapacity(names.count)
        for (i, name) in names.enumerated() {
            guard let tensor = byName[name] else { throw IOError.missingTensor(name) }
            let torchData = tensor.data
            if i < plan.count {
                // Count alone is not sufficient. `fromTorchLayout` un-transposes a
                // `.linear` using the PLAN's dims, so a tensor stored `[in, out]`
                // instead of `[out, in]` has an identical element count, sails past
                // the check above, and comes back silently scrambled. Compare the
                // stored dimensions against what this position is supposed to look
                // like in torch layout. Squeezed, so a writer that spells a bias
                // `[1, C, 1, 1]` rather than `[C]` is still accepted — the axes that
                // carry no values are not worth rejecting a file over.
                let expectedDims = WeightTensorSpec.squeeze(Self.torchShape(for: plan[i]))
                let actualDims = WeightTensorSpec.squeeze(tensor.shape)
                guard actualDims == expectedDims else {
                    throw IOError.tensorDimsMismatch(
                        name: name, expected: Self.torchShape(for: plan[i]), got: tensor.shape
                    )
                }
                // Reverse the PyTorch layout back to the engine's native flat order.
                weights.append(Self.fromTorchLayout(kind: plan[i].kind, nativeShape: plan[i].shape, torchData: torchData))
            } else {
                weights.append(torchData) // velocity: stored native flat
            }
        }

        // Remove the value head's shared offset unless the file is marked as
        // already centered (see `ValueHeadRecentering`). Runs here, on the
        // decoded native-layout values, so every loader — model files,
        // sessions' champion and trainer files, CLIs — sees the same weights.
        let valueHeadCentering: ValueHeadCentering
        switch valueHead {
        case .recenterUnlessMarked:
            valueHeadCentering = try ValueHeadRecentering.apply(
                to: &weights,
                architecture: architecture,
                includesVelocity: hasVelocity,
                markedCentered: try ValueHeadRecentering.isMarkedCentered(md[ValueHeadRecentering.metadataKey])
            )
        case .asStored:
            valueHeadCentering = .keptAsStored
        }

        let trainerSchedule = try TrainerScheduleState.decode(fromMetadata: md)
        if trainerSchedule != nil && !hasVelocity {
            throw IOError.trainerScheduleWithoutVelocity
        }
        let trainerPolicyTailPrecision: ChessNetwork.PolicyTailPrecision?
        if let raw = md[Key.trainerPolicyTailPrecision] {
            guard let precision = ChessNetwork.PolicyTailPrecision(rawValue: raw) else {
                throw IOError.malformedTrainerPolicyTailPrecision(raw)
            }
            trainerPolicyTailPrecision = precision
        } else {
            trainerPolicyTailPrecision = nil
        }
        let fileLineage = try lineage(fromMetadata: md, formatVersion: architectureFormat.formatVersion)
        let provenance = ModelCheckpointFile.SafetensorsProvenance(
            contentSHA256: md[SafetensorsFile.contentHashKey],
            lineage: fileLineage,
            derivationHistory: try LineageTracker.ParentFile.derivationHistory(lineage: fileLineage, metadata: md)
        )
        let metadata = ModelCheckpointMetadata(
            creator: md[Key.creator] ?? "",
            trainingStep: md[Key.trainingStep].flatMap { Int($0) },
            parentModelID: md[Key.parentModelID] ?? "",
            notes: md[Key.notes] ?? "",
            trainerSchedule: trainerSchedule,
            trainerPolicyTailPrecision: trainerPolicyTailPrecision
        )
        let file = ModelCheckpointFile(
            modelID: md[Key.modelID] ?? "",
            createdAtUnix: md[Key.createdAt].flatMap { Int64($0) } ?? 0,
            metadata: metadata,
            weights: weights,
            architecture: architecture,
            valueHeadCentering: valueHeadCentering,
            architectureFormat: architectureFormat,
            safetensorsProvenance: provenance
        )
        return Decoded(file: file, architecture: architecture, hasVelocity: hasVelocity,
                       architectureFormat: architectureFormat)
    }

    // MARK: - Lineage

    /// The lineage a safetensors `__metadata__` map carries, under the file's
    /// own format version: required (and decoded strictly) from
    /// `ArchitectureFormat.lineageRequiredFromVersion`, reported unrecorded
    /// for older files. Shared by the full decode and the header-only reader.
    static func lineage(fromMetadata md: [String: String], formatVersion: Int) throws -> LineageRecord.Presence {
        guard formatVersion >= ArchitectureFormat.lineageRequiredFromVersion else {
            return .unrecorded(formatVersion: formatVersion)
        }
        guard let text = md[LineageRecord.metadataKey] else {
            throw IOError.missingLineage(formatVersion: formatVersion)
        }
        do {
            return .recorded(try LineageRecord.decode(jsonText: text))
        } catch {
            throw IOError.malformedLineage(String(describing: error))
        }
    }

    /// Header-only read of a safetensors model file's identity and lineage:
    /// model ID, content hash, trainer clock (`trainer_completed_steps`, or
    /// the plain file's `training_step`) and lineage, with no tensor decode.
    static func readParentFile(at url: URL) throws -> LineageTracker.ParentFile {
        try readParentFile(fromMetadata: try ModelFileCatalog.headerMetadata(at: url), source: url.lastPathComponent)
    }

    /// `readParentFile(at:)` over a header's `__metadata__` already read, so
    /// a caller that needs other keys of the same header reads it once.
    /// `source` names the file in errors.
    static func readParentFile(fromMetadata md: [String: String], source: String) throws -> LineageTracker.ParentFile {
        let version = try ArchitectureFormat.safetensorsFormatVersion(
            metadataValue: md[Key.formatVersion], source: source)
        guard let modelID = md[Key.modelID] else {
            throw IOError.missingModelID(source: source)
        }
        let fileLineage = try lineage(fromMetadata: md, formatVersion: version)
        return LineageTracker.ParentFile(
            modelID: modelID,
            contentSHA256: md[SafetensorsFile.contentHashKey],
            trainerCompletedSteps: try trainerClock(fromMetadata: md, source: source),
            lineage: fileLineage,
            derivationHistory: try LineageTracker.ParentFile.derivationHistory(lineage: fileLineage, metadata: md)
        )
    }

    /// A file's trainer clock: `trainer_completed_steps` on a trainer-state
    /// file, otherwise the `training_step` its weights were taken at, nil
    /// when it states neither. A value that is present but not an integer
    /// is an error, never read as absent.
    static func trainerClock(fromMetadata md: [String: String], source: String) throws -> Int? {
        if let schedule = try TrainerScheduleState.decode(fromMetadata: md) {
            return schedule.completedTrainSteps
        }
        guard let raw = md[Key.trainingStep] else { return nil }
        guard let value = Int(raw) else {
            throw IOError.malformedTrainingStep(raw, source: source)
        }
        return value
    }

    // MARK: - Resume provenance

    /// Where a corpus-replay checkpoint stood in its corpus. Read from the
    /// file's lineage record, or — for a file written before lineage — from
    /// the `replay_*` / `built_by_*` keys corpus replay wrote then.
    struct ReplayResumeMetadata: Sendable {
        var corpusID: String
        /// Where the corpus was when the checkpoint was written — a hint;
        /// the corpus is matched by `corpusID`. Nil when a file written
        /// before lineage does not record it.
        var corpusPath: String?
        var nextGameIndex: Int
        var epoch: Int
        /// Positions in the replay buffer at the save, and its capacity.
        /// Nil when a file written before lineage does not record them.
        var populatedPlies: Int?
        var capacity: Int?
        var builtByBuild: Int?
        var builtByGit: String?
    }

    /// Why a file cannot be exact-resumed by corpus replay.
    enum ReplayResumeError: Error, CustomStringConvertible {
        case noCorpusPosition(file: String)
        case noLegacyResumeMetadata(file: String)
        /// A file written before lineage names its corpus but not this
        /// part of its position.
        case legacyKeyMissing(file: String, key: String)
        /// A file written before lineage holds `value` for `key`, which is
        /// not an integer.
        case legacyKeyMalformed(file: String, key: String, value: String)

        var description: String {
            switch self {
            case .noCorpusPosition(let file):
                return "\(file) carries a lineage without a corpus position (not a corpus-replay checkpoint)"
            case .noLegacyResumeMetadata(let file):
                return "\(file) carries no replay_* resume metadata (not a corpus-replay checkpoint)"
            case let .legacyKeyMissing(file, key):
                return "\(file) names its corpus (replay_corpus_id) but has no \(key), so where it stood in the "
                    + "corpus is unknown"
            case let .legacyKeyMalformed(file, key, value):
                return "\(file) has \(key) \"\(value)\", which is not an integer"
            }
        }
    }

    /// Where a corpus-replay checkpoint stood in its corpus: from its
    /// lineage record (`fed.corpus`) when it has one, else — for a file
    /// written before lineage — from the legacy `replay_*` keys. The header
    /// is read once, for both.
    static func replayResumePoint(at url: URL) throws -> ReplayResumeMetadata {
        let md = try ModelFileCatalog.headerMetadata(at: url)
        let parent = try readParentFile(fromMetadata: md, source: url.lastPathComponent)
        switch parent.lineage {
        case .recorded(let record):
            guard let corpus = record.fed.corpus else {
                throw ReplayResumeError.noCorpusPosition(file: url.lastPathComponent)
            }
            return ReplayResumeMetadata(
                corpusID: corpus.corpusID,
                corpusPath: corpus.corpusPath,
                nextGameIndex: corpus.nextGameIndex,
                epoch: corpus.epoch,
                populatedPlies: corpus.populatedPlies,
                capacity: corpus.bufferCapacity,
                builtByBuild: record.build.buildNumber,
                builtByGit: record.build.gitHash
            )
        case .unrecorded:
            guard let legacy = try legacyResumeMetadata(fromMetadata: md, file: url.lastPathComponent) else {
                throw ReplayResumeError.noLegacyResumeMetadata(file: url.lastPathComponent)
            }
            return legacy
        }
    }

    /// The resume point a file written before lineage records in its
    /// `replay_*` / `built_by_*` header keys, or nil when it names no corpus
    /// (`replay_corpus_id`; not a corpus-replay checkpoint). The position
    /// (`replay_next_game_index`, `replay_epoch`) is required: a missing or
    /// non-integer one throws, never reads as game 0 of epoch 0, which would
    /// silently retrain the corpus from its start. The buffer fill and
    /// capacity and the build number are nil when absent and throw when
    /// present but not an integer.
    static func legacyResumeMetadata(fromMetadata md: [String: String], file: String) throws -> ReplayResumeMetadata? {
        guard let corpusID = md["replay_corpus_id"] else { return nil }
        func optionalInt(_ key: String) throws -> Int? {
            guard let text = md[key] else { return nil }
            guard let value = Int(text) else {
                throw ReplayResumeError.legacyKeyMalformed(file: file, key: key, value: text)
            }
            return value
        }
        func requiredInt(_ key: String) throws -> Int {
            guard let value = try optionalInt(key) else {
                throw ReplayResumeError.legacyKeyMissing(file: file, key: key)
            }
            return value
        }
        return ReplayResumeMetadata(
            corpusID: corpusID,
            corpusPath: md["replay_corpus_path"],
            nextGameIndex: try requiredInt("replay_next_game_index"),
            epoch: try requiredInt("replay_epoch"),
            populatedPlies: try optionalInt("replay_populated_plies"),
            capacity: try optionalInt("replay_capacity"),
            builtByBuild: try optionalInt("built_by_build"),
            builtByGit: md["built_by_git"]
        )
    }

    // MARK: - PyTorch layout transforms

    /// Native engine layout -> PyTorch state_dict layout (for the on-disk file).
    /// Only Linear weights need a data transpose; biases reshape to 1-D; conv
    /// (OIHW) and BN params ([C]) already match torch.
    /// The shape a plan entry takes ON DISK, in PyTorch state_dict layout.
    ///
    /// Single definition shared by the writer (`toTorchLayout`) and the
    /// load-time dimension guard in `decode`, so the two cannot drift apart —
    /// a guard computing "expected" differently from how the file is actually
    /// written would either reject valid files or wave through the very
    /// transposition it exists to catch.
    static func torchShape(kind: WeightKind, nativeShape: [Int]) -> [Int] {
        switch kind {
        case .linear:
            return [nativeShape[1], nativeShape[0]]   // native [in, out] -> torch [out, in]
        case .bias:
            return [nativeShape.reduce(1, *)]         // [1,N,1,1] / [1,N] -> [N]
        case .conv, .bnAffine, .bnRunningStat, .scalar:
            return nativeShape
        }
    }

    static func torchShape(for spec: WeightTensorSpec) -> [Int] {
        torchShape(kind: spec.kind, nativeShape: spec.shape)
    }

    static func toTorchLayout(kind: WeightKind, nativeShape: [Int], data: [Float]) -> (shape: [Int], data: [Float]) {
        let shape = torchShape(kind: kind, nativeShape: nativeShape)
        switch kind {
        case .linear:
            // native [in, out] -> torch [out, in]
            return (shape, transpose2D(data, rows: nativeShape[0], cols: nativeShape[1]))
        case .bias, .conv, .bnAffine, .bnRunningStat, .scalar:
            return (shape, data)
        }
    }

    /// PyTorch layout -> native engine flat order (for loading into the graph,
    /// and for weight initialization, which draws in the on-disk order).
    static func fromTorchLayout(kind: WeightKind, nativeShape: [Int], torchData: [Float]) -> [Float] {
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
