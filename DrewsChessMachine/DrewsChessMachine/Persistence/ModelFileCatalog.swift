import Foundation

/// One `.safetensors` model file, identified by the metadata inside it —
/// never by its filename, which can repeat across segments and be
/// overwritten in place (see CLAUDE.md, "Identify checkpoints by safetensors
/// `__metadata__`").
struct ModelFileEntry: Sendable, Identifiable, Equatable {
    var id: URL { url }
    let url: URL
    let modelID: String
    /// Nil for files that record no step (a fresh build, a champion export).
    let trainingStep: Int?
    let createdAt: Date?
    let architectureLabel: String
    let fileModifiedAt: Date
    /// The `model_id` this model was forked from; nil when the file names
    /// none (a fresh build, or a session champion, which records "").
    var parentModelID: String? = nil
    /// Who wrote the file: "manual", "replay", "train-vs-uci", "sigusr2", …
    var creator: String? = nil
    /// The header's `content_sha256`: the hash of the file's data region,
    /// which identifies its exact weights (a rolling `--out-model` file and
    /// the step file of the same save share it). Nil when the header
    /// records none.
    var contentSHA256: String? = nil
    /// What the header's `dcm_lineage` record says. The catalog always sets
    /// it; nil only for an entry built elsewhere (a test).
    var lineage: ModelFileLineageFacts? = nil
}

/// What a model file's header says about its lineage (follow-lineage plan
/// §3.2), from its `dcm_lineage` record — the single source of truth; the
/// flat mirror keys (`lineage_run_id`, `cum_trainer_step`, …) are never
/// read.
enum ModelFileLineageFacts: Sendable, Equatable {
    case recorded(ModelFileLineagePosition)
    /// Written at a format version from before lineage records.
    case unrecorded(formatVersion: Int)
    /// The record (or the format version, trainer clock or derivation
    /// history read with it) does not decode. The file still lists: the
    /// file picker shows what it always showed.
    case unreadable(reason: String)
}

/// Where a file sits in its run, from its `dcm_lineage` record.
struct ModelFileLineagePosition: Sendable, Equatable {
    let lineageRunID: String
    let segmentID: String
    let segmentIndex: Int
    let segmentStartedUnix: Int64
    /// Earlier segments' IDs, oldest first, then this file's own.
    let segmentChain: [String]
    /// One per earlier segment, aligned with `segmentChain`: the segment
    /// that resumed it, the step it handed on at and when the resuming
    /// segment started. Read by the follow-lineage fork check
    /// (`ModelLineageTip`), which needs them to see a parent that kept
    /// training after a child resumed it.
    let handoffs: [LineageHandoff]
    /// The segment's own step count at the save (the CLI files'
    /// `training_step`).
    let segmentLocalStep: Int
    /// Nil when the run continues history no record counted.
    let cumTrainerStep: Int?
    let recordedUnix: Int64
    let pathKind: LineageRecord.PathKind
}

extension ModelFileLineagePosition {
    /// The position `record` describes. Each earlier segment's summary holds
    /// the step of the file that segment was resumed from; the segment that
    /// resumed it is the next summary, or the record's own segment for the
    /// last one.
    init(record: LineageRecord) {
        let earlier = record.segments
        var handoffs: [LineageHandoff] = []
        for (index, segment) in earlier.enumerated() {
            let resumedByID: String
            let resumedByStartedUnix: Int64
            if index + 1 < earlier.count {
                resumedByID = earlier[index + 1].segmentID
                resumedByStartedUnix = earlier[index + 1].startedUnix
            } else {
                resumedByID = record.run.segmentID
                resumedByStartedUnix = record.run.segmentStartedUnix
            }
            handoffs.append(LineageHandoff(
                fromSegmentID: segment.segmentID,
                atLocalStep: segment.segmentLocalStep,
                toSegmentID: resumedByID,
                toStartedUnix: resumedByStartedUnix))
        }
        self.init(
            lineageRunID: record.run.lineageRunID,
            segmentID: record.run.segmentID,
            segmentIndex: record.run.segmentIndex,
            segmentStartedUnix: record.run.segmentStartedUnix,
            segmentChain: earlier.map(\.segmentID) + [record.run.segmentID],
            handoffs: handoffs,
            segmentLocalStep: record.steps.segmentLocalStep,
            cumTrainerStep: record.steps.cumTrainerStep,
            recordedUnix: record.run.recordedUnix,
            pathKind: record.invocation.pathKind)
    }
}

/// Segment `fromSegmentID` was resumed by `toSegmentID`, which started at
/// `toStartedUnix` from `fromSegmentID`'s file at `atLocalStep`.
struct LineageHandoff: Sendable, Equatable, Hashable {
    let fromSegmentID: String
    let atLocalStep: Int
    let toSegmentID: String
    let toStartedUnix: Int64
}

/// Every file of one model line (one `model_id`), newest step first.
struct ModelLine: Sendable, Identifiable, Equatable {
    var id: String { modelID }
    let modelID: String
    let files: [ModelFileEntry]

    /// The line's most advanced file: the highest training step, then the
    /// newest.
    var latest: ModelFileEntry {
        files[0]
    }

    /// The newest modification time among the line's files. Not necessarily
    /// `latest`'s: `latest` is chosen by training step, and an earlier step
    /// can be written after a later one.
    var newestFileModifiedAt: Date {
        files.reduce(latest.fileModifiedAt) { max($0, $1.fileModifiedAt) }
    }
}

/// A self-play session's champion file.
struct SessionChampion: Sendable, Equatable {
    let sessionName: String
    let entry: ModelFileEntry
}

/// A `.safetensors` file the catalog could not read, and why.
struct UnreadableModelFile: Sendable, Identifiable, Equatable {
    var id: URL { url }
    let url: URL
    let reason: String
}

enum ModelFileCatalogError: LocalizedError, Equatable {
    case notSafetensors(file: String, detail: String)
    case missingModelID(file: String)

    var errorDescription: String? {
        switch self {
        case .notSafetensors(let file, let detail):
            return "\(file) is not a readable safetensors model: \(detail)"
        case .missingModelID(let file):
            return "\(file) records no model_id"
        }
    }
}

/// Lists the model files in a folder grouped into lines by `model_id`, for
/// finding the latest model of each line quickly. Reads only each file's
/// safetensors header (an 8-byte length and a JSON block), never the
/// weights, so scanning thousands of files is cheap.
enum ModelFileCatalog {

    struct Scan: Sendable {
        /// Newest activity first.
        let lines: [ModelLine]
        /// Files that could not be read, with the reason, by filename.
        let unreadable: [UnreadableModelFile]
    }

    /// Scan `directory` (not recursive) off the caller's thread.
    static func scan(directory: URL) async throws -> Scan {
        try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                continuation.resume(with: Result { try scanSynchronously(directory: directory) })
            }
        }
    }

    /// One full read of the folder: `ModelFolderHeaderCache` with nothing
    /// cached, so the catalog and the Lichess bot's lineage follower share
    /// one listing filter and one header reader.
    static func scanSynchronously(directory: URL) throws -> Scan {
        let scan = try ModelFolderHeaderCache.scanSynchronously(directory: directory, previous: .empty)
        var byModel: [String: [ModelFileEntry]] = [:]
        for entry in scan.entries {
            byModel[entry.modelID, default: []].append(entry)
        }
        let unreadable = scan.unreadable
        let lines = byModel.map { modelID, files in
            ModelLine(modelID: modelID, files: files.sorted(by: isMoreAdvanced))
        }
        .sorted { $0.newestFileModifiedAt > $1.newestFileModifiedAt }
        return Scan(lines: lines, unreadable: unreadable.sorted { $0.url.lastPathComponent < $1.url.lastPathComponent })
    }

    /// Higher step first; files without a step after those with one; ties
    /// by newest file.
    static func isMoreAdvanced(_ lhs: ModelFileEntry, _ rhs: ModelFileEntry) -> Bool {
        switch (lhs.trainingStep, rhs.trainingStep) {
        case let (left?, right?) where left != right:
            return left > right
        case (.some, .none):
            return true
        case (.none, .some):
            return false
        default:
            return lhs.fileModifiedAt > rhs.fileModifiedAt
        }
    }

    static func entry(for url: URL) throws -> ModelFileEntry {
        try entry(for: url, metadata: try headerMetadata(at: url))
    }

    /// The entry for `url` from its header's `__metadata__`, already read.
    static func entry(for url: URL, metadata: [String: String]) throws -> ModelFileEntry {
        guard let modelID = metadata["model_id"], !modelID.isEmpty else {
            throw ModelFileCatalogError.missingModelID(file: url.lastPathComponent)
        }
        let values = try url.resourceValues(forKeys: [.contentModificationDateKey])
        guard let modified = values.contentModificationDate else {
            throw ModelFileCatalogError.notSafetensors(file: url.lastPathComponent, detail: "no modification date")
        }
        let name = url.lastPathComponent
        let label: String
        if metadata[SafetensorsModelIO.Key.architecture] != nil {
            // Same format-version gate as a full load. Display-only, so legacy
            // resolutions are not logged here — the real load logs them.
            do {
                label = try SafetensorsModelIO.decodeArchitecture(fromMetadata: metadata, source: name).architecture.shortLabel
            } catch {
                throw ModelFileCatalogError.notSafetensors(file: name, detail: "unreadable architecture: \(String(describing: error))")
            }
        } else {
            label = "no architecture recorded"
        }
        // The file's trainer step where it records one, else the step it
        // states (`ModelFileStepReading`): one rule for files before and
        // after format v11, so the number shown is the trainer step wherever
        // one is known. Order within a line (one model ID, one writer) is
        // the same as by the raw value. The reading decodes the lineage
        // record only when it needs it; a record it needs that does not
        // decode makes this an error entry, as an unreadable architecture
        // does. Display-only, so its legacy entry is not logged here.
        let trainingStep: Int?
        do {
            trainingStep = try SafetensorsModelIO.trainingStepReading(fromMetadata: metadata, source: name)
                .trainerStepOrStatedStep
        } catch {
            throw ModelFileCatalogError.notSafetensors(file: name, detail: "unreadable training step: \(String(describing: error))")
        }
        var createdAt: Date?
        if let text = metadata["created_at_unix"] {
            guard let seconds = TimeInterval(text) else {
                throw ModelFileCatalogError.notSafetensors(file: name, detail: "created_at_unix \"\(text)\" is not a number")
            }
            createdAt = Date(timeIntervalSince1970: seconds)
        }
        // The one header-only lineage reader. Anything it refuses (the
        // record, or the format version, trainer clock or derivation history
        // read with it) makes the lineage unreadable, not the file.
        let lineage: ModelFileLineageFacts
        do {
            switch try SafetensorsModelIO.readParentFile(fromMetadata: metadata, source: name).lineage {
            case .recorded(let record):
                lineage = .recorded(ModelFileLineagePosition(record: record))
            case .unrecorded(let formatVersion):
                lineage = .unrecorded(formatVersion: formatVersion)
            }
        } catch {
            lineage = .unreadable(reason: String(describing: error))
        }
        return ModelFileEntry(
            url: url,
            modelID: modelID,
            trainingStep: trainingStep,
            createdAt: createdAt,
            architectureLabel: label,
            fileModifiedAt: modified,
            parentModelID: metadata["parent_model_id"].flatMap { $0.isEmpty ? nil : $0 },
            creator: metadata["creator"].flatMap { $0.isEmpty ? nil : $0 },
            contentSHA256: metadata[SafetensorsFile.contentHashKey],
            lineage: lineage
        )
    }

    /// The `__metadata__` map from a safetensors header, reading only the
    /// header.
    static func headerMetadata(at url: URL) throws -> [String: String] {
        let name = url.lastPathComponent
        let handle = try FileHandle(forReadingFrom: url)
        defer {
            do {
                try handle.close()
            } catch {
                SessionLogger.shared.log("[MODELS] closing \(name) failed: \(error.localizedDescription)")
            }
        }
        guard let lengthBytes = try handle.read(upToCount: 8), lengthBytes.count == 8 else {
            throw ModelFileCatalogError.notSafetensors(file: name, detail: "shorter than the header length")
        }
        var length: UInt64 = 0
        for (index, byte) in lengthBytes.enumerated() {
            length |= UInt64(byte) << (8 * UInt64(index))
        }
        guard length > 0, length < maximumHeaderBytes else {
            throw ModelFileCatalogError.notSafetensors(file: name, detail: "implausible header length \(length)")
        }
        guard let headerBytes = try handle.read(upToCount: Int(length)), headerBytes.count == Int(length) else {
            throw ModelFileCatalogError.notSafetensors(file: name, detail: "truncated header")
        }
        let object: Any
        do {
            object = try JSONSerialization.jsonObject(with: headerBytes)
        } catch {
            throw ModelFileCatalogError.notSafetensors(file: name, detail: "header is not JSON")
        }
        guard let header = object as? [String: Any] else {
            throw ModelFileCatalogError.notSafetensors(file: name, detail: "header is not a JSON object")
        }
        guard let metadata = header["__metadata__"] as? [String: String] else {
            throw ModelFileCatalogError.notSafetensors(file: name, detail: "no __metadata__ string map")
        }
        return metadata
    }

    /// `scanSessionChampions` off the caller's thread.
    static func scanSessionChampionsInBackground(directory: URL) async throws -> (champions: [SessionChampion], unreadable: [UnreadableModelFile]) {
        try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                continuation.resume(with: Result { try scanSessionChampions(directory: directory) })
            }
        }
    }

    /// Every session's champion (`<session>.dcmsession/champion.safetensors`),
    /// newest session first. A session without a readable champion is
    /// reported, not skipped.
    static func scanSessionChampions(directory: URL) throws -> (champions: [SessionChampion], unreadable: [UnreadableModelFile]) {
        let sessions = try FileManager.default.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: [.contentModificationDateKey],
            options: [.skipsHiddenFiles]
        ).filter { $0.pathExtension == "dcmsession" }
        var champions: [SessionChampion] = []
        var unreadable: [UnreadableModelFile] = []
        for session in sessions {
            let champion = session.appendingPathComponent("champion.safetensors")
            do {
                champions.append(SessionChampion(sessionName: session.deletingPathExtension().lastPathComponent, entry: try entry(for: champion)))
            } catch {
                unreadable.append(UnreadableModelFile(url: champion, reason: error.localizedDescription))
            }
        }
        return (champions.sorted { $0.entry.fileModifiedAt > $1.entry.fileModifiedAt }, unreadable)
    }

    /// Headers are small (the architecture JSON and a tensor index); a
    /// length beyond this means the file is not a safetensors model.
    private static let maximumHeaderBytes: UInt64 = 64 << 20
}
