import AppKit
import Foundation

// MARK: - Errors

enum CheckpointManagerError: LocalizedError {
    case directoryCreationFailed(URL, Error)
    case writeFailed(URL, Error)
    case readFailed(URL, Error)
    case targetAlreadyExists(URL)
    case verificationBytesDiffer(tensorIndex: Int, offset: Int)
    case verificationTensorCountMismatch(expected: Int, got: Int)
    case verificationTensorSizeMismatch(tensorIndex: Int, expected: Int, got: Int)
    case verificationForwardPassValueDiffers
    case verificationForwardPassPolicyDiffers(index: Int)
    case verificationForwardPassFailed(Error)
    case verificationScratchBuildFailed(Error)
    case loadWeightsFailed(Error)
    case sessionWriteFailed(String)

    var errorDescription: String? {
        switch self {
        case .directoryCreationFailed(let url, let err):
            return "Could not create directory \(url.path): \(err.localizedDescription)"
        case .writeFailed(let url, let err):
            return "Write failed for \(url.lastPathComponent): \(err.localizedDescription)"
        case .readFailed(let url, let err):
            return "Read failed for \(url.lastPathComponent): \(err.localizedDescription)"
        case .targetAlreadyExists(let url):
            return "Target already exists (never overwriting): \(url.lastPathComponent)"
        case .verificationBytesDiffer(let tensorIndex, let offset):
            return "Post-save weight byte compare failed at tensor \(tensorIndex) offset \(offset)"
        case .verificationTensorCountMismatch(let expected, let got):
            return "Post-save tensor count mismatch: expected \(expected), got \(got)"
        case .verificationTensorSizeMismatch(let tensorIndex, let expected, let got):
            return "Post-save tensor \(tensorIndex) element count mismatch: expected \(expected), got \(got)"
        case .verificationForwardPassValueDiffers:
            return "Post-save forward-pass verification: value head output differs"
        case .verificationForwardPassPolicyDiffers(let index):
            return "Post-save forward-pass verification: policy output differs at index \(index)"
        case .verificationForwardPassFailed(let err):
            return "Post-save forward-pass verification raised: \(err.localizedDescription)"
        case .verificationScratchBuildFailed(let err):
            return "Could not build scratch network for verification: \(err.localizedDescription)"
        case .loadWeightsFailed(let err):
            return "Could not load weights into scratch network: \(err.localizedDescription)"
        case .sessionWriteFailed(let detail):
            return "Session write failed: \(detail)"
        }
    }
}

// MARK: - Paths

/// Canonical locations for all checkpoint files. Hard-coded under
/// `~/Library/Application Support/DrewsChessMachine/` so there is
/// exactly one place to look for saved state. A `Reveal Saves`
/// button in the UI opens the relevant subfolder since
/// `Application Support` is hidden by default in Finder.
enum CheckpointPaths {
    /// Root: `~/Library/Application Support/DrewsChessMachine/`.
    static var rootURL: URL {
        let fm = FileManager.default
        let support = fm.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? URL(fileURLWithPath: NSHomeDirectory(), isDirectory: true)
                .appendingPathComponent("Library/Application Support", isDirectory: true)
        return support.appendingPathComponent("DrewsChessMachine", isDirectory: true)
    }

    /// `~/Library/Application Support/DrewsChessMachine/Sessions/`.
    static var sessionsDir: URL {
        rootURL.appendingPathComponent("Sessions", isDirectory: true)
    }

    /// `~/Library/Application Support/DrewsChessMachine/Models/`.
    static var modelsDir: URL {
        rootURL.appendingPathComponent("Models", isDirectory: true)
    }

    /// Create all checkpoint subdirectories if they don't already
    /// exist. Idempotent and cheap — called from every save path.
    static func ensureDirectories() throws {
        let fm = FileManager.default
        do {
            try fm.createDirectory(at: sessionsDir, withIntermediateDirectories: true)
        } catch {
            throw CheckpointManagerError.directoryCreationFailed(sessionsDir, error)
        }
        do {
            try fm.createDirectory(at: modelsDir, withIntermediateDirectories: true)
        } catch {
            throw CheckpointManagerError.directoryCreationFailed(modelsDir, error)
        }
    }

    /// POSIX/UTC timestamp formatter used as the leading sort key in
    /// every generated filename so Finder's alphabetical order is
    /// also chronological order.
    private static let filenameTimestampFormatter: DateFormatter = {
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.timeZone = TimeZone(identifier: "UTC")
        f.dateFormat = "yyyyMMdd-HHmmss"
        return f
    }()

    /// Build a checkpoint filename stem of the form
    /// `YYYYMMDD-HHMMSS-<modelID>-<trigger>`. `trigger` is one of
    /// `manual`, `promote`, or another short tag describing why the
    /// file was written.
    static func makeFilename(
        modelID: String,
        trigger: String,
        ext: String,
        at date: Date = Date()
    ) -> String {
        let ts = filenameTimestampFormatter.string(from: date)
        let safeID = modelID.isEmpty ? "unknown" : modelID
        return "\(ts)-\(safeID)-\(trigger).\(ext)"
    }

    /// Build a session directory name. Uses `sessionID` (which is
    /// stable across autosaves in the same run) rather than a fresh
    /// model ID so multiple autosaves during one session cluster
    /// together alphabetically without colliding.
    static func makeSessionDirectoryName(
        sessionID: String,
        trigger: String,
        at date: Date = Date()
    ) -> String {
        let ts = filenameTimestampFormatter.string(from: date)
        let safeID = sessionID.isEmpty ? "unknown" : sessionID
        return "\(ts)-\(safeID)-\(trigger).dcmsession"
    }
}

// MARK: - Load result shapes

/// Result of reading a session directory. Weights aren't loaded
/// into any live network yet — the caller decides where they go.
struct LoadedSession {
    let directoryURL: URL
    let state: SessionCheckpointState
    let championFile: ModelCheckpointFile
    let trainerFile: ModelCheckpointFile
}

// MARK: - Save / Load / Verify

/// Top-level save and load orchestration. Every save runs the full
/// atomic-write + self-verify sequence before it is considered
/// successful; any verification failure removes the partial file
/// and leaves prior saves untouched. Nothing on disk is ever
/// overwritten — every save lands under a unique timestamped name
/// in the canonical Library folders.
enum CheckpointManager {
    // MARK: Save a single model

    /// Write a standalone `.dcmmodel` into `Models/`, run
    /// post-save verification, and return the final URL. Callers
    /// pass already-exported weights (the live network should have
    /// been paused before the export) — this function never
    /// touches the caller's network for reads.
    ///
    /// Runs synchronously in the caller's task context. Callers
    /// should invoke via `Task.detached` to keep MPSGraph work off
    /// the main actor.
    static func saveModel(
        weights: [[Float]],
        modelID: String,
        createdAtUnix: Int64,
        metadata: ModelCheckpointMetadata,
        trigger: String,
        at date: Date = Date()
    ) throws -> URL {
        try CheckpointPaths.ensureDirectories()

        let file = ModelCheckpointFile(
            modelID: modelID,
            createdAtUnix: createdAtUnix,
            metadata: metadata,
            weights: weights
        )
        let encoded = try file.encode()

        let filename = CheckpointPaths.makeFilename(
            modelID: modelID,
            trigger: trigger,
            ext: "dcmmodel",
            at: date
        )
        let finalURL = CheckpointPaths.modelsDir.appendingPathComponent(filename)
        let tmpURL = finalURL.appendingPathExtension("tmp")

        if FileManager.default.fileExists(atPath: finalURL.path) {
            // Never overwrite. Timestamp-to-the-second collisions are
            // extraordinarily unlikely but refuse cleanly if it ever
            // happens rather than silently stomping prior history.
            throw CheckpointManagerError.targetAlreadyExists(finalURL)
        }

        do {
            try encoded.write(to: tmpURL, options: [.atomic])
        } catch {
            try? FileManager.default.removeItem(at: tmpURL)
            throw CheckpointManagerError.writeFailed(tmpURL, error)
        }

        // Verify BEFORE the rename so a failed check leaves nothing
        // with the final name.
        do {
            try verifyModelFile(at: tmpURL, expectedWeights: weights)
        } catch {
            try? FileManager.default.removeItem(at: tmpURL)
            throw error
        }

        do {
            try FileManager.default.moveItem(at: tmpURL, to: finalURL)
        } catch {
            try? FileManager.default.removeItem(at: tmpURL)
            throw CheckpointManagerError.writeFailed(finalURL, error)
        }

        return finalURL
    }

    // MARK: Save a session

    /// Write a `.dcmsession` directory containing champion and
    /// trainer `.dcmmodel` files plus `session.json`, run post-save
    /// verification on both model files, and return the final
    /// directory URL. Like `saveModel`, takes already-exported
    /// weights — callers handle any gate pausing needed to snapshot
    /// live networks safely.
    static func saveSession(
        championWeights: [[Float]],
        championID: String,
        championMetadata: ModelCheckpointMetadata,
        championCreatedAtUnix: Int64,
        trainerWeights: [[Float]],
        trainerID: String,
        trainerMetadata: ModelCheckpointMetadata,
        trainerCreatedAtUnix: Int64,
        state: SessionCheckpointState,
        trigger: String,
        at date: Date = Date()
    ) throws -> URL {
        try CheckpointPaths.ensureDirectories()

        let dirName = CheckpointPaths.makeSessionDirectoryName(
            sessionID: state.sessionID,
            trigger: trigger,
            at: date
        )
        let finalDirURL = CheckpointPaths.sessionsDir.appendingPathComponent(dirName, isDirectory: true)
        let tmpDirURL = CheckpointPaths.sessionsDir.appendingPathComponent(dirName + ".tmp", isDirectory: true)

        if FileManager.default.fileExists(atPath: finalDirURL.path) {
            throw CheckpointManagerError.targetAlreadyExists(finalDirURL)
        }

        // Encode everything up front so we don't leave a half-filled
        // tmp directory around on an encoding failure.
        let championFile = ModelCheckpointFile(
            modelID: championID,
            createdAtUnix: championCreatedAtUnix,
            metadata: championMetadata,
            weights: championWeights
        )
        let championEncoded = try championFile.encode()

        let trainerFile = ModelCheckpointFile(
            modelID: trainerID,
            createdAtUnix: trainerCreatedAtUnix,
            metadata: trainerMetadata,
            weights: trainerWeights
        )
        let trainerEncoded = try trainerFile.encode()

        let stateEncoded = try state.encode()

        // Build the staging directory, write the three files, verify.
        let fm = FileManager.default
        do {
            try fm.createDirectory(at: tmpDirURL, withIntermediateDirectories: true)
        } catch {
            throw CheckpointManagerError.directoryCreationFailed(tmpDirURL, error)
        }

        func cleanupTmp() {
            try? fm.removeItem(at: tmpDirURL)
        }

        let championTmpURL = SessionCheckpointLayout.championURL(in: tmpDirURL)
        let trainerTmpURL = SessionCheckpointLayout.trainerURL(in: tmpDirURL)
        let stateTmpURL = SessionCheckpointLayout.stateURL(in: tmpDirURL)

        do {
            try championEncoded.write(to: championTmpURL, options: [.atomic])
            try trainerEncoded.write(to: trainerTmpURL, options: [.atomic])
            try stateEncoded.write(to: stateTmpURL, options: [.atomic])
        } catch {
            cleanupTmp()
            throw CheckpointManagerError.writeFailed(tmpDirURL, error)
        }

        do {
            try verifyModelFile(at: championTmpURL, expectedWeights: championWeights)
            try verifyModelFile(at: trainerTmpURL, expectedWeights: trainerWeights)
            // Round-trip session.json: decode the bytes we just wrote
            // and confirm they reproduce the struct. Catches JSON
            // encoder/decoder asymmetries (e.g. Float precision).
            let writtenStateData = try Data(contentsOf: stateTmpURL)
            let writtenState = try SessionCheckpointState.decode(writtenStateData)
            guard writtenState == state else {
                throw CheckpointManagerError.sessionWriteFailed(
                    "session.json round-trip decoded to a different struct"
                )
            }
        } catch {
            cleanupTmp()
            throw error
        }

        do {
            try fm.moveItem(at: tmpDirURL, to: finalDirURL)
        } catch {
            cleanupTmp()
            throw CheckpointManagerError.writeFailed(finalDirURL, error)
        }

        return finalDirURL
    }

    // MARK: Load

    /// Parse a `.dcmmodel` file from disk. Runs the full decode
    /// pipeline including SHA-256 and arch checks. Returns the
    /// parsed struct; the caller is responsible for loading the
    /// weights into a live network.
    static func loadModelFile(at url: URL) throws -> ModelCheckpointFile {
        let data: Data
        do {
            data = try Data(contentsOf: url)
        } catch {
            throw CheckpointManagerError.readFailed(url, error)
        }
        return try ModelCheckpointFile.decode(data)
    }

    /// Parse a `.dcmsession` directory from disk. Reads all three
    /// files, decodes them, and returns them together. No weights
    /// are loaded into any live network — the caller decides the
    /// restore path.
    static func loadSession(at directoryURL: URL) throws -> LoadedSession {
        let (stateData, championData, trainerData) = try SessionCheckpointLayout.readAll(from: directoryURL)
        let state = try SessionCheckpointState.decode(stateData)
        let championFile = try ModelCheckpointFile.decode(championData)
        let trainerFile = try ModelCheckpointFile.decode(trainerData)
        return LoadedSession(
            directoryURL: directoryURL,
            state: state,
            championFile: championFile,
            trainerFile: trainerFile
        )
    }

    // MARK: Verification

    /// Post-save verification pipeline — runs on every save before
    /// the temp file is renamed into place. Two checks, in order:
    ///
    /// 1. **Bit-exact weight round-trip.** Re-read the file from
    ///    disk and byte-compare every weight tensor against what
    ///    was passed to `saveModel`. Catches file-format bugs.
    ///
    /// 2. **Forward-pass round-trip.** Build a throwaway
    ///    inference network, load the round-tripped weights into
    ///    it, run a forward pass on the starting position, then
    ///    load the ORIGINAL pre-save weights into the same network
    ///    and run the same pass. Bit-compare the policy and value
    ///    outputs. Catches `loadWeights` + `exportWeights`
    ///    regressions that leave MPS state in a subtly wrong
    ///    condition the tensor read-back wouldn't notice.
    ///
    /// Any failure throws — the save path then deletes the tmp
    /// file and surfaces the error. The scratch network is built
    /// fresh on every call; at ~100 ms per build it's acceptable
    /// because saves are infrequent.
    static func verifyModelFile(
        at url: URL,
        expectedWeights: [[Float]]
    ) throws {
        // 1. Re-read and byte-compare.
        let data: Data
        do {
            data = try Data(contentsOf: url)
        } catch {
            throw CheckpointManagerError.readFailed(url, error)
        }
        let readBack: ModelCheckpointFile
        do {
            readBack = try ModelCheckpointFile.decode(data)
        } catch {
            throw error
        }

        guard readBack.weights.count == expectedWeights.count else {
            throw CheckpointManagerError.verificationTensorCountMismatch(
                expected: expectedWeights.count,
                got: readBack.weights.count
            )
        }
        for (i, (fresh, onDisk)) in zip(expectedWeights, readBack.weights).enumerated() {
            guard fresh.count == onDisk.count else {
                throw CheckpointManagerError.verificationTensorSizeMismatch(
                    tensorIndex: i,
                    expected: fresh.count,
                    got: onDisk.count
                )
            }
            for j in 0..<fresh.count where fresh[j].bitPattern != onDisk[j].bitPattern {
                throw CheckpointManagerError.verificationBytesDiffer(
                    tensorIndex: i,
                    offset: j
                )
            }
        }

        // 2. Forward-pass round-trip through a throwaway inference
        //    network. Compares a "load pre-save weights" run against
        //    a "load post-save weights" run so any divergence in
        //    loadWeights → graph state is caught end-to-end.
        let scratch: ChessMPSNetwork
        do {
            scratch = try ChessMPSNetwork(.randomWeights)
        } catch {
            throw CheckpointManagerError.verificationScratchBuildFailed(error)
        }

        let testBoard = BoardEncoder.encode(.starting)

        let preValue: Float
        let prePolicy: [Float]
        do {
            try scratch.network.loadWeights(expectedWeights)
            let result = try scratch.evaluate(board: testBoard)
            preValue = result.value
            prePolicy = Array(result.policy)
        } catch {
            throw CheckpointManagerError.verificationForwardPassFailed(error)
        }

        let postValue: Float
        let postPolicy: [Float]
        do {
            try scratch.network.loadWeights(readBack.weights)
            let result = try scratch.evaluate(board: testBoard)
            postValue = result.value
            postPolicy = Array(result.policy)
        } catch {
            throw CheckpointManagerError.verificationForwardPassFailed(error)
        }

        guard preValue.bitPattern == postValue.bitPattern else {
            throw CheckpointManagerError.verificationForwardPassValueDiffers
        }
        guard prePolicy.count == postPolicy.count else {
            throw CheckpointManagerError.verificationForwardPassPolicyDiffers(index: -1)
        }
        for i in 0..<prePolicy.count where prePolicy[i].bitPattern != postPolicy[i].bitPattern {
            throw CheckpointManagerError.verificationForwardPassPolicyDiffers(index: i)
        }
    }

    // MARK: Finder reveal

    /// Open the given checkpoint folder or file in Finder. Called
    /// from the `Reveal Saves` button. Main actor because
    /// `NSWorkspace` expects it.
    @MainActor
    static func revealInFinder(_ url: URL) {
        NSWorkspace.shared.activateFileViewerSelecting([url])
    }
}
