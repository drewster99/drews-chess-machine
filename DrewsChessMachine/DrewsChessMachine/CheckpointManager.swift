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
    case fsyncFailed(URL, Error)
    case replayVerificationFailed(String)
    case sessionReplayMismatch(detail: String)

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
        case .fsyncFailed(let url, let err):
            return "Could not flush \(url.lastPathComponent) to stable storage: \(err.localizedDescription)"
        case .replayVerificationFailed(let detail):
            return "Post-save replay-buffer verification failed: \(detail)"
        case .sessionReplayMismatch(let detail):
            return "Session/replay-buffer mismatch on load: \(detail)"
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

    /// Remove orphan `.tmp` directories and files left behind by a
    /// save that was interrupted mid-flight (process killed, kernel
    /// panic, power loss). Runs once at app launch — no save or load
    /// has a chance to contend with a `*.tmp` debris path. Each
    /// removal is logged via `[CLEANUP]`; individual failures log
    /// `[CLEANUP-ERR]` and do not abort the sweep, since a stuck
    /// orphan should not prevent the app from starting.
    ///
    /// Two sweep patterns:
    /// - `Sessions/<name>.tmp` — `saveSession`'s staging directory
    ///   suffix. Entries here are directories.
    /// - `Models/<name>.dcmmodel.tmp` — `saveModel`'s tmp file
    ///   suffix. Entries here are files.
    static func cleanupOrphans() {
        let fm = FileManager.default
        let sweep = { (root: URL, suffix: String) in
            let entries: [URL]
            do {
                entries = try fm.contentsOfDirectory(
                    at: root,
                    includingPropertiesForKeys: nil,
                    options: [.skipsHiddenFiles]
                )
            } catch {
                SessionLogger.shared.log(
                    "[CLEANUP-ERR] Could not list \(root.lastPathComponent): \(error.localizedDescription)"
                )
                return
            }
            for entry in entries where entry.lastPathComponent.hasSuffix(suffix) {
                do {
                    try fm.removeItem(at: entry)
                    SessionLogger.shared.log(
                        "[CLEANUP] Removed orphan \(entry.lastPathComponent)"
                    )
                } catch {
                    SessionLogger.shared.log(
                        "[CLEANUP-ERR] Could not remove \(entry.lastPathComponent): \(error.localizedDescription)"
                    )
                }
            }
        }
        sweep(sessionsDir, ".tmp")
        sweep(modelsDir, ".dcmmodel.tmp")
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
    /// URL of the replay-buffer binary, if one exists in the session
    /// directory and the state flags it as present. `nil` for older
    /// sessions or sessions saved without a replay buffer.
    let replayBufferURL: URL?
}

/// Lightweight, weights-free projection of a `SessionCheckpointState`
/// — exactly the fields the auto-resume sheet wants to surface up
/// front: when training started, what was saved, how much progress
/// has accumulated, and which build produced the save. Built from
/// the session's `session.json` alone so the resume prompt can
/// render rich context without paying the cost of loading either
/// `.dcmmodel` or the replay buffer.
///
/// `arenaCount` and `promotionCount` are derived from
/// `state.arenaHistory` so the sheet doesn't need to redo the same
/// reduction at every render.
struct SessionResumeSummary: Sendable, Equatable {
    let sessionID: String
    let sessionStartUnix: Int64
    let savedAtUnix: Int64
    let elapsedTrainingSec: Double
    let trainingSteps: Int
    let trainingPositionsSeen: Int
    let selfPlayGames: Int
    let selfPlayMoves: Int
    let replayBufferTotalPositionsAdded: Int?
    let arenaCount: Int
    let promotionCount: Int
    let buildNumber: Int?
    let buildGitHash: String?
    let buildGitDirty: Bool?
    let buildTimestamp: String?

    init(state: SessionCheckpointState) {
        self.sessionID = state.sessionID
        self.sessionStartUnix = state.sessionStartUnix
        self.savedAtUnix = state.savedAtUnix
        self.elapsedTrainingSec = state.elapsedTrainingSec
        self.trainingSteps = state.trainingSteps
        self.trainingPositionsSeen = state.trainingPositionsSeen
        self.selfPlayGames = state.selfPlayGames
        self.selfPlayMoves = state.selfPlayMoves
        self.replayBufferTotalPositionsAdded = state.replayBufferTotalPositionsAdded
        self.arenaCount = state.arenaHistory.count
        self.promotionCount = state.arenaHistory.lazy.filter { $0.promoted }.count
        self.buildNumber = state.buildNumber
        self.buildGitHash = state.buildGitHash
        self.buildGitDirty = state.buildGitDirty
        self.buildTimestamp = state.buildTimestamp
    }
}

// MARK: - Save / Load / Verify

/// Top-level save and load orchestration. Every save runs the full
/// atomic-write + self-verify sequence before it is considered
/// successful; any verification failure removes the partial file
/// and leaves prior saves untouched. Nothing on disk is ever
/// overwritten — every save lands under a unique timestamped name
/// in the canonical Library folders.
enum CheckpointManager {
    // MARK: Low-level durability helpers

    /// Force every dirty page for the file or directory at `url` out
    /// to stable storage, bypassing the drive's write cache. This is
    /// the "your data is on the platter, for real" guarantee on Apple
    /// filesystems — regular `fsync(2)` only commits to the device's
    /// cache, which can still be lost on a power-cut before the drive
    /// flushes its cache on its own schedule.
    ///
    /// Implementation: open the path read-only (works for both files
    /// and directories on macOS), issue `fcntl(fd, F_FULLFSYNC)`. If
    /// F_FULLFSYNC is not supported (some network filesystems, etc.),
    /// fall back to a regular `fsync`. Throws if neither succeeds.
    ///
    /// Called from `saveSession` on every file inside the staging
    /// directory, on the staging directory itself just before the
    /// atomic rename, and on the parent `Sessions` directory after
    /// the rename.
    static func fullSyncPath(_ url: URL) throws {
        let fd = open(url.path, O_RDONLY)
        guard fd >= 0 else {
            let err = NSError(
                domain: NSPOSIXErrorDomain,
                code: Int(errno),
                userInfo: [NSLocalizedDescriptionKey: String(cString: strerror(errno))]
            )
            throw CheckpointManagerError.fsyncFailed(url, err)
        }
        defer { close(fd) }
        if fcntl(fd, F_FULLFSYNC) == -1 {
            // Fall back to regular fsync — F_FULLFSYNC is not supported
            // on every filesystem (notably some network mounts).
            if fsync(fd) == -1 {
                let err = NSError(
                    domain: NSPOSIXErrorDomain,
                    code: Int(errno),
                    userInfo: [NSLocalizedDescriptionKey: String(cString: strerror(errno))]
                )
                throw CheckpointManagerError.fsyncFailed(url, err)
            }
        }
    }

    /// Verify a freshly-restored replay buffer's lifetime counter
    /// matches what `session.json` said it should be. Used at session
    /// load time as a defense-in-depth cross-check after the buffer's
    /// own SHA and size guards have already succeeded.
    ///
    /// Only `totalPositionsAdded` is checked, not `storedCount` or
    /// `capacity` — those two intentionally diverge when loading a
    /// larger saved ring into a smaller live one (see
    /// `ReplayBuffer.restore`'s skip-oldest-entries logic). The
    /// lifetime counter survives the restore verbatim and is an
    /// effectively unique fingerprint across sessions, so a mismatch
    /// here strongly implies a file-pairing error (replay buffer
    /// from one save paired with session.json from another) or
    /// residual corruption that happened to SHA-match.
    ///
    /// A missing `replayBufferTotalPositionsAdded` in `state`
    /// (Optional for back-compat with older session.json files) skips
    /// the check rather than forcing a mismatch.
    static func verifyReplayBufferMatchesSession(
        buffer: ReplayBuffer,
        state: SessionCheckpointState
    ) throws {
        guard let expected = state.replayBufferTotalPositionsAdded else {
            return
        }
        let snap = buffer.stateSnapshot()
        guard expected == snap.totalPositionsAdded else {
            throw CheckpointManagerError.sessionReplayMismatch(
                detail: "totalPositionsAdded: session.json says \(expected), replay buffer file says \(snap.totalPositionsAdded)"
            )
        }
    }

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
    ) async throws -> URL {
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

        // Flush tmp file to platter before verify + rename so a
        // crash after verify-returns can't leave a torn file behind.
        do {
            try fullSyncPath(tmpURL)
        } catch {
            try? FileManager.default.removeItem(at: tmpURL)
            throw error
        }

        // Verify BEFORE the rename so a failed check leaves nothing
        // with the final name.
        do {
            try await verifyModelFile(at: tmpURL, expectedWeights: weights)
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

        // Flush parent directory so the rename (directory-entry
        // change) is durable.
        do {
            try fullSyncPath(CheckpointPaths.modelsDir)
        } catch {
            SessionLogger.shared.log(
                "[CHECKPOINT] fullSyncPath(modelsDir) failed after rename: \(error.localizedDescription) — file visible but parent-directory flush not guaranteed"
            )
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
        replayBuffer: ReplayBuffer? = nil,
        trigger: String,
        at date: Date = Date()
    ) async throws -> URL {
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
        let bufferTmpURL = SessionCheckpointLayout.replayBufferURL(in: tmpDirURL)
        let wantsReplayBuffer = replayBuffer != nil && state.hasReplayBuffer == true

        do {
            try championEncoded.write(to: championTmpURL, options: [.atomic])
            try trainerEncoded.write(to: trainerTmpURL, options: [.atomic])
            try stateEncoded.write(to: stateTmpURL, options: [.atomic])
        } catch {
            cleanupTmp()
            throw CheckpointManagerError.writeFailed(tmpDirURL, error)
        }

        // Optional replay-buffer dump. Written only when the caller
        // passes a buffer AND the state flags `hasReplayBuffer == true`.
        // Errors propagate — the tmp dir is cleaned up and the save
        // fails, rather than silently producing a session whose
        // session.json promises a replay buffer that isn't there.
        // ReplayBuffer.write itself calls handle.synchronize() before
        // close; the fullSyncPath below adds F_FULLFSYNC on top for
        // drive-cache-bypass durability.
        //
        // The returned `writtenSnap` captures the ring state that
        // was actually serialized into the file (captured atomically
        // under the write lock). We save it for the post-fsync
        // verification phase below because concurrent self-play
        // appends may advance the live ring past that state between
        // now and then, and the verify needs ground truth from the
        // moment of write, not the current live moment.
        var writtenSnap: ReplayBuffer.StateSnapshot? = nil
        if let replayBuffer, wantsReplayBuffer {
            do {
                writtenSnap = try replayBuffer.write(to: bufferTmpURL)
            } catch {
                cleanupTmp()
                throw CheckpointManagerError.writeFailed(bufferTmpURL, error)
            }
        }

        // F_FULLFSYNC every file we just wrote. `Data.write(...,
        // options: [.atomic])` gives an atomic rename on top of a
        // normal write, but does NOT imply platter-level durability —
        // the bytes may still sit in the VFS page cache or in the
        // drive's write cache when control returns. Without this step,
        // a crash between write-returns and the kernel's eventual
        // flush leaves a file that the subsequent tmp-dir rename
        // commits as if it were valid, even though its contents are
        // torn. We also fsync the replay buffer — its write already
        // calls `synchronize()` internally, but `F_FULLFSYNC` is
        // stronger (bypasses the drive's own cache) and matches the
        // treatment the other three files get.
        do {
            try fullSyncPath(championTmpURL)
            try fullSyncPath(trainerTmpURL)
            try fullSyncPath(stateTmpURL)
            if wantsReplayBuffer {
                try fullSyncPath(bufferTmpURL)
            }
        } catch {
            cleanupTmp()
            throw error
        }

        do {
            try await verifyModelFile(at: championTmpURL, expectedWeights: championWeights)
            try await verifyModelFile(at: trainerTmpURL, expectedWeights: trainerWeights)
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
            // Replay-buffer verification: re-load the file we just
            // wrote into a scratch ReplayBuffer. The scratch restore
            // runs the full v4 validation stack — magic, version,
            // size-equality, upper-bound caps, SHA-256 trailer verify
            // — so a mismatch throws a specific PersistenceError.
            // We then compare the restored storedCount and lifetime
            // counter against what the live in-memory buffer reports;
            // any drift here indicates the write path produced bytes
            // that don't round-trip, which the SHA alone cannot catch
            // if the write is internally consistent but wrong.
            //
            // Scratch capacity is sized to the live buffer's current
            // `storedCount`, not `capacity` — a 1 M-slot ring holding
            // 300 K positions would otherwise allocate 5 GB of empty
            // ring during verify on top of the 5 GB live ring. The
            // scratch only needs enough slots to hold the saved data.
            // We compare `live.storedCount == got.storedCount` and
            // `live.totalPositionsAdded == got.totalPositionsAdded`
            // (both survive the restore verbatim); the live ring's
            // `capacity` field is intentionally NOT compared — it
            // reflects ring-allocation size, which the scratch
            // deliberately differs on.
            if wantsReplayBuffer, let written = writtenSnap {
                let scratchCapacity = max(1, written.storedCount)
                let scratch = ReplayBuffer(capacity: scratchCapacity)
                do {
                    try scratch.restore(from: bufferTmpURL)
                } catch {
                    throw CheckpointManagerError.replayVerificationFailed(
                        "scratch restore failed: \(error.localizedDescription)"
                    )
                }
                let got = scratch.stateSnapshot()
                guard written.storedCount == got.storedCount,
                      written.totalPositionsAdded == got.totalPositionsAdded else {
                    throw CheckpointManagerError.replayVerificationFailed(
                        "counter round-trip mismatch: written=(stored=\(written.storedCount), total=\(written.totalPositionsAdded)) scratch=(stored=\(got.storedCount), total=\(got.totalPositionsAdded))"
                    )
                }
            }
        } catch {
            cleanupTmp()
            throw error
        }

        // Flush the tmp directory's metadata (file entries, mtimes)
        // to stable storage before the atomic rename commits the
        // bundle. Without this, a crash between rename-commit and the
        // directory flush can leave the final-named directory whose
        // file metadata hasn't yet been written — it appears in
        // listings but the file sizes/mtimes may be wrong.
        do {
            try fullSyncPath(tmpDirURL)
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

        // Flush the parent `Sessions/` directory so the rename itself
        // (which is a directory-entry change in the parent) lands on
        // stable storage. If we skip this and the machine loses power
        // before the parent's directory block is flushed, the session
        // can disappear entirely even though the files inside it are
        // fully durable.
        do {
            try fullSyncPath(CheckpointPaths.sessionsDir)
        } catch {
            // At this point the rename has already succeeded and the
            // session is visible under its final name, so we don't
            // remove it — log and continue. Worst case: the session
            // survives the current process but not a power-cut within
            // the next few seconds. Acceptable given the rename is
            // already committed in the filesystem's in-memory view.
            SessionLogger.shared.log(
                "[CHECKPOINT] fullSyncPath(sessionsDir) failed after rename: \(error.localizedDescription) — session is still visible but flush-to-disk of directory entry is not guaranteed"
            )
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

    /// Read just `session.json` from a `.dcmsession` directory and
    /// return a lightweight `SessionResumeSummary`. Skips the two
    /// `.dcmmodel` weight files entirely — this is the fast path
    /// the auto-resume sheet uses to populate the prompt with live
    /// counters and build info before the user has decided whether
    /// to actually resume. A few KB of JSON, no Metal allocation,
    /// no replay-buffer rehydration.
    ///
    /// Throws `SessionCheckpointError.missingSessionJSON` if the
    /// state file is absent (callers fall back to a minimal sheet
    /// in that case rather than blocking the prompt). Other
    /// decode failures bubble through with their underlying
    /// `SessionCheckpointError.invalidJSON` payload so a corrupted
    /// pointer-target gets a useful log line.
    static func peekSessionMetadata(at directoryURL: URL) throws -> SessionResumeSummary {
        let normalizedDir = URL(fileURLWithPath: directoryURL.path, isDirectory: true)
        let stateURL = SessionCheckpointLayout.stateURL(in: normalizedDir)
        guard FileManager.default.fileExists(atPath: stateURL.path) else {
            throw SessionCheckpointError.missingSessionJSON
        }
        let stateData = try Data(contentsOf: stateURL)
        let state = try SessionCheckpointState.decode(stateData)
        return SessionResumeSummary(state: state)
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
        let bufferURL = SessionCheckpointLayout.replayBufferURL(in: directoryURL)
        let bufferPresent = (state.hasReplayBuffer == true)
            && FileManager.default.fileExists(atPath: bufferURL.path)
        return LoadedSession(
            directoryURL: directoryURL,
            state: state,
            championFile: championFile,
            trainerFile: trainerFile,
            replayBufferURL: bufferPresent ? bufferURL : nil
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
    ) async throws {
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
            try await scratch.loadWeights(expectedWeights)
            let result = try await scratch.evaluate(board: testBoard)
            preValue = result.value
            prePolicy = result.policy
        } catch {
            throw CheckpointManagerError.verificationForwardPassFailed(error)
        }

        let postValue: Float
        let postPolicy: [Float]
        do {
            try await scratch.loadWeights(readBack.weights)
            let result = try await scratch.evaluate(board: testBoard)
            postValue = result.value
            postPolicy = result.policy
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
