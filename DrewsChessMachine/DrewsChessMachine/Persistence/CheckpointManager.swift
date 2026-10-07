import AppKit
import Foundation

// MARK: - Errors

enum CheckpointManagerError: LocalizedError {
    case directoryCreationFailed(URL, Error)
    case writeFailed(URL, Error)
    case readFailed(URL, Error)
    case targetAlreadyExists(URL)
    case stagingPathAlreadyExists(URL)
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
    case chartFileWriteFailed(URL, Error)
    case chartFileVerifyFailed(detail: String)

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
        case .stagingPathAlreadyExists(let url):
            return "Save staging path already exists: \(url.path). It is either another save in progress or debris from an interrupted one; this save neither reuses nor deletes it. Remove it by hand once no save is running."
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
        case .chartFileWriteFailed(let url, let err):
            return "Chart-file write failed for \(url.lastPathComponent): \(err.localizedDescription)"
        case .chartFileVerifyFailed(let detail):
            return "Chart-file post-write verification failed: \(detail)"
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

    /// `~/Library/Application Support/DrewsChessMachine/Analyses/`.
    /// Output directory for offline replay-buffer analyzer runs (see
    /// `ReplayBufferAnalyzer` + the `Analyze Replay Buffer…` debug menu).
    /// Created on-demand by the analyzer's write path; not part of
    /// `ensureDirectories()` because no normal session flow writes here.
    static var analysesDir: URL {
        rootURL.appendingPathComponent("Analyses", isDirectory: true)
    }

    /// Create all checkpoint subdirectories if they don't already
    /// exist. Idempotent and cheap — called from every save path.
    static func ensureDirectories() throws {
        try ensureDirectory(sessionsDir)
        try ensureDirectory(modelsDir)
    }

    /// Create `directory` (and any missing parents) if it does not
    /// already exist. Idempotent. The save paths call this for the
    /// one directory they write into, which is the canonical
    /// `Sessions/` or `Models/` folder in the app and a temporary
    /// folder in tests.
    static func ensureDirectory(_ directory: URL) throws {
        do {
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        } catch {
            throw CheckpointManagerError.directoryCreationFailed(directory, error)
        }
    }

    // MARK: Staging names

    /// Path extension every save stages under before its final rename:
    /// `saveSession` builds `Sessions/<name>.dcmsession.tmp/` and
    /// `saveModel` writes `Models/<name>.safetensors.tmp`. The launch
    /// sweep below derives the names it recognizes from this, so the
    /// writer and the sweeper cannot drift apart.
    static let stagingPathExtension = "tmp"

    /// Name suffix of `saveSession`'s staging directory.
    static var sessionStagingSuffix: String { ".dcmsession.\(stagingPathExtension)" }

    /// Name suffixes of `saveModel`'s staging file: the current
    /// `.safetensors` format and the legacy `.dcmmodel` format, whose
    /// interrupted saves may still be on disk from older builds.
    static var modelStagingSuffixes: [String] {
        [".safetensors.\(stagingPathExtension)", ".dcmmodel.\(stagingPathExtension)"]
    }

    // MARK: Staging ownership

    /// Create `url` as a new, empty regular file and return a handle
    /// open for writing it, with the file's identity. Fails with
    /// `.stagingPathAlreadyExists` if anything at all is already at
    /// `url`.
    ///
    /// A save's error paths delete its staging item, so the save must
    /// know the item is its own. A plain write would silently adopt
    /// whatever was already at the staging path — debris from an
    /// interrupted save, or the live staging file of a save running in
    /// another instance — and the error path would then delete it.
    /// `FileSafety.createNewFile` makes the existence check and the
    /// creation one atomic step, so success proves this call created
    /// the file, and the returned identity lets the cleanup prove it is
    /// still that file.
    static func createStagingFileExclusively(at url: URL) throws -> FileSafety.NewFile {
        do {
            return try FileSafety.createNewFile(at: url)
        } catch FileSafetyError.alreadyExists {
            throw CheckpointManagerError.stagingPathAlreadyExists(url)
        } catch {
            throw CheckpointManagerError.writeFailed(url, error)
        }
    }

    /// Create `url` as a new, empty directory and return its identity.
    /// Fails with `.stagingPathAlreadyExists` if anything at all is
    /// already at `url`. The directory counterpart of
    /// `createStagingFileExclusively(at:)`, for the same reason:
    /// `FileSafety.createNewDirectory` refuses an existing path
    /// atomically, so success proves this call owns the directory.
    /// (`FileManager.createDirectory` with intermediate directories
    /// succeeds on an existing directory, which is exactly the adoption
    /// this exists to prevent.)
    static func createStagingDirectoryExclusively(at url: URL) throws -> FileSafety.FileIdentity {
        do {
            return try FileSafety.createNewDirectory(at: url)
        } catch FileSafetyError.alreadyExists {
            throw CheckpointManagerError.stagingPathAlreadyExists(url)
        } catch {
            throw CheckpointManagerError.directoryCreationFailed(url, error)
        }
    }

    /// Remove a staging item this save created and still owns, on a
    /// failure path — only if it is still the item with `identity`
    /// (`FileSafety.removeOwnedItem`), so a folder or file swapped in
    /// at the staging path is never deleted. Never throws — the save is
    /// already failing with its own error — but logs every problem: a
    /// removal failure leaves a large item behind, and an owned item
    /// that has vanished or been replaced means something else touched
    /// it while the save was running.
    static func removeOwnedStagingItem(at url: URL, identity: FileSafety.FileIdentity) {
        do {
            if try FileSafety.removeOwnedItem(at: url, identity: identity) == .alreadyGone {
                SessionLogger.shared.log(
                    "[CHECKPOINT-CLEANUP] staging item \(url.lastPathComponent) was already gone when this save went to remove it"
                )
            }
        } catch {
            SessionLogger.shared.log(
                "[CHECKPOINT-CLEANUP] failed to remove staging item \(url.lastPathComponent): \(error.localizedDescription)"
            )
        }
    }

    // MARK: Launch-time orphan sweep

    /// How long a staging item must have gone unmodified before the
    /// launch-time sweep treats it as debris from a dead save rather
    /// than a save still running in another process.
    ///
    /// "This process just launched" does not mean "no save is
    /// running": two app instances share one `Application Support`
    /// folder, and a second instance
    /// launched while the first is mid-save would delete the first
    /// one's staging folder out from under it — the first save then
    /// fails at its rename after doing all of its writing and
    /// verification, or its error cleanup trips over a folder that is
    /// already gone. So the sweep claims only items whose newest
    /// modification is older than this bound.
    ///
    /// A live save keeps its staging item fresh: every file it writes
    /// bumps the modification date of the item or one of its direct
    /// children, and its longest write-free stretch is the
    /// post-write verification read of the replay buffer. This bound
    /// sits far beyond that stretch. The cost of a long bound is only
    /// that a real orphan survives until a later launch, which delays
    /// reclaiming its space but loses nothing. The age is wall-clock,
    /// so a machine that slept through a save ages that save's staging
    /// item while it sleeps; a second instance launched right after a
    /// long sleep is the one case this bound does not cover.
    static let orphanStagingMinimumAge: TimeInterval = 60 * 60

    /// What kind of staging debris a directory is swept for.
    enum OrphanStagingKind: Sendable, Equatable {
        /// `saveSession`'s staging folder in `Sessions/`. Must be a
        /// real directory.
        case sessionDirectory
        /// `saveModel`'s staging file in `Models/`. Must be a regular
        /// file.
        case modelFile
        /// `FileSafety`'s hidden staging file for a write staged and
        /// renamed into place (`FileSafety.temporarySibling(of:)`) — e.g.
        /// a corpus-replay or train-vs-UCI rolling `--out-model` written
        /// into `Models/`. Swept in both folders. Must be a regular file.
        case fileSafetyStagingFile

        /// `true` when `name` is a staging name of this kind: for the
        /// save kinds, a non-hidden name ending in one of the kind's
        /// staging suffixes with a non-empty stem in front of it (saves
        /// never stage under a hidden name); for `FileSafety`'s staging
        /// files, exactly the shape `FileSafety` builds.
        func matchesName(_ name: String) -> Bool {
            let suffixes: [String]
            switch self {
            case .sessionDirectory: suffixes = [CheckpointPaths.sessionStagingSuffix]
            case .modelFile: suffixes = CheckpointPaths.modelStagingSuffixes
            case .fileSafetyStagingFile:
                return FileSafety.destinationName(ofTemporarySiblingName: name) != nil
            }
            guard !name.hasPrefix(".") else { return false }
            return suffixes.contains { name.count > $0.count && name.hasSuffix($0) }
        }
    }

    /// The on-disk facts the sweep decides on, gathered by
    /// `inspectOrphanStagingCandidate(at:)`. Split out from the I/O so
    /// the decision is a pure function the tests can drive directly.
    struct OrphanStagingCandidate: Sendable, Equatable {
        let name: String
        let isDirectory: Bool
        let isRegularFile: Bool
        let isSymbolicLink: Bool
        /// Newest modification date of the item itself and, for a
        /// directory, of every entry directly inside it — the last
        /// moment any save could have been working in it.
        let lastActivity: Date
    }

    /// The sweep's decision for one directory entry.
    enum OrphanStagingVerdict: Sendable, Equatable {
        /// Not a staging name of the swept kind — an ordinary saved
        /// session or model, left alone without comment.
        case notStaging
        /// A staging name, but not provably dead debris; kept and
        /// logged with the reason.
        case keep(reason: String)
        /// Dead debris of the expected kind; removed.
        case remove
    }

    /// Decide whether the launch sweep may remove `candidate`. Only an
    /// entry with the staging name of `kind`, of exactly that kind's
    /// filesystem type (never a symbolic link), and unmodified for at
    /// least `minimumAge` is removable — anything less could be a save
    /// in flight in another instance, or something that is not a
    /// save's staging item at all.
    static func orphanStagingVerdict(
        for candidate: OrphanStagingCandidate,
        kind: OrphanStagingKind,
        now: Date,
        minimumAge: TimeInterval
    ) -> OrphanStagingVerdict {
        guard kind.matchesName(candidate.name) else { return .notStaging }
        if candidate.isSymbolicLink {
            return .keep(reason: "is a symbolic link, not a staging item a save creates")
        }
        switch kind {
        case .sessionDirectory:
            guard candidate.isDirectory else {
                return .keep(reason: "is not a directory, but session staging is always a directory")
            }
        case .modelFile:
            guard candidate.isRegularFile else {
                return .keep(reason: "is not a regular file, but model staging is always a regular file")
            }
        case .fileSafetyStagingFile:
            guard candidate.isRegularFile else {
                return .keep(reason: "is not a regular file, but FileSafety only ever stages regular files (a folder with this name is one whose removal was interrupted; see documentation/disk-cleanup.md)")
            }
        }
        let age = now.timeIntervalSince(candidate.lastActivity)
        guard age >= minimumAge else {
            return .keep(
                reason: "last modified \(Int(age.rounded()))s ago, younger than the \(Int(minimumAge))s orphan age — may be a save in progress in another instance"
            )
        }
        return .remove
    }

    /// One inspected sweep entry: the facts the verdict is decided on,
    /// and the identity of the item they were read from — the only item
    /// the sweep may then remove.
    struct InspectedOrphanStagingEntry: Sendable, Equatable {
        let candidate: OrphanStagingCandidate
        let identity: FileSafety.FileIdentity
    }

    /// Read the facts `orphanStagingVerdict` needs for the entry at
    /// `url`, and the identity of the item they describe. Throws when
    /// any of them cannot be read, or when the item at `url` changed
    /// while they were being read; the sweep keeps such an entry, since
    /// it cannot prove the entry is debris.
    ///
    /// The identity is read (`lstat`) before the facts and checked again
    /// after them, so the facts are known to belong to the item whose
    /// identity the removal later insists on.
    static func inspectOrphanStagingCandidate(at url: URL) throws -> InspectedOrphanStagingEntry {
        guard let item = try FileSafety.existingItem(at: url) else {
            throw CheckpointManagerError.readFailed(
                url,
                CocoaError(.fileReadNoSuchFile, userInfo: [NSLocalizedDescriptionKey: "it is no longer there"])
            )
        }
        let isDirectory = item.kind == .directory
        let isRegularFile = item.kind == .regularFile
        let isSymbolicLink = item.kind == .symbolicLink
        guard let modified = try url.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate else {
            throw CheckpointManagerError.readFailed(
                url,
                CocoaError(.fileReadUnknown, userInfo: [NSLocalizedDescriptionKey: "modification date unavailable"])
            )
        }
        var lastActivity = modified
        if isDirectory && !isSymbolicLink {
            let children = try FileManager.default.contentsOfDirectory(
                at: url,
                includingPropertiesForKeys: [.contentModificationDateKey],
                options: []
            )
            for child in children {
                guard let childModified = try child.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate else {
                    throw CheckpointManagerError.readFailed(
                        child,
                        CocoaError(.fileReadUnknown, userInfo: [NSLocalizedDescriptionKey: "modification date unavailable"])
                    )
                }
                lastActivity = max(lastActivity, childModified)
            }
        }
        guard try FileSafety.existingItem(at: url)?.identity == item.identity else {
            throw CheckpointManagerError.readFailed(
                url,
                CocoaError(.fileReadUnknown, userInfo: [NSLocalizedDescriptionKey: "it was replaced while being inspected"])
            )
        }
        return InspectedOrphanStagingEntry(
            candidate: OrphanStagingCandidate(
                name: url.lastPathComponent,
                isDirectory: isDirectory,
                isRegularFile: isRegularFile,
                isSymbolicLink: isSymbolicLink,
                lastActivity: lastActivity
            ),
            identity: item.identity
        )
    }

    /// Remove staging debris left behind by a save or a staged file
    /// write that was interrupted mid-flight (process killed, kernel
    /// panic, power loss). Runs once at GUI launch.
    ///
    /// Launch is not a quiet moment: another app instance sharing this
    /// `Application Support` folder may be mid-save right now, and a
    /// headless trainer may be writing its rolling model into
    /// `Models/`. So an entry is removed only when
    /// `orphanStagingVerdict` proves it is dead debris — a
    /// `saveSession` staging directory (`Sessions/<name>.dcmsession.tmp`),
    /// a `saveModel` staging file (`Models/<name>.safetensors.tmp`, or
    /// the legacy `.dcmmodel.tmp`), or a `FileSafety` staging file
    /// (`.<name>.<UUID>.tmp`, hidden, in either folder), of exactly that
    /// filesystem type, unmodified for at least `minimumAge`. The
    /// removal is identity-checked (`FileSafety.removeOwnedItem` with the
    /// identity read at inspection), so an item that took the entry's
    /// place after it was judged is refused, not removed. Every removal
    /// logs `[CLEANUP]`; every staging-named entry that is kept logs
    /// `[CLEANUP]` with the reason; failures log `[CLEANUP-ERR]` and do
    /// not abort the sweep, since a stuck orphan should not prevent the
    /// app from starting.
    ///
    /// The directories, clock, and age bound are parameters so tests
    /// can sweep a temporary folder; the app uses the defaults.
    static func cleanupOrphans(
        sessionsDirectory: URL = CheckpointPaths.sessionsDir,
        modelsDirectory: URL = CheckpointPaths.modelsDir,
        now: Date = Date(),
        minimumAge: TimeInterval = CheckpointPaths.orphanStagingMinimumAge
    ) {
        sweepOrphanStaging(in: sessionsDirectory, kinds: [.sessionDirectory, .fileSafetyStagingFile],
                           now: now, minimumAge: minimumAge)
        sweepOrphanStaging(in: modelsDirectory, kinds: [.modelFile, .fileSafetyStagingFile],
                           now: now, minimumAge: minimumAge)
    }

    private static func sweepOrphanStaging(
        in directory: URL,
        kinds: [OrphanStagingKind],
        now: Date,
        minimumAge: TimeInterval
    ) {
        let entries: [URL]
        do {
            // Hidden entries included: FileSafety's staging files are
            // hidden. The save kinds' `matchesName` rejects hidden names.
            entries = try FileManager.default.contentsOfDirectory(
                at: directory,
                includingPropertiesForKeys: nil,
                options: []
            )
        } catch {
            SessionLogger.shared.log(
                "[CLEANUP-ERR] Could not list \(directory.lastPathComponent): \(error.localizedDescription)"
            )
            return
        }
        for entry in entries {
            guard let kind = kinds.first(where: { $0.matchesName(entry.lastPathComponent) }) else { continue }
            let inspected: InspectedOrphanStagingEntry
            do {
                inspected = try inspectOrphanStagingCandidate(at: entry)
            } catch {
                SessionLogger.shared.log(
                    "[CLEANUP] Kept \(entry.lastPathComponent): could not read its type, identity and modification date (\(error.localizedDescription)), so it cannot be proven to be debris"
                )
                continue
            }
            switch orphanStagingVerdict(for: inspected.candidate, kind: kind, now: now, minimumAge: minimumAge) {
            case .notStaging:
                continue
            case .keep(let reason):
                SessionLogger.shared.log("[CLEANUP] Kept \(entry.lastPathComponent): \(reason)")
            case .remove:
                removeInspectedOrphan(inspected, at: entry)
            }
        }
    }

    /// Remove the sweep entry at `url` only if it is still the item
    /// `inspected` describes, logging the outcome. Internal rather than
    /// private so tests can swap the entry between inspection and
    /// removal.
    static func removeInspectedOrphan(_ inspected: InspectedOrphanStagingEntry, at url: URL) {
        do {
            switch try FileSafety.removeOwnedItem(at: url, identity: inspected.identity) {
            case .removed:
                SessionLogger.shared.log("[CLEANUP] Removed orphan \(url.lastPathComponent)")
            case .alreadyGone:
                SessionLogger.shared.log(
                    "[CLEANUP] Orphan \(url.lastPathComponent) was already gone when the sweep went to remove it"
                )
            }
        } catch {
            SessionLogger.shared.log(
                "[CLEANUP-ERR] Could not remove \(url.lastPathComponent): \(error.localizedDescription)"
            )
        }
    }

    // MARK: Automatic-save retention

    /// The automatic session saves that share one retention pool, capped
    /// by `max_periodic_autosaves_kept`: periodic autosaves and
    /// promotion saves. Every other save — `-manual` (File ▸ Save
    /// Session), `-sigusr2` (checkpoint-and-shut-down), and any trigger
    /// added later — is a deliberate "keep this" save and is never a
    /// pool member, so the sweep can never select one.
    ///
    /// The pool is defined by disk tag, not by `SessionSaveTrigger`,
    /// because the arena's post-promotion save does not go through a
    /// `SessionSaveTrigger`: it writes the shared
    /// `SessionSaveTrigger.promotionDiskTag` directly. "Promote Trainee
    /// Now" (`SessionSaveTrigger.manualPromote`) writes the same tag and
    /// is therefore treated exactly like an arena promotion (owner
    /// decision 2026-10-01).
    ///
    /// This is also the single rule for *when* the sweep runs: after a
    /// successful save whose disk tag maps to a kind here
    /// (`init(diskTag:)`), and after no other.
    enum AutomaticSaveKind: Sendable, Equatable, CaseIterable {
        case periodic
        case promotion

        /// The tag this kind's saves carry in their folder name.
        var diskTag: String {
            switch self {
            case .periodic: SessionSaveTrigger.periodic.diskTag
            case .promotion: SessionSaveTrigger.promotionDiskTag
            }
        }

        /// The kind whose saves carry `diskTag`, or nil when saves with
        /// that tag are outside the retention pool.
        init?(diskTag: String) {
            guard let kind = Self.allCases.first(where: { $0.diskTag == diskTag }) else { return nil }
            self = kind
        }

        /// `-<diskTag>.dcmsession`, the end of every folder name of this
        /// kind.
        var folderNameSuffix: String { "-\(diskTag).dcmsession" }
    }

    /// Owner-imposed kill switch for automatic-save pruning (owner
    /// decision 2026-10-01). While this is `true`,
    /// `automaticSavePruningDecision` answers `.forcedOff` for every
    /// automatic save, whatever `automatic_save_pruning_enabled` and
    /// `max_periodic_autosaves_kept` say, so `pruneAutomaticSaves` is
    /// never reached and no session folder is ever deleted by the app.
    /// The retention code stays in the build — reviewed and tested — but
    /// nothing runs it.
    ///
    /// Why a constant on top of the setting's off default: the setting
    /// lives in UserDefaults, is restored from a resumed session's
    /// `session.json`, and can be set by a `parameters.json` or CLI
    /// override, so any of those could switch on the deletion of whole
    /// session folders — from every session, not just the running one —
    /// without anyone deciding to in this build. The switch stays thrown
    /// until session saves leave the replay buffer out by default (plan
    /// #8, decision D-8) and there is more confidence in deleting saves
    /// automatically.
    ///
    /// Changing this to `false` is the only way to re-enable pruning.
    /// `AutomaticSavePruningGateTests` pins the shipped value, so that
    /// change has to come with a deliberate, visible test edit.
    static let automaticSavePruningForcedOff = true

    /// Whether a retention sweep runs after an automatic save and, when it
    /// does not, why. Produced only by `automaticSavePruningDecision`.
    enum AutomaticSavePruningDecision: Sendable, Equatable {
        /// Run `pruneAutomaticSaves` with this cap.
        case prune(keeping: Int)
        /// The build's kill switch (`automaticSavePruningForcedOff`) is
        /// thrown; the setting and the cap were not consulted.
        case forcedOff
        /// `automatic_save_pruning_enabled` is off.
        case disabledBySetting
        /// `max_periodic_autosaves_kept` is 0, which means unlimited.
        case unlimitedCap

        /// Log-ready account of the decision together with the inputs
        /// behind it, so a skipped sweep and the Play-and-Train start
        /// line read the same way and always show both settings.
        func logDescription(settingEnabled: Bool, cap: Int) -> String {
            let reason: String
            switch self {
            case .prune(let keeping):
                reason = "on, keeping the newest \(keeping) automatic saves"
            case .forcedOff:
                reason = "off: automatic-save pruning is forced off in this build"
            case .disabledBySetting:
                reason = "off: disabled by setting"
            case .unlimitedCap:
                reason = "off: cap is 0 (unlimited)"
            }
            return "\(reason) (\(AutomaticSavePruningEnabled.id)=\(settingEnabled) \(MaxPeriodicAutosavesKept.id)=\(cap))"
        }
    }

    /// The single rule for whether automatic-save pruning runs: only when
    /// the build does not force it off, the setting is on, and the cap is
    /// above zero. The kill switch is checked first and short-circuits, so
    /// a forced-off build never acts on the setting or the cap.
    ///
    /// `forcedOff` is a parameter rather than a read of
    /// `automaticSavePruningForcedOff` so tests can reach the paths a
    /// build with the switch lifted would take. The one production caller,
    /// `SessionController.currentAutomaticSavePruningDecision` (behind both
    /// the sweep trigger and the Play-and-Train start line), passes the
    /// constant.
    static func automaticSavePruningDecision(
        forcedOff: Bool,
        settingEnabled: Bool,
        cap: Int
    ) -> AutomaticSavePruningDecision {
        if forcedOff { return .forcedOff }
        guard settingEnabled else { return .disabledBySetting }
        guard cap > 0 else { return .unlimitedCap }
        return .prune(keeping: cap)
    }

    /// `true` when `text` has exactly the shape `format` (a
    /// `DateFormatter` pattern of digit fields and literal separators,
    /// such as `filenameTimestampFormat`) renders: an ASCII digit for
    /// every pattern letter, the identical character for every other.
    private static func matchesDigitPattern(_ text: Substring, format: String) -> Bool {
        guard text.count == format.count else { return false }
        return zip(text, format).allSatisfy { character, formatCharacter in
            formatCharacter.isLetter
                ? (character.isASCII && character.isWholeNumber)
                : character == formatCharacter
        }
    }

    /// `true` when `sessionID` has the shape `ModelIDMinter.mint()`
    /// gives every Play-and-Train session: `yyyymmdd-N-XXXX` (UTC
    /// date, positive per-day counter, base62 random suffix).
    ///
    /// The retention sweep deletes a folder only when the session ID
    /// in its name matches the one inside its `session.json`, and that
    /// cross-check only proves anything for an ID that names a single
    /// session. A minted ID does: the random suffix keeps two sessions
    /// from sharing one even when they start on the same day on
    /// different machines. The placeholder IDs the save path falls back
    /// to when a session has no ID (in `makeSessionDirectoryName` and
    /// in `buildCurrentSessionState`) are shared by every session that
    /// hits them, and a folder carrying one is a sign that something
    /// went wrong when it was written — the kind of folder a person
    /// should look at before it goes. Neither placeholder has this
    /// shape, so such folders are kept and logged.
    static func isMintedSessionID(_ sessionID: String) -> Bool {
        let parts = sessionID.split(separator: "-", omittingEmptySubsequences: false)
        guard parts.count == 3 else { return false }
        let datePart = parts[0]
        let counterPart = parts[1]
        let suffixPart = parts[2]
        guard matchesDigitPattern(datePart, format: "yyyyMMdd") else { return false }
        guard !counterPart.isEmpty,
              counterPart.allSatisfy({ $0.isASCII && $0.isWholeNumber }),
              let counter = Int(counterPart), counter > 0 else { return false }
        guard !suffixPart.isEmpty,
              suffixPart.allSatisfy({ $0.isASCII && ($0.isLetter || $0.isWholeNumber) }) else { return false }
        return true
    }

    /// The parts of a folder name `makeSessionDirectoryName` produces
    /// for an automatic save.
    struct AutomaticSaveFolderName: Sendable, Equatable {
        let folderName: String
        /// Whatever sits between the timestamp and the trigger tag —
        /// a minted session ID for every save of a properly started
        /// session, a placeholder otherwise.
        let sessionID: String
        let kind: AutomaticSaveKind
    }

    /// Split `name` into its session ID and kind when it has exactly
    /// the shape `YYYYMMDD-HHMMSS-<sessionID>-(periodic|promote).dcmsession`:
    /// a bare UTC timestamp first, nothing in front of it, a non-empty
    /// session ID, and an automatic-save tag as the very end. Anything
    /// else — a manual or signal save, a staging `.tmp` folder, a
    /// renamed or hand-made folder — returns nil and is never a pool
    /// member. The session ID is not checked here; a placeholder ID
    /// still parses, so the sweep can report that folder as kept.
    static func parseAutomaticSaveFolderName(_ name: String) -> AutomaticSaveFolderName? {
        for kind in AutomaticSaveKind.allCases where name.hasSuffix(kind.folderNameSuffix) {
            let stem = name.dropLast(kind.folderNameSuffix.count)
            let timestamp = stem.prefix(filenameTimestampFormat.count)
            guard matchesDigitPattern(timestamp, format: filenameTimestampFormat) else { return nil }
            let afterTimestamp = stem.dropFirst(filenameTimestampFormat.count)
            guard afterTimestamp.first == "-" else { return nil }
            let sessionID = afterTimestamp.dropFirst()
            guard !sessionID.isEmpty else { return nil }
            return AutomaticSaveFolderName(folderName: name, sessionID: String(sessionID), kind: kind)
        }
        return nil
    }

    /// Whether a name-matched automatic-save folder is provably the save
    /// its name says it is. Only `.verified` folders are counted in the
    /// pool, ranked, and ever deleted; every other case is kept and
    /// logged with `keptReason`.
    enum AutomaticSaveVerification: Sendable, Equatable {
        /// A real directory holding a regular `session.json` whose
        /// `sessionID` is the minted ID in the folder name. Carries the
        /// directory's identity as inspected, so the deletion can prove
        /// the folder at that path is still this one.
        case verified(identity: FileSafety.FileIdentity)
        /// The ID in the folder name is not a minted session ID.
        case placeholderSessionID
        /// The entry was listed but was gone by the time it was inspected.
        case vanished
        case unreadableFolder(detail: String)
        case symbolicLink
        case notADirectory(kind: FileSafety.ItemKind)
        case missingSessionJSON
        case sessionJSONNotARegularFile(kind: FileSafety.ItemKind)
        case unreadableSessionJSON(detail: String)
        case sessionIDMismatch(found: String)

        var isVerified: Bool {
            if case .verified = self { return true }
            return false
        }

        /// Why a non-verified folder is kept, for the `[PRUNE]` log.
        var keptReason: String {
            switch self {
            case .verified: "verified"
            case .placeholderSessionID: "the session ID in its name is a placeholder, not a minted session ID"
            case .vanished: "it was no longer there when inspected"
            case .unreadableFolder(let detail): "its file type could not be read: \(detail)"
            case .symbolicLink: "it is a symbolic link, not a session folder a save creates"
            case .notADirectory(let kind): "it is a \(kind), not a directory"
            case .missingSessionJSON: "it has no \(SessionCheckpointLayout.stateFilename) inside"
            case .sessionJSONNotARegularFile(let kind): "its \(SessionCheckpointLayout.stateFilename) is a \(kind), not a regular file"
            case .unreadableSessionJSON(let detail): "its \(SessionCheckpointLayout.stateFilename) could not be read: \(detail)"
            case .sessionIDMismatch(let found): "its \(SessionCheckpointLayout.stateFilename) names session \(found)"
            }
        }
    }

    /// The only field of `session.json` verification reads. Decoding
    /// just this key, rather than the whole `SessionCheckpointState`,
    /// keeps verification independent of the format version and of
    /// every other field, so a folder written by any older build still
    /// verifies.
    private struct SessionIdentityProbe: Decodable {
        let sessionID: String
    }

    /// Establish whether the folder at `url` is the save `parsed` names,
    /// by its contents and not just its name: the name's session ID
    /// must be a minted one, and the entry must be a real directory
    /// (not a symbolic link, read with `lstat`) holding a regular
    /// `session.json` whose `sessionID` is exactly that ID.
    static func inspectAutomaticSaveFolder(
        at url: URL,
        parsed: AutomaticSaveFolderName
    ) -> AutomaticSaveVerification {
        guard isMintedSessionID(parsed.sessionID) else { return .placeholderSessionID }
        let folder: FileSafety.ExistingItem
        do {
            guard let item = try FileSafety.existingItem(at: url) else { return .vanished }
            folder = item
        } catch {
            return .unreadableFolder(detail: error.localizedDescription)
        }
        switch folder.kind {
        case .directory:
            break
        case .symbolicLink:
            return .symbolicLink
        case .regularFile, .fifo, .socket, .characterDevice, .blockDevice, .unknown:
            return .notADirectory(kind: folder.kind)
        }
        let stateURL = SessionCheckpointLayout.stateURL(in: url)
        let stateItem: FileSafety.ExistingItem
        do {
            guard let item = try FileSafety.existingItem(at: stateURL) else { return .missingSessionJSON }
            stateItem = item
        } catch {
            return .unreadableSessionJSON(detail: error.localizedDescription)
        }
        guard stateItem.kind == .regularFile else { return .sessionJSONNotARegularFile(kind: stateItem.kind) }
        let probe: SessionIdentityProbe
        do {
            probe = try JSONDecoder().decode(SessionIdentityProbe.self, from: Data(contentsOf: stateURL))
        } catch {
            return .unreadableSessionJSON(detail: error.localizedDescription)
        }
        guard probe.sessionID == parsed.sessionID else { return .sessionIDMismatch(found: probe.sessionID) }
        return .verified(identity: folder.identity)
    }

    /// One name-matched automatic-save folder and what inspection found.
    struct AutomaticSaveCandidate: Sendable, Equatable {
        let folder: AutomaticSaveFolderName
        let verification: AutomaticSaveVerification

        var folderName: String { folder.folderName }
    }

    /// The outcome of one retention sweep. Pure data so the decision can
    /// be tested without a filesystem.
    struct AutomaticSavePrunePlan: Sendable, Equatable {
        /// Verified folders inside the newest `keep`, newest first.
        var keptWithinCap: [AutomaticSaveCandidate] = []
        /// Verified folders beyond the cap that are protected, newest
        /// first.
        var keptProtected: [AutomaticSaveCandidate] = []
        /// Verified, unprotected folders beyond the cap, newest first.
        var delete: [AutomaticSaveCandidate] = []
        /// Name-matched folders that did not verify, in input order.
        /// Never counted in the pool.
        var keptUnverified: [AutomaticSaveCandidate] = []

        /// How many verified folders of `kind` the pool held — the
        /// pool size the `[PRUNE]` summary reports per kind.
        func verifiedCount(of kind: AutomaticSaveKind) -> Int {
            (keptWithinCap + keptProtected + delete).filter { $0.folder.kind == kind }.count
        }
    }

    /// Decide which automatic-save folders to delete. Only verified
    /// folders form the pool; periodic and promotion saves of every
    /// session are ranked together, newest first by folder name, which
    /// is chronological because every name starts with its fixed-width
    /// UTC `YYYYMMDD-HHMMSS` timestamp — so the trigger and the session
    /// never affect the order. A protected folder holds its rank:
    /// inside the cap it is simply kept, beyond it it is kept without
    /// pulling any older folder back under the cap. `keep <= 0` means
    /// unlimited and deletes nothing.
    static func planAutomaticSavePrune(
        candidates: [AutomaticSaveCandidate],
        keep: Int,
        protectedFolderNames: Set<String>
    ) -> AutomaticSavePrunePlan {
        var plan = AutomaticSavePrunePlan()
        var verified: [AutomaticSaveCandidate] = []
        for candidate in candidates {
            if candidate.verification.isVerified {
                verified.append(candidate)
            } else {
                plan.keptUnverified.append(candidate)
            }
        }
        verified.sort { $0.folderName > $1.folderName }
        for (rank, candidate) in verified.enumerated() {
            if keep <= 0 || rank < keep {
                plan.keptWithinCap.append(candidate)
            } else if protectedFolderNames.contains(candidate.folderName) {
                plan.keptProtected.append(candidate)
            } else {
                plan.delete.append(candidate)
            }
        }
        return plan
    }

    /// Enforce the automatic-save retention cap over `directory`: keep
    /// the `keep` most recent automatic saves — periodic and promotion
    /// saves together, from every session that ever wrote one there —
    /// and delete the older ones. `keep <= 0` means unlimited: it logs
    /// that and returns without listing the folder.
    ///
    /// The pool is global by owner decision (2026-10-01). It replaced
    /// `prunePeriodicAutosaves`, which already spanned every session
    /// but covered periodic saves only: after each successful periodic
    /// save it listed `Sessions/`, took every non-hidden entry whose
    /// name ended in `-periodic.dcmsession` — by suffix alone, whatever
    /// its type, with no `session.json` check and no session-ID check —
    /// ranked them by name, and removed every one beyond the newest
    /// `keep` by path (`FileManager.removeItem`, recursive), sparing
    /// only the save just written; the resume-pointer target had no
    /// protection. Promotion saves were never pruned, so `Sessions/`
    /// grew without bound over a long campaign. One cap over
    /// everything automatic bounds the disk the autosaves can take,
    /// whatever the number of runs and promotions.
    ///
    /// What stays out of the pool, or is never deleted from it:
    /// - every folder whose name is not exactly an automatic-save name
    ///   (`parseAutomaticSaveFolderName`) — `-manual` and `-sigusr2`
    ///   saves, staging folders, anything renamed or hand-made;
    /// - every folder that does not verify
    ///   (`inspectAutomaticSaveFolder`): a placeholder session ID in
    ///   the name, a symbolic link, not a directory, no regular
    ///   `session.json`, or a `session.json` naming another session.
    ///   These are kept, logged with the reason, and not counted;
    /// - `justWritten` and the current `LastSessionPointer` target,
    ///   read at sweep time — a resume must always find the folder it
    ///   would reach for, even when a different save wrote the pointer.
    ///   Protection compares folder names, not full paths, so a pointer
    ///   spelled through a different path to the same folder still
    ///   protects it.
    ///
    /// A deletion goes through `FileSafety.removeOwnedItem` with the
    /// identity recorded during inspection, so a folder swapped in at
    /// that path between inspection and deletion is refused, not
    /// removed.
    ///
    /// Logs one `[PRUNE] retention:` summary per sweep, a `[PRUNE]`
    /// line for every kept-for-a-reason folder and every removal (with
    /// its session ID and trigger), and `[PRUNE-ERR]` for a failure,
    /// which does not abort the sweep — except an unreadable resume
    /// pointer, which stops it before anything is listed (its target could
    /// not be protected). Safe to call off the main actor
    /// — file system and UserDefaults work only.
    static func pruneAutomaticSaves(
        keeping keep: Int,
        protecting justWritten: URL,
        in directory: URL = CheckpointPaths.sessionsDir,
        lastSessionPointerDefaults: UserDefaults = .standard
    ) {
        guard keep > 0 else {
            SessionLogger.shared.log(
                "[PRUNE] retention: cap=\(keep) (unlimited) — nothing pruned after \(justWritten.lastPathComponent)"
            )
            return
        }
        // The resume pointer's target is protected below. A pointer that is
        // stored but unreadable must not be taken for "no pointer", which
        // would leave that target deletable: stop instead. Every session save
        // rewrites the pointer, so pruning resumes once a save succeeds in
        // rewriting it.
        let resumePointer: LastSessionPointer?
        do {
            resumePointer = try LastSessionPointer.stored(in: lastSessionPointerDefaults)
        } catch {
            SessionLogger.shared.log(
                "[PRUNE-ERR] retention sweep after \(justWritten.lastPathComponent) aborted, nothing pruned: \(error.localizedDescription); the resume target cannot be protected. Pruning resumes after a session save rewrites the pointer."
            )
            return
        }
        let entries: [URL]
        do {
            entries = try FileManager.default.contentsOfDirectory(
                at: directory,
                includingPropertiesForKeys: nil,
                options: [.skipsHiddenFiles]
            )
        } catch {
            SessionLogger.shared.log(
                "[PRUNE-ERR] Could not list \(directory.lastPathComponent) for automatic-save retention: \(error.localizedDescription)"
            )
            return
        }
        var urlsByName: [String: URL] = [:]
        var candidates: [AutomaticSaveCandidate] = []
        for entry in entries {
            guard let parsed = parseAutomaticSaveFolderName(entry.lastPathComponent) else { continue }
            urlsByName[parsed.folderName] = entry
            candidates.append(AutomaticSaveCandidate(
                folder: parsed,
                verification: inspectAutomaticSaveFolder(at: entry, parsed: parsed)
            ))
        }

        var protectionReasons: [String: [String]] = [:]
        protectionReasons[justWritten.standardizedFileURL.lastPathComponent, default: []].append("the save just written")
        if let pointer = resumePointer {
            protectionReasons[pointer.directoryURL.standardizedFileURL.lastPathComponent, default: []]
                .append("the resume pointer's target")
        }

        let plan = planAutomaticSavePrune(
            candidates: candidates,
            keep: keep,
            protectedFolderNames: Set(protectionReasons.keys)
        )
        SessionLogger.shared.log(
            "[PRUNE] retention: cap=\(keep) periodic=\(plan.verifiedCount(of: .periodic)) promote=\(plan.verifiedCount(of: .promotion)) kept=\(plan.keptWithinCap.count) protected=\(plan.keptProtected.count) unverified=\(plan.keptUnverified.count) deleting=\(plan.delete.count)"
        )
        for candidate in plan.keptUnverified {
            SessionLogger.shared.log(
                "[PRUNE] Kept \(candidate.folderName) (session=\(candidate.folder.sessionID) trigger=\(candidate.folder.kind.diskTag)), not counted: \(candidate.verification.keptReason)"
            )
        }
        for candidate in plan.keptProtected {
            let reasons = protectionReasons[candidate.folderName, default: []].joined(separator: " and ")
            SessionLogger.shared.log(
                "[PRUNE] Kept \(candidate.folderName) (session=\(candidate.folder.sessionID) trigger=\(candidate.folder.kind.diskTag)) beyond the cap: it is \(reasons)"
            )
        }
        for candidate in plan.delete {
            let name = candidate.folderName
            let label = "session=\(candidate.folder.sessionID) trigger=\(candidate.folder.kind.diskTag)"
            guard case .verified(let identity) = candidate.verification, let url = urlsByName[name] else {
                SessionLogger.shared.log("[PRUNE-ERR] Planned deletion \(name) (\(label)) has no verified listed folder; skipped")
                continue
            }
            do {
                switch try FileSafety.removeOwnedItem(at: url, identity: identity) {
                case .removed:
                    SessionLogger.shared.log("[PRUNE] Removed \(name) (\(label) retention cap=\(keep))")
                case .alreadyGone:
                    SessionLogger.shared.log("[PRUNE] \(name) (\(label)) was already gone when the sweep went to remove it")
                }
            } catch FileSafetyError.fileChangedSinceWritten {
                SessionLogger.shared.log(
                    "[PRUNE-ERR] Did not remove \(name) (\(label)): the item at that path is no longer the folder that was inspected"
                )
            } catch {
                SessionLogger.shared.log(
                    "[PRUNE-ERR] Could not remove \(name) (\(label)): \(error.localizedDescription)"
                )
            }
        }
    }

    // MARK: Filenames

    /// `DateFormatter` pattern of the timestamp that leads every
    /// generated file and folder name.
    private static let filenameTimestampFormat = "yyyyMMdd-HHmmss"

    /// POSIX/UTC timestamp formatter used as the leading sort key in
    /// every generated filename so Finder's alphabetical order is
    /// also chronological order.
    private static let filenameTimestampFormatter: DateFormatter = {
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.timeZone = TimeZone(identifier: "UTC")
        f.dateFormat = filenameTimestampFormat
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
    /// URLs of the optional chart-companion JSON files, when
    /// `state.hasChartData == true` and both files exist on disk.
    /// `nil` for older sessions, sessions saved with chart
    /// collection disabled, or sessions whose chart files were
    /// removed externally. Loader callers JSON-decode them via
    /// `readChartFile(_:from:)` (see `ChartFileFormat.swift`) and
    /// then hand the arrays to `ChartCoordinator.seedFromRestoredSession`.
    let chartDataURLs: (training: URL, progressRate: URL)?
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
    /// Count of `TrainingChartSample`s saved with the session.
    /// nil for older sessions or sessions saved with chart
    /// collection disabled. Lets the resume sheet surface
    /// "12,453 chart samples" so the user knows up front whether
    /// the chart trajectory will come back on resume.
    let trainingChartSampleCount: Int?
    let arenaCount: Int
    let promotionCount: Int
    /// Champion `ModelID` description (`yyyymmdd-N-XXXX`) at save time —
    /// the live champion the session would resume into. Surfaced in the
    /// resume sheet's Models block so the user can verify the lineage
    /// before confirming the resume.
    let championID: String
    /// Trainer `ModelID` description at save time. Distinct from
    /// `championID` between promotions (every promotion forks a fresh
    /// next-generation trainer ID off the promoted champion).
    let trainerID: String
    /// Architecture snapshot persisted in `session.json`. nil on
    /// sessions saved before the field landed; the resume sheet
    /// renders "unknown" in that case.
    let architecture: ArchitectureMetadata?
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
        self.trainingChartSampleCount = state.trainingChartSampleCount
        self.arenaCount = state.arenaHistory.count
        self.promotionCount = state.arenaHistory.lazy.filter { $0.promoted }.count
        self.championID = state.championID
        self.trainerID = state.trainerID
        self.architecture = state.architecture
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
        do {
            try FileSafety.fullSync(at: url)
        } catch {
            throw CheckpointManagerError.fsyncFailed(url, error)
        }
    }

    /// Verify a freshly-restored replay buffer's lifetime counter
    /// matches what `session.json` said it should be. Used at session
    /// load time as a defense-in-depth cross-check after the buffer's
    /// own SHA and size guards have already succeeded. It runs after the
    /// restore has filled the buffer; the GUI resume instead checks the
    /// same count before the restore mutates anything
    /// (`ReplayBuffer.restore(from:expectedTotalPositionsAdded:)`).
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
    /// The file is staged at `<final>.tmp`, which this call creates
    /// exclusively (`createStagingFileExclusively`) before doing any
    /// other work: if anything already occupies that path the save
    /// fails with `.stagingPathAlreadyExists` and leaves it untouched.
    /// Every failure after the claim removes the staging file — it is
    /// provably this call's own — and a successful rename hands it
    /// over, after which nothing here touches that path again.
    ///
    /// `modelsDirectory` is the canonical `Models/` folder in the app;
    /// tests pass a temporary folder.
    ///
    /// Runs synchronously in the caller's task context. Callers
    /// should invoke via `Task.detached` to keep MPSGraph work off
    /// the main actor.
    static func saveModel(
        weights: [[Float]],
        modelID: String,
        createdAtUnix: Int64,
        metadata: ModelCheckpointMetadata,
        architecture: NetworkArchitecture = .current,
        lineage: LineageRecord,
        trigger: String,
        at date: Date = Date(),
        modelsDirectory: URL = CheckpointPaths.modelsDir
    ) async throws -> URL {
        try CheckpointPaths.ensureDirectory(modelsDirectory)

        let filename = CheckpointPaths.makeFilename(
            modelID: modelID,
            trigger: trigger,
            ext: "safetensors",
            at: date
        )
        let finalURL = modelsDirectory.appendingPathComponent(filename)
        let tmpURL = finalURL.appendingPathExtension(CheckpointPaths.stagingPathExtension)

        if FileManager.default.fileExists(atPath: finalURL.path) {
            // Never overwrite. Timestamp-to-the-second collisions are
            // extraordinarily unlikely but refuse cleanly if it ever
            // happens rather than silently stomping prior history.
            throw CheckpointManagerError.targetAlreadyExists(finalURL)
        }

        let staging = try CheckpointPaths.createStagingFileExclusively(at: tmpURL)
        // From here until the rename succeeds, `tmpURL` is this call's
        // own, and every exit removes it. After the rename the flag is
        // cleared: the path is free again, and whatever appears there
        // later belongs to someone else.
        var ownsStagingFile = true
        defer {
            if ownsStagingFile {
                CheckpointPaths.removeOwnedStagingItem(at: tmpURL, identity: staging.identity)
            }
        }

        let encoded = try SafetensorsModelIO.encode(
            modelID: modelID,
            createdAtUnix: createdAtUnix,
            metadata: metadata,
            weights: weights,
            architecture: architecture,
            includesVelocity: false,
            lineage: lineage
        )

        do {
            try staging.handle.write(contentsOf: encoded)
            try staging.handle.close()
        } catch {
            throw CheckpointManagerError.writeFailed(tmpURL, error)
        }

        // Flush tmp file to platter before verify + rename so a
        // crash after verify-returns can't leave a torn file behind.
        try fullSyncPath(tmpURL)

        // Verify BEFORE the rename so a failed check leaves nothing
        // with the final name.
        try await verifyModelFile(at: tmpURL, expectedWeights: weights, architecture: architecture)

        // Exclusive rename: the existence check at the top is only a
        // fast path, and this refuses atomically if anything took the
        // final name since.
        do {
            try FileSafety.renameWithoutReplacing(from: tmpURL, to: finalURL)
        } catch FileSafetyError.alreadyExists {
            throw CheckpointManagerError.targetAlreadyExists(finalURL)
        } catch {
            throw CheckpointManagerError.writeFailed(finalURL, error)
        }
        ownsStagingFile = false

        // Flush parent directory so the rename (directory-entry
        // change) is durable.
        do {
            try fullSyncPath(modelsDirectory)
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
    ///
    /// Everything is staged in `<final>.tmp/`, which this call creates
    /// exclusively (`createStagingDirectoryExclusively`) before any
    /// other filesystem work: if anything already occupies that path
    /// the save fails with `.stagingPathAlreadyExists` and leaves it
    /// untouched. Every failure after the claim removes the staging
    /// directory — it is provably this call's own — and a successful
    /// rename hands it over.
    ///
    /// `lineage` is the run's record at this save, written into the
    /// trainer file and session.json. `championLineage` is the record of
    /// the weights the champion file holds, which are not the trainer's
    /// whenever training has moved on since the champion was built, loaded
    /// or promoted (`SessionController.championFileLineageRecord`); it
    /// carries no trainer state. A writer whose champion is the trainer's
    /// weights at the save passes the run's record without its trainer
    /// state.
    ///
    /// `onReplayBufferWritten` runs as soon as the replay buffer file is
    /// written (and before the save's verification re-reads it), for a
    /// caller that holds self-play paused so the written buffer is the one
    /// its record describes and can let self-play go as early as possible;
    /// it does not run when the save writes no buffer or fails before the
    /// write completes.
    ///
    /// `sessionsDirectory` is the canonical `Sessions/` folder in the
    /// app; tests pass a temporary folder.
    static func saveSession(
        championWeights: [[Float]],
        championID: String,
        championMetadata: ModelCheckpointMetadata,
        championCreatedAtUnix: Int64,
        trainerWeights: [[Float]],
        trainerID: String,
        trainerMetadata: ModelCheckpointMetadata,
        trainerCreatedAtUnix: Int64,
        state stateWithoutLineage: SessionCheckpointState,
        lineage: LineageRecord,
        championLineage: LineageRecord,
        architecture: NetworkArchitecture = .current,
        replayBuffer: ReplayBuffer? = nil,
        chartSnapshot: ChartCoordinatorSnapshot? = nil,
        trigger: String,
        at date: Date = Date(),
        sessionsDirectory: URL = CheckpointPaths.sessionsDir,
        onReplayBufferWritten: (@Sendable () -> Void)? = nil
    ) async throws -> URL {
        // The trainer file and session.json carry the run's record; the
        // champion file carries its own weights' record.
        let state = stateWithoutLineage.withLineage(lineage)
        try CheckpointPaths.ensureDirectory(sessionsDirectory)

        let dirName = CheckpointPaths.makeSessionDirectoryName(
            sessionID: state.sessionID,
            trigger: trigger,
            at: date
        )
        let finalDirURL = sessionsDirectory.appendingPathComponent(dirName, isDirectory: true)
        let tmpDirURL = sessionsDirectory.appendingPathComponent(
            dirName + "." + CheckpointPaths.stagingPathExtension,
            isDirectory: true
        )

        if FileManager.default.fileExists(atPath: finalDirURL.path) {
            throw CheckpointManagerError.targetAlreadyExists(finalDirURL)
        }

        guard trainerWeights.count > championWeights.count else {
            throw CheckpointManagerError.sessionWriteFailed(
                "trainer.dcmmodel must contain full trainer state, including optimizer velocity; got \(trainerWeights.count) tensors for trainer and \(championWeights.count) tensors for champion"
            )
        }

        let stagingDirectoryIdentity = try CheckpointPaths.createStagingDirectoryExclusively(at: tmpDirURL)

        // Single deferred cleanup that covers every throw site between
        // here and the successful rename `tmpDirURL → finalDirURL`.
        // Manual cleanup calls at every error branch were easy to miss —
        // and a missed branch leaks a multi-GB staging directory
        // (champion + trainer + replay buffer) under `Sessions/*.tmp`.
        //
        // The cleanup is gated on ownership, not on the path existing:
        // the exclusive create above proves the directory is this
        // call's, and the flag is cleared the moment the rename hands
        // it over. An existence check alone would also delete a
        // directory this call never made, such as debris from an
        // interrupted save or another instance's live staging.
        var ownsStagingDirectory = true
        defer {
            if ownsStagingDirectory {
                CheckpointPaths.removeOwnedStagingItem(at: tmpDirURL, identity: stagingDirectoryIdentity)
            }
        }

        // Encode the model files before writing anything into the
        // staging directory; an encoding failure exits through the
        // cleanup above. session.json is encoded LATER — after the
        // replay buffer has been written — so its
        // `replayBuffer*` counters can be derived from the snapshot
        // the buffer write captured atomically under its own lock,
        // rather than from a stale snapshot the caller took before
        // self-play paused. See the writtenSnap section below.
        let championEncoded = try SafetensorsModelIO.encode(
            modelID: championID,
            createdAtUnix: championCreatedAtUnix,
            metadata: championMetadata,
            weights: championWeights,
            architecture: architecture,
            includesVelocity: false,
            lineage: championLineage
        )

        // Trainer file = base weights (trainables + BN running stats) followed by
        // optimizer velocity (one per trainable, trainable order) — named
        // opt.<trainable>.velocity by SafetensorsModelIO.
        let trainerEncoded = try SafetensorsModelIO.encode(
            modelID: trainerID,
            createdAtUnix: trainerCreatedAtUnix,
            metadata: trainerMetadata,
            weights: trainerWeights,
            architecture: architecture,
            includesVelocity: true,
            lineage: lineage
        )

        let championTmpURL = SessionCheckpointLayout.championURL(in: tmpDirURL)
        let trainerTmpURL = SessionCheckpointLayout.trainerURL(in: tmpDirURL)
        let stateTmpURL = SessionCheckpointLayout.stateURL(in: tmpDirURL)
        let bufferTmpURL = SessionCheckpointLayout.replayBufferURL(in: tmpDirURL)
        let trainingChartTmpURL = SessionCheckpointLayout.trainingChartURL(in: tmpDirURL)
        let progressRateChartTmpURL = SessionCheckpointLayout.progressRateChartURL(in: tmpDirURL)
        let wantsReplayBuffer = replayBuffer != nil && state.hasReplayBuffer == true
        // Chart files are only written when the caller passed a
        // snapshot AND that snapshot has at least one ring populated.
        // `ChartCoordinator.buildSnapshot()` already returns nil when
        // collection is off or the rings are empty, so this flag
        // typically tracks "the caller wanted to save chart data and
        // had something to save." Older sessions that never saw a
        // snapshot leave the flag false, the chart files absent, and
        // `state.hasChartData` nil — load-time code falls back to the
        // existing behavior (chart pane starts fresh).
        let wantsChartData = chartSnapshot != nil

        do {
            try championEncoded.write(to: championTmpURL, options: [.atomic])
            try trainerEncoded.write(to: trainerTmpURL, options: [.atomic])
        } catch {
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
        // was actually serialized into the file, captured atomically
        // under the buffer's write lock. We use it twice below:
        //
        // 1. To override the caller-supplied `state.replayBuffer*`
        //    counters before encoding session.json. The caller
        //    captures those fields BEFORE self-play is paused (and
        //    well before this function runs); self-play continues
        //    appending positions in that window, so a separate
        //    `stateSnapshot()` would diverge from the file we just
        //    wrote. ReplayBuffer.write's contract specifically
        //    requires using its return value for downstream
        //    consistency checks for exactly this reason. Without
        //    this override, restore-time `verifyReplayBufferMatchesSession`
        //    would (correctly) reject the session as mismatched.
        //
        // 2. As ground truth for the post-fsync round-trip verify
        //    further down — concurrent self-play appends may have
        //    already advanced the live ring past that state by then.
        var writtenSnap: ReplayBuffer.StateSnapshot? = nil
        if let replayBuffer, wantsReplayBuffer {
            do {
                writtenSnap = try replayBuffer.write(to: bufferTmpURL)
            } catch {
                throw CheckpointManagerError.writeFailed(bufferTmpURL, error)
            }
            onReplayBufferWritten?()
        }

        // Optional chart-data dump. Two plain-JSON files
        // (`training_chart.json`, `progress_rate_chart.json`) carry
        // the two `ChartCoordinator` rings + the small auxiliary
        // chart state needed to restore a chart trajectory across
        // save/resume. Inline JSON for the `arenaChartEvents` and
        // `legalMassMaxAllTime` fields rides in session.json
        // alongside everything else (small enough not to need a side
        // file). Skipped entirely when `chartSnapshot` is nil — older
        // sessions, sessions saved with chart collection off, and
        // CLI runs without a heartbeat all land here.
        if let chartSnapshot {
            do {
                try writeChartFile(
                    chartSnapshot.trainingSamples,
                    to: trainingChartTmpURL
                )
                try writeChartFile(
                    chartSnapshot.progressRateSamples,
                    to: progressRateChartTmpURL
                )
            } catch {
                throw CheckpointManagerError.chartFileWriteFailed(tmpDirURL, error)
            }
        }

        // Now encode and write session.json. If we wrote a buffer,
        // its post-write snapshot supersedes the caller's pre-pause
        // numbers so the two files match bit-for-bit on the
        // `totalPositionsAdded` cross-check at load time. Chart
        // counts go in here too — the values come from the snapshot
        // we just serialized, so they're guaranteed to match the
        // file contents.
        var effectiveState = state
        if let writtenSnap {
            effectiveState.replayBufferStoredCount = writtenSnap.storedCount
            effectiveState.replayBufferCapacity = writtenSnap.capacity
            effectiveState.replayBufferTotalPositionsAdded = writtenSnap.totalPositionsAdded
        }
        if let chartSnapshot {
            effectiveState = effectiveState.withChartData(
                hasChartData: true,
                trainingChartSampleCount: chartSnapshot.trainingSamples.count,
                progressRateSampleCount: chartSnapshot.progressRateSamples.count,
                arenaChartEvents: chartSnapshot.arenaChartEvents,
                legalMassMaxAllTime: chartSnapshot.legalMassMaxAllTime
            )
        }
        let stateEncoded: Data
        do {
            stateEncoded = try effectiveState.encode()
        } catch {
            throw error
        }
        do {
            try stateEncoded.write(to: stateTmpURL, options: [.atomic])
        } catch {
            throw CheckpointManagerError.writeFailed(tmpDirURL, error)
        }

        // manifest.json — the Load Session picker's at-a-glance summary,
        // derived from the EXACT session.json bytes just written (single
        // extraction code path; cannot drift from the file). A manifest
        // failure must not abort the save: the manifest is regenerable
        // derived data (the picker's indexer rebuilds it from
        // session.json), the session files are the artifact. Log loudly
        // and continue.
        do {
            let manifestData = try SessionManifest.makeManifestData(
                sessionJSON: stateEncoded,
                folderName: dirName,
                sessionDirURL: tmpDirURL,
                sessionJSONURL: stateTmpURL
            )
            try manifestData.write(
                to: tmpDirURL.appendingPathComponent("manifest.json"),
                options: [.atomic]
            )
        } catch {
            SessionLogger.shared.log(
                "[CHECKPOINT] manifest.json write FAILED (session save continues; picker will index from session.json): \(error.localizedDescription)"
            )
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
            if wantsChartData {
                try fullSyncPath(trainingChartTmpURL)
                try fullSyncPath(progressRateChartTmpURL)
            }
        } catch {
            throw error
        }

        do {
            try await verifyModelFile(at: championTmpURL, expectedWeights: championWeights, architecture: architecture)
            // Trainer file is v2 layout: trainables + bn + velocity.
            // The inference scratch network used by `verifyModelFile`
            // can only load trainables + bn, so the forward-pass
            // round-trip is scoped to that base prefix. The base
            // length is exactly the champion file's tensor count
            // (champion is the same architecture, base-only). The
            // bit-compare step inside `verifyModelFile` still
            // covers the full v2 payload, so velocity tensor
            // integrity on disk is checked unchanged.
            try await verifyModelFile(
                at: trainerTmpURL,
                expectedWeights: trainerWeights,
                architecture: architecture,
                forwardPassPrefixCount: championWeights.count
            )
            // Round-trip session.json: decode the bytes we just wrote
            // and confirm they reproduce the struct. Catches JSON
            // encoder/decoder asymmetries (e.g. Float precision).
            // Compare against `effectiveState` (the struct we actually
            // serialized), not the caller-supplied `state` — the two
            // differ when a replay-buffer save updated the
            // `replayBuffer*` counters from the post-write snapshot.
            let writtenStateData = try Data(contentsOf: stateTmpURL)
            let writtenState = try SessionCheckpointState.decode(writtenStateData)
            guard writtenState == effectiveState else {
                throw CheckpointManagerError.sessionWriteFailed(
                    "session.json round-trip decoded to a different struct"
                )
            }
            // Replay-buffer verification: re-load the file we just
            // wrote into a scratch ReplayBuffer. The scratch restore
            // runs the full current-version validation stack — magic, version,
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
                let scratch: ReplayBuffer
                do {
                    // Match the just-written file's per-position stride (per
                    // architecture — e.g. basic20 = 1280) so the verify
                    // restore doesn't reject it on a stride mismatch.
                    let scratchFloatsPerBoard = try ReplayBuffer.peekFloatsPerBoard(at: bufferTmpURL)
                    // Restored and compared, never sampled.
                    scratch = ReplayBuffer(capacity: scratchCapacity, floatsPerBoard: scratchFloatsPerBoard, sampler: DCMRandom.seededFromSystem())
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
            // Chart-file verification: re-read each file and confirm
            // the count of decoded samples matches what we just
            // wrote. Element-wise equality is intentionally NOT
            // checked here — `Double.nan == Double.nan` is `false`
            // per IEEE 754, and `gNorm` legitimately becomes NaN
            // when the network diverges, so an element-wise check
            // would flag healthy NaN-bearing samples as round-trip
            // failures. Bit-pattern equality is verified in unit
            // tests instead.
            if wantsChartData, let chartSnapshot {
                do {
                    let restoredTraining = try readChartFile(
                        [TrainingChartSample].self, from: trainingChartTmpURL
                    )
                    guard restoredTraining.count == chartSnapshot.trainingSamples.count else {
                        throw CheckpointManagerError.chartFileVerifyFailed(
                            detail: "training_chart.json count round-trip mismatch: written=\(chartSnapshot.trainingSamples.count) scratch=\(restoredTraining.count)"
                        )
                    }
                    let restoredProgress = try readChartFile(
                        [ProgressRateSample].self, from: progressRateChartTmpURL
                    )
                    guard restoredProgress.count == chartSnapshot.progressRateSamples.count else {
                        throw CheckpointManagerError.chartFileVerifyFailed(
                            detail: "progress_rate_chart.json count round-trip mismatch: written=\(chartSnapshot.progressRateSamples.count) scratch=\(restoredProgress.count)"
                        )
                    }
                } catch let err as CheckpointManagerError {
                    throw err
                } catch {
                    throw CheckpointManagerError.chartFileVerifyFailed(
                        detail: "scratch read failed: \(error.localizedDescription)"
                    )
                }
            }
        } catch {
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
            throw error
        }

        // Exclusive rename, as in `saveModel`: refuses atomically if
        // anything took the final name since the check at the top.
        do {
            try FileSafety.renameWithoutReplacing(from: tmpDirURL, to: finalDirURL)
        } catch FileSafetyError.alreadyExists {
            throw CheckpointManagerError.targetAlreadyExists(finalDirURL)
        } catch {
            throw CheckpointManagerError.writeFailed(finalDirURL, error)
        }
        ownsStagingDirectory = false

        // Flush the parent `Sessions/` directory so the rename itself
        // (which is a directory-entry change in the parent) lands on
        // stable storage. If we skip this and the machine loses power
        // before the parent's directory block is flushed, the session
        // can disappear entirely even though the files inside it are
        // fully durable.
        do {
            try fullSyncPath(sessionsDirectory)
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
    /// Decode a model file of either format: native safetensors (current) or
    /// the legacy custom `.dcmmodel` binary, detected by its `DCMMODEL` magic.
    /// Existing models stay loadable; new saves are safetensors.
    static func decodeAnyModelFile(_ data: Data) throws -> ModelCheckpointFile {
        try decodeAnyModelFile(data, valueHead: .recenterUnlessMarked)
    }

    static func decodeAnyModelFile(_ data: Data, valueHead: ValueHeadDecoding) throws -> ModelCheckpointFile {
        try decodeAnyModelFile(data, valueHead: valueHead, source: SafetensorsModelIO.unnamedSource)
    }

    /// `decodeAnyModelFile(_:valueHead:)` for bytes read from `source` — the
    /// file name that format-version errors and the legacy-resolution log
    /// line report.
    static func decodeAnyModelFile(_ data: Data, valueHead: ValueHeadDecoding, source: String) throws -> ModelCheckpointFile {
        if data.count >= 8, Array(data.prefix(8)) == ModelCheckpointFile.magic {
            return try ModelCheckpointFile.decode(data, valueHead: valueHead)
        }
        return try SafetensorsModelIO.decode(data, valueHead: valueHead, source: source).file
    }

    /// A model file exactly as stored, for the numerics audit: the value
    /// head is not recentered, so the audit sees the offset the file holds.
    /// Never used to play or train.
    static func loadModelFileAsStored(at url: URL) throws -> ModelCheckpointFile {
        let data: Data
        do {
            data = try Data(contentsOf: url)
        } catch {
            throw CheckpointManagerError.readFailed(url, error)
        }
        let file = try decodeAnyModelFile(data, valueHead: .asStored, source: url.lastPathComponent)
        file.architectureFormat?.logLegacyResolutions()
        return file
    }

    static func loadModelFile(at url: URL) throws -> ModelCheckpointFile {
        try loadModelFile(fromBytes: try readModelFileBytes(at: url), source: url.lastPathComponent)
    }

    /// A model file's bytes, a failed read reported as `readFailed` with
    /// the file's URL.
    static func readModelFileBytes(at url: URL) throws -> Data {
        do {
            return try Data(contentsOf: url)
        } catch {
            throw CheckpointManagerError.readFailed(url, error)
        }
    }

    /// `loadModelFile(at:)` over bytes already read, so a caller that also
    /// needs the bytes (to hash exactly what was decoded) reads the file
    /// once. `source` names the file in errors and log lines.
    static func loadModelFile(fromBytes data: Data, source: String) throws -> ModelCheckpointFile {
        let file = try decodeAnyModelFile(data, valueHead: .recenterUnlessMarked, source: source)
        file.architectureFormat?.logLegacyResolutions()
        logValueHeadCentering(file, source: source)
        return file
    }

    /// Write the one `[NUMERICS]` line for a file whose value head decode
    /// recentered (see `ValueHeadRecentering`); silent otherwise. Called by
    /// the loaders that know the file's name, once per load — not by
    /// `decodeAnyModelFile`, which the post-save verification also runs.
    static func logValueHeadCentering(_ file: ModelCheckpointFile, source: String) {
        // Every decoded file carries a centering result; only files built in
        // memory for a save have none, and those are never loaded here.
        guard let centering = file.valueHeadCentering else {
            preconditionFailure("a decoded model file (\(source)) has no value-head centering result")
        }
        guard let line = ValueHeadRecentering.logLine(for: centering, source: source) else { return }
        SessionLogger.shared.log(line)
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
        let championFile = try decodeAnyModelFile(
            championData, valueHead: .recenterUnlessMarked,
            source: "\(directoryURL.lastPathComponent) champion")
        let trainerFile = try decodeAnyModelFile(
            trainerData, valueHead: .recenterUnlessMarked,
            source: "\(directoryURL.lastPathComponent) trainer")
        championFile.architectureFormat?.logLegacyResolutions()
        trainerFile.architectureFormat?.logLegacyResolutions()
        logValueHeadCentering(championFile, source: "\(directoryURL.lastPathComponent) champion")
        logValueHeadCentering(trainerFile, source: "\(directoryURL.lastPathComponent) trainer")
        let bufferURL = SessionCheckpointLayout.replayBufferURL(in: directoryURL)
        let bufferPresent = (state.hasReplayBuffer == true)
            && FileManager.default.fileExists(atPath: bufferURL.path)
        let trainingChartURL = SessionCheckpointLayout.trainingChartURL(in: directoryURL)
        let progressRateChartURL = SessionCheckpointLayout.progressRateChartURL(in: directoryURL)
        let chartFilesPresent = (state.hasChartData == true)
            && FileManager.default.fileExists(atPath: trainingChartURL.path)
            && FileManager.default.fileExists(atPath: progressRateChartURL.path)
        return LoadedSession(
            directoryURL: directoryURL,
            state: state,
            championFile: championFile,
            trainerFile: trainerFile,
            replayBufferURL: bufferPresent ? bufferURL : nil,
            chartDataURLs: chartFilesPresent
                ? (training: trainingChartURL, progressRate: progressRateChartURL)
                : nil
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
    /// - parameter forwardPassPrefixCount: When non-nil, the forward-
    ///   pass round-trip step loads only the first `forwardPassPrefixCount`
    ///   tensors of `expectedWeights` (and of the readback) into a
    ///   throwaway inference scratch network. Used for the trainer
    ///   `.dcmmodel` v2 layout: the file carries trainables + bn +
    ///   velocity (= total 2·trainables + bn), but the inference
    ///   scratch network only has trainables + bn slots, so a full
    ///   payload load would throw "Weight load mismatch: expected N
    ///   tensors, got 2·trainables + bn." When nil (default), the
    ///   full payload is loaded — appropriate for the champion file
    ///   which contains only trainables + bn. The bit-compare step
    ///   above always covers the full payload regardless of this
    ///   parameter, so velocity tensor integrity is still verified
    ///   on disk; only the forward-pass behavioral check is scoped
    ///   to the inference-loadable subset.
    static func verifyModelFile(
        at url: URL,
        expectedWeights: [[Float]],
        architecture: NetworkArchitecture = .current,
        forwardPassPrefixCount: Int? = nil
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
            readBack = try decodeAnyModelFile(data, valueHead: .recenterUnlessMarked, source: url.lastPathComponent)
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
            scratch = try ChessMPSNetwork(.overwrittenByLoad, arch: architecture)
            scratch.network.commandQueue.label = "verifyModelFile scratch"
        } catch {
            throw CheckpointManagerError.verificationScratchBuildFailed(error)
        }

        let testBoard = BoardEncoder.encode(.starting, encoding: architecture.inputEncoding)

        // For v2 trainer files the scratch inference network can
        // only load the base prefix (trainables + bn), not the
        // velocity tail — slice both payloads identically before
        // the round-trip so they compare apples to apples.
        let prePayload: [[Float]]
        let postPayload: [[Float]]
        if let prefix = forwardPassPrefixCount {
            guard prefix >= 0,
                  prefix <= expectedWeights.count,
                  prefix <= readBack.weights.count else {
                throw CheckpointManagerError.verificationTensorCountMismatch(
                    expected: prefix,
                    got: min(expectedWeights.count, readBack.weights.count)
                )
            }
            prePayload = Array(expectedWeights.prefix(prefix))
            postPayload = Array(readBack.weights.prefix(prefix))
        } else {
            prePayload = expectedWeights
            postPayload = readBack.weights
        }

        // `nonisolated(unsafe)` for the `var` so each can be mutated
        // from inside the `@Sendable` consume closure. Safe because
        // the await suspends this task for the closure window.
        nonisolated(unsafe) var preValue: Float = 0
        nonisolated(unsafe) var prePolicy: [Float] = []
        do {
            try await scratch.loadWeights(prePayload)
            try await scratch.evaluate(board: testBoard) { policyBuf, value in
                prePolicy = Array(policyBuf)
                preValue = value
            }
        } catch {
            throw CheckpointManagerError.verificationForwardPassFailed(error)
        }

        nonisolated(unsafe) var postValue: Float = 0
        nonisolated(unsafe) var postPolicy: [Float] = []
        do {
            try await scratch.loadWeights(postPayload)
            try await scratch.evaluate(board: testBoard) { policyBuf, value in
                postPolicy = Array(policyBuf)
                postValue = value
            }
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
