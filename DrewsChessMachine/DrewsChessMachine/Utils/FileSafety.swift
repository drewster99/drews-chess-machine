import Darwin
import Foundation

/// The one implementation of every low-level file operation that must never
/// touch something the caller does not own.
///
/// Each rule here was paid for. A writer that removed whatever sat at the
/// requested path before writing deleted a whole documentation folder when it
/// was handed a folder path. `Data.write(options: .atomic)` silently replaces
/// whatever exists, so step-enumerated training checkpoints written that way
/// let a resumed segment that reused an earlier segment's stem overwrite that
/// segment's checkpoints (step numbers restart in every run).
/// `FileManager.createFile(atPath:contents:)` empties an existing regular file
/// (a previous run's results, or another process's live session log), follows
/// a symbolic link and empties the link's target, and reports failure only
/// through a `Bool` that callers ignored — after which a nil `FileHandle`
/// dropped hours of output. And a save's failure path that deleted "its"
/// staging path by name deleted whatever was there, including debris from
/// another save or the live staging of a second app instance.
///
/// So creating and publishing are built on kernel-enforced primitives, with no
/// check-then-act step at all. Replacing and removing a file the caller owns
/// cannot be: no system call takes "only if it is still this inode", so those
/// check the item by path with `lstat` and then act on the path. Each
/// bullet below states the window that leaves.
///
/// - **Exclusive creation.** A new file is opened with `O_CREAT | O_EXCL |
///   O_NOFOLLOW | O_CLOEXEC`, a new directory with `mkdir`; both fail with
///   `EEXIST` when anything at all — file, folder, symbolic link, even a
///   dangling one — is already there, and never open, truncate or follow it.
///   No window.
/// - **Publishing without overwriting.** A complete new file is staged in a
///   per-call unique hidden sibling and moved into place with
///   `renamex_np(RENAME_EXCL)`, which atomically refuses when the destination
///   exists. That gives "never overwrite" and "never a torn file under the
///   final name" at once; `Data.WritingOptions.withoutOverwriting` cannot be
///   combined with `.atomic`, and a plain exclusive create leaves a torn file
///   at the final name if the process dies mid-write. No window.
/// - **Replacing only a regular file.** A replacement is staged the same way
///   and moved over the destination with plain `rename`. The destination's
///   type (and, when the caller passes the identity it recorded when it wrote
///   the file, its identity) is checked with `lstat` before staging and again
///   right after, so the staging write is not part of the window. The window
///   is the gap between that second `lstat` and the `rename`: a directory
///   swapped in there is still safe (`rename` fails with `EISDIR` rather than
///   touching it), but any other item swapped in there — a different regular
///   file, a symbolic link (the link itself, never its target), a FIFO —
///   would be replaced.
/// - **Truncating only a regular file**, and only on request: the path is
///   opened with `O_NOFOLLOW` (a symbolic link is an error, never followed)
///   and `O_NONBLOCK` (a FIFO with no reader is an error, not a hang inside
///   `open`), with no `O_TRUNC`. The descriptor's type is checked with
///   `fstat` first and only then is the file cut to zero with `ftruncate`, so
///   a non-regular item is never modified even transiently. No window: every
///   check is on the open descriptor. The cost is the one the other
///   operations avoid — the file is rewritten in place, so a crash mid-write
///   leaves it torn.
/// - **Removing only what the caller created**, proven by identity (device
///   and inode recorded at creation), never by name alone. A regular file is
///   checked with `lstat` and then `unlink`ed by path; any non-directory item
///   swapped in between those two calls would be the one unlinked. A
///   directory is never deleted by path: it is first moved to a private
///   hidden name with `renamex_np(RENAME_EXCL)`, the item now under that name
///   is checked, and only then is it deleted with its contents — whatever
///   was moved by mistake is moved back untouched. Its only exposure is that
///   it is briefly absent from its path, and, if moving a mistaken item back
///   fails (its path was taken again meanwhile), that item is left under the
///   hidden name and reported rather than deleted.
///
/// Staging leftovers: a process killed between staging a file and renaming
/// it into place leaves the hidden `.<name>.<UUID>.tmp` sibling behind
/// (`temporarySibling(of:)`), and one killed while deleting a directory can
/// leave that directory, partly emptied, under the same kind of name. Nothing
/// here can clean those up — a leftover is indistinguishable from another
/// process's write in flight. The app's launch sweep
/// (`CheckpointPaths.cleanupOrphans`) removes aged staging *files* of exactly
/// this shape from `Models/` and `Sessions/`; `CorpusValidator` reports any
/// `.tmp` in a corpus folder; anywhere else (Presets, a CLI's output
/// folder) a leftover stays until removed by hand.
///
/// Every refusal and every failed system call is a thrown `FileSafetyError`
/// naming the path and the reason; nothing returns an optional handle and
/// nothing is skipped silently. `FileHandle` write failures propagate
/// unchanged (they carry the real `errno`), so callers that classify disk-full
/// failures keep working; see `FileSafetyError.systemCallFailed` for the rest.
///
/// Identity checks cannot see inode reuse: if the caller's file is deleted
/// and an unrelated file is created that happens to receive the same inode
/// number on the same device, the two are indistinguishable here. That needs
/// the original to be deleted first, so it never turns a refusal into the
/// destruction of a file that was still present.
enum FileSafety {

    /// Identifies one file independently of the path used to reach it: two
    /// paths that differ in case on a case-insensitive volume, run through a
    /// symbolic link, or are hard links to the same file all share one
    /// identity.
    struct FileIdentity: Hashable, Sendable {
        let device: dev_t
        let inode: ino_t
    }

    /// What kind of item is at a path, read with `lstat` so a symbolic link
    /// is reported as itself and never as whatever it points at.
    enum ItemKind: Equatable, Sendable, CustomStringConvertible {
        case regularFile
        case directory
        case symbolicLink
        case fifo
        case socket
        case characterDevice
        case blockDevice
        case unknown

        init(mode: mode_t) {
            switch mode & S_IFMT {
            case S_IFREG: self = .regularFile
            case S_IFDIR: self = .directory
            case S_IFLNK: self = .symbolicLink
            case S_IFIFO: self = .fifo
            case S_IFSOCK: self = .socket
            case S_IFCHR: self = .characterDevice
            case S_IFBLK: self = .blockDevice
            default: self = .unknown
            }
        }

        var description: String {
            switch self {
            case .regularFile: return "regular file"
            case .directory: return "directory"
            case .symbolicLink: return "symbolic link"
            case .fifo: return "FIFO"
            case .socket: return "socket"
            case .characterDevice: return "character device"
            case .blockDevice: return "block device"
            case .unknown: return "file of unknown type"
            }
        }
    }

    /// An item that exists at a path: its kind and identity, both from
    /// `lstat` (a symbolic link's own identity, not its target's).
    struct ExistingItem: Equatable, Sendable {
        let kind: ItemKind
        let identity: FileIdentity
    }

    /// A file this process just created exclusively: a handle open for
    /// writing it (which closes the descriptor when released) and the
    /// identity that later proves the file is still this one.
    struct NewFile {
        let handle: FileHandle
        let identity: FileIdentity
    }

    /// What `openForWriting` does when a regular file already exists at the
    /// destination. Anything that exists and is *not* a regular file is
    /// refused under both policies.
    enum ExistingRegularFilePolicy: Sendable, Equatable {
        /// Fail with `.alreadyExists`; the file is untouched.
        case refuse
        /// Empty the existing regular file and write from its start.
        case truncate
    }

    /// The outcome of `removeOwnedItem` when it does not throw.
    enum OwnedItemRemoval: Sendable, Equatable {
        case removed
        /// Nothing was at the path any more.
        case alreadyGone
    }

    /// Permission bits for a newly created file (before the umask).
    static let newFileMode: mode_t = 0o644
    /// Permission bits for a newly created directory (before the umask).
    static let newDirectoryMode: mode_t = 0o755

    // MARK: - Inspecting

    /// What is at `url`, or nil when nothing is. Does not follow a final
    /// symbolic link.
    static func existingItem(at url: URL) throws -> ExistingItem? {
        var info = stat()
        guard lstat(url.path, &info) == 0 else {
            let code = errno
            if code == ENOENT { return nil }
            throw FileSafetyError.systemCallFailed(path: url.path, call: "lstat", errnoValue: code)
        }
        return ExistingItem(kind: ItemKind(mode: info.st_mode),
                            identity: FileIdentity(device: info.st_dev, inode: info.st_ino))
    }

    /// The identity of the file `url` resolves to, following symbolic links,
    /// or nil when nothing exists there (including a dangling link).
    static func resolvedIdentity(at url: URL) throws -> FileIdentity? {
        var info = stat()
        guard stat(url.path, &info) == 0 else {
            let code = errno
            if code == ENOENT { return nil }
            throw FileSafetyError.systemCallFailed(path: url.path, call: "stat", errnoValue: code)
        }
        return FileIdentity(device: info.st_dev, inode: info.st_ino)
    }

    /// The identity of the file open on `descriptor`. `path` only labels a
    /// failure.
    static func identity(ofOpenFileDescriptor descriptor: Int32, path: String) throws -> FileIdentity {
        var info = stat()
        guard fstat(descriptor, &info) == 0 else {
            let code = errno
            throw FileSafetyError.systemCallFailed(path: path, call: "fstat", errnoValue: code)
        }
        return FileIdentity(device: info.st_dev, inode: info.st_ino)
    }

    // MARK: - Comparing paths

    /// True when `first` and `second` may name one file — the one check for
    /// "these two paths must not be the same file" (two outputs of one run,
    /// or an output and an input it would destroy).
    ///
    /// They may when their paths, with symbolic links resolved (in every
    /// component that exists — a missing tail is kept as written,
    /// `pathForComparison`) and `.`/`..` removed, are equal ignoring case, or
    /// when both exist and resolve to the
    /// same device and inode. Case is ignored because APFS and HFS+ volumes are
    /// case-insensitive by default; on a case-sensitive volume that refuses two
    /// genuinely different files whose names differ only in case, which costs
    /// a caller nothing but a rename. The identity check catches what no path
    /// comparison can: hard links, and links resolved differently than above.
    ///
    /// A path check, so it shares the check-then-act window of the type's
    /// other path checks: a file linked into place between this call and the
    /// caller's open is not seen. Callers that open both paths compare the
    /// open descriptors' identities as well.
    static func mayNameTheSameFile(_ first: URL, _ second: URL) throws -> Bool {
        let firstPath = try pathForComparison(first)
        let secondPath = try pathForComparison(second)
        if firstPath.compare(secondPath, options: [.caseInsensitive]) == .orderedSame { return true }
        guard let firstIdentity = try resolvedIdentity(at: first),
              let secondIdentity = try resolvedIdentity(at: second) else {
            return false
        }
        return firstIdentity == secondIdentity
    }

    /// `url` with `.`/`..` removed and every symbolic link resolved in the
    /// part of the path that exists. `resolvingSymlinksInPath()` resolves
    /// nothing at all when the final item does not exist, so a file not yet
    /// created inside a linked folder would keep the link's spelling; here the
    /// deepest existing ancestor is resolved and the missing tail re-appended.
    private static func pathForComparison(_ url: URL) throws -> String {
        var existingAncestor = url.standardizedFileURL
        var missingTail: [String] = []
        while try existingItem(at: existingAncestor) == nil {
            let parent = existingAncestor.deletingLastPathComponent()
            guard parent.path != existingAncestor.path else { break }
            missingTail.insert(existingAncestor.lastPathComponent, at: 0)
            existingAncestor = parent
        }
        var resolved = existingAncestor.resolvingSymlinksInPath()
        for component in missingTail {
            resolved.appendPathComponent(component)
        }
        return resolved.standardizedFileURL.path
    }

    // MARK: - Creating

    /// Create `url` as a new, empty regular file and return a handle open for
    /// writing it, with the new file's identity. Throws `.alreadyExists` —
    /// naming what is there — when anything at all already exists at `url`;
    /// that item is never opened, truncated or followed.
    static func createNewFile(at url: URL) throws -> NewFile {
        try createNewFile(at: url, additionalOpenFlags: 0)
    }

    /// `createNewFile`, with extra `open` flags (a lock request).
    private static func createNewFile(at url: URL, additionalOpenFlags: Int32) throws -> NewFile {
        let descriptor = Darwin.open(url.path,
                                     O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC | additionalOpenFlags,
                                     newFileMode)
        guard descriptor >= 0 else {
            let code = errno
            throw failureForExclusiveCreate(at: url, errnoValue: code, call: "open")
        }
        let identity: FileIdentity
        do {
            identity = try self.identity(ofOpenFileDescriptor: descriptor, path: url.path)
        } catch {
            // Without the identity there is no proof that the name still
            // refers to the file this call created, so it is left in place
            // (empty) and reported rather than unlinked by name.
            closeDescriptorAfterFailure(descriptor, path: url.path)
            reportProblem("left the empty file \(url.path) in place: its identity could not be read, so it cannot be proven to be the file this call created")
            throw error
        }
        return NewFile(handle: FileHandle(fileDescriptor: descriptor, closeOnDealloc: true), identity: identity)
    }

    /// Create `url` as a new, empty directory and return its identity.
    /// Throws `.alreadyExists` when anything at all is already at `url`
    /// (`mkdir` refuses atomically; `FileManager.createDirectory` with
    /// intermediate directories would succeed on an existing directory and so
    /// adopt it).
    @discardableResult
    static func createNewDirectory(at url: URL) throws -> FileIdentity {
        guard mkdir(url.path, newDirectoryMode) == 0 else {
            let code = errno
            throw failureForExclusiveCreate(at: url, errnoValue: code, call: "mkdir")
        }
        guard let created = try existingItem(at: url), created.kind == .directory else {
            throw FileSafetyError.fileChangedSinceWritten(path: url.path)
        }
        return created.identity
    }

    /// Open `url` for writing, creating it when absent. An existing regular
    /// file is refused (`.alreadyExists`) or emptied according to `policy`;
    /// any other existing item (directory, symbolic link, FIFO, socket,
    /// device) is always refused and left untouched — as `.alreadyExists`
    /// under `.refuse` and `.notARegularFile` under `.truncate`. The returned
    /// handle owns the descriptor and closes it when released.
    static func openForWriting(at url: URL, existingRegularFile policy: ExistingRegularFilePolicy) throws -> FileHandle {
        switch policy {
        case .refuse:
            return try createNewFile(at: url).handle

        case .truncate:
            let descriptor = Darwin.open(url.path, O_WRONLY | O_CREAT | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC, newFileMode)
            guard descriptor >= 0 else {
                let code = errno
                if let existing = existingItemForDiagnosis(at: url), existing.kind != .regularFile {
                    throw FileSafetyError.notARegularFile(path: url.path, kind: existing.kind)
                }
                throw FileSafetyError.systemCallFailed(path: url.path, call: "open", errnoValue: code)
            }
            var info = stat()
            guard fstat(descriptor, &info) == 0 else {
                let code = errno
                closeDescriptorAfterFailure(descriptor, path: url.path)
                throw FileSafetyError.systemCallFailed(path: url.path, call: "fstat", errnoValue: code)
            }
            let kind = ItemKind(mode: info.st_mode)
            guard kind == .regularFile else {
                closeDescriptorAfterFailure(descriptor, path: url.path)
                throw FileSafetyError.notARegularFile(path: url.path, kind: kind)
            }
            guard ftruncate(descriptor, 0) == 0 else {
                let code = errno
                closeDescriptorAfterFailure(descriptor, path: url.path)
                throw FileSafetyError.systemCallFailed(path: url.path, call: "ftruncate", errnoValue: code)
            }
            // O_NONBLOCK stays set: it only guarded the open against a FIFO,
            // and it has no effect on reads or writes of a regular file.
            return FileHandle(fileDescriptor: descriptor, closeOnDealloc: true)
        }
    }

    /// A file opened for writing by `openForWritingReportingCreation`.
    struct OpenedForWriting {
        let handle: FileHandle
        /// The open file's identity, read from the open descriptor.
        let identity: FileIdentity
        /// True only when this call's exclusive create made the file, so the
        /// caller may remove it (by `identity`) if it abandons the write.
        /// False when an existing regular file was opened and emptied.
        let createdByThisCall: Bool
    }

    /// `openForWriting`, reporting whether the file is new — for a caller
    /// that opens several outputs and must undo exactly what it created when
    /// a later one is refused. The file is first created exclusively; only
    /// when that finds an existing regular file and `policy` is `.truncate`
    /// is that file opened and emptied (anything else there is refused, as in
    /// `openForWriting`). Should the existing file vanish between the two
    /// steps, the truncating open creates it and it is still reported as not
    /// created here — the direction that leaves a file rather than deletes one.
    static func openForWritingReportingCreation(at url: URL,
                                                existingRegularFile policy: ExistingRegularFilePolicy) throws -> OpenedForWriting {
        do {
            let created = try createNewFile(at: url)
            return OpenedForWriting(handle: created.handle, identity: created.identity, createdByThisCall: true)
        } catch FileSafetyError.alreadyExists(let path, let kind) {
            guard policy == .truncate else { throw FileSafetyError.alreadyExists(path: path, kind: kind) }
            guard kind == .regularFile else { throw FileSafetyError.notARegularFile(path: path, kind: kind) }
            let handle = try openForWriting(at: url, existingRegularFile: .truncate)
            let identity = try self.identity(ofOpenFileDescriptor: handle.fileDescriptor, path: url.path)
            return OpenedForWriting(handle: handle, identity: identity, createdByThisCall: false)
        }
    }

    /// Exclusively create `<stem>.<pathExtension>` in `directory`, or — when
    /// that name is taken by anything at all — `<stem>-2.<pathExtension>`,
    /// `<stem>-3.<pathExtension>`, … up to `<stem>-<maxAttempts>`. Existing
    /// items are never opened or modified, and two processes that pick the
    /// same stem in the same instant each end up with their own file. Returns
    /// the handle and the URL actually created.
    static func createNewFileWithNumericSuffix(
        in directory: URL,
        stem: String,
        pathExtension: String,
        maxAttempts: Int
    ) throws -> (handle: FileHandle, url: URL) {
        precondition(maxAttempts >= 1, "createNewFileWithNumericSuffix needs at least one attempt")
        for attempt in 1...maxAttempts {
            let name = attempt == 1 ? "\(stem).\(pathExtension)" : "\(stem)-\(attempt).\(pathExtension)"
            let url = directory.appendingPathComponent(name, isDirectory: false)
            do {
                return (try createNewFile(at: url).handle, url)
            } catch FileSafetyError.alreadyExists {
                continue
            }
        }
        throw FileSafetyError.noFreeNumericSuffix(
            directory: directory.path, stem: stem, pathExtension: pathExtension, maxAttempts: maxAttempts)
    }

    // MARK: - Exclusively locked files (a writer's file that others must leave alone)

    /// What `openExistingRegularFileWithExclusiveLock` found.
    enum ExistingFileLockAttempt {
        /// The lock was free and is now held by `handle` (open read-write)
        /// for as long as the handle stays open; `identity` is the file's.
        case locked(handle: FileHandle, identity: FileIdentity)
        /// Another open file holds the lock — in this process or another.
        /// Nothing was opened or changed.
        case heldByAnotherOpenFile
        /// Nothing is at the path.
        case gone
    }

    /// `createNewFile`, also taking an exclusive lock on the new file
    /// (`O_EXLOCK`), held until the returned handle is closed — explicitly,
    /// by its release, or by the kernel when the process exits however it
    /// exits. It marks the file as in use by a live writer:
    /// `openExistingRegularFileWithExclusiveLock` elsewhere finds it held.
    ///
    /// The lock is BSD `flock`-style: it belongs to this one open file, so
    /// opening and closing the same file elsewhere in this process (a
    /// whole-file read, say) leaves it in place. That is why it is not an
    /// `fcntl` lock, which any close of the file by this process would drop.
    ///
    /// `open` creates the file and then takes the lock, so another process
    /// can open the new, empty file in between; this call waits for the lock
    /// in that case (no `O_NONBLOCK`) rather than failing, and the other side
    /// sees an empty file, which no recovery treats as a shard. A volume that
    /// cannot lock (some network mounts) fails the call with the system's
    /// error — there is no unlocked fallback — and can leave the new empty
    /// file behind, which this does not remove (its identity is unknown).
    static func createNewFileHoldingExclusiveLock(at url: URL) throws -> NewFile {
        try createNewFile(at: url, additionalOpenFlags: O_EXLOCK)
    }

    /// Open the existing regular file at `url` read-write and take its
    /// exclusive lock without waiting — the counterpart to
    /// `createNewFileHoldingExclusiveLock` for a caller that may change the
    /// file only when no writer holds it. Returns `.heldByAnotherOpenFile`
    /// when the lock is taken, `.gone` when nothing is at `url`. A symbolic
    /// link, directory or other non-regular item is refused
    /// (`.notARegularFile`); so is a file replaced at `url` while it was
    /// being opened (`.fileChangedSinceWritten`). A volume that cannot lock
    /// is an error (`.systemCallFailed`), never treated as unlocked.
    static func openExistingRegularFileWithExclusiveLock(at url: URL) throws -> ExistingFileLockAttempt {
        let descriptor = Darwin.open(url.path, O_RDWR | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC | O_EXLOCK)
        guard descriptor >= 0 else {
            let code = errno
            switch code {
            case EAGAIN:
                // EWOULDBLOCK, the same value: the lock is held.
                return .heldByAnotherOpenFile
            case ENOENT:
                return .gone
            case ELOOP:
                throw FileSafetyError.notARegularFile(path: url.path, kind: .symbolicLink)
            case EISDIR:
                throw FileSafetyError.notARegularFile(path: url.path, kind: .directory)
            default:
                // A FIFO or device cannot take the lock (ENOTSUP) before its
                // type can be checked on a descriptor, so name what is there
                // when it is not a regular file; otherwise the failure is the
                // system's own (on a regular file, ENOTSUP means the volume
                // cannot lock).
                if let existing = existingItemForDiagnosis(at: url), existing.kind != .regularFile {
                    throw FileSafetyError.notARegularFile(path: url.path, kind: existing.kind)
                }
                throw FileSafetyError.systemCallFailed(path: url.path, call: "open", errnoValue: code)
            }
        }
        var info = stat()
        guard fstat(descriptor, &info) == 0 else {
            let code = errno
            closeDescriptorAfterFailure(descriptor, path: url.path)
            throw FileSafetyError.systemCallFailed(path: url.path, call: "fstat", errnoValue: code)
        }
        // `O_NONBLOCK` lets the open succeed on a FIFO with no writer, so the
        // type is checked on the descriptor before the file is ever used.
        let kind = ItemKind(mode: info.st_mode)
        guard kind == .regularFile else {
            closeDescriptorAfterFailure(descriptor, path: url.path)
            throw FileSafetyError.notARegularFile(path: url.path, kind: kind)
        }
        let identity = FileIdentity(device: info.st_dev, inode: info.st_ino)
        let atPath: ExistingItem?
        do {
            atPath = try existingItem(at: url)
        } catch {
            closeDescriptorAfterFailure(descriptor, path: url.path)
            throw error
        }
        guard let atPath, atPath.identity == identity else {
            closeDescriptorAfterFailure(descriptor, path: url.path)
            throw FileSafetyError.fileChangedSinceWritten(path: url.path)
        }
        // O_NONBLOCK stays set: it has no effect on reads or writes of a
        // regular file.
        return .locked(handle: FileHandle(fileDescriptor: descriptor, closeOnDealloc: true), identity: identity)
    }

    // MARK: - Writing whole files

    /// Write `data` to a brand-new file at `url`, durably (`fullSync`). The
    /// file is created exclusively (`createNewFile`), so nothing that already
    /// exists at `url` is ever touched. On a failure after creation the
    /// partial file — provably this call's own — is removed. Returns the new
    /// file's identity.
    ///
    /// The file appears under `url` as soon as it is created, so a crash
    /// mid-write leaves a torn file there; `publishNewFile` is the version
    /// that never shows a partial file under the final name.
    @discardableResult
    static func writeNewFile(_ data: Data, at url: URL) throws -> FileIdentity {
        let created = try createNewFile(at: url)
        var handleIsOpen = true
        do {
            try created.handle.write(contentsOf: data)
            try fullSync(fileDescriptor: created.handle.fileDescriptor, path: url.path)
            handleIsOpen = false
            try created.handle.close()
            return created.identity
        } catch {
            if handleIsOpen {
                do {
                    try created.handle.close()
                } catch let closeError {
                    reportProblem("closing \(url.path) after a failed write: \(closeError.localizedDescription)")
                }
            }
            removeOwnedItemAfterFailure(at: url, identity: created.identity)
            throw error
        }
    }

    /// Atomically publish `data` as a brand-new file at `destination`: staged
    /// durably in a temporary sibling, then moved into place with
    /// `renameWithoutReplacing`. Throws `.alreadyExists` when anything at all
    /// already exists at `destination`; the staged temporary is removed on
    /// every failure. Returns the published file's identity.
    @discardableResult
    static func publishNewFile(_ data: Data, to destination: URL) throws -> FileIdentity {
        let temporary = temporarySibling(of: destination)
        let identity = try writeNewFile(data, at: temporary)
        do {
            try renameWithoutReplacing(from: temporary, to: destination)
        } catch {
            removeOwnedItemAfterFailure(at: temporary, identity: identity)
            throw error
        }
        return identity
    }

    /// Move the item at `source` to `destination` only if nothing at all is
    /// at `destination` (`renamex_np` with `RENAME_EXCL`: the check and the
    /// move are one atomic step). Throws `.alreadyExists`, naming what is
    /// there, otherwise. Works for files and directories on the same volume.
    static func renameWithoutReplacing(from source: URL, to destination: URL) throws {
        guard renamex_np(source.path, destination.path, UInt32(RENAME_EXCL)) == 0 else {
            let code = errno
            throw failureForExclusiveCreate(at: destination, errnoValue: code, call: "renamex_np")
        }
    }

    /// Atomically write `data` at `destination`, replacing it only if it is a
    /// regular file — and, when `expectedIdentity` is given, only if it is
    /// still that very file (`.fileChangedSinceWritten` otherwise). When
    /// nothing is there — including when a file the caller owned has since
    /// been deleted — the file is published with `publishNewFile`, which
    /// overwrites nothing. A directory, symbolic link or special file is never
    /// replaced (`.notARegularFile`). Returns the written file's identity.
    ///
    /// The destination is checked before staging (so a refused destination
    /// costs no write) and again after (so the seconds a large write can take
    /// are not part of the unguarded window); see the type's documentation
    /// for the window that remains.
    @discardableResult
    static func replaceRegularFile(_ data: Data,
                                   at destination: URL,
                                   expectedIdentity: FileIdentity?) throws -> FileIdentity {
        guard let existing = try existingItem(at: destination) else {
            return try publishNewFile(data, to: destination)
        }
        try requireReplaceable(existing, at: destination, expectedIdentity: expectedIdentity)
        let temporary = temporarySibling(of: destination)
        let identity = try writeNewFile(data, at: temporary)
        do {
            // Nothing there any more is fine: the rename below then creates
            // the file, exactly as the publish above would have.
            if let current = try existingItem(at: destination) {
                try requireReplaceable(current, at: destination, expectedIdentity: expectedIdentity)
            }
        } catch {
            removeOwnedItemAfterFailure(at: temporary, identity: identity)
            throw error
        }
        // Plain rename atomically replaces a regular file and fails with
        // EISDIR rather than touching a directory, should one appear there
        // between the check above and this call.
        guard Darwin.rename(temporary.path, destination.path) == 0 else {
            let code = errno
            removeOwnedItemAfterFailure(at: temporary, identity: identity)
            if let racing = existingItemForDiagnosis(at: destination), racing.kind != .regularFile {
                throw FileSafetyError.notARegularFile(path: destination.path, kind: racing.kind)
            }
            throw FileSafetyError.systemCallFailed(path: destination.path, call: "rename", errnoValue: code)
        }
        return identity
    }

    /// `replaceRegularFile`'s rule for the item found at `destination`: a
    /// regular file, and the expected one when an identity is given.
    private static func requireReplaceable(_ item: ExistingItem,
                                           at destination: URL,
                                           expectedIdentity: FileIdentity?) throws {
        guard item.kind == .regularFile else {
            throw FileSafetyError.notARegularFile(path: destination.path, kind: item.kind)
        }
        if let expectedIdentity, item.identity != expectedIdentity {
            throw FileSafetyError.fileChangedSinceWritten(path: destination.path)
        }
    }

    // MARK: - Removing

    /// Remove the item at `url` only if it is still the one with `identity`
    /// — a file or directory the caller created. A directory is removed with
    /// its contents, but only after it has been moved to a private name and
    /// proven there to be the caller's (`deleteMovedAsideDirectory`), so a
    /// directory swapped in at `url` is never the one emptied. Throws
    /// `.fileChangedSinceWritten` when something else is at the path now (it
    /// is left alone), and returns `.alreadyGone` when nothing is.
    @discardableResult
    static func removeOwnedItem(at url: URL, identity: FileIdentity) throws -> OwnedItemRemoval {
        guard let current = try existingItem(at: url) else { return .alreadyGone }
        guard current.identity == identity else {
            throw FileSafetyError.fileChangedSinceWritten(path: url.path)
        }
        switch current.kind {
        case .regularFile:
            guard unlink(url.path) == 0 else {
                let code = errno
                throw FileSafetyError.systemCallFailed(path: url.path, call: "unlink", errnoValue: code)
            }
        case .directory:
            // A recursive delete by path takes as long as the folder is big,
            // and a folder swapped in at `url` during it would be emptied.
            // Moving the folder aside first makes the deletion act on a name
            // no other process knows.
            let movedAside = temporarySibling(of: url)
            guard renamex_np(url.path, movedAside.path, UInt32(RENAME_EXCL)) == 0 else {
                let code = errno
                if code == ENOENT { return .alreadyGone }
                throw failureForExclusiveCreate(at: movedAside, errnoValue: code, call: "renamex_np")
            }
            try deleteMovedAsideDirectory(at: movedAside, movedFrom: url, expectedIdentity: identity)
        case .symbolicLink, .fifo, .socket, .characterDevice, .blockDevice, .unknown:
            // Only regular files and directories are ever created here, so a
            // matching identity on anything else is not the caller's item.
            throw FileSafetyError.fileChangedSinceWritten(path: url.path)
        }
        return .removed
    }

    /// The second half of `removeOwnedItem` for a directory: `movedAside` is
    /// what was just renamed away from `originalURL`. It is deleted, with its
    /// contents, only if it is the directory with `expectedIdentity`.
    /// Anything else — an item swapped in at `originalURL` between the
    /// identity check and the rename, and so moved by mistake — is moved back
    /// to `originalURL` untouched and `.fileChangedSinceWritten` is thrown.
    /// Internal rather than private so tests can drive the mismatch, which in
    /// `removeOwnedItem` needs a race to reach.
    static func deleteMovedAsideDirectory(at movedAside: URL,
                                          movedFrom originalURL: URL,
                                          expectedIdentity: FileIdentity) throws {
        let moved: ExistingItem?
        do {
            moved = try existingItem(at: movedAside)
        } catch {
            // Unverifiable, so it must not be deleted.
            moveBackAfterRefusedRemoval(from: movedAside, to: originalURL)
            throw error
        }
        guard let moved else {
            reportProblem("\(movedAside.path), moved there from \(originalURL.path) to be removed, vanished before it could be checked")
            throw FileSafetyError.fileChangedSinceWritten(path: originalURL.path)
        }
        guard moved.kind == .directory, moved.identity == expectedIdentity else {
            moveBackAfterRefusedRemoval(from: movedAside, to: originalURL)
            throw FileSafetyError.fileChangedSinceWritten(path: originalURL.path)
        }
        try FileManager.default.removeItem(at: movedAside)
    }

    /// Put an item that `removeOwnedItem` moved aside but may not delete
    /// back where it was, without replacing anything that has appeared there
    /// since. A failure is reported, not thrown: the caller is already
    /// throwing the refusal, and the item is intact under `movedAside`.
    private static func moveBackAfterRefusedRemoval(from movedAside: URL, to originalURL: URL) {
        do {
            try renameWithoutReplacing(from: movedAside, to: originalURL)
        } catch {
            reportProblem("left \(movedAside.path) in place, intact: it was moved there from \(originalURL.path) to be removed, turned out not to be the item to remove, and could not be moved back: \(error.localizedDescription)")
        }
    }

    // MARK: - Durability

    /// Force the file open on `descriptor` to stable storage with
    /// `F_FULLFSYNC` (past the drive's write cache), falling back to `fsync`
    /// on filesystems that lack it (some network mounts). `path` only labels
    /// a failure.
    static func fullSync(fileDescriptor descriptor: Int32, path: String) throws {
        if fcntl(descriptor, F_FULLFSYNC) == -1 {
            guard fsync(descriptor) == 0 else {
                let code = errno
                throw FileSafetyError.systemCallFailed(path: path, call: "fsync", errnoValue: code)
            }
        }
    }

    /// `fullSync(fileDescriptor:path:)` for the file or directory at `url`,
    /// opened read-only for the purpose (a directory's flush makes a rename
    /// inside it durable).
    static func fullSync(at url: URL) throws {
        let descriptor = Darwin.open(url.path, O_RDONLY | O_CLOEXEC)
        guard descriptor >= 0 else {
            let code = errno
            throw FileSafetyError.systemCallFailed(path: url.path, call: "open", errnoValue: code)
        }
        do {
            try fullSync(fileDescriptor: descriptor, path: url.path)
        } catch {
            closeDescriptorAfterFailure(descriptor, path: url.path)
            throw error
        }
        guard close(descriptor) == 0 else {
            let code = errno
            throw FileSafetyError.systemCallFailed(path: url.path, call: "close", errnoValue: code)
        }
    }

    // MARK: - Staging names

    /// A hidden, per-call unique sibling of `destination` for staging a
    /// write: `.<destination name>.<UUID>.tmp`. The `.tmp` extension keeps a
    /// leftover (process killed between staging and rename) recognizable to
    /// `CorpusValidator`'s stray-temp check, the exact shape lets the launch
    /// sweep recognize one (`destinationName(ofTemporarySiblingName:)`), and
    /// the leading dot keeps it out of folder listings that skip hidden
    /// files.
    static func temporarySibling(of destination: URL) -> URL {
        destination.deletingLastPathComponent()
            .appendingPathComponent(temporarySiblingName(forDestinationName: destination.lastPathComponent,
                                                         uniqueToken: UUID().uuidString))
    }

    /// How many UTF-8 bytes a temporary sibling's name adds to its
    /// destination's name. A staged write to a destination whose name is
    /// within `NAME_MAX` but not within `NAME_MAX` minus this fails with
    /// `ENAMETOOLONG` at staging, so a caller that validates a file name
    /// it will write this way must leave this much room.
    static let temporarySiblingNameOverhead: Int =
        FileSafety.temporarySiblingName(forDestinationName: "", uniqueToken: UUID().uuidString).utf8.count

    /// The destination name a temporary sibling named `name` was staging, or
    /// nil when `name` is not exactly a name `temporarySibling(of:)` makes:
    /// a leading `.`, a non-empty destination name, a `.`, an upper-case
    /// UUID as `UUID().uuidString` spells it, and `.tmp`. Recognizes only
    /// what this type writes, so a sweep keyed on it never claims anyone
    /// else's hidden file.
    static func destinationName(ofTemporarySiblingName name: String) -> String? {
        let suffix = ".\(temporarySiblingExtension)"
        let uuidLength = UUID().uuidString.count
        guard name.hasSuffix(suffix) else { return nil }
        let withoutSuffix = name.dropLast(suffix.count)
        guard withoutSuffix.count > uuidLength else { return nil }
        let uniqueToken = String(withoutSuffix.suffix(uuidLength))
        guard let uuid = UUID(uuidString: uniqueToken), uuid.uuidString == uniqueToken else { return nil }
        // What is left is `.<destination name>.`; at least one character
        // between the dots.
        let enclosedDestinationName = withoutSuffix.dropLast(uuidLength)
        guard enclosedDestinationName.count >= 3 else { return nil }
        let destinationName = String(enclosedDestinationName.dropFirst().dropLast())
        // Rebuilding the name proves every separator is where the builder
        // puts it, so nothing but an exact match is ever claimed.
        guard temporarySiblingName(forDestinationName: destinationName, uniqueToken: uniqueToken) == name else {
            return nil
        }
        return destinationName
    }

    /// The extension every temporary sibling carries.
    private static let temporarySiblingExtension = "tmp"

    /// The one definition of a temporary sibling's name, shared by the
    /// builder, the overhead and the recognizer above so they cannot drift.
    private static func temporarySiblingName(forDestinationName destinationName: String, uniqueToken: String) -> String {
        ".\(destinationName).\(uniqueToken).\(temporarySiblingExtension)"
    }

    // MARK: - Failure helpers

    /// The error for an exclusive create or rename that failed: when
    /// something is in the way, name it (that is the actionable fact);
    /// otherwise report the system error.
    private static func failureForExclusiveCreate(at url: URL, errnoValue: Int32, call: String) -> FileSafetyError {
        if errnoValue == EEXIST, let existing = existingItemForDiagnosis(at: url) {
            return .alreadyExists(path: url.path, kind: existing.kind)
        }
        return .systemCallFailed(path: url.path, call: call, errnoValue: errnoValue)
    }

    /// `existingItem(at:)` for explaining a failure that is already being
    /// thrown. When the inspection itself fails, that is reported and nil is
    /// returned, so the caller throws the original system error instead of
    /// the inspection's.
    private static func existingItemForDiagnosis(at url: URL) -> ExistingItem? {
        do {
            return try existingItem(at: url)
        } catch {
            reportProblem("could not inspect \(url.path) to explain a failed operation: \(error.localizedDescription)")
            return nil
        }
    }

    /// `removeOwnedItem` on a failure path, where the original error is what
    /// the caller throws: any problem is reported rather than thrown.
    private static func removeOwnedItemAfterFailure(at url: URL, identity: FileIdentity) {
        do {
            if try removeOwnedItem(at: url, identity: identity) == .alreadyGone {
                reportProblem("\(url.path), created by this operation, was already gone when it went to remove it")
            }
        } catch {
            reportProblem("left \(url.path) in place: \(error.localizedDescription)")
        }
    }

    private static func closeDescriptorAfterFailure(_ descriptor: Int32, path: String) {
        if close(descriptor) != 0 {
            let code = errno
            reportProblem("closing \(path) after a failure: \(String(cString: strerror(code)))")
        }
    }

    /// Report a problem met while cleaning up after, or explaining, a failure
    /// that is already being thrown — to the session log and stderr, so it
    /// is visible in the GUI's log and in a headless tool's terminal.
    private static func reportProblem(_ message: String) {
        SessionLogger.shared.log("[FILE-SAFETY] \(message)")
        FileHandle.standardError.write(Data("[FILE-SAFETY] \(message)\n".utf8))
    }
}

/// Every refusal and failure from `FileSafety`.
enum FileSafetyError: LocalizedError, Equatable {
    /// Something already exists where a brand-new file or directory was to
    /// be created (or renamed to); it was not opened or modified.
    case alreadyExists(path: String, kind: FileSafety.ItemKind)
    /// The path holds something other than a regular file, so it is never
    /// replaced, truncated or written.
    case notARegularFile(path: String, kind: FileSafety.ItemKind)
    /// The item at the path is no longer the one the caller created, so the
    /// caller no longer owns it and must not replace or remove it.
    case fileChangedSinceWritten(path: String)
    /// Every candidate name for a numeric-suffix create was taken.
    case noFreeNumericSuffix(directory: String, stem: String, pathExtension: String, maxAttempts: Int)
    /// A system call failed for a reason other than the above. `errnoValue`
    /// is the real `errno`; `CorpusReplayRunner.isOutOfSpace` reads it to
    /// classify a disk-full (`ENOSPC`) failure.
    case systemCallFailed(path: String, call: String, errnoValue: Int32)

    /// True for the refusals to touch an item the caller does not own
    /// (`.alreadyExists`, `.notARegularFile`, `.fileChangedSinceWritten`) —
    /// as opposed to an I/O failure. Long-running writers halt on these
    /// rather than treating them as a transient save failure.
    var isOwnershipRefusal: Bool {
        switch self {
        case .alreadyExists, .notARegularFile, .fileChangedSinceWritten:
            return true
        case .noFreeNumericSuffix, .systemCallFailed:
            return false
        }
    }

    var errorDescription: String? {
        switch self {
        case .alreadyExists(let path, let kind):
            if kind == .regularFile {
                return "\(path) already exists; refusing to overwrite it"
            }
            return "\(path) already exists and is a \(kind); refusing to write over it"
        case .notARegularFile(let path, let kind):
            return "\(path) is a \(kind), not a regular file; refusing to replace or write to it"
        case .fileChangedSinceWritten(let path):
            return "refusing to replace or remove \(path): it is no longer the item this process created there "
                + "(something else replaced it)"
        case .noFreeNumericSuffix(let directory, let stem, let pathExtension, let maxAttempts):
            return "no free name for \(stem).\(pathExtension) in \(directory): all \(maxAttempts) candidate names (the plain name, then -2, -3, …) are taken"
        case .systemCallFailed(let path, let call, let errnoValue):
            if let code = POSIXErrorCode(rawValue: errnoValue) {
                return "\(call) failed for \(path): \(POSIXError(code).localizedDescription) (errno \(errnoValue))"
            }
            return "\(call) failed for \(path): errno \(errnoValue)"
        }
    }
}
