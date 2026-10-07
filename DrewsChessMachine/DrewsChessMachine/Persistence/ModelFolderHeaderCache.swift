import Foundation

/// What a scan knows about a listed model file without reading it: the file
/// the path resolves to (device and inode, following a symbolic link the way
/// the header read and the load do), its size and its modification time. A
/// file renamed over the path (a rolling `--out-model` save) is a new inode,
/// so it never matches its predecessor even at the same size and second.
struct ModelFileFingerprint: Hashable, Sendable {
    let identity: FileSafety.FileIdentity
    let size: Int64
    let modifiedAt: Date
}

/// One model folder's headers, read incrementally (follow-lineage plan
/// §3.2). A scan lists the folder exactly as `ModelFileCatalog` always has
/// (non-hidden items named `*.safetensors`, not recursive), `stat`s each,
/// and reads a header only when the file's fingerprint differs from the
/// previous scan's — so a folder of thousands of checkpoints costs one
/// listing and one `stat` per file once its headers are known.
///
/// A file that could not be read is remembered by its fingerprint too: a
/// half-copied file is read again as soon as it grows, and not on every
/// scan while it sits unchanged. An item that resolves to something other
/// than a regular file (a folder named `x.safetensors`, a dangling link) is
/// listed as unreadable with that reason, never dropped.
struct ModelFolderHeaderCache: Sendable {

    /// What a listed file turned out to be.
    enum Outcome: Sendable, Equatable {
        case entry(ModelFileEntry)
        case unreadable(UnreadableModelFile)
    }

    private struct Item: Sendable {
        let fingerprint: ModelFileFingerprint
        let outcome: Outcome
    }

    private let items: [URL: Item]

    /// A cache that knows nothing: the next scan reads every header.
    static let empty = ModelFolderHeaderCache(items: [:])

    private init(items: [URL: Item]) {
        self.items = items
    }

    /// The fingerprint `url` had when the scan that built this cache read it,
    /// or nil when that scan did not list it (or could not stat it).
    func fingerprint(of url: URL) -> ModelFileFingerprint? {
        items[url]?.fingerprint
    }

    /// Scan `directory`, reusing `previous` for every file whose fingerprint
    /// is unchanged. Throws when the folder itself cannot be listed — never
    /// "empty".
    static func scanSynchronously(directory: URL, previous: ModelFolderHeaderCache) throws -> ModelFolderScan {
        let started = ContinuousClock.now
        let urls: [URL]
        do {
            urls = try FileManager.default.contentsOfDirectory(
                at: directory,
                includingPropertiesForKeys: [.contentModificationDateKey, .isRegularFileKey],
                options: [.skipsHiddenFiles]
            ).filter { $0.pathExtension == "safetensors" }
        } catch {
            throw ModelFolderScanError.folderUnreadable(directory: directory, reason: error.localizedDescription)
        }

        var items: [URL: Item] = [:]
        var entries: [ModelFileEntry] = []
        var unreadable: [UnreadableModelFile] = []
        var headersRead = 0
        var reused = 0
        for url in urls {
            let resolved: FileSafety.ResolvedItem?
            do {
                resolved = try FileSafety.resolvedItem(at: url)
            } catch {
                unreadable.append(UnreadableModelFile(url: url, reason: error.localizedDescription))
                continue
            }
            guard let resolved else {
                unreadable.append(UnreadableModelFile(url: url, reason: "\(url.lastPathComponent) is a symbolic link to nothing"))
                continue
            }
            let fingerprint = ModelFileFingerprint(identity: resolved.identity, size: resolved.size, modifiedAt: resolved.modifiedAt)
            let outcome: Outcome
            if let known = previous.items[url], known.fingerprint == fingerprint {
                outcome = known.outcome
                reused += 1
            } else if resolved.kind != .regularFile {
                outcome = .unreadable(UnreadableModelFile(url: url, reason: "\(url.lastPathComponent) is a \(resolved.kind), not a model file"))
            } else {
                headersRead += 1
                do {
                    outcome = .entry(try ModelFileCatalog.entry(for: url))
                } catch {
                    outcome = .unreadable(UnreadableModelFile(url: url, reason: error.localizedDescription))
                }
            }
            items[url] = Item(fingerprint: fingerprint, outcome: outcome)
            switch outcome {
            case .entry(let entry):
                entries.append(entry)
            case .unreadable(let file):
                unreadable.append(file)
            }
        }
        let elapsed = ContinuousClock.now - started
        return ModelFolderScan(
            cache: ModelFolderHeaderCache(items: items),
            entries: entries,
            unreadable: unreadable,
            listed: urls.count,
            headersRead: headersRead,
            reused: reused,
            elapsedMilliseconds: Double(elapsed.components.seconds) * 1000 + Double(elapsed.components.attoseconds) / 1e15
        )
    }

    /// `scanSynchronously` off the caller's thread, for callers on the main
    /// actor (a settings row resolving a lineage).
    static func scan(directory: URL, previous: ModelFolderHeaderCache) async throws -> ModelFolderScan {
        try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                continuation.resume(with: Result { try scanSynchronously(directory: directory, previous: previous) })
            }
        }
    }
}

/// One scan of a model folder.
struct ModelFolderScan: Sendable {
    /// What the next scan reuses.
    let cache: ModelFolderHeaderCache
    let entries: [ModelFileEntry]
    let unreadable: [UnreadableModelFile]
    /// Items named `*.safetensors` in the folder.
    let listed: Int
    /// Headers read by this scan (new or changed files).
    let headersRead: Int
    /// Files whose earlier result was reused unchanged.
    let reused: Int
    let elapsedMilliseconds: Double
}

enum ModelFolderScanError: LocalizedError, Equatable {
    case folderUnreadable(directory: URL, reason: String)

    var errorDescription: String? {
        switch self {
        case .folderUnreadable(let directory, let reason):
            return "The folder \(directory.path) can't be listed: \(reason)"
        }
    }
}
