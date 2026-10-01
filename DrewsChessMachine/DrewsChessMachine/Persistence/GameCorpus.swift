import Foundation

/// Filesystem locations for the standalone corpus store. Mirrors
/// `CheckpointManager`'s Application Support layout but kept independent so a
/// corpus never lives inside a session folder.
enum CorpusPaths {
    static var rootURL: URL {
        let fm = FileManager.default
        let support = fm.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? URL(fileURLWithPath: NSHomeDirectory(), isDirectory: true)
                .appendingPathComponent("Library/Application Support", isDirectory: true)
        return support.appendingPathComponent("DrewsChessMachine", isDirectory: true)
    }

    /// `~/Library/Application Support/DrewsChessMachine/Corpora/`.
    static var corporaDir: URL {
        rootURL.appendingPathComponent("Corpora", isDirectory: true)
    }
}

/// Mints corpus and source identifiers. Uses a UTC timestamp + random suffix
/// (no `UserDefaults` counter) so it is safe to call off the main actor,
/// unlike `ModelID.mint()`.
enum CorpusID {
    private static let base62: [Character] =
        Array("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")

    private static let stampFormatter: DateFormatter = {
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.timeZone = TimeZone(identifier: "UTC")
        f.dateFormat = "yyyyMMdd-HHmmss"
        return f
    }()

    static func mintCorpus(now: Date = Date()) -> String {
        "\(stampFormatter.string(from: now))-\(randomSuffix(6))"
    }

    static func mintSource() -> String {
        "src-\(randomSuffix(8))"
    }

    private static func randomSuffix(_ length: Int) -> String {
        var chars: [Character] = []
        chars.reserveCapacity(length)
        for _ in 0..<length {
            chars.append(base62[Int.random(in: 0..<base62.count)])
        }
        return String(chars)
    }
}

/// One ingestion event's provenance (a self-play recording session or a PGN
/// import). Append-only within `CorpusMetadata.sources`; every field beyond
/// the identity is Optional so the schema can grow additively.
struct CorpusSource: Codable, Equatable, Sendable {
    var sourceID: String
    var kind: String                 // "selfPlay" | "pgnImport"
    var addedAtUnix: Int64
    var appBuildNumber: Int?
    var appGitHash: String?
    var inputFilename: String?
    var inputURL: String?
    var shardSoftLimitBytes: Int?
    var minRating: Int?
    var timeControls: [String]?
    var maxGames: Int?
    var gamesAdded: Int?
    var pliesAdded: Int?
    var complete: Bool?
}

/// The single provenance file (`corpus.json`) for a corpus. Holds only the
/// non-reconstructable metadata; the shard list and counts are derived from
/// the shard files themselves.
struct CorpusMetadata: Codable, Equatable, Sendable {
    static let currentFormatVersion = 1
    var formatVersion: Int
    var corpusID: String
    var name: String?
    var comment: String?
    var state: String                // "recording" | "sealed"
    var createdAtUnix: Int64
    var sources: [CorpusSource]
}

/// What `GameCorpus.recoverOpenShard(at:)` did with one unsealed `.open`
/// shard.
enum OpenShardRecovery: Equatable, Sendable {
    /// The shard held at least one complete game: it was truncated to its
    /// last complete game (dropping `discardedTailBytes` of torn tail) and
    /// sealed under `sealedShard`.
    case sealed(openShard: URL, sealedShard: URL, gameCount: Int, plyCount: Int, discardedTailBytes: Int)
    /// The shard held no complete game: it was deleted (its header plus
    /// `discardedTailBytes` of torn tail).
    case removedEmpty(openShard: URL, discardedTailBytes: Int)

    /// One line describing the outcome, for logs and reports.
    var summary: String {
        switch self {
        case let .sealed(openShard, sealedShard, gameCount, plyCount, discardedTailBytes):
            return "\(openShard.lastPathComponent): sealed as \(sealedShard.lastPathComponent) with "
                + "\(gameCount) complete game(s), \(plyCount) plies; dropped \(discardedTailBytes) byte(s) of "
                + "incomplete tail"
        case let .removedEmpty(openShard, discardedTailBytes):
            return "\(openShard.lastPathComponent): removed — it held no complete game (header plus "
                + "\(discardedTailBytes) byte(s) of incomplete tail)"
        }
    }
}

/// A corpus as seen by a reader: its metadata and sealed shards, read without
/// modifying anything (see `GameCorpus.openReadOnly`). A value snapshot taken
/// at open time; shards a writer seals afterwards are not in it.
struct GameCorpusReadOnlyView: Sendable {
    let directory: URL
    let metadata: CorpusMetadata
    /// Sealed shard files in stable (sequence) order.
    let sealedShardURLs: [URL]
    /// Unsealed `.open` shards present at open time, which this view neither
    /// reads nor touches — a live writer's shard or a crash leftover.
    let ignoredOpenShardURLs: [URL]

    var corpusID: String { metadata.corpusID }
}

/// A standalone, append-only game corpus on disk: a directory under `Corpora/`
/// holding a `corpus.json` and a series of self-describing shard files.
///
/// Single-writer — recording will drive it from a serial async queue, so it is
/// intentionally not `Sendable` and must not be shared across threads without
/// external serialization (that wrapper is wired in the recording step, not
/// here). `state` moves `recording` → `sealed`; a sealed corpus is frozen for
/// replay.
final class GameCorpus {
    let directory: URL
    let corpusID: String
    private(set) var metadata: CorpusMetadata
    private let shardSoftLimitBytes: Int
    private var nextShardSeq: UInt32
    private var currentWriter: ShardWriter?
    private var currentSourceID: String?

    static let metadataFilename = "corpus.json"
    static let shardExtension = "dcmgames"
    static let defaultShardSoftLimitBytes = 64 * 1024 * 1024

    private init(directory: URL,
                 metadata: CorpusMetadata,
                 shardSoftLimitBytes: Int,
                 nextShardSeq: UInt32) {
        self.directory = directory
        self.corpusID = metadata.corpusID
        self.metadata = metadata
        self.shardSoftLimitBytes = max(1, shardSoftLimitBytes)
        self.nextShardSeq = nextShardSeq
    }

    // MARK: Create / open

    /// Create a new corpus directory under `parentDirectory` (defaults to the
    /// shared `Corpora/` store) in the `recording` state.
    static func create(name: String?,
                       comment: String?,
                       shardSoftLimitBytes: Int = defaultShardSoftLimitBytes,
                       parentDirectory: URL? = nil) throws -> GameCorpus {
        let corpusID = CorpusID.mintCorpus()
        let base = parentDirectory ?? CorpusPaths.corporaDir
        let dir = base.appendingPathComponent(corpusID, isDirectory: true)
        do {
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        } catch {
            throw GameCorpusError.ioFailed("create corpus dir: \(error.localizedDescription)")
        }
        let meta = CorpusMetadata(formatVersion: CorpusMetadata.currentFormatVersion,
                                  corpusID: corpusID,
                                  name: name,
                                  comment: comment,
                                  state: "recording",
                                  createdAtUnix: Int64(Date().timeIntervalSince1970),
                                  sources: [])
        let corpus = GameCorpus(directory: dir,
                                metadata: meta,
                                shardSoftLimitBytes: shardSoftLimitBytes,
                                nextShardSeq: 0)
        try corpus.writeMetadata()
        return corpus
    }

    /// Open an existing corpus directory **for writing**. Any leftover `.open`
    /// shard is recovered: scanned to its last complete game, truncated, and
    /// sealed (or deleted when it holds no game), leaving the corpus consistent
    /// with only sealed shards.
    ///
    /// That recovery modifies the corpus, and it cannot tell a crash leftover
    /// from a shard another process is still appending to — run against a
    /// corpus that is being recorded or imported into, it truncates and seals
    /// the writer's live shard out from under it. So only the process that
    /// owns the corpus as its single writer may call this. Anything that only
    /// reads a corpus (corpus replay, inspection) uses `openReadOnly`, which
    /// never changes a byte on disk. No app path opens an existing corpus for
    /// writing today; the explicit, operator-invoked recovery of crash
    /// leftovers is `--validate-corpus <dir> --fix` (`CorpusValidator`), which
    /// shares `recoverOpenShard(at:)` with this.
    static func open(directory: URL,
                     shardSoftLimitBytes: Int = defaultShardSoftLimitBytes) throws -> GameCorpus {
        let metaURL = directory.appendingPathComponent(metadataFilename)
        let data: Data
        do { data = try Data(contentsOf: metaURL) }
        catch { throw GameCorpusError.ioFailed("read corpus.json: \(error.localizedDescription)") }
        let meta: CorpusMetadata
        do { meta = try JSONDecoder().decode(CorpusMetadata.self, from: data) }
        catch { throw GameCorpusError.corruptMetadata("corpus.json decode: \(error.localizedDescription)") }

        let nextSeq = try highestShardSeq(in: directory).map { $0 + 1 } ?? 0
        let corpus = GameCorpus(directory: directory,
                                metadata: meta,
                                shardSoftLimitBytes: shardSoftLimitBytes,
                                nextShardSeq: nextSeq)
        try corpus.recoverOpenShardsIfPresent()
        return corpus
    }

    /// Open an existing corpus for reading only: decode `corpus.json` and list
    /// the sealed shards, changing nothing on disk.
    ///
    /// Unlike `open(directory:)`, no `.open` shard is recovered, truncated,
    /// sealed, renamed or deleted. An `.open` shard is either a writer's live
    /// shard (a recording or import still in progress) or a crash leftover,
    /// and a reader cannot tell which — so it never touches one, and it never
    /// reads one either (its tail may be mid-append). Those shards are
    /// returned in `ignoredOpenShardURLs` so the caller can say loudly which
    /// games it is not seeing.
    static func openReadOnly(directory: URL) throws -> GameCorpusReadOnlyView {
        let metadata = try Self.loadMetadata(directory: directory)
        let entries: [URL]
        do {
            entries = try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
        } catch {
            throw GameCorpusError.ioFailed("list corpus directory \(directory.path): \(error.localizedDescription)")
        }
        return GameCorpusReadOnlyView(directory: directory,
                                      metadata: metadata,
                                      sealedShardURLs: Self.sealedShardURLs(among: entries),
                                      ignoredOpenShardURLs: Self.openShardURLs(among: entries))
    }

    // MARK: Recording

    /// Begin an ingestion source and open its first shard. Returns the new
    /// `sourceID`. One source maps to one recording session or one import.
    @discardableResult
    func beginSource(kind: String,
                     inputFilename: String? = nil,
                     inputURL: String? = nil,
                     minRating: Int? = nil,
                     timeControls: [String]? = nil,
                     maxGames: Int? = nil) throws -> String {
        guard currentWriter == nil, currentSourceID == nil else {
            throw GameCorpusError.invalidState("a source is already in progress")
        }
        guard metadata.state == "recording" else {
            throw GameCorpusError.invalidState("corpus is sealed")
        }
        let sourceID = CorpusID.mintSource()
        let source = CorpusSource(sourceID: sourceID,
                                  kind: kind,
                                  addedAtUnix: Int64(Date().timeIntervalSince1970),
                                  appBuildNumber: BuildInfo.buildNumber,
                                  appGitHash: BuildInfo.gitHash,
                                  inputFilename: inputFilename,
                                  inputURL: inputURL,
                                  shardSoftLimitBytes: shardSoftLimitBytes,
                                  minRating: minRating,
                                  timeControls: timeControls,
                                  maxGames: maxGames,
                                  gamesAdded: 0,
                                  pliesAdded: 0,
                                  complete: false)
        metadata.sources.append(source)
        currentSourceID = sourceID
        try writeMetadata()
        try openNewShard()
        return sourceID
    }

    /// Append one game to the current source's open shard, sealing and rotating
    /// to a fresh shard once the soft byte limit is crossed (always on a
    /// whole-game boundary).
    func append(_ game: GameRecord) throws {
        try append(framed: GameCorpusShardFormat.encodeFramedRecord(game), plyCount: game.moves.count)
    }

    /// Append a pre-encoded framed game (encoded off the writer thread by the
    /// caller, e.g. the parallel PGN importer). Mirrors `append(_:)` exactly but
    /// skips re-encoding; seals and rotates on the same whole-game boundary.
    func append(framed frame: Data, plyCount: Int) throws {
        guard let writer = currentWriter else {
            throw GameCorpusError.invalidState("no source in progress; call beginSource first")
        }
        try writer.appendFramed(frame, plyCount: plyCount)
        if !metadata.sources.isEmpty {
            let i = metadata.sources.count - 1
            metadata.sources[i].gamesAdded = (metadata.sources[i].gamesAdded ?? 0) + 1
            metadata.sources[i].pliesAdded = (metadata.sources[i].pliesAdded ?? 0) + plyCount
        }
        if writer.byteCount >= shardSoftLimitBytes {
            _ = try writer.seal(sealUnix: Int64(Date().timeIntervalSince1970))
            currentWriter = nil
            try openNewShard()
        }
    }

    /// Seal the current source's open shard (or discard it if empty) and mark
    /// the source complete in `corpus.json`.
    func finishSource() throws {
        if let writer = currentWriter {
            if writer.gameCount > 0 {
                _ = try writer.seal(sealUnix: Int64(Date().timeIntervalSince1970))
            } else {
                try writer.discardEmpty()
            }
            currentWriter = nil
        }
        if !metadata.sources.isEmpty {
            metadata.sources[metadata.sources.count - 1].complete = true
        }
        currentSourceID = nil
        try writeMetadata()
    }

    /// Finish any in-progress source and mark the corpus frozen for replay.
    func seal() throws {
        if currentWriter != nil || currentSourceID != nil {
            try finishSource()
        }
        metadata.state = "sealed"
        try writeMetadata()
    }

    // MARK: Reading

    /// Sealed shard files in stable (sequence) order.
    func sealedShardURLs() throws -> [URL] {
        let fm = FileManager.default
        let entries = try fm.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
        return Self.sealedShardURLs(among: entries)
    }

    /// The sealed shard files among a corpus directory's `entries`, in stable
    /// (sequence) order.
    static func sealedShardURLs(among entries: [URL]) -> [URL] {
        entries
            .filter { $0.pathExtension == shardExtension && $0.lastPathComponent.hasPrefix("shard-") }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
    }

    /// The unsealed (`.open`) shard files among a corpus directory's
    /// `entries`, in stable order — the same selection crash recovery acts on.
    static func openShardURLs(among entries: [URL]) -> [URL] {
        entries
            .filter { $0.pathExtension == "open" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
    }

    /// Read every game from every sealed shard, in shard order.
    ///
    /// Loads all games into memory — for true replay the feeder streams
    /// shard-by-shard; this convenience is for tests and small corpora.
    func allGames() throws -> [GameRecord] {
        var games: [GameRecord] = []
        for url in try sealedShardURLs() {
            let shard = try GameCorpusShardIO.readSealed(at: url)
            games.append(contentsOf: shard.games)
        }
        return games
    }

    // MARK: Internals

    private func openNewShard() throws {
        guard let sourceID = currentSourceID else {
            throw GameCorpusError.invalidState("no source in progress")
        }
        let seq = nextShardSeq
        // Zero-pad the sequence by hand rather than via String(format:) to dodge
        // the printf %u/UInt32 CVarArg width pitfall.
        let seqDigits = String(seq)
        let padded = String(repeating: "0", count: max(0, 5 - seqDigits.count)) + seqDigits
        let name = "shard-\(padded).\(Self.shardExtension).open"
        let url = directory.appendingPathComponent(name)
        let header = GameCorpusShardFormat.FrontHeader(corpusID: corpusID,
                                                       sourceID: sourceID,
                                                       shardSeq: seq,
                                                       createdAtUnix: Int64(Date().timeIntervalSince1970))
        currentWriter = try ShardWriter(creatingAt: url, header: header)
        nextShardSeq += 1
    }

    private func writeMetadata() throws {
        try Self.persistMetadata(metadata, to: directory)
    }

    /// Load and decode `corpus.json` from a corpus directory **without** opening
    /// the corpus for writing — unlike `open(directory:)`, this has no side
    /// effects (it does not recover/seal any leftover `.open` shard). Used by
    /// read-only consumers such as `CorpusValidator`.
    static func loadMetadata(directory: URL) throws -> CorpusMetadata {
        let metaURL = directory.appendingPathComponent(metadataFilename)
        let data: Data
        do { data = try Data(contentsOf: metaURL) }
        catch { throw GameCorpusError.ioFailed("read corpus.json: \(error.localizedDescription)") }
        do { return try JSONDecoder().decode(CorpusMetadata.self, from: data) }
        catch { throw GameCorpusError.corruptMetadata("corpus.json decode: \(error.localizedDescription)") }
    }

    /// Atomically write `corpus.json` in the canonical on-disk form (pretty,
    /// sorted keys, fsync'd, temp-file + atomic replace). The single writer for
    /// corpus metadata: the instance recorder path and `CorpusValidator`'s
    /// metadata-repair path both go through here so they produce byte-identical
    /// files.
    ///
    /// Replaces `corpus.json` only when it is a regular file (or absent):
    /// `replaceItemAt` over a directory of that name would swap the directory
    /// out of existence. The staging file gets a per-call unique name and is
    /// created exclusively, so failure cleanup removes only the file this call
    /// created — never a pre-existing `corpus.json.tmp`, and never a folder of
    /// that name, which a recursive `removeItem` cleanup would delete. All of
    /// that is `FileSafety.replaceRegularFile`.
    static func persistMetadata(_ metadata: CorpusMetadata, to directory: URL) throws {
        let url = directory.appendingPathComponent(metadataFilename)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data: Data
        do { data = try encoder.encode(metadata) }
        catch { throw GameCorpusError.corruptMetadata("encode corpus.json: \(error.localizedDescription)") }
        do {
            try FileSafety.replaceRegularFile(data, at: url, expectedIdentity: nil)
        } catch {
            throw GameCorpusError.ioFailed("write corpus.json: \(error.localizedDescription)")
        }
    }

    private func recoverOpenShardsIfPresent() throws {
        let fm = FileManager.default
        let entries = try fm.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
        for openURL in Self.openShardURLs(among: entries) {
            try Self.recoverOpenShard(at: openURL)
        }
    }

    /// Recover one unsealed `.open` shard left by a crash: scan it to its
    /// last complete game, truncate the torn tail, and seal it — or, when it
    /// holds no complete game, delete it. Logs the outcome as
    /// `[CORPUS-RECOVERY]` and returns it.
    ///
    /// This rewrites the shard, and nothing distinguishes a crash leftover
    /// from the live shard of a recording or import still appending to it:
    /// run against a corpus that is being written, it truncates and seals that
    /// writer's shard out from under it. Callers — `GameCorpus.open` and
    /// `CorpusValidator`'s `fix` pass — must only run it when nothing is
    /// writing to the corpus. Only a regular file is ever recovered; anything
    /// else with a `.open` name is refused, untouched.
    @discardableResult
    static func recoverOpenShard(at openURL: URL) throws -> OpenShardRecovery {
        guard let item = try FileSafety.existingItem(at: openURL) else {
            throw GameCorpusError.ioFailed("recover \(openURL.lastPathComponent): it no longer exists")
        }
        guard item.kind == .regularFile else {
            throw FileSafetyError.notARegularFile(path: openURL.path, kind: item.kind)
        }
        let scan = try GameCorpusShardIO.scanOpenShard(at: openURL)
        let writer = try ShardWriter(resumingAt: openURL,
                                     validByteCount: scan.validByteCount,
                                     gameCount: scan.gameCount,
                                     plyCount: scan.plyCount)
        let discardedTailBytes = scan.fileSize - scan.validByteCount
        let recovery: OpenShardRecovery
        if scan.gameCount > 0 {
            let sealedURL = try writer.seal(sealUnix: Int64(Date().timeIntervalSince1970))
            recovery = .sealed(openShard: openURL,
                               sealedShard: sealedURL,
                               gameCount: scan.gameCount,
                               plyCount: scan.plyCount,
                               discardedTailBytes: discardedTailBytes)
        } else {
            try writer.discardEmpty()
            recovery = .removedEmpty(openShard: openURL, discardedTailBytes: discardedTailBytes)
        }
        SessionLogger.shared.log("[CORPUS-RECOVERY] \(recovery.summary)")
        return recovery
    }

    private static func highestShardSeq(in directory: URL) throws -> UInt32? {
        let fm = FileManager.default
        let entries = try fm.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
        var maxSeq: UInt32? = nil
        for url in entries {
            let name = url.lastPathComponent
            guard name.hasPrefix("shard-") else { continue }
            let digits = name.dropFirst("shard-".count).prefix { $0.isNumber }
            if let seq = UInt32(digits) {
                maxSeq = Swift.max(maxSeq ?? 0, seq)
            }
        }
        return maxSeq
    }
}
