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

    /// Open an existing corpus directory. Any leftover `.open` shard (from a
    /// crash) is recovered: scanned to its last complete game, truncated, and
    /// sealed, leaving the corpus consistent with only sealed shards.
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
        return entries
            .filter { $0.pathExtension == Self.shardExtension && $0.lastPathComponent.hasPrefix("shard-") }
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
        let url = directory.appendingPathComponent(Self.metadataFilename)
        let tmp = url.appendingPathExtension("tmp")
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data: Data
        do { data = try encoder.encode(metadata) }
        catch { throw GameCorpusError.corruptMetadata("encode corpus.json: \(error.localizedDescription)") }
        do {
            try data.write(to: tmp, options: [.atomic])
            try corpusFullSync(tmp)
            if FileManager.default.fileExists(atPath: url.path) {
                _ = try FileManager.default.replaceItemAt(url, withItemAt: tmp)
            } else {
                try FileManager.default.moveItem(at: tmp, to: url)
            }
        } catch {
            do { try FileManager.default.removeItem(at: tmp) } catch { /* best-effort cleanup */ }
            throw GameCorpusError.ioFailed("write corpus.json: \(error.localizedDescription)")
        }
    }

    private func recoverOpenShardsIfPresent() throws {
        let fm = FileManager.default
        let entries = try fm.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
        let openShards = entries
            .filter { $0.pathExtension == "open" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        for openURL in openShards {
            let scan = try GameCorpusShardIO.scanOpenShard(at: openURL)
            let writer = try ShardWriter(resumingAt: openURL,
                                         validByteCount: scan.validByteCount,
                                         gameCount: scan.gameCount,
                                         plyCount: scan.plyCount)
            if scan.gameCount > 0 {
                _ = try writer.seal(sealUnix: Int64(Date().timeIntervalSince1970))
            } else {
                try writer.discardEmpty()
            }
        }
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
