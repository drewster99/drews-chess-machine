import Foundation

/// Severity of a single corpus-validation finding.
enum CorpusValidationSeverity: String, Sendable {
    /// A data/integrity problem — the corpus is not trustworthy as-is. Never
    /// auto-fixed (fixing would mean fabricating or discarding game data).
    case error
    /// A metadata inconsistency or oddity — usually derivable from the shards
    /// and therefore auto-fixable.
    case warning
    /// A benign note, no action needed.
    case info
}

/// One issue found while validating a corpus.
struct CorpusValidationFinding: Sendable {
    var severity: CorpusValidationSeverity
    /// Stable slug for the kind of problem (e.g. `source-game-count`).
    var code: String
    /// Human-readable description including the specific values.
    var message: String
    /// Whether `validate(fix: true)` can repair this.
    var fixable: Bool
    /// Set true once a fix pass has repaired it.
    var fixed: Bool = false
}

/// The outcome of validating one corpus directory.
struct CorpusValidationReport: Sendable {
    var corpusID: String
    var directory: URL
    /// Number of `shard-*.dcmgames` files found.
    var shardCount: Int
    /// Games summed across all readable shard trailers.
    var totalGames: Int
    /// Plies summed across all readable shard trailers.
    var totalPlies: Int
    /// Whether shard bodies were SHA-256 + per-record-CRC verified (vs a fast
    /// header/trailer-only read).
    var integrityVerified: Bool
    var findings: [CorpusValidationFinding]

    var errorCount: Int { findings.filter { $0.severity == .error }.count }
    var warningCount: Int { findings.filter { $0.severity == .warning }.count }
    /// Errors + warnings that were not fixed. `info` never counts against validity.
    var unresolvedProblemCount: Int {
        findings.filter { ($0.severity == .error || $0.severity == .warning) && !$0.fixed }.count
    }
    /// Valid when nothing above `info` remains unresolved.
    var isValid: Bool { unresolvedProblemCount == 0 }
}

/// Validates a `Corpora/` corpus directory and optionally repairs its metadata.
///
/// Checks two independent things: (1) **shard integrity** — every sealed shard's
/// front magic/version, corpus-ID stamp, trailer magic, and (in full mode) its
/// whole-shard SHA-256 and every record's CRC; and (2) **metadata consistency**
/// — that `corpus.json`'s per-source `gamesAdded`/`pliesAdded` match what the
/// shards actually hold, that every shard's `sourceID` is declared, that shard
/// sequence numbers line up, and that the recording state is coherent.
///
/// A crash or a disk-full recording can leave two repairable things behind:
///
/// - **Unsealed `.open` shards.** A recording or import that died mid-shard
///   leaves its shard `.open`, with a possibly torn last record. Nothing else
///   recovers these: corpus replay opens corpora read-only (it cannot tell a
///   crash leftover from a live writer's shard), and no app path reopens an
///   existing corpus for writing. `fix: true` is the explicit place they are
///   recovered, through the same code `GameCorpus.open` uses
///   (`GameCorpus.recoverOpenShard`): each is truncated to its last complete
///   game and sealed, or deleted when it holds no complete game, and each
///   outcome is logged and reported. Like that recovery, a `fix` run must only
///   be made when nothing is writing to the corpus. Without `fix` they are
///   only reported, byte-for-byte untouched.
/// - **Stale `corpus.json` counts** (e.g. `gamesAdded: 0`) even though the
///   sealed shards are intact — re-derivable, so `fix: true` recomputes the
///   per-source counts from the shard trailers (after any `.open` recovery,
///   so recovered games are counted) and rewrites `corpus.json`.
///
/// **Sealed shard bytes are never modified** — a genuine data problem (bad
/// SHA/CRC, missing shards, corpus-ID mismatch) is reported as an `error`,
/// never silently "repaired".
enum CorpusValidator {

    /// Validate the corpus rooted at `directory`.
    ///
    /// - Parameters:
    ///   - directory: the corpus folder (contains `corpus.json` + shards).
    ///   - verifyIntegrity: when `true` (default) every sealed shard body is read
    ///     and SHA-256 + per-record CRC verified — authoritative but O(corpus
    ///     bytes). When `false`, only the fixed-size front header and trailer of
    ///     each shard are read (fast, counts-only, no body integrity).
    ///   - fix: when `true`, repair the fixable findings: recover every `.open`
    ///     shard (seal it at its last complete game, or delete it when it holds
    ///     none — only safe when nothing is writing to the corpus), then
    ///     rewrite per-source `gamesAdded`/`pliesAdded` from the shard trailers
    ///     and persist `corpus.json`. Never modifies a sealed shard.
    /// - Returns: a structured report. Throws when `corpus.json` itself cannot
    ///   be read or decoded, or when the corpus directory cannot be listed
    ///   (without either, there is nothing trustworthy to validate against — a
    ///   permission or path failure here must surface, not be swallowed into an
    ///   empty listing that would masquerade as "no shards present"); every
    ///   other problem is a finding, not a throw.
    @discardableResult
    static func validate(directory: URL,
                         verifyIntegrity: Bool = true,
                         fix: Bool = false) throws -> CorpusValidationReport {
        var metadata = try GameCorpus.loadMetadata(directory: directory)
        var findings: [CorpusValidationFinding] = []

        func listEntries() throws -> [URL] {
            do {
                return try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
            } catch {
                throw GameCorpusError.ioFailed("list corpus directory \(directory.path): \(error.localizedDescription)")
            }
        }
        var entries = try listEntries()

        if metadata.formatVersion != CorpusMetadata.currentFormatVersion {
            // Informational, not a validity failure: a routine bump of
            // `currentFormatVersion` must not flip every previously-written corpus
            // to invalid. The shards are still readable — the version delta is a
            // note, not a data problem.
            findings.append(CorpusValidationFinding(
                severity: .info, code: "format-version",
                message: "corpus.json formatVersion is \(metadata.formatVersion), expected \(CorpusMetadata.currentFormatVersion)",
                fixable: false))
        }
        if metadata.state != "recording" && metadata.state != "sealed" {
            findings.append(CorpusValidationFinding(
                severity: .warning, code: "unknown-state",
                message: "corpus state '\(metadata.state)' is neither 'recording' nor 'sealed'",
                fixable: false))
        }

        // Leftover open shards: a crashed recording or import, or the live shard
        // of one still running. Their presence alongside a "sealed" state is
        // contradictory. Recovered only by `fix` (see the type doc); the
        // listing is redone afterwards so the shards it sealed are validated
        // and counted below like any other.
        let openShards = GameCorpus.openShardURLs(among: entries)
        let openShardSeverity: CorpusValidationSeverity = metadata.state == "sealed" ? .error : .warning
        if !openShards.isEmpty {
            if fix {
                for openURL in openShards {
                    do {
                        let recovery = try GameCorpus.recoverOpenShard(at: openURL)
                        findings.append(CorpusValidationFinding(
                            severity: openShardSeverity, code: "open-shard-present",
                            message: "recovered unsealed shard \(recovery.summary)",
                            fixable: true, fixed: true))
                    } catch {
                        SessionLogger.shared.log(
                            "[CORPUS-RECOVERY] \(openURL.lastPathComponent): NOT recovered — \(error.localizedDescription)")
                        findings.append(CorpusValidationFinding(
                            severity: .error, code: "open-shard-unrecoverable",
                            message: "\(openURL.lastPathComponent): could not be recovered: \(error.localizedDescription)",
                            fixable: false))
                    }
                }
                entries = try listEntries()
            } else {
                let names = openShards.map { $0.lastPathComponent }.joined(separator: ", ")
                findings.append(CorpusValidationFinding(
                    severity: openShardSeverity,
                    code: "open-shard-present",
                    message: "\(openShards.count) unsealed .open shard(s) present (\(names)); their games are not "
                        + "readable until recovered. Each is a crash leftover or the live shard of a recording/import "
                        + "still writing; once nothing is writing to this corpus, --fix seals each at its last "
                        + "complete game (or removes one that holds none)",
                    fixable: true))
            }
        }
        let tmpFiles = entries.filter { $0.pathExtension == "tmp" }
        if !tmpFiles.isEmpty {
            findings.append(CorpusValidationFinding(
                severity: .warning, code: "stray-temp-file",
                message: "leftover temp file(s): \(tmpFiles.map { $0.lastPathComponent }.sorted().joined(separator: ", "))",
                fixable: false))
        }

        let shardURLs = GameCorpus.sealedShardURLs(among: entries)

        func filenameSeq(_ url: URL) -> Int? {
            let digits = url.lastPathComponent.dropFirst("shard-".count).prefix { $0.isNumber }
            return Int(digits)
        }

        var totalGames = 0
        var totalPlies = 0
        var perSource: [String: (games: Int, plies: Int)] = [:]
        var seenSeqs: [Int] = []

        for url in shardURLs {
            let name = url.lastPathComponent
            let header: GameCorpusShardFormat.FrontHeader
            let gameCount: Int
            let plyCount: Int
            do {
                if verifyIntegrity {
                    let sealed = try GameCorpusShardIO.readSealed(at: url)
                    header = sealed.header; gameCount = sealed.gameCount; plyCount = sealed.plyCount
                } else {
                    let counts = try GameCorpusShardIO.readSealedHeaderAndCounts(at: url)
                    header = counts.header; gameCount = counts.gameCount; plyCount = counts.plyCount
                }
            } catch {
                findings.append(CorpusValidationFinding(
                    severity: .error, code: "shard-unreadable",
                    message: "\(name): \(error.localizedDescription)", fixable: false))
                continue
            }

            totalGames += gameCount
            totalPlies += plyCount
            perSource[header.sourceID, default: (0, 0)].games += gameCount
            perSource[header.sourceID, default: (0, 0)].plies += plyCount
            if let seq = filenameSeq(url) { seenSeqs.append(seq) }

            if header.corpusID != metadata.corpusID {
                findings.append(CorpusValidationFinding(
                    severity: .error, code: "shard-corpus-id-mismatch",
                    message: "\(name): front-header corpusID '\(header.corpusID)' != corpus.json '\(metadata.corpusID)'",
                    fixable: false))
            }
            if let seq = filenameSeq(url), Int(header.shardSeq) != seq {
                findings.append(CorpusValidationFinding(
                    severity: .warning, code: "shard-seq-filename-mismatch",
                    message: "\(name): front-header shardSeq \(header.shardSeq) != filename index \(seq)",
                    fixable: false))
            }
            if !metadata.sources.contains(where: { $0.sourceID == header.sourceID }) {
                findings.append(CorpusValidationFinding(
                    severity: .warning, code: "orphan-shard-source",
                    message: "\(name): sourceID '\(header.sourceID)' has no entry in corpus.json sources",
                    fixable: false))
            }
        }

        // Sequence duplicates are always wrong; gaps can be benign (a discarded
        // empty shard still consumes a sequence number), so they are info only.
        let sortedSeqs = seenSeqs.sorted()
        let dupCounts = Dictionary(sortedSeqs.map { ($0, 1) }, uniquingKeysWith: +)
        let dups = dupCounts.filter { $0.value > 1 }.keys.sorted()
        if !dups.isEmpty {
            findings.append(CorpusValidationFinding(
                severity: .error, code: "duplicate-shard-seq",
                message: "duplicate shard sequence number(s): \(dups.map(String.init).joined(separator: ", "))",
                fixable: false))
        }
        if let lo = sortedSeqs.first, let hi = sortedSeqs.last {
            let missing = Set(lo...hi).subtracting(sortedSeqs).sorted()
            if !missing.isEmpty {
                findings.append(CorpusValidationFinding(
                    severity: .info, code: "shard-sequence-gap",
                    message: "gap(s) in shard sequence \(lo)...\(hi): missing \(missing.map(String.init).joined(separator: ", ")) (benign if an empty shard was discarded)",
                    fixable: false))
            }
        }

        // Per-source count reconciliation — the fixable class.
        var fixedAny = false
        for i in metadata.sources.indices {
            let sid = metadata.sources[i].sourceID
            guard let actual = perSource[sid] else {
                if (metadata.sources[i].gamesAdded ?? 0) > 0 {
                    findings.append(CorpusValidationFinding(
                        severity: .error, code: "source-shards-missing",
                        message: "source \(sid) claims \(metadata.sources[i].gamesAdded ?? 0) games but no shards for it are present",
                        fixable: false))
                }
                continue
            }
            if metadata.sources[i].gamesAdded != actual.games {
                var f = CorpusValidationFinding(
                    severity: .warning, code: "source-game-count",
                    message: "source \(sid): corpus.json gamesAdded=\(metadata.sources[i].gamesAdded.map(String.init) ?? "nil") but shards hold \(actual.games)",
                    fixable: true)
                if fix { metadata.sources[i].gamesAdded = actual.games; f.fixed = true; fixedAny = true }
                findings.append(f)
            }
            if metadata.sources[i].pliesAdded != actual.plies {
                var f = CorpusValidationFinding(
                    severity: .warning, code: "source-ply-count",
                    message: "source \(sid): corpus.json pliesAdded=\(metadata.sources[i].pliesAdded.map(String.init) ?? "nil") but shards hold \(actual.plies)",
                    fixable: true)
                if fix { metadata.sources[i].pliesAdded = actual.plies; f.fixed = true; fixedAny = true }
                findings.append(f)
            }
        }

        if metadata.state == "sealed" {
            for source in metadata.sources where source.complete != true {
                findings.append(CorpusValidationFinding(
                    severity: .warning, code: "sealed-source-incomplete",
                    message: "corpus is sealed but source \(source.sourceID) is not marked complete",
                    fixable: false))
            }
        }

        if fix && fixedAny {
            try GameCorpus.persistMetadata(metadata, to: directory)
        }

        return CorpusValidationReport(
            corpusID: metadata.corpusID,
            directory: directory,
            shardCount: shardURLs.count,
            totalGames: totalGames,
            totalPlies: totalPlies,
            integrityVerified: verifyIntegrity,
            findings: findings)
    }
}
