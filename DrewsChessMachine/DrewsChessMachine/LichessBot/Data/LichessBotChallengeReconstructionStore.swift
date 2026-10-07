import Foundation

/// Keeps `Challenges/reconstructed-from-protocol.json` (challenge-log plan
/// §3.7, "Running it") in step with the protocol log.
///
/// The file is derived and regenerable: a pure function of the protocol day
/// files, the account id and the live log's first entry. So the store
/// rebuilds it only when one of those differs from what the stored file
/// records — its `algorithmVersion`, `ourAccountID` or `liveLogFirstEntryAt`,
/// the list of input files, or any input's size (protocol files only grow)
/// — and writes it only when the new bytes differ from the old. A rerun
/// with unchanged inputs writes nothing, not even a modification time.
///
/// **Two instances.** Both may run this at once against the same folder.
/// Each computes the same bytes from the same inputs, and the one that
/// loses the race to create the file finds it already there
/// (`FileSafetyError.alreadyExists`): it reads the file back and reports
/// `unchanged` when the bytes are its own, or replaces it when they are not
/// (the other instance read the protocol files a moment earlier, before a
/// line was appended). The race never becomes an error or a lost write.
///
/// The file is published whole (`FileSafety.publishNewFile`: staged, then
/// moved into place without replacing), so a reader — including the other
/// instance reading back after losing the race — never sees a partial file.
/// A stored file that doesn't decode (from a crash before this store
/// existed, a newer build's shape, or damage) is simply stale: it is rebuilt
/// and replaced, since nothing in it is the only copy of anything.
///
/// Synchronous: the controller runs it on its general file queue
/// (`LichessBotFileQueue`), never on the main actor. It reads the protocol
/// files and writes only the reconstructed file; it never touches the live
/// challenge log, game records, journals, PGNs or `challenge-outcomes.json`.
/// The caller has run `LichessBotDataDirectory.createDirectories()`, so
/// `Challenges/` exists.
enum LichessBotChallengeReconstructionStore {

    /// What `update` did to the file.
    enum Outcome: String, Sendable, Equatable {
        /// The file was created or replaced with new bytes.
        case written
        /// The file already held exactly these bytes (or was fresh for
        /// these inputs); nothing was written.
        case unchanged
    }

    struct Result: Sendable, Equatable {
        let outcome: Outcome
        let reconstruction: LichessBotChallengeReconstruction
        /// The decoding error of a stored file that was rebuilt because it
        /// didn't decode, for the caller to log; nil when it decoded or
        /// there was none.
        let undecodableStoredFile: String?
    }

    /// One protocol day file as listed, before it is read.
    struct ListedInput: Sendable, Equatable {
        let name: String
        let url: URL
        let byteCount: Int
    }

    /// Bring the reconstructed file up to date and return what it holds.
    /// `ourAccountID` must be the configured account (non-empty);
    /// `liveLogFirstEntryAt` is the `at` of the first line of the oldest
    /// challenge-log day file, or nil when there is no live log.
    static func update(in directory: LichessBotDataDirectory,
                       ourAccountID: String,
                       liveLogFirstEntryAt: Date?) throws -> Result {
        guard !ourAccountID.isEmpty else {
            throw LichessBotChallengeReconstructionError.noAccountID
        }
        let url = directory.reconstructedChallengesURL
        let listed = try listInputs(in: directory, liveLogFirstEntryAt: liveLogFirstEntryAt)
        let stored = try readStored(at: url)
        if let stored, case .decoded(let decoded) = stored.contents,
           isFresh(decoded, listed: listed, ourAccountID: ourAccountID, liveLogFirstEntryAt: liveLogFirstEntryAt) {
            return Result(outcome: .unchanged, reconstruction: decoded, undecodableStoredFile: nil)
        }
        let files = try listed.map { input in
            LichessBotProtocolDayFile(name: input.name, data: try readRegularFile(at: input.url))
        }
        let reconstruction = LichessBotChallengeReconstruction.build(
            from: files, ourAccountID: ourAccountID, liveLogFirstEntryAt: liveLogFirstEntryAt
        )
        let bytes = try reconstruction.encoded()
        let outcome = try writeIfChanged(bytes, at: url, storedBytes: stored?.bytes)
        var undecodableStoredFile: String?
        if let stored, case .undecodable(let reason) = stored.contents {
            undecodableStoredFile = reason
        }
        return Result(outcome: outcome, reconstruction: reconstruction, undecodableStoredFile: undecodableStoredFile)
    }

    /// Whether `stored` is what a rebuild from `listed` would produce: same
    /// algorithm, account and cutoff, and the same files at the same sizes.
    /// Sizes rather than hashes, so the check costs one `lstat` per file:
    /// protocol files are only ever appended to, so an unchanged size means
    /// unchanged bytes.
    static func isFresh(_ stored: LichessBotChallengeReconstruction,
                        listed: [ListedInput],
                        ourAccountID: String,
                        liveLogFirstEntryAt: Date?) -> Bool {
        guard stored.algorithmVersion == LichessBotChallengeReconstruction.currentAlgorithmVersion,
              stored.ourAccountID == ourAccountID,
              stored.liveLogFirstEntryAt.map(LichessBotChallengeReconstruction.timestampText)
                == liveLogFirstEntryAt.map(LichessBotChallengeReconstruction.timestampText) else {
            return false
        }
        return stored.inputs.map(\.file) == listed.map(\.name)
            && stored.inputs.map(\.byteCount) == listed.map(\.byteCount)
    }

    /// The protocol day files the reconstruction reads
    /// (`LichessBotChallengeReconstruction.inputFileNames`), with their
    /// sizes. A missing `Protocol/` folder is no input. A day-file name that
    /// is not a regular file (a link is not followed) is refused, as the
    /// protocol log's own appends refuse it.
    static func listInputs(in directory: LichessBotDataDirectory, liveLogFirstEntryAt: Date?) throws -> [ListedInput] {
        let folder = directory.protocolDirectory
        guard let folderItem = try FileSafety.existingItem(at: folder) else {
            return []
        }
        guard folderItem.kind == .directory else {
            throw LichessBotChallengeReconstructionError.protocolFolderIsNotADirectory(path: folder.path, kind: folderItem.kind)
        }
        let names = LichessBotChallengeReconstruction.inputFileNames(
            from: try FileManager.default.contentsOfDirectory(atPath: folder.path),
            liveLogFirstEntryAt: liveLogFirstEntryAt
        )
        return try names.map { name in
            let url = folder.appendingPathComponent(name, isDirectory: false)
            return ListedInput(name: name, url: url, byteCount: try regularFileSize(at: url))
        }
    }

    // MARK: Private

    private struct Stored {
        enum Contents {
            case decoded(LichessBotChallengeReconstruction)
            /// Stale whatever its inputs, so rebuilt; the decoding error.
            case undecodable(String)
        }

        let bytes: Data
        let contents: Contents
    }

    private static func readStored(at url: URL) throws -> Stored? {
        guard try FileSafety.existingItem(at: url) != nil else { return nil }
        let bytes = try readRegularFile(at: url)
        do {
            return Stored(bytes: bytes, contents: .decoded(try LichessBotChallengeReconstruction.decode(bytes)))
        } catch {
            return Stored(bytes: bytes, contents: .undecodable(String(describing: error)))
        }
    }

    /// Write `bytes` unless the file already holds them. `storedBytes` is
    /// what the file held when read (nil: there was no file).
    static func writeIfChanged(_ bytes: Data, at url: URL, storedBytes: Data?) throws -> Outcome {
        if let storedBytes {
            if storedBytes == bytes {
                return .unchanged
            }
            try FileSafety.replaceRegularFile(bytes, at: url, expectedIdentity: nil)
            return .written
        }
        do {
            try FileSafety.publishNewFile(bytes, to: url)
            return .written
        } catch FileSafetyError.alreadyExists {
            // Another instance created it between the read and the publish.
            if try readRegularFile(at: url) == bytes {
                return .unchanged
            }
            // No expected identity: the file is derived, and whichever
            // instance read the newer protocol bytes should win; refusing to
            // replace the other instance's file would leave the older one.
            try FileSafety.replaceRegularFile(bytes, at: url, expectedIdentity: nil)
            return .written
        }
    }

    /// The bytes of the regular file at `url`; anything else is refused
    /// (a symbolic link is not followed).
    private static func readRegularFile(at url: URL) throws -> Data {
        guard let item = try FileSafety.existingItem(at: url) else {
            throw LichessBotChallengeReconstructionError.inputRemoved(path: url.path)
        }
        guard item.kind == .regularFile else {
            throw FileSafetyError.notARegularFile(path: url.path, kind: item.kind)
        }
        return try Data(contentsOf: url)
    }

    private static func regularFileSize(at url: URL) throws -> Int {
        guard let item = try FileSafety.existingItem(at: url) else {
            throw LichessBotChallengeReconstructionError.inputRemoved(path: url.path)
        }
        guard item.kind == .regularFile else {
            throw FileSafetyError.notARegularFile(path: url.path, kind: item.kind)
        }
        let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
        guard let size = attributes[.size] as? NSNumber else {
            throw LichessBotChallengeReconstructionError.sizeUnavailable(path: url.path)
        }
        return size.intValue
    }

    // MARK: Log line

    /// `[LICHESS-BOT] challenge history reconstructed: …` (§3.7). `games`
    /// is the join of the filed games' ids with the rows
    /// (`LichessBotReconstructedChallengeLookup.gameCounts`). Sizes are
    /// base-2 megabytes.
    static func summaryLine(_ result: Result, games: LichessBotReconstructedGameCounts) -> String {
        let reconstruction = result.reconstruction
        let counts = reconstruction.counts
        let megabytes = String(format: "%.2f", Double(reconstruction.inputByteCount) / 1_048_576)
        return "[LICHESS-BOT] challenge history reconstructed: inputs=\(reconstruction.inputs.count) files (\(megabytes) MB) "
            + "rows=\(reconstruction.rows.count) (outgoing created \(counts.outgoingCreatedRows), not created \(counts.notCreatedRows), incoming \(counts.incomingRows)); "
            + "games: incoming \(games[.incoming]), matchmaking \(games[.matchmaking]), queue \(games[.challengeQueue]), "
            + "casual resend \(games[.matchmakingCasualResend]), operator (inferred) \(games[.operatorInferred]), "
            + "outgoing sender not determined \(games[.outgoingSenderNotDetermined]), unknown \(games[.unknown]); "
            + result.outcome.rawValue
    }
}

enum LichessBotChallengeReconstructionError: LocalizedError, Equatable {
    /// No Lichess account is configured, so no direction can be decided.
    case noAccountID
    case protocolFolderIsNotADirectory(path: String, kind: FileSafety.ItemKind)
    /// A listed file was gone when it was read.
    case inputRemoved(path: String)
    case sizeUnavailable(path: String)

    var errorDescription: String? {
        switch self {
        case .noAccountID:
            return "No Lichess account is configured, so past challenges can't be reconstructed"
        case .protocolFolderIsNotADirectory(let path, let kind):
            return "\(path) is a \(kind), not a folder; past challenges can't be reconstructed from it"
        case .inputRemoved(let path):
            return "\(path) was removed while past challenges were being reconstructed"
        case .sizeUnavailable(let path):
            return "the size of \(path) could not be read"
        }
    }
}
