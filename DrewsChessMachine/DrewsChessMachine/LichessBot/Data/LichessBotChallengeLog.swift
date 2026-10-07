import Foundation

// The challenge log (challenge-log plan §3.1–§3.3): an append-only record of
// every challenge's lifecycle, in both directions,
// `Challenges/challenges-YYYYMMDD.jsonl`, one file per UTC day, kept
// forever. It is the record of challenges; the in-memory
// `LichessBotChallengeLedger` is its fold.
//
// The types here are a persistence schema, kept separate from the API models
// (as the journal's are) so the file format changes only deliberately. Values
// Lichess defines stay open (`LichessBotOpenValue`), so an unfamiliar value
// is kept verbatim rather than refused.

// MARK: - Line schema

/// One line of `Challenges/challenges-YYYYMMDD.jsonl`.
struct LichessBotChallengeLogEntry: Sendable, Codable, Equatable {
    /// Bumped whenever a case or a field is added, so an older build can
    /// tell "written by a newer build" (skipped and counted) from
    /// corruption. Unlike the journal, where a newer case makes the whole
    /// file undecodable, one newer line here must not cost a whole day.
    static let currentSchemaVersion = 1

    let schemaVersion: Int
    let at: Date
    /// The writing build (`BuildInfo.buildNumber`, `BuildInfo.gitHash`).
    let build: Int
    let gitHash: String
    let event: LichessBotChallengeLogEvent

    /// An entry written by this build.
    init(at: Date, event: LichessBotChallengeLogEvent) {
        self.init(schemaVersion: Self.currentSchemaVersion, at: at, build: BuildInfo.buildNumber, gitHash: BuildInfo.gitHash, event: event)
    }

    init(schemaVersion: Int, at: Date, build: Int, gitHash: String, event: LichessBotChallengeLogEvent) {
        self.schemaVersion = schemaVersion
        self.at = at
        self.build = build
        self.gitHash = gitHash
        self.event = event
    }
}

/// One lifecycle fact of a challenge.
enum LichessBotChallengeLogEvent: Sendable, Codable, Equatable {
    // Outgoing: DCM's own sends.
    /// Lichess created the challenge (the POST answered with it).
    case outgoingCreated(challenge: LichessBotChallengeSnapshot, sender: LichessBotChallengeSender,
                         request: LichessBotOutgoingChallenge, opponentKind: LichessBotChallengeOpponentKind, creditCost: Int)
    /// A send that created no challenge. `attemptID` names the attempt
    /// (there is no challenge id); `opponentID` is lowercased.
    case outgoingNotCreated(attemptID: UUID, opponentID: String, sender: LichessBotChallengeSender,
                            request: LichessBotOutgoingChallenge, opponentKind: LichessBotChallengeOpponentKind?,
                            reason: LichessBotChallengeNotCreatedReason, creditCost: Int)
    /// DCM's own challenge, echoed on the event stream, with no created
    /// line for it in the run that saw it (an unmatched echo, §3.4).
    case outgoingSeenWithoutCreatedLine(challenge: LichessBotChallengeSnapshot, attribution: LichessBotEchoAttribution)
    case withdrawalRequested(challengeID: String, reason: LichessBotWithdrawalReason)
    case withdrawalResult(challengeID: String, result: LichessBotWithdrawalResult)
    // Incoming.
    case incomingReceived(challenge: LichessBotChallengeSnapshot)
    case incomingDecided(challengeID: String, decision: LichessBotIncomingDecisionRecord)
    case incomingResponseFailed(challengeID: String, error: String)
    // Either direction, as Lichess reported it on the event stream.
    case declinedOnLichess(challengeID: String, reason: LichessBotDeclineReasonRecord, text: String?)
    case canceledOnLichess(challengeID: String)
    /// A `gameStart` whose game id is this challenge's id.
    case gameStarted(challengeID: String)
    // The writer's own housekeeping.
    /// An unterminated final line left by an interrupted write was cut
    /// before this append; its bytes are kept here.
    case unterminatedLineCut(byteCount: Int, base64: String)
}

/// One side of a challenge, as Lichess described it.
struct LichessBotChallengeParty: Sendable, Codable, Equatable, Hashable {
    let id: String
    let name: String
    let title: String?
    let rating: Int?
    let provisional: Bool?

    init(id: String, name: String, title: String?, rating: Int?, provisional: Bool?) {
        self.id = id
        self.name = name
        self.title = title
        self.rating = rating
        self.provisional = provisional
    }

    init(_ user: LichessBotChallengeUser) {
        self.init(id: user.id, name: user.name, title: user.title, rating: user.rating, provisional: user.provisional)
    }
}

/// A challenge's terms and parties, as Lichess sent them.
struct LichessBotChallengeSnapshot: Sendable, Codable, Equatable, Hashable {
    let id: String
    let challenger: LichessBotChallengeParty
    /// Nil for an open challenge, which names no opponent.
    let destUser: LichessBotChallengeParty?
    let variant: LichessBotOpenValue<LichessBotVariantKey>
    let rated: Bool
    let speed: LichessBotOpenValue<LichessBotSpeed>
    let timeControlType: LichessBotOpenValue<LichessBotTimeControlType>
    let limitSeconds: Int?
    let incrementSeconds: Int?
    let daysPerTurn: Int?
    let color: LichessBotOpenValue<LichessBotChallengeColorName>
    let finalColor: LichessBotOpenValue<LichessBotColorName>?
    let initialFen: String?
    let rematchOf: String?

    init(id: String, challenger: LichessBotChallengeParty, destUser: LichessBotChallengeParty?,
         variant: LichessBotOpenValue<LichessBotVariantKey>, rated: Bool, speed: LichessBotOpenValue<LichessBotSpeed>,
         timeControlType: LichessBotOpenValue<LichessBotTimeControlType>, limitSeconds: Int?, incrementSeconds: Int?,
         daysPerTurn: Int?, color: LichessBotOpenValue<LichessBotChallengeColorName>,
         finalColor: LichessBotOpenValue<LichessBotColorName>?, initialFen: String?, rematchOf: String?) {
        self.id = id
        self.challenger = challenger
        self.destUser = destUser
        self.variant = variant
        self.rated = rated
        self.speed = speed
        self.timeControlType = timeControlType
        self.limitSeconds = limitSeconds
        self.incrementSeconds = incrementSeconds
        self.daysPerTurn = daysPerTurn
        self.color = color
        self.finalColor = finalColor
        self.initialFen = initialFen
        self.rematchOf = rematchOf
    }

    /// The one mapping from the API model.
    init(_ challenge: LichessBotChallenge) {
        self.init(
            id: challenge.id,
            challenger: LichessBotChallengeParty(challenge.challenger),
            destUser: challenge.destUser.map(LichessBotChallengeParty.init),
            variant: challenge.variant.key,
            rated: challenge.rated,
            speed: challenge.speed,
            timeControlType: challenge.timeControl.type,
            limitSeconds: challenge.timeControl.limit?.value,
            incrementSeconds: challenge.timeControl.increment?.value,
            daysPerTurn: challenge.timeControl.daysPerTurn,
            color: challenge.color,
            finalColor: challenge.finalColor,
            initialFen: challenge.initialFen,
            rematchOf: challenge.rematchOf
        )
    }
}

/// Who sent one of DCM's challenges.
enum LichessBotChallengeSender: Sendable, Codable, Equatable, Hashable {
    /// The operator, from the Challenge sheet.
    case challengeSheet
    /// The operator's Resend as Casual after a rated challenge was declined.
    case casualResendOffer
    /// The operator's challenge queue.
    case challengeQueue
    /// Matchmaking, and what started the pass.
    case matchmaking(trigger: LichessBotMatchmakingTrigger, fillMode: LichessBotMatchmakingSettings.FillMode)
    /// Matchmaking's automatic casual resend after a rated decline.
    case matchmakingCasualResend
}

/// What started a matchmaking pass (OD-13: otherwise unknowable afterwards).
enum LichessBotMatchmakingTrigger: String, Sendable, Codable, Equatable, Hashable, CaseIterable {
    case automaticPass
    case fillOpenSlots
}

/// Why a send created no challenge.
enum LichessBotChallengeNotCreatedReason: Sendable, Codable, Equatable, Hashable {
    /// The player was offline when DCM checked, so nothing was posted.
    case opponentOffline
    /// Lichess answered the POST with a refusal.
    case refused(LichessBotChallengeRefusal)
    /// Lichess's answer never arrived, so a challenge may exist after all;
    /// its echo, if one comes, is attributed to this send (§3.4).
    case noAnswer(error: String)
}

/// Who sent a challenge seen only as its echo.
enum LichessBotEchoAttribution: Sendable, Codable, Equatable, Hashable {
    /// Exactly one send to that player in the window had no answer.
    case unansweredSend(attemptID: UUID, sender: LichessBotChallengeSender)
    /// No send of this run explains it.
    case notRecorded
}

/// Why DCM withdrew one of its challenges.
enum LichessBotWithdrawalReason: Sendable, Codable, Equatable, Hashable {
    case operatorCancel
    case unansweredTimeout(seconds: Int)
    case goingOffline
    /// Created while DCM was going offline, and withdrawn at once.
    case wentOfflineWhileSending
}

/// What Lichess answered to a withdrawal.
enum LichessBotWithdrawalResult: Sendable, Codable, Equatable, Hashable {
    case confirmed
    /// Lichess answered 400/404: expired, or already answered.
    case alreadyGone(message: String?)
    case failed(error: String)
    /// The app shut down before an answer came.
    case abandonedAtShutdown
}

/// DCM's decision on an incoming challenge, as persisted.
enum LichessBotIncomingDecisionRecord: Sendable, Codable, Equatable, Hashable {
    case accept
    case decline(reason: LichessBotDeclineReason, rule: String)
    case ignore(rule: String)

    /// The one mapping from the policy's decision.
    init(_ decision: LichessBotChallengeDecision) {
        switch decision {
        case .accept:
            self = .accept
        case .decline(let reason, let rule):
            self = .decline(reason: reason, rule: rule)
        case .ignore(let rule):
            self = .ignore(rule: rule)
        }
    }
}

// MARK: - Reading

/// What reading the challenge log's day files found (§3.2's decoding rule).
struct LichessBotChallengeLogContents: Sendable, Equatable {
    /// A day file that was read.
    struct DayFile: Sendable, Equatable {
        let name: String
        let byteCount: Int
        /// Complete lines, including skipped ones; blank lines not counted.
        let lineCount: Int
        /// Lines written by a newer build (a higher `schemaVersion`):
        /// skipped, never treated as corruption.
        let skippedNewerLines: Int
        /// An unterminated final line (an interrupted append), dropped. The
        /// next append cuts it and records it as `unterminatedLineCut`.
        let droppedTrailingByteCount: Int
        /// The time of the file's first entry this build could read; nil
        /// when it has none.
        let firstEntryAt: Date?
    }

    /// A day file left out of the contents, and why.
    struct LeftOutFile: Sendable, Equatable {
        let name: String
        let reason: String
    }

    /// Every entry of every file read, in file-name order (oldest day
    /// first), then line order.
    var entries: [LichessBotChallengeLogEntry] = []
    var filesRead: [DayFile] = []
    var filesLeftOut: [LeftOutFile] = []

    /// When the live log begins: the reconstruction's cutoff (§3.7). The
    /// first entry of the oldest day file with one; nil when no day file
    /// holds an entry (no live log yet). When a day file older than that
    /// was left out (corrupt), its first entry can't be read, so the start
    /// of its UTC day stands in: the cutoff is then never later than the
    /// real first entry, so a fact the live log may hold is never also
    /// rebuilt from the protocol log (at worst a few rebuilt facts of that
    /// day are missing until the file is repaired).
    var liveLogFirstEntryAt: Date? {
        let firstRead = filesRead.first { $0.firstEntryAt != nil }
        let olderLeftOut = filesLeftOut
            .filter { leftOut in firstRead.map { leftOut.name < $0.name } ?? true }
            .compactMap { LichessBotChallengeLog.utcDayStart(ofDayFileName: $0.name) }
            .min()
        switch (firstRead?.firstEntryAt, olderLeftOut) {
        case (let read?, let leftOut?): return min(read, leftOut)
        case (let read?, nil): return read
        case (nil, let leftOut?): return leftOut
        case (nil, nil): return nil
        }
    }

    var lineCount: Int { filesRead.reduce(0) { $0 + $1.lineCount } }
    var byteCount: Int { filesRead.reduce(0) { $0 + $1.byteCount } }
    var skippedNewerLines: Int { filesRead.reduce(0) { $0 + $1.skippedNewerLines } }
}

/// One day file's lines, decoded by §3.2's rule.
struct LichessBotChallengeLogDayFileDecoding: Sendable, Equatable {
    let entries: [LichessBotChallengeLogEntry]
    let lineCount: Int
    let skippedNewerLines: Int
    let droppedTrailingByteCount: Int
}

enum LichessBotChallengeLogError: LocalizedError, Equatable {
    /// `Challenges/` exists but is not a folder (a symbolic link is not
    /// followed), so the log can't be listed.
    case folderIsNotADirectory(path: String, kind: FileSafety.ItemKind)

    var errorDescription: String? {
        switch self {
        case .folderIsNotADirectory(let path, let kind):
            return "\(path) is a \(kind), not a folder; the challenge log can't be read from it"
        }
    }
}

// MARK: - Writer

/// Writes the challenge log, and reads it back.
///
/// `record` is synchronous, so the controller's one funnel can call it in
/// the same step as its in-memory update; the append is enqueued on the
/// general file queue (`LichessBotFileQueue`, the one the protocol log uses),
/// so entries land in the order they were recorded and nothing runs on the
/// main actor or a cooperative-pool thread. Nothing on the play path waits on
/// that queue (journals have their own).
///
/// **Durability: `F_FULLFSYNC` after every append** (OD-3), and the folder
/// too when an append creates a day file, so the new file's name survives a
/// power loss with its contents. Challenge facts are rare (Lichess caps sends
/// at 25 a minute) and a full sync measured a few milliseconds here, and this
/// is the record the owner wants kept, so the protocol log's no-sync risk is
/// not repeated. Each sync is timed: one over `slowSyncThreshold` is logged,
/// and `logSummary()` reports the count and the slowest.
///
/// Appends go through `LichessBotJSONLines.append`, which takes the day
/// file's lock for its tail check, cut and write: two instances can append
/// to the same day file (a withdrawal's result can land after another
/// instance has taken over the bot), and their lines then interleave whole.
/// A fragment an interrupted append left is cut and recorded first, as an
/// `unterminatedLineCut` line ahead of the entry, and raised as an alarm.
///
/// A failed append (or an entry that fails to encode) is reported through
/// `onWriteFailure`, which the controller raises as an alarm; nothing is
/// dropped silently. An entry recorded after the file queue was closed is
/// not written; the queue writes the refusal to the session log with the
/// entry's full line, so it is still on record.
final class LichessBotChallengeLog: Sendable {
    private let directory: LichessBotDataDirectory
    private let fileQueue: LichessBotFileQueue
    private let systemCalls: LichessBotJSONLines.AppendSystemCalls
    private let onWriteFailure: @Sendable (Error) -> Void
    /// Append count and slowest sync, for `logSummary()`. Changed only on
    /// the file queue; read anywhere.
    private let syncStatistics = SyncBox(SyncStatistics())

    /// A sync slower than this is logged as it happens.
    static let slowSyncThreshold: TimeInterval = 0.1

    struct SyncStatistics: Sendable, Equatable {
        var appendCount = 0
        var slowestSyncSeconds: TimeInterval = 0
    }

    /// `systemCalls` is `.system` in production; tests pass recording calls.
    init(directory: LichessBotDataDirectory,
         fileQueue: LichessBotFileQueue,
         systemCalls: LichessBotJSONLines.AppendSystemCalls,
         onWriteFailure: @escaping @Sendable (Error) -> Void) {
        self.directory = directory
        self.fileQueue = fileQueue
        self.systemCalls = systemCalls
        self.onWriteFailure = onWriteFailure
    }

    /// Append `event`, timestamped `at`, to the day file for `at`'s UTC day.
    func record(_ event: LichessBotChallengeLogEvent, at: Date) {
        let entry = LichessBotChallengeLogEntry(at: at, event: event)
        let line: Data
        do {
            line = try LichessBotJSONLines.encodeLine(entry)
        } catch {
            onWriteFailure(error)
            return
        }
        let url = directory.challengeLogURL(for: at)
        let onWriteFailure = self.onWriteFailure
        let systemCalls = timed(self.systemCalls)
        let syncStatistics = self.syncStatistics
        let lineText = String(decoding: line.dropLast(), as: UTF8.self)
        fileQueue.enqueue("challenge log entry \(lineText)") {
            do {
                try LichessBotJSONLines.append(to: url, synchronization: .fullSync, systemCalls: systemCalls) { cut in
                    var data = Data()
                    if !cut.isEmpty {
                        SessionLogger.shared.log("[ALARM] LICHESS-BOT \(url.lastPathComponent): cut an unterminated final line of \(cut.count) bytes left by an interrupted write; base64 \(cut.base64EncodedString())")
                        data.append(try LichessBotJSONLines.encodeLine(LichessBotChallengeLogEntry(
                            at: Date(),
                            event: .unterminatedLineCut(byteCount: cut.count, base64: cut.base64EncodedString())
                        )))
                    }
                    data.append(line)
                    return data
                }
                syncStatistics.modify { $0.appendCount += 1 }
            } catch {
                onWriteFailure(error)
            }
        }
    }

    /// `calls` with its two syncs timed into `syncStatistics`, each one over
    /// `slowSyncThreshold` logged.
    private func timed(_ calls: LichessBotJSONLines.AppendSystemCalls) -> LichessBotJSONLines.AppendSystemCalls {
        let syncStatistics = self.syncStatistics
        @Sendable func measure(_ what: String, _ body: () throws -> Void) rethrows {
            let start = ContinuousClock.now
            try body()
            let elapsed = ContinuousClock.now - start
            let seconds = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) / 1e18
            syncStatistics.modify { $0.slowestSyncSeconds = max($0.slowestSyncSeconds, seconds) }
            if seconds > Self.slowSyncThreshold {
                SessionLogger.shared.log("[LICHESS-BOT] challenge log: slow sync \(String(format: "%.1f", seconds * 1000)) ms (\(what))")
            }
        }
        return LichessBotJSONLines.AppendSystemCalls(
            write: calls.write,
            fsync: calls.fsync,
            fullSync: { handle, path in try measure("file") { try calls.fullSync(handle, path) } },
            fullSyncDirectory: { folder in try measure("folder") { try calls.fullSyncDirectory(folder) } }
        )
    }

    /// Appends made so far and the slowest sync.
    var statistics: SyncStatistics {
        syncStatistics.value
    }

    /// Log the appends made and the slowest sync (at shutdown).
    func logSummary() {
        let statistics = syncStatistics.value
        SessionLogger.shared.log("[LICHESS-BOT] challenge log: \(statistics.appendCount) appends, slowest sync \(String(format: "%.1f", statistics.slowestSyncSeconds * 1000)) ms")
    }

    /// Wait until every entry recorded so far has been appended (or has
    /// failed). Throws `LichessBotFileQueueError.closed` once the queue is
    /// closed.
    func flush() async throws {
        try await fileQueue.run {}
    }

    /// Read every day file, on the file queue (so it sees every append
    /// enqueued before it, and none after).
    func readAll() async throws -> LichessBotChallengeLogContents {
        let directory = self.directory
        return try await fileQueue.run {
            try Self.readAll(in: directory)
        }
    }

    // MARK: Reading (pure, synchronous)

    /// `challenges-YYYYMMDD.jsonl`, exactly: what `challengeLogURL(for:)`
    /// names. Anything else in `Challenges/` (the reconstructed history) is
    /// not a day file.
    static func isDayFileName(_ name: String) -> Bool {
        let prefix = "challenges-"
        let suffix = ".jsonl"
        guard name.hasPrefix(prefix), name.hasSuffix(suffix) else { return false }
        let stamp = name.dropFirst(prefix.count).dropLast(suffix.count)
        return stamp.utf8.count == 8 && stamp.utf8.allSatisfy { (UInt8(ascii: "0")...UInt8(ascii: "9")).contains($0) }
    }

    /// Read every day file in `directory.challengesDirectory`, oldest day
    /// first. A missing folder is an empty log. A day file that is not a
    /// regular file, can't be read, or holds a corrupt line is left out
    /// (named, with the reason), and the rest are still read. Throws only
    /// when the folder itself can't be listed.
    static func readAll(in directory: LichessBotDataDirectory) throws -> LichessBotChallengeLogContents {
        let folder = directory.challengesDirectory
        guard let folderItem = try FileSafety.existingItem(at: folder) else {
            return LichessBotChallengeLogContents()
        }
        guard folderItem.kind == .directory else {
            throw LichessBotChallengeLogError.folderIsNotADirectory(path: folder.path, kind: folderItem.kind)
        }
        let names = try FileManager.default.contentsOfDirectory(atPath: folder.path).filter(isDayFileName).sorted()
        var contents = LichessBotChallengeLogContents()
        for name in names {
            let url = folder.appendingPathComponent(name, isDirectory: false)
            do {
                guard let item = try FileSafety.existingItem(at: url) else {
                    contents.filesLeftOut.append(.init(name: name, reason: "removed while the log was being read"))
                    continue
                }
                guard item.kind == .regularFile else {
                    throw FileSafetyError.notARegularFile(path: url.path, kind: item.kind)
                }
                let data = try Data(contentsOf: url)
                let decoded = try decodeDayFile(data, fileName: name)
                contents.entries.append(contentsOf: decoded.entries)
                contents.filesRead.append(.init(
                    name: name, byteCount: data.count, lineCount: decoded.lineCount,
                    skippedNewerLines: decoded.skippedNewerLines, droppedTrailingByteCount: decoded.droppedTrailingByteCount,
                    firstEntryAt: decoded.entries.first?.at
                ))
            } catch {
                contents.filesLeftOut.append(.init(name: name, reason: error.localizedDescription))
            }
        }
        return contents
    }

    /// 00:00 UTC of the day a `challenges-YYYYMMDD.jsonl` name stands for,
    /// or nil for any other name.
    static func utcDayStart(ofDayFileName name: String) -> Date? {
        guard isDayFileName(name) else { return nil }
        let stamp = String(name.dropFirst("challenges-".count).prefix(8))
        guard let year = Int(stamp.prefix(4)), let month = Int(stamp.dropFirst(4).prefix(2)), let day = Int(stamp.suffix(2)) else {
            return nil
        }
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = .gmt
        return calendar.date(from: DateComponents(year: year, month: month, day: day))
    }

    /// The part of a line decoded first, so a newer build's line is told
    /// from a corrupt one before its event is looked at.
    private struct SchemaVersionProbe: Decodable {
        let schemaVersion: Int
    }

    /// One day file's lines, by §3.2's decoding rule:
    /// 1. each line's `schemaVersion` is decoded first;
    /// 2. a line with a higher version than this build writes is skipped
    ///    and counted;
    /// 3. a line at or below this build's version must decode; anything
    ///    else is corruption, thrown as
    ///    `LichessBotJSONLinesError.undecodableLine` (file and line), so the
    ///    caller leaves the whole file out;
    /// 4. an unterminated final line is dropped and its size returned.
    static func decodeDayFile(_ data: Data, fileName: String) throws -> LichessBotChallengeLogDayFileDecoding {
        let decoder = LichessBotJSONLines.makeDecoder()
        var entries: [LichessBotChallengeLogEntry] = []
        var lineCount = 0
        var skippedNewerLines = 0
        let dropped = try LichessBotJSONLines.forEachCompleteLine(in: data) { lineNumber, line in
            lineCount += 1
            do {
                let version = try decoder.decode(SchemaVersionProbe.self, from: line).schemaVersion
                if version > LichessBotChallengeLogEntry.currentSchemaVersion {
                    skippedNewerLines += 1
                    return
                }
                entries.append(try decoder.decode(LichessBotChallengeLogEntry.self, from: line))
            } catch {
                throw LichessBotJSONLinesError.undecodableLine(file: fileName, lineNumber: lineNumber, detail: String(describing: error))
            }
        }
        return LichessBotChallengeLogDayFileDecoding(
            entries: entries, lineCount: lineCount, skippedNewerLines: skippedNewerLines, droppedTrailingByteCount: dropped
        )
    }
}
