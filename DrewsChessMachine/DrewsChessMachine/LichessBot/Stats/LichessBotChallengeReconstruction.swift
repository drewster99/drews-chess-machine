import CryptoKit
import Foundation

// Back-fill (challenge-log plan §3.7): the challenges DCM sent and received
// before the challenge log existed, rebuilt from the protocol log
// (`Protocol/events-YYYYMMDD.jsonl`).
//
// The protocol log was written for people, not as a record: free-text
// messages, companion lines with no challenge id, never synchronized to
// disk. So every reconstructed row carries the lines it was built from
// (`evidence`) and how its origin is known (`LichessBotReconstructionConfidence`),
// and anything the lines don't settle is left unpaired and counted
// (`unexplainedLines`), never guessed.
//
// The message forms parsed here are historical formats, frozen with the plan
// (§3.7) and matched exactly. Later wording changes don't matter: once the
// live challenge log exists, the inputs stop at its first entry
// (`liveLogFirstEntryAt`), so only lines written by the builds that produced
// these forms are ever read.
//
// Everything here is a pure function of its inputs. The output has no
// generation time and is encoded with sorted keys, so the same inputs always
// give the same bytes; `LichessBotChallengeReconstructionStore` relies on
// that to write the file only when it changes.

// MARK: - Evidence and inputs

/// One protocol-log line: a day file's name and the line's number in it
/// (from 1, blank lines counted, as `LichessBotJSONLines.forEachCompleteLine`
/// numbers them). Day-file names sort by date, so this order is the order the
/// lines were recorded in.
struct LichessBotProtocolLineReference: Sendable, Codable, Hashable, Comparable, CustomStringConvertible {
    let file: String
    let line: Int

    static func < (lhs: Self, rhs: Self) -> Bool {
        if lhs.file != rhs.file { return lhs.file < rhs.file }
        return lhs.line < rhs.line
    }

    var description: String { "\(file):\(line)" }
}

/// A protocol day file handed to the reconstruction: its name and its bytes.
struct LichessBotProtocolDayFile: Sendable, Equatable {
    let name: String
    let data: Data
}

/// One protocol day file the reconstruction read, as recorded in its output.
/// `byteCount` is what the store compares to decide whether the output is
/// stale (protocol files only grow); `sha256` lets a reader prove which bytes
/// produced the rows.
struct LichessBotReconstructionInput: Sendable, Codable, Equatable {
    let file: String
    let byteCount: Int
    /// Lowercase hexadecimal SHA-256 of the file's bytes.
    let sha256: String
    /// An unterminated final line (an append in flight or interrupted),
    /// left out, as every JSON Lines reader does.
    let droppedTrailingByteCount: Int
    /// Complete lines that are not protocol entries, by line number. Left
    /// out and listed rather than failing the whole reconstruction: one bad
    /// line must not cost every past challenge.
    let undecodableLines: [Int]
}

// MARK: - Rows

/// Who sent a reconstructed outgoing challenge. Coarser than
/// `LichessBotChallengeSender`: the protocol log can't tell the Challenge
/// sheet from the operator's Resend as Casual, and it never recorded a
/// matchmaking pass's trigger or fill mode (a Fill Open Slots bracket can
/// contain an automatic pass that was already running, so the bracket can't
/// attribute sends either).
enum LichessBotReconstructedSender: String, Sendable, Codable, Equatable, Hashable, CaseIterable {
    /// The operator: the Challenge sheet or Resend as Casual. Known only
    /// from the absence of every other sender's line, so always
    /// `inferredFromAbsence`.
    case byOperator
    case challengeQueue
    case matchmaking
    case matchmakingCasualResend
}

/// How a reconstructed row's origin is known (§3.7 step 7).
enum LichessBotReconstructionConfidence: String, Sendable, Codable, Equatable, Hashable, CaseIterable {
    /// A line carrying the challenge id says it. Every direction comes from
    /// one (the challenge on the event stream, or DCM's own send line), so an
    /// incoming challenge's origin is certain.
    case certain
    /// A companion line with no id, paired with its send by player name and
    /// adjacency (within `LichessBotChallengeReconstruction.companionWindow`).
    case paired
    /// No companion line exists, so the operator sent it. A lost protocol
    /// write (the log is never synchronized to disk) would make this wrong;
    /// `LichessBotPickLineCheck` records the second line that would also have
    /// had to be lost for a matchmaking send.
    case inferredFromAbsence
}

/// What the protocol log says about who sent an outgoing challenge or
/// attempt.
enum LichessBotReconstructedSenderFinding: Sendable, Codable, Equatable, Hashable {
    case attributed(sender: LichessBotReconstructedSender, confidence: LichessBotReconstructionConfidence)
    /// A companion line could belong to this send or to another send to the
    /// same player inside the window, so it was left unpaired and the sender
    /// is not claimed.
    case ambiguousCompanion
    /// DCM's own challenge, seen on the event stream, with no "challenge
    /// sent to" line naming it: sent by another client using the token, or
    /// its line was lost.
    case noSendLine
}

/// Whether a "matchmaking pick: <name> …" line precedes a send to that
/// player within `LichessBotChallengeReconstruction.pickWindow` (§3.7 step 7).
/// Every matchmaking pass writes one before it sends, so an operator send
/// inferred from absence that has none is wrong only if two separate lines
/// were both lost.
enum LichessBotPickLineCheck: Sendable, Codable, Equatable, Hashable {
    case found(LichessBotProtocolLineReference)
    case notFound
}

/// Why a reconstructed send created no challenge.
enum LichessBotReconstructedNotCreatedReason: Sendable, Codable, Equatable, Hashable {
    /// "challenge outcome: <id> offline".
    case opponentOffline
    /// "challenge outcome: <id> refused: …", which spells out the whole
    /// refusal.
    case refused(LichessBotChallengeRefusal)
    /// "<id> is at its bot-game limit (<n>) until <time>" with no outcome
    /// line beside it: written before outcome lines existed, only from a
    /// refused POST. Lichess's HTTP status and message were not logged, so
    /// no `LichessBotChallengeRefusal` is claimed; the time is kept as
    /// logged (local time, as the line printed it).
    case botGameLimit(gamesPlayed: Int, untilAsLogged: String)
}

/// A reconstructed row's state.
enum LichessBotReconstructedChallengeState: Sendable, Codable, Equatable {
    /// A challenge Lichess created: the live ledger's state, decided by the
    /// ledger's own fold (`LichessBotChallengeLedgerRow.state`) so the
    /// precedence has one definition.
    case challenge(LichessBotChallengeLogState)
    /// A send that created no challenge.
    case notCreated(LichessBotReconstructedNotCreatedReason)
}

/// What a reconstructed row is keyed by.
enum LichessBotReconstructedRowKey: Sendable, Codable, Hashable {
    /// A challenge, by its Lichess id (which is also its game's id).
    case challenge(id: String)
    /// A send that created no challenge has no id; it is named by the line
    /// that records its outcome.
    case notCreatedAttempt(LichessBotProtocolLineReference)
}

/// The other player in a reconstructed row.
struct LichessBotReconstructedOpponent: Sendable, Codable, Equatable, Hashable {
    /// Lowercased: Lichess ids are lowercased usernames, and the protocol
    /// messages mix ids and display names.
    let id: String
    /// As the evidence spelled it (the challenge's display name when the
    /// event stream had it).
    let name: String
}

/// One reconstructed challenge, or one send that created none.
struct LichessBotReconstructedChallengeRow: Sendable, Codable, Equatable {
    let key: LichessBotReconstructedRowKey
    let direction: LichessBotChallengeLogDirection
    /// The first evidence line's time.
    let firstAt: Date
    /// Nil only for an open challenge (one that names no opponent) DCM
    /// sent, which no line here names a player for.
    let opponent: LichessBotReconstructedOpponent?
    /// The challenge as the event stream described it; nil when no
    /// `challenge` event was logged for it, and for a send that created none.
    let challenge: LichessBotChallengeSnapshot?
    /// The terms DCM's "challenge sent to" line gave; nil when there is no
    /// such line (incoming challenges, echo-only and not-created rows).
    let sendTerms: LichessBotOutgoingChallenge?
    /// Who sent it; nil for an incoming challenge, whose challenger is
    /// `opponent`.
    let sender: LichessBotReconstructedSenderFinding?
    /// Nil when the row has no send line to check (incoming, echo-only).
    let pickLineCheck: LichessBotPickLineCheck?
    /// DCM's latest decision on an incoming challenge; nil when none was
    /// logged, and for outgoing rows.
    let decision: LichessBotIncomingDecisionRecord?
    let state: LichessBotReconstructedChallengeState
    /// What an accepted row's outranked facts said (the ledger's notes).
    let notes: [LichessBotChallengeLedgerNote]
    /// Every line the row was built from, in record order.
    let evidence: [LichessBotProtocolLineReference]

    /// How the row's origin is known: certain for incoming; the sender's
    /// confidence for outgoing; nil when the sender is not determined.
    var originConfidence: LichessBotReconstructionConfidence? {
        switch direction {
        case .incoming:
            return .certain
        case .outgoing:
            switch sender {
            case .attributed(_, let confidence)?:
                return confidence
            case .ambiguousCompanion?, .noSendLine?, nil:
                return nil
            }
        }
    }
}

/// A line of a frozen form that the reconstruction could not use, and why.
/// Listed with its reference so it can be found, never dropped silently.
struct LichessBotReconstructionUnexplainedLine: Sendable, Codable, Equatable, Hashable {
    enum Reason: String, Sendable, Codable, Equatable, Hashable, CaseIterable {
        /// A sender companion with no unpaired send to that player within
        /// the window before it.
        case companionWithoutSend
        /// A sender companion with two or more candidate sends.
        case companionAmbiguous
        /// A failure companion with no unpaired not-created attempt to that
        /// player within the window before it (the send failed before
        /// reaching Lichess, or failed without an answer).
        case failureCompanionWithoutAttempt
        /// A failure companion with two or more candidate attempts.
        case failureCompanionAmbiguous
        /// "withdrawing unanswered challenge to <name>" with no open
        /// outgoing challenge to that player at that point.
        case timeoutWithdrawalWithoutOpenChallenge
        /// The same, with two or more open challenges to that player.
        case timeoutWithdrawalAmbiguous
        /// A line naming a challenge id that no reconstructed challenge has
        /// (or one of the wrong direction).
        case namesNoMatchingChallenge
        /// A second "challenge sent to" line for an id.
        case repeatedSendLine
        /// DCM's send line for an id the event stream showed as incoming.
        case sendLineForIncomingChallenge
        /// An event-stream line that does not decode as an event.
        case undecodableStreamEvent
        /// A frozen form whose details don't parse.
        case unparsableDetails
    }

    let line: LichessBotProtocolLineReference
    let reason: Reason
}

/// The reconstruction's tallies, for the log line and the Challenge Log
/// window's footer.
struct LichessBotChallengeReconstructionCounts: Sendable, Codable, Equatable {
    /// Challenges DCM sent that Lichess created (send line or event stream).
    var outgoingCreatedRows = 0
    var notCreatedRows = 0
    var incomingRows = 0
    /// "challenge sent to" lines used.
    var sendLines = 0
    /// Outgoing challenge rows with no send line.
    var outgoingChallengesWithoutSendLine = 0
    var companionsPaired = 0
    var failureCompanionsPaired = 0
    var notCreatedFromOutcomeLines = 0
    var notCreatedFromBotLimitLines = 0
    /// Bot-limit lines with an outcome line beside them: the same refusal,
    /// counted once (as the outcome line's attempt).
    var botLimitLinesBesideOutcomeLines = 0
    /// "<us>: accept" decision lines, which older builds logged on their own
    /// echoes: not incoming decisions, skipped.
    var skippedOwnEchoAcceptLines = 0
    /// "<us>: ignore: our own outgoing challenge" and any other decision
    /// line on DCM's own echo: skipped.
    var skippedOwnEchoOtherDecisionLines = 0
    /// Entries at or after `liveLogFirstEntryAt`, left to the live log.
    var entriesAtOrAfterCutoff = 0
}

// MARK: - The reconstruction

/// `Challenges/reconstructed-from-protocol.json`: past challenges rebuilt
/// from the protocol log by Algorithm v1 (§3.7).
struct LichessBotChallengeReconstruction: Sendable, Codable, Equatable {
    /// Bumped whenever the algorithm or the output's shape changes, so a
    /// stored file from another version is regenerated.
    static let currentAlgorithmVersion = 1

    /// How long after its send (or not-created attempt) a companion line may
    /// come and still pair with it. Each companion is written synchronously
    /// right after its send on the main actor, so in practice it is the very
    /// next challenge line.
    static let companionWindow: TimeInterval = 5
    /// How long after its outcome-less bot-limit line an outcome line may
    /// come and still be the same refusal (written in the same synchronous
    /// step).
    static let botLimitOutcomeWindow: TimeInterval = 5
    /// How long before a send a matchmaking pick line may come (the pick,
    /// the player-status check and the POST, which can take up to its
    /// request timeout).
    static let pickWindow: TimeInterval = 60

    let algorithmVersion: Int
    /// The bot's account id the directions were decided against. A
    /// different id changes every direction, so it regenerates.
    let ourAccountID: String
    /// Nil when there was no live log: every entry was read.
    let liveLogFirstEntryAt: Date?
    /// The day files read, in name order.
    let inputs: [LichessBotReconstructionInput]
    /// Sorted by first evidence.
    let rows: [LichessBotReconstructedChallengeRow]
    /// Sorted by line.
    let unexplainedLines: [LichessBotReconstructionUnexplainedLine]
    let counts: LichessBotChallengeReconstructionCounts

    var inputByteCount: Int {
        inputs.reduce(0) { $0 + $1.byteCount }
    }

    /// The unexplained lines with `reason` (the list is the one record of
    /// them; this only counts).
    func unexplainedLineCount(_ reason: LichessBotReconstructionUnexplainedLine.Reason) -> Int {
        unexplainedLines.reduce(0) { $0 + ($1.reason == reason ? 1 : 0) }
    }

    // MARK: Inputs

    /// `events-YYYYMMDD.jsonl`, exactly: what
    /// `LichessBotDataDirectory.protocolLogURL(for:)` names.
    static func isProtocolDayFileName(_ name: String) -> Bool {
        dayStamp(ofProtocolDayFileName: name) != nil
    }

    private static func dayStamp(ofProtocolDayFileName name: String) -> Substring? {
        let prefix = "events-"
        let suffix = ".jsonl"
        guard name.hasPrefix(prefix), name.hasSuffix(suffix) else { return nil }
        let stamp = name.dropFirst(prefix.count).dropLast(suffix.count)
        guard stamp.utf8.count == 8,
              stamp.utf8.allSatisfy({ (UInt8(ascii: "0")...UInt8(ascii: "9")).contains($0) }) else { return nil }
        return stamp
    }

    /// The protocol day files that can hold entries before the cutoff: every
    /// one when there is no live log, else those dated on or before the
    /// cutoff's UTC day. Name order.
    static func inputFileNames<Names: Sequence>(from names: Names, liveLogFirstEntryAt: Date?) -> [String] where Names.Element == String {
        let lastStamp = liveLogFirstEntryAt.map(LichessBotDataDirectory.utcDayStamp(for:))
        return names.filter { name in
            guard let stamp = dayStamp(ofProtocolDayFileName: name) else { return false }
            guard let lastStamp else { return true }
            return stamp <= lastStamp
        }.sorted()
    }

    /// A timestamp as the output encodes it (milliseconds). Two cutoffs are
    /// the same cutoff when these agree, which survives a round trip
    /// through the file.
    static func timestampText(_ date: Date) -> String {
        date.formatted(Date.ISO8601FormatStyle(includingFractionalSeconds: true))
    }

    // MARK: Encoding

    /// The file's bytes: sorted keys, pretty-printed, newline-terminated, no
    /// generation time, so equal reconstructions are equal bytes.
    func encoded() throws -> Data {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601withFractionalSeconds
        encoder.outputFormatting = [.sortedKeys, .prettyPrinted, .withoutEscapingSlashes]
        var data = try encoder.encode(self)
        data.append(UInt8(ascii: "\n"))
        return data
    }

    static func decode(_ data: Data) throws -> LichessBotChallengeReconstruction {
        try LichessBotJSONLines.makeDecoder().decode(LichessBotChallengeReconstruction.self, from: data)
    }

    // MARK: Algorithm v1

    /// Rebuild past challenges from `files` (§3.7). Files are read in name
    /// order, lines in file order; only entries before `liveLogFirstEntryAt`
    /// are used. Every player-name comparison is case-insensitive.
    static func build(from files: [LichessBotProtocolDayFile],
                      ourAccountID: String,
                      liveLogFirstEntryAt: Date?) -> LichessBotChallengeReconstruction {
        var builder = Builder(ourAccountKey: ourAccountID.lowercased(), cutoff: liveLogFirstEntryAt)
        for file in files.sorted(by: { $0.name < $1.name }) {
            builder.read(file)
        }
        return builder.finish(ourAccountID: ourAccountID)
    }
}

// MARK: - Parsing the frozen message forms

/// One `.challenge` protocol message of a form §3.7 uses.
enum LichessBotReconstructionMessage: Sendable, Equatable {
    /// "challenge sent to <name>" (fields `id`, `rated`, `clock` L+I,
    /// `color`).
    case send(name: String, challengeID: String, terms: LichessBotOutgoingChallenge)
    /// "matchmaking sent a challenge to <name>", "matchmaking resent a
    /// challenge to <name> as casual", "challenge queue: sent <name>".
    case senderCompanion(name: String, sender: LichessBotReconstructedSender)
    /// "matchmaking send to <name> failed: …" / "… stopped: …"; "challenge
    /// queue: skipped <name>: …" / "dropped <name>: …" / "stopped at <name>,
    /// which waits again: …"; "matchmaking: casual resend to <name> not
    /// sent: …".
    case failureCompanion(name: String, sender: LichessBotReconstructedSender)
    /// "matchmaking pick: <name> (…".
    case matchmakingPick(name: String)
    /// "challenge outcome: <id> offline" / "challenge outcome: <id>
    /// refused: …".
    case notCreatedOutcome(opponentID: String, reason: LichessBotReconstructedNotCreatedReason)
    /// "<id> is at its bot-game limit (<n>) until <time>".
    case botGameLimit(userID: String, gamesPlayed: Int, untilAsLogged: String)
    /// "withdrew challenge <id> on going offline".
    case withdrewOnGoingOffline(challengeID: String)
    /// "withdrawing unanswered challenge to <name> after <n> s".
    case withdrawingUnanswered(name: String, seconds: Int)
    /// "outgoing challenge accepted; game <id>" (field `challenge`).
    case outgoingAccepted(challengeID: String)
    /// "<challenger>: accept | decline (<key>): <rule> | ignore: <rule>"
    /// (field `challenge`).
    case decision(challengerID: String, challengeID: String, decision: LichessBotIncomingDecisionRecord)
    /// A frozen form whose details don't parse.
    case unparsable

    /// The form `message` has, or nil when it is none of them (most
    /// `.challenge` messages are about other things).
    static func parse(_ message: String, fields: [String: String]) -> LichessBotReconstructionMessage? {
        if let rest = message.dropPrefix("challenge sent to ") {
            return parseSend(name: rest, fields: fields)
        }
        if let rest = message.dropPrefix("matchmaking sent a challenge to ") {
            return playerName(rest).map { .senderCompanion(name: $0, sender: .matchmaking) } ?? .unparsable
        }
        if let rest = message.dropPrefix("matchmaking resent a challenge to ") {
            guard let name = rest.dropSuffix(" as casual").flatMap(playerName) else { return .unparsable }
            return .senderCompanion(name: name, sender: .matchmakingCasualResend)
        }
        if let rest = message.dropPrefix("challenge queue: sent ") {
            return playerName(rest).map { .senderCompanion(name: $0, sender: .challengeQueue) } ?? .unparsable
        }
        if let rest = message.dropPrefix("matchmaking send to ") {
            guard let space = rest.firstIndex(of: " "), let name = playerName(rest[..<space]) else { return .unparsable }
            let outcome = rest[rest.index(after: space)...]
            guard outcome.hasPrefix("failed: ") || outcome.hasPrefix("stopped: ") else { return .unparsable }
            return .failureCompanion(name: name, sender: .matchmaking)
        }
        for prefix in ["challenge queue: skipped ", "challenge queue: dropped "] {
            if let rest = message.dropPrefix(prefix) {
                guard let colon = rest.range(of: ": "), let name = playerName(rest[..<colon.lowerBound]) else { return .unparsable }
                return .failureCompanion(name: name, sender: .challengeQueue)
            }
        }
        if let rest = message.dropPrefix("challenge queue: stopped at ") {
            guard let marker = rest.range(of: ", which waits again: "), let name = playerName(rest[..<marker.lowerBound]) else { return .unparsable }
            return .failureCompanion(name: name, sender: .challengeQueue)
        }
        if let rest = message.dropPrefix("matchmaking: casual resend to ") {
            guard let marker = rest.range(of: " not sent: "), let name = playerName(rest[..<marker.lowerBound]) else { return .unparsable }
            return .failureCompanion(name: name, sender: .matchmakingCasualResend)
        }
        if let rest = message.dropPrefix("matchmaking pick: ") {
            guard let marker = rest.range(of: " ("), let name = playerName(rest[..<marker.lowerBound]) else { return .unparsable }
            return .matchmakingPick(name: name)
        }
        if let rest = message.dropPrefix("challenge outcome: ") {
            return parseOutcome(rest)
        }
        if let marker = message.range(of: " is at its bot-game limit (") {
            return parseBotLimit(name: message[..<marker.lowerBound], rest: message[marker.upperBound...])
        }
        if let rest = message.dropPrefix("withdrew challenge ") {
            guard let id = rest.dropSuffix(" on going offline").flatMap(playerName) else { return .unparsable }
            return .withdrewOnGoingOffline(challengeID: id)
        }
        if let rest = message.dropPrefix("withdrawing unanswered challenge to ") {
            guard let marker = rest.range(of: " after "),
                  let name = playerName(rest[..<marker.lowerBound]),
                  let seconds = rest[marker.upperBound...].dropSuffix(" s").flatMap(strictInt) else { return .unparsable }
            return .withdrawingUnanswered(name: name, seconds: seconds)
        }
        if let rest = message.dropPrefix("outgoing challenge accepted; game ") {
            guard let id = playerName(rest), fields["challenge"] == id else { return .unparsable }
            return .outgoingAccepted(challengeID: id)
        }
        if let challengeID = fields["challenge"], let colon = message.range(of: ": ") {
            return parseDecision(challenger: message[..<colon.lowerBound], rest: message[colon.upperBound...], challengeID: challengeID)
        }
        return nil
    }

    private static func parseSend(name: Substring, fields: [String: String]) -> LichessBotReconstructionMessage {
        guard let name = playerName(name),
              let id = fields["id"].flatMap({ playerName(Substring($0)) }),
              let rated = fields["rated"].flatMap(strictBool),
              let clock = fields["clock"], let plus = clock.firstIndex(of: "+"),
              let limit = strictInt(clock[..<plus]),
              let increment = strictInt(clock[clock.index(after: plus)...]),
              let color = fields["color"].flatMap(LichessBotChallengeColorName.init(rawValue:)) else {
            return .unparsable
        }
        return .send(name: name, challengeID: id,
                     terms: LichessBotOutgoingChallenge(rated: rated, clockLimitSeconds: limit, clockIncrementSeconds: increment, color: color))
    }

    /// "<id> offline", "<id> refused: <label>, HTTP <status>[: <text>];
    /// counted <n> credits (worst case)"; nil for the outcome lines that
    /// aren't a not-created send ("<id> challenge <x> created; …", "<x>
    /// accepted", …).
    private static func parseOutcome(_ rest: Substring) -> LichessBotReconstructionMessage? {
        guard let space = rest.firstIndex(of: " "), let opponentID = playerName(rest[..<space]) else { return nil }
        let tail = rest[rest.index(after: space)...]
        if tail == "offline" {
            return .notCreatedOutcome(opponentID: opponentID.lowercased(), reason: .opponentOffline)
        }
        guard let refusalText = tail.dropPrefix("refused: ") else { return nil }
        guard let refusal = parseRefusal(refusalText) else { return .unparsable }
        return .notCreatedOutcome(opponentID: opponentID.lowercased(), reason: .refused(refusal))
    }

    /// `LichessBotController.describe(.refused(_))` followed by the credit
    /// note, read back exactly: the kind by its label, the status, and the
    /// text when there was one.
    private static func parseRefusal(_ text: Substring) -> LichessBotChallengeRefusal? {
        guard let counted = text.range(of: "; counted ", options: .backwards) else { return nil }
        let note = text[counted.upperBound...]
        guard let credits = note.dropSuffix(" credits (worst case)"), strictInt(credits) != nil else { return nil }
        let body = text[..<counted.lowerBound]
        for kind in LichessBotChallengeRefusal.Kind.allCases {
            guard let afterLabel = body.dropPrefix("\(kind.label), HTTP ") else { continue }
            let digits = afterLabel.prefix { $0.isASCII && $0.isNumber }
            guard let status = strictInt(digits) else { return nil }
            let after = afterLabel[digits.endIndex...]
            if after.isEmpty {
                return LichessBotChallengeRefusal(kind: kind, httpStatus: status, text: nil)
            }
            guard let message = after.dropPrefix(": ") else { return nil }
            return LichessBotChallengeRefusal(kind: kind, httpStatus: status, text: String(message))
        }
        return nil
    }

    private static func parseBotLimit(name: Substring, rest: Substring) -> LichessBotReconstructionMessage {
        guard let userID = playerName(name),
              let close = rest.range(of: ") until "),
              let games = strictInt(rest[..<close.lowerBound]) else { return .unparsable }
        let until = rest[close.upperBound...]
        guard !until.isEmpty else { return .unparsable }
        return .botGameLimit(userID: userID.lowercased(), gamesPlayed: games, untilAsLogged: String(until))
    }

    /// `LichessBotController.describe(_ decision:)`, read back; nil when the
    /// challenger part isn't a player name (another form with a `challenge`
    /// field, such as "outgoing challenge declined: …").
    private static func parseDecision(challenger: Substring, rest: Substring, challengeID: String) -> LichessBotReconstructionMessage? {
        guard let challengerID = playerName(challenger) else { return nil }
        let decision: LichessBotIncomingDecisionRecord
        if rest == "accept" {
            decision = .accept
        } else if let declined = rest.dropPrefix("decline (") {
            guard let close = declined.range(of: "): "),
                  let reason = LichessBotDeclineReason(rawValue: String(declined[..<close.lowerBound])) else { return .unparsable }
            decision = .decline(reason: reason, rule: String(declined[close.upperBound...]))
        } else if let rule = rest.dropPrefix("ignore: ") {
            decision = .ignore(rule: String(rule))
        } else {
            return nil
        }
        return .decision(challengerID: challengerID.lowercased(), challengeID: challengeID, decision: decision)
    }

    /// A Lichess username or id as the messages print it: non-empty, no
    /// space, no colon.
    private static func playerName(_ text: Substring) -> String? {
        guard !text.isEmpty, !text.contains(" "), !text.contains(":") else { return nil }
        return String(text)
    }

    /// Decimal digits only (no sign, no space), as DCM printed counts.
    private static func strictInt(_ text: Substring) -> Int? {
        guard !text.isEmpty, text.allSatisfy({ $0.isASCII && $0.isNumber }) else { return nil }
        return Int(text)
    }

    private static func strictBool(_ text: String) -> Bool? {
        switch text {
        case "true": return true
        case "false": return false
        default: return nil
        }
    }
}

private extension StringProtocol {
    func dropPrefix(_ prefix: String) -> SubSequence? {
        hasPrefix(prefix) ? dropFirst(prefix.count) : nil
    }

    func dropSuffix(_ suffix: String) -> SubSequence? {
        hasSuffix(suffix) ? dropLast(suffix.count) : nil
    }
}

// MARK: - The builder

private extension LichessBotChallengeReconstruction {

    /// A line's reference and time. Ordered by the reference alone (record
    /// order), never by time: the wall clock can step back.
    struct Evidence {
        let line: LichessBotProtocolLineReference
        let at: Date

        static func recordOrder(_ lhs: Evidence, _ rhs: Evidence) -> Bool {
            lhs.line < rhs.line
        }
    }

    /// A "challenge sent to" line.
    struct Send {
        let evidence: Evidence
        let name: String
        let challengeID: String
        let terms: LichessBotOutgoingChallenge
        var companion: (evidence: Evidence, sender: LichessBotReconstructedSender)?
        var wasAmbiguousCandidate = false
        var nameKey: String { name.lowercased() }
    }

    /// A send that created no challenge.
    struct Attempt {
        /// The outcome line, or the bot-limit line when there is none.
        let evidence: Evidence
        let opponentName: String
        var reason: LichessBotReconstructedNotCreatedReason
        let fromOutcomeLine: Bool
        var botLimitLine: Evidence?
        var companion: (evidence: Evidence, sender: LichessBotReconstructedSender)?
        var wasAmbiguousCandidate = false
        var nameKey: String { opponentName.lowercased() }
    }

    /// A companion line (sender or failure).
    struct Companion {
        let evidence: Evidence
        let name: String
        let sender: LichessBotReconstructedSender
        var nameKey: String { name.lowercased() }
    }

    /// A withdrawal fact for a challenge.
    struct Withdrawal {
        let evidence: Evidence
        let reason: LichessBotWithdrawalReason
        let result: (at: Date, result: LichessBotWithdrawalResult)?
    }

    /// Everything the lines say about one challenge id.
    struct ChallengeEvidence {
        var snapshot: LichessBotChallengeSnapshot?
        var snapshotLines: [Evidence] = []
        var sendIndex: Int?
        var declines: [(evidence: Evidence, reason: LichessBotDeclineReasonRecord, text: String?)] = []
        var cancels: [Evidence] = []
        var gameStarts: [Evidence] = []
        var acceptances: [Evidence] = []
        var decisions: [(evidence: Evidence, decision: LichessBotIncomingDecisionRecord)] = []
        var withdrawals: [Withdrawal] = []
        var offlineWithdrawals: [Evidence] = []
    }

    struct Builder {
        let ourAccountKey: String
        let cutoff: Date?
        var inputs: [LichessBotReconstructionInput] = []
        var counts = LichessBotChallengeReconstructionCounts()
        var unexplained: [LichessBotReconstructionUnexplainedLine] = []

        var challenges: [String: ChallengeEvidence] = [:]
        var sends: [Send] = []
        var senderCompanions: [Companion] = []
        var failureCompanions: [Companion] = []
        var picksByName: [String: [Evidence]] = [:]
        var outcomeAttempts: [Attempt] = []
        var botLimitLines: [(evidence: Evidence, userID: String, gamesPlayed: Int, untilAsLogged: String)] = []
        var offlineWithdrawalLines: [(evidence: Evidence, challengeID: String)] = []
        var timeoutLines: [(evidence: Evidence, name: String, seconds: Int)] = []
        var acceptanceLines: [(evidence: Evidence, challengeID: String)] = []
        var decisionLines: [(evidence: Evidence, challengeID: String, decision: LichessBotIncomingDecisionRecord)] = []

        init(ourAccountKey: String, cutoff: Date?) {
            self.ourAccountKey = ourAccountKey
            self.cutoff = cutoff
        }

        mutating func note(_ line: LichessBotProtocolLineReference, _ reason: LichessBotReconstructionUnexplainedLine.Reason) {
            unexplained.append(LichessBotReconstructionUnexplainedLine(line: line, reason: reason))
        }

        // MARK: Reading

        mutating func read(_ file: LichessBotProtocolDayFile) {
            let decoder = LichessBotJSONLines.makeDecoder()
            var undecodable: [Int] = []
            var entries: [(LichessBotProtocolLineReference, LichessBotProtocolEntry)] = []
            let dropped = LichessBotJSONLines.forEachCompleteLine(in: file.data) { lineNumber, line in
                do {
                    entries.append((LichessBotProtocolLineReference(file: file.name, line: lineNumber),
                                    try decoder.decode(LichessBotProtocolEntry.self, from: line)))
                } catch {
                    undecodable.append(lineNumber)
                }
            }
            let digest = SHA256.hash(data: file.data).map { String(format: "%02x", $0) }.joined()
            inputs.append(LichessBotReconstructionInput(
                file: file.name, byteCount: file.data.count, sha256: digest,
                droppedTrailingByteCount: dropped, undecodableLines: undecodable
            ))
            for (reference, entry) in entries {
                if let cutoff, entry.at >= cutoff {
                    counts.entriesAtOrAfterCutoff += 1
                    continue
                }
                take(entry, Evidence(line: reference, at: entry.at))
            }
        }

        mutating func take(_ entry: LichessBotProtocolEntry, _ evidence: Evidence) {
            switch entry.kind {
            case .stream:
                guard entry.fields["stream"] == "event", entry.message.hasPrefix("{") else { return }
                takeStreamEvent(entry.message, evidence)
            case .challenge:
                guard let message = LichessBotReconstructionMessage.parse(entry.message, fields: entry.fields) else { return }
                takeChallengeMessage(message, evidence)
            case .request, .rateLimit, .breaker, .account, .model, .game, .lifecycle, .anomaly:
                return
            }
        }

        mutating func takeStreamEvent(_ message: String, _ evidence: Evidence) {
            let event: LichessBotEvent
            do {
                event = try LichessBotEvent.decode(Data(message.utf8))
            } catch {
                note(evidence.line, .undecodableStreamEvent)
                return
            }
            switch event {
            case .challenge(let challenge, _):
                challenges[challenge.id, default: ChallengeEvidence()].snapshotLines.append(evidence)
                if challenges[challenge.id]?.snapshot == nil {
                    challenges[challenge.id]?.snapshot = LichessBotChallengeSnapshot(challenge)
                }
            case .challengeDeclined(let reference):
                challenges[reference.id, default: ChallengeEvidence()].declines.append(
                    (evidence, LichessBotDeclineReasonRecord(reasonKey: reference.declineReasonKey), reference.declineReason))
            case .challengeCanceled(let reference):
                challenges[reference.id, default: ChallengeEvidence()].cancels.append(evidence)
            case .gameStart(let info):
                challenges[info.gameId, default: ChallengeEvidence()].gameStarts.append(evidence)
            case .gameFinish, .unknown:
                return
            }
        }

        mutating func takeChallengeMessage(_ message: LichessBotReconstructionMessage, _ evidence: Evidence) {
            switch message {
            case .send(let name, let challengeID, let terms):
                sends.append(Send(evidence: evidence, name: name, challengeID: challengeID, terms: terms))
            case .senderCompanion(let name, let sender):
                senderCompanions.append(Companion(evidence: evidence, name: name, sender: sender))
            case .failureCompanion(let name, let sender):
                failureCompanions.append(Companion(evidence: evidence, name: name, sender: sender))
            case .matchmakingPick(let name):
                picksByName[name.lowercased(), default: []].append(evidence)
            case .notCreatedOutcome(let opponentID, let reason):
                outcomeAttempts.append(Attempt(evidence: evidence, opponentName: opponentID, reason: reason, fromOutcomeLine: true))
            case .botGameLimit(let userID, let gamesPlayed, let untilAsLogged):
                botLimitLines.append((evidence, userID, gamesPlayed, untilAsLogged))
            case .withdrewOnGoingOffline(let challengeID):
                offlineWithdrawalLines.append((evidence, challengeID))
            case .withdrawingUnanswered(let name, let seconds):
                timeoutLines.append((evidence, name, seconds))
            case .outgoingAccepted(let challengeID):
                acceptanceLines.append((evidence, challengeID))
            case .decision(let challengerID, let challengeID, let decision):
                guard challengerID != ourAccountKey else {
                    // Older builds decided on their own echo ("<us>: accept");
                    // current ones log "<us>: ignore: our own outgoing
                    // challenge". Neither is an incoming decision.
                    if decision == .accept {
                        counts.skippedOwnEchoAcceptLines += 1
                    } else {
                        counts.skippedOwnEchoOtherDecisionLines += 1
                    }
                    return
                }
                decisionLines.append((evidence, challengeID, decision))
            case .unparsable:
                note(evidence.line, .unparsableDetails)
            }
        }

        // MARK: Folding

        mutating func finish(ourAccountID: String) -> LichessBotChallengeReconstruction {
            pairSenderCompanions()
            let attempts = pairFailureCompanions(on: attemptsWithBotLimitLines())
            attachSends()
            attachOfflineWithdrawals()
            attachTimeoutWithdrawals()
            attachAcceptances()
            attachDecisions()

            var anchoredRows: [(firstLine: LichessBotProtocolLineReference, row: LichessBotReconstructedChallengeRow)] = []
            for (id, evidence) in challenges {
                guard let anchor = firstEvidence(of: evidence), let rowDirection = direction(of: evidence) else {
                    // Answers, game starts and the like for an id with no
                    // challenge event and no send line (a tournament game,
                    // or a challenge from before the log began).
                    for line in evidence.declines.map({ $0.evidence }) + evidence.cancels + evidence.gameStarts {
                        note(line.line, .namesNoMatchingChallenge)
                    }
                    continue
                }
                let row = challengeRow(id: id, evidence, anchor: anchor, direction: rowDirection)
                anchoredRows.append((row.evidence.reduce(anchor.line) { min($0, $1) }, row))
            }
            for attempt in attempts {
                let row = attemptRow(attempt)
                anchoredRows.append((row.evidence.reduce(attempt.evidence.line) { min($0, $1) }, row))
            }
            // Distinct rows never share a first line: each line is evidence
            // for one row only (pick lines, which several sends can share,
            // are kept out of `evidence`).
            let rows = anchoredRows.sorted { $0.firstLine < $1.firstLine }.map { $0.row }
            for row in rows {
                switch (row.key, row.direction) {
                case (.notCreatedAttempt, _): counts.notCreatedRows += 1
                case (.challenge, .outgoing): counts.outgoingCreatedRows += 1
                case (.challenge, .incoming): counts.incomingRows += 1
                }
                if row.sender == .noSendLine {
                    counts.outgoingChallengesWithoutSendLine += 1
                }
            }
            unexplained.sort { lhs, rhs in
                if lhs.line != rhs.line { return lhs.line < rhs.line }
                return lhs.reason.rawValue < rhs.reason.rawValue
            }
            return LichessBotChallengeReconstruction(
                algorithmVersion: LichessBotChallengeReconstruction.currentAlgorithmVersion,
                ourAccountID: ourAccountID,
                liveLogFirstEntryAt: cutoff,
                inputs: inputs,
                rows: rows,
                unexplainedLines: unexplained,
                counts: counts
            )
        }

        /// Step 3: each sender companion pairs with the latest unpaired send
        /// to the same player at most `companionWindow` before it. Two or
        /// more candidates: left unpaired, never guessed.
        mutating func pairSenderCompanions() {
            var sendIndicesByName: [String: [Int]] = [:]
            for index in sends.indices {
                sendIndicesByName[sends[index].nameKey, default: []].append(index)
            }
            for companion in senderCompanions {
                let candidates = (sendIndicesByName[companion.nameKey] ?? []).filter { index in
                    let send = sends[index]
                    return send.companion == nil
                        && send.evidence.line < companion.evidence.line
                        && companion.evidence.at.timeIntervalSince(send.evidence.at) <= LichessBotChallengeReconstruction.companionWindow
                }
                switch candidates.count {
                case 0:
                    note(companion.evidence.line, .companionWithoutSend)
                case 1:
                    sends[candidates[0]].companion = (companion.evidence, companion.sender)
                    counts.companionsPaired += 1
                default:
                    for index in candidates {
                        sends[index].wasAmbiguousCandidate = true
                    }
                    note(companion.evidence.line, .companionAmbiguous)
                }
            }
        }

        /// Step 5: the not-created attempts. Every outcome line is one. A
        /// bot-limit line (written only from a refused POST) is the same
        /// refusal as an outcome line for that player right after it, and
        /// otherwise (before outcome lines existed) an attempt of its own.
        mutating func attemptsWithBotLimitLines() -> [Attempt] {
            var attempts = outcomeAttempts
            for limit in botLimitLines {
                let match = attempts.indices.first { index in
                    let attempt = attempts[index]
                    guard attempt.fromOutcomeLine, attempt.botLimitLine == nil,
                          attempt.nameKey == limit.userID,
                          limit.evidence.line < attempt.evidence.line,
                          case .refused = attempt.reason else { return false }
                    return attempt.evidence.at.timeIntervalSince(limit.evidence.at) <= LichessBotChallengeReconstruction.botLimitOutcomeWindow
                }
                if let match {
                    attempts[match].botLimitLine = limit.evidence
                    counts.botLimitLinesBesideOutcomeLines += 1
                } else {
                    attempts.append(Attempt(
                        evidence: limit.evidence, opponentName: limit.userID,
                        reason: .botGameLimit(gamesPlayed: limit.gamesPlayed, untilAsLogged: limit.untilAsLogged),
                        fromOutcomeLine: false
                    ))
                }
            }
            attempts.sort { Evidence.recordOrder($0.evidence, $1.evidence) }
            for attempt in attempts {
                if attempt.fromOutcomeLine {
                    counts.notCreatedFromOutcomeLines += 1
                } else {
                    counts.notCreatedFromBotLimitLines += 1
                }
            }
            return attempts
        }

        /// Step 5's sender: a failure companion pairs with the latest
        /// unpaired attempt to the same player at most `companionWindow`
        /// before it, by the same rule as step 3.
        mutating func pairFailureCompanions(on attempts: [Attempt]) -> [Attempt] {
            var attempts = attempts
            var indicesByName: [String: [Int]] = [:]
            for index in attempts.indices {
                indicesByName[attempts[index].nameKey, default: []].append(index)
            }
            for companion in failureCompanions {
                let candidates = (indicesByName[companion.nameKey] ?? []).filter { index in
                    let attempt = attempts[index]
                    return attempt.companion == nil
                        && attempt.evidence.line < companion.evidence.line
                        && companion.evidence.at.timeIntervalSince(attempt.evidence.at) <= LichessBotChallengeReconstruction.companionWindow
                }
                switch candidates.count {
                case 0:
                    note(companion.evidence.line, .failureCompanionWithoutAttempt)
                case 1:
                    attempts[candidates[0]].companion = (companion.evidence, companion.sender)
                    counts.failureCompanionsPaired += 1
                default:
                    for index in candidates {
                        attempts[index].wasAmbiguousCandidate = true
                    }
                    note(companion.evidence.line, .failureCompanionAmbiguous)
                }
            }
            return attempts
        }

        /// Step 2: each send line names its challenge.
        mutating func attachSends() {
            for index in sends.indices {
                let send = sends[index]
                if challenges[send.challengeID]?.sendIndex != nil {
                    note(send.evidence.line, .repeatedSendLine)
                    continue
                }
                if let snapshot = challenges[send.challengeID]?.snapshot, !isOurs(snapshot) {
                    note(send.evidence.line, .sendLineForIncomingChallenge)
                    continue
                }
                challenges[send.challengeID, default: ChallengeEvidence()].sendIndex = index
                counts.sendLines += 1
            }
        }

        func isOurs(_ snapshot: LichessBotChallengeSnapshot) -> Bool {
            snapshot.challenger.id.lowercased() == ourAccountKey
        }

        func direction(of evidence: ChallengeEvidence) -> LichessBotChallengeLogDirection? {
            if let snapshot = evidence.snapshot {
                return isOurs(snapshot) ? .outgoing : .incoming
            }
            return evidence.sendIndex == nil ? nil : .outgoing
        }

        /// Step 6: "withdrew challenge <id> on going offline" is logged only
        /// after Lichess confirmed the cancel.
        mutating func attachOfflineWithdrawals() {
            for line in offlineWithdrawalLines {
                guard let evidence = challenges[line.challengeID], direction(of: evidence) == .outgoing else {
                    note(line.evidence.line, .namesNoMatchingChallenge)
                    continue
                }
                challenges[line.challengeID]?.offlineWithdrawals.append(line.evidence)
                challenges[line.challengeID]?.withdrawals.append(
                    Withdrawal(evidence: line.evidence, reason: .goingOffline, result: (at: line.evidence.at, result: .confirmed)))
            }
        }

        /// Step 6: "withdrawing unanswered challenge to <name> after N s"
        /// names a player, not a challenge: it belongs to the outgoing
        /// challenge to that player still open at that line. Its result is
        /// the `challengeCanceled` that follows, if one does.
        mutating func attachTimeoutWithdrawals() {
            var idsByOpponent: [String: [String]] = [:]
            for (id, evidence) in challenges where direction(of: evidence) == .outgoing {
                guard let key = opponent(of: evidence)?.id else { continue }
                idsByOpponent[key, default: []].append(id)
            }
            for line in timeoutLines {
                let open = (idsByOpponent[line.name.lowercased()] ?? []).filter { id in
                    guard let evidence = challenges[id], let start = firstEvidence(of: evidence), start.line < line.evidence.line else { return false }
                    let ends = evidence.gameStarts + evidence.declines.map { $0.evidence } + evidence.cancels + evidence.offlineWithdrawals
                    return !ends.contains { $0.line < line.evidence.line }
                }.sorted()
                switch open.count {
                case 0:
                    note(line.evidence.line, .timeoutWithdrawalWithoutOpenChallenge)
                case 1:
                    let id = open[0]
                    let cancel = challenges[id]?.cancels.filter { line.evidence.line < $0.line }.min(by: Evidence.recordOrder)
                    challenges[id]?.withdrawals.append(Withdrawal(
                        evidence: line.evidence,
                        reason: .unansweredTimeout(seconds: line.seconds),
                        result: cancel.map { (at: $0.at, result: LichessBotWithdrawalResult.confirmed) }
                    ))
                default:
                    note(line.evidence.line, .timeoutWithdrawalAmbiguous)
                }
            }
        }

        mutating func attachAcceptances() {
            for line in acceptanceLines {
                guard let evidence = challenges[line.challengeID], direction(of: evidence) == .outgoing else {
                    note(line.evidence.line, .namesNoMatchingChallenge)
                    continue
                }
                challenges[line.challengeID]?.acceptances.append(line.evidence)
            }
        }

        /// Step 4: decisions on challenges someone else sent.
        mutating func attachDecisions() {
            for line in decisionLines {
                guard let evidence = challenges[line.challengeID], direction(of: evidence) == .incoming else {
                    note(line.evidence.line, .namesNoMatchingChallenge)
                    continue
                }
                challenges[line.challengeID]?.decisions.append((line.evidence, line.decision))
            }
        }

        /// The earliest of the challenge event and the send line: what makes
        /// an id a challenge row. Nil when it has neither.
        func firstEvidence(of evidence: ChallengeEvidence) -> Evidence? {
            var candidates = evidence.snapshotLines
            if let index = evidence.sendIndex {
                candidates.append(sends[index].evidence)
            }
            return candidates.min(by: Evidence.recordOrder)
        }

        func opponent(of evidence: ChallengeEvidence) -> LichessBotReconstructedOpponent? {
            if let snapshot = evidence.snapshot {
                let party = isOurs(snapshot) ? snapshot.destUser : snapshot.challenger
                if let party {
                    return LichessBotReconstructedOpponent(id: party.id.lowercased(), name: party.name)
                }
            }
            if let index = evidence.sendIndex {
                return LichessBotReconstructedOpponent(id: sends[index].nameKey, name: sends[index].name)
            }
            return nil
        }

        /// The latest pick line for `name` at or before `evidence`, within
        /// `pickWindow`.
        func pickLineCheck(name: String, before evidence: Evidence) -> LichessBotPickLineCheck {
            let pick = (picksByName[name.lowercased()] ?? []).last { pick in
                pick.line < evidence.line && evidence.at.timeIntervalSince(pick.at) <= LichessBotChallengeReconstruction.pickWindow
            }
            return pick.map { .found($0.line) } ?? .notFound
        }

        func senderFinding(companion: (evidence: Evidence, sender: LichessBotReconstructedSender)?,
                           wasAmbiguousCandidate: Bool) -> LichessBotReconstructedSenderFinding {
            if let companion {
                return .attributed(sender: companion.sender, confidence: .paired)
            }
            if wasAmbiguousCandidate {
                return .ambiguousCompanion
            }
            return .attributed(sender: .byOperator, confidence: .inferredFromAbsence)
        }

        /// `anchor` is `firstEvidence(of:)` and `rowDirection` is
        /// `direction(of:)`; both exist for every challenge row.
        func challengeRow(id: String, _ evidence: ChallengeEvidence, anchor: Evidence,
                          direction rowDirection: LichessBotChallengeLogDirection) -> LichessBotReconstructedChallengeRow {
            let send = evidence.sendIndex.map { sends[$0] }

            // The state comes from the live ledger's own fold, fed the facts
            // the protocol lines prove, so the precedence (§3.4) has one
            // definition.
            var facts: [LichessBotChallengeLedgerFact] = []
            if let snapshot = evidence.snapshot, let first = evidence.snapshotLines.min(by: Evidence.recordOrder) {
                let event: LichessBotChallengeLogEvent = rowDirection == .outgoing
                    ? .outgoingSeenWithoutCreatedLine(challenge: snapshot, attribution: .notRecorded)
                    : .incomingReceived(challenge: snapshot)
                facts.append(LichessBotChallengeLedgerFact(at: first.at, event: event))
            }
            for decline in evidence.declines {
                facts.append(LichessBotChallengeLedgerFact(at: decline.evidence.at, event: .declinedOnLichess(challengeID: id, reason: decline.reason, text: decline.text)))
            }
            for cancel in evidence.cancels {
                facts.append(LichessBotChallengeLedgerFact(at: cancel.at, event: .canceledOnLichess(challengeID: id)))
            }
            for start in evidence.gameStarts {
                facts.append(LichessBotChallengeLedgerFact(at: start.at, event: .gameStarted(challengeID: id)))
            }
            for withdrawal in evidence.withdrawals {
                facts.append(LichessBotChallengeLedgerFact(at: withdrawal.evidence.at, event: .withdrawalRequested(challengeID: id, reason: withdrawal.reason)))
                if let result = withdrawal.result {
                    facts.append(LichessBotChallengeLedgerFact(at: result.at, event: .withdrawalResult(challengeID: id, result: result.result)))
                }
            }
            for decision in evidence.decisions {
                facts.append(LichessBotChallengeLedgerFact(at: decision.evidence.at, event: .incomingDecided(challengeID: id, decision: decision.decision)))
            }

            var state: LichessBotChallengeLogState = .open
            var notes: [LichessBotChallengeLedgerNote] = []
            var latestDecision: LichessBotIncomingDecisionRecord?
            if let firstFact = facts.first {
                var row = LichessBotChallengeLedgerRow(key: .challenge(id: id), firstFact: firstFact)
                for fact in facts.dropFirst() {
                    row.add(fact)
                }
                state = row.state
                notes = row.notes
                latestDecision = row.decisions.last?.decision
            }
            // A row with no challenge event has no fact that tells the
            // ledger its direction, but its send line does.
            if state == .canceledOnLichessDirectionNotRecorded, rowDirection == .outgoing {
                state = .canceledOnLichessWithoutRecordedWithdrawal
            }
            // "outgoing challenge accepted" proves an acceptance whose game
            // start may not have been logged.
            if !evidence.acceptances.isEmpty, evidence.gameStarts.isEmpty {
                state = .accepted(gameStarted: false)
            }

            var lines = evidence.snapshotLines + evidence.declines.map { $0.evidence } + evidence.cancels + evidence.gameStarts
                + evidence.acceptances + evidence.decisions.map { $0.evidence } + evidence.withdrawals.map(\.evidence)
            if let send {
                lines.append(send.evidence)
                if let companion = send.companion {
                    lines.append(companion.evidence)
                }
            }
            let sorted = Array(Set(lines.map(\.line))).sorted()
            let firstAt = lines.reduce(anchor) { Evidence.recordOrder($1, $0) ? $1 : $0 }.at

            let sender: LichessBotReconstructedSenderFinding?
            let pickCheck: LichessBotPickLineCheck?
            switch rowDirection {
            case .incoming:
                sender = nil
                pickCheck = nil
            case .outgoing:
                if let send {
                    sender = senderFinding(companion: send.companion, wasAmbiguousCandidate: send.wasAmbiguousCandidate)
                    pickCheck = pickLineCheck(name: send.name, before: send.evidence)
                } else {
                    sender = .noSendLine
                    pickCheck = nil
                }
            }

            return LichessBotReconstructedChallengeRow(
                key: .challenge(id: id),
                direction: rowDirection,
                firstAt: firstAt,
                opponent: opponent(of: evidence),
                challenge: evidence.snapshot,
                sendTerms: send?.terms,
                sender: sender,
                pickLineCheck: pickCheck,
                decision: rowDirection == .incoming ? latestDecision : nil,
                state: .challenge(state),
                notes: notes,
                evidence: sorted
            )
        }

        func attemptRow(_ attempt: Attempt) -> LichessBotReconstructedChallengeRow {
            var lines = [attempt.evidence]
            if let limit = attempt.botLimitLine {
                lines.append(limit)
            }
            if let companion = attempt.companion {
                lines.append(companion.evidence)
            }
            lines.sort(by: Evidence.recordOrder)
            return LichessBotReconstructedChallengeRow(
                key: .notCreatedAttempt(attempt.evidence.line),
                direction: .outgoing,
                firstAt: lines[0].at,
                opponent: LichessBotReconstructedOpponent(id: attempt.nameKey, name: attempt.opponentName),
                challenge: nil,
                sendTerms: nil,
                sender: senderFinding(companion: attempt.companion, wasAmbiguousCandidate: attempt.wasAmbiguousCandidate),
                pickLineCheck: pickLineCheck(name: attempt.opponentName, before: lines[0]),
                decision: nil,
                state: .notCreated(attempt.reason),
                notes: [],
                evidence: lines.map(\.line)
            )
        }
    }
}

// MARK: - Joining games

/// What the reconstruction says about how a game started (game id ==
/// challenge id).
enum LichessBotReconstructedGameOrigin: Sendable, Equatable {
    /// DCM accepted a challenge someone sent it (certain).
    case incoming
    /// DCM's own challenge, and who sent it.
    case outgoing(LichessBotReconstructedSenderFinding)
    /// No reconstructed challenge has the game's id.
    case noReconstructedChallenge
}

/// The categories the reconstruction log line counts games in.
enum LichessBotReconstructedGameCategory: String, Sendable, CaseIterable {
    case incoming
    case matchmaking
    case challengeQueue
    case matchmakingCasualResend
    case operatorInferred
    /// DCM's challenge, but the sender is not determined (no send line, or
    /// an ambiguous companion).
    case outgoingSenderNotDetermined
    /// No reconstructed challenge has the game's id.
    case unknown

    init(_ origin: LichessBotReconstructedGameOrigin) {
        switch origin {
        case .incoming:
            self = .incoming
        case .outgoing(.attributed(let sender, _)):
            switch sender {
            case .byOperator: self = .operatorInferred
            case .challengeQueue: self = .challengeQueue
            case .matchmaking: self = .matchmaking
            case .matchmakingCasualResend: self = .matchmakingCasualResend
            }
        case .outgoing(.ambiguousCompanion), .outgoing(.noSendLine):
            self = .outgoingSenderNotDetermined
        case .noReconstructedChallenge:
            self = .unknown
        }
    }
}

/// Games counted by category.
struct LichessBotReconstructedGameCounts: Sendable, Equatable {
    private(set) var byCategory: [LichessBotReconstructedGameCategory: Int] = Dictionary(
        uniqueKeysWithValues: LichessBotReconstructedGameCategory.allCases.map { ($0, 0) }
    )

    subscript(category: LichessBotReconstructedGameCategory) -> Int {
        byCategory[category] ?? 0
    }

    var total: Int {
        byCategory.values.reduce(0, +)
    }

    mutating func add(_ category: LichessBotReconstructedGameCategory) {
        byCategory[category, default: 0] += 1
    }
}

/// The reconstruction's challenge rows by id: the join of past games to
/// their challenges (§3.6 step 3, §3.7 "Game origins for past games"),
/// computed in memory.
struct LichessBotReconstructedChallengeLookup: Sendable {
    private let rowsByChallengeID: [String: LichessBotReconstructedChallengeRow]

    init(_ reconstruction: LichessBotChallengeReconstruction) {
        var rows: [String: LichessBotReconstructedChallengeRow] = [:]
        for row in reconstruction.rows {
            if case .challenge(let id) = row.key {
                rows[id] = row
            }
        }
        rowsByChallengeID = rows
    }

    func row(challengeID: String) -> LichessBotReconstructedChallengeRow? {
        rowsByChallengeID[challengeID]
    }

    /// How the game `gameID` started, by its challenge (ids compared
    /// exactly: Lichess ids are case-sensitive).
    func gameOrigin(gameID: String) -> LichessBotReconstructedGameOrigin {
        guard let row = rowsByChallengeID[gameID] else { return .noReconstructedChallenge }
        switch row.direction {
        case .incoming:
            return .incoming
        case .outgoing:
            guard let sender = row.sender else { return .outgoing(.noSendLine) }
            return .outgoing(sender)
        }
    }

    /// Every id in `gameIDs` counted once per occurrence (pass each game
    /// once, e.g. the index rows' ids).
    func gameCounts<GameIDs: Sequence>(gameIDs: GameIDs) -> LichessBotReconstructedGameCounts where GameIDs.Element == String {
        var counts = LichessBotReconstructedGameCounts()
        for gameID in gameIDs {
            counts.add(LichessBotReconstructedGameCategory(gameOrigin(gameID: gameID)))
        }
        return counts
    }
}
