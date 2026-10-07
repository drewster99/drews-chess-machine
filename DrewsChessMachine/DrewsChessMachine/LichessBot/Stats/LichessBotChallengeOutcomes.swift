import Foundation

/// How one of DCM's outgoing challenges ended. The single classification the
/// outcome log, its rolling counts and the Overview all read.
enum LichessBotChallengeOutcome: Sendable, Hashable, Codable {
    /// The challenged player accepted; a game followed.
    case accepted
    /// The challenged player declined, with Lichess's reason key.
    case declined(LichessBotDeclineReasonRecord)
    /// Withdrawn — by the operator, by the automatic timeout, by DCM going
    /// offline — or expired on Lichess.
    case canceled
    /// The player was offline when DCM checked, so nothing was posted.
    case offline
    /// Lichess refused the challenge POST: no challenge was created. Some
    /// refusals still spend credits; see
    /// `LichessBotChallengeRefusal.spendsCredits`.
    case refused(LichessBotChallengeRefusal)

    /// A challenge was created on Lichess.
    var challengeWasCreated: Bool {
        switch self {
        case .accepted, .declined, .canceled:
            return true
        case .offline, .refused:
            return false
        }
    }
}

/// A decline's reason as Lichess reported it. Lichess documents the keys in
/// `LichessBotDeclineReason`; a key outside them, or a decline event with
/// none, is kept as reported rather than folded into `generic`.
///
/// Saved records (`challenge-outcomes.json`, the challenge log) store the
/// form Swift synthesizes for these cases — `{"known":{"_0":"noBot"}}`,
/// `{"unrecognized":{"_0":"…"}}`, `{"unstated":{}}` — and still do;
/// `StoredForm` keeps that shape. Reading normalizes: builds before the
/// event-key fix stored the event stream's lowercased keys (`nobot`,
/// `timecontrol`, `toofast`, `tooslow`, `onlybot`) as unrecognized, and
/// such a record reads as its known reason through `init(reasonKey:)`, so a
/// reason is counted under one case whichever build wrote it. Nothing is
/// rewritten for this: `challenge-outcomes.json` carries the known form
/// after its next ordinary save, and append-only lines stay as written.
enum LichessBotDeclineReasonRecord: Sendable, Hashable, Codable {
    case known(LichessBotDeclineReason)
    /// A key outside `LichessBotDeclineReason`, as sent. Built only through
    /// `init(reasonKey:)`, so it never holds a key that names a reason.
    case unrecognized(String)
    /// The decline event carried no reason key.
    case unstated

    init(reasonKey: String?) {
        guard let reasonKey else {
            self = .unstated
            return
        }
        if let known = LichessBotDeclineReason(lichessKey: reasonKey) {
            self = .known(known)
        } else {
            self = .unrecognized(reasonKey)
        }
    }

    /// The stored shape: the one Swift synthesizes for this enum's cases,
    /// kept so files written before and after the normalizing decode read
    /// alike.
    private enum StoredForm: Codable {
        case known(LichessBotDeclineReason)
        case unrecognized(String)
        case unstated
    }

    init(from decoder: Decoder) throws {
        switch try StoredForm(from: decoder) {
        case .known(let reason):
            self = .known(reason)
        case .unrecognized(let key):
            self.init(reasonKey: key)
        case .unstated:
            self = .unstated
        }
    }

    func encode(to encoder: Encoder) throws {
        let stored: StoredForm
        switch self {
        case .known(let reason):
            stored = .known(reason)
        case .unrecognized(let key):
            stored = .unrecognized(key)
        case .unstated:
            stored = .unstated
        }
        try stored.encode(to: encoder)
    }

    /// The reason for tables and logs: a known reason in the API
    /// documentation's spelling (`noBot`), any other key as Lichess sent it.
    var keyText: String {
        switch self {
        case .known(let reason): return reason.rawValue
        case .unrecognized(let key): return key
        case .unstated: return "(none)"
        }
    }
}

/// Lichess's refusal of a challenge POST, with its text kept verbatim.
struct LichessBotChallengeRefusal: Sendable, Hashable, Codable {
    enum Kind: String, Sendable, Hashable, Codable, CaseIterable {
        /// HTTP 429.
        case rateLimited
        /// HTTP 400 because a bot-vs-bot daily game limit is reached (the
        /// challenged bot's, or DCM's own).
        case botDailyGameLimit
        /// HTTP 400 for any other reason.
        case badRequest
        /// Any other non-2xx status.
        case otherHTTPStatus

        var label: String {
            switch self {
            case .rateLimited: return "rate limit (429)"
            case .botDailyGameLimit: return "bot daily game limit (400)"
            case .badRequest: return "other 400"
            case .otherHTTPStatus: return "other HTTP status"
            }
        }
    }

    let kind: Kind
    let httpStatus: Int
    /// Lichess's message (or the gate's description of a 429); nil when the
    /// response carried none.
    let text: String?

    /// Whether Lichess charged the challenge's credits before refusing it.
    /// Lichess charges when the request passes its per-user challenge rate
    /// limit, before it checks the bot-vs-bot daily game limit and the
    /// challenged player's preferences, so those refusals cost the same as
    /// a created challenge. A 429 is that rate limit refusing, and an
    /// authorization failure never reaches it, so neither costs anything.
    /// A few other 400s (no such user, challenging oneself) are raised
    /// before the charge; they are indistinguishable here from the charged
    /// ones except by wording, so every other 400 is charged: the worst
    /// case, so the budget shown is never understated.
    var spendsCredits: Bool {
        switch kind {
        case .botDailyGameLimit, .badRequest:
            return true
        case .rateLimited, .otherHTTPStatus:
            return false
        }
    }

    /// Lichess's wording for a bot-vs-bot daily limit refusal, which names
    /// whichever side reached it.
    static let botDailyGameLimitMarker = "games against other bots today"

    /// The refusal a failed challenge POST represents, or nil when the error
    /// is not an answer from Lichess refusing it: a transport failure, the
    /// gate being closed before the request went out, a cancellation, or an
    /// undecodable success. In those cases whether a challenge exists is not
    /// known, so the caller logs rather than records it.
    static func classify(postError error: Error) -> LichessBotChallengeRefusal? {
        if let gateError = error as? LichessBotGateError {
            switch gateError {
            case .rateLimited:
                return LichessBotChallengeRefusal(kind: .rateLimited, httpStatus: 429, text: gateError.localizedDescription)
            case .closed:
                return nil
            }
        }
        guard let apiError = error as? LichessBotAPIError else { return nil }
        switch apiError {
        case .http(let status, let message):
            if status == 429 {
                return LichessBotChallengeRefusal(kind: .rateLimited, httpStatus: status, text: message)
            }
            if status == 400 {
                if let message, message.contains(botDailyGameLimitMarker) {
                    return LichessBotChallengeRefusal(kind: .botDailyGameLimit, httpStatus: status, text: message)
                }
                return LichessBotChallengeRefusal(kind: .badRequest, httpStatus: status, text: message)
            }
            return LichessBotChallengeRefusal(kind: .otherHTTPStatus, httpStatus: status, text: message)
        case .unauthorized(let status, let message):
            return LichessBotChallengeRefusal(kind: .otherHTTPStatus, httpStatus: status, text: message)
        case .undecodableResponse, .invalidURL, .tooManyIDs:
            return nil
        }
    }
}

/// Who a challenge went to, as far as Lichess's credit costs care.
enum LichessBotChallengeOpponentKind: Sendable, Hashable, Codable {
    case bot
    case human
}

/// Lichess's challenge credits: every charged challenge spends some, within
/// a per-minute and a per-day budget. DCM counts them over rolling windows
/// of those lengths, which never undercount a window Lichess resets.
enum LichessBotChallengeCredits {
    static let perDay = 200
    static let perMinute = 25
    static let dayWindow: TimeInterval = 24 * 3600
    static let minuteWindow: TimeInterval = 60

    /// What a charged challenge to `kind` costs, at most. Lichess charges
    /// nothing when the challenged player (bot or human) follows DCM's
    /// account, but no API reports whether another player follows the
    /// token's account (`following` in a profile is the other direction),
    /// so this is always the non-follower cost: the worst case, so the
    /// budget shown is never understated.
    static func cost(for kind: LichessBotChallengeOpponentKind) -> Int {
        switch kind {
        case .bot:
            return 1
        case .human:
            return 5
        }
    }
}

/// One outgoing challenge attempt that reached a result, or is waiting for
/// one.
struct LichessBotChallengeOutcomeRecord: Sendable, Hashable, Codable, Identifiable {
    let id: UUID
    let sentAt: Date
    /// Lowercased Lichess user id.
    let opponentID: String
    let opponentKind: LichessBotChallengeOpponentKind
    /// The created challenge's id; nil when none was created.
    let challengeID: String?
    /// Credits spent, at most: the opponent's cost for a created challenge
    /// or a refusal Lichess charged for, zero otherwise.
    let creditCost: Int
    /// Nil while a created challenge waits for an answer.
    var outcome: LichessBotChallengeOutcome?
    var resolvedAt: Date?
}

/// The outgoing-challenge outcome log (persisted to
/// `challenge-outcomes.json`, like the player notes): one record per
/// attempt, kept for the rolling day and pruned after it. Pure; callers
/// pass the time, so the rolling windows are tested with a fixed clock.
struct LichessBotChallengeOutcomeLog: Sendable, Codable, Equatable {
    /// Oldest first.
    private(set) var records: [LichessBotChallengeOutcomeRecord] = []

    /// Rolling counts over the last day (the credits also over the last
    /// minute), for the Overview.
    struct Summary: Sendable, Equatable {
        var creditsLastDay = 0
        var creditsLastMinute = 0
        var accepted = 0
        var declined = 0
        var canceled = 0
        var offline = 0
        var refused = 0
        /// Created challenges still waiting for an answer.
        var pending = 0
        var declinedByReason: [LichessBotDeclineReasonRecord: Int] = [:]
        var refusedByKind: [LichessBotChallengeRefusal.Kind: Int] = [:]

        /// Accepted over created challenges that were answered (accepted,
        /// declined or canceled); nil when none was.
        var acceptanceRate: Double? {
            let answered = accepted + declined + canceled
            guard answered > 0 else { return nil }
            return Double(accepted) / Double(answered)
        }
    }

    /// A challenge Lichess created: pending until `resolve`.
    mutating func recordCreated(challengeID: String, opponentID: String, kind: LichessBotChallengeOpponentKind, at now: Date, makeID: () -> UUID = { UUID() }) {
        records.append(LichessBotChallengeOutcomeRecord(
            id: makeID(), sentAt: now, opponentID: opponentID.lowercased(), opponentKind: kind,
            challengeID: challengeID, creditCost: LichessBotChallengeCredits.cost(for: kind),
            outcome: nil, resolvedAt: nil
        ))
    }

    /// An attempt that created no challenge (`offline` or `refused`); a
    /// refusal Lichess charged for costs the opponent's credits. Returns
    /// false, recording nothing, for an outcome that implies a created
    /// challenge.
    @discardableResult
    mutating func recordNotCreated(opponentID: String, kind: LichessBotChallengeOpponentKind, outcome: LichessBotChallengeOutcome, at now: Date, makeID: () -> UUID = { UUID() }) -> Bool {
        guard !outcome.challengeWasCreated else { return false }
        records.append(LichessBotChallengeOutcomeRecord(
            id: makeID(), sentAt: now, opponentID: opponentID.lowercased(), opponentKind: kind,
            challengeID: nil, creditCost: Self.creditCost(notCreated: outcome, kind: kind), outcome: outcome, resolvedAt: now
        ))
        return true
    }

    /// Whether `resolve` would change a record.
    func canResolve(challengeID: String, outcome: LichessBotChallengeOutcome) -> Bool {
        resolvableIndex(challengeID: challengeID, outcome: outcome) != nil
    }

    private func resolvableIndex(challengeID: String, outcome: LichessBotChallengeOutcome) -> Int? {
        guard outcome.challengeWasCreated,
              let index = records.lastIndex(where: { $0.challengeID == challengeID }) else { return nil }
        switch (records[index].outcome, outcome) {
        case (.none, _), (.canceled?, .accepted):
            return index
        default:
            return nil
        }
    }

    /// Credits counted for an attempt that created no challenge.
    static func creditCost(notCreated outcome: LichessBotChallengeOutcome, kind: LichessBotChallengeOpponentKind) -> Int {
        guard case .refused(let refusal) = outcome, refusal.spendsCredits else { return 0 }
        return LichessBotChallengeCredits.cost(for: kind)
    }

    /// Set a pending challenge's outcome. Only the first answer counts: a
    /// challenge already resolved (say, accepted, then reported canceled by
    /// a late cleanup) keeps its outcome. The one exception is a game
    /// starting for a challenge recorded as canceled: DCM infers some
    /// cancellations (Lichess no longer knowing a challenge DCM tried to
    /// withdraw, or a withdrawal on going offline that lost the race with
    /// an acceptance), and a started game is proof it was accepted.
    /// Returns whether a record changed.
    @discardableResult
    mutating func resolve(challengeID: String, outcome: LichessBotChallengeOutcome, at now: Date) -> Bool {
        guard let index = resolvableIndex(challengeID: challengeID, outcome: outcome) else { return false }
        records[index].outcome = outcome
        records[index].resolvedAt = now
        return true
    }

    /// Drop records sent more than a day before `now`.
    mutating func prune(now: Date) {
        records.removeAll { now.timeIntervalSince($0.sentAt) >= LichessBotChallengeCredits.dayWindow }
    }

    /// Counts over records sent within the last day (credits also within
    /// the last minute) of `now`.
    func summary(now: Date) -> Summary {
        var summary = Summary()
        for record in records where now.timeIntervalSince(record.sentAt) < LichessBotChallengeCredits.dayWindow {
            summary.creditsLastDay += record.creditCost
            if now.timeIntervalSince(record.sentAt) < LichessBotChallengeCredits.minuteWindow {
                summary.creditsLastMinute += record.creditCost
            }
            switch record.outcome {
            case .none:
                summary.pending += 1
            case .accepted:
                summary.accepted += 1
            case .declined(let reason):
                summary.declined += 1
                summary.declinedByReason[reason, default: 0] += 1
            case .canceled:
                summary.canceled += 1
            case .offline:
                summary.offline += 1
            case .refused(let refusal):
                summary.refused += 1
                summary.refusedByKind[refusal.kind, default: 0] += 1
            }
        }
        return summary
    }

    static func load(from url: URL) throws -> LichessBotChallengeOutcomeLog {
        guard FileManager.default.fileExists(atPath: url.path) else {
            return LichessBotChallengeOutcomeLog()
        }
        return try decoder.decode(LichessBotChallengeOutcomeLog.self, from: Data(contentsOf: url))
    }

    func save(to url: URL) throws {
        try LichessBotAtomicWrite.write(Self.encoder.encode(self), to: url)
    }

    private static let encoder: JSONEncoder = {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        return encoder
    }()

    private static let decoder: JSONDecoder = {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return decoder
    }()
}
