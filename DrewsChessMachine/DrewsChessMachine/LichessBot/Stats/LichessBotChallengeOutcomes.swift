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
    /// Lichess refused the challenge POST: no challenge was created, and no
    /// challenge credits were spent.
    case refused(LichessBotChallengeRefusal)

    /// A challenge was created on Lichess (and so spent credits).
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
enum LichessBotDeclineReasonRecord: Sendable, Hashable, Codable {
    case known(LichessBotDeclineReason)
    case unrecognized(String)
    /// The decline event carried no reason key.
    case unstated

    init(reasonKey: String?) {
        guard let reasonKey else {
            self = .unstated
            return
        }
        if let known = LichessBotDeclineReason(rawValue: reasonKey) {
            self = .known(known)
        } else {
            self = .unrecognized(reasonKey)
        }
    }

    /// The key as Lichess spells it, for tables and logs.
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
    /// `following` is whether DCM's account follows them; nil when Lichess
    /// didn't say.
    case human(following: Bool?)
}

/// Lichess's challenge credits: every created challenge spends some, within
/// a per-minute and a per-rolling-day budget.
enum LichessBotChallengeCredits {
    static let perDay = 200
    static let perMinute = 25
    static let dayWindow: TimeInterval = 24 * 3600
    static let minuteWindow: TimeInterval = 60

    /// What creating a challenge to `kind` costs. A human whose follow state
    /// Lichess didn't report is charged the non-followed cost, the most it
    /// can be, so the budget shown is never understated; the caller logs
    /// that the follow state was unknown.
    static func cost(for kind: LichessBotChallengeOpponentKind) -> Int {
        switch kind {
        case .bot:
            return 1
        case .human(following: true):
            return 0
        case .human(following: false), .human(following: nil):
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
    /// Credits spent: the opponent's cost for a created challenge, zero
    /// otherwise.
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

    /// An attempt that created no challenge (`offline` or `refused`).
    /// Returns false, recording nothing, for an outcome that implies a
    /// created challenge.
    @discardableResult
    mutating func recordNotCreated(opponentID: String, kind: LichessBotChallengeOpponentKind, outcome: LichessBotChallengeOutcome, at now: Date, makeID: () -> UUID = { UUID() }) -> Bool {
        guard !outcome.challengeWasCreated else { return false }
        records.append(LichessBotChallengeOutcomeRecord(
            id: makeID(), sentAt: now, opponentID: opponentID.lowercased(), opponentKind: kind,
            challengeID: nil, creditCost: 0, outcome: outcome, resolvedAt: now
        ))
        return true
    }

    /// Set a pending challenge's outcome. Only the first answer counts: a
    /// challenge already resolved (say, accepted, then reported canceled by
    /// a late cleanup) keeps its outcome. Returns whether a record changed.
    @discardableResult
    mutating func resolve(challengeID: String, outcome: LichessBotChallengeOutcome, at now: Date) -> Bool {
        guard outcome.challengeWasCreated,
              let index = records.lastIndex(where: { $0.challengeID == challengeID }),
              records[index].outcome == nil else { return false }
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
