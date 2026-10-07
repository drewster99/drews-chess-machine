import Foundation

/// One row of the Challenge Log window (challenge-log plan §3.9): a challenge
/// or a send that created none, from the live challenge log's ledger or
/// rebuilt from the protocol log. A plain value built by `rows(ledger:…)`,
/// the one place the two sources are merged; the window only filters, sorts
/// and shows these.
///
/// Every column's "not recorded" is its own case rather than an empty value,
/// so the window always says why a cell has nothing to show.
struct LichessBotChallengeLogRow: Identifiable, Sendable, Equatable {

    /// What a row is keyed by. A challenge's key is its Lichess id whichever
    /// source holds it, so a live row and a rebuilt row for the same
    /// challenge collide (the live one wins). Sends that created none are
    /// keyed by their own source's name for them, so a live attempt and a
    /// rebuilt one never collide.
    enum Key: Sendable, Hashable {
        case challenge(id: String)
        /// A live send that created no challenge, by its attempt id.
        case liveAttempt(UUID)
        /// A rebuilt send that created no challenge, by the protocol line
        /// recording its outcome.
        case reconstructedAttempt(LichessBotProtocolLineReference)
    }

    /// The other player.
    struct Player: Sendable, Equatable {
        /// As recorded: a Lichess id (lowercase) where the source had one.
        let id: String
        let name: String
        let title: String?
        let rating: Int?
    }

    /// Who the challenge was with.
    enum Opponent: Sendable, Equatable {
        case player(Player)
        /// An open challenge, which names no opponent.
        case openChallenge
        /// No fact the row holds names the other player.
        case notRecorded
    }

    /// What was asked for.
    struct Terms: Sendable, Equatable {
        let rated: Bool
        /// Real-time clock; nil for correspondence or unlimited.
        let limitSeconds: Int?
        let incrementSeconds: Int?
        /// Correspondence; nil otherwise.
        let daysPerTurn: Int?
        /// The color asked for (DCM's when DCM sent it, the challenger's
        /// when someone else did).
        let color: LichessBotOpenValue<LichessBotChallengeColorName>
    }

    /// The "Sender / Decision" column: who sent DCM's challenge, or what
    /// DCM decided on someone else's.
    enum Initiative: Sendable, Equatable {
        /// DCM's challenge, sent from this part of DCM (live log).
        case sentBy(LichessBotChallengeSender)
        /// DCM's challenge as the protocol log tells it.
        case reconstructedSender(LichessBotReconstructedSenderFinding)
        /// DCM's challenge, seen only as its echo, with no send of the run
        /// that saw it to explain it.
        case senderNotRecorded
        /// Someone else's challenge, and DCM's latest decision on it.
        case decided(LichessBotIncomingDecisionRecord)
        /// Someone else's challenge, with no decision recorded.
        case noDecisionRecorded
        /// No fact says which way the challenge went (only Lichess's
        /// answers, or facts of both directions).
        case directionNotRecorded
    }

    /// Where the challenge's life ended up.
    enum State: Sendable, Equatable {
        /// The ledger's state (§3.4). `isPending` is whether the challenge
        /// is among the controller's unanswered challenges right now, which
        /// tells "Waiting" from "No answer recorded" for an open one.
        case challenge(LichessBotChallengeLogState, isPending: Bool)
        /// A rebuilt send that created no challenge.
        case reconstructedNotCreated(LichessBotReconstructedNotCreatedReason)
    }

    /// Lichess challenge credits the send cost.
    enum Credits: Sendable, Equatable {
        case spent(Int)
        /// Someone else's challenge: DCM spent nothing.
        case notApplicable
        /// DCM's challenge whose cost no fact records (an echo, or a row
        /// rebuilt from the protocol log).
        case notRecorded
    }

    /// Where the row came from.
    enum Source: Sendable, Equatable {
        case live
        /// Rebuilt from the protocol log, with how its origin is known (nil
        /// when its sender is not determined).
        case reconstructed(LichessBotReconstructionConfidence?)
    }

    let key: Key
    /// The row's first fact.
    let at: Date
    /// Nil when no fact says (`Initiative.directionNotRecorded`).
    let direction: LichessBotChallengeLogDirection?
    let opponent: Opponent
    /// Nil when no fact records them.
    let terms: Terms?
    let initiative: Initiative
    let state: State
    /// What outranked facts said on the way to an accepted state.
    let notes: [LichessBotChallengeLedgerNote]
    /// What only a code bug or a lost line explains (live rows only).
    let anomalies: [LichessBotChallengeLedgerAnomaly]
    let credits: Credits
    let source: Source

    var id: Key { key }

    /// The game the challenge became, when it was accepted (a challenge's
    /// game has the challenge's id).
    var gameID: String? {
        guard case .challenge(let id) = key, case .challenge(.accepted, _) = state else { return nil }
        return id
    }

    var opponentSortKey: String {
        switch opponent {
        case .player(let player): return player.name.lowercased()
        case .openChallenge, .notRecorded: return ""
        }
    }

    var ratingSortKey: Int {
        guard case .player(let player) = opponent, let rating = player.rating else { return Int.min }
        return rating
    }

    var isReconstructed: Bool {
        if case .reconstructed = source { return true }
        return false
    }

    // MARK: - Building

    /// Every row, newest first: the ledger's rows, plus the rebuilt rows
    /// whose challenge the ledger doesn't hold (live rows win on an id both
    /// hold; sends that created none never collide). `pendingChallengeIDs`
    /// are the controller's unanswered challenges.
    static func rows(ledger: LichessBotChallengeLedger?,
                     reconstruction: LichessBotChallengeReconstruction?,
                     pendingChallengeIDs: Set<String>) -> [LichessBotChallengeLogRow] {
        var rows: [LichessBotChallengeLogRow] = []
        var liveChallengeIDs: Set<String> = []
        for ledgerRow in ledger?.rows ?? [] {
            let row = LichessBotChallengeLogRow(live: ledgerRow, pendingChallengeIDs: pendingChallengeIDs)
            if case .challenge(let id) = row.key {
                liveChallengeIDs.insert(id)
            }
            rows.append(row)
        }
        for rebuilt in reconstruction?.rows ?? [] {
            if case .challenge(let id) = rebuilt.key, liveChallengeIDs.contains(id) {
                continue
            }
            rows.append(LichessBotChallengeLogRow(reconstructed: rebuilt, pendingChallengeIDs: pendingChallengeIDs))
        }
        return rows.sorted { lhs, rhs in
            if lhs.at != rhs.at { return lhs.at > rhs.at }
            return lhs.tieBreak < rhs.tieBreak
        }
    }

    /// A total order between rows that start at the same moment.
    private var tieBreak: String {
        switch key {
        case .challenge(let id): return "c:\(id)"
        case .liveAttempt(let id): return "l:\(id.uuidString)"
        case .reconstructedAttempt(let line): return "r:\(line.file):\(String(format: "%09d", line.line))"
        }
    }

    init(key: Key, at: Date, direction: LichessBotChallengeLogDirection?, opponent: Opponent, terms: Terms?,
         initiative: Initiative, state: State, notes: [LichessBotChallengeLedgerNote],
         anomalies: [LichessBotChallengeLedgerAnomaly], credits: Credits, source: Source) {
        self.key = key
        self.at = at
        self.direction = direction
        self.opponent = opponent
        self.terms = terms
        self.initiative = initiative
        self.state = state
        self.notes = notes
        self.anomalies = anomalies
        self.credits = credits
        self.source = source
    }

    /// A live ledger row.
    init(live row: LichessBotChallengeLedgerRow, pendingChallengeIDs: Set<String>) {
        let key: Key
        let isPending: Bool
        switch row.key {
        case .challenge(let id):
            key = .challenge(id: id)
            isPending = pendingChallengeIDs.contains(id)
        case .attempt(let id):
            key = .liveAttempt(id)
            isPending = false
        }
        let direction = row.direction
        let notCreated = row.notCreatedFacts.first
        let snapshot = row.snapshot

        let opponent: Opponent
        if let notCreated {
            opponent = .player(Player(id: notCreated.opponentID, name: notCreated.opponentID, title: nil, rating: nil))
        } else {
            opponent = Self.opponent(of: snapshot, direction: direction)
        }

        let terms: Terms?
        if let snapshot {
            terms = Terms(snapshot)
        } else if let notCreated {
            terms = Terms(notCreated.request)
        } else {
            terms = nil
        }

        let initiative: Initiative
        let credits: Credits
        switch direction {
        case .outgoing?:
            initiative = row.sender.map(Initiative.sentBy) ?? .senderNotRecorded
            if let created = row.createdFacts.first {
                credits = .spent(created.creditCost)
            } else if let notCreated {
                credits = .spent(notCreated.creditCost)
            } else {
                credits = .notRecorded
            }
        case .incoming?:
            initiative = row.decisions.last.map { Initiative.decided($0.decision) } ?? .noDecisionRecorded
            credits = .notApplicable
        case nil:
            initiative = .directionNotRecorded
            credits = .notRecorded
        }

        self.init(key: key, at: row.firstAt, direction: direction, opponent: opponent, terms: terms,
                  initiative: initiative, state: .challenge(row.state, isPending: isPending),
                  notes: row.notes, anomalies: row.anomalies, credits: credits, source: .live)
    }

    /// A row rebuilt from the protocol log.
    init(reconstructed row: LichessBotReconstructedChallengeRow, pendingChallengeIDs: Set<String>) {
        let key: Key
        let isPending: Bool
        switch row.key {
        case .challenge(let id):
            key = .challenge(id: id)
            isPending = pendingChallengeIDs.contains(id)
        case .notCreatedAttempt(let line):
            key = .reconstructedAttempt(line)
            isPending = false
        }

        let opponent: Opponent
        if let snapshot = row.challenge, case .player(let party) = Self.opponent(of: snapshot, direction: row.direction) {
            opponent = .player(party)
        } else if let named = row.opponent {
            opponent = .player(Player(id: named.id, name: named.name, title: nil, rating: nil))
        } else {
            opponent = Self.opponent(of: row.challenge, direction: row.direction)
        }

        let terms: Terms?
        if let snapshot = row.challenge {
            terms = Terms(snapshot)
        } else if let sendTerms = row.sendTerms {
            terms = Terms(sendTerms)
        } else {
            terms = nil
        }

        let initiative: Initiative
        let credits: Credits
        switch row.direction {
        case .outgoing:
            initiative = row.sender.map(Initiative.reconstructedSender) ?? .senderNotRecorded
            credits = .notRecorded
        case .incoming:
            initiative = row.decision.map(Initiative.decided) ?? .noDecisionRecorded
            credits = .notApplicable
        }

        let state: State
        switch row.state {
        case .challenge(let logState):
            state = .challenge(logState, isPending: isPending)
        case .notCreated(let reason):
            state = .reconstructedNotCreated(reason)
        }

        self.init(key: key, at: row.firstAt, direction: row.direction, opponent: opponent, terms: terms,
                  initiative: initiative, state: state, notes: row.notes, anomalies: [],
                  credits: credits, source: .reconstructed(row.originConfidence))
    }

    /// The other player in a challenge's snapshot: the challenged player of
    /// DCM's challenge, the challenger of someone else's. With no direction
    /// the other side can't be told.
    private static func opponent(of snapshot: LichessBotChallengeSnapshot?, direction: LichessBotChallengeLogDirection?) -> Opponent {
        guard let snapshot else { return .notRecorded }
        switch direction {
        case .outgoing?:
            guard let destUser = snapshot.destUser else { return .openChallenge }
            return .player(Player(destUser))
        case .incoming?:
            return .player(Player(snapshot.challenger))
        case nil:
            return .notRecorded
        }
    }
}

extension LichessBotChallengeLogRow.Player {
    init(_ party: LichessBotChallengeParty) {
        self.init(id: party.id, name: party.name, title: party.title, rating: party.rating)
    }
}

extension LichessBotChallengeLogRow.Terms {
    init(_ snapshot: LichessBotChallengeSnapshot) {
        self.init(rated: snapshot.rated, limitSeconds: snapshot.limitSeconds, incrementSeconds: snapshot.incrementSeconds,
                  daysPerTurn: snapshot.daysPerTurn, color: snapshot.color)
    }

    init(_ request: LichessBotOutgoingChallenge) {
        self.init(rated: request.rated, limitSeconds: request.clockLimitSeconds, incrementSeconds: request.clockIncrementSeconds,
                  daysPerTurn: nil, color: LichessBotOpenValue(request.color))
    }
}

// MARK: - Filters

/// The kinds of state the Challenge Log window filters by.
enum LichessBotChallengeLogStateKind: String, Sendable, Hashable, CaseIterable {
    /// Open, and among the controller's unanswered challenges.
    case waiting
    /// Open, and not pending: no answer was recorded.
    case noAnswerRecorded
    case accepted
    case declined
    /// DCM withdrew it (with or without a recorded reason).
    case withdrawn
    /// The challenger canceled it, or Lichess reported it canceled with no
    /// direction recorded.
    case canceled
    case notCreated
    /// Someone else's challenge DCM decided on, with nothing after.
    case decidedByDCM

    init(_ state: LichessBotChallengeLogRow.State) {
        switch state {
        case .challenge(let logState, let isPending):
            switch logState {
            case .open: self = isPending ? .waiting : .noAnswerRecorded
            case .accepted: self = .accepted
            case .declined: self = .declined
            case .withdrawn, .canceledOnLichessWithoutRecordedWithdrawal: self = .withdrawn
            case .canceledByChallenger, .canceledOnLichessDirectionNotRecorded: self = .canceled
            case .notCreated: self = .notCreated
            case .incomingDecided: self = .decidedByDCM
            }
        case .reconstructedNotCreated:
            self = .notCreated
        }
    }
}

/// The kinds of sender (or, for someone else's challenge, "incoming") the
/// Challenge Log window filters by.
enum LichessBotChallengeLogSenderKind: String, Sendable, Hashable, CaseIterable {
    case challengeSheet
    case casualResendOffer
    case challengeQueue
    case matchmaking
    case matchmakingCasualResend
    /// Rebuilt from the protocol log, which can't tell the Challenge sheet
    /// from Resend as Casual.
    case byOperator
    /// Someone else's challenge.
    case incoming
    /// The sender (or the direction) is not recorded.
    case notRecorded

    init(_ initiative: LichessBotChallengeLogRow.Initiative) {
        switch initiative {
        case .sentBy(let sender):
            switch sender {
            case .challengeSheet: self = .challengeSheet
            case .casualResendOffer: self = .casualResendOffer
            case .challengeQueue: self = .challengeQueue
            case .matchmaking: self = .matchmaking
            case .matchmakingCasualResend: self = .matchmakingCasualResend
            }
        case .reconstructedSender(.attributed(let sender, _)):
            switch sender {
            case .byOperator: self = .byOperator
            case .challengeQueue: self = .challengeQueue
            case .matchmaking: self = .matchmaking
            case .matchmakingCasualResend: self = .matchmakingCasualResend
            }
        case .reconstructedSender(.ambiguousCompanion), .reconstructedSender(.noSendLine), .senderNotRecorded, .directionNotRecorded:
            self = .notRecorded
        case .decided, .noDecisionRecorded:
            self = .incoming
        }
    }
}

/// What the Challenge Log window shows (challenge-log plan §3.9).
struct LichessBotChallengeLogFilter: Sendable, Equatable {

    enum DateRange: String, Sendable, Hashable, CaseIterable {
        case last24Hours
        case last7Days
        case last30Days
        case all

        /// How far back from now; nil for all.
        var lookback: TimeInterval? {
            switch self {
            case .last24Hours: return 24 * 3600
            case .last7Days: return 7 * 24 * 3600
            case .last30Days: return 30 * 24 * 3600
            case .all: return nil
            }
        }
    }

    enum DirectionChoice: String, Sendable, Hashable, CaseIterable {
        case all
        case outgoing
        case incoming
    }

    /// One state kind, or every state.
    enum StateChoice: Sendable, Hashable {
        case all
        case only(LichessBotChallengeLogStateKind)
    }

    /// One sender kind, or every sender.
    enum SenderChoice: Sendable, Hashable {
        case all
        case only(LichessBotChallengeLogSenderKind)
    }

    var dateRange: DateRange = .all
    var direction: DirectionChoice = .all
    var state: StateChoice = .all
    var sender: SenderChoice = .all
    var includeReconstructed = true
    /// Matched case-insensitively against the opponent's id and name;
    /// empty matches every row.
    var opponentSearch = ""

    /// The rows this filter shows, in their given order. A row whose
    /// opponent is not recorded never matches a non-empty search, and a
    /// row whose direction is not recorded shows only under "All".
    func apply(to rows: [LichessBotChallengeLogRow], now: Date) -> [LichessBotChallengeLogRow] {
        let earliest = dateRange.lookback.map { now.addingTimeInterval(-$0) }
        let search = opponentSearch.trimmingCharacters(in: .whitespaces).lowercased()
        return rows.filter { row in
            if let earliest, row.at < earliest { return false }
            if !includeReconstructed, row.isReconstructed { return false }
            switch direction {
            case .all: break
            case .outgoing: guard row.direction == .outgoing else { return false }
            case .incoming: guard row.direction == .incoming else { return false }
            }
            if case .only(let kind) = state, LichessBotChallengeLogStateKind(row.state) != kind { return false }
            if case .only(let kind) = sender, LichessBotChallengeLogSenderKind(row.initiative) != kind { return false }
            if !search.isEmpty {
                guard case .player(let player) = row.opponent,
                      player.id.lowercased().contains(search) || player.name.lowercased().contains(search) else { return false }
            }
            return true
        }
    }
}

/// The footer's tallies of the shown rows.
struct LichessBotChallengeLogCounts: Sendable, Equatable {
    var shown = 0
    var outgoing = 0
    var incoming = 0
    var directionNotRecorded = 0
    var live = 0
    var reconstructed = 0
    /// Credits the shown rows record DCM spending.
    var creditsSpent = 0

    init() {}

    init(_ rows: [LichessBotChallengeLogRow]) {
        for row in rows {
            shown += 1
            switch row.direction {
            case .outgoing?: outgoing += 1
            case .incoming?: incoming += 1
            case nil: directionNotRecorded += 1
            }
            if row.isReconstructed {
                reconstructed += 1
            } else {
                live += 1
            }
            if case .spent(let credits) = row.credits {
                creditsSpent += credits
            }
        }
    }
}

extension LichessBotGameOriginCategory {
    /// The category a game from DCM's challenge shows, by who sent it. The
    /// same mapping as `LichessBotGameOriginDisplay.display(_:basis:)`
    /// (pinned by a test), for views that show a sender without a game.
    init(sender: LichessBotChallengeSender) {
        switch sender {
        case .challengeSheet: self = .challengeSheet
        case .casualResendOffer: self = .casualResendOffer
        case .challengeQueue: self = .challengeQueue
        case .matchmaking: self = .matchmaking
        case .matchmakingCasualResend: self = .matchmakingCasualResend
        }
    }
}
