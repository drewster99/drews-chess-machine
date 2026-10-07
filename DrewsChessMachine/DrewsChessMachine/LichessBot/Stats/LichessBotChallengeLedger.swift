import Foundation

/// The fold of the challenge log (challenge-log plan §3.4): one row per
/// challenge, keyed by its id, or per send that created none, keyed by its
/// attempt id. A pure value; the controller holds one, built from the day
/// files and then updated by its one funnel as facts are recorded.
///
/// **Order independence.** Facts reach the ledger in any order: in the POST
/// race the game can start before the line saying the challenge was created,
/// and a challenge's facts can span two day files. So a row keeps its facts
/// in one canonical order (time, then a fixed tie-break) whatever order they
/// were applied in, and everything about the row — its state, sender, notes
/// and anomalies — is derived from them, never from the last fact read. Any
/// permutation of the same facts gives an identical row.
struct LichessBotChallengeLedger: Sendable, Equatable {

    /// How completely the ledger's facts were loaded from disk.
    enum LoadStatus: Sendable, Equatable {
        /// Every day file was read.
        case complete
        /// Some day files were left out (named, with why); their facts are
        /// missing, so a miss on this ledger proves nothing.
        case partial(filesLeftOut: [LichessBotChallengeLogContents.LeftOutFile])
        /// The log could not be read at all; the ledger holds only facts
        /// recorded since.
        case failed(reason: String)
    }

    private(set) var loadStatus: LoadStatus
    private(set) var rowsByKey: [LichessBotChallengeLedgerRowKey: LichessBotChallengeLedgerRow] = [:]
    /// The writer's `unterminatedLineCut` repairs, in canonical order. They
    /// belong to no challenge.
    private(set) var unterminatedLineCuts: [LichessBotChallengeLedgerFact] = []

    /// An empty ledger.
    init(loadStatus: LoadStatus) {
        self.loadStatus = loadStatus
    }

    /// The ledger of what the reader found: `complete` when no day file was
    /// left out, else `partial`.
    init(contents: LichessBotChallengeLogContents) {
        self.init(loadStatus: contents.filesLeftOut.isEmpty ? .complete : .partial(filesLeftOut: contents.filesLeftOut))
        apply(contentsOf: contents.entries)
    }

    /// Fold one entry in.
    mutating func apply(_ entry: LichessBotChallengeLogEntry) {
        let fact = LichessBotChallengeLedgerFact(at: entry.at, event: entry.event)
        guard let key = LichessBotChallengeLedgerRowKey(entry.event) else {
            LichessBotChallengeLedgerFact.insert(fact, into: &unterminatedLineCuts)
            return
        }
        if var row = rowsByKey.removeValue(forKey: key) {
            row.add(fact)
            rowsByKey[key] = row
        } else {
            rowsByKey[key] = LichessBotChallengeLedgerRow(key: key, firstFact: fact)
        }
    }

    mutating func apply<Entries: Sequence>(contentsOf entries: Entries) where Entries.Element == LichessBotChallengeLogEntry {
        for entry in entries {
            apply(entry)
        }
    }

    func row(challengeID: String) -> LichessBotChallengeLedgerRow? {
        rowsByKey[.challenge(id: challengeID)]
    }

    func row(attemptID: UUID) -> LichessBotChallengeLedgerRow? {
        rowsByKey[.attempt(id: attemptID)]
    }

    /// Every row, in no particular order: for a caller that filters before
    /// it sorts (`rows` sorts every row on each access).
    var unorderedRows: Dictionary<LichessBotChallengeLedgerRowKey, LichessBotChallengeLedgerRow>.Values {
        rowsByKey.values
    }

    /// Every row, oldest first fact first.
    var rows: [LichessBotChallengeLedgerRow] {
        rowsByKey.values.sorted { lhs, rhs in
            if lhs.firstAt != rhs.firstAt { return lhs.firstAt < rhs.firstAt }
            return lhs.key.sortKey < rhs.key.sortKey
        }
    }
}

/// What a ledger row is keyed by.
enum LichessBotChallengeLedgerRowKey: Sendable, Hashable {
    /// A challenge Lichess created (or reported), by its id.
    case challenge(id: String)
    /// A send that created no challenge, by its attempt id.
    case attempt(id: UUID)

    /// The row an event belongs to; nil for the writer's housekeeping.
    init?(_ event: LichessBotChallengeLogEvent) {
        switch event {
        case .outgoingCreated(let challenge, _, _, _, _),
             .outgoingSeenWithoutCreatedLine(let challenge, _),
             .incomingReceived(let challenge):
            self = .challenge(id: challenge.id)
        case .outgoingNotCreated(let attemptID, _, _, _, _, _, _):
            self = .attempt(id: attemptID)
        case .withdrawalRequested(let challengeID, _),
             .withdrawalResult(let challengeID, _),
             .incomingDecided(let challengeID, _),
             .incomingResponseFailed(let challengeID, _),
             .declinedOnLichess(let challengeID, _, _),
             .canceledOnLichess(let challengeID),
             .gameStarted(let challengeID):
            self = .challenge(id: challengeID)
        case .unterminatedLineCut:
            return nil
        }
    }

    /// A total order for ties between rows' first facts.
    fileprivate var sortKey: String {
        switch self {
        case .challenge(let id): return "challenge:\(id)"
        case .attempt(let id): return "attempt:\(id.uuidString)"
        }
    }
}

/// One fact of a ledger row: an entry's time and event.
struct LichessBotChallengeLedgerFact: Sendable, Equatable {
    let at: Date
    let event: LichessBotChallengeLogEvent

    /// Insert `fact` into `facts`, kept in the canonical order: by time,
    /// then by the event's description, so equal times still have one
    /// order whatever order the facts arrived in.
    static func insert(_ fact: LichessBotChallengeLedgerFact, into facts: inout [LichessBotChallengeLedgerFact]) {
        let index = facts.firstIndex { fact.sorts(before: $0) } ?? facts.endIndex
        facts.insert(fact, at: index)
    }

    private func sorts(before other: LichessBotChallengeLedgerFact) -> Bool {
        if at != other.at { return at < other.at }
        return String(describing: event) < String(describing: other.event)
    }
}

/// Which way a challenge went.
enum LichessBotChallengeLogDirection: String, Sendable, Codable, Equatable, Hashable, CaseIterable {
    /// DCM challenged someone.
    case outgoing
    /// Someone challenged DCM.
    case incoming
}

/// Where a challenge's life ended up (§3.4), decided by the strongest fact
/// present, never by the last one read.
enum LichessBotChallengeLogState: Sendable, Codable, Equatable {
    /// No terminal fact. The UI shows Waiting while the challenge is
    /// pending, else "No answer recorded" — explicit, never blank.
    case open
    /// Accepted. The ledger derives it only from a `gameStarted` fact
    /// (`gameStarted == true`); the reconstructed history (§3.7) can also
    /// know an acceptance whose game start it never saw.
    case accepted(gameStarted: Bool)
    case declined(LichessBotDeclineReasonRecord)
    /// Incoming, withdrawn by the challenger.
    case canceledByChallenger
    /// DCM withdrew it: why, and what Lichess answered (nil when no answer
    /// was recorded).
    case withdrawn(LichessBotWithdrawalReason, LichessBotWithdrawalResult?)
    /// The send created no challenge.
    case notCreated(LichessBotChallengeNotCreatedReason)
    /// Incoming, decided by DCM, and nothing after.
    case incomingDecided(LichessBotIncomingDecisionRecord)
    /// Outgoing, reported canceled by Lichess with no withdrawal fact naming
    /// it. Only the challenger can cancel, so DCM or another client using
    /// the token withdrew it, but why was not recorded. Shown as
    /// "Withdrawn (reason not recorded)".
    case canceledOnLichessWithoutRecordedWithdrawal
    /// Reported canceled by Lichess, and no fact says which way the
    /// challenge went (no created, echo or received line, or facts that
    /// disagree), so neither of the two cancel states can be claimed.
    case canceledOnLichessDirectionNotRecorded
}

/// A fact that a stronger one outranked in an accepted row, kept so the row
/// still says what happened on the way.
enum LichessBotChallengeLedgerNote: Sendable, Codable, Equatable {
    /// A withdrawal was attempted, but the game started anyway (the cancel
    /// lost the race with the acceptance, or Lichess answered "already
    /// gone").
    case withdrawalAttempted(LichessBotWithdrawalReason, LichessBotWithdrawalResult?)
    /// A decline was reported, and the game still started.
    case declinedOnLichess(LichessBotDeclineReasonRecord)
    /// A cancel was reported, and the game still started.
    case canceledOnLichess
    /// Incoming: DCM did not accept it, so the operator accepted it outside
    /// DCM (by hand on lichess.org). The decision is DCM's latest.
    case acceptedOutsideDCM(dcmDecided: LichessBotIncomingDecisionRecord)
}

/// Something in a row's facts that only a code bug or a lost line explains.
/// Never hidden, never guessed around.
enum LichessBotChallengeLedgerAnomaly: Sendable, Equatable {
    /// More than one `outgoingCreated` line for the id; the first is used.
    case repeatedCreatedLines(count: Int)
    /// More than one `outgoingNotCreated` line for the attempt; the first
    /// is used.
    case repeatedNotCreatedLines(count: Int)
    /// Facts of both directions for one id; the direction is not claimed.
    case conflictingDirections
    /// A withdrawal result with no withdrawal request for the id.
    case withdrawalResultWithoutRequest
}

/// One challenge, or one send that created none, folded from its facts.
struct LichessBotChallengeLedgerRow: Sendable, Equatable {

    /// An `outgoingCreated` fact.
    struct Created: Sendable, Equatable {
        let at: Date
        let challenge: LichessBotChallengeSnapshot
        let sender: LichessBotChallengeSender
        let request: LichessBotOutgoingChallenge
        let opponentKind: LichessBotChallengeOpponentKind
        let creditCost: Int
    }

    /// An `outgoingNotCreated` fact.
    struct NotCreated: Sendable, Equatable {
        let at: Date
        let opponentID: String
        let sender: LichessBotChallengeSender
        let request: LichessBotOutgoingChallenge
        let opponentKind: LichessBotChallengeOpponentKind?
        let reason: LichessBotChallengeNotCreatedReason
        let creditCost: Int
    }

    let key: LichessBotChallengeLedgerRowKey
    /// Every fact, in canonical order (`LichessBotChallengeLedgerFact.insert`).
    /// Never empty: a row exists only because a fact named it.
    private(set) var facts: [LichessBotChallengeLedgerFact]
    /// The earliest fact's time.
    private(set) var firstAt: Date
    /// The latest fact's time.
    private(set) var lastAt: Date

    init(key: LichessBotChallengeLedgerRowKey, firstFact: LichessBotChallengeLedgerFact) {
        self.key = key
        self.facts = [firstFact]
        self.firstAt = firstFact.at
        self.lastAt = firstFact.at
    }

    mutating func add(_ fact: LichessBotChallengeLedgerFact) {
        LichessBotChallengeLedgerFact.insert(fact, into: &facts)
        firstAt = min(firstAt, fact.at)
        lastAt = max(lastAt, fact.at)
    }

    // MARK: Facts by kind (canonical order)

    var createdFacts: [Created] {
        facts.compactMap { fact in
            guard case .outgoingCreated(let challenge, let sender, let request, let opponentKind, let creditCost) = fact.event else { return nil }
            return Created(at: fact.at, challenge: challenge, sender: sender, request: request, opponentKind: opponentKind, creditCost: creditCost)
        }
    }

    var notCreatedFacts: [NotCreated] {
        facts.compactMap { fact in
            guard case .outgoingNotCreated(_, let opponentID, let sender, let request, let opponentKind, let reason, let creditCost) = fact.event else { return nil }
            return NotCreated(at: fact.at, opponentID: opponentID, sender: sender, request: request,
                              opponentKind: opponentKind, reason: reason, creditCost: creditCost)
        }
    }

    var echoFacts: [(at: Date, challenge: LichessBotChallengeSnapshot, attribution: LichessBotEchoAttribution)] {
        facts.compactMap { fact in
            guard case .outgoingSeenWithoutCreatedLine(let challenge, let attribution) = fact.event else { return nil }
            return (fact.at, challenge, attribution)
        }
    }

    var receivedFacts: [(at: Date, challenge: LichessBotChallengeSnapshot)] {
        facts.compactMap { fact in
            guard case .incomingReceived(let challenge) = fact.event else { return nil }
            return (fact.at, challenge)
        }
    }

    var withdrawalRequests: [(at: Date, reason: LichessBotWithdrawalReason)] {
        facts.compactMap { fact in
            guard case .withdrawalRequested(_, let reason) = fact.event else { return nil }
            return (fact.at, reason)
        }
    }

    var withdrawalResults: [(at: Date, result: LichessBotWithdrawalResult)] {
        facts.compactMap { fact in
            guard case .withdrawalResult(_, let result) = fact.event else { return nil }
            return (fact.at, result)
        }
    }

    /// Every decision DCM made on it: each is a real request, so a
    /// reconnect's replay that DCM decided again is recorded again.
    var decisions: [(at: Date, decision: LichessBotIncomingDecisionRecord)] {
        facts.compactMap { fact in
            guard case .incomingDecided(_, let decision) = fact.event else { return nil }
            return (fact.at, decision)
        }
    }

    var responseFailures: [(at: Date, error: String)] {
        facts.compactMap { fact in
            guard case .incomingResponseFailed(_, let error) = fact.event else { return nil }
            return (fact.at, error)
        }
    }

    /// The first decline Lichess reported.
    var declined: (at: Date, reason: LichessBotDeclineReasonRecord, text: String?)? {
        for fact in facts {
            if case .declinedOnLichess(_, let reason, let text) = fact.event {
                return (fact.at, reason, text)
            }
        }
        return nil
    }

    /// When Lichess first reported it canceled.
    var canceledAt: Date? {
        facts.first { if case .canceledOnLichess = $0.event { return true }; return false }?.at
    }

    /// When its game was first seen starting.
    var gameStartedAt: Date? {
        facts.first { if case .gameStarted = $0.event { return true }; return false }?.at
    }

    // MARK: Derived

    /// The facts' direction: outgoing for DCM's sends, echoes and
    /// withdrawals, incoming for received challenges and DCM's decisions on
    /// them. Nil when no fact says, or when facts say both (an anomaly).
    var direction: LichessBotChallengeLogDirection? {
        switch (hasOutgoingFacts, hasIncomingFacts) {
        case (true, false): return .outgoing
        case (false, true): return .incoming
        case (false, false), (true, true): return nil
        }
    }

    private var hasOutgoingFacts: Bool {
        facts.contains { fact in
            switch fact.event {
            case .outgoingCreated, .outgoingNotCreated, .outgoingSeenWithoutCreatedLine, .withdrawalRequested, .withdrawalResult:
                return true
            case .incomingReceived, .incomingDecided, .incomingResponseFailed, .declinedOnLichess, .canceledOnLichess, .gameStarted, .unterminatedLineCut:
                return false
            }
        }
    }

    private var hasIncomingFacts: Bool {
        facts.contains { fact in
            switch fact.event {
            case .incomingReceived, .incomingDecided, .incomingResponseFailed:
                return true
            case .outgoingCreated, .outgoingNotCreated, .outgoingSeenWithoutCreatedLine, .withdrawalRequested, .withdrawalResult,
                 .declinedOnLichess, .canceledOnLichess, .gameStarted, .unterminatedLineCut:
                return false
            }
        }
    }

    /// The challenge as Lichess described it: from the first
    /// `outgoingCreated`, else the first echo, else the first
    /// `incomingReceived`. Nil for a send that created none, and for a
    /// challenge known only by later facts.
    var snapshot: LichessBotChallengeSnapshot? {
        createdFacts.first?.challenge ?? echoFacts.first?.challenge ?? receivedFacts.first?.challenge
    }

    /// Who sent it: the first `outgoingCreated`'s sender. An echo line gives
    /// the sender only when there is no `outgoingCreated` (a teardown during
    /// the POST writes both, and the created line knows better), and only
    /// when it was matched to an unanswered send. A not-created attempt's
    /// sender is its own. Nil for incoming challenges and unattributed echoes.
    var sender: LichessBotChallengeSender? {
        if let created = createdFacts.first {
            return created.sender
        }
        if let notCreated = notCreatedFacts.first {
            return notCreated.sender
        }
        for echo in echoFacts {
            if case .unansweredSend(_, let sender) = echo.attribution {
                return sender
            }
        }
        return nil
    }

    /// The state, by the fold's precedence (§3.4):
    /// 1. a started game → accepted; it outranks everything (the other
    ///    facts become `notes`);
    /// 2. a send that created nothing → notCreated;
    /// 3. a decline → declined;
    /// 4. a withdrawal request → withdrawn (the first request's reason, the
    ///    last result's answer, nil when none was recorded);
    /// 5. a cancel → canceled by the challenger (incoming), withdrawn with no
    ///    recorded reason (outgoing), or direction not recorded;
    /// 6. a decision → incomingDecided (the latest);
    /// 7. otherwise open.
    var state: LichessBotChallengeLogState {
        if gameStartedAt != nil {
            return .accepted(gameStarted: true)
        }
        if let notCreated = notCreatedFacts.first {
            return .notCreated(notCreated.reason)
        }
        if let declined {
            return .declined(declined.reason)
        }
        if let request = withdrawalRequests.first {
            return .withdrawn(request.reason, withdrawalResults.last?.result)
        }
        if canceledAt != nil {
            switch direction {
            case .incoming?: return .canceledByChallenger
            case .outgoing?: return .canceledOnLichessWithoutRecordedWithdrawal
            case nil: return .canceledOnLichessDirectionNotRecorded
            }
        }
        if let decision = decisions.last {
            return .incomingDecided(decision.decision)
        }
        return .open
    }

    /// What an accepted row's outranked facts said; empty unless accepted.
    var notes: [LichessBotChallengeLedgerNote] {
        guard gameStartedAt != nil else { return [] }
        var notes: [LichessBotChallengeLedgerNote] = []
        if let request = withdrawalRequests.first {
            notes.append(.withdrawalAttempted(request.reason, withdrawalResults.last?.result))
        }
        if let declined {
            notes.append(.declinedOnLichess(declined.reason))
        }
        if canceledAt != nil {
            notes.append(.canceledOnLichess)
        }
        if direction == .incoming,
           let latest = decisions.last,
           !decisions.contains(where: { $0.decision == .accept }) {
            notes.append(.acceptedOutsideDCM(dcmDecided: latest.decision))
        }
        return notes
    }

    var anomalies: [LichessBotChallengeLedgerAnomaly] {
        var anomalies: [LichessBotChallengeLedgerAnomaly] = []
        let createdCount = createdFacts.count
        if createdCount > 1 {
            anomalies.append(.repeatedCreatedLines(count: createdCount))
        }
        let notCreatedCount = notCreatedFacts.count
        if notCreatedCount > 1 {
            anomalies.append(.repeatedNotCreatedLines(count: notCreatedCount))
        }
        if hasOutgoingFacts && hasIncomingFacts {
            anomalies.append(.conflictingDirections)
        }
        if !withdrawalResults.isEmpty && withdrawalRequests.isEmpty {
            anomalies.append(.withdrawalResultWithoutRequest)
        }
        return anomalies
    }
}
