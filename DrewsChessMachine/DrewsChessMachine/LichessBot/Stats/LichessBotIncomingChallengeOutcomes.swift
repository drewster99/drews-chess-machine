import Foundation

/// How a challenge another player sent DCM ended, as DCM's records tell it.
/// The single classification the incoming counts on the Overview read; the
/// incoming counterpart of `LichessBotChallengeOutcome`.
enum LichessBotIncomingChallengeOutcome: Sendable, Hashable {
    /// Its game started (or, for a challenge rebuilt from the protocol log,
    /// the acceptance was logged though the game start may not have been).
    case accepted
    /// DCM accepted it, Lichess raised no error, and no game start has been
    /// recorded. Normally a matter of seconds; a count that stays up means
    /// accepted challenges whose games never began.
    case acceptedNoGameStarted
    /// Declined: Lichess's reason when it reported the decline, else the
    /// reason DCM declined with.
    case declined(LichessBotDeclineReasonRecord)
    /// The challenger withdrew it, or Lichess reported it canceled
    /// (expired) before any answer.
    case canceledByChallenger
    /// DCM's policy left it unanswered on purpose.
    case ignored
    /// DCM's latest answer was sent and Lichess (or the transport) failed
    /// it, and nothing after says how the challenge ended.
    case answerFailed(LichessBotIncomingAnswerFailure)
    /// No decision by DCM is recorded and nothing ended it: still waiting,
    /// or expired without Lichess reporting it. The records can't tell the
    /// two apart.
    case notAnswered
}

/// A failed accept or decline of an incoming challenge.
struct LichessBotIncomingAnswerFailure: Sendable, Hashable {
    /// Which answer failed: only these two send anything (an ignore sends
    /// nothing, so it can't fail).
    enum Answer: String, Sendable, Hashable, CaseIterable {
        case accept
        case decline

        var label: String {
            switch self {
            case .accept: return "accept failed"
            case .decline: return "decline failed"
            }
        }
    }

    let answer: Answer
    let at: Date
    /// The error as recorded (redacted). The challenge log keeps only this
    /// text, not an HTTP status or kind, so failures aren't broken down
    /// further.
    let error: String
}

/// One incoming challenge received within the rolling day.
struct LichessBotIncomingChallengeRecord: Sendable, Hashable, Identifiable {
    /// The challenge id.
    let id: String
    let receivedAt: Date
    /// Lowercased Lichess user id; nil when no record names the challenger.
    let challengerID: String?
    let outcome: LichessBotIncomingChallengeOutcome
}

/// The incoming challenges of the rolling day, folded from the challenge log
/// (plus the rebuilt history for the part of the day before the live log
/// began), by the same rules and window as the outgoing outcome log
/// (`LichessBotChallengeOutcomeLog.fold`). Pure; callers pass the time.
struct LichessBotIncomingChallengeOutcomeLog: Sendable, Equatable {
    /// Oldest first; ties broken by challenge id so a refold is equal.
    let records: [LichessBotIncomingChallengeRecord]

    /// Rolling counts over the last day, for the Overview.
    struct Summary: Sendable, Equatable {
        var received = 0
        var accepted = 0
        var acceptedNoGameStarted = 0
        var declined = 0
        var canceledByChallenger = 0
        var ignored = 0
        var answerFailed = 0
        var notAnswered = 0
        var declinedByReason: [LichessBotDeclineReasonRecord: Int] = [:]
        var answerFailedByAnswer: [LichessBotIncomingAnswerFailure.Answer: Int] = [:]
    }

    /// Counts over records received within the last day of `now`.
    func summary(now: Date) -> Summary {
        var summary = Summary()
        for record in records where now.timeIntervalSince(record.receivedAt) < LichessBotChallengeCredits.dayWindow {
            summary.received += 1
            switch record.outcome {
            case .accepted:
                summary.accepted += 1
            case .acceptedNoGameStarted:
                summary.acceptedNoGameStarted += 1
            case .declined(let reason):
                summary.declined += 1
                summary.declinedByReason[reason, default: 0] += 1
            case .canceledByChallenger:
                summary.canceledByChallenger += 1
            case .ignored:
                summary.ignored += 1
            case .answerFailed(let failure):
                summary.answerFailed += 1
                summary.answerFailedByAnswer[failure.answer, default: 0] += 1
            case .notAnswered:
                summary.notAnswered += 1
            }
        }
        return summary
    }

    /// The most recent failed answer, by when it failed, among the records
    /// received within the last day of `now`: the same records `summary`
    /// counts. The log is refolded only when a fact arrives, so records age
    /// out between refolds and the window is applied at the reader's time.
    func latestAnswerFailure(now: Date) -> (record: LichessBotIncomingChallengeRecord, failure: LichessBotIncomingAnswerFailure)? {
        var latest: (record: LichessBotIncomingChallengeRecord, failure: LichessBotIncomingAnswerFailure)?
        for record in records where now.timeIntervalSince(record.receivedAt) < LichessBotChallengeCredits.dayWindow {
            guard case .answerFailed(let failure) = record.outcome else { continue }
            if let current = latest, current.failure.at >= failure.at { continue }
            latest = (record, failure)
        }
        return latest
    }

    static func fold(
        ledger: LichessBotChallengeLedger,
        history: LichessBotChallengeReconstruction?,
        liveLogFirstEntryAt: Date?,
        now: Date
    ) -> LichessBotIncomingChallengeOutcomeLog {
        let window = LichessBotChallengeCredits.dayWindow
        var records: [LichessBotIncomingChallengeRecord] = []
        // A row whose latest fact is a day old can't have been received
        // within the day: skipped before its facts are read, since the
        // ledger holds every challenge ever and this runs for every fact.
        for row in ledger.unorderedRows where now.timeIntervalSince(row.lastAt) < window {
            guard let record = record(from: row), now.timeIntervalSince(record.receivedAt) < window else { continue }
            records.append(record)
        }
        if let history {
            for row in history.rows where now.timeIntervalSince(row.firstAt) < window {
                if let liveLogFirstEntryAt, row.firstAt >= liveLogFirstEntryAt { continue }
                if case .challenge(let id) = row.key, ledger.row(challengeID: id) != nil { continue }
                if let record = record(from: row) {
                    records.append(record)
                }
            }
        }
        return LichessBotIncomingChallengeOutcomeLog(records: records.sorted { lhs, rhs in
            if lhs.receivedAt != rhs.receivedAt { return lhs.receivedAt < rhs.receivedAt }
            return lhs.id < rhs.id
        })
    }

    // MARK: - Live rows

    private static func record(from row: LichessBotChallengeLedgerRow) -> LichessBotIncomingChallengeRecord? {
        guard row.direction == .incoming, case .challenge(let id) = row.key else { return nil }
        let outcome: LichessBotIncomingChallengeOutcome
        switch row.state {
        case .accepted:
            outcome = .accepted
        case .declined(let reason):
            outcome = .declined(reason)
        case .canceledByChallenger:
            outcome = .canceledByChallenger
        case .incomingDecided(let decision):
            outcome = Self.outcome(decidedBy: decision, decidedAt: row.decisions.last?.at, failures: row.responseFailures)
        case .open:
            outcome = .notAnswered
        case .withdrawn, .notCreated, .canceledOnLichessWithoutRecordedWithdrawal, .canceledOnLichessDirectionNotRecorded:
            // Outgoing-only states, or a direction no fact records; an
            // incoming row never folds to them.
            return nil
        }
        let received = row.receivedFacts.first
        return LichessBotIncomingChallengeRecord(
            id: id, receivedAt: received?.at ?? row.firstAt,
            challengerID: received?.challenge.challenger.id.lowercased(), outcome: outcome
        )
    }

    /// The outcome of a row whose last fact is DCM's decision: a failure
    /// recorded at or after the decision means the decision's answer
    /// didn't go through.
    private static func outcome(
        decidedBy decision: LichessBotIncomingDecisionRecord,
        decidedAt: Date?,
        failures: [(at: Date, error: String)]
    ) -> LichessBotIncomingChallengeOutcome {
        let failureAfterDecision: (at: Date, error: String)? = {
            guard let decidedAt, let last = failures.last, last.at >= decidedAt else { return nil }
            return last
        }()
        switch decision {
        case .accept:
            if let failure = failureAfterDecision {
                return .answerFailed(LichessBotIncomingAnswerFailure(answer: .accept, at: failure.at, error: failure.error))
            }
            return .acceptedNoGameStarted
        case .decline(let reason, _):
            if let failure = failureAfterDecision {
                return .answerFailed(LichessBotIncomingAnswerFailure(answer: .decline, at: failure.at, error: failure.error))
            }
            return .declined(.known(reason))
        case .ignore:
            return .ignored
        }
    }

    // MARK: - Rebuilt rows

    private static func record(from row: LichessBotReconstructedChallengeRow) -> LichessBotIncomingChallengeRecord? {
        guard row.direction == .incoming,
              case .challenge(let id) = row.key,
              case .challenge(let state) = row.state else { return nil }
        let outcome: LichessBotIncomingChallengeOutcome
        switch state {
        case .accepted:
            outcome = .accepted
        case .declined(let reason):
            outcome = .declined(reason)
        case .canceledByChallenger:
            outcome = .canceledByChallenger
        case .incomingDecided(let decision):
            // The rebuilt history keeps no response failures.
            outcome = Self.outcome(decidedBy: decision, decidedAt: nil, failures: [])
        case .open:
            outcome = .notAnswered
        case .withdrawn, .notCreated, .canceledOnLichessWithoutRecordedWithdrawal, .canceledOnLichessDirectionNotRecorded:
            return nil
        }
        return LichessBotIncomingChallengeRecord(
            id: id, receivedAt: row.firstAt,
            challengerID: (row.opponent?.id ?? row.challenge?.challenger.id)?.lowercased(), outcome: outcome
        )
    }
}
