import CryptoKit
import Foundation

/// The outgoing-challenge outcome log as a fold of the challenge log
/// (challenge-log plan §3.8, OD-2, P6).
///
/// Until P6 the outcome log was its own file, `challenge-outcomes.json`,
/// rewritten on every change and pruned to the last day, fed by its own
/// calls beside every challenge fact. Two records of the same facts can
/// disagree, so it is now derived: the challenge log's rows of the last day,
/// plus the challenges rebuilt from the protocol log for the part of that day
/// the live log doesn't cover yet (the first day after this build starts
/// recording). The old file is left on disk untouched, and
/// `LichessBotChallengeOutcomeLog.load(from:)` still reads it; nothing
/// writes it any more.
///
/// The fold keeps the old log's meaning:
/// - a created challenge is pending until it is accepted (its game started),
///   declined, or withdrawn or canceled (all `.canceled`); a started game
///   outranks an earlier withdrawal, as `resolve` let an acceptance replace
///   an inferred cancel;
/// - a send to an offline player is `.offline` and costs nothing; a refused
///   POST is `.refused`, charged when Lichess charges for it;
/// - a send whose POST got no answer is left out, as the old log left it
///   out (whether a challenge exists is unknown; its echo, if it comes,
///   records it).
/// Record ids are derived from the row (the attempt id, or a hash of the
/// challenge id or the protocol line), so a refold keeps each record's id
/// and the Overview's lists keep their identity.
extension LichessBotChallengeOutcomeLog {

    static func fold(
        ledger: LichessBotChallengeLedger,
        history: LichessBotChallengeReconstruction?,
        liveLogFirstEntryAt: Date?,
        now: Date
    ) -> LichessBotChallengeOutcomeLog {
        var records: [LichessBotChallengeOutcomeRecord] = []
        var liveChallengeIDs: Set<String> = []
        for row in ledger.rows {
            if case .challenge(let id) = row.key {
                liveChallengeIDs.insert(id)
            }
            if let record = record(from: row) {
                records.append(record)
            }
        }
        if let history {
            for row in history.rows {
                if let liveLogFirstEntryAt, row.firstAt >= liveLogFirstEntryAt { continue }
                if case .challenge(let id) = row.key, liveChallengeIDs.contains(id) { continue }
                if let record = record(from: row) {
                    records.append(record)
                }
            }
        }
        var log = LichessBotChallengeOutcomeLog(records: records.sorted { $0.sentAt < $1.sentAt })
        log.prune(now: now)
        return log
    }

    // MARK: - Live rows

    private static func record(from row: LichessBotChallengeLedgerRow) -> LichessBotChallengeOutcomeRecord? {
        if let created = row.createdFacts.first {
            // DCM only ever challenges a named player, so a created challenge
            // always has one; an open challenge is not DCM's and has no place
            // here.
            guard let opponent = created.challenge.destUser else { return nil }
            let resolution = resolution(of: row)
            return LichessBotChallengeOutcomeRecord(
                id: stableID(for: "challenge:\(created.challenge.id)"), sentAt: created.at,
                opponentID: opponent.id.lowercased(), opponentKind: created.opponentKind,
                challengeID: created.challenge.id, creditCost: created.creditCost,
                outcome: resolution?.outcome, resolvedAt: resolution?.at
            )
        }
        if let notCreated = row.notCreatedFacts.first, case .attempt(let attemptID) = row.key {
            let outcome: LichessBotChallengeOutcome
            switch notCreated.reason {
            case .opponentOffline:
                outcome = .offline
            case .refused(let refusal):
                outcome = .refused(refusal)
            case .noAnswer:
                return nil
            }
            return LichessBotChallengeOutcomeRecord(
                id: attemptID, sentAt: notCreated.at, opponentID: notCreated.opponentID.lowercased(),
                // This build always records the kind. A line without one is
                // labeled at the worst case; its cost is the charge the line
                // recorded either way.
                opponentKind: notCreated.opponentKind ?? .human, challengeID: nil, creditCost: notCreated.creditCost,
                outcome: outcome, resolvedAt: notCreated.at
            )
        }
        return nil
    }

    /// A created challenge's answer and when it came, or nil while pending.
    private static func resolution(of row: LichessBotChallengeLedgerRow) -> (outcome: LichessBotChallengeOutcome, at: Date)? {
        switch row.state {
        case .accepted:
            guard let at = row.gameStartedAt else { return nil }
            return (.accepted, at)
        case .declined(let reason):
            guard let at = row.declined?.at else { return nil }
            return (.declined(reason), at)
        case .withdrawn:
            guard let at = row.withdrawalRequests.first?.at else { return nil }
            return (.canceled, at)
        case .canceledByChallenger, .canceledOnLichessWithoutRecordedWithdrawal, .canceledOnLichessDirectionNotRecorded:
            guard let at = row.canceledAt else { return nil }
            return (.canceled, at)
        case .open, .notCreated, .incomingDecided:
            return nil
        }
    }

    // MARK: - Rebuilt rows

    private static func record(from row: LichessBotReconstructedChallengeRow) -> LichessBotChallengeOutcomeRecord? {
        guard row.direction == .outgoing else { return nil }
        switch (row.key, row.state) {
        case (.challenge(let id), .challenge(let state)):
            guard let opponentID = row.opponent?.id ?? row.challenge?.destUser?.id else { return nil }
            // Rebuilt rows carry no cost; a bot's challenge costs a bot's
            // credits, anyone else's the worst case (a human's).
            let kind: LichessBotChallengeOpponentKind = row.challenge?.destUser?.title == "BOT" ? .bot : .human
            let outcome: LichessBotChallengeOutcome?
            switch state {
            case .accepted: outcome = .accepted
            case .declined(let reason): outcome = .declined(reason)
            case .withdrawn, .canceledByChallenger, .canceledOnLichessWithoutRecordedWithdrawal, .canceledOnLichessDirectionNotRecorded:
                outcome = .canceled
            case .open, .notCreated, .incomingDecided:
                outcome = nil
            }
            return LichessBotChallengeOutcomeRecord(
                id: stableID(for: "challenge:\(id)"), sentAt: row.firstAt, opponentID: opponentID.lowercased(),
                opponentKind: kind, challengeID: id, creditCost: LichessBotChallengeCredits.cost(for: kind),
                // The rebuilt row doesn't keep when the answer came; its
                // first evidence stands in, which only the Overview's
                // ordering reads.
                outcome: outcome, resolvedAt: outcome == nil ? nil : row.firstAt
            )
        case (.notCreatedAttempt(let line), .notCreated(let reason)):
            guard let opponentID = row.opponent?.id else { return nil }
            let outcome: LichessBotChallengeOutcome
            let kind: LichessBotChallengeOpponentKind
            switch reason {
            case .opponentOffline:
                outcome = .offline
                kind = .human
            case .refused(let refusal):
                outcome = .refused(refusal)
                // A bot-vs-bot limit refusal is about a bot; any other
                // refusal is costed at the worst case.
                kind = refusal.kind == .botDailyGameLimit ? .bot : .human
            case .botGameLimit:
                // Logged only from a refused POST answered 400 for the
                // bot-vs-bot daily limit, before outcome lines existed.
                outcome = .refused(LichessBotChallengeRefusal(kind: .botDailyGameLimit, httpStatus: 400, text: nil))
                kind = .bot
            }
            return LichessBotChallengeOutcomeRecord(
                id: stableID(for: "attempt:\(line.description)"), sentAt: row.firstAt, opponentID: opponentID.lowercased(),
                opponentKind: kind, challengeID: nil, creditCost: LichessBotChallengeOutcomeLog.creditCost(notCreated: outcome, kind: kind),
                outcome: outcome, resolvedAt: row.firstAt
            )
        case (.challenge, .notCreated), (.notCreatedAttempt, .challenge):
            return nil
        }
    }

    /// A UUID derived from `key`, so a refold gives a record the same id.
    private static func stableID(for key: String) -> UUID {
        let digest = Array(SHA256.hash(data: Data(key.utf8)))
        return UUID(uuid: (digest[0], digest[1], digest[2], digest[3], digest[4], digest[5], digest[6], digest[7],
                           digest[8], digest[9], digest[10], digest[11], digest[12], digest[13], digest[14], digest[15]))
    }
}
