import SwiftUI

/// The one place a Challenge Log row's values become words and glyphs
/// (challenge-log plan §3.9). Every state, sender and source is said
/// explicitly, never left blank. Glyphs for directions and senders come from
/// `LichessBotGameOriginStyle`, so they mean the same as on games.
enum LichessBotChallengeLogStyle {

    // MARK: Direction

    static func directionSystemImage(_ direction: LichessBotChallengeLogDirection?) -> String {
        switch direction {
        case .outgoing?: return LichessBotGameOriginStyle.systemImage(for: .outgoingSenderNotRecorded)
        case .incoming?: return LichessBotGameOriginStyle.systemImage(for: .incoming)
        case nil: return LichessBotGameOriginStyle.systemImage(for: .unknown)
        }
    }

    static func directionText(_ direction: LichessBotChallengeLogDirection?) -> String {
        switch direction {
        case .outgoing?: return "Outgoing: DCM challenged the player"
        case .incoming?: return "Incoming: the player challenged DCM"
        case nil: return "Direction not recorded"
        }
    }

    // MARK: Opponent and terms

    static func opponentText(_ opponent: LichessBotChallengeLogRow.Opponent) -> String {
        switch opponent {
        case .player(let player): return player.name
        case .openChallenge: return "Open challenge"
        case .notRecorded: return "Not recorded"
        }
    }

    static func ratingText(_ opponent: LichessBotChallengeLogRow.Opponent) -> String {
        guard case .player(let player) = opponent, let rating = player.rating else { return "" }
        return String(format: "%4d", rating)
    }

    /// "3+2 · rated · random": the clock, rated or casual, and the color
    /// asked for.
    static func termsText(_ terms: LichessBotChallengeLogRow.Terms?) -> String {
        guard let terms else { return "Not recorded" }
        let clock: String
        if let limit = terms.limitSeconds, let increment = terms.incrementSeconds {
            clock = "\(clockMinutesText(limitSeconds: limit))+\(increment)"
        } else if let days = terms.daysPerTurn {
            clock = days == 1 ? "1 day per move" : "\(days) days per move"
        } else {
            clock = "unlimited"
        }
        return "\(clock) · \(terms.rated ? "rated" : "casual") · \(terms.color.raw)"
    }

    /// A clock's initial time in minutes, as Lichess writes it: whole
    /// minutes plain, the quarter-minute limits as fractions.
    static func clockMinutesText(limitSeconds: Int) -> String {
        switch limitSeconds {
        case 15: return "¼"
        case 30: return "½"
        case 45: return "¾"
        default:
            if limitSeconds % 60 == 0 { return "\(limitSeconds / 60)" }
            return String(format: "%g", Double(limitSeconds) / 60)
        }
    }

    // MARK: Sender / decision

    static func initiativeText(_ initiative: LichessBotChallengeLogRow.Initiative) -> String {
        switch initiative {
        case .sentBy(let sender):
            let label = LichessBotGameOriginStyle.shortLabel(for: LichessBotGameOriginCategory(sender: sender))
            guard case .matchmaking(let trigger, let fillMode) = sender else { return label }
            return "\(label) (\(triggerText(trigger)), \(fillModeText(fillMode)))"
        case .reconstructedSender(.attributed(let sender, let confidence)):
            let marker = confidence == .inferredFromAbsence ? "≈" : ""
            return reconstructedSenderText(sender) + marker
        case .reconstructedSender(.ambiguousCompanion):
            return "Sender not determined (ambiguous line)"
        case .reconstructedSender(.noSendLine):
            return "Sender not recorded (no send line)"
        case .senderNotRecorded:
            return "Sender not recorded"
        case .decided(.accept):
            return "DCM: accept"
        case .decided(.decline(let reason, _)):
            return "DCM: decline (\(reason.rawValue))"
        case .decided(.ignore):
            return "DCM: ignore"
        case .noDecisionRecorded:
            return "No decision recorded"
        case .directionNotRecorded:
            return "Direction not recorded"
        }
    }

    /// The rule behind DCM's decision, or how a rebuilt sender is known.
    static func initiativeHelp(_ initiative: LichessBotChallengeLogRow.Initiative) -> String {
        switch initiative {
        case .sentBy(let sender):
            return LichessBotGameOriginStyle.longLabel(for: LichessBotGameOriginCategory(sender: sender))
        case .reconstructedSender(.attributed(_, let confidence)):
            return "Reconstructed from the protocol log (\(LichessBotGameOriginStyle.confidenceText(confidence)))"
        case .reconstructedSender(.ambiguousCompanion):
            return "A sender's line could belong to this send or another send to the same player, so the sender is not claimed"
        case .reconstructedSender(.noSendLine):
            return "DCM's challenge was seen on the event stream with no 'challenge sent to' line: sent by another client using the token, or its line was lost"
        case .senderNotRecorded:
            return "DCM's challenge was seen only as its echo, and no send of the run that saw it explains it"
        case .decided(.accept):
            return "DCM accepted it"
        case .decided(.decline(_, let rule)):
            return "DCM declined it: \(rule)"
        case .decided(.ignore(let rule)):
            return "DCM ignored it: \(rule)"
        case .noDecisionRecorded:
            return "No decision by DCM was recorded"
        case .directionNotRecorded:
            return "No fact says which way the challenge went"
        }
    }

    static func reconstructedSenderText(_ sender: LichessBotReconstructedSender) -> String {
        switch sender {
        case .byOperator: return "Operator (sheet or resend)"
        case .challengeQueue: return LichessBotGameOriginStyle.shortLabel(for: .challengeQueue)
        case .matchmaking: return LichessBotGameOriginStyle.shortLabel(for: .matchmaking)
        case .matchmakingCasualResend: return LichessBotGameOriginStyle.shortLabel(for: .matchmakingCasualResend)
        }
    }

    static func triggerText(_ trigger: LichessBotMatchmakingTrigger) -> String {
        switch trigger {
        case .automaticPass: return "automatic pass"
        case .fillOpenSlots: return "Fill Open Slots"
        }
    }

    static func fillModeText(_ fillMode: LichessBotMatchmakingSettings.FillMode) -> String {
        switch fillMode {
        case .everyFreeSlot: return "every free slot"
        case .onlyWhenIdle: return "only when idle"
        }
    }

    // MARK: State

    static func stateText(_ state: LichessBotChallengeLogRow.State) -> String {
        switch state {
        case .challenge(let logState, let isPending):
            switch logState {
            case .open:
                return isPending ? "Waiting" : "No answer recorded"
            case .accepted(let gameStarted):
                return gameStarted ? "Accepted" : "Accepted (game start not seen)"
            case .declined(let reason):
                return "Declined (\(reason.keyText))"
            case .canceledByChallenger:
                return "Canceled by the challenger"
            case .withdrawn(let reason, let result):
                switch result {
                case .confirmed?: return "Withdrawn (\(withdrawalReasonText(reason)))"
                case nil: return "Withdrawn (\(withdrawalReasonText(reason))), result not recorded"
                case .alreadyGone?: return "No longer on Lichess"
                case .failed?: return "Withdrawal failed (\(withdrawalReasonText(reason)))"
                case .abandonedAtShutdown?: return "Withdrawn (\(withdrawalReasonText(reason))), abandoned at shutdown"
                }
            case .notCreated(.opponentOffline):
                return "Not sent: opponent offline"
            case .notCreated(.refused(let refusal)):
                return "Refused: \(refusal.kind.label)"
            case .notCreated(.noAnswer):
                return "No answer from Lichess"
            case .incomingDecided(.accept):
                return "Accepted by DCM, no game recorded"
            case .incomingDecided(.decline(let reason, _)):
                return "Declined by DCM (\(reason.rawValue))"
            case .incomingDecided(.ignore):
                return "Ignored by DCM"
            case .canceledOnLichessWithoutRecordedWithdrawal:
                return "Withdrawn (reason not recorded)"
            case .canceledOnLichessDirectionNotRecorded:
                return "Canceled on Lichess (direction not recorded)"
            }
        case .reconstructedNotCreated(.opponentOffline):
            return "Not sent: opponent offline"
        case .reconstructedNotCreated(.refused(let refusal)):
            return "Refused: \(refusal.kind.label)"
        case .reconstructedNotCreated(.botGameLimit(let gamesPlayed, _)):
            return "Refused: bot game limit (\(gamesPlayed))"
        }
    }

    /// What the state's short text leaves out: Lichess's messages, DCM's
    /// rules, and the notes and anomalies of the row's facts.
    static func stateHelp(_ row: LichessBotChallengeLogRow) -> String {
        var lines: [String] = []
        switch row.state {
        case .challenge(let logState, let isPending):
            switch logState {
            case .open:
                lines.append(isPending ? "DCM is waiting for an answer" : "No answer to this challenge was recorded")
            case .accepted(let gameStarted):
                lines.append(gameStarted ? "Accepted; its game started" : "Accepted; its game start was not seen")
            case .declined(let reason):
                lines.append("Declined on Lichess, reason key \(reason.keyText)")
            case .canceledByChallenger:
                lines.append("The challenger withdrew it")
            case .withdrawn(let reason, let result):
                lines.append("DCM withdrew it (\(withdrawalReasonText(reason))); Lichess: \(withdrawalResultText(result))")
            case .notCreated(.opponentOffline):
                lines.append("The player was offline when DCM checked, so nothing was posted")
            case .notCreated(.refused(let refusal)):
                lines.append("Lichess refused the POST (HTTP \(refusal.httpStatus)): \(refusal.text ?? "no message")")
            case .notCreated(.noAnswer(let error)):
                lines.append("Lichess' answer never arrived, so a challenge may exist: \(error)")
            case .incomingDecided(let decision):
                lines.append(initiativeHelp(.decided(decision)) + "; nothing after was recorded")
            case .canceledOnLichessWithoutRecordedWithdrawal:
                lines.append("Lichess reported DCM's challenge canceled with no withdrawal recorded: DCM or another client using the token withdrew it, but why was not recorded")
            case .canceledOnLichessDirectionNotRecorded:
                lines.append("Lichess reported it canceled, and no fact says which way it went")
            }
        case .reconstructedNotCreated(.opponentOffline):
            lines.append("The player was offline when DCM checked, so nothing was posted")
        case .reconstructedNotCreated(.refused(let refusal)):
            lines.append("Lichess refused the POST (HTTP \(refusal.httpStatus)): \(refusal.text ?? "no message")")
        case .reconstructedNotCreated(.botGameLimit(let gamesPlayed, let untilAsLogged)):
            lines.append("The player was at its bot-game limit (\(gamesPlayed)) until \(untilAsLogged), as logged; Lichess' answer was not logged")
        }
        lines.append(contentsOf: row.notes.map(noteText))
        lines.append(contentsOf: row.anomalies.map(anomalyText))
        return lines.joined(separator: "\n")
    }

    static func withdrawalReasonText(_ reason: LichessBotWithdrawalReason) -> String {
        switch reason {
        case .operatorCancel: return "by the operator"
        case .unansweredTimeout(let seconds): return "unanswered after \(seconds) s"
        case .goingOffline: return "going offline"
        case .wentOfflineWhileSending: return "went offline while sending"
        }
    }

    static func withdrawalResultText(_ result: LichessBotWithdrawalResult?) -> String {
        switch result {
        case .confirmed?: return "withdrawn"
        case .alreadyGone(let message)?: return "already gone (expired or answered)\(message.map { ": \($0)" } ?? "")"
        case .failed(let error)?: return "the withdrawal failed: \(error)"
        case .abandonedAtShutdown?: return "no answer before shutdown"
        case nil: return "result not recorded"
        }
    }

    static func noteText(_ note: LichessBotChallengeLedgerNote) -> String {
        switch note {
        case .withdrawalAttempted(let reason, let result):
            return "Note: a withdrawal was attempted (\(withdrawalReasonText(reason)); \(withdrawalResultText(result))), and the game started anyway"
        case .declinedOnLichess(let reason):
            return "Note: a decline (\(reason.keyText)) was reported, and the game started anyway"
        case .canceledOnLichess:
            return "Note: a cancel was reported, and the game started anyway"
        case .acceptedOutsideDCM(let decided):
            return "Note: accepted outside DCM (DCM decided: \(initiativeText(.decided(decided))))"
        }
    }

    static func anomalyText(_ anomaly: LichessBotChallengeLedgerAnomaly) -> String {
        switch anomaly {
        case .repeatedCreatedLines(let count): return "Anomaly: \(count) created lines for this challenge; the first is used"
        case .repeatedNotCreatedLines(let count): return "Anomaly: \(count) not-created lines for this attempt; the first is used"
        case .conflictingDirections: return "Anomaly: facts of both directions for this challenge"
        case .withdrawalResultWithoutRequest: return "Anomaly: a withdrawal result with no withdrawal request"
        }
    }

    // MARK: Credits and source

    static func creditsText(_ credits: LichessBotChallengeLogRow.Credits) -> String {
        switch credits {
        case .spent(let count): return String(format: "%3d", count)
        case .notApplicable, .notRecorded: return "–"
        }
    }

    static func creditsHelp(_ credits: LichessBotChallengeLogRow.Credits) -> String {
        switch credits {
        case .spent(let count): return "\(count) Lichess challenge credit(s), as counted when it was sent (the worst case)"
        case .notApplicable: return "Someone else's challenge costs DCM no credits"
        case .notRecorded: return "Not recorded"
        }
    }

    static func sourceText(_ source: LichessBotChallengeLogRow.Source) -> String {
        switch source {
        case .live: return "Live"
        case .reconstructed(let confidence?): return "Reconstructed (\(LichessBotGameOriginStyle.confidenceWord(confidence)))"
        case .reconstructed(nil): return "Reconstructed (sender not determined)"
        }
    }

    static func sourceHelp(_ source: LichessBotChallengeLogRow.Source) -> String {
        switch source {
        case .live: return "From the challenge log, recorded as it happened"
        case .reconstructed(let confidence?): return "Reconstructed from the protocol log (\(LichessBotGameOriginStyle.confidenceText(confidence)))"
        case .reconstructed(nil): return "Reconstructed from the protocol log; its sender is not determined"
        }
    }

    // MARK: Footer

    /// A load or rebuild that left something out, or failed.
    static let warningColor = Color.red

    /// The shown rows' tallies, of how many rows in all.
    static func countsText(_ counts: LichessBotChallengeLogCounts, totalRowCount: Int) -> String {
        var parts = [
            "\(counts.shown) of \(totalRowCount) shown",
            "outgoing \(counts.outgoing)",
            "incoming \(counts.incoming)",
        ]
        if counts.directionNotRecorded > 0 {
            parts.append("direction not recorded \(counts.directionNotRecorded)")
        }
        parts.append("live \(counts.live)")
        parts.append("reconstructed \(counts.reconstructed)")
        parts.append("credits recorded \(counts.creditsSpent)")
        return parts.joined(separator: " · ")
    }

    /// The challenge log's load: files, lines, newer lines skipped, and
    /// any file left out with why.
    static func ledgerStatusText(ledger: LichessBotChallengeLedger?, loadedFiles: LichessBotChallengeLogRecorder.LoadedFiles?) -> String {
        guard let ledger else { return "Challenge log: loading…" }
        let read = loadedFiles.map {
            "\($0.fileCount) file(s), \($0.lineCount) line(s), \($0.skippedNewerLines) newer line(s) skipped"
        }
        switch ledger.loadStatus {
        case .complete:
            return "Challenge log at load: \(read ?? "loaded")"
        case .partial(let filesLeftOut):
            let leftOut = filesLeftOut.map { "\($0.name) (\($0.reason))" }.joined(separator: "; ")
            return "Challenge log at load: \(read ?? "loaded"); \(filesLeftOut.count) file(s) left out: \(leftOut)"
        case .failed(let reason):
            return "Challenge log: the read failed (\(reason)); only this run's challenges are shown"
        }
    }

    /// Whether the ledger's load left something out (drawn as a warning).
    static func ledgerStatusIsWarning(_ ledger: LichessBotChallengeLedger?) -> Bool {
        switch ledger?.loadStatus {
        case .complete?, nil: return false
        case .partial?, .failed?: return true
        }
    }

    /// The history rebuilt from the protocol log: where its rebuild stands,
    /// and what it holds.
    static func historyStatusText(_ status: LichessBotController.ChallengeHistoryStatus, history: LichessBotChallengeReconstruction?) -> String {
        let held = history.map { history in
            let unexplained = history.unexplainedLines.count
            return "\(history.rows.count) row(s) from \(history.inputs.count) protocol file(s)"
                + (unexplained > 0 ? ", \(unexplained) line(s) left unexplained" : "")
                + (history.liveLogFirstEntryAt.map { ", up to the live log's start \($0.formatted(date: .abbreviated, time: .standard))" } ?? ", no live log yet")
        }
        switch status {
        case .notBuilt:
            return "Reconstructed history: not built yet (it is built once the challenge log loads)"
        case .noAccount:
            return "Reconstructed history: not built — no Lichess account is configured, so no direction can be told"
        case .rebuilding:
            return "Reconstructed history: rebuilding…" + (held.map { " (showing \($0))" } ?? "")
        case .ready(let outcome, let at):
            let what = outcome == .written ? "rebuilt" : "up to date"
            return "Reconstructed history: \(held ?? "empty"); \(what) at \(at.formatted(date: .omitted, time: .standard))"
        case .failed(let reason):
            return "Reconstructed history: the rebuild failed (\(reason))" + (held.map { "; showing \($0)" } ?? "")
        }
    }

    static func historyStatusIsWarning(_ status: LichessBotController.ChallengeHistoryStatus) -> Bool {
        switch status {
        case .failed, .noAccount: return true
        case .notBuilt, .rebuilding, .ready: return false
        }
    }

    // MARK: Filter labels

    static func label(_ range: LichessBotChallengeLogFilter.DateRange) -> String {
        switch range {
        case .last24Hours: return "24 h"
        case .last7Days: return "7 days"
        case .last30Days: return "30 days"
        case .all: return "All"
        }
    }

    static func label(_ direction: LichessBotChallengeLogFilter.DirectionChoice) -> String {
        switch direction {
        case .all: return "All"
        case .outgoing: return "Outgoing"
        case .incoming: return "Incoming"
        }
    }

    static func label(_ kind: LichessBotChallengeLogStateKind) -> String {
        switch kind {
        case .waiting: return "Waiting"
        case .noAnswerRecorded: return "No answer recorded"
        case .accepted: return "Accepted"
        case .declined: return "Declined"
        case .withdrawn: return "Withdrawn"
        case .canceled: return "Canceled"
        case .notCreated: return "Not created"
        case .decidedByDCM: return "Decided by DCM"
        }
    }

    static func label(_ kind: LichessBotChallengeLogSenderKind) -> String {
        switch kind {
        case .challengeSheet: return LichessBotGameOriginStyle.shortLabel(for: .challengeSheet)
        case .casualResendOffer: return LichessBotGameOriginStyle.shortLabel(for: .casualResendOffer)
        case .challengeQueue: return LichessBotGameOriginStyle.shortLabel(for: .challengeQueue)
        case .matchmaking: return LichessBotGameOriginStyle.shortLabel(for: .matchmaking)
        case .matchmakingCasualResend: return LichessBotGameOriginStyle.shortLabel(for: .matchmakingCasualResend)
        case .byOperator: return reconstructedSenderText(.byOperator)
        case .incoming: return LichessBotGameOriginStyle.shortLabel(for: .incoming)
        case .notRecorded: return "Not recorded"
        }
    }
}
