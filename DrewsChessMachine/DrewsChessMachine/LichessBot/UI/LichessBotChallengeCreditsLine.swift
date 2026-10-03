import SwiftUI

/// Challenge credits and outgoing-challenge outcomes over the rolling day,
/// under matchmaking's status on the Overview, as two labeled lines:
///
/// - **Challenge credits used** — Lichess's challenge budget: at most
///   `perDay` credits per rolling day and `perMinute` per minute, with the
///   per-challenge cost from `LichessBotChallengeCredits.cost(for:)`. Shown as
///   "N of M" with the window named, so the two budgets cannot be confused.
/// - **Challenges, last 24 h** — accepted, declined and refused counts, each
///   number directly before its own word, then the acceptance rate.
///
/// Every count is padded with figure spaces to the width of its budget, so
/// the words after a number hold their place as the number grows. Words
/// within one item are joined by no-break spaces, so a narrow Overview wraps
/// a line only between items, never inside one.
struct LichessBotChallengeCreditsLine: View {
    let controller: LichessBotController

    var body: some View {
        // The rolling windows shrink as records age out, with no state
        // change to redraw them, so they are re-read on a timer.
        TimelineView(.periodic(from: .now, by: 10)) { context in
            let lines = Self.lines(log: controller.challengeOutcomeLog, now: context.date)
            VStack(alignment: .leading, spacing: 2) {
                Text(lines.credits)
                    .help("Lichess's challenge budget: \(LichessBotChallengeCredits.perDay) credits per rolling 24 hours and \(LichessBotChallengeCredits.perMinute) per minute. A challenge to a bot costs \(LichessBotChallengeCredits.cost(for: .bot)) credit, to a human \(LichessBotChallengeCredits.cost(for: .human)), and nothing to a player who follows DCM, which no API reports, so these are the most DCM can have used. A refusal for the bot daily game limit or another 400 is charged too; a 429 is not.")
                Text(lines.outcomes)
                    .help("Outgoing challenges over the last 24 hours. Accepted / declined: the opponent's answer. Refused: Lichess rejected the challenge request itself (for example a daily limit). Acceptance: accepted over answered (accepted, declined or canceled).")
            }
            .font(.callout)
            .monospacedDigit()
            .foregroundStyle(.secondary)
            .fixedSize(horizontal: false, vertical: true)
        }
    }

    /// U+2007 FIGURE SPACE: as wide as a digit under `.monospacedDigit()`,
    /// and a no-break space.
    static let figureSpace = "\u{2007}"
    /// U+00A0 NO-BREAK SPACE, between the words of one item.
    static let noBreakSpace = "\u{00A0}"

    /// `value` right-aligned with figure spaces to the digit count of
    /// `widest`; a value wider than `widest` is shown in full.
    static func padded(_ value: Int, toWidthOf widest: Int) -> String {
        let text = String(value)
        return String(repeating: figureSpace, count: max(0, String(widest).count - text.count)) + text
    }

    /// `words` joined with no-break spaces.
    private static func item(_ words: String...) -> String {
        words.joined(separator: noBreakSpace)
    }

    static func lines(log: LichessBotChallengeOutcomeLog?, now: Date) -> (credits: String, outcomes: String) {
        guard let log else { return ("Challenge outcomes not loaded", "") }
        let summary = log.summary(now: now)
        let perDay = LichessBotChallengeCredits.perDay
        let perMinute = LichessBotChallengeCredits.perMinute
        // Answered challenges are bounded by those charged in the day, so
        // the day's budget is the widest any count can be.
        let rate: String
        if let acceptanceRate = summary.acceptanceRate {
            rate = padded(Int((acceptanceRate * 100).rounded()), toWidthOf: 100) + "%"
        } else {
            rate = String(repeating: figureSpace, count: String(100).count - 1) + "–" + figureSpace
        }
        let separator = "  ·  "
        let credits = "Challenge credits used: "
            + item(padded(summary.creditsLastDay, toWidthOf: perDay), "of", String(perDay), "in", "24", "h")
            + separator
            + item(padded(summary.creditsLastMinute, toWidthOf: perMinute), "of", String(perMinute), "in", "the", "last", "minute")
        let outcomes = "Challenges, last 24 h: "
            + item(padded(summary.accepted, toWidthOf: perDay), "accepted")
            + separator + item(padded(summary.declined, toWidthOf: perDay), "declined")
            + separator + item(padded(summary.refused, toWidthOf: perDay), "refused")
            + separator + item(rate, "acceptance")
        return (credits, outcomes)
    }
}
