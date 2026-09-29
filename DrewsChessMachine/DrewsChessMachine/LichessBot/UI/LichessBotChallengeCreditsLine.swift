import SwiftUI

/// Challenge credits and outgoing-challenge outcomes over the rolling day,
/// under matchmaking's status on the Overview: credits spent (at most)
/// against Lichess's daily and per-minute budgets, then accepted, declined and
/// refused counts and the acceptance rate.
struct LichessBotChallengeCreditsLine: View {
    let controller: LichessBotController

    var body: some View {
        // The rolling windows shrink as records age out, with no state
        // change to redraw them, so they are re-read on a timer.
        TimelineView(.periodic(from: .now, by: 10)) { context in
            Text(Self.text(log: controller.challengeOutcomeLog, now: context.date))
                .font(.system(.callout, design: .monospaced))
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .help("Lichess allows \(LichessBotChallengeCredits.perDay) challenge credits per day and \(LichessBotChallengeCredits.perMinute) per minute: a challenge to a bot costs \(LichessBotChallengeCredits.cost(for: .bot)), to a human \(LichessBotChallengeCredits.cost(for: .human)), and nothing to a player who follows DCM, which no API reports, so the counts here are the most it can have been. A refusal for the bot daily game limit or another 400 is charged too; a 429 is not. Acceptance is accepted over answered (accepted, declined or canceled).")
        }
    }

    private static func text(log: LichessBotChallengeOutcomeLog?, now: Date) -> String {
        guard let log else { return "challenge outcomes not loaded" }
        let summary = log.summary(now: now)
        let rate = summary.acceptanceRate.map { String(format: "%3.0f%%", $0 * 100) } ?? "  –"
        return "credits \(pad(summary.creditsLastDay, 3))/\(LichessBotChallengeCredits.perDay) 24 h"
            + "  \(pad(summary.creditsLastMinute, 2))/\(LichessBotChallengeCredits.perMinute) min"
            + "  accepted \(pad(summary.accepted, 3))"
            + "  declined \(pad(summary.declined, 3))"
            + "  refused \(pad(summary.refused, 3))"
            + "  acceptance \(rate)"
    }

    private static func pad(_ value: Int, _ width: Int) -> String {
        String(format: "%\(width)d", value)
    }
}
