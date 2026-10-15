import SwiftUI

/// Challenges other players sent DCM over the rolling day, by outcome, with
/// declines broken down by reason and failed answers by which answer
/// failed, and the most recent failure's text. The incoming counterpart of
/// `LichessBotChallengeOutcomesCard`, laid out the same way so the two read
/// side by side.
struct LichessBotIncomingChallengeOutcomesCard: View {
    let controller: LichessBotController

    private struct Row: Identifiable {
        let id: String
        let label: String
        let count: Int
        /// A breakdown row under its outcome.
        let isDetail: Bool
    }

    var body: some View {
        GroupBox(content: {
            TimelineView(.periodic(from: .now, by: 30)) { context in
                VStack(alignment: .leading, spacing: 8) {
                    Text("Challenge outcomes haven't loaded.")
                        .foregroundStyle(.secondary)
                        .shown(controller.incomingChallengeOutcomeLog == nil)
                    Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 2) {
                        ForEach(Self.rows(controller.incomingChallengeOutcomeLog?.summary(now: context.date))) { row in
                            GridRow {
                                Text(row.label)
                                    .padding(.leading, row.isDetail ? 16 : 0)
                                    .foregroundStyle(row.isDetail ? Color.secondary : Color.primary)
                                Text(String(format: "%4d", row.count))
                                    .font(.system(.callout, design: .monospaced))
                                    .foregroundStyle(row.isDetail && row.count == 0 ? Color.secondary : Color.primary)
                                    .gridColumnAlignment(.trailing)
                            }
                            .font(.callout)
                        }
                    }
                    .shown(controller.incomingChallengeOutcomeLog != nil)
                    Text(Self.latestFailureText(controller.incomingChallengeOutcomeLog))
                        .font(.callout)
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                        .lineLimit(3)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }
        }, label: {
            Text("Incoming challenges, last 24 h")
                .help("Challenges other players sent DCM over the last 24 hours. Declined: Lichess's reason when it reported the decline, else the reason DCM declined with. Not answered: no decision by DCM recorded; still waiting, or expired without Lichess reporting it. Answer failed: DCM's accept or decline was sent and failed.")
        })
    }

    private static func rows(_ summary: LichessBotIncomingChallengeOutcomeLog.Summary?) -> [Row] {
        guard let summary else { return [] }
        var rows: [Row] = []
        rows.append(Row(id: "received", label: "Received", count: summary.received, isDetail: false))
        rows.append(Row(id: "accepted", label: "Accepted", count: summary.accepted, isDetail: false))
        rows.append(Row(id: "acceptedNoGame", label: "Accepted, no game started", count: summary.acceptedNoGameStarted, isDetail: false))
        rows.append(Row(id: "declined", label: "Declined", count: summary.declined, isDetail: false))
        for reason in LichessBotDeclineReason.allCases {
            rows.append(Row(id: "declined.\(reason.rawValue)", label: reason.rawValue, count: summary.declinedByReason[.known(reason), default: 0], isDetail: true))
        }
        let others = summary.declinedByReason.filter {
            if case .known = $0.key { return false }
            return true
        }
        for (reason, count) in others.sorted(by: { $0.key.keyText < $1.key.keyText }) {
            rows.append(Row(id: "declined.other.\(reason.keyText)", label: reason.keyText, count: count, isDetail: true))
        }
        rows.append(Row(id: "canceled", label: "Canceled by the challenger", count: summary.canceledByChallenger, isDetail: false))
        rows.append(Row(id: "ignored", label: "Ignored by DCM", count: summary.ignored, isDetail: false))
        rows.append(Row(id: "failed", label: "Answer failed", count: summary.answerFailed, isDetail: false))
        for answer in LichessBotIncomingAnswerFailure.Answer.allCases {
            rows.append(Row(id: "failed.\(answer.rawValue)", label: answer.label, count: summary.answerFailedByAnswer[answer, default: 0], isDetail: true))
        }
        rows.append(Row(id: "notAnswered", label: "Not answered", count: summary.notAnswered, isDetail: false))
        return rows
    }

    private static func latestFailureText(_ log: LichessBotIncomingChallengeOutcomeLog?) -> String {
        guard let log else { return "" }
        guard let latest = log.latestAnswerFailure else { return "No failed answers." }
        let time = latest.failure.at.formatted(date: .omitted, time: .standard)
        let challenger = latest.record.challengerID ?? "challenger not recorded"
        return "Latest failed answer (\(time), \(challenger), \(latest.failure.answer.label)): \(latest.failure.error)"
    }
}
