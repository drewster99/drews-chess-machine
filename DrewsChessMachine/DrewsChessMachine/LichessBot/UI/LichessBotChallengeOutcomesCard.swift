import SwiftUI

/// Outgoing challenges over the rolling day, by outcome, with declines
/// broken down by Lichess's reason key and refusals by kind, and the most
/// recent refusal's text.
struct LichessBotChallengeOutcomesCard: View {
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
                        .shown(controller.challengeOutcomeLog == nil)
                    Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 2) {
                        ForEach(Self.rows(controller.challengeOutcomeLog?.summary(now: context.date))) { row in
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
                    .shown(controller.challengeOutcomeLog != nil)
                    Text(Self.latestRefusalText(controller.challengeOutcomeLog))
                        .font(.callout)
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                        .lineLimit(3)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }
        }, label: {
            HStack {
                Text("Outgoing challenges, last 24 h")
                Spacer()
                Button("Challenge Log…") {
                    LichessBotChallengeLogWindowController.open(controller: controller)
                }
                .controlSize(.small)
                .help("Every challenge DCM sent or received, with how each ended")
            }
        })
    }

    private static func rows(_ summary: LichessBotChallengeOutcomeLog.Summary?) -> [Row] {
        guard let summary else { return [] }
        var rows: [Row] = []
        rows.append(Row(id: "accepted", label: "Accepted", count: summary.accepted, isDetail: false))
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
        rows.append(Row(id: "canceled", label: "Canceled or timed out", count: summary.canceled, isDetail: false))
        rows.append(Row(id: "offline", label: "Offline", count: summary.offline, isDetail: false))
        rows.append(Row(id: "refused", label: "POST refused", count: summary.refused, isDetail: false))
        for kind in LichessBotChallengeRefusal.Kind.allCases {
            rows.append(Row(id: "refused.\(kind.rawValue)", label: kind.label, count: summary.refusedByKind[kind, default: 0], isDetail: true))
        }
        rows.append(Row(id: "pending", label: "Waiting for an answer", count: summary.pending, isDetail: false))
        return rows
    }

    private static func latestRefusalText(_ log: LichessBotChallengeOutcomeLog?) -> String {
        guard let log else { return "" }
        let latest = log.records.reversed().lazy.compactMap { record -> (record: LichessBotChallengeOutcomeRecord, refusal: LichessBotChallengeRefusal)? in
            guard case .refused(let refusal) = record.outcome else { return nil }
            return (record, refusal)
        }.first
        guard let latest else { return "No refusals." }
        let time = latest.record.sentAt.formatted(date: .omitted, time: .standard)
        return "Latest refusal (\(time), \(latest.record.opponentID), HTTP \(latest.refusal.httpStatus)): \(latest.refusal.text ?? "no message")"
    }
}
