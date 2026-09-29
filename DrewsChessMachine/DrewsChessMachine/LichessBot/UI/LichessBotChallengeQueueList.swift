import SwiftUI

/// The challenge queue on the Overview (plan §7.3 A): each entry's
/// opponent, clock, and why it waits or was skipped, with per-entry Cancel
/// and Clear Queue. Hidden while the queue is empty.
struct LichessBotChallengeQueueList: View {
    let controller: LichessBotController

    var body: some View {
        let entries = controller.challengeQueue.entries
        let waitReason = controller.challengeQueueWaitReason
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 8) {
                Text("Challenge queue")
                    .font(.callout.weight(.semibold))
                Text("\(entries.count)")
                    .font(.system(.callout, design: .monospaced))
                    .foregroundStyle(.secondary)
                Spacer()
                Button("Clear Queue") {
                    controller.clearChallengeQueue()
                }
                .help("Remove every entry; challenges already sent stay pending")
            }
            Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 2) {
                ForEach(entries) { entry in
                    GridRow {
                        LichessBotFavoriteStar(controller: controller, userID: entry.userID)
                        Text(entry.username)
                            .lineLimit(1)
                        Text(entry.request.clockText)
                            .font(.system(.callout, design: .monospaced))
                            .gridColumnAlignment(.trailing)
                        Text(entry.request.rated ? "rated" : "casual")
                            .foregroundStyle(.secondary)
                        Text(Self.statusText(entry.status, waitReason: waitReason))
                            .foregroundStyle(Self.statusColor(entry.status))
                            .lineLimit(1)
                        Button("Cancel") {
                            controller.cancelQueuedChallenge(entry.id)
                        }
                        .help(entry.status == .sending ? "Remove it; the challenge being sent still goes out and shows as pending" : "Remove it from the queue")
                    }
                    .font(.callout)
                }
            }
        }
        .shown(!entries.isEmpty)
    }

    private static func statusText(_ status: LichessBotChallengeQueue.Status, waitReason: String?) -> String {
        switch status {
        case .waiting:
            return waitReason ?? "waiting"
        case .sending:
            return "sending…"
        case .skipped(let reason):
            return "skipped: \(reason)"
        }
    }

    private static func statusColor(_ status: LichessBotChallengeQueue.Status) -> Color {
        switch status {
        case .waiting, .sending:
            return .secondary
        case .skipped:
            return .orange
        }
    }
}

/// Matchmaking's state on the Overview (plan §7.3 B): on or off, what the
/// latest pass did, and challenges sent in the last hour against the cap.
struct LichessBotMatchmakingStatusLine: View {
    let controller: LichessBotController

    var body: some View {
        // The hourly count falls as sends age out, with no state change to
        // redraw it, so it is re-read on a timer.
        TimelineView(.periodic(from: .now, by: 30)) { context in
            HStack(spacing: 8) {
                Text(controller.settings.matchmaking.enabled ? "Matchmaking on" : "Matchmaking off")
                    .font(.callout.weight(.semibold))
                    .foregroundStyle(controller.settings.matchmaking.enabled ? Color.primary : Color.secondary)
                Text(hourText(now: context.date))
                    .font(.system(.callout, design: .monospaced))
                    .foregroundStyle(.secondary)
                    .help("Matchmaking challenges that reached Lichess in the last hour, against the per-hour cap in Settings ▸ Matchmaking")
                Text(controller.isFillingOpenSlots ? "filling open slots…" : (controller.matchmakingStatus ?? ""))
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
                    .textSelection(.enabled)
            }
        }
    }

    private func hourText(now: Date) -> String {
        let sent = controller.matchmakingRateLimiter.challengesInLastHour(now: now)
        return "\(sent)/\(controller.settings.matchmaking.maxChallengesPerHour) this hour"
    }
}
