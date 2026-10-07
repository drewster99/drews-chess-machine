import SwiftUI

/// The follow-lineage source on the Overview's model card (follow-lineage
/// plan §3.9): the lineage followed, its newest file, when it was last
/// checked, the outcome — orange while the source can't vouch for a newer
/// file and the last good generation keeps playing — and Check Now (OD-12).
/// Takes the controller's state as values, so every outcome can be drawn
/// without a running bot.
struct LichessBotLineageFollowStatusView: View {
    /// The lineage the applied settings follow.
    let followed: LichessBotFollowedLineage?
    /// The running bot's last check, of whatever lineage it checked.
    let status: LichessBotLineageFollowStatus?
    let playingGenerationID: Int?
    let isRunning: Bool
    let checkNow: @MainActor () async -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(followingText)
                .font(.system(.caption, design: .monospaced))
                .foregroundStyle(.secondary)
                .help(helpText)
            Text(outcomeText)
                .font(.caption)
                .foregroundStyle(isProblem ? Color.orange : Color.secondary)
                .textSelection(.enabled)
            HStack {
                Text(checkedText)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Button("Check Now") {
                    Task { @MainActor in
                        await checkNow()
                    }
                }
                .controlSize(.small)
                .disabled(!isRunning)
            }
        }
    }

    /// The status, when it is about the lineage the settings follow.
    private var currentStatus: LichessBotLineageFollowStatus? {
        guard let status, status.followed == followed else { return nil }
        return status
    }

    private var followingText: String {
        guard let followed else { return "No lineage chosen" }
        return "following run \(followed.lineageRunID.prefix(8)) from segment \(followed.anchorSegmentID.prefix(8))"
    }

    private var helpText: String {
        guard let followed else { return "" }
        return "run \(followed.lineageRunID)\nsegment \(followed.anchorSegmentID)"
    }

    private var outcomeText: String {
        guard let currentStatus else {
            return isRunning ? "Not checked yet" : "Checked while the bot is online"
        }
        let outcome = currentStatus.outcome
        if outcome.isProblem, let playingGenerationID {
            return "\(outcome.description) — still playing generation \(playingGenerationID)"
        }
        return outcome.description
    }

    private var checkedText: String {
        guard let currentStatus else { return "" }
        return "checked \(currentStatus.checkedAt.formatted(date: .omitted, time: .standard))"
    }

    private var isProblem: Bool {
        currentStatus?.outcome.isProblem ?? false
    }
}
