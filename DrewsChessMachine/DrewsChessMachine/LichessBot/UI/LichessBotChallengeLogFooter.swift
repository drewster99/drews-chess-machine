import SwiftUI

/// The Challenge Log window's footer (challenge-log plan §3.9): the shown
/// rows' counts, how the challenge log loaded, and where the history rebuilt
/// from the protocol log stands, with its Rebuild button.
struct LichessBotChallengeLogFooter: View {
    let controller: LichessBotController
    let counts: LichessBotChallengeLogCounts
    let totalRowCount: Int

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(LichessBotChallengeLogStyle.countsText(counts, totalRowCount: totalRowCount))
                .font(.system(.callout, design: .monospaced))
            Text(LichessBotChallengeLogStyle.ledgerStatusText(ledger: controller.challengeLedger, loadedFiles: controller.challengeLogRecorder.loadedFiles))
                .foregroundStyle(LichessBotChallengeLogStyle.ledgerStatusIsWarning(controller.challengeLedger) ? LichessBotChallengeLogStyle.warningColor : Color.secondary)
                .textSelection(.enabled)
            HStack(spacing: 8) {
                Text(LichessBotChallengeLogStyle.historyStatusText(controller.challengeHistoryStatus, history: controller.challengeHistory))
                    .foregroundStyle(LichessBotChallengeLogStyle.historyStatusIsWarning(controller.challengeHistoryStatus) ? LichessBotChallengeLogStyle.warningColor : Color.secondary)
                    .textSelection(.enabled)
                Spacer()
                Button("Rebuild") {
                    SessionLogger.shared.log("[BUTTON] Rebuild Lichess bot challenge history")
                    Task { @MainActor in
                        await controller.rebuildChallengeHistory()
                    }
                }
                .disabled(controller.challengeLedger == nil || controller.challengeHistoryStatus == .rebuilding)
                .help("Rebuild the history of challenges from before the challenge log began, from the protocol log. It is written only when it changes.")
            }
        }
        .font(.callout)
        .lineLimit(2)
    }
}
