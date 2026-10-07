import SwiftUI

/// The model card's switch and refresh status (follow-lineage plan §3.9,
/// §3.10). While the applied settings name another source than the playing
/// generation's, the bot is building that source's generation in the poll
/// loop and new games keep starting on the old one; this says so. The last
/// failed refresh or switch, and when it is retried, show until one
/// succeeds.
struct LichessBotModelSwitchStatusView: View {
    let controller: LichessBotController

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(switchingText)
                .font(.callout)
                .foregroundStyle(.orange)
                .shown(isSwitching)
            Text(failureText)
                .font(.caption)
                .foregroundStyle(.orange)
                .textSelection(.enabled)
                .shown(hasFailure)
        }
    }

    private var hasFailure: Bool {
        controller.modelRefreshFailure != nil
    }

    private var isSwitching: Bool {
        guard let playing = controller.generation else { return false }
        return playing.generationSource != controller.settings.model.generationSource
    }

    private var switchingText: String {
        guard let playing = controller.generation else { return "" }
        return "Switching to \(controller.settings.model.source.displayName) — still playing \(playing.sourceKind.displayName) \(playing.modelID) (generation \(playing.generationID))"
    }

    private var failureText: String {
        guard let failure = controller.modelRefreshFailure else { return "" }
        return "Last attempt failed: \(failure.text) · retry at \(failure.retryAt.formatted(date: .omitted, time: .standard))"
    }
}
