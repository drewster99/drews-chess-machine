import SwiftUI

/// The Lichess Bot window's sidebar footer: which app version and build the
/// bot is running. A bot runs for days, so the build it was launched from can
/// be older than the one just built; this answers "which build is playing?"
/// without opening the session log.
struct LichessBotBuildInfoFooter: View {
    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text("Version \(AppVersionText.marketingVersionWithBundleVersion)")
            Text(AppVersionText.buildLine)
        }
        .font(LichessBotStatsStyle.noteFont.monospacedDigit())
        .foregroundStyle(LichessBotStatsStyle.neutral)
        .textSelection(.enabled)
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
    }
}
