import SwiftUI

/// A game's anomaly count for the live grid's tile: the same anomalies the
/// game window's header counts, so a tile never shows a game as fine while
/// its window lists problems. Empty (and taking no room) without any.
struct LichessBotGameAnomalyBadge: View {
    let anomalies: [LichessBotLiveGame.Anomaly]

    var body: some View {
        Label("\(anomalies.count)", systemImage: "exclamationmark.triangle.fill")
            .font(.system(.caption, design: .monospaced).weight(.semibold))
            .foregroundStyle(.red)
            .help(anomalies.map(\.text).joined(separator: "\n"))
            .accessibilityLabel("\(anomalies.count) anomalies")
            .shown(!anomalies.isEmpty)
    }
}
