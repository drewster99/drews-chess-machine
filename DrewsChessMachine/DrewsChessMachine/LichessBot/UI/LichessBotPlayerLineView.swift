import SwiftUI

/// A player's name, title and rating, with a clock. For the opponent it
/// also shows DCM's head-to-head record against them (plan §14.3a).
struct LichessBotPlayerLineView: View {
    let player: LichessBotLiveGame.Player?
    let isOurs: Bool
    let headToHead: (wins: Int, draws: Int, losses: Int)?
    let clockMilliseconds: Int?
    let clockReceivedAt: Date?
    let clockRunning: Bool

    var body: some View {
        HStack(spacing: 8) {
            Circle()
                .fill(isOurs ? Color.accentColor : Color.secondary)
                .frame(width: 8, height: 8)
            Text(player?.title ?? "")
                .font(.callout.weight(.semibold))
                .foregroundStyle(.orange)
                .shown(player?.title != nil)
            Text(player?.name ?? "—")
                .font(.body.weight(.medium))
                .lineLimit(1)
            Text(player?.rating.map { String(format: "%4d", $0) } ?? "")
                .font(.system(.callout, design: .monospaced))
                .foregroundStyle(.secondary)
            Text(headToHeadText)
                .font(.system(.caption, design: .monospaced))
                .foregroundStyle(.secondary)
                .help("DCM's record against this opponent: wins–draws–losses")
                .shown(headToHead != nil)
            Spacer(minLength: 8)
            LichessBotClockView(milliseconds: clockMilliseconds, receivedAt: clockReceivedAt, isRunning: clockRunning, isOurs: isOurs)
        }
    }

    private var headToHeadText: String {
        guard let headToHead else { return "" }
        return "vs: \(headToHead.wins)–\(headToHead.draws)–\(headToHead.losses)"
    }
}
