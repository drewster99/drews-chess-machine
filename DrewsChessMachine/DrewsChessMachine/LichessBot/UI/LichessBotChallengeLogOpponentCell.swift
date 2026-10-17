import SwiftUI

/// A Challenge Log row's opponent: the favorite star, the title and the
/// name, or why no player is named.
struct LichessBotChallengeLogOpponentCell: View {
    let controller: LichessBotController
    let opponent: LichessBotChallengeLogRow.Opponent

    var body: some View {
        HStack(spacing: 4) {
            LichessBotFavoriteStar(controller: controller, userID: player?.id)
            Text(player?.title ?? "")
                .foregroundStyle(.orange)
                .shown(player?.title != nil)
            Button {
                LichessBotOpponentWindowLauncher.open(opponentID: player?.id, name: player?.name, controller: controller)
            } label: {
                Text(LichessBotChallengeLogStyle.opponentText(opponent))
                    .foregroundStyle(player == nil ? Color.secondary : Color.primary)
                    .lineLimit(1)
            }
            .buttonStyle(.link)
            .disabled(player == nil)
            .help("Open this opponent's breakdown")
        }
    }

    private var player: LichessBotChallengeLogRow.Player? {
        guard case .player(let player) = opponent else { return nil }
        return player
    }
}
