import SwiftUI

/// The marker for a game against a human (`LichessBotLiveGameStyle`): a
/// glyph alone, with its meaning as help and accessibility label, so it fits
/// a tile's crowded header without making the opponent's name wrap. Kept in
/// the hierarchy and collapsed when the opponent isn't a human (or isn't
/// known yet), like the anomaly badge.
struct LichessBotHumanOpponentGlyph: View {
    let isHuman: Bool

    var body: some View {
        Image(systemName: LichessBotLiveGameStyle.humanOpponentSystemImage)
            .foregroundStyle(LichessBotLiveGameStyle.humanOpponent)
            .help(LichessBotLiveGameStyle.humanOpponentHelp)
            .accessibilityLabel(LichessBotLiveGameStyle.humanOpponentLabel)
            .shown(isHuman)
    }
}
