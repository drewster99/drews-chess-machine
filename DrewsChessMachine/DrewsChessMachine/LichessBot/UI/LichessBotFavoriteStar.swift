import SwiftUI

/// A toggleable favorite star for one Lichess player (plan §7.2), shown
/// beside usernames throughout the bot's UI. Favorites persist in
/// `player-notes.json`; until they load, the star is disabled.
struct LichessBotFavoriteStar: View {
    let controller: LichessBotController
    /// Lichess user id (any case; stored lowercased). Nil for a player
    /// without an account (Lichess's AI): the star is hidden.
    let userID: String?

    var body: some View {
        let isFavorite = userID.map { controller.playerNotes?.isFavorite($0) == true } ?? false
        Button {
            if let userID {
                controller.toggleFavorite(userID)
            }
        } label: {
            Image(systemName: isFavorite ? "star.fill" : "star")
                .foregroundStyle(isFavorite ? Color.yellow : Color.secondary)
        }
        .buttonStyle(.plain)
        .disabled(controller.playerNotes == nil)
        .shown(userID != nil)
        .help(isFavorite ? "Remove from favorites" : "Add to favorites")
        .accessibilityLabel(isFavorite ? "Favorite; remove" : "Not a favorite; add")
    }
}
