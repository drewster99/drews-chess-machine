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
        let isFavorite: Bool = userID.map { controller.playerNotes?.isFavorite($0) == true } ?? false
        Button(
            action: {
                if let userID {
                    controller.toggleFavorite(userID)
                }
            },
            label: {
                LichessBotFavoriteStarSymbol(isFavorite: isFavorite)
            }
        )
        .buttonStyle(.plain)
        .disabled(controller.playerNotes == nil)
        .shown(userID != nil)
        .help(LichessBotFavoriteStarSymbol.helpText(isFavorite: isFavorite))
        .accessibilityLabel(LichessBotFavoriteStarSymbol.accessibilityText(isFavorite: isFavorite))
    }
}

/// The star itself: filled and yellow for a favorite.
struct LichessBotFavoriteStarSymbol: View {
    let isFavorite: Bool

    var body: some View {
        let symbolName: String = isFavorite ? "star.fill" : "star"
        let color: Color = isFavorite ? Color.yellow : Color.secondary
        Image(systemName: symbolName)
            .foregroundStyle(color)
    }

    static func helpText(isFavorite: Bool) -> String {
        isFavorite ? "Remove from favorites" : "Add to favorites"
    }

    static func accessibilityText(isFavorite: Bool) -> String {
        isFavorite ? "Favorite; remove" : "Not a favorite; add"
    }
}
