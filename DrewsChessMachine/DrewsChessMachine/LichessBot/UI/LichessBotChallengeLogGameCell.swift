import SwiftUI

/// A Challenge Log row's game: a link to it on lichess.org once the
/// challenge was accepted (the game has the challenge's id), else a dash.
struct LichessBotChallengeLogGameCell: View {
    let gameID: String?

    var body: some View {
        ZStack(alignment: .leading) {
            Text("–")
                .foregroundStyle(.secondary)
                .help("No game: the challenge was not accepted")
                .shown(gameID == nil)
            Button(gameID ?? "") {
                if let gameID {
                    LichessBotLinks.openGame(gameID)
                }
            }
            .buttonStyle(.link)
            .help("Open on lichess.org")
            .shown(gameID != nil)
        }
    }
}
