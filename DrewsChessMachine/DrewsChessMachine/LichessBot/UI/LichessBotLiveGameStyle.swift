import SwiftUI

/// The live games' colors and glyphs, in one place, so a game against a
/// human looks the same in the grid and the single view (owner request
/// 2026-10-13: "highlight games vs humans"). Every color is a system
/// semantic color, so light and dark mode both follow the system.
enum LichessBotLiveGameStyle {

    /// A game against a human: the marker glyph and a live tile's border
    /// and tint. Purple carries no other meaning in the live views (green
    /// is a win or a live game, red a loss, orange a title), so it reads as
    /// "who" rather than "how it is going".
    static let humanOpponent = Color.purple
    static let humanOpponentSystemImage = "person.fill"
    static let humanOpponentLabel = "Human opponent"
    static let humanOpponentHelp = "DCM is playing a human, not a bot"

    /// A live tile's border: against a bot (or an opponent not known yet),
    /// or against a human.
    static let liveBorder = Color.green.opacity(0.6)
    static let liveHumanBorder = humanOpponent.opacity(0.85)
    /// A live tile's background against a human; a live bot game's tile has
    /// none. Faint, so the board stays easy to read.
    static let liveHumanBackground = humanOpponent.opacity(0.10)
}
