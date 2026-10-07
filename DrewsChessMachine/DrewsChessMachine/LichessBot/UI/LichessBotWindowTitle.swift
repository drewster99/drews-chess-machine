import Foundation

/// The Lichess Bot window's title. The root view's `navigationTitle`
/// replaces the window's own title (`sceneBridgingOptions` `.title`), so the
/// window's name has to lead that title too, or the window list, the Window
/// menu and the Dock lose which window this is.
enum LichessBotWindowTitle {
    static let name = "Lichess Bot"

    /// "Lichess Bot — DrewsChessMachine — Online — 2 games in progress":
    /// which window, which account, what it is doing, and how busy it is.
    static func text(account: String, connection: LichessBotController.ConnectionState, gamesInProgress: Int) -> String {
        "\(name) — \(account) — \(connection.label) — \(gamesInProgress) game\(gamesInProgress == 1 ? "" : "s") in progress"
    }
}
