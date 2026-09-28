import AppKit
import Foundation

/// Opens Lichess pages in the browser.
@MainActor
enum LichessBotLinks {
    static func openGame(_ gameID: String) {
        open("https://lichess.org/\(gameID)")
    }

    static func openUser(_ username: String) {
        open("https://lichess.org/@/\(username)")
    }

    private static func open(_ text: String) {
        guard let url = URL(string: text) else {
            SessionLogger.shared.log("[LICHESS-BOT] could not build a URL from \(text)")
            return
        }
        NSWorkspace.shared.open(url)
    }
}
