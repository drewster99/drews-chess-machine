import AppKit
import Foundation

/// Opens Lichess pages in the browser, and copies their addresses.
@MainActor
enum LichessBotLinks {
    static func openGame(_ gameID: String) {
        open("https://lichess.org/\(gameID)")
    }

    static func openUser(_ username: String) {
        open(userAddress(username))
    }

    /// Puts the user's profile address on the clipboard, replacing what was there.
    static func copyUser(_ username: String) {
        let address = userAddress(username)
        let pasteboard = NSPasteboard.general
        pasteboard.clearContents()
        guard pasteboard.setString(address, forType: .string) else {
            SessionLogger.shared.log("[LICHESS-BOT] could not copy \(address) to the clipboard")
            return
        }
        SessionLogger.shared.log("[BUTTON] Copy Lichess profile link \(address)")
    }

    /// A user's profile page: the one place its address is spelled out, so
    /// opening and copying always agree.
    static func userAddress(_ username: String) -> String {
        "https://lichess.org/@/\(username)"
    }

    private static func open(_ text: String) {
        guard let url = URL(string: text) else {
            SessionLogger.shared.log("[LICHESS-BOT] could not build a URL from \(text)")
            return
        }
        NSWorkspace.shared.open(url)
    }
}
