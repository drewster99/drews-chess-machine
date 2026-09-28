import AppKit
import SwiftUI

/// A standalone window for one Lichess game (plan §14.3a). Several can be
/// open; each keeps its game until closed, even after the game leaves the
/// live grid.
@MainActor
final class LichessBotGameWindowController: NSWindowController, NSWindowDelegate {
    let gameID: String

    init(game: LichessBotLiveGame, controller: LichessBotController) {
        gameID = game.id
        let view = LichessBotGameWindowView(game: game, controller: controller)
        let hosting = NSHostingController(rootView: view)
        let window = NSWindow(contentViewController: hosting)
        window.setContentSize(NSSize(width: 980, height: 720))
        window.minSize = NSSize(width: 720, height: 560)
        window.title = "Lichess game \(game.id)"
        window.isReleasedWhenClosed = false
        window.center()
        super.init(window: window)
        window.delegate = self
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) {
        fatalError("init(coder:) is not supported")
    }

    func windowWillClose(_ notification: Notification) {
        LichessBotGameWindowLauncher.unregister(self)
    }
}

/// The pop-out window's content.
struct LichessBotGameWindowView: View {
    let game: LichessBotLiveGame
    let controller: LichessBotController

    var body: some View {
        LichessBotGameDetailView(
            controller: controller,
            game: game,
            headToHead: controller.headToHead(against: game.opponent?.id),
            onPopOut: nil,
            claimsKeyboardShortcuts: true
        )
        .frame(minWidth: 720, minHeight: 560)
    }
}

/// Opens (or focuses) the window for a game.
@MainActor
enum LichessBotGameWindowLauncher {
    private static var controllers: [String: LichessBotGameWindowController] = [:]

    static func open(game: LichessBotLiveGame, controller: LichessBotController) {
        SessionLogger.shared.log("[BUTTON] Pop out Lichess game \(game.id)")
        if let existing = controllers[game.id] {
            existing.showWindow(nil)
            existing.window?.makeKeyAndOrderFront(nil)
            return
        }
        let windowController = LichessBotGameWindowController(game: game, controller: controller)
        controllers[game.id] = windowController
        windowController.showWindow(nil)
        windowController.window?.makeKeyAndOrderFront(nil)
    }

    static func unregister(_ windowController: LichessBotGameWindowController) {
        if controllers[windowController.gameID] === windowController {
            controllers[windowController.gameID] = nil
        }
    }
}
