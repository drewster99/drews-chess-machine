import AppKit
import SwiftUI

/// A standalone window with one opponent's full breakdown (plan §14.3c):
/// their public profile, DCM's head-to-head, streaks and notables, the record
/// split by time control and by model, trend charts, and every game DCM has
/// played them. One window per opponent; reopening the same opponent brings
/// its window forward. Opened by clicking an opponent in the Opponents pane,
/// the recent-games list, or the complete-games window.
@MainActor
final class LichessBotOpponentDetailWindowController: NSWindowController, NSWindowDelegate {
    let opponentID: String

    init(opponentID: String, name: String, controller: LichessBotController) {
        self.opponentID = opponentID.lowercased()
        let view = LichessBotOpponentDetailView(controller: controller, opponentID: self.opponentID)
        let hosting = NSHostingController(rootView: view)
        let window = NSWindow(contentViewController: hosting)
        window.setContentSize(NSSize(width: 1040, height: 760))
        window.minSize = NSSize(width: 760, height: 520)
        window.title = "Opponent: \(name)"
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
        LichessBotOpponentWindowLauncher.unregister(self)
    }
}

/// Opens (or focuses) the detail window for one opponent.
@MainActor
enum LichessBotOpponentWindowLauncher {
    private static var controllers: [String: LichessBotOpponentDetailWindowController] = [:]

    /// A game with no opponent id (an aborted challenge that never named one)
    /// can't be profiled, so the caller passes nil and nothing opens.
    static func open(opponentID: String?, name: String?, controller: LichessBotController) {
        guard let opponentID, !opponentID.isEmpty else { return }
        let id = opponentID.lowercased()
        SessionLogger.shared.log("[BUTTON] Open Lichess opponent \(id)")
        if let existing = controllers[id] {
            existing.showWindow(nil)
            existing.window?.makeKeyAndOrderFront(nil)
            return
        }
        let windowController = LichessBotOpponentDetailWindowController(opponentID: id, name: name ?? id, controller: controller)
        controllers[id] = windowController
        windowController.showWindow(nil)
        windowController.window?.makeKeyAndOrderFront(nil)
    }

    static func unregister(_ windowController: LichessBotOpponentDetailWindowController) {
        if controllers[windowController.opponentID] === windowController {
            controllers[windowController.opponentID] = nil
        }
    }
}
