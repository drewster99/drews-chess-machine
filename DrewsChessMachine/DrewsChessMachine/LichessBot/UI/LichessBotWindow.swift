import AppKit
import SwiftUI

/// The Lichess Bot window (plan §14.1). Closing it never stops the bot: the
/// bot belongs to the app.
@MainActor
final class LichessBotWindowController: NSWindowController, NSWindowDelegate {
    init(controller: LichessBotController) {
        let hosting = NSHostingController(rootView: LichessBotRootView(controller: controller))
        // The root view's `navigationTitle` (the account, its state and the
        // games in progress) becomes the window's title.
        hosting.sceneBridgingOptions.insert(.title)
        let window = NSWindow(contentViewController: hosting)
        window.setContentSize(NSSize(width: 1180, height: 820))
        window.minSize = NSSize(width: 900, height: 600)
        window.title = LichessBotWindowTitle.name
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
        LichessBotWindowLauncher.unregister(self)
    }
}

/// Opens the single Lichess Bot window, or brings it forward.
@MainActor
enum LichessBotWindowLauncher {
    private static var current: LichessBotWindowController?

    static func openWindow(controller: LichessBotController) {
        SessionLogger.shared.log("[BUTTON] Open Lichess Bot")
        if let current {
            current.showWindow(nil)
            current.window?.makeKeyAndOrderFront(nil)
            return
        }
        let windowController = LichessBotWindowController(controller: controller)
        current = windowController
        windowController.showWindow(nil)
        windowController.window?.makeKeyAndOrderFront(nil)
    }

    static func unregister(_ windowController: LichessBotWindowController) {
        if current === windowController {
            current = nil
        }
    }
}
