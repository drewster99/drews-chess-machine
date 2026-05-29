import AppKit
import SwiftUI

/// Standalone window for the Lichess Probe Monitor — parallel to
/// `TacticalProbeMonitorWindowController` but observes the 200-puzzle
/// per-theme history. The history and watcher are owned by
/// `SessionController` so opening/closing the window doesn't start or
/// stop ticking; the window is purely an observer.
@MainActor
final class LichessProbeMonitorWindowController: NSWindowController, NSWindowDelegate {
    let history: LichessProbeHistory

    init(history: LichessProbeHistory, onProbeNow: @escaping @MainActor () -> Void) {
        self.history = history
        let view = LichessProbeMonitorView(history: history, onProbeNow: onProbeNow)
        let hosting = NSHostingController(rootView: view)
        let window = NSWindow(contentViewController: hosting)
        window.setContentSize(NSSize(width: 880, height: 380))
        window.minSize = NSSize(width: 760, height: 280)
        window.title = "Lichess Probe Monitor"
        window.isReleasedWhenClosed = false
        window.center()
        super.init(window: window)
        window.delegate = self
    }

    required init?(coder: NSCoder) {
        fatalError(
            "init(coder:) not supported for LichessProbeMonitorWindowController"
        )
    }

    func windowWillClose(_ notification: Notification) {
        LichessProbeMonitorWindowRegistry.shared.unregister(self)
    }
}

/// Keeps `LichessProbeMonitorWindowController` instances alive for as
/// long as their window is on-screen. Without this, a menu-driven
/// `showWindow(nil)` would let ARC tear down the controller as soon as
/// the launching closure returned. Mirrors the existing
/// `TacticalProbeMonitorWindowRegistry`.
@MainActor
final class LichessProbeMonitorWindowRegistry {
    static let shared = LichessProbeMonitorWindowRegistry()
    private var controllers: [LichessProbeMonitorWindowController] = []

    private init() {}

    func register(_ controller: LichessProbeMonitorWindowController) {
        controllers.append(controller)
    }

    func unregister(_ controller: LichessProbeMonitorWindowController) {
        controllers.removeAll { $0 === controller }
    }

    /// First currently-open monitor window. Used by the menu launcher
    /// so a second click brings the existing window to the front
    /// rather than stacking a duplicate.
    var firstOpen: LichessProbeMonitorWindowController? {
        controllers.first
    }
}

/// Bridges the Debug menu button to the window launcher.
@MainActor
enum LichessProbeMonitorLauncher {
    static func openWindow(sessionController: SessionController) {
        SessionLogger.shared.log("[BUTTON] Open Lichess Probe Monitor")
        if let existing = LichessProbeMonitorWindowRegistry.shared.firstOpen {
            existing.showWindow(nil)
            existing.window?.makeKeyAndOrderFront(nil)
            return
        }
        let controller = LichessProbeMonitorWindowController(
            history: sessionController.lichessProbeHistory,
            onProbeNow: { [weak sessionController] in
                sessionController?.triggerLichessProbeNow()
            }
        )
        LichessProbeMonitorWindowRegistry.shared.register(controller)
        controller.showWindow(nil)
        controller.window?.makeKeyAndOrderFront(nil)
    }
}
