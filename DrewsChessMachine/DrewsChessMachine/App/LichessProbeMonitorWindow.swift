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

    init(
        history: LichessProbeHistory,
        onProbeNow: @escaping @MainActor () -> Void,
        onOpenDetail: @escaping @MainActor () -> Void
    ) {
        self.history = history
        let view = LichessProbeMonitorView(
            history: history,
            onProbeNow: onProbeNow,
            onOpenDetail: onOpenDetail
        )
        let hosting = NSHostingController(rootView: view)
        let window = NSWindow(contentViewController: hosting)
        window.setContentSize(NSSize(width: 1000, height: 380))
        window.minSize = NSSize(width: 880, height: 280)
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

/// Single-instance registry for the Lichess Probe Monitor window. Holds
/// at most one controller so menu-driven opens never stack duplicate
/// windows — a `register` call when a window is already alive is a
/// programmer-error precondition. The launcher's `firstOpen` check
/// upstream guarantees the precondition holds in normal flow.
@MainActor
final class LichessProbeMonitorWindowRegistry {
    static let shared = LichessProbeMonitorWindowRegistry()
    private(set) var controller: LichessProbeMonitorWindowController?

    private init() {}

    func register(_ controller: LichessProbeMonitorWindowController) {
        precondition(
            self.controller == nil,
            "LichessProbeMonitorWindowRegistry: a window is already registered"
        )
        self.controller = controller
    }

    func unregister(_ controller: LichessProbeMonitorWindowController) {
        if self.controller === controller {
            self.controller = nil
        }
    }
}

/// Bridges the Debug menu button to the window launcher.
@MainActor
enum LichessProbeMonitorLauncher {
    static func openWindow(sessionController: SessionController) {
        SessionLogger.shared.log("[BUTTON] Open Lichess Probe Monitor")
        if let existing = LichessProbeMonitorWindowRegistry.shared.controller {
            existing.showWindow(nil)
            existing.window?.makeKeyAndOrderFront(nil)
            return
        }
        let controller = LichessProbeMonitorWindowController(
            history: sessionController.lichessProbeHistory,
            onProbeNow: { [weak sessionController] in
                sessionController?.triggerLichessProbeNow()
            },
            onOpenDetail: { [weak sessionController] in
                guard let session = sessionController else { return }
                LichessProbeDetailLauncher.openWindow(sessionController: session)
            }
        )
        LichessProbeMonitorWindowRegistry.shared.register(controller)
        controller.showWindow(nil)
        controller.window?.makeKeyAndOrderFront(nil)
    }
}
