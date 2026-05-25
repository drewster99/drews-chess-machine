import AppKit
import SwiftUI

/// Standalone window for the Tactical Probe Monitor. Mirrors the
/// `LogAnalysisWindowController` pattern: NSWindowController +
/// NSWindowDelegate, kept alive by `TacticalProbeMonitorWindowRegistry`
/// while the window is open. Owns the `TacticalProbeHistory` store and
/// the `TacticalProbeWatcher` driver; the watcher's tick loop starts
/// on window open and is cancelled in `windowWillClose(_:)`.
@MainActor
final class TacticalProbeMonitorWindowController: NSWindowController, NSWindowDelegate {
    let history: TacticalProbeHistory
    let watcher: TacticalProbeWatcher

    init(sessionController: SessionController) {
        let history = TacticalProbeHistory()
        self.history = history
        let watcher = TacticalProbeWatcher(
            sessionController: sessionController,
            history: history
        )
        self.watcher = watcher
        let view = TacticalProbeMonitorView(history: history)
        let hosting = NSHostingController(rootView: view)
        let window = NSWindow(contentViewController: hosting)
        window.setContentSize(NSSize(width: 1080, height: 460))
        window.minSize = NSSize(width: 820, height: 320)
        window.title = "Tactical Probe Monitor"
        window.isReleasedWhenClosed = false
        window.center()
        super.init(window: window)
        window.delegate = self
        watcher.start()
    }

    required init?(coder: NSCoder) {
        fatalError("init(coder:) not supported for TacticalProbeMonitorWindowController")
    }

    func windowWillClose(_ notification: Notification) {
        watcher.stop()
        TacticalProbeMonitorWindowRegistry.shared.unregister(self)
    }
}

/// Keeps open `TacticalProbeMonitorWindowController` instances alive
/// for as long as their window is on-screen. Without this, a
/// menu-driven `showWindow(nil)` call would let ARC tear down the
/// controller (and its window) the moment the launching closure
/// returned. Same pattern as `LogAnalysisWindowRegistry`.
@MainActor
final class TacticalProbeMonitorWindowRegistry {
    static let shared = TacticalProbeMonitorWindowRegistry()
    private var controllers: [TacticalProbeMonitorWindowController] = []

    private init() {}

    func register(_ controller: TacticalProbeMonitorWindowController) {
        controllers.append(controller)
    }

    func unregister(_ controller: TacticalProbeMonitorWindowController) {
        controllers.removeAll { $0 === controller }
    }
}

/// Bridges the Debug menu button to the window launcher. Validates
/// the precondition (a champion network must exist — the watcher
/// no-ops gracefully on tick if it disappears mid-session, but
/// opening with no network would show 7 empty rows forever) and
/// surfaces an NSAlert when blocked. Otherwise builds the controller,
/// registers it, shows the window.
@MainActor
enum TacticalProbeMonitorLauncher {
    static func openWindow(sessionController: SessionController) {
        guard sessionController.network != nil else {
            let alert = NSAlert()
            alert.messageText = "No champion network"
            alert.informativeText = "Build a network (or load a model / session) before opening the Tactical Probe Monitor."
            alert.alertStyle = .warning
            alert.addButton(withTitle: "OK")
            alert.runModal()
            return
        }

        SessionLogger.shared.log("[BUTTON] Open Tactical Probe Monitor")
        let controller = TacticalProbeMonitorWindowController(sessionController: sessionController)
        TacticalProbeMonitorWindowRegistry.shared.register(controller)
        controller.showWindow(nil)
        controller.window?.makeKeyAndOrderFront(nil)
    }
}
