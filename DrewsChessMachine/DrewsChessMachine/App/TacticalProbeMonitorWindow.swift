import AppKit
import SwiftUI

/// Standalone window for the Tactical Probe Monitor. The history and
/// watcher are owned by `SessionController` (always-on, life-of-session)
/// so opening or closing this window doesn't start/stop ticking — the
/// window is purely an observer. Multiple monitor windows can be
/// opened simultaneously; they all observe the same shared history.
/// Kept alive by `TacticalProbeMonitorWindowRegistry` while open.
@MainActor
final class TacticalProbeMonitorWindowController: NSWindowController, NSWindowDelegate {
    let history: TacticalProbeHistory

    init(history: TacticalProbeHistory) {
        self.history = history
        let view = TacticalProbeMonitorView(history: history)
        let hosting = NSHostingController(rootView: view)
        let window = NSWindow(contentViewController: hosting)
        window.setContentSize(NSSize(width: 1220, height: 460))
        window.minSize = NSSize(width: 1000, height: 320)
        window.title = "Tactical Probe Monitor"
        window.isReleasedWhenClosed = false
        window.center()
        super.init(window: window)
        window.delegate = self
    }

    required init?(coder: NSCoder) {
        fatalError("init(coder:) not supported for TacticalProbeMonitorWindowController")
    }

    func windowWillClose(_ notification: Notification) {
        // Watcher lives on SessionController; closing the window is a
        // pure observer-detach. Nothing to stop here.
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

/// Bridges the Debug menu button to the window launcher. The watcher
/// keeps running regardless of whether a window is open — opening a
/// window with no network yet is fine: the existing history (empty
/// or otherwise) renders; subsequent ticks fill rows in once a network
/// is built.
@MainActor
enum TacticalProbeMonitorLauncher {
    static func openWindow(sessionController: SessionController) {
        SessionLogger.shared.log("[BUTTON] Open Tactical Probe Monitor")
        let controller = TacticalProbeMonitorWindowController(
            history: sessionController.tacticalProbeHistory
        )
        TacticalProbeMonitorWindowRegistry.shared.register(controller)
        controller.showWindow(nil)
        controller.window?.makeKeyAndOrderFront(nil)
    }
}
