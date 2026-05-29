import AppKit
import SwiftUI

/// Standalone window for the Lichess Probe **Detail** view — the per-
/// puzzle drill-down companion of the aggregate `LichessProbeMonitor`.
/// Observes the same shared `LichessProbeHistory`, so opening/closing
/// either window has no effect on watcher cadence.
@MainActor
final class LichessProbeDetailWindowController: NSWindowController, NSWindowDelegate {
    let history: LichessProbeHistory

    init(
        history: LichessProbeHistory,
        onProbeNow: @escaping @MainActor () -> Void,
        onExport: @escaping @MainActor () -> Void
    ) {
        self.history = history
        let view = LichessProbeDetailView(
            history: history,
            onProbeNow: onProbeNow,
            onExport: onExport
        )
        let hosting = NSHostingController(rootView: view)
        let window = NSWindow(contentViewController: hosting)
        window.setContentSize(NSSize(width: 1100, height: 700))
        window.minSize = NSSize(width: 900, height: 400)
        window.title = "Lichess Probe Detail"
        window.isReleasedWhenClosed = false
        window.center()
        super.init(window: window)
        window.delegate = self
    }

    required init?(coder: NSCoder) {
        fatalError(
            "init(coder:) not supported for LichessProbeDetailWindowController"
        )
    }

    func windowWillClose(_ notification: Notification) {
        LichessProbeDetailWindowRegistry.shared.unregister(self)
    }
}

/// Strong-ref registry so menu-driven opens survive past the launcher
/// closure. Mirrors the existing `LichessProbeMonitorWindowRegistry`.
@MainActor
final class LichessProbeDetailWindowRegistry {
    static let shared = LichessProbeDetailWindowRegistry()
    private var controllers: [LichessProbeDetailWindowController] = []

    private init() {}

    func register(_ controller: LichessProbeDetailWindowController) {
        controllers.append(controller)
    }

    func unregister(_ controller: LichessProbeDetailWindowController) {
        controllers.removeAll { $0 === controller }
    }

    var firstOpen: LichessProbeDetailWindowController? {
        controllers.first
    }
}

@MainActor
enum LichessProbeDetailLauncher {
    static func openWindow(sessionController: SessionController) {
        SessionLogger.shared.log("[BUTTON] Open Lichess Probe Detail")
        if let existing = LichessProbeDetailWindowRegistry.shared.firstOpen {
            existing.showWindow(nil)
            existing.window?.makeKeyAndOrderFront(nil)
            return
        }
        let controller = LichessProbeDetailWindowController(
            history: sessionController.lichessProbeHistory,
            onProbeNow: { [weak sessionController] in
                sessionController?.triggerLichessProbeNow()
            },
            onExport: { [weak sessionController] in
                guard let session = sessionController else { return }
                LichessProbeExporter.exportLatest(
                    history: session.lichessProbeHistory
                )
            }
        )
        LichessProbeDetailWindowRegistry.shared.register(controller)
        controller.showWindow(nil)
        controller.window?.makeKeyAndOrderFront(nil)
    }
}
