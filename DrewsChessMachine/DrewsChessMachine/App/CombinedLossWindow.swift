import AppKit
import SwiftUI

/// Standalone, freely-resizable window overlaying training total loss
/// against held-out eval loss on one dual-axis plot. Parallel in
/// structure to `LichessProbeMonitorWindowController`: the chart data
/// (the chart coordinator's training ring + the wide Lichess probe's
/// overall series) is owned by `SessionController` / `ChartCoordinator`,
/// so opening/closing the window is purely observational — it neither
/// starts nor stops any training or probing.
@MainActor
final class CombinedLossWindowController: NSWindowController, NSWindowDelegate {
    init(coordinator: ChartCoordinator, evalHistory: LichessProbeHistory) {
        let view = CombinedLossChartView(coordinator: coordinator, evalHistory: evalHistory)
        let hosting = NSHostingController(rootView: view)
        // The root SwiftUI view fills available space (`maxHeight: .infinity`),
        // which makes its fitting height unbounded. With the hosting
        // controller's default `.preferredContentSize` sizing, AppKit feeds
        // that unbounded height back as the window's content size during the
        // first user-driven resize/move layout pass, so the window snaps to
        // full screen height (under the dock and past it) and can't be made
        // smaller again. Detaching content-driven sizing makes the window
        // size purely user-controlled, with `setContentSize`/`minSize` below
        // as the only policy.
        hosting.sizingOptions = []
        let window = NSWindow(contentViewController: hosting)
        window.setContentSize(NSSize(width: 900, height: 520))
        window.minSize = NSSize(width: 520, height: 320)
        window.title = "Training vs Eval Loss"
        window.isReleasedWhenClosed = false
        window.center()
        super.init(window: window)
        window.delegate = self
    }

    required init?(coder: NSCoder) {
        fatalError("init(coder:) not supported for CombinedLossWindowController")
    }

    func windowWillClose(_ notification: Notification) {
        CombinedLossWindowRegistry.shared.unregister(self)
    }
}

/// Single-instance registry so menu-driven opens never stack duplicate
/// windows. Mirrors `LichessProbeMonitorWindowRegistry`.
@MainActor
final class CombinedLossWindowRegistry {
    static let shared = CombinedLossWindowRegistry()
    private(set) var controller: CombinedLossWindowController?

    private init() {}

    func register(_ controller: CombinedLossWindowController) {
        precondition(
            self.controller == nil,
            "CombinedLossWindowRegistry: a window is already registered"
        )
        self.controller = controller
    }

    func unregister(_ controller: CombinedLossWindowController) {
        if self.controller === controller {
            self.controller = nil
        }
    }
}

/// Bridges the Performance menu button to the window launcher.
@MainActor
enum CombinedLossWindowLauncher {
    static func openWindow(sessionController: SessionController) {
        SessionLogger.shared.log("[BUTTON] Open Training vs Eval Loss")
        if let existing = CombinedLossWindowRegistry.shared.controller {
            existing.showWindow(nil)
            existing.window?.makeKeyAndOrderFront(nil)
            return
        }
        guard let coordinator = sessionController.chartCoordinator else {
            // No chart coordinator wired yet — nothing to plot. Audible
            // cue rather than a silent no-op so a too-early open is
            // noticed; the menu item is otherwise gated on networkReady.
            NSSound.beep()
            SessionLogger.shared.log("[BUTTON] Open Training vs Eval Loss: no chart coordinator; ignoring")
            return
        }
        // Fill historical (stepless) training samples with interpolated
        // steps so the whole trajectory shows on the shared step axis.
        sessionController.backfillTrainingStepsIfNeeded()

        let controller = CombinedLossWindowController(
            coordinator: coordinator,
            evalHistory: sessionController.lichessProbeWideHistory
        )
        CombinedLossWindowRegistry.shared.register(controller)
        controller.showWindow(nil)
        controller.window?.makeKeyAndOrderFront(nil)
    }
}
