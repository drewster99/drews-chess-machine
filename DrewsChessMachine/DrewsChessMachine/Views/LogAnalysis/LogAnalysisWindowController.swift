import AppKit
import SwiftUI

/// Standalone window that opens the current session log and, in
/// parallel, runs `cat <log> | ~/.local/bin/claude -p "..."` to
/// generate an analysis. Top pane shows the raw log in a
/// monospaced font; bottom pane shows Claude's markdown-formatted
/// response as it arrives. Each analysis window is independent —
/// a new one is built every time the user selects Debug > Analyze
/// Log so a long-running analysis doesn't block a second pass.
@MainActor
final class LogAnalysisWindowController: NSWindowController, NSWindowDelegate {
    /// The view model is owned here (rather than only inside the
    /// SwiftUI view tree) so we can reach in on window-close and
    /// terminate the running `claude` subprocess. Otherwise a
    /// user closing the window mid-analysis would leave a zombie
    /// process until it finished on its own.
    private let viewModel: LogAnalysisViewModel

    init(logPath: String, logContent: String, claudePath: String) {
        let vm = LogAnalysisViewModel(
            logPath: logPath,
            logContent: logContent,
            claudePath: claudePath
        )
        self.viewModel = vm
        let view = LogAnalysisView(viewModel: vm)
        let hosting = NSHostingController(rootView: view)
        let window = NSWindow(contentViewController: hosting)
        window.setContentSize(NSSize(width: 900, height: 800))
        window.minSize = NSSize(width: 500, height: 400)
        window.title = "Log Analysis — \((logPath as NSString).lastPathComponent)"
        window.isReleasedWhenClosed = false
        window.center()
        super.init(window: window)
        window.delegate = self
        vm.start()
    }

    required init?(coder: NSCoder) {
        fatalError("init(coder:) not supported for LogAnalysisWindowController")
    }

    func windowWillClose(_ notification: Notification) {
        viewModel.cancel()
        LogAnalysisWindowRegistry.shared.unregister(self)
    }
}
