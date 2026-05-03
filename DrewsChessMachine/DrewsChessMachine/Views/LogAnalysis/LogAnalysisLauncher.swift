import AppKit
import Foundation

/// Bridges the SwiftUI Debug menu button to the window launcher.
/// Handles the three preconditions (claude binary exists and is
/// executable, there's an active log, the log can be read) and
/// surfaces a clear NSAlert for any failure instead of silently
/// doing nothing.
@MainActor
enum LogAnalysisLauncher {
    static func openWindow() {
        let home = NSString(string: "~").expandingTildeInPath
        let claudePath = "\(home)/.local/bin/claude"
        let fm = FileManager.default

        guard fm.fileExists(atPath: claudePath),
              fm.isExecutableFile(atPath: claudePath) else {
            presentAlert(
                title: "Claude CLI not found",
                info: "~/.local/bin/claude does not exist or is not executable."
            )
            return
        }

        guard let logPath = SessionLogger.shared.activeLogPath else {
            presentAlert(
                title: "No active session log",
                info: "The session logger has not opened a file. Start the app session logging and try again."
            )
            return
        }

        let logContent: String
        do {
            let data = try Data(contentsOf: URL(fileURLWithPath: logPath))
            logContent = String(data: data, encoding: .utf8) ?? ""
        } catch {
            presentAlert(
                title: "Could not read session log",
                info: "\(logPath): \(error.localizedDescription)"
            )
            return
        }

        let controller = LogAnalysisWindowController(
            logPath: logPath,
            logContent: logContent,
            claudePath: claudePath
        )
        LogAnalysisWindowRegistry.shared.register(controller)
        controller.showWindow(nil)
        controller.window?.makeKeyAndOrderFront(nil)
    }

    private static func presentAlert(title: String, info: String) {
        let alert = NSAlert()
        alert.messageText = title
        alert.informativeText = info
        alert.alertStyle = .warning
        alert.addButton(withTitle: "OK")
        alert.runModal()
    }
}
