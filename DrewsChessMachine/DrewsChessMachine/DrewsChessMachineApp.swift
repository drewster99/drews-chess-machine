import SwiftUI

@main
struct DrewsChessMachineApp: App {
    init() {
        // Start the session logger before any view work so every event
        // from this launch — button taps, arena results, periodic
        // stats — lands in a single `dcm_log_yyyymmdd-HHMMSS.txt`
        // file under the app's Library/Logs directory.
        SessionLogger.shared.start()
        SessionLogger.shared.log("[APP] launched")
        if let path = SessionLogger.shared.activeLogPath {
            print("[APP] session log: \(path)")
        } else {
            print("[APP] session log: (failed to open)")
        }
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
