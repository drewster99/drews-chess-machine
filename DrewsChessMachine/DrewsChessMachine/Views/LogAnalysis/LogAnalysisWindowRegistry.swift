import Foundation

/// Keeps `LogAnalysisWindowController` instances alive for as long
/// as their window is open. Without this a menu-driven
/// `showWindow(nil)` call would leave the controller unretained
/// and let ARC tear it down the moment the launching closure
/// returned, taking the window with it.
@MainActor
final class LogAnalysisWindowRegistry {
    static let shared = LogAnalysisWindowRegistry()
    private var controllers: [LogAnalysisWindowController] = []

    private init() {}

    func register(_ controller: LogAnalysisWindowController) {
        controllers.append(controller)
    }

    func unregister(_ controller: LogAnalysisWindowController) {
        controllers.removeAll { $0 === controller }
    }
}
