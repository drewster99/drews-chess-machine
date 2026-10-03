import SwiftUI

/// The bot's status in the main window's title bar (plan §14.2): a state
/// dot, the state, and the number of games in progress. Click to open the
/// Lichess Bot window; the context menu has the on/off controls.
struct LichessBotStatusChip: View {
    let controller: LichessBotController

    var body: some View {
        Button(
            action: {
                LichessBotWindowLauncher.openWindow(controller: controller)
            },
            label: {
                HStack(spacing: 5) {
                    Circle()
                        .fill(LichessBotStatusStyle.color(for: controller.connection))
                        .frame(width: 8, height: 8)
                    Text(text)
                        .font(.callout)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                }
                .padding(.horizontal, 6)
                .padding(.vertical, 2)
                .background(Capsule().fill(Color.gray.opacity(0.12)))
            }
        )
        .buttonStyle(.plain)
        .help("Lichess bot — click to open")
        .contextMenu {
            Button("Open Lichess Bot") {
                LichessBotWindowLauncher.openWindow(controller: controller)
            }
            Divider()
            Button("Go Online") {
                Task { await controller.goOnline() }
            }
            .disabled(controller.isRunning)
            Button("Stop Accepting (Drain)") {
                Task { await controller.drain() }
            }
            .disabled(controller.connection != .online)
            Button("Go Offline") {
                Task { await controller.goOffline() }
            }
            .disabled(!controller.isRunning)
        }
        .accessibilityLabel("Lichess bot: \(text)")
        .task {
            await controller.noteLeftoverJournalsAtLaunch()
        }
    }

    private var text: String {
        let games = controller.activeGameIDs.count
        let base = "Lichess: \(controller.connection.label)"
        if games > 0 {
            return "\(base) · \(games) game\(games == 1 ? "" : "s")"
        }
        // Unfinished games may be running on DCM's clock; finished ones
        // only wait to be filed. Both are settled by going online.
        var lastRun: [String] = []
        let unfinished = controller.leftoverGamesFromLastRun.count
        if unfinished > 0 {
            lastRun.append("\(unfinished) unfinished")
        }
        let toFile = controller.finishedGamesAwaitingFilingFromLastRun.count
        if toFile > 0 {
            lastRun.append("\(toFile) to file")
        }
        if !lastRun.isEmpty {
            return "\(base) · \(lastRun.joined(separator: ", ")) from last run"
        }
        return base
    }
}
