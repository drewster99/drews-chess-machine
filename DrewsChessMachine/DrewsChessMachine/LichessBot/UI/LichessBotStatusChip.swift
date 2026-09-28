import SwiftUI

/// The bot's status in the main window's title bar (plan §14.2): a state dot
/// and "Online · 2 games". Click to open the Lichess Bot window; the context
/// menu has the on/off controls.
struct LichessBotStatusChip: View {
    let controller: LichessBotController

    var body: some View {
        Button {
            LichessBotWindowLauncher.openWindow(controller: controller)
        } label: {
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
    }

    private var text: String {
        let games = controller.activeGameIDs.count
        let base = "Lichess: \(controller.connection.label)"
        return games > 0 ? "\(base) · \(games) game\(games == 1 ? "" : "s")" : base
    }
}
