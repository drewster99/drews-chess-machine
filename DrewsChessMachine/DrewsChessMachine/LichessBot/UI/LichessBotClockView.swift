import SwiftUI

/// One player's clock. While it is that player's turn in a live game it
/// counts down locally from the server's last reading; the next server
/// update corrects it.
struct LichessBotClockView: View {
    let milliseconds: Int?
    let receivedAt: Date?
    let isRunning: Bool
    let isOurs: Bool

    var body: some View {
        TimelineView(.periodic(from: .now, by: isRunning ? 0.1 : 3600)) { context in
            Text(Self.format(remaining(at: context.date)))
                .font(.system(.title3, design: .monospaced).weight(isRunning ? .semibold : .regular))
                .foregroundStyle(color(at: context.date))
                .padding(.horizontal, 8)
                .padding(.vertical, 2)
                .background(
                    RoundedRectangle(cornerRadius: 4)
                        .fill(Color.accentColor.opacity(isRunning ? 0.15 : 0))
                )
        }
        .accessibilityLabel(isOurs ? "DCM clock" : "Opponent clock")
    }

    private func remaining(at date: Date) -> Int? {
        guard let milliseconds else { return nil }
        guard isRunning, let receivedAt else { return milliseconds }
        return max(0, milliseconds - Int(date.timeIntervalSince(receivedAt) * 1000))
    }

    private func color(at date: Date) -> Color {
        guard let remaining = remaining(at: date) else { return .secondary }
        return remaining < 10_000 ? .red : .primary
    }

    static func format(_ milliseconds: Int?) -> String {
        guard let milliseconds else { return "–:––" }
        let totalSeconds = milliseconds / 1000
        if totalSeconds < 10 {
            return String(format: "0:%02d.%d", totalSeconds, (milliseconds % 1000) / 100)
        }
        if totalSeconds >= 3600 {
            return String(format: "%d:%02d:%02d", totalSeconds / 3600, (totalSeconds / 60) % 60, totalSeconds % 60)
        }
        return String(format: "%d:%02d", totalSeconds / 60, totalSeconds % 60)
    }
}
