import SwiftUI

/// Start / back / forward / live buttons for browsing a game's positions,
/// and a banner saying which ply is shown and how many were played (plan
/// §14.3a). Browsing is for viewing only; nothing here changes the game.
struct GameBrowseControlsView: View {
    @Binding var cursor: GameBrowseCursor
    let totalPlies: Int
    /// Arrow-key shortcuts; only one view in a window should claim them.
    let claimsKeyboardShortcuts: Bool

    var body: some View {
        HStack(spacing: 6) {
            Button(
                action: { cursor.goToStart(totalPlies: totalPlies) },
                label: { Image(systemName: "backward.end.fill") }
            )
            .help("First position")
            .disabled(cursor.displayedPlyCount(totalPlies: totalPlies) == 0)
            .keyboardShortcut(claimsKeyboardShortcuts ? KeyboardShortcut(.home, modifiers: []) : nil)

            Button(
                action: { cursor.stepBack(totalPlies: totalPlies) },
                label: { Image(systemName: "chevron.left") }
            )
            .help("Previous position")
            .disabled(cursor.displayedPlyCount(totalPlies: totalPlies) == 0)
            .keyboardShortcut(claimsKeyboardShortcuts ? KeyboardShortcut(.leftArrow, modifiers: []) : nil)

            Button(
                action: { cursor.stepForward(totalPlies: totalPlies) },
                label: { Image(systemName: "chevron.right") }
            )
            .help("Next position")
            .disabled(cursor.isLive)
            .keyboardShortcut(claimsKeyboardShortcuts ? KeyboardShortcut(.rightArrow, modifiers: []) : nil)

            Button(
                action: { cursor.goLive() },
                label: { Image(systemName: "forward.end.fill") }
            )
            .help("Back to the live position")
            .disabled(cursor.isLive)
            .keyboardShortcut(claimsKeyboardShortcuts ? KeyboardShortcut(.end, modifiers: []) : nil)

            Text(bannerText)
                .font(.system(.callout, design: .monospaced))
                .foregroundStyle(cursor.isLive ? Color.secondary : Color.orange)
                .lineLimit(1)
            Spacer(minLength: 0)
        }
        .buttonStyle(.borderless)
    }

    private var bannerText: String {
        let displayed = cursor.displayedPlyCount(totalPlies: totalPlies)
        if cursor.isLive {
            return String(format: "Live · %3d plies played", totalPlies)
        }
        return String(format: "Viewing after ply %3d · %3d played", displayed, totalPlies)
    }
}
