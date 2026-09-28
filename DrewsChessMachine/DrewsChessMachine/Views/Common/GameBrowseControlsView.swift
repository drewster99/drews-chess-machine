import SwiftUI

/// Start / back / forward / live buttons for browsing a game's positions,
/// and the "Viewing ply N · live at ply M" banner while not live (plan
/// §14.3a). Browsing is for viewing only; nothing here changes the game.
struct GameBrowseControlsView: View {
    @Binding var cursor: GameBrowseCursor
    let totalPlies: Int
    /// Arrow-key shortcuts; only one view in a window should claim them.
    let claimsKeyboardShortcuts: Bool

    var body: some View {
        HStack(spacing: 6) {
            Button {
                cursor.goToStart(totalPlies: totalPlies)
            } label: {
                Image(systemName: "backward.end.fill")
            }
            .help("First position")
            .disabled(cursor.displayedPlyCount(totalPlies: totalPlies) == 0)
            .keyboardShortcut(claimsKeyboardShortcuts ? KeyboardShortcut(.home, modifiers: []) : nil)

            Button {
                cursor.stepBack(totalPlies: totalPlies)
            } label: {
                Image(systemName: "chevron.left")
            }
            .help("Previous position")
            .disabled(cursor.displayedPlyCount(totalPlies: totalPlies) == 0)
            .keyboardShortcut(claimsKeyboardShortcuts ? KeyboardShortcut(.leftArrow, modifiers: []) : nil)

            Button {
                cursor.stepForward(totalPlies: totalPlies)
            } label: {
                Image(systemName: "chevron.right")
            }
            .help("Next position")
            .disabled(cursor.isLive)
            .keyboardShortcut(claimsKeyboardShortcuts ? KeyboardShortcut(.rightArrow, modifiers: []) : nil)

            Button {
                cursor.goLive()
            } label: {
                Image(systemName: "forward.end.fill")
            }
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
            return String(format: "Live · ply %3d", totalPlies)
        }
        return String(format: "Viewing ply %3d · live at ply %3d", displayed, totalPlies)
    }
}
