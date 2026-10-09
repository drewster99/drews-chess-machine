import SwiftUI

/// Copies a Lichess user's profile address, then shows a checkmark for a
/// moment so the copy is visibly confirmed.
struct LichessBotCopyProfileLinkButton: View {
    let username: String
    /// Set by a copy; the checkmark shows until the confirmation period
    /// passes. A new copy restarts the period.
    @State private var copiedAt: Date?

    private static let copiedConfirmation: Duration = .seconds(1.5)

    var body: some View {
        Button {
            LichessBotLinks.copyUser(username)
            copiedAt = Date()
        } label: {
            Image(systemName: copiedAt == nil ? "doc.on.doc" : "checkmark")
                .frame(width: 16)
        }
        .buttonStyle(.borderless)
        .help(copiedAt == nil ? "Copy the link to \(username)'s Lichess page" : "Copied")
        .task(id: copiedAt) {
            await endConfirmation()
        }
    }

    private func endConfirmation() async {
        guard copiedAt != nil else { return }
        do {
            try await Task.sleep(for: Self.copiedConfirmation)
        } catch {
            // Cancelled by a newer copy (which starts its own period) or by
            // the view going away; either way this period is over.
            return
        }
        copiedAt = nil
    }
}
