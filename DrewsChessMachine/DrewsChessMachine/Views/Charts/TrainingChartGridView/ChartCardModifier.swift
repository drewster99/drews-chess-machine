import SwiftUI

/// Background card chrome shared by every grid tile.
extension View {
    func chartCard() -> some View {
        self
            .padding(6)
            .background(Color(nsColor: .controlBackgroundColor))
    }
}
