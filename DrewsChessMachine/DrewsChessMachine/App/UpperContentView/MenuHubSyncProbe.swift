import SwiftUI

/// Hidden zero-sized view that drives `syncMenuCommandHubState()`
/// off a single Equatable signature, replacing what used to be a
/// 13-deep `.onChange` chain on `body`'s tail.
struct MenuHubSyncProbe: View {
    let signature: MenuHubSignature
    let onSignatureChanged: () -> Void

    var body: some View {
        Color.clear
            .frame(width: 0, height: 0)
            .onChange(of: signature) { _, _ in onSignatureChanged() }
    }
}
