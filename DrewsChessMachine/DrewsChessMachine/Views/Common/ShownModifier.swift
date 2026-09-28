import SwiftUI

extension View {
    /// Keep the view in the hierarchy — so its identity and state stay
    /// stable — but hide and collapse it when `isShown` is false. The
    /// project's alternative to `if`-gating visible content.
    func shown(_ isShown: Bool) -> some View {
        opacity(isShown ? 1 : 0)
            .frame(width: isShown ? nil : 0, height: isShown ? nil : 0)
            .allowsHitTesting(isShown)
            .accessibilityHidden(!isShown)
    }
}
