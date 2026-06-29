import SwiftUI

/// Small activity spinner for the top-bar status chips.
///
/// Exists because SwiftUI's `ProgressView` spinner on macOS is backed
/// by `NSProgressIndicator`, which ignores `.tint` — it always draws
/// in the system gray, which is nearly invisible against the chips'
/// saturated green/blue/orange backgrounds. This draws a simple
/// three-quarter arc in an explicit color instead, so the chip's
/// foreground color actually applies and the motion reads clearly on
/// any chip background, in light and dark mode alike.
struct ChipActivitySpinner: View {
    /// Stroke color — pass the chip's foreground color.
    let color: Color

    @State private var rotating = false

    var body: some View {
        Circle()
            .trim(from: 0, to: 0.75)
            .stroke(color, style: StrokeStyle(lineWidth: 1.8, lineCap: .round))
            .frame(width: 11, height: 11)
            .rotationEffect(.degrees(rotating ? 360 : 0))
            .animation(
                .linear(duration: 0.9).repeatForever(autoreverses: false),
                value: rotating
            )
            .onAppear { rotating = true }
    }
}
