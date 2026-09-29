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
///
/// The rotation angle is computed from the wall clock inside a
/// `TimelineView` rather than started as a `repeatForever` implicit
/// animation from `onAppear`. The implicit-animation version also
/// captured the view's own layout position at the moment it appeared
/// (the chip is laid out inside a status bar that is still settling),
/// so instead of spinning in place the arc endlessly re-played a slide
/// from its initial position into the chip. Deriving the angle from
/// time has no animation transaction at all, so only rotation moves.
struct ChipActivitySpinner: View {
    /// Stroke color — pass the chip's foreground color.
    let color: Color

    /// Seconds per full revolution.
    private static let revolutionSeconds: Double = 0.9

    var body: some View {
        TimelineView(.animation) { context in
            let seconds = context.date.timeIntervalSinceReferenceDate
            let fraction = seconds.truncatingRemainder(dividingBy: Self.revolutionSeconds)
                / Self.revolutionSeconds
            Circle()
                .trim(from: 0, to: 0.75)
                .stroke(color, style: StrokeStyle(lineWidth: 1.8, lineCap: .round))
                .frame(width: 11, height: 11)
                .rotationEffect(.degrees(fraction * 360))
        }
    }
}
