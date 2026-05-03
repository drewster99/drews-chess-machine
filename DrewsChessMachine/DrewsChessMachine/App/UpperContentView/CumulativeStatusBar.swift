import SwiftUI

/// Top status bar — sums across all completed Play-and-Train
/// segments + the in-flight one. Always shows the right-side
/// chips (session-status + arena countdown). The left-side
/// history cells (training time, steps, positions, rates,
/// run/arena/promotion counts, score) are supplied by the caller
/// as a `@ViewBuilder` closure so this shell stays decoupled
/// from the parent's state-heavy snapshot fields. The bar
/// renders only when there is either a history to summarize OR a
/// runnable arena setup, so a fresh-never-trained session shows
/// no bar at all.
struct CumulativeStatusBar<History: View, Chips: View>: View {
    /// `true` when there is anything to summarize (any completed
    /// run or any training-step count) — drives the visibility of
    /// the left-side history cells.
    let hasHistory: Bool
    /// `true` when the parent should render *something*, even if
    /// `hasHistory` is false (e.g. an arena can be kicked off so
    /// the right-side chips remain useful). Computed by the
    /// caller as `hasHistory || canRunArena`.
    let isVisible: Bool
    @ViewBuilder var historyCells: () -> History
    @ViewBuilder var rightChips: () -> Chips

    var body: some View {
        if isVisible {
            HStack(spacing: 16) {
                if hasHistory {
                    historyCells()
                }
                Spacer()
                rightChips()
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
            .background(Color.secondary.opacity(0.10))
            .clipShape(RoundedRectangle(cornerRadius: 6))
        }
    }
}
