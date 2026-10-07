import SwiftUI

/// DCM's record and statistics from the saved game records
/// (`LICHESS_BOT_RECORD_STATS_PLAN.md` §5): the period table and the
/// statistics panel beside (or above) the recent games.
struct LichessBotRecordCard: View {
    let controller: LichessBotController
    /// The card's content height, set by dragging its bottom edge and
    /// remembered across launches (a viewing preference, not a bot setting).
    /// The declared default applies only while the key holds no value, so
    /// an operator who already dragged it keeps their height (OD-19).
    @AppStorage("lichessBot.overview.recordCardHeight") private var contentHeight: Double = 560

    /// Short enough to keep the card compact, never so short the period
    /// table's rows are cut off.
    private static let contentHeightRange: ClosedRange<Double> = 160...1600

    var body: some View {
        GroupBox("Record") {
            VStack(spacing: 4) {
                // Applied here rather than passed in, so dragging the handle
                // doesn't recompute anything on every frame.
                LichessBotRecordCardContent(controller: controller)
                    .frame(height: min(max(contentHeight, Self.contentHeightRange.lowerBound), Self.contentHeightRange.upperBound), alignment: .top)
                LichessBotHeightResizeHandle(height: $contentHeight, range: Self.contentHeightRange)
            }
        }
    }
}
