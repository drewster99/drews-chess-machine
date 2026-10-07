import SwiftUI

/// The account's rating in each speed, its rated games there (from
/// Lichess), and its unrated games, games today and games in the last 24
/// hours there. Lichess reports unrated games only as an account-wide total,
/// so those three columns count DCM's own games — every game this BOT
/// account plays goes through DCM.
struct LichessBotAccountRatingsGrid: View {
    let controller: LichessBotController

    var body: some View {
        // Today and the last 24 hours move with the clock, not with any
        // state change, so the counts are re-read every minute — on the
        // minute, so Today turns over at midnight. The content reads the
        // controller in its own body, so a game starting or being filed
        // redraws it too.
        TimelineView(.everyMinute) { context in
            LichessBotAccountRatingsGridContent(controller: controller, now: context.date)
        }
    }
}
