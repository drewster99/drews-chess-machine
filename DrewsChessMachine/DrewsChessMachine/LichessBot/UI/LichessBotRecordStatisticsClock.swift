import SwiftUI

/// The Record card's clock, attached to the bot window
/// (`LICHESS_BOT_RECORD_STATS_PLAN.md` §4.3): while the window is open, every
/// 60 s the statistics recompute if a period boundary passed (the rolling
/// hour, midnight, a week, month or year) or the time zone changed.
///
/// Its own task, not a step of the window's loading task, whose awaits run
/// in sequence. `Task.sleep` runs on the continuous clock, which keeps
/// counting while the Mac sleeps, so a wake is caught at the next tick; a
/// wall-clock or time-zone change recomputes at once through the system
/// notifications the pipeline starts observing here.
struct LichessBotRecordStatisticsClock: ViewModifier {
    let pipeline: LichessBotRecordStatisticsPipeline

    static let tickInterval: Duration = .seconds(60)

    func body(content: Content) -> some View {
        content.task {
            pipeline.observeSystemTimeChanges()
            await pipeline.runClock(tickInterval: Self.tickInterval)
        }
    }
}
