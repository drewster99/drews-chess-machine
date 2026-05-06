import Foundation

/// One point on the Progress rate chart. Sampled once per second
/// from the heartbeat during a Play and Train session; cleared at
/// each new session. The `*MovesPerHour` fields are *rolling-window*
/// rates — over the last 3 minutes leading up to `timestamp`, not
/// lifetime averages — so the chart shows how throughput changes
/// over time rather than asymptoting to the session mean.
///
/// On-disk schema note: same warning as `TrainingChartSample` —
/// fields are part of the JSON envelope persisted in
/// `progress_rate_chart.json` inside `.dcmsession` bundles. Treat
/// them as additive-Optional only and bump `formatVersion` for any
/// breaking change.
struct ProgressRateSample: Identifiable, Sendable, Codable, Equatable {
    /// Monotonic identity for SwiftUI's `ForEach` / `Chart` — the
    /// index the sample was appended at. Stable for the life of
    /// the session and never reused.
    let id: Int
    /// Wall-clock instant this sample was taken. Used as the
    /// reference point when locating the 3-minute-ago sample that
    /// defines the lower edge of this point's rolling window.
    let timestamp: Date
    /// Seconds elapsed since `sessionStart`. Used as the X
    /// coordinate on the chart so each session starts fresh at 0
    /// rather than showing wall-clock time.
    let elapsedSec: Double
    /// Cumulative self-play moves at `timestamp`. Stored per
    /// sample so the next tick can subtract from "the sample that
    /// was 3 minutes ago" to get a windowed delta without needing
    /// a parallel cumulative-counters buffer.
    let selfPlayCumulativeMoves: Int
    /// Cumulative training-positions at `timestamp` — training
    /// steps × batch size. Same reason to store per-sample as
    /// `selfPlayCumulativeMoves`.
    let trainingCumulativeMoves: Int
    /// Rolling self-play moves/hr over the last 3 minutes. Before
    /// the session has 3 minutes of data, the window covers
    /// "everything so far" and this degrades gracefully to the
    /// lifetime rate; after 3 minutes it's strictly a 3-minute
    /// trailing average.
    let selfPlayMovesPerHour: Double
    /// Rolling training moves/hr over the same window.
    let trainingMovesPerHour: Double
    /// Sum of `selfPlayMovesPerHour` and `trainingMovesPerHour`.
    /// Derived rather than stored so changes to the definition of
    /// "combined" only have to happen in one place.
    var combinedMovesPerHour: Double {
        selfPlayMovesPerHour + trainingMovesPerHour
    }
}
