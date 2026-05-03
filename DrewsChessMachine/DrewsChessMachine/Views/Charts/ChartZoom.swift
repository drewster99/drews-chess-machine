import Foundation

/// Discrete zoom stops for the training chart grid's horizontal
/// time window. A user can zoom in (narrower window = higher time
/// resolution) or out (wider window = more history in view) one
/// stop at a time via ⌘= / ⌘- keyboard shortcuts or the View menu.
///
/// The stops are not evenly spaced — they follow human-friendly
/// durations (quarter-hour, hour, day) so the label above the
/// chart grid reads "2h" or "7d" rather than a computed number
/// of seconds.
///
/// Zoom-in is unconstrained all the way to the tightest stop
/// (15 minutes). Zoom-out is capped at `autoIndex(forDataSec:) + 3`
/// — three stops past the tightest window that fully fits the
/// current data span. This prevents the common footgun of zooming
/// out to "30 days" and then staring at a near-empty chart with
/// all the interesting structure compressed into a pixel at the
/// right edge.
enum ChartZoom {

    /// Stops in seconds, ascending. Must stay sorted.
    static let stops: [TimeInterval] = [
        15 * 60,               // 15m
        30 * 60,               // 30m
        60 * 60,               // 1h
        2  * 3600,             // 2h
        4  * 3600,             // 4h
        8  * 3600,             // 8h
        12 * 3600,             // 12h
        24 * 3600,             // 24h (1d)
        2  * 86400,            // 2d
        3  * 86400,            // 3d
        4  * 86400,            // 4d
        5  * 86400,            // 5d
        6  * 86400,            // 6d
        7  * 86400,            // 7d
        10 * 86400,            // 10d
        20 * 86400,            // 20d
        30 * 86400             // 30d
    ]

    /// Labels in 1:1 correspondence with `stops`. Displayed as the
    /// bold indicator above the upper-left chart.
    static let labels: [String] = [
        "15m", "30m", "1h", "2h", "4h", "8h", "12h", "24h",
        "2d", "3d", "4d", "5d", "6d", "7d", "10d", "20d", "30d"
    ]

    /// Default zoom index for a new session. `1` = 30 minutes, which
    /// matches the pre-zoom `progressRateVisibleDomainSec` constant
    /// (1800s) exactly, so upgrading a session that predates the
    /// zoom feature doesn't visually jump on first render.
    static let defaultIndex: Int = 1

    /// Smallest stop index that fully contains `dataSec` — i.e. the
    /// stop the "Auto" button picks. With e.g. 29 minutes of data,
    /// that's index 1 (30m). With 31 minutes, index 2 (1h).
    ///
    /// When the data already exceeds the widest stop (30 days),
    /// Auto pins to the last index rather than throwing.
    static func autoIndex(forDataSec dataSec: Double) -> Int {
        let clampedData = max(0, dataSec)
        for (idx, stop) in stops.enumerated() where stop >= clampedData {
            return idx
        }
        return stops.count - 1
    }

    /// Widest zoom-out index the UI is allowed to reach given
    /// `dataSec` — three stops past the fitting Auto index. Clamped
    /// to the last index so you can't walk past 30d.
    static func maxZoomOutIndex(forDataSec dataSec: Double) -> Int {
        let fit = autoIndex(forDataSec: dataSec)
        return min(fit + 3, stops.count - 1)
    }

    /// Clamp an arbitrary `idx` into the legal range given current
    /// `dataSec`. Returns a value in `[0, maxZoomOutIndex(forDataSec:)]`.
    /// Useful when data has shrunk (e.g. after a Clear action) and
    /// a prior manual zoom-out is no longer allowed.
    static func clamp(_ idx: Int, forDataSec dataSec: Double) -> Int {
        let maxIdx = maxZoomOutIndex(forDataSec: dataSec)
        return min(max(idx, 0), maxIdx)
    }
}
