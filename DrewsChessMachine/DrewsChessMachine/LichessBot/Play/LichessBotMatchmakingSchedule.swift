import Foundation

/// Matchmaking's hourly cap by time of day: the full
/// `maxChallengesPerHour` inside the peak window, `offPeakRatePercent` of it
/// outside (`LichessBotMatchmakingSettings.peakStartHour` explains the
/// defaults). Pure; the caller passes the time and calendar.
///
/// Bots refuse DCM's challenges far more often in the evening, once many
/// have played their 100 bot games for the day, and those caps clear
/// overnight. Scaling the one hourly cap the rate limiter already enforces
/// (`LichessBotMatchmakingRateLimiter`) is the whole mechanism: the
/// spacing, slot filling and Fill Open Slots all stop at that cap.
///
/// Hours are read with `Calendar.component(.hour, from:)`, the wall-clock
/// hour in the calendar's time zone, so a DST change moves the window with
/// local time: on the fall-back night the repeated 01:00 hour is the same
/// hour twice, and on spring-forward night a window starting at the missing
/// 02:00 starts when the clock reads 03:00.
enum LichessBotMatchmakingSchedule {

    struct Rate: Sendable, Equatable {
        /// The cap on challenges per rolling hour now.
        let hourlyCap: Int
        let isPeak: Bool
        /// The percentage of `maxChallengesPerHour` in force: 100 at peak.
        let percent: Int
        /// When the window next opens or closes; nil when every hour is
        /// peak (equal start and end hours).
        let changesAt: Date?

        /// For the matchmaking status line: "peak 10/h" or "off-peak 40%: 4/h".
        var text: String {
            isPeak ? "peak \(hourlyCap)/h" : "off-peak \(percent)%: \(hourlyCap)/h"
        }
    }

    /// Whether `hour` (0–23, local) lies in the window from `start` up to,
    /// not including, `end`, wrapping past midnight when `end` < `start`.
    /// Equal hours make every hour peak.
    static func isPeak(hour: Int, start: Int, end: Int) -> Bool {
        if start == end { return true }
        if start < end { return hour >= start && hour < end }
        return hour >= start || hour < end
    }

    /// `maxChallengesPerHour` scaled by `percent`, to the nearest whole
    /// challenge (halves up), and at least one while `percent` is above
    /// zero: only a 0% rate means "send nothing off-peak", so a small peak
    /// cap (1 or 2 an hour) still sends off-peak at a reduced rate.
    static func scaledCap(_ maxChallengesPerHour: Int, percent: Int) -> Int {
        guard percent > 0 else { return 0 }
        return max(1, (maxChallengesPerHour * percent + 50) / 100)
    }

    static func rate(settings: LichessBotMatchmakingSettings, now: Date, calendar: Calendar = .current) -> Rate {
        let hour = calendar.component(.hour, from: now)
        let peak = isPeak(hour: hour, start: settings.peakStartHour, end: settings.peakEndHour)
        let changesAt: Date?
        if settings.peakStartHour == settings.peakEndHour {
            changesAt = nil
        } else {
            let nextHour = peak ? settings.peakEndHour : settings.peakStartHour
            // `.nextTime`: a boundary hour skipped by DST takes the next
            // existing time.
            changesAt = calendar.nextDate(after: now, matching: DateComponents(hour: nextHour, minute: 0, second: 0), matchingPolicy: .nextTime)
        }
        let percent = peak ? 100 : settings.offPeakRatePercent
        return Rate(hourlyCap: scaledCap(settings.maxChallengesPerHour, percent: percent), isPeak: peak, percent: percent, changesAt: changesAt)
    }
}
