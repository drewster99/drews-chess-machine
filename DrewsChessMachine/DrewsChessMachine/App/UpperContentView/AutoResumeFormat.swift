import Foundation

/// Pure-data formatters used by the Auto-Resume sheet's various
/// rows. Pulled into a shared namespace so both the per-section
/// View structs and the surrounding sheet body can call them
/// without duplicating logic.
enum AutoResumeFormat {
    /// Group-3 thousands separator using a cached static formatter.
    /// Unlike a one-off `NumberFormatter()` per call this avoids
    /// the non-trivial allocation cost on every sheet re-render
    /// while the countdown ticks.
    private static let groupedNumberFormatter: NumberFormatter = {
        let f = NumberFormatter()
        f.numberStyle = .decimal
        f.groupingSeparator = ","
        return f
    }()

    /// Compact-count formatter for the resume sheet. Below 1M the
    /// number renders with thousands separators (`12,926`); 1M
    /// and up abbreviates with one decimal place so close values
    /// rendering identically (e.g. two ~4.1M counters from the
    /// replay-ratio controller doing its job) is the user's signal
    /// to look at the underlying log for exact figures.
    static func count(_ value: Int) -> String {
        let absVal = abs(value)
        if absVal >= 1_000_000_000 {
            let scaled = Double(value) / 1_000_000_000.0
            return String(format: "%.1fB", scaled)
        }
        if absVal >= 1_000_000 {
            let scaled = Double(value) / 1_000_000.0
            return String(format: "%.1fM", scaled)
        }
        return groupedNumberFormatter.string(from: NSNumber(value: value))
            ?? String(value)
    }

    /// Active-training duration as `Hh Mm` (or `Mm Ss` for short
    /// runs). Uses `elapsedTrainingSec` from the session, which
    /// already excludes idle stretches per the field's doc-comment.
    static func activeDuration(_ seconds: Double) -> String {
        let total = max(0, Int(seconds.rounded()))
        let hours = total / 3600
        let minutes = (total % 3600) / 60
        let secs = total % 60
        if hours > 0 {
            return "\(hours)h \(minutes)m"
        }
        if minutes > 0 {
            return "\(minutes)m \(secs)s"
        }
        return "\(secs)s"
    }

    /// Human-friendly "N minutes ago" string for the sheet body.
    /// Kept deliberately simple — not worth dragging in
    /// RelativeDateTimeFormatter for a single-use string whose
    /// units only need to span seconds through days.
    static func relativeAgo(savedAtUnix: Int64) -> String {
        let deltaSec = max(0, Int(Date().timeIntervalSince1970) - Int(savedAtUnix))
        if deltaSec < 60 {
            return "\(deltaSec) second\(deltaSec == 1 ? "" : "s") ago"
        }
        let minutes = deltaSec / 60
        if minutes < 60 {
            return "\(minutes) minute\(minutes == 1 ? "" : "s") ago"
        }
        let hours = minutes / 60
        if hours < 24 {
            return "\(hours) hour\(hours == 1 ? "" : "s") ago"
        }
        let days = hours / 24
        return "\(days) day\(days == 1 ? "" : "s") ago"
    }

    /// "Started Apr 30, 2026 at 8:00 AM (10h ago)" line. Rendered
    /// only when the session.json peek succeeded — without the
    /// session-start timestamp the line would be empty noise.
    static func startedLine(sessionStartUnix: Int64) -> String {
        let date = Date(timeIntervalSince1970: TimeInterval(sessionStartUnix))
        let f = DateFormatter()
        f.dateStyle = .medium
        f.timeStyle = .short
        let absolute = f.string(from: date)
        let ago = relativeAgo(savedAtUnix: sessionStartUnix)
        return "Started \(absolute) (\(ago))"
    }
}
