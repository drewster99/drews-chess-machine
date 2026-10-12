import Foundation

/// How the bot's model views write a time (issue #20,
/// documentation/plans-active/LICHESS_BOT_MODEL_IN_USE_PLAN.md "Times"):
/// one fixed-width local form, so file times line up in columns, and the
/// same form with its age for the model in use, where "how long ago" is
/// the question. Main-actor: the views are, and `DateFormatter` isn't
/// `Sendable`.
@MainActor
enum LichessBotModelTimeText {
    /// `date` as fixed-width local date and time ("2026-10-12 13:05").
    static func absolute(_ date: Date) -> String {
        dateFormatter.string(from: date)
    }

    /// `date` with its age at `now` ("2026-10-12 13:05 · 23 min ago"). A
    /// date after `now` (another Mac's clock wrote the file) shows no age
    /// rather than a negative one.
    static func withAge(_ date: Date, now: Date) -> String {
        guard let age = ageText(from: date, to: now) else {
            return absolute(date)
        }
        return "\(absolute(date)) · \(age)"
    }

    /// "<1 min ago", "23 min ago", "5 h 12 min ago", "3 d 4 h ago"; nil
    /// when `date` is after `now`.
    static func ageText(from date: Date, to now: Date) -> String? {
        let seconds = now.timeIntervalSince(date)
        guard seconds >= 0 else { return nil }
        let minutes = Int(seconds / 60)
        if minutes < 1 {
            return "<1 min ago"
        }
        if minutes < 60 {
            return "\(minutes) min ago"
        }
        let hours = minutes / 60
        if hours < 48 {
            return "\(hours) h \(minutes % 60) min ago"
        }
        return "\(hours / 24) d \(hours % 24) h ago"
    }

    private static let dateFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "yyyy-MM-dd HH:mm"
        return formatter
    }()
}
