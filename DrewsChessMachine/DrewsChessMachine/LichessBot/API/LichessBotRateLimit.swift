import Foundation

/// Lichess rate-limit rules, as pure functions (plan §5).
///
/// Lichess asks clients to make one request at a time and, after a 429, to
/// wait before retrying, longer for some limits. No numeric limits are
/// published, and they change, so the design treats not triggering a 429 as
/// a hard constraint and a 429, when it happens, as a full stop.
enum LichessBotRateLimit {
    /// The shortest cooldown after any 429, whatever `Retry-After` says.
    /// Deliberately a constant and not a setting: the floor follows Lichess's
    /// own guidance on how long to wait, and a shorter pause risks a longer
    /// lockout.
    static let minimumCooldown: Duration = .seconds(60)

    /// Cooldown to apply after a 429: the server's `Retry-After` when it asks
    /// for longer than the floor, otherwise the floor.
    static func cooldown(retryAfter: Duration?) -> Duration {
        guard let retryAfter, retryAfter > minimumCooldown else {
            return minimumCooldown
        }
        return retryAfter
    }

    /// Parse an HTTP `Retry-After` header, which is either delta-seconds
    /// (`"120"`) or an HTTP-date (`"Wed, 21 Oct 2015 07:28:00 GMT"`) (plan
    /// E27). Returns nil for an absent or unparseable header; a date in the
    /// past yields zero.
    static func parseRetryAfter(_ header: String?, now: Date) -> Duration? {
        guard let text = header?.trimmingCharacters(in: .whitespaces), !text.isEmpty else {
            return nil
        }
        if let seconds = Int(text) {
            return seconds >= 0 ? .seconds(seconds) : nil
        }
        // RFC 7231 IMF-fixdate, the HTTP-date form servers send. Built per
        // call: `DateFormatter` is not `Sendable`, and a 429 is rare enough
        // that the construction cost is irrelevant.
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = TimeZone(identifier: "GMT")
        formatter.dateFormat = "EEE, dd MMM yyyy HH:mm:ss zzz"
        guard let date = formatter.date(from: text) else {
            return nil
        }
        let delta = date.timeIntervalSince(now)
        return delta > 0 ? .seconds(delta) : .zero
    }
}
