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

    /// How the request gate answers request-rate 429s (owner decisions
    /// 2026-10-26, `LICHESS_BOT_CIRCUIT_BREAKERS_PLAN.md`): `breakerCount`
    /// of them inside `breakerWindow` trip the rate-limit breaker, and every
    /// cooldown adds `cooldownMargin` to what Lichess asked for.
    struct Policy: Sendable, Equatable {
        let breakerCount: Int
        let breakerWindow: Duration
        let cooldownMargin: Duration
    }

    /// Cooldown to apply after a 429: the longest wait Lichess named (its
    /// `Retry-After` header, or the body's `ratelimit.seconds`), never less
    /// than the floor, plus `margin`. Lichess asks for "a full minute"; the
    /// margin is DCM's own caution on top.
    static func cooldown(retryAfter: Duration?, bodyWait: Duration?, margin: Duration) -> Duration {
        let asked = [retryAfter, bodyWait].compactMap { $0 }.max() ?? .zero
        return max(asked, minimumCooldown) + margin
    }

    /// The wait a 429's body names, `{"ratelimit": {"seconds": N}}` (lila's
    /// `JsonLimited`, any limit key), or nil when the body has none: not
    /// JSON, no `ratelimit` object, or no non-negative number there.
    static func bodyWait(from body: Data) -> Duration? {
        guard !body.isEmpty else { return nil }
        let object: Any
        do {
            object = try JSONSerialization.jsonObject(with: body)
        } catch {
            // Not JSON (a proxy's page): it names no wait.
            return nil
        }
        guard let fields = object as? [String: Any],
              let rateLimit = fields["ratelimit"] as? [String: Any],
              let seconds = rateLimit["seconds"] as? NSNumber,
              seconds.doubleValue >= 0
        else {
            return nil
        }
        return .seconds(seconds.doubleValue)
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
