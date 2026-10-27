@testable import DrewsChessMachine

extension LichessBotRateLimit.Policy {
    /// The policy the gate tests run under: the breaker at two 429s in an
    /// hour (the default), no cooldown margin, so cooldowns are exactly what
    /// Lichess asked for.
    static let testPolicy = LichessBotRateLimit.Policy(breakerCount: 2, breakerWindow: .seconds(3600), cooldownMargin: .zero)
}
