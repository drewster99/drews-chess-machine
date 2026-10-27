import SwiftUI

/// The circuit breakers (owner decisions 2026-10-26). A trip stops new games
/// only: games in progress always finish.
struct LichessBotBreakerSettingsSection: View {
    @Binding var settings: LichessBotBreakerSettings
    @Binding var connection: LichessBotConnectionSettings

    var body: some View {
        Section("Rate limits (HTTP 429)") {
            LichessBotIntegerField(label: "Added to every cooldown", value: $connection.rateLimitCooldownMarginSeconds, unit: "s")
            LichessBotIntegerField(label: "No new games after a cooldown for", value: $connection.postRateLimitDrainMinutes, unit: "min")
            LichessBotBreakerSettingsRow(
                title: "Rate-limit breaker", countUnit: "429s",
                count: $settings.rateLimit.count,
                windowLabel: "Within", windowUnit: "min", window: $settings.rateLimit.windowMinutes,
                resumesAutomatically: $settings.rateLimit.resumesAutomatically,
                resumeAfterMinutes: $settings.rateLimit.resumeAfterMinutes)
        }
        Section("Circuit breakers — stop new games; games in progress finish") {
            LichessBotBreakerSettingsRow(
                title: "Lost on time", countUnit: "losses",
                count: $settings.lostOnTime.count,
                windowLabel: "Among the last", windowUnit: "games", window: $settings.lostOnTime.windowGames,
                resumesAutomatically: $settings.lostOnTime.resumesAutomatically,
                resumeAfterMinutes: $settings.lostOnTime.resumeAfterMinutes)
            LichessBotBreakerSettingsRow(
                title: "Move posts with no answer", countUnit: "posts",
                count: $settings.movePostFailures.count,
                windowLabel: "Within", windowUnit: "min", window: $settings.movePostFailures.windowMinutes,
                resumesAutomatically: $settings.movePostFailures.resumesAutomatically,
                resumeAfterMinutes: $settings.movePostFailures.resumeAfterMinutes)
            LichessBotBreakerSettingsRow(
                title: "Event-stream reconnects", countUnit: "times",
                count: $settings.reconnectStorm.count,
                windowLabel: "Within", windowUnit: "min", window: $settings.reconnectStorm.windowMinutes,
                resumesAutomatically: $settings.reconnectStorm.resumesAutomatically,
                resumeAfterMinutes: $settings.reconnectStorm.resumeAfterMinutes)
        }
    }
}
