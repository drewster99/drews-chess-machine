import SwiftUI

/// One circuit breaker's settings: when it trips, and whether it resumes by
/// itself (`LICHESS_BOT_CIRCUIT_BREAKERS_PLAN.md`).
struct LichessBotBreakerSettingsRow: View {
    let title: String
    let countUnit: String
    @Binding var count: Int
    let windowLabel: String
    let windowUnit: String
    @Binding var window: Int
    @Binding var resumesAutomatically: Bool
    @Binding var resumeAfterMinutes: Int

    var body: some View {
        LichessBotIntegerField(label: "\(title): trips at", value: $count, unit: countUnit)
        LichessBotIntegerField(label: windowLabel, value: $window, unit: windowUnit)
        Toggle("Resume new games by itself (off: finish games, then go offline until you go online)", isOn: $resumesAutomatically)
        LichessBotIntegerField(label: "Resume after", value: $resumeAfterMinutes, unit: "min")
            .disabled(!resumesAutomatically)
    }
}
