import SwiftUI

/// Tones for incoming challenges, split by bot and human challengers.
struct LichessBotAlertSettingsSection: View {
    @Binding var settings: LichessBotAlertSettings
    @State private var soundNames: [String] = []
    @State private var problem: String?

    var body: some View {
        Section("Alerts — applies to the next challenge") {
            LichessBotAlertSoundRow(label: "Challenge from a bot", soundNames: soundNames, selection: $settings.botChallengeSoundName, problem: $problem)
            LichessBotAlertSoundRow(label: "Challenge from a human", soundNames: soundNames, selection: $settings.humanChallengeSoundName, problem: $problem)
            Text(problem ?? "")
                .foregroundStyle(.red)
                .shown(problem != nil)
        }
        .task {
            do {
                soundNames = try LichessBotSystemSounds.availableNames()
            } catch {
                problem = "Could not list system sounds: \(error.localizedDescription)"
            }
        }
    }
}
