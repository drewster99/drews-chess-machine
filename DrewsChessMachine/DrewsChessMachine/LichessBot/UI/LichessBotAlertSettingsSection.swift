import SwiftUI

/// Tones for incoming challenges, split by bot and human challengers, and
/// for new human games from an accepted challenge, split by who challenged
/// (`LichessBotGameStartAlert`).
struct LichessBotAlertSettingsSection: View {
    @Binding var settings: LichessBotAlertSettings
    @State private var soundNames: [String] = []
    @State private var problem: String?

    var body: some View {
        Section("Alerts — applies to the next challenge or game") {
            LichessBotAlertSoundRow(label: "Challenge from a bot", soundNames: soundNames, selection: $settings.botChallengeSoundName, problem: $problem)
            LichessBotAlertSoundRow(label: "Challenge from a human", soundNames: soundNames, selection: $settings.humanChallengeSoundName, problem: $problem)
            LichessBotRepeatingAlertSoundRow(label: "Human accepted our challenge", soundNames: soundNames, sound: $settings.humanAcceptedOurChallengeSound, problem: $problem)
            LichessBotRepeatingAlertSoundRow(label: "We accepted a human's challenge", soundNames: soundNames, sound: $settings.weAcceptedHumanChallengeSound, problem: $problem)
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
