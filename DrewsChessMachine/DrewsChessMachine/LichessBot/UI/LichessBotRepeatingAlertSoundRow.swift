import SwiftUI

/// One repeating alert: a picker of system sounds (or none), how many times
/// it plays, and a preview that plays it exactly as an alert would — the
/// same player, the same repeats one after another.
struct LichessBotRepeatingAlertSoundRow: View {
    let label: String
    let soundNames: [String]
    @Binding var sound: LichessBotRepeatingAlertSound
    @Binding var problem: String?
    /// The preview's own player, so a preview never queues behind (or cuts
    /// off) a live game's alert.
    @State private var previewPlayer = LichessBotRepeatingSoundPlayer()

    var body: some View {
        HStack {
            Picker(label, selection: $sound.soundName) {
                Text("None").tag(String?.none)
                ForEach(soundNames, id: \.self) { name in
                    Text(name).tag(String?.some(name))
                }
            }
            Stepper(value: $sound.repeatCount, in: LichessBotLimits.alertSoundRepeatCountRange) {
                Text("× \(sound.repeatCount)")
                    .monospacedDigit()
                    .frame(minWidth: 32, alignment: .trailing)
            }
            .disabled(sound.soundName == nil)
            .help("How many times the sound plays, one after another")
            .accessibilityLabel("\(label) repeats")
            .accessibilityValue("\(sound.repeatCount)")
            Button("▶") {
                preview()
            }
            .disabled(sound.soundName == nil)
            .help("Play this sound as the alert would")
            .accessibilityLabel("Preview \(label)")
        }
    }

    /// Plays the chosen sound its chosen number of times, replacing any
    /// preview still playing.
    private func preview() {
        guard let soundName = sound.soundName else { return }
        previewPlayer.stopAll()
        let reportProblem = $problem
        do {
            try previewPlayer.enqueue(soundName: soundName, times: sound.repeatCount) { error in
                reportProblem.wrappedValue = error.localizedDescription
            }
            problem = nil
        } catch {
            problem = error.localizedDescription
        }
    }
}
