import SwiftUI

/// One alert tone: a picker of system sounds (or none) and a preview button.
struct LichessBotAlertSoundRow: View {
    let label: String
    let soundNames: [String]
    @Binding var selection: String?
    @Binding var problem: String?

    var body: some View {
        HStack {
            Picker(label, selection: $selection) {
                Text("None").tag(String?.none)
                ForEach(soundNames, id: \.self) { name in
                    Text(name).tag(String?.some(name))
                }
            }
            Button("▶") {
                preview()
            }
            .disabled(selection == nil)
            .help("Play this sound")
            .accessibilityLabel("Preview \(label)")
        }
    }

    private func preview() {
        guard let selection else { return }
        do {
            try LichessBotSystemSounds.play(named: selection)
            problem = nil
        } catch {
            problem = error.localizedDescription
        }
    }
}
