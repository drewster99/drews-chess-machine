import SwiftUI

/// Stored training preferences found unusable at launch (wrong type, or
/// outside the parameter's declared range). The app runs on each one's
/// default meanwhile; the stored value stays as found until the user resets
/// it here.
struct InvalidStoredSettingsSheet: View {
    @Bindable var trainingParams: TrainingParameters
    let onClose: () -> Void
    @State private var resetError: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Some saved settings can't be used")
                .font(.title3.weight(.semibold))
            Text("These stored values are the wrong type or outside the range the app accepts. Until you reset one, the app uses the value shown under Reset to. Nothing has been changed on disk.")
                .font(.callout)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            List(trainingParams.invalidStoredSettings) { setting in
                HStack(alignment: .top) {
                    InvalidSettingRow(setting: setting, replacementLabel: "Reset to")
                    Spacer()
                    Button("Reset") {
                        do {
                            try trainingParams.resetInvalidStoredSetting(id: setting.id)
                            resetError = nil
                        } catch {
                            resetError = "Couldn't reset \(setting.id): \(error)"
                        }
                    }
                }
            }
            .frame(minHeight: 160)
            Text(resetError ?? " ")
                .font(.callout)
                .foregroundStyle(.red)
                .opacity(resetError == nil ? 0 : 1)
            HStack {
                Button("Reset All") {
                    for setting in trainingParams.invalidStoredSettings {
                        do {
                            try trainingParams.resetInvalidStoredSetting(id: setting.id)
                        } catch {
                            resetError = "Couldn't reset \(setting.id): \(error)"
                            return
                        }
                    }
                    resetError = nil
                }
                .disabled(trainingParams.invalidStoredSettings.isEmpty)
                Spacer()
                Button("Close", action: onClose)
                    .keyboardShortcut(.defaultAction)
            }
        }
        .padding(20)
        .frame(minWidth: 560, minHeight: 360)
    }
}
