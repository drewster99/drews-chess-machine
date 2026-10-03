import SwiftUI

/// Stored training preferences found unusable at launch (wrong type, or
/// outside the parameter's declared range). The app starts on each one's
/// default; the stored value stays as found until the user resets it here,
/// and a reset rewrites only the stored value
/// (`TrainingParameters.resetInvalidStoredSetting`), never the value the
/// running app is using.
struct InvalidStoredSettingsSheet: View {
    @Bindable var trainingParams: TrainingParameters
    let onClose: () -> Void
    @State private var resetError: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Some saved settings can't be used")
                .font(.title3.weight(.semibold))
            Text("These saved values are the wrong type or outside the range the app accepts, so the app started with the value shown under Reset to instead. Reset saves that value in place of the unusable one for future launches; it does not change the value this run is using. Nothing is saved until you reset.")
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
