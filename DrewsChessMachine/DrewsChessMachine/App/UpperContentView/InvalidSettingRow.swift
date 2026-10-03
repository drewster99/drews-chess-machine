import SwiftUI

/// One unusable stored or saved setting: what it is, what was found, why it
/// cannot be used, and the valid value offered in its place. Shared by the
/// stored-preferences list and the session-resume review so both read alike.
struct InvalidSettingRow: View {
    let setting: InvalidStoredSetting
    /// Label for the offered value: "Reset to" for a stored preference,
    /// "Replace with" for a saved session value.
    let replacementLabel: String

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(setting.name)
                .font(.headline)
            Grid(alignment: .leadingFirstTextBaseline, horizontalSpacing: 8, verticalSpacing: 2) {
                GridRow {
                    Text("Found")
                        .foregroundStyle(.secondary)
                        .gridColumnAlignment(.trailing)
                    Text(setting.found)
                        .monospacedDigit()
                        .textSelection(.enabled)
                }
                GridRow {
                    Text("Problem")
                        .foregroundStyle(.secondary)
                    Text(setting.problem)
                        .fixedSize(horizontal: false, vertical: true)
                }
                GridRow {
                    Text(replacementLabel)
                        .foregroundStyle(.secondary)
                    Text(setting.replacement)
                        .monospacedDigit()
                        .textSelection(.enabled)
                }
            }
            .font(.callout)
            Text(setting.id)
                .font(.caption.monospaced())
                .foregroundStyle(.tertiary)
        }
        .padding(.vertical, 4)
    }
}
