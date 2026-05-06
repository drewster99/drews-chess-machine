import SwiftUI

/// One row of the Arena popover Options section. Centralized so
/// the three fields read identically: aligned label on the left,
/// fixed-width text field, optional hint on the right, and an
/// inline red rounded-rect overlay when the field's parse failed
/// on the last Save attempt.
struct ArenaPopoverField: View {
    let label: String
    @Binding var text: String
    let error: Bool
    let placeholder: String
    let width: CGFloat
    var hint: String? = nil

    var body: some View {
        HStack(spacing: 8) {
            Text(label)
                .frame(width: 110, alignment: .trailing)
            // Use ParameterTextField. macOS bare `TextField(text:)`
            // doesn't commit the binding until Return or focus loss;
            // clicking Save with a still-focused field would lose
            // any typed value. ParameterTextField's internal
            // `@FocusState` + `onChange(of: isFocused)` ensures the
            // binding is current by the time `arenaPopoverSave()`
            // reads it. The onCommit closure is a no-op because
            // parse/validate/apply runs transactionally on Save.
            ParameterTextField(
                placeholder: placeholder,
                text: $text,
                width: width
            ) { _ in /* no-op: Save handler does parse + apply */ }
                .overlay(
                    RoundedRectangle(cornerRadius: 6)
                        .stroke(Color.red, lineWidth: error ? 2 : 0)
                )
            if let hint {
                Text(hint)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer()
        }
    }
}
