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
            TextField(placeholder, text: $text)
                .textFieldStyle(.roundedBorder)
                .frame(width: width)
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
