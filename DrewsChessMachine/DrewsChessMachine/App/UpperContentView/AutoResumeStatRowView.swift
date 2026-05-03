import SwiftUI

/// One row in the Auto-Resume sheet's progress block: a fixed-width
/// secondary-color label on the left and a monospaced-digit value
/// on the right. Lets the rows align cleanly even when the values
/// have differing digit counts.
struct AutoResumeStatRowView: View {
    let label: String
    let value: String

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            Text(label)
                .font(.callout)
                .foregroundStyle(.secondary)
                .frame(width: 76, alignment: .leading)
            Text(value)
                .font(.callout)
                .monospacedDigit()
        }
    }
}
