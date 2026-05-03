import SwiftUI

/// Compact tile header — title on the left, value on the right.
/// The header layout is identical across every tile, so this one
/// view replaces a repetitive `HStack { Text(...).font(.caption2)... }`
/// pattern in each chart subview.
struct ChartTileHeader: View {
    let title: String
    let value: String

    var body: some View {
        HStack(spacing: 4) {
            Text(title)
                .font(.caption2)
                .foregroundStyle(.secondary)
            Spacer()
            Text(value)
                .font(.caption2)
                .monospacedDigit()
                .foregroundStyle(.primary)
        }
    }
}
