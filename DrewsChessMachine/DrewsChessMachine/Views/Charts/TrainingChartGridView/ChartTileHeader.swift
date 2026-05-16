import SwiftUI

/// Compact tile header — title on the left, value on the right.
/// When `titleHelp` is non-nil, the title becomes a small button
/// that opens a popover with the supplied description; the visual
/// affordance is a dotted underline. Mirrors the same pattern in
/// `FastLineChart` so migrated and still-on-Swift-Charts tiles
/// behave identically.
struct ChartTileHeader: View {
    let title: String
    let value: String
    var titleHelp: AttributedString? = nil

    @State private var showingTitleHelp = false

    var body: some View {
        HStack(spacing: 4) {
            titleLabel
            Spacer()
            Text(value)
                .font(.caption2)
                .monospacedDigit()
                .foregroundStyle(.primary)
        }
    }

    @ViewBuilder
    private var titleLabel: some View {
        if let titleHelp {
            Button(action: { showingTitleHelp = true }) {
                Text(title)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .underline(true, pattern: .dot)
            }
            .buttonStyle(.plain)
            .help("Click for description")
            .popover(isPresented: $showingTitleHelp, arrowEdge: .top) {
                Text(titleHelp)
                    .font(.caption)
                    .textSelection(.enabled)
                    .padding(10)
                    .frame(maxWidth: 320, alignment: .leading)
            }
        } else {
            Text(title)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .lineLimit(1)
        }
    }
}
