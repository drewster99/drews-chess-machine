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
            }
            .buttonStyle(.plain)
            .help("Click for description")
            .popover(isPresented: $showingTitleHelp, arrowEdge: .top) {
                // `.fixedSize(horizontal: false, vertical: true)` is
                // load-bearing — without it the popover container
                // sizes to the Text's natural single-line intrinsic
                // width and then truncates with an ellipsis. Pin
                // horizontal to the proposed 320 and let vertical
                // grow to whatever the wrapped block needs.
                Text(titleHelp)
                    .font(.caption)
                    .textSelection(.enabled)
                    .fixedSize(horizontal: false, vertical: true)
                    .frame(width: 320, alignment: .leading)
                    .padding(10)
            }
        } else {
            Text(title)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .lineLimit(1)
        }
    }
}
