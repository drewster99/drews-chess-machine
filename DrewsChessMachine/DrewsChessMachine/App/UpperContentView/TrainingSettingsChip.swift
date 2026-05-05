import SwiftUI

/// Top-bar chip that anchors the `TrainingSettingsPopover`. Mirrors
/// `ArenaCountdownChip`'s rounded-rect look so the right-hand cluster
/// of chips reads as one row, but doesn't carry a `TimelineView` —
/// there is no countdown to display, just a stable label that opens
/// the optimizer-knobs editor on click.
struct TrainingSettingsChip<PopoverContent: View>: View {
    @Binding var showPopover: Bool
    @ViewBuilder var popoverContent: () -> PopoverContent

    var body: some View {
        Button {
            showPopover.toggle()
        } label: {
            HStack(spacing: 6) {
                Image(systemName: "slider.horizontal.3")
                Text("Training")
                    .font(.callout)
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 4)
            .background(
                RoundedRectangle(cornerRadius: 8)
                    .fill(Color.secondary.opacity(0.12))
            )
        }
        .buttonStyle(.plain)
        .popover(isPresented: $showPopover, arrowEdge: .top) {
            popoverContent()
        }
    }
}
