import SwiftUI

/// Top-bar chip showing the live HH:MM:SS countdown to the next
/// auto-arena. Tapping opens the supplied popover content (the
/// Arena settings editor). The label updates on its own 1 Hz
/// `TimelineView` schedule so the rest of the status bar isn't
/// forced to invalidate every second.
struct ArenaCountdownChip<PopoverContent: View>: View {
    let isArenaRunning: Bool
    /// `at` is the date the parent's TimelineView ticked at;
    /// returning the formatted countdown text lets the parent keep
    /// its single source of truth for "what should this chip read
    /// right now."
    let countdownText: (Date) -> String
    @Binding var showPopover: Bool
    @ViewBuilder var popoverContent: () -> PopoverContent

    var body: some View {
        Button {
            if !isArenaRunning { showPopover.toggle() }
        } label: {
            HStack(spacing: 6) {
                Image(systemName: "flag.checkered")
                TimelineView(.periodic(from: .now, by: 1.0)) { context in
                    Text(countdownText(context.date))
                        .font(.callout)
                        .monospacedDigit()
                }
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 4)
            .background(
                RoundedRectangle(cornerRadius: 8)
                    .fill(Color.secondary.opacity(0.12))
            )
        }
        .buttonStyle(.plain)
        .disabled(isArenaRunning)
        .popover(isPresented: $showPopover, arrowEdge: .top) {
            popoverContent()
        }
    }
}
