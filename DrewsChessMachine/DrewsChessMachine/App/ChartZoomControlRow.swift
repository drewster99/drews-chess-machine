import SwiftUI

/// Compact row — same font size and weight for every element so
/// it lays out as one tight cluster on the left rather than the
/// bold-zoom + tiny-hint + far-right-Auto layout it used to be.
struct ChartZoomControlRow: View {
    @Bindable var coordinator: ChartCoordinator

    var body: some View {
        HStack(spacing: 8) {
            Text(ChartZoom.labels[coordinator.chartZoomIdx])
                .font(.caption.bold())
                .monospacedDigit()
                .foregroundStyle(.secondary)
            Text("⌘=")
                .font(.caption)
                .foregroundStyle(coordinator.canZoomIn ? Color.secondary : Color.secondary.opacity(0.4))
            Text("⌘-")
                .font(.caption)
                .foregroundStyle(coordinator.canZoomOut ? Color.secondary : Color.secondary.opacity(0.4))
            Toggle(isOn: Binding(
                get: { coordinator.chartZoomAuto },
                set: { coordinator.setAutoZoom($0) }
            )) {
                Text("Auto")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            .toggleStyle(.checkbox)
            Spacer()
        }
        .padding(.horizontal, 4)
    }
}
