import SwiftUI

/// Standalone-window root view for the Tactical Probe Monitor.
/// Shows one row per probe in `TacticalProbeData.standardSet`,
/// updating every 15s via the owning window controller's watcher.
///
/// First-tick UI: rows render with "—" cells and a flat stub spark
/// until the initial probe batch lands (typically within a few
/// hundred ms of opening). After that, each tick appends one sample
/// per probe and the spark redraws.
///
/// Reads from `TacticalProbeHistory` via `@Bindable` so SwiftUI
/// re-renders the affected rows when the watcher appends. The history
/// store is owned by the window controller; this view doesn't create
/// or mutate it.
struct TacticalProbeMonitorView: View {
    @Bindable var history: TacticalProbeHistory

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            header
            Divider()
            ScrollView(.vertical, showsIndicators: true) {
                LazyVStack(alignment: .leading, spacing: 0) {
                    headerRow
                        .padding(.vertical, 4)
                        .padding(.horizontal, 12)
                        .background(Color(NSColor.controlBackgroundColor).opacity(0.5))
                    Divider()
                    ForEach(TacticalProbeData.standardSet, id: \.name) { probe in
                        rowOrPlaceholder(probe: probe)
                            .padding(.horizontal, 12)
                        Divider()
                    }
                }
            }
            footer
        }
    }

    // MARK: Header band

    @ViewBuilder
    private var header: some View {
        HStack(spacing: 16) {
            Text("Tactical Probe Monitor")
                .font(.system(.title2).weight(.semibold))
            Spacer()
            Text(totalTicksString)
                .font(.system(.caption, design: .monospaced))
                .foregroundStyle(.secondary)
            Button("Clear history") {
                history.clearAll()
            }
            .controlSize(.small)
            .disabled(history.entries.isEmpty)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
    }

    private var totalTicksString: String {
        // Max series length across all probes is the tick count
        // (each tick appends one entry per probe). Using max rather
        // than first-probe handles the transient case where one
        // probe failed and hasn't been recorded yet.
        let n = history.entries.values.map(\.count).max() ?? 0
        return "ticks: \(n) / cap \(history.maxEntriesPerProbe)"
    }

    // MARK: Column header row

    @ViewBuilder
    private var headerRow: some View {
        HStack(spacing: 8) {
            Text("VERDICT")
                .frame(width: 100, alignment: .center)
            Text("PROBE")
                .frame(width: 130, alignment: .leading)
            Text("ACTUAL")
                .frame(width: 80, alignment: .leading)
            Text("EXPECTED")
                .frame(width: 80, alignment: .leading)
            metricHeader(label: "PROB", valueWidth: 56)
            metricHeader(label: "RANK", valueWidth: 38)
            metricHeader(label: "ENTROPY %", valueWidth: 56)
            Text("W / D / L")
                .frame(width: 120, alignment: .trailing)
        }
        .font(.system(.caption2, design: .monospaced).weight(.semibold))
        .foregroundStyle(.secondary)
    }

    @ViewBuilder
    private func metricHeader(label: String, valueWidth: CGFloat) -> some View {
        HStack(spacing: 4) {
            Text(label)
                .frame(width: valueWidth, alignment: .trailing)
            Spacer().frame(width: 10)              // arrow column placeholder
            Text("trend")
                .frame(width: 80, alignment: .leading)
                .foregroundStyle(.secondary.opacity(0.6))
        }
    }

    // MARK: Per-probe row dispatcher

    @ViewBuilder
    private func rowOrPlaceholder(probe: TacticalProbe) -> some View {
        if let pair = history.latestPair(probe.name) {
            TacticalProbeRowView(
                probeName: probe.name,
                current: pair.current,
                previous: pair.previous,
                probSeries: history.sparkSeries(probe.name, metric: .expectedProb),
                rankSeries: history.sparkSeries(probe.name, metric: .expectedRank),
                entropyPctSeries: history.sparkSeries(probe.name, metric: .entropyPercent)
            )
        } else {
            // No tick recorded yet for this probe — render the row
            // with neutral placeholders. Keeps the column alignment
            // intact and prevents flicker once the first tick lands.
            placeholderRow(probe: probe)
        }
    }

    @ViewBuilder
    private func placeholderRow(probe: TacticalProbe) -> some View {
        HStack(spacing: 8) {
            Text("— ")
                .font(.system(.caption2, design: .monospaced))
                .foregroundStyle(.secondary)
                .frame(width: 100, alignment: .center)
            Text(probe.shortDescription)
                .font(.system(.body))
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .frame(width: 130, alignment: .leading)
            Text("—")
                .font(.system(.body, design: .monospaced))
                .foregroundStyle(.secondary)
                .frame(width: 80, alignment: .leading)
            Text(probe.acceptable.sorted(by: { $0.notation < $1.notation }).first?.notation ?? "—")
                .font(.system(.body, design: .monospaced))
                .foregroundStyle(.secondary)
                .frame(width: 80, alignment: .leading)
            placeholderCell(valueWidth: 56)   // prob
            placeholderCell(valueWidth: 38)   // rank
            placeholderCell(valueWidth: 56)   // entropy %
            Text("—")
                .font(.system(.caption, design: .monospaced))
                .foregroundStyle(.secondary)
                .frame(width: 120, alignment: .trailing)
        }
        .padding(.vertical, 2)
    }

    @ViewBuilder
    private func placeholderCell(valueWidth: CGFloat) -> some View {
        HStack(spacing: 4) {
            Text("—")
                .font(.system(.body, design: .monospaced))
                .foregroundStyle(.secondary)
                .frame(width: valueWidth, alignment: .trailing)
            Spacer().frame(width: 10)
            TacticalProbeSparkView(values: [], stroke: .secondary)
                .frame(width: 80)
        }
    }

    // MARK: Footer

    @ViewBuilder
    private var footer: some View {
        VStack(alignment: .leading, spacing: 2) {
            Divider()
            HStack {
                Text("Tap any row to view the position. Color on metrics = direction since prior tick: green = value up, red = value down, gray = first tick or unchanged. ACTUAL move is green when it matches an expected move.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Spacer()
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 6)
        }
    }
}
