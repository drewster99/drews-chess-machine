import SwiftUI

/// Standalone-window root view for the Lichess Probe Monitor.
/// Eight rows, one per theme bucket in the 200-puzzle Lichess set —
/// `mateIn1`, `hangingPiece`, `fork`, `pin`, `skewer`, `opening`,
/// `middlegame`, `endgame` — each showing the latest argmax-correct
/// fraction, top-5-correct fraction, a small spark line, and the error
/// count for that tick.
///
/// First-tick UI: rows render with "—" cells and a flat stub spark
/// until the initial 200-probe batch lands (a few seconds at the
/// network's serial forward-pass rate). After that, each tick appends
/// one aggregate per theme and the spark redraws.
struct LichessProbeMonitorView: View {
    @Bindable var history: LichessProbeHistory
    let onProbeNow: @MainActor () -> Void

    init(
        history: LichessProbeHistory,
        onProbeNow: @escaping @MainActor () -> Void = {}
    ) {
        self.history = history
        self.onProbeNow = onProbeNow
    }

    /// Stable display order — tactical themes first, phase themes
    /// second, alphabetical inside each group. Matches the priority
    /// order the curation script uses to assign puzzles to buckets.
    private let themeOrder: [ProbeCategory] = [
        .lichessMateIn1,
        .lichessHangingPiece,
        .lichessFork,
        .lichessPin,
        .lichessSkewer,
        .lichessOpening,
        .lichessMiddlegame,
        .lichessEndgame
    ]

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
                    ForEach(themeOrder, id: \.self) { theme in
                        rowOrPlaceholder(theme: theme)
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
            Text("Lichess Probe Monitor — 200 puzzles")
                .font(.system(.title2).weight(.semibold))
            Spacer()
            Button("Clear history") {
                history.clearAll()
            }
            .controlSize(.small)
            .disabled(history.entries.isEmpty)
            Text(totalTicksString)
                .font(.system(.caption, design: .monospaced))
                .foregroundStyle(.secondary)
            Text(totalScoreString)
                .font(.system(.caption, design: .monospaced).weight(.semibold))
                .foregroundStyle(.primary)
            Button("Probe now") {
                onProbeNow()
            }
            .controlSize(.small)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
    }

    private var totalTicksString: String {
        "ticks: \(history.tickCount) / cap \(history.maxEntriesPerTheme)"
    }

    private var totalScoreString: String {
        guard let correct = history.totalArgmaxCorrect,
              let total = history.totalLatestProbes, total > 0 else {
            return "score: — / —"
        }
        let pct = 100.0 * Double(correct) / Double(total)
        return String(format: "score: %d / %d (%.1f%%)", correct, total, pct)
    }

    // MARK: Column header row

    @ViewBuilder
    private var headerRow: some View {
        HStack(spacing: 8) {
            Text("THEME")
                .frame(width: 160, alignment: .leading)
            Text("ARGMAX")
                .frame(width: 100, alignment: .trailing)
            Text("ARGMAX %")
                .frame(width: 80, alignment: .trailing)
            Text("TOP-5 %")
                .frame(width: 80, alignment: .trailing)
            Text("trend (argmax %)")
                .frame(width: 180, alignment: .leading)
            Text("ERR")
                .frame(width: 40, alignment: .trailing)
        }
        .font(.system(.caption2, design: .monospaced).weight(.semibold))
        .foregroundStyle(.secondary)
    }

    // MARK: Per-theme row dispatcher

    @ViewBuilder
    private func rowOrPlaceholder(theme: ProbeCategory) -> some View {
        if let pair = history.latestPair(theme) {
            row(theme: theme, current: pair.current, previous: pair.previous)
        } else {
            placeholderRow(theme: theme)
        }
    }

    @ViewBuilder
    private func row(
        theme: ProbeCategory,
        current: LichessProbeHistory.Entry,
        previous: LichessProbeHistory.Entry?
    ) -> some View {
        let agg = current.aggregate
        let argmaxStr = "\(agg.argmaxCorrect) / \(agg.total)"
        let argmaxPct = String(format: "%.1f%%", 100.0 * Double(agg.argmaxCorrectFraction))
        let top5Pct = String(format: "%.1f%%", 100.0 * Double(agg.top5CorrectFraction))
        let series = history.argmaxFractionSeries(theme)
        let trendColor = deltaColor(
            current: agg.argmaxCorrectFraction,
            previous: previous?.aggregate.argmaxCorrectFraction
        )

        HStack(spacing: 8) {
            Text(themeLabel(theme))
                .font(.system(.body))
                .frame(width: 160, alignment: .leading)
            Text(argmaxStr)
                .font(.system(.body, design: .monospaced))
                .foregroundStyle(trendColor)
                .frame(width: 100, alignment: .trailing)
            Text(argmaxPct)
                .font(.system(.body, design: .monospaced))
                .foregroundStyle(trendColor)
                .frame(width: 80, alignment: .trailing)
            Text(top5Pct)
                .font(.system(.body, design: .monospaced))
                .foregroundStyle(.secondary)
                .frame(width: 80, alignment: .trailing)
            TacticalProbeSparkView(values: series, stroke: trendColor)
                .frame(width: 180, height: 24)
            errorCell(count: agg.errored)
                .frame(width: 40, alignment: .trailing)
        }
        .padding(.vertical, 4)
    }

    @ViewBuilder
    private func placeholderRow(theme: ProbeCategory) -> some View {
        HStack(spacing: 8) {
            Text(themeLabel(theme))
                .font(.system(.body))
                .foregroundStyle(.secondary)
                .frame(width: 160, alignment: .leading)
            Text("—")
                .font(.system(.body, design: .monospaced))
                .foregroundStyle(.secondary)
                .frame(width: 100, alignment: .trailing)
            Text("—")
                .font(.system(.body, design: .monospaced))
                .foregroundStyle(.secondary)
                .frame(width: 80, alignment: .trailing)
            Text("—")
                .font(.system(.body, design: .monospaced))
                .foregroundStyle(.secondary)
                .frame(width: 80, alignment: .trailing)
            TacticalProbeSparkView(values: [], stroke: .secondary)
                .frame(width: 180, height: 24)
            Text("—")
                .font(.system(.caption, design: .monospaced))
                .foregroundStyle(.secondary)
                .frame(width: 40, alignment: .trailing)
        }
        .padding(.vertical, 4)
    }

    // MARK: Footer

    @ViewBuilder
    private var footer: some View {
        VStack(alignment: .leading, spacing: 2) {
            Divider()
            HStack {
                Text(
                    "Each row aggregates 25 Lichess puzzles in that theme bucket. "
                    + "ARGMAX = network's #1 legal move matches the bookmove. "
                    + "TOP-5 = bookmove is among the network's top-5. "
                    + "Green/red = direction since prior tick."
                )
                .font(.caption)
                .foregroundStyle(.secondary)
                Spacer()
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 6)
        }
    }

    // MARK: Helpers

    private func themeLabel(_ theme: ProbeCategory) -> String {
        switch theme {
        case .lichessMateIn1:      return "Mate in 1"
        case .lichessHangingPiece: return "Hanging piece"
        case .lichessFork:         return "Fork"
        case .lichessPin:          return "Pin"
        case .lichessSkewer:       return "Skewer"
        case .lichessOpening:      return "Opening"
        case .lichessMiddlegame:   return "Middlegame"
        case .lichessEndgame:      return "Endgame"
        default:                   return theme.rawValue
        }
    }

    private func deltaColor(current: Float, previous: Float?) -> Color {
        guard let prev = previous else { return .primary }
        if current > prev + 1e-6 { return .green }
        if current < prev - 1e-6 { return .red }
        return .primary
    }

    /// Error-count cell. Single `Text` so view identity is stable across
    /// ticks; the shape style is type-erased through `AnyShapeStyle` so
    /// the ternary can switch between `.secondary` (a
    /// `HierarchicalShapeStyle`) and `Color.red` without Swift's
    /// overload resolution complaining.
    private func errorCell(count: Int) -> some View {
        let style: AnyShapeStyle = count == 0
            ? AnyShapeStyle(HierarchicalShapeStyle.secondary)
            : AnyShapeStyle(Color.red)
        return Text(count == 0 ? "—" : "\(count)")
            .font(.system(.caption, design: .monospaced))
            .foregroundStyle(style)
    }
}
