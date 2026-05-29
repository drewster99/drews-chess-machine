import SwiftUI

/// Standalone-window root view for the Lichess Probe **Detail** monitor.
/// Shows every one of the 200 puzzles from the latest watcher tick,
/// grouped into eight theme sections of 25 rows each. Each row carries
/// the original Lichess puzzle id, rating, expected (book) move, the
/// network's actual top-1 move, the masked-prob and rank of the
/// expected move, a verdict badge, and the value head's W/D/L.
///
/// Reads from `LichessProbeHistory.latestPerPuzzleResults` (replaced
/// wholesale each tick) plus `LichessProbeData.metadata` for the static
/// (id, rating, theme, FEN, bookmove UCI) fields. No own copy of the
/// data — fully derived from the observed history.
struct LichessProbeDetailView: View {
    @Bindable var history: LichessProbeHistory
    let onProbeNow: @MainActor () -> Void
    let onExport: @MainActor () -> Void

    init(
        history: LichessProbeHistory,
        onProbeNow: @escaping @MainActor () -> Void = {},
        onExport: @escaping @MainActor () -> Void = {}
    ) {
        self.history = history
        self.onProbeNow = onProbeNow
        self.onExport = onExport
    }

    /// Tap-to-toggle popover state must live on a row-scoped struct,
    /// not on `LichessProbeDetailView` itself — otherwise every row
    /// would share the same toggle and clicking one would open them
    /// all. Each row instantiates its own.
    fileprivate struct DetailRowView: View {
        let result: ProbeResult
        @State private var isShowingBoardPopover = false

        var body: some View {
            let meta = LichessProbeData.metadata[result.probe.name]
            let expectedNotation = result.probe.acceptable
                .sorted(by: { $0.notation < $1.notation })
                .first?.notation ?? "—"
            let actualNotation = result.topMoves.first?.move.notation ?? "—"
            let actualMatchesExpected = result.topMoves.first.map { top in
                result.probe.acceptable.contains(top.move)
            } ?? false
            let probStr = String(format: "%.3f", result.expectedProb)
            let rankStr = result.expectedRank.map(String.init) ?? "—"
            let w = String(format: "%.2f", result.valueWDL.win)
            let d = String(format: "%.2f", result.valueWDL.draw)
            let l = String(format: "%.2f", result.valueWDL.loss)

            HStack(spacing: 8) {
                Text(meta?.id ?? "—")
                    .font(.system(.caption, design: .monospaced))
                    .frame(width: 60, alignment: .leading)
                Text(meta.map { "\($0.rating)" } ?? "—")
                    .font(.system(.caption, design: .monospaced))
                    .foregroundStyle(.secondary)
                    .frame(width: 56, alignment: .trailing)
                Text(expectedNotation)
                    .font(.system(.body, design: .monospaced))
                    .frame(width: 90, alignment: .leading)
                Text(actualNotation)
                    .font(.system(.body, design: .monospaced))
                    .foregroundStyle(actualMatchesExpected ? Color.green : Color.primary)
                    .frame(width: 90, alignment: .leading)
                Text(probStr)
                    .font(.system(.body, design: .monospaced))
                    .frame(width: 64, alignment: .trailing)
                Text(rankStr)
                    .font(.system(.body, design: .monospaced))
                    .foregroundStyle(LichessProbeDetailView.rankColor(result.expectedRank))
                    .frame(width: 48, alignment: .trailing)
                LichessProbeDetailView.verdictBadge(result.verdict)
                    .frame(width: 100, alignment: .center)
                Text("\(w)/\(d)/\(l)")
                    .font(.system(.caption, design: .monospaced))
                    .foregroundStyle(.secondary)
                    .frame(width: 130, alignment: .trailing)
            }
            .padding(.vertical, 2)
            .contentShape(Rectangle())
            .onTapGesture {
                isShowingBoardPopover.toggle()
            }
            .popover(isPresented: $isShowingBoardPopover, arrowEdge: .leading) {
                TacticalProbeBoardPopover(result: result)
            }
        }
    }

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
                    columnHeader
                        .padding(.vertical, 4)
                        .padding(.horizontal, 12)
                        .background(Color(NSColor.controlBackgroundColor).opacity(0.5))
                    Divider()
                    ForEach(themeGroups, id: \.theme) { group in
                        themeSectionHeader(group)
                            .padding(.horizontal, 12)
                            .padding(.vertical, 6)
                            .background(Color(NSColor.controlBackgroundColor).opacity(0.3))
                        Divider()
                        ForEach(group.results, id: \.probe.name) { result in
                            DetailRowView(result: result)
                                .padding(.horizontal, 12)
                            Divider()
                        }
                    }
                    if history.latestPerPuzzleResults.isEmpty {
                        emptyState
                            .padding(.horizontal, 16)
                            .padding(.vertical, 32)
                    }
                }
            }
            footer
        }
    }

    // MARK: Header

    @ViewBuilder
    private var header: some View {
        HStack(spacing: 16) {
            Text("Lichess Probe Detail — 200 puzzles")
                .font(.system(.title2).weight(.semibold))
            Spacer()
            tickMetadataText
            Button("Probe now") {
                onProbeNow()
            }
            .controlSize(.small)
            Button("Export latest…") {
                onExport()
            }
            .controlSize(.small)
            .disabled(history.latestPerPuzzleResults.isEmpty)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
    }

    @ViewBuilder
    private var tickMetadataText: some View {
        if let ts = history.latestTickTimestamp {
            VStack(alignment: .trailing, spacing: 1) {
                Text("tick: \(Self.timestampFormatter.string(from: ts))")
                    .font(.system(.caption, design: .monospaced))
                    .foregroundStyle(.secondary)
                Text("model: \(history.latestTickModelLabel ?? "<unknown>")")
                    .font(.system(.caption, design: .monospaced))
                    .foregroundStyle(.secondary)
            }
        } else {
            Text("no tick yet")
                .font(.system(.caption, design: .monospaced))
                .foregroundStyle(.secondary)
        }
    }

    // MARK: Column header row

    @ViewBuilder
    private var columnHeader: some View {
        HStack(spacing: 8) {
            Text("ID")
                .frame(width: 60, alignment: .leading)
            Text("RATING")
                .frame(width: 56, alignment: .trailing)
            Text("EXPECTED")
                .frame(width: 90, alignment: .leading)
            Text("ACTUAL #1")
                .frame(width: 90, alignment: .leading)
            Text("PROB")
                .frame(width: 64, alignment: .trailing)
            Text("RANK")
                .frame(width: 48, alignment: .trailing)
            Text("VERDICT")
                .frame(width: 100, alignment: .center)
            Text("W / D / L")
                .frame(width: 130, alignment: .trailing)
        }
        .font(.system(.caption2, design: .monospaced).weight(.semibold))
        .foregroundStyle(.secondary)
    }

    @ViewBuilder
    private func themeSectionHeader(_ group: ThemeGroup) -> some View {
        HStack(spacing: 12) {
            Text(themeLabel(group.theme))
                .font(.system(.body).weight(.semibold))
            Text("\(group.correctCount) / \(group.results.count) argmax-correct")
                .font(.system(.caption, design: .monospaced))
                .foregroundStyle(.secondary)
            Spacer()
        }
    }

    // MARK: Per-row styling (shared with DetailRowView via static fns)

    @ViewBuilder
    fileprivate static func verdictBadge(_ verdict: ProbeVerdict) -> some View {
        let (label, fill, fg) = verdictStyle(verdict)
        Text(label)
            .font(.system(.caption2, design: .monospaced).weight(.semibold))
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(fill.opacity(0.25))
            .foregroundStyle(fg)
            .clipShape(RoundedRectangle(cornerRadius: 4))
    }

    fileprivate static func verdictStyle(_ v: ProbeVerdict) -> (String, Color, Color) {
        switch v {
        case .correctAndConfident: return ("TOP·CONF", .green, .green)
        case .correctButFlat:      return ("TOP·FLAT", .yellow, .orange)
        case .correctInTop5:       return ("TOP-5", .blue, .blue)
        case .wrong:               return ("WRONG", .red, .red)
        case .error:               return ("ERR", .gray, .gray)
        }
    }

    fileprivate static func rankColor(_ rank: Int?) -> Color {
        guard let rank else { return .secondary }
        if rank == 1 { return .green }
        if rank <= 5 { return .blue }
        return .red
    }

    // MARK: Empty state

    @ViewBuilder
    private var emptyState: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("No probe tick has landed yet.")
                .font(.system(.body))
            Text(
                "The watcher fires once on app launch and every 30 minutes; "
                + "press \"Probe now\" to fire one immediately."
            )
            .font(.caption)
            .foregroundStyle(.secondary)
        }
    }

    // MARK: Footer

    @ViewBuilder
    private var footer: some View {
        VStack(alignment: .leading, spacing: 2) {
            Divider()
            HStack {
                Text(
                    "200 puzzles from the bundled Lichess CC0 set, grouped by theme. "
                    + "ACTUAL #1 turns green when it matches EXPECTED. "
                    + "RANK colored green=#1, blue=top-5, red=#6+. "
                    + "PROB is legal-masked mass on the bookmove. "
                    + "Export writes the full per-puzzle JSON."
                )
                .font(.caption)
                .foregroundStyle(.secondary)
                Spacer()
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 6)
        }
    }

    // MARK: Grouping

    private struct ThemeGroup: Identifiable {
        let theme: ProbeCategory
        let results: [ProbeResult]
        var id: ProbeCategory { theme }
        var correctCount: Int {
            results.reduce(0) { acc, r in
                switch r.verdict {
                case .correctAndConfident, .correctButFlat:
                    return acc + 1
                default:
                    return acc
                }
            }
        }
    }

    private var themeGroups: [ThemeGroup] {
        var grouped: [ProbeCategory: [ProbeResult]] = [:]
        for r in history.latestPerPuzzleResults {
            grouped[r.probe.category, default: []].append(r)
        }
        return themeOrder.compactMap { theme in
            guard let results = grouped[theme] else { return nil }
            // Sort within theme by expected-move rank ascending (best
            // matches first), nil ranks at the end (errored / fixture
            // bugs), with puzzle id as the stable tie-breaker so the
            // order is reproducible across ticks even when the network
            // ranks many puzzles identically.
            let sorted = results.sorted { a, b in
                let ra = a.expectedRank ?? Int.max
                let rb = b.expectedRank ?? Int.max
                if ra != rb { return ra < rb }
                let aid = LichessProbeData.metadata[a.probe.name]?.id ?? a.probe.name
                let bid = LichessProbeData.metadata[b.probe.name]?.id ?? b.probe.name
                return aid < bid
            }
            return ThemeGroup(theme: theme, results: sorted)
        }
    }

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

    private static let timestampFormatter: DateFormatter = {
        let df = DateFormatter()
        df.dateFormat = "yyyy-MM-dd HH:mm:ss"
        return df
    }()
}
