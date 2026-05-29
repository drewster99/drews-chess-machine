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

    /// Click-toggled popover that explains what the RATING column
    /// actually measures. Hosted on the column header — row taps are
    /// reserved for the board popover.
    @State private var showRatingExplanation = false

    /// Loaded comparison snapshot from a previously-exported JSON.
    /// `nil` when no comparison is active. When set, each row grows
    /// 5 extra columns on the right (Actual #1, Prob, Rank, Verdict,
    /// W/D/L for the comparison entry), and the live row's rank /
    /// prob delta against the comparison is color-coded in the
    /// comparison cell.
    @State private var comparison: LichessProbeComparison?

    init(
        history: LichessProbeHistory,
        onProbeNow: @escaping @MainActor () -> Void = {},
        onExport: @escaping @MainActor () -> Void = {}
    ) {
        self.history = history
        self.onProbeNow = onProbeNow
        self.onExport = onExport
    }

    /// One-line summary used as both the hover tooltip (via `.help()`)
    /// on the column header and on each row's rating value cell.
    /// Short enough to fit a single tooltip line but specific enough to
    /// distinguish from any other "rating" the project might use.
    fileprivate static let ratingTooltip =
        "Lichess Glicko-2 puzzle rating (Elo-like; higher = harder)"

    /// Tap-to-toggle popover state must live on a row-scoped struct,
    /// not on `LichessProbeDetailView` itself — otherwise every row
    /// would share the same toggle and clicking one would open them
    /// all. Each row instantiates its own.
    ///
    /// `comparisonActive` flags whether ANY comparison is loaded; the
    /// 5 right-hand columns are rendered iff it's true. `comparison`
    /// carries the matched-by-puzzle-id entry from the loaded snapshot
    /// — `nil` when comparison is loaded but the puzzle isn't present
    /// in it (the row's right-hand columns render blank to keep the
    /// table aligned).
    fileprivate struct DetailRowView: View {
        let result: ProbeResult
        let comparisonActive: Bool
        let comparison: LichessProbeComparison.LoadedPuzzleEntry?
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
                    .help(LichessProbeDetailView.ratingTooltip)
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
                if comparisonActive {
                    Divider().frame(height: 18)
                    comparisonCells(currentResult: result)
                }
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

        /// Right-hand comparison cells. Rendered in two shapes:
        ///  * matched entry exists → 5 colored cells with the
        ///    comparison values, with PROB and RANK tinted vs the
        ///    current row (green = comparison was better, red =
        ///    comparison was worse, primary = same).
        ///  * matched entry missing → 5 blank "—" cells of the same
        ///    widths so the table stays aligned across rows.
        @ViewBuilder
        private func comparisonCells(currentResult: ProbeResult) -> some View {
            if let cmp = comparison {
                let cmpProbeResult = cmp.probeResult
                let cmpActual = cmpProbeResult.topMoves.first?.notation ?? "—"
                let cmpProbStr = String(format: "%.3f", cmpProbeResult.expectedProb)
                let cmpRankStr = cmpProbeResult.expectedRank.map(String.init) ?? "—"
                let cmpWdl = cmpProbeResult.valueWdl
                let cmpW = cmpWdl.map { String(format: "%.2f", $0.win) } ?? "—"
                let cmpD = cmpWdl.map { String(format: "%.2f", $0.draw) } ?? "—"
                let cmpL = cmpWdl.map { String(format: "%.2f", $0.loss) } ?? "—"

                Text(cmpActual)
                    .font(.system(.body, design: .monospaced))
                    .frame(width: 90, alignment: .leading)
                Text(cmpProbStr)
                    .font(.system(.body, design: .monospaced))
                    .foregroundStyle(
                        LichessProbeDetailView.comparisonProbDeltaColor(
                            cmp: cmpProbeResult.expectedProb,
                            current: currentResult.expectedProb
                        )
                    )
                    .frame(width: 64, alignment: .trailing)
                Text(cmpRankStr)
                    .font(.system(.body, design: .monospaced))
                    .foregroundStyle(
                        LichessProbeDetailView.comparisonRankDeltaColor(
                            cmp: cmpProbeResult.expectedRank,
                            current: currentResult.expectedRank
                        )
                    )
                    .frame(width: 48, alignment: .trailing)
                LichessProbeDetailView.verdictBadge(
                    verdictFromRawValue(cmpProbeResult.verdict)
                )
                .frame(width: 100, alignment: .center)
                Text("\(cmpW)/\(cmpD)/\(cmpL)")
                    .font(.system(.caption, design: .monospaced))
                    .foregroundStyle(.secondary)
                    .frame(width: 130, alignment: .trailing)
            } else {
                Text("—").font(.system(.body, design: .monospaced))
                    .foregroundStyle(.secondary)
                    .frame(width: 90, alignment: .leading)
                Text("—").font(.system(.body, design: .monospaced))
                    .foregroundStyle(.secondary)
                    .frame(width: 64, alignment: .trailing)
                Text("—").font(.system(.body, design: .monospaced))
                    .foregroundStyle(.secondary)
                    .frame(width: 48, alignment: .trailing)
                Text("—").font(.system(.body, design: .monospaced))
                    .foregroundStyle(.secondary)
                    .frame(width: 100, alignment: .center)
                Text("—").font(.system(.caption, design: .monospaced))
                    .foregroundStyle(.secondary)
                    .frame(width: 130, alignment: .trailing)
            }
        }

        /// Map the comparison's stored verdict string back to a
        /// `ProbeVerdict` so the shared `verdictBadge` helper can
        /// render it. Unknown strings fall through to `.error`.
        private func verdictFromRawValue(_ raw: String) -> ProbeVerdict {
            ProbeVerdict(rawValue: raw) ?? .error
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
            // ScrollView covers both axes so the optional 5 comparison
            // columns (which push total row width well past the
            // default window) can be reached by horizontal scrolling
            // without resizing the window first. Vertical is the
            // primary scroll; horizontal only kicks in when content
            // exceeds the viewport width.
            ScrollView([.vertical, .horizontal], showsIndicators: true) {
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
                            DetailRowView(
                                result: result,
                                comparisonActive: comparison != nil,
                                comparison: comparisonEntry(for: result)
                            )
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

    /// Look up the comparison's matched entry for a live row by puzzle
    /// id. Returns nil when no comparison is loaded OR when the loaded
    /// comparison doesn't include this puzzle (regenerated bundle,
    /// older curation snapshot, etc.).
    private func comparisonEntry(
        for result: ProbeResult
    ) -> LichessProbeComparison.LoadedPuzzleEntry? {
        guard let comparison else { return nil }
        guard let meta = LichessProbeData.metadata[result.probe.name] else {
            return nil
        }
        return comparison.byPuzzleId[meta.id]
    }

    // MARK: Header

    @ViewBuilder
    private var header: some View {
        HStack(spacing: 16) {
            Text("Lichess Probe Detail — 200 puzzles")
                .font(.system(.title2).weight(.semibold))
            Spacer()
            tickMetadataText
            comparisonMetadataText
            Button("Probe now") {
                onProbeNow()
            }
            .controlSize(.small)
            Button("Compare…") {
                if let loaded = LichessProbeComparisonLoader.loadFromFile() {
                    comparison = loaded
                }
            }
            .controlSize(.small)
            if comparison != nil {
                Button("Clear compare") {
                    SessionLogger.shared.log("[TACTICAL-LICHESS] compare cleared")
                    comparison = nil
                }
                .controlSize(.small)
            }
            Button("Export latest…") {
                onExport()
            }
            .controlSize(.small)
            .disabled(history.latestPerPuzzleResults.isEmpty)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
    }

    /// Right-aligned metadata block describing the currently-loaded
    /// comparison snapshot — filename, model label, tick timestamp.
    /// Renders nothing when no comparison is active.
    @ViewBuilder
    private var comparisonMetadataText: some View {
        if let cmp = comparison {
            VStack(alignment: .trailing, spacing: 1) {
                Text("cmp: \(cmp.sourceURL.lastPathComponent)")
                    .font(.system(.caption, design: .monospaced))
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .frame(maxWidth: 300, alignment: .trailing)
                Text("cmp model: \(cmp.payload.modelLabel ?? "<unknown>")")
                    .font(.system(.caption, design: .monospaced))
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }
        }
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
            ratingColumnHeaderCell
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
            if comparison != nil {
                Divider().frame(height: 14)
                Text("CMP ACTUAL #1")
                    .frame(width: 90, alignment: .leading)
                Text("CMP PROB")
                    .frame(width: 64, alignment: .trailing)
                Text("CMP RANK")
                    .frame(width: 48, alignment: .trailing)
                Text("CMP VERDICT")
                    .frame(width: 100, alignment: .center)
                Text("CMP W / D / L")
                    .frame(width: 130, alignment: .trailing)
            }
        }
        .font(.system(.caption2, design: .monospaced).weight(.semibold))
        .foregroundStyle(.secondary)
    }

    /// "RATING" column header rendered as a clickable label. Hover
    /// shows the short tooltip via `.help()`; click toggles a popover
    /// with a longer explanation. The text is underlined with a dotted
    /// pattern so the user has a visual cue this header is interactive
    /// — without that, the popover affordance is invisible.
    @ViewBuilder
    private var ratingColumnHeaderCell: some View {
        Text("RATING")
            .underline(true, pattern: .dot)
            .help(Self.ratingTooltip)
            .contentShape(Rectangle())
            .onTapGesture {
                showRatingExplanation.toggle()
            }
            .popover(isPresented: $showRatingExplanation, arrowEdge: .top) {
                ratingExplanationPopover
            }
    }

    /// Expanded explanation of the RATING column, shown when the user
    /// clicks the "RATING" header. Wider tooltip-equivalent — covers
    /// the rating family, the curation filter range, and the "higher =
    /// harder" interpretation in a single read.
    @ViewBuilder
    private var ratingExplanationPopover: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Lichess puzzle rating")
                .font(.headline)
            Text(
                "Each Lichess puzzle is rated with the Glicko-2 system — "
                + "the same family as chess Elo, with a reliability deviation "
                + "tracked alongside the number. The rating moves based on how "
                + "often users at known ratings solve the puzzle."
            )
            .font(.body)
            Text(
                "This probe set was filtered to ratings 800–1800 during "
                + "curation, with rating deviation ≤ 90 and ≥ 200 plays, "
                + "so the rating is well-stabilised for the puzzles you see here."
            )
            .font(.body)
            Text("Higher = harder.")
                .font(.body).bold()
        }
        .padding(16)
        .frame(width: 360)
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

    /// Color for the comparison's PROB cell. Higher prob = better
    /// (more mass on the bookmove). Comparison higher than current
    /// reads as "the snapshot was already better than current here"
    /// — green; comparison lower reads as "we've improved since the
    /// snapshot" — red. Tied (within a small epsilon) renders
    /// primary so visual noise is suppressed when the difference is
    /// not meaningful.
    fileprivate static func comparisonProbDeltaColor(
        cmp: Float,
        current: Float
    ) -> Color {
        let epsilon: Float = 0.001
        if cmp > current + epsilon { return .green }
        if cmp < current - epsilon { return .red }
        return .primary
    }

    /// Color for the comparison's RANK cell. Lower rank = better
    /// (bookmove higher in the legal-masked ranking). Same delta
    /// semantic as PROB: comparison lower rank = comparison was
    /// better = green; comparison higher rank = comparison was
    /// worse = red; equal = primary. nil-rank either side renders
    /// secondary so missing data doesn't masquerade as a delta.
    fileprivate static func comparisonRankDeltaColor(
        cmp: Int?,
        current: Int?
    ) -> Color {
        guard let cmpRank = cmp, let currentRank = current else {
            return .secondary
        }
        if cmpRank < currentRank { return .green }
        if cmpRank > currentRank { return .red }
        return .primary
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
                    + "Export writes the full per-puzzle JSON. "
                    + "Compare loads a previously-exported JSON into 5 right-hand columns; "
                    + "CMP PROB / CMP RANK turn green when the comparison was better "
                    + "than current, red when worse."
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
