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
    /// Optional parallel WIDE-set history. When present, a second OVERALL
    /// trend chart for the ~4,435-puzzle set is shown beneath the 200-set
    /// chart. nil ⇒ no wide chart (200-set-only window).
    let wideHistory: LichessProbeHistory?
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
        wideHistory: LichessProbeHistory? = nil,
        onProbeNow: @escaping @MainActor () -> Void = {},
        onExport: @escaping @MainActor () -> Void = {}
    ) {
        self.history = history
        self.wideHistory = wideHistory
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
            // ScrollView is vertical-only — the earlier two-axis form
            // (`[.vertical, .horizontal]`) caused SwiftUI to center the
            // table horizontally when the content was narrower than
            // the viewport, breaking the table's leading alignment.
            // With comparison loaded the row width grows past the
            // default window; resize the window wider to reach the
            // rightmost columns rather than scrolling horizontally.
            ScrollView(.vertical, showsIndicators: true) {
                LazyVStack(alignment: .leading, spacing: 0) {
                    overallSummaryBand
                        .padding(.vertical, 6)
                        .padding(.horizontal, 12)
                        .background(Color(NSColor.controlBackgroundColor).opacity(0.6))
                    Divider()
                    LichessProbeOverallTrendChart(
                        history: history,
                        cmpNll: comparison?.overallSummary.meanNegLogProb,
                        cmpElo: comparison.map(Self.cmpMlePuzzleElo)
                    )
                    .padding(.vertical, 6)
                    .padding(.horizontal, 12)
                    .background(Color(NSColor.controlBackgroundColor).opacity(0.4))
                    Divider()
                    if let wideHistory {
                        Text("WIDE PROBE SET (\(LichessProbeData.wideSet.count) puzzles)")
                            .font(.system(.caption, design: .default).weight(.semibold))
                            .foregroundStyle(.secondary)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(.horizontal, 12)
                            .padding(.top, 6)
                        LichessProbeOverallTrendChart(
                            history: wideHistory,
                            cmpNll: nil,
                            cmpElo: nil
                        )
                        .padding(.vertical, 6)
                        .padding(.horizontal, 12)
                        .background(Color(NSColor.controlBackgroundColor).opacity(0.4))
                        Divider()
                    }
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
                // Step + cumulative-progress line. Mirrors the status
                // bar cells so the Detail window stands alone as a
                // record of "where in training was this tick taken."
                // Each part is "—" when nil (champion-target probes
                // before Play-and-Train, or pre-checkpoint-controller
                // session boot).
                Text(progressMetadataLine)
                    .font(.system(.caption, design: .monospaced))
                    .foregroundStyle(.secondary)
            }
        } else {
            Text("no tick yet")
                .font(.system(.caption, design: .monospaced))
                .foregroundStyle(.secondary)
        }
    }

    /// "step: 10451 positions: 42.8M active: 2:14:07 arenas: 15 (0 promoted)"
    /// — composed inline so each nil-typed field is handled with a
    /// short "—" sentinel rather than dropping the whole row.
    private var progressMetadataLine: String {
        let stepStr = history.latestTickTrainingStep.map(String.init) ?? "—"
        let posStr = history.latestTickPositionsTrained
            .map { Self.compactCount($0) } ?? "—"
        let activeStr = history.latestTickActiveTrainingSec
            .map { Self.formatHMS(seconds: $0) } ?? "—"
        let arenaStr = history.latestTickArenaCount.map(String.init) ?? "—"
        let promoStr = history.latestTickPromotionCount.map(String.init) ?? "—"
        return "step: \(stepStr)  positions: \(posStr)"
            + "  active: \(activeStr)  arenas: \(arenaStr) (\(promoStr) promoted)"
    }

    /// "1,234,567" → "1.2M", "42,000" → "42.0K", "789" → "789".
    /// Matches the compact formatter the status bar uses so the
    /// Detail window's progress line reads the same as the bar.
    fileprivate static func compactCount(_ n: Int) -> String {
        let d = Double(n)
        if abs(d) >= 1e9 { return String(format: "%.2fB", d / 1e9) }
        if abs(d) >= 1e6 { return String(format: "%.1fM", d / 1e6) }
        if abs(d) >= 1e3 { return String(format: "%.1fK", d / 1e3) }
        return "\(n)"
    }

    /// Seconds → "H:MM:SS" (or "MM:SS" under one hour). Mirrors the
    /// status bar's `GameWatcher.Snapshot.formatHMS` shape without
    /// dragging that type into the Detail view's link surface.
    fileprivate static func formatHMS(seconds: Double) -> String {
        let total = Int(seconds.rounded())
        let h = total / 3600
        let m = (total % 3600) / 60
        let s = total % 60
        if h > 0 { return String(format: "%d:%02d:%02d", h, m, s) }
        return String(format: "%d:%02d", m, s)
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
        let liveAgg = liveAggregate(for: group)
        let cmpAgg = comparison?.aggregatesByCategory[group.theme]
        VStack(alignment: .leading, spacing: 2) {
            HStack(spacing: 12) {
                Text(themeLabel(group.theme))
                    .font(.system(.body).weight(.semibold))
                    .frame(width: 180, alignment: .leading)
                aggregateMetricCells(
                    total: liveAgg.total,
                    argmax: liveAgg.argmaxCorrect,
                    top5: liveAgg.top5Correct,
                    avgProb: liveAgg.avgExpectedProb,
                    avgRank: liveAgg.avgExpectedRank,
                    meanNll: liveAgg.meanNegLogProb,
                    showLabels: true,
                    vsProb: nil,
                    vsRank: nil,
                    vsNll: nil
                )
            }
            if comparison != nil {
                HStack(spacing: 12) {
                    Text("cmp")
                        .font(.system(.caption, design: .monospaced))
                        .foregroundStyle(.secondary)
                        .frame(width: 180, alignment: .leading)
                    if let c = cmpAgg {
                        aggregateMetricCells(
                            total: c.total,
                            argmax: c.argmaxCorrect,
                            top5: c.top5Correct,
                            avgProb: c.avgExpectedProb,
                            avgRank: c.avgExpectedRank,
                            meanNll: c.meanNegLogProb,
                            showLabels: false,
                            vsProb: liveAgg.avgExpectedProb,
                            vsRank: liveAgg.avgExpectedRank,
                            vsNll: liveAgg.meanNegLogProb
                        )
                    } else {
                        // Comparison loaded but no entries for this
                        // theme — render placeholder cells of matching
                        // widths so the row stays aligned with above.
                        Text("—").font(.system(.caption, design: .monospaced))
                            .foregroundStyle(.secondary)
                            .frame(width: 180, alignment: .leading)
                        Text("—").font(.system(.caption, design: .monospaced))
                            .foregroundStyle(.secondary)
                            .frame(width: 180, alignment: .leading)
                        Text("—").font(.system(.caption, design: .monospaced))
                            .foregroundStyle(.secondary)
                            .frame(width: 150, alignment: .leading)
                        Text("—").font(.system(.caption, design: .monospaced))
                            .foregroundStyle(.secondary)
                            .frame(width: 150, alignment: .leading)
                        Text("—").font(.system(.caption, design: .monospaced))
                            .foregroundStyle(.secondary)
                            .frame(width: 130, alignment: .leading)
                    }
                }
                HStack(spacing: 12) {
                    Text("Δ live − cmp")
                        .font(.system(.caption, design: .monospaced))
                        .foregroundStyle(.secondary)
                        .frame(width: 180, alignment: .leading)
                    diffMetricCells(
                        liveTotal: liveAgg.total,
                        liveArgmax: liveAgg.argmaxCorrect,
                        liveTop5: liveAgg.top5Correct,
                        liveProb: liveAgg.avgExpectedProb,
                        liveRank: liveAgg.avgExpectedRank,
                        liveNll: liveAgg.meanNegLogProb,
                        cmpTotal: cmpAgg?.total,
                        cmpArgmax: cmpAgg?.argmaxCorrect,
                        cmpTop5: cmpAgg?.top5Correct,
                        cmpProb: cmpAgg?.avgExpectedProb,
                        cmpRank: cmpAgg?.avgExpectedRank,
                        cmpNll: cmpAgg?.meanNegLogProb
                    )
                }
            }
        }
    }

    /// Build the live per-theme `LichessProbeHistory.Aggregate` for a
    /// rendered theme group on the fly. The live history exposes only
    /// the time-series of `Aggregate`s per theme; the section header
    /// wants the latest tick's value, which is just the last entry.
    /// Falls back to a zero aggregate if the series for this theme
    /// hasn't ticked yet — keeps the layout stable while early.
    private func liveAggregate(for group: ThemeGroup) -> LichessProbeHistory.Aggregate {
        if let latest = history.latest(group.theme)?.aggregate {
            return latest
        }
        return LichessProbeHistory.Aggregate(
            theme: group.theme,
            total: group.results.count,
            argmaxCorrect: 0,
            top5Correct: 0,
            errored: 0,
            sumExpectedProb: 0,
            sumExpectedRank: 0,
            countWithRank: 0,
            sumNegLogProb: 0
        )
    }

    /// 4 fixed-width metric cells (argmax/top-5/avg prob/avg rank) for
    /// either a per-theme aggregate or the all-200 overall summary.
    /// `showLabels = true` prefixes the metric labels ("argmax …"); the
    /// cmp sub-row passes `false` so its values align under the live
    /// row's values without label repetition.
    ///
    /// When `vsProb` / `vsRank` are non-nil, the avg-prob / avg-rank
    /// cells are color-coded against those baselines (the cmp row
    /// passes the live row's avg as the baseline so green = comparison
    /// better, red = comparison worse — matching the per-row CMP cell
    /// logic).
    @ViewBuilder
    private func aggregateMetricCells(
        total: Int,
        argmax: Int,
        top5: Int,
        avgProb: Float,
        avgRank: Float?,
        meanNll: Double,
        showLabels: Bool,
        vsProb: Float?,
        vsRank: Float?,
        vsNll: Double?
    ) -> some View {
        let argmaxPct = total > 0
            ? String(format: "%.1f%%", 100.0 * Double(argmax) / Double(total))
            : "—"
        let top5Pct = total > 0
            ? String(format: "%.1f%%", 100.0 * Double(top5) / Double(total))
            : "—"
        let argmaxText = showLabels
            ? "argmax \(argmax)/\(total) (\(argmaxPct))"
            : "\(argmax)/\(total) (\(argmaxPct))"
        let top5Text = showLabels
            ? "top-5 \(top5)/\(total) (\(top5Pct))"
            : "\(top5)/\(total) (\(top5Pct))"
        let probText = showLabels
            ? "avg prob \(String(format: "%.3f", avgProb))"
            : String(format: "%.3f", avgProb)
        let rankText: String = {
            if let r = avgRank {
                return showLabels
                    ? "avg rank \(String(format: "%.2f", r))"
                    : String(format: "%.2f", r)
            } else {
                return showLabels ? "avg rank —" : "—"
            }
        }()
        let nllText = showLabels
            ? "NLL \(String(format: "%.3f", meanNll))"
            : String(format: "%.3f", meanNll)

        Text(argmaxText)
            .font(.system(.caption, design: .monospaced))
            .frame(width: 180, alignment: .leading)
        Text(top5Text)
            .font(.system(.caption, design: .monospaced))
            .frame(width: 180, alignment: .leading)
        Text(probText)
            .font(.system(.caption, design: .monospaced))
            .foregroundStyle(Self.summaryProbColor(value: avgProb, vs: vsProb))
            .frame(width: 150, alignment: .leading)
        Text(rankText)
            .font(.system(.caption, design: .monospaced))
            .foregroundStyle(Self.summaryRankColor(value: avgRank, vs: vsRank))
            .frame(width: 150, alignment: .leading)
        Text(nllText)
            .font(.system(.caption, design: .monospaced))
            .foregroundStyle(Self.summaryNllColor(value: meanNll, vs: vsNll))
            .frame(width: 130, alignment: .leading)
    }

    /// Color for an NLL cell. Lower is better (closer to 0 = perfect
    /// prediction), so `value < vs - epsilon` → green (this side
    /// better); `value > vs + epsilon` → red. Tied within ±0.005
    /// renders primary (NLL changes of less than ~0.005 nats over
    /// 200 probes are dominated by sampling noise on a single
    /// errored probe).
    fileprivate static func summaryNllColor(value: Double, vs: Double?) -> Color {
        guard let vs else { return .primary }
        if value < vs - 0.005 { return .green }
        if value > vs + 0.005 { return .red }
        return .primary
    }

    /// One cell rendering the MLE puzzle-Elo: `pElo 845`, or the
    /// floor / ceiling / no-data sentinels. Width matches the NLL
    /// cell so the OVERALL band's metric grid stays uniform.
    @ViewBuilder
    private func puzzleEloCell(elo: Double, showLabel: Bool, vsElo: Double?) -> some View {
        Text(Self.formatPuzzleElo(elo, showLabel: showLabel))
            .font(.system(.caption, design: .monospaced))
            .foregroundStyle(Self.summaryEloColor(value: elo, vs: vsElo))
            .frame(width: 130, alignment: .leading)
    }

    /// "pElo 845" / "pElo <floor" / "pElo >ceil" / "pElo —"
    /// (NaN). The floor / ceiling sentinels render when MLE
    /// goes unbounded — all-wrong or all-correct over the
    /// probe set. Without a label prefix, the cmp/diff rows
    /// drop "pElo " to align under the live row.
    fileprivate static func formatPuzzleElo(_ elo: Double, showLabel: Bool) -> String {
        let prefix = showLabel ? "pElo " : ""
        if elo.isNaN { return "\(prefix)—" }
        if elo == -.infinity { return "\(prefix)<floor" }
        if elo == .infinity { return "\(prefix)>ceil" }
        return "\(prefix)\(String(format: "%.0f", elo))"
    }

    /// Color for a pElo cell. Higher is better. `±5` Elo threshold
    /// is wider than typical Elo noise on small probe sets — keeps
    /// the cell from flickering on sub-significant moves.
    fileprivate static func summaryEloColor(value: Double, vs: Double?) -> Color {
        guard let vs, value.isFinite, vs.isFinite else { return .primary }
        if value > vs + 5 { return .green }
        if value < vs - 5 { return .red }
        return .primary
    }

    /// "Δ +12" / "Δ -34" / "Δ —" pElo diff cell. Both sides must
    /// be finite for a numeric delta — sentinel-vs-sentinel (e.g.
    /// both "<floor") renders "Δ 0" if they match, "Δ —" otherwise.
    @ViewBuilder
    private func puzzleEloDiffCell(live: Double, cmp: Double) -> some View {
        let cell = Self.diffCellForElo(live: live, cmp: cmp)
        Text(cell.text)
            .font(.system(.caption, design: .monospaced))
            .foregroundStyle(cell.color)
            .frame(width: 130, alignment: .leading)
    }

    /// Pure-data helper for `puzzleEloDiffCell` — `(text, color)`
    /// the renderer can drop into a single `Text` cell without
    /// branching inside the ViewBuilder (which chokes on
    /// deferred-init `let` patterns).
    private static func diffCellForElo(live: Double, cmp: Double) -> DiffCell {
        if live.isFinite && cmp.isFinite {
            let d = live - cmp
            let signed = signedInt(Int(d.rounded()))
            let color: Color = {
                if d > 5 { return .green }
                if d < -5 { return .red }
                return .primary
            }()
            return DiffCell(text: "Δ \(signed)", color: color)
        }
        if live == cmp {
            return DiffCell(text: "Δ 0", color: .primary)
        }
        return DiffCell(text: "Δ —", color: .secondary)
    }

    /// Build the per-puzzle (rating, correct) pairs needed by
    /// `mlePuzzleElo`. Falls back to skipping any puzzle whose
    /// metadata lookup fails (defensive — the 200-puzzle bundle is
    /// stable but the lookup is by `probe.name`, which is a constructed
    /// string).
    private func liveMlePuzzleElo() -> Double {
        let pairs: [(rating: Int, correct: Bool)] = history.latestPerPuzzleResults.compactMap {
            guard let meta = LichessProbeData.metadata[$0.probe.name] else { return nil }
            let isArgmaxCorrect = $0.verdict == .correctAndConfident
                || $0.verdict == .correctButFlat
            return (rating: meta.rating, correct: isArgmaxCorrect)
        }
        return LichessProbeHistory.mlePuzzleElo(pairs: pairs)
    }

    /// Comparison-side pElo, computed from the loaded snapshot's
    /// own `rating_glicko2` field (NOT the live bundle's metadata —
    /// the snapshot's ratings are what the network was scored
    /// against at export time, even if the bundle has since
    /// changed). Returns NaN if the snapshot is missing the rating
    /// field on every puzzle (older schema-v2 exports).
    private static func cmpMlePuzzleElo(_ cmp: LichessProbeComparison) -> Double {
        let pairs: [(rating: Int, correct: Bool)] = cmp.payload.puzzles.compactMap {
            guard let rating = $0.puzzle.ratingGlicko2 else { return nil }
            let isArgmaxCorrect = $0.probeResult.verdict == "correctAndConfident"
                || $0.probeResult.verdict == "correctButFlat"
            return (rating: rating, correct: isArgmaxCorrect)
        }
        return LichessProbeHistory.mlePuzzleElo(pairs: pairs)
    }

    /// Color for an avg-prob cell that's possibly a cmp value:
    /// `value > vs + epsilon` is "this side is better" → green;
    /// `value < vs - epsilon` is "this side is worse" → red.
    /// `vs == nil` (live row) → primary.
    fileprivate static func summaryProbColor(value: Float, vs: Float?) -> Color {
        guard let vs else { return .primary }
        let epsilon: Float = 0.001
        if value > vs + epsilon { return .green }
        if value < vs - epsilon { return .red }
        return .primary
    }

    /// Color for an avg-rank cell. Lower rank = better, so:
    /// `value < vs` → green (this side better); `value > vs` → red.
    fileprivate static func summaryRankColor(value: Float?, vs: Float?) -> Color {
        guard let value, let vs else { return .secondary }
        if value < vs - 0.05 { return .green }
        if value > vs + 0.05 { return .red }
        return .primary
    }

    /// 4 fixed-width cells aligned under the live/cmp rows showing the
    /// `live - cmp` delta for each metric. Sign + value with a "Δ"
    /// prefix. Color follows the same "live-better = green" semantic as
    /// the per-row CMP cell: positive delta on argmax/top-5/prob is
    /// live-better-than-cmp = green; positive delta on rank is
    /// live-worse-than-cmp (rank lower = better) = red.
    ///
    /// `cmp == nil` (per-theme cmp aggregate missing for a theme that
    /// exists live) renders four "—" placeholders so the row stays
    /// dimensionally aligned with the live / cmp rows above.
    /// Three-string view-model for one diff cell. Built ahead-of-time
    /// so `diffMetricCells` can stay a pure tuple-of-`Text` ViewBuilder
    /// (which is the only shape SwiftUI's result builder reliably
    /// composes across multiple sibling cells without choking on opaque
    /// return types).
    private struct DiffCell {
        let text: String
        let color: Color
    }

    private static func diffCellForCount(
        liveTotal: Int,
        liveCount: Int,
        cmpTotal: Int?,
        cmpCount: Int?
    ) -> DiffCell {
        guard let cmpCount, let cmpTotal else {
            return DiffCell(text: "Δ —", color: .secondary)
        }
        let dCount = liveCount - cmpCount
        let livePct = liveTotal > 0 ? 100.0 * Double(liveCount) / Double(liveTotal) : 0
        let cmpPct = cmpTotal > 0 ? 100.0 * Double(cmpCount) / Double(cmpTotal) : 0
        let dPp = livePct - cmpPct
        return DiffCell(
            text: "Δ \(signedInt(dCount)) (\(signedPp(dPp)))",
            color: countBetterColor(delta: dCount)
        )
    }

    private static func diffCellForProb(live: Float, cmp: Float?) -> DiffCell {
        guard let cmp else { return DiffCell(text: "Δ —", color: .secondary) }
        let d = live - cmp
        return DiffCell(
            text: "Δ \(signedFloat(d, decimals: 3))",
            color: probBetterColor(delta: d)
        )
    }

    private static func diffCellForRank(live: Float?, cmp: Float?) -> DiffCell {
        guard let live, let cmp else { return DiffCell(text: "Δ —", color: .secondary) }
        let d = live - cmp
        return DiffCell(
            text: "Δ \(signedFloat(d, decimals: 2))",
            color: rankBetterColor(delta: d)
        )
    }

    private static func diffCellForNll(live: Double, cmp: Double?) -> DiffCell {
        guard let cmp else { return DiffCell(text: "Δ —", color: .secondary) }
        let d = live - cmp
        // Lower NLL = better, so positive delta = live worse = red.
        let color: Color
        if d < -0.005 { color = .green }
        else if d > 0.005 { color = .red }
        else { color = .primary }
        let signed: String = {
            let abs = String(format: "%.3f", abs(d))
            if d > 0 { return "+\(abs)" }
            if d < 0 { return "-\(abs)" }
            return abs
        }()
        return DiffCell(text: "Δ \(signed)", color: color)
    }

    @ViewBuilder
    private func diffMetricCells(
        liveTotal: Int,
        liveArgmax: Int,
        liveTop5: Int,
        liveProb: Float,
        liveRank: Float?,
        liveNll: Double,
        cmpTotal: Int?,
        cmpArgmax: Int?,
        cmpTop5: Int?,
        cmpProb: Float?,
        cmpRank: Float?,
        cmpNll: Double?
    ) -> some View {
        let argmax = Self.diffCellForCount(
            liveTotal: liveTotal, liveCount: liveArgmax,
            cmpTotal: cmpTotal, cmpCount: cmpArgmax
        )
        let top5 = Self.diffCellForCount(
            liveTotal: liveTotal, liveCount: liveTop5,
            cmpTotal: cmpTotal, cmpCount: cmpTop5
        )
        let prob = Self.diffCellForProb(live: liveProb, cmp: cmpProb)
        let rank = Self.diffCellForRank(live: liveRank, cmp: cmpRank)
        let nll = Self.diffCellForNll(live: liveNll, cmp: cmpNll)

        Text(argmax.text)
            .font(.system(.caption, design: .monospaced))
            .foregroundStyle(argmax.color)
            .frame(width: 180, alignment: .leading)
        Text(top5.text)
            .font(.system(.caption, design: .monospaced))
            .foregroundStyle(top5.color)
            .frame(width: 180, alignment: .leading)
        Text(prob.text)
            .font(.system(.caption, design: .monospaced))
            .foregroundStyle(prob.color)
            .frame(width: 150, alignment: .leading)
        Text(rank.text)
            .font(.system(.caption, design: .monospaced))
            .foregroundStyle(rank.color)
            .frame(width: 150, alignment: .leading)
        Text(nll.text)
            .font(.system(.caption, design: .monospaced))
            .foregroundStyle(nll.color)
            .frame(width: 130, alignment: .leading)
    }

    /// "+3" / "0" / "-2" with an explicit sign on positives.
    fileprivate static func signedInt(_ v: Int) -> String {
        v > 0 ? "+\(v)" : "\(v)"
    }

    /// "+1.0pp" / "0.0pp" / "-2.5pp" — percentage-points formatter with
    /// an explicit sign on positives.
    fileprivate static func signedPp(_ v: Double) -> String {
        let abs = String(format: "%.1f", abs(v))
        if v > 0 { return "+\(abs)pp" }
        if v < 0 { return "-\(abs)pp" }
        return "0.0pp"
    }

    /// "+0.123" / "0.000" / "-0.045" — float formatter with an explicit
    /// sign on positives.
    fileprivate static func signedFloat(_ v: Float, decimals: Int) -> String {
        let format = "%.\(decimals)f"
        let abs = String(format: format, abs(v))
        if v > 0 { return "+\(abs)" }
        if v < 0 { return "-\(abs)" }
        return abs
    }

    /// For argmax/top-5 deltas: positive count = live improved = green.
    fileprivate static func countBetterColor(delta: Int) -> Color {
        if delta > 0 { return .green }
        if delta < 0 { return .red }
        return .primary
    }

    /// For avg-prob delta: positive = live more confident in bookmove
    /// = better = green. Tied within ±0.001 renders primary.
    fileprivate static func probBetterColor(delta: Float) -> Color {
        let epsilon: Float = 0.001
        if delta > epsilon { return .green }
        if delta < -epsilon { return .red }
        return .primary
    }

    /// For avg-rank delta: positive = live higher rank = worse = red
    /// (lower rank is better). Tied within ±0.05 renders primary.
    fileprivate static func rankBetterColor(delta: Float) -> Color {
        if delta < -0.05 { return .green }
        if delta > 0.05 { return .red }
        return .primary
    }

    // MARK: Overall summary band

    /// Top "OVERALL" summary band — same metric layout as the per-theme
    /// section header but folded across all 200 puzzles. Renders one
    /// row for live data and (when a comparison is loaded) a second
    /// "cmp" row aligned under it with avg-prob / avg-rank color-coded
    /// against the live row.
    @ViewBuilder
    private var overallSummaryBand: some View {
        let liveAggregates = themeOrder.compactMap {
            history.latest($0)?.aggregate
        }
        let liveSummary = LichessProbeOverallSummary(folding: liveAggregates)
        let liveElo = liveMlePuzzleElo()
        VStack(alignment: .leading, spacing: 2) {
            HStack(spacing: 12) {
                Text("OVERALL")
                    .font(.system(.caption, design: .monospaced).weight(.bold))
                    .frame(width: 180, alignment: .leading)
                aggregateMetricCells(
                    total: liveSummary.totalProbes,
                    argmax: liveSummary.argmaxCorrect,
                    top5: liveSummary.top5Correct,
                    avgProb: liveSummary.avgExpectedProb,
                    avgRank: liveSummary.avgExpectedRank,
                    meanNll: liveSummary.meanNegLogProb,
                    showLabels: true,
                    vsProb: nil,
                    vsRank: nil,
                    vsNll: nil
                )
                puzzleEloCell(elo: liveElo, showLabel: true, vsElo: nil)
            }
            if let cmp = comparison {
                let cmpSummary = cmp.overallSummary
                let cmpElo = Self.cmpMlePuzzleElo(cmp)
                HStack(spacing: 12) {
                    Text("cmp")
                        .font(.system(.caption, design: .monospaced))
                        .foregroundStyle(.secondary)
                        .frame(width: 180, alignment: .leading)
                    aggregateMetricCells(
                        total: cmpSummary.totalProbes,
                        argmax: cmpSummary.argmaxCorrect,
                        top5: cmpSummary.top5Correct,
                        avgProb: cmpSummary.avgExpectedProb,
                        avgRank: cmpSummary.avgExpectedRank,
                        meanNll: cmpSummary.meanNegLogProb,
                        showLabels: false,
                        vsProb: liveSummary.avgExpectedProb,
                        vsRank: liveSummary.avgExpectedRank,
                        vsNll: liveSummary.meanNegLogProb
                    )
                    puzzleEloCell(elo: cmpElo, showLabel: false, vsElo: liveElo)
                }
                HStack(spacing: 12) {
                    Text("Δ live − cmp")
                        .font(.system(.caption, design: .monospaced))
                        .foregroundStyle(.secondary)
                        .frame(width: 180, alignment: .leading)
                    diffMetricCells(
                        liveTotal: liveSummary.totalProbes,
                        liveArgmax: liveSummary.argmaxCorrect,
                        liveTop5: liveSummary.top5Correct,
                        liveProb: liveSummary.avgExpectedProb,
                        liveRank: liveSummary.avgExpectedRank,
                        liveNll: liveSummary.meanNegLogProb,
                        cmpTotal: cmpSummary.totalProbes,
                        cmpArgmax: cmpSummary.argmaxCorrect,
                        cmpTop5: cmpSummary.top5Correct,
                        cmpProb: cmpSummary.avgExpectedProb,
                        cmpRank: cmpSummary.avgExpectedRank,
                        cmpNll: cmpSummary.meanNegLogProb
                    )
                    puzzleEloDiffCell(live: liveElo, cmp: cmpElo)
                }
            }
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
