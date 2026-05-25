import AppKit
import Charts
import SwiftUI

/// Sheet that displays the full per-session arena history. The list
/// is ordered newest-first so the most recent result is visible
/// without scrolling.
///
/// The visible row is a glanceable two-lane summary — status badge,
/// score, W/D/L proportional bar, Elo±CI on lane 1; index, date,
/// games+duration, step on lane 2. Long ModelIDs and the per-color
/// (W/B) breakdown are hidden by default and revealed by clicking
/// any row, which opens an independent, movable detail window with
/// all the diagnostic detail plus a "Copy details" button.
///
/// A sparkline above the list shows the score trend across the
/// session, with one dot per arena colored by status. Tapping a dot
/// scrolls the list to that row, so the trend view doubles as a
/// navigation index.
struct ArenaHistoryView: View {
    let history: [TournamentRecord]
    /// Total games configured for tournaments at the time the
    /// sheet was opened — used in the popover so the "X / Y games"
    /// readout matches what the rest of the UI reports.
    let configuredGamesPerTournament: Int
    /// Promote threshold from `TrainingParameters`. Drives the
    /// status classification (high outlier vs noise band) and the
    /// dashed reference line in the sparkline.
    let promoteThreshold: Double
    let onClose: () -> Void
    /// Sheet title — defaults to "Arena History". Overridden when the
    /// view is reused for filtered subsets (e.g. promotions only).
    var title: String = "Arena History"
    /// Noun used in the header's count text. Singular form; rendered
    /// with a simple "+s" pluralization in `headerBar`. Defaults to
    /// "tournament" so existing call sites keep their wording.
    var unitNoun: String = "tournament"
    /// Optional parallel array mapping `history[i]` to the arena
    /// number it should display as ("#23", "#56", …). When nil, rows
    /// fall back to `i + 1` — i.e., the row's index in `history`.
    /// Required length, if non-nil, equals `history.count`; mismatch
    /// is treated as nil so a render never crashes from a stale
    /// caller-side filter.
    var displayIndices: [Int]? = nil

    /// Row the list should scroll to. Set when a sparkline dot is
    /// tapped so the list jumps to that arena; the change drives the
    /// `.onChange` scroll below. (Arena detail itself now opens as an
    /// independent window — see `ArenaDetailWindowManager`.)
    @State private var scrollTargetID: UUID?

    /// Drives the 3D arena win-rate surface sheet — the per-ply
    /// win-rate chart stacked across every arena along a Z axis.
    @State private var showSurface = false

    var body: some View {
        VStack(spacing: 0) {
            headerBar
            Divider()
            if history.isEmpty {
                emptyState
            } else {
                ArenaTrendSparkline(
                    history: history,
                    promoteThreshold: promoteThreshold,
                    onTapRecord: { id in
                        scrollTargetID = id
                    }
                )
                .frame(height: 60)
                .padding(.horizontal, 16)
                .padding(.vertical, 8)

                Divider()

                historyList
            }
        }
        .frame(minWidth: 600, idealWidth: 760, minHeight: 380, idealHeight: 600)
        .sheet(isPresented: $showSurface) {
            ArenaSurfaceView(
                history: history,
                onClose: { showSurface = false }
            )
        }
    }

    @ViewBuilder
    private var headerBar: some View {
        HStack {
            Text(title)
                .font(.title2.weight(.semibold))
            Spacer()
            Text("\(history.count) \(unitNoun)\(history.count == 1 ? "" : "s")")
                .font(.callout)
                .foregroundStyle(.secondary)
            Button("3D Surface") { showSurface = true }
            Button("Close", action: onClose)
                .keyboardShortcut(.cancelAction)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
    }

    @ViewBuilder
    private var emptyState: some View {
        VStack {
            Spacer()
            Text("No data to display")
                .foregroundStyle(.secondary)
            Spacer()
        }
        .frame(maxWidth: .infinity)
    }

    @ViewBuilder
    private var historyList: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 0) {
                    // Newest first so the recent picture is visible
                    // without scrolling. Reverse the backing array;
                    // the original `id` from `TournamentRecord` is
                    // stable so SwiftUI can diff updates.
                    ForEach(Array(history.enumerated().reversed()), id: \.element.id) { idx, record in
                        row(idx: idx, record: record)
                        Divider()
                    }
                }
            }
            .onChange(of: scrollTargetID) { _, newID in
                guard let id = newID else { return }
                // Animate the scroll so a sparkline-tap that jumps to
                // an offscreen row doesn't make the popover appear out
                // of nowhere — the row visibly slides into view first.
                withAnimation(.easeInOut(duration: 0.2)) {
                    proxy.scrollTo(id, anchor: .center)
                }
            }
        }
    }

    @ViewBuilder
    private func row(idx: Int, record: TournamentRecord) -> some View {
        ArenaHistoryRow(
            index: displayIndex(forRowAt: idx),
            record: record,
            promoteThreshold: promoteThreshold,
            configuredGamesPerTournament: configuredGamesPerTournament,
            rowParity: idx % 2
        )
        .id(record.id)
    }

    /// Pick the arena number to render in the row's "#N" badge. Uses
    /// the caller-supplied `displayIndices` when its length matches
    /// `history`; otherwise falls back to the row's 1-based position
    /// — matches the pre-parameter behavior.
    private func displayIndex(forRowAt idx: Int) -> Int {
        if let displayIndices, displayIndices.count == history.count {
            return displayIndices[idx]
        }
        return idx + 1
    }
}

// MARK: - Status classification

/// Status bucket used by both the row's edge accent and the
/// sparkline's dot color. Lifted to file scope so the two views
/// share one classifier and stay visually consistent.
fileprivate enum ArenaStatusKind {
    case promoted, highOutlier, lowOutlier, neutral

    /// Edge-accent / dot color. Promoted is the strongest signal
    /// (saturated green), high outliers (kept-but-near-threshold)
    /// are amber to distinguish "almost promoted" from "promoted",
    /// low outliers are red, and the noise band is invisible.
    var accentColor: Color {
        switch self {
        case .promoted:    return .green
        case .highOutlier: return .orange
        case .lowOutlier:  return .red
        case .neutral:     return .clear
        }
    }
}

fileprivate func arenaStatusKind(
    for record: TournamentRecord,
    promoteThreshold: Double
) -> ArenaStatusKind {
    if record.promoted { return .promoted }
    guard record.gamesPlayed > 0 else { return .neutral }
    // Symmetric "blowout" floor below 0.5: if the candidate scored
    // at least as far below 0.5 as `promoteThreshold` is above it,
    // flag it as a low-side outlier so a quick scan picks out
    // arenas where the candidate got crushed.
    let highCutoff = promoteThreshold
    let lowCutoff = max(0, 1.0 - promoteThreshold)
    if record.score >= highCutoff { return .highOutlier }
    if record.score <= lowCutoff { return .lowOutlier }
    return .neutral
}

// MARK: - Trend sparkline

/// 60pt-tall score trend across all completed arenas in this
/// session. One dot per arena colored by status. Dashed reference
/// lines at the promote threshold and its mirror (1 − threshold)
/// give the noise band a visible boundary. Tap a dot → invokes
/// `onTapRecord` with that arena's stable `TournamentRecord.id`,
/// which the parent uses to scroll-and-popover.
private struct ArenaTrendSparkline: View {
    let history: [TournamentRecord]
    let promoteThreshold: Double
    let onTapRecord: (UUID) -> Void

    private struct Point: Identifiable {
        let id: UUID
        let index: Int
        let score: Double
        let kind: ArenaStatusKind
    }

    private var points: [Point] {
        history.enumerated().map { idx, rec in
            Point(
                id: rec.id,
                index: idx + 1,
                score: rec.score,
                kind: arenaStatusKind(for: rec, promoteThreshold: promoteThreshold)
            )
        }
    }

    private func color(for kind: ArenaStatusKind) -> Color {
        switch kind {
        case .promoted:    return .green
        case .highOutlier: return .orange
        case .lowOutlier:  return .red
        case .neutral:     return .secondary
        }
    }

    var body: some View {
        let pts: [Point] = points
        Chart {
            chartContent(pts: pts)
        }
        .chartYScale(domain: 0...1)
        .chartXScale(domain: 0.5...(Double(max(pts.count, 1)) + 0.5))
        .chartXAxis(.hidden)
        .chartYAxis {
            AxisMarks(position: .leading, values: [0, 0.5, 1.0]) { value in
                AxisGridLine()
                AxisValueLabel {
                    if let v = value.as(Double.self) {
                        Text(String(format: "%.1f", v))
                            .font(.system(size: 7))
                            .monospacedDigit()
                    }
                }
            }
        }
        .chartOverlay { proxy in
            GeometryReader { geo in
                Rectangle()
                    .fill(Color.clear)
                    .contentShape(Rectangle())
                    .onTapGesture { location in
                        handleTap(location: location, proxy: proxy, geo: geo, pts: pts)
                    }
            }
        }
    }

    @ChartContentBuilder
    private func chartContent(pts: [Point]) -> some ChartContent {
        let lowRef = max(0, 1.0 - promoteThreshold)
        // Faint connecting line so the eye reads the time series
        // even when most dots are clustered near 0.5.
        ForEach(pts) { p in
            LineMark(
                x: .value("Arena", p.index),
                y: .value("Score", p.score)
            )
            .foregroundStyle(Color.gray.opacity(0.35))
            .lineStyle(StrokeStyle(lineWidth: 1))
        }
        // Promote-threshold reference (top) and its mirror (bottom)
        // bracket the noise band visually.
        RuleMark(y: .value("Promote", promoteThreshold))
            .foregroundStyle(Color.green.opacity(0.45))
            .lineStyle(StrokeStyle(lineWidth: 1, dash: [3, 3]))
        RuleMark(y: .value("Floor", lowRef))
            .foregroundStyle(Color.red.opacity(0.45))
            .lineStyle(StrokeStyle(lineWidth: 1, dash: [3, 3]))
        ForEach(pts) { p in
            PointMark(
                x: .value("Arena", p.index),
                y: .value("Score", p.score)
            )
            .foregroundStyle(color(for: p.kind))
            .symbolSize(36)
        }
    }

    /// Convert a tap into a chart-X coordinate, snap to the nearest
    /// arena index, and fire `onTapRecord` if the tap landed within
    /// a capture radius of that dot. The radius keeps fat-finger
    /// taps in the right neighborhood without grabbing taps that
    /// aren't really on a dot.
    private func handleTap(
        location: CGPoint,
        proxy: ChartProxy,
        geo: GeometryProxy,
        pts: [Point]
    ) {
        let origin: CGPoint = proxy.plotFrame.map { geo[$0].origin } ?? .zero
        let xInPlot = location.x - origin.x
        guard let xVal: Double = proxy.value(atX: xInPlot) else { return }
        let nearest = pts.min { a, b in
            abs(Double(a.index) - xVal) < abs(Double(b.index) - xVal)
        }
        guard let n = nearest else { return }
        // A real `guard let` here — the prior `?? 0` fallback collapsed
        // `dotX` to the chart's `origin.x` whenever `proxy.position(forX:)`
        // returned nil, which could spuriously match taps near the
        // chart's left edge or reject taps elsewhere that genuinely
        // landed near a dot.
        guard let plotX = proxy.position(forX: Double(n.index)) else { return }
        let dotX = plotX + origin.x
        if abs(location.x - dotX) <= 14 {
            onTapRecord(n.id)
        }
    }
}

// MARK: - Row

private struct ArenaHistoryRow: View {
    let index: Int
    let record: TournamentRecord
    let promoteThreshold: Double
    let configuredGamesPerTournament: Int
    /// 0 or 1, used for subtle alternating row backgrounds. Computed
    /// in the parent off the original (chronological) index so the
    /// stripe pattern stays stable across re-renders even though
    /// the visible order is reversed.
    let rowParity: Int

    @State private var hovering = false

    private static let metaDateFmt: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "EEE h:mm a"
        return f
    }()

    private var statusKind: ArenaStatusKind {
        arenaStatusKind(for: record, promoteThreshold: promoteThreshold)
    }

    private var statusBadgeText: String {
        if record.promoted {
            switch record.promotionKind {
            case .manual: return "PROMOTED ★ (manual)"
            case .automatic, .none: return "PROMOTED ★"
            }
        }
        return "kept"
    }

    private var statusBadgeColor: Color {
        record.promoted ? .green : .secondary
    }

    private var eloColor: Color {
        switch statusKind {
        case .promoted:    return .green
        case .highOutlier: return .orange
        case .lowOutlier:  return .red
        case .neutral:     return .secondary
        }
    }

    private var compactDuration: String {
        // "M:SS" → "Xm Ys" so the row's metadata line reads as
        // a duration rather than as a clock time.
        let total = Int(record.durationSec.rounded())
        let m = total / 60
        let s = total % 60
        if m == 0 { return "\(s)s" }
        return "\(m)m \(s)s"
    }

    private var dateText: String {
        record.finishedAt.map { Self.metaDateFmt.string(from: $0) } ?? "—"
    }

    var body: some View {
        let elo = record.eloSummary
        let eloCI = ArenaEloStats.formatEloWithCI(elo)
        let scoreText = String(format: "%.2f", record.score)
        let stepText = "step " + record.finishedAtStep.formatted(.number.grouping(.automatic))

        HStack(spacing: 0) {
            // Color edge accent — the glanceable status signal.
            // Wide enough to catch the eye when scanning a long
            // list but narrow enough that a noise-band (clear) edge
            // doesn't leave a visible gap.
            Rectangle()
                .fill(statusKind.accentColor)
                .frame(width: 4)

            VStack(alignment: .leading, spacing: 4) {
                // Lane 1 — primary info: status badge, score,
                // W/D/L bar, Elo±CI. The bar and Elo flex to
                // fill available width; the badge and score stay
                // at intrinsic size.
                HStack(alignment: .center, spacing: 12) {
                    ArenaHistoryStatusBadge(text: statusBadgeText, color: statusBadgeColor)
                    Text(scoreText)
                        .font(.system(.title3, design: .monospaced).weight(.bold))
                        .foregroundStyle(statusBadgeColor)
                        .frame(minWidth: 56, alignment: .leading)
                    WLDBar(
                        wins: record.candidateWins,
                        draws: record.draws,
                        losses: record.championWins
                    )
                    .frame(height: 8)
                    .frame(maxWidth: .infinity)
                    Text(eloCI)
                        .font(.system(.callout, design: .monospaced))
                        .foregroundStyle(eloColor)
                        .lineLimit(1)
                        .fixedSize(horizontal: true, vertical: false)
                }
                // Lane 2 — secondary metadata. Same single-line
                // layout the user reads after they've absorbed
                // the lane-1 headline.
                HStack(spacing: 6) {
                    Text("#\(index)")
                        .foregroundStyle(.secondary)
                    Text("·")
                        .foregroundStyle(.tertiary)
                    Text(dateText)
                    Text("·")
                        .foregroundStyle(.tertiary)
                    Text("\(record.gamesPlayed) games")
                    Text("·")
                        .foregroundStyle(.tertiary)
                    Text(compactDuration)
                    Text("·")
                        .foregroundStyle(.tertiary)
                    Text(stepText)
                    Spacer(minLength: 0)
                }
                .font(.system(.caption, design: .monospaced))
                .foregroundStyle(.secondary)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 10)
        }
        .background(rowBackground)
        .contentShape(Rectangle())
        .onHover { hovering = $0 }
        .onTapGesture {
            ArenaDetailWindowManager.shared.open(
                record: record,
                index: index,
                configuredGamesPerTournament: configuredGamesPerTournament
            )
        }
    }

    private var rowBackground: Color {
        if hovering { return Color.gray.opacity(0.10) }
        if rowParity == 0 { return Color.gray.opacity(0.04) }
        return Color.clear
    }
}

// MARK: - Status badge

/// "PROMOTED ★" / "kept" pill. Visual styling driven entirely by
/// the inputs so SwiftUI sees a stable view tree.
private struct ArenaHistoryStatusBadge: View {
    let text: String
    let color: Color

    var body: some View {
        Text(text)
            .font(.system(.caption, design: .default).weight(.semibold))
            .foregroundStyle(color)
            .padding(.horizontal, 8)
            .padding(.vertical, 3)
            .background(
                RoundedRectangle(cornerRadius: 4)
                    .fill(color.opacity(0.12))
            )
            .frame(minWidth: 90, alignment: .center)
    }
}

// MARK: - W/D/L proportional bar

/// Three-color proportional bar showing wins / draws / losses for
/// the candidate. Width-flex; height fixed by the caller. Empty
/// (no games) renders as a neutral gray track so the layout slot
/// doesn't collapse mid-list.
private struct WLDBar: View {
    let wins: Int
    let draws: Int
    let losses: Int

    var body: some View {
        GeometryReader { geo in
            let total = wins + draws + losses
            if total == 0 {
                RoundedRectangle(cornerRadius: 2)
                    .fill(Color.gray.opacity(0.15))
            } else {
                let w = geo.size.width * CGFloat(wins) / CGFloat(total)
                let d = geo.size.width * CGFloat(draws) / CGFloat(total)
                let l = geo.size.width * CGFloat(losses) / CGFloat(total)
                HStack(spacing: 0) {
                    Color.green.opacity(0.85).frame(width: w)
                    Color.gray.opacity(0.55).frame(width: d)
                    Color.red.opacity(0.85).frame(width: l)
                }
                .clipShape(RoundedRectangle(cornerRadius: 2))
            }
        }
    }
}

// MARK: - Detail popover

// MARK: - Detail window

/// Opens and retains independent arena-detail windows. Each detail is
/// a real, movable, resizable `NSWindow` that stays open after the
/// Arena History sheet is dismissed. One window per arena record —
/// re-requesting an already-open record brings its window forward
/// rather than stacking a duplicate.
@MainActor
final class ArenaDetailWindowManager: NSObject, NSWindowDelegate {
    static let shared = ArenaDetailWindowManager()

    /// Open detail windows keyed by record id. This dictionary is the
    /// sole strong reference keeping each window alive; the entry is
    /// dropped in `windowWillClose`, at which point the window
    /// deallocates.
    private var windows: [UUID: NSWindow] = [:]

    /// Top-left point the next cascaded window should take. Seeded from
    /// the first (centered) window and stepped by `cascadeTopLeft(from:)`
    /// each time another opens, so multiple detail windows fan out
    /// down-and-right instead of stacking on one spot. Once every
    /// window has closed, the next first-window open re-centers and
    /// re-seeds it.
    private var cascadeOrigin = NSPoint.zero

    func open(record: TournamentRecord, index: Int, configuredGamesPerTournament: Int) {
        if let existing = windows[record.id] {
            existing.makeKeyAndOrderFront(nil)
            return
        }
        let hasSummary = record.extendedSummary != nil
        let content = ArenaDetailView(
            record: record,
            index: index,
            configuredGamesPerTournament: configuredGamesPerTournament
        )
        let window = NSWindow(contentViewController: NSHostingController(rootView: content))
        window.title = "Arena #\(index)"
        window.styleMask = [.titled, .closable, .miniaturizable, .resizable]
        // We own the only strong reference (`windows`); dropping it in
        // `windowWillClose` is what frees the window, so AppKit must
        // not also release it on close.
        window.isReleasedWhenClosed = false
        window.delegate = self
        window.contentMinSize = NSSize(width: 360, height: 320)
        window.setContentSize(NSSize(
            width: hasSummary ? 700 : 380,
            height: hasSummary ? 760 : 440
        ))
        if windows.isEmpty {
            // First window: center it, then seed the cascade origin
            // from its frame. `cascadeTopLeft(from: .zero)` is a no-op
            // move that only returns the next cascade point.
            window.center()
            cascadeOrigin = window.cascadeTopLeft(from: .zero)
        } else {
            // Subsequent windows step down-and-right from the last,
            // so opening several arenas leaves them all visible.
            cascadeOrigin = window.cascadeTopLeft(from: cascadeOrigin)
        }
        windows[record.id] = window
        window.makeKeyAndOrderFront(nil)
    }

    func windowWillClose(_ notification: Notification) {
        guard let closing = notification.object as? NSWindow else { return }
        windows = windows.filter { $0.value !== closing }
    }
}

/// Per-arena detail view — the content of the independent window
/// `ArenaDetailWindowManager` opens when an arena-history row is
/// clicked. Surfaces every field on `TournamentRecord` that the
/// row's two-lane summary leaves out — the per-color (W/B)
/// breakdown, the three ModelIDs, the breakdown charts, and a
/// "Copy details" button that puts a structured plain-text summary
/// on the pasteboard for log/commit-message use.
private struct ArenaDetailView: View {
    let record: TournamentRecord
    let index: Int
    let configuredGamesPerTournament: Int

    private static let dateFmt: DateFormatter = {
        let f = DateFormatter()
        f.dateStyle = .medium
        f.timeStyle = .medium
        return f
    }()

    private var headline: String {
        if record.promoted {
            switch record.promotionKind {
            case .manual:               return "Arena #\(index) — PROMOTED manually"
            case .automatic, .none:     return "Arena #\(index) — PROMOTED automatically"
            }
        }
        return "Arena #\(index) — kept"
    }

    private var headlineColor: Color {
        record.promoted ? .green : .primary
    }

    private var scoreCI: String {
        ArenaEloStats.formatScorePercentWithCI(record.eloSummary)
    }

    private var eloCI: String {
        ArenaEloStats.formatEloWithCI(record.eloSummary)
    }

    private var durationStr: String {
        ArenaLogFormatter.formatDuration(record.durationSec)
    }

    private var drawRatePct: String {
        let f = ArenaLogFormatter.drawRateFraction(record: record)
        return String(format: "%.1f%%", f * 100)
    }

    private var whiteN: Int {
        record.candidateWinsAsWhite + record.candidateLossesAsWhite + record.candidateDrawsAsWhite
    }

    private var blackN: Int {
        record.candidateWinsAsBlack + record.candidateLossesAsBlack + record.candidateDrawsAsBlack
    }

    var body: some View {
        ScrollView {
            detailContent
        }
    }

    @ViewBuilder
    private var detailContent: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(headline)
                .font(.headline)
                .foregroundStyle(headlineColor)

            Divider()

            VStack(alignment: .leading, spacing: 6) {
                detailRow("Finished", record.finishedAt.map { Self.dateFmt.string(from: $0) } ?? "—")
                detailRow("Step", record.finishedAtStep.formatted(.number.grouping(.automatic)))
                detailRow("Games", "\(record.gamesPlayed) / \(configuredGamesPerTournament)")
                detailRow("Duration", durationStr)
            }

            Divider()

            VStack(alignment: .leading, spacing: 6) {
                detailRow("Score", scoreCI)
                detailRow("W·D·L", "\(record.candidateWins) · \(record.draws) · \(record.championWins)")
                detailRow("Elo (95%)", eloCI)
                detailRow("Draw rate", drawRatePct)
            }

            Divider()

            VStack(alignment: .leading, spacing: 6) {
                Text("Per-side breakdown")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)
                detailRow("As white", sideBreakdown(
                    wins: record.candidateWinsAsWhite,
                    draws: record.candidateDrawsAsWhite,
                    losses: record.candidateLossesAsWhite,
                    score: record.candidateScoreAsWhite,
                    n: whiteN
                ))
                detailRow("As black", sideBreakdown(
                    wins: record.candidateWinsAsBlack,
                    draws: record.candidateDrawsAsBlack,
                    losses: record.candidateLossesAsBlack,
                    score: record.candidateScoreAsBlack,
                    n: blackN
                ))
            }

            Divider()

            VStack(alignment: .leading, spacing: 6) {
                Text("Model IDs")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)
                detailRow("Trainer", record.candidateID?.description ?? "—")
                detailRow("Champion", record.championID?.description ?? "—")
                if record.promoted, let pid = record.promotedID {
                    detailRow("→ Promoted", pid.description, valueColor: .green)
                }
            }

            if let ext = record.extendedSummary {
                Divider()

                ArenaBreakdownsView(summary: ext)
            }

            Divider()

            HStack {
                Spacer()
                Button("Copy details") {
                    let pb = NSPasteboard.general
                    pb.clearContents()
                    pb.setString(structuredSummary(), forType: .string)
                }
                .controlSize(.small)
            }
        }
        .padding(16)
    }

    @ViewBuilder
    private func detailRow(_ label: String, _ value: String, valueColor: Color = .primary) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            Text(label)
                .font(.caption)
                .foregroundStyle(.secondary)
                .frame(width: 84, alignment: .leading)
            Text(value)
                .font(.system(.callout, design: .monospaced))
                .foregroundStyle(valueColor)
                .lineLimit(1)
                .truncationMode(.middle)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private func sideBreakdown(wins: Int, draws: Int, losses: Int, score: Double, n: Int) -> String {
        guard n > 0 else { return "—" }
        let pct = String(format: "%.1f%%", score * 100)
        return "\(wins)W · \(draws)D · \(losses)L  (\(pct))"
    }

    /// Multi-line plain-text summary suitable for pasting into a
    /// log line, commit message, or chat. Stable layout so external
    /// tooling (or a future "diff two arenas" gist) can parse it.
    private func structuredSummary() -> String {
        var lines: [String] = []
        lines.append(headline)
        lines.append("")
        if let d = record.finishedAt {
            lines.append("Finished:    \(Self.dateFmt.string(from: d))")
        }
        lines.append("Step:        " + record.finishedAtStep.formatted(.number.grouping(.automatic)))
        lines.append("Games:       \(record.gamesPlayed) / \(configuredGamesPerTournament) (\(durationStr))")
        lines.append("")
        lines.append("Score:       \(scoreCI)")
        lines.append("Elo (95%):   \(eloCI)")
        lines.append("Result:      \(record.candidateWins)W / \(record.draws)D / \(record.championWins)L")
        lines.append("Draw rate:   \(drawRatePct)")
        lines.append("")
        lines.append("By side:")
        lines.append("  As white:  " + sideBreakdown(
            wins: record.candidateWinsAsWhite,
            draws: record.candidateDrawsAsWhite,
            losses: record.candidateLossesAsWhite,
            score: record.candidateScoreAsWhite,
            n: whiteN
        ))
        lines.append("  As black:  " + sideBreakdown(
            wins: record.candidateWinsAsBlack,
            draws: record.candidateDrawsAsBlack,
            losses: record.candidateLossesAsBlack,
            score: record.candidateScoreAsBlack,
            n: blackN
        ))
        lines.append("")
        lines.append("Trainer:     " + (record.candidateID?.description ?? "—"))
        lines.append("Champion:    " + (record.championID?.description ?? "—"))
        if record.promoted, let pid = record.promotedID {
            lines.append("Promoted:    " + pid.description)
        }
        return lines.joined(separator: "\n")
    }
}
