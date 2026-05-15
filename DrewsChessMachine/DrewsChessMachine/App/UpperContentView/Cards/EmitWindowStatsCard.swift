import SwiftUI

/// "1-min emit" — what the last minute of self-play actually pushed
/// into the replay buffer. Three sections:
///
/// 1. **Total plies** — W / D / L position counts + share of all
///    emitted plies in the window.
/// 2. **By ply phase** — the same 5-bucket cutoffs the per-batch
///    `phase_by_ply` histogram uses (open ≤20, early 21–60,
///    mid 61–150, late 151–300, end 301+). Per bucket: position
///    count, share-of-window-plies, and the within-bucket W/D/L
///    composition.
/// 3. **By material phase** — same 5-bucket structure but bucketed
///    by non-pawn piece count (open ≥14, early 12–13, mid 8–11,
///    late 4–7, end ≤3).
///
/// Hidden by default; toggled via View > Show Emit Window Stats.
/// Sits at the top of column 2 (existing Forward Pass / Value Head
/// column). Status-bar typography matching `SelfPlayStatsCard` and
/// `ResultsCard`.
struct EmitWindowStatsCard: View {
    let snapshot: ParallelWorkerStatsBox.Snapshot?

    /// Per-bucket-row hover state. Stable across re-renders because
    /// the card is the SwiftUI identity boundary; @State persists for
    /// the life of this view's identity. Set to a unique
    /// "<section>.<bucket>" key (e.g. "ply.open" or "material.late")
    /// when the cursor enters the corresponding bucket label, nil
    /// otherwise. Drives the opacity of the tooltip overlay so the
    /// view tree stays stable (no `if`-gated visible content per the
    /// project's view-stability rule).
    @State private var hoveredBucketID: String?

    private static let dash = "—"
    private static let labelWidth: CGFloat = 90
    /// Width of the "plies (X.X%)" cell — sized for "1,234,567
    /// (100.0%)" worst case.
    private static let pliesCellWidth: CGFloat = 130
    /// Width of each W% / D% / L% within-bucket cell.
    private static let breakdownCellWidth: CGFloat = 56
    private static let cellSpacing: CGFloat = 6

    // MARK: - Bucket range tooltips
    //
    // Cutoffs match `PhaseHistogram.plyBucket` / `materialBucket`,
    // which themselves match `ReplayBuffer.computeBatchStats`'s
    // per-batch phase histograms. Keep these in sync if any cutoff
    // ever moves.

    private static let plyTooltips: [String: String] = [
        "open":  "ply 0–20",
        "early": "ply 21–60",
        "mid":   "ply 61–150",
        "late":  "ply 151–300",
        "end":   "ply 301+",
    ]
    private static let materialTooltips: [String: String] = [
        "open":  "≥14 non-pawn pieces",
        "early": "12–13 non-pawn pieces",
        "mid":   "8–11 non-pawn pieces",
        "late":  "4–7 non-pawn pieces",
        "end":   "≤3 non-pawn pieces",
    ]

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("1-min emit window")
                .font(.caption2)
                .foregroundStyle(.secondary)

            totalPliesSection
            phaseSection(
                title: "By ply phase",
                sectionID: "ply",
                tooltips: Self.plyTooltips,
                w: snapshot?.emitWindowPhaseByPlyW ?? .zero,
                d: snapshot?.emitWindowPhaseByPlyD ?? .zero,
                l: snapshot?.emitWindowPhaseByPlyL ?? .zero
            )
            phaseSection(
                title: "By material phase",
                sectionID: "material",
                tooltips: Self.materialTooltips,
                w: snapshot?.emitWindowPhaseByMaterialW ?? .zero,
                d: snapshot?.emitWindowPhaseByMaterialD ?? .zero,
                l: snapshot?.emitWindowPhaseByMaterialL ?? .zero
            )
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 6)
        .background(Color.secondary.opacity(0.10))
        .clipShape(RoundedRectangle(cornerRadius: 6))
    }

    // MARK: - Section 1: total plies (W / D / L)

    @ViewBuilder
    private var totalPliesSection: some View {
        let totalW = totalPositions(of: snapshot?.emitWindowPhaseByPlyW)
        let totalD = totalPositions(of: snapshot?.emitWindowPhaseByPlyD)
        let totalL = totalPositions(of: snapshot?.emitWindowPhaseByPlyL)
        let total = totalW + totalD + totalL

        VStack(alignment: .leading, spacing: 1) {
            Text("Total plies")
                .font(.caption2)
                .foregroundStyle(.secondary)
            simpleRow(label: "W", count: totalW, total: total)
            simpleRow(label: "D", count: totalD, total: total)
            simpleRow(label: "L", count: totalL, total: total)
        }
    }

    @ViewBuilder
    private func simpleRow(label: String, count: Int, total: Int) -> some View {
        HStack(spacing: Self.cellSpacing) {
            Text(label)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .frame(width: Self.labelWidth, alignment: .trailing)
            Text(pliesAndPct(count, of: total))
                .font(.system(.callout, design: .monospaced).weight(.semibold))
                .monospacedDigit()
                .lineLimit(1)
                .frame(width: Self.pliesCellWidth, alignment: .trailing)
            Spacer(minLength: 0)
        }
    }

    // MARK: - Section 2 / 3: per-bucket phase tables

    @ViewBuilder
    private func phaseSection(
        title: String,
        sectionID: String,
        tooltips: [String: String],
        w: PhaseHistogram,
        d: PhaseHistogram,
        l: PhaseHistogram
    ) -> some View {
        let totalAll = w.total + d.total + l.total

        VStack(alignment: .leading, spacing: 1) {
            // Section title.
            Text(title)
                .font(.caption2)
                .foregroundStyle(.secondary)

            // Header strip aligned with the data cells below.
            HStack(spacing: Self.cellSpacing) {
                Text("")
                    .frame(width: Self.labelWidth, alignment: .trailing)
                Text("plies")
                    .frame(width: Self.pliesCellWidth, alignment: .trailing)
                Text("W%")
                    .frame(width: Self.breakdownCellWidth, alignment: .trailing)
                Text("D%")
                    .frame(width: Self.breakdownCellWidth, alignment: .trailing)
                Text("L%")
                    .frame(width: Self.breakdownCellWidth, alignment: .trailing)
                Spacer(minLength: 0)
            }
            .font(.caption2)
            .foregroundStyle(.secondary)

            phaseRow(sectionID: sectionID, tooltip: tooltips["open"]  ?? "", label: "open",  bucket: \.open,  w: w, d: d, l: l, totalAll: totalAll)
            phaseRow(sectionID: sectionID, tooltip: tooltips["early"] ?? "", label: "early", bucket: \.early, w: w, d: d, l: l, totalAll: totalAll)
            phaseRow(sectionID: sectionID, tooltip: tooltips["mid"]   ?? "", label: "mid",   bucket: \.mid,   w: w, d: d, l: l, totalAll: totalAll)
            phaseRow(sectionID: sectionID, tooltip: tooltips["late"]  ?? "", label: "late",  bucket: \.late,  w: w, d: d, l: l, totalAll: totalAll)
            phaseRow(sectionID: sectionID, tooltip: tooltips["end"]   ?? "", label: "end",   bucket: \.end,   w: w, d: d, l: l, totalAll: totalAll)
        }
    }

    @ViewBuilder
    private func phaseRow(
        sectionID: String,
        tooltip: String,
        label: String,
        bucket: KeyPath<PhaseHistogram, Int>,
        w: PhaseHistogram,
        d: PhaseHistogram,
        l: PhaseHistogram,
        totalAll: Int
    ) -> some View {
        let wB = w[keyPath: bucket]
        let dB = d[keyPath: bucket]
        let lB = l[keyPath: bucket]
        let bucketTotal = wB + dB + lB
        let rowID = "\(sectionID).\(label)"
        let isHovering = hoveredBucketID == rowID

        HStack(spacing: Self.cellSpacing) {
            Text(label)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .frame(width: Self.labelWidth, alignment: .trailing)
                // Zero-delay hover tooltip showing the bucket's
                // numeric range. Anchored to the trailing edge of
                // the label and offset further right so the popup
                // floats next to (but not over) the data cells.
                // Always rendered for view-stability; opacity 0
                // when not hovering (the project's pattern for
                // conditional UI inside an animated container).
                // `.allowsHitTesting(false)` keeps the popup from
                // re-triggering hover events on itself.
                .overlay(alignment: .trailing) {
                    Text(tooltip)
                        .font(.caption2)
                        .foregroundStyle(.primary)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 3)
                        .background(.regularMaterial)
                        .clipShape(RoundedRectangle(cornerRadius: 4))
                        .shadow(radius: 2)
                        .fixedSize()
                        .opacity(isHovering ? 1 : 0)
                        .offset(x: 4, y: -22)
                        .allowsHitTesting(false)
                }
                .onHover { inside in
                    if inside {
                        hoveredBucketID = rowID
                    } else if hoveredBucketID == rowID {
                        hoveredBucketID = nil
                    }
                }
            Text(pliesAndPct(bucketTotal, of: totalAll))
                .font(.system(.callout, design: .monospaced))
                .monospacedDigit()
                .lineLimit(1)
                .frame(width: Self.pliesCellWidth, alignment: .trailing)
            Text(pctOnly(wB, of: bucketTotal))
                .font(.system(.callout, design: .monospaced))
                .monospacedDigit()
                .lineLimit(1)
                .frame(width: Self.breakdownCellWidth, alignment: .trailing)
            Text(pctOnly(dB, of: bucketTotal))
                .font(.system(.callout, design: .monospaced))
                .monospacedDigit()
                .lineLimit(1)
                .frame(width: Self.breakdownCellWidth, alignment: .trailing)
            Text(pctOnly(lB, of: bucketTotal))
                .font(.system(.callout, design: .monospaced))
                .monospacedDigit()
                .lineLimit(1)
                .frame(width: Self.breakdownCellWidth, alignment: .trailing)
            Spacer(minLength: 0)
        }
    }

    // MARK: - Formatters

    private func totalPositions(of h: PhaseHistogram?) -> Int {
        h?.total ?? 0
    }

    /// "12,345 (15.2%)" or "—" when total is zero.
    private func pliesAndPct(_ count: Int, of total: Int) -> String {
        guard total > 0 else { return Self.dash }
        let countStr = count.formatted(.number.grouping(.automatic))
        let pct = Double(count) / Double(total) * 100
        return String(format: "%@ (%.1f%%)", countStr, pct)
    }

    /// "15.2%" or "—" when total is zero.
    private func pctOnly(_ count: Int, of total: Int) -> String {
        guard total > 0 else { return Self.dash }
        let pct = Double(count) / Double(total) * 100
        return String(format: "%.1f%%", pct)
    }
}
