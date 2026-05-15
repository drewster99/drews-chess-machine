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

    private static let dash = "—"
    /// Wide enough to fit the longest bucket label with its range
    /// suffix in caption2 — e.g. "late (151–300)" or "early (12–13)".
    private static let labelWidth: CGFloat = 120
    /// Width of the "plies (X.X%)" cell — sized for "1,234,567
    /// (100.0%)" worst case.
    private static let pliesCellWidth: CGFloat = 130
    /// Width of each W% / D% / L% within-bucket cell.
    private static let breakdownCellWidth: CGFloat = 56
    private static let cellSpacing: CGFloat = 6

    // MARK: - Bucket range labels
    //
    // Range strings appended after each bucket label so the operator
    // can read the cutoff at a glance. Values match
    // `PhaseHistogram.plyBucket` / `materialBucket`, which themselves
    // match `ReplayBuffer.computeBatchStats`. Keep these in sync if
    // any cutoff ever moves.

    private static let plyLabels: [String: String] = [
        "open":  "open (0–20)",
        "early": "early (21–60)",
        "mid":   "mid (61–150)",
        "late":  "late (151–300)",
        "end":   "end (301+)",
    ]
    private static let materialLabels: [String: String] = [
        "open":  "open (≥14)",
        "early": "early (12–13)",
        "mid":   "mid (8–11)",
        "late":  "late (4–7)",
        "end":   "end (≤3)",
    ]

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("1-min emit window")
                .font(.caption2)
                .foregroundStyle(.secondary)

            totalPliesSection
            phaseSection(
                title: "By ply phase",
                labels: Self.plyLabels,
                w: snapshot?.emitWindowPhaseByPlyW ?? .zero,
                d: snapshot?.emitWindowPhaseByPlyD ?? .zero,
                l: snapshot?.emitWindowPhaseByPlyL ?? .zero
            )
            phaseSection(
                title: "By material phase",
                labels: Self.materialLabels,
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
        labels: [String: String],
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

            phaseRow(label: labels["open"]  ?? "open",  bucket: \.open,  w: w, d: d, l: l, totalAll: totalAll)
            phaseRow(label: labels["early"] ?? "early", bucket: \.early, w: w, d: d, l: l, totalAll: totalAll)
            phaseRow(label: labels["mid"]   ?? "mid",   bucket: \.mid,   w: w, d: d, l: l, totalAll: totalAll)
            phaseRow(label: labels["late"]  ?? "late",  bucket: \.late,  w: w, d: d, l: l, totalAll: totalAll)
            phaseRow(label: labels["end"]   ?? "end",   bucket: \.end,   w: w, d: d, l: l, totalAll: totalAll)
        }
    }

    @ViewBuilder
    private func phaseRow(
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

        HStack(spacing: Self.cellSpacing) {
            Text(label)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .frame(width: Self.labelWidth, alignment: .trailing)
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
