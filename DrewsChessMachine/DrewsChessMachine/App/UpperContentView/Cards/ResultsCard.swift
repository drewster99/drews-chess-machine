import SwiftUI

/// Self Play Results card — Overall vs Kept side-by-side, with
/// the Checkmate / Draw breakdown sub-rows indented underneath.
/// Replaces the per-line text block that used to live inside
/// `playAndTrainStatsText`.
///
/// "Overall" denominator = `selfPlayGames`; "Kept" denominator =
/// `emittedGames`. Decisive sub-rows (white/black checkmate) read
/// the same numerator in both columns because decisives bypass the
/// draw-keep filter — only their *percentage* differs (smaller
/// denominator under Kept, so the share rises). Draw sub-rows
/// diverge in both numerator and percentage when the filter is
/// active.
///
/// Visual style matches `SelfPlayStatsCard` and the upper status bar.
///
/// Each value cell is a 2-column sub-grid: a right-aligned count and
/// a right-aligned percent. This keeps both the digits column and
/// the percent column visually aligned across rows. Rendering them
/// as a single combined string ("48,801 (10.2%)") would right-align
/// only the closing paren; the count's right edge would float by the
/// percent string's variable width and the comma-separated digits
/// would never line up between rows.
struct ResultsCard: View {
    let snapshot: ParallelWorkerStatsBox.Snapshot?

    private static let dash = "—"
    private static let labelWidth: CGFloat = 110
    /// Width of the count sub-cell. Sized for a 7-digit count
    /// ("1,234,567") at callout-monospaced — about 9 chars · ~7.8 pt
    /// = ~70 pt. Comfortable headroom for realistic session sizes.
    private static let countWidth: CGFloat = 78
    /// Width of the percent sub-cell. Sized for the worst-case
    /// "(100.0%)" (8 chars · ~7.8 pt = ~62 pt).
    private static let pctWidth: CGFloat = 64
    /// Spacing between the count and percent sub-cells inside each
    /// value cell.
    private static let countPctSpacing: CGFloat = 6

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Results")
                .font(.caption2)
                .foregroundStyle(.secondary)

            // Header strip — column titles aligned over the combined
            // (count + percent) value cell width.
            HStack(spacing: 0) {
                Text("")
                    .frame(width: Self.labelWidth, alignment: .trailing)
                Text("Overall")
                    .frame(width: Self.valueWidth, alignment: .trailing)
                Text("Kept")
                    .frame(width: Self.valueWidth, alignment: .trailing)
            }
            .font(.caption2)
            .foregroundStyle(.secondary)
            .padding(.bottom, 1)

            // Checkmate (decisive) — parent + 2 indented children.
            row(
                label: "Checkmate",
                overall: cells(checkmateOverall, of: overallTotal),
                kept: cells(checkmateKept, of: keptTotal),
                indent: false
            )
            row(
                label: "white",
                overall: cells(whiteOverall, of: overallTotal),
                kept: cells(whiteKept, of: keptTotal),
                indent: true
            )
            row(
                label: "black",
                overall: cells(blackOverall, of: overallTotal),
                kept: cells(blackKept, of: keptTotal),
                indent: true
            )

            // Draw (filtered) — parent + 4 indented children.
            row(
                label: "Draw",
                overall: cells(drawOverall, of: overallTotal),
                kept: cells(drawKept, of: keptTotal),
                indent: false
            )
            row(
                label: "stalemate",
                overall: cells(stalemateOverall, of: overallTotal),
                kept: cells(stalemateKept, of: keptTotal),
                indent: true
            )
            row(
                label: "insufficient",
                overall: cells(insufficientOverall, of: overallTotal),
                kept: cells(insufficientKept, of: keptTotal),
                indent: true
            )
            row(
                label: "50-move rule",
                overall: cells(fiftyMoveOverall, of: overallTotal),
                kept: cells(fiftyMoveKept, of: keptTotal),
                indent: true
            )
            row(
                label: "threefold rep",
                overall: cells(threefoldOverall, of: overallTotal),
                kept: cells(threefoldKept, of: keptTotal),
                indent: true
            )
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 6)
        .background(Color.secondary.opacity(0.10))
        .clipShape(RoundedRectangle(cornerRadius: 6))
    }

    // MARK: - Layout

    /// Combined (count + percent) cell width — the header titles
    /// "Overall" / "Kept" align over this.
    private static var valueWidth: CGFloat {
        countWidth + countPctSpacing + pctWidth
    }

    /// One value cell: count right-aligned in its sub-cell, percent
    /// right-aligned in its sub-cell. Both monospaced so digits stack
    /// vertically across rows.
    @ViewBuilder
    private func valueCell(_ v: ValueCells, indent: Bool) -> some View {
        HStack(spacing: Self.countPctSpacing) {
            Text(v.count)
                .font(.system(.callout, design: .monospaced).weight(indent ? .regular : .semibold))
                .monospacedDigit()
                .lineLimit(1)
                .frame(width: Self.countWidth, alignment: .trailing)
            Text(v.pct)
                .font(.system(.callout, design: .monospaced).weight(indent ? .regular : .semibold))
                .monospacedDigit()
                .lineLimit(1)
                .frame(width: Self.pctWidth, alignment: .trailing)
        }
    }

    /// Indent flag: parent rows render in the card's primary value
    /// weight (semibold callout); indented sub-rows step down to a
    /// lighter `.regular` weight at the same size so the hierarchy
    /// reads at a glance without changing column alignment.
    @ViewBuilder
    private func row(label: String, overall: ValueCells, kept: ValueCells, indent: Bool) -> some View {
        HStack(spacing: 0) {
            Text(indent ? "  \(label)" : label)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .frame(width: Self.labelWidth, alignment: .trailing)
            valueCell(overall, indent: indent)
            valueCell(kept, indent: indent)
        }
    }

    // MARK: - Numerators

    private var overallTotal: Int { snapshot?.selfPlayGames ?? 0 }
    private var keptTotal: Int { snapshot?.emittedGames ?? 0 }

    private var whiteOverall: Int { snapshot?.whiteCheckmates ?? 0 }
    private var blackOverall: Int { snapshot?.blackCheckmates ?? 0 }
    private var checkmateOverall: Int { whiteOverall + blackOverall }

    private var whiteKept: Int { snapshot?.emittedWhiteCheckmates ?? 0 }
    private var blackKept: Int { snapshot?.emittedBlackCheckmates ?? 0 }
    private var checkmateKept: Int { whiteKept + blackKept }

    private var stalemateOverall: Int { snapshot?.stalemates ?? 0 }
    private var insufficientOverall: Int { snapshot?.insufficientMaterialDraws ?? 0 }
    private var fiftyMoveOverall: Int { snapshot?.fiftyMoveDraws ?? 0 }
    private var threefoldOverall: Int { snapshot?.threefoldRepetitionDraws ?? 0 }
    private var drawOverall: Int { stalemateOverall + insufficientOverall + fiftyMoveOverall + threefoldOverall }

    private var stalemateKept: Int { snapshot?.emittedStalemates ?? 0 }
    private var insufficientKept: Int { snapshot?.emittedInsufficientMaterialDraws ?? 0 }
    private var fiftyMoveKept: Int { snapshot?.emittedFiftyMoveDraws ?? 0 }
    private var threefoldKept: Int { snapshot?.emittedThreefoldRepetitionDraws ?? 0 }
    private var drawKept: Int { stalemateKept + insufficientKept + fiftyMoveKept + threefoldKept }

    // MARK: - Formatters

    /// Two strings: the count cell ("47,351", "0", or "—") and the
    /// percent cell ("(19.1%)" or empty when the count cell is "0"
    /// or "—"). Held as a struct so the view tree stays
    /// declarative — the row builder picks the right sub-cells out.
    private struct ValueCells {
        let count: String
        let pct: String
    }

    private func cells(_ count: Int, of total: Int) -> ValueCells {
        guard total > 0 else { return ValueCells(count: Self.dash, pct: "") }
        if count == 0 { return ValueCells(count: "0", pct: "") }
        let pct = Double(count) / Double(total) * 100
        let countStr = count.formatted(.number.grouping(.automatic))
        return ValueCells(count: countStr, pct: String(format: "(%.1f%%)", pct))
    }
}
