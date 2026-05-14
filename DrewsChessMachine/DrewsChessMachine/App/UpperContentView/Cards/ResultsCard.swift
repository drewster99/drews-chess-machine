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
struct ResultsCard: View {
    let snapshot: ParallelWorkerStatsBox.Snapshot?

    private static let dash = "—"
    private static let labelWidth: CGFloat = 110
    /// Wide enough to fit a 7-digit count plus a 5-character percent
    /// suffix at callout-monospaced size — e.g. "1,234,567 (99.9%)"
    /// (~17 chars · ~7.8 pt = ~133 pt) without `.lineLimit(1)` ever
    /// truncating mid-cell at realistic session sizes.
    private static let valueWidth: CGFloat = 140

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Results")
                .font(.caption2)
                .foregroundStyle(.secondary)

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
                overall: countAndPct(checkmateOverall, of: overallTotal),
                kept: countAndPct(checkmateKept, of: keptTotal),
                indent: false
            )
            row(
                label: "white",
                overall: countAndPct(whiteOverall, of: overallTotal),
                kept: countAndPct(whiteKept, of: keptTotal),
                indent: true
            )
            row(
                label: "black",
                overall: countAndPct(blackOverall, of: overallTotal),
                kept: countAndPct(blackKept, of: keptTotal),
                indent: true
            )

            // Draw (filtered) — parent + 4 indented children.
            row(
                label: "Draw",
                overall: countAndPct(drawOverall, of: overallTotal),
                kept: countAndPct(drawKept, of: keptTotal),
                indent: false
            )
            row(
                label: "stalemate",
                overall: countAndPct(stalemateOverall, of: overallTotal),
                kept: countAndPct(stalemateKept, of: keptTotal),
                indent: true
            )
            row(
                label: "insufficient",
                overall: countAndPct(insufficientOverall, of: overallTotal),
                kept: countAndPct(insufficientKept, of: keptTotal),
                indent: true
            )
            row(
                label: "50-move rule",
                overall: countAndPct(fiftyMoveOverall, of: overallTotal),
                kept: countAndPct(fiftyMoveKept, of: keptTotal),
                indent: true
            )
            row(
                label: "threefold rep",
                overall: countAndPct(threefoldOverall, of: overallTotal),
                kept: countAndPct(threefoldKept, of: keptTotal),
                indent: true
            )
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 6)
        .background(Color.secondary.opacity(0.10))
        .clipShape(RoundedRectangle(cornerRadius: 6))
    }

    // MARK: - Layout

    /// Indent flag: parent rows render in the card's primary value
    /// weight (semibold callout); indented sub-rows step down to a
    /// lighter `.regular` weight at the same size so the hierarchy
    /// reads at a glance without changing column alignment.
    @ViewBuilder
    private func row(label: String, overall: String, kept: String, indent: Bool) -> some View {
        HStack(spacing: 0) {
            Text(indent ? "  \(label)" : label)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .frame(width: Self.labelWidth, alignment: .trailing)
            Text(overall)
                .font(.system(.callout, design: .monospaced).weight(indent ? .regular : .semibold))
                .monospacedDigit()
                .lineLimit(1)
                .frame(width: Self.valueWidth, alignment: .trailing)
            Text(kept)
                .font(.system(.callout, design: .monospaced).weight(indent ? .regular : .semibold))
                .monospacedDigit()
                .lineLimit(1)
                .frame(width: Self.valueWidth, alignment: .trailing)
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

    /// "47,351 (19.1%)" or "—" when either count or denominator is zero.
    private func countAndPct(_ count: Int, of total: Int) -> String {
        guard total > 0 else { return Self.dash }
        if count == 0 { return "0" }
        let pct = Double(count) / Double(total) * 100
        let countStr = count.formatted(.number.grouping(.automatic))
        return String(format: "%@ (%.1f%%)", countStr, pct)
    }
}
