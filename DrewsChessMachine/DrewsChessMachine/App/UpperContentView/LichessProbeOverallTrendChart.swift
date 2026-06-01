import Charts
import SwiftUI

/// Two stacked compact line charts plotting the OVERALL 200-puzzle
/// summary across every recorded tick of the Lichess probe watcher
/// (default cadence: every 200 trainer SGD steps).
///
/// Top: mean per-probe `−log p_bookmove` in nats — the cross-entropy
/// of the bookmove evaluated against all 200 puzzles. Lower is
/// better. This is the same loss the trainer minimizes on its replay
/// buffer, computed on these held-out positions, so it's the closest
/// thing the project has to a "test loss" trajectory.
///
/// Bottom: MLE puzzle-Elo fitted by Bradley-Terry on the
/// per-puzzle (rating, correct) outcomes. Higher is better. Only
/// finite-valued samples are plotted; ticks where every puzzle was
/// wrong (−∞) or every puzzle was right (+∞) are dropped from the
/// series so the y-axis doesn't blow up. The set's rating range is
/// 800–1800, so values pin near 800 until the network is solving
/// at least one puzzle.
///
/// X axis is the trainer step (`ChessTrainer.completedTrainSteps`) at
/// which each tick was recorded — the same index the rest of the app's
/// telemetry uses, so the trajectory lines up with steps/positions
/// elsewhere. If any sample lacks a step (e.g. a manual "Probe now"
/// taken before training started, or the steps aren't monotonic) the
/// chart falls back to plotting against tick index and labels the axis
/// accordingly.
///
/// When a comparison snapshot is loaded, a dashed horizontal
/// reference line at the cmp's value is overlaid on each chart so
/// the live trajectory's distance from cmp reads at a glance.
struct LichessProbeOverallTrendChart: View {
    @Bindable var history: LichessProbeHistory
    /// Optional comparison snapshot's overall values, drawn as
    /// dashed horizontal reference lines. nil = no comparison
    /// loaded (don't draw the reference).
    let cmpNll: Double?
    let cmpElo: Double?

    /// X-axis positions (one per sample, parallel to the input array)
    /// plus the axis label. Uses the per-tick trainer step when every
    /// sample carries a non-nil, non-decreasing step; otherwise falls
    /// back to tick index (0-based). Pure + static so it can be unit
    /// tested without a SwiftUI host.
    static func xPositions(
        for samples: [LichessProbeHistory.OverallTickSample]
    ) -> (xs: [Double], label: String) {
        let steps = samples.map(\.trainingStep)
        let allPresent = steps.allSatisfy { $0 != nil }
        var nonDecreasing = true
        if allPresent {
            var prev = Int.min
            for case let s? in steps {
                if s < prev { nonDecreasing = false; break }
                prev = s
            }
        }
        if allPresent && nonDecreasing {
            return (steps.map { Double($0 ?? 0) }, "trainer step")
        }
        return ((0..<samples.count).map(Double.init), "tick #")
    }

    var body: some View {
        let samples = history.overallSeries
        if samples.isEmpty {
            placeholder
        } else {
            let x = Self.xPositions(for: samples)
            VStack(alignment: .leading, spacing: 4) {
                nllChart(samples: samples, xs: x.xs, xLabel: x.label)
                eloChart(samples: samples, xs: x.xs, xLabel: x.label)
            }
        }
    }

    @ViewBuilder
    private var placeholder: some View {
        Text("OVERALL trend — waiting for first tick "
            + "(cadence: every 200 trainer steps; click Probe now for an immediate sample)")
            .font(.system(.caption, design: .monospaced))
            .foregroundStyle(.secondary)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.vertical, 8)
    }

    @ViewBuilder
    private func nllChart(
        samples: [LichessProbeHistory.OverallTickSample],
        xs: [Double],
        xLabel: String
    ) -> some View {
        VStack(alignment: .leading, spacing: 1) {
            HStack(spacing: 8) {
                Text("OVERALL NLL (nats, lower = better)")
                    .font(.system(.caption, design: .monospaced).weight(.semibold))
                latestValueLabel(
                    "live",
                    String(format: "%.3f", samples.last!.meanNegLogProb)
                )
                if let cmp = cmpNll {
                    latestValueLabel(
                        "cmp",
                        String(format: "%.3f", cmp)
                    )
                }
                Spacer()
            }
            Chart {
                ForEach(Array(samples.enumerated()), id: \.offset) { i, s in
                    LineMark(
                        x: .value(xLabel, xs[i]),
                        y: .value("NLL", s.meanNegLogProb)
                    )
                    .foregroundStyle(Color.blue)
                    .lineStyle(StrokeStyle(lineWidth: 1.5))
                }
                if let cmp = cmpNll {
                    RuleMark(y: .value("cmp", cmp))
                        .foregroundStyle(Color.orange.opacity(0.7))
                        .lineStyle(StrokeStyle(lineWidth: 1, dash: [4, 3]))
                        .annotation(position: .top, alignment: .trailing) {
                            Text("cmp")
                                .font(.system(.caption2, design: .monospaced))
                                .foregroundStyle(.orange)
                        }
                }
                // Uniform-over-30-legals reference line (≈ ln 30).
                RuleMark(y: .value("uniform", log(30.0)))
                    .foregroundStyle(Color.gray.opacity(0.45))
                    .lineStyle(StrokeStyle(lineWidth: 1, dash: [2, 3]))
                    .annotation(position: .top, alignment: .trailing) {
                        Text("uniform ~3.4")
                            .font(.system(.caption2, design: .monospaced))
                            .foregroundStyle(.secondary)
                    }
            }
            .frame(height: 110)
            .chartXAxisLabel(position: .bottom, alignment: .leading) {
                Text(xLabel)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
        }
    }

    @ViewBuilder
    private func eloChart(
        samples: [LichessProbeHistory.OverallTickSample],
        xs: [Double],
        xLabel: String
    ) -> some View {
        // Only finite-valued samples are plotted. If the entire
        // series is pinned at the floor / ceiling sentinel, the
        // chart renders empty with a clarifying caption. Keep each
        // surviving sample's original index so it maps back to the
        // matching x-position.
        let finite = samples.enumerated().filter { $0.element.puzzleElo.isFinite }
        VStack(alignment: .leading, spacing: 1) {
            HStack(spacing: 8) {
                Text("OVERALL puzzle-Elo (higher = better, 800–1800 set range)")
                    .font(.system(.caption, design: .monospaced).weight(.semibold))
                latestValueLabel("live", Self.formatElo(samples.last!.puzzleElo))
                if let cmp = cmpElo {
                    latestValueLabel("cmp", Self.formatElo(cmp))
                }
                Spacer()
            }
            if finite.isEmpty {
                Text("All samples at floor (every puzzle wrong) — Elo MLE is unbounded below.")
                    .font(.system(.caption2, design: .monospaced))
                    .foregroundStyle(.secondary)
                    .frame(height: 60, alignment: .center)
                    .frame(maxWidth: .infinity, alignment: .leading)
            } else {
                Chart {
                    ForEach(finite, id: \.offset) { idx, s in
                        LineMark(
                            x: .value(xLabel, xs[idx]),
                            y: .value("pElo", s.puzzleElo)
                        )
                        .foregroundStyle(Color.green)
                        .lineStyle(StrokeStyle(lineWidth: 1.5))
                    }
                    if let cmp = cmpElo, cmp.isFinite {
                        RuleMark(y: .value("cmp", cmp))
                            .foregroundStyle(Color.orange.opacity(0.7))
                            .lineStyle(StrokeStyle(lineWidth: 1, dash: [4, 3]))
                            .annotation(position: .top, alignment: .trailing) {
                                Text("cmp")
                                    .font(.system(.caption2, design: .monospaced))
                                    .foregroundStyle(.orange)
                            }
                    }
                }
                .frame(height: 110)
            }
        }
    }

    @ViewBuilder
    private func latestValueLabel(_ label: String, _ value: String) -> some View {
        HStack(spacing: 2) {
            Text(label)
                .font(.system(.caption2, design: .monospaced))
                .foregroundStyle(.secondary)
            Text(value)
                .font(.system(.caption, design: .monospaced).weight(.semibold))
        }
    }

    /// Mirror of `LichessProbeDetailView.formatPuzzleElo` without
    /// the "pElo" prefix — used in the chart header's value chip.
    private static func formatElo(_ elo: Double) -> String {
        if elo.isNaN { return "—" }
        if elo == -.infinity { return "<floor" }
        if elo == .infinity { return ">ceil" }
        return String(format: "%.0f", elo)
    }
}
