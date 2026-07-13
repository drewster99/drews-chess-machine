import Foundation

/// Formats the periodic `[VS-UCI-STATS]` block for train-vs-UCI runs:
/// first one **per-kind summary line** (aggregating that kind's `n`
/// instances), then one **per-instance line** per engine process, so a
/// wedged or slow instance inside a kind is visible at a glance.
///
/// Each line reports cumulative games / plies / W-L-D (from the
/// trainer's perspective) plus games/hr and plies/hr over the window
/// since the previous emit. Pure and side-effect-free so the format is
/// unit-testable without a live run; the runner owns the cadence and
/// the previous-snapshot bookkeeping.
enum TrainVsUciStatsFormatter {

    /// Build the stats block. `previous` is the snapshot from the prior
    /// emit (empty on the first emit — rates then cover the whole run),
    /// `intervalSec` the wall-clock seconds the rate window spans.
    /// Instance order within the block follows `current`'s order; kinds
    /// appear in first-appearance order.
    static func lines(
        current: [TrainVsUciDriver.SlotStats],
        previous: [TrainVsUciDriver.SlotStats],
        intervalSec: Double
    ) -> [String] {
        guard !current.isEmpty else { return [] }
        let window = max(0.001, intervalSec)

        // Previous counts keyed by instance label (labels are unique and
        // stable for the life of the run).
        var prevByLabel: [String: TrainVsUciDriver.SlotStats] = [:]
        for p in previous { prevByLabel[p.instanceLabel] = p }

        // Group by kind, preserving first-appearance order.
        var kindOrder: [String] = []
        var byKind: [String: [TrainVsUciDriver.SlotStats]] = [:]
        for s in current {
            if byKind[s.kind] == nil { kindOrder.append(s.kind) }
            byKind[s.kind, default: []].append(s)
        }

        // Per-hour rate over the window, humanized with a magnitude suffix
        // (M / k) so the per-kind summaries read in millions — e.g. a
        // ~2900 plies/sec pool prints `p/hr=10.4M` — while the much smaller
        // per-instance lines stay legible as `k`. See `humanizedPerHour`.
        func rate(_ delta: Int) -> String {
            humanizedPerHour(Double(delta) / window * 3600)
        }

        var out: [String] = []
        // Per-kind summary lines.
        for kind in kindOrder {
            let group = byKind[kind] ?? []
            let games = group.reduce(0) { $0 + $1.gamesCompleted }
            let plies = group.reduce(0) { $0 + $1.pliesPlayed }
            let wins = group.reduce(0) { $0 + $1.trainerWins }
            let losses = group.reduce(0) { $0 + $1.trainerLosses }
            let draws = group.reduce(0) { $0 + $1.draws }
            let aborted = group.reduce(0) { $0 + $1.aborted }
            let capDropped = group.reduce(0) { $0 + $1.capDropped }
            let prevGames = group.reduce(0) { $0 + (prevByLabel[$1.instanceLabel]?.gamesCompleted ?? 0) }
            let prevPlies = group.reduce(0) { $0 + (prevByLabel[$1.instanceLabel]?.pliesPlayed ?? 0) }
            out.append(
                "[VS-UCI-STATS] \(kind) n=\(group.count):"
                + " games=\(games) plies=\(plies)"
                + " g/hr=\(rate(games - prevGames)) p/hr=\(rate(plies - prevPlies))"
                + " W-L-D=\(wins)-\(losses)-\(draws)"
                + (aborted > 0 ? " aborted=\(aborted)" : "")
                + (capDropped > 0 ? " capDropped=\(capDropped)" : "")
            )
        }
        // Per-instance breakdown lines.
        for s in current {
            let prev = prevByLabel[s.instanceLabel]
            out.append(
                "[VS-UCI-STATS]   \(s.instanceLabel):"
                + " games=\(s.gamesCompleted) plies=\(s.pliesPlayed)"
                + " g/hr=\(rate(s.gamesCompleted - (prev?.gamesCompleted ?? 0)))"
                + " p/hr=\(rate(s.pliesPlayed - (prev?.pliesPlayed ?? 0)))"
                + " W-L-D=\(s.trainerWins)-\(s.trainerLosses)-\(s.draws)"
                + (s.aborted > 0 ? " aborted=\(s.aborted)" : "")
                + (s.capDropped > 0 ? " capDropped=\(s.capDropped)" : "")
            )
        }
        return out
    }

    /// Formats a per-hour throughput rate with a magnitude suffix and one
    /// decimal: `≥1e6 → "10.4M"`, `≥1e3 → "394.5k"`, else a bare integer.
    /// Keeps the millions-scale pool summaries reading "in millions per
    /// hour" without an unwieldy 8-digit integer, while per-instance rates
    /// fall back to `k` (or raw) at their smaller scale.
    static func humanizedPerHour(_ perHour: Double) -> String {
        let magnitude = abs(perHour)
        if magnitude >= 1_000_000 {
            return String(format: "%.1fM", perHour / 1_000_000)
        }
        if magnitude >= 1_000 {
            return String(format: "%.1fk", perHour / 1_000)
        }
        return String(format: "%.0f", perHour)
    }
}
