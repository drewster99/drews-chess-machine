import Foundation

/// Periodic driver for the 200-puzzle Lichess probe set. Same
/// life-of-session shape as `TacticalProbeWatcher` — one task that
/// fires `intervalSec` apart, plus an on-demand `triggerOnce()` for the
/// monitor's "Probe now" button.
///
/// Per tick: snapshot the trainer's current weights into the dedicated
/// probe-inference network (or read the live champion, depending on
/// `session.probeNetworkTarget`), then run all 200 puzzles serially
/// through `TacticalProbeRunner.run` and fold the 200 results into 8
/// per-theme aggregates that get appended to `LichessProbeHistory`.
///
/// Cost note: 200 probes × 2 forward passes each = 400 forward passes
/// per tick. At the 30-minute default cadence that's ~13 single-position
/// forward passes per minute — well under the per-ply rate the batched
/// self-play evaluator already sustains. We deliberately do NOT log
/// per-probe lines for the periodic ticks (only a one-line `[TACTICAL-
/// LICHESS]` summary), to keep the session log readable; the manual
/// "Run Lichess Probe" menu item produces a denser per-theme breakdown
/// instead, see `SessionController.runLichessProbe()`.
///
/// @MainActor isolated — same pattern as the small-set watcher.
@MainActor
final class LichessProbeWatcher {
    private weak var sessionController: SessionController?
    private let history: LichessProbeHistory
    private let intervalSec: TimeInterval
    private var task: Task<Void, Never>?

    init(
        sessionController: SessionController,
        history: LichessProbeHistory,
        intervalSec: TimeInterval = 30.0 * 60.0
    ) {
        self.sessionController = sessionController
        self.history = history
        self.intervalSec = intervalSec
    }

    /// Begin the periodic loop. Runs an immediate first tick so the
    /// monitor populates within a couple of seconds of opening (200
    /// probes take a few seconds total, not the few-hundred-ms of the
    /// 9-probe set). Re-entrant: a second `start()` while running is
    /// a no-op.
    func start() {
        guard task == nil else { return }
        task = Task { [weak self, intervalSec] in
            await self?.tickOnce()
            while !Task.isCancelled {
                let nanos = UInt64(intervalSec * 1_000_000_000)
                do {
                    try await Task.sleep(nanoseconds: nanos)
                } catch {
                    return
                }
                if Task.isCancelled { return }
                await self?.tickOnce()
            }
        }
    }

    /// Cancel the loop. In-flight probes finish (they're cheap and
    /// already submitted to the network's executionQueue); the next
    /// tick won't fire.
    func stop() {
        task?.cancel()
        task = nil
    }

    /// Fire one cycle on demand, independent of the periodic loop.
    /// Used by the monitor's "Probe now" button and by the manual
    /// "Run Lichess Probe" menu item's foreground hook (when the
    /// session also wants the history updated as a side effect).
    func triggerOnce() {
        Task { [weak self] in
            await self?.tickOnce()
        }
    }

    /// One tick: pick network, run 200 probes, aggregate, record.
    /// Mirrors `TacticalProbeWatcher.tickOnce()` for the network-target
    /// selection and trainer-snapshot path so the two watchers behave
    /// identically with respect to the existing
    /// `probeNetworkTarget` toggle.
    private func tickOnce() async {
        guard let session = sessionController else { return }
        let target = session.probeNetworkTarget

        let net: ChessMPSNetwork
        switch target {
        case .champion:
            guard let championNet = session.network else { return }
            net = championNet
        case .candidate:
            guard let trainer = session.trainer,
                  let probeNet = session.probeInferenceNetwork else { return }
            do {
                let weights = try await trainer.network.exportWeights()
                try await probeNet.loadWeights(weights)
            } catch {
                SessionLogger.shared.log(
                    "[TACTICAL-LICHESS] monitor trainer-snapshot failed: \(error.localizedDescription)"
                )
                return
            }
            probeNet.identifier = trainer.identifier
            net = probeNet
        }

        // Run the full battery. 200 serial forward passes — the
        // batched evaluator isn't wired through `TacticalProbeRunner`
        // because per-probe top-5 / entropy bookkeeping wants the raw
        // logits per position rather than a folded batch result.
        let probes = LichessProbeData.largeSet
        var resultsByCategory: [ProbeCategory: [ProbeResult]] = [:]
        var allResults: [ProbeResult] = []
        allResults.reserveCapacity(probes.count)
        for probe in probes {
            let r = await TacticalProbeRunner.run(probe, against: net)
            resultsByCategory[probe.category, default: []].append(r)
            allResults.append(r)
        }

        let aggregates = foldAggregates(resultsByCategory)
        let modelLabel = net.identifier?.description ?? "<no-id>"
        history.record(aggregates, allResults: allResults, modelLabel: modelLabel)
        logTickSummary(aggregates, modelLabel: modelLabel)
    }

    /// Fold per-category arrays of `ProbeResult` into one
    /// `LichessProbeHistory.Aggregate` per category. Verdicts:
    ///   - `argmaxCorrect` counts `.correctAndConfident` + `.correctButFlat`
    ///   - `top5Correct`   adds `.correctInTop5` on top
    ///   - `errored`       counts `.error`
    /// `.wrong` falls in none.
    ///
    /// Also accumulates `sumExpectedProb` over every probe (errored
    /// probes have `expectedProb = 0` by construction) and
    /// `sumExpectedRank` / `countWithRank` over probes that produced a
    /// non-nil rank — the source of the monitor's continuous-valued
    /// AVG PROB and AVG RANK columns.
    private func foldAggregates(
        _ resultsByCategory: [ProbeCategory: [ProbeResult]]
    ) -> [LichessProbeHistory.Aggregate] {
        var out: [LichessProbeHistory.Aggregate] = []
        out.reserveCapacity(resultsByCategory.count)
        for (cat, results) in resultsByCategory {
            var argmaxCorrect = 0
            var top5Correct = 0
            var errored = 0
            var sumProb: Float = 0
            var sumRank = 0
            var countRank = 0
            for r in results {
                sumProb += r.expectedProb
                if let rank = r.expectedRank {
                    sumRank += rank
                    countRank += 1
                }
                switch r.verdict {
                case .correctAndConfident, .correctButFlat:
                    argmaxCorrect += 1
                    top5Correct += 1
                case .correctInTop5:
                    top5Correct += 1
                case .wrong:
                    break
                case .error:
                    errored += 1
                }
            }
            out.append(LichessProbeHistory.Aggregate(
                theme: cat,
                total: results.count,
                argmaxCorrect: argmaxCorrect,
                top5Correct: top5Correct,
                errored: errored,
                sumExpectedProb: sumProb,
                sumExpectedRank: sumRank,
                countWithRank: countRank
            ))
        }
        return out
    }

    /// One `[TACTICAL-LICHESS]` summary line per tick — total score
    /// plus per-theme `correct/total` so the log is grep-friendly.
    private func logTickSummary(
        _ aggregates: [LichessProbeHistory.Aggregate],
        modelLabel: String
    ) {
        var totalCorrect = 0
        var totalProbes = 0
        var perThemeParts: [String] = []

        // Iterate in a stable order (rawValue alphabetical) so the
        // log line is reproducible across ticks.
        for agg in aggregates.sorted(by: { $0.theme.rawValue < $1.theme.rawValue }) {
            totalCorrect += agg.argmaxCorrect
            totalProbes += agg.total
            perThemeParts.append(
                "\(agg.theme.rawValue)=\(agg.argmaxCorrect)/\(agg.total)"
            )
        }
        let pct = totalProbes > 0
            ? String(format: "%.1f", 100.0 * Double(totalCorrect) / Double(totalProbes))
            : "0.0"
        SessionLogger.shared.log(
            "[TACTICAL-LICHESS] tick \(totalCorrect)/\(totalProbes) (\(pct)%) model=\(modelLabel) "
            + perThemeParts.joined(separator: " ")
        )
    }
}
