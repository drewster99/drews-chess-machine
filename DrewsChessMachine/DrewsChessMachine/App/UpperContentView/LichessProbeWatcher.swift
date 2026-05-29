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

    /// Begin the periodic loop. Two-phase startup:
    ///  1. Poll every 5 seconds until the network needed by the current
    ///     `probeNetworkTarget` is built (training has started).
    ///  2. Sleep 60 seconds — a warm-up so the first tick lands well
    ///     into training rather than right at the launch instant when
    ///     the network is still random / tunable.
    /// After the first tick, the regular `intervalSec` cadence kicks in.
    /// Re-entrant: a second `start()` while running is a no-op.
    func start() {
        guard task == nil else { return }
        task = Task { [weak self, intervalSec] in
            await self?.waitForFirstTickConditions()
            if Task.isCancelled { return }

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

    /// Poll-until-ready + 60s warm-up before the first tick. Same
    /// shape as `TacticalProbeWatcher.waitForFirstTickConditions`,
    /// just keyed on the Lichess-dedicated inference network.
    @MainActor
    private func waitForFirstTickConditions() async {
        while !Task.isCancelled {
            if canTickNow() { break }
            do {
                try await Task.sleep(nanoseconds: 5_000_000_000)
            } catch {
                return
            }
        }
        if Task.isCancelled { return }
        do {
            try await Task.sleep(nanoseconds: 60_000_000_000)
        } catch {
            return
        }
    }

    /// Whether `tickOnce` would currently be able to run a probe.
    /// Mirrors the guards inside `tickOnce`.
    @MainActor
    private func canTickNow() -> Bool {
        guard let session = sessionController else { return false }
        switch session.probeNetworkTarget {
        case .champion:
            return session.network != nil
        case .candidate:
            return session.trainer != nil
                && session.lichessProbeInferenceNetwork != nil
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
            // Use the dedicated `lichessProbeInferenceNetwork`. Each
            // probe consumer — `fireCandidateProbeIfNeeded`,
            // `TacticalProbeWatcher`, `LichessProbeWatcher` — now owns
            // its own inference network so none can overwrite another's
            // weight snapshot mid-cycle (which was the original race:
            // one tick's snapshot landing inside another tick's
            // loadWeights would leave the wrong weights in place for
            // the remainder of the first tick's probe loop). A single
            // watcher's tickOnce is serial against itself.
            guard let trainer = session.trainer,
                  let probeNet = session.lichessProbeInferenceNetwork else { return }
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
        var allResults: [ProbeResult] = []
        allResults.reserveCapacity(probes.count)
        for probe in probes {
            let r = await TacticalProbeRunner.run(probe, against: net)
            allResults.append(r)
        }

        let aggregates = LichessProbeHistory.aggregates(from: allResults)
        let modelLabel = net.identifier?.description ?? "<no-id>"
        history.record(aggregates, allResults: allResults, modelLabel: modelLabel)
        logTickSummary(aggregates, modelLabel: modelLabel)
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
