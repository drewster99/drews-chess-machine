import Foundation

/// Step-triggered driver for the 200-puzzle Lichess probe set. Same
/// life-of-session shape as `TacticalProbeWatcher` — one task that
/// fires every `triggerEverySteps` trainer SGD steps, plus an
/// on-demand `triggerOnce()` for the monitor's "Probe now" button.
///
/// Per tick: snapshot the trainer's current weights into the dedicated
/// probe-inference network (or read the live champion, depending on
/// `session.probeNetworkTarget`), then run all 200 puzzles serially
/// through `TacticalProbeRunner.run` and fold the 200 results into 8
/// per-theme aggregates that get appended to `LichessProbeHistory`.
///
/// Cadence rationale: 30-minute time-based ticks decoupled the probe
/// from training progress — at fast-warmup early steps the probe
/// would lag the rapidly-shifting weights, and during slow / paused
/// training it would fire on a stale snapshot. Tying the cadence to
/// trainer steps (default 400) keeps probe density tracking actual
/// learning progress, so each chart point represents a fixed amount
/// of gradient applied rather than a fixed amount of wall-clock time.
///
/// Cost note: 200 probes × 2 forward passes each = 400 forward passes
/// per tick. At the default 400-step cadence and ~2.5 trainer steps/s
/// that's a tick every ~160 s (roughly 540 ticks/day at sustained
/// training, vs. 48 ticks/day under the prior 30-minute time-based
/// cadence — about 11× denser sampling, which is the whole point of
/// switching to step-based triggering). Forward-pass load is still
/// well under the per-ply rate the batched self-play evaluator
/// already sustains. We deliberately do NOT log per-probe lines for
/// the periodic ticks (only a one-line `[TACTICAL-LICHESS]` summary)
/// to keep the session log readable; the manual "Run Lichess Probe"
/// menu item produces a denser per-theme breakdown instead — see
/// `SessionController.runLichessProbe()`.
///
/// @MainActor isolated — same pattern as the small-set watcher.
@MainActor
final class LichessProbeWatcher {
    private weak var sessionController: SessionController?
    private let history: LichessProbeHistory
    /// Fire one tick every time the trainer's `completedTrainSteps`
    /// has advanced by this many steps since the last tick.
    private let triggerEverySteps: Int
    /// Polling cadence for the step-watch loop. Faster than the
    /// expected tick interval (~160s at 400 steps and ~2.5 steps/s)
    /// so a crossing is detected within a few seconds. Smaller
    /// values are wasted CPU on a sleeping task; larger values
    /// would smear the tick boundary across multiple chart points.
    private let pollIntervalSec: TimeInterval = 2.0
    private var task: Task<Void, Never>?

    init(
        sessionController: SessionController,
        history: LichessProbeHistory,
        triggerEverySteps: Int = 400
    ) {
        self.sessionController = sessionController
        self.history = history
        self.triggerEverySteps = max(1, triggerEverySteps)
    }

    /// Begin the step-watching loop. Three-phase startup:
    ///  1. Poll every 5 seconds until the network needed by the
    ///     current `probeNetworkTarget` is built (training has
    ///     started).
    ///  2. Sleep 60 seconds — a warm-up so the first tick lands
    ///     past the launch instant when the network is still
    ///     random / tunable.
    ///  3. Fire one baseline tick so the chart has a first data
    ///     point, then enter the step-watch loop: every
    ///     `pollIntervalSec` seconds, read the trainer step count
    ///     and fire when it's advanced by `triggerEverySteps`.
    /// Re-entrant: a second `start()` while running is a no-op.
    func start() {
        guard task == nil else { return }
        task = Task { [weak self] in
            await self?.waitForFirstTickConditions()
            if Task.isCancelled { return }

            // Baseline tick + record the step the first tick was
            // anchored at, so the cadence stays "every N steps from
            // first tick" rather than "every N steps from 0" (which
            // could spam several initial ticks if startup conditions
            // delay the watcher past N steps).
            await self?.tickOnce()
            var lastTriggeredStep = await self?.currentTrainerStep() ?? 0

            while !Task.isCancelled {
                do {
                    let nanos = UInt64(
                        (self?.pollIntervalSec ?? 2.0) * 1_000_000_000
                    )
                    try await Task.sleep(nanoseconds: nanos)
                } catch {
                    return
                }
                if Task.isCancelled { return }
                guard let self else { return }
                guard let step = await self.currentTrainerStep() else { continue }
                // Trainer rebuild / fresh session can roll step back —
                // re-anchor so we don't lock out probes by holding a
                // historical high-water mark.
                if step < lastTriggeredStep {
                    lastTriggeredStep = step
                    continue
                }
                if step - lastTriggeredStep >= self.triggerEverySteps {
                    await self.tickOnce()
                    // Snap to the most recent multiple-of-N boundary
                    // at or below `step`, so a slow tick (the 200
                    // forward passes block the loop for a few seconds)
                    // doesn't accumulate "missed" boundaries and
                    // double-fire. Worst case the chart point's step
                    // anchor is N-1 steps behind the snap; that's
                    // already implicit in the cadence quantization.
                    lastTriggeredStep = step - (step % self.triggerEverySteps)
                }
            }
        }
    }

    /// Convenience accessor used by the step-watch loop. Returns nil
    /// when no trainer is yet attached to the session (e.g. between
    /// Play-and-Train stop and restart).
    @MainActor
    private func currentTrainerStep() -> Int? {
        sessionController?.trainer?.completedTrainSteps
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
        // Snapshot the four progress fields the JSON export carries —
        // captured at tick time (not export time) so a later
        // "Export latest…" reports values consistent with the probed
        // weights even if many minutes of training have happened
        // since. Each is nil-safe at its own scope: trainer-step needs
        // a live trainer, positions-trained derives from it, the
        // checkpoint-controller fields need an attached controller.
        let trainingStep = session.trainer?.completedTrainSteps
        let positionsTrained = trainingStep.map {
            $0 * TrainingParameters.shared.trainingBatchSize
        }
        let activeTrainingSec = session.checkpoint?.cumulativeActiveTrainingSec
        let arenaCount = session.tournamentHistory.count
        let promotionCount = session.tournamentHistory.lazy.filter { $0.promoted }.count
        history.record(
            aggregates,
            allResults: allResults,
            modelLabel: modelLabel,
            trainingStep: trainingStep,
            positionsTrained: positionsTrained,
            activeTrainingSec: activeTrainingSec,
            arenaCount: arenaCount,
            promotionCount: promotionCount
        )
        logTickSummary(
            aggregates,
            allResults: allResults,
            modelLabel: modelLabel,
            trainingStep: trainingStep
        )
    }

    /// One `[TACTICAL-LICHESS]` summary line per tick — overall
    /// argmax / top-5 / avg-prob / avg-rank / NLL / puzzle-Elo, then
    /// the per-theme `correct/total` breakdown. Grep-friendly,
    /// reproducible across ticks (themes sorted alphabetically by
    /// rawValue). Mirrors the OVERALL band of the Detail window so
    /// the session log carries the same metrics the UI shows.
    private func logTickSummary(
        _ aggregates: [LichessProbeHistory.Aggregate],
        allResults: [ProbeResult],
        modelLabel: String,
        trainingStep: Int?
    ) {
        let overall = LichessProbeOverallSummary(folding: aggregates)
        let totalCorrect = overall.argmaxCorrect
        let top5Correct = overall.top5Correct
        let totalProbes = overall.totalProbes
        let pct: (Int) -> String = { num in
            totalProbes > 0
                ? String(format: "%.1f", 100.0 * Double(num) / Double(totalProbes))
                : "0.0"
        }
        let avgProbStr = String(format: "%.3f", overall.avgExpectedProb)
        let avgRankStr = overall.avgExpectedRank.map { String(format: "%.2f", $0) }
            ?? "--"
        let nllStr = String(format: "%.3f", overall.meanNegLogProb)
        let pairs: [(rating: Int, correct: Bool)] = allResults.compactMap {
            guard let meta = LichessProbeData.metadata[$0.probe.name] else { return nil }
            let isArgmaxCorrect = $0.verdict == .correctAndConfident
                || $0.verdict == .correctButFlat
            return (rating: meta.rating, correct: isArgmaxCorrect)
        }
        let elo = LichessProbeHistory.mlePuzzleElo(pairs: pairs)
        let eloStr: String = {
            if elo.isNaN { return "--" }
            if elo == -.infinity { return "<floor" }
            if elo == .infinity { return ">ceil" }
            return String(format: "%.0f", elo)
        }()
        let stepStr = trainingStep.map(String.init) ?? "--"

        var perThemeParts: [String] = []
        for agg in aggregates.sorted(by: { $0.theme.rawValue < $1.theme.rawValue }) {
            perThemeParts.append(
                "\(agg.theme.rawValue)=\(agg.argmaxCorrect)/\(agg.total)"
            )
        }

        SessionLogger.shared.log(
            "[TACTICAL-LICHESS] tick"
            + " step=\(stepStr)"
            + " argmax=\(totalCorrect)/\(totalProbes)(\(pct(totalCorrect))%)"
            + " top5=\(top5Correct)/\(totalProbes)(\(pct(top5Correct))%)"
            + " avgProb=\(avgProbStr) avgRank=\(avgRankStr)"
            + " NLL=\(nllStr) pElo=\(eloStr)"
            + " model=\(modelLabel) "
            + perThemeParts.joined(separator: " ")
        )
    }
}
