import Foundation

/// Driver that runs the full 7-probe battery every
/// `triggerEverySteps` trainer SGD steps (default 100) for the life
/// of the session. Each tick reads the live champion (or trainer
/// snapshot, depending on `session.probeNetworkTarget`) and appends
/// one snapshot to the shared `TacticalProbeHistory`. Gracefully
/// no-ops when the network is nil.
///
/// Cadence rationale: time-based ticks (the prior 10-minute default)
/// decoupled probe density from training progress — fast warm-up
/// steps blew past several ticks of "the network just changed",
/// while a paused trainer kept emitting stale-data ticks. Tying
/// cadence to trainer SGD steps means each chart point represents
/// a fixed amount of gradient applied, which is what we actually
/// want to compare across runs.
///
/// At ~2.5 trainer steps/s and the 100-step default, this fires
/// roughly every 40 seconds during sustained training — faster than
/// the Lichess watcher (200 steps, ~80 s) so the smaller probe set
/// gets denser sampling for the spark trends. Probe cost is a small
/// fraction of one tick's interval at this cadence.
///
/// @MainActor isolated because both the network handle and the
/// history store live on the main actor. The forward passes
/// themselves are async and bounce off the network's `executionQueue`,
/// so the main actor is only briefly held to dispatch and to write
/// the result back into the history.
///
/// Two-phase lifecycle: `start()` kicks the loop; `stop()` cancels it.
/// The window controller calls `stop()` from `windowWillClose(_:)`
/// so the watcher doesn't survive the window. Re-entrant — calling
/// `start()` while already running is a no-op.
@MainActor
final class TacticalProbeWatcher {
    private weak var sessionController: SessionController?
    private let history: TacticalProbeHistory
    /// Fire one tick every time the trainer's `completedTrainSteps`
    /// has advanced by this many steps since the last tick. See
    /// `LichessProbeWatcher` for the cadence rationale; this watcher
    /// uses a lower step count because the 7-probe battery is far
    /// cheaper and benefits more from dense sampling.
    private let triggerEverySteps: Int
    /// Polling cadence for the step-watch loop. 2.0s is faster than
    /// the expected tick interval (~40s at 100 steps and ~2.5 steps/s)
    /// so a crossing is detected within a few seconds. Mirrors the
    /// Lichess watcher's choice so both monitors share latency
    /// characteristics.
    private let pollIntervalSec: TimeInterval = 2.0
    private var task: Task<Void, Never>?

    init(
        sessionController: SessionController,
        history: TacticalProbeHistory,
        triggerEverySteps: Int = 100
    ) {
        self.sessionController = sessionController
        self.history = history
        self.triggerEverySteps = max(1, triggerEverySteps)
    }

    /// Begin the step-watching loop. Three-phase startup:
    ///  1. Poll every 5 seconds until the network needed by the
    ///     current `probeNetworkTarget` is ready (training has
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
            // Phase 1+2: wait for training, then warm up.
            await self?.waitForFirstTickConditions()
            if Task.isCancelled { return }

            await self?.tickOnce()
            var lastTriggeredStep = await self?.currentTrainerStep() ?? 0

            while !Task.isCancelled {
                do {
                    let nanos = UInt64(
                        (self?.pollIntervalSec ?? 2.0) * 1_000_000_000
                    )
                    try await Task.sleep(nanoseconds: nanos)
                } catch {
                    // sleep throws CancellationError on stop; bail.
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
                    // Snap to the most recent multiple-of-N boundary at
                    // or below `step` so a slow tick (probe forward
                    // passes block the loop for a moment) doesn't
                    // accumulate "missed" boundaries and double-fire.
                    lastTriggeredStep = step - (step % self.triggerEverySteps)
                }
            }
        }
    }

    /// Convenience accessor used by the step-watch loop. Returns nil
    /// when no trainer is yet attached to the session.
    @MainActor
    private func currentTrainerStep() -> Int? {
        sessionController?.trainer?.completedTrainSteps
    }

    /// Poll-until-ready + 60s warm-up before the first tick.
    /// "Ready" means the network the current `probeNetworkTarget`
    /// would consume is built — `network` (the champion) for
    /// `.champion`, `tacticalProbeInferenceNetwork` plus a non-nil
    /// trainer for `.candidate`. Once ready, sleeps 60s and returns.
    /// One-time per watcher: if training is stopped after the first
    /// tick, the periodic loop continues without re-running this gate.
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

    /// Whether `tickOnce` would currently be able to run a probe
    /// against a valid network. Mirrors the guards inside `tickOnce`.
    @MainActor
    private func canTickNow() -> Bool {
        guard let session = sessionController else { return false }
        switch session.probeNetworkTarget {
        case .champion:
            return session.network != nil
        case .candidate:
            return session.trainer != nil
                && session.tacticalProbeInferenceNetwork != nil
        }
    }

    /// Cancel the loop. Outstanding probe forward passes that have
    /// already launched run to completion (they're cheap), but the
    /// loop won't fire another tick. Safe to call multiple times.
    func stop() {
        task?.cancel()
        task = nil
    }

    /// Fire a single probe cycle on demand, independent of the
    /// periodic loop. Used by the "Probe now" button in the monitor
    /// window header so the user can refresh the displayed stats
    /// without waiting up to `intervalSec` for the next scheduled
    /// tick. The on-demand cycle runs concurrently with any
    /// in-flight scheduled tick — they share the network's
    /// `executionQueue` so they serialize naturally and there's
    /// no risk of duplicate batches landing on the same forward
    /// pass; the displayed sample count just advances by one extra
    /// entry for each manual fire.
    func triggerOnce() {
        Task { [weak self] in
            await self?.tickOnce()
        }
    }

    /// One tick: pick the network to probe based on
    /// `session.probeNetworkTarget` (default `.candidate`, controlled
    /// by the existing main-UI Picker), run all probes, append to
    /// history. No-ops cleanly when the network needed isn't loaded.
    ///
    /// On `.candidate`: snapshots the trainer's CURRENT weights into
    /// the dedicated `tacticalProbeInferenceNetwork` (an inference-mode
    /// network — necessary because the trainer's own network uses
    /// training-mode BN with fresh batch stats, which would give
    /// nonsense outputs on a single-position forward pass). A
    /// separate network — not the candidate-test probe's
    /// `probeInferenceNetwork` — so the two consumers can't overwrite
    /// each other's weight snapshot mid-cycle. Each tick gets a fresh
    /// snapshot, so the monitor tracks the live trainer's evolving
    /// policy between promotions — that is what makes the periodic
    /// cadence meaningful (otherwise the probe shows the same frozen
    /// champion until a promotion fires).
    ///
    /// On `.champion`: reads `session.network` directly, same as the
    /// pre-change behavior. Useful for "what is the deployed network
    /// actually doing" between promotions; probe values are stable
    /// (identical every tick) until a promotion overwrites
    /// `session.network`.
    ///
    /// Logs `[TACTICAL]` on the trainer-snapshot failure path only —
    /// `TacticalProbeRunner.run` swallows per-probe forward-pass
    /// failures into the result's `.error` verdict so per-probe
    /// failures are visible in the monitor window directly.
    private func tickOnce() async {
        guard let session = sessionController else { return }
        let target = session.probeNetworkTarget

        let net: ChessMPSNetwork
        switch target {
        case .champion:
            guard let championNet = session.network else { return }
            net = championNet
        case .candidate:
            // Use the dedicated `tacticalProbeInferenceNetwork`, not
            // the shared `probeInferenceNetwork`. Snapshotting trainer
            // weights into the shared net while
            // `fireCandidateProbeIfNeeded` is doing the same can leave
            // the wrong weights in place mid-cycle. Owning a separate
            // network here costs one extra `ChessMPSNetwork` instance
            // but eliminates the race for good.
            guard let trainer = session.trainer,
                  let probeNet = session.tacticalProbeInferenceNetwork else { return }
            do {
                let weights = try await trainer.network.exportWeights()
                try await probeNet.loadWeights(weights)
            } catch {
                SessionLogger.shared.log(
                    "[TACTICAL] monitor trainer-snapshot failed: \(error.localizedDescription)"
                )
                return
            }
            // Inherit the trainer's ID so any future per-tick record
            // points back to a specific weight snapshot.
            probeNet.identifier = trainer.identifier
            net = probeNet
        }

        let probes = TacticalProbeData.standardSet
        var results: [ProbeResult] = []
        results.reserveCapacity(probes.count)
        for probe in probes {
            let r = await TacticalProbeRunner.run(probe, against: net)
            // Network may have been replaced (promotion) while a probe
            // was in flight. The result still describes a real forward
            // pass against the network that existed at submit time —
            // record it and continue.
            results.append(r)
        }
        history.record(results)
    }
}
