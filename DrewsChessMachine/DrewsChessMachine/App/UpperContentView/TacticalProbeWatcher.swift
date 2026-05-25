import Foundation

/// Driver that runs the full 7-probe battery every `intervalSec`
/// while the Tactical Probe Monitor window is open. Each tick reads
/// the live champion `ChessMPSNetwork` off `SessionController` (via a
/// weak ref so the window doesn't keep the session alive after the
/// app's main lifecycle ends) and appends one snapshot to the shared
/// `TacticalProbeHistory`. Gracefully no-ops when the network is nil.
///
/// @MainActor isolated because both the network handle and the
/// history store live on the main actor. The forward passes
/// themselves are async and bounce off the network's `executionQueue`,
/// so the main actor is only briefly held to dispatch and to write
/// the result back into the history. Probe cost is ~3ms × 7 ≈ 21ms
/// per tick (a small fraction of the 15s cadence).
///
/// Two-phase lifecycle: `start()` kicks the loop; `stop()` cancels it.
/// The window controller calls `stop()` from `windowWillClose(_:)`
/// so the watcher doesn't survive the window. Re-entrant — calling
/// `start()` while already running is a no-op.
@MainActor
final class TacticalProbeWatcher {
    private weak var sessionController: SessionController?
    private let history: TacticalProbeHistory
    private let intervalSec: TimeInterval
    private var task: Task<Void, Never>?

    init(
        sessionController: SessionController,
        history: TacticalProbeHistory,
        intervalSec: TimeInterval = 15.0
    ) {
        self.sessionController = sessionController
        self.history = history
        self.intervalSec = intervalSec
    }

    /// Begin the periodic loop. Runs an immediate first tick (so the
    /// window populates within a few hundred ms of opening) and then
    /// sleeps `intervalSec` between subsequent ticks. Re-entrant: a
    /// second `start()` while running returns without doing anything.
    func start() {
        guard task == nil else { return }
        task = Task { [weak self, intervalSec] in
            // Fire immediately on open.
            await self?.tickOnce()
            while !Task.isCancelled {
                let nanos = UInt64(intervalSec * 1_000_000_000)
                do {
                    try await Task.sleep(nanoseconds: nanos)
                } catch {
                    // sleep throws CancellationError on stop; bail.
                    return
                }
                if Task.isCancelled { return }
                await self?.tickOnce()
            }
        }
    }

    /// Cancel the loop. Outstanding probe forward passes that have
    /// already launched run to completion (they're cheap), but the
    /// loop won't fire another tick. Safe to call multiple times.
    func stop() {
        task?.cancel()
        task = nil
    }

    /// One tick: read the live champion, run all probes, append to
    /// history. No-ops cleanly when the network is gone (e.g. the
    /// session has been torn down but the window is still open). Logs
    /// nothing here — the monitor window is the visible record; the
    /// per-error log lines come from `runTacticalProbe` itself on
    /// forward-pass failures.
    private func tickOnce() async {
        guard let session = sessionController, let net = session.network else { return }
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
