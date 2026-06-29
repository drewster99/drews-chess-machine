import AppKit
import Darwin
import Foundation

/// Singleton that funnels mid-run early-stop signals (SIGUSR1, SIGHUP)
/// and AppKit termination hooks (`applicationShouldTerminate`,
/// `applicationWillTerminate`) into a single `result.json` flush path
/// before process exit.
///
/// Why this exists: a CLI training run owns a `CliTrainingRecorder`
/// that buffers stats / arena results / candidate probes in memory
/// and writes them to disk only at the timer-expired branch (see
/// `ContentView.swift` near the `Darwin._exit(0)` for `.timerExpired`).
/// Without this coordinator, autotrain's mid-run hard-reject path
/// (`kill <pid>`) destroys all that buffered telemetry, leaving only
/// a stub commentary for analysis. With it, autotrain can `kill -USR1
/// <pid>` and get a fully-populated `result.json` with
/// `termination_reason: "SIGUSR1-user-requested"` — same data shape
/// as a timer-expired run, just over a truncated window.
///
/// All paths are idempotent: the first signal that arrives wins, and
/// the coordinator guarantees exactly one flush+exit. This matters
/// because SIGHUP and `applicationWillTerminate` can both fire on
/// macOS logout.
@MainActor
final class EarlyStopCoordinator {
    static let shared = EarlyStopCoordinator()

    /// Set by `ContentView` when a CLI training run starts. Captures the
    /// recorder, output URL, and run start time as a closure that, when
    /// invoked, performs the same flush sequence the timer-expired
    /// branch performs — with a different `terminationReason`.
    ///
    /// Cleared back to `nil` when the run completes through any path
    /// (timer expiry, legal-mass collapse, user stop). Nil also means
    /// "no CLI run in flight" — the coordinator's signal handlers
    /// then just exit cleanly without trying to flush.
    var earlyStopHandler: ((CliTrainingRecorder.TerminationReason) -> Void)?

    /// Set by `SessionController` while a training session is active.
    /// SIGUSR2 invokes it to write a full `.dcmsession` checkpoint and —
    /// on success — shut the process down cleanly, so the session can be
    /// resumed on next launch. Unlike `earlyStopHandler` (CLI-only), this
    /// is registered for GUI sessions too. Nil ⇒ no active session to save.
    var saveSessionHandler: (() -> Void)?

    private var sigusr1Source: DispatchSourceSignal?
    private var sighupSource: DispatchSourceSignal?
    private var sigusr2Source: DispatchSourceSignal?
    private var stopRequested: Bool = false

    private init() {}

    /// Wires libc + Dispatch so SIGUSR1 and SIGHUP route into the
    /// coordinator on the main queue. Call once from
    /// `applicationDidFinishLaunching`. Calling more than once is a
    /// programmer error (each `DispatchSource.makeSignalSource` consumes
    /// the signal exclusively).
    func installSignalHandlers() {
        // Tell libc to ignore the signals so DispatchSource has
        // exclusive ownership. Without SIG_IGN the default disposition
        // (terminate the process for SIGUSR1; same for SIGHUP) races
        // against the dispatch handler and the process can exit before
        // the handler runs.
        signal(SIGUSR1, SIG_IGN)
        signal(SIGHUP, SIG_IGN)

        let usr1 = DispatchSource.makeSignalSource(signal: SIGUSR1, queue: .main)
        usr1.setEventHandler {
            // DispatchSource handlers on `.main` ARE on the main run
            // loop, but Swift concurrency doesn't know that — wrap in
            // a Task @MainActor to satisfy isolation.
            Task.detached { @MainActor in
                EarlyStopCoordinator.shared.requestEarlyStop(reason: .sigusr1Requested)
            }
        }
        usr1.activate()
        sigusr1Source = usr1

        let hup = DispatchSource.makeSignalSource(signal: SIGHUP, queue: .main)
        hup.setEventHandler {
            Task.detached { @MainActor in
                EarlyStopCoordinator.shared.requestEarlyStop(reason: .sighupReceived)
            }
        }
        hup.activate()
        sighupSource = hup

        // SIGUSR2 — "checkpoint now and (on success) shut down": write a full
        // .dcmsession so the next launch auto-resumes, then exit. Distinct from
        // SIGUSR1/SIGHUP (which only flush CLI metrics and die without saving).
        signal(SIGUSR2, SIG_IGN)
        let usr2 = DispatchSource.makeSignalSource(signal: SIGUSR2, queue: .main)
        usr2.setEventHandler {
            Task.detached { @MainActor in
                EarlyStopCoordinator.shared.requestSessionSave()
            }
        }
        usr2.activate()
        sigusr2Source = usr2
    }

    /// Idempotent early-stop entry point. Multiple signals + AppDelegate
    /// paths can converge here without double-flushing. After the first
    /// call, subsequent calls are no-ops.
    func requestEarlyStop(reason: CliTrainingRecorder.TerminationReason) {
        guard !stopRequested else { return }
        stopRequested = true

        if let handler = earlyStopHandler {
            // Hand off to ContentView's flush closure. The closure is
            // expected to call `Darwin._exit(0)` after writing — control
            // does not return.
            handler(reason)
            return
        }

        // No CLI run in flight. For signal-driven paths the user clearly
        // wants the process down; exit. For AppKit-driven paths
        // (`appWillTerminate`) AppKit's normal teardown handles exit;
        // we just no-op so the regular `.terminateNow` reply runs.
        switch reason {
        case .sigusr1Requested, .sighupReceived:
            Darwin._exit(0)
        default:
            return
        }
    }

    /// Whether a CLI training run is currently flushable. Used by
    /// `AppDelegate.applicationShouldTerminate` to decide between
    /// `.terminateLater` (let the flush finish) and `.terminateNow`.
    var hasActiveCLIFlush: Bool {
        earlyStopHandler != nil
    }

    /// SIGUSR2 entry point — request a "save session, then (on success) shut
    /// down." Re-entrancy is handled by the handler's own in-flight guard, so
    /// repeated signals during a save are no-ops and a signal after a *failed*
    /// save can retry. With no active session (`saveSessionHandler` nil) there's
    /// nothing to checkpoint, so it logs and stays alive rather than killing the
    /// process on a stray signal.
    func requestSessionSave() {
        guard let saveSessionHandler else {
            SessionLogger.shared.log("[SIGUSR2] no active session to save; staying running")
            return
        }
        saveSessionHandler()
    }
}
