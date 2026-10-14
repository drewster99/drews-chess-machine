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
/// and writes them to disk only when the run ends (`AutoTrainTermination`,
/// shared by the time-limit, step-limit and collapse paths).
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
    static let shared = EarlyStopCoordinator(terminateApp: deferredToTheRunLoop { NSApp.terminate(nil) })

    /// `action` run later from the main run loop instead of now. SIGTERM's
    /// handler runs inside a main-actor task; `NSApp.terminate` called there
    /// spins AppKit's quit loop inside that task, so the main actor stays
    /// occupied and the bot's reply to `.terminateLater` (another main-actor
    /// task) can never run: the process hangs half-quit with Quit disabled
    /// (seen 2026-10-13). Run from a run-loop block, the quit loop holds no
    /// task, and the reply runs.
    static func deferredToTheRunLoop(_ action: @escaping @MainActor () -> Void) -> @MainActor () -> Void {
        {
            RunLoop.main.perform {
                MainActor.assumeIsolated { action() }
            }
        }
    }

    /// Set by `SessionController` when a CLI training run starts. Captures
    /// the run's `AutoTrainTermination` and start time as a closure that,
    /// when invoked, performs the same flush-and-exit the time-limit path
    /// performs — with a different `terminationReason`.
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
    private var sigtermSource: DispatchSourceSignal?
    private var stopRequested: Bool = false
    /// AppKit was told `.terminateLater` and is waiting for the reply
    /// (`quitDeferred`, whatever started the quit): `applicationShouldTerminate`
    /// then doesn't start a second quit. Cleared only by `quitWasCancelled`.
    private var quitInProgress = false
    /// A SIGTERM has asked AppKit to quit (the request runs later, from the
    /// run loop): a repeat only logs. Cleared with `quitInProgress` when the
    /// quit is cancelled. Kept apart from `quitInProgress` because AppKit's
    /// `applicationShouldTerminate` for this very request must still reach
    /// the bot.
    private var terminationRequested = false
    /// What SIGTERM does: ask AppKit to quit, exactly as ⌘Q does.
    private let terminateApp: @MainActor () -> Void

    /// The process's coordinator; `terminateApp` is AppKit's quit.
    init(terminateApp: @escaping @MainActor () -> Void) {
        self.terminateApp = terminateApp
    }

    /// Wires libc + Dispatch so SIGUSR1, SIGHUP, SIGUSR2 and SIGTERM route
    /// into the coordinator on the main queue. Call once from
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

        // SIGTERM — "quit nicely from outside" (owner request 2026-10-10):
        // the same path as ⌘Q, so a `kill <pid>` gets whatever ⌘Q does in
        // this process. A CLI training run flushes and exits; with Lichess
        // games in play the bot stops taking games, withdraws its pending
        // challenges, finishes every running game however long it takes,
        // then quits. A signal names one process, unlike an Apple Event
        // `quit` addressed to the app (the bot and a training window can
        // be two instances of it). `kill -9` still ends the process at once.
        // CLI corpus-replay and train-vs-UCI runs exit before AppKit
        // launches, so this never reaches them; they keep the default.
        signal(SIGTERM, SIG_IGN)
        let term = DispatchSource.makeSignalSource(signal: SIGTERM, queue: .main)
        term.setEventHandler {
            Task.detached { @MainActor in
                EarlyStopCoordinator.shared.handleTerminationSignal()
            }
        }
        term.activate()
        sigtermSource = term
    }

    /// The app answered a quit with `.terminateLater` (running Lichess
    /// games finish first): until it is cancelled, a SIGTERM only logs, so
    /// it never starts a second quit over the pending one.
    func quitDeferred() {
        quitInProgress = true
    }

    /// Whether a deferred quit is pending, so `applicationShouldTerminate`
    /// doesn't start a second one.
    var isQuitInProgress: Bool { quitInProgress }

    /// A quit was cancelled (the operator cancelled "Finishing games"):
    /// the next SIGTERM or ⌘Q asks to quit again.
    func quitWasCancelled() {
        guard quitInProgress || terminationRequested else { return }
        quitInProgress = false
        terminationRequested = false
        SessionLogger.shared.log("[APP] quit cancelled; a new SIGTERM or ⌘Q will quit again")
    }

    /// SIGTERM's handler. During a GUI `--train` run it stops the run
    /// through the early-stop flush, labelled `SIGTERM-received` (autotrain's
    /// watchdog sends it); otherwise the first asks AppKit to quit, and a
    /// repeat while a quit is pending only says so.
    func handleTerminationSignal() {
        if quitInProgress || terminationRequested {
            SessionLogger.shared.log("[APP] SIGTERM again: already quitting (running Lichess games finish first; kill -9 ends the process now)")
            return
        }
        if hasActiveCLIFlush {
            SessionLogger.shared.log("[APP] SIGTERM: stopping the --train run (result written as SIGTERM-received)")
            requestEarlyStop(reason: .sigtermReceived)
            return
        }
        // Only `terminationRequested`, never `quitInProgress`: AppKit's
        // `applicationShouldTerminate` for this request must reach the bot
        // (a pending quit there returns `.terminateLater` without asking it,
        // and nothing would ever answer); the delegate marks the quit pending
        // when the bot defers it.
        SessionLogger.shared.log("[APP] SIGTERM: quitting as ⌘Q does")
        terminationRequested = true
        terminateApp()
    }

    /// Idempotent early-stop entry point. Multiple signals + AppDelegate
    /// paths can converge here without double-flushing. After the first
    /// call, subsequent calls are no-ops.
    func requestEarlyStop(reason: CliTrainingRecorder.TerminationReason) {
        guard !stopRequested else { return }
        stopRequested = true

        if let handler = earlyStopHandler {
            // Hand off to the session's flush closure. It writes the results
            // and ends the process through `AutoTrainTermination`, or — when
            // another termination path already holds the run's claim and is
            // doing exactly that — returns, and that path exits.
            handler(reason)
            return
        }

        // No CLI run in flight. For signal-driven paths the user clearly
        // wants the process down; exit. For AppKit-driven paths
        // (`appWillTerminate`) AppKit's normal teardown handles exit;
        // we just no-op so the regular `.terminateNow` reply runs.
        switch reason {
        case .sigusr1Requested, .sighupReceived, .sigtermReceived:
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
