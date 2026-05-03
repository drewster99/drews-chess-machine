import AppKit
import Foundation

/// Bridges AppKit lifecycle events into the early-stop flush path.
///
/// - `applicationDidFinishLaunching` opts the process out of macOS's
///   sudden-termination optimization (we are explicitly NOT a candidate;
///   training runs hold uncommitted in-memory state) and installs
///   SIGUSR1 / SIGHUP handlers via `EarlyStopCoordinator`.
/// - `applicationShouldTerminate` short-circuits AppKit's normal
///   teardown when a CLI training run is in flight, requesting a
///   flush+exit through the coordinator. The coordinator's handler
///   calls `Darwin._exit(0)` after writing `result.json`, so AppKit's
///   own teardown never runs in that case (which is what we want — we
///   already have a clean snapshot on disk).
/// - `applicationWillTerminate` is a belt-and-suspenders flush in
///   case `applicationShouldTerminate` was bypassed (e.g. SIGTERM).
final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        // Two layers of opt-out: the Info.plist key (set via build
        // settings) tells the OS at app-launch time that we don't
        // support sudden termination; the runtime call confirms it
        // for the current process. Reference-counted, but starting
        // from "off" via Info.plist means a single disable call
        // here is enough to stay off for the life of the process.
        ProcessInfo.processInfo.disableSuddenTermination()
        EarlyStopCoordinator.shared.installSignalHandlers()
        SessionLogger.shared.log("[APP] sudden termination disabled; SIGUSR1/SIGHUP handlers installed")
    }

    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        // If a CLI training run is in flight, route through the
        // early-stop path so result.json gets a clean flush before
        // exit. The coordinator's handler calls Darwin._exit(0) — we
        // never return to AppKit's normal terminate path. If no
        // CLI run is in flight, fall through to .terminateNow.
        if EarlyStopCoordinator.shared.hasActiveCLIFlush {
            EarlyStopCoordinator.shared.requestEarlyStop(reason: .appWillTerminate)
            // requestEarlyStop above invokes the handler which exits;
            // unreached. If for some reason the handler returned (it
            // shouldn't), .terminateNow lets AppKit finish teardown.
            return .terminateNow
        }
        return .terminateNow
    }

    func applicationWillTerminate(_ notification: Notification) {
        // Belt-and-suspenders. applicationShouldTerminate is the
        // primary path; this fires on any AppKit-driven teardown
        // that didn't go through that callback (rare but possible
        // for some logout flows).
        if EarlyStopCoordinator.shared.hasActiveCLIFlush {
            EarlyStopCoordinator.shared.requestEarlyStop(reason: .appWillTerminate)
        }
    }
}
