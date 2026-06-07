import Foundation

// MARK: - Session Logger

/// Thread-safe file logger that writes one line per event to a fresh
/// timestamped file each app launch.
///
/// Call `SessionLogger.shared.start()` once at app launch, then
/// `SessionLogger.shared.log(...)` from any thread or actor. Writes
/// are serialized via a private serial dispatch queue. Each write
/// hits the OS file handle immediately, but the explicit `fsync` is
/// coalesced — a write schedules an idle flush 0.5 s out, cancelling
/// any prior pending flush, so bursts (per-step STATS, BATCH-STATS)
/// collapse to one `synchronize()` at the tail of the burst. A
/// normal app exit funnels through `shutdown()` for a final
/// synchronous flush; only a hard kernel crash can lose the last
/// 0.5 s of log tail. `log(...)` dispatches asynchronously —
/// callers never block waiting for disk I/O, which keeps
/// Swift-concurrency tasks free to keep making progress.
///
/// Log files land in the user's Library/Logs directory under a
/// `DrewsChessMachine` subfolder — in a sandboxed build that
/// resolves to
/// `~/Library/Containers/<bundle-id>/Data/Library/Logs/DrewsChessMachine/`.
/// Filenames follow the pattern `dcm_log_yyyymmdd-HHMMSS.txt` using
/// the session's launch time.
final class SessionLogger: @unchecked Sendable {
    static let shared = SessionLogger()

    private let queue = DispatchQueue(label: "drewschess.sessionlogger.serial")
    private var fileHandle: FileHandle?
    private var fileURL: URL?
    private var didLogStartupFailure = false
    /// Once-only breadcrumb flag for the idle-flush `synchronize()`
    /// path. Same shape as `didLogStartupFailure`: queue-protected,
    /// flipped before the stderr write so a persistently-failing
    /// fsync can't recursively spam. Without this, a broken
    /// `synchronize()` (handle invalidated, device gone, disk full)
    /// silently disabled every coalesced flush and the user only
    /// noticed when the post-crash log tail came up short.
    private var didLogFsyncFailure = false

    /// Idle-flush coalescer. Each successful write cancels the previous
    /// pending flush and schedules a new one 0.5 s out. A burst of
    /// writes therefore produces at most one `synchronize()` per ~0.5 s
    /// of idle, instead of one per line. On hard kernel-level crash we
    /// lose at most the last 0.5 s of log tail; on a normal app exit
    /// `shutdown()` does a final synchronous flush. Mutated only inside
    /// the serial queue.
    private var pendingFlush: DispatchWorkItem?

    /// Local-time formatter for the filename stamp — the log file
    /// sits in the user's own Library/Logs folder, so local time is
    /// what they'll expect when eyeballing filenames.
    private static let filenameFormatter: DateFormatter = {
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.dateFormat = "yyyyMMdd-HHmmss"
        return f
    }()

    /// Per-line timestamp formatter: `HH:mm:ss.SSS` local time.
    /// Milliseconds included because human-scale events (button taps,
    /// arena starts) can easily fire inside the same second.
    private static let lineTimestampFormatter: DateFormatter = {
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.dateFormat = "HH:mm:ss.SSS"
        return f
    }()

    private init() {}

    /// Open the session log file. Safe to call exactly once at app
    /// launch; calling again is a no-op after the first success. If
    /// the file can't be opened (disk full, permissions issue, etc.)
    /// the error is printed to stderr and all subsequent `log` calls
    /// silently drop — the logger never crashes or escalates a log
    /// failure into an app-level error.
    func start() {
        queue.sync {
            if fileHandle != nil { return }

            do {
                let libraryURL = try FileManager.default.url(
                    for: .libraryDirectory,
                    in: .userDomainMask,
                    appropriateFor: nil,
                    create: true
                )
                let logsDir = libraryURL
                    .appendingPathComponent("Logs", isDirectory: true)
                    .appendingPathComponent("DrewsChessMachine", isDirectory: true)
                try FileManager.default.createDirectory(
                    at: logsDir,
                    withIntermediateDirectories: true
                )

                let stamp = Self.filenameFormatter.string(from: Date())
                let fileName = "dcm_log_\(stamp).txt"
                let url = logsDir.appendingPathComponent(fileName)

                FileManager.default.createFile(atPath: url.path, contents: nil)
                let handle = try FileHandle(forWritingTo: url)

                self.fileHandle = handle
                self.fileURL = url
            } catch {
                if !didLogStartupFailure {
                    didLogStartupFailure = true
                    FileHandle.standardError.write(
                        Data("SessionLogger: failed to open log file: \(error)\n".utf8)
                    )
                }
            }
        }
    }

    /// Write a line to the session log. The timestamp and trailing
    /// newline are added automatically — callers pass the bare
    /// message (typically `"[TAG] details"`). Safe to call from any
    /// thread; no-op before `start()` or after a startup failure.
    /// Dispatches asynchronously to the serial queue so callers never
    /// wait on disk I/O, which is the whole point of preferring a
    /// queue over an `NSLock` here.
    func log(_ message: String) {
        let timestamp = Self.lineTimestampFormatter.string(from: Date())
        let line = "\(timestamp)  \(message)\n"
        let data = Data(line.utf8)

        queue.async { [weak self] in
            guard let self else { return }
            guard let fileHandle = self.fileHandle else { return }
            do {
                try fileHandle.write(contentsOf: data)
            } catch {
                // Swallow — a logger that can't write should never bring
                // down the app. Print once to stderr so there's at least
                // one breadcrumb if logging completely fails.
                if !self.didLogStartupFailure {
                    self.didLogStartupFailure = true
                    FileHandle.standardError.write(
                        Data("SessionLogger: write failed: \(error)\n".utf8)
                    )
                }
            }
        }
    }

    /// Emit the authoritative `[ARCH]` line describing a network architecture
    /// that has just become live — built, loaded from a model file, resumed
    /// from a session, or the trainer forked at training start. This is the
    /// single source of truth for "what architecture is actually running."
    ///
    /// The `[APP] launched` banner can only report `NetworkArchitecture.current`
    /// (the compile-time default preset), because at launch no model exists
    /// yet. The architecture is runtime-configurable, so the *live* arch
    /// routinely differs from that default — every transition that makes one
    /// live logs one of these so a run's true encoding (e.g. `full10ply200(200)`
    /// vs the default `basic30(30)`), topology, and parameter count are never
    /// in doubt. `event` names the transition and any model id / file.
    func logArchitecture(event: String, arch: NetworkArchitecture) {
        log("[ARCH] \(event) | \(arch.architectureSummary)")
    }

    /// Path of the active log file, if any. Useful for surfacing the
    /// location to the user (e.g. via a "Reveal in Finder" menu item)
    /// or for debugging from LLDB.
    var activeLogPath: String? {
        queue.sync { fileURL?.path }
    }

    /// Synchronously flush and close the log file. Called from
    /// AppDelegate.applicationWillTerminate so a normal app exit
    /// preserves the full tail that the idle-flush coalescer might
    /// otherwise drop. `queue.sync` FIFOs behind any in-flight write,
    /// so this never deadlocks. Not invoked on `_exit(2)` paths
    /// (early-stop coordinator); tail loss on those paths is
    /// consistent with the existing best-effort log posture.
    func shutdown() {
        queue.sync {
            pendingFlush?.cancel()
            pendingFlush = nil
            try? fileHandle?.synchronize()
            try? fileHandle?.close()
            fileHandle = nil
        }
    }
}
