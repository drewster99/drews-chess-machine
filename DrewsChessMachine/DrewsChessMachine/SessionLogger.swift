import Foundation

// MARK: - Session Logger

/// Thread-safe file logger that writes one line per event to a fresh
/// timestamped file each app launch.
///
/// Call `SessionLogger.shared.start()` once at app launch, then
/// `SessionLogger.shared.log(...)` from any thread or actor. Writes
/// are serialized via an internal lock and flushed to disk after
/// each line so a crash mid-session still leaves a usable log.
///
/// Log files land in the user's Library/Logs directory under a
/// `DrewsChessMachine` subfolder — in a sandboxed build that
/// resolves to
/// `~/Library/Containers/<bundle-id>/Data/Library/Logs/DrewsChessMachine/`.
/// Filenames follow the pattern `dcm_log_yyyymmdd-HHMMSS.txt` using
/// the session's launch time.
final class SessionLogger: @unchecked Sendable {
    static let shared = SessionLogger()

    private let lock = NSLock()
    private var fileHandle: FileHandle?
    private var fileURL: URL?
    private var didLogStartupFailure = false

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
        lock.lock()
        defer { lock.unlock() }

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

    /// Write a line to the session log. The timestamp and trailing
    /// newline are added automatically — callers pass the bare
    /// message (typically `"[TAG] details"`). Safe to call from any
    /// thread; no-op before `start()` or after a startup failure.
    func log(_ message: String) {
        let timestamp = Self.lineTimestampFormatter.string(from: Date())
        let line = "\(timestamp)  \(message)\n"
        let data = Data(line.utf8)

        lock.lock()
        defer { lock.unlock() }

        guard let fileHandle else { return }
        do {
            try fileHandle.write(contentsOf: data)
            try fileHandle.synchronize()
        } catch {
            // Swallow — a logger that can't write should never bring
            // down the app. Print once to stderr so there's at least
            // one breadcrumb if logging completely fails.
            if !didLogStartupFailure {
                didLogStartupFailure = true
                FileHandle.standardError.write(
                    Data("SessionLogger: write failed: \(error)\n".utf8)
                )
            }
        }
    }

    /// Path of the active log file, if any. Useful for surfacing the
    /// location to the user (e.g. via a "Reveal in Finder" menu item)
    /// or for debugging from LLDB.
    var activeLogPath: String? {
        lock.lock()
        defer { lock.unlock() }
        return fileURL?.path
    }
}
