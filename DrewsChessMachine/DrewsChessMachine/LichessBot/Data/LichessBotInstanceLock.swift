import Darwin
import Foundation

enum LichessBotInstanceLockError: LocalizedError, Equatable {
    /// Another process holds the lock. `holder` is what that process wrote
    /// about itself, for the message only.
    case heldByAnotherProcess(holder: String)
    case systemError(operation: String, code: Int32)

    var errorDescription: String? {
        switch self {
        case .heldByAnotherProcess(let holder):
            return "The bot is already online in another DrewsChessMachine process (\(holder))"
        case .systemError(let operation, let code):
            return "Bot instance lock: \(operation) failed: \(String(cString: strerror(code)))"
        }
    }
}

/// Who holds the lock, recorded in the lock file for error messages. The
/// file's contents are never the lock itself.
struct LichessBotInstanceLockHolder: Sendable, Codable, Equatable {
    let pid: Int32
    let launchedAt: Date
    let build: Int
    let hostName: String

    static var current: LichessBotInstanceLockHolder {
        LichessBotInstanceLockHolder(
            pid: getpid(),
            launchedAt: Date(),
            build: BuildInfo.buildNumber,
            hostName: ProcessInfo.processInfo.hostName
        )
    }

    var summary: String {
        "pid \(pid), build \(build), on \(hostName), since \(launchedAt.formatted(date: .abbreviated, time: .standard))"
    }
}

/// An exclusive `flock` on `LichessBot/bot.lock`, held for as long as the
/// bot is online (plan §6.1 A).
///
/// The kernel drops a `flock` when its process exits or crashes, so there
/// is no stale-lock problem. `flock` locks belong to an open file
/// description, so a second acquisition fails even within one process.
/// The lock also keeps two processes from writing the same journals.
final class LichessBotInstanceLock: @unchecked Sendable {
    // `descriptor` is written only in `init` and `release`; `release` runs
    // once, guarded by `released`.
    private var descriptor: Int32
    private let released = SyncBox(false)
    let url: URL

    private init(descriptor: Int32, url: URL) {
        self.descriptor = descriptor
        self.url = url
    }

    /// Take the lock, or throw `heldByAnotherProcess` naming the holder.
    static func acquire(at url: URL, holder: LichessBotInstanceLockHolder) throws -> LichessBotInstanceLock {
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        let fd = open(url.path, O_RDWR | O_CREAT | O_CLOEXEC, 0o644)
        guard fd >= 0 else {
            throw LichessBotInstanceLockError.systemError(operation: "open", code: errno)
        }
        guard flock(fd, LOCK_EX | LOCK_NB) == 0 else {
            let code = errno
            let description = readHolder(fd)
            close(fd)
            if code == EWOULDBLOCK {
                throw LichessBotInstanceLockError.heldByAnotherProcess(holder: description)
            }
            throw LichessBotInstanceLockError.systemError(operation: "flock", code: code)
        }
        let lock = LichessBotInstanceLock(descriptor: fd, url: url)
        do {
            try lock.writeHolder(holder)
        } catch {
            lock.release()
            throw error
        }
        return lock
    }

    func release() {
        let first = released.mutate { released -> Bool in
            defer { released = true }
            return !released
        }
        guard first else { return }
        // Closing the descriptor drops the lock even if the explicit
        // unlock failed; either failure is logged, not thrown, because the
        // caller is going offline regardless.
        if flock(descriptor, LOCK_UN) != 0 {
            SessionLogger.shared.log("[ALARM] LICHESS-BOT instance lock: unlock failed: \(String(cString: strerror(errno)))")
        }
        if close(descriptor) != 0 {
            SessionLogger.shared.log("[ALARM] LICHESS-BOT instance lock: close failed: \(String(cString: strerror(errno)))")
        }
        descriptor = -1
    }

    deinit {
        release()
    }

    private func writeHolder(_ holder: LichessBotInstanceLockHolder) throws {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        let data = try encoder.encode(holder)
        guard ftruncate(descriptor, 0) == 0 else {
            throw LichessBotInstanceLockError.systemError(operation: "ftruncate", code: errno)
        }
        let written = data.withUnsafeBytes { buffer in
            pwrite(descriptor, buffer.baseAddress, buffer.count, 0)
        }
        guard written == data.count else {
            throw LichessBotInstanceLockError.systemError(operation: "pwrite", code: errno)
        }
    }

    /// The holder's own description, or a note saying it couldn't be read.
    private static func readHolder(_ fd: Int32) -> String {
        var buffer = [UInt8](repeating: 0, count: 4096)
        let count = buffer.withUnsafeMutableBytes { pread(fd, $0.baseAddress, $0.count, 0) }
        guard count > 0 else {
            return "holder unknown: the lock file is empty"
        }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        do {
            return try decoder.decode(LichessBotInstanceLockHolder.self, from: Data(buffer[0..<count])).summary
        } catch {
            return "holder unknown: \(error.localizedDescription)"
        }
    }
}
