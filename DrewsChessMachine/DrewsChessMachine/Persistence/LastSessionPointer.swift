import Foundation

/// Persisted "most recently saved session" pointer used to drive
/// the app-launch auto-resume prompt. Stored in `UserDefaults`
/// under a single key; updated on every successful session save
/// regardless of trigger (manual, post-promotion, periodic) so the
/// pointer always names the freshest on-disk session.
///
/// The directory URL is stored as a plain file-system path string
/// rather than a security-scoped bookmark: the target always lives
/// under the app's own `Application Support` folder, which is
/// readable without a bookmark even if the app is later sandboxed,
/// so the extra machinery would just add failure modes without
/// buying anything.
struct LastSessionPointer: Codable, Equatable, Sendable {

    /// UserDefaults key the pointer is persisted under. Singular —
    /// only one pointer is tracked at a time (the latest save
    /// wins).
    static let userDefaultsKey = "DrewsChessMachine.LastSessionPointer.v1"

    /// Stable session identifier (matches the sessionID inside the
    /// session's own `session.json`). Written into the resume
    /// prompt so the user sees which session they are about to
    /// continue.
    let sessionID: String

    /// Path to the `.dcmsession` directory on disk. Stored as a
    /// plain filesystem path (not a bookmark) — see the type
    /// doc-comment for the rationale.
    let directoryPath: String

    /// Unix timestamp of when the save completed. Used in the
    /// resume prompt's human-readable "saved N minutes ago" label
    /// and for staleness diagnostics in the session log.
    let savedAtUnix: Int64

    /// Which save path wrote this pointer. One of `"manual"`,
    /// `"periodic"`, `"promote"`, `"post-promotion"`. Purely
    /// informational — the resume flow treats all four the same way.
    let trigger: String

    /// Reconstruct the directory URL from the stored path.
    var directoryURL: URL {
        URL(fileURLWithPath: directoryPath, isDirectory: true)
    }

    /// `true` if the directory named by this pointer still exists
    /// on disk. A pointer that names a missing directory is stale
    /// (the user deleted the session manually) and should surface
    /// as "no session to resume".
    var directoryExists: Bool {
        var isDir: ObjCBool = false
        let exists = FileManager.default.fileExists(atPath: directoryPath, isDirectory: &isDir)
        return exists && isDir.boolValue
    }

    // MARK: - Persistence

    /// Read the pointer currently stored in the given defaults,
    /// or `nil` if none has been set (first launch or the user
    /// never saved).
    static func read(from defaults: UserDefaults = .standard) -> LastSessionPointer? {
        guard let data = defaults.data(forKey: userDefaultsKey) else {
            return nil
        }
        do {
            return try JSONDecoder().decode(LastSessionPointer.self, from: data)
        } catch {
            // Corrupt pointer — route through SessionLogger (the
            // singleton is queue-protected and the app starts the
            // logger before any LastSessionPointer call site runs;
            // the old `print` line went to stdout, which the app's
            // log file never captures). Return nil so the caller
            // falls back to the no-saved-session launch state.
            // Deliberately do not rewrite / clear the key: a future
            // build with a different schema might still be able to
            // read it.
            SessionLogger.shared.log(
                "[CHECKPOINT] LastSessionPointer decode failed: \(error.localizedDescription)"
            )
            return nil
        }
    }

    /// Encode and store `self` in the given defaults. Any encode
    /// failure is logged and the stored value is left unchanged —
    /// a failure to update the pointer must not break the save
    /// path that called us, and the next save will retry.
    func write(to defaults: UserDefaults = .standard) {
        do {
            let data = try JSONEncoder().encode(self)
            defaults.set(data, forKey: Self.userDefaultsKey)
        } catch {
            SessionLogger.shared.log(
                "[CHECKPOINT] LastSessionPointer encode failed: \(error.localizedDescription)"
            )
        }
    }

    /// Remove any stored pointer. Intended for the "user manually
    /// deleted the target" cleanup path and for tests.
    static func clear(in defaults: UserDefaults = .standard) {
        defaults.removeObject(forKey: userDefaultsKey)
    }
}
