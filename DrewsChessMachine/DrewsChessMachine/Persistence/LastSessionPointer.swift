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

    /// The pointer stored in `defaults`: nil only when nothing is stored
    /// under the key. A stored value that is not pointer data, or that does
    /// not decode, throws — so a caller that must not mistake "unreadable"
    /// for "none" (the automatic-save sweep, which protects the pointer's
    /// target) can tell them apart.
    static func stored(in defaults: UserDefaults) throws -> LastSessionPointer? {
        guard let object = defaults.object(forKey: userDefaultsKey) else {
            return nil
        }
        guard let data = object as? Data else {
            throw LastSessionPointerError.notData(typeDescription: String(describing: type(of: object)))
        }
        do {
            return try JSONDecoder().decode(LastSessionPointer.self, from: data)
        } catch {
            throw LastSessionPointerError.undecodable(detail: error.localizedDescription)
        }
    }

    /// Read the pointer currently stored in the given defaults,
    /// or `nil` if none has been set (first launch or the user
    /// never saved) or the stored value cannot be read.
    static func read(from defaults: UserDefaults = .standard) -> LastSessionPointer? {
        do {
            return try stored(in: defaults)
        } catch {
            // Unreadable pointer — logged, and nil so the launch flow
            // falls back to the no-saved-session state. Deliberately do
            // not rewrite / clear the key: a future build with a
            // different schema might still be able to read it.
            SessionLogger.shared.log(
                "[CHECKPOINT] LastSessionPointer unreadable: \(error.localizedDescription)"
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

/// Why a stored `LastSessionPointer` could not be read.
enum LastSessionPointerError: LocalizedError, Equatable {
    /// Something other than encoded pointer data is stored under the key.
    case notData(typeDescription: String)
    /// The stored data does not decode as a pointer.
    case undecodable(detail: String)

    var errorDescription: String? {
        switch self {
        case .notData(let typeDescription):
            return "the stored resume pointer is a \(typeDescription), not encoded pointer data"
        case .undecodable(let detail):
            return "the stored resume pointer does not decode: \(detail)"
        }
    }
}
