import Foundation

// MARK: - Model ID

/// Stable per-model identifier in the format `yyyymmdd-N-XXXX` for a
/// lineage root, with optional trainer-generation suffix
/// `-<generation>` (for example `20260418-1-AbCd-2`).
///
/// The three components are:
/// - **`yyyymmdd`** — the UTC date the ID was minted on.
/// - **`N`** — a per-day counter, persisted across app launches in
///   `UserDefaults`. Resets automatically when the date rolls over.
/// - **`XXXX`** — a 4-character base62 random suffix (digits + upper
///   + lower alphabet) used to deduplicate IDs that happen to be
///   minted on the same day with the same counter across multiple
///   machines or concurrent processes.
///
/// IDs are minted at well-defined events — Build Network, Play and
/// Train start, and arena snapshots — and inherited verbatim for
/// most weight copies. See `sampling-parameters.md` and
/// `MODEL_IDS.md` for the full mint / inherit rule set.
struct ModelID: Sendable, Equatable, Hashable, CustomStringConvertible {
    let value: String

    var description: String { value }

    /// Root lineage ID with any trainer-generation suffix removed.
    var lineageRoot: String {
        guard let generation else { return value }
        let suffix = "-\(generation)"
        guard value.hasSuffix(suffix) else { return value }
        return String(value.dropLast(suffix.count))
    }

    /// Optional trainer-generation number. Nil for a lineage root /
    /// champion that has never been forked into a mutable trainer.
    var generation: Int? {
        let parts = value.split(separator: "-", omittingEmptySubsequences: false)
        guard parts.count >= 4, let parsed = Int(parts.last ?? "") else {
            return nil
        }
        return parsed > 0 ? parsed : nil
    }
}

// MARK: - Model ID Minter

/// Namespace for minting new `ModelID` values. Main-actor-isolated so
/// the `UserDefaults` per-date counter increment is race-free; all
/// callers (Build button, Play-and-Train start, arena coordinator)
/// already run on the main actor at the moment they need to mint.
enum ModelIDMinter {
    /// Base62 alphabet used by the random suffix: digits, then
    /// uppercase, then lowercase. Fixed ordering so the suffix
    /// distribution is uniform across all 62 characters.
    private static let base62Alphabet: [Character] = Array(
        "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    )

    /// Number of characters in the random suffix. Four base62 chars
    /// give 62⁴ ≈ 14.8M combinations per (date, counter) pair — more
    /// than enough dedup headroom against simultaneous mints from
    /// multiple machines.
    private static let suffixLength = 4

    /// `UserDefaults` key prefix for the per-date counter. The date
    /// string is appended at mint time so the counter resets at the
    /// UTC day rollover without any explicit cleanup.
    private static let counterKeyPrefix = "ModelIDMinter.counter."

    /// POSIX-locale UTC formatter — fixed format so two users on
    /// different locales mint identically-shaped IDs, and the counter
    /// rolls over on the same instant globally.
    private static let dateFormatter: DateFormatter = {
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.timeZone = TimeZone(identifier: "UTC")
        f.dateFormat = "yyyyMMdd"
        return f
    }()

    /// Mint a fresh `ModelID`. Reads the per-date counter, increments
    /// it, persists the new value, and formats the result together
    /// with a random base62 suffix.
    @MainActor
    static func mint() -> ModelID {
        let dateString = Self.dateFormatter.string(from: Date())
        let counterKey = "\(Self.counterKeyPrefix)\(dateString)"
        let counter = Self.nextCounter(forKey: counterKey)
        let suffix = Self.randomSuffix()
        return ModelID(value: "\(dateString)-\(counter)-\(suffix)")
    }

    /// Mint the next mutable trainer generation for a lineage.
    ///
    /// Examples:
    /// - `20260418-1-AbCd` -> `20260418-1-AbCd-1`
    /// - `20260418-1-AbCd-1` -> `20260418-1-AbCd-2`
    @MainActor
    static func mintTrainerGeneration(from base: ModelID) -> ModelID {
        let nextGeneration = (base.generation ?? 0) + 1
        return ModelID(value: "\(base.lineageRoot)-\(nextGeneration)")
    }

    @MainActor
    private static func nextCounter(forKey key: String) -> Int {
        let defaults = UserDefaults.standard
        let current = defaults.integer(forKey: key)
        let next = current + 1
        defaults.set(next, forKey: key)
        return next
    }

    private static func randomSuffix() -> String {
        var chars: [Character] = []
        chars.reserveCapacity(Self.suffixLength)
        for _ in 0..<Self.suffixLength {
            let index = Int.random(in: 0..<Self.base62Alphabet.count)
            chars.append(Self.base62Alphabet[index])
        }
        return String(chars)
    }
}
