import Foundation

/// Tracks the diversity of games in a rolling window by comparing
/// move sequences. Designed for use in self-play (shared across
/// workers) and arena evaluation (one per tournament).
///
/// For each recorded game the tracker computes:
/// - Whether the full move sequence is an exact duplicate of any
///   game already in the window (via FNV-1a hash comparison).
/// - The **divergence ply** — the length of the longest prefix the
///   new game shares with any stored game. A low divergence ply
///   means games branch early (high diversity); a high value means
///   games follow similar lines deep into the middlegame or endgame.
///
/// Thread-safe via `NSLock`. All stored data is bounded by the
/// window size.
final class GameDiversityTracker: @unchecked Sendable {

    /// Immutable snapshot of the tracker's current state, safe to
    /// read from any thread (typically the UI heartbeat).
    struct Snapshot: Sendable {
        /// Number of games currently stored in the rolling window.
        let gamesInWindow: Int
        /// Number of distinct game hashes in the window.
        let uniqueGames: Int
        /// `uniqueGames / gamesInWindow * 100`, or 100 when empty.
        let uniquePercent: Double
        /// Mean divergence ply across all games in the window.
        /// Zero when fewer than 2 games have been recorded.
        let avgDivergencePly: Double
    }

    private let windowSize: Int
    private let lock = NSLock()

    // Circular buffers — pre-allocated at init, indexed by writeIndex.
    private var sequences: [[Int16]]      // policy-index per move
    private var hashes: [UInt64]          // FNV-1a of the full sequence
    private var divergencePlies: [Int]    // max shared prefix at record time
    private var writeIndex: Int = 0
    private var stored: Int = 0

    /// Create a tracker with the given rolling-window capacity.
    /// - Parameter windowSize: Maximum number of games to retain.
    ///   Older games are overwritten in FIFO order. Defaults to 200.
    init(windowSize: Int = 200) {
        precondition(windowSize > 0, "GameDiversityTracker window must be > 0")
        self.windowSize = windowSize
        self.sequences = Array(repeating: [], count: windowSize)
        self.hashes = Array(repeating: 0, count: windowSize)
        self.divergencePlies = Array(repeating: 0, count: windowSize)
    }

    /// Record a completed game's move sequence.
    ///
    /// - Parameter moves: The full ordered list of moves played in
    ///   the game (both players interleaved, white first). The
    ///   tracker extracts each move's `policyIndex` for compact
    ///   storage and comparison.
    func recordGame(moves: [ChessMove]) {
        let indices = moves.map { Int16(clamping: $0.policyIndex) }
        let hash = Self.fnv1a(indices)

        lock.lock()
        defer { lock.unlock() }

        // Find the longest shared prefix with any stored game.
        var maxPrefix = 0
        for i in 0..<stored {
            let other = sequences[i]
            var shared = 0
            let limit = min(indices.count, other.count)
            while shared < limit && indices[shared] == other[shared] {
                shared += 1
            }
            if shared > maxPrefix { maxPrefix = shared }
        }

        sequences[writeIndex] = indices
        hashes[writeIndex] = hash
        divergencePlies[writeIndex] = maxPrefix
        writeIndex = (writeIndex + 1) % windowSize
        if stored < windowSize { stored += 1 }
    }

    /// Take an immutable snapshot of the current diversity metrics.
    func snapshot() -> Snapshot {
        lock.lock()
        defer { lock.unlock() }

        guard stored > 0 else {
            return Snapshot(gamesInWindow: 0, uniqueGames: 0,
                            uniquePercent: 100, avgDivergencePly: 0)
        }

        var hashSet = Set<UInt64>(minimumCapacity: stored)
        var divergenceSum = 0
        for i in 0..<stored {
            hashSet.insert(hashes[i])
            divergenceSum += divergencePlies[i]
        }

        let unique = hashSet.count
        return Snapshot(
            gamesInWindow: stored,
            uniqueGames: unique,
            uniquePercent: Double(unique) / Double(stored) * 100,
            avgDivergencePly: Double(divergenceSum) / Double(stored)
        )
    }

    /// Reset the tracker, discarding all stored games. Used when a
    /// new arena tournament starts so its diversity stats are
    /// isolated from the previous tournament.
    func reset() {
        lock.lock()
        defer { lock.unlock() }
        for i in 0..<stored { sequences[i] = [] }
        stored = 0
        writeIndex = 0
    }

    // MARK: - FNV-1a hash

    /// FNV-1a 64-bit hash over a sequence of Int16 policy indices.
    /// Chosen for speed, simplicity, and good distribution — the
    /// tracker only needs collision resistance within a 200-game
    /// window, not cryptographic strength.
    private static func fnv1a(_ values: [Int16]) -> UInt64 {
        var hash: UInt64 = 0xcbf29ce484222325  // FNV offset basis
        let prime: UInt64 = 0x100000001b3       // FNV prime
        for value in values {
            let lo = UInt8(truncatingIfNeeded: UInt16(bitPattern: value))
            let hi = UInt8(truncatingIfNeeded: UInt16(bitPattern: value) >> 8)
            hash ^= UInt64(lo)
            hash &*= prime
            hash ^= UInt64(hi)
            hash &*= prime
        }
        return hash
    }
}
