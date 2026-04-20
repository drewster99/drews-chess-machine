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
/// Thread-safe via a private serial `DispatchQueue`. All stored data
/// is bounded by the window size.
final class GameDiversityTracker: @unchecked Sendable {

    /// Divergence-ply bucket upper bounds (inclusive). A game's
    /// divergence ply falls into the lowest bucket whose upper bound
    /// it does not exceed. The last bucket catches everything above
    /// the highest bound.
    ///
    /// Chosen to make the "policy collapse" tail (bucket 5) visually
    /// pop even at low counts: the first four buckets cover the range
    /// where diverse self-play sits in steady state (0–20 plies) and
    /// the last two highlight the pathological deep-share regime.
    static let histogramBounds: [Int] = [2, 5, 10, 20, 40]
    /// Human-readable labels for each histogram bucket, aligned with
    /// `histogramBounds` plus a trailing "41+" overflow bucket. Used
    /// by the UI to render axis labels on the diversity chart.
    static let histogramLabels: [String] = ["0-2", "3-5", "6-10", "11-20", "21-40", "41+"]
    /// Number of buckets in the divergence histogram.
    static var histogramBucketCount: Int { histogramBounds.count + 1 }

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
        /// Count of games in each divergence-ply bucket. Aligned
        /// with `GameDiversityTracker.histogramLabels`. Sums to
        /// `gamesInWindow`. Empty only when the window holds zero
        /// games; otherwise always `histogramBucketCount` entries.
        let divergenceHistogram: [Int]
    }

    private let windowSize: Int
    private let queue = DispatchQueue(label: "drewschess.gamediversitytracker.serial")

    // Circular buffers — pre-allocated at init, indexed by writeIndex.
    private var sequences: [[Int16]]      // policy-index per move
    private var hashes: [UInt64]          // FNV-1a of the full sequence
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
    }

    /// Record a completed game's move sequence.
    ///
    /// - Parameter moves: The full ordered list of moves played in
    ///   the game (both players interleaved, white first). The
    ///   tracker extracts each move's `PolicyEncoding.policyIndex`
    ///   for compact storage and comparison. Player alternates white
    ///   then black starting from index 0; the encoding is
    ///   deterministic per (move, player) so any two games that play
    ///   the same physical move at the same ply produce the same
    ///   stored index, which is what diversity comparison needs.
    func recordGame(moves: [ChessMove]) {
        let indices = moves.enumerated().map { (i, move) -> Int16 in
            let player: PieceColor = (i % 2 == 0) ? .white : .black
            return Int16(clamping: PolicyEncoding.policyIndex(move, currentPlayer: player))
        }
        let hash = Self.fnv1a(indices)

        queue.async { [weak self] in
            guard let self else { return }
            self.sequences[self.writeIndex] = indices
            self.hashes[self.writeIndex] = hash
            self.writeIndex = (self.writeIndex + 1) % self.windowSize
            if self.stored < self.windowSize { self.stored += 1 }
        }
    }

    /// Take an immutable snapshot of the current diversity metrics.
    ///
    /// Divergence-ply values are recomputed against the *current*
    /// window contents on every call, not cached from the moment each
    /// game was inserted. This matters because as old games age out of
    /// the FIFO and new ones come in, the set of peers each game can be
    /// compared against changes — a cached at-insert-time value would
    /// drift out of sync with the live window. The pairwise scan is
    /// O(n²) in window size; with n=200 and ~150-ply games it's well
    /// under a millisecond on Apple Silicon.
    func snapshot() -> Snapshot {
        queue.sync {
            guard stored > 0 else {
                return Snapshot(
                    gamesInWindow: 0,
                    uniqueGames: 0,
                    uniquePercent: 100,
                    avgDivergencePly: 0,
                    divergenceHistogram: Array(repeating: 0, count: Self.histogramBucketCount)
                )
            }

            // Symmetric pairwise prefix scan: each unordered pair (i, j)
            // is compared once and the shared length is folded into both
            // games' running max. A game's divergence ply is the longest
            // prefix it shares with any other game currently stored.
            var maxPrefix = Array(repeating: 0, count: stored)
            var hashSet = Set<UInt64>(minimumCapacity: stored)
            for i in 0..<stored {
                hashSet.insert(hashes[i])
                let seqI = sequences[i]
                for j in (i + 1)..<stored {
                    let seqJ = sequences[j]
                    var shared = 0
                    let limit = min(seqI.count, seqJ.count)
                    while shared < limit && seqI[shared] == seqJ[shared] {
                        shared += 1
                    }
                    if shared > maxPrefix[i] { maxPrefix[i] = shared }
                    if shared > maxPrefix[j] { maxPrefix[j] = shared }
                }
            }

            var divergenceSum = 0
            var histogram = Array(repeating: 0, count: Self.histogramBucketCount)
            let bounds = Self.histogramBounds
            for i in 0..<stored {
                let ply = maxPrefix[i]
                divergenceSum += ply
                // Linear scan over bounds — 5 compares, fastest for this
                // tiny count. Matches `bucketIndex(for:)` semantics.
                var bucket = bounds.count  // overflow bucket by default
                for (idx, upper) in bounds.enumerated() where ply <= upper {
                    bucket = idx
                    break
                }
                histogram[bucket] += 1
            }

            let unique = hashSet.count
            return Snapshot(
                gamesInWindow: stored,
                uniqueGames: unique,
                uniquePercent: Double(unique) / Double(stored) * 100,
                avgDivergencePly: Double(divergenceSum) / Double(stored),
                divergenceHistogram: histogram
            )
        }
    }

    /// Reset the tracker, discarding all stored games. Used when a
    /// new arena tournament starts so its diversity stats are
    /// isolated from the previous tournament.
    func reset() {
        queue.async { [weak self] in
            guard let self else { return }
            for i in 0..<self.stored { self.sequences[i] = [] }
            self.stored = 0
            self.writeIndex = 0
        }
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
