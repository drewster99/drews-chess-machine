import Foundation

// MARK: - Extended arena summary

/// Per-bucket W/D/L tally for one game-length range, from the
/// candidate's perspective. `upperInclusive == nil` marks the
/// open-ended overflow bucket (e.g. "200+ plies").
struct ArenaWDLByLengthBucket: Sendable {
    let lowerInclusive: Int
    /// nil for the overflow bucket.
    let upperInclusive: Int?
    let wins: Int
    let draws: Int
    let losses: Int

    var count: Int { wins + draws + losses }
}

/// Per-bucket mean of the candidate's value-head scalar
/// (`p_win - p_loss ∈ [-1, +1]`, side-to-move == candidate), bucketed
/// by absolute ply index. `lowerInclusive` and `upperInclusive` are
/// 0-indexed ply numbers.
struct ArenaValueByPlyBucket: Sendable {
    let lowerInclusive: Int
    let upperInclusive: Int
    let mean: Float
    let count: Int
}

/// Per-bucket mean of the candidate's value-head scalar, bucketed by
/// the sample's position within its game (`ply / gameLength`, in
/// percent). 20 fixed buckets of width 5%; the last bucket is
/// inclusive of 100 so the very last candidate ply of a finished
/// game lands cleanly.
struct ArenaValueByProgressBucket: Sendable {
    let lowerPercent: Int     // inclusive
    let upperPercent: Int     // exclusive, except final bucket which is inclusive of 100
    let mean: Float
    let count: Int
}

/// Aggregated breakdowns appended to the post-arena `[ARENA]` block.
/// Computed once from `[TournamentGameRecord]` at end-of-arena and
/// not persisted on `TournamentRecord` — these are diagnostic-only
/// and would bloat the session-file schema if stored.
struct ArenaExtendedSummary: Sendable {
    let wdlByLength: [ArenaWDLByLengthBucket]
    let valueByPly: [ArenaValueByPlyBucket]
    let valueByProgress: [ArenaValueByProgressBucket]
}

// MARK: - Aggregation

enum ArenaSummaryAggregator {

    // Bucket widths chosen to be human-readable in the log block:
    //   - length:   20 plies (roughly one "phase" of a chess game)
    //   - ply:      5 plies  (matches user's explicit request)
    //   - progress: 5%       (matches user's explicit request → 20
    //               buckets exactly covering [0, 100])
    //
    // length buckets are 0-19, 20-39, ..., 180-199, then an open-ended
    // overflow bucket "200+". 10 closed buckets + 1 overflow.
    static let lengthBucketWidth = 20
    static let closedLengthBucketCount = 10
    static let plyBucketWidth = 5
    static let progressBucketWidth = 5
    static let progressBucketCount = 100 / progressBucketWidth  // 20

    /// Pure function: aggregate per-game records into the three
    /// breakdowns. Empty `records` yields empty arrays; ply / progress
    /// buckets with zero samples are filtered out, so a downstream
    /// log block can render only populated rows.
    static func aggregate(records: [TournamentGameRecord]) -> ArenaExtendedSummary {
        // (1) W/D/L by length.
        // closedLengthBucketCount closed buckets + 1 overflow at end.
        var wdl: [(w: Int, d: Int, l: Int)] = Array(
            repeating: (0, 0, 0),
            count: closedLengthBucketCount + 1
        )

        // (2) Mean candidate value by absolute ply. Growing as we see
        // longer games; pre-reserve a generous capacity.
        var plyAccum: [(sum: Double, count: Int)] = []
        plyAccum.reserveCapacity(40)

        // (3) Mean candidate value by progress-percent. Fixed
        // `progressBucketCount` buckets of width `progressBucketWidth`.
        var progressAccum: [(sum: Double, count: Int)] = Array(
            repeating: (0, 0),
            count: progressBucketCount
        )

        for rec in records {
            let length = rec.moveHistory.count
            let lengthBucketIdx = min(length / lengthBucketWidth, closedLengthBucketCount)
            updateWDL(&wdl[lengthBucketIdx], result: rec.result, candidateIsWhite: rec.aIsWhite)

            for sample in rec.candidateValueSamples {
                // Absolute-ply bucket.
                let pBucket = max(0, sample.ply) / plyBucketWidth
                while plyAccum.count <= pBucket {
                    plyAccum.append((0, 0))
                }
                plyAccum[pBucket].sum += Double(sample.value)
                plyAccum[pBucket].count += 1

                // Progress-percent bucket. Requires length >= 1; the
                // candidate samples loop already implies length >= 1
                // (a sample exists only if a candidate ply happened),
                // but guard explicitly for safety.
                if length > 0 {
                    let pct = (sample.ply * 100) / length
                    let progBucket = min(pct / progressBucketWidth, progressBucketCount - 1)
                    progressAccum[progBucket].sum += Double(sample.value)
                    progressAccum[progBucket].count += 1
                }
            }
        }

        // Build length output (always include every bucket so the log
        // table is comparable across arenas; rendering-time filtering
        // is the caller's choice).
        var wdlOut: [ArenaWDLByLengthBucket] = []
        wdlOut.reserveCapacity(closedLengthBucketCount + 1)
        for i in 0..<closedLengthBucketCount {
            let lo = i * lengthBucketWidth
            let hi = lo + lengthBucketWidth - 1
            wdlOut.append(ArenaWDLByLengthBucket(
                lowerInclusive: lo,
                upperInclusive: hi,
                wins: wdl[i].w,
                draws: wdl[i].d,
                losses: wdl[i].l
            ))
        }
        wdlOut.append(ArenaWDLByLengthBucket(
            lowerInclusive: closedLengthBucketCount * lengthBucketWidth,
            upperInclusive: nil,
            wins: wdl[closedLengthBucketCount].w,
            draws: wdl[closedLengthBucketCount].d,
            losses: wdl[closedLengthBucketCount].l
        ))

        // Build ply output (drop empty trailing buckets).
        var plyOut: [ArenaValueByPlyBucket] = []
        plyOut.reserveCapacity(plyAccum.count)
        for (i, entry) in plyAccum.enumerated() where entry.count > 0 {
            let lo = i * plyBucketWidth
            let hi = lo + plyBucketWidth - 1
            plyOut.append(ArenaValueByPlyBucket(
                lowerInclusive: lo,
                upperInclusive: hi,
                mean: Float(entry.sum / Double(entry.count)),
                count: entry.count
            ))
        }

        // Build progress output (drop empty buckets so a draw-heavy
        // very-short-games arena doesn't render mostly-empty rows).
        var progressOut: [ArenaValueByProgressBucket] = []
        progressOut.reserveCapacity(progressBucketCount)
        for (i, entry) in progressAccum.enumerated() where entry.count > 0 {
            let lo = i * progressBucketWidth
            let hi = lo + progressBucketWidth
            progressOut.append(ArenaValueByProgressBucket(
                lowerPercent: lo,
                upperPercent: hi,
                mean: Float(entry.sum / Double(entry.count)),
                count: entry.count
            ))
        }

        return ArenaExtendedSummary(
            wdlByLength: wdlOut,
            valueByPly: plyOut,
            valueByProgress: progressOut
        )
    }

    private static func updateWDL(
        _ bucket: inout (w: Int, d: Int, l: Int),
        result: GameResult,
        candidateIsWhite: Bool
    ) {
        switch result {
        case .checkmate(let winner):
            let candidateWon = (winner == .white && candidateIsWhite)
                || (winner == .black && !candidateIsWhite)
            if candidateWon {
                bucket.w += 1
            } else {
                bucket.l += 1
            }
        case .stalemate,
             .drawByFiftyMoveRule,
             .drawByInsufficientMaterial,
             .drawByThreefoldRepetition:
            bucket.d += 1
        }
    }
}
