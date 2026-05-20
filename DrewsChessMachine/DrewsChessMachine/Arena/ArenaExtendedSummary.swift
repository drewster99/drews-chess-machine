import Foundation

// MARK: - Extended arena summary

/// Per-bucket W/D/L tally for one game-length range, from the
/// candidate's perspective. `upperInclusive == nil` marks the
/// open-ended overflow bucket (e.g. "200+ plies").
struct ArenaWDLByLengthBucket: Sendable, Codable, Equatable {
    let lowerInclusive: Int
    /// nil for the overflow bucket.
    let upperInclusive: Int?
    let wins: Int
    let draws: Int
    let losses: Int

    var count: Int { wins + draws + losses }

    /// AlphaZero-style mean score `(W + 0.5·D) / N` for this length
    /// bucket — same identity used to decide promotion at the
    /// tournament level. 0 when the bucket is empty.
    var candidateScore: Double {
        guard count > 0 else { return 0 }
        return (Double(wins) + 0.5 * Double(draws)) / Double(count)
    }
}

/// Per-bucket statistics for candidate-to-move plies bucketed by
/// absolute ply index.
///
/// Carries two distinct signals over the same sample population:
///   - `mean` — mean of the candidate network's value-head scalar
///     (`p_win - p_loss ∈ [-1, +1]`) at those plies. The network's
///     own self-assessment.
///   - `wins / draws / losses` — each ply attributed to its game's
///     eventual outcome from the candidate's perspective. The
///     derived `candidateScore = (W + 0.5·D) / N` is the same
///     metric the arena uses to score a tournament, but bucketed
///     by ply so the user can see *at which point in the game* the
///     candidate is actually winning rather than just *what it
///     thinks*. The two signals together form a calibration view.
struct ArenaValueByPlyBucket: Sendable, Codable, Equatable {
    let lowerInclusive: Int
    let upperInclusive: Int
    /// Mean candidate value scalar over plies in this bucket.
    let mean: Float
    /// Per-bucket W/D/L attribution: each candidate-to-move ply
    /// that fell in this bucket adds one to W, D, or L depending on
    /// the eventual outcome of the game it came from. Sum =
    /// `count`.
    let wins: Int
    let draws: Int
    let losses: Int

    var count: Int { wins + draws + losses }

    /// AlphaZero-style score over plies in this bucket. 0 when the
    /// bucket is empty.
    var candidateScore: Double {
        guard count > 0 else { return 0 }
        return (Double(wins) + 0.5 * Double(draws)) / Double(count)
    }
}

/// Per-bucket statistics for candidate-to-move plies bucketed by
/// position within the game (`ply / gameLength`, in percent). 20
/// fixed buckets of width 5%; the last bucket is inclusive of 100
/// so the very last candidate ply of a finished game lands cleanly.
///
/// Same dual signal as `ArenaValueByPlyBucket` — see that doc for
/// the rationale on carrying both `mean` and the W/D/L attribution.
struct ArenaValueByProgressBucket: Sendable, Codable, Equatable {
    let lowerPercent: Int     // inclusive
    let upperPercent: Int     // exclusive, except final bucket which is inclusive of 100
    let mean: Float
    let wins: Int
    let draws: Int
    let losses: Int

    var count: Int { wins + draws + losses }

    var candidateScore: Double {
        guard count > 0 else { return 0 }
        return (Double(wins) + 0.5 * Double(draws)) / Double(count)
    }
}

/// Aggregated breakdowns appended to the post-arena `[ARENA]` block
/// and persisted on `TournamentRecord` so the row's detail popover
/// can render histograms after a session save/load round-trip.
struct ArenaExtendedSummary: Sendable, Codable, Equatable {
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

    /// Outcome category for a single game, from the candidate's
    /// perspective. Used both to update the length-bucket W/D/L
    /// tally and to attribute every per-ply sample from the same
    /// game to the same outcome bucket.
    private enum CandidateOutcome { case win, draw, loss }

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

        // (2) Mean candidate value + W/D/L attribution by absolute
        // ply. Growing as we see longer games; pre-reserve a
        // generous capacity.
        var plyAccum: [PlyAccumulator] = []
        plyAccum.reserveCapacity(40)

        // (3) Same payload, bucketed by progress percent. Fixed
        // `progressBucketCount` buckets of width `progressBucketWidth`.
        var progressAccum: [PlyAccumulator] = Array(
            repeating: PlyAccumulator(),
            count: progressBucketCount
        )

        for rec in records {
            let length = rec.moveHistory.count
            let lengthBucketIdx = min(length / lengthBucketWidth, closedLengthBucketCount)
            let outcome = candidateOutcome(result: rec.result, candidateIsWhite: rec.aIsWhite)
            updateWDL(&wdl[lengthBucketIdx], outcome: outcome)

            for sample in rec.candidateValueSamples {
                // Absolute-ply bucket.
                let pBucket = max(0, sample.ply) / plyBucketWidth
                while plyAccum.count <= pBucket {
                    plyAccum.append(PlyAccumulator())
                }
                plyAccum[pBucket].add(value: sample.value, outcome: outcome)

                // Progress-percent bucket. Requires length >= 1; the
                // candidate samples loop already implies length >= 1
                // (a sample exists only if a candidate ply happened),
                // but guard explicitly for safety.
                if length > 0 {
                    let pct = (sample.ply * 100) / length
                    let progBucket = min(pct / progressBucketWidth, progressBucketCount - 1)
                    progressAccum[progBucket].add(value: sample.value, outcome: outcome)
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
                mean: entry.mean,
                wins: entry.wins,
                draws: entry.draws,
                losses: entry.losses
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
                mean: entry.mean,
                wins: entry.wins,
                draws: entry.draws,
                losses: entry.losses
            ))
        }

        return ArenaExtendedSummary(
            wdlByLength: wdlOut,
            valueByPly: plyOut,
            valueByProgress: progressOut
        )
    }

    /// Mutable running accumulator for one ply / progress bucket.
    /// Tracks the value-scalar sum (for `mean`) and the per-outcome
    /// counts (for `candidateScore`) without exposing the raw
    /// running sum to callers.
    private struct PlyAccumulator {
        var sum: Double = 0
        var wins: Int = 0
        var draws: Int = 0
        var losses: Int = 0

        var count: Int { wins + draws + losses }
        var mean: Float {
            guard count > 0 else { return 0 }
            return Float(sum / Double(count))
        }

        mutating func add(value: Float, outcome: CandidateOutcome) {
            sum += Double(value)
            switch outcome {
            case .win:  wins += 1
            case .draw: draws += 1
            case .loss: losses += 1
            }
        }
    }

    private static func candidateOutcome(
        result: GameResult,
        candidateIsWhite: Bool
    ) -> CandidateOutcome {
        switch result {
        case .checkmate(let winner):
            let won = (winner == .white && candidateIsWhite)
                || (winner == .black && !candidateIsWhite)
            return won ? .win : .loss
        case .stalemate,
             .drawByFiftyMoveRule,
             .drawByInsufficientMaterial,
             .drawByThreefoldRepetition:
            return .draw
        }
    }

    private static func updateWDL(
        _ bucket: inout (w: Int, d: Int, l: Int),
        outcome: CandidateOutcome
    ) {
        switch outcome {
        case .win:  bucket.w += 1
        case .draw: bucket.d += 1
        case .loss: bucket.l += 1
        }
    }
}
