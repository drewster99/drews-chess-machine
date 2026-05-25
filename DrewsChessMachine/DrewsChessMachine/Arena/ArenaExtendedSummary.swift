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
/// Carries four signals over the same sample population:
///   - `mean` — mean of the candidate network's value-head scalar
///     (`p_win - p_loss ∈ [-1, +1]`) at those plies. The network's
///     own self-assessment of the position.
///   - `meanPolicyProbability` — mean probability the candidate's
///     policy placed on the move it actually played. How committed
///     the policy was, distinct from how good it judged the
///     position to be.
///   - `meanMaterialAdvantage` — mean candidate material advantage
///     (standard piece values, candidate minus opponent) at those
///     plies. Feeds the 3D arena surface's material-advantage metric.
///   - `wins / draws / losses` — each ply attributed to its game's
///     eventual outcome from the candidate's perspective. The
///     derived `candidateScore = (W + 0.5·D) / N` is the same
///     metric the arena uses to score a tournament, but bucketed
///     by ply so the user can see *at which point in the game* the
///     candidate is actually winning rather than just *what it
///     thinks*. The signals together form a calibration view.
struct ArenaValueByPlyBucket: Sendable, Codable, Equatable {
    let lowerInclusive: Int
    let upperInclusive: Int
    /// Mean candidate value scalar over plies in this bucket.
    let mean: Float
    /// Mean chosen-move policy probability over plies in this bucket
    /// — how committed the candidate's policy was, on average, to the
    /// moves it actually played here. `nil` for summaries persisted
    /// before this field was added.
    let meanPolicyProbability: Float?
    /// Mean candidate material advantage over plies in this bucket,
    /// in standard piece values. `nil` for summaries persisted before
    /// this field was added.
    let meanMaterialAdvantage: Float?
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
/// Same signals as `ArenaValueByPlyBucket` — see that doc for the
/// rationale on carrying `mean`, `meanPolicyProbability`, and the
/// W/D/L attribution.
struct ArenaValueByProgressBucket: Sendable, Codable, Equatable {
    let lowerPercent: Int     // inclusive
    let upperPercent: Int     // exclusive, except final bucket which is inclusive of 100
    let mean: Float
    /// Mean chosen-move policy probability over plies in this bucket.
    /// `nil` for summaries persisted before this field was added.
    let meanPolicyProbability: Float?
    let wins: Int
    let draws: Int
    let losses: Int

    var count: Int { wins + draws + losses }

    var candidateScore: Double {
        guard count > 0 else { return 0 }
        return (Double(wins) + 0.5 * Double(draws)) / Double(count)
    }
}

/// Per-bucket statistics for candidate-to-move plies bucketed by the
/// candidate's material advantage — its standard piece-value sum
/// minus the opponent's. `advantage` is clamped to ±9 by the
/// aggregator, so the two end buckets are overflow ("a queen up or
/// more" / "a queen down or more").
///
/// Carries the same value + W/D/L signals as `ArenaValueByPlyBucket`:
/// `mean` is the value-head self-assessment at this material level,
/// `candidateScore` is how those games actually scored. Comparing the
/// two against the material axis is a calibration view orthogonal to
/// the by-ply / by-progress phase charts.
struct ArenaValueByMaterialAdvantageBucket: Sendable, Codable, Equatable {
    /// Material advantage this bucket represents, clamped to ±9.
    let advantage: Int
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

/// Per-bucket statistics for candidate-to-move plies bucketed by the
/// total non-king material on the board — both sides summed, standard
/// piece values. A game-phase axis: a high total marks the opening,
/// a low total the endgame.
///
/// Same value + W/D/L signals as `ArenaValueByPlyBucket`.
struct ArenaValueByTotalMaterialBucket: Sendable, Codable, Equatable {
    let lowerInclusive: Int
    let upperInclusive: Int
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
/// and persisted on `TournamentRecord` so the row's detail view can
/// render histograms after a session save/load round-trip.
struct ArenaExtendedSummary: Sendable, Codable, Equatable {
    let wdlByLength: [ArenaWDLByLengthBucket]
    let valueByPly: [ArenaValueByPlyBucket]
    let valueByProgress: [ArenaValueByProgressBucket]
    /// Candidate value + arena-style score bucketed by material
    /// advantage. Empty for summaries persisted before the material
    /// breakdowns existed (see the custom decoder below).
    let valueByMaterialAdvantage: [ArenaValueByMaterialAdvantageBucket]
    /// Same, bucketed by total material on the board.
    let valueByTotalMaterial: [ArenaValueByTotalMaterialBucket]

    init(
        wdlByLength: [ArenaWDLByLengthBucket],
        valueByPly: [ArenaValueByPlyBucket],
        valueByProgress: [ArenaValueByProgressBucket],
        valueByMaterialAdvantage: [ArenaValueByMaterialAdvantageBucket],
        valueByTotalMaterial: [ArenaValueByTotalMaterialBucket]
    ) {
        self.wdlByLength = wdlByLength
        self.valueByPly = valueByPly
        self.valueByProgress = valueByProgress
        self.valueByMaterialAdvantage = valueByMaterialAdvantage
        self.valueByTotalMaterial = valueByTotalMaterial
    }

    private enum CodingKeys: String, CodingKey {
        case wdlByLength, valueByPly, valueByProgress
        case valueByMaterialAdvantage, valueByTotalMaterial
    }

    /// Custom decoder so summaries persisted before the material
    /// breakdowns existed still load: the two material arrays default
    /// to empty when their keys are absent. The three original arrays
    /// stay required — every persisted summary has carried them.
    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        wdlByLength = try container.decode([ArenaWDLByLengthBucket].self, forKey: .wdlByLength)
        valueByPly = try container.decode([ArenaValueByPlyBucket].self, forKey: .valueByPly)
        valueByProgress = try container.decode([ArenaValueByProgressBucket].self, forKey: .valueByProgress)
        valueByMaterialAdvantage = try container.decodeIfPresent(
            [ArenaValueByMaterialAdvantageBucket].self,
            forKey: .valueByMaterialAdvantage
        ) ?? []
        valueByTotalMaterial = try container.decodeIfPresent(
            [ArenaValueByTotalMaterialBucket].self,
            forKey: .valueByTotalMaterial
        ) ?? []
    }
}

// MARK: - Aggregation

enum ArenaSummaryAggregator {

    // Bucket widths chosen to be human-readable in the log block:
    //   - length:   `lengthBucketWidth` plies (roughly one "phase"
    //               of a chess game)
    //   - ply:      `plyBucketWidth` plies
    //   - progress: `progressBucketWidth` % (chosen to divide 100
    //               evenly → `progressBucketCount` buckets covering
    //               [0, 100])
    //
    // length buckets are `closedLengthBucketCount` closed
    // `lengthBucketWidth`-wide buckets starting at 0, then an
    // open-ended overflow bucket above
    // `closedLengthBucketCount * lengthBucketWidth`.
    static let lengthBucketWidth = 20
    static let closedLengthBucketCount = 10
    static let plyBucketWidth = 20
    static let progressBucketWidth = 5
    static let progressBucketCount = 100 / progressBucketWidth  // 20

    // Material-advantage axis: one bucket per integer point of
    // advantage, clamped to ±this — the end buckets are overflow
    // ("a queen up or more"). 2·clamp + 1 buckets total.
    static let materialAdvantageClamp = 9
    // Total-material axis: bucket width
    // (`totalMaterialBucketWidth`) in standard piece-value points.
    // Total non-king material starts at the sum of the standard
    // material values for both sides (per `BoardEncoder.materialValue`)
    // and can exceed it after promotions, so the total-material
    // accumulator grows dynamically rather than assuming a fixed
    // maximum.
    static let totalMaterialBucketWidth = 6

    /// Outcome category for a single game, from the candidate's
    /// perspective. Used both to update the length-bucket W/D/L
    /// tally and to attribute every per-ply sample from the same
    /// game to the same outcome bucket.
    private enum CandidateOutcome { case win, draw, loss }

    /// Pure function: aggregate per-game records into the breakdowns.
    /// Empty `records` yields empty arrays; ply / progress / material
    /// buckets with zero samples are filtered out, so a downstream log
    /// block can render only populated rows.
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

        // (4) Same payload, bucketed by material advantage. Fixed
        // `2·clamp + 1` buckets; index `clampedAdvantage + clamp`.
        var materialAccum: [PlyAccumulator] = Array(
            repeating: PlyAccumulator(),
            count: 2 * materialAdvantageClamp + 1
        )

        // (5) Same payload, bucketed by total material on the board.
        // Grows as we see material-heavy positions (promotions can
        // push a total well past the 78-point starting material).
        var totalMaterialAccum: [PlyAccumulator] = []
        totalMaterialAccum.reserveCapacity(16)

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
                plyAccum[pBucket].add(sample: sample, outcome: outcome)

                // Progress-percent bucket. Requires length >= 1; the
                // candidate samples loop already implies length >= 1
                // (a sample exists only if a candidate ply happened),
                // but guard explicitly for safety.
                if length > 0 {
                    let pct = (sample.ply * 100) / length
                    let progBucket = min(pct / progressBucketWidth, progressBucketCount - 1)
                    progressAccum[progBucket].add(sample: sample, outcome: outcome)
                }

                // Material-advantage bucket (clamped to ±clamp).
                let clampedAdv = max(
                    -materialAdvantageClamp,
                    min(materialAdvantageClamp, sample.materialAdvantage)
                )
                materialAccum[clampedAdv + materialAdvantageClamp].add(sample: sample, outcome: outcome)

                // Total-material bucket (dynamically grown).
                let tBucket = max(0, sample.totalMaterial) / totalMaterialBucketWidth
                while totalMaterialAccum.count <= tBucket {
                    totalMaterialAccum.append(PlyAccumulator())
                }
                totalMaterialAccum[tBucket].add(sample: sample, outcome: outcome)
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
                meanPolicyProbability: entry.meanPolicy,
                meanMaterialAdvantage: entry.meanMaterial,
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
                meanPolicyProbability: entry.meanPolicy,
                wins: entry.wins,
                draws: entry.draws,
                losses: entry.losses
            ))
        }

        // Build material-advantage output (drop empty buckets).
        var materialOut: [ArenaValueByMaterialAdvantageBucket] = []
        materialOut.reserveCapacity(materialAccum.count)
        for (i, entry) in materialAccum.enumerated() where entry.count > 0 {
            materialOut.append(ArenaValueByMaterialAdvantageBucket(
                advantage: i - materialAdvantageClamp,
                mean: entry.mean,
                wins: entry.wins,
                draws: entry.draws,
                losses: entry.losses
            ))
        }

        // Build total-material output (drop empty buckets).
        var totalMaterialOut: [ArenaValueByTotalMaterialBucket] = []
        totalMaterialOut.reserveCapacity(totalMaterialAccum.count)
        for (i, entry) in totalMaterialAccum.enumerated() where entry.count > 0 {
            let lo = i * totalMaterialBucketWidth
            let hi = lo + totalMaterialBucketWidth - 1
            totalMaterialOut.append(ArenaValueByTotalMaterialBucket(
                lowerInclusive: lo,
                upperInclusive: hi,
                mean: entry.mean,
                wins: entry.wins,
                draws: entry.draws,
                losses: entry.losses
            ))
        }

        return ArenaExtendedSummary(
            wdlByLength: wdlOut,
            valueByPly: plyOut,
            valueByProgress: progressOut,
            valueByMaterialAdvantage: materialOut,
            valueByTotalMaterial: totalMaterialOut
        )
    }

    /// Mutable running accumulator for one ply / progress / material
    /// bucket. Tracks the value-scalar, policy-probability and
    /// material-advantage sums (for the three means) plus the
    /// per-outcome counts (for `candidateScore`).
    private struct PlyAccumulator {
        var sum: Double = 0
        var policySum: Double = 0
        var materialSum: Double = 0
        var wins: Int = 0
        var draws: Int = 0
        var losses: Int = 0

        var count: Int { wins + draws + losses }
        var mean: Float {
            guard count > 0 else { return 0 }
            return Float(sum / Double(count))
        }
        var meanPolicy: Float {
            guard count > 0 else { return 0 }
            return Float(policySum / Double(count))
        }
        var meanMaterial: Float {
            guard count > 0 else { return 0 }
            return Float(materialSum / Double(count))
        }

        mutating func add(sample: CandidateValueSample, outcome: CandidateOutcome) {
            sum += Double(sample.value)
            policySum += Double(sample.policyProbability)
            materialSum += Double(sample.materialAdvantage)
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

// MARK: - Material

/// Material accounting for an arena position. Pure, so it can be
/// unit-tested independently of the tournament driver.
enum ArenaMaterial {
    /// Sum the standard piece values on `board`, returning the
    /// candidate's edge (its material minus the opponent's) and the
    /// total non-king material on the board. `candidateColor` is the
    /// side the candidate is playing — at a candidate-to-move ply
    /// that's the side to move, so `advantage` shares the perspective
    /// of the value-head scalar.
    static func summary(board: [Piece?], candidateColor: PieceColor) -> (advantage: Int, total: Int) {
        var advantage = 0
        var total = 0
        for square in board {
            guard let piece = square else { continue }
            let value = piece.type.materialValue
            total += value
            advantage += (piece.color == candidateColor) ? value : -value
        }
        return (advantage, total)
    }
}
