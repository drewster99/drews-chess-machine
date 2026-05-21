import Foundation

/// Pure-Swift formatters for the `[ARENA]` session-log block. Split
/// out of `logArenaResult` so the string-building logic can be
/// exercised by XCTest without spinning up a `ContentView` or any
/// Metal/MPSGraph dependencies.
///
/// Two outputs share a single record input:
///   1. `formatHumanReadable(...)` → array of lines for the
///      multi-line display block (Games / Result / Score / Elo /
///      Draw rate / By side / params / IDs / diversity / Verdict).
///   2. `formatKVLine(...)` → single `[ARENA] #N kv …` line carrying
///      every parse-target key=value pair for grep-based tooling.
///
/// Both functions take only plain-value context parameters — no
/// ChessTrainer, no MPSNetwork handles — so tests can feed synthetic
/// records through without any engine scaffolding.
enum ArenaLogFormatter {

    /// The subset of per-session parameters that appear in the
    /// human-readable block's parameter line. Pulled out as a
    /// struct so tests don't need to construct a trainer or
    /// sampling schedule to exercise the formatter.
    struct Parameters: Sendable {
        let batchSize: Int
        let learningRate: Float
        let promoteThreshold: Double
        let tournamentGames: Int
        let spStartTau: Float
        let spFloorTau: Float
        let spDecayPerPly: Float
        let arStartTau: Float
        let arFloorTau: Float
        let arDecayPerPly: Float
        let workerCount: Int
        let buildNumber: Int
    }

    /// Diversity snapshot fields needed for the "diversity:" line.
    /// Mirror of `GameDiversityTracker.Snapshot` reduced to the
    /// four scalars the log prints — keeps the formatter decoupled
    /// from the tracker's concrete type for testing.
    struct Diversity: Sendable {
        let uniqueGames: Int
        let gamesInWindow: Int
        let uniquePercent: Double
        let avgDivergencePly: Double
    }

    // MARK: - Human-readable block

    /// Build the multi-line human-readable `[ARENA]` block.
    /// Returns the lines as an array so the caller can `print` /
    /// `SessionLogger.log` each one independently (matching the
    /// pre-extraction call pattern).
    ///
    /// `extended` carries the post-arena breakdowns appended after
    /// the core block: W/D/L by game-length bucket, candidate value
    /// scalar by absolute ply, and candidate value scalar by game
    /// progress percentage. Pass `nil` to skip the breakdown lines
    /// (e.g. tests, legacy callers).
    static func formatHumanReadable(
        record: TournamentRecord,
        index: Int,
        candidateID: String,
        championID: String,
        trainerID: String,
        parameters p: Parameters,
        diversity d: Diversity,
        extended: ArenaExtendedSummary? = nil
    ) -> [String] {
        let durationStr = formatDuration(record.durationSec)
        let verdictStr = formatVerdict(record: record)

        let lrStr = String(format: "%.1e", p.learningRate)
        let threshStr = String(format: "%.2f", p.promoteThreshold)
        let spTauStr = String(
            format: "%.2f/%.2f/%.3f",
            Double(p.spStartTau),
            Double(p.spFloorTau),
            Double(p.spDecayPerPly)
        )
        let arTauStr = String(
            format: "%.2f/%.2f/%.3f",
            Double(p.arStartTau),
            Double(p.arFloorTau),
            Double(p.arDecayPerPly)
        )
        let divStr = String(format: "unique=%d/%d (%.0f%%) avgDiverge=%.1f",
                            d.uniqueGames, d.gamesInWindow,
                            d.uniquePercent, d.avgDivergencePly)

        let elo = record.eloSummary
        let scoreCI = ArenaEloStats.formatScorePercentWithCI(elo)
        let eloCI = ArenaEloStats.formatEloWithCI(elo)
        let drawRatePct = drawRateFraction(record: record) * 100.0

        let gamesStr = "\(record.gamesPlayed)/\(p.tournamentGames)"
        let resultStr = "\(record.candidateWins)W / \(record.draws)D / \(record.championWins)L"

        let whiteN = record.candidateWinsAsWhite + record.candidateLossesAsWhite + record.candidateDrawsAsWhite
        let blackN = record.candidateWinsAsBlack + record.candidateLossesAsBlack + record.candidateDrawsAsBlack
        let whiteScoreStr: String = whiteN > 0
            ? String(format: "%.1f%%", record.candidateScoreAsWhite * 100)
            : "—"
        let blackScoreStr: String = blackN > 0
            ? String(format: "%.1f%%", record.candidateScoreAsBlack * 100)
            : "—"
        let whiteWDL = "\(record.candidateWinsAsWhite)W/\(record.candidateDrawsAsWhite)D/\(record.candidateLossesAsWhite)L"
        let blackWDL = "\(record.candidateWinsAsBlack)W/\(record.candidateDrawsAsBlack)D/\(record.candidateLossesAsBlack)L"

        var lines: [String] = [
            "[ARENA] #\(index) Candidate vs Champion @ step \(record.finishedAtStep)",
            "[ARENA]     Games: \(gamesStr)",
            "[ARENA]     Result: \(resultStr)",
            "[ARENA]     Score: \(scoreCI)",
            "[ARENA]     Elo diff: \(eloCI)",
            "[ARENA]     Draw rate: \(String(format: "%.1f%%", drawRatePct))",
            "[ARENA]     By side:",
            "[ARENA]       Candidate as white: \(whiteScoreStr)  (\(whiteWDL), n=\(whiteN))",
            "[ARENA]       Candidate as black: \(blackScoreStr)  (\(blackWDL), n=\(blackN))"
        ]

        if let ext = extended {
            lines.append(contentsOf: formatExtendedBlock(extended: ext))
        }

        lines.append(contentsOf: [
            "[ARENA]     batch=\(p.batchSize) lr=\(lrStr) promote>=\(threshStr) games=\(p.tournamentGames) sp.tau=\(spTauStr) ar.tau=\(arTauStr) workers=\(p.workerCount) build=\(p.buildNumber)",
            "[ARENA]     candidate=\(candidateID)  champion=\(championID)  trainer=\(trainerID)",
            "[ARENA]     diversity: \(divStr)",
            "[ARENA]     Verdict: \(verdictStr)    dur=\(durationStr)"
        ])

        return lines
    }

    /// Render the three post-arena breakdowns as `[ARENA]` lines.
    /// Inserted between the "By side" block and the parameters line
    /// so the headline scoring is still at the top, with detail
    /// breakdowns flowing below it.
    ///
    /// The per-ply and per-progress sections render two metrics per
    /// bucket: `v=` is the mean candidate value-head scalar (the
    /// network's own assessment), and `score=` is the arena-style
    /// `(W + 0.5·D) / N` over candidate plies in the bucket — the
    /// same identity used to score the tournament itself, just
    /// attributed per ply. Pairing them is the calibration signal
    /// — when `v` stays high but `score` is near 0.5 across many
    /// arenas, the network is consistently over-rating its own
    /// position.
    static func formatExtendedBlock(extended ext: ArenaExtendedSummary) -> [String] {
        var lines: [String] = []

        // (1) W/D/L by game length. Always render every bucket so the
        // table is visually comparable arena-to-arena — a row with
        // n=0 still shows up as "0W/0D/0L (n=0)".
        if !ext.wdlByLength.isEmpty {
            lines.append("[ARENA]     WDL by game length (candidate perspective):")
            let labelWidth = ext.wdlByLength.map { lengthBucketLabel($0).count }.max() ?? 0
            for bucket in ext.wdlByLength {
                let label = lengthBucketLabel(bucket)
                    .padding(toLength: labelWidth, withPad: " ", startingAt: 0)
                let scoreStr = bucket.count > 0
                    ? String(format: " score=%.3f", bucket.candidateScore)
                    : ""
                lines.append(
                    "[ARENA]       \(label)  \(bucket.wins)W/\(bucket.draws)D/\(bucket.losses)L  (n=\(bucket.count)\(scoreStr))"
                )
            }
        }

        // (2) Candidate value scalar + arena-style score by absolute
        // ply. Drops empty trailing buckets in the aggregator; render
        // the rest.
        if !ext.valueByPly.isEmpty {
            lines.append("[ARENA]     Value + score by ply (20-ply buckets, candidate-to-move only):")
            let labelWidth = ext.valueByPly.map { plyBucketLabel($0).count }.max() ?? 0
            for bucket in ext.valueByPly {
                let label = plyBucketLabel(bucket)
                    .padding(toLength: labelWidth, withPad: " ", startingAt: 0)
                let pol = bucket.meanPolicyProbability
                    .map { String(format: "pol=%.3f ", $0) } ?? ""
                lines.append(
                    "[ARENA]       \(label)  v=\(formatSignedValue(bucket.mean)) \(pol)score=\(formatScore(bucket.candidateScore))  (n=\(bucket.count))"
                )
            }
        }

        // (3) Candidate value scalar + score by progress through game.
        if !ext.valueByProgress.isEmpty {
            lines.append("[ARENA]     Value + score by game progress (5% buckets):")
            let labelWidth = ext.valueByProgress.map { progressBucketLabel($0).count }.max() ?? 0
            for bucket in ext.valueByProgress {
                let label = progressBucketLabel(bucket)
                    .padding(toLength: labelWidth, withPad: " ", startingAt: 0)
                let pol = bucket.meanPolicyProbability
                    .map { String(format: "pol=%.3f ", $0) } ?? ""
                lines.append(
                    "[ARENA]       \(label)  v=\(formatSignedValue(bucket.mean)) \(pol)score=\(formatScore(bucket.candidateScore))  (n=\(bucket.count))"
                )
            }
        }

        return lines
    }

    // MARK: - KV single-line dump

    /// Build the single-line key=value arena summary that external
    /// tooling greps. Elo endpoints render as literal `"nan"` when
    /// the score CI is degenerate (p ∈ {0, 1}), which lets a
    /// downstream parser distinguish "undefined" from "unavailable"
    /// — displayed Elo of "—" in the human-readable block maps to
    /// `"nan"` here.
    static func formatKVLine(
        record: TournamentRecord,
        index: Int,
        candidateID: String,
        championID: String,
        trainerID: String,
        buildNumber: Int
    ) -> String {
        let elo = record.eloSummary
        let eloStr = elo.elo.map { String(format: "%+d", Int($0.rounded())) } ?? "nan"
        let eloLoStr = elo.eloLo.map { String(format: "%+d", Int($0.rounded())) } ?? "nan"
        let eloHiStr = elo.eloHi.map { String(format: "%+d", Int($0.rounded())) } ?? "nan"
        let kindKV = record.promotionKind?.rawValue ?? "none"
        let drawRateFrac = drawRateFraction(record: record)

        return "[ARENA] #\(index) kv step=\(record.finishedAtStep) games=\(record.gamesPlayed) w=\(record.candidateWins) d=\(record.draws) l=\(record.championWins) "
            + "score=\(String(format: "%.4f", record.score)) elo=\(eloStr) elo_lo=\(eloLoStr) elo_hi=\(eloHiStr) "
            + "draw_rate=\(String(format: "%.4f", drawRateFrac)) "
            + "cand_white_w=\(record.candidateWinsAsWhite) cand_white_d=\(record.candidateDrawsAsWhite) cand_white_l=\(record.candidateLossesAsWhite) "
            + "cand_black_w=\(record.candidateWinsAsBlack) cand_black_d=\(record.candidateDrawsAsBlack) cand_black_l=\(record.candidateLossesAsBlack) "
            + "cand_white_score=\(String(format: "%.4f", record.candidateScoreAsWhite)) cand_black_score=\(String(format: "%.4f", record.candidateScoreAsBlack)) "
            + "promoted=\(record.promoted ? 1 : 0) kind=\(kindKV) dur_sec=\(String(format: "%.1f", record.durationSec)) build=\(buildNumber) "
            + "candidate=\(candidateID) champion=\(championID) trainer=\(trainerID)"
    }

    // MARK: - Shared helpers

    /// Format a duration in seconds as `M:SS` (minutes:seconds),
    /// matching the legacy log format.
    static func formatDuration(_ sec: Double) -> String {
        let m = Int(sec) / 60
        let s = Int(sec) % 60
        return String(format: "%d:%02d", m, s)
    }

    /// Render the verdict marker (PROMOTED / kept / with kind
    /// suffix and promoted-ID tail). Extracted so tests can
    /// exercise every branch of the three-way switch without
    /// rebuilding the full log block.
    static func formatVerdict(record: TournamentRecord) -> String {
        let kindSuffix: String
        switch record.promotionKind {
        case .automatic: kindSuffix = "(auto)"
        case .manual:    kindSuffix = "(manual)"
        case .none:      kindSuffix = ""
        }
        if record.promoted, let pid = record.promotedID {
            return "PROMOTED\(kindSuffix)=\(pid.description)"
        } else if record.promoted {
            return "PROMOTED\(kindSuffix)"
        } else {
            return "kept"
        }
    }

    /// Draw rate as a fraction in `[0, 1]`. Guarded against empty
    /// tournaments — returns 0 rather than dividing by zero.
    static func drawRateFraction(record: TournamentRecord) -> Double {
        guard record.gamesPlayed > 0 else { return 0 }
        return Double(record.draws) / Double(record.gamesPlayed)
    }

    // MARK: - Extended-block helpers

    /// Render a length-bucket label, e.g. "0-19 plies", "180-199 plies",
    /// or "200+ plies" for the overflow bucket.
    static func lengthBucketLabel(_ bucket: ArenaWDLByLengthBucket) -> String {
        if let hi = bucket.upperInclusive {
            return "\(bucket.lowerInclusive)-\(hi) plies:"
        }
        return "\(bucket.lowerInclusive)+ plies:"
    }

    /// Render an absolute-ply bucket label, e.g. "ply  0- 19:" or
    /// "ply 100-119:". Width-padded so the digits column-align in
    /// the log block.
    static func plyBucketLabel(_ bucket: ArenaValueByPlyBucket) -> String {
        let lo = String(format: "%3d", bucket.lowerInclusive)
        let hi = String(format: "%3d", bucket.upperInclusive)
        return "ply \(lo)-\(hi):"
    }

    /// Render a progress-percent bucket label, e.g. "  0- 5%:" or
    /// " 95-100%:". The numeric edges from the aggregator are
    /// `[lower, upper)` for all but the last bucket, which is
    /// `[95, 100]` — both render the same in the log.
    static func progressBucketLabel(_ bucket: ArenaValueByProgressBucket) -> String {
        let lo = String(format: "%3d", bucket.lowerPercent)
        let hi = String(format: "%3d", bucket.upperPercent)
        return "\(lo)-\(hi)%:"
    }

    /// Format a value in `[-1, +1]` as a signed three-decimal-place
    /// number, e.g. `+0.123` or `-0.045`. `%+0.3f` would suffice; we
    /// route through `String(format:)` to match the rest of this
    /// formatter's idiom.
    static func formatSignedValue(_ v: Float) -> String {
        String(format: "%+0.3f", Double(v))
    }

    /// Format an arena-style score in `[0, 1]` as a fixed-width
    /// `0.NNN`. Unsigned to distinguish it visually from the signed
    /// value-scalar column when both appear side by side.
    static func formatScore(_ s: Double) -> String {
        String(format: "%0.3f", s)
    }
}
