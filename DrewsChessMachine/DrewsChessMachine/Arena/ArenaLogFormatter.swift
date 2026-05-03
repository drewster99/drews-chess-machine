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
    static func formatHumanReadable(
        record: TournamentRecord,
        index: Int,
        candidateID: String,
        championID: String,
        trainerID: String,
        parameters p: Parameters,
        diversity d: Diversity
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

        return [
            "[ARENA] #\(index) Candidate vs Champion @ step \(record.finishedAtStep)",
            "[ARENA]     Games: \(gamesStr)",
            "[ARENA]     Result: \(resultStr)",
            "[ARENA]     Score: \(scoreCI)",
            "[ARENA]     Elo diff: \(eloCI)",
            "[ARENA]     Draw rate: \(String(format: "%.1f%%", drawRatePct))",
            "[ARENA]     By side:",
            "[ARENA]       Candidate as white: \(whiteScoreStr)  (\(whiteWDL), n=\(whiteN))",
            "[ARENA]       Candidate as black: \(blackScoreStr)  (\(blackWDL), n=\(blackN))",
            "[ARENA]     batch=\(p.batchSize) lr=\(lrStr) promote>=\(threshStr) games=\(p.tournamentGames) sp.tau=\(spTauStr) ar.tau=\(arTauStr) workers=\(p.workerCount) build=\(p.buildNumber)",
            "[ARENA]     candidate=\(candidateID)  champion=\(championID)  trainer=\(trainerID)",
            "[ARENA]     diversity: \(divStr)",
            "[ARENA]     Verdict: \(verdictStr)    dur=\(durationStr)"
        ]
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
}
