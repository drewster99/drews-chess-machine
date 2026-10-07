import Foundation

/// The Record card panel's tabs (`LICHESS_BOT_RECORD_STATS_PLAN.md` §5.1,
/// §11): the first pass in the order OD-21 shipped them, then the later
/// panes. Raw values are stable identifiers: the selected tab is remembered
/// in the defaults.
enum LichessBotRecordPane: String, CaseIterable, Sendable, Identifiable {
    case timeControls
    case models
    case selfAssessment
    case endings
    case opponentStrength
    case moveChoice
    case clock
    case gameLength
    case openings
    case opponents
    case botHealth

    var id: String { rawValue }

    /// The first pass (§9 P3–P8), the picker's first section.
    static let firstPass: [LichessBotRecordPane] = [.timeControls, .models, .selfAssessment, .endings, .opponentStrength]
    /// The later panes (§11 L1–L6), its second section.
    static let later: [LichessBotRecordPane] = [.moveChoice, .clock, .gameLength, .openings, .opponents, .botHealth]

    var label: String {
        switch self {
        case .timeControls: return "Time controls"
        case .models: return "Models"
        case .selfAssessment: return "Self-assessment"
        case .endings: return "Endings"
        case .opponentStrength: return "Opponent strength"
        case .moveChoice: return "Move choice"
        case .clock: return "Clock"
        case .gameLength: return "Game length"
        case .openings: return "Openings"
        case .opponents: return "Opponents"
        case .botHealth: return "Bot health"
        }
    }
}

/// The `[LICHESS-BOT] record stats` session-log line (§4.3): the all-time,
/// all-games numbers, so a wrong number is visible in the log and
/// `scripts/lichess_bot_record_stats.py` can be compared with it. ASCII
/// signs only (the log is grepped and diffed), unlike the card's true minus.
enum LichessBotRecordStatsLogLine {
    static func text(_ statistics: LichessBotRecordStatistics, reason: String, milliseconds: Double) -> String {
        let row = statistics[.all].periodRows.allTime
        let tally = row.record.all
        let brier = statistics[.all].byPeriod.allTime.selfAssessment.calibration.first { $0.moveNumber == 20 }?.brier
        return "[LICHESS-BOT] record stats (\(reason)): games=\(tally.scored)"
            + " W-D-L=\(tally.wins)-\(tally.draws)-\(tally.losses)"
            + " score=\(tally.score.map { String(format: "%.1f%%", 100 * $0) } ?? "-")"
            + " perf=\(perf(row.performance))"
            + " rating=\(rating(row.ratingChange))"
            + " brier@20=\(brier.map { String(format: "%.3f", $0) } ?? "-")"
            + " not-counted=\(statistics[.all].notCounted)"
            + " rated-without-diff=\(statistics[.all].ratedWithoutRatingChange)"
            + " ms=\(String(format: "%.1f", milliseconds))"
    }

    private static func perf(_ estimate: LichessBotRatingEstimate) -> String {
        switch estimate {
        case .none: return "-"
        case .estimate(let value): return "\(Int(value.rounded()))"
        case .atLeast(let value): return ">=\(Int(value.rounded()))"
        case .atMost(let value): return "<=\(Int(value.rounded()))"
        }
    }

    private static func rating(_ change: LichessBotRatingChange) -> String {
        if change.ratedGames == 0 { return "-" }
        if change.gamesWithChange == 0 { return "-*" }
        let sign = change.total > 0 ? "+" : ""
        return "\(sign)\(change.total)" + (change.gamesWithoutChange > 0 ? "*" : "")
    }
}
