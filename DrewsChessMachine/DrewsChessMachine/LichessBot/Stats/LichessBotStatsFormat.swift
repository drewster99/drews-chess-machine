import Foundation

/// Number formatting shared by every Record card pane, so one quantity is
/// written one way everywhere (`LICHESS_BOT_RECORD_STATS_PLAN.md` §4.1).
/// Pure text: the views set the monospaced font that makes padded columns
/// line up.
enum LichessBotStatsFormat {
    /// What a cell shows when there is nothing to compute from.
    static let missing = "–"
    /// The true minus sign (U+2212), as wide as "+" in a monospaced font,
    /// so signed columns align.
    static let minus = "\u{2212}"

    /// "+12", "−7", "0".
    static func signed(_ value: Int) -> String {
        if value > 0 { return "+\(value)" }
        if value < 0 { return "\(minus)\(-value)" }
        return "0"
    }

    /// A signed value rounded to an integer.
    static func signed(_ value: Double) -> String {
        signed(Int(value.rounded()))
    }

    /// Rating change (§3.3): "–" with no rated game; "–*" with rated games
    /// of which none has a recorded change (never "0", which would claim
    /// no change); otherwise the signed sum, with "*" when some rated game
    /// lacks a change.
    static func ratingChange(_ change: LichessBotRatingChange) -> String {
        if change.ratedGames == 0 {
            return missing
        }
        if change.gamesWithChange == 0 {
            return missing + "*"
        }
        return signed(change.total) + (change.gamesWithoutChange > 0 ? "*" : "")
    }

    /// The tooltip that goes with `ratingChange`.
    static func ratingChangeHelp(_ change: LichessBotRatingChange) -> String {
        if change.ratedGames == 0 {
            return "No rated game"
        }
        return "\(change.gamesWithChange) of \(change.ratedGames) rated games have a rating change (Lichess's change at the time of the game)"
    }

    /// "1734", "≥1890", "≤1100", "–".
    static func estimate(_ estimate: LichessBotRatingEstimate) -> String {
        switch estimate {
        case .none: return missing
        case .estimate(let value): return "\(Int(value.rounded()))"
        case .atLeast(let value): return "≥\(Int(value.rounded()))"
        case .atMost(let value): return "≤\(Int(value.rounded()))"
        }
    }

    /// A signed offset estimate (the 50% point): "+37", "≥+120", "≤−80".
    static func signedEstimate(_ estimate: LichessBotRatingEstimate) -> String {
        switch estimate {
        case .none: return missing
        case .estimate(let value): return signed(value)
        case .atLeast(let value): return "≥\(signed(value))"
        case .atMost(let value): return "≤\(signed(value))"
        }
    }

    /// "62.5%", or "–" for no value.
    static func percent(_ fraction: Double?) -> String {
        guard let fraction else { return missing }
        return String(format: "%.1f%%", 100 * fraction)
    }

    /// A Wilson interval as whole percentages: "41–80%".
    static func interval(_ interval: LichessBotScoreInterval?) -> String {
        guard let interval else { return missing }
        return String(format: "%.0f–%.0f%%", 100 * interval.lower, 100 * interval.upper)
    }

    /// An average rounded to an integer, or "–".
    static func average(_ value: Double?) -> String {
        guard let value else { return missing }
        return "\(Int(value.rounded()))"
    }

    /// A probability or mean with three decimals, or "–".
    static func decimal(_ value: Double?, places: Int = 3) -> String {
        guard let value else { return missing }
        let text = String(format: "%.\(places)f", value)
        return value < 0 ? minus + text.dropFirst() : text
    }

    /// `count` left-padded with spaces to `width` digits (spaces are as wide
    /// as digits in the monospaced font every table sets).
    static func padded(_ count: Int, width: Int) -> String {
        let text = String(count)
        return String(repeating: " ", count: max(0, width - text.count)) + text
    }

    /// `text` left-padded with spaces to `width` characters.
    static func padded(_ text: String, width: Int) -> String {
        String(repeating: " ", count: max(0, width - text.count)) + text
    }

    /// "W–D–L" with each count padded to `width` digits.
    static func tally(_ tally: LichessBotResultTally, width: Int) -> String {
        "\(padded(tally.wins, width: width))–\(padded(tally.draws, width: width))–\(padded(tally.losses, width: width))"
    }

    /// Digits of the largest count among `tallies`, for `tally(_:width:)`.
    static func countWidth(_ tallies: [LichessBotResultTally]) -> Int {
        String(tallies.flatMap { [$0.wins, $0.draws, $0.losses] }.max() ?? 0).count
    }

    /// The account's current rating in one speed (§3.5): Lichess's number,
    /// "?" appended while it is provisional; "–" with no rating there or
    /// before the account has loaded.
    static func accountRating(_ perf: LichessBotPerfRating?) -> String {
        guard let rating = perf?.rating else { return missing }
        return "\(rating)" + (perf?.prov == true ? "?" : "")
    }

    /// "x blown of y held wins (z%)".
    static func held(turned: Int, held: Int, verb: String, noun: String) -> String {
        guard held > 0 else { return "No \(noun)" }
        return "\(turned) \(verb) of \(held) \(noun) (\(percent(Double(turned) / Double(held))))"
    }
}
