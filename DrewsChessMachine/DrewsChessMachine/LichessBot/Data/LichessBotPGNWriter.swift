import Foundation

/// PGN for a finished game, with DCM annotations (plan §10.1).
///
/// Standard Seven Tag Roster plus Lichess's usual extras, and `DCM*` tags
/// naming the model that played. Clock comments use the `[%clk h:mm:ss]`
/// convention Lichess and most viewers read; our moves also carry the value
/// head's W/D/L and the sampling details. If the move list could not be
/// replayed past some ply, the movetext stops there with a comment saying
/// so — it never contains an unverified move.
enum LichessBotPGNWriter {

    static func pgn(for record: LichessBotGameRecord) -> String {
        var tags: [(String, String)] = [
            ("Event", "\(record.setup.rated ? "Rated" : "Casual") \(record.setup.perf ?? record.setup.speed) game"),
            ("Site", record.url),
            // UTC, as Lichess's own PGN has it (its Date equals UTCDate), and
            // plan E50.
            ("Date", dateString(record.createdAt, format: "yyyy.MM.dd", timeZone: .gmt)),
            ("Round", "-"),
            ("White", displayName(record.ourColor == .white ? record.us : record.opponent)),
            ("Black", displayName(record.ourColor == .black ? record.us : record.opponent)),
            ("Result", record.outcome.pgnResult),
            ("UTCDate", dateString(record.createdAt, format: "yyyy.MM.dd", timeZone: .gmt)),
            ("UTCTime", dateString(record.createdAt, format: "HH:mm:ss", timeZone: .gmt)),
        ]
        let white = record.ourColor == .white ? record.us : record.opponent
        let black = record.ourColor == .black ? record.us : record.opponent
        tags.append(("WhiteElo", white.ratingBefore.map(String.init) ?? "?"))
        tags.append(("BlackElo", black.ratingBefore.map(String.init) ?? "?"))
        if let diff = white.ratingDiff { tags.append(("WhiteRatingDiff", signed(diff))) }
        if let diff = black.ratingDiff { tags.append(("BlackRatingDiff", signed(diff))) }
        if let title = white.title { tags.append(("WhiteTitle", title)) }
        if let title = black.title { tags.append(("BlackTitle", title)) }
        tags.append(("Variant", record.setup.variant == "standard" ? "Standard" : record.setup.variant))
        if !LichessBotPositionTracker.isStandardStart(record.setup.initialFen) {
            tags.append(("SetUp", "1"))
            tags.append(("FEN", record.setup.initialFen))
        }
        tags.append(("TimeControl", timeControl(record.setup)))
        if let eco = record.openingECO { tags.append(("ECO", eco)) }
        if let opening = record.openingName { tags.append(("Opening", opening)) }
        tags.append(("Termination", termination(record.outcome.status)))
        tags.append(("LichessStatus", record.outcome.status))
        tags.append(("DCMBuilds", record.builds.map(String.init).joined(separator: ",")))
        tags.append(("DCMModelIDs", orderedUnique(record.generations.map(\.modelID)).joined(separator: ",")))
        tags.append(("DCMSources", orderedUnique(record.generations.map(\.sourceKind.rawValue)).joined(separator: ",")))
        // New games only: a record from before origins were recorded has
        // none, and its PGN is never rewritten (OD-10).
        if let origin = record.origin {
            tags.append(("DCMOrigin", origin.token))
        }

        var text = tags.map { "[\($0.0) \"\(escape($0.1))\"]" }.joined(separator: "\n")
        text += "\n\n"
        text += wrap(movetextTokens(record) + [record.outcome.pgnResult])
        text += "\n"
        return text
    }

    // MARK: - Movetext

    private static func movetextTokens(_ record: LichessBotGameRecord) -> [String] {
        var tokens: [String] = []
        for move in record.moves {
            guard let san = move.san else {
                tokens.append("{ The move list could not be replayed from ply \(move.ply). }")
                break
            }
            if move.color == .white {
                tokens.append("\(move.ply / 2 + 1).")
            } else if tokens.isEmpty || tokens.last?.hasSuffix("}") == true {
                tokens.append("\(move.ply / 2 + 1)...")
            }
            tokens.append(san)
            if let comment = comment(for: move) {
                tokens.append(comment)
            }
        }
        return tokens
    }

    private static func comment(for move: LichessBotGameRecord.Move) -> String? {
        var parts: [String] = []
        let clock = move.color == .white ? move.whiteClockMilliseconds : move.blackClockMilliseconds
        if let clock {
            parts.append("[%clk \(clockString(milliseconds: clock))]")
        }
        if let decision = move.decision {
            var text = String(
                format: "DCM W/D/L %.3f/%.3f/%.3f p=%.3f tau=%.2f legal=%d",
                decision.win, decision.draw, decision.loss,
                decision.chosenProbability, decision.temperature, decision.legalMoveCount
            )
            if let generationID = move.generationID {
                text += " gen=\(generationID)"
            }
            parts.append(text)
        }
        if move.offeredDraw == true {
            parts.append("draw offered")
        }
        return parts.isEmpty ? nil : "{ \(parts.joined(separator: " ")) }"
    }

    /// The conventional PGN movetext width.
    static let maximumMovetextLineLength = 80

    /// Join tokens into lines no longer than `maximumMovetextLineLength`
    /// where possible (a single long comment may exceed it).
    private static func wrap(_ tokens: [String]) -> String {
        var lines: [String] = []
        var line = ""
        for token in tokens {
            if line.isEmpty {
                line = token
            } else if line.count + 1 + token.count <= maximumMovetextLineLength {
                line += " " + token
            } else {
                lines.append(line)
                line = token
            }
        }
        if !line.isEmpty {
            lines.append(line)
        }
        return lines.joined(separator: "\n")
    }

    // MARK: - Tag helpers

    static func clockString(milliseconds: Int) -> String {
        let totalSeconds = max(0, milliseconds) / 1000
        return String(format: "%d:%02d:%02d", totalSeconds / 3600, (totalSeconds / 60) % 60, totalSeconds % 60)
    }

    private static func timeControl(_ setup: LichessBotGameRecord.Setup) -> String {
        guard let initial = setup.clockInitialMilliseconds, let increment = setup.clockIncrementMilliseconds else {
            return "-"
        }
        return "\(initial / 1000)+\(increment / 1000)"
    }

    /// PGN `Termination`, following Lichess's own export.
    private static func termination(_ status: String) -> String {
        switch LichessBotGameStatusName(rawValue: status) {
        case .mate, .resign, .stalemate, .draw, .insufficientMaterialClaim, .variantEnd:
            return "Normal"
        case .outOfTime:
            return "Time forfeit"
        case .timeout:
            return "Abandoned"
        case .cheat:
            return "Rules infraction"
        case .aborted, .noStart:
            return "Unterminated"
        case .created, .started, .unknownFinish, .none:
            return "Unknown"
        }
    }

    private static func displayName(_ player: LichessBotGameRecord.Player) -> String {
        if let name = player.name {
            return name
        }
        if let level = player.aiLevel {
            return "lichess AI level \(level)"
        }
        return player.id ?? "?"
    }

    private static func signed(_ value: Int) -> String {
        value >= 0 ? "+\(value)" : "\(value)"
    }

    private static func escape(_ value: String) -> String {
        value.replacingOccurrences(of: "\\", with: "\\\\").replacingOccurrences(of: "\"", with: "\\\"")
    }

    private static func dateString(_ date: Date, format: String, timeZone: TimeZone) -> String {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.calendar = Calendar(identifier: .gregorian)
        formatter.timeZone = timeZone
        formatter.dateFormat = format
        return formatter.string(from: date)
    }

    private static func orderedUnique(_ values: [String]) -> [String] {
        var seen: Set<String> = []
        return values.filter { seen.insert($0).inserted }
    }
}
