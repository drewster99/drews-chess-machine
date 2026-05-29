import Foundation

/// Large probe set sourced from the Lichess puzzle database (CC0). Two
/// hundred positions across eight theme buckets — `mateIn1`,
/// `hangingPiece`, `fork`, `pin`, `skewer`, `opening`, `middlegame`,
/// `endgame`, 25 puzzles each — filtered to puzzle ratings 800-1800 so a
/// healthy network produces a moving score (target floor ~30%, ceiling
/// ~80%) rather than saturating. Parallel to `TacticalProbeData` and
/// produces the same `TacticalProbe` shape so probe-running infrastructure
/// can run either set.
///
/// The bundled `lichess_probes_200.json` resource is the source of
/// truth. Regenerate with `scripts/curate_lichess_probes.py` against a
/// fresh download of `lichess_db_puzzle.csv.zst` if the set needs to
/// rotate. Selection is deterministic (sort eligible by puzzle id, take
/// 25 evenly-spaced) so re-running on the same DB snapshot yields the
/// same 200 puzzles.
///
/// Each JSON entry stores the FEN of the position the solver sees (i.e.
/// the Lichess "setup move" has already been applied) and the UCI of
/// the single best move. Single-best-move grading: a probe scores
/// `correct` iff the network's argmax over legal moves equals the
/// stored UCI move.
///
/// Loading is lazy and one-shot — the JSON parse happens at the first
/// touch of `largeSet`. Failures are `preconditionFailure` because the
/// JSON ships in the app bundle; a malformed payload would mean a
/// broken build, not a runtime condition to recover from.
enum LichessProbeData {

    /// Bundled 200-puzzle set, parsed at first access.
    static let largeSet: [TacticalProbe] = loadFromBundle()

    // MARK: - JSON shape

    private struct BundleJSON: Decodable {
        let puzzles: [PuzzleJSON]
    }

    private struct PuzzleJSON: Decodable {
        let id: String
        let theme: String
        let rating: Int
        let fen: String
        let bestMoveUci: String

        enum CodingKeys: String, CodingKey {
            case id
            case theme
            case rating
            case fen
            case bestMoveUci = "best_move_uci"
        }
    }

    // MARK: - Bundle load

    private static func loadFromBundle() -> [TacticalProbe] {
        guard let url = Bundle.main.url(
            forResource: "lichess_probes_200",
            withExtension: "json"
        ) else {
            preconditionFailure(
                "LichessProbeData: lichess_probes_200.json not in bundle"
            )
        }

        let data: Data
        do {
            data = try Data(contentsOf: url)
        } catch {
            preconditionFailure(
                "LichessProbeData: read failed for \(url.path): \(error)"
            )
        }

        let decoded: BundleJSON
        do {
            decoded = try JSONDecoder().decode(BundleJSON.self, from: data)
        } catch {
            preconditionFailure(
                "LichessProbeData: JSON decode failed: \(error)"
            )
        }

        var probes: [TacticalProbe] = []
        probes.reserveCapacity(decoded.puzzles.count)

        for entry in decoded.puzzles {
            let state: GameState
            do {
                state = try FENParser.parse(entry.fen)
            } catch {
                preconditionFailure(
                    "LichessProbeData: bad FEN for \(entry.id): \(error)"
                )
            }

            guard let move = parseUCI(entry.bestMoveUci) else {
                preconditionFailure(
                    "LichessProbeData: bad UCI '\(entry.bestMoveUci)' for \(entry.id)"
                )
            }

            guard let category = themeToCategory(entry.theme) else {
                preconditionFailure(
                    "LichessProbeData: unknown theme '\(entry.theme)' for \(entry.id)"
                )
            }

            // Name includes the puzzle id and rating so a `[TACTICAL]`
            // log line points straight back to the source puzzle (the
            // Lichess URL is `https://lichess.org/training/<id>`).
            let name = "[lichess \(entry.id), r\(entry.rating)] \(entry.theme): \(move.notation)"

            probes.append(TacticalProbe(
                name: name,
                shortDescription: entry.theme,
                category: category,
                state: state,
                acceptable: [move]
            ))
        }

        return probes
    }

    // MARK: - UCI move parsing

    /// Parse a UCI long-algebraic move (e.g. `"e2e4"`, `"a7a8q"`) into a
    /// `ChessMove`. Returns nil for any structural problem — caller is
    /// expected to treat a nil as a fixture bug since bundled puzzle
    /// moves are pre-validated by python-chess in the curation script.
    static func parseUCI(_ uci: String) -> ChessMove? {
        let chars = Array(uci)
        guard chars.count == 4 || chars.count == 5 else { return nil }
        guard let from = squareIndex(file: chars[0], rank: chars[1]) else {
            return nil
        }
        guard let to = squareIndex(file: chars[2], rank: chars[3]) else {
            return nil
        }
        var promotion: PieceType?
        if chars.count == 5 {
            switch chars[4] {
            case "q": promotion = .queen
            case "r": promotion = .rook
            case "b": promotion = .bishop
            case "n": promotion = .knight
            default: return nil
            }
        }
        return ChessMove(
            fromRow: from / 8,
            fromCol: from % 8,
            toRow: to / 8,
            toCol: to % 8,
            promotion: promotion
        )
    }

    private static func squareIndex(file: Character, rank: Character) -> Int? {
        guard let fileAscii = file.asciiValue,
              fileAscii >= 0x61, fileAscii <= 0x68,
              let rankAscii = rank.asciiValue,
              rankAscii >= 0x31, rankAscii <= 0x38
        else { return nil }
        let col = Int(fileAscii - 0x61)
        let rankNum = Int(rankAscii - 0x30)  // 1...8
        let row = 8 - rankNum
        return row * 8 + col
    }

    // MARK: - Theme mapping

    private static func themeToCategory(_ theme: String) -> ProbeCategory? {
        switch theme {
        case "mateIn1":      return .lichessMateIn1
        case "hangingPiece": return .lichessHangingPiece
        case "fork":         return .lichessFork
        case "pin":          return .lichessPin
        case "skewer":       return .lichessSkewer
        case "opening":      return .lichessOpening
        case "middlegame":   return .lichessMiddlegame
        case "endgame":      return .lichessEndgame
        default:             return nil
        }
    }
}
