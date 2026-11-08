import CryptoKit
import Foundation

/// One bundled puzzle test set: who it is, and its probes. Its identity
/// comes from the set file's own `metadata` block (written by
/// `scripts/curate_lichess_probes.py`), never from code, so a renamed or
/// regenerated set is described by its file.
struct ProbeTestSet: Sendable {
    /// Stable name, e.g. `lichess-wide`. A regenerated set may keep its id;
    /// `fingerprintSHA256` tells the versions apart.
    let id: String
    let title: String
    let description: String
    /// SHA-256 (lowercase hex) of the puzzle list in file order, one line per
    /// puzzle: `id \t theme \t rating \t fen \t best_move_uci \n`. Results
    /// from two files are comparable only when their fingerprints match.
    let fingerprintSHA256: String
    let probes: [TacticalProbe]
}

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
/// touch of `largeSet`. Bundled-set failures are `preconditionFailure`
/// because the JSON ships in the app bundle; a malformed payload would
/// mean a broken build. A set file named on the command line
/// (`loadProbeSetFile`) goes through the same parser and throws instead.
enum LichessProbeData {

    /// Bundled 200-puzzle set, parsed at first access.
    static var largeSet: [TacticalProbe] { loaded.testSet.probes }

    /// The 200-puzzle set with its identity.
    static var set200: ProbeTestSet { loaded.testSet }

    /// The wide set with its identity.
    static var wide: ProbeTestSet { loadedWide.testSet }

    /// The sets every model file DCM writes is evaluated on, in the order
    /// their results are recorded (`ModelTestSetEvaluator`).
    static var modelFileTestSets: [ProbeTestSet] { [set200, wide] }

    /// The ~4,435-puzzle WIDE longitudinal probe set (rating 400–3200,
    /// flat per-100 density 550–2849, mate-weighted), from the bundled
    /// `lichess_probes_wide.json`. Runs in parallel with `largeSet` as a
    /// fixed long-term yardstick — the 200-set is left completely
    /// untouched. See `LichessProbeWatcher`.
    static var wideSet: [TacticalProbe] { loadedWide.testSet.probes }

    /// Sidecar metadata keyed by `TacticalProbe.name` so the detail
    /// window and the JSON exporter can surface the original Lichess
    /// puzzle id, rating, theme, FEN, and bookmove UCI without
    /// re-parsing the bundle JSON or trying to recover those fields
    /// from `probe.name` by regex.
    /// Per-puzzle metadata for BOTH sets, merged. Keyed by
    /// `TacticalProbe.name` (which embeds the unique Lichess puzzle id),
    /// so 200-set and wide-set entries don't collide; on a shared puzzle
    /// the 200-set's entry wins (identical content anyway).
    static var metadata: [String: PuzzleMetadata] { mergedMetadata }

    private static let mergedMetadata: [String: PuzzleMetadata] =
        loaded.metadata.merging(loadedWide.metadata) { existing, _ in existing }

    /// Per-puzzle metadata. Stored verbatim from the bundle JSON so
    /// exports round-trip cleanly back to the source.
    struct PuzzleMetadata: Sendable {
        let id: String
        let theme: String              // raw Lichess theme string
        let rating: Int
        let fen: String                // post-setup-move FEN
        let bestMoveUci: String
    }

    /// One parsed set file: the set with its identity, and per-puzzle
    /// metadata keyed by `TacticalProbe.name`. The bundled sets and a
    /// `--probe-set-file` set are both this, from `parseProbeSet`.
    struct ParsedProbeSet: Sendable {
        let testSet: ProbeTestSet
        let metadata: [String: PuzzleMetadata]
    }

    private static let loaded: ParsedProbeSet = loadFromBundle(resource: "lichess_probes_200")
    private static let loadedWide: ParsedProbeSet = loadFromBundle(resource: "lichess_probes_wide")

    // MARK: - JSON shape

    private struct SetFileJSON: Decodable {
        let metadata: SetMetadataJSON
        let puzzles: [PuzzleJSON]
    }

    /// The identity fields of the set file's `metadata` block; its other
    /// (provenance) fields are not read.
    private struct SetMetadataJSON: Decodable {
        let id: String
        let title: String
        let description: String
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

    // MARK: - Parsing

    /// Why a set file could not be read or parsed.
    enum ProbeSetFileError: LocalizedError, Equatable {
        case unreadable(path: String, reason: String)
        case notASetFile(reason: String)
        case noPuzzles
        case duplicatePuzzleID(String)
        case badFEN(puzzleID: String, fen: String, reason: String)
        case badMove(puzzleID: String, uci: String)
        case unknownTheme(puzzleID: String, theme: String)

        var errorDescription: String? {
            switch self {
            case let .unreadable(path, reason):
                return "cannot read the puzzle set file \(path): \(reason)"
            case let .notASetFile(reason):
                return "not a puzzle set file (top-level metadata {id, title, description} + puzzles [{id, theme, rating, fen, best_move_uci}]): \(reason)"
            case .noPuzzles:
                return "the puzzle set file holds no puzzles"
            case let .duplicatePuzzleID(id):
                return "puzzle \(id) appears more than once"
            case let .badFEN(puzzleID, fen, reason):
                return "puzzle \(puzzleID): bad FEN '\(fen)': \(reason)"
            case let .badMove(puzzleID, uci):
                return "puzzle \(puzzleID): bad UCI move '\(uci)'"
            case let .unknownTheme(puzzleID, theme):
                return "puzzle \(puzzleID): theme '\(theme)' is not one of the Lichess probe themes"
            }
        }
    }

    /// Read and parse a set file named on the command line
    /// (`--probe-set-file`). Every failure is a thrown `ProbeSetFileError`:
    /// a user's file is input, not part of the build.
    static func loadProbeSetFile(at url: URL) throws -> ParsedProbeSet {
        let data: Data
        do {
            data = try Data(contentsOf: url)
        } catch {
            throw ProbeSetFileError.unreadable(path: url.path, reason: error.localizedDescription)
        }
        return try parseProbeSet(jsonData: data)
    }

    /// The one parser for a set file's JSON (bundled or not): decode, turn
    /// each puzzle into a `TacticalProbe` plus its metadata, and compute the
    /// set's fingerprint over the puzzles in file order. Puzzle ids must be
    /// unique: a probe's name embeds its id, and the metadata is keyed by it.
    static func parseProbeSet(jsonData: Data) throws -> ParsedProbeSet {
        let decoded: SetFileJSON
        do {
            decoded = try JSONDecoder().decode(SetFileJSON.self, from: jsonData)
        } catch let error as DecodingError {
            throw ProbeSetFileError.notASetFile(reason: describe(error))
        } catch {
            throw ProbeSetFileError.notASetFile(reason: error.localizedDescription)
        }
        guard !decoded.puzzles.isEmpty else { throw ProbeSetFileError.noPuzzles }

        var probes: [TacticalProbe] = []
        var metadata: [String: PuzzleMetadata] = [:]
        var puzzleIDs: Set<String> = []
        var fingerprintText = ""
        probes.reserveCapacity(decoded.puzzles.count)
        metadata.reserveCapacity(decoded.puzzles.count)
        puzzleIDs.reserveCapacity(decoded.puzzles.count)

        for entry in decoded.puzzles {
            guard puzzleIDs.insert(entry.id).inserted else {
                throw ProbeSetFileError.duplicatePuzzleID(entry.id)
            }
            let state: GameState
            do {
                state = try FENParser.parse(entry.fen)
            } catch {
                throw ProbeSetFileError.badFEN(puzzleID: entry.id, fen: entry.fen, reason: "\(error)")
            }
            guard let move = parseUCI(entry.bestMoveUci) else {
                throw ProbeSetFileError.badMove(puzzleID: entry.id, uci: entry.bestMoveUci)
            }
            guard let category = ProbeCategory(lichessThemeID: entry.theme) else {
                throw ProbeSetFileError.unknownTheme(puzzleID: entry.id, theme: entry.theme)
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
            metadata[name] = PuzzleMetadata(
                id: entry.id,
                theme: entry.theme,
                rating: entry.rating,
                fen: entry.fen,
                bestMoveUci: entry.bestMoveUci
            )
            fingerprintText += Self.fingerprintLine(id: entry.id, theme: entry.theme, rating: entry.rating, fen: entry.fen, bestMoveUci: entry.bestMoveUci)
        }

        let testSet = ProbeTestSet(
            id: decoded.metadata.id,
            title: decoded.metadata.title,
            description: decoded.metadata.description,
            fingerprintSHA256: Self.sha256Hex(fingerprintText),
            probes: probes
        )
        return ParsedProbeSet(testSet: testSet, metadata: metadata)
    }

    /// A decoding failure as one line: what went wrong and where.
    private static func describe(_ error: DecodingError) -> String {
        func path(_ codingPath: [CodingKey]) -> String {
            codingPath.isEmpty ? "top level" : codingPath.map { $0.intValue.map { "[\($0)]" } ?? $0.stringValue }.joined(separator: ".")
        }
        switch error {
        case let .typeMismatch(_, context), let .valueNotFound(_, context), let .dataCorrupted(context):
            return "\(context.debugDescription) (at \(path(context.codingPath)))"
        case let .keyNotFound(key, context):
            return "missing key '\(key.stringValue)' (at \(path(context.codingPath)))"
        @unknown default:
            return "\(error)"
        }
    }

    // MARK: - Bundle load

    /// The bundled sets ship in the app, so a missing or malformed one is a
    /// broken build, not a runtime condition: it stops here.
    private static func loadFromBundle(resource: String) -> ParsedProbeSet {
        guard let url = Bundle.main.url(
            forResource: resource,
            withExtension: "json"
        ) else {
            preconditionFailure(
                "LichessProbeData: \(resource).json not in bundle"
            )
        }
        do {
            return try loadProbeSetFile(at: url)
        } catch {
            preconditionFailure(
                "LichessProbeData: bundled \(resource).json: \(error.localizedDescription)"
            )
        }
    }

    // MARK: - Fingerprint

    /// One puzzle's line in a set's fingerprint (`ProbeTestSet.fingerprintSHA256`).
    static func fingerprintLine(id: String, theme: String, rating: Int, fen: String, bestMoveUci: String) -> String {
        "\(id)\t\(theme)\t\(rating)\t\(fen)\t\(bestMoveUci)\n"
    }

    static func sha256Hex(_ text: String) -> String {
        SHA256.hash(data: Data(text.utf8)).map { String(format: "%02x", $0) }.joined()
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
}
