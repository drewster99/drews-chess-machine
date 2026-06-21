import Foundation

struct PGNImportConfig: Sendable {
    var inputPath: String
    var corpusName: String?
    var minRating: Int?
    var maxGames: Int?
    var minPlies: Int
    var timeControlClasses: [String]?   // e.g. ["blitz","rapid"]; nil = all
    var shardSoftLimitMB: Int = 64
}

enum PGNImportError: LocalizedError {
    case openFailed(String)
    var errorDescription: String? {
        switch self {
        case .openFailed(let s): return "PGN import open failed: \(s)"
        }
    }
}

/// Streaming PGN → game-corpus importer. Decompresses `.pgn.zst` via the
/// `zstd` CLI (or reads a plain `.pgn` directly), parses games one at a time,
/// converts each SAN move list to `ChessMove`s by matching against the
/// engine's legal moves, applies rating / time-control / length filters, and
/// appends the survivors to a fresh corpus. Standard-start games only (FEN
/// setups, variants, and any game with an unparseable move are skipped).
enum PGNImporter {

    // MARK: - CLI entry

    static func runImportAndExit(config: PGNImportConfig) -> Never {
        SessionLogger.shared.start()
        SessionLogger.shared.log("[PGN] importing \(config.inputPath)")
        do {
            let summary = try runImport(config: config)
            let line = "[PGN] done: imported=\(summary.imported) skipped=\(summary.skipped) parseErrors=\(summary.parseErrors) → corpus \(summary.corpusID)"
            print(line)
            SessionLogger.shared.log(line)
        } catch {
            FileHandle.standardError.write(Data("pgn-import: failed: \(error.localizedDescription)\n".utf8))
            SessionLogger.shared.log("[PGN] failed: \(error.localizedDescription)")
            SessionLogger.shared.shutdown()
            Darwin.exit(34)
        }
        SessionLogger.shared.shutdown()
        Darwin.exit(0)
    }

    struct Summary {
        var imported = 0
        var skipped = 0
        var parseErrors = 0
        var corpusID = ""
    }

    // MARK: - Import

    static func runImport(config: PGNImportConfig) throws -> Summary {
        let expanded = (config.inputPath as NSString).expandingTildeInPath
        let inputURL = URL(fileURLWithPath: expanded)

        let corpus = try GameCorpus.create(
            name: config.corpusName ?? inputURL.lastPathComponent,
            comment: "Imported from \(inputURL.lastPathComponent) by build \(BuildInfo.buildNumber) (\(BuildInfo.gitHash))",
            shardSoftLimitBytes: max(1, config.shardSoftLimitMB) * 1024 * 1024
        )
        try corpus.beginSource(
            kind: "pgnImport",
            inputFilename: inputURL.lastPathComponent,
            inputURL: inputURL.absoluteString,
            minRating: config.minRating,
            timeControls: config.timeControlClasses,
            maxGames: config.maxGames
        )

        var summary = Summary()
        summary.corpusID = corpus.corpusID
        var done = false

        try streamGames(at: expanded) { tags, movetext in
            if done { return }
            switch importOneGame(tags: tags, movetext: movetext, config: config) {
            case .imported(let game):
                do {
                    try corpus.append(game)
                    summary.imported += 1
                    if let mx = config.maxGames, summary.imported >= mx { done = true }
                    if summary.imported % 10_000 == 0 {
                        SessionLogger.shared.log("[PGN] imported \(summary.imported) games…")
                    }
                } catch {
                    SessionLogger.shared.log("[PGN] append failed: \(error.localizedDescription)")
                    summary.parseErrors += 1
                }
            case .skipped:
                summary.skipped += 1
            case .parseError:
                summary.parseErrors += 1
            }
        }

        try corpus.finishSource()
        return summary
    }

    private enum GameOutcomeResult {
        case imported(GameRecord)
        case skipped
        case parseError
    }

    private static func importOneGame(tags: [String: String], movetext: String, config: PGNImportConfig) -> GameOutcomeResult {
        // Standard start only.
        if tags["FEN"] != nil || tags["SetUp"] == "1" { return .skipped }
        if let variant = tags["Variant"], variant.lowercased() != "standard" { return .skipped }

        guard let result = tags["Result"], let outcome = outcome(forResult: result) else { return .skipped }

        if let minRating = config.minRating {
            let w = tags["WhiteElo"].flatMap { Int($0) }
            let b = tags["BlackElo"].flatMap { Int($0) }
            guard let wr = w, let br = b, wr >= minRating, br >= minRating else { return .skipped }
        }

        if let classes = config.timeControlClasses {
            let cls = timeControlClass(tags["TimeControl"] ?? "")
            guard classes.contains(cls) else { return .skipped }
        }

        let tokens = sanTokens(from: movetext)
        guard !tokens.isEmpty else { return .skipped }

        let engine = ChessGameEngine()
        var moves: [ChessMove] = []
        moves.reserveCapacity(tokens.count)
        for token in tokens {
            let legal = MoveGenerator.legalMoves(for: engine.state)
            guard let move = matchSAN(token, state: engine.state, legal: legal) else {
                return .parseError
            }
            moves.append(move)
            do {
                try engine.applyMoveAndAdvance(move)
            } catch {
                return .parseError
            }
        }

        guard moves.count >= config.minPlies else { return .skipped }
        return .imported(GameRecord(moves: moves, outcome: outcome, terminationReason: nil))
    }

    // MARK: - Streaming reader

    /// Decompress (if `.zst`) and stream the file, invoking `onGame(tags,
    /// movetext)` for each complete game.
    private static func streamGames(at path: String, onGame: (_ tags: [String: String], _ movetext: String) -> Void) throws {
        let process = Process()
        let usesZstd = path.hasSuffix(".zst")
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        process.arguments = usesZstd ? ["zstd", "-dc", path] : ["cat", path]
        let pipe = Pipe()
        process.standardOutput = pipe
        do {
            try process.run()
        } catch {
            throw PGNImportError.openFailed(error.localizedDescription)
        }
        let handle = pipe.fileHandleForReading

        // Per-game accumulation state.
        var tags: [String: String] = [:]
        var movetext = ""
        var inMoves = false

        func finalizeGame() {
            if !movetext.isEmpty || !tags.isEmpty {
                if !movetext.isEmpty { onGame(tags, movetext) }
                tags = [:]
                movetext = ""
            }
            inMoves = false
        }

        func processLine(_ raw: String) {
            let line = raw.trimmingCharacters(in: .whitespaces)
            if line.hasPrefix("[") && line.hasSuffix("]") {
                // A tag line that arrives after movetext begins a new game.
                if inMoves { finalizeGame() }
                if let (k, v) = parseTag(line) { tags[k] = v }
            } else if line.isEmpty {
                if inMoves { finalizeGame() }
                // blank between tags and movetext: ignore (movetext begins on
                // the next non-tag line).
            } else {
                inMoves = true
                movetext += " " + line
            }
        }

        var carry = Data()
        let newline: UInt8 = 0x0A
        while true {
            let chunk = handle.availableData
            if chunk.isEmpty { break }
            carry.append(chunk)
            while let nl = carry.firstIndex(of: newline) {
                let lineData = carry.subdata(in: carry.startIndex..<nl)
                carry.removeSubrange(carry.startIndex...nl)
                processLine(String(decoding: lineData, as: UTF8.self))
            }
        }
        if !carry.isEmpty {
            processLine(String(decoding: carry, as: UTF8.self))
        }
        finalizeGame()
        process.waitUntilExit()
    }

    private static func parseTag(_ line: String) -> (String, String)? {
        // [Key "Value"]
        var s = line
        s.removeFirst()       // [
        s.removeLast()        // ]
        guard let spaceIdx = s.firstIndex(of: " ") else { return nil }
        let key = String(s[s.startIndex..<spaceIdx])
        var value = String(s[s.index(after: spaceIdx)...]).trimmingCharacters(in: .whitespaces)
        if value.hasPrefix("\"") { value.removeFirst() }
        if value.hasSuffix("\"") { value.removeLast() }
        return (key, value)
    }

    // MARK: - Movetext → SAN tokens

    static func sanTokens(from movetext: String) -> [String] {
        let noComments = removeBraced(movetext, open: "{", close: "}")
        let noVariations = removeBraced(noComments, open: "(", close: ")")
        var result: [String] = []
        for raw in noVariations.split(whereSeparator: { $0 == " " || $0 == "\n" || $0 == "\t" || $0 == "\r" }) {
            if raw.first == "$" { continue }                       // NAG
            let token = stripMoveNumberPrefix(String(raw))
            if token.isEmpty { continue }
            if token == "1-0" || token == "0-1" || token == "1/2-1/2" || token == "*" { continue }
            result.append(token)
        }
        return result
    }

    private static func removeBraced(_ s: String, open: Character, close: Character) -> String {
        var out = ""
        out.reserveCapacity(s.count)
        var depth = 0
        for ch in s {
            if ch == open {
                depth += 1
            } else if ch == close {
                if depth > 0 { depth -= 1 }
            } else if depth == 0 {
                out.append(ch)
            }
        }
        return out
    }

    private static func stripMoveNumberPrefix(_ s: String) -> String {
        let chars = Array(s)
        var i = 0
        while i < chars.count && chars[i].isNumber { i += 1 }
        guard i > 0 else { return s }
        // Bare "1-0"/"0-1" start with a digit but are results, not move numbers
        // — only treat as a move number if a dot follows the digits.
        guard i < chars.count && chars[i] == "." else { return s }
        while i < chars.count && chars[i] == "." { i += 1 }
        return String(chars[i...])
    }

    // MARK: - SAN → ChessMove

    /// Match a SAN token against the legal moves of `state`. Returns the unique
    /// matching `ChessMove`, or nil if the token doesn't resolve.
    static func matchSAN(_ token: String, state: GameState, legal: [ChessMove]) -> ChessMove? {
        var san = token
        // Strip check/mate/annotation suffixes.
        while let last = san.last, "+#!?".contains(last) { san.removeLast() }
        if san.isEmpty { return nil }

        // Castling.
        if san == "O-O" || san == "0-0" {
            return legal.first { m in isKing(m, state) && m.toCol == m.fromCol + 2 }
        }
        if san == "O-O-O" || san == "0-0-0" {
            return legal.first { m in isKing(m, state) && m.toCol == m.fromCol - 2 }
        }

        var chars = Array(san)

        // Promotion: "=Q" or a trailing piece letter ("e8Q").
        var promotion: PieceType? = nil
        if let eq = chars.firstIndex(of: "=") {
            if eq + 1 < chars.count { promotion = sanPiece(chars[eq + 1]) }
            chars = Array(chars[0..<eq])
        } else if let last = chars.last, "NBRQ".contains(last), chars.count >= 3 {
            promotion = sanPiece(last)
            chars.removeLast()
        }

        // Leading piece letter (absent ⇒ pawn).
        var pieceType: PieceType = .pawn
        var idx = 0
        if let first = chars.first, "NBRQK".contains(first), let p = sanPiece(first) {
            pieceType = p
            idx = 1
        }

        // Strip capture markers; what's left is [disambig]? dest.
        let rest = Array(chars[idx...]).filter { $0 != "x" }
        guard rest.count >= 2 else { return nil }
        guard let dest = square(file: rest[rest.count - 2], rank: rest[rest.count - 1]) else { return nil }

        var disFile: Int? = nil
        var disRow: Int? = nil
        for ch in rest[0..<(rest.count - 2)] {
            if let c = col(forFile: ch) {
                disFile = c
            } else if let n = ch.wholeNumberValue, (1...8).contains(n) {
                disRow = 8 - n
            }
        }

        let candidates = legal.filter { m in
            guard m.toRow == dest.row, m.toCol == dest.col else { return false }
            guard m.promotion == promotion else { return false }
            guard let piece = state.board[m.fromRow * 8 + m.fromCol], piece.type == pieceType else { return false }
            if let df = disFile, m.fromCol != df { return false }
            if let dr = disRow, m.fromRow != dr { return false }
            return true
        }
        return candidates.first
    }

    private static func isKing(_ m: ChessMove, _ state: GameState) -> Bool {
        state.board[m.fromRow * 8 + m.fromCol]?.type == .king
    }

    private static func sanPiece(_ ch: Character) -> PieceType? {
        switch ch {
        case "N": return .knight
        case "B": return .bishop
        case "R": return .rook
        case "Q": return .queen
        case "K": return .king
        default:  return nil
        }
    }

    private static let files = Array("abcdefgh")

    private static func col(forFile ch: Character) -> Int? {
        files.firstIndex(of: ch)
    }

    /// Algebraic (file,rank) → (row,col) in the engine's convention: row 0 is
    /// rank 8, col 0 is file a (matches `FENParser`/`BoardEncoder`).
    private static func square(file: Character, rank: Character) -> (row: Int, col: Int)? {
        guard let col = col(forFile: file),
              let rankNum = rank.wholeNumberValue, (1...8).contains(rankNum) else { return nil }
        return (row: 8 - rankNum, col: col)
    }

    // MARK: - Result / time-control classification

    private static func outcome(forResult r: String) -> GameOutcome? {
        switch r {
        case "1-0":     return .whiteWin
        case "0-1":     return .blackWin
        case "1/2-1/2": return .draw
        default:        return nil
        }
    }

    /// Lichess-style time-control class from a `TimeControl` tag like "600+0".
    static func timeControlClass(_ tc: String) -> String {
        let parts = tc.split(separator: "+")
        let base = Int(parts.first.map(String.init) ?? "") ?? 0
        let inc = parts.count > 1 ? (Int(parts[1]) ?? 0) : 0
        let estimated = base + 40 * inc
        switch estimated {
        case ..<29:   return "ultrabullet"
        case ..<179:  return "bullet"
        case ..<479:  return "blitz"
        case ..<1499: return "rapid"
        default:      return "classical"
        }
    }
}
