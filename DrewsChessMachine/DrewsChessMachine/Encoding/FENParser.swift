import Foundation

/// Parses standard FEN (Forsyth-Edwards Notation) into a `GameState`.
///
/// Used by `LichessProbeData` to ingest the bundled 200-puzzle probe set,
/// where each puzzle stores its starting position as a FEN string sourced
/// from the Lichess puzzle database. The parser is deliberately structural
/// — it validates field shape (six fields, eight ranks summing to eight,
/// recognisable piece letters, etc.) but does NOT validate chess legality
/// (kings present, no double-checks, en passant only when stm could
/// capture, etc.). Bundle data has already been verified upstream by
/// python-chess in the curation script, and the parser is not meant for
/// user input.
///
/// Fullmove number (sixth FEN field) is parsed for validation only —
/// `GameState` does not store it. Halfmove clock IS persisted.
///
/// `repetitionCount` and `recentRepetitionMask` default to 0 on the
/// resulting `GameState`. FEN by definition does not carry game history,
/// so the position is treated as the first occurrence with no prior
/// recent repetitions — the same as the project's existing tactical-
/// probe fixtures (which all construct `GameState` directly without
/// game history).
enum FENParser {

    enum ParseError: Swift.Error, CustomStringConvertible {
        case wrongFieldCount(Int)
        case wrongRankCount(Int)
        case rankFileSumMismatch(rankIndex: Int, sum: Int)
        case unknownPieceCharacter(Character)
        case badSideToMove(String)
        case badCastlingField(String)
        case badEnPassantField(String)
        case badHalfmoveClock(String)
        case badFullmoveNumber(String)

        var description: String {
            switch self {
            case .wrongFieldCount(let n):
                return "FEN must have 6 space-separated fields, got \(n)"
            case .wrongRankCount(let n):
                return "FEN board must have 8 ranks, got \(n)"
            case .rankFileSumMismatch(let r, let sum):
                return "FEN rank \(r) files sum to \(sum), expected 8"
            case .unknownPieceCharacter(let c):
                return "FEN piece character '\(c)' is not a known piece"
            case .badSideToMove(let s):
                return "FEN side-to-move must be 'w' or 'b', got '\(s)'"
            case .badCastlingField(let s):
                return "FEN castling field invalid: '\(s)'"
            case .badEnPassantField(let s):
                return "FEN en-passant field invalid: '\(s)'"
            case .badHalfmoveClock(let s):
                return "FEN halfmove clock not an integer: '\(s)'"
            case .badFullmoveNumber(let s):
                return "FEN fullmove number not an integer: '\(s)'"
            }
        }
    }

    /// Parse a standard FEN string into a `GameState`. Throws `ParseError`
    /// on any structural problem.
    static func parse(_ fen: String) throws -> GameState {
        let fields = fen.split(separator: " ", omittingEmptySubsequences: true)
        guard fields.count == 6 else {
            throw ParseError.wrongFieldCount(fields.count)
        }

        let board = try parseBoard(String(fields[0]))
        let stm = try parseSideToMove(String(fields[1]))
        let castling = try parseCastling(String(fields[2]))
        let ep = try parseEnPassant(String(fields[3]))
        let halfmoveClock = try parseHalfmoveClock(String(fields[4]))
        _ = try parseFullmoveNumber(String(fields[5]))

        return GameState(
            board: board,
            currentPlayer: stm,
            whiteKingsideCastle: castling.whiteKingside,
            whiteQueensideCastle: castling.whiteQueenside,
            blackKingsideCastle: castling.blackKingside,
            blackQueensideCastle: castling.blackQueenside,
            enPassantSquare: ep,
            halfmoveClock: halfmoveClock
        )
    }

    // MARK: - Encoding

    /// Serialize a `GameState` back to FEN — the inverse of `parse`, used to log
    /// reproducible positions (e.g. a move-generation cross-check divergence).
    /// `GameState` doesn't track the fullmove number, so a constant placeholder
    /// is emitted for it; that field doesn't affect move generation and the FEN
    /// still round-trips through `parse`.
    static func fen(from state: GameState) -> String {
        var ranks: [String] = []
        ranks.reserveCapacity(8)
        for row in 0..<8 {
            var rank = ""
            var empty = 0
            for col in 0..<8 {
                if let piece = state.board[row * 8 + col] {
                    if empty > 0 { rank += String(empty); empty = 0 }
                    rank.append(fenChar(for: piece))
                } else {
                    empty += 1
                }
            }
            if empty > 0 { rank += String(empty) }
            ranks.append(rank)
        }
        let placement = ranks.joined(separator: "/")
        let side = state.currentPlayer == .white ? "w" : "b"
        var castling = ""
        if state.whiteKingsideCastle { castling += "K" }
        if state.whiteQueensideCastle { castling += "Q" }
        if state.blackKingsideCastle { castling += "k" }
        if state.blackQueensideCastle { castling += "q" }
        if castling.isEmpty { castling = "-" }
        let ep: String
        if let e = state.enPassantSquare {
            ep = "\(Character(UnicodeScalar(UInt8(97 + e.col))))\(8 - e.row)"
        } else {
            ep = "-"
        }
        return "\(placement) \(side) \(castling) \(ep) \(state.halfmoveClock) 1"
    }

    private static func fenChar(for piece: Piece) -> Character {
        let base: Character
        switch piece.type {
        case .pawn:   base = "p"
        case .knight: base = "n"
        case .bishop: base = "b"
        case .rook:   base = "r"
        case .queen:  base = "q"
        case .king:   base = "k"
        }
        return piece.color == .white ? Character(base.uppercased()) : base
    }

    // MARK: - Field parsers

    private static func parseBoard(_ field: String) throws -> [Piece?] {
        let ranks = field.split(separator: "/", omittingEmptySubsequences: false)
        guard ranks.count == 8 else {
            throw ParseError.wrongRankCount(ranks.count)
        }

        var board: [Piece?] = Array(repeating: nil, count: 64)
        for (rankIndex, rank) in ranks.enumerated() {
            // FEN's first rank token is rank 8 (board row 0).
            var col = 0
            for ch in rank {
                if let empties = ch.wholeNumberValue, empties >= 1, empties <= 8 {
                    col += empties
                    continue
                }
                let piece = try pieceForFENChar(ch)
                guard col < 8 else {
                    throw ParseError.rankFileSumMismatch(
                        rankIndex: rankIndex,
                        sum: col + 1
                    )
                }
                board[rankIndex * 8 + col] = piece
                col += 1
            }
            guard col == 8 else {
                throw ParseError.rankFileSumMismatch(
                    rankIndex: rankIndex,
                    sum: col
                )
            }
        }
        return board
    }

    private static func pieceForFENChar(_ ch: Character) throws -> Piece {
        let color: PieceColor = ch.isUppercase ? .white : .black
        let type: PieceType
        switch ch.lowercased() {
        case "p": type = .pawn
        case "n": type = .knight
        case "b": type = .bishop
        case "r": type = .rook
        case "q": type = .queen
        case "k": type = .king
        default:  throw ParseError.unknownPieceCharacter(ch)
        }
        return Piece(type: type, color: color)
    }

    private static func parseSideToMove(_ field: String) throws -> PieceColor {
        switch field {
        case "w": return .white
        case "b": return .black
        default:  throw ParseError.badSideToMove(field)
        }
    }

    private struct CastlingRights {
        var whiteKingside: Bool = false
        var whiteQueenside: Bool = false
        var blackKingside: Bool = false
        var blackQueenside: Bool = false
    }

    private static func parseCastling(_ field: String) throws -> CastlingRights {
        if field == "-" { return CastlingRights() }
        var r = CastlingRights()
        for ch in field {
            switch ch {
            case "K": r.whiteKingside = true
            case "Q": r.whiteQueenside = true
            case "k": r.blackKingside = true
            case "q": r.blackQueenside = true
            default:  throw ParseError.badCastlingField(field)
            }
        }
        return r
    }

    private static func parseEnPassant(_ field: String) throws -> (row: Int, col: Int)? {
        if field == "-" { return nil }
        guard field.count == 2 else {
            throw ParseError.badEnPassantField(field)
        }
        let chars = Array(field)
        guard
            let fileAscii = chars[0].asciiValue,
            fileAscii >= 0x61, fileAscii <= 0x68,
            let rankAscii = chars[1].asciiValue,
            rankAscii >= 0x31, rankAscii <= 0x38
        else {
            throw ParseError.badEnPassantField(field)
        }
        let col = Int(fileAscii - 0x61)
        let rank = Int(rankAscii - 0x30)  // 1...8
        let row = 8 - rank                 // rank 8 -> row 0
        return (row: row, col: col)
    }

    private static func parseHalfmoveClock(_ field: String) throws -> Int {
        guard let n = Int(field), n >= 0 else {
            throw ParseError.badHalfmoveClock(field)
        }
        return n
    }

    private static func parseFullmoveNumber(_ field: String) throws -> Int {
        guard let n = Int(field), n >= 1 else {
            throw ParseError.badFullmoveNumber(field)
        }
        return n
    }
}
