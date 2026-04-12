import Foundation

// MARK: - Chess Types

enum PieceColor: Sendable, Hashable {
    case white, black

    var opposite: PieceColor {
        switch self {
        case .white: return .black
        case .black: return .white
        }
    }
}

/// Piece types ordered to match tensor plane indices (0-5).
enum PieceType: Int, Sendable, CaseIterable, Hashable {
    case pawn = 0
    case knight = 1
    case bishop = 2
    case rook = 3
    case queen = 4
    case king = 5
}

struct Piece: Sendable, Hashable {
    let type: PieceType
    let color: PieceColor

    /// Asset catalog image name (e.g., "wK", "bP").
    var assetName: String {
        let colorPrefix = color == .white ? "w" : "b"
        let pieceCode: String
        switch type {
        case .pawn:   pieceCode = "P"
        case .knight: pieceCode = "N"
        case .bishop: pieceCode = "B"
        case .rook:   pieceCode = "R"
        case .queen:  pieceCode = "Q"
        case .king:   pieceCode = "K"
        }
        return "\(colorPrefix)\(pieceCode)"
    }
}

/// Complete game state needed for tensor encoding and move generation.
/// Board is stored in absolute coordinates: row 0 = rank 8, row 7 = rank 1.
struct GameState: Sendable {
    /// 64-square board, indexed as row * 8 + col.
    /// row 0 = rank 8, row 7 = rank 1, col 0 = a-file.
    /// Stored flat (instead of nested 8×8) so applyMove only triggers a single
    /// CoW copy of one array, not eight inner arrays plus the outer.
    let board: [Piece?]
    let currentPlayer: PieceColor
    let whiteKingsideCastle: Bool
    let whiteQueensideCastle: Bool
    let blackKingsideCastle: Bool
    let blackQueensideCastle: Bool
    /// En passant target square (where the capturing pawn lands), or nil.
    let enPassantSquare: (row: Int, col: Int)?
    /// Moves since last pawn move or capture (for fifty-move rule).
    let halfmoveClock: Int

    /// Convenience: read the piece at (row, col). Equivalent to board[row * 8 + col].
    @inline(__always)
    func piece(at row: Int, _ col: Int) -> Piece? {
        board[row * 8 + col]
    }

    static let starting: GameState = {
        var b: [Piece?] = Array(repeating: nil, count: 64)
        let backRank: [PieceType] = [.rook, .knight, .bishop, .queen, .king, .bishop, .knight, .rook]
        for col in 0..<8 {
            b[0 * 8 + col] = Piece(type: backRank[col], color: .black)
            b[1 * 8 + col] = Piece(type: .pawn, color: .black)
            b[6 * 8 + col] = Piece(type: .pawn, color: .white)
            b[7 * 8 + col] = Piece(type: backRank[col], color: .white)
        }
        return GameState(
            board: b,
            currentPlayer: .white,
            whiteKingsideCastle: true,
            whiteQueensideCastle: true,
            blackKingsideCastle: true,
            blackQueensideCastle: true,
            enPassantSquare: nil,
            halfmoveClock: 0
        )
    }()
}

// MARK: - Board Encoder

/// Encodes chess positions into the 18x8x8 tensor format expected by the network.
///
/// Always encoded from the current player's perspective:
/// - Board flipped vertically if black is playing (so current player is always at bottom)
/// - Planes 0-5: current player's pieces (pawn, knight, bishop, rook, queen, king)
/// - Planes 6-11: opponent's pieces (same order)
/// - Plane 12-13: current player's castling rights (kingside, queenside)
/// - Plane 14-15: opponent's castling rights (kingside, queenside)
/// - Plane 16: en passant target square
/// - Plane 17: halfmove clock (normalized 0.0-1.0)
enum BoardEncoder {

    /// Encode a game state into an 18x8x8 = 1,152 float tensor.
    static func encode(_ state: GameState) -> [Float] {
        var tensor = [Float](repeating: 0, count: 18 * 64)
        let flip = state.currentPlayer == .black

        // Planes 0-11: pieces
        for row in 0..<8 {
            let sourceRow = flip ? (7 - row) : row
            let sourceRowBase = sourceRow * 8
            let destRowBase = row * 8
            for col in 0..<8 {
                guard let piece = state.board[sourceRowBase + col] else { continue }

                let isMine = piece.color == state.currentPlayer
                let plane = (isMine ? 0 : 6) + piece.type.rawValue
                tensor[plane * 64 + destRowBase + col] = 1.0
            }
        }

        // Planes 12-15: castling rights (from current player's perspective)
        let myKingside: Bool
        let myQueenside: Bool
        let oppKingside: Bool
        let oppQueenside: Bool

        if flip {
            myKingside = state.blackKingsideCastle
            myQueenside = state.blackQueensideCastle
            oppKingside = state.whiteKingsideCastle
            oppQueenside = state.whiteQueensideCastle
        } else {
            myKingside = state.whiteKingsideCastle
            myQueenside = state.whiteQueensideCastle
            oppKingside = state.blackKingsideCastle
            oppQueenside = state.blackQueensideCastle
        }

        if myKingside { fillPlane(&tensor, plane: 12) }
        if myQueenside { fillPlane(&tensor, plane: 13) }
        if oppKingside { fillPlane(&tensor, plane: 14) }
        if oppQueenside { fillPlane(&tensor, plane: 15) }

        // Plane 16: en passant target square
        if let ep = state.enPassantSquare {
            let epRow = flip ? (7 - ep.row) : ep.row
            tensor[16 * 64 + epRow * 8 + ep.col] = 1.0
        }

        // Plane 17: halfmove clock (normalized, 100 = fifty-move rule limit)
        let normalized = Float(min(state.halfmoveClock, 100)) / 100.0
        if normalized > 0 {
            fillPlane(&tensor, plane: 17, value: normalized)
        }

        return tensor
    }

    /// Convenience: encode the starting position.
    static func encodeStartingPosition() -> [Float] {
        encode(.starting)
    }

    // MARK: - Piece Lookup

    /// Piece symbols for the starting position, used by the board visualization.
    /// Row 0 = rank 8 (top), row 7 = rank 1 (bottom). Indexed as row * 8 + col.
    static let startingPieces: [String?] = GameState.starting.board.map { $0?.assetName }

    // MARK: - Move Decoding

    /// Decode a policy index (0-4095) into source and destination square names.
    /// Index encoding: from_square * 64 + to_square
    static func decodeMove(index: Int) -> (from: String, to: String) {
        let fromSquare = index / 64
        let toSquare = index % 64
        return (squareName(fromSquare), squareName(toSquare))
    }

    /// Convert a square index (0-63) to algebraic notation (e.g., 0 = "a8", 63 = "h1").
    /// Squares numbered row-by-row from rank 8: 0=a8, 7=h8, 8=a7, ..., 56=a1, 63=h1.
    static func squareName(_ square: Int) -> String {
        let file = square % 8
        let rank = 8 - (square / 8)
        let fileChar = String(UnicodeScalar(UInt8(97 + file)))  // 97 = 'a'
        return "\(fileChar)\(rank)"
    }

    // MARK: - Private Helpers

    private static func fillPlane(_ tensor: inout [Float], plane: Int, value: Float = 1.0) {
        let start = plane * 64
        for i in start..<(start + 64) {
            tensor[i] = value
        }
    }
}
