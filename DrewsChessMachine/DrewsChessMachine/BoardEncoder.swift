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
    /// Number of times this exact position has occurred *previously* in
    /// the game (excluding the current visit). Saturated at 2 — a value
    /// of 2 means the next visit would force a 3-fold draw claim. Drives
    /// `BoardEncoder` planes 18 and 19. Default 0 so existing test/UI
    /// constructions of `GameState` (which don't track game history)
    /// produce the correct "no repetitions" encoding without breaking.
    /// `ChessGameEngine` populates the actual count after each move.
    let repetitionCount: Int

    /// Explicit memberwise initializer with default for `repetitionCount`
    /// so legacy callsites (tests, UI editable position, applyMove that
    /// doesn't know about game history) keep compiling without changes.
    /// `ChessGameEngine` is the only caller that supplies a non-default
    /// value, derived from its `positionCounts` table after each move.
    init(
        board: [Piece?],
        currentPlayer: PieceColor,
        whiteKingsideCastle: Bool,
        whiteQueensideCastle: Bool,
        blackKingsideCastle: Bool,
        blackQueensideCastle: Bool,
        enPassantSquare: (row: Int, col: Int)?,
        halfmoveClock: Int,
        repetitionCount: Int = 0
    ) {
        self.board = board
        self.currentPlayer = currentPlayer
        self.whiteKingsideCastle = whiteKingsideCastle
        self.whiteQueensideCastle = whiteQueensideCastle
        self.blackKingsideCastle = blackKingsideCastle
        self.blackQueensideCastle = blackQueensideCastle
        self.enPassantSquare = enPassantSquare
        self.halfmoveClock = halfmoveClock
        self.repetitionCount = repetitionCount
    }

    /// Convenience: read the piece at (row, col). Equivalent to board[row * 8 + col].
    @inline(__always)
    func piece(at row: Int, _ col: Int) -> Piece? {
        board[row * 8 + col]
    }

    /// Return a copy with `repetitionCount` replaced. Used by
    /// `ChessGameEngine` to layer the rep count onto a state produced
    /// by `MoveGenerator.applyMove` (which has no history awareness).
    func withRepetitionCount(_ count: Int) -> GameState {
        GameState(
            board: board,
            currentPlayer: currentPlayer,
            whiteKingsideCastle: whiteKingsideCastle,
            whiteQueensideCastle: whiteQueensideCastle,
            blackKingsideCastle: blackKingsideCastle,
            blackQueensideCastle: blackQueensideCastle,
            enPassantSquare: enPassantSquare,
            halfmoveClock: halfmoveClock,
            repetitionCount: count
        )
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

/// Encodes chess positions into the 20x8x8 tensor format expected by the network.
///
/// Always encoded from the current player's perspective:
/// - Board flipped vertically if black is playing (so current player is always at bottom)
/// - Planes 0-5: current player's pieces (pawn, knight, bishop, rook, queen, king)
/// - Planes 6-11: opponent's pieces (same order)
/// - Plane 12-13: current player's castling rights (kingside, queenside)
/// - Plane 14-15: opponent's castling rights (kingside, queenside)
/// - Plane 16: en passant target square
/// - Plane 17: halfmove clock, normalized as `min(clock, 99) / 99` (Leela-style)
/// - Plane 18: 1.0 if current position has occurred ≥1 time before in this game
///   (this is at least the 2nd visit — a repeat that signals possible shuffling)
/// - Plane 19: 1.0 if current position has occurred ≥2 times before
///   (this is at least the 3rd visit — the game is at the 3-fold draw threshold)
enum BoardEncoder {

    /// Number of floats one encoded position occupies: 20 planes × 64 squares = 1280.
    static let tensorLength = ChessNetwork.inputPlanes * ChessNetwork.boardSize * ChessNetwork.boardSize

    /// Encode a game state into a caller-owned slice of `tensorLength` (= 1,280) floats.
    ///
    /// The per-move inference hot path uses this variant so the encoded
    /// tensor can live in a pre-allocated per-game scratch buffer,
    /// avoiding a fresh `[Float](tensorLength)` allocation on every ply. The
    /// buffer is zero-filled in place first — callers do not need to
    /// clear it themselves — then the occupancy, castling, en-passant
    /// and halfmove planes are written according to the standard
    /// encoding. The buffer must have at least `tensorLength` elements.
    static func encode(
        _ state: GameState,
        into buffer: UnsafeMutableBufferPointer<Float>
    ) {
        precondition(
            buffer.count >= tensorLength,
            "BoardEncoder.encode(into:): buffer must hold at least \(tensorLength) floats (got \(buffer.count))"
        )
        // Pointer form once, then reuse — `UnsafeMutableBufferPointer`
        // subscripting bounds-checks every access, which adds up over
        // `tensorLength` writes per ply.
        guard let base = buffer.baseAddress else { return }

        // Zero the full tensorLength region. Sparse planes (pieces, EP)
        // rely on this being cleared first. Uses `update` (not
        // `initialize`) because callers pass in already-initialized
        // storage — a slice of a reused `[Float]` array or a
        // previously-initialized `UnsafeMutablePointer` allocation.
        base.update(repeating: 0, count: tensorLength)

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
                base[plane * 64 + destRowBase + col] = 1.0
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

        if myKingside { fillPlane(base, plane: 12) }
        if myQueenside { fillPlane(base, plane: 13) }
        if oppKingside { fillPlane(base, plane: 14) }
        if oppQueenside { fillPlane(base, plane: 15) }

        // Plane 16: en passant target square
        if let ep = state.enPassantSquare {
            let epRow = flip ? (7 - ep.row) : ep.row
            base[16 * 64 + epRow * 8 + ep.col] = 1.0
        }

        // Plane 17: halfmove clock, normalized as `min(clock, 99) / 99`.
        // Saturates at 1.0 on the 99th ply of no progress — the last ply
        // before either side can claim the 50-move rule on their next
        // turn. Matches Leela Chess Zero's `rule50_count / 99` convention:
        // putting the saturation point at the move-decision boundary
        // rather than at the rule-firing moment gives the value head a
        // signal aligned with when a player must actually act on the
        // approaching draw, not one ply later. The actual 50-move-rule
        // game logic (in ChessGameEngine) still fires at clock >= 100;
        // only the normalization of this *input feature* changes.
        let normalized = Float(min(state.halfmoveClock, 99)) / 99.0
        if normalized > 0 {
            fillPlane(base, plane: 17, value: normalized)
        }

        // Planes 18 and 19: threefold-repetition signals. Always-fill
        // pattern (no skip-if-zero optimization) so each plane is
        // self-contained and doesn't depend on the leading clear at
        // line 136 — easier to reason about and immune to the silent
        // failure mode where someone bumps `tensorLength` without
        // updating the leading clear's count. Cost is 128 extra writes
        // per encode, negligible against the existing tensorLength-float clear.
        //
        // The rep count is read from the GameState itself (populated
        // by ChessGameEngine after every move from its positionCounts
        // table). For tests / UI editable positions / .starting where
        // the count defaults to 0, both planes are zero — the correct
        // "no repetition history" encoding.
        let repCount = state.repetitionCount
        fillPlane(base, plane: 18, value: repCount >= 1 ? 1.0 : 0.0)
        fillPlane(base, plane: 19, value: repCount >= 2 ? 1.0 : 0.0)
    }

    /// Encode a game state into a 20×8×8 = 1,280 float tensor.
    ///
    /// Allocating variant — delegates to `encode(_:into:)` so both
    /// paths share the same encoding logic. Used by non-hot-path
    /// callers (tests, the Forward Pass demo UI). Hot-path callers
    /// should use `encode(_:into:)` with a pre-allocated scratch.
    static func encode(_ state: GameState) -> [Float] {
        var tensor = [Float](repeating: 0, count: tensorLength)
        tensor.withUnsafeMutableBufferPointer { buf in
            encode(state, into: buf)
        }
        return tensor
    }

    /// Convenience: encode the starting position.
    static func encodeStartingPosition() -> [Float] {
        encode(.starting)
    }

    /// Reconstruct a "synthetic white-to-move" `GameState` from a raw
    /// encoded tensor.
    ///
    /// The encoding is always from the current player's perspective
    /// (board flipped for black-to-move so the mover sits at rows 6-7)
    /// and the policy-index encoding runs in the same encoder frame.
    /// That means the mover's color is **not recoverable** from the
    /// tensor alone — and doesn't need to be for either legal-move
    /// enumeration or policy-index computation. Labeling the mover as
    /// white (no flip) yields a state whose `MoveGenerator.legalMoves`
    /// returns moves in encoder-frame coordinates, which are exactly
    /// what `PolicyEncoding.policyIndex(_:currentPlayer: .white)`
    /// expects to produce the same flat indices the network is
    /// already emitting.
    ///
    /// Used by `ChessTrainer.legalMassSnapshot` to compute how much
    /// softmax mass the current policy places on the legal-move set
    /// for a sampled batch of replay-buffer positions. The repetition
    /// planes (18/19) and `halfmoveClock`'s exact value don't affect
    /// legality (the 50-move-rule fires at clock ≥ 100, handled
    /// separately in `MoveGenerator`), so we ignore them.
    ///
    /// - Parameter buffer: Exactly `tensorLength` (1280) floats in the
    ///   NCHW row-major layout produced by `encode(_:into:)`.
    /// - Returns: A `GameState` with `currentPlayer = .white` whose
    ///   `MoveGenerator.legalMoves` output lines up with the policy
    ///   indices stored for this position in the replay buffer.
    static func decodeSynthetic(
        from buffer: UnsafePointer<Float>
    ) -> GameState {
        var board: [Piece?] = Array(repeating: nil, count: 64)

        // Planes 0-5: mover's pieces (pawn..king) — labeled as white.
        for plane in 0..<6 {
            let pieceType = PieceType(rawValue: plane)!
            for row in 0..<8 {
                for col in 0..<8 {
                    if buffer[plane * 64 + row * 8 + col] > 0.5 {
                        board[row * 8 + col] = Piece(type: pieceType, color: .white)
                    }
                }
            }
        }
        // Planes 6-11: opponent's pieces — labeled as black.
        for plane in 6..<12 {
            let pieceType = PieceType(rawValue: plane - 6)!
            for row in 0..<8 {
                for col in 0..<8 {
                    if buffer[plane * 64 + row * 8 + col] > 0.5 {
                        board[row * 8 + col] = Piece(type: pieceType, color: .black)
                    }
                }
            }
        }

        // Planes 12-15: castling rights (mover's kingside, mover's
        // queenside, opp kingside, opp queenside). Plane-is-solid-1 =
        // right available. Read a single corner square as a cheap
        // probe — `fillPlane` writes the whole 64-square plane so any
        // cell carries the flag.
        let myKingside = buffer[12 * 64] > 0.5
        let myQueenside = buffer[13 * 64] > 0.5
        let oppKingside = buffer[14 * 64] > 0.5
        let oppQueenside = buffer[15 * 64] > 0.5

        // Plane 16: en passant target (single cell). Scan the whole
        // plane since we don't know which cell is hot — under a legal
        // encoding at most one cell is set.
        var enPassant: (row: Int, col: Int)?
        for idx in 0..<64 where buffer[16 * 64 + idx] > 0.5 {
            enPassant = (idx / 8, idx % 8)
            break
        }

        // Plane 17: halfmove clock, normalized as min(clock,99)/99.
        // Round-trip to an integer clock for the reconstructed state.
        // The exact value only matters for 50-move-rule termination,
        // not legality.
        let clockProbe = buffer[17 * 64]
        let halfmoveClock = Int((clockProbe * 99).rounded())

        return GameState(
            board: board,
            currentPlayer: .white,
            whiteKingsideCastle: myKingside,
            whiteQueensideCastle: myQueenside,
            blackKingsideCastle: oppKingside,
            blackQueensideCastle: oppQueenside,
            enPassantSquare: enPassant,
            halfmoveClock: halfmoveClock,
            repetitionCount: 0
        )
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

    private static func fillPlane(
        _ base: UnsafeMutablePointer<Float>,
        plane: Int,
        value: Float = 1.0
    ) {
        let start = plane * 64
        for i in start..<(start + 64) {
            base[i] = value
        }
    }
}
