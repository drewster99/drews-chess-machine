import Accelerate
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

    /// Standard chess material value, used by the arena
    /// material-advantage breakdowns. Kings are 0 — both sides always
    /// have exactly one, so a king contributes a constant that cancels
    /// out of any candidate-minus-opponent difference.
    var materialValue: Int {
        switch self {
        case .pawn:   return 1
        case .knight: return 3
        case .bishop: return 3
        case .rook:   return 5
        case .queen:  return 9
        case .king:   return 0
        }
    }
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

    /// Packed 10-bit mask of which recent prior positions equal this
    /// position under `PositionKey` semantics (board + side-to-move +
    /// all four castling rights + en-passant target). Bit `i` (0-indexed)
    /// is 1 iff the position `i + 1` plies ago is a strict chess-rules
    /// duplicate of the current position. Drives `BoardEncoder` planes
    /// 20–29 — plane `20 + i` is all-1 iff bit `i` is set.
    ///
    /// Unlike `repetitionCount` (which counts total occurrences regardless
    /// of when they happened), this carries the *temporal pattern* of
    /// recent repetitions — e.g. a 2-ply cycle visited three times sets
    /// bit 1 and bit 3 simultaneously. The intended use is letting the
    /// network see "I'm caught in an N-ply shuffle" as a distinct feature
    /// from "I've been at this position before."
    ///
    /// Cleared (along with `repetitionCount`) on any irreversible move
    /// (halfmove clock = 0), matching `ChessGameEngine.positionCounts`
    /// semantics — positions before an irreversible move can never recur.
    /// Default 0 so existing test/UI constructions of `GameState` produce
    /// the correct "no recent repetitions" encoding without breaking.
    let recentRepetitionMask: UInt16

    /// Explicit memberwise initializer with defaults for `repetitionCount`
    /// and `recentRepetitionMask` so legacy callsites (tests, UI editable
    /// position, applyMove that doesn't know about game history) keep
    /// compiling without changes. `ChessGameEngine` is the only caller
    /// that supplies non-default values, derived from its `positionCounts`
    /// table and `recentPositionKeys` window after each move.
    init(
        board: [Piece?],
        currentPlayer: PieceColor,
        whiteKingsideCastle: Bool,
        whiteQueensideCastle: Bool,
        blackKingsideCastle: Bool,
        blackQueensideCastle: Bool,
        enPassantSquare: (row: Int, col: Int)?,
        halfmoveClock: Int,
        repetitionCount: Int = 0,
        recentRepetitionMask: UInt16 = 0
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
        self.recentRepetitionMask = recentRepetitionMask
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
            repetitionCount: count,
            recentRepetitionMask: recentRepetitionMask
        )
    }

    /// Return a copy with `recentRepetitionMask` replaced. Used by
    /// `ChessGameEngine` to layer the temporal-repetition signal onto
    /// a state after each move, in parallel with `withRepetitionCount`.
    func withRecentRepetitionMask(_ mask: UInt16) -> GameState {
        GameState(
            board: board,
            currentPlayer: currentPlayer,
            whiteKingsideCastle: whiteKingsideCastle,
            whiteQueensideCastle: whiteQueensideCastle,
            blackKingsideCastle: blackKingsideCastle,
            blackQueensideCastle: blackQueensideCastle,
            enPassantSquare: enPassantSquare,
            halfmoveClock: halfmoveClock,
            repetitionCount: repetitionCount,
            recentRepetitionMask: mask
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

/// Encodes chess positions into the 30x8x8 tensor format expected by the network.
///
/// Always encoded from the current player's perspective:
/// - Board flipped vertically if black is playing (so current player is always at bottom)
/// - Planes 0-5: current player's pieces (pawn, knight, bishop, rook, queen, king)
/// - Planes 6-11: opponent's pieces (same order)
/// - Planes 12-13: current player's castling rights (kingside, queenside)
/// - Planes 14-15: opponent's castling rights (kingside, queenside)
/// - Plane 16: en passant target square
/// - Plane 17: halfmove clock, normalized as `min(clock, 99) / 99` (Leela-style)
/// - Plane 18: 1.0 if current position has occurred ≥1 time before in this game
///   (this is at least the 2nd visit — a repeat that signals possible shuffling)
/// - Plane 19: 1.0 if current position has occurred ≥2 times before
///   (this is at least the 3rd visit — the game is at the 3-fold draw threshold)
/// - Planes 20-29: temporal-repetition history, broadcast-scalar (all-0 or
///   all-1 across 64 cells). Plane `20 + i` is all-1 iff the position
///   `i + 1` plies ago is a strict chess-rules duplicate (under
///   `PositionKey` semantics: board + side-to-move + all four castling
///   rights + en-passant target) of the current position. Index 0 = 1 ply
///   ago (the most recent prior position); index 9 = 10 plies ago. Zero-
///   padded when fewer than `i + 1` plies of history are available (game
///   start, or after an irreversible move that clears the window). Drives
///   the network's awareness of the *temporal pattern* of repetitions —
///   distinguishing a 2-ply shuffle (bits 1 and 3 set after two visits)
///   from a longer maneuvering cycle — which planes 18-19 cannot express.
enum BoardEncoder {

    /// Number of floats one encoded position occupies for `encoding`:
    /// `planeCount × 64` (basic20 → 1280, basic30 → 1920).
    static func tensorLength(for encoding: InputEncoding) -> Int {
        encoding.planeCount * ChessNetwork.boardSize * ChessNetwork.boardSize
    }

    /// Encode a game state into a caller-owned slice of `tensorLength` floats.
    ///
    /// The per-move inference hot path uses this variant so the encoded
    /// tensor can live in a pre-allocated per-game scratch buffer,
    /// avoiding a fresh `[Float](tensorLength)` allocation on every ply. The
    /// buffer is zero-filled in place first — callers do not need to
    /// clear it themselves — then the occupancy, castling, en-passant
    /// and halfmove planes are written according to the standard
    /// encoding. The buffer must have at least `tensorLength` elements.
    static func encode(
        _ current: GameState,
        history: [GameState] = [],
        perspective: PieceColor? = nil,
        into buffer: UnsafeMutableBufferPointer<Float>,
        encoding: InputEncoding
    ) {
        let tensorLength = Self.tensorLength(for: encoding)
        precondition(
            buffer.count >= tensorLength,
            "BoardEncoder.encode(into:): buffer must hold at least \(tensorLength) floats (got \(buffer.count))"
        )
        // Pointer form once, then reuse — `UnsafeMutableBufferPointer`
        // subscripting bounds-checks every access, which adds up over
        // `tensorLength` writes per ply. The precondition above pins
        // `buffer.count >= tensorLength`, so `baseAddress` is non-nil
        // here; a nil here means the caller's invariant is broken.
        guard let base = buffer.baseAddress else {
            preconditionFailure(
                "BoardEncoder.encode(into:): buffer baseAddress is nil "
                + "despite count=\(buffer.count) >= \(tensorLength); upstream invariant violated."
            )
        }

        // Zero the full tensorLength region. Sparse planes (pieces, EP)
        // rely on this being cleared first. Uses `update` (not
        // `initialize`) because callers pass in already-initialized
        // storage — a slice of a reused `[Float]` array or a
        // previously-initialized `UnsafeMutablePointer` allocation.
        base.update(repeating: 0, count: tensorLength)

        // All frames are written from the ply-N mover's perspective. For
        // single-frame encodings that's just `current.currentPlayer`; the
        // history path passes the ply-N mover explicitly so an odd,
        // opponent-to-move prior frame still shows *our* pieces in planes
        // 0–5, oriented to our side.
        let persp = perspective ?? current.currentPlayer

        switch encoding {
        case .basic20:
            writeBasicBlock(current, perspective: persp, planeBase: 0,
                            includeTemporalRepetition: false, base: base)
        case .basic30:
            writeBasicBlock(current, perspective: persp, planeBase: 0,
                            includeTemporalRepetition: true, base: base)
        case .full10ply200:
            // Frame 0 = current (ply N); frame f = history[f-1] = ply N-f
            // when available. All share `persp`. Absent frames stay zero
            // from the leading clear (the "no frame here" signal). Each
            // frame is the 20-plane basic20 set — no per-frame temporal-
            // repetition block.
            let stride = encoding.planesPerFrame
            writeBasicBlock(current, perspective: persp, planeBase: 0,
                            includeTemporalRepetition: false, base: base)
            let available = min(history.count, encoding.historyFrameCount - 1)
            for f in 0..<available {
                writeBasicBlock(history[f], perspective: persp,
                                planeBase: (f + 1) * stride,
                                includeTemporalRepetition: false, base: base)
            }
        case .full10Ply10Reps210:
            // Same 10-frame basic20 stack as full10ply200 (mirrored, not
            // shared, so full10ply200's path stays byte-identical), then the
            // CURRENT position's 10 temporal-repetition planes appended at the
            // tail (planes 200–209), read from the engine-maintained mask —
            // bit-for-bit identical to basic30's planes 20–29. History frames
            // carry no reps. The training path reproduces this same tail from
            // stored priors in `ReplayBuffer.appendRepetitionTail`.
            let stride = encoding.planesPerFrame
            writeBasicBlock(current, perspective: persp, planeBase: 0,
                            includeTemporalRepetition: false, base: base)
            let available = min(history.count, encoding.historyFrameCount - 1)
            for f in 0..<available {
                writeBasicBlock(history[f], perspective: persp,
                                planeBase: (f + 1) * stride,
                                includeTemporalRepetition: false, base: base)
            }
            let repBase = encoding.historyFrameCount * encoding.planesPerFrame
            let recentMask = current.recentRepetitionMask
            if recentMask != 0 {
                for i in 0..<10 where (recentMask >> i) & 1 == 1 {
                    fillPlane(base, plane: repBase + i)
                }
            }
        }
    }

    /// Write one 20-plane `basic20` block — optionally plus the 10
    /// temporal-repetition planes (20–29) for basic30 — starting at
    /// `planeBase`, oriented to `perspective`.
    ///
    /// Piece placement (mine 0–5 / opponent's 6–11), the vertical flip,
    /// castling assignment, and en-passant orientation are all keyed to
    /// `perspective`, NOT `state.currentPlayer`. That's what lets a prior
    /// history frame whose own mover was the opponent still render from our
    /// side. The halfmove-clock and repetition planes carry the frame's own
    /// values. `base` must already be zero-cleared across the full tensor.
    private static func writeBasicBlock(
        _ state: GameState,
        perspective: PieceColor,
        planeBase: Int,
        includeTemporalRepetition: Bool,
        base: UnsafeMutablePointer<Float>
    ) {
        let flip = perspective == .black

        // Planes [+0 … +11]: pieces.
        for row in 0..<8 {
            let sourceRow = flip ? (7 - row) : row
            let sourceRowBase = sourceRow * 8
            let destRowBase = row * 8
            for col in 0..<8 {
                guard let piece = state.board[sourceRowBase + col] else { continue }

                let isMine = piece.color == perspective
                let plane = planeBase + (isMine ? 0 : 6) + piece.type.rawValue
                base[plane * 64 + destRowBase + col] = 1.0
            }
        }

        // Planes [+12 … +15]: castling rights (from `perspective`).
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

        if myKingside { fillPlane(base, plane: planeBase + 12) }
        if myQueenside { fillPlane(base, plane: planeBase + 13) }
        if oppKingside { fillPlane(base, plane: planeBase + 14) }
        if oppQueenside { fillPlane(base, plane: planeBase + 15) }

        // Plane [+16]: en passant target square.
        if let ep = state.enPassantSquare {
            let epRow = flip ? (7 - ep.row) : ep.row
            base[(planeBase + 16) * 64 + epRow * 8 + ep.col] = 1.0
        }

        // Plane [+17]: halfmove clock, normalized as `min(clock, 99) / 99`
        // (Leela's rule50 convention — saturation at the move-decision
        // boundary). The real 50-move-rule logic still fires at clock >=
        // 100 in ChessGameEngine; only this input feature's scale changes.
        let normalized = Float(min(state.halfmoveClock, 99)) / 99.0
        if normalized > 0 {
            fillPlane(base, plane: planeBase + 17, value: normalized)
        }

        // Planes [+18, +19]: threefold-repetition signals. Always-fill (no
        // skip-if-zero) so each plane is self-contained. Read from the
        // frame's own GameState; .starting / tests / UI positions default
        // to 0 → both planes zero.
        let repCount = state.repetitionCount
        fillPlane(base, plane: planeBase + 18, value: repCount >= 1 ? 1.0 : 0.0)
        fillPlane(base, plane: planeBase + 19, value: repCount >= 2 ? 1.0 : 0.0)

        // Planes [+20 … +29]: temporal-repetition history (basic30 only).
        // Plane +20+i is all-1 iff bit i of recentRepetitionMask is set
        // (the position i+1 plies ago is a PositionKey duplicate of this
        // frame). Skip-if-zero — the leading clear already zeroed the region.
        if includeTemporalRepetition {
            let recentMask = state.recentRepetitionMask
            if recentMask != 0 {
                for i in 0..<10 where (recentMask >> i) & 1 == 1 {
                    fillPlane(base, plane: planeBase + 20 + i)
                }
            }
        }
    }

    /// Encode a game state into a `tensorLength`-float tensor.
    ///
    /// Allocating variant — delegates to `encode(_:into:)` so both
    /// paths share the same encoding logic. Used by non-hot-path
    /// callers (tests, the Forward Pass demo UI). Hot-path callers
    /// should use `encode(_:into:)` with a pre-allocated scratch.
    static func encode(
        _ current: GameState,
        history: [GameState] = [],
        perspective: PieceColor? = nil,
        encoding: InputEncoding
    ) -> [Float] {
        var tensor = [Float](repeating: 0, count: tensorLength(for: encoding))
        tensor.withUnsafeMutableBufferPointer { buf in
            encode(current, history: history, perspective: perspective, into: buf, encoding: encoding)
        }
        return tensor
    }

    /// Convenience: encode the starting position.
    static func encodeStartingPosition(encoding: InputEncoding) -> [Float] {
        encode(.starting, encoding: encoding)
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
    /// - Parameter buffer: Exactly `tensorLength` floats in the
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
            guard let pieceType = PieceType(rawValue: plane) else {
                preconditionFailure("plane \(plane) is out of PieceType's raw-value range (0..<6)")
            }
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
            guard let pieceType = PieceType(rawValue: plane - 6) else {
                preconditionFailure("plane \(plane) - 6 = \(plane - 6) is out of PieceType's raw-value range (0..<6)")
            }
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
        // not legality. The encode-side writes `min(clock,99)/99.0` from
        // a non-negative Int, so under non-corrupt operation clockProbe
        // is finite and in [0, 1]. An out-of-range or NaN probe means
        // the underlying replay-buffer / Metal staging memory is
        // corrupt — surface it at the boundary rather than silently
        // clamping (and note `Int(Float.nan)` would otherwise trap).
        let clockProbe = buffer[17 * 64]
        precondition(
            clockProbe.isFinite && clockProbe >= 0 && clockProbe <= 1.0001,
            "BoardEncoder.decodeSynthetic: plane-17 clock probe out of expected [0,1] range (got \(clockProbe)); replay-buffer or Metal staging memory is corrupt."
        )
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
        var v = value
        vDSP_vfill(&v, base.advanced(by: plane * 64), 1, 64)
    }
}
