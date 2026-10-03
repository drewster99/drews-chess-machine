import Foundation

// MARK: - Game Result

enum GameResult: Sendable, Equatable {
    case checkmate(winner: PieceColor)
    case stalemate
    case drawByFiftyMoveRule
    case drawByInsufficientMaterial
    case drawByThreefoldRepetition
}

/// Raw game result used at game-result boundaries — wraps an ordinary
/// `GameResult` for games that ended via the chess rules and adds the
/// `terminatedEarly` case for self-play games that hit the configured
/// `selfPlayMaxPliesPerGame` cap before terminating naturally.
///
/// `ChessMachine.beginNewGame` only ever returns `.terminatedNormally`
/// — the human-play paths it drives (PlayController, Play Game, Play
/// Continuous) should run to a real chess termination. The
/// `.terminatedEarly` case is produced by the self-play stats path
/// (`ParallelWorkerStatsBox.recordDroppedGame`) for its rolling-window
/// ring, so `snapshot()` can tally cap-dropped games alongside W/D/L
/// counts. Self-play and arena themselves call `ChessGameEngine`
/// directly, not `ChessMachine`.
enum RawGameResult: Sendable {
    /// Self-play game stopped because `moveHistory.count` reached the
    /// configured max-plies cap before the chess engine declared a
    /// natural termination. Carries no payload — the game is discarded.
    case terminatedEarly
    /// Game ended via the chess rules (checkmate / stalemate / one of
    /// the four draw types). Payload is the rule that fired.
    case terminatedNormally(GameResult)
}

/// A draw condition the engine's own rules detect in the current position.
/// Under `.automatic` adjudication the first one found ends the game; under
/// `.serverAuthoritative` it is only reported (via
/// `ChessGameEngine.drawCondition`) so a caller can compare its view with an
/// external authority's.
enum ChessDrawCondition: String, Sendable, Equatable, Codable {
    case fiftyMoveRule
    case threefoldRepetition
    case insufficientMaterial

    var gameResult: GameResult {
        switch self {
        case .fiftyMoveRule: return .drawByFiftyMoveRule
        case .threefoldRepetition: return .drawByThreefoldRepetition
        case .insufficientMaterial: return .drawByInsufficientMaterial
        }
    }
}

/// Who decides that a game has ended in a draw.
enum ChessGameAdjudication: Sendable, Equatable {
    /// The engine ends the game on every termination it detects: checkmate,
    /// stalemate, the fifty-move rule, threefold repetition and insufficient
    /// material. Self-play, arena and human play all use this.
    case automatic
    /// An external authority decides draws: the Lichess server, the chess GUI
    /// driving `--uci`, or — in corpus replay — the recorded game itself,
    /// whose players may have played on past an unclaimed draw. The engine
    /// ends the game only when the side to move has no legal move
    /// (checkmate or stalemate), because then there is literally no move to
    /// make. Draw conditions never end the game locally, so the engine can
    /// never refuse a move while the authority still considers the game live
    /// — which matters wherever the two rule sets' definitions differ (the
    /// en-passant component of `PositionKey`, insufficient-material cases).
    case serverAuthoritative
}

// MARK: - Position Key (for repetition detection)

/// A hashable identifier for a chess position. Two positions match for the
/// purposes of FIDE Article 9.2 (threefold repetition) when piece placement,
/// side to move, all four castling rights, and en passant target are equal.
///
/// Note: this implementation includes the en passant square verbatim. The
/// strict FIDE rule says the ep target only differentiates positions when
/// a capture is actually playable; we err on the conservative side and
/// treat any ep target difference as different positions, which can miss
/// a tiny number of legitimate threefold draws but never declares a wrong
/// one.
struct PositionKey: Hashable {
    let board: [Piece?]
    let currentPlayer: PieceColor
    let whiteKingsideCastle: Bool
    let whiteQueensideCastle: Bool
    let blackKingsideCastle: Bool
    let blackQueensideCastle: Bool
    let enPassantSquareIndex: Int?

    init(from state: GameState) {
        self.board = state.board
        self.currentPlayer = state.currentPlayer
        self.whiteKingsideCastle = state.whiteKingsideCastle
        self.whiteQueensideCastle = state.whiteQueensideCastle
        self.blackKingsideCastle = state.blackKingsideCastle
        self.blackQueensideCastle = state.blackQueensideCastle
        if let ep = state.enPassantSquare {
            self.enPassantSquareIndex = ep.row * 8 + ep.col
        } else {
            self.enPassantSquareIndex = nil
        }
    }
}

// MARK: - Errors

enum ChessGameError: LocalizedError {
    case gameAlreadyOver
    case illegalMove(ChessMove)

    var errorDescription: String? {
        switch self {
        case .gameAlreadyOver:
            return "The game has already ended"
        case .illegalMove(let move):
            return "Illegal move for the current position: \(move)"
        }
    }
}

// MARK: - Chess Game Engine

/// Manages a chess game between two players: applies moves, updates state,
/// detects game-ending conditions (checkmate, stalemate, draws).
///
/// The engine owns the authoritative legal-move list for the current
/// side-to-move (`currentLegalMoves`), computed once at init and refreshed
/// inside `applyMoveAndAdvance` after each move. Every incoming move is
/// validated against that list before apply — callers can hand moves in
/// from any source (self-play player, arena player, UI, tests, file load)
/// and the engine rejects anything illegal with
/// `ChessGameError.illegalMove` instead of trusting the caller and
/// potentially trapping inside `MoveGenerator.applyMove`'s
/// `preconditionFailure`.
/// `MoveGenerator.legalMoves(for:)` still runs exactly once per ply — the
/// list the engine produces after applying move N is reused both for
/// end-of-game detection and as the guard for move N+1.
final class ChessGameEngine {
    private(set) var state: GameState
    private(set) var result: GameResult?
    private(set) var moveHistory: [ChessMove] = []

    /// Who decides draws; see `ChessGameAdjudication`. Fixed for the life of
    /// the engine.
    let adjudication: ChessGameAdjudication

    /// Legal moves for `state`'s side-to-move. Refreshed inside
    /// `applyMoveAndAdvance`; callers can read this instead of calling
    /// `MoveGenerator.legalMoves(for: engine.state)` themselves.
    private(set) var currentLegalMoves: [ChessMove]

    /// Tally of how many times each position has appeared since the last
    /// irreversible move. Cleared whenever halfmoveClock resets to 0
    /// (pawn moves and captures), since no prior position can recur after
    /// an irreversible move.
    private var positionCounts: [PositionKey: Int] = [:]

    /// Occurrences of the current position since the last irreversible move,
    /// including this one — `positionCounts` for the current key, cached so
    /// `drawCondition` needs no second hash of the board per ply.
    private var currentPositionVisits: Int = 1

    /// Ordered window of up to the `recentPositionKeyWindow` most recent
    /// prior positions, with index 0 = position 1 ply ago, index
    /// `recentPositionKeyWindow - 1` = oldest. Drives
    /// `GameState.recentRepetitionMask` and `BoardEncoder` planes
    /// 20–29. Cleared in the same conditions as `positionCounts`
    /// (halfmoveClock = 0 after an irreversible move) — entries from
    /// before such a move can never match a future position.
    private var recentPositionKeys: [PositionKey] = []

    /// Hard cap on `recentPositionKeys` length, matching the 10 binary
    /// repetition-history planes (20–29) the encoder fills. Exposed as a
    /// static so tests can read the same constant the implementation does.
    static let recentPositionKeyWindow: Int = 10

    /// Ordered window of up to `recentStateWindow` most recent *prior* full
    /// game states, index 0 = the position 1 ply ago, index
    /// `recentStateWindow - 1` = the oldest retained. Each stored `GameState`
    /// already carries the `repetitionCount`/`recentRepetitionMask` it had at
    /// the time, so a history frame reflects the board exactly as it stood.
    ///
    /// Unlike `recentPositionKeys`, this is **never cleared on irreversible
    /// moves**: a history-stacking input encoding must show the true prior
    /// board across pawn moves and captures, not a window reset to empty.
    /// Feeds `BoardEncoder`'s history-aware encode path for the
    /// `full10ply200` encoding (current frame + 9 prior frames).
    private(set) var recentStates: [GameState] = []

    /// Hard cap on `recentStates` length. Sized to the deepest history-
    /// stacking `InputEncoding` — `full10ply200` needs the current position
    /// plus 9 prior frames. Deliberately distinct from
    /// `recentPositionKeyWindow` (different size *and* different clearing
    /// semantics: this window does not clear on irreversible moves).
    static let recentStateWindow: Int = 9

    init(state: GameState = .starting, adjudication: ChessGameAdjudication = .automatic) {
        // The starting position has occurred zero times before — fold that
        // into the state itself so encoders downstream see a consistent
        // `repetitionCount` and `recentRepetitionMask`. Every state
        // subsequently produced by applyMoveAndAdvance also carries both.
        let seeded = state.withRepetitionCount(0).withRecentRepetitionMask(0)
        self.state = seeded
        self.adjudication = adjudication
        self.currentLegalMoves = MoveGenerator.legalMoves(for: seeded)
        positionCounts[PositionKey(from: state)] = 1
    }

    /// Apply a move and recompute the result given the legal moves available
    /// in the *new* position.
    ///
    /// Throws `ChessGameError.gameAlreadyOver` if a result has already been
    /// latched, or `ChessGameError.illegalMove` if `move` is not in
    /// `currentLegalMoves`. On success, `state`, `currentLegalMoves`, and
    /// (if game-ending) `result` are all updated before return.
    ///
    /// Returns the legal moves for the next position so the caller can hand
    /// them straight back to the next player; the same value is also
    /// available via the `currentLegalMoves` property.
    @discardableResult
    func applyMoveAndAdvance(_ move: ChessMove) throws -> [ChessMove] {
        guard result == nil else {
            throw ChessGameError.gameAlreadyOver
        }
        guard currentLegalMoves.contains(move) else {
            throw ChessGameError.illegalMove(move)
        }

        // Snapshot the pre-apply position's key *before* mutating state.
        // After the move lands, this becomes "1 ply ago" relative to the
        // new state and goes at the front of `recentPositionKeys` (unless
        // the move is irreversible, in which case the window is cleared
        // and the snapshot is discarded — see below).
        let priorKey = PositionKey(from: state)

        // Push the pre-apply state into the non-clearing history window for
        // history-stacking encodings. Done unconditionally and *outside* the
        // irreversible-move branch below: a history frame must show the board
        // as it actually stood, even across a pawn move or capture. The
        // stored state already carries its as-of-then repetition fields.
        recentStates.insert(state, at: 0)
        if recentStates.count > Self.recentStateWindow {
            recentStates.removeLast(recentStates.count - Self.recentStateWindow)
        }

        let appliedState = MoveGenerator.applyMove(move, to: state)
        moveHistory.append(move)

        // Pawn moves and captures reset halfmoveClock to 0 and make all
        // prior positions unreachable. Drop both the position-counts table
        // and the ordered window so neither grows unbounded across long
        // games and so prior positions can never accidentally match.
        // `priorKey` is intentionally NOT prepended in this branch — it
        // came from before the irreversible move and is no longer
        // reachable from any future state.
        if appliedState.halfmoveClock == 0 {
            positionCounts.removeAll(keepingCapacity: true)
            recentPositionKeys.removeAll(keepingCapacity: true)
        } else {
            // Prepend the pre-apply key and truncate to the window size.
            // After this, recentPositionKeys[i] == PositionKey of the
            // position (i + 1) plies before appliedState.
            recentPositionKeys.insert(priorKey, at: 0)
            if recentPositionKeys.count > Self.recentPositionKeyWindow {
                recentPositionKeys.removeLast(
                    recentPositionKeys.count - Self.recentPositionKeyWindow
                )
            }
        }
        let key = PositionKey(from: appliedState)
        let totalVisits = (positionCounts[key] ?? 0) + 1
        positionCounts[key] = totalVisits
        currentPositionVisits = totalVisits

        // Compute the temporal-repetition mask (one bit per windowed
        // prior position): bit `i` is set iff `recentPositionKeys[i] == key`.
        // After an irreversible move the window is empty so the mask is
        // necessarily 0 — matching the chess-rules fact that no prior
        // position can equal the new one across a pawn move or capture.
        var mask: UInt16 = 0
        for i in 0..<recentPositionKeys.count {
            if recentPositionKeys[i] == key {
                mask |= UInt16(1) << i
            }
        }

        // `repetitionCount` is occurrences *before* the current visit,
        // saturated at 2 (the encoder's plane representation only
        // distinguishes 0 / ≥1 / ≥2). Layer it and the temporal mask
        // onto the state so the encoder reads consistent values via
        // `state.repetitionCount` and `state.recentRepetitionMask`.
        let priorOccurrences = min(totalVisits - 1, 2)
        state = appliedState
            .withRepetitionCount(priorOccurrences)
            .withRecentRepetitionMask(mask)

        let nextMoves = MoveGenerator.legalMoves(for: state)
        currentLegalMoves = nextMoves
        updateResult(nextLegalMoves: nextMoves)
        return nextMoves
    }

    // MARK: - Game End Detection

    /// The draw condition the engine's own rules detect in the current
    /// position, checked in the order `.automatic` adjudication applies them
    /// (fifty-move rule, then threefold repetition, then insufficient
    /// material). Nil when the side to move has no legal move: checkmate and
    /// stalemate take precedence over every draw rule, so a mating move that
    /// also completes the fifty-move count is a checkmate, not a draw.
    ///
    /// Reported under either adjudication mode. Under `.automatic`, a non-nil
    /// value after `applyMoveAndAdvance` means that move ended the game in
    /// this draw. A position handed to `init` is never adjudicated, so it can
    /// report a draw condition while `result` stays nil.
    var drawCondition: ChessDrawCondition? {
        guard !currentLegalMoves.isEmpty else { return nil }
        if state.halfmoveClock >= 100 {
            return .fiftyMoveRule
        }
        if currentPositionVisits >= 3 {
            return .threefoldRepetition
        }
        if isInsufficientMaterial() {
            return .insufficientMaterial
        }
        return nil
    }

    private func updateResult(nextLegalMoves: [ChessMove]) {
        if nextLegalMoves.isEmpty {
            if MoveGenerator.isInCheck(state, color: state.currentPlayer) {
                result = .checkmate(winner: state.currentPlayer.opposite)
            } else {
                result = .stalemate
            }
            return
        }
        switch adjudication {
        case .automatic:
            if let draw = drawCondition {
                result = draw.gameResult
            }
        case .serverAuthoritative:
            break
        }
    }

    /// Detects positions where neither side can checkmate (FIDE Article 5.2.2):
    /// K vs K, K+B vs K, K+N vs K, and K+B(s) vs K+B(s) when all bishops sit
    /// on a single color complex.
    private func isInsufficientMaterial() -> Bool {
        var whiteKnights = 0
        var blackKnights = 0
        var whiteBishops = 0
        var blackBishops = 0
        // Track square colors of all bishops on the board (0 = light, 1 = dark).
        var bishopSquareColors: Set<Int> = []

        for row in 0..<8 {
            let rowBase = row * 8
            for col in 0..<8 {
                guard let piece = state.board[rowBase + col] else { continue }
                switch piece.type {
                case .pawn, .rook, .queen:
                    // Any of these on either side → mate is possible.
                    return false
                case .king:
                    continue
                case .knight:
                    if piece.color == .white { whiteKnights += 1 } else { blackKnights += 1 }
                case .bishop:
                    if piece.color == .white { whiteBishops += 1 } else { blackBishops += 1 }
                    bishopSquareColors.insert((row + col) & 1)
                }
            }
        }

        let whiteMinors = whiteKnights + whiteBishops
        let blackMinors = blackKnights + blackBishops

        // K vs K
        if whiteMinors == 0 && blackMinors == 0 { return true }
        // K+minor vs K (single bishop or single knight)
        if whiteMinors == 1 && blackMinors == 0 { return true }
        if whiteMinors == 0 && blackMinors == 1 { return true }

        // K+bishop(s) vs K+bishop(s) — drawn iff every bishop on the board
        // (both colors) is on the same color complex. No knights allowed:
        // K+N+N vs K is technically winnable with cooperation and FIDE does
        // not treat it as a forced draw, and K+B vs K+N can mate.
        if whiteKnights == 0 && blackKnights == 0 && bishopSquareColors.count == 1 {
            return true
        }

        return false
    }
}
