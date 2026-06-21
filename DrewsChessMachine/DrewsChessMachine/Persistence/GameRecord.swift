import Foundation

/// The objective outcome of a recorded game, from White's point of view.
///
/// The per-position value target (current-player perspective) is derived at
/// replay time when the game is re-encoded, so the corpus stores only this
/// single side-independent result rather than a per-ply signed outcome.
enum GameOutcome: UInt8, Sendable, Equatable {
    case whiteWin = 0
    case draw = 1
    case blackWin = 2
}

/// Why a recorded game ended. Optional provenance only — training reads
/// `GameOutcome`, never this. Imported PGN games may not carry a reason.
enum GameTerminationReason: UInt8, Sendable, Equatable {
    case checkmate = 0
    case stalemate = 1
    case fiftyMoveRule = 2
    case insufficientMaterial = 3
    case threefoldRepetition = 4
    case maxPlies = 5
    case resignation = 6
    case timeout = 7
    case unknown = 255
}

/// One recorded game: an intrinsic, architecture-independent description of a
/// single game as a start position plus the moves played and the result.
///
/// Deliberately holds no encoded tensors and no policy indices — a game is
/// replayed through `ChessGameEngine` + the target architecture's
/// `BoardEncoder` at training time, so the same corpus feeds any network
/// shape. `sourceID` is a shard-level property (every game in a shard shares
/// one ingestion source), so it is not stored per record.
struct GameRecord: Sendable, Equatable {
    /// Standard start position when nil; a FEN only for imported setups.
    var startFEN: String?
    /// The moves played, in game order.
    var moves: [ChessMove]
    /// Objective result from White's point of view.
    var outcome: GameOutcome
    /// Optional provenance: how the game ended.
    var terminationReason: GameTerminationReason?

    init(startFEN: String? = nil,
         moves: [ChessMove],
         outcome: GameOutcome,
         terminationReason: GameTerminationReason? = nil) {
        self.startFEN = startFEN
        self.moves = moves
        self.outcome = outcome
        self.terminationReason = terminationReason
    }
}

extension GameRecord {
    /// Build a record from a game that ended via the chess rules.
    ///
    /// Self-play games that hit the max-plies cap are discarded *before* they
    /// reach recording (the draw-keep / cap filters run at game end), so only
    /// natural `GameResult` terminations are mapped here.
    init(moves: [ChessMove], result: GameResult, startFEN: String? = nil) {
        let outcome: GameOutcome
        let reason: GameTerminationReason
        switch result {
        case .checkmate(let winner):
            outcome = (winner == .white) ? .whiteWin : .blackWin
            reason = .checkmate
        case .stalemate:
            outcome = .draw
            reason = .stalemate
        case .drawByFiftyMoveRule:
            outcome = .draw
            reason = .fiftyMoveRule
        case .drawByInsufficientMaterial:
            outcome = .draw
            reason = .insufficientMaterial
        case .drawByThreefoldRepetition:
            outcome = .draw
            reason = .threefoldRepetition
        }
        self.init(startFEN: startFEN, moves: moves, outcome: outcome, terminationReason: reason)
    }
}

/// Bijective 16-bit packing of a `ChessMove` for the corpus move stream.
///
/// Layout (low → high bits): `to`(6) | `from`(6) | `promo`(3), top bit unused.
/// `from`/`to` are square indices `row*8+col` (0–63). Castling and en passant
/// carry no extra bits — they are inferred from board state when the move list
/// is replayed. Promotion codes: 0 none, 1 knight, 2 bishop, 3 rook, 4 queen
/// (the only legal promotion targets). The mapping is written as an explicit
/// switch rather than via `PieceType.rawValue` so a future reordering of
/// `PieceType` cannot silently change the on-disk format.
enum PackedMove {
    static func pack(_ move: ChessMove) -> UInt16 {
        let fromSq = UInt16(move.fromRow * 8 + move.fromCol)
        let toSq = UInt16(move.toRow * 8 + move.toCol)
        let promo: UInt16
        switch move.promotion {
        case nil:            promo = 0
        case .knight?:       promo = 1
        case .bishop?:       promo = 2
        case .rook?:         promo = 3
        case .queen?:        promo = 4
        case .pawn?, .king?: promo = 0
        }
        return (promo << 12) | (fromSq << 6) | toSq
    }

    static func unpack(_ packed: UInt16) throws -> ChessMove {
        let toSq = Int(packed & 0x3F)
        let fromSq = Int((packed >> 6) & 0x3F)
        let promoCode = Int((packed >> 12) & 0x7)
        let promotion: PieceType?
        switch promoCode {
        case 0: promotion = nil
        case 1: promotion = .knight
        case 2: promotion = .bishop
        case 3: promotion = .rook
        case 4: promotion = .queen
        default: throw GameCorpusError.corruptMove(promoCode: promoCode)
        }
        return ChessMove(fromRow: fromSq / 8,
                         fromCol: fromSq % 8,
                         toRow: toSq / 8,
                         toCol: toSq % 8,
                         promotion: promotion)
    }
}
