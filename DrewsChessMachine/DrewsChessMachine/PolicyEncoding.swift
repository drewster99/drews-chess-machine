import Foundation

// MARK: - Policy Encoding (AlphaZero-shape, 76-channel)
//
// The network's policy head emits raw logits laid out as
// `[batch, policyChannels, 8, 8]`, flattened in NCHW row-major order to
// `[batch, policySize]` where `policySize = policyChannels * 64 = 4864`.
// Each output cell at `(channel, row, col)` is the logit for "move of
// type `channel` from square `(row, col)`" in the current player's
// encoder frame (board flipped vertically when black is to move).
//
// `PolicyEncoding` is the bijection between Swift `ChessMove` values
// and these (channel, row, col) triples. It is the single source of
// truth for the encoding — every site that needs to convert between
// moves and policy indices must call into here, both at training-time
// (one-hot target construction) and at inference-time (per-move logit
// lookup during sampling).
//
// ## Channel layout (locked spec — do not reorder)
//
// Channels 0–55: queen-style moves (8 directions × 7 distances).
//   Direction order, clockwise from "up": N, NE, E, SE, S, SW, W, NW.
//   `channel = direction_index * 7 + (distance - 1)` for
//   `direction_index ∈ 0..<8`, `distance ∈ 1..7`.
//   Covers all sliding-piece moves (R/B/Q), king's normal moves,
//   pawn pushes, pawn diagonal captures, and queen-promotions
//   ARE NOT encoded here — promotions get dedicated channels (see
//   below). 1-step pawn moves into the last rank that have NO
//   promotion field set are also not represented here; the move
//   generator always sets the promotion field for last-rank-bound
//   pawn moves, so no such "promotion-less" moves exist in practice.
//
// Channels 56–63: knight moves (the 8 L-shapes, clockwise from
//   "up-right knight jump"). `channel = 56 + jump_index`.
//
// Channels 64–72: underpromotions (3 pieces × 3 directions).
//   Pieces: knight=0, rook=1, bishop=2.
//   Directions: forward=0, capture-left=1, capture-right=2.
//   `channel = 64 + piece_index * 3 + direction_index`.
//
// Channels 73–75: queen-promotions (3 directions; same direction
//   encoding as underpromotions). `channel = 73 + direction_index`.
//
// Total: 56 + 8 + 9 + 3 = 76 channels × 64 squares = 4864 logits.
//
// ## Encoder frame
//
// Encoding/decoding happens in the encoder's vertically-flipped frame
// (matching `BoardEncoder.encode`'s perspective flip when black is to
// move). The current player is always at the "bottom" (rows 6-7) and
// "forward" is always toward row 0. This means promotions always have
// `from_row=1, to_row=0` in encoder frame, and the "left" / "right"
// capture directions are always relative to the current player's POV.
// The mapping back to absolute board coordinates happens in `decode`,
// which un-flips for black before returning a `ChessMove`.
enum PolicyEncoding {

    // MARK: - Channel-block boundaries (derived from the spec)

    /// Total number of policy channels. Mirrors `ChessNetwork.policyChannels`.
    static let channelCount = 76

    /// First underpromotion channel (knight forward). Channels 64–72.
    private static let underpromotionBase = 64

    /// First queen-promotion channel. Channels 73–75.
    private static let queenPromotionBase = 73

    /// First knight-move channel. Channels 56–63.
    private static let knightBase = 56

    // MARK: - Direction tables

    /// Queen-style directions in encoder frame, indexed 0..7.
    /// Order: N, NE, E, SE, S, SW, W, NW (clockwise from "up").
    /// `(dr, dc)` gives the per-step displacement.
    private static let queenDirections: [(dr: Int, dc: Int)] = [
        (-1,  0), // 0: N
        (-1,  1), // 1: NE
        ( 0,  1), // 2: E
        ( 1,  1), // 3: SE
        ( 1,  0), // 4: S
        ( 1, -1), // 5: SW
        ( 0, -1), // 6: W
        (-1, -1)  // 7: NW
    ]

    /// Knight jumps in encoder frame, indexed 0..7. Order: clockwise
    /// from "up-right knight jump" (-2, +1). Each entry is a (dr, dc)
    /// displacement for a single L-shape.
    private static let knightJumps: [(dr: Int, dc: Int)] = [
        (-2,  1), // 0: up-right
        (-1,  2), // 1: right-up
        ( 1,  2), // 2: right-down
        ( 2,  1), // 3: down-right
        ( 2, -1), // 4: down-left
        ( 1, -2), // 5: left-down
        (-1, -2), // 6: left-up
        (-2, -1)  // 7: up-left
    ]

    // MARK: - Public API

    /// Encode a legal `ChessMove` into `(channel, row, col)` in the
    /// current player's encoder frame.
    ///
    /// Asserts on moves that don't correspond to any of the 76 channel
    /// classes — moves drawn from `MoveGenerator.legalMoves(for:)`
    /// always do, so this should never fire in practice.
    static func encode(_ move: ChessMove, currentPlayer: PieceColor) -> (channel: Int, row: Int, col: Int) {
        let flip = currentPlayer == .black
        let fromRow = flip ? (7 - move.fromRow) : move.fromRow
        let toRow   = flip ? (7 - move.toRow)   : move.toRow
        let fromCol = move.fromCol
        let toCol   = move.toCol
        let dr = toRow - fromRow
        let dc = toCol - fromCol

        // 1) Promotions (Q/R/B/N) get dedicated channels regardless of
        //    geometric direction. Direction within the promotion blocks
        //    is forward (dc=0), capture-left (dc=-1), capture-right
        //    (dc=+1). In encoder frame, "forward" for the current
        //    player is always dr = -1 (toward row 0). Verify both axes
        //    so a malformed promotion move (wrong rank delta) fails
        //    here at encode time rather than producing a junk index
        //    that would mis-train and never round-trip.
        if let promo = move.promotion {
            precondition(dr == -1,
                "PolicyEncoding.encode: promotion must have dr=-1 in encoder frame (got \(dr)) for \(move.notation)"
            )
            let directionIndex: Int
            switch dc {
            case  0: directionIndex = 0  // forward
            case -1: directionIndex = 1  // capture-left
            case  1: directionIndex = 2  // capture-right
            default:
                preconditionFailure(
                    "PolicyEncoding.encode: promotion with non-promotion-direction dc=\(dc)"
                )
            }
            let channel: Int
            switch promo {
            case .queen:
                channel = queenPromotionBase + directionIndex
            case .knight:
                channel = underpromotionBase + 0 * 3 + directionIndex
            case .rook:
                channel = underpromotionBase + 1 * 3 + directionIndex
            case .bishop:
                channel = underpromotionBase + 2 * 3 + directionIndex
            case .pawn, .king:
                preconditionFailure(
                    "PolicyEncoding.encode: invalid promotion piece \(promo)"
                )
            }
            return (channel: channel, row: fromRow, col: fromCol)
        }

        // 2) Knight moves — exactly one of the 8 L-shapes.
        if let jumpIndex = knightJumps.firstIndex(where: { $0.dr == dr && $0.dc == dc }) {
            return (channel: knightBase + jumpIndex, row: fromRow, col: fromCol)
        }

        // 3) Queen-style: pure direction × distance. Distance = max(|dr|, |dc|).
        //    For a true queen-style move, dr and dc must each be either
        //    0 or ±distance, with at least one nonzero.
        let absR = abs(dr)
        let absC = abs(dc)
        let distance = max(absR, absC)
        precondition(distance >= 1 && distance <= 7,
            "PolicyEncoding.encode: invalid distance \(distance) for non-knight move \(move.notation)")
        precondition(
            (absR == 0 || absR == distance) && (absC == 0 || absC == distance),
            "PolicyEncoding.encode: non-orthogonal/non-diagonal displacement (\(dr), \(dc)) for \(move.notation)"
        )
        let stepDr = dr == 0 ? 0 : dr / absR
        let stepDc = dc == 0 ? 0 : dc / absC
        guard let dirIndex = queenDirections.firstIndex(where: { $0.dr == stepDr && $0.dc == stepDc }) else {
            preconditionFailure(
                "PolicyEncoding.encode: unrecognized queen-style direction (\(stepDr), \(stepDc)) for \(move.notation)"
            )
        }
        let channel = dirIndex * 7 + (distance - 1)
        return (channel: channel, row: fromRow, col: fromCol)
    }

    /// Flat policy index: `channel * 64 + row * 8 + col`. Matches NCHW
    /// row-major flatten of the network's `[batch, 76, 8, 8]` policy
    /// output. Index range: `0..<policySize` (= 4864).
    static func policyIndex(_ move: ChessMove, currentPlayer: PieceColor) -> Int {
        let (chan, r, c) = encode(move, currentPlayer: currentPlayer)
        return chan * 64 + r * 8 + c
    }

    /// Decode a `(channel, fromRow, fromCol)` triple to a `ChessMove`
    /// based purely on geometry — does **not** check whether the move
    /// is legal in any specific board state. Returns nil only when:
    ///   - the (channel, row, col) triple itself is out of range; or
    ///   - the destination square computed from the channel's geometry
    ///     would be off-board.
    ///
    /// Used by the Forward Pass / Candidate Test diagnostic to display
    /// top-K raw policy cells regardless of legality — the diagnostic
    /// "is the network learning what's a valid move?" depends on
    /// seeing illegal-but-geometrically-valid candidates surface in the
    /// top-K. For decision-making (e.g., the self-play sampler), callers
    /// should use `decode(channel:row:col:state:)` which adds the
    /// legality filter on top of this geometric decode.
    static func geometricDecode(
        channel: Int,
        row: Int,
        col: Int,
        currentPlayer: PieceColor
    ) -> ChessMove? {
        guard channel >= 0, channel < channelCount else { return nil }
        guard row >= 0, row < 8, col >= 0, col < 8 else { return nil }

        let flip = currentPlayer == .black
        let promotion: PieceType?
        let toRow_enc: Int
        let toCol_enc: Int

        if channel >= queenPromotionBase {
            // Channels 73–75: queen-promotion. Forward / capture-left / capture-right.
            let directionIndex = channel - queenPromotionBase
            let (dr, dc) = promotionDirection(directionIndex)
            toRow_enc = row + dr
            toCol_enc = col + dc
            promotion = .queen
        } else if channel >= underpromotionBase {
            // Channels 64–72: underpromotion (N/R/B × forward/cap-left/cap-right).
            let offset = channel - underpromotionBase
            let pieceIndex = offset / 3
            let directionIndex = offset % 3
            let (dr, dc) = promotionDirection(directionIndex)
            toRow_enc = row + dr
            toCol_enc = col + dc
            switch pieceIndex {
            case 0: promotion = .knight
            case 1: promotion = .rook
            case 2: promotion = .bishop
            default: return nil
            }
        } else if channel >= knightBase {
            // Channels 56–63: knight jumps.
            let jumpIndex = channel - knightBase
            let jump = knightJumps[jumpIndex]
            toRow_enc = row + jump.dr
            toCol_enc = col + jump.dc
            promotion = nil
        } else {
            // Channels 0–55: queen-style (direction × distance).
            let dirIndex = channel / 7
            let distance = (channel % 7) + 1
            let dir = queenDirections[dirIndex]
            toRow_enc = row + dir.dr * distance
            toCol_enc = col + dir.dc * distance
            promotion = nil
        }

        // Off-board guard.
        guard toRow_enc >= 0, toRow_enc < 8, toCol_enc >= 0, toCol_enc < 8 else {
            return nil
        }

        // Un-flip rows back to absolute board coordinates if black to move.
        let fromRowAbs = flip ? (7 - row) : row
        let toRowAbs = flip ? (7 - toRow_enc) : toRow_enc
        return ChessMove(
            fromRow: fromRowAbs,
            fromCol: col,
            toRow: toRowAbs,
            toCol: toCol_enc,
            promotion: promotion
        )
    }

    /// Decode a `(channel, fromRow, fromCol)` triple back to the
    /// matching legal `ChessMove` in absolute board coordinates.
    /// Returns nil for any of:
    ///   - invalid `(channel, row, col)` triple (out of range);
    ///   - destination square would be off-board;
    ///   - the resulting move is not actually legal in `state`.
    ///
    /// Off-board guards (in `geometricDecode`) run BEFORE the
    /// legal-moves filter so callers scanning every cell of the policy
    /// output get clean nils for the many cells that don't correspond
    /// to anything realizable.
    static func decode(channel: Int, row: Int, col: Int, state: GameState) -> ChessMove? {
        guard let candidate = geometricDecode(
            channel: channel,
            row: row,
            col: col,
            currentPlayer: state.currentPlayer
        ) else { return nil }

        // Final disambiguation: only return the move if it's actually
        // legal in `state`. Catches channel-shape moves that happen to
        // be illegal here (e.g., underpromotion channel from a square
        // with no pawn, or queenside-castle channel after castling
        // rights are gone).
        let legalMoves = MoveGenerator.legalMoves(for: state)
        return legalMoves.contains(candidate) ? candidate : nil
    }

    // MARK: - Helpers

    /// `(dr, dc)` for promotion direction in encoder frame:
    /// 0 = forward (no capture), 1 = capture-left, 2 = capture-right.
    /// Forward is always dr = -1 (toward row 0) since promotions only
    /// happen from row 1 to row 0 in encoder frame.
    @inline(__always)
    private static func promotionDirection(_ index: Int) -> (dr: Int, dc: Int) {
        switch index {
        case 0: return (-1,  0)  // forward
        case 1: return (-1, -1)  // capture-left
        case 2: return (-1,  1)  // capture-right
        default:
            preconditionFailure("PolicyEncoding.promotionDirection: invalid direction index \(index)")
        }
    }
}
