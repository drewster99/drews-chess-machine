import Foundation

/// Hand-built starter set of unambiguous tactical positions for the
/// `[TACTICAL]` probe. Each fixture's "right move" was verified by
/// enumerating legal alternatives during construction; the comments
/// next to each builder summarize that check so a future reviewer
/// can re-walk it without re-deriving.
///
/// No FEN parser. Positions are constructed cell-by-cell with the
/// same `Piece(type:color:)` + `GameState(...)` primitives the
/// existing engine diagnostics use. Castling rights and en-passant
/// are turned off everywhere — these positions are about pure
/// tactics and the network's input planes already encode "no
/// castling, no en-passant" as the all-zero default.
enum TacticalProbeData {

    /// The probes a single click of "Run Tactical Probe" feeds the
    /// network. Keep this list short and high-signal — the goal is
    /// "does the network commit to the obviously-right move," not
    /// coverage of every tactical motif.
    static let standardSet: [TacticalProbe] = [
        Self.makeDefensiveOnlyKingEscape(),
        Self.makeForcedPromotion(),
        Self.makeHangingQueenCapture(),
        Self.makeHangingKnightCapture(),
        Self.makeHangingRookCapture(),
        Self.makeBackRankMate(),
        Self.makeKingQueenMate(),
        Self.makeTwoRookLadderMate(),
        Self.makeAvoidStalemateQueenMate()
    ]

    // MARK: - Square helpers

    /// Algebraic-to-board-index converter. The board is laid out as
    /// `row * 8 + col` where `row 0 = rank 8` (Black's back rank) and
    /// `row 7 = rank 1` (White's back rank), `col 0 = a-file`. So
    /// `"a8"` -> 0, `"h1"` -> 63. Bad input is a programmer error in
    /// fixture data — a precondition crashes loudly rather than
    /// silently producing an off-board move.
    private static func sq(_ alg: String) -> Int {
        let chars = Array(alg.lowercased())
        precondition(
            chars.count == 2,
            "TacticalProbeData.sq: '\(alg)' must be 2 chars"
        )
        guard let fileAscii = chars[0].asciiValue,
              fileAscii >= 0x61, fileAscii <= 0x68,
              let rankAscii = chars[1].asciiValue,
              rankAscii >= 0x31, rankAscii <= 0x38
        else {
            preconditionFailure(
                "TacticalProbeData.sq: '\(alg)' not a valid algebraic square"
            )
        }
        let file = Int(fileAscii - 0x61)     // 'a' -> 0
        let rank = Int(rankAscii - 0x30)     // '1' -> 1
        let row = 8 - rank                   // rank 8 -> row 0
        return row * 8 + file
    }

    /// Build a `ChessMove` from algebraic from/to squares plus an
    /// optional promotion piece.
    private static func mv(
        _ from: String,
        _ to: String,
        promote: PieceType? = nil
    ) -> ChessMove {
        let f = sq(from)
        let t = sq(to)
        return ChessMove(
            fromRow: f / 8,
            fromCol: f % 8,
            toRow: t / 8,
            toCol: t % 8,
            promotion: promote
        )
    }

    /// Place pieces at algebraic squares and return the resulting
    /// `[Piece?]` board. Anything not listed is empty. Both lists
    /// are `(square, type)` pairs. Side to move comes from `toMove`.
    /// All castling rights are turned off — these are post-opening
    /// composed positions, not real-game continuations.
    private static func placement(
        white: [(square: String, type: PieceType)],
        black: [(square: String, type: PieceType)],
        toMove: PieceColor
    ) -> GameState {
        var board: [Piece?] = Array(repeating: nil, count: 64)
        for entry in white {
            board[sq(entry.square)] = Piece(type: entry.type, color: .white)
        }
        for entry in black {
            board[sq(entry.square)] = Piece(type: entry.type, color: .black)
        }
        return GameState(
            board: board,
            currentPlayer: toMove,
            whiteKingsideCastle: false,
            whiteQueensideCastle: false,
            blackKingsideCastle: false,
            blackQueensideCastle: false,
            enPassantSquare: nil,
            halfmoveClock: 0
        )
    }

    // MARK: - Fixtures

    /// K+Q vs K mate-in-1. White Kg6 walls in g7/h7/f7; Qd1 swings
    /// up the d-file to d8 and gives mate along the 8th rank.
    /// Verified unique mate-in-1 over all queen moves: other
    /// checks (Qg1+, Qh5+, Qg4+, Qe1) all let the king to f8.
    private static func makeKingQueenMate() -> TacticalProbe {
        let state = placement(
            white: [("g6", .king), ("d1", .queen)],
            black: [("g8", .king)],
            toMove: .white
        )
        return TacticalProbe(
            name: "K+Q mate, Qd1-d8#",
            shortDescription: "K+Q mate",
            category: .mateInOne,
            state: state,
            acceptable: [mv("d1", "d8")]
        )
    }

    /// Classic back-rank mate. Black Kg8 is hemmed in by its own
    /// pawns f7/g7/h7; white Rook lifts from a1 to a8 along the
    /// 8th rank. White king on g1 (out of the action). Unique
    /// mate among rook moves: Rb1..Rh1 / Ra2..Ra7 all miss.
    private static func makeBackRankMate() -> TacticalProbe {
        let state = placement(
            white: [("g1", .king), ("a1", .rook)],
            black: [
                ("g8", .king),
                ("f7", .pawn),
                ("g7", .pawn),
                ("h7", .pawn)
            ],
            toMove: .white
        )
        return TacticalProbe(
            name: "Back-rank mate, Ra1-a8#",
            shortDescription: "Back-rank mate",
            category: .mateInOne,
            state: state,
            acceptable: [mv("a1", "a8")]
        )
    }

    /// Two-rook ladder mate. White Ra7 cuts off the 7th rank;
    /// Re6 lifts to e8 and delivers mate. Unique among rook moves:
    /// Rh6+/Ra8 both let the king out via g8.
    private static func makeTwoRookLadderMate() -> TacticalProbe {
        let state = placement(
            white: [("h1", .king), ("a7", .rook), ("e6", .rook)],
            black: [("h8", .king)],
            toMove: .white
        )
        return TacticalProbe(
            name: "Two-rook ladder, Re6-e8#",
            shortDescription: "Two-rook ladder",
            category: .mateInOne,
            state: state,
            acceptable: [mv("e6", "e8")]
        )
    }

    /// Hanging Black queen on e5, no defenders nearby. White Nc4
    /// is the only piece that can capture in one move. White king
    /// safely on h1 (queen does not attack the h1-a8 diagonal —
    /// e5 to h1 is row+4 col+3, not in line). Verified: of Nc4's
    /// knight moves (a3/a5/b2/b6/d2/d6/e3/e5), only Nxe5 captures
    /// the queen.
    private static func makeHangingQueenCapture() -> TacticalProbe {
        let state = placement(
            white: [("h1", .king), ("c4", .knight)],
            black: [("h8", .king), ("e5", .queen)],
            toMove: .white
        )
        return TacticalProbe(
            name: "Free queen, Nc4xe5",
            shortDescription: "Free queen",
            category: .hangingPieceCapture,
            state: state,
            acceptable: [mv("c4", "e5")]
        )
    }

    /// Hanging Black knight on c6, no defenders. White Be4 sights
    /// it down the a8-h1 diagonal (e4-d5-c6). Bishop is the only
    /// white piece in capture range; no other bishop diagonal nor
    /// king move captures.
    private static func makeHangingKnightCapture() -> TacticalProbe {
        let state = placement(
            white: [("h1", .king), ("e4", .bishop)],
            black: [("h8", .king), ("c6", .knight)],
            toMove: .white
        )
        return TacticalProbe(
            name: "Free knight, Be4xc6",
            shortDescription: "Free knight",
            category: .hangingPieceCapture,
            state: state,
            acceptable: [mv("e4", "c6")]
        )
    }

    /// Hanging Black rook on a2, no defenders. White Rf2 takes
    /// along the 2nd rank. Only white rook in range; king moves
    /// don't capture.
    private static func makeHangingRookCapture() -> TacticalProbe {
        let state = placement(
            white: [("h1", .king), ("f2", .rook)],
            black: [("h8", .king), ("a2", .rook)],
            toMove: .white
        )
        return TacticalProbe(
            name: "Free rook, Rf2xa2",
            shortDescription: "Free rook",
            category: .hangingPieceCapture,
            state: state,
            acceptable: [mv("f2", "a2")]
        )
    }

    /// Forced promotion. White pawn on b7 has nothing in front,
    /// kings far away. Promote-to-queen is obviously best —
    /// promote-to-rook/bishop/knight all win eventually but are
    /// strictly worse. Strict fixture: only b8=Q is accepted, so a
    /// `correctButFlat` here would mean the network spreads probability
    /// across the four promotion variants instead of committing to
    /// queen.
    private static func makeForcedPromotion() -> TacticalProbe {
        let state = placement(
            white: [("h1", .king), ("b7", .pawn)],
            black: [("h8", .king)],
            toMove: .white
        )
        return TacticalProbe(
            name: "Forced promotion, b7-b8=Q",
            shortDescription: "Forced promotion",
            category: .forcedPromotion,
            state: state,
            acceptable: [mv("b7", "b8", promote: .queen)]
        )
    }

    /// Stalemate-avoidance probe. White Kf6 + Qg3 vs lone Black Kh8.
    /// Two visually-attractive Q moves dominate the candidate list:
    ///   * `Qg7#` — mate (Q defended by Kf6; Black k has no legal
    ///     reply: g7 occupied/defended, g8 attacked diagonal from Q,
    ///     h7 attacked diagonal from Q).
    ///   * `Qg6` — stalemate (Black k's three adjacent squares all
    ///     covered: g7 by Kf6 + Qg6 file g, g8 by Q file g, h7 by Q
    ///     diagonal; not in check, so it's stalemate not mate).
    /// Q does not have a clear path to other mating squares — Qa8+
    /// is the closest but Kf6 doesn't cover h7 so Black k escapes
    /// Kh7. So `Qg7#` is the unique mate; a network that picks
    /// Qg6 traded mate for a half-point. Tests "does the policy
    /// dome look one move ahead past the obvious-looking Q-jump."
    private static func makeAvoidStalemateQueenMate() -> TacticalProbe {
        let state = placement(
            white: [("f6", .king), ("g3", .queen)],
            black: [("h8", .king)],
            toMove: .white
        )
        return TacticalProbe(
            name: "Avoid stalemate, Qg3-g7#",
            shortDescription: "Avoid stalemate (Qg7#)",
            category: .avoidStalemate,
            state: state,
            acceptable: [mv("g3", "g7")]
        )
    }

    /// Defensive must-find probe. White Kh1 is in double-rook
    /// trouble: BR a1 gives check along rank 1; BR g3 covers file
    /// g (including g1 and g2). Black king Kh8 sits out of the
    /// action. White has no defenders, no interpositions, and
    /// cannot capture either rook. The only legal escape from
    /// check is `Kh2` — h2 is not attacked by either rook (Ra1
    /// covers file a + rank 1, Rg3 covers file g + rank 3). Tests
    /// whether the network can find the single forced defensive
    /// reply rather than picking any plausible-looking move.
    private static func makeDefensiveOnlyKingEscape() -> TacticalProbe {
        let state = placement(
            white: [("h1", .king)],
            black: [("h8", .king), ("a1", .rook), ("g3", .rook)],
            toMove: .white
        )
        return TacticalProbe(
            name: "Defensive only-move, Kh1-h2",
            shortDescription: "Defensive only-move (Kh2)",
            category: .defensiveMustFind,
            state: state,
            acceptable: [mv("h1", "h2")]
        )
    }
}
