//
//  PolicyEncodingTests.swift
//  DrewsChessMachineTests
//
//  Round-trip and edge-case tests for the AlphaZero-shape 76-channel
//  policy encoding bijection. The encoding lives at the boundary
//  between the network's flat 4864-cell policy output and Swift's
//  ChessMove value type — a bug here would silently mistrain policy
//  targets and mis-sample moves at inference. Tests focus on:
//   - round-trip: encode(move) → decode → original move
//   - bijection: distinct legal moves never share an index
//   - underpromotion: all 4 promotion variants produce distinct
//     indices and round-trip to the right piece type
//   - decode rejects off-board cells with nil
//   - decode rejects illegal-here cells with nil
//   - both colors handled correctly (encoder-frame perspective flip)
//

import XCTest
@testable import DrewsChessMachine

final class PolicyEncodingTests: XCTestCase {

    // MARK: - Round-trip across every legal move from many positions

    func testRoundTripStartingPosition() {
        assertRoundTripsAllLegalMoves(in: .starting)
    }

    func testRoundTripAfter1e4() {
        // White plays 1.e4 → black to move position.
        let after1e4 = MoveGenerator.applyMove(
            ChessMove(fromRow: 6, fromCol: 4, toRow: 4, toCol: 4, promotion: nil),
            to: .starting
        )
        assertRoundTripsAllLegalMoves(in: after1e4)
    }

    func testRoundTripCastlingPositions() {
        // Set up a position where both colors can castle either side.
        // Clear pieces between king and rook for both colors.
        var board: [Piece?] = Array(repeating: nil, count: 64)
        // Black back rank
        board[0 * 8 + 0] = Piece(type: .rook, color: .black)
        board[0 * 8 + 4] = Piece(type: .king, color: .black)
        board[0 * 8 + 7] = Piece(type: .rook, color: .black)
        // White back rank
        board[7 * 8 + 0] = Piece(type: .rook, color: .white)
        board[7 * 8 + 4] = Piece(type: .king, color: .white)
        board[7 * 8 + 7] = Piece(type: .rook, color: .white)

        for player in [PieceColor.white, .black] {
            let state = GameState(
                board: board,
                currentPlayer: player,
                whiteKingsideCastle: true,
                whiteQueensideCastle: true,
                blackKingsideCastle: true,
                blackQueensideCastle: true,
                enPassantSquare: nil,
                halfmoveClock: 0
            )
            assertRoundTripsAllLegalMoves(in: state)
        }
    }

    func testRoundTripAllPromotionsOnEveryFile() {
        // Set up white pawns on rank 7 (row 1) for files a-h, with a
        // single white king and a single black king to make the
        // position legal. Each pawn can promote forward to its own
        // 8th-rank square. Verify all 4 promotion piece variants per
        // file produce 4 distinct policy indices and round-trip.
        var board: [Piece?] = Array(repeating: nil, count: 64)
        // Kings out of the way
        board[7 * 8 + 0] = Piece(type: .king, color: .white)
        board[0 * 8 + 7] = Piece(type: .king, color: .black)
        // Pawns on rank 7 for files b-g (avoid file 0 and 7 because of king positions)
        for col in 1..<7 {
            board[1 * 8 + col] = Piece(type: .pawn, color: .white)
        }
        let state = GameState(
            board: board,
            currentPlayer: .white,
            whiteKingsideCastle: false,
            whiteQueensideCastle: false,
            blackKingsideCastle: false,
            blackQueensideCastle: false,
            enPassantSquare: nil,
            halfmoveClock: 0
        )
        let legalMoves = MoveGenerator.legalMoves(for: state)

        // Group by (fromCol, toCol) and verify each (col → col)
        // promotion bundle has exactly 4 variants with distinct indices.
        var promotionsByFromCol: [Int: [ChessMove]] = [:]
        for move in legalMoves where move.promotion != nil && move.fromRow == 1 && move.toRow == 0 && move.toCol == move.fromCol {
            promotionsByFromCol[move.fromCol, default: []].append(move)
        }
        for (col, promoMoves) in promotionsByFromCol {
            XCTAssertEqual(promoMoves.count, 4, "Expected 4 promotion variants on file \(col), got \(promoMoves.count)")
            let indices = promoMoves.map { PolicyEncoding.policyIndex($0, currentPlayer: .white) }
            XCTAssertEqual(Set(indices).count, 4,
                           "All 4 promotions on file \(col) must produce distinct policy indices, got \(indices)")
            // Round-trip each
            for move in promoMoves {
                let (chan, r, c) = PolicyEncoding.encode(move, currentPlayer: .white)
                let decoded = PolicyEncoding.decode(channel: chan, row: r, col: c, state: state)
                XCTAssertEqual(decoded, move,
                               "Promotion round-trip failed: \(move.notation) → (\(chan),\(r),\(c)) → \(String(describing: decoded?.notation))")
            }
        }
        XCTAssertGreaterThan(promotionsByFromCol.count, 0, "No promotion moves found in test position")
    }

    // MARK: - Bijection: no two distinct legal moves share an index

    func testNoLegalMovesShareIndexAcrossManyPositions() {
        let positions = handcraftedTestPositions()
        for state in positions {
            let legalMoves = MoveGenerator.legalMoves(for: state)
            let indices = legalMoves.map { PolicyEncoding.policyIndex($0, currentPlayer: state.currentPlayer) }
            XCTAssertEqual(Set(indices).count, indices.count,
                           "Distinct legal moves produced colliding policy indices in \(legalMoves.count)-move position")
        }
    }

    // MARK: - Channel-range coverage

    func testQueenStyleChannelsExercised() {
        // From the starting position, the rook a1 doesn't move, but
        // queen-style channels are exercised via knights, bishops,
        // pawns. We don't need every direction × distance, just that
        // the queen-style (channels 0..55) bucket gets meaningful
        // hits.
        let legalMoves = MoveGenerator.legalMoves(for: .starting)
        let channels = legalMoves.map { PolicyEncoding.encode($0, currentPlayer: .white).channel }
        let queenStyleHits = channels.filter { $0 < 56 }
        XCTAssertGreaterThan(queenStyleHits.count, 0,
                             "Expected queen-style channel hits at starting position (pawn pushes, etc.)")
    }

    func testKnightChannelsExercised() {
        let legalMoves = MoveGenerator.legalMoves(for: .starting)
        let channels = legalMoves.map { PolicyEncoding.encode($0, currentPlayer: .white).channel }
        let knightHits = channels.filter { $0 >= 56 && $0 < 64 }
        XCTAssertEqual(knightHits.count, 4,
                       "Starting position has 4 knight moves (b1→a3,c3 + g1→f3,h3); got \(knightHits.count)")
    }

    func testCastlingDecodesAsQueenStyleDistance2() {
        // White kingside castle: king e1 (row 7, col 4) → g1 (row 7, col 6).
        // dr=0, dc=+2. Queen-style E direction (idx 2), distance 2.
        // Channel = 2 * 7 + (2 - 1) = 15.
        let castle = ChessMove(fromRow: 7, fromCol: 4, toRow: 7, toCol: 6, promotion: nil)
        let (chan, r, c) = PolicyEncoding.encode(castle, currentPlayer: .white)
        XCTAssertEqual(chan, 15, "Kingside castle should encode to channel 15 (E direction, distance 2)")
        XCTAssertEqual(r, 7)
        XCTAssertEqual(c, 4)
    }

    // MARK: - Off-board guard

    func testDecodeRejectsOffBoardDestinations() {
        // Channel 0 is N direction distance 1 (dr=-1, dc=0). From row 0
        // this would compute toRow=-1, which is off-board. Decode
        // should return nil rather than crashing.
        let result = PolicyEncoding.decode(channel: 0, row: 0, col: 0, state: .starting)
        XCTAssertNil(result, "Off-board destination should decode to nil")
    }

    func testDecodeRejectsOutOfRangeChannel() {
        XCTAssertNil(PolicyEncoding.decode(channel: -1, row: 0, col: 0, state: .starting))
        XCTAssertNil(PolicyEncoding.decode(channel: PolicyEncoding.channelCount, row: 0, col: 0, state: .starting))
        XCTAssertNil(PolicyEncoding.decode(channel: 9999, row: 0, col: 0, state: .starting))
    }

    func testDecodeRejectsOutOfRangeRowCol() {
        XCTAssertNil(PolicyEncoding.decode(channel: 0, row: -1, col: 0, state: .starting))
        XCTAssertNil(PolicyEncoding.decode(channel: 0, row: 8, col: 0, state: .starting))
        XCTAssertNil(PolicyEncoding.decode(channel: 0, row: 0, col: -1, state: .starting))
        XCTAssertNil(PolicyEncoding.decode(channel: 0, row: 0, col: 8, state: .starting))
    }

    // MARK: - Decode rejects illegal-here cells

    func testDecodeRejectsIllegalMoveAtCurrentPosition() {
        // Channel 1 (N direction, distance 2) from (7, 4) = e1 → e3.
        // White king cannot move 2 squares forward. Cell is
        // geometrically valid but illegal at the starting position.
        let result = PolicyEncoding.decode(channel: 1, row: 7, col: 4, state: .starting)
        XCTAssertNil(result, "Geometrically-valid but illegal move should decode to nil")
    }

    // MARK: - Both colors

    func testEncodingIsConsistentForBothColors() {
        // Same physical position, different sides to move → different
        // perspective flip → same channel value if the move shape is
        // identical from each player's POV.
        let after1e4 = MoveGenerator.applyMove(
            ChessMove(fromRow: 6, fromCol: 4, toRow: 4, toCol: 4, promotion: nil),
            to: .starting
        )
        // Black is now to move. Black's e7 pawn moving e7-e5 should
        // encode to the same channel as white's e2-e4 (both are
        // "queen-style N direction, distance 2 from row 6, col 4 in
        // encoder frame").
        let blackE7E5 = ChessMove(fromRow: 1, fromCol: 4, toRow: 3, toCol: 4, promotion: nil)
        let whiteE2E4 = ChessMove(fromRow: 6, fromCol: 4, toRow: 4, toCol: 4, promotion: nil)
        let (blackChan, blackR, blackC) = PolicyEncoding.encode(blackE7E5, currentPlayer: .black)
        let (whiteChan, whiteR, whiteC) = PolicyEncoding.encode(whiteE2E4, currentPlayer: .white)
        XCTAssertEqual(blackChan, whiteChan, "Symmetric pawn pushes should hit the same channel")
        XCTAssertEqual(blackR, whiteR, "Symmetric pawn pushes should hit the same encoder-frame row")
        XCTAssertEqual(blackC, whiteC, "Symmetric pawn pushes should hit the same column")
        // And the round-trip works for black
        let decoded = PolicyEncoding.decode(channel: blackChan, row: blackR, col: blackC, state: after1e4)
        XCTAssertEqual(decoded, blackE7E5)
    }

    // MARK: - Helpers

    /// Round-trip every legal move in `state` and assert encode/decode
    /// is the identity. Common helper used across the position-specific tests.
    private func assertRoundTripsAllLegalMoves(in state: GameState, file: StaticString = #filePath, line: UInt = #line) {
        let legalMoves = MoveGenerator.legalMoves(for: state)
        for move in legalMoves {
            let (chan, r, c) = PolicyEncoding.encode(move, currentPlayer: state.currentPlayer)
            XCTAssertGreaterThanOrEqual(chan, 0, "Channel out of range for \(move.notation)", file: file, line: line)
            XCTAssertLessThan(chan, PolicyEncoding.channelCount, "Channel out of range for \(move.notation)", file: file, line: line)
            XCTAssertGreaterThanOrEqual(r, 0, "Row out of range for \(move.notation)", file: file, line: line)
            XCTAssertLessThan(r, 8, "Row out of range for \(move.notation)", file: file, line: line)
            XCTAssertGreaterThanOrEqual(c, 0, "Col out of range for \(move.notation)", file: file, line: line)
            XCTAssertLessThan(c, 8, "Col out of range for \(move.notation)", file: file, line: line)
            let decoded = PolicyEncoding.decode(channel: chan, row: r, col: c, state: state)
            XCTAssertEqual(decoded, move,
                           "Round-trip mismatch: \(move.notation) → (\(chan),\(r),\(c)) → \(String(describing: decoded?.notation))",
                           file: file, line: line)
        }
        XCTAssertGreaterThan(legalMoves.count, 0,
                             "Test position has no legal moves — likely a setup mistake",
                             file: file, line: line)
    }

    /// A handful of distinct positions to drive the bijection test.
    /// Mix of starting position, after-1.e4, and a position with
    /// multiple promotion possibilities, plus a knight-in-corner case.
    private func handcraftedTestPositions() -> [GameState] {
        var positions: [GameState] = [.starting]

        // After 1.e4
        let after1e4 = MoveGenerator.applyMove(
            ChessMove(fromRow: 6, fromCol: 4, toRow: 4, toCol: 4, promotion: nil),
            to: .starting
        )
        positions.append(after1e4)

        // After 1.e4 e5
        let after1e4e5 = MoveGenerator.applyMove(
            ChessMove(fromRow: 1, fromCol: 4, toRow: 3, toCol: 4, promotion: nil),
            to: after1e4
        )
        positions.append(after1e4e5)

        // Promotion-position: white pawns on rank 7
        var promoBoard: [Piece?] = Array(repeating: nil, count: 64)
        promoBoard[7 * 8 + 0] = Piece(type: .king, color: .white)
        promoBoard[0 * 8 + 7] = Piece(type: .king, color: .black)
        for col in 1..<7 {
            promoBoard[1 * 8 + col] = Piece(type: .pawn, color: .white)
        }
        positions.append(GameState(
            board: promoBoard,
            currentPlayer: .white,
            whiteKingsideCastle: false, whiteQueensideCastle: false,
            blackKingsideCastle: false, blackQueensideCastle: false,
            enPassantSquare: nil, halfmoveClock: 0
        ))

        // Corner-knight position: knight at a8 (row 0, col 0), kings.
        var knightBoard: [Piece?] = Array(repeating: nil, count: 64)
        knightBoard[0 * 8 + 0] = Piece(type: .knight, color: .white)
        knightBoard[7 * 8 + 0] = Piece(type: .king, color: .white)
        knightBoard[7 * 8 + 7] = Piece(type: .king, color: .black)
        positions.append(GameState(
            board: knightBoard,
            currentPlayer: .white,
            whiteKingsideCastle: false, whiteQueensideCastle: false,
            blackKingsideCastle: false, blackQueensideCastle: false,
            enPassantSquare: nil, halfmoveClock: 0
        ))

        return positions
    }
}
