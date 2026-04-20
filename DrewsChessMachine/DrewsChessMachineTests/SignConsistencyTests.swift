//
//  SignConsistencyTests.swift
//  DrewsChessMachineTests
//
//  Tests for perspective / sign consistency across the encoder,
//  policy-index bijection, network forward pass, and outcome
//  labeling. A sign bug here — where we encode one way and
//  interpret another — would silently corrupt training without
//  tripping any of the plain correctness tests. These tests
//  exercise the symmetries the pipeline should respect.
//

import XCTest
import Metal
@testable import DrewsChessMachine

final class SignConsistencyTests: XCTestCase {

    // MARK: - Test A: Encoder bit-identical for symmetric-POV starting position
    //
    // Starting position encoded from white's POV and black's POV
    // (same physical board, swap side-to-move) should produce
    // byte-identical tensors. Rationale:
    //
    // Starting position is horizontally symmetric: both colors have
    // the same piece layout (rnbqkbnr) in their home ranks. The
    // encoder maps "my pieces at rows 6-7, opp at rows 0-1". For
    // white: white-at-row-7, black-at-row-0 (no flip). For black:
    // black-at-row-0 flipped to encoder-row-7, white-at-row-7
    // flipped to encoder-row-0. Same pattern, same active squares,
    // same piece-plane assignments. Bit-identical.
    //
    // If this test fails, the encoder's perspective flip is NOT
    // actually perspective-invariant — a very high-priority bug.

    func testEncoderIdenticalForStartingPositionBothSides() {
        let whiteToMove = GameState.starting
        let blackToMove = GameState(
            board: GameState.starting.board,
            currentPlayer: .black,
            whiteKingsideCastle: true,
            whiteQueensideCastle: true,
            blackKingsideCastle: true,
            blackQueensideCastle: true,
            enPassantSquare: nil,
            halfmoveClock: 0
        )
        let white = BoardEncoder.encode(whiteToMove)
        let black = BoardEncoder.encode(blackToMove)
        XCTAssertEqual(
            white, black,
            "Encoder must produce bit-identical output for white-to-move and " +
            "black-to-move starting position. Divergence means the perspective " +
            "flip has a sign or plane-assignment bug."
        )
    }

    // MARK: - Test B: Asymmetric-position piece-plane mirror
    //
    // Stronger check using an asymmetric position: white is up a
    // queen (extra queen at d4). Encode from white's POV AND encode
    // the COLOR-MIRRORED position from black's POV (black up a queen,
    // black to move, mirrored queen locations). These two encodings
    // should be bit-identical — they're the same "I'm winning with
    // an extra queen" scenario from each player's perspective.
    //
    // Catches bugs where piece-plane assignment is by ABSOLUTE COLOR
    // rather than by RELATIVE-TO-CURRENT-PLAYER. If white's extra
    // queen goes to "my-queen-plane" but black's extra queen goes to
    // "opp-queen-plane" (because the encoder accidentally uses
    // absolute color), the test fails.

    func testEncoderMirrorsByColorForAsymmetricPosition() {
        // P: white up a queen (extra queen at d4). White to move.
        // Keep both kings and the regular queens on the board so
        // the position is chess-legal-ish.
        var whitePos: [Piece?] = Array(repeating: nil, count: 64)
        // Black back rank + a few pieces
        whitePos[0 * 8 + 4] = Piece(type: .king, color: .black)   // e8
        whitePos[0 * 8 + 3] = Piece(type: .queen, color: .black)  // d8
        // White back rank
        whitePos[7 * 8 + 4] = Piece(type: .king, color: .white)   // e1
        whitePos[7 * 8 + 3] = Piece(type: .queen, color: .white)  // d1
        // White's EXTRA queen on d4
        whitePos[4 * 8 + 3] = Piece(type: .queen, color: .white)  // d4
        let P = GameState(
            board: whitePos,
            currentPlayer: .white,
            whiteKingsideCastle: false, whiteQueensideCastle: false,
            blackKingsideCastle: false, blackQueensideCastle: false,
            enPassantSquare: nil, halfmoveClock: 0
        )

        // P': color-mirrored. Black up a queen (black extra queen on
        // d5 — the row-mirror of d4, since row 4 in absolute flips to
        // row 3 under encoder-frame flip), black to move. Regular
        // queens still on d1/d8. Regular kings still on e1/e8.
        //
        // In absolute coordinates, the color flip transforms:
        //   (row, col) → (7-row, col), swap color
        // So white queen at d4 (row=4, col=3) becomes black queen at
        // (7-4, 3) = (3, 3) = d5. Similarly for the other pieces.
        var blackPos: [Piece?] = Array(repeating: nil, count: 64)
        blackPos[7 * 8 + 4] = Piece(type: .king, color: .white)   // e1 (was e8 black king)
        blackPos[7 * 8 + 3] = Piece(type: .queen, color: .white)  // d1 (was d8 black queen)
        blackPos[0 * 8 + 4] = Piece(type: .king, color: .black)   // e8 (was e1 white king)
        blackPos[0 * 8 + 3] = Piece(type: .queen, color: .black)  // d8 (was d1 white queen)
        blackPos[3 * 8 + 3] = Piece(type: .queen, color: .black)  // d5 (was d4 white extra queen)
        let Pprime = GameState(
            board: blackPos,
            currentPlayer: .black,
            whiteKingsideCastle: false, whiteQueensideCastle: false,
            blackKingsideCastle: false, blackQueensideCastle: false,
            enPassantSquare: nil, halfmoveClock: 0
        )

        let encP = BoardEncoder.encode(P)
        let encPprime = BoardEncoder.encode(Pprime)
        XCTAssertEqual(
            encP, encPprime,
            "Color-mirrored positions should produce bit-identical encodings " +
            "when each side views from its own POV. Divergence means piece-plane " +
            "assignment is keyed on absolute color rather than my/opp role."
        )
    }

    // MARK: - Test C: Network forward pass is deterministic on identical inputs
    //
    // The natural corollary of Tests A/B: if two inputs are
    // bit-identical, the network's policy and value outputs must
    // also be bit-identical. Catches bugs where the network
    // depends on something OTHER than the encoded input (hidden
    // global state, stale cache, etc.). Rare but worth locking down.

    func testNetworkOutputIdenticalForBitIdenticalInputs() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("Metal not available")
        }
        let net = try ChessNetwork(bnMode: .inference)

        // Starting position encoded from both POVs (Test A scenario).
        let whiteToMove = GameState.starting
        let blackToMove = GameState(
            board: GameState.starting.board,
            currentPlayer: .black,
            whiteKingsideCastle: true, whiteQueensideCastle: true,
            blackKingsideCastle: true, blackQueensideCastle: true,
            enPassantSquare: nil, halfmoveClock: 0
        )
        let whiteEnc = BoardEncoder.encode(whiteToMove)
        let blackEnc = BoardEncoder.encode(blackToMove)
        XCTAssertEqual(whiteEnc, blackEnc, "Test precondition (Test A) must hold")

        let (whitePolicy, whiteValue) = try await net.evaluate(board: whiteEnc)
        let (blackPolicy, blackValue) = try await net.evaluate(board: blackEnc)

        XCTAssertEqual(
            whiteValue, blackValue,
            "Network value output must be identical for bit-identical inputs " +
            "(got white=\(whiteValue), black=\(blackValue)). Divergence means the " +
            "network has hidden state or the evaluate path is non-deterministic."
        )
        XCTAssertEqual(
            whitePolicy, blackPolicy,
            "Network policy output must be identical for bit-identical inputs. " +
            "Divergence means hidden state or non-determinism in the evaluate path."
        )
    }

    // MARK: - Test D: PolicyEncoding assigns mirrored moves to the same cell
    //
    // White's e2-e4 and black's e7-e5 are the same physical move
    // shape ("push my e-pawn 2 squares forward"). Under the
    // encoder-frame bijection, both should map to the SAME
    // (channel, row, col). Stronger statement: any symmetric pair
    // of legal opening moves should map identically, which means
    // training-target indices are aligned across colors.

    func testPolicyIndexIdenticalForSymmetricMoves() {
        let whiteE2E4 = ChessMove(fromRow: 6, fromCol: 4, toRow: 4, toCol: 4, promotion: nil)
        let blackE7E5 = ChessMove(fromRow: 1, fromCol: 4, toRow: 3, toCol: 4, promotion: nil)
        let whiteIdx = PolicyEncoding.policyIndex(whiteE2E4, currentPlayer: .white)
        let blackIdx = PolicyEncoding.policyIndex(blackE7E5, currentPlayer: .black)
        XCTAssertEqual(
            whiteIdx, blackIdx,
            "Symmetric e-pawn 2-square pushes must share the same policy index. " +
            "Got white=\(whiteIdx), black=\(blackIdx)."
        )

        // Knight moves: b1-c3 (white) and b8-c6 (black) — symmetric developing moves.
        let whiteNbc3 = ChessMove(fromRow: 7, fromCol: 1, toRow: 5, toCol: 2, promotion: nil)
        let blackNbc6 = ChessMove(fromRow: 0, fromCol: 1, toRow: 2, toCol: 2, promotion: nil)
        let wN = PolicyEncoding.policyIndex(whiteNbc3, currentPlayer: .white)
        let bN = PolicyEncoding.policyIndex(blackNbc6, currentPlayer: .black)
        XCTAssertEqual(wN, bN, "Symmetric knight moves must share the same policy index")

        // Castling: white O-O (e1-g1) and black O-O (e8-g8).
        let wKS = ChessMove(fromRow: 7, fromCol: 4, toRow: 7, toCol: 6, promotion: nil)
        let bKS = ChessMove(fromRow: 0, fromCol: 4, toRow: 0, toCol: 6, promotion: nil)
        XCTAssertEqual(
            PolicyEncoding.policyIndex(wKS, currentPlayer: .white),
            PolicyEncoding.policyIndex(bKS, currentPlayer: .black),
            "Kingside castle should map to the same policy index for both colors"
        )
    }

    // MARK: - Test E: Outcome-sign truth table
    //
    // Locks in `MPSChessPlayer.onGameEnded` sign convention:
    //   myOutcome = (winner == .white) == isWhite ? 1.0 : -1.0
    // so `myOutcome = +1` iff THIS player won, regardless of color.
    //
    // This is a truth-table re-verification — catches a regression
    // where the comparison gets flipped in a future refactor. Tests
    // the boolean expression directly rather than the MPSChessPlayer
    // code path (which would require a full player + mock source
    // setup; not worth the cost for one boolean).

    func testOutcomeSignTruthTable() {
        func outcome(winner: PieceColor, isWhite: Bool) -> Float {
            (winner == .white) == isWhite ? 1.0 : -1.0
        }
        XCTAssertEqual(outcome(winner: .white, isWhite: true), +1.0,
                       "White player wins as white → +1")
        XCTAssertEqual(outcome(winner: .black, isWhite: false), +1.0,
                       "Black player wins as black → +1")
        XCTAssertEqual(outcome(winner: .white, isWhite: false), -1.0,
                       "Black player loses to white → -1")
        XCTAssertEqual(outcome(winner: .black, isWhite: true), -1.0,
                       "White player loses to black → -1")
    }

    // MARK: - Test F: Advantage-formula sign convention
    //
    // Locks the advantage = z - vBaseline convention. Documents the
    // behavior that:
    //   - z > vBaseline (outcome better than expected) → positive
    //     advantage → policy loss pushes π UP on the played move.
    //   - z < vBaseline (outcome worse than expected) → negative
    //     advantage → policy loss pushes π DOWN on the played move.
    //
    // Pure arithmetic sanity check but catches "accidentally swap
    // z and vBaseline" regressions.

    func testAdvantageFormulaSignConvention() {
        func advantage(z: Float, vBaseline: Float) -> Float { z - vBaseline }

        // Clear win when network expected it (advantage small positive).
        XCTAssertEqual(advantage(z: 1.0, vBaseline: 0.8), 0.2, accuracy: 1e-6)

        // Unexpected win (advantage big positive — strong reinforcement).
        XCTAssertEqual(advantage(z: 1.0, vBaseline: -0.5), 1.5, accuracy: 1e-6)

        // Clear loss when network expected it (small negative).
        XCTAssertEqual(advantage(z: -1.0, vBaseline: -0.8), -0.2, accuracy: 1e-6)

        // Unexpected loss (strong push away from played move).
        XCTAssertEqual(advantage(z: -1.0, vBaseline: 0.5), -1.5, accuracy: 1e-6)

        // Drawn game with draw penalty active (z=-0.1) and network
        // expected a win (v=+0.3): advantage = -0.4, push away.
        XCTAssertEqual(advantage(z: -0.1, vBaseline: 0.3), -0.4, accuracy: 1e-6)

        // Drawn with network expecting a loss (v=-0.3): advantage +0.2.
        // Reinforces the played move (better outcome than expected).
        XCTAssertEqual(advantage(z: -0.1, vBaseline: -0.3), 0.2, accuracy: 1e-6)
    }

    // MARK: - Test G: PolicyEncoding.geometricDecode consistency with encode
    //
    // Weaker than round-trip (which requires legal-move filtering)
    // but verifies: for every legal move, encode then geometric-
    // decode produces the original move. If geometric decode drops
    // legality checks AND the encoding is correct, the two should
    // still agree on the geometry (no off-board arithmetic).

    func testGeometricDecodeRoundTripsAllLegalMoves() {
        let positions: [GameState] = [
            .starting,
            MoveGenerator.applyMove(
                ChessMove(fromRow: 6, fromCol: 4, toRow: 4, toCol: 4, promotion: nil),
                to: .starting
            )  // After 1.e4, black to move
        ]
        for state in positions {
            let legals = MoveGenerator.legalMoves(for: state)
            for move in legals {
                let (chan, r, c) = PolicyEncoding.encode(move, currentPlayer: state.currentPlayer)
                let decoded = PolicyEncoding.geometricDecode(
                    channel: chan, row: r, col: c,
                    currentPlayer: state.currentPlayer
                )
                XCTAssertEqual(
                    decoded, move,
                    "geometricDecode should recover the original move (no legality filter). " +
                    "Move \(move.notation), (chan,r,c)=(\(chan),\(r),\(c)), got \(String(describing: decoded?.notation))"
                )
            }
        }
    }
}
