import XCTest
@testable import DrewsChessMachine

final class MoveGeneratorCastlingTests: XCTestCase {

    func testKingsideCastleRequiresRookOnHomeSquare() {
        var board = [Piece?](repeating: nil, count: 64)
        board[7 * 8 + 4] = Piece(type: .king, color: .white)
        board[0 * 8 + 4] = Piece(type: .king, color: .black)

        let state = GameState(
            board: board,
            currentPlayer: .white,
            whiteKingsideCastle: true,
            whiteQueensideCastle: false,
            blackKingsideCastle: false,
            blackQueensideCastle: false,
            enPassantSquare: nil,
            halfmoveClock: 0
        )

        let legalMoves = MoveGenerator.legalMoves(for: state)
        XCTAssertFalse(
            legalMoves.contains(
                ChessMove(fromRow: 7, fromCol: 4, toRow: 7, toCol: 6, promotion: nil)
            ),
            "Castling rights alone must not allow kingside castling when the rook is missing"
        )
    }

    func testQueensideCastleRequiresRookOnHomeSquare() {
        var board = [Piece?](repeating: nil, count: 64)
        board[7 * 8 + 4] = Piece(type: .king, color: .white)
        board[0 * 8 + 4] = Piece(type: .king, color: .black)

        let state = GameState(
            board: board,
            currentPlayer: .white,
            whiteKingsideCastle: false,
            whiteQueensideCastle: true,
            blackKingsideCastle: false,
            blackQueensideCastle: false,
            enPassantSquare: nil,
            halfmoveClock: 0
        )

        let legalMoves = MoveGenerator.legalMoves(for: state)
        XCTAssertFalse(
            legalMoves.contains(
                ChessMove(fromRow: 7, fromCol: 4, toRow: 7, toCol: 2, promotion: nil)
            ),
            "Castling rights alone must not allow queenside castling when the rook is missing"
        )
    }
}
