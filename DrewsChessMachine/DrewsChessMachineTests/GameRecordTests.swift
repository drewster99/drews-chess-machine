import XCTest
@testable import DrewsChessMachine

final class GameRecordTests: XCTestCase {

    /// The packed move must be a bijection across every (from, to, promotion)
    /// — getting this wrong silently corrupts the recorded policy target.
    func testMoveCodecRoundTripAllSquaresAndPromotions() throws {
        let promos: [PieceType?] = [nil, .knight, .bishop, .rook, .queen]
        for from in 0..<64 {
            for to in 0..<64 {
                for promo in promos {
                    let move = ChessMove(fromRow: from / 8, fromCol: from % 8,
                                         toRow: to / 8, toCol: to % 8,
                                         promotion: promo)
                    let unpacked = try PackedMove.unpack(PackedMove.pack(move))
                    XCTAssertEqual(unpacked, move, "round-trip failed for \(move.notation)")
                }
            }
        }
    }

    func testUnpackRejectsInvalidPromotionCode() {
        for badCode in UInt16(5)...7 {
            let packed = badCode << 12   // from/to = 0, promo code 5/6/7
            XCTAssertThrowsError(try PackedMove.unpack(packed)) { error in
                guard let e = error as? GameCorpusError, case .corruptMove = e else {
                    return XCTFail("expected corruptMove, got \(error)")
                }
            }
        }
    }

    func testResultMappingFromGameResult() {
        let m = [ChessMove(fromRow: 1, fromCol: 4, toRow: 3, toCol: 4, promotion: nil)]
        XCTAssertEqual(GameRecord(moves: m, result: .checkmate(winner: .white)).outcome, .whiteWin)
        XCTAssertEqual(GameRecord(moves: m, result: .checkmate(winner: .black)).outcome, .blackWin)
        XCTAssertEqual(GameRecord(moves: m, result: .stalemate).outcome, .draw)
        XCTAssertEqual(GameRecord(moves: m, result: .drawByThreefoldRepetition).outcome, .draw)
        XCTAssertEqual(GameRecord(moves: m, result: .checkmate(winner: .white)).terminationReason, .checkmate)
        XCTAssertEqual(GameRecord(moves: m, result: .drawByFiftyMoveRule).terminationReason, .fiftyMoveRule)
        XCTAssertEqual(GameRecord(moves: m, result: .drawByInsufficientMaterial).terminationReason, .insufficientMaterial)
    }
}
