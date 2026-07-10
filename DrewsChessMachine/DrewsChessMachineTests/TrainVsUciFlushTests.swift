import XCTest
@testable import DrewsChessMachine

/// Verifies `ActiveGame.flushTrainerSide` — the one-sided flush the
/// train-vs-UCI driver uses. That driver records ONLY the trainer's
/// plies (on-policy, outcome-only), so exactly one colour's scratch is
/// populated. The two-sided `flush` cannot be used there: it sources
/// each game-ply by parity and would read the *unrecorded* opposite
/// side's scratch for half the rows (corrupt data). These tests pin
/// that `flushTrainerSide`:
///   - appends exactly the trainer side's recorded plies (never the
///     unrecorded opposite side — the core correctness bug),
///   - signs every row's outcome from the trainer's colour (+1 win /
///     -1 loss / 0 draw), for a white trainer AND a black trainer,
///   - returns nil (and appends nothing) when the trainer side recorded
///     no plies.
///
/// Shares one `ChessMPSNetwork` (basic30, single-frame — the encoding
/// train-vs-UCI supports) to amortize MPS graph build time; the network
/// is never invoked by the methods under test.
final class TrainVsUciFlushTests: XCTestCase {

    private static var sharedNetwork: ChessMPSNetwork = {
        do {
            return try ChessMPSNetwork(.randomWeights)
        } catch {
            fatalError("TrainVsUciFlushTests: ChessMPSNetwork(.randomWeights) failed: \(error)")
        }
    }()

    private func makeGame(capPlies: Int) -> ActiveGame {
        ActiveGame(
            workerId: 7,
            whiteNetwork: Self.sharedNetwork,
            blackNetwork: Self.sharedNetwork,
            capPlies: capPlies,
            schedule: .uniform
        )
    }

    private func fakeBoardBytes(seed: UInt32) -> [Float] {
        var arr = [Float](repeating: 0, count: BoardEncoder.tensorLength(for: .basic30))
        for i in 0..<arr.count {
            arr[i] = Float((Int(seed) &+ i) % 1000) * 0.001
        }
        return arr
    }

    private func record(_ game: ActiveGame, side: PieceColor, seed: UInt32) {
        var bytes = fakeBoardBytes(seed: seed)
        bytes.withUnsafeMutableBufferPointer { buf in
            game.recordPly(
                side: side,
                encodedBoardSrc: UnsafePointer(buf.baseAddress!),
                policyIndex: Int(seed),
                samplingTau: 1.0,
                materialCount: 32
            )
        }
    }

    /// Sample `n` positions and return their outcome (`z`) values.
    private func sampledOutcomes(_ buffer: ReplayBuffer, count n: Int) -> [Float] {
        var boards = [Float](repeating: 0, count: n * ReplayBuffer.defaultFloatsPerBoard)
        var moves = [Int32](repeating: 0, count: n)
        var zs = [Float](repeating: 0, count: n)
        _ = boards.withUnsafeMutableBufferPointer { bBuf in
            moves.withUnsafeMutableBufferPointer { mBuf in
                zs.withUnsafeMutableBufferPointer { zBuf in
                    buffer.sample(count: n, intoBoards: bBuf.baseAddress!, moves: mBuf.baseAddress!, zs: zBuf.baseAddress!)
                }
            }
        }
        return zs
    }

    // MARK: - White trainer

    func test_whiteTrainer_win_appendsOnlyWhitePliesWithPlusOne() {
        let g = makeGame(capPlies: 40)
        // Record ONLY the trainer's (white) plies — the opponent's black
        // plies are deliberately not recorded.
        for i in 0..<5 { record(g, side: .white, seed: UInt32(i + 1)) }

        let buffer = ReplayBuffer(capacity: 100)
        let stats = g.flushTrainerSide(
            buffer: buffer, result: .checkmate(winner: .white),
            trainerColor: .white, totalPlies: 9)

        // Exactly the 5 recorded white plies — NOT 10, and NOT reading
        // the unrecorded black scratch.
        XCTAssertEqual(stats?.positions, 5)
        XCTAssertEqual(buffer.count, 5, "must append only the trainer's recorded plies")
        for z in sampledOutcomes(buffer, count: 5) {
            XCTAssertEqual(z, 1.0, "white trainer won → every row outcome must be +1")
        }
    }

    func test_whiteTrainer_loss_isNegativeOne() {
        let g = makeGame(capPlies: 40)
        for i in 0..<5 { record(g, side: .white, seed: UInt32(i + 1)) }

        let buffer = ReplayBuffer(capacity: 100)
        _ = g.flushTrainerSide(
            buffer: buffer, result: .checkmate(winner: .black),
            trainerColor: .white, totalPlies: 10)

        XCTAssertEqual(buffer.count, 5)
        for z in sampledOutcomes(buffer, count: 5) {
            XCTAssertEqual(z, -1.0, "white trainer lost → every row outcome must be -1")
        }
    }

    // MARK: - Black trainer (sign is from black's perspective)

    func test_blackTrainer_win_appendsOnlyBlackPliesWithPlusOne() {
        let g = makeGame(capPlies: 40)
        // Trainer plays black: record ONLY black plies.
        for i in 0..<5 { record(g, side: .black, seed: UInt32(i + 50)) }

        let buffer = ReplayBuffer(capacity: 100)
        let stats = g.flushTrainerSide(
            buffer: buffer, result: .checkmate(winner: .black),
            trainerColor: .black, totalPlies: 9)

        XCTAssertEqual(stats?.positions, 5)
        XCTAssertEqual(buffer.count, 5)
        for z in sampledOutcomes(buffer, count: 5) {
            XCTAssertEqual(z, 1.0, "black trainer won → +1 from black's perspective")
        }
    }

    func test_blackTrainer_loss_isNegativeOne() {
        let g = makeGame(capPlies: 40)
        for i in 0..<5 { record(g, side: .black, seed: UInt32(i + 50)) }

        let buffer = ReplayBuffer(capacity: 100)
        _ = g.flushTrainerSide(
            buffer: buffer, result: .checkmate(winner: .white),
            trainerColor: .black, totalPlies: 10)

        XCTAssertEqual(buffer.count, 5)
        for z in sampledOutcomes(buffer, count: 5) {
            XCTAssertEqual(z, -1.0, "black trainer lost → -1 from black's perspective")
        }
    }

    // MARK: - Draws

    func test_draw_isZeroForEitherColor() {
        for trainerColor in [PieceColor.white, .black] {
            let g = makeGame(capPlies: 40)
            for i in 0..<4 { record(g, side: trainerColor, seed: UInt32(i + 1)) }
            let buffer = ReplayBuffer(capacity: 100)
            _ = g.flushTrainerSide(
                buffer: buffer, result: .drawByFiftyMoveRule,
                trainerColor: trainerColor, totalPlies: 8)
            XCTAssertEqual(buffer.count, 4)
            for z in sampledOutcomes(buffer, count: 4) {
                XCTAssertEqual(z, 0.0, "draw → 0 for a \(trainerColor) trainer")
            }
        }
    }

    // MARK: - Empty recorded side

    func test_emptyTrainerSide_returnsNilAndAppendsNothing() {
        let g = makeGame(capPlies: 40)
        // Record white plies but flush as the BLACK trainer — black has
        // recorded nothing, so there is nothing to flush.
        for i in 0..<3 { record(g, side: .white, seed: UInt32(i + 1)) }

        let buffer = ReplayBuffer(capacity: 100)
        let stats = g.flushTrainerSide(
            buffer: buffer, result: .checkmate(winner: .white),
            trainerColor: .black, totalPlies: 5)

        XCTAssertNil(stats, "no plies recorded for the trainer's colour → nil")
        XCTAssertEqual(buffer.count, 0, "nothing should be appended")
    }
}
