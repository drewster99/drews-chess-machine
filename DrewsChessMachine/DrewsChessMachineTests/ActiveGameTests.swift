import XCTest
@testable import DrewsChessMachine

/// Tests for `ActiveGame` — the per-game state holder used by the
/// upcoming tick-based self-play driver. Verifies the lifecycle
/// (init / record / flush / reset / cap-grow) without touching the
/// driver itself, so the tick-driver phase can land knowing this
/// layer is sound.
///
/// The network references on `ActiveGame` are stored but never
/// invoked by the methods under test (init, resetForNewGame,
/// recordPly, flush, deinit). Tests share one `ChessMPSNetwork` to
/// amortize MPS graph build time.
final class ActiveGameTests: XCTestCase {

    private static var sharedNetwork: ChessMPSNetwork = {
        // Force-tried: there's no usable fallback if `.randomWeights` fails
        // (it would mean MPS / Metal is broken on the test host). All
        // existing network-bearing tests in this target follow the same
        // pattern (`try!` on init for the shared instance).
        do {
            return try ChessMPSNetwork(.randomWeights)
        } catch {
            fatalError("ActiveGameTests: ChessMPSNetwork(.randomWeights) failed: \(error)")
        }
    }()

    /// Convenience: an `ActiveGame` with both sides on the shared
    /// network, given a per-game cap and a default `.uniform` schedule.
    private func makeGame(capPlies: Int) -> ActiveGame {
        ActiveGame(
            workerId: 42,
            whiteNetwork: Self.sharedNetwork,
            blackNetwork: Self.sharedNetwork,
            capPlies: capPlies,
            schedule: .uniform
        )
    }

    /// One ply-worth of fake encoded-board bytes. The driver supplies
    /// real `BoardEncoder.encode` output here; for layout-only tests
    /// any reproducible content works.
    private func fakeBoardBytes(seed: UInt32) -> [Float] {
        var arr = [Float](repeating: 0, count: BoardEncoder.tensorLength)
        // Deterministic small-period pattern so different seeds produce
        // distinguishable bytes without pulling in an RNG.
        for i in 0..<arr.count {
            arr[i] = Float((Int(seed) &+ i) % 1000) * 0.001
        }
        return arr
    }

    /// Wraps `recordPly` so each test reads cleanly without raw-pointer
    /// boilerplate. The driver itself passes a `UnsafePointer<Float>`
    /// slice of its tick scratch.
    private func record(
        _ game: ActiveGame,
        side: PieceColor,
        seed: UInt32,
        policyIndex: Int = 0,
        tau: Float = 1.0,
        material: UInt8 = 32
    ) {
        var bytes = fakeBoardBytes(seed: seed)
        bytes.withUnsafeMutableBufferPointer { buf in
            game.recordPly(
                side: side,
                encodedBoardSrc: UnsafePointer(buf.baseAddress!),
                policyIndex: policyIndex,
                samplingTau: tau,
                materialCount: material
            )
        }
    }

    // MARK: - Init

    func test_init_capacityIsCeilOfHalf() {
        // capPlies=150 -> perSideCap=(150+1)/2=75
        let g = makeGame(capPlies: 150)
        XCTAssertEqual(g.perSideCap, 75)
        XCTAssertEqual(g.maxPliesCap, 150)
        XCTAssertEqual(g.whitePliesRecorded, 0)
        XCTAssertEqual(g.blackPliesRecorded, 0)
        XCTAssertEqual(g.totalPliesPlayed, 0)
        XCTAssertEqual(g.intraWorkerGameIndex, 0)
        XCTAssertEqual(g.workerId, 42)
    }

    func test_init_oddCapRoundsUp() {
        // capPlies=151 -> perSideCap=(151+1)/2=76 — covers the odd-ply case.
        let g = makeGame(capPlies: 151)
        XCTAssertEqual(g.perSideCap, 76)
    }

    // MARK: - recordPly

    func test_recordPly_advancesCorrectSide() {
        let g = makeGame(capPlies: 10)
        record(g, side: .white, seed: 1)
        XCTAssertEqual(g.whitePliesRecorded, 1)
        XCTAssertEqual(g.blackPliesRecorded, 0)
        XCTAssertEqual(g.totalPliesPlayed, 1)

        record(g, side: .black, seed: 2)
        XCTAssertEqual(g.whitePliesRecorded, 1)
        XCTAssertEqual(g.blackPliesRecorded, 1)
        XCTAssertEqual(g.totalPliesPlayed, 2)
    }

    func test_recordPly_manyPliesEachSide() {
        // capPlies=20 → perSideCap=10. Drive 10 each side.
        let g = makeGame(capPlies: 20)
        for i in 0..<10 {
            record(g, side: .white, seed: UInt32(i * 2 + 1))
            record(g, side: .black, seed: UInt32(i * 2 + 2))
        }
        XCTAssertEqual(g.whitePliesRecorded, 10)
        XCTAssertEqual(g.blackPliesRecorded, 10)
        XCTAssertEqual(g.totalPliesPlayed, 20)
    }

    // MARK: - resetForNewGame

    func test_reset_bumpsIntraWorkerGameIndex() {
        let g = makeGame(capPlies: 10)
        XCTAssertEqual(g.intraWorkerGameIndex, 0)
        g.resetForNewGame(maxPliesCap: 10, schedule: .uniform)
        XCTAssertEqual(g.intraWorkerGameIndex, 1)
        g.resetForNewGame(maxPliesCap: 10, schedule: .uniform)
        XCTAssertEqual(g.intraWorkerGameIndex, 2)
    }

    func test_reset_zerosPlyCounters() {
        let g = makeGame(capPlies: 10)
        record(g, side: .white, seed: 1)
        record(g, side: .black, seed: 2)
        record(g, side: .white, seed: 3)
        XCTAssertEqual(g.totalPliesPlayed, 3)

        g.resetForNewGame(maxPliesCap: 10, schedule: .uniform)
        XCTAssertEqual(g.whitePliesRecorded, 0)
        XCTAssertEqual(g.blackPliesRecorded, 0)
        XCTAssertEqual(g.totalPliesPlayed, 0)
    }

    func test_reset_doesNotShrinkOnSmallerCap() {
        // Allocate at cap=20 (perSideCap=10), reset at cap=10
        // (perSideCap=5). Slot stays at 10 — never shrink.
        let g = makeGame(capPlies: 20)
        XCTAssertEqual(g.perSideCap, 10)
        g.resetForNewGame(maxPliesCap: 10, schedule: .uniform)
        XCTAssertEqual(g.perSideCap, 10, "Reset to smaller cap must NOT shrink")
        XCTAssertEqual(g.maxPliesCap, 10, "maxPliesCap field DOES follow the new value")
    }

    func test_reset_growsOnLargerCap() {
        // Allocate at cap=10 (perSideCap=5), reset at cap=30 (perSideCap=15).
        let g = makeGame(capPlies: 10)
        XCTAssertEqual(g.perSideCap, 5)
        g.resetForNewGame(maxPliesCap: 30, schedule: .uniform)
        XCTAssertEqual(g.perSideCap, 15, "Reset to larger cap must grow")
        XCTAssertEqual(g.maxPliesCap, 30)
        // After grow, recording 15 per side must succeed (no overflow).
        for i in 0..<15 {
            record(g, side: .white, seed: UInt32(i * 2 + 1))
            record(g, side: .black, seed: UInt32(i * 2 + 2))
        }
        XCTAssertEqual(g.whitePliesRecorded, 15)
        XCTAssertEqual(g.blackPliesRecorded, 15)
    }

    // MARK: - Flush

    func test_flush_emptyGame_returnsNil() {
        let g = makeGame(capPlies: 10)
        let buffer = ReplayBuffer(capacity: 100)
        let result = g.flush(buffer: buffer, result: .stalemate)
        XCTAssertNil(result, "Flush of an empty game should return nil")
        XCTAssertEqual(buffer.count, 0)
    }

    func test_flush_whiteCheckmate_appendsAllPliesAndCorrectOutcomes() {
        // 6 white plies + 6 black plies in alternation; .checkmate(.white)
        // ⇒ white outcome=+1, black outcome=-1. Total 12 positions
        // appended to the buffer.
        let g = makeGame(capPlies: 20)
        for i in 0..<6 {
            record(g, side: .white, seed: UInt32(i * 2 + 1), policyIndex: 100 + i)
            record(g, side: .black, seed: UInt32(i * 2 + 2), policyIndex: 200 + i)
        }
        let buffer = ReplayBuffer(capacity: 100)
        let stats = g.flush(buffer: buffer, result: .checkmate(winner: .white))
        XCTAssertNotNil(stats)
        XCTAssertEqual(stats?.positions, 12)
        XCTAssertEqual(buffer.count, 12, "Buffer should hold 6 white + 6 black plies")
    }

    func test_flush_draw_appendsWithZeroOutcomes() {
        let g = makeGame(capPlies: 20)
        for i in 0..<3 {
            record(g, side: .white, seed: UInt32(i + 1))
            record(g, side: .black, seed: UInt32(i + 100))
        }
        let buffer = ReplayBuffer(capacity: 100)
        let stats = g.flush(buffer: buffer, result: .drawByThreefoldRepetition)
        XCTAssertEqual(stats?.positions, 6)
        XCTAssertEqual(buffer.count, 6)
        // Spot-check: sample N positions; every outcome should be 0
        // for a draw (no per-side sign flip).
        let n = 6
        var boards = [Float](repeating: 0, count: n * ReplayBuffer.floatsPerBoard)
        var moves  = [Int32](repeating: 0, count: n)
        var zs     = [Float](repeating: 0, count: n)
        let ok = boards.withUnsafeMutableBufferPointer { bBuf in
            moves.withUnsafeMutableBufferPointer { mBuf in
                zs.withUnsafeMutableBufferPointer { zBuf in
                    buffer.sample(count: n, intoBoards: bBuf.baseAddress!, moves: mBuf.baseAddress!, zs: zBuf.baseAddress!)
                }
            }
        }
        XCTAssertTrue(ok)
        for z in zs {
            XCTAssertEqual(z, 0.0, "Draw outcome must be exactly 0 on both sides")
        }
    }

    func test_flush_blackCheckmate_appendsWithSignFlippedOutcomes() {
        // 4 white + 4 black; .checkmate(.black) ⇒ white outcome=-1,
        // black outcome=+1. We sample everything and assert at least
        // one of each sign appears.
        let g = makeGame(capPlies: 20)
        for i in 0..<4 {
            record(g, side: .white, seed: UInt32(i + 1))
            record(g, side: .black, seed: UInt32(i + 100))
        }
        let buffer = ReplayBuffer(capacity: 100)
        _ = g.flush(buffer: buffer, result: .checkmate(winner: .black))
        XCTAssertEqual(buffer.count, 8)

        // Sample-with-replacement N times; over 200 draws we should see
        // both -1 and +1.
        let n = 200
        var boards = [Float](repeating: 0, count: n * ReplayBuffer.floatsPerBoard)
        var moves  = [Int32](repeating: 0, count: n)
        var zs     = [Float](repeating: 0, count: n)
        _ = boards.withUnsafeMutableBufferPointer { bBuf in
            moves.withUnsafeMutableBufferPointer { mBuf in
                zs.withUnsafeMutableBufferPointer { zBuf in
                    buffer.sample(count: n, intoBoards: bBuf.baseAddress!, moves: mBuf.baseAddress!, zs: zBuf.baseAddress!)
                }
            }
        }
        let sawPos = zs.contains { $0 > 0.5 }
        let sawNeg = zs.contains { $0 < -0.5 }
        let sawZero = zs.contains { $0 == 0.0 }
        XCTAssertTrue(sawPos, "Should sample at least one +1 outcome (black wins)")
        XCTAssertTrue(sawNeg, "Should sample at least one -1 outcome (white loses)")
        XCTAssertFalse(sawZero, "Non-draw flush should not produce any zero outcomes")
    }

    // MARK: - Reset across games (no data leak from prior game)

    func test_reset_thenFlush_secondGameWritesOnlySecondGamesPlies() {
        let g = makeGame(capPlies: 20)
        // First game: record 4 plies (2 white, 2 black) and flush.
        record(g, side: .white, seed: 1)
        record(g, side: .black, seed: 2)
        record(g, side: .white, seed: 3)
        record(g, side: .black, seed: 4)
        let buffer = ReplayBuffer(capacity: 100)
        _ = g.flush(buffer: buffer, result: .stalemate)
        XCTAssertEqual(buffer.count, 4)

        // Reset for a new game; record only 2 plies (1 each side) and
        // flush. Buffer should grow by exactly 2.
        g.resetForNewGame(maxPliesCap: 20, schedule: .uniform)
        record(g, side: .white, seed: 99)
        record(g, side: .black, seed: 100)
        _ = g.flush(buffer: buffer, result: .stalemate)
        XCTAssertEqual(buffer.count, 6, "Second flush should append exactly 2 new positions")
    }
}
