//
//  Full10Ply10Reps210EncodingTests.swift
//  DrewsChessMachineTests
//
//  `full10Ply10Reps210` = `full10ply200`'s 200 stacked planes + the 10 basic30
//  temporal-repetition planes appended at 200–209 (describing the CURRENT
//  position only). The reps are NOT stored in the replay buffer — it stores the
//  same 20-plane basic20 frames as full10ply200. The inference path fills the
//  tail from the live `GameState.recentRepetitionMask`; the training path
//  recomputes it from stored priors in `ReplayBuffer.appendRepetitionTail`.
//
//  The correctness gate (`test_reconstruction_with_recomputed_reps_equals_bakeIn`)
//  drives a knight-shuffle game with real repetitions through `ChessGameEngine`,
//  stores its basic20 frames, reconstructs every ply through the buffer, and
//  asserts the full 210-plane reconstruction equals the bake-in encode — which
//  derives its tail from `recentRepetitionMask`. So recompute == basic30 mask,
//  EXACTLY, including across an irreversible-move boundary.
//

import XCTest
@testable import DrewsChessMachine

final class Full10Ply10Reps210EncodingTests: XCTestCase {

    // MARK: - Structure

    func testStructure() {
        let e = InputEncoding.full10Ply10Reps210
        XCTAssertEqual(e.planeCount, 210)
        XCTAssertEqual(e.historyFrameCount, 10)
        XCTAssertEqual(e.planesPerFrame, 20)
        XCTAssertEqual(e.tailPlaneCount, 10)
        XCTAssertEqual(BoardEncoder.tensorLength(for: e), 210 * 64)
        // Generalized frame+tail invariant for every encoding.
        for enc in InputEncoding.allCases {
            XCTAssertEqual(enc.historyFrameCount * enc.planesPerFrame + enc.tailPlaneCount,
                           enc.planeCount, "frame+tail invariant for \(enc.rawValue)")
        }
        // Isolation guard: the three pre-existing encodings have no tail.
        XCTAssertEqual(InputEncoding.basic20.tailPlaneCount, 0)
        XCTAssertEqual(InputEncoding.basic30.tailPlaneCount, 0)
        XCTAssertEqual(InputEncoding.full10ply200.tailPlaneCount, 0)
    }

    /// Bake-in composition: planes 0–199 equal the full10ply200 encoding of the
    /// same position; planes 200–209 equal basic30's planes 20–29.
    func testBakeInComposition() throws {
        let engine = ChessGameEngine()
        for m in Self.knightCycle + Self.knightCycle { _ = try engine.applyMoveAndAdvance(m) }
        let state = engine.state
        let history = engine.recentStates

        let full210 = BoardEncoder.encode(state, history: history, encoding: .full10Ply10Reps210)
        let full200 = BoardEncoder.encode(state, history: history, encoding: .full10ply200)
        let basic30 = BoardEncoder.encode(state, encoding: .basic30)

        XCTAssertEqual(full210.count, 210 * 64)
        XCTAssertEqual(Array(full210[0 ..< 200 * 64]), full200,
                       "planes 0–199 must equal the full10ply200 encoding")
        XCTAssertEqual(Array(full210[200 * 64 ..< 210 * 64]),
                       Array(basic30[20 * 64 ..< 30 * 64]),
                       "tail planes 200–209 must equal basic30 planes 20–29")
        // The double knight cycle repeats the start position 4 and 8 plies ago.
        XCTAssertEqual(sumPlane(full210, 203), 64.0, "plane 203 (4 plies ago) all-1")
        XCTAssertEqual(sumPlane(full210, 207), 64.0, "plane 207 (8 plies ago) all-1")
    }

    // MARK: - Correctness gate: buffer recompute == bake-in (== basic30 mask)

    func test_reconstruction_with_recomputed_reps_equals_bakeIn() {
        let captured = playKnightCycleGame()
        XCTAssertGreaterThan(captured.count, 8)

        let workerId: UInt16 = 7
        let gameIndex: UInt32 = 1
        let buffer = ReplayBuffer(capacity: 1024, inputEncoding: .full10Ply10Reps210)
        // Stored frame is the 20-plane basic20 block — identical to full10ply200.
        XCTAssertEqual(buffer.floatsPerBoard, BoardEncoder.tensorLength(for: .basic20))
        XCTAssertEqual(buffer.reconstructedStride,
                       BoardEncoder.tensorLength(for: .full10Ply10Reps210))
        appendGame(captured, to: buffer, whiteOutcome: 0.0,
                   workerId: workerId, gameIndex: gameIndex)

        let packedId = ReplayBuffer.packWorkerGameId(workerId: workerId, gameIndex: gameIndex)
        let stride = BoardEncoder.tensorLength(for: .full10Ply10Reps210)

        var sawNonzeroRepTail = false
        for p in captured {
            let expected = BoardEncoder.encode(
                p.state, history: p.history, encoding: .full10Ply10Reps210)
            var got = [Float](repeating: .nan, count: stride)
            let ok = got.withUnsafeMutableBufferPointer { buf -> Bool in
                buffer.reconstructStack(forWorkerGameId: packedId,
                                        plyIndex: UInt16(p.gameTotalPly),
                                        into: buf.baseAddress!)
            }
            XCTAssertTrue(ok, "no resident slot for ply \(p.gameTotalPly)")
            XCTAssertEqual(got, expected,
                "recomputed reconstruction != bake-in at ply \(p.gameTotalPly) (mover=\(p.mover))")
            if Array(expected[200 * 64 ..< 210 * 64]).contains(where: { $0 != 0 }) {
                sawNonzeroRepTail = true
            }
        }
        XCTAssertTrue(sawNonzeroRepTail,
            "test must exercise at least one non-zero recomputed rep tail")
    }

    /// Isolation: a `full10ply200` buffer is byte-for-byte unchanged — no tail
    /// applied, reconstruction == bake-in exactly as before.
    func test_full10ply200_buffer_unaffected() {
        let captured = playKnightCycleGame()
        let buffer = ReplayBuffer(capacity: 1024, inputEncoding: .full10ply200)
        appendGame(captured, to: buffer, whiteOutcome: 0.0, workerId: 9, gameIndex: 1)
        let packedId = ReplayBuffer.packWorkerGameId(workerId: 9, gameIndex: 1)
        let stride = BoardEncoder.tensorLength(for: .full10ply200)
        for p in captured {
            let expected = BoardEncoder.encode(
                p.state, history: p.history, encoding: .full10ply200)
            var got = [Float](repeating: .nan, count: stride)
            _ = got.withUnsafeMutableBufferPointer { buf in
                buffer.reconstructStack(forWorkerGameId: packedId,
                                        plyIndex: UInt16(p.gameTotalPly),
                                        into: buf.baseAddress!)
            }
            XCTAssertEqual(got, expected,
                "full10ply200 reconstruction changed at ply \(p.gameTotalPly)")
        }
    }

    // MARK: - Game capture + buffer population (mirrors ReplayHistoryReconstructionTests)

    private struct CapturedPly {
        let gameTotalPly: Int
        let mover: PieceColor
        let state: GameState
        let history: [GameState]
    }

    /// Nf3 Nc6 Ng1 Nb8 — returns to the starting position after 4 plies.
    private static let knightCycle: [ChessMove] = [
        ChessMove(fromRow: 7, fromCol: 6, toRow: 5, toCol: 5, promotion: nil), // Nf3
        ChessMove(fromRow: 0, fromCol: 1, toRow: 2, toCol: 2, promotion: nil), // Nc6
        ChessMove(fromRow: 5, fromCol: 5, toRow: 7, toCol: 6, promotion: nil), // Ng1
        ChessMove(fromRow: 2, fromCol: 2, toRow: 0, toCol: 1, promotion: nil), // Nb8
    ]

    /// Two full knight cycles (8 plies): the start position recurs 4 and 8
    /// plies ago, so ply 8's mask has bits 3 and 7 set (planes 203, 207).
    private func playKnightCycleGame() -> [CapturedPly] {
        let engine = ChessGameEngine()
        var captured: [CapturedPly] = []
        func cap(_ ply: Int) {
            captured.append(CapturedPly(gameTotalPly: ply, mover: engine.state.currentPlayer,
                                        state: engine.state, history: engine.recentStates))
        }
        cap(0)
        var ply = 0
        for m in Self.knightCycle + Self.knightCycle {
            do { _ = try engine.applyMoveAndAdvance(m) }
            catch { XCTFail("applyMoveAndAdvance threw at ply \(ply): \(error)"); break }
            ply += 1
            cap(ply)
        }
        return captured
    }

    /// Stored single frame = frame 0 of the inference stack = the basic20
    /// encode of the position (identical to full10ply200's stored frame).
    private func storedFrame(for ply: CapturedPly) -> [Float] {
        BoardEncoder.encode(ply.state, encoding: .basic20)
    }

    /// Append one game as `ActiveGame.flush` does: one merged block in reverse
    /// game-ply order, per-row outcomes, stride = the buffer's stored stride.
    private func appendGame(_ captured: [CapturedPly], to buffer: ReplayBuffer,
                            whiteOutcome: Float, workerId: UInt16, gameIndex: UInt32) {
        let total = captured.count
        guard total > 0 else { return }
        let stride = buffer.floatsPerBoard
        var boards = [Float](repeating: 0, count: total * stride)
        var policy = [Int32](repeating: 0, count: total)
        var plies = [UInt16](repeating: 0, count: total)
        var taus = [Float](repeating: 1.0, count: total)
        var hashes = [UInt64](repeating: 0, count: total)
        var materials = [UInt8](repeating: 0, count: total)
        var outcomes = [Float](repeating: 0, count: total)
        for dst in 0..<total {
            let p = captured[total - 1 - dst]
            let frame = storedFrame(for: p)
            XCTAssertEqual(frame.count, stride, "stored frame stride mismatch")
            for k in 0..<stride { boards[dst * stride + k] = frame[k] }
            policy[dst] = Int32(p.gameTotalPly)
            plies[dst] = UInt16(p.gameTotalPly)
            hashes[dst] = frame.withUnsafeBufferPointer {
                ReplayBuffer.hashBoard($0.baseAddress!, count: stride)
            }
            outcomes[dst] = (p.mover == .white) ? whiteOutcome : -whiteOutcome
        }
        boards.withUnsafeBufferPointer { b in
        policy.withUnsafeBufferPointer { pi in
        plies.withUnsafeBufferPointer { pl in
        taus.withUnsafeBufferPointer { t in
        hashes.withUnsafeBufferPointer { h in
        materials.withUnsafeBufferPointer { m in
        outcomes.withUnsafeBufferPointer { o in
            buffer.append(
                boards: b.baseAddress!, policyIndices: pi.baseAddress!,
                plyIndices: pl.baseAddress!, samplingTaus: t.baseAddress!,
                stateHashes: h.baseAddress!, materialCounts: m.baseAddress!,
                gameLength: UInt16(total), workerId: workerId,
                intraWorkerGameIndex: gameIndex, outcomes: o.baseAddress!,
                count: total)
        }}}}}}}
    }

    private func sumPlane(_ tensor: [Float], _ plane: Int) -> Float {
        var s: Float = 0
        for i in (plane * 64)..<(plane * 64 + 64) { s += tensor[i] }
        return s
    }
}
