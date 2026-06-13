import XCTest
@testable import DrewsChessMachine

/// Bit-exact tests for the replay-buffer history-reconstruction
/// optimization (store each ply once as its single mover-relative frame;
/// stack the `historyFrameCount`-frame network input at sample time).
///
/// These are pure-logic and Metal-free: they drive a deterministic game
/// through `ChessGameEngine`, store the single mover-relative frames into a
/// `ReplayBuffer` constructed for `.full10ply200` exactly the way
/// `ActiveGame.flush` does (one merged, reverse-game-ply-ordered append
/// with per-row outcomes), then assert that the buffer's reconstruction of
/// each position equals the on-the-fly bake-in
/// `BoardEncoder.encode(state, history:…, encoding: .full10ply200)` for
/// that ply — for BOTH white-to-move and black-to-move positions, and at
/// game start (where leading frames must be zero).
///
/// A `basic30` case guards the single-frame regression: a stored position
/// must round-trip identically (gather-of-1 == stored frame).
final class ReplayHistoryReconstructionTests: XCTestCase {

    // MARK: - Deterministic game capture

    /// One captured position: the engine state and history window AS THEY
    /// STOOD when it was that mover's turn (before the move was applied), so
    /// the bake-in can be recomputed independently of the buffer.
    private struct CapturedPly {
        let gameTotalPly: Int
        let mover: PieceColor
        let state: GameState
        let history: [GameState]
    }

    /// Play a deterministic game (first legal move by a stable ordering) for
    /// up to `maxPlies` plies, capturing each ply's `(state, history)`.
    private func playDeterministicGame(maxPlies: Int) -> [CapturedPly] {
        let engine = ChessGameEngine()
        var captured: [CapturedPly] = []
        var ply = 0
        while ply < maxPlies {
            guard engine.result == nil else { break }
            let legal = engine.currentLegalMoves
            guard let move = legal.sorted(by: Self.moveOrder).first else { break }
            captured.append(CapturedPly(
                gameTotalPly: ply,
                mover: engine.state.currentPlayer,
                state: engine.state,
                history: engine.recentStates
            ))
            do {
                _ = try engine.applyMoveAndAdvance(move)
            } catch {
                XCTFail("applyMoveAndAdvance threw at ply \(ply): \(error)")
                break
            }
            ply += 1
        }
        return captured
    }

    /// Stable total ordering on moves so the game is reproducible.
    private static func moveOrder(_ a: ChessMove, _ b: ChessMove) -> Bool {
        let ka = [a.fromRow, a.fromCol, a.toRow, a.toCol, a.promotion?.rawValue ?? -1]
        let kb = [b.fromRow, b.fromCol, b.toRow, b.toCol, b.promotion?.rawValue ?? -1]
        for (x, y) in zip(ka, kb) where x != y { return x < y }
        return false
    }

    // MARK: - Buffer population mimicking ActiveGame.flush

    /// The single mover-relative stored frame for a position = frame 0 of
    /// the inference stack = the plain `basic20` encode of that state from
    /// its own mover's perspective.
    private func storedFrame(for ply: CapturedPly) -> [Float] {
        BoardEncoder.encode(ply.state, encoding: .basic20)
    }

    /// Append one whole game to `buffer` exactly as `ActiveGame.flush`
    /// does: one merged block in reverse game-ply order (last ply first,
    /// opening ply last), per-row outcomes (white rows +whiteOutcome, black
    /// rows −whiteOutcome). Stride is the buffer's stored stride.
    private func appendGame(
        _ captured: [CapturedPly],
        to buffer: ReplayBuffer,
        whiteOutcome: Float,
        workerId: UInt16,
        gameIndex: UInt32
    ) {
        let total = captured.count
        guard total > 0 else { return }
        let stride = buffer.floatsPerBoard

        var boards = [Float](repeating: 0, count: total * stride)
        var policy = [Int32](repeating: 0, count: total)
        var plies = [UInt16](repeating: 0, count: total)
        let taus = [Float](repeating: 1.0, count: total)
        var hashes = [UInt64](repeating: 0, count: total)
        let materials = [UInt8](repeating: 0, count: total)
        var outcomes = [Float](repeating: 0, count: total)

        // dst row 0 = last ply played; dst row total-1 = opening ply.
        for dst in 0..<total {
            let p = captured[total - 1 - dst]
            let frame = storedFrame(for: p)
            XCTAssertEqual(frame.count, stride, "stored frame stride mismatch")
            for k in 0..<stride { boards[dst * stride + k] = frame[k] }
            policy[dst] = Int32(p.gameTotalPly)
            plies[dst] = UInt16(p.gameTotalPly)
            // Each frame hashed over its stored bytes, matching ActiveGame.
            hashes[dst] = frame.withUnsafeBufferPointer {
                ReplayBuffer.hashBoard($0.baseAddress!, count: stride)
            }
            outcomes[dst] = (p.mover == .white) ? whiteOutcome : -whiteOutcome
        }

        let gameLength = UInt16(total)
        boards.withUnsafeBufferPointer { b in
            policy.withUnsafeBufferPointer { pi in
                plies.withUnsafeBufferPointer { pl in
                    taus.withUnsafeBufferPointer { t in
                        hashes.withUnsafeBufferPointer { h in
                            materials.withUnsafeBufferPointer { m in
                                outcomes.withUnsafeBufferPointer { o in
                                    buffer.append(
                                        boards: b.baseAddress!,
                                        policyIndices: pi.baseAddress!,
                                        plyIndices: pl.baseAddress!,
                                        samplingTaus: t.baseAddress!,
                                        stateHashes: h.baseAddress!,
                                        materialCounts: m.baseAddress!,
                                        gameLength: gameLength,
                                        workerId: workerId,
                                        intraWorkerGameIndex: gameIndex,
                                        outcomes: o.baseAddress!,
                                        count: total
                                    )
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // MARK: - Tests

    /// Core: reconstruction == bake-in, bit-exact, for every ply of a
    /// deterministic game — covering both white-to-move and black-to-move
    /// positions and the game-start zeroing of leading history frames.
    func test_reconstruction_equals_bakeIn_for_every_ply_full10ply200() {
        let captured = playDeterministicGame(maxPlies: 16)
        XCTAssertGreaterThan(captured.count, 12,
            "need >12 plies to exercise both the zero-padded game start and a full 10-frame stack")

        let workerId: UInt16 = 7
        let gameIndex: UInt32 = 1
        // Capacity comfortably larger than the game so nothing is evicted.
        let buffer = ReplayBuffer(capacity: 1024, inputEncoding: .full10ply200)
        appendGame(captured, to: buffer, whiteOutcome: 1.0,
                   workerId: workerId, gameIndex: gameIndex)

        let packedId = ReplayBuffer.packWorkerGameId(workerId: workerId, gameIndex: gameIndex)
        let stride = BoardEncoder.tensorLength(for: .full10ply200)

        var sawWhite = false
        var sawBlack = false
        var sawZeroPadded = false
        var sawFullStack = false

        for p in captured {
            // Bake-in: the network input the on-the-fly encoder would build.
            let expected = BoardEncoder.encode(
                p.state, history: p.history, encoding: .full10ply200)
            XCTAssertEqual(expected.count, stride)

            var got = [Float](repeating: .nan, count: stride)
            let ok = got.withUnsafeMutableBufferPointer { buf -> Bool in
                buffer.reconstructStack(
                    forWorkerGameId: packedId,
                    plyIndex: UInt16(p.gameTotalPly),
                    into: buf.baseAddress!)
            }
            XCTAssertTrue(ok, "no resident slot for ply \(p.gameTotalPly)")

            XCTAssertEqual(got, expected,
                "reconstruction != bake-in at ply \(p.gameTotalPly) (mover=\(p.mover))")

            if p.mover == .white { sawWhite = true } else { sawBlack = true }
            // A ply with fewer than 9 priors must have trailing zero frames;
            // verify at least the deepest history frame is zero for an early
            // ply and non-trivial for a deep ply.
            let lastFrameBase = (10 - 1) * 20 * 64
            let lastFrameAllZero = (0..<(20 * 64)).allSatisfy { expected[lastFrameBase + $0] == 0 }
            if p.gameTotalPly < 9 { if lastFrameAllZero { sawZeroPadded = true } }
            if p.gameTotalPly >= 9 { if !lastFrameAllZero { sawFullStack = true } }
        }

        XCTAssertTrue(sawWhite, "test must cover a white-to-move position")
        XCTAssertTrue(sawBlack, "test must cover a black-to-move position")
        XCTAssertTrue(sawZeroPadded, "test must cover a zero-padded game-start position")
        XCTAssertTrue(sawFullStack, "test must cover a fully-populated 10-frame stack")
    }

    /// Game-start zeroing: the opening ply (ply 0) has NO priors, so frames
    /// 1…9 must all be zero, and frame 0 must equal the stored single frame.
    func test_gameStart_leadingZeroFrames_full10ply200() {
        let captured = playDeterministicGame(maxPlies: 6)
        XCTAssertGreaterThan(captured.count, 0)

        let workerId: UInt16 = 3
        let gameIndex: UInt32 = 9
        let buffer = ReplayBuffer(capacity: 256, inputEncoding: .full10ply200)
        appendGame(captured, to: buffer, whiteOutcome: 0.0,
                   workerId: workerId, gameIndex: gameIndex)

        let packedId = ReplayBuffer.packWorkerGameId(workerId: workerId, gameIndex: gameIndex)
        let stride = BoardEncoder.tensorLength(for: .full10ply200)
        let frameFloats = 20 * 64

        var got = [Float](repeating: .nan, count: stride)
        let ok = got.withUnsafeMutableBufferPointer { buf -> Bool in
            buffer.reconstructStack(forWorkerGameId: packedId, plyIndex: 0, into: buf.baseAddress!)
        }
        XCTAssertTrue(ok)

        // Frame 0 == the stored opening-position single frame.
        let openingFrame = storedFrame(for: captured[0])
        for k in 0..<frameFloats {
            XCTAssertEqual(got[k], openingFrame[k], "frame 0 mismatch at \(k)")
        }
        // Frames 1…9 all zero.
        for f in 1..<10 {
            for k in 0..<frameFloats {
                XCTAssertEqual(got[f * frameFloats + k], 0,
                    "expected zero in absent frame \(f) at \(k)")
            }
        }
    }

    /// Eviction / absent prior: if a position's chronological prior is not
    /// resident (evicted by the ring), reconstruction must zero from the
    /// first absent frame onward. We force this by storing ONLY a suffix of
    /// the game (the priors of the earliest stored ply are absent), then
    /// reconstructing that earliest stored ply.
    func test_absentPrior_zeroesFromThatFrameOnward_full10ply200() {
        let captured = playDeterministicGame(maxPlies: 14)
        XCTAssertGreaterThan(captured.count, 12)

        // Store only plies [4 ... end] of the game, in the same reverse
        // merged order flush would use for that sub-block. The earliest
        // stored ply (gameTotalPly == 4) then has its priors (plies 0–3)
        // absent, so reconstructing it must keep frame 0..(N) valid up to
        // where the prior exists, then zero the rest.
        let suffix = Array(captured[4...])
        let workerId: UInt16 = 11
        let gameIndex: UInt32 = 2
        let buffer = ReplayBuffer(capacity: 512, inputEncoding: .full10ply200)
        appendGame(suffix, to: buffer, whiteOutcome: 1.0,
                   workerId: workerId, gameIndex: gameIndex)

        let packedId = ReplayBuffer.packWorkerGameId(workerId: workerId, gameIndex: gameIndex)
        let stride = BoardEncoder.tensorLength(for: .full10ply200)
        let frameFloats = 20 * 64

        let earliest = suffix[0]                // gameTotalPly == 4
        var got = [Float](repeating: .nan, count: stride)
        let ok = got.withUnsafeMutableBufferPointer { buf -> Bool in
            buffer.reconstructStack(
                forWorkerGameId: packedId,
                plyIndex: UInt16(earliest.gameTotalPly),
                into: buf.baseAddress!)
        }
        XCTAssertTrue(ok)

        // The earliest stored ply has gameTotalPly == 4 and sits at the top
        // of the reverse-ordered block, so the slot one forward (its f=1
        // prior, ply 3) is NOT resident. Frame 0 must be valid; frames 1…9
        // must be zero because the f=1 prior is absent (the gather stops at
        // the first invalid frame and zeroes from there).
        let openingFrame = storedFrame(for: earliest)
        for k in 0..<frameFloats {
            XCTAssertEqual(got[k], openingFrame[k], "frame 0 mismatch at \(k)")
        }
        for f in 1..<10 {
            for k in 0..<frameFloats {
                XCTAssertEqual(got[f * frameFloats + k], 0,
                    "expected zero in absent-prior frame \(f) at \(k)")
            }
        }
    }

    /// basic30 single-frame regression: a stored position round-trips
    /// identically through reconstruction (gather-of-1 == stored frame),
    /// and the stored/reconstructed strides are equal (no decoupling for
    /// single-frame encodings).
    func test_basic30_singleFrame_roundTrips_identically() {
        let captured = playDeterministicGame(maxPlies: 8)
        XCTAssertGreaterThan(captured.count, 4)

        let workerId: UInt16 = 5
        let gameIndex: UInt32 = 1
        let buffer = ReplayBuffer(capacity: 256, inputEncoding: .basic30)
        XCTAssertEqual(buffer.floatsPerBoard, BoardEncoder.tensorLength(for: .basic30))
        XCTAssertEqual(buffer.reconstructedStride, buffer.floatsPerBoard,
            "single-frame encoding must not decouple stored/reconstructed stride")

        // For basic30 the stored frame is the full basic30 encode (single
        // frame). Mimic flush with that stride.
        let total = captured.count
        let stride = buffer.floatsPerBoard
        var boards = [Float](repeating: 0, count: total * stride)
        let policy = [Int32](repeating: 0, count: total)
        var plies = [UInt16](repeating: 0, count: total)
        let taus = [Float](repeating: 1.0, count: total)
        var hashes = [UInt64](repeating: 0, count: total)
        let materials = [UInt8](repeating: 0, count: total)
        var outcomes = [Float](repeating: 0, count: total)
        var perPlyExpected: [Int: [Float]] = [:]
        for dst in 0..<total {
            let p = captured[total - 1 - dst]
            let frame = BoardEncoder.encode(p.state, history: p.history, encoding: .basic30)
            perPlyExpected[p.gameTotalPly] = frame
            for k in 0..<stride { boards[dst * stride + k] = frame[k] }
            plies[dst] = UInt16(p.gameTotalPly)
            hashes[dst] = frame.withUnsafeBufferPointer {
                ReplayBuffer.hashBoard($0.baseAddress!, count: stride)
            }
            outcomes[dst] = (p.mover == .white) ? 1.0 : -1.0
        }
        boards.withUnsafeBufferPointer { b in
            policy.withUnsafeBufferPointer { pi in
                plies.withUnsafeBufferPointer { pl in
                    taus.withUnsafeBufferPointer { t in
                        hashes.withUnsafeBufferPointer { h in
                            materials.withUnsafeBufferPointer { m in
                                outcomes.withUnsafeBufferPointer { o in
                                    buffer.append(
                                        boards: b.baseAddress!,
                                        policyIndices: pi.baseAddress!,
                                        plyIndices: pl.baseAddress!,
                                        samplingTaus: t.baseAddress!,
                                        stateHashes: h.baseAddress!,
                                        materialCounts: m.baseAddress!,
                                        gameLength: UInt16(total),
                                        workerId: workerId,
                                        intraWorkerGameIndex: gameIndex,
                                        outcomes: o.baseAddress!,
                                        count: total
                                    )
                                }
                            }
                        }
                    }
                }
            }
        }

        let packedId = ReplayBuffer.packWorkerGameId(workerId: workerId, gameIndex: gameIndex)
        for p in captured {
            var got = [Float](repeating: .nan, count: stride)
            let ok = got.withUnsafeMutableBufferPointer { buf -> Bool in
                buffer.reconstructStack(
                    forWorkerGameId: packedId,
                    plyIndex: UInt16(p.gameTotalPly),
                    into: buf.baseAddress!)
            }
            XCTAssertTrue(ok)
            XCTAssertEqual(got, perPlyExpected[p.gameTotalPly],
                "basic30 single-frame round-trip mismatch at ply \(p.gameTotalPly)")
        }
    }
}
