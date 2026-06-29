import XCTest
@testable import DrewsChessMachine

/// Behavioral probe for the repetition planes (input planes 18..29).
///
/// These are SELF-CONTAINED, repeatable tests: each builds a fresh basic30
/// network in-test (no on-disk champion) and asserts the probe MACHINERY
/// produces complete, finite output for every position/variant, then writes
/// a report for inspection. The exact magnitudes are not asserted (random
/// weights), so the same probe can be pointed at a trained champion manually
/// to answer the research question below — but the TEST contract is the
/// structural one, which is deterministic and champion-independent.
///
/// Research question (manual use): with a trained network, does the value
/// head's W/D/L distribution AND the policy softmax shift when the repetition
/// planes (18..19 = "this position has been seen N times before" count flags;
/// 20..29 = "i plies ago was a strict-rule duplicate" temporal mask) are
/// flipped from "fresh position" to "this position is about to repeat" while
/// the rest of the board encoding is held constant?
///
/// If yes (large policy KL and/or large value shift, especially
/// toward draw in the pattern that signals threefold imminence),
/// the network has learned to read those planes.
///
/// If no (KL ≈ 0, Δ(p_win − p_loss) ≈ 0 across all positions and
/// patterns), the network is effectively ignoring those planes.
///
/// This is NOT a generated unit-test pass/fail check. It's a
/// diagnostic that writes its full output to a known path so the
/// outer harness can read it. The test itself always passes; the
/// signal is the file contents.
final class RepPlaneProbeTests: XCTestCase {

    /// Where the probe report lands. Picked up after the test runs.
    private let outputPath = "/tmp/rep_plane_probe_results.txt"

    func test_probeRepetitionPlanes_writeReport() async throws {
        // Self-contained: build the probe network in-test so the result is
        // repeatable and independent of any on-disk champion. The assertions
        // below verify the probe's OUTPUT data (completeness + finiteness),
        // then the report is written for the curious.
        let mps = try makeProbeNetwork()

        // ----- 3. Build a diverse set of probe positions by walking
        // a few random self-play games from .starting and snapshotting
        // every K plies. Deterministic seed for repeatability.
        let probeStates = buildProbeStates(targetCount: 24, seed: 0xC0FFEE)
        XCTAssertGreaterThan(probeStates.count, 0)

        // ----- 4. For each position, encode once normally, then
        // build the variant patterns by copying + overwriting just
        // the rep planes (indices 18..29 inclusive). All other bytes
        // are identical, so any difference in network output is
        // entirely due to the rep-plane bits.
        let perBoard = BoardEncoder.tensorLength(for: .basic30)
        XCTAssertEqual(perBoard, 30 * 64)
        let planeFloats = 64

        var report = ProbeReport()
        report.modelPath = Self.probeNetLabel
        report.modelID = Self.probeNetLabel
        report.numPositions = probeStates.count

        for (posIdx, state) in probeStates.enumerated() {
            var base = [Float](repeating: 0, count: perBoard)
            base.withUnsafeMutableBufferPointer { buf in
                BoardEncoder.encode(state, into: buf, encoding: .basic30)
            }
            // Force planes 18..29 to zero so the baseline is
            // "fresh position, no history." This matters for
            // positions that we sampled mid-game where the
            // GameState might have non-zero rep tracking already.
            for plane in 18...29 {
                for j in 0..<planeFloats {
                    base[plane * planeFloats + j] = 0
                }
            }

            // Evaluate the baseline.
            let baseOut = try await evalBoth(network: mps.network, board: base)

            // Patterns we test, all IN-DISTRIBUTION (= combinations the
            // network has actually seen in training data per the full-
            // buffer analysis in tools/full_buffer_analysis.py):
            //   - rep18 alone: 0.36% of buffer (seen long-ago)
            //   - rep18 + mask t-4: 3.73% (4-ply knight shuffle)
            //   - rep18 + mask t-8: 0.20% (8-ply rotation)
            //
            // Plane 19 and odd-ply masks (t-1, t-3, t-5, t-7, t-9) are
            // mathematically/structurally impossible to fire in real
            // games — testing them measures OOD response, not learned
            // behavior, so they are deliberately excluded.
            let patterns: [(name: String, modify: (inout [Float]) -> Void)] = [
                (
                    name: "rep18=1 ONLY (seen long-ago, no recent cycle)",
                    modify: { b in
                        setPlane(&b, index: 18, value: 1)
                    }
                ),
                (
                    name: "rep18 + mask t-4 (4-ply knight shuffle — 3.73% of buffer)",
                    modify: { b in
                        setPlane(&b, index: 18, value: 1)
                        setPlane(&b, index: 23, value: 1)
                    }
                ),
                (
                    name: "rep18 + mask t-8 (8-ply rotation — 0.20% of buffer)",
                    modify: { b in
                        setPlane(&b, index: 18, value: 1)
                        setPlane(&b, index: 27, value: 1)
                    }
                ),
            ]

            var perPos = ProbeReport.PositionResult(ply: posIdx, fen: shortFenForReport(state))
            perPos.baselineValueWDL = baseOut.wdl
            perPos.baselineValueScalar = baseOut.wdl.win - baseOut.wdl.loss
            perPos.baselineTopMoveLogit = baseOut.policy.max() ?? 0

            for pat in patterns {
                var variant = base
                pat.modify(&variant)
                let varOut = try await evalBoth(network: mps.network, board: variant)
                let kl = klDivergence(from: softmax(baseOut.policy), to: softmax(varOut.policy))
                let dWin = varOut.wdl.win - baseOut.wdl.win
                let dDraw = varOut.wdl.draw - baseOut.wdl.draw
                let dLoss = varOut.wdl.loss - baseOut.wdl.loss
                let dScalar = (varOut.wdl.win - varOut.wdl.loss) - (baseOut.wdl.win - baseOut.wdl.loss)
                perPos.variants.append(.init(
                    name: pat.name,
                    klPolicy: kl,
                    dWin: dWin,
                    dDraw: dDraw,
                    dLoss: dLoss,
                    dScalar: dScalar
                ))
            }
            report.positions.append(perPos)
        }

        // ----- 5. Check the data -----
        XCTAssertEqual(report.positions.count, probeStates.count,
                       "every probe position must produce a result")
        let variantCount = report.positions.first?.variants.count ?? 0
        XCTAssertGreaterThan(variantCount, 0, "each position must probe at least one variant")
        for pos in report.positions {
            let w = pos.baselineValueWDL
            XCTAssertTrue(w.win.isFinite && w.draw.isFinite && w.loss.isFinite,
                          "baseline WDL must be finite at ply \(pos.ply)")
            XCTAssertEqual(Double(w.win + w.draw + w.loss), 1.0, accuracy: 1e-3,
                           "WDL softmax must sum to 1 at ply \(pos.ply)")
            XCTAssertEqual(pos.variants.count, variantCount,
                           "every position must probe the same variants")
            for v in pos.variants {
                XCTAssertTrue(v.klPolicy.isFinite && v.klPolicy >= 0,
                              "policy KL must be finite and non-negative (\(v.name))")
                XCTAssertTrue(v.dWin.isFinite && v.dDraw.isFinite && v.dLoss.isFinite && v.dScalar.isFinite,
                              "value deltas must be finite (\(v.name))")
            }
        }

        // ----- 6. Write the report, then verify the file on disk -----
        let text = report.formatted()
        try text.write(toFile: outputPath, atomically: true, encoding: .utf8)
        let readback = try String(contentsOfFile: outputPath, encoding: .utf8)
        XCTAssertFalse(readback.isEmpty, "report file must be non-empty")
        XCTAssertTrue(readback.contains("probe positions: \(probeStates.count)"),
                      "report file must record the probed position count")
        print(text)   // Echo so it lands in xcresult logs too.
    }

    // MARK: - Constructed knight-shuffle endgame probe

    /// Builds a real K+N+3P vs K+N+3P endgame, plays the 4-ply knight
    /// shuffle Nb1-c3 Nb8-c6 Nc3-b1 Nc6-b8 through ChessGameEngine so
    /// the rep state is real (rep_count=1, mask t-4 set), then asks the
    /// network: how does this position score WITH the rep planes set
    /// (real) vs zeroed (counterfactual)? Tests behavior on a concrete
    /// in-distribution rep pattern in the kind of position where it
    /// would naturally arise (low-material drawish endgame), not a
    /// random opening position.
    func test_constructedKnightShuffleEndgame_probe() async throws {
        let mps = try makeProbeNetwork()

        // Build the endgame position.
        //   row 0 = rank 8 … row 7 = rank 1, col 0 = a-file.
        // White: K@g1, N@b1, P@a2,g2,h2
        // Black: K@g8, N@b8, P@a7,g7,h7
        var board: [Piece?] = Array(repeating: nil, count: 64)
        board[7 * 8 + 6] = Piece(type: .king,   color: .white)  // g1
        board[7 * 8 + 1] = Piece(type: .knight, color: .white)  // b1
        board[6 * 8 + 0] = Piece(type: .pawn,   color: .white)  // a2
        board[6 * 8 + 6] = Piece(type: .pawn,   color: .white)  // g2
        board[6 * 8 + 7] = Piece(type: .pawn,   color: .white)  // h2
        board[0 * 8 + 6] = Piece(type: .king,   color: .black)  // g8
        board[0 * 8 + 1] = Piece(type: .knight, color: .black)  // b8
        board[1 * 8 + 0] = Piece(type: .pawn,   color: .black)  // a7
        board[1 * 8 + 6] = Piece(type: .pawn,   color: .black)  // g7
        board[1 * 8 + 7] = Piece(type: .pawn,   color: .black)  // h7

        let initialState = GameState(
            board: board,
            currentPlayer: .white,
            whiteKingsideCastle: false,
            whiteQueensideCastle: false,
            blackKingsideCastle: false,
            blackQueensideCastle: false,
            enPassantSquare: nil,
            halfmoveClock: 0
        )
        let engine = ChessGameEngine(state: initialState)
        let moves: [(name: String, m: ChessMove)] = [
            ("Nb1-c3", ChessMove(fromRow: 7, fromCol: 1, toRow: 5, toCol: 2, promotion: nil)),
            ("Nb8-c6", ChessMove(fromRow: 0, fromCol: 1, toRow: 2, toCol: 2, promotion: nil)),
            ("Nc3-b1", ChessMove(fromRow: 5, fromCol: 2, toRow: 7, toCol: 1, promotion: nil)),
            ("Nc6-b8", ChessMove(fromRow: 2, fromCol: 2, toRow: 0, toCol: 1, promotion: nil)),
        ]
        for (name, move) in moves {
            do {
                _ = try engine.applyMoveAndAdvance(move)
            } catch {
                XCTFail("Knight shuffle move \(name) failed: \(error)")
                return
            }
        }

        let postState = engine.state
        XCTAssertEqual(postState.repetitionCount, 1,
                       "after 4-ply knight shuffle, rep_count should be 1")
        XCTAssertEqual(postState.recentRepetitionMask, UInt16(1) << 3,
                       "after 4-ply shuffle, mask t-4 (bit 3) should be the only bit set")

        // Encode WITH rep planes (real game history) and WITHOUT (zeroed).
        let perBoard = BoardEncoder.tensorLength(for: .basic30)
        var withRep = [Float](repeating: 0, count: perBoard)
        withRep.withUnsafeMutableBufferPointer { BoardEncoder.encode(postState, into: $0, encoding: .basic30) }
        var withoutRep = withRep
        for plane in 18...29 {
            setPlane(&withoutRep, index: plane, value: 0)
        }

        let outWith    = try await evalBoth(network: mps.network, board: withRep)
        let outWithout = try await evalBoth(network: mps.network, board: withoutRep)

        let kl = klDivergence(from: softmax(outWithout.policy), to: softmax(outWith.policy))
        let dWin = outWith.wdl.win - outWithout.wdl.win
        let dDraw = outWith.wdl.draw - outWithout.wdl.draw
        let dLoss = outWith.wdl.loss - outWithout.wdl.loss
        let dScalar = (outWith.wdl.win - outWith.wdl.loss) - (outWithout.wdl.win - outWithout.wdl.loss)

        // Check the data: both evaluations valid, deltas finite, KL sane.
        for (label, o) in [("with-rep", outWith), ("without-rep", outWithout)] {
            XCTAssertTrue(o.wdl.win.isFinite && o.wdl.draw.isFinite && o.wdl.loss.isFinite,
                          "\(label) WDL must be finite")
            XCTAssertEqual(Double(o.wdl.win + o.wdl.draw + o.wdl.loss), 1.0, accuracy: 1e-3,
                           "\(label) WDL softmax must sum to 1")
        }
        XCTAssertTrue(kl.isFinite && kl >= 0, "policy KL must be finite and non-negative")
        XCTAssertTrue(dWin.isFinite && dDraw.isFinite && dLoss.isFinite && dScalar.isFinite,
                      "value deltas must be finite")

        var s = "Constructed knight-shuffle endgame probe\n"
        s += "==========================================\n"
        s += "modelID: \(Self.probeNetLabel)\n\n"
        s += "Position: white K@g1 N@b1 P@a2,g2,h2 vs black K@g8 N@b8 P@a7,g7,h7\n"
        s += "  10 pieces, white to move, in-distribution endgame material\n"
        s += "Shuffle: \(moves.map(\.name).joined(separator: " · "))\n"
        s += "  post-shuffle rep_count=\(postState.repetitionCount), "
            + "rep_mask=0b\(String(postState.recentRepetitionMask, radix: 2)) (bit 3 = mask t-4)\n\n"

        s += "Comparison: same board, rep planes zeroed vs real rep state\n"
        s += "  baseline (rep=0):  W=\(fmt(outWithout.wdl.win)) D=\(fmt(outWithout.wdl.draw)) L=\(fmt(outWithout.wdl.loss)) scalar=\(fmtS(outWithout.wdl.win - outWithout.wdl.loss))\n"
        s += "  with rep planes:   W=\(fmt(outWith.wdl.win))    D=\(fmt(outWith.wdl.draw))    L=\(fmt(outWith.wdl.loss))    scalar=\(fmtS(outWith.wdl.win - outWith.wdl.loss))\n"
        s += "  ΔWin=\(fmtS(dWin))  ΔDraw=\(fmtS(dDraw))  ΔLoss=\(fmtS(dLoss))  Δscalar=\(fmtS(dScalar))\n"
        s += "  KL(policy) = \(String(format: "%.4f", kl)) nats\n\n"

        s += "Interpretation:\n"
        s += "  Expected if network correctly learned the rep signal:\n"
        s += "    - p_draw should INCREASE when rep planes set (this is a draw shuffle)\n"
        s += "    - p_win and p_loss should both decrease\n"
        s += "    - Policy may shift to either ESCAPE the cycle (if winning) or REPEAT it (if drawing)\n"

        let knightPath = "/tmp/knight_shuffle_probe.txt"
        try s.write(toFile: knightPath, atomically: true, encoding: .utf8)
        XCTAssertFalse(try String(contentsOfFile: knightPath, encoding: .utf8).isEmpty,
                       "knight-shuffle report file must be non-empty")
        print(s)
    }

    // MARK: - Halfmove-clock sensitivity probe

    /// For each of ~20 random positions, set plane 17 (halfmove clock,
    /// normalized 0..1) to a series of test values and measure how the
    /// network's value-head WDL distribution shifts. If the network has
    /// learned the 50-move-rule signal, p_draw should rise monotonically
    /// with the halfmove value.
    func test_halfmoveProbe_writeReport() async throws {
        let mps = try makeProbeNetwork()

        let probeStates = buildProbeStates(targetCount: 20, seed: 0xBADCAFE_1234567)
        let testValues: [Float] = [0.00, 0.25, 0.50, 0.75, 0.90, 0.99]
        let perBoard = BoardEncoder.tensorLength(for: .basic30)

        struct HalfmoveAgg {
            var sumKL: Double = 0
            var maxKL: Float = 0
            var sumDWin: Double = 0
            var sumDDraw: Double = 0
            var sumDLoss: Double = 0
            var sumDScalar: Double = 0
            var n: Int = 0
        }
        var aggs: [HalfmoveAgg] = Array(repeating: HalfmoveAgg(), count: testValues.count)
        var perPosLines: [String] = []

        for (posIdx, state) in probeStates.enumerated() {
            var base = [Float](repeating: 0, count: perBoard)
            base.withUnsafeMutableBufferPointer { BoardEncoder.encode(state, into: $0, encoding: .basic30) }
            // Zero rep planes and halfmove so baseline is "fresh, no recent rep, halfmove=0"
            for plane in 17...29 {
                setPlane(&base, index: plane, value: 0)
            }
            let baseOut = try await evalBoth(network: mps.network, board: base)

            var line = "ply \(posIdx)  stm=\(state.currentPlayer == .white ? "w" : "b")  baseline WDL=(\(fmt(baseOut.wdl.win)), \(fmt(baseOut.wdl.draw)), \(fmt(baseOut.wdl.loss)))"
            for (vi, v) in testValues.enumerated() {
                var variant = base
                setPlane(&variant, index: 17, value: v)
                let varOut = try await evalBoth(network: mps.network, board: variant)
                let kl = klDivergence(from: softmax(baseOut.policy), to: softmax(varOut.policy))
                let dW = varOut.wdl.win - baseOut.wdl.win
                let dD = varOut.wdl.draw - baseOut.wdl.draw
                let dL = varOut.wdl.loss - baseOut.wdl.loss
                let dS = (varOut.wdl.win - varOut.wdl.loss) - (baseOut.wdl.win - baseOut.wdl.loss)
                aggs[vi].sumKL += Double(kl)
                aggs[vi].maxKL = max(aggs[vi].maxKL, kl)
                aggs[vi].sumDWin += Double(dW)
                aggs[vi].sumDDraw += Double(dD)
                aggs[vi].sumDLoss += Double(dL)
                aggs[vi].sumDScalar += Double(dS)
                aggs[vi].n += 1
                line += "  | hm=\(String(format: "%.2f", v)) KL=\(String(format: "%.3f", kl)) ΔD=\(fmtS(dD))"
            }
            perPosLines.append(line)
        }

        // Check the data: every value tested at every position, all finite.
        for (vi, a) in aggs.enumerated() {
            XCTAssertEqual(a.n, probeStates.count,
                           "halfmove value index \(vi) must be tested at every position")
            XCTAssertTrue(a.maxKL.isFinite && a.maxKL >= 0, "maxKL must be finite/non-negative")
            XCTAssertTrue(a.sumKL.isFinite && a.sumDWin.isFinite && a.sumDDraw.isFinite
                          && a.sumDLoss.isFinite && a.sumDScalar.isFinite,
                          "aggregated deltas must be finite at value index \(vi)")
        }

        var s = "Halfmove-clock sensitivity probe\n"
        s += "================================\n"
        s += "modelID: \(Self.probeNetLabel)\n"
        s += "probe positions: \(probeStates.count)\n"
        s += "halfmove values tested (plane 17, broadcast): \(testValues.map { String(format: "%.2f", $0) }.joined(separator: ", "))\n"
        s += "(network's halfmove encoding: min(clock, 99) / 99 → so plane value 0.50 ≈ halfmove 50, 0.99 ≈ halfmove 98)\n\n"

        s += "=== aggregated across all \(probeStates.count) positions ===\n"
        s += "  " + padR("plane value", 14) + padR("equiv. hm-clock", 16)
            + padR("meanKL", 10) + padR("maxKL", 10)
            + padR("meanΔWin", 11) + padR("meanΔDraw", 12) + padR("meanΔLoss", 11) + padR("meanΔScalar", 12) + "\n"
        for (vi, v) in testValues.enumerated() {
            let a = aggs[vi]
            let nF = Double(max(a.n, 1))
            let equiv = Int((v * 99).rounded())
            s += "  " + padR(String(format: "%.2f", v), 14)
                + padR("≈\(equiv)", 16)
                + padR(String(format: "%.4f", a.sumKL / nF), 10)
                + padR(String(format: "%.4f", a.maxKL), 10)
                + padR(String(format: "%+.4f", a.sumDWin / nF), 11)
                + padR(String(format: "%+.4f", a.sumDDraw / nF), 12)
                + padR(String(format: "%+.4f", a.sumDLoss / nF), 11)
                + padR(String(format: "%+.4f", a.sumDScalar / nF), 12) + "\n"
        }

        s += "\nInterpretation:\n"
        s += "  Healthy: meanΔDraw should INCREASE monotonically with plane value.\n"
        s += "    At hm=0.99 (≈ halfmove 98, one move from 50-move-rule draw), p_draw\n"
        s += "    should be substantially above baseline (training data: 97% draw at hm≥75).\n"
        s += "  Flat / noisy ΔDraw → network has not learned the halfmove signal well\n"
        s += "    (consistent with the rarity: only 0.04% of training has hm≥75).\n"
        s += "    Threshold planes (hm≥50, hm≥75) might help by making the signal categorical.\n\n"

        s += "=== per-position (first 10 only for brevity) ===\n"
        for line in perPosLines.prefix(10) {
            s += line + "\n"
        }

        let halfmovePath = "/tmp/halfmove_probe.txt"
        try s.write(toFile: halfmovePath, atomically: true, encoding: .utf8)
        XCTAssertFalse(try String(contentsOfFile: halfmovePath, encoding: .utf8).isEmpty,
                       "halfmove report file must be non-empty")
        print(s)
    }

    // MARK: - Helpers

    /// A short, fixed label for the self-contained probe network.
    private static let probeNetLabel = "self-contained probe (random-init basic30)"

    /// Build a fresh basic30 inference network for the probe — no on-disk
    /// champion. These probes assert that the probe MACHINERY produces
    /// complete, finite output for every position/variant: a repeatable
    /// contract that doesn't depend on whatever model happens to be on disk.
    /// (Random weights are sufficient for that contract — the magnitudes
    /// aren't asserted. "Did the *trained* champion learn plane X?" is a
    /// manual research question, not a deterministic unit test.)
    private func makeProbeNetwork() throws -> ChessMPSNetwork {
        let arch = NetworkArchitecture.preset(.v3_8block_3x3)
        precondition(arch.inputEncoding == .basic30,
                     "probe encodes basic30; the probe network must be a basic30 arch")
        return try ChessMPSNetwork(.randomWeights, arch: arch)
    }

    // MARK: - Probe-state generator

    private func buildProbeStates(targetCount: Int, seed: UInt64) -> [GameState] {
        var rng = SeededGenerator(seed: seed)
        var out: [GameState] = [.starting]
        var state: GameState = .starting
        // Snapshot every ~3 plies along a random walk. Reset on
        // terminal positions to keep generating fresh ones.
        var pliesSinceSnapshot = 0
        while out.count < targetCount {
            let moves = MoveGenerator.legalMoves(for: state)
            guard let move = moves.randomElement(using: &rng) else {
                state = .starting
                pliesSinceSnapshot = 0
                continue
            }
            state = MoveGenerator.applyMove(move, to: state)
            pliesSinceSnapshot += 1
            if pliesSinceSnapshot >= 3 {
                out.append(state)
                pliesSinceSnapshot = 0
            }
        }
        return out
    }

    // MARK: - Forward pass

    private struct EvalOut {
        let policy: [Float]                              // [4864] logits
        let wdl: (win: Float, draw: Float, loss: Float)  // softmax over the 3 logits
    }

    private func evalBoth(network: ChessNetwork, board: [Float]) async throws -> EvalOut {
        // policy via evaluate(board:consume:) — the consume closure
        // gets policy logits + the derived scalar (which we discard;
        // we re-derive from WDL ourselves).
        nonisolated(unsafe) var policyArr: [Float] = []
        try await network.evaluate(board: board) { polBuf, _ in
            policyArr = Array(polBuf)
        }
        let wdl = try await network.evaluateValueDistribution(board: board)
        return EvalOut(policy: policyArr, wdl: wdl)
    }
}

// MARK: - Free helpers

private func setPlane(_ board: inout [Float], index plane: Int, value: Float) {
    let base = plane * 64
    for j in 0..<64 { board[base + j] = value }
}

private func softmax(_ x: [Float]) -> [Float] {
    let m = x.max() ?? 0
    var ex = x.map { Float(exp(Double($0 - m))) }
    let s = ex.reduce(0, +)
    if s > 0 { for i in 0..<ex.count { ex[i] /= s } }
    return ex
}

/// Symmetric-undefined KL(p || q) in nats. Skip terms with p≈0.
private func klDivergence(from p: [Float], to q: [Float]) -> Float {
    precondition(p.count == q.count)
    var acc: Double = 0
    let eps: Double = 1e-12
    for i in 0..<p.count {
        let pi = Double(p[i])
        if pi <= eps { continue }
        let qi = max(Double(q[i]), eps)
        acc += pi * (log(pi) - log(qi))
    }
    return Float(acc)
}

private func shortFenForReport(_ state: GameState) -> String {
    // Use the side-to-move + piece count as a short identifier
    // (we don't have a FEN serializer wired up here, and the
    // identity of each probe position isn't load-bearing for
    // the diagnostic — only the shift in network output is).
    let side = state.currentPlayer == .white ? "w" : "b"
    let pieces = state.board.reduce(into: 0) { $0 += ($1 == nil ? 0 : 1) }
    return "stm=\(side) pieces=\(pieces)"
}

// MARK: - Tiny seeded RNG (xorshift) for reproducible position walks.

private struct SeededGenerator: RandomNumberGenerator {
    var state: UInt64
    init(seed: UInt64) { state = seed == 0 ? 0xDEADBEEF : seed }
    mutating func next() -> UInt64 {
        state ^= state << 13
        state ^= state >> 7
        state ^= state << 17
        return state
    }
}

// MARK: - Report shape

private struct ProbeReport {
    var modelPath: String = ""
    var modelID: String = ""
    var numPositions: Int = 0
    var positions: [PositionResult] = []

    struct PositionResult {
        var ply: Int
        var fen: String
        var baselineValueWDL: (win: Float, draw: Float, loss: Float) = (0, 0, 0)
        var baselineValueScalar: Float = 0
        var baselineTopMoveLogit: Float = 0
        var variants: [VariantResult] = []
    }

    struct VariantResult {
        var name: String
        var klPolicy: Float
        var dWin: Float
        var dDraw: Float
        var dLoss: Float
        var dScalar: Float
    }

    func formatted() -> String {
        var s = ""
        s += "Repetition-plane behavioral probe\n"
        s += "=================================\n"
        s += "modelID: \(modelID)\n"
        s += "modelPath: \(modelPath)\n"
        s += "probe positions: \(numPositions)\n\n"

        // Aggregate stats per variant pattern
        let variantNames = positions.first?.variants.map(\.name) ?? []
        s += "=== aggregated across all \(positions.count) positions ===\n"
        s += "  " + padR("variant", 50)
            + " " + padL("meanKL", 12)
            + " " + padL("maxKL", 12)
            + " " + padL("meanΔWin", 12)
            + " " + padL("meanΔDraw", 12)
            + " " + padL("meanΔScalar", 12)
            + "\n"
        for (i, name) in variantNames.enumerated() {
            var sumKL: Double = 0
            var maxKL: Float = 0
            var sumDW: Double = 0
            var sumDD: Double = 0
            var sumDS: Double = 0
            var n = 0
            for pos in positions where i < pos.variants.count {
                let v = pos.variants[i]
                sumKL += Double(v.klPolicy)
                maxKL = max(maxKL, v.klPolicy)
                sumDW += Double(v.dWin)
                sumDD += Double(v.dDraw)
                sumDS += Double(v.dScalar)
                n += 1
            }
            let nF = Double(max(n, 1))
            s += "  " + padR(name, 50)
                + " " + padL(String(format: "%.4e", sumKL / nF), 12)
                + " " + padL(String(format: "%.4e", maxKL), 12)
                + " " + padL(String(format: "%+.4f", sumDW / nF), 12)
                + " " + padL(String(format: "%+.4f", sumDD / nF), 12)
                + " " + padL(String(format: "%+.4f", sumDS / nF), 12)
                + "\n"
        }

        s += "\n=== per-position detail ===\n"
        for pos in positions {
            let wdl = pos.baselineValueWDL
            s += "ply " + padL(String(pos.ply), 3) + "  " + pos.fen
                + "  baseline WDL=(" + String(format: "%.3f", wdl.win)
                + ", " + String(format: "%.3f", wdl.draw)
                + ", " + String(format: "%.3f", wdl.loss)
                + ") scalar=" + String(format: "%+.3f", pos.baselineValueScalar) + "\n"
            for v in pos.variants {
                s += "    " + padR(v.name, 48)
                    + "  KL=" + String(format: "%.4e", v.klPolicy)
                    + "  ΔW=" + String(format: "%+.4f", v.dWin)
                    + "  ΔD=" + String(format: "%+.4f", v.dDraw)
                    + "  ΔL=" + String(format: "%+.4f", v.dLoss)
                    + "  Δscalar=" + String(format: "%+.4f", v.dScalar) + "\n"
            }
        }
        return s
    }
}

/// Left-align: pad string on the right with spaces to reach `len`.
/// Truncates if already longer.
private func padR(_ s: String, _ len: Int) -> String {
    if s.count >= len { return s }
    return s + String(repeating: " ", count: len - s.count)
}

/// Right-align: pad string on the left with spaces to reach `len`.
/// Truncates the leading side if already longer.
private func padL(_ s: String, _ len: Int) -> String {
    if s.count >= len { return s }
    return String(repeating: " ", count: len - s.count) + s
}

private func fmt(_ x: Float) -> String { String(format: "%.3f", x) }
private func fmtS(_ x: Float) -> String { String(format: "%+.3f", x) }
