//
//  ReplayBufferSamplingConstraintsTests.swift
//  DrewsChessMachineTests
//
//  Covers the composition-aware batch sampler added to `ReplayBuffer`:
//   - no-op constraints ⇒ a valid uniform batch (and `compositionSnapshot`
//     reflects the appended games)
//   - `maxDrawPercent` is a hard ceiling (and freed slots go to decisive)
//   - `maxPerGame` is a hard per-batch cap (no game over the cap in a batch)
//   - `targetMeanGameLengthPlies` pulls the position-weighted sampled mean
//     down toward the target; 0 / a target above the natural mean is a no-op
//   - the composition aggregates survive a write→restore round-trip
//   - empty / under-filled buffer behaves as before
//

import XCTest
@testable import DrewsChessMachine

final class ReplayBufferSamplingConstraintsTests: XCTestCase {

    private var tempFile: URL!

    override func setUpWithError() throws {
        tempFile = FileManager.default.temporaryDirectory
            .appendingPathComponent("dcm-replay-sampling-test-\(UUID().uuidString).bin")
    }
    override func tearDownWithError() throws {
        if let p = tempFile?.path, FileManager.default.fileExists(atPath: p) {
            try? FileManager.default.removeItem(at: tempFile)
        }
    }

    // MARK: - Helpers

    /// Append one synthetic game of `length` positions with the given
    /// outcome (+1 / 0 / -1) and identity. Board content is filler — only
    /// the metadata (game length, outcome, worker/game id) matters here.
    private func appendGame(
        to buffer: ReplayBuffer, length: Int, outcome: Float,
        workerId: UInt16, gameIndex: UInt32
    ) {
        precondition(length > 0)
        let fpb = ReplayBuffer.floatsPerBoard
        var boards = [Float](repeating: 0, count: length * fpb)
        // Make boards distinct so the hash dict doesn't collapse them.
        for i in 0..<length { boards[i * fpb] = Float(workerId) * 1e6 + Float(gameIndex) * 1e3 + Float(i) }
        var moves = [Int32](repeating: 0, count: length)
        for i in 0..<length { moves[i] = Int32(i) }
        var plies = [UInt16](repeating: 0, count: length)
        for i in 0..<length { plies[i] = UInt16(min(i, Int(UInt16.max))) }
        let taus = [Float](repeating: 1.0, count: length)
        var hashes = [UInt64](repeating: 0, count: length)
        for i in 0..<length { hashes[i] = (UInt64(workerId) << 40) | (UInt64(gameIndex) << 16) | UInt64(i & 0xFFFF) }
        let mats = [UInt8](repeating: 16, count: length)
        boards.withUnsafeBufferPointer { b in
        moves.withUnsafeBufferPointer { m in
        plies.withUnsafeBufferPointer { pl in
        taus.withUnsafeBufferPointer { t in
        hashes.withUnsafeBufferPointer { h in
        mats.withUnsafeBufferPointer { ma in
            buffer.append(
                boards: b.baseAddress!, policyIndices: m.baseAddress!,
                plyIndices: pl.baseAddress!, samplingTaus: t.baseAddress!,
                stateHashes: h.baseAddress!, materialCounts: ma.baseAddress!,
                gameLength: UInt16(min(length, Int(UInt16.max))),
                workerId: workerId, intraWorkerGameIndex: gameIndex,
                outcome: outcome, count: length
            )
        }}}}}}
    }

    /// Draw one batch and return per-position (outcome, gameLength, workerGameId).
    private func drawBatch(
        _ buffer: ReplayBuffer, count: Int
    ) -> (ok: Bool, zs: [Float], lens: [UInt16], gameIds: [UInt32]) {
        let fpb = ReplayBuffer.floatsPerBoard
        var boards = [Float](repeating: 0, count: count * fpb)
        var moves = [Int32](repeating: 0, count: count)
        var zs = [Float](repeating: 0, count: count)
        var lens = [UInt16](repeating: 0, count: count)
        var gids = [UInt32](repeating: 0, count: count)
        let ok = boards.withUnsafeMutableBufferPointer { b -> Bool in
        moves.withUnsafeMutableBufferPointer { m in
        zs.withUnsafeMutableBufferPointer { z in
        lens.withUnsafeMutableBufferPointer { l in
        gids.withUnsafeMutableBufferPointer { g in
            buffer.sample(
                count: count, intoBoards: b.baseAddress!, moves: m.baseAddress!,
                zs: z.baseAddress!, gameLengths: l.baseAddress!, workerGameIds: g.baseAddress!
            )
        }}}}}
        return (ok, zs, lens, gids)
    }

    private func constraints(maxPerGame: Int = .max, maxDrawPercent: Int = 100, targetLen: Int = 0)
        -> ReplayBuffer.SamplingConstraints {
        ReplayBuffer.SamplingConstraints(
            maxPerGame: maxPerGame, maxDrawPercent: maxDrawPercent,
            targetMeanGameLengthPlies: targetLen
        )
    }

    /// A buffer with `nDrawGames` games of `drawLen` (outcome 0) and
    /// `nDecGames` games of `decLen` (alternating +1 / -1).
    private func makeMixedBuffer(
        capacity: Int, nDrawGames: Int, drawLen: Int, nDecGames: Int, decLen: Int
    ) -> ReplayBuffer {
        let buf = ReplayBuffer(capacity: capacity)
        var gi: UInt32 = 0
        for _ in 0..<nDrawGames { appendGame(to: buf, length: drawLen, outcome: 0, workerId: 0, gameIndex: gi); gi += 1 }
        for k in 0..<nDecGames { appendGame(to: buf, length: decLen, outcome: k % 2 == 0 ? 1 : -1, workerId: 1, gameIndex: gi); gi += 1 }
        return buf
    }

    // MARK: - Tests

    func testNoOpConstraintsProduceValidUniformBatch() {
        // 10 drawn 40-ply games + 10 decisive 30-ply games = 700 positions.
        let buf = makeMixedBuffer(capacity: 10_000, nDrawGames: 10, drawLen: 40, nDecGames: 10, decLen: 30)
        XCTAssertEqual(buf.count, 700)
        let comp = buf.compositionSnapshot()
        XCTAssertEqual(comp.storedCount, 700)
        XCTAssertEqual(comp.distinctResidentGames, 20)
        XCTAssertEqual(comp.drawPositions, 400)
        XCTAssertEqual(comp.winPositions + comp.lossPositions, 300)
        // 10 wins of 30 + 10 losses of 30, alternating ⇒ but appendGame uses k%2:
        // k=0,2,4,6,8 → win (5 games × 30 = 150); k=1,3,5,7,9 → loss (150).
        XCTAssertEqual(comp.winPositions, 150)
        XCTAssertEqual(comp.lossPositions, 150)
        XCTAssertEqual(comp.sumGameLengthOverResidentPositions, 400 * 40 + 300 * 30)
        // game-weighted mean = 700/20 = 35; position-weighted = (16000+9000)/700 ≈ 35.71
        XCTAssertEqual(comp.meanGameLengthPerGame, 35.0, accuracy: 1e-9)
        XCTAssertEqual(comp.meanGameLengthPerSampledPosition, Double(400*40 + 300*30) / 700.0, accuracy: 1e-9)

        buf.setSamplingConstraints(.unconstrained)
        let b = drawBatch(buf, count: 256)
        XCTAssertTrue(b.ok)
        // Every sampled position must correspond to a real resident game.
        for z in b.zs { XCTAssertTrue(z == 0 || z == 1 || z == -1) }
        for l in b.lens { XCTAssertTrue(l == 40 || l == 30) }
    }

    func testDrawPercentCapIsCeiling() {
        // 85% draws: 17 drawn 100-ply games + 3 decisive 100-ply games (per side mix).
        let buf = makeMixedBuffer(capacity: 100_000, nDrawGames: 17, drawLen: 100, nDecGames: 6, decLen: 100)
        // 1700 draw positions, 600 decisive ⇒ 1700/2300 ≈ 73.9% draws.
        for cap in [0, 25, 50, 70] {
            buf.setSamplingConstraints(constraints(maxDrawPercent: cap))
            let b = drawBatch(buf, count: 512)
            XCTAssertTrue(b.ok)
            let drawn = b.zs.filter { $0 == 0 }.count
            // The achieved draw fraction must not exceed the cap (within a
            // 1-position rounding slack on the stratum split).
            XCTAssertLessThanOrEqual(drawn, Int((Double(cap) / 100.0 * 512).rounded()) + 1,
                "cap=\(cap)%: drawn=\(drawn)/512 exceeds ceiling")
            // Freed slots must be decisive (not silently dropped).
            XCTAssertEqual(drawn + b.zs.filter { $0 != 0 }.count, 512)
        }
    }

    func testMaxPerGameCapIsHard() {
        // 5 games of 200 plies each = 1000 positions, all from distinct games.
        let buf = ReplayBuffer(capacity: 10_000)
        for gi in 0..<5 { appendGame(to: buf, length: 200, outcome: gi % 2 == 0 ? 1 : -1, workerId: 0, gameIndex: UInt32(gi)) }
        XCTAssertEqual(buf.count, 1000)
        for cap in [1, 4, 16, 64] {
            buf.setSamplingConstraints(constraints(maxPerGame: cap))
            let b = drawBatch(buf, count: 256)
            XCTAssertTrue(b.ok)
            var perGame: [UInt32: Int] = [:]
            for g in b.gameIds { perGame[g, default: 0] += 1 }
            for (g, n) in perGame {
                XCTAssertLessThanOrEqual(n, cap, "cap=\(cap): game \(g) drawn \(n) times")
            }
        }
    }

    func testTargetLengthPullsMeanDown() {
        // Drawn games with a broad length spread (so there's something for
        // the tilt to bite on), all in one outcome class so the test
        // exercises the length tilt in isolation (no draw cap).
        let buf = ReplayBuffer(capacity: 2_000_000)
        var gi: UInt32 = 0
        for len in [50, 100, 200, 400, 800] {
            for _ in 0..<20 { appendGame(to: buf, length: len, outcome: 0, workerId: 0, gameIndex: gi); gi += 1 }
        }
        let comp = buf.compositionSnapshot()
        let naturalMean = comp.meanGameLengthPerSampledPosition  // position-weighted, dominated by 800s
        XCTAssertGreaterThan(naturalMean, 400)

        // Target above the natural mean ⇒ no-op (β == 0).
        buf.setSamplingConstraints(constraints(targetLen: Int(naturalMean) + 200))
        let none = drawBatch(buf, count: 4096)
        XCTAssertTrue(none.ok)
        var noneSum = 0.0
        for l in none.lens { noneSum += Double(l) }
        XCTAssertEqual(noneSum / Double(none.lens.count), naturalMean, accuracy: naturalMean * 0.1)

        // Target well below ⇒ achieved mean must move strongly toward it.
        var prevAchieved = naturalMean
        for target in [400, 200, 100] {
            buf.setSamplingConstraints(constraints(targetLen: target))
            var sum = 0.0; var n = 0
            for _ in 0..<4 {
                let b = drawBatch(buf, count: 4096)
                XCTAssertTrue(b.ok)
                for l in b.lens { sum += Double(l); n += 1 }
            }
            let achieved = sum / Double(n)
            XCTAssertLessThan(achieved, naturalMean * 0.85,
                "target=\(target): achieved mean \(achieved) not pulled below natural \(naturalMean)")
            XCTAssertLessThan(achieved, prevAchieved + 1.0,
                "target=\(target): achieved mean \(achieved) not monotone (prev \(prevAchieved))")
            XCTAssertGreaterThanOrEqual(achieved, 45.0)  // can't go below the shortest game (50)
            prevAchieved = achieved
        }
    }

    func testCompositionAggregatesSurviveRoundTrip() throws {
        let buf = makeMixedBuffer(capacity: 5_000, nDrawGames: 7, drawLen: 50, nDecGames: 5, decLen: 33)
        let before = buf.compositionSnapshot()
        try buf.write(to: tempFile)
        let restored = ReplayBuffer(capacity: 5_000)
        try restored.restore(from: tempFile)
        let after = restored.compositionSnapshot()
        XCTAssertEqual(before, after)
        // Constrained sampling works on the restored buffer.
        restored.setSamplingConstraints(constraints(maxPerGame: 2, maxDrawPercent: 30))
        let b = drawBatch(restored, count: 128)
        XCTAssertTrue(b.ok)
        XCTAssertLessThanOrEqual(b.zs.filter { $0 == 0 }.count, Int((0.30 * 128).rounded()) + 1)
        var perGame: [UInt32: Int] = [:]
        for g in b.gameIds { perGame[g, default: 0] += 1 }
        for (_, n) in perGame { XCTAssertLessThanOrEqual(n, 2) }
    }

    func testUnderFillReturnsFalse() {
        let buf = ReplayBuffer(capacity: 10_000)
        appendGame(to: buf, length: 10, outcome: 0, workerId: 0, gameIndex: 0)
        // Even unconstrained, asking for more than is held returns false.
        let b = drawBatch(buf, count: 100)
        XCTAssertFalse(b.ok)
        // Constrained path: a stratum with zero supply just clamps; the
        // call still succeeds when there's enough total supply.
        appendGame(to: buf, length: 100, outcome: 0, workerId: 0, gameIndex: 1)  // 110 draws, 0 decisive
        buf.setSamplingConstraints(constraints(maxDrawPercent: 50))  // wants decisive, has none
        let b2 = drawBatch(buf, count: 64)
        XCTAssertTrue(b2.ok)
        XCTAssertEqual(b2.zs.filter { $0 == 0 }.count, 64)  // all draws, decisive stratum empty
    }

    // MARK: - SamplingResult / [SAMPLER] diagnostics

    /// No-op fast path: `wasConstrainedPath` is false and `wasDegraded`
    /// never fires (the [SAMPLER] line won't be emitted on these batches).
    func testSamplingResultNoOpPath() {
        let buf = makeMixedBuffer(capacity: 10_000, nDrawGames: 10, drawLen: 40, nDecGames: 10, decLen: 30)
        buf.setSamplingConstraints(.unconstrained)
        let b = drawBatch(buf, count: 128)
        XCTAssertTrue(b.ok)
        let sr = buf.lastSamplingResult()
        XCTAssertTrue(sr.didSample)
        XCTAssertFalse(sr.wasConstrainedPath)
        XCTAssertFalse(sr.wasDegraded)
        XCTAssertEqual(sr.batchSize, 128)
    }

    /// Buffer is heavily draw-skewed (1700 draws, only 200 decisive).
    /// With a draw cap of 25% on a batch of 512, the decisive stratum
    /// requests `bDec = 384` positions, but the buffer only has 200
    /// decisive — `bDec` clamps down to 200 and the slack moves into
    /// `bDraw`, pushing achievedDrawCount above the requested ceiling.
    /// That's a *cap violation* on the draw ceiling; the trainer
    /// surfaces this on [SAMPLER].
    func testSamplingResultReportsDrawStratumOverflow() {
        // 17 drawn 100-ply + 2 decisive 100-ply ⇒ 1700 draws, 200 decisive.
        // 200 < 384 (the requested bDec) forces the clamp branch in sample().
        let buf = makeMixedBuffer(capacity: 100_000, nDrawGames: 17, drawLen: 100, nDecGames: 2, decLen: 100)
        let batchSize = 512
        buf.setSamplingConstraints(constraints(maxDrawPercent: 25))
        let b = drawBatch(buf, count: batchSize)
        XCTAssertTrue(b.ok)
        let sr = buf.lastSamplingResult()
        XCTAssertTrue(sr.didSample)
        XCTAssertTrue(sr.wasConstrainedPath)
        XCTAssertEqual(sr.batchSize, batchSize)
        // Request was 25% of 512 = 128 draws.
        XCTAssertEqual(sr.requestedDrawCount, Int((0.25 * 512).rounded()))
        // After the clamp: bDec = 200, deficit = 184, bDraw = 128 + 184 = 312.
        XCTAssertEqual(sr.achievedDrawCount, 312,
            "expected post-clamp bDraw = 312, got \(sr.achievedDrawCount)")
        XCTAssertGreaterThan(sr.achievedDrawCount, sr.requestedDrawCount,
            "achieved=\(sr.achievedDrawCount) should overshoot requested=\(sr.requestedDrawCount)")
        XCTAssertTrue(sr.wasDegraded)
    }

    /// Buffer is mostly decisive. With a *high* draw cap of 90%, the
    /// draw stratum gets requested ~460 positions but the buffer only
    /// has ~80 drawn — `bDraw` clamps down, the decisive stratum eats
    /// the slack. Achieved < requested. Still flagged as degraded so
    /// the operator notices the buffer can't support the cap.
    func testSamplingResultReportsDrawStratumUndershoot() {
        // 2 drawn 40-ply + 20 decisive 40-ply ⇒ 80 draws, 800 decisive.
        let buf = makeMixedBuffer(capacity: 50_000, nDrawGames: 2, drawLen: 40, nDecGames: 20, decLen: 40)
        let batchSize = 512
        buf.setSamplingConstraints(constraints(maxDrawPercent: 90))
        let b = drawBatch(buf, count: batchSize)
        XCTAssertTrue(b.ok)
        let sr = buf.lastSamplingResult()
        XCTAssertTrue(sr.wasConstrainedPath)
        XCTAssertEqual(sr.requestedDrawCount, Int((0.90 * 512).rounded()))
        XCTAssertLessThan(sr.achievedDrawCount, sr.requestedDrawCount,
            "achieved=\(sr.achievedDrawCount) should undershoot requested=\(sr.requestedDrawCount)")
        XCTAssertTrue(sr.wasDegraded)
    }

    /// Length target *above* the shortest resident game is feasible —
    /// β bisection converges and `lengthTargetInfeasible` stays false.
    /// `shortestResidentLength` is still reported (for [SAMPLER]'s
    /// orientation context).
    func testSamplingResultLengthTargetFeasible() {
        // Lengths 50, 100, 200, 400. Shortest = 50.
        let buf = ReplayBuffer(capacity: 1_000_000)
        var gi: UInt32 = 0
        for len in [50, 100, 200, 400] {
            for _ in 0..<20 { appendGame(to: buf, length: len, outcome: 0, workerId: 0, gameIndex: gi); gi += 1 }
        }
        // Target 120 — above shortest (50), below natural position-
        // weighted mean ⇒ a finite β exists.
        buf.setSamplingConstraints(constraints(targetLen: 120))
        let b = drawBatch(buf, count: 1024)
        XCTAssertTrue(b.ok)
        let sr = buf.lastSamplingResult()
        XCTAssertTrue(sr.wasConstrainedPath)
        XCTAssertFalse(sr.lengthTargetInfeasible,
            "shortest=50, target=120 should be feasible")
        XCTAssertEqual(sr.shortestResidentLength, 50)
    }

    /// Length target *at or below* the shortest resident game is
    /// infeasible — the position-weighted tilted mean cannot fall
    /// below the shortest game length even at β→∞. The solver must
    /// flag this rather than runaway-grow the bracket; the trainer
    /// surfaces it on [SAMPLER] so the operator can either widen the
    /// buffer's length distribution or relax the target.
    func testSamplingResultLengthTargetInfeasibleBelowShortest() {
        // Shortest resident length = 50; target 30 is unreachable.
        let buf = ReplayBuffer(capacity: 1_000_000)
        var gi: UInt32 = 0
        for len in [50, 100, 200, 400] {
            for _ in 0..<20 { appendGame(to: buf, length: len, outcome: 0, workerId: 0, gameIndex: gi); gi += 1 }
        }
        buf.setSamplingConstraints(constraints(targetLen: 30))
        let b = drawBatch(buf, count: 1024)
        XCTAssertTrue(b.ok)
        let sr = buf.lastSamplingResult()
        XCTAssertTrue(sr.wasConstrainedPath)
        XCTAssertTrue(sr.lengthTargetInfeasible,
            "shortest=50, target=30 must be flagged infeasible")
        XCTAssertEqual(sr.shortestResidentLength, 50)
        XCTAssertTrue(sr.wasDegraded)
        // The achieved batch should still complete (clamped β rejects
        // all but the shortest games) — the achieved mean should be
        // near 50, not the natural mean.
        XCTAssertLessThan(sr.achievedMeanGameLength, 80,
            "with infeasible target the achieved mean should still collapse near shortest")
    }

    /// Length target equal to the shortest resident length is also
    /// infeasible whenever any longer game is present (equality only
    /// reachable at β = ∞).
    func testSamplingResultLengthTargetInfeasibleAtShortest() {
        let buf = ReplayBuffer(capacity: 1_000_000)
        var gi: UInt32 = 0
        for len in [50, 100, 200] {
            for _ in 0..<10 { appendGame(to: buf, length: len, outcome: 0, workerId: 0, gameIndex: gi); gi += 1 }
        }
        buf.setSamplingConstraints(constraints(targetLen: 50))
        let b = drawBatch(buf, count: 256)
        XCTAssertTrue(b.ok)
        let sr = buf.lastSamplingResult()
        XCTAssertTrue(sr.lengthTargetInfeasible)
        XCTAssertEqual(sr.shortestResidentLength, 50)
    }

    /// `BatchStatsSummary` carries the constraints that produced its
    /// histograms (the post-sampling-constraints caption). Verified by
    /// checking the JSON line contents — `applied` toggles with the
    /// constraint state.
    func testBatchStatsSummaryCarriesConstraints() {
        let buf = makeMixedBuffer(capacity: 10_000, nDrawGames: 10, drawLen: 40, nDecGames: 10, decLen: 30)
        // No-op path.
        buf.setSamplingConstraints(.unconstrained)
        let summary1 = sampleAndCompute(buf, count: 128)
        XCTAssertFalse(summary1.samplingConstraintsApplied)
        let json1 = summary1.jsonLine()
        XCTAssertTrue(json1.contains("\"sampling_constraints\":{\"applied\":false"),
            "no-op path should emit applied:false; got: \(json1.prefix(200))")
        // Constrained path.
        buf.setSamplingConstraints(constraints(maxPerGame: 5, maxDrawPercent: 50, targetLen: 35))
        let summary2 = sampleAndCompute(buf, count: 128)
        XCTAssertTrue(summary2.samplingConstraintsApplied)
        let json2 = summary2.jsonLine()
        XCTAssertTrue(json2.contains("\"applied\":true"), "got: \(json2.prefix(200))")
        XCTAssertTrue(json2.contains("\"max_per_game\":5"))
        XCTAssertTrue(json2.contains("\"max_draw_pct\":50"))
        XCTAssertTrue(json2.contains("\"target_length\":35"))
    }

    /// Sample one batch with the full metadata pointers filled, then
    /// hand the same buffers to `computeBatchStats` so the test can
    /// inspect the resulting summary. Single sample call — keeps
    /// `_lastSamplingResult` aligned with the summary returned.
    private func sampleAndCompute(_ buffer: ReplayBuffer, count: Int) -> ReplayBuffer.BatchStatsSummary {
        let fpb = ReplayBuffer.floatsPerBoard
        let b = UnsafeMutablePointer<Float>.allocate(capacity: count * fpb)
        let m = UnsafeMutablePointer<Int32>.allocate(capacity: count)
        let z = UnsafeMutablePointer<Float>.allocate(capacity: count)
        let pl = UnsafeMutablePointer<UInt16>.allocate(capacity: count)
        let gl = UnsafeMutablePointer<UInt16>.allocate(capacity: count)
        let ta = UnsafeMutablePointer<Float>.allocate(capacity: count)
        let ha = UnsafeMutablePointer<UInt64>.allocate(capacity: count)
        let wi = UnsafeMutablePointer<UInt32>.allocate(capacity: count)
        let ma = UnsafeMutablePointer<UInt8>.allocate(capacity: count)
        defer {
            b.deallocate(); m.deallocate(); z.deallocate()
            pl.deallocate(); gl.deallocate(); ta.deallocate()
            ha.deallocate(); wi.deallocate(); ma.deallocate()
        }
        // sample() unconditionally writes to every slot — uninitialized
        // POD storage is fine to pass in; computeBatchStats reads only
        // what sample just wrote.
        let ok = buffer.sample(
            count: count, intoBoards: b, moves: m, zs: z,
            plies: pl, gameLengths: gl, taus: ta, hashes: ha,
            workerGameIds: wi, materialCounts: ma
        )
        XCTAssertTrue(ok, "sample under-fill in sampleAndCompute helper")
        return buffer.computeBatchStats(
            step: 0, batchSize: count,
            plies: pl, gameLengths: gl, taus: ta,
            hashes: ha, workerGameIds: wi,
            materialCounts: ma, zs: z
        )
    }
}
