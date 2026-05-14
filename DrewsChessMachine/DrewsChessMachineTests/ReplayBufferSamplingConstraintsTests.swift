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

    /// No-op fast path also populates the per-batch achievement counters
    /// (so the popover's "Last batch" readout works regardless of
    /// constraint state). W/D/L sum to the batch size, distinct game
    /// count and max-per-game are in plausible ranges, and the
    /// achieved-mean-game-length comes out near the buffer's
    /// position-weighted mean.
    func testSamplingResultNoOpPathPopulatesAchievementCounters() {
        // 10 drawn 40-ply + 10 decisive 30-ply = 700 positions across 20 games.
        // Buffer composition: 400 draws, 150 wins, 150 losses; position-
        // weighted mean game length ≈ (400*40 + 300*30) / 700 ≈ 35.71.
        let buf = makeMixedBuffer(capacity: 10_000, nDrawGames: 10, drawLen: 40, nDecGames: 10, decLen: 30)
        buf.setSamplingConstraints(.unconstrained)
        let batchSize = 4096
        let b = drawBatch(buf, count: batchSize)
        XCTAssertTrue(b.ok)
        let sr = buf.lastSamplingResult()
        XCTAssertFalse(sr.wasConstrainedPath)
        // W + D + L must sum to batch size.
        XCTAssertEqual(sr.achievedWinCount + sr.achievedDrawCount + sr.achievedLossCount, batchSize)
        // Distinct games in batch ≤ resident game count and > 0.
        XCTAssertGreaterThan(sr.distinctGamesInBatch, 0)
        XCTAssertLessThanOrEqual(sr.distinctGamesInBatch, 20)
        // Max per game is at least the average and at most the batch size.
        let avgPerGame = Double(batchSize) / Double(sr.distinctGamesInBatch)
        XCTAssertGreaterThanOrEqual(Double(sr.achievedMaxPerGame), avgPerGame)
        XCTAssertLessThanOrEqual(sr.achievedMaxPerGame, batchSize)
        // Achieved mean game length should be close to the position-
        // weighted mean of the buffer (35.71). 4096 samples ⇒ tight CLT.
        let bufferMean = Double(400 * 40 + 300 * 30) / 700.0
        XCTAssertEqual(sr.achievedMeanGameLength, bufferMean, accuracy: 2.0,
            "fast-path achieved mean \(sr.achievedMeanGameLength) should track buffer mean \(bufferMean)")
        XCTAssertEqual(sr.achievedMeanSamplesPerGame, avgPerGame, accuracy: 1e-9)
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

    // MARK: - K-aware stratum sizing (regression for the budget-hit bug)

    /// Tracks the new `residentDecisiveGameCount` field that backs the
    /// K-aware stratum ceiling — verifies it counts only games whose
    /// outcome is decisive, mirrors the resident-set lifecycle through
    /// FIFO eviction, and survives the persistence round-trip.
    func testResidentDecisiveGameCountTracking() throws {
        // 6 drawn games + 4 decisive games — decisive count should be 4
        // regardless of insertion order. Sizes are mixed to also exercise
        // the per-position incrementing path during restore (one slot at
        // a time, count=1, so the "is this a new resident game?" branch
        // fires exactly once per game on rebuild).
        let buf = ReplayBuffer(capacity: 10_000)
        var gi: UInt32 = 0
        for _ in 0..<3 { appendGame(to: buf, length: 40, outcome: 0, workerId: 0, gameIndex: gi); gi += 1 }
        for _ in 0..<2 { appendGame(to: buf, length: 30, outcome: 1, workerId: 0, gameIndex: gi); gi += 1 }
        for _ in 0..<3 { appendGame(to: buf, length: 40, outcome: 0, workerId: 0, gameIndex: gi); gi += 1 }
        for _ in 0..<2 { appendGame(to: buf, length: 30, outcome: -1, workerId: 0, gameIndex: gi); gi += 1 }

        let snap1 = buf.compositionSnapshot()
        XCTAssertEqual(snap1.distinctResidentGames, 10)
        XCTAssertEqual(snap1.residentDecisiveGameCount, 4)

        // Persistence round-trip: rebuilt from the per-slot loop in
        // `restore`, which exercises the per-position increment path.
        try buf.write(to: tempFile)
        let restored = ReplayBuffer(capacity: 10_000)
        try restored.restore(from: tempFile)
        let snap2 = restored.compositionSnapshot()
        XCTAssertEqual(snap2, snap1, "decisive-game count must round-trip")
        XCTAssertEqual(snap2.residentDecisiveGameCount, 4)

        // FIFO eviction: fill past capacity and confirm the decisive
        // count tracks resident-set turnover correctly.
        let small = ReplayBuffer(capacity: 240)
        var gi2: UInt32 = 100
        // Slot ranges (capacity=240, ring starts at slot 0):
        //   gi=100 (draw 40):    slots 0-39
        //   gi=101 (draw 40):    slots 40-79
        //   gi=102 (draw 40):    slots 80-119
        //   gi=103 (decisive 30): slots 120-149
        //   gi=104 (decisive 30): slots 150-179
        //   gi=105 (draw 60):    slots 180-239   — buffer now full, writeIndex wraps to 0
        for _ in 0..<3 { appendGame(to: small, length: 40, outcome: 0, workerId: 0, gameIndex: gi2); gi2 += 1 }
        for _ in 0..<2 { appendGame(to: small, length: 30, outcome: 1, workerId: 0, gameIndex: gi2); gi2 += 1 }
        XCTAssertEqual(small.compositionSnapshot().residentDecisiveGameCount, 2)
        appendGame(to: small, length: 60, outcome: 0, workerId: 0, gameIndex: gi2); gi2 += 1
        XCTAssertEqual(small.count, 240)
        XCTAssertEqual(small.compositionSnapshot().residentDecisiveGameCount, 2)

        // Add a 120-ply drawn game — wraps to slot 0 and evicts slots
        // 0-119, which is exactly the first three drawn games. The two
        // decisive games (gi=103, gi=104) survive untouched.
        appendGame(to: small, length: 120, outcome: 0, workerId: 0, gameIndex: gi2); gi2 += 1
        let snap3 = small.compositionSnapshot()
        XCTAssertEqual(snap3.distinctResidentGames, 4,
            "after eviction expect gi=103, 104, 105, 106 resident (got distinct=\(snap3.distinctResidentGames))")
        XCTAssertEqual(snap3.residentDecisiveGameCount, 2,
            "decisive games (gi=103, 104) untouched by this eviction")

        // Add a 60-ply drawn game — evicts slots 120-179, which is
        // both decisive games end-to-end. Decisive count must drop to 0.
        appendGame(to: small, length: 60, outcome: 0, workerId: 0, gameIndex: gi2); gi2 += 1
        let snap4 = small.compositionSnapshot()
        XCTAssertEqual(snap4.residentDecisiveGameCount, 0,
            "decisive count must drop to zero once both decisive games have been fully evicted")
        XCTAssertEqual(snap4.distinctResidentGames, 3,
            "after second eviction expect gi=105, 106, 107 resident")
    }

    /// Direct regression for the bug analysed under
    /// "max_draw_percent_per_batch=65 but achieved ~87%". Reproduces
    /// the production regime in miniature: a draw-dominated buffer
    /// where the requested decisive stratum exceeds K · resident
    /// decisive game count. Before the fix the rejection loop
    /// exhausted `attemptBudget` filling the decisive stratum, then the
    /// uniform-fill fallback ignored the K cap AND the draw cap AND
    /// the length tilt. After the fix:
    ///   - `attemptBudgetHit` is false (the stratum sizing knew the
    ///     decisive ceiling up front);
    ///   - `achievedMaxPerGame ≤ K` (the K cap is honoured in every
    ///     emitted slot, not silently dropped);
    ///   - `wasDegraded` is true because the K-aware reflow pushed the
    ///     achieved draw count above the requested draw count — the
    ///     `[SAMPLER]` log fires so the operator sees the buffer can't
    ///     support the cap *given the K constraint*.
    func testKAwareStratumSizingPreventsBudgetHitAndPreservesCapInDrawSkewedBuffer() {
        // Buffer: 10 decisive × 50-ply + 200 draw × 80-ply.
        // 500 decisive positions, 16,000 draw positions ⇒ 97.0% draws.
        // With K=2: decisiveReachable = 2·10 = 20 (well below 500);
        //          drawReachable     = 2·200 = 400 (well below 16,000).
        let buf = ReplayBuffer(capacity: 50_000)
        var gi: UInt32 = 0
        for k in 0..<10 {
            appendGame(to: buf, length: 50, outcome: k % 2 == 0 ? 1 : -1, workerId: 0, gameIndex: gi)
            gi += 1
        }
        for _ in 0..<200 {
            appendGame(to: buf, length: 80, outcome: 0, workerId: 1, gameIndex: gi)
            gi += 1
        }
        XCTAssertEqual(buf.count, 500 + 200 * 80)
        XCTAssertEqual(buf.compositionSnapshot().residentDecisiveGameCount, 10)

        // batch=200: requestedDecisive = 70 (> reachable=20); strata
        // jointly reach 20 + 400 = 420 ≥ 200, so no under-fill. The
        // K-aware reflow pushes 50 slots from decisive into draws,
        // landing at bDec=20, bDraw=180.
        let batchSize = 200
        buf.setSamplingConstraints(constraints(maxPerGame: 2, maxDrawPercent: 65, targetLen: 0))
        let b = drawBatch(buf, count: batchSize)
        XCTAssertTrue(b.ok)
        let sr = buf.lastSamplingResult()

        // Core regression assertions: the bug's signature is a true
        // budget hit + K-cap violation. Both must be absent.
        XCTAssertFalse(sr.attemptBudgetHit,
            "K-aware sizing must avoid the attempt-budget fallback in this regime")
        XCTAssertLessThanOrEqual(sr.achievedMaxPerGame, 2,
            "K=2 cap must hold for every emitted position; got maxG=\(sr.achievedMaxPerGame)")
        var perGame: [UInt32: Int] = [:]
        for g in b.gameIds { perGame[g, default: 0] += 1 }
        for (g, n) in perGame {
            XCTAssertLessThanOrEqual(n, 2,
                "K=2 cap violated for game \(g): \(n) positions")
        }

        // Achievement counters land at the K-aware ceiling for decisive
        // and absorb the deficit into draws. Allow a small slack for
        // the standard 1-position rounding within the sampler.
        XCTAssertEqual((sr.achievedWinCount + sr.achievedLossCount), 20,
            "decisive stratum should hit exactly K · residentDecisiveGameCount = 20")
        XCTAssertEqual(sr.achievedDrawCount, 180,
            "draw stratum should absorb the decisive deficit and land at 180")
        XCTAssertEqual((sr.achievedWinCount + sr.achievedLossCount) + sr.achievedDrawCount, batchSize)

        // Degraded flag fires (achieved draws > requested draws) so the
        // trainer surfaces a [SAMPLER] line — the operator must still
        // see the cap couldn't be honoured under the K constraint.
        XCTAssertTrue(sr.wasDegraded,
            "wasDegraded must fire so [SAMPLER] surfaces the cap mismatch")
        XCTAssertGreaterThan(sr.achievedDrawCount, sr.requestedDrawCount)
        XCTAssertEqual(sr.requestedDrawCount, Int((0.65 * Double(batchSize)).rounded()))
    }

    /// Symmetric coverage for the rare draw-scarce case: a decisive-
    /// dominated buffer where the K cap is the binding ceiling on the
    /// *draw* stratum. Confirms the K-aware reflow is bidirectional.
    func testKAwareStratumSizingHandlesDecisiveSkewedBuffer() {
        // Buffer: 100 decisive × 80-ply + 5 draw × 40-ply.
        // 8000 decisive positions, 200 draw positions ⇒ ~2.4% draws.
        // With K=2: drawReachable = 2·5 = 10; decisiveReachable = 2·100 = 200.
        let buf = ReplayBuffer(capacity: 50_000)
        var gi: UInt32 = 0
        for k in 0..<100 {
            appendGame(to: buf, length: 80, outcome: k % 2 == 0 ? 1 : -1, workerId: 0, gameIndex: gi)
            gi += 1
        }
        for _ in 0..<5 {
            appendGame(to: buf, length: 40, outcome: 0, workerId: 1, gameIndex: gi)
            gi += 1
        }
        let batchSize = 128
        // Request 70% draws (89 positions) — far above the reachable
        // draw ceiling of 10. Decisive stratum absorbs the deficit.
        buf.setSamplingConstraints(constraints(maxPerGame: 2, maxDrawPercent: 70, targetLen: 0))
        let b = drawBatch(buf, count: batchSize)
        XCTAssertTrue(b.ok)
        let sr = buf.lastSamplingResult()

        XCTAssertFalse(sr.attemptBudgetHit)
        XCTAssertLessThanOrEqual(sr.achievedMaxPerGame, 2)
        XCTAssertEqual(sr.achievedDrawCount, 10,
            "draw stratum should hit exactly K · residentDrawGameCount = 10")
        XCTAssertEqual((sr.achievedWinCount + sr.achievedLossCount), batchSize - 10)
        XCTAssertTrue(sr.wasDegraded)
        XCTAssertLessThan(sr.achievedDrawCount, sr.requestedDrawCount)
    }
}
