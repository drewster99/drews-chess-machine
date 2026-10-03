//
//  ReplayBufferResumeEquivalenceTests.swift
//  DrewsChessMachineTests
//
//  The state-level half of an exact corpus-replay resume (determinism plan
//  C1 #4/#5, C6): a buffer refilled after a resume — fresh ring, written from
//  slot 0, holding the same positions in the same age order as the buffer
//  it replaces — and given the saved sampler state must draw exactly the
//  batches the uninterrupted buffer draws, including through the
//  material-bucket stratified path and through later inserts and evictions.
//

import XCTest
@testable import DrewsChessMachine

final class ReplayBufferResumeEquivalenceTests: XCTestCase {

    // MARK: - Synthetic games

    /// One synthetic game: every row carries a hash unique to (game, ply), so
    /// a batch's hashes identify exactly which positions it drew.
    private struct SyntheticGame {
        let index: Int
        let length: Int
        let outcome: Float
        let materialCount: UInt8
    }

    /// A fixed, varied sequence of games: lengths 3…40, all three outcomes,
    /// every active material bucket.
    private func games(count: Int) -> [SyntheticGame] {
        (0..<count).map { i in
            SyntheticGame(
                index: i,
                length: 3 + (i * 7) % 38,
                outcome: [Float(1), 0, -1][i % 3],
                materialCount: [UInt8(2), 7, 12, 18][(i / 3) % 4]
            )
        }
    }

    private func append(_ game: SyntheticGame, to buffer: ReplayBuffer) {
        let fpb = ReplayBuffer.defaultFloatsPerBoard
        let n = game.length
        var boards = [Float](repeating: 0, count: n * fpb)
        for i in 0..<n { boards[i * fpb] = Float(game.index) * 1e3 + Float(i) }
        let moves = (0..<n).map { Int32($0) }
        let plies = (0..<n).map { UInt16($0) }
        let taus = [Float](repeating: 1, count: n)
        let hashes = (0..<n).map { (UInt64(game.index) << 16) | UInt64($0) }
        let mats = [UInt8](repeating: game.materialCount, count: n)
        let outcomes = [Float](repeating: game.outcome, count: n)
        boards.withUnsafeBufferPointer { b in
        moves.withUnsafeBufferPointer { m in
        plies.withUnsafeBufferPointer { pl in
        taus.withUnsafeBufferPointer { t in
        hashes.withUnsafeBufferPointer { h in
        mats.withUnsafeBufferPointer { ma in
        outcomes.withUnsafeBufferPointer { o in
            guard let b = b.baseAddress, let m = m.baseAddress, let pl = pl.baseAddress,
                  let t = t.baseAddress, let h = h.baseAddress, let ma = ma.baseAddress,
                  let o = o.baseAddress else {
                preconditionFailure("non-empty arrays have base addresses")
            }
            buffer.append(
                boards: b, policyIndices: m, plyIndices: pl, samplingTaus: t,
                stateHashes: h, materialCounts: ma, gameLength: UInt16(n),
                workerId: 0, intraWorkerGameIndex: UInt32(game.index),
                outcomes: o, count: n)
        }}}}}}}
    }

    /// The hashes of one drawn batch, in batch-slot order.
    private func drawHashes(_ buffer: ReplayBuffer, count: Int) -> [UInt64] {
        let fpb = ReplayBuffer.defaultFloatsPerBoard
        var boards = [Float](repeating: 0, count: count * fpb)
        var moves = [Int32](repeating: 0, count: count)
        var zs = [Float](repeating: 0, count: count)
        var hashes = [UInt64](repeating: 0, count: count)
        let ok = boards.withUnsafeMutableBufferPointer { b -> Bool in
        moves.withUnsafeMutableBufferPointer { m in
        zs.withUnsafeMutableBufferPointer { z in
        hashes.withUnsafeMutableBufferPointer { h in
            guard let b = b.baseAddress, let m = m.baseAddress, let z = z.baseAddress, let h = h.baseAddress else {
                preconditionFailure("non-empty arrays have base addresses")
            }
            return buffer.sample(count: count, intoBoards: b, moves: m, zs: z, hashes: h)
        }}}}
        XCTAssertTrue(ok, "buffer held fewer than \(count) positions")
        return hashes
    }

    private static func stratified() -> ReplayBuffer.SamplingConstraints {
        let n = ReplayBufferAnalyzer.materialBuckets.count
        var weights = [Float](repeating: 0, count: n)
        for i in 0..<(n - 1) { weights[i] = 1 / Float(n - 1) }
        return ReplayBuffer.SamplingConstraints(
            maxPerGame: .max, maxDrawPercent: 100, targetMeanGameLengthPlies: 0,
            materialBucketWeights: weights)
    }

    private static func lengthTilted() -> ReplayBuffer.SamplingConstraints {
        ReplayBuffer.SamplingConstraints(
            maxPerGame: .max, maxDrawPercent: 100, targetMeanGameLengthPlies: 15,
            materialBucketWeights: nil)
    }

    // MARK: - The equivalence

    /// Runs the uninterrupted buffer for `gamesBeforeSave` games (drawing a
    /// batch after each game once it can), then rebuilds a second buffer the
    /// way a resume does — a fresh ring fed only the later games, enough to
    /// fill it — with the saved sampler state, and checks both draw the same
    /// batches while the same later games keep arriving.
    private func assertRefilledBufferMatches(
        constraints: ReplayBuffer.SamplingConstraints,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        let capacity = 400
        let batch = 16
        let all = games(count: 240)
        let gamesBeforeSave = 120
        let gamesAfterSave = all.count - gamesBeforeSave

        let uninterrupted = ReplayBuffer(capacity: capacity, sampler: DCMRandom(seed: 0xD5C0FFEE))
        uninterrupted.setSamplingConstraints(constraints)
        for game in all.prefix(gamesBeforeSave) {
            append(game, to: uninterrupted)
            if uninterrupted.count >= batch { _ = drawHashes(uninterrupted, count: batch) }
        }
        let savedSampler = uninterrupted.samplerState()

        // The refill: the newest games that cover the whole ring, oldest
        // first, into a fresh buffer whose own sampler seed is irrelevant
        // once the saved state is restored.
        var refillStart = gamesBeforeSave
        var plies = 0
        while refillStart > 0 && plies < capacity {
            refillStart -= 1
            plies += all[refillStart].length
        }
        XCTAssertGreaterThan(plies, capacity, "the refill must overfill the ring", file: file, line: line)
        let refilled = ReplayBuffer(capacity: capacity, sampler: DCMRandom(seed: 1))
        refilled.setSamplingConstraints(constraints)
        for game in all[refillStart..<gamesBeforeSave] { append(game, to: refilled) }
        refilled.restoreSamplerState(savedSampler)
        XCTAssertEqual(refilled.count, uninterrupted.count, file: file, line: line)

        for (k, game) in all.suffix(gamesAfterSave).enumerated() {
            let a = drawHashes(uninterrupted, count: batch)
            let b = drawHashes(refilled, count: batch)
            XCTAssertEqual(a, b, "batch \(k) after the resume differs", file: file, line: line)
            append(game, to: uninterrupted)
            append(game, to: refilled)
        }
        XCTAssertEqual(uninterrupted.samplerState(), refilled.samplerState(), file: file, line: line)
    }

    func testRefilledBufferDrawsTheSameUniformBatches() {
        assertRefilledBufferMatches(constraints: .unconstrained)
    }

    func testRefilledBufferDrawsTheSameMaterialStratifiedBatches() {
        assertRefilledBufferMatches(constraints: Self.stratified())
    }

    func testRefilledBufferDrawsTheSameLengthTiltedBatches() {
        assertRefilledBufferMatches(constraints: Self.lengthTilted())
    }
}
