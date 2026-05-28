//
//  ReplayBufferMaterialBucketTests.swift
//  DrewsChessMachineTests
//
//  Covers the material-bucket index + stratified sampling path added
//  to `ReplayBuffer`:
//   - bucket boundary mapping matches `ReplayBufferAnalyzer.materialBuckets`
//   - bucket index is maintained correctly on insert and on FIFO eviction
//   - bucket index is rebuilt correctly on session restore from disk
//   - stratified sample emits a balanced per-bucket distribution
//   - empty-bucket deficit is reallocated to other active buckets
//   - the legacy fast path is bit-for-bit unchanged when stratification
//     is off (regression guard for the new SamplingResult / fast-path
//     bookkeeping)
//

import XCTest
@testable import DrewsChessMachine

final class ReplayBufferMaterialBucketTests: XCTestCase {

    private var tempFile: URL!

    override func setUpWithError() throws {
        tempFile = FileManager.default.temporaryDirectory
            .appendingPathComponent("dcm-bucket-test-\(UUID().uuidString).bin")
    }

    override func tearDownWithError() throws {
        if let p = tempFile?.path, FileManager.default.fileExists(atPath: p) {
            try? FileManager.default.removeItem(at: tempFile)
        }
    }

    // MARK: - Helpers

    /// Append a synthetic game where every position has the given
    /// non-pawn material count. Mirrors the helper in
    /// `ReplayBufferSamplingConstraintsTests` but parameterizes the
    /// material count so tests can drive the bucket index directly.
    private func appendGame(
        to buffer: ReplayBuffer,
        length: Int,
        outcome: Float,
        workerId: UInt16,
        gameIndex: UInt32,
        materialCount: UInt8
    ) {
        precondition(length > 0)
        let fpb = ReplayBuffer.floatsPerBoard
        var boards = [Float](repeating: 0, count: length * fpb)
        for i in 0..<length { boards[i * fpb] = Float(workerId) * 1e6 + Float(gameIndex) * 1e3 + Float(i) }
        var moves = [Int32](repeating: 0, count: length)
        for i in 0..<length { moves[i] = Int32(i) }
        var plies = [UInt16](repeating: 0, count: length)
        for i in 0..<length { plies[i] = UInt16(min(i, Int(UInt16.max))) }
        let taus = [Float](repeating: 1.0, count: length)
        var hashes = [UInt64](repeating: 0, count: length)
        for i in 0..<length { hashes[i] = (UInt64(workerId) << 40) | (UInt64(gameIndex) << 16) | UInt64(i & 0xFFFF) }
        let mats = [UInt8](repeating: materialCount, count: length)
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

    private func drawBatch(
        _ buffer: ReplayBuffer, count: Int
    ) -> (ok: Bool, materialCounts: [UInt8]) {
        let fpb = ReplayBuffer.floatsPerBoard
        var boards = [Float](repeating: 0, count: count * fpb)
        var moves = [Int32](repeating: 0, count: count)
        var zs = [Float](repeating: 0, count: count)
        var mats = [UInt8](repeating: 0, count: count)
        let ok = boards.withUnsafeMutableBufferPointer { b -> Bool in
        moves.withUnsafeMutableBufferPointer { m in
        zs.withUnsafeMutableBufferPointer { z in
        mats.withUnsafeMutableBufferPointer { ma in
            buffer.sample(
                count: count,
                intoBoards: b.baseAddress!,
                moves: m.baseAddress!,
                zs: z.baseAddress!,
                materialCounts: ma.baseAddress!
            )
        }}}}
        return (ok, mats)
    }

    // MARK: - Tests

    /// Every boundary on either side of every bucket edge maps to the
    /// expected bucket. Catches off-by-one regressions in
    /// `materialBucketIndex(for:)`.
    func testMaterialBucketBoundariesMatchAnalyzer() {
        XCTAssertEqual(ReplayBuffer.materialBucketIndex(for: 0), 0)
        XCTAssertEqual(ReplayBuffer.materialBucketIndex(for: 4), 0)
        XCTAssertEqual(ReplayBuffer.materialBucketIndex(for: 5), 1)
        XCTAssertEqual(ReplayBuffer.materialBucketIndex(for: 8), 1)
        XCTAssertEqual(ReplayBuffer.materialBucketIndex(for: 9), 2)
        XCTAssertEqual(ReplayBuffer.materialBucketIndex(for: 14), 2)
        XCTAssertEqual(ReplayBuffer.materialBucketIndex(for: 15), 3)
        XCTAssertEqual(ReplayBuffer.materialBucketIndex(for: 22), 3)
        XCTAssertEqual(ReplayBuffer.materialBucketIndex(for: 23), 4)
        // Out-of-range above the analyzer's max should still land in
        // the last bucket (clamp semantics).
        XCTAssertEqual(ReplayBuffer.materialBucketIndex(for: 30), 4)
        XCTAssertEqual(ReplayBuffer.materialBucketIndex(for: 250), 4)
    }

    /// Insertion populates the per-bucket index in lock-step with
    /// `materialCountStorage` — every appended position lands in its
    /// correct bucket and the `residentPerBucket` surface matches.
    func testBucketIndexInsertionTracksMaterialCount() {
        let buf = ReplayBuffer(capacity: 10_000)
        // Three games, one per active bucket, of length 30.
        appendGame(to: buf, length: 30, outcome: 1, workerId: 0, gameIndex: 0, materialCount: 2)
        appendGame(to: buf, length: 30, outcome: 0, workerId: 1, gameIndex: 0, materialCount: 7)
        appendGame(to: buf, length: 30, outcome: -1, workerId: 2, gameIndex: 0, materialCount: 12)
        appendGame(to: buf, length: 30, outcome: 0, workerId: 3, gameIndex: 0, materialCount: 18)
        let snap = buf.compositionSnapshot()
        XCTAssertEqual(snap.storedCount, 120)
        XCTAssertEqual(snap.residentPerBucket.count, ReplayBufferAnalyzer.materialBuckets.count)
        XCTAssertEqual(snap.residentPerBucket[0], 30, "bucket 0–4 should hold materialCount=2 game")
        XCTAssertEqual(snap.residentPerBucket[1], 30, "bucket 5–8 should hold materialCount=7 game")
        XCTAssertEqual(snap.residentPerBucket[2], 30, "bucket 9–14 should hold materialCount=12 game")
        XCTAssertEqual(snap.residentPerBucket[3], 30, "bucket 15–22 should hold materialCount=18 game")
        XCTAssertEqual(snap.residentPerBucket[4], 0, "bucket 23–30 should be empty (geometrically unreachable)")
    }

    /// FIFO eviction removes evicted slots from their bucket and the
    /// newly-inserted slots show up in the right bucket. Sized so the
    /// ring wraps several times.
    func testBucketIndexHonorsRingEviction() {
        let buf = ReplayBuffer(capacity: 60)  // tight ring → forced eviction
        // Round 1: fill the ring with a 60-position game in bucket 3.
        appendGame(to: buf, length: 60, outcome: 0, workerId: 0, gameIndex: 0, materialCount: 18)
        var snap = buf.compositionSnapshot()
        XCTAssertEqual(snap.residentPerBucket[3], 60)
        XCTAssertEqual(snap.residentPerBucket[0], 0)
        // Round 2: overwrite with a 60-position game in bucket 0.
        appendGame(to: buf, length: 60, outcome: 0, workerId: 0, gameIndex: 1, materialCount: 2)
        snap = buf.compositionSnapshot()
        XCTAssertEqual(snap.residentPerBucket[0], 60, "bucket 0–4 should hold the new game")
        XCTAssertEqual(snap.residentPerBucket[3], 0, "bucket 15–22 should be empty after eviction")
        // Round 3: half-overwrite — 30 of bucket 1 lands on top of the
        // first 30 slots of round 2.
        appendGame(to: buf, length: 30, outcome: 1, workerId: 0, gameIndex: 2, materialCount: 7)
        snap = buf.compositionSnapshot()
        XCTAssertEqual(snap.residentPerBucket[1], 30, "30 freshly-bucket-1 positions")
        XCTAssertEqual(snap.residentPerBucket[0], 30, "30 surviving bucket-0 positions")
        XCTAssertEqual(snap.storedCount, 60)
    }

    /// Round-tripping a buffer through `write` + `restore` rebuilds
    /// the bucket index. Both the disk-format invariant (no layout
    /// change) and the in-memory derived state are checked.
    func testBucketIndexRebuildOnRestore() throws {
        let cap = 5_000
        let buf = ReplayBuffer(capacity: cap)
        appendGame(to: buf, length: 100, outcome: 1, workerId: 0, gameIndex: 0, materialCount: 2)
        appendGame(to: buf, length: 100, outcome: 0, workerId: 1, gameIndex: 0, materialCount: 7)
        appendGame(to: buf, length: 100, outcome: -1, workerId: 2, gameIndex: 0, materialCount: 12)
        appendGame(to: buf, length: 100, outcome: 0, workerId: 3, gameIndex: 0, materialCount: 18)
        let original = buf.compositionSnapshot()
        try buf.write(to: tempFile)

        // Fresh buffer, same capacity, restore from disk.
        let restored = ReplayBuffer(capacity: cap)
        try restored.restore(from: tempFile)
        let after = restored.compositionSnapshot()
        XCTAssertEqual(after.residentPerBucket, original.residentPerBucket,
            "restore should rebuild materialBucketSlots exactly")
        XCTAssertEqual(after.storedCount, original.storedCount)
    }

    /// With the toggle off, `sample(...)` takes the bit-for-bit
    /// uniform fast path (`wasConstrainedPath == false`) and the
    /// `achievedBucketCounts` track the buffer's natural mix within
    /// statistical noise. Regression guard.
    func testFastPathStillUsedWhenStratificationOff() {
        let buf = ReplayBuffer(capacity: 10_000)
        // Skewed buffer: 80% bucket 0, 20% bucket 3.
        for i in 0..<8 {
            appendGame(to: buf, length: 100, outcome: 0,
                workerId: 0, gameIndex: UInt32(i), materialCount: 2)
        }
        for i in 0..<2 {
            appendGame(to: buf, length: 100, outcome: 0,
                workerId: 1, gameIndex: UInt32(i), materialCount: 18)
        }
        // Default constraints — fast path.
        buf.setSamplingConstraints(.unconstrained)
        let drawn = drawBatch(buf, count: 1000)
        XCTAssertTrue(drawn.ok)
        let result = buf.lastSamplingResult()
        XCTAssertFalse(result.wasConstrainedPath, "fast path expected for unconstrained")
        XCTAssertNil(result.constraints.materialBucketWeights)
        // Achieved counts should follow the natural 80/20 split within
        // a tolerance set by binomial noise.
        let counts = result.achievedBucketCounts
        XCTAssertEqual(counts.count, ReplayBufferAnalyzer.materialBuckets.count)
        XCTAssertEqual(counts.reduce(0, +), 1000)
        XCTAssertGreaterThan(counts[0], 700, "bucket 0 should dominate (80% natural)")
        XCTAssertLessThan(counts[0], 900)
        XCTAssertGreaterThan(counts[3], 100, "bucket 3 should be present (20% natural)")
        XCTAssertLessThan(counts[3], 300)
    }

    /// With stratification on and all 4 active buckets populated, the
    /// achieved per-bucket counts should be ~equal for a balanced
    /// target. Sample size large enough that statistical noise is
    /// below the assertion tolerance.
    func testStratifiedSampleReachesBalancedTargetMix() {
        let buf = ReplayBuffer(capacity: 20_000)
        // Skewed buffer: 1500 positions of each bucket → balanced
        // *resident* mix; the test is checking that the path produces
        // a balanced *batch* regardless of resident weight, so we'll
        // make residency skewed too.
        let mats: [UInt8] = [2, 7, 12, 18]
        // Unequal residency: 800 / 400 / 200 / 100 of each bucket.
        let perBucketGames = [8, 4, 2, 1]
        let gameLen = 100
        var gi: UInt32 = 0
        for (b, n) in perBucketGames.enumerated() {
            for _ in 0..<n {
                appendGame(to: buf, length: gameLen, outcome: 0,
                    workerId: UInt16(b), gameIndex: gi, materialCount: mats[b])
                gi &+= 1
            }
        }
        let totalResident = perBucketGames.reduce(0, +) * gameLen
        XCTAssertEqual(buf.count, totalResident)
        // Balanced V1 target.
        let n = ReplayBufferAnalyzer.materialBuckets.count
        let activeCount = Float(n - 1)
        var weights = [Float](repeating: 0, count: n)
        for i in 0..<(n - 1) { weights[i] = 1.0 / activeCount }
        buf.setSamplingConstraints(
            ReplayBuffer.SamplingConstraints(
                maxPerGame: .max,
                maxDrawPercent: 100,
                targetMeanGameLengthPlies: 0,
                materialBucketWeights: weights
            )
        )

        let batchSize = 1000
        let drawn = drawBatch(buf, count: batchSize)
        XCTAssertTrue(drawn.ok)
        let result = buf.lastSamplingResult()
        XCTAssertTrue(result.wasConstrainedPath, "stratified path expected when weights set")
        XCTAssertNotNil(result.constraints.materialBucketWeights)
        let counts = result.achievedBucketCounts
        XCTAssertEqual(counts.reduce(0, +), batchSize)
        // Active buckets each get ~250. Allow ±2 (rounding correction
        // may shift +/-1 between adjacent buckets; combined with
        // sample-with-replacement bookkeeping that's well under 1%).
        for i in 0..<(n - 1) {
            XCTAssertEqual(counts[i], 250, accuracy: 2,
                "bucket \(i) achieved \(counts[i]), expected ~250")
        }
        // Unreachable 5th bucket should remain zero.
        XCTAssertEqual(counts[4], 0)
    }

    /// When one of the active buckets is empty, its share should be
    /// reallocated to the remaining active buckets — the batch size
    /// should still be filled exactly. Tests the slack-redistribution
    /// loop.
    func testStratifiedSampleRedistributesEmptyBucket() {
        let buf = ReplayBuffer(capacity: 10_000)
        // Only 3 of 4 active buckets populated. Bucket 3 (15–22)
        // intentionally empty.
        for i in 0..<5 {
            appendGame(to: buf, length: 100, outcome: 0,
                workerId: 0, gameIndex: UInt32(i), materialCount: 2)
        }
        for i in 0..<5 {
            appendGame(to: buf, length: 100, outcome: 0,
                workerId: 1, gameIndex: UInt32(i), materialCount: 7)
        }
        for i in 0..<5 {
            appendGame(to: buf, length: 100, outcome: 0,
                workerId: 2, gameIndex: UInt32(i), materialCount: 12)
        }

        // Balanced target across all 4 active buckets (V1 default).
        let n = ReplayBufferAnalyzer.materialBuckets.count
        let activeCount = Float(n - 1)
        var weights = [Float](repeating: 0, count: n)
        for i in 0..<(n - 1) { weights[i] = 1.0 / activeCount }
        buf.setSamplingConstraints(
            ReplayBuffer.SamplingConstraints(
                maxPerGame: .max,
                maxDrawPercent: 100,
                targetMeanGameLengthPlies: 0,
                materialBucketWeights: weights
            )
        )

        let batchSize = 600
        let drawn = drawBatch(buf, count: batchSize)
        XCTAssertTrue(drawn.ok)
        let result = buf.lastSamplingResult()
        let counts = result.achievedBucketCounts
        XCTAssertEqual(counts.reduce(0, +), batchSize,
            "stratified batch should fill exactly even with one bucket empty")
        // Empty bucket gets 0.
        XCTAssertEqual(counts[3], 0)
        XCTAssertEqual(counts[4], 0)
        // The 3 non-empty buckets absorb the deficit. Each should be
        // ~200 (batchSize / 3). The bucket-4 deficit reallocation runs
        // round-robin so the resulting per-bucket counts should be
        // within 1 of each other.
        XCTAssertEqual(counts[0], 200, accuracy: 1)
        XCTAssertEqual(counts[1], 200, accuracy: 1)
        XCTAssertEqual(counts[2], 200, accuracy: 1)
    }

    /// `SamplingConstraints.fromCurrentParameters()` should produce
    /// the V1 balanced target when the singleton toggle is on, and
    /// `nil` when it's off.
    @MainActor
    func testFromCurrentParametersMapsToggleToWeights() {
        let p = TrainingParameters.shared
        let originalToggle = p.replayBufferStratifyByMaterial
        defer { p.replayBufferStratifyByMaterial = originalToggle }

        p.replayBufferStratifyByMaterial = false
        let off = ReplayBuffer.SamplingConstraints.fromCurrentParameters()
        XCTAssertNil(off.materialBucketWeights)

        p.replayBufferStratifyByMaterial = true
        let on = ReplayBuffer.SamplingConstraints.fromCurrentParameters()
        let n = ReplayBufferAnalyzer.materialBuckets.count
        let weights = try! XCTUnwrap(on.materialBucketWeights,
            "stratification ON should produce non-nil weights")
        XCTAssertEqual(weights.count, n)
        let expectedShare = Float(1.0) / Float(n - 1)
        for i in 0..<(n - 1) {
            XCTAssertEqual(weights[i], expectedShare, accuracy: 1e-6,
                "bucket \(i) weight should be 1/activeCount")
        }
        XCTAssertEqual(weights[n - 1], 0,
            "structurally-unreachable bucket should have weight 0")
    }

    /// `isNoOp(forBatchSize:)` must return false whenever the
    /// stratification weights are set, even if every other knob is
    /// at its default. Guards against a regression where the sampler
    /// silently takes the legacy fast path while the user expects
    /// stratification.
    func testStratifiedNeverTakesFastPath() {
        let active = SamplingResultIsNoOpHelper.constraintsWithBalancedWeights()
        let inactive = ReplayBuffer.SamplingConstraints(
            maxPerGame: .max, maxDrawPercent: 100, targetMeanGameLengthPlies: 0
        )
        XCTAssertFalse(active.isNoOp(forBatchSize: 4096))
        XCTAssertTrue(inactive.isNoOp(forBatchSize: 4096))
    }
}

/// Small adapter so the noop-detection test can construct a
/// fully-formed `SamplingConstraints` without going through the
/// `@MainActor fromCurrentParameters()` path.
private enum SamplingResultIsNoOpHelper {
    static func constraintsWithBalancedWeights() -> ReplayBuffer.SamplingConstraints {
        let n = ReplayBufferAnalyzer.materialBuckets.count
        var weights = [Float](repeating: 0, count: n)
        let share = Float(1.0) / Float(n - 1)
        for i in 0..<(n - 1) { weights[i] = share }
        return ReplayBuffer.SamplingConstraints(
            maxPerGame: .max,
            maxDrawPercent: 100,
            targetMeanGameLengthPlies: 0,
            materialBucketWeights: weights
        )
    }
}
