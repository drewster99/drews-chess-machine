//
//  BatchHashChainTests.swift
//  DrewsChessMachineTests
//
//  Batch hashes and their window chain (GPU fault forensics plan, Part B):
//  the same bytes always hash the same, any changed value changes the hash,
//  and a chain matches between two runs exactly when every batch of the
//  window matched — including a run that resumed at the window boundary, as
//  every exact resume from a checkpoint does.
//

import XCTest
@testable import DrewsChessMachine

final class BatchHashChainTests: XCTestCase {

    private static func batch(seed: Int, count: Int = 8) -> (boards: [Float], moves: [Int32], outcomes: [Float]) {
        let boards = (0..<(count * 30)).map { Float(($0 * 7 + seed * 13) % 23) / 23 }
        let moves = (0..<count).map { Int32(($0 * 31 + seed) % 4864) }
        let outcomes = (0..<count).map { Float((($0 + seed) % 3) - 1) }
        return (boards, moves, outcomes)
    }

    func testSameBytesSameHashAndAnyChangedValueChangesIt() {
        let b = Self.batch(seed: 1)
        let hash = BatchHashChain.batchHash(boards: b.boards, moves: b.moves, outcomes: b.outcomes)
        XCTAssertEqual(hash.count, 64)
        XCTAssertEqual(hash, BatchHashChain.batchHash(boards: b.boards, moves: b.moves, outcomes: b.outcomes))

        var boards = b.boards
        boards[17] = boards[17].nextUp
        XCTAssertNotEqual(hash, BatchHashChain.batchHash(boards: boards, moves: b.moves, outcomes: b.outcomes))
        var moves = b.moves
        moves[3] += 1
        XCTAssertNotEqual(hash, BatchHashChain.batchHash(boards: b.boards, moves: moves, outcomes: b.outcomes))
        var outcomes = b.outcomes
        outcomes[0] = -0.013 // the draw-penalty rewrite of a drawn position
        XCTAssertNotEqual(hash, BatchHashChain.batchHash(boards: b.boards, moves: b.moves, outcomes: outcomes))
    }

    func testWindowSeedsDifferPerWindow() {
        XCTAssertNotEqual(BatchHashChain.windowSeed(windowStart: 0), BatchHashChain.windowSeed(windowStart: 1_000))
        XCTAssertEqual(BatchHashChain.windowSeed(windowStart: 45_000), BatchHashChain.windowSeed(windowStart: 45_000))
    }

    private static func run(steps: ClosedRange<Int>, changingStep: Int? = nil) async -> [Int: BatchHashChain.Entry] {
        let chain = BatchHashChain()
        for step in steps {
            let b = batch(seed: step == changingStep ? -step : step)
            chain.submit(trainerStep: step, boards: b.boards, moves: b.moves, outcomes: b.outcomes)
        }
        var out: [Int: BatchHashChain.Entry] = [:]
        for entry in await chain.recentEntries() {
            out[entry.trainerStep] = entry
        }
        return out
    }

    func testAResumeAtTheWindowBoundaryReproducesTheOriginalChain() async {
        let original = await Self.run(steps: 995...1_210)
        let resumed = await Self.run(steps: 1_001...1_210)
        for step in 1_001...1_210 {
            XCTAssertEqual(resumed[step], original[step], "step \(step)")
            XCTAssertNotNil(resumed[step]?.chain, "step \(step)")
        }
    }

    func testAMidWindowStartIsPartialUntilTheNextWindow() async {
        let entries = await Self.run(steps: 1_500...2_003)
        XCTAssertNil(entries[1_500]?.chain)
        XCTAssertNil(entries[2_000]?.chain)
        XCTAssertNotNil(entries[2_001]?.chain)
        XCTAssertNotNil(entries[2_003]?.chain)
        XCTAssertEqual(entries[1_500]?.logLine.hasSuffix("batchChain=partial"), true)
    }

    func testOneDifferentBatchChangesEveryLaterChainInItsWindowOnly() async {
        let original = await Self.run(steps: 1_001...2_100)
        let changed = await Self.run(steps: 1_001...2_100, changingStep: 1_500)
        XCTAssertEqual(changed[1_499]?.chain, original[1_499]?.chain)
        XCTAssertNotEqual(changed[1_500]?.batchHash, original[1_500]?.batchHash)
        XCTAssertNotEqual(changed[1_501]?.chain, original[1_501]?.chain)
        XCTAssertEqual(changed[1_501]?.batchHash, original[1_501]?.batchHash)
        XCTAssertNotEqual(changed[2_000]?.chain, original[2_000]?.chain)
        // The next window starts clean.
        XCTAssertEqual(changed[2_001]?.chain, original[2_001]?.chain)
    }

    func testAClockRewindMakesTheChainPartial() async {
        let chain = BatchHashChain()
        for step in [1_001, 1_002, 1_003, 1_002, 1_003] {
            let b = Self.batch(seed: step)
            chain.submit(trainerStep: step, boards: b.boards, moves: b.moves, outcomes: b.outcomes)
        }
        let entries = await chain.recentEntries()
        XCTAssertEqual(entries.map(\.trainerStep), [1_001, 1_002, 1_003, 1_002, 1_003])
        XCTAssertNotNil(entries[2].chain)
        XCTAssertNil(entries[3].chain, "a rewound step can't continue the chain")
        XCTAssertNil(entries[4].chain)
        let latest = await chain.entry(forTrainerStep: 1_003)
        XCTAssertNil(latest?.chain, "the newest entry for a step wins")
    }

    func testLinesEveryHundredStepsAndTheirFormat() {
        XCTAssertFalse(BatchHashChain.logsLine(atTrainerStep: 0))
        XCTAssertFalse(BatchHashChain.logsLine(atTrainerStep: 150))
        XCTAssertTrue(BatchHashChain.logsLine(atTrainerStep: 45_900))
        let entry = BatchHashChain.Entry(
            trainerStep: 45_900, batchHash: String(repeating: "ab", count: 32), chain: String(repeating: "cd", count: 32))
        XCTAssertEqual(entry.logLine,
                       "[BATCH-HASH] trainerStep=45900 batchHash=abababababababab batchChain=cdcdcdcdcdcdcdcd")
    }
}
