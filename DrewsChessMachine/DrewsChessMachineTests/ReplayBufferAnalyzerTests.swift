//
//  ReplayBufferAnalyzerTests.swift
//  DrewsChessMachineTests
//
//  The entropy probe's picks are a function of its stream and of the
//  buffer's positions in age order — never of where the ring happens to
//  store them. An uninterrupted run whose ring has wrapped and a resumed run
//  whose restore compacted the same positions oldest-first must probe the
//  same positions at the same step.
//

import XCTest
@testable import DrewsChessMachine

final class ReplayBufferAnalyzerTests: XCTestCase {

    private static let encoding = InputEncoding.basic30

    /// Append `rows` one position per game, each board filled with its row
    /// number so every position is distinct, all in one material bucket.
    private static func append(rows: Range<Int>, to buffer: ReplayBuffer) {
        for row in rows {
            let board = [Float](repeating: Float(row), count: buffer.floatsPerBoard)
            var move = Int32(row)
            var ply = UInt16(0)
            var tau = Float(1)
            var hash = UInt64(row)
            var material = UInt8(32)
            var outcome = Float(0)
            board.withUnsafeBufferPointer { boardPointer in
                guard let boardBase = boardPointer.baseAddress else {
                    preconditionFailure("a board has floatsPerBoard > 0 floats")
                }
                buffer.append(
                    boards: boardBase, policyIndices: &move, plyIndices: &ply, samplingTaus: &tau,
                    stateHashes: &hash, materialCounts: &material, gameLength: 1,
                    workerId: 0, intraWorkerGameIndex: UInt32(row), outcomes: &outcome, count: 1)
            }
        }
    }

    func testEntropyProbePicksDependOnAgeOrderNotRingLayout() {
        let capacity = 8
        // Wrapped: twelve positions through an eight-slot ring, so the oldest
        // surviving position sits mid-ring.
        let wrapped = ReplayBuffer(capacity: capacity, inputEncoding: Self.encoding, sampler: DCMRandom(seed: 1))
        Self.append(rows: 0..<12, to: wrapped)
        // Compacted: the same surviving positions, oldest in slot 0 — the
        // layout a restore produces.
        let compacted = ReplayBuffer(capacity: capacity, inputEncoding: Self.encoding, sampler: DCMRandom(seed: 1))
        Self.append(rows: 4..<12, to: compacted)
        XCTAssertEqual(wrapped.count, compacted.count)

        let wrappedPicks = ReplayBufferAnalyzer.entropyProbeSamples(
            buffer: wrapped, perBucketTarget: 3, sampleRandom: DCMRandom(seed: 77))
        let compactedPicks = ReplayBufferAnalyzer.entropyProbeSamples(
            buffer: compacted, perBucketTarget: 3, sampleRandom: DCMRandom(seed: 77))
        XCTAssertEqual(wrappedPicks.flatMap(\.boards).count, 3)
        XCTAssertEqual(wrappedPicks, compactedPicks)
    }
}
