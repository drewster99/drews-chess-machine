//
//  ReplayBufferRestoreExpectationTests.swift
//  DrewsChessMachineTests
//
//  A GUI resume pairs a session's replay buffer file with its session.json
//  by the buffer's lifetime position count. The check used to run after
//  the restore had already filled the ring, so a mismatched file (another
//  session's buffer, or one that SHA-matched after damage) left its
//  positions in the live buffer while the resume reported "continuing with
//  empty buffer". The expected count is now checked from the file's header
//  before anything is mutated.
//

import XCTest
@testable import DrewsChessMachine

@MainActor
final class ReplayBufferRestoreExpectationTests: XCTestCase {

    private func writtenBuffer() throws -> (url: URL, totalPositionsAdded: Int, cleanup: () throws -> Void) {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("ReplayBufferRestoreExpectationTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
        let url = directory.appendingPathComponent("replay_buffer.bin")
        let buffer = try ResumeEquivalenceTests.fixtureBuffer(sampler: DCMRandom(seed: 5))
        let written = try buffer.write(to: url)
        return (url, written.totalPositionsAdded, { try FileManager.default.removeItem(at: directory) })
    }

    func testRestoreWithTheExpectedTotalSucceeds() throws {
        let file = try writtenBuffer()
        defer { do { try file.cleanup() } catch { XCTFail("\(error)") } }
        let restored = ReplayBuffer(capacity: 512, inputEncoding: ResumeEquivalenceTests.architecture.inputEncoding,
                                    sampler: DCMRandom(seed: 6))
        try restored.restore(from: file.url, expectedTotalPositionsAdded: file.totalPositionsAdded)
        XCTAssertEqual(restored.stateSnapshot().totalPositionsAdded, file.totalPositionsAdded)
    }

    func testRestoreWithMismatchedTotalLeavesBufferEmpty() throws {
        let file = try writtenBuffer()
        defer { do { try file.cleanup() } catch { XCTFail("\(error)") } }
        let restored = ReplayBuffer(capacity: 512, inputEncoding: ResumeEquivalenceTests.architecture.inputEncoding,
                                    sampler: DCMRandom(seed: 6))
        XCTAssertThrowsError(try restored.restore(from: file.url,
                                                  expectedTotalPositionsAdded: file.totalPositionsAdded + 1))
        XCTAssertEqual(restored.stateSnapshot().storedCount, 0)
        XCTAssertEqual(restored.stateSnapshot().totalPositionsAdded, 0)
    }
}
