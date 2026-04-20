//
//  ReplayBufferTests.swift
//  DrewsChessMachineTests
//
//  Round-trip and version-rejection tests for `ReplayBuffer`'s on-disk
//  persistence (format v3). Because the buffer is the unit of training
//  data, a write/read regression here would corrupt every saved
//  session. Tests cover:
//   - empty buffer round-trips cleanly
//   - single-position round-trip preserves exact float bytes
//   - small buffer round-trip preserves all fields
//   - older format versions cleanly reject with `unsupportedVersion`
//

import XCTest
@testable import DrewsChessMachine

final class ReplayBufferTests: XCTestCase {

    // Per-test temp file path. Cleaned up in tearDown.
    private var tempFile: URL!

    override func setUpWithError() throws {
        let dir = FileManager.default.temporaryDirectory
        tempFile = dir.appendingPathComponent("dcm-replay-test-\(UUID().uuidString).bin")
    }

    override func tearDownWithError() throws {
        if let path = tempFile?.path, FileManager.default.fileExists(atPath: path) {
            try? FileManager.default.removeItem(at: tempFile)
        }
    }

    // MARK: - Round-trip

    func testEmptyBufferWriteRead() throws {
        let buffer = ReplayBuffer(capacity: 100)
        try buffer.write(to: tempFile)

        let restored = ReplayBuffer(capacity: 100)
        try restored.restore(from: tempFile)

        XCTAssertEqual(restored.count, 0, "Restored buffer should have no positions")
        XCTAssertEqual(restored.totalPositionsAdded, 0)
    }

    func testSinglePositionWriteRead() throws {
        let buffer = ReplayBuffer(capacity: 100)

        // Append a single fake position with deterministic content.
        let boardFloats = makeFakeBoard(seed: 42)
        var move: Int32 = 1234
        var vBaseline: Float = 0.42

        boardFloats.withUnsafeBufferPointer { boardsBuf in
            withUnsafePointer(to: &move) { moveP in
                withUnsafePointer(to: &vBaseline) { vP in
                    buffer.append(
                        boards: boardsBuf.baseAddress!,
                        policyIndices: moveP,
                        vBaselines: vP,
                        outcome: 1.0,
                        count: 1
                    )
                }
            }
        }

        try buffer.write(to: tempFile)

        let restored = ReplayBuffer(capacity: 100)
        try restored.restore(from: tempFile)

        XCTAssertEqual(restored.count, 1)

        // Sample what we wrote and compare bytes.
        var sampledBoard = [Float](repeating: 0, count: ReplayBuffer.floatsPerBoard)
        var sampledMove: Int32 = 0
        var sampledZ: Float = 0
        var sampledV: Float = 0

        let success = sampledBoard.withUnsafeMutableBufferPointer { boardBuf -> Bool in
            withUnsafeMutablePointer(to: &sampledMove) { moveP in
                withUnsafeMutablePointer(to: &sampledZ) { zP in
                    withUnsafeMutablePointer(to: &sampledV) { vP in
                        restored.sample(
                            count: 1,
                            intoBoards: boardBuf.baseAddress!,
                            moves: moveP,
                            zs: zP,
                            vBaselines: vP
                        )
                    }
                }
            }
        }

        XCTAssertTrue(success)
        XCTAssertEqual(sampledMove, 1234, "Move index round-trip")
        XCTAssertEqual(sampledZ, 1.0, "Outcome round-trip")
        XCTAssertEqual(sampledV, 0.42, accuracy: 1e-6, "vBaseline round-trip")
        XCTAssertEqual(sampledBoard, boardFloats, "Board floats round-trip exactly")
    }

    // MARK: - Version rejection

    func testV2FileRejectedWithUnsupportedVersion() throws {
        // Synthesize a v2 file header by hand and attempt to restore.
        // v2 layout: 8-byte magic, 4-byte version=2, 4-byte pad,
        // then 5 × Int64 fields.
        let magic = "DCMRPBUF".data(using: .utf8)!
        var version: UInt32 = 2
        var pad: UInt32 = 0
        var fpb: Int64 = 1152  // v2 board stride
        var cap: Int64 = 100
        var stored: Int64 = 0
        var writeIdx: Int64 = 0
        var totalAdded: Int64 = 0

        var header = Data()
        header.append(magic)
        withUnsafeBytes(of: &version) { header.append(contentsOf: $0) }
        withUnsafeBytes(of: &pad) { header.append(contentsOf: $0) }
        withUnsafeBytes(of: &fpb) { header.append(contentsOf: $0) }
        withUnsafeBytes(of: &cap) { header.append(contentsOf: $0) }
        withUnsafeBytes(of: &stored) { header.append(contentsOf: $0) }
        withUnsafeBytes(of: &writeIdx) { header.append(contentsOf: $0) }
        withUnsafeBytes(of: &totalAdded) { header.append(contentsOf: $0) }

        try header.write(to: tempFile)

        let restored = ReplayBuffer(capacity: 100)
        XCTAssertThrowsError(try restored.restore(from: tempFile)) { error in
            guard case ReplayBuffer.PersistenceError.unsupportedVersion(let v) = error else {
                XCTFail("Expected unsupportedVersion(2), got \(error)")
                return
            }
            XCTAssertEqual(v, 2, "Should report the rejected version number")
        }
    }

    func testBadMagicRejected() throws {
        // Write 8 bytes of wrong magic followed by zeroes.
        var data = Data(count: 64)
        let bogus = "BOGUSMAG".data(using: .utf8)!
        data.replaceSubrange(0..<8, with: bogus)
        try data.write(to: tempFile)

        let restored = ReplayBuffer(capacity: 100)
        XCTAssertThrowsError(try restored.restore(from: tempFile)) { error in
            guard case ReplayBuffer.PersistenceError.badMagic = error else {
                XCTFail("Expected badMagic, got \(error)")
                return
            }
        }
    }

    func testTruncatedHeaderRejected() throws {
        // Only 16 bytes — well short of the 56-byte header.
        let short = Data(count: 16)
        try short.write(to: tempFile)

        let restored = ReplayBuffer(capacity: 100)
        XCTAssertThrowsError(try restored.restore(from: tempFile)) { error in
            guard case ReplayBuffer.PersistenceError.truncatedHeader = error else {
                XCTFail("Expected truncatedHeader, got \(error)")
                return
            }
        }
    }

    // MARK: - Helpers

    private func makeFakeBoard(seed: UInt64) -> [Float] {
        var rng = SeededRNG(seed: seed)
        let count = ReplayBuffer.floatsPerBoard
        var board = [Float](repeating: 0, count: count)
        for i in 0..<count {
            // Deterministic float in [0, 1).
            board[i] = Float(rng.next()) / Float(UInt32.max)
        }
        return board
    }
}

/// Deterministic RNG so the test sequence is reproducible.
private struct SeededRNG {
    var state: UInt64
    init(seed: UInt64) { self.state = seed | 1 }
    mutating func next() -> UInt32 {
        state &*= 6364136223846793005
        state &+= 1442695040888963407
        return UInt32(state >> 32)
    }
}
