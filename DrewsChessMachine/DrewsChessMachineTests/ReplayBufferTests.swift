//
//  ReplayBufferTests.swift
//  DrewsChessMachineTests
//
//  Round-trip, integrity, and corruption-rejection tests for
//  `ReplayBuffer`'s on-disk persistence (current format v6). Because
//  the buffer is the unit of training data, a write/read regression
//  here would corrupt every saved session. Tests cover:
//   - empty buffer round-trips cleanly through SHA verify
//   - single-position round-trip preserves exact float bytes
//   - older format versions cleanly reject with
//     `unsupportedVersion`
//   - tampered body byte → hashMismatch
//   - file truncated by one byte → sizeMismatch
//   - file with trailing garbage → sizeMismatch
//   - header with Int64.max capacity → upperBoundExceeded (not crash)
//   - bad magic / truncated header still rejected
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
        var ply: UInt16 = 0
        var tau: Float = 1.0
        var hash: UInt64 = 0xCAFE_BABE_DEAD_BEEF
        var mat: UInt8 = 32

        boardFloats.withUnsafeBufferPointer { boardsBuf in
            withUnsafePointer(to: &move) { moveP in
                withUnsafePointer(to: &ply) { plyP in
                    withUnsafePointer(to: &tau) { tauP in
                        withUnsafePointer(to: &hash) { hashP in
                            withUnsafePointer(to: &mat) { matP in
                                buffer.append(
                                    boards: boardsBuf.baseAddress!,
                                    policyIndices: moveP,
                                    plyIndices: plyP,
                                    samplingTaus: tauP,
                                    stateHashes: hashP,
                                    materialCounts: matP,
                                    gameLength: 1,
                                    workerId: 0,
                                    intraWorkerGameIndex: 0,
                                    outcome: 1.0,
                                    count: 1
                                )
                            }
                        }
                    }
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

        let success = sampledBoard.withUnsafeMutableBufferPointer { boardBuf -> Bool in
            withUnsafeMutablePointer(to: &sampledMove) { moveP in
                withUnsafeMutablePointer(to: &sampledZ) { zP in
                    restored.sample(
                        count: 1,
                        intoBoards: boardBuf.baseAddress!,
                        moves: moveP,
                        zs: zP
                    )
                }
            }
        }

        XCTAssertTrue(success)
        XCTAssertEqual(sampledMove, 1234, "Move index round-trip")
        XCTAssertEqual(sampledZ, 1.0, "Outcome round-trip")
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

    func testV3FileRejectedWithUnsupportedVersion() throws {
        // Synthesize a v3-formatted file (no SHA trailer) and attempt to
        // restore. v3 shared the current header layout but had no
        // SHA-256 trailer and no size-invariant check. The reader
        // must reject v3 cleanly via `unsupportedVersion(3)`.
        let magic = "DCMRPBUF".data(using: .utf8)!
        var version: UInt32 = 3
        var pad: UInt32 = 0
        var fpb: Int64 = Int64(ReplayBuffer.floatsPerBoard)
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
                XCTFail("Expected unsupportedVersion(3), got \(error)")
                return
            }
            XCTAssertEqual(v, 3, "Should report the rejected version number")
        }
    }

    func testV4FileRejectedWithUnsupportedVersion() throws {
        let magic = "DCMRPBUF".data(using: .utf8)!
        var version: UInt32 = 4
        var pad: UInt32 = 0
        var fpb: Int64 = Int64(ReplayBuffer.floatsPerBoard)
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
                XCTFail("Expected unsupportedVersion(4), got \(error)")
                return
            }
            XCTAssertEqual(v, 4)
        }
    }

    func testV5FileRejectedWithUnsupportedVersion() throws {
        let magic = "DCMRPBUF".data(using: .utf8)!
        var version: UInt32 = 5
        var pad: UInt32 = 0
        var fpb: Int64 = Int64(ReplayBuffer.floatsPerBoard)
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
                XCTFail("Expected unsupportedVersion(5), got \(error)")
                return
            }
            XCTAssertEqual(v, 5)
        }
    }

    // MARK: - Current-format integrity checks

    func testSHAMismatchRejected() throws {
        // Write a valid file with one real position, then flip a
        // single byte somewhere in the body (past the 56-byte header
        // and before the 32-byte trailer). Decode must throw
        // hashMismatch — the SHA-256 trailer is the first line of
        // defense against any unnoticed corruption.
        let buffer = ReplayBuffer(capacity: 10)
        try appendOnePosition(to: buffer, seed: 7)
        try buffer.write(to: tempFile)

        var bytes = try Data(contentsOf: tempFile)
        XCTAssertGreaterThan(bytes.count, 56 + 32, "File should have a body")

        // Flip the lowest-order bit of the first body byte (offset 56).
        let tamperOffset = 56
        bytes[tamperOffset] ^= 0x01
        try bytes.write(to: tempFile)

        let restored = ReplayBuffer(capacity: 10)
        XCTAssertThrowsError(try restored.restore(from: tempFile)) { error in
            guard case ReplayBuffer.PersistenceError.hashMismatch = error else {
                XCTFail("Expected hashMismatch, got \(error)")
                return
            }
        }
    }

    func testSizeMismatchOnTruncation() throws {
        // Truncate a valid file by one byte. The size-equality
        // check runs before the SHA pass, so the error must be
        // sizeMismatch (not hashMismatch).
        let buffer = ReplayBuffer(capacity: 10)
        try appendOnePosition(to: buffer, seed: 11)
        try buffer.write(to: tempFile)

        let original = try Data(contentsOf: tempFile)
        let truncated = original.prefix(original.count - 1)
        try truncated.write(to: tempFile)

        let restored = ReplayBuffer(capacity: 10)
        XCTAssertThrowsError(try restored.restore(from: tempFile)) { error in
            guard case ReplayBuffer.PersistenceError.sizeMismatch = error else {
                XCTFail("Expected sizeMismatch, got \(error)")
                return
            }
        }
    }

    func testSizeMismatchOnTrailingGarbage() throws {
        // Append a single byte past the valid end. The SHA trailer
        // is still at its correct offset, but the file is now one
        // byte too long — strict `==` size check must reject.
        let buffer = ReplayBuffer(capacity: 10)
        try appendOnePosition(to: buffer, seed: 13)
        try buffer.write(to: tempFile)

        var bytes = try Data(contentsOf: tempFile)
        bytes.append(0xFF)
        try bytes.write(to: tempFile)

        let restored = ReplayBuffer(capacity: 10)
        XCTAssertThrowsError(try restored.restore(from: tempFile)) { error in
            guard case ReplayBuffer.PersistenceError.sizeMismatch = error else {
                XCTFail("Expected sizeMismatch, got \(error)")
                return
            }
        }
    }

    func testV6FileRejectedWithUnsupportedVersion() throws {
        // v6 was the last format before the W/D/L value-head rewrite
        // dropped the per-slot vBaseline column (v7). Loading a v6
        // file must fail cleanly via `unsupportedVersion(6)`.
        let magic = "DCMRPBUF".data(using: .utf8)!
        var version: UInt32 = 6
        var pad: UInt32 = 0
        var fpb: Int64 = Int64(ReplayBuffer.floatsPerBoard)
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
                XCTFail("Expected unsupportedVersion(6), got \(error)")
                return
            }
            XCTAssertEqual(v, 6)
        }
    }

    func testUpperBoundRejectedOnCapacity() throws {
        // Synthesize a current-version header with capacity = Int64.max.
        // The upper-bound cap check runs after the header is parsed
        // but before any allocation or seek arithmetic — it must
        // throw upperBoundExceeded, not crash.
        let magic = "DCMRPBUF".data(using: .utf8)!
        var version: UInt32 = 7
        var pad: UInt32 = 0
        var fpb: Int64 = Int64(ReplayBuffer.floatsPerBoard)
        var cap: Int64 = .max
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
        // No SHA trailer — but the upper-bound check runs before the
        // SHA pass, so this rejects at the cap check regardless.
        try header.write(to: tempFile)

        let restored = ReplayBuffer(capacity: 100)
        XCTAssertThrowsError(try restored.restore(from: tempFile)) { error in
            guard case ReplayBuffer.PersistenceError.upperBoundExceeded(let field, _, _) = error else {
                XCTFail("Expected upperBoundExceeded, got \(error)")
                return
            }
            XCTAssertEqual(field, "capacity",
                "Should name the offending field as 'capacity'")
        }
    }

    // MARK: - Append helper

    /// Push one deterministic position into `buffer` so it has a
    /// non-empty body (and therefore a SHA trailer that actually
    /// depends on content). Used by the integrity-check tests that
    /// need something to corrupt.
    private func appendOnePosition(to buffer: ReplayBuffer, seed: UInt64) throws {
        let boardFloats = makeFakeBoard(seed: seed)
        var move: Int32 = Int32(seed % UInt64(ChessNetwork.policySize))
        var ply: UInt16 = UInt16(seed % 100)
        var tau: Float = 1.0
        var hash: UInt64 = seed
        var mat: UInt8 = 32
        boardFloats.withUnsafeBufferPointer { boardsBuf in
            withUnsafePointer(to: &move) { moveP in
                withUnsafePointer(to: &ply) { plyP in
                    withUnsafePointer(to: &tau) { tauP in
                        withUnsafePointer(to: &hash) { hashP in
                            withUnsafePointer(to: &mat) { matP in
                                buffer.append(
                                    boards: boardsBuf.baseAddress!,
                                    policyIndices: moveP,
                                    plyIndices: plyP,
                                    samplingTaus: tauP,
                                    stateHashes: hashP,
                                    materialCounts: matP,
                                    gameLength: 1,
                                    workerId: 0,
                                    intraWorkerGameIndex: UInt32(seed & 0xFFFF),
                                    outcome: 1.0,
                                    count: 1
                                )
                            }
                        }
                    }
                }
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
