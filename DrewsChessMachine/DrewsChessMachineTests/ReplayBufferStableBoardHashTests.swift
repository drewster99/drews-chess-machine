//
//  ReplayBufferStableBoardHashTests.swift
//  DrewsChessMachineTests
//
//  `ReplayBuffer.hashBoard` used Swift's `Hasher`, which is keyed randomly in
//  every process. The per-slot hashes are saved with the buffer and restored
//  as they are, so after a resume a position already in the buffer and the
//  same position inserted again landed under two different keys. The hash is
//  now a fixed function of the board bytes, files written with it are format
//  v8, and a v7 file (process-keyed hashes) has every hash recomputed from its
//  stored board on load.
//

import CryptoKit
import XCTest
@testable import DrewsChessMachine

final class ReplayBufferStableBoardHashTests: XCTestCase {

    private var root: URL!

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory
            .appendingPathComponent("ReplayBufferStableBoardHashTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try FileManager.default.removeItem(at: root)
    }

    // MARK: - The hash is a fixed function of the bytes

    /// Golden values computed by an independent implementation of the same
    /// definition (SplitMix64 mix over little-endian 8-byte words from a fixed
    /// key, a 4-byte tail word, then the byte count).
    func testBoardHashIsAFixedFunctionOfTheBoardBytes() {
        let patterned = (0..<1920).map { Float($0 % 97) * 0.25 - 3.0 }
        XCTAssertEqual(Self.hash(patterned), 0x8045_6DBA_D029_9BE2)
        XCTAssertEqual(Self.hash([1, 2, 3, 4, 5, 6, 7]), 0x5853_FAF7_B0B1_D9C5, "odd float count: 4-byte tail word")
        XCTAssertEqual(Self.hash([]), 0x2CB0_F69F_4ABE_A221)
        XCTAssertEqual(Self.hash([Float](repeating: 0, count: 1920)), 0x32F9_0D03_4454_4646)
    }

    // MARK: - File format

    func testABufferIsWrittenAsFormatEight() throws {
        let url = root.appendingPathComponent("buffer.bin")
        let buffer = ReplayBuffer(capacity: 16)
        try Self.appendGame(to: buffer, boards: Self.distinctBoards(count: 3, floatsPerBoard: buffer.floatsPerBoard),
                            hashes: [11, 22, 33])
        try buffer.write(to: url)
        XCTAssertEqual(try Self.fileVersion(at: url), 8)
    }

    /// A format-v7 file carries hashes from a process-keyed `Hasher`, which no
    /// later process can reproduce. Loading it recomputes every slot's hash from
    /// the slot's stored board, so the restored positions count under the same
    /// keys as new inserts of the same boards.
    func testAFormatSevenBufferHasItsHashesRecomputedFromTheBoardsOnLoad() throws {
        let source = ReplayBuffer(capacity: 16)
        let boards = Self.distinctBoards(count: 3, floatsPerBoard: source.floatsPerBoard)
        let staleHashes: [UInt64] = [0xDEAD_0001, 0xDEAD_0002, 0xDEAD_0003]
        try Self.appendGame(to: source, boards: boards, hashes: staleHashes)
        let url = root.appendingPathComponent("legacy.bin")
        try source.write(to: url)
        try Self.rewriteAsFormatSeven(url: url)

        let restored = ReplayBuffer(capacity: 16)
        try restored.restore(from: url)

        for (index, board) in boards.enumerated() {
            let stable = Self.hash(board)
            XCTAssertEqual(restored.bufferedPositionStats(forHash: stable)?.count, 1,
                           "slot \(index) must count under the stable hash of its board")
            XCTAssertNil(restored.bufferedPositionStats(forHash: staleHashes[index]),
                         "the file's process-keyed hash \(index) must not survive the load")
        }
        XCTAssertEqual(restored.uniquePositionCount, boards.count)
    }

    /// A current-format file is trusted as written: its hashes were produced by
    /// the stable hash, so the load does not spend time recomputing them.
    func testAFormatEightBufferKeepsItsSavedHashes() throws {
        let source = ReplayBuffer(capacity: 16)
        let boards = Self.distinctBoards(count: 2, floatsPerBoard: source.floatsPerBoard)
        let savedHashes: [UInt64] = [0xABCD_0001, 0xABCD_0002]
        try Self.appendGame(to: source, boards: boards, hashes: savedHashes)
        let url = root.appendingPathComponent("current.bin")
        try source.write(to: url)

        let restored = ReplayBuffer(capacity: 16)
        try restored.restore(from: url)
        for hash in savedHashes {
            XCTAssertEqual(restored.bufferedPositionStats(forHash: hash)?.count, 1)
        }
    }

    // MARK: - Helpers

    private static func hash(_ floats: [Float]) -> UInt64 {
        floats.withUnsafeBufferPointer { buffer in
            guard let base = buffer.baseAddress else {
                return withUnsafePointer(to: Float(0)) { ReplayBuffer.hashBoard($0, count: 0) }
            }
            return ReplayBuffer.hashBoard(base, count: buffer.count)
        }
    }

    private static func distinctBoards(count: Int, floatsPerBoard: Int) -> [[Float]] {
        (0..<count).map { boardIndex in
            (0..<floatsPerBoard).map { Float(($0 &* 31 &+ boardIndex &* 7) % 11) }
        }
    }

    private enum FixtureError: Error { case emptyColumn, mismatchedColumns }

    private static func appendGame(to buffer: ReplayBuffer, boards: [[Float]], hashes: [UInt64]) throws {
        let count = boards.count
        guard count > 0, hashes.count == count else { throw FixtureError.mismatchedColumns }
        let flat = boards.flatMap { $0 }
        let policy = [Int32](repeating: 0, count: count)
        let plies = (0..<count).map { UInt16($0) }
        let taus = [Float](repeating: 1, count: count)
        let materials = [UInt8](repeating: 32, count: count)
        let outcomes = [Float](repeating: 0, count: count)
        try flat.withUnsafeBufferPointer { boardsBuffer in
            try policy.withUnsafeBufferPointer { policyBuffer in
                try plies.withUnsafeBufferPointer { pliesBuffer in
                    try taus.withUnsafeBufferPointer { tausBuffer in
                        try hashes.withUnsafeBufferPointer { hashesBuffer in
                            try materials.withUnsafeBufferPointer { materialsBuffer in
                                try outcomes.withUnsafeBufferPointer { outcomesBuffer in
                                    guard let boardsBase = boardsBuffer.baseAddress,
                                          let policyBase = policyBuffer.baseAddress,
                                          let pliesBase = pliesBuffer.baseAddress,
                                          let tausBase = tausBuffer.baseAddress,
                                          let hashesBase = hashesBuffer.baseAddress,
                                          let materialsBase = materialsBuffer.baseAddress,
                                          let outcomesBase = outcomesBuffer.baseAddress else {
                                        throw FixtureError.emptyColumn
                                    }
                                    buffer.append(
                                        boards: boardsBase,
                                        policyIndices: policyBase,
                                        plyIndices: pliesBase,
                                        samplingTaus: tausBase,
                                        stateHashes: hashesBase,
                                        materialCounts: materialsBase,
                                        gameLength: UInt16(count),
                                        workerId: 1,
                                        intraWorkerGameIndex: 0,
                                        outcomes: outcomesBase,
                                        count: count
                                    )
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    private static let versionByteOffset = 8
    private static let trailerByteCount = 32

    private static func fileVersion(at url: URL) throws -> UInt32 {
        let data = try Data(contentsOf: url)
        return data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: versionByteOffset, as: UInt32.self) }
    }

    /// Rewrite a buffer file's version field as 7 and re-seal its SHA-256
    /// trailer, giving exactly the bytes an earlier build would have written
    /// for the same contents (the layout did not change between the two
    /// formats; only what the hash column means did).
    private static func rewriteAsFormatSeven(url: URL) throws {
        var data = try Data(contentsOf: url)
        var seven: UInt32 = 7
        withUnsafeBytes(of: &seven) { bytes in
            data.replaceSubrange(versionByteOffset..<(versionByteOffset + 4), with: bytes)
        }
        let contentEnd = data.count - trailerByteCount
        let digest = SHA256.hash(data: data.prefix(contentEnd))
        data.replaceSubrange(contentEnd..<data.count, with: Data(digest))
        try FileManager.default.removeItem(at: url)
        try data.write(to: url, options: .withoutOverwriting)
    }
}
