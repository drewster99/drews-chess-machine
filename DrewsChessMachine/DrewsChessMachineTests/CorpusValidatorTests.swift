import XCTest
@testable import DrewsChessMachine

/// Exercises `CorpusValidator`: a clean corpus validates with correct totals,
/// stale `corpus.json` counts (the disk-full / crash signature) are flagged and
/// repaired by `--fix`, and a corrupted shard body is caught by the integrity
/// pass but (by design) not by the fast counts-only pass.
final class CorpusValidatorTests: XCTestCase {

    private var tmpRoot: URL!

    override func setUpWithError() throws {
        tmpRoot = FileManager.default.temporaryDirectory
            .appendingPathComponent("dcm-corpus-validator-tests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: tmpRoot, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        if let tmpRoot { try? FileManager.default.removeItem(at: tmpRoot) }
    }

    /// Build a small sealed corpus: one source per entry in `gamesPerSource`,
    /// each game holding `pliesPerGame` moves. Returns the corpus directory.
    private func makeCorpus(gamesPerSource: [Int], pliesPerGame: Int) throws -> URL {
        let corpus = try GameCorpus.create(name: "validator-test",
                                           comment: nil,
                                           shardSoftLimitBytes: 1 << 20,
                                           parentDirectory: tmpRoot)
        let move = ChessMove(fromRow: 1, fromCol: 4, toRow: 3, toCol: 4, promotion: nil)
        let moves = Array(repeating: move, count: pliesPerGame)
        for (idx, count) in gamesPerSource.enumerated() {
            try corpus.beginSource(kind: "test-\(idx)")
            for _ in 0..<count {
                try corpus.append(GameRecord(moves: moves, outcome: .draw))
            }
            try corpus.finishSource()
        }
        try corpus.seal()
        return corpus.directory
    }

    func testCleanCorpusValidatesWithCorrectTotals() throws {
        let dir = try makeCorpus(gamesPerSource: [5, 3], pliesPerGame: 4)
        let report = try CorpusValidator.validate(directory: dir)
        XCTAssertTrue(report.isValid, "clean corpus should validate; findings: \(report.findings.map(\.message))")
        XCTAssertEqual(report.errorCount, 0)
        XCTAssertEqual(report.totalGames, 8)
        XCTAssertEqual(report.totalPlies, 32)
        XCTAssertEqual(report.shardCount, 2)
        XCTAssertTrue(report.integrityVerified)
    }

    func testStaleCountsAreFlaggedAndFixed() throws {
        let dir = try makeCorpus(gamesPerSource: [7], pliesPerGame: 3)

        // Simulate the disk-full / crash signature: zero the persisted counts
        // while the sealed shards still hold the real games.
        var meta = try GameCorpus.loadMetadata(directory: dir)
        meta.sources[0].gamesAdded = 0
        meta.sources[0].pliesAdded = 0
        try GameCorpus.persistMetadata(meta, to: dir)

        // Without --fix: flagged as fixable, corpus not valid, but the totals in
        // the report come from the shards so they are already correct.
        let before = try CorpusValidator.validate(directory: dir, fix: false)
        XCTAssertFalse(before.isValid)
        XCTAssertTrue(before.findings.contains { $0.code == "source-game-count" && $0.fixable && !$0.fixed })
        XCTAssertTrue(before.findings.contains { $0.code == "source-ply-count" && $0.fixable && !$0.fixed })
        XCTAssertEqual(before.totalGames, 7)
        XCTAssertEqual(before.totalPlies, 21)

        // With --fix: repaired and persisted.
        let fixed = try CorpusValidator.validate(directory: dir, fix: true)
        XCTAssertTrue(fixed.findings.contains { $0.code == "source-game-count" && $0.fixed })
        XCTAssertTrue(fixed.isValid)

        let after = try GameCorpus.loadMetadata(directory: dir)
        XCTAssertEqual(after.sources[0].gamesAdded, 7)
        XCTAssertEqual(after.sources[0].pliesAdded, 21)

        // A fresh validation of the repaired corpus is clean.
        let reval = try CorpusValidator.validate(directory: dir)
        XCTAssertTrue(reval.isValid, "post-fix corpus should be clean; findings: \(reval.findings.map(\.message))")
    }

    func testCorruptShardByteIsDetectedByIntegrityCheckOnly() throws {
        let dir = try makeCorpus(gamesPerSource: [4], pliesPerGame: 5)
        let shard = try XCTUnwrap(
            FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)
                .first { $0.pathExtension == "dcmgames" && $0.lastPathComponent.hasPrefix("shard-") })

        // Flip the last body byte (just before the 64-byte trailer): definitely
        // inside a record, so it breaks the whole-shard SHA-256 (and a record CRC)
        // without touching the front header or trailer counts.
        var bytes = try Data(contentsOf: shard)
        let flipAt = bytes.count - GameCorpusShardFormat.trailerSize - 1
        XCTAssertGreaterThan(flipAt, GameCorpusShardFormat.frontHeaderSize)
        bytes[flipAt] ^= 0xFF
        try bytes.write(to: shard)

        // Full integrity mode reads the body and catches it as an error.
        let full = try CorpusValidator.validate(directory: dir, verifyIntegrity: true)
        XCTAssertFalse(full.isValid)
        XCTAssertTrue(full.findings.contains { $0.code == "shard-unreadable" && $0.severity == .error })

        // Quick mode reads only header + trailer, so a body-byte flip slips past
        // it (documented tradeoff).
        let quick = try CorpusValidator.validate(directory: dir, verifyIntegrity: false)
        XCTAssertFalse(quick.findings.contains { $0.code == "shard-unreadable" })
    }
}
