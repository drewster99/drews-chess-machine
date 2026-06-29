import XCTest
@testable import DrewsChessMachine

final class GameCorpusShardTests: XCTestCase {

    // A fresh temp dir per test instance (XCTest news up one instance per test).
    private let tempDir = FileManager.default.temporaryDirectory
        .appendingPathComponent("dcm-shard-tests-\(UUID().uuidString)", isDirectory: true)

    override func setUpWithError() throws {
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        do { try FileManager.default.removeItem(at: tempDir) } catch { /* best-effort cleanup */ }
    }

    private func sampleGame(_ seed: Int) -> GameRecord {
        var moves: [ChessMove] = []
        let n = 4 + (seed % 11)
        for i in 0..<n {
            let f = (seed * 7 + i) % 64
            let t = (seed * 13 + i + 1) % 64
            let promo: PieceType? = (i % 4 == 3) ? .queen : nil
            moves.append(ChessMove(fromRow: f / 8, fromCol: f % 8, toRow: t / 8, toCol: t % 8, promotion: promo))
        }
        let outcome: GameOutcome = [.whiteWin, .draw, .blackWin][seed % 3]
        return GameRecord(moves: moves, outcome: outcome, terminationReason: .checkmate)
    }

    private func makeHeader() -> GameCorpusShardFormat.FrontHeader {
        GameCorpusShardFormat.FrontHeader(corpusID: "20260620-000000-ABC123",
                                          sourceID: "src-TESTSRC1",
                                          shardSeq: 0,
                                          createdAtUnix: 1_700_000_000)
    }

    func testWriteSealReadRoundTrip() throws {
        let openURL = tempDir.appendingPathComponent("shard-00000.dcmgames.open")
        let writer = try ShardWriter(creatingAt: openURL, header: makeHeader())
        let games = (0..<25).map { sampleGame($0) }
        for g in games { try writer.append(g) }
        XCTAssertEqual(writer.gameCount, games.count)

        let sealedURL = try writer.seal(sealUnix: 1_700_000_500)
        XCTAssertEqual(sealedURL.lastPathComponent, "shard-00000.dcmgames")
        XCTAssertFalse(FileManager.default.fileExists(atPath: openURL.path))

        let shard = try GameCorpusShardIO.readSealed(at: sealedURL)
        XCTAssertEqual(shard.games, games)
        XCTAssertEqual(shard.gameCount, games.count)
        XCTAssertEqual(shard.header.corpusID, "20260620-000000-ABC123")
        XCTAssertEqual(shard.header.sourceID, "src-TESTSRC1")
    }

    func testSHATamperIsDetected() throws {
        let openURL = tempDir.appendingPathComponent("shard-00001.dcmgames.open")
        let writer = try ShardWriter(creatingAt: openURL, header: makeHeader())
        for g in (0..<5).map({ sampleGame($0) }) { try writer.append(g) }
        let sealedURL = try writer.seal(sealUnix: 1)

        var bytes = try Data(contentsOf: sealedURL)
        // Flip a body byte (just past the 256-byte front header).
        let flipIndex = bytes.startIndex + GameCorpusShardFormat.frontHeaderSize + 8
        bytes[flipIndex] ^= 0xFF
        try bytes.write(to: sealedURL)

        XCTAssertThrowsError(try GameCorpusShardIO.readSealed(at: sealedURL)) { error in
            guard let e = error as? GameCorpusError, case .shardSHAMismatch = e else {
                return XCTFail("expected shardSHAMismatch, got \(error)")
            }
        }
    }

    func testCleanOpenShardScanCountsAll() throws {
        let openURL = tempDir.appendingPathComponent("shard-00002.dcmgames.open")
        let writer = try ShardWriter(creatingAt: openURL, header: makeHeader())
        let games = (0..<8).map { sampleGame($0) }
        for g in games { try writer.append(g) }
        try writer.closeWithoutSealing()

        let scan = try GameCorpusShardIO.scanOpenShard(at: openURL)
        XCTAssertEqual(scan.gameCount, games.count)
        XCTAssertEqual(scan.validByteCount, scan.fileSize)
    }

    func testOpenShardScanRecoversTornTail() throws {
        let openURL = tempDir.appendingPathComponent("shard-00003.dcmgames.open")
        let writer = try ShardWriter(creatingAt: openURL, header: makeHeader())
        let games = (0..<10).map { sampleGame($0) }
        for g in games { try writer.append(g) }
        try writer.closeWithoutSealing()

        // Lop off the tail to simulate a torn final record.
        let full = try Data(contentsOf: openURL)
        try full.subdata(in: full.startIndex..<(full.endIndex - 5)).write(to: openURL)

        let scan = try GameCorpusShardIO.scanOpenShard(at: openURL)
        XCTAssertEqual(scan.gameCount, games.count - 1)
        XCTAssertGreaterThanOrEqual(scan.validByteCount, GameCorpusShardFormat.frontHeaderSize)
        XCTAssertLessThan(scan.validByteCount, scan.fileSize)
    }
}
