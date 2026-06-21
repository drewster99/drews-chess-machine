import XCTest
@testable import DrewsChessMachine

final class GameCorpusTests: XCTestCase {

    private let tempDir = FileManager.default.temporaryDirectory
        .appendingPathComponent("dcm-corpus-tests-\(UUID().uuidString)", isDirectory: true)

    override func setUpWithError() throws {
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        do { try FileManager.default.removeItem(at: tempDir) } catch { /* best-effort cleanup */ }
    }

    private func sampleGame(_ seed: Int) -> GameRecord {
        var moves: [ChessMove] = []
        let n = 4 + (seed % 9)
        for i in 0..<n {
            let f = (seed * 5 + i) % 64
            let t = (seed * 11 + i + 3) % 64
            moves.append(ChessMove(fromRow: f / 8, fromCol: f % 8, toRow: t / 8, toCol: t % 8, promotion: nil))
        }
        let outcome: GameOutcome = [.whiteWin, .draw, .blackWin][seed % 3]
        return GameRecord(moves: moves, outcome: outcome, terminationReason: .checkmate)
    }

    /// Record `games` into a fresh sealed corpus and return its directory. The
    /// corpus object is released at return so no file handles remain open.
    private func recordSealedCorpus(games: [GameRecord], shardSoftLimitBytes: Int) throws -> URL {
        let corpus = try GameCorpus.create(name: "test",
                                           comment: "round-trip",
                                           shardSoftLimitBytes: shardSoftLimitBytes,
                                           parentDirectory: tempDir)
        try corpus.beginSource(kind: "selfPlay")
        for g in games { try corpus.append(g) }
        try corpus.seal()
        return corpus.directory
    }

    func testCreateRecordSealReopenRoundTrip() throws {
        let games = (0..<40).map { sampleGame($0) }
        // Small soft limit forces rotation into multiple shards.
        let dir = try recordSealedCorpus(games: games, shardSoftLimitBytes: 512)

        let reopened = try GameCorpus.open(directory: dir)
        XCTAssertEqual(reopened.metadata.state, "sealed")
        XCTAssertEqual(reopened.metadata.sources.count, 1)
        XCTAssertEqual(reopened.metadata.sources.first?.complete, true)
        XCTAssertEqual(reopened.metadata.sources.first?.gamesAdded, games.count)

        let shardURLs = try reopened.sealedShardURLs()
        XCTAssertGreaterThan(shardURLs.count, 1, "expected rotation into multiple shards")

        XCTAssertEqual(try reopened.allGames(), games)
    }

    func testCorpusMetadataJSONRoundTrips() throws {
        let dir = try recordSealedCorpus(games: (0..<3).map { sampleGame($0) },
                                         shardSoftLimitBytes: 1 << 20)
        let metaURL = dir.appendingPathComponent(GameCorpus.metadataFilename)
        let meta = try JSONDecoder().decode(CorpusMetadata.self, from: try Data(contentsOf: metaURL))
        XCTAssertEqual(meta.state, "sealed")
        XCTAssertEqual(meta.name, "test")
        XCTAssertEqual(meta.comment, "round-trip")
        XCTAssertEqual(meta.sources.first?.appBuildNumber, BuildInfo.buildNumber)
        XCTAssertEqual(meta.sources.first?.appGitHash, BuildInfo.gitHash)
    }

    func testCrashRecoverySealsOpenShardOnReopen() throws {
        let corpus = try GameCorpus.create(name: "crash", comment: nil, parentDirectory: tempDir)
        let dir = corpus.directory
        let games = (0..<6).map { sampleGame($0) }

        // Hand-create an unsealed shard as if a crash hit mid-recording.
        let openURL = dir.appendingPathComponent("shard-00000.dcmgames.open")
        let header = GameCorpusShardFormat.FrontHeader(corpusID: corpus.corpusID,
                                                       sourceID: "src-CRASH01",
                                                       shardSeq: 0,
                                                       createdAtUnix: 1)
        let writer = try ShardWriter(creatingAt: openURL, header: header)
        for g in games { try writer.append(g) }
        try writer.closeWithoutSealing()
        XCTAssertTrue(FileManager.default.fileExists(atPath: openURL.path))

        let reopened = try GameCorpus.open(directory: dir)
        XCTAssertFalse(FileManager.default.fileExists(atPath: openURL.path),
                       "open shard should be recovered + sealed away")
        XCTAssertEqual(try reopened.sealedShardURLs().count, 1)
        XCTAssertEqual(try reopened.allGames(), games)
    }

    func testCrashRecoveryDropsTornTailGame() throws {
        let corpus = try GameCorpus.create(name: "crash2", comment: nil, parentDirectory: tempDir)
        let dir = corpus.directory
        let games = (0..<6).map { sampleGame($0) }

        let openURL = dir.appendingPathComponent("shard-00000.dcmgames.open")
        let header = GameCorpusShardFormat.FrontHeader(corpusID: corpus.corpusID,
                                                       sourceID: "src-CRASH02",
                                                       shardSeq: 0,
                                                       createdAtUnix: 1)
        let writer = try ShardWriter(creatingAt: openURL, header: header)
        for g in games { try writer.append(g) }
        try writer.closeWithoutSealing()

        // Lop off the tail to simulate a torn final record.
        let full = try Data(contentsOf: openURL)
        try full.subdata(in: full.startIndex..<(full.endIndex - 6)).write(to: openURL)

        let reopened = try GameCorpus.open(directory: dir)
        XCTAssertEqual(try reopened.allGames().count, games.count - 1)
    }
}
