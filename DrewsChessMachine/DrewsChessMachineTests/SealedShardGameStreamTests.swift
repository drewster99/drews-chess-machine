import XCTest
import CryptoKit
@testable import DrewsChessMachine

/// Corpus replay reads each sealed shard through `SealedShardGameStream`
/// (game by game, through a bounded window) instead of `readSealed` (every
/// game decoded at once). The replay's `[BATCH-HASH]` lines, its resume
/// bookkeeping and its skip-or-refuse handling of a bad shard all depend on
/// the stream returning exactly the games `readSealed` returns, in the same
/// order, and on it refusing exactly the shards `readSealed` refuses before
/// returning any game. These tests pin that on shards the app's own
/// `ShardWriter` writes.
final class SealedShardGameStreamTests: XCTestCase {

    private let tempDir = FileManager.default.temporaryDirectory
        .appendingPathComponent("dcm-shard-stream-tests-\(UUID().uuidString)", isDirectory: true)

    override func setUpWithError() throws {
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        do { try FileManager.default.removeItem(at: tempDir) } catch { /* best-effort cleanup */ }
    }

    /// A game of `moveCount` moves with varied squares and promotions; every
    /// fifth game starts from a FEN and every seventh has no termination
    /// reason, so every optional field of the record format is crossed.
    private func game(_ seed: Int, moveCount: Int) -> GameRecord {
        var moves: [ChessMove] = []
        moves.reserveCapacity(moveCount)
        for i in 0..<moveCount {
            let from = (seed * 7 + i) % 64
            let to = (seed * 13 + i * 5 + 1) % 64
            let promotion: PieceType? = (i % 9 == 4) ? [.queen, .rook, .bishop, .knight][(seed + i) % 4] : nil
            moves.append(ChessMove(fromRow: from / 8, fromCol: from % 8, toRow: to / 8, toCol: to % 8,
                                   promotion: promotion))
        }
        let outcome: GameOutcome = [.whiteWin, .draw, .blackWin][seed % 3]
        let fen: String? = seed % 5 == 0 ? "8/8/8/4k3/8/8/4K3/8 w - - 0 \(seed + 1)" : nil
        let reason: GameTerminationReason? = seed % 7 == 0 ? nil : .checkmate
        return GameRecord(startFEN: fen, moves: moves, outcome: outcome, terminationReason: reason)
    }

    /// Games whose records straddle many read-chunk boundaries, plus one
    /// record larger than a whole chunk (the window must grow to hold it).
    private func gamesSpanningSeveralChunks() -> [GameRecord] {
        let chunk = GameCorpusShardIO.SealedShardBodyCursor.readChunkByteCount
        var games: [GameRecord] = []
        for seed in 0..<9_000 { games.append(game(seed, moveCount: 40 + seed % 170)) }
        // ~1.5 chunks of moves in one record (2 bytes per move).
        games.insert(game(9_001, moveCount: chunk * 3 / 4), at: 2_500)
        games.append(game(9_002, moveCount: 0))
        return games
    }

    private func writeSealedShard(_ games: [GameRecord], seq: Int = 0) throws -> URL {
        let openURL = tempDir.appendingPathComponent("shard-0000\(seq).dcmgames.open")
        let header = GameCorpusShardFormat.FrontHeader(corpusID: "20261122-000000-STREAM",
                                                       sourceID: "src-STREAM01",
                                                       shardSeq: UInt32(seq),
                                                       createdAtUnix: 1_700_000_000)
        let writer = try ShardWriter(creatingAt: openURL, header: header)
        for g in games { try writer.append(g) }
        return try writer.seal(sealUnix: 1_700_000_100)
    }

    private func drain(_ stream: GameCorpusShardIO.SealedShardGameStream) throws -> [GameRecord] {
        var games: [GameRecord] = []
        while let g = try stream.nextGame() { games.append(g) }
        return games
    }

    /// The stream returns exactly the all-at-once read's games, in order,
    /// across chunk boundaries and an over-chunk record, then ends (and keeps
    /// returning nil).
    func testStreamReturnsExactlyTheVerifiedReadsGames() throws {
        let games = gamesSpanningSeveralChunks()
        let url = try writeSealedShard(games)
        let fileSize = try XCTUnwrap(try FileManager.default.attributesOfItem(atPath: url.path)[.size] as? Int)
        XCTAssertGreaterThan(fileSize, 3 * GameCorpusShardIO.SealedShardBodyCursor.readChunkByteCount,
                             "the shard must span several read chunks to exercise the window")

        let verified = try GameCorpusShardIO.readSealed(at: url)
        XCTAssertEqual(verified.games, games)

        let stream = try GameCorpusShardIO.SealedShardGameStream(verifyingSealedShardAt: url)
        XCTAssertEqual(stream.gameCount, games.count)
        XCTAssertEqual(try drain(stream), verified.games)
        XCTAssertEqual(stream.gamesRead, games.count)
        XCTAssertNil(try stream.nextGame(), "an ended stream stays ended")
    }

    /// A resume part-way into a shard skips games through the stream; the
    /// next game is the one at that index of the shard.
    func testSkipThenReadContinuesAtTheSkippedIndex() throws {
        let games = gamesSpanningSeveralChunks()
        let url = try writeSealedShard(games)
        for skip in [0, 1, 2_500, 2_501, games.count - 1] {
            let stream = try GameCorpusShardIO.SealedShardGameStream(verifyingSealedShardAt: url)
            try stream.skipGames(skip)
            XCTAssertEqual(try drain(stream), Array(games[skip...]), "skip \(skip)")
        }
        let stream = try GameCorpusShardIO.SealedShardGameStream(verifyingSealedShardAt: url)
        XCTAssertThrowsError(try stream.skipGames(games.count + 1), "skipping past the end is an error")
    }

    /// A shard that fails verification is refused at open — before any game
    /// is returned — with the same error the all-at-once read throws: a hash
    /// mismatch, a record whose CRC fails with the hash re-sealed over it
    /// (only a writer defect produces that), a trailer game count that
    /// disagrees with the records, and a file cut short.
    func testStreamRefusesAtOpenWhatTheVerifiedReadRefuses() throws {
        let games = (0..<300).map { game($0, moveCount: 30 + $0 % 50) }
        let frontSize = GameCorpusShardFormat.frontHeaderSize
        let trailerSize = GameCorpusShardFormat.trailerSize

        func reseal(_ bytes: Data, gameCount: Int? = nil) throws -> Data {
            let body = bytes.prefix(bytes.count - trailerSize)
            var trailer = try GameCorpusShardFormat.decodeTrailer(bytes.suffix(trailerSize))
            trailer.sha256 = Data(SHA256.hash(data: body))
            if let gameCount { trailer.gameCount = gameCount }
            return body + (try GameCorpusShardFormat.encodeTrailer(trailer))
        }

        let pristine = try Data(contentsOf: try writeSealedShard(games, seq: 1))
        var shaTamper = pristine
        shaTamper[shaTamper.startIndex + frontSize + 20] ^= 0xFF
        // The last record's CRC field is the last 4 body bytes.
        var crcTamper = pristine
        crcTamper[crcTamper.startIndex + crcTamper.count - trailerSize - 1] ^= 0xFF
        let crcTamperResealed = try reseal(crcTamper)
        let countMismatch = try reseal(pristine, gameCount: games.count + 1)
        let truncated = pristine.prefix(frontSize + trailerSize - 1)

        let cases: [(String, Data, (GameCorpusError) -> Bool)] = [
            ("SHA tamper", shaTamper, { if case .shardSHAMismatch = $0 { return true }; return false }),
            ("CRC tamper, re-sealed", crcTamperResealed, { if case .recordCRCMismatch = $0 { return true }; return false }),
            ("game count", countMismatch, { if case .gameCountMismatch = $0 { return true }; return false }),
            ("truncated", truncated, { if case .truncatedHeader = $0 { return true }; return false }),
        ]
        for (name, bytes, matches) in cases {
            let url = tempDir.appendingPathComponent("case-\(name.filter(\.isLetter)).dcmgames")
            try bytes.write(to: url)
            XCTAssertThrowsError(try GameCorpusShardIO.readSealed(at: url), name) { error in
                XCTAssertTrue((error as? GameCorpusError).map(matches) ?? false, "\(name): readSealed threw \(error)")
            }
            XCTAssertThrowsError(try GameCorpusShardIO.SealedShardGameStream(verifyingSealedShardAt: url), name) { error in
                XCTAssertTrue((error as? GameCorpusError).map(matches) ?? false, "\(name): the stream threw \(error)")
            }
        }
    }

    /// A shard changed in place after it verified at open is not fed: the
    /// stream returns the games before the change and then throws
    /// `streamedReadFailedAfterVerification`, never the changed game.
    func testShardChangedAfterOpenThrowsInsteadOfFeedingTheChange() throws {
        let games = gamesSpanningSeveralChunks()
        let url = try writeSealedShard(games)
        let stream = try GameCorpusShardIO.SealedShardGameStream(verifyingSealedShardAt: url)
        XCTAssertEqual(try stream.nextGame(), games[0])

        // Flip a byte inside the last record's payload, far past the window
        // the first read filled.
        let handle = try FileHandle(forUpdating: url)
        let size = try handle.seekToEnd()
        let offset = size - UInt64(GameCorpusShardFormat.trailerSize) - 6
        try handle.seek(toOffset: offset)
        let original = try XCTUnwrap(try handle.read(upToCount: 1))
        try handle.seek(toOffset: offset)
        try handle.write(contentsOf: Data([original[original.startIndex] ^ 0xFF]))
        try handle.close()

        var returned = [games[0]]
        do {
            while let g = try stream.nextGame() { returned.append(g) }
            XCTFail("a changed shard must not stream to its end")
        } catch let error as GameCorpusError {
            guard case .streamedReadFailedAfterVerification = error else {
                return XCTFail("expected streamedReadFailedAfterVerification, got \(error)")
            }
        }
        XCTAssertEqual(returned, Array(games.prefix(returned.count)), "only verified games were returned")
        XCTAssertLessThan(returned.count, games.count)
        XCTAssertThrowsError(try stream.nextGame(), "a failed stream keeps failing")
    }
}
