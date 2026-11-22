import XCTest
import CryptoKit
@testable import DrewsChessMachine

/// Corpus replay reads each sealed shard through `SealedShardGameStream`
/// (raw bytes read and verified once, games decoded one at a time) instead of
/// `readSealed` (every game decoded at once). The replay's `[BATCH-HASH]`
/// lines, its resume bookkeeping and its skip-or-refuse handling of a bad
/// shard all depend on the stream returning exactly the games `readSealed`
/// returns, in the same order; on it refusing exactly the shards `readSealed`
/// refuses, before returning any game; and on it feeding only bytes that
/// passed the SHA-256. These tests pin that on shards the app's own
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

    /// Many games of varied lengths, one long one and one empty one.
    private func variedGames() -> [GameRecord] {
        var games = (0..<3_000).map { game($0, moveCount: 40 + $0 % 170) }
        games.insert(game(3_001, moveCount: 20_000), at: 1_500)
        games.append(game(3_002, moveCount: 0))
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

    private func drain(_ stream: GameCorpusShardIO.SealedShardGameStream) -> [GameRecord] {
        var games: [GameRecord] = []
        while let g = stream.nextGame() { games.append(g) }
        return games
    }

    /// `bytes` with its trailer rewritten over its own body: the SHA-256
    /// recomputed and, when given, another game count — what a writer that
    /// sealed bad records would have produced.
    private func resealed(_ bytes: Data, gameCount: Int? = nil) throws -> Data {
        let trailerSize = GameCorpusShardFormat.trailerSize
        let body = bytes.prefix(bytes.count - trailerSize)
        var trailer = try GameCorpusShardFormat.decodeTrailer(bytes.suffix(trailerSize))
        trailer.sha256 = Data(SHA256.hash(data: body))
        if let gameCount { trailer.gameCount = gameCount }
        return body + (try GameCorpusShardFormat.encodeTrailer(trailer))
    }

    /// The stream returns exactly the all-at-once read's games, in order,
    /// then ends (and keeps returning nil).
    func testStreamReturnsExactlyTheVerifiedReadsGames() throws {
        let games = variedGames()
        let url = try writeSealedShard(games)
        let verified = try GameCorpusShardIO.readSealed(at: url)
        XCTAssertEqual(verified.games, games)

        let stream = try GameCorpusShardIO.SealedShardGameStream(verifyingSealedShardAt: url)
        XCTAssertEqual(stream.gameCount, games.count)
        XCTAssertEqual(drain(stream), verified.games)
        XCTAssertEqual(stream.gamesRead, games.count)
        XCTAssertNil(stream.nextGame(), "an ended stream stays ended")
    }

    /// A resume part-way into a shard skips games through the stream; the
    /// next game is the one at that index of the shard.
    func testSkipThenReadContinuesAtTheSkippedIndex() throws {
        let games = variedGames()
        let url = try writeSealedShard(games)
        for skip in [0, 1, 1_500, 1_501, games.count - 1, games.count] {
            let stream = try GameCorpusShardIO.SealedShardGameStream(verifyingSealedShardAt: url)
            stream.skipGames(skip)
            XCTAssertEqual(drain(stream), Array(games[skip...]), "skip \(skip)")
        }
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

        let pristine = try Data(contentsOf: try writeSealedShard(games, seq: 1))
        var shaTamper = pristine
        shaTamper[shaTamper.startIndex + frontSize + 20] ^= 0xFF
        // The last record's CRC field is the last 4 body bytes.
        var crcTamper = pristine
        crcTamper[crcTamper.startIndex + crcTamper.count - trailerSize - 1] ^= 0xFF
        let crcTamperResealed = try resealed(crcTamper)
        let countMismatch = try resealed(pristine, gameCount: games.count + 1)
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

    /// A record whose move count is far larger than its payload — with a
    /// matching CRC and the hash re-sealed over it, so it reaches the decoder
    /// — is refused as truncated instead of reserving space for ~4 billion
    /// moves (which would abort the process).
    func testHugeMoveCountRecordIsRefusedNotAllocated() throws {
        let games = (0..<10).map { game($0, moveCount: 20) }
        let pristine = try Data(contentsOf: try writeSealedShard(games, seq: 2))

        // flags, outcome, reason, moveCount = UInt32.max, no moves.
        let payload = Data([0, GameOutcome.draw.rawValue, 0, 0xFF, 0xFF, 0xFF, 0xFF])
        var frame = Data()
        withUnsafeBytes(of: UInt32(payload.count).littleEndian) { frame.append(contentsOf: $0) }
        frame.append(payload)
        withUnsafeBytes(of: CRC32.checksum(payload).littleEndian) { frame.append(contentsOf: $0) }
        let trailerSize = GameCorpusShardFormat.trailerSize
        let withHugeRecord = pristine.prefix(pristine.count - trailerSize) + frame + pristine.suffix(trailerSize)
        let url = tempDir.appendingPathComponent("huge-move-count.dcmgames")
        try (try resealed(withHugeRecord, gameCount: games.count + 1)).write(to: url)

        for (name, read) in [("readSealed", { _ = try GameCorpusShardIO.readSealed(at: url) }),
                             ("stream", { _ = try GameCorpusShardIO.SealedShardGameStream(verifyingSealedShardAt: url) })] {
            XCTAssertThrowsError(try read(), name) { error in
                guard case GameCorpusError.truncatedRecord? = error as? GameCorpusError else {
                    return XCTFail("\(name): expected truncatedRecord, got \(error)")
                }
            }
        }
    }

    /// A shard rewritten in place after it verified at open — a later record
    /// replaced by a valid one of the same length with a matching CRC, which
    /// a second read of the file could not tell from the original without
    /// the hash — does not reach the feed: the stream returns the games of
    /// the bytes it verified.
    func testShardRewrittenAfterOpenStillFeedsTheVerifiedGames() throws {
        var games = (0..<2_000).map { game($0, moveCount: 60) }
        // The last game's outcome is the one byte the rewrite changes.
        games[games.count - 1] = GameRecord(moves: games[games.count - 1].moves, outcome: .whiteWin,
                                            terminationReason: .checkmate)
        let url = try writeSealedShard(games, seq: 3)
        let stream = try GameCorpusShardIO.SealedShardGameStream(verifyingSealedShardAt: url)
        XCTAssertEqual(stream.nextGame(), games[0])

        let replacementGame = GameRecord(moves: games[games.count - 1].moves, outcome: .blackWin,
                                         terminationReason: .checkmate)
        let replacement = GameCorpusShardFormat.encodeFramedRecord(replacementGame)
        XCTAssertEqual(replacement.count, GameCorpusShardFormat.encodeFramedRecord(games[games.count - 1]).count)
        let handle = try FileHandle(forUpdating: url)
        let size = try handle.seekToEnd()
        try handle.seek(toOffset: size - UInt64(GameCorpusShardFormat.trailerSize) - UInt64(replacement.count))
        try handle.write(contentsOf: replacement)
        try handle.close()
        XCTAssertThrowsError(try GameCorpusShardIO.readSealed(at: url), "the rewritten file itself fails its hash")

        XCTAssertEqual([games[0]] + drain(stream), games, "the stream feeds the bytes it verified, not the rewrite")
    }
}
