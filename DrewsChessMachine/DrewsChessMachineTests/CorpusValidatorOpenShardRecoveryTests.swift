//
//  CorpusValidatorOpenShardRecoveryTests.swift
//  DrewsChessMachineTests
//
//  Corpus replay opens corpora read-only, and no app path reopens an existing
//  corpus for writing, so nothing recovered the `.open` shard a crashed
//  recording or import leaves behind: its games stayed unreadable for good.
//  `--validate-corpus <dir> --fix` is now the explicit place that recovers
//  them, through the same recovery `GameCorpus.open` uses. These pin both
//  halves: without `fix` an `.open` shard is only reported and every byte of
//  the corpus is left as it was; with `fix` a shard with complete games is
//  truncated to its last complete game and sealed, one with none is removed,
//  each outcome is reported as fixed, and the recovered games are then
//  validated and counted like any other sealed shard.
//

import XCTest
@testable import DrewsChessMachine

final class CorpusValidatorOpenShardRecoveryTests: XCTestCase {

    private var root: URL!

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory
            .appendingPathComponent("CorpusValidatorOpenShardRecoveryTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try FileManager.default.removeItem(at: root)
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

    /// Every entry's name and bytes, to compare the corpus directory before
    /// and after validation. A corpus directory holds only files.
    private func snapshot(_ directory: URL) throws -> [String: Data] {
        var result: [String: Data] = [:]
        for name in try FileManager.default.contentsOfDirectory(atPath: directory.path) {
            let url = directory.appendingPathComponent(name)
            var isDirectory: ObjCBool = false
            guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory), !isDirectory.boolValue else {
                XCTFail("\(name) is not a regular file")
                continue
            }
            result[name] = try Data(contentsOf: url)
        }
        return result
    }

    /// What a crashed recording leaves: one sealed shard (games 0..<5), then
    /// `shard-00001.dcmgames.open` holding games 5..<9 of the same source
    /// with the last record torn, and `shard-00002.dcmgames.open` holding
    /// only its header. `corpus.json` counts only the sealed shard's games.
    private func makeCrashedCorpus() throws -> (directory: URL, tornShard: URL, emptyShard: URL) {
        let corpus = try GameCorpus.create(name: "crashed", comment: nil, parentDirectory: root)
        let sourceID = try corpus.beginSource(kind: "selfPlay")
        for seed in 0..<5 { try corpus.append(sampleGame(seed)) }
        try corpus.finishSource()

        func header(seq: UInt32) -> GameCorpusShardFormat.FrontHeader {
            GameCorpusShardFormat.FrontHeader(corpusID: corpus.corpusID,
                                              sourceID: sourceID,
                                              shardSeq: seq,
                                              createdAtUnix: 1)
        }

        let tornShard = corpus.directory.appendingPathComponent("shard-00001.dcmgames.open")
        let tornWriter = try ShardWriter(creatingAt: tornShard, header: header(seq: 1))
        for seed in 5..<9 { try tornWriter.append(sampleGame(seed)) }
        try tornWriter.closeWithoutSealing()
        let full = try Data(contentsOf: tornShard)
        try full.subdata(in: full.startIndex..<(full.endIndex - 6)).write(to: tornShard)

        let emptyShard = corpus.directory.appendingPathComponent("shard-00002.dcmgames.open")
        let emptyWriter = try ShardWriter(creatingAt: emptyShard, header: header(seq: 2))
        try emptyWriter.closeWithoutSealing()

        return (corpus.directory, tornShard, emptyShard)
    }

    func testWithoutFixOpenShardsAreOnlyReportedAndNothingChanges() throws {
        let (directory, _, _) = try makeCrashedCorpus()
        let before = try snapshot(directory)

        let report = try CorpusValidator.validate(directory: directory, fix: false)

        XCTAssertEqual(try snapshot(directory), before, "a report-only run must leave every byte of the corpus as it was")
        let openFindings = report.findings.filter { $0.code == "open-shard-present" }
        XCTAssertEqual(openFindings.count, 1)
        XCTAssertEqual(openFindings.first?.fixable, true)
        XCTAssertEqual(openFindings.first?.fixed, false)
        XCTAssertFalse(report.isValid, "an unrecovered .open shard is an unresolved problem")
        XCTAssertEqual(report.totalGames, 5, "only the sealed shard's games are readable")
    }

    func testFixSealsTheTornShardAtItsLastCompleteGameAndRemovesTheEmptyOne() throws {
        let (directory, tornShard, emptyShard) = try makeCrashedCorpus()

        let report = try CorpusValidator.validate(directory: directory, fix: true)

        XCTAssertFalse(FileManager.default.fileExists(atPath: tornShard.path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: emptyShard.path))
        let sealed = directory.appendingPathComponent("shard-00001.dcmgames")
        XCTAssertEqual(try GameCorpusShardIO.readSealed(at: sealed).games, (5..<8).map { sampleGame($0) },
                       "the torn last game is dropped; every complete one is kept")
        XCTAssertFalse(FileManager.default.fileExists(atPath: directory.appendingPathComponent("shard-00002.dcmgames").path),
                       "a shard with no complete game is removed, not sealed")

        let recoveries = report.findings.filter { $0.code == "open-shard-present" }
        XCTAssertEqual(recoveries.count, 2, "one finding per recovered shard")
        XCTAssertTrue(recoveries.allSatisfy(\.fixed))
        XCTAssertTrue(report.isValid, "findings: \(report.findings.map(\.message))")
        XCTAssertEqual(report.totalGames, 8)
        XCTAssertEqual(report.shardCount, 2)

        let metadata = try GameCorpus.loadMetadata(directory: directory)
        XCTAssertEqual(metadata.sources.first?.gamesAdded, 8, "corpus.json counts include the recovered games")

        let revalidated = try CorpusValidator.validate(directory: directory)
        XCTAssertTrue(revalidated.isValid, "findings: \(revalidated.findings.map(\.message))")
        XCTAssertEqual(revalidated.totalGames, 8)
    }
}
