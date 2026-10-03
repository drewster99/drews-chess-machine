//
//  CorpusValidatorLiveWriterTests.swift
//  DrewsChessMachineTests
//
//  `--validate-corpus <dir> --fix` recovers `.open` shards: a crashed
//  recording or import leaves its shard `.open` with a possibly torn last
//  record, and the fix truncates it to its last complete game and seals it,
//  or deletes it when it holds none. A live writer's shard looks exactly the
//  same on disk, and recovering one destroys data: a mid-record shard is cut
//  and sealed while the writer keeps appending after the trailer (the sealed
//  SHA no longer matches), and a header-only shard — the live shard right
//  after a rotation — is deleted, so every later game goes into an unlinked
//  file and the writer's own seal then fails.
//
//  A shard writer now holds an exclusive lock on its `.open` file for the
//  file's whole life (`O_EXLOCK`, a `flock`-style lock owned by that open
//  file and released by the kernel when the file is closed or the process
//  exits, however it exits). Recovery takes the same lock without waiting
//  and skips any shard whose lock is held. These tests prove each half:
//
//  - a live writer's shard is held (from this process, from a separate read
//    of the same file, and from another process), and `--fix` leaves it
//    byte-for-byte untouched, does not rewrite `corpus.json` under it, and
//    the writer carries on and seals normally afterwards;
//  - a shard whose writer has gone — closed without sealing, or a helper
//    process that held the lock and exited — is recovered as before.
//

import XCTest
import Darwin
@testable import DrewsChessMachine

final class CorpusValidatorLiveWriterTests: XCTestCase {

    private var root: URL!

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory
            .appendingPathComponent("CorpusValidatorLiveWriterTests-\(UUID().uuidString)", isDirectory: true)
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

    private func shardURL(_ directory: URL, _ seq: Int, open: Bool) -> URL {
        directory.appendingPathComponent("shard-0000\(seq).dcmgames" + (open ? ".open" : ""))
    }

    /// Try to take the exclusive lock on `url` from a new open file in this
    /// process, without waiting. True when the lock was free (it is released
    /// again before returning).
    private func lockIsFree(_ url: URL) throws -> Bool {
        let descriptor = Darwin.open(url.path, O_RDONLY | O_NONBLOCK | O_EXLOCK | O_CLOEXEC)
        if descriptor >= 0 {
            XCTAssertEqual(Darwin.close(descriptor), 0)
            return true
        }
        let code = errno
        guard code == EWOULDBLOCK else {
            throw FileSafetyError.systemCallFailed(path: url.path, call: "open", errnoValue: code)
        }
        return false
    }

    // MARK: - A live writer's shard is never recovered

    func testFixLeavesALiveWritersMidRecordShardUntouched() throws {
        let corpus = try GameCorpus.create(name: "live", comment: nil, parentDirectory: root)
        try withExtendedLifetime(corpus) {
            try corpus.beginSource(kind: "selfPlay")
            for seed in 0..<3 { try corpus.append(sampleGame(seed)) }
            let live = shardURL(corpus.directory, 0, open: true)
            let before = try Data(contentsOf: live)

            let report = try CorpusValidator.validate(directory: corpus.directory, fix: true)

            XCTAssertEqual(try Data(contentsOf: live), before, "a live writer's shard must be left byte-for-byte")
            XCTAssertFalse(FileManager.default.fileExists(atPath: shardURL(corpus.directory, 0, open: false).path),
                           "a live writer's shard must not be sealed")
            let inUse = report.findings.filter { $0.code == "open-shard-in-use" }
            XCTAssertEqual(inUse.count, 1, "\(report.findings.map(\.code))")
            XCTAssertEqual(inUse.first?.fixed, false)
            XCTAssertFalse(report.isValid)

            try corpus.append(sampleGame(3))
            try corpus.finishSource()
            let sealed = try GameCorpusShardIO.readSealed(at: shardURL(corpus.directory, 0, open: false))
            XCTAssertEqual(sealed.games, (0..<4).map { sampleGame($0) }, "the writer seals all of its games, SHA intact")
        }
    }

    func testFixDoesNotDeleteALiveWritersHeaderOnlyShardAfterRotation() throws {
        let corpus = try GameCorpus.create(name: "rotating", comment: nil, shardSoftLimitBytes: 1, parentDirectory: root)
        try withExtendedLifetime(corpus) {
            try corpus.beginSource(kind: "selfPlay")
            try corpus.append(sampleGame(0))
            try corpus.append(sampleGame(1))
            let live = shardURL(corpus.directory, 2, open: true)
            let before = try Data(contentsOf: live)
            XCTAssertEqual(before.count, GameCorpusShardFormat.frontHeaderSize, "the live shard holds only its header")

            try CorpusValidator.validate(directory: corpus.directory, fix: true)

            XCTAssertEqual(try Data(contentsOf: live), before, "the live header-only shard must not be deleted")
            try corpus.append(sampleGame(2))
            try corpus.finishSource()
            XCTAssertEqual(try GameCorpusShardIO.readSealed(at: shardURL(corpus.directory, 2, open: false)).games,
                           [sampleGame(2)])
            XCTAssertEqual(try corpus.allGames(), (0..<3).map { sampleGame($0) }, "no game is lost")
        }
    }

    func testFixDoesNotRewriteCorpusJsonWhileAWriterIsLive() throws {
        let corpus = try GameCorpus.create(name: "rotating", comment: nil, shardSoftLimitBytes: 1, parentDirectory: root)
        try withExtendedLifetime(corpus) {
            try corpus.beginSource(kind: "selfPlay")
            try corpus.append(sampleGame(0))
            try corpus.append(sampleGame(1))
            let metadataURL = corpus.directory.appendingPathComponent(GameCorpus.metadataFilename)
            let before = try Data(contentsOf: metadataURL)

            let report = try CorpusValidator.validate(directory: corpus.directory, fix: true)

            XCTAssertEqual(try Data(contentsOf: metadataURL), before,
                           "the live writer's in-memory counts are the authority; corpus.json must not be rewritten under it")
            XCTAssertTrue(report.findings.contains { $0.code == "counts-not-repaired-writer-active" },
                          "\(report.findings.map(\.code))")
        }
    }

    func testOpeningACorpusForWritingRefusesWhileAWriterIsLive() throws {
        let corpus = try GameCorpus.create(name: "live", comment: nil, parentDirectory: root)
        try withExtendedLifetime(corpus) {
            try corpus.beginSource(kind: "selfPlay")
            try corpus.append(sampleGame(0))
            let live = shardURL(corpus.directory, 0, open: true)
            let before = try Data(contentsOf: live)
            XCTAssertThrowsError(try GameCorpus.open(directory: corpus.directory))
            XCTAssertEqual(try Data(contentsOf: live), before)
        }
    }

    // MARK: - The lock itself

    func testALiveWritersShardIsLockedEvenAfterASeparateReadOfIt() throws {
        let corpus = try GameCorpus.create(name: "live", comment: nil, parentDirectory: root)
        try withExtendedLifetime(corpus) {
            try corpus.beginSource(kind: "selfPlay")
            try corpus.append(sampleGame(0))
            let live = shardURL(corpus.directory, 0, open: true)
            XCTAssertFalse(try lockIsFree(live), "the writer holds its open shard's lock")
            // A whole-file read opens and closes its own descriptor; with a
            // `flock`-style lock that must not release the writer's lock
            // (an `fcntl` lock would be dropped here).
            _ = try Data(contentsOf: live)
            XCTAssertFalse(try lockIsFree(live), "reading the shard elsewhere must not drop the writer's lock")
            try corpus.finishSource()
        }
    }

    func testClosingWithoutSealingReleasesTheLockAndFixRecoversTheShard() throws {
        let corpus = try GameCorpus.create(name: "crashed", comment: nil, parentDirectory: root)
        let sourceID = try corpus.beginSource(kind: "selfPlay")
        try corpus.finishSource()
        let leftover = shardURL(corpus.directory, 1, open: true)
        let writer = try ShardWriter(creatingAt: leftover,
                                     header: .init(corpusID: corpus.corpusID, sourceID: sourceID, shardSeq: 1, createdAtUnix: 1))
        try writer.append(sampleGame(0))
        try writer.closeWithoutSealing()
        XCTAssertTrue(try lockIsFree(leftover), "a closed writer holds no lock")

        try CorpusValidator.validate(directory: corpus.directory, fix: true)

        XCTAssertFalse(FileManager.default.fileExists(atPath: leftover.path))
        XCTAssertEqual(try GameCorpusShardIO.readSealed(at: shardURL(corpus.directory, 1, open: false)).games, [sampleGame(0)])
    }

    /// Another process holding the lock — the case that matters in practice
    /// (a GUI recording while `--validate-corpus --fix` runs from a shell).
    /// The helper takes a `flock` exclusive lock, the same lock `O_EXLOCK`
    /// takes, and holds it until its stdin closes. While it holds it, `--fix`
    /// must skip the shard; once it has exited, the kernel has released the
    /// lock and `--fix` recovers the shard.
    func testFixSkipsAShardLockedByAnotherProcessAndRecoversItAfterThatProcessExits() throws {
        let corpus = try GameCorpus.create(name: "crashed", comment: nil, parentDirectory: root)
        let sourceID = try corpus.beginSource(kind: "selfPlay")
        try corpus.finishSource()
        let shard = shardURL(corpus.directory, 1, open: true)
        let writer = try ShardWriter(creatingAt: shard,
                                     header: .init(corpusID: corpus.corpusID, sourceID: sourceID, shardSeq: 1, createdAtUnix: 1))
        try writer.append(sampleGame(0))
        try writer.closeWithoutSealing()

        let helper = Process()
        helper.executableURL = URL(fileURLWithPath: "/usr/bin/perl")
        helper.arguments = [
            "-e",
            #"use Fcntl qw(:flock); open(my $f, "<", $ARGV[0]) or die "open: $!"; "#
                + #"flock($f, LOCK_EX) or die "flock: $!"; $| = 1; print "locked\n"; <STDIN>; exit 0;"#,
            shard.path,
        ]
        let toHelper = Pipe()
        let fromHelper = Pipe()
        helper.standardInput = toHelper
        helper.standardOutput = fromHelper
        try helper.run()
        let ready = fromHelper.fileHandleForReading.availableData
        XCTAssertEqual(String(decoding: ready, as: UTF8.self), "locked\n", "the helper must hold the lock before --fix runs")
        XCTAssertFalse(try lockIsFree(shard), "the helper process holds the shard's lock")

        let before = try Data(contentsOf: shard)
        let whileHeld = try CorpusValidator.validate(directory: corpus.directory, fix: true)
        XCTAssertEqual(try Data(contentsOf: shard), before, "--fix must not touch a shard another process holds")
        XCTAssertTrue(whileHeld.findings.contains { $0.code == "open-shard-in-use" }, "\(whileHeld.findings.map(\.code))")

        try toHelper.fileHandleForWriting.close()
        helper.waitUntilExit()
        XCTAssertEqual(helper.terminationStatus, 0)
        XCTAssertTrue(try lockIsFree(shard), "the kernel releases the lock when its holder exits")

        try CorpusValidator.validate(directory: corpus.directory, fix: true)
        XCTAssertFalse(FileManager.default.fileExists(atPath: shard.path))
        XCTAssertEqual(try GameCorpusShardIO.readSealed(at: shardURL(corpus.directory, 1, open: false)).games, [sampleGame(0)])
    }
}
