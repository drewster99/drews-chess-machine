//
//  LichessBotChallengeLogWriterTests.swift
//  DrewsChessMachineTests
//
//  `LichessBotChallengeLog`'s writing (challenge-log plan §3.3): one file per
//  UTC day whatever the Mac's zone, `F_FULLFSYNC` after every append (and the
//  folder for a new day file), a torn tail cut and recorded as an
//  `unterminatedLineCut` line, failures reported, nothing written once the
//  file queue is closed — and what it writes reads back into the ledger.
//  Every test works in its own temporary folder.
//

import XCTest
@testable import DrewsChessMachine

final class LichessBotChallengeLogWriterTests: XCTestCase {

    private typealias F = LichessBotChallengeLogFixtures

    private var tempRoot: URL!
    private var savedDefaultTimeZone: TimeZone!

    override func setUpWithError() throws {
        tempRoot = FileManager.default.temporaryDirectory
            .appendingPathComponent("LichessBotChallengeLogWriterTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: tempRoot, withIntermediateDirectories: true)
        savedDefaultTimeZone = NSTimeZone.default
    }

    override func tearDownWithError() throws {
        NSTimeZone.default = savedDefaultTimeZone
        try FileManager.default.removeItem(at: tempRoot)
    }

    private var directory: LichessBotDataDirectory { LichessBotDataDirectory(root: tempRoot) }

    /// One call the append made after taking the lock.
    private enum RecordedCall: Equatable {
        case write
        case fsync
        case fullSync(name: String)
        case fullSyncDirectory(name: String)
    }

    private static func recordingSystemCalls(into calls: SyncBox<[RecordedCall]>) -> LichessBotJSONLines.AppendSystemCalls {
        let system = LichessBotJSONLines.AppendSystemCalls.system
        return LichessBotJSONLines.AppendSystemCalls(
            write: { data, handle in
                calls.modify { $0.append(.write) }
                try system.write(data, handle)
            },
            fsync: { handle in
                calls.modify { $0.append(.fsync) }
                try system.fsync(handle)
            },
            fullSync: { handle, path in
                calls.modify { $0.append(.fullSync(name: (path as NSString).lastPathComponent)) }
                try system.fullSync(handle, path)
            },
            fullSyncDirectory: { folder in
                calls.modify { $0.append(.fullSyncDirectory(name: folder.lastPathComponent)) }
                try system.fullSyncDirectory(folder)
            }
        )
    }

    private func makeLog(fileQueue: LichessBotFileQueue = LichessBotFileQueue(),
                         calls: SyncBox<[RecordedCall]> = SyncBox([]),
                         failures: SyncBox<[String]>) -> LichessBotChallengeLog {
        LichessBotChallengeLog(directory: directory, fileQueue: fileQueue, systemCalls: Self.recordingSystemCalls(into: calls)) { error in
            failures.modify { $0.append(error.localizedDescription) }
        }
    }

    func testEntriesGoToTheirUTCDayFileInAnyTimeZone() async throws {
        NSTimeZone.default = try XCTUnwrap(TimeZone(identifier: "America/Los_Angeles"))
        let failures = SyncBox<[String]>([])
        let log = makeLog(failures: failures)
        // 23:30 and 00:30 UTC: one local day in Los Angeles, two UTC days.
        log.record(F.created("first"), at: F.at(11.5 * 3600))
        log.record(.gameStarted(challengeID: "first"), at: F.at(12.5 * 3600))
        try await log.flush()
        XCTAssertEqual(failures.value, [])
        let contents = try await log.readAll()
        XCTAssertEqual(contents.filesRead.map(\.name), ["challenges-20261005.jsonl", "challenges-20261006.jsonl"])
        XCTAssertEqual(contents.entries.map(\.event), [F.created("first"), .gameStarted(challengeID: "first")])
        XCTAssertEqual(contents.entries.map(\.build), [BuildInfo.buildNumber, BuildInfo.buildNumber])
        let ledger = LichessBotChallengeLedger(contents: contents)
        XCTAssertEqual(ledger.loadStatus, .complete)
        XCTAssertEqual(ledger.row(challengeID: "first")?.state, .accepted(gameStarted: true), "a row spans two day files")
    }

    func testEveryAppendIsFullySyncedAndANewDayFilesFolderToo() async throws {
        let failures = SyncBox<[String]>([])
        let calls = SyncBox<[RecordedCall]>([])
        let log = makeLog(calls: calls, failures: failures)
        log.record(F.created(), at: F.at(0))
        log.record(.canceledOnLichess(challengeID: "AbCd1234"), at: F.at(1))
        try await log.flush()
        XCTAssertEqual(failures.value, [])
        XCTAssertEqual(calls.value, [
            // The append created `Challenges/` too: its entry in the bot folder.
            .fullSyncDirectory(name: tempRoot.lastPathComponent),
            .write, .fullSync(name: "challenges-20261005.jsonl"), .fullSyncDirectory(name: "Challenges"),
            .write, .fullSync(name: "challenges-20261005.jsonl"),
        ])
        XCTAssertEqual(log.statistics.appendCount, 2)
    }

    func testATornTailIsCutAndRecordedAheadOfTheEntry() async throws {
        let failures = SyncBox<[String]>([])
        let log = makeLog(failures: failures)
        log.record(F.created(), at: F.at(0))
        try await log.flush()
        let url = directory.challengeLogURL(for: F.at(0))
        let fragment = Data(#"{"at":"2026-10-05T12:00:01.000Z","build":24"#.utf8)
        let otherWriter = try FileSafety.openForAppending(at: url)
        try otherWriter.handle.write(contentsOf: fragment)
        try otherWriter.handle.close()

        log.record(.gameStarted(challengeID: "AbCd1234"), at: F.at(2))
        try await log.flush()

        XCTAssertEqual(failures.value, [])
        let contents = try await log.readAll()
        XCTAssertEqual(contents.filesRead.first?.droppedTrailingByteCount, 0)
        let events = contents.entries.map(\.event)
        XCTAssertEqual(events.count, 3)
        XCTAssertEqual(events.first, F.created())
        XCTAssertEqual(events[1], .unterminatedLineCut(byteCount: fragment.count, base64: fragment.base64EncodedString()))
        XCTAssertEqual(events.last, .gameStarted(challengeID: "AbCd1234"))
        let ledger = LichessBotChallengeLedger(contents: contents)
        XCTAssertEqual(ledger.unterminatedLineCuts.map(\.event), [events[1]], "a repair belongs to no challenge")
        XCTAssertEqual(ledger.rowsByKey.count, 1)
    }

    func testAWriteFailureIsReported() async throws {
        // `Challenges` is a file, so no day file can be created under it.
        try Data("not a folder".utf8).write(to: tempRoot.appendingPathComponent("Challenges", isDirectory: false))
        let failures = SyncBox<[String]>([])
        let log = makeLog(failures: failures)
        log.record(F.created(), at: F.at(0))
        try await log.flush()
        XCTAssertEqual(failures.value.count, 1)
        XCTAssertEqual(log.statistics.appendCount, 0)
    }

    func testNothingIsWrittenAfterTheFileQueueCloses() async throws {
        let fileQueue = LichessBotFileQueue()
        let failures = SyncBox<[String]>([])
        let log = makeLog(fileQueue: fileQueue, failures: failures)
        log.record(F.created(), at: F.at(0))
        await fileQueue.close(reason: "test shutdown")
        log.record(.gameStarted(challengeID: "AbCd1234"), at: F.at(1))
        // The refusal is logged by the queue; give it its turn.
        await fileQueue.close(reason: "test shutdown, again")
        XCTAssertEqual(failures.value, [])
        let contents = try LichessBotChallengeLog.readAll(in: directory)
        XCTAssertEqual(contents.entries.map(\.event), [F.created()], "the entry recorded after closing is not written")
        XCTAssertEqual(log.statistics.appendCount, 1)
    }

    func testTheFileItWritesIsWhatTheEarlierLinesLookLike() async throws {
        let failures = SyncBox<[String]>([])
        let log = makeLog(failures: failures)
        log.record(.gameStarted(challengeID: "AbCd1234"), at: F.at(0))
        try await log.flush()
        let text = try String(contentsOf: directory.challengeLogURL(for: F.at(0)), encoding: .utf8)
        XCTAssertEqual(text, #"{"at":"2026-10-05T12:00:00.000Z","build":\#(BuildInfo.buildNumber),"event":{"gameStarted":{"challengeID":"AbCd1234"}},"gitHash":"\#(BuildInfo.gitHash)","schemaVersion":1}"# + "\n")
    }
}
