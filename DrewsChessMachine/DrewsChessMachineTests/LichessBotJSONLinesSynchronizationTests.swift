//
//  LichessBotJSONLinesSynchronizationTests.swift
//  DrewsChessMachineTests
//
//  `LichessBotJSONLines.append`'s `Synchronization` reaches the right call —
//  nothing, `fsync`, or `F_FULLFSYNC` plus the folder when the append made
//  the file — and the journal and protocol log, which append through it,
//  refuse a symbolic link at their paths with a reported failure and leave
//  its target alone (challenge-log plan §3.3, §5).
//

import XCTest
@testable import DrewsChessMachine

final class LichessBotJSONLinesSynchronizationTests: XCTestCase {

    private var tempRoot: URL!

    override func setUpWithError() throws {
        tempRoot = FileManager.default.temporaryDirectory
            .appendingPathComponent("LichessBotJSONLinesSynchronizationTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: tempRoot, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try FileManager.default.removeItem(at: tempRoot)
    }

    private var directory: LichessBotDataDirectory { LichessBotDataDirectory(root: tempRoot) }

    /// One call `append` made after taking the lock.
    private enum RecordedCall: Equatable {
        case write(byteCount: Int)
        case fsync
        case fullSync(path: String)
        case fullSyncDirectory(path: String)
    }

    /// The real calls, each recorded before it runs.
    private static func recordingSystemCalls(into calls: SyncBox<[RecordedCall]>) -> LichessBotJSONLines.AppendSystemCalls {
        let system = LichessBotJSONLines.AppendSystemCalls.system
        return LichessBotJSONLines.AppendSystemCalls(
            write: { data, handle in
                calls.modify { $0.append(.write(byteCount: data.count)) }
                try system.write(data, handle)
            },
            fsync: { handle in
                calls.modify { $0.append(.fsync) }
                try system.fsync(handle)
            },
            fullSync: { handle, path in
                calls.modify { $0.append(.fullSync(path: path)) }
                try system.fullSync(handle, path)
            },
            fullSyncDirectory: { folder in
                calls.modify { $0.append(.fullSyncDirectory(path: folder.path)) }
                try system.fullSyncDirectory(folder)
            }
        )
    }

    private static let line = Data("{\"m\":1}\n".utf8)

    private func appendRecording(to url: URL, _ synchronization: LichessBotJSONLines.Synchronization) throws -> [RecordedCall] {
        let calls = SyncBox<[RecordedCall]>([])
        try LichessBotJSONLines.append(to: url, synchronization: synchronization, systemCalls: Self.recordingSystemCalls(into: calls)) { _ in
            Self.line
        }
        return calls.value
    }

    func testNoneOnlyWrites() throws {
        let url = tempRoot.appendingPathComponent("none.jsonl")
        XCTAssertEqual(try appendRecording(to: url, .none), [.write(byteCount: Self.line.count)])
        XCTAssertEqual(try appendRecording(to: url, .none), [.write(byteCount: Self.line.count)])
    }

    func testFsyncSynchronizesTheFileAndNeverTheFolder() throws {
        let url = tempRoot.appendingPathComponent("fsync.jsonl")
        XCTAssertEqual(try appendRecording(to: url, .fsync), [.write(byteCount: Self.line.count), .fsync])
        XCTAssertEqual(try appendRecording(to: url, .fsync), [.write(byteCount: Self.line.count), .fsync])
    }

    func testFullSyncFlushesTheFolderOnlyWhenTheAppendCreatedTheFile() throws {
        let url = tempRoot.appendingPathComponent("Challenges/full.jsonl")
        XCTAssertEqual(try appendRecording(to: url, .fullSync), [
            .write(byteCount: Self.line.count),
            .fullSync(path: url.path),
            .fullSyncDirectory(path: url.deletingLastPathComponent().path),
        ], "a new file's name is in its folder, which is flushed too")
        XCTAssertEqual(try appendRecording(to: url, .fullSync), [
            .write(byteCount: Self.line.count),
            .fullSync(path: url.path),
        ], "an existing file's name is already durable")
        XCTAssertEqual(try Data(contentsOf: url), Self.line + Self.line)
    }

    // MARK: - The writers refuse a symbolic link

    func testAJournalPathThatIsASymbolicLinkIsRefusedAndItsTargetKept() async throws {
        try directory.createDirectories()
        let target = tempRoot.appendingPathComponent("elsewhere.jsonl")
        let targetBytes = Self.line + Data("{\"unterminated".utf8)
        try targetBytes.write(to: target)
        let journalURL = directory.inProgressJournalURL(gameID: "g1")
        try FileManager.default.createSymbolicLink(at: journalURL, withDestinationURL: target)
        let failures = SyncBox<[String]>([])
        let writer = LichessBotJournalWriter(
            directory: directory,
            fileQueue: LichessBotFileQueue(),
            onWriteFailure: { gameID, error in failures.modify { $0.append("\(gameID): \(error.localizedDescription)") } },
            onGameFinished: { _ in }
        )
        await writer.gameEvent(gameID: "g1", .action("must not reach the target"))
        let expected = FileSafetyError.notARegularFile(path: journalURL.path, kind: .symbolicLink)
        XCTAssertEqual(failures.value, ["g1: \(expected.localizedDescription)"])
        XCTAssertEqual(try Data(contentsOf: target), targetBytes, "the target is neither cut nor written")
        XCTAssertEqual(try FileSafety.existingItem(at: journalURL)?.kind, .symbolicLink)
    }

    func testAProtocolLogPathThatIsASymbolicLinkIsRefusedAndItsTargetKept() async throws {
        try directory.createDirectories()
        let target = tempRoot.appendingPathComponent("elsewhere.jsonl")
        let targetBytes = Self.line + Data("{\"unterminated".utf8)
        try targetBytes.write(to: target)
        let failures = SyncBox<[String]>([])
        let log = LichessBotProtocolLog(directory: directory, fileQueue: LichessBotFileQueue()) { error in
            failures.modify { $0.append(error.localizedDescription) }
        }
        let day = Date()
        let logURL = log.fileURL(for: day)
        try FileManager.default.createSymbolicLink(at: logURL, withDestinationURL: target)
        log.record(.game, "must not reach the target", at: day)
        try await log.flush()
        XCTAssertEqual(failures.value, [FileSafetyError.notARegularFile(path: logURL.path, kind: .symbolicLink).localizedDescription])
        XCTAssertEqual(try Data(contentsOf: target), targetBytes, "the target is neither cut nor written")
        XCTAssertEqual(try FileSafety.existingItem(at: logURL)?.kind, .symbolicLink)
    }
}
