//
//  LichessBotAppendPathCompatibilityTests.swift
//  DrewsChessMachineTests
//
//  Journals and protocol-log day files written by builds before the locked
//  append path (challenge-log plan P1) stay readable, and are appended to in
//  place exactly as before: the existing bytes are kept byte for byte, the
//  file is the same file (same inode, same permissions), and the new lines
//  have the same format. The fixtures are literal lines in the format those
//  builds wrote, so a format change shows up here rather than on the
//  owner's data.
//

import XCTest
import Darwin
@testable import DrewsChessMachine

final class LichessBotAppendPathCompatibilityTests: XCTestCase {

    private var tempRoot: URL!

    override func setUpWithError() throws {
        tempRoot = FileManager.default.temporaryDirectory
            .appendingPathComponent("LichessBotAppendPathCompatibilityTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: tempRoot, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try FileManager.default.removeItem(at: tempRoot)
    }

    private var directory: LichessBotDataDirectory { LichessBotDataDirectory(root: tempRoot) }

    /// 2026-10-05T12:00:00Z, the day of the protocol fixtures.
    private static let day = Date(timeIntervalSince1970: 1_791_201_600)

    private static func lines(_ lines: [String]) -> Data {
        Data(lines.map { $0 + "\n" }.joined().utf8)
    }

    private static let oldJournalLines = [
        #"{"at":"2026-10-05T12:00:00.000Z","event":{"header":{"build":2400,"gameID":"g1","gitHash":"0123abc","resumed":false,"schemaVersion":1}}}"#,
        #"{"at":"2026-10-05T12:00:00.125Z","event":{"streamOpened":{"attempt":0}}}"#,
        #"{"at":"2026-10-05T12:00:01.500Z","event":{"keepAlive":{}}}"#,
        #"{"at":"2026-10-05T12:00:02.000Z","event":{"action":{"_0":"written by an earlier build"}}}"#,
    ]

    private static let oldProtocolLines = [
        #"{"at":"2026-10-05T12:00:00.000Z","fields":{"id":"AbCd1234","rated":"true"},"kind":"challenge","message":"challenge sent to someone"}"#,
        #"{"at":"2026-10-05T12:00:03.250Z","fields":{},"gameID":"g1","kind":"game","message":"game started"}"#,
    ]

    private func identityAndMode(of url: URL) throws -> (FileSafety.FileIdentity, mode_t) {
        var info = stat()
        guard lstat(url.path, &info) == 0 else {
            throw FileSafetyError.systemCallFailed(path: url.path, call: "lstat", errnoValue: errno)
        }
        return (FileSafety.FileIdentity(device: info.st_dev, inode: info.st_ino), info.st_mode)
    }

    private func makeJournalWriter(failures: SyncBox<[String]>) -> LichessBotJournalWriter {
        LichessBotJournalWriter(
            directory: directory,
            fileQueue: LichessBotFileQueue(),
            onWriteFailure: { gameID, error in failures.modify { $0.append("\(gameID): \(error)") } },
            onGameFinished: { _ in }
        )
    }

    private func makeProtocolLog(failures: SyncBox<[String]>) -> LichessBotProtocolLog {
        LichessBotProtocolLog(directory: directory, fileQueue: LichessBotFileQueue()) { error in
            failures.modify { $0.append(String(describing: error)) }
        }
    }

    // MARK: - The fixtures are what the earlier builds wrote

    func testTheOldLinesDecodeAndReencodeByteForByte() throws {
        let journal = try LichessBotJSONLines.decode(LichessBotJournalEntry.self, from: Self.lines(Self.oldJournalLines), fileName: "j")
        XCTAssertEqual(journal.elements.count, Self.oldJournalLines.count)
        XCTAssertEqual(try journal.elements.map { try LichessBotJSONLines.encodeLine($0) }.reduce(Data(), +), Self.lines(Self.oldJournalLines))
        let log = try LichessBotJSONLines.decode(LichessBotProtocolEntry.self, from: Self.lines(Self.oldProtocolLines), fileName: "p")
        XCTAssertEqual(log.elements.count, Self.oldProtocolLines.count)
        XCTAssertEqual(try log.elements.map { try LichessBotJSONLines.encodeLine($0) }.reduce(Data(), +), Self.lines(Self.oldProtocolLines))
    }

    // MARK: - Journals

    func testAnOldJournalIsAppendedInPlaceAndStaysReadable() async throws {
        try directory.createDirectories()
        let url = directory.inProgressJournalURL(gameID: "g1")
        let oldBytes = Self.lines(Self.oldJournalLines)
        try oldBytes.write(to: url)
        let (identityBefore, modeBefore) = try identityAndMode(of: url)
        let failures = SyncBox<[String]>([])
        let writer = makeJournalWriter(failures: failures)

        await writer.gameEvent(gameID: "g1", .action("after the upgrade"))
        await writer.gameEvent(gameID: "g1", .action("and again"))

        XCTAssertEqual(failures.value, [])
        let after = try Data(contentsOf: url)
        XCTAssertEqual(after.prefix(oldBytes.count), oldBytes, "the earlier build's bytes are kept exactly")
        let (identityAfter, modeAfter) = try identityAndMode(of: url)
        XCTAssertEqual(identityAfter, identityBefore, "appended in place, never replaced")
        XCTAssertEqual(modeAfter, modeBefore)
        let read = try LichessBotJournal.read(url)
        XCTAssertEqual(read.droppedTrailingByteCount, 0)
        let added = read.elements.dropFirst(Self.oldJournalLines.count).map(\.event)
        guard added.count == 3,
              case .header(let schemaVersion, "g1", BuildInfo.buildNumber, BuildInfo.gitHash, true) = added[0],
              case .action("after the upgrade") = added[1],
              case .action("and again") = added[2] else {
            return XCTFail("unexpected appended entries: \(added)")
        }
        XCTAssertEqual(schemaVersion, LichessBotJournal.schemaVersion)
    }

    func testAnOldJournalWithATornTailIsCutRecordedAndKeptReadable() async throws {
        try directory.createDirectories()
        let url = directory.inProgressJournalURL(gameID: "g1")
        let completeBytes = Self.lines(Self.oldJournalLines)
        let fragment = Data(#"{"at":"2026-10-05T12:00:03.000Z","event":{"streamLi"#.utf8)
        try (completeBytes + fragment).write(to: url)
        XCTAssertEqual(try LichessBotJournal.read(url).droppedTrailingByteCount, fragment.count, "readable before the append")
        let failures = SyncBox<[String]>([])
        let writer = makeJournalWriter(failures: failures)

        await writer.gameEvent(gameID: "g1", .action("after the crash"))

        XCTAssertEqual(failures.value, [])
        XCTAssertEqual(try Data(contentsOf: url).prefix(completeBytes.count), completeBytes)
        let read = try LichessBotJournal.read(url)
        XCTAssertEqual(read.droppedTrailingByteCount, 0)
        let added = read.elements.dropFirst(Self.oldJournalLines.count).map(\.event)
        guard added.count == 3,
              case .header(_, _, _, _, true) = added[0],
              case .anomaly(let note) = added[1],
              case .action("after the crash") = added[2] else {
            return XCTFail("unexpected appended entries: \(added)")
        }
        XCTAssertTrue(note.contains(fragment.base64EncodedString()), note)
    }

    // MARK: - Protocol log

    func testAnOldProtocolDayFileIsAppendedInPlaceAndStaysReadable() async throws {
        try directory.createDirectories()
        let failures = SyncBox<[String]>([])
        let log = makeProtocolLog(failures: failures)
        let url = log.fileURL(for: Self.day)
        let oldBytes = Self.lines(Self.oldProtocolLines)
        try oldBytes.write(to: url)
        let (identityBefore, modeBefore) = try identityAndMode(of: url)

        log.record(.game, "after the upgrade", at: Self.day)
        log.record(.game, "and again", at: Self.day)
        try await log.flush()

        XCTAssertEqual(failures.value, [])
        let after = try Data(contentsOf: url)
        XCTAssertEqual(after.prefix(oldBytes.count), oldBytes, "the earlier build's bytes are kept exactly")
        let (identityAfter, modeAfter) = try identityAndMode(of: url)
        XCTAssertEqual(identityAfter, identityBefore, "appended in place, never replaced")
        XCTAssertEqual(modeAfter, modeBefore)
        let read = try await log.entries(on: Self.day)
        XCTAssertEqual(read.droppedTrailingByteCount, 0)
        XCTAssertEqual(read.elements.map(\.message), ["challenge sent to someone", "game started", "after the upgrade", "and again"])
        XCTAssertEqual(read.elements.map(\.kind), [.challenge, .game, .game, .game], "no repair note on a clean file")
        let newLine = try LichessBotJSONLines.encodeLine(read.elements[2])
        XCTAssertEqual(after.dropFirst(oldBytes.count).prefix(newLine.count), newLine, "new lines keep the old format")
    }

    func testAnOldProtocolDayFileWithATornTailIsCutRecordedAndKeptReadable() async throws {
        try directory.createDirectories()
        let failures = SyncBox<[String]>([])
        let log = makeProtocolLog(failures: failures)
        let url = log.fileURL(for: Self.day)
        let completeBytes = Self.lines(Self.oldProtocolLines)
        let fragment = Data(#"{"at":"2026-10-05T12:00:04"#.utf8)
        try (completeBytes + fragment).write(to: url)

        log.record(.game, "after the crash", at: Self.day)
        try await log.flush()

        XCTAssertEqual(failures.value, [])
        XCTAssertEqual(try Data(contentsOf: url).prefix(completeBytes.count), completeBytes)
        let read = try await log.entries(on: Self.day)
        XCTAssertEqual(read.droppedTrailingByteCount, 0)
        XCTAssertEqual(read.elements.map(\.kind), [.challenge, .game, .anomaly, .game])
        XCTAssertEqual(read.elements[2].fields["base64"], fragment.base64EncodedString())
        XCTAssertEqual(read.elements[2].fields["bytes"], "\(fragment.count)")
        XCTAssertEqual(read.elements.last?.message, "after the crash")
    }

    /// A day file this launch has already appended to, and that another
    /// instance then left a torn line in (it crashed mid-append), is still
    /// repaired on the next append: the tail is checked every time, not only
    /// on a launch's first append to a file.
    func testAFragmentAnotherInstanceLeftIsCutEvenAfterThisLaunchAppended() async throws {
        try directory.createDirectories()
        let failures = SyncBox<[String]>([])
        let log = makeProtocolLog(failures: failures)
        let url = log.fileURL(for: Self.day)
        log.record(.game, "this launch's first line", at: Self.day)
        try await log.flush()
        let fragment = Data(#"{"at":"2026-10-05T12:00:05.000Z","fields":{},"kind":"ga"#.utf8)
        let otherInstance = try FileSafety.openForAppending(at: url)
        try otherInstance.handle.write(contentsOf: fragment)
        try otherInstance.handle.close()

        log.record(.game, "this launch's second line", at: Self.day)
        try await log.flush()

        XCTAssertEqual(failures.value, [])
        let read = try await log.entries(on: Self.day)
        XCTAssertEqual(read.droppedTrailingByteCount, 0)
        XCTAssertEqual(read.elements.map(\.kind), [.game, .anomaly, .game])
        XCTAssertEqual(read.elements[1].fields["base64"], fragment.base64EncodedString())
    }

    /// A day file the new path creates gets the same permissions as one the
    /// earlier builds created (`open` with mode 0644, before the umask).
    func testANewDayFileGetsTheSamePermissionsAsBefore() async throws {
        try directory.createDirectories()
        let failures = SyncBox<[String]>([])
        let log = makeProtocolLog(failures: failures)
        log.record(.game, "first line of a new day", at: Self.day)
        try await log.flush()
        XCTAssertEqual(failures.value, [])
        let oldStyle = tempRoot.appendingPathComponent("old-style.jsonl")
        let descriptor = open(oldStyle.path, O_WRONLY | O_APPEND | O_CREAT | O_CLOEXEC, 0o644)
        XCTAssertGreaterThanOrEqual(descriptor, 0)
        XCTAssertEqual(close(descriptor), 0)
        XCTAssertEqual(try identityAndMode(of: log.fileURL(for: Self.day)).1, try identityAndMode(of: oldStyle).1)
    }
}
