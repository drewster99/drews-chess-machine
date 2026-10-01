import XCTest
@testable import DrewsChessMachine

/// The Lichess server's game id became part of file paths unchecked:
/// `InProgress/<gameId>.journal.jsonl` and `Games/YYYY/MM/<stamp>-<gameId>.*`.
/// An id with `/` or `..` could read, write or move files outside the bot's
/// folders. These pin the rule (non-empty, ASCII letters/digits, bounded
/// length) and that every production path builder — journal append, journal
/// read, finalize — refuses a bad id before touching the disk. Every test
/// works in its own temporary directory.
final class LichessBotGameIDPathSafetyTests: XCTestCase {

    private var tempRoot: URL!

    override func setUpWithError() throws {
        tempRoot = FileManager.default.temporaryDirectory
            .appendingPathComponent("LichessBotGameIDPathSafetyTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: tempRoot, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try FileManager.default.removeItem(at: tempRoot)
    }

    /// The bot's data folder sits one level inside `tempRoot`, so a `..`
    /// escape lands somewhere the test can watch.
    private var directory: LichessBotDataDirectory {
        LichessBotDataDirectory(root: tempRoot.appendingPathComponent("LichessBot", isDirectory: true))
    }

    private static let unsafeIDs = [
        "", ".", "..", "../account", "../../escape", "a/b", "/abs", ".hidden",
        "abc def", "é1", "abc\n", "abc\"", "a-b", "a_b", "a.b",
        "abcdefghijklm",
    ]

    private func assertUnsafe(_ error: Error, _ gameID: String, file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(error as? LichessBotGameIDError, .unsafeForFilePath(gameID: gameID), file: file, line: line)
    }

    private func allFiles(under folder: URL) throws -> [String] {
        let enumerator = try XCTUnwrap(FileManager.default.enumerator(atPath: folder.path), "cannot enumerate \(folder.path)")
        return enumerator.compactMap { $0 as? String }.sorted()
    }

    // MARK: - The rule

    func testRealAndFixtureShapedIDsAreSafe() {
        for gameID in ["wtRWmfWC", "abcd1234", "YfjTIV43", "g1", "game0001", "abcdefghijkl"] {
            XCTAssertTrue(LichessBotGameIDPathSafety.isSafe(gameID), gameID)
        }
    }

    func testIDsThatCouldEscapeHideOrInjectAreUnsafe() {
        for gameID in Self.unsafeIDs {
            XCTAssertFalse(LichessBotGameIDPathSafety.isSafe(gameID), gameID.debugDescription)
        }
    }

    func testValidatedBuildersThrowForUnsafeIDs() {
        let createdAt = Date(timeIntervalSince1970: 1_700_000_000)
        for gameID in Self.unsafeIDs {
            XCTAssertThrowsError(try directory.validatedInProgressJournalURL(gameID: gameID)) { assertUnsafe($0, gameID) }
            XCTAssertThrowsError(try LichessBotDataDirectory.validatedFileStem(gameID: gameID, createdAt: createdAt)) { assertUnsafe($0, gameID) }
        }
    }

    func testValidatedBuildersMatchTheUncheckedOnesForASafeID() throws {
        let createdAt = Date(timeIntervalSince1970: 1_700_000_000)
        XCTAssertEqual(try directory.validatedInProgressJournalURL(gameID: "abcd1234"),
                       directory.inProgressJournalURL(gameID: "abcd1234"))
        XCTAssertEqual(try LichessBotDataDirectory.validatedFileStem(gameID: "abcd1234", createdAt: createdAt),
                       LichessBotDataDirectory.fileStem(gameID: "abcd1234", createdAt: createdAt))
    }

    // MARK: - Production paths refuse before touching the disk

    func testJournalAppendRefusesATraversalIDAndWritesNothing() async throws {
        try directory.createDirectories()
        let writer = LichessBotJournalWriter(
            directory: directory,
            fileQueue: LichessBotFileQueue(),
            onWriteFailure: { _, _ in },
            onGameFinished: { _ in }
        )
        let before = try allFiles(under: tempRoot)
        do {
            try await writer.append([], gameID: "../escape", synchronize: false)
            XCTFail("a traversal game id must not be appended")
        } catch {
            assertUnsafe(error, "../escape")
        }
        XCTAssertEqual(try allFiles(under: tempRoot), before, "nothing may be created inside or outside the bot's folders")
    }

    /// `InProgress/../escape.journal.jsonl` is the data root's
    /// `escape.journal.jsonl`; a file planted there must not be read.
    func testJournalReadRefusesATraversalIDInsteadOfReadingOutsideInProgress() async throws {
        try directory.createDirectories()
        let planted = directory.root.appendingPathComponent("escape.journal.jsonl")
        try Data("{}\n".utf8).write(to: planted)
        let store = LichessBotRecordStore(directory: directory, fileQueue: LichessBotFileQueue(), ourAccountID: "drewschessmachine")
        do {
            _ = try await store.readJournal(gameID: "../escape")
            XCTFail("a traversal game id must not be read")
        } catch {
            assertUnsafe(error, "../escape")
        }
    }

    func testFinalizeRefusesATraversalIDAndMovesNothing() async throws {
        try directory.createDirectories()
        let planted = directory.root.appendingPathComponent("escape.journal.jsonl")
        try Data("{}\n".utf8).write(to: planted)
        let store = LichessBotRecordStore(directory: directory, fileQueue: LichessBotFileQueue(), ourAccountID: "drewschessmachine")
        let before = try allFiles(under: tempRoot)
        do {
            _ = try await store.finalize(gameID: "../escape", export: nil, exportUnavailableReason: "test")
            XCTFail("a traversal game id must not be finalized")
        } catch {
            assertUnsafe(error, "../escape")
        }
        XCTAssertEqual(try allFiles(under: tempRoot), before)
    }
}
