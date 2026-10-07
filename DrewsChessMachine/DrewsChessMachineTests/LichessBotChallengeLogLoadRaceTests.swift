import XCTest
@testable import DrewsChessMachine

/// The challenge-log load's ordering promise (challenge-log plan §3.4): the
/// read is enqueued on the file queue at a known point on the main actor, so
/// a fact recorded after `load()` started is either in the files the read
/// sees or held for the ledger — never both. If the read were enqueued only
/// after the call left the main actor, a fact recorded in that gap would
/// have its append enqueued first, the read would see it, and the held copy
/// would be folded on top: the ledger would count it twice.
@MainActor
final class LichessBotChallengeLogLoadRaceTests: XCTestCase {
    private var tempRoot: URL!

    override func setUpWithError() throws {
        tempRoot = FileManager.default.temporaryDirectory
            .appendingPathComponent("LichessBotChallengeLogLoadRaceTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: tempRoot, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try FileManager.default.removeItem(at: tempRoot)
    }

    func testAFactRecordedWhileTheLoadStartsIsFoldedOnce() async throws {
        let directory = LichessBotDataDirectory(root: tempRoot)
        try directory.createDirectories()
        let recorder = LichessBotChallengeLogRecorder(directory: directory, fileQueue: LichessBotFileQueue(), systemCalls: .system) { Date() }
        let loading = Task { await recorder.load() }
        // Let the load run to its first suspension.
        await Task.yield()
        recorder.record(.canceledOnLichess(challengeID: "AbCd1234"))
        await loading.value
        try await recorder.flush()

        let row = try XCTUnwrap(recorder.ledger?.row(challengeID: "AbCd1234"))
        XCTAssertEqual(row.facts.count, 1, "one fact recorded, one fact in the ledger")
        let onDisk = try await LichessBotChallengeLog(directory: directory, fileQueue: LichessBotFileQueue(), systemCalls: .system) { _ in }.readAll()
        XCTAssertEqual(onDisk.entries.count, 1, "and one line in the files")
    }
}
