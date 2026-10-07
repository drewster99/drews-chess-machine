import XCTest
@testable import DrewsChessMachine

/// The reconstruction cutoff on the first run (challenge-log plan §3.7).
/// With no challenge-log day file at load, the cutoff must follow the first
/// live entry this run writes; left nil, a rebuild in the same launch also
/// rebuilds that launch's not-created sends from the protocol log, and the
/// fold and the Challenge Log window count each of them twice (sends that
/// created no challenge are keyed by attempt id live and by protocol line
/// rebuilt, so nothing dedupes them).
@MainActor
final class LichessBotChallengeLogFirstRunCutoffTests: XCTestCase {
    private var tempRoot: URL!

    override func setUpWithError() throws {
        tempRoot = FileManager.default.temporaryDirectory
            .appendingPathComponent("LichessBotChallengeLogFirstRunCutoffTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: tempRoot, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try FileManager.default.removeItem(at: tempRoot)
    }

    func testFirstRunRebuildDoesNotDoubleCountANotCreatedSend() async throws {
        let directory = LichessBotDataDirectory(root: tempRoot)
        try directory.createDirectories()
        let fileQueue = LichessBotFileQueue()
        let recorder = LichessBotChallengeLogRecorder(directory: directory, fileQueue: fileQueue, systemCalls: .system) { Date() }
        let protocolLog = LichessBotProtocolLog(directory: directory, fileQueue: fileQueue) { _ in }
        await recorder.load()
        XCTAssertNil(recorder.liveLogFirstEntryAt, "empty folder: no live log yet")

        // What sendChallenge does for an offline opponent.
        recorder.record(.outgoingNotCreated(
            attemptID: UUID(), opponentID: "maia1", sender: .challengeQueue, request: LichessBotChallengeLogFixtures.request,
            opponentKind: .bot, reason: .opponentOffline, creditCost: 0))
        protocolLog.record(.challenge, "challenge outcome: maia1 offline", fields: ["credits_day": "0/200", "credits_minute": "0/25"])
        try await recorder.flush()
        _ = try await fileQueue.run { 0 }

        let cutoff = recorder.liveLogFirstEntryAt
        let result = try LichessBotChallengeReconstructionStore.update(in: directory, ourAccountID: "dcmbot", liveLogFirstEntryAt: cutoff)
        let ledger = try XCTUnwrap(recorder.ledger)
        let folded = LichessBotChallengeOutcomeLog.fold(ledger: ledger, history: result.reconstruction, liveLogFirstEntryAt: cutoff, now: Date())
        XCTAssertNotNil(cutoff, "the live log now has an entry; the cutoff should say so")
        XCTAssertEqual(folded.records.count, 1, "one send, one record")
        let rows = LichessBotChallengeLogRow.rows(ledger: ledger, reconstruction: result.reconstruction, pendingChallengeIDs: [])
        XCTAssertEqual(rows.count, 1, "one send, one Challenge Log row")
    }

    /// A failed load leaves a ledger of this run's facts only, so the
    /// cutoff is when the load began: every fact the ledger holds was
    /// recorded at or after it, and the protocol log covers what came
    /// before.
    func testAFailedLoadSetsTheCutoffToWhenTheLoadBegan() async throws {
        let directory = LichessBotDataDirectory(root: tempRoot)
        // `Challenges` is a file, so the folder can't be listed.
        try Data("not a folder".utf8).write(to: directory.challengesDirectory)
        let loadBegan = Date(timeIntervalSince1970: 1_790_000_000)
        let clock = SyncBox(loadBegan)
        let recorder = LichessBotChallengeLogRecorder(directory: directory, fileQueue: LichessBotFileQueue(), systemCalls: .system) { clock.value }
        await recorder.load()
        guard case .failed? = recorder.ledger?.loadStatus else {
            return XCTFail("the load fails: \(String(describing: recorder.ledger?.loadStatus))")
        }
        XCTAssertEqual(recorder.liveLogFirstEntryAt, loadBegan)
        clock.value = loadBegan.addingTimeInterval(60)
        recorder.record(.canceledOnLichess(challengeID: "AbCd1234"))
        XCTAssertEqual(recorder.liveLogFirstEntryAt, loadBegan, "a later entry doesn't move it")
    }
}
