//
//  LichessBotUnmatchedEchoTests.swift
//  DrewsChessMachineTests
//
//  `LichessBotChallengeLogRecorder`'s bookkeeping (challenge-log plan §3.4):
//  our own challenges' echoes are held until their created line and written
//  only when unmatched (attributed to the single unanswered send to that
//  player, compared case-insensitively), game starts are written for known
//  ids and remembered for ids named later, replays write nothing, and a fact
//  recorded while the ledger loads is in it exactly once. Each test has its
//  own temporary data folder and a test clock.
//

import XCTest
@testable import DrewsChessMachine

@MainActor
final class LichessBotUnmatchedEchoTests: XCTestCase {

    private typealias F = LichessBotChallengeLogFixtures

    private var tempRoot: URL!
    private var now = F.start

    override func setUpWithError() throws {
        tempRoot = FileManager.default.temporaryDirectory
            .appendingPathComponent("LichessBotUnmatchedEchoTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: tempRoot, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try FileManager.default.removeItem(at: tempRoot)
    }

    private var directory: LichessBotDataDirectory { LichessBotDataDirectory(root: tempRoot) }

    private func makeRecorder(fileQueue: LichessBotFileQueue = LichessBotFileQueue(), alarms: SyncBox<[String]> = SyncBox([])) async -> LichessBotChallengeLogRecorder {
        let recorder = LichessBotChallengeLogRecorder(directory: directory, fileQueue: fileQueue, systemCalls: .system) { [unowned self] in
            self.now
        }
        recorder.alarmSink = { text in alarms.modify { $0.append(text) } }
        await recorder.load()
        return recorder
    }

    private func writtenEvents(_ recorder: LichessBotChallengeLogRecorder) async throws -> [LichessBotChallengeLogEvent] {
        try await recorder.flush()
        return try LichessBotChallengeLog.readAll(in: directory).entries.map(\.event)
    }

    private func noAnswerSend(to opponentID: String, attempt: UInt8, sender: LichessBotChallengeSender = .challengeQueue) -> LichessBotChallengeLogEvent {
        .outgoingNotCreated(attemptID: F.attemptID(attempt), opponentID: opponentID, sender: sender, request: F.request,
                            opponentKind: .bot, reason: .noAnswer(error: "timed out"), creditCost: 1)
    }

    func testAnEchoMatchedByItsCreatedLineWritesNothing() async throws {
        let recorder = await makeRecorder()
        recorder.noteOwnEcho(F.snapshot())
        recorder.record(F.created())
        now = now.addingTimeInterval(LichessBotChallengeLogRecorder.echoMatchWindow + 1)
        recorder.writeExpiredEchoes()
        recorder.writeAllHeldEchoes()
        let written1 = try await writtenEvents(recorder)
        XCTAssertEqual(written1, [F.created()])
    }

    func testAnUnmatchedEchoIsAttributedToTheOneUnansweredSendCaseInsensitively() async throws {
        let recorder = await makeRecorder()
        recorder.record(noAnswerSend(to: "Maia1", attempt: 1))
        now = now.addingTimeInterval(1)
        recorder.noteOwnEcho(F.snapshot())
        now = now.addingTimeInterval(LichessBotChallengeLogRecorder.echoMatchWindow - 2)
        recorder.writeExpiredEchoes()
        let written2 = try await writtenEvents(recorder)
        XCTAssertEqual(written2.count, 1, "still inside the window: held")
        now = now.addingTimeInterval(2)
        recorder.writeExpiredEchoes()
        let events = try await writtenEvents(recorder)
        XCTAssertEqual(events.last, .outgoingSeenWithoutCreatedLine(challenge: F.snapshot(), attribution: .unansweredSend(attemptID: F.attemptID(1), sender: .challengeQueue)))
        XCTAssertEqual(recorder.ledger?.row(challengeID: "AbCd1234")?.sender, .challengeQueue)
    }

    func testAnEchoWithNoneOrTwoUnansweredSendsIsNotAttributed() async throws {
        let none = await makeRecorder()
        none.noteOwnEcho(F.snapshot())
        now = now.addingTimeInterval(LichessBotChallengeLogRecorder.echoMatchWindow)
        none.writeExpiredEchoes()
        let written3 = try await writtenEvents(none)
        XCTAssertEqual(written3, [.outgoingSeenWithoutCreatedLine(challenge: F.snapshot(), attribution: .notRecorded)])

        try FileManager.default.removeItem(at: tempRoot)
        try FileManager.default.createDirectory(at: tempRoot, withIntermediateDirectories: true)
        now = F.start
        let two = await makeRecorder()
        two.record(noAnswerSend(to: "maia1", attempt: 1))
        two.record(noAnswerSend(to: "maia1", attempt: 2))
        two.noteOwnEcho(F.snapshot())
        now = now.addingTimeInterval(LichessBotChallengeLogRecorder.echoMatchWindow)
        two.writeExpiredEchoes()
        let written4 = try await writtenEvents(two)
        XCTAssertEqual(written4.last, .outgoingSeenWithoutCreatedLine(challenge: F.snapshot(), attribution: .notRecorded))
    }

    func testTeardownWritesEveryHeldEcho() async throws {
        let recorder = await makeRecorder()
        recorder.noteOwnEcho(F.snapshot(id: "one"))
        recorder.noteOwnEcho(F.snapshot(id: "two"))
        recorder.writeAllHeldEchoes()
        let events = try await writtenEvents(recorder)
        XCTAssertEqual(Set(events.compactMap { event -> String? in
            guard case .outgoingSeenWithoutCreatedLine(let challenge, .notRecorded) = event else { return nil }
            return challenge.id
        }), ["one", "two"])
    }

    func testAnEchoOfAChallengeTheLogAlreadyHoldsIsNotHeld() async throws {
        // An earlier run's created line, on disk before this run loads.
        let earlier = await makeRecorder()
        earlier.record(F.created())
        try await earlier.flush()
        let recorder = await makeRecorder()
        XCTAssertNotNil(recorder.ledger?.row(challengeID: "AbCd1234"))
        recorder.noteOwnEcho(F.snapshot())
        recorder.writeAllHeldEchoes()
        let written5 = try await writtenEvents(recorder)
        XCTAssertEqual(written5, [F.created()], "a reconnect's replay writes nothing")
    }

    func testAGameStartBeforeItsCreatedLineIsWrittenRightAfterIt() async throws {
        let recorder = await makeRecorder()
        // The POST race with the echo lost: nothing names the id yet.
        recorder.noteGameStart(gameID: "AbCd1234")
        let written6 = try await writtenEvents(recorder)
        XCTAssertEqual(written6, [])
        recorder.record(F.created())
        let written7 = try await writtenEvents(recorder)
        XCTAssertEqual(written7, [F.created(), .gameStarted(challengeID: "AbCd1234")])
        XCTAssertEqual(recorder.ledger?.row(challengeID: "AbCd1234")?.state, .accepted(gameStarted: true))
    }

    func testAGameStartWithItsEchoHeldIsWrittenAtOnce() async throws {
        let recorder = await makeRecorder()
        recorder.noteOwnEcho(F.snapshot())
        recorder.noteGameStart(gameID: "AbCd1234")
        recorder.record(F.created())
        let written8 = try await writtenEvents(recorder)
        XCTAssertEqual(written8, [.gameStarted(challengeID: "AbCd1234"), F.created()])
        recorder.noteGameStart(gameID: "AbCd1234")
        let written9 = try await writtenEvents(recorder)
        XCTAssertEqual(written9.count, 2, "a replayed gameStart writes nothing")
    }

    func testAnUnknownGameStartWritesNothing() async throws {
        let recorder = await makeRecorder()
        recorder.noteGameStart(gameID: "tournament1")
        let written10 = try await writtenEvents(recorder)
        XCTAssertEqual(written10, [])
    }

    func testAReplayedIncomingChallengeIsWrittenOnce() async throws {
        let recorder = await makeRecorder()
        let incoming = F.snapshot(id: "InCo5678", challenger: F.maia, destUser: F.ourAccount)
        recorder.noteIncomingChallenge(incoming)
        recorder.noteIncomingChallenge(incoming)
        let written11 = try await writtenEvents(recorder)
        XCTAssertEqual(written11, [.incomingReceived(challenge: incoming)])
    }

    /// A fact recorded while the load's read waits on the file queue is
    /// folded on top of the loaded ledger exactly once.
    func testAFactRecordedDuringTheLoadIsInTheLedgerOnce() async throws {
        let fileQueue = LichessBotFileQueue()
        let recorder = LichessBotChallengeLogRecorder(directory: directory, fileQueue: fileQueue, systemCalls: .system) { [unowned self] in
            self.now
        }
        let gate = DispatchSemaphore(value: 0)
        fileQueue.enqueue("hold the queue") { gate.wait() }
        let load = Task { await recorder.load() }
        // Let the load enqueue its read behind the held queue.
        for _ in 0..<50 { await Task.yield() }
        recorder.record(F.created())
        gate.signal()
        await load.value
        let row = try XCTUnwrap(recorder.ledger?.row(challengeID: "AbCd1234"))
        XCTAssertEqual(row.facts.count, 1)
        let written12 = try await writtenEvents(recorder)
        XCTAssertEqual(written12, [F.created()])
    }

    func testAFailedLoadKeepsThisRunsFactsAndRaisesAnAlarm() async throws {
        try Data("not a folder".utf8).write(to: tempRoot.appendingPathComponent("Challenges", isDirectory: false))
        let alarms = SyncBox<[String]>([])
        let recorder = await makeRecorder(alarms: alarms)
        guard case .failed = recorder.ledger?.loadStatus else {
            return XCTFail("expected a failed load, got \(String(describing: recorder.ledger?.loadStatus))")
        }
        XCTAssertEqual(alarms.value.count, 1)
        recorder.record(F.created())
        XCTAssertEqual(recorder.ledger?.rowsByKey.count, 1)
    }
}
