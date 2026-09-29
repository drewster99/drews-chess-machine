import XCTest
@testable import DrewsChessMachine

/// Regression tests for defects found in the pre-ship review of the Lichess
/// bot (resync storms, foreign moves, capacity races, challenge-answer
/// races, journal resurrection, secret redaction).
final class LichessBotReviewFixTests: XCTestCase {

    // MARK: - Game session

    private func makeSession(server: LichessBotFakeGameServer, observer: LichessBotRecordingGameObserver, time: LichessBotManualTime) -> LichessBotGameSession {
        var settings = LichessBotSettings.testBaseline()
        settings.chat.greetingEnabled = false
        let frozen = settings
        let source = LichessBotScriptedMoveSource()
        return LichessBotGameSession(
            gameID: LichessBotFakeGameServer.gameID,
            ourAccountID: LichessBotFakeGameServer.botID,
            api: server,
            moveSource: source,
            latestMoveSource: { source },
            settingsProvider: { frozen },
            observer: observer,
            time: time,
            onTurnStatus: { _, _ in }
        )
    }

    private func stoppedReasons(_ observer: LichessBotRecordingGameObserver) -> [String] {
        observer.events.value.compactMap { event in
            if case .stoppedMoving(let reason) = event { return reason }
            return nil
        }
    }

    private func waitUntil(_ description: String, advancing time: LichessBotManualTime? = nil, _ condition: () async -> Bool) async throws {
        for _ in 0..<4000 {
            if await condition() { return }
            time?.advance(by: .seconds(2))
            try await Task.sleep(for: .milliseconds(3))
        }
        XCTFail("timed out waiting until \(description)")
        throw CancellationError()
    }

    /// A move that keeps being rejected must not loop forever: resyncs back
    /// off, and after a few rejections at one ply the session stops moving.
    func testRepeatedRejectionsStopMovingInsteadOfLooping() async throws {
        let server = try LichessBotFakeGameServer()
        await server.rejectAllMoves()
        let observer = LichessBotRecordingGameObserver()
        let time = LichessBotManualTime()
        let session = makeSession(server: server, observer: observer, time: time)
        let run = Task { await session.run() }
        try await waitUntil("the session stops moving", advancing: time) { !stoppedReasons(observer).isEmpty }
        run.cancel()
        await run.value
        let record = await server.record()
        XCTAssertEqual(record.moveAttempts, 3, "gives up after the per-ply rejection limit")
        XCTAssertLessThanOrEqual(record.streamOpens, 3)
    }

    /// Plan §6.1 B: a move on our side that this client never sent means
    /// another client is playing the account — stop moving at once.
    func testForeignMoveOnOurSideStopsMoving() async throws {
        let server = try LichessBotFakeGameServer()
        await server.setScript(afterOurMoves: [.opponentReplies, .opponentThinks])
        let observer = LichessBotRecordingGameObserver()
        let time = LichessBotManualTime()
        let session = makeSession(server: server, observer: observer, time: time)
        let run = Task { await session.run() }
        try await waitUntil("two moves are posted") { await server.record().acceptedPlies == [0, 2] }
        // Black (the opponent) replies and "someone" plays white's move, in
        // one state.
        try await server.injectMoves(["h7h6", "h2h3"])
        try await waitUntil("the session stops moving") { !stoppedReasons(observer).isEmpty }
        XCTAssertTrue(stoppedReasons(observer).first?.contains("did not send") == true)
        run.cancel()
        await run.value
    }

    /// A 5xx is Lichess failing, not refusing the move: it is retried, and
    /// never counted toward giving up on the game.
    func testServerErrorsOnAMoveAreRetriedNotCountedAsRejections() async throws {
        let server = try LichessBotFakeGameServer()
        await server.setScript(
            moveOutcomes: [.rejected(status: 503), .rejected(status: 502), .rejected(status: 503), .rejected(status: 500)],
            afterOurMoves: [.finish(status: "resign", winner: "white")]
        )
        let observer = LichessBotRecordingGameObserver()
        let time = LichessBotManualTime()
        let session = makeSession(server: server, observer: observer, time: time)
        let run = Task { await session.run() }
        try await waitUntil("the game ends", advancing: time) { observer.finishedStatus != nil }
        await run.value
        let record = await server.record()
        XCTAssertEqual(record.acceptedPlies, [0])
        XCTAssertEqual(record.moveAttempts, 5)
        XCTAssertEqual(stoppedReasons(observer), [])
    }

    /// The end of a post-429 hold must not undo a stop that "one game"
    /// (or a drain) made.
    func testRateLimitHoldEndingKeepsADrainInForce() async throws {
        let time = LichessBotManualTime()
        let manager = LichessBotSessionManager(
            accountAPI: LichessBotFakeAccountAPI(script: []),
            gameAPI: try LichessBotFakeGameServer(),
            gate: LichessBotRequestGate(time: time, breakerWindow: .seconds(3600)) { _ in },
            slots: LichessBotModelSlots(provider: LichessBotFakeModelProvider.unbuildableChampion(), time: time) { _ in },
            ourAccountID: LichessBotFakeGameServer.botID,
            time: time,
            settingsProvider: { LichessBotSettings.testBaseline() },
            gameObserver: LichessBotRecordingGameObserver(),
            onEvent: { _ in }
        )
        await manager.setRateLimitHold(true)
        var accepting = await manager.isAcceptingNewGames
        XCTAssertFalse(accepting)
        await manager.setAcceptingNewGames(false)
        await manager.setRateLimitHold(false)
        accepting = await manager.isAcceptingNewGames
        XCTAssertFalse(accepting, "the drain is still in force")
        await manager.setAcceptingNewGames(true)
        accepting = await manager.isAcceptingNewGames
        XCTAssertTrue(accepting)
    }

    /// A resync in a game resumed after a relaunch must not mistake DCM's
    /// own moves from before the relaunch for another client's.
    func testResyncInAResumedGameIsNotAForeignMove() async throws {
        let server = try LichessBotFakeGameServer(initialTokens: ["e2e4", "e7e5"])
        await server.setScript(afterOurMoves: [.opponentThinks])
        let observer = LichessBotRecordingGameObserver()
        let time = LichessBotManualTime()
        let session = makeSession(server: server, observer: observer, time: time)
        let run = Task { await session.run() }
        try await waitUntil("the first move is posted") { await server.record().acceptedPlies == [2] }
        await server.sendBogusState(["e2e4", "e7e5", "a1a8"])
        try await waitUntil("the stream reopens after the divergence") { await server.record().streamOpens == 2 }
        // The reopened stream's `gameFull`, whose moves are the ones checked
        // for foreign play, is streamed before the sentinel.
        await server.sendSentinel("sentinel-after-resync")
        try await waitUntil("the reopened stream's gameFull has been handled") { observer.receivedLines.contains { $0.contains("sentinel-after-resync") } }
        XCTAssertEqual(stoppedReasons(observer), [])
        run.cancel()
        await run.value
    }

    /// A failed sync must not leave a half-applied position to extend next
    /// time.
    func testFailedSyncLeavesAnEmptyTracker() throws {
        let tracker = try LichessBotPositionTracker(initialFen: "startpos")
        try tracker.sync(to: ["e2e4"])
        XCTAssertThrowsError(try tracker.sync(to: ["e2e4", "e7e5", "e1e8"]))
        XCTAssertEqual(tracker.ply, 0)
        XCTAssertEqual(try tracker.sync(to: ["e2e4", "e7e5"]), .extended(fromPly: 0, toPly: 2), "the next sync starts from the beginning")
    }

    // MARK: - Manager

    private func challengeLine(id: String, challenger: String) -> String {
        #"{"type":"challenge","challenge":{"id":"\#(id)","status":"created","challenger":{"id":"\#(challenger)","name":"\#(challenger)","rating":1500},"variant":{"key":"standard"},"rated":false,"speed":"blitz","timeControl":{"type":"clock","limit":300,"increment":3},"color":"random","direction":"in"},"compat":{"bot":true,"board":true}}"#
    }

    /// Two challenges arriving before the first game starts can't both take
    /// the one free slot.
    func testAcceptedChallengesCountBeforeTheirGameStarts() async throws {
        let provider = try await LichessBotFakeModelProvider.randomChampion()
        let account = LichessBotFakeAccountAPI(script: [.open(lines: [challengeLine(id: "c1", challenger: "alice"), challengeLine(id: "c2", challenger: "bob")])])
        let time = LichessBotManualTime()
        let events = SyncBox<[LichessBotManagerEvent]>([])
        let manager = LichessBotSessionManager(
            accountAPI: account,
            gameAPI: try LichessBotFakeGameServer(),
            gate: LichessBotRequestGate(time: time, breakerWindow: .seconds(3600)) { _ in },
            slots: LichessBotModelSlots(provider: provider, time: time) { _ in },
            ourAccountID: LichessBotFakeGameServer.botID,
            time: time,
            settingsProvider: { LichessBotSettings.testBaseline() },
            gameObserver: LichessBotRecordingGameObserver(),
            onEvent: { event in events.modify { $0.append(event) } }
        )
        await manager.setOneGameMode(true)
        let run = Task { await manager.run() }
        try await waitUntil("both challenges are answered") {
            let accepted = await account.accepted.count
            let declined = await account.declined.count
            return accepted + declined == 2
        }
        let accepted = await account.accepted
        let reasons = await account.declineReasons
        XCTAssertEqual(accepted, ["c1"])
        XCTAssertEqual(reasons, [.later])
        run.cancel()
        await run.value
    }

    /// Lichess echoes our own outgoing challenge on the event stream without
    /// a `direction` field (observed live); it must never be answered.
    func testOwnOutgoingChallengeEchoIsIgnored() async throws {
        let echo = #"{"type":"challenge","challenge":{"id":"ZQRPySw4","status":"created","challenger":{"id":"drewschessmachine","name":"DrewsChessMachine","title":"BOT","rating":3000,"provisional":true},"destUser":{"id":"dala-700","name":"dala-700","title":"BOT","rating":837},"variant":{"key":"standard"},"rated":false,"speed":"blitz","timeControl":{"type":"clock","limit":300,"increment":3},"color":"random","finalColor":"black"},"compat":{"bot":true,"board":true}}"#
        let account = LichessBotFakeAccountAPI(script: [.openThenClose(lines: [echo])])
        let time = LichessBotManualTime()
        let events = SyncBox<[LichessBotManagerEvent]>([])
        let manager = LichessBotSessionManager(
            accountAPI: account,
            gameAPI: try LichessBotFakeGameServer(),
            gate: LichessBotRequestGate(time: time, breakerWindow: .seconds(3600)) { _ in },
            slots: LichessBotModelSlots(provider: LichessBotFakeModelProvider.unbuildableChampion(), time: time) { _ in },
            ourAccountID: LichessBotFakeGameServer.botID,
            time: time,
            settingsProvider: { LichessBotSettings.testBaseline() },
            gameObserver: LichessBotRecordingGameObserver(),
            onEvent: { event in events.modify { $0.append(event) } }
        )
        let run = Task { await manager.run() }
        // The manager reports the stream's end only after it has finished
        // handling every line the stream carried, the echo included.
        try await waitUntil("the stream's only line has been handled") {
            events.value.contains { event in
                if case .eventStreamEnded = event { return true }
                return false
            }
        }
        run.cancel()
        await run.value
        let accepted = await account.accepted
        let declined = await account.declined.count
        XCTAssertEqual(accepted, [])
        XCTAssertEqual(declined, 0)
    }

    /// The answer to our challenge can arrive before its POST returns; it
    /// is still reported when the challenge is noted.
    func testChallengeAnswerBeforeNoteIsReported() async throws {
        let account = LichessBotFakeAccountAPI(script: [.open(lines: [])])
        let time = LichessBotManualTime()
        let events = SyncBox<[LichessBotManagerEvent]>([])
        let manager = LichessBotSessionManager(
            accountAPI: account,
            gameAPI: try LichessBotFakeGameServer(),
            gate: LichessBotRequestGate(time: time, breakerWindow: .seconds(3600)) { _ in },
            slots: LichessBotModelSlots(provider: LichessBotFakeModelProvider.unbuildableChampion(), time: time) { _ in },
            ourAccountID: LichessBotFakeGameServer.botID,
            time: time,
            settingsProvider: { LichessBotSettings.testBaseline() },
            gameObserver: LichessBotRecordingGameObserver(),
            onEvent: { event in events.modify { $0.append(event) } }
        )
        let run = Task { await manager.run() }
        try await waitUntil("the stream opens") { await account.opens == 1 }
        await account.send(#"{"type":"challengeDeclined","challenge":{"id":"ch9","declineReasonKey":"later"}}"#)
        // The answer must be handled before the challenge is noted; the
        // sentinel is read only after it has been.
        await account.sendSentinel("sentinel-after-decline")
        try await waitUntil("the decline has been handled") {
            events.value.contains { $0.isEventStreamLine(containing: "sentinel-after-decline") }
        }
        await manager.noteOutgoingChallenge(id: "ch9")
        let outcomes = events.value.compactMap { event -> LichessBotOutgoingChallengeOutcome? in
            if case .outgoingChallengeResolved(_, let outcome) = event { return outcome }
            return nil
        }
        XCTAssertEqual(outcomes, [.declined(reason: "later", reasonKey: "later")])
        let pending = await manager.outgoingChallengeID
        XCTAssertNil(pending)
        run.cancel()
        await run.value
    }

    // MARK: - Data layer

    private func temporaryDirectory() throws -> LichessBotDataDirectory {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("LichessBotReviewFixTests-\(UUID().uuidString)", isDirectory: true)
        addTeardownBlock {
            do {
                try FileManager.default.removeItem(at: root)
            } catch {
                XCTFail("cleanup failed: \(error)")
            }
        }
        return LichessBotDataDirectory(root: root)
    }

    /// A late write for a game whose journal was filed must not recreate a
    /// headerless fragment in InProgress/.
    func testLateWriteDoesNotRecreateAFiledJournal() async throws {
        let directory = try temporaryDirectory()
        let failures = SyncBox<[String]>([])
        let writer = LichessBotJournalWriter(
            directory: directory,
            fileQueue: LichessBotFileQueue(),
            onWriteFailure: { gameID, error in failures.modify { $0.append("\(gameID): \(error)") } },
            onGameFinished: { _ in }
        )
        await writer.gameEvent(gameID: "g1", .action("first"))
        let url = directory.inProgressJournalURL(gameID: "g1")
        XCTAssertTrue(FileManager.default.fileExists(atPath: url.path))
        try FileManager.default.removeItem(at: url)
        await writer.gameEvent(gameID: "g1", .action("late"))
        XCTAssertFalse(FileManager.default.fileExists(atPath: url.path))
        XCTAssertEqual(failures.value, [])
    }

    /// A stray fragment (no gameFull) must never replace a filed record.
    func testFragmentNeverOverwritesAFiledRecord() async throws {
        let directory = try temporaryDirectory()
        try directory.createDirectories()
        let createdAt: Int64 = 1_759_000_000_000
        let full = #"{"type":"gameFull","id":"g1","variant":{"key":"standard"},"clock":{"initial":180000,"increment":2000},"speed":"blitz","rated":false,"createdAt":\#(createdAt),"white":{"id":"drewschessmachine","name":"DrewsChessMachine"},"black":{"id":"alice","name":"Alice"},"initialFen":"startpos","state":{"type":"gameState","moves":"","wtime":180000,"btime":180000,"winc":2000,"binc":2000,"status":"started"}}"#
        let at = Date(timeIntervalSince1970: 1_759_000_001)
        func write(_ entries: [LichessBotJournalEntry]) throws {
            var data = Data()
            for entry in entries {
                data.append(try LichessBotJSONLines.encodeLine(entry))
            }
            try data.write(to: directory.inProgressJournalURL(gameID: "g1"))
        }
        let header = LichessBotJournalEntry(at: at, event: .header(schemaVersion: 1, gameID: "g1", build: 1, gitHash: "t", resumed: false))
        try write([header, .init(at: at, event: .streamLine(raw: full)), .init(at: at, event: .finished(status: "aborted", winner: nil, localDrawCondition: nil))])
        let store = LichessBotRecordStore(directory: directory, fileQueue: LichessBotFileQueue(), ourAccountID: "drewschessmachine")
        let first = try await store.finalize(gameID: "g1", export: nil, exportUnavailableReason: "test")
        let original = try Data(contentsOf: first.recordURL)

        try write([header, .init(at: at, event: .action("late chat"))])
        let exportJSON = #"{"id":"g1","rated":false,"variant":"standard","speed":"blitz","perf":"blitz","createdAt":\#(createdAt),"status":"aborted","players":{"white":{"user":{"id":"drewschessmachine","name":"DrewsChessMachine"}},"black":{"user":{"id":"alice","name":"Alice"}}},"moves":""}"#
        let second = try await store.finalize(gameID: "g1", export: try LichessBotGameExport.decode(Data(exportJSON.utf8)), exportUnavailableReason: nil)
        XCTAssertEqual(second.recordURL, first.recordURL)
        XCTAssertEqual(try Data(contentsOf: first.recordURL), original, "the complete record is untouched")
        XCTAssertTrue(second.journalURL.lastPathComponent.contains("fragment"))
        let remaining = try await store.inProgressGameIDs()
        XCTAssertEqual(remaining, [])
    }

    // MARK: - Redaction (E26)

    func testFullIdIsRedacted() {
        let line = #"{"type":"gameStart","game":{"gameId":"abcd1234","fullId":"abcd1234wxyz","color":"white"}}"#
        let redacted = LichessBotRedaction.redact(line)
        XCTAssertFalse(redacted.contains("wxyz"))
        XCTAssertTrue(redacted.contains(#""gameId":"abcd1234""#))
    }
}
