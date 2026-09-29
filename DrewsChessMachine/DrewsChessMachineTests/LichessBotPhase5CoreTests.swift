import XCTest
@testable import DrewsChessMachine

/// Phase 5's non-UI additions: request records for the protocol
/// transcript, outgoing challenges, online bots, keep-alive gap statistics,
/// "Play one game", and journaling around finalize (plan §7.1, §14.3a).
final class LichessBotPhase5CoreTests: XCTestCase {

    // MARK: - Request records (plan §14.3a)

    private func makeClient(records: SyncBox<[LichessBotRequestRecord]>, _ handler: @escaping LichessBotScriptedTransport.Handler) throws -> LichessBotAPIClient {
        let gate = LichessBotRequestGate(time: LichessBotVirtualTime(), breakerWindow: .seconds(3600)) { _ in }
        return LichessBotAPIClient(
            baseURL: try LichessBotAPIClient.lichessBaseURL(),
            token: "lip_SECRETTOKEN",
            transport: LichessBotScriptedTransport(handler: handler),
            gate: gate,
            onRequest: { record in records.modify { $0.append(record) } }
        )
    }

    func testEveryRequestIsRecordedWithoutTheToken() async throws {
        let records = SyncBox<[LichessBotRequestRecord]>([])
        let client = try makeClient(records: records) { request in
            if request.url?.path == "/api/bot/game/g1/chat" {
                return (400, Data(#"{"error":"Too long"}"#.utf8), [:])
            }
            return (200, Data(#"{"ok":true}"#.utf8), [:])
        }
        try await client.makeMove(gameID: "g1", uci: "e2e4", offeringDraw: true)
        do {
            try await client.chat(gameID: "g1", room: .player, text: "hello there")
            XCTFail("the scripted 400 should throw")
        } catch LichessBotAPIError.http(let status, _) {
            XCTAssertEqual(status, 400)
        }

        let recorded = records.value
        XCTAssertEqual(recorded.count, 2)
        XCTAssertEqual(recorded[0].gameID, "g1")
        XCTAssertEqual(recorded[0].method, "POST")
        XCTAssertEqual(recorded[0].path, "/api/bot/game/g1/move/e2e4?offeringDraw=true")
        XCTAssertEqual(recorded[0].status, 200)
        XCTAssertEqual(recorded[0].networkProtocol, "h2")
        XCTAssertNotNil(recorded[0].roundTripMilliseconds)
        XCTAssertEqual(recorded[1].formFields, ["room": "player", "text": "hello there"])
        XCTAssertEqual(recorded[1].status, 400)
        XCTAssertEqual(recorded[1].errorMessage, "Too long")

        let encoded = String(decoding: try JSONEncoder().encode(recorded), as: UTF8.self)
        XCTAssertFalse(encoded.contains("SECRETTOKEN"))
    }

    /// The token-test body *is* the token: it is never recorded.
    func testTokenTestBodyIsNotRecorded() async throws {
        let records = SyncBox<[LichessBotRequestRecord]>([])
        let client = try makeClient(records: records) { _ in
            (200, Data(#"{"lip_SECRETTOKEN":{"userId":"drewschessmachine","scopes":"bot:play","expires":null}}"#.utf8), [:])
        }
        _ = try await client.testToken()
        let recorded = try XCTUnwrap(records.value.first)
        XCTAssertEqual(recorded.formFields, [:])
        XCTAssertFalse(String(decoding: try JSONEncoder().encode(recorded), as: UTF8.self).contains("SECRETTOKEN"))
    }

    /// A request the gate refuses is still recorded, with no status.
    func testRefusedRequestIsRecorded() async throws {
        let records = SyncBox<[LichessBotRequestRecord]>([])
        let gate = LichessBotRequestGate(time: LichessBotVirtualTime(), breakerWindow: .seconds(3600)) { _ in }
        let client = LichessBotAPIClient(
            baseURL: try LichessBotAPIClient.lichessBaseURL(),
            token: "lip_x",
            transport: LichessBotScriptedTransport { _ in (200, Data(), [:]) },
            gate: gate,
            onRequest: { record in records.modify { $0.append(record) } }
        )
        await gate.close(reason: "offline")
        do {
            try await client.resign(gameID: "g1")
            XCTFail("a closed gate should refuse")
        } catch LichessBotGateError.closed {
        }
        let recorded = try XCTUnwrap(records.value.first)
        XCTAssertNil(recorded.status)
        XCTAssertNotNil(recorded.failure)
        XCTAssertEqual(recorded.gameID, "g1")
    }

    // MARK: - Outgoing challenges and online bots (plan §7.1)

    func testChallengeRequestShapeAndBothResponseShapes() async throws {
        let challengeJSON = #"{"id":"ch1","status":"created","challenger":{"id":"drewschessmachine","name":"DrewsChessMachine","title":"BOT"},"variant":{"key":"standard"},"rated":false,"speed":"blitz","timeControl":{"type":"clock","limit":300,"increment":3},"color":"random","direction":"out"}"#
        for body in [challengeJSON, #"{"challenge":"# + challengeJSON + "}"] {
            let records = SyncBox<[LichessBotRequestRecord]>([])
            let client = try makeClient(records: records) { _ in (200, Data(body.utf8), [:]) }
            let created = try await client.challenge(
                username: "alice",
                request: LichessBotOutgoingChallenge(rated: false, clockLimitSeconds: 300, clockIncrementSeconds: 3, color: .random)
            )
            XCTAssertEqual(created.id, "ch1")
            let recorded = try XCTUnwrap(records.value.first)
            XCTAssertEqual(recorded.path, "/api/challenge/alice")
            XCTAssertEqual(recorded.formFields, ["rated": "false", "clock.limit": "300", "clock.increment": "3", "color": "random", "variant": "standard"])
        }
    }

    func testUnexpectedChallengeResponseIsAnError() {
        XCTAssertThrowsError(try LichessBotOutgoingChallenge.decodeCreated(Data(#"{"ok":true}"#.utf8)))
    }

    func testOnlineBotsParseNDJSONIncludingAnUnterminatedLastLine() async throws {
        let records = SyncBox<[LichessBotRequestRecord]>([])
        let body = #"{"id":"bota","username":"BotA","title":"BOT","perfs":{"blitz":{"rating":1800,"games":50}}}"# + "\n" + #"{"id":"botb","username":"BotB","title":"BOT"}"#
        let client = try makeClient(records: records) { request in
            XCTAssertNil(request.value(forHTTPHeaderField: "Authorization"), "online bots needs no token")
            return (200, Data(body.utf8), [:])
        }
        let bots = try await client.onlineBots(count: 50)
        XCTAssertEqual(bots.map(\.id), ["bota", "botb"])
        XCTAssertEqual(bots[0].rating("blitz")?.rating, 1800)
        XCTAssertTrue(bots.allSatisfy(\.isBot))
    }

    // MARK: - Keep-alive gap statistics

    func testGapTrackerSummarizesPerWindowAndFlagsLongGaps() {
        var tracker = LichessBotStreamGapTracker(window: .seconds(60), longGap: .seconds(14))
        var reports: [LichessBotStreamGapTracker.Report] = []
        for second in stride(from: 0, through: 56, by: 7) {
            reports += tracker.arrival(at: .seconds(second))
        }
        XCTAssertEqual(reports, [])
        reports += tracker.arrival(at: .seconds(76))
        XCTAssertEqual(reports.count, 2)
        XCTAssertEqual(reports[0], .longGap(seconds: 20))
        guard case .summary(let summary) = reports[1] else {
            return XCTFail("expected a summary")
        }
        XCTAssertEqual(summary.gapCount, 9)
        XCTAssertEqual(summary.maximumSeconds, 20)
        XCTAssertEqual(summary.windowSeconds, 76)

        tracker.reset()
        XCTAssertEqual(tracker.arrival(at: .seconds(500)), [], "the silence across a reconnect is not a gap")
    }

    // MARK: - Manager: outgoing challenges and one game

    private func makeManager(script: [LichessBotFakeAccountAPI.Connection], provider: LichessBotFakeModelProvider, server: LichessBotFakeGameServer, events: SyncBox<[LichessBotManagerEvent]>, time: LichessBotManualTime) -> (LichessBotSessionManager, LichessBotFakeAccountAPI) {
        var settings = LichessBotSettings.testBaseline()
        settings.chat.greetingEnabled = false
        let frozen = settings
        let account = LichessBotFakeAccountAPI(script: script)
        let manager = LichessBotSessionManager(
            accountAPI: account,
            gameAPI: server,
            gate: LichessBotRequestGate(time: time, breakerWindow: .seconds(3600)) { _ in },
            slots: LichessBotModelSlots(provider: provider, time: time) { _ in },
            ourAccountID: LichessBotFakeGameServer.botID,
            time: time,
            settingsProvider: { frozen },
            gameObserver: LichessBotRecordingGameObserver(),
            onEvent: { event in events.modify { $0.append(event) } }
        )
        return (manager, account)
    }

    private func waitUntil(_ description: String, _ condition: () async -> Bool) async throws {
        for _ in 0..<4000 {
            if await condition() { return }
            try await Task.sleep(for: .milliseconds(5))
        }
        XCTFail("timed out waiting until \(description)")
        throw CancellationError()
    }

    private func outcomes(_ events: SyncBox<[LichessBotManagerEvent]>) -> [LichessBotOutgoingChallengeOutcome] {
        events.value.compactMap { event in
            if case .outgoingChallengeResolved(_, let outcome) = event { return outcome }
            return nil
        }
    }

    private func resolvedChallengeIDs(_ events: SyncBox<[LichessBotManagerEvent]>) -> [String] {
        events.value.compactMap { event in
            if case .outgoingChallengeResolved(let challengeID, _) = event { return challengeID }
            return nil
        }
    }

    func testOutgoingChallengeDeclineAndCancelAreReported() async throws {
        let events = SyncBox<[LichessBotManagerEvent]>([])
        let time = LichessBotManualTime()
        let (manager, account) = makeManager(
            script: [.open(lines: [])],
            provider: LichessBotFakeModelProvider.unbuildableChampion(),
            server: try LichessBotFakeGameServer(),
            events: events,
            time: time
        )
        let run = Task { await manager.run() }
        try await waitUntil("the stream opens") { await account.opens == 1 }

        await manager.noteOutgoingChallenge(id: "ch1")
        await account.send(#"{"type":"challengeDeclined","challenge":{"id":"ch1","declineReason":"I'm not accepting challenges right now.","declineReasonKey":"later"}}"#)
        try await waitUntil("the decline is reported") { outcomes(events).count == 1 }
        XCTAssertEqual(outcomes(events), [.declined(reason: "I'm not accepting challenges right now.", reasonKey: "later")])
        let pendingAfterDecline = await manager.outgoingChallengeID
        XCTAssertNil(pendingAfterDecline)

        await manager.noteOutgoingChallenge(id: "ch2")
        await account.send(#"{"type":"challengeCanceled","challenge":{"id":"ch2"}}"#)
        try await waitUntil("the cancel is reported") { outcomes(events).count == 2 }
        XCTAssertEqual(outcomes(events).last, .canceled)

        await account.send(#"{"type":"challengeDeclined","challenge":{"id":"someone-elses"}}"#)
        // The stream is handled in order, so once the answer to a challenge
        // of ours sent after it is reported, the foreign one has been
        // handled too.
        await manager.noteOutgoingChallenge(id: "ch3")
        await account.send(#"{"type":"challengeDeclined","challenge":{"id":"ch3","declineReasonKey":"generic"}}"#)
        try await waitUntil("the later decline of our own challenge is reported") { resolvedChallengeIDs(events).contains("ch3") }
        XCTAssertEqual(outcomes(events).count, 3, "only our pending challenge is tracked")
        XCTAssertEqual(resolvedChallengeIDs(events), ["ch1", "ch2", "ch3"])
        run.cancel()
        await run.value
    }

    /// "Play one game": the accepted challenge's game starts, the outgoing
    /// challenge resolves as accepted, and the manager stops accepting.
    func testOneGameModeStopsAcceptingAfterTheGameStarts() async throws {
        let events = SyncBox<[LichessBotManagerEvent]>([])
        let time = LichessBotManualTime()
        let server = try LichessBotFakeGameServer()
        await server.setScript(afterOurMoves: [.opponentThinks])
        let provider = try await LichessBotFakeModelProvider.randomChampion()
        let (manager, account) = makeManager(script: [.open(lines: [])], provider: provider, server: server, events: events, time: time)
        await manager.setOneGameMode(true)
        let run = Task { await manager.run() }
        try await waitUntil("the stream opens") { await account.opens == 1 }

        await manager.noteOutgoingChallenge(id: LichessBotFakeGameServer.gameID)
        await account.send(#"{"type":"gameStart","game":{"gameId":"\#(LichessBotFakeGameServer.gameID)","opponent":{"id":"alice"}}}"#)
        try await waitUntil("the one game starts") {
            events.value.contains { event in
                if case .oneGameStarted = event { return true }
                return false
            }
        }
        XCTAssertEqual(outcomes(events), [.accepted(gameID: LichessBotFakeGameServer.gameID)])
        let stillOneGame = await manager.isOneGameMode
        XCTAssertFalse(stillOneGame)

        let challenge = #"{"type":"challenge","challenge":{"id":"c9","status":"created","challenger":{"id":"bob","name":"bob","rating":1500},"variant":{"key":"standard"},"rated":false,"speed":"blitz","timeControl":{"type":"clock","limit":300,"increment":3},"color":"random","direction":"in"},"compat":{"bot":true,"board":true}}"#
        await account.send(challenge)
        try await waitUntil("the new challenge is answered") { await account.declined.count == 1 }
        let reasons = await account.declineReasons
        XCTAssertEqual(reasons, [.later])

        await manager.abandonAllSessions()
        run.cancel()
        await run.value
    }

    // MARK: - Journal around finalize

    func testNothingIsJournaledForAFiledGame() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("LichessBotPhase5CoreTests-\(UUID().uuidString)", isDirectory: true)
        defer {
            do {
                try FileManager.default.removeItem(at: root)
            } catch {
                XCTFail("cleanup failed: \(error)")
            }
        }
        let directory = LichessBotDataDirectory(root: root)
        let failures = SyncBox<[String]>([])
        let writer = LichessBotJournalWriter(
            directory: directory,
            fileQueue: LichessBotFileQueue(),
            onWriteFailure: { gameID, error in failures.modify { $0.append("\(gameID): \(error)") } },
            onGameFinished: { _ in }
        )
        let request = LichessBotRequestRecord(
            // Whole seconds: the journal stores times to the millisecond.
            startedAt: Date(timeIntervalSince1970: 1_759_000_000), gameID: "g1", label: "move", method: "POST", path: "/api/bot/game/g1/move/e2e4",
            formFields: [:], status: 200, queuedMilliseconds: 1, roundTripMilliseconds: 40, networkProtocol: "h2",
            errorMessage: nil, failure: nil
        )
        await writer.gameEvent(gameID: "g1", .keepAlive(receivedAt: Date()))
        await writer.recordRequest(request)
        let journal = try LichessBotJournal.read(directory.inProgressJournalURL(gameID: "g1"))
        XCTAssertEqual(journal.elements.count, 3, "header, keep-alive, request")
        guard case .keepAlive = journal.elements[1].event, case .request(let recorded) = journal.elements[2].event else {
            return XCTFail("unexpected journal: \(journal.elements.map(\.event))")
        }
        XCTAssertEqual(recorded, request)

        try FileManager.default.removeItem(at: directory.inProgressJournalURL(gameID: "g1"))
        writer.markFinalized(gameID: "g1")
        await writer.recordRequest(request)
        await writer.gameEvent(gameID: "g1", .action("late"))
        XCTAssertFalse(FileManager.default.fileExists(atPath: directory.inProgressJournalURL(gameID: "g1").path), "a filed game's journal is never recreated")
        XCTAssertEqual(failures.value, [])
    }
}
