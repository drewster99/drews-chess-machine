import XCTest
@testable import DrewsChessMachine

/// Live-game resume and anomaly de-duplication, per-opponent accounting of
/// outgoing challenges, and alarm de-duplication (Lichess bot plan §7, §13,
/// §14.3a).
final class LichessBotGroupBFixTests: XCTestCase {

    // MARK: - Live game

    private func stateLine(moves: String, status: String, winner: String? = nil) -> Data {
        let winnerField = winner.map { #","winner":"\#($0)""# } ?? ""
        return Data(#"{"type":"gameState","moves":"\#(moves)","wtime":300000,"btime":300000,"winc":3000,"binc":3000,"status":"\#(status)"\#(winnerField)}"#.utf8)
    }

    @MainActor
    func testResumedSessionUndoesLeftUnfinishedAndKeepsPacing() {
        let game = LichessBotLiveGame(id: "g1", startedAt: Date(), ourAccountID: "drewschessmachine")
        game.apply(.streamLine(stateLine(moves: "e2e4", status: "started"), receivedAt: Date()))
        game.holdsMoves = true
        game.moveDelaySeconds = 5
        game.markSessionEnded("offline")
        XCTAssertTrue(game.isFinished)
        XCTAssertEqual(game.status, LichessBotLiveGame.leftUnfinishedStatus)

        game.resumeFollowing()
        XCTAssertFalse(game.isFinished)
        XCTAssertEqual(game.status, "started")
        XCTAssertTrue(game.holdsMoves, "the operator's pacing is kept")
        XCTAssertEqual(game.moveDelaySeconds, 5)
    }

    @MainActor
    func testAGameThatReallyEndedIsNotResumed() {
        let game = LichessBotLiveGame(id: "g1", startedAt: Date(), ourAccountID: "drewschessmachine")
        game.apply(.streamLine(stateLine(moves: "e2e4", status: "resign", winner: "black"), receivedAt: Date()))
        XCTAssertTrue(game.isFinished)
        game.resumeFollowing()
        XCTAssertTrue(game.isFinished)
        XCTAssertEqual(game.status, "resign")
    }

    @MainActor
    func testUnreplayableMoveIsReportedOnce() {
        let game = LichessBotLiveGame(id: "g1", startedAt: Date(), ourAccountID: "drewschessmachine")
        let line = stateLine(moves: "e2e4 e2e4", status: "started")
        game.apply(.streamLine(line, receivedAt: Date()))
        game.apply(.streamLine(line, receivedAt: Date()))
        XCTAssertEqual(game.anomalies.count, 1)
    }

    // MARK: - Manager harness

    private struct Harness {
        let account: LichessBotFakeAccountAPI
        let manager: LichessBotSessionManager
        let events: SyncBox<[LichessBotManagerEvent]>
        let time: LichessBotManualTime
    }

    private func makeHarness(provider: any LichessBotModelProvider, configure: (inout LichessBotSettings) -> Void) async throws -> Harness {
        var settings = LichessBotSettings.testBaseline()
        settings.chat.greetingEnabled = false
        configure(&settings)
        let frozen = settings
        let time = LichessBotManualTime()
        let account = LichessBotFakeAccountAPI(script: [.open(lines: [])])
        let gate = LichessBotRequestGate(time: time, breakerWindow: .seconds(3600)) { _ in }
        let slots = try await LichessBotModelSlots.prepare(for: frozen.model, provider: provider, time: time, log: { _ in })
        let events = SyncBox<[LichessBotManagerEvent]>([])
        let manager = LichessBotSessionManager(
            accountAPI: account,
            gameAPI: try LichessBotFakeGameServer(),
            gate: gate,
            slots: slots,
            ourAccountID: LichessBotFakeGameServer.botID,
            time: time,
            settingsProvider: { frozen },
            gameObserver: LichessBotRecordingGameObserver(),
            onEvent: { event in events.modify { $0.append(event) } },
            journalReader: LichessBotTestJournalReaders.none
        )
        return Harness(account: account, manager: manager, events: events, time: time)
    }

    private func waitUntil(_ description: String, _ condition: () async -> Bool) async throws {
        for _ in 0..<4000 {
            if await condition() { return }
            try await Task.sleep(for: .milliseconds(5))
        }
        XCTFail("timed out waiting until \(description)")
        throw CancellationError()
    }

    private func challengeLine(id: String, challenger: String) -> String {
        #"{"type":"challenge","challenge":{"id":"\#(id)","status":"created","challenger":{"id":"\#(challenger)","name":"\#(challenger)","rating":1500},"variant":{"key":"standard","name":"x","short":"x"},"rated":false,"speed":"blitz","timeControl":{"type":"clock","limit":300,"increment":3,"show":"5+3"},"color":"random","direction":"in"},"compat":{"bot":true,"board":true}}"#
    }

    private func declineRules(_ events: [LichessBotManagerEvent]) -> [String] {
        events.compactMap { event in
            if case .challengeDecision(_, _, .decline(_, let rule)) = event { return rule }
            return nil
        }
    }

    // MARK: - Per-opponent accounting

    func testPendingOutgoingChallengeCountsAgainstThatOpponent() async throws {
        let h = try await makeHarness(provider: try await LichessBotFakeModelProvider.randomChampion()) { settings in
            settings.challenge.maxConcurrentGames = 4
            settings.challenge.maxSimultaneousGamesPerOpponent = 1
        }
        let run = Task { await h.manager.run() }
        defer { run.cancel() }
        try await waitUntil("the event stream is open") { await h.account.opens == 1 }

        await h.manager.noteOutgoingChallenge(id: "mine", opponentID: "Bob")
        await h.account.send(challengeLine(id: "c1", challenger: "bob"))
        try await waitUntil("the incoming challenge is declined") { await h.account.declined.count == 1 }
        let reasons = await h.account.declineReasons
        XCTAssertEqual(reasons, [.later])
        XCTAssertEqual(declineRules(h.events.value), ["already playing this opponent"])

        let blocked = await h.manager.reserveOutgoingChallenge(against: "BOB", perOpponentLimit: 1)
        XCTAssertEqual(blocked.committedBefore, 1)
        XCTAssertFalse(blocked.reserved)
        let other = await h.manager.reserveOutgoingChallenge(against: "alice", perOpponentLimit: 1)
        XCTAssertTrue(other.reserved)
    }

    func testReservationCountsUntilReleasedOrSent() async throws {
        let h = try await makeHarness(provider: try await LichessBotFakeModelProvider.randomChampion()) { settings in
            settings.challenge.maxConcurrentGames = 4
            settings.challenge.maxSimultaneousGamesPerOpponent = 1
        }
        let run = Task { await h.manager.run() }
        defer { run.cancel() }
        try await waitUntil("the event stream is open") { await h.account.opens == 1 }

        let first = await h.manager.reserveOutgoingChallenge(against: "bob", perOpponentLimit: 1)
        XCTAssertEqual(first.committedBefore, 0)
        XCTAssertTrue(first.reserved)

        // An incoming challenge from the same player during the POST.
        await h.account.send(challengeLine(id: "c1", challenger: "bob"))
        try await waitUntil("the incoming challenge is declined") { await h.account.declined.count == 1 }
        XCTAssertEqual(declineRules(h.events.value), ["already playing this opponent"])

        let second = await h.manager.reserveOutgoingChallenge(against: "bob", perOpponentLimit: 1)
        XCTAssertEqual(second.committedBefore, 1)
        XCTAssertFalse(second.reserved)

        await h.manager.releaseOutgoingChallengeReservation(against: "bob")
        let afterRelease = await h.manager.reserveOutgoingChallenge(against: "bob", perOpponentLimit: 1)
        XCTAssertEqual(afterRelease.committedBefore, 0)
        XCTAssertTrue(afterRelease.reserved)

        // Sending turns the reservation into one pending challenge.
        await h.manager.noteSentChallenge(id: "sent", opponentID: "bob")
        let ids = await h.manager.outgoingChallengeIDs
        XCTAssertEqual(ids, ["sent"])
        let afterSend = await h.manager.reserveOutgoingChallenge(against: "bob", perOpponentLimit: 2)
        XCTAssertEqual(afterSend.committedBefore, 1, "counted once, not as both a reservation and a pending challenge")
    }

    // MARK: - Alarms

    @MainActor
    func testRepeatedAlarmIsOneRow() throws {
        let defaults = try makeTemporaryDefaults()
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("LichessBotGroupBFixTests-\(UUID().uuidString)", isDirectory: true)
        let controller = LichessBotController(
            modelProvider: LichessBotFakeModelProvider(snapshot: nil),
            defaults: defaults,
            dataDirectory: LichessBotDataDirectory(root: root)
        )
        addTeardownBlock { @MainActor in
            // Raising an alarm records a protocol event, appended on the
            // controller's file queue. Shutting down writes what is queued
            // and refuses anything later, so nothing races the removal.
            await controller.shutdown(reason: "test teardown")
            if FileManager.default.fileExists(atPath: root.path) {
                do {
                    try FileManager.default.removeItem(at: root)
                } catch {
                    XCTFail("cleanup failed: \(error)")
                }
            }
        }
        // Favorites aren't loaded, so each toggle raises the same alarm.
        controller.toggleFavorite("x")
        controller.toggleFavorite("x")
        XCTAssertEqual(controller.alarms.count, 1)
        XCTAssertEqual(controller.alarms.first?.repeatCount, 2)
    }
}
