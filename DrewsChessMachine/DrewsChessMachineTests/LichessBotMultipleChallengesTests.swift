import XCTest
@testable import DrewsChessMachine

/// Several outgoing challenges may be pending at once; each is resolved on
/// its own (plan §7.1).
final class LichessBotMultipleChallengesTests: XCTestCase {

    private func waitUntil(_ description: String, _ condition: () async -> Bool) async throws {
        for _ in 0..<4000 {
            if await condition() { return }
            try await Task.sleep(for: .milliseconds(5))
        }
        XCTFail("timed out waiting until \(description)")
        throw CancellationError()
    }

    func testEachPendingChallengeResolvesIndependently() async throws {
        let events = SyncBox<[LichessBotManagerEvent]>([])
        let time = LichessBotManualTime()
        var settings = LichessBotSettings.testBaseline()
        settings.chat.greetingEnabled = false
        let frozen = settings
        let account = LichessBotFakeAccountAPI(script: [.open(lines: [])])
        let manager = LichessBotSessionManager(
            accountAPI: account,
            gameAPI: try LichessBotFakeGameServer(),
            gate: LichessBotRequestGate(time: time, breakerWindow: .seconds(3600)) { _ in },
            slots: try await LichessBotModelSlots.prepare(for: LichessBotModelSettings.testBaseline(), provider: try await LichessBotFakeModelProvider.randomChampion(), time: time, folderScanner: LichessBotNoModelsFolderScanner(), log: { _ in }),
            ourAccountID: LichessBotFakeGameServer.botID,
            time: time,
            settingsProvider: { frozen },
            gameObserver: LichessBotRecordingGameObserver(),
            onEvent: { event in events.modify { $0.append(event) } },
            journalReader: LichessBotTestJournalReaders.none
        )
        let run = Task { await manager.run() }
        try await waitUntil("the stream opens") { await account.opens == 1 }

        await manager.noteOutgoingChallenge(id: "ch1")
        await manager.noteOutgoingChallenge(id: "ch2")
        let bothPending = await manager.outgoingChallengeIDs
        XCTAssertEqual(bothPending, ["ch1", "ch2"])

        await account.send(#"{"type":"challengeDeclined","challenge":{"id":"ch1","declineReasonKey":"later"}}"#)
        try await waitUntil("the decline is processed") { await manager.outgoingChallengeIDs == ["ch2"] }
        let afterDecline = await manager.outgoingChallengeIDs
        XCTAssertEqual(afterDecline, ["ch2"], "declining one leaves the other pending")

        await manager.clearOutgoingChallenge(id: "ch2")
        let afterClear = await manager.outgoingChallengeIDs
        XCTAssertEqual(afterClear, [])
        let resolved = events.value.compactMap { event -> String? in
            if case .outgoingChallengeResolved(let id, _) = event { return id }
            return nil
        }
        XCTAssertEqual(resolved, ["ch1"])
        run.cancel()
        await run.value
    }

    /// Regression (live, 2026-09-28: 8–9 games at once): pending outgoing
    /// challenges hold their slots, so an incoming challenge is declined
    /// when they already fill the concurrent-game limit.
    func testPendingOutgoingChallengesCountTowardTheLimit() async throws {
        let time = LichessBotManualTime()
        let events = SyncBox<[LichessBotManagerEvent]>([])
        var settings = LichessBotSettings.testBaseline()
        settings.chat.greetingEnabled = false
        settings.challenge.maxConcurrentGames = 1
        let frozen = settings
        let account = LichessBotFakeAccountAPI(script: [.open(lines: [])])
        let manager = LichessBotSessionManager(
            accountAPI: account,
            gameAPI: try LichessBotFakeGameServer(),
            gate: LichessBotRequestGate(time: time, breakerWindow: .seconds(3600)) { _ in },
            slots: try await LichessBotModelSlots.prepare(for: LichessBotModelSettings.testBaseline(), provider: try await LichessBotFakeModelProvider.randomChampion(), time: time, folderScanner: LichessBotNoModelsFolderScanner(), log: { _ in }),
            ourAccountID: LichessBotFakeGameServer.botID,
            time: time,
            settingsProvider: { frozen },
            gameObserver: LichessBotRecordingGameObserver(),
            onEvent: { event in events.modify { $0.append(event) } },
            journalReader: LichessBotTestJournalReaders.none
        )
        let run = Task { await manager.run() }
        try await waitUntil("the stream opens") { await account.opens == 1 }

        await manager.noteOutgoingChallenge(id: "mine")
        let challenge = #"{"type":"challenge","challenge":{"id":"c9","status":"created","challenger":{"id":"bob","name":"bob","rating":1500},"variant":{"key":"standard"},"rated":false,"speed":"blitz","timeControl":{"type":"clock","limit":300,"increment":3},"color":"random","direction":"in"},"compat":{"bot":true,"board":true}}"#
        await account.send(challenge)
        try await waitUntil("the challenge is answered") { await account.declined.count == 1 }
        let reasons = await account.declineReasons
        XCTAssertEqual(reasons, [.later])
        // `.later` is shared by several rules; pin this one.
        let rules = events.value.compactMap { event -> LichessBotChallengeDecision? in
            if case .challengeDecision("c9", _, let decision) = event { return decision }
            return nil
        }
        XCTAssertEqual(rules, [.decline(.later, rule: "at the concurrent-game limit")])
        run.cancel()
        await run.value
    }
}
