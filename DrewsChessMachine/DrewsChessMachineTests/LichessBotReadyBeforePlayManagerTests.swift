import XCTest
@testable import DrewsChessMachine

/// The session manager never waits on, builds or declines for the model:
/// the slots always hold a generation (built before going online), a
/// challenge is answered from the policy alone, and a game starts on the
/// current generation even while a source switch is building (follow-lineage
/// plan §3.10, OD-18, OD-19).
final class LichessBotReadyBeforePlayManagerTests: XCTestCase {

    private struct Harness {
        let account: LichessBotFakeAccountAPI
        let manager: LichessBotSessionManager
        let slots: LichessBotModelSlots
        let events: SyncBox<[LichessBotManagerEvent]>
        let server: LichessBotFakeGameServer
    }

    private func makeHarness(provider: LichessBotHoldableModelProvider, lines: [String] = []) async throws -> Harness {
        var settings = LichessBotSettings.testBaseline()
        settings.chat.greetingEnabled = false
        let frozen = settings
        let time = LichessBotManualTime()
        let account = LichessBotFakeAccountAPI(script: [.open(lines: lines)])
        let slots = try await LichessBotModelSlots.prepare(for: frozen.model, provider: provider, time: time, log: { _ in })
        let events = SyncBox<[LichessBotManagerEvent]>([])
        let server = try LichessBotFakeGameServer()
        await server.setScript(afterOurMoves: [.opponentThinks])
        let manager = LichessBotSessionManager(
            accountAPI: account,
            gameAPI: server,
            gate: LichessBotRequestGate(time: time, breakerWindow: .seconds(3600)) { _ in },
            slots: slots,
            ourAccountID: LichessBotFakeGameServer.botID,
            time: time,
            settingsProvider: { frozen },
            gameObserver: LichessBotRecordingGameObserver(),
            onEvent: { event in events.modify { $0.append(event) } },
            journalReader: LichessBotTestJournalReaders.none
        )
        return Harness(account: account, manager: manager, slots: slots, events: events, server: server)
    }

    private func waitUntil(_ description: String, _ condition: () async -> Bool) async throws {
        for _ in 0..<4000 {
            if await condition() { return }
            try await Task.sleep(for: .milliseconds(5))
        }
        XCTFail("timed out waiting until \(description)")
        throw CancellationError()
    }

    private func challengeLine(id: String, challenger: String, variant: String = "standard") -> String {
        #"{"type":"challenge","challenge":{"id":"\#(id)","status":"created","challenger":{"id":"\#(challenger)","name":"\#(challenger)","rating":1500},"variant":{"key":"\#(variant)","name":"x","short":"x"},"rated":false,"speed":"blitz","timeControl":{"type":"clock","limit":300,"increment":3,"show":"5+3"},"color":"random","direction":"in"},"compat":{"bot":true,"board":true}}"#
    }

    private func decisions(_ h: Harness) -> [(id: String, decision: LichessBotChallengeDecision)] {
        h.events.value.compactMap { event in
            if case .challengeDecision(let id, _, let decision) = event { return (id, decision) }
            return nil
        }
    }

    private func anomalies(_ h: Harness) -> [String] {
        h.events.value.compactMap { event in
            if case .anomaly(let text) = event { return text }
            return nil
        }
    }

    /// Starts a switch to the live trainer whose snapshot is held, and waits
    /// until it is building.
    private func startHeldSwitch(_ h: Harness, provider: LichessBotHoldableModelProvider) async throws -> Task<Void, Error> {
        provider.holdSnapshots()
        var trainer = LichessBotModelSettings.testBaseline()
        trainer.source = .liveTrainer
        let frozenTrainer = trainer
        let slots = h.slots
        let switching = Task {
            try await slots.refreshIfDue(for: frozenTrainer)
        }
        try await waitUntil("the switch's snapshot is held") { provider.snapshotsHeld.value == 1 }
        return switching
    }

    func testChallengeDuringASourceSwitchIsAcceptedOnTheCurrentGeneration() async throws {
        let provider = try await LichessBotHoldableModelProvider.make()
        let h = try await makeHarness(provider: provider)
        let run = Task { await h.manager.run() }
        try await waitUntil("the event stream is open") { await h.account.opens == 1 }
        let switching = try await startHeldSwitch(h, provider: provider)

        await h.account.send(challengeLine(id: "c1", challenger: "alice"))
        try await waitUntil("the challenge is answered") {
            let accepted = await h.account.accepted
            let declined = await h.account.declined
            return accepted.count + declined.count == 1
        }
        let accepted = await h.account.accepted
        XCTAssertEqual(accepted, ["c1"], "accepted while the switch is still building")
        XCTAssertEqual(provider.snapshotsHeld.value, 1, "the switch was still held when the challenge was accepted")
        XCTAssertEqual(provider.championSnapshots.value, 1, "the acceptance built nothing: the one champion snapshot is the prepared one")
        let playing = await h.slots.current.info
        XCTAssertEqual(playing.generationID, 1)

        provider.releaseSnapshots()
        try await switching.value
        run.cancel()
        await run.value
    }

    func testGameStartDuringASourceSwitchPlaysTheCurrentGeneration() async throws {
        let provider = try await LichessBotHoldableModelProvider.make()
        let h = try await makeHarness(provider: provider)
        let run = Task { await h.manager.run() }
        try await waitUntil("the event stream is open") { await h.account.opens == 1 }
        let switching = try await startHeldSwitch(h, provider: provider)

        await h.account.send(#"{"type":"gameStart","game":{"gameId":"\#(LichessBotFakeGameServer.gameID)","color":"white","opponent":{"id":"alice","username":"alice","rating":1500},"isMyTurn":true}}"#)
        try await waitUntil("the game session starts") {
            h.events.value.contains { event in
                if case .gameSessionStarted = event { return true }
                return false
            }
        }
        let started = h.events.value.compactMap { event -> LichessBotGenerationInfo? in
            if case .gameSessionStarted(_, let generation, _) = event { return generation }
            return nil
        }
        XCTAssertEqual(started.map(\.generationID), [1], "the game plays the generation that exists, not the one building")
        XCTAssertEqual(started.map(\.sourceKind), [.champion])
        XCTAssertEqual(provider.snapshotsHeld.value, 1, "the switch was still held when the game started")
        try await waitUntil("the first move is posted") { await h.server.record().acceptedPlies == [0] }

        await h.manager.abandonAllSessions()
        provider.releaseSnapshots()
        try await switching.value
        run.cancel()
        await run.value
    }

    func testNoChallengeIsEverDeclinedForTheModel() async throws {
        let provider = try await LichessBotHoldableModelProvider.make()
        let h = try await makeHarness(provider: provider)
        let run = Task { await h.manager.run() }
        try await waitUntil("the event stream is open") { await h.account.opens == 1 }

        // Every switch to the live trainer fails: there is no trainer.
        provider.trainerExists.value = false
        var trainer = LichessBotModelSettings.testBaseline()
        trainer.source = .liveTrainer
        let challenges = [
            challengeLine(id: "c1", challenger: "alice"),
            challengeLine(id: "c2", challenger: "bob", variant: "chess960"),
            challengeLine(id: "c3", challenger: "carol"),
            challengeLine(id: "c4", challenger: "dave"),
        ]
        for (index, line) in challenges.enumerated() {
            do {
                try await h.slots.refreshIfDue(for: trainer)
                XCTFail("the switch must fail without a trainer")
            } catch {
                XCTAssertEqual(error as? LichessBotModelError, .noTrainer)
            }
            await h.account.send(line)
            try await waitUntil("challenge \(index + 1) is decided") { decisions(h).count == index + 1 }
        }
        try await waitUntil("every challenge is answered") {
            let accepted = await h.account.accepted
            let declined = await h.account.declined
            return accepted.count + declined.count == challenges.count
        }
        let accepted = await h.account.accepted
        XCTAssertEqual(accepted, ["c1", "c3"], "playable challenges are accepted while the switch keeps failing")
        let rules = decisions(h).compactMap { entry -> String? in
            if case .decline(_, let rule) = entry.decision { return rule }
            return nil
        }
        XCTAssertEqual(rules.count, 2, "the chess960 challenge and the one past the concurrent-game limit: \(rules)")
        XCTAssertFalse(rules.contains { $0.localizedCaseInsensitiveContains("model") }, "\(rules)")
        XCTAssertFalse(anomalies(h).contains { $0.localizedCaseInsensitiveContains("model") }, "\(anomalies(h))")
        let playing = await h.slots.current.info
        XCTAssertEqual(playing.sourceKind, .champion, "the failing switch kept the champion playing")
        run.cancel()
        await run.value
    }
}
