import XCTest
@testable import DrewsChessMachine

/// An event stream scripted one connection at a time. When the script runs
/// out, opening fails with a closed gate, which stops the manager.
actor LichessBotFakeAccountAPI: LichessBotAccountAPI {
    enum Connection: Sendable {
        /// Send these lines, then keep the stream open until the test
        /// closes it.
        case open(lines: [String])
        /// Send these lines, then close the stream at once.
        case openThenClose(lines: [String])
        /// The connection attempt fails.
        case fail
    }

    private var script: [Connection]
    private var continuation: LichessBotChunkStream.Continuation?
    private(set) var opens = 0
    private(set) var accepted: [String] = []
    private(set) var declined: [(id: String, reason: LichessBotDeclineReason)] = []

    init(script: [Connection]) {
        self.script = script
    }

    func openEventStream() async throws -> LichessBotChunkStream {
        guard !script.isEmpty else {
            throw LichessBotGateError.closed(reason: "script exhausted")
        }
        let connection = script.removeFirst()
        let (stream, continuation) = LichessBotChunkStream.makeStream()
        switch connection {
        case .fail:
            throw URLError(.networkConnectionLost)
        case .open(let lines):
            opens += 1
            self.continuation = continuation
            for line in lines {
                continuation.yield(Data((line + "\n").utf8))
            }
        case .openThenClose(let lines):
            opens += 1
            for line in lines {
                continuation.yield(Data((line + "\n").utf8))
            }
            continuation.finish()
        }
        return stream
    }

    func send(_ line: String) {
        continuation?.yield(Data((line + "\n").utf8))
    }

    /// Send a line of an event type the manager doesn't know, named
    /// `marker`, after every line sent so far. The manager reads its event
    /// stream one line at a time, reports each line read, and finishes
    /// handling it before reading the next, so once this line is reported
    /// (`LichessBotManagerEvent.isEventStreamLine(containing:)`) every
    /// earlier line has been fully handled. The manager journals the unknown
    /// type as an anomaly and otherwise ignores it.
    func sendSentinel(_ marker: String) {
        send(#"{"type":"\#(marker)"}"#)
    }

    func closeCurrentStream() {
        continuation?.finish()
        continuation = nil
    }

    func acceptChallenge(id: String) async throws {
        accepted.append(id)
    }

    func declineChallenge(id: String, reason: LichessBotDeclineReason) async throws {
        declined.append((id, reason))
    }

    var declineReasons: [LichessBotDeclineReason] {
        declined.map(\.reason)
    }
}

extension LichessBotManagerEvent {
    /// This is the manager reporting that it read an event-stream line
    /// containing `marker`.
    func isEventStreamLine(containing marker: String) -> Bool {
        guard case .eventStreamLine(let data, _) = self else { return false }
        return String(decoding: data, as: UTF8.self).contains(marker)
    }
}

/// Champion weights from a real random-weight network, or none at all.
final class LichessBotFakeModelProvider: LichessBotModelProvider, @unchecked Sendable {
    private let snapshot: LichessBotWeightsSnapshot?
    let snapshotCount = SyncBox<Int>(0)

    init(snapshot: LichessBotWeightsSnapshot?) {
        self.snapshot = snapshot
    }

    static func randomChampion() async throws -> LichessBotFakeModelProvider {
        let network = try ChessMPSNetwork(.randomWeights)
        let weights = try await network.exportWeights()
        return LichessBotFakeModelProvider(snapshot: LichessBotWeightsSnapshot(
            weights: weights,
            architecture: network.arch,
            modelID: "20260928-1-TEST",
            trainingStep: nil
        ))
    }

    /// A champion that exists but whose weights can't build a network:
    /// enough for decisions that must never build one.
    static func unbuildableChampion() -> LichessBotFakeModelProvider {
        LichessBotFakeModelProvider(snapshot: LichessBotWeightsSnapshot(
            weights: [],
            architecture: .current,
            modelID: "20260928-1-TEST",
            trainingStep: nil
        ))
    }

    func championModelID() async -> String? {
        snapshot?.modelID
    }

    func championSnapshot() async throws -> LichessBotWeightsSnapshot {
        snapshotCount.modify { $0 += 1 }
        guard let snapshot else { throw LichessBotModelError.noChampion }
        return snapshot
    }

    func trainerAvailable() async -> Bool {
        false
    }

    func trainerSnapshot() async throws -> LichessBotWeightsSnapshot {
        throw LichessBotModelError.noTrainer
    }
}

/// `LichessBotSessionManager` — challenge responses, game sessions, gate
/// eligibility and takeover detection (plan §5.2, §6.1, §7, E15).
final class LichessBotSessionManagerTests: XCTestCase {

    private struct Harness {
        let account: LichessBotFakeAccountAPI
        let manager: LichessBotSessionManager
        let gate: LichessBotRequestGate
        let time: LichessBotManualTime
        let events: SyncBox<[LichessBotManagerEvent]>
        let observer: LichessBotRecordingGameObserver
    }

    private func makeHarness(
        script: [LichessBotFakeAccountAPI.Connection],
        provider: LichessBotFakeModelProvider = LichessBotFakeModelProvider(snapshot: nil),
        gameServer: LichessBotFakeGameServer? = nil,
        configure: (inout LichessBotSettings) -> Void = { _ in }
    ) throws -> Harness {
        var settings = LichessBotSettings.testBaseline()
        settings.chat.greetingEnabled = false
        configure(&settings)
        let frozen = settings
        let time = LichessBotManualTime()
        let account = LichessBotFakeAccountAPI(script: script)
        let gate = LichessBotRequestGate(time: time, breakerWindow: .seconds(3600)) { _ in }
        let slots = LichessBotModelSlots(provider: provider, time: time) { _ in }
        let events = SyncBox<[LichessBotManagerEvent]>([])
        let observer = LichessBotRecordingGameObserver()
        let manager = LichessBotSessionManager(
            accountAPI: account,
            gameAPI: try gameServer ?? LichessBotFakeGameServer(),
            gate: gate,
            slots: slots,
            ourAccountID: LichessBotFakeGameServer.botID,
            time: time,
            settingsProvider: { frozen },
            gameObserver: observer,
            onEvent: { event in events.modify { $0.append(event) } }
        )
        return Harness(account: account, manager: manager, gate: gate, time: time, events: events, observer: observer)
    }

    private func waitUntil(_ description: String, advancing time: LichessBotManualTime? = nil, by step: Duration = .seconds(2), _ condition: () async -> Bool) async throws {
        for _ in 0..<4000 {
            if await condition() { return }
            time?.advance(by: step)
            try await Task.sleep(for: .milliseconds(5))
        }
        XCTFail("timed out waiting until \(description)")
        throw CancellationError()
    }

    private func stopReason(_ h: Harness) -> String? {
        h.events.value.compactMap { event in
            if case .stopped(let reason) = event { return reason }
            return nil
        }.last
    }

    private func challengeLine(id: String, variant: String = "standard", rated: Bool = false, challenger: String = "alice", title: String? = nil) -> String {
        let titleField = title.map { #","title":"\#($0)""# } ?? ""
        return #"{"type":"challenge","challenge":{"id":"\#(id)","status":"created","challenger":{"id":"\#(challenger)","name":"\#(challenger)","rating":1500\#(titleField)},"variant":{"key":"\#(variant)","name":"x","short":"x"},"rated":\#(rated),"speed":"blitz","timeControl":{"type":"clock","limit":300,"increment":3,"show":"5+3"},"color":"random","direction":"in"},"compat":{"bot":true,"board":true}}"#
    }

    private func gameStartLine(gameID: String = LichessBotFakeGameServer.gameID, opponent: String = "alice") -> String {
        #"{"type":"gameStart","game":{"gameId":"\#(gameID)","color":"white","opponent":{"id":"\#(opponent)","username":"\#(opponent)","rating":1500},"isMyTurn":true}}"#
    }

    // MARK: - Challenges

    func testDeclinesWhatDCMCannotPlay() async throws {
        let provider = LichessBotFakeModelProvider.unbuildableChampion()
        let h = try makeHarness(
            script: [.openThenClose(lines: [
                challengeLine(id: "c1", variant: "chess960"),
                challengeLine(id: "c2", rated: true),
            ])],
            provider: provider
        )
        let run = Task { await h.manager.run() }
        try await waitUntil("the manager stops", advancing: h.time) { stopReason(h) != nil }
        await run.value
        let reasons = await h.account.declineReasons
        let accepted = await h.account.accepted
        XCTAssertEqual(reasons, [.standard, .casual])
        XCTAssertEqual(accepted, [])
        XCTAssertEqual(provider.snapshotCount.value, 0, "declines never build a network")
    }

    func testDeclinesLaterWithoutAModel() async throws {
        let h = try makeHarness(script: [.openThenClose(lines: [challengeLine(id: "c1")])])
        let run = Task { await h.manager.run() }
        try await waitUntil("the manager stops", advancing: h.time) { stopReason(h) != nil }
        await run.value
        let reasons = await h.account.declineReasons
        XCTAssertEqual(reasons, [.later])
    }

    func testDrainingDeclinesLater() async throws {
        let provider = LichessBotFakeModelProvider.unbuildableChampion()
        let h = try makeHarness(script: [.openThenClose(lines: [challengeLine(id: "c1")])], provider: provider)
        await h.manager.setAcceptingNewGames(false)
        let run = Task { await h.manager.run() }
        try await waitUntil("the manager stops", advancing: h.time) { stopReason(h) != nil }
        await run.value
        let reasons = await h.account.declineReasons
        XCTAssertEqual(reasons, [.later])
    }

    /// Past the per-minute budget, challenges are left unanswered rather
    /// than spending requests (plan §5.3).
    func testChallengeResponseBudget() async throws {
        let budget = LichessBotChallengeSettings.testBaseline().challengeResponseBudgetPerMinute
        let lines = (0...budget).map { challengeLine(id: "c\($0)", variant: "chess960") }
        let h = try makeHarness(script: [.open(lines: lines)])
        let run = Task { await h.manager.run() }
        try await waitUntil("every challenge is handled") {
            h.events.value.filter { event in
                if case .challengeDecision = event { return true }
                return false
            }.count == budget + 1
        }
        let responded = await h.account.declined.count
        XCTAssertEqual(responded, budget)
        run.cancel()
        await run.value
    }

    /// Accepting builds the model generation first (E15); the game that
    /// follows plays with that same generation, and its end is reported.
    func testAcceptsThenPlaysTheGame() async throws {
        let provider = try await LichessBotFakeModelProvider.randomChampion()
        let server = try LichessBotFakeGameServer()
        await server.setScript(afterOurMoves: [.finish(status: "resign", winner: "white")])
        let h = try makeHarness(
            script: [.open(lines: [challengeLine(id: "c1"), gameStartLine()])],
            provider: provider,
            gameServer: server
        )
        let run = Task { await h.manager.run() }
        try await waitUntil("the game session ends") {
            h.events.value.contains { event in
                if case .gameSessionEnded = event { return true }
                return false
            }
        }
        let accepted = await h.account.accepted
        let record = await server.record()
        XCTAssertEqual(accepted, ["c1"])
        XCTAssertEqual(record.acceptedPlies, [0])
        XCTAssertEqual(provider.snapshotCount.value, 1, "one snapshot serves both the acceptance and the game")
        XCTAssertEqual(h.observer.finishedStatus, "resign")
        let activeGames = await h.manager.activeGameCount
        XCTAssertEqual(activeGames, 0)
        let gate = await h.gate.snapshot()
        XCTAssertEqual(gate.gamesAwaitingOurMove, 0, "a finished game no longer holds housekeeping back")
        run.cancel()
        await run.value
    }

    /// The same game replayed on a reconnect does not start a second
    /// session.
    func testRepeatedGameStartStartsOneSession() async throws {
        let provider = try await LichessBotFakeModelProvider.randomChampion()
        let server = try LichessBotFakeGameServer()
        await server.setScript(afterOurMoves: [.opponentThinks])
        let h = try makeHarness(script: [.open(lines: [gameStartLine(), gameStartLine()])], provider: provider, gameServer: server)
        let run = Task { await h.manager.run() }
        try await waitUntil("the first move is posted") { await server.record().acceptedPlies == [0] }
        // Both gameStart lines were sent before the sentinel, so the
        // duplicate has been handled once the sentinel is read.
        await h.account.sendSentinel("sentinel-after-duplicate-game-start")
        try await waitUntil("the duplicate gameStart has been handled") {
            h.events.value.contains { $0.isEventStreamLine(containing: "sentinel-after-duplicate-game-start") }
        }
        let started = h.events.value.filter { event in
            if case .gameSessionStarted = event { return true }
            return false
        }.count
        XCTAssertEqual(started, 1)
        let record = await server.record()
        XCTAssertEqual(record.streamOpens, 1)
        await h.manager.abandonAllSessions()
        run.cancel()
        await run.value
    }

    /// E22: a `gameFinish` on the event stream ends a session whose game
    /// stream never delivered the final state.
    func testGameFinishEndsASessionWithoutAFinalState() async throws {
        let provider = try await LichessBotFakeModelProvider.randomChampion()
        let server = try LichessBotFakeGameServer()
        await server.setScript(afterOurMoves: [.opponentThinks])
        let h = try makeHarness(script: [.open(lines: [gameStartLine()])], provider: provider, gameServer: server)
        let run = Task { await h.manager.run() }
        try await waitUntil("the first move is posted") { await server.record().acceptedPlies == [0] }
        await server.endSilently(status: "resign", winner: "white")
        await h.account.send(#"{"type":"gameFinish","game":{"gameId":"\#(LichessBotFakeGameServer.gameID)","opponent":{"id":"alice"},"status":{"id":31,"name":"resign"}}}"#)
        try await waitUntil("the game session ends", advancing: h.time, by: .milliseconds(500)) {
            h.events.value.contains { event in
                if case .gameSessionEnded = event { return true }
                return false
            }
        }
        XCTAssertEqual(h.observer.finishedStatus, "resign")
        run.cancel()
        await run.value
    }

    // MARK: - Takeover (plan §6.1)

    func testRepeatedImmediateServerClosesAreATakeover() async throws {
        let h = try makeHarness(script: Array(repeating: .openThenClose(lines: []), count: 5))
        let run = Task { await h.manager.run() }
        try await waitUntil("the manager stops", advancing: h.time, by: .milliseconds(500)) { stopReason(h) != nil }
        await run.value
        let opens = await h.account.opens
        XCTAssertEqual(opens, LichessBotSessionManager.takeoverThreshold)
        XCTAssertTrue(h.events.value.contains { event in
            if case .takeoverSuspected = event { return true }
            return false
        })
    }

    func testFailedConnectsAreNotATakeover() async throws {
        let h = try makeHarness(script: Array(repeating: .fail, count: 5))
        let run = Task { await h.manager.run() }
        try await waitUntil("the manager stops", advancing: h.time, by: .seconds(10)) { stopReason(h) != nil }
        await run.value
        XCTAssertEqual(stopReason(h), "request gate closed: script exhausted")
    }

    /// A stream that lasted resets the count of short ones.
    func testALongStreamResetsTheTakeoverCount() async throws {
        let threshold = LichessBotSessionManager.takeoverThreshold
        let short = Array(repeating: LichessBotFakeAccountAPI.Connection.openThenClose(lines: []), count: threshold - 1)
        let h = try makeHarness(script: short + [.open(lines: [])] + short)
        let run = Task { await h.manager.run() }
        try await waitUntil("the long stream opens", advancing: h.time, by: .milliseconds(500)) { await h.account.opens == threshold }
        // Keep-alives hold the stream open past the takeover window.
        for _ in 0..<40 {
            await h.account.send("")
            h.time.advance(by: .seconds(1))
            try await Task.sleep(for: .milliseconds(2))
        }
        await h.account.closeCurrentStream()
        try await waitUntil("the manager stops", advancing: h.time, by: .milliseconds(500)) { stopReason(h) != nil }
        await run.value
        XCTAssertEqual(stopReason(h), "request gate closed: script exhausted")
    }
}
