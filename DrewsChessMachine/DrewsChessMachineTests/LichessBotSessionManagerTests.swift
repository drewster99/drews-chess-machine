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
    /// Challenges whose acceptance Lichess refuses, with the error the
    /// client throws for it. Still listed in `accepted`: the request was
    /// sent.
    private var acceptRefusals: [String: LichessBotAPIError] = [:]
    /// Errors the next attempts to accept or decline a challenge throw, in
    /// order; once used up, attempts succeed. Every attempt is still listed
    /// in `accepted` / `declined`: the request was sent.
    private var acceptFailures: [String: [any Error]] = [:]
    private var declineFailures: [String: [any Error]] = [:]
    /// Challenges whose next acceptance (after any queued failures) stays
    /// in flight until the test releases it.
    private var acceptHolds: Set<String> = []
    private var heldAcceptances: [String: CheckedContinuation<Void, any Error>] = [:]

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

    func refuseAcceptance(of id: String, with error: LichessBotAPIError) {
        acceptRefusals[id] = error
    }

    func failAcceptance(of id: String, with errors: [any Error]) {
        acceptFailures[id, default: []].append(contentsOf: errors)
    }

    func failDecline(of id: String, with errors: [any Error]) {
        declineFailures[id, default: []].append(contentsOf: errors)
    }

    func holdNextAcceptance(of id: String) {
        acceptHolds.insert(id)
    }

    func isHoldingAcceptance(of id: String) -> Bool {
        heldAcceptances[id] != nil
    }

    func releaseHeldAcceptance(of id: String, failingWith error: any Error) {
        heldAcceptances.removeValue(forKey: id)?.resume(throwing: error)
    }

    func acceptChallenge(id: String) async throws {
        accepted.append(id)
        if let refusal = acceptRefusals[id] {
            throw refusal
        }
        if let failure = acceptFailures[id]?.first {
            acceptFailures[id]?.removeFirst()
            throw failure
        }
        if acceptHolds.remove(id) != nil {
            try await withCheckedThrowingContinuation { heldAcceptances[id] = $0 }
        }
    }

    func declineChallenge(id: String, reason: LichessBotDeclineReason) async throws {
        declined.append((id, reason))
        if let failure = declineFailures[id]?.first {
            declineFailures[id]?.removeFirst()
            throw failure
        }
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
        let network = try ChessMPSNetwork(.randomWeights(initSeed: 1))
        let weights = try await network.exportWeights()
        return LichessBotFakeModelProvider(snapshot: LichessBotWeightsSnapshot(
            weights: weights,
            architecture: network.arch,
            modelID: "20260928-1-TEST",
            trainingStep: nil,
            trainingHistory: .unknown
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
        provider: LichessBotFakeModelProvider,
        gameServer: LichessBotFakeGameServer? = nil,
        configure: (inout LichessBotSettings) -> Void = { _ in }
    ) async throws -> Harness {
        var settings = LichessBotSettings.testBaseline()
        settings.chat.greetingEnabled = false
        configure(&settings)
        let frozen = settings
        let time = LichessBotManualTime()
        let account = LichessBotFakeAccountAPI(script: script)
        let gate = LichessBotRequestGate(time: time, rateLimitPolicy: .testPolicy) { _ in }
        let slots = try await LichessBotModelSlots.prepare(for: settings.model, provider: provider, time: time, folderScanner: LichessBotNoModelsFolderScanner(), log: { _ in })
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
            onEvent: { event in events.modify { $0.append(event) } },
            journalReader: LichessBotTestJournalReaders.none
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
        let provider = try await LichessBotFakeModelProvider.randomChampion()
        let h = try await makeHarness(
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
        XCTAssertEqual(provider.snapshotCount.value, 1, "declines build nothing beyond the generation prepared before going online")
    }

    func testDrainingDeclinesLater() async throws {
        let provider = try await LichessBotFakeModelProvider.randomChampion()
        let h = try await makeHarness(script: [.openThenClose(lines: [challengeLine(id: "c1")])], provider: provider)
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
        let h = try await makeHarness(script: [.open(lines: lines)], provider: try await LichessBotFakeModelProvider.randomChampion())
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

    /// The game that follows an acceptance plays with the generation
    /// prepared before going online (E15), and its end is reported.
    func testAcceptsThenPlaysTheGame() async throws {
        let provider = try await LichessBotFakeModelProvider.randomChampion()
        let server = try LichessBotFakeGameServer()
        await server.setScript(afterOurMoves: [.finish(status: "resign", winner: "white")])
        let h = try await makeHarness(
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
        let h = try await makeHarness(script: [.open(lines: [gameStartLine(), gameStartLine()])], provider: provider, gameServer: server)
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
        let h = try await makeHarness(script: [.open(lines: [gameStartLine()])], provider: provider, gameServer: server)
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
        let h = try await makeHarness(script: Array(repeating: .openThenClose(lines: []), count: 5), provider: try await LichessBotFakeModelProvider.randomChampion())
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
        let h = try await makeHarness(script: Array(repeating: .fail, count: 5), provider: try await LichessBotFakeModelProvider.randomChampion())
        let run = Task { await h.manager.run() }
        try await waitUntil("the manager stops", advancing: h.time, by: .seconds(10)) { stopReason(h) != nil }
        await run.value
        XCTAssertEqual(stopReason(h), "request gate closed: script exhausted")
    }

    /// A stream that lasted resets the count of short ones.
    func testALongStreamResetsTheTakeoverCount() async throws {
        let threshold = LichessBotSessionManager.takeoverThreshold
        let short = Array(repeating: LichessBotFakeAccountAPI.Connection.openThenClose(lines: []), count: threshold - 1)
        let h = try await makeHarness(script: short + [.open(lines: [])] + short, provider: try await LichessBotFakeModelProvider.randomChampion())
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

    // MARK: - Challenge responses that got no answer

    /// What URLSession throws when Lichess never answered (2026-10-26:
    /// `NSURLErrorDomain -1001`, POSIX 60 underneath).
    private static let noAnswer = URLError(.timedOut)

    private func responseFailures(_ h: Harness) -> [String] {
        h.events.value.compactMap { event in
            if case .challengeResponseFailed(let challengeID, _) = event { return challengeID }
            return nil
        }
    }

    private func challengeCanceledLine(id: String) -> String {
        #"{"type":"challengeCanceled","challenge":{"id":"\#(id)","status":"canceled"}}"#
    }

    /// Let the manager's clock run on, a second at a time.
    private func advance(_ h: Harness, seconds: Int) async throws {
        for _ in 0..<seconds {
            h.time.advance(by: .seconds(1))
            try await Task.sleep(for: .milliseconds(3))
        }
    }

    private func waitForEventLine(_ h: Harness, containing marker: String) async throws {
        await h.account.sendSentinel(marker)
        try await waitUntil("the manager reads \(marker)") {
            h.events.value.contains { $0.isEventStreamLine(containing: marker) }
        }
    }

    /// Protects: an acceptance Lichess never answered is sent again while
    /// the challenge is still open.
    /// Fails if: a failed acceptance is reported and dropped, leaving the
    /// challenger waiting (2026-10-26, GeoModPhreak's NbJuXlWI).
    func testAcceptanceThatGotNoAnswerIsRetried() async throws {
        let h = try await makeHarness(script: [.open(lines: [challengeLine(id: "c1")])], provider: try await LichessBotFakeModelProvider.randomChampion())
        await h.account.failAcceptance(of: "c1", with: [Self.noAnswer])
        let run = Task { await h.manager.run() }
        try await waitUntil("the acceptance is retried", advancing: h.time, by: .seconds(1)) { await h.account.accepted.count == 2 }
        try await advance(h, seconds: 30)
        let accepted = await h.account.accepted
        XCTAssertEqual(accepted, ["c1", "c1"])
        XCTAssertEqual(responseFailures(h), [], "an acceptance that went through on a retry is no failure")
        run.cancel()
        await run.value
    }

    /// Protects: retries are bounded, and the failure is reported once,
    /// after the last attempt.
    /// Fails if: the retries never stop, or each failed attempt is reported
    /// as the challenge's failure.
    func testAcceptanceRetriesStopAtTheLimitAndReportOnce() async throws {
        let h = try await makeHarness(script: [.open(lines: [challengeLine(id: "c1")])], provider: try await LichessBotFakeModelProvider.randomChampion())
        await h.account.failAcceptance(of: "c1", with: Array(repeating: Self.noAnswer, count: 10))
        let run = Task { await h.manager.run() }
        try await waitUntil("the failure is reported", advancing: h.time, by: .seconds(1)) { !responseFailures(h).isEmpty }
        try await advance(h, seconds: 60)
        let accepted = await h.account.accepted
        XCTAssertEqual(accepted.count, 1 + LichessBotSessionManager.challengeResponseRetryDelays.count)
        XCTAssertEqual(responseFailures(h), ["c1"])
        run.cancel()
        await run.value
    }

    /// Protects: a refusal Lichess sent (an HTTP status) is final.
    /// Fails if: every failure, answered or not, is retried.
    func testAcceptanceLichessRefusedIsNotRetried() async throws {
        let h = try await makeHarness(script: [.open(lines: [challengeLine(id: "c1")])], provider: try await LichessBotFakeModelProvider.randomChampion())
        await h.account.failAcceptance(of: "c1", with: [LichessBotAPIError.http(status: 400, message: "challenge not found")])
        let run = Task { await h.manager.run() }
        try await waitUntil("the failure is reported") { !responseFailures(h).isEmpty }
        try await advance(h, seconds: 60)
        let accepted = await h.account.accepted
        XCTAssertEqual(accepted, ["c1"])
        XCTAssertEqual(responseFailures(h), ["c1"])
        run.cancel()
        await run.value
    }

    /// Protects: a challenge canceled while its retry waits is not accepted
    /// again.
    /// Fails if: the retry doesn't check that the challenge is still open.
    func testAcceptanceRetryStopsWhenTheChallengeIsCanceled() async throws {
        let h = try await makeHarness(script: [.open(lines: [challengeLine(id: "c1")])], provider: try await LichessBotFakeModelProvider.randomChampion())
        await h.account.failAcceptance(of: "c1", with: [Self.noAnswer])
        let run = Task { await h.manager.run() }
        try await waitUntil("the first attempt is made") { await h.account.accepted.count == 1 }
        await h.account.send(challengeCanceledLine(id: "c1"))
        try await waitForEventLine(h, containing: "after-cancel")
        try await advance(h, seconds: 60)
        let accepted = await h.account.accepted
        XCTAssertEqual(accepted, ["c1"])
        run.cancel()
        await run.value
    }

    /// Protects: an acceptance that did get through (its `gameStart`
    /// arrived) is not sent again.
    /// Fails if: the retry ignores a game that started meanwhile.
    func testAcceptanceRetryStopsWhenTheGameStarts() async throws {
        let server = try LichessBotFakeGameServer()
        await server.setScript(afterOurMoves: [.finish(status: "resign", winner: "white")])
        let gameID = LichessBotFakeGameServer.gameID
        let h = try await makeHarness(script: [.open(lines: [challengeLine(id: gameID)])], provider: try await LichessBotFakeModelProvider.randomChampion(), gameServer: server)
        await h.account.failAcceptance(of: gameID, with: [Self.noAnswer])
        let run = Task { await h.manager.run() }
        try await waitUntil("the first attempt is made") { await h.account.accepted.count == 1 }
        await h.account.send(gameStartLine())
        try await waitUntil("the game session ends") {
            h.events.value.contains { event in
                if case .gameSessionEnded = event { return true }
                return false
            }
        }
        try await advance(h, seconds: 60)
        let accepted = await h.account.accepted
        XCTAssertEqual(accepted, [gameID])
        XCTAssertEqual(responseFailures(h), [])
        run.cancel()
        await run.value
    }

    /// Protects: a bot that started draining while an acceptance waited to
    /// be retried takes no new game; the challenge is declined `later`.
    /// Fails if: the retry accepts regardless, or drops the challenge
    /// unanswered.
    func testAcceptanceRetryWhileDrainingDeclinesLater() async throws {
        let h = try await makeHarness(script: [.open(lines: [challengeLine(id: "c1")])], provider: try await LichessBotFakeModelProvider.randomChampion())
        await h.account.failAcceptance(of: "c1", with: [Self.noAnswer])
        let run = Task { await h.manager.run() }
        try await waitUntil("the first attempt is made") { await h.account.accepted.count == 1 }
        await h.manager.setAcceptingNewGames(false)
        try await waitUntil("the retry declines", advancing: h.time, by: .seconds(1)) { await h.account.declined.count == 1 }
        try await advance(h, seconds: 30)
        let accepted = await h.account.accepted
        let reasons = await h.account.declineReasons
        XCTAssertEqual(accepted, ["c1"])
        XCTAssertEqual(reasons, [.later])
        run.cancel()
        await run.value
    }

    /// Protects: the same challenge arriving again on the stream (a
    /// reconnect re-sends open challenges) while its acceptance waits to be
    /// retried gets no second decision of its own.
    /// Fails if: the repeat is decided afresh, sending another acceptance
    /// (or, at capacity, a decline of the challenge being accepted) beside
    /// the retry.
    func testChallengeRepeatedWhileItsAcceptanceWaitsIsNotAnsweredTwice() async throws {
        let h = try await makeHarness(script: [.open(lines: [challengeLine(id: "c1")])], provider: try await LichessBotFakeModelProvider.randomChampion())
        await h.account.failAcceptance(of: "c1", with: [Self.noAnswer])
        let run = Task { await h.manager.run() }
        try await waitUntil("the first attempt is made") { await h.account.accepted.count == 1 }
        await h.account.send(challengeLine(id: "c1"))
        try await waitForEventLine(h, containing: "after-repeat")
        try await advance(h, seconds: 30)
        let accepted = await h.account.accepted
        let declined = await h.account.declined.count
        XCTAssertEqual(accepted, ["c1", "c1"], "the first attempt and its one retry")
        XCTAssertEqual(declined, 0)
        run.cancel()
        await run.value
    }

    /// Protects: a retry whose request is in flight when the run ends sends
    /// nothing more for the runtime that has gone; its failure is final.
    /// Fails if: the retry, out of the run's reach while its request is
    /// out, schedules another attempt (an acceptance for a bot that went
    /// offline).
    func testRetryInFlightWhenTheRunEndsSendsNoMore() async throws {
        let h = try await makeHarness(script: [.open(lines: [challengeLine(id: "c1")])], provider: try await LichessBotFakeModelProvider.randomChampion())
        await h.account.failAcceptance(of: "c1", with: [Self.noAnswer])
        await h.account.holdNextAcceptance(of: "c1")
        let run = Task { await h.manager.run() }
        try await waitUntil("the retry is in flight", advancing: h.time, by: .seconds(1)) { await h.account.isHoldingAcceptance(of: "c1") }
        run.cancel()
        await run.value
        await h.account.releaseHeldAcceptance(of: "c1", failingWith: Self.noAnswer)
        try await waitUntil("the retry's failure is reported") { responseFailures(h) == ["c1"] }
        try await advance(h, seconds: 30)
        let accepted = await h.account.accepted
        XCTAssertEqual(accepted, ["c1", "c1"])
    }

    /// Protects: a decline Lichess never answered is sent again.
    /// Fails if: a failed decline is reported and dropped.
    func testDeclineThatGotNoAnswerIsRetried() async throws {
        let h = try await makeHarness(script: [.open(lines: [challengeLine(id: "c1")])], provider: try await LichessBotFakeModelProvider.randomChampion())
        await h.account.failDecline(of: "c1", with: [Self.noAnswer])
        await h.manager.setAcceptingNewGames(false)
        let run = Task { await h.manager.run() }
        try await waitUntil("the decline is retried", advancing: h.time, by: .seconds(1)) { await h.account.declined.count == 2 }
        try await advance(h, seconds: 30)
        let reasons = await h.account.declineReasons
        XCTAssertEqual(reasons, [.later, .later])
        XCTAssertEqual(responseFailures(h), [])
        run.cancel()
        await run.value
    }

    /// Protects: a challenge canceled while its decline waits to be retried
    /// is not declined again.
    /// Fails if: the decline retry doesn't check that the challenge is
    /// still open.
    func testDeclineRetryStopsWhenTheChallengeIsCanceled() async throws {
        let h = try await makeHarness(script: [.open(lines: [challengeLine(id: "c1")])], provider: try await LichessBotFakeModelProvider.randomChampion())
        await h.account.failDecline(of: "c1", with: [Self.noAnswer])
        await h.manager.setAcceptingNewGames(false)
        let run = Task { await h.manager.run() }
        try await waitUntil("the first attempt is made") { await h.account.declined.count == 1 }
        await h.account.send(challengeCanceledLine(id: "c1"))
        try await waitForEventLine(h, containing: "after-cancel")
        try await advance(h, seconds: 60)
        let declined = await h.account.declined.count
        XCTAssertEqual(declined, 1)
        run.cancel()
        await run.value
    }
}
