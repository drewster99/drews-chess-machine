import XCTest
@testable import DrewsChessMachine

/// Wraps `LichessBotFakeGameServer` to fail or hold chosen calls and to add
/// raw lines to the open game stream.
actor LichessBotFaultyGameAPI: LichessBotGameAPI {
    let server: LichessBotFakeGameServer
    private var failures: [String: [any Error]] = [:]
    private var armedHolds: Set<String> = []
    private var heldCalls: [String: CheckedContinuation<Void, Never>] = [:]
    private(set) var attempts: [String] = []
    private var injector: LichessBotChunkStream.Continuation?
    private var forwarding: Task<Void, Never>?

    init(server: LichessBotFakeGameServer) {
        self.server = server
    }

    /// The next calls named `call` throw these errors, one each, in order.
    func fail(_ call: String, with errors: [any Error]) {
        failures[call, default: []].append(contentsOf: errors)
    }

    /// The next call named `call` waits for `release(_:)` before going on.
    func hold(_ call: String) {
        armedHolds.insert(call)
    }

    func isHolding(_ call: String) -> Bool {
        heldCalls[call] != nil
    }

    func release(_ call: String) {
        heldCalls.removeValue(forKey: call)?.resume()
    }

    /// Add `line` to the open game stream, as if Lichess had sent it.
    func inject(_ line: String) {
        injector?.yield(Data((line + "\n").utf8))
    }

    private func intercept(_ call: String) async throws {
        attempts.append(call)
        if armedHolds.remove(call) != nil {
            await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
                heldCalls[call] = continuation
            }
        }
        if let error = failures[call]?.first {
            failures[call]?.removeFirst()
            throw error
        }
    }

    func openGameStream(gameID: String) async throws -> LichessBotChunkStream {
        try await intercept("openGameStream")
        let inner = try await server.openGameStream(gameID: gameID)
        forwarding?.cancel()
        injector?.finish()
        let (stream, continuation) = LichessBotChunkStream.makeStream()
        injector = continuation
        forwarding = Task {
            do {
                for try await chunk in inner {
                    continuation.yield(chunk)
                }
                continuation.finish()
            } catch {
                continuation.finish(throwing: error)
            }
        }
        return stream
    }

    func makeMove(gameID: String, uci: String, offeringDraw: Bool) async throws {
        try await intercept("makeMove")
        try await server.makeMove(gameID: gameID, uci: uci, offeringDraw: offeringDraw)
    }

    func respondToDraw(gameID: String, accept: Bool) async throws {
        try await intercept("respondToDraw")
        try await server.respondToDraw(gameID: gameID, accept: accept)
    }

    func respondToTakeback(gameID: String, accept: Bool) async throws {
        try await intercept("respondToTakeback")
        try await server.respondToTakeback(gameID: gameID, accept: accept)
    }

    func resign(gameID: String) async throws {
        try await intercept("resign")
        try await server.resign(gameID: gameID)
    }

    func abort(gameID: String) async throws {
        try await intercept("abort")
        try await server.abort(gameID: gameID)
    }

    func claimVictory(gameID: String) async throws {
        try await intercept("claimVictory")
        try await server.claimVictory(gameID: gameID)
    }

    func claimDraw(gameID: String) async throws {
        try await intercept("claimDraw")
        try await server.claimDraw(gameID: gameID)
    }

    func chat(gameID: String, room: LichessBotChatRoom, text: String) async throws {
        try await intercept("chat")
        try await server.chat(gameID: gameID, room: room, text: text)
    }
}

/// A move source with a chosen generation identity (the scripted source's
/// is fixed to champion generation 1). Plays the first legal move in UCI
/// order.
final class LichessBotGenerationMoveSource: LichessBotMoveSource, @unchecked Sendable {
    let info: LichessBotGenerationInfo
    let decidedPlies = SyncBox<[Int]>([])

    init(generationID: Int, sourceKind: LichessBotModelSourceKind, modelID: String) {
        info = LichessBotGenerationInfo(
            generationID: generationID,
            sourceKind: sourceKind,
            modelID: modelID,
            trainingStep: nil,
            snapshotAt: Date(timeIntervalSince1970: 0),
            architectureSummary: "test",
            filePath: nil,
            fileSHA256: nil
        )
    }

    func decide(_ request: LichessBotMoveRequest, schedule: SamplingSchedule) async throws -> LichessBotMoveDecision {
        decidedPlies.modify { $0.append(request.ply) }
        guard let uci = request.legalMoves.map(\.uci).sorted().first else {
            throw LichessBotMoveChooserError.noLegalMoves
        }
        return LichessBotMoveDecision(
            uci: uci, san: uci, chosenProbability: 1, topMoves: [],
            win: 0.3, draw: 0.4, loss: 0.3,
            temperature: schedule.floorTau, legalMoveCount: request.legalMoves.count, randomish: false,
            encodeMilliseconds: 0, inferenceMilliseconds: 0, sampleMilliseconds: 0
        )
    }
}

/// `LichessBotGameSession` when requests fail or wait: unplayable games,
/// victory claims, optional actions, command replies, the resign streak,
/// held moves, mid-game refresh and foreign moves.
final class LichessBotGameSessionFaultTests: XCTestCase {

    private struct Harness {
        let server: LichessBotFakeGameServer
        let api: LichessBotFaultyGameAPI
        let observer: LichessBotRecordingGameObserver
        let time: LichessBotManualTime
        let session: LichessBotGameSession
    }

    private func makeHarness(
        server: LichessBotFakeGameServer,
        source: any LichessBotMoveSource = LichessBotScriptedMoveSource(),
        latestMoveSource: @escaping @Sendable () async -> (any LichessBotMoveSource)? = { nil },
        pacing: SyncBox<LichessBotMovePacingSnapshot> = SyncBox(LichessBotMovePacingSnapshot()),
        configure: (inout LichessBotSettings) -> Void = { _ in }
    ) -> Harness {
        var settings = LichessBotSettings()
        settings.chat.greetingEnabled = false
        settings.chat.goodbyeEnabled = false
        configure(&settings)
        let frozen = settings
        let observer = LichessBotRecordingGameObserver()
        let time = LichessBotManualTime()
        let api = LichessBotFaultyGameAPI(server: server)
        let session = LichessBotGameSession(
            gameID: LichessBotFakeGameServer.gameID,
            ourAccountID: LichessBotFakeGameServer.botID,
            api: api,
            moveSource: source,
            latestMoveSource: latestMoveSource,
            settingsProvider: { frozen },
            observer: observer,
            time: time,
            onTurnStatus: { _, status in observer.turnStatuses.modify { $0.append(status) } },
            pacing: { pacing.value }
        )
        return Harness(server: server, api: api, observer: observer, time: time, session: session)
    }

    /// Poll `condition`, optionally advancing the manual clock each time.
    private func waitUntil(_ description: String, advancing time: LichessBotManualTime? = nil, by step: Duration = .seconds(2), _ condition: () async -> Bool) async throws {
        for _ in 0..<2000 {
            if await condition() { return }
            time?.advance(by: step)
            try await Task.sleep(for: .milliseconds(5))
        }
        XCTFail("timed out waiting until \(description)")
        throw CancellationError()
    }

    private func heldPlies(_ observer: LichessBotRecordingGameObserver) -> [Int] {
        observer.events.value.compactMap { event in
            if case .moveHeld(let ply, _, _) = event { return ply }
            return nil
        }
    }

    private func stopReasons(_ observer: LichessBotRecordingGameObserver) -> [String] {
        observer.events.value.compactMap { event in
            if case .stoppedMoving(let reason) = event { return reason }
            return nil
        }
    }

    // MARK: - A1: unplayable games

    /// An abort that got no answer is tried again on the next `gameFull`,
    /// instead of the game being left to run out of time.
    func testUnplayableAbortIsRetriedAfterATransportFailure() async throws {
        let server = try LichessBotFakeGameServer(variantKey: "chess960")
        let h = makeHarness(server: server)
        await h.api.fail("abort", with: [URLError(.networkConnectionLost)])
        let run = Task { await h.session.run() }
        try await waitUntil("the game ends", advancing: h.time) { h.observer.finishedStatus != nil }
        await run.value
        let attempts = await h.api.attempts
        let record = await server.record()
        XCTAssertEqual(attempts.filter { $0 == "abort" }.count, 2)
        XCTAssertEqual(record.calls, ["abort"])
        XCTAssertEqual(h.observer.finishedStatus, "aborted")
    }

    /// A 5xx is Lichess failing, not refusing: the abort is retried, never
    /// turned into a resignation.
    func testUnplayableAbortServerErrorIsRetriedNotResigned() async throws {
        let server = try LichessBotFakeGameServer(variantKey: "chess960")
        let h = makeHarness(server: server)
        await h.api.fail("abort", with: [LichessBotAPIError.http(status: 502, message: nil)])
        let run = Task { await h.session.run() }
        try await waitUntil("the game ends", advancing: h.time) { h.observer.finishedStatus != nil }
        await run.value
        let record = await server.record()
        XCTAssertEqual(record.calls, ["abort"])
        XCTAssertEqual(h.observer.finishedStatus, "aborted")
    }

    // MARK: - A2 / A10: victory claims

    /// A victory claim that got no answer says nothing about the win, so no
    /// draw is claimed.
    func testVictoryClaimThatGetsNoAnswerDoesNotClaimADraw() async throws {
        let server = try LichessBotFakeGameServer()
        await server.setScript(afterOurMoves: [.opponentGone(claimWinInSeconds: 10)])
        let h = makeHarness(server: server)
        await h.api.fail("claimVictory", with: [LichessBotAPIError.http(status: 503, message: nil)])
        let run = Task { await h.session.run() }
        try await waitUntil("the claim fails", advancing: h.time) { h.observer.anomalies.contains { $0.hasPrefix("victory claim failed") } }
        let attempts = await h.api.attempts
        XCTAssertFalse(attempts.contains("claimDraw"))
        XCTAssertNil(h.observer.finishedStatus)
        run.cancel()
        await run.value
    }

    /// The opponent returning cancels a claim already in flight: a refusal
    /// arriving afterwards claims no draw.
    func testOpponentReturningCancelsAnInFlightVictoryClaim() async throws {
        let server = try LichessBotFakeGameServer()
        await server.setScript(afterOurMoves: [.opponentGone(claimWinInSeconds: 10)])
        let h = makeHarness(server: server)
        await h.api.hold("claimVictory")
        await h.api.fail("claimVictory", with: [LichessBotAPIError.http(status: 400, message: "cannot claim victory")])
        let run = Task { await h.session.run() }
        try await waitUntil("the claim is in flight", advancing: h.time) { await h.api.isHolding("claimVictory") }
        await h.api.inject(#"{"type":"opponentGone","gone":false}"#)
        try await waitUntil("the opponent's return is handled") { h.observer.actions.contains { $0.hasPrefix("opponent returned") } }
        await h.api.release("claimVictory")
        try await waitUntil("the refusal is handled") { h.observer.actions.contains { $0.contains("cancelled meanwhile") } }
        let attempts = await h.api.attempts
        XCTAssertFalse(attempts.contains("claimDraw"))
        run.cancel()
        await run.value
    }

    // MARK: - A3 / A11: mid-game refresh

    /// Mid-game refresh applies only to a live-trainer game, whatever the
    /// stored toggle says.
    func testMidGameRefreshDoesNotApplyToAChampionGame() async throws {
        let server = try LichessBotFakeGameServer()
        await server.setScript(afterOurMoves: [.opponentReplies, .finish(status: "resign", winner: "white")])
        let pinned = LichessBotScriptedMoveSource()
        let newer = LichessBotGenerationMoveSource(generationID: 2, sourceKind: .champion, modelID: "20260928-2-CHM2")
        let refreshes = SyncBox(0)
        let h = makeHarness(server: server, source: pinned, latestMoveSource: {
            refreshes.modify { $0 += 1 }
            return newer
        }, configure: { $0.model.midGameRefresh = true })
        let run = Task { await h.session.run() }
        try await waitUntil("the game ends") { h.observer.finishedStatus != nil }
        await run.value
        XCTAssertEqual(refreshes.value, 0)
        XCTAssertEqual(pinned.decidedPlies.value, [0, 2])
        XCTAssertEqual(newer.decidedPlies.value, [])
    }

    /// A live-trainer game with mid-game refresh plays the newest built
    /// snapshot, and chat names the model actually playing.
    func testMidGameRefreshSwitchesALiveTrainerGameAndChatNamesTheNewModel() async throws {
        let server = try LichessBotFakeGameServer()
        await server.setScript(afterOurMoves: [.opponentReplies, .finish(status: "resign", winner: "white")])
        let first = LichessBotGenerationMoveSource(generationID: 1, sourceKind: .liveTrainer, modelID: "20260928-1-LIV1")
        let second = LichessBotGenerationMoveSource(generationID: 2, sourceKind: .liveTrainer, modelID: "20260928-2-LIV2")
        let h = makeHarness(server: server, source: first, latestMoveSource: { second }, configure: {
            $0.model.source = .liveTrainer
            $0.model.midGameRefresh = true
            $0.chat.goodbyeEnabled = true
            $0.chat.goodbyeTemplate = "bye from {modelID}"
        })
        let run = Task { await h.session.run() }
        try await waitUntil("the game ends") { h.observer.finishedStatus != nil }
        await run.value
        let record = await server.record()
        XCTAssertEqual(second.decidedPlies.value, [0, 2])
        XCTAssertEqual(first.decidedPlies.value, [])
        XCTAssertEqual(record.chats.last, "bye from 20260928-2-LIV2")
    }

    // MARK: - A5: optional actions

    /// A failed takeback acceptance keeps the stream and the allowance: a
    /// later state accepts the still-pending proposal.
    func testFailedTakebackAcceptanceKeepsTheStreamAndTheAllowance() async throws {
        let server = try LichessBotFakeGameServer()
        await server.setScript(afterOurMoves: [.opponentReplies, .opponentProposesTakeback(plies: 2), .opponentReplies, .finish(status: "resign", winner: "white")])
        let h = makeHarness(server: server, configure: { $0.play.maxTakebacksAcceptedPerGame = 1 })
        await h.api.fail("respondToTakeback", with: [URLError(.timedOut)])
        let run = Task { await h.session.run() }
        try await waitUntil("the acceptance fails") { h.observer.anomalies.contains { $0.hasPrefix("accepting the takeback failed") } }
        let beforeResync = await server.record()
        XCTAssertEqual(beforeResync.streamOpens, 1, "a failed optional action keeps the stream")
        h.session.requestResync(reason: "test: deliver the proposal again")
        try await waitUntil("the game ends", advancing: h.time, by: .milliseconds(500)) { h.observer.finishedStatus != nil }
        await run.value
        let record = await server.record()
        XCTAssertEqual(record.calls, ["takeback:true"])
        XCTAssertEqual(record.acceptedPlies, [0, 2, 2, 4])
    }

    /// A failed resignation plays the decided move on our clock, and a later
    /// reading resigns.
    func testFailedResignationPlaysTheMoveAndResignsLater() async throws {
        let server = try LichessBotFakeGameServer()
        let lost = LichessBotScriptedMoveSource(win: 0, draw: 0, loss: 1)
        let h = makeHarness(server: server, source: lost, configure: {
            $0.play.resignEnabled = true
            $0.play.resignConsecutiveMoves = 1
            $0.play.resignMinimumPly = 0
        })
        await h.api.fail("resign", with: [URLError(.timedOut)])
        let run = Task { await h.session.run() }
        try await waitUntil("the game ends") { h.observer.finishedStatus != nil }
        await run.value
        let record = await server.record()
        XCTAssertEqual(record.acceptedPlies, [0])
        XCTAssertEqual(record.calls, ["resign"])
        XCTAssertEqual(record.streamOpens, 1)
        XCTAssertEqual(h.observer.finishedStatus, "resign")
    }

    // MARK: - A6: command replies

    /// A command reply waiting in the gate doesn't hold up reading the
    /// stream, and so our next move.
    func testCommandReplyDoesNotHoldUpOurNextMove() async throws {
        let server = try LichessBotFakeGameServer()
        await server.setScript(afterOurMoves: [.opponentThinks, .opponentThinks])
        let h = makeHarness(server: server)
        let run = Task { await h.session.run() }
        try await waitUntil("the first move is posted") { await server.record().acceptedPlies == [0] }
        await h.api.hold("chat")
        await h.api.inject(#"{"type":"chatLine","room":"player","username":"Alice","text":"!help"}"#)
        try await waitUntil("the reply is in flight") { await h.api.isHolding("chat") }
        try await server.injectMoves(["a7a5"])
        try await waitUntil("our next move is posted while the reply waits") { await server.record().acceptedPlies == [0, 2] }
        let stillHeld = await h.api.isHolding("chat")
        XCTAssertTrue(stillHeld)
        await h.api.release("chat")
        try await waitUntil("the reply is sent") { await server.record().chats.first?.hasPrefix("Commands:") == true }
        run.cancel()
        await run.value
    }

    // MARK: - A7: resign streak

    /// A ply decided again after a rejected move counts once toward the
    /// resign streak.
    func testReDecidedPlyIsOneReadingInTheResignStreak() async throws {
        let server = try LichessBotFakeGameServer()
        await server.setScript(moveOutcomes: [.rejected(status: 400)], afterOurMoves: [.finish(status: "outoftime", winner: "white")])
        let lost = LichessBotScriptedMoveSource(win: 0, draw: 0, loss: 1)
        let h = makeHarness(server: server, source: lost, configure: {
            $0.play.resignEnabled = true
            $0.play.resignConsecutiveMoves = 2
            $0.play.resignMinimumPly = 0
        })
        let run = Task { await h.session.run() }
        try await waitUntil("the game ends") { h.observer.finishedStatus != nil }
        await run.value
        let record = await server.record()
        XCTAssertEqual(record.calls, [])
        XCTAssertEqual(record.acceptedPlies, [0])
        XCTAssertEqual(h.observer.finishedStatus, "outoftime")
    }

    // MARK: - A9: held moves and the gate

    /// A move held for the operator is not reported as awaiting our move
    /// (which would hold back housekeeping) until it is released.
    func testHeldMoveIsNotReportedAsAwaitingUntilReleased() async throws {
        let server = try LichessBotFakeGameServer()
        await server.setScript(afterOurMoves: [.opponentReplies, .opponentReplies, .finish(status: "resign", winner: "white")])
        let pacing = SyncBox(LichessBotMovePacingSnapshot(delaySeconds: 0, holds: true, releaseRequested: false))
        let h = makeHarness(server: server, pacing: pacing)
        let run = Task { await h.session.run() }
        try await waitUntil("the second move is held and reported as not awaiting", advancing: h.time, by: .milliseconds(250)) {
            self.heldPlies(h.observer) == [2] && h.observer.turnStatuses.value.last?.awaitingOurMove == false
        }
        let heldIndex = h.observer.turnStatuses.value.count
        pacing.value = LichessBotMovePacingSnapshot(delaySeconds: 0, holds: false, releaseRequested: true)
        try await waitUntil("the held move is posted", advancing: h.time, by: .milliseconds(250)) { await server.record().acceptedPlies == [0, 2] }
        let afterHold = h.observer.turnStatuses.value.dropFirst(heldIndex)
        XCTAssertTrue(afterHold.contains { $0.awaitingOurMove }, "a released move awaits its post")
        pacing.value = LichessBotMovePacingSnapshot()
        run.cancel()
        await run.value
    }

    // MARK: - A4: released held moves

    /// While a released held move may be waiting in the gate, a takeback
    /// proposal is left unanswered.
    func testTakebackIsNotAcceptedWhileAReleasedMoveIsPending() async throws {
        let server = try LichessBotFakeGameServer()
        await server.setScript(afterOurMoves: [.opponentReplies, .opponentThinks])
        let pacing = SyncBox(LichessBotMovePacingSnapshot(delaySeconds: 0, holds: true, releaseRequested: false))
        let h = makeHarness(server: server, pacing: pacing, configure: { $0.play.maxTakebacksAcceptedPerGame = 1 })
        let run = Task { await h.session.run() }
        try await waitUntil("the second move is held", advancing: h.time, by: .milliseconds(250)) { self.heldPlies(h.observer) == [2] }
        await h.api.hold("makeMove")
        pacing.value = LichessBotMovePacingSnapshot(delaySeconds: 0, holds: false, releaseRequested: true)
        try await waitUntil("the released move is in flight", advancing: h.time, by: .milliseconds(250)) { await h.api.isHolding("makeMove") }
        await h.api.inject(#"{"type":"gameState","moves":"a2a3 a7a5","wtime":180000,"btime":180000,"winc":2000,"binc":2000,"status":"started","btakeback":true}"#)
        // The loop handles lines in order, so once this later line is seen
        // the proposal before it has been handled.
        await h.api.inject(#"{"type":"chatLine","room":"spectator","username":"Watcher","text":"sentinel"}"#)
        try await waitUntil("the proposal has been handled") { h.observer.receivedLines.contains { $0.contains("sentinel") } }
        let attempts = await h.api.attempts
        XCTAssertFalse(attempts.contains("respondToTakeback"))
        await h.api.release("makeMove")
        try await waitUntil("the held move is posted") { await server.record().acceptedPlies == [0, 2] }
        pacing.value = LichessBotMovePacingSnapshot()
        run.cancel()
        await run.value
    }

    /// A released held move whose POST was refused for the rate limit is not
    /// sent again into a different position at the same ply; the stream is
    /// reopened and the move decided afresh.
    func testReleasedHeldMoveIsNotPostedIntoADifferentPositionAtTheSamePly() async throws {
        let server = try LichessBotFakeGameServer()
        await server.setScript(afterOurMoves: [.opponentReplies, .opponentThinks])
        let pacing = SyncBox(LichessBotMovePacingSnapshot(delaySeconds: 0, holds: true, releaseRequested: false))
        let h = makeHarness(server: server, pacing: pacing)
        let run = Task { await h.session.run() }
        try await waitUntil("the second move is held", advancing: h.time, by: .milliseconds(250)) { self.heldPlies(h.observer) == [2] }
        await h.api.hold("makeMove")
        await h.api.fail("makeMove", with: [LichessBotGateError.rateLimited(cooldown: .seconds(60))])
        pacing.value = LichessBotMovePacingSnapshot(delaySeconds: 0, holds: false, releaseRequested: true)
        try await waitUntil("the released move is in flight", advancing: h.time, by: .milliseconds(250)) { await h.api.isHolding("makeMove") }
        pacing.value = LichessBotMovePacingSnapshot()
        // The same ply, reached through a different reply.
        await h.api.inject(#"{"type":"gameState","moves":"a2a3 a7a6","wtime":180000,"btime":180000,"winc":2000,"binc":2000,"status":"started"}"#)
        await h.api.inject(#"{"type":"chatLine","room":"spectator","username":"Watcher","text":"sentinel"}"#)
        try await waitUntil("the new position has been handled") { h.observer.receivedLines.contains { $0.contains("sentinel") } }
        await h.api.release("makeMove")
        try await waitUntil("the move is posted after a fresh gameFull", advancing: h.time, by: .milliseconds(500)) { await server.record().acceptedPlies == [0, 2] }
        let attempts = await h.api.attempts
        XCTAssertEqual(attempts.filter { $0 == "openGameStream" || $0 == "makeMove" }, ["openGameStream", "makeMove", "makeMove", "openGameStream", "makeMove"])
        run.cancel()
        await run.value
    }

    // MARK: - A10: foreign moves

    /// A state carrying a move this client didn't send and a finished status
    /// still finishes the game.
    func testForeignMoveInAFinishingStateStillFinishesTheGame() async throws {
        let server = try LichessBotFakeGameServer()
        await server.setScript(afterOurMoves: [.opponentReplies, .opponentThinks])
        let h = makeHarness(server: server)
        let run = Task { await h.session.run() }
        try await waitUntil("two moves are posted") { await server.record().acceptedPlies == [0, 2] }
        await server.endSilently(status: "resign", winner: "black")
        try await server.injectMoves(["h7h6", "h2h3"])
        try await waitUntil("the game ends") { h.observer.finishedStatus != nil }
        await run.value
        XCTAssertEqual(h.observer.finishedStatus, "resign")
        XCTAssertTrue(stopReasons(h.observer).contains { $0.contains("did not send") })
    }
}
