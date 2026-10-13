//
//  LichessBotStuckGameTests.swift
//  DrewsChessMachineTests
//
//  The stuck-game watchdog, the operator's Abort, and the grace setting
//  (documentation/plans-active/LICHESS_BOT_STUCK_GAME_PLAN.md). Game
//  0qgNk2Ex stayed `started` for hours after its opponent never answered
//  1. e4, with only keep-alives on its stream; these pin that such a game is
//  ended by a client action — abort while a side has yet to move, otherwise
//  a victory claim — and never by a resignation or a draw claim.
//

import XCTest
@testable import DrewsChessMachine

final class LichessBotStuckGameTests: XCTestCase {

    private struct Harness {
        let server: LichessBotFakeGameServer
        let observer: LichessBotRecordingGameObserver
        let time: LichessBotManualTime
        let session: LichessBotGameSession
    }

    private static let graceSeconds = 300

    private func makeHarness(
        server: LichessBotFakeGameServer,
        pacing: LichessBotMovePacingSnapshot = LichessBotMovePacingSnapshot()
    ) -> Harness {
        var settings = LichessBotSettings.testBaseline()
        settings.chat.greetingEnabled = false
        settings.chat.goodbyeEnabled = false
        settings.play.stuckGameGraceSeconds = Self.graceSeconds
        // These tests move the manual clock by minutes at a time. The stream
        // watchdog runs on the same clock; with its usual limits every
        // advance would also drop and reopen the stream, which is not what
        // is under test here.
        settings.connection.gameStreamResyncSeconds = 1_000_000
        settings.connection.eventStreamStallTimeoutSeconds = 1_000_000
        let frozen = settings
        let observer = LichessBotRecordingGameObserver()
        let time = LichessBotManualTime()
        let source = LichessBotScriptedMoveSource()
        let session = LichessBotGameSession(
            gameID: LichessBotFakeGameServer.gameID,
            ourAccountID: LichessBotFakeGameServer.botID,
            api: server,
            moveSource: source,
            latestMoveSource: { source },
            settingsProvider: { frozen },
            observer: observer,
            time: time,
            onTurnStatus: { _, status in observer.turnStatuses.modify { $0.append(status) } },
            pacing: { pacing },
            carryover: .newGame
        )
        return Harness(server: server, observer: observer, time: time, session: session)
    }

    /// Poll `condition` without moving the manual clock.
    private func waitUntil(_ description: String, _ condition: () async -> Bool) async throws {
        for _ in 0..<2000 {
            if await condition() { return }
            try await Task.sleep(for: .milliseconds(5))
        }
        XCTFail("timed out waiting until \(description)")
        throw CancellationError()
    }

    /// Stream a sentinel line after everything streamed so far and wait
    /// until the session reads it: every earlier line, and the watchdog's
    /// check after the sentinel itself, have then been fully handled (the
    /// session handles one stream item at a time).
    private func waitUntilEarlierLinesAreHandled(_ harness: Harness, marker: String) async throws {
        await harness.server.sendSentinel(marker)
        try await waitUntil("the sentinel \(marker) is read") {
            harness.observer.receivedLines.contains { $0.contains(marker) }
        }
        // The line is reported before it is handled; a second sentinel is
        // reported only after the first one's handling (and check) ended.
        await harness.server.sendSentinel(marker + "-after")
        try await waitUntil("the sentinel \(marker)-after is read") {
            harness.observer.receivedLines.contains { $0.contains(marker + "-after") }
        }
    }

    /// Wait until the session has opened its stream and handled the
    /// `gameFull`. The fake server drops lines sent before the session opens
    /// a stream (it has no continuation yet to yield them to), so a sentinel
    /// sent right after starting `run()` can be lost; wait for the open
    /// first.
    private func waitUntilTheGameIsTracked(_ harness: Harness) async throws {
        try await waitUntil("the session opens the game stream") { await harness.server.record().streamOpens >= 1 }
        try await waitUntilEarlierLinesAreHandled(harness, marker: "sentinel-start")
    }

    private func finishRun(_ run: Task<Void, Never>, harness: Harness) async throws {
        try await waitUntil("the session ends") { await harness.session.isFinished }
        await run.value
    }

    /// Ten plies of an ordinary opening; White is to move after them.
    private static let tenPlies = ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "d2d3", "f8c5", "b1c3", "d7d6"]

    // MARK: - Watchdog

    /// The 0qgNk2Ex case: DCM played 1. e4 and the opponent never moved.
    /// Keep-alives alone drive the watchdog, which aborts exactly once the
    /// opponent's whole clock plus the grace has passed — not before.
    func testAGameStuckAtPlyOneIsAbortedAfterTheClockAndGrace() async throws {
        let server = try LichessBotFakeGameServer(botIsWhite: true, initialTokens: ["e2e4"])
        await server.setClockMilliseconds(white: 30_000, black: 30_000)
        let h = makeHarness(server: server)
        let run = Task { await h.session.run() }
        try await waitUntilTheGameIsTracked(h)

        // One second short of 30 s + 300 s.
        h.time.advance(by: .seconds(329))
        await server.sendKeepAlive()
        try await waitUntilEarlierLinesAreHandled(h, marker: "sentinel-before-deadline")
        let early = await server.record()
        XCTAssertEqual(early.calls, [], "nothing before the opponent's clock plus the grace has passed")

        h.time.advance(by: .seconds(2))
        await server.sendKeepAlive()
        try await finishRun(run, harness: h)

        let record = await server.record()
        XCTAssertEqual(record.calls, ["abort"])
        XCTAssertEqual(record.moveAttempts, 0)
        XCTAssertEqual(h.observer.finishedStatus, "aborted")
        XCTAssertTrue(h.observer.actions.contains("aborted (stuck game)"))
        XCTAssertTrue(h.observer.actions.contains { $0.hasPrefix("opponent still to move at ply 1") && $0.hasSuffix("aborting") })
    }

    /// Past the first moves, with the opponent's clock spent: a victory
    /// claim once the grace has passed. A refused claim is tried again no
    /// sooner than one grace period later; never a draw claim, never a
    /// resignation.
    func testAGameStuckAtPlyTenClaimsVictoryAndRetriesAfterARefusal() async throws {
        let server = try LichessBotFakeGameServer(botIsWhite: false, initialTokens: Self.tenPlies)
        await server.setClockMilliseconds(white: 0, black: 120_000)
        await server.setVictoryClaimSucceeds(false)
        let h = makeHarness(server: server)
        let run = Task { await h.session.run() }
        try await waitUntilTheGameIsTracked(h)

        h.time.advance(by: .seconds(299))
        try await waitUntilEarlierLinesAreHandled(h, marker: "sentinel-before-deadline")
        let early = await server.record()
        XCTAssertEqual(early.calls, [])

        h.time.advance(by: .seconds(2))
        await server.sendKeepAlive()
        try await waitUntil("the first claim is refused") {
            h.observer.anomalies.contains { $0.hasPrefix("stuck game: victory claim failed") }
        }
        try await waitUntilEarlierLinesAreHandled(h, marker: "sentinel-after-first-claim")
        let afterFirst = await server.record()
        XCTAssertEqual(afterFirst.calls, ["claimVictory"])

        // One second short of a grace period after the refused claim.
        h.time.advance(by: .seconds(299))
        try await waitUntilEarlierLinesAreHandled(h, marker: "sentinel-before-retry")
        let beforeRetry = await server.record()
        XCTAssertEqual(beforeRetry.calls, ["claimVictory"], "no retry sooner than one grace period")

        h.time.advance(by: .seconds(2))
        try await waitUntilEarlierLinesAreHandled(h, marker: "sentinel-after-retry")
        XCTAssertTrue(h.observer.actions.contains { $0.hasSuffix("claiming victory (attempt 2)") })
        try await waitUntil("the second claim is refused") {
            h.observer.anomalies.filter { $0.hasPrefix("stuck game: victory claim failed") }.count == 2
        }
        try await waitUntilEarlierLinesAreHandled(h, marker: "sentinel-after-second-claim")
        let afterRetry = await server.record()
        XCTAssertEqual(afterRetry.calls, ["claimVictory", "claimVictory"])

        await server.setVictoryClaimSucceeds(true)
        h.time.advance(by: .seconds(Self.graceSeconds + 1))
        await server.sendKeepAlive()
        try await finishRun(run, harness: h)

        let record = await server.record()
        XCTAssertEqual(record.calls, ["claimVictory", "claimVictory", "claimVictory"])
        XCTAssertFalse(record.calls.contains("claimDraw"))
        XCTAssertFalse(record.calls.contains("resign"))
        XCTAssertFalse(record.calls.contains("abort"), "both sides have moved: never an abort")
        XCTAssertEqual(h.observer.finishedStatus, "timeout")
        XCTAssertTrue(h.observer.actions.contains("claimed victory (stuck game)"))
    }

    /// An opponent move before the deadline starts a new turn: the old
    /// deadline no longer applies, and the new one is counted from the
    /// state that made it the opponent's turn again.
    func testAnOpponentMoveBeforeTheDeadlineResetsIt() async throws {
        let server = try LichessBotFakeGameServer(botIsWhite: true, initialTokens: ["e2e4"])
        await server.setClockMilliseconds(white: 30_000, black: 30_000)
        await server.setScript(afterOurMoves: [.opponentThinks])
        let h = makeHarness(server: server)
        let run = Task { await h.session.run() }
        try await waitUntilTheGameIsTracked(h)

        h.time.advance(by: .seconds(200))
        try await server.injectMoves(["e7e5"])
        try await waitUntil("DCM answers at ply 2") { await server.record().acceptedPlies == [2] }
        try await waitUntilEarlierLinesAreHandled(h, marker: "sentinel-after-reply")

        // Past the first turn's deadline (330 s), short of the new one's
        // (200 s + 330 s).
        h.time.advance(by: .seconds(200))
        await server.sendKeepAlive()
        try await waitUntilEarlierLinesAreHandled(h, marker: "sentinel-past-the-old-deadline")
        let early = await server.record()
        XCTAssertEqual(early.calls, [], "the opponent's move reset the watchdog")

        h.time.advance(by: .seconds(131))
        await server.sendKeepAlive()
        try await finishRun(run, harness: h)
        let record = await server.record()
        XCTAssertEqual(record.calls, ["claimVictory"], "three plies played: a victory claim, not an abort")
    }

    /// Our own turn never triggers the watchdog, however long it lasts: a
    /// move held for the operator keeps the stream being read while DCM is
    /// on move, with the opponent's clock long spent.
    func testOurOwnTurnNeverTriggersIt() async throws {
        let server = try LichessBotFakeGameServer(botIsWhite: true, initialTokens: ["e2e4", "e7e5"])
        // Our clock large enough that the held move's clock floor never
        // releases it while the manual clock runs on.
        await server.setClockMilliseconds(white: 100_000_000, black: 0)
        var pacing = LichessBotMovePacingSnapshot()
        pacing.holds = true
        let h = makeHarness(server: server, pacing: pacing)
        let run = Task { await h.session.run() }
        try await waitUntil("DCM holds its move") {
            h.observer.events.value.contains { event in
                if case .moveHeld = event { return true }
                return false
            }
        }
        for step in 0..<5 {
            h.time.advance(by: .seconds(Self.graceSeconds * 2))
            await server.sendKeepAlive()
            try await waitUntilEarlierLinesAreHandled(h, marker: "sentinel-\(step)")
        }
        let record = await server.record()
        XCTAssertEqual(record.calls, [])
        XCTAssertEqual(record.moveAttempts, 0)
        run.cancel()
        await run.value
    }

    /// A game that finishes never triggers it, even when the finishing
    /// state arrives after the opponent's clock plus the grace has passed.
    ///
    /// No stream item arrives between tracking the turn and the deadline,
    /// so the only check after the deadline is the one run after the
    /// finishing state itself — and only `finished` stands between it and
    /// an abort: `finish()` clears the timed turn, and the check returns on
    /// `finished`. With both removed, that check logs "opponent still to
    /// move" before starting its request, and the assertion on the actions
    /// fails deterministically (the abort itself may lose its race with the
    /// session's exit). Each of the two alone is enough to keep it quiet;
    /// the task's own `!finished` re-check is a third layer.
    func testAGameThatFinishesNeverTriggersIt() async throws {
        let server = try LichessBotFakeGameServer(botIsWhite: true, initialTokens: ["e2e4"])
        await server.setClockMilliseconds(white: 30_000, black: 30_000)
        let h = makeHarness(server: server)
        let run = Task { await h.session.run() }
        try await waitUntilTheGameIsTracked(h)

        // Past 30 s + 300 s, with nothing streamed meanwhile.
        h.time.advance(by: .seconds(400))
        await server.finishGame(status: "outoftime", winner: "white")
        try await finishRun(run, harness: h)

        let record = await server.record()
        XCTAssertEqual(record.calls, [])
        XCTAssertEqual(h.observer.finishedStatus, "outoftime")
        XCTAssertFalse(h.observer.actions.contains { $0.hasPrefix("opponent still to move") })
    }

    /// A reopened stream's `gameFull` for the same turn keeps the turn's
    /// start and the clock it began with: the watchdog acts at the original
    /// deadline, not one counted from the reopen or from the new, lower
    /// clock.
    func testAReopenedStreamKeepsTheOriginalDeadline() async throws {
        let server = try LichessBotFakeGameServer(botIsWhite: true, initialTokens: ["e2e4"])
        await server.setClockMilliseconds(white: 30_000, black: 30_000)
        let h = makeHarness(server: server)
        let run = Task { await h.session.run() }
        try await waitUntilTheGameIsTracked(h)

        h.time.advance(by: .seconds(200))
        var elapsedSeconds = 200
        await server.setClockMilliseconds(white: 30_000, black: 5_000)
        h.session.requestResync(reason: "test reopen")
        // The stream watchdog checks once a (manual) second.
        for _ in 0..<20 {
            if await server.record().streamOpens == 2 { break }
            h.time.advance(by: .seconds(1))
            elapsedSeconds += 1
            try await Task.sleep(for: .milliseconds(20))
        }
        let reopened = await server.record()
        XCTAssertEqual(reopened.streamOpens, 2, "the stream was reopened")
        try await waitUntilEarlierLinesAreHandled(h, marker: "sentinel-after-reopen")

        // The original deadline is 0 + 30 s + 300 s. Counted from the
        // reopened clock it would be 305 s; from the reopen, past 500 s.
        h.time.advance(by: .seconds(329 - elapsedSeconds))
        await server.sendKeepAlive()
        try await waitUntilEarlierLinesAreHandled(h, marker: "sentinel-before-deadline")
        let early = await server.record()
        XCTAssertEqual(early.calls, [])

        h.time.advance(by: .seconds(2))
        await server.sendKeepAlive()
        try await finishRun(run, harness: h)
        let record = await server.record()
        XCTAssertEqual(record.calls, ["abort"])
    }

    /// A draw offer without a move is a new state for the same turn; it
    /// doesn't restart the watchdog.
    func testADrawOfferOnTheSameTurnKeepsTheDeadline() async throws {
        let server = try LichessBotFakeGameServer(botIsWhite: true, initialTokens: ["e2e4"])
        await server.setClockMilliseconds(white: 30_000, black: 30_000)
        let h = makeHarness(server: server)
        let run = Task { await h.session.run() }
        try await waitUntilTheGameIsTracked(h)

        h.time.advance(by: .seconds(200))
        await server.opponentOffersDrawWithoutMoving()
        try await waitUntilEarlierLinesAreHandled(h, marker: "sentinel-after-offer")

        h.time.advance(by: .seconds(129))
        await server.sendKeepAlive()
        try await waitUntilEarlierLinesAreHandled(h, marker: "sentinel-before-deadline")
        let early = await server.record()
        XCTAssertEqual(early.calls, [])

        h.time.advance(by: .seconds(2))
        await server.sendKeepAlive()
        try await finishRun(run, harness: h)
        let record = await server.record()
        XCTAssertEqual(record.calls, ["abort"], "aborted at the original deadline; the offer is never answered")
    }

    /// The watchdog's request runs off the stream loop: while it waits (the
    /// gate can hold it through a cooldown), the stream is still read, and
    /// an opponent move is answered at once. The move starts a new turn,
    /// which cancels the stale claim; no draw claim or resignation follows.
    func testTheStreamIsReadWhileTheWatchdogsRequestIsPending() async throws {
        let server = try LichessBotFakeGameServer(botIsWhite: true, initialTokens: ["e2e4", "e7e5", "g1f3"])
        await server.setClockMilliseconds(white: 30_000, black: 30_000)
        await server.setScript(afterOurMoves: [.opponentThinks])
        await server.holdVictoryClaims()
        let h = makeHarness(server: server)
        let run = Task { await h.session.run() }
        try await waitUntilTheGameIsTracked(h)

        h.time.advance(by: .seconds(331))
        await server.sendKeepAlive()
        try await waitUntil("the claim is held") { await server.heldVictoryClaimCount == 1 }

        try await server.injectMoves(["b8c6"])
        try await waitUntil("DCM answers at ply 4") { await server.record().acceptedPlies == [4] }
        let heldCount = await server.heldVictoryClaimCount
        XCTAssertEqual(heldCount, 1, "our move went out while the claim was still pending")

        await server.setVictoryClaimSucceeds(false)
        await server.releaseHeldVictoryClaims()
        try await waitUntil("the stale claim is reported cancelled") {
            h.observer.actions.contains { $0.hasPrefix("stuck game: victory claim at ply 3 cancelled") }
        }
        try await waitUntilEarlierLinesAreHandled(h, marker: "sentinel-end")
        let record = await server.record()
        XCTAssertEqual(record.calls, ["claimVictory"])
        XCTAssertFalse(h.observer.anomalies.contains { $0.hasPrefix("stuck game") }, "a cancelled claim is not a failure")
        run.cancel()
        await run.value
    }

    /// An opponent-gone claim scheduled for the same turn is cancelled when
    /// the watchdog claims: two claims racing could leave the loser refused
    /// while the game still looks live, and the opponent-gone path would
    /// then claim a draw.
    func testTheWatchdogsClaimCancelsAPendingOpponentGoneClaim() async throws {
        let server = try LichessBotFakeGameServer(botIsWhite: true, initialTokens: ["e2e4", "e7e5"])
        await server.setClockMilliseconds(white: 30_000, black: 30_000)
        await server.setScript(afterOurMoves: [.opponentGone(claimWinInSeconds: 100_000)])
        let h = makeHarness(server: server)
        let run = Task { await h.session.run() }
        try await waitUntil("the opponent-gone claim is scheduled") {
            h.observer.actions.contains { $0.hasPrefix("opponent gone") }
        }

        h.time.advance(by: .seconds(331))
        await server.sendKeepAlive()
        try await finishRun(run, harness: h)
        let record = await server.record()
        XCTAssertEqual(record.calls, ["claimVictory"])
        XCTAssertTrue(h.observer.actions.contains("opponent-gone claim cancelled: the stuck-game victory claim replaced it"))
        XCTAssertTrue(h.observer.actions.contains("claimed victory (stuck game)"))
        XCTAssertEqual(h.observer.finishedStatus, "timeout")
    }

    // MARK: - Manual abort

    func testAbortNowPostsTheAbortAndLogsIt() async throws {
        let server = try LichessBotFakeGameServer(botIsWhite: true, initialTokens: ["e2e4"])
        let h = makeHarness(server: server)
        let run = Task { await h.session.run() }
        try await waitUntilTheGameIsTracked(h)
        try await h.session.abortNow()
        try await finishRun(run, harness: h)
        let record = await server.record()
        XCTAssertEqual(record.calls, ["abort"])
        XCTAssertTrue(h.observer.actions.contains("aborted (operator)"))
        XCTAssertEqual(h.observer.finishedStatus, "aborted")
    }

    /// The tile's and game view's Abort is offered only while the game is
    /// in progress and a side has yet to make its first move.
    @MainActor
    func testAbortIsOfferedOnlyWhileAbortable() {
        let game = LichessBotLiveGame(id: "g1", startedAt: Date(), ourAccountID: LichessBotFakeGameServer.botID)
        game.apply(.streamLine(Self.gameFullLine(), receivedAt: Date()))
        XCTAssertTrue(game.isAbortable, "no move yet")
        game.apply(.streamLine(Self.stateLine(["e2e4"]), receivedAt: Date()))
        XCTAssertTrue(game.isAbortable, "only White has moved")
        game.apply(.streamLine(Self.stateLine(["e2e4", "e7e5"]), receivedAt: Date()))
        XCTAssertFalse(game.isAbortable, "both sides have moved")

        let finished = LichessBotLiveGame(id: "g2", startedAt: Date(), ourAccountID: LichessBotFakeGameServer.botID)
        finished.apply(.streamLine(Self.gameFullLine(), receivedAt: Date()))
        finished.apply(.streamLine(Self.stateLine(["e2e4"], status: "aborted"), receivedAt: Date()))
        XCTAssertTrue(finished.isFinished)
        XCTAssertFalse(finished.isAbortable, "a finished game is never offered an abort")
    }

    private static func stateLine(_ tokens: [String], status: String = "started") -> Data {
        Data(#"{"type":"gameState","moves":"\#(tokens.joined(separator: " "))","wtime":170000,"btime":165000,"winc":2000,"binc":2000,"status":"\#(status)"}"#.utf8)
    }

    private static func gameFullLine() -> Data {
        Data(#"{"type":"gameFull","id":"g1","variant":{"key":"standard"},"clock":{"initial":180000,"increment":2000},"speed":"blitz","rated":false,"createdAt":1700000000000,"white":{"id":"drewschessmachine","name":"DrewsChessMachine","title":"BOT","rating":1500},"black":{"id":"alice","name":"Alice","rating":1600},"initialFen":"startpos","state":{"type":"gameState","moves":"","wtime":180000,"btime":180000,"winc":2000,"binc":2000,"status":"started"}}"#.utf8)
    }

    // MARK: - Settings

    /// Settings saved before the grace existed load with its default,
    /// reported as filled from the defaults.
    func testSavedSettingsWithoutTheGraceLoadWithTheDefault() throws {
        let defaults = try makeTemporaryDefaults()
        let data = try JSONEncoder().encode(LichessBotSettings())
        var object = try XCTUnwrap(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        var play = try XCTUnwrap(object["play"] as? [String: Any])
        play["stuckGameGraceSeconds"] = nil
        object["play"] = play
        defaults.set(try JSONSerialization.data(withJSONObject: object), forKey: LichessBotSettingsStore.defaultsKey)

        let result = try LichessBotSettingsStore.loadReporting(from: defaults)
        XCTAssertEqual(result.settings.play.stuckGameGraceSeconds, 300)
        XCTAssertEqual(result.filledFromDefaults, ["play.stuckGameGraceSeconds"])
        XCTAssertEqual(result.ignoredSavedKeys, [])
    }

    func testTheGraceIsValidatedToItsRange() {
        func problems(grace: Int) -> [String] {
            var settings = LichessBotSettings()
            settings.play.stuckGameGraceSeconds = grace
            return settings.validationProblems().filter { $0.contains("stuck-game grace") }
        }
        XCTAssertEqual(problems(grace: 59).count, 1)
        XCTAssertEqual(problems(grace: 3601).count, 1)
        XCTAssertEqual(problems(grace: 60), [])
        XCTAssertEqual(problems(grace: 3600), [])
        XCTAssertEqual(LichessBotSettings().play.stuckGameGraceSeconds, 300)
    }
}
