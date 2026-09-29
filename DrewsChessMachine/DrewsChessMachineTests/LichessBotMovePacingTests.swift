import XCTest
@testable import DrewsChessMachine

/// Plan §14.3c: the operator's per-game move delay / hold, operator chat
/// rules, and form-body encoding.
final class LichessBotMovePacingTests: XCTestCase {

    private func makeSession(server: LichessBotFakeGameServer, observer: LichessBotRecordingGameObserver, time: LichessBotManualTime, pacing: SyncBox<LichessBotMovePacingSnapshot>) -> LichessBotGameSession {
        var settings = LichessBotSettings()
        settings.chat.greetingEnabled = false
        settings.chat.goodbyeEnabled = false
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
            onTurnStatus: { _, _ in },
            pacing: { pacing.value }
        )
    }

    /// Poll `condition`, nudging the manual clock by a quarter second (the
    /// paced move's own polling step) each time.
    private func waitUntil(_ description: String, time: LichessBotManualTime, _ condition: () async -> Bool) async throws {
        for _ in 0..<2000 {
            if await condition() { return }
            time.advance(by: .milliseconds(250))
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

    private func releaseReasons(_ observer: LichessBotRecordingGameObserver) -> [String] {
        observer.events.value.compactMap { event in
            if case .moveReleased(_, let reason) = event { return reason }
            return nil
        }
    }

    /// DCM's first move is never held; later moves wait for Play move while
    /// the stream keeps running, then post.
    func testHoldWaitsForTheOperatorAfterTheFirstMove() async throws {
        let server = try LichessBotFakeGameServer()
        await server.setScript(afterOurMoves: [.opponentReplies, .opponentReplies, .finish(status: "resign", winner: "white")])
        let observer = LichessBotRecordingGameObserver()
        let time = LichessBotManualTime()
        let pacing = SyncBox(LichessBotMovePacingSnapshot(delaySeconds: 0, holds: true, releaseRequested: false))
        let session = makeSession(server: server, observer: observer, time: time, pacing: pacing)
        let run = Task { await session.run() }

        try await waitUntil("the second move is held", time: time) { self.heldPlies(observer) == [2] }
        var record = await server.record()
        XCTAssertEqual(record.acceptedPlies, [0], "the first move is never held")

        for _ in 0..<8 {
            time.advance(by: .milliseconds(250))
            try await Task.sleep(for: .milliseconds(5))
        }
        record = await server.record()
        XCTAssertEqual(record.acceptedPlies, [0], "a held move waits for the operator")

        pacing.value = LichessBotMovePacingSnapshot(delaySeconds: 0, holds: false, releaseRequested: true)
        // With the hold off, the script's reply to the released move is
        // answered at once, so the list can already have moved past ply 2.
        try await waitUntil("the held move is posted", time: time) { await server.record().acceptedPlies.starts(with: [0, 2]) }
        XCTAssertEqual(releaseReasons(observer), ["the operator played it"])

        pacing.value = LichessBotMovePacingSnapshot()
        try await waitUntil("the game ends", time: time) { await server.record().acceptedPlies == [0, 2, 4] }
        run.cancel()
        await run.value
    }

    /// A plain delay posts on its own once it has elapsed.
    func testDelayPostsAfterItElapses() async throws {
        let server = try LichessBotFakeGameServer()
        await server.setScript(afterOurMoves: [.opponentReplies, .finish(status: "resign", winner: "white")])
        let observer = LichessBotRecordingGameObserver()
        let time = LichessBotManualTime()
        let pacing = SyncBox(LichessBotMovePacingSnapshot(delaySeconds: 2, holds: false, releaseRequested: false))
        let session = makeSession(server: server, observer: observer, time: time, pacing: pacing)
        let run = Task { await session.run() }
        try await waitUntil("the delayed move is posted", time: time) { await server.record().acceptedPlies == [0, 2] }
        XCTAssertEqual(releaseReasons(observer), ["the 2 s delay elapsed"])
        await run.value
    }

    func testClockFloorIncludesTwiceTheRoundTrip() {
        XCTAssertEqual(LichessBotMovePacingSnapshot.clockFloorMilliseconds(lastMoveRoundTripMilliseconds: nil), 30_000)
        XCTAssertEqual(LichessBotMovePacingSnapshot.clockFloorMilliseconds(lastMoveRoundTripMilliseconds: 150), 30_300)
    }

    // MARK: - Operator chat

    func testOperatorChatLimitsAndLinkWarning() {
        XCTAssertEqual(LichessBotOperatorChat.problem(with: ""), "the message is empty")
        XCTAssertNil(LichessBotOperatorChat.problem(with: "Good game!"))
        XCTAssertNotNil(LichessBotOperatorChat.problem(with: String(repeating: "𝕏", count: 71)), "142 UTF-16 units")
        XCTAssertTrue(LichessBotOperatorChat.looksLikeLink("see lichess.org"))
        XCTAssertTrue(LichessBotOperatorChat.looksLikeLink("https://example"))
        XCTAssertFalse(LichessBotOperatorChat.looksLikeLink("e.g. 6 super + 12 performance"))
    }

    // MARK: - Form bodies

    /// `+` must be percent-encoded, or a form decoder reads it as a space.
    func testFormBodyEncodesPlusAndSpaces() throws {
        let body = try LichessBotAPIClient.formBody([("room", "player"), ("text", "6 super + 12 performance & more=yes")])
        XCTAssertEqual(String(decoding: body, as: UTF8.self), "room=player&text=6%20super%20%2B%2012%20performance%20%26%20more%3Dyes")
    }
}
