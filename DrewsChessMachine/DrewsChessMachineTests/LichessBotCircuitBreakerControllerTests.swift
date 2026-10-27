import XCTest
@testable import DrewsChessMachine

/// The circuit breakers seen from the controller
/// (`LICHESS_BOT_CIRCUIT_BREAKERS_PLAN.md`): a trip stops new games and
/// never abandons a game in progress (owner decision 2026-10-26).
@MainActor
final class LichessBotCircuitBreakerControllerTests: XCTestCase {

    private func waitUntil(_ description: String, _ condition: () -> Bool) async throws {
        for _ in 0..<6000 {
            if condition() { return }
            try await Task.sleep(for: .milliseconds(10))
        }
        XCTFail("timed out waiting until \(description)")
        throw CancellationError()
    }

    /// Protects: a breaker that waits for the operator (lost on time, as the
    /// owner set it) stops new games while the other game plays on, goes to
    /// Error only once that game has finished, and going online ends it.
    /// Fails if: a trip tears the runtime down (abandoning the game in
    /// progress, as the old breakers did), or the bot never stops, or going
    /// online leaves the hold in place.
    func testALostOnTimeTripFinishesTheOtherGameThenStopsUntilTheOperatorGoesOnline() async throws {
        let defaults = try makeTemporaryDefaults()
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("LichessBotCircuitBreakerControllerTests-\(UUID().uuidString)", isDirectory: true)
        var settings = LichessBotSettings.testBaseline()
        settings.chat.greetingEnabled = false
        settings.chat.goodbyeEnabled = false
        settings.connection.preventSleepWhileOnline = false
        settings.challenge.outgoingChallengeTimeoutSeconds = 0
        settings.breakers.lostOnTime = LichessBotLostOnTimeBreakerSettings(count: 1, windowGames: 30, resumesAutomatically: false, resumeAfterMinutes: 120)
        try LichessBotSettingsStore.save(settings, to: defaults)
        let lichess = LichessBotResumeFakeLichess()
        lichess.resignedExports.modify { $0.formUnion(["cbob", "ccarol"]) }
        let model = try await LichessBotFakeModelProvider.randomChampion()
        let token = LichessBotResumeFakeLichess.token
        let controller = LichessBotController(
            modelProvider: model,
            defaults: defaults,
            dataDirectory: LichessBotDataDirectory(root: root),
            services: LichessBotControllerServices(
                openBotWindow: { _ in },
                makeTransport: { lichess },
                readToken: { _ in token }
            ),
            finishedGameHold: LichessBotController.finishedGameHold,
            postGameChatFetchDelays: [.milliseconds(100)]
        )
        addTeardownBlock { @MainActor in
            controller.abandonAndStop()
            await controller.shutdown(reason: "test teardown")
            if FileManager.default.fileExists(atPath: root.path) {
                do {
                    try FileManager.default.removeItem(at: root)
                } catch {
                    XCTFail("cleanup failed: \(error)")
                }
            }
        }
        await controller.loadPlayerNotes()
        await controller.goOnline()
        try await waitUntil("the event stream is open") { lichess.eventStreamsOpened.value > 0 }
        lichess.startGame(against: "bob")
        lichess.startGame(against: "carol")
        try await waitUntil("both games are listed") { controller.games.count == 2 }
        try await waitUntil("both game streams are open") { lichess.gameStreamIsOpen("cbob") && lichess.gameStreamIsOpen("ccarol") }

        // DCM (black) loses cbob on time.
        lichess.sendGameLine("cbob", #"{"type":"gameState","moves":"","wtime":180000,"btime":0,"winc":2000,"binc":2000,"status":"outoftime","winner":"white"}"#)
        try await waitUntil("the breaker holds new games") { controller.newGameHolds.reasons == [.lostOnTime] }
        try await waitUntil("the bot drains") { controller.connection == .draining }
        XCTAssertTrue(controller.isRunning, "the trip must not stop the bot while a game is in progress")
        XCTAssertTrue(controller.activeGameIDs.contains("ccarol"), "the other game is still being played")
        XCTAssertTrue(lichess.gameStreamIsOpen("ccarol"))

        lichess.opponentResigns("carol")
        try await waitUntil("the bot stops once the last game has finished") {
            if case .error = controller.connection { return true }
            return false
        }
        guard case .error(let text) = controller.connection else { return }
        XCTAssertTrue(text.contains("lost-on-time breaker"), text)
        XCTAssertFalse(controller.isRunning)

        await controller.goOnline()
        XCTAssertEqual(controller.newGameHolds.reasons, [], "going online is the operator's resume")
        XCTAssertEqual(controller.connection, .online)
    }
}
