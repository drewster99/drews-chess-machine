import Observation
import XCTest
@testable import DrewsChessMachine

/// The once-a-second poll must not tell the views the game list changed when
/// it didn't. An in-place mutation of an `@Observable` property notifies
/// every reader even when it removes nothing, and every reader of `games`
/// (the Live section's game picker, the single game view, the grid) then
/// re-renders once a second, visible or not.
@MainActor
final class LichessBotPollObservationTests: XCTestCase {

    private func waitUntil(_ description: String, _ condition: () -> Bool) async throws {
        for _ in 0..<3000 {
            if condition() { return }
            try await Task.sleep(for: .milliseconds(10))
        }
        XCTFail("timed out waiting until \(description)")
        throw CancellationError()
    }

    private func makeOnlineController(
        lichess: LichessBotFakeLichess,
        modelProvider: LichessBotFakeModelProvider,
        finishedGameHold: Duration
    ) async throws -> LichessBotController {
        let defaults = try makeTemporaryDefaults()
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("LichessBotPollObservationTests-\(UUID().uuidString)", isDirectory: true)
        var settings = LichessBotSettings.testBaseline()
        settings.chat.greetingEnabled = false
        settings.connection.preventSleepWhileOnline = false
        settings.challenge.outgoingChallengeTimeoutSeconds = 0
        try LichessBotSettingsStore.save(settings, to: defaults)
        let token = LichessBotFakeLichess.token
        let controller = LichessBotController(
            modelProvider: modelProvider,
            defaults: defaults,
            dataDirectory: LichessBotDataDirectory(root: root),
            services: LichessBotControllerServices(
                makeTransport: { lichess },
                readToken: { _ in token }
            ),
            finishedGameHold: finishedGameHold
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
        XCTAssertEqual(controller.connection, .online)
        try await waitUntil("the event stream is open") { lichess.eventStreamIsOpen }
        return controller
    }

    /// True once `games` reports a change after tracking starts.
    private func trackGames(of controller: LichessBotController) -> SyncBox<Bool> {
        let changed = SyncBox(false)
        withObservationTracking {
            _ = controller.games
        } onChange: {
            changed.value = true
        }
        return changed
    }

    /// Long enough for at least two polls.
    private let twoPolls = Duration.milliseconds(2500)

    func testThePollDoesNotReportAChangeWhenThereAreNoGames() async throws {
        let lichess = LichessBotFakeLichess()
        let controller = try await makeOnlineController(
            lichess: lichess,
            modelProvider: LichessBotFakeModelProvider(snapshot: nil),
            finishedGameHold: LichessBotController.finishedGameHold)
        let changed = trackGames(of: controller)
        try await Task.sleep(for: twoPolls)
        XCTAssertFalse(changed.value, "the poll reported a change to an unchanged game list")
    }

    func testThePollDoesNotReportAChangeWhileAFinishedGameIsRetained() async throws {
        let lichess = LichessBotFakeLichess()
        let controller = try await makeOnlineController(
            lichess: lichess,
            modelProvider: try await LichessBotFakeModelProvider.randomChampion(),
            finishedGameHold: .milliseconds(50))
        let request = LichessBotOutgoingChallenge(rated: false, clockLimitSeconds: 300, clockIncrementSeconds: 3, color: .random)
        try controller.enqueueChallenges(to: [LichessBotChallengeQueue.Player(username: "bob", userID: "bob")], request: request)
        try await waitUntil("the challenge is pending") { controller.pendingChallenges.count == 1 }
        let bobGame = LichessBotFakeLichess.challengeID(for: "bob")
        lichess.startGame(against: "bob")
        try await waitUntil("bob's game is in progress") { controller.activeGameIDs.contains(bobGame) && lichess.gameStreamIsOpen(bobGame) }
        lichess.endGame(against: "bob")
        try await waitUntil("bob's game is finished") { controller.games.first { $0.id == bobGame }?.isFinished == true }
        try await waitUntil("bob's game session has ended") { !controller.activeGameIDs.contains(bobGame) }
        // Past the hold, with the game still inside its retention window.
        try await Task.sleep(for: .milliseconds(300))
        XCTAssertTrue(controller.games.contains { $0.id == bobGame })

        let changed = trackGames(of: controller)
        try await Task.sleep(for: twoPolls)
        XCTAssertFalse(changed.value, "the poll reported a change to an unchanged game list")
        XCTAssertTrue(controller.games.contains { $0.id == bobGame }, "the retained game is still listed")
    }
}
