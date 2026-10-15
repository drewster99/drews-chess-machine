import XCTest
@testable import DrewsChessMachine

/// Saving new matchmaking settings is seen by the next automatic pass at
/// once. A pass stopped by the hourly cap schedules the next one for when
/// the oldest send leaves the hour — up to an hour away. Before this was
/// fixed, raising the cap (owner report 2026-10-14) changed nothing until
/// then: no send, and the status kept the old pass's "the cap" text.
@MainActor
final class LichessBotMatchmakingSettingsChangeTests: XCTestCase {

    private static let botsNDJSON = [
        #"{"id":"fitbot","username":"FitBot","title":"BOT","perfs":{"blitz":{"games":50,"rating":1600,"rd":60,"prog":0}}}"#,
        #"{"id":"secondbot","username":"SecondBot","title":"BOT","perfs":{"blitz":{"games":50,"rating":1550,"rd":60,"prog":0}}}"#,
    ].joined(separator: "\n") + "\n"

    private func waitUntil(_ description: String, _ condition: () -> Bool) async throws {
        for _ in 0..<3000 {
            if condition() { return }
            try await Task.sleep(for: .milliseconds(10))
        }
        XCTFail("timed out waiting until \(description)")
        throw CancellationError()
    }

    func testRaisingTheHourlyCapLetsTheNextPassSendWithoutWaitingOutTheOldCap() async throws {
        let lichess = LichessBotFakeLichess(accountPerfsJSON: #"{"blitz":{"games":100,"rating":1500,"rd":50,"prog":0}}"#)
        lichess.onlineBotsNDJSON.value = Self.botsNDJSON

        let defaults = try makeTemporaryDefaults()
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("LichessBotMatchmakingSettingsChangeTests-\(UUID().uuidString)", isDirectory: true)
        var settings = LichessBotSettings.testBaseline()
        settings.chat.greetingEnabled = false
        settings.connection.preventSleepWhileOnline = false
        settings.challenge.outgoingChallengeTimeoutSeconds = 0
        settings.challenge.maxConcurrentGames = 3
        settings.challenge.gamesReservedForHumans = 0
        settings.matchmaking.enabled = true
        settings.matchmaking.fillMode = .everyFreeSlot
        settings.matchmaking.timeControls = [.blitz5plus3]
        settings.matchmaking.rated = false
        settings.matchmaking.maxChallengesPerHour = 1
        // Every hour is peak, so the cap doesn't depend on when the test runs.
        settings.matchmaking.peakStartHour = 0
        settings.matchmaking.peakEndHour = 0
        try LichessBotSettingsStore.save(settings, to: defaults)
        let token = LichessBotFakeLichess.token
        let controller = LichessBotController(
            modelProvider: try await LichessBotFakeModelProvider.randomChampion(),
            defaults: defaults,
            dataDirectory: LichessBotDataDirectory(root: root),
            services: LichessBotControllerServices(
                openBotWindow: { _ in },
                makeTransport: { lichess },
                readToken: { _ in token }
            )
        )
        addTeardownBlock { @MainActor in
            controller.abandonAndStop()
            // Writes the stopped runtime already queued land; anything later
            // is refused, so nothing races the removal of its folder.
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
        try await waitUntil("the first automatic pass sends") { lichess.challengedNames.value.count == 1 }
        try await waitUntil("the next pass stops at the hourly cap") {
            controller.matchmakingStatus?.contains("the cap") == true
        }
        XCTAssertEqual(lichess.challengedNames.value.count, 1)

        var raised = controller.settings
        raised.matchmaking.maxChallengesPerHour = 3
        try controller.updateSettings(raised)

        // The send spacing (20 s) still applies; the old cap's hour must not.
        try await waitUntil("a pass under the raised cap sends the second challenge") {
            lichess.challengedNames.value.count == 2
        }
        XCTAssertEqual(Set(lichess.challengedNames.value), ["FitBot", "SecondBot"])
        XCTAssertFalse(controller.matchmakingStatus?.contains("the cap") == true,
                       "the status still reports the old cap: \(controller.matchmakingStatus ?? "none")")
    }
}
