import XCTest
@testable import DrewsChessMachine

/// A bot refused for Lichess's bot-vs-bot daily limit is left alone until
/// the time Lichess gave, and the status says so, even when the player-notes
/// file (where limits are also saved) couldn't be loaded.
@MainActor
final class LichessBotBotLimitTests: XCTestCase {

    private func waitUntil(_ description: String, _ condition: () -> Bool) async throws {
        for _ in 0..<3000 {
            if condition() { return }
            try await Task.sleep(for: .milliseconds(10))
        }
        XCTFail("timed out waiting until \(description)")
        throw CancellationError()
    }

    private static let limitUntil = Date().addingTimeInterval(3 * 3600)

    private static var refusalText: String {
        let formatter = ISO8601DateFormatter()
        return "maia1 played 150 games against other bots today, please wait until \(formatter.string(from: limitUntil)) to challenge them."
    }

    /// An online controller whose player-notes file is unreadable.
    private func makeOnlineControllerWithUnreadableNotes(lichess: LichessBotResumeFakeLichess, configure: (inout LichessBotSettings) -> Void) async throws -> LichessBotController {
        let defaults = try makeTemporaryDefaults()
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("LichessBotBotLimitTests-\(UUID().uuidString)", isDirectory: true)
        let directory = LichessBotDataDirectory(root: root)
        try directory.createDirectories()
        try Data("not player notes".utf8).write(to: directory.playerNotesURL)
        var settings = LichessBotSettings.testBaseline()
        settings.chat.greetingEnabled = false
        settings.connection.preventSleepWhileOnline = false
        settings.challenge.outgoingChallengeTimeoutSeconds = 0
        configure(&settings)
        try LichessBotSettingsStore.save(settings, to: defaults)
        let token = LichessBotResumeFakeLichess.token
        let controller = LichessBotController(
            modelProvider: try await LichessBotFakeModelProvider.randomChampion(),
            defaults: defaults,
            dataDirectory: directory,
            services: LichessBotControllerServices(
                makeTransport: { lichess },
                readToken: { _ in token }
            ),
            finishedGameHold: LichessBotController.finishedGameHold
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
        XCTAssertNil(controller.playerNotes, "the fixture's notes file is unreadable")
        await controller.goOnline()
        XCTAssertEqual(controller.connection, .online)
        try await waitUntil("the event stream is open") { lichess.eventStreamsOpened.value > 0 }
        return controller
    }

    private let request = LichessBotOutgoingChallenge(rated: false, clockLimitSeconds: 300, clockIncrementSeconds: 3, color: .random)

    func testABotAtItsLimitIsNotChallengedAgainWhenTheNotesFileIsUnreadable() async throws {
        let lichess = LichessBotResumeFakeLichess()
        let refusal = Self.refusalText
        lichess.challengeRefusals.modify { $0["maia1"] = refusal }
        let controller = try await makeOnlineControllerWithUnreadableNotes(lichess: lichess) { _ in }
        let maia = LichessBotChallengeQueue.Player(username: "maia1", userID: "maia1")
        try controller.enqueueChallenges(to: [maia], request: request)
        try await waitUntil("the first challenge is refused") { lichess.challengedNames.value.count == 1 && !controller.challengeQueue.hasEntriesToSend }
        try controller.enqueueChallenges(to: [maia], request: request)
        try await waitUntil("the second queue entry is settled") { !controller.challengeQueue.hasEntriesToSend }
        try await Task.sleep(for: .milliseconds(300))
        XCTAssertEqual(lichess.challengedNames.value, ["maia1"], "the limit Lichess gave is honored without asking again")
    }

    func testMatchmakingSaysWhenItsPickIsAtTheBotLimit() async throws {
        let lichess = LichessBotResumeFakeLichess(accountPerfsJSON: #"{"blitz":{"games":100,"rating":1500,"rd":50,"prog":0}}"#)
        let refusal = Self.refusalText
        lichess.challengeRefusals.modify { $0["maia1"] = refusal }
        lichess.onlineBotsNDJSON.value = #"{"id":"maia1","username":"maia1","title":"BOT","perfs":{"blitz":{"games":50,"rating":1500,"rd":60,"prog":0}}}"# + "\n"
        let controller = try await makeOnlineControllerWithUnreadableNotes(lichess: lichess) { settings in
            settings.matchmaking.timeControls = [.blitz5plus3]
            settings.matchmaking.rated = false
            settings.matchmaking.enabled = false
        }
        await controller.fillOpenSlots()
        XCTAssertEqual(lichess.challengedNames.value, ["maia1"])
        let status = try XCTUnwrap(controller.matchmakingStatus)
        XCTAssertTrue(status.contains("maia1 is at Lichess's bot-vs-bot daily limit until"), status)
    }
}
