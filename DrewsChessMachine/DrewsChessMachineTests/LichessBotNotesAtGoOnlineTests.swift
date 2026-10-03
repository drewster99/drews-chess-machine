import XCTest
@testable import DrewsChessMachine

/// Player notes (favorites, bot-vs-bot limits, decline cool-downs) and the
/// challenge-outcome log are what the running bot consults and records into.
/// The bot can go online from the main window's status chip without its own
/// window ever opening, so going online loads them itself; and a bot limit
/// Lichess reported while the notes weren't loaded reaches the notes, and
/// their file, once they are.
@MainActor
final class LichessBotNotesAtGoOnlineTests: XCTestCase {

    private func waitUntil(_ description: String, _ condition: () throws -> Bool) async throws {
        for _ in 0..<1500 {
            if try condition() { return }
            try await Task.sleep(for: .milliseconds(20))
        }
        XCTFail("timed out waiting until \(description)")
        throw CancellationError()
    }

    private static let limitUntil = Date().addingTimeInterval(3 * 3600)

    private static var refusalText: String {
        let formatter = ISO8601DateFormatter()
        return "maia1 played 150 games against other bots today, please wait until \(formatter.string(from: limitUntil)) to challenge them."
    }

    private let request = LichessBotOutgoingChallenge(rated: false, clockLimitSeconds: 300, clockIncrementSeconds: 3, color: .random)

    /// A controller over a fresh data folder (prepared by `prepare` before
    /// the controller exists), not yet online; everything is removed after
    /// the test.
    private func makeController(lichess: LichessBotResumeFakeLichess, prepare: (LichessBotDataDirectory) throws -> Void) throws -> (controller: LichessBotController, directory: LichessBotDataDirectory) {
        let suite = "LichessBotNotesAtGoOnlineTests-\(UUID().uuidString)"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("LichessBotNotesAtGoOnlineTests-\(UUID().uuidString)", isDirectory: true)
        let directory = LichessBotDataDirectory(root: root)
        try directory.createDirectories()
        try prepare(directory)
        var settings = LichessBotSettings.testBaseline()
        settings.chat.greetingEnabled = false
        settings.connection.preventSleepWhileOnline = false
        settings.challenge.outgoingChallengeTimeoutSeconds = 0
        try LichessBotSettingsStore.save(settings, to: defaults)
        let token = LichessBotResumeFakeLichess.token
        let controller = LichessBotController(
            modelProvider: LichessBotFakeModelProvider(snapshot: nil),
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
            defaults.removePersistentDomain(forName: suite)
            if FileManager.default.fileExists(atPath: root.path) {
                do {
                    try FileManager.default.removeItem(at: root)
                } catch {
                    XCTFail("cleanup failed: \(error)")
                }
            }
        }
        return (controller, directory)
    }

    func testGoingOnlineWithoutTheBotWindowLoadsNotesAndChallengeOutcomes() async throws {
        let lichess = LichessBotResumeFakeLichess()
        let (controller, _) = try makeController(lichess: lichess) { directory in
            var notes = LichessBotPlayerNotes()
            notes.toggleFavorite("maia1")
            try notes.save(to: directory.playerNotesURL)
        }
        // The bot window's `.task` never ran: going online from the status
        // chip is the only call.
        await controller.goOnline()
        XCTAssertEqual(controller.connection, .online)
        let notes = try XCTUnwrap(controller.playerNotes, "going online loads the player notes")
        XCTAssertTrue(notes.isFavorite("maia1"), "the favorites on disk are what matchmaking sees")
        XCTAssertNotNil(controller.challengeOutcomeLog, "going online loads the challenge-outcome log")
    }

    func testABotLimitRecordedBeforeTheNotesLoadedReachesTheNotesAndTheirFile() async throws {
        let lichess = LichessBotResumeFakeLichess()
        let refusal = Self.refusalText
        lichess.challengeRefusals.modify { $0["maia1"] = refusal }
        let (controller, directory) = try makeController(lichess: lichess) { directory in
            try Data("not player notes".utf8).write(to: directory.playerNotesURL)
        }
        await controller.goOnline()
        XCTAssertEqual(controller.connection, .online)
        try await waitUntil("the event stream is open") { lichess.eventStreamsOpened.value > 0 }
        XCTAssertNil(controller.playerNotes, "the fixture's notes file is unreadable")
        let maia = LichessBotChallengeQueue.Player(username: "maia1", userID: "maia1")
        try controller.enqueueChallenges(to: [maia], request: request)
        try await waitUntil("the challenge is refused for the bot limit") { controller.botLimitEnds("maia1", now: Date()) != nil }

        // The operator repairs the file; the notes load (the bot window
        // opening, say).
        var repaired = LichessBotPlayerNotes()
        repaired.toggleFavorite("bob")
        try repaired.save(to: directory.playerNotesURL)
        await controller.loadPlayerNotes()

        let notes = try XCTUnwrap(controller.playerNotes)
        XCTAssertTrue(notes.isFavorite("bob"))
        let inNotes = try XCTUnwrap(notes.botLimitUntil["maia1"], "the limit Lichess gave reaches the loaded notes")
        XCTAssertEqual(inNotes.timeIntervalSince(Self.limitUntil), 0, accuracy: 1)
        try await waitUntil("the limit reaches the notes file") {
            try LichessBotPlayerNotes.load(from: directory.playerNotesURL).botLimitUntil["maia1"] != nil
        }
        XCTAssertTrue(try LichessBotPlayerNotes.load(from: directory.playerNotesURL).isFavorite("bob"), "the file keeps its favorites")
    }
}
