import XCTest
@testable import DrewsChessMachine

/// A game session that ends without ever seeing its game finish — Lichess
/// closes the game stream and then answers every reopen with a 404, as it
/// does for a game it deleted — is handed to filing in the same runtime. The
/// reconciler leaves a game alone while it has a live owner, so if the
/// session's end handed nothing over, the journal would sit in
/// `InProgress/` until the next go-online's launch recovery, which may be
/// days away.
@MainActor
final class LichessBotSessionEndFilingTests: XCTestCase {

    private func waitUntil(_ description: String, timeout: Duration = .seconds(30), _ condition: () throws -> Bool) async throws {
        let deadline = ContinuousClock.now + timeout
        while ContinuousClock.now < deadline {
            if try condition() { return }
            try await Task.sleep(for: .milliseconds(20))
        }
        XCTFail("timed out waiting until \(description)")
        throw CancellationError()
    }

    private func recordURLs(in root: URL) throws -> [URL] {
        let games = LichessBotDataDirectory(root: root).gamesDirectory
        guard FileManager.default.fileExists(atPath: games.path) else { return [] }
        guard let enumerator = FileManager.default.enumerator(at: games, includingPropertiesForKeys: nil) else {
            throw CocoaError(.fileReadUnknown, userInfo: [NSFilePathErrorKey: games.path])
        }
        return enumerator.compactMap { $0 as? URL }.filter { $0.pathExtension == "json" }
    }

    func testASessionEndingWithoutAFinishIsHandedToFiling() async throws {
        let defaults = try makeTemporaryDefaults()
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("LichessBotSessionEndFilingTests-\(UUID().uuidString)", isDirectory: true)
        var settings = LichessBotSettings.testBaseline()
        settings.chat.greetingEnabled = false
        settings.chat.goodbyeEnabled = false
        settings.connection.preventSleepWhileOnline = false
        settings.connection.exportMinimumSpacingSeconds = 1
        settings.connection.reconnectInitialSeconds = 1
        settings.challenge.outgoingChallengeTimeoutSeconds = 0
        try LichessBotSettingsStore.save(settings, to: defaults)
        let lichess = LichessBotForwardingFakeLichess()
        // Lichess's export knows how the game ended; the game stream never
        // said so.
        lichess.base.resignedExports.modify { $0.insert("cbob") }
        let model = try await LichessBotFakeModelProvider.randomChampion()
        let token = LichessBotResumeFakeLichess.token
        let controller = LichessBotController(
            modelProvider: model,
            defaults: defaults,
            dataDirectory: LichessBotDataDirectory(root: root),
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
        await controller.goOnline()
        XCTAssertEqual(controller.connection, .online)
        try await waitUntil("the event stream is open") { lichess.base.eventStreamsOpened.value > 0 }
        lichess.base.startGame(against: "bob")
        try await waitUntil("cbob is in progress") { controller.activeGameIDs.contains("cbob") }
        try await waitUntil("cbob's game stream is open") { lichess.base.gameStreamIsOpen("cbob") }

        lichess.endGameStream("cbob")
        try await waitUntil("cbob's session has ended") { !controller.activeGameIDs.contains("cbob") }
        XCTAssertGreaterThan(lichess.goneGameStreamRefusals.value["cbob", default: 0], 0, "the session ended on the reopen's 404")

        // Launch recovery would file it only 30 s after go-online; the
        // session's end hands it over well before that.
        try await waitUntil("cbob is filed", timeout: .seconds(10)) { try !recordURLs(in: root).isEmpty }
        // Filing writes the record before it moves the journal out.
        let journalURL = LichessBotDataDirectory(root: root).inProgressJournalURL(gameID: "cbob")
        try await waitUntil("cbob's journal has left InProgress/", timeout: .seconds(5)) { !FileManager.default.fileExists(atPath: journalURL.path) }
    }
}
