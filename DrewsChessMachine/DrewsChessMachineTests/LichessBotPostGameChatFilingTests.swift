import XCTest
@testable import DrewsChessMachine

/// Chat sent after a game ends (an opponent's "gg") reaches the game's
/// record: Lichess closes the game stream at the finish, so those lines come
/// only from the post-game chat fetches, and the game is filed only after the
/// last of them.
@MainActor
final class LichessBotPostGameChatFilingTests: XCTestCase {

    private func waitUntil(_ description: String, _ condition: () -> Bool) async throws {
        for _ in 0..<3000 {
            if condition() { return }
            try await Task.sleep(for: .milliseconds(10))
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

    func testChatFetchedAfterTheFirstPostGameFetchReachesTheRecord() async throws {
        let defaults = try makeTemporaryDefaults()
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("LichessBotPostGameChatFilingTests-\(UUID().uuidString)", isDirectory: true)
        var settings = LichessBotSettings.testBaseline()
        settings.chat.greetingEnabled = false
        settings.chat.goodbyeEnabled = false
        settings.connection.preventSleepWhileOnline = false
        settings.connection.exportMinimumSpacingSeconds = 1
        settings.challenge.outgoingChallengeTimeoutSeconds = 0
        try LichessBotSettingsStore.save(settings, to: defaults)
        let lichess = LichessBotResumeFakeLichess()
        // Nothing yet at the first fetch; the opponent's "gg" by the second.
        lichess.chatFetchResponses.modify { $0["cbob"] = ["[]", #"[{"text":"gg","user":"bob"}]"#] }
        lichess.resignedExports.modify { $0.insert("cbob") }
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
            finishedGameHold: LichessBotController.finishedGameHold,
            postGameChatFetchDelays: [.milliseconds(300), .milliseconds(1500)]
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
        try await waitUntil("cbob is listed") { controller.games.count == 1 }
        try await waitUntil("cbob's game stream is open") { lichess.gameStreamIsOpen("cbob") }
        lichess.opponentResigns("bob")

        var records: [URL] = []
        for _ in 0..<1500 where records.isEmpty {
            records = try recordURLs(in: root)
            try await Task.sleep(for: .milliseconds(20))
        }
        let recordURL = try XCTUnwrap(records.first, "the game was never filed")
        let record = try LichessBotIndex.readRecord(at: recordURL)
        XCTAssertTrue(record.chat.contains { $0.username == "bob" && $0.text == "gg" }, "\(record.chat)")
    }
}
