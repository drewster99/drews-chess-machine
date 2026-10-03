import XCTest
@testable import DrewsChessMachine

/// Quitting while the bot is offline still waits for the bot's shutdown
/// when it has been used this launch. Going offline starts withdrawing our
/// unanswered challenges, and those withdrawals — and the protocol-log,
/// player-notes and outcome-log writes already queued — must finish before
/// the process exits: a challenge left standing can be accepted into a game
/// nobody plays. A launch that never touched the bot quits at once, without
/// creating its data folder.
@MainActor
final class LichessBotOfflineQuitTests: XCTestCase {

    private func waitUntil(_ description: String, _ condition: () -> Bool) async throws {
        for _ in 0..<3000 {
            if condition() { return }
            try await Task.sleep(for: .milliseconds(10))
        }
        XCTFail("timed out waiting until \(description)")
        throw CancellationError()
    }

    private let request = LichessBotOutgoingChallenge(rated: false, clockLimitSeconds: 300, clockIncrementSeconds: 3, color: .random)

    /// A controller on `transport` whose quit replies are recorded in
    /// `replies`, with its own defaults suite and data folder, removed after
    /// the test.
    private func makeController(transport: any LichessBotTransport, token: String, replies: SyncBox<[Bool]>, root: URL) throws -> LichessBotController {
        let suite = "LichessBotOfflineQuitTests-\(UUID().uuidString)"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        var settings = LichessBotSettings.testBaseline()
        settings.chat.greetingEnabled = false
        settings.connection.preventSleepWhileOnline = false
        settings.challenge.outgoingChallengeTimeoutSeconds = 0
        try LichessBotSettingsStore.save(settings, to: defaults)
        let controller = LichessBotController(
            modelProvider: LichessBotFakeModelProvider(snapshot: nil),
            defaults: defaults,
            dataDirectory: LichessBotDataDirectory(root: root),
            services: LichessBotControllerServices(
                makeTransport: { transport },
                readToken: { _ in token },
                replyToTerminate: { shouldTerminate in
                    replies.modify { $0.append(shouldTerminate) }
                }
            )
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
        return controller
    }

    /// Every protocol-log message written under `root`.
    private func protocolMessages(root: URL) throws -> [String] {
        let folder = root.appendingPathComponent("Protocol", isDirectory: true)
        guard FileManager.default.fileExists(atPath: folder.path) else { return [] }
        var messages: [String] = []
        for name in try FileManager.default.contentsOfDirectory(atPath: folder.path).sorted() where name.hasSuffix(".jsonl") {
            let url = folder.appendingPathComponent(name)
            let decoded = try LichessBotJSONLines.decode(LichessBotProtocolEntry.self, from: Data(contentsOf: url), fileName: name)
            messages += decoded.elements.map(\.message)
        }
        return messages
    }

    private func makeRoot() -> URL {
        FileManager.default.temporaryDirectory.appendingPathComponent("LichessBotOfflineQuitTests-\(UUID().uuidString)", isDirectory: true)
    }

    func testQuitWhileOfflineWaitsForChallengeWithdrawals() async throws {
        let lichess = LichessBotWithdrawalHoldingLichess()
        addTeardownBlock {
            lichess.release.open()
        }
        let replies = SyncBox<[Bool]>([])
        let root = makeRoot()
        let controller = try makeController(transport: lichess, token: LichessBotFakeLichess.token, replies: replies, root: root)
        await controller.goOnline()
        XCTAssertEqual(controller.connection, .online)
        try await waitUntil("the event stream is open") { lichess.base.eventStreamIsOpen }
        try await controller.sendChallenge(to: "bob", request: request)
        try await waitUntil("the challenge is pending") { controller.pendingChallenges.count == 1 }
        let challengeID = LichessBotFakeLichess.challengeID(for: "bob")

        await controller.goOffline()
        XCTAssertEqual(controller.connection, .offline)
        try await waitUntil("the withdrawal reaches Lichess") { lichess.withdrawalsReceived.value == [challengeID] }

        XCTAssertEqual(controller.applicationShouldTerminate(), .terminateLater, "the quit waits for the bot's shutdown")
        try await Task.sleep(for: .milliseconds(300))
        XCTAssertEqual(replies.value, [], "the quit was answered while the withdrawal was still unanswered")

        lichess.release.open()
        try await waitUntil("the quit is answered") { !replies.value.isEmpty }
        XCTAssertEqual(replies.value, [true])
        XCTAssertTrue(controller.isShutDown)
        XCTAssertTrue(try protocolMessages(root: root).contains("withdrew challenge \(challengeID) on going offline"), "the withdrawal finished before the quit was answered")
    }

    func testQuitWithoutEverUsingTheBotIsImmediate() async throws {
        let replies = SyncBox<[Bool]>([])
        let root = makeRoot()
        let controller = try makeController(transport: LichessBotResumeFakeLichess(), token: LichessBotResumeFakeLichess.token, replies: replies, root: root)
        XCTAssertEqual(controller.applicationShouldTerminate(), .terminateNow)
        XCTAssertEqual(replies.value, [])
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.path), "a launch that never used the bot leaves no data folder")
    }
}
