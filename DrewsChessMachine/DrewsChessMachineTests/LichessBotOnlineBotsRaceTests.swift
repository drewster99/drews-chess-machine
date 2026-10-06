import XCTest
@testable import DrewsChessMachine

/// `LichessBotFakeLichess`, except that `GET /api/bot/online` is held in
/// the transport until the test opens `release`, so a test can keep the
/// online-bots fetch in flight while matchmaking runs.
final class LichessBotOnlineBotsHeldLichess: LichessBotTransport, @unchecked Sendable {
    let lichess: LichessBotFakeLichess
    let release = LichessBotTestLatch()
    /// `GET /api/bot/online` requests that have reached the transport.
    let onlineBotsRequests = SyncBox<Int>(0)

    init(lichess: LichessBotFakeLichess) {
        self.lichess = lichess
    }

    func data(for request: URLRequest) async throws -> LichessBotTransportResponse {
        let path = request.url.flatMap { URLComponents(url: $0, resolvingAgainstBaseURL: true) }?.path
        if request.httpMethod == "GET", path == "/api/bot/online" {
            onlineBotsRequests.modify { $0 += 1 }
            await release.wait()
        }
        return try await lichess.data(for: request)
    }

    func stream(for request: URLRequest) async throws -> (chunks: LichessBotChunkStream, response: HTTPURLResponse) {
        try await lichess.stream(for: request)
    }
}

/// A matchmaking pass that starts while the online-bots fetch is in flight
/// waits for that fetch instead of deciding without a list. Going Online
/// with matchmaking on starts the first automatic pass and the first
/// online-bots fetch on the same poll; a pass that skipped the fetch in
/// flight would find no list, report "failed: the online-bots list has not
/// loaded", and wait a minute before trying again.
@MainActor
final class LichessBotOnlineBotsRaceTests: XCTestCase {

    private func waitUntil(_ description: String, _ condition: () -> Bool) async throws {
        for _ in 0..<3000 {
            if condition() { return }
            try await Task.sleep(for: .milliseconds(10))
        }
        XCTFail("timed out waiting until \(description)")
        throw CancellationError()
    }

    func testAPassStartedDuringTheOnlineBotsFetchWaitsForItAndSends() async throws {
        let lichess = LichessBotFakeLichess(accountPerfsJSON: #"{"blitz":{"games":100,"rating":1500,"rd":50,"prog":0}}"#)
        lichess.onlineBotsNDJSON.value = #"{"id":"fitbot","username":"FitBot","title":"BOT","perfs":{"blitz":{"games":50,"rating":1600,"rd":60,"prog":0}}}"# + "\n"
        let held = LichessBotOnlineBotsHeldLichess(lichess: lichess)

        let defaults = try makeTemporaryDefaults()
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("LichessBotOnlineBotsRaceTests-\(UUID().uuidString)", isDirectory: true)
        var settings = LichessBotSettings.testBaseline()
        settings.chat.greetingEnabled = false
        settings.connection.preventSleepWhileOnline = false
        settings.challenge.outgoingChallengeTimeoutSeconds = 0
        settings.challenge.maxConcurrentGames = 2
        settings.challenge.gamesReservedForHumans = 1
        // On: the poll loop starts the automatic pass and the fetch itself.
        settings.matchmaking.enabled = true
        settings.matchmaking.fillMode = .everyFreeSlot
        settings.matchmaking.timeControls = [.blitz5plus3]
        settings.matchmaking.rated = false
        try LichessBotSettingsStore.save(settings, to: defaults)
        let token = LichessBotFakeLichess.token
        let controller = LichessBotController(
            modelProvider: try await LichessBotFakeModelProvider.randomChampion(),
            defaults: defaults,
            dataDirectory: LichessBotDataDirectory(root: root),
            services: LichessBotControllerServices(
                makeTransport: { held },
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
        // Registered last, so it runs first: never leave a request parked in
        // the transport.
        addTeardownBlock { @MainActor in
            held.release.open()
        }

        await controller.loadPlayerNotes()
        await controller.goOnline()
        XCTAssertEqual(controller.connection, .online)
        try await waitUntil("the online-bots fetch is in flight") { held.onlineBotsRequests.value == 1 }
        // Long enough for the automatic pass started with the fetch to have
        // decided, had it not waited for the fetch.
        try await Task.sleep(for: .milliseconds(1500))
        let statusWhileHeld = controller.matchmakingStatus
        XCTAssertFalse(statusWhileHeld?.contains("online-bots") == true, "the pass decided without waiting for the fetch: \(statusWhileHeld ?? "no status")")
        XCTAssertEqual(lichess.challengedNames.value, [], "nothing can be picked before the list arrives")

        held.release.open()
        try await waitUntil("the pass reports challenging FitBot from the fetched list") {
            controller.matchmakingStatus?.hasSuffix("sent a challenge to FitBot") == true
        }
        XCTAssertEqual(lichess.challengedNames.value, ["FitBot"])
        XCTAssertEqual(controller.pendingChallenges.map(\.username), ["FitBot"])
        XCTAssertNotNil(controller.onlineBotsFetchedAt)
        XCTAssertEqual(held.onlineBotsRequests.value, 1, "the pass joined the fetch under way rather than starting another")
    }
}
