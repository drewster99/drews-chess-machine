import XCTest
@testable import DrewsChessMachine

/// `LichessBotFakeLichess`, except that withdrawing a challenge
/// (`POST /api/challenge/<id>/cancel`) waits for `release` before it is
/// answered; a withdrawal cancelled while it waits throws
/// `CancellationError`, as the real transport does.
final class LichessBotWithdrawalHoldingLichess: LichessBotTransport, @unchecked Sendable {
    let base = LichessBotFakeLichess()
    let release = LichessBotTestLatch()
    /// Challenge ids whose withdrawal reached this transport.
    let withdrawalsReceived = SyncBox<[String]>([])

    func data(for request: URLRequest) async throws -> LichessBotTransportResponse {
        let segments = (request.url?.path ?? "").split(separator: "/").map(String.init)
        if request.httpMethod == "POST", segments.count == 4, segments[0] == "api", segments[1] == "challenge", segments[3] == "cancel" {
            withdrawalsReceived.modify { $0.append(segments[2]) }
            await release.wait()
            try Task.checkCancellation()
        }
        return try await base.data(for: request)
    }

    func stream(for request: URLRequest) async throws -> (chunks: LichessBotChunkStream, response: HTTPURLResponse) {
        try await base.stream(for: request)
    }
}

/// Quitting withdraws our unanswered challenges, and shutdown waits for the
/// withdrawals, within its limit, before the process may exit: a challenge
/// left standing can be accepted into a game nobody plays.
@MainActor
final class LichessBotQuitWithdrawalTests: XCTestCase {

    private func waitUntil(_ description: String, _ condition: () -> Bool) async throws {
        for _ in 0..<3000 {
            if condition() { return }
            try await Task.sleep(for: .milliseconds(10))
        }
        XCTFail("timed out waiting until \(description)")
        throw CancellationError()
    }

    private let request = LichessBotOutgoingChallenge(rated: false, clockLimitSeconds: 300, clockIncrementSeconds: 3, color: .random)

    /// An online controller on `lichess`, with its own defaults suite and
    /// data folder, removed after the test once the latch is open.
    private func makeOnlineController(
        lichess: LichessBotWithdrawalHoldingLichess,
        configure: (inout LichessBotSettings) -> Void = { _ in },
        makeController: (UserDefaults, LichessBotDataDirectory, LichessBotControllerServices) -> LichessBotController
    ) async throws -> (controller: LichessBotController, root: URL) {
        let defaults = try makeTemporaryDefaults()
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("LichessBotQuitWithdrawalTests-\(UUID().uuidString)", isDirectory: true)
        var settings = LichessBotSettings.testBaseline()
        settings.chat.greetingEnabled = false
        settings.connection.preventSleepWhileOnline = false
        settings.challenge.outgoingChallengeTimeoutSeconds = 0
        configure(&settings)
        try LichessBotSettingsStore.save(settings, to: defaults)
        let token = LichessBotFakeLichess.token
        let controller = makeController(
            defaults,
            LichessBotDataDirectory(root: root),
            LichessBotControllerServices(
                makeTransport: { lichess },
                readToken: { _ in token }
            )
        )
        addTeardownBlock { @MainActor in
            lichess.release.open()
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
        try await waitUntil("the event stream is open") { lichess.base.eventStreamIsOpen }
        return (controller, root)
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

    func testShutdownWaitsForTheWithdrawalOfAnUnansweredChallenge() async throws {
        let lichess = LichessBotWithdrawalHoldingLichess()
        let model = try await LichessBotFakeModelProvider.randomChampion()
        let (controller, root) = try await makeOnlineController(lichess: lichess) { defaults, directory, services in
            LichessBotController(
                modelProvider: model,
                defaults: defaults,
                dataDirectory: directory,
                services: services
            )
        }
        try await controller.sendChallenge(to: "bob", request: request)
        try await waitUntil("the challenge is pending") { controller.pendingChallenges.count == 1 }
        let challengeID = LichessBotFakeLichess.challengeID(for: "bob")

        controller.abandonAndStop()
        try await waitUntil("the withdrawal reaches Lichess") { lichess.withdrawalsReceived.value == [challengeID] }
        let finished = SyncBox(false)
        let shutdown = Task { @MainActor in
            await controller.shutdown(reason: "test quit")
            finished.value = true
        }
        try await Task.sleep(for: .milliseconds(300))
        XCTAssertFalse(finished.value, "shutdown returned while the withdrawal was still unanswered")

        lichess.release.open()
        await shutdown.value
        XCTAssertTrue(try protocolMessages(root: root).contains("withdrew challenge \(challengeID) on going offline"))
    }

    func testShutdownGivesUpOnAWithdrawalAfterItsLimitAndLogsIt() async throws {
        let lichess = LichessBotWithdrawalHoldingLichess()
        let limit = Duration.milliseconds(500)
        let model = try await LichessBotFakeModelProvider.randomChampion()
        let (controller, root) = try await makeOnlineController(lichess: lichess) { defaults, directory, services in
            LichessBotController(
                modelProvider: model,
                defaults: defaults,
                dataDirectory: directory,
                services: services,
                challengeWithdrawalShutdownLimit: limit
            )
        }
        try await controller.sendChallenge(to: "bob", request: request)
        try await waitUntil("the challenge is pending") { controller.pendingChallenges.count == 1 }
        let challengeID = LichessBotFakeLichess.challengeID(for: "bob")

        controller.abandonAndStop()
        try await waitUntil("the withdrawal reaches Lichess") { lichess.withdrawalsReceived.value == [challengeID] }
        let started = ContinuousClock.now
        await controller.shutdown(reason: "test quit")
        XCTAssertLessThan(ContinuousClock.now - started, .seconds(5), "shutdown waited past its limit")
        XCTAssertTrue(
            try protocolMessages(root: root).contains("withdrawal of challenge \(challengeID) abandoned: the bot shut down before Lichess answered; the challenge may still stand"),
            "the abandoned withdrawal is logged")
    }
}
