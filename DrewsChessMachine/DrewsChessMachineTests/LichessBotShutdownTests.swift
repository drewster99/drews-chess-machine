import XCTest
@testable import DrewsChessMachine

/// A Lichess that holds every request until the test opens `release`, then
/// answers 404, so a test can keep a request in flight across a shutdown.
/// Streams are not offered.
final class LichessBotHeldLichess: LichessBotTransport, @unchecked Sendable {
    static let token = "lip_SHUTDOWNTEST"

    enum HeldLichessError: Error {
        case streamsNotOffered
    }

    let release = LichessBotTestLatch()
    /// Requests that have reached the transport (and are held there until
    /// `release` opens).
    let requestsReceived = SyncBox<Int>(0)

    func data(for request: URLRequest) async throws -> LichessBotTransportResponse {
        requestsReceived.modify { $0 += 1 }
        await release.wait()
        return LichessBotTransportResponse(body: Data(#"{"error":"Not found"}"#.utf8), response: try lichessBotTestResponse(status: 404), networkProtocolName: "h2")
    }

    func stream(for request: URLRequest) async throws -> (chunks: LichessBotChunkStream, response: HTTPURLResponse) {
        throw HeldLichessError.streamsNotOffered
    }
}

/// The controller's shutdown and the file queue's close: work queued before
/// them reaches the disk, and nothing reaches the data folder after them,
/// even from work that was already under way (a request in flight whose
/// gate events and request record are protocol-log appends). Without this,
/// deleting a test's temporary data folder raced those appends, and an
/// append recreating `Protocol/` mid-removal failed the cleanup.
@MainActor
final class LichessBotShutdownTests: XCTestCase {

    private func waitUntil(_ description: String, _ condition: () -> Bool) async throws {
        for _ in 0..<3000 {
            if condition() { return }
            try await Task.sleep(for: .milliseconds(10))
        }
        XCTFail("timed out waiting until \(description)")
        throw CancellationError()
    }

    private func makeTemporaryRoot() -> URL {
        FileManager.default.temporaryDirectory.appendingPathComponent("LichessBotShutdownTests-\(UUID().uuidString)", isDirectory: true)
    }

    private func removeIfPresent(_ root: URL) {
        guard FileManager.default.fileExists(atPath: root.path) else { return }
        do {
            try FileManager.default.removeItem(at: root)
        } catch {
            XCTFail("cleanup failed: \(error)")
        }
    }

    /// A controller with its own defaults suite and data folder (both
    /// removed after the test) and the given transport and stored token.
    private func makeController(root: URL, transport: any LichessBotTransport, token: String?) throws -> LichessBotController {
        let suite = "LichessBotShutdownTests-\(UUID().uuidString)"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        try LichessBotSettingsStore.save(LichessBotSettings.testBaseline(), to: defaults)
        let controller = LichessBotController(
            modelProvider: LichessBotFakeModelProvider(snapshot: nil),
            defaults: defaults,
            dataDirectory: LichessBotDataDirectory(root: root),
            services: LichessBotControllerServices(
                makeTransport: { transport },
                readToken: { _ in token }
            )
        )
        addTeardownBlock { @MainActor in
            await controller.shutdown(reason: "test teardown")
            defaults.removePersistentDomain(forName: suite)
            self.removeIfPresent(root)
        }
        return controller
    }

    /// Every protocol-log message on disk from `start` to now, read straight
    /// from the day files (the log's own reader goes through the file queue,
    /// which a shutdown closes).
    private func protocolMessages(_ log: LichessBotProtocolLog, since start: Date) throws -> [String] {
        let urls = Set([log.fileURL(for: start), log.fileURL(for: Date())]).sorted { $0.path < $1.path }
        var messages: [String] = []
        for url in urls where FileManager.default.fileExists(atPath: url.path) {
            let decoded = try LichessBotJSONLines.decode(LichessBotProtocolEntry.self, from: Data(contentsOf: url), fileName: url.lastPathComponent)
            messages += decoded.elements.map(\.message)
        }
        return messages
    }

    /// Work enqueued before `close` runs; work after it is refused — `run`
    /// throws, `enqueue` doesn't run its body — and writes nothing.
    func testAClosedFileQueueRunsEarlierWorkAndRefusesLaterWork() async throws {
        let root = makeTemporaryRoot()
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        addTeardownBlock { @MainActor in
            self.removeIfPresent(root)
        }
        let beforeURL = root.appendingPathComponent("before-close.txt", isDirectory: false)
        let enqueuedAfterURL = root.appendingPathComponent("enqueued-after-close.txt", isDirectory: false)
        let runAfterURL = root.appendingPathComponent("run-after-close.txt", isDirectory: false)
        let writeFailures = SyncBox<[String]>([])
        let queue = LichessBotFileQueue()

        queue.enqueue("test write before close") {
            do {
                try Data("before".utf8).write(to: beforeURL)
            } catch {
                writeFailures.modify { $0.append("before close: \(error)") }
            }
        }
        await queue.close(reason: "test")
        queue.enqueue("test write after close") {
            do {
                try Data("after".utf8).write(to: enqueuedAfterURL)
            } catch {
                writeFailures.modify { $0.append("after close: \(error)") }
            }
        }
        // Runs after the enqueued body's turn (the queue is serial), so when
        // it returns, that body has been refused or run.
        do {
            try await queue.run {
                try Data("after".utf8).write(to: runAfterURL)
            }
            XCTFail("a closed queue ran work")
        } catch LichessBotFileQueueError.closed(_, let reason) {
            XCTAssertEqual(reason, "test")
        }
        // Closing again keeps the first reason.
        await queue.close(reason: "second close")
        do {
            try await queue.run {}
            XCTFail("a closed queue ran work")
        } catch LichessBotFileQueueError.closed(_, let reason) {
            XCTAssertEqual(reason, "test")
        }

        XCTAssertEqual(writeFailures.value, [])
        XCTAssertTrue(FileManager.default.fileExists(atPath: beforeURL.path), "work enqueued before the close runs")
        XCTAssertFalse(FileManager.default.fileExists(atPath: enqueuedAfterURL.path), "enqueued work after the close is refused")
        XCTAssertFalse(FileManager.default.fileExists(atPath: runAfterURL.path), "run after the close is refused")
    }

    /// The late-write path that failed test cleanups: a request in flight
    /// when the controller shuts down finishes afterwards, and its gate
    /// events and request record are protocol-log appends. Shutdown must
    /// write what was recorded before it, then let nothing reach the folder
    /// — here the folder is deleted right after the shutdown, and must not
    /// come back when the request finishes.
    func testARequestFinishingAfterShutdownWritesNothing() async throws {
        let root = makeTemporaryRoot()
        let lichess = LichessBotHeldLichess()
        let controller = try makeController(root: root, transport: lichess, token: LichessBotHeldLichess.token)
        addTeardownBlock { @MainActor in
            // Never leave a request parked in the transport.
            lichess.release.open()
        }
        let startedAt = Date()
        // Favorites aren't loaded, so this raises an alarm: a protocol event
        // recorded before the shutdown.
        controller.toggleFavorite("x")
        controller.loadOpponentProfile("alice")
        try await waitUntil("the profile request is in flight") { lichess.requestsReceived.value == 1 }

        await controller.shutdown(reason: "test")

        let messages = try protocolMessages(controller.protocolLog, since: startedAt)
        XCTAssertTrue(messages.contains("Favorites aren't loaded; not changing them"), "an event recorded before the shutdown is on disk: \(messages)")
        XCTAssertTrue(messages.contains("shutting down: test"), "the shutdown's own event is on disk: \(messages)")
        try FileManager.default.removeItem(at: root)

        lichess.release.open()
        try await waitUntil("the profile request has finished") {
            controller.opponentProfiles["alice"] != LichessBotController.OpponentProfile.loading
        }
        // The request's records were enqueued before the profile changed, so
        // this runs after their turn: when it throws, they have been refused.
        do {
            try await controller.protocolLog.flush()
            XCTFail("the file queue ran work after the shutdown")
        } catch is LichessBotFileQueueError {
            // Expected: the queue closed with the shutdown.
        }
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.path), "a write reached the data folder after the shutdown")
    }

    /// A controller that has shut down refuses to go online, visibly.
    func testAShutDownControllerDoesNotGoOnline() async throws {
        let lichess = LichessBotFakeLichess()
        let controller = try makeController(root: makeTemporaryRoot(), transport: lichess, token: LichessBotFakeLichess.token)
        await controller.shutdown(reason: "test")
        await controller.goOnline()
        XCTAssertEqual(controller.connection, .error("The bot has shut down"))
        XCTAssertFalse(lichess.eventStreamIsOpen)
        XCTAssertFalse(controller.isRunning)
    }
}
