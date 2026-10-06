import XCTest
@testable import DrewsChessMachine

/// `LichessBotResumeFakeLichess`, forwarded, recording what the model
/// provider had been asked for at the moment the event stream was requested:
/// the bot must hold a built model generation before it listens for
/// challenges (follow-lineage plan §3.10).
final class LichessBotModelFirstFakeLichess: LichessBotTransport, @unchecked Sendable {
    let base = LichessBotResumeFakeLichess()
    /// The provider's snapshot count, read when an event-stream request
    /// reaches this transport.
    private let snapshotCount: @Sendable () -> Int
    /// One entry per event-stream request: the snapshot count at that moment.
    let snapshotCountsAtEventStreamRequests = SyncBox<[Int]>([])

    init(snapshotCount: @escaping @Sendable () -> Int) {
        self.snapshotCount = snapshotCount
    }

    var eventStreamRequestCount: Int {
        snapshotCountsAtEventStreamRequests.value.count
    }

    func data(for request: URLRequest) async throws -> LichessBotTransportResponse {
        try await base.data(for: request)
    }

    func stream(for request: URLRequest) async throws -> (chunks: LichessBotChunkStream, response: HTTPURLResponse) {
        if request.url?.path == "/api/stream/event" {
            let count = snapshotCount()
            snapshotCountsAtEventStreamRequests.modify { $0.append(count) }
        }
        return try await base.stream(for: request)
    }

    /// Challenge responses and challenges sent: every POST except the token
    /// check.
    var challengeTraffic: [String] {
        base.postedPaths.value.filter { $0.hasPrefix("/api/challenge") }
    }
}

/// Going online builds the model generation before anything that talks to
/// Lichess's streams, for every source; a failed build leaves the bot
/// offline with the error (follow-lineage plan §3.10, OD-18, OD-19).
@MainActor
final class LichessBotGoOnlineModelFirstTests: XCTestCase {

    private func waitUntil(_ description: String, _ condition: () throws -> Bool) async throws {
        for _ in 0..<3000 {
            if try condition() { return }
            try await Task.sleep(for: .milliseconds(10))
        }
        XCTFail("timed out waiting until \(description)")
        throw CancellationError()
    }

    private struct Installation {
        let defaults: UserDefaults
        let root: URL
        var directory: LichessBotDataDirectory { LichessBotDataDirectory(root: root) }
    }

    /// A defaults suite holding `settings` and a fresh data folder, removed
    /// after the test.
    private func makeInstallation(configure: (inout LichessBotSettings) -> Void = { _ in }) throws -> Installation {
        let defaults = try makeTemporaryDefaults()
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("LichessBotGoOnlineModelFirstTests-\(UUID().uuidString)", isDirectory: true)
        var settings = LichessBotSettings.testBaseline()
        settings.chat.greetingEnabled = false
        settings.chat.goodbyeEnabled = false
        settings.connection.preventSleepWhileOnline = false
        settings.challenge.outgoingChallengeTimeoutSeconds = 0
        configure(&settings)
        try LichessBotSettingsStore.save(settings, to: defaults)
        addTeardownBlock {
            if FileManager.default.fileExists(atPath: root.path) {
                do {
                    try FileManager.default.removeItem(at: root)
                } catch {
                    XCTFail("cleanup failed: \(error)")
                }
            }
        }
        return Installation(defaults: defaults, root: root)
    }

    /// A controller over the installation, not yet online; shut down after
    /// the test.
    private func makeController(_ installation: Installation, transport: any LichessBotTransport, modelProvider: any LichessBotModelProvider) -> LichessBotController {
        let token = LichessBotResumeFakeLichess.token
        let controller = LichessBotController(
            modelProvider: modelProvider,
            defaults: installation.defaults,
            dataDirectory: installation.directory,
            services: LichessBotControllerServices(
                makeTransport: { transport },
                readToken: { _ in token }
            ),
            finishedGameHold: LichessBotController.finishedGameHold
        )
        addTeardownBlock { @MainActor in
            controller.abandonAndStop()
            await controller.shutdown(reason: "test teardown")
        }
        return controller
    }

    /// Waits until the instance lock can be taken (the failed start releases
    /// it on the journal queue), then gives it back.
    private func assertInstanceLockReleased(_ installation: Installation) async throws {
        var lastError: Error?
        for _ in 0..<500 {
            do {
                let lock = try LichessBotInstanceLock.acquire(at: installation.directory.lockURL, holder: .current)
                lock.release()
                return
            } catch {
                lastError = error
            }
            try await Task.sleep(for: .milliseconds(10))
        }
        XCTFail("the instance lock was never released: \(String(describing: lastError))")
    }

    // MARK: - Written before the change (they fail on the lazy build)

    func testGoingOnlineBuildsTheModelBeforeOpeningTheEventStream() async throws {
        let installation = try makeInstallation()
        let provider = try await LichessBotFakeModelProvider.randomChampion()
        let lichess = LichessBotModelFirstFakeLichess(snapshotCount: { provider.snapshotCount.value })
        let controller = makeController(installation, transport: lichess, modelProvider: provider)
        await controller.goOnline()
        XCTAssertEqual(controller.connection, .online)
        try await waitUntil("the event stream is requested") { lichess.eventStreamRequestCount > 0 }
        XCTAssertEqual(lichess.snapshotCountsAtEventStreamRequests.value.first, 1, "the champion was snapshotted (and its generation built) before the event stream was opened")
    }

    func testModelBuildFailureLeavesTheBotOfflineWithTheError() async throws {
        let installation = try makeInstallation()
        let provider = LichessBotFakeModelProvider(snapshot: nil)
        let lichess = LichessBotModelFirstFakeLichess(snapshotCount: { provider.snapshotCount.value })
        let controller = makeController(installation, transport: lichess, modelProvider: provider)
        await controller.goOnline()
        guard case .error(let text) = controller.connection else {
            return XCTFail("expected Error, got \(controller.connection)")
        }
        XCTAssertTrue(text.contains("No champion"), text)
        XCTAssertFalse(controller.isRunning)
        XCTAssertEqual(lichess.eventStreamRequestCount, 0, "no event stream is opened without a model")
        XCTAssertEqual(lichess.challengeTraffic, [], "nothing is accepted, declined or sent")
        try await assertInstanceLockReleased(installation)
    }

    func testModelBuildFailureKeepsTheLeftoverGamesReport() async throws {
        let installation = try makeInstallation()
        try installation.directory.createDirectories()
        // A journal with no lines yet: a game that never recorded a finish.
        try Data().write(to: installation.directory.inProgressJournalURL(gameID: "cbob"))
        let provider = LichessBotFakeModelProvider(snapshot: nil)
        let lichess = LichessBotModelFirstFakeLichess(snapshotCount: { provider.snapshotCount.value })
        let controller = makeController(installation, transport: lichess, modelProvider: provider)
        await controller.noteLeftoverJournalsAtLaunch()
        XCTAssertEqual(controller.leftoverGamesFromLastRun, ["cbob"])
        await controller.goOnline()
        guard case .error = controller.connection else {
            return XCTFail("expected Error, got \(controller.connection)")
        }
        XCTAssertEqual(controller.leftoverGamesFromLastRun, ["cbob"], "a start that failed resumed nothing, so the report stands")
    }

    // MARK: - Cancelling, settings and shutdown while the model builds

    /// Goes online in the background with the provider's snapshots held, and
    /// waits until the model build is waiting on them.
    private func goOnlineHeld(_ controller: LichessBotController, provider: LichessBotHoldableModelProvider) async throws -> Task<Void, Never> {
        provider.holdSnapshots()
        let goingOnline = Task { @MainActor in
            await controller.goOnline()
        }
        try await waitUntil("the model build is held") { provider.snapshotsHeld.value == 1 }
        return goingOnline
    }

    func testGoOfflineWhileThePreparingModelCancelsGoingOnline() async throws {
        let installation = try makeInstallation()
        let provider = try await LichessBotHoldableModelProvider.make()
        let lichess = LichessBotModelFirstFakeLichess(snapshotCount: { provider.championSnapshots.value })
        let controller = makeController(installation, transport: lichess, modelProvider: provider)
        let goingOnline = try await goOnlineHeld(controller, provider: provider)
        XCTAssertEqual(controller.connection, .connecting)
        XCTAssertTrue(controller.canGoOffline, "Go Offline is enabled while going online")

        await controller.goOffline()
        XCTAssertEqual(controller.connection, .connecting, "cancelling waits for the step in flight")
        XCTAssertEqual(controller.goingOnlineStatusText, "Cancelling…")
        await controller.goOffline()
        XCTAssertEqual(controller.connection, .connecting, "a second press changes nothing")

        provider.releaseSnapshots()
        await goingOnline.value
        XCTAssertEqual(controller.connection, .offline, "a cancel is not an error")
        XCTAssertFalse(controller.isRunning)
        XCTAssertNil(controller.generation, "the late build is discarded")
        XCTAssertNil(controller.goingOnlineStep)
        XCTAssertFalse(controller.goingOnlineCancelRequested)
        XCTAssertEqual(lichess.eventStreamRequestCount, 0)
        XCTAssertEqual(lichess.challengeTraffic, [])
        try await assertInstanceLockReleased(installation)

        await controller.goOnline()
        XCTAssertEqual(controller.connection, .online, "the cancel did not reach the next attempt")
    }

    func testGoOfflineAfterThePreparedModelStillCancels() async throws {
        let installation = try makeInstallation()
        let provider = try await LichessBotHoldableModelProvider.make()
        let lichess = LichessBotModelFirstFakeLichess(snapshotCount: { provider.championSnapshots.value })
        let controller = makeController(installation, transport: lichess, modelProvider: provider)
        let goingOnline = try await goOnlineHeld(controller, provider: provider)

        // Hold the journal queue, so going online stops at its read of the
        // leftover journals (seeding today's counts), after the model.
        let journalHold = DispatchSemaphore(value: 0)
        let journalHeld = SyncBox(false)
        controller.journalQueue.enqueue("test: hold the journal queue") {
            journalHeld.value = true
            journalHold.wait()
        }
        try await waitUntil("the journal queue is held") { journalHeld.value }
        provider.releaseSnapshots()
        try await waitUntil("the model is prepared") { controller.goingOnlineStep == .startingSession }

        await controller.goOffline()
        XCTAssertEqual(controller.connection, .connecting)
        journalHold.signal()
        await goingOnline.value
        XCTAssertEqual(controller.connection, .offline)
        XCTAssertFalse(controller.isRunning)
        XCTAssertEqual(lichess.eventStreamRequestCount, 0, "a press after the model was built still stops before any stream opens")
        try await assertInstanceLockReleased(installation)
    }

    func testSettingsAppliedWhilePreparingReachTheRuntime() async throws {
        let installation = try makeInstallation()
        let provider = try await LichessBotHoldableModelProvider.make()
        let lichess = LichessBotModelFirstFakeLichess(snapshotCount: { provider.championSnapshots.value })
        let controller = makeController(installation, transport: lichess, modelProvider: provider)
        XCTAssertTrue(controller.settings.challenge.acceptBots, "the baseline accepts bots")
        let goingOnline = try await goOnlineHeld(controller, provider: provider)

        var changed = controller.settings
        changed.challenge.acceptBots = false
        try controller.updateSettings(changed)
        provider.releaseSnapshots()
        await goingOnline.value
        XCTAssertEqual(controller.connection, .online)
        try await waitUntil("the event stream is open") { lichess.base.eventStreamIsOpen }

        lichess.base.sendEvent(#"{"type":"challenge","challenge":{"id":"cbot","status":"created","challenger":{"id":"botty","name":"botty","title":"BOT","rating":1500},"variant":{"key":"standard","name":"x","short":"x"},"rated":false,"speed":"blitz","timeControl":{"type":"clock","limit":300,"increment":3,"show":"5+3"},"color":"random","direction":"in"},"compat":{"bot":true,"board":true}}"#)
        try await waitUntil("the challenge is answered") { !lichess.challengeTraffic.isEmpty }
        XCTAssertEqual(lichess.challengeTraffic, ["/api/challenge/cbot/decline"], "the bot runs on the settings applied while its model was built")
    }

    func testShutdownWhilePreparingStopsGoingOnline() async throws {
        let installation = try makeInstallation()
        let provider = try await LichessBotHoldableModelProvider.make()
        let lichess = LichessBotModelFirstFakeLichess(snapshotCount: { provider.championSnapshots.value })
        let controller = makeController(installation, transport: lichess, modelProvider: provider)
        let goingOnline = try await goOnlineHeld(controller, provider: provider)

        await controller.shutdown(reason: "test quit while preparing")
        provider.releaseSnapshots()
        await goingOnline.value
        XCTAssertEqual(controller.connection, .error("The bot has shut down"))
        XCTAssertFalse(controller.isRunning)
        XCTAssertEqual(lichess.eventStreamRequestCount, 0)
    }

    func testGoingOnlineStepIsReportedWhilePreparing() async throws {
        let installation = try makeInstallation()
        let provider = try await LichessBotHoldableModelProvider.make()
        let lichess = LichessBotModelFirstFakeLichess(snapshotCount: { provider.championSnapshots.value })
        let controller = makeController(installation, transport: lichess, modelProvider: provider)
        let goingOnline = try await goOnlineHeld(controller, provider: provider)

        guard case .preparingModel(.champion, _) = controller.goingOnlineStep else {
            provider.releaseSnapshots()
            await goingOnline.value
            return XCTFail("expected the model step, got \(String(describing: controller.goingOnlineStep))")
        }
        let status = try XCTUnwrap(controller.goingOnlineStatusText)
        XCTAssertTrue(status.hasPrefix("Preparing model: Champion"), status)

        provider.releaseSnapshots()
        await goingOnline.value
        XCTAssertEqual(controller.connection, .online)
        XCTAssertNil(controller.goingOnlineStep)
        XCTAssertNil(controller.goingOnlineStatusText)
    }

    func testEverySourceIsBuiltBeforeGoingOnline() async throws {
        for source in [LichessBotModelSourceKind.champion, .trainerSnapshot, .liveTrainer] {
            let installation = try makeInstallation { settings in
                settings.model.source = source
            }
            let provider = try await LichessBotHoldableModelProvider.make()
            let lichess = LichessBotModelFirstFakeLichess(snapshotCount: {
                source == .champion ? provider.championSnapshots.value : provider.trainerSnapshots.value
            })
            let controller = makeController(installation, transport: lichess, modelProvider: provider)
            await controller.goOnline()
            XCTAssertEqual(controller.connection, .online, "\(source)")
            try await waitUntil("the event stream is requested (\(source))") { lichess.eventStreamRequestCount > 0 }
            XCTAssertEqual(lichess.snapshotCountsAtEventStreamRequests.value.first, 1, "\(source): built before the event stream")
            try await waitUntil("the poll publishes the generation (\(source))") { controller.generation != nil }
            XCTAssertEqual(controller.generation?.sourceKind, source)
        }

        // A file: a missing one stops going online before any stream (the
        // build comes first); a real one plays.
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent("LichessBotGoOnlineModelFirstTests-models-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: false)
        addTeardownBlock {
            do {
                try FileManager.default.removeItem(at: folder)
            } catch {
                XCTFail("cleanup failed: \(error)")
            }
        }
        let missing = folder.appendingPathComponent("absent.safetensors")
        let missingInstallation = try makeInstallation { settings in
            settings.model.source = .file
            settings.model.filePath = missing.path
        }
        let missingProvider = try await LichessBotHoldableModelProvider.make()
        let missingLichess = LichessBotModelFirstFakeLichess(snapshotCount: { 0 })
        let missingController = makeController(missingInstallation, transport: missingLichess, modelProvider: missingProvider)
        await missingController.goOnline()
        guard case .error = missingController.connection else {
            return XCTFail("a missing model file must stop going online, got \(missingController.connection)")
        }
        XCTAssertEqual(missingLichess.eventStreamRequestCount, 0)

        let file = folder.appendingPathComponent("model.safetensors")
        try await LichessBotTestModelFiles.writeRandomModel(to: file)
        let fileInstallation = try makeInstallation { settings in
            settings.model.source = .file
            settings.model.filePath = file.path
        }
        let fileProvider = try await LichessBotHoldableModelProvider.make()
        let fileLichess = LichessBotModelFirstFakeLichess(snapshotCount: { 0 })
        let fileController = makeController(fileInstallation, transport: fileLichess, modelProvider: fileProvider)
        await fileController.goOnline()
        XCTAssertEqual(fileController.connection, .online)
        try await waitUntil("the poll publishes the file's generation") { fileController.generation != nil }
        XCTAssertEqual(fileController.generation?.sourceKind, .file)
        XCTAssertEqual(fileController.generation?.filePath, file.path)
        XCTAssertEqual(fileController.generation?.generationID, 1)
    }
}
