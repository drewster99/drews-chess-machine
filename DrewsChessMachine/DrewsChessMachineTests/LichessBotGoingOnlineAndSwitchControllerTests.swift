import XCTest
@testable import DrewsChessMachine

/// Going online and switching model sources, seen from the controller
/// (follow-lineage plan §3.10; review of the ready-before-play change):
/// - a cancel late in going online leaves the leftover-games report up;
/// - a failed source switch publishes its failure, alarms and backs off
///   while the old generation keeps playing;
/// - a model-settings change clears the old failure at once and switches at
///   the next tick, and a success clears the failure.
@MainActor
final class LichessBotGoingOnlineAndSwitchControllerTests: XCTestCase {

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

    private func makeInstallation(configure: (inout LichessBotSettings) -> Void = { _ in }) throws -> Installation {
        let defaults = try makeTemporaryDefaults()
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("LichessBotGoingOnlineAndSwitchControllerTests-\(UUID().uuidString)", isDirectory: true)
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

    private func makeController(
        _ installation: Installation,
        modelProvider: any LichessBotModelProvider,
        goingOnlineStepBegan: @escaping @MainActor @Sendable (LichessBotGoingOnlineStep) async -> Void = { _ in }
    ) -> (LichessBotController, LichessBotModelFirstFakeLichess) {
        let lichess = LichessBotModelFirstFakeLichess(snapshotCount: { 0 })
        let token = LichessBotResumeFakeLichess.token
        var services = LichessBotControllerServices(
            makeTransport: { lichess },
            readToken: { _ in token }
        )
        services.goingOnlineStepBegan = goingOnlineStepBegan
        let controller = LichessBotController(
            modelProvider: modelProvider,
            defaults: installation.defaults,
            dataDirectory: installation.directory,
            services: services,
            finishedGameHold: LichessBotController.finishedGameHold
        )
        addTeardownBlock { @MainActor in
            controller.abandonAndStop()
            await controller.shutdown(reason: "test teardown")
        }
        return (controller, lichess)
    }

    // MARK: - Cancelling late in going online

    /// Regression: going online cleared the leftover-games report before its
    /// last stop checks, so a Go Offline while it read today's games lost
    /// the report of a game that may still be running on Lichess.
    func testCancelWhileReadingTodaysGamesKeepsTheLeftoverGamesReport() async throws {
        let installation = try makeInstallation()
        try installation.directory.createDirectories()
        // A journal with no lines yet: a game that never recorded a finish.
        try Data().write(to: installation.directory.inProgressJournalURL(gameID: "cbob"))
        let provider = try await LichessBotHoldableModelProvider.make()
        let hold = LichessBotTestLatch()
        let held = SyncBox(false)
        let (controller, lichess) = makeController(installation, modelProvider: provider, goingOnlineStepBegan: { step in
            guard step == .readingTodaysGames else { return }
            held.value = true
            await hold.wait()
        })
        await controller.noteLeftoverJournalsAtLaunch()
        XCTAssertEqual(controller.leftoverGamesFromLastRun, ["cbob"])

        let goingOnline = Task { @MainActor in
            await controller.goOnline()
        }
        try await waitUntil("going online reads today's games") { held.value }
        await controller.goOffline()
        hold.open()
        await goingOnline.value

        XCTAssertEqual(controller.connection, .offline)
        XCTAssertFalse(controller.isRunning)
        XCTAssertEqual(lichess.eventStreamRequestCount, 0)
        XCTAssertEqual(controller.leftoverGamesFromLastRun, ["cbob"], "a cancelled start resumed nothing, so the report stands")
    }

    // MARK: - Source switches in the poll loop

    private func goOnlineOnChampion() async throws -> (LichessBotController, LichessBotHoldableModelProvider) {
        let installation = try makeInstallation()
        let provider = try await LichessBotHoldableModelProvider.make()
        let (controller, _) = makeController(installation, modelProvider: provider)
        await controller.goOnline()
        XCTAssertEqual(controller.connection, .online)
        try await waitUntil("the poll publishes the champion's generation") { controller.generation?.sourceKind == .champion }
        return (controller, provider)
    }

    private func switchingAlarmCount(_ controller: LichessBotController) -> Int {
        controller.alarms.filter { $0.text.hasPrefix("Model refresh failed") }.reduce(0) { $0 + $1.repeatCount }
    }

    func testFailedSwitchKeepsPlayingPublishesTheFailureAlarmsAndBacksOff() async throws {
        let (controller, provider) = try await goOnlineOnChampion()
        provider.trainerExists.value = false

        var settings = controller.settings
        settings.model.source = .liveTrainer
        try controller.updateSettings(settings)
        try await waitUntil("the switch fails") { controller.modelRefreshFailure != nil }

        let failure = try XCTUnwrap(controller.modelRefreshFailure)
        XCTAssertTrue(failure.text.contains("No trainer"), failure.text)
        XCTAssertGreaterThan(failure.retryAt.timeIntervalSinceNow, 10, "the retry waits out the backoff, not the next poll")
        XCTAssertEqual(switchingAlarmCount(controller), 1)
        XCTAssertEqual(controller.generation?.sourceKind, .champion, "the old generation keeps playing")
        XCTAssertEqual(controller.connection, .online)

        // The poll ticks every second; the backoff keeps it from retrying.
        try await Task.sleep(for: .seconds(3))
        XCTAssertEqual(switchingAlarmCount(controller), 1, "no retry before the backoff ends")
        XCTAssertEqual(provider.trainerSnapshots.value, 1)
    }

    func testModelSettingsChangeClearsTheFailureAndSwitchesAtTheNextTick() async throws {
        let (controller, provider) = try await goOnlineOnChampion()
        provider.trainerExists.value = false
        var settings = controller.settings
        settings.model.source = .liveTrainer
        try controller.updateSettings(settings)
        try await waitUntil("the switch fails") { controller.modelRefreshFailure != nil }

        // The trainer is back; a new model setting is a new attempt at once,
        // and the old failure (its text and retry time) goes with it.
        provider.trainerExists.value = true
        provider.holdSnapshots()
        settings.model.liveTrainerRefreshIntervalSeconds += 60
        try controller.updateSettings(settings)
        try await waitUntil("the new attempt is building") { provider.snapshotsHeld.value == 1 }
        XCTAssertNil(controller.modelRefreshFailure, "the failure of the old settings is not shown during the new attempt")
        XCTAssertEqual(controller.generation?.sourceKind, .champion, "the old generation plays while the switch builds")

        provider.releaseSnapshots()
        try await waitUntil("the switch lands") { controller.generation?.sourceKind == .liveTrainer }
        XCTAssertNil(controller.modelRefreshFailure)
        XCTAssertEqual(controller.generation?.generationID, 2)
    }
}

/// `LichessBotModelSlots.refreshIfDue` while a build is in flight: it joins
/// the build and reports that build's outcome, never a success before the
/// build ends (review of the ready-before-play change, m4).
final class LichessBotModelSlotsInFlightRefreshTests: XCTestCase {

    private func liveTrainer() -> LichessBotModelSettings {
        var settings = LichessBotModelSettings.testBaseline()
        settings.source = .liveTrainer
        return settings
    }

    private func waitUntil(_ description: String, _ condition: () async -> Bool) async throws {
        for _ in 0..<4000 {
            if await condition() { return }
            try await Task.sleep(for: .milliseconds(5))
        }
        XCTFail("timed out waiting until \(description)")
        throw CancellationError()
    }

    func testRefreshDuringABuildJoinsItsSuccess() async throws {
        let provider = try await LichessBotHoldableModelProvider.make()
        let slots = try await LichessBotModelSlots.prepare(for: .testBaseline(), provider: provider, time: LichessBotManualTime(), log: { _ in })
        provider.holdSnapshots()
        let target = liveTrainer()
        let first = Task { try await slots.refreshIfDue(for: target) }
        try await waitUntil("the switch is building") { provider.snapshotsHeld.value == 1 }
        let second = Task { try await slots.refreshIfDue(for: target) }
        // The second call must still be waiting: give it the chance to
        // return early.
        try await Task.sleep(for: .milliseconds(200))
        provider.releaseSnapshots()
        let firstOutcome = try await first.value
        let secondOutcome = try await second.value
        guard case .built(let built) = firstOutcome, case .joinedBuildInFlight(let joined) = secondOutcome else {
            return XCTFail("expected built then joined, got \(firstOutcome) and \(secondOutcome)")
        }
        XCTAssertEqual(built, joined)
        XCTAssertEqual(joined.sourceKind, .liveTrainer)
        XCTAssertEqual(provider.trainerSnapshots.value, 1, "one build serves both")
    }

    func testRefreshDuringABuildJoinsItsFailure() async throws {
        let provider = try await LichessBotHoldableModelProvider.make()
        let slots = try await LichessBotModelSlots.prepare(for: .testBaseline(), provider: provider, time: LichessBotManualTime(), log: { _ in })
        provider.trainerExists.value = false
        provider.holdSnapshots()
        let target = liveTrainer()
        let first = Task { try await slots.refreshIfDue(for: target) }
        try await waitUntil("the switch is building") { provider.snapshotsHeld.value == 1 }
        let second = Task { try await slots.refreshIfDue(for: target) }
        try await Task.sleep(for: .milliseconds(200))
        provider.releaseSnapshots()
        for (name, task) in [("first", first), ("second", second)] {
            do {
                let outcome = try await task.value
                XCTFail("\(name): expected the build's failure, got \(outcome)")
            } catch let error as LichessBotModelError {
                XCTAssertEqual(error, .noTrainer, name)
            }
        }
        let info = await slots.current.info
        XCTAssertEqual(info.sourceKind, .champion, "the old generation keeps playing")
    }
}
