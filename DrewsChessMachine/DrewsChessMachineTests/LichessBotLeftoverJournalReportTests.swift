import XCTest
@testable import DrewsChessMachine

/// What a launch says, while offline, about the journals the last run left
/// in `InProgress/`. A quit doesn't wait to file a finished game that is
/// still collecting its post-game chat (launch recovery files it), so a
/// cleanly finished game is left there on many ordinary quits: that is not
/// an alarm, and the status chip doesn't call it unfinished. A game whose
/// journal never recorded a finish may still be running on Lichess on DCM's
/// clock, and is alarmed; one whose journal can't be read is alarmed with
/// the reason, since nobody can tell which it is. And a report computed
/// while the bot was going online is dropped: going online resumes or files
/// those games itself.
@MainActor
final class LichessBotLeftoverJournalReportTests: XCTestCase {

    private func waitUntil(_ description: String, _ condition: () -> Bool) async throws {
        for _ in 0..<3000 {
            if condition() { return }
            try await Task.sleep(for: .milliseconds(10))
        }
        XCTFail("timed out waiting until \(description)")
        throw CancellationError()
    }

    /// A defaults suite and a data folder that several controllers (launches)
    /// share, removed after the test.
    private struct Installation {
        let defaults: UserDefaults
        let root: URL
        var directory: LichessBotDataDirectory { LichessBotDataDirectory(root: root) }
    }

    private func makeInstallation() throws -> Installation {
        let defaults = try makeTemporaryDefaults()
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("LichessBotLeftoverJournalReportTests-\(UUID().uuidString)", isDirectory: true)
        var settings = LichessBotSettings.testBaseline()
        settings.chat.greetingEnabled = false
        settings.chat.goodbyeEnabled = false
        settings.connection.preventSleepWhileOnline = false
        settings.challenge.outgoingChallengeTimeoutSeconds = 0
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

    /// A controller (one app launch) over the installation, not yet online;
    /// shut down after the test.
    private func makeController(_ installation: Installation, transport: any LichessBotTransport, modelProvider: LichessBotFakeModelProvider, postGameChatFetchDelays: [Duration] = LichessBotController.postGameChatFetchDelays) -> LichessBotController {
        let token = LichessBotResumeFakeLichess.token
        let controller = LichessBotController(
            modelProvider: modelProvider,
            defaults: installation.defaults,
            dataDirectory: installation.directory,
            services: LichessBotControllerServices(
                makeTransport: { transport },
                readToken: { _ in token }
            ),
            finishedGameHold: LichessBotController.finishedGameHold,
            postGameChatFetchDelays: postGameChatFetchDelays
        )
        addTeardownBlock { @MainActor in
            controller.abandonAndStop()
            await controller.shutdown(reason: "test teardown")
        }
        return controller
    }

    /// The first launch finishes `cbob` (the opponent resigns) and then
    /// stops while the game still waits for its post-game chat fetches: the
    /// files a quit leaves (the quit stops at once; launch recovery files
    /// the game).
    private func leaveAFinishedGameUnfiled(_ installation: Installation) async throws {
        let lichess = LichessBotResumeFakeLichess()
        let model = try await LichessBotFakeModelProvider.randomChampion()
        let first = makeController(installation, transport: lichess, modelProvider: model, postGameChatFetchDelays: [.seconds(3600)])
        await first.goOnline()
        XCTAssertEqual(first.connection, .online)
        try await waitUntil("the event stream is open") { lichess.eventStreamsOpened.value > 0 }
        lichess.startGame(against: "bob")
        try await waitUntil("cbob is listed") { first.games.count == 1 }
        try await waitUntil("cbob's game stream is open") { lichess.gameStreamIsOpen("cbob") }
        lichess.opponentResigns("bob")
        try await waitUntil("cbob has finished") { first.games.first?.finishedAt != nil }
        try await waitUntil("cbob's session has ended") { first.activeGameIDs.isEmpty }
        first.abandonAndStop()
        await first.shutdown(reason: "simulated quit")
        XCTAssertTrue(FileManager.default.fileExists(atPath: installation.directory.inProgressJournalURL(gameID: "cbob").path), "the quit left cbob's journal unfiled")
    }

    func testAFinishedGameLeftByQuitIsNotAlarmedAtNextLaunch() async throws {
        let installation = try makeInstallation()
        try await leaveAFinishedGameUnfiled(installation)

        let second = makeController(installation, transport: LichessBotResumeFakeLichess(), modelProvider: LichessBotFakeModelProvider(snapshot: nil))
        await second.noteLeftoverJournalsAtLaunch()
        XCTAssertEqual(second.alarms.map(\.text), [], "a game that finished cleanly is not alarmed")
        XCTAssertEqual(second.leftoverGamesFromLastRun, [], "a game that finished is not unfinished")
    }

    func testAFinishedGameLeftByQuitIsCountedToFileAtNextLaunch() async throws {
        let installation = try makeInstallation()
        try await leaveAFinishedGameUnfiled(installation)

        let second = makeController(installation, transport: LichessBotResumeFakeLichess(), modelProvider: LichessBotFakeModelProvider(snapshot: nil))
        await second.noteLeftoverJournalsAtLaunch()
        XCTAssertEqual(second.finishedGamesAwaitingFilingFromLastRun, ["cbob"])
        await second.goOnline()
        XCTAssertEqual(second.connection, .online)
        XCTAssertEqual(second.finishedGamesAwaitingFilingFromLastRun, [], "going online files them")
    }

    func testAnUnreadableLeftoverJournalIsAlarmedWithItsReason() async throws {
        let installation = try makeInstallation()
        try installation.directory.createDirectories()
        try Data("not a journal line\n".utf8).write(to: installation.directory.inProgressJournalURL(gameID: "cbob"))

        let controller = makeController(installation, transport: LichessBotResumeFakeLichess(), modelProvider: LichessBotFakeModelProvider(snapshot: nil))
        await controller.noteLeftoverJournalsAtLaunch()
        XCTAssertEqual(controller.leftoverGamesFromLastRun, ["cbob"], "a game that may still be live is reported as unfinished")
        XCTAssertTrue(
            controller.alarms.contains { $0.text.contains("cbob") && $0.text.contains("can't tell whether it finished") },
            "\(controller.alarms.map(\.text))")
    }

    func testTheLeftoverReportIsDroppedWhileGoingOnline() async throws {
        let installation = try makeInstallation()
        try installation.directory.createDirectories()
        // A journal with no lines yet: a game that never recorded a finish.
        try Data().write(to: installation.directory.inProgressJournalURL(gameID: "cbob"))
        let lichess = LichessBotForwardingFakeLichess(holdsAccount: true)
        let controller = makeController(installation, transport: lichess, modelProvider: LichessBotFakeModelProvider(snapshot: nil))
        addTeardownBlock {
            lichess.accountRelease.open()
        }

        let goingOnline = Task { @MainActor in
            await controller.goOnline()
        }
        try await waitUntil("going online is waiting for the account") { lichess.accountRequestsReceived.value > 0 }
        XCTAssertEqual(controller.connection, .connecting)
        await controller.noteLeftoverJournalsAtLaunch()
        XCTAssertFalse(controller.alarms.contains { $0.text.contains("last run left") }, "\(controller.alarms.map(\.text))")
        XCTAssertEqual(controller.leftoverGamesFromLastRun, [])

        lichess.accountRelease.open()
        await goingOnline.value
        XCTAssertEqual(controller.connection, .online)
        XCTAssertEqual(controller.leftoverGamesFromLastRun, [])
    }
}
