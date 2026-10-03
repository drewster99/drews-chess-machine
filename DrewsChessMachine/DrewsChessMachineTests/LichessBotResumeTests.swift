import XCTest
@testable import DrewsChessMachine

/// Games left running when the app stopped, picked up again by the next
/// launch (or the next go-online in the same launch): they are resumed, not
/// started over (plan §10.2 launch recovery; owner report 2026-10-02).
@MainActor
final class LichessBotResumeTests: XCTestCase {

    private func waitUntil(_ description: String, _ condition: () -> Bool) async throws {
        for _ in 0..<3000 {
            if condition() { return }
            try await Task.sleep(for: .milliseconds(10))
        }
        XCTFail("timed out waiting until \(description)")
        throw CancellationError()
    }

    /// One "installation": a defaults suite and a data folder that several
    /// controllers (launches) share, removed after the test.
    private struct Installation {
        let suite: String
        let defaults: UserDefaults
        let root: URL
        var directory: LichessBotDataDirectory { LichessBotDataDirectory(root: root) }
    }

    private func makeInstallation(configure: (inout LichessBotSettings) -> Void = { _ in }) throws -> Installation {
        let suite = "LichessBotResumeTests-\(UUID().uuidString)"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("LichessBotResumeTests-\(UUID().uuidString)", isDirectory: true)
        var settings = LichessBotSettings.testBaseline()
        settings.chat.greetingEnabled = false
        settings.connection.preventSleepWhileOnline = false
        settings.challenge.outgoingChallengeTimeoutSeconds = 0
        configure(&settings)
        try LichessBotSettingsStore.save(settings, to: defaults)
        addTeardownBlock {
            defaults.removePersistentDomain(forName: suite)
            if FileManager.default.fileExists(atPath: root.path) {
                do {
                    try FileManager.default.removeItem(at: root)
                } catch {
                    XCTFail("cleanup failed: \(error)")
                }
            }
        }
        return Installation(suite: suite, defaults: defaults, root: root)
    }

    /// A controller (one app launch) over the installation, online.
    private func launch(_ installation: Installation, lichess: LichessBotResumeFakeLichess, modelProvider: LichessBotFakeModelProvider, oneGame: Bool = false) async throws -> LichessBotController {
        let token = LichessBotResumeFakeLichess.token
        let controller = LichessBotController(
            modelProvider: modelProvider,
            defaults: installation.defaults,
            dataDirectory: installation.directory,
            services: LichessBotControllerServices(
                makeTransport: { lichess },
                readToken: { _ in token }
            ),
            finishedGameHold: LichessBotController.finishedGameHold
        )
        addTeardownBlock { @MainActor in
            controller.abandonAndStop()
            await controller.shutdown(reason: "test teardown")
        }
        let streamsBefore = lichess.eventStreamsOpened.value
        await controller.loadPlayerNotes()
        await controller.goOnline(oneGame: oneGame)
        XCTAssertEqual(controller.connection, .online)
        try await waitUntil("this launch's event stream is open") { lichess.eventStreamsOpened.value > streamsBefore }
        return controller
    }

    /// The app stops with its games still running (Xcode stopping a run, a
    /// crash, a force quit): nothing is drained or filed.
    private func stop(_ controller: LichessBotController) async {
        controller.abandonAndStop()
        await controller.shutdown(reason: "simulated app exit")
    }

    private func journalExists(_ installation: Installation, _ gameID: String) -> Bool {
        FileManager.default.fileExists(atPath: installation.directory.inProgressJournalURL(gameID: gameID).path)
    }

    private func protocolMessages(_ controller: LichessBotController, since start: Date) async throws -> [String] {
        try await controller.protocolLog.flush()
        let log = controller.protocolLog
        let urls = Set([log.fileURL(for: start), log.fileURL(for: Date())]).sorted { $0.path < $1.path }
        var messages: [String] = []
        for url in urls where FileManager.default.fileExists(atPath: url.path) {
            let decoded = try LichessBotJSONLines.decode(LichessBotProtocolEntry.self, from: Data(contentsOf: url), fileName: url.lastPathComponent)
            // Launches share the day's file: only this launch's entries.
            messages += decoded.elements.filter { $0.at >= start }.map(\.message)
        }
        return messages
    }

    /// Start `c<opponent>` on `controller` and wait until its journal and
    /// game stream exist.
    private func startAndWait(_ opponent: String, on controller: LichessBotController, lichess: LichessBotResumeFakeLichess, installation: Installation) async throws {
        let count = controller.games.count
        lichess.startGame(against: opponent)
        try await waitUntil("c\(opponent) is listed") { controller.games.count == count + 1 }
        try await waitUntil("c\(opponent)'s journal exists") { journalExists(installation, "c\(opponent)") }
    }

    func testAResumedGameKeepsItsStartTimeAndIsLoggedAsResumed() async throws {
        let installation = try makeInstallation()
        let lichess = LichessBotResumeFakeLichess()
        let model = try await LichessBotFakeModelProvider.randomChampion()
        let first = try await launch(installation, lichess: lichess, modelProvider: model)
        try await startAndWait("bob", on: first, lichess: lichess, installation: installation)
        await stop(first)

        let relaunchedAt = Date()
        let second = try await launch(installation, lichess: lichess, modelProvider: model)
        lichess.startGame(against: "bob")
        try await waitUntil("cbob is listed again") { second.games.count == 1 }
        let game = try XCTUnwrap(second.games.first)
        XCTAssertLessThan(game.startedAt, relaunchedAt, "a resumed game keeps the time it started, not the relaunch time")
        let messages = try await protocolMessages(second, since: relaunchedAt)
        XCTAssertTrue(messages.contains("game resumed"), "\(messages)")
        XCTAssertFalse(messages.contains("game started"), "\(messages)")
    }

    func testAResumedGameCountsAgainstItsOpponentToday() async throws {
        let installation = try makeInstallation()
        let lichess = LichessBotResumeFakeLichess()
        let model = try await LichessBotFakeModelProvider.randomChampion()
        let first = try await launch(installation, lichess: lichess, modelProvider: model)
        try await startAndWait("bob", on: first, lichess: lichess, installation: installation)
        await stop(first)

        let second = try await launch(installation, lichess: lichess, modelProvider: model)
        lichess.startGame(against: "bob")
        try await waitUntil("cbob is listed again") { second.games.count == 1 }
        try await waitUntil("today's games against bob count the resumed game") { second.gamesTodayByOpponent["bob"] == 1 }
    }

    func testPlayOneGameIsNotUsedUpByAResumedGame() async throws {
        let installation = try makeInstallation()
        let lichess = LichessBotResumeFakeLichess()
        let model = try await LichessBotFakeModelProvider.randomChampion()
        let first = try await launch(installation, lichess: lichess, modelProvider: model)
        try await startAndWait("bob", on: first, lichess: lichess, installation: installation)
        await stop(first)

        let second = try await launch(installation, lichess: lichess, modelProvider: model, oneGame: true)
        lichess.startGame(against: "bob")
        try await waitUntil("cbob is listed again") { second.games.count == 1 }
        try await Task.sleep(for: .milliseconds(300))
        XCTAssertEqual(second.connection, .online, "the one game asked for has not started yet")
        XCTAssertTrue(second.oneGameRequested)
    }

    func testResumedGamesAreListedInTheOrderTheyStarted() async throws {
        let installation = try makeInstallation()
        let lichess = LichessBotResumeFakeLichess()
        let model = try await LichessBotFakeModelProvider.randomChampion()
        let first = try await launch(installation, lichess: lichess, modelProvider: model)
        try await startAndWait("bob", on: first, lichess: lichess, installation: installation)
        try await Task.sleep(for: .milliseconds(1100))
        try await startAndWait("carol", on: first, lichess: lichess, installation: installation)
        await stop(first)

        let second = try await launch(installation, lichess: lichess, modelProvider: model)
        lichess.startGame(against: "carol")
        try await waitUntil("ccarol is listed again") { second.games.count == 1 }
        lichess.startGame(against: "bob")
        try await waitUntil("cbob is listed again") { second.games.count == 2 }
        XCTAssertEqual(second.games.map(\.id), ["cbob", "ccarol"], "oldest first, by when each game started")
    }

    func testAGameResumedInTheSameLaunchKeepsItsLiveGameObject() async throws {
        let installation = try makeInstallation()
        let lichess = LichessBotResumeFakeLichess()
        let model = try await LichessBotFakeModelProvider.randomChampion()
        let controller = try await launch(installation, lichess: lichess, modelProvider: model)
        try await startAndWait("bob", on: controller, lichess: lichess, installation: installation)
        let held = try XCTUnwrap(controller.games.first)
        controller.abandonAndStop()
        XCTAssertEqual(controller.connection, .offline)
        // A pop-out window would still hold `held`.
        controller.dismissGame("cbob")
        XCTAssertTrue(controller.games.isEmpty)
        let streamsBefore = lichess.eventStreamsOpened.value
        await controller.goOnline()
        try await waitUntil("the event stream is open again") { lichess.eventStreamsOpened.value > streamsBefore }
        lichess.startGame(against: "bob")
        try await waitUntil("cbob is listed again") { controller.games.count == 1 }
        XCTAssertTrue(controller.games.first === held, "the object a pop-out window holds is listed again, so it keeps updating")
    }

    func testALaunchWithGamesLeftFromTheLastRunSaysSoWhileOffline() async throws {
        let installation = try makeInstallation()
        let lichess = LichessBotResumeFakeLichess()
        let model = try await LichessBotFakeModelProvider.randomChampion()
        let first = try await launch(installation, lichess: lichess, modelProvider: model)
        try await startAndWait("bob", on: first, lichess: lichess, installation: installation)
        await stop(first)

        let second = LichessBotController(
            modelProvider: model,
            defaults: installation.defaults,
            dataDirectory: installation.directory,
            services: LichessBotControllerServices(
                makeTransport: { lichess },
                readToken: { _ in LichessBotResumeFakeLichess.token }
            ),
            finishedGameHold: LichessBotController.finishedGameHold
        )
        addTeardownBlock { @MainActor in
            second.abandonAndStop()
            await second.shutdown(reason: "test teardown")
        }
        await second.noteLeftoverJournalsAtLaunch()
        XCTAssertEqual(second.leftoverGamesFromLastRun, ["cbob"])
        XCTAssertTrue(second.alarms.contains { $0.text.contains("cbob") && $0.text.contains("go online to resume") }, "\(second.alarms.map(\.text))")
        await second.goOnline()
        XCTAssertEqual(second.leftoverGamesFromLastRun, [], "going online resumes or files them")
    }

    /// The move DCM posted in `gameID`, from the fake's POST paths.
    private func postedMove(_ lichess: LichessBotResumeFakeLichess, _ gameID: String) throws -> String {
        let prefix = "/api/bot/game/\(gameID)/move/"
        let path = try XCTUnwrap(lichess.postedPaths.value.first { $0.hasPrefix(prefix) }, "no move posted in \(gameID)")
        return String(path.dropFirst(prefix.count))
    }

    func testAResumedGameShowsItsEarlierDecisionsAndChat() async throws {
        let installation = try makeInstallation { $0.chat.greetingEnabled = true }
        let lichess = LichessBotResumeFakeLichess()
        lichess.movesByGame.modify { $0["cbob"] = "e2e4" }
        let model = try await LichessBotFakeModelProvider.randomChampion()
        let first = try await launch(installation, lichess: lichess, modelProvider: model)
        try await startAndWait("bob", on: first, lichess: lichess, installation: installation)
        try await waitUntil("DCM plays its first move") { lichess.postCount(containing: "/api/bot/game/cbob/move/") > 0 }
        lichess.sendGameLine("cbob", #"{"type":"chatLine","room":"player","username":"bob","text":"hello there"}"#)
        try await waitUntil("bob's chat is shown") { first.games.first?.chat.contains { $0.text == "hello there" } == true }
        try await Task.sleep(for: .milliseconds(300))
        let ourMove = try postedMove(lichess, "cbob")
        await stop(first)

        lichess.movesByGame.modify { $0["cbob"] = "e2e4 \(ourMove)" }
        let second = try await launch(installation, lichess: lichess, modelProvider: model)
        lichess.startGame(against: "bob")
        try await waitUntil("cbob is listed again") { second.games.count == 1 }
        try await waitUntil("cbob's game stream is open again") { lichess.gameStreamIsOpen("cbob") }
        try await Task.sleep(for: .milliseconds(300))
        let game = try XCTUnwrap(second.games.first)
        XCTAssertNotNil(game.decisions[1], "DCM's decision at ply 1, made before the relaunch, is shown")
        XCTAssertTrue(game.chat.contains { $0.username == "bob" && $0.text == "hello there" }, "\(game.chat)")
        XCTAssertTrue(game.chat.contains { $0.origin == .greeting }, "\(game.chat)")
    }

    func testTheTakebackAllowanceIsNotRenewedByAResume() async throws {
        let installation = try makeInstallation { $0.play.maxTakebacksAcceptedPerGame = 1 }
        let lichess = LichessBotResumeFakeLichess()
        lichess.movesByGame.modify { $0["cbob"] = "e2e4" }
        lichess.opponentProposesTakeback.modify { $0.insert("cbob") }
        let model = try await LichessBotFakeModelProvider.randomChampion()
        let first = try await launch(installation, lichess: lichess, modelProvider: model)
        try await startAndWait("bob", on: first, lichess: lichess, installation: installation)
        try await waitUntil("the takeback is accepted") { lichess.postCount(containing: "/api/bot/game/cbob/takeback/yes") == 1 }
        try await Task.sleep(for: .milliseconds(300))
        await stop(first)

        // The opponent proposes another takeback after the relaunch.
        let second = try await launch(installation, lichess: lichess, modelProvider: model)
        lichess.startGame(against: "bob")
        try await waitUntil("cbob is listed again") { second.games.count == 1 }
        try await waitUntil("DCM plays on in the resumed game") { lichess.postCount(containing: "/api/bot/game/cbob/move/") > 0 }
        XCTAssertEqual(lichess.postCount(containing: "/api/bot/game/cbob/takeback/yes"), 1, "the game's one takeback was spent before the relaunch")
    }

    func testAResumedGameIsNotGreetedAgain() async throws {
        let installation = try makeInstallation { $0.chat.greetingEnabled = true }
        let lichess = LichessBotResumeFakeLichess()
        let model = try await LichessBotFakeModelProvider.randomChampion()
        let first = try await launch(installation, lichess: lichess, modelProvider: model)
        try await startAndWait("bob", on: first, lichess: lichess, installation: installation)
        try await waitUntil("the greeting is sent") { lichess.postCount(containing: "/api/bot/game/cbob/chat") > 0 }
        try await Task.sleep(for: .milliseconds(300))
        let greetingPosts = lichess.postCount(containing: "/api/bot/game/cbob/chat")
        await stop(first)

        // White moved while the app was down: DCM resumes at ply 1, where a
        // new game would still be greeted.
        lichess.movesByGame.modify { $0["cbob"] = "e2e4" }
        let second = try await launch(installation, lichess: lichess, modelProvider: model)
        lichess.startGame(against: "bob")
        try await waitUntil("DCM plays its move in the resumed game") { lichess.postCount(containing: "/api/bot/game/cbob/move/") > 0 }
        try await Task.sleep(for: .milliseconds(300))
        XCTAssertEqual(lichess.postCount(containing: "/api/bot/game/cbob/chat"), greetingPosts, "the opponent was greeted before the relaunch")
        XCTAssertEqual(second.games.count, 1)
    }
}
