import XCTest
@testable import DrewsChessMachine

/// A Lichess for a whole controller runtime: token check, account, an event
/// stream and game streams the test writes to, player statuses, challenge
/// creation and cancellation, and the online-bots list. Every other POST
/// succeeds and every other GET is a 404.
final class LichessBotFakeLichess: LichessBotTransport, @unchecked Sendable {
    static let token = "lip_CONTROLLERTEST"
    static let botID = "drewschessmachine"

    /// Usernames challenged, in order, as the path named them.
    let challengedNames = SyncBox<[String]>([])
    /// Challenge ids withdrawn.
    let canceledIDs = SyncBox<[String]>([])
    /// NDJSON body of `GET /api/bot/online`.
    let onlineBotsNDJSON = SyncBox<String>("")
    /// The `perfs` object of `GET /api/account`.
    let accountPerfsJSON: String
    private let eventContinuation = SyncBox<LichessBotChunkStream.Continuation?>(nil)
    private let gameContinuations = SyncBox<[String: LichessBotChunkStream.Continuation]>([:])

    init(accountPerfsJSON: String = "{}") {
        self.accountPerfsJSON = accountPerfsJSON
    }

    /// The id Lichess gives our challenge to `username` (and the game's id
    /// if it is accepted).
    static func challengeID(for username: String) -> String {
        "c" + username.lowercased()
    }

    var eventStreamIsOpen: Bool {
        eventContinuation.value != nil
    }

    func gameStreamIsOpen(_ gameID: String) -> Bool {
        gameContinuations.value[gameID] != nil
    }

    func sendEvent(_ line: String) {
        eventContinuation.value?.yield(Data((line + "\n").utf8))
    }

    func sendGameLine(_ gameID: String, _ line: String) {
        gameContinuations.value[gameID]?.yield(Data((line + "\n").utf8))
    }

    /// Our challenge to `opponent` was accepted: its game starts.
    func startGame(against opponent: String) {
        sendEvent(#"{"type":"gameStart","game":{"gameId":"\#(Self.challengeID(for: opponent))","opponent":{"id":"\#(opponent.lowercased())"}}}"#)
    }

    /// The opponent resigns the game against them.
    func endGame(against opponent: String) {
        sendGameLine(Self.challengeID(for: opponent), #"{"type":"gameState","moves":"","wtime":180000,"btime":180000,"winc":2000,"binc":2000,"status":"resign","winner":"black"}"#)
    }

    func data(for request: URLRequest) async throws -> LichessBotTransportResponse {
        let (status, body) = answer(request)
        return LichessBotTransportResponse(body: Data(body.utf8), response: try lichessBotTestResponse(status: status), networkProtocolName: "h2")
    }

    func stream(for request: URLRequest) async throws -> (chunks: LichessBotChunkStream, response: HTTPURLResponse) {
        let path = request.url?.path ?? ""
        let (chunks, continuation) = LichessBotChunkStream.makeStream()
        if path == "/api/stream/event" {
            eventContinuation.value = continuation
            return (chunks, try lichessBotTestResponse(status: 200))
        }
        let gamePrefix = "/api/bot/game/stream/"
        if path.hasPrefix(gamePrefix) {
            let gameID = String(path.dropFirst(gamePrefix.count))
            gameContinuations.modify { $0[gameID] = continuation }
            continuation.yield(Data((Self.gameFullJSON(gameID: gameID) + "\n").utf8))
            return (chunks, try lichessBotTestResponse(status: 200))
        }
        continuation.yield(Data(#"{"error":"Not found"}"#.utf8))
        continuation.finish()
        return (chunks, try lichessBotTestResponse(status: 404))
    }

    /// DCM plays black, so it waits for the opponent's first move and the
    /// test decides when the game ends.
    private static func gameFullJSON(gameID: String) -> String {
        let opponent = gameID.hasPrefix("c") ? String(gameID.dropFirst()) : gameID
        let white = #"{"id":"\#(opponent)","name":"\#(opponent)","title":"BOT","rating":1500}"#
        let black = #"{"id":"\#(botID)","name":"DrewsChessMachine","title":"BOT","rating":1500}"#
        let state = #"{"type":"gameState","moves":"","wtime":180000,"btime":180000,"winc":2000,"binc":2000,"status":"started"}"#
        return #"{"type":"gameFull","id":"\#(gameID)","variant":{"key":"standard","name":"Standard","short":"Std"},"clock":{"initial":180000,"increment":2000},"speed":"blitz","perf":{"name":"Blitz"},"rated":false,"createdAt":1700000000000,"white":\#(white),"black":\#(black),"initialFen":"startpos","state":\#(state)}"#
    }

    private func answer(_ request: URLRequest) -> (Int, String) {
        let method = request.httpMethod ?? "GET"
        let components = request.url.flatMap { URLComponents(url: $0, resolvingAgainstBaseURL: true) }
        let path = components?.path ?? ""
        let segments = path.split(separator: "/").map(String.init)
        switch (method, path) {
        case ("POST", "/api/token/test"):
            return (200, #"{"\#(Self.token)":{"userId":"\#(Self.botID)","scopes":"bot:play,challenge:write","expires":null}}"#)
        case ("GET", "/api/account"):
            return (200, #"{"id":"\#(Self.botID)","username":"DrewsChessMachine","title":"BOT","perfs":\#(accountPerfsJSON)}"#)
        case ("GET", "/api/users/status"):
            let ids = components?.queryItems?.first { $0.name == "ids" }?.value?.split(separator: ",").map(String.init) ?? []
            let entries = ids.map { #"{"id":"\#($0.lowercased())","name":"\#($0)","title":"BOT","online":true}"# }
            return (200, "[" + entries.joined(separator: ",") + "]")
        case ("GET", "/api/bot/online"):
            return (200, onlineBotsNDJSON.value)
        default:
            break
        }
        if method == "POST", segments.count == 4, segments[0] == "api", segments[1] == "challenge", segments[3] == "cancel" {
            canceledIDs.modify { $0.append(segments[2]) }
            return (200, #"{"ok":true}"#)
        }
        if method == "POST", segments.count == 3, segments[0] == "api", segments[1] == "challenge" {
            let username = segments[2]
            challengedNames.modify { $0.append(username) }
            let id = Self.challengeID(for: username)
            return (200, #"{"id":"\#(id)","status":"created","challenger":{"id":"\#(Self.botID)","name":"DrewsChessMachine","rating":1500},"destUser":{"id":"\#(username.lowercased())","name":"\#(username)","rating":1500},"variant":{"key":"standard"},"rated":false,"speed":"blitz","timeControl":{"type":"clock","limit":300,"increment":3},"color":"random","direction":"out"}"#)
        }
        if method == "POST" {
            return (200, #"{"ok":true}"#)
        }
        return (404, #"{"error":"Not found"}"#)
    }
}

/// The controller's challenge queue, matchmaking's Fill Open Slots, and the
/// single game view's hold on a finished game, over a whole runtime against
/// `LichessBotFakeLichess` (plan §7.3, §14.3a).
@MainActor
final class LichessBotChallengeQueueControllerTests: XCTestCase {

    private func waitUntil(_ description: String, _ condition: () -> Bool) async throws {
        for _ in 0..<3000 {
            if condition() { return }
            try await Task.sleep(for: .milliseconds(10))
        }
        XCTFail("timed out waiting until \(description)")
        throw CancellationError()
    }

    /// An online controller with its own defaults suite, data folder and
    /// fake Lichess, all removed after the test.
    private func makeOnlineController(
        lichess: LichessBotFakeLichess = LichessBotFakeLichess(),
        modelProvider: LichessBotFakeModelProvider = LichessBotFakeModelProvider(snapshot: nil),
        finishedGameHold: Duration = LichessBotController.finishedGameHold,
        configure: (inout LichessBotSettings) -> Void
    ) async throws -> LichessBotController {
        let suite = "LichessBotChallengeQueueControllerTests-\(UUID().uuidString)"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("LichessBotChallengeQueueControllerTests-\(UUID().uuidString)", isDirectory: true)
        var settings = LichessBotSettings()
        settings.chat.greetingEnabled = false
        settings.connection.preventSleepWhileOnline = false
        // No automatic withdrawal: the tests decide when challenges end.
        settings.challenge.outgoingChallengeTimeoutSeconds = 0
        configure(&settings)
        try LichessBotSettingsStore.save(settings, to: defaults)
        let token = LichessBotFakeLichess.token
        let controller = LichessBotController(
            modelProvider: modelProvider,
            defaults: defaults,
            dataDirectory: LichessBotDataDirectory(root: root),
            services: LichessBotControllerServices(
                makeTransport: { lichess },
                readToken: { _ in token }
            ),
            finishedGameHold: finishedGameHold
        )
        addTeardownBlock { @MainActor in
            controller.abandonAndStop()
            // Let journal writes the stopped runtime already queued land
            // before its folder is removed.
            try await Task.sleep(for: .milliseconds(300))
            defaults.removePersistentDomain(forName: suite)
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
        XCTAssertTrue(controller.hasChallengeScope)
        try await waitUntil("the event stream is open") { lichess.eventStreamIsOpen }
        return controller
    }

    private let request = LichessBotOutgoingChallenge(rated: false, clockLimitSeconds: 300, clockIncrementSeconds: 3, color: .random)

    private func players(_ names: String...) -> [LichessBotChallengeQueue.Player] {
        names.map { LichessBotChallengeQueue.Player(username: $0, userID: $0) }
    }

    /// With one slot already taken by a challenge sent directly, four
    /// selected players fill the two free slots in order and the other two
    /// wait for a slot.
    func testSelectingMorePlayersThanFreeSlotsSendsThoseThatFitAndQueuesTheRest() async throws {
        let lichess = LichessBotFakeLichess()
        let controller = try await makeOnlineController(lichess: lichess) { settings in
            settings.challenge.maxConcurrentGames = 3
        }
        try await controller.sendChallenge(to: "zed", request: request)

        let result = try controller.enqueueChallenges(to: players("alice", "bob", "carol", "dave"), request: request)
        XCTAssertEqual(result.added, ["alice", "bob", "carol", "dave"])
        try await waitUntil("the free slots are filled and the rest wait") {
            controller.challengeQueueWaitReason == LichessBotChallengeQueue.waitingForSlotReason
        }

        XCTAssertEqual(lichess.challengedNames.value, ["zed", "alice", "bob"])
        XCTAssertEqual(controller.pendingChallenges.map(\.username), ["zed", "alice", "bob"])
        XCTAssertEqual(controller.challengeQueue.entries.map(\.username), ["carol", "dave"])
        XCTAssertEqual(controller.challengeQueue.entries.map(\.status), [.waiting, .waiting])
        XCTAssertEqual(controller.freeChallengeSlots, 0)

        let again = try controller.enqueueChallenges(to: players("bob", "carol", "erin"), request: request)
        XCTAssertEqual(again.alreadyPending, ["bob"])
        XCTAssertEqual(again.alreadyQueued, ["carol"])
        XCTAssertEqual(again.added, ["erin"])
    }

    /// A game from a queued challenge holds its slot while it starts and
    /// while it is played; when it ends, the next entry is sent.
    func testAGameEndingSendsTheNextQueuedEntry() async throws {
        let lichess = LichessBotFakeLichess()
        let controller = try await makeOnlineController(lichess: lichess, modelProvider: try await LichessBotFakeModelProvider.randomChampion()) { settings in
            settings.challenge.maxConcurrentGames = 1
        }
        try controller.enqueueChallenges(to: players("bob", "carol"), request: request)
        try await waitUntil("bob's challenge is pending") { controller.pendingChallenges.map(\.username) == ["bob"] }

        lichess.startGame(against: "bob")
        let gameID = LichessBotFakeLichess.challengeID(for: "bob")
        try await waitUntil("bob's game is in progress") { controller.activeGameIDs.contains(gameID) && lichess.gameStreamIsOpen(gameID) }
        XCTAssertEqual(lichess.challengedNames.value, ["bob"], "the starting and running game hold the only slot")
        XCTAssertEqual(controller.challengeQueue.entries.map(\.username), ["carol"])
        XCTAssertEqual(controller.challengeQueueWaitReason, LichessBotChallengeQueue.waitingForSlotReason)

        lichess.endGame(against: "bob")
        try await waitUntil("carol is challenged") { lichess.challengedNames.value == ["bob", "carol"] }
        try await waitUntil("carol's challenge is pending") { controller.pendingChallenges.map(\.username) == ["carol"] }
        XCTAssertTrue(controller.challengeQueue.isEmpty)
        XCTAssertFalse(controller.activeGameIDs.contains(gameID))
    }

    /// A decline frees the slot for the next entry, and starts the bot's
    /// decline cool-down in the player notes.
    func testADeclineSendsTheNextEntryAndRecordsTheCooldown() async throws {
        let lichess = LichessBotFakeLichess()
        let controller = try await makeOnlineController(lichess: lichess) { settings in
            settings.challenge.maxConcurrentGames = 1
        }
        try controller.enqueueChallenges(to: players("bob", "carol"), request: request)
        try await waitUntil("bob's challenge is pending") { controller.pendingChallenges.map(\.username) == ["bob"] }

        lichess.sendEvent(#"{"type":"challengeDeclined","challenge":{"id":"\#(LichessBotFakeLichess.challengeID(for: "bob"))","declineReason":"I'm not accepting challenges at the moment.","declineReasonKey":"later"}}"#)
        try await waitUntil("carol is challenged") { lichess.challengedNames.value == ["bob", "carol"] }
        let cooldownEnds = try XCTUnwrap(controller.playerNotes?.declineCooldownEnds("bob", now: Date()))
        XCTAssertGreaterThan(cooldownEnds.timeIntervalSinceNow, 5 * 3600, "the default cool-down is hours long")
        XCTAssertNil(controller.playerNotes?.declineCooldownEnds("carol", now: Date()))
    }

    /// Go Offline empties the queue and withdraws the challenges waiting for
    /// an answer.
    func testGoingOfflineClearsTheQueueAndWithdrawsPendingChallenges() async throws {
        let lichess = LichessBotFakeLichess()
        let controller = try await makeOnlineController(lichess: lichess) { settings in
            settings.challenge.maxConcurrentGames = 2
        }
        try controller.enqueueChallenges(to: players("bob", "carol", "dave"), request: request)
        try await waitUntil("two challenges are pending") { controller.pendingChallenges.count == 2 }
        XCTAssertEqual(controller.challengeQueue.entries.map(\.username), ["dave"])

        await controller.goOffline()
        XCTAssertEqual(controller.connection, .offline)
        XCTAssertTrue(controller.challengeQueue.isEmpty)
        XCTAssertTrue(controller.pendingChallenges.isEmpty)
        let expected = Set(["bob", "carol"].map { LichessBotFakeLichess.challengeID(for: $0) })
        try await waitUntil("both pending challenges are withdrawn") { Set(lichess.canceledIDs.value) == expected }
        XCTAssertEqual(lichess.challengedNames.value, ["bob", "carol"], "dave was never sent")
    }

    /// Fill Open Slots challenges fitting online bots, never takes the slot
    /// reserved for humans, and skips DCM itself and bots outside the
    /// rating window.
    func testFillOpenSlotsLeavesTheReservedHumanSlotFree() async throws {
        let lichess = LichessBotFakeLichess(accountPerfsJSON: #"{"blitz":{"games":100,"rating":1500,"rd":50,"prog":0}}"#)
        lichess.onlineBotsNDJSON.value = [
            #"{"id":"drewschessmachine","username":"DrewsChessMachine","title":"BOT","perfs":{"blitz":{"games":100,"rating":1500,"rd":50,"prog":0}}}"#,
            #"{"id":"fitbot","username":"FitBot","title":"BOT","perfs":{"blitz":{"games":50,"rating":1600,"rd":60,"prog":0}}}"#,
            #"{"id":"farbot","username":"FarBot","title":"BOT","perfs":{"blitz":{"games":50,"rating":2500,"rd":60,"prog":0}}}"#,
        ].joined(separator: "\n") + "\n"
        let controller = try await makeOnlineController(lichess: lichess) { settings in
            settings.challenge.maxConcurrentGames = 2
            settings.challenge.gamesReservedForHumans = 1
            settings.matchmaking.timeControls = [.blitz5plus3]
        }
        await controller.fillOpenSlots()

        XCTAssertEqual(lichess.challengedNames.value, ["FitBot"])
        XCTAssertEqual(controller.pendingChallenges.map(\.username), ["FitBot"])
        XCTAssertEqual(controller.pendingChallenges.first?.request, LichessBotOutgoingChallenge(rated: false, clockLimitSeconds: 300, clockIncrementSeconds: 3, color: .random))
        XCTAssertEqual(controller.freeChallengeSlots, 1, "one slot is still free, but it is reserved for humans")
        XCTAssertFalse(controller.isFillingOpenSlots)
        XCTAssertEqual(controller.matchmakingRateLimiter.challengesInLastHour(now: Date()), 1)
    }

    /// The single view stays on a followed game after it ends until the hold
    /// is over, then moves to the next game in progress.
    func testTheSingleViewHoldsAFinishedGameBeforeMovingOn() async throws {
        let hold = Duration.milliseconds(1500)
        let lichess = LichessBotFakeLichess()
        let controller = try await makeOnlineController(lichess: lichess, modelProvider: try await LichessBotFakeModelProvider.randomChampion(), finishedGameHold: hold) { settings in
            settings.challenge.maxConcurrentGames = 2
        }
        try controller.enqueueChallenges(to: players("bob", "carol"), request: request)
        try await waitUntil("both challenges are pending") { controller.pendingChallenges.count == 2 }
        let bobGame = LichessBotFakeLichess.challengeID(for: "bob")
        let carolGame = LichessBotFakeLichess.challengeID(for: "carol")
        lichess.startGame(against: "bob")
        try await waitUntil("bob's game is in progress") { controller.activeGameIDs.contains(bobGame) }
        lichess.startGame(against: "carol")
        try await waitUntil("carol's game is in progress") { controller.activeGameIDs.contains(carolGame) && lichess.gameStreamIsOpen(bobGame) }
        XCTAssertNil(controller.focusedGameID)
        XCTAssertEqual(controller.displayedGame?.id, bobGame, "the first-started game is followed")

        lichess.endGame(against: "bob")
        // The hold starts when the final state is applied to the game.
        try await waitUntil("bob's game is finished") { controller.games.first { $0.id == bobGame }?.isFinished == true }
        let finishedAt = ContinuousClock.now
        try await waitUntil("bob's game session has ended") { !controller.activeGameIDs.contains(bobGame) }
        XCTAssertEqual(controller.displayedGame?.id, bobGame, "the finished game stays on screen after its session ends")
        try await waitUntil("the view moves on") { controller.displayedGame?.id == carolGame }
        XCTAssertGreaterThanOrEqual(ContinuousClock.now - finishedAt, hold - .milliseconds(100), "not before the hold ends")
        XCTAssertTrue(controller.games.contains { $0.id == bobGame }, "the finished game is still listed")
    }
}
