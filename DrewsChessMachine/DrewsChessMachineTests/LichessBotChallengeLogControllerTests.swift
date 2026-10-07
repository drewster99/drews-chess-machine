//
//  LichessBotChallengeLogControllerTests.swift
//  DrewsChessMachineTests
//
//  The controller's challenge log (challenge-log plan §3.4) over a whole
//  runtime against `LichessBotFakeLichess`: every send path writes
//  `outgoingCreated` with its sender, withdrawals write their request and
//  Lichess's answer, stream answers and game starts reach the ledger,
//  incoming challenges and DCM's decisions are recorded, and every file is
//  written under the test's temporary data folder.
//

import XCTest
@testable import DrewsChessMachine

@MainActor
final class LichessBotChallengeLogControllerTests: XCTestCase {

    private static let fitBotLine = #"{"id":"fitbot","username":"FitBot","title":"BOT","perfs":{"blitz":{"games":50,"rating":1600,"rd":60,"prog":0}}}"#
    private let request = LichessBotOutgoingChallenge(rated: false, clockLimitSeconds: 300, clockIncrementSeconds: 3, color: .random)

    private func waitUntil(_ description: String, _ condition: () -> Bool) async throws {
        for _ in 0..<3000 {
            if condition() { return }
            try await Task.sleep(for: .milliseconds(10))
        }
        XCTFail("timed out waiting until \(description)")
        throw CancellationError()
    }

    private struct Online {
        let controller: LichessBotController
        let lichess: LichessBotFakeLichess
        let root: URL
    }

    /// An online controller with its own defaults suite, data folder and
    /// fake Lichess (FitBot is the one matchmaking candidate), all removed
    /// after the test.
    private func makeOnline(configure: (inout LichessBotSettings) -> Void = { _ in }) async throws -> Online {
        let lichess = LichessBotFakeLichess(accountPerfsJSON: #"{"blitz":{"games":100,"rating":1500,"rd":50,"prog":0}}"#)
        lichess.onlineBotsNDJSON.value = Self.fitBotLine + "\n"
        let defaults = try makeTemporaryDefaults()
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("LichessBotChallengeLogControllerTests-\(UUID().uuidString)", isDirectory: true)
        var settings = LichessBotSettings.testBaseline()
        settings.chat.greetingEnabled = false
        settings.connection.preventSleepWhileOnline = false
        settings.challenge.outgoingChallengeTimeoutSeconds = 0
        settings.challenge.maxConcurrentGames = 3
        settings.challenge.gamesReservedForHumans = 0
        settings.matchmaking.enabled = false
        settings.matchmaking.timeControls = [.blitz5plus3]
        settings.matchmaking.rated = false
        configure(&settings)
        try LichessBotSettingsStore.save(settings, to: defaults)
        let token = LichessBotFakeLichess.token
        let controller = LichessBotController(
            modelProvider: try await LichessBotFakeModelProvider.randomChampion(),
            defaults: defaults,
            dataDirectory: LichessBotDataDirectory(root: root),
            services: LichessBotControllerServices(makeTransport: { lichess }, readToken: { _ in token })
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
        XCTAssertEqual(controller.connection, .online)
        XCTAssertNotNil(controller.challengeLedger, "going online loads the challenge log")
        try await waitUntil("the event stream is open") { lichess.eventStreamIsOpen }
        return Online(controller: controller, lichess: lichess, root: root)
    }

    private func row(_ online: Online, _ username: String) -> LichessBotChallengeLedgerRow? {
        online.controller.challengeLedger?.row(challengeID: LichessBotFakeLichess.challengeID(for: username))
    }

    /// The day files' events, after every queued append has landed.
    private func writtenEvents(_ online: Online) async throws -> [LichessBotChallengeLogEvent] {
        try await online.controller.challengeLogRecorder.flush()
        return try LichessBotChallengeLog.readAll(in: LichessBotDataDirectory(root: online.root)).entries.map(\.event)
    }

    // MARK: - Senders

    func testEachSendPathRecordsItsSender() async throws {
        let online = try await makeOnline()
        let controller = online.controller
        try await controller.sendChallenge(to: "alice", request: request)
        XCTAssertEqual(row(online, "alice")?.sender, .challengeSheet)
        XCTAssertEqual(row(online, "alice")?.state, .open)

        try controller.enqueueChallenges(to: [LichessBotChallengeQueue.Player(username: "bob", userID: "bob")], request: request)
        try await waitUntil("the queue sends bob") { self.row(online, "bob") != nil }
        XCTAssertEqual(row(online, "bob")?.sender, .challengeQueue)

        await controller.fillOpenSlots()
        XCTAssertEqual(row(online, "FitBot")?.sender, .matchmaking(trigger: .fillOpenSlots, fillMode: .everyFreeSlot))

        let events = try await writtenEvents(online)
        let createdIDs = events.compactMap { event -> String? in
            guard case .outgoingCreated(let challenge, _, _, _, _) = event else { return nil }
            return challenge.id
        }
        XCTAssertEqual(createdIDs, ["calice", "cbob", "cfitbot"])
    }

    func testTheOperatorsResendAsCasualIsItsOwnSender() async throws {
        let online = try await makeOnline()
        let rated = LichessBotOutgoingChallenge(rated: true, clockLimitSeconds: 180, clockIncrementSeconds: 2, color: .white)
        try await online.controller.sendChallenge(to: "bob", request: rated)
        online.lichess.sendEvent(#"{"type":"challengeDeclined","challenge":{"id":"cbob","declineReason":"Casual please","declineReasonKey":"casual"}}"#)
        try await waitUntil("the manual offer is made") { online.controller.casualResendOffer != nil }
        XCTAssertEqual(row(online, "bob")?.state, .declined(.known(.casual)))
        await online.controller.resendAsCasual()
        let events = try await writtenEvents(online)
        let senders = events.compactMap { event -> LichessBotChallengeSender? in
            guard case .outgoingCreated(_, let sender, _, _, _) = event else { return nil }
            return sender
        }
        XCTAssertEqual(senders, [.challengeSheet, .casualResendOffer])
    }

    // MARK: - Answers and withdrawals

    func testAnOperatorCancelRecordsTheWithdrawalAndItsAnswer() async throws {
        let online = try await makeOnline()
        try await online.controller.sendChallenge(to: "alice", request: request)
        await online.controller.cancelChallenge(id: "calice")
        XCTAssertEqual(row(online, "alice")?.state, .withdrawn(.operatorCancel, .confirmed))
        await online.controller.cancelChallenge(id: "nobody")
        let events = try await writtenEvents(online)
        XCTAssertFalse(events.contains(.withdrawalRequested(challengeID: "nobody", reason: .operatorCancel)), "a cancel of an id that isn't pending writes nothing")
    }

    func testGoingOfflineRecordsItsWithdrawals() async throws {
        let online = try await makeOnline()
        try await online.controller.sendChallenge(to: "alice", request: request)
        await online.controller.goOffline()
        try await waitUntil("the withdrawal is answered") {
            self.row(online, "alice")?.withdrawalResults.isEmpty == false
        }
        XCTAssertEqual(row(online, "alice")?.state, .withdrawn(.goingOffline, .confirmed))
    }

    func testAGameStartIsRecordedAsAccepted() async throws {
        let online = try await makeOnline()
        try await online.controller.sendChallenge(to: "alice", request: request)
        online.lichess.startGame(against: "alice")
        try await waitUntil("the game starts") { self.row(online, "alice")?.state == .accepted(gameStarted: true) }
        XCTAssertEqual(row(online, "alice")?.sender, .challengeSheet)
    }

    func testOurOwnEchoAfterTheCreatedLineWritesNothing() async throws {
        let online = try await makeOnline()
        try await online.controller.sendChallenge(to: "alice", request: request)
        online.lichess.sendEvent(#"{"type":"challenge","challenge":{"id":"calice","status":"created","challenger":{"id":"drewschessmachine","name":"DrewsChessMachine","rating":1500},"destUser":{"id":"alice","name":"alice","rating":1500},"variant":{"key":"standard"},"rated":false,"speed":"blitz","timeControl":{"type":"clock","limit":300,"increment":3},"color":"random"}}"#)
        online.lichess.startGame(against: "alice")
        try await waitUntil("the game starts") { self.row(online, "alice")?.state == .accepted(gameStarted: true) }
        await online.controller.goOffline()
        let events = try await writtenEvents(online)
        XCTAssertFalse(events.contains { if case .outgoingSeenWithoutCreatedLine = $0 { return true }; return false },
                       "an echo matched by its created line is never written, even at teardown")
    }

    // MARK: - Incoming

    func testAnIncomingChallengeAndDCMsDecisionAreRecorded() async throws {
        let online = try await makeOnline()
        online.lichess.sendEvent(#"{"type":"challenge","challenge":{"id":"inco1","status":"created","challenger":{"id":"carol","name":"Carol","rating":1500},"destUser":{"id":"drewschessmachine","name":"DrewsChessMachine","rating":1500},"variant":{"key":"chess960"},"rated":false,"speed":"blitz","timeControl":{"type":"clock","limit":300,"increment":3},"color":"random"}}"#)
        try await waitUntil("the decision is recorded") {
            online.controller.challengeLedger?.row(challengeID: "inco1")?.decisions.isEmpty == false
        }
        let incoming = try XCTUnwrap(online.controller.challengeLedger?.row(challengeID: "inco1"))
        XCTAssertEqual(incoming.direction, .incoming)
        guard case .incomingDecided(.decline) = incoming.state else {
            return XCTFail("a chess960 challenge is declined, got \(incoming.state)")
        }
        online.lichess.sendEvent(#"{"type":"challengeCanceled","challenge":{"id":"inco1"}}"#)
        try await waitUntil("the cancel is recorded") {
            online.controller.challengeLedger?.row(challengeID: "inco1")?.state == .canceledByChallenger
        }
    }

    func testEveryFileIsUnderTheTemporaryDataFolder() async throws {
        let online = try await makeOnline()
        try await online.controller.sendChallenge(to: "alice", request: request)
        let events = try await writtenEvents(online)
        XCTAssertFalse(events.isEmpty)
        let challenges = LichessBotDataDirectory(root: online.root).challengesDirectory
        let names = try FileManager.default.contentsOfDirectory(atPath: challenges.path)
        XCTAssertEqual(names.filter(LichessBotChallengeLog.isDayFileName).count, 1)
        XCTAssertNotEqual(online.root.standardizedFileURL, LichessBotDataDirectory.standard.root.standardizedFileURL)
    }
}
