//
//  LichessBotOutcomeLogFromChallengeLogTests.swift
//  DrewsChessMachineTests
//
//  The controller's outcome log after P6 (challenge-log plan §3.8): a fold
//  of the challenge log, kept current as challenges are sent and answered,
//  with `challenge-outcomes.json` neither written nor read — an old file
//  left from an earlier build stays exactly as it was.
//

import XCTest
@testable import DrewsChessMachine

@MainActor
final class LichessBotOutcomeLogFromChallengeLogTests: XCTestCase {

    private let request = LichessBotOutgoingChallenge(rated: false, clockLimitSeconds: 300, clockIncrementSeconds: 3, color: .random)

    private func waitUntil(_ description: String, _ condition: () -> Bool) async throws {
        for _ in 0..<3000 {
            if condition() { return }
            try await Task.sleep(for: .milliseconds(10))
        }
        XCTFail("timed out waiting until \(description)")
        throw CancellationError()
    }

    func testTheOutcomeLogFollowsTheChallengeLogAndTheOldFileIsLeftAlone() async throws {
        let lichess = LichessBotFakeLichess()
        let defaults = try makeTemporaryDefaults()
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("LichessBotOutcomeLogFromChallengeLogTests-\(UUID().uuidString)", isDirectory: true)
        let directory = LichessBotDataDirectory(root: root)
        try directory.createDirectories()
        // An outcome file an earlier build wrote.
        let oldBytes = Data(#"{"records":[]}"#.utf8)
        try oldBytes.write(to: directory.challengeOutcomesURL)
        var settings = LichessBotSettings.testBaseline()
        settings.chat.greetingEnabled = false
        settings.connection.preventSleepWhileOnline = false
        settings.challenge.outgoingChallengeTimeoutSeconds = 0
        try LichessBotSettingsStore.save(settings, to: defaults)
        let token = LichessBotFakeLichess.token
        let controller = LichessBotController(
            modelProvider: try await LichessBotFakeModelProvider.randomChampion(),
            defaults: defaults,
            dataDirectory: directory,
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
        try await waitUntil("the event stream is open") { lichess.eventStreamIsOpen }
        XCTAssertNotNil(controller.challengeOutcomeLog)

        try await controller.sendChallenge(to: "alice", request: request)
        XCTAssertEqual(controller.challengeOutcomeLog?.records.map(\.challengeID), ["calice"])
        XCTAssertEqual(controller.challengeOutcomeLog?.summary(now: Date()).pending, 1)

        lichess.sendEvent(#"{"type":"challengeDeclined","challenge":{"id":"calice","declineReason":"No bots","declineReasonKey":"nobot"}}"#)
        try await waitUntil("the decline is folded in") { controller.challengeOutcomeLog?.summary(now: Date()).declined == 1 }
        XCTAssertEqual(controller.challengeOutcomeLog?.records.first?.outcome, .declined(.known(.noBot)))

        await controller.goOffline()
        XCTAssertEqual(try Data(contentsOf: directory.challengeOutcomesURL), oldBytes, "the old outcome file is neither written nor replaced")
    }
}
