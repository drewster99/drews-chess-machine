//
//  LichessBotTestWindowTests.swift
//  DrewsChessMachineTests
//
//  A test must never open the real Lichess Bot window. Going offline (or
//  quitting) with a game in play brings the window up so the operator sees
//  the "Finishing games" state; in the test host that put a live-looking bot
//  window on the owner's screen ("Lichess Bot — DrewsChessMachine —
//  Offline", game calice vs alice, 2026-10-10), which stayed after the test
//  ended and was mistaken for the running bot.
//

import AppKit
import XCTest
@testable import DrewsChessMachine

@MainActor
final class LichessBotTestWindowTests: XCTestCase {

    private let request = LichessBotOutgoingChallenge(rated: false, clockLimitSeconds: 300, clockIncrementSeconds: 3, color: .random)

    /// Every Lichess Bot window this process has open.
    private static func botWindows() -> [NSWindow] {
        NSApp.windows.filter { $0.windowController is LichessBotWindowController }
    }

    override func tearDown() async throws {
        // Clean up after a failure too: a window left open is exactly the
        // defect under test.
        for window in Self.botWindows() {
            window.close()
        }
        try await super.tearDown()
    }

    private func waitUntil(_ description: String, _ condition: () -> Bool) async throws {
        for _ in 0..<3000 {
            if condition() { return }
            try await Task.sleep(for: .milliseconds(10))
        }
        XCTFail("timed out waiting until \(description)")
        throw CancellationError()
    }

    func testGoingOfflineWithAGameInPlayOpensNoWindowInATest() async throws {
        XCTAssertEqual(Self.botWindows(), [], "no bot window before the test")
        let lichess = LichessBotFakeLichess(accountPerfsJSON: #"{"blitz":{"games":100,"rating":1500,"rd":50,"prog":0}}"#)
        let defaults = try makeTemporaryDefaults()
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("LichessBotTestWindowTests-\(UUID().uuidString)", isDirectory: true)
        var settings = LichessBotSettings.testBaseline()
        settings.chat.greetingEnabled = false
        settings.connection.preventSleepWhileOnline = false
        settings.challenge.outgoingChallengeTimeoutSeconds = 0
        settings.challenge.gamesReservedForHumans = 0
        settings.matchmaking.enabled = false
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
        await controller.goOnline()
        try await controller.sendChallenge(to: "alice", request: request)
        lichess.startGame(against: "alice")
        try await waitUntil("the game is in play") { controller.games.count == 1 }

        await controller.goOffline()

        XCTAssertEqual(Self.botWindows(), [], "going offline in a test opened the real Lichess Bot window")
    }
}
