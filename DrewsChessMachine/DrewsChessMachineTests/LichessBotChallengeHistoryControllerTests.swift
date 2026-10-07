//
//  LichessBotChallengeHistoryControllerTests.swift
//  DrewsChessMachineTests
//
//  The controller's back-fill (challenge-log plan §3.7, "Running it"): once
//  the challenge log loads, the history is rebuilt from the protocol log
//  into the data folder's `Challenges/reconstructed-from-protocol.json`, and
//  a rerun with unchanged inputs writes nothing. Every file is in the
//  test's temporary data folder.
//

import XCTest
@testable import DrewsChessMachine

@MainActor
final class LichessBotChallengeHistoryControllerTests: XCTestCase {

    private func makeController(accountID: String) async throws -> (LichessBotController, LichessBotDataDirectory) {
        let defaults = try makeTemporaryDefaults()
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("LichessBotChallengeHistoryControllerTests-\(UUID().uuidString)", isDirectory: true)
        var settings = LichessBotSettings.testBaseline()
        settings.connection.expectedAccountID = accountID
        try LichessBotSettingsStore.save(settings, to: defaults)
        let directory = LichessBotDataDirectory(root: root)
        let controller = LichessBotController(
            modelProvider: try await LichessBotFakeModelProvider.randomChampion(),
            defaults: defaults,
            dataDirectory: directory
        )
        addTeardownBlock { @MainActor in
            await controller.shutdown(reason: "test teardown")
            if FileManager.default.fileExists(atPath: root.path) {
                do {
                    try FileManager.default.removeItem(at: root)
                } catch {
                    XCTFail("cleanup failed: \(error)")
                }
            }
        }
        return (controller, directory)
    }

    private func writeProtocolDay(_ directory: LichessBotDataDirectory) throws {
        try directory.createDirectories()
        let event = #"{"type":"challenge","challenge":{"id":"rIn","status":"created","challenger":{"id":"carol","name":"Carol","rating":1500},"destUser":{"id":"drewschessmachine","name":"DrewsChessMachine","rating":1500},"variant":{"key":"standard"},"rated":true,"speed":"blitz","timeControl":{"type":"clock","limit":180,"increment":2},"color":"random"}}"#
        let entry = LichessBotProtocolEntry(at: LichessBotChallengeLogFixtures.start, kind: .stream, gameID: nil, message: event, fields: ["stream": "event"])
        try LichessBotJSONLines.encodeLine(entry).write(to: directory.protocolLogURL(for: LichessBotChallengeLogFixtures.start))
    }

    private func waitForHistory(_ controller: LichessBotController) async throws {
        for _ in 0..<2000 {
            if case .ready = controller.challengeHistoryStatus { return }
            if case .failed(let reason) = controller.challengeHistoryStatus {
                return XCTFail("rebuild failed: \(reason)")
            }
            try await Task.sleep(for: .milliseconds(10))
        }
        XCTFail("timed out waiting for the challenge history")
    }

    func testLoadingTheChallengeLogRebuildsTheHistoryIntoTheDataFolder() async throws {
        let (controller, directory) = try await makeController(accountID: "drewschessmachine")
        try writeProtocolDay(directory)
        await controller.loadChallengeLog()
        try await waitForHistory(controller)
        XCTAssertEqual(controller.challengeHistory?.rows.count, 1)
        XCTAssertEqual(controller.challengeHistory?.rows.first?.direction, .incoming)
        let url = directory.reconstructedChallengesURL
        XCTAssertEqual(try FileSafety.existingItem(at: url)?.kind, .regularFile)
        XCTAssertTrue(url.path.hasPrefix(directory.root.path))

        let before = try FileManager.default.attributesOfItem(atPath: url.path)[.modificationDate] as? Date
        await controller.rebuildChallengeHistory()
        guard case .ready(.unchanged, _) = controller.challengeHistoryStatus else {
            return XCTFail("a rerun with unchanged inputs must write nothing, got \(controller.challengeHistoryStatus)")
        }
        let after = try FileManager.default.attributesOfItem(atPath: url.path)[.modificationDate] as? Date
        XCTAssertEqual(before, after)
    }
}
