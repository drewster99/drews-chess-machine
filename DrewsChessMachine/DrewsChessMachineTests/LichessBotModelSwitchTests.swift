import XCTest
@testable import DrewsChessMachine

/// The model-switch record (issue #20, LICHESS_BOT_MODEL_IN_USE_PLAN.md):
/// the first generation is a switch from nothing; each later generation
/// records the one it replaced and the reason; a failed build changes
/// nothing.
final class LichessBotModelSwitchTests: XCTestCase {

    private func prepare(_ settings: LichessBotModelSettings, provider: LichessBotHoldableModelProvider,
                         time: LichessBotManualTime, log: @escaping @Sendable (String) -> Void = { _ in }) async throws -> LichessBotModelSlots {
        try await LichessBotModelSlots.prepare(for: settings, provider: provider, time: time,
                                               folderScanner: LichessBotNoModelsFolderScanner(), log: log)
    }

    private func liveTrainer(interval: Int) -> LichessBotModelSettings {
        var settings = LichessBotModelSettings.testBaseline()
        settings.source = .liveTrainer
        settings.liveTrainerRefreshIntervalSeconds = interval
        return settings
    }

    func testTheFirstGenerationIsASwitchFromNothing() async throws {
        let provider = try await LichessBotHoldableModelProvider.make()
        let slots = try await prepare(liveTrainer(interval: 120), provider: provider, time: LichessBotManualTime())
        let record = await slots.lastSwitch
        let current = await slots.current.info
        XCTAssertEqual(record.to, current)
        XCTAssertNil(record.from)
        XCTAssertEqual(record.reason, LichessBotModelSlots.goingOnlineReason)
    }

    func testALaterGenerationRecordsTheOneItReplaced() async throws {
        let provider = try await LichessBotHoldableModelProvider.make()
        let time = LichessBotManualTime()
        let lines = SyncBox<[String]>([])
        let slots = try await prepare(liveTrainer(interval: 60), provider: provider, time: time,
                                      log: { line in lines.mutate { $0.append(line) } })
        let first = await slots.current.info
        time.advance(by: .seconds(61))
        try await slots.refreshIfDue(for: liveTrainer(interval: 60))
        let record = await slots.lastSwitch
        let second = await slots.current.info
        XCTAssertEqual(second.generationID, 2, "the interval elapsed, so a new generation was built")
        XCTAssertEqual(record.to, second)
        XCTAssertEqual(record.from, first)
        XCTAssertEqual(record.reason, "live-trainer interval elapsed")
        XCTAssertTrue(lines.value.contains { $0.contains("model generation 2 ready") && $0.contains("replacing generation 1 \(first.modelID)") },
                      "the ready line names the generation it replaced: \(lines.value)")
    }

    func testARefreshThatBuildsNothingLeavesTheRecord() async throws {
        let provider = try await LichessBotHoldableModelProvider.make()
        let time = LichessBotManualTime()
        let slots = try await prepare(liveTrainer(interval: 120), provider: provider, time: time)
        let before = await slots.lastSwitch
        time.advance(by: .seconds(10))
        try await slots.refreshIfDue(for: liveTrainer(interval: 120))
        let after = await slots.lastSwitch
        XCTAssertEqual(after, before)
    }

    func testAFailedBuildLeavesTheRecord() async throws {
        let provider = try await LichessBotHoldableModelProvider.make()
        let time = LichessBotManualTime()
        let slots = try await prepare(liveTrainer(interval: 60), provider: provider, time: time)
        let before = await slots.lastSwitch
        provider.trainerExists.mutate { $0 = false }
        time.advance(by: .seconds(61))
        do {
            try await slots.refreshIfDue(for: liveTrainer(interval: 60))
            XCTFail("a refresh with no trainer must throw")
        } catch LichessBotModelError.noTrainer {
            // The expected failure; the record must not move.
        } catch {
            XCTFail("unexpected error: \(error)")
        }
        let after = await slots.lastSwitch
        XCTAssertEqual(after, before)
    }
}
