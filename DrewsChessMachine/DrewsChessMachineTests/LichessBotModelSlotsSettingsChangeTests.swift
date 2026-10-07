import XCTest
@testable import DrewsChessMachine

/// Model settings that don't change which weights are played — the refresh
/// interval, the mid-game toggle, a file path left over while another
/// source is chosen — are adopted without building a new generation
/// (follow-lineage plan §3.4, "Settings comparison fix").
final class LichessBotModelSlotsSettingsChangeTests: XCTestCase {

    private func prepare(_ settings: LichessBotModelSettings, provider: LichessBotHoldableModelProvider, time: LichessBotManualTime) async throws -> LichessBotModelSlots {
        try await LichessBotModelSlots.prepare(for: settings, provider: provider, time: time, folderScanner: LichessBotNoModelsFolderScanner(), log: { _ in })
    }

    private func liveTrainer(interval: Int = 120, filePath: String? = nil) -> LichessBotModelSettings {
        var settings = LichessBotModelSettings.testBaseline()
        settings.source = .liveTrainer
        settings.liveTrainerRefreshIntervalSeconds = interval
        settings.filePath = filePath
        return settings
    }

    func testIntervalEditDoesNotRebuild() async throws {
        let provider = try await LichessBotHoldableModelProvider.make()
        let time = LichessBotManualTime()
        let slots = try await prepare(liveTrainer(interval: 120), provider: provider, time: time)
        time.advance(by: .seconds(10))
        try await slots.refreshIfDue(for: liveTrainer(interval: 300))
        let info = await slots.current.info
        XCTAssertEqual(info.generationID, 1, "an interval edit builds nothing")
        XCTAssertEqual(provider.trainerSnapshots.value, 1)
    }

    func testTogglingMidGameRefreshDoesNotRebuild() async throws {
        let provider = try await LichessBotHoldableModelProvider.make()
        let time = LichessBotManualTime()
        var settings = LichessBotModelSettings.testBaseline()
        settings.midGameRefresh = false
        let slots = try await prepare(settings, provider: provider, time: time)
        settings.midGameRefresh = true
        try await slots.refreshIfDue(for: settings)
        let info = await slots.current.info
        XCTAssertEqual(info.generationID, 1, "the toggle builds nothing")
        XCTAssertEqual(provider.championSnapshots.value, 1)
    }

    func testLeftoverFilePathDoesNotRebuildTheLiveTrainer() async throws {
        let provider = try await LichessBotHoldableModelProvider.make()
        let time = LichessBotManualTime()
        let slots = try await prepare(liveTrainer(filePath: "/models/a.safetensors"), provider: provider, time: time)
        try await slots.refreshIfDue(for: liveTrainer(filePath: "/models/b.safetensors"))
        let info = await slots.current.info
        XCTAssertEqual(info.generationID, 1, "a file path the live trainer doesn't use builds nothing")
        XCTAssertEqual(provider.trainerSnapshots.value, 1)
    }
}
