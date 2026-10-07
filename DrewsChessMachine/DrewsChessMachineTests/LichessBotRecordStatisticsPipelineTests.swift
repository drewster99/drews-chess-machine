import XCTest
@testable import DrewsChessMachine

/// The Record card's statistics pipeline (`LICHESS_BOT_RECORD_STATS_PLAN.md`
/// §4.3): computed off the main actor from each new index, late outcomes
/// dropped, the clock tick, failures, shutdown, and the remembered panel
/// selections.
@MainActor
final class LichessBotRecordStatisticsPipelineTests: XCTestCase {
    private typealias Fixtures = LichessBotStatsFixtures

    /// A controller isolated from the app's real settings, bot data,
    /// Keychain and network, on the defaults suite `suite` and a temporary
    /// data folder removed after the test.
    private func makeController(suite: TemporaryDefaultsSuite) throws -> LichessBotController {
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite.suiteName))
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("LichessBotRecordStatisticsPipelineTests-\(UUID().uuidString)", isDirectory: true)
        try LichessBotSettingsStore.save(LichessBotSettings.testBaseline(), to: defaults)
        let controller = LichessBotController(
            modelProvider: LichessBotFakeModelProvider(snapshot: nil),
            defaults: defaults,
            dataDirectory: LichessBotDataDirectory(root: root),
            services: LichessBotControllerServices(
                makeTransport: { LichessBotFakeLichess() },
                readToken: { _ in nil }
            )
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
        return controller
    }

    private func makePipeline(compute: @escaping LichessBotRecordStatisticsPipeline.Compute = { rows, now, calendar in
        try LichessBotRecordStatistics.compute(rows: rows, now: now, calendar: calendar)
    }) throws -> LichessBotRecordStatisticsPipeline {
        let pipeline = LichessBotRecordStatisticsPipeline(
            defaults: try makeTemporaryDefaults(),
            compute: compute,
            calendar: { Fixtures.utcCalendar }
        )
        addTeardownBlock { @MainActor in
            await pipeline.shutdown(reason: "test teardown")
        }
        return pipeline
    }

    private func rows(_ count: Int) throws -> [LichessBotGameSummary] {
        try (0..<count).map { try Fixtures.row(id: "r\($0)", at: Date().addingTimeInterval(Double(-$0 * 60)), score: 1) }
    }

    private func games(_ state: LichessBotRecordStatisticsState) -> Int? {
        guard case .ready(let statistics) = state else { return nil }
        return statistics[.all].periodRows.allTime.record.all.scored
    }

    func testAFiledGameUpdatesTheSnapshot() async throws {
        let controller = try makeController(suite: try makeTemporaryDefaultsSuite())
        XCTAssertEqual(controller.recordStatistics.state, .loading)
        await controller.refreshIndex()
        await controller.recordStatistics.latestComputation?.value
        XCTAssertEqual(games(controller.recordStatistics.state), 0, "an empty folder is a ready snapshot of no games")

        // File a record the way the store does, then reload the index.
        let record = try Fixtures.record(ourColor: .white, plies: 20, status: "resign", winner: "white") { _ in .postedWithoutDecision }
        let folder = controller.dataDirectory.gamesMonthDirectory(createdAt: record.createdAt)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601withFractionalSeconds
        try encoder.encode(record).write(to: folder.appendingPathComponent("\(record.gameID).json"))
        await controller.refreshIndex()
        await controller.recordStatistics.latestComputation?.value
        XCTAssertEqual(games(controller.recordStatistics.state), 1)
    }

    func testALateOlderResultIsDropped() async throws {
        let release = DispatchSemaphore(value: 0)
        let pipeline = try makePipeline { rows, now, calendar in
            if rows.count == 1 {
                release.wait()
            }
            return try LichessBotRecordStatistics.compute(rows: rows, now: now, calendar: calendar)
        }
        pipeline.indexChanged(rows: try rows(1))
        let first = pipeline.latestComputation
        pipeline.indexChanged(rows: try rows(3))
        let second = pipeline.latestComputation
        release.signal()
        await first?.value
        await second?.value
        XCTAssertEqual(pipeline.requestCount, 2)
        XCTAssertEqual(pipeline.appliedOutcomeCount, 1, "the first request's result arrived after the second was made")
        XCTAssertEqual(games(pipeline.state), 3)
    }

    func testALateOlderErrorIsDropped() async throws {
        struct Boom: Error {}
        let release = DispatchSemaphore(value: 0)
        let pipeline = try makePipeline { rows, now, calendar in
            if rows.count == 1 {
                release.wait()
                throw Boom()
            }
            return try LichessBotRecordStatistics.compute(rows: rows, now: now, calendar: calendar)
        }
        pipeline.indexChanged(rows: try rows(1))
        let first = pipeline.latestComputation
        pipeline.indexChanged(rows: try rows(2))
        let second = pipeline.latestComputation
        release.signal()
        await first?.value
        await second?.value
        XCTAssertEqual(pipeline.appliedOutcomeCount, 1)
        XCTAssertEqual(games(pipeline.state), 2)
    }

    func testTheClockTickRecomputesOnceValidUntilPassesAndNotBefore() async throws {
        let pipeline = try makePipeline()
        pipeline.indexChanged(rows: try rows(2))
        await pipeline.latestComputation?.value
        guard case .ready(let statistics) = pipeline.state else { return XCTFail("expected a snapshot") }
        XCTAssertEqual(pipeline.requestCount, 1)
        pipeline.clockTick(now: statistics.validUntil.addingTimeInterval(-1))
        XCTAssertEqual(pipeline.requestCount, 1, "nothing has changed yet")
        pipeline.clockTick(now: statistics.validUntil)
        XCTAssertEqual(pipeline.requestCount, 2)
        await pipeline.latestComputation?.value
        XCTAssertEqual(games(pipeline.state), 2)
    }

    func testTheClockTickRecomputesWhenTheTimeZoneDiffers() async throws {
        var zone = Fixtures.utcCalendar
        let box = SyncBox(zone)
        let pipeline = LichessBotRecordStatisticsPipeline(defaults: try makeTemporaryDefaults(), calendar: { box.value })
        addTeardownBlock { @MainActor in await pipeline.shutdown(reason: "test teardown") }
        pipeline.indexChanged(rows: try rows(1))
        await pipeline.latestComputation?.value
        pipeline.clockTick(now: Date())
        XCTAssertEqual(pipeline.requestCount, 1)
        zone.timeZone = try XCTUnwrap(TimeZone(identifier: "Pacific/Auckland"))
        box.value = zone
        pipeline.clockTick(now: Date())
        XCTAssertEqual(pipeline.requestCount, 2)
        await pipeline.latestComputation?.value
        guard case .ready(let statistics) = pipeline.state else { return XCTFail("expected a snapshot") }
        XCTAssertEqual(statistics.timeZoneIdentifier, "Pacific/Auckland")
    }

    func testAFailedStateIsShownAndNotRetriedByTheTick() async throws {
        let pipeline = try makePipeline { _, now, _ in
            throw LichessBotStatsPeriods.CalendarError.noWeekInterval(now)
        }
        pipeline.indexChanged(rows: try rows(1))
        await pipeline.latestComputation?.value
        guard case .failed(let text) = pipeline.state else { return XCTFail("expected a failure, got \(pipeline.state)") }
        XCTAssertTrue(text.contains("no week"), text)
        pipeline.clockTick(now: Date().addingTimeInterval(400 * 86_400))
        XCTAssertEqual(pipeline.requestCount, 1, "a failure waits for the next index or time change")
        // The next index change retries.
        pipeline.indexChanged(rows: try rows(1))
        XCTAssertEqual(pipeline.requestCount, 2)
        await pipeline.latestComputation?.value
    }

    func testNothingIsScheduledAfterShutdown() async throws {
        let controller = try makeController(suite: try makeTemporaryDefaultsSuite())
        await controller.shutdown(reason: "test")
        controller.recordStatistics.indexChanged(rows: try rows(2))
        XCTAssertEqual(controller.recordStatistics.requestCount, 0)
        XCTAssertNil(controller.recordStatistics.latestComputation)
        XCTAssertEqual(controller.recordStatistics.state, .loading)
    }

    func testRememberedSelectionsPersistAcrossControllers() throws {
        let suite = try makeTemporaryDefaultsSuite()
        let first = try makeController(suite: suite)
        XCTAssertEqual(first.recordStatistics.rememberedPane, .timeControls)
        XCTAssertEqual(first.recordStatistics.rememberedPeriod, .allTime)
        XCTAssertEqual(first.recordStatistics.rememberedFilter, .all)
        first.recordStatistics.rememberedPane = .selfAssessment
        first.recordStatistics.rememberedPeriod = .thisMonth
        first.recordStatistics.rememberedFilter = .rated
        let second = try makeController(suite: suite)
        XCTAssertEqual(second.recordStatistics.rememberedPane, .selfAssessment)
        XCTAssertEqual(second.recordStatistics.rememberedPeriod, .thisMonth)
        XCTAssertEqual(second.recordStatistics.rememberedFilter, .rated)
    }

    func testRememberedSelectionsThatNoLongerExistAreIgnored() throws {
        let suite = try makeTemporaryDefaultsSuite()
        suite.defaults.set("openings", forKey: LichessBotRecordStatisticsPipeline.paneKey)
        suite.defaults.set("lastDecade", forKey: LichessBotRecordStatisticsPipeline.periodKey)
        suite.defaults.set("blitzOnly", forKey: LichessBotRecordStatisticsPipeline.filterKey)
        let controller = try makeController(suite: suite)
        XCTAssertEqual(controller.recordStatistics.rememberedPane, .timeControls)
        XCTAssertEqual(controller.recordStatistics.rememberedPeriod, .allTime)
        XCTAssertEqual(controller.recordStatistics.rememberedFilter, .all)
    }

    func testTheLogLineCarriesTheAllTimeNumbers() throws {
        let rows = [
            try Fixtures.row(id: "a", at: Date(timeIntervalSince1970: 1_759_500_000), score: 1, opponentRating: 1500, ourRatingDiff: 7),
            try Fixtures.row(id: "b", at: Date(timeIntervalSince1970: 1_759_500_100), score: 0, opponentRating: 1500, ourRatingDiff: -9),
            try Fixtures.row(id: "c", at: Date(timeIntervalSince1970: 1_759_500_200), score: nil, status: "aborted"),
        ]
        let statistics = try LichessBotRecordStatistics.compute(rows: rows, now: Date(timeIntervalSince1970: 1_759_510_000), calendar: Fixtures.utcCalendar)
        XCTAssertEqual(
            LichessBotRecordStatsLogLine.text(statistics, reason: "index", milliseconds: 1.25),
            "[LICHESS-BOT] record stats (index): games=2 W-D-L=1-0-1 score=50.0% perf=1500 rating=-2 brier@20=- not-counted=1 rated-without-diff=0 ms=1.2"
        )
    }
}
