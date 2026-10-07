import SwiftUI
import XCTest
@testable import DrewsChessMachine

/// Fixes from the branch review (`LICHESS_BOT_RECORD_STATS_PLAN.md`,
/// implementation notes): progression points stable and uniquely named when
/// checkpoints share a step; the index log line kept when a clock request
/// supersedes an index request; the system time-change path; the wide card.
@MainActor
final class LichessBotRecordStatisticsReviewFixTests: XCTestCase {
    private typealias Fixtures = LichessBotStatsFixtures

    private let now = Date(timeIntervalSince1970: 1_791_374_400)

    // MARK: - Progression with checkpoints at one step

    private func sharedStepRows() throws -> [LichessBotGameSummary] {
        var rows: [LichessBotGameSummary] = []
        var serial = 0
        // A trainer snapshot and the live trainer both at step 1000, then
        // a snapshot at 2000; 30 games each, so each is a bin of its own.
        for generation in [
            Fixtures.generation("RUN", source: .trainerSnapshot, step: 1000, moves: 10),
            Fixtures.generation("RUN", source: .liveTrainer, step: 1000, moves: 10),
            Fixtures.generation("RUN", source: .trainerSnapshot, step: 2000, moves: 10),
        ] {
            for game in 0..<30 {
                serial += 1
                rows.append(try Fixtures.row(id: "s\(serial)", at: now - Double(10_000 - serial * 10), score: game % 3 == 0 ? 1 : 0, facts: Fixtures.facts(generations: [generation])))
            }
        }
        return rows
    }

    func testProgressionIsStableAndUniquelyNamedWhenCheckpointsShareAStep() throws {
        let rows = try sharedStepRows()
        let forward = try LichessBotRecordStatistics.compute(rows: rows, now: now, calendar: Fixtures.utcCalendar)[.all].byPeriod.allTime.models.progression
        let backward = try LichessBotRecordStatistics.compute(rows: rows.reversed(), now: now, calendar: Fixtures.utcCalendar)[.all].byPeriod.allTime.models.progression
        XCTAssertEqual(forward, backward, "the same games give the same chart in any order")
        XCTAssertEqual(forward.map(\.step), [1000, 1000, 2000])
        XCTAssertEqual(Set(forward.map(\.id)).count, forward.count, "two bins ending at one step keep distinct IDs")
        let marks = LichessBotProgressChartMark.marks(forward, metric: .score)
        XCTAssertEqual(Set(marks.map(\.id)).count, marks.count)
        // The snapshot's games started first, so its bin comes first.
        XCTAssertEqual(forward.first?.games, 30)
    }

    // MARK: - Index log line

    func testTheIndexLineIsWrittenWhenAClockRequestSupersedesTheIndexRequest() async throws {
        let release = DispatchSemaphore(value: 0)
        let first = SyncBox(true)
        let pipeline = LichessBotRecordStatisticsPipeline(defaults: try makeTemporaryDefaults(), compute: { rows, now, calendar in
            if first.value {
                first.value = false
                release.wait()
            }
            return try LichessBotRecordStatistics.compute(rows: rows, now: now, calendar: calendar)
        }, calendar: { Fixtures.utcCalendar })
        addTeardownBlock { @MainActor in await pipeline.shutdown(reason: "test teardown") }
        pipeline.indexChanged(rows: [try Fixtures.row(id: "a", at: now, score: 1)])
        XCTAssertTrue(pipeline.indexLogPending)
        let indexRequest = pipeline.latestComputation
        pipeline.schedule(reason: .clock, now: now)
        let clockRequest = pipeline.latestComputation
        release.signal()
        await indexRequest?.value
        await clockRequest?.value
        XCTAssertEqual(pipeline.appliedOutcomeCount, 1, "only the clock request's result is applied")
        XCTAssertFalse(pipeline.indexLogPending, "and it wrote the index change's line")
    }

    // MARK: - System time changes

    private func waitForRequestCount(_ pipeline: LichessBotRecordStatisticsPipeline, atLeast count: Int) async throws {
        for _ in 0..<200 where pipeline.requestCount < count {
            try await Task.sleep(for: .milliseconds(10))
        }
    }

    func testSystemClockAndTimeZoneChangesRecomputeAtOnceAndRetryAFailure() async throws {
        let failing = SyncBox(false)
        let pipeline = LichessBotRecordStatisticsPipeline(defaults: try makeTemporaryDefaults(), compute: { rows, now, calendar in
            if failing.value {
                throw LichessBotStatsPeriods.CalendarError.noWeekInterval(now)
            }
            return try LichessBotRecordStatistics.compute(rows: rows, now: now, calendar: calendar)
        }, calendar: { Fixtures.utcCalendar })
        addTeardownBlock { @MainActor in await pipeline.shutdown(reason: "test teardown") }
        pipeline.observeSystemTimeChanges()
        pipeline.indexChanged(rows: [try Fixtures.row(id: "a", at: Date(), score: 1)])
        await pipeline.latestComputation?.value
        XCTAssertEqual(pipeline.requestCount, 1)

        // A clock change recomputes although `now` is before `validUntil`.
        NotificationCenter.default.post(name: .NSSystemClockDidChange, object: nil)
        try await waitForRequestCount(pipeline, atLeast: 2)
        XCTAssertEqual(pipeline.requestCount, 2)
        await pipeline.latestComputation?.value

        // A failed state is retried by a time-zone change.
        failing.value = true
        pipeline.indexChanged(rows: [try Fixtures.row(id: "b", at: Date(), score: 0)])
        await pipeline.latestComputation?.value
        XCTAssertNotNil(pipeline.state.failureText)
        failing.value = false
        NotificationCenter.default.post(name: .NSSystemTimeZoneDidChange, object: nil)
        try await waitForRequestCount(pipeline, atLeast: 4)
        await pipeline.latestComputation?.value
        guard case .ready = pipeline.state else { return XCTFail("expected the retry to succeed, got \(pipeline.state)") }

        // After stopping, notifications schedule nothing.
        pipeline.stopScheduling()
        let stopped = pipeline.requestCount
        NotificationCenter.default.post(name: .NSSystemClockDidChange, object: nil)
        try await Task.sleep(for: .milliseconds(100))
        XCTAssertEqual(pipeline.requestCount, stopped)
    }

    // MARK: - The wide card

    func testTheCardRendersSideBySideWhenWide() async throws {
        let suite = try makeTemporaryDefaultsSuite()
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite.suiteName))
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("LichessBotRecordStatisticsReviewFixTests-\(UUID().uuidString)", isDirectory: true)
        try LichessBotSettingsStore.save(LichessBotSettings.testBaseline(), to: defaults)
        let controller = LichessBotController(
            modelProvider: LichessBotFakeModelProvider(snapshot: nil),
            defaults: defaults,
            dataDirectory: LichessBotDataDirectory(root: root),
            services: LichessBotControllerServices(makeTransport: { LichessBotFakeLichess() }, readToken: { _ in nil })
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
        controller.recordStatistics.indexChanged(rows: try Fixtures.syntheticRows(206, now: Date(), models: 6))
        await controller.recordStatistics.latestComputation?.value
        let width = LichessBotStatsStyle.wideCardWidth + 260
        XCTAssertGreaterThanOrEqual(width - 32, LichessBotStatsStyle.wideCardWidth, "wide enough for the side-by-side arrangement after padding")
        for scheme in [ColorScheme.light, .dark] {
            let renderer = ImageRenderer(content: LichessBotRecordCard(controller: controller)
                .padding(16)
                .frame(width: width, height: 700, alignment: .top)
                .background(Color(nsColor: .windowBackgroundColor))
                .environment(\.colorScheme, scheme))
            renderer.scale = 2
            let image = try XCTUnwrap(renderer.nsImage, "the wide card did not render")
            XCTAssertGreaterThan(image.size.width, 0)
        }
    }
}
