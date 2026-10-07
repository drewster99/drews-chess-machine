import SwiftUI
import XCTest
@testable import DrewsChessMachine

/// Render smoke tests for the Overview's Record card
/// (`LICHESS_BOT_RECORD_STATS_PLAN.md` §7 P3–P8): the card at the narrowest
/// detail pane (680 points, stacked) and at the default window width
/// (1,180 points), light and dark, in each state (loading, ready, failed),
/// with no games, a 206-game synthetic record and all wins, on every pane.
/// Each must render a non-empty image; the images are written to a
/// temporary folder (printed) for visual inspection.
@MainActor
final class LichessBotRecordCardRenderTests: XCTestCase {
    private typealias Fixtures = LichessBotStatsFixtures

    private let narrow = CGSize(width: 680, height: 640)
    private let wide = CGSize(width: 1_180, height: 640)

    private func render<Content: View>(_ view: Content, size: CGSize, name: String, scheme: ColorScheme) throws {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent("LichessBotRenders", isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let renderer = ImageRenderer(content: view
            .padding(16)
            .frame(width: size.width, height: size.height, alignment: .top)
            .background(Color(nsColor: .windowBackgroundColor))
            .environment(\.colorScheme, scheme))
        renderer.scale = 2
        let image = try XCTUnwrap(renderer.nsImage, "\(name) did not render")
        let tiff = try XCTUnwrap(image.tiffRepresentation)
        let bitmap = try XCTUnwrap(NSBitmapImageRep(data: tiff))
        XCTAssertGreaterThan(bitmap.pixelsWide, 0)
        let png = try XCTUnwrap(bitmap.representation(using: .png, properties: [:]))
        let url = folder.appendingPathComponent("record-\(name)-\(scheme == .dark ? "dark" : "light").png")
        try png.write(to: url)
        print("LICHESS-BOT-RENDER \(url.path)")
    }

    private func makeController() throws -> LichessBotController {
        let suite = try makeTemporaryDefaultsSuite()
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite.suiteName))
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("LichessBotRecordCardRenderTests-\(UUID().uuidString)", isDirectory: true)
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

    /// Every pane at both widths in both schemes.
    private func renderEveryPane(_ controller: LichessBotController, name: String) throws {
        for pane in LichessBotRecordPane.allCases {
            controller.recordStatistics.rememberedPane = pane
            for (size, width) in [(narrow, "680"), (wide, "1180")] {
                for scheme in [ColorScheme.light, .dark] {
                    try render(LichessBotRecordCard(controller: controller), size: size, name: "\(name)-\(pane.rawValue)-\(width)", scheme: scheme)
                }
            }
        }
    }

    func testLoadingState() throws {
        let controller = try makeController()
        XCTAssertEqual(controller.recordStatistics.state, .loading)
        for scheme in [ColorScheme.light, .dark] {
            try render(LichessBotRecordCard(controller: controller), size: narrow, name: "loading-680", scheme: scheme)
            try render(LichessBotRecordCard(controller: controller), size: wide, name: "loading-1180", scheme: scheme)
        }
    }

    func testNoGames() async throws {
        let controller = try makeController()
        await controller.refreshIndex()
        await controller.recordStatistics.latestComputation?.value
        guard case .ready = controller.recordStatistics.state else { return XCTFail("expected a snapshot") }
        try renderEveryPane(controller, name: "empty")
    }

    func testSyntheticRecordOnEveryPaneFilterAndPeriod() async throws {
        let controller = try makeController()
        controller.recordStatistics.indexChanged(rows: try Fixtures.syntheticRows(206, now: Date(), models: 6))
        await controller.recordStatistics.latestComputation?.value
        guard case .ready = controller.recordStatistics.state else { return XCTFail("expected a snapshot") }
        try renderEveryPane(controller, name: "206")
        controller.recordStatistics.rememberedFilter = .rated
        controller.recordStatistics.rememberedPeriod = .today
        try renderEveryPane(controller, name: "206-rated-today")
    }

    func testAllWins() async throws {
        let controller = try makeController()
        controller.recordStatistics.indexChanged(rows: try Fixtures.syntheticRows(12, now: Date(), models: 2, allWins: true))
        await controller.recordStatistics.latestComputation?.value
        try renderEveryPane(controller, name: "all-wins")
    }

    /// `ImageRenderer` draws neither a `ScrollView`'s content nor AppKit
    /// controls (segmented pickers), so the card's own renders show the
    /// frame only. The tables and panes are drawn here on their own, at the
    /// narrow width, so their columns can be checked by eye.
    func testTablesAndPanesForInspection() throws {
        let statistics = try LichessBotRecordStatistics.compute(
            rows: try Fixtures.syntheticRows(206, now: Date(), models: 6),
            now: Date(),
            calendar: .current
        )
        let controller = try makeController()
        // The period table's ideal width sets `wideCardWidth` (§5.1): the
        // side-by-side arrangement must leave the table its full width
        // beside the recent games' minimum.
        let gridWidth = NSHostingView(rootView: LichessBotRecordPeriodGrid(rows: statistics[.all].periodRows)).fittingSize.width
        print("LICHESS-BOT-RECORD period grid ideal width: \(gridWidth)")
        XCTAssertLessThanOrEqual(gridWidth + LichessBotStatsStyle.recentGamesMinimumWidth + 17, LichessBotStatsStyle.wideCardWidth)
        for scheme in [ColorScheme.light, .dark] {
            for filter in LichessBotStatsFilter.allCases {
                try render(LichessBotRecordPeriodGrid(rows: statistics[filter].periodRows), size: narrow, name: "grid-\(filter.rawValue)", scheme: scheme)
            }
            try render(LichessBotRecordFootnote(statistics: statistics[.all]), size: CGSize(width: 680, height: 80), name: "footnote", scheme: scheme)
            try renderPanes(statistics, controller: controller, scheme: scheme)
        }
    }

    /// The panel (no scroll view of its own) on each pane.
    private func renderPanes(_ statistics: LichessBotRecordStatistics, controller: LichessBotController, scheme: ColorScheme) throws {
        for pane in LichessBotRecordPane.allCases {
            controller.recordStatistics.rememberedPane = pane
            try render(
                LichessBotRecordPanel(account: controller.account, pipeline: controller.recordStatistics, statistics: statistics),
                size: CGSize(width: 680, height: 900), name: "pane-\(pane.rawValue)", scheme: scheme
            )
        }
    }

    func testFailedState() async throws {
        let controller = try makeController()
        let pipeline = LichessBotRecordStatisticsPipeline(defaults: try makeTemporaryDefaults(), compute: { _, now, _ in
            throw LichessBotStatsPeriods.CalendarError.noWeekInterval(now)
        })
        addTeardownBlock { @MainActor in await pipeline.shutdown(reason: "test teardown") }
        pipeline.indexChanged(rows: [])
        await pipeline.latestComputation?.value
        XCTAssertNotNil(pipeline.state.failureText)
        for scheme in [ColorScheme.light, .dark] {
            try render(
                LichessBotRecordCardLayout(statistics: LichessBotRecordStatisticsColumn(account: nil, pipeline: pipeline), recentGames: LichessBotRecentGamesSection(controller: controller)).frame(height: 400),
                size: narrow, name: "failed-680", scheme: scheme
            )
        }
    }
}
