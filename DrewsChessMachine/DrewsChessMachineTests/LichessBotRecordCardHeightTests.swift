import SwiftUI
import XCTest
@testable import DrewsChessMachine

/// The Record card's height follows its content: a pane taller than the
/// dragged height grows the card instead of being drawn past its slot, over
/// the recent games stacked below (owner report 2026-10-08: the Opponents
/// pane overlapped the recent games).
@MainActor
final class LichessBotRecordCardHeightTests: XCTestCase {
    private typealias Fixtures = LichessBotStatsFixtures

    /// 60 games, each against its own opponent, so the Opponents pane lists
    /// 60 rows; the card is narrow (stacked) and dragged short.
    private func controller() throws -> (LichessBotController, UserDefaults) {
        let suite = try makeTemporaryDefaultsSuite()
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite.suiteName))
        defaults.set(160.0, forKey: "lichessBot.overview.recordCardHeight")
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("LichessBotRecordCardHeightTests-\(UUID().uuidString)", isDirectory: true)
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
        return (controller, defaults)
    }

    private func cardHeight(_ controller: LichessBotController, defaults: UserDefaults) -> CGFloat {
        NSHostingView(rootView: LichessBotRecordCard(controller: controller)
            .frame(width: 900)
            .defaultAppStorage(defaults)).fittingSize.height
    }

    func testATallPaneGrowsTheCard() async throws {
        let (controller, defaults) = try controller()
        let now = Date()
        let rows = try (0..<60).map { index in
            try Fixtures.row(id: "g\(index)", at: now.addingTimeInterval(-Double(index) * 600), score: index.isMultiple(of: 2) ? 1 : 0,
                             opponentID: "opponent\(index)", opponentName: "Opponent \(index)")
        }
        controller.recordStatistics.indexChanged(rows: rows)
        await controller.recordStatistics.latestComputation?.value

        guard case .ready(let statistics) = controller.recordStatistics.state else { return XCTFail("expected a snapshot") }
        func heights(_ pane: LichessBotRecordPane) -> (pane: CGFloat, card: CGFloat) {
            controller.recordStatistics.rememberedPane = pane
            let panel = NSHostingView(rootView: LichessBotRecordPanel(account: nil, pipeline: controller.recordStatistics, statistics: statistics)
                .frame(width: 900)).fittingSize.height
            return (panel, cardHeight(controller, defaults: defaults))
        }
        let short = heights(.timeControls)
        let tall = heights(.opponents)
        let minimum = LichessBotStatsStyle.paneMinimumHeight
        XCTAssertGreaterThan(tall.pane, minimum, "the opponents pane is taller than the pane minimum")
        // The card is dragged short, so its height is its content's: switching
        // panes changes it by exactly the panes' difference, each pane taking
        // at least the pane minimum.
        XCTAssertEqual(tall.card - short.card, max(tall.pane, minimum) - max(short.pane, minimum), accuracy: 1,
                       "panes \(short.pane) → \(tall.pane), card \(short.card) → \(tall.card)")
    }
}
