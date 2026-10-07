//
//  LichessBotChallengeLogViewRenderTests.swift
//  DrewsChessMachineTests
//
//  Render smoke tests for the Challenge Log window (challenge-log plan
//  §3.9): the whole window's view over a real controller with an empty log,
//  live rows only, rebuilt rows only and both; the table with a row in every
//  state; the filter bar; and the outcomes card with its "Challenge Log…"
//  button. Each is drawn off-screen in light and dark and must produce a
//  non-empty image; the images are written to a temporary folder (printed)
//  for inspection. Every controller uses a temporary data folder and
//  defaults suite, and nothing is written anywhere else.
//

import AppKit
import SwiftUI
import XCTest
@testable import DrewsChessMachine

@MainActor
final class LichessBotChallengeLogViewRenderTests: XCTestCase {

    private typealias R = LichessBotChallengeLogRowFixtures

    private static let renderFolder = FileManager.default.temporaryDirectory.appendingPathComponent("LichessBotRenders", isDirectory: true)
    private static let windowSize = CGSize(width: 1500, height: 560)

    /// Through a hosting view, which draws the AppKit-backed `Table` that
    /// `ImageRenderer` can't. The window is never ordered on screen.
    private func renderHosted<Content: View>(_ view: Content, size: CGSize, name: String, scheme: ColorScheme) async throws {
        try FileManager.default.createDirectory(at: Self.renderFolder, withIntermediateDirectories: true)
        let host = NSHostingView(rootView: view
            .frame(width: size.width, height: size.height)
            .background(Color(nsColor: .windowBackgroundColor)))
        let window = NSWindow(contentRect: NSRect(origin: .zero, size: size), styleMask: [.titled], backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        window.appearance = NSAppearance(named: scheme == .dark ? .darkAqua : .aqua)
        window.contentView = host
        defer {
            window.contentView = nil
        }
        host.layoutSubtreeIfNeeded()
        // The view builds its rows in main-actor tasks started by `onChange`.
        try await Task.sleep(for: .milliseconds(400))
        host.layoutSubtreeIfNeeded()
        let bitmap = try XCTUnwrap(host.bitmapImageRepForCachingDisplay(in: host.bounds), "\(name) did not render")
        host.cacheDisplay(in: host.bounds, to: bitmap)
        XCTAssertGreaterThan(bitmap.pixelsWide, 0)
        let png = try XCTUnwrap(bitmap.representation(using: .png, properties: [:]))
        let url = Self.renderFolder.appendingPathComponent("\(name)-\(scheme == .dark ? "dark" : "light").png")
        try png.write(to: url)
        print("LICHESS-BOT-RENDER \(url.path)")
    }

    private func makeController() throws -> (LichessBotController, LichessBotDataDirectory) {
        let defaults = try makeTemporaryDefaults()
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("LichessBotChallengeLogViewRenderTests-\(UUID().uuidString)", isDirectory: true)
        var settings = LichessBotSettings.testBaseline()
        settings.connection.expectedAccountID = R.ourAccountID
        try LichessBotSettingsStore.save(settings, to: defaults)
        let directory = LichessBotDataDirectory(root: root)
        let controller = LichessBotController(
            modelProvider: LichessBotFakeModelProvider(snapshot: nil),
            defaults: defaults,
            dataDirectory: directory,
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
        return (controller, directory)
    }

    /// Protocol lines the history is rebuilt from, written into the
    /// temporary data folder before the challenge log loads.
    private func writeProtocolDay(_ directory: LichessBotDataDirectory) throws {
        try directory.createDirectories()
        try R.protocolDayData().write(to: directory.protocolLogURL(for: LichessBotChallengeLogFixtures.start))
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

    /// Load the log (rebuilding the history), then record `live` facts.
    private func prepare(_ controller: LichessBotController, live: Bool) async throws {
        await controller.loadChallengeLog()
        try await waitForHistory(controller)
        if live {
            for entry in R.ledgerEntries() {
                controller.challengeLogRecorder.record(entry.event)
            }
        }
    }

    private func renderWindow(_ controller: LichessBotController, name: String) async throws {
        for scheme in [ColorScheme.light, .dark] {
            try await renderHosted(LichessBotChallengeLogView(controller: controller), size: Self.windowSize, name: name, scheme: scheme)
        }
    }

    func testEmptyLogRenders() async throws {
        let (controller, _) = try makeController()
        try await prepare(controller, live: false)
        XCTAssertEqual(controller.challengeLedger?.rows.count, 0)
        XCTAssertEqual(controller.challengeHistory?.rows.count, 0)
        try await renderWindow(controller, name: "challenge-log-empty")
    }

    func testLiveOnlyLogRenders() async throws {
        let (controller, _) = try makeController()
        try await prepare(controller, live: true)
        XCTAssertEqual(controller.challengeLedger?.rows.count, R.ledger().rows.count)
        XCTAssertEqual(controller.challengeHistory?.rows.count, 0)
        try await renderWindow(controller, name: "challenge-log-live")
    }

    func testReconstructedOnlyLogRenders() async throws {
        let (controller, directory) = try makeController()
        try writeProtocolDay(directory)
        try await prepare(controller, live: false)
        XCTAssertEqual(controller.challengeLedger?.rows.count, 0)
        XCTAssertEqual(controller.challengeHistory?.rows.count, try R.reconstruction().rows.count)
        try await renderWindow(controller, name: "challenge-log-reconstructed")
    }

    func testMixedLogRenders() async throws {
        let (controller, directory) = try makeController()
        try writeProtocolDay(directory)
        try await prepare(controller, live: true)
        let rows = LichessBotChallengeLogRow.rows(ledger: controller.challengeLedger, reconstruction: controller.challengeHistory, pendingChallengeIDs: [])
        XCTAssertTrue(rows.contains { !$0.isReconstructed })
        XCTAssertTrue(rows.contains(where: \.isReconstructed))
        XCTAssertEqual(rows.filter { $0.key == .challenge(id: R.sharedID) }.map(\.source), [.live])
        try await renderWindow(controller, name: "challenge-log-mixed")
    }

    func testTableRendersARowInEveryStateAndTheFilterBarRenders() async throws {
        let (controller, _) = try makeController()
        let rows = LichessBotChallengeLogRow.rows(ledger: R.ledger(), reconstruction: try R.reconstruction(), pendingChallengeIDs: [R.pendingID])
        let kinds = Set(rows.map { LichessBotChallengeLogStateKind($0.state) })
        XCTAssertEqual(kinds, Set(LichessBotChallengeLogStateKind.allCases))
        for scheme in [ColorScheme.light, .dark] {
            try await renderHosted(LichessBotChallengeLogTableRenderHost(controller: controller, rows: rows),
                                   size: Self.windowSize, name: "challenge-log-table", scheme: scheme)
            try await renderHosted(LichessBotChallengeLogFilterBarRenderHost(), size: CGSize(width: 1500, height: 40),
                                   name: "challenge-log-filter-bar", scheme: scheme)
            try await renderHosted(LichessBotChallengeOutcomesCard(controller: controller), size: CGSize(width: 420, height: 160),
                                   name: "challenge-outcomes-card", scheme: scheme)
        }
    }
}

/// The table with fixed rows and its own sort state.
private struct LichessBotChallengeLogTableRenderHost: View {
    let controller: LichessBotController
    let rows: [LichessBotChallengeLogRow]
    @State private var sortOrder = [KeyPathComparator(\LichessBotChallengeLogRow.at, order: .reverse)]

    var body: some View {
        LichessBotChallengeLogTable(controller: controller, rows: rows, sortOrder: $sortOrder)
    }
}

/// The filter bar with its own filter state.
private struct LichessBotChallengeLogFilterBarRenderHost: View {
    @State private var filter = LichessBotChallengeLogFilter()

    var body: some View {
        LichessBotChallengeLogFilterBar(filter: $filter)
    }
}
