//
//  LichessBotGameOriginViewRenderTests.swift
//  DrewsChessMachineTests
//
//  Render smoke tests for how a game's origin is shown (challenge-log plan
//  §3.9): the glyph, the label and the detail's Origin row for every
//  category and every basis (and a live game's "not yet known"), the Recent
//  games list's glyph column, a tile and a game detail with and without an
//  origin, and the All Games window. Each is drawn off-screen in light and
//  dark and must produce a non-empty image; the images are written to a
//  temporary folder (printed) for inspection. Every controller uses a
//  temporary data folder and defaults suite.
//

import AppKit
import SwiftUI
import XCTest
@testable import DrewsChessMachine

@MainActor
final class LichessBotGameOriginViewRenderTests: XCTestCase {

    private static let renderFolder = FileManager.default.temporaryDirectory.appendingPathComponent("LichessBotRenders", isDirectory: true)

    private func render<Content: View>(_ view: Content, size: CGSize, name: String, scheme: ColorScheme) throws {
        try FileManager.default.createDirectory(at: Self.renderFolder, withIntermediateDirectories: true)
        let renderer = ImageRenderer(content: view
            .frame(width: size.width, height: size.height)
            .background(Color(nsColor: .windowBackgroundColor))
            .environment(\.colorScheme, scheme))
        renderer.scale = 2
        let image = try XCTUnwrap(renderer.nsImage, "\(name) did not render")
        let tiff = try XCTUnwrap(image.tiffRepresentation)
        let bitmap = try XCTUnwrap(NSBitmapImageRep(data: tiff))
        let png = try XCTUnwrap(bitmap.representation(using: .png, properties: [:]))
        XCTAssertGreaterThan(bitmap.pixelsWide, 0)
        let url = Self.renderFolder.appendingPathComponent("\(name)-\(scheme == .dark ? "dark" : "light").png")
        try png.write(to: url)
        print("LICHESS-BOT-RENDER \(url.path)")
    }

    /// AppKit-backed content (a `Table`) through a hosting view, which
    /// `ImageRenderer` can't draw. The window is never ordered on screen.
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
        // The views' `onChange` refreshes run as main-actor tasks.
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

    private func makeController() throws -> LichessBotController {
        let defaults = try makeTemporaryDefaults()
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("LichessBotGameOriginViewRenderTests-\(UUID().uuidString)", isDirectory: true)
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

    /// Every category with every basis it can have, plus a live game's
    /// "not yet known" (nil).
    private static func everyDisplay() -> [LichessBotGameOriginDisplay?] {
        let bases: [LichessBotOriginBasis] = [
            .recorded, .challengeLog, .reconstructed(.certain), .reconstructed(.paired), .reconstructed(.inferredFromAbsence),
        ]
        var displays: [LichessBotGameOriginDisplay?] = []
        for category in LichessBotGameOriginCategory.allCases where category != .unknown {
            for basis in bases {
                displays.append(LichessBotGameOriginDisplay(category: category, detail: "detail of \(category.rawValue)", basis: basis))
            }
        }
        let reasons: [LichessBotOriginUnknownReason] = [.playedBeforeOriginsWereRecorded, .notRecorded, .gap(.noChallengeRecord), .gap(.challengeLogIncomplete)]
        for reason in reasons {
            displays.append(LichessBotGameOriginDisplay(category: .unknown, detail: "game abc", basis: .unknown(reason)))
        }
        displays.append(nil)
        return displays
    }

    func testOriginBuildingBlocksRenderForEveryCategoryAndBasis() throws {
        let displays = Self.everyDisplay()
        for scheme in [ColorScheme.light, .dark] {
            try render(LichessBotGameOriginRenderSheet(displays: displays), size: CGSize(width: 900, height: CGFloat(displays.count) * 22 + 20),
                       name: "origin-building-blocks", scheme: scheme)
        }
    }

    // MARK: Games

    private static let gameFullJSON = #"{"type":"gameFull","id":"abcd1234","variant":{"key":"standard"},"clock":{"initial":300000,"increment":3000},"speed":"blitz","rated":false,"createdAt":1700000000000,"white":{"id":"drewschessmachine","name":"DrewsChessMachine","title":"BOT","rating":1500},"black":{"id":"alice","name":"Alice","rating":1612},"initialFen":"startpos","state":{"type":"gameState","moves":"e2e4 e7e5","wtime":300000,"btime":300000,"winc":3000,"binc":3000,"status":"started"}}"#

    private func makeGame(origin: LichessBotGameOrigin?) throws -> LichessBotLiveGame {
        let game = LichessBotLiveGame(id: "abcd1234", startedAt: Date(), ourAccountID: "drewschessmachine")
        let full = Data(Self.gameFullJSON.utf8)
        guard case .gameFull(let decoded) = try LichessBotGameStreamLine.decode(full) else {
            throw CocoaError(.coderReadCorrupt)
        }
        game.apply(.streamOpened(attempt: 0))
        game.apply(.streamLine(full, receivedAt: Date()))
        game.apply(.gameInfo(decoded, ourColor: .white))
        if let origin {
            game.setOrigin(origin)
        }
        return game
    }

    func testTileAndDetailRenderWithAndWithoutAnOrigin() throws {
        let controller = try makeController()
        let cases: [(String, LichessBotGameOrigin?)] = [
            ("not-yet-known", nil),
            ("matchmaking", .outgoingChallengeAccepted(challengeID: "abcd1234", sender: .matchmaking(trigger: .automaticPass, fillMode: .everyFreeSlot))),
            ("incoming", .acceptedIncomingChallenge(challengeID: "abcd1234", challengerID: "alice")),
            ("undetermined", .undetermined(source: LichessBotOpenValue(.friend), gap: .noChallengeRecord)),
        ]
        for (name, origin) in cases {
            let game = try makeGame(origin: origin)
            XCTAssertEqual(game.originDisplay, origin.map { LichessBotGameOriginDisplay.display($0, basis: .recorded) })
            for scheme in [ColorScheme.light, .dark] {
                try render(LichessBotGameTileView(controller: controller, game: game, onOpen: {}, onDismiss: {}),
                           size: CGSize(width: 300, height: 420), name: "origin-tile-\(name)", scheme: scheme)
                try render(LichessBotGameDetailView(controller: controller, game: game, headToHead: nil, onPopOut: {}, claimsKeyboardShortcuts: false),
                           size: CGSize(width: 1000, height: 720), name: "origin-detail-\(name)", scheme: scheme)
            }
        }
    }

    func testRecentGamesListRendersItsOriginColumn() throws {
        let controller = try makeController()
        let summaries = try ["g1", "g2", "g3"].map(Self.summary)
        for scheme in [ColorScheme.light, .dark] {
            try render(LichessBotRecentGamesList(controller: controller, rows: summaries), size: CGSize(width: 640, height: 200),
                       name: "origin-recent-games", scheme: scheme)
        }
    }

    func testAllGamesWindowRenders() async throws {
        let controller = try makeController()
        await controller.refreshIndex()
        for scheme in [ColorScheme.light, .dark] {
            try await renderHosted(LichessBotAllGamesView(controller: controller), size: CGSize(width: 1080, height: 400),
                                   name: "origin-all-games", scheme: scheme)
        }
    }

    private static let gameFullTemplate = #"{"type":"gameFull","id":"GAMEID","variant":{"key":"standard"},"clock":{"initial":180000,"increment":2000},"speed":"blitz","perf":{"name":"Blitz"},"rated":false,"createdAt":1759000000000,"white":{"id":"drewschessmachine","name":"drewschessmachine","title":"BOT","rating":1500},"black":{"id":"alice","name":"Alice","rating":1600},"initialFen":"startpos","state":{"type":"gameState","moves":"","wtime":180000,"btime":180000,"winc":2000,"binc":2000,"status":"started"}}"#

    private static func summary(_ gameID: String) throws -> LichessBotGameSummary {
        var at = Date(timeIntervalSince1970: 1_759_000_000)
        func next() -> Date {
            at = at.addingTimeInterval(1)
            return at
        }
        let entries: [LichessBotJournalEntry] = [
            .init(at: next(), event: .header(schemaVersion: LichessBotJournal.schemaVersion, gameID: gameID, build: 1, gitHash: "test", resumed: false)),
            .init(at: next(), event: .streamOpened(attempt: 0)),
            .init(at: next(), event: .streamLine(raw: gameFullTemplate.replacingOccurrences(of: "GAMEID", with: gameID))),
            .init(at: next(), event: .finished(status: "resign", winner: "white", localDrawCondition: nil)),
        ]
        let record = try LichessBotRecordBuilder.build(
            gameID: gameID, journal: .init(elements: entries, droppedTrailingByteCount: 0),
            export: nil, exportUnavailableReason: "test", ourAccountID: "drewschessmachine", checkedAt: Date()
        )
        return LichessBotGameSummary(record: record)
    }
}

/// Every origin building block, one display per row: glyph, label, and the
/// detail's Origin row.
private struct LichessBotGameOriginRenderSheet: View {
    let displays: [LichessBotGameOriginDisplay?]

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            ForEach(displays.indices, id: \.self) { index in
                HStack(spacing: 16) {
                    LichessBotGameOriginGlyph(display: displays[index])
                        .frame(width: 20)
                    LichessBotGameOriginLabel(display: displays[index])
                        .frame(width: 240, alignment: .leading)
                    LichessBotGameOriginDetailLine(display: displays[index])
                }
            }
        }
        .padding(10)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
    }
}
