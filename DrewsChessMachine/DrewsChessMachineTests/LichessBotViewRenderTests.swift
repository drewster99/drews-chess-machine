import SwiftUI
import XCTest
@testable import DrewsChessMachine

/// Render smoke tests for the Lichess bot's game views: each view is laid
/// out and drawn off-screen from a synthetic game, in light and dark, and
/// must produce a non-empty image. The images are written to a temporary
/// folder (printed) for visual inspection.
@MainActor
final class LichessBotViewRenderTests: XCTestCase {

    private func makeGame() throws -> LichessBotLiveGame {
        let game = LichessBotLiveGame(id: "abcd1234", startedAt: Date(), ourAccountID: "drewschessmachine")
        let fullJSON = #"{"type":"gameFull","id":"abcd1234","variant":{"key":"standard"},"clock":{"initial":300000,"increment":3000},"speed":"blitz","rated":false,"createdAt":1700000000000,"white":{"id":"drewschessmachine","name":"DrewsChessMachine","title":"BOT","rating":1500},"black":{"id":"alice","name":"Alice","rating":1612},"initialFen":"startpos","state":{"type":"gameState","moves":"","wtime":300000,"btime":300000,"winc":3000,"binc":3000,"status":"started"}}"#
        let full = Data(fullJSON.utf8)
        guard case .gameFull(let decoded) = try LichessBotGameStreamLine.decode(full) else {
            throw CocoaError(.coderReadCorrupt)
        }
        game.apply(.streamOpened(attempt: 0))
        game.apply(.streamLine(full, receivedAt: Date()))
        game.apply(.gameInfo(decoded, ourColor: .white))
        let tokens = ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6", "e1h1"]
        for count in 1...tokens.count {
            if count % 2 == 1 {
                let decision = LichessBotMoveDecision(
                    uci: tokens[count - 1], san: "", chosenProbability: 0.41,
                    topMoves: [LichessBotMoveCandidate(uci: tokens[count - 1], probability: 0.41), LichessBotMoveCandidate(uci: "d2d4", probability: 0.22)],
                    win: 0.38, draw: 0.35, loss: 0.27, temperature: 0.5, legalMoveCount: 30, randomish: false,
                    encodeMilliseconds: 0.2, inferenceMilliseconds: 3.1, sampleMilliseconds: 0.1
                )
                game.apply(.moveDecided(ply: count - 1, decision: decision, generation: LichessBotGenerationInfo(
                    generationID: 1, sourceKind: .champion, modelID: "20260928-1-TEST", trainingStep: nil,
                    snapshotAt: Date(), architectureSummary: "test", filePath: nil, fileSHA256: nil
                )))
                game.applyRequest(LichessBotRequestRecord(
                    startedAt: Date(), gameID: "abcd1234", label: "move", method: "POST",
                    path: "/api/bot/game/abcd1234/move/\(tokens[count - 1])", formFields: [:], status: 200,
                    queuedMilliseconds: 0.4, roundTripMilliseconds: 48, networkProtocol: "h2", errorMessage: nil, failure: nil
                ))
            }
            let state = #"{"type":"gameState","moves":"\#(tokens.prefix(count).joined(separator: " "))","wtime":\#(300000 - count * 4000),"btime":\#(300000 - count * 5000),"winc":3000,"binc":3000,"status":"started"}"#
            game.apply(.streamLine(Data(state.utf8), receivedAt: Date()))
            game.apply(.keepAlive(receivedAt: Date()))
        }
        return game
    }

    private func render<Content: View>(_ view: Content, size: CGSize, name: String, scheme: ColorScheme) throws -> URL {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent("LichessBotRenders", isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
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
        let url = folder.appendingPathComponent("\(name)-\(scheme == .dark ? "dark" : "light").png")
        try png.write(to: url)
        print("LICHESS-BOT-RENDER \(url.path)")
        return url
    }

    func testGameViewsRender() throws {
        let game = try makeGame()
        for scheme in [ColorScheme.light, .dark] {
            _ = try render(
                LichessBotGameDetailView(game: game, headToHead: (2, 1, 3), onPopOut: {}, claimsKeyboardShortcuts: false),
                size: CGSize(width: 1000, height: 720), name: "detail", scheme: scheme
            )
            _ = try render(
                LichessBotGameTileView(game: game, isFocused: true, onFocus: {}, onDismiss: {}),
                size: CGSize(width: 300, height: 420), name: "tile", scheme: scheme
            )
        }
    }
}
