import SwiftUI
import XCTest
@testable import DrewsChessMachine

/// Render smoke tests for the Lichess bot's Settings screen: every tab is
/// laid out and drawn off-screen at about the bot window's minimum detail
/// size, in light and dark, and must produce a non-empty image — so a crash
/// or a layout trap in any tab is caught. The Account tab is drawn with no
/// token saved and with a verified BOT token. The images are written to a
/// temporary folder (printed) for visual inspection.
///
/// Also checks that the remembered tab persists in the controller's own
/// defaults and that a remembered tab that no longer exists is ignored.
@MainActor
final class LichessBotSettingsViewRenderTests: XCTestCase {

    /// The bot window's minimum content size less the widest sidebar.
    private let minimumDetailSize = CGSize(width: 680, height: 570)

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

    /// A controller isolated from the app's real settings, bot data,
    /// Keychain and network: the defaults suite `suite`, opened afresh as a
    /// later launch would open it, and a temporary data folder, both removed
    /// after the test. `storedToken` is
    /// what the controller finds "in the Keychain" — nil for none; the fake
    /// Lichess verifies `LichessBotFakeLichess.token` as a BOT account's.
    private func makeController(suite: TemporaryDefaultsSuite, storedToken: String?) throws -> LichessBotController {
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite.suiteName))
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("LichessBotSettingsViewRenderTests-\(UUID().uuidString)", isDirectory: true)
        try LichessBotSettingsStore.save(LichessBotSettings.testBaseline(), to: defaults)
        let controller = LichessBotController(
            modelProvider: LichessBotFakeModelProvider(snapshot: nil),
            defaults: defaults,
            dataDirectory: LichessBotDataDirectory(root: root),
            services: LichessBotControllerServices(
                makeTransport: { LichessBotFakeLichess() },
                readToken: { _ in storedToken }
            )
        )
        addTeardownBlock { @MainActor in
            // Shutting down writes the protocol events already queued and
            // refuses later ones, so nothing races the folder's removal.
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

    /// Every tab, while the token check has not run: the view opens on the
    /// remembered tab.
    func testEveryTabRenders() throws {
        let controller = try makeController(suite: try makeTemporaryDefaultsSuite(), storedToken: nil)
        for tab in LichessBotSettingsTab.allCases {
            controller.rememberedSettingsTab = tab
            XCTAssertEqual(LichessBotSettingsTab.opening(tokenState: controller.tokenState, remembered: controller.rememberedSettingsTab), tab)
            for scheme in [ColorScheme.light, .dark] {
                _ = try render(
                    LichessBotSettingsView(controller: controller),
                    size: minimumDetailSize, name: "settings-\(tab.rawValue)", scheme: scheme
                )
            }
        }
    }

    func testAccountTabRendersWithNoToken() async throws {
        let controller = try makeController(suite: try makeTemporaryDefaultsSuite(), storedToken: nil)
        controller.rememberedSettingsTab = .games
        await controller.refreshTokenState()
        XCTAssertEqual(controller.tokenState, LichessBotController.TokenState.none)
        XCTAssertEqual(LichessBotSettingsTab.opening(tokenState: controller.tokenState, remembered: controller.rememberedSettingsTab), .account)
        for scheme in [ColorScheme.light, .dark] {
            _ = try render(
                LichessBotSettingsView(controller: controller),
                size: minimumDetailSize, name: "settings-account-no-token", scheme: scheme
            )
        }
    }

    func testAccountTabRendersWithAVerifiedBotToken() async throws {
        let controller = try makeController(suite: try makeTemporaryDefaultsSuite(), storedToken: LichessBotFakeLichess.token)
        controller.rememberedSettingsTab = .account
        await controller.refreshTokenState()
        guard case .saved = controller.tokenState else {
            XCTFail("the fake token must verify; token state is \(controller.tokenState)")
            return
        }
        XCTAssertEqual(controller.account?.isBot, true)
        XCTAssertFalse(controller.canUpgradeToBot, "a BOT account hides the upgrade box")
        for scheme in [ColorScheme.light, .dark] {
            _ = try render(
                LichessBotSettingsView(controller: controller),
                size: minimumDetailSize, name: "settings-account-bot-token", scheme: scheme
            )
        }
    }

    func testRememberedTabPersistsInTheControllersDefaults() throws {
        let suite = try makeTemporaryDefaultsSuite()
        let first = try makeController(suite: suite, storedToken: nil)
        XCTAssertEqual(first.rememberedSettingsTab, .games, "with nothing remembered, Settings opens on Games")
        first.rememberedSettingsTab = .chat
        let second = try makeController(suite: suite, storedToken: nil)
        XCTAssertEqual(second.rememberedSettingsTab, .chat)
    }

    func testRememberedTabThatNoLongerExistsIsIgnored() throws {
        let suite = try makeTemporaryDefaultsSuite()
        suite.defaults.set("tournaments", forKey: "lichessBot.settings.tab")
        let controller = try makeController(suite: suite, storedToken: nil)
        XCTAssertEqual(controller.rememberedSettingsTab, .games)
    }
}
