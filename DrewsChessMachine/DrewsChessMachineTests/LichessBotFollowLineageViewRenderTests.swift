import SwiftUI
import XCTest
@testable import DrewsChessMachine

/// The follow-lineage UI (follow-lineage plan §3.9): the model settings with
/// each source, the Overview's follow status in every outcome, and the
/// lineage picker's follow mode, which offers only training segments that
/// record a followable lineage and says why every other row is not offered.
/// Images go to a temporary folder (printed) for visual inspection. The
/// picker itself is not drawn: it reads the real models folder when it
/// appears; its follow-mode rules are checked directly.
@MainActor
final class LichessBotFollowLineageViewRenderTests: XCTestCase {

    private let lineage = LichessBotFollowedLineage(lineageRunID: "783BF744-5FCB-4869-BBE8-7126EDD0C72B", anchorSegmentID: "0E4B3E7C-6A0B-4F7E-9A57-2E9D1F3C5B21")

    private func render<Content: View>(_ view: Content, size: CGSize, name: String, scheme: ColorScheme) throws {
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
    }

    private func temporaryFolder() throws -> URL {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent("LichessBotFollowLineageViewRenderTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: false)
        addTeardownBlock {
            do {
                try FileManager.default.removeItem(at: folder)
            } catch {
                XCTFail("cleanup failed: \(error)")
            }
        }
        return folder
    }

    /// An offline controller over a private defaults suite and data folder.
    private func makeController() throws -> LichessBotController {
        let defaults = try makeTemporaryDefaults()
        let root = try temporaryFolder()
        try LichessBotSettingsStore.save(LichessBotSettings.testBaseline(), to: defaults)
        var services = LichessBotControllerServices(
            makeTransport: { LichessBotFakeLichess() },
            readToken: { _ in nil }
        )
        services.makeModelFolderScanner = { LichessBotNoModelsFolderScanner() }
        let controller = LichessBotController(
            modelProvider: LichessBotFakeModelProvider(snapshot: nil),
            defaults: defaults,
            dataDirectory: LichessBotDataDirectory(root: root),
            services: services
        )
        addTeardownBlock { @MainActor in
            await controller.shutdown(reason: "test teardown")
        }
        return controller
    }

    func testModelSettingsRenderWithEverySource() throws {
        let controller = try makeController()
        let models = try temporaryFolder()
        for source in LichessBotModelSourceKind.allCases {
            var settings = LichessBotModelSettings.testBaseline()
            settings.source = source
            settings.followedLineage = source == .followLineage ? lineage : nil
            let section = Form {
                LichessBotModelSettingsSection(settings: .constant(settings), controller: controller, modelsDirectory: models)
            }
            .formStyle(.grouped)
            for scheme in [ColorScheme.light, .dark] {
                try render(section, size: CGSize(width: 680, height: 420), name: "model-settings-\(source.rawValue)", scheme: scheme)
            }
        }
    }

    func testFollowStatusRendersInEveryOutcome() throws {
        let file = LichessBotLineageFile(ModelLineageTip.Candidate(
            entry: ModelFileEntry(
                url: URL(fileURLWithPath: "/models/20261005-lrBleaky-cyc1-replay-step31000.safetensors"),
                modelID: "20261006-6-Xxvl", trainingStep: 31000, createdAt: nil, architectureLabel: "a",
                fileModifiedAt: Date(timeIntervalSince1970: 0)),
            position: ModelFileLineagePosition(
                lineageRunID: lineage.lineageRunID, segmentID: lineage.anchorSegmentID, segmentIndex: 0, segmentStartedUnix: 0,
                segmentChain: [lineage.anchorSegmentID], handoffs: [], segmentLocalStep: 31000, cumTrainerStep: 31000,
                recordedUnix: 0, pathKind: .replay),
            contentSHA256: "1447ee85e98b0000"))
        let outcomes: [(String, LichessBotLineageFollowStatus.Outcome)] = [
            ("following", .following(newest: file)),
            ("keep-playing", .keepPlaying(newestOnDisk: file)),
            ("no-files", .noFiles),
            ("fork", .fork(continuations: ["segment A (20261006-2-LEFT) at step 1000, left.safetensors", "segment B (20261006-3-RGHT) at step 2000, right.safetensors"])),
            ("conflict", .conflict(files: [file, file])),
            ("folder-unreadable", .folderUnreadable("permission denied")),
            ("newest-failed", .newestFailedToLoad(file: file, reason: "content_sha256 mismatch")),
        ]
        for (name, outcome) in outcomes {
            let status = LichessBotLineageFollowStatus(
                followed: lineage, checkedAt: Date(timeIntervalSince1970: 1_790_000_000), outcome: outcome,
                candidateCount: 2, excluded: [.otherPathKind: 1], listed: 4462, headersRead: 1, reused: 4461, scanMilliseconds: 13)
            let view = LichessBotLineageFollowStatusView(followed: lineage, status: status, playingGenerationID: 3, isRunning: true, checkNow: {})
            for scheme in [ColorScheme.light, .dark] {
                try render(view, size: CGSize(width: 420, height: 90), name: "follow-status-\(name)", scheme: scheme)
            }
        }
        try render(LichessBotLineageFollowStatusView(followed: lineage, status: nil, playingGenerationID: nil, isRunning: false, checkNow: {}), size: CGSize(width: 420, height: 90), name: "follow-status-offline", scheme: .light)
    }

    func testLinePickerFollowModeOffersOnlyFollowableSegments() throws {
        func entry(_ modelID: String, lineage facts: ModelFileLineageFacts?) -> ModelFileEntry {
            ModelFileEntry(
                url: URL(fileURLWithPath: "/models/\(modelID).safetensors"), modelID: modelID, trainingStep: 1000,
                createdAt: nil, architectureLabel: "a", fileModifiedAt: Date(timeIntervalSince1970: 0),
                contentSHA256: "sha", lineage: facts)
        }
        func position(_ pathKind: LineageRecord.PathKind) -> ModelFileLineagePosition {
            ModelFileLineagePosition(
                lineageRunID: lineage.lineageRunID, segmentID: lineage.anchorSegmentID, segmentIndex: 0, segmentStartedUnix: 0,
                segmentChain: [lineage.anchorSegmentID], handoffs: [], segmentLocalStep: 1000, cumTrainerStep: 1000,
                recordedUnix: 0, pathKind: pathKind)
        }
        func segment(_ latest: ModelFileEntry) -> ModelLineageNode {
            ModelLineageNode(id: latest.modelID, kind: .segment(line: ModelLine(modelID: latest.modelID, files: [latest]), path: [latest.modelID], isUntrained: false, isBranchTip: true), children: nil)
        }
        let follow = LichessBotModelLinePickerPurpose.chooseLineageToFollow
        let replay = segment(entry("20261006-6-Xxvl", lineage: .recorded(position(.replay))))
        XCTAssertNotNil(follow.selectableEntry(for: replay))
        XCTAssertNil(follow.unselectableReason(for: replay))
        XCTAssertEqual(follow.selectableEntry(for: replay).flatMap(LichessBotModelLinePickerPurpose.followedLineage(of:)), lineage)
        XCTAssertNotNil(follow.selectableEntry(for: segment(entry("20261006-7-VSUC", lineage: .recorded(position(.vsuci))))))

        let unfollowable: [(ModelLineageNode, String)] = [
            (segment(entry("20261006-1-GUIS", lineage: .recorded(position(.gui)))), "A GUI run"),
            (segment(entry("20261006-2-DERV", lineage: .recorded(position(.derive)))), "Written by derive"),
            (segment(entry("20260701-1-OLDF", lineage: .unrecorded(formatVersion: 6))), "Written before lineage records"),
            (segment(entry("20261006-3-BADL", lineage: .unreadable(reason: "malformed"))), "Unreadable lineage: malformed"),
            (ModelLineageNode(id: "file", kind: .file(entry("20261006-6-Xxvl", lineage: .recorded(position(.replay)))), children: nil), "Choose the segment row"),
        ]
        for (node, reason) in unfollowable {
            XCTAssertNil(follow.selectableEntry(for: node), node.id)
            XCTAssertTrue(follow.unselectableReason(for: node)?.hasPrefix(reason) == true, "\(node.id): \(String(describing: follow.unselectableReason(for: node)))")
        }

        // Choosing a file still offers every row that names a file.
        let chooseFile = LichessBotModelLinePickerPurpose.chooseFile
        for (node, _) in unfollowable {
            XCTAssertNotNil(chooseFile.selectableEntry(for: node), node.id)
            XCTAssertNil(chooseFile.unselectableReason(for: node), node.id)
        }
    }
}
