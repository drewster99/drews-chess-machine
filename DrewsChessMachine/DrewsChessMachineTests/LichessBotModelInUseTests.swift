import SwiftUI
import XCTest
@testable import DrewsChessMachine

/// What the model-in-use views say (issue #20,
/// LICHESS_BOT_MODEL_IN_USE_PLAN.md): the name and identity of the model
/// playing, file times and ages, the switch, older generations games still
/// play, the newest file's "(in use)" mark, and renders of the row and line.
@MainActor
final class LichessBotModelInUseTests: XCTestCase {

    private let now = Date(timeIntervalSince1970: 1_791_800_000)
    private let followed = LichessBotFollowedLineage(lineageRunID: "E709F4EB-1D95-4166-90FA-F0D31002BB89",
                                                     anchorSegmentID: "0E4B3E7C-6A0B-4F7E-9A57-2E9D1F3C5B21")

    private func lineageFile(sha: String, name: String = "20261011-zlra-elitecyc-latest.safetensors") -> LichessBotLineageFile {
        LichessBotLineageFile(ModelLineageTip.Candidate(
            entry: ModelFileEntry(
                url: URL(fileURLWithPath: "/models/\(name)"), modelID: "20261012-2-dMNh", trainingStep: 28000,
                createdAt: nil, architectureLabel: "a", fileModifiedAt: now.addingTimeInterval(-600)),
            position: ModelFileLineagePosition(
                lineageRunID: followed.lineageRunID, segmentID: followed.anchorSegmentID, segmentIndex: 0, segmentStartedUnix: 0,
                segmentChain: [followed.anchorSegmentID], handoffs: [], segmentLocalStep: 28000, cumTrainerStep: 28000,
                recordedUnix: 0, pathKind: .replay),
            contentSHA256: sha))
    }

    /// A generation of the followed lineage built from content `sha`.
    private func followedGeneration(id: Int, sha: String, snapshotAt: Date, modified: Date?,
                                    file: String = "20261011-zlra-elitecyc-latest.safetensors") -> LichessBotGenerationInfo {
        var info = LichessBotGenerationInfo(
            generationID: id, sourceKind: .followLineage, modelID: "20261012-2-dMNh", trainingStep: 28000,
            snapshotAt: snapshotAt, architectureSummary: "test", filePath: "/models/\(file)", fileSHA256: sha)
        let position = ModelFileLineagePosition(
            lineageRunID: followed.lineageRunID, segmentID: followed.anchorSegmentID, segmentIndex: 0, segmentStartedUnix: 0,
            segmentChain: [followed.anchorSegmentID], handoffs: [], segmentLocalStep: 28000, cumTrainerStep: 28000,
            recordedUnix: 0, pathKind: .replay)
        info.lineage = LichessBotGenerationLineage(position: position, contentSHA256: sha, followed: followed)
        info.fileModifiedAt = modified
        return info
    }

    private func champion(id: Int, snapshotAt: Date) -> LichessBotGenerationInfo {
        LichessBotGenerationInfo(generationID: id, sourceKind: .champion, modelID: "20261001-1-Ejp0", trainingStep: nil,
                                 snapshotAt: snapshotAt, architectureSummary: "test", filePath: nil, fileSHA256: nil)
    }

    // MARK: - Times

    func testAgeText() {
        XCTAssertEqual(LichessBotModelTimeText.ageText(from: now.addingTimeInterval(-30), to: now), "<1 min ago")
        XCTAssertEqual(LichessBotModelTimeText.ageText(from: now.addingTimeInterval(-23 * 60), to: now), "23 min ago")
        XCTAssertEqual(LichessBotModelTimeText.ageText(from: now.addingTimeInterval(-(5 * 3600 + 12 * 60)), to: now), "5 h 12 min ago")
        XCTAssertEqual(LichessBotModelTimeText.ageText(from: now.addingTimeInterval(-(76 * 3600)), to: now), "3 d 4 h ago")
        XCTAssertEqual(LichessBotModelTimeText.ageText(from: now, to: now), "<1 min ago")
        XCTAssertNil(LichessBotModelTimeText.ageText(from: now.addingTimeInterval(60), to: now), "a future time has no age")
    }

    func testWithAgeIsTheAbsoluteFormPlusTheAge() {
        let date = now.addingTimeInterval(-23 * 60)
        XCTAssertEqual(LichessBotModelTimeText.withAge(date, now: now), "\(LichessBotModelTimeText.absolute(date)) · 23 min ago")
        let future = now.addingTimeInterval(600)
        XCTAssertEqual(LichessBotModelTimeText.withAge(future, now: now), LichessBotModelTimeText.absolute(future))
        XCTAssertEqual(LichessBotModelTimeText.absolute(date).count, "yyyy-MM-dd HH:mm".count, "fixed width")
    }

    // MARK: - Names and the switch

    func testNameIsTheFileOrTheSource() {
        let file = followedGeneration(id: 1, sha: "aa", snapshotAt: now, modified: now)
        XCTAssertEqual(LichessBotModelInUseText.name(file), "20261011-zlra-elitecyc-latest.safetensors")
        XCTAssertEqual(LichessBotModelInUseText.name(champion(id: 1, snapshotAt: now)), "Champion")
    }

    func testIdentityPrefersTheCumulativeStep() {
        XCTAssertEqual(LichessBotModelInUseText.identity(followedGeneration(id: 1, sha: "aa", snapshotAt: now, modified: now)),
                       "20261012-2-dMNh · cum 28000")
        let stepOnly = LichessBotStatsFixtures.generationInfo(id: 1, modelID: "20261012-1-AbCd", sha: "bb", step: 500)
        XCTAssertEqual(LichessBotModelInUseText.identity(stepOnly), "20261012-1-AbCd · step 500")
        XCTAssertEqual(LichessBotModelInUseText.identity(champion(id: 1, snapshotAt: now)), "20261001-1-Ejp0")
    }

    func testFileModifiedOnlyForAFile() {
        let modified = now.addingTimeInterval(-600)
        XCTAssertEqual(LichessBotModelInUseText.fileModified(followedGeneration(id: 1, sha: "aa", snapshotAt: now, modified: modified), now: now),
                       "file modified \(LichessBotModelTimeText.withAge(modified, now: now))")
        XCTAssertNil(LichessBotModelInUseText.fileModified(champion(id: 1, snapshotAt: now), now: now))
    }

    func testTheFirstGenerationReadsSinceGoingOnline() {
        let at = now.addingTimeInterval(-3600)
        let first = LichessBotModelSwitch(to: champion(id: 1, snapshotAt: at), from: nil, reason: LichessBotModelSlots.goingOnlineReason)
        XCTAssertEqual(LichessBotModelInUseText.switchText(first, now: now), "since going online \(LichessBotModelTimeText.withAge(at, now: now))")
        XCTAssertEqual(LichessBotModelInUseText.onlineLine(first, now: now),
                       "Playing Champion (20261001-1-Ejp0) · since going online \(LichessBotModelTimeText.withAge(at, now: now))")
    }

    func testALaterSwitchNamesTheGenerationItReplaced() {
        let old = followedGeneration(id: 1, sha: "aa", snapshotAt: now.addingTimeInterval(-7200), modified: now.addingTimeInterval(-7300),
                                     file: "20261011-zlra-elitecyc-latest-step27000.safetensors")
        let modified = now.addingTimeInterval(-660)
        let new = followedGeneration(id: 2, sha: "bb", snapshotAt: now.addingTimeInterval(-600), modified: modified)
        let record = LichessBotModelSwitch(to: new, from: old, reason: "newer file of the followed lineage")
        let switched = "switched \(LichessBotModelTimeText.withAge(new.snapshotAt, now: now)) from 20261011-zlra-elitecyc-latest-step27000.safetensors (20261012-2-dMNh)"
        XCTAssertEqual(LichessBotModelInUseText.switchText(record, now: now), switched)
        XCTAssertEqual(LichessBotModelInUseText.onlineLine(record, now: now),
                       "Playing 20261011-zlra-elitecyc-latest.safetensors (20261012-2-dMNh · cum 28000) · file modified \(LichessBotModelTimeText.withAge(modified, now: now)) · \(switched)")
    }

    // MARK: - Older generations games still play

    private func game(_ id: String, moves: [LichessBotGenerationInfo]) -> LichessBotLiveGame {
        let game = LichessBotLiveGame(id: id, startedAt: now, ourAccountID: "drewschessmachine")
        for (ply, generation) in moves.enumerated() {
            game.apply(.moveDecided(ply: ply * 2, decision: LichessBotStatsFixtures.decision("e2e4", win: 0.3, draw: 0.4, loss: 0.3),
                                    generation: generation))
        }
        return game
    }

    func testOlderGenerationLinesCountGamesByTheirLatestMove() {
        let first = champion(id: 1, snapshotAt: now.addingTimeInterval(-7200))
        let second = champion(id: 2, snapshotAt: now.addingTimeInterval(-3600))
        let current = champion(id: 3, snapshotAt: now.addingTimeInterval(-60))
        let games = [
            game("a", moves: [first, second]),          // latest move on 2
            game("b", moves: [second]),                 // on 2
            game("c", moves: [first]),                  // on 1
            game("d", moves: [first, current]),         // moved to the current one
            game("e", moves: []),                       // no move of ours yet
        ]
        XCTAssertEqual(LichessBotModelInUseText.olderGenerationLines(games: games, current: current), [
            "1 game in progress still on generation 1 (20261001-1-Ejp0)",
            "2 games in progress still on generation 2 (20261001-1-Ejp0)",
        ])
        XCTAssertEqual(LichessBotModelInUseText.olderGenerationLines(games: [game("d", moves: [current])], current: current), [])
    }

    func testAResumedGameWithAReusedGenerationIDIsStillOlder() {
        // Generation IDs restart each time the bot goes online: a game
        // resumed after a relaunch can carry the ID the current one has.
        let before = champion(id: 1, snapshotAt: now.addingTimeInterval(-86_400))
        let current = followedGeneration(id: 1, sha: "aa", snapshotAt: now, modified: now)
        XCTAssertEqual(LichessBotModelInUseText.olderGenerationLines(games: [game("r", moves: [before])], current: current),
                       ["1 game in progress still on generation 1 (20261001-1-Ejp0)"])
    }

    // MARK: - The newest file's mark

    func testNewestFileMark() {
        let newest = lineageFile(sha: "bb")
        let playingNewest = LichessBotModelSwitch(to: followedGeneration(id: 2, sha: "bb", snapshotAt: now, modified: now), from: nil, reason: "r")
        let playingOlder = LichessBotModelSwitch(to: followedGeneration(id: 1, sha: "aa", snapshotAt: now, modified: now), from: nil, reason: "r")
        XCTAssertEqual(LichessBotModelInUseText.newestFileMark(newest: newest, followed: followed, modelInUse: playingNewest, appliedLineage: followed), "(in use)")
        XCTAssertEqual(LichessBotModelInUseText.newestFileMark(newest: newest, followed: followed, modelInUse: playingOlder, appliedLineage: followed),
                       "(next game once loaded)")
        XCTAssertNil(LichessBotModelInUseText.newestFileMark(newest: newest, followed: followed, modelInUse: nil, appliedLineage: followed), "offline")
        let other = LichessBotFollowedLineage(lineageRunID: "other", anchorSegmentID: "other")
        XCTAssertNil(LichessBotModelInUseText.newestFileMark(newest: newest, followed: followed, modelInUse: playingNewest, appliedLineage: other),
                     "a lineage chosen in the draft but not applied")
        XCTAssertNil(LichessBotModelInUseText.newestFileMark(newest: newest, followed: followed, modelInUse: playingNewest, appliedLineage: nil))
    }

    func testTheAppliedLineageCountsOnlyUnderTheFollowLineageSource() {
        var model = LichessBotModelSettings.testBaseline()
        model.followedLineage = followed
        model.source = .followLineage
        XCTAssertEqual(LichessBotModelInUseText.appliedFollowedLineage(of: model), followed)
        model.source = .file
        XCTAssertNil(LichessBotModelInUseText.appliedFollowedLineage(of: model), "a lineage left in the settings under another source plays nothing")
    }

    func testPlaysIsTheSlotsRule() {
        let file = lineageFile(sha: "bb")
        XCTAssertTrue(followedGeneration(id: 1, sha: "bb", snapshotAt: now, modified: now).plays(file, of: followed))
        XCTAssertFalse(followedGeneration(id: 1, sha: "aa", snapshotAt: now, modified: now).plays(file, of: followed))
        XCTAssertFalse(followedGeneration(id: 1, sha: "bb", snapshotAt: now, modified: now)
            .plays(file, of: LichessBotFollowedLineage(lineageRunID: "x", anchorSegmentID: "y")))
        XCTAssertFalse(champion(id: 1, snapshotAt: now).plays(file, of: followed), "no lineage")
    }

    // MARK: - Codable

    func testGenerationInfoWithoutFileModifiedAtDecodesAndADateRoundTrips() throws {
        let info = followedGeneration(id: 4, sha: "aa", snapshotAt: now, modified: now.addingTimeInterval(-90))
        let data = try JSONEncoder().encode(info)
        XCTAssertEqual(try JSONDecoder().decode(LichessBotGenerationInfo.self, from: data), info)
        var object = try XCTUnwrap(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertNotNil(object.removeValue(forKey: "fileModifiedAt"), "the key this test removes")
        let older = try JSONDecoder().decode(LichessBotGenerationInfo.self, from: try JSONSerialization.data(withJSONObject: object))
        XCTAssertNil(older.fileModifiedAt)
        XCTAssertEqual(older.modelID, info.modelID)
    }

    // MARK: - Renders

    private func render<Content: View>(_ view: Content, size: CGSize, name: String) throws {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent("LichessBotRenders", isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        for scheme in [ColorScheme.light, .dark] {
            let renderer = ImageRenderer(content: view
                .frame(width: size.width, height: size.height)
                .background(Color(nsColor: .windowBackgroundColor))
                .environment(\.colorScheme, scheme))
            renderer.scale = 2
            let image = try XCTUnwrap(renderer.nsImage, "\(name) did not render")
            let bitmap = try XCTUnwrap(NSBitmapImageRep(data: try XCTUnwrap(image.tiffRepresentation)))
            XCTAssertGreaterThan(bitmap.pixelsWide, 0)
            let url = folder.appendingPathComponent("\(name)-\(scheme == .dark ? "dark" : "light").png")
            try XCTUnwrap(bitmap.representation(using: .png, properties: [:])).write(to: url)
            print("LICHESS-BOT-RENDER \(url.path)")
        }
    }

    func testTheInUseRowRendersOnlineOfflineAndWithoutAFile() throws {
        let longName = "20261011-zlra-elitecyc-latest-a-very-long-rolling-file-name-for-overflow.safetensors"
        let cases: [(String, LichessBotModelSwitch?)] = [
            ("offline", nil),
            ("champion", LichessBotModelSwitch(to: champion(id: 1, snapshotAt: now.addingTimeInterval(-60)), from: nil, reason: "r")),
            ("long-file", LichessBotModelSwitch(
                to: followedGeneration(id: 2, sha: "bb", snapshotAt: now.addingTimeInterval(-600), modified: now.addingTimeInterval(-700), file: longName),
                from: followedGeneration(id: 1, sha: "aa", snapshotAt: now.addingTimeInterval(-7200), modified: nil), reason: "r")),
        ]
        for (name, modelInUse) in cases {
            let row = Form { LabeledContent("In use") { LichessBotModelInUseDetails(modelInUse: modelInUse, now: now) } }
                .formStyle(.grouped)
            try render(row, size: CGSize(width: 680, height: 160), name: "model-in-use-row-\(name)")
        }
    }

    func testTheOnlineLineRendersAFirstGenerationASwitchAndOlderGames() throws {
        let first = champion(id: 1, snapshotAt: now.addingTimeInterval(-7200))
        let current = followedGeneration(id: 2, sha: "bb", snapshotAt: now.addingTimeInterval(-600), modified: now.addingTimeInterval(-700))
        let cases: [(String, LichessBotModelSwitch, [LichessBotLiveGame])] = [
            ("first", LichessBotModelSwitch(to: first, from: nil, reason: "r"), []),
            ("switch", LichessBotModelSwitch(to: current, from: first, reason: "r"), []),
            ("older-games", LichessBotModelSwitch(to: current, from: first, reason: "r"), [game("a", moves: [first]), game("b", moves: [first])]),
        ]
        for (name, record, games) in cases {
            let line = LichessBotModelInUseLineText(modelInUse: record, gamesInProgress: games, now: now)
            try render(line, size: CGSize(width: 900, height: 70), name: "model-in-use-line-\(name)")
        }
    }

    func testTheNewestLineRendersWithEachMark() throws {
        let newest = lineageFile(sha: "bb")
        let status = LichessBotLineageFollowStatus(
            followed: followed, checkedAt: now, outcome: .following(newest: newest),
            candidateCount: 2, excluded: [:], listed: 40, headersRead: 1, reused: 39, scanMilliseconds: 3)
        for mark in ["(in use)", "(next game once loaded)", nil] as [String?] {
            let summary = LichessBotFollowedLineageSummary(followedLineage: followed, status: status, resolution: .idle, newestFileMark: mark)
            try render(summary, size: CGSize(width: 520, height: 90), name: "newest-line-\(mark ?? "offline")")
        }
    }
}
