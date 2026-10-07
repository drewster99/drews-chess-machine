//
//  LichessBotDataDirectoryChallengePathsTests.swift
//  DrewsChessMachineTests
//
//  The challenge log's place in the bot's data folder (challenge-log plan
//  §3.1): `Challenges/challenges-YYYYMMDD.jsonl` named by UTC day exactly as
//  the protocol log names its day files, the derived
//  `reconstructed-from-protocol.json` beside them, and `Challenges/` among
//  the folders `createDirectories()` makes.
//

import XCTest
@testable import DrewsChessMachine

final class LichessBotDataDirectoryChallengePathsTests: XCTestCase {

    private var tempRoot: URL!
    private var savedDefaultTimeZone: TimeZone!

    override func setUpWithError() throws {
        tempRoot = FileManager.default.temporaryDirectory
            .appendingPathComponent("LichessBotDataDirectoryChallengePathsTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: tempRoot, withIntermediateDirectories: true)
        savedDefaultTimeZone = NSTimeZone.default
    }

    override func tearDownWithError() throws {
        NSTimeZone.default = savedDefaultTimeZone
        try FileManager.default.removeItem(at: tempRoot)
    }

    private var directory: LichessBotDataDirectory { LichessBotDataDirectory(root: tempRoot) }

    func testChallengePathsSitInTheirOwnFolder() {
        let challenges = tempRoot.appendingPathComponent("Challenges", isDirectory: true)
        XCTAssertEqual(directory.challengesDirectory, challenges)
        XCTAssertEqual(directory.reconstructedChallengesURL,
                       challenges.appendingPathComponent("reconstructed-from-protocol.json", isDirectory: false))
    }

    /// 2026-10-05 23:30 and 2026-10-06 00:30 UTC fall on two UTC days but on
    /// one local day in Los Angeles; the files follow the UTC day, whatever
    /// the Mac's zone.
    func testDayFilesAreNamedByUTCDayInAnyTimeZone() throws {
        let lateOnTheFifth = Date(timeIntervalSince1970: 1_791_243_000)
        let earlyOnTheSixth = Date(timeIntervalSince1970: 1_791_246_600)
        for zoneIdentifier in ["America/Los_Angeles", "Asia/Tokyo", "UTC"] {
            NSTimeZone.default = try XCTUnwrap(TimeZone(identifier: zoneIdentifier))
            XCTAssertEqual(directory.challengeLogURL(for: lateOnTheFifth).lastPathComponent, "challenges-20261005.jsonl", zoneIdentifier)
            XCTAssertEqual(directory.challengeLogURL(for: earlyOnTheSixth).lastPathComponent, "challenges-20261006.jsonl", zoneIdentifier)
            XCTAssertEqual(directory.protocolLogURL(for: lateOnTheFifth).lastPathComponent, "events-20261005.jsonl", zoneIdentifier)
            XCTAssertEqual(directory.protocolLogURL(for: earlyOnTheSixth).lastPathComponent, "events-20261006.jsonl", zoneIdentifier)
        }
        XCTAssertEqual(directory.challengeLogURL(for: lateOnTheFifth).deletingLastPathComponent(), directory.challengesDirectory)
        XCTAssertEqual(directory.protocolLogURL(for: lateOnTheFifth).deletingLastPathComponent(), directory.protocolDirectory)
    }

    func testTheProtocolLogNamesItsFilesThroughTheDataDirectory() {
        let log = LichessBotProtocolLog(directory: directory, fileQueue: LichessBotFileQueue()) { _ in }
        let date = Date(timeIntervalSince1970: 1_791_243_000)
        XCTAssertEqual(log.fileURL(for: date), directory.protocolLogURL(for: date))
    }

    func testCreateDirectoriesMakesTheChallengesFolder() throws {
        try directory.createDirectories()
        XCTAssertEqual(try FileSafety.existingItem(at: directory.challengesDirectory)?.kind, .directory)
        try directory.createDirectories()
        XCTAssertEqual(try FileSafety.existingItem(at: directory.challengesDirectory)?.kind, .directory, "creating again is harmless")
    }
}
