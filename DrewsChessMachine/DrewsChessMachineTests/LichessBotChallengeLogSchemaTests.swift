//
//  LichessBotChallengeLogSchemaTests.swift
//  DrewsChessMachineTests
//
//  The challenge log's line format (challenge-log plan §3.2) and its
//  decoding rule. A golden line per event case pins the format byte for
//  byte; a line from a newer build is skipped and counted, never treated as
//  corruption; a complete bad line is reported with file and line and costs
//  its file; an unterminated tail is dropped and reported.
//

import XCTest
@testable import DrewsChessMachine

final class LichessBotChallengeLogSchemaTests: XCTestCase {

    private typealias F = LichessBotChallengeLogFixtures

    private var tempRoot: URL!

    override func setUpWithError() throws {
        tempRoot = FileManager.default.temporaryDirectory
            .appendingPathComponent("LichessBotChallengeLogSchemaTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: tempRoot, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try FileManager.default.removeItem(at: tempRoot)
    }

    private var directory: LichessBotDataDirectory { LichessBotDataDirectory(root: tempRoot) }

    // MARK: - Golden lines

    private static let snapshotJSON = #"{"challenger":{"id":"drewschessmachine","name":"DrewsChessMachine","provisional":false,"rating":1500,"title":"BOT"},"color":"random","destUser":{"id":"maia1","name":"maia1","rating":1400,"title":"BOT"},"finalColor":"white","id":"AbCd1234","incrementSeconds":2,"limitSeconds":180,"rated":true,"speed":"blitz","timeControlType":"clock","variant":"standard"}"#
    private static let requestJSON = #"{"clockIncrementSeconds":2,"clockLimitSeconds":180,"color":"random","rated":true}"#

    private static func line(_ eventJSON: String) -> String {
        #"{"at":"2026-10-05T12:00:00.000Z","build":2400,"event":"# + eventJSON + #","gitHash":"0123abc","schemaVersion":1}"#
    }

    /// Every event case (and each nested case), with the exact line it
    /// writes.
    private static let goldens: [(LichessBotChallengeLogEvent, String)] = [
        (F.created(), line(#"{"outgoingCreated":{"challenge":"# + snapshotJSON + #","creditCost":1,"opponentKind":{"bot":{}},"request":"# + requestJSON + #","sender":{"matchmaking":{"fillMode":"everyFreeSlot","trigger":"automaticPass"}}}}"#)),
        (.outgoingCreated(challenge: F.snapshot(), sender: .challengeSheet, request: F.request, opponentKind: .human, creditCost: 5),
         line(#"{"outgoingCreated":{"challenge":"# + snapshotJSON + #","creditCost":5,"opponentKind":{"human":{}},"request":"# + requestJSON + #","sender":{"challengeSheet":{}}}}"#)),
        (.outgoingNotCreated(attemptID: F.attemptID(1), opponentID: "maia1", sender: .casualResendOffer, request: F.request, opponentKind: .bot,
                             reason: .refused(LichessBotChallengeRefusal(kind: .botDailyGameLimit, httpStatus: 400, text: "too many games against other bots today")), creditCost: 1),
         line(#"{"outgoingNotCreated":{"attemptID":"00000000-0000-0000-0000-000000000001","creditCost":1,"opponentID":"maia1","opponentKind":{"bot":{}},"reason":{"refused":{"_0":{"httpStatus":400,"kind":"botDailyGameLimit","text":"too many games against other bots today"}}},"request":"# + requestJSON + #","sender":{"casualResendOffer":{}}}}"#)),
        (.outgoingNotCreated(attemptID: F.attemptID(2), opponentID: "maia1", sender: .challengeQueue, request: F.request, opponentKind: nil,
                             reason: .opponentOffline, creditCost: 0),
         line(#"{"outgoingNotCreated":{"attemptID":"00000000-0000-0000-0000-000000000002","creditCost":0,"opponentID":"maia1","reason":{"opponentOffline":{}},"request":"# + requestJSON + #","sender":{"challengeQueue":{}}}}"#)),
        (.outgoingNotCreated(attemptID: F.attemptID(3), opponentID: "maia1", sender: .matchmaking(trigger: .fillOpenSlots, fillMode: .onlyWhenIdle),
                             request: F.request, opponentKind: .bot, reason: .noAnswer(error: "timed out"), creditCost: 1),
         line(#"{"outgoingNotCreated":{"attemptID":"00000000-0000-0000-0000-000000000003","creditCost":1,"opponentID":"maia1","opponentKind":{"bot":{}},"reason":{"noAnswer":{"error":"timed out"}},"request":"# + requestJSON + #","sender":{"matchmaking":{"fillMode":"onlyWhenIdle","trigger":"fillOpenSlots"}}}}"#)),
        (.outgoingSeenWithoutCreatedLine(challenge: F.snapshot(), attribution: .unansweredSend(attemptID: F.attemptID(3), sender: .matchmakingCasualResend)),
         line(#"{"outgoingSeenWithoutCreatedLine":{"attribution":{"unansweredSend":{"attemptID":"00000000-0000-0000-0000-000000000003","sender":{"matchmakingCasualResend":{}}}},"challenge":"# + snapshotJSON + #"}}"#)),
        (.outgoingSeenWithoutCreatedLine(challenge: F.snapshot(), attribution: .notRecorded),
         line(#"{"outgoingSeenWithoutCreatedLine":{"attribution":{"notRecorded":{}},"challenge":"# + snapshotJSON + #"}}"#)),
        (.withdrawalRequested(challengeID: "AbCd1234", reason: .operatorCancel),
         line(#"{"withdrawalRequested":{"challengeID":"AbCd1234","reason":{"operatorCancel":{}}}}"#)),
        (.withdrawalRequested(challengeID: "AbCd1234", reason: .unansweredTimeout(seconds: 60)),
         line(#"{"withdrawalRequested":{"challengeID":"AbCd1234","reason":{"unansweredTimeout":{"seconds":60}}}}"#)),
        (.withdrawalRequested(challengeID: "AbCd1234", reason: .goingOffline),
         line(#"{"withdrawalRequested":{"challengeID":"AbCd1234","reason":{"goingOffline":{}}}}"#)),
        (.withdrawalRequested(challengeID: "AbCd1234", reason: .wentOfflineWhileSending),
         line(#"{"withdrawalRequested":{"challengeID":"AbCd1234","reason":{"wentOfflineWhileSending":{}}}}"#)),
        (.withdrawalResult(challengeID: "AbCd1234", result: .confirmed),
         line(#"{"withdrawalResult":{"challengeID":"AbCd1234","result":{"confirmed":{}}}}"#)),
        (.withdrawalResult(challengeID: "AbCd1234", result: .alreadyGone(message: "Not found")),
         line(#"{"withdrawalResult":{"challengeID":"AbCd1234","result":{"alreadyGone":{"message":"Not found"}}}}"#)),
        (.withdrawalResult(challengeID: "AbCd1234", result: .alreadyGone(message: nil)),
         line(#"{"withdrawalResult":{"challengeID":"AbCd1234","result":{"alreadyGone":{}}}}"#)),
        (.withdrawalResult(challengeID: "AbCd1234", result: .failed(error: "HTTP 500")),
         line(#"{"withdrawalResult":{"challengeID":"AbCd1234","result":{"failed":{"error":"HTTP 500"}}}}"#)),
        (.withdrawalResult(challengeID: "AbCd1234", result: .abandonedAtShutdown),
         line(#"{"withdrawalResult":{"challengeID":"AbCd1234","result":{"abandonedAtShutdown":{}}}}"#)),
        (.incomingReceived(challenge: F.snapshot()),
         line(#"{"incomingReceived":{"challenge":"# + snapshotJSON + #"}}"#)),
        (.incomingDecided(challengeID: "AbCd1234", decision: .accept),
         line(#"{"incomingDecided":{"challengeID":"AbCd1234","decision":{"accept":{}}}}"#)),
        (.incomingDecided(challengeID: "AbCd1234", decision: .decline(reason: .tooFast, rule: "speed bullet")),
         line(#"{"incomingDecided":{"challengeID":"AbCd1234","decision":{"decline":{"reason":"tooFast","rule":"speed bullet"}}}}"#)),
        (.incomingDecided(challengeID: "AbCd1234", decision: .ignore(rule: "own outgoing challenge")),
         line(#"{"incomingDecided":{"challengeID":"AbCd1234","decision":{"ignore":{"rule":"own outgoing challenge"}}}}"#)),
        (.incomingResponseFailed(challengeID: "AbCd1234", error: "HTTP 500"),
         line(#"{"incomingResponseFailed":{"challengeID":"AbCd1234","error":"HTTP 500"}}"#)),
        (.declinedOnLichess(challengeID: "AbCd1234", reason: .known(.later), text: "Not now"),
         line(#"{"declinedOnLichess":{"challengeID":"AbCd1234","reason":{"known":{"_0":"later"}},"text":"Not now"}}"#)),
        (.declinedOnLichess(challengeID: "AbCd1234", reason: .unrecognized("newKey"), text: nil),
         line(#"{"declinedOnLichess":{"challengeID":"AbCd1234","reason":{"unrecognized":{"_0":"newKey"}}}}"#)),
        (.declinedOnLichess(challengeID: "AbCd1234", reason: .unstated, text: nil),
         line(#"{"declinedOnLichess":{"challengeID":"AbCd1234","reason":{"unstated":{}}}}"#)),
        (.canceledOnLichess(challengeID: "AbCd1234"),
         line(#"{"canceledOnLichess":{"challengeID":"AbCd1234"}}"#)),
        (.gameStarted(challengeID: "AbCd1234"),
         line(#"{"gameStarted":{"challengeID":"AbCd1234"}}"#)),
        (.unterminatedLineCut(byteCount: 5, base64: "eyJhdCI="),
         line(#"{"unterminatedLineCut":{"base64":"eyJhdCI=","byteCount":5}}"#)),
    ]

    func testEveryEventCaseHasAGoldenLine() {
        // One entry per case of the enum, so a new case can't go unpinned.
        func caseName(_ event: LichessBotChallengeLogEvent) -> String {
            String(String(describing: event).prefix { $0 != "(" })
        }
        let covered = Set(Self.goldens.map { caseName($0.0) })
        XCTAssertEqual(covered, [
            "outgoingCreated", "outgoingNotCreated", "outgoingSeenWithoutCreatedLine", "withdrawalRequested",
            "withdrawalResult", "incomingReceived", "incomingDecided", "incomingResponseFailed",
            "declinedOnLichess", "canceledOnLichess", "gameStarted", "unterminatedLineCut",
        ])
    }

    func testGoldenLinesEncodeAndDecodeByteForByte() throws {
        for (event, golden) in Self.goldens {
            let entry = F.entry(event, at: 0)
            let encoded = try LichessBotJSONLines.encodeLine(entry)
            XCTAssertEqual(String(decoding: encoded, as: UTF8.self), golden + "\n")
            let decoded = try LichessBotChallengeLog.decodeDayFile(Data((golden + "\n").utf8), fileName: "golden")
            XCTAssertEqual(decoded.entries, [entry])
        }
    }

    func testTheSnapshotMapsTheAPIChallenge() throws {
        let json = #"{"id":"AbCd1234","url":"https://lichess.org/AbCd1234","status":"created","challenger":{"id":"drewschessmachine","name":"DrewsChessMachine","rating":1500,"title":"BOT","provisional":false,"online":true},"destUser":{"id":"maia1","name":"maia1","rating":1400,"title":"BOT","online":true},"variant":{"key":"standard","name":"Standard","short":"Std"},"rated":true,"speed":"blitz","timeControl":{"type":"clock","limit":180,"increment":2,"show":"3+2"},"color":"random","finalColor":"white","perf":{"icon":";","name":"Blitz"},"direction":"out"}"#
        let challenge = try JSONDecoder().decode(LichessBotChallenge.self, from: Data(json.utf8))
        XCTAssertEqual(LichessBotChallengeSnapshot(challenge), F.snapshot())
    }

    func testTheDecisionRecordMapsEveryPolicyDecision() {
        XCTAssertEqual(LichessBotIncomingDecisionRecord(.accept), .accept)
        XCTAssertEqual(LichessBotIncomingDecisionRecord(.decline(.rated, rule: "rated off")), .decline(reason: .rated, rule: "rated off"))
        XCTAssertEqual(LichessBotIncomingDecisionRecord(.ignore(rule: "budget")), .ignore(rule: "budget"))
    }

    func testAnEntryWrittenByThisBuildCarriesTheCurrentVersionAndBuild() {
        let entry = LichessBotChallengeLogEntry(at: F.start, event: .gameStarted(challengeID: "x"))
        XCTAssertEqual(entry.schemaVersion, LichessBotChallengeLogEntry.currentSchemaVersion)
        XCTAssertEqual(entry.build, BuildInfo.buildNumber)
        XCTAssertEqual(entry.gitHash, BuildInfo.gitHash)
    }

    // MARK: - Decoding rule

    func testALineFromANewerBuildIsSkippedAndCounted() throws {
        let newer = #"{"at":"2026-10-05T12:00:01.000Z","build":9999,"event":{"aCaseThisBuildDoesNotKnow":{}},"gitHash":"f","schemaVersion":"# + "\(LichessBotChallengeLogEntry.currentSchemaVersion + 1)}"
        let data = Data((Self.goldens[0].1 + "\n" + newer + "\n" + Self.goldens[10].1 + "\n").utf8)
        let decoded = try LichessBotChallengeLog.decodeDayFile(data, fileName: "challenges-20261005.jsonl")
        XCTAssertEqual(decoded.entries.map(\.event), [Self.goldens[0].0, Self.goldens[10].0])
        XCTAssertEqual(decoded.skippedNewerLines, 1)
        XCTAssertEqual(decoded.lineCount, 3)
        XCTAssertEqual(decoded.droppedTrailingByteCount, 0)
    }

    func testACompleteBadLineAtTheCurrentVersionIsReportedWithFileAndLine() {
        let unknownCase = #"{"at":"2026-10-05T12:00:01.000Z","build":2400,"event":{"aCaseThisBuildDoesNotKnow":{}},"gitHash":"0123abc","schemaVersion":1}"#
        let noVersion = #"{"at":"2026-10-05T12:00:01.000Z","build":2400,"event":{"gameStarted":{"challengeID":"x"}},"gitHash":"0123abc"}"#
        for (bad, expectedLine) in [(unknownCase, 2), (noVersion, 2), ("not json", 2)] {
            let data = Data((Self.goldens[0].1 + "\n" + bad + "\n" + Self.goldens[1].1 + "\n").utf8)
            XCTAssertThrowsError(try LichessBotChallengeLog.decodeDayFile(data, fileName: "challenges-20261005.jsonl")) { error in
                guard case LichessBotJSONLinesError.undecodableLine(let file, let lineNumber, _) = error else {
                    return XCTFail("unexpected error \(error)")
                }
                XCTAssertEqual(file, "challenges-20261005.jsonl")
                XCTAssertEqual(lineNumber, expectedLine)
            }
        }
    }

    func testAnUnterminatedTailIsDroppedAndReported() throws {
        let fragment = #"{"at":"2026-10-05T12:00:01.000Z","bui"#
        let decoded = try LichessBotChallengeLog.decodeDayFile(Data((Self.goldens[0].1 + "\n" + fragment).utf8), fileName: "x")
        XCTAssertEqual(decoded.entries.count, 1)
        XCTAssertEqual(decoded.droppedTrailingByteCount, fragment.utf8.count)
    }

    // MARK: - Reading the folder

    private func writeDayFile(_ name: String, _ lines: [String], trailing: String = "") throws {
        try FileManager.default.createDirectory(at: directory.challengesDirectory, withIntermediateDirectories: true)
        let text = lines.map { $0 + "\n" }.joined() + trailing
        try Data(text.utf8).write(to: directory.challengesDirectory.appendingPathComponent(name))
    }

    func testAMissingFolderIsAnEmptyLog() throws {
        XCTAssertEqual(try LichessBotChallengeLog.readAll(in: directory), LichessBotChallengeLogContents())
    }

    func testDayFilesAreReadOldestFirstAndOtherFilesIgnored() throws {
        try writeDayFile("challenges-20261006.jsonl", [Self.goldens[10].1])
        try writeDayFile("challenges-20261005.jsonl", [Self.goldens[0].1, Self.goldens[1].1])
        try writeDayFile("reconstructed-from-protocol.json", ["{}"])
        try writeDayFile("challenges-2026105.jsonl", ["not a day file name"])
        let contents = try LichessBotChallengeLog.readAll(in: directory)
        XCTAssertEqual(contents.filesRead.map(\.name), ["challenges-20261005.jsonl", "challenges-20261006.jsonl"])
        XCTAssertEqual(contents.entries.map(\.event), [Self.goldens[0].0, Self.goldens[1].0, Self.goldens[10].0])
        XCTAssertEqual(contents.lineCount, 3)
        XCTAssertEqual(contents.filesLeftOut, [])
    }

    func testACorruptOrNonRegularDayFileIsLeftOutAndTheRestRead() throws {
        try writeDayFile("challenges-20261004.jsonl", [Self.goldens[0].1, "not json"])
        try writeDayFile("challenges-20261005.jsonl", [Self.goldens[10].1], trailing: #"{"at""#)
        let target = tempRoot.appendingPathComponent("elsewhere.jsonl")
        try Data((Self.goldens[1].1 + "\n").utf8).write(to: target)
        try FileManager.default.createSymbolicLink(at: directory.challengesDirectory.appendingPathComponent("challenges-20261006.jsonl"),
                                                   withDestinationURL: target)
        let contents = try LichessBotChallengeLog.readAll(in: directory)
        XCTAssertEqual(contents.filesRead.map(\.name), ["challenges-20261005.jsonl"])
        XCTAssertEqual(contents.filesRead.first?.droppedTrailingByteCount, #"{"at""#.utf8.count)
        XCTAssertEqual(contents.entries.map(\.event), [Self.goldens[10].0])
        XCTAssertEqual(contents.filesLeftOut.map(\.name), ["challenges-20261004.jsonl", "challenges-20261006.jsonl"])
        XCTAssertTrue(contents.filesLeftOut[0].reason.contains("line 2"), contents.filesLeftOut[0].reason)
        XCTAssertTrue(contents.filesLeftOut[1].reason.contains("symbolic link"), contents.filesLeftOut[1].reason)
    }

    func testAChallengesPathThatIsNotAFolderCannotBeRead() throws {
        try Data("x".utf8).write(to: tempRoot.appendingPathComponent("Challenges", isDirectory: false))
        XCTAssertThrowsError(try LichessBotChallengeLog.readAll(in: directory)) { error in
            XCTAssertEqual(error as? LichessBotChallengeLogError,
                           .folderIsNotADirectory(path: directory.challengesDirectory.path, kind: .regularFile))
        }
    }
}
