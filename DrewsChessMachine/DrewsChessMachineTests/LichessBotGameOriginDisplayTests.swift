//
//  LichessBotGameOriginDisplayTests.swift
//  DrewsChessMachineTests
//
//  The one resolver of what a game shows for how it began (challenge-log
//  plan §3.6), step by step: the record's determined origin, then the live
//  challenge log, then the challenge rebuilt from the protocol log (with its
//  confidence), then the record's undetermined gap, then unknown — split by
//  whether the game was created before the live log began. Also the live
//  log's first entry as the reader computes it (the reconstruction cutoff).
//

import XCTest
@testable import DrewsChessMachine

final class LichessBotGameOriginDisplayTests: XCTestCase {

    private typealias F = LichessBotChallengeLogFixtures

    private static let us = "drewschessmachine"

    private func ledger(_ events: [LichessBotChallengeLogEvent]) -> LichessBotChallengeLedger {
        var ledger = LichessBotChallengeLedger(loadStatus: .complete)
        for (index, event) in events.enumerated() {
            ledger.apply(F.entry(event, at: TimeInterval(index)))
        }
        return ledger
    }

    // MARK: - A small protocol log to rebuild from

    private static func party(_ id: String) -> String {
        #"{"name":"\#(id)","title":"BOT","id":"\#(id)","rating":1500}"#
    }

    private static func challengeEvent(id: String, challenger: String, dest: String) -> String {
        #"{"type":"challenge","challenge":{"id":"\#(id)","status":"created","challenger":\#(party(challenger)),"destUser":\#(party(dest)),"variant":{"key":"standard"},"rated":true,"speed":"blitz","timeControl":{"type":"clock","limit":180,"increment":2},"color":"random"}}"#
    }

    /// `rIn` came in; `rMatch` was matchmaking's (a companion line beside
    /// its send); `rOperator` was a send with no companion (inferred).
    private func reconstruction() throws -> LichessBotReconstructedChallengeLookup {
        var entries: [LichessBotProtocolEntry] = []
        func add(_ kind: LichessBotProtocolEventKind, _ message: String, _ fields: [String: String], _ seconds: TimeInterval) {
            entries.append(LichessBotProtocolEntry(at: F.at(seconds), kind: kind, gameID: nil, message: message, fields: fields))
        }
        add(.stream, Self.challengeEvent(id: "rIn", challenger: "carol", dest: Self.us), ["stream": "event"], 0)
        add(.stream, Self.challengeEvent(id: "rMatch", challenger: Self.us, dest: "fitbot"), ["stream": "event"], 10)
        add(.challenge, "challenge sent to FitBot", ["id": "rMatch", "rated": "true", "clock": "180+2", "color": "random"], 10.1)
        add(.challenge, "matchmaking sent a challenge to FitBot", ["clock": "3+2", "rated": "true"], 10.2)
        add(.stream, Self.challengeEvent(id: "rOperator", challenger: Self.us, dest: "bob"), ["stream": "event"], 20)
        add(.challenge, "challenge sent to bob", ["id": "rOperator", "rated": "true", "clock": "180+2", "color": "random"], 20.1)
        var data = Data()
        for entry in entries {
            data.append(try LichessBotJSONLines.encodeLine(entry))
        }
        let built = LichessBotChallengeReconstruction.build(
            from: [LichessBotProtocolDayFile(name: "events-20261005.jsonl", data: data)], ourAccountID: Self.us, liveLogFirstEntryAt: nil)
        return LichessBotReconstructedChallengeLookup(built)
    }

    private func resolve(_ gameID: String, createdAt: Date = F.start, recorded: LichessBotGameOrigin? = nil,
                         ledger: LichessBotChallengeLedger? = nil, reconstruction: LichessBotReconstructedChallengeLookup? = nil,
                         liveLogFirstEntryAt: Date? = nil) -> LichessBotGameOriginDisplay {
        LichessBotGameOriginDisplay.resolve(gameID: gameID, createdAt: createdAt, recorded: recorded, ledger: ledger,
                                            reconstruction: reconstruction, liveLogFirstEntryAt: liveLogFirstEntryAt)
    }

    // MARK: - The order

    func testTheRecordsDeterminedOriginComesFirst() throws {
        let recorded = LichessBotGameOrigin.outgoingChallengeAccepted(challengeID: "AbCd1234", sender: .challengeQueue)
        let shown = resolve("AbCd1234", recorded: recorded, ledger: ledger([F.created(sender: .challengeSheet)]), reconstruction: try reconstruction())
        XCTAssertEqual(shown.category, .challengeQueue)
        XCTAssertEqual(shown.basis, .recorded)
    }

    func testTheLiveChallengeLogComesSecondEvenOverAnUndeterminedRecord() {
        let undetermined = LichessBotGameOrigin.undetermined(source: LichessBotOpenValue(.friend), gap: .noChallengeRecord)
        let shown = resolve("AbCd1234", recorded: undetermined, ledger: ledger([F.created()]))
        XCTAssertEqual(shown.category, .matchmaking)
        XCTAssertEqual(shown.basis, .challengeLog)
        XCTAssertTrue(shown.detail.contains("automatic pass, every free slot"), shown.detail)
        XCTAssertEqual(resolve("InCo5678", ledger: ledger([F.received()])).category, .incoming)
    }

    func testTheReconstructionComesThirdWithItsConfidence() throws {
        let lookup = try reconstruction()
        let incoming = resolve("rIn", reconstruction: lookup)
        XCTAssertEqual(incoming.category, .incoming)
        XCTAssertEqual(incoming.basis, .reconstructed(.certain))
        let matchmaking = resolve("rMatch", reconstruction: lookup)
        XCTAssertEqual(matchmaking.category, .matchmaking)
        XCTAssertEqual(matchmaking.basis, .reconstructed(.paired))
        let byOperator = resolve("rOperator", reconstruction: lookup)
        XCTAssertEqual(byOperator.category, .challengeSheet)
        XCTAssertEqual(byOperator.basis, .reconstructed(.inferredFromAbsence))
    }

    func testTheRecordsGapComesFourth() throws {
        let undetermined = LichessBotGameOrigin.undetermined(source: nil, gap: .challengeLogIncomplete)
        let shown = resolve("nowhere", recorded: undetermined, ledger: ledger([]), reconstruction: try reconstruction())
        XCTAssertEqual(shown.category, .unknown)
        XCTAssertEqual(shown.basis, .unknown(.gap(.challengeLogIncomplete)))
    }

    func testUnknownIsSplitByWhenTheGameWasCreated() {
        let firstEntry = F.at(3600)
        XCTAssertEqual(resolve("old", createdAt: F.at(0), liveLogFirstEntryAt: firstEntry).basis, .unknown(.playedBeforeOriginsWereRecorded))
        XCTAssertEqual(resolve("new", createdAt: F.at(7200), liveLogFirstEntryAt: firstEntry).basis, .unknown(.notRecorded))
        XCTAssertEqual(resolve("anyGame", createdAt: F.at(7200), liveLogFirstEntryAt: nil).basis, .unknown(.playedBeforeOriginsWereRecorded),
                       "with no challenge log at all, nothing was recorded yet")
    }

    func testEveryCategoryHasADisplayFromAnOrigin() {
        let origins: [(LichessBotGameOrigin, LichessBotGameOriginCategory)] = [
            (.acceptedIncomingChallenge(challengeID: "a", challengerID: "c"), .incoming),
            (.outgoingChallengeAccepted(challengeID: "a", sender: .challengeSheet), .challengeSheet),
            (.outgoingChallengeAccepted(challengeID: "a", sender: .casualResendOffer), .casualResendOffer),
            (.outgoingChallengeAccepted(challengeID: "a", sender: .challengeQueue), .challengeQueue),
            (.outgoingChallengeAccepted(challengeID: "a", sender: .matchmaking(trigger: .fillOpenSlots, fillMode: .onlyWhenIdle)), .matchmaking),
            (.outgoingChallengeAccepted(challengeID: "a", sender: .matchmakingCasualResend), .matchmakingCasualResend),
            (.outgoingChallengeSenderNotRecorded(challengeID: "a"), .outgoingSenderNotRecorded),
            (.tournament(source: LichessBotOpenValue(.swiss), tournamentID: "t"), .tournament),
            (.undetermined(source: nil, gap: .noChallengeRecord), .unknown),
        ]
        XCTAssertEqual(Set(origins.map(\.1)), Set(LichessBotGameOriginCategory.allCases))
        for (origin, category) in origins {
            XCTAssertEqual(LichessBotGameOriginDisplay.display(origin, basis: .recorded).category, category)
        }
    }

    // MARK: - The live log's first entry (the cutoff)

    func testTheLiveLogBeginsAtTheOldestDayFilesFirstEntry() {
        let read = [
            LichessBotChallengeLogContents.DayFile(name: "challenges-20261005.jsonl", byteCount: 1, lineCount: 1, skippedNewerLines: 0, droppedTrailingByteCount: 0, firstEntryAt: F.at(100)),
            LichessBotChallengeLogContents.DayFile(name: "challenges-20261006.jsonl", byteCount: 1, lineCount: 1, skippedNewerLines: 0, droppedTrailingByteCount: 0, firstEntryAt: F.at(90_000)),
        ]
        XCTAssertEqual(LichessBotChallengeLogContents(entries: [], filesRead: read, filesLeftOut: []).liveLogFirstEntryAt, F.at(100))
        XCTAssertNil(LichessBotChallengeLogContents().liveLogFirstEntryAt)
    }

    func testAnOlderLeftOutDayFileMovesTheCutoffToItsDayStart() throws {
        let read = [LichessBotChallengeLogContents.DayFile(name: "challenges-20261006.jsonl", byteCount: 1, lineCount: 1, skippedNewerLines: 0, droppedTrailingByteCount: 0, firstEntryAt: F.at(90_000))]
        let leftOut = [LichessBotChallengeLogContents.LeftOutFile(name: "challenges-20261005.jsonl", reason: "corrupt")]
        let contents = LichessBotChallengeLogContents(entries: [], filesRead: read, filesLeftOut: leftOut)
        XCTAssertEqual(contents.liveLogFirstEntryAt, Date(timeIntervalSince1970: 1_791_158_400), "2026-10-05T00:00:00Z")
    }
}
