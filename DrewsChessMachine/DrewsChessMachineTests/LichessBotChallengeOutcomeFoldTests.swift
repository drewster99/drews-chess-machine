//
//  LichessBotChallengeOutcomeFoldTests.swift
//  DrewsChessMachineTests
//
//  The outcome log as a fold of the challenge log (challenge-log plan §3.8,
//  OD-2, P6): each row state maps to the outcome the old log recorded, a
//  started game outranks a withdrawal, unanswered sends and other players'
//  challenges stay out, the rolling day is kept, rebuilt rows fill only the
//  part of the day before the live log began (live rows win on an id), and
//  a refold keeps every record's id.
//

import XCTest
@testable import DrewsChessMachine

final class LichessBotChallengeOutcomeFoldTests: XCTestCase {

    private typealias F = LichessBotChallengeLogFixtures

    private func ledger(_ events: [(LichessBotChallengeLogEvent, TimeInterval)]) -> LichessBotChallengeLedger {
        var ledger = LichessBotChallengeLedger(loadStatus: .complete)
        for (event, seconds) in events {
            ledger.apply(F.entry(event, at: seconds))
        }
        return ledger
    }

    private func fold(_ ledger: LichessBotChallengeLedger, history: LichessBotChallengeReconstruction? = nil,
                      liveLogFirstEntryAt: Date? = nil, now: Date = F.at(600)) -> LichessBotChallengeOutcomeLog {
        LichessBotChallengeOutcomeLog.fold(ledger: ledger, history: history, liveLogFirstEntryAt: liveLogFirstEntryAt, now: now)
    }

    private func created(_ id: String) -> LichessBotChallengeLogEvent {
        F.created(id)
    }

    func testEachStateMapsToTheOldOutcome() {
        let log = fold(ledger([
            (created("open"), 1),
            (created("acc"), 2), (.gameStarted(challengeID: "acc"), 3),
            (created("dec"), 4), (.declinedOnLichess(challengeID: "dec", reason: .known(.noBot), text: nil), 5),
            (created("wd"), 6), (.withdrawalRequested(challengeID: "wd", reason: .operatorCancel), 7),
            (created("can"), 8), (.canceledOnLichess(challengeID: "can"), 9),
            (created("late"), 10), (.withdrawalRequested(challengeID: "late", reason: .goingOffline), 11), (.gameStarted(challengeID: "late"), 12),
        ]))
        let outcomes = Dictionary(uniqueKeysWithValues: log.records.compactMap { record in record.challengeID.map { ($0, record.outcome) } })
        XCTAssertEqual(outcomes["open"], .some(nil))
        XCTAssertEqual(outcomes["acc"], .accepted)
        XCTAssertEqual(outcomes["dec"], .declined(.known(.noBot)))
        XCTAssertEqual(outcomes["wd"], .canceled)
        XCTAssertEqual(outcomes["can"], .canceled)
        XCTAssertEqual(outcomes["late"], .accepted, "a started game outranks the withdrawal, as an acceptance replaced an inferred cancel")
        XCTAssertEqual(log.records.first { $0.challengeID == "acc" }?.resolvedAt, F.at(3))
        XCTAssertEqual(log.summary(now: F.at(600)).creditsLastDay, 6, "six bot challenges at one credit each")
    }

    func testNotCreatedAttemptsAndWhatStaysOut() {
        let refusal = LichessBotChallengeRefusal(kind: .botDailyGameLimit, httpStatus: 400, text: "limit")
        let log = fold(ledger([
            (.outgoingNotCreated(attemptID: F.attemptID(1), opponentID: "Maia1", sender: .challengeSheet, request: F.request,
                                 opponentKind: .bot, reason: .opponentOffline, creditCost: 0), 1),
            (.outgoingNotCreated(attemptID: F.attemptID(2), opponentID: "maia1", sender: .challengeSheet, request: F.request,
                                 opponentKind: .bot, reason: .refused(refusal), creditCost: 1), 2),
            (.outgoingNotCreated(attemptID: F.attemptID(3), opponentID: "maia1", sender: .challengeSheet, request: F.request,
                                 opponentKind: .bot, reason: .noAnswer(error: "timed out"), creditCost: 1), 3),
            (.outgoingSeenWithoutCreatedLine(challenge: F.snapshot(id: "echoOnly"), attribution: .notRecorded), 4),
            (F.received("incoming1"), 5),
        ]))
        XCTAssertEqual(log.records.map(\.outcome), [.offline, .refused(refusal)], "no-answer sends, echoes and incoming challenges stay out")
        XCTAssertEqual(log.records.map(\.id), [F.attemptID(1), F.attemptID(2)])
        XCTAssertEqual(log.records.first?.opponentID, "maia1")
        XCTAssertEqual(log.summary(now: F.at(600)).creditsLastDay, 1)
    }

    func testOnlyTheRollingDayIsKept() {
        let day = LichessBotChallengeCredits.dayWindow
        let log = fold(ledger([(created("old"), 0), (created("new"), day)]), now: F.at(day + 10))
        XCTAssertEqual(log.records.compactMap(\.challengeID), ["new"])
    }

    func testARefoldKeepsEveryRecordsID() {
        let first = fold(ledger([(created("a"), 1), (created("b"), 2)]))
        let again = fold(ledger([(created("b"), 2), (created("a"), 1), (.gameStarted(challengeID: "a"), 3)]))
        XCTAssertEqual(first.records.map(\.id), again.records.map(\.id))
    }

    // MARK: - Rebuilt rows

    private func history() throws -> LichessBotChallengeReconstruction {
        let party = { (id: String) in #"{"name":"\#(id)","title":"BOT","id":"\#(id)","rating":1500}"# }
        func event(_ id: String, at seconds: TimeInterval) -> LichessBotProtocolEntry {
            let json = #"{"type":"challenge","challenge":{"id":"\#(id)","status":"created","challenger":\#(party("drewschessmachine")),"destUser":\#(party("fitbot")),"variant":{"key":"standard"},"rated":true,"speed":"blitz","timeControl":{"type":"clock","limit":180,"increment":2},"color":"random"}}"#
            return LichessBotProtocolEntry(at: F.at(seconds), kind: .stream, gameID: nil, message: json, fields: ["stream": "event"])
        }
        var data = Data()
        for entry in [event("rBefore", at: 100), event("shared", at: 110), event("rAfter", at: 400)] {
            data.append(try LichessBotJSONLines.encodeLine(entry))
        }
        return LichessBotChallengeReconstruction.build(
            from: [LichessBotProtocolDayFile(name: "events-20261005.jsonl", data: data)], ourAccountID: "drewschessmachine", liveLogFirstEntryAt: nil)
    }

    func testRebuiltRowsFillOnlyTheTimeBeforeTheLiveLog() throws {
        let live = ledger([(created("shared"), 300)])
        let log = fold(live, history: try history(), liveLogFirstEntryAt: F.at(300))
        XCTAssertEqual(log.records.compactMap(\.challengeID), ["rBefore", "shared"],
                       "rAfter is after the live log began; shared is the live row's")
        XCTAssertEqual(log.records.first { $0.challengeID == "shared" }?.sentAt, F.at(300), "the live row wins on an id both hold")
        XCTAssertEqual(log.records.first?.opponentKind, .bot)
    }
}
