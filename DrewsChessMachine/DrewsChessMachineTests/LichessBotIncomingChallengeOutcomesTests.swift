//
//  LichessBotIncomingChallengeOutcomesTests.swift
//  DrewsChessMachineTests
//
//  The incoming-challenge counts as a fold of the challenge log: each row
//  state (and each decision DCM made with nothing after it) maps to one
//  outcome, a failure after DCM's answer marks that answer failed, declines
//  are counted by reason, DCM's own challenges stay out, and only
//  challenges received within the rolling day count.
//

import XCTest
@testable import DrewsChessMachine

final class LichessBotIncomingChallengeOutcomesTests: XCTestCase {

    private typealias F = LichessBotChallengeLogFixtures

    private func ledger(_ events: [(LichessBotChallengeLogEvent, TimeInterval)]) -> LichessBotChallengeLedger {
        var ledger = LichessBotChallengeLedger(loadStatus: .complete)
        for (event, seconds) in events {
            ledger.apply(F.entry(event, at: seconds))
        }
        return ledger
    }

    private func fold(_ ledger: LichessBotChallengeLedger, now: Date = F.at(600)) -> LichessBotIncomingChallengeOutcomeLog {
        LichessBotIncomingChallengeOutcomeLog.fold(ledger: ledger, history: nil, liveLogFirstEntryAt: nil, now: now)
    }

    private func outcomes(_ log: LichessBotIncomingChallengeOutcomeLog) -> [String: LichessBotIncomingChallengeOutcome] {
        Dictionary(uniqueKeysWithValues: log.records.map { ($0.id, $0.outcome) })
    }

    func testEachRowStateMapsToItsOutcome() {
        let log = fold(ledger([
            (F.received("open"), 1),
            (F.received("acc"), 2), (.incomingDecided(challengeID: "acc", decision: .accept), 3), (.gameStarted(challengeID: "acc"), 4),
            (F.received("dec"), 5), (.declinedOnLichess(challengeID: "dec", reason: .known(.tooFast), text: nil), 6),
            (F.received("can"), 7), (.canceledOnLichess(challengeID: "can"), 8),
            (F.received("ign"), 9), (.incomingDecided(challengeID: "ign", decision: .ignore(rule: "busy")), 10),
            (F.received("noGame"), 11), (.incomingDecided(challengeID: "noGame", decision: .accept), 12),
            (F.received("dcmDec"), 13), (.incomingDecided(challengeID: "dcmDec", decision: .decline(reason: .noBot, rule: "humans only")), 14),
        ]))
        let byID = outcomes(log)
        XCTAssertEqual(byID["open"], .notAnswered)
        XCTAssertEqual(byID["acc"], .accepted)
        XCTAssertEqual(byID["dec"], .declined(.known(.tooFast)))
        XCTAssertEqual(byID["can"], .canceledByChallenger)
        XCTAssertEqual(byID["ign"], .ignored)
        XCTAssertEqual(byID["noGame"], .acceptedNoGameStarted)
        XCTAssertEqual(byID["dcmDec"], .declined(.known(.noBot)), "with no decline event, DCM's own reason counts")
        XCTAssertEqual(log.records.first { $0.id == "acc" }?.challengerID, F.maia.id)

        let summary = log.summary(now: F.at(600))
        XCTAssertEqual(summary.received, 7)
        XCTAssertEqual(summary.accepted, 1)
        XCTAssertEqual(summary.acceptedNoGameStarted, 1)
        XCTAssertEqual(summary.declined, 2)
        XCTAssertEqual(summary.canceledByChallenger, 1)
        XCTAssertEqual(summary.ignored, 1)
        XCTAssertEqual(summary.answerFailed, 0)
        XCTAssertEqual(summary.notAnswered, 1, "an unanswered challenge is the pending count")
    }

    func testAFailureAfterTheAnswerMarksThatAnswerFailed() {
        let log = fold(ledger([
            (F.received("accFail"), 1), (.incomingDecided(challengeID: "accFail", decision: .accept), 2),
            (.incomingResponseFailed(challengeID: "accFail", error: "HTTP 429"), 3),
            (F.received("decFail"), 4), (.incomingDecided(challengeID: "decFail", decision: .decline(reason: .later, rule: "full")), 5),
            (.incomingResponseFailed(challengeID: "decFail", error: "timed out"), 6),
            (F.received("retried"), 7), (.incomingResponseFailed(challengeID: "retried", error: "timed out"), 8),
            (.incomingDecided(challengeID: "retried", decision: .accept), 9),
            (F.received("failedThenStarted"), 10), (.incomingDecided(challengeID: "failedThenStarted", decision: .accept), 11),
            (.incomingResponseFailed(challengeID: "failedThenStarted", error: "timed out"), 12), (.gameStarted(challengeID: "failedThenStarted"), 13),
        ]))
        let byID = outcomes(log)
        XCTAssertEqual(byID["accFail"], .answerFailed(LichessBotIncomingAnswerFailure(answer: .accept, at: F.at(3), error: "HTTP 429")))
        XCTAssertEqual(byID["decFail"], .answerFailed(LichessBotIncomingAnswerFailure(answer: .decline, at: F.at(6), error: "timed out")))
        XCTAssertEqual(byID["retried"], .acceptedNoGameStarted, "a failure before the latest decision doesn't fail it")
        XCTAssertEqual(byID["failedThenStarted"], .accepted, "a started game outranks a failed answer")

        let summary = log.summary(now: F.at(600))
        XCTAssertEqual(summary.answerFailed, 2)
        XCTAssertEqual(summary.answerFailedByAnswer, [.accept: 1, .decline: 1])
        XCTAssertEqual(log.latestAnswerFailure?.record.id, "decFail")
        XCTAssertEqual(log.latestAnswerFailure?.failure.error, "timed out")
    }

    func testDeclinesAreCountedByReason() {
        let log = fold(ledger([
            (F.received("a"), 1), (.declinedOnLichess(challengeID: "a", reason: .known(.noBot), text: nil), 2),
            (F.received("b"), 3), (.declinedOnLichess(challengeID: "b", reason: .known(.noBot), text: nil), 4),
            (F.received("c"), 5), (.declinedOnLichess(challengeID: "c", reason: .unrecognized("someNewKey"), text: nil), 6),
            (F.received("d"), 7), (.declinedOnLichess(challengeID: "d", reason: .unstated, text: nil), 8),
            (F.received("e"), 9), (.incomingDecided(challengeID: "e", decision: .decline(reason: .rated, rule: "casual only")), 10),
        ]))
        let summary = log.summary(now: F.at(600))
        XCTAssertEqual(summary.declined, 5)
        XCTAssertEqual(summary.declinedByReason, [
            .known(.noBot): 2, .unrecognized("someNewKey"): 1, .unstated: 1, .known(.rated): 1,
        ])
    }

    func testDCMsOwnChallengesStayOut() {
        let log = fold(ledger([
            (F.created("mine"), 1), (.gameStarted(challengeID: "mine"), 2),
            (.outgoingSeenWithoutCreatedLine(challenge: F.snapshot(id: "echo"), attribution: .notRecorded), 3),
            (F.received("theirs"), 4),
        ]))
        XCTAssertEqual(log.records.map(\.id), ["theirs"])
    }

    func testOnlyChallengesReceivedWithinTheRollingDayCount() {
        let day = LichessBotChallengeCredits.dayWindow
        // "edge" is exactly a day old at `now`, so it is out, as the
        // outgoing log's `prune` drops a record exactly a day old.
        let ledger = ledger([(F.received("edge"), 0), (F.received("inside"), 1), (F.received("late"), day)])
        let log = fold(ledger, now: F.at(day))
        XCTAssertEqual(log.records.map(\.id), ["inside", "late"])
        XCTAssertEqual(log.summary(now: F.at(day)).received, 2)
        XCTAssertEqual(log.summary(now: F.at(day + 1)).received, 1, "the summary applies the window at its own time")
    }

    func testAnOldChallengeAnsweredRecentlyIsCountedByWhenItWasReceived() {
        let day = LichessBotChallengeCredits.dayWindow
        let log = fold(ledger([(F.received("old"), 0), (.canceledOnLichess(challengeID: "old"), day + 5)]), now: F.at(day + 10))
        XCTAssertTrue(log.records.isEmpty)
    }
}
