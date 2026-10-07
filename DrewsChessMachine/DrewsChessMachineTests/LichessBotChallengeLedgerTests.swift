//
//  LichessBotChallengeLedgerTests.swift
//  DrewsChessMachineTests
//
//  The pure fold of challenge-log facts into rows (challenge-log plan
//  §3.4): every state, the fold's precedence (the strongest fact wins, never
//  the last read), the sender rule, notes and anomalies, the load status,
//  and order independence — every permutation of a row's facts folds to the
//  same row.
//

import XCTest
@testable import DrewsChessMachine

final class LichessBotChallengeLedgerTests: XCTestCase {

    private typealias F = LichessBotChallengeLogFixtures

    private static let id = "AbCd1234"
    private static let incomingID = "InCo5678"

    /// A ledger of `events`, the nth one at second n.
    private func ledger(_ events: [LichessBotChallengeLogEvent]) -> LichessBotChallengeLedger {
        var ledger = LichessBotChallengeLedger(loadStatus: .complete)
        for (index, event) in events.enumerated() {
            ledger.apply(F.entry(event, at: TimeInterval(index)))
        }
        return ledger
    }

    private func row(_ events: [LichessBotChallengeLogEvent], id: String = LichessBotChallengeLedgerTests.id) throws -> LichessBotChallengeLedgerRow {
        try XCTUnwrap(ledger(events).row(challengeID: id))
    }

    // MARK: - Outgoing

    func testACreatedChallengeIsOpenWithItsSender() throws {
        let row = try self.row([F.created()])
        XCTAssertEqual(row.state, .open)
        XCTAssertEqual(row.direction, .outgoing)
        XCTAssertEqual(row.sender, F.automaticPass)
        XCTAssertEqual(row.snapshot, F.snapshot())
        XCTAssertEqual(row.notes, [])
        XCTAssertEqual(row.anomalies, [])
    }

    func testAStartedGameIsAcceptedWhicheverFactCameFirst() throws {
        XCTAssertEqual(try row([F.created(), .gameStarted(challengeID: Self.id)]).state, .accepted(gameStarted: true))
        let raced = try row([.gameStarted(challengeID: Self.id), F.created()])
        XCTAssertEqual(raced.state, .accepted(gameStarted: true), "the POST race: gameStarted before outgoingCreated")
        XCTAssertEqual(raced.sender, F.automaticPass)
    }

    func testDeclinesKeepLichesssReason() throws {
        for reason in [LichessBotDeclineReasonRecord.known(.tooFast), .unrecognized("newKey"), .unstated] {
            XCTAssertEqual(try row([F.created(), .declinedOnLichess(challengeID: Self.id, reason: reason, text: nil)]).state, .declined(reason))
        }
    }

    func testEveryWithdrawalReasonAndResult() throws {
        let reasons: [LichessBotWithdrawalReason] = [.operatorCancel, .unansweredTimeout(seconds: 60), .goingOffline, .wentOfflineWhileSending]
        let results: [LichessBotWithdrawalResult?] = [nil, .confirmed, .alreadyGone(message: "Not found"), .alreadyGone(message: nil), .failed(error: "HTTP 500"), .abandonedAtShutdown]
        for reason in reasons {
            for result in results {
                var events: [LichessBotChallengeLogEvent] = [F.created(), .withdrawalRequested(challengeID: Self.id, reason: reason)]
                if let result {
                    events.append(.withdrawalResult(challengeID: Self.id, result: result))
                    events.append(.canceledOnLichess(challengeID: Self.id))
                }
                XCTAssertEqual(try row(events).state, .withdrawn(reason, result), "\(reason) \(String(describing: result))")
            }
        }
    }

    func testTheLatestWithdrawalResultIsTheAnswer() throws {
        let row = try self.row([
            F.created(),
            .withdrawalRequested(challengeID: Self.id, reason: .operatorCancel),
            .withdrawalResult(challengeID: Self.id, result: .failed(error: "timeout")),
            .withdrawalRequested(challengeID: Self.id, reason: .goingOffline),
            .withdrawalResult(challengeID: Self.id, result: .confirmed),
        ])
        XCTAssertEqual(row.state, .withdrawn(.operatorCancel, .confirmed), "the first request's reason, the last answer")
    }

    func testAnOutgoingCancelWithNoWithdrawalIsWithdrawnForAnUnrecordedReason() throws {
        XCTAssertEqual(try row([F.created(), .canceledOnLichess(challengeID: Self.id)]).state, .canceledOnLichessWithoutRecordedWithdrawal)
        XCTAssertEqual(try row([.canceledOnLichess(challengeID: Self.id)]).state, .canceledOnLichessDirectionNotRecorded,
                       "with no fact saying which way it went, neither cancel state is claimed")
    }

    func testEveryNotCreatedReasonIsItsOwnAttemptRow() throws {
        let reasons: [LichessBotChallengeNotCreatedReason] = [
            .opponentOffline,
            .refused(LichessBotChallengeRefusal(kind: .rateLimited, httpStatus: 429, text: nil)),
            .noAnswer(error: "timed out"),
        ]
        var events: [LichessBotChallengeLogEvent] = []
        for (index, reason) in reasons.enumerated() {
            events.append(.outgoingNotCreated(attemptID: F.attemptID(UInt8(index + 1)), opponentID: "maia1", sender: .challengeSheet,
                                              request: F.request, opponentKind: .bot, reason: reason, creditCost: 0))
        }
        let ledger = self.ledger(events)
        XCTAssertEqual(ledger.rowsByKey.count, reasons.count)
        for (index, reason) in reasons.enumerated() {
            let row = try XCTUnwrap(ledger.row(attemptID: F.attemptID(UInt8(index + 1))))
            XCTAssertEqual(row.state, .notCreated(reason))
            XCTAssertEqual(row.sender, .challengeSheet)
            XCTAssertEqual(row.direction, .outgoing)
            XCTAssertNil(row.snapshot)
        }
    }

    // MARK: - Echoes and the sender rule

    func testAnEchoGivesTheSenderOnlyWithoutACreatedLine() throws {
        let attributed = try row([.outgoingSeenWithoutCreatedLine(challenge: F.snapshot(), attribution: .unansweredSend(attemptID: F.attemptID(1), sender: .challengeQueue))])
        XCTAssertEqual(attributed.sender, .challengeQueue)
        XCTAssertEqual(attributed.direction, .outgoing)
        let unattributed = try row([.outgoingSeenWithoutCreatedLine(challenge: F.snapshot(), attribution: .notRecorded)])
        XCTAssertNil(unattributed.sender)
        // Teardown during the POST: the echo is flushed as not recorded,
        // then the POST's created line lands.
        let both = try row([.outgoingSeenWithoutCreatedLine(challenge: F.snapshot(), attribution: .notRecorded), F.created(sender: .challengeSheet)])
        XCTAssertEqual(both.sender, .challengeSheet)
        XCTAssertEqual(both.anomalies, [])
    }

    func testTwoCreatedLinesKeepTheFirstAndAreAnAnomaly() throws {
        let row = try self.row([F.created(sender: .challengeSheet), F.created(sender: .challengeQueue)])
        XCTAssertEqual(row.sender, .challengeSheet)
        XCTAssertEqual(row.anomalies, [.repeatedCreatedLines(count: 2)])
    }

    // MARK: - Incoming

    func testIncomingDecisionsAndACancelByTheChallenger() throws {
        let decisions: [LichessBotIncomingDecisionRecord] = [.accept, .decline(reason: .tooFast, rule: "bullet"), .ignore(rule: "budget")]
        for decision in decisions {
            let decided = try row([F.received(), .incomingDecided(challengeID: Self.incomingID, decision: decision)], id: Self.incomingID)
            XCTAssertEqual(decided.state, .incomingDecided(decision))
            XCTAssertEqual(decided.direction, .incoming)
            XCTAssertNil(decided.sender)
            let canceled = try row([F.received(), .incomingDecided(challengeID: Self.incomingID, decision: decision),
                                    .canceledOnLichess(challengeID: Self.incomingID)], id: Self.incomingID)
            XCTAssertEqual(canceled.state, .canceledByChallenger)
        }
        XCTAssertEqual(try row([F.received()], id: Self.incomingID).state, .open)
    }

    func testAReplayedReceivedChallengeIsStillOneRow() throws {
        let ledger = self.ledger([F.received(), F.received(), .incomingDecided(challengeID: Self.incomingID, decision: .accept)])
        XCTAssertEqual(ledger.rowsByKey.count, 1)
        XCTAssertEqual(ledger.row(challengeID: Self.incomingID)?.state, .incomingDecided(.accept))
    }

    func testTheLatestDecisionIsTheState() throws {
        let row = try self.row([F.received(),
                           .incomingDecided(challengeID: Self.incomingID, decision: .ignore(rule: "budget")),
                           .incomingDecided(challengeID: Self.incomingID, decision: .accept)], id: Self.incomingID)
        XCTAssertEqual(row.state, .incomingDecided(.accept))
        XCTAssertEqual(row.decisions.count, 2, "each decision is a real request, so each is kept")
    }

    // MARK: - Precedence pairs

    func testAGameThatStartedAfterAWithdrawalIsAcceptedWithANote() throws {
        let row = try self.row([F.created(sender: .challengeSheet),
                           .withdrawalRequested(challengeID: Self.id, reason: .operatorCancel),
                           .withdrawalResult(challengeID: Self.id, result: .alreadyGone(message: "already accepted")),
                           .gameStarted(challengeID: Self.id)])
        XCTAssertEqual(row.state, .accepted(gameStarted: true))
        XCTAssertEqual(row.notes, [.withdrawalAttempted(.operatorCancel, .alreadyGone(message: "already accepted"))])
        XCTAssertEqual(row.sender, .challengeSheet)
    }

    func testAnIncomingChallengeDCMDeclinedThatStartedWasAcceptedOutsideDCM() throws {
        let decline = LichessBotIncomingDecisionRecord.decline(reason: .tooFast, rule: "bullet")
        let row = try self.row([F.received(), .incomingDecided(challengeID: Self.incomingID, decision: decline),
                           .declinedOnLichess(challengeID: Self.incomingID, reason: .known(.tooFast), text: nil),
                           .gameStarted(challengeID: Self.incomingID)], id: Self.incomingID)
        XCTAssertEqual(row.state, .accepted(gameStarted: true))
        XCTAssertEqual(row.notes, [.declinedOnLichess(.known(.tooFast)), .acceptedOutsideDCM(dcmDecided: decline)])
        let acceptedByDCM = try self.row([F.received(), .incomingDecided(challengeID: Self.incomingID, decision: .accept),
                                          .gameStarted(challengeID: Self.incomingID)], id: Self.incomingID)
        XCTAssertEqual(acceptedByDCM.notes, [])
    }

    func testAWithdrawalRequestWithNoResultHasNoRecordedAnswer() throws {
        XCTAssertEqual(try row([F.created(), .withdrawalRequested(challengeID: Self.id, reason: .goingOffline)]).state, .withdrawn(.goingOffline, nil))
    }

    func testDeclineOutranksWithdrawalAndCancel() throws {
        let row = try self.row([F.created(), .withdrawalRequested(challengeID: Self.id, reason: .unansweredTimeout(seconds: 60)),
                           .declinedOnLichess(challengeID: Self.id, reason: .known(.later), text: nil),
                           .canceledOnLichess(challengeID: Self.id)])
        XCTAssertEqual(row.state, .declined(.known(.later)))
    }

    func testAWithdrawalResultWithoutARequestIsAnAnomalyNotAWithdrawal() throws {
        let row = try self.row([F.created(), .withdrawalResult(challengeID: Self.id, result: .confirmed)])
        XCTAssertEqual(row.state, .open)
        XCTAssertEqual(row.anomalies, [.withdrawalResultWithoutRequest])
    }

    func testFactsOfBothDirectionsAreAnAnomaly() throws {
        let row = try self.row([F.created(), .incomingDecided(challengeID: Self.id, decision: .accept), .canceledOnLichess(challengeID: Self.id)])
        XCTAssertNil(row.direction)
        XCTAssertEqual(row.anomalies, [.conflictingDirections])
        XCTAssertEqual(row.state, .canceledOnLichessDirectionNotRecorded)
    }

    // MARK: - Order independence

    func testEveryPermutationOfARowsFactsFoldsToTheSameLedger() {
        let entries: [LichessBotChallengeLogEntry] = [
            F.entry(.outgoingSeenWithoutCreatedLine(challenge: F.snapshot(), attribution: .notRecorded), at: 0),
            F.entry(F.created(sender: .challengeQueue), at: 1),
            F.entry(.withdrawalRequested(challengeID: Self.id, reason: .operatorCancel), at: 2),
            F.entry(.withdrawalResult(challengeID: Self.id, result: .alreadyGone(message: nil)), at: 3),
            F.entry(.gameStarted(challengeID: Self.id), at: 4),
            // Two facts at one instant: the tie-break still gives one order.
            F.entry(.canceledOnLichess(challengeID: Self.id), at: 4),
        ]
        var reference = LichessBotChallengeLedger(loadStatus: .complete)
        reference.apply(contentsOf: entries)
        var permutationCount = 0
        for permutation in Self.permutations(of: entries) {
            var ledger = LichessBotChallengeLedger(loadStatus: .complete)
            ledger.apply(contentsOf: permutation)
            XCTAssertEqual(ledger, reference)
            permutationCount += 1
        }
        XCTAssertEqual(permutationCount, 720)
        let row = reference.row(challengeID: Self.id)
        XCTAssertEqual(row?.state, .accepted(gameStarted: true))
        XCTAssertEqual(row?.sender, .challengeQueue)
        XCTAssertEqual(row?.notes, [.withdrawalAttempted(.operatorCancel, .alreadyGone(message: nil)), .canceledOnLichess])
        XCTAssertEqual(row?.firstAt, F.at(0))
    }

    private static func permutations<Element>(of elements: [Element]) -> [[Element]] {
        guard let first = elements.first else { return [[]] }
        var result: [[Element]] = []
        for rest in permutations(of: Array(elements.dropFirst())) {
            for index in 0...rest.count {
                var permutation = rest
                permutation.insert(first, at: index)
                result.append(permutation)
            }
        }
        return result
    }

    // MARK: - Load status and housekeeping

    func testLoadStatusFollowsTheFilesLeftOut() {
        let complete = LichessBotChallengeLedger(contents: LichessBotChallengeLogContents(entries: [F.entry(F.created(), at: 0)]))
        XCTAssertEqual(complete.loadStatus, .complete)
        XCTAssertEqual(complete.rowsByKey.count, 1)
        let leftOut = [LichessBotChallengeLogContents.LeftOutFile(name: "challenges-20261004.jsonl", reason: "line 2 is not a valid entry")]
        let partial = LichessBotChallengeLedger(contents: LichessBotChallengeLogContents(entries: [], filesRead: [], filesLeftOut: leftOut))
        XCTAssertEqual(partial.loadStatus, .partial(filesLeftOut: leftOut))
        XCTAssertEqual(LichessBotChallengeLedger(loadStatus: .failed(reason: "unreadable")).rowsByKey.count, 0)
    }

    func testRepairsBelongToNoRow() {
        let ledger = self.ledger([.unterminatedLineCut(byteCount: 3, base64: "e30="), F.created()])
        XCTAssertEqual(ledger.rowsByKey.count, 1)
        XCTAssertEqual(ledger.unterminatedLineCuts.map(\.event), [.unterminatedLineCut(byteCount: 3, base64: "e30=")])
    }

    func testRowsAreListedOldestFirst() {
        let ledger = self.ledger([F.created("second"), F.received("first")])
        XCTAssertEqual(ledger.rows.map(\.key), [.challenge(id: "second"), .challenge(id: "first")], "by first fact time, not id")
    }
}
