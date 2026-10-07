//
//  LichessBotChallengeLogRowTests.swift
//  DrewsChessMachineTests
//
//  The Challenge Log window's rows (challenge-log plan §3.9): built from the
//  live ledger plus the history rebuilt from the protocol log, live rows
//  winning on an id both hold; every column's value, including each "not
//  recorded"; the filters; the footer's counts.
//

import XCTest
@testable import DrewsChessMachine

final class LichessBotChallengeLogRowTests: XCTestCase {

    private typealias F = LichessBotChallengeLogFixtures
    private typealias R = LichessBotChallengeLogRowFixtures

    private func row(_ rows: [LichessBotChallengeLogRow], _ key: LichessBotChallengeLogRow.Key,
                     file: StaticString = #filePath, line: UInt = #line) throws -> LichessBotChallengeLogRow {
        let matching = rows.filter { $0.key == key }
        XCTAssertEqual(matching.count, 1, "rows keyed \(key)", file: file, line: line)
        return try XCTUnwrap(matching.first, file: file, line: line)
    }

    private func liveRows() -> [LichessBotChallengeLogRow] {
        LichessBotChallengeLogRow.rows(ledger: R.ledger(), reconstruction: nil, pendingChallengeIDs: [R.pendingID])
    }

    func testNoSourcesGiveNoRows() {
        XCTAssertEqual(LichessBotChallengeLogRow.rows(ledger: nil, reconstruction: nil, pendingChallengeIDs: []), [])
        XCTAssertEqual(LichessBotChallengeLogRow.rows(ledger: LichessBotChallengeLedger(loadStatus: .complete), reconstruction: nil, pendingChallengeIDs: []), [])
    }

    func testAcceptedOutgoingRowCarriesEveryColumn() throws {
        let accepted = try row(liveRows(), .challenge(id: R.sharedID))
        XCTAssertEqual(accepted.at, F.at(1000))
        XCTAssertEqual(accepted.direction, .outgoing)
        XCTAssertEqual(accepted.opponent, .player(.init(id: "maia1", name: "maia1", title: "BOT", rating: 1400)))
        XCTAssertEqual(accepted.terms, .init(rated: true, limitSeconds: 180, incrementSeconds: 2, daysPerTurn: nil, color: .init(.random)))
        XCTAssertEqual(accepted.initiative, .sentBy(F.automaticPass))
        XCTAssertEqual(accepted.state, .challenge(.accepted(gameStarted: true), isPending: false))
        XCTAssertEqual(accepted.gameID, R.sharedID)
        XCTAssertEqual(accepted.credits, .spent(1))
        XCTAssertEqual(accepted.source, .live)
        XCTAssertEqual(LichessBotChallengeLogStateKind(accepted.state), .accepted)
        XCTAssertEqual(LichessBotChallengeLogSenderKind(accepted.initiative), .matchmaking)
    }

    func testOpenRowIsWaitingOnlyWhilePending() throws {
        let pending = try row(liveRows(), .challenge(id: R.pendingID))
        XCTAssertEqual(pending.state, .challenge(.open, isPending: true))
        XCTAssertEqual(LichessBotChallengeLogStateKind(pending.state), .waiting)
        XCTAssertNil(pending.gameID)

        let notPending = try row(LichessBotChallengeLogRow.rows(ledger: R.ledger(), reconstruction: nil, pendingChallengeIDs: []),
                                 .challenge(id: R.pendingID))
        XCTAssertEqual(notPending.state, .challenge(.open, isPending: false))
        XCTAssertEqual(LichessBotChallengeLogStateKind(notPending.state), .noAnswerRecorded)
    }

    func testNotCreatedAttemptNamesTheOpponentByIDAndTheTermsFromTheRequest() throws {
        let attempt = try row(liveRows(), .liveAttempt(F.attemptID(1)))
        XCTAssertEqual(attempt.direction, .outgoing)
        XCTAssertEqual(attempt.opponent, .player(.init(id: "offlinebot", name: "offlinebot", title: nil, rating: nil)))
        XCTAssertEqual(attempt.terms, .init(F.request))
        XCTAssertEqual(attempt.initiative, .sentBy(.challengeQueue))
        XCTAssertEqual(attempt.state, .challenge(.notCreated(.opponentOffline), isPending: false))
        XCTAssertEqual(attempt.credits, .spent(0))
        XCTAssertEqual(LichessBotChallengeLogStateKind(attempt.state), .notCreated)
    }

    func testIncomingRowShowsTheChallengerAndDCMsDecisionAndCostsNothing() throws {
        let incoming = try row(liveRows(), .challenge(id: "InCo5678"))
        XCTAssertEqual(incoming.direction, .incoming)
        XCTAssertEqual(incoming.opponent, .player(.init(id: "maia1", name: "maia1", title: "BOT", rating: 1400)))
        XCTAssertEqual(incoming.initiative, .decided(.decline(reason: .tooFast, rule: "too fast")))
        XCTAssertEqual(incoming.state, .challenge(.incomingDecided(.decline(reason: .tooFast, rule: "too fast")), isPending: false))
        XCTAssertEqual(incoming.credits, .notApplicable)
        XCTAssertEqual(LichessBotChallengeLogSenderKind(incoming.initiative), .incoming)
        XCTAssertEqual(LichessBotChallengeLogStateKind(incoming.state), .decidedByDCM)
    }

    func testARowWithNoDirectionSaysSoInEveryColumn() throws {
        let gone = try row(liveRows(), .challenge(id: "Gone0001"))
        XCTAssertNil(gone.direction)
        XCTAssertEqual(gone.opponent, .notRecorded)
        XCTAssertNil(gone.terms)
        XCTAssertEqual(gone.initiative, .directionNotRecorded)
        XCTAssertEqual(gone.state, .challenge(.canceledOnLichessDirectionNotRecorded, isPending: false))
        XCTAssertEqual(gone.credits, .notRecorded)
        XCTAssertEqual(LichessBotChallengeLogSenderKind(gone.initiative), .notRecorded)
        XCTAssertEqual(LichessBotChallengeLogStateKind(gone.state), .canceled)
    }

    func testAnUnattributedEchoHasNoSenderAndNoRecordedCost() throws {
        let echo = try row(liveRows(), .challenge(id: "Echo0001"))
        XCTAssertEqual(echo.direction, .outgoing)
        XCTAssertEqual(echo.initiative, .senderNotRecorded)
        XCTAssertEqual(echo.credits, .notRecorded)
        XCTAssertEqual(echo.state, .challenge(.open, isPending: false))
    }

    func testWithdrawnRowKeepsReasonAndResult() throws {
        let withdrawn = try row(liveRows(), .challenge(id: "Wdr00001"))
        XCTAssertEqual(withdrawn.initiative, .sentBy(.casualResendOffer))
        XCTAssertEqual(withdrawn.state, .challenge(.withdrawn(.operatorCancel, .alreadyGone(message: "not found")), isPending: false))
        XCTAssertEqual(LichessBotChallengeLogStateKind(withdrawn.state), .withdrawn)
        XCTAssertEqual(LichessBotChallengeLogSenderKind(withdrawn.initiative), .casualResendOffer)
    }

    func testDeclinedRowKeepsLichesssReasonKey() throws {
        let declined = try row(liveRows(), .challenge(id: "Decl0001"))
        XCTAssertEqual(declined.state, .challenge(.declined(.unrecognized("nobot")), isPending: false))
        XCTAssertEqual(LichessBotChallengeLogStateKind(declined.state), .declined)
        XCTAssertNil(declined.gameID)
    }

    func testReconstructedRowsCarryTheirConfidenceAndNoRecordedCredits() throws {
        let rows = LichessBotChallengeLogRow.rows(ledger: nil, reconstruction: try R.reconstruction(), pendingChallengeIDs: [])
        XCTAssertTrue(rows.allSatisfy(\.isReconstructed))

        let matchmaking = try row(rows, .challenge(id: "RecMm001"))
        XCTAssertEqual(matchmaking.direction, .outgoing)
        XCTAssertEqual(matchmaking.opponent, .player(.init(id: "edwardkillick", name: "EdwardKillick", title: "BOT", rating: 1500)))
        XCTAssertEqual(matchmaking.initiative, .reconstructedSender(.attributed(sender: .matchmaking, confidence: .paired)))
        XCTAssertEqual(matchmaking.state, .challenge(.accepted(gameStarted: true), isPending: false))
        XCTAssertEqual(matchmaking.gameID, "RecMm001")
        XCTAssertEqual(matchmaking.credits, .notRecorded)
        XCTAssertEqual(matchmaking.source, .reconstructed(.paired))

        let byOperator = try row(rows, .challenge(id: "RecOp001"))
        XCTAssertEqual(byOperator.opponent, .player(.init(id: "operator1", name: "Operator1", title: nil, rating: nil)))
        XCTAssertEqual(byOperator.terms, .init(LichessBotOutgoingChallenge(rated: true, clockLimitSeconds: 180, clockIncrementSeconds: 2, color: .random)))
        XCTAssertEqual(byOperator.initiative, .reconstructedSender(.attributed(sender: .byOperator, confidence: .inferredFromAbsence)))
        XCTAssertEqual(byOperator.source, .reconstructed(.inferredFromAbsence))
        XCTAssertEqual(LichessBotChallengeLogSenderKind(byOperator.initiative), .byOperator)

        let incoming = try row(rows, .challenge(id: "RecIn001"))
        XCTAssertEqual(incoming.direction, .incoming)
        XCTAssertEqual(incoming.initiative, .decided(.accept))
        XCTAssertEqual(incoming.credits, .notApplicable)
        XCTAssertEqual(incoming.source, .reconstructed(.certain))

        let attempts = rows.filter { if case .reconstructedAttempt = $0.key { return true }; return false }
        XCTAssertEqual(attempts.count, 1)
        let attempt = try XCTUnwrap(attempts.first)
        XCTAssertEqual(attempt.state, .reconstructedNotCreated(.opponentOffline))
        XCTAssertEqual(LichessBotChallengeLogStateKind(attempt.state), .notCreated)

        let canceled = try row(rows, .challenge(id: "RecCx001"))
        XCTAssertEqual(canceled.state, .challenge(.canceledOnLichessWithoutRecordedWithdrawal, isPending: false))
        XCTAssertEqual(LichessBotChallengeLogStateKind(canceled.state), .withdrawn)
    }

    func testLiveRowsWinOnAnIDBothHoldAndAttemptsNeverCollide() throws {
        let reconstruction = try R.reconstruction()
        let ledger = R.ledger()
        let rows = LichessBotChallengeLogRow.rows(ledger: ledger, reconstruction: reconstruction, pendingChallengeIDs: [R.pendingID])

        let shared = try row(rows, .challenge(id: R.sharedID))
        XCTAssertEqual(shared.source, .live)
        XCTAssertEqual(shared.at, F.at(1000))
        // Every live row, plus every rebuilt row but the shared one; both
        // offline attempts (one per source) are kept.
        XCTAssertEqual(rows.count, ledger.rows.count + reconstruction.rows.count - 1)
        XCTAssertEqual(rows.filter { if case .liveAttempt = $0.key { return true }; return false }.count, 1)
        XCTAssertEqual(rows.filter { if case .reconstructedAttempt = $0.key { return true }; return false }.count, 1)
        XCTAssertEqual(Set(rows.map(\.key)).count, rows.count)
    }

    func testRowsAreNewestFirst() throws {
        let rows = LichessBotChallengeLogRow.rows(ledger: R.ledger(), reconstruction: try R.reconstruction(), pendingChallengeIDs: [])
        XCTAssertEqual(rows.map(\.at), rows.map(\.at).sorted(by: >))
        XCTAssertEqual(rows.first?.key, .challenge(id: "Wdr00001"))
    }

    // MARK: Filter

    private func mixedRows() throws -> [LichessBotChallengeLogRow] {
        LichessBotChallengeLogRow.rows(ledger: R.ledger(), reconstruction: try R.reconstruction(), pendingChallengeIDs: [R.pendingID])
    }

    func testTheDefaultFilterShowsEverything() throws {
        let rows = try mixedRows()
        XCTAssertEqual(LichessBotChallengeLogFilter().apply(to: rows, now: F.at(2000)), rows)
    }

    func testDateRangeIsMeasuredBackFromNow() throws {
        let rows = try mixedRows()
        var filter = LichessBotChallengeLogFilter()
        filter.dateRange = .last24Hours
        // Every fixture row lies within the day before F.at(2000)…
        XCTAssertEqual(filter.apply(to: rows, now: F.at(2000)).count, rows.count)
        // …and a day after the live rows, only rows at or after the cut
        // remain (none here: the newest live row is at F.at(1060)).
        XCTAssertEqual(filter.apply(to: rows, now: F.at(1060 + 24 * 3600 + 1)).count, 0)
        XCTAssertEqual(filter.apply(to: rows, now: F.at(1060 + 24 * 3600)).map(\.key), [.challenge(id: "Wdr00001")])
        filter.dateRange = .all
        XCTAssertEqual(filter.apply(to: rows, now: F.at(10_000_000)).count, rows.count)
        XCTAssertEqual(LichessBotChallengeLogFilter.DateRange.last7Days.lookback, 7 * 24 * 3600)
        XCTAssertEqual(LichessBotChallengeLogFilter.DateRange.last30Days.lookback, 30 * 24 * 3600)
    }

    func testDirectionStateSenderAndReconstructedFilters() throws {
        let rows = try mixedRows()
        var filter = LichessBotChallengeLogFilter()
        filter.direction = .incoming
        XCTAssertEqual(Set(filter.apply(to: rows, now: F.at(2000)).map(\.key)), [.challenge(id: "InCo5678"), .challenge(id: "RecIn001")])
        filter.direction = .outgoing
        XCTAssertTrue(filter.apply(to: rows, now: F.at(2000)).allSatisfy { $0.direction == .outgoing })
        XCTAssertFalse(filter.apply(to: rows, now: F.at(2000)).contains { $0.key == .challenge(id: "Gone0001") })

        filter = LichessBotChallengeLogFilter()
        filter.state = .only(.waiting)
        XCTAssertEqual(filter.apply(to: rows, now: F.at(2000)).map(\.key), [.challenge(id: R.pendingID)])
        filter.state = .only(.notCreated)
        XCTAssertEqual(filter.apply(to: rows, now: F.at(2000)).count, 2)

        filter = LichessBotChallengeLogFilter()
        filter.sender = .only(.byOperator)
        XCTAssertEqual(Set(filter.apply(to: rows, now: F.at(2000)).map(\.key)).count, 3)
        XCTAssertTrue(filter.apply(to: rows, now: F.at(2000)).allSatisfy(\.isReconstructed))

        filter = LichessBotChallengeLogFilter()
        filter.includeReconstructed = false
        XCTAssertEqual(filter.apply(to: rows, now: F.at(2000)).count, R.ledger().rows.count)
        XCTAssertFalse(filter.apply(to: rows, now: F.at(2000)).contains(where: \.isReconstructed))
    }

    func testOpponentSearchIsCaseInsensitiveOnIDAndNameAndNeverMatchesAnUnnamedRow() throws {
        let rows = try mixedRows()
        var filter = LichessBotChallengeLogFilter()
        filter.opponentSearch = "  EDWARD "
        XCTAssertEqual(filter.apply(to: rows, now: F.at(2000)).map(\.key), [.challenge(id: "RecMm001")])
        filter.opponentSearch = "operator1"
        XCTAssertEqual(filter.apply(to: rows, now: F.at(2000)).map(\.key), [.challenge(id: "RecOp001")])
        filter.opponentSearch = "zzz"
        XCTAssertEqual(filter.apply(to: rows, now: F.at(2000)), [])
        filter.opponentSearch = ""
        XCTAssertEqual(filter.apply(to: rows, now: F.at(2000)).count, rows.count)
    }

    func testCountsTallyTheShownRows() throws {
        let rows = try mixedRows()
        let counts = LichessBotChallengeLogCounts(rows)
        XCTAssertEqual(counts.shown, rows.count)
        XCTAssertEqual(counts.outgoing + counts.incoming + counts.directionNotRecorded, rows.count)
        XCTAssertEqual(counts.incoming, 2)
        XCTAssertEqual(counts.directionNotRecorded, 1)
        XCTAssertEqual(counts.live, R.ledger().rows.count)
        XCTAssertEqual(counts.reconstructed, rows.count - counts.live)
        // The four live created rows (shared, pending, declined, withdrawn)
        // cost 1 each and the offline attempt 0; an echo and rebuilt rows
        // record none.
        XCTAssertEqual(counts.creditsSpent, 4)
        XCTAssertEqual(LichessBotChallengeLogCounts([]), LichessBotChallengeLogCounts())
    }

    func testSenderKindsCoverEveryInitiative() {
        let senders: [LichessBotChallengeSender] = [.challengeSheet, .casualResendOffer, .challengeQueue, F.automaticPass,
                                                    .matchmaking(trigger: .fillOpenSlots, fillMode: .onlyWhenIdle), .matchmakingCasualResend]
        XCTAssertEqual(senders.map { LichessBotChallengeLogSenderKind(.sentBy($0)) },
                       [.challengeSheet, .casualResendOffer, .challengeQueue, .matchmaking, .matchmaking, .matchmakingCasualResend])
        XCTAssertEqual(LichessBotReconstructedSender.allCases.map { LichessBotChallengeLogSenderKind(.reconstructedSender(.attributed(sender: $0, confidence: .paired))) },
                       [.byOperator, .challengeQueue, .matchmaking, .matchmakingCasualResend])
        XCTAssertEqual(LichessBotChallengeLogSenderKind(.reconstructedSender(.ambiguousCompanion)), .notRecorded)
        XCTAssertEqual(LichessBotChallengeLogSenderKind(.reconstructedSender(.noSendLine)), .notRecorded)
        XCTAssertEqual(LichessBotChallengeLogSenderKind(.senderNotRecorded), .notRecorded)
        XCTAssertEqual(LichessBotChallengeLogSenderKind(.noDecisionRecorded), .incoming)
    }
}
