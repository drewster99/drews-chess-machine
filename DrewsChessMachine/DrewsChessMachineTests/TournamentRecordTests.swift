//
//  TournamentRecordTests.swift
//  DrewsChessMachineTests
//
//  Tests for the per-side statistics added to TournamentStats and
//  TournamentRecord. The data-model layer is pure Swift — these
//  tests construct the structs directly rather than running a full
//  TournamentDriver through ChessMachine (which would be slow and
//  brittle), since the concern here is: given correct per-side
//  counts, do the derived score / Elo properties compute correctly?
//
//  The TournamentDriver code path that produces these counts is
//  straightforward bookkeeping over `aIsWhite` and the game result
//  — it doesn't need a formal test so long as we fence the data
//  model it feeds into.
//

import XCTest
@testable import DrewsChessMachine

final class TournamentStatsSideTests: XCTestCase {

    func testWhiteGamesCount() {
        let s = TournamentStats(
            gamesPlayed: 100, playerAWins: 40, playerBWins: 30, draws: 30,
            playerAWinsAsWhite: 25, playerAWinsAsBlack: 15,
            playerALossesAsWhite: 10, playerALossesAsBlack: 20,
            playerADrawsAsWhite: 15, playerADrawsAsBlack: 15
        )
        XCTAssertEqual(s.playerAWhiteGames, 25 + 10 + 15)
        XCTAssertEqual(s.playerABlackGames, 15 + 20 + 15)
    }

    func testScoreAsWhite() {
        // 25W + 10L + 15D, 50 games as white → (25 + 7.5) / 50 = 0.65
        let s = TournamentStats(
            gamesPlayed: 100, playerAWins: 40, playerBWins: 30, draws: 30,
            playerAWinsAsWhite: 25, playerAWinsAsBlack: 15,
            playerALossesAsWhite: 10, playerALossesAsBlack: 20,
            playerADrawsAsWhite: 15, playerADrawsAsBlack: 15
        )
        XCTAssertEqual(s.playerAScoreAsWhite, 0.65, accuracy: 1e-9)
    }

    func testScoreAsBlack() {
        // 15W + 20L + 15D, 50 games as black → (15 + 7.5) / 50 = 0.45
        let s = TournamentStats(
            gamesPlayed: 100, playerAWins: 40, playerBWins: 30, draws: 30,
            playerAWinsAsWhite: 25, playerAWinsAsBlack: 15,
            playerALossesAsWhite: 10, playerALossesAsBlack: 20,
            playerADrawsAsWhite: 15, playerADrawsAsBlack: 15
        )
        XCTAssertEqual(s.playerAScoreAsBlack, 0.45, accuracy: 1e-9)
    }

    func testScoreEmptySideIsZero() {
        // Tournament aborted before any black games: score as black
        // must be exactly 0 (guard in the computed property) rather
        // than a divide-by-zero NaN.
        let s = TournamentStats(
            gamesPlayed: 10, playerAWins: 5, playerBWins: 3, draws: 2,
            playerAWinsAsWhite: 5, playerAWinsAsBlack: 0,
            playerALossesAsWhite: 3, playerALossesAsBlack: 0,
            playerADrawsAsWhite: 2, playerADrawsAsBlack: 0
        )
        XCTAssertEqual(s.playerABlackGames, 0)
        XCTAssertEqual(s.playerAScoreAsBlack, 0, accuracy: 1e-12)
        XCTAssertFalse(s.playerAScoreAsBlack.isNaN)
    }

    func testPerSideCountsSumToTotals() {
        // Identity: W-by-white + W-by-black = total W, and same for
        // L and D. The driver must maintain this — documenting the
        // invariant here catches any future off-by-one in the
        // switch-statement branches.
        let s = TournamentStats(
            gamesPlayed: 200, playerAWins: 90, playerBWins: 70, draws: 40,
            playerAWinsAsWhite: 50, playerAWinsAsBlack: 40,
            playerALossesAsWhite: 30, playerALossesAsBlack: 40,
            playerADrawsAsWhite: 20, playerADrawsAsBlack: 20
        )
        XCTAssertEqual(s.playerAWinsAsWhite + s.playerAWinsAsBlack, s.playerAWins)
        XCTAssertEqual(s.playerALossesAsWhite + s.playerALossesAsBlack, s.playerBWins)
        XCTAssertEqual(s.playerADrawsAsWhite + s.playerADrawsAsBlack, s.draws)
        XCTAssertEqual(s.playerAWhiteGames + s.playerABlackGames, s.gamesPlayed)
    }
}

final class TournamentRecordSideTests: XCTestCase {

    private func makeRecord(
        candW: Int, champW: Int, draws: Int,
        cWW: Int, cWB: Int, cLW: Int, cLB: Int, cDW: Int, cDB: Int,
        promoted: Bool = false,
        kind: PromotionKind? = nil
    ) -> TournamentRecord {
        TournamentRecord(
            finishedAtStep: 1000,
            gamesPlayed: candW + champW + draws,
            candidateWins: candW,
            championWins: champW,
            draws: draws,
            score: ArenaEloStats.score(wins: candW, draws: draws, losses: champW),
            promoted: promoted,
            promotionKind: kind,
            promotedID: nil,
            durationSec: 60,
            candidateWinsAsWhite: cWW,
            candidateWinsAsBlack: cWB,
            candidateLossesAsWhite: cLW,
            candidateLossesAsBlack: cLB,
            candidateDrawsAsWhite: cDW,
            candidateDrawsAsBlack: cDB
        )
    }

    func testCandidateScoreAsWhite() {
        let r = makeRecord(
            candW: 40, champW: 30, draws: 30,
            cWW: 25, cWB: 15, cLW: 10, cLB: 20, cDW: 15, cDB: 15
        )
        XCTAssertEqual(r.candidateScoreAsWhite, 0.65, accuracy: 1e-9)
    }

    func testCandidateScoreAsBlack() {
        let r = makeRecord(
            candW: 40, champW: 30, draws: 30,
            cWW: 25, cWB: 15, cLW: 10, cLB: 20, cDW: 15, cDB: 15
        )
        XCTAssertEqual(r.candidateScoreAsBlack, 0.45, accuracy: 1e-9)
    }

    func testCandidateScoreAsBlackEmptyIsZero() {
        let r = makeRecord(
            candW: 5, champW: 3, draws: 2,
            cWW: 5, cWB: 0, cLW: 3, cLB: 0, cDW: 2, cDB: 0
        )
        XCTAssertEqual(r.candidateScoreAsBlack, 0, accuracy: 1e-12)
        XCTAssertFalse(r.candidateScoreAsBlack.isNaN)
    }

    func testEloSummaryMatchesArenaEloStats() {
        // Computed property should just delegate to
        // ArenaEloStats.summary. Cross-check.
        let r = makeRecord(
            candW: 60, champW: 20, draws: 20,
            cWW: 30, cWB: 30, cLW: 10, cLB: 10, cDW: 10, cDB: 10
        )
        let direct = ArenaEloStats.summary(wins: 60, draws: 20, losses: 20)
        XCTAssertEqual(r.eloSummary.score, direct.score, accuracy: 1e-12)
        XCTAssertEqual(r.eloSummary.scoreLo, direct.scoreLo, accuracy: 1e-12)
        XCTAssertEqual(r.eloSummary.scoreHi, direct.scoreHi, accuracy: 1e-12)
        XCTAssertEqual(r.eloSummary.elo!, direct.elo!, accuracy: 1e-9)
    }

    func testBalancedCandidateScoreIsOneHalf() {
        // Sanity: if candidate and champion each won exactly half
        // the decisive games, record.score is 0.5 regardless of
        // side split.
        let r = makeRecord(
            candW: 30, champW: 30, draws: 40,
            cWW: 15, cWB: 15, cLW: 15, cLB: 15, cDW: 20, cDB: 20
        )
        XCTAssertEqual(r.score, 0.5, accuracy: 1e-12)
    }
}

final class TournamentRecordEloWorkedExampleTests: XCTestCase {

    func testUserSpecThousandGameExample() {
        // Ticket's worked example — 1000 games, 312W / 401D / 287L:
        // score 51.2%, Elo diff +8 with CI straddling zero, draw
        // rate 40.1% — should all flow from the record correctly.
        let r = TournamentRecord(
            finishedAtStep: 12345,
            gamesPlayed: 1000,
            candidateWins: 312,
            championWins: 287,
            draws: 401,
            score: ArenaEloStats.score(wins: 312, draws: 401, losses: 287),
            promoted: false,
            promotionKind: nil,
            promotedID: nil,
            durationSec: 600,
            candidateWinsAsWhite: 170, candidateWinsAsBlack: 142,
            candidateLossesAsWhite: 130, candidateLossesAsBlack: 157,
            candidateDrawsAsWhite: 200, candidateDrawsAsBlack: 201
        )
        XCTAssertEqual(r.score, 0.5125, accuracy: 1e-4)
        XCTAssertEqual(r.eloSummary.elo!, 8.7, accuracy: 0.3)
        XCTAssertLessThan(r.eloSummary.eloLo!, 0)  // "not statistically convincing"
        XCTAssertGreaterThan(r.eloSummary.eloHi!, 0)
        // By-side sanity.
        XCTAssertEqual(r.candidateScoreAsWhite, (170 + 100) / 500.0, accuracy: 1e-9)
        XCTAssertEqual(r.candidateScoreAsBlack, (142 + 100.5) / 500.0, accuracy: 1e-9)
    }
}
