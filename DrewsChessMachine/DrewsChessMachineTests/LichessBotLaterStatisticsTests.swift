import XCTest
@testable import DrewsChessMachine

/// The later panes (`LICHESS_BOT_RECORD_STATS_PLAN.md` §11, L1–L6): move
/// choice, clock, game length, openings, opponents and bot health — their
/// facts from records and their statistics from rows.
final class LichessBotLaterStatisticsTests: XCTestCase {
    private typealias Fixtures = LichessBotStatsFixtures

    /// Wednesday 2026-10-07 12:00 UTC.
    private let now = Date(timeIntervalSince1970: 1_791_374_400)
    private let model = Fixtures.generationInfo(id: 1, modelID: "20261001-1-AAAA")

    private func later(_ rows: [LichessBotGameSummary], filter: LichessBotStatsFilter = .all) throws -> LichessBotLaterBreakdowns {
        try LichessBotRecordStatistics.compute(rows: rows, now: now, calendar: Fixtures.utcCalendar)[filter].byPeriod.allTime.later
    }

    // MARK: - Facts from records

    func testMoveChoiceAndClockFactsFromARecord() throws {
        // DCM (White) plays its policy's top move on every other decision;
        // decisions at plies 0 and 4 record no top moves; ply 8 is randomish.
        let record = try Fixtures.record(ourColor: .white, plies: 42, status: "resign", winner: "white") { ply in
            .decidedWith(makeDecision: { uci in
                let top: [LichessBotMoveCandidate]
                switch ply {
                case 0, 4: top = []
                default: top = [LichessBotMoveCandidate(uci: ply % 4 == 2 ? uci : "a2a3", probability: 0.5)]
                }
                return LichessBotMoveDecision(
                    uci: uci, san: uci, chosenProbability: 0.25, topMoves: top,
                    win: 0.5, draw: 0.2, loss: 0.3, temperature: 0.5, legalMoveCount: 20, randomish: ply == 8,
                    encodeMilliseconds: 0.1, inferenceMilliseconds: 2, sampleMilliseconds: 0.1
                )
            }, generation: self.model)
        }
        let moves = try XCTUnwrap(LichessBotGameFacts(record: record).moves)
        // 21 decisions; 19 with top moves; top played at plies 2, 6, 10, … 38 (10 of them).
        XCTAssertEqual(moves.choice, LichessBotMoveChoiceFacts(decisionsWithTopMoves: 19, topMoveChosen: 10, sumChosenProbability: 21 * 0.25, randomish: 1))
        // The fixture's server clocks are constant (White 170 s), so each
        // move after DCM's first used exactly the 2 s increment.
        let clock = try XCTUnwrap(moves.clock)
        XCTAssertEqual(clock.thinkTimeMoves, 20)
        XCTAssertEqual(clock.sumThinkMilliseconds, 40_000)
        XCTAssertEqual(clock.finalClockMilliseconds, 170_000)
        let facts = LichessBotGameFacts(record: record)
        XCTAssertEqual(facts.rejectedMoves, 0)
        XCTAssertEqual(facts.streamReconnects, 0)
        XCTAssertNil(facts.openingECO, "no export, no opening")
    }

    // MARK: - Statistics from rows

    func testMoveChoiceOverallAndByModel() throws {
        let rows = [
            try Fixtures.row(id: "a", at: now - 600, score: 1, facts: Fixtures.facts(ourMovesWithDecision: 10, generations: [Fixtures.generation("A", step: 1, moves: 10)], choice: LichessBotMoveChoiceFacts(decisionsWithTopMoves: 10, topMoveChosen: 9, sumChosenProbability: 6, randomish: 0))),
            try Fixtures.row(id: "b", at: now - 700, score: 0, facts: Fixtures.facts(ourMovesWithDecision: 20, generations: [Fixtures.generation("B", step: 1, moves: 20)], choice: LichessBotMoveChoiceFacts(decisionsWithTopMoves: 10, topMoveChosen: 5, sumChosenProbability: 8, randomish: 2))),
        ]
        let choice = try later(rows).moveChoice
        XCTAssertEqual(choice.overall.decisions, 30)
        XCTAssertEqual(try XCTUnwrap(choice.overall.topMoveShare), 14.0 / 20, accuracy: 1e-12)
        XCTAssertEqual(try XCTUnwrap(choice.overall.meanChosenProbability), 14.0 / 30, accuracy: 1e-12)
        XCTAssertEqual(choice.overall.randomish, 2)
        XCTAssertEqual(choice.byModel.map(\.modelID), ["B", "A"], "most decisions first")
        XCTAssertNil(LichessBotMoveChoiceLine().topMoveShare)
    }

    func testClockRowsPerSpeed() throws {
        let clock = { (moves: Int, sum: Double, final: Int?) in LichessBotClockFacts(thinkTimeMoves: moves, sumThinkMilliseconds: sum, finalClockMilliseconds: final) }
        let rows = [
            try Fixtures.row(id: "a", at: now - 600, score: 1, speed: "rapid", facts: Fixtures.facts(clock: clock(10, 50_000, 300_000))),
            try Fixtures.row(id: "b", at: now - 700, score: 0, speed: "rapid", status: "outoftime", facts: Fixtures.facts(clock: clock(30, 210_000, 0))),
            try Fixtures.row(id: "c", at: now - 800, score: 0, speed: "bullet", facts: Fixtures.facts(clock: clock(5, 4_000, nil))),
            try Fixtures.row(id: "d", at: now - 900, score: 1, speed: "correspondence", facts: Fixtures.facts(clock: nil)),
        ]
        let clockRows = try later(rows).clock
        XCTAssertEqual(clockRows.map(\.speed), ["bullet", "rapid"], "Lichess's order; no clock, no row")
        let rapid = clockRows[1]
        XCTAssertEqual(rapid.games, 2)
        XCTAssertEqual(try XCTUnwrap(rapid.meanThinkSeconds), 260.0 / 40, accuracy: 1e-12)
        XCTAssertEqual(try XCTUnwrap(rapid.meanFinalClockSeconds), 150, accuracy: 1e-12)
        XCTAssertEqual(rapid.flagged, 1)
        XCTAssertNil(clockRows[0].meanFinalClockSeconds)
    }

    func testGameLengthAndShortLosses() throws {
        let rows = [
            try Fixtures.row(id: "w1", at: now - 100, score: 1, plies: 40),
            try Fixtures.row(id: "w2", at: now - 200, score: 1, plies: 61),
            try Fixtures.row(id: "w3", at: now - 300, score: 1, plies: 50),
            try Fixtures.row(id: "l1", at: now - 400, score: 0, plies: 29),
            try Fixtures.row(id: "l2", at: now - 500, score: 0, plies: 30),
            try Fixtures.row(id: "l3", at: now - 50, score: 0, plies: 12),
            try Fixtures.row(id: "x", at: now - 60, score: nil, status: "aborted", plies: 1),
        ]
        let length = try later(rows).gameLength
        XCTAssertEqual(length.rows.map(\.label), ["Won", "Drawn", "Lost"])
        XCTAssertEqual(length.rows.map(\.games), [3, 0, 3])
        XCTAssertEqual(try XCTUnwrap(length.rows[0].meanPlies), 151.0 / 3, accuracy: 1e-12)
        XCTAssertEqual(length.rows[0].medianPlies, 50)
        XCTAssertNil(length.rows[1].meanPlies)
        XCTAssertEqual(length.rows[2].medianPlies, 29)
        XCTAssertEqual(length.shortLosses.map(\.gameID), ["l3", "l1"], "under 30 plies, most recent first")
        XCTAssertEqual(length.shortLossCount, 2)
    }

    func testOpeningsByFamilyAndColor() throws {
        let rows = [
            try Fixtures.row(id: "a", at: now - 100, score: 1, color: .white, facts: Fixtures.facts(openingECO: "B90", openingName: "Sicilian Defense: Najdorf Variation")),
            try Fixtures.row(id: "b", at: now - 200, score: 0, color: .black, facts: Fixtures.facts(openingECO: "B20", openingName: "Sicilian Defense")),
            try Fixtures.row(id: "c", at: now - 300, score: 0.5, color: .white, facts: Fixtures.facts(openingECO: "C50", openingName: "Italian Game: Giuoco Piano")),
            try Fixtures.row(id: "d", at: now - 400, score: 1, color: .white, facts: Fixtures.facts()),
            try Fixtures.row(id: "e", at: now - 500, score: 1),
        ]
        let openings = try later(rows).openings
        XCTAssertEqual(openings.rows.map(\.family), ["Sicilian Defense", "Italian Game"])
        XCTAssertEqual(openings.rows[0].ecoRange, "B20–B90")
        XCTAssertEqual(openings.rows[0].asWhite, LichessBotResultTally(wins: 1, draws: 0, losses: 0, unscored: 0))
        XCTAssertEqual(openings.rows[0].asBlack, LichessBotResultTally(wins: 0, draws: 0, losses: 1, unscored: 0))
        XCTAssertEqual(openings.rows[1].ecoRange, "C50")
        XCTAssertEqual(openings.gamesWithoutOpening, 2)
        XCTAssertEqual(LichessBotOpeningRow.family(of: "Queen's Gambit Declined: Exchange Variation, Positional Variation"), "Queen's Gambit Declined")
        XCTAssertEqual(LichessBotOpeningRow.family(of: "King's Pawn Game"), "King's Pawn Game")
    }

    func testOpponentsStreaksAndBestWin() throws {
        let rows = [
            // Oldest first: W W L L L W D W W (the last two are the current run).
            try Fixtures.row(id: "g1", at: now - 900, score: 1, opponentID: "alice", opponentName: "Alice", opponentRating: 1700),
            try Fixtures.row(id: "g2", at: now - 800, score: 1, opponentID: "bob", opponentName: "Bob", opponentRating: 1900),
            try Fixtures.row(id: "g3", at: now - 700, score: 0, opponentID: "alice", opponentName: "Alice", opponentRating: 1700),
            try Fixtures.row(id: "g4", at: now - 600, score: 0, opponentID: "alice", opponentName: "Alice", opponentRating: 1700),
            try Fixtures.row(id: "g5", at: now - 500, score: 0, opponentID: "carol", opponentName: "Carol", opponentRating: 2000),
            try Fixtures.row(id: "g6", at: now - 400, score: 1, opponentID: "dave", opponentName: "Dave", opponentRating: 1900),
            try Fixtures.row(id: "g7", at: now - 300, score: 0.5, kind: .lichessAI, opponentID: nil, opponentName: nil, opponentRating: nil, status: "draw"),
            try Fixtures.row(id: "g8", at: now - 200, score: 1, opponentID: "alice", opponentName: "Alice", opponentRating: 1700),
            try Fixtures.row(id: "g9", at: now - 100, score: 1, opponentID: "bob", opponentName: "Bob", opponentRating: 1500),
            try Fixtures.row(id: "ab", at: now - 50, score: nil, opponentID: "erin", opponentName: "Erin", status: "aborted"),
        ]
        let opponents = try later(rows).opponents
        XCTAssertEqual(opponents.mostPlayed.first?.name, "Alice")
        XCTAssertEqual(opponents.mostPlayed.first?.tally, LichessBotResultTally(wins: 2, draws: 0, losses: 2, unscored: 0))
        XCTAssertEqual(opponents.opponentCount, 5, "Lichess AI has no account and is not an opponent row")
        XCTAssertEqual(opponents.currentStreak, LichessBotStreak(ourScore: 1, length: 2))
        XCTAssertEqual(opponents.longestWinStreak, 2)
        XCTAssertEqual(opponents.longestLossStreak, 3)
        // Bob (1900, g2) and Dave (1900, g6) tie: the more recent win.
        XCTAssertEqual(opponents.highestRatedWin?.gameID, "g6")
        XCTAssertEqual(opponents.highestRatedWin?.rating, 1900)
    }

    func testHealthCountsEveryGame() throws {
        let rows = [
            try Fixtures.row(id: "a", at: now - 100, score: 1, facts: Fixtures.facts(rejectedMoves: 2, streamReconnects: 1)),
            try Fixtures.row(id: "b", at: now - 200, score: nil, status: "aborted", facts: Fixtures.facts(rejectedMoves: 0, streamReconnects: 3)),
            try Fixtures.row(id: "c", at: now - 300, score: 0),
        ]
        let health = try later(rows).health
        XCTAssertEqual(health.games, 3)
        XCTAssertEqual(health.rejectedMoves, 2)
        XCTAssertEqual(health.streamReconnects, 4)
        XCTAssertEqual(health.rowsWithoutFacts, 1)
        XCTAssertEqual(health.anomalies, 0)
        XCTAssertEqual(health.reconciliationCorrected, 0)
    }

    func testPanesAndTheirText() {
        XCTAssertEqual(LichessBotRecordPane.firstPass + LichessBotRecordPane.later, LichessBotRecordPane.allCases, "every pane in the picker, once")
        XCTAssertEqual(LichessBotStatsFormat.seconds(6.54), "6.5 s")
        XCTAssertEqual(LichessBotStatsFormat.seconds(-1.25), "\u{2212}1.2 s")
        XCTAssertEqual(LichessBotStatsFormat.seconds(nil), "–")
        XCTAssertEqual(LichessBotStatsFormat.streak(LichessBotStreak(ourScore: 1, length: 3)), "3 wins")
        XCTAssertEqual(LichessBotStatsFormat.streak(LichessBotStreak(ourScore: 0, length: 1)), "1 loss")
        XCTAssertEqual(LichessBotStatsFormat.streak(LichessBotStreak(ourScore: 0.5, length: 2)), "2 draws")
        XCTAssertEqual(LichessBotStatsFormat.streak(nil), "none")
    }

    func testNoGamesGivesEmptyLaterPanes() throws {
        let empty = try later([])
        XCTAssertEqual(empty.moveChoice.overall, LichessBotMoveChoiceLine())
        XCTAssertEqual(empty.clock, [])
        XCTAssertEqual(empty.gameLength.rows.map(\.games), [0, 0, 0])
        XCTAssertEqual(empty.openings.rows, [])
        XCTAssertNil(empty.opponents.currentStreak)
        XCTAssertNil(empty.opponents.highestRatedWin)
        XCTAssertEqual(empty.health, LichessBotHealthStatistics())
    }
}
