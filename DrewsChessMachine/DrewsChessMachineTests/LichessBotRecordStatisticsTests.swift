import XCTest
@testable import DrewsChessMachine

/// The Record card's statistics from hand-built index rows
/// (`LICHESS_BOT_RECORD_STATS_PLAN.md` §3, §4.4).
final class LichessBotRecordStatisticsTests: XCTestCase {
    private typealias Fixtures = LichessBotStatsFixtures

    /// Wednesday 2026-10-07 12:00 UTC.
    private let now = Date(timeIntervalSince1970: 1_791_374_400)
    private let hour: TimeInterval = 3600

    private func compute(_ rows: [LichessBotGameSummary]) throws -> LichessBotRecordStatistics {
        try LichessBotRecordStatistics.compute(rows: rows, now: now, calendar: Fixtures.utcCalendar)
    }

    // MARK: - Period table

    func testPeriodTableColumns() throws {
        let rows = [
            // Last hour: a win vs 1600 (+8) and a loss vs 1400 (−9).
            try Fixtures.row(id: "a", at: now - 0.2 * hour, score: 1, opponentRating: 1600, ourRatingDiff: 8),
            try Fixtures.row(id: "b", at: now - 0.5 * hour, score: 0, color: .black, kind: .human, opponentRating: 1400, ourRatingDiff: -9),
            // Today: a draw vs Lichess AI (no rating), casual.
            try Fixtures.row(id: "c", at: now - 5 * hour, score: 0.5, rated: false, kind: .lichessAI, opponentID: nil, opponentName: nil, opponentRating: nil, status: "draw"),
            // This week (Monday): a rated win without a recorded diff.
            try Fixtures.row(id: "d", at: now - 40 * hour, score: 1, opponentRating: 1500, ourRatingDiff: nil),
            // Aborted today: never counted.
            try Fixtures.row(id: "e", at: now - 2 * hour, score: nil, status: "aborted"),
        ]
        let stats = try compute(rows)[.all]
        let lastHour = stats.periodRows.lastHour
        XCTAssertEqual(lastHour.record.all.scored, 2)
        XCTAssertEqual(lastHour.record.all.score, 0.5)
        XCTAssertEqual(lastHour.ratedOpponentGames, 2)
        XCTAssertEqual(lastHour.opponentAverage, 1500)
        // 1 of 2 against 1400 and 1600: the mean, by symmetry.
        guard case .estimate(let perf) = lastHour.performance else { return XCTFail("expected an estimate") }
        XCTAssertEqual(perf, 1500, accuracy: LichessBotEloMath.solverTolerance)
        XCTAssertEqual(lastHour.ratingChange, LichessBotRatingChange(ratedGames: 2, gamesWithChange: 2, total: -1))
        XCTAssertEqual(LichessBotStatsFormat.ratingChange(lastHour.ratingChange), "\u{2212}1")

        let today = stats.periodRows.today
        // Games = scored only (OD-2): the aborted game is not in it.
        XCTAssertEqual(today.record.all.scored, 3)
        XCTAssertEqual(today.record.all.unscored, 1)
        // Lichess AI: in W–D–L and splits, not in Perf or Opp avg.
        XCTAssertEqual(today.record.versusLichessAI.draws, 1)
        XCTAssertEqual(today.ratedOpponentGames, 2)
        XCTAssertEqual(today.ratingChange.ratedGames, 2, "casual games never count toward Rating ±")

        let week = stats.periodRows.thisWeek
        XCTAssertEqual(week.ratingChange, LichessBotRatingChange(ratedGames: 3, gamesWithChange: 2, total: -1))
        XCTAssertEqual(LichessBotStatsFormat.ratingChange(week.ratingChange), "\u{2212}1*")
        XCTAssertEqual(LichessBotStatsFormat.ratingChangeHelp(week.ratingChange), "2 of 3 rated games have a rating change (Lichess' change at the time of the game)")
        XCTAssertEqual(stats.notCounted, 1)
        XCTAssertEqual(stats.ratedWithoutRatingChange, 1)
        XCTAssertEqual(stats.rowsWithoutMoveData, 4, "hand-built rows without facts")
        XCTAssertEqual(stats.periodRows.allTime.record.all.scored, 4)
    }

    func testRatingChangeIsDashStarWhenNoRatedGameHasADiff() throws {
        let stats = try compute([
            try Fixtures.row(id: "a", at: now - hour, score: 1),
            try Fixtures.row(id: "b", at: now - hour, score: 0),
        ])[.all]
        let change = stats.periodRows.today.ratingChange
        XCTAssertEqual(change, LichessBotRatingChange(ratedGames: 2, gamesWithChange: 0, total: 0))
        XCTAssertEqual(LichessBotStatsFormat.ratingChange(change), "–*")
        let casual = try compute([try Fixtures.row(id: "c", at: now - hour, score: 1, rated: false)])[.all]
        XCTAssertEqual(LichessBotStatsFormat.ratingChange(casual.periodRows.today.ratingChange), "–")
    }

    func testEveryFilterEqualsComputeOverThePrefilteredRows() throws {
        var rows: [LichessBotGameSummary] = []
        for index in 0..<40 {
            rows.append(try Fixtures.row(
                id: "g\(index)",
                at: now - Double(index * 7) * hour,
                score: [1, 0.5, 0, nil][index % 4],
                speed: ["blitz", "rapid", "bullet"][index % 3],
                rated: index % 5 != 0,
                color: index % 2 == 0 ? .white : .black,
                opponentRating: 1400 + index * 13,
                ourRatingBefore: 1500 + index,
                ourRatingDiff: index % 7 == 0 ? nil : index - 20,
                facts: Fixtures.facts(generations: [Fixtures.generation("M\(index % 3)", step: index * 100, moves: 5)])
            ))
        }
        let stats = try compute(rows)
        for filter in LichessBotStatsFilter.allCases {
            let alone = try compute(rows.filter(filter.includes))
            XCTAssertEqual(stats[filter], alone[.all], filter.rawValue)
        }
        XCTAssertEqual(stats.recordSpeeds, ["blitz", "rapid", "bullet"])
    }

    // MARK: - Time controls

    func testTimeControlRowsIncludingAnUnknownSpeed() throws {
        let rows = [
            try Fixtures.row(id: "a", at: now - hour, score: 1, speed: "blitz", ourRatingDiff: 6),
            try Fixtures.row(id: "b", at: now - 30 * hour, score: 0, speed: "blitz", ourRatingDiff: -5),
            try Fixtures.row(id: "c", at: now - hour, score: 0.5, speed: "hyperBullet"),
        ]
        let stats = try compute(rows)
        let today = stats[.all].byPeriod.today.timeControls
        XCTAssertEqual(today["blitz"]?.tally, LichessBotResultTally(wins: 1, draws: 0, losses: 0, unscored: 0))
        XCTAssertEqual(today["blitz"]?.ratingChange.total, 6)
        XCTAssertEqual(stats[.all].byPeriod.thisWeek.timeControls["blitz"]?.ratingChange.total, 1)
        XCTAssertEqual(today["hyperBullet"]?.tally.draws, 1)
        XCTAssertEqual(
            LichessBotTimeControlOrder.speeds(recordSpeeds: stats.recordSpeeds, accountSpeeds: ["rapid", "puzzle", "chess960"]),
            ["blitz", "rapid", "hyperBullet"]
        )
    }

    func testSparklineIsCappedAndOldestFirst() throws {
        var rows: [LichessBotGameSummary] = []
        for index in 0..<130 {
            rows.append(try Fixtures.row(id: String(format: "r%03d", index), at: now - Double(index) * hour, score: 1, speed: "rapid", ourRatingBefore: 2000 - index))
        }
        rows.append(try Fixtures.row(id: "casual", at: now, score: 1, speed: "rapid", rated: false, ourRatingBefore: 9999))
        rows.append(try Fixtures.row(id: "noRating", at: now, score: 1, speed: "rapid", ourRatingBefore: nil))
        let trend = try XCTUnwrap(try compute(rows).ratingTrends["rapid"])
        XCTAssertEqual(trend.count, LichessBotRecordStatistics.ratingTrendLimit)
        XCTAssertEqual(trend.first?.rating, 2000 - 99)
        XCTAssertEqual(trend.last?.rating, 2000)
        XCTAssertEqual(trend.map(\.createdAt), trend.map(\.createdAt).sorted())
    }

    // MARK: - Models

    func testModelKeysAttributionMixedAndNoModelRow() throws {
        let sha = String(repeating: "cd", count: 32)
        let rows = [
            // The same file reloaded (two generations, one hash): one key.
            try Fixtures.row(id: "a", at: now - 1 * hour, score: 1, facts: Fixtures.facts(generations: [
                Fixtures.generation("FILE", source: .file, step: 900, sha: sha, moves: 10),
                Fixtures.generation("FILE", source: .file, step: 900, sha: sha, moves: 5),
            ])),
            // A tie between two snapshots: the later one wins; mixed.
            try Fixtures.row(id: "b", at: now - 2 * hour, score: 0, facts: Fixtures.facts(generations: [
                Fixtures.generation("RUN", step: 100, moves: 6),
                Fixtures.generation("RUN", step: 200, moves: 6),
            ])),
            // A majority for the earlier snapshot; mixed.
            try Fixtures.row(id: "c", at: now - 3 * hour, score: 0.5, facts: Fixtures.facts(generations: [
                Fixtures.generation("RUN", step: 100, moves: 9),
                Fixtures.generation("RUN", step: 200, moves: 2),
            ])),
            // DCM never moved: its own row.
            try Fixtures.row(id: "d", at: now - 4 * hour, score: 1, facts: Fixtures.facts(ourMoveCount: 0, ourMovesWithDecision: 0)),
            // Without facts: also no model.
            try Fixtures.row(id: "e", at: now - 5 * hour, score: 0),
        ]
        let models = try compute(rows)[.all].byPeriod.allTime.models
        XCTAssertEqual(models.groups.map(\.modelID), ["FILE", "RUN"], "newest last game first")
        let file = models.groups[0]
        XCTAssertEqual(file.checkpoints.map(\.key), [.file(sha256: sha)])
        XCTAssertEqual(file.line.tally.wins, 1)
        XCTAssertEqual(file.line.mixedGames, 0, "one key is one model, not a mix")
        let run = models.groups[1]
        XCTAssertEqual(run.checkpoints.map(\.trainingStep), [100, 200])
        XCTAssertEqual(run.checkpoints[0].line.tally, LichessBotResultTally(wins: 0, draws: 1, losses: 0, unscored: 0))
        XCTAssertEqual(run.checkpoints[1].line.tally, LichessBotResultTally(wins: 0, draws: 0, losses: 1, unscored: 0))
        XCTAssertEqual(run.line.mixedGames, 2)
        XCTAssertEqual(run.checkpoints.map(\.line.mixedGames), [1, 1])
        XCTAssertEqual(run.line.firstGame, now - 3 * hour)
        XCTAssertEqual(run.line.lastGame, now - 2 * hour)
        XCTAssertEqual(models.noModelRecorded, LichessBotResultTally(wins: 1, draws: 0, losses: 1, unscored: 0))
        XCTAssertEqual(models.progression, [], "no run has two points")
    }

    func testProgressionMergesCheckpointsUntilThirtyGames() throws {
        var rows: [LichessBotGameSummary] = []
        var serial = 0
        // Checkpoints at steps 1000…6000 with 12, 12, 12, 20, 10, 5 games.
        for (step, games) in [(1000, 12), (2000, 12), (3000, 12), (4000, 20), (5000, 10), (6000, 5)] {
            for game in 0..<games {
                serial += 1
                rows.append(try Fixtures.row(
                    id: "p\(serial)", at: now - Double(serial) * 60, score: game % 2 == 0 ? 1 : 0,
                    facts: Fixtures.facts(generations: [Fixtures.generation("RUN", step: step, moves: 10)])
                ))
            }
        }
        let points = try compute(rows)[.all].byPeriod.allTime.models.progression
        // 12+12+12 = 36 ≥ 30 → one point; 20+10 = 30 → one point; 5 left.
        XCTAssertEqual(points.map(\.step), [3000, 5000, 6000])
        XCTAssertEqual(points.map(\.firstStep), [1000, 4000, 6000])
        XCTAssertEqual(points.map(\.games), [36, 30, 5])
        XCTAssertEqual(points.map(\.checkpointCount), [3, 2, 1])
        XCTAssertTrue(points.allSatisfy { !$0.stepIsCumulative })
        XCTAssertNotNil(points[0].interval)
    }

    // MARK: - Self-assessment

    func testCalibrationBrierSkillAndReliability() throws {
        let checkpoint = { (win: Float, draw: Float, loss: Float) in
            LichessBotGameMoveFacts.Checkpoint(moveNumber: 10, win: win, draw: draw, loss: loss)
        }
        let rows = [
            try Fixtures.row(id: "w", at: now - hour, score: 1, facts: Fixtures.facts(
                checkpoints: [checkpoint(0.5, 0.25, 0.25)],
                buckets: [.init(index: 6, positions: 2, sumExpected: 1.3)]
            )),
            try Fixtures.row(id: "l", at: now - hour, score: 0, facts: Fixtures.facts(
                checkpoints: [checkpoint(0.5, 0.25, 0.25)],
                checkpointsWithoutDecision: [20],
                buckets: [.init(index: 6, positions: 1, sumExpected: 0.6), .init(index: 1, positions: 3, sumExpected: 0.45)]
            )),
        ]
        let assessment = try compute(rows)[.all].byPeriod.allTime.selfAssessment
        let ten = try XCTUnwrap(assessment.calibration.first { $0.moveNumber == 10 })
        XCTAssertEqual(ten.games, 2)
        XCTAssertEqual(try XCTUnwrap(ten.meanPredictedExpected), 0.625, accuracy: 1e-9)
        XCTAssertEqual(try XCTUnwrap(ten.meanActualScore), 0.5, accuracy: 1e-9)
        // Win: (0.5−1)² + 0.25² + 0.25² = 0.375; loss: 0.5² + 0.25² + (0.25−1)² = 0.875.
        XCTAssertEqual(try XCTUnwrap(ten.brier), (0.375 + 0.875) / 2, accuracy: 1e-9)
        // Base rates ½ / 0 / ½: Brier_ref = 1 − ½ = ½.
        XCTAssertEqual(try XCTUnwrap(ten.skill), 1 - ((0.375 + 0.875) / 2) / 0.5, accuracy: 1e-9)
        XCTAssertEqual(ten.actualFrequencies, LichessBotOutcomeTriple(win: 0.5, draw: 0, loss: 0.5))
        let twenty = try XCTUnwrap(assessment.calibration.first { $0.moveNumber == 20 })
        XCTAssertEqual(twenty.games, 0)
        XCTAssertEqual(twenty.missing, 1)
        XCTAssertNil(twenty.brier)
        XCTAssertEqual(assessment.calibration.map(\.moveNumber), [10, 20, 40])

        XCTAssertEqual(assessment.reliability.map(\.index), [1, 6])
        XCTAssertEqual(assessment.reliability[0].positions, 3)
        XCTAssertEqual(assessment.reliability[0].meanPredicted, 0.15, accuracy: 1e-12)
        XCTAssertEqual(assessment.reliability[0].meanActual, 0, accuracy: 1e-12)
        XCTAssertEqual(assessment.reliability[1].positions, 3)
        XCTAssertEqual(assessment.reliability[1].meanPredicted, 1.9 / 3, accuracy: 1e-12)
        XCTAssertEqual(assessment.reliability[1].meanActual, 2.0 / 3, accuracy: 1e-12)
    }

    func testSkillIsUndefinedWhenEveryResultIsTheSame() throws {
        let rows = try (0..<3).map { index in
            try Fixtures.row(id: "w\(index)", at: now - hour, score: 1, facts: Fixtures.facts(
                checkpoints: [.init(moveNumber: 10, win: 0.7, draw: 0.2, loss: 0.1)]
            ))
        }
        let ten = try XCTUnwrap(try compute(rows)[.all].byPeriod.allTime.selfAssessment.calibration.first)
        XCTAssertNotNil(ten.brier)
        XCTAssertNil(ten.skill)
    }

    func testBlownWinsSavesAndDecisivePlies() throws {
        let rows = [
            try Fixtures.row(id: "blownDraw", at: now - 1 * hour, score: 0.5, status: "draw", plies: 60, facts: Fixtures.facts(heldWinStartPly: 20)),
            try Fixtures.row(id: "blownLoss", at: now - 2 * hour, score: 0, plies: 50, facts: Fixtures.facts(heldWinStartPly: 14, decisive: .atPly(40))),
            try Fixtures.row(id: "converted", at: now - 3 * hour, score: 1, plies: 40, facts: Fixtures.facts(heldWinStartPly: 10, decisive: .atPly(10))),
            try Fixtures.row(id: "saved", at: now - 4 * hour, score: 1, plies: 70, facts: Fixtures.facts(heldLossStartPly: 30, decisive: .atPly(60))),
            try Fixtures.row(id: "lostHeld", at: now - 5 * hour, score: 0, plies: 30, facts: Fixtures.facts(heldLossStartPly: 12, decisive: .never)),
            try Fixtures.row(id: "noDecisions", at: now - 6 * hour, score: 1, plies: 30, facts: Fixtures.facts(ourMovesWithDecision: 0, decisive: .noDecisions)),
        ]
        let assessment = try compute(rows)[.all].byPeriod.allTime.selfAssessment
        XCTAssertEqual(assessment.heldWins.held, 3)
        XCTAssertEqual(assessment.heldWins.turned, 2)
        XCTAssertEqual(assessment.heldWins.recentTurned.map(\.gameID), ["blownDraw", "blownLoss"])
        XCTAssertEqual(assessment.heldWins.recentTurned.map(\.startPly), [20, 14])
        XCTAssertEqual(assessment.heldLosses.held, 2)
        XCTAssertEqual(assessment.heldLosses.turned, 1)
        XCTAssertEqual(assessment.heldLosses.recentTurned.map(\.gameID), ["saved"])
        XCTAssertEqual(assessment.decisiveWins, LichessBotDecisiveSummary(games: 2, meanPly: 35, medianPly: 35, meanLead: 20, never: 0, noData: 1))
        XCTAssertEqual(assessment.decisiveLosses, LichessBotDecisiveSummary(games: 1, meanPly: 40, medianPly: 40, meanLead: 10, never: 1, noData: 0))
        XCTAssertEqual(LichessBotStatsFormat.held(turned: 2, held: 3, verb: "blown", noun: "held wins"), "2 blown of 3 held wins (66.7%)")
    }

    // MARK: - Endings

    func testEndingsMatrix() throws {
        let rows = [
            try Fixtures.row(id: "m1", at: now - hour, score: 1, status: "mate"),
            try Fixtures.row(id: "m2", at: now - hour, score: 0, status: "mate"),
            try Fixtures.row(id: "t", at: now - hour, score: 0.5, status: "draw", facts: Fixtures.facts(localDrawCondition: .threefoldRepetition)),
            try Fixtures.row(id: "a", at: now - hour, score: 0.5, status: "draw", facts: Fixtures.facts()),
            try Fixtures.row(id: "x", at: now - hour, score: nil, status: "aborted"),
            try Fixtures.row(id: "c", at: now - hour, score: 1, status: "cheat"),
        ]
        let endings = try compute(rows)[.all].byPeriod.today.endings
        XCTAssertEqual(endings.rows.map(\.ending), [.checkmate, .threefoldRepetition, .agreedOrOtherDraw, .other(status: "cheat"), .notCounted])
        XCTAssertEqual(endings.rows[0], LichessBotEndingRow(ending: .checkmate, wins: 1, draws: 0, losses: 1, unscored: 0))
        XCTAssertEqual(endings.rows.last?.unscored, 1)
        XCTAssertEqual([endings.wins, endings.draws, endings.losses], [2, 2, 1])
    }

    // MARK: - Opponent strength

    func testBandsAtEveryEdgeAndTheFiftyPercentPoint() throws {
        XCTAssertEqual([-401, -400, -301, -300, -101, -100, -1, 0, 99, 100, 399, 400, 1000, -2000].map(LichessBotRatingBand.index(gap:)),
                       [-5, -4, -4, -3, -2, -1, -1, 0, 0, 1, 3, 4, 4, -5])
        XCTAssertEqual(LichessBotRatingBand.label(index: -5), "< \u{2212}400")
        XCTAssertEqual(LichessBotRatingBand.label(index: -4), "\u{2212}400…\u{2212}301")
        XCTAssertEqual(LichessBotRatingBand.label(index: -1), "\u{2212}100…\u{2212}1")
        XCTAssertEqual(LichessBotRatingBand.label(index: 0), "0…99")
        XCTAssertEqual(LichessBotRatingBand.label(index: 4), "≥ 400")

        let rows = [
            // Gap −1: band −1 (floored, not truncated to 0).
            try Fixtures.row(id: "a", at: now - hour, score: 1, opponentRating: 1499, ourRatingBefore: 1500),
            try Fixtures.row(id: "b", at: now - hour, score: 0, opponentRating: 1600, ourRatingBefore: 1500),
            try Fixtures.row(id: "c", at: now - hour, score: 0.5, opponentRating: 1700, ourRatingBefore: 1500),
            // Missing our rating: left out of the bands.
            try Fixtures.row(id: "d", at: now - hour, score: 1, opponentRating: 1700, ourRatingBefore: nil),
        ]
        let strength = try compute(rows)[.all].byPeriod.today.opponentStrength
        XCTAssertEqual(strength.bands.map(\.index), [-1, 1, 2])
        XCTAssertEqual(strength.games, 3)
        XCTAssertEqual(strength.gamesWithoutRatings, 1)
        XCTAssertEqual(strength.bands[0].meanExpected, 1 / (1 + pow(10, -1.0 / 400)), accuracy: 1e-12)
        guard case .estimate(let offset) = strength.fiftyPercentPoint else { return XCTFail("expected an estimate") }
        // The offset solves Σ 1/(1 + 10^((dᵢ − Δ)/400)) = 1.5.
        let sum = [-1.0, 100, 200].reduce(0.0) { $0 + 1 / (1 + pow(10, ($1 - offset) / 400)) }
        XCTAssertEqual(sum, 1.5, accuracy: 1e-4)
    }

    // MARK: - Degenerate inputs

    func testNoGamesAtAll() throws {
        let stats = try compute([])
        for filter in LichessBotStatsFilter.allCases {
            for period in LichessBotStatsPeriod.allCases {
                let row = stats[filter].periodRows[period]
                XCTAssertEqual(row.record.all.games, 0)
                XCTAssertNil(row.record.all.score)
                XCTAssertEqual(row.performance, .none)
                XCTAssertNil(row.opponentAverage)
                XCTAssertEqual(LichessBotStatsFormat.ratingChange(row.ratingChange), "–")
                let panes = stats[filter].byPeriod[period]
                XCTAssertEqual(panes.timeControls, [:])
                XCTAssertEqual(panes.models.groups, [])
                XCTAssertEqual(panes.endings.rows, [])
                XCTAssertEqual(panes.opponentStrength.fiftyPercentPoint, .none)
                XCTAssertEqual(panes.selfAssessment.calibration.map(\.games), [0, 0, 0])
            }
        }
        XCTAssertEqual(stats.ratingTrends, [:])
        XCTAssertEqual(stats.timeZoneIdentifier, "GMT")
        XCTAssertEqual(stats.validUntil, now + 12 * hour, "the next midnight")
    }

    func testAllAbortedAndAllWins() throws {
        let aborted = try compute([
            try Fixtures.row(id: "a", at: now - hour, score: nil, status: "aborted"),
            try Fixtures.row(id: "b", at: now - hour, score: nil, status: "noStart"),
        ])[.all]
        XCTAssertEqual(aborted.periodRows.today.record.all.scored, 0)
        XCTAssertEqual(aborted.notCounted, 2)
        XCTAssertEqual(aborted.periodRows.today.performance, .none)
        XCTAssertEqual(aborted.ratedWithoutRatingChange, 0, "aborted rated games change no rating")

        let wins = try compute([
            try Fixtures.row(id: "a", at: now - hour, score: 1, opponentRating: 1700),
            try Fixtures.row(id: "b", at: now - hour, score: 1, opponentRating: 1700),
        ])[.all]
        XCTAssertEqual(wins.periodRows.today.record.all.score, 1)
        guard case .atLeast(let bound) = wins.periodRows.today.performance else { return XCTFail("expected a bound") }
        XCTAssertEqual(bound, 1700 + 400 * log10(3), accuracy: 1e-9)
        XCTAssertEqual(LichessBotStatsFormat.estimate(wins.periodRows.today.performance), "≥1891")
    }
}
