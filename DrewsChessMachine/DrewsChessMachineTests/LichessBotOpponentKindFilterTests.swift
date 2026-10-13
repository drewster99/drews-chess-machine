import XCTest
@testable import DrewsChessMachine

/// The Opponents filter (`LichessBotOpponentKindFilter`): which games count
/// by opponent kind, how it combines with the Rated / Casual Games filter
/// on the Record card (every number, the rating trend and the footnote
/// count included), the Model menu under it, its remembered selection, and
/// the All Games window's rows and summary. A game against Lichess' AI
/// counts only under All opponents.
@MainActor
final class LichessBotOpponentKindFilterTests: XCTestCase {
    private typealias Fixtures = LichessBotStatsFixtures

    private struct NoSnapshot: Error {}

    private static func key(_ modelID: String, step: Int) -> LichessBotModelKey {
        .snapshot(sourceKind: .trainerSnapshot, modelID: modelID, trainingStep: step)
    }

    /// Six games, one minute apart, newest first:
    /// - "bot-rated": a bot, rated, won, rating 1500 before, +8 recorded; model A.
    /// - "bot-casual": a bot, casual, lost; model A.
    /// - "human-rated": a human, rated, drawn, rating 1510 before, no
    ///   rating change recorded; model B.
    /// - "human-casual-win" / "human-casual-loss": a human, casual; model B.
    /// - "lichess-ai": Lichess' AI, casual, won; no account, no model.
    private func rows() throws -> [LichessBotGameSummary] {
        let now = Date()
        func at(_ minutes: Int) -> Date { now.addingTimeInterval(Double(-minutes * 60)) }
        let modelA = Fixtures.facts(generations: [Fixtures.generation("A", step: 1000, moves: 20)])
        let modelB = Fixtures.facts(generations: [Fixtures.generation("B", step: 2000, moves: 20)])
        return [
            try Fixtures.row(id: "bot-rated", at: at(1), score: 1, rated: true, kind: .bot, opponentID: "somebot", ourRatingBefore: 1500, ourRatingDiff: 8, facts: modelA),
            try Fixtures.row(id: "bot-casual", at: at(2), score: 0, rated: false, kind: .bot, opponentID: "somebot", facts: modelA),
            try Fixtures.row(id: "human-rated", at: at(3), score: 0.5, rated: true, kind: .human, opponentID: "someone", ourRatingBefore: 1510, ourRatingDiff: nil, facts: modelB),
            try Fixtures.row(id: "human-casual-win", at: at(4), score: 1, rated: false, kind: .human, opponentID: "someone", facts: modelB),
            try Fixtures.row(id: "human-casual-loss", at: at(5), score: 0, rated: false, kind: .human, opponentID: "someone", facts: modelB),
            try Fixtures.row(id: "lichess-ai", at: at(6), score: 1, rated: false, kind: .lichessAI, opponentID: nil, opponentName: nil, opponentRating: nil, facts: nil),
        ]
    }

    private func makePipeline(defaults: UserDefaults) -> LichessBotRecordStatisticsPipeline {
        let pipeline = LichessBotRecordStatisticsPipeline(defaults: defaults, calendar: { Fixtures.utcCalendar })
        addTeardownBlock { @MainActor in await pipeline.shutdown(reason: "test teardown") }
        return pipeline
    }

    /// The statistics after selecting `opponents` and waiting for the
    /// recompute.
    private func statistics(_ pipeline: LichessBotRecordStatisticsPipeline, opponents: LichessBotOpponentKindFilter) async throws -> LichessBotRecordStatistics {
        pipeline.rememberedOpponents = opponents
        await pipeline.latestComputation?.value
        guard case .ready(let statistics) = pipeline.state else {
            XCTFail("expected a snapshot, got \(pipeline.state)")
            throw NoSnapshot()
        }
        return statistics
    }

    // MARK: - The predicate

    func testEachKindMatchesOnlyItsOwnFilterAndLichessAIOnlyAll() throws {
        let rows = try rows()
        func ids(_ filter: LichessBotOpponentKindFilter) -> [String] {
            rows.filter { filter.includes($0) }.map(\.gameID)
        }
        XCTAssertEqual(ids(.all), ["bot-rated", "bot-casual", "human-rated", "human-casual-win", "human-casual-loss", "lichess-ai"])
        XCTAssertEqual(ids(.bots), ["bot-rated", "bot-casual"])
        XCTAssertEqual(ids(.humans), ["human-rated", "human-casual-win", "human-casual-loss"])
    }

    func testRawValuesLabelsAndLogSuffixesAreStable() {
        XCTAssertEqual(LichessBotOpponentKindFilter.allCases.map(\.rawValue), ["all", "bots", "humans"])
        XCTAssertEqual(LichessBotOpponentKindFilter.allCases.map(\.label), ["All opponents", "Bots", "Humans"])
        XCTAssertEqual(LichessBotOpponentKindFilter.allCases.map(\.logSuffix), ["", " opponents=bots", " opponents=humans"])
    }

    // MARK: - Record card: combined with the Games filter

    /// Every Games filter × Opponents filter combination: game counts and
    /// W–D–L.
    func testTheRecordCardCombinesOpponentsWithRatedAndCasual() async throws {
        let pipeline = makePipeline(defaults: try makeTemporaryDefaults())
        pipeline.indexChanged(rows: try rows())
        await pipeline.latestComputation?.value

        struct Expected {
            let games: Int
            let wins: Int
            let draws: Int
            let losses: Int
        }
        let expected: [LichessBotOpponentKindFilter: [LichessBotStatsFilter: Expected]] = [
            .all: [
                .all: Expected(games: 6, wins: 3, draws: 1, losses: 2),
                .rated: Expected(games: 2, wins: 1, draws: 1, losses: 0),
                .casual: Expected(games: 4, wins: 2, draws: 0, losses: 2),
            ],
            .bots: [
                .all: Expected(games: 2, wins: 1, draws: 0, losses: 1),
                .rated: Expected(games: 1, wins: 1, draws: 0, losses: 0),
                .casual: Expected(games: 1, wins: 0, draws: 0, losses: 1),
            ],
            .humans: [
                .all: Expected(games: 3, wins: 1, draws: 1, losses: 1),
                .rated: Expected(games: 1, wins: 0, draws: 1, losses: 0),
                .casual: Expected(games: 2, wins: 1, draws: 0, losses: 1),
            ],
        ]
        for opponents in LichessBotOpponentKindFilter.allCases {
            let statistics = try await statistics(pipeline, opponents: opponents)
            for games in LichessBotStatsFilter.allCases {
                let want = try XCTUnwrap(expected[opponents]?[games])
                let record = statistics[games].periodRows.allTime.record.all
                let label = "\(opponents.rawValue) × \(games.rawValue)"
                XCTAssertEqual(record.games, want.games, label)
                XCTAssertEqual(record.wins, want.wins, label)
                XCTAssertEqual(record.draws, want.draws, label)
                XCTAssertEqual(record.losses, want.losses, label)
                // The same numbers in the panes as in the period table.
                XCTAssertEqual(statistics[games].byPeriod.allTime.timeControls["blitz"]?.tally.games ?? 0, want.games, label)
            }
        }
    }

    /// The footnote's "rated games without a rating change" and the rating
    /// trend follow the Opponents filter (the trend ignores the Games
    /// filter by design, so only the Opponents filter narrows it).
    func testTheFootnoteCountAndTheRatingTrendFollowOpponents() async throws {
        let pipeline = makePipeline(defaults: try makeTemporaryDefaults())
        pipeline.indexChanged(rows: try rows())
        await pipeline.latestComputation?.value

        let all = try await statistics(pipeline, opponents: .all)
        XCTAssertEqual(all[.all].ratedWithoutRatingChange, 1)
        XCTAssertEqual(all.ratingTrends["blitz"]?.map(\.gameID), ["human-rated", "bot-rated"])

        let bots = try await statistics(pipeline, opponents: .bots)
        XCTAssertEqual(bots[.all].ratedWithoutRatingChange, 0)
        XCTAssertEqual(bots[.rated].ratedWithoutRatingChange, 0)
        XCTAssertEqual(bots.ratingTrends["blitz"]?.map(\.rating), [1500])
        XCTAssertFalse(LichessBotRecordFootnote.text(bots[.all]).contains("rating change"))

        let humans = try await statistics(pipeline, opponents: .humans)
        XCTAssertEqual(humans[.all].ratedWithoutRatingChange, 1)
        XCTAssertEqual(humans[.rated].ratedWithoutRatingChange, 1)
        XCTAssertEqual(humans[.casual].ratedWithoutRatingChange, 0)
        XCTAssertEqual(humans.ratingTrends["blitz"]?.map(\.rating), [1510])
        XCTAssertTrue(LichessBotRecordFootnote.text(humans[.all]).contains("* 1 rated game without a rating change from Lichess"))
    }

    /// The Opponents and Model filters combine.
    func testOpponentsCombineWithTheModelFilter() async throws {
        let pipeline = makePipeline(defaults: try makeTemporaryDefaults())
        pipeline.indexChanged(rows: try rows())
        await pipeline.latestComputation?.value
        pipeline.rememberedModel = .model(Self.key("B", step: 2000))
        await pipeline.latestComputation?.value

        let humans = try await statistics(pipeline, opponents: .humans)
        XCTAssertEqual(humans[.all].periodRows.allTime.record.all.games, 3)
        let bots = try await statistics(pipeline, opponents: .bots)
        XCTAssertEqual(bots[.all].periodRows.allTime.record.all.games, 0, "model B played no bot")
        XCTAssertEqual(pipeline.rememberedModel, .model(Self.key("B", step: 2000)),
                       "a model with games, none against this kind of opponent, keeps its selection")
    }

    // MARK: - Model menu

    /// The menu lists the models that played the selected kind of
    /// opponent; `everyModel` lists every model whatever the filter.
    func testTheModelMenuFollowsOpponentsButEveryModelDoesNot() throws {
        let rows = try rows()
        let now = Date()
        let humans = try LichessBotStatsModelMenu.make(rows: rows, opponents: .humans, now: now, calendar: Fixtures.utcCalendar)
        XCTAssertEqual(humans.choices(filter: .all, period: .allTime).map(\.key), [Self.key("B", step: 2000)])
        XCTAssertEqual(humans.choices(filter: .all, period: .allTime).map(\.games), [3])
        XCTAssertEqual(humans.choices(filter: .rated, period: .allTime).map(\.games), [1])
        XCTAssertEqual(Set(humans.everyModel.map(\.key)), [Self.key("A", step: 1000), Self.key("B", step: 2000)])

        let bots = try LichessBotStatsModelMenu.make(rows: rows, opponents: .bots, now: now, calendar: Fixtures.utcCalendar)
        XCTAssertEqual(bots.choices(filter: .all, period: .allTime).map(\.key), [Self.key("A", step: 1000)])
        XCTAssertEqual(bots.choices(filter: .casual, period: .allTime).map(\.games), [1])

        let all = try LichessBotStatsModelMenu.make(rows: rows, opponents: .all, now: now, calendar: Fixtures.utcCalendar)
        XCTAssertEqual(Set(all.choices(filter: .all, period: .allTime).map(\.key)), [Self.key("A", step: 1000), Self.key("B", step: 2000)])
    }

    // MARK: - Remembered

    func testTheSelectionIsRememberedAndAnUnknownValueIgnored() throws {
        let defaults = try makeTemporaryDefaults()
        let pipeline = makePipeline(defaults: defaults)
        XCTAssertEqual(pipeline.rememberedOpponents, .all)
        pipeline.rememberedOpponents = .humans
        XCTAssertEqual(defaults.string(forKey: LichessBotRecordStatisticsPipeline.opponentsKey), "humans")
        XCTAssertEqual(makePipeline(defaults: defaults).rememberedOpponents, .humans)

        defaults.set("robots", forKey: LichessBotRecordStatisticsPipeline.opponentsKey)
        XCTAssertEqual(makePipeline(defaults: defaults).rememberedOpponents, .all)
    }

    // MARK: - All Games window

    func testTheAllGamesRowsAndSummaryFollowOpponentsAndOrigin() throws {
        let summaries = try rows()
        let origins: [String: LichessBotGameOriginDisplay] = [
            "bot-rated": LichessBotGameOriginDisplay(category: .matchmaking, detail: "d", basis: .recorded),
            "bot-casual": LichessBotGameOriginDisplay(category: .incoming, detail: "d", basis: .recorded),
            "human-rated": LichessBotGameOriginDisplay(category: .incoming, detail: "d", basis: .recorded),
            "human-casual-win": LichessBotGameOriginDisplay(category: .incoming, detail: "d", basis: .recorded),
            "human-casual-loss": LichessBotGameOriginDisplay(category: .incoming, detail: "d", basis: .recorded),
            "lichess-ai": LichessBotGameOriginDisplay(category: .challengeSheet, detail: "d", basis: .recorded),
        ]

        let all = LichessBotAllGamesRow.rows(summaries, origins: origins, filter: .all, opponents: .all)
        XCTAssertEqual(all.count, 6)
        XCTAssertEqual(LichessBotAllGamesRow.summary(filedCount: 6, shown: all, filter: .all, opponents: .all),
                       "6 filed games: Incoming 4 · Sheet 1 · Matchmaking 1")

        let bots = LichessBotAllGamesRow.rows(summaries, origins: origins, filter: .all, opponents: .bots)
        XCTAssertEqual(bots.map(\.id), ["bot-rated", "bot-casual"])
        XCTAssertEqual(LichessBotAllGamesRow.summary(filedCount: 6, shown: bots, filter: .all, opponents: .bots),
                       "2 of 6 filed games: Incoming 1 · Matchmaking 1")

        let humans = LichessBotAllGamesRow.rows(summaries, origins: origins, filter: .all, opponents: .humans)
        XCTAssertEqual(humans.map(\.id), ["human-rated", "human-casual-win", "human-casual-loss"])

        // Both filters at once: incoming games against bots.
        let incomingBots = LichessBotAllGamesRow.rows(summaries, origins: origins, filter: .only(.incoming), opponents: .bots)
        XCTAssertEqual(incomingBots.map(\.id), ["bot-casual"])
        XCTAssertEqual(LichessBotAllGamesRow.summary(filedCount: 6, shown: incomingBots, filter: .only(.incoming), opponents: .bots),
                       "1 of 6 filed games: Incoming 1")

        // Lichess' AI: shown under All opponents only.
        XCTAssertEqual(LichessBotAllGamesRow.rows(summaries, origins: origins, filter: .only(.challengeSheet), opponents: .all).map(\.id), ["lichess-ai"])
        XCTAssertEqual(LichessBotAllGamesRow.rows(summaries, origins: origins, filter: .only(.challengeSheet), opponents: .bots).count, 0)
        XCTAssertEqual(LichessBotAllGamesRow.rows(summaries, origins: origins, filter: .only(.challengeSheet), opponents: .humans).count, 0)
    }
}
