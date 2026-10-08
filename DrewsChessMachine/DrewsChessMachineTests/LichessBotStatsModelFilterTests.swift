import XCTest
@testable import DrewsChessMachine

/// The Record card's Model filter: games counted by the model they are
/// attributed to (majority of DCM's moves, `LichessBotModelAttribution`), the
/// model list, and the pipeline recomputing and remembering the selection.
@MainActor
final class LichessBotStatsModelFilterTests: XCTestCase {
    private typealias Fixtures = LichessBotStatsFixtures

    private static let start = Date(timeIntervalSince1970: 1_791_000_000)

    /// Game "a": model A only. "b": model B only. "c": A 5 moves, B 15 (B by
    /// majority, mixed). "d": no recorded model.
    private func rows() throws -> [LichessBotGameSummary] {
        let a = Fixtures.generation("A", step: 1000, moves: 20)
        let b = Fixtures.generation("B", step: 2000, moves: 20)
        return [
            try Fixtures.row(id: "a", at: Self.start, score: 1, facts: Fixtures.facts(generations: [a])),
            try Fixtures.row(id: "b", at: Self.start.addingTimeInterval(60), score: 0, facts: Fixtures.facts(generations: [b])),
            try Fixtures.row(id: "c", at: Self.start.addingTimeInterval(120), score: 0.5, facts: Fixtures.facts(generations: [
                Fixtures.generation("A", step: 1000, moves: 5), Fixtures.generation("B", step: 2000, moves: 15),
            ])),
            try Fixtures.row(id: "d", at: Self.start.addingTimeInterval(180), score: 1, facts: nil),
        ]
    }

    private static func key(_ modelID: String, step: Int) -> LichessBotModelKey {
        .snapshot(sourceKind: .trainerSnapshot, modelID: modelID, trainingStep: step)
    }

    func testASelectionCountsTheMajorityModelsGames() throws {
        let rows = try rows()
        XCTAssertEqual(rows.filter { LichessBotStatsModelSelection.all.includes($0) }.map(\.gameID), ["a", "b", "c", "d"])
        XCTAssertEqual(rows.filter { LichessBotStatsModelSelection.model(Self.key("A", step: 1000)).includes($0) }.map(\.gameID), ["a"])
        XCTAssertEqual(rows.filter { LichessBotStatsModelSelection.model(Self.key("B", step: 2000)).includes($0) }.map(\.gameID), ["b", "c"])
    }

    func testTheChoicesListEveryAttributedModelNewestFirst() throws {
        let choices = LichessBotStatsModelChoice.choices(from: try rows())
        XCTAssertEqual(choices.map(\.key), [Self.key("B", step: 2000), Self.key("A", step: 1000)])
        XCTAssertEqual(choices.map(\.games), [2, 1])
        XCTAssertEqual(choices.first?.menuLabel, "B · step 2,000 · \(LichessBotModelSourceKind.trainerSnapshot.displayName) (2 games)")
    }

    func testTheSelectionRoundTripsThroughJSON() throws {
        for selection in [LichessBotStatsModelSelection.all, .model(Self.key("A", step: 1000)), .model(.file(sha256: "ab12"))] {
            XCTAssertEqual(try JSONDecoder().decode(LichessBotStatsModelSelection.self, from: JSONEncoder().encode(selection)), selection)
        }
    }

    /// Selecting a model recomputes from its games only, lists every model
    /// whatever the selection, and is remembered by a new pipeline.
    func testThePipelineFiltersRecomputesAndRemembers() async throws {
        let defaults = try makeTemporaryDefaults()
        let pipeline = LichessBotRecordStatisticsPipeline(defaults: defaults, calendar: { Fixtures.utcCalendar })
        addTeardownBlock { @MainActor in await pipeline.shutdown(reason: "test teardown") }
        pipeline.indexChanged(rows: try rows())
        await pipeline.latestComputation?.value
        guard case .ready(let all) = pipeline.state else { return XCTFail("expected a snapshot") }
        XCTAssertEqual(pipeline.modelChoices.count, 2)

        pipeline.rememberedModel = .model(Self.key("B", step: 2000))
        await pipeline.latestComputation?.value
        guard case .ready(let onlyB) = pipeline.state else { return XCTFail("expected a snapshot") }
        XCTAssertEqual(onlyB[.all].periodRows.allTime.record.all.games, 2)
        XCTAssertEqual(all[.all].periodRows.allTime.record.all.games, 4)
        XCTAssertEqual(pipeline.modelChoices.count, 2, "the menu lists every model, whatever the selection")

        let reopened = LichessBotRecordStatisticsPipeline(defaults: defaults, calendar: { Fixtures.utcCalendar })
        addTeardownBlock { @MainActor in await reopened.shutdown(reason: "test teardown") }
        XCTAssertEqual(reopened.rememberedModel, .model(Self.key("B", step: 2000)))
    }

    /// A remembered model no game is attributed to falls back to every model.
    func testAModelWithNoGamesFallsBackToAll() async throws {
        let pipeline = LichessBotRecordStatisticsPipeline(defaults: try makeTemporaryDefaults(), calendar: { Fixtures.utcCalendar })
        addTeardownBlock { @MainActor in await pipeline.shutdown(reason: "test teardown") }
        pipeline.indexChanged(rows: try rows())
        await pipeline.latestComputation?.value
        pipeline.rememberedModel = .model(Self.key("Gone", step: 1))
        await pipeline.latestComputation?.value
        XCTAssertEqual(pipeline.rememberedModel, .all)
        await pipeline.latestComputation?.value
        guard case .ready(let statistics) = pipeline.state else { return XCTFail("expected a snapshot") }
        XCTAssertEqual(statistics[.all].periodRows.allTime.record.all.games, 4)
    }
}
