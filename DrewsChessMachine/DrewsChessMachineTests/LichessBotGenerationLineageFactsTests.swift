import XCTest
@testable import DrewsChessMachine

/// A generation's lineage (follow-lineage plan §3.6) reaches the per-game
/// facts and the Models pane (`LICHESS_BOT_RECORD_STATS_PLAN.md` §3.8): runs
/// are grouped by lineage run ID and placed by cumulative trainer step.
final class LichessBotGenerationLineageFactsTests: XCTestCase {
    private typealias Fixtures = LichessBotStatsFixtures

    private func lineage(run: String, segment: Int, cumStep: Int?) -> LichessBotGenerationLineage {
        LichessBotGenerationLineage(
            lineageRunID: run,
            segmentID: "\(run)-seg\(segment)",
            segmentIndex: segment,
            segmentChain: ["\(run)-seg\(segment)"],
            segmentLocalStep: 500,
            recordedUnix: 1_759_500_000,
            cumTrainerStep: cumStep,
            contentSHA256: String(repeating: "aa", count: 32),
            followed: nil
        )
    }

    func testTheGenerationsLineageReachesTheFacts() throws {
        var info = Fixtures.generationInfo(id: 1, modelID: "20261005-1-LINE", sha: String(repeating: "bc", count: 32), step: 500)
        info.lineage = lineage(run: "RUN-A", segment: 2, cumStep: 120_500)
        let record = try Fixtures.record(ourColor: .white, plies: 20, status: "resign", winner: "white") { _ in
            .decided(win: 0.5, draw: 0.2, loss: 0.3, generation: info)
        }
        let generation = try XCTUnwrap(LichessBotGameFacts(record: record).moves?.generations.first)
        XCTAssertEqual(generation.lineageRunID, "RUN-A")
        XCTAssertEqual(generation.segmentIndex, 2)
        XCTAssertEqual(generation.cumTrainerStep, 120_500)
        XCTAssertEqual(generation.modelKey, .file(sha256: String(repeating: "bc", count: 32)), "the file's own hash, never the header's content hash")
        // Without a lineage the fields are nil.
        let plain = try Fixtures.record(ourColor: .white, plies: 20, status: "resign", winner: "white") { _ in
            .decided(win: 0.5, draw: 0.2, loss: 0.3, generation: Fixtures.generationInfo(id: 1, modelID: "M"))
        }
        let plainGeneration = try XCTUnwrap(LichessBotGameFacts(record: plain).moves?.generations.first)
        XCTAssertNil(plainGeneration.lineageRunID)
        XCTAssertNil(plainGeneration.cumTrainerStep)
    }

    func testProgressionFollowsTheRunAcrossModelIDsByCumulativeStep() throws {
        // One run, two segments (two model IDs), checkpoints at cumulative
        // steps 100,000 / 110,000 / 120,000, 30 games each.
        var rows: [LichessBotGameSummary] = []
        var serial = 0
        let now = Date(timeIntervalSince1970: 1_791_374_400)
        for (modelID, cum, sha) in [("SEG1", 100_000, "01"), ("SEG1", 110_000, "02"), ("SEG2", 120_000, "03")] {
            for game in 0..<30 {
                serial += 1
                rows.append(try Fixtures.row(
                    id: "l\(serial)", at: now - Double(10_000 - serial), score: game % 2 == 0 ? 1 : 0,
                    facts: Fixtures.facts(generations: [Fixtures.generation(modelID, source: .followLineage, step: 500, sha: String(repeating: sha, count: 32), lineageRunID: "RUN-A", cumTrainerStep: cum, moves: 10)])
                ))
            }
        }
        let models = try LichessBotRecordStatistics.compute(rows: rows, now: now, calendar: Fixtures.utcCalendar)[.all].byPeriod.allTime.models
        XCTAssertEqual(Set(models.groups.map(\.modelID)), ["SEG1", "SEG2"], "the table still groups by model ID")
        XCTAssertEqual(models.progression.map(\.series), ["RUN-A", "RUN-A", "RUN-A"], "the chart follows the run across segments")
        XCTAssertEqual(models.progression.map(\.step), [100_000, 110_000, 120_000])
        XCTAssertTrue(models.progression.allSatisfy(\.stepIsCumulative))
        let checkpoint = try XCTUnwrap(models.groups.first { $0.modelID == "SEG2" }?.checkpoints.first)
        XCTAssertTrue(checkpoint.label.contains("cum \(120_000.formatted())"), checkpoint.label)
    }
}
