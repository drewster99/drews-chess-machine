//
//  ProbeHistoryPersistenceTests.swift
//  DrewsChessMachineTests
//
//  Round-trip tests for persisting the Lichess (200-puzzle) and tactical
//  probe-monitor histories into the session checkpoint. The runtime
//  `ProbeResult` is not Codable (it carries a GameState, a move set, and
//  a tuple); persistence stores a scalar mirror + the probe name and
//  reconstructs the position from the immutable fixture sets on load.
//  These tests guard:
//   - ProbeResultCodable encode→decode→reconstruct preserves scalars and
//     round-trips top-move UCI back to the exact ChessMove
//   - the Lichess/tactical snapshots survive JSON round-trip, including
//     the per-tick trainingStep and non-finite puzzle-Elo values (the
//     bug that would otherwise abort a session save in early training)
//   - the live history objects round-trip through makeSnapshot/restore
//   - the OVERALL chart x-axis helper picks trainer step vs tick index
//

import XCTest
@testable import DrewsChessMachine

@MainActor
final class ProbeHistoryPersistenceTests: XCTestCase {

    // MARK: - Helpers

    private func tacticalIndex() -> [String: TacticalProbe] {
        Dictionary(
            uniqueKeysWithValues: TacticalProbeData.standardSet.map { ($0.name, $0) }
        )
    }

    private func makeResult(
        probe: TacticalProbe,
        move: ChessMove,
        verdict: ProbeVerdict = .correctAndConfident,
        expectedRank: Int? = 1,
        expectedProb: Float = 0.73
    ) -> ProbeResult {
        ProbeResult(
            probe: probe,
            topMoves: [ProbeResult.TopMoveEntry(move: move, prob: expectedProb)],
            expectedRank: expectedRank,
            expectedProb: expectedProb,
            legalCount: 20,
            legalEntropyNats: 1.5,
            uniformLegalEntropy: 3.0,
            illegalMass: 0.01,
            valueWDL: (win: 0.6, draw: 0.3, loss: 0.1),
            verdict: verdict
        )
    }

    // MARK: - ProbeResultCodable round-trip

    func testProbeResultCodableRoundTripTacticalFixture() throws {
        let probe = try XCTUnwrap(TacticalProbeData.standardSet.first)
        let move = try XCTUnwrap(probe.acceptable.first)
        let result = makeResult(probe: probe, move: move)

        let codable = ProbeResultCodable(result)
        let data = try JSONEncoder().encode(codable)
        let decoded = try JSONDecoder().decode(ProbeResultCodable.self, from: data)
        XCTAssertEqual(decoded, codable)

        let restored = try XCTUnwrap(decoded.reconstruct(using: tacticalIndex()))
        XCTAssertEqual(restored.probe.name, probe.name)
        XCTAssertEqual(restored.verdict, result.verdict)
        XCTAssertEqual(restored.expectedRank, result.expectedRank)
        XCTAssertEqual(restored.expectedProb, result.expectedProb)
        XCTAssertEqual(restored.legalCount, result.legalCount)
        XCTAssertEqual(restored.illegalMass, result.illegalMass)
        XCTAssertEqual(restored.valueWDL.win, result.valueWDL.win)
        XCTAssertEqual(restored.valueWDL.draw, result.valueWDL.draw)
        XCTAssertEqual(restored.valueWDL.loss, result.valueWDL.loss)
        XCTAssertEqual(restored.topMoves.count, 1)
        // The whole point of name-based reconstruction: the UCI round-trip
        // yields the identical ChessMove, so popover arrow / acceptable-set
        // membership checks still hold post-resume.
        XCTAssertEqual(restored.topMoves.first?.move, move)
    }

    func testReconstructReturnsNilForUnknownProbe() throws {
        let probe = try XCTUnwrap(TacticalProbeData.standardSet.first)
        let move = try XCTUnwrap(probe.acceptable.first)
        let codable = ProbeResultCodable(makeResult(probe: probe, move: move))
        // Empty index → name doesn't resolve → skipped, no crash.
        XCTAssertNil(codable.reconstruct(using: [:]))
    }

    // MARK: - OverallTickSample non-finite handling

    func testOverallSampleNaNAndNilStepRoundTrip() throws {
        let now = Date(timeIntervalSince1970: 1)
        let sample = LichessProbeHistory.OverallTickSample(
            timestamp: now, meanNegLogProb: 3.4, puzzleElo: .nan, trainingStep: nil
        )
        // Must not throw despite NaN (default JSONEncoder would on a raw Double).
        let data = try JSONEncoder().encode(sample)
        let decoded = try JSONDecoder().decode(
            LichessProbeHistory.OverallTickSample.self, from: data
        )
        XCTAssertTrue(decoded.puzzleElo.isNaN)
        XCTAssertNil(decoded.trainingStep)
        XCTAssertEqual(decoded.meanNegLogProb, 3.4)
    }

    // MARK: - Snapshot JSON round-trip

    func testLichessSnapshotJSONRoundTrip() throws {
        let probe = try XCTUnwrap(LichessProbeData.largeSet.first)
        let move = try XCTUnwrap(probe.acceptable.first)
        let now = Date(timeIntervalSince1970: 1_000_000)

        let agg = LichessProbeHistory.Aggregate(
            theme: .lichessFork, total: 25, argmaxCorrect: 5, top5Correct: 12,
            errored: 0, sumExpectedProb: 1.2, sumExpectedRank: 60,
            countWithRank: 25, sumNegLogProb: 90.0
        )
        let entry = LichessProbeHistory.Entry(timestamp: now, aggregate: agg)
        let overallFinite = LichessProbeHistory.OverallTickSample(
            timestamp: now, meanNegLogProb: 3.5, puzzleElo: 712, trainingStep: 4200
        )
        let overallNegInf = LichessProbeHistory.OverallTickSample(
            timestamp: now, meanNegLogProb: 3.9, puzzleElo: -.infinity, trainingStep: 4300
        )
        let snapshot = LichessProbeHistorySnapshot(
            perTheme: ["lichessFork": [entry]],
            overall: [overallFinite, overallNegInf],
            latestResults: [ProbeResultCodable(makeResult(probe: probe, move: move))],
            latestTimestamp: now,
            latestModelLabel: "20260601-1-ABCD",
            latestTrainingStep: 4300,
            latestPositionsTrained: 100_000,
            latestActiveTrainingSec: 3600,
            latestArenaCount: 3,
            latestPromotionCount: 1
        )

        let data = try JSONEncoder().encode(snapshot)
        let decoded = try JSONDecoder().decode(LichessProbeHistorySnapshot.self, from: data)
        // Equatable holds: finite values match; -inf == -inf is true.
        XCTAssertEqual(decoded, snapshot)
        XCTAssertEqual(decoded.overall[0].trainingStep, 4200)
        XCTAssertEqual(decoded.overall[1].puzzleElo, -.infinity)
    }

    // MARK: - Live history round-trip

    func testTacticalHistoryRecordSnapshotRestore() throws {
        let history = TacticalProbeHistory()
        let probe = try XCTUnwrap(TacticalProbeData.standardSet.first)
        let move = try XCTUnwrap(probe.acceptable.first)
        history.record([makeResult(probe: probe, move: move)])
        XCTAssertEqual(history.entries[probe.name]?.count, 1)

        let data = try JSONEncoder().encode(history.makeSnapshot())
        let decoded = try JSONDecoder().decode(TacticalProbeHistorySnapshot.self, from: data)

        let restored = TacticalProbeHistory()
        restored.restore(from: decoded)
        let series = try XCTUnwrap(restored.entries[probe.name])
        XCTAssertEqual(series.count, 1)
        XCTAssertEqual(series.first?.result.probe.name, probe.name)
        XCTAssertEqual(series.first?.result.topMoves.first?.move, move)
        XCTAssertEqual(series.first?.result.verdict, .correctAndConfident)
    }

    func testLichessHistoryRecordSnapshotRestore() throws {
        let history = LichessProbeHistory()
        let probes = Array(LichessProbeData.largeSet.prefix(2))
        XCTAssertEqual(probes.count, 2)
        let results = try probes.map { probe -> ProbeResult in
            let move = try XCTUnwrap(probe.acceptable.first)
            return makeResult(probe: probe, move: move)
        }
        history.record(
            LichessProbeHistory.aggregates(from: results),
            allResults: results,
            modelLabel: "m",
            trainingStep: 1234,
            positionsTrained: 5000,
            activeTrainingSec: 60,
            arenaCount: 2,
            promotionCount: 1
        )
        XCTAssertEqual(history.overallSeries.count, 1)

        let data = try JSONEncoder().encode(history.makeSnapshot())
        let decoded = try JSONDecoder().decode(LichessProbeHistorySnapshot.self, from: data)

        let restored = LichessProbeHistory()
        restored.restore(from: decoded)
        XCTAssertEqual(restored.overallSeries.count, 1)
        XCTAssertEqual(restored.overallSeries.first?.trainingStep, 1234)
        XCTAssertEqual(restored.latestPerPuzzleResults.count, 2)
        XCTAssertEqual(restored.latestTickModelLabel, "m")
        XCTAssertEqual(restored.latestTickTrainingStep, 1234)
    }

    // MARK: - OVERALL chart x-axis helper

    func testXPositionsUsesStepsWhenMonotonic() {
        let now = Date(timeIntervalSince1970: 0)
        let samples = [100, 200, 350].map {
            LichessProbeHistory.OverallTickSample(
                timestamp: now, meanNegLogProb: 3, puzzleElo: 700, trainingStep: $0
            )
        }
        let result = LichessProbeOverallTrendChart.xPositions(for: samples)
        XCTAssertEqual(result.xs, [100, 200, 350])
        XCTAssertEqual(result.label, "trainer step")
    }

    func testXPositionsFallsBackToIndexWhenStepMissing() {
        let now = Date(timeIntervalSince1970: 0)
        let samples = [
            LichessProbeHistory.OverallTickSample(
                timestamp: now, meanNegLogProb: 3, puzzleElo: 700, trainingStep: 100
            ),
            LichessProbeHistory.OverallTickSample(
                timestamp: now, meanNegLogProb: 3, puzzleElo: 700, trainingStep: nil
            ),
        ]
        let result = LichessProbeOverallTrendChart.xPositions(for: samples)
        XCTAssertEqual(result.xs, [0, 1])
        XCTAssertEqual(result.label, "tick #")
    }

    func testXPositionsFallsBackWhenNonMonotonic() {
        let now = Date(timeIntervalSince1970: 0)
        let samples = [300, 100].map {
            LichessProbeHistory.OverallTickSample(
                timestamp: now, meanNegLogProb: 3, puzzleElo: 700, trainingStep: $0
            )
        }
        let result = LichessProbeOverallTrendChart.xPositions(for: samples)
        XCTAssertEqual(result.xs, [0, 1])
        XCTAssertEqual(result.label, "tick #")
    }

    // MARK: - OVERALL chart EMA overlay helper

    /// The first EMA value seeds from the first sample, and the output length
    /// matches the input — the invariants every downstream plot relies on.
    func testEmaSeedsFromFirstSampleAndPreservesLength() {
        let ys = [3.0, 7.0, 1.0, 9.0, 2.0]
        let out = LichessProbeOverallTrendChart.ema(ys, span: 10)
        XCTAssertEqual(out.count, ys.count)
        XCTAssertEqual(out[0], 3.0, "EMA[0] must equal the first sample")
    }

    /// Exact recurrence at span 3 → alpha = 2/(3+1) = 0.5, so each output is the
    /// midpoint of the current sample and the previous EMA. Hand-computed:
    /// [0, 0.5·10+0.5·0=5, 0.5·10+0.5·5=7.5, 0.5·0+0.5·7.5=3.75].
    func testEmaKnownRecurrenceSpan3() {
        let out = LichessProbeOverallTrendChart.ema([0, 10, 10, 0], span: 3)
        XCTAssertEqual(out, [0, 5, 7.5, 3.75])
    }

    /// EMA of a constant series is that constant at every point (any span).
    func testEmaOfConstantIsConstant() {
        let out = LichessProbeOverallTrendChart.ema([4, 4, 4, 4], span: 25)
        for v in out { XCTAssertEqual(v, 4, accuracy: 1e-12) }
    }

    /// Degenerate inputs are returned unchanged: too few points to smooth, or a
    /// non-positive span (alpha would be ill-defined).
    func testEmaDegenerateInputsReturnedUnchanged() {
        XCTAssertEqual(LichessProbeOverallTrendChart.ema([], span: 25), [])
        XCTAssertEqual(LichessProbeOverallTrendChart.ema([5], span: 25), [5])
        XCTAssertEqual(LichessProbeOverallTrendChart.ema([1, 2, 3], span: 0), [1, 2, 3])
    }
}
