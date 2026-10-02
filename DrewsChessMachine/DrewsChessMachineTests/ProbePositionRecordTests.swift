//
//  ProbePositionRecordTests.swift
//  DrewsChessMachineTests
//
//  Pins the `--probe-positions-out` per-position record: the bookmove NLL
//  uses the one shared definition (`ProbeBookmoveNLL`), so averaging the
//  records of a battery reproduces the battery's summary `nll` exactly —
//  the property that makes position-level paired comparisons between
//  checkpoints consistent with the battery-level numbers already recorded.
//

import Foundation
import XCTest
@testable import DrewsChessMachine

final class ProbePositionRecordTests: XCTestCase {

    func testNLLUsesTheFlooredNegativeLog() {
        XCTAssertEqual(ProbeBookmoveNLL.nats(expectedProb: 0.5), -log(Double(Float(0.5))))
        XCTAssertEqual(ProbeBookmoveNLL.nats(expectedProb: 1), 0)
        XCTAssertEqual(
            ProbeBookmoveNLL.nats(expectedProb: 0),
            -log(Double(ProbeBookmoveNLL.probabilityFloor))
        )
    }

    func testRecordCarriesTheResultAndPuzzleMetadata() throws {
        let probe = try XCTUnwrap(LichessProbeData.wideSet.first)
        let bookmove = try XCTUnwrap(probe.acceptable.first)
        let result = syntheticResult(for: probe, expectedProb: 0.625, rank: 1, top1: bookmove)

        let record = ProbeModelCLI.positionRecord(
            result, index: 7, setLabel: "wide", logitAbsMax: 12.5,
            modelID: "20261002-1-test", modelPath: "/models/x.safetensors"
        )

        XCTAssertEqual(record["index"] as? Int, 7)
        XCTAssertEqual(record["set"] as? String, "wide")
        XCTAssertEqual(record["modelID"] as? String, "20261002-1-test")
        XCTAssertEqual(record["name"] as? String, probe.name)
        XCTAssertEqual(record["expectedRank"] as? Int, 1)
        XCTAssertEqual(record["expectedProb"] as? Double, Double(Float(0.625)))
        XCTAssertEqual(record["nll"] as? Double, ProbeBookmoveNLL.nats(expectedProb: 0.625))
        XCTAssertEqual(record["top1Move"] as? String, bookmove.uci)
        XCTAssertEqual(record["top1Prob"] as? Double, Double(Float(0.625)))
        XCTAssertEqual(record["logitAbsMax"] as? Double, 12.5)
        XCTAssertEqual(record["verdict"] as? String, ProbeVerdict.correctAndConfident.rawValue)

        let meta = try XCTUnwrap(LichessProbeData.metadata[probe.name])
        XCTAssertEqual(record["puzzleId"] as? String, meta.id)
        XCTAssertEqual(record["theme"] as? String, meta.theme)
        XCTAssertEqual(record["rating"] as? Int, meta.rating)

        // The record must serialize as the CLI writes it.
        XCTAssertNoThrow(try JSONSerialization.data(withJSONObject: record, options: [.sortedKeys]))
    }

    func testErroredResultOmitsRankTop1AndLogitMax() throws {
        let probe = try XCTUnwrap(LichessProbeData.wideSet.first)
        let record = ProbeModelCLI.positionRecord(
            TacticalProbeRunner.errorResult(for: probe), index: 0, setLabel: "wide", logitAbsMax: nil,
            modelID: "m", modelPath: "/m"
        )
        XCTAssertNil(record["expectedRank"])
        XCTAssertNil(record["top1Move"])
        XCTAssertNil(record["top1Prob"])
        XCTAssertNil(record["logitAbsMax"])
        XCTAssertEqual(record["verdict"] as? String, ProbeVerdict.error.rawValue)
        XCTAssertEqual(record["nll"] as? Double, -log(Double(ProbeBookmoveNLL.probabilityFloor)))
    }

    /// The battery summary's `nll` (live/CLI aggregation path) equals the mean
    /// of the per-position records' `nll` for the same results.
    func testMeanOfRecordNLLEqualsTheBatterySummaryNLL() throws {
        let probes = Array(LichessProbeData.wideSet.prefix(64))
        XCTAssertEqual(probes.count, 64)
        let probabilities: [Float] = [0.9, 0.31, 0.05, 0, 0.6, 0.002, 1, 0.47]
        var results: [ProbeResult] = []
        for (i, probe) in probes.enumerated() {
            let p = probabilities[i % probabilities.count]
            let bookmove = try XCTUnwrap(probe.acceptable.first)
            results.append(syntheticResult(for: probe, expectedProb: p, rank: p >= 0.5 ? 1 : 3, top1: bookmove))
        }

        let summaryNLL = LichessProbeOverallSummary(
            folding: LichessProbeHistory.aggregates(from: results)
        ).meanNegLogProb
        let records = results.enumerated().map { index, result in
            ProbeModelCLI.positionRecord(
                result, index: index, setLabel: "wide", logitAbsMax: nil, modelID: "m", modelPath: "/m"
            )
        }
        let recordNLLs = try records.map { try XCTUnwrap($0["nll"] as? Double) }
        let meanRecordNLL = recordNLLs.reduce(0, +) / Double(recordNLLs.count)

        XCTAssertEqual(meanRecordNLL, summaryNLL, accuracy: 1e-12)
    }

    // MARK: - Helpers

    /// A result whose legal count and entropy are real for the position and
    /// whose bookmove probability / rank are the given values.
    private func syntheticResult(
        for probe: TacticalProbe, expectedProb: Float, rank: Int, top1: ChessMove
    ) -> ProbeResult {
        let legalCount = MoveGenerator.legalMoves(for: probe.state).count
        let verdict: ProbeVerdict
        switch rank {
        case 1: verdict = expectedProb >= 0.5 ? .correctAndConfident : .correctButFlat
        case 2...5: verdict = .correctInTop5
        default: verdict = .wrong
        }
        return ProbeResult(
            probe: probe,
            topMoves: [ProbeResult.TopMoveEntry(move: top1, prob: expectedProb)],
            expectedRank: rank,
            expectedProb: expectedProb,
            legalCount: legalCount,
            legalEntropyNats: 1.25,
            uniformLegalEntropy: Float(log(Double(legalCount))),
            illegalMass: 0.001,
            valueWDL: (win: 0.4, draw: 0.2, loss: 0.4),
            verdict: verdict
        )
    }
}
