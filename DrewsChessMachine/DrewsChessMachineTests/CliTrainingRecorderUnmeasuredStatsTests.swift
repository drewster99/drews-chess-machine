import XCTest
@testable import DrewsChessMachine

/// `results.json` from the CLI runners (corpus replay, train-vs-UCI) must
/// encode on every step. Diagnostic values are measured only on stats steps;
/// the trainer leaves them NaN otherwise, and JSON cannot encode NaN, so the
/// runners record them as nil ("not measured") there.
final class CliTrainingRecorderUnmeasuredStatsTests: XCTestCase {

    private func line(policyEntropy: Double?, valueMean: Double?) -> CliTrainingRecorder.StatsLine {
        CliTrainingRecorder.StatsLine(
            elapsedSec: 1, steps: 1, positionsFed: 4096, bufferCount: 4096, bufferCapacity: 8192,
            policyLoss: 2.3, valueLoss: 0.9, policyEntropy: policyEntropy,
            policyIllegalMassPenalty: 0.001, gradGlobalNorm: 4.8, playedMoveProb: nil,
            valueMean: valueMean, valueAbsMean: nil, valueProbWin: nil, valueProbDraw: nil, valueProbLoss: nil,
            policyLogitMean: nil, valueLogitMean: nil,
            batchSize: 4096, learningRate: 1e-3, gradClipMaxNorm: 30, weightDecayC: 5e-4, dropoutRate: 0,
            entropyRegularizationCoeff: 0, drawPenalty: 0, policyLossWeight: 1, valueLossWeight: 1,
            lrEffectiveBase: 1e-3, momentumEffective: 0.9, buildNumber: 1, trainerID: "20260928-1-TEST",
            positionsProduced: 4096,
            lineageTotals: CliTrainingRecorder.LineageTotals(cumTrainerStep: nil, cumTrainStepSec: nil, cumGames: nil)
        )
    }

    func testANonStatsStepWithUnmeasuredDiagnosticsEncodes() throws {
        let recorder = CliTrainingRecorder()
        recorder.appendStats(line(policyEntropy: nil, valueMean: nil))
        XCTAssertNoThrow(try recorder.encodedJSONData(totalTrainingSeconds: 1))
    }

    /// The failure the runners used to hit: a NaN diagnostic makes the whole
    /// file unwritable. Pins why they must pass nil instead.
    func testANaNDiagnosticMakesTheFileUnwritable() {
        let recorder = CliTrainingRecorder()
        recorder.appendStats(line(policyEntropy: .nan, valueMean: .nan))
        XCTAssertThrowsError(try recorder.encodedJSONData(totalTrainingSeconds: 1))
    }
}
