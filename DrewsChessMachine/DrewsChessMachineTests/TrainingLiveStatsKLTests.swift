import XCTest
@testable import DrewsChessMachine

/// The KL probe's results reach the rolling KL windows — and from there the
/// `[STATS]` `kl=` field and the Policy KL charts — on every step that
/// carries them. The probe runs when the completed-step count is a multiple
/// of `kl_probe_interval` (trainer steps 1, 101, 201, …) while diagnostics
/// run on multiples of `batch_stats_interval` (default 10); `recordStep`
/// once appended KL only on diagnostics steps, which under the defaults no
/// probe step is, so every probe value was dropped and the charts stayed
/// blank.
final class TrainingLiveStatsKLTests: XCTestCase {

    private func makeTiming(hasDiagnostics: Bool, klMean: Double?, klStdDev: Double?) -> TrainStepTiming {
        TrainStepTiming(
            dataPrepMs: 1, gpuRunMs: 2, readbackMs: 0.1, queueWaitMs: 0, totalMs: 3,
            klMean: klMean,
            klStdDev: klStdDev,
            loss: 1, policyLoss: 1, valueLoss: 0.5,
            policyEntropy: hasDiagnostics ? 2 : .nan,
            illegalMassPenalty: 0.01,
            policyNonNegligibleCount: .nan,
            policyNonNegligibleIllegalCount: .nan,
            gradGlobalNorm: 3,
            gradientCap: .hardMaxOnly(hardMax: 15),
            valueMean: .nan,
            valueAbsMean: .nan,
            valueProbWin: .nan, valueProbDraw: .nan, valueProbLoss: .nan,
            freshBaselineMs: nil,
            policyHeadWeightNorm: .nan,
            policyLogitAbsMax: .nan,
            playedMoveProb: .nan,
            playedMoveProbPosAdv: .nan,
            playedMoveProbNegAdv: .nan,
            advantageMean: .nan, advantageStd: .nan, advantageMin: .nan, advantageMax: .nan,
            advantageFracPositive: .nan, advantageFracSmall: .nan,
            advantageRaw: nil,
            policyLossWin: nil, policyLossLoss: nil,
            velocityNorm: .nan,
            sampledBatchMeanGameLength: .nan,
            sampledBatchDrawFraction: .nan,
            hasDiagnostics: hasDiagnostics
        )
    }

    /// The two schedules don't line up: a probe step is not a diagnostics
    /// step. This is the situation the fix has to handle.
    func testProbeStepsAreNotDiagnosticsSteps() {
        // The probe's index is the completed-step count before the step.
        XCTAssertTrue(ChessTrainer.isKLProbeStep(stepIndex: 100, interval: 100))
        XCTAssertFalse(ChessTrainer.isDiagnosticsStep(trainerStep: 101, batchStatsInterval: 50))
    }

    func testKLFromAProbeStepWithoutDiagnosticsIsRecorded() throws {
        let box = TrainingLiveStatsBox(rollingWindow: 100)
        box.recordStep(makeTiming(hasDiagnostics: false, klMean: 0.002, klStdDev: 0.001))
        box.recordStep(makeTiming(hasDiagnostics: false, klMean: nil, klStdDev: nil))
        box.recordStep(makeTiming(hasDiagnostics: true, klMean: nil, klStdDev: nil))
        box.recordStep(makeTiming(hasDiagnostics: false, klMean: 0.004, klStdDev: 0.003))
        let snap = box.snapshot()
        XCTAssertEqual(try XCTUnwrap(snap.rollingKLMean), 0.003, accuracy: 1e-12, "the mean of the two probe values only")
        XCTAssertEqual(try XCTUnwrap(snap.rollingKLStdDev), 0.002, accuracy: 1e-12)
    }

    func testNoProbeStepLeavesKLEmpty() {
        let box = TrainingLiveStatsBox(rollingWindow: 100)
        box.recordStep(makeTiming(hasDiagnostics: true, klMean: nil, klStdDev: nil))
        box.recordStep(makeTiming(hasDiagnostics: false, klMean: nil, klStdDev: nil))
        let snap = box.snapshot()
        XCTAssertNil(snap.rollingKLMean)
        XCTAssertNil(snap.rollingKLStdDev)
    }

    /// A probe whose readback failed reports a non-finite value; it must not
    /// enter the mean.
    func testNonFiniteKLIsNotRecorded() throws {
        let box = TrainingLiveStatsBox(rollingWindow: 100)
        box.recordStep(makeTiming(hasDiagnostics: false, klMean: .nan, klStdDev: .infinity))
        box.recordStep(makeTiming(hasDiagnostics: false, klMean: 0.01, klStdDev: 0.02))
        let snap = box.snapshot()
        XCTAssertEqual(try XCTUnwrap(snap.rollingKLMean), 0.01, accuracy: 1e-12)
        XCTAssertEqual(try XCTUnwrap(snap.rollingKLStdDev), 0.02, accuracy: 1e-12)
    }
}
