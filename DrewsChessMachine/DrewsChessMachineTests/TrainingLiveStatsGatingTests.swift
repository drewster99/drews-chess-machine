import XCTest
@testable import DrewsChessMachine

/// Guards the telemetry-gating invariant in `TrainingLiveStatsBox.recordStep`
/// (see GPU_UTILIZATION_PLAN.md, Phase 1): on a non-stats step the diagnostic
/// `TrainStepTiming` fields are `.nan` / `nil` (their graph reductions weren't
/// encoded), and `recordStep` must:
///   - still fold the loss / grad-norm / timing values into their rolling means
///     every step (those are valid on every step), AND
///   - NOT fold the `.nan` diagnostic placeholders into the diagnostic rolling
///     means (which would poison them), AND
///   - NOT overwrite `lastTiming` with a non-stats step (so the "Last Step" UI
///     readout never shows a NaN entropy/advantage).
final class TrainingLiveStatsGatingTests: XCTestCase {

    /// Build a `TrainStepTiming`. Diagnostic fields default to `.nan` / `nil`
    /// (the non-stats shape); pass `hasDiagnostics: true` with explicit values
    /// to simulate a stats step.
    private func makeTiming(
        hasDiagnostics: Bool,
        policyLoss: Float,
        gradGlobalNorm: Float,
        policyEntropy: Float = .nan,
        valueMean: Float = .nan,
        advantageRaw: [Float]? = nil
    ) -> TrainStepTiming {
        TrainStepTiming(
            dataPrepMs: 1, gpuRunMs: 2, readbackMs: 0.1, queueWaitMs: 0, totalMs: 3,
            loss: policyLoss, policyLoss: policyLoss, valueLoss: 0.5,
            policyEntropy: policyEntropy,
            illegalMassPenalty: 0.01,
            policyNonNegligibleCount: .nan,
            policyNonNegligibleIllegalCount: .nan,
            gradGlobalNorm: gradGlobalNorm,
            valueMean: valueMean,
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
            advantageRaw: advantageRaw,
            policyLossWin: nil, policyLossLoss: nil,
            velocityNorm: .nan,
            hasDiagnostics: hasDiagnostics
        )
    }

    func testNonStatsStepsDoNotPoisonDiagnosticMeans() {
        let box = TrainingLiveStatsBox(rollingWindow: 100)

        // One stats step with real diagnostics, then five non-stats steps whose
        // diagnostic fields are .nan. The loss differs so we can tell whether
        // the loss window accumulated every step (it should) vs. only the stats
        // step (it should not).
        box.recordStep(makeTiming(
            hasDiagnostics: true,
            policyLoss: 1.0,
            gradGlobalNorm: 3.0,
            policyEntropy: 2.0,
            valueMean: 0.5,
            advantageRaw: [0.1, -0.2, 0.3]
        ))
        for _ in 0..<5 {
            box.recordStep(makeTiming(
                hasDiagnostics: false,
                policyLoss: 2.0,
                gradGlobalNorm: 4.0
            ))
        }

        let snap = box.snapshot()

        // All six steps counted.
        XCTAssertEqual(snap.stats.steps, 6)

        // Diagnostic means reflect ONLY the single real (stats) sample — not
        // poisoned by the five .nan placeholders.
        XCTAssertEqual(try XCTUnwrap(snap.rollingPolicyEntropy), 2.0, accuracy: 1e-6)
        XCTAssertEqual(try XCTUnwrap(snap.rollingValueMean), 0.5, accuracy: 1e-6)

        // Every-step means reflect ALL six steps: (1.0 + 5·2.0)/6 = 1.8333…,
        // (3.0 + 5·4.0)/6 = 3.8333…. If gating wrongly skipped them on
        // non-stats steps these would read 1.0 / 3.0.
        XCTAssertEqual(try XCTUnwrap(snap.rollingPolicyLoss), 11.0 / 6.0, accuracy: 1e-5)
        XCTAssertEqual(try XCTUnwrap(snap.rollingGradGlobalNorm), 23.0 / 6.0, accuracy: 1e-5)

        // `lastTiming` retains the stats step (real entropy), never a non-stats
        // NaN step.
        let last = try? XCTUnwrap(snap.lastTiming)
        XCTAssertEqual(last?.policyEntropy, 2.0)
        XCTAssertTrue(last?.hasDiagnostics ?? false)
    }

    func testOnlyNonStatsStepsLeaveDiagnosticMeansEmpty() {
        let box = TrainingLiveStatsBox(rollingWindow: 100)
        for _ in 0..<4 {
            box.recordStep(makeTiming(hasDiagnostics: false, policyLoss: 1.0, gradGlobalNorm: 2.0))
        }
        let snap = box.snapshot()

        XCTAssertEqual(snap.stats.steps, 4)
        // No stats step ever recorded → diagnostic windows are empty (nil mean),
        // NOT a NaN.
        XCTAssertNil(snap.rollingPolicyEntropy)
        XCTAssertNil(snap.rollingValueMean)
        // Loss window still populated every step.
        XCTAssertEqual(try XCTUnwrap(snap.rollingPolicyLoss), 1.0, accuracy: 1e-6)
        // No stats step → no `lastTiming` surfaced.
        XCTAssertNil(snap.lastTiming)
    }
}
