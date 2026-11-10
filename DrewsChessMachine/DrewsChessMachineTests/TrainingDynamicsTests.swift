import XCTest
@testable import DrewsChessMachine

/// Rules of the `[DYNAMICS]` accumulator (`TrainingDynamics.swift`).
final class TrainingDynamicsTests: XCTestCase {

    /// One tensor's six sums in `TrainingDynamicsSums` order.
    private func sums(g2: Float, gv: Float, vOld2: Float, vNew2: Float, dw2: Float, w2: Float) -> [Float] {
        [g2, gv, vOld2, vNew2, dw2, w2]
    }

    // MARK: - Layout

    /// Protects: a BN's γ/β and an FC's weight/bias report as one layer; other names stand alone.
    /// Fails if: the suffix rule changes or layers are emitted out of first-appearance order.
    func testLayoutGroupsWeightAndBiasIntoOneLayerInFirstAppearanceOrder() {
        let layout = TrainingDynamicsLayout(tensorNames: [
            "stem.conv.weight", "stem.bn.weight", "stem.bn.bias",
            "blocks.0.se_attenuate.fc1.weight", "blocks.0.se_attenuate.fc1.bias",
            "value.fc1.weight", "value.fc1.bias", "odd_tensor"
        ])
        XCTAssertEqual(layout.layerNames,
                       ["stem.conv", "stem.bn", "blocks.0.se_attenuate.fc1", "value.fc1", "odd_tensor"])
        XCTAssertEqual(layout.layerIndexOfTensor, [0, 1, 1, 2, 2, 3, 3, 4])
    }

    /// Protects: a name that is only the suffix is not reduced to an empty layer name.
    /// Fails if: `.weight` alone becomes "".
    func testLayerNameKeepsANameThatIsOnlyTheSuffix() {
        XCTAssertEqual(TrainingDynamicsLayout.layerName(ofTensor: ".weight"), ".weight")
        XCTAssertEqual(TrainingDynamicsLayout.layerName(ofTensor: "a.weights"), "a.weights")
    }

    // MARK: - Recording

    /// Protects: a step vector of the wrong length is a layout bug and is refused.
    /// Fails if: a short or long vector is silently accepted.
    func testRecordRefusesAWrongLengthVector() {
        var window = TrainingDynamicsWindow(layout: TrainingDynamicsLayout(tensorNames: ["a.weight"]))
        XCTAssertThrowsError(try window.record(stepSums: [1, 2, 3])) { error in
            XCTAssertEqual(error as? TrainingDynamicsWindow.RecordError, .wrongLength(expected: 6, got: 3))
        }
        XCTAssertEqual(window.steps, 0)
    }

    /// Protects: a step with any non-finite sum adds nothing and is counted.
    /// Fails if: a NaN reaches the means, or the step is dropped without a count.
    func testNonFiniteStepIsCountedAndNotAccumulated() throws {
        var window = TrainingDynamicsWindow(layout: TrainingDynamicsLayout(tensorNames: ["a.weight"]))
        try window.record(stepSums: sums(g2: .nan, gv: 0, vOld2: 1, vNew2: 1, dw2: 1, w2: 1))
        XCTAssertEqual(window.steps, 0)
        XCTAssertEqual(window.nonFiniteSteps, 1)
        XCTAssertEqual(window.line(trainerStep: 7),
                       "[DYNAMICS] trainerStep=7 steps=0 nonFinite=1 all=g:-,v:-,dww:-,cos:- a=g:-,v:-,dww:-,cos:-")
    }

    /// Protects: mean and max of ‖g‖, mean ‖v‖, mean and max ‖Δw‖/‖w‖ over the window.
    /// Fails if: a mean is taken over squares, or the max is not the per-step max.
    func testMeansAndMaxesOverSteps() throws {
        var window = TrainingDynamicsWindow(layout: TrainingDynamicsLayout(tensorNames: ["a.weight"]))
        // ‖g‖ 1 then 3; ‖v‖ 2 then 4; ‖Δw‖/‖w‖ 0.5 then 0.25.
        try window.record(stepSums: sums(g2: 1, gv: 0, vOld2: 0, vNew2: 4, dw2: 4, w2: 16))
        try window.record(stepSums: sums(g2: 9, gv: 0, vOld2: 0, vNew2: 16, dw2: 1, w2: 16))
        let layer = window.layers[0]
        XCTAssertEqual(layer.steps, 2)
        XCTAssertEqual(layer.gradientNormSum / 2, 2, accuracy: 1e-12)
        XCTAssertEqual(layer.gradientNormMax, 3, accuracy: 1e-12)
        XCTAssertEqual(layer.velocityNormSum / 2, 3, accuracy: 1e-12)
        XCTAssertEqual(layer.relativeUpdateSum / 2, 0.375, accuracy: 1e-12)
        XCTAssertEqual(layer.relativeUpdateMax, 0.5, accuracy: 1e-12)
    }

    /// Protects: cos(g, v_old) — parallel 1, opposed −1, undefined when either norm is 0.
    /// Fails if: the cosine uses v_new, is not normalized, or a zero momentum counts as 0.
    func testCosineEdgeCases() throws {
        let layout = TrainingDynamicsLayout(tensorNames: ["a.weight"])
        // Parallel: g = v_old = (1, 0): Σg·v = 1, norms 1.
        var parallel = TrainingDynamicsWindow(layout: layout)
        try parallel.record(stepSums: sums(g2: 1, gv: 1, vOld2: 1, vNew2: 1, dw2: 0, w2: 1))
        XCTAssertEqual(parallel.layers[0].cosineSum, 1, accuracy: 1e-12)
        // Opposed, with magnitudes: g = (2, 0), v_old = (−3, 0): Σg·v = −6.
        var opposed = TrainingDynamicsWindow(layout: layout)
        try opposed.record(stepSums: sums(g2: 4, gv: -6, vOld2: 9, vNew2: 1, dw2: 0, w2: 1))
        XCTAssertEqual(opposed.layers[0].cosineSum, -1, accuracy: 1e-12)
        // Zero incoming momentum (first step after a reset): undefined.
        var fresh = TrainingDynamicsWindow(layout: layout)
        try fresh.record(stepSums: sums(g2: 1, gv: 0, vOld2: 0, vNew2: 1, dw2: 0, w2: 1))
        XCTAssertEqual(fresh.layers[0].cosineSteps, 0)
        XCTAssertTrue(TrainingDynamicsWindow.fields(fresh.layers[0]).hasSuffix("cos:-"))
        // Rounding past ±1 is clamped.
        var rounded = TrainingDynamicsWindow(layout: layout)
        try rounded.record(stepSums: sums(g2: 1, gv: 1.000001, vOld2: 1, vNew2: 1, dw2: 0, w2: 1))
        XCTAssertEqual(rounded.layers[0].cosineSum, 1, accuracy: 0)
    }

    /// Protects: zero weights leave the relative update undefined rather than infinite.
    /// Fails if: ‖Δw‖/0 reaches the line.
    func testZeroWeightNormLeavesRelativeUpdateUndefined() throws {
        var window = TrainingDynamicsWindow(layout: TrainingDynamicsLayout(tensorNames: ["a.weight"]))
        try window.record(stepSums: sums(g2: 1, gv: 0, vOld2: 0, vNew2: 1, dw2: 1, w2: 0))
        XCTAssertEqual(window.layers[0].relativeUpdateSteps, 0)
        XCTAssertTrue(TrainingDynamicsWindow.fields(window.layers[0]).contains("dww:-"))
    }

    /// Protects: a layer's sums are the sums over its tensors (norms of the
    /// concatenation, not sums of norms), and `all` covers every tensor.
    /// Fails if: per-tensor norms are added, or `all` misses a layer.
    func testLayerAndWholeNetworkCombineSumsBeforeTakingNorms() throws {
        let layout = TrainingDynamicsLayout(tensorNames: ["a.weight", "a.bias", "b.weight"])
        var window = TrainingDynamicsWindow(layout: layout)
        try window.record(stepSums:
            sums(g2: 9, gv: 0, vOld2: 0, vNew2: 0, dw2: 0, w2: 1)
            + sums(g2: 16, gv: 0, vOld2: 0, vNew2: 0, dw2: 0, w2: 1)
            + sums(g2: 144, gv: 0, vOld2: 0, vNew2: 0, dw2: 0, w2: 1))
        XCTAssertEqual(window.layers[0].gradientNormSum, 5, accuracy: 1e-12)   // √(9 + 16)
        XCTAssertEqual(window.layers[1].gradientNormSum, 12, accuracy: 1e-12)
        XCTAssertEqual(window.network.gradientNormSum, 13, accuracy: 1e-12)    // √169
    }

    // MARK: - Line and drain

    /// Protects: the line's exact shape (dashboards parse it) and nil for an empty window.
    /// Fails if: a field is renamed, reordered or a layer is omitted.
    func testLineFormat() throws {
        var window = TrainingDynamicsWindow(layout: TrainingDynamicsLayout(tensorNames: ["a.weight"]))
        XCTAssertNil(window.line(trainerStep: 1))
        try window.record(stepSums: sums(g2: 4, gv: 2, vOld2: 1, vNew2: 9, dw2: 0.01, w2: 100))
        XCTAssertEqual(window.line(trainerStep: 1000),
                       "[DYNAMICS] trainerStep=1000 steps=1 nonFinite=0 all=g:2/2,v:3,dww:0.01/0.01,cos:1 a=g:2/2,v:3,dww:0.01/0.01,cos:1")
    }

    /// Protects: draining returns everything since the previous drain and starts a fresh window.
    /// Fails if: a drain repeats old steps or loses the ones recorded after it.
    func testDrainResetsTheWindow() throws {
        let log = TrainingDynamicsLog(layout: TrainingDynamicsLayout(tensorNames: ["a.weight"]))
        try log.record(stepSums: sums(g2: 1, gv: 0, vOld2: 0, vNew2: 1, dw2: 0, w2: 1))
        try log.record(stepSums: sums(g2: 1, gv: 0, vOld2: 0, vNew2: 1, dw2: 0, w2: 1))
        XCTAssertTrue(log.drainLine(trainerStep: 2)?.contains("steps=2") == true)
        XCTAssertNil(log.drainLine(trainerStep: 2))
        try log.record(stepSums: sums(g2: 1, gv: 0, vOld2: 0, vNew2: 1, dw2: 0, w2: 1))
        XCTAssertTrue(log.drainLine(trainerStep: 3)?.contains("steps=1") == true)
    }
}
