//
//  SharpnessVectorMathTests.swift
//  DrewsChessMachineTests
//
//  The `--sharpness` vector math (`SharpnessVectorMath`): dot products and
//  norms over tensor-by-tensor vectors, axpy, normalization, the
//  central-difference accumulation, the Rayleigh quotient, the convergence
//  test, group masking, the seeded Gaussian start vector, and the
//  trainable-tensor group classification (`SharpnessParameterGroup`). Pure:
//  no Metal, no files. Ends with a power iteration on a known symmetric
//  matrix driven through the same helpers the CLI uses.
//

import Foundation
import XCTest
@testable import DrewsChessMachine

final class SharpnessVectorMathTests: XCTestCase {

    // MARK: - Reductions

    /// Protects: the dot product sums every element of every tensor.
    /// Fails if: a tensor or a tail element is skipped, or the tensors are
    /// paired out of order.
    func testDotSumsEveryElementOfEveryTensor() throws {
        let a: [[Float]] = [[1, 2], [3], [4, 5, 6]]
        let b: [[Float]] = [[7, 8], [9], [10, 11, 12]]
        XCTAssertEqual(try SharpnessVectorMath.dot(a, b), 7 + 16 + 27 + 40 + 55 + 72)
    }

    /// Protects: reductions accumulate in Double, so many small addends are
    /// not lost against a large running total.
    /// Fails if: the sum runs in Float (2^24 + 1 + 1 + … stays 2^24 in Float).
    func testDotAccumulatesInDouble() throws {
        let count = 1_000
        let a: [[Float]] = [[Float(1 << 24)] + [Float](repeating: 1, count: count)]
        let ones: [[Float]] = [[Float](repeating: 1, count: count + 1)]
        XCTAssertEqual(try SharpnessVectorMath.dot(a, ones), Double(1 << 24) + Double(count))
    }

    /// Protects: vectors with different tensor counts or element counts are
    /// refused, never silently truncated.
    /// Fails if: the layout check is missing from `dot`, `adding`,
    /// `difference` or `accumulateDifference`.
    func testMismatchedLayoutsAreRefused() {
        let a: [[Float]] = [[1, 2], [3]]
        let fewerTensors: [[Float]] = [[1, 2]]
        let shorterTensor: [[Float]] = [[1, 2], []]
        XCTAssertThrowsError(try SharpnessVectorMath.dot(a, fewerTensors))
        XCTAssertThrowsError(try SharpnessVectorMath.dot(a, shorterTensor))
        XCTAssertThrowsError(try SharpnessVectorMath.adding(a, scaledBy: 1, to: shorterTensor))
        XCTAssertThrowsError(try SharpnessVectorMath.difference(a, fewerTensors))
        var accumulator = SharpnessVectorMath.zeroAccumulator(shapedLike: a)
        XCTAssertThrowsError(try SharpnessVectorMath.accumulateDifference(into: &accumulator, plus: a, minus: shorterTensor,
                                                                          scale: 1))
        var shortAccumulator = SharpnessVectorMath.zeroAccumulator(shapedLike: fewerTensors)
        XCTAssertThrowsError(try SharpnessVectorMath.accumulate(into: &shortAccumulator, a, scale: 1))
    }

    /// Protects: the norm is the Euclidean norm over all elements.
    /// Fails if: it is per-tensor, squared, or skips a tensor.
    func testNormIsEuclideanOverAllElements() {
        XCTAssertEqual(SharpnessVectorMath.norm([[3], [0, 4]]), 5)
        XCTAssertEqual(SharpnessVectorMath.norm([[], []]), 0)
    }

    // MARK: - Linear combinations

    /// Protects: `adding(x, scaledBy: α, to: y)` is y + αx, element by element.
    /// Fails if: the roles of x and y are swapped or α is applied to y.
    func testAddingIsAxpy() throws {
        let x: [[Float]] = [[1, -2], [0.5]]
        let y: [[Float]] = [[10, 20], [30]]
        XCTAssertEqual(try SharpnessVectorMath.adding(x, scaledBy: -2, to: y), [[8, 24], [29]])
    }

    /// Protects: normalization gives a unit vector in the same direction and
    /// returns the norm it divided by.
    /// Fails if: it divides by the squared norm or loses the sign.
    func testNormalizedIsAUnitVectorInTheSameDirection() throws {
        let (unit, norm) = try SharpnessVectorMath.normalized([[0, -3], [4]])
        XCTAssertEqual(norm, 5)
        XCTAssertEqual(unit, [[0, -0.6], [0.8]])
        XCTAssertEqual(SharpnessVectorMath.norm(unit), 1, accuracy: 1e-7)
    }

    /// Protects: a zero or non-finite vector is refused rather than turned
    /// into zeros or NaNs that power iteration would read as λ = 0 or NaN.
    /// Fails if: `normalized` divides by zero or infinity without checking.
    func testNormalizingAZeroOrNonFiniteVectorThrows() {
        XCTAssertThrowsError(try SharpnessVectorMath.normalized([[0, 0], [0]]))
        XCTAssertThrowsError(try SharpnessVectorMath.normalized([[Float.nan]]))
        XCTAssertThrowsError(try SharpnessVectorMath.normalized([[Float.infinity, 1]]))
    }

    // MARK: - Central differences

    /// Protects: K batches each adding (plus − minus)·1/(2εK) leave the mean
    /// central difference in the accumulator.
    /// Fails if: the difference's sign is flipped, the scale is applied
    /// twice, or later batches overwrite instead of adding.
    func testAccumulatedCentralDifferencesAreTheMeanOverBatches() throws {
        let epsilon = 0.5
        let batches = 2
        let scale = 1 / (2 * epsilon * Double(batches))
        var sum = SharpnessVectorMath.zeroAccumulator(shapedLike: [[0, 0], [0]])
        try SharpnessVectorMath.accumulateDifference(into: &sum, plus: [[3, 1], [2]], minus: [[1, 1], [0]], scale: scale)
        try SharpnessVectorMath.accumulateDifference(into: &sum, plus: [[5, 0], [4]], minus: [[1, 2], [0]], scale: scale)
        // Batch 0: (2, 0, 2) / 1; batch 1: (4, −2, 4) / 1; mean (3, −1, 3).
        XCTAssertEqual(SharpnessVectorMath.narrowed(sum), [[3, -1], [3]])
    }

    /// Protects: the central difference of the gradient of a quadratic
    /// ½wᵀAw is exactly A·v (no truncation error at any ε).
    /// Fails if: the difference is one-sided or divided by ε instead of 2ε.
    func testCentralDifferenceOfAQuadraticsGradientIsTheMatrixProduct() throws {
        let matrix: [[Double]] = [[2, 1, 0], [1, 3, -1], [0, -1, 4]]
        func gradient(_ w: [[Float]]) -> [[Float]] {
            let flat = w.flatMap { $0 }.map(Double.init)
            let product = matrix.map { row in zip(row, flat).reduce(0) { $0 + $1.0 * $1.1 } }
            return [[Float(product[0]), Float(product[1])], [Float(product[2])]]
        }
        let w: [[Float]] = [[0.25, -0.5], [1]]
        let v: [[Float]] = [[1, 0], [0]]
        let epsilon = 0.125
        var sum = SharpnessVectorMath.zeroAccumulator(shapedLike: w)
        try SharpnessVectorMath.accumulateDifference(
            into: &sum,
            plus: gradient(try SharpnessVectorMath.adding(v, scaledBy: epsilon, to: w)),
            minus: gradient(try SharpnessVectorMath.adding(v, scaledBy: -epsilon, to: w)),
            scale: 1 / (2 * epsilon))
        XCTAssertEqual(SharpnessVectorMath.narrowed(sum), [[2, 1], [0]])
    }

    /// Protects: `accumulate` with scale 1/K builds the mean of K vectors.
    /// Fails if: it overwrites or ignores the scale.
    func testAccumulateBuildsTheMean() throws {
        var sum = SharpnessVectorMath.zeroAccumulator(shapedLike: [[0], [0, 0]])
        try SharpnessVectorMath.accumulate(into: &sum, [[2], [4, -6]], scale: 0.5)
        try SharpnessVectorMath.accumulate(into: &sum, [[4], [0, 2]], scale: 0.5)
        XCTAssertEqual(SharpnessVectorMath.narrowed(sum), [[3], [2, -2]])
    }

    // MARK: - Rayleigh quotient and convergence

    /// Protects: the Rayleigh quotient is vᵀ(Hv)/vᵀv, so a non-unit vector
    /// gives the same estimate as its unit direction.
    /// Fails if: the denominator is dropped or is ‖v‖ instead of ‖v‖².
    func testRayleighQuotientIsScaleInvariant() throws {
        let vector: [[Float]] = [[2, 0], [0]]
        let product: [[Float]] = [[14, 0], [0]]   // H·v with eigenvalue 7
        XCTAssertEqual(try SharpnessVectorMath.rayleighQuotient(vector: vector, product: product), 7)
    }

    /// Protects: a zero vector has no Rayleigh quotient.
    /// Fails if: 0/0 is returned as NaN instead of an error.
    func testRayleighQuotientOfAZeroVectorThrows() {
        XCTAssertThrowsError(try SharpnessVectorMath.rayleighQuotient(vector: [[0]], product: [[1]]))
    }

    /// Protects: convergence is relative to the newer estimate's magnitude,
    /// works for negative estimates, and never accepts a non-finite one.
    /// Fails if: the test is absolute, uses the wrong side of the
    /// inequality, or lets NaN through (NaN comparisons are false either way,
    /// so the guard matters only in the true direction).
    func testConvergenceIsRelativeAndRejectsNonFinite() {
        XCTAssertTrue(SharpnessVectorMath.hasConverged(previous: 100, current: 100.05, relativeTolerance: 1e-3))
        XCTAssertFalse(SharpnessVectorMath.hasConverged(previous: 100, current: 100.2, relativeTolerance: 1e-3))
        XCTAssertTrue(SharpnessVectorMath.hasConverged(previous: -50, current: -50.01, relativeTolerance: 1e-3))
        XCTAssertFalse(SharpnessVectorMath.hasConverged(previous: 0.001, current: 0.002, relativeTolerance: 1e-3))
        XCTAssertTrue(SharpnessVectorMath.hasConverged(previous: 0, current: 0, relativeTolerance: 1e-3))
        XCTAssertFalse(SharpnessVectorMath.hasConverged(previous: .nan, current: 1, relativeTolerance: 1))
        XCTAssertFalse(SharpnessVectorMath.hasConverged(previous: .infinity, current: .infinity, relativeTolerance: 1))
    }

    // MARK: - Groups and masking

    /// Protects: masking zeroes exactly the tensors outside the group and
    /// keeps the layout.
    /// Fails if: it drops tensors (breaking the layout) or zeroes the kept ones.
    func testMaskingZeroesTensorsOutsideTheGroup() throws {
        let x: [[Float]] = [[1, 2], [3], [4, 5]]
        XCTAssertEqual(try SharpnessVectorMath.masked(x, keeping: [0, 2]), [[1, 2], [0], [4, 5]])
    }

    /// Protects: a group that selects no tensor of the vector is an error.
    /// Fails if: power iteration could start on an all-zero subspace.
    func testMaskingWithAnEmptyGroupThrows() {
        XCTAssertThrowsError(try SharpnessVectorMath.masked([[1]], keeping: []))
        XCTAssertThrowsError(try SharpnessVectorMath.masked([[1]], keeping: [5]))
    }

    /// Protects: every trainable tensor of a real architecture belongs to
    /// exactly one group, and each family lands where the doc says.
    /// Fails if: a tensor family is unclassified, or BN affine parameters in
    /// the heads are counted as head weights.
    func testEveryTrainableTensorOfThePresetsHasOneGroup() throws {
        for preset in NetworkArchitecture.Preset.allCases {
            let plan = NetworkArchitecture.preset(preset).trainableTensorPlan()
            let groups = try SharpnessParameterGroup.tensorIndices(of: plan)
            let assigned = groups.values.reduce(0) { $0 + $1.count }
            XCTAssertEqual(assigned, plan.count, "\(preset.rawValue)")
            XCTAssertEqual(Set(groups.values.flatMap { $0 }), Set(plan.indices), "\(preset.rawValue)")
        }
        func group(_ name: String, _ kind: WeightKind) throws -> SharpnessParameterGroup {
            try SharpnessParameterGroup.group(of: WeightTensorSpec(name: name, shape: [1], kind: kind))
        }
        XCTAssertEqual(try group("stem.conv.weight", .conv), .stem)
        XCTAssertEqual(try group("stem.bn.weight", .bnAffine), .normAffine)
        XCTAssertEqual(try group("blocks.2.conv1.weight", .conv), .tower)
        XCTAssertEqual(try group("blocks.0.se_scalebias.fc1.bias", .bias), .tower)
        XCTAssertEqual(try group("blocks.1.rezero_alpha", .scalar), .tower)
        XCTAssertEqual(try group("blocks.1.res_ln.weight", .bnAffine), .normAffine)
        XCTAssertEqual(try group("feature_skip.conv.weight", .conv), .tower)
        XCTAssertEqual(try group("policy.conv.bias", .bias), .policyHead)
        XCTAssertEqual(try group("policy.pre_bn.bias", .bnAffine), .normAffine)
        XCTAssertEqual(try group("value.wdl_fc2.weight", .linear), .valueHead)
    }

    /// Protects: an unknown tensor family or a running statistic is refused,
    /// never put in some group by default.
    /// Fails if: classification has a catch-all.
    func testUnknownTensorsAndRunningStatisticsAreRefused() {
        XCTAssertThrowsError(try SharpnessParameterGroup.group(
            of: WeightTensorSpec(name: "mystery.weight", shape: [1], kind: .conv)))
        XCTAssertThrowsError(try SharpnessParameterGroup.group(
            of: WeightTensorSpec(name: "stem.bn.running_mean", shape: [1], kind: .bnRunningStat)))
    }

    // MARK: - Start vector

    /// Protects: the start vector is a pure function of the stream state, is
    /// zero outside the group, and covers odd-length tensors.
    /// Fails if: it draws from an unseeded source, skips a tensor's last
    /// element, or fills tensors outside the group.
    func testGaussianStartVectorIsSeededAndMasked() throws {
        let layout: [[Float]] = [[0, 0, 0], [0, 0], [0]]
        var first = DCMRandom(seed: 42)
        var second = DCMRandom(seed: 42)
        var other = DCMRandom(seed: 43)
        let a = try SharpnessVectorMath.gaussianVector(shapedLike: layout, keeping: [0, 2], using: &first)
        let b = try SharpnessVectorMath.gaussianVector(shapedLike: layout, keeping: [0, 2], using: &second)
        let c = try SharpnessVectorMath.gaussianVector(shapedLike: layout, keeping: [0, 2], using: &other)
        XCTAssertEqual(a, b)
        XCTAssertNotEqual(a, c)
        XCTAssertEqual(a[1], [0, 0])
        XCTAssertTrue(a[0].allSatisfy { $0 != 0 && $0.isFinite })
        XCTAssertTrue(a[2][0] != 0 && a[2][0].isFinite)
    }

    /// Protects: the draw is roughly standard normal (mean 0, variance 1),
    /// so no direction is favoured.
    /// Fails if: Box–Muller is mis-scaled (variance 2, or uniform output).
    func testGaussianStartVectorHasUnitVariance() throws {
        let count = 200_000
        var random = DCMRandom(seed: 7)
        let draw = try SharpnessVectorMath.gaussianVector(
            shapedLike: [[Float](repeating: 0, count: count)], keeping: [0], using: &random)[0]
        let mean = draw.reduce(0.0) { $0 + Double($1) } / Double(count)
        let variance = draw.reduce(0.0) { $0 + (Double($1) - mean) * (Double($1) - mean) } / Double(count)
        XCTAssertEqual(mean, 0, accuracy: 0.01)
        XCTAssertEqual(variance, 1, accuracy: 0.02)
    }

    // MARK: - Difference

    /// Protects: the determinism check reports the largest gap and bit
    /// identity, and tells −0 from +0 in bits.
    /// Fails if: it compares values instead of bits, or misses the max.
    func testDifferenceReportsMaxGapAndBitIdentity() throws {
        let same = try SharpnessVectorMath.difference([[1, 2], [3]], [[1, 2], [3]])
        XCTAssertEqual(same.maxAbsolute, 0)
        XCTAssertTrue(same.bitIdentical)
        let signedZero = try SharpnessVectorMath.difference([[0]], [[-0.0]])
        XCTAssertEqual(signedZero.maxAbsolute, 0)
        XCTAssertFalse(signedZero.bitIdentical)
        let apart = try SharpnessVectorMath.difference([[1, 2], [3]], [[1, 2.5], [2]])
        XCTAssertEqual(apart.maxAbsolute, 1)
        XCTAssertFalse(apart.bitIdentical)
    }

    /// Protects: a NaN in either vector shows up as a NaN gap, not zero.
    /// Fails if: `max` silently skips NaN.
    func testDifferencePropagatesNaN() throws {
        XCTAssertTrue(try SharpnessVectorMath.difference([[1, .nan]], [[1, 1]]).maxAbsolute.isNaN)
    }

    // MARK: - End to end

    /// Protects: power iteration through these helpers, on a central-difference
    /// product of a quadratic's gradient, finds the largest eigenvalue of a
    /// known symmetric matrix, and restricted to a group finds that block's.
    /// Fails if: the helpers compose wrongly (wrong normalization, product
    /// not re-projected onto the group, Rayleigh quotient on the wrong pair).
    func testPowerIterationFindsTheLargestEigenvalue() throws {
        // Block diagonal: tensor 0 is a 2×2 block with eigenvalues 1 and 3,
        // tensor 1 a 1×1 block with eigenvalue 5.
        let matrix: [[Double]] = [[2, 1, 0], [1, 2, 0], [0, 0, 5]]
        func gradient(_ w: [[Float]]) -> [[Float]] {
            let flat = w.flatMap { $0 }.map(Double.init)
            let product = matrix.map { row in zip(row, flat).reduce(0) { $0 + $1.0 * $1.1 } }
            return [[Float(product[0]), Float(product[1])], [Float(product[2])]]
        }
        let w: [[Float]] = [[0.3, -0.2], [0.7]]
        func largestEigenvalue(keeping group: Set<Int>) throws -> Double {
            var random = DCMRandom(seed: 1)
            var vector = try SharpnessVectorMath.normalized(
                SharpnessVectorMath.gaussianVector(shapedLike: w, keeping: group, using: &random)).unit
            var lambda = 0.0
            for _ in 0..<200 {
                let epsilon = 0.01
                var sum = SharpnessVectorMath.zeroAccumulator(shapedLike: w)
                try SharpnessVectorMath.accumulateDifference(
                    into: &sum,
                    plus: gradient(try SharpnessVectorMath.adding(vector, scaledBy: epsilon, to: w)),
                    minus: gradient(try SharpnessVectorMath.adding(vector, scaledBy: -epsilon, to: w)),
                    scale: 1 / (2 * epsilon))
                let product = try SharpnessVectorMath.masked(SharpnessVectorMath.narrowed(sum), keeping: group)
                lambda = try SharpnessVectorMath.rayleighQuotient(vector: vector, product: product)
                vector = try SharpnessVectorMath.normalized(product).unit
            }
            return lambda
        }
        XCTAssertEqual(try largestEigenvalue(keeping: [0, 1]), 5, accuracy: 1e-3)
        XCTAssertEqual(try largestEigenvalue(keeping: [0]), 3, accuracy: 1e-3)
    }
}
