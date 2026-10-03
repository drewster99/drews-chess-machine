//
//  DCMNormalMathTests.swift
//  DrewsChessMachineTests
//
//  The restricted-domain normal transform (determinism plan A5, D-3): exhaustive
//  accuracy over every grid input, golden normal pairs, and the draw count.
//

import XCTest
@testable import DrewsChessMachine

final class DCMNormalMathTests: XCTestCase {

    /// `ln((k + 1) / gridCount)` for every grid index, against Foundation's
    /// double `log`: within a few double ULPs relative everywhere, which is
    /// orders of magnitude finer than the `Float` the normals are rounded to.
    func testLogMatchesFoundationOnEveryGridInput() {
        let scale = Double(DCMNormalMath.gridCount)
        var worstRelative = 0.0
        var worstIndex: UInt32 = 0
        for k in 0..<UInt32(DCMNormalMath.gridCount) {
            let ours = DCMNormalMath.logOfGridUniform(k)
            let reference = Foundation.log(Double(k + 1) / scale)
            if reference == 0 {
                XCTAssertEqual(ours, 0, "ln(1) must be exactly 0")
                continue
            }
            let relative = abs(ours - reference) / abs(reference)
            if relative > worstRelative {
                worstRelative = relative
                worstIndex = k
            }
        }
        XCTAssertLessThanOrEqual(worstRelative, 4 * Double.ulpOfOne,
                                 "worst relative error \(worstRelative) at k=\(worstIndex)")
    }

    /// `(cos θ, sin θ)` for every grid angle, against Foundation's double
    /// `cos`/`sin`. Absolute error (both cross zero), bounded by a few ULPs of
    /// one.
    func testCosSinMatchFoundationOnEveryGridInput() {
        let turn = 2 * Double.pi / Double(DCMNormalMath.gridCount)
        var worst = 0.0
        var worstIndex: UInt32 = 0
        for k in 0..<UInt32(DCMNormalMath.gridCount) {
            let (c, s) = DCMNormalMath.cosSinOfGridAngle(k)
            let angle = Double(k) * turn
            let error = max(abs(c - Foundation.cos(angle)), abs(s - Foundation.sin(angle)))
            if error > worst {
                worst = error
                worstIndex = k
            }
        }
        XCTAssertLessThanOrEqual(worst, 8 * Double.ulpOfOne, "worst absolute error \(worst) at k=\(worstIndex)")
    }

    /// The grid's exact points come out exact: the quadrant boundaries.
    func testCosSinAtQuadrantBoundaries() {
        let quarter = UInt32(DCMNormalMath.gridCount / 4)
        XCTAssertTrue(DCMNormalMath.cosSinOfGridAngle(0) == (1, 0))
        XCTAssertTrue(DCMNormalMath.cosSinOfGridAngle(quarter) == (-0.0, 1))
        XCTAssertTrue(DCMNormalMath.cosSinOfGridAngle(2 * quarter) == (-1, -0.0))
        XCTAssertTrue(DCMNormalMath.cosSinOfGridAngle(3 * quarter) == (0, -1))
    }

    /// Each normal of a pair is the correctly rounded `Float` of the Box–Muller
    /// value computed with Foundation, to within one `Float` ULP — or, where
    /// the value is near zero, within a few double ULPs of the radius: at the
    /// grid's quarter turns the true cosine is exactly zero, which this
    /// transform returns, while Foundation's reference angle carries the
    /// rounding of `π` and so comes out a few double ULPs away from zero.
    func testNormalPairWithinOneFloatULPOfFoundation() {
        let scale = Double(DCMNormalMath.gridCount)
        let turn = 2 * Double.pi / scale
        let angleIndices: [UInt32] = [0, 1, 12_345, 4_194_303, 4_194_304, 9_999_999, 16_777_215]
        for radiusIndex in stride(from: UInt32(0), to: UInt32(DCMNormalMath.gridCount), by: 4099) {
            for angleIndex in angleIndices {
                let (first, second) = DCMNormalMath.standardNormalPair(radiusIndex: radiusIndex, angleIndex: angleIndex)
                let radius = (-2 * Foundation.log(Double(radiusIndex + 1) / scale)).squareRoot()
                let angle = Double(angleIndex) * turn
                let expectedFirst = Float(radius * Foundation.cos(angle))
                let expectedSecond = Float(radius * Foundation.sin(angle))
                let nearZero = Float(radius * 8 * Double.ulpOfOne)
                XCTAssertLessThanOrEqual(abs(first - expectedFirst), max(expectedFirst.ulp, nearZero),
                                         "k1=\(radiusIndex) k2=\(angleIndex)")
                XCTAssertLessThanOrEqual(abs(second - expectedSecond), max(expectedSecond.ulp, nearZero),
                                         "k1=\(radiusIndex) k2=\(angleIndex)")
            }
        }
    }

    /// Pinned bit patterns, computed by an independent Python implementation
    /// of the same generator, transform and operation order.
    func testNormalPairGoldenBits() {
        var fromFortyTwo = DCMRandom(seed: 42)
        let expectedFortyTwo: [(UInt32, UInt32)] = [
            (0xbfce7e1a, 0x3fc46a16), (0x3f481cf6, 0xbecce62b), (0x3c82046f, 0xbe025d79), (0x3ef455c0, 0xbf282161),
            (0xbf23b311, 0xbebd1151), (0xbe624c24, 0x3f5882c6), (0xbe93f262, 0x3f19a635), (0x3f180a25, 0xbf12d7ad),
        ]
        for (index, expected) in expectedFortyTwo.enumerated() {
            let (first, second) = fromFortyTwo.nextStandardNormalPair()
            XCTAssertEqual(first.bitPattern, expected.0, "seed 42 pair \(index) first")
            XCTAssertEqual(second.bitPattern, expected.1, "seed 42 pair \(index) second")
        }
        var fromWord = DCMRandom(seed: 0x0123_4567_89AB_CDEF)
        let expectedWord: [(UInt32, UInt32)] = [
            (0x3f71051e, 0x3e0ea4c1), (0x3f89d339, 0x3f5e37d9), (0x3db1f130, 0x3f048de8), (0xbf541d97, 0xbbb94983),
        ]
        for (index, expected) in expectedWord.enumerated() {
            let (first, second) = fromWord.nextStandardNormalPair()
            XCTAssertEqual(first.bitPattern, expected.0, "seed 0x0123456789ABCDEF pair \(index) first")
            XCTAssertEqual(second.bitPattern, expected.1, "seed 0x0123456789ABCDEF pair \(index) second")
        }
    }

    /// A pair always consumes exactly two draws, so a stream's position after
    /// N pairs is fixed whatever the values were.
    func testPairConsumesExactlyTwoDraws() {
        var viaPairs = DCMRandom(seed: 99)
        var viaDraws = DCMRandom(seed: 99)
        for _ in 0..<1000 {
            _ = viaPairs.nextStandardNormalPair()
            _ = viaDraws.next()
            _ = viaDraws.next()
        }
        XCTAssertEqual(viaPairs, viaDraws)
    }

    /// Distribution sanity: mean ≈ 0 and variance ≈ 1 over a large sample.
    func testNormalsHaveStandardMoments() {
        let values = WeightInitScheme.standardNormals(seed: 2024, count: 1 << 20)
        let count = Double(values.count)
        let mean = values.reduce(0.0) { $0 + Double($1) } / count
        let variance = values.reduce(0.0) { $0 + (Double($1) - mean) * (Double($1) - mean) } / count
        XCTAssertEqual(mean, 0, accuracy: 0.005)
        XCTAssertEqual(variance, 1, accuracy: 0.01)
        XCTAssertTrue(values.allSatisfy { $0.isFinite })
    }
}
