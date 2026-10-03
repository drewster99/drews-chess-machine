//
//  DCMNormalMath.swift
//  DrewsChessMachine
//
//  The standard-normal transform behind `DCMRandom.nextStandardNormalPair`
//  (determinism plan, A5 and decision D-3).
//

import Foundation

/// Box–Muller on a fixed grid of uniforms, computed with nothing but IEEE-754
/// double `+ − × ÷` and `sqrt` — operations every Apple chip and OS build
/// rounds identically — so a seed gives bit-identical normals everywhere.
///
/// Why not the system math library: vForce (`vvlogf`, `vvcosf`) and libm
/// (`log`, `cos`) are accurate only to about an ULP and their exact results may change
/// with an OS update or differ between chips. Weight initialization and move
/// sampling must reproduce from a recorded seed on every machine the lineage
/// runs on, so the transcendental functions here are our own.
///
/// The inputs are restricted by construction, which is what makes small,
/// exact range reductions possible:
/// - the radius uniform is `u = (k + 1) / gridCount` for a grid index `k`, so
///   it is never zero, never denormal and at most one — `ln u` is always
///   finite;
/// - the angle is `θ = 2π · k / gridCount`, so the quadrant and octant fold is
///   exact integer arithmetic on `k`.
///
/// Every function works in `Double` and is far more accurate than the `Float`
/// result needs (pinned by exhaustive tests over every grid input), so the one
/// `Float` rounding at the end decides the output. The operation order is
/// part of the init scheme: any change gives different bits and requires a new
/// scheme ID (see `WeightInitScheme`).
enum DCMNormalMath {
    /// Bits in each grid index: the top bits of a 64-bit draw.
    static let gridBits = 24
    /// Number of grid points.
    static let gridCount = 1 << gridBits

    // `ln 2`, the nearest double.
    private static let ln2 = 0.693_147_180_559_945_3
    // `√2`, the nearest double: mantissas above it are halved so the series
    // argument stays small.
    private static let sqrtTwo = 1.414_213_562_373_095_1

    /// `ln((k + 1) / gridCount)` for a grid index `k`.
    ///
    /// `k + 1 = 2^e · f` with `f` in `[1, 2)`, both exact (a power-of-two
    /// split); `f` is folded into `(√½, √2]` by an exact halving, then
    /// `ln f = 2·atanh(s)` with `s = (f − 1)/(f + 1)`, summed as the odd series
    /// `Σ s^(2n+1)/(2n+1)` in Horner form. On that interval `s` is small enough
    /// that the truncated tail is far below a double ULP.
    static func logOfGridUniform(_ k: UInt32) -> Double {
        precondition(Int(k) < gridCount, "logOfGridUniform: k \(k) is outside the 24-bit grid")
        let m = UInt64(k) + 1
        var exponent = 63 - m.leadingZeroBitCount
        var mantissa = Double(m) / Double(UInt64(1) << UInt64(exponent))
        if mantissa > sqrtTwo {
            mantissa *= 0.5
            exponent += 1
        }
        let s = (mantissa - 1) / (mantissa + 1)
        let s2 = s * s
        var series = 1.0 / 25.0
        series = series * s2 + 1.0 / 23.0
        series = series * s2 + 1.0 / 21.0
        series = series * s2 + 1.0 / 19.0
        series = series * s2 + 1.0 / 17.0
        series = series * s2 + 1.0 / 15.0
        series = series * s2 + 1.0 / 13.0
        series = series * s2 + 1.0 / 11.0
        series = series * s2 + 1.0 / 9.0
        series = series * s2 + 1.0 / 7.0
        series = series * s2 + 1.0 / 5.0
        series = series * s2 + 1.0 / 3.0
        series = series * s2 + 1.0
        let lnMantissa = 2 * s * series
        return Double(exponent - gridBits) * ln2 + lnMantissa
    }

    /// `(cos θ, sin θ)` for `θ = 2π · k / gridCount`.
    ///
    /// The quadrant is the index's top two bits; within it the index is folded
    /// onto the first octant by its exact complement against a quarter turn,
    /// so the Taylor series only ever see `x ≤ π/4`. The step `π / (gridCount/2)`
    /// is an exact power-of-two scaling of the double `π`, so `x` carries one
    /// rounding. The series run to the term whose successor is far below a
    /// double ULP on that interval.
    static func cosSinOfGridAngle(_ k: UInt32) -> (cos: Double, sin: Double) {
        precondition(Int(k) < gridCount, "cosSinOfGridAngle: k \(k) is outside the 24-bit grid")
        let quarter = UInt32(gridCount / 4)
        let eighth = UInt32(gridCount / 8)
        let quadrant = k / quarter
        let withinQuadrant = k % quarter
        let complemented = withinQuadrant > eighth
        let reduced = complemented ? quarter - withinQuadrant : withinQuadrant
        let x = Double(reduced) * (Double.pi / Double(gridCount / 2))
        let x2 = x * x

        var sinSeries = -1.0 / 121_645_100_408_832_000.0
        sinSeries = sinSeries * x2 + 1.0 / 355_687_428_096_000.0
        sinSeries = sinSeries * x2 - 1.0 / 1_307_674_368_000.0
        sinSeries = sinSeries * x2 + 1.0 / 6_227_020_800.0
        sinSeries = sinSeries * x2 - 1.0 / 39_916_800.0
        sinSeries = sinSeries * x2 + 1.0 / 362_880.0
        sinSeries = sinSeries * x2 - 1.0 / 5_040.0
        sinSeries = sinSeries * x2 + 1.0 / 120.0
        sinSeries = sinSeries * x2 - 1.0 / 6.0
        sinSeries = sinSeries * x2 + 1.0
        let sinX = x * sinSeries

        var cosSeries = 1.0 / 6_402_373_705_728_000.0
        cosSeries = cosSeries * x2 - 1.0 / 20_922_789_888_000.0
        cosSeries = cosSeries * x2 + 1.0 / 87_178_291_200.0
        cosSeries = cosSeries * x2 - 1.0 / 479_001_600.0
        cosSeries = cosSeries * x2 + 1.0 / 3_628_800.0
        cosSeries = cosSeries * x2 - 1.0 / 40_320.0
        cosSeries = cosSeries * x2 + 1.0 / 720.0
        cosSeries = cosSeries * x2 - 1.0 / 24.0
        cosSeries = cosSeries * x2 + 1.0 / 2.0
        let cosX = 1.0 - x2 * cosSeries

        // θ within the quadrant is either x or π/2 − x.
        let c = complemented ? sinX : cosX
        let s = complemented ? cosX : sinX
        switch quadrant {
        case 0: return (c, s)
        case 1: return (-s, c)
        case 2: return (-c, -s)
        default: return (s, -c)
        }
    }

    /// Two independent standard normals from a radius index and an angle
    /// index: `r = √(−2 ln u)`, then `(r cos θ, r sin θ)`, each rounded once
    /// to `Float`.
    static func standardNormalPair(radiusIndex: UInt32, angleIndex: UInt32) -> (Float, Float) {
        let radius = (-2 * logOfGridUniform(radiusIndex)).squareRoot()
        let (cosTheta, sinTheta) = cosSinOfGridAngle(angleIndex)
        return (Float(radius * cosTheta), Float(radius * sinTheta))
    }
}

extension DCMRandom {
    /// Two independent standard normals: the radius index is the top grid
    /// bits of one draw, the angle index the top grid bits of the next (see
    /// `DCMNormalMath`). Always exactly two draws, so a stream's position
    /// after N pairs is fixed.
    mutating func nextStandardNormalPair() -> (Float, Float) {
        let radiusIndex = UInt32(next() >> 40)
        let angleIndex = UInt32(next() >> 40)
        return DCMNormalMath.standardNormalPair(radiusIndex: radiusIndex, angleIndex: angleIndex)
    }
}
