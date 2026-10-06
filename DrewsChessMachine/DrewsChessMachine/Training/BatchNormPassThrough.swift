import Foundation

/// How much gradient a batch-norm channel passes through the activation it
/// feeds — the activation-aware "parked" classification `LayerHealth` and the
/// `dead_channels` training-health rule use for every activation function.
///
/// The β/|γ| dead / mostly-off bands of `LayerHealth` mean something only
/// for ReLU and leaky ReLU, whose derivative is a step: a channel whose
/// BN-normalized input z ~ N(0, 1) lands below zero after `γz + β` passes no
/// gradient (ReLU) or only the α leak (leaky ReLU), and Φ(β/|γ|) is the share
/// of inputs on the unit-slope side. SiLU and GELU have no step, so a tower of
/// them was unchecked. This type measures the same thing for every function:
///
/// - **pass-through** P = E|f'(γz + β)| over z ~ N(0, 1): the fraction of the
///   incoming gradient the channel passes back on average;
/// - **excess pass-through** X = (P − floor) / (1 − floor), where the floor is
///   what the function passes whatever its input (0 for ReLU, SiLU and GELU,
///   α for leaky ReLU). For ReLU and leaky ReLU X is exactly Φ(β/|γ|).
///
/// A channel is **parked** when X < Φ(−3) and **mostly off** when
/// Φ(−3) ≤ X < Φ(−2). For ReLU and leaky ReLU those are, by construction,
/// `LayerHealth`'s dead (β/|γ| < −3) and mostly-off (−3 ≤ β/|γ| < −2) bands —
/// `band` decides them from β/|γ| itself so the counts agree exactly, γ = 0
/// included (always on when β > 0, else dead). For SiLU, whose derivative is
/// not scale-invariant, the same X line sits at a β that depends on |γ|
/// (about −9 at |γ| = 1); a parked SiLU or GELU channel passes less than
/// Φ(−3) of its gradient and outputs about 0.
///
/// The Gaussian input is a model, not a measurement: BN makes each channel's
/// input mean 0 and variance 1 over its running statistics, but the true
/// distribution need not be normal.
///
/// This is the single source of the model and of its numerical integral;
/// `experiments/20261005-lr-schedule-ab/bn_liveness.py` mirrors it, and
/// `BatchNormPassThroughTests` checks the two agree on reference values the
/// Python produced and on arm B-silu's real checkpoints.
enum BatchNormPassThrough {

    /// X below this: parked. Φ(−3), `LayerHealth.deadBetaOverAbsGamma` as
    /// excess pass-through.
    static let parkedBelow: Double = standardNormalCDF(LayerHealth.deadBetaOverAbsGamma)
    /// X below this (and not parked): mostly off. Φ(−2),
    /// `LayerHealth.mostlyOffBetaOverAbsGamma` as excess pass-through.
    static let mostlyOffBelow: Double = standardNormalCDF(LayerHealth.mostlyOffBetaOverAbsGamma)

    enum Band: String, Sendable, Equatable {
        case parked
        case mostlyOff = "mostly_off"
        case passing
    }

    // MARK: Integration constants (SiLU, GELU)
    //
    // Y = γz + β ~ N(β, γ²) is integrated in y over β ± sigmaSpan·|γ|, clipped
    // to ±saturation; beyond it |f'(y)| equals 1 (above) or 0 (below) to well
    // under double precision, so the mass there is added in closed form.
    // Panels are at most `panelWidthY` wide in y and |γ| / `panelsPerSigma`
    // wide relative to the Gaussian, with a panel edge at f''s sign change so
    // |f'| is smooth inside each panel; each panel is a 16-point
    // Gauss–Legendre rule.

    static let sigmaSpan: Double = 9
    static let saturation: Double = 50
    static let panelWidthY: Double = 0.25
    static let panelsPerSigma: Double = 4
    static let gaussLegendrePointCount = 16

    /// The one y where SiLU's derivative changes sign (negative below it).
    static let siluDerivativeRoot: Double = derivativeRoot(of: siluDerivative, bracket: (-2, -1))
    /// The one y where exact GELU's derivative changes sign.
    static let geluDerivativeRoot: Double = derivativeRoot(of: geluDerivative, bracket: (-1, -0.5))

    /// 16-point Gauss–Legendre nodes and weights on [−1, 1], ascending.
    static let gaussLegendre: (nodes: [Double], weights: [Double]) = gaussLegendreRule(pointCount: gaussLegendrePointCount)

    // MARK: Model

    /// Whether `activation` has a pass-through model. Every function does;
    /// `does_not_apply` is not a function.
    static func isModeled(_ activation: ActivationFunction) -> Bool {
        switch activation {
        case .relu, .leakyRelu, .silu, .gelu: return true
        case .doesNotApply: return false
        }
    }

    /// What the function passes whatever its input.
    static func passThroughFloor(_ activation: ActivationFunction) -> Double {
        switch activation {
        case .leakyRelu: return ActivationFunction.leakyReLUNegativeSlope
        case .relu, .silu, .gelu, .doesNotApply: return 0
        }
    }

    /// X for one finite channel; γ = 0 is the constant input β.
    static func excessPassThrough(activation: ActivationFunction, gamma: Double, beta: Double) -> Double {
        switch activation {
        case .relu, .leakyRelu:
            if gamma == 0 { return beta > 0 ? 1 : 0 }
            return standardNormalCDF(beta / abs(gamma))
        case .silu:
            return smoothPassThrough(derivative: siluDerivative, root: siluDerivativeRoot, gamma: gamma, beta: beta)
        case .gelu:
            return smoothPassThrough(derivative: geluDerivative, root: geluDerivativeRoot, gamma: gamma, beta: beta)
        case .doesNotApply:
            preconditionFailure("BatchNormPassThrough: 'does_not_apply' is not an activation function; callers check isModeled")
        }
    }

    /// P = floor + (1 − floor)·X for one finite channel.
    static func passThrough(activation: ActivationFunction, gamma: Double, beta: Double) -> Double {
        let base = passThroughFloor(activation)
        return base + (1 - base) * excessPassThrough(activation: activation, gamma: gamma, beta: beta)
    }

    /// The parked / mostly-off band of one finite channel. ReLU and leaky
    /// ReLU are decided from β/|γ| with `LayerHealth`'s own comparisons, so
    /// their parked count equals the dead count exactly.
    static func band(activation: ActivationFunction, gamma: Double, beta: Double) -> Band {
        switch activation {
        case .relu, .leakyRelu:
            if gamma == 0 { return beta > 0 ? .passing : .parked }
            let ratio = beta / abs(gamma)
            if ratio < LayerHealth.deadBetaOverAbsGamma { return .parked }
            if ratio < LayerHealth.mostlyOffBetaOverAbsGamma { return .mostlyOff }
            return .passing
        case .silu, .gelu:
            let x = excessPassThrough(activation: activation, gamma: gamma, beta: beta)
            if x < parkedBelow { return .parked }
            if x < mostlyOffBelow { return .mostlyOff }
            return .passing
        case .doesNotApply:
            preconditionFailure("BatchNormPassThrough: 'does_not_apply' is not an activation function; callers check isModeled")
        }
    }

    // MARK: Functions

    /// Φ(x), accurate in the far lower tail (erfc, not 1 + erf).
    static func standardNormalCDF(_ x: Double) -> Double {
        0.5 * erfc(-x / 2.0.squareRoot())
    }

    /// 1 / (1 + e^−y) without overflow for any finite y.
    static func logistic(_ y: Double) -> Double {
        let e = exp(-abs(y))
        return y >= 0 ? 1 / (1 + e) : e / (1 + e)
    }

    static func siluDerivative(_ y: Double) -> Double {
        let s = logistic(y)
        return s * (1 + y * (1 - s))
    }

    /// Exact (erf) GELU, as the network builds it: d/dy [y·Φ(y)] = Φ(y) + y·φ(y).
    static func geluDerivative(_ y: Double) -> Double {
        standardNormalCDF(y) + y * exp(-0.5 * y * y) / (2 * Double.pi).squareRoot()
    }

    /// E|f'(Y)| for Y ~ N(β, γ²), f SiLU or GELU; γ = 0 is the constant input β.
    static func smoothPassThrough(
        derivative: (Double) -> Double,
        root: Double,
        gamma: Double,
        beta: Double
    ) -> Double {
        let sigma = abs(gamma)
        if sigma == 0 { return abs(derivative(beta)) }
        let lo = max(beta - sigmaSpan * sigma, -saturation)
        let hi = min(beta + sigmaSpan * sigma, saturation)
        let saturatedMass = hi == saturation ? standardNormalCDF((beta - saturation) / sigma) : 0
        if lo >= hi { return saturatedMass }
        let width = min(panelWidthY, sigma / panelsPerSigma)
        let edges = lo < root && root < hi ? [lo, root, hi] : [lo, hi]
        let normalization = 1 / (sigma * (2 * Double.pi).squareRoot())
        let (nodes, weights) = gaussLegendre
        var total: Double = 0
        for segment in 0..<(edges.count - 1) {
            let a = edges[segment]
            let b = edges[segment + 1]
            let panels = max(1, Int(((b - a) / width).rounded(.up)))
            let step = (b - a) / Double(panels)
            for panel in 0..<panels {
                let left = a + Double(panel) * step
                let right = panel == panels - 1 ? b : a + Double(panel + 1) * step
                let middle = 0.5 * (left + right)
                let half = 0.5 * (right - left)
                for index in 0..<nodes.count {
                    let y = middle + half * nodes[index]
                    let z = (y - beta) / sigma
                    total += abs(derivative(y)) * exp(-0.5 * z * z) * normalization * half * weights[index]
                }
            }
        }
        return total + saturatedMass
    }

    // MARK: Numerical helpers

    /// The one sign change of `f` inside `bracket`, by bisection (f < 0
    /// below it). The brackets are fixed and checked by the tests.
    static func derivativeRoot(of f: (Double) -> Double, bracket: (Double, Double)) -> Double {
        var lo = bracket.0
        var hi = bracket.1
        for _ in 0..<200 {
            let mid = 0.5 * (lo + hi)
            if f(mid) < 0 {
                lo = mid
            } else {
                hi = mid
            }
        }
        return 0.5 * (lo + hi)
    }

    /// Gauss–Legendre nodes and weights on [−1, 1] by Newton iteration on the
    /// Legendre polynomial, ascending.
    static func gaussLegendreRule(pointCount n: Int) -> (nodes: [Double], weights: [Double]) {
        var nodes = [Double](repeating: 0, count: n)
        var weights = [Double](repeating: 0, count: n)
        let half = (n + 1) / 2
        for i in 1...half {
            var z = cos(Double.pi * (Double(i) - 0.25) / (Double(n) + 0.5))
            var derivative: Double = 0
            for _ in 0..<100 {
                var p1: Double = 1
                var p2: Double = 0
                for j in 1...n {
                    let p3 = p2
                    p2 = p1
                    p1 = ((2 * Double(j) - 1) * z * p2 - (Double(j) - 1) * p3) / Double(j)
                }
                derivative = Double(n) * (z * p1 - p2) / (z * z - 1)
                let previous = z
                z = previous - p1 / derivative
                if abs(z - previous) <= 1e-15 { break }
            }
            nodes[i - 1] = -z
            nodes[n - i] = z
            let weight = 2 / ((1 - z * z) * derivative * derivative)
            weights[i - 1] = weight
            weights[n - i] = weight
        }
        return (nodes, weights)
    }
}
