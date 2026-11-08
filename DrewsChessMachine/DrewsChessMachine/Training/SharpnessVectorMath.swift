import Foundation

/// The vector math of `--sharpness` (`SharpnessCLI`): the trainable
/// parameters as one flat vector held tensor by tensor (`[[Float]]`, the
/// layout the trainer exports weights and velocity in), and the pieces of a
/// power iteration on finite-difference Hessian-vector products over it.
/// Pure — no GPU, no files — so every rule is pinned by unit tests
/// (`SharpnessVectorMathTests`).
///
/// Why these definitions:
///
/// - **Accumulation in `Double`.** One vector is ~8 million floats; a `Float`
///   running sum of that many squares loses every addend below half an ULP
///   of the total, biasing norms and dot products low (the same reason the
///   training graph widens its gradient-norm reduction to fp32). Every
///   reduction here sums in `Double`.
/// - **Central differences.** Hv ≈ (g(w + εv) − g(w − εv)) / 2ε. Its
///   truncation error is O(ε²) (the forward difference's is O(ε)), and the
///   subtraction is done in `Double` from the two `Float` gradients, so the
///   only rounding is the gradients' own.
/// - **Rayleigh quotient** λ = vᵀHv / vᵀv: the eigenvalue estimate power
///   iteration reports each step. It is exact for an eigenvector and, for a
///   symmetric operator, its error is second order in the eigenvector error,
///   so it settles before the vector does.
/// - **Convergence.** Two consecutive estimates within a relative tolerance
///   of the newer one. Relative, because the scale of λ is what is being
///   measured and is unknown in advance.
enum SharpnessVectorMath {

    enum VectorError: LocalizedError, Equatable {
        /// Two vectors that must have the same layout do not.
        case layoutMismatch(detail: String)
        /// A vector expected to have a direction is zero (or not finite).
        case zeroOrNonFiniteNorm(Double)
        /// A group mask leaves no element of the vector free.
        case emptyGroup

        var errorDescription: String? {
            switch self {
            case .layoutMismatch(let detail):
                return "vector layouts differ: \(detail)"
            case .zeroOrNonFiniteNorm(let norm):
                return "vector norm is \(norm); it has no direction to normalize"
            case .emptyGroup:
                return "the parameter group selects no tensor"
            }
        }
    }

    /// Throws unless `a` and `b` hold the same number of tensors with the same
    /// element count each.
    static func requireSameLayout(_ a: [[Float]], _ b: [[Float]]) throws {
        guard a.count == b.count else {
            throw VectorError.layoutMismatch(detail: "\(a.count) vs \(b.count) tensors")
        }
        for index in a.indices where a[index].count != b[index].count {
            throw VectorError.layoutMismatch(
                detail: "tensor \(index) has \(a[index].count) vs \(b[index].count) elements")
        }
    }

    /// Σ aᵢbᵢ over every element, summed in `Double`.
    static func dot(_ a: [[Float]], _ b: [[Float]]) throws -> Double {
        try requireSameLayout(a, b)
        var sum = 0.0
        for index in a.indices {
            a[index].withUnsafeBufferPointer { x in
                b[index].withUnsafeBufferPointer { y in
                    for element in x.indices {
                        sum += Double(x[element]) * Double(y[element])
                    }
                }
            }
        }
        return sum
    }

    /// The Euclidean norm over every element, summed in `Double`.
    static func norm(_ a: [[Float]]) -> Double {
        var sum = 0.0
        for tensor in a {
            tensor.withUnsafeBufferPointer { x in
                for value in x {
                    let widened = Double(value)
                    sum += widened * widened
                }
            }
        }
        return sum.squareRoot()
    }

    /// `y + α·x`, element by element (computed in `Double`, stored as `Float`).
    static func adding(_ x: [[Float]], scaledBy alpha: Double, to y: [[Float]]) throws -> [[Float]] {
        try requireSameLayout(x, y)
        var result = y
        for index in result.indices {
            x[index].withUnsafeBufferPointer { source in
                result[index].withUnsafeMutableBufferPointer { target in
                    for element in target.indices {
                        target[element] = Float(Double(target[element]) + alpha * Double(source[element]))
                    }
                }
            }
        }
        return result
    }

    /// `α·x`.
    static func scaled(_ x: [[Float]], by alpha: Double) -> [[Float]] {
        x.map { tensor in tensor.map { Float(alpha * Double($0)) } }
    }

    /// `x / ‖x‖`, with the norm it divided by. Throws for a zero or
    /// non-finite norm: such a vector has no direction, and silently
    /// returning zeros would make power iteration report λ = 0.
    static func normalized(_ x: [[Float]]) throws -> (unit: [[Float]], norm: Double) {
        let length = norm(x)
        guard length.isFinite, length > 0 else { throw VectorError.zeroOrNonFiniteNorm(length) }
        return (scaled(x, by: 1 / length), length)
    }

    /// Adds `(plus − minus) · scale` into `accumulator`, element by element in
    /// `Double`. With `scale = 1 / (2ε·K)` and K batches each adding their
    /// gradient pair, the accumulator ends as the mean over batches of the
    /// central-difference Hessian-vector product.
    static func accumulateDifference(into accumulator: inout [[Double]], plus: [[Float]], minus: [[Float]],
                                     scale: Double) throws {
        try requireSameLayout(plus, minus)
        guard accumulator.count == plus.count else {
            throw VectorError.layoutMismatch(detail: "accumulator \(accumulator.count) vs \(plus.count) tensors")
        }
        for index in plus.indices {
            guard accumulator[index].count == plus[index].count else {
                throw VectorError.layoutMismatch(
                    detail: "accumulator tensor \(index) has \(accumulator[index].count) vs \(plus[index].count) elements")
            }
            plus[index].withUnsafeBufferPointer { p in
                minus[index].withUnsafeBufferPointer { m in
                    accumulator[index].withUnsafeMutableBufferPointer { target in
                        for element in target.indices {
                            target[element] += (Double(p[element]) - Double(m[element])) * scale
                        }
                    }
                }
            }
        }
    }

    /// Adds `x · scale` into `accumulator` (the mean of K gradients is K
    /// calls with `scale = 1 / K`).
    static func accumulate(into accumulator: inout [[Double]], _ x: [[Float]], scale: Double) throws {
        guard accumulator.count == x.count else {
            throw VectorError.layoutMismatch(detail: "accumulator \(accumulator.count) vs \(x.count) tensors")
        }
        for index in x.indices {
            guard accumulator[index].count == x[index].count else {
                throw VectorError.layoutMismatch(
                    detail: "accumulator tensor \(index) has \(accumulator[index].count) vs \(x[index].count) elements")
            }
            x[index].withUnsafeBufferPointer { source in
                accumulator[index].withUnsafeMutableBufferPointer { target in
                    for element in target.indices {
                        target[element] += Double(source[element]) * scale
                    }
                }
            }
        }
    }

    /// A zero accumulator with `layout`'s shape.
    static func zeroAccumulator(shapedLike layout: [[Float]]) -> [[Double]] {
        layout.map { [Double](repeating: 0, count: $0.count) }
    }

    /// The accumulator narrowed to `Float`, the trainer's vector layout.
    static func narrowed(_ accumulator: [[Double]]) -> [[Float]] {
        accumulator.map { tensor in tensor.map { Float($0) } }
    }

    /// The Rayleigh quotient vᵀHv / vᵀv of `vector` given `product` = H·vector.
    static func rayleighQuotient(vector: [[Float]], product: [[Float]]) throws -> Double {
        let denominator = try dot(vector, vector)
        guard denominator.isFinite, denominator > 0 else { throw VectorError.zeroOrNonFiniteNorm(denominator.squareRoot()) }
        return try dot(vector, product) / denominator
    }

    /// Whether two consecutive eigenvalue estimates agree to `relativeTolerance`
    /// of the newer one's magnitude. A newer estimate of exactly zero agrees
    /// only with an exact zero (there is no scale to be relative to).
    static func hasConverged(previous: Double, current: Double, relativeTolerance: Double) -> Bool {
        guard previous.isFinite, current.isFinite else { return false }
        return abs(current - previous) <= relativeTolerance * abs(current)
    }

    /// `x` with every tensor outside `selected` set to zero: the projection
    /// onto the subspace of the selected tensors. Throws when `selected`
    /// holds no tensor index of `x` (an empty subspace has no eigenvalue).
    static func masked(_ x: [[Float]], keeping selected: Set<Int>) throws -> [[Float]] {
        guard selected.contains(where: { x.indices.contains($0) }) else { throw VectorError.emptyGroup }
        return x.enumerated().map { index, tensor in
            selected.contains(index) ? tensor : [Float](repeating: 0, count: tensor.count)
        }
    }

    /// A vector shaped like `layout` with independent standard-normal entries
    /// in the `selected` tensors and zeros elsewhere, drawn from `random` by
    /// the Box–Muller transform (two uniforms per pair of normals, tensors in
    /// order, elements in order) — the same stream state always gives the same
    /// vector. Power iteration only needs a start with a component along the
    /// top eigenvector; an isotropic draw has one with probability 1.
    static func gaussianVector(shapedLike layout: [[Float]], keeping selected: Set<Int>,
                               using random: inout DCMRandom) throws -> [[Float]] {
        guard selected.contains(where: { layout.indices.contains($0) }) else { throw VectorError.emptyGroup }
        var result: [[Float]] = []
        result.reserveCapacity(layout.count)
        for (index, tensor) in layout.enumerated() {
            guard selected.contains(index) else {
                result.append([Float](repeating: 0, count: tensor.count))
                continue
            }
            var values = [Float](repeating: 0, count: tensor.count)
            var element = 0
            while element < values.count {
                // 1 − u ∈ (0, 1], so the log is finite.
                let radius = (-2 * Foundation.log(1 - random.nextUnitDouble())).squareRoot()
                let angle = 2 * Double.pi * random.nextUnitDouble()
                values[element] = Float(radius * Foundation.cos(angle))
                if element + 1 < values.count {
                    values[element + 1] = Float(radius * Foundation.sin(angle))
                }
                element += 2
            }
            result.append(values)
        }
        return result
    }

    /// The largest absolute element-wise difference between `a` and `b`, and
    /// whether they are bit-for-bit equal (`-0` and `+0` differ in bits).
    static func difference(_ a: [[Float]], _ b: [[Float]]) throws -> (maxAbsolute: Double, bitIdentical: Bool) {
        try requireSameLayout(a, b)
        var largest = 0.0
        var identical = true
        for index in a.indices {
            a[index].withUnsafeBufferPointer { x in
                b[index].withUnsafeBufferPointer { y in
                    for element in x.indices {
                        if x[element].bitPattern != y[element].bitPattern { identical = false }
                        let gap = abs(Double(x[element]) - Double(y[element]))
                        if !(gap <= largest) { largest = gap }   // a NaN gap propagates
                    }
                }
            }
        }
        return (largest, identical)
    }
}

/// The trainable-parameter groups `--sharpness-groups` restricts power
/// iteration to. Every trainable tensor belongs to exactly one, by its
/// `weightTensorPlan()` name and kind:
///
/// - `normAffine`: every BatchNorm / LayerNorm γ and β, wherever it sits
///   (stem, tower, tower end, heads) — the scale-and-shift parameters, whose
///   curvature behaves unlike a convolution's (a BN-normalized conv is
///   scale-invariant; its γ is not).
/// - `stem`: the stem convolution.
/// - `tower`: the residual blocks' convolutions, squeeze-excitation layers,
///   ReZero α and skip projections, and the feature-skip compress conv.
/// - `policyHead` / `valueHead`: each head's convolutions, fully connected
///   layers and biases.
///
/// A tensor no rule claims is an error (`unclassified`), never a silent
/// member of some group: a new tensor family must be placed deliberately.
enum SharpnessParameterGroup: String, CaseIterable, Sendable {
    case stem
    case tower
    case normAffine = "norm_affine"
    case policyHead = "policy_head"
    case valueHead = "value_head"

    struct UnclassifiedTensor: LocalizedError, Equatable {
        let name: String
        var errorDescription: String? { "trainable tensor '\(name)' belongs to no sharpness parameter group" }
    }

    /// The group of the trainable tensor `spec`. BN running statistics are
    /// not trainable and are refused like an unknown name.
    static func group(of spec: WeightTensorSpec) throws -> SharpnessParameterGroup {
        switch spec.kind {
        case .bnRunningStat:
            throw UnclassifiedTensor(name: spec.name)
        case .bnAffine:
            return .normAffine
        case .conv, .linear, .bias, .scalar:
            if spec.name.hasPrefix("stem.") { return .stem }
            if spec.name.hasPrefix("blocks.") || spec.name.hasPrefix("feature_skip.") { return .tower }
            if spec.name.hasPrefix("policy.") { return .policyHead }
            if spec.name.hasPrefix("value.") { return .valueHead }
            throw UnclassifiedTensor(name: spec.name)
        }
    }

    /// The trainable-tensor indices of each group, for a plan in trainable
    /// order (`trainableTensorPlan()`). Every index lands in exactly one
    /// group; a group the architecture lacks maps to an empty set.
    static func tensorIndices(of plan: [WeightTensorSpec]) throws -> [SharpnessParameterGroup: Set<Int>] {
        var indices: [SharpnessParameterGroup: Set<Int>] = [:]
        for group in allCases { indices[group] = [] }
        for (index, spec) in plan.enumerated() {
            indices[try group(of: spec), default: []].insert(index)
        }
        return indices
    }
}
