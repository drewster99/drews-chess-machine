//
//  GraphAuditReference.swift
//  DrewsChessMachine
//
//  Float64 CPU references for every operation `ForwardGraphOps` records: the
//  forward result and the vector-Jacobian product (the operands' gradient
//  given the output's), computed from the operand values MPSGraph itself held.
//  `GraphAudit` compares MPSGraph's own results against these, op by op, at
//  the batch sizes the app runs.
//
//  Why float64 and why per op: MPSGraph returned a wrong fp32 convolution
//  weight gradient once the problem passed 2³¹ elements (R-elitecyc-fp32;
//  `ConvolutionWeightGradientWorkaround`). The error depended on dtype, size
//  and the graph around the op, so it only shows at real sizes inside the real
//  graph, and only an exact reference separates "wrong" from "rounded". Each
//  op is recomputed from its own MPSGraph operands, so an error is pinned to
//  the op that made it rather than smeared through the network.
//
//  Semantics follow `GraphOperationKind`'s documentation (MPSGraph's):
//  NumPy-style broadcasting, reductions keep their axes, population variance,
//  NCHW / OIHW convolution with explicit padding.
//

import Accelerate
import Foundation

/// A dense row-major float64 tensor.
struct Tensor64: Equatable {
    var shape: [Int]
    var values: [Double]

    init(shape: [Int], values: [Double]) {
        precondition(shape.reduce(1, *) == values.count, "shape \(shape) does not hold \(values.count) values")
        self.shape = shape
        self.values = values
    }

    init(zeros shape: [Int]) {
        self.init(shape: shape, values: [Double](repeating: 0, count: shape.reduce(1, *)))
    }

    var count: Int { values.count }

    /// Row-major strides.
    static func strides(_ shape: [Int]) -> [Int] {
        var strides = [Int](repeating: 1, count: shape.count)
        var running = 1
        for axis in stride(from: shape.count - 1, through: 0, by: -1) {
            strides[axis] = running
            running *= shape[axis]
        }
        return strides
    }
}

enum GraphAuditReferenceError: Error, LocalizedError, Equatable {
    case shapeMismatch(String)
    case unsupported(String)

    var errorDescription: String? {
        switch self {
        case .shapeMismatch(let detail): return "graph audit reference: shape mismatch: \(detail)"
        case .unsupported(let detail): return "graph audit reference: unsupported \(detail)"
        }
    }
}

/// The float64 references.
enum GraphAuditReference {

    // MARK: - Rounding to a graph data type

    /// `value` rounded to `dataType` (round to nearest, ties to even), as a
    /// Double. The operands come from MPSGraph tensors of their own type, so
    /// the conversion to Float first is exact for them.
    static func rounded(_ value: Double, to dataType: GraphDataType) -> Double {
        switch dataType {
        case .float32:
            return Double(Float(value))
        case .float16:
            return Double(Float(Float16(Float(value))))
        case .bfloat16:
            return Double(bfloat16Rounded(Float(value)))
        }
    }

    /// Float rounded to bfloat16 (RNE), as a Float. NaN stays NaN.
    static func bfloat16Rounded(_ x: Float) -> Float {
        if x.isNaN { return x }
        let bits = x.bitPattern
        let rounding: UInt32 = 0x7FFF + ((bits >> 16) & 1)
        let (sum, _) = bits.addingReportingOverflow(rounding)
        return Float(bitPattern: sum & 0xFFFF_0000)
    }

    // MARK: - Broadcasting

    /// NumPy broadcast of two shapes, nil when they don't broadcast.
    static func broadcastShape(_ a: [Int], _ b: [Int]) -> [Int]? {
        let rank = max(a.count, b.count)
        let pa = [Int](repeating: 1, count: rank - a.count) + a
        let pb = [Int](repeating: 1, count: rank - b.count) + b
        var out: [Int] = []
        for (x, y) in zip(pa, pb) {
            if x == y || y == 1 { out.append(x) } else if x == 1 { out.append(y) } else { return nil }
        }
        return out
    }

    /// Calls `body(flat, indices)` for every element of a tensor of shape
    /// `outShape`, in row-major order, where `indices[s]` is the element of
    /// source `s` (shape `sourceShapes[s]`, broadcast to `outShape`) that
    /// element reads. The indices are streamed, never materialized: an index
    /// array costs 8 bytes per element, and at batch 4096 a 128-channel
    /// activation has 33.5M elements (the first audit's full-size runs
    /// exhausted memory with them).
    static func walk(_ outShape: [Int], _ sourceShapes: [[Int]],
                     _ body: (Int, UnsafeMutableBufferPointer<Int>) -> Void) {
        let rank = outShape.count
        let n = sourceShapes.count
        var effective = [Int](repeating: 0, count: rank * n)
        for (source, shape) in sourceShapes.enumerated() {
            let padded = [Int](repeating: 1, count: rank - shape.count) + shape
            let strides = Tensor64.strides(padded)
            for axis in 0..<rank { effective[source * rank + axis] = padded[axis] == 1 ? 0 : strides[axis] }
        }
        let total = outShape.reduce(1, *)
        var counter = [Int](repeating: 0, count: rank)
        var indices = [Int](repeating: 0, count: max(n, 1))
        indices.withUnsafeMutableBufferPointer { index in
            for flat in 0..<total {
                body(flat, index)
                // Advance the multi-index, last axis fastest.
                var axis = rank - 1
                while axis >= 0 {
                    counter[axis] += 1
                    for source in 0..<n { index[source] += effective[source * rank + axis] }
                    if counter[axis] < outShape[axis] { break }
                    for source in 0..<n { index[source] -= effective[source * rank + axis] * counter[axis] }
                    counter[axis] = 0
                    axis -= 1
                }
            }
        }
    }

    /// For every element of a tensor of shape `outShape`, the index of the
    /// element of `source` it reads, materialized. Only for small tensors (the
    /// batch axes of a matrix multiplication); everything else uses `walk`.
    static func broadcastIndices(from sourceShape: [Int], to outShape: [Int]) -> [Int] {
        let rank = outShape.count
        let padded = [Int](repeating: 1, count: rank - sourceShape.count) + sourceShape
        let sourceStrides = Tensor64.strides(padded)
        let effective = (0..<rank).map { padded[$0] == 1 ? 0 : sourceStrides[$0] }
        let total = outShape.reduce(1, *)
        var indices = [Int](repeating: 0, count: total)
        var counter = [Int](repeating: 0, count: rank)
        var sourceIndex = 0
        for flat in 0..<total {
            indices[flat] = sourceIndex
            // Increment the multi-index (last axis fastest).
            var axis = rank - 1
            while axis >= 0 {
                counter[axis] += 1
                sourceIndex += effective[axis]
                if counter[axis] < outShape[axis] { break }
                sourceIndex -= effective[axis] * counter[axis]
                counter[axis] = 0
                axis -= 1
            }
        }
        return indices
    }

    /// `gradient` (shape `outShape`) summed down to `targetShape` — the
    /// gradient of an operand that was broadcast to `outShape`.
    static func reduce(_ gradient: Tensor64, toShape targetShape: [Int]) -> Tensor64 {
        if gradient.shape == targetShape { return gradient }
        var out = Tensor64(zeros: targetShape)
        walk(gradient.shape, [targetShape]) { flat, index in
            out.values[index[0]] += gradient.values[flat]
        }
        return out
    }

    static func broadcast(_ tensor: Tensor64, to shape: [Int]) -> Tensor64 {
        if tensor.shape == shape { return tensor }
        var out = Tensor64(zeros: shape)
        walk(shape, [tensor.shape]) { flat, index in out.values[flat] = tensor.values[index[0]] }
        return out
    }

    // MARK: - Elementwise

    enum Binary { case add, subtract, multiply, divide }

    static func binary(_ op: Binary, _ a: Tensor64, _ b: Tensor64) throws -> Tensor64 {
        guard let shape = broadcastShape(a.shape, b.shape) else {
            throw GraphAuditReferenceError.shapeMismatch("\(a.shape) and \(b.shape) don't broadcast")
        }
        var out = [Double](repeating: 0, count: shape.reduce(1, *))
        walk(shape, [a.shape, b.shape]) { k, index in
            let x = a.values[index[0]], y = b.values[index[1]]
            switch op {
            case .add: out[k] = x + y
            case .subtract: out[k] = x - y
            case .multiply: out[k] = x * y
            case .divide: out[k] = x / y
            }
        }
        return Tensor64(shape: shape, values: out)
    }

    static func binaryGradients(_ op: Binary, _ a: Tensor64, _ b: Tensor64, dY: Tensor64) throws
        -> (Tensor64, Tensor64) {
        guard let shape = broadcastShape(a.shape, b.shape), shape == dY.shape else {
            throw GraphAuditReferenceError.shapeMismatch("binary gradient: \(a.shape), \(b.shape), dY \(dY.shape)")
        }
        var da = Tensor64(zeros: a.shape)
        var db = Tensor64(zeros: b.shape)
        walk(shape, [a.shape, b.shape]) { k, index in
            let ia = index[0], ib = index[1]
            let g = dY.values[k], x = a.values[ia], y = b.values[ib]
            switch op {
            case .add:
                da.values[ia] += g; db.values[ib] += g
            case .subtract:
                da.values[ia] += g; db.values[ib] -= g
            case .multiply:
                da.values[ia] += g * y; db.values[ib] += g * x
            case .divide:
                da.values[ia] += g / y; db.values[ib] -= g * x / (y * y)
            }
        }
        return (da, db)
    }

    enum Unary { case tanh, sigmoid, erf, relu, leakyReLU(Double) }

    static func unary(_ op: Unary, _ x: Tensor64) -> Tensor64 {
        var out = x
        for k in 0..<x.count {
            let v = x.values[k]
            switch op {
            case .tanh: out.values[k] = Foundation.tanh(v)
            case .sigmoid: out.values[k] = 1 / (1 + Foundation.exp(-v))
            case .erf: out.values[k] = Foundation.erf(v)
            case .relu: out.values[k] = v > 0 ? v : 0
            case .leakyReLU(let alpha): out.values[k] = v >= 0 ? v : alpha * v
            }
        }
        return out
    }

    /// The input gradient of a unary op. At an activation's kink (x exactly 0)
    /// MPSGraph's convention is not documented; the reference takes the
    /// right-hand slope for ReLU (0 at x = 0) and leaky ReLU (1 at x = 0).
    static func unaryGradient(_ op: Unary, _ x: Tensor64, dY: Tensor64) -> Tensor64 {
        var out = x
        for k in 0..<x.count {
            let v = x.values[k], g = dY.values[k]
            switch op {
            case .tanh:
                let t = Foundation.tanh(v); out.values[k] = g * (1 - t * t)
            case .sigmoid:
                let s = 1 / (1 + Foundation.exp(-v)); out.values[k] = g * s * (1 - s)
            case .erf:
                out.values[k] = g * 2 / Double.pi.squareRoot() * Foundation.exp(-v * v)
            case .relu:
                out.values[k] = v > 0 ? g : 0
            case .leakyReLU(let alpha):
                out.values[k] = v >= 0 ? g : alpha * g
            }
        }
        return out
    }

    // MARK: - Reductions

    /// Mean or population variance over `axes`, keeping them as size 1.
    static func reduceMean(_ x: Tensor64, axes: [Int]) -> Tensor64 {
        let outShape = x.shape.enumerated().map { axes.contains($0.offset) ? 1 : $0.element }
        let n = Double(axes.reduce(1) { $0 * x.shape[$1] })
        let summed = reduce(x, toShape: outShape)
        return Tensor64(shape: outShape, values: summed.values.map { $0 / n })
    }

    static func reduceVariance(_ x: Tensor64, axes: [Int]) -> Tensor64 {
        let mean = reduceMean(x, axes: axes)
        let meanB = broadcast(mean, to: x.shape)
        var squares = x
        for k in 0..<x.count {
            let d = x.values[k] - meanB.values[k]
            squares.values[k] = d * d
        }
        return reduceMean(squares, axes: axes)
    }

    static func reduceSum(_ x: Tensor64, axis: Int) -> Tensor64 {
        var outShape = x.shape
        outShape[axis] = 1
        return reduce(x, toShape: outShape)
    }

    static func meanGradient(_ x: Tensor64, axes: [Int], dY: Tensor64) -> Tensor64 {
        let n = Double(axes.reduce(1) { $0 * x.shape[$1] })
        let g = broadcast(dY, to: x.shape)
        return Tensor64(shape: x.shape, values: g.values.map { $0 / n })
    }

    static func varianceGradient(_ x: Tensor64, axes: [Int], dY: Tensor64) -> Tensor64 {
        let n = Double(axes.reduce(1) { $0 * x.shape[$1] })
        let meanB = broadcast(reduceMean(x, axes: axes), to: x.shape)
        let g = broadcast(dY, to: x.shape)
        var out = x
        for k in 0..<x.count {
            out.values[k] = g.values[k] * 2 * (x.values[k] - meanB.values[k]) / n
        }
        return out
    }

    // MARK: - Normalize

    /// (x − mean) / sqrt(variance + ε) · γ + β, every operand broadcast to x.
    static func normalize(_ x: Tensor64, mean: Tensor64, variance: Tensor64, gamma: Tensor64, beta: Tensor64,
                          epsilon: Double) -> Tensor64 {
        var out = x
        walk(x.shape, [mean.shape, variance.shape, gamma.shape, beta.shape]) { k, i in
            let s = (variance.values[i[1]] + epsilon).squareRoot()
            out.values[k] = (x.values[k] - mean.values[i[0]]) / s * gamma.values[i[2]] + beta.values[i[3]]
        }
        return out
    }

    /// Gradients of `normalize` with respect to x, mean, variance, γ and β.
    static func normalizeGradients(_ x: Tensor64, mean: Tensor64, variance: Tensor64, gamma: Tensor64,
                                   beta: Tensor64, epsilon: Double, dY: Tensor64)
        -> (dx: Tensor64, dMean: Tensor64, dVariance: Tensor64, dGamma: Tensor64, dBeta: Tensor64) {
        var dx = Tensor64(zeros: x.shape)
        var dMean = Tensor64(zeros: mean.shape)
        var dVariance = Tensor64(zeros: variance.shape)
        var dGamma = Tensor64(zeros: gamma.shape)
        var dBeta = Tensor64(zeros: beta.shape)
        walk(x.shape, [mean.shape, variance.shape, gamma.shape, beta.shape]) { k, i in
            let vePlus = variance.values[i[1]] + epsilon
            let s = vePlus.squareRoot()
            let centered = x.values[k] - mean.values[i[0]]
            let gy = dY.values[k]
            let gammaK = gamma.values[i[2]]
            dx.values[k] = gy * gammaK / s
            dMean.values[i[0]] -= gy * gammaK / s
            dVariance.values[i[1]] += gy * gammaK * centered * (-0.5) / (vePlus * s)
            dGamma.values[i[2]] += gy * centered / s
            dBeta.values[i[3]] += gy
        }
        return (dx, dMean, dVariance, dGamma, dBeta)
    }

    // MARK: - Softmax

    static func softmax(_ x: Tensor64, axis: Int) -> Tensor64 {
        let (outer, length, inner) = axisSplit(x.shape, axis: axis)
        var out = x
        for o in 0..<outer {
            for i in 0..<inner {
                var maxValue = -Double.infinity
                for l in 0..<length { maxValue = max(maxValue, x.values[(o * length + l) * inner + i]) }
                var total = 0.0
                for l in 0..<length {
                    let e = Foundation.exp(x.values[(o * length + l) * inner + i] - maxValue)
                    out.values[(o * length + l) * inner + i] = e
                    total += e
                }
                for l in 0..<length { out.values[(o * length + l) * inner + i] /= total }
            }
        }
        return out
    }

    static func softmaxGradient(_ x: Tensor64, axis: Int, dY: Tensor64) -> Tensor64 {
        let y = softmax(x, axis: axis)
        let (outer, length, inner) = axisSplit(x.shape, axis: axis)
        var out = x
        for o in 0..<outer {
            for i in 0..<inner {
                var dot = 0.0
                for l in 0..<length {
                    let k = (o * length + l) * inner + i
                    dot += dY.values[k] * y.values[k]
                }
                for l in 0..<length {
                    let k = (o * length + l) * inner + i
                    out.values[k] = y.values[k] * (dY.values[k] - dot)
                }
            }
        }
        return out
    }

    private static func axisSplit(_ shape: [Int], axis: Int) -> (outer: Int, length: Int, inner: Int) {
        let outer = shape[..<axis].reduce(1, *)
        let inner = shape[(axis + 1)...].reduce(1, *)
        return (outer, shape[axis], inner)
    }

    // MARK: - Concat and slice

    static func concat(_ parts: [Tensor64], dimension: Int) throws -> Tensor64 {
        guard let first = parts.first else { throw GraphAuditReferenceError.unsupported("empty concat") }
        var outShape = first.shape
        outShape[dimension] = parts.reduce(0) { $0 + $1.shape[dimension] }
        let (outer, _, inner) = axisSplit(outShape, axis: dimension)
        var out = [Double]()
        out.reserveCapacity(outShape.reduce(1, *))
        for o in 0..<outer {
            for part in parts {
                let length = part.shape[dimension]
                let base = o * length * inner
                out.append(contentsOf: part.values[base..<(base + length * inner)])
            }
        }
        return Tensor64(shape: outShape, values: out)
    }

    static func concatGradients(_ parts: [Tensor64], dimension: Int, dY: Tensor64) -> [Tensor64] {
        var start = 0
        return parts.map { part in
            let length = part.shape[dimension]
            defer { start += length }
            return slice(dY, dimension: dimension, start: start, length: length)
        }
    }

    static func slice(_ x: Tensor64, dimension: Int, start: Int, length: Int) -> Tensor64 {
        let (outer, full, inner) = axisSplit(x.shape, axis: dimension)
        var outShape = x.shape
        outShape[dimension] = length
        var out = [Double]()
        out.reserveCapacity(outer * length * inner)
        for o in 0..<outer {
            let base = (o * full + start) * inner
            out.append(contentsOf: x.values[base..<(base + length * inner)])
        }
        return Tensor64(shape: outShape, values: out)
    }

    static func sliceGradient(inputShape: [Int], dimension: Int, start: Int, dY: Tensor64) -> Tensor64 {
        let (outer, full, inner) = axisSplit(inputShape, axis: dimension)
        let length = dY.shape[dimension]
        var out = Tensor64(zeros: inputShape)
        for o in 0..<outer {
            for l in 0..<length {
                for i in 0..<inner {
                    out.values[(o * full + start + l) * inner + i] = dY.values[(o * length + l) * inner + i]
                }
            }
        }
        return out
    }

    // MARK: - Matrix multiplication

    /// C[cOffset...] = op(A[aOffset...]) · op(B[bOffset...]) + beta · C, row-major,
    /// where op transposes when asked. Built on `vDSP_mmulD` (the CBLAS
    /// interface is deprecated in this SDK); a transposed operand is copied
    /// out transposed first. beta is 0 (overwrite) or 1 (accumulate).
    private static func gemm(transposeA: Bool, transposeB: Bool, m: Int, n: Int, k: Int,
                             a: [Double], aOffset: Int, lda: Int,
                             b: [Double], bOffset: Int, ldb: Int,
                             beta: Double, c: inout [Double], cOffset: Int, ldc: Int) {
        precondition(beta == 0 || beta == 1, "gemm accumulates (1) or overwrites (0)")
        precondition(ldc == n, "gemm writes a dense row-major C")
        // op(A) as a dense m×k row-major matrix, op(B) as k×n.
        let opA = denseMatrix(a, offset: aOffset, ld: lda, rows: m, cols: k, transposed: transposeA)
        let opB = denseMatrix(b, offset: bOffset, ld: ldb, rows: k, cols: n, transposed: transposeB)
        var product = [Double](repeating: 0, count: m * n)
        vDSP_mmulD(opA, 1, opB, 1, &product, 1, vDSP_Length(m), vDSP_Length(n), vDSP_Length(k))
        if beta == 0 {
            for i in 0..<(m * n) { c[cOffset + i] = product[i] }
        } else {
            for i in 0..<(m * n) { c[cOffset + i] += product[i] }
        }
    }

    /// The rows × cols matrix op(X): X itself (stored rows × ld from `offset`)
    /// or, when `transposed`, the transpose of the stored cols × rows matrix.
    private static func denseMatrix(_ x: [Double], offset: Int, ld: Int, rows: Int, cols: Int,
                                    transposed: Bool) -> [Double] {
        var out = [Double](repeating: 0, count: rows * cols)
        if transposed {
            // Stored as cols × rows with leading dimension ld.
            for r in 0..<cols {
                let base = offset + r * ld
                for q in 0..<rows { out[q * cols + r] = x[base + q] }
            }
        } else {
            for r in 0..<rows {
                let base = offset + r * ld
                for q in 0..<cols { out[r * cols + q] = x[base + q] }
            }
        }
        return out
    }

    /// [..., M, K] × [..., K, N] with the leading axes broadcast.
    static func matmul(_ a: Tensor64, _ b: Tensor64) throws -> Tensor64 {
        let (batch, m, k, n, ia, ib) = try matmulLayout(a.shape, b.shape)
        var out = [Double](repeating: 0, count: batch.reduce(1, *) * m * n)
        for (index, (offA, offB)) in zip(ia, ib).enumerated() {
            gemm(transposeA: false, transposeB: false, m: m, n: n, k: k,
                 a: a.values, aOffset: offA * m * k, lda: k, b: b.values, bOffset: offB * k * n, ldb: n,
                 beta: 0, c: &out, cOffset: index * m * n, ldc: n)
        }
        return Tensor64(shape: batch + [m, n], values: out)
    }

    static func matmulGradients(_ a: Tensor64, _ b: Tensor64, dY: Tensor64) throws -> (Tensor64, Tensor64) {
        let (_, m, k, n, ia, ib) = try matmulLayout(a.shape, b.shape)
        var da = Tensor64(zeros: a.shape)
        var db = Tensor64(zeros: b.shape)
        for (index, (offA, offB)) in zip(ia, ib).enumerated() {
            // dA += dY · Bᵀ ; dB += Aᵀ · dY (beta 1: broadcast operands accumulate).
            gemm(transposeA: false, transposeB: true, m: m, n: k, k: n,
                 a: dY.values, aOffset: index * m * n, lda: n, b: b.values, bOffset: offB * k * n, ldb: n,
                 beta: 1, c: &da.values, cOffset: offA * m * k, ldc: k)
            gemm(transposeA: true, transposeB: false, m: k, n: n, k: m,
                 a: a.values, aOffset: offA * m * k, lda: k, b: dY.values, bOffset: index * m * n, ldb: n,
                 beta: 1, c: &db.values, cOffset: offB * k * n, ldc: n)
        }
        return (da, db)
    }

    /// The broadcast batch shape, M, K, N, and for each output batch element the
    /// batch element of a and of b it reads.
    private static func matmulLayout(_ sa: [Int], _ sb: [Int]) throws
        -> (batch: [Int], m: Int, k: Int, n: Int, ia: [Int], ib: [Int]) {
        guard sa.count >= 2, sb.count >= 2, sa[sa.count - 1] == sb[sb.count - 2] else {
            throw GraphAuditReferenceError.shapeMismatch("matmul \(sa) × \(sb)")
        }
        let batchA = Array(sa.dropLast(2)), batchB = Array(sb.dropLast(2))
        guard let batch = broadcastShape(batchA, batchB) else {
            throw GraphAuditReferenceError.shapeMismatch("matmul batch \(batchA) × \(batchB)")
        }
        let ia = batch.isEmpty ? [0] : broadcastIndices(from: batchA, to: batch)
        let ib = batch.isEmpty ? [0] : broadcastIndices(from: batchB, to: batch)
        return (batch, sa[sa.count - 2], sa[sa.count - 1], sb[sb.count - 1], ia, ib)
    }

    // MARK: - Convolution

    struct ConvolutionGeometry: Equatable {
        let strideY: Int, strideX: Int, dilationY: Int, dilationX: Int
        let paddingTop: Int, paddingBottom: Int, paddingLeft: Int, paddingRight: Int
    }

    /// Byte budget for one im2col block; the batch is processed in blocks.
    static let im2colBlockBytes = 384 * 1024 * 1024

    /// NCHW input, OIHW weights, groups 1.
    static func convolution(_ x: Tensor64, weights w: Tensor64, geometry g: ConvolutionGeometry) throws -> Tensor64 {
        let layout = try convolutionLayout(x.shape, w.shape, g)
        var y = Tensor64(zeros: [layout.n, layout.o, layout.ho, layout.wo])
        try forEachBlock(layout) { first, count in
            let cols = im2col(x, layout: layout, geometry: g, first: first, count: count)
            let p = count * layout.ho * layout.wo
            var block = [Double](repeating: 0, count: layout.o * p)
            gemm(transposeA: false, transposeB: false, m: layout.o, n: p, k: layout.k,
                 a: w.values, aOffset: 0, lda: layout.k, b: cols, bOffset: 0, ldb: p,
                 beta: 0, c: &block, cOffset: 0, ldc: p)
            scatter(block, into: &y.values, layout: layout, first: first, count: count)
        }
        return y
    }

    static func convolutionGradients(_ x: Tensor64, weights w: Tensor64, geometry g: ConvolutionGeometry,
                                     dY: Tensor64) throws -> (dx: Tensor64, dWeights: Tensor64) {
        let layout = try convolutionLayout(x.shape, w.shape, g)
        guard dY.shape == [layout.n, layout.o, layout.ho, layout.wo] else {
            throw GraphAuditReferenceError.shapeMismatch("convolution dY \(dY.shape)")
        }
        var dx = Tensor64(zeros: x.shape)
        var dw = Tensor64(zeros: w.shape)
        try forEachBlock(layout) { first, count in
            let cols = im2col(x, layout: layout, geometry: g, first: first, count: count)
            let p = count * layout.ho * layout.wo
            let gBlock = gather(dY.values, layout: layout, first: first, count: count)
            var dCols = [Double](repeating: 0, count: layout.k * p)
            // dW[O, K] += dY[O, P] · colsᵀ
            gemm(transposeA: false, transposeB: true, m: layout.o, n: layout.k, k: p,
                 a: gBlock, aOffset: 0, lda: p, b: cols, bOffset: 0, ldb: p,
                 beta: 1, c: &dw.values, cOffset: 0, ldc: layout.k)
            // dCols[K, P] = Wᵀ · dY
            gemm(transposeA: true, transposeB: false, m: layout.k, n: p, k: layout.o,
                 a: w.values, aOffset: 0, lda: layout.k, b: gBlock, bOffset: 0, ldb: p,
                 beta: 0, c: &dCols, cOffset: 0, ldc: p)
            col2im(dCols, into: &dx.values, layout: layout, geometry: g, first: first, count: count)
        }
        return (dx, dw)
    }

    private struct ConvolutionLayout {
        let n: Int, c: Int, h: Int, w: Int
        let o: Int, kh: Int, kw: Int
        let ho: Int, wo: Int
        var k: Int { c * kh * kw }
    }

    private static func convolutionLayout(_ xs: [Int], _ ws: [Int], _ g: ConvolutionGeometry) throws
        -> ConvolutionLayout {
        guard xs.count == 4, ws.count == 4, xs[1] == ws[1] else {
            throw GraphAuditReferenceError.shapeMismatch("convolution input \(xs), weights \(ws) (groups 1)")
        }
        let ho = (xs[2] + g.paddingTop + g.paddingBottom - g.dilationY * (ws[2] - 1) - 1) / g.strideY + 1
        let wo = (xs[3] + g.paddingLeft + g.paddingRight - g.dilationX * (ws[3] - 1) - 1) / g.strideX + 1
        return ConvolutionLayout(n: xs[0], c: xs[1], h: xs[2], w: xs[3], o: ws[0], kh: ws[2], kw: ws[3],
                                 ho: ho, wo: wo)
    }

    private static func forEachBlock(_ layout: ConvolutionLayout, _ body: (Int, Int) throws -> Void) throws {
        let perSample = layout.k * layout.ho * layout.wo * MemoryLayout<Double>.size
        let block = max(1, im2colBlockBytes / max(1, perSample))
        var first = 0
        while first < layout.n {
            let count = min(block, layout.n - first)
            try body(first, count)
            first += count
        }
    }

    /// cols[K, P] for samples first..<first+count; K = (c, ky, kx), P = (n, oy, ox).
    private static func im2col(_ x: Tensor64, layout l: ConvolutionLayout, geometry g: ConvolutionGeometry,
                               first: Int, count: Int) -> [Double] {
        let p = count * l.ho * l.wo
        var cols = [Double](repeating: 0, count: l.k * p)
        for c in 0..<l.c {
            for ky in 0..<l.kh {
                for kx in 0..<l.kw {
                    let row = (c * l.kh + ky) * l.kw + kx
                    for s in 0..<count {
                        let xBase = ((first + s) * l.c + c) * l.h * l.w
                        for oy in 0..<l.ho {
                            let iy = oy * g.strideY - g.paddingTop + ky * g.dilationY
                            guard iy >= 0, iy < l.h else { continue }
                            for ox in 0..<l.wo {
                                let ix = ox * g.strideX - g.paddingLeft + kx * g.dilationX
                                guard ix >= 0, ix < l.w else { continue }
                                cols[row * p + (s * l.ho + oy) * l.wo + ox] = x.values[xBase + iy * l.w + ix]
                            }
                        }
                    }
                }
            }
        }
        return cols
    }

    private static func col2im(_ dCols: [Double], into dx: inout [Double], layout l: ConvolutionLayout,
                               geometry g: ConvolutionGeometry, first: Int, count: Int) {
        let p = count * l.ho * l.wo
        for c in 0..<l.c {
            for ky in 0..<l.kh {
                for kx in 0..<l.kw {
                    let row = (c * l.kh + ky) * l.kw + kx
                    for s in 0..<count {
                        let xBase = ((first + s) * l.c + c) * l.h * l.w
                        for oy in 0..<l.ho {
                            let iy = oy * g.strideY - g.paddingTop + ky * g.dilationY
                            guard iy >= 0, iy < l.h else { continue }
                            for ox in 0..<l.wo {
                                let ix = ox * g.strideX - g.paddingLeft + kx * g.dilationX
                                guard ix >= 0, ix < l.w else { continue }
                                dx[xBase + iy * l.w + ix] += dCols[row * p + (s * l.ho + oy) * l.wo + ox]
                            }
                        }
                    }
                }
            }
        }
    }

    /// Output block [O, P] → NCHW.
    private static func scatter(_ block: [Double], into y: inout [Double], layout l: ConvolutionLayout,
                                first: Int, count: Int) {
        let hw = l.ho * l.wo
        let p = count * hw
        for o in 0..<l.o {
            for s in 0..<count {
                let dst = ((first + s) * l.o + o) * hw
                let src = o * p + s * hw
                for q in 0..<hw { y[dst + q] = block[src + q] }
            }
        }
    }

    /// NCHW → block [O, P].
    private static func gather(_ y: [Double], layout l: ConvolutionLayout, first: Int, count: Int) -> [Double] {
        let hw = l.ho * l.wo
        let p = count * hw
        var block = [Double](repeating: 0, count: l.o * p)
        for o in 0..<l.o {
            for s in 0..<count {
                let src = ((first + s) * l.o + o) * hw
                let dst = o * p + s * hw
                for q in 0..<hw { block[dst + q] = y[src + q] }
            }
        }
        return block
    }

    // MARK: - Comparison

    /// How far `actual` (MPSGraph) is from `reference` (float64).
    struct Comparison: Codable, Equatable {
        /// ‖actual − reference‖₂ / ‖reference‖₂ (nil when the reference is 0).
        let relativeError: Double?
        let maxAbsoluteError: Double
        let referenceNorm: Double
        let referenceMaxAbs: Double
        /// The relative error of `reference` rounded to the result's own data
        /// type: the least any result of that type can show.
        let roundingFloor: Double?
        let nonFiniteActual: Int
        let count: Int

        enum CodingKeys: String, CodingKey {
            case relativeError = "rel_err", maxAbsoluteError = "max_abs_err", referenceNorm = "ref_norm"
            case referenceMaxAbs = "ref_max_abs", roundingFloor = "rounding_floor"
            case nonFiniteActual = "non_finite", count
        }
    }

    static func compare(actual: [Double], reference: [Double], dataType: GraphDataType) -> Comparison {
        precondition(actual.count == reference.count)
        var diff2 = 0.0, ref2 = 0.0, round2 = 0.0, maxAbs = 0.0, refMax = 0.0
        var nonFinite = 0
        for k in 0..<reference.count {
            let a = actual[k], r = reference[k]
            if !a.isFinite { nonFinite += 1; continue }
            let d = a - r
            diff2 += d * d
            ref2 += r * r
            maxAbs = max(maxAbs, abs(d))
            refMax = max(refMax, abs(r))
            let q = rounded(r, to: dataType) - r
            round2 += q * q
        }
        let refNorm = ref2.squareRoot()
        return Comparison(
            relativeError: refNorm > 0 ? diff2.squareRoot() / refNorm : nil,
            maxAbsoluteError: maxAbs, referenceNorm: refNorm, referenceMaxAbs: refMax,
            roundingFloor: refNorm > 0 ? round2.squareRoot() / refNorm : nil,
            nonFiniteActual: nonFinite, count: reference.count)
    }
}
