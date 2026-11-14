// Minimal: Y = conv(pre(X0), W); loss = sum(R * post(Y)). Compare graph dW with corr(X, dY) read from the same graph.
import Foundation
import Metal
import MetalPerformanceShadersGraph
struct LCG { var s: UInt64
    mutating func u() -> Float { s = s &* 6364136223846793005 &+ 1442695040888963407; return Float(s >> 40) / Float(1 << 24) }
    mutating func n() -> Float { let a = max(u(), 1e-7), b = u(); return sqrt(-2 * log(a)) * cos(2 * .pi * b) }
}
func read(_ td: MPSGraphTensorData, count: Int) -> [Float] { var o = [Float](repeating: 0, count: count); td.mpsndarray().readBytes(&o, strideBytes: nil); return o }
let device = MTLCreateSystemDefaultDevice()!
let queue = device.makeCommandQueue()!
let a = CommandLine.arguments
let n = Int(a[1])!, k = Int(a[2])!, C = Int(a[3])!, pre = a[4], post = a[5]
var r = LCG(s: 3)
let sh: [NSNumber] = [NSNumber(value: n), NSNumber(value: C), 8, 8]
let g = MPSGraph()
let x0v = (0..<(n * C * 64)).map { _ in r.n() }
let X0 = g.constant(x0v.withUnsafeBufferPointer { Data(buffer: $0) }, shape: sh, dataType: .float32)
var X: MPSGraphTensor
switch pre {
case "none": X = X0
case "silu": X = g.multiplication(X0, g.sigmoid(with: X0, name: nil), name: nil)
case "relu": X = g.reLU(with: X0, name: nil)
case "bn": X = g.normalize(X0, mean: g.mean(of: X0, axes: [0, 2, 3], name: nil), variance: g.variance(of: X0, axes: [0, 2, 3], name: nil), gamma: nil, beta: nil, epsilon: 1e-5, name: nil)
case "scale": X = g.multiplication(X0, g.constant(0.5, dataType: .float32), name: nil)
default: fatalError()
}
let wv = (0..<(C * C * k * k)).map { _ in r.n() * 0.03 }
let W = g.variable(with: wv.withUnsafeBufferPointer { Data(buffer: $0) }, shape: [NSNumber(value: C), NSNumber(value: C), NSNumber(value: k), NSNumber(value: k)], dataType: .float32, name: nil)
let d = MPSGraphConvolution2DOpDescriptor(strideInX: 1, strideInY: 1, dilationRateInX: 1, dilationRateInY: 1, groups: 1, paddingStyle: .explicit, dataLayout: .NCHW, weightsLayout: .OIHW)!
let p = (k - 1) / 2; d.paddingLeft = p; d.paddingRight = p; d.paddingTop = p; d.paddingBottom = p
let Y = g.convolution2D(X, weights: W, descriptor: d, name: nil)
var Z: MPSGraphTensor
switch post {
case "none": Z = Y
case "square": Z = g.square(with: Y, name: nil)
case "bn": Z = g.normalize(Y, mean: g.mean(of: Y, axes: [0, 2, 3], name: nil), variance: g.variance(of: Y, axes: [0, 2, 3], name: nil), gamma: nil, beta: nil, epsilon: 1e-5, name: nil)
default: fatalError()
}
let rv = (0..<(n * C * 64)).map { _ in r.n() }
let R = g.constant(rv.withUnsafeBufferPointer { Data(buffer: $0) }, shape: sh, dataType: .float32)
let loss = g.reductionSum(with: g.reshape(g.multiplication(Z, R, name: nil), shape: [-1], name: nil), axis: 0, name: nil)
let grads = g.gradients(of: loss, with: [W, Y], name: nil)
let res = g.run(with: queue, feeds: [:], targetTensors: [grads[W]!, grads[Y]!, X, Y], targetOperations: nil)
let dW = read(res[grads[W]!]!, count: wv.count), dYg = read(res[grads[Y]!]!, count: n * C * 64), Xv = read(res[X]!, count: n * C * 64), Yv = read(res[Y]!, count: n * C * 64)
// CPU dY (float64) for post=bn: dY = (R - mean(R) - yhat*mean(R*yhat)) / sqrt(var+eps), per channel over (n,h,w); for post=none dY = R.
var dY = [Float](repeating: 0, count: n * C * 64)
if post == "bn" {
  let M = Double(n * 64)
  for c in 0..<C {
    var mu = 0.0; for b in 0..<n { for j in 0..<64 { mu += Double(Yv[(b * C + c) * 64 + j]) } }; mu /= M
    var v = 0.0; for b in 0..<n { for j in 0..<64 { let q = Double(Yv[(b * C + c) * 64 + j]) - mu; v += q * q } }; v /= M
    let sd = sqrt(v + 1e-5)
    var mr = 0.0, mry = 0.0
    for b in 0..<n { for j in 0..<64 { let i = (b * C + c) * 64 + j; let yh = (Double(Yv[i]) - mu) / sd; mr += Double(rv[i]); mry += Double(rv[i]) * yh } }
    mr /= M; mry /= M
    for b in 0..<n { for j in 0..<64 { let i = (b * C + c) * 64 + j; let yh = (Double(Yv[i]) - mu) / sd; dY[i] = Float((Double(rv[i]) - mr - yh * mry) / sd) } }
  }
} else { dY = rv }
var ed = 0.0, nd = 0.0; for i in 0..<dY.count { ed += pow(Double(dYg[i] - dY[i]), 2); nd += Double(dY[i]) * Double(dY[i]) }
print(String(format: "   graph dY vs CPU dY rel err=%.3e", sqrt(ed / nd)))
var err = 0.0, ref = 0.0
for t in 0..<4 { let o = (t * 37) % C, i = (t * 53 + 7) % C
    for ky in 0..<k { for kx in 0..<k {
        var s = 0.0
        for b in 0..<n { for yy in 0..<8 { let sy = yy + ky - p; if sy < 0 || sy >= 8 { continue }
            for xx in 0..<8 { let sx = xx + kx - p; if sx < 0 || sx >= 8 { continue }
                s += Double(dY[((b * C + o) * 8 + yy) * 8 + xx]) * Double(Xv[((b * C + i) * 8 + sy) * 8 + sx]) } } }
        let idx = ((o * C + i) * k + ky) * k + kx
        err += pow(Double(dW[idx]) - s, 2); ref += s * s
    }}
}
print(String(format: "n=%5d k=%d C=%3d pre=%-5@ post=%-6@ rel err=%.3e", n, k, C, pre, post, sqrt(err / ref)))
