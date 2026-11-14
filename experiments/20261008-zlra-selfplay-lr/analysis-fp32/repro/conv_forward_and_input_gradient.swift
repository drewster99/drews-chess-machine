// fp32 batch 4096, 9x9, 128ch, Y also read: check forward Y and input gradient dX against float64 CPU.
import Foundation
import Metal
import MetalPerformanceShadersGraph
struct LCG { var s: UInt64
    mutating func u() -> Float { s = s &* 6364136223846793005 &+ 1442695040888963407; return Float(s >> 40) / Float(1 << 24) }
    mutating func n() -> Float { let a = max(u(), 1e-7), b = u(); return sqrt(-2 * log(a)) * cos(2 * .pi * b) }
}
func read(_ td: MPSGraphTensorData, _ count: Int) -> [Float] { var o = [Float](repeating: 0, count: count); td.mpsndarray().readBytes(&o, strideBytes: nil); return o }
let device = MTLCreateSystemDefaultDevice()!, queue = device.makeCommandQueue()!
let n = Int(CommandLine.arguments[1])!, k = 9, C = 128, p = 4
let inferenceOnly = CommandLine.arguments.count > 2
var r = LCG(s: 3)
let xv = (0..<(n * C * 64)).map { _ in r.n() }, wv = (0..<(C * C * k * k)).map { _ in r.n() * 0.03 }, rv = (0..<(n * C * 64)).map { _ in r.n() * 0.1 }
let g = MPSGraph(); let sh: [NSNumber] = [NSNumber(value: n), NSNumber(value: C), 8, 8]
let X = g.variable(with: xv.withUnsafeBufferPointer { Data(buffer: $0) }, shape: sh, dataType: .float32, name: nil)
let W = g.variable(with: wv.withUnsafeBufferPointer { Data(buffer: $0) }, shape: [128, 128, 9, 9], dataType: .float32, name: nil)
let R = g.constant(rv.withUnsafeBufferPointer { Data(buffer: $0) }, shape: sh, dataType: .float32)
let d = MPSGraphConvolution2DOpDescriptor(strideInX: 1, strideInY: 1, dilationRateInX: 1, dilationRateInY: 1, groups: 1, paddingStyle: .explicit, dataLayout: .NCHW, weightsLayout: .OIHW)!
d.paddingLeft = p; d.paddingRight = p; d.paddingTop = p; d.paddingBottom = p
let Y = g.convolution2D(X, weights: W, descriptor: d, name: nil)
var targets = [Y]; var gX: MPSGraphTensor? = nil
if !inferenceOnly {
    let loss = g.reductionSum(with: g.reshape(g.multiplication(Y, R, name: nil), shape: [-1], name: nil), axis: 0, name: nil)
    let gr = g.gradients(of: loss, with: [W, X], name: nil); gX = gr[X]!; targets += [gr[W]!, gX!]
}
let res = g.run(with: queue, feeds: [:], targetTensors: targets, targetOperations: nil)
let Yv = read(res[Y]!, n * C * 64)
var ey = 0.0, ny = 0.0, ex = 0.0, nx = 0.0
let dX = gX.map { read(res[$0]!, n * C * 64) }
for t in 0..<400 { let b = (t * 977) % n, c = (t * 31) % C, yy = t % 8, xx = (t / 8) % 8
    var s = 0.0
    for i in 0..<C { for ky in 0..<k { let sy = yy + ky - p; if sy < 0 || sy >= 8 { continue }
        for kx in 0..<k { let sx = xx + kx - p; if sx < 0 || sx >= 8 { continue }
            s += Double(wv[((c * C + i) * k + ky) * k + kx]) * Double(xv[((b * C + i) * 8 + sy) * 8 + sx]) } } }
    let yi = ((b * C + c) * 8 + yy) * 8 + xx; ey += pow(Double(Yv[yi]) - s, 2); ny += s * s
    if let dX { // dX[b,i,sy,sx] = sum_o sum_ky,kx R[b,o,sy-ky+p,sx-kx+p] W[o,i,ky,kx]
        var q = 0.0
        for o in 0..<C { for ky in 0..<k { let oy = yy - ky + p; if oy < 0 || oy >= 8 { continue }
            for kx in 0..<k { let ox = xx - kx + p; if ox < 0 || ox >= 8 { continue }
                q += Double(rv[((b * C + o) * 8 + oy) * 8 + ox]) * Double(wv[((o * C + c) * k + ky) * k + kx]) } } }
        ex += pow(Double(dX[yi]) - q, 2); nx += q * q
    }
}
print(String(format: "n=%d %@ forward Y relerr=%.2e  input-gradient dX relerr=%@", n, inferenceOnly ? "inference-only" : "with gradients", sqrt(ey / ny), dX == nil ? "n/a" : String(format: "%.2e", sqrt(ex / nx))))
