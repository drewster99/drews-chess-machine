// Y = conv(X, W) built several ways; loss = sum(R * Y); Y is also a graph target (the trigger).
// dW from the graph vs exact float64 corr(X, R) (dY = R exactly, so the reference needs no MPSGraph gradient).
import Foundation
import Metal
import MetalPerformanceShadersGraph
struct LCG { var s: UInt64
    mutating func u() -> Float { s = s &* 6364136223846793005 &+ 1442695040888963407; return Float(s >> 40) / Float(1 << 24) }
    mutating func n() -> Float { let a = max(u(), 1e-7), b = u(); return sqrt(-2 * log(a)) * cos(2 * .pi * b) }
}
func h16(_ x: Float, _ dt: MPSDataType) -> UInt16 {
    if dt == .float16 { return Float16(x).bitPattern }
    let b = x.bitPattern; return UInt16((b + 0x7FFF + ((b >> 16) & 1)) >> 16)
}
func back(_ h: UInt16, _ dt: MPSDataType) -> Float { dt == .float16 ? Float(Float16(bitPattern: h)) : Float(bitPattern: UInt32(h) << 16) }
func rnd(_ v: [Float], _ dt: MPSDataType) -> [Float] { dt == .float32 ? v : v.map { back(h16($0, dt), dt) } }
func data(_ v: [Float], _ dt: MPSDataType) -> Data {
    if dt == .float32 { return v.withUnsafeBufferPointer { Data(buffer: $0) } }
    let h = v.map { h16($0, dt) }; return h.withUnsafeBufferPointer { Data(buffer: $0) }
}
func read(_ td: MPSGraphTensorData, _ count: Int, _ dt: MPSDataType) -> [Float] {
    if dt == .float32 { var o = [Float](repeating: 0, count: count); td.mpsndarray().readBytes(&o, strideBytes: nil); return o }
    var h = [UInt16](repeating: 0, count: count); td.mpsndarray().readBytes(&h, strideBytes: nil); return h.map { back($0, dt) }
}
let device = MTLCreateSystemDefaultDevice()!
let queue = device.makeCommandQueue()!
let a = CommandLine.arguments
let n = Int(a[1])!, k = Int(a[2])!, C = Int(a[3])!, variant = a[4]
let dt: MPSDataType = a[5] == "bf16" ? .bFloat16 : (a[5] == "fp16" ? .float16 : .float32)
var r = LCG(s: 3)
let xv = rnd((0..<(n * C * 64)).map { _ in r.n() }, dt)
let wv = rnd((0..<(C * C * k * k)).map { _ in r.n() * 0.03 }, dt)
let rv = rnd((0..<(n * C * 64)).map { _ in r.n() * 0.1 }, dt)
let g = MPSGraph()
let sh: [NSNumber] = [NSNumber(value: n), NSNumber(value: C), 8, 8]
let X = g.constant(data(xv, dt), shape: sh, dataType: dt)
let W = g.variable(with: data(wv, dt), shape: [NSNumber(value: C), NSNumber(value: C), NSNumber(value: k), NSNumber(value: k)], dataType: dt, name: nil)
let R = g.constant(data(rv, dt), shape: sh, dataType: dt)
let d = MPSGraphConvolution2DOpDescriptor(strideInX: 1, strideInY: 1, dilationRateInX: 1, dilationRateInY: 1, groups: 1, paddingStyle: .explicit, dataLayout: .NCHW, weightsLayout: .OIHW)!
let p = (k - 1) / 2; d.paddingLeft = p; d.paddingRight = p; d.paddingTop = p; d.paddingBottom = p
var Y: MPSGraphTensor
var extraTarget: MPSGraphTensor
switch variant {
case "plain":
    Y = g.convolution2D(X, weights: W, descriptor: d, name: nil); extraTarget = Y
case "identity":   // BN-side consumers read an identity copy; the loss path reads Y
    Y = g.convolution2D(X, weights: W, descriptor: d, name: nil); extraTarget = g.identity(with: Y, name: nil)
case "split2", "split4":   // same conv over batch slices, concatenated: each slice's weight gradient is below 2^31
    let parts = variant == "split2" ? 2 : 4
    let slices = g.split(X, numSplits: parts, axis: 0, name: nil)
    Y = g.concatTensors(slices.map { g.convolution2D($0, weights: W, descriptor: d, name: nil) }, dimension: 0, name: nil)
    extraTarget = Y
case "chan2", "chan4":   // split input channels: Y = sum_g conv(X[:, g], W[:, g])
    let G = variant == "chan2" ? 2 : 4, cg = C / G
    var acc: MPSGraphTensor? = nil
    for gi in 0..<G {
        let xs = g.sliceTensor(X, dimension: 1, start: gi * cg, length: cg, name: nil)
        let ws = g.sliceTensor(W, dimension: 1, start: gi * cg, length: cg, name: nil)
        let part = g.convolution2D(xs, weights: ws, descriptor: d, name: nil)
        acc = acc.map { g.addition($0, part, name: nil) } ?? part
    }
    Y = acc!; extraTarget = Y
case "plusZero":
    let c0 = g.convolution2D(X, weights: W, descriptor: d, name: nil)
    Y = g.addition(c0, g.constant(0, dataType: dt), name: nil); extraTarget = Y
case "noTarget":
    Y = g.convolution2D(X, weights: W, descriptor: d, name: nil); extraTarget = X
default: fatalError()
}
let loss = g.reductionSum(with: g.cast(g.reshape(g.multiplication(Y, R, name: nil), shape: [-1], name: nil), to: .float32, name: nil), axis: 0, name: nil)
let gW = g.gradients(of: loss, with: [W], name: nil)[W]!
let res = g.run(with: queue, feeds: [:], targetTensors: [gW, extraTarget], targetOperations: nil)
let dW = read(res[gW]!, wv.count, dt)
var err = 0.0, ref = 0.0
for t in 0..<4 { let o = (t * 37) % C, i = (t * 53 + 7) % C
    for ky in 0..<k { for kx in 0..<k {
        var s = 0.0
        for b in 0..<n { for yy in 0..<8 { let sy = yy + ky - p; if sy < 0 || sy >= 8 { continue }
            for xx in 0..<8 { let sx = xx + kx - p; if sx < 0 || sx >= 8 { continue }
                s += Double(rv[((b * C + o) * 8 + yy) * 8 + xx]) * Double(xv[((b * C + i) * 8 + sy) * 8 + sx]) } } }
        let idx = ((o * C + i) * k + ky) * k + kx
        err += pow(Double(dW[idx]) - s, 2); ref += s * s
    }}
}
print(String(format: "%@ n=%5d k=%d C=%d %-9@ elements=%.3e relerr=%.3e", a[5], n, k, C, variant, Double(n * 64 * C * k * k), sqrt(err / ref)))
