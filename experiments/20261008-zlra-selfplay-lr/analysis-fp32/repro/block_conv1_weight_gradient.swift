import Foundation
import Metal
import MetalPerformanceShadersGraph
struct LCG { var s: UInt64
    mutating func u() -> Float { s = s &* 6364136223846793005 &+ 1442695040888963407; return Float(s >> 40) / Float(1 << 24) }
    mutating func n() -> Float { let a = max(u(), 1e-7), b = u(); return sqrt(-2 * log(a)) * cos(2 * .pi * b) }
}
func bf16(_ x: Float) -> UInt16 { let b = x.bitPattern; return UInt16((b + 0x7FFF + ((b >> 16) & 1)) >> 16) }
func rb(_ v: [Float]) -> [Float] { v.map { Float(bitPattern: UInt32(bf16($0)) << 16) } }
func data(_ v: [Float], _ dt: MPSDataType) -> Data {
    if dt == .float32 { return v.withUnsafeBufferPointer { Data(buffer: $0) } }
    let h = v.map(bf16); return h.withUnsafeBufferPointer { Data(buffer: $0) }
}
func read(_ td: MPSGraphTensorData, count: Int) -> [Float] {
    if td.dataType == .float32 { var o = [Float](repeating: 0, count: count); td.mpsndarray().readBytes(&o, strideBytes: nil); return o }
    var h = [UInt16](repeating: 0, count: count); td.mpsndarray().readBytes(&h, strideBytes: nil)
    return h.map { Float(bitPattern: UInt32($0) << 16) }
}
let device = MTLCreateSystemDefaultDevice()!
let queue = device.makeCommandQueue()!
let C = 128, k = 9
let args = CommandLine.arguments
let n = Int(args[1])!
let dt: MPSDataType = args[2] == "bf16" ? .bFloat16 : .float32
let compiled = args.count > 3 && args[3] == "exe"
let split = args.count > 4 ? Int(args[4])! : 1
func conv(_ g: MPSGraph, _ x: MPSGraphTensor, _ w: MPSGraphTensor) -> MPSGraphTensor {
    let d = MPSGraphConvolution2DOpDescriptor(strideInX: 1, strideInY: 1, dilationRateInX: 1, dilationRateInY: 1, groups: 1, paddingStyle: .explicit, dataLayout: .NCHW, weightsLayout: .OIHW)!
    let p = (k - 1) / 2; d.paddingLeft = p; d.paddingRight = p; d.paddingTop = p; d.paddingBottom = p
    return g.convolution2D(x, weights: w, descriptor: d, name: nil)
}
func silu(_ g: MPSGraph, _ x: MPSGraphTensor) -> MPSGraphTensor { g.multiplication(x, g.sigmoid(with: x, name: nil), name: nil) }
func bn(_ g: MPSGraph, _ x: MPSGraphTensor, _ gamma: MPSGraphTensor, _ beta: MPSGraphTensor) -> MPSGraphTensor {
    g.normalize(x, mean: g.mean(of: x, axes: [0, 2, 3], name: nil), variance: g.variance(of: x, axes: [0, 2, 3], name: nil), gamma: gamma, beta: beta, epsilon: 1e-5, name: nil)
}
var r = LCG(s: 11)
let sh: [NSNumber] = [NSNumber(value: n), NSNumber(value: C), 8, 8]
let g = MPSGraph()
var h = g.constant(data(rb((0..<(n * C * 64)).map { _ in r.n() * 3 + 0.5 }), dt), shape: sh, dataType: dt)
var W1s: [MPSGraphTensor] = [], X1s: [MPSGraphTensor] = [], Z1s: [MPSGraphTensor] = [], all: [MPSGraphTensor] = []
for _ in 0..<3 {
    let cs: [NSNumber] = [1, NSNumber(value: C), 1, 1]
    func vec(_ base: Float) -> MPSGraphTensor { g.variable(with: data(rb((0..<C).map { _ in base + r.n() * 0.3 }), dt), shape: cs, dataType: dt, name: nil) }
    let g1 = vec(1), b1 = vec(0), g2 = vec(1), b2 = vec(0)
    let ws: [NSNumber] = [NSNumber(value: C), NSNumber(value: C), NSNumber(value: k), NSNumber(value: k)]
    let sd = sqrt(2.0 / Float(C * k * k))
    let W1 = g.variable(with: data(rb((0..<(C * C * k * k)).map { _ in r.n() * sd }), dt), shape: ws, dataType: dt, name: nil)
    let W2 = g.variable(with: data(rb((0..<(C * C * k * k)).map { _ in r.n() * sd }), dt), shape: ws, dataType: dt, name: nil)
    let x1 = silu(g, bn(g, h, g1, b1))
    var z1: MPSGraphTensor
    if split == 1 { z1 = conv(g, x1, W1) } else {
        let cg = C / split; var acc: MPSGraphTensor? = nil
        for gi in 0..<split {
            let part = conv(g, g.sliceTensor(x1, dimension: 1, start: gi * cg, length: cg, name: nil), g.sliceTensor(W1, dimension: 1, start: gi * cg, length: cg, name: nil))
            acc = acc.map { g.addition($0, part, name: nil) } ?? part
        }
        z1 = acc!
    }
    var t = silu(g, bn(g, z1, g2, b2))
    t = conv(g, t, W2)
    h = g.addition(h, g.multiplication(t, g.constant(0.5, dataType: dt), name: nil), name: nil)
    W1s.append(W1); X1s.append(x1); Z1s.append(z1); all += [g1, b1, W1, g2, b2, W2]
}
let rr = g.constant(data(rb((0..<(n * C * 64)).map { _ in r.n() }), dt), shape: sh, dataType: dt)
let loss = g.reductionSum(with: g.cast(g.reshape(g.multiplication(h, rr, name: nil), shape: [-1], name: nil), to: .float32, name: nil), axis: 0, name: nil)
let grads = g.gradients(of: loss, with: all + Z1s, name: nil)
let targets = W1s.map { grads[$0]! } + Z1s.map { grads[$0]! } + X1s
var outs: [MPSGraphTensorData]
if compiled {
    let cd = MPSGraphCompilationDescriptor(); cd.optimizationLevel = .level1
    let exe = g.compile(with: MPSGraphDevice(mtlDevice: device), feeds: [:], targetTensors: targets, targetOperations: nil, compilationDescriptor: cd)
    outs = exe.run(with: queue, inputs: [], results: nil, executionDescriptor: nil)
    let t0 = Date(); for _ in 0..<5 { _ = exe.run(with: queue, inputs: [], results: nil, executionDescriptor: nil) }
    print(String(format: "  exe time per run (5 runs, incl. readout targets): %.0f ms", Date().timeIntervalSince(t0) * 200))
} else {
    let res = g.run(with: queue, feeds: [:], targetTensors: targets, targetOperations: nil); outs = targets.map { res[$0]! }
}
let p = (k - 1) / 2
for b in 0..<3 {
    let dW = read(outs[b], count: C * C * k * k), dZ = read(outs[3 + b], count: n * C * 64), X = read(outs[6 + b], count: n * C * 64)
    var err = 0.0, ref = 0.0
    for t in 0..<6 { let o = (t * 37) % C, i = (t * 53 + 7) % C
        for ky in 0..<k { for kx in 0..<k {
            var s = 0.0
            for bb in 0..<n { for yy in 0..<8 { let sy = yy + ky - p; if sy < 0 || sy >= 8 { continue }
                for xx in 0..<8 { let sx = xx + kx - p; if sx < 0 || sx >= 8 { continue }
                    s += Double(dZ[((bb * C + o) * 8 + yy) * 8 + xx]) * Double(X[((bb * C + i) * 8 + sy) * 8 + sx]) } } }
            let idx = ((o * C + i) * k + ky) * k + kx
            err += pow(Double(dW[idx]) - s, 2); ref += s * s
        }}
    }
    print(String(format: "split=\(split) %@ n=%d %@ block %d: graph dW1 vs corr(X1,dZ1) rel err = %.3e", args[2], n, compiled ? "exe" : "run", b, sqrt(err / ref)))
}
