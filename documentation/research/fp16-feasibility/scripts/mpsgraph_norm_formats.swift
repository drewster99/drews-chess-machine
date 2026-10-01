import Foundation
import Metal
import MetalPerformanceShadersGraph

guard let device = MTLCreateSystemDefaultDevice() else { fatalError("no Metal device") }
let gdev = MPSGraphDevice(mtlDevice: device)

// Feed fp32, cast in-graph to the format under test (as our network casts its input), run
// mean/variance/normalize in that format, widen results to fp32 for readback.
func run(_ values: [Float], _ dtype: MPSDataType, eps: Double = 1e-5) -> (mean: Float, variance: Float, normalized: [Float]) {
    let g = MPSGraph()
    let n = values.count
    let x32 = g.placeholder(shape: [NSNumber(value: n)], dataType: .float32, name: "x")
    let x = dtype == .float32 ? x32 : g.cast(x32, to: dtype, name: "cast")
    let mean = g.mean(of: x, axes: [0], name: "mean")
    let variance = g.variance(of: x, mean: mean, axes: [0], name: "var")
    let norm = g.normalize(x, mean: mean, variance: variance, gamma: nil, beta: nil, epsilon: Float(eps), name: "norm")
    let w = { (t: MPSGraphTensor) in dtype == .float32 ? t : g.cast(t, to: .float32, name: nil) }
    let (m32, v32, n32) = (w(mean), w(variance), w(norm))
    let td = MPSGraphTensorData(device: gdev, data: values.withUnsafeBufferPointer { Data(buffer: $0) }, shape: [NSNumber(value: n)], dataType: .float32)
    let out = g.run(feeds: [x32: td], targetTensors: [m32, v32, n32], targetOperations: nil)
    func read(_ t: MPSGraphTensor, _ c: Int) -> [Float] { var a = [Float](repeating: 0, count: c); out[t]!.mpsndarray().readBytes(&a, strideBytes: nil); return a }
    return (read(m32, 1)[0], read(v32, 1)[0], read(n32, n))
}

func exact(_ v: [Float], eps: Double = 1e-5) -> [Double] {
    let d = v.map(Double.init); let m = d.reduce(0, +) / Double(d.count)
    let s = d.map { ($0 - m) * ($0 - m) }.reduce(0, +) / Double(d.count)
    return d.map { ($0 - m) / (s + eps).squareRoot() }
}

func report(_ label: String, _ values: [Float], eps: Double = 1e-5) {
    let ref = exact(values, eps: eps)
    var line = "\(label):"
    for (name, dt) in [("fp32", MPSDataType.float32), ("fp16", .float16), ("bf16", .bFloat16)] {
        let r = run(values, dt, eps: eps)
        let err = zip(r.normalized, ref).map { abs(Double($0) - $1) }.max()!
        line += " | \(name) var \(r.variance) maxErr \(String(format: "%.3g", err))"
    }
    print(line)
}

var rng = SystemRandomNumberGenerator()
func gauss(_ n: Int, mean: Float, std: Float) -> [Float] {
    (0..<n).map { _ in let u1 = Float.random(in: 1e-7..<1, using: &rng), u2 = Float.random(in: 0..<1, using: &rng)
        return mean + std * (-2 * log(u1)).squareRoot() * cos(2 * .pi * u2) }
}

print("== variance boundary (fp16 max 65504): ±s gives variance s²")
for s: Float in [250, 255, 256, 257, 260] { report(String(format: "±%.0f (var %.0f)", s, s*s), [Float](repeating: s, count: 32) + [Float](repeating: -s, count: 32)) }
print("== isolated large values (x² overflows alone)")
report("one 377 among 63×0.5", [377] + [Float](repeating: 0.5, count: 63))
report("one 1000 among 63×0", [1000] + [Float](repeating: 0, count: 63))
print("== mean ≫ std (cancellation); 4096 samples, std 1")
for ratio: Float in [1, 8, 31.6, 100, 1000] { report(String(format: "|mean|/std %.1f", ratio), gauss(4096, mean: ratio, std: 1)) }
print("== tiny variance vs epsilon 1e-5 (fp16-subnormal)")
for std: Float in [1e-1, 1e-2, 3e-3, 1e-3] { report(String(format: "std %.0e", std), gauss(4096, mean: 0, std: std)) }
print("== typical activation, 128 channels (LayerNorm-like), std 1, one hot channel 30")
report("std 1 + hot 30", gauss(127, mean: 0, std: 1) + [30])
