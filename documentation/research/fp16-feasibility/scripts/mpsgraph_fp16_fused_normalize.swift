import Foundation
import Metal
import MetalPerformanceShadersGraph
let gdev = MPSGraphDevice(mtlDevice: MTLCreateSystemDefaultDevice()!)
func run(_ values: [Float], _ dtype: MPSDataType, withStats: Bool) -> [Float] {
    let g = MPSGraph(); let n = values.count
    let x32 = g.placeholder(shape: [NSNumber(value: n)], dataType: .float32, name: nil)
    let x = g.cast(x32, to: dtype, name: nil)
    let m = g.mean(of: x, axes: [0], name: nil); let v = g.variance(of: x, mean: m, axes: [0], name: nil)
    let o = g.cast(g.normalize(x, mean: m, variance: v, gamma: nil, beta: nil, epsilon: 1e-5, name: nil), to: .float32, name: nil)
    let targets = withStats ? [g.cast(m, to: .float32, name: nil), g.cast(v, to: .float32, name: nil), o] : [o]
    let td = MPSGraphTensorData(device: gdev, data: values.withUnsafeBufferPointer { Data(buffer: $0) }, shape: [NSNumber(value: n)], dataType: .float32)
    let r = g.run(feeds: [x32: td], targetTensors: targets, targetOperations: nil)
    var a = [Float](repeating: 0, count: n); r[o]!.mpsndarray().readBytes(&a, strideBytes: nil); return a
}
for s: Float in [170, 175, 178, 180, 180.5, 180.9, 181] { let vv = [Float](repeating: s, count: 32) + [Float](repeating: -s, count: 32); print("±\(s)", "fused", Set(run(vv, .float16, withStats: true)).sorted(), "plain", Set(run(vv, .float16, withStats: false)).sorted(), "bf16 fused", Set(run(vv, .bFloat16, withStats: true)).sorted()) }
let v = [Float](repeating: 250, count: 32) + [Float](repeating: -250, count: 32)
for ws in [false, true] { let o = run(v, .float16, withStats: ws); print("fp16 withStats=\(ws): first", o[0], "last", o[63], "distinct", Set(o).sorted()) }
let v2 = [Float](repeating: 100, count: 32) + [Float](repeating: -100, count: 32)
for ws in [false, true] { let o = run(v2, .float16, withStats: ws); print("±100 fp16 withStats=\(ws): distinct", Set(o).sorted()) }
let v3 = [Float](repeating: 16, count: 32) + [Float](repeating: -16, count: 32)
for ws in [false, true] { let o = run(v3, .float16, withStats: ws); print("±16 fp16 withStats=\(ws): distinct", Set(o).sorted()) }
