import Foundation
import Metal
import MetalPerformanceShadersGraph
let gdev = MPSGraphDevice(mtlDevice: MTLCreateSystemDefaultDevice()!)
func norm(_ values: [Float], _ dtype: MPSDataType) -> [Float] {
    let g = MPSGraph(); let n = values.count
    let x32 = g.placeholder(shape: [NSNumber(value: n)], dataType: .float32, name: nil)
    let x = dtype == .float32 ? x32 : g.cast(x32, to: dtype, name: nil)
    let m = g.mean(of: x, axes: [0], name: nil); let v = g.variance(of: x, mean: m, axes: [0], name: nil)
    var o = g.normalize(x, mean: m, variance: v, gamma: nil, beta: nil, epsilon: 1e-5, name: nil)
    if dtype != .float32 { o = g.cast(o, to: .float32, name: nil) }
    let td = MPSGraphTensorData(device: gdev, data: values.withUnsafeBufferPointer { Data(buffer: $0) }, shape: [NSNumber(value: n)], dataType: .float32)
    let r = g.run(feeds: [x32: td], targetTensors: [o], targetOperations: nil)
    var a = [Float](repeating: 0, count: n); r[o]!.mpsndarray().readBytes(&a, strideBytes: nil); return a
}
func show(_ label: String, _ v: [Float]) {
    let ref = norm(v, .float32)
    for (nm, dt) in [("fp16", MPSDataType.float16), ("bf16", .bFloat16)] {
        let o = norm(v, dt); let e = zip(o, ref).map { abs($0 - $1) }; let i = e.firstIndex(of: e.max()!)!
        print(label, nm, "maxErr", e.max()!, "at", i, "x", v[i], "got", o[i], "fp32", ref[i], "| nonfinite", o.filter { !$0.isFinite }.count)
    }
}
show("±250 n64", [Float](repeating: 250, count: 32) + [Float](repeating: -250, count: 32))
var g = SystemRandomNumberGenerator()
let gs = (0..<4096).map { _ -> Float in let u1 = Float.random(in: 1e-7..<1, using: &g), u2 = Float.random(in: 0..<1, using: &g); return (-2*log(u1)).squareRoot()*cos(2 * .pi*u2) }
show("std1e-2 n4096", gs.map { $0 * 1e-2 })
show("std1e-2 n64", Array(gs.prefix(64)).map { $0 * 1e-2 })
show("std1 n4096", gs)
show("mean31.6 n4096", gs.map { $0 + 31.6 })
show("mean31.6 n64", Array(gs.prefix(64)).map { $0 + 31.6 })
