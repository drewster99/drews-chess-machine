import Foundation
import Metal
import MetalPerformanceShadersGraph
let gdev = MPSGraphDevice(mtlDevice: MTLCreateSystemDefaultDevice()!)
// Do fp16 subnormals (below 6.1e-5) survive MPSGraph elementwise ops, matmul and reductions?
let vals: [Float] = [3e-5, 1e-5, 1e-6, 1e-7, 6e-8]
let g = MPSGraph(); let n = vals.count
let x32 = g.placeholder(shape: [1, NSNumber(value: n)], dataType: .float32, name: nil)
let x = g.cast(x32, to: .float16, name: nil)
let one = g.constant(1.0, dataType: .float16)
let mul = g.multiplication(x, one, name: nil)
let add = g.addition(x, g.constant(0.0, dataType: .float16), name: nil)
let eye = g.constant(1.0, shape: [1, 1], dataType: .float16)
let mm = g.matrixMultiplication(primary: g.transpose(x, permutation: [1, 0], name: nil), secondary: eye, name: nil)
let sum = g.reductionSum(with: x, axis: 1, name: nil)
let big = g.multiplication(x, g.constant(1000.0, dataType: .float16), name: nil)
let outs = [x, mul, add, mm, sum, big].map { g.cast($0, to: .float32, name: nil) }
let td = MPSGraphTensorData(device: gdev, data: vals.withUnsafeBufferPointer { Data(buffer: $0) }, shape: [1, NSNumber(value: n)], dataType: .float32)
let r = g.run(feeds: [x32: td], targetTensors: outs, targetOperations: nil)
for (name, t) in zip(["cast", "x*1", "x+0", "matmul", "sum", "x*1000"], outs) {
    let c = name == "sum" ? 1 : n; var a = [Float](repeating: 0, count: c); r[t]!.mpsndarray().readBytes(&a, strideBytes: nil); print(name, a)
}
print("expected sum", vals.reduce(0, +), "expected x*1000", vals.map { $0 * 1000 })
