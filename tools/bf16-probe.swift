#!/usr/bin/env swift
//
//  bf16-probe.swift
//
//  Standalone diagnostic: reports every signal we might use to decide whether to
//  run the chess network's GPU compute in bfloat16 vs float32 on THIS machine.
//
//  Run on a few different Macs (M1, M2, M3, Intel if any) and compare:
//
//      swift bf16-probe.swift                 # defaults: N=2048, depth=8, iters=20
//      swift bf16-probe.swift 4096 8 30       # N depth iters overrides
//
//  It prints three independent capability signals plus a real timing:
//    1. MTLGPUFamily.supportsFamily(...)  — incl. .apple6 (bfloat TYPE) and
//       .apple8 (~M2, our proxy for native bf16 arithmetic)
//    2. sysctl hw.optional.arm.FEAT_BF16  — the CPU NEON "native bf16" flag (M2=1)
//    3. a chained-matmul micro-benchmark, fp32 vs bf16, GFLOPS + speedup ratio
//
//  The speedup ratio is the only thing that measures the real criterion
//  ("is bf16 actually faster here?"); the family/sysctl flags are the cheap
//  static proxies we're validating against it.
//

import Foundation
import Metal
import MetalPerformanceShaders
import MetalPerformanceShadersGraph

// MARK: - sysctl helpers

func sysctlInt(_ name: String) -> Int? {
    var size = 0
    guard sysctlbyname(name, nil, &size, nil, 0) == 0, size > 0 else { return nil }
    var buf = [UInt8](repeating: 0, count: size)
    guard sysctlbyname(name, &buf, &size, nil, 0) == 0 else { return nil }
    switch size {
    case 4: return Int(buf.withUnsafeBytes { $0.load(as: Int32.self) })
    case 8: return Int(buf.withUnsafeBytes { $0.load(as: Int64.self) })
    default: return nil
    }
}

func sysctlString(_ name: String) -> String? {
    var size = 0
    guard sysctlbyname(name, nil, &size, nil, 0) == 0, size > 0 else { return nil }
    var buf = [CChar](repeating: 0, count: size)
    guard sysctlbyname(name, &buf, &size, nil, 0) == 0 else { return nil }
    return String(cString: buf)
}

// MARK: - args

let args = CommandLine.arguments
let N = args.count > 1 ? (Int(args[1]) ?? 2048) : 2048
let depth = args.count > 2 ? (Int(args[2]) ?? 8) : 8
let iters = args.count > 3 ? (Int(args[3]) ?? 20) : 20
let warmup = 3

// MARK: - device + environment

guard let device = MTLCreateSystemDefaultDevice() else {
    print("No Metal device."); exit(1)
}
guard let queue = device.makeCommandQueue() else {
    print("No command queue."); exit(1)
}

let osv = ProcessInfo.processInfo.operatingSystemVersion

print("=== Environment ===")
print("  Device:            \(device.name)")
print("  Chip:              \(sysctlString("machdep.cpu.brand_string") ?? "?")")
print("  hw.model:          \(sysctlString("hw.model") ?? "?")")
print("  macOS:             \(osv.majorVersion).\(osv.minorVersion).\(osv.patchVersion)")
print("  unifiedMemory:     \(device.hasUnifiedMemory)")
print("")

print("=== Signal 1: MTLGPUFamily.supportsFamily ===")
let families: [(String, MTLGPUFamily)] = [
    ("apple6 (bfloat TYPE floor)", .apple6),
    ("apple7 (~M1)",               .apple7),
    ("apple8 (~M2, native-bf16 proxy)", .apple8),
    ("apple9 (~M3)",               .apple9),
    ("mac2",                       .mac2),
    ("metal3",                     .metal3),
]
for (label, fam) in families {
    print("  \(device.supportsFamily(fam) ? "YES" : "no ")  \(label)")
}
// Speculative scan for GPU families newer than this SDK's named cases — e.g. an
// M5-only matrix-unit family that would distinguish it from M4 (which reports
// "native" on every other signal but has no fast GPU bf16 matmul). rawValue init
// is failable, so this compiles on any SDK and only prints families it knows.
print("  -- speculative higher families (rawValue probe) --")
for raw in [1010, 1011, 1012, 2003, 5002, 5003] {
    if let fam = MTLGPUFamily(rawValue: raw) {
        print("    \(device.supportsFamily(fam) ? "YES" : "no ")  rawValue \(raw)")
    }
}
print("")

print("=== Signal 2: sysctl native-bf16 flags ===")
let featBF16 = sysctlInt("hw.optional.arm.FEAT_BF16")
print("  hw.optional.arm.FEAT_BF16:  \(featBF16.map(String.init) ?? "absent")  (1 = native bf16, M2+)")
print("  hw.optional.arm.FEAT_FP16:  \(sysctlInt("hw.optional.arm.FEAT_FP16").map(String.init) ?? "absent")")
print("")

// MARK: - Signal 3: micro-benchmark (chained matmul, fp32 vs bf16)

func makeInputND(_ n: Int) -> MPSGraphTensorData {
    // Scale entries to ~1/sqrt(n) so chained matmuls preserve O(1) magnitude
    // (no overflow that would skew bf16 timing/results).
    let scale = 1.0 / Float(n).squareRoot()
    var data = [Float](repeating: 0, count: n * n)
    for i in 0..<(n * n) { data[i] = Float.random(in: -scale...scale) }
    let desc = MPSNDArrayDescriptor(dataType: .float32, shape: [NSNumber(value: n), NSNumber(value: n)])
    let nd = MPSNDArray(device: device, descriptor: desc)
    data.withUnsafeMutableBytes { raw in
        guard let base = raw.baseAddress else { return }
        nd.writeBytes(base, strideBytes: nil)
    }
    return MPSGraphTensorData(nd)
}

func benchmark(_ compute: MPSDataType, n: Int, depth: Int, iters: Int) -> Double {
    let graph = MPSGraph()
    let a = graph.placeholder(shape: [NSNumber(value: n), NSNumber(value: n)], dataType: .float32, name: "a")
    let b = graph.placeholder(shape: [NSNumber(value: n), NSNumber(value: n)], dataType: .float32, name: "b")
    let aC = compute == .float32 ? a : graph.cast(a, to: compute, name: "ac")
    let bC = compute == .float32 ? b : graph.cast(b, to: compute, name: "bc")
    var m = aC
    for _ in 0..<depth { m = graph.matrixMultiplication(primary: m, secondary: bC, name: nil) }
    let out = compute == .float32 ? m : graph.cast(m, to: .float32, name: "out")

    let feeds = [a: makeInputND(n), b: makeInputND(n)]

    func runOnce() {
        autoreleasepool {
            _ = graph.run(with: queue, feeds: feeds, targetTensors: [out], targetOperations: nil)
        }
    }

    for _ in 0..<warmup { runOnce() }

    var times: [Double] = []
    for _ in 0..<iters {
        let t0 = DispatchTime.now().uptimeNanoseconds
        runOnce()
        let t1 = DispatchTime.now().uptimeNanoseconds
        times.append(Double(t1 - t0) / 1e9)
    }
    times.sort()
    return times[times.count / 2]   // median seconds
}

print("=== Signal 3: micro-benchmark (chained matmul) ===")
print("  config: N=\(N), depth=\(depth) matmuls, iters=\(iters) (warmup \(warmup))")

let flopsPerIter = 2.0 * Double(N) * Double(N) * Double(N) * Double(depth)  // 2N^3 per matmul
let tF32 = benchmark(.float32, n: N, depth: depth, iters: iters)
let tBF16 = benchmark(.bFloat16, n: N, depth: depth, iters: iters)
let gflopsF32 = flopsPerIter / tF32 / 1e9
let gflopsBF16 = flopsPerIter / tBF16 / 1e9
let speedup = tF32 / tBF16

print(String(format: "  fp32:   %.2f ms/iter   %.0f GFLOP/s", tF32 * 1e3, gflopsF32))
print(String(format: "  bf16:   %.2f ms/iter   %.0f GFLOP/s", tBF16 * 1e3, gflopsBF16))
print(String(format: "  bf16 speedup vs fp32:  %.2fx", speedup))
print("")

print("=== Verdict ===")
let familyNative = device.supportsFamily(.apple8)
let sysctlNative = (featBF16 ?? 0) == 1
print("  apple8 (family proxy):   \(familyNative ? "native-likely" : "not-native")")
print("  FEAT_BF16 (sysctl):      \(sysctlNative ? "native" : "not-native/absent")")
print(String(format: "  benchmark says bf16 is:  %@ (%.2fx)",
             speedup > 1.10 ? "FASTER" : (speedup < 0.95 ? "SLOWER" : "same/marginal"), speedup))
print("  -> Use bf16 if the benchmark shows a real speedup; the two flags should agree on M2+.")
