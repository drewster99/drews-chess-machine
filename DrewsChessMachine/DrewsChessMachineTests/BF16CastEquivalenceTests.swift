import XCTest
import MetalPerformanceShadersGraph
@testable import DrewsChessMachine

/// Characterizes MPSGraph's bf16↔fp32 `cast` against the CPU bit helpers
/// (`float32ToBFloat16Bits` / `bFloat16BitsToFloat32`) — the exact math the
/// `writeInferenceInput` / `readFloats` conversion loops use — and guards that
/// the optimized CPU loop is bit-identical to the original conversion.
///
/// Findings (each measured with a *single* cast — see the round-trip caveat):
///  - **bf16 → fp32** (readback) is a lossless widening: bit-exact on GPU and
///    CPU (`testGPUCastBF16ToFP32IsExact`).
///  - **fp32 → bf16** (input) narrows with **round-to-nearest-even**, matching
///    our CPU helper bit-for-bit, including ties-to-even
///    (`testGPUCastFP32ToBF16MatchesCPURoundToEven`). MPSGraph does *not*
///    truncate.
///
/// So the MPSGraph `cast` is a bit-exact substitute for the CPU conversion in
/// both directions. The shipping implementation uses this: the fp32 board
/// input is narrowed to the compute dtype by an in-graph `cast` (the input
/// boundary is fp32, the CPU just memcpys), and the wide policy output is
/// `cast` back to fp32 in-graph so its readback is a raw memcpy too. What
/// remains on the CPU is the bf16→fp32 widen of the small value + WDL
/// readbacks and the trainer's scalar loss reads — too tiny to be worth a GPU
/// dispatch; `testReadFloatsBF16WidenMatchesReference` guards that
/// (bare-pointer `while` loop) against the reference conversion.
///
/// Round-trip caveat: a `cast(cast(x, .bFloat16), .float32)` chain in a single
/// graph does NOT reproduce these per-direction results — MPSGraph's optimizer
/// fuses back-to-back casts. Each boundary in production is a single cast, so
/// the single-cast measurements below are the relevant ones.
final class BF16CastEquivalenceTests: XCTestCase {

    // MARK: - Reference (the original conversion, as an oracle)

    private func cpuBF16BitsToFP32(_ bits: [UInt16]) -> [Float] {
        bits.map { ChessNetwork.bFloat16BitsToFloat32($0) }
    }

    /// A spread of fp32 values: exact-in-bf16 values, both signs, zero, very
    /// small / large magnitudes, deliberate tie cases (low 16 mantissa bits =
    /// 0x8000), and a deterministic pseudo-random spread.
    private func testValues() -> [Float] {
        var v: [Float] = [
            0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 2.0, 0.25, 0.04, -0.04,
            3.14159265, -2.71828, 1e-3, -1e-3, 1e3, -1e3, 1e-30, 1e30,
        ]
        for hi: UInt16 in [0x3F80, 0x3F00, 0xBF80, 0x4040, 0x3DCC] {
            v.append(ChessNetwork.bFloat16BitsToFloat32(hi))
        }
        for hi: UInt16 in [0x3F80, 0x3F00, 0xBF80, 0x4048, 0x3E00] {
            let tieBits = (UInt32(hi) << 16) | 0x0000_8000
            v.append(Float(bitPattern: tieBits))
        }
        var state: UInt64 = 0x9E3779B97F4A7C15
        for _ in 0..<512 {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            let mantissa = Float(state >> 40) / Float(1 << 24)   // [0,1)
            let sign: Float = (state & 1) == 0 ? 1 : -1
            let scale = Float(1) + Float((state >> 8) & 0x3F)     // 1..64
            v.append(sign * mantissa * scale)
        }
        return v
    }

    /// Cast fp32 → bf16 on the GPU and read back the raw bf16 *bits* (a single
    /// cast, no round-trip — avoids the double-cast fusion artifact).
    private func gpuFP32ToBF16Bits(_ values: [Float]) throws -> [UInt16] {
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue() else {
            throw XCTSkip("Metal not available")
        }
        let n = values.count
        let graph = MPSGraph()
        let input = graph.placeholder(shape: [NSNumber(value: n)], dataType: .float32, name: "in")
        let output = graph.cast(input, to: .bFloat16, name: "to_bf16")

        let inDesc = MPSNDArrayDescriptor(dataType: .float32, shape: [NSNumber(value: n)])
        let inND = MPSNDArray(device: device, descriptor: inDesc)
        var local = values
        local.withUnsafeMutableBytes { buf in
            if let base = buf.baseAddress { inND.writeBytes(base, strideBytes: nil) }
        }
        let results = graph.run(with: queue, feeds: [input: MPSGraphTensorData(inND)],
                                targetTensors: [output], targetOperations: nil)
        guard let outData = results[output] else { XCTFail("no output"); return [] }
        var bits = [UInt16](repeating: 0, count: n)
        bits.withUnsafeMutableBytes { buf in
            if let base = buf.baseAddress { outData.mpsndarray().readBytes(base, strideBytes: nil) }
        }
        return bits
    }

    // MARK: - GPU cast characterization

    /// bf16 → fp32 is a lossless widening; the GPU `cast` matches the CPU
    /// `bFloat16BitsToFloat32` bit-for-bit.
    func testGPUCastBF16ToFP32IsExact() throws {
        let bits = testValues().map { ChessNetwork.float32ToBFloat16Bits($0) }
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue() else {
            throw XCTSkip("Metal not available")
        }
        let n = bits.count
        let graph = MPSGraph()
        let input = graph.placeholder(shape: [NSNumber(value: n)], dataType: .bFloat16, name: "in")
        let output = graph.cast(input, to: .float32, name: "to_fp32")

        let inDesc = MPSNDArrayDescriptor(dataType: .bFloat16, shape: [NSNumber(value: n)])
        let inND = MPSNDArray(device: device, descriptor: inDesc)
        var localBits = bits
        localBits.withUnsafeMutableBytes { buf in
            if let base = buf.baseAddress { inND.writeBytes(base, strideBytes: nil) }
        }
        let results = graph.run(with: queue, feeds: [input: MPSGraphTensorData(inND)],
                                targetTensors: [output], targetOperations: nil)
        guard let outData = results[output] else { XCTFail("no output"); return }
        var gpu = [Float](repeating: 0, count: n)
        gpu.withUnsafeMutableBytes { buf in
            if let base = buf.baseAddress { outData.mpsndarray().readBytes(base, strideBytes: nil) }
        }

        let cpu = cpuBF16BitsToFP32(bits)
        for i in 0..<n {
            XCTAssertEqual(
                gpu[i].bitPattern, cpu[i].bitPattern,
                "bf16→fp32 mismatch at \(i): bits=0x\(String(bits[i], radix: 16)) gpu=\(gpu[i]) cpu=\(cpu[i])"
            )
        }
    }

    /// fp32 → bf16: MPSGraph rounds to nearest, ties-to-even — bit-identical
    /// to the CPU `float32ToBFloat16Bits` on every value (no truncation, no
    /// half-ULP correction required).
    func testGPUCastFP32ToBF16MatchesCPURoundToEven() throws {
        let values = testValues()
        let gpu = try gpuFP32ToBF16Bits(values)
        guard !gpu.isEmpty else { return }   // skipped (no Metal)
        for i in 0..<values.count {
            let cpu = ChessNetwork.float32ToBFloat16Bits(values[i])
            XCTAssertEqual(
                gpu[i], cpu,
                "fp32→bf16 mismatch at \(i): in=\(values[i]) " +
                "gpu=0x\(String(gpu[i], radix: 16)) cpu=0x\(String(cpu, radix: 16))"
            )
        }
    }

    /// Regression guard for the bare-pointer rewrite of `readFloats`' bf16
    /// widen. After the GPU-cast offload, the *input* and the wide *policy*
    /// readback go through the graph; what stays on the CPU is the
    /// bf16→fp32 widen for the small value + WDL readbacks and the
    /// trainer's scalar loss reads. This feeds known bf16 bit patterns and
    /// asserts both `readFloats` overloads widen them bit-identically to the
    /// reference `bFloat16BitsToFloat32` (the original loop's math).
    func testReadFloatsBF16WidenMatchesReference() throws {
        guard ChessNetwork.dataType == .bFloat16 else {
            throw XCTSkip("bf16 readFloats path inactive (dataType=\(ChessNetwork.dataType))")
        }
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("Metal not available")
        }
        let bits = testValues().map { ChessNetwork.float32ToBFloat16Bits($0) }
        let n = bits.count

        let desc = MPSNDArrayDescriptor(dataType: .bFloat16, shape: [NSNumber(value: n)])
        let nda = MPSNDArray(device: device, descriptor: desc)
        var localBits = bits
        localBits.withUnsafeMutableBytes { buf in
            if let base = buf.baseAddress { nda.writeBytes(base, strideBytes: nil) }
        }
        let td = MPSGraphTensorData(nda)

        let expected = cpuBF16BitsToFP32(bits)

        let outArray = ChessNetwork.readFloats(from: td, count: n)
        XCTAssertEqual(outArray.count, n)
        for i in 0..<n {
            XCTAssertEqual(
                outArray[i].bitPattern, expected[i].bitPattern,
                "readFloats([Float]) bf16 widen mismatch at \(i): bits=0x\(String(bits[i], radix: 16))"
            )
        }

        var outPtr = [Float](repeating: 0, count: n)
        outPtr.withUnsafeMutableBufferPointer { buf in
            if let base = buf.baseAddress {
                ChessNetwork.readFloats(from: td, into: base, count: n)
            }
        }
        for i in 0..<n {
            XCTAssertEqual(
                outPtr[i].bitPattern, expected[i].bitPattern,
                "readFloats(into:) bf16 widen mismatch at \(i): bits=0x\(String(bits[i], radix: 16))"
            )
        }
    }
}
