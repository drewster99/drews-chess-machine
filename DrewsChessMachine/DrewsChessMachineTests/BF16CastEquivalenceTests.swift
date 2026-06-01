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

    /// Decisive check for the input GPU-cast offload. The pre-offload path fed
    /// a bf16 board straight into a bf16 stem conv; the new path feeds fp32 and
    /// narrows with an in-graph `cast` immediately before the same conv. The
    /// standalone cast is bit-exact, but if MPSGraph fuses `cast → conv` with
    /// different rounding/accumulation than a plain bf16-input conv, the whole
    /// forward pass (and tactical move rankings) shifts. This builds both graphs
    /// with identical bf16 weights, feeds the *same* board (bf16 bits to one,
    /// the fp32 source to the other), and asserts the conv outputs are
    /// bit-identical. The conv is the only consumer of the cast, so equality
    /// here means the entire forward is unchanged.
    func testCastThenConvMatchesDirectBF16Conv() throws {
        guard ChessNetwork.dataType == .bFloat16 else {
            throw XCTSkip("bf16 path inactive (dataType=\(ChessNetwork.dataType))")
        }
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue() else {
            throw XCTSkip("Metal not available")
        }
        let inC = ChessNetwork.inputPlanes
        let outC = 128
        let batch = 4
        let side = ChessNetwork.boardSize

        var s: UInt64 = 0x9E3779B97F4A7C15
        func rnd() -> Float {
            s = s &* 6364136223846793005 &+ 1442695040888963407
            return (Float(s >> 40) / Float(1 << 24)) * 2 - 1   // [-1, 1)
        }
        let boardN = batch * inC * side * side
        var boardFP32 = [Float](repeating: 0, count: boardN)
        for i in 0..<boardN { boardFP32[i] = rnd() }
        let wN = outC * inC * 3 * 3
        var wBits = [UInt16](repeating: 0, count: wN)
        for i in 0..<wN { wBits[i] = ChessNetwork.float32ToBFloat16Bits(rnd() * 0.1) }
        let wData = wBits.withUnsafeBytes { Data($0) }

        let wShape: [NSNumber] = [NSNumber(value: outC), NSNumber(value: inC), 3, 3]
        let boardShape: [NSNumber] = [
            NSNumber(value: batch), NSNumber(value: inC),
            NSNumber(value: side), NSNumber(value: side)
        ]
        guard let conv = MPSGraphConvolution2DOpDescriptor(
            strideInX: 1, strideInY: 1, dilationRateInX: 1, dilationRateInY: 1,
            groups: 1, paddingLeft: 1, paddingRight: 1, paddingTop: 1, paddingBottom: 1,
            paddingStyle: .explicit, dataLayout: .NCHW, weightsLayout: .OIHW
        ) else { XCTFail("conv descriptor"); return }

        // OLD path: bf16 placeholder fed the bf16 board directly.
        let gOld = MPSGraph()
        let pOld = gOld.placeholder(shape: boardShape, dataType: .bFloat16, name: "in")
        let wOld = gOld.variable(with: wData, shape: wShape, dataType: .bFloat16, name: "w")
        let outOld = gOld.convolution2D(pOld, weights: wOld, descriptor: conv, name: "conv")
        let boardBF16 = boardFP32.map { ChessNetwork.float32ToBFloat16Bits($0) }
        let oldInND = MPSNDArray(device: device,
                                 descriptor: MPSNDArrayDescriptor(dataType: .bFloat16, shape: boardShape))
        var bbits = boardBF16
        bbits.withUnsafeMutableBytes { if let b = $0.baseAddress { oldInND.writeBytes(b, strideBytes: nil) } }
        let rOld = gOld.run(with: queue, feeds: [pOld: MPSGraphTensorData(oldInND)],
                            targetTensors: [outOld], targetOperations: nil)

        // NEW path: fp32 placeholder -> cast(bf16) -> same conv.
        let gNew = MPSGraph()
        let pNew = gNew.placeholder(shape: boardShape, dataType: .float32, name: "in")
        let cNew = gNew.cast(pNew, to: .bFloat16, name: "board_input_cast")
        let wNew = gNew.variable(with: wData, shape: wShape, dataType: .bFloat16, name: "w")
        let outNew = gNew.convolution2D(cNew, weights: wNew, descriptor: conv, name: "conv")
        let newInND = MPSNDArray(device: device,
                                 descriptor: MPSNDArrayDescriptor(dataType: .float32, shape: boardShape))
        var bf = boardFP32
        bf.withUnsafeMutableBytes { if let b = $0.baseAddress { newInND.writeBytes(b, strideBytes: nil) } }
        let rNew = gNew.run(with: queue, feeds: [pNew: MPSGraphTensorData(newInND)],
                            targetTensors: [outNew], targetOperations: nil)

        guard let dOld = rOld[outOld], let dNew = rNew[outNew] else { XCTFail("no output"); return }
        let outElems = batch * outC * side * side
        var bitsOld = [UInt16](repeating: 0, count: outElems)
        var bitsNew = [UInt16](repeating: 0, count: outElems)
        bitsOld.withUnsafeMutableBytes { if let b = $0.baseAddress { dOld.mpsndarray().readBytes(b, strideBytes: nil) } }
        bitsNew.withUnsafeMutableBytes { if let b = $0.baseAddress { dNew.mpsndarray().readBytes(b, strideBytes: nil) } }

        var mismatches = 0
        var maxAbsDiff: Float = 0
        for i in 0..<outElems where bitsOld[i] != bitsNew[i] {
            mismatches += 1
            let a = ChessNetwork.bFloat16BitsToFloat32(bitsOld[i])
            let b = ChessNetwork.bFloat16BitsToFloat32(bitsNew[i])
            maxAbsDiff = max(maxAbsDiff, abs(a - b))
        }
        XCTAssertEqual(
            mismatches, 0,
            "cast→conv differs from direct bf16 conv on \(mismatches)/\(outElems) elements " +
            "(maxAbsDiff=\(maxAbsDiff)) — the input offload changed the forward pass."
        )
    }

    /// Backward-pass sibling of `testCastThenConvMatchesDirectBF16Conv`. The
    /// offload also inserts the `cast` before the stem conv in the *training*
    /// graph, and MPSGraph builds a separate backward graph — a `cast` ahead of
    /// the conv could in principle make autodiff select a different
    /// weight-gradient kernel. This builds the same two graphs (bf16-direct vs
    /// fp32+cast) with identical bf16 weights, takes the gradient of `sum(conv²)`
    /// w.r.t. the conv weights in each, and asserts the gradients are
    /// bit-identical. `sum(conv²)` (not `sum(conv)`) is used so the weight
    /// gradient actually depends on the conv output, making it sensitive to any
    /// forward *or* backward divergence the cast might introduce.
    func testCastThenConvGradientMatchesDirectBF16Conv() throws {
        guard ChessNetwork.dataType == .bFloat16 else {
            throw XCTSkip("bf16 path inactive (dataType=\(ChessNetwork.dataType))")
        }
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue() else {
            throw XCTSkip("Metal not available")
        }
        let inC = ChessNetwork.inputPlanes
        let outC = 128
        let batch = 4
        let side = ChessNetwork.boardSize

        var s: UInt64 = 0x9E3779B97F4A7C15
        func rnd() -> Float {
            s = s &* 6364136223846793005 &+ 1442695040888963407
            return (Float(s >> 40) / Float(1 << 24)) * 2 - 1   // [-1, 1)
        }
        let boardN = batch * inC * side * side
        var boardFP32 = [Float](repeating: 0, count: boardN)
        for i in 0..<boardN { boardFP32[i] = rnd() }
        let wN = outC * inC * 3 * 3
        var wBits = [UInt16](repeating: 0, count: wN)
        for i in 0..<wN { wBits[i] = ChessNetwork.float32ToBFloat16Bits(rnd() * 0.1) }
        let wData = wBits.withUnsafeBytes { Data($0) }

        let wShape: [NSNumber] = [NSNumber(value: outC), NSNumber(value: inC), 3, 3]
        let boardShape: [NSNumber] = [
            NSNumber(value: batch), NSNumber(value: inC),
            NSNumber(value: side), NSNumber(value: side)
        ]
        guard let conv = MPSGraphConvolution2DOpDescriptor(
            strideInX: 1, strideInY: 1, dilationRateInX: 1, dilationRateInY: 1,
            groups: 1, paddingLeft: 1, paddingRight: 1, paddingTop: 1, paddingBottom: 1,
            paddingStyle: .explicit, dataLayout: .NCHW, weightsLayout: .OIHW
        ) else { XCTFail("conv descriptor"); return }

        // Build a graph whose target is d(sum(conv²))/d(weights), cast to fp32.
        // `wire` maps the placeholder to the conv input (identity for the
        // bf16-direct path, a cast for the fp32 path).
        func gradGraph(
            inputDataType: MPSDataType,
            wire: (MPSGraph, MPSGraphTensor) -> MPSGraphTensor
        ) -> (graph: MPSGraph, input: MPSGraphTensor, gradF32: MPSGraphTensor)? {
            let g = MPSGraph()
            let p = g.placeholder(shape: boardShape, dataType: inputDataType, name: "in")
            let convIn = wire(g, p)
            let w = g.variable(with: wData, shape: wShape, dataType: .bFloat16, name: "w")
            let out = g.convolution2D(convIn, weights: w, descriptor: conv, name: "conv")
            let sq = g.multiplication(out, out, name: "sq")
            // gradients(of: sq) seeds `sq` with ones → d(Σ sq)/d(w) = d(Σ conv²)/d(w).
            let grads = g.gradients(of: sq, with: [w], name: "grads")
            guard let dW = grads[w] else { return nil }
            let dWf32 = g.cast(dW, to: .float32, name: "dW_f32")
            return (g, p, dWf32)
        }

        guard let old = gradGraph(inputDataType: .bFloat16, wire: { _, p in p }) else {
            XCTFail("old gradient missing"); return
        }
        let boardBF16 = boardFP32.map { ChessNetwork.float32ToBFloat16Bits($0) }
        let oldInND = MPSNDArray(device: device,
                                 descriptor: MPSNDArrayDescriptor(dataType: .bFloat16, shape: boardShape))
        var bbits = boardBF16
        bbits.withUnsafeMutableBytes { if let b = $0.baseAddress { oldInND.writeBytes(b, strideBytes: nil) } }
        let rOld = old.graph.run(with: queue, feeds: [old.input: MPSGraphTensorData(oldInND)],
                                 targetTensors: [old.gradF32], targetOperations: nil)

        guard let new = gradGraph(inputDataType: .float32,
                                  wire: { g, p in g.cast(p, to: .bFloat16, name: "board_input_cast") }) else {
            XCTFail("new gradient missing"); return
        }
        let newInND = MPSNDArray(device: device,
                                 descriptor: MPSNDArrayDescriptor(dataType: .float32, shape: boardShape))
        var bf = boardFP32
        bf.withUnsafeMutableBytes { if let b = $0.baseAddress { newInND.writeBytes(b, strideBytes: nil) } }
        let rNew = new.graph.run(with: queue, feeds: [new.input: MPSGraphTensorData(newInND)],
                                 targetTensors: [new.gradF32], targetOperations: nil)

        guard let dOld = rOld[old.gradF32], let dNew = rNew[new.gradF32] else {
            XCTFail("no gradient output"); return
        }
        var gradOld = [Float](repeating: 0, count: wN)
        var gradNew = [Float](repeating: 0, count: wN)
        gradOld.withUnsafeMutableBytes { if let b = $0.baseAddress { dOld.mpsndarray().readBytes(b, strideBytes: nil) } }
        gradNew.withUnsafeMutableBytes { if let b = $0.baseAddress { dNew.mpsndarray().readBytes(b, strideBytes: nil) } }

        var mismatches = 0
        var maxAbsDiff: Float = 0
        for i in 0..<wN where gradOld[i].bitPattern != gradNew[i].bitPattern {
            mismatches += 1
            maxAbsDiff = max(maxAbsDiff, abs(gradOld[i] - gradNew[i]))
        }
        XCTAssertEqual(
            mismatches, 0,
            "cast→conv weight-gradient differs from direct bf16 conv on \(mismatches)/\(wN) elements " +
            "(maxAbsDiff=\(maxAbsDiff)) — the input offload changed the backward pass."
        )
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
