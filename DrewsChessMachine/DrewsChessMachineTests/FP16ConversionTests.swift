import XCTest
import Accelerate
import Metal
import MetalPerformanceShaders
import MetalPerformanceShadersGraph
@testable import DrewsChessMachine

/// Guards the fp16 (`ComputeDataType.float16`) host-side conversion paths.
///
/// fp16 weights/feeds are narrowed two different ways in the codebase, and the
/// two MUST agree bit-for-bit or a scalar feed and a bulk feed of the same
/// value would diverge:
///  - **vImage** (`vImageConvert_PlanarFtoPlanar16F`) in `makeWeightData` and
///    `ChessTrainer.writeRealValuedFeed`'s staging fill.
///  - **native `Float16`** (`Float16(x).bitPattern`) in
///    `ChessTrainer.writeScalarFeed`.
///
/// Both are IEEE-754 round-to-nearest-even; `testFP16VImageNarrowMatchesNative`
/// pins that. The widen side (half → fp32) is a lossless conversion used by
/// `readFloats`; `testFP16RoundTripWidenMatchesNative` pins it against the
/// standard-library widen. `testMPSDataTypeMappingIsExhaustive` guards the
/// `ComputeDataType → MPSDataType` switch stays total and correct.
final class FP16ConversionTests: XCTestCase {

    /// A spread of fp32 values exercising both signs, zero, exact-in-fp16
    /// values, subnormal/underflow magnitudes, overflow-to-inf magnitudes,
    /// deliberate round-to-even tie cases, and a deterministic pseudo-random
    /// fill in fp16's normal range.
    private func testValues() -> [Float] {
        var v: [Float] = [
            0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 2.0, 0.25, 0.04, -0.04,
            3.14159265, -2.71828, 1e-3, -1e-3, 1e3, -1e3,
            // fp16 edges: max finite (65504), just over (→ +inf), tiny
            // (→ subnormal), far underflow (→ 0).
            65504.0, -65504.0, 70000.0, -70000.0, 6.1035e-5, 1e-8, -1e-8, 1e30,
        ]
        // Tie cases: an fp32 value exactly halfway between two fp16 values
        // (the dropped mantissa bits are 0x1000 in fp16's 13-dropped-bit grid).
        // Build by widening an fp16 bit pattern then nudging the fp32 mantissa.
        for halfBits: UInt16 in [0x3C00, 0x3800, 0xBC00, 0x4900, 0x0400] {
            let base = Float(Float16(bitPattern: halfBits))
            let midpoint = base.nextUp
            v.append((base + midpoint) * 0.5)
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

    /// vImage narrowing (the bulk weight/feed path) must equal native
    /// `Float16` narrowing (the scalar feed path) bit-for-bit, so the two
    /// conversion sites never disagree on the same value.
    func testFP16VImageNarrowMatchesNative() {
        let values = testValues()
        let count = values.count
        var viaVImage = [UInt16](repeating: 0, count: count)
        values.withUnsafeBufferPointer { src in
            viaVImage.withUnsafeMutableBufferPointer { dst in
                var srcImg = vImage_Buffer(
                    data: UnsafeMutableRawPointer(mutating: src.baseAddress),
                    height: 1,
                    width: vImagePixelCount(count),
                    rowBytes: count * MemoryLayout<Float>.size
                )
                var dstImg = vImage_Buffer(
                    data: dst.baseAddress,
                    height: 1,
                    width: vImagePixelCount(count),
                    rowBytes: count * MemoryLayout<UInt16>.size
                )
                _ = vImageConvert_PlanarFtoPlanar16F(&srcImg, &dstImg, 0)
            }
        }
        for i in 0..<count {
            let native = Float16(values[i]).bitPattern
            // -0.0 and +0.0 both narrow to a zero half; treat the two zero
            // encodings as equal (sign of zero is not load-bearing for feeds).
            let bothZero = (viaVImage[i] & 0x7FFF) == 0 && (native & 0x7FFF) == 0
            XCTAssertTrue(
                viaVImage[i] == native || bothZero,
                "fp16 narrow mismatch at \(values[i]): vImage=0x\(String(viaVImage[i], radix: 16)) native=0x\(String(native, radix: 16))"
            )
        }
    }

    /// The fp16 weight round-trip (`makeWeightData` → `decodeWeightData`) must
    /// widen back to exactly what the standard-library `Float(Float16(x))`
    /// produces — the widen is lossless, so the only loss is the single narrow.
    func testFP16RoundTripWidenMatchesNative() {
        let values = testValues()
        let data = ChessNetwork.makeWeightData(values, dataType: .float16)
        XCTAssertEqual(data.count, values.count * 2, "fp16 packs to 2 bytes/element")
        let decoded = ChessNetwork.decodeWeightData(data, dataType: .float16)
        XCTAssertEqual(decoded.count, values.count)
        for i in 0..<values.count {
            let reference = Float(Float16(values[i]))
            if reference.isNaN {
                XCTAssertTrue(decoded[i].isNaN, "expected NaN at \(values[i])")
            } else {
                XCTAssertEqual(
                    decoded[i], reference, accuracy: 0,
                    "fp16 round-trip at \(values[i]): got \(decoded[i]) want \(reference)"
                )
            }
        }
    }

    /// The compute-dtype → MPSDataType switch must stay total and map each
    /// case to its matching MPS type. A new `ComputeDataType` case forces this
    /// to be revisited (the `arch(for:)` builder fails to compile otherwise).
    func testMPSDataTypeMappingIsExhaustive() throws {
        for dtype in ComputeDataType.allCases {
            let arch = try makeArch(dtype)
            let mapped = ChessNetwork.mpsDataType(for: arch)
            switch dtype {
            case .float32: XCTAssertEqual(mapped, .float32)
            case .bFloat16: XCTAssertEqual(mapped, .bFloat16)
            case .float16: XCTAssertEqual(mapped, .float16)
            }
        }
    }

    /// fp16 element width is 2 bytes, like bf16, and its relative epsilon is
    /// the 10-bit-mantissa value the save-verification tolerance keys off.
    func testFP16WeightMetadata() {
        XCTAssertEqual(ChessNetwork.bytesPerWeightElement(for: .float16), 2)
        XCTAssertEqual(ChessNetwork.weightRelativeEpsilon(for: .float16), 0x1p-10)
    }

    /// The two `readFloats` overloads' fp16 GPU-readback branches must both
    /// widen half → fp32 exactly as the standard library does. This is the
    /// hot inference path (value + WDL readback in `evaluate(batchBoards:)`),
    /// so a regression here silently corrupts every fp16 evaluation. Mirrors
    /// `BF16CastEquivalenceTests.testReadFloatsBF16WidenMatchesReference`; not
    /// gated on the default arch's dtype since the dtype is passed explicitly.
    func testReadFloatsFP16WidenMatchesReference() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("Metal not available")
        }
        let bits = testValues().map { Float16($0).bitPattern }
        let n = bits.count

        let desc = MPSNDArrayDescriptor(dataType: .float16, shape: [NSNumber(value: n)])
        let nda = MPSNDArray(device: device, descriptor: desc)
        var localBits = bits
        localBits.withUnsafeMutableBytes { buf in
            if let base = buf.baseAddress { nda.writeBytes(base, strideBytes: nil) }
        }
        let td = MPSGraphTensorData(nda)

        // Reference widen: the lossless half → fp32 the stdlib produces.
        let expected = bits.map { Float(Float16(bitPattern: $0)) }

        let outArray = ChessNetwork.readFloats(from: td, count: n, dataType: .float16)
        XCTAssertEqual(outArray.count, n)
        for i in 0..<n {
            XCTAssertEqual(
                outArray[i].bitPattern, expected[i].bitPattern,
                "readFloats([Float]) fp16 widen mismatch at \(i): bits=0x\(String(bits[i], radix: 16))"
            )
        }

        var outPtr = [Float](repeating: 0, count: n)
        outPtr.withUnsafeMutableBufferPointer { buf in
            if let base = buf.baseAddress {
                ChessNetwork.readFloats(from: td, into: base, count: n, dataType: .float16)
            }
        }
        for i in 0..<n {
            XCTAssertEqual(
                outPtr[i].bitPattern, expected[i].bitPattern,
                "readFloats(into:) fp16 widen mismatch at \(i): bits=0x\(String(bits[i], radix: 16))"
            )
        }
    }

    /// A `.float16` architecture must survive the JSON round-trip that backs
    /// safetensors metadata + session save/load — i.e. the new enum case
    /// encodes to its `"float16"` key and decodes back. Guards the persistence
    /// path for fp16 models without depending on which presets ship fp16.
    func testFP16ArchitectureJSONRoundTrips() throws {
        var arch = NetworkArchitecture.current
        arch.computeDataType = .float16
        let data = try JSONEncoder().encode(arch)
        let json = String(decoding: data, as: UTF8.self)
        XCTAssertTrue(
            json.contains("\"compute_data_type\":\"float16\""),
            "expected float16 compute_data_type key in JSON; got: \(json)"
        )
        let decoded = try JSONDecoder().decode(NetworkArchitecture.self, from: data)
        XCTAssertEqual(decoded.computeDataType, .float16)
        XCTAssertEqual(decoded, arch)
    }

    /// Build the current default architecture with `computeDataType` overridden
    /// to the requested dtype, so the mapping test covers every case without
    /// depending on which presets happen to use which precision.
    private func makeArch(_ dtype: ComputeDataType) throws -> NetworkArchitecture {
        var arch = NetworkArchitecture.current
        arch.computeDataType = dtype
        return arch
    }
}
