import XCTest
import Accelerate
import MetalPerformanceShaders
@testable import DrewsChessMachine

/// The engine narrows fp32 → fp16/bf16 at more than one site:
///   - fp16 scalar feed: native `Float16(v).bitPattern` (`writeScalarFeed`)
///   - fp16 bulk: vImage `PlanarFtoPlanar16F` (`makeWeightData` / the
///     `writeRealValuedFeed` staging path)
///   - bf16 scalar + bulk: `ChessNetwork.float32ToBFloat16Bits`
///     (`writeScalarFeed`, `makeWeightData`)
/// If any two of these disagreed on the same value, a weight saved/loaded
/// through one path would differ from the same value fed through another —
/// silent, hard-to-find drift. This suite pins every production narrowing
/// (and the widen back) to agree across the whole route matrix, over a value
/// battery covering both signs, zeros, subnormals, round-to-even ties,
/// overflow-to-inf, infinities, and NaN.
final class FP16Bf16NarrowEquivalenceTests: XCTestCase {

    /// fp32 values exercising the corners of both half formats.
    private func values() -> [Float] {
        var v: [Float] = [
            0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 2.0, 0.25, 0.04, -0.04,
            3.14159265, -2.71828, 1e-3, -1e-3, 1e3, -1e3,
            65504.0, -65504.0, 70000.0, -70000.0,      // fp16 max / overflow→inf
            6.1035e-5, 1e-8, -1e-8, 1e30, -1e30,        // fp16 subnormal / underflow / bf16-range
            .infinity, -.infinity, .nan,
        ]
        // fp16 round-to-even tie cases (dropped mantissa exactly 0x1000).
        for halfBits: UInt16 in [0x3C00, 0x3800, 0xBC00, 0x4900, 0x0400] {
            let base = Float(Float16(bitPattern: halfBits))
            v.append((base + base.nextUp) * 0.5)
        }
        // Deterministic spread through both formats' normal ranges.
        var state: UInt64 = 0x9E3779B97F4A7C15
        for _ in 0..<256 {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            let mantissa = Float(state >> 40) / Float(1 << 24)
            let sign: Float = (state & 1) == 0 ? 1 : -1
            let scale = Float(1) + Float((state >> 8) & 0x3F)
            v.append(sign * mantissa * scale)
        }
        return v
    }

    private func isFP16NaN(_ bits: UInt16) -> Bool { (bits & 0x7C00) == 0x7C00 && (bits & 0x03FF) != 0 }
    private func isBF16NaN(_ bits: UInt16) -> Bool { (bits & 0x7F80) == 0x7F80 && (bits & 0x007F) != 0 }
    private func isZero(_ bits: UInt16) -> Bool { (bits & 0x7FFF) == 0 }

    private func makeFP16Bits(_ v: Float) -> UInt16 {
        ChessNetwork.makeWeightData([v], dataType: .float16)
            .withUnsafeBytes { $0.loadUnaligned(fromByteOffset: 0, as: UInt16.self) }
    }
    private func makeBF16Bits(_ v: Float) -> UInt16 {
        ChessNetwork.makeWeightData([v], dataType: .bFloat16)
            .withUnsafeBytes { $0.loadUnaligned(fromByteOffset: 0, as: UInt16.self) }
    }

    /// fp16: the native scalar narrow (`Float16.bitPattern`) and the bulk
    /// vImage narrow (`makeWeightData`) must produce identical half bits.
    func testFP16ScalarAndBulkNarrowAgree() {
        for v in values() {
            let scalar = Float16(v).bitPattern
            let bulk = makeFP16Bits(v)
            if v.isNaN {
                XCTAssertTrue(isFP16NaN(scalar) && isFP16NaN(bulk),
                              "fp16 NaN narrow: scalar=0x\(String(scalar, radix: 16)) bulk=0x\(String(bulk, radix: 16))")
            } else if scalar != bulk {
                // The only legitimate divergence is the sign of zero.
                XCTAssertTrue(isZero(scalar) && isZero(bulk),
                              "fp16 narrow mismatch at \(v): scalar=0x\(String(scalar, radix: 16)) bulk=0x\(String(bulk, radix: 16))")
            }
        }
    }

    /// bf16: the scalar/bulk narrow both route through `float32ToBFloat16Bits`,
    /// so `makeWeightData` must equal it bit-for-bit. Guards against a future
    /// `makeWeightData` that swaps in a different bf16 narrowing.
    func testBF16ScalarAndBulkNarrowAgree() {
        for v in values() {
            let scalar = ChessNetwork.float32ToBFloat16Bits(v)
            let bulk = makeBF16Bits(v)
            if v.isNaN {
                XCTAssertTrue(isBF16NaN(scalar) && isBF16NaN(bulk),
                              "bf16 NaN narrow: scalar=0x\(String(scalar, radix: 16)) bulk=0x\(String(bulk, radix: 16))")
            } else {
                XCTAssertEqual(scalar, bulk,
                               "bf16 narrow mismatch at \(v): scalar=0x\(String(scalar, radix: 16)) bulk=0x\(String(bulk, radix: 16))")
            }
        }
    }

    /// Bulk narrowing one element at a time must equal narrowing the whole
    /// array at once, for both formats — i.e. each element converts
    /// independently with no cross-element contamination in the vImage/loop path.
    func testBulkNarrowIsElementwiseIndependent() {
        let vs = values()
        for dtype in [MPSDataType.float16, .bFloat16] {
            let batch = ChessNetwork.makeWeightData(vs, dataType: dtype)
            let batchBits: [UInt16] = batch.withUnsafeBytes { Array($0.bindMemory(to: UInt16.self)) }
            XCTAssertEqual(batchBits.count, vs.count, "\(dtype) packs one half per element")
            for (i, v) in vs.enumerated() {
                let single = (dtype == .float16) ? makeFP16Bits(v) : makeBF16Bits(v)
                XCTAssertEqual(batchBits[i], single,
                               "\(dtype) elementwise mismatch at index \(i) (\(v)): batch=0x\(String(batchBits[i], radix: 16)) single=0x\(String(single, radix: 16))")
            }
        }
    }

    /// Round-trip (narrow then widen) through the production weight codec must
    /// equal the reference widen of the narrowed value, for both formats — the
    /// widen is lossless, so the only loss is the single narrow.
    func testRoundTripWidenMatchesReference() {
        let vs = values()
        // fp16 reference: standard-library Float16 round-trip.
        let fp16Decoded = ChessNetwork.decodeWeightData(
            ChessNetwork.makeWeightData(vs, dataType: .float16), dataType: .float16)
        // bf16 reference: the engine's own bit-exact widen of its own narrow.
        let bf16Decoded = ChessNetwork.decodeWeightData(
            ChessNetwork.makeWeightData(vs, dataType: .bFloat16), dataType: .bFloat16)
        for (i, v) in vs.enumerated() {
            let fp16Ref = Float(Float16(v))
            if fp16Ref.isNaN {
                XCTAssertTrue(fp16Decoded[i].isNaN, "fp16 expected NaN at \(v)")
            } else {
                XCTAssertEqual(fp16Decoded[i], fp16Ref, accuracy: 0, "fp16 round-trip at \(v)")
            }
            let bf16Ref = ChessNetwork.bFloat16BitsToFloat32(ChessNetwork.float32ToBFloat16Bits(v))
            if bf16Ref.isNaN {
                XCTAssertTrue(bf16Decoded[i].isNaN, "bf16 expected NaN at \(v)")
            } else {
                XCTAssertEqual(bf16Decoded[i], bf16Ref, accuracy: 0, "bf16 round-trip at \(v)")
            }
        }
    }
}
