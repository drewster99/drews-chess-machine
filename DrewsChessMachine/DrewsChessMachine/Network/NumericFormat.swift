import Foundation

/// A floating-point format the network can compute in, with the limits the
/// numerics audit checks values against. Mirrors `ComputeDataType`.
enum NumericFormat: String, Codable, CaseIterable, Sendable {
    case fp32
    case bf16
    case fp16

    /// The architecture setting that builds a network computing in this format.
    var computeDataType: ComputeDataType {
        switch self {
        case .fp32: return .float32
        case .bf16: return .bFloat16
        case .fp16: return .float16
        }
    }

    /// Largest finite value.
    var maxFinite: Double {
        switch self {
        case .fp32: return Double(Float.greatestFiniteMagnitude)
        // bf16 keeps fp32's exponent with a shorter fraction, so its largest
        // finite value is fp32's with the dropped fraction bits cleared.
        case .bf16: return Double(Float(bitPattern: 0x7F7F_0000))
        case .fp16: return Double(Float16.greatestFiniteMagnitude)
        }
    }

    /// Smallest positive normal value; below it precision falls off
    /// (subnormals) until values flush to zero.
    var minNormal: Double {
        switch self {
        case .fp32, .bf16: return Double(Float.leastNormalMagnitude)
        case .fp16: return Double(Float16.leastNormalMagnitude)
        }
    }

    /// Stored fraction bits (the implicit leading bit excluded).
    var fractionBits: Int {
        switch self {
        case .fp32: return Float.significandBitCount
        case .bf16: return Float.significandBitCount - 16
        case .fp16: return Float16.significandBitCount
        }
    }

    /// Round an fp32 value to this format and back (round to nearest, ties
    /// to even), as storing it in this format would.
    func round(_ value: Float) -> Float {
        switch self {
        case .fp32:
            return value
        case .bf16:
            return ChessNetwork.bFloat16BitsToFloat32(ChessNetwork.float32ToBFloat16Bits(value))
        case .fp16:
            return Float(Float16(value))
        }
    }

    /// Distance between neighbouring representable values at magnitude `m`:
    /// the step every value of that size is rounded to.
    func step(atMagnitude magnitude: Double) -> Double {
        let smallestExponent = log2(minNormal)
        let exponent = magnitude > 0 ? max(floor(log2(magnitude)), smallestExponent) : smallestExponent
        return pow(2, exponent - Double(fractionBits))
    }
}

/// A numerics-audit check's outcome.
enum NumericsVerdict: String, Codable, Sendable, Comparable {
    case fine
    case degraded
    case bad = "BAD"

    private var severity: Int {
        switch self {
        case .fine: return 0
        case .degraded: return 1
        case .bad: return 2
        }
    }

    static func < (lhs: NumericsVerdict, rhs: NumericsVerdict) -> Bool {
        lhs.severity < rhs.severity
    }
}
