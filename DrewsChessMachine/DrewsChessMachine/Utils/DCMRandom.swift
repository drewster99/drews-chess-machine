//
//  DCMRandom.swift
//  DrewsChessMachine
//
//  The seeded generator behind every random draw that has to reproduce:
//  replay-buffer sampling, dropout seeding, move sampling, weight
//  initialization (see documentation/plans-active/DETERMINISM_RESUME_LINEAGE_PLAN.md,
//  Part A). Identity values — ModelIDs, corpus IDs, UUIDs — and the Lichess
//  bot's jitter deliberately stay on the system generator: they must differ
//  between reruns of the same seed.
//

import Foundation

/// SplitMix64 (Steele, Lea & Flood 2014), exactly as in the reference
/// `splitmix64.c`. Used for two things only: expanding one seed word into the
/// four state words of `DCMRandom`, and mixing a parent seed with a stream
/// name's hash into a child seed (`DCMRandomStreams.childSeed`). Both uses are
/// pinned by golden tests, so the constants and the operation order here can
/// never change without breaking every recorded seed.
struct DCMSplitMix64: Sendable {
    private(set) var state: UInt64

    init(state: UInt64) {
        self.state = state
    }

    /// Advances the state by the golden-ratio increment and returns the
    /// mixed value — the reference `next()`.
    mutating func next() -> UInt64 {
        state = state &+ 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }

    /// The first output of a SplitMix64 started at `value`: a bijective
    /// mixing function on 64-bit words. This is the `splitmix64(x)` of the
    /// child-seed derivation.
    static func mix(_ value: UInt64) -> UInt64 {
        var generator = DCMSplitMix64(state: value)
        return generator.next()
    }
}

/// Why a `DCMRandom` state could not be built or restored.
enum DCMRandomError: Error, Equatable, LocalizedError {
    /// All four state words are zero: xoshiro256** would output zero forever.
    case allZeroState

    var errorDescription: String? {
        switch self {
        case .allZeroState:
            return "A DCMRandom state cannot be all zero words; xoshiro256** would only ever output zero from it."
        }
    }
}

/// xoshiro256** (Blackman & Vigna 2018) seeded by SplitMix64.
///
/// Every draw that decides training math, or that must match a saved state on
/// resume, uses this type's own draw methods — `nextBounded`,
/// `nextUnitDouble`, `nextUnitFloat` and `stableShuffle` — never the standard
/// library's `using:` algorithms (`Int.random(in:using:)`,
/// `Float.random(in:using:)`, `shuffled(using:)`). Those are not documented as
/// stable across Swift versions: a toolchain upgrade may change how many
/// words they consume and how they map them, which would silently change a
/// "seeded" run. The `RandomNumberGenerator` conformance exists so the
/// standard `using:` APIs still work, and they are acceptable only where no
/// saved state and no golden result depends on the draw sequence.
///
/// A seeded stream also needs a fixed draw order: no draw may happen inside
/// iteration over a `Dictionary` or `Set` (their order differs per process);
/// iterate sorted keys instead.
///
/// Probe isolation: a probe, diagnostic or observer never draws from a
/// training stream (or advances the graph's dropout RNG beyond the step's
/// own advance); it gets its own `probe.<name>.<step>` stream, so turning a
/// probe on or off cannot change a run.
///
/// The state is four 64-bit words. `Codable` writes them as decimal strings,
/// because many JSON readers (Python's included) parse numbers as doubles and
/// would silently drop the low bits of a word above 2^53.
struct DCMRandom: RandomNumberGenerator, Codable, Sendable, Equatable {
    private(set) var s0: UInt64
    private(set) var s1: UInt64
    private(set) var s2: UInt64
    private(set) var s3: UInt64

    /// Expands `seed` into the four state words with SplitMix64, the seeding
    /// the xoshiro authors recommend. SplitMix64's outputs from consecutive
    /// states are distinct, so at most one of the four words can be zero and
    /// the state can never be all zero.
    /// A generator seeded from the system's random source, for draws that
    /// no saved state or reproducible result depends on — interactive play
    /// (UCI, human play, the Lichess bot) and objects that are built with a
    /// generator but never draw from it. Every seeded training stream comes
    /// from `DCMRandomStreams` instead.
    static func seededFromSystem() -> DCMRandom {
        DCMRandom(seed: UInt64.random(in: UInt64.min...UInt64.max))
    }

    init(seed: UInt64) {
        var expander = DCMSplitMix64(state: seed)
        s0 = expander.next()
        s1 = expander.next()
        s2 = expander.next()
        s3 = expander.next()
        precondition(s0 | s1 | s2 | s3 != 0, "SplitMix64 produced an all-zero xoshiro256** state")
    }

    /// Restores a state, e.g. one read back from a checkpoint.
    init(s0: UInt64, s1: UInt64, s2: UInt64, s3: UInt64) throws {
        guard s0 | s1 | s2 | s3 != 0 else { throw DCMRandomError.allZeroState }
        self.s0 = s0
        self.s1 = s1
        self.s2 = s2
        self.s3 = s3
    }

    // MARK: - Generator

    private static func rotateLeft(_ value: UInt64, by count: UInt64) -> UInt64 {
        (value << count) | (value >> (64 - count))
    }

    /// The reference xoshiro256** `next()`.
    mutating func next() -> UInt64 {
        let result = Self.rotateLeft(s1 &* 5, by: 7) &* 9
        let shifted = s1 << 17
        s2 ^= s0
        s3 ^= s1
        s1 ^= s2
        s0 ^= s3
        s2 ^= shifted
        s3 = Self.rotateLeft(s3, by: 45)
        return result
    }

    /// Advances the state by 2^128 draws — the reference `jump()`, for
    /// carving non-overlapping subsequences out of one stream.
    mutating func jump() {
        let jumpPolynomial: [UInt64] = [0x180E_C6D3_3CFD_0ABA, 0xD5A6_1266_F0C9_392C, 0xA958_2618_E03F_C9AA, 0x39AB_DC45_29B1_661C]
        var jumped: (UInt64, UInt64, UInt64, UInt64) = (0, 0, 0, 0)
        // The state advances once per polynomial bit, set or clear; only set
        // bits fold the current state into the result.
        for word in jumpPolynomial {
            for bit in 0..<UInt64(64) {
                if word & (1 << bit) != 0 {
                    jumped.0 ^= s0
                    jumped.1 ^= s1
                    jumped.2 ^= s2
                    jumped.3 ^= s3
                }
                _ = next()
            }
        }
        (s0, s1, s2, s3) = jumped
    }

    // MARK: - Stable draws

    /// A uniform value in `0..<upperBound`, by Lemire's nearly-divisionless
    /// method (2019): unbiased, and usually one draw. The rejection loop's
    /// draw count depends on the values drawn, which is fine — a stream is
    /// restored from its saved state, never by counting draws.
    mutating func nextBounded(_ upperBound: UInt64) -> UInt64 {
        precondition(upperBound > 0, "nextBounded needs a positive bound")
        var product = next().multipliedFullWidth(by: upperBound)
        if product.low < upperBound {
            let threshold = (0 &- upperBound) % upperBound
            while product.low < threshold {
                product = next().multipliedFullWidth(by: upperBound)
            }
        }
        return product.high
    }

    /// A uniform value in `0..<upperBound`; the same draw as the `UInt64` form.
    mutating func nextBounded(_ upperBound: Int) -> Int {
        precondition(upperBound > 0, "nextBounded needs a positive bound")
        return Int(nextBounded(UInt64(upperBound)))
    }

    /// A uniform `Double` in `[0, 1)`: the top 53 bits scaled by 2^-53, so
    /// every value is exactly representable.
    mutating func nextUnitDouble() -> Double {
        Double(next() >> 11) * 0x1p-53
    }

    /// A uniform `Float` in `[0, 1)`: the top 24 bits scaled by 2^-24, so
    /// every value is exactly representable.
    mutating func nextUnitFloat() -> Float {
        Float(next() >> 40) * 0x1p-24
    }

    // MARK: - Codable

    private enum CodingKeys: String, CodingKey {
        case s0, s1, s2, s3
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        func word(_ key: CodingKeys) throws -> UInt64 {
            let text = try container.decode(String.self, forKey: key)
            guard let value = UInt64(strictDecimal: text) else {
                throw DecodingError.dataCorruptedError(
                    forKey: key, in: container,
                    debugDescription: "\"\(text)\" is not a decimal UInt64 state word")
            }
            return value
        }
        try self.init(s0: word(.s0), s1: word(.s1), s2: word(.s2), s3: word(.s3))
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(String(s0), forKey: .s0)
        try container.encode(String(s1), forKey: .s1)
        try container.encode(String(s2), forKey: .s2)
        try container.encode(String(s3), forKey: .s3)
    }
}

extension MutableCollection where Self: RandomAccessCollection {
    /// Fisher–Yates shuffle drawing with `DCMRandom.nextBounded`, from the
    /// last position down: position `i` swaps with a uniform position in
    /// `0...i`. Unlike `shuffle(using:)`, its draw sequence is ours and is
    /// pinned by a golden test.
    mutating func stableShuffle(using generator: inout DCMRandom) {
        guard count > 1 else { return }
        for offset in stride(from: count - 1, to: 0, by: -1) {
            let other = generator.nextBounded(offset + 1)
            swapAt(index(startIndex, offsetBy: offset), index(startIndex, offsetBy: other))
        }
    }
}
