//
//  StandardPathForwardPinTests.swift
//  DrewsChessMachineTests
//
//  Pins the forward-pass output of the standard weight-storage path (every
//  variable in the compute dtype) bit for bit. The values were recorded from
//  the build that still carried the fp32-storage / cast-in-forward option, so
//  these tests prove that removing that option left the default graph's
//  arithmetic untouched: same weights in, same bits out.
//
//  The weights are a deterministic function of each tensor's position in the
//  architecture's weight plan (running variances kept positive), so no random
//  source is involved anywhere.
//

import XCTest
import Metal
@testable import DrewsChessMachine

final class StandardPathForwardPinTests: XCTestCase {

    private func requireMetal() throws {
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("Metal not available") }
    }

    private struct Fixture {
        let label: String
        let dtype: ComputeDataType
        let policyStyle: PolicyHeadStyle
        let tail: ChessNetwork.PolicyTailPrecision
    }

    /// FNV-1a over the bit patterns of every policy logit and the value scalar.
    private static func fingerprint(policy: UnsafeBufferPointer<Float>, value: Float) -> UInt64 {
        var hash: UInt64 = 0xcbf2_9ce4_8422_2325
        func mix(_ bits: UInt32) {
            for shift in stride(from: 0, to: 32, by: 8) {
                hash ^= UInt64((bits >> UInt32(shift)) & 0xff)
                hash = hash &* 0x0000_0100_0000_01b3
            }
        }
        for logit in policy { mix(logit.bitPattern) }
        mix(value.bitPattern)
        return hash
    }

    private static func deterministicWeights(for arch: NetworkArchitecture) -> [[Float]] {
        arch.weightTensorPlan().enumerated().map { tensorIndex, spec in
            if spec.name.hasSuffix("running_var") {
                return (0..<spec.elementCount).map { Float(1 + ($0 + tensorIndex) % 5) * 0.25 }
            }
            return (0..<spec.elementCount).map { element in
                Float((tensorIndex * 31 + element * 17) % 23 - 11) * 0.01
            }
        }
    }

    private func forwardFingerprint(_ fixture: Fixture) async throws -> UInt64 {
        var arch = NetworkArchitecture.current
        arch.computeDataType = fixture.dtype
        arch.policyHeadStyle = fixture.policyStyle
        let net = try ChessNetwork(arch: arch, policyTailPrecision: fixture.tail)
        try await net.loadWeights(Self.deterministicWeights(for: arch))
        let board = BoardEncoder.encode(.starting, encoding: arch.inputEncoding)
        let result = SyncBox<UInt64?>(nil)
        try await net.evaluate(board: board) { policy, value in
            result.value = Self.fingerprint(policy: policy, value: value)
        }
        guard let fingerprint = result.value else {
            XCTFail("\(fixture.label): evaluate returned without calling its consumer")
            return 0
        }
        return fingerprint
    }

    /// Recorded from the build before the option's removal.
    private static let pinned: [String: UInt64] = [
        "bf16 current mixed": 0x966f2ababab684cd,
        "bf16 current fp32-tail": 0x23959220b867ff1,
        "bf16 fc_bottleneck fp32-tail": 0xeb8f57854b4559ba,
        "fp32 current mixed": 0xc4fd75f94037a94f,
    ]

    func testStandardPathForwardIsBitIdenticalToThePinnedOutputs() async throws {
        try requireMetal()
        let fixtures: [Fixture] = [
            .init(label: "bf16 current mixed", dtype: .bFloat16, policyStyle: NetworkArchitecture.current.policyHeadStyle, tail: .mixedFinalProjection),
            .init(label: "bf16 current fp32-tail", dtype: .bFloat16, policyStyle: NetworkArchitecture.current.policyHeadStyle, tail: .float32FromPreBatchNorm),
            .init(label: "bf16 fc_bottleneck fp32-tail", dtype: .bFloat16, policyStyle: .fcBottleneck, tail: .float32FromPreBatchNorm),
            .init(label: "fp32 current mixed", dtype: .float32, policyStyle: NetworkArchitecture.current.policyHeadStyle, tail: .mixedFinalProjection),
        ]
        for fixture in fixtures {
            let fingerprint = try await forwardFingerprint(fixture)
            XCTAssertEqual(Self.pinned[fixture.label], fingerprint,
                           "\(fixture.label): got 0x\(String(fingerprint, radix: 16))")
        }
    }
}
