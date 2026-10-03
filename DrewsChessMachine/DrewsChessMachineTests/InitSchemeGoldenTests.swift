//
//  InitSchemeGoldenTests.swift
//  DrewsChessMachineTests
//
//  Pins of the published `dcm-init-1` weight-initialization scheme
//  (determinism plan B1.1). Every value here was computed by an independent
//  Python implementation of the scheme (generator, child-seed derivation,
//  normal transform, fill order, fp32 scaling, bf16 rounding). A change to any
//  of those breaks these pins — which is the point: a published scheme never
//  changes; a different algorithm needs a new scheme ID.
//

import CryptoKit
import MetalPerformanceShaders
import XCTest
@testable import DrewsChessMachine

final class InitSchemeGoldenTests: XCTestCase {

    func testSchemeIdentifier() {
        XCTAssertEqual(WeightInitScheme.current, "dcm-init-1")
    }

    func testTensorSeedsUnderInitSeedFortyTwo() {
        let expected: [(String, UInt64)] = [
            ("stem.conv.weight", 0xdd01_3ebe_fbb3_b66f),
            ("blocks.0.conv1.weight", 0x6bae_333f_965b_7ff7),
            ("blocks.3.se_scalebias.fc2.weight", 0x00bb_3eb6_c04d_0524),
            ("value.wdl_fc2.weight", 0xcd02_29de_c427_6c5f),
            ("policy.conv.weight", 0x15f7_2813_6dd0_86da),
            ("value.fc1.weight", 0x898a_6dba_3e87_9a96),
        ]
        for (name, seed) in expected {
            XCTAssertEqual(WeightInitScheme.tensorSeed(initSeed: 42, tensorName: name), seed, name)
        }
        XCTAssertEqual(WeightInitScheme.bnCalibrationSeed(initSeed: 42), 0x5a98_1c36_fdcb_f5bb)
    }

    func testConvTensorGoldens() throws {
        let spec = WeightTensorSpec(name: "blocks.0.conv1.weight", shape: [16, 8, 3, 3], kind: .conv)
        let values = try WeightInitScheme.nativeValues(initSeed: 42, spec: spec, distribution: .heNormal)
        try assertGoldens(
            values,
            first8: [0x3da25fba, 0x3eb31c6b, 0xbea52cfa, 0x3dc7d003, 0xbd6513c8, 0x3cfdbf32, 0x3e2870b2, 0x3e1517b4],
            sha256: "5ddf942c0cadcc37b9565b3364d4f7b1f1faf9f9270a7d0619f397e091c6ec51",
            bf16First8: [0x3da2, 0x3eb3, 0xbea5, 0x3dc8, 0xbd65, 0x3cfe, 0x3e28, 0x3e15],
            bf16SHA256: "d9934bd290f7f1b3965419994f28f231f62cccdc6f1293e341b943d501168ab9")
    }

    /// An FC weight: drawn row-major in the on-disk `[out, in]` layout, then
    /// transposed to the native `[in, out]` order these goldens are in.
    func testFCTensorGoldens() throws {
        let spec = WeightTensorSpec(name: "value.fc1.weight", shape: [64, 32], kind: .linear)
        let values = try WeightInitScheme.nativeValues(initSeed: 42, spec: spec, distribution: .heNormal)
        try assertGoldens(
            values,
            first8: [0x3ec813a8, 0xbd13d85d, 0x3dd096cb, 0xbce5e117, 0x3e34ac8e, 0xbe84a868, 0x3dbd2ab1, 0xbc35fdff],
            sha256: "03d373d1eeb22d73480509f9f63c4b76fc0d9a5f69afaa0bc99d3cbc265a47a4",
            bf16First8: [0x3ec8, 0xbd14, 0x3dd1, 0xbce6, 0x3e35, 0xbe85, 0x3dbd, 0xbc36],
            bf16SHA256: "1556020eb054d0db90a11ee675ce0e1d07a4d40a6113da04e3c7a59324203e81")
    }

    /// A zero-β SE FC2: the Glorot draw of the whole matrix with its β
    /// columns zeroed.
    func testZeroBetaSEFC2Goldens() throws {
        var group = NetworkArchitecture.current.blockGroups[0]
        group.channels = 16
        group.seStyle = .scaleAndBias
        group.seReductionRatio = 4
        group.seBetaInit = .zero
        let spec = WeightTensorSpec(name: "blocks.0.se_scalebias.fc2.weight", shape: [4, 32], kind: .linear)
        let values = try WeightInitScheme.seFC2NativeValues(initSeed: 42, spec: spec, group: group)
        try assertGoldens(
            values,
            first8: [0xbd1e90d8, 0x3e632111, 0xbc0496cf, 0x3cdd72e8, 0x3de97445, 0xbe8ab424, 0xbdab78d2, 0xbe2988a9],
            sha256: "37af67e49e000b529ebc8a200fd752cf473d1d9466410c11db820dac01191c98",
            bf16First8: [0xbd1f, 0x3e63, 0xbc05, 0x3cdd, 0x3de9, 0xbe8b, 0xbdab, 0xbe2a],
            bf16SHA256: "26bc1b10a8c421be6190f97dbf1fa0ea21388b4d1ec975554f3f1bfdd2dd04af")
        for row in 0..<4 {
            for column in 16..<32 {
                XCTAssertEqual(values[row * 32 + column].bitPattern, 0, "β column \(column) of row \(row)")
            }
        }
    }

    /// An odd element count drops the last pair's second normal.
    func testOddCountTensorGoldens() throws {
        let spec = WeightTensorSpec(name: "odd.conv", shape: [1, 1, 1, 3], kind: .conv)
        let values = try WeightInitScheme.nativeValues(initSeed: 7, spec: spec, distribution: .heNormal)
        XCTAssertEqual(values.map(\.bitPattern), [0x3f042de5, 0xbeb7ec6b, 0x3f80a8a2])
    }

    /// Kinds with no random role are refused rather than given a made-up
    /// distribution.
    func testNonRandomKindsAreRefused() {
        let bias = WeightTensorSpec(name: "value.fc1.bias", shape: [1, 32], kind: .bias)
        XCTAssertThrowsError(try WeightInitScheme.nativeValues(initSeed: 1, spec: bias, distribution: .heNormal)) { error in
            XCTAssertEqual(error as? WeightInitError, .notRandomlyInitialized(name: "value.fc1.bias", kind: .bias))
        }
    }

    /// Per-role standard deviation from the tensor's own fans.
    func testRoleStandardDeviations() throws {
        let conv = WeightTensorSpec(name: "c", shape: [32, 16, 3, 3], kind: .conv)
        XCTAssertEqual(try WeightInitScheme.standardDeviation(of: conv, distribution: .heNormal),
                       (2.0 / Float(16 * 9)).squareRoot())
        let fc = WeightTensorSpec(name: "f", shape: [64, 128], kind: .linear)
        XCTAssertEqual(try WeightInitScheme.standardDeviation(of: fc, distribution: .heNormal),
                       (2.0 / Float(64)).squareRoot())
        XCTAssertEqual(try WeightInitScheme.standardDeviation(of: fc, distribution: .glorotNormal),
                       (2.0 / Float(64 + 128)).squareRoot())
        let big = WeightTensorSpec(name: "big.conv", shape: [128, 128, 3, 3], kind: .conv)
        let values = try WeightInitScheme.nativeValues(initSeed: 3, spec: big, distribution: .heNormal)
        let mean = values.reduce(0.0) { $0 + Double($1) } / Double(values.count)
        let variance = values.reduce(0.0) { $0 + (Double($1) - mean) * (Double($1) - mean) } / Double(values.count)
        let expected = Double((2.0 / Float(128 * 9)).squareRoot())
        XCTAssertEqual(variance.squareRoot(), expected, accuracy: expected * 0.02)
    }

    // MARK: Helpers

    private func assertGoldens(
        _ values: [Float], first8: [UInt32], sha256: String, bf16First8: [UInt16], bf16SHA256: String,
        file: StaticString = #filePath, line: UInt = #line
    ) throws {
        XCTAssertEqual(values.prefix(8).map(\.bitPattern), first8, "fp32 first values", file: file, line: line)
        let fp32Data = ChessNetwork.makeWeightData(values, dataType: .float32)
        XCTAssertEqual(Self.hex(SHA256.hash(data: fp32Data)), sha256, "fp32 tensor SHA-256", file: file, line: line)
        let bf16Data = ChessNetwork.makeWeightData(values, dataType: .bFloat16)
        let bf16Words = bf16Data.withUnsafeBytes { Array($0.bindMemory(to: UInt16.self)) }
        XCTAssertEqual(Array(bf16Words.prefix(8)), bf16First8, "bf16 first values", file: file, line: line)
        XCTAssertEqual(Self.hex(SHA256.hash(data: bf16Data)), bf16SHA256, "bf16 tensor SHA-256", file: file, line: line)
    }

    private static func hex(_ digest: SHA256.Digest) -> String {
        digest.map { String(format: "%02x", $0) }.joined()
    }
}
