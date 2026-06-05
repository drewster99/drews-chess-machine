//
//  ArchitectureConfigTests.swift
//  DrewsChessMachineTests
//
//  architecture.json round-trip + validation-on-load.
//

import XCTest
@testable import DrewsChessMachine

final class ArchitectureConfigTests: XCTestCase {

    func testRoundTripThroughFile() throws {
        let arch = NetworkArchitecture.preset(.v4_8block_3x3)
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("arch_rt_\(UUID().uuidString).json")
        defer { do { try FileManager.default.removeItem(at: url) } catch {} }

        try ArchitectureConfig.writeTemplate(arch, to: url)
        let loaded = try ArchitectureConfig.load(from: url)
        XCTAssertEqual(loaded, arch)
    }

    func testLoadRejectsInvalidArchitecture() throws {
        // Encode an invalid arch (even kernel) directly, bypassing validate(),
        // then confirm load() validates and rejects it.
        var bad = NetworkArchitecture.current
        bad.towerConvKernelSize = 4
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("arch_bad_\(UUID().uuidString).json")
        defer { do { try FileManager.default.removeItem(at: url) } catch {} }

        try JSONEncoder().encode(bad).write(to: url)
        XCTAssertThrowsError(try ArchitectureConfig.load(from: url))
    }
}
