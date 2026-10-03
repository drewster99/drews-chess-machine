//
//  NetworkArchitectureTowerShapeTests.swift
//  DrewsChessMachineTests
//
//  The tower's shape is checked before anything walks it block by block.
//  The Build New Model screen validates whatever the user types, so a
//  negative block count, a count near `Int.max`, or a channel or kernel size
//  whose parameter count does not fit in an `Int` must come back as a
//  validation error. Before the check, `validate()` itself expanded the tower
//  (`expandedBlocks`, through the skip-projection check), so a negative count
//  trapped on a reversed range, a huge count tried to allocate the whole
//  expansion, and an oversized channel count passed validation and then
//  trapped in `parameterCount`.
//
//  There is deliberately no cap on block count, channels or kernel size:
//  anything whose arithmetic fits is a legal architecture, and whether it
//  fits this Mac is a build-time question (`ModelSizeGuidance`).
//

import XCTest
@testable import DrewsChessMachine

final class NetworkArchitectureTowerShapeTests: XCTestCase {

    /// The current preset with its tower replaced by one group per entry of
    /// `counts`, each at `channels` wide (same width, so no skip
    /// projections unless a test changes one).
    private func architecture(counts: [Int], channels: Int = 128) -> NetworkArchitecture {
        var arch = NetworkArchitecture.current
        let base = arch.blockGroups[0]
        arch.blockGroups = counts.map { count in
            var group = base
            group.count = count
            group.channels = channels
            return group
        }
        return arch
    }

    func testNegativeBlockCountIsAValidationError() {
        let arch = architecture(counts: [2, -1])
        XCTAssertThrowsError(try arch.validate()) { error in
            XCTAssertEqual(error as? NetworkArchitectureError,
                           .nonPositive(field: "blockGroups[1].count", value: -1))
        }
    }

    func testAnIntMaxBlockCountIsRefusedWithoutExpandingTheTower() {
        let arch = architecture(counts: [Int.max])
        XCTAssertThrowsError(try arch.validate()) { error in
            XCTAssertTrue(String(describing: error).contains("overflow"),
                          "an Int.max-block tower's parameter count cannot be represented: \(error)")
        }
    }

    func testTwoNearMaxBlockCountsDoNotOverflow() {
        let half = Int.max / 2 + 1
        let arch = architecture(counts: [half, half])
        XCTAssertThrowsError(try arch.validate()) { error in
            XCTAssertTrue(String(describing: error).contains("overflow"),
                          "the total block count overflows Int: \(error)")
        }
    }

    func testAChannelCountWhoseParameterCountOverflowsIsRefused() {
        let arch = architecture(counts: [5], channels: 1 << 31)
        XCTAssertThrowsError(try arch.validate()) { error in
            XCTAssertTrue(String(describing: error).contains("overflow"),
                          "a conv of (2^31)^2 · 7 · 7 weights overflows Int: \(error)")
        }
    }

    func testAKernelSizeWhoseParameterCountOverflowsIsRefused() {
        var arch = architecture(counts: [5])
        arch.blockGroups[0].conv1KernelSize = (1 << 31) + 1
        XCTAssertThrowsError(try arch.validate()) { error in
            XCTAssertTrue(String(describing: error).contains("overflow"),
                          "a (2^31+1)^2 kernel at 128 channels overflows Int: \(error)")
        }
    }

    /// A tower far too large for any Mac but whose arithmetic fits is a
    /// legal architecture: it validates, and its parameter count is computed
    /// group by group, never by walking every block.
    func testAHugeRepresentableTowerValidatesWithoutBeingExpanded() throws {
        let blocks = 1 << 44
        let one = architecture(counts: [1], channels: 16)
        let two = architecture(counts: [2], channels: 16)
        let huge = architecture(counts: [blocks], channels: 16)
        try huge.validate()
        let perBlock = two.parameterCount - one.parameterCount
        XCTAssertEqual(huge.parameterCount, one.parameterCount + (blocks - 1) * perBlock)
        XCTAssertEqual(huge.numBlocks, blocks)
    }

    // MARK: - The per-group formulas agree with the block-by-block ones

    /// Towers that exercise every case of the per-group rules: uniform,
    /// width staircases (first block of a group projects), a group of one
    /// block at the end, and the final-block feature skip widening a
    /// multi-block last group or a single-block last group.
    private var variedArchitectures: [NetworkArchitecture] {
        var result = NetworkArchitecture.Preset.allCases.map { NetworkArchitecture.preset($0) }
        for counts in [[2, 3, 1], [1, 1, 4], [3, 2, 2]] {
            for finalBlockSkip in [false, true] {
                var arch = architecture(counts: counts, channels: 32)
                for (index, width) in [16, 32, 32].enumerated() {
                    arch.blockGroups[index].channels = width
                }
                if finalBlockSkip {
                    arch.featureSkipSource = .stemOutput
                    arch.featureSkipFusion = .concatDirect
                    arch.featureSkipToFinalBlock = true
                }
                result.append(arch)
            }
        }
        return result
    }

    func testGroupsWithSkipProjectionMatchesPerBlockIndices() throws {
        for arch in variedArchitectures {
            try arch.validate()
            let projectedBlocks = arch.skipProjectionBlockIndices
            let expected = Set(arch.blockGroups.indices.filter { group in
                let range = arch.blockRange(ofGroup: group)
                return projectedBlocks.contains { range.contains($0) }
            })
            XCTAssertEqual(arch.groupsWithSkipProjection, expected, arch.architectureSummary)
        }
    }

    func testParameterCountBreakdownMatchesTheWeightPlan() throws {
        for arch in variedArchitectures {
            try arch.validate()
            var blockToGroup: [Int] = []
            for (group, spec) in arch.blockGroups.enumerated() {
                blockToGroup += Array(repeating: group, count: spec.count)
            }
            var perGroup = Array(repeating: 0, count: arch.blockGroups.count)
            var stem = 0, towerEndBN = 0, featureSkip = 0, policy = 0, value = 0, total = 0
            for spec in arch.weightTensorPlan() {
                total += spec.elementCount
                if spec.name.hasPrefix("stem.") {
                    stem += spec.elementCount
                } else if spec.name.hasPrefix("blocks.") {
                    let digits = spec.name.dropFirst("blocks.".count).prefix(while: \.isNumber)
                    let block = try XCTUnwrap(Int(digits))
                    perGroup[blockToGroup[block]] += spec.elementCount
                } else if spec.name.hasPrefix("tower_final_bn") {
                    towerEndBN += spec.elementCount
                } else if spec.name.hasPrefix("feature_skip.") {
                    featureSkip += spec.elementCount
                } else if spec.name.hasPrefix("policy.") {
                    policy += spec.elementCount
                } else if spec.name.hasPrefix("value.") {
                    value += spec.elementCount
                } else {
                    XCTFail("unclassified tensor \(spec.name)")
                }
            }
            let breakdown = arch.parameterCountBreakdown
            XCTAssertEqual(breakdown, NetworkArchitecture.ParameterCountBreakdown(
                stem: stem, perGroup: perGroup, towerEndBN: towerEndBN,
                featureSkip: featureSkip, policy: policy, value: value, total: total
            ), arch.architectureSummary)
        }
    }

    /// The skip-projection rows of the Build screen are drawn for any count
    /// the user types; a group with no blocks has no projection and passes
    /// its input width through.
    func testGroupsWithSkipProjectionIsDefinedForInvalidCounts() {
        var arch = architecture(counts: [1, -1, 0, 2], channels: 32)
        arch.blockGroups[0].channels = 16
        arch.blockGroups[1].channels = 64
        XCTAssertEqual(arch.groupsWithSkipProjection, [3])
        XCTAssertEqual(architecture(counts: [Int.max, Int.max]).groupsWithSkipProjection, [])
    }

    func testValidateTowerShapeReportsAnOverflowingTotal() {
        let half = Int.max / 2 + 1
        XCTAssertThrowsError(try architecture(counts: [half, half]).validateTowerShape()) { error in
            XCTAssertEqual(error as? NetworkArchitectureError,
                           .arithmeticOverflow(quantity: "the total block count"))
        }
    }
}
