//
//  NetworkArchitectureTests.swift
//  DrewsChessMachineTests
//
//  Phase 1 of the safetensors / runtime-architecture work. NetworkArchitecture
//  is a pure value type that reproduces — independently of the live graph
//  builder — the derived quantities currently computed from ChessNetwork's
//  static constants: parameter count, arch hash, and the ordered weight-tensor
//  plan. These tests pin those derivations to the documented, known-good values
//  so the struct can't silently drift from what ChessNetwork actually builds.
//

import XCTest
@testable import DrewsChessMachine

final class NetworkArchitectureTests: XCTestCase {

    // MARK: archHash reproduces documented values

    func testArchHashMatchesDocumentedCurrent() {
        // 5-block 7×7 v4 — the current champion architecture.
        XCTAssertEqual(NetworkArchitecture.preset(.v4_5block_7x7).archHash, 0xdf23_a86c)
        XCTAssertEqual(NetworkArchitecture.preset(.v4_5block_7x7).archHashHex, "0xdf23a86c")
    }

    func testArchHashMatchesDocumented12Block() {
        // 12-block 3×3 v4 — the bf16 baseline ("Session A").
        XCTAssertEqual(NetworkArchitecture.preset(.v4_12block_3x3).archHash, 0xbad3_2ced)
    }

    func testArchHashIgnoresKernelAndSEByDesign() {
        // archHash is a coarse tag: kernel size and SE shape are NOT mixed in,
        // so the 5-block 7×7 and a hypothetical 5-block 3×3 share a hash. The
        // explicit embedded config carries precise identity.
        var a = NetworkArchitecture.preset(.v4_5block_7x7)
        var b = a
        b.towerConvKernelSize = 3
        XCTAssertEqual(a.archHash, b.archHash)
        // But block count IS mixed in.
        a.numBlocks = 6
        XCTAssertNotEqual(a.archHash, b.archHash)
    }

    // MARK: parameterCount reproduces known values

    func testParameterCountKnownValues() {
        XCTAssertEqual(NetworkArchitecture.preset(.v4_5block_7x7).parameterCount, 8_445_748)
        XCTAssertEqual(NetworkArchitecture.preset(.v4_12block_3x3).parameterCount, 3_898_139)
        // Predicted 8-block v4 3×3 (the proposed re-run).
        XCTAssertEqual(NetworkArchitecture.preset(.v4_8block_3x3).parameterCount, 2_664_087)
    }

    /// parameterCount must equal the summed element counts of the weight plan —
    /// this cross-validates every shape in the plan against the param formula.
    func testParameterCountEqualsPlanElementSum() {
        for p in NetworkArchitecture.Preset.allCases {
            let arch = NetworkArchitecture.preset(p)
            let planSum = arch.weightTensorPlan().reduce(0) { $0 + $1.elementCount }
            XCTAssertEqual(planSum, arch.parameterCount, "preset \(p.rawValue)")
        }
    }

    // MARK: weightTensorPlan structure

    func testPlanTensorCountFormula() {
        // v4: 25 fixed tensors + 15 per block (matches README weight-tensor counts).
        for p in NetworkArchitecture.Preset.allCases {
            let arch = NetworkArchitecture.preset(p)
            XCTAssertEqual(arch.weightTensorPlan().count, 25 + 15 * arch.numBlocks, "preset \(p.rawValue)")
        }
        XCTAssertEqual(NetworkArchitecture.preset(.v4_5block_7x7).weightTensorPlan().count, 100)
        XCTAssertEqual(NetworkArchitecture.preset(.v4_12block_3x3).weightTensorPlan().count, 205)
        XCTAssertEqual(NetworkArchitecture.preset(.v4_8block_3x3).weightTensorPlan().count, 145)
    }

    func testPlanOrderTrainablesThenRunningStats() throws {
        // exportWeights / loadWeights order = all trainables, then all BN
        // running stats. So no running-stat tensor may precede a non-running
        // tensor.
        let plan = NetworkArchitecture.current.weightTensorPlan()
        XCTAssertEqual(plan.first?.name, "stem.conv.weight")
        let firstRunning = try XCTUnwrap(plan.firstIndex { $0.kind == .bnRunningStat })
        let lastNonRunning = try XCTUnwrap(plan.lastIndex { $0.kind != .bnRunningStat })
        XCTAssertLessThan(lastNonRunning, firstRunning)
    }

    func testPlanNamesAreUnique() {
        for p in NetworkArchitecture.Preset.allCases {
            let names = NetworkArchitecture.preset(p).weightTensorPlan().map(\.name)
            XCTAssertEqual(Set(names).count, names.count, "duplicate tensor name in preset \(p.rawValue)")
        }
    }

    func testPlanFlagsDivergentModules() {
        // The materially-different modules carry their flag tokens so a torch
        // consumer can't mistake them for stock components.
        let names = NetworkArchitecture.current.weightTensorPlan().map(\.name)
        XCTAssertTrue(names.contains("blocks.0.se_scalebias.fc2.weight"))
        XCTAssertTrue(names.contains("blocks.0.rezero_alpha"))
        XCTAssertTrue(names.contains("value.wdl_fc2.weight"))
        XCTAssertTrue(names.contains("blocks.0.conv1.weight"))
    }

    // MARK: validation

    func testPresetsValidate() throws {
        for p in NetworkArchitecture.Preset.allCases {
            XCTAssertNoThrow(try NetworkArchitecture.preset(p).validate(), "preset \(p.rawValue)")
        }
    }

    func testValidationRejectsEvenKernel() {
        var a = NetworkArchitecture.current
        a.towerConvKernelSize = 4
        XCTAssertThrowsError(try a.validate()) { error in
            XCTAssertEqual(error as? NetworkArchitectureError, .kernelMustBeOdd(4))
        }
    }

    func testValidationRejectsChannelsNotDivisibleByReduction() {
        var a = NetworkArchitecture.current
        a.channels = 130 // not divisible by 4
        XCTAssertThrowsError(try a.validate())
    }

    func testValidationRejectsFixedFieldChange() {
        var a = NetworkArchitecture.current
        a.inputPlanes = 31
        XCTAssertThrowsError(try a.validate()) { error in
            XCTAssertEqual(error as? NetworkArchitectureError,
                           .fixedFieldChanged(field: "inputPlanes", expected: 30, got: 31))
        }
    }

    func testNoneSEAllowsAnyChannels() throws {
        var a = NetworkArchitecture.current
        a.se = .none
        a.channels = 130
        XCTAssertNoThrow(try a.validate())
    }

    func testValidationRejectsZeroReductionEvenWithoutSE() {
        // The residual block computes channels / seReductionRatio regardless of
        // SE style, so a zero ratio must be rejected to avoid a build-time
        // divide-by-zero — even when se == .none.
        var a = NetworkArchitecture.current
        a.se = .none
        a.seReductionRatio = 0
        XCTAssertThrowsError(try a.validate()) { error in
            XCTAssertEqual(error as? NetworkArchitectureError,
                           .nonPositive(field: "seReductionRatio", value: 0))
        }
    }

    // MARK: Codable round-trip (this is what rides in safetensors __metadata__)

    func testCodableRoundTrip() throws {
        for p in NetworkArchitecture.Preset.allCases {
            let original = NetworkArchitecture.preset(p)
            let data = try JSONEncoder().encode(original)
            let decoded = try JSONDecoder().decode(NetworkArchitecture.self, from: data)
            XCTAssertEqual(original, decoded, "preset \(p.rawValue)")
        }
    }

    // MARK: parameterCount responds to SE style

    func testSEStyleAffectsParameterCount() {
        let base = NetworkArchitecture.current
        var attenuate = base; attenuate.se = .attenuateOnly
        var noSE = base; noSE.se = .none
        // scale-and-bias has the widest FC2 (2C), so most params; none has fewest.
        XCTAssertGreaterThan(base.parameterCount, attenuate.parameterCount)
        XCTAssertGreaterThan(attenuate.parameterCount, noSE.parameterCount)
    }

    // MARK: guard against static-constant / preset drift

    /// ChessNetwork keeps its static arch constants as the *default* arch for
    /// external callers; this asserts they stay in lockstep with the
    /// NetworkArchitecture.current preset (the single source of truth).
    func testStaticConstantsMatchCurrentPreset() {
        let cur = NetworkArchitecture.current
        XCTAssertEqual(ChessNetwork.channels, cur.channels)
        XCTAssertEqual(ChessNetwork.numBlocks, cur.numBlocks)
        XCTAssertEqual(ChessNetwork.towerConvKernelSize, cur.towerConvKernelSize)
        XCTAssertEqual(ChessNetwork.inputPlanes, cur.inputPlanes)
        XCTAssertEqual(ChessNetwork.boardSize, cur.boardSize)
        XCTAssertEqual(ChessNetwork.policyChannels, cur.policyChannels)
        XCTAssertEqual(ChessNetwork.policySize, cur.policySize)
        XCTAssertEqual(ChessNetwork.valueHeadClasses, cur.valueHeadClasses)
        XCTAssertEqual(ChessNetwork.seReductionRatio, cur.seReductionRatio)
        XCTAssertEqual(ChessNetwork.valueHeadConvChannels, cur.valueHeadConvChannels)
        XCTAssertEqual(ChessNetwork.valueHeadHiddenUnits, cur.valueHeadHiddenUnits)
        XCTAssertEqual(ChessNetwork.architectureVersion, cur.architectureVersion)
        XCTAssertEqual(ChessNetwork.parameterCount, cur.parameterCount)
    }
}
