//
//  NetworkArchitectureTests.swift
//  DrewsChessMachineTests
//
//  NetworkArchitecture is a pure value type that reproduces — independently of
//  the live graph builder — the derived quantities for every preset: parameter
//  count and the ordered weight-tensor plan. These tests pin those to the
//  documented, known-good values (verified against the real saved files) so the
//  struct can't silently drift from what ChessNetwork actually builds. Identity
//  is the config value itself (no arch hash, per plan §6).
//

import XCTest
@testable import DrewsChessMachine

final class NetworkArchitectureTests: XCTestCase {

    // MARK: parameterCount reproduces known values (verified vs real files)

    func testParameterCountKnownValues() {
        XCTAssertEqual(NetworkArchitecture.preset(.v3_8block_3x3).parameterCount, 2_483_667)
        XCTAssertEqual(NetworkArchitecture.preset(.v3_16block_3x3).parameterCount, 4_934_867)
        XCTAssertEqual(NetworkArchitecture.preset(.v4_12block_3x3).parameterCount, 3_898_139)
        XCTAssertEqual(NetworkArchitecture.preset(.v4_5block_7x7).parameterCount, 8_445_748)
        XCTAssertEqual(NetworkArchitecture.preset(.v4_8block_3x3).parameterCount, 2_664_087)
    }

    /// parameterCount must equal the summed element counts of the weight plan —
    /// cross-validates every shape in the plan against the param formula.
    func testParameterCountEqualsPlanElementSum() {
        for p in NetworkArchitecture.Preset.allCases {
            let arch = NetworkArchitecture.preset(p)
            let planSum = arch.weightTensorPlan().reduce(0) { $0 + $1.elementCount }
            XCTAssertEqual(planSum, arch.parameterCount, "preset \(p.rawValue)")
        }
    }

    // MARK: weightTensorPlan tensor counts (= champion tensor counts on disk)

    func testPlanTensorCountKnownValues() {
        XCTAssertEqual(NetworkArchitecture.preset(.v3_8block_3x3).weightTensorPlan().count, 128)
        XCTAssertEqual(NetworkArchitecture.preset(.v3_16block_3x3).weightTensorPlan().count, 245)
        XCTAssertEqual(NetworkArchitecture.preset(.v4_12block_3x3).weightTensorPlan().count, 205)
        XCTAssertEqual(NetworkArchitecture.preset(.v4_5block_7x7).weightTensorPlan().count, 100)
        XCTAssertEqual(NetworkArchitecture.preset(.v4_8block_3x3).weightTensorPlan().count, 145)
    }

    func testPlanOrderTrainablesThenRunningStats() throws {
        // exportWeights / loadWeights order = all trainables, then all BN running
        // stats; no running-stat tensor may precede a non-running tensor.
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

    func testV4PlanFlagsDivergentModules() {
        // v4: scale-and-bias SE, ReZero, WDL value head, tower-end BN.
        let names = NetworkArchitecture.preset(.v4_5block_7x7).weightTensorPlan().map(\.name)
        XCTAssertTrue(names.contains("blocks.0.se_scalebias.fc2.weight"))
        XCTAssertTrue(names.contains("blocks.0.rezero_alpha"))
        XCTAssertTrue(names.contains("value.wdl_fc2.weight"))
        XCTAssertTrue(names.contains("tower_final_bn.weight"))
        XCTAssertTrue(names.contains("policy.pre_conv.weight"))   // intermediate_conv
    }

    func testV3PlanReflectsPostActivation() {
        // v3: post-activation (no tower-end BN), no ReZero, attenuate-only SE.
        let names8 = NetworkArchitecture.preset(.v3_8block_3x3).weightTensorPlan().map(\.name)
        XCTAssertTrue(names8.contains("blocks.0.se_attenuate.fc2.weight"))
        XCTAssertFalse(names8.contains(where: { $0.hasSuffix("rezero_alpha") }))
        XCTAssertFalse(names8.contains("tower_final_bn.weight"))
        XCTAssertTrue(names8.contains("value.wdl_fc2.weight"))
        // 8-block-v3 uses the simple policy head (single conv, no pre-conv).
        XCTAssertTrue(names8.contains("policy.conv.weight"))
        XCTAssertFalse(names8.contains("policy.pre_conv.weight"))
        // 16-block-v3 uses the intermediate-conv policy head.
        let names16 = NetworkArchitecture.preset(.v3_16block_3x3).weightTensorPlan().map(\.name)
        XCTAssertTrue(names16.contains("policy.pre_conv.weight"))
    }

    // MARK: validation

    func testPresetsValidate() throws {
        for p in NetworkArchitecture.Preset.allCases {
            XCTAssertNoThrow(try NetworkArchitecture.preset(p).validate(), "preset \(p.rawValue)")
        }
    }

    func testValidationRejectsEvenKernel() {
        var a = NetworkArchitecture.current
        a.blockConv1KernelSize = 4
        XCTAssertThrowsError(try a.validate()) { error in
            XCTAssertEqual(error as? NetworkArchitectureError,
                           .kernelMustBeOdd(field: "blockConv1KernelSize", value: 4))
        }
    }

    func testValidationRejectsChannelsNotDivisibleByReduction() {
        var a = NetworkArchitecture.current
        a.channels = 130 // not divisible by 4 (SE on)
        XCTAssertThrowsError(try a.validate())
    }

    func testNoneSEAllowsNonDivisibleChannels() throws {
        var a = NetworkArchitecture.current
        a.blockSeStyle = .none
        a.channels = 130
        XCTAssertNoThrow(try a.validate())
    }

    func testValidationRejectsZeroReductionEvenWithoutSE() {
        // The residual block computes channels / ratio regardless of SE style,
        // so a zero ratio must be rejected even when se == .none.
        var a = NetworkArchitecture.current
        a.blockSeStyle = .none
        a.blockSeReductionRatio = 0
        XCTAssertThrowsError(try a.validate()) { error in
            XCTAssertEqual(error as? NetworkArchitectureError,
                           .nonPositive(field: "blockSeReductionRatio", value: 0))
        }
    }

    func testValidationRejectsValueConvChannelsExceedingChannels() {
        var a = NetworkArchitecture.current
        a.valueHeadConvChannels = a.channels + 1
        XCTAssertThrowsError(try a.validate())
    }

    // MARK: derived scalars

    func testDerivedScalars() {
        let wdl = NetworkArchitecture.preset(.v4_5block_7x7)
        XCTAssertEqual(wdl.valueHeadClasses, 3)
        XCTAssertEqual(wdl.inputPlanes, 30)
        XCTAssertTrue(wdl.hasTowerEndBN)        // pre-activation
        XCTAssertFalse(wdl.hasStemActivation)
        let v3 = NetworkArchitecture.preset(.v3_8block_3x3)
        XCTAssertFalse(v3.hasTowerEndBN)        // post-activation
        XCTAssertTrue(v3.hasStemActivation)
        XCTAssertEqual(v3.architectureVersionLabel, 3)
        XCTAssertEqual(wdl.architectureVersionLabel, 4)
    }

    // MARK: Codable (rides in safetensors __metadata__ as canonical JSON)

    func testCodableRoundTrip() throws {
        for p in NetworkArchitecture.Preset.allCases {
            let original = NetworkArchitecture.preset(p)
            let data = try JSONEncoder().encode(original)
            let decoded = try JSONDecoder().decode(NetworkArchitecture.self, from: data)
            XCTAssertEqual(original, decoded, "preset \(p.rawValue)")
        }
    }

    func testSnakeCaseKeys() throws {
        let enc = JSONEncoder()
        enc.outputFormatting = [.sortedKeys]
        let json = String(decoding: try enc.encode(NetworkArchitecture.current), as: UTF8.self)
        XCTAssertTrue(json.contains("\"input_encoding\""))
        XCTAssertTrue(json.contains("\"num_blocks\""))
        XCTAssertTrue(json.contains("\"block_conv1_kernel_size\""))
        XCTAssertTrue(json.contains("\"compute_data_type\""))
        XCTAssertFalse(json.contains("\"numBlocks\""))
    }

    /// Canonical (sortedKeys) encoding is deterministic and independent of struct
    /// field order — the property that lets arch identity be stable (plan §6).
    func testCanonicalEncodingDeterministic() throws {
        let enc = JSONEncoder()
        enc.outputFormatting = [.sortedKeys]
        let a = try enc.encode(NetworkArchitecture.current)
        let b = try enc.encode(NetworkArchitecture.current)
        XCTAssertEqual(a, b)
    }

    // MARK: parameterCount responds to SE style

    func testSEStyleAffectsParameterCount() {
        let base = NetworkArchitecture.current          // scale_and_bias
        var attenuate = base; attenuate.blockSeStyle = .attenuateOnly
        var noSE = base; noSE.blockSeStyle = .none
        // scale-and-bias has the widest FC2 (2C) → most params; none → fewest.
        XCTAssertGreaterThan(base.parameterCount, attenuate.parameterCount)
        XCTAssertGreaterThan(attenuate.parameterCount, noSE.parameterCount)
    }

    // MARK: guard against ChessNetwork static-constant / preset drift

    /// ChessNetwork keeps its static arch constants as the *default* arch; this
    /// asserts they stay in lockstep with `NetworkArchitecture.current` (the
    /// single source of truth). Current is v4_5block_7x7 (all kernels 7).
    func testStaticConstantsMatchCurrentPreset() {
        let cur = NetworkArchitecture.current
        XCTAssertEqual(ChessNetwork.channels, cur.channels)
        XCTAssertEqual(ChessNetwork.numBlocks, cur.numBlocks)
        XCTAssertEqual(ChessNetwork.towerConvKernelSize, cur.blockConv1KernelSize)
        XCTAssertEqual(ChessNetwork.towerConvKernelSize, cur.blockConv2KernelSize)
        XCTAssertEqual(ChessNetwork.towerConvKernelSize, cur.stemConvKernelSize)
        XCTAssertEqual(ChessNetwork.inputPlanes, cur.inputPlanes)
        XCTAssertEqual(ChessNetwork.boardSize, cur.boardSize)
        XCTAssertEqual(ChessNetwork.policyChannels, cur.policyChannels)
        XCTAssertEqual(ChessNetwork.policySize, cur.policySize)
        XCTAssertEqual(ChessNetwork.valueHeadClasses, cur.valueHeadClasses)
        XCTAssertEqual(ChessNetwork.seReductionRatio, cur.blockSeReductionRatio)
        XCTAssertEqual(ChessNetwork.valueHeadConvChannels, cur.valueHeadConvChannels)
        XCTAssertEqual(ChessNetwork.valueHeadHiddenUnits, cur.valueHeadHiddenUnits)
        XCTAssertEqual(ChessNetwork.architectureVersion, cur.architectureVersionLabel)
        XCTAssertEqual(ChessNetwork.parameterCount, cur.parameterCount)
    }
}
