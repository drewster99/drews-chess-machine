import XCTest
@testable import DrewsChessMachine

/// The numerics audit's static checks on synthetic tensors whose answers are
/// worked out by hand, the format limits they rest on, the position set's
/// fixed parts, and that analysis taps exist only when asked for.
final class NumericsAuditStaticTests: XCTestCase {

    // MARK: - Formats

    func testFormatLimitsAndSteps() {
        XCTAssertEqual(NumericFormat.fp16.maxFinite, 65504)
        XCTAssertEqual(NumericFormat.fp16.minNormal, 6.103515625e-5)
        XCTAssertEqual(NumericFormat.bf16.fractionBits, 7)
        XCTAssertEqual(NumericFormat.fp16.fractionBits, 10)
        // The steps quoted by the head-offset research.
        XCTAssertEqual(NumericFormat.bf16.step(atMagnitude: 510), 2)
        XCTAssertEqual(NumericFormat.bf16.step(atMagnitude: 600), 4)
        XCTAssertEqual(NumericFormat.bf16.step(atMagnitude: 42), 0.25)
        XCTAssertEqual(NumericFormat.fp16.step(atMagnitude: 510), 0.25)
        XCTAssertEqual(NumericFormat.bf16.step(atMagnitude: 30), 0.125)
    }

    func testRoundingMatchesTheFormats() {
        // Halfway between 1 and the next bf16 value: ties to even, so 1.
        XCTAssertEqual(NumericFormat.bf16.round(1 + 1.0 / 256), 1)
        XCTAssertEqual(NumericFormat.bf16.round(510.55), 510)
        XCTAssertEqual(NumericFormat.bf16.round(511.4), 512)
        XCTAssertEqual(NumericFormat.fp16.round(65519), 65504)
        XCTAssertTrue(NumericFormat.fp16.round(65520).isInfinite)
        XCTAssertEqual(NumericFormat.fp32.round(0.1), 0.1)
    }

    // MARK: - Tensor fitness

    func testFP16FlushesTinyValuesAndOverflowsLargeOnes() throws {
        let tiny = NumericsAudit.tensorReport(name: "tiny", values: [1e-8, 1e-8, 1, -1])
        let fp16Tiny = try XCTUnwrap(tiny.fitness.first { $0.format == .fp16 })
        XCTAssertEqual(fp16Tiny.underflowFraction, 0.5)
        XCTAssertEqual(fp16Tiny.flushToZeroFraction, 0.5)
        XCTAssertEqual(fp16Tiny.verdict, .bad)
        let bf16Tiny = try XCTUnwrap(tiny.fitness.first { $0.format == .bf16 })
        XCTAssertEqual(bf16Tiny.flushToZeroFraction, 0)

        let large = NumericsAudit.tensorReport(name: "large", values: [1e5, -3, 2])
        let fp16Large = try XCTUnwrap(large.fitness.first { $0.format == .fp16 })
        XCTAssertTrue(fp16Large.overflows)
        XCTAssertEqual(fp16Large.verdict, .bad)
        let bf16Large = try XCTUnwrap(large.fitness.first { $0.format == .bf16 })
        XCTAssertFalse(bf16Large.overflows)
    }

    func testRoundingAgainstSpreadFlagsAnOffsetTensor() throws {
        // A tensor riding on a large shared offset: its spread is tiny next
        // to the bf16 step at its size, so rounding swamps it.
        let offset = (0..<64).map { Float(512) + Float($0 % 4) * 0.25 }
        let report = NumericsAudit.tensorReport(name: "offset", values: offset)
        let bf16 = try XCTUnwrap(report.fitness.first { $0.format == .bf16 })
        XCTAssertEqual(bf16.verdict, .bad)
        let fp32 = try XCTUnwrap(report.fitness.first { $0.format == .fp32 })
        XCTAssertEqual(fp32.roundingRMSToSpread, 0)
        XCTAssertEqual(fp32.verdict, .fine)
    }

    // MARK: - Head shared offset

    /// Rows of `base` sum to zero across the three classes, so the mean row
    /// is exactly the added offset.
    private let base: [[Float]] = [[1, -1, 0], [0, 1, -1], [-1, 0, 1], [1, 0, -1]]

    func testValueHeadSharedOffsetIsMeasuredExactly() throws {
        let weights = base.flatMap { row in row.map { $0 + 10 } }
        let report = try XCTUnwrap(NumericsAudit.sharedOffset(
            weightName: "value_wdl_fc2_weights", weights: weights, layout: .inputMajor(outputs: 3),
            biasName: "value_wdl_fc2_bias", bias: [0, Float(log(6.0)), 0], biasInitMean: log(6.0) / 3
        ))
        XCTAssertEqual(report.meanRowNorm, 20, accuracy: 1e-9)
        XCTAssertEqual(report.residualNormMedian, sqrt(3), accuracy: 1e-9)
        XCTAssertEqual(report.initExpectedRatio, 1 / sqrt(2), accuracy: 1e-12)
        XCTAssertEqual(report.ratioToInitExpectation, (20 / sqrt(3)) * sqrt(2), accuracy: 1e-9)
        XCTAssertEqual(report.biasMean, log(6.0) / 3, accuracy: 1e-6)
        XCTAssertEqual(report.verdict, .bad)
    }

    func testCenteredValueHeadHasNoSharedOffset() throws {
        let report = try XCTUnwrap(NumericsAudit.sharedOffset(
            weightName: "value_wdl_fc2_weights", weights: base.flatMap { $0 }, layout: .inputMajor(outputs: 3),
            biasName: "value_wdl_fc2_bias", bias: [0, 0, 0], biasInitMean: 0
        ))
        XCTAssertEqual(report.meanRowNorm, 0, accuracy: 1e-12)
        XCTAssertEqual(report.verdict, .fine)
    }

    func testOutputMajorLayoutAveragesAcrossOutputs() throws {
        // Three outputs over two inputs, stored [outputs, inputs].
        let report = try XCTUnwrap(NumericsAudit.sharedOffset(
            weightName: "policy_conv_weights", weights: [1, 2, 3, 4, 5, 6], layout: .outputMajor(outputs: 3),
            biasName: "policy_conv_bias", bias: [0, 0, 0], biasInitMean: 0
        ))
        XCTAssertEqual(report.meanRowNorm, 5, accuracy: 1e-12)
        XCTAssertEqual(report.residualNormMedian, sqrt(8), accuracy: 1e-12)
    }

    // MARK: - Batch-norm statistics, ReZero, masters

    func testBatchNormMeanToSpread() {
        let report = NumericsAudit.batchNormStats(layerName: "b", mean: [0, 32, 100], variance: [1, 1, 1e-6])
        XCTAssertEqual(report.maxMeanToStd, 100_000, accuracy: 1e-3)
        XCTAssertEqual(report.maxMeanToStdChannel, 2)
        XCTAssertEqual(report.medianMeanToStd, 32, accuracy: 1e-9)
        XCTAssertEqual(report.varianceBelowNormalCount["fp16"], 1)
        XCTAssertEqual(report.varianceBelowNormalCount["bf16"], 0)
        XCTAssertEqual(report.verdict, .bad)

        let calm = NumericsAudit.batchNormStats(layerName: "c", mean: [0.5, -1], variance: [1, 4])
        XCTAssertEqual(calm.verdict, .fine)
    }

    func testReZeroBoundSaturatesInBF16First() {
        let ceiling = 1 / sqrt(5.0)
        let report = NumericsAudit.reZeroReport(name: "block0_res_scale", alpha: 4 * ceiling, ceiling: ceiling)
        XCTAssertEqual(report.tanhValue, tanh(4.0), accuracy: 1e-12)
        XCTAssertEqual(report.saturatesInFormat["bf16"], true)
        XCTAssertEqual(report.saturatesInFormat["fp16"], false)
        XCTAssertEqual(report.saturatesInFormat["fp32"], false)
    }

    func testMasterDivergenceInBF16Steps() throws {
        let divergence = NumericsAudit.masterDivergence(names: ["w"], working: [[1]], masters: [[1.001]])
        let top = try XCTUnwrap(divergence.first)
        XCTAssertEqual(top.maxAbsDifference, Double(Float(1.001)) - 1, accuracy: 1e-12)
        XCTAssertEqual(top.maxDifferenceInBF16Steps, (Double(Float(1.001)) - 1) / (1.0 / 128), accuracy: 1e-9)
    }

    // MARK: - Whole static pass

    func testRunStaticFindsTheHotSpotsByName() throws {
        let names = ["value_wdl_fc2_weights", "value_wdl_fc2_bias", "blk_bn_running_mean", "blk_bn_running_var"]
        let weights: [[Float]] = [
            base.flatMap { row in row.map { $0 + 10 } },
            [0, Float(log(6.0)), 0],
            [100, 0],
            [1, 1],
        ]
        let result = try NumericsAudit.runStatic(names: names, weights: weights, arch: .current, masters: nil, mastersNote: "none")
        XCTAssertEqual(result.tensors.map(\.name), names)
        XCTAssertEqual(result.valueHeadOffset?.verdict, .bad)
        XCTAssertNil(result.policyHeadOffset)
        XCTAssertEqual(result.batchNormStats.map(\.layerName), ["blk_bn"])
        XCTAssertEqual(result.batchNormStats.first?.verdict, .bad)
        XCTAssertNil(result.masterDivergence)
        XCTAssertEqual(result.mastersNote, "none")

        let findings = NumericsAudit.collectFindings(staticResult: result, dynamicResult: nil)
        XCTAssertTrue(findings.contains { $0.area == "head shared offset" && $0.verdict == .bad })
        XCTAssertEqual(findings.first?.verdict, .bad, "findings are sorted worst first")
    }

    func testRunStaticRejectsMismatchedNames() {
        XCTAssertThrowsError(try NumericsAudit.runStatic(names: ["a"], weights: [], arch: .current, masters: nil, mastersNote: nil))
    }

    // MARK: - Position set

    func testPositionSetWithoutCorpusOrLichessIsJustTheStart() throws {
        let set = try NumericsAudit.buildPositionSet(encoding: NetworkArchitecture.current.inputEncoding, corpusShardURL: nil, lichessDirectory: nil)
        XCTAssertEqual(set.positions.count, 1)
        XCTAssertEqual(set.summary.total, 1)
        XCTAssertEqual(set.positions[0].legalPolicyIndices.count, 20)
        XCTAssertNil(set.positions[0].valueTarget)
        XCTAssertEqual(set.summary.corpusNote, "no corpus shard was given")
        XCTAssertEqual(set.summary.lichessNote, "no Lichess bot data directory was given")
    }

    // MARK: - Analysis taps

    func testAnalysisTapsExistOnlyWhenRequested() throws {
        let production = try ChessNetwork(arch: .current, bnMode: .inference)
        XCTAssertTrue(production.analysisTapReadbacks.isEmpty)

        let audit = try ChessNetwork(arch: .current, bnMode: .inference, analysisTaps: true)
        let names = Set(audit.analysisTapReadbacks.map(\.name))
        for expected in ["stem_bn_input", "stem_output", "block0_output", "tower_output", "policy_logits", "value_logits", "value_probs", "value_fc1_act"] {
            XCTAssertTrue(names.contains(expected), "missing tap \(expected)")
        }
        XCTAssertTrue(audit.analysisTapReadbacks.allSatisfy { $0.tensor.dataType == .float32 })
        // Taps add readbacks, never variables: the weight layout is unchanged.
        XCTAssertEqual(audit.trainableVariables.count, production.trainableVariables.count)
        XCTAssertEqual(audit.bnRunningStatsVariables.count, production.bnRunningStatsVariables.count)
    }
}
