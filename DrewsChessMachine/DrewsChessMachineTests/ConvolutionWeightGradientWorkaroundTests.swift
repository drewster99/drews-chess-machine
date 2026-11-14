import XCTest
@testable import DrewsChessMachine

/// The rule that decides which convolutions a training graph splits to avoid
/// MPSGraph's wrong fp32 weight gradient (`ConvolutionWeightGradientWorkaround`).
/// Pure: no GPU.
final class ConvolutionWeightGradientWorkaroundTests: XCTestCase {

    /// One pre-activation block of 128 channels (9×9 by default: the shape of
    /// the convolutions R-elitecyc trains) on a 5×5 stem over the 24-plane
    /// input, 1×1 heads, in the compute dtype given.
    private func architecture(_ dtype: ComputeDataType, tail: PolicyTailPrecisionSetting? = nil,
                              policyPreConvChannels: Int = 32, blockKernelSize: Int = 9) throws -> NetworkArchitecture {
        var arch = NetworkArchitecture(
            inputEncoding: .basic24, channels: 128, numBlocks: 1, stemConvKernelSize: 5,
            activationFunction: .relu, blockActivationStyle: .pre,
            blockSkipMerge: .cleanAdd, blockUseRezero: false, rezeroAlphaInit: 0.5,
            blockConv1KernelSize: blockKernelSize, blockConv2KernelSize: blockKernelSize,
            blockSeStyle: .none, blockSeReductionRatio: 4,
            policyHeadStyle: .intermediateConv, policyPreConvChannels: policyPreConvChannels,
            valueHeadStyle: .wdlSoftmax, valueHeadConvChannels: 4, valueHeadHiddenUnits: 16,
            computeDataType: .float32)
        arch.clearActivationSitesTheTopologyLacks()
        try arch.setActivationAtEveryExistingSite(.relu)
        if dtype != .float32 {
            arch = try arch.withComputeDataType(dtype, tail: tail)
        }
        try arch.validate()
        return arch
    }

    /// Protects: an fp32 architecture splits exactly the convolutions whose
    /// element count passes 2³¹ at the planned batch (4096), each into two
    /// groups; the 5×5 / 24-channel stem passes only from batch 55,925 and is
    /// left alone.
    /// Fails if: a qualifying convolution is left out (its fp32 gradient would be
    /// wrong), a convolution is split needlessly, or the group count changes.
    func testFloat32ArchitectureSplitsEveryConvolutionThatPassesTheLimitAtThePlannedBatch() throws {
        let plan = try ConvolutionWeightGradientWorkaround.plan(for: architecture(.float32))
        XCTAssertEqual(plan.splits, [
            .init(weightName: "blocks.0.conv1.weight", inputChannels: 128, kernelSize: 9, groups: 2),
            .init(weightName: "blocks.0.conv2.weight", inputChannels: 128, kernelSize: 9, groups: 2),
        ])
    }

    /// Protects: bf16 and fp16 graphs are never split (their kernel is
    /// correct), so their training graphs stay exactly as before.
    /// Fails if: the rule stops gating on the convolution's dtype.
    func testReducedPrecisionArchitecturesSplitNothing() throws {
        for (dtype, tail) in [(ComputeDataType.bFloat16, PolicyTailPrecisionSetting.mixedFinalProjection),
                              (.bFloat16, .float32FromPreBatchNorm),
                              (.float16, .mixedFinalProjection),
                              (.float16, .float32FromPreBatchNorm)] {
            let plan = try ConvolutionWeightGradientWorkaround.plan(for: architecture(dtype, tail: tail))
            XCTAssertTrue(plan.isEmpty, "\(dtype) / \(tail): \(plan.splits)")
            XCTAssertEqual(plan.batchLimit, .everyAcceptedBatch, "\(dtype) / \(tail)")
        }
    }

    /// Protects: under the `fp32_from_pre_bn` tail the final policy
    /// convolution runs in fp32 inside a bf16 network, so it is covered when
    /// its shape qualifies; under `mixed_final_projection` it runs in bf16 and
    /// is not.
    /// Fails if: the rule looks only at the model's compute dtype.
    func testFloat32PolicyTailConvolutionInAReducedPrecisionNetworkIsCovered() throws {
        let wide = 8194   // 1×1 over 8194 channels: 4096 × 64 × 8194 > 2³¹ − 1
        let fp32Tail = try ConvolutionWeightGradientWorkaround.plan(
            for: architecture(.bFloat16, tail: .float32FromPreBatchNorm, policyPreConvChannels: wide))
        XCTAssertEqual(fp32Tail.splits, [
            .init(weightName: "policy.conv.weight", inputChannels: wide, kernelSize: 1, groups: 2),
        ])
        let mixedTail = try ConvolutionWeightGradientWorkaround.plan(
            for: architecture(.bFloat16, tail: .mixedFinalProjection, policyPreConvChannels: wide))
        XCTAssertTrue(mixedTail.isEmpty, "\(mixedTail.splits)")
    }

    /// Protects: the size boundary. At the planned batch (4096) an 8×8
    /// convolution passes 2³¹ − 1 exactly when C_in·k² ≥ 8192.
    /// Fails if: the limit, the planned batch or the board size changes, or the
    /// comparison becomes ≥.
    func testAffectedExactlyWhenInputChannelsTimesKernelAreaReach8192() {
        XCTAssertEqual(ConvolutionWeightGradientWorkaround.plannedTrainingBatchSize, 4096)
        let cases: [(inputChannels: Int, kernelSize: Int, affected: Bool)] = [
            (8191, 1, false), (8192, 1, true),
            (910, 3, false), (911, 3, true),     // 8190 / 8199
            (327, 5, false), (328, 5, true),     // 8175 / 8200
            (167, 7, false), (168, 7, true),     // 8183 / 8232
            (101, 9, false), (102, 9, true),     // 8181 / 8262
        ]
        for c in cases {
            XCTAssertEqual(ConvolutionWeightGradientWorkaround.isAffected(inputChannels: c.inputChannels, kernelSize: c.kernelSize),
                           c.affected, "C_in \(c.inputChannels), k \(c.kernelSize)")
        }
    }

    /// Protects: the R-elitecyc shape is over the limit at its own batch (the
    /// measured failure) and the batch where it starts (3237).
    /// Fails if: `elementCount` computes a different product.
    func testElementCountMatchesTheMeasuredBoundary() {
        XCTAssertEqual(ConvolutionWeightGradientWorkaround.elementCount(batch: 4096, inputChannels: 128, kernelSize: 9),
                       2_717_908_992)
        XCTAssertLessThanOrEqual(
            ConvolutionWeightGradientWorkaround.elementCount(batch: 3236, inputChannels: 128, kernelSize: 9),
            ConvolutionWeightGradientWorkaround.elementLimit)
        XCTAssertGreaterThan(
            ConvolutionWeightGradientWorkaround.elementCount(batch: 3237, inputChannels: 128, kernelSize: 9),
            ConvolutionWeightGradientWorkaround.elementLimit)
    }

    /// Protects: a product past `Int.max` saturates instead of trapping or
    /// wrapping to a small (unaffected-looking) value.
    func testElementCountSaturatesOnOverflow() {
        XCTAssertEqual(ConvolutionWeightGradientWorkaround.elementCount(batch: Int.max / 2, inputChannels: 128, kernelSize: 9),
                       Int.max)
    }

    /// Protects: the group count is the smallest divisor ≥ 2, and a single
    /// channel has none.
    func testGroupCountIsTheSmallestDivisorOfAtLeastTwo() {
        let cases: [(Int, Int?)] = [(1, nil), (0, nil), (2, 2), (3, 3), (9, 3), (24, 2), (25, 5), (49, 7), (97, 97), (128, 2)]
        for (channels, expected) in cases {
            XCTAssertEqual(ConvolutionWeightGradientWorkaround.groupCount(inputChannels: channels), expected, "\(channels)")
        }
    }

    /// Protects: the notice names every split convolution, its event and the
    /// batch limit, and there is none for a plan that neither splits nor limits.
    func testNoticeNamesEverySplitTheEventAndTheBatchLimit() throws {
        let plan = try ConvolutionWeightGradientWorkaround.plan(for: architecture(.float32))
        let notice = try XCTUnwrap(plan.notice(event: "--new-model test"))
        XCTAssertTrue(notice.hasPrefix("[ARCH] fp32 conv weight-gradient workaround (--new-model test): 2 convolution(s)"),
                      notice)
        XCTAssertTrue(notice.contains("Training refuses batches above 32768"), notice)
        for split in plan.splits {
            XCTAssertTrue(notice.contains(split.weightName), "\(split.weightName) missing from \(notice)")
        }
        XCTAssertNil(ConvolutionWeightGradientWorkaround.Plan.none.notice(event: "x"))
    }

    /// Protects: `groups(forWeight:)` reports 1 for a convolution the plan
    /// does not split.
    func testUnsplitConvolutionHasOneGroup() throws {
        let plan = try ConvolutionWeightGradientWorkaround.plan(for: architecture(.float32))
        XCTAssertEqual(plan.groups(forWeight: "blocks.0.conv1.weight"), 2)
        XCTAssertEqual(plan.groups(forWeight: "value.conv.weight"), 1)
    }

    /// Protects: the first batch at which a shape passes the limit, for the
    /// three tower shapes in use (9×9, 7×7, 3×3 at 128 channels).
    /// Fails if: `firstWrongBatch` is off by one or computes another product.
    func testFirstWrongBatchForTheTowerShapesInUse() {
        XCTAssertEqual(ConvolutionWeightGradientWorkaround.firstWrongBatch(inputChannels: 128, kernelSize: 9), 3237)
        XCTAssertEqual(ConvolutionWeightGradientWorkaround.firstWrongBatch(inputChannels: 128, kernelSize: 7), 5350)
        XCTAssertEqual(ConvolutionWeightGradientWorkaround.firstWrongBatch(inputChannels: 128, kernelSize: 3), 29_128)
        for (cin, k) in [(128, 9), (128, 7), (128, 3), (24, 5), (1, 1)] {
            let first = ConvolutionWeightGradientWorkaround.firstWrongBatch(inputChannels: cin, kernelSize: k)
            XCTAssertLessThanOrEqual(
                ConvolutionWeightGradientWorkaround.elementCount(batch: first - 1, inputChannels: cin, kernelSize: k),
                ConvolutionWeightGradientWorkaround.elementLimit, "\(cin) / \(k)")
            XCTAssertGreaterThan(
                ConvolutionWeightGradientWorkaround.elementCount(batch: first, inputChannels: cin, kernelSize: k),
                ConvolutionWeightGradientWorkaround.elementLimit, "\(cin) / \(k)")
        }
    }

    /// Protects: a plan with splits covers batches up to the batch the split
    /// was measured correct at, and refuses the next one.
    /// Fails if: the limit is dropped, moved, or the comparison becomes ≥.
    func testPlanWithSplitsRefusesBatchesAboveTheMeasuredSplitRange() throws {
        let plan = try ConvolutionWeightGradientWorkaround.plan(for: architecture(.float32))
        XCTAssertEqual(plan.batchLimit, .upTo(largestCorrectBatch: 32_768, source: .splitConvolutionsUnmeasured))
        XCTAssertNoThrow(try plan.requireCorrectWeightGradients(batchSize: 4096))
        XCTAssertNoThrow(try plan.requireCorrectWeightGradients(batchSize: 32_768))
        XCTAssertThrowsError(try plan.requireCorrectWeightGradients(batchSize: 32_769)) { error in
            XCTAssertEqual(error as? ConvolutionWeightGradientWorkaround.WorkaroundError,
                           .batchAboveCorrectLimit(batchSize: 32_769, largestCorrectBatch: 32_768,
                                                   source: .splitConvolutionsUnmeasured))
        }
    }

    /// Protects: an fp32 3×3 / 128 tower is not split at the planned batch
    /// (it passes the limit only from 29,128), and a step at 29,128 — where
    /// its unsplit gradient would be wrong — is refused, naming the convolution.
    /// Fails if: the guard ignores unsplit convolutions, or uses the wrong one.
    func testUnsplitPlanRefusesTheFirstBatchItsConvolutionsGetWrong() throws {
        let plan = try ConvolutionWeightGradientWorkaround.plan(for: architecture(.float32, blockKernelSize: 3))
        XCTAssertTrue(plan.isEmpty, "\(plan.splits)")
        let source = ConvolutionWeightGradientWorkaround.BatchLimitSource.unsplitConvolution(
            weightName: "blocks.0.conv1.weight", inputChannels: 128, kernelSize: 3)
        XCTAssertEqual(plan.batchLimit, .upTo(largestCorrectBatch: 29_127, source: source))
        XCTAssertNoThrow(try plan.requireCorrectWeightGradients(batchSize: 29_127))
        XCTAssertThrowsError(try plan.requireCorrectWeightGradients(batchSize: 29_128)) { error in
            XCTAssertEqual(error as? ConvolutionWeightGradientWorkaround.WorkaroundError,
                           .batchAboveCorrectLimit(batchSize: 29_128, largestCorrectBatch: 29_127, source: source))
        }
        let notice = try XCTUnwrap(plan.notice(event: "x"))
        XCTAssertTrue(notice.contains("no convolution needs splitting at the planned batch 4096"), notice)
        XCTAssertTrue(notice.contains("Training refuses batches above 29127"), notice)
    }

    /// Protects: a network with no fp32 convolution never refuses a batch.
    /// Fails if: the guard fires on bf16 networks.
    func testReducedPrecisionPlanAcceptsEveryBatch() throws {
        let plan = try ConvolutionWeightGradientWorkaround.plan(for: architecture(.bFloat16, tail: .mixedFinalProjection))
        XCTAssertNoThrow(try plan.requireCorrectWeightGradients(batchSize: ConvolutionWeightGradientWorkaround.batchCeiling))
        XCTAssertNil(plan.notice(event: "x"))
    }
}
