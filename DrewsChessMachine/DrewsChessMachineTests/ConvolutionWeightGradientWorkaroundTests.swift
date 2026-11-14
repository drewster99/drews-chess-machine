import XCTest
@testable import DrewsChessMachine

/// The rule that decides which convolutions a training graph splits to avoid
/// MPSGraph's wrong fp32 weight gradient (`ConvolutionWeightGradientWorkaround`).
/// Pure: no GPU.
final class ConvolutionWeightGradientWorkaroundTests: XCTestCase {

    /// One pre-activation block of 9×9 / 128 channels on a 5×5 stem over the
    /// 24-plane input, 1×1 heads: the shape of the convolutions R-elitecyc
    /// trains, in the compute dtype given.
    private func architecture(_ dtype: ComputeDataType, tail: PolicyTailPrecisionSetting? = nil,
                              policyPreConvChannels: Int = 32) throws -> NetworkArchitecture {
        var arch = NetworkArchitecture(
            inputEncoding: .basic24, channels: 128, numBlocks: 1, stemConvKernelSize: 5,
            activationFunction: .relu, blockActivationStyle: .pre,
            blockSkipMerge: .cleanAdd, blockUseRezero: false, rezeroAlphaInit: 0.5,
            blockConv1KernelSize: 9, blockConv2KernelSize: 9,
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
    /// element count can pass 2³¹ at the batch ceiling, each into two groups.
    /// Fails if: a qualifying convolution is left out (its fp32 gradient would be
    /// wrong), a 1×1 head convolution is split needlessly, or the group count
    /// changes.
    func testFloat32ArchitectureSplitsEveryConvolutionThatCanPassTheLimit() throws {
        let plan = try ConvolutionWeightGradientWorkaround.plan(for: architecture(.float32))
        XCTAssertEqual(plan.splits, [
            .init(weightName: "stem.conv.weight", inputChannels: 24, kernelSize: 5, groups: 2),
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
        }
    }

    /// Protects: under the `fp32_from_pre_bn` tail the final policy
    /// convolution runs in fp32 inside a bf16 network, so it is covered when
    /// its shape qualifies; under `mixed_final_projection` it runs in bf16 and
    /// is not.
    /// Fails if: the rule looks only at the model's compute dtype.
    func testFloat32PolicyTailConvolutionInAReducedPrecisionNetworkIsCovered() throws {
        let wide = 520   // 1×1 over 520 channels: 520 > 512
        let fp32Tail = try ConvolutionWeightGradientWorkaround.plan(
            for: architecture(.bFloat16, tail: .float32FromPreBatchNorm, policyPreConvChannels: wide))
        XCTAssertEqual(fp32Tail.splits, [
            .init(weightName: "policy.conv.weight", inputChannels: wide, kernelSize: 1, groups: 2),
        ])
        let mixedTail = try ConvolutionWeightGradientWorkaround.plan(
            for: architecture(.bFloat16, tail: .mixedFinalProjection, policyPreConvChannels: wide))
        XCTAssertTrue(mixedTail.isEmpty, "\(mixedTail.splits)")
    }

    /// Protects: the size boundary. At the 65,536 batch ceiling an 8×8
    /// convolution passes 2³¹ − 1 exactly when C_in·k² > 512.
    /// Fails if: the limit, the ceiling or the board size changes, or the
    /// comparison becomes ≥.
    func testAffectedExactlyWhenInputChannelsTimesKernelAreaExceed512() {
        XCTAssertEqual(ConvolutionWeightGradientWorkaround.batchCeiling, 65_536)
        let cases: [(inputChannels: Int, kernelSize: Int, affected: Bool)] = [
            (512, 1, false), (513, 1, true),
            (56, 3, false), (57, 3, true),       // 504 / 513
            (20, 5, false), (21, 5, true),       // 500 / 525
            (10, 7, false), (11, 7, true),       // 490 / 539
            (6, 9, false), (7, 9, true),         // 486 / 567
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

    /// Protects: the notice names every split convolution and its event, and
    /// there is none without splits.
    func testNoticeNamesEverySplitAndTheEvent() throws {
        let plan = try ConvolutionWeightGradientWorkaround.plan(for: architecture(.float32))
        let notice = try XCTUnwrap(plan.notice(event: "--new-model test"))
        XCTAssertTrue(notice.hasPrefix("[ARCH] fp32 conv weight-gradient workaround (--new-model test): 3 convolution(s)"),
                      notice)
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
}
