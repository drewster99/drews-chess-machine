//
//  ConvolutionWeightGradientWorkaround.swift
//  DrewsChessMachine
//
//  Which convolutions a training graph splits over their input channels to avoid
//  a wrong MPSGraph fp32 weight gradient, and into how many groups.
//
//  The defect (measured on macOS 27.2 26B123, Apple M5 Max; standalone repro in
//  `experiments/20261008-zlra-selfplay-lr/analysis-fp32/repro/`): MPSGraph's fp32
//  convolution weight gradient is wrong once batch × 64 × C_in × k² passes
//  2³¹ − 1 for 128-channel inputs and the convolution's output has another
//  consumer. A training-mode BatchNorm reads its input three times (batch mean,
//  batch variance, normalize), so every BatchNorm-fed convolution qualifies. At
//  batch 4096, 9×9 and 128 channels the gradient's relative error is 0.75; at
//  batch 8192 MPS tries to allocate 8 bytes per element of that product (40 GiB)
//  and aborts. The forward output and the input gradient stay exact; bf16 and
//  fp16 take another kernel and are correct at every size measured.
//  R-elitecyc-fp32 trained 28,814 steps on it and lost 330–380 pElo.
//
//  Why an input-channel split: Y = Σ_g conv(X[:, g], W[:, g]) is the same
//  function, and each slice's weight gradient is that slice of the true gradient,
//  so nothing about training changes but MPSGraph's kernel choice. It is exact in
//  measurement (relative error 7e-6 inside the real block structure at batch
//  4096, exact at batches 4096–32768 with two halves). Channel counts are static
//  when the graph is built and the batch is not (the input placeholder's batch
//  axis is -1 and one trainer runs several batch sizes), which rules out a batch
//  split. Measurements also show the split works by changing the kernel MPSGraph
//  selects, not by shrinking the element count below 2³¹ (two halves are exact
//  at 1.09e10 elements per half), so two groups are used.
//
//  Which convolutions: every one that runs in fp32 in a training graph and whose
//  element count can pass the limit at the largest batch the app accepts
//  (`TrainingBatchSize`'s declared upper bound — the one source of the batch
//  ceiling; at 65,536 that is C_in·k² > 512). Measured shapes below the size limit
//  never failed, and 5×5 / 24 channels was exact even past it, so the rule is
//  deliberately wider than the defect: a split convolution costs some speed (about
//  20% in a 3-block microbenchmark), an unsplit wrong one costs the run.
//  Inference graphs are never split: they have no weight gradient, and Core ML
//  export records only inference graphs.
//

import Foundation

enum ConvolutionWeightGradientWorkaround {

    /// The largest element count MPSGraph's fp32 weight-gradient path handles
    /// correctly (a 32-bit signed index).
    static let elementLimit = Int(Int32.max)

    /// Squares per board — the H·W of every convolution in the network.
    static let boardSquares = NetworkArchitecture.boardSize * NetworkArchitecture.boardSize

    /// The largest training batch any path accepts.
    static let batchCeiling = TrainingBatchSize.declaredClosedRange.upperBound

    /// One convolution the training graph splits.
    struct SplitConvolution: Equatable, Sendable {
        /// The weight's plan (safetensors) name, e.g. `blocks.0.conv1.weight`.
        let weightName: String
        let inputChannels: Int
        let kernelSize: Int
        /// How many input-channel groups the convolution is built as.
        let groups: Int
    }

    /// The split decision for one training graph, by weight name.
    struct Plan: Equatable, Sendable {
        let splits: [SplitConvolution]

        static let none = Plan(splits: [])

        /// The number of input-channel groups for the convolution whose weight
        /// is `weightName`: 1 when it is not split.
        func groups(forWeight weightName: String) -> Int {
            splits.first { $0.weightName == weightName }?.groups ?? 1
        }

        var isEmpty: Bool { splits.isEmpty }

        /// The `[ARCH]` notice for a graph built (or a model to be trained) with
        /// these splits, tagged with what produced it (`training graph`,
        /// `--new-model <name>`, `--derive-model <file>`, `Build Network`); nil
        /// when there are none.
        func notice(event: String) -> String? {
            guard !splits.isEmpty else { return nil }
            let list = splits.map { "\($0.weightName) (\($0.inputChannels)ch \($0.kernelSize)x\($0.kernelSize) -> \($0.groups) groups)" }
                .joined(separator: ", ")
            return "[ARCH] fp32 conv weight-gradient workaround (\(event)): \(splits.count) convolution(s) are split over their "
                + "input channels in training graphs, because MPSGraph computes a wrong fp32 weight gradient once "
                + "batch x 64 x C_in x k^2 passes 2^31 (batch ceiling \(batchCeiling)): \(list). Same function and "
                + "gradients; inference graphs are unchanged."
        }
    }

    enum WorkaroundError: Error, LocalizedError, Equatable {
        /// An affected convolution whose input-channel count has no divisor ≥ 2.
        case cannotSplit(weightName: String, inputChannels: Int, kernelSize: Int)
        /// The graph builder met an affected fp32 convolution the plan does not
        /// split — the plan and the builder disagree about where fp32 runs.
        case unplannedAffectedConvolution(weightName: String, inputChannels: Int, kernelSize: Int)
        /// A convolution weight without a static four-axis shape.
        case weightShapeUnknown(weightName: String)
        /// A training step larger than the batch ceiling the plan was made for,
        /// on a network with an fp32 convolution.
        case batchAboveCeiling(batchSize: Int)

        var errorDescription: String? {
            switch self {
            case let .cannotSplit(name, cin, k):
                return "\(name) (\(cin) input channels, \(k)x\(k)) runs in fp32 and can pass MPSGraph's 2^31-element "
                    + "fp32 weight-gradient limit at batch \(batchCeiling), but \(cin) input channels cannot be split "
                    + "into groups; training this architecture in fp32 would give a wrong gradient"
            case let .unplannedAffectedConvolution(name, cin, k):
                return "internal error: \(name) (\(cin) input channels, \(k)x\(k)) is built in fp32 in a training graph "
                    + "and can pass the 2^31-element limit, but the weight-gradient workaround plan does not split it"
            case let .batchAboveCeiling(batchSize):
                return "training batch \(batchSize) is above \(batchCeiling), the largest batch the fp32 conv "
                    + "weight-gradient workaround covers; an fp32 convolution could get a wrong weight gradient"
            case let .weightShapeUnknown(name):
                return "internal error: convolution weight \(name) has no static [C_out, C_in, k, k] shape"
            }
        }
    }

    /// batch × 64 × C_in × k², saturating instead of overflowing.
    static func elementCount(batch: Int, inputChannels: Int, kernelSize: Int) -> Int {
        let factors = [batch, boardSquares, inputChannels, kernelSize, kernelSize]
        var product = 1
        for factor in factors {
            let (result, overflow) = product.multipliedReportingOverflow(by: factor)
            if overflow { return Int.max }
            product = result
        }
        return product
    }

    /// Whether an fp32 convolution with this shape needs the split.
    static func isAffected(inputChannels: Int, kernelSize: Int) -> Bool {
        elementCount(batch: batchCeiling, inputChannels: inputChannels, kernelSize: kernelSize) > elementLimit
    }

    /// The smallest divisor of `inputChannels` that is at least 2; nil when there
    /// is none (one input channel).
    static func groupCount(inputChannels: Int) -> Int? {
        guard inputChannels >= 2 else { return nil }
        var divisor = 2
        while divisor * divisor <= inputChannels {
            if inputChannels % divisor == 0 { return divisor }
            divisor += 1
        }
        return inputChannels
    }

    /// Whether the convolution whose weight is `weightName` runs in fp32 in a
    /// graph of `architecture`: every convolution on an fp32 model, and the final
    /// policy projection under the `fp32_from_pre_bn` tail (its input, weights and
    /// bias are widened to fp32; `ChessNetwork.policyHead`). The graph builder
    /// re-checks this against each convolution's actual input dtype.
    static func runsInFloat32(weightName: String, architecture: NetworkArchitecture) -> Bool {
        if architecture.computeDataType == .float32 { return true }
        return weightName == "policy.conv.weight" && architecture.policyTailPrecision == .float32FromPreBatchNorm
    }

    /// Whether any convolution of `architecture` runs in fp32 — the networks a
    /// training step above `batchCeiling` is refused on.
    static func anyConvolutionRunsInFloat32(_ architecture: NetworkArchitecture) -> Bool {
        architecture.weightTensorPlan().contains {
            $0.kind == .conv && runsInFloat32(weightName: $0.name, architecture: architecture)
        }
    }

    /// The splits a training graph of `architecture` is built with. Throws when
    /// an affected convolution cannot be split.
    static func plan(for architecture: NetworkArchitecture) throws -> Plan {
        var splits: [SplitConvolution] = []
        for spec in architecture.weightTensorPlan() where spec.kind == .conv {
            // Native conv shape: [C_out, C_in, k, k].
            let inputChannels = spec.shape[1]
            let kernelSize = spec.shape[2]
            guard runsInFloat32(weightName: spec.name, architecture: architecture),
                  isAffected(inputChannels: inputChannels, kernelSize: kernelSize) else { continue }
            guard let groups = groupCount(inputChannels: inputChannels) else {
                throw WorkaroundError.cannotSplit(weightName: spec.name, inputChannels: inputChannels, kernelSize: kernelSize)
            }
            splits.append(SplitConvolution(weightName: spec.name, inputChannels: inputChannels,
                                           kernelSize: kernelSize, groups: groups))
        }
        return Plan(splits: splits)
    }
}
