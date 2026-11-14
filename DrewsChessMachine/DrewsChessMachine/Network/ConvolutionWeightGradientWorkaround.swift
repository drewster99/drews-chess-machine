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
//  Which convolutions (owner decision 2026-11-14): those that run in fp32 in a
//  training graph and pass the limit at the planned batch, 4096 — the batch every
//  run uses. Planning for the largest batch the app accepts (65,536) split far
//  more than the defect needs and doubled a 16-block 3×3 net's step time, for
//  batches nobody trains at. Measured shapes below the size limit never failed,
//  and 5×5 / 24 channels was exact even past it, so even at 4096 the rule is
//  wider than the defect: a split convolution costs some speed, an unsplit wrong
//  one costs the run. Inference graphs are never split: they have no weight
//  gradient, and Core ML export records only inference graphs.
//
//  The guard: each plan also records the largest batch its graph trains
//  correctly — the batch just below where the first unsplit fp32 convolution
//  would pass the limit, and no more than the batch split convolutions were
//  measured correct at. A training step above it is refused
//  (`Plan.requireCorrectWeightGradients(batchSize:)`, one comparison per step),
//  so a batch the plan did not cover stops training instead of training on a
//  wrong gradient. Training at such a batch means re-planning: building graphs
//  for the batches in use, or a larger planned batch.
//
import Foundation


enum ConvolutionWeightGradientWorkaround {

    /// The largest element count MPSGraph's fp32 weight-gradient path handles
    /// correctly (a 32-bit signed index).
    static let elementLimit = Int(Int32.max)

    /// Squares per board — the H·W of every convolution in the network.
    static let boardSquares = NetworkArchitecture.boardSize * NetworkArchitecture.boardSize

    /// The training batch the split is planned for (owner decision 2026-11-14:
    /// every run trains at 4096). A training step above a plan's
    /// `largestCorrectBatch` is refused, so a different batch is caught, never
    /// trained on a wrong gradient.
    static let plannedTrainingBatchSize = 4096

    /// The largest batch a split convolution was measured correct at (two
    /// input-channel groups, 9×9 / 128 channels; `analysis-fp32/README.md`
    /// "Workarounds"). Larger batches are unmeasured, so a plan with splits
    /// refuses them.
    static let splitMeasuredCorrectThroughBatch = 32_768

    /// The largest training batch any path accepts.
    static let batchCeiling = TrainingBatchSize.declaredClosedRange.upperBound

    /// What a split costs in training speed, said wherever the split is announced
    /// (the `[ARCH]` notice and Build New Model's warning) so nobody trains in fp32
    /// without knowing the price. The cost depends on the shape and is not modeled;
    /// these are the measured cases (corpus replay, batch 4096, macOS 27.2,
    /// M5 Max; `experiments/20261008-zlra-selfplay-lr/analysis-fp32/README.md`
    /// "Speed"). Re-measure when the rule or MPSGraph changes.
    static let speedPenaltyNote = "Splitting costs training speed: measured at batch 4096, about +10% per step on 3 blocks "
        + "of 9x9 / 128 channels and about 2x on 16 blocks of 3x3 / 128 channels (fp32 unsplit is already about 2x bf16)."

    /// One convolution the training graph splits.
    struct SplitConvolution: Equatable, Sendable {
        /// The weight's plan (safetensors) name, e.g. `blocks.0.conv1.weight`.
        let weightName: String
        let inputChannels: Int
        let kernelSize: Int
        /// How many input-channel groups the convolution is built as.
        let groups: Int
    }

    /// What sets a plan's largest correct batch.
    enum BatchLimitSource: Equatable, Sendable {
        /// An fp32 convolution left unsplit at the planned batch, which passes
        /// the element limit at the next batch up.
        case unsplitConvolution(weightName: String, inputChannels: Int, kernelSize: Int)
        /// The split convolutions, beyond the batch they were measured correct at.
        case splitConvolutionsUnmeasured
    }

    /// The batches a training graph computes correct fp32 weight gradients for.
    enum TrainingBatchLimit: Equatable, Sendable {
        /// Every batch the app accepts: no convolution runs in fp32, or none can
        /// pass the limit below the batch ceiling.
        case everyAcceptedBatch
        /// Batches up to `largestCorrectBatch`; a larger step is refused.
        case upTo(largestCorrectBatch: Int, source: BatchLimitSource)
    }

    /// The split decision for one training graph, by weight name, and the
    /// batches it covers.
    struct Plan: Equatable, Sendable {
        let splits: [SplitConvolution]
        let batchLimit: TrainingBatchLimit

        static let none = Plan(splits: [], batchLimit: .everyAcceptedBatch)

        /// The number of input-channel groups for the convolution whose weight
        /// is `weightName`: 1 when it is not split.
        func groups(forWeight weightName: String) -> Int {
            splits.first { $0.weightName == weightName }?.groups ?? 1
        }

        var isEmpty: Bool { splits.isEmpty }

        /// Refuses a training step this plan's graph would compute a wrong fp32
        /// weight gradient for. Called before every SGD step: one comparison.
        func requireCorrectWeightGradients(batchSize: Int) throws {
            guard case let .upTo(largestCorrectBatch, source) = batchLimit,
                  batchSize > largestCorrectBatch else { return }
            throw WorkaroundError.batchAboveCorrectLimit(
                batchSize: batchSize, largestCorrectBatch: largestCorrectBatch, source: source)
        }

        /// The batch-limit sentence of the notice and Build New Model's warning;
        /// nil when every accepted batch is covered.
        var batchLimitSentence: String? {
            guard case let .upTo(largestCorrectBatch, source) = batchLimit else { return nil }
            let reason: String
            switch source {
            case let .unsplitConvolution(name, cin, k):
                reason = "\(name) (\(cin)ch \(k)x\(k)), unsplit at the planned batch \(plannedTrainingBatchSize), "
                    + "would get a wrong gradient above it"
            case .splitConvolutionsUnmeasured:
                reason = "the split convolutions are measured correct only up to it"
            }
            return "Training refuses batches above \(largestCorrectBatch): \(reason)."
        }

        /// The `[ARCH]` notice for a graph built (or a model to be trained) with
        /// this plan, tagged with what produced it (`training graph`,
        /// `--new-model <name>`, `--derive-model <file>`, `Build Network`); nil
        /// when the plan neither splits nor limits the batch.
        func notice(event: String) -> String? {
            guard let limit = batchLimitSentence else { return nil }
            let prefix = "[ARCH] fp32 conv weight-gradient workaround (\(event)): "
            let cause = "MPSGraph computes a wrong fp32 weight gradient once batch x 64 x C_in x k^2 passes 2^31"
            guard !splits.isEmpty else {
                return prefix + "no convolution needs splitting at the planned batch \(plannedTrainingBatchSize) "
                    + "(\(cause)). \(limit)"
            }
            let list = splits.map { "\($0.weightName) (\($0.inputChannels)ch \($0.kernelSize)x\($0.kernelSize) -> \($0.groups) groups)" }
                .joined(separator: ", ")
            return prefix + "\(splits.count) convolution(s) are split over their input channels in training graphs, "
                + "because \(cause) (planned batch \(plannedTrainingBatchSize)): \(list). Same function and gradients; "
                + "inference graphs are unchanged. \(limit) \(ConvolutionWeightGradientWorkaround.speedPenaltyNote)"
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
        /// A training step larger than the largest batch the graph's fp32 weight
        /// gradients are correct for.
        case batchAboveCorrectLimit(batchSize: Int, largestCorrectBatch: Int, source: BatchLimitSource)

        var errorDescription: String? {
            switch self {
            case let .cannotSplit(name, cin, k):
                return "\(name) (\(cin) input channels, \(k)x\(k)) runs in fp32 and passes MPSGraph's 2^31-element "
                    + "fp32 weight-gradient limit at the planned batch \(plannedTrainingBatchSize), but \(cin) input "
                    + "channels cannot be split into groups; training this architecture in fp32 would give a wrong gradient"
            case let .unplannedAffectedConvolution(name, cin, k):
                return "internal error: \(name) (\(cin) input channels, \(k)x\(k)) is built in fp32 in a training graph "
                    + "and passes the 2^31-element limit at the planned batch, but the weight-gradient workaround plan "
                    + "does not split it"
            case let .batchAboveCorrectLimit(batchSize, largest, source):
                let reason: String
                switch source {
                case let .unsplitConvolution(weightName, cin, k):
                    reason = "\(weightName) (\(cin) input channels, \(k)x\(k)) was left unsplit for the planned batch "
                        + "\(plannedTrainingBatchSize) and passes MPSGraph's 2^31-element fp32 weight-gradient limit above "
                        + "batch \(largest)"
                case .splitConvolutionsUnmeasured:
                    reason = "the split convolutions are measured correct only up to batch \(largest)"
                }
                return "training batch \(batchSize) refused: \(reason); its fp32 weight gradient would be wrong. "
                    + "Train at batch \(largest) or less, or re-plan ConvolutionWeightGradientWorkaround for this batch"
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

    /// The smallest batch at which this shape passes the element limit.
    static func firstWrongBatch(inputChannels: Int, kernelSize: Int) -> Int {
        let elementsPerPosition = elementCount(batch: 1, inputChannels: inputChannels, kernelSize: kernelSize)
        precondition(elementsPerPosition > 0, "a convolution has at least one input channel and kernel tap")
        return elementLimit / elementsPerPosition + 1
    }

    /// Whether an fp32 convolution with this shape needs the split at the
    /// planned batch.
    static func isAffected(inputChannels: Int, kernelSize: Int) -> Bool {
        elementCount(batch: plannedTrainingBatchSize, inputChannels: inputChannels, kernelSize: kernelSize) > elementLimit
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

    /// The splits a training graph of `architecture` is built with and the
    /// batches it then trains correctly. Throws when an affected convolution
    /// cannot be split.
    static func plan(for architecture: NetworkArchitecture) throws -> Plan {
        var splits: [SplitConvolution] = []
        // Largest correct batch, and what sets it, over every fp32 convolution.
        var limit: (batch: Int, source: BatchLimitSource)?
        for spec in architecture.weightTensorPlan() where spec.kind == .conv {
            // Native conv shape: [C_out, C_in, k, k].
            let inputChannels = spec.shape[1]
            let kernelSize = spec.shape[2]
            guard runsInFloat32(weightName: spec.name, architecture: architecture) else { continue }
            let candidate: (batch: Int, source: BatchLimitSource)
            if isAffected(inputChannels: inputChannels, kernelSize: kernelSize) {
                guard let groups = groupCount(inputChannels: inputChannels) else {
                    throw WorkaroundError.cannotSplit(weightName: spec.name, inputChannels: inputChannels,
                                                      kernelSize: kernelSize)
                }
                splits.append(SplitConvolution(weightName: spec.name, inputChannels: inputChannels,
                                               kernelSize: kernelSize, groups: groups))
                candidate = (splitMeasuredCorrectThroughBatch, .splitConvolutionsUnmeasured)
            } else {
                candidate = (firstWrongBatch(inputChannels: inputChannels, kernelSize: kernelSize) - 1,
                             .unsplitConvolution(weightName: spec.name, inputChannels: inputChannels,
                                                 kernelSize: kernelSize))
            }
            if limit.map({ candidate.batch < $0.batch }) ?? true {
                limit = candidate
            }
        }
        guard let limit, limit.batch < batchCeiling else {
            return Plan(splits: splits, batchLimit: .everyAcceptedBatch)
        }
        return Plan(splits: splits, batchLimit: .upTo(largestCorrectBatch: limit.batch, source: limit.source))
    }
}
