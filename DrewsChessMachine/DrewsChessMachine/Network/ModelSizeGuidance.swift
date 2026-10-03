//
//  ModelSizeGuidance.swift
//  DrewsChessMachine
//
//  How large a model this Mac can train, and where a given parameter count
//  sits against that.
//

import Foundation

/// Size guidance for a model about to be built, scaled to this Mac's
/// physical memory — the one source the Build New Model screen's readout,
/// the `[ARCH] size guidance` line of `--new-model`, `--derive-model` and
/// GUI builds, and the build-time memory refusal all read.
///
/// **Guidance, not limits** (owner decision, 2026-10-03). There is no cap on
/// block count, channels or kernel size: `NetworkArchitecture.validate()`
/// refuses only what cannot exist (a non-positive count, a count that
/// overflows `Int`). This adds the one build-time refusal — a model whose
/// training state cannot fit in physical memory at all — and otherwise only
/// says what to expect, so an unusual model stays buildable.
///
/// The thresholds are the owner's measurements on the reference machine
/// (`referencePhysicalMemoryBytes`): the top of the recommended range, and
/// the largest model the reference batch trains. Every threshold scales
/// linearly with installed memory. A model above the reference-batch level
/// still trains at a smaller batch, by the owner's working rule
/// `maximumParameters(batch) = maximumParameters(referenceBatch) ×
/// √(referenceBatch / batch)`. Batches below `smallestRecommendedBatchSize`
/// are not recommended, so a model too large even there is "likely too
/// large to train on this Mac" — a warning, not a refusal.
struct ModelSizeGuidance: Equatable, Sendable {

    /// Where a parameter count sits on this Mac.
    enum Verdict: Equatable, Sendable {
        /// At or below the recommended size.
        case withinRecommendedSize
        /// Above the recommended size, but trains at the reference batch.
        case trainsAtReferenceBatch
        /// Too large for the reference batch; trains at `batchSize`, the
        /// largest power-of-two batch, at least the smallest recommended
        /// one, that fits.
        case trainsAtReducedBatch(batchSize: Int)
        /// Too large even at the smallest recommended batch.
        case likelyTooLargeToTrain
        /// The training state alone exceeds physical memory: not buildable.
        case trainingStateExceedsPhysicalMemory
    }

    /// Bytes of training state per parameter: the fp32 working weights, the
    /// fp32 master weights, the momentum velocity and the gradient.
    static let trainingBytesPerParameter = 4 * MemoryLayout<Float>.size
    /// The machine the thresholds below were measured on.
    static let referencePhysicalMemoryBytes: UInt64 = 64 << 30
    /// The top of the recommended range on the reference machine.
    static let recommendedParametersAtReferenceMemory = 15_000_000.0
    /// The largest model the reference batch trains on the reference machine.
    static let referenceBatchParametersAtReferenceMemory = 20_000_000.0
    static let referenceBatchSize = 4096
    static let smallestRecommendedBatchSize = 512

    let parameterCount: Int
    let physicalMemoryBytes: UInt64
    let verdict: Verdict

    init(parameterCount: Int, physicalMemoryBytes: UInt64) {
        self.parameterCount = parameterCount
        self.physicalMemoryBytes = physicalMemoryBytes
        let maximumParametersInMemory = physicalMemoryBytes / UInt64(Self.trainingBytesPerParameter)
        let scale = Double(physicalMemoryBytes) / Double(Self.referencePhysicalMemoryBytes)
        let parameters = Double(parameterCount)
        if parameterCount.magnitude > maximumParametersInMemory {
            verdict = .trainingStateExceedsPhysicalMemory
        } else if parameters <= Self.recommendedParametersAtReferenceMemory * scale {
            verdict = .withinRecommendedSize
        } else if parameters <= Self.maximumParameters(atBatchSize: Self.referenceBatchSize, memoryScale: scale) {
            verdict = .trainsAtReferenceBatch
        } else if let batchSize = Self.reducedBatchSizes.first(where: {
            parameters <= Self.maximumParameters(atBatchSize: $0, memoryScale: scale)
        }) {
            verdict = .trainsAtReducedBatch(batchSize: batchSize)
        } else {
            verdict = .likelyTooLargeToTrain
        }
    }

    /// Guidance for `parameterCount` on the Mac this process runs on.
    static func forThisMac(parameterCount: Int) -> ModelSizeGuidance {
        ModelSizeGuidance(parameterCount: parameterCount, physicalMemoryBytes: ProcessInfo.processInfo.physicalMemory)
    }

    /// The power-of-two batches below the reference batch, largest first,
    /// down to the smallest recommended one.
    static var reducedBatchSizes: [Int] {
        Array(sequence(first: referenceBatchSize / 2, next: { $0 / 2 })
            .prefix(while: { $0 >= smallestRecommendedBatchSize }))
    }

    /// The largest parameter count that trains at `batchSize` on a machine
    /// with `memoryScale` times the reference memory.
    static func maximumParameters(atBatchSize batchSize: Int, memoryScale: Double) -> Double {
        referenceBatchParametersAtReferenceMemory * memoryScale
            * (Double(referenceBatchSize) / Double(batchSize)).squareRoot()
    }

    /// This Mac's memory as a multiple of the reference machine's.
    var memoryScale: Double {
        Double(physicalMemoryBytes) / Double(Self.referencePhysicalMemoryBytes)
    }

    var recommendedMaximumParameters: Double { Self.recommendedParametersAtReferenceMemory * memoryScale }

    func maximumParameters(atBatchSize batchSize: Int) -> Double {
        Self.maximumParameters(atBatchSize: batchSize, memoryScale: memoryScale)
    }

    /// The training state's size. A `Double`, because for a model this
    /// refuses it need not fit in an `Int`.
    var trainingStateBytes: Double { Double(parameterCount) * Double(Self.trainingBytesPerParameter) }

    /// Refuse a model whose training state cannot fit in physical memory —
    /// the one size a build refuses.
    func requireTrainingStateFitsInPhysicalMemory() throws {
        guard verdict != .trainingStateExceedsPhysicalMemory else {
            throw NetworkArchitectureError.trainingStateExceedsPhysicalMemory(
                parameterCount: parameterCount,
                trainingStateBytes: trainingStateBytes,
                physicalMemoryBytes: physicalMemoryBytes
            )
        }
    }

    /// One sentence for the Build New Model screen.
    var readout: String {
        let memory = Self.gigabytesText(Double(physicalMemoryBytes))
        let recommended = Self.parametersText(recommendedMaximumParameters)
        let atReference = Self.parametersText(maximumParameters(atBatchSize: Self.referenceBatchSize))
        switch verdict {
        case .withinRecommendedSize:
            return "Within the recommended size for this Mac (up to \(recommended) parameters with \(memory) of memory)."
        case .trainsAtReferenceBatch:
            return "Above the recommended \(recommended) parameters for this Mac (\(memory) of memory); "
                + "trains at batch \(Self.referenceBatchSize) (up to \(atReference))."
        case .trainsAtReducedBatch(let batchSize):
            return "Too large for batch \(Self.referenceBatchSize) on this Mac (up to \(atReference) with \(memory) of memory); "
                + "fits at batch \(batchSize) (up to \(Self.parametersText(maximumParameters(atBatchSize: batchSize)))) and below."
        case .likelyTooLargeToTrain:
            let atSmallest = Self.parametersText(maximumParameters(atBatchSize: Self.smallestRecommendedBatchSize))
            return "Likely too large to train on this Mac (\(memory) of memory): above \(atSmallest) parameters, "
                + "the most that fits at batch \(Self.smallestRecommendedBatchSize); smaller batches are not recommended."
        case .trainingStateExceedsPhysicalMemory:
            return "Cannot be trained on this Mac: its training state needs \(Self.gigabytesText(trainingStateBytes)), "
                + "more than the \(memory) of physical memory."
        }
    }

    /// The machine-readable token for `verdict` in the log line.
    var verdictToken: String {
        switch verdict {
        case .withinRecommendedSize: return "within_recommended"
        case .trainsAtReferenceBatch: return "trains_at_batch_\(Self.referenceBatchSize)"
        case .trainsAtReducedBatch(let batchSize): return "trains_at_batch_\(batchSize)"
        case .likelyTooLargeToTrain: return "likely_too_large"
        case .trainingStateExceedsPhysicalMemory: return "exceeds_physical_memory"
        }
    }

    /// The `[ARCH] size guidance` line for a model being built or written
    /// by `event` (e.g. `--new-model <name>`).
    func logLine(event: String) -> String {
        "[ARCH] size guidance (\(event)): parameters=\(parameterCount) "
            + "physical_memory=\(Self.gigabytesText(Double(physicalMemoryBytes)).replacingOccurrences(of: " ", with: "")) "
            + "recommended_max=\(Int(recommendedMaximumParameters.rounded(.down))) "
            + "batch\(Self.referenceBatchSize)_max=\(Int(maximumParameters(atBatchSize: Self.referenceBatchSize).rounded(.down))) "
            + "batch\(Self.smallestRecommendedBatchSize)_max="
            + "\(Int(maximumParameters(atBatchSize: Self.smallestRecommendedBatchSize).rounded(.down))) "
            + "verdict=\(verdictToken) | \(readout)"
    }

    /// `bytes` in base-2 gigabytes: whole when integral, else with one
    /// decimal.
    static func gigabytesText(_ bytes: Double) -> String {
        let gigabytes = bytes / Double(1 << 30)
        return gigabytes == gigabytes.rounded()
            ? String(format: "%.0f GB", gigabytes)
            : String(format: "%.1f GB", gigabytes)
    }

    /// A parameter count in millions, with one decimal.
    static func parametersText(_ parameters: Double) -> String {
        String(format: "%.1fM", parameters / 1_000_000)
    }
}
