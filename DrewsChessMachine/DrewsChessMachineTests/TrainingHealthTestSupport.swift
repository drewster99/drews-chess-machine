import Foundation
import XCTest
@testable import DrewsChessMachine

/// Shared builders for the training-health tests.
enum TrainingHealthTestSupport {

    static func config(
        enabled: Bool = true,
        checkIntervalSteps: Int = 1000,
        learningGraceSteps: Int = 1000,
        lrWarmupSteps: Int = 1000,
        actions: TrainingHealthActions = TrainingHealthActions { _ in .log }
    ) throws -> TrainingHealthConfig {
        try TrainingHealthConfig(
            enabled: enabled, checkIntervalSteps: checkIntervalSteps, learningGraceSteps: learningGraceSteps,
            lrWarmupSteps: lrWarmupSteps, momentumCoefficient: 0.85,
            legalMassStallThreshold: 0.99, legalMassStallEvaluations: 8, actions: actions)
    }

    static func stamp(generation: Int = 0, stepsTrained: Int = 10_000, lastRecorded: Int? = nil) -> TrainingHealthStamp {
        TrainingHealthStamp(
            runID: UUID(), generation: generation, stepsTrainedByThisProcess: stepsTrained,
            lastRecordedTrainerStep: lastRecorded)
    }

    /// A window of one or more records summarized the way the monitor does.
    static func window(
        _ records: [TrainingHealthStepRecord],
        history: [TrainingHealthReferenceEntry] = []
    ) -> TrainingHealthWindowStatistics {
        TrainingHealthWindowStatistics.make(records: records, history: history)
    }

    static func record(
        _ step: Int,
        loss: Float? = 4.0,
        illegal: Float? = 0.01,
        gradient: Float? = 1.0,
        offset: Float? = nil,
        ms: Double? = 10
    ) -> TrainingHealthStepRecord {
        // A record with `offset` is a diagnostic step, which in-app carries
        // every diagnostic field together; rules 10–12's inputs get healthy
        // values so they are measured (not "no data") and raise nothing.
        TrainingHealthStepRecord(
            trainerStep: step, loss: loss, illegalMassPenalty: illegal, gradGlobalNorm: gradient,
            totalMs: ms, policyLogitMean: offset,
            policyEntropy: offset == nil ? nil : 2.0,
            valueAbsMean: offset == nil ? nil : 0.3,
            valueProbDraw: offset == nil ? nil : 0.5)
    }

    /// A healthy reference history: one entry every `stride` steps before
    /// `before`.
    static func history(before: Int, count: Int = 20, stride: Int = 50, loss: Float = 4.0, gradient: Float = 1.0)
        -> [TrainingHealthReferenceEntry] {
        (1...count).map { index in
            TrainingHealthReferenceEntry(trainerStep: before - index * stride, loss: loss, gradGlobalNorm: gradient)
        }.reversed()
    }

    static func liveObservation(
        step: Int,
        records: [TrainingHealthStepRecord]? = nil,
        history: [TrainingHealthReferenceEntry] = [],
        digest: LayerHealthDigest? = nil,
        stepsTrained: Int = 10_000
    ) -> TrainingHealthObservation {
        TrainingHealthObservation(
            trainerStep: step, window: window(records ?? [record(step)], history: history), layerHealth: digest,
            stamp: stamp(stepsTrained: stepsTrained), effectiveLearningRate: 0.1, effectiveMomentum: 0.85)
    }

    static func checkpointObservation(step: Int, digest: LayerHealthDigest, stepsTrained: Int = 10_000) -> TrainingHealthObservation {
        TrainingHealthObservation(
            trainerStep: step, window: nil, layerHealth: digest, stamp: stamp(stepsTrained: stepsTrained),
            effectiveLearningRate: nil, effectiveMomentum: nil)
    }

    static func deadDigest(
        dead: Int,
        channels: Int = 1040,
        sites: [(String, Int, Int?)] = [],
        tier: LayerHealthDigest.Tier = .live
    ) -> LayerHealthDigest {
        LayerHealthDigest(
            tier: tier,
            deadChannels: LayerHealthDigest.DeadChannels(
                modeledSiteCount: 9, modeledChannelCount: channels, parkedChannelCount: dead,
                sites: sites.map { LayerHealthDigest.SiteDeadChannels(site: $0.0, parkedChannelCount: $0.1, channelCount: $0.2) },
                coversEveryActivation: true),
            nonFiniteValueCount: 0, runningVariance: nil, valueFC1: nil)
    }

    static func valueFC1Digest(zero: Int, units: Int = 128) -> LayerHealthDigest {
        .valueFC1Only(LayerHealthDigest.ValueFC1Velocity(zeroVelocityUnitCount: zero, unitCount: units))
    }

    static func runningVarianceDigest(_ ratio: Double) -> LayerHealthDigest {
        LayerHealthDigest(
            tier: .live, deadChannels: nil, nonFiniteValueCount: 0,
            runningVariance: LayerHealthDigest.RunningVarianceRunaway(maxOverMedian: ratio, site: "blocks.0.bn2"),
            valueFC1: nil)
    }

    /// A test-bundle resource's text; fails loudly when the file is missing.
    static func resourceText(_ name: String, extension fileExtension: String, file: StaticString = #filePath, line: UInt = #line) throws -> String {
        let bundle = Bundle(for: BundleMarker.self)
        guard let url = bundle.url(forResource: name, withExtension: fileExtension) else {
            XCTFail("test resource \(name).\(fileExtension) is not in the test bundle", file: file, line: line)
            throw ResourceError.missing("\(name).\(fileExtension)")
        }
        return try String(contentsOf: url, encoding: .utf8)
    }

    static func resourceData(_ name: String, extension fileExtension: String, file: StaticString = #filePath, line: UInt = #line) throws -> Data {
        let bundle = Bundle(for: BundleMarker.self)
        guard let url = bundle.url(forResource: name, withExtension: fileExtension) else {
            XCTFail("test resource \(name).\(fileExtension) is not in the test bundle", file: file, line: line)
            throw ResourceError.missing("\(name).\(fileExtension)")
        }
        return try Data(contentsOf: url)
    }

    enum ResourceError: Error {
        case missing(String)
    }

    final class BundleMarker {}

    /// Collects a monitor's lines for assertions.
    final class LineSink: @unchecked Sendable {
        private let box = SyncBox<[TrainingHealthLogLine]>([])
        var lines: [TrainingHealthLogLine] { box.value }
        var texts: [String] { box.value.map(\.text) }
        var sink: TrainingHealthLogSink {
            let box = self.box
            return { line in box.modify { $0.append(line) } }
        }
    }

    /// A TrainStepTiming with the lean fields set (the rest not measured).
    static func timing(
        loss: Float = 4.0,
        illegal: Float = 0.01,
        gradient: Float = 1.0,
        totalMs: Double = 10,
        hasDiagnostics: Bool = false,
        policyLogitMean: Float = .nan
    ) -> TrainStepTiming {
        var timing = TrainStepTiming(
            dataPrepMs: 1, gpuRunMs: 2, readbackMs: 0.1, queueWaitMs: 0, totalMs: totalMs,
            loss: loss, policyLoss: loss, valueLoss: 0.5,
            policyEntropy: .nan,
            illegalMassPenalty: illegal,
            policyNonNegligibleCount: .nan,
            policyNonNegligibleIllegalCount: .nan,
            gradGlobalNorm: gradient,
            valueMean: .nan,
            valueAbsMean: .nan,
            valueProbWin: .nan, valueProbDraw: .nan, valueProbLoss: .nan,
            freshBaselineMs: nil,
            policyHeadWeightNorm: .nan,
            policyLogitAbsMax: .nan,
            playedMoveProb: .nan,
            playedMoveProbPosAdv: .nan,
            playedMoveProbNegAdv: .nan,
            advantageMean: .nan, advantageStd: .nan, advantageMin: .nan, advantageMax: .nan,
            advantageFracPositive: .nan, advantageFracSmall: .nan,
            advantageRaw: nil,
            policyLossWin: nil, policyLossLoss: nil,
            velocityNorm: .nan,
            sampledBatchMeanGameLength: .nan,
            sampledBatchDrawFraction: .nan,
            hasDiagnostics: hasDiagnostics)
        timing.policyLogitMean = policyLogitMean
        return timing
    }
}
