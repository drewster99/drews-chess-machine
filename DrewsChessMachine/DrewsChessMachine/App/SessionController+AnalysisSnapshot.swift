import Foundation

/// The analyzers' one weights snapshot per network (`AnalyzedNetworkSnapshot`):
/// what the main actor knows about the network is captured first, then the
/// weights are exported once off the main actor and the init reference is
/// built from the same architecture.
extension SessionController {

    /// A network to analyze, as the main actor sees it.
    struct AnalysisTarget: Sendable {
        enum Source: Sendable {
            /// The champion. Its weights change only on a promotion, so the
            /// step its origin states is the step of what is exported —
            /// checked after the export by its model ID.
            case champion(ChessNetwork, trainingStep: Int?, stepSource: String)
            /// The trainer: weights and step are read as one pair.
            case trainer(ChessTrainer)
        }

        let source: Source
        let modelID: String?
        /// How the weights' run drew its starting weights, nil when unknown.
        let initialization: ModelInitRecord?

        var role: AnalyzedNetworkSnapshot.Role {
            switch source {
            case .champion: return .champion
            case .trainer: return .trainer
            }
        }

        var modelLabel: String {
            AnalyzedNetworkSnapshot.modelLabel(role: role, modelID: modelID)
        }
    }

    /// The champion as an analysis target, or nil when none is loaded.
    func championAnalysisTarget() -> AnalysisTarget? {
        guard let champion = network else { return nil }
        let trainingStep: Int?
        let stepSource: String
        do {
            trainingStep = try Self.championFileTrainingStep(origin: championOrigin)
            switch championOrigin {
            case .built:
                stepSource = "built in this process; never trained"
            case .file:
                stepSource = trainingStep == nil
                    ? "loaded or promoted from a file that states no training step"
                    : "the training step its source file or promotion states"
            case nil:
                stepSource = "the champion's origin is not recorded"
            }
        } catch {
            trainingStep = nil
            stepSource = "not known: \(error.localizedDescription)"
        }
        let initialization: ModelInitRecord?
        switch championOrigin {
        case .built(let record): initialization = record
        case .file(let source, _): initialization = source.lineage.record?.rng.initialization
        case nil: initialization = nil
        }
        return AnalysisTarget(
            source: .champion(champion.network, trainingStep: trainingStep, stepSource: stepSource),
            modelID: champion.identifier?.description,
            initialization: initialization
        )
    }

    /// The trainer as an analysis target, or nil when none exists.
    func trainerAnalysisTarget() -> AnalysisTarget? {
        guard let trainer else { return nil }
        return AnalysisTarget(
            source: .trainer(trainer),
            modelID: trainer.identifier?.description,
            initialization: lineageTracker?.startingInitialization
        )
    }

    /// Export the target's weights once and build its init reference.
    nonisolated static func takeAnalysisSnapshot(of target: AnalysisTarget) async throws -> AnalyzedNetworkSnapshot {
        let network: ChessNetwork
        let weights: [[Float]]
        let trainingStep: Int?
        let stepSource: String
        switch target.source {
        case .champion(let champion, let step, let source):
            network = champion
            weights = try await champion.exportWeights()
            trainingStep = step
            stepSource = source
        case .trainer(let trainer):
            network = trainer.network
            let pair = try await trainer.exportWeightsWithCompletedSteps()
            weights = pair.weights
            trainingStep = pair.completedSteps
            stepSource = "the trainer's completed SGD steps, read with the weights as one pair"
        }
        let takenAt = Date()
        let names = (network.trainableVariables + network.bnRunningStatsVariables).map { $0.operation.name }
        guard weights.count == names.count else {
            throw AnalysisSnapshotError.weightCountMismatch(expected: names.count, got: weights.count)
        }
        let reference = try await AnalysisInitReference.build(
            architecture: network.arch, initialization: target.initialization, names: names)
        return AnalyzedNetworkSnapshot(
            role: target.role,
            modelID: target.modelID,
            architecture: network.arch,
            policyTailPrecision: network.policyTailPrecision,
            names: names,
            weights: weights,
            trainableCount: network.trainableVariables.count,
            trainingStep: trainingStep,
            trainingStepSource: stepSource,
            takenAt: takenAt,
            initReference: reference
        )
    }

    /// `takeAnalysisSnapshot`, then — for the champion — a check on the main
    /// actor that no promotion replaced it during the export, so the step its
    /// origin stated still describes the exported weights.
    func analysisSnapshot(of target: AnalysisTarget) async throws -> AnalyzedNetworkSnapshot {
        let snapshot = try await Self.takeAnalysisSnapshot(of: target)
        if case .champion = target.source, network?.identifier?.description != target.modelID {
            throw AnalysisSnapshotError.championReplaced(was: target.modelID, now: network?.identifier?.description)
        }
        return snapshot
    }
}

enum AnalysisSnapshotError: LocalizedError {
    case weightCountMismatch(expected: Int, got: Int)
    /// A promotion replaced the champion while it was being exported.
    case championReplaced(was: String?, now: String?)

    var errorDescription: String? {
        switch self {
        case .weightCountMismatch(let expected, let got):
            return "The weight export returned \(got) tensors; the network has \(expected) variables"
        case .championReplaced(let was, let now):
            return "The champion changed from \(was ?? "<no-id>") to \(now ?? "<no-id>") during the export; run the analysis again"
        }
    }
}
