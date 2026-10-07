import Foundation

/// The analyzers' one weights snapshot per network (`AnalyzedNetworkSnapshot`).
///
/// Whose weights a snapshot holds — model ID, init seed, training step — is
/// read on the main actor immediately before the export and checked again
/// immediately after it, with the replacement-window test the Lichess bot's
/// trainer snapshots use (`LichessBotSessionModelProvider`): a replacement in
/// progress at the start, or one that overlapped the export, refuses the
/// snapshot instead of labeling the weights with an identity they do not
/// have. Nothing about the weights is captured at the button press, so a
/// request reads the identity when it exports.
///
/// The init reference is taken after that check, outside the trainer's
/// queue, from the request's `AnalysisInitReferenceCache`: a replacement
/// during the reference build does not touch the exported weights, and the
/// networks of one request that share an architecture and seed share one
/// build.
extension SessionController {

    /// A network's snapshot with the optimizer state taken in the same cut.
    struct AnalysisCapture: Sendable {
        let snapshot: AnalyzedNetworkSnapshot
        let optimizerState: AnalysisOptimizerState
    }

    /// Snapshot the session's `role` network as it is now, its init
    /// reference from `initReferences` (one cache per request).
    func analysisSnapshot(
        of role: AnalyzedNetworkSnapshot.Role,
        initReferences: AnalysisInitReferenceCache
    ) async throws -> AnalysisCapture {
        switch role {
        case .champion: return try await championAnalysisSnapshot(initReferences: initReferences)
        case .trainer: return try await trainerAnalysisSnapshot(initReferences: initReferences)
        }
    }

    /// The champion. Its weights change only by a replacement (a load or a
    /// promotion), which `championWeightIdentity` brackets from before the
    /// load until the new origin is recorded, so the origin read before the
    /// export describes the exported weights when no replacement was open at
    /// the start and none began before the end.
    private func championAnalysisSnapshot(initReferences: AnalysisInitReferenceCache) async throws -> AnalysisCapture {
        guard let champion = network else { throw AnalysisSnapshotError.noNetwork(.champion) }
        let identityBefore = championWeightIdentity
        guard !identityBefore.awaitingOrigin else {
            throw AnalysisSnapshotError.weightsBeingReplaced(.champion)
        }
        let modelID = champion.identifier
        let described = Self.championAnalysisDescription(origin: championOrigin)
        let weights = try await champion.exportWeights()
        guard network === champion, championWeightIdentity == identityBefore, champion.identifier == modelID else {
            throw AnalysisSnapshotError.changedDuringExport(.champion)
        }
        let takenAt = Date()
        // `ChessMPSNetwork.network` is a `let` whose variable lists are set
        // only at build, so reading them here is safe.
        let net = champion.network
        let snapshot = try await Self.makeAnalysisSnapshot(
            role: .champion, modelID: modelID?.description, architecture: net.arch,
            names: (net.trainableVariables + net.bnRunningStatsVariables).map { $0.operation.name },
            weights: weights, trainableCount: net.trainableVariables.count,
            trainingStep: described.trainingStep, trainingStepSource: described.stepSource,
            takenAt: takenAt, initialization: described.initialization, initReferences: initReferences)
        return AnalysisCapture(snapshot: snapshot, optimizerState: .inferenceNetwork)
    }

    /// The trainer. Weights, step, masters and velocity are one cut
    /// (`ChessTrainer.exportAnalysisState()`). The model ID and the init seed
    /// (`lineageTracker`, which a Play-and-Train start begins in the same
    /// main-actor turn as it stamps the trainer's ID) are read just before it
    /// and hold for its weights only when no weight replacement — reset,
    /// load, promotion rewind, batch-size sweep — was open at the start or
    /// began before the end (`ChessTrainer.weightIdentityState`). A sweep's
    /// reset stays open until the next Play-and-Train start stamps an ID, so
    /// sweep-drawn weights are never labeled with the run's seed.
    private func trainerAnalysisSnapshot(initReferences: AnalysisInitReferenceCache) async throws -> AnalysisCapture {
        let cut = try await trainerCut()
        let state = cut.state
        let snapshot = try await Self.makeAnalysisSnapshot(
            role: .trainer, modelID: cut.modelID, architecture: state.architecture,
            names: state.names, weights: state.weights,
            trainableCount: state.trainableCount, trainingStep: state.completedSteps,
            trainingStepSource: "the trainer's completed SGD steps, read with the weights in one trainer-queue turn",
            takenAt: cut.takenAt, initialization: cut.initialization, initReferences: initReferences)
        return AnalysisCapture(snapshot: snapshot,
                               optimizerState: .trainer(masters: state.masters, velocity: state.velocity))
    }

    /// The trainer's weights, step, masters and velocity as one cut, with
    /// the identity read around it (see `trainerAnalysisSnapshot`). Exposed
    /// apart from the snapshot for a reader that needs only the weights —
    /// the single replay analysis's entropy probe — so it gets the same
    /// identity-checked cut without paying for an init reference it never
    /// reads.
    struct TrainerCut: Sendable {
        let state: ChessTrainer.AnalysisState
        let modelID: String?
        /// How the weights' run drew its starting weights, nil when unknown.
        let initialization: ModelInitRecord?
        let takenAt: Date

        var modelLabel: String { AnalyzedNetworkSnapshot.modelLabel(role: .trainer, modelID: modelID) }
    }

    func trainerCut() async throws -> TrainerCut {
        guard let trainer else { throw AnalysisSnapshotError.noNetwork(.trainer) }
        let identityBefore = trainer.weightIdentityState
        guard !identityBefore.awaitingIdentity else {
            throw AnalysisSnapshotError.weightsBeingReplaced(.trainer)
        }
        let modelID = trainer.identifier
        let initialization = lineageTracker?.startingInitialization
        let state = try await trainer.exportAnalysisState()
        guard self.trainer === trainer, trainer.weightIdentityState == identityBefore, trainer.identifier == modelID else {
            throw AnalysisSnapshotError.changedDuringExport(.trainer)
        }
        return TrainerCut(state: state, modelID: modelID?.description, initialization: initialization, takenAt: Date())
    }

    /// What the champion's origin says about its weights: the step (the
    /// champion save's reading, `championFileTrainingStep`), where it comes
    /// from, and the init seed.
    private static func championAnalysisDescription(origin: ChampionOrigin?)
        -> (trainingStep: Int?, stepSource: String, initialization: ModelInitRecord?) {
        guard let origin else {
            return (nil, "not known: \(LineageSegmentError.noChampionOrigin.localizedDescription)", nil)
        }
        let step = championFileTrainingStep(recordedOrigin: origin)
        switch origin {
        case .built(let record):
            return (step, "built in this process; never trained", record)
        case .file(let source, _):
            return (step,
                    step == nil
                        ? "loaded or promoted from a file that states no training step"
                        : "the training step its source file or promotion states",
                    source.lineage.record?.rng.initialization)
        }
    }

    /// The snapshot with its init reference, taken from `initReferences` and
    /// checked to describe exactly these variables. `nonisolated` so the
    /// reference's build and comparisons run off the main actor (its graph
    /// build runs on GCD).
    nonisolated private static func makeAnalysisSnapshot(
        role: AnalyzedNetworkSnapshot.Role, modelID: String?, architecture: NetworkArchitecture,
        names: [String], weights: [[Float]],
        trainableCount: Int, trainingStep: Int?, trainingStepSource: String, takenAt: Date,
        initialization: ModelInitRecord?, initReferences: AnalysisInitReferenceCache
    ) async throws -> AnalyzedNetworkSnapshot {
        guard weights.count == names.count else {
            throw AnalysisSnapshotError.weightCountMismatch(expected: names.count, got: weights.count)
        }
        let reference = try await initReferences.reference(architecture: architecture, initialization: initialization)
        try reference.requireMatches(variableNames: names, trainableCount: trainableCount)
        return AnalyzedNetworkSnapshot(
            role: role, modelID: modelID, architecture: architecture,
            names: names, weights: weights,
            trainableCount: trainableCount, trainingStep: trainingStep,
            trainingStepSource: trainingStepSource, takenAt: takenAt, initReference: reference)
    }
}

enum AnalysisSnapshotError: LocalizedError {
    case weightCountMismatch(expected: Int, got: Int)
    /// No such network is loaded.
    case noNetwork(AnalyzedNetworkSnapshot.Role)
    /// A weight replacement has begun and its identity is not recorded yet.
    case weightsBeingReplaced(AnalyzedNetworkSnapshot.Role)
    /// A replacement began, or the network or its ID changed, during the export.
    case changedDuringExport(AnalyzedNetworkSnapshot.Role)

    var errorDescription: String? {
        switch self {
        case .weightCountMismatch(let expected, let got):
            return "The weight export returned \(got) tensors; the network has \(expected) variables"
        case .noNetwork(let role):
            return "No \(role.rawValue) network is loaded"
        case .weightsBeingReplaced(.champion):
            return "The champion's weights are being replaced (a load or promotion), or the last replacement failed, so their origin is not recorded; run the analysis after the replacement finishes, or after the next load, build or promotion"
        case .weightsBeingReplaced(.trainer):
            return "The trainer's weights are being replaced (a promotion, reset, load or batch-size sweep) and its ID is not stamped yet — after a sweep that happens at the next Play-and-Train start — so they can't be attributed"
        case .changedDuringExport(let role):
            return "The \(role.rawValue)'s weights or identity changed during the export; run the analysis again"
        }
    }
}
