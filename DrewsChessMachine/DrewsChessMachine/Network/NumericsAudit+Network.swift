import Foundation

extension NumericsAudit {

    /// Audit a live network's weights from its analysis snapshot (one
    /// export, shared with the other analyses of the same request) over the
    /// standard position set (start position, the Lichess bot's filed games
    /// under `lichessDirectory`, and a corpus sample when `corpusShardURL` is
    /// given). The result's step is the snapshot's: the step the audited
    /// weights were taken at.
    static func run(
        snapshot: AnalyzedNetworkSnapshot,
        modelLabel: String,
        masters: [[Float]]?,
        mastersNote: String?,
        velocity: LayerHealth.VelocitySource,
        corpusShardURL: URL?,
        lichessDirectory: LichessBotDataDirectory?
    ) async throws -> Result {
        let positions = try await buildPositionSetOffPool(
            encoding: snapshot.architecture.inputEncoding,
            corpusShardURL: corpusShardURL,
            lichessDirectory: lichessDirectory
        )
        return try await run(
            names: snapshot.names,
            weights: snapshot.weights,
            arch: snapshot.architecture,
            masters: masters,
            mastersNote: mastersNote,
            velocity: velocity,
            positions: positions,
            dynamicSkippedReason: nil,
            policyTailPrecision: snapshot.policyTailPrecision,
            modelLabel: modelLabel,
            modelID: snapshot.modelID,
            trainingStep: snapshot.trainingStep
        )
    }

    /// `buildPositionSet` on a GCD queue: it reads and replays game files
    /// synchronously.
    static func buildPositionSetOffPool(
        encoding: InputEncoding,
        corpusShardURL: URL?,
        lichessDirectory: LichessBotDataDirectory?
    ) async throws -> PositionSet {
        try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    continuation.resume(returning: try buildPositionSet(
                        encoding: encoding,
                        corpusShardURL: corpusShardURL,
                        lichessDirectory: lichessDirectory
                    ))
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    /// Masters for a trainer, or why they weren't read. Masters are read only
    /// while training is stopped (the read races SGD otherwise); a trainer
    /// computing in fp32 keeps none.
    static func trainerMasters(
        trainer: ChessTrainer,
        trainingIsRunning: Bool
    ) async throws -> (masters: [[Float]]?, note: String?) {
        guard !trainingIsRunning else {
            return (nil, "training is running, so the fp32 masters can't be read safely")
        }
        let masters = try await trainer.readMasterValues()
        guard !masters.isEmpty else {
            return (nil, "the trainer computes in fp32 and keeps no separate masters")
        }
        return (masters, nil)
    }

    /// A trainer's optimizer velocity for the layer-health checks — the
    /// tail of its trainer-state export, one tensor per trainable — or why
    /// it wasn't read. Like the masters, it is read only while training is
    /// stopped: the export needs training paused.
    /// `networkTensorCount` is the trainables plus the BN running statistics,
    /// which the export carries before the velocity.
    static func trainerVelocity(
        trainer: ChessTrainer,
        networkTensorCount: Int,
        trainableCount: Int,
        trainingIsRunning: Bool
    ) async throws -> LayerHealth.VelocitySource {
        guard !trainingIsRunning else {
            return .unavailable(reason: "training is running, so the trainer's velocity can't be read safely")
        }
        let state = try await trainer.exportTrainerWeights()
        guard state.count == networkTensorCount + trainableCount else {
            throw NumericsAuditError.trainerStateCountMismatch(tensors: state.count, expected: networkTensorCount + trainableCount)
        }
        return .trainerVelocity(Array(state.suffix(trainableCount)))
    }

    /// Why an audit of the champion has no velocity for the layer-health
    /// checks: an inference network holds no optimizer state.
    static let liveNetworkVelocityNote = "an inference network holds no optimizer state; velocity lives in the trainer"

    /// Why a champion has no masters to compare.
    static let championMastersNote = "the champion is an inference network; it has no fp32 masters"
}
