import Foundation

extension NumericsAudit {

    /// Audit a live network's weights from its analysis snapshot (one export,
    /// shared with the other analyses of the same request), comparing them
    /// with `optimizerState` — read in the same cut — over the standard
    /// position set (start position, the Lichess bot's filed games under
    /// `lichessDirectory`, and a corpus sample when `corpusShardURL` is
    /// given). The result's step is the snapshot's: the step the audited
    /// weights were taken at. The head-bias init means come from the
    /// snapshot's init reference.
    static func run(
        snapshot: AnalyzedNetworkSnapshot,
        optimizerState: AnalysisOptimizerState,
        modelLabel: String,
        corpusShardURL: URL?,
        lichessDirectory: LichessBotDataDirectory?
    ) async throws -> Result {
        let positions = try await buildPositionSetOffPool(
            encoding: snapshot.architecture.inputEncoding,
            corpusShardURL: corpusShardURL,
            lichessDirectory: lichessDirectory
        )
        let optimizer = optimizerInputs(optimizerState)
        return try await run(
            names: snapshot.names,
            weights: snapshot.weights,
            arch: snapshot.architecture,
            initReference: snapshot.initReference,
            masters: optimizer.masters,
            mastersNote: optimizer.mastersNote,
            velocity: optimizer.velocity,
            positions: positions,
            dynamicSkippedReason: nil,
            modelLabel: modelLabel,
            modelID: snapshot.modelID,
            trainingStep: snapshot.trainingStep
        )
    }

    /// The masters and velocity an audit compares, from the optimizer state
    /// taken with the snapshot's weights, or why there are none.
    static func optimizerInputs(_ state: AnalysisOptimizerState)
        -> (masters: [[Float]]?, mastersNote: String?, velocity: LayerHealth.VelocitySource) {
        switch state {
        case .inferenceNetwork:
            return (nil, championMastersNote, .unavailable(reason: liveNetworkVelocityNote))
        case .trainer(let masters, let velocity):
            return (masters, masters == nil ? fp32TrainerMastersNote : nil, .trainerVelocity(velocity))
        }
    }

    /// Why a trainer has no masters to compare.
    static let fp32TrainerMastersNote = "the trainer computes in fp32 and keeps no separate masters"

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

    /// Why an audit of the champion has no velocity for the layer-health
    /// checks: an inference network holds no optimizer state.
    static let liveNetworkVelocityNote = "an inference network holds no optimizer state; velocity lives in the trainer"

    /// Why a champion has no masters to compare.
    static let championMastersNote = "the champion is an inference network; it has no fp32 masters"
}
