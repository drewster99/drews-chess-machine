import Foundation

extension NumericsAudit {

    /// Audit a live network: its current weights, named by its graph
    /// variables, over the standard position set (start position, the
    /// Lichess bot's filed games under `lichessDirectory`, and a corpus
    /// sample when `corpusShardURL` is given).
    static func run(
        network: ChessNetwork,
        modelLabel: String,
        modelID: String?,
        trainingStep: Int?,
        masters: [[Float]]?,
        mastersNote: String?,
        corpusShardURL: URL?,
        lichessDirectory: LichessBotDataDirectory?
    ) async throws -> Result {
        let weights = try await network.exportWeights()
        let names = (network.trainableVariables + network.bnRunningStatsVariables).map { $0.operation.name }
        let positions = try await buildPositionSetOffPool(
            encoding: network.arch.inputEncoding,
            corpusShardURL: corpusShardURL,
            lichessDirectory: lichessDirectory
        )
        return try await run(
            names: names,
            weights: weights,
            arch: network.arch,
            masters: masters,
            mastersNote: mastersNote,
            positions: positions,
            dynamicSkippedReason: nil,
            modelLabel: modelLabel,
            modelID: modelID,
            trainingStep: trainingStep
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

    /// Why a champion has no masters to compare.
    static let championMastersNote = "the champion is an inference network; it has no fp32 masters"
}
