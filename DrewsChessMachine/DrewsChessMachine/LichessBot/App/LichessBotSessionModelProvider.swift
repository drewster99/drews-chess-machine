import Foundation

enum LichessBotSessionModelError: LocalizedError, Equatable {
    case sessionGone
    case missingModelID(source: String)

    var errorDescription: String? {
        switch self {
        case .sessionGone:
            return "The training session is not available"
        case .missingModelID(let source):
            return "The \(source) network has no ModelID, so games played with it could not be attributed"
        }
    }
}

/// Champion and trainer weights for the Lichess bot, taken from the live
/// `SessionController` (plan §9). Created with the app-level bot controller
/// and attached to the session once the main view creates it; until then
/// every source reports the session as unavailable.
///
/// Weights come from `exportWeights()`, the export used by human play and
/// the probe watchers: it runs on the network's own execution queue and, for
/// the trainer, holds the weight lock SGD uses, so it is safe while training
/// runs and never needs the self-play or training gates.
@MainActor
final class LichessBotSessionModelProvider: LichessBotModelProvider {
    private weak var session: SessionController?

    init() {}

    func attach(session: SessionController) {
        self.session = session
    }

    func championModelID() async -> String? {
        session?.network?.identifier.map { "\($0)" }
    }

    func championSnapshot() async throws -> LichessBotWeightsSnapshot {
        guard let session else { throw LichessBotSessionModelError.sessionGone }
        guard let champion = session.network else { throw LichessBotModelError.noChampion }
        guard let identifier = champion.identifier else {
            throw LichessBotSessionModelError.missingModelID(source: "champion")
        }
        let weights = try await champion.exportWeights()
        return LichessBotWeightsSnapshot(weights: weights, architecture: champion.arch, modelID: "\(identifier)", trainingStep: nil)
    }

    func trainerAvailable() async -> Bool {
        session?.trainer != nil
    }

    func trainerSnapshot() async throws -> LichessBotWeightsSnapshot {
        guard let session else { throw LichessBotSessionModelError.sessionGone }
        guard let trainer = session.trainer else { throw LichessBotModelError.noTrainer }
        guard let identifier = trainer.identifier else {
            throw LichessBotSessionModelError.missingModelID(source: "trainer")
        }
        let step = trainer.completedTrainSteps
        let weights = try await trainer.network.exportWeights()
        return LichessBotWeightsSnapshot(weights: weights, architecture: trainer.arch, modelID: "\(identifier)", trainingStep: step)
    }
}
