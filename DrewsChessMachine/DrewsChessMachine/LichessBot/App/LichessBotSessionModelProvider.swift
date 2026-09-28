import Foundation

enum LichessBotSessionModelError: LocalizedError, Equatable {
    case sessionGone
    case missingModelID(source: String)
    case trainerChangedDuringExport

    var errorDescription: String? {
        switch self {
        case .sessionGone:
            return "The training session is not available"
        case .missingModelID(let source):
            return "The \(source) network has no ModelID, so games played with it could not be attributed"
        case .trainerChangedDuringExport:
            return "The trainer's weights were being replaced (a promotion, reset or load) during the export, so they can't be attributed; the next snapshot will be"
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
        // A promotion, reset or load replaces the weights before it stamps
        // the new identity. One in progress, or one that overlapped the
        // export, leaves the weights unattributable.
        let before = trainer.weightIdentityState
        guard !before.awaitingIdentity else {
            throw LichessBotSessionModelError.trainerChangedDuringExport
        }
        let export = try await trainer.exportWeightsWithCompletedSteps()
        guard trainer.weightIdentityState == before, trainer.identifier == identifier else {
            throw LichessBotSessionModelError.trainerChangedDuringExport
        }
        return LichessBotWeightsSnapshot(weights: export.weights, architecture: trainer.arch, modelID: "\(identifier)", trainingStep: export.completedSteps)
    }
}
