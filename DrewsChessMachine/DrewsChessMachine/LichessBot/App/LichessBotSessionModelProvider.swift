import Foundation

enum LichessBotSessionModelError: LocalizedError, Equatable {
    case sessionGone
    case missingModelID(source: String)
    case trainerChangedDuringExport
    case championChangedDuringExport

    var errorDescription: String? {
        switch self {
        case .sessionGone:
            return "The training session is not available"
        case .missingModelID(let source):
            return "The \(source) network has no ModelID, so games played with it could not be attributed"
        case .trainerChangedDuringExport:
            return "The trainer's weights were being replaced (a promotion, reset or load) during the export, so they can't be attributed; the next snapshot will be"
        case .championChangedDuringExport:
            return "The champion's weights were being replaced (a promotion or load) during the export, or the last replacement failed, so they can't be attributed until a replacement completes"
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
        // A promotion or load replaces the champion's weights before it
        // stamps the new identity and records their origin
        // (`SessionController.championWeightIdentity`). One in progress, or
        // one that overlapped the export, would label the new weights with
        // the old model ID — and the model ID alone can't tell, since every
        // checkpoint of a run shares it.
        let before = session.championWeightIdentity
        guard !before.awaitingOrigin else {
            throw LichessBotSessionModelError.championChangedDuringExport
        }
        let weights = try await champion.exportWeights()
        guard session.network === champion, session.championWeightIdentity == before, champion.identifier == identifier else {
            throw LichessBotSessionModelError.championChangedDuringExport
        }
        return LichessBotWeightsSnapshot(weights: weights, architecture: champion.arch, modelID: "\(identifier)", trainingStep: nil)
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
