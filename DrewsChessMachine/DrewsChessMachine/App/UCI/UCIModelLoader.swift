import Foundation

/// Resolves the weight checkpoint to use for a UCI session and loads
/// it into a fresh inference network.
///
/// Two entry shapes:
///
/// - `loadExplicit(modelFileURL:)` — caller passed `--model <path>`.
///   The path must name a `.dcmmodel` file (champion or trainer
///   variant; we slice off optimizer velocity if present). Session
///   directories are deliberately NOT accepted here — a `.dcmsession`
///   has two `.dcmmodel` files inside and the user can name whichever
///   they want directly.
///
/// - `loadDefault()` — no `--model` flag. Resolves to the most
///   recently saved session via `LastSessionPointer`, then loads
///   that session's `trainer.dcmmodel` (the trainee, which is
///   generally the most up-to-date weights). Hard-errors if no
///   pointer exists or the pointed-to directory is missing — the
///   caller is the UCI pre-flight, which prints the error and exits.
enum UCIModelLoader {

    enum LoadError: Swift.Error, CustomStringConvertible {
        case noSavedSession
        case sessionDirectoryMissing(URL)
        case modelFileMissing(URL)
        case explicitMustBeDcmmodel(URL)
        case weightCountTooSmall(have: Int, need: Int)

        var description: String {
            switch self {
            case .noSavedSession:
                return "No saved session found. Use --model <path.dcmmodel> or save a session first."
            case .sessionDirectoryMissing(let url):
                return "The most recent session directory is missing on disk: \(url.path)"
            case .modelFileMissing(let url):
                return "Model file does not exist: \(url.path)"
            case .explicitMustBeDcmmodel(let url):
                return "--model expects a .dcmmodel file; got: \(url.path) (session directories are not accepted)"
            case .weightCountTooSmall(let have, let need):
                return "Model file has \(have) tensors but the network needs at least \(need)"
            }
        }
    }

    /// Description of a successfully loaded model — used by the UCI
    /// engine to populate the `id name` handshake line and the
    /// session-log banner.
    struct Loaded {
        let network: ChessMPSNetwork
        let modelID: String
        let sourceLabel: String
    }

    /// Resolve and load using the precedence: explicit `--model` if
    /// given, otherwise the latest session's trainer file.
    static func resolveAndLoad(explicitPath: String?) async throws -> Loaded {
        if let path = explicitPath {
            return try await loadExplicit(path: path)
        }
        return try await loadDefault()
    }

    private static func loadExplicit(path: String) async throws -> Loaded {
        let expanded = (path as NSString).expandingTildeInPath
        let url = URL(fileURLWithPath: expanded)
        guard url.pathExtension.lowercased() == "dcmmodel" else {
            throw LoadError.explicitMustBeDcmmodel(url)
        }
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw LoadError.modelFileMissing(url)
        }
        let file = try CheckpointManager.loadModelFile(at: url)
        let network = try await buildAndLoad(weights: file.weights)
        return Loaded(
            network: network,
            modelID: file.modelID,
            sourceLabel: "--model \(url.path)"
        )
    }

    private static func loadDefault() async throws -> Loaded {
        guard let pointer = LastSessionPointer.read() else {
            throw LoadError.noSavedSession
        }
        let dir = pointer.directoryURL
        guard pointer.directoryExists else {
            throw LoadError.sessionDirectoryMissing(dir)
        }
        let trainerURL = SessionCheckpointLayout.trainerURL(in: dir)
        guard FileManager.default.fileExists(atPath: trainerURL.path) else {
            throw LoadError.modelFileMissing(trainerURL)
        }
        let file = try CheckpointManager.loadModelFile(at: trainerURL)
        let network = try await buildAndLoad(weights: file.weights)
        return Loaded(
            network: network,
            modelID: file.modelID,
            sourceLabel: "session \(pointer.sessionID) trainer"
        )
    }

    /// Build a fresh `ChessMPSNetwork` (inference BN mode) and load the
    /// supplied weight tensors into it. Trainer-file weights include
    /// optimizer velocity buffers after the base trainables + BN
    /// running stats; UCI inference only needs the base block, so we
    /// take the leading prefix of length
    /// `trainableVariables.count + bnRunningStatsVariables.count`.
    /// Champion-file weights are exactly that length already, so the
    /// prefix is a no-op for them.
    private static func buildAndLoad(weights: [[Float]]) async throws -> ChessMPSNetwork {
        let network = try ChessMPSNetwork(.randomWeights)
        let baseCount = network.network.trainableVariables.count
            + network.network.bnRunningStatsVariables.count
        guard weights.count >= baseCount else {
            throw LoadError.weightCountTooSmall(have: weights.count, need: baseCount)
        }
        let baseWeights = Array(weights.prefix(baseCount))
        try await network.network.loadWeights(baseWeights)
        return network
    }
}
