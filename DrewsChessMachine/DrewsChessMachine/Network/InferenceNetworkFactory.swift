import Foundation

/// Builds stand-alone inference `ChessMPSNetwork`s for callers that play
/// moves on a private copy of some weights — human play's opponent slots and
/// the Lichess bot's model slots.
///
/// Building a network constructs a full MPSGraph, which is long synchronous
/// work. It runs on a GCD global queue and re-enters Swift concurrency
/// through a continuation, so it never occupies a cooperative-pool thread
/// (a `Task.detached` still runs on that pool).
enum InferenceNetworkFactory {

    /// A `.randomWeights` network of `arch`: untrained weights from a freshly
    /// drawn init seed, usable at once.
    static func build(arch: NetworkArchitecture) async throws -> ChessMPSNetwork {
        try await build(mode: .randomWeights(initSeed: WeightInitialization.drawnInitSeed()), arch: arch)
    }

    /// An `.overwrittenByLoad` network of `arch`: for a mirror whose weights
    /// are overwritten before every use. It draws no weights and refuses to
    /// evaluate until the first load.
    static func buildAwaitingLoad(arch: NetworkArchitecture) async throws -> ChessMPSNetwork {
        try await build(mode: .overwrittenByLoad, arch: arch)
    }

    /// A network of `arch` carrying `weights` (the layout `exportWeights()`
    /// produces and `loadWeights(_:)` accepts).
    static func build(loading weights: [[Float]], arch: NetworkArchitecture) async throws -> ChessMPSNetwork {
        let network = try await buildAwaitingLoad(arch: arch)
        try await network.loadWeights(weights)
        return network
    }

    private static func build(mode: NetworkInitMode, arch: NetworkArchitecture) async throws -> ChessMPSNetwork {
        try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    continuation.resume(returning: try ChessMPSNetwork(mode, arch: arch))
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }
}
