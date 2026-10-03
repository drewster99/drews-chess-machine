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

    /// An `.overwrittenByLoad` network of `arch`, with no weights loaded. For a
    /// mirror whose weights are overwritten before every use.
    static func build(arch: NetworkArchitecture) async throws -> ChessMPSNetwork {
        try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    continuation.resume(returning: try ChessMPSNetwork(.overwrittenByLoad, arch: arch))
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    /// A network of `arch` carrying `weights` (the layout `exportWeights()`
    /// produces and `loadWeights(_:)` accepts).
    static func build(loading weights: [[Float]], arch: NetworkArchitecture) async throws -> ChessMPSNetwork {
        let network = try await build(arch: arch)
        try await network.loadWeights(weights)
        return network
    }
}
