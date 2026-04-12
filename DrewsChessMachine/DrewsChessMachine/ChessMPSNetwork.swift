import Foundation
import Metal
import MetalPerformanceShadersGraph

// MARK: - Errors

enum ChessMPSNetworkError: LocalizedError {
    case packageLoadingNotImplemented

    var errorDescription: String? {
        switch self {
        case .packageLoadingNotImplemented:
            return "Loading from MPSGraph package is not yet implemented"
        }
    }
}

// MARK: - Init Mode

/// How to initialize the network weights.
enum NetworkInitMode {
    /// He-initialized random weights — untrained network.
    case randomWeights
    /// Load a previously serialized MPSGraphExecutable package.
    case package(URL)
}

// MARK: - Chess MPS Network

/// Wraps the neural network graph for chess position evaluation.
///
/// Provides a clean interface over ChessNetwork (MPSGraph construction + inference).
/// Can be initialized with random weights for training or loaded from a serialized package.
///
/// Marked @unchecked Sendable — Metal objects aren't Sendable, but access is
/// serialized (players take turns, one evaluation at a time).
final class ChessMPSNetwork: @unchecked Sendable {
    private let network: ChessNetwork

    /// Time taken to build the graph and initialize weights, in milliseconds.
    let buildTimeMs: Double

    /// Create a network with the specified initialization mode.
    init(_ mode: NetworkInitMode) throws {
        let start = CFAbsoluteTimeGetCurrent()

        switch mode {
        case .randomWeights:
            network = try ChessNetwork()

        case .package:
            // Deserializing a saved MPSGraphExecutable is tracked in
            // ROADMAP.md ("Compiled MPSGraphExecutable"); not implemented
            // yet. Steps once we get there:
            //   1. Load MPSGraphExecutable from the package URL
            //   2. Use executable.run() instead of graph.run() for inference
            throw ChessMPSNetworkError.packageLoadingNotImplemented
        }

        buildTimeMs = (CFAbsoluteTimeGetCurrent() - start) * 1000
    }

    /// Run the forward pass on a board tensor.
    /// - Parameter board: 18×8×8 = 1,152 floats (from BoardEncoder.encode)
    /// - Returns: Policy probabilities (4096 values) and position value in [-1, +1]
    func evaluate(board: [Float]) throws -> (policy: [Float], value: Float) {
        try network.evaluate(board: board)
    }
}
