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
    let network: ChessNetwork

    /// Time taken to build the graph and initialize weights, in milliseconds.
    let buildTimeMs: Double

    /// Optional stable identity assigned externally by the UI layer.
    /// See `ModelID` and `sampling-parameters.md` for the mint /
    /// inherit rules. Mutated at well-defined checkpoint events
    /// (Build, arena snapshot, promotion). Nil until a caller sets it.
    var identifier: ModelID?

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
    ///
    /// **Not re-entrant.** The returned policy buffer aliases the
    /// underlying `ChessNetwork`'s shared readback scratch and is valid
    /// only until the next `evaluate` call on this wrapper. See
    /// `ChessNetwork.evaluate(board:)` for the full contract.
    ///
    /// - Parameter board: `inputPlanes`×8×8 = 1,280 floats (from `BoardEncoder.encode`).
    /// - Returns: `policySize` (4,864) raw policy logits and the scalar value head.
    func evaluate(
        board: UnsafeBufferPointer<Float>
    ) async throws -> (policy: [Float], value: Float) {
        try await network.evaluate(board: board)
    }

    /// `[Float]`-input overload for non-hot-path callers (the Forward
    /// Pass demo, tests). Delegates to the pointer-based primary entry
    /// point — no copy on `.float32`.
    func evaluate(
        board: [Float]
    ) async throws -> (policy: [Float], value: Float) {
        try await network.evaluate(board: board)
    }

    /// Run a batched forward pass.
    ///
    /// **Not re-entrant.** Both returned buffers alias the underlying
    /// `ChessNetwork`'s shared batched readback scratch and are valid
    /// only until the next batched `evaluate` call on this wrapper. See
    /// `ChessNetwork.evaluate(batchBoards:count:)` for the full contract.
    ///
    /// - Parameters:
    ///   - batchBoards: `count * BoardEncoder.tensorLength` floats (currently
    ///                  `count * 1280`), NCHW order, positions laid
    ///                  out back-to-back.
    ///   - count: batch size; must be >= 1.
    /// - Returns: `policy` — `count * policySize` logits; `values` — `count`
    ///            scalars in [-1, +1].
    func evaluate(
        batchBoards: UnsafeBufferPointer<Float>,
        count: Int
    ) async throws -> (policy: [Float], values: [Float]) {
        try await network.evaluate(batchBoards: batchBoards, count: count)
    }

    func evaluate(
        batchBoards: [Float],
        count: Int
    ) async throws -> (policy: [Float], values: [Float]) {
        try await network.evaluate(batchBoards: batchBoards, count: count)
    }

    func exportWeights() async throws -> [[Float]] {
        try await network.exportWeights()
    }

    func loadWeights(_ weights: [[Float]]) async throws {
        try await network.loadWeights(weights)
    }
}
