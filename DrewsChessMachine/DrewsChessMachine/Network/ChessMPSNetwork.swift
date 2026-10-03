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
    /// He-initialized random weights — untrained network. `initSeed` is the
    /// model's init seed: the parent of its initialization streams, starting
    /// with the batch-norm calibration walk
    /// (`DCMRandomStreams.batchNormCalibrationGenerator(initSeed:)`). A
    /// network whose weights and batch-norm statistics are replaced by a
    /// load right after it is built (a load container) passes
    /// `RunRandomSeed.systemDrawnSeed()`: nothing of its initialization
    /// survives.
    case randomWeights(initSeed: UInt64)

    /// A container whose weights and batch-norm statistics are replaced by a
    /// load before first use (inference mirrors, arena and probe networks,
    /// checkpoint verification, a run continuing from a model file). Nothing
    /// of its initialization survives the load, so it is built from a
    /// system-drawn init seed. (Per-tensor initialization will make this a
    /// mode that skips initialization altogether.)
    static var overwrittenByLoad: NetworkInitMode {
        .randomWeights(initSeed: RunRandomSeed.systemDrawnSeed())
    }
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

    /// The architecture this network was built from. Convenience for the
    /// common `someMPSNet.network.arch` hop. Source of the input encoding
    /// every board fed to this network must be encoded with.
    var arch: NetworkArchitecture { network.arch }

    /// Input encoding this network expects — every `BoardEncoder.encode`
    /// feeding it must use this so the tensor width matches the stem.
    var inputEncoding: InputEncoding { network.arch.inputEncoding }

    /// Time taken to build the graph and initialize weights, in milliseconds.
    let buildTimeMs: Double

    /// Optional stable identity assigned externally by the UI layer.
    /// See `ModelID` and `sampling-parameters.md` for the mint /
    /// inherit rules. Mutated at well-defined checkpoint events
    /// (Build, arena snapshot, promotion). Nil until a caller sets it.
    var identifier: ModelID?

    /// Create a network with the specified initialization mode.
    ///
    /// **`.randomWeights` includes a one-shot BN warmup pass.** A
    /// freshly-He-init `ChessNetwork(bnMode: .inference)` has BN
    /// running stats at (0, 1), which makes the BN op effectively
    /// identity — and that is *not* what a deep residual+SE tower
    /// needs to keep activation variance bounded. Without the warmup
    /// the first inference pass produces a softmax dominated by a
    /// single randomly-chosen channel, sending self-play games into
    /// near-uniform legal-move sampling and starving the trainer of
    /// signal. The warmup runs ONE batched forward through a sibling
    /// training-mode network on a varied batch of chess positions
    /// (random walk from `.starting`), reads out per-BN-layer
    /// `batch_mean` / `batch_var`, and writes them into this network's
    /// `running_mean` / `running_var`. End result: BN actually
    /// normalizes from the very first forward pass.
    ///
    /// The training-mode sibling and the calibration batch both live
    /// only for the duration of this initializer — once warmup is done
    /// they're released. Cost is one extra graph build plus one extra
    /// batched forward, ~10s of ms; happens exactly once per fresh
    /// random-init network.
    ///
    /// `.package` (not yet implemented) would skip warmup — a loaded
    /// package already carries trained running stats. Sites that
    /// build a `.randomWeights` container and immediately call
    /// `loadWeights(_:)` on it (candidate inference network, arena
    /// snapshot, checkpoint verification scratch) pay the warmup cost
    /// even though the loaded weights overwrite it; the cost is
    /// negligible and the alternative — multiple init modes one of
    /// which is silently broken until `loadWeights` is called — is
    /// the kind of footgun this engine has spent debugging effort on
    /// before.
    init(_ mode: NetworkInitMode, arch: NetworkArchitecture = .current) throws {
        let start = CFAbsoluteTimeGetCurrent()

        switch mode {
        case .randomWeights(let initSeed):
            let net = try ChessNetwork(arch: arch)
            net.commandQueue.label = "ChessMPSNetwork.net(init)"
            try Self.calibrateBNRunningStats(
                into: net,
                random: DCMRandomStreams.batchNormCalibrationGenerator(initSeed: initSeed)
            )
            network = net

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

    /// Number of plies in the BN warmup batch. ~64 plies × 64 spatial
    /// cells = 4096 samples per channel, plenty for stable batch-stat
    /// estimation. Larger batches give marginal gains at proportional
    /// cost.
    private static let warmupBatchSize: Int = 64

    /// Synthesize a varied batch of encoded chess positions for BN
    /// warmup. Walks one random game from `.starting`, each move a uniform
    /// pick from `random` over `MoveGenerator.legalMoves` (whose order is
    /// pinned by test), restarting on terminal positions, until the batch is
    /// full. Outputs are concatenated `BoardEncoder.encode` tensors in the
    /// position-major NCHW layout the network expects. A function of the
    /// encoding and `random`'s state alone.
    static func warmupBatch(encoding: InputEncoding, random: inout DCMRandom) -> [Float] {
        let perBoard = BoardEncoder.tensorLength(for: encoding)
        // Prior frames a history-stacking encoding needs (0 for single-frame).
        let historyDepth = max(0, encoding.historyFrameCount - 1)
        var out = [Float](repeating: 0, count: warmupBatchSize * perBoard)
        var state = GameState.starting
        // Rolling window of recent prior states (most-recent-first), mirroring
        // ChessGameEngine.recentStates so warmup inputs carry realistic history
        // for history-stacking encodings. Empty for single-frame encodings, so
        // their warmup is byte-identical to before. Cleared on game reset.
        var history: [GameState] = []
        var ply = 0
        out.withUnsafeMutableBufferPointer { buf in
            // `out` is `warmupBatchSize * perBoard` floats, statically
            // non-empty; a nil baseAddress here means the array's
            // backing storage is broken.
            guard let base = buf.baseAddress else {
                preconditionFailure(
                    "warmupBatch: out buffer baseAddress is nil "
                    + "(count=\(buf.count)); upstream invariant violated."
                )
            }
            while ply < warmupBatchSize {
                let slot = UnsafeMutableBufferPointer<Float>(
                    start: base.advanced(by: ply * perBoard),
                    count: perBoard
                )
                BoardEncoder.encode(state, history: history, into: slot, encoding: encoding)
                ply += 1
                let legalMoves = MoveGenerator.legalMoves(for: state)
                guard !legalMoves.isEmpty else {
                    // No legal moves (mate/stalemate) — reset to the opening
                    // and keep generating warmup positions. New game → no history.
                    state = .starting
                    history.removeAll(keepingCapacity: true)
                    continue
                }
                let move = legalMoves[random.nextBounded(legalMoves.count)]
                if historyDepth > 0 {
                    history.insert(state, at: 0)
                    if history.count > historyDepth {
                        history.removeLast(history.count - historyDepth)
                    }
                }
                state = MoveGenerator.applyMove(move, to: state)
            }
        }
        return out
    }

    /// Build a sibling training-mode network sharing `inference`'s
    /// trainable weights, run one batched forward on the warmup batch,
    /// read out per-layer batch_mean / batch_var, and write them into
    /// `inference`'s BN running stats. Synchronous wrapper around the
    /// async export/compute/load primitives — called from the
    /// non-async `init(_:)` so the network is fully calibrated by the
    /// time it's handed back to the caller.
    private static func calibrateBNRunningStats(into inference: ChessNetwork, random: DCMRandom) throws {
        // Build a parallel training-mode network. Its trainable weights
        // are independently He-init at first; we copy `inference`'s in
        // so the batch stats reflect the inference network's actual
        // weight realization (different random seed → different
        // activation distribution → different needed running stats).
        let trainingNet = try ChessNetwork(arch: inference.arch, bnMode: .training)
        trainingNet.commandQueue.label = "calibrateBNRunningStats trainingNet"
        var walkRandom = random
        let boards = warmupBatch(encoding: inference.arch.inputEncoding, random: &walkRandom)

        // Bridge async → sync via a semaphore + a Sendable holder for
        // the caught error. The work runs on each network's private
        // execution queue, so we cannot deadlock on the caller's
        // queue; the semaphore just blocks the calling thread until
        // the chained operations complete. The Task writes the error
        // box exactly once (before signaling) and the calling thread
        // reads it exactly once (after waiting), so the explicit
        // happens-before of the semaphore is the only synchronization
        // needed.
        final class ErrorBox: @unchecked Sendable {
            nonisolated(unsafe) var value: Error?
        }
        let errorBox = ErrorBox()
        let semaphore = DispatchSemaphore(value: 0)
        // Match the priority of the typical caller (network builds run
        // from user-initiated UI flows or test runners) so the
        // semaphore wait below isn't a priority-inversion landmine.
        Task.detached(priority: .userInitiated) { [boards, trainingNet, inference] in
            do {
                let weights = try await inference.exportWeights()
                try await trainingNet.loadWeights(weights)
                let stats = try await trainingNet.computeBatchStats(
                    boards: boards,
                    count: warmupBatchSize
                )
                try await inference.loadBNRunningStats(
                    means: stats.means,
                    vars: stats.vars
                )
            } catch {
                errorBox.value = error
            }
            semaphore.signal()
        }
        semaphore.wait()
        if let caught = errorBox.value {
            throw caught
        }
    }

    /// Run the forward pass on a board tensor and hand the policy/value
    /// readback to `consume` synchronously, inside the underlying
    /// `ChessNetwork`'s `executionQueue` work block. See
    /// `ChessNetwork.evaluate(board:consume:)` for the full contract
    /// (closure-validity window, non-throwing requirement).
    ///
    /// - Parameter board: `inputPlanes`×8×8 = 1,920 floats (from `BoardEncoder.encode`).
    /// - Parameter consume: receives `policySize` (4,864) raw policy logits and the derived scalar value `p_win − p_loss ∈ [−1, +1]` (the W/D/L head's softmax · `[+1, 0, −1]`). The full `(p_win, p_draw, p_loss)` distribution is not carried by this closure — call `evaluateValueDistribution(board:)` for that.
    func evaluate(
        board: [Float],
        consume: @Sendable @escaping (UnsafeBufferPointer<Float>, Float) -> Void
    ) async throws {
        try await network.evaluate(board: board, consume: consume)
    }

    /// Run the forward pass on a board tensor and hand `consume` the policy
    /// logits plus the value head's full `(p_win, p_draw, p_loss)` — one
    /// forward pass, same as `evaluate(board:consume:)`. Passthrough to
    /// `ChessNetwork.evaluateWithValueDistribution(board:consume:)`; same
    /// closure-validity contract.
    func evaluateWithValueDistribution(
        board: [Float],
        consume: @Sendable @escaping (UnsafeBufferPointer<Float>, (win: Float, draw: Float, loss: Float)) -> Void
    ) async throws {
        try await network.evaluateWithValueDistribution(board: board, consume: consume)
    }

    /// Forward-only pass returning the value head's W/D/L softmax
    /// `(p_win, p_draw, p_loss)` for a single position — passthrough to
    /// `ChessNetwork.evaluateValueDistribution(board:)`. For diagnostics
    /// (the candidate-test probe / Run Forward Pass panel); the hot
    /// inference paths use the scalar via `evaluate(board:consume:)`.
    func evaluateValueDistribution(board: [Float]) async throws -> (win: Float, draw: Float, loss: Float) {
        try await network.evaluateValueDistribution(board: board)
    }

    /// Run a batched forward pass and hand the policy / value / W-D-L
    /// readback to `consume` synchronously, inside the underlying
    /// `ChessNetwork`'s `executionQueue` work block. See
    /// `ChessNetwork.evaluateBatched(batchBoards:count:consume:)` for
    /// the full contract.
    ///
    /// - Parameters:
    ///   - batchBoards: `count * BoardEncoder.tensorLength` floats,
    ///                  NCHW order, positions laid out back-to-back.
    ///   - count: batch size; must be >= 1.
    ///   - consume: receives `policy` (`count * policySize` logits,
    ///              position-major), `values` (`count` scalars in
    ///              [-1, +1]), and `wdlProbs` (`count * 3` softmax
    ///              probabilities, position-major, slot order
    ///              `[win, draw, loss]`). Callers that don't need
    ///              `wdlProbs` ignore it via `{ _, _, _ in ... }`.
    func evaluateBatched(
        batchBoards: [Float],
        count: Int,
        consume: @Sendable @escaping (
            UnsafeBufferPointer<Float>,
            UnsafeBufferPointer<Float>,
            UnsafeBufferPointer<Float>
        ) -> Void
    ) async throws {
        try await network.evaluateBatched(batchBoards: batchBoards, count: count, consume: consume)
    }

    /// Pointer-flavored sibling of the `[Float]` overload above. The
    /// tick-driver self-play path holds its batched-input scratch as
    /// an `UnsafeMutablePointer<Float>` field and calls this method
    /// directly per tick, avoiding the per-fire `[Float]` allocation
    /// the `[Float]` overload's `withUnsafeBufferPointer` round-trip
    /// would otherwise require. See
    /// `ChessNetwork.evaluateBatched(batchBoardsPointer:floatCount:count:consume:)`
    /// for the lifetime contract and Sendable handling.
    func evaluateBatched(
        batchBoardsPointer: UnsafePointer<Float>,
        floatCount: Int,
        count: Int,
        consume: @Sendable @escaping (
            UnsafeBufferPointer<Float>,
            UnsafeBufferPointer<Float>,
            UnsafeBufferPointer<Float>
        ) -> Void
    ) async throws {
        try await network.evaluateBatched(
            batchBoardsPointer: batchBoardsPointer,
            floatCount: floatCount,
            count: count,
            consume: consume
        )
    }

    func exportWeights() async throws -> [[Float]] {
        try await network.exportWeights()
    }

    func loadWeights(_ weights: [[Float]]) async throws {
        try await network.loadWeights(weights)
    }
}
