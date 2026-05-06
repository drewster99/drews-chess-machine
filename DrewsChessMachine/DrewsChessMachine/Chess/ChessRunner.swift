import Foundation

// MARK: - Chess Runner

/// UI-facing wrapper around ChessMPSNetwork. Adds timing and move extraction
/// for the forward pass demo in ContentView.
///
/// Marked @unchecked Sendable — access serialized via disabled UI buttons.
final class ChessRunner: @unchecked Sendable {
    private let network: ChessMPSNetwork

    init(network: ChessMPSNetwork) {
        self.network = network
    }

    /// Run the forward pass on a board position.
    ///
    /// The network emits raw policy logits (no softmax in the graph). For
    /// the UI demo we softmax over all `policySize` (4864) slots once here
    /// so the displayed percentages are real probabilities. Self-play does
    /// not go through this path; it consumes the logits directly via
    /// MPSChessPlayer.
    ///
    /// `pieces` is the unflipped display board used to look up ghost-piece
    /// icons for each arrow. `state` is the source GameState the board was
    /// encoded from; `PolicyEncoding.decode` uses it both to interpret the
    /// (channel, row, col) cells back into absolute coordinates AND to
    /// filter to the legal subset (so absurd top-K cells like "knight jump
    /// from a square with no piece" don't appear as ghost arrows).
    func evaluate(
        board: [Float],
        state: GameState,
        pieces: [Piece?]
    ) async throws -> InferenceResult {
        let start = CFAbsoluteTimeGetCurrent()
        let (logits, value) = try await network.evaluate(board: board)
        let inferenceTimeMs = (CFAbsoluteTimeGetCurrent() - start) * 1000
        return Self.makeInferenceResult(
            logits: logits,
            value: value,
            state: state,
            pieces: pieces,
            inferenceTimeMs: inferenceTimeMs
        )
    }

    /// Package raw logits + value from any forward pass into the
    /// `InferenceResult` the UI consumes: softmaxes the logits, extracts
    /// the top-4 LEGAL move visualizations, and records the inference
    /// wall time. Shared between the inference-network path
    /// (`evaluate(board:state:pieces:)`, which uses the frozen-BN inference
    /// network) and the Candidate test probe path in ContentView, which
    /// runs forward directly on the trainer's internal training-mode
    /// network so the user sees the candidate-in-training's opinion
    /// without any weight copy.
    static func makeInferenceResult(
        logits: [Float],
        value: Float,
        state: GameState,
        pieces: [Piece?],
        inferenceTimeMs: Double
    ) -> InferenceResult {
        let policy = softmax(logits)
        return InferenceResult(
            topMoves: extractTopMoves(from: policy, state: state, pieces: pieces, count: 4),
            logits: logits,
            policy: policy,
            value: value,
            inferenceTimeMs: inferenceTimeMs
        )
    }

    /// Numerically stable softmax over the full vector.
    static func softmax(_ logits: [Float]) -> [Float] {
        guard let maxLogit = logits.max() else { return logits }
        var out = logits.map { expf($0 - maxLogit) }
        let sum = out.reduce(0, +)
        if sum > 0 {
            for i in out.indices { out[i] /= sum }
        }
        return out
    }

    // MARK: - Move Extraction

    /// Extract the top-K policy cells by raw probability — INCLUDING
    /// illegal candidates. This is the diagnostic for "is the network
    /// learning what's a valid move?" — illegal-but-geometrically-
    /// valid candidates surfacing in the top-K signal that the policy
    /// hasn't yet learned to suppress them.
    ///
    /// Iterates the full `policy` vector rather than the legal-move
    /// subset so cells the network favors but that aren't legal here
    /// still appear (with `isLegal: false` on the resulting
    /// `MoveVisualization`). Off-board geometry → cell skipped.
    /// Geometric-decode result legality is checked once via
    /// `MoveGenerator.legalMoves(for: state)`.
    ///
    /// Cost: O(policySize) for the sort, O(legalMoves) once for the
    /// legality lookup set. Trivial for a 4864-cell policy.
    static func extractTopMoves(
        from policy: [Float],
        state: GameState,
        pieces: [Piece?],
        count: Int
    ) -> [MoveVisualization] {
        // Pre-compute a hashable set of legal moves so per-cell
        // legality lookup is O(1) instead of O(legalMoves).
        let legalSet = Set(MoveGenerator.legalMoves(for: state))

        // Sort ALL policy cells by probability descending. The prior
        // implementation took only `count * 4` as headroom under the
        // assumption that few top cells would decode to off-board
        // geometry. That breaks for catastrophically collapsed
        // policies where a single off-board cell holds ~100% of the
        // probability mass (observed with policy collapse) — in that
        // case the result list came back empty or near-empty because
        // the top N cells were all off-board and we ran out of
        // headroom. Sorting the full 4864-cell vector costs nothing
        // (policy-size array, one pass) and guarantees we'll always
        // find `count` on-board cells if they exist in the
        // distribution at all.
        let topByProb = policy.indices
            .map { (index: $0, prob: policy[$0]) }
            .sorted { $0.prob > $1.prob }

        var results: [MoveVisualization] = []
        results.reserveCapacity(count)
        for entry in topByProb {
            if results.count >= count { break }
            let chan = entry.index / 64
            let row = (entry.index % 64) / 8
            let col = entry.index % 8
            guard let candidate = PolicyEncoding.geometricDecode(
                channel: chan, row: row, col: col,
                currentPlayer: state.currentPlayer
            ) else { continue }  // off-board — skip
            let isLegal = legalSet.contains(candidate)
            let piece = pieces[candidate.fromRow * 8 + candidate.fromCol]
            results.append(MoveVisualization(
                fromRow: candidate.fromRow,
                fromCol: candidate.fromCol,
                toRow: candidate.toRow,
                toCol: candidate.toCol,
                probability: entry.prob,
                piece: piece?.assetName,
                isLegal: isLegal
            ))
        }
        return results
    }

    // MARK: - Result Type

    struct InferenceResult: Sendable {
        let topMoves: [MoveVisualization]
        /// Pre-softmax raw policy logits (length `policySize`). Kept
        /// alongside `policy` so consumers that want the unnormalized
        /// per-channel landscape (e.g. the policy-channels panel's
        /// per-channel min-max heatmap) don't have to log-transform
        /// the softmaxed values back into logit-space — log() of
        /// near-zero probabilities loses precision and goes to -inf
        /// where the softmax floored. Same source vector that gets
        /// softmaxed into `policy` below.
        let logits: [Float]
        /// Softmax-over-all-`policySize`-cells of `logits`. The UI's
        /// top-K display, the per-cell percentage readouts, and the
        /// "Top 100 sum" stats all consume this — they all want
        /// real probabilities, not raw logits.
        let policy: [Float]
        let value: Float
        let inferenceTimeMs: Double
    }
}
