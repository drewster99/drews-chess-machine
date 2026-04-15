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
    /// The network emits raw policy logits (no softmax in the graph). For the
    /// UI demo we softmax over all 4096 slots once here so the displayed
    /// percentages are real probabilities. Self-play does not go through
    /// this path; it consumes the logits directly via MPSChessPlayer.
    ///
    /// `pieces` is the unflipped display board used to look up ghost-piece
    /// icons for each arrow. `flip` must match whatever `BoardEncoder.encode`
    /// did to produce `board`: true when the position was black-to-move, so
    /// the network's policy frame is vertically mirrored relative to the
    /// display. We un-mirror the extracted move coordinates here so arrows
    /// land on the right squares regardless of side-to-move.
    func evaluate(
        board: [Float],
        pieces: [Piece?],
        flip: Bool
    ) throws -> InferenceResult {
        let start = CFAbsoluteTimeGetCurrent()
        let (policyBuf, value) = try network.evaluate(board: board)
        // Copy the non-owning policy view into an owned array before
        // returning — the Forward Pass UI is the only consumer and is
        // cold path, so the one-time copy doesn't matter, and this
        // frees the next `evaluate` call to reuse the network's
        // scratch without invalidating our result.
        let logits = Array(policyBuf)
        let inferenceTimeMs = (CFAbsoluteTimeGetCurrent() - start) * 1000
        return Self.makeInferenceResult(
            logits: logits,
            value: value,
            pieces: pieces,
            flip: flip,
            inferenceTimeMs: inferenceTimeMs
        )
    }

    /// Package raw logits + value from any forward pass into the
    /// `InferenceResult` the UI consumes: softmaxes the logits, extracts
    /// the top-4 move visualizations (un-flipped when needed), and records
    /// the inference wall time. Shared between the inference-network path
    /// (`evaluate(board:pieces:flip:)`, which uses the frozen-BN inference
    /// network) and the Candidate test probe path in ContentView, which
    /// runs forward directly on the trainer's internal training-mode
    /// network so the user sees the candidate-in-training's opinion
    /// without any weight copy.
    static func makeInferenceResult(
        logits: [Float],
        value: Float,
        pieces: [Piece?],
        flip: Bool,
        inferenceTimeMs: Double
    ) -> InferenceResult {
        let policy = softmax(logits)
        return InferenceResult(
            topMoves: extractTopMoves(from: policy, pieces: pieces, flip: flip, count: 4),
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

    static func extractTopMoves(
        from policy: [Float],
        pieces: [Piece?],
        flip: Bool,
        count: Int
    ) -> [MoveVisualization] {
        let indexed = policy.indices.map { (index: $0, prob: policy[$0]) }
        return indexed
            .sorted { $0.prob > $1.prob }
            .prefix(count)
            .map { entry in
                // The policy index is `fromSquare * 64 + toSquare` in the
                // network's (possibly flipped) coordinate frame. We decode
                // into that frame first, then un-flip rows back to the
                // display frame so the arrows land on the right squares.
                let netFromRow = (entry.index / 64) / 8
                let netFromCol = (entry.index / 64) % 8
                let netToRow = (entry.index % 64) / 8
                let netToCol = (entry.index % 64) % 8
                let fromRow = flip ? (7 - netFromRow) : netFromRow
                let toRow = flip ? (7 - netToRow) : netToRow
                let piece = pieces[fromRow * 8 + netFromCol]
                return MoveVisualization(
                    fromRow: fromRow,
                    fromCol: netFromCol,
                    toRow: toRow,
                    toCol: netToCol,
                    probability: entry.prob,
                    piece: piece?.assetName
                )
            }
    }

    // MARK: - Result Type

    struct InferenceResult: Sendable {
        let topMoves: [MoveVisualization]
        let policy: [Float]
        let value: Float
        let inferenceTimeMs: Double
    }
}
