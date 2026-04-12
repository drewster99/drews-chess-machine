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
    func evaluate(board: [Float]) throws -> InferenceResult {
        let start = CFAbsoluteTimeGetCurrent()
        let (logits, value) = try network.evaluate(board: board)
        let inferenceTimeMs = (CFAbsoluteTimeGetCurrent() - start) * 1000

        let policy = Self.softmax(logits)
        return InferenceResult(
            topMoves: Self.extractTopMoves(from: policy, count: 4),
            policy: policy,
            value: value,
            inferenceTimeMs: inferenceTimeMs
        )
    }

    /// Numerically stable softmax over the full vector.
    private static func softmax(_ logits: [Float]) -> [Float] {
        guard let maxLogit = logits.max() else { return logits }
        var out = logits.map { expf($0 - maxLogit) }
        let sum = out.reduce(0, +)
        if sum > 0 {
            for i in out.indices { out[i] /= sum }
        }
        return out
    }

    // MARK: - Move Extraction

    private static func extractTopMoves(from policy: [Float], count: Int) -> [MoveVisualization] {
        let indexed = policy.indices.map { (index: $0, prob: policy[$0]) }
        return indexed
            .sorted { $0.prob > $1.prob }
            .prefix(count)
            .map { entry in
                let fromSquare = entry.index / 64
                let toSquare = entry.index % 64
                let fromRow = fromSquare / 8
                let fromCol = fromSquare % 8
                return MoveVisualization(
                    fromRow: fromRow,
                    fromCol: fromCol,
                    toRow: toSquare / 8,
                    toCol: toSquare % 8,
                    probability: entry.prob,
                    piece: BoardEncoder.startingPieces[fromSquare]
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
