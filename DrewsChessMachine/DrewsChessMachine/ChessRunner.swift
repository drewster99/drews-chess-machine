import Foundation

// MARK: - Errors

enum ChessRunnerError: LocalizedError {
    case networkNotBuilt

    var errorDescription: String? {
        switch self {
        case .networkNotBuilt:
            return "Network has not been built — call buildNetwork() first"
        }
    }
}

// MARK: - Chess Runner

/// UI-facing wrapper around ChessMPSNetwork. Adds timing and move extraction
/// for the forward pass demo in ContentView.
///
/// Marked @unchecked Sendable — access serialized via disabled UI buttons.
final class ChessRunner: @unchecked Sendable {
    private var network: ChessMPSNetwork?
    private(set) var networkBuildTimeMs: Double = 0

    var isReady: Bool { network != nil }

    /// Build a new network with random weights.
    func buildNetwork() throws {
        let net = try ChessMPSNetwork(.randomWeights)
        networkBuildTimeMs = net.buildTimeMs
        network = net
    }

    /// Run the forward pass on a board position.
    func evaluate(board: [Float]) throws -> InferenceResult {
        guard let network else {
            throw ChessRunnerError.networkNotBuilt
        }

        let start = CFAbsoluteTimeGetCurrent()
        let (policy, value) = try network.evaluate(board: board)
        let inferenceTimeMs = (CFAbsoluteTimeGetCurrent() - start) * 1000

        return InferenceResult(
            topMoves: Self.extractTopMoves(from: policy, count: 4),
            policy: policy,
            value: value,
            inferenceTimeMs: inferenceTimeMs
        )
    }

    // MARK: - Move Extraction

    private static func extractTopMoves(from policy: [Float], count: Int) -> [MoveVisualization] {
        policy.enumerated()
            .sorted { $0.element > $1.element }
            .prefix(count)
            .map { index, prob in
                let fromSquare = index / 64
                let toSquare = index % 64
                let fromRow = fromSquare / 8
                let fromCol = fromSquare % 8
                return MoveVisualization(
                    fromRow: fromRow,
                    fromCol: fromCol,
                    toRow: toSquare / 8,
                    toCol: toSquare % 8,
                    probability: prob,
                    piece: BoardEncoder.startingPieces[fromRow][fromCol]
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
