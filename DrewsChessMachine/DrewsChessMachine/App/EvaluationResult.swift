import Foundation

struct EvaluationResult: Sendable {
    let topMoves: [MoveVisualization]
    let textOutput: String
    let inputTensor: [Float]
    /// Full forward-pass result surfaced alongside the formatted
    /// text for the CLI recorder. The on-screen UI only reads
    /// `topMoves` and `textOutput`; the recorder needs the raw
    /// policy vector to extract a top-10 list and the full set of
    /// policy stats without re-parsing the formatted string.
    /// Nil when the forward pass threw — in that case the probe
    /// retains its prior on-screen state and the recorder simply
    /// skips the event rather than logging a half-populated entry.
    let rawInference: ChessRunner.InferenceResult?
}
