import Foundation

/// Channel name lookup for the 30 input planes. Planes 0–15 are pieces +
/// castling, 16 en passant, 17 halfmove clock, 18–19 threefold-repetition
/// counts (≥1× / ≥2× before), and 20–29 the 10-ply repetition temporal
/// pattern — plane `20 + i` is all-1 iff the position `i + 1` plies ago is
/// a strict chess-rules duplicate of the current position.
///
/// Must stay in lockstep with `ChessNetwork.inputPlanes` — `count` is
/// asserted equal to `ChessNetwork.inputPlanes` at app launch so a drift
/// fails loudly instead of trapping later inside SwiftUI's layout pass.
enum TensorChannelNames {
    static let names = [
        "My Pawns", "My Knights", "My Bishops",
        "My Rooks", "My Queens", "My King",
        "Opp Pawns", "Opp Knights", "Opp Bishops",
        "Opp Rooks", "Opp Queens", "Opp King",
        "My Kingside Castle", "My Queenside Castle",
        "Opp Kingside Castle", "Opp Queenside Castle",
        "En Passant", "Halfmove Clock",
        "Repetition ≥1×", "Repetition ≥2×",
        "Repetition Same as −1 ply",
        "Repetition Same as −2 plies",
        "Repetition Same as −3 plies",
        "Repetition Same as −4 plies",
        "Repetition Same as −5 plies",
        "Repetition Same as −6 plies",
        "Repetition Same as −7 plies",
        "Repetition Same as −8 plies",
        "Repetition Same as −9 plies",
        "Repetition Same as −10 plies"
    ]

    /// Short labels for the strip thumbnails.
    static let shortNames = [
        "♙", "♘", "♗", "♖", "♕", "♔",
        "♟", "♞", "♝", "♜", "♛", "♚",
        "K-side", "Q-side", "K-side", "Q-side",
        "e.p.", "50-mv",
        "rep≥1", "rep≥2",
        "rep−1", "rep−2", "rep−3", "rep−4", "rep−5",
        "rep−6", "rep−7", "rep−8", "rep−9", "rep−10"
    ]
}
