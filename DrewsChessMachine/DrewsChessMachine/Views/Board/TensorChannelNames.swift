import Foundation

/// Channel name lookup for the 20 input planes (post-arch-refresh —
/// added planes 18 and 19 for threefold-repetition signals).
enum TensorChannelNames {
    static let names = [
        "My Pawns", "My Knights", "My Bishops",
        "My Rooks", "My Queens", "My King",
        "Opp Pawns", "Opp Knights", "Opp Bishops",
        "Opp Rooks", "Opp Queens", "Opp King",
        "My Kingside Castle", "My Queenside Castle",
        "Opp Kingside Castle", "Opp Queenside Castle",
        "En Passant", "Halfmove Clock",
        "Repetition ≥1×", "Repetition ≥2×"
    ]

    /// Short labels for the strip thumbnails.
    static let shortNames = [
        "♙", "♘", "♗", "♖", "♕", "♔",
        "♟", "♞", "♝", "♜", "♛", "♚",
        "K-side", "Q-side", "K-side", "Q-side",
        "e.p.", "50-mv",
        "rep≥1", "rep≥2"
    ]
}
