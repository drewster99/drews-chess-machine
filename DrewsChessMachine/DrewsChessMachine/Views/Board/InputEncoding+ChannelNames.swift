import Foundation

/// Per-plane channel names for the board / forward-pass visualizers, derived
/// from the encoding itself.
///
/// This replaces the old standalone `TensorChannelNames` list, which was
/// hardcoded to exactly the 30 planes of the *default* arch — redundant with
/// `planeGroups`, prone to drift, and a crash waiting to happen the moment a
/// model had more than 30 planes (full10ply200 = 200, full10Ply10Reps210 =
/// 210). Deriving the names here makes the encoding the single source of
/// truth: `channelNames.count == planeCount` for EVERY encoding, present or
/// future, so the UI can never index past the end.
///
/// The base names describe one `basic30` block. Multi-frame encodings repeat
/// the 20-plane `basic20` sub-block once per stacked frame (frame-tagged), then
/// append any non-stacked tail (e.g. full10Ply10Reps210's 10 repetition planes
/// = the `basic30` temporal-repetition planes 20–29).
extension InputEncoding {

    /// Full human-readable name per plane. `count == planeCount`.
    var channelNames: [String] {
        compose(base: Self.basic30ChannelNames) { frame, name in
            frame == 0 ? "[ply N] \(name)" : "[ply N-\(frame)] \(name)"
        }
    }

    /// Compact per-plane label for the strip thumbnails. `count == planeCount`.
    var shortChannelNames: [String] {
        compose(base: Self.basic30ShortChannelNames) { frame, name in
            frame == 0 ? name : "−\(frame) \(name)"
        }
    }

    private func compose(
        base base30: [String],
        frameTag: (_ frame: Int, _ name: String) -> String
    ) -> [String] {
        switch self {
        case .basic20:
            return Array(base30.prefix(20))
        case .basic30:
            return base30
        case .full10ply200, .full10Ply10Reps210:
            let frameBlock = Array(base30.prefix(20))   // the basic20 sub-block
            var out: [String] = []
            out.reserveCapacity(planeCount)
            for f in 0..<historyFrameCount {
                out += frameBlock.map { frameTag(f, $0) }
            }
            if tailPlaneCount > 0 {
                // Non-stacked tail = the basic30 temporal-repetition planes 20…29.
                out += Array(base30[20..<(20 + tailPlaneCount)])
            }
            return out
        }
    }

    private static let basic30ChannelNames = [
        "My Pawns", "My Knights", "My Bishops",
        "My Rooks", "My Queens", "My King",
        "Opp Pawns", "Opp Knights", "Opp Bishops",
        "Opp Rooks", "Opp Queens", "Opp King",
        "My Kingside Castle", "My Queenside Castle",
        "Opp Kingside Castle", "Opp Queenside Castle",
        "En Passant", "Halfmove Clock",
        "Repetition ≥1×", "Repetition ≥2×",
        "Repetition Same as −1 ply", "Repetition Same as −2 plies",
        "Repetition Same as −3 plies", "Repetition Same as −4 plies",
        "Repetition Same as −5 plies", "Repetition Same as −6 plies",
        "Repetition Same as −7 plies", "Repetition Same as −8 plies",
        "Repetition Same as −9 plies", "Repetition Same as −10 plies",
    ]

    private static let basic30ShortChannelNames = [
        "♙", "♘", "♗", "♖", "♕", "♔",
        "♟", "♞", "♝", "♜", "♛", "♚",
        "K-side", "Q-side", "K-side", "Q-side",
        "e.p.", "50-mv",
        "rep≥1", "rep≥2",
        "rep−1", "rep−2", "rep−3", "rep−4", "rep−5",
        "rep−6", "rep−7", "rep−8", "rep−9", "rep−10",
    ]
}
