import SwiftUI

/// Hover overlay rendered to the right of the main chess board when
/// the user mouses over a square. Shows a horizontal row of three
/// mini-board tiles, one per top-3 channel (ranked by per-channel
/// logit at the hovered from-square). Each tile draws the channel's
/// geometric move as a green arrow from the hovered square plus a
/// label with the channel name and softmax probability.
///
/// Replaces `MainTextPanel` while hovering — the parent gates the
/// swap on (cursor-over-board AND inference result has raw logits).
/// Restores `MainTextPanel` on un-hover.
///
/// All math is over the per-channel softmax (each tile's slice sums
/// to 1) for the displayed prob, but the *ranking* of channels
/// is by raw logit at the hovered square. Logits and softmax
/// agree on ordering at a fixed cell since softmax is monotonic;
/// using logits avoids re-computing per-channel softmax on every
/// hover tick.
struct HoverPolicyOverlay: View {
    /// Hovered square in absolute (display) coordinates: row 0 =
    /// rank 8, row 7 = rank 1.
    let hoveredRow: Int
    let hoveredCol: Int
    /// Side to move. Drives the encoder-frame row flip when
    /// computing per-channel logit indices and the geometric-move
    /// decoder for arrow placement.
    let currentPlayer: PieceColor
    /// Pieces on the displayed board (absolute coords). Used as the
    /// background for each tile so the arrow lands on the right
    /// physical square.
    let pieces: [Piece?]
    /// Raw 4864 logits. The ranking source.
    let policyLogits: [Float]
    /// Optional pre-softmaxed policy. When provided we use it to
    /// display the global softmax probability of each top channel's
    /// hovered cell — that's the most directly comparable number to
    /// the upper Policy Head top-K display. nil → fall back to the
    /// raw logit value as the display number.
    let policyProbs: [Float]?

    /// Top-3 channels for the hovered square, sorted by logit
    /// descending. Cached as `@State` so we recompute only when one
    /// of the inputs actually changes — not on every parent
    /// re-render. The 76-channel sort + two geometric decodes per
    /// channel is sub-millisecond, but the typical case is a parent
    /// re-render where none of those inputs moved (e.g. an unrelated
    /// SwiftUI state change), so caching is the SwiftUI-correct shape.
    ///
    /// We watch `policyLogits.count` to catch the inference-completed
    /// transition (0 → `policySize`) and `policyLogits.first` as a
    /// cheap content fingerprint to catch the case where a new
    /// inference produces a fresh array of the same length while the
    /// cursor stays over the board (different position → different
    /// logit[0] in any realistic scenario). Watching the full 4864-
    /// element array would be far more work than the recompute we're
    /// avoiding.
    @State private var topChannels: [TopChannel] = []

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(alignment: .firstTextBaseline) {
                Text("Top 3 channels at \(squareName(hoveredRow, hoveredCol))")
                    .font(.headline)
                Spacer()
                Text("\(currentPlayer == .white ? "White" : "Black") to move")
                    .font(.system(.caption, design: .monospaced))
                    .foregroundStyle(.secondary)
            }
            HStack(alignment: .top, spacing: 12) {
                ForEach(topChannels) { ch in
                    tile(ch)
                }
            }
        }
        .padding(8)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .onAppear { topChannels = computeTopChannels() }
        .onChange(of: hoveredRow) { _, _ in topChannels = computeTopChannels() }
        .onChange(of: hoveredCol) { _, _ in topChannels = computeTopChannels() }
        .onChange(of: currentPlayer) { _, _ in topChannels = computeTopChannels() }
        .onChange(of: policyLogits.count) { _, _ in topChannels = computeTopChannels() }
        .onChange(of: policyLogits.first) { _, _ in topChannels = computeTopChannels() }
    }

    private func computeTopChannels() -> [TopChannel] {
        // Defensive: parent gates on `policyLogits.count == policySize`,
        // but @State recomputes can fire mid-transition (e.g. between
        // an inference clear and the next result). Bail before the
        // loop when the array isn't sized for the full channel sweep.
        let required = PolicyEncoding.channelCount * 64
        guard policyLogits.count >= required else { return [] }
        let flip = currentPlayer == .black
        let encoderRow = flip ? (7 - hoveredRow) : hoveredRow
        // Build (channel, logit) pairs for every channel that's
        // geometrically valid from this from-square.
        var pairs: [(chan: Int, logit: Float, idx: Int)] = []
        pairs.reserveCapacity(PolicyEncoding.channelCount)
        for chan in 0..<PolicyEncoding.channelCount {
            let idx = chan * 64 + encoderRow * 8 + hoveredCol
            let logit = policyLogits[idx]
            // Skip channels whose geometric move is off-board from
            // this square — a north-7 move from rank 7 would land
            // on rank 14, which is meaningless to draw an arrow for.
            if PolicyEncoding.geometricDecode(
                channel: chan,
                row: encoderRow,
                col: hoveredCol,
                currentPlayer: currentPlayer
            ) == nil {
                continue
            }
            pairs.append((chan, logit, idx))
        }
        let top = pairs
            .sorted { $0.logit > $1.logit }
            .prefix(3)
        return top.map { p in
            let move = PolicyEncoding.geometricDecode(
                channel: p.chan,
                row: encoderRow,
                col: hoveredCol,
                currentPlayer: currentPlayer
            )
            let prob: Float? = policyProbs.map { probs in
                p.idx < probs.count ? probs[p.idx] : 0
            }
            return TopChannel(
                channel: p.chan,
                logit: p.logit,
                prob: prob,
                move: move,
                label: HoverPolicyOverlay.channelLabel(for: p.chan)
            )
        }
    }

    @ViewBuilder
    private func tile(_ ch: TopChannel) -> some View {
        VStack(alignment: .center, spacing: 4) {
            // Human-readable move-type title above the board.
            // Matches what a chess player would say at the board:
            // "Northeast, 3 squares" / "Knight" / "Promote to Queen"
            // / "Promote to Rook". The encoder-frame canonical name
            // (`32 S5` etc.) lives below the board with the stats.
            Text(Self.humanReadableTitle(for: ch.channel))
                .font(.system(.subheadline))
                .frame(maxWidth: .infinity, alignment: .center)
            ChessBoardView(
                pieces: pieces,
                overlay: arrowOverlay(for: ch)
            )
            .aspectRatio(1, contentMode: .fit)
            .frame(maxWidth: .infinity)
            .clipShape(RoundedRectangle(cornerRadius: 4))
            .overlay(
                RoundedRectangle(cornerRadius: 4)
                    .stroke(Color.gray.opacity(0.3), lineWidth: 0.5)
            )
            // Channel index + spec name + this cell's stats below
            // the board. Centered horizontally to match the title;
            // sized closer to the title font (.callout / .caption)
            // and at .secondary contrast — the previous 9pt/8pt
            // tertiary was too small to read at the new tile size.
            Text(ch.label)
                .font(.system(.callout, design: .monospaced))
                .lineLimit(1)
                .frame(maxWidth: .infinity, alignment: .center)
            Text(probLabel(ch))
                .font(.system(.callout, design: .monospaced))
                .frame(maxWidth: .infinity, alignment: .center)
        }
    }

    private func probLabel(_ ch: TopChannel) -> String {
        if let p = ch.prob {
            return String(format: "logit %+.3f · prob %.3f%%", ch.logit, p * 100)
        }
        return String(format: "logit %+.3f", ch.logit)
    }

    /// Friendly, plain-English description of a channel's move type
    /// for the title above each tile. The arrow on the mini-board
    /// already shows the geometric direction visually, so the title
    /// stays high-level — direction + distance for queen-style,
    /// just "Knight" for knight channels, and the standard chess
    /// "Promote to {Piece}" wording for promotions.
    fileprivate static func humanReadableTitle(for channel: Int) -> String {
        if channel < 56 {
            let dirs = ["North", "Northeast", "East", "Southeast",
                        "South", "Southwest", "West", "Northwest"]
            let dirIdx = channel / 7
            let dist = channel % 7 + 1
            let plural = dist == 1 ? "" : "s"
            return "\(dirs[dirIdx]), \(dist) square\(plural)"
        }
        if channel < 64 {
            return "Knight"
        }
        if channel < 73 {
            let pieces = ["Knight", "Rook", "Bishop"]
            let off = channel - 64
            return "Promote to \(pieces[off / 3])"
        }
        return "Promote to Queen"
    }

    /// Build a `.topMoves` overlay containing exactly one entry —
    /// the arrow for this channel's geometric move from the
    /// hovered square. `MoveVisualization.probability` is set to
    /// 1.0 so the arrow renders at full saturation regardless of
    /// the channel's actual mass; the textual `probLabel` carries
    /// the magnitude information.
    private func arrowOverlay(for ch: TopChannel) -> ChessBoardView.Overlay {
        guard let move = ch.move else {
            return .none
        }
        let piece = pieces[move.fromRow * 8 + move.fromCol]
        return .topMoves([
            MoveVisualization(
                fromRow: move.fromRow,
                fromCol: move.fromCol,
                toRow: move.toRow,
                toCol: move.toCol,
                probability: 1.0,
                piece: piece?.assetName,
                isLegal: true,
                promotion: move.promotion
            )
        ])
    }

    private func squareName(_ row: Int, _ col: Int) -> String {
        let files: [Character] = ["a", "b", "c", "d", "e", "f", "g", "h"]
        let rank = 8 - row
        return "\(files[col])\(rank)"
    }

    fileprivate struct TopChannel: Identifiable {
        let channel: Int
        let logit: Float
        let prob: Float?
        let move: ChessMove?
        let label: String
        var id: Int { channel }
    }

    /// Channel-name decoder. Mirrors `PolicyChannelsPanel.label(for:)`
    /// but lifted here so the two views can reuse the same naming
    /// without circular module dependencies. Format: "<idx> <name>"
    /// where name is the spec's encoder-frame canonical string
    /// (queen-style direction × distance, knight L-shape, or
    /// promotion piece × direction).
    fileprivate static func channelLabel(for channel: Int) -> String {
        let queenDirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        let knightDirs = ["UR", "RU", "RD", "DR", "DL", "LD", "LU", "UL"]
        let promoDirs = ["F", "CL", "CR"]
        let underpromo = ["uN", "uR", "uB"]
        if channel < 56 {
            return "\(channel) \(queenDirs[channel / 7])\(channel % 7 + 1)"
        }
        if channel < 64 {
            return "\(channel) Kn-\(knightDirs[channel - 56])"
        }
        if channel < 73 {
            let off = channel - 64
            return "\(channel) \(underpromo[off / 3])-\(promoDirs[off % 3])"
        }
        return "\(channel) QP-\(promoDirs[channel - 73])"
    }
}
