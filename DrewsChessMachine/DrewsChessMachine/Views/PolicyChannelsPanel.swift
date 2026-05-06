import SwiftUI

/// Inline panel that decomposes the network's 4864 policy logits
/// into the 76 AlphaZero-shape output channels and renders each as
/// an 8×8 mini-board overlay on the same position currently shown
/// on the main chess board.
///
/// Driven entirely off the existing `inferenceResult` pipeline: in
/// Candidate Test mode the trainer's network is auto-re-evaluated
/// on the editable board every time the trainer's weights change
/// (the existing Probe loop), so this panel updates "live" as the
/// network learns — without any new self-play hook, snapshot box,
/// or background eval loop. In pure Forward Pass mode it updates
/// only when the user clicks Run Forward Pass (or edits the board,
/// which auto-triggers a re-eval).
///
/// Per-tile rendering: each tile shows softmax over THAT channel's
/// own 64 cells (so the 64 cells of one tile sum to 1), then
/// normalized to the per-tile max so the channel's brightest
/// from-square renders fully opaque. This means every tile is
/// independently scaled — a channel with tiny global mass still
/// shows where ITS preferred from-squares are. The tile's
/// per-channel max probability and the channel's GLOBAL share of
/// total policy mass are both shown beneath the board so the user
/// can read both halves of the story (where would this channel
/// fire AND does this channel matter overall).
///
/// Sections mirror `PolicyEncoding`'s blocks so co-located tiles
/// share semantics:
///   - 0..55  queen-style (8 directions × 7 distances)
///   - 56..63 knight (8 jumps)
///   - 64..72 underpromotion (3 pieces × 3 directions)
///   - 73..75 queen-promotion (3 directions)
struct PolicyChannelsPanel: View {
    /// Board pieces in absolute coordinates (row 0 = rank 8). Same
    /// array fed to the parent `ChessBoardView`, so each tile shows
    /// the same arrangement and brightness lines up visually with
    /// the actual squares.
    let pieces: [Piece?]
    /// Side to move for the position the policy was evaluated at.
    /// Drives the encoder-frame un-flip when decoding logits to
    /// per-square brightness for black-to-move positions.
    let currentPlayer: PieceColor
    /// 4864 raw logits from the network's policy head. Pass nil to
    /// render an empty placeholder (no inference result available).
    let policyLogits: [Float]?

    /// Tile color for the channel-cell overlay. Red so the
    /// channel-grid is visually distinct from the blue input-tensor
    /// channel strip rendered elsewhere in the UI.
    private static let tileTint: Color = .red

    /// Adaptive grid sizing — `.adaptive(minimum:)` lets SwiftUI
    /// pack as many tiles per row as the available width allows
    /// while keeping each tile at least this wide. The full-area
    /// layout (panel taking over the chart pane) gets many columns;
    /// any narrower layout falls back to fewer.
    private static let minTileSide: CGFloat = 96

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .firstTextBaseline) {
                Text("Policy channels")
                    .font(.headline)
                Text(headerStatus)
                    .font(.system(.caption, design: .monospaced))
                    .foregroundStyle(.secondary)
                Spacer()
                Text("Per-channel softmax (each tile sums to 1) → per-tile normalize. Brightness = relative within the channel. mass% = channel's share of total policy mass.")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.trailing)
                    .frame(maxWidth: 540, alignment: .trailing)
            }

            if let logits = policyLogits, logits.count == ChessNetwork.policySize {
                let chans = Self.computeChannels(
                    from: logits,
                    currentPlayer: currentPlayer
                )
                ScrollView {
                    VStack(alignment: .leading, spacing: 14) {
                        section(
                            title: "Queen-style (8 directions × 7 distances)",
                            channels: Array(chans[0..<56])
                        )
                        section(
                            title: "Knight (8 jumps)",
                            channels: Array(chans[56..<64])
                        )
                        section(
                            title: "Underpromotion (knight / rook / bishop × forward / cap-left / cap-right)",
                            channels: Array(chans[64..<73])
                        )
                        section(
                            title: "Queen-promotion (forward / cap-left / cap-right)",
                            channels: Array(chans[73..<76])
                        )
                    }
                    .padding(.bottom, 8)
                }
            } else {
                Spacer()
                Text("No inference result.\nSwitch to Candidate Test, or click Run Forward Pass.")
                    .font(.body)
                    .foregroundStyle(.tertiary)
                    .multilineTextAlignment(.center)
                    .frame(maxWidth: .infinity)
                Spacer()
            }
        }
        .padding(8)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var headerStatus: String {
        guard let logits = policyLogits, logits.count == ChessNetwork.policySize else {
            return ""
        }
        let side = currentPlayer == .white ? "White" : "Black"
        return "— \(side) to move"
    }

    @ViewBuilder
    private func section(title: String, channels: [ChannelData]) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title)
                .font(.subheadline)
                .foregroundStyle(.secondary)
            let gridColumns = [
                GridItem(.adaptive(minimum: Self.minTileSide), spacing: 6, alignment: .top)
            ]
            LazyVGrid(columns: gridColumns, alignment: .leading, spacing: 8) {
                ForEach(channels) { ch in
                    channelTile(ch)
                }
            }
        }
    }

    @ViewBuilder
    private func channelTile(_ ch: ChannelData) -> some View {
        VStack(spacing: 1) {
            ChessBoardView(
                pieces: pieces,
                overlay: .channel(ch.cellValues),
                channelColor: Self.tileTint
            )
            .aspectRatio(1, contentMode: .fit)
            .clipShape(RoundedRectangle(cornerRadius: 3))
            .overlay(
                RoundedRectangle(cornerRadius: 3)
                    .stroke(Color.gray.opacity(0.3), lineWidth: 0.5)
            )
            Text(ch.label)
                .font(.system(size: 9, design: .monospaced))
                .lineLimit(1)
            // peak% = brightest cell's per-channel softmax probability;
            // mass% = channel's share of the global policy distribution.
            // Two distinct signals: peak says "how concentrated is the
            // pick within this channel" and mass says "how much is the
            // network actually betting on this channel as a whole".
            Text(String(format: "peak %.1f%% · mass %.2f%%",
                        ch.peakProb * 100, ch.globalMass * 100))
                .font(.system(size: 8, design: .monospaced))
                .foregroundStyle(.tertiary)
        }
    }

    // MARK: - Channel computation

    fileprivate struct ChannelData: Identifiable {
        let id: Int
        let cellValues: [Float]   // 64 entries in absolute (display) coordinates, scaled to [0,1] via per-channel max
        let peakProb: Float       // max per-channel softmax probability (= max cellValues pre-normalization)
        let globalMass: Float     // sum of GLOBAL softmax probs in this channel
        let label: String
    }

    /// Decompose the 4864 raw logits into 76 displayable channel
    /// tiles. Two normalizations happen here:
    ///
    /// 1. **Global softmax** over all 4864 cells, summed per channel,
    ///    gives `globalMass` — "what fraction of total policy
    ///    probability does this channel hold." Useful as a relevance
    ///    indicator: dominant channels show high mass, dead ones near
    ///    zero.
    /// 2. **Per-channel softmax** over each channel's 64 cells gives a
    ///    proper [0, 1] distribution INSIDE that channel ("if I pick
    ///    this channel, what's the spatial preference?"). Then divide
    ///    by the channel's max so the brightest cell renders fully
    ///    opaque — every tile auto-scales for visibility, even
    ///    channels with tiny global mass still show where their
    ///    preferred from-squares are.
    ///
    /// Cells are also transposed from encoder frame to absolute board
    /// coordinates (row un-flip) when black is to move, so every tile
    /// renders right-side-up versus the parent board's pieces.
    fileprivate static func computeChannels(
        from logits: [Float],
        currentPlayer: PieceColor
    ) -> [ChannelData] {
        let n = logits.count
        precondition(n == ChessNetwork.policySize,
                     "PolicyChannelsPanel expects \(ChessNetwork.policySize) logits, got \(n)")

        // --- Global softmax pass (for per-channel mass) -------------
        var maxLogit: Float = -.infinity
        for i in 0..<n where logits[i] > maxLogit {
            maxLogit = logits[i]
        }
        var globalProbs = [Float](repeating: 0, count: n)
        var globalSum: Float = 0
        for i in 0..<n {
            let e = expf(logits[i] - maxLogit)
            globalProbs[i] = e
            globalSum += e
        }
        let invGlobalSum: Float = globalSum > 0 ? 1 / globalSum : 0
        for i in 0..<n {
            globalProbs[i] *= invGlobalSum
        }

        // --- Per-channel softmax + display-frame transpose ----------
        let flip = currentPlayer == .black
        var result: [ChannelData] = []
        result.reserveCapacity(PolicyEncoding.channelCount)
        for chan in 0..<PolicyEncoding.channelCount {
            let base = chan * 64

            // Stable softmax over JUST this channel's 64 cells.
            var maxIn: Float = -.infinity
            for i in 0..<64 where logits[base + i] > maxIn {
                maxIn = logits[base + i]
            }
            var probs = [Float](repeating: 0, count: 64)
            var sum: Float = 0
            for i in 0..<64 {
                let e = expf(logits[base + i] - maxIn)
                probs[i] = e
                sum += e
            }
            let invSum: Float = sum > 0 ? 1 / sum : 0
            var peakProb: Float = 0
            for i in 0..<64 {
                probs[i] *= invSum
                if probs[i] > peakProb { peakProb = probs[i] }
            }
            let normInv: Float = peakProb > 0 ? 1 / peakProb : 0

            // Transpose encoder rows to absolute (display) rows, scale
            // to [0, 1] by the per-channel peak, and accumulate the
            // global mass while we're walking the channel.
            var cells = [Float](repeating: 0, count: 64)
            var globalMass: Float = 0
            for r in 0..<8 {
                let displayRow = flip ? (7 - r) : r
                for c in 0..<8 {
                    cells[displayRow * 8 + c] = probs[r * 8 + c] * normInv
                    globalMass += globalProbs[base + r * 8 + c]
                }
            }
            result.append(ChannelData(
                id: chan,
                cellValues: cells,
                peakProb: peakProb,
                globalMass: globalMass,
                label: Self.label(for: chan)
            ))
        }
        return result
    }

    // MARK: - Channel labels

    /// Direction order for queen-style: N, NE, E, SE, S, SW, W, NW.
    /// Mirrors `PolicyEncoding.queenDirections`.
    private static let queenDirNames = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    /// Knight-jump labels in `PolicyEncoding.knightJumps` order.
    private static let knightLabels = ["UR", "RU", "RD", "DR", "DL", "LD", "LU", "UL"]
    /// Promotion direction: forward / capture-left / capture-right.
    private static let promoDirLabels = ["F", "CL", "CR"]
    /// Underpromotion piece prefixes in `PolicyEncoding`'s order.
    private static let underpromoPieceLabels = ["uN", "uR", "uB"]

    fileprivate static func label(for channel: Int) -> String {
        if channel < 56 {
            let dir = channel / 7
            let dist = channel % 7 + 1
            return "\(channel) \(queenDirNames[dir])\(dist)"
        }
        if channel < 64 {
            return "\(channel) \(knightLabels[channel - 56])"
        }
        if channel < 73 {
            let off = channel - 64
            let piece = off / 3
            let dir = off % 3
            return "\(channel) \(underpromoPieceLabels[piece])-\(promoDirLabels[dir])"
        }
        let dir = channel - 73
        return "\(channel) QP-\(promoDirLabels[dir])"
    }
}
