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
/// Per-tile rendering: each cell's brightness is the channel's
/// logit at that square min-max normalized within the channel —
/// `(logit − channel_min) / (channel_max − channel_min)`. We
/// deliberately avoid softmax-based brightness here: early in
/// training the per-channel softmax is essentially uniform
/// (1/64 ≈ 1.56% per cell), and any per-tile peak normalization
/// of a uniform distribution produces solid-color tiles with no
/// per-cell variation. Min-max on the raw logits guarantees
/// visible per-cell structure as long as the channel has any
/// spread at all (always true post random init). The trade-off
/// is that absolute magnitude is lost from the visual — a barely-
/// trained channel and a sharp channel both render their peak at
/// full brightness — so the per-tile `peak %` (channel-softmax
/// concentration) and `mass %` (channel's share of the GLOBAL
/// 4864-cell distribution) are shown beneath each tile to keep
/// the magnitude story available textually.
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

    /// Brightness floor applied to every tile after per-channel
    /// normalization. Cells whose normalized value falls strictly
    /// below this threshold render with no red tint at all (the
    /// underlying chess-square color shows through). Slider in the
    /// header drives this; persisted across launches via
    /// `@AppStorage` so the user's preferred cut-off survives a
    /// session restart. 0 = show everything (default), 1 = show
    /// only cells at the per-channel max.
    @AppStorage("policyChannelsRedThreshold") private var redThreshold: Double = 0

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .firstTextBaseline, spacing: 12) {
                Text("Policy channels")
                    .font(.headline)
                Text(headerStatus)
                    .font(.system(.caption, design: .monospaced))
                    .foregroundStyle(.secondary)

                // Per-cell brightness floor. Drag right to hide
                // weak cells and reveal only the per-channel peaks
                // — at 1.0, only the brightest cell of each tile
                // remains red.
                HStack(spacing: 6) {
                    Text("min")
                        .font(.system(.caption, design: .monospaced))
                        .foregroundStyle(.secondary)
                    Slider(value: $redThreshold, in: 0...1)
                        .frame(width: 220)
                    Text(String(format: "%.2f", redThreshold))
                        .font(.system(.caption, design: .monospaced))
                        .foregroundStyle(.secondary)
                        .frame(width: 36, alignment: .trailing)
                }

                Spacer()
                Text("Per-channel logit min-max → cell brightness = (logit − channel_min) / (channel_max − channel_min). peak% = per-channel softmax concentration; mass% = channel's share of total policy mass.")
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
                overlay: .channel(thresholded(ch.cellValues)),
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

    /// Apply the per-cell brightness floor. Cells whose normalized
    /// value is strictly below `redThreshold` are zeroed out so
    /// `ChessBoardView`'s `.channel` overlay (which short-circuits
    /// at `value > 0.001`) skips drawing them entirely. At
    /// `redThreshold == 0` we hand back the input unchanged so the
    /// allocation is skipped on the no-cutoff fast path.
    private func thresholded(_ values: [Float]) -> [Float] {
        guard redThreshold > 0 else { return values }
        let t = Float(redThreshold)
        return values.map { $0 >= t ? $0 : 0 }
    }

    // MARK: - Channel computation

    fileprivate struct ChannelData: Identifiable {
        let id: Int
        let cellValues: [Float]   // 64 entries in absolute (display) coords, [0, 1] from logit min-max
        let peakProb: Float       // per-channel softmax peak probability (concentration indicator)
        let globalMass: Float     // sum of GLOBAL softmax probs in this channel
        let label: String
    }

    /// Decompose the 4864 raw logits into 76 displayable channel
    /// tiles. Three independent quantities are computed per channel:
    ///
    /// 1. **`cellValues`** — the actual heatmap. Each cell's
    ///    brightness is the logit at that square min-max normalized
    ///    within the channel: `(logit − chMin) / (chMax − chMin)`.
    ///    Always uses the full [0, 1] range as long as the channel
    ///    has any logit spread (degenerate `chMax == chMin` collapses
    ///    to all-zero). Crucially, this does NOT depend on the
    ///    softmax flatness — early-training near-uniform softmax
    ///    distributions still produce meaningful per-cell variation
    ///    because the underlying logits always have *some* spread
    ///    (random init alone is enough). Trade-off: absolute
    ///    magnitude is lost from the visual; we surface it via the
    ///    peak/mass labels below.
    /// 2. **`peakProb`** — max probability of the per-channel softmax
    ///    over the 64 cells. Uniform = 1/64 ≈ 1.56%, perfectly
    ///    concentrated = 1.0. Tells the user how spiky the channel
    ///    actually is, independent of the visual brightness pattern.
    /// 3. **`globalMass`** — channel's share of the global softmax
    ///    over all 4864 cells. Uniform across channels = 1/76 ≈ 1.32%.
    ///    Tells the user how much the channel matters overall.
    ///
    /// Cells are transposed from encoder frame to absolute board
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

        // --- Per-channel passes -------------------------------------
        // Smallest meaningful logit span. Below this we treat the
        // channel as degenerate (all-equal) and render it as all-zero
        // — anything finer is denormalized noise that would just
        // amplify into arbitrary bright patterns under min-max.
        let degenerateSpanThreshold: Float = 1e-6
        let flip = currentPlayer == .black
        var result: [ChannelData] = []
        result.reserveCapacity(PolicyEncoding.channelCount)
        for chan in 0..<PolicyEncoding.channelCount {
            let base = chan * 64

            // Per-channel logit min/max (drives the visualization).
            var chMin: Float = .infinity
            var chMax: Float = -.infinity
            for i in 0..<64 {
                let l = logits[base + i]
                if l < chMin { chMin = l }
                if l > chMax { chMax = l }
            }
            let span = chMax - chMin
            let invSpan: Float = span > degenerateSpanThreshold ? 1 / span : 0

            // Per-channel softmax (drives the `peak %` label only).
            var pkSum: Float = 0
            var pkProbs = [Float](repeating: 0, count: 64)
            for i in 0..<64 {
                let e = expf(logits[base + i] - chMax)
                pkProbs[i] = e
                pkSum += e
            }
            let pkInv: Float = pkSum > 0 ? 1 / pkSum : 0
            var peakProb: Float = 0
            for i in 0..<64 {
                pkProbs[i] *= pkInv
                if pkProbs[i] > peakProb { peakProb = pkProbs[i] }
            }

            // Build the display-frame heatmap (logit min-max) and
            // accumulate the global mass in one pass.
            var cells = [Float](repeating: 0, count: 64)
            var globalMass: Float = 0
            for r in 0..<8 {
                let displayRow = flip ? (7 - r) : r
                for c in 0..<8 {
                    let i = r * 8 + c
                    cells[displayRow * 8 + c] = (logits[base + i] - chMin) * invSpan
                    globalMass += globalProbs[base + i]
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
