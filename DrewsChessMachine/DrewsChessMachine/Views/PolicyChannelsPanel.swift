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
/// One tile per channel:
///   - Mini chess board with the same piece arrangement as the
///     parent board (so the user can mentally co-locate "this
///     channel firing from a knight square" with the actual knight).
///   - `.channel` overlay: per-square brightness = global softmax
///     probability of (this channel × this from-square) within the
///     full 4864-cell distribution, normalized to the global max so
///     the brightest cell anywhere in the panel is fully opaque and
///     dead channels visibly stay dim.
///   - Channel index + spec name (e.g. "0 N1", "57 RU", "64 uN-F",
///     "73 QP-F") and per-channel total mass% so the user can spot
///     dominant channels at a glance.
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

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("Policy channels")
                    .font(.headline)
                Spacer()
                Text(headerStatus)
                    .font(.system(.caption, design: .monospaced))
                    .foregroundStyle(.secondary)
            }
            Text("Per channel: cell brightness = global softmax probability of channel × from-square (normalized to overall max). Mass% = total prob in the channel.")
                .font(.caption2)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)

            if let logits = policyLogits, logits.count == ChessNetwork.policySize {
                ScrollView {
                    let chans = Self.computeChannels(
                        from: logits,
                        currentPlayer: currentPlayer
                    )
                    VStack(alignment: .leading, spacing: 12) {
                        section(
                            title: "Queen-style",
                            channels: Array(chans[0..<56]),
                            columns: 7
                        )
                        section(
                            title: "Knight",
                            channels: Array(chans[56..<64]),
                            columns: 4
                        )
                        section(
                            title: "Underpromotion",
                            channels: Array(chans[64..<73]),
                            columns: 3
                        )
                        section(
                            title: "Queen-promotion",
                            channels: Array(chans[73..<76]),
                            columns: 3
                        )
                    }
                    .padding(.bottom, 8)
                }
            } else {
                Spacer()
                Text("No inference result.\nSwitch to Candidate Test, or click Run Forward Pass.")
                    .font(.caption)
                    .foregroundStyle(.tertiary)
                    .multilineTextAlignment(.center)
                    .frame(maxWidth: .infinity)
                Spacer()
            }
        }
        .padding(8)
        .frame(minWidth: 360, idealWidth: 380, maxWidth: 440)
    }

    private var headerStatus: String {
        guard let logits = policyLogits, logits.count == ChessNetwork.policySize else {
            return ""
        }
        let side = currentPlayer == .white ? "White" : "Black"
        return "\(side) to move"
    }

    @ViewBuilder
    private func section(
        title: String,
        channels: [ChannelData],
        columns: Int
    ) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title)
                .font(.subheadline)
                .foregroundStyle(.secondary)
            let gridColumns = Array(
                repeating: GridItem(.flexible(), spacing: 4, alignment: .top),
                count: columns
            )
            LazyVGrid(columns: gridColumns, alignment: .leading, spacing: 6) {
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
                overlay: .channel(ch.cellValues)
            )
            .aspectRatio(1, contentMode: .fit)
            .clipShape(RoundedRectangle(cornerRadius: 2))
            .overlay(
                RoundedRectangle(cornerRadius: 2)
                    .stroke(Color.gray.opacity(0.25), lineWidth: 0.5)
            )
            Text(ch.label)
                .font(.system(size: 8, design: .monospaced))
                .lineLimit(1)
            Text(String(format: "%.2f%%", ch.totalMass * 100))
                .font(.system(size: 7, design: .monospaced))
                .foregroundStyle(.tertiary)
        }
    }

    // MARK: - Channel computation

    fileprivate struct ChannelData: Identifiable {
        let id: Int
        let cellValues: [Float]   // 64 entries in absolute (display) coordinates, normalized to [0,1] vs global max
        let totalMass: Float      // sum of softmax probs in this channel
        let label: String
    }

    /// One global softmax pass + per-channel slicing, returning all
    /// 76 tiles ready to hand to `ChessBoardView`'s `.channel` overlay.
    ///
    /// Cells are transposed from encoder frame to absolute board
    /// coordinates: when black is to move, the network emits logits
    /// with rows vertically flipped (per `BoardEncoder` and
    /// `PolicyEncoding`'s "encoder frame" convention), but the
    /// parent `ChessBoardView` shows pieces in absolute coordinates
    /// (row 0 = rank 8). Without the un-flip, every black-to-move
    /// channel would render upside-down versus the pieces on its
    /// own board — extremely confusing. We do the flip here, once
    /// per re-eval, so tile renderers can stay identity-mapped.
    fileprivate static func computeChannels(
        from logits: [Float],
        currentPlayer: PieceColor
    ) -> [ChannelData] {
        let n = logits.count
        precondition(n == ChessNetwork.policySize,
                     "PolicyChannelsPanel expects \(ChessNetwork.policySize) logits, got \(n)")

        // Stable softmax over all 4864 cells.
        var maxLogit: Float = -.infinity
        for i in 0..<n where logits[i] > maxLogit {
            maxLogit = logits[i]
        }
        var probs = [Float](repeating: 0, count: n)
        var sum: Float = 0
        for i in 0..<n {
            let e = expf(logits[i] - maxLogit)
            probs[i] = e
            sum += e
        }
        let invSum: Float = sum > 0 ? 1 / sum : 0
        var globalMaxProb: Float = 0
        for i in 0..<n {
            probs[i] *= invSum
            if probs[i] > globalMaxProb { globalMaxProb = probs[i] }
        }
        let normInv: Float = globalMaxProb > 0 ? 1 / globalMaxProb : 0

        let flip = currentPlayer == .black

        var result: [ChannelData] = []
        result.reserveCapacity(PolicyEncoding.channelCount)
        for chan in 0..<PolicyEncoding.channelCount {
            var cells = [Float](repeating: 0, count: 64)
            var mass: Float = 0
            let base = chan * 64
            for r in 0..<8 {
                let displayRow = flip ? (7 - r) : r
                for c in 0..<8 {
                    let p = probs[base + r * 8 + c]
                    mass += p
                    cells[displayRow * 8 + c] = p * normInv
                }
            }
            result.append(ChannelData(
                id: chan,
                cellValues: cells,
                totalMass: mass,
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
