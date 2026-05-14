import SwiftUI

/// Chess board flanked by chevrons that walk through the available
/// overlays (plain board, top-moves arrows, then the per-channel
/// tensor views). The board itself accepts drag input in
/// forward-pass editor mode, and overlays a "N concurrent games"
/// placeholder when multi-worker self-play makes a single live
/// board meaningless.
///
/// When `humanMoveActive` is true, the board also accepts tap input
/// for human-vs-network play: the caller paints `selectedFromSquare`
/// (the user's first tap) and `legalMoveTargets` (the destinations
/// that would be legal from there) so the user gets visual feedback
/// without the controller having to re-derive squares from board
/// coordinates. `pendingPromotion` surfaces a piece-picker overlay
/// when a pawn promotion is being chosen.
struct LiveBoardWithNavigationView: View {
    let pieces: [Piece?]
    let overlay: ChessBoardView.Overlay
    let selectedOverlay: Int
    let inferenceResultPresent: Bool
    let forwardPassEditable: Bool
    let realTraining: Bool
    let isCandidateTestActive: Bool
    let workerCount: Int
    let onNavigate: (Int) -> Void
    let onApplyFreePlacementDrag: (Int?, Int?) -> Void
    let squareIndex: (CGPoint, CGFloat) -> Int?
    /// Called as the cursor moves over (and off of) the board.
    /// Argument is the hovered square index (0..<64, row*8+col) or
    /// nil when the cursor leaves the board. Drives the upper
    /// "top-3 channels at this square" overlay in `UpperContentView`.
    /// Always wired (the SwiftUI hit layer is shared with the
    /// drag editor and `forwardPassEditable` only gates drags, not
    /// hover) so the overlay can fire in any mode that produces an
    /// inference result.
    let onHoverSquare: (Int?) -> Void

    // MARK: - Human-vs-network play (all optional)

    /// True while a human-vs-network game is awaiting the user's
    /// move. Routes board taps to `onTapSquare` instead of the
    /// forward-pass editor.
    var humanMoveActive: Bool = false
    /// Square (0..<64, visual coordinates) the user picked first
    /// this turn. Drawn as a thick yellow ring. `nil` when no
    /// from-piece is selected.
    var selectedFromSquare: Int?
    /// Visual squares (0..<64) that are legal destinations from
    /// `selectedFromSquare`. Drawn as translucent dots. Empty
    /// outside human play or when no from-piece is selected.
    var legalMoveTargets: Set<Int> = []
    /// Non-nil while the human has chosen a (from, to) pair that
    /// requires picking a promotion piece — overlays a four-button
    /// picker centered on the board.
    var pendingPromotion: PlayController.PendingPromotion?
    /// Side the human is playing, used to pick a tint and to label
    /// the picker pieces. `nil` outside human-play mode.
    var humanColor: PieceColor?
    /// Promotion-target square in visual coordinates, used to anchor
    /// the picker near the destination. `nil` when no picker is
    /// active. Decoupled from `pendingPromotion.toRow`/`.toCol`
    /// (which are in logical coordinates) so the caller does the
    /// visual flip exactly once.
    var promotionVisualSquare: Int?
    /// Tap callback: receives the 0..<64 visual square index. Only
    /// fires when `humanMoveActive` and no promotion picker is open.
    var onTapSquare: (Int) -> Void = { _ in }
    /// Picker callback: user chose a piece type for the pending
    /// promotion. Fires only while `pendingPromotion != nil`.
    var onSelectPromotion: (PieceType) -> Void = { _ in }
    /// Picker callback: user dismissed the promotion picker without
    /// committing. The caller leaves the from-square selected so
    /// the user can pick a different move.
    var onCancelPromotion: () -> Void = { }

    var body: some View {
        HStack(spacing: 8) {
            // Left chevron is enabled whenever we can step toward a
            // lower-index mode. -1 (plain board) is the floor and is
            // always reachable — independent of inferenceResult and
            // showForwardPassUI — so we only gate left when we're
            // already at the floor.
            let leftDisabled = selectedOverlay <= -1
            Button(
                action: { onNavigate(-1) },
                label: {
                    Image(systemName: "chevron.left").font(.title3).frame(width: 24)
                }
            )
            .buttonStyle(.plain)
            .disabled(leftDisabled)
            .opacity(leftDisabled ? 0.2 : 0.6)

            ChessBoardView(pieces: pieces, overlay: overlay)
                .overlay {
                    // Transparent hit layer that converts drag
                    // coordinates to squares and routes edits through
                    // the supplied callback. Sized to match the
                    // board's square frame via the overlay modifier,
                    // so local coordinates map 1:1 onto board squares.
                    // Disabled outside forward-pass mode so
                    // game/training views aren't hijacked.
                    GeometryReader { geo in
                        let boardSize = min(geo.size.width, geo.size.height)
                        // Single hit layer: hover always live, drag
                        // gated by `forwardPassEditable`. We can't
                        // split these into two stacked Color.clear
                        // layers because the upper one (whichever it
                        // is) consumes hover events and starves the
                        // other — combining both modifiers on the
                        // same view lets SwiftUI route hover and
                        // gesture independently.
                        Color.clear
                            .contentShape(Rectangle())
                            .onContinuousHover { phase in
                                switch phase {
                                case .active(let pt):
                                    onHoverSquare(squareIndex(pt, boardSize))
                                case .ended:
                                    onHoverSquare(nil)
                                }
                            }
                            .gesture(
                                DragGesture(minimumDistance: 0)
                                    .onEnded { value in
                                        if humanMoveActive {
                                            if pendingPromotion != nil {
                                                // Picker is open and the tap landed
                                                // somewhere on the board. SwiftUI
                                                // hit-testing routes taps on the
                                                // four piece buttons (rendered as
                                                // a later overlay) to those
                                                // buttons first, so this branch
                                                // only fires on a tap *outside*
                                                // the picker — i.e. the user
                                                // changed their mind and wants to
                                                // pick a different move. Clear
                                                // the picker; keep the from-piece
                                                // selected so the user can choose
                                                // a different destination.
                                                onCancelPromotion()
                                                return
                                            }
                                            if let sq = squareIndex(value.location, boardSize) {
                                                onTapSquare(sq)
                                            }
                                            return
                                        }
                                        guard forwardPassEditable else { return }
                                        let fromSq = squareIndex(value.startLocation, boardSize)
                                        let toSq = squareIndex(value.location, boardSize)
                                        onApplyFreePlacementDrag(fromSq, toSq)
                                    }
                            )
                    }
                }
                .overlay {
                    // Human-play highlights: the selected from-square
                    // (yellow ring) and a translucent dot on every
                    // legal target. Drawn into a single `Canvas`
                    // overlay so the rendering doesn't fight the
                    // board's primary `Canvas` for z-order, and so
                    // both passes (ring + dots) share one geometry
                    // pass.
                    if humanMoveActive
                        || selectedFromSquare != nil
                        || !legalMoveTargets.isEmpty {
                        Canvas { ctx, size in
                            let boardSize = min(size.width, size.height)
                            let cellSize = boardSize / 8
                            Self.drawHumanPlayHighlights(
                                ctx: &ctx,
                                cellSize: cellSize,
                                selectedFromSquare: selectedFromSquare,
                                legalMoveTargets: legalMoveTargets
                            )
                        }
                        .allowsHitTesting(false)
                    }
                }
                .overlay {
                    // Multi-worker placeholder — the live animated
                    // game board only works with one driving worker
                    // (N=1), because a single `GameWatcher` can't
                    // track multiple concurrent games without
                    // flicker. When N>1 we still show the board slot
                    // (so the Candidate test picker remains usable
                    // and the layout doesn't shift) but overlay a
                    // centered label indicating how many workers are
                    // running. Hidden in candidate-test mode so the
                    // probe board stays clean.
                    if realTraining
                        && !isCandidateTestActive
                        && workerCount > 1 {
                        Text("N = \(workerCount) concurrent games\nLive board hidden")
                            .font(.system(.body, design: .monospaced))
                            .multilineTextAlignment(.center)
                            .foregroundStyle(.white)
                            .padding(14)
                            .background(
                                RoundedRectangle(cornerRadius: 10)
                                    .fill(Color.black.opacity(0.7))
                            )
                    }
                }
                .overlay {
                    if let promotion = pendingPromotion,
                       let visualSquare = promotionVisualSquare,
                       let color = humanColor {
                        promotionPicker(
                            promotion: promotion,
                            visualSquare: visualSquare,
                            color: color
                        )
                    }
                }

            // Right chevron walks up through Top Moves / channels. Both
            // of those require an inferenceResult to render meaningful
            // content, so right is disabled either when we are at the
            // ceiling or when there is no inference data to step into.
            let rightDisabled = !inferenceResultPresent || selectedOverlay >= ChessNetwork.inputPlanes
            Button(
                action: { onNavigate(1) },
                label: {
                    Image(systemName: "chevron.right").font(.title3).frame(width: 24)
                }
            )
            .buttonStyle(.plain)
            .disabled(rightDisabled)
            .opacity(rightDisabled ? 0.2 : 0.6)
        }
    }

    // MARK: - Promotion picker

    @ViewBuilder
    private func promotionPicker(
        promotion: PlayController.PendingPromotion,
        visualSquare: Int,
        color: PieceColor
    ) -> some View {
        GeometryReader { geo in
            let boardSize = min(geo.size.width, geo.size.height)
            let cellSize = boardSize / 8
            let promoRow = visualSquare / 8
            let promoCol = visualSquare % 8
            // Anchor the picker just below the destination square,
            // or just above if there isn't enough room beneath it.
            // The picker is 4 cells wide and clamped horizontally so
            // it never spills off the board.
            let pickerWidth = cellSize * 4
            let halfPicker = pickerWidth / 2
            let cellCenterX = (CGFloat(promoCol) + 0.5) * cellSize
            let rawX = cellCenterX - halfPicker
            let clampedX = min(max(rawX, 0), boardSize - pickerWidth)
            let preferBelow = promoRow <= 3
            let y = preferBelow
                ? CGFloat(promoRow + 1) * cellSize
                : CGFloat(promoRow) * cellSize - cellSize
            HStack(spacing: 0) {
                ForEach(PromotionPickerView.pieceChoices, id: \.self) { type in
                    let label = PromotionPickerView.assetName(type: type, color: color)
                    Button(
                        action: { onSelectPromotion(type) },
                        label: {
                            Image(label)
                                .resizable()
                                .aspectRatio(contentMode: .fit)
                                .padding(4)
                                .frame(width: cellSize, height: cellSize)
                                .background(Color.white.opacity(0.85))
                        }
                    )
                    .buttonStyle(.plain)
                    .help(PromotionPickerView.displayName(type))
                }
            }
            .background(
                RoundedRectangle(cornerRadius: 6)
                    .fill(Color.black.opacity(0.35))
            )
            .overlay(
                RoundedRectangle(cornerRadius: 6)
                    .stroke(Color.black, lineWidth: 1)
            )
            .position(x: clampedX + halfPicker, y: y + cellSize / 2)
        }
        .allowsHitTesting(true)
    }

    // MARK: - Highlight drawing

    private static func drawHumanPlayHighlights(
        ctx: inout GraphicsContext,
        cellSize: CGFloat,
        selectedFromSquare: Int?,
        legalMoveTargets: Set<Int>
    ) {
        if let sel = selectedFromSquare {
            let row = sel / 8
            let col = sel % 8
            let rect = CGRect(
                x: CGFloat(col) * cellSize,
                y: CGFloat(row) * cellSize,
                width: cellSize,
                height: cellSize
            )
            ctx.stroke(
                Path(rect.insetBy(dx: cellSize * 0.06, dy: cellSize * 0.06)),
                with: .color(.yellow),
                lineWidth: max(2, cellSize * 0.06)
            )
        }
        let dotRadius = cellSize * 0.18
        for tgt in legalMoveTargets {
            let row = tgt / 8
            let col = tgt % 8
            let cx = (CGFloat(col) + 0.5) * cellSize
            let cy = (CGFloat(row) + 0.5) * cellSize
            let dotRect = CGRect(
                x: cx - dotRadius,
                y: cy - dotRadius,
                width: dotRadius * 2,
                height: dotRadius * 2
            )
            ctx.fill(
                Path(ellipseIn: dotRect),
                with: .color(.green.opacity(0.55))
            )
        }
    }
}

// MARK: - Promotion picker helpers

private enum PromotionPickerView {
    /// Order pieces by what users pick most often: queen, then rook,
    /// then knight, then bishop. Underpromotion to knight is the
    /// only commonly-useful underpromotion (smothered mate /
    /// fork escape), so it sits ahead of bishop.
    static let pieceChoices: [PieceType] = [.queen, .rook, .knight, .bishop]

    static func displayName(_ type: PieceType) -> String {
        switch type {
        case .queen: return "Queen"
        case .rook: return "Rook"
        case .bishop: return "Bishop"
        case .knight: return "Knight"
        case .pawn: return "Pawn"
        case .king: return "King"
        }
    }

    static func assetName(type: PieceType, color: PieceColor) -> String {
        let colorPrefix = color == .white ? "w" : "b"
        let code: String
        switch type {
        case .queen: code = "Q"
        case .rook: code = "R"
        case .bishop: code = "B"
        case .knight: code = "N"
        case .pawn: code = "P"
        case .king: code = "K"
        }
        return "\(colorPrefix)\(code)"
    }
}
