import SwiftUI

/// Chess board flanked by chevrons that walk through the available
/// overlays (plain board, top-moves arrows, then the per-channel
/// tensor views). The board itself accepts drag input in
/// forward-pass editor mode, and overlays a "N concurrent games"
/// placeholder when multi-worker self-play makes a single live
/// board meaningless.
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
                                        guard forwardPassEditable else { return }
                                        let fromSq = squareIndex(value.startLocation, boardSize)
                                        let toSq = squareIndex(value.location, boardSize)
                                        onApplyFreePlacementDrag(fromSq, toSq)
                                    }
                            )
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
}
