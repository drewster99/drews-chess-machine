import SwiftUI

/// Draws the starting chess position with a switchable overlay:
/// - `.topMoves`: gradient arrows from source to destination + ghost pieces
/// - `.channel`: blue highlights on active squares of a tensor channel
struct ChessBoardView: View {
    /// 64-square flat board, indexed as row * 8 + col.
    var pieces: [Piece?] = GameState.starting.board
    var overlay: Overlay = .none
    /// Tint for the `.channel` overlay's per-square fills. Defaults
    /// to blue (matches the input-tensor channel strip elsewhere in
    /// the UI). The policy-channels panel passes red so an active-
    /// channel grid is visually distinct from an input-tensor grid
    /// at a glance.
    var channelColor: Color = .blue

    private static let lightSquare = Color(red: 0.94, green: 0.85, blue: 0.71)
    private static let darkSquare = Color(red: 0.71, green: 0.53, blue: 0.39)

    private static let allAssetNames = [
        "wP", "wN", "wB", "wR", "wQ", "wK",
        "bP", "bN", "bB", "bR", "bQ", "bK"
    ]

    var body: some View {
        Canvas(opaque: true, rendersAsynchronously: true) { context, size in
            let squareSize = min(size.width, size.height) / 8

            // 1. Board squares
            for row in 0..<8 {
                for col in 0..<8 {
                    let rect = CGRect(
                        x: CGFloat(col) * squareSize,
                        y: CGFloat(row) * squareSize,
                        width: squareSize,
                        height: squareSize
                    )
                    let isLight = (row + col) % 2 == 0
                    context.fill(
                        Path(rect),
                        with: .color(isLight ? Self.lightSquare : Self.darkSquare)
                    )
                }
            }

            // 2. Channel overlay (drawn before pieces so pieces stay readable)
            if case .channel(let values) = overlay {
                for row in 0..<8 {
                    for col in 0..<8 {
                        let value = values[row * 8 + col]
                        if value > 0.001 {
                            let rect = CGRect(
                                x: CGFloat(col) * squareSize,
                                y: CGFloat(row) * squareSize,
                                width: squareSize,
                                height: squareSize
                            )
                            context.fill(
                                Path(rect),
                                with: .color(channelColor.opacity(Double(value) * 0.6))
                            )
                        }
                    }
                }
            }

            // 3. Pieces at their actual positions
            let pieceInset = squareSize * 0.08
            for row in 0..<8 {
                let rowBase = row * 8
                for col in 0..<8 {
                    if let piece = pieces[rowBase + col] {
                        let rect = CGRect(
                            x: CGFloat(col) * squareSize + pieceInset,
                            y: CGFloat(row) * squareSize + pieceInset,
                            width: squareSize - pieceInset * 2,
                            height: squareSize - pieceInset * 2
                        )
                        if let resolved = context.resolveSymbol(id: piece.assetName) {
                            context.draw(resolved, in: rect)
                        }
                    }
                }
            }

            // 4. Top moves overlay (drawn after pieces — arrows on top)
            if case .topMoves(let moves) = overlay {
                // Arrows — weakest first so strongest renders on top
                for (rank, move) in moves.enumerated().reversed() {
                    let fromCenter = CGPoint(
                        x: (CGFloat(move.fromCol) + 0.5) * squareSize,
                        y: (CGFloat(move.fromRow) + 0.5) * squareSize
                    )
                    let toCenter = CGPoint(
                        x: (CGFloat(move.toCol) + 0.5) * squareSize,
                        y: (CGFloat(move.toRow) + 0.5) * squareSize
                    )

                    let color = Self.arrowColor(forRank: rank, of: moves.count)
                    let path = Self.arrowPath(
                        from: fromCenter,
                        to: toCenter,
                        shaftWidth: squareSize * 0.28,
                        headWidth: squareSize * 0.55,
                        headLength: squareSize * 0.4
                    )

                    context.fill(
                        path,
                        with: .linearGradient(
                            Gradient(stops: [
                                .init(color: color.opacity(1.0), location: 0),
                                .init(color: color.opacity(0.6), location: 1)
                            ]),
                            startPoint: fromCenter,
                            endPoint: toCenter
                        )
                    )
                }

                // Ghost pieces at target squares (25% opacity).
                // Same inset as the live pieces drawn above.
                let ghostInset = squareSize * 0.08
                for move in moves {
                    guard let assetName = move.piece,
                          let resolved = context.resolveSymbol(id: assetName) else { continue }
                    let rect = CGRect(
                        x: CGFloat(move.toCol) * squareSize + ghostInset,
                        y: CGFloat(move.toRow) * squareSize + ghostInset,
                        width: squareSize - ghostInset * 2,
                        height: squareSize - ghostInset * 2
                    )
                    context.drawLayer { ghostContext in
                        ghostContext.opacity = 0.25
                        ghostContext.draw(resolved, in: rect)
                    }
                }
            }
        } symbols: {
            ForEach(Self.allAssetNames, id: \.self) { name in
                Image(name)
                    .resizable()
                    .tag(name)
            }
        }
        .aspectRatio(1, contentMode: .fit)
    }

    // MARK: - Arrow Color

    private static func arrowColor(forRank rank: Int, of total: Int) -> Color {
        let t = Double(rank) / Double(max(total - 1, 1))
        let hue = 0.33 + t * 0.35
        let saturation = 0.85 - t * 0.2
        let brightness = 0.75 - t * 0.1
        return Color(hue: hue, saturation: saturation, brightness: brightness)
    }

    // MARK: - Arrow Geometry

    private static func arrowPath(
        from: CGPoint,
        to: CGPoint,
        shaftWidth: CGFloat,
        headWidth: CGFloat,
        headLength: CGFloat
    ) -> Path {
        let dx = to.x - from.x
        let dy = to.y - from.y
        let length = sqrt(dx * dx + dy * dy)
        guard length > 0.001 else { return Path() }

        let ux = dx / length
        let uy = dy / length
        let px = -uy
        let py = ux

        let shaftHalf = shaftWidth / 2
        let headHalf = headWidth / 2
        let headStart = max(length - headLength, length * 0.5)

        var path = Path()

        path.move(to: CGPoint(x: from.x + px * shaftHalf, y: from.y + py * shaftHalf))
        path.addLine(to: CGPoint(
            x: from.x + ux * headStart + px * shaftHalf,
            y: from.y + uy * headStart + py * shaftHalf
        ))
        path.addLine(to: CGPoint(
            x: from.x + ux * headStart + px * headHalf,
            y: from.y + uy * headStart + py * headHalf
        ))
        path.addLine(to: to)
        path.addLine(to: CGPoint(
            x: from.x + ux * headStart - px * headHalf,
            y: from.y + uy * headStart - py * headHalf
        ))
        path.addLine(to: CGPoint(
            x: from.x + ux * headStart - px * shaftHalf,
            y: from.y + uy * headStart - py * shaftHalf
        ))
        path.addLine(to: CGPoint(x: from.x - px * shaftHalf, y: from.y - py * shaftHalf))

        path.closeSubpath()
        return path
    }
}

extension ChessBoardView {
    /// What to draw on top of the pieces.
    enum Overlay {
        case none
        case topMoves([MoveVisualization])
        case channel([Float])
    }
}
