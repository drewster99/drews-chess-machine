import SwiftUI

/// The chess board rendered inside the dedicated human-vs-network
/// window. Replaces the generic `LiveBoardWithNavigationView` for
/// this surface because the human-play board needs features that the
/// shared board doesn't: a last-move destination highlight (green),
/// an in-check king highlight (red), and sliding piece animation
/// keyed on stable per-piece identities (not the cross-fade the
/// shared board does on the entire `[Piece?]` array).
///
/// Coordinate convention: every square index this view receives is
/// in **visual** coordinates — the caller has already applied the
/// "human-plays-black → rotate 180°" flip (`63 - sq`) so a square
/// index of 0 always means the top-left of what the user sees and 63
/// the bottom-right. Logical chess coordinates never leak in.
///
/// Animation strategy: pieces are not part of the `Canvas` like in
/// `ChessBoardView` — they live in a `ZStack` of position-modified
/// `Image` views, one per piece, keyed by a stable `UUID`. On every
/// board update, a small reconcile pass matches each piece in the
/// new board to the closest old piece of the same type+color (and
/// falls back to promotion-aware pawn → new-piece matching), so the
/// moving piece's UUID survives across the move. SwiftUI then
/// animates the `.position(...)` change for that id, producing a
/// smooth slide. Captured pieces (whose ids no longer appear in the
/// new state) fade out via `.transition(.opacity)`; fresh pieces
/// (game reset, post-promotion replacement that didn't match) fade
/// in the same way.
struct HumanPlayBoardView: View {
    /// 64-square board in visual coordinates.
    let pieces: [Piece?]

    /// Selected from-square (visual frame). `nil` when no piece is
    /// picked or human play isn't active. Drawn as a thick yellow
    /// ring.
    let selectedFromSquare: Int?
    /// Legal destinations from `selectedFromSquare` (visual frame).
    /// Drawn as translucent green dots in the cell centers.
    let legalMoveTargets: Set<Int>
    /// Destination square of the most recent move (visual frame).
    /// Drawn as a translucent green fill on the whole cell so the
    /// user can always see where the last piece landed — even after
    /// the slide animation finishes and even after the network has
    /// played its reply. `nil` before any move in the current game.
    let lastMoveDestinationSquare: Int?
    /// Square of the king currently in check (visual frame). Drawn
    /// as a red fill underneath the king sprite. `nil` when neither
    /// side is in check (or when the game has ended).
    let checkSquare: Int?

    /// True while it's the human's turn and the controller has
    /// surfaced legal moves. Gates the tap-input layer.
    let humanMoveActive: Bool
    /// Color the human is playing — needed to label the promotion
    /// picker pieces. `nil` outside human play.
    let humanColor: PieceColor?
    /// Pending promotion choice from the controller, in logical
    /// coordinates. Drives the picker overlay.
    let pendingPromotion: PlayController.PendingPromotion?
    /// Promotion-target square in visual coordinates, used by the
    /// picker to anchor itself near the destination. Decoupled from
    /// `pendingPromotion.toRow`/`.toCol` (logical) so the caller does
    /// the flip exactly once.
    let promotionVisualSquare: Int?

    /// Tap callback: receives a 0..<64 visual square index.
    let onTapSquare: (Int) -> Void
    /// Promotion picker chose a piece.
    let onSelectPromotion: (PieceType) -> Void
    /// Promotion picker dismissed without a choice.
    let onCancelPromotion: () -> Void

    private static let lightSquare = Color(red: 0.94, green: 0.85, blue: 0.71)
    private static let darkSquare = Color(red: 0.71, green: 0.53, blue: 0.39)

    /// Stable-identity piece records for the animation layer. Seeded
    /// in `.onAppear` and reconciled on every `pieces` update so the
    /// moving piece keeps its UUID across a move. Sole owner of
    /// "which piece is which" — `pieces: [Piece?]` carries no
    /// identity by itself.
    @State private var tracked: [TrackedPiece] = []

    var body: some View {
        GeometryReader { geo in
            let boardSize = min(geo.size.width, geo.size.height)
            let cellSize = boardSize / 8
            ZStack {
                // 1. Squares + last-move/check fills (Canvas — single
                //    geometry pass for the entire 8x8 grid).
                Canvas(opaque: true, rendersAsynchronously: false) { ctx, _ in
                    Self.drawSquares(
                        ctx: &ctx,
                        cellSize: cellSize,
                        lastMoveTo: lastMoveDestinationSquare,
                        checkSquare: checkSquare
                    )
                }
                .frame(width: boardSize, height: boardSize)

                // 2. Pieces as animatable SwiftUI views (one Image
                //    per piece, position-keyed, stable id).
                pieceLayer(cellSize: cellSize)

                // 3. Selection ring + legal-target dots.
                Canvas { ctx, _ in
                    Self.drawSelectionAndTargets(
                        ctx: &ctx,
                        cellSize: cellSize,
                        selectedFromSquare: selectedFromSquare,
                        legalMoveTargets: legalMoveTargets
                    )
                }
                .frame(width: boardSize, height: boardSize)
                .allowsHitTesting(false)

                // 4. Tap hit layer.
                Color.clear
                    .contentShape(Rectangle())
                    .frame(width: boardSize, height: boardSize)
                    .gesture(
                        DragGesture(minimumDistance: 0)
                            .onEnded { value in
                                guard humanMoveActive else { return }
                                if pendingPromotion != nil {
                                    // Picker is open; SwiftUI hit-tests
                                    // its buttons first, so reaching
                                    // this branch means the user tapped
                                    // outside the picker — treat as
                                    // cancel. Keep selection so they
                                    // can pick a different move.
                                    onCancelPromotion()
                                    return
                                }
                                if let sq = Self.squareIndex(at: value.location, boardSize: boardSize) {
                                    onTapSquare(sq)
                                }
                            }
                    )

                // 5. Promotion picker.
                if pendingPromotion != nil,
                   let visualSquare = promotionVisualSquare,
                   let color = humanColor {
                    promotionPicker(
                        visualSquare: visualSquare,
                        color: color,
                        boardSize: boardSize,
                        cellSize: cellSize
                    )
                }
            }
            .frame(width: boardSize, height: boardSize)
        }
        .aspectRatio(1, contentMode: .fit)
        .onAppear {
            tracked = Self.seedTracked(from: pieces)
        }
        .onChange(of: pieces) { _, newPieces in
            tracked = Self.reconcile(old: tracked, newBoard: newPieces)
        }
    }

    // MARK: - Piece layer

    @ViewBuilder
    private func pieceLayer(cellSize: CGFloat) -> some View {
        let pieceSize = cellSize * 0.84
        ZStack {
            ForEach(tracked) { tp in
                let row = tp.square / 8
                let col = tp.square % 8
                Image(tp.piece.assetName)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(width: pieceSize, height: pieceSize)
                    .position(
                        x: (CGFloat(col) + 0.5) * cellSize,
                        y: (CGFloat(row) + 0.5) * cellSize
                    )
                    .transition(.opacity)
            }
        }
        // Drive position interpolation for ids that survive across
        // updates, and fade for ids that appear / disappear. The
        // duration is matched to a human's "I just saw that move"
        // glance — too fast and the slide is unreadable, too slow and
        // the AI's reply feels laggy on top of the existing 2-second
        // delay.
        .animation(.easeInOut(duration: 0.3), value: tracked)
        .allowsHitTesting(false)
    }

    // MARK: - Square / highlight drawing

    private static func drawSquares(
        ctx: inout GraphicsContext,
        cellSize: CGFloat,
        lastMoveTo: Int?,
        checkSquare: Int?
    ) {
        for row in 0..<8 {
            for col in 0..<8 {
                let rect = CGRect(
                    x: CGFloat(col) * cellSize,
                    y: CGFloat(row) * cellSize,
                    width: cellSize,
                    height: cellSize
                )
                let isLight = (row + col) % 2 == 0
                ctx.fill(
                    Path(rect),
                    with: .color(isLight ? lightSquare : darkSquare)
                )
            }
        }
        // Last-move destination — translucent green fill on the cell.
        if let to = lastMoveTo {
            let row = to / 8, col = to % 8
            let rect = CGRect(
                x: CGFloat(col) * cellSize,
                y: CGFloat(row) * cellSize,
                width: cellSize,
                height: cellSize
            )
            ctx.fill(Path(rect), with: .color(Color.green.opacity(0.42)))
        }
        // Check — drawn last so a king sitting on the previous move's
        // destination still reads as "in check" rather than "just
        // moved there".
        if let king = checkSquare {
            let row = king / 8, col = king % 8
            let rect = CGRect(
                x: CGFloat(col) * cellSize,
                y: CGFloat(row) * cellSize,
                width: cellSize,
                height: cellSize
            )
            ctx.fill(Path(rect), with: .color(Color.red.opacity(0.55)))
        }
    }

    private static func drawSelectionAndTargets(
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

    // MARK: - Promotion picker

    @ViewBuilder
    private func promotionPicker(
        visualSquare: Int,
        color: PieceColor,
        boardSize: CGFloat,
        cellSize: CGFloat
    ) -> some View {
        let promoRow = visualSquare / 8
        let promoCol = visualSquare % 8
        let pickerWidth = cellSize * 4
        let halfPicker = pickerWidth / 2
        let cellCenterX = (CGFloat(promoCol) + 0.5) * cellSize
        let rawX = cellCenterX - halfPicker
        let clampedX = min(max(rawX, 0), boardSize - pickerWidth)
        // Place the picker just below the destination row when the
        // destination sits in the top half (more room below); flip
        // to "just above" otherwise. Same logic as the shared board.
        let preferBelow = promoRow <= 3
        let y = preferBelow
            ? CGFloat(promoRow + 1) * cellSize
            : CGFloat(promoRow) * cellSize - cellSize
        HStack(spacing: 0) {
            ForEach(HumanPlayPromotionPicker.pieceChoices, id: \.self) { type in
                let label = HumanPlayPromotionPicker.assetName(type: type, color: color)
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
                .help(HumanPlayPromotionPicker.displayName(type))
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

    // MARK: - Hit testing

    private static func squareIndex(at point: CGPoint, boardSize: CGFloat) -> Int? {
        guard boardSize > 0 else { return nil }
        guard point.x >= 0, point.y >= 0, point.x < boardSize, point.y < boardSize else {
            return nil
        }
        let squareSize = boardSize / 8
        let col = Int(point.x / squareSize)
        let row = Int(point.y / squareSize)
        guard (0..<8).contains(col), (0..<8).contains(row) else { return nil }
        return row * 8 + col
    }

    // MARK: - Piece tracking / reconcile

    /// Tracked piece — a unique id, a piece, and a square. Identity
    /// is the id; the `piece` and `square` change as the same id
    /// moves around the board (a slide) or morphs (promotion).
    /// Equatable on all fields so SwiftUI's `.animation(value:)`
    /// fires when a piece moves, gets captured, or gets promoted.
    struct TrackedPiece: Identifiable, Equatable {
        let id: UUID
        let piece: Piece
        let square: Int
    }

    static func seedTracked(from board: [Piece?]) -> [TrackedPiece] {
        var out: [TrackedPiece] = []
        for idx in 0..<board.count {
            if let p = board[idx] {
                out.append(TrackedPiece(id: UUID(), piece: p, square: idx))
            }
        }
        return out
    }

    /// Reconcile the old `tracked` list against a new `[Piece?]`
    /// board so SwiftUI can animate position changes for ids that
    /// survive the update.
    ///
    /// Algorithm:
    /// 1. Squares whose piece is identical to the old square keep
    ///    their tracked id verbatim.
    /// 2. Remaining new squares are matched against unused old
    ///    pieces of the same `Piece` (type+color), choosing the
    ///    Manhattan-closest old position as the source. This
    ///    correctly tracks normal moves, captures (the captured
    ///    piece is simply unmatched and falls off the list), castles
    ///    (king and rook are unique types so each matches its only
    ///    available source), and en passant (the captured pawn isn't
    ///    in the new board, so it just disappears).
    /// 3. Anything still unmatched on the new side falls into the
    ///    promotion fallback: a new non-pawn piece can claim an
    ///    unused same-color pawn as its source so the promoting pawn
    ///    "morphs" into a queen/rook/knight/bishop in place. Square
    ///    moves alongside the morph.
    /// 4. Anything still unmatched after that gets a fresh UUID —
    ///    pieces appearing out of thin air (game reset to a position
    ///    that wasn't a strict superset of the old one, etc.).
    ///
    /// This is O(64²) worst case but the inner loops are tiny — a
    /// chess board has at most 32 pieces and `seedTracked` runs
    /// once per board update, not per frame.
    static func reconcile(old: [TrackedPiece], newBoard: [Piece?]) -> [TrackedPiece] {
        var oldByPos: [Int: TrackedPiece] = [:]
        for tp in old { oldByPos[tp.square] = tp }

        var result: [TrackedPiece] = []
        var consumed: Set<UUID> = []
        var unmatchedNew: [(Int, Piece)] = []

        // Pass 1: identical (square, piece) pairs.
        for idx in 0..<newBoard.count {
            guard let p = newBoard[idx] else { continue }
            if let old = oldByPos[idx], old.piece == p {
                result.append(TrackedPiece(id: old.id, piece: p, square: idx))
                consumed.insert(old.id)
            } else {
                unmatchedNew.append((idx, p))
            }
        }

        // Pass 2: same-piece, different-square matching (slides /
        // captures / castles / en passant).
        var available: [TrackedPiece] = old.filter { !consumed.contains($0.id) }
        for (newSquare, p) in unmatchedNew {
            // Find available pieces with the exact same type+color
            // and pick the closest one as the source. Pawn promotion
            // is handled in the next pass after this one, so do not
            // match a non-pawn here using a pawn source.
            let sameTypeIndexes = available.indices.filter { available[$0].piece == p }
            if !sameTypeIndexes.isEmpty {
                let targetRow = newSquare / 8
                let targetCol = newSquare % 8
                var bestI = sameTypeIndexes[0]
                var bestDist = Int.max
                for i in sameTypeIndexes {
                    let src = available[i]
                    let d = abs(src.square / 8 - targetRow)
                        + abs(src.square % 8 - targetCol)
                    if d < bestDist {
                        bestDist = d
                        bestI = i
                    }
                }
                let src = available.remove(at: bestI)
                result.append(TrackedPiece(id: src.id, piece: p, square: newSquare))
            } else {
                // Defer to the promotion pass.
                result.append(TrackedPiece(id: UUID(), piece: p, square: newSquare))
            }
        }

        // Pass 3: promotion fixup. Any TrackedPiece in `result` with
        // a freshly-minted id that is a non-pawn and whose color has
        // an unmatched pawn in `available` claims that pawn's id —
        // the promoting pawn morphs into the new piece type. We do
        // this as a post-pass rather than inside pass 2 because
        // promotion is the rare case and the simpler same-type
        // matching above handles every other move type cleanly.
        for i in result.indices {
            let tp = result[i]
            guard tp.piece.type != .pawn else { continue }
            // Was this id minted fresh in pass 2 (no prior owner)?
            guard !old.contains(where: { $0.id == tp.id }) else { continue }
            if let promoPawnI = available.firstIndex(
                where: { $0.piece.color == tp.piece.color && $0.piece.type == .pawn }
            ) {
                let pawn = available.remove(at: promoPawnI)
                result[i] = TrackedPiece(id: pawn.id, piece: tp.piece, square: tp.square)
            }
        }

        return result
    }
}

// MARK: - Promotion picker helpers

private enum HumanPlayPromotionPicker {
    /// Order by what users pick most often: queen → rook → knight → bishop.
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
