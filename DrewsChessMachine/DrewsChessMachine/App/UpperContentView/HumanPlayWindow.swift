import AppKit
import SwiftUI

/// Standalone window that hosts a human-vs-network game. Created by
/// `HumanPlayWindowLauncher.openOrFocus(...)` once
/// `PlayController.materializeTask` succeeds; closed either by the
/// user (X button), by `Stop Game` from the in-window toolbar or
/// the Chess menu, or by a Reset (which re-opens with a fresh game).
///
/// Owns no game state directly: the rendered board, side-to-move,
/// legal-move highlights, and pending promotion all read from the
/// shared `PlayController` (`@MainActor @Observable`) and the
/// `GameWatcher` snapshot (polled on a 100 ms timer because
/// `GameWatcher` is intentionally not `@Observable` — its mutations
/// fire from the ChessMachine delegate queue at game-loop rate, and
/// the project decouples UI redraw from that rate via polling). The
/// window's lifecycle is owned by the controller + registry pattern
/// used elsewhere in the project (see `LogAnalysisWindowController`):
/// the registry holds the strong reference so the controller doesn't
/// dealloc the moment SwiftUI lets go of the hosting view, and the
/// controller unregisters in `windowWillClose`.
@MainActor
final class HumanPlayWindowController: NSWindowController, NSWindowDelegate {
    private let playController: PlayController
    private let session: SessionController
    private let gameWatcher: GameWatcher

    init(
        playController: PlayController,
        session: SessionController,
        gameWatcher: GameWatcher
    ) {
        self.playController = playController
        self.session = session
        self.gameWatcher = gameWatcher
        let view = HumanPlayWindowView(
            playController: playController,
            session: session,
            gameWatcher: gameWatcher
        )
        let hosting = NSHostingController(rootView: view)
        let window = NSWindow(contentViewController: hosting)
        window.setContentSize(NSSize(width: 720, height: 860))
        window.minSize = NSSize(width: 520, height: 660)
        window.title = "Chess — Human vs Network"
        // `isReleasedWhenClosed = false` lets the controller manage
        // the window's lifetime via the registry rather than handing
        // it over to AppKit's release-on-close behavior — needed for
        // the `windowWillClose` delegate to fire safely and for the
        // registry to be the single source of truth on whether a
        // window is currently open.
        window.isReleasedWhenClosed = false
        window.center()
        super.init(window: window)
        window.delegate = self
    }

    required init?(coder: NSCoder) {
        fatalError("init(coder:) not supported for HumanPlayWindowController")
    }

    func windowWillClose(_ notification: Notification) {
        // Closing the window terminates the game in flight. `stop` is
        // idempotent — if the game already ended (checkmate / draw /
        // user clicked Stop Game), this is a no-op.
        if playController.isPlayingHuman {
            playController.stop(gameWatcher: gameWatcher)
        }
        HumanPlayWindowRegistry.shared.unregister(self)
    }
}

/// One-window-at-a-time strong-reference holder for the human-play
/// window. Matches `LogAnalysisWindowRegistry`'s shape but enforces
/// a single instance because multi-window human play would require
/// per-window PlayController state (currently shared).
@MainActor
final class HumanPlayWindowRegistry {
    static let shared = HumanPlayWindowRegistry()
    private var current: HumanPlayWindowController?

    private init() {}

    func register(_ controller: HumanPlayWindowController) {
        current = controller
    }

    func unregister(_ controller: HumanPlayWindowController) {
        if current === controller {
            current = nil
        }
    }

    func focusExisting() -> Bool {
        guard let c = current else { return false }
        c.window?.makeKeyAndOrderFront(nil)
        return true
    }
}

/// Bridges `PlayController.start(...)`'s success path to a freshly
/// opened (or already-open and refocused) human-play window. Mirrors
/// the static-launcher pattern from `LogAnalysisLauncher` so the
/// `PlayController` doesn't need to import AppKit.
@MainActor
enum HumanPlayWindowLauncher {
    static func openOrFocus(
        controller: PlayController,
        session: SessionController,
        gameWatcher: GameWatcher
    ) {
        // A second Start while a window is already open (e.g. via
        // Reset) re-uses the existing window — the registry's single-
        // instance invariant matches `PlayController`'s
        // single-game-at-a-time invariant.
        if HumanPlayWindowRegistry.shared.focusExisting() {
            return
        }
        let win = HumanPlayWindowController(
            playController: controller,
            session: session,
            gameWatcher: gameWatcher
        )
        HumanPlayWindowRegistry.shared.register(win)
        win.showWindow(nil)
        win.window?.makeKeyAndOrderFront(nil)
    }
}

// MARK: - SwiftUI content

/// SwiftUI content for the human-play window. Three stacked regions:
///   - top banner: big game-over message (with the specific draw
///     reason or the winning side), or a CHECK call-out while a
///     game is in progress and the side-to-move's king is attacked,
///     or a smaller "Your move / Network thinking…" status.
///   - middle: the human-play board (own dedicated view —
///     `HumanPlayBoardView` — for animated pieces, last-move and
///     in-check highlights).
///   - bottom: an info row (ply count, material totals + advantage,
///     last move in algebraic notation) and the toolbar (Reset /
///     Stop, build stamp).
///
/// State sources:
///   - `playController` (`@Bindable`): reactive — selected from-
///     square, legal-target highlights, pending promotion, the
///     `isPlayingHuman` flag.
///   - `gameWatcher` (polled): live board, side-to-move, move count,
///     last applied move, end-of-game `GameResult`, and last game's
///     stats. Polled on a 100 ms timer because the watcher fires its
///     mutations from a non-SwiftUI dispatch queue and is
///     intentionally not `@Observable` (the project's UI decouples
///     redraw from game-loop rate).
fileprivate struct HumanPlayWindowView: View {
    @Bindable var playController: PlayController
    let session: SessionController
    let gameWatcher: GameWatcher

    /// Mirrored snapshot of the watcher. Refreshed by the
    /// `Combine`-driven polling subscription wired in via
    /// `.onReceive(pollTimer)` below. Seeded from the watcher in
    /// `.onAppear` so the window doesn't flash the default starting
    /// position before the first publisher tick lands.
    @State private var snapshot: GameWatcher.Snapshot = .init()

    /// 10 Hz polling timer for the watcher snapshot. Matches the
    /// `snapshotTimer` pattern used in `UpperContentView`.
    private let pollTimer = Timer.publish(
        every: 0.1, on: .main, in: .common
    ).autoconnect()

    var body: some View {
        VStack(spacing: 12) {
            bannerRow
            boardView
            statusRow
            toolbarRow
        }
        .padding(16)
        .frame(minWidth: 520, minHeight: 660)
        .onAppear { snapshot = gameWatcher.snapshot() }
        .onReceive(pollTimer) { _ in refreshSnapshot() }
    }

    // MARK: - Top banner

    /// Centered top banner with two stacked lines: a big primary
    /// line (game result / CHECK / status) and a small subtitle
    /// line (who the human is playing as). The big line's content
    /// changes but its minimum height is fixed so the board doesn't
    /// jump up/down as the message changes.
    private var bannerRow: some View {
        VStack(spacing: 2) {
            bannerPrimaryText
                .frame(maxWidth: .infinity)
                .frame(minHeight: 40)
            Text(humanLabel)
                .font(.subheadline)
                .foregroundStyle(.secondary)
        }
    }

    @ViewBuilder
    private var bannerPrimaryText: some View {
        if let result = snapshot.result {
            Text(gameOverText(result))
                .font(.title.weight(.bold))
                .foregroundStyle(.primary)
                .multilineTextAlignment(.center)
        } else if computeInCheckColor() != nil {
            Text("CHECK")
                .font(.title.weight(.bold))
                .foregroundStyle(.red)
        } else if !playController.isPlayingHuman {
            Text("Waiting…")
                .font(.title3)
                .foregroundStyle(.secondary)
        } else if !playController.pendingLegalMoves.isEmpty {
            Text("Your move")
                .font(.title3)
                .foregroundStyle(.secondary)
        } else {
            Text("Network thinking…")
                .font(.title3)
                .foregroundStyle(.secondary)
        }
    }

    private var humanLabel: String {
        playController.humanColor == .white ? "You play White" : "You play Black"
    }

    /// Full game-over banner text. Includes the specific draw
    /// reason ("Draw by 50-move rule", "Draw by threefold
    /// repetition", etc.) so the user knows *why* the game ended.
    private func gameOverText(_ r: GameResult) -> String {
        switch r {
        case .checkmate(let winner):
            return winner == .white ? "White wins by checkmate" : "Black wins by checkmate"
        case .stalemate:
            return "Draw by stalemate"
        case .drawByFiftyMoveRule:
            return "Draw by 50-move rule"
        case .drawByInsufficientMaterial:
            return "Draw by insufficient material"
        case .drawByThreefoldRepetition:
            return "Draw by threefold repetition"
        }
    }

    /// Color of the side that's currently in check, or nil if
    /// neither side is. After `didApplyMove`, `state.currentPlayer`
    /// is the side that just received the move — the one whose
    /// king might be in check. The game-over case is excluded
    /// because checkmate already produces a definitive banner.
    private func computeInCheckColor() -> PieceColor? {
        guard snapshot.result == nil else { return nil }
        let p = snapshot.state.currentPlayer
        return MoveGenerator.isInCheck(snapshot.state, color: p) ? p : nil
    }

    // MARK: - Board

    /// The board, oriented so the human's pieces sit at the bottom.
    /// All square indices passed into `HumanPlayBoardView` are in
    /// visual coordinates (already 180°-flipped for a black-playing
    /// human); the tap callback inverts the flip back to logical
    /// before handing the square to `PlayController.tapSquare`.
    private var boardView: some View {
        let humanColor = playController.humanColor
        let humanBoardFlipped = (humanColor == .black)
        let pieces: [Piece?] = humanBoardFlipped
            ? Array(snapshot.state.board.reversed())
            : snapshot.state.board
        let selectedVisual: Int? = playController.selectedFromSquare.map { sq in
            humanBoardFlipped ? 63 - sq : sq
        }
        let humanPlayActive = playController.isPlayingHuman
            && !playController.pendingLegalMoves.isEmpty
        let legalTargetsVisual: Set<Int> = humanPlayActive
            ? Self.legalTargetsVisual(
                from: playController.selectedFromSquare,
                pending: playController.pendingLegalMoves,
                flipped: humanBoardFlipped
            )
            : []
        let promotionVisualSquare: Int? = playController.pendingPromotion.map { p in
            let logical = p.toRow * 8 + p.toCol
            return humanBoardFlipped ? 63 - logical : logical
        }
        let lastToVisual: Int? = snapshot.lastMove.map { mv in
            let logical = mv.toRow * 8 + mv.toCol
            return humanBoardFlipped ? 63 - logical : logical
        }
        let checkVisual: Int? = inCheckKingVisualSquare(flipped: humanBoardFlipped)
        return HumanPlayBoardView(
            pieces: pieces,
            selectedFromSquare: selectedVisual,
            legalMoveTargets: legalTargetsVisual,
            lastMoveDestinationSquare: lastToVisual,
            checkSquare: checkVisual,
            humanMoveActive: humanPlayActive,
            humanColor: playController.isPlayingHuman ? humanColor : nil,
            pendingPromotion: playController.pendingPromotion,
            promotionVisualSquare: promotionVisualSquare,
            onTapSquare: { visualSq in
                let logical = humanBoardFlipped ? 63 - visualSq : visualSq
                playController.tapSquare(logical, in: snapshot.state.board)
            },
            onSelectPromotion: { type in
                playController.selectPromotion(type)
            },
            onCancelPromotion: {
                playController.cancelPromotion()
            }
        )
    }

    /// Visual square (0..<64) of the king belonging to the side in
    /// check, or nil if neither side is in check. Used to drive the
    /// red board fill in `HumanPlayBoardView`.
    private func inCheckKingVisualSquare(flipped: Bool) -> Int? {
        guard let color = computeInCheckColor() else { return nil }
        for i in 0..<64 {
            if let p = snapshot.state.board[i], p.type == .king, p.color == color {
                return flipped ? 63 - i : i
            }
        }
        return nil
    }

    // MARK: - Status row (ply / material / last move)

    private var statusRow: some View {
        HStack(alignment: .center, spacing: 18) {
            statusBlock(title: "Ply", value: plyText)
            Divider().frame(height: 32)
            materialBlock
            Divider().frame(height: 32)
            statusBlock(title: "Last move", value: lastMoveText)
            Spacer(minLength: 0)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func statusBlock(title: String, value: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title)
                .font(.caption2)
                .foregroundStyle(.secondary)
            Text(value)
                .font(.body.monospacedDigit())
        }
    }

    private var materialBlock: some View {
        let (white, black) = materialCounts(snapshot.state.board)
        let advantage = white - black
        let advText: String
        if advantage > 0 {
            advText = "W +\(advantage)"
        } else if advantage < 0 {
            advText = "B +\(abs(advantage))"
        } else {
            advText = "even"
        }
        return VStack(alignment: .leading, spacing: 2) {
            Text("Material")
                .font(.caption2)
                .foregroundStyle(.secondary)
            HStack(spacing: 6) {
                Text("W \(white)   B \(black)")
                    .font(.body.monospacedDigit())
                Text("(\(advText))")
                    .font(.body.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
        }
    }

    /// Ply count to display. Mid-game: `moveCount` (the live ply
    /// counter). Post-game: `lastGameStats.totalMoves` (the final
    /// ply count, because `GameWatcher` zeros `moveCount` at game
    /// end so the session-cumulative `totalMoves + moveCount` math
    /// elsewhere doesn't double-count).
    private var plyText: String {
        if snapshot.result != nil, let stats = snapshot.lastGameStats {
            return "\(stats.totalMoves)"
        }
        return "\(snapshot.moveCount)"
    }

    private var lastMoveText: String {
        snapshot.lastMove?.notation ?? "—"
    }

    /// Standard piece-value sums (P=1, N=3, B=3, R=5, Q=9). King is
    /// not counted — there is always exactly one king per side and
    /// including it would only inflate both totals by the same
    /// constant, flattening the displayed advantage.
    private func materialCounts(_ board: [Piece?]) -> (white: Int, black: Int) {
        var w = 0
        var b = 0
        for square in board {
            guard let p = square else { continue }
            let v: Int
            switch p.type {
            case .pawn:   v = 1
            case .knight: v = 3
            case .bishop: v = 3
            case .rook:   v = 5
            case .queen:  v = 9
            case .king:   v = 0
            }
            if p.color == .white { w += v } else { b += v }
        }
        return (w, b)
    }

    // MARK: - Toolbar

    private var toolbarRow: some View {
        HStack(spacing: 12) {
            // Reset only re-launches an in-flight game's opponent
            // settings (`PlayController.reset` stops + starts using
            // `lastOpponentChoice` / `lastHumanColor`, both cleared
            // at game-end cleanup). Mirror the Chess menu's gate:
            // enabled only while a game is actually running.
            Button("Reset Game") {
                playController.reset(session: session, gameWatcher: gameWatcher)
            }
            .disabled(!playController.isPlayingHuman)
            Button("Stop Game") {
                playController.stop(gameWatcher: gameWatcher)
            }
            .disabled(!playController.isPlayingHuman)
            Spacer()
            Text("Build \(BuildInfo.buildNumber) · \(BuildInfo.gitHash)")
                .font(.caption2.monospacedDigit())
                .foregroundStyle(.tertiary)
        }
    }

    // MARK: - Polling

    /// Pull the latest snapshot from the watcher and mirror it to
    /// `@State` if any user-visible field changed. `GameState` and
    /// `GameResult` aren't `Equatable` (only `Sendable`), so the
    /// snapshot can't be compared wholesale — dedup on the fields
    /// that actually drive the visible UI.
    private func refreshSnapshot() {
        let s = gameWatcher.snapshot()
        let resultChanged: Bool = {
            switch (snapshot.result, s.result) {
            case (nil, nil): return false
            case (nil, _), (_, nil): return true
            default: return false
            }
        }()
        if s.state.board != snapshot.state.board
            || s.moveCount != snapshot.moveCount
            || s.isPlaying != snapshot.isPlaying
            || s.lastMove != snapshot.lastMove
            || resultChanged
        {
            snapshot = s
        }
    }

    // MARK: - Helpers

    /// Project legal moves into visual coordinates given the
    /// selected from-square and whether the board is rendered 180°
    /// rotated (human plays black).
    static func legalTargetsVisual(
        from: Int?,
        pending: [ChessMove],
        flipped: Bool
    ) -> Set<Int> {
        guard let from else { return [] }
        let fromRow = from / 8
        let fromCol = from % 8
        var out: Set<Int> = []
        for move in pending where move.fromRow == fromRow && move.fromCol == fromCol {
            let logical = move.toRow * 8 + move.toCol
            out.insert(flipped ? 63 - logical : logical)
        }
        return out
    }
}
