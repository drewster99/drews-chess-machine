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
        window.setContentSize(NSSize(width: 720, height: 800))
        window.minSize = NSSize(width: 480, height: 560)
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

/// SwiftUI content for the human-play window. Renders the game
/// board, a toolbar with Reset / Stop / Resign-or-game-over status,
/// and the same promotion picker overlay used by the inline board in
/// the main window.
///
/// State sources:
///   - `playController` (`@Bindable`): reactive — selected from-
///     square, legal-target highlights, pending promotion, the
///     `isPlayingHuman` flag.
///   - `gameWatcher` (polled): the live board, side-to-move, move
///     count, and end-of-game `GameResult`. Polled on a 100 ms timer
///     because the watcher fires its mutations from a non-SwiftUI
///     dispatch queue and is intentionally not `@Observable` (the
///     project's UI decouples redraw from game-loop rate).
///
/// Animation: the board's `pieces` array is the animation key for a
/// short `.easeInOut` transition, so every move (AI's 2-second-
/// delayed reply OR the human's own move after submission) cross-
/// fades the affected squares. A piece truly "sliding" between
/// squares would need `matchedGeometryEffect` per piece identity
/// (not done here to keep the change small); the cross-fade is enough
/// to register "something moved on the board".
fileprivate struct HumanPlayWindowView: View {
    @Bindable var playController: PlayController
    let session: SessionController
    let gameWatcher: GameWatcher

    /// Mirrored snapshot of the watcher. Refreshed by the
    /// `Combine`-driven polling subscription wired in via
    /// `.onReceive(pollTimer)` below; SwiftUI redraws when this
    /// value's `state.board` or other observed fields change. Seeded
    /// from the watcher in `.onAppear` so the window doesn't flash
    /// the default starting position before the first publisher tick
    /// lands.
    @State private var snapshot: GameWatcher.Snapshot = .init()

    /// 10 Hz polling timer for the watcher snapshot. SwiftUI manages
    /// the subscription lifecycle through `.onReceive` — the
    /// publisher is created with the view struct and torn down when
    /// the view disappears. Matches the `snapshotTimer` pattern used
    /// in `UpperContentView` (and avoids the `@State<Timer?>`
    /// Sendable headache that the manual `Timer.scheduledTimer` path
    /// would land us in under Swift 6 strict concurrency).
    private let pollTimer = Timer.publish(
        every: 0.1, on: .main, in: .common
    ).autoconnect()

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            statusRow
            boardView
            toolbarRow
        }
        .padding(16)
        .frame(minWidth: 480, minHeight: 560)
        .onAppear { snapshot = gameWatcher.snapshot() }
        .onReceive(pollTimer) { _ in refreshSnapshot() }
    }

    // MARK: status row

    private var statusRow: some View {
        HStack(spacing: 8) {
            Text(humanLabel)
                .font(.headline)
            Spacer()
            Text(statusText)
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .monospacedDigit()
        }
    }

    private var humanLabel: String {
        playController.humanColor == .white ? "You play White" : "You play Black"
    }

    private var statusText: String {
        if let result = snapshot.result {
            return "Game over — \(describe(result))"
        }
        if !playController.isPlayingHuman {
            return "Waiting…"
        }
        if !playController.pendingLegalMoves.isEmpty {
            return "Your move (move \(snapshot.moveCount + 1))"
        }
        return "Network thinking… (move \(snapshot.moveCount + 1))"
    }

    private func describe(_ r: GameResult) -> String {
        switch r {
        case .checkmate(let winner):
            return winner == .white ? "White wins" : "Black wins"
        case .stalemate:
            return "Draw (stalemate)"
        case .drawByFiftyMoveRule:
            return "Draw (50-move rule)"
        case .drawByInsufficientMaterial:
            return "Draw (insufficient material)"
        case .drawByThreefoldRepetition:
            return "Draw (threefold repetition)"
        }
    }

    // MARK: board

    /// The board, oriented so the human's pieces sit at the bottom.
    /// Click routing translates visual → logical (the encoder frame)
    /// before reaching `PlayController.tapSquare`.
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
        // Disambiguate via a typed local because the call site has
        // many parameters and Swift's overload resolution occasionally
        // mis-parses bare `.none` as `Optional.none` (= nil) against
        // the non-Optional enum case. Belt + suspenders.
        let noneOverlay: ChessBoardView.Overlay = .none
        return LiveBoardWithNavigationView(
            pieces: pieces,
            overlay: noneOverlay,
            selectedOverlay: -1,
            inferenceResultPresent: false,
            forwardPassEditable: false,
            realTraining: false,
            isCandidateTestActive: false,
            workerCount: 1,
            onNavigate: { _ in },
            onApplyFreePlacementDrag: { _, _ in },
            squareIndex: { point, size in
                Self.squareIndex(at: point, boardSize: size)
            },
            onHoverSquare: { _ in },
            humanMoveActive: humanPlayActive,
            selectedFromSquare: selectedVisual,
            legalMoveTargets: legalTargetsVisual,
            pendingPromotion: playController.pendingPromotion,
            humanColor: playController.isPlayingHuman ? humanColor : nil,
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
        // Animate the cross-fade between consecutive positions. Key
        // on the encoder-frame board so a Reset (which replaces the
        // whole array) animates too. Duration kept short so a human
        // move feels immediate while the AI's response (already
        // 2 seconds delayed) cross-fades cleanly into the new
        // position.
        .animation(.easeInOut(duration: 0.35), value: snapshot.state.board)
        .aspectRatio(1, contentMode: .fit)
    }

    // MARK: toolbar

    private var toolbarRow: some View {
        HStack(spacing: 12) {
            // Reset only re-launches an in-flight game's opponent
            // settings (`PlayController.reset` stops + starts using
            // `lastOpponentChoice` / `lastHumanColor`, both of which
            // are cleared at game-end cleanup). Mirror the Chess menu's
            // gate: enabled only while a human game is actually
            // running — post-game the user re-opens Chess > Play….
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

    // MARK: polling

    /// Pull the latest snapshot from the watcher and mirror it to
    /// `@State` if any user-visible field changed. Fired by the
    /// `.onReceive(pollTimer)` subscription in `body`. `GameState` and
    /// `GameResult` aren't `Equatable` (only `Sendable`), so the
    /// snapshot can't be compared wholesale — dedup on the fields
    /// that actually drive the visible UI: the board array
    /// (`Piece` is `Hashable`, so `[Piece?]` is `Equatable`), the
    /// move count, the play flag, and a simple "result-nil-or-not"
    /// signal. A missed update (e.g. a draw-by-50-move reason
    /// changing without the move count moving) is impossible — the
    /// move count changes monotonically across any game-state
    /// change, and the `result-nil-or-not` flip catches the
    /// game-end transition.
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
            || resultChanged
        {
            snapshot = s
        }
    }

    // MARK: helpers (mirrors of UpperContentView's private statics —
    // inlined here so this file doesn't have to widen those helpers'
    // access for one call site each)

    /// Project `pending` legal moves into visual coordinates given the
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

    /// Convert a point in the board-overlay's local coordinate space
    /// to a 0–63 square index, or nil if outside the board.
    static func squareIndex(at point: CGPoint, boardSize: CGFloat) -> Int? {
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
}
