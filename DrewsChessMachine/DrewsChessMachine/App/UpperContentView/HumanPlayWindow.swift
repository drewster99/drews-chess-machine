import AppKit
import Combine
import SwiftUI

/// Standalone window that hosts a human-vs-network game. Created by
/// `HumanPlayWindowLauncher.openOrFocus(...)` once
/// `PlayController.materializeTask` succeeds; closed either by the
/// user (X button), by `Stop Game` from the in-window toolbar or
/// the Chess menu, or by a Reset (which re-opens with a fresh game).
///
/// Owns no game state directly: the rendered board, side-to-move,
/// legal-move highlights, and pending promotion all read from the
/// shared `PlayController` (`@MainActor @Observable`) and from the
/// per-game `HumanPlayPacer` (also `@MainActor @Observable`) that
/// the controller owns. `GameWatcher.changes` events are handed
/// directly to the pacer's `ingest(...)`; the pacer's state machine
/// decides whether each watcher snapshot advances the displayed
/// board, so the user sees per-ply animations in strict order even
/// when the main actor is laggy. The window's lifecycle is owned by
/// the controller + registry pattern used elsewhere in the project
/// (see `LogAnalysisWindowController`): the registry holds the
/// strong reference so the controller doesn't dealloc the moment
/// SwiftUI lets go of the hosting view, and the controller
/// unregisters in `windowWillClose`.
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
        // The move-list strip on the right needs
        // `HumanPlayWindowView.historyPanelWidth` of fixed width
        // without squeezing the board into an unreadable size.
        window.setContentSize(NSSize(width: 920, height: 860))
        window.minSize = NSSize(width: 720, height: 660)
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
///     `isPlayingHuman` flag, and the per-game `HumanPlayPacer`.
///   - `playController.pacer.displayedSnapshot` (`@Observable`):
///     the board / status snapshot the user is currently looking at.
///     Distinct from `gameWatcher.snapshot()` — the watcher may have
///     already absorbed the AI's reply before the pacer has
///     permitted the user's move to finish animating, and the pacer
///     suppresses those premature updates so the board stays
///     in-order.
///   - `gameWatcher.changes`: every emission is handed verbatim to
///     `pacer.ingest(...)` (no `.throttle`); the pacer's state
///     machine decides whether each snapshot actually advances the
///     display.
fileprivate struct HumanPlayWindowView: View {
    @Bindable var playController: PlayController
    let session: SessionController
    let gameWatcher: GameWatcher

    /// Currently-selected ply in the move-history sidebar. Set by
    /// tapping a half-move cell; cleared after a successful Revert
    /// to here. Plies are 1-based and match `HumanPlayPacer.HistoryEntry.plyNumber`.
    @State private var selectedHistoryPly: Int?

    /// Empty snapshot used when no game is active and
    /// `playController.pacer` is `nil` (e.g., the window briefly
    /// outlives a `stop()`). The view's body uses `??` to fall back
    /// to this so layout stays valid in that transient state.
    private static let emptySnapshot = GameWatcher.Snapshot()

    /// Convenience: the snapshot the board + banner + status row
    /// render from. Sourced from the pacer rather than directly from
    /// the watcher so the user sees moves in strict ply order.
    private var snapshot: GameWatcher.Snapshot {
        playController.pacer?.displayedSnapshot ?? Self.emptySnapshot
    }

    /// Fixed strip width for the move-history panel that sits to the
    /// right of the board. Sized to comfortably fit two columns of
    /// algebraic-notation entries (e.g. "23. e2e4=Q  Qxa1#") at the
    /// monospaced body font without truncation, with room for the
    /// header and a vertical scroller.
    static let historyPanelWidth: CGFloat = 200

    var body: some View {
        VStack(spacing: 12) {
            bannerRow
            HStack(alignment: .top, spacing: 12) {
                boardView
                moveHistoryPanel
                    .frame(width: Self.historyPanelWidth)
            }
            statusRow
            tauControlRow
            toolbarRow
        }
        .padding(16)
        .frame(minWidth: 720, minHeight: 660)
        // `.onReceive` is intentionally un-throttled: this window owns
        // its own `GameWatcher` instance (one watcher per Human-vs-
        // Network game), so emissions land at human-pacing rates
        // rather than self-play / arena rates. `pacer.ingest` is
        // cheap (a state-machine `switch`) and `gameWatcher.snapshot()`
        // is a single locked read of a small struct.
        // `.receive(on: DispatchQueue.main)` stays because `.onReceive`
        // doesn't guarantee main-actor delivery on every Combine source.
        .onReceive(gameWatcher.changes.receive(on: DispatchQueue.main)) { _ in
            playController.pacer?.ingest(gameWatcher.snapshot())
        }
    }

    // MARK: - Top banner

    /// Centered top banner with three stacked lines: a big primary
    /// line (game result / CHECK / status), a small subtitle line
    /// (who the human is playing as), and a metadata row showing
    /// the half-move clock + threefold counter so the user can see
    /// how close the position is to a 50-move or 3-fold draw. The
    /// big line's content changes but its minimum height is fixed
    /// so the board doesn't jump up/down as the message changes.
    private var bannerRow: some View {
        VStack(spacing: 2) {
            bannerPrimaryText
                .frame(maxWidth: .infinity)
                .frame(minHeight: 40)
            Text(humanLabel)
                .font(.subheadline)
                .foregroundStyle(.secondary)
            opponentLabelText
            drawCountersRow
        }
    }

    /// Single line under the "You play White/Black" subtitle that
    /// surfaces which network the AI side is using — champion or
    /// trainer snapshot at its ModelID, live trainer, or the
    /// filename of a loaded `.dcmmodel`. Sourced from
    /// `PlayController.currentOpponentDescription`, which is set
    /// in `materializeOpponentSource(...)` after the inference
    /// network is built (so the ModelID reflects the actual
    /// game-start snapshot rather than whatever the live champion
    /// has drifted to since). Hidden when the controller has no
    /// description (between games or before the network has
    /// materialized).
    @ViewBuilder
    private var opponentLabelText: some View {
        if let description = playController.currentOpponentDescription {
            (Text("vs ") + Text(description).fontDesign(.monospaced))
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    /// One-line row directly under the human-color label. Surfaces the
    /// half-move clock (toward the 50-move rule's 100-ply limit) and the
    /// current position's repetition count (toward the 3-fold draw at 3
    /// visits). Color-coded so the user notices when either approaches
    /// its draw threshold:
    ///   * halfmoveClock ≥ 80 → orange ("draw window opening")
    ///   * halfmoveClock ≥ 95 → red ("imminent")
    ///   * repetitionCount ≥ 1 → orange ("one visit short of forcing 3-fold")
    /// `repetitionCount` on `GameState` is occurrences BEFORE the current
    /// visit, so the displayed "visits" is `repetitionCount + 1` (the +1
    /// accounts for the current visit being on the board).
    private var drawCountersRow: some View {
        let hmc = snapshot.state.halfmoveClock
        let visits = snapshot.state.repetitionCount + 1
        let hmcColor: Color = {
            switch hmc {
            case 95...: return .red
            case 80...: return .orange
            default: return .secondary
            }
        }()
        let visitsColor: Color = visits >= 2 ? .orange : .secondary
        return HStack(spacing: 12) {
            Text("Half-move clock: ")
                .foregroundStyle(.secondary)
                + Text("\(hmc)/100")
                .foregroundStyle(hmcColor)
                .monospacedDigit()
            Text("Threefold: ")
                .foregroundStyle(.secondary)
                + Text("\(visits)/3")
                .foregroundStyle(visitsColor)
                .monospacedDigit()
        }
        .font(.caption)
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
    /// king might be in check.
    ///
    /// We deliberately do NOT bail out when `snapshot.result` is set:
    /// for `.checkmate`, the mated side IS still in check (that's the
    /// definition), and the user wants the red king square to stay
    /// visible through the result transition so the visual makes
    /// clear which king got mated. For `.stalemate` and
    /// `.drawByInsufficientMaterial` the side to move is by
    /// definition not in check, so `MoveGenerator.isInCheck` returns
    /// false and this method naturally returns nil → no red square.
    /// `.drawByFiftyMoveRule` and `.drawByThreefoldRepetition` are
    /// the edge cases that *can* coincide with a check (the engine
    /// fires those checks unconditionally — see ChessGameEngine
    /// lines 242, 247); we still show red in that case, which is
    /// accurate (the king genuinely IS in check, the rules just say
    /// the game is drawn anyway).
    private func computeInCheckColor() -> PieceColor? {
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
        let lastFromVisual: Int? = snapshot.lastMove.map { mv in
            let logical = mv.fromRow * 8 + mv.fromCol
            return humanBoardFlipped ? 63 - logical : logical
        }
        let checkVisual: Int? = inCheckKingVisualSquare(flipped: humanBoardFlipped)
        return HumanPlayBoardView(
            pieces: pieces,
            selectedFromSquare: selectedVisual,
            legalMoveTargets: legalTargetsVisual,
            lastMoveDestinationSquare: lastToVisual,
            lastMoveSourceSquare: lastFromVisual,
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
            },
            onAnimationCompleted: {
                playController.pacer?.onAnimationCompleted()
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

    // MARK: - Move history panel

    /// Right-side scrollable list of all moves made in the current
    /// game (or the just-finished game if `phase == .gameOver`).
    /// Auto-scrolls so the latest pair is always visible. Reads
    /// straight from the pacer's `history`, which is the same list
    /// that drove `displayedSnapshot.lastMove` for each row — the user
    /// sees every move on the board and in the list in the same
    /// order.
    private var moveHistoryPanel: some View {
        let history = playController.pacer?.history ?? []
        let pairs = Self.pairRows(from: history)
        let totalPlies = history.count
        let canRevertSelected = (selectedHistoryPly.map { $0 >= 1 && $0 < totalPlies }) ?? false
        return VStack(alignment: .leading, spacing: 4) {
            Text("Moves")
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
            ScrollViewReader { proxy in
                ScrollView(.vertical, showsIndicators: true) {
                    LazyVStack(alignment: .leading, spacing: 2) {
                        ForEach(pairs) { row in
                            moveHistoryRow(row)
                                .id(row.id)
                        }
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal, 6)
                    .padding(.vertical, 4)
                }
                .background(
                    RoundedRectangle(cornerRadius: 6)
                        .fill(Color.secondary.opacity(0.08))
                )
                .overlay(
                    RoundedRectangle(cornerRadius: 6)
                        .stroke(Color.secondary.opacity(0.25), lineWidth: 1)
                )
                .onChange(of: pairs.last?.id) {
                    guard let lastID = pairs.last?.id else { return }
                    withAnimation(.easeOut(duration: 0.15)) {
                        proxy.scrollTo(lastID, anchor: .bottom)
                    }
                }
                // Drop the selection if the history shrank past it
                // (typically because a fresh game / revert reset the
                // pacer's history list). Without this, a stale
                // highlight would survive across games and the
                // Revert button would either dim silently or operate
                // on a ply that no longer exists.
                .onChange(of: totalPlies) {
                    if let sel = selectedHistoryPly, sel > totalPlies {
                        selectedHistoryPly = nil
                    }
                }
            }
            Button(
                action: {
                    guard let ply = selectedHistoryPly else { return }
                    playController.revertToHistoryPly(
                        ply,
                        session: session,
                        gameWatcher: gameWatcher
                    )
                    selectedHistoryPly = nil
                },
                label: {
                    Text("Revert to here")
                        .frame(maxWidth: .infinity)
                }
            )
            .controlSize(.small)
            .disabled(!canRevertSelected)
            .help(canRevertSelected
                ? "Remove every move played after ply \(selectedHistoryPly ?? 0) and resume from there."
                : "Tap any move except the last to enable Revert to here."
            )
        }
    }

    private func moveHistoryRow(_ row: MoveRow) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 6) {
            Text("\(row.moveNumber).")
                .font(.system(.callout, design: .monospaced))
                .foregroundStyle(.secondary)
                .frame(width: 28, alignment: .trailing)
            moveHistoryHalfCell(text: row.whiteText, ply: row.whitePly)
            moveHistoryHalfCell(text: row.blackText, ply: row.blackPly)
        }
    }

    /// A single half-move cell. Tappable when it carries a ply (an
    /// actual played move): tap sets `selectedHistoryPly` to that
    /// ply, which (a) highlights the cell and (b) enables the Revert
    /// to here button below the list. Empty cells (the unanswered
    /// white half before black has replied) stay rendered for
    /// alignment but are disabled.
    @ViewBuilder
    private func moveHistoryHalfCell(text: String, ply: Int?) -> some View {
        let isSelected = ply != nil && ply == selectedHistoryPly
        Button(
            action: {
                guard let ply else { return }
                // Toggle: tapping the already-selected cell deselects
                // it. Avoids needing a separate Cancel control.
                if selectedHistoryPly == ply {
                    selectedHistoryPly = nil
                } else {
                    selectedHistoryPly = ply
                }
            },
            label: {
                Text(text)
                    .font(.system(.callout, design: .monospaced))
                    .foregroundStyle(text.isEmpty ? Color.clear : Color.primary)
                    .frame(maxWidth: .infinity, minHeight: 18, alignment: .leading)
                    .padding(.horizontal, 4)
                    .background(
                        RoundedRectangle(cornerRadius: 3)
                            .fill(isSelected
                                ? Color.accentColor.opacity(0.30)
                                : Color.clear)
                    )
                    .contentShape(Rectangle())
            }
        )
        .buttonStyle(.plain)
        .disabled(ply == nil || text.isEmpty)
    }

    /// One row of the paired move-history list: "1. e2-e4  e7-e5".
    /// `blackText` is empty when only the white half of the pair has
    /// been played (the row appears the moment white moves; the black
    /// half fills in when black responds). `whitePly` / `blackPly`
    /// carry the 1-based ply numbers from `HumanPlayPacer.HistoryEntry`
    /// for half-move selection; nil when that half wasn't played.
    private struct MoveRow: Identifiable, Equatable {
        let id: Int
        let moveNumber: Int
        let whitePly: Int?
        let blackPly: Int?
        let whiteText: String
        let blackText: String
    }

    /// Pair an unbounded history of `HistoryEntry` (one per ply) into
    /// per-full-move rows. White always starts a new row; an
    /// unanswered white move sits alone with empty `blackText`. If the
    /// game began with a black move (engine quirk / loaded position),
    /// the leading half-move pairs into row 1 with an empty white
    /// slot rather than splitting it into a single row that contains
    /// only the black half.
    private static func pairRows(from entries: [HumanPlayPacer.HistoryEntry]) -> [MoveRow] {
        var rows: [MoveRow] = []
        var currentNumber = 0
        var currentWhite = ""
        var currentBlack = ""
        var currentWhitePly: Int?
        var currentBlackPly: Int?
        var haveOpen = false

        for entry in entries {
            if entry.side == .white {
                if haveOpen {
                    rows.append(MoveRow(
                        id: currentNumber,
                        moveNumber: currentNumber,
                        whitePly: currentWhitePly,
                        blackPly: currentBlackPly,
                        whiteText: currentWhite,
                        blackText: currentBlack
                    ))
                }
                currentNumber = rows.count + 1
                currentWhite = entry.move.notation
                currentWhitePly = entry.plyNumber
                currentBlack = ""
                currentBlackPly = nil
                haveOpen = true
            } else {
                if !haveOpen {
                    currentNumber = rows.count + 1
                    currentWhite = ""
                    currentWhitePly = nil
                    haveOpen = true
                }
                currentBlack = entry.move.notation
                currentBlackPly = entry.plyNumber
                rows.append(MoveRow(
                    id: currentNumber,
                    moveNumber: currentNumber,
                    whitePly: currentWhitePly,
                    blackPly: currentBlackPly,
                    whiteText: currentWhite,
                    blackText: currentBlack
                ))
                haveOpen = false
            }
        }
        if haveOpen {
            rows.append(MoveRow(
                id: currentNumber,
                moveNumber: currentNumber,
                whitePly: currentWhitePly,
                blackPly: currentBlackPly,
                whiteText: currentWhite,
                blackText: currentBlack
            ))
        }
        return rows
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

    /// Ply count to display. Always the displayed game length — the
    /// number of plies in the pacer's `history` (the move list), which
    /// is seeded with any reverted-in prefix and grows as play continues.
    ///
    /// Earlier this switched to `lastGameStats.totalMoves` once the game
    /// ended; that under-reported any reverted game, because the engine's
    /// per-loop counters restart at 0 after a Revert (the `runGameLoop`
    /// total excludes the seeded prefix) while the move list, board, and
    /// live counter all include it. `HumanPlayPlyReadout` documents the
    /// choice; `history.count` keeps "Ply" equal to the move list in
    /// every state — mid-game, game-over, reverted or not.
    private var plyText: String {
        let displayedHistoryPlies = playController.pacer?.history.count ?? snapshot.moveCount
        let engineLoopTotalMoves = snapshot.result != nil ? snapshot.lastGameStats?.totalMoves : nil
        return "\(HumanPlayPlyReadout.displayedPlyCount(displayedHistoryPlies: displayedHistoryPlies, engineLoopTotalMoves: engineLoopTotalMoves))"
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

    // MARK: - AI sampling temperature

    /// Slider + monospaced readout for the AI's per-ply sampling
    /// temperature. Lives outside the toolbar so the slider has room
    /// to breathe. Updates live: the AI's `MPSChessPlayer` re-reads
    /// from the shared SyncBox at the top of every `sampleMove`, so a
    /// slider move between the user's submission and the AI's next
    /// move takes effect immediately. Persists across launches via
    /// `UserDefaults`.
    private var tauControlRow: some View {
        HStack(spacing: 10) {
            Text("AI τ")
                .font(.callout.weight(.semibold))
                .foregroundStyle(.secondary)
                .frame(width: 36, alignment: .leading)
            Slider(
                value: $playController.humanPlayTau,
                in: PlayController.humanPlayTauMin...PlayController.humanPlayTauMax,
                step: 0.05
            )
            .frame(maxWidth: 260)
            Text(String(format: "%.2f", playController.humanPlayTau))
                .font(.body.monospacedDigit())
                .frame(width: 48, alignment: .trailing)
            Button("Reset τ") {
                playController.humanPlayTau = 1.0
            }
            .controlSize(.small)
            .disabled(playController.humanPlayTau == 1.0)
            Spacer(minLength: 0)
        }
    }

    // MARK: - Toolbar

    private var toolbarRow: some View {
        HStack(spacing: 12) {
            // Reset stays enabled across natural game-end so the user
            // can launch a fresh game without re-opening the setup
            // popover — `PlayController.reset` works both mid-game and
            // post-game (gates on `canReset` = remembered opponent
            // settings, which the cleanup intentionally preserves).
            // Label flips to "Play Again" post-game so the action
            // reads naturally as "start a new game" rather than as
            // "abort this one and restart" — same code path, different
            // mental model.
            // Stop only applies mid-game; there's nothing to stop once
            // the result banner is up.
            Button(playController.isPlayingHuman ? "Reset Game" : "Play Again") {
                playController.reset(session: session, gameWatcher: gameWatcher)
            }
            .disabled(!playController.canReset)
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
