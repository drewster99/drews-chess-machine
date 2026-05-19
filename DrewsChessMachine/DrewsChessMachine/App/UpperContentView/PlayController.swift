import AppKit
import Observation
import SwiftUI
import UniformTypeIdentifiers

/// Which opponent the user has chosen for a human-vs-network game.
enum HumanPlayOpponentChoice: Sendable, Hashable {
    /// Snapshot the current champion's (`session.network`) weights into
    /// a dedicated inference network and play against that snapshot.
    /// Frozen at game start; an arena promotion mid-game won't disturb
    /// the in-progress game.
    case championSnapshot
    /// Snapshot the trainer's current SGD weights into a dedicated
    /// inference network and play against that snapshot. Frozen at
    /// game start; subsequent training steps don't disturb the
    /// in-progress game.
    case trainerSnapshot
    /// Play against a persistent inference-mode mirror that re-overlays
    /// the trainer's *current* weights before every AI move. The human
    /// watches the trainer evolve game-by-game. The mirror lives for
    /// the controller's lifetime — lazy-built on first use, reused
    /// across games and resets.
    case liveTrainer
    /// Play against weights loaded from a `.dcmmodel` file on disk —
    /// either a freestanding model under `Models/` or one of the two
    /// `.dcmmodel` files inside a `.dcmsession` directory.
    case loadedFile
}

/// Drives the human-vs-network "Play" mode: setup popover, opponent
/// network sourcing, tap-based move entry on the board, and the
/// life-cycle of the `ChessMachine` game task.
///
/// Owned by `UpperContentView` as `@State`. The board UI reads
/// `pendingLegalMoves` / `selectedFromSquare` / `pendingPromotion` to
/// render highlights and prompts; the menu DSL flips
/// `isSetupVisible` to surface the configuration popover.
@MainActor
@Observable
final class PlayController {

    // MARK: - Setup popover state

    /// Drives the `.popover(isPresented:)` shown from the Chess menu's
    /// Play… item.
    var isSetupVisible: Bool = false

    /// Last opponent choice the user picked in the setup popover.
    /// Persisted across opens within a single launch.
    var opponentChoice: HumanPlayOpponentChoice = .championSnapshot

    /// Color the human plays as. Default white so the human moves
    /// first — least surprising on a brand-new session.
    var humanColor: PieceColor = .white

    /// `.dcmmodel` URL the user selected for `opponentChoice == .loadedFile`.
    /// Nil until a file is chosen via the picker.
    var loadedFileURL: URL?

    /// Short display label for the loaded file (file's last path
    /// component, or the parent `.dcmsession` directory's name plus
    /// the inner file). Computed on selection so the popover doesn't
    /// re-walk the URL every render.
    var loadedFileLabel: String?

    /// Status text shown in the setup popover when a file pick or
    /// game start fails. Cleared on the next popover open.
    var setupErrorText: String?

    // MARK: - Active game state

    /// True once `start(...)` has constructed players and launched
    /// the game task. Drives the "Resign / Stop" button visibility
    /// and the board's tap-input enable.
    var isPlayingHuman: Bool = false

    /// The legal-move list for the current human turn. Empty while
    /// it's the opponent's turn or no game is running. Used by the
    /// board to highlight the moves emanating from
    /// `selectedFromSquare`.
    var pendingLegalMoves: [ChessMove] = []

    /// Square the user tapped first this turn (their own piece) — the
    /// "from" half of a two-tap move. `nil` between turns and after a
    /// move submission. UI shows a highlight ring on this square and
    /// dots on every legal destination.
    var selectedFromSquare: Int?

    /// Set when the user selects a destination square that has both
    /// promotion and non-promotion variants in the legal list — i.e.,
    /// a pawn reaching the last rank. The board overlays a
    /// piece-picker; tapping one of the four options resolves the
    /// promotion and submits the resulting `ChessMove`.
    var pendingPromotion: PendingPromotion?

    /// `MoveEvaluationSource` the AI side plays through. Held for the
    /// duration of the game; released on stop. For the three snapshot
    /// paths (`.championSnapshot` / `.trainerSnapshot` / `.loadedFile`)
    /// this is a `DirectMoveEvaluationSource` wrapping a freshly-built
    /// inference network that the source owns; for `.liveTrainer` it's
    /// a `LiveTrainerMoveEvaluationSource` that overlays trainer
    /// weights onto `liveTrainerMirrorNetwork` before every AI move.
    private var opponentSource: MoveEvaluationSource?

    /// Persistent inference-mode mirror used by the `.liveTrainer`
    /// path. Lazy-built on the first `.liveTrainer` materialize and
    /// reused for the controller's lifetime — building a fresh
    /// `ChessMPSNetwork` is non-trivial (graph compilation), and the
    /// per-move re-overlay path only needs the graph (the weights
    /// get replaced anyway).
    private var liveTrainerMirrorNetwork: ChessMPSNetwork?

    /// Remembered between `start(...)` and the next explicit `stop(...)`
    /// so `reset(...)` can relaunch a fresh game with the same opponent
    /// choice and side without re-asking the user. Intentionally
    /// preserved across natural game-end (the launchGame cleanup
    /// leaves them in place) so post-game Reset still has settings to
    /// relaunch with. Explicit `stop()` — Stop Game / window close /
    /// next `start()` — clears them.
    private var lastOpponentChoice: HumanPlayOpponentChoice?
    private var lastHumanColor: PieceColor?

    /// True iff a remembered opponent choice + side are on file —
    /// i.e., Reset Game has a fresh game to launch. Becomes true at
    /// `start(...)` and stays true through natural game-end so the
    /// post-game window can still relaunch; falls back to false on
    /// explicit `stop()`.
    var canReset: Bool {
        lastOpponentChoice != nil && lastHumanColor != nil
    }

    // MARK: - Live AI sampling temperature

    /// Sampling temperature (tau) the AI uses for every ply of a
    /// human-vs-network game. 1.0 reproduces the unmodified policy
    /// softmax; values < 1 concentrate on the top-1 move (sharper,
    /// stronger but more predictable play); values > 1 flatten the
    /// distribution toward uniform (weaker, more varied play).
    ///
    /// Live-updating: `MPSChessPlayer` reads from `humanPlayTauBox` at
    /// the top of each `sampleMove`, so changing this between the AI's
    /// moves takes effect on the very next AI move. Persists across
    /// launches via `UserDefaults`.
    var humanPlayTau: Float {
        didSet {
            // NaN / infinity: collapse to the default before the
            // re-entrancy check below would loop forever (NaN != NaN
            // is true in Swift, so a NaN write would fail every
            // equality check). UserDefaults is the only path that
            // could deliver an invalid value here — Sliders honor
            // their bounded range — but a single defensive line is
            // cheaper than depending on every caller staying healthy.
            if !humanPlayTau.isFinite {
                humanPlayTau = 1.0
                return
            }
            let clamped = max(Self.humanPlayTauMin, min(humanPlayTau, Self.humanPlayTauMax))
            if clamped != humanPlayTau {
                humanPlayTau = clamped  // re-enters didSet, then commits below
                return
            }
            UserDefaults.standard.set(Double(humanPlayTau), forKey: Self.humanPlayTauKey)
            humanPlayTauBox.value = humanPlayTau
        }
    }

    /// Lock-protected mirror of `humanPlayTau`, handed to the AI's
    /// `MPSChessPlayer.tauOverride` so the game-task thread can read a
    /// value the main actor wrote without a data race on the
    /// observable property.
    let humanPlayTauBox: SyncBox<Float>

    static let humanPlayTauMin: Float = 0.05
    static let humanPlayTauMax: Float = 3.0
    private static let humanPlayTauKey = "humanPlayTau"

    init() {
        let stored = UserDefaults.standard.object(forKey: Self.humanPlayTauKey) as? Double
        let initial: Float
        if let stored, stored.isFinite {
            initial = max(Self.humanPlayTauMin, min(Float(stored), Self.humanPlayTauMax))
        } else {
            initial = 1.0
        }
        self.humanPlayTau = initial
        self.humanPlayTauBox = SyncBox(initial)
    }

    /// Monotonically incremented by `launchGame(...)`. The launched
    /// game task captures the value at creation time and checks it
    /// inside its `MainActor.run` cleanup block — if the controller's
    /// generation has moved on (the user pressed Reset and a fresh
    /// game is running), the stale cleanup becomes a no-op so it
    /// doesn't clobber the newer game's `isPlayingHuman` / watcher /
    /// `gameTask` state.
    private var gameGeneration: UInt = 0

    /// The active `HumanChessPlayer` instance for this game. Holds
    /// the suspended continuation while it's the user's turn.
    private var humanPlayer: HumanChessPlayer?

    /// `Task` running `ChessMachine.beginNewGame`. Cancelled by
    /// `stop()` and by a new `start(...)` invocation. Holds the
    /// `ChessMachine` alive for the duration of the game.
    private var gameTask: Task<Void, Never>?

    /// `Task` running the up-front opponent-network materialization
    /// (the part that snapshots trainer weights or loads a `.dcmmodel`
    /// from disk and builds a fresh `ChessMPSNetwork`). Held so
    /// `stop()` can cancel a Play that the user changed their mind
    /// about before the game task ever started. `nil` once the
    /// materialization resolves (either way).
    private var materializeTask: Task<Void, Never>?

    /// State machine that paces the human-vs-network play loop —
    /// owns the displayed board snapshot, the per-ply phase, and the
    /// AI permission gate. Allocated per game in `start(...)`, torn
    /// down in `stop(...)`. `HumanPlayWindowView` reads from it via
    /// `@Bindable` (the controller is `@Observable`), and the AI's
    /// `UIGatedMoveEvaluationSource` parks on its
    /// `awaitAIPermission()` before every forward pass so the AI's
    /// reply is always sequenced after the user has seen their own
    /// move animate. `nil` between games.
    private(set) var pacer: HumanPlayPacer?

    // MARK: - Promotion picker

    struct PendingPromotion: Equatable {
        let fromRow: Int
        let fromCol: Int
        let toRow: Int
        let toCol: Int
        /// The four candidate ChessMoves that share the same
        /// (from, to) and differ only in `promotion`. Kept rather
        /// than re-derived so the submit path doesn't have to
        /// re-scan the legal-move list.
        let options: [ChessMove]
    }

    // MARK: - Setup popover entry points

    /// Called by the Chess menu's Play… item. Resets transient setup
    /// state (error text), then shows the popover. If a game is
    /// already in flight, refuses with a status update rather than
    /// silently overlaying a second setup on top.
    func openSetupPopover() {
        if isPlayingHuman {
            setupErrorText = "A human game is already in progress. Stop it first."
            isSetupVisible = true
            return
        }
        setupErrorText = nil
        isSetupVisible = true
    }

    /// Show an `NSOpenPanel` rooted at `Models/`, filtered to
    /// `.dcmmodel`. The Sessions directory is reachable by the user
    /// navigating up one level — `.dcmmodel` files inside a
    /// `.dcmsession` directory are picked the same way.
    func pickModelFile() {
        let panel = NSOpenPanel()
        panel.title = "Choose a saved model"
        panel.message = "Pick a .dcmmodel file (Models/ or inside a .dcmsession/)"
        panel.allowedContentTypes = [
            UTType(filenameExtension: "dcmmodel") ?? .data
        ]
        panel.allowsMultipleSelection = false
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        // Default to Models/ — the user can step up to the parent and
        // descend into Sessions/<name>.dcmsession/ to pick either of
        // the inner `champion.dcmmodel` / `trainer.dcmmodel` files.
        panel.directoryURL = CheckpointPaths.modelsDir
        panel.canCreateDirectories = false

        let response = panel.runModal()
        guard response == .OK, let url = panel.url else { return }
        loadedFileURL = url
        loadedFileLabel = Self.describeModelFile(url)
        // Picking a file implies the user wants the loaded-file
        // option — flip the radio so the popover doesn't require a
        // separate tap.
        opponentChoice = .loadedFile
        setupErrorText = nil
    }

    private static func describeModelFile(_ url: URL) -> String {
        let parent = url.deletingLastPathComponent()
        if parent.pathExtension == "dcmsession" {
            return "\(parent.lastPathComponent) / \(url.lastPathComponent)"
        }
        return url.lastPathComponent
    }

    // MARK: - Start / stop game

    /// Spin up the opponent network according to `opponentChoice`,
    /// construct a `HumanChessPlayer` for the user side and an
    /// `MPSChessPlayer` for the AI side, and launch the game task.
    /// `session` is the `SessionController` that owns `network` and
    /// `trainer` — passed in by `UpperContentView` rather than held as
    /// a property so the controller has no retain cycle on the
    /// session.
    ///
    /// `initialState` and `seededHistory` are non-default only for the
    /// Revert to here path: `initialState` is the position obtained
    /// by replaying the kept prefix, and `seededHistory` is the
    /// matching pacer history that should appear in the move-list
    /// sidebar from the moment the revert finishes. For a standard
    /// fresh game both stay at their defaults (`.starting` / empty).
    func start(
        session: SessionController,
        gameWatcher: GameWatcher,
        initialState: GameState = .starting,
        seededHistory: [HumanPlayPacer.HistoryEntry] = []
    ) {
        guard !isPlayingHuman else {
            setupErrorText = "A human game is already running."
            return
        }
        setupErrorText = nil

        let opponent = opponentChoice
        let humanIsWhite = (humanColor == .white)
        let chosenURL = loadedFileURL
        let isRevert = !seededHistory.isEmpty

        // Snapshot the user's two simple sources (champion / loaded file
        // weights) up front, while the trainer path needs an async
        // export. Build the opponent network on a detached task so the
        // popover can dismiss and the UI can render "loading" instead
        // of spinning the main actor.
        isSetupVisible = false
        if isRevert {
            SessionLogger.shared.log(
                "[BUTTON] Chess > Revert (opponent=\(Self.label(for: opponent)) humanColor=\(humanIsWhite ? "white" : "black") plies=\(seededHistory.count))"
            )
        } else {
            SessionLogger.shared.log(
                "[BUTTON] Chess > Play (opponent=\(Self.label(for: opponent)) humanColor=\(humanIsWhite ? "white" : "black"))"
            )
        }

        // Seed game state immediately so the UI shows the starting (or
        // reverted-to) position the moment Play / Revert is clicked,
        // even before the network is materialized. Also flip
        // `isPlaying` synchronously so the menu and `isBusy` gates
        // pick it up before the next heartbeat — same reason the
        // existing `playSingleGame` refreshes `gameSnapshot` right
        // after `markPlaying(true)`.
        if isRevert {
            gameWatcher.seedFreshGame(
                state: initialState,
                lastMove: seededHistory.last?.move,
                moveCount: seededHistory.count
            )
        } else {
            gameWatcher.resetCurrentGame()
        }
        gameWatcher.markPlaying(true)
        isPlayingHuman = true
        lastOpponentChoice = opponent
        lastHumanColor = humanColor
        // Bump now (before the launchGame from any prior game's
        // already-queued cleanup `MainActor.run` runs) so the stale
        // cleanup finds a mismatched generation and bails.
        gameGeneration &+= 1

        // Stand up a fresh pacer for this game and seed it with the
        // just-seeded board snapshot + any kept history (Revert path)
        // so the human-play window has a consistent `displayedSnapshot`
        // and move list from the moment it opens. The pacer is also
        // handed to the AI's gated source below so the AI side blocks
        // on the pacer's `.aiThinking` transition for each ply.
        let pacer = HumanPlayPacer()
        pacer.start(
            humanColor: humanColor,
            initialSnapshot: gameWatcher.snapshot(),
            seedingHistory: seededHistory
        )
        self.pacer = pacer

        materializeTask = Task { [weak self] in
            guard let self else { return }
            let result = await self.materializeOpponentSource(
                choice: opponent,
                session: session,
                loadedFileURL: chosenURL
            )
            // Note: we deliberately do NOT clear `self.materializeTask`
            // here. After a Start→Stop→Start sequence the field may
            // already hold the *next* task, and a stale clear would
            // strand it (a subsequent Stop wouldn't see it to cancel).
            // `start()` and `stop()` are the only writers; a stale
            // post-completion reference is a tiny leak that the next
            // `start()` overwrites.
            //
            // The user may have hit Stop while the materialize was in
            // flight. `stop()` clears `isPlayingHuman` (and cancels
            // this task), so if either is true here we've already
            // been cancelled and must not flip the gates back on or
            // launch the game.
            if Task.isCancelled || !self.isPlayingHuman {
                return
            }
            switch result {
            case .failure(let error):
                self.setupErrorText = "Could not start game: \(error.localizedDescription)"
                self.isSetupVisible = true
                gameWatcher.markPlaying(false)
                self.isPlayingHuman = false
                self.lastOpponentChoice = nil
                self.lastHumanColor = nil
                return
            case .success(let source):
                // Gate the AI side on the pacer's `.aiThinking`
                // transition so the AI's forward pass never runs in
                // parallel with the UI rendering of the human's move.
                // The pacer's `awaitAIPermission()` is
                // cancellation-aware, so a Stop / Reset cancels the
                // game Task and surfaces `CancellationError` cleanly.
                guard let pacer = self.pacer else {
                    // `start()` always sets `pacer` before the
                    // materialize task runs, and only `stop()` clears
                    // it. If `pacer` is nil here we've been cancelled —
                    // bail without launching the game.
                    return
                }
                let gated = UIGatedMoveEvaluationSource(
                    wrapping: source,
                    pacer: pacer
                )
                self.opponentSource = gated
                self.launchGame(
                    source: gated,
                    humanIsWhite: humanIsWhite,
                    gameWatcher: gameWatcher,
                    initialState: initialState
                )
                // The window controller observes this controller's
                // `@Observable` state plus polls the gameWatcher
                // snapshot on its own timer. Opening it AFTER
                // `launchGame` (rather than in `start(...)`) means the
                // window only appears if the opponent network actually
                // materialized — a failure puts the popover back up
                // with the error rather than orphaning an empty
                // window.
                HumanPlayWindowLauncher.openOrFocus(
                    controller: self,
                    session: session,
                    gameWatcher: gameWatcher
                )
            }
        }
    }

    /// User pressed Reset / on-board Reset. Tear down the current game
    /// and immediately relaunch a fresh game against the same opponent
    /// type and side. Re-running through `start(...)` means snapshot
    /// opponents (`.championSnapshot` / `.trainerSnapshot` /
    /// `.loadedFile`) re-snapshot weights automatically and the
    /// `.liveTrainer` mirror is reused.
    ///
    /// Works mid-game (cancels the in-flight game first) and post-
    /// natural-game-end (the game task already cleaned up `gameTask` /
    /// `opponentSource` but intentionally left `lastOpponentChoice` /
    /// `lastHumanColor` and the .gameOver pacer in place so the user
    /// could still see the result banner and click Reset). In the
    /// post-end branch we tear down the .gameOver pacer here so
    /// `start(...)` can stand up a fresh one.
    func reset(session: SessionController, gameWatcher: GameWatcher) {
        guard let choice = lastOpponentChoice, let color = lastHumanColor else { return }
        SessionLogger.shared.log("[BUTTON] Chess > Reset Game")
        // Capture the saved values before `stop` / start would clear
        // them, then restore onto the bound popover state so
        // `start(...)` reads the same choice + color it just had.
        let savedChoice = choice
        let savedColor = color
        if isPlayingHuman {
            stop(gameWatcher: gameWatcher)
        } else {
            // Game already ended; the launchGame cleanup ran but left
            // the pacer in .gameOver so the result banner stayed
            // visible. Tear it down here before the new start.
            pacer?.stop()
            pacer = nil
            opponentSource = nil
        }
        opponentChoice = savedChoice
        humanColor = savedColor
        start(session: session, gameWatcher: gameWatcher)
    }

    /// User picked a ply in the move-history sidebar and clicked
    /// "Revert to here". Tear down the current game (mid-game or
    /// post-game), reconstruct the game state by replaying the kept
    /// `1...plyNumber` prefix from the standard starting position,
    /// then launch a fresh game from that position with the kept
    /// history pre-loaded into the pacer's history list and
    /// sidebar.
    ///
    /// Constraints:
    /// - `plyNumber >= 1` (the first ply cannot be removed; reverting
    ///   to it keeps it and removes everything after).
    /// - `plyNumber < pacer.history.count` (reverting to the last
    ///   played ply is a no-op — nothing to remove).
    ///
    /// Settings (`lastOpponentChoice`, `lastHumanColor`,
    /// `humanPlayTau`) are intentionally preserved across revert so
    /// the user keeps their opponent + side + sampling temperature.
    /// The AI re-snapshots weights for snapshot opponent modes (same
    /// as Reset Game), so a revert against `.trainerSnapshot` gets
    /// the trainer's current weights, not a stale copy from the
    /// pre-revert game.
    func revertToHistoryPly(_ plyNumber: Int, session: SessionController, gameWatcher: GameWatcher) {
        guard let oldPacer = pacer else { return }
        guard plyNumber >= 1, plyNumber < oldPacer.history.count else { return }
        guard let choice = lastOpponentChoice, let color = lastHumanColor else { return }

        let keptMoves = Array(oldPacer.history.prefix(plyNumber))

        // Reconstruct the game state at the revert point by replaying
        // the kept moves from the standard starting position. The
        // alternative — querying the engine for the historical state
        // at a specific ply — isn't supported (the engine doesn't
        // keep per-ply snapshots), so the replay is the source of
        // truth.
        var revertState = GameState.starting
        for entry in keptMoves {
            revertState = MoveGenerator.applyMove(entry.move, to: revertState)
        }

        // Tear down the current game (preserve settings — `stop()`
        // would clear lastOpponentChoice / lastHumanColor, which we
        // need to feed back into `start(...)`). Mirror `stop()`'s
        // cancellation work without the settings wipe.
        if isPlayingHuman {
            isPlayingHuman = false
            materializeTask?.cancel()
            materializeTask = nil
            humanPlayer?.cancelPendingChoice()
            gameTask?.cancel()
            gameTask = nil
            humanPlayer = nil
        }
        oldPacer.stop()
        pacer = nil
        opponentSource = nil
        pendingLegalMoves = []
        selectedFromSquare = nil
        pendingPromotion = nil

        // Restore opponent + side (the popover state may have drifted
        // since the original `start(...)`, but `lastOpponentChoice` /
        // `lastHumanColor` hold the values the game was actually
        // playing under) and launch with the revert position + kept
        // history seeded into the new pacer.
        opponentChoice = choice
        humanColor = color
        start(
            session: session,
            gameWatcher: gameWatcher,
            initialState: revertState,
            seededHistory: keptMoves
        )
    }

    /// User pressed Stop / Resign. Cancels the game task, which
    /// surfaces `CancellationError` through the suspended human
    /// continuation; the `ChessMachine` loop then unwinds cleanly.
    ///
    /// `gameWatcher` is rolled back here in the materialize-was-still-
    /// in-flight branch (no `gameTask` to run its own cleanup); when a
    /// game task is alive its catch block handles the rollback and
    /// the duplicate call is a harmless no-op.
    func stop(gameWatcher: GameWatcher) {
        guard isPlayingHuman else { return }
        SessionLogger.shared.log("[BUTTON] Chess > Stop human game")
        // Drop `isPlayingHuman` first so the materialize-task success
        // branch (if it's racing this) sees the cancelled state and
        // bails out before flipping the play gates back on.
        isPlayingHuman = false
        let materializeWasPending = (materializeTask != nil && gameTask == nil)
        materializeTask?.cancel()
        materializeTask = nil
        humanPlayer?.cancelPendingChoice()
        gameTask?.cancel()
        gameTask = nil
        humanPlayer = nil
        // Tear down the pacer *before* releasing the gated source so
        // any parked `awaitAIPermission()` resumes with
        // `CancellationError` and the game `Task`'s unwind path
        // proceeds cleanly. The gated source holds a strong reference
        // to the pacer; clearing `opponentSource` afterwards drops
        // the last live reference.
        pacer?.stop()
        pacer = nil
        opponentSource = nil
        lastOpponentChoice = nil
        lastHumanColor = nil
        // Note: `liveTrainerMirrorNetwork` is intentionally NOT cleared.
        // It persists for the controller's lifetime so subsequent
        // `.liveTrainer` games skip the graph-build cost.
        pendingLegalMoves = []
        selectedFromSquare = nil
        pendingPromotion = nil
        if materializeWasPending {
            gameWatcher.markPlaying(false)
        }
    }

    private func launchGame(
        source: MoveEvaluationSource,
        humanIsWhite: Bool,
        gameWatcher: GameWatcher,
        initialState: GameState
    ) {
        let humanLabel = humanIsWhite ? "White (you)" : "Black (you)"
        let aiLabel = humanIsWhite ? "Black (network)" : "White (network)"

        // HumanChessPlayer is `@unchecked Sendable` (lock-protected
        // continuation slot), so the controller can both keep a
        // reference to it for `submit` / `cancelPendingChoice` and
        // hand it into the game Task without a `sending` violation.
        let human = HumanChessPlayer(
            name: humanLabel,
            onTurnBegin: { [weak self] legal in
                guard let self else { return }
                self.pendingLegalMoves = legal
                self.selectedFromSquare = nil
                self.pendingPromotion = nil
            },
            onTurnEnd: { [weak self] in
                guard let self else { return }
                self.pendingLegalMoves = []
                self.selectedFromSquare = nil
                self.pendingPromotion = nil
            }
        )
        humanPlayer = human

        // Capture the current generation (bumped by `start(...)`) so
        // the cleanup block can detect "a newer game has taken over"
        // and skip its cleanup (otherwise a Reset between cancel and
        // cleanup would have the stale cleanup clobber the new game's
        // state — see `gameGeneration`'s doc).
        let myGeneration = gameGeneration

        // MPSChessPlayer's per-game scratch isn't `Sendable`, so the AI
        // player is constructed inside the Task closure and never
        // crosses an isolation boundary. The captured Sendable inputs —
        // `source`, `human`, `gameWatcher`, `aiLabel`, `humanIsWhite`,
        // `myGeneration` — produce the AI side and the (white, black)
        // pair entirely within the task.
        let tauBox = humanPlayTauBox
        gameTask = Task { [weak self, source, human, gameWatcher, aiLabel, humanIsWhite, myGeneration, tauBox, initialState] in
            let ai = MPSChessPlayer(name: aiLabel, source: source, tauOverride: tauBox)
            let machine = ChessMachine()
            machine.delegate = gameWatcher
            do {
                // The two ChessPlayer-typed args are constructed at
                // their concrete types (HumanChessPlayer / MPSChessPlayer)
                // and converted to `any ChessPlayer` at the call site.
                // Combining them through a single existential local
                // before the call ran into a Swift 6 strict-concurrency
                // "sending non-Sendable existential" diagnostic, so
                // pass them positionally instead.
                let raw: RawGameResult
                if humanIsWhite {
                    raw = try await machine.beginNewGame(white: human, black: ai, initialState: initialState)
                } else {
                    raw = try await machine.beginNewGame(white: ai, black: human, initialState: initialState)
                }
                // Log the natural game end so a "Waiting…" or otherwise
                // surprising terminal state in the window is traceable to
                // a concrete engine result in the session log. Without
                // this, the only signal a human game ended was the
                // absence of a Stop log line.
                SessionLogger.shared.log("[CHESS] human-vs-network game ended: \(Self.describe(raw))")
            } catch is CancellationError {
                // User stopped or a new game replaced this one. The
                // GameWatcher state is already coherent (last applied
                // move + isPlaying=false on stop()) so just clean up
                // the active-play state here.
            } catch {
                SessionLogger.shared.log(
                    "[CHESS] human-vs-network game ended with error: \(error.localizedDescription)"
                )
            }
            await MainActor.run {
                guard let self else {
                    // Controller is gone — best-effort wind-down of
                    // the watcher (still strongly retained by this
                    // closure) so the rest of the app's "is a game
                    // running" gates clear.
                    gameWatcher.markPlaying(false)
                    return
                }
                guard self.gameGeneration == myGeneration else {
                    // A newer game (via Reset) has taken over. Leave
                    // `isPlayingHuman`, `gameTask`, the watcher, etc.
                    // alone so the new game's `start(...)` state wins.
                    return
                }
                gameWatcher.markPlaying(false)
                self.isPlayingHuman = false
                self.pendingLegalMoves = []
                self.selectedFromSquare = nil
                self.pendingPromotion = nil
                self.humanPlayer = nil
                // The pacer is intentionally left alive after a
                // natural game end so the window can keep rendering
                // the final position + game-over banner from its
                // `.gameOver` phase. `stop()` (Stop / Reset / window
                // close while still playing) and the next `start()`
                // are the only paths that tear it down.
                //
                // `lastOpponentChoice` / `lastHumanColor` are
                // intentionally NOT cleared here — they're what Reset
                // Game uses to relaunch with the same settings, and
                // we want Reset to work post-game (the on-board
                // toolbar gates on `canReset`, not `isPlayingHuman`).
                // Explicit `stop()` clears them.
                self.opponentSource = nil
                self.gameTask = nil
            }
        }
    }

    // MARK: - Board tap handling

    /// User tapped a square (0..<64) on the board. Two-tap selection:
    ///   1. First tap on one of the user's own pieces selects it
    ///      (sets `selectedFromSquare`).
    ///   2. Second tap on a legal destination submits the move (or
    ///      surfaces a promotion picker if the destination is the
    ///      last rank for a pawn).
    /// Tapping a different own piece while a selection is active
    /// switches the selection. Tapping an illegal target clears the
    /// selection.
    func tapSquare(_ square: Int, in board: [Piece?]) {
        guard isPlayingHuman, !pendingLegalMoves.isEmpty else { return }
        // While the promotion picker is active, board taps should not
        // re-enter the from/to flow — the picker overlay consumes
        // them via `selectPromotion` instead.
        guard pendingPromotion == nil else { return }
        guard (0..<64).contains(square) else { return }

        let row = square / 8
        let col = square % 8

        if let from = selectedFromSquare {
            // A from-square is already chosen; decide whether `square` is
            // a legal destination, a re-pick of a different own piece, or
            // a deselect tap.
            let candidateMoves = pendingLegalMoves.filter {
                $0.fromRow == from / 8 && $0.fromCol == from % 8
                    && $0.toRow == row && $0.toCol == col
            }
            if candidateMoves.isEmpty {
                // Not a legal destination. If the tap landed on another of
                // the user's own pieces, switch selection; otherwise
                // deselect.
                if let piece = board[square],
                   piece.color == humanColor,
                   pendingLegalMoves.contains(where: { $0.fromRow == row && $0.fromCol == col }) {
                    selectedFromSquare = square
                } else {
                    selectedFromSquare = nil
                }
                return
            }
            // Normal move: a single legal move with matching (from, to).
            // Promotion: multiple legal moves sharing (from, to), each with
            // a different promotion piece — surface the picker.
            if candidateMoves.count == 1, candidateMoves[0].promotion == nil {
                submit(candidateMoves[0])
            } else {
                pendingPromotion = PendingPromotion(
                    fromRow: from / 8,
                    fromCol: from % 8,
                    toRow: row,
                    toCol: col,
                    options: candidateMoves
                )
            }
        } else {
            // No from-square yet. Accept only taps on the user's own
            // pieces that have at least one legal move.
            guard let piece = board[square], piece.color == humanColor else { return }
            guard pendingLegalMoves.contains(where: { $0.fromRow == row && $0.fromCol == col }) else { return }
            selectedFromSquare = square
        }
    }

    /// Resolve a pending promotion by piece type. Submits the matching
    /// `ChessMove` and clears the picker. No-op if no promotion is
    /// pending or the requested type isn't among the candidates.
    func selectPromotion(_ pieceType: PieceType) {
        guard let pending = pendingPromotion else { return }
        guard let move = pending.options.first(where: { $0.promotion == pieceType }) else { return }
        pendingPromotion = nil
        submit(move)
    }

    /// User pressed Escape (or tapped outside) while the promotion
    /// picker was open. Clears the picker without submitting a move;
    /// the user remains parked on the same turn with their from-square
    /// still selected.
    func cancelPromotion() {
        pendingPromotion = nil
    }

    private func submit(_ move: ChessMove) {
        guard let human = humanPlayer else { return }
        let accepted = human.submit(move)
        if !accepted {
            // Defensive: the legal-move list shouldn't drift between
            // `onTurnBegin` and submit, but if it does we leave the
            // selection up so the user can pick again rather than the
            // game stalling.
            selectedFromSquare = nil
            pendingPromotion = nil
            return
        }
        // `onTurnEnd` clears the rest of the per-turn state.
    }

    // MARK: - Opponent source materialization

    /// Build the `MoveEvaluationSource` the AI side will play through.
    /// Snapshot cases export the relevant weights and overlay them
    /// onto a freshly-built inference network; `.liveTrainer` lazy-
    /// builds (or reuses) the persistent mirror and returns a
    /// per-move-overlaying source.
    ///
    /// Runs on the main actor since `PlayController` is `@MainActor`;
    /// the long-running MPSGraph build is delegated to
    /// `Task.detached` inside `buildInferenceNetwork(...)` so the main
    /// actor is yielded for the duration of the build.
    private func materializeOpponentSource(
        choice: HumanPlayOpponentChoice,
        session: SessionController,
        loadedFileURL: URL?
    ) async -> Result<MoveEvaluationSource, Error> {
        switch choice {
        case .championSnapshot:
            guard let champion = session.network else {
                return .failure(PlayControllerError.noChampionAvailable)
            }
            do {
                let weights = try await champion.exportWeights()
                let net = try await Self.buildInferenceNetwork(loading: weights)
                return .success(DirectMoveEvaluationSource(network: net))
            } catch {
                return .failure(error)
            }

        case .trainerSnapshot:
            guard let trainer = session.trainer else {
                return .failure(PlayControllerError.noTrainerAvailable)
            }
            do {
                let weights = try await trainer.network.exportWeights()
                let net = try await Self.buildInferenceNetwork(loading: weights)
                return .success(DirectMoveEvaluationSource(network: net))
            } catch {
                return .failure(error)
            }

        case .liveTrainer:
            guard let trainer = session.trainer else {
                return .failure(PlayControllerError.noTrainerAvailable)
            }
            do {
                if liveTrainerMirrorNetwork == nil {
                    liveTrainerMirrorNetwork = try await Self.buildBareInferenceNetwork()
                }
                guard let mirror = liveTrainerMirrorNetwork else {
                    return .failure(PlayControllerError.noTrainerAvailable)
                }
                return .success(LiveTrainerMoveEvaluationSource(
                    trainer: trainer,
                    mirror: mirror
                ))
            } catch {
                return .failure(error)
            }

        case .loadedFile:
            guard let url = loadedFileURL else {
                return .failure(PlayControllerError.noFileSelected)
            }
            do {
                let file = try CheckpointManager.loadModelFile(at: url)
                let net = try await Self.buildInferenceNetwork(loading: file.weights)
                return .success(DirectMoveEvaluationSource(network: net))
            } catch {
                return .failure(error)
            }
        }
    }

    /// Build a fresh `.randomWeights` `ChessMPSNetwork` and overlay the
    /// supplied weights. Runs on a detached `.userInitiated` task so
    /// the MPSGraph build (long synchronous work) never sits on the
    /// Swift Concurrency executor. Mirrors the pattern
    /// `SessionController.performBuild()` uses, plus an immediate
    /// `loadWeights` to overwrite the randomly-initialized graph.
    private nonisolated static func buildInferenceNetwork(loading weights: [[Float]]) async throws -> ChessMPSNetwork {
        let net = try await Task.detached(priority: .userInitiated) {
            try ChessMPSNetwork(.randomWeights)
        }.value
        try await net.loadWeights(weights)
        return net
    }

    /// Build a `.randomWeights` `ChessMPSNetwork` *without* overlaying
    /// weights. Used as the lazy initializer for the live-trainer
    /// mirror — the mirror's initial weights are immaterial because
    /// `LiveTrainerMoveEvaluationSource.evaluate` overwrites them on
    /// every AI move via `loadWeights(...)`.
    private nonisolated static func buildBareInferenceNetwork() async throws -> ChessMPSNetwork {
        try await Task.detached(priority: .userInitiated) {
            try ChessMPSNetwork(.randomWeights)
        }.value
    }

    /// Compact human-readable form of a `RawGameResult` for the session log.
    /// Mirrors the banner copy in `HumanPlayWindow` so a log reader can
    /// cross-check the displayed game-over message.
    private static func describe(_ raw: RawGameResult) -> String {
        switch raw {
        case .terminatedEarly:
            return "terminatedEarly"
        case .terminatedNormally(let result):
            switch result {
            case .checkmate(let winner):
                return "checkmate (\(winner == .white ? "white" : "black") wins)"
            case .stalemate:
                return "stalemate"
            case .drawByFiftyMoveRule:
                return "drawByFiftyMoveRule"
            case .drawByInsufficientMaterial:
                return "drawByInsufficientMaterial"
            case .drawByThreefoldRepetition:
                return "drawByThreefoldRepetition"
            }
        }
    }

    private static func label(for choice: HumanPlayOpponentChoice) -> String {
        switch choice {
        case .championSnapshot: return "champion-snapshot"
        case .trainerSnapshot: return "trainer-snapshot"
        case .liveTrainer: return "live-trainer"
        case .loadedFile: return "loaded-file"
        }
    }
}

// MARK: - Errors

enum PlayControllerError: LocalizedError {
    case noChampionAvailable
    case noTrainerAvailable
    case noFileSelected

    var errorDescription: String? {
        switch self {
        case .noChampionAvailable:
            return "No champion network has been built. Build or load one first."
        case .noTrainerAvailable:
            return "The trainer hasn't been built yet. Start Play-and-Train at least once, then stop, then try again."
        case .noFileSelected:
            return "No .dcmmodel file was selected."
        }
    }
}

// MARK: - Setup popover view

/// Configuration popover surfaced from Chess > Play…. Lets the user
/// pick the opponent (champion snapshot / trainer snapshot / live
/// trainer / loaded file) and side, then hit Start to launch the game.
struct PlaySetupPopover: View {
    @Bindable var controller: PlayController
    let championAvailable: Bool
    let trainerAvailable: Bool
    let onStart: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Play vs. Network")
                .font(.headline)

            VStack(alignment: .leading, spacing: 8) {
                Text("Opponent")
                    .font(.subheadline.weight(.semibold))
                Picker("Opponent", selection: $controller.opponentChoice) {
                    Text("Champion (snapshot)")
                        .tag(HumanPlayOpponentChoice.championSnapshot)
                        .help(championAvailable
                            ? "Freeze the current champion's weights at game start."
                            : "No champion network built yet"
                        )
                    Text("Trainer (snapshot)")
                        .tag(HumanPlayOpponentChoice.trainerSnapshot)
                        .help(trainerAvailable
                            ? "Freeze the trainer's current weights at game start."
                            : "No trainer built yet"
                        )
                    Text("Trainer (live)")
                        .tag(HumanPlayOpponentChoice.liveTrainer)
                        .help(trainerAvailable
                            ? "Re-snapshot the trainer's weights before every AI move — watch training evolve game-by-game."
                            : "No trainer built yet"
                        )
                    Text("Load Saved Model…")
                        .tag(HumanPlayOpponentChoice.loadedFile)
                }
                .pickerStyle(.radioGroup)
                .labelsHidden()

                if controller.opponentChoice == .loadedFile {
                    HStack(spacing: 8) {
                        Button(action: { controller.pickModelFile() }, label: {
                            Text(controller.loadedFileURL == nil ? "Choose .dcmmodel…" : "Change…")
                        })
                        if let label = controller.loadedFileLabel {
                            Text(label)
                                .font(.system(.body, design: .monospaced))
                                .foregroundStyle(.secondary)
                                .lineLimit(1)
                                .truncationMode(.middle)
                        }
                    }
                }
            }

            Divider()

            VStack(alignment: .leading, spacing: 8) {
                Text("Play as")
                    .font(.subheadline.weight(.semibold))
                Picker("Play as", selection: $controller.humanColor) {
                    Text("White").tag(PieceColor.white)
                    Text("Black").tag(PieceColor.black)
                }
                .pickerStyle(.segmented)
                .labelsHidden()
            }

            if let error = controller.setupErrorText {
                Text(error)
                    .font(.callout)
                    .foregroundStyle(.red)
                    .fixedSize(horizontal: false, vertical: true)
            }

            HStack {
                Button("Cancel") { controller.isSetupVisible = false }
                    .keyboardShortcut(.cancelAction)
                Spacer()
                Button("Start Game") { onStart() }
                    .keyboardShortcut(.defaultAction)
                    .disabled(!isStartEnabled)
            }
        }
        .padding(16)
        .frame(width: 360)
    }

    private var isStartEnabled: Bool {
        switch controller.opponentChoice {
        case .championSnapshot: return championAvailable
        case .trainerSnapshot: return trainerAvailable
        case .liveTrainer: return trainerAvailable
        case .loadedFile: return controller.loadedFileURL != nil
        }
    }
}

// MARK: - On-board human-play toolbar

/// Small Reset / Stop row rendered below the chess board while a
/// human game is in flight. Mirrors Chess menu's Reset Game / Stop
/// Game items so the user doesn't have to leave the board to issue
/// either command. The `controller` binding lets the toolbar fade in
/// and out automatically as `isPlayingHuman` toggles.
struct HumanPlayToolbar: View {
    @Bindable var controller: PlayController
    let onReset: () -> Void
    let onStop: () -> Void

    var body: some View {
        HStack(spacing: 12) {
            Button("Reset Game") { onReset() }
            Button("Stop Game") { onStop() }
        }
        .controlSize(.small)
    }
}
