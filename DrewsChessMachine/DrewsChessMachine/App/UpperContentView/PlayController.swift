import AppKit
import Observation
import SwiftUI
import UniformTypeIdentifiers

/// Which opponent the user has chosen for a human-vs-network game.
enum HumanPlayOpponentChoice: Sendable, Hashable {
    /// Play the live champion (`session.network`) directly.
    case champion
    /// Snapshot the trainer's current SGD weights into a dedicated
    /// inference network and play against that snapshot. Captured at
    /// game start; subsequent training steps don't disturb the
    /// in-progress game.
    case trainer
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
    var opponentChoice: HumanPlayOpponentChoice = .champion

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

    /// Snapshot of the network the AI side is playing against. Held
    /// for the duration of the game; released on stop. Built fresh
    /// every game (even for `.champion`, where it's just an alias for
    /// `session.network`, the snapshot is the alias) so the game
    /// task always has a stable reference even if the live champion
    /// is later replaced.
    private var opponentNetwork: ChessMPSNetwork?

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
    func start(session: SessionController, gameWatcher: GameWatcher) {
        guard !isPlayingHuman else {
            setupErrorText = "A human game is already running."
            return
        }
        setupErrorText = nil

        let opponent = opponentChoice
        let humanIsWhite = (humanColor == .white)
        let chosenURL = loadedFileURL

        // Snapshot the user's two simple sources (champion / loaded file
        // weights) up front, while the trainer path needs an async
        // export. Build the opponent network on a detached task so the
        // popover can dismiss and the UI can render "loading" instead
        // of spinning the main actor.
        isSetupVisible = false
        SessionLogger.shared.log(
            "[BUTTON] Chess > Play (opponent=\(Self.label(for: opponent)) humanColor=\(humanIsWhite ? "white" : "black"))"
        )

        // Reset board state immediately so the UI shows the starting
        // position the moment Play is clicked, even before the
        // network is materialized. Also flip `isPlaying` synchronously
        // so the menu and `isBusy` gates pick it up before the next
        // heartbeat — same reason the existing `playSingleGame`
        // refreshes `gameSnapshot` right after `markPlaying(true)`.
        gameWatcher.resetCurrentGame()
        gameWatcher.markPlaying(true)
        isPlayingHuman = true

        materializeTask = Task { [weak self] in
            let result = await Self.materializeOpponentNetwork(
                choice: opponent,
                session: session,
                loadedFileURL: chosenURL
            )
            guard let self else { return }
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
                return
            case .success(let network):
                self.opponentNetwork = network
                self.launchGame(
                    network: network,
                    humanIsWhite: humanIsWhite,
                    gameWatcher: gameWatcher
                )
            }
        }
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
        opponentNetwork = nil
        pendingLegalMoves = []
        selectedFromSquare = nil
        pendingPromotion = nil
        if materializeWasPending {
            gameWatcher.markPlaying(false)
        }
    }

    private func launchGame(
        network: ChessMPSNetwork,
        humanIsWhite: Bool,
        gameWatcher: GameWatcher
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

        isPlayingHuman = true
        gameWatcher.markPlaying(true)

        // The MPSChessPlayer's per-game scratch isn't `Sendable`, so it
        // (and the `DirectMoveEvaluationSource` that wraps the
        // inference network) is constructed inside the Task closure
        // and never crosses an isolation boundary. The captured
        // sendable inputs — `network`, `human`, `gameWatcher`,
        // `aiLabel`, `humanIsWhite` — produce the AI side and the
        // (white, black) pair entirely within the task.
        gameTask = Task { [weak self, network, human, gameWatcher, aiLabel, humanIsWhite] in
            let source = DirectMoveEvaluationSource(network: network)
            let ai = MPSChessPlayer(name: aiLabel, source: source)
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
                if humanIsWhite {
                    _ = try await machine.beginNewGame(white: human, black: ai)
                } else {
                    _ = try await machine.beginNewGame(white: ai, black: human)
                }
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
                gameWatcher.markPlaying(false)
                guard let self else { return }
                self.isPlayingHuman = false
                self.pendingLegalMoves = []
                self.selectedFromSquare = nil
                self.pendingPromotion = nil
                self.humanPlayer = nil
                self.opponentNetwork = nil
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

    // MARK: - Opponent network materialization

    private nonisolated static func materializeOpponentNetwork(
        choice: HumanPlayOpponentChoice,
        session: SessionController,
        loadedFileURL: URL?
    ) async -> Result<ChessMPSNetwork, Error> {
        switch choice {
        case .champion:
            // The champion network is the live inference net; play
            // games go straight against it (same arrangement as the
            // existing Debug > Play Game).
            if let net = await session.network {
                return .success(net)
            }
            return .failure(PlayControllerError.noChampionAvailable)

        case .trainer:
            do {
                guard let trainer = await session.trainer else {
                    return .failure(PlayControllerError.noTrainerAvailable)
                }
                let weights = try await trainer.network.exportWeights()
                let net = try await buildInferenceNetwork(loading: weights)
                return .success(net)
            } catch {
                return .failure(error)
            }

        case .loadedFile:
            guard let url = loadedFileURL else {
                return .failure(PlayControllerError.noFileSelected)
            }
            do {
                let file = try CheckpointManager.loadModelFile(at: url)
                let net = try await buildInferenceNetwork(loading: file.weights)
                return .success(net)
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

    private static func label(for choice: HumanPlayOpponentChoice) -> String {
        switch choice {
        case .champion: return "champion"
        case .trainer: return "trainer"
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
/// pick the opponent (current champion / current trainer / loaded
/// file) and side, then hit Start to launch the game.
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
                    Text("Current Champion").tag(HumanPlayOpponentChoice.champion)
                        .help(championAvailable ? "" : "No champion network built yet")
                    Text("Current Trainer").tag(HumanPlayOpponentChoice.trainer)
                        .help(trainerAvailable ? "" : "No trainer built yet")
                    Text("Load Saved Model…").tag(HumanPlayOpponentChoice.loadedFile)
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
        case .champion: return championAvailable
        case .trainer: return trainerAvailable
        case .loadedFile: return controller.loadedFileURL != nil
        }
    }
}
