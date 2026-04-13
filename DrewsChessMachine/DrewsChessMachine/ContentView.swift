import SwiftUI

// MARK: - Result Types

struct EvaluationResult: Sendable {
    let topMoves: [MoveVisualization]
    let textOutput: String
    let inputTensor: [Float]
}

/// Live progress reported from inside a sweep task. The trainer fires the
/// progress callback before each step; we publish that to the UI so the
/// user can see "currently sweeping batch=X, step Y, elapsed Z".
struct SweepProgress: Sendable {
    let batchSize: Int
    let stepsSoFar: Int
    let elapsedSec: Double
}

/// Lock-protected holder for the sweep's shared state across the worker
/// thread and the SwiftUI main thread. The worker writes (progress, row
/// completions, errors); the main-thread heartbeat polls and lifts those
/// values into `@State`. The Stop button writes `cancel()` directly here
/// rather than going through Task cancellation — Swift's unstructured
/// `Task { }` doesn't inherit cancellation, so leaning on Task.isCancelled
/// inside a worker spawned from inside another Task is unreliable.
/// A locked Bool that the worker polls between steps is simpler and works.
final class CancelBox: @unchecked Sendable {
    private let lock = NSLock()
    private var _cancelled = false
    private var _progress: SweepProgress?
    private var _completedRows: [SweepRow] = []
    private var _rowPeakBytes: UInt64 = 0

    var isCancelled: Bool {
        lock.lock(); defer { lock.unlock() }
        return _cancelled
    }

    func cancel() {
        lock.lock(); defer { lock.unlock() }
        _cancelled = true
    }

    func updateProgress(_ p: SweepProgress) {
        lock.lock(); defer { lock.unlock() }
        _progress = p
    }

    var latestProgress: SweepProgress? {
        lock.lock(); defer { lock.unlock() }
        return _progress
    }

    func appendRow(_ r: SweepRow) {
        lock.lock(); defer { lock.unlock() }
        _completedRows.append(r)
    }

    var completedRows: [SweepRow] {
        lock.lock(); defer { lock.unlock() }
        return _completedRows
    }

    /// Update the per-row peak with a new sample. The sweep's worker
    /// thread reads and resets this between rows via `takeRowPeak()`.
    /// Called from both the UI heartbeat (every ~100ms) and from the
    /// trainer at row boundaries — whichever produces the higher value
    /// wins for that row.
    func recordPeakSample(_ bytes: UInt64) {
        lock.lock(); defer { lock.unlock() }
        if bytes > _rowPeakBytes { _rowPeakBytes = bytes }
    }

    /// Read the peak observed during the just-finished row and reset the
    /// accumulator for the next one.
    func takeRowPeak() -> UInt64 {
        lock.lock(); defer { lock.unlock() }
        let peak = _rowPeakBytes
        _rowPeakBytes = 0
        return peak
    }
}

// MARK: - Game Watcher

/// Holds live game state mutated by the ChessMachine delegate queue.
///
/// **Not @Observable.** SwiftUI doesn't observe its mutations directly —
/// instead, ContentView polls `snapshot()` on a 100ms timer and copies the
/// values into local @State. That decouples UI redraw frequency from game
/// throughput: continuous self-play can run hundreds of moves per second
/// while the UI updates at most 10 times per second, and the game loop
/// never waits for SwiftUI invalidation.
///
/// Mutations come from two sources:
/// 1. The ChessMachine delegate queue (didApplyMove, gameEnded, playerErrored).
/// 2. Direct calls from ContentView actions (resetCurrentGame, markPlaying).
/// Both go through `lock`, so reads from any thread are safe.
final class GameWatcher: ChessMachineDelegate, @unchecked Sendable {
    /// Snapshot of all displayable values, taken atomically. Sendable so it
    /// can flow from any thread to the main actor.
    struct Snapshot: Sendable {
        var state: GameState = .starting
        var result: GameResult?
        var moveCount = 0
        var isPlaying = false
        var lastGameStats: GameStats?

        var totalGames = 0
        var totalMoves = 0
        var totalGameTimeMs: Double = 0
        var totalWhiteThinkMs: Double = 0
        var totalBlackThinkMs: Double = 0

        /// Cumulative wall-clock seconds spent in active play (between
        /// markPlaying(true) and the matching markPlaying(false) /
        /// gameEnded / playerErrored). Idle time between Play clicks
        /// and the small inter-game pauses in continuous play are excluded.
        var activePlaySeconds: Double = 0
        /// Set to the wall-clock time at which play started; cleared on stop.
        /// While non-nil, the live "active seconds" includes (now - this).
        var currentPlayStartTime: CFAbsoluteTime?

        var whiteCheckmates = 0
        var blackCheckmates = 0
        var stalemates = 0
        var fiftyMoveDraws = 0
        var insufficientMaterialDraws = 0
        var threefoldRepetitionDraws = 0
    }

    private let lock = NSLock()
    private var s = Snapshot()

    func snapshot() -> Snapshot {
        lock.lock()
        defer { lock.unlock() }
        return s
    }

    func resetCurrentGame() {
        lock.lock()
        defer { lock.unlock() }
        s.state = .starting
        s.result = nil
        s.moveCount = 0
        // Keep lastGameStats — show previous game until the next one ends
    }

    func resetAll() {
        lock.lock()
        defer { lock.unlock() }
        s = Snapshot()
    }

    func markPlaying(_ playing: Bool) {
        lock.lock()
        defer { lock.unlock() }
        setPlayingLocked(playing)
    }

    /// Toggle isPlaying and update the active-play stopwatch. Caller must
    /// already hold `lock`. Idempotent: calling with the same value twice
    /// is a no-op for the stopwatch.
    private func setPlayingLocked(_ playing: Bool) {
        if playing {
            if s.currentPlayStartTime == nil {
                s.currentPlayStartTime = CFAbsoluteTimeGetCurrent()
            }
        } else if let start = s.currentPlayStartTime {
            s.activePlaySeconds += max(0, CFAbsoluteTimeGetCurrent() - start)
            s.currentPlayStartTime = nil
        }
        s.isPlaying = playing
    }

    // MARK: - Delegate (called on ChessMachine.delegateQueue, never main)

    func chessMachine(_ machine: ChessMachine, didApplyMove move: ChessMove, newState: GameState) {
        lock.lock()
        defer { lock.unlock() }
        s.state = newState
        s.moveCount += 1
    }

    func chessMachine(
        _ machine: ChessMachine,
        gameEndedWith result: GameResult,
        finalState: GameState,
        stats: GameStats
    ) {
        lock.lock()
        defer { lock.unlock() }
        s.result = result
        s.state = finalState
        s.lastGameStats = stats
        setPlayingLocked(false)

        s.totalGames += 1
        s.totalMoves += stats.totalMoves
        s.totalGameTimeMs += stats.totalGameTimeMs
        s.totalWhiteThinkMs += stats.whiteThinkingTimeMs
        s.totalBlackThinkMs += stats.blackThinkingTimeMs
        // Move counting handed off to totalMoves; zero the per-game counter
        // atomically so display helpers using `totalMoves + moveCount` don't
        // double-count between gameEnded and the next resetCurrentGame call.
        s.moveCount = 0

        switch result {
        case .checkmate(let winner):
            if winner == .white {
                s.whiteCheckmates += 1
            } else {
                s.blackCheckmates += 1
            }
        case .stalemate:
            s.stalemates += 1
        case .drawByFiftyMoveRule:
            s.fiftyMoveDraws += 1
        case .drawByInsufficientMaterial:
            s.insufficientMaterialDraws += 1
        case .drawByThreefoldRepetition:
            s.threefoldRepetitionDraws += 1
        }
    }

    func chessMachine(_ machine: ChessMachine, playerErrored player: any ChessPlayer, error: any Error) {
        lock.lock()
        defer { lock.unlock() }
        setPlayingLocked(false)
    }
}

// MARK: - Snapshot Display Helpers

extension GameWatcher.Snapshot {
    var avgMoveTimeMs: Double {
        totalMoves > 0 ? (totalWhiteThinkMs + totalBlackThinkMs) / Double(totalMoves) : 0
    }

    var avgGameTimeMs: Double {
        totalGames > 0 ? totalGameTimeMs / Double(totalGames) : 0
    }

    /// Discrete throughput: based only on completed-game time. Equivalent to
    /// 3600 / avgGameTimeSec. Updates only when a game ends, so it never
    /// drifts mid-game.
    func gamesPerHour() -> Double {
        guard totalGames > 0, totalGameTimeMs > 0 else { return 0 }
        let totalGameSec = totalGameTimeMs / 1000
        return Double(totalGames) / (totalGameSec / 3600)
    }

    /// Live throughput: includes the in-progress game's moves so the rate
    /// updates smoothly. Denominator is active play time, so idle gaps
    /// between Play clicks don't depress the value.
    func movesPerHour(now: CFAbsoluteTime) -> Double {
        let moves = totalMoves + moveCount
        let secs = activeSeconds(now: now)
        guard moves > 0, secs > 0 else { return 0 }
        return Double(moves) / (secs / 3600)
    }

    /// Live cumulative seconds spent in active play.
    func activeSeconds(now: CFAbsoluteTime) -> Double {
        var sec = activePlaySeconds
        if let start = currentPlayStartTime {
            sec += max(0, now - start)
        }
        return sec
    }

    /// True once any play has started — used to decide whether to show
    /// "--" for time and rates.
    var hasSession: Bool {
        activePlaySeconds > 0 || currentPlayStartTime != nil
    }

    /// Fixed-layout stats text. Every section is always present — values show
    /// "--" when no data exists. Structure never changes so layout stays stable.
    func statsText(continuousPlay: Bool) -> String {
        let dash = "--"
        let now = CFAbsoluteTimeGetCurrent()

        // Status line
        let status: String
        if isPlaying {
            let turn = state.currentPlayer == .white ? "White" : "Black"
            let check = MoveGenerator.isInCheck(state, color: state.currentPlayer) ? " CHECK" : ""
            status = "\(turn) to move (move \(moveCount + 1))\(check)"
        } else if let result {
            switch result {
            case .checkmate(let winner):
                status = "\(winner == .white ? "White" : "Black") wins by checkmate"
            case .stalemate:
                status = "Draw by stalemate"
            case .drawByFiftyMoveRule:
                status = "Draw by fifty-move rule"
            case .drawByInsufficientMaterial:
                status = "Draw by insufficient material"
            case .drawByThreefoldRepetition:
                status = "Draw by threefold repetition"
            }
        } else {
            status = dash
        }

        // Session — totalMoves only counts completed games' moves; add the
        // in-progress game's moveCount so the displayed count and rate update
        // smoothly instead of jumping at game-end.
        let liveMoves = totalMoves + moveCount
        let sGames = totalGames > 0 ? totalGames.formatted(.number.grouping(.automatic)) : dash
        let sMoves = liveMoves > 0 ? liveMoves.formatted(.number.grouping(.automatic)) : dash
        let sTime = hasSession ? Self.formatHMS(seconds: activeSeconds(now: now)) : dash
        let sAvgMove = totalMoves > 0 ? String(format: "%.2f ms", avgMoveTimeMs) : dash
        let sAvgGame = totalGames > 0 ? String(format: "%.1f ms", avgGameTimeMs) : dash
        let sMovesHr = liveMoves > 0
            ? Int(movesPerHour(now: now).rounded()).formatted(.number.grouping(.automatic))
            : dash
        let sGamesHr = totalGames > 0
            ? Int(gamesPerHour().rounded()).formatted(.number.grouping(.automatic))
            : dash

        // Last Game — only shown when not in active continuous play. During
        // continuous play it has no value (it's already part of session totals
        // and changes too fast to read).
        let lastGameSection: String
        if continuousPlay {
            lastGameSection = ""
        } else if let stats = lastGameStats {
            let lgMoves = String(format: "%d (%dW + %dB)", stats.totalMoves, stats.whiteMoves, stats.blackMoves)
            let lgTime = String(format: "%.1f ms", stats.totalGameTimeMs)
            let lgAvg = String(format: "%.2f ms", stats.avgMoveTimeMs)
            let lgWAvg = String(format: "%.2f ms", stats.avgWhiteMoveTimeMs)
            let lgBAvg = String(format: "%.2f ms", stats.avgBlackMoveTimeMs)
            lastGameSection = """
                Last Game
                  Moves:     \(lgMoves)
                  Time:      \(lgTime)
                  Avg move:  \(lgAvg)
                  White avg: \(lgWAvg)
                  Black avg: \(lgBAvg)


                """
        } else {
            lastGameSection = ""
        }

        return """
            Status: \(status)

            \(lastGameSection)Session
              Games:     \(sGames)
              Moves:     \(sMoves)
              Time:      \(sTime)
              Avg move:  \(sAvgMove)
              Avg game:  \(sAvgGame)
              Moves/hr:  \(sMovesHr)
              Games/hr:  \(sGamesHr)

            Results
              Checkmate:      \(whiteCheckmates + blackCheckmates)\(pct(whiteCheckmates + blackCheckmates))
                White wins:     \(whiteCheckmates)\(pct(whiteCheckmates))
                Black wins:     \(blackCheckmates)\(pct(blackCheckmates))
              Stalemate:      \(stalemates)\(pct(stalemates))
              50-move draw:   \(fiftyMoveDraws)\(pct(fiftyMoveDraws))
              Threefold rep:  \(threefoldRepetitionDraws)\(pct(threefoldRepetitionDraws))
              Insufficient:   \(insufficientMaterialDraws)\(pct(insufficientMaterialDraws))
            """
    }

    private func pct(_ count: Int) -> String {
        guard totalGames > 0 else { return "" }
        return String(format: "  (%.1f%%)", Double(count) / Double(totalGames) * 100)
    }

    static func formatHMS(seconds: Double) -> String {
        let totalTenths = Int((seconds * 10).rounded())
        let tenths = totalTenths % 10
        let totalSeconds = totalTenths / 10
        let h = totalSeconds / 3600
        let m = (totalSeconds % 3600) / 60
        let s = totalSeconds % 60
        return String(format: "%02d:%02d:%02d.%d", h, m, s, tenths)
    }
}

// MARK: - Content View

struct ContentView: View {
    // Network
    @State private var network: ChessMPSNetwork?
    @State private var runner: ChessRunner?
    @State private var networkStatus = ""
    @State private var isBuilding = false

    // Inference
    @State private var inferenceResult: EvaluationResult?
    @State private var isEvaluating = false
    @State private var selectedOverlay = 0
    /// The position the forward-pass demo is evaluating. Seeded to the
    /// starting position on launch and NEVER auto-reset — free-placement
    /// edits persist across Build Network, mode switches, and re-runs, so
    /// the user can tinker with a position and come back to it. Game mode
    /// doesn't read this (it shows `gameSnapshot.state.board` instead), and
    /// training modes ignore it entirely.
    @State private var editableState: GameState = .starting
    /// Set when `reevaluateForwardPass()` is called while an inference is
    /// already in flight. The in-flight task checks this on completion and
    /// re-runs itself once more so the latest edit is always reflected
    /// without us having to block drags on `isEvaluating`.
    @State private var pendingReeval = false

    // Game — gameWatcher is mutated by the delegate queue and is NOT
    // SwiftUI-observed. A 100ms timer copies its `snapshot()` into
    // `gameSnapshot`, which is what the body actually reads. This caps UI
    // refresh rate regardless of game throughput.
    //
    // `gameWatcher` MUST be `@State`, not `let`. SwiftUI may reconstruct
    // ContentView's struct across body invocations; a plain `let` initializer
    // would build a fresh `GameWatcher` each time and any in-flight machine
    // (which only holds the delegate via `weak`) would lose its delegate.
    @State private var gameWatcher = GameWatcher()
    @State private var gameSnapshot = GameWatcher.Snapshot()
    @State private var continuousPlay = false
    @State private var continuousTask: Task<Void, Never>?

    // Training — trainer is built lazily on first use. It owns its own
    // training-mode ChessNetwork internally (not shared with the inference
    // network used by Play / Forward Pass), so its weight updates do NOT
    // flow into inference. The inference network keeps frozen-stats BN
    // for fast play; the trainer measures realistic training-step costs
    // through batch-stats BN and the full backward graph.
    @State private var trainer: ChessTrainer?
    @State private var isTrainingOnce = false
    @State private var continuousTraining = false
    @State private var trainingTask: Task<Void, Never>?
    @State private var lastTrainStep: TrainStepTiming?
    @State private var trainingStats: TrainingRunStats?
    @State private var trainingError: String?
    /// Lock-protected live-stats holder shared with the background training
    /// task (continuous or self-play). The worker writes via `recordStep`
    /// with no main-actor hop; the 60 Hz `snapshotTimer` poller mirrors the
    /// latest values into `trainingStats` / `lastTrainStep` /
    /// `realRollingPolicyLoss` / `realRollingValueLoss` only when the step
    /// count has actually advanced.
    @State private var trainingBox: TrainingLiveStatsBox?
    nonisolated static let trainingBatchSize = 256

    // Real (self-play) training — generates games, labels positions from the
    // final outcome, pushes them through the shared trainer. Shares the
    // lazily-built `trainer` with the random-data training path above so the
    // trainer's MPSGraph is built at most once per session. Only one training
    // mode is allowed to run at a time (enforced by the button hide rules and
    // by `isBusy`), so there's no cross-mode concurrency on the trainer.
    @State private var realTraining = false
    @State private var realTrainingTask: Task<Void, Never>?
    @State private var replayBuffer: ReplayBuffer?
    /// Rolling-window averages of the most recent self-play training losses,
    /// split into the policy (outcome-weighted cross-entropy) and value
    /// (bounded MSE) components. Mirrored from `trainingBox` by the 60 Hz
    /// poller. The windows themselves live inside the box — these are just
    /// the most recent display values. Split so we can tell whether an
    /// oscillating total-loss plot is the bounded value term going unstable
    /// (training problem) or the unbounded policy term bouncing around
    /// (usually just metric noise).
    @State private var realRollingPolicyLoss: Double?
    @State private var realRollingValueLoss: Double?
    nonisolated static let replayBufferCapacity = 50_000
    /// Don't start sampling training batches until the buffer holds at least
    /// this many positions. At 16× the batch size, each draw covers ~6% of
    /// the buffer, so consecutive batches share few enough samples to keep
    /// gradient estimates meaningfully decorrelated. Also guarantees the
    /// `ReplayBuffer.sample` call inside the train loop can never return
    /// nil, since `minBufferBeforeTraining >= trainingBatchSize` by
    /// construction.
    nonisolated static let minBufferBeforeTraining = trainingBatchSize * 16
    /// Training steps to run between self-play games. Kept modest so the
    /// driver alternates visibly between play and train rather than
    /// disappearing into a long training run.
    nonisolated static let trainStepsPerGame = 10
    /// Size of the rolling-loss window displayed in the Self-Play training
    /// column. 512 steps × 256 batch = ~131k positions averaged per reported
    /// number, which should be more than enough to smooth through batch-
    /// composition noise and show the real underlying trend.
    nonisolated static let rollingLossWindow = 512

    // Batch-size sweep — runs each size in `sweepSizes` for ~15 s, then
    // displays the throughput table. Driven by its own task / cancel path
    // so it can share the unified Stop button.
    @State private var sweepRunning = false
    @State private var sweepTask: Task<Void, Never>?
    @State private var sweepResults: [SweepRow] = []
    @State private var sweepProgress: SweepProgress?
    @State private var sweepCancelBox: CancelBox?
    @State private var sweepDeviceCaps: DeviceMemoryCaps?
    nonisolated static let sweepSizes: [Int] = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    nonisolated static let sweepSecondsPerSize: Double = 1.0

    /// 100 ms heartbeat that pulls the latest snapshot from `gameWatcher`
    /// into `gameSnapshot`. Standard SwiftUI Combine timer pattern — the
    /// publisher is created when the view struct is initialized and SwiftUI
    /// manages the subscription lifecycle via `.onReceive` below.
    private let snapshotTimer = Timer.publish(
        every: 1.0/60.0, on: .main, in: .common
    ).autoconnect()

    private var networkReady: Bool { network != nil }
    private var isBusy: Bool {
        isBuilding
            || isEvaluating
            || gameSnapshot.isPlaying
            || continuousPlay
            || isTrainingOnce
            || continuousTraining
            || sweepRunning
            || realTraining
    }
    private var isGameMode: Bool { gameSnapshot.isPlaying || gameSnapshot.totalGames > 0 }
    private var isTrainingMode: Bool {
        isTrainingOnce
            || continuousTraining
            || realTraining
            || trainingStats != nil
            || lastTrainStep != nil
            || sweepRunning
            || !sweepResults.isEmpty
    }

    private var displayedPieces: [Piece?] {
        if isGameMode { return gameSnapshot.state.board }
        return editableState.board
    }

    /// Whether the forward-pass free-placement editor should be live: a
    /// forward pass can run, the user isn't watching a game, and no
    /// training mode is active. Used to gate both the drag gesture and the
    /// side-to-move picker so they don't fight with other modes.
    private var forwardPassEditable: Bool {
        networkReady && !isGameMode && !isTrainingMode
    }

    /// Binding for the side-to-move segmented picker. Writes rebuild
    /// `editableState` with the new current-player (nothing else changes)
    /// and kick off an auto re-eval so the arrows update for the new
    /// perspective.
    private var sideToMoveBinding: Binding<PieceColor> {
        Binding(
            get: { editableState.currentPlayer },
            set: { newValue in
                editableState = GameState(
                    board: editableState.board,
                    currentPlayer: newValue,
                    whiteKingsideCastle: editableState.whiteKingsideCastle,
                    whiteQueensideCastle: editableState.whiteQueensideCastle,
                    blackKingsideCastle: editableState.blackKingsideCastle,
                    blackQueensideCastle: editableState.blackQueensideCastle,
                    enPassantSquare: editableState.enPassantSquare,
                    halfmoveClock: editableState.halfmoveClock
                )
                reevaluateForwardPass()
            }
        )
    }

    private var overlayLabel: String {
        if selectedOverlay == 0 { return "Top Moves" }
        return "Channel \(selectedOverlay - 1): \(TensorChannelNames.names[selectedOverlay - 1])"
    }

    private var currentOverlay: BoardOverlay {
        if isGameMode { return .none }
        guard let result = inferenceResult else { return .none }
        if selectedOverlay == 0 {
            return .topMoves(result.topMoves)
        } else {
            let start = (selectedOverlay - 1) * 64
            return .channel(Array(result.inputTensor[start..<start + 64]))
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Chess Neural Network")
                .font(.title2)
                .fontWeight(.semibold)

            Text(
                "Forward pass through a ~2.9M parameter convolutional network using MPSGraph " +
                "on the GPU. Weights are randomly initialized (He initialization) — no training " +
                "has occurred."
            )
                .font(.callout)
                .foregroundStyle(.secondary)

            // Buttons
            HStack(spacing: 8) {
                Button("Build Network") { buildNetwork() }
                    .disabled(isBusy || networkReady)

                Button("Run Forward Pass") { runForwardPass() }
                    .disabled(isBusy || !networkReady)
                    .keyboardShortcut(.return)

                Divider().frame(height: 20)

                Button("Play Game") { playSingleGame() }
                    .disabled(isBusy || !networkReady)

                if !continuousPlay && !continuousTraining && !realTraining {
                    Button("Play Continuous") { startContinuousPlay() }
                        .disabled(isBusy || !networkReady)
                }

                Divider().frame(height: 20)

                Button("Train Once") { trainOnce() }
                    .disabled(isBusy || !networkReady)

                if !continuousTraining && !continuousPlay && !sweepRunning && !realTraining {
                    Button("Train Continuous") { startContinuousTraining() }
                        .disabled(isBusy || !networkReady)
                }

                if !realTraining && !continuousPlay && !continuousTraining && !sweepRunning {
                    Button("Train on Self-Play") { startRealTraining() }
                        .disabled(isBusy || !networkReady)
                }

                if !sweepRunning && !continuousPlay && !continuousTraining && !realTraining {
                    Button("Sweep Batch Sizes") { startSweep() }
                        .disabled(isBusy || !networkReady)
                }

                // Single unified Stop button — handles whichever continuous
                // loop is currently running (play, training, self-play
                // training, or sweep). Bound to escape so the same shortcut
                // works in every mode.
                if continuousPlay || continuousTraining || sweepRunning || realTraining {
                    Button("Stop") { stopAnyContinuous() }
                        .keyboardShortcut(.escape, modifiers: [])
                }

                if isBusy {
                    ProgressView().controlSize(.small)
                    Text(busyLabel).foregroundStyle(.secondary)
                }
            }

            // Board + text side by side
            HStack(alignment: .top, spacing: 24) {
                VStack(spacing: 6) {
                    if inferenceResult != nil, !isGameMode {
                        Text(overlayLabel)
                            .font(.system(.subheadline, design: .monospaced))
                    }

                    if forwardPassEditable {
                        Picker("To move", selection: sideToMoveBinding) {
                            Text("White").tag(PieceColor.white)
                            Text("Black").tag(PieceColor.black)
                        }
                        .pickerStyle(.segmented)
                        .labelsHidden()
                        .frame(maxWidth: 160)
                    }

                    HStack(spacing: 8) {
                        let leftDisabled = inferenceResult == nil || selectedOverlay == 0 || isGameMode
                        Button(
                            action: { navigateOverlay(-1) },
                            label: {
                                Image(systemName: "chevron.left").font(.title3).frame(width: 24)
                            }
                        )
                        .buttonStyle(.plain)
                        .disabled(leftDisabled)
                        .opacity(leftDisabled ? 0.2 : 0.6)

                        ChessBoardView(pieces: displayedPieces, overlay: currentOverlay)
                            .overlay {
                                // Transparent hit layer that converts drag
                                // coordinates to squares and routes edits
                                // through `applyFreePlacementDrag`. Sized to
                                // match the board's square frame via the
                                // overlay modifier, so local coordinates map
                                // 1:1 onto board squares. Disabled outside
                                // forward-pass mode so game/training views
                                // aren't hijacked.
                                GeometryReader { geo in
                                    let boardSize = min(geo.size.width, geo.size.height)
                                    Color.clear
                                        .contentShape(Rectangle())
                                        .gesture(
                                            DragGesture(minimumDistance: 0)
                                                .onEnded { value in
                                                    let fromSq = Self.squareIndex(
                                                        at: value.startLocation,
                                                        boardSize: boardSize
                                                    )
                                                    let toSq = Self.squareIndex(
                                                        at: value.location,
                                                        boardSize: boardSize
                                                    )
                                                    applyFreePlacementDrag(from: fromSq, to: toSq)
                                                }
                                        )
                                        .allowsHitTesting(forwardPassEditable)
                                }
                            }

                        let rightDisabled = inferenceResult == nil || selectedOverlay == 18 || isGameMode
                        Button(
                            action: { navigateOverlay(1) },
                            label: {
                                Image(systemName: "chevron.right").font(.title3).frame(width: 24)
                            }
                        )
                        .buttonStyle(.plain)
                        .disabled(rightDisabled)
                        .opacity(rightDisabled ? 0.2 : 0.6)
                    }
                }
                .frame(minWidth: 320, maxWidth: 420)

                // Text panel — two fixed-width columns (game stats + training
                // stats) so a mode that shows one never changes size when a
                // mode that shows both (real-training) is active. Each column
                // is gated independently: whichever is relevant for the
                // current mode is rendered, the other is simply omitted. In
                // real-training mode both are shown side-by-side.
                ScrollView {
                    VStack(alignment: .leading, spacing: 8) {
                        if !networkStatus.isEmpty {
                            Text(networkStatus)
                                .foregroundStyle(.secondary)
                        }

                        // Fixed-width columns so the panel never reflows
                        // between modes OR between game results. The game
                        // column has to be wide enough for the longest
                        // Status line the formatter can produce ("Status:
                        // Draw by insufficient material" ≈ 37 chars at
                        // ~8pt/char in monospaced body) — otherwise a draw
                        // by insufficient material or threefold repetition
                        // swells the left column at game end and pushes the
                        // training column rightward.
                        HStack(alignment: .top, spacing: 16) {
                            if isGameMode {
                                Text(gameSnapshot.statsText(
                                    continuousPlay: continuousPlay || realTraining
                                ))
                                .frame(minWidth: 330, alignment: .topLeading)
                            }
                            if isTrainingMode {
                                Text(trainingStatsText())
                                    .frame(minWidth: 260, alignment: .topLeading)
                            }
                            if !isGameMode, !isTrainingMode, let result = inferenceResult {
                                Text(result.textOutput)
                            }
                        }

                        if let trainingError {
                            Text(trainingError).foregroundStyle(.red)
                        }
                    }
                    .font(.system(.body, design: .monospaced))
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
            }
            .layoutPriority(1)

            // Input tensor channel strip
            if let result = inferenceResult, !isGameMode {
                Divider()
                HStack(spacing: 2) {
                    ForEach(0..<18, id: \.self) { channel in
                        let start = channel * 64
                        let isSelected = selectedOverlay == channel + 1
                        VStack(spacing: 1) {
                            ChannelBoardView(values: Array(result.inputTensor[start..<start + 64]))
                                .frame(width: 40, height: 40)
                                .clipShape(RoundedRectangle(cornerRadius: 2))
                                .overlay(
                                    RoundedRectangle(cornerRadius: 2)
                                        .stroke(
                                            isSelected ? Color.accentColor : Color.gray.opacity(0.2),
                                            lineWidth: isSelected ? 2 : 0.5
                                        )
                                )
                            Text(TensorChannelNames.shortNames[channel])
                                .font(.system(size: 8))
                                .foregroundStyle(isSelected ? .primary : .tertiary)
                                .lineLimit(1)
                        }
                    }
                }
            }
        }
        .padding(24)
        // Window minWidth raised from 900 to 1060 to give the two-column
        // text panel (game column 330pt wide enough for the longest
        // Status: line + training column 260pt) enough room alongside the
        // 320-420pt board without horizontal clipping in real-training
        // mode where both columns are visible at once.
        .frame(minWidth: 1060, minHeight: 600)
        .focusable()
        .focusEffectDisabled()
        .onKeyPress(.leftArrow) { navigateOverlay(-1); return .handled }
        .onKeyPress(.rightArrow) { navigateOverlay(1); return .handled }
        .onReceive(snapshotTimer) { _ in
            // Pull the latest game state into @State at most every 100ms.
            // Cheap (single locked struct copy) and bounds UI work even
            // when the game loop is doing hundreds of moves per second.
            gameSnapshot = gameWatcher.snapshot()
            // Same heartbeat pulls the sweep's worker-thread progress and
            // any newly-completed rows into @State so the table grows live.
            if sweepRunning, let box = sweepCancelBox {
                sweepProgress = box.latestProgress
                // Sample process resident memory and feed it into the
                // sweep's per-row peak. The trainer also samples at row
                // boundaries — we just contribute extra samples while a
                // row is in flight so we don't miss mid-step spikes.
                box.recordPeakSample(ChessTrainer.currentPhysFootprintBytes())
                let rows = box.completedRows
                if rows.count != sweepResults.count {
                    sweepResults = rows
                }
            }
            // Same heartbeat pulls live training stats out of the
            // lock-protected box the background training task is writing
            // into. Guarded on step count so a mid-run session that
            // hasn't advanced since the last tick doesn't trigger a
            // useless redraw — and an idle box (after Stop or before
            // first step) stays silent.
            if let box = trainingBox {
                let snap = box.snapshot()
                if snap.stats.steps != (trainingStats?.steps ?? -1) {
                    trainingStats = snap.stats
                    lastTrainStep = snap.lastTiming
                    realRollingPolicyLoss = snap.rollingPolicyLoss
                    realRollingValueLoss = snap.rollingValueLoss
                }
                if let err = snap.error, trainingError == nil {
                    trainingError = err
                }
            }
        }
    }

    private var busyLabel: String {
        if isBuilding { return "Building network..." }
        if realTraining {
            // Show both self-play and training progress at once — the
            // driver alternates between them, so either number alone is
            // misleading.
            let games = gameSnapshot.totalGames
            let buf = replayBuffer?.count ?? 0
            let steps = trainingStats?.steps ?? 0
            return "Self-play: \(games) games, \(buf) positions, \(steps) train steps..."
        }
        if gameSnapshot.isPlaying { return "Game \(gameSnapshot.totalGames + 1), move \(gameSnapshot.moveCount)..." }
        if sweepRunning {
            if let p = sweepProgress {
                return String(format: "Sweep batch size %d, step %d, %.1f s",
                              p.batchSize, p.stepsSoFar, p.elapsedSec)
            }
            return "Sweep starting..."
        }
        if continuousTraining {
            return "Training step \((trainingStats?.steps ?? 0) + 1)..."
        }
        if isTrainingOnce { return "Training one batch..." }
        return "Running inference..."
    }

    // MARK: - Navigation

    private func navigateOverlay(_ direction: Int) {
        guard inferenceResult != nil, !isGameMode else { return }
        let next = selectedOverlay + direction
        if next >= 0, next <= 18 { selectedOverlay = next }
    }

    // MARK: - Actions

    /// Wipe every piece of training/sweep display state. Called when
    /// switching modes (forward pass, play game, build network) so the
    /// previous run's table doesn't linger and hide what the user actually
    /// just did.
    private func clearTrainingDisplay() {
        trainingStats = nil
        lastTrainStep = nil
        trainingError = nil
        trainingBox = nil
        sweepResults = []
        sweepProgress = nil
        sweepDeviceCaps = nil
        // Real-training state — dropped when switching modes so the next
        // run starts from a fresh rolling-loss average and nil buffer
        // reference. The previous run's final numbers are the last thing
        // the user saw; a fresh mode shouldn't inherit them.
        replayBuffer = nil
        realRollingPolicyLoss = nil
        realRollingValueLoss = nil
    }

    private func buildNetwork() {
        isBuilding = true
        networkStatus = ""
        // Drop the trainer (it owns graph state we're about to invalidate
        // by rebuilding) and wipe all training/sweep display state.
        trainer = nil
        clearTrainingDisplay()

        Task {
            let result = await Task.detached(priority: .userInitiated) {
                Self.performBuild()
            }.value

            switch result {
            case .success(let net):
                network = net
                runner = ChessRunner(network: net)
                networkStatus = """
                    Network built in \(String(format: "%.1f", net.buildTimeMs)) ms
                    Parameters: ~2,917,383 (~2.9M)
                    Architecture: 18x8x8 -> stem(128)
                      -> 8 res blocks -> policy(4096) + value(1)
                    """
            case .failure(let error):
                network = nil
                runner = nil
                networkStatus = "Build failed: \(error.localizedDescription)"
            }
            isBuilding = false
        }
    }

    private func runForwardPass() {
        gameWatcher.resetAll()
        gameSnapshot = gameWatcher.snapshot()
        clearTrainingDisplay()
        // Explicit "Run Forward Pass" click resets the overlay to Top Moves;
        // auto re-evals from drag edits preserve whatever overlay the user
        // is currently inspecting.
        selectedOverlay = 0
        reevaluateForwardPass()
    }

    /// Kick off (or coalesce) a forward pass on the current `editableState`.
    /// Called both explicitly from the "Run Forward Pass" button and
    /// automatically after a free-placement drag or side-to-move toggle.
    /// If an inference is already in flight, sets `pendingReeval` so the
    /// in-flight task re-runs once more on completion — that way rapid
    /// drags always resolve to a final inference reflecting the last edit,
    /// without us needing to block the UI on `isEvaluating`.
    private func reevaluateForwardPass() {
        guard let runner else { return }
        if isEvaluating {
            pendingReeval = true
            return
        }
        isEvaluating = true
        let state = editableState
        Task {
            let evalResult = await Task.detached(priority: .userInitiated) {
                Self.performInference(with: runner, state: state)
            }.value
            inferenceResult = evalResult
            isEvaluating = false
            if pendingReeval {
                pendingReeval = false
                reevaluateForwardPass()
            }
        }
    }

    /// Apply one free-placement drag: pick up the piece at `from`, drop it
    /// at `to`. `from` or `to` may be nil (drag started or ended outside
    /// the board, e.g. off-edge), in which case the gesture is either a
    /// no-op (no source) or a deletion (no destination — piece removed).
    /// Captures replace whatever sat on the destination square. Castling
    /// rights, en-passant, and halfmove clock are carried through
    /// unchanged — the network just reads whatever encoding falls out,
    /// which is exactly what a free-placement "what does the net think of
    /// this position" tool should do. Triggers an auto re-eval afterward.
    private func applyFreePlacementDrag(from: Int?, to: Int?) {
        guard let from else { return }
        guard (0..<64).contains(from) else { return }
        if let to, to == from { return }  // Tap without movement — nothing to do.
        var board = editableState.board
        let piece = board[from]
        guard piece != nil else { return }  // Empty square dragged — nothing to do.
        board[from] = nil
        if let to, (0..<64).contains(to) {
            board[to] = piece
        }
        // else: dragged off the board → deletion.
        editableState = GameState(
            board: board,
            currentPlayer: editableState.currentPlayer,
            whiteKingsideCastle: editableState.whiteKingsideCastle,
            whiteQueensideCastle: editableState.whiteQueensideCastle,
            blackKingsideCastle: editableState.blackKingsideCastle,
            blackQueensideCastle: editableState.blackQueensideCastle,
            enPassantSquare: editableState.enPassantSquare,
            halfmoveClock: editableState.halfmoveClock
        )
        reevaluateForwardPass()
    }

    /// Convert a point in the board-overlay's local coordinate space into
    /// a 0-63 square index. Returns nil if the point lies outside the
    /// board's square frame, which the drag handler treats as "off-board"
    /// — a no-op on drag-start, a deletion on drag-end.
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

    private func playSingleGame() {
        inferenceResult = nil
        clearTrainingDisplay()
        gameWatcher.resetCurrentGame()
        gameWatcher.markPlaying(true)
        // Synchronously refresh the snapshot so isBusy reflects the new
        // playing state immediately — the polling task only runs every
        // 100ms, which would otherwise leave a window where the Play
        // button stayed enabled and a fast double-click could spawn two
        // concurrent ChessMachine instances against the same gameWatcher.
        gameSnapshot = gameWatcher.snapshot()

        Task { [network] in
            guard let network else { return }
            let machine = ChessMachine()
            machine.delegate = gameWatcher
            let white = MPSChessPlayer(name: "White", network: network)
            let black = MPSChessPlayer(name: "Black", network: network)
            do {
                let task = try machine.beginNewGame(white: white, black: black)
                _ = await task.value
            } catch {
                gameWatcher.markPlaying(false)
            }
        }
    }

    private func startContinuousPlay() {
        inferenceResult = nil
        clearTrainingDisplay()
        gameWatcher.resetAll()
        continuousPlay = true

        continuousTask = Task { [network] in
            guard let network else { return }

            while !Task.isCancelled {
                gameWatcher.resetCurrentGame()
                gameWatcher.markPlaying(true)

                let machine = ChessMachine()
                machine.delegate = gameWatcher
                let white = MPSChessPlayer(name: "White", network: network)
                let black = MPSChessPlayer(name: "Black", network: network)
                do {
                    let task = try machine.beginNewGame(white: white, black: black)
                    _ = await task.value
                } catch {
                    gameWatcher.markPlaying(false)
                    break
                }

                do {
                    try await Task.sleep(for: .milliseconds(1))
                } catch {
                    break
                }
            }

            await MainActor.run { continuousPlay = false }
        }
    }

    private func stopContinuousPlay() {
        continuousTask?.cancel()
        continuousTask = nil
    }

    /// Stop whichever continuous loop (play, train, or sweep) is currently
    /// active. Bound to escape via the unified Stop button.
    private func stopAnyContinuous() {
        if continuousPlay { stopContinuousPlay() }
        if continuousTraining { stopContinuousTraining() }
        if sweepRunning { stopSweep() }
        if realTraining { stopRealTraining() }
    }

    // MARK: - Training Actions

    /// Build (or reuse) the trainer. The trainer manages its own
    /// training-mode network internally — it doesn't share weights with
    /// the inference network used by Play / Forward Pass — so the inference
    /// network can keep its frozen-stats BN for fast play while the trainer
    /// measures realistic training-step costs through batch-stats BN.
    private func ensureTrainer() -> ChessTrainer? {
        if let trainer { return trainer }
        do {
            let t = try ChessTrainer()
            trainer = t
            return t
        } catch {
            trainingError = "Trainer init failed: \(error.localizedDescription)"
            return nil
        }
    }

    private func trainOnce() {
        guard let trainer = ensureTrainer() else { return }
        // Switching modes — clear any stale game/inference output and
        // start a fresh stats run (single-step still uses TrainingRunStats
        // so the formatter has one path to render).
        inferenceResult = nil
        gameWatcher.resetAll()
        gameSnapshot = gameWatcher.snapshot()
        clearTrainingDisplay()
        isTrainingOnce = true

        Task { [trainer] in
            let result = await Task.detached(priority: .userInitiated) {
                Self.runOneTrainStep(trainer: trainer)
            }.value
            await MainActor.run {
                switch result {
                case .success(let timing):
                    var stats = TrainingRunStats()
                    stats.record(timing)
                    trainingStats = stats
                    lastTrainStep = timing
                case .failure(let error):
                    trainingError = error.localizedDescription
                }
                isTrainingOnce = false
            }
        }
    }

    private func startContinuousTraining() {
        guard let trainer = ensureTrainer() else { return }
        inferenceResult = nil
        gameWatcher.resetAll()
        gameSnapshot = gameWatcher.snapshot()
        clearTrainingDisplay()

        // Seed trainingStats with a fresh zero so the formatter shows
        // "Steps done: 0" immediately; the heartbeat poller replaces it
        // with the real stats out of the box as soon as the first step
        // lands.
        let box = TrainingLiveStatsBox(rollingWindow: Self.rollingLossWindow)
        trainingBox = box
        trainingStats = TrainingRunStats()
        continuousTraining = true

        trainingTask = Task { [trainer, box] in
            var shouldStop = false
            while !Task.isCancelled && !shouldStop {
                let result = await Task.detached(priority: .userInitiated) {
                    Self.runOneTrainStep(trainer: trainer)
                }.value
                switch result {
                case .success(let timing):
                    box.recordStep(timing)
                case .failure(let error):
                    box.recordError(error.localizedDescription)
                    shouldStop = true
                }
            }
            await MainActor.run { continuousTraining = false }
        }
    }

    private func stopContinuousTraining() {
        trainingTask?.cancel()
        trainingTask = nil
    }

    nonisolated private static func runOneTrainStep(trainer: ChessTrainer) -> Result<TrainStepTiming, Error> {
        Result { try trainer.trainStep(batchSize: trainingBatchSize) }
    }

    // MARK: - Real Training (Self-Play) Actions

    /// Kick off real-data training: self-play games against the inference
    /// network feed a replay buffer; the trainer samples minibatches out of
    /// the buffer and steps SGD on them. Alternates one game → K train
    /// steps → next game, so the UI shows both activities live. Trainer
    /// weights are reset at the top of the loop so each run starts from
    /// fresh random init and the rolling loss curve is meaningful.
    private func startRealTraining() {
        guard let trainer = ensureTrainer(), let network else { return }
        inferenceResult = nil
        gameWatcher.resetAll()
        gameSnapshot = gameWatcher.snapshot()
        clearTrainingDisplay()

        let buffer = ReplayBuffer(capacity: Self.replayBufferCapacity)
        replayBuffer = buffer
        let box = TrainingLiveStatsBox(rollingWindow: Self.rollingLossWindow)
        trainingBox = box
        realRollingPolicyLoss = nil
        realRollingValueLoss = nil
        trainingStats = TrainingRunStats()
        realTraining = true

        realTrainingTask = Task { [trainer, network, buffer, box] in
            // Fresh weights each run — otherwise the loss curve inherits
            // whatever weights the previous session (random-data training,
            // a sweep, or a prior self-play run) left behind and the user
            // can't tell whether real data is actually driving loss down.
            do {
                try await Task.detached(priority: .userInitiated) {
                    try trainer.resetNetwork()
                }.value
            } catch {
                box.recordError("Reset failed: \(error.localizedDescription)")
                await MainActor.run {
                    realTraining = false
                    realTrainingTask = nil
                }
                return
            }

            var shouldStop = false
            while !Task.isCancelled && !shouldStop {
                // --- Play one self-play game with positions going into the
                //     buffer via each MPSChessPlayer's onGameEnded push. ---
                gameWatcher.resetCurrentGame()
                gameWatcher.markPlaying(true)

                let machine = ChessMachine()
                machine.delegate = gameWatcher
                let white = MPSChessPlayer(
                    name: "White",
                    network: network,
                    replayBuffer: buffer
                )
                let black = MPSChessPlayer(
                    name: "Black",
                    network: network,
                    replayBuffer: buffer
                )
                do {
                    let task = try machine.beginNewGame(white: white, black: black)
                    _ = await task.value
                } catch {
                    gameWatcher.markPlaying(false)
                    break
                }

                if Task.isCancelled { break }

                // --- Train K steps, if the buffer is warm enough for the
                //     samples to cover more than one game's positions. ---
                if buffer.count < Self.minBufferBeforeTraining { continue }

                for _ in 0..<Self.trainStepsPerGame {
                    if Task.isCancelled { break }
                    guard let batch = buffer.sample(count: Self.trainingBatchSize) else { break }

                    let result = await Task.detached(priority: .userInitiated) {
                        Result { try trainer.trainStep(batch: batch) }
                    }.value

                    switch result {
                    case .success(let timing):
                        box.recordStep(timing)
                    case .failure(let error):
                        box.recordError(error.localizedDescription)
                        shouldStop = true
                    }
                    if shouldStop { break }
                }
            }

            await MainActor.run {
                realTraining = false
                realTrainingTask = nil
            }
        }
    }

    private func stopRealTraining() {
        realTrainingTask?.cancel()
        realTrainingTask = nil
    }

    // MARK: - Sweep Actions

    private func startSweep() {
        guard let trainer = ensureTrainer() else { return }
        inferenceResult = nil
        gameWatcher.resetAll()
        gameSnapshot = gameWatcher.snapshot()
        clearTrainingDisplay()
        sweepRunning = true
        // Snapshot device caps once at sweep start so the header has a
        // stable reference point regardless of what else is running.
        sweepDeviceCaps = trainer.deviceMemoryCaps()

        let sizes = Self.sweepSizes
        let secondsPerSize = Self.sweepSecondsPerSize
        let cancelBox = CancelBox()
        sweepCancelBox = cancelBox

        sweepTask = Task { [trainer, cancelBox] in
            // Reset the trainer's internal weights so loss starts fresh
            // and small batches don't inherit overfit weights from prior
            // continuous-training runs.
            do {
                try await Task.detached(priority: .userInitiated) {
                    try trainer.resetNetwork()
                }.value
            } catch {
                await MainActor.run {
                    trainingError = "Reset failed: \(error.localizedDescription)"
                    sweepRunning = false
                    sweepCancelBox = nil
                }
                return
            }

            let result = await Task.detached(priority: .userInitiated) {
                Self.runSweep(
                    trainer: trainer,
                    sizes: sizes,
                    secondsPerSize: secondsPerSize,
                    cancelBox: cancelBox
                )
            }.value

            await MainActor.run {
                // Pull any final completed rows out of the box (the
                // heartbeat may have a stale cached snapshot).
                sweepResults = cancelBox.completedRows
                if case .failure(let error) = result {
                    trainingError = "Sweep failed: \(error.localizedDescription)"
                }
                sweepProgress = nil
                sweepCancelBox = nil
                sweepRunning = false
            }
        }
    }

    private func stopSweep() {
        // Flip the box directly — the worker polls this between steps and
        // breaks out of the loops. Cancelling the Swift Task wouldn't help
        // because Task.isCancelled doesn't propagate to the unstructured
        // detached worker we spawned, and the worker doesn't await anything
        // it could check Task.isCancelled on.
        sweepCancelBox?.cancel()
    }

    nonisolated private static func runSweep(
        trainer: ChessTrainer,
        sizes: [Int],
        secondsPerSize: Double,
        cancelBox: CancelBox
    ) -> Result<[SweepRow], Error> {
        Result {
            try trainer.runSweep(
                sizes: sizes,
                targetSecondsPerSize: secondsPerSize,
                cancelled: { cancelBox.isCancelled },
                progress: { batchSize, stepsSoFar, elapsed in
                    cancelBox.updateProgress(
                        SweepProgress(
                            batchSize: batchSize,
                            stepsSoFar: stepsSoFar,
                            elapsedSec: elapsed
                        )
                    )
                },
                recordPeakSampleNow: {
                    // Worker-thread sample — guarantees every row gets a
                    // fresh reading at start and end even if no UI
                    // heartbeat fired during the row's lifetime.
                    cancelBox.recordPeakSample(ChessTrainer.currentPhysFootprintBytes())
                },
                consumeRowPeak: {
                    cancelBox.takeRowPeak()
                },
                onRowCompleted: { row in
                    // Worker thread — push the completed row into the box
                    // so the heartbeat can pick it up. Lets the table grow
                    // one row at a time as the sweep progresses.
                    cancelBox.appendRow(row)
                }
            )
        }
    }

    // MARK: - Training Stats Display

    private func trainingStatsText() -> String {
        let dash = "--"

        // Sweep results trump the per-step display. Once a sweep starts or
        // completes, the table is what the user came here for.
        if sweepRunning || !sweepResults.isEmpty {
            return sweepStatsText()
        }

        let isSelfPlay = realTraining || replayBuffer != nil
        let mode: String
        if isSelfPlay {
            mode = "Self-Play"
        } else if continuousTraining {
            mode = "Continuous"
        } else {
            mode = "Single Step"
        }
        var lines: [String] = []
        lines.append("Training (\(mode))")
        lines.append("  Batch size:  \(Self.trainingBatchSize)")
        // Learning rate — read off the trainer so we can't drift out of
        // sync with what the graph is actually applying. Shown in every
        // training mode since it's the same knob across all of them.
        let lrStr: String
        if let trainer {
            lrStr = String(format: "%.1e", trainer.learningRate)
        } else {
            lrStr = dash
        }
        lines.append("  Learn rate: \(lrStr)")

        // Self-Play adds two extra header lines (replay buffer fill, rolling
        // loss). Both are present from the first render of a self-play run,
        // so they don't cause mid-run layout shifts. They're omitted in
        // single-step / continuous modes because those modes have no replay
        // buffer and no meaningful rolling-loss window separate from the
        // last-step loss shown below.
        if isSelfPlay {
            let bufCount = replayBuffer?.count ?? 0
            let bufCap = replayBuffer?.capacity ?? Self.replayBufferCapacity
            let bufStr = String(format: "%6d / %d", bufCount, bufCap)
            lines.append("  Buffer:     \(bufStr)")
            let policyStr: String
            if let loss = realRollingPolicyLoss {
                policyStr = String(format: "%+.6f", loss)
            } else {
                policyStr = dash
            }
            let valueStr: String
            if let loss = realRollingValueLoss {
                valueStr = String(format: "%+.6f", loss)
            } else {
                valueStr = dash
            }
            lines.append("  Roll policy: \(policyStr)")
            lines.append("  Roll value:  \(valueStr)")
        }
        lines.append("")

        if let last = lastTrainStep {
            lines.append("Last Step")
            lines.append(String(format: "  Total:       %7.2f ms", last.totalMs))
            lines.append(String(format: "  Data prep:   %7.2f ms", last.dataPrepMs))
            lines.append(String(format: "  GPU run:     %7.2f ms", last.gpuRunMs))
            lines.append(String(format: "  Readback:    %7.2f ms", last.readbackMs))
            lines.append(String(format: "  Loss:        %+.6f", last.loss))
            lines.append(String(format: "    policy:    %+.6f", last.policyLoss))
            lines.append(String(format: "    value:     %+.6f", last.valueLoss))
            lines.append("")
        }

        if let stats = trainingStats {
            let stepsStr = stats.steps.formatted(.number.grouping(.automatic))
            // Train time is the sum of per-step wall times — in real-
            // training mode this excludes self-play, so the rate numbers
            // below reflect trainer throughput rather than session clock.
            let trainTimeStr = stats.steps > 0
                ? String(format: "%.2f s", stats.trainingSeconds)
                : dash
            let avgTotal = stats.steps > 0 ? String(format: "%7.2f ms", stats.avgStepMs) : dash
            let avgGpu = stats.steps > 0 ? String(format: "%7.2f ms", stats.avgGpuMs) : dash
            let minStr = stats.steps > 0 ? String(format: "%7.2f ms", stats.minStepMs) : dash
            let maxStr = stats.steps > 0 ? String(format: "%7.2f ms", stats.maxStepMs) : dash
            let rateStr = stats.steps > 0 ? String(format: "%.2f", stats.stepsPerSecond) : dash
            let posRateStr: String
            if stats.steps > 0 {
                let posPerSec = stats.positionsPerSecond(batchSize: Self.trainingBatchSize)
                posRateStr = Int(posPerSec.rounded())
                    .formatted(.number.grouping(.automatic))
            } else {
                posRateStr = dash
            }
            let projStr = stats.steps > 0 ? String(format: "%.2f s", stats.projectedSecPer250Steps) : dash

            lines.append("Run Totals")
            lines.append("  Steps done:  \(stepsStr)")
            lines.append("  Train time:  \(trainTimeStr)")
            lines.append("  Avg total:   \(avgTotal)")
            lines.append("  Avg GPU:     \(avgGpu)")
            lines.append("  Min step:    \(minStr)")
            lines.append("  Max step:    \(maxStr)")
            lines.append("  Steps/sec:   \(rateStr)")
            lines.append("  Pos/sec:     \(posRateStr)")
            lines.append("  Proj 250×:   \(projStr)")
        }

        return lines.joined(separator: "\n")
    }

    /// Format the sweep results as a fixed-column monospaced table.
    /// Updates live as rows complete; after the run finishes, includes
    /// the throughput peak.
    private func sweepStatsText() -> String {
        var lines: [String] = []
        lines.append("Batch Size Sweep (training-mode BN)")
        lines.append(String(format: "  Target: %.0f s per size", Self.sweepSecondsPerSize))
        if let caps = sweepDeviceCaps {
            lines.append(String(
                format: "  Device:  recommendedMaxWorkingSetSize=%.2f GB,  maxBufferLength=%.2f GB",
                Self.bytesToGB(caps.recommendedMaxWorkingSet),
                Self.bytesToGB(caps.maxBufferLength)
            ))
            lines.append(String(
                format: "           currentAllocatedSize=%.2f GB (at sweep start)",
                Self.bytesToGB(caps.currentAllocated)
            ))
        }
        lines.append("")

        lines.append(" Batch    Warmup    Steps    Time   Avg/step   Avg GPU    Pos/sec     Loss      Peak")
        lines.append(" -----    ------    -----    ----   --------   -------    -------     ----      ----")

        for row in sweepResults {
            switch row {
            case .completed(let r):
                let posPerSec = Int(r.positionsPerSec.rounded())
                    .formatted(.number.grouping(.automatic))
                    .padding(toLength: 9, withPad: " ", startingAt: 0)
                lines.append(String(
                    format: "%6d  %7.1f ms %6d %6.1fs  %7.2f ms %7.2f ms  %@  %+.3f  %6.2f GB",
                    r.batchSize,
                    r.warmupMs,
                    r.steps,
                    r.elapsedSec,
                    r.avgStepMs,
                    r.avgGpuMs,
                    posPerSec,
                    r.lastLoss,
                    Self.bytesToGB(r.peakResidentBytes)
                ))
            case .skipped(let s):
                let reason: String
                if s.exceededWorkingSet && s.exceededBufferLength {
                    reason = "working-set & buffer cap"
                } else if s.exceededWorkingSet {
                    reason = "working-set cap"
                } else {
                    reason = "buffer cap"
                }
                lines.append(String(
                    format: "%6d  skipped — est RAM %6.2f GB, max buf %6.2f GB  [%@]",
                    s.batchSize,
                    Self.bytesToGB(s.estimatedBytes),
                    Self.bytesToGB(s.largestBufferBytes),
                    reason
                ))
            }
        }

        if sweepRunning {
            lines.append("")
            if let p = sweepProgress {
                lines.append(String(
                    format: "  Running: batch size %d, %d steps, %.1fs",
                    p.batchSize, p.stepsSoFar, p.elapsedSec
                ))
            } else {
                lines.append("  Starting...")
            }
        } else if !sweepResults.isEmpty {
            let completed: [SweepResult] = sweepResults.compactMap {
                if case .completed(let r) = $0 { return r } else { return nil }
            }
            if let best = completed.max(by: { $0.positionsPerSec < $1.positionsPerSec }) {
                lines.append("")
                lines.append(String(
                    format: "  Best: batch size %d at %d positions/sec",
                    best.batchSize,
                    Int(best.positionsPerSec.rounded())
                ))
            }
        }

        return lines.joined(separator: "\n")
    }

    private static func bytesToGB(_ bytes: UInt64) -> Double {
        Double(bytes) / 1_073_741_824.0
    }

    // MARK: - Background Work

    nonisolated private static func performBuild() -> Result<ChessMPSNetwork, Error> {
        Result { try ChessMPSNetwork(.randomWeights) }
    }

    nonisolated private static func performInference(
        with runner: ChessRunner,
        state: GameState
    ) -> EvaluationResult {
        var lines: [String] = []
        var topMoves: [MoveVisualization] = []
        let board = BoardEncoder.encode(state)
        let flip = state.currentPlayer == .black

        do {
            let inference = try runner.evaluate(board: board, pieces: state.board, flip: flip)
            topMoves = inference.topMoves

            lines.append(String(format: "Forward pass: %.2f ms", inference.inferenceTimeMs))
            lines.append("")
            lines.append("Value Head")
            lines.append(String(format: "  Output: %+.6f", inference.value))
            lines.append(String(format: "  %.1f%% win / %.1f%% loss",
                                (inference.value + 1) / 2 * 100, (1 - inference.value) / 2 * 100))
            lines.append("")
            lines.append("Policy Head (Top 4)")
            for (rank, move) in inference.topMoves.enumerated() {
                let fromName = BoardEncoder.squareName(move.fromRow * 8 + move.fromCol)
                let toName = BoardEncoder.squareName(move.toRow * 8 + move.toCol)
                let rankCol = String(rank + 1).padding(toLength: 4, withPad: " ", startingAt: 0)
                let moveCol = "\(fromName)-\(toName)".padding(toLength: 8, withPad: " ", startingAt: 0)
                lines.append("  \(rankCol)\(moveCol)\(String(format: "%.4f%%", move.probability * 100))")
            }
            lines.append("")
            lines.append("Policy Stats")
            lines.append(String(format: "  Sum: %.6f", inference.policy.reduce(0, +)))
            let nonZeroCount = inference.policy.filter { $0 > 1e-10 }.count
            lines.append(String(format: "  Non-negligible: %d / 4096", nonZeroCount))
            if let maxProb = inference.policy.max(), let minProb = inference.policy.min() {
                lines.append(String(format: "  Min: %.6f", minProb))
                lines.append(String(format: "  Max: %.6f", maxProb))
            }
        } catch {
            lines.append("Error: \(error.localizedDescription)")
        }

        return EvaluationResult(topMoves: topMoves, textOutput: lines.joined(separator: "\n"), inputTensor: board)
    }
}

#Preview {
    ContentView()
}
