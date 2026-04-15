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

/// Which board the Play and Train UI is showing. `.gameRun` is the
/// live self-play game (the only option before this feature existed);
/// `.candidateTest` swaps in the free-placement forward-pass editor so
/// the user can probe a fixed test position and watch the network's
/// evaluation of it drift as training progresses in the background.
enum PlayAndTrainBoardMode: Sendable, Hashable {
    case gameRun
    case candidateTest
}

// MARK: - Arena Tournament Types

/// Live view of an in-progress arena tournament. Updated by the driver
/// task after each finished game via a lock-protected `TournamentLiveBox`
/// and mirrored into `@State` by the UI heartbeat. "Candidate" and
/// "champion" here refer to the two networks being compared — not to
/// any chess color. Colors alternate every game.
struct TournamentProgress: Sendable {
    let currentGame: Int
    let totalGames: Int
    let candidateWins: Int
    let championWins: Int
    let draws: Int
    /// Wall-clock time the tournament started, captured once in
    /// `runArenaTournament` before the first game and carried through
    /// every per-game update so the busy label can compute live
    /// elapsed time via `Date().timeIntervalSince(startTime)` on each
    /// render without needing a separate ticking timer.
    let startTime: Date

    /// AlphaZero-style score: (wins + 0.5 * draws) / games_played.
    /// Pure draws → 0.5. Candidate sweeping every decisive game with
    /// zero losses → 1.0. Used with a 0.55 promotion threshold.
    var candidateScore: Double {
        let played = currentGame > 0 ? currentGame : 1
        return (Double(candidateWins) + 0.5 * Double(draws)) / Double(played)
    }
}

/// Historical record of a completed tournament, appended to the
/// `tournamentHistory` array after each arena finishes. Displayed in
/// the training stats text panel so the user can see promotion
/// decisions across the session.
struct TournamentRecord: Sendable, Identifiable {
    let id = UUID()
    /// `trainingStats.steps` at the moment the tournament finished.
    let finishedAtStep: Int
    let candidateWins: Int
    let championWins: Int
    let draws: Int
    let score: Double
    let promoted: Bool
    /// Total wall-clock time the tournament took from the initial
    /// trainer → candidate sync through the last game. Promotion copy
    /// after the score threshold check is not included.
    let durationSec: Double
}

/// Lock-protected holder for live tournament progress, shared between
/// the driver task (writer, one update per finished game) and the UI
/// heartbeat (reader, polling at 60 Hz). Same pattern as
/// `TrainingLiveStatsBox` and `CancelBox` — an `NSLock` guards all
/// state, so the class is safely `@unchecked Sendable`.
final class TournamentLiveBox: @unchecked Sendable {
    private let lock = NSLock()
    private var _progress: TournamentProgress?

    func update(_ progress: TournamentProgress) {
        lock.lock()
        defer { lock.unlock() }
        _progress = progress
    }

    func snapshot() -> TournamentProgress? {
        lock.lock()
        defer { lock.unlock() }
        return _progress
    }

    func clear() {
        lock.lock()
        defer { lock.unlock() }
        _progress = nil
    }
}

// MARK: - Parallel Worker Coordination

/// Request/ack gate used to briefly pause one of the parallel Play and
/// Train workers (self-play or training) while another task needs
/// exclusive access to one of the four shared networks for a few
/// milliseconds — e.g. arena start copying trainer → candidate and
/// champion → arena champion, or arena end copying candidate →
/// champion on promotion. A coordinator calls `pauseAndWait()` to
/// request + wait for the worker to enter its wait state, does its
/// protected work, then calls `resume()`. The worker polls
/// `isRequestedToPause` at natural iteration boundaries (between
/// games for self-play, between SGD steps for training) and spins
/// on a 5 ms sleep until released.
///
/// Cancellation-safe: the worker's spin-wait checks `Task.isCancelled`
/// on every iteration so clicking Stop during a pause exits the
/// wait loop immediately. Lock-protected state, `@unchecked Sendable`.
final class WorkerPauseGate: @unchecked Sendable {
    private let lock = NSLock()
    private var _requested = false
    private var _isWaiting = false

    /// Polled by the worker at each iteration boundary.
    var isRequestedToPause: Bool {
        lock.lock()
        defer { lock.unlock() }
        return _requested
    }

    /// Called by the worker when it enters its spin-wait state, so
    /// the coordinator knows it's safe to start the protected work.
    func markWaiting() {
        lock.lock()
        defer { lock.unlock() }
        _isWaiting = true
    }

    /// Called by the worker when it leaves its spin-wait state and
    /// resumes normal iteration.
    func markRunning() {
        lock.lock()
        defer { lock.unlock() }
        _isWaiting = false
    }

    /// Coordinator: flip the pause request and spin-wait until the
    /// worker has acknowledged by entering its wait state. Returns
    /// once it's safe to perform the protected work.
    func pauseAndWait() async {
        setRequested(true)
        while !Task.isCancelled {
            if readIsWaiting() { return }
            try? await Task.sleep(for: .milliseconds(5))
        }
    }

    /// Synchronous helpers so `pauseAndWait()` doesn't hold an
    /// `NSLock` across an `await` — Swift 6 strict concurrency
    /// forbids calling `NSLock.lock()` from an async context.
    private func setRequested(_ value: Bool) {
        lock.lock()
        defer { lock.unlock() }
        _requested = value
    }

    private func readIsWaiting() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return _isWaiting
    }

    /// Coordinator: release the worker. Clears the request flag so
    /// the worker's next spin-wait iteration sees it and resumes.
    func resume() {
        lock.lock()
        defer { lock.unlock() }
        _requested = false
    }
}

/// Lock-protected trigger inbox for the arena coordinator task. The
/// training worker fires the trigger when the 30-minute auto cadence
/// elapses; the UI fires it via the Run Arena button. The arena
/// coordinator task polls `consume()` in its main loop and runs an
/// arena whenever the trigger is pending. `recordArenaCompleted()`
/// resets the "last arena" timestamp so the auto-fire math stays
/// accurate across both the automatic and manual paths.
final class ArenaTriggerBox: @unchecked Sendable {
    private let lock = NSLock()
    private var _pending = false
    private var _lastArenaTime: Date

    init(startTime: Date = Date()) {
        self._lastArenaTime = startTime
    }

    /// Check whether enough wall clock has elapsed since the last
    /// arena to auto-trigger another one. Returns false if a trigger
    /// is already pending, so the training worker doesn't queue
    /// multiple auto-triggers for the same deadline.
    func shouldAutoTrigger(interval: TimeInterval) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        if _pending { return false }
        return Date().timeIntervalSince(_lastArenaTime) >= interval
    }

    /// Set the pending flag. The arena coordinator's next poll will
    /// consume it and start an arena.
    func trigger() {
        lock.lock()
        defer { lock.unlock() }
        _pending = true
    }

    /// Poll the trigger. Returns true and clears the pending flag if
    /// a trigger was waiting; returns false otherwise.
    func consume() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        if _pending {
            _pending = false
            return true
        }
        return false
    }

    /// Record that an arena just finished. Resets the wall-clock
    /// reference for subsequent `shouldAutoTrigger` checks so the
    /// next auto-fire happens `interval` seconds from now, not from
    /// the previous last-arena time.
    func recordArenaCompleted() {
        lock.lock()
        defer { lock.unlock() }
        _lastArenaTime = Date()
    }

    /// True if a trigger is currently pending (used for UI
    /// disable-while-queued semantics).
    var isPending: Bool {
        lock.lock()
        defer { lock.unlock() }
        return _pending
    }
}

/// Lock-protected flag indicating an arena tournament is currently
/// in progress. Used to mutually exclude the Candidate test probe
/// from the arena, since both touch the candidate inference network
/// and the probe can't write while the arena is reading.
final class ArenaActiveFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var _active = false

    var isActive: Bool {
        lock.lock()
        defer { lock.unlock() }
        return _active
    }

    func set() {
        lock.lock()
        defer { lock.unlock() }
        _active = true
    }

    func clear() {
        lock.lock()
        defer { lock.unlock() }
        _active = false
    }
}

/// Lock-protected rolling counters for the parallel self-play and
/// training workers. Each worker increments its own counters after
/// finishing one unit of work (one game for self-play, one SGD step
/// for training), and the UI heartbeat reads both to compute the
/// live positions-per-second rates shown in the busy label. Values
/// are monotonic for the life of the Play and Train session; wall-
/// clock rate is computed against `sessionStart`.
final class ParallelWorkerStatsBox: @unchecked Sendable {
    private let lock = NSLock()
    private var _selfPlayGames: Int = 0
    private var _selfPlayPositions: Int = 0
    private var _trainingSteps: Int = 0
    let sessionStart: Date

    init(sessionStart: Date = Date()) {
        self.sessionStart = sessionStart
    }

    func recordSelfPlayGame(positions: Int) {
        lock.lock()
        defer { lock.unlock() }
        _selfPlayGames += 1
        _selfPlayPositions += positions
    }

    func recordTrainingStep() {
        lock.lock()
        defer { lock.unlock() }
        _trainingSteps += 1
    }

    struct Snapshot: Sendable {
        let selfPlayGames: Int
        let selfPlayPositions: Int
        let trainingSteps: Int
        let sessionStart: Date
    }

    func snapshot() -> Snapshot {
        lock.lock()
        defer { lock.unlock() }
        return Snapshot(
            selfPlayGames: _selfPlayGames,
            selfPlayPositions: _selfPlayPositions,
            trainingSteps: _trainingSteps,
            sessionStart: sessionStart
        )
    }
}

// MARK: - Content View

struct ContentView: View {
    // Network
    @State private var network: ChessMPSNetwork?
    @State private var runner: ChessRunner?
    @State private var networkStatus = ""
    @State private var isBuilding = false
    /// Separate inference-mode network used *only* by the Candidate test
    /// probe during Play and Train. Distinct from `network` (the
    /// "champion" used by self-play / Play Game / Run Forward Pass)
    /// because we explicitly want the champion to stay frozen at
    /// whatever weights it was built with until a future arena-based
    /// promotion step decides otherwise. The trainer's current SGD
    /// state is copied into this candidate inference network after
    /// each training block, so the probe reflects "what the
    /// candidate-in-training thinks right now" without disturbing the
    /// network that's actually generating self-play data.
    ///
    /// Cached across Play and Train sessions so the ~100 ms
    /// MPSGraph-build cost only happens once per app launch, not on
    /// every start.
    @State private var candidateInferenceNetwork: ChessMPSNetwork?
    /// ChessRunner wrapping `candidateInferenceNetwork`. Built alongside
    /// the network and used by `fireCandidateProbeIfNeeded` via the
    /// same `performInference` code path as the pure forward-pass
    /// mode — no special-case trainer-probe branch needed now that
    /// the candidate has a dedicated inference-mode network of its own.
    @State private var candidateRunner: ChessRunner?
    /// Fourth network — dedicated to the arena's "champion side"
    /// player. During arena, champion is copied into this network
    /// once at the start (under a brief self-play pause) and the
    /// arena games run on this network alone, leaving the real
    /// champion free for continuous self-play throughout the
    /// tournament. Built lazily on the first Play and Train session
    /// and cached for the life of the app.
    @State private var arenaChampionNetwork: ChessMPSNetwork?
    /// Live-progress snapshot from the parallel workers, mirrored
    /// from `parallelWorkerStatsBox` by the heartbeat. Nil outside of
    /// Play and Train sessions.
    @State private var parallelStats: ParallelWorkerStatsBox.Snapshot?
    /// Lock-protected counter box shared across the parallel self-
    /// play and training worker tasks. Writers (workers) call
    /// `recordSelfPlayGame` / `recordTrainingStep`; the UI heartbeat
    /// polls `snapshot()` and mirrors into `parallelStats` so the
    /// busy label shows live positions/sec rates.
    @State private var parallelWorkerStatsBox: ParallelWorkerStatsBox?
    /// Shared cancellation-aware flag set while an arena tournament
    /// is in flight. The Candidate test probe checks this and skips
    /// firing so probe and arena never contend on the candidate
    /// inference network.
    @State private var arenaActiveFlag: ArenaActiveFlag?
    /// Trigger inbox the arena coordinator polls. Set by the training
    /// worker's 30-minute auto check and by the Run Arena button.
    @State private var arenaTriggerBox: ArenaTriggerBox?
    /// True while an arena is running — mirror of `arenaActiveFlag`
    /// maintained by the heartbeat for UI purposes (disabling the
    /// Run Arena button, suppressing probe activity on screen).
    @State private var isArenaRunning: Bool = false

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
    /// Which board the Play and Train view is showing. `.gameRun` is the
    /// live self-play game (current behavior); `.candidateTest` shows the
    /// editable forward-pass board alongside the still-running training
    /// loop so the user can watch the network's evaluation of a fixed
    /// test position evolve as the weights update.
    @State private var playAndTrainBoardMode: PlayAndTrainBoardMode = .gameRun
    /// Set when the user edits the candidate test board (drag, side-to-move
    /// toggle, Board picker flip) while Play and Train is running. The
    /// Play and Train driver task checks this at natural gap points (end
    /// of game, end of training block) and fires a forward-pass probe
    /// there — cooperatively, so inference never races with self-play or
    /// training on the shared network graph.
    @State private var candidateProbeDirty: Bool = false
    /// Wall-clock timestamp of the last candidate-test probe. Combined
    /// with `candidateProbeIntervalSec` to enforce the 15-second cadence:
    /// gap-point checks fire a probe whenever this elapsed interval has
    /// passed, regardless of whether the user has edited anything.
    @State private var lastCandidateProbeTime: Date = .distantPast
    /// Number of candidate-test probes that have actually fired since
    /// Play and Train started. Displayed in the training stats text so
    /// the user can confirm probes are running — the visible arrows may
    /// barely change between 15-second probes (network deltas per 10
    /// training steps are tiny), and without a counter it's impossible
    /// to distinguish "firing but imperceptible" from "stuck".
    @State private var candidateProbeCount: Int = 0

    // MARK: - Arena Tournament State
    //
    // Arena tournaments run inside the Play and Train driver task every
    // `stepsPerTournament` SGD steps. They play N games candidate vs
    // champion, alternating colors, pause self-play + training for the
    // duration, and either promote the candidate into the champion
    // (AlphaZero-style 0.55 score threshold) or leave the champion
    // alone. History is appended to `tournamentHistory` for display.

    /// Live progress mirrored from `tournamentBox` by the heartbeat.
    /// Non-nil while a tournament is running; nil otherwise.
    @State private var tournamentProgress: TournamentProgress?
    /// Lock-protected box the driver task writes into after each arena
    /// game completes. The heartbeat polls it and lifts the latest
    /// progress into `tournamentProgress` so the busy label and the
    /// text panel update live without cross-actor hops per game.
    @State private var tournamentBox: TournamentLiveBox?
    /// History of all completed tournaments in this session. Appended
    /// after each arena finishes. In-memory only for now — disk
    /// persistence is deferred.
    @State private var tournamentHistory: [TournamentRecord] = []
    /// Total number of games a single arena plays. 200 gives us enough
    /// decisive games (~26 at the current ~13% decisive rate with
    /// random networks) for the 0.55 score threshold to be meaningful.
    /// Colors alternate every game, so candidate and champion each get
    /// 100 games as white and 100 as black.
    nonisolated static let tournamentGames = 200
    /// Candidate-score threshold for promotion. The AlphaZero paper's
    /// default. Demands the candidate score at least 110/200 points,
    /// which in a draw-heavy regime translates to winning the large
    /// majority of decisive games.
    nonisolated static let tournamentPromoteThreshold = 0.55
    /// Wall-clock seconds between automatic arena tournaments in
    /// parallel mode. Checked inside the training worker between SGD
    /// steps; when `now - lastTournamentTime >= secondsPerTournament`
    /// and no arena is already running, a new arena is spawned.
    /// 30 minutes is the default — long enough that arenas are
    /// consequential events rather than noise, short enough that a
    /// session hits several of them. Also available on demand via
    /// the Run Arena button regardless of this cadence.
    nonisolated static let secondsPerTournament: TimeInterval = 30 * 60
    /// Minimum wall-clock interval between scheduled candidate-test probes.
    /// Actual cadence drifts slightly — a probe only fires at the next
    /// driver gap after the interval has elapsed — and that's fine: this
    /// is a cheap eval-drift visualization, not a precision timer.
    nonisolated static let candidateProbeIntervalSec: TimeInterval = 15
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
        // Candidate test mode pulls from the editable state even though
        // Play and Train is running a game in the background — that's
        // the whole point of the mode, to look at a fixed test position.
        if isCandidateTestActive { return editableState.board }
        if isGameMode { return gameSnapshot.state.board }
        return editableState.board
    }

    /// True when the Play and Train Board picker is currently showing
    /// the editable Candidate test board (as opposed to the live
    /// self-play game). Centralizes the "override game mode with forward-
    /// pass UI" decision so every site that needs to branch on it reads
    /// the same condition.
    private var isCandidateTestActive: Bool {
        realTraining && playAndTrainBoardMode == .candidateTest
    }

    /// True when the forward-pass UI elements (overlay, channel strip,
    /// side-to-move picker, chevrons, editable drag) should be visible —
    /// either in pure forward-pass mode or in Candidate test mode during
    /// Play and Train. Pure training modes (Train Once, Train Continuous,
    /// Sweep) don't set this, and neither does game-run-mode Play and
    /// Train.
    private var showForwardPassUI: Bool {
        isCandidateTestActive || (!isGameMode && !isTrainingMode)
    }

    /// Whether the forward-pass free-placement editor should be live: the
    /// forward-pass UI is visible AND the network is built so inference
    /// can actually run. Gates the drag gesture, the side-to-move picker,
    /// and anything else that directly mutates `editableState`.
    private var forwardPassEditable: Bool {
        networkReady && showForwardPassUI
    }

    /// Binding that drives the Play and Train Board picker. Writes that
    /// flip the mode to `.candidateTest` mark the probe dirty so the
    /// driver fires an immediate forward pass on the next gap — that way
    /// the user sees arrows the moment they switch, rather than having
    /// to wait up to 15 s for the interval probe to trigger.
    private var playAndTrainBoardBinding: Binding<PlayAndTrainBoardMode> {
        Binding(
            get: { playAndTrainBoardMode },
            set: { newValue in
                playAndTrainBoardMode = newValue
                if newValue == .candidateTest {
                    candidateProbeDirty = true
                }
            }
        )
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
                requestForwardPassReeval()
            }
        )
    }

    /// Route a forward-pass re-eval request through the correct path for
    /// the current mode. In pure forward-pass mode we fire immediately
    /// via `reevaluateForwardPass()`, which runs on a detached task. In
    /// Candidate test mode during Play and Train we set
    /// `candidateProbeDirty` instead — the Play and Train driver task
    /// picks it up at the next cooperative gap point, so the probe
    /// never races with self-play or training on the shared network.
    private func requestForwardPassReeval() {
        if isCandidateTestActive {
            candidateProbeDirty = true
            return
        }
        reevaluateForwardPass()
    }

    private var overlayLabel: String {
        if selectedOverlay == 0 { return "Top Moves" }
        return "Channel \(selectedOverlay - 1): \(TensorChannelNames.names[selectedOverlay - 1])"
    }

    private var currentOverlay: BoardOverlay {
        // Candidate test mode overrides the game-mode blackout: we want
        // to see the forward-pass arrows/channels on the edit board even
        // though a game is running in the background.
        if !showForwardPassUI { return .none }
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
            Text("Drew's Chess Machine")
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
                    Button("Play and Train") { startRealTraining() }
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

                // Run Arena — visible only during Play and Train. Fires
                // an arena immediately (outside the 30-minute auto
                // cadence) and is disabled while one is already
                // running so the user can't queue overlapping
                // tournaments. Writes to the trigger box; the arena
                // coordinator task picks it up on its next poll.
                //
                // Disabled check is based on `isArenaRunning` (a @State
                // mirror maintained by runArenaParallel) so SwiftUI
                // actually re-evaluates it when state changes. There's
                // a small window (~500 ms, the arena coordinator poll
                // interval) between the user clicking and the button
                // disabling; repeated clicks during that window just
                // re-set the trigger flag, which is idempotent.
                if realTraining {
                    Button("Run Arena") {
                        SessionLogger.shared.log("[BUTTON] Run Arena")
                        guard !isArenaRunning else { return }
                        arenaTriggerBox?.trigger()
                    }
                    .disabled(isArenaRunning)
                }

                if isBusy {
                    ProgressView().controlSize(.small)
                    busyLabelView
                }
            }

            // Board + text side by side
            HStack(alignment: .top, spacing: 24) {
                VStack(spacing: 6) {
                    if realTraining {
                        Picker("Board", selection: playAndTrainBoardBinding) {
                            Text("Game run").tag(PlayAndTrainBoardMode.gameRun)
                            Text("Candidate test").tag(PlayAndTrainBoardMode.candidateTest)
                        }
                        .pickerStyle(.segmented)
                        .labelsHidden()
                        .frame(maxWidth: 240)
                    }

                    if inferenceResult != nil, showForwardPassUI {
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
                        let leftDisabled = inferenceResult == nil || selectedOverlay == 0 || !showForwardPassUI
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

                        let rightDisabled = inferenceResult == nil || selectedOverlay == 18 || !showForwardPassUI
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
                            // Candidate test mode replaces the game-stats
                            // column with the inference-result column so
                            // the user sees what the network thinks of the
                            // probe position alongside the running training
                            // stats. Same min-width as the game column so
                            // the overall text panel doesn't reflow when
                            // toggling between Game run and Candidate test.
                            if isGameMode && !isCandidateTestActive {
                                Text(gameSnapshot.statsText(
                                    continuousPlay: continuousPlay || realTraining
                                ))
                                .frame(minWidth: 330, alignment: .topLeading)
                            }
                            if isCandidateTestActive, let result = inferenceResult {
                                Text(result.textOutput)
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
            if let result = inferenceResult, showForwardPassUI {
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
            // Arena progress mirror — cheap lock read, only updates the
            // @State when the game index has advanced (or transitioned
            // between non-running and running), so no redundant view
            // invalidations between tournament games.
            if let tBox = tournamentBox {
                let snap = tBox.snapshot()
                if snap?.currentGame != tournamentProgress?.currentGame
                    || (snap == nil) != (tournamentProgress == nil) {
                    tournamentProgress = snap
                }
            }
            // Parallel worker counters mirror — only updates @State
            // when totals have actually advanced so the body isn't
            // re-evaluated when nothing's changed. The sessionStart
            // timestamp is embedded in the snapshot so the busy
            // label can compute rates on every render.
            if let pBox = parallelWorkerStatsBox {
                let snap = pBox.snapshot()
                if snap.selfPlayGames != (parallelStats?.selfPlayGames ?? -1)
                    || snap.trainingSteps != (parallelStats?.trainingSteps ?? -1) {
                    parallelStats = snap
                }
            }
        }
    }

    /// Colored status line for the busy row. Returns a `Text` so the
    /// arena path can use multiple foreground colors in one line — the
    /// header and elapsed-time suffix in an "arena running" accent
    /// color, and the running score emphasized in green (at or above
    /// the promotion threshold) or red (below). All non-arena states
    /// fall through to `busyLabel` rendered in the usual secondary
    /// color. The promotion threshold is read from
    /// `Self.tournamentPromoteThreshold` so flipping it in one place
    /// re-colors the UI automatically.
    private var busyLabelView: Text {
        if let tp = tournamentProgress {
            let elapsed = Date().timeIntervalSince(tp.startTime)
            let scorePercent = tp.candidateScore * 100
            let thresholdPercent = Self.tournamentPromoteThreshold * 100
            let scoreColor: Color = scorePercent >= thresholdPercent ? .green : .red

            let head = Text(String(
                format: "Arena game %d/%d  candidate %d-%d-%d  score ",
                tp.currentGame, tp.totalGames,
                tp.candidateWins, tp.championWins, tp.draws
            ))
            .foregroundStyle(Color.blue)

            let score = Text(String(format: "%.2f%%", scorePercent))
                .foregroundStyle(scoreColor)
                .bold()

            let tail = Text("  " + Self.formatElapsed(elapsed))
                .foregroundStyle(Color.blue)

            return head + score + tail
        }
        return Text(busyLabel).foregroundStyle(.secondary)
    }

    private var busyLabel: String {
        if isBuilding { return "Building network..." }
        if realTraining {
            // Parallel mode: self-play and training advance independently,
            // each on its own task. Show both rates in the same units
            // (positions/sec) so you can see the training loop racing
            // ahead of the self-play loop. Rates are computed against
            // the session start time captured in `parallelStats` so they
            // stabilize quickly and reflect actual wall-clock throughput.
            let games = gameSnapshot.totalGames
            let buf = replayBuffer?.count ?? 0
            let steps = trainingStats?.steps ?? 0
            if let ps = parallelStats {
                let elapsed = max(0.1, Date().timeIntervalSince(ps.sessionStart))
                let spMovesPerSec = Double(ps.selfPlayPositions) / elapsed
                let trMovesPerSec = Double(ps.trainingSteps * Self.trainingBatchSize) / elapsed
                return String(
                    format: "Self-play: %d games, %@ moves/s  ·  Train: %d steps, %@ moves/s  ·  Buf: %d",
                    games,
                    Int(spMovesPerSec.rounded()).formatted(.number.grouping(.automatic)),
                    steps,
                    Int(trMovesPerSec.rounded()).formatted(.number.grouping(.automatic)),
                    buf
                )
            }
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
        SessionLogger.shared.log("[BUTTON] Build Network")
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
                net.identifier = ModelIDMinter.mint()
                network = net
                runner = ChessRunner(network: net)
                let idStr = net.identifier?.description ?? "?"
                networkStatus = """
                    Network built in \(String(format: "%.1f", net.buildTimeMs)) ms
                    ID: \(idStr)
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
        SessionLogger.shared.log("[BUTTON] Run Forward Pass")
        gameWatcher.resetAll()
        gameSnapshot = gameWatcher.snapshot()
        clearTrainingDisplay()
        // Explicit "Run Forward Pass" click resets the overlay to Top Moves;
        // auto re-evals from drag edits preserve whatever overlay the user
        // is currently inspecting.
        selectedOverlay = 0
        reevaluateForwardPass()
    }

    /// Cooperative candidate-test probe, called from the Play and Train
    /// driver task at natural gap points (end of a self-play game, end of
    /// a training block). Fires a forward pass on the current editable
    /// state iff Candidate test mode is active AND either the user has
    /// dirtied the probe (drag / side-to-move / Board-picker flip) or the
    /// 15-second interval has elapsed since the last probe.
    ///
    /// Serialization: this runs from the driver task, which is paused on
    /// the `await` for the duration of the inference. No game or training
    /// step can run concurrently with the probe, so there's no contention
    /// on the shared ChessNetwork graph — which is exactly why we chose
    /// the cooperative-gap model over a parallel timer.
    @MainActor
    private func fireCandidateProbeIfNeeded() async {
        // Guards: Candidate test active, candidate runner built, trainer
        // and candidate network both available for the trainer → candidate
        // sync. All of these are normally true during Play and Train —
        // the early-return cases cover "just-started race before the
        // candidate was built" or "trainer wasn't initialized." We also
        // skip if an arena is running because the arena is currently
        // using the candidate inference network as its player-A; probe
        // writes to the same network would race with arena reads.
        guard
            isCandidateTestActive,
            let candidateRunner,
            let trainer,
            let candidateInference = candidateInferenceNetwork,
            arenaActiveFlag?.isActive != true
        else { return }
        let now = Date()
        let dirty = candidateProbeDirty
        let intervalElapsed = now.timeIntervalSince(lastCandidateProbeTime)
            >= Self.candidateProbeIntervalSec
        guard dirty || intervalElapsed else { return }

        let state = editableState
        let result: EvaluationResult
        do {
            // Snapshot the trainer's current state into the candidate
            // inference network, then immediately run the probe. Doing
            // the copy here — rather than after every training block —
            // means the ~11.6 MB trainer → candidate transfer happens
            // only when the probe is actually about to fire (every 15 s
            // or on drag/side-to-move/mode-flip), not at the ~per-second
            // cadence of training blocks. Copy cost is still trivial but
            // the pattern is cleaner: "no one looks at the candidate
            // except the probe, so no one needs to update it except the
            // probe."
            //
            // Both the sync and the forward pass run on a detached task
            // so we don't stall MainActor while they execute. The driver
            // task is awaiting this whole method, so no other user of
            // the trainer or candidate network can run concurrently.
            result = try await Task.detached(priority: .userInitiated) {
                let weights = try trainer.network.exportWeights()
                try candidateInference.network.loadWeights(weights)
                return Self.performInference(with: candidateRunner, state: state)
            }.value
        } catch {
            // Leave probe state unchanged so the previous result stays
            // on screen; the next gap-point call will retry. The error
            // lands in trainingBox via the driver loop's existing
            // plumbing if something structural broke.
            return
        }
        // Probe is a transient read-only snapshot, not a checkpoint —
        // candidateInference inherits the trainer's current ID rather
        // than minting a fresh one. (Arena snapshots, by contrast,
        // do mint — see runArenaParallel.)
        candidateInference.identifier = trainer.identifier
        inferenceResult = result
        candidateProbeDirty = false
        lastCandidateProbeTime = Date()
        candidateProbeCount += 1
    }

    /// Run one arena tournament in parallel mode — 200 games between
    /// the candidate (synced from trainer at start) and the arena
    /// champion (synced from the real champion at start), while
    /// self-play and training continue running in the background.
    /// Promotes the candidate into the real champion iff the score
    /// meets the 0.55 threshold.
    ///
    /// Synchronization: this is called from the arena coordinator
    /// task, which is a peer to the self-play and training workers.
    /// Training and self-play are briefly paused at arena start so
    /// the method can take trainer and champion snapshots into
    /// dedicated arena-only networks; after that the arena runs
    /// exclusively on those two snapshots and doesn't touch the
    /// "live" trainer or champion again until promotion (which
    /// briefly re-pauses self-play to write into the champion).
    /// Candidate test probes skip while `arenaFlag.isActive` so
    /// they don't race with arena reads of the candidate inference
    /// network.
    @MainActor
    private func runArenaParallel(
        trainer: ChessTrainer,
        champion: ChessMPSNetwork,
        candidateInference: ChessMPSNetwork,
        arenaChampion: ChessMPSNetwork,
        tBox: TournamentLiveBox,
        selfPlayGate: WorkerPauseGate,
        trainingGate: WorkerPauseGate,
        arenaFlag: ArenaActiveFlag
    ) async {
        let steps = trainingStats?.steps ?? 0

        let trainerIDStart = trainer.identifier?.description ?? "?"
        let championIDStart = champion.identifier?.description ?? "?"
        SessionLogger.shared.log(
            "[ARENA] start  step=\(steps) trainer=\(trainerIDStart) champion=\(championIDStart)"
        )

        // Mark arena active and seed live progress. Arena-active
        // suppresses the candidate test probe for the duration so
        // probe and arena don't race on the candidate inference
        // network. isArenaRunning is @State mirror the UI reads to
        // disable the Run Arena button and adjust the busy label.
        arenaFlag.set()
        isArenaRunning = true

        let totalGames = Self.tournamentGames
        let startTime = Date()
        tBox.update(TournamentProgress(
            currentGame: 0,
            totalGames: totalGames,
            candidateWins: 0,
            championWins: 0,
            draws: 0,
            startTime: startTime
        ))
        tournamentProgress = tBox.snapshot()

        // --- Trainer → candidate inference snapshot ---
        //
        // Pause the training worker briefly (a few ms, at most one
        // SGD step) so we can export trainer weights without racing
        // against a concurrent `trainer.trainStep`. Release training
        // as soon as the snapshot lands — the rest of the arena
        // runs on `candidateInference`, not on `trainer`, so training
        // can continue through the 200 games.
        await trainingGate.pauseAndWait()
        if Task.isCancelled {
            trainingGate.resume()
            cleanupArenaState(arenaFlag: arenaFlag, tBox: tBox)
            return
        }
        do {
            try await Task.detached(priority: .userInitiated) {
                let weights = try trainer.network.exportWeights()
                try candidateInference.network.loadWeights(weights)
            }.value
        } catch {
            trainingBox?.recordError("Arena candidate sync failed: \(error.localizedDescription)")
            trainingGate.resume()
            cleanupArenaState(arenaFlag: arenaFlag, tBox: tBox)
            return
        }
        trainingGate.resume()

        // Mint a fresh ID for the arena candidate — this snapshot
        // represents the specific checkpoint being evaluated in this
        // tournament. The trainer keeps its own ID (the current training
        // lineage) and continues SGD in the background.
        candidateInference.identifier = ModelIDMinter.mint()

        // --- Champion → arena champion snapshot ---
        //
        // Same pattern for self-play: brief pause, copy weights from
        // the real champion into the arena-only champion network,
        // release. Arena games from here on only read
        // `arenaChampion`, leaving the real champion free for
        // continuous self-play through the tournament.
        await selfPlayGate.pauseAndWait()
        if Task.isCancelled {
            selfPlayGate.resume()
            cleanupArenaState(arenaFlag: arenaFlag, tBox: tBox)
            return
        }
        do {
            try await Task.detached(priority: .userInitiated) {
                let weights = try champion.network.exportWeights()
                try arenaChampion.network.loadWeights(weights)
            }.value
        } catch {
            trainingBox?.recordError("Arena champion sync failed: \(error.localizedDescription)")
            selfPlayGate.resume()
            cleanupArenaState(arenaFlag: arenaFlag, tBox: tBox)
            return
        }
        selfPlayGate.resume()

        // arenaChampion is a pure copy of champion's current weights,
        // so it inherits the champion's ID verbatim — no mint.
        arenaChampion.identifier = champion.identifier

        // --- Tournament games ---
        //
        // Cancellation: the detached tournament task is wrapped in
        // `withTaskCancellationHandler` so clicking Stop flips a
        // `CancelBox` that `TournamentDriver.run` checks between
        // games. The worst-case delay from Stop to actually breaking
        // out is one in-flight arena game (~400 ms).
        let cancelBox = CancelBox()
        let stats = await withTaskCancellationHandler {
            await Task.detached(priority: .userInitiated) {
                [arenaChampion, candidateInference, tBox, cancelBox] in
                let driver = TournamentDriver()
                driver.delegate = nil
                return await driver.run(
                    playerA: { MPSChessPlayer(name: "Candidate", network: candidateInference, schedule: .arena) },
                    playerB: { MPSChessPlayer(name: "Champion", network: arenaChampion, schedule: .arena) },
                    games: totalGames,
                    isCancelled: { cancelBox.isCancelled },
                    onGameCompleted: { gameIndex, aWins, bWins, draws in
                        tBox.update(TournamentProgress(
                            currentGame: gameIndex,
                            totalGames: totalGames,
                            candidateWins: aWins,
                            championWins: bWins,
                            draws: draws,
                            startTime: startTime
                        ))
                    }
                )
            }.value
        } onCancel: {
            cancelBox.cancel()
        }

        // --- Score and promotion ---
        let playedGames = stats.gamesPlayed
        let score: Double
        if playedGames > 0 {
            score = (Double(stats.playerAWins) + 0.5 * Double(stats.draws)) / Double(playedGames)
        } else {
            score = 0
        }
        var promoted = false
        if playedGames >= totalGames && score >= Self.tournamentPromoteThreshold {
            // Pause self-play briefly, copy candidate inference into
            // the real champion, release. Self-play's next game uses
            // the new weights.
            await selfPlayGate.pauseAndWait()
            if !Task.isCancelled {
                do {
                    try await Task.detached(priority: .userInitiated) {
                        [candidateInference, champion] in
                        let weights = try candidateInference.network.exportWeights()
                        try champion.network.loadWeights(weights)
                    }.value
                    // Promoted: champion now holds the arena candidate's
                    // exact weights, so it inherits that snapshot ID.
                    champion.identifier = candidateInference.identifier
                    promoted = true
                } catch {
                    trainingBox?.recordError("Promotion copy failed: \(error.localizedDescription)")
                }
            }
            selfPlayGate.resume()
        }

        // Append to history and clear arena state.
        let durationSec = Date().timeIntervalSince(startTime)
        let record = TournamentRecord(
            finishedAtStep: steps,
            candidateWins: stats.playerAWins,
            championWins: stats.playerBWins,
            draws: stats.draws,
            score: score,
            promoted: promoted,
            durationSec: durationSec
        )
        tournamentHistory.append(record)
        logArenaResult(
            record: record,
            index: tournamentHistory.count,
            trainer: trainer,
            candidate: candidateInference,
            championSide: arenaChampion
        )
        cleanupArenaState(arenaFlag: arenaFlag, tBox: tBox)
    }

    /// Emit a three-line summary of a just-finished arena tournament to
    /// stdout (visible in Xcode's console). First line has the human-
    /// readable result, second line has the training + sampling
    /// parameters, third line has the model IDs of both arena sides
    /// plus the trainer's current lineage ID. Running a long
    /// Play-and-Train session leaves behind a greppable audit trail
    /// of `[ARENA] …` lines correlating each result with its
    /// configuration and the exact checkpoint it evaluated. Kept
    /// deliberately on stdout rather than the UI
    /// `TrainingLiveStatsBox` so external tooling can `tee` the log.
    @MainActor
    private func logArenaResult(
        record: TournamentRecord,
        index: Int,
        trainer: ChessTrainer,
        candidate: ChessMPSNetwork,
        championSide: ChessMPSNetwork
    ) {
        let durMin = Int(record.durationSec) / 60
        let durSec = Int(record.durationSec) % 60
        let durationStr = String(format: "%d:%02d", durMin, durSec)
        let statusStr = record.promoted ? "PROMOTED" : "kept"
        let sp = SamplingSchedule.selfPlay
        let ar = SamplingSchedule.arena
        let lrStr = String(format: "%.1e", trainer.learningRate)
        let scoreStr = String(format: "%.3f", record.score)
        let threshStr = String(format: "%.2f", Self.tournamentPromoteThreshold)
        let spTauStr = String(
            format: "%.2f/%.2f@%d",
            Double(sp.openingTau),
            Double(sp.mainTau),
            sp.openingPliesPerPlayer
        )
        let arTauStr = String(
            format: "%.2f/%.2f@%d",
            Double(ar.openingTau),
            Double(ar.mainTau),
            ar.openingPliesPerPlayer
        )
        let candidateIDStr = candidate.identifier?.description ?? "?"
        let championIDStr = championSide.identifier?.description ?? "?"
        let trainerIDStr = trainer.identifier?.description ?? "?"

        let wld = "\(record.candidateWins)-\(record.championWins)-\(record.draws)"
        let header = "[ARENA] #\(index) @ step \(record.finishedAtStep)  W/L/D=\(wld)  score=\(scoreStr)  \(statusStr)  dur=\(durationStr)"
        let params = "        batch=\(Self.trainingBatchSize) lr=\(lrStr) promote>=\(threshStr) games=\(Self.tournamentGames) sp.tau=\(spTauStr) ar.tau=\(arTauStr)"
        let ids = "        candidate=\(candidateIDStr)  champion=\(championIDStr)  trainer=\(trainerIDStr)"

        print(header)
        print(params)
        print(ids)

        // Mirror the same three lines into the session log so the
        // on-disk file carries the full arena history even when the
        // Xcode console isn't being captured.
        SessionLogger.shared.log(header)
        SessionLogger.shared.log(params)
        SessionLogger.shared.log(ids)
    }

    /// Release arena-active state on an early return from
    /// `runArenaParallel`. Clears the active flag (so the candidate
    /// probe resumes), clears the live-progress box and mirror (so
    /// the busy label reverts to normal Play and Train mode), and
    /// resets the UI's `isArenaRunning` mirror.
    @MainActor
    private func cleanupArenaState(arenaFlag: ArenaActiveFlag, tBox: TournamentLiveBox) {
        arenaFlag.clear()
        isArenaRunning = false
        tBox.clear()
        tournamentProgress = nil
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
                // Pure forward-pass mode runs through the champion via
                // `runner`. Candidate test mode takes a different path
                // via `fireCandidateProbeIfNeeded`, which uses
                // `candidateRunner` → the dedicated candidate inference
                // network.
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
        requestForwardPassReeval()
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
        SessionLogger.shared.log("[BUTTON] Play Game")
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
        SessionLogger.shared.log("[BUTTON] Play Continuous")
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
        SessionLogger.shared.log("[BUTTON] Stop")
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
        SessionLogger.shared.log("[BUTTON] Train Once")
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
        SessionLogger.shared.log("[BUTTON] Train Continuous")
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

    /// Kick off real-data training in parallel mode: self-play, training,
    /// and arena coordination run as three independent tasks inside a
    /// `TaskGroup`, sharing state only through the lock-protected
    /// replay buffer, stats boxes, pause gates, and arena-trigger box.
    /// Self-play plays one game at a time on the champion network and
    /// streams labeled positions into the replay buffer. Training runs
    /// a tight-loop SGD on the trainer network, sampling the buffer
    /// for each batch. The arena coordinator sleeps until triggered
    /// (either by the 30-minute auto-fire or the Run Arena button),
    /// then runs 200 games between the candidate inference network
    /// and a fourth "arena champion" network — both snapshots taken
    /// under brief per-worker pauses so game play and training never
    /// actually stop, even during a tournament.
    private func startRealTraining() {
        SessionLogger.shared.log("[BUTTON] Play and Train")
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
        playAndTrainBoardMode = .gameRun
        candidateProbeDirty = false
        lastCandidateProbeTime = .distantPast
        candidateProbeCount = 0
        tournamentHistory = []
        tournamentProgress = nil
        let tBox = TournamentLiveBox()
        tournamentBox = tBox
        let pStatsBox = ParallelWorkerStatsBox(sessionStart: Date())
        parallelWorkerStatsBox = pStatsBox
        parallelStats = pStatsBox.snapshot()
        let selfPlayGate = WorkerPauseGate()
        let trainingGate = WorkerPauseGate()
        let arenaFlag = ArenaActiveFlag()
        arenaActiveFlag = arenaFlag
        let triggerBox = ArenaTriggerBox()
        arenaTriggerBox = triggerBox
        isArenaRunning = false
        realTraining = true

        realTrainingTask = Task(priority: .userInitiated) {
            [trainer, network, buffer, box, tBox, pStatsBox,
             selfPlayGate, trainingGate, arenaFlag, triggerBox] in

            // --- Setup: build any missing networks, reset the trainer ---

            let needsCandidateBuild = await MainActor.run { candidateInferenceNetwork == nil }
            if needsCandidateBuild {
                do {
                    let built = try await Task.detached(priority: .userInitiated) {
                        try ChessMPSNetwork(.randomWeights)
                    }.value
                    await MainActor.run {
                        candidateInferenceNetwork = built
                        candidateRunner = ChessRunner(network: built)
                    }
                } catch {
                    box.recordError("Candidate network init failed: \(error.localizedDescription)")
                    await MainActor.run {
                        realTraining = false
                        realTrainingTask = nil
                    }
                    return
                }
            }

            let needsArenaChampionBuild = await MainActor.run { arenaChampionNetwork == nil }
            if needsArenaChampionBuild {
                do {
                    let built = try await Task.detached(priority: .userInitiated) {
                        try ChessMPSNetwork(.randomWeights)
                    }.value
                    await MainActor.run {
                        arenaChampionNetwork = built
                    }
                } catch {
                    box.recordError("Arena champion init failed: \(error.localizedDescription)")
                    await MainActor.run {
                        realTraining = false
                        realTrainingTask = nil
                    }
                    return
                }
            }

            // Reset the trainer's graph AND initialize its weights from
            // the champion. Pre-ID this was `resetNetwork()` alone, which
            // re-randomized the trainer independently of the champion —
            // meaning step 0 of every Play-and-Train session pitted two
            // different random inits against each other in the arena.
            // Copying champion weights makes arena-at-step-0 a fair
            // tie by construction and establishes the trainer's
            // starting point as a true fork of the champion.
            do {
                try await Task.detached(priority: .userInitiated) {
                    try trainer.resetNetwork()
                    let championWeights = try network.network.exportWeights()
                    try trainer.network.loadWeights(championWeights)
                }.value
            } catch {
                box.recordError("Reset failed: \(error.localizedDescription)")
                await MainActor.run {
                    realTraining = false
                    realTrainingTask = nil
                }
                return
            }

            // Mint a fresh ID for the trainer now that its weights are
            // loaded from the champion. The trainer's ID represents the
            // "current training lineage" — stable from here until the
            // next Play-and-Train restart; individual arena snapshots
            // get their own minted IDs inside `runArenaParallel`.
            await MainActor.run {
                trainer.identifier = ModelIDMinter.mint()
            }

            // Grab the candidate inference network and arena champion
            // network references on the main actor once — both are
            // now guaranteed non-nil from the setup above. The
            // workers capture them as values for the duration of the
            // session.
            let candidateInference = await MainActor.run { candidateInferenceNetwork }
            let arenaChampion = await MainActor.run { arenaChampionNetwork }
            guard let candidateInference, let arenaChampion else {
                box.recordError("Networks missing after setup")
                await MainActor.run {
                    realTraining = false
                    realTrainingTask = nil
                }
                return
            }

            // --- Spawn the three worker tasks ---

            await withTaskGroup(of: Void.self) { group in
                // Self-play worker: plays one game at a time on the
                // champion, streams positions into the replay buffer.
                // Checks `selfPlayGate` between games so the arena
                // coordinator can briefly pause for the champion →
                // arena-champion snapshot and for promotion.
                //
                // `gameWatcher` is captured once here — GameWatcher is
                // @unchecked Sendable so we can call its methods from
                // any context, avoiding a MainActor hop per game to
                // re-read the @State reference.
                group.addTask(priority: .userInitiated) {
                    [network, buffer, pStatsBox, selfPlayGate, gameWatcher] in
                    while !Task.isCancelled {
                        // Pause gate check (between games).
                        if selfPlayGate.isRequestedToPause {
                            selfPlayGate.markWaiting()
                            while selfPlayGate.isRequestedToPause && !Task.isCancelled {
                                try? await Task.sleep(for: .milliseconds(5))
                            }
                            selfPlayGate.markRunning()
                        }
                        if Task.isCancelled { break }

                        gameWatcher.resetCurrentGame()
                        gameWatcher.markPlaying(true)

                        let machine = ChessMachine()
                        machine.delegate = gameWatcher
                        let white = MPSChessPlayer(
                            name: "White",
                            network: network,
                            replayBuffer: buffer,
                            schedule: .selfPlay
                        )
                        let black = MPSChessPlayer(
                            name: "Black",
                            network: network,
                            replayBuffer: buffer,
                            schedule: .selfPlay
                        )

                        do {
                            let task = try machine.beginNewGame(white: white, black: black)
                            _ = await task.value
                        } catch {
                            gameWatcher.markPlaying(false)
                            break
                        }

                        // Record the completed game for the parallel
                        // rate display. Position count is the total
                        // plies across both players in this game.
                        let positions = white.recordedPliesCount + black.recordedPliesCount
                        pStatsBox.recordSelfPlayGame(positions: positions)
                    }
                }

                // Training worker: tight-loop SGD on the trainer,
                // sampling batches from the replay buffer. Fires the
                // candidate probe at its own 15 s cadence between
                // steps, and nudges the arena trigger box when the
                // 30 min auto cadence elapses. Pauses at `trainingGate`
                // so the arena coordinator can briefly snapshot
                // trainer weights.
                group.addTask(priority: .userInitiated) {
                    [trainer, buffer, box, pStatsBox, trainingGate, triggerBox] in
                    while !Task.isCancelled {
                        // Pause gate check (between steps).
                        if trainingGate.isRequestedToPause {
                            trainingGate.markWaiting()
                            while trainingGate.isRequestedToPause && !Task.isCancelled {
                                try? await Task.sleep(for: .milliseconds(5))
                            }
                            trainingGate.markRunning()
                        }
                        if Task.isCancelled { break }

                        // Wait for the replay buffer to warm up before
                        // starting to train — the first few games
                        // haven't produced enough decorrelated samples
                        // yet. Short sleep + retry keeps the worker
                        // responsive to Stop and to pause requests.
                        if buffer.count < Self.minBufferBeforeTraining {
                            try? await Task.sleep(for: .milliseconds(100))
                            continue
                        }

                        guard let batch = buffer.sample(count: Self.trainingBatchSize) else {
                            try? await Task.sleep(for: .milliseconds(100))
                            continue
                        }

                        // The enclosing worker already runs at
                        // `.userInitiated` so there's nothing to escape
                        // to — run the SGD step inline and skip the
                        // per-step detached-task + continuation
                        // allocation pair.
                        let result: Result<TrainStepTiming, Error>
                        do {
                            result = .success(try trainer.trainStep(batch: batch))
                        } catch {
                            result = .failure(error)
                        }

                        switch result {
                        case .success(let timing):
                            box.recordStep(timing)
                            pStatsBox.recordTrainingStep()
                        case .failure(let error):
                            box.recordError(error.localizedDescription)
                            return
                        }

                        // Candidate-test probe firing check. Method
                        // guards internally on all preconditions
                        // including arena-active, so an unconditional
                        // call here is safe — it no-ops when nothing
                        // is due or when the arena is running.
                        await fireCandidateProbeIfNeeded()

                        // Auto-trigger the arena on the 30-min cadence.
                        // Fires the trigger inbox; the arena coordinator
                        // task picks it up and runs the tournament.
                        if triggerBox.shouldAutoTrigger(interval: Self.secondsPerTournament) {
                            triggerBox.trigger()
                        }
                    }
                }

                // Arena coordinator: polls the trigger inbox and runs
                // a tournament whenever one is pending. Blocks its
                // own loop (not the worker tasks) during arena
                // execution. Both the 30-minute auto-fire and the
                // Run Arena button enter here via `triggerBox.trigger()`.
                group.addTask(priority: .userInitiated) {
                    [trainer, network, tBox, selfPlayGate, trainingGate, arenaFlag, triggerBox,
                     candidateInference, arenaChampion] in
                    while !Task.isCancelled {
                        if triggerBox.consume() {
                            await self.runArenaParallel(
                                trainer: trainer,
                                champion: network,
                                candidateInference: candidateInference,
                                arenaChampion: arenaChampion,
                                tBox: tBox,
                                selfPlayGate: selfPlayGate,
                                trainingGate: trainingGate,
                                arenaFlag: arenaFlag
                            )
                            triggerBox.recordArenaCompleted()
                        } else {
                            try? await Task.sleep(for: .milliseconds(500))
                        }
                    }
                }

                // Periodic session-log ticker. Fires at 30s, 1m, 2m,
                // 5m, 15m, 30m, 60m from Play-and-Train start and
                // then once per hour for the rest of the session.
                // Each wake-up snapshots the thread-safe stats boxes
                // and writes one `[STATS]` line to the on-disk
                // session log. Identifiers are pulled through a brief
                // MainActor hop since they live on classes whose var
                // mutation is otherwise main-actor-driven.
                group.addTask(priority: .utility) {
                    [trainer, network, box, pStatsBox, buffer] in
                    let sessionStart = Date()
                    let fixedTargets: [TimeInterval] = [30, 60, 120, 300, 900, 1800, 3600]

                    func logOne(elapsedTarget: TimeInterval) async {
                        let trainingSnap = box.snapshot()
                        let parallelSnap = pStatsBox.snapshot()
                        let bufCount = buffer.count
                        let bufCap = buffer.capacity
                        let (trainerID, championID) = await MainActor.run {
                            (
                                trainer.identifier?.description ?? "?",
                                network.identifier?.description ?? "?"
                            )
                        }
                        let policyStr: String
                        if let p = trainingSnap.rollingPolicyLoss {
                            policyStr = String(format: "%+.4f", p)
                        } else {
                            policyStr = "--"
                        }
                        let valueStr: String
                        if let v = trainingSnap.rollingValueLoss {
                            valueStr = String(format: "%+.4f", v)
                        } else {
                            valueStr = "--"
                        }
                        let entropyStr: String
                        if let e = trainingSnap.rollingPolicyEntropy {
                            entropyStr = String(format: "%.4f", e)
                        } else {
                            entropyStr = "--"
                        }
                        let h = Int(elapsedTarget) / 3600
                        let m = (Int(elapsedTarget) % 3600) / 60
                        let s = Int(elapsedTarget) % 60
                        let elapsedStr = String(format: "%d:%02d:%02d", h, m, s)
                        let line = "[STATS] elapsed=\(elapsedStr) steps=\(trainingSnap.stats.steps) spGames=\(parallelSnap.selfPlayGames) spMoves=\(parallelSnap.selfPlayPositions) buffer=\(bufCount)/\(bufCap) pLoss=\(policyStr) vLoss=\(valueStr) pEnt=\(entropyStr) trainer=\(trainerID) champion=\(championID)"
                        SessionLogger.shared.log(line)
                    }

                    func sleepUntil(elapsed: TimeInterval) async -> Bool {
                        let remaining = elapsed - Date().timeIntervalSince(sessionStart)
                        if remaining > 0 {
                            do {
                                try await Task.sleep(for: .seconds(remaining))
                            } catch {
                                return false
                            }
                        }
                        return !Task.isCancelled
                    }

                    // Fixed schedule.
                    for target in fixedTargets {
                        if Task.isCancelled { return }
                        guard await sleepUntil(elapsed: target) else { return }
                        await logOne(elapsedTarget: target)
                    }

                    // Hourly after the fixed schedule (2h, 3h, 4h, ...).
                    var hourNumber = 2
                    while !Task.isCancelled {
                        let target = TimeInterval(hourNumber) * 3600
                        guard await sleepUntil(elapsed: target) else { return }
                        await logOne(elapsedTarget: target)
                        hourNumber += 1
                    }
                }

                // Wait for all four tasks to complete (only happens
                // on cancellation since each loops forever).
                for await _ in group { }
            }

            await MainActor.run {
                realTraining = false
                realTrainingTask = nil
                isArenaRunning = false
                arenaActiveFlag = nil
                arenaTriggerBox = nil
                parallelWorkerStatsBox = nil
                parallelStats = nil
            }
        }
    }

    private func stopRealTraining() {
        realTrainingTask?.cancel()
        realTrainingTask = nil
    }

    // MARK: - Sweep Actions

    private func startSweep() {
        SessionLogger.shared.log("[BUTTON] Sweep Batch Sizes")
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
        let trainerIDStr = trainer?.identifier?.description ?? dash
        let championIDStr = network?.identifier?.description ?? dash
        lines.append("  Trainer ID:  \(trainerIDStr)")
        lines.append("  Champion ID: \(championIDStr)")
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
            // Rolling total derived from the two component windows —
            // since the mean operator is linear, mean(policy + value)
            // equals mean(policy) + mean(value), so no third window is
            // needed on the TrainingLiveStatsBox side. Only display it
            // when both components have at least one sample; otherwise
            // the components disagree on sample count and the derived
            // sum would be misleading.
            let totalStr: String
            if let p = realRollingPolicyLoss, let v = realRollingValueLoss {
                totalStr = String(format: "%+.6f", p + v)
            } else {
                totalStr = dash
            }
            lines.append("  Loss total:  \(totalStr)")
            lines.append("    Loss policy:   \(policyStr)")
            lines.append("    Loss value:    \(valueStr)")
            // Candidate-test probe counter + time-since-last, so the user
            // can distinguish "probes firing but imperceptible" from "probes
            // stuck". Shown in both Game run and Candidate test modes so
            // the count is visible while Play and Train is running; the
            // count only advances when Candidate test is active and a
            // gap check actually fires a probe.
            let probeLine: String
            if candidateProbeCount == 0 {
                probeLine = dash
            } else {
                let ageSec = Date().timeIntervalSince(lastCandidateProbeTime)
                probeLine = String(format: "%4d  (last %5.1f s ago)", candidateProbeCount, ageSec)
            }
            lines.append("  Probes:      \(probeLine)")
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
            lines.append(String(format: "    entropy:   %.6f", last.policyEntropy))
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
            let movesSecStr: String
            let movesHrStr: String
            if stats.steps > 0 {
                let movesPerSec = stats.positionsPerSecond(batchSize: Self.trainingBatchSize)
                movesSecStr = Int(movesPerSec.rounded())
                    .formatted(.number.grouping(.automatic))
                movesHrStr = Int((movesPerSec * 3600).rounded())
                    .formatted(.number.grouping(.automatic))
            } else {
                movesSecStr = dash
                movesHrStr = dash
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
            // Moves/sec and moves/hr match the game side's session-
            // stats format (which shows Games/hr and Moves/hr) so the
            // two tables speak the same language. "Moves" here means
            // the same thing as "positions consumed": each training
            // sample is one position, which is equivalent to one
            // move played in the original game.
            lines.append("  Moves/sec:   \(movesSecStr)")
            lines.append("  Moves/hr:    \(movesHrStr)")
            lines.append("  Proj 250×:   \(projStr)")
        }

        // Arena history — present only in self-play runs. One line per
        // completed tournament, newest last. Promotion marker is
        // visually distinct so you can scan for it.
        if isSelfPlay, !tournamentHistory.isEmpty {
            lines.append("")
            lines.append("Arena History")
            for (idx, record) in tournamentHistory.enumerated() {
                let number = String(format: "%2d", idx + 1)
                let stepStr = record.finishedAtStep.formatted(.number.grouping(.automatic))
                let scoreStr = String(format: "%.3f", record.score)
                let marker = record.promoted ? "PROMOTED" : "kept"
                let durStr = Self.formatElapsed(record.durationSec)
                lines.append(String(
                    format: "  #%@ @ %@ steps  %d-%d-%d  score %@  %@  (%@)",
                    number, stepStr,
                    record.candidateWins, record.championWins, record.draws,
                    scoreStr, marker, durStr
                ))
            }
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

    /// Render an elapsed-time interval as a compact fixed-width string.
    /// Under one minute: `"12.3s"` (1-decimal seconds). One minute and
    /// up: `"1:22"` (m:ss). Keeps the arena busy label stable in width
    /// whether the tournament has been running for 8 seconds or 2
    /// minutes.
    private static func formatElapsed(_ seconds: Double) -> String {
        let s = max(0, seconds)
        if s < 60 {
            return String(format: "%4.1fs", s)
        }
        let totalSec = Int(s)
        return String(format: "%d:%02d", totalSec / 60, totalSec % 60)
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
            lines.append(String(format: "  %.3f%% win / %.3f%% loss",
                                (inference.value + 1) / 2 * 100, (1 - inference.value) / 2 * 100))
            lines.append("")
            lines.append("Policy Head (Top 4)")
            for (rank, move) in inference.topMoves.enumerated() {
                let fromName = BoardEncoder.squareName(move.fromRow * 8 + move.fromCol)
                let toName = BoardEncoder.squareName(move.toRow * 8 + move.toCol)
                let rankCol = String(rank + 1).padding(toLength: 4, withPad: " ", startingAt: 0)
                let moveCol = "\(fromName)-\(toName)".padding(toLength: 8, withPad: " ", startingAt: 0)
                lines.append("  \(rankCol)\(moveCol)\(String(format: "%.6f%%", move.probability * 100))")
            }
            // Sum of the top-100 move probabilities. With a freshly-
            // initialized network this sits near 100/4096 ≈ 2.44%; as the
            // policy head learns to concentrate mass on promising moves,
            // this number climbs — a cheap scalar that changes visibly
            // between candidate-test probes even when the top-4 move
            // ordering stays stable.
            let top100Sum = inference.policy.sorted(by: >).prefix(100).reduce(0, +)
            lines.append(String(format: "  Top 100 sum: %.6f%%", top100Sum * 100))
            lines.append("")
            lines.append("Policy Stats")
            lines.append(String(format: "  Sum: %.8f", inference.policy.reduce(0, +)))
            let nonZeroCount = inference.policy.filter { $0 > 1e-10 }.count
            lines.append(String(format: "  Non-negligible: %d / 4096", nonZeroCount))
            if let maxProb = inference.policy.max(), let minProb = inference.policy.min() {
                lines.append(String(format: "  Min: %.8f", minProb))
                lines.append(String(format: "  Max: %.8f", maxProb))
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
