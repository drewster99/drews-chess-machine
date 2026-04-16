import Charts
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
/// evaluation of it drift as training progresses in the background;
/// `.progressRate` replaces the board with a line chart of rolling
/// moves/hr for self-play, training, and combined, sampled once per
/// second across the life of the session.
enum PlayAndTrainBoardMode: Sendable, Hashable {
    case gameRun
    case candidateTest
    case progressRate
}

/// One point on the Progress rate chart. Sampled once per second
/// from the heartbeat during a Play and Train session; cleared at
/// each new session. The `*MovesPerHour` fields are *rolling-window*
/// rates — over the last 3 minutes leading up to `timestamp`, not
/// lifetime averages — so the chart shows how throughput changes
/// over time rather than asymptoting to the session mean.
struct ProgressRateSample: Identifiable, Sendable {
    /// Monotonic identity for SwiftUI's `ForEach` / `Chart` — the
    /// index the sample was appended at. Stable for the life of
    /// the session and never reused.
    let id: Int
    /// Wall-clock instant this sample was taken. Used as the
    /// reference point when locating the 3-minute-ago sample that
    /// defines the lower edge of this point's rolling window.
    let timestamp: Date
    /// Seconds elapsed since `sessionStart`. Used as the X
    /// coordinate on the chart so each session starts fresh at 0
    /// rather than showing wall-clock time.
    let elapsedSec: Double
    /// Cumulative self-play moves at `timestamp`. Stored per
    /// sample so the next tick can subtract from "the sample that
    /// was 3 minutes ago" to get a windowed delta without needing
    /// a parallel cumulative-counters buffer.
    let selfPlayCumulativeMoves: Int
    /// Cumulative training-positions at `timestamp` — training
    /// steps × batch size. Same reason to store per-sample as
    /// `selfPlayCumulativeMoves`.
    let trainingCumulativeMoves: Int
    /// Rolling self-play moves/hr over the last 3 minutes. Before
    /// the session has 3 minutes of data, the window covers
    /// "everything so far" and this degrades gracefully to the
    /// lifetime rate; after 3 minutes it's strictly a 3-minute
    /// trailing average.
    let selfPlayMovesPerHour: Double
    /// Rolling training moves/hr over the same window.
    let trainingMovesPerHour: Double
    /// Sum of `selfPlayMovesPerHour` and `trainingMovesPerHour`.
    /// Derived rather than stored so changes to the definition of
    /// "combined" only have to happen in one place.
    var combinedMovesPerHour: Double {
        selfPlayMovesPerHour + trainingMovesPerHour
    }
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
    /// ID of the candidate network when a promotion happened — the
    /// model the champion was replaced with. `nil` when `promoted`
    /// is false, so the arena history / logs can surface "which
    /// network just became the champion" alongside the kept/PROMOTED
    /// marker. Captured at the moment of promotion, before any
    /// subsequent trainer re-mint can change the candidate's ID.
    let promotedID: ModelID?
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

    /// Bounded variant of `pauseAndWait` — returns `true` if the
    /// worker entered its wait state within `timeoutMs`, `false`
    /// on timeout (or task cancellation). Used by code paths that
    /// must not deadlock if the worker has exited its loop without
    /// acknowledging the pause (e.g. a Play-and-Train session
    /// ending mid-save via `realTrainingTask.cancel()` — that
    /// cancellation does not propagate to unstructured save
    /// Tasks, so they need their own escape hatch). On timeout
    /// the request flag is cleared so a later-returning worker
    /// doesn't get stuck in a stale pause request.
    func pauseAndWait(timeoutMs: Int) async -> Bool {
        setRequested(true)
        let deadline = Date().addingTimeInterval(Double(timeoutMs) / 1000.0)
        while !Task.isCancelled {
            if readIsWaiting() { return true }
            if Date() >= deadline {
                setRequested(false)
                return false
            }
            try? await Task.sleep(for: .milliseconds(5))
        }
        setRequested(false)
        return false
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

/// Lock-protected current-N holder shared between the SwiftUI
/// Stepper (which mutates the value on the main actor) and the
/// concurrent self-play worker tasks (which poll it between games).
/// Workers above the current count idle in their pause loop until
/// either the count grows enough to include them or the session
/// stops. Decoupling the box from `@State selfPlayWorkerCount` is
/// what lets the value cross the actor boundary without forcing
/// every worker to hop back to the main actor on each game.
final class WorkerCountBox: @unchecked Sendable {
    private let lock = NSLock()
    private var _count: Int

    init(initial: Int) {
        precondition(initial >= 1, "WorkerCountBox initial count must be >= 1")
        _count = initial
    }

    var count: Int {
        lock.lock()
        defer { lock.unlock() }
        return _count
    }

    /// Set the active worker count. Clamped at the bottom to 1 so a
    /// stuck Stepper or a sloppy caller can never zero out self-play
    /// (the upper bound is enforced by the Stepper and the spawn
    /// loop's `absoluteMaxSelfPlayWorkers` constant, not here).
    func set(_ value: Int) {
        lock.lock()
        defer { lock.unlock() }
        _count = max(1, value)
    }
}

/// Lock-protected holder for the live training-step delay in
/// milliseconds. The training worker reads this at the bottom of
/// every step to decide how long to pause before looping; the
/// Stepper in the Play and Train row writes through it whenever
/// the user nudges the value. Decoupled from `@State
/// trainingStepDelayMs` so the worker task doesn't have to hop back
/// to the main actor to read a single Int per step. Clamped at the
/// bottom to 0 — negative delays are meaningless.
final class TrainingStepDelayBox: @unchecked Sendable {
    private let lock = NSLock()
    private var _ms: Int

    init(initial: Int) {
        _ms = max(0, initial)
    }

    var milliseconds: Int {
        lock.lock()
        defer { lock.unlock() }
        return _ms
    }

    func set(_ value: Int) {
        lock.lock()
        defer { lock.unlock() }
        _ms = max(0, value)
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
    /// the previous last-arena time. Also clears the pending flag:
    /// the training worker runs in parallel with the arena and can
    /// stamp `_pending` mid-arena once elapsed time crosses `interval`
    /// against the stale `_lastArenaTime`. Without clearing it here,
    /// that stale trigger would fire a back-to-back arena the instant
    /// the coordinator loops back.
    func recordArenaCompleted() {
        lock.lock()
        defer { lock.unlock() }
        _lastArenaTime = Date()
        _pending = false
    }

    /// True if a trigger is currently pending (used for UI
    /// disable-while-queued semantics).
    var isPending: Bool {
        lock.lock()
        defer { lock.unlock() }
        return _pending
    }
}

/// Lock-protected user-override inbox for an in-flight arena
/// tournament. Exactly two user actions can end a tournament early:
/// `abort()` ends it with no promotion regardless of the score, and
/// `promote()` ends it early and forces promotion regardless of the
/// score. The decision is set-once — whichever of the two buttons
/// lands first wins, and the second is a no-op — so rapid conflicting
/// clicks can't produce contradictory state. `runArenaParallel`
/// clears the box at the start of every tournament and consumes
/// the decision once the driver returns.
final class ArenaOverrideBox: @unchecked Sendable {
    enum Decision: Sendable {
        case abort
        case promote
    }

    private let lock = NSLock()
    private var _decision: Decision?

    /// Request abort: end the current tournament early with no
    /// promotion. No-op if a decision (abort or promote) is already
    /// set for this tournament.
    func abort() {
        lock.lock()
        defer { lock.unlock() }
        if _decision == nil {
            _decision = .abort
        }
    }

    /// Request forced promotion: end the current tournament early
    /// and promote the candidate unconditionally. No-op if a decision
    /// (abort or promote) is already set for this tournament.
    func promote() {
        lock.lock()
        defer { lock.unlock() }
        if _decision == nil {
            _decision = .promote
        }
    }

    /// True once either `abort()` or `promote()` has been called,
    /// until `consume()` resets the box. Polled by the tournament
    /// driver's `isCancelled` closure so the game loop breaks out
    /// between games the moment the user clicks one of the buttons.
    var isActive: Bool {
        lock.lock()
        defer { lock.unlock() }
        return _decision != nil
    }

    /// Read-and-clear the decision. Returns `nil` if no override
    /// was set (normal tournament completion), or the decision the
    /// user made. Called once by `runArenaParallel` after the
    /// driver returns, both to branch on the decision and to reset
    /// the box for the next tournament.
    func consume() -> Decision? {
        lock.lock()
        defer { lock.unlock() }
        let d = _decision
        _decision = nil
        return d
    }
}

/// One refresh of the memory stats line shown in the top busy row
/// during Play and Train. Sampled every ~10 s by the snapshot
/// timer (not every frame) so the displayed numbers stay stable.
/// Bytes are stored at the granularity the source APIs return so
/// the formatter can round consistently regardless of when it ran.
struct MemoryStatsSnapshot: Sendable {
    /// Process resident memory from `task_info(TASK_VM_INFO).phys_footprint`.
    let appFootprintBytes: UInt64
    /// `MTLDevice.currentAllocatedSize` — the live GPU working set.
    let gpuAllocatedBytes: UInt64
    /// `MTLDevice.recommendedMaxWorkingSetSize` — the soft cap Metal
    /// asks us to stay under for this device.
    let gpuMaxTargetBytes: UInt64
    /// Total physical memory available to the process, from
    /// `ProcessInfo.processInfo.physicalMemory`. On Apple Silicon
    /// this is the unified-memory total (CPU and GPU draw from the
    /// same pool), so it doubles as "GPU total RAM" for the user-
    /// facing display.
    let gpuTotalBytes: UInt64
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
    /// Rolling-window length in seconds for the "recent" rate column
    /// shown next to the lifetime rates in the Session panel. Fixed at
    /// 10 minutes so short-term throughput shifts (e.g. after an arena
    /// pause, or when training/self-play contention changes) become
    /// visible without having to watch the lifetime number drift.
    static let recentWindow: TimeInterval = 600

    /// One completed game, stored in the rolling window. Drops out of
    /// the window once its `timestamp` is more than `recentWindow`
    /// seconds behind `Date()`. Storage is O(games in the last 10
    /// minutes); at typical self-play rates this is a few thousand
    /// records max.
    private struct GameRecord {
        let timestamp: Date
        let moves: Int
        let durationMs: Double
    }

    private let lock = NSLock()
    private var _totalGames: Int = 0
    private var _totalMoves: Int = 0
    private var _totalGameWallMs: Double = 0
    private var _whiteCheckmates: Int = 0
    private var _blackCheckmates: Int = 0
    private var _stalemates: Int = 0
    private var _fiftyMoveDraws: Int = 0
    private var _threefoldRepetitionDraws: Int = 0
    private var _insufficientMaterialDraws: Int = 0
    private var _trainingSteps: Int = 0
    private var _recentGames: [GameRecord] = []
    /// Wall-clock anchor used as the denominator for every session
    /// rate the UI shows (games/hr, moves/hr, steps/sec, avg move ms,
    /// "Total session time", ...). Initially set to the moment the
    /// box is created, then advanced by `markWorkersStarted()` to
    /// the moment the worker task group actually begins — Play-and-
    /// Train setup (network builds, trainer reset, weight copies)
    /// would otherwise inflate the denominator by several seconds
    /// and pull every rate down proportionally.
    private var _sessionStart: Date

    init(sessionStart: Date = Date()) {
        self._sessionStart = sessionStart
    }

    /// Advance `sessionStart` to `Date()`. Called once from inside
    /// the Play-and-Train task, immediately before the worker group
    /// is spawned, so that rate denominators only cover the window
    /// in which workers are actually running.
    func markWorkersStarted() {
        lock.lock()
        defer { lock.unlock() }
        _sessionStart = Date()
    }

    /// Record one completed self-play game. Called from every worker
    /// at game-end with the game's total moves, wall-clock duration,
    /// and final result. Bumps lifetime totals, the per-outcome
    /// counters, and the rolling 10-minute window. Thread-safe via
    /// the box's `NSLock`.
    func recordCompletedGame(moves: Int, durationMs: Double, result: GameResult) {
        lock.lock()
        defer { lock.unlock() }
        _totalGames += 1
        _totalMoves += moves
        _totalGameWallMs += durationMs

        switch result {
        case .checkmate(let winner):
            if winner == .white {
                _whiteCheckmates += 1
            } else {
                _blackCheckmates += 1
            }
        case .stalemate:
            _stalemates += 1
        case .drawByFiftyMoveRule:
            _fiftyMoveDraws += 1
        case .drawByInsufficientMaterial:
            _insufficientMaterialDraws += 1
        case .drawByThreefoldRepetition:
            _threefoldRepetitionDraws += 1
        }

        let now = Date()
        _recentGames.append(GameRecord(timestamp: now, moves: moves, durationMs: durationMs))
        pruneRecentLocked(now: now)
    }

    func recordTrainingStep() {
        lock.lock()
        defer { lock.unlock() }
        _trainingSteps += 1
    }

    /// Drop rolling-window entries older than `now - recentWindow`.
    /// Caller must hold `lock`. Records are appended in monotonic
    /// timestamp order so this is a prefix removal — O(k) where k is
    /// the expired count.
    private func pruneRecentLocked(now: Date) {
        let cutoff = now.addingTimeInterval(-Self.recentWindow)
        while let first = _recentGames.first, first.timestamp < cutoff {
            _recentGames.removeFirst()
        }
    }

    struct Snapshot: Sendable {
        // Lifetime totals
        let selfPlayGames: Int
        let selfPlayPositions: Int
        let totalGameWallMs: Double
        let whiteCheckmates: Int
        let blackCheckmates: Int
        let stalemates: Int
        let fiftyMoveDraws: Int
        let threefoldRepetitionDraws: Int
        let insufficientMaterialDraws: Int
        let trainingSteps: Int
        let sessionStart: Date
        // Rolling 10-minute window aggregates
        let recentGames: Int
        let recentMoves: Int
        let recentGameWallMs: Double
        /// Effective denominator for rolling rate calculations, in
        /// seconds: `min(recentWindow, now - oldest-entry-timestamp)`.
        /// Starts at 0 and grows to `recentWindow` over the first 10
        /// minutes of a session. `0` means the window holds no games.
        let recentWindowSeconds: Double
    }

    func snapshot() -> Snapshot {
        lock.lock()
        defer { lock.unlock() }
        let now = Date()
        pruneRecentLocked(now: now)

        var recentMoves = 0
        var recentGameWallMs: Double = 0
        for r in _recentGames {
            recentMoves += r.moves
            recentGameWallMs += r.durationMs
        }
        let recentWindowSec: Double
        if let oldest = _recentGames.first?.timestamp {
            recentWindowSec = min(Self.recentWindow, now.timeIntervalSince(oldest))
        } else {
            recentWindowSec = 0
        }

        return Snapshot(
            selfPlayGames: _totalGames,
            selfPlayPositions: _totalMoves,
            totalGameWallMs: _totalGameWallMs,
            whiteCheckmates: _whiteCheckmates,
            blackCheckmates: _blackCheckmates,
            stalemates: _stalemates,
            fiftyMoveDraws: _fiftyMoveDraws,
            threefoldRepetitionDraws: _threefoldRepetitionDraws,
            insufficientMaterialDraws: _insufficientMaterialDraws,
            trainingSteps: _trainingSteps,
            sessionStart: _sessionStart,
            recentGames: _recentGames.count,
            recentMoves: recentMoves,
            recentGameWallMs: recentGameWallMs,
            recentWindowSeconds: recentWindowSec
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
    /// Additional inference networks for concurrent self-play workers
    /// 1..N-1 (worker 0 still uses `network`, the champion). One
    /// entry per extra worker beyond the first; built lazily the
    /// first time a Play and Train session needs more than one
    /// worker and cached for the life of the app. Kept in weight
    /// lock-step with `network` via the session-start fork and the
    /// arena promotion branch — each secondary receives a
    /// `loadWeights` copy of whatever goes into the champion.
    @State private var secondarySelfPlayNetworks: [ChessMPSNetwork] = []
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
    /// User-override inbox for an in-flight arena. The Abort and
    /// Promote buttons (visible only while an arena is running) write
    /// to this box; `runArenaParallel` polls it to break the game
    /// loop early and to branch on promote-vs-no-promote once the
    /// driver returns. Nil between Play-and-Train sessions.
    @State private var arenaOverrideBox: ArenaOverrideBox?
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
    nonisolated static let trainerLearningRateDefault: Float = 0.1
    nonisolated static let trainingBatchSize = 1024

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
    nonisolated static let replayBufferCapacity = 1_000_000
    /// Don't start sampling training batches until the buffer holds at least
    /// this many positions — the greater of a 25k-position floor and 20% of
    /// the ring's capacity. The floor keeps small buffers from training on a
    /// tiny, heavily-correlated warmup cohort; the 20% fraction keeps larger
    /// buffers from starting to train before the ring has enough diversity
    /// to produce meaningfully decorrelated minibatches. `minBufferBeforeTraining
    /// >= trainingBatchSize` by construction, so the `ReplayBuffer.sample`
    /// call inside the train loop can never return nil.
    nonisolated static let minBufferBeforeTraining = max(25_000, replayBufferCapacity / 5)
    /// Default number of active self-play workers when a new
    /// Play and Train session starts. The Stepper and
    /// `@State selfPlayWorkerCount` below both default to this
    /// value — it's the *initial* setting, **not** an upper
    /// bound. The user can raise or lower the live count at any
    /// time via the Stepper, and changes take effect at each
    /// worker's next game-end check. Edit to change the default.
    nonisolated static let initialSelfPlayWorkerCount: Int = 5
    /// Hard ceiling on how many self-play workers can run
    /// concurrently in a single session. We pre-build this many
    /// inference networks (one champion plus
    /// `absoluteMaxSelfPlayWorkers - 1` secondaries) and spawn
    /// this many worker tasks at session start, idling any above
    /// the current active count. The only reason this is capped
    /// at all is the steady-state memory footprint — each extra
    /// inference network costs ~12 MB of unified memory plus
    /// some MPSGraph scratch. Raise for headroom, lower to save
    /// memory. Must be ≥ `initialSelfPlayWorkerCount`.
    nonisolated static let absoluteMaxSelfPlayWorkers: Int = 16
    /// Current active self-play worker count for the running
    /// session. The Stepper writes through `workerCountBinding`
    /// which updates this @State and `workerCountBox` atomically;
    /// workers poll the box at the top of each iteration to
    /// decide whether to play another game or sit in their idle
    /// wait state. Defaults to `initialSelfPlayWorkerCount`;
    /// bounded at runtime by `absoluteMaxSelfPlayWorkers`.
    @State private var selfPlayWorkerCount: Int = Self.initialSelfPlayWorkerCount
    /// Upper bound on the adjustable training-step delay. 500 ms
    /// already turns a ~60 steps/s training worker into roughly
    /// 2 steps/s, which is as slow as anyone reasonably wants to
    /// crawl the learning rate while still making progress.
    nonisolated static let stepDelayMaxMs: Int = 2000
    /// Discrete ladder of valid training-step delay values in
    /// milliseconds. Fine-grained 5 ms increments at the low end
    /// where small delays matter most, then 25 ms increments all
    /// the way up to `stepDelayMaxMs`. The Stepper's +/- clicks
    /// walk this ladder one rung at a time via
    /// `trainingStepDelayBinding`.
    nonisolated static let stepDelayLadder: [Int] =
        [0, 5, 10, 15, 20] + Array(stride(from: 25, through: Self.stepDelayMaxMs, by: 25))
    /// Current training-step delay in milliseconds, written by
    /// the Stepper via `trainingStepDelayBinding` and mirrored into
    /// `trainingStepDelayBox` so the training worker task reads
    /// the live value between steps. Always a member of
    /// `stepDelayLadder`. Persists across sessions (@State) so the
    /// user doesn't have to re-pick the delay on every start. The
    /// 50 ms default holds the training worker to a modest ~20
    /// steps/s ceiling so it doesn't starve the N self-play workers
    /// of GPU time on a fresh session start; the user can drop it
    /// to 0 ms once the self-play throughput has stabilized.
    @AppStorage("trainingStepDelayMs") private var trainingStepDelayMs: Int = 50
    /// Shared lock-protected mirror of `trainingStepDelayMs` that
    /// the training worker task reads at the bottom of each step
    /// to decide how long to sleep before looping. Allocated at
    /// session start, cleared on session end.
    @State private var trainingStepDelayBox: TrainingStepDelayBox?
    /// Shared lock-protected mirror of `selfPlayWorkerCount` that
    /// the self-play worker tasks read between games. The Stepper
    /// updates `selfPlayWorkerCount` AND this box atomically (via
    /// the binding side-effect); workers poll the box at the top
    /// of each iteration to decide whether to play another game
    /// or stay in their idle wait state. Allocated at session
    /// start, cleared on session end.
    @State private var workerCountBox: WorkerCountBox?
    /// Cached memory-stats line shown in the top busy row during
    /// Play and Train. Refreshed at most every
    /// `memoryStatsRefreshSec` seconds via
    /// `refreshMemoryStatsIfNeeded()` (called from the snapshot
    /// timer) so the displayed numbers don't churn at 60 Hz.
    @State private var memoryStatsSnap: MemoryStatsSnapshot?
    /// Wall-clock timestamp of the most recent `memoryStatsSnap`
    /// refresh. Defaults to `.distantPast` so the first refresh
    /// always fires. Compared against `now - memoryStatsRefreshSec`
    /// inside the heartbeat to decide whether to take a new sample.
    @State private var memoryStatsLastFetch: Date = .distantPast
    /// How long the cached memory stats are reused before the
    /// next sample. The user explicitly asked for a 10-second
    /// cadence — RAM and GPU footprint don't change visibly more
    /// often than that, and resampling 60×/s would just churn
    /// the display.
    nonisolated static let memoryStatsRefreshSec: Double = 10
    /// Previous `ProcessUsageSample` held so the next heartbeat
    /// can compute %CPU and %GPU from the delta. `nil` until the
    /// first successful sample lands; after that it rolls forward
    /// at the `usageStatsRefreshSec` cadence.
    @State private var lastUsageSample: ProcessUsageSample?
    /// Wall-clock timestamp of the most recent usage refresh.
    /// Defaults to `.distantPast` so the first heartbeat tick
    /// always fires a sample.
    @State private var usageStatsLastFetch: Date = .distantPast
    /// Last-computed %CPU over the real wall-clock elapsed since
    /// the previous sample. Relative to one core, so on multi-core
    /// CPUs this can exceed 100% (matching `top`'s convention).
    /// `nil` until the second sample lands — no delta to divide
    /// from a single reading.
    @State private var cpuPercent: Double?
    /// Last-computed %GPU over the same interval as `cpuPercent`.
    /// Relative to one GPU engine; workloads that keep several
    /// engines busy can exceed 100%. `nil` until the second sample.
    @State private var gpuPercent: Double?
    /// Cadence at which the CPU/GPU utilisation refreshes.
    /// The user asked for ~5 s; the computed percentages always
    /// divide by the actual wall-clock gap between samples, so
    /// heartbeat drift or a paused app don't bias the result.
    nonisolated static let usageStatsRefreshSec: Double = 5
    /// Append-only list of progress-rate samples for the Progress
    /// rate chart. Grows by one entry per `progressRateRefreshSec`
    /// during a Play and Train session; cleared to `[]` at each
    /// new session start in `startRealTraining()`.
    @State private var progressRateSamples: [ProgressRateSample] = []
    /// Wall-clock timestamp of the last progress-rate sample.
    /// Defaults to `.distantPast` so the first heartbeat tick of
    /// a new session always fires.
    @State private var progressRateLastFetch: Date = .distantPast
    /// Next `id` to assign when appending to `progressRateSamples`.
    /// Monotonic counter, reset alongside the sample list.
    @State private var progressRateNextId: Int = 0
    /// Current scroll position on the Progress rate chart, in
    /// elapsed seconds since session start. Acts as the X
    /// coordinate that maps to the *left edge* of the visible
    /// window (so `scrollX = 0` shows t=0..window, and
    /// `scrollX = lastElapsed - window` shows the latest
    /// window). Two-way bound to `.chartScrollPosition(x:)` so
    /// `Swift Charts` handles the native scroll gestures
    /// (trackpad two-finger scroll, scroll wheel, arrow keys)
    /// without any custom DragGesture of our own.
    @State private var progressRateScrollX: Double = 0
    /// Whether the chart should auto-advance `progressRateScrollX`
    /// to keep the newest sample in view. Starts `true`; flips
    /// to `false` when the user scrolls backward past a small
    /// tolerance, and flips back to `true` when they scroll
    /// forward to the right edge again. Without this, a user
    /// scrolling back to inspect history would get yanked
    /// forward on every 1 Hz tick.
    @State private var progressRateFollowLatest: Bool = true
    /// Cadence for the progress-rate sampler: one sample per
    /// second. Matches the user's spec.
    nonisolated static let progressRateRefreshSec: Double = 1.0
    /// Rolling window width used to compute each sample's
    /// moves/hr from the delta between "now" and "the sample
    /// closest to 3 minutes ago". 180 s, as requested.
    nonisolated static let progressRateWindowSec: Double = 180.0
    /// Visible X-axis length shown on the Progress rate chart
    /// at any one time, in elapsed seconds. The chart scrolls
    /// horizontally through the full session's data in chunks
    /// of this size. 10 minutes matches the existing "last 10m"
    /// rolling column in the Self Play stats panel, so the eye
    /// can correlate chart movement with the numeric column.
    nonisolated static let progressRateVisibleDomainSec: Double = 600
    /// Wall-clock seconds the Play and Train Session panel waits
    /// after session start before showing rate-based stats fields
    /// (Moves/hr, Games/hr in both lifetime and 10-min columns).
    /// Below this threshold the very first game's near-zero
    /// elapsed denominator would print absurd millions-of-moves/hr
    /// values; the dashes fade in once the session has had enough
    /// wall clock for the rates to be meaningful. Per-game and
    /// per-move averages aren't gated — they don't divide by wall
    /// clock so they're correct from the first completed game.
    nonisolated static let statsWarmupSeconds: Double = 5.0

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

    // MARK: - Checkpoint state (save / load models and sessions)

    /// Stable session identifier, minted at Play-and-Train start and
    /// carried through every autosave and manual session save for
    /// the life of that run. Re-minted on the next start. Used as
    /// the middle token in `.dcmsession` directory names so
    /// successive saves from the same run cluster together
    /// alphabetically in Finder.
    @State private var currentSessionID: String?

    /// Wall-clock anchor for the current session, captured at
    /// startRealTraining time before worker setup. Used by the
    /// session-save path to compute `elapsedTrainingSec`.
    @State private var currentSessionStart: Date?

    /// A parsed session that was loaded from disk but not yet
    /// applied. The user loads a session while Play-and-Train is
    /// stopped; the next `startRealTraining` call consumes this
    /// and seeds the trainer / counters / IDs from it, then
    /// clears it.
    @State private var pendingLoadedSession: LoadedSession?

    /// A parsed standalone model that was loaded from disk but
    /// not yet applied. Consumed by a follow-up network build
    /// or by `startRealTraining` to initialize the champion's
    /// weights from the loaded file. Cleared on apply.
    @State private var pendingLoadedModel: ModelCheckpointFile?

    /// Last user-facing checkpoint status message. Shown briefly
    /// in the busy row. Cleared after a few seconds.
    @State private var checkpointStatusMessage: String?

    /// True when the last checkpoint status message represents an
    /// error (shown in red rather than secondary).
    @State private var checkpointStatusIsError: Bool = false

    /// Drives the Load Model file importer sheet.
    @State private var showingLoadModelImporter: Bool = false

    /// Drives the Load Session file importer sheet.
    @State private var showingLoadSessionImporter: Bool = false

    /// Whether an autosave is currently in flight, so repeated
    /// promotions or rapid manual saves don't overlap. Advisory only
    /// — the save path is idempotent except for the "already exists"
    /// check which throws cleanly.
    @State private var checkpointSaveInFlight: Bool = false

    /// Live reference to worker 0's self-play pause gate for the
    /// current Play-and-Train session. Set at session start by
    /// `startRealTraining` and cleared at session end. Used by
    /// the checkpoint save path to briefly pause champion exports
    /// without having to reach into the task-group closure.
    @State private var activeSelfPlayGate: WorkerPauseGate?

    /// Live reference to the training worker's pause gate for
    /// the current Play-and-Train session. Set at session start
    /// and cleared at session end. Used by the checkpoint save
    /// path to briefly pause trainer weight exports.
    @State private var activeTrainingGate: WorkerPauseGate?

    /// Replay-ratio controller that tracks the 1-minute rolling
    /// ratio of training consumption to self-play production and
    /// auto-adjusts the training step delay to keep them balanced.
    /// Created at session start, polled by the UI heartbeat for
    /// display, and cleared at session end.
    @State private var replayRatioController: ReplayRatioController?
    /// Latest snapshot from `replayRatioController`, mirrored by
    /// the heartbeat for UI display.
    @State private var replayRatioSnapshot: ReplayRatioController.RatioSnapshot?
    /// User-adjustable target consumption/production ratio.
    /// Default 1.0 = balanced. Values >1 let training outpace
    /// self-play (higher replay ratio); <1 the opposite. Persisted
    /// in session checkpoints.
    @AppStorage("replayRatioTarget") private var replayRatioTarget: Double = 1.0
    @AppStorage("trainerLearningRate") private var trainerLearningRate: Double = Double(trainerLearningRateDefault)
    /// Whether the ratio controller auto-adjusts the training step
    /// delay. When off, the manual Stepper controls the delay
    /// directly. Persisted in session checkpoints.
    @State private var replayRatioAutoAdjust: Bool = true

    /// 100 ms heartbeat that pulls the latest snapshot from `gameWatcher`
    /// into `gameSnapshot`. Standard SwiftUI Combine timer pattern — the
    /// publisher is created when the view struct is initialized and SwiftUI
    /// manages the subscription lifecycle via `.onReceive` below.
    private let snapshotTimer = Timer.publish(
        every: 1.0/60.0, on: .main, in: .common
    ).autoconnect()

    /// Default on/off toggle for "autosave the full session after
    /// every arena promotion." Off would skip the save; on writes a
    /// `.dcmsession` next to every manual save. Defaulting to true
    /// means promoted models are never lost by default.
    nonisolated static let autosaveSessionsOnPromote: Bool = true

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
    private var isGameMode: Bool {
        gameSnapshot.isPlaying
            || gameSnapshot.totalGames > 0
            || realTraining
    }
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

    /// True when the Play and Train Board picker is currently on the
    /// Progress rate line-chart tab. The chart takes over the board
    /// slot in the left column, so the live game board and the
    /// forward-pass editor are both suppressed while this is active.
    private var isProgressRateActive: Bool {
        realTraining && playAndTrainBoardMode == .progressRate
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

    /// Binding for the live worker count Stepper. Writes update both
    /// `@State selfPlayWorkerCount` (so the body re-renders with the
    /// new N immediately) and `workerCountBox` (so the self-play
    /// workers see the change on their next game-end check). The
    /// box is nil between sessions, so writes outside Play and
    /// Train just update the @State and take effect on the next
    /// session start.
    private var workerCountBinding: Binding<Int> {
        Binding(
            get: { selfPlayWorkerCount },
            set: { newValue in
                let clamped = max(1, min(Self.absoluteMaxSelfPlayWorkers, newValue))
                selfPlayWorkerCount = clamped
                workerCountBox?.set(clamped)
            }
        )
    }

    /// Binding for the training-step delay Stepper. The Stepper is
    /// configured with `step: 1` over the full 0...stepDelayMaxMs
    /// range, but the valid values are the discrete rungs in
    /// `stepDelayLadder`. This binding translates a raw Stepper
    /// delta (current ± 1) into "advance/retreat one ladder rung",
    /// snapping the displayed value to the nearest rung and writing
    /// through to both `@State trainingStepDelayMs` (so the row
    /// re-renders immediately) and `trainingStepDelayBox` (so the
    /// training worker task sees the new delay on its next step).
    /// The box is nil between sessions, so writes outside Play and
    /// Train just update the @State and take effect when the next
    /// session starts.
    private var trainingStepDelayBinding: Binding<Int> {
        Binding(
            get: { trainingStepDelayMs },
            set: { newValue in
                let ladder = Self.stepDelayLadder
                let current = trainingStepDelayMs
                // `trainingStepDelayMs` is seeded with a ladder rung
                // (0) and every write below snaps to another ladder
                // rung, so `firstIndex(of:)` is invariant-guaranteed
                // non-nil. A violation here would be a programmer
                // error, so crash loudly rather than silently drifting.
                guard let currentIdx = ladder.firstIndex(of: current) else {
                    preconditionFailure("trainingStepDelayMs (\(current)) is not a rung in stepDelayLadder")
                }
                let nextIdx: Int
                if newValue > current {
                    nextIdx = min(currentIdx + 1, ladder.count - 1)
                } else if newValue < current {
                    nextIdx = max(currentIdx - 1, 0)
                } else {
                    nextIdx = currentIdx
                }
                let snapped = ladder[nextIdx]
                trainingStepDelayMs = snapped
                trainingStepDelayBox?.set(snapped)
                replayRatioController?.manualDelayMs = snapped
            }
        )
    }

    /// Binding for the replay-ratio target stepper. Writes through
    /// to both `@State replayRatioTarget` (so the label re-renders
    /// immediately) and the live controller (so the next training
    /// step sees the updated target).
    private var replayRatioTargetBinding: Binding<Double> {
        Binding(
            get: { replayRatioTarget },
            set: { newValue in
                let clamped = max(0.1, min(5.0, newValue))
                replayRatioTarget = clamped
                replayRatioController?.targetRatio = clamped
            }
        )
    }

    /// Binding for the replay-ratio auto-adjust toggle. Mirrors
    /// the toggle into both `@State` and the live controller, and
    /// when turning auto OFF, also syncs the current manual delay
    /// into the controller so subsequent delay reads fall back to
    /// the Stepper's value.
    private var replayRatioAutoAdjustBinding: Binding<Bool> {
        Binding(
            get: { replayRatioAutoAdjust },
            set: { newValue in
                replayRatioAutoAdjust = newValue
                replayRatioController?.autoAdjust = newValue
                if !newValue {
                    replayRatioController?.manualDelayMs = trainingStepDelayMs
                }
            }
        )
    }

    /// Scratch string for the learning rate text field. Seeded from
    /// the trainer's current LR when Play-and-Train starts; the user
    /// edits freely without the binding reformatting mid-keystroke.
    /// The value is parsed and applied only on Enter (via `.onSubmit`
    /// on the TextField). Invalid input reverts to the current LR.
    @State private var learningRateEditText: String = ""

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

                // Abort / Promote — visible only while an arena is in
                // flight, so the user can terminate it early without
                // waiting for the full 200-game tournament. Abort ends
                // with no promotion regardless of score; Promote ends
                // early and forcibly promotes the candidate. Both go
                // through `ArenaOverrideBox` which the driver's
                // `isCancelled` closure polls between games, so the
                // actual end-of-arena happens one in-flight game later
                // (the same ~400 ms granularity Stop has).
                if realTraining && isArenaRunning {
                    Button("Abort Arena") {
                        SessionLogger.shared.log("[BUTTON] Abort Arena")
                        arenaOverrideBox?.abort()
                    }
                    Button("Promote") {
                        SessionLogger.shared.log("[BUTTON] Promote")
                        arenaOverrideBox?.promote()
                    }
                }

                if isBusy {
                    ProgressView().controlSize(.small)
                    busyLabelView
                }
            }

            // Checkpoint row: save/load buttons for models and
            // sessions plus a Reveal button that opens the
            // Library folder in Finder. See ROADMAP "Model and
            // session save/load" for format and trigger details.
            HStack(spacing: 8) {
                // Save Session is only meaningful while a
                // Play-and-Train session is running, and never
                // during an arena (mid-arena saves would have to
                // snapshot a live candidate whose weights are
                // being modified by the trainer).
                if realTraining {
                    Button("Save Session") {
                        SessionLogger.shared.log("[BUTTON] Save Session")
                        handleSaveSessionManual()
                    }
                    .disabled(isArenaRunning || checkpointSaveInFlight)
                }

                // Save Champion as a standalone .dcmmodel works
                // whenever the network exists. It's race-safe
                // during Play-and-Train because we pause
                // worker 0 via `activeSelfPlayGate`. We
                // additionally disable during an arena because
                // `runArenaParallel` is already using the same
                // gate as its own coordinator, and
                // `WorkerPauseGate` is not reentrant for
                // multiple concurrent pauseAndWait callers. In
                // other active modes (Play Game, Play
                // Continuous, Run Forward Pass) there is no
                // coordinating gate and a concurrent `evaluate`
                // would race against the export, so we disable
                // the button there too.
                if networkReady {
                    Button("Save Champion") {
                        SessionLogger.shared.log("[BUTTON] Save Champion")
                        handleSaveChampionAsModel()
                    }
                    .disabled(
                        checkpointSaveInFlight
                        || isArenaRunning
                        || (isBusy && !realTraining)
                    )
                }

                Divider().frame(height: 20)

                // Load is only offered when nothing else is
                // running — we don't support hot-swapping weights
                // into an in-flight training session.
                if !realTraining && !continuousPlay && !continuousTraining && !sweepRunning && !gameSnapshot.isPlaying {
                    Button("Load Session…") {
                        SessionLogger.shared.log("[BUTTON] Load Session")
                        showingLoadSessionImporter = true
                    }
                    .disabled(isBuilding || checkpointSaveInFlight)

                    Button("Load Model…") {
                        SessionLogger.shared.log("[BUTTON] Load Model")
                        showingLoadModelImporter = true
                    }
                    .disabled(isBuilding || checkpointSaveInFlight || !networkReady)
                }

                Divider().frame(height: 20)

                // Always available so the user can open Finder to
                // the canonical save location even when nothing is
                // saved yet.
                Button("Reveal Saves") {
                    handleRevealSaves()
                }

                if let msg = checkpointStatusMessage {
                    Text(msg)
                        .font(.callout)
                        .foregroundStyle(checkpointStatusIsError ? .red : .secondary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                }
            }
            .fileImporter(
                isPresented: $showingLoadModelImporter,
                allowedContentTypes: [.data, .item],
                allowsMultipleSelection: false,
                onCompletion: { result in
                    handleLoadModelPickResult(result)
                }
            )
            .fileImporter(
                isPresented: $showingLoadSessionImporter,
                allowedContentTypes: [.folder],
                allowsMultipleSelection: false,
                onCompletion: { result in
                    handleLoadSessionPickResult(result)
                }
            )
            .fileDialogDefaultDirectory(
                showingLoadModelImporter
                    ? CheckpointPaths.modelsDir
                    : CheckpointPaths.sessionsDir
            )

            // Board + text side by side
            HStack(alignment: .top, spacing: 24) {
                VStack(spacing: 6) {
                    if realTraining {
                        Picker("Board", selection: playAndTrainBoardBinding) {
                            Text("Game run").tag(PlayAndTrainBoardMode.gameRun)
                            Text("Candidate test").tag(PlayAndTrainBoardMode.candidateTest)
                            Text("Progress rate").tag(PlayAndTrainBoardMode.progressRate)
                        }
                        .pickerStyle(.segmented)
                        .labelsHidden()
                        .frame(maxWidth: 360)
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

                    if isProgressRateActive {
                        progressRateChartView
                    } else {
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
                            .overlay {
                                // Multi-worker placeholder — the live
                                // animated game board only works with
                                // one driving worker (N=1), because a
                                // single `GameWatcher` can't track
                                // multiple concurrent games without
                                // flicker. When N>1 we still show the
                                // board slot (so the Candidate test
                                // picker remains usable and the layout
                                // doesn't shift) but overlay a centered
                                // label indicating how many workers
                                // are running. Hidden in candidate-test
                                // mode so the probe board stays clean.
                                if realTraining
                                    && !isCandidateTestActive
                                    && selfPlayWorkerCount > 1 {
                                    Text("N = \(selfPlayWorkerCount) concurrent games\nLive board hidden")
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
                                // Play and Train uses the aggregate
                                // stats text — all N workers feed
                                // into `parallelStats`, and the text
                                // builder shows a Status line only
                                // when N=1. Non-realTraining modes
                                // (Play Game / Play Continuous) still
                                // use `gameSnapshot.statsText`, which
                                // is their single source of truth.
                                if realTraining, let session = parallelStats {
                                    let column = playAndTrainStatsText(
                                        game: gameSnapshot,
                                        session: session
                                    )
                                    // Split layout: header Text, then the
                                    // Concurrency control row with the live
                                    // N Stepper, then the body Text. Zero
                                    // spacing so the three pieces read as a
                                    // single continuous block. The HStack's
                                    // leading "  " mirrors the body's two-
                                    // space label indent, and the minWidth
                                    // on the value Text keeps the Stepper
                                    // from jittering horizontally when the
                                    // count changes width (1 ↔ 16).
                                    VStack(alignment: .leading, spacing: 0) {
                                        Text(column.header)
                                        HStack(spacing: 6) {
                                            Text("  Concurrency:")
                                            Text("\(selfPlayWorkerCount)")
                                                .monospacedDigit()
                                                .frame(minWidth: 24, alignment: .trailing)
                                            Stepper(
                                                "Concurrency",
                                                value: workerCountBinding,
                                                in: 1...Self.absoluteMaxSelfPlayWorkers
                                            )
                                            .labelsHidden()
                                        }
                                        Text(column.body)
                                    }
                                    .frame(minWidth: 330, alignment: .topLeading)
                                } else {
                                    Text(gameSnapshot.statsText(
                                        continuousPlay: continuousPlay || realTraining
                                    ))
                                    .frame(minWidth: 330, alignment: .topLeading)
                                }
                            }
                            if isCandidateTestActive, let result = inferenceResult {
                                Text(result.textOutput)
                                    .frame(minWidth: 330, alignment: .topLeading)
                            }
                            if isTrainingMode {
                                let column = trainingStatsText()
                                // Split layout mirroring the Self Play
                                // column: header, then the Step Delay row
                                // (only during Play and Train — the delay
                                // only affects the live training worker),
                                // then the body. Sweep mode still runs
                                // through this branch but `realTraining`
                                // is false there, so the control row is
                                // omitted and the sweep table renders
                                // exactly as before.
                                VStack(alignment: .leading, spacing: 0) {
                                    Text(column.header)
                                    if realTraining {
                                        HStack(spacing: 6) {
                                            Text("  Step Delay:")
                                            if let snap = replayRatioSnapshot, snap.autoAdjust {
                                                Text("\(snap.computedDelayMs)")
                                                    .monospacedDigit()
                                                    .frame(minWidth: 32, alignment: .trailing)
                                                Text("ms (auto)")
                                                    .foregroundStyle(.secondary)
                                            } else {
                                                Text("\(trainingStepDelayMs)")
                                                    .monospacedDigit()
                                                    .frame(minWidth: 32, alignment: .trailing)
                                                Text("ms")
                                            }
                                            Stepper(
                                                "Step Delay",
                                                value: trainingStepDelayBinding,
                                                in: 0...Self.stepDelayMaxMs
                                            )
                                            .labelsHidden()
                                            .disabled(replayRatioAutoAdjust)
                                        }
                                        HStack(spacing: 6) {
                                            Text("  Replay Ratio:")
                                            if let snap = replayRatioSnapshot {
                                                Text(String(format: "%.2f", snap.currentRatio))
                                                    .monospacedDigit()
                                                    .frame(minWidth: 40, alignment: .trailing)
                                                    .foregroundStyle(
                                                        abs(snap.currentRatio - snap.targetRatio) < 0.3
                                                            ? Color.primary : Color.red
                                                    )
                                            } else {
                                                Text("--")
                                                    .monospacedDigit()
                                                    .frame(minWidth: 40, alignment: .trailing)
                                            }
                                            Text("target:")
                                                .foregroundStyle(.secondary)
                                            Text(String(format: "%.1f", replayRatioTarget))
                                                .monospacedDigit()
                                                .frame(minWidth: 24, alignment: .trailing)
                                            Stepper(
                                                "Target Ratio",
                                                value: replayRatioTargetBinding,
                                                in: 0.1...5.0,
                                                step: 0.1
                                            )
                                            .labelsHidden()
                                            Toggle("Auto", isOn: replayRatioAutoAdjustBinding)
                                                .toggleStyle(.checkbox)
                                        }
                                        HStack(spacing: 6) {
                                            Text("  Learn Rate:")
                                            TextField("LR", text: $learningRateEditText)
                                                .monospacedDigit()
                                                .frame(width: 80)
                                                .textFieldStyle(.roundedBorder)
                                                .onSubmit {
                                                    if let parsed = Float(learningRateEditText),
                                                       parsed > 0, parsed.isFinite {
                                                        trainer?.learningRate = parsed
                                                        trainerLearningRate = Double(parsed)
                                                    }
                                                    learningRateEditText = String(
                                                        format: "%.1e",
                                                        trainer?.learningRate ?? Self.trainerLearningRateDefault
                                                    )
                                                }
                                        }
                                    }
                                    Text(column.body)
                                }
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
            // timestamp is embedded in the snapshot so the Session
            // panel and busy label can compute wall-clock rates on
            // every render. Dirty check compares the fields that
            // advance on self-play and training events; if either
            // has changed (or the rolling-window count has shifted
            // because an entry aged out), push a new snapshot.
            if let pBox = parallelWorkerStatsBox {
                let snap = pBox.snapshot()
                let prev = parallelStats
                // `sessionStart` is included in the dirty check so
                // the one-time shift performed by
                // `markWorkersStarted()` (right before the worker
                // group spawns) lands in @State immediately, even
                // if no game or training step has recorded yet.
                let changed = snap.selfPlayGames != (prev?.selfPlayGames ?? -1)
                    || snap.trainingSteps != (prev?.trainingSteps ?? -1)
                    || snap.recentGames != (prev?.recentGames ?? -1)
                    || snap.sessionStart != prev?.sessionStart
                if changed {
                    parallelStats = snap
                }
            }
            // Memory stats refresh. Throttled internally to
            // `memoryStatsRefreshSec` so this is a cheap timestamp
            // compare on most heartbeats.
            refreshMemoryStatsIfNeeded()
            // Process %CPU / %GPU refresh — separate (5 s) cadence
            // from memory stats (10 s) so the utilisation line
            // updates twice as often without dragging the heavier
            // Metal property reads along with it.
            refreshUsagePercentsIfNeeded()
            // Progress-rate chart sampler. 1 Hz during Play and
            // Train; each sample carries the moves/hr averaged
            // over the last 3 minutes of work. No-op outside of
            // realTraining.
            refreshProgressRateIfNeeded()
            // Replay-ratio snapshot for the UI.
            if let rc = replayRatioController {
                replayRatioSnapshot = rc.snapshot()
            }
        }
    }

    /// Sample app and GPU memory at most every
    /// `memoryStatsRefreshSec` seconds, caching the result in
    /// `memoryStatsSnap` for the busy label to read. Cheap on a
    /// no-op tick (a single timestamp diff) so it's fine to call
    /// from the 60 Hz heartbeat. The actual sampling reads
    /// `task_info` and a couple of `MTLDevice` properties via
    /// the trainer's existing helpers.
    private func refreshMemoryStatsIfNeeded() {
        let now = Date()
        if now.timeIntervalSince(memoryStatsLastFetch) < Self.memoryStatsRefreshSec {
            return
        }
        let app = ChessTrainer.currentPhysFootprintBytes()
        let caps = trainer?.deviceMemoryCaps()
        memoryStatsSnap = MemoryStatsSnapshot(
            appFootprintBytes: app,
            gpuAllocatedBytes: caps?.currentAllocated ?? 0,
            gpuMaxTargetBytes: caps?.recommendedMaxWorkingSet ?? 0,
            gpuTotalBytes: ProcessInfo.processInfo.physicalMemory
        )
        memoryStatsLastFetch = now
    }

    /// Append a new progress-rate sample at most once per
    /// `progressRateRefreshSec` during a Play and Train session.
    /// The moves/hr fields are computed over a real trailing
    /// 3-minute window: we walk backward from the newest stored
    /// sample until we find the first one whose `timestamp` is
    /// still inside the window, then subtract its cumulative
    /// counters from the current cumulative counters and divide
    /// by the actual elapsed seconds between the two samples.
    ///
    /// Before the session has 3 minutes of history, the window
    /// shrinks gracefully to "whatever we have" — the first
    /// sample reports zero (no earlier sample to subtract from),
    /// the second reports over ~1 s, and so on until the window
    /// reaches its full 180 s width.
    ///
    /// No-op outside of `realTraining`. Sampler state is cleared
    /// by `startRealTraining()` so each session's chart starts
    /// fresh from t=0.
    private func refreshProgressRateIfNeeded() {
        guard realTraining else { return }
        let now = Date()
        if now.timeIntervalSince(progressRateLastFetch) < Self.progressRateRefreshSec {
            return
        }
        progressRateLastFetch = now

        guard let session = parallelStats else { return }
        let elapsed = max(0, now.timeIntervalSince(session.sessionStart))
        let curSp = session.selfPlayPositions
        let curTr = (trainingStats?.steps ?? 0) * Self.trainingBatchSize

        // Walk newest → oldest, recording the last sample we see
        // that still falls inside the 3-minute window. Breaks out
        // as soon as we hit a sample older than the cutoff — the
        // list is timestamp-sorted, so anything older is also
        // out of window. Bounded at ~180 iterations per call in
        // steady state regardless of total session length.
        let cutoff = now.addingTimeInterval(-Self.progressRateWindowSec)
        var windowStart: ProgressRateSample?
        for sample in progressRateSamples.reversed() {
            if sample.timestamp >= cutoff {
                windowStart = sample
            } else {
                break
            }
        }

        let spRate: Double
        let trRate: Double
        if let ws = windowStart {
            let dt = now.timeIntervalSince(ws.timestamp)
            if dt > 0 {
                let spDelta = max(0, curSp - ws.selfPlayCumulativeMoves)
                let trDelta = max(0, curTr - ws.trainingCumulativeMoves)
                spRate = Double(spDelta) / dt * 3600
                trRate = Double(trDelta) / dt * 3600
            } else {
                spRate = 0
                trRate = 0
            }
        } else {
            // First sample of the session — nothing to diff
            // against yet. Rate reads as zero for this one tick
            // and the chart picks up real values from the next.
            spRate = 0
            trRate = 0
        }

        let sample = ProgressRateSample(
            id: progressRateNextId,
            timestamp: now,
            elapsedSec: elapsed,
            selfPlayCumulativeMoves: curSp,
            trainingCumulativeMoves: curTr,
            selfPlayMovesPerHour: spRate,
            trainingMovesPerHour: trRate
        )
        progressRateSamples.append(sample)
        progressRateNextId += 1

        // Auto-follow: pin the scroll position so the latest
        // sample sits at the right edge of the visible window.
        // Disabled when the user has manually scrolled backward
        // (the binding on `.chartScrollPosition(x:)` flips this
        // flag when it sees a backward jump), so inspecting
        // history doesn't fight the 1 Hz tick.
        if progressRateFollowLatest {
            progressRateScrollX = max(0, sample.elapsedSec - Self.progressRateVisibleDomainSec)
        }
    }

    /// Format a number of elapsed seconds for the Progress rate
    /// chart's X-axis. Picks a display granularity that matches
    /// the magnitude of the value so early-session axis labels
    /// read "0:15 / 0:30 / 0:45" rather than "0.0 / 0.0 / 0.0":
    ///
    /// * < 60 s: "0:SS"
    /// * < 3600 s: "M:SS"
    /// * ≥ 3600 s: "H:MM:SS"
    ///
    /// Negative values are clamped to 0 — shouldn't happen given
    /// the sampler only produces non-negative elapsed values, but
    /// the chart's axis automatic-ticks can overshoot into negative
    /// space briefly during pan gestures at the left edge.
    static func formatElapsedAxis(_ seconds: Double) -> String {
        let secs = max(0, Int(seconds.rounded()))
        let h = secs / 3600
        let m = (secs % 3600) / 60
        let s = secs % 60
        if h > 0 {
            return String(format: "%d:%02d:%02d", h, m, s)
        } else if secs >= 60 {
            return String(format: "%d:%02d", m, s)
        } else {
            return String(format: "0:%02d", s)
        }
    }

    /// The Progress rate chart. Three line series (self-play,
    /// training, combined) plotted against elapsed session time.
    /// Horizontal scrolling is native: `chartScrollableAxes` +
    /// `chartXVisibleDomain` set a 10-minute visible window and
    /// let Swift Charts handle the trackpad / mouse / keyboard
    /// scroll gestures. `chartScrollPosition(x:)` is two-way
    /// bound to `progressRateScrollX`; the binding's setter is
    /// where we detect a manual scroll and pause auto-follow so
    /// reading history doesn't fight the 1 Hz sampler tick.
    private var progressRateChartView: some View {
        Chart(progressRateSamples) { sample in
            LineMark(
                x: .value("Elapsed", sample.elapsedSec),
                y: .value("Moves/hr", sample.combinedMovesPerHour)
            )
            .foregroundStyle(by: .value("Series", "Combined"))

            LineMark(
                x: .value("Elapsed", sample.elapsedSec),
                y: .value("Moves/hr", sample.selfPlayMovesPerHour)
            )
            .foregroundStyle(by: .value("Series", "Self-play"))

            LineMark(
                x: .value("Elapsed", sample.elapsedSec),
                y: .value("Moves/hr", sample.trainingMovesPerHour)
            )
            .foregroundStyle(by: .value("Series", "Training"))
        }
        .chartForegroundStyleScale([
            "Self-play": Color.blue,
            "Training": Color.orange,
            "Combined": Color.green
        ])
        .chartXAxis {
            AxisMarks(values: .automatic(desiredCount: 6)) { value in
                AxisGridLine()
                AxisTick()
                AxisValueLabel {
                    if let secs = value.as(Double.self) {
                        Text(Self.formatElapsedAxis(secs))
                            .monospacedDigit()
                    }
                }
            }
        }
        .chartYAxis {
            AxisMarks(position: .leading, values: .automatic(desiredCount: 6)) { value in
                AxisGridLine()
                AxisTick()
                AxisValueLabel {
                    if let v = value.as(Double.self) {
                        Text(v.formatted(.number.notation(.compactName)))
                            .monospacedDigit()
                    }
                }
            }
        }
        .chartXAxisLabel("Session time", position: .bottom, alignment: .center)
        .chartYAxisLabel("Moves / hour", position: .leading, alignment: .center)
        .chartLegend(position: .bottom, alignment: .center, spacing: 10)
        .chartScrollableAxes(.horizontal)
        .chartXVisibleDomain(length: Self.progressRateVisibleDomainSec)
        .chartScrollPosition(x: progressRateScrollBinding)
        .frame(height: 320)
    }

    /// Two-way binding between `progressRateScrollX` and the
    /// chart's `chartScrollPosition(x:)`. The setter is where
    /// user scrolls land: if the new position is more than 1 s
    /// away from the "latest" scroll position, auto-follow
    /// pauses; if they scroll back to (within 1 s of) the right
    /// edge, it resumes. 1 s matches the sampler cadence, so
    /// one tick of slack lines up with one natural scroll-by-
    /// one-sample gesture.
    private var progressRateScrollBinding: Binding<Double> {
        Binding(
            get: { progressRateScrollX },
            set: { newValue in
                // Swift Charts echoes the binding back with its own
                // (sometimes identical) value on the same frame we
                // advanced auto-follow. Without this guard the chart's
                // internal onChange(of: ChartScrollPositionConfiguration)
                // sees two writes per frame and logs a warning. Only
                // propagate the write if the position actually moved.
                if progressRateScrollX == newValue {
                    return
                }
                progressRateScrollX = newValue
                let latest = progressRateSamples.last?.elapsedSec ?? 0
                let latestScrollX = max(0, latest - Self.progressRateVisibleDomainSec)
                progressRateFollowLatest = abs(newValue - latestScrollX) < 1.0
            }
        )
    }

    /// Sample process CPU + GPU time at most every
    /// `usageStatsRefreshSec` seconds, compute the percentage over
    /// the real wall-clock elapsed since the previous sample, and
    /// publish the result into `cpuPercent` / `gpuPercent`. The
    /// math always uses the real `timestamp` delta (not the nominal
    /// cadence), so a paused heartbeat, a missed tick, or a session
    /// restart doesn't skew the reading. If the gap between samples
    /// is more than 3× the cadence — e.g. the app was idle for a
    /// while — the previous sample is discarded rather than used,
    /// because an interval much larger than the polling window is
    /// usually not what the user wants averaged over.
    private func refreshUsagePercentsIfNeeded() {
        let now = Date()
        if now.timeIntervalSince(usageStatsLastFetch) < Self.usageStatsRefreshSec {
            return
        }
        usageStatsLastFetch = now
        guard let sample = ChessTrainer.sampleCurrentProcessUsage() else {
            return
        }
        if let prev = lastUsageSample {
            let wallDeltaS = sample.timestamp.timeIntervalSince(prev.timestamp)
            let maxUsefulGapS = Self.usageStatsRefreshSec * 3
            if wallDeltaS > 0 && wallDeltaS <= maxUsefulGapS {
                let wallDeltaNs = wallDeltaS * 1_000_000_000
                let cpuDeltaNs = sample.cpuNs >= prev.cpuNs
                    ? Double(sample.cpuNs - prev.cpuNs)
                    : 0
                let gpuDeltaNs = sample.gpuNs >= prev.gpuNs
                    ? Double(sample.gpuNs - prev.gpuNs)
                    : 0
                cpuPercent = cpuDeltaNs / wallDeltaNs * 100
                gpuPercent = gpuDeltaNs / wallDeltaNs * 100
            }
        }
        lastUsageSample = sample
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
        // Tabular figures so the elapsed timer and memory sizes
        // don't jitter as digits roll. `monospacedDigit()` keeps
        // letters in the normal proportional font while forcing
        // digits to a fixed cell width — less jarring than
        // switching the whole label to a monospaced face.
        return Text(busyLabel)
            .foregroundStyle(.secondary)
            .monospacedDigit()
    }

    private var busyLabel: String {
        if isBuilding { return "Building network..." }
        if realTraining {
            // Play and Train (no arena): show total session time
            // and a memory-usage line. The detailed self-play and
            // training rates that used to live here have moved
            // into the Self Play and Training panels below; the
            // top row's job now is "how long has this session
            // been running, and is memory healthy."
            //
            // The session time uses wall clock since
            // `parallelStats.sessionStart`, formatted as HH:MM:SS
            // for legibility (this is the same denominator the
            // panels use, so the user can correlate). The memory
            // stats are sampled out-of-band by
            // `refreshMemoryStatsIfNeeded()` at ~10 s intervals
            // and read here from `memoryStatsSnap`. Missing
            // (nil) memory snapshot just renders without that
            // section so the line still shows session time
            // immediately on startup.
            let timeStr: String
            if let ps = parallelStats {
                let elapsed = max(0, Date().timeIntervalSince(ps.sessionStart))
                timeStr = "Total session time: \(GameWatcher.Snapshot.formatHMS(seconds: elapsed))"
            } else {
                timeStr = "Total session time: --"
            }
            let memLine: String
            if let mem = memoryStatsSnap {
                let appGB = Self.bytesToGB(mem.appFootprintBytes)
                let gpuGB = Self.bytesToGB(mem.gpuAllocatedBytes)
                let gpuMaxGB = Self.bytesToGB(mem.gpuMaxTargetBytes)
                let gpuTotalGB = Self.bytesToGB(mem.gpuTotalBytes)
                let gpuPct = mem.gpuMaxTargetBytes > 0
                    ? Int((Double(mem.gpuAllocatedBytes) / Double(mem.gpuMaxTargetBytes) * 100).rounded())
                    : 0
                memLine = String(
                    format: "%@  ·  App: %.2f GB  ·  GPU RAM: %.2f / %.2f GB (%d%%)  ·  Total: %.1f GB",
                    timeStr, appGB, gpuGB, gpuMaxGB, gpuPct, gpuTotalGB
                )
            } else {
                memLine = timeStr
            }
            // Second line: %CPU and %GPU time utilisation since the
            // previous usage sample. "GPU" here is the GPU-time
            // percentage (different from "GPU RAM" on line 1, which
            // is the unified-memory working set). `%4.0f%%` pads to
            // 5 characters so values from "   0%" through "1600%"
            // align under the monospacedDigit() renderer.
            let cpuStr = cpuPercent.map { String(format: "%4.0f%%", $0) } ?? "  --%"
            let gpuUsageStr = gpuPercent.map { String(format: "%4.0f%%", $0) } ?? "  --%"
            let usageLine = "CPU: \(cpuStr)  ·  GPU: \(gpuUsageStr)"
            return "\(memLine)\n\(usageLine)"
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

    // MARK: - Checkpoint save / load handlers

    /// Publish a user-visible status line in the checkpoint row
    /// and clear it after a few seconds. Safe to call repeatedly
    /// — the latest message wins.
    @MainActor
    private func setCheckpointStatus(_ message: String, isError: Bool) {
        checkpointStatusMessage = message
        checkpointStatusIsError = isError
        // Auto-clear after 6 seconds. Grabs the current message at
        // schedule time so a later message isn't wiped out by an
        // earlier one's timer.
        let snapshotMessage = message
        Task {
            try? await Task.sleep(for: .seconds(isError ? 12 : 6))
            if checkpointStatusMessage == snapshotMessage {
                checkpointStatusMessage = nil
                checkpointStatusIsError = false
            }
        }
    }

    /// Build the Codable snapshot of the current session state,
    /// including counters, hyperparameters, and arena history.
    /// Called at save time with the live state read off the main
    /// actor. `championIDOverride` / `trainerIDOverride` let the
    /// caller inject specific IDs when the on-disk identity should
    /// differ from the live network identifiers (not currently used
    /// but kept for future "rename on save" flows).
    @MainActor
    private func buildCurrentSessionState(
        championID: String,
        trainerID: String
    ) -> SessionCheckpointState {
        let now = Date()
        let sessionStart = currentSessionStart ?? (parallelStats?.sessionStart ?? now)
        let elapsedSec = max(0, now.timeIntervalSince(sessionStart))
        let snap = parallelStats
        let trainingSnap = trainingStats
        let history = tournamentHistory.map { record in
            ArenaHistoryEntryCodable(
                finishedAtStep: record.finishedAtStep,
                candidateWins: record.candidateWins,
                championWins: record.championWins,
                draws: record.draws,
                score: record.score,
                promoted: record.promoted,
                promotedID: record.promotedID?.description,
                durationSec: record.durationSec
            )
        }
        let lr = trainer?.learningRate ?? Self.trainerLearningRateDefault
        return SessionCheckpointState(
            formatVersion: SessionCheckpointState.currentFormatVersion,
            sessionID: currentSessionID ?? "unknown-session",
            savedAtUnix: Int64(now.timeIntervalSince1970),
            sessionStartUnix: Int64(sessionStart.timeIntervalSince1970),
            elapsedTrainingSec: elapsedSec,
            trainingSteps: trainingSnap?.steps ?? 0,
            selfPlayGames: snap?.selfPlayGames ?? 0,
            selfPlayMoves: snap?.selfPlayPositions ?? 0,
            trainingPositionsSeen: (trainingSnap?.steps ?? 0) * Self.trainingBatchSize,
            batchSize: Self.trainingBatchSize,
            learningRate: lr,
            promoteThreshold: Self.tournamentPromoteThreshold,
            arenaGames: Self.tournamentGames,
            selfPlayTau: TauConfigCodable(SamplingSchedule.selfPlay),
            arenaTau: TauConfigCodable(SamplingSchedule.arena),
            selfPlayWorkerCount: selfPlayWorkerCount,
            replayRatioTarget: replayRatioTarget,
            replayRatioAutoAdjust: replayRatioAutoAdjust,
            championID: championID,
            trainerID: trainerID,
            arenaHistory: history
        )
    }

    /// Manual "Save Champion as Model" — writes a standalone
    /// `.dcmmodel` containing the current champion's weights.
    /// If Play-and-Train is active, pauses self-play worker 0
    /// briefly so the export doesn't race with in-flight
    /// inference calls on the shared champion graph, then
    /// resumes. Uses `pauseAndWait(timeoutMs:)` so a
    /// mid-save session end can't deadlock the save task.
    private func handleSaveChampionAsModel() {
        guard let champion = network else {
            setCheckpointStatus("No network to save", isError: true)
            return
        }
        let championID = champion.identifier?.description ?? "unknown"
        // Snapshot the active self-play gate up front. If there
        // is no active session, we can safely export directly —
        // nobody is racing against us.
        let gate = activeSelfPlayGate
        checkpointSaveInFlight = true
        setCheckpointStatus("Saving champion…", isError: false)

        Task {
            // Pause worker 0 if a session is running. Bail with a
            // user-visible error on timeout (indicates the session
            // has already ended or the worker is stuck — either way
            // we shouldn't spin forever).
            if let gate {
                let acquired = await gate.pauseAndWait(timeoutMs: Self.saveGateTimeoutMs)
                if !acquired {
                    checkpointSaveInFlight = false
                    setCheckpointStatus("Save aborted: could not pause self-play (timeout)", isError: true)
                    return
                }
            }

            var championWeights: [[Float]] = []
            var exportError: Error?
            do {
                championWeights = try await Task.detached(priority: .userInitiated) {
                    try champion.network.exportWeights()
                }.value
            } catch {
                exportError = error
            }
            gate?.resume()

            if let exportError {
                checkpointSaveInFlight = false
                setCheckpointStatus("Save failed (export): \(exportError.localizedDescription)", isError: true)
                SessionLogger.shared.log("[CHECKPOINT] Save champion export failed: \(exportError.localizedDescription)")
                return
            }

            let metadata = ModelCheckpointMetadata(
                creator: "manual",
                trainingStep: trainingStats?.steps,
                parentModelID: "",
                notes: "Manual Save Champion export"
            )
            let createdAtUnix = Int64(Date().timeIntervalSince1970)

            let outcome: Result<URL, Error> = await Task.detached(priority: .userInitiated) {
                do {
                    let url = try CheckpointManager.saveModel(
                        weights: championWeights,
                        modelID: championID,
                        createdAtUnix: createdAtUnix,
                        metadata: metadata,
                        trigger: "manual"
                    )
                    return .success(url)
                } catch {
                    return .failure(error)
                }
            }.value
            checkpointSaveInFlight = false
            switch outcome {
            case .success(let url):
                setCheckpointStatus("Saved \(url.lastPathComponent)", isError: false)
                SessionLogger.shared.log("[CHECKPOINT] Saved champion: \(url.lastPathComponent)")
            case .failure(let error):
                setCheckpointStatus("Save failed: \(error.localizedDescription)", isError: true)
                SessionLogger.shared.log("[CHECKPOINT] Save champion failed: \(error.localizedDescription)")
            }
        }
    }

    /// Upper bound on how long a save path will wait for a
    /// worker to acknowledge a pause request. Has to cover one
    /// in-flight self-play game or training step, so 15 s is a
    /// comfortable margin above the worst-case game length at
    /// typical self-play rates. On timeout the save bails with
    /// a user-visible error rather than blocking forever.
    nonisolated static let saveGateTimeoutMs: Int = 15_000

    /// Manual "Save Session" — writes a full `.dcmsession` with
    /// champion and trainer model files plus `session.json`.
    /// Requires an active Play-and-Train session and an available
    /// trainer. Briefly pauses both self-play worker 0 and the
    /// training gate to snapshot the two networks' weights.
    private func handleSaveSessionManual() {
        guard realTraining,
              let champion = network,
              let trainer,
              let selfPlayGate = activeSelfPlayGate,
              let trainingGate = activeTrainingGate else {
            setCheckpointStatus("No active session to save", isError: true)
            return
        }
        saveSessionInternal(
            champion: champion,
            trainer: trainer,
            selfPlayGate: selfPlayGate,
            trainingGate: trainingGate,
            trigger: "manual"
        )
    }

    /// Shared save-session internal used by both the manual save
    /// button and the post-promotion autosave hook. Handles the
    /// gate dance, exports both networks, builds the session
    /// state on the main actor, and fires off the actual write to
    /// a detached task.
    private func saveSessionInternal(
        champion: ChessMPSNetwork,
        trainer: ChessTrainer,
        selfPlayGate: WorkerPauseGate,
        trainingGate: WorkerPauseGate,
        trigger: String
    ) {
        let championID = champion.identifier?.description ?? "unknown"
        let trainerID = trainer.identifier?.description ?? "unknown"
        checkpointSaveInFlight = true
        setCheckpointStatus("Saving session (\(trigger))…", isError: false)

        // Build the state snapshot on the main actor before
        // jumping to detached work.
        let sessionState = buildCurrentSessionState(
            championID: championID,
            trainerID: trainerID
        )
        let trainingStep = trainingStats?.steps ?? 0

        Task {
            // Pause self-play briefly so the champion export is
            // race-free, snapshot weights, then resume. Uses the
            // bounded variant so a session end mid-save doesn't
            // spin forever waiting for workers that have exited.
            let selfPlayAcquired = await selfPlayGate.pauseAndWait(timeoutMs: Self.saveGateTimeoutMs)
            guard selfPlayAcquired else {
                checkpointSaveInFlight = false
                setCheckpointStatus("Save aborted: could not pause self-play (timeout)", isError: true)
                SessionLogger.shared.log("[CHECKPOINT] Save session aborted at self-play pause timeout")
                return
            }
            var championWeights: [[Float]] = []
            var championError: Error?
            do {
                championWeights = try await Task.detached(priority: .userInitiated) {
                    try champion.network.exportWeights()
                }.value
            } catch {
                championError = error
            }
            selfPlayGate.resume()

            if let championError {
                checkpointSaveInFlight = false
                setCheckpointStatus("Save failed (champion export): \(championError.localizedDescription)", isError: true)
                SessionLogger.shared.log("[CHECKPOINT] Save session failed at champion export: \(championError.localizedDescription)")
                return
            }

            // Pause training briefly to snapshot trainer weights.
            let trainingAcquired = await trainingGate.pauseAndWait(timeoutMs: Self.saveGateTimeoutMs)
            guard trainingAcquired else {
                checkpointSaveInFlight = false
                setCheckpointStatus("Save aborted: could not pause training (timeout)", isError: true)
                SessionLogger.shared.log("[CHECKPOINT] Save session aborted at training pause timeout")
                return
            }
            var trainerWeights: [[Float]] = []
            var trainerError: Error?
            do {
                trainerWeights = try await Task.detached(priority: .userInitiated) {
                    try trainer.network.exportWeights()
                }.value
            } catch {
                trainerError = error
            }
            trainingGate.resume()

            if let trainerError {
                checkpointSaveInFlight = false
                setCheckpointStatus("Save failed (trainer export): \(trainerError.localizedDescription)", isError: true)
                SessionLogger.shared.log("[CHECKPOINT] Save session failed at trainer export: \(trainerError.localizedDescription)")
                return
            }

            // Final write + verify on a detached task so UI stays
            // responsive during the ~150 ms scratch-network build.
            let championMetadata = ModelCheckpointMetadata(
                creator: trigger,
                trainingStep: trainingStep,
                parentModelID: "",
                notes: "Session checkpoint (\(trigger))"
            )
            let trainerMetadata = ModelCheckpointMetadata(
                creator: trigger,
                trainingStep: trainingStep,
                parentModelID: championID,
                notes: "Trainer lineage at session checkpoint (\(trigger))"
            )
            let now = Int64(Date().timeIntervalSince1970)
            let outcome: Result<URL, Error> = await Task.detached(priority: .userInitiated) {
                do {
                    let url = try CheckpointManager.saveSession(
                        championWeights: championWeights,
                        championID: championID,
                        championMetadata: championMetadata,
                        championCreatedAtUnix: now,
                        trainerWeights: trainerWeights,
                        trainerID: trainerID,
                        trainerMetadata: trainerMetadata,
                        trainerCreatedAtUnix: now,
                        state: sessionState,
                        trigger: trigger
                    )
                    return .success(url)
                } catch {
                    return .failure(error)
                }
            }.value

            checkpointSaveInFlight = false
            switch outcome {
            case .success(let url):
                setCheckpointStatus("Saved \(url.lastPathComponent)", isError: false)
                SessionLogger.shared.log("[CHECKPOINT] Saved session: \(url.lastPathComponent)")
            case .failure(let error):
                setCheckpointStatus("Save failed: \(error.localizedDescription)", isError: true)
                SessionLogger.shared.log("[CHECKPOINT] Save session failed: \(error.localizedDescription)")
            }
        }
    }

    /// Load a standalone `.dcmmodel` into the current champion
    /// network. Triggered from the Load Model file importer. The
    /// network must exist (loading into a built network preserves
    /// the existing graph compilation; we don't rebuild).
    private func handleLoadModelPickResult(_ result: Result<[URL], Error>) {
        switch result {
        case .failure(let error):
            setCheckpointStatus("Load cancelled: \(error.localizedDescription)", isError: true)
        case .success(let urls):
            guard let url = urls.first else { return }
            loadModelFrom(url: url)
        }
    }

    private func loadModelFrom(url: URL) {
        guard let champion = network else {
            setCheckpointStatus("Build the network first before loading weights", isError: true)
            return
        }

        checkpointSaveInFlight = true
        setCheckpointStatus("Loading \(url.lastPathComponent)…", isError: false)

        Task {
            // Keep the security scope open across the entire
            // detached read+load so files picked from outside the
            // sandbox (Downloads, AirDrop, external volumes) stay
            // accessible until the work finishes. Start/stop must
            // happen inside the detached closure to bracket the
            // actual I/O.
            let outcome: Result<ModelCheckpointFile, Error> = await Task.detached(priority: .userInitiated) {
                let scopeAccessed = url.startAccessingSecurityScopedResource()
                defer {
                    if scopeAccessed {
                        url.stopAccessingSecurityScopedResource()
                    }
                }
                do {
                    let file = try CheckpointManager.loadModelFile(at: url)
                    try champion.network.loadWeights(file.weights)
                    return .success(file)
                } catch {
                    return .failure(error)
                }
            }.value
            checkpointSaveInFlight = false
            switch outcome {
            case .success(let file):
                champion.identifier = ModelID(value: file.modelID)
                networkStatus = "Loaded model \(file.modelID)\nFrom: \(url.lastPathComponent)"
                setCheckpointStatus("Loaded \(file.modelID)", isError: false)
                SessionLogger.shared.log("[CHECKPOINT] Loaded model: \(url.lastPathComponent) → \(file.modelID)")
                inferenceResult = nil
            case .failure(let error):
                setCheckpointStatus("Load failed: \(error.localizedDescription)", isError: true)
                SessionLogger.shared.log("[CHECKPOINT] Load model failed: \(error.localizedDescription)")
            }
        }
    }

    /// Load a `.dcmsession` directory. Parses everything, loads
    /// champion weights immediately into the live champion
    /// network, and stores the session state + trainer weights
    /// in `pendingLoadedSession` so the next Play-and-Train start
    /// resumes from them.
    private func handleLoadSessionPickResult(_ result: Result<[URL], Error>) {
        switch result {
        case .failure(let error):
            setCheckpointStatus("Load cancelled: \(error.localizedDescription)", isError: true)
        case .success(let urls):
            guard let url = urls.first else { return }
            loadSessionFrom(url: url)
        }
    }

    private func loadSessionFrom(url: URL) {
        guard let champion = network else {
            setCheckpointStatus("Build the network first before loading a session", isError: true)
            return
        }

        checkpointSaveInFlight = true
        setCheckpointStatus("Loading session \(url.lastPathComponent)…", isError: false)

        Task {
            let outcome: Result<LoadedSession, Error> = await Task.detached(priority: .userInitiated) {
                let scopeAccessed = url.startAccessingSecurityScopedResource()
                defer {
                    if scopeAccessed {
                        url.stopAccessingSecurityScopedResource()
                    }
                }
                do {
                    let loaded = try CheckpointManager.loadSession(at: url)
                    // Apply champion weights immediately; trainer
                    // weights are held for the next startRealTraining.
                    try champion.network.loadWeights(loaded.championFile.weights)
                    return .success(loaded)
                } catch {
                    return .failure(error)
                }
            }.value
            checkpointSaveInFlight = false
            switch outcome {
            case .success(let loaded):
                champion.identifier = ModelID(value: loaded.championFile.modelID)
                pendingLoadedSession = loaded
                networkStatus = """
                    Loaded session \(loaded.state.sessionID)
                    Champion: \(loaded.championFile.modelID)
                    Trainer: \(loaded.trainerFile.modelID)
                    Steps: \(loaded.state.trainingSteps) / Games: \(loaded.state.selfPlayGames)
                    Click Play and Train to resume.
                    """
                setCheckpointStatus("Loaded session \(loaded.state.sessionID) — click Play and Train to resume", isError: false)
                SessionLogger.shared.log("[CHECKPOINT] Loaded session: \(url.lastPathComponent)")
                inferenceResult = nil
            case .failure(let error):
                setCheckpointStatus("Load failed: \(error.localizedDescription)", isError: true)
                SessionLogger.shared.log("[CHECKPOINT] Load session failed: \(error.localizedDescription)")
            }
        }
    }

    /// Open Finder pointed at the checkpoint root so the user can
    /// browse saved sessions and models even though Application
    /// Support is hidden by default. Creates the folder if it
    /// doesn't exist yet so the button always works.
    private func handleRevealSaves() {
        do {
            try CheckpointPaths.ensureDirectories()
        } catch {
            setCheckpointStatus("Could not create save folder: \(error.localizedDescription)", isError: true)
            return
        }
        CheckpointManager.revealInFinder(CheckpointPaths.rootURL)
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
        secondarySelfPlayNetworks: [ChessMPSNetwork],
        secondarySelfPlayGates: [WorkerPauseGate],
        tBox: TournamentLiveBox,
        selfPlayGate: WorkerPauseGate,
        trainingGate: WorkerPauseGate,
        arenaFlag: ArenaActiveFlag,
        overrideBox: ArenaOverrideBox
    ) async {
        // Clear any stale decision from a previous tournament so this
        // run starts with a clean override slate. Normal completion
        // `consume()`s the box at the end, but early-return paths
        // (cancellation, sync errors) don't — clearing here keeps
        // all exit paths honest.
        _ = overrideBox.consume()
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
        // Capture trainer weights here and hold them through the
        // rest of the arena. At arena end we use them to autosave
        // the session without needing another training pause
        // (and therefore without any gate interaction from the
        // autosave task, which is critical to avoid deadlocking
        // a save that runs past a session cancel — unstructured
        // save tasks don't inherit realTrainingTask cancellation).
        var trainerSnapshotWeights: [[Float]] = []
        do {
            trainerSnapshotWeights = try await Task.detached(priority: .userInitiated) {
                let weights = try trainer.network.exportWeights()
                try candidateInference.network.loadWeights(weights)
                return weights
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
                [arenaChampion, candidateInference, tBox, cancelBox, overrideBox] in
                let driver = TournamentDriver()
                driver.delegate = nil
                return await driver.run(
                    playerA: { MPSChessPlayer(name: "Candidate", network: candidateInference, schedule: .arena) },
                    playerB: { MPSChessPlayer(name: "Champion", network: arenaChampion, schedule: .arena) },
                    games: totalGames,
                    // The driver checks this between games. Either a
                    // task-cancel (session Stop) or a user Abort /
                    // Promote click breaks the game loop early; the
                    // caller below disambiguates the two via the
                    // override box's `consume()`.
                    isCancelled: { cancelBox.isCancelled || overrideBox.isActive },
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
        //
        // Branch on the user override first. `.abort` ends the
        // tournament with no promotion regardless of score; `.promote`
        // forces promotion regardless of score and games played; `nil`
        // is the normal path where the usual score-threshold check
        // decides. The consume also clears the box for the next
        // tournament.
        let overrideDecision = overrideBox.consume()
        let playedGames = stats.gamesPlayed
        let score: Double
        if playedGames > 0 {
            score = (Double(stats.playerAWins) + 0.5 * Double(stats.draws)) / Double(playedGames)
        } else {
            score = 0
        }
        var promoted = false
        var promotedID: ModelID?
        let shouldPromote: Bool
        switch overrideDecision {
        case .abort:
            shouldPromote = false
        case .promote:
            shouldPromote = true
        case .none:
            shouldPromote = playedGames >= totalGames && score >= Self.tournamentPromoteThreshold
        }
        // Holds the new champion weights if promotion succeeds,
        // so we can hand them to a detached autosave task at the
        // end without needing to re-read them from the live
        // network (which would race against self-play again).
        var promotedChampionWeights: [[Float]] = []
        if shouldPromote {
            // Pause every self-play worker briefly, copy candidate
            // inference into the champion AND every secondary
            // self-play network, release. Each worker's next game
            // uses the new weights. All workers must be paused
            // before the loadWeights calls because loadWeights
            // mutates MPSGraph variable state in place, and any
            // concurrent `evaluate` on the same network would race
            // the assign ops. Pause worker 0 first, then the
            // secondaries; resume in reverse so the primary is
            // always the last one released (mirrors the
            // "worker 0 leads" pattern used elsewhere for UI
            // ownership).
            await selfPlayGate.pauseAndWait()
            for gate in secondarySelfPlayGates {
                await gate.pauseAndWait()
            }
            if !Task.isCancelled {
                do {
                    promotedChampionWeights = try await Task.detached(priority: .userInitiated) {
                        [candidateInference, champion, secondarySelfPlayNetworks] in
                        let weights = try candidateInference.network.exportWeights()
                        try champion.network.loadWeights(weights)
                        for net in secondarySelfPlayNetworks {
                            try net.network.loadWeights(weights)
                        }
                        return weights
                    }.value
                    // Promoted: champion now holds the arena candidate's
                    // exact weights, so it inherits that snapshot ID.
                    // Secondaries are pure mirrors of the champion and
                    // don't carry their own identity.
                    champion.identifier = candidateInference.identifier
                    promoted = true
                    promotedID = candidateInference.identifier
                } catch {
                    trainingBox?.recordError("Promotion copy failed: \(error.localizedDescription)")
                }
            }
            for gate in secondarySelfPlayGates.reversed() {
                gate.resume()
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
            promotedID: promotedID,
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

        // Post-promotion autosave. Fires a detached Task that
        // writes a full session snapshot using the weights we
        // already captured above under the arena-start training
        // pause and the promotion self-play pause. The detached
        // task touches no live networks and no pause gates, so
        // it is safe to run past a session cancel — unstructured
        // save tasks don't inherit `realTrainingTask`
        // cancellation, and any post-return gate interaction
        // here would potentially deadlock against workers that
        // have already exited their loops.
        if promoted && Self.autosaveSessionsOnPromote && !promotedChampionWeights.isEmpty {
            let championID = champion.identifier?.description ?? "unknown"
            let trainerID = trainer.identifier?.description ?? "unknown"
            let sessionState = buildCurrentSessionState(
                championID: championID,
                trainerID: trainerID
            )
            let championMetadata = ModelCheckpointMetadata(
                creator: "promote",
                trainingStep: trainingStats?.steps ?? 0,
                parentModelID: "",
                notes: "Post-arena autosave after promotion"
            )
            let trainerMetadata = ModelCheckpointMetadata(
                creator: "promote",
                trainingStep: trainingStats?.steps ?? 0,
                parentModelID: championID,
                notes: "Trainer lineage at arena-start pause"
            )
            let createdAtUnix = Int64(Date().timeIntervalSince1970)
            // Copy captured arrays for clean Sendable semantics
            // (they're already Sendable but this makes the
            // transfer to the detached task explicit).
            let championWeightsSnapshot = promotedChampionWeights
            let trainerWeightsSnapshot = trainerSnapshotWeights

            // Fire-and-forget detached task. The closure captures
            // only Sendable value types (weight arrays, metadata
            // structs, the session state snapshot, and a few
            // strings) — explicitly NOT `self` — so we can safely
            // run past a session end without touching any View
            // @State. Outcome is audited through SessionLogger,
            // which is NSLock-guarded and thread-safe. The user
            // sees success via "Reveal Saves" finding the
            // timestamped folder; failures show up in the
            // session log.
            Task.detached(priority: .utility) {
                do {
                    let url = try CheckpointManager.saveSession(
                        championWeights: championWeightsSnapshot,
                        championID: championID,
                        championMetadata: championMetadata,
                        championCreatedAtUnix: createdAtUnix,
                        trainerWeights: trainerWeightsSnapshot,
                        trainerID: trainerID,
                        trainerMetadata: trainerMetadata,
                        trainerCreatedAtUnix: createdAtUnix,
                        state: sessionState,
                        trigger: "promote"
                    )
                    SessionLogger.shared.log("[CHECKPOINT] Autosaved session: \(url.lastPathComponent)")
                } catch {
                    SessionLogger.shared.log("[CHECKPOINT] Autosave session failed: \(error.localizedDescription)")
                }
            }
        }
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
        // Promotion marker carries the promoted model's ID inline so
        // the single-line header identifies exactly which checkpoint
        // just took over as the champion — the same ID that already
        // shows up in the trailing `ids` line, pulled forward so a
        // `grep PROMOTED` on the session log is self-contained.
        let statusStr: String
        if record.promoted, let pid = record.promotedID {
            statusStr = "PROMOTED=\(pid.description)"
        } else if record.promoted {
            statusStr = "PROMOTED"
        } else {
            statusStr = "kept"
        }
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
            let t = try ChessTrainer(learningRate: Float(trainerLearningRate))
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
        precondition(
            Self.absoluteMaxSelfPlayWorkers >= 1,
            "absoluteMaxSelfPlayWorkers must be >= 1; got \(Self.absoluteMaxSelfPlayWorkers)"
        )
        // Snap the live N into the [1, absoluteMaxSelfPlayWorkers] range
        // before doing anything else. The Stepper enforces this
        // for user input but `selfPlayWorkerCount` is plain @State
        // so the value could in principle be edited elsewhere.
        let initialWorkerCount = max(1, min(Self.absoluteMaxSelfPlayWorkers, selfPlayWorkerCount))
        if initialWorkerCount != selfPlayWorkerCount {
            selfPlayWorkerCount = initialWorkerCount
        }
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
        learningRateEditText = String(format: "%.1e", trainer.learningRate)
        tournamentHistory = []
        tournamentProgress = nil
        let tBox = TournamentLiveBox()
        tournamentBox = tBox
        let pStatsBox = ParallelWorkerStatsBox(sessionStart: Date())
        parallelWorkerStatsBox = pStatsBox
        parallelStats = pStatsBox.snapshot()
        // Reset progress-rate sampler state so the new session's
        // chart starts fresh at t=0. Leaving old samples in place
        // would show up as a visible "step" from the previous
        // session's trailing values to the new session's zero
        // reading.
        progressRateSamples = []
        progressRateLastFetch = .distantPast
        progressRateNextId = 0
        progressRateScrollX = 0
        progressRateFollowLatest = true
        let selfPlayGate = WorkerPauseGate()
        // One pause gate per secondary self-play worker (workers
        // 1..absoluteMaxSelfPlayWorkers-1). Worker 0 uses `selfPlayGate`.
        // We size to the hard maximum (not the current N) because
        // every potentially-active worker is spawned at session
        // start and idles in its own gate's wait state until the
        // user grows N enough to include it. Each secondary worker
        // polls its own gate, so the arena coordinator can still
        // pause exactly the workers whose networks a given sync
        // point touches. Promotion pauses all of them; the
        // champion → arena-champion snapshot only pauses worker 0
        // because secondary workers don't touch `network`.
        let secondarySelfPlayGates: [WorkerPauseGate] = (0..<max(0, Self.absoluteMaxSelfPlayWorkers - 1))
            .map { _ in WorkerPauseGate() }
        // Shared current-N holder. Workers poll this to decide
        // whether to play another game or sit in their idle wait.
        // The Stepper writes through it (and to `@State
        // selfPlayWorkerCount` simultaneously). Exposed via @State
        // so the UI can disable the buttons when the box is gone
        // (between sessions).
        let countBox = WorkerCountBox(initial: initialWorkerCount)
        workerCountBox = countBox
        // Shared live delay holder. The Stepper writes through to
        // both `@State trainingStepDelayMs` and this box; the
        // training worker reads from the box at the bottom of each
        // step to decide how long to pause.
        let stepDelayBox = TrainingStepDelayBox(initial: trainingStepDelayMs)
        trainingStepDelayBox = stepDelayBox
        let ratioController = ReplayRatioController(
            batchSize: Self.trainingBatchSize,
            targetRatio: replayRatioTarget,
            autoAdjust: replayRatioAutoAdjust,
            initialDelayMs: trainingStepDelayMs,
            maxDelayMs: Self.stepDelayMaxMs
        )
        replayRatioController = ratioController
        let trainingGate = WorkerPauseGate()
        let arenaFlag = ArenaActiveFlag()
        arenaActiveFlag = arenaFlag
        let triggerBox = ArenaTriggerBox()
        arenaTriggerBox = triggerBox
        let overrideBox = ArenaOverrideBox()
        arenaOverrideBox = overrideBox
        isArenaRunning = false
        realTraining = true

        // Expose the two gates the checkpoint save path needs and
        // anchor the session ID + wall clock. `currentSessionID`
        // is either a fresh mint or the loaded session's ID when
        // resuming. `currentSessionStart` is back-dated on
        // resume by the loaded session's `elapsedTrainingSec`, so
        // successive save-resume-save cycles accumulate elapsed
        // time monotonically. This anchor is only read by the
        // save path (`buildCurrentSessionState`) — the parallel
        // worker stats box keeps its own fresh `sessionStart`
        // anchor for rate-display purposes, so games/hr doesn't
        // get polluted by the back-dated hours.
        activeSelfPlayGate = selfPlayGate
        activeTrainingGate = trainingGate
        if let resumed = pendingLoadedSession {
            currentSessionID = resumed.state.sessionID
            currentSessionStart = Date().addingTimeInterval(-resumed.state.elapsedTrainingSec)
            replayRatioTarget = resumed.state.replayRatioTarget ?? 1.0
            replayRatioAutoAdjust = resumed.state.replayRatioAutoAdjust ?? true
        } else {
            currentSessionID = ModelIDMinter.mint().value
            currentSessionStart = Date()
        }

        realTrainingTask = Task(priority: .userInitiated) {
            [trainer, network, buffer, box, tBox, pStatsBox,
             selfPlayGate, secondarySelfPlayGates, trainingGate, arenaFlag, triggerBox, overrideBox, countBox] in

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

            // Build secondary self-play networks up to the hard
            // maximum. Worker 0 uses `network`, so we need
            // `absoluteMaxSelfPlayWorkers - 1` extras. The array is
            // @State so it persists across sessions — if a prior
            // session already built enough, we reuse them. We
            // pre-build to the *max*, not the current N, because
            // every worker is spawned at session start and idles
            // until the user grows N enough to include it. That
            // means the first session pays the full one-time
            // build cost, and subsequent sessions are instant.
            let neededSecondaryCount = max(0, Self.absoluteMaxSelfPlayWorkers - 1)
            let existingSecondaryCount = await MainActor.run { secondarySelfPlayNetworks.count }
            if existingSecondaryCount < neededSecondaryCount {
                let toBuild = neededSecondaryCount - existingSecondaryCount
                do {
                    let built = try await Task.detached(priority: .userInitiated) {
                        var out: [ChessMPSNetwork] = []
                        out.reserveCapacity(toBuild)
                        for _ in 0..<toBuild {
                            out.append(try ChessMPSNetwork(.randomWeights))
                        }
                        return out
                    }.value
                    await MainActor.run {
                        secondarySelfPlayNetworks.append(contentsOf: built)
                    }
                } catch {
                    box.recordError("Secondary self-play network init failed: \(error.localizedDescription)")
                    await MainActor.run {
                        realTraining = false
                        realTrainingTask = nil
                    }
                    return
                }
            }

            // Grab the secondary self-play networks for this
            // session — `absoluteMaxSelfPlayWorkers - 1` of them, one per
            // potential worker beyond worker 0. Workers above the
            // currently active count idle until the user grows N
            // via the Stepper. The pause-gate array captured
            // above already carries a matching count.
            let secondaries: [ChessMPSNetwork] = await MainActor.run {
                Array(secondarySelfPlayNetworks.prefix(max(0, Self.absoluteMaxSelfPlayWorkers - 1)))
            }
            guard secondaries.count == max(0, Self.absoluteMaxSelfPlayWorkers - 1) else {
                box.recordError("Secondary self-play networks missing after setup")
                await MainActor.run {
                    realTraining = false
                    realTrainingTask = nil
                }
                return
            }

            // Reset the trainer's graph AND initialize its weights.
            // Two paths:
            //
            // (1) Normal start: copy champion weights into the trainer
            //     and into every secondary self-play network. This
            //     makes arena-at-step-0 a fair tie by construction and
            //     establishes the trainer's starting point as a true
            //     fork of the champion.
            //
            // (2) Resume from a loaded `.dcmsession`: copy champion
            //     weights into the secondaries (the champion has
            //     already been loaded from disk into `network` at
            //     file-load time), but load the trainer from the
            //     session's `trainer.dcmmodel` payload so its
            //     mid-training divergence from the champion is
            //     preserved. Without this branch, resume would
            //     throw away the trainer's in-flight SGD progress
            //     every time.
            let resumedTrainerWeights: [[Float]]? = await MainActor.run {
                pendingLoadedSession?.trainerFile.weights
            }
            do {
                try await Task.detached(priority: .userInitiated) {
                    [secondaries, resumedTrainerWeights] in
                    try trainer.resetNetwork()
                    let championWeights = try network.network.exportWeights()
                    if let trainerWeights = resumedTrainerWeights {
                        try trainer.network.loadWeights(trainerWeights)
                    } else {
                        try trainer.network.loadWeights(championWeights)
                    }
                    for net in secondaries {
                        try net.network.loadWeights(championWeights)
                    }
                }.value
            } catch {
                box.recordError("Reset failed: \(error.localizedDescription)")
                await MainActor.run {
                    realTraining = false
                    realTrainingTask = nil
                }
                return
            }

            // Trainer ID: on a fresh start, mint a new one. On a
            // resume, inherit the trainer ID from the loaded session
            // so the audit trail stays continuous.
            await MainActor.run {
                if let resumed = pendingLoadedSession {
                    trainer.identifier = ModelID(value: resumed.trainerFile.modelID)
                } else {
                    trainer.identifier = ModelIDMinter.mint()
                }
                // Consume the pending load — from here on, the
                // running session owns the restored state.
                pendingLoadedSession = nil
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

            // --- Spawn the worker tasks ---
            //
            // absoluteMaxSelfPlayWorkers self-play tasks, one training
            // worker, one arena coordinator, one session-log
            // ticker. The Stepper picks how many of the self-play
            // tasks are *active* at any moment — the rest sit in
            // their pause gate's wait state until the user raises
            // N enough to include them.

            // Anchor the session wall-clock to *now*, after all the
            // synchronous and MainActor-hop setup above has finished.
            // Rate denominators ("steps/sec", "games/hr", "avg move
            // ms", "Total session time", ...) are computed as `Date()
            // - sessionStart`, so leaving the original `Date()`-at-
            // button-press anchor in place would bake the setup
            // delay into every average for the life of the session.
            pStatsBox.markWorkersStarted()

            await withTaskGroup(of: Void.self) { group in
                // Self-play workers: each plays one game at a time
                // on its own dedicated inference network, streams
                // positions into the shared replay buffer, and
                // polls (a) its own `WorkerPauseGate` for arena
                // coordination and (b) the shared `WorkerCountBox`
                // for live N adjustment. Workers above the current
                // active count idle in their pause-wait state;
                // workers within the count play games normally.
                //
                // Worker 0 uses `network` (the champion, which also
                // serves as the arena-champion snapshot source);
                // workers 1..absoluteMaxSelfPlayWorkers-1 use dedicated
                // secondary inference networks mirrored from the
                // champion at session start and at every promotion.
                //
                // Each worker allocates its two `MPSChessPlayer`
                // instances once before the loop and reuses them
                // across every game — `ChessMachine.beginNewGame`
                // calls `onNewGame` internally, which resets
                // per-game scratch state while keeping backing
                // storage alive (see `MPSChessPlayer.onNewGame`).
                //
                // Worker 0 is always the one wired to `GameWatcher`
                // for the live-game animated board. The board view
                // is hidden when `selfPlayWorkerCount > 1`, but we
                // keep feeding GameWatcher so dropping back to N=1
                // restores a valid live state immediately on the
                // next game (no warmup needed). All workers — not
                // just worker 0 — contribute identically to the
                // aggregate stats box via `recordCompletedGame`.
                for workerIndex in 0..<Self.absoluteMaxSelfPlayWorkers {
                    let workerNetwork: ChessMPSNetwork = workerIndex == 0
                        ? network
                        : secondaries[workerIndex - 1]
                    let workerGate: WorkerPauseGate = workerIndex == 0
                        ? selfPlayGate
                        : secondarySelfPlayGates[workerIndex - 1]
                    let isWorker0 = workerIndex == 0

                    group.addTask(priority: .userInitiated) {
                        [workerNetwork, workerGate, buffer, pStatsBox, gameWatcher,
                         isWorker0, countBox] in

                        // Reusable players, allocated once per
                        // worker. `ChessMachine.beginNewGame` calls
                        // `onNewGame` on each before the next game
                        // starts, resetting per-game scratch state
                        // without reallocating.
                        let white = MPSChessPlayer(
                            name: "White",
                            network: workerNetwork,
                            replayBuffer: buffer,
                            schedule: .selfPlay
                        )
                        let black = MPSChessPlayer(
                            name: "Black",
                            network: workerNetwork,
                            replayBuffer: buffer,
                            schedule: .selfPlay
                        )

                        while !Task.isCancelled {
                            // Combined wait check. The worker is
                            // ready to play iff (a) no arena
                            // coordinator has requested a pause
                            // AND (b) its index is below the
                            // current active count. Either
                            // condition false → enter the wait
                            // state (`markWaiting` so arena
                            // `pauseAndWait` succeeds even on
                            // permanently-inactive workers) and
                            // poll until both clear.
                            var shouldPlay = !workerGate.isRequestedToPause
                                && countBox.count > workerIndex
                            if !shouldPlay {
                                workerGate.markWaiting()
                                while !Task.isCancelled && !shouldPlay {
                                    try? await Task.sleep(for: .milliseconds(50))
                                    shouldPlay = !workerGate.isRequestedToPause
                                        && countBox.count > workerIndex
                                }
                                workerGate.markRunning()
                            }
                            if Task.isCancelled { break }

                            // Live-display decision: only wire
                            // `GameWatcher` when the current
                            // concurrency is exactly 1 and this is
                            // worker 0. Evaluated here (not at
                            // spawn) so toggling N between 1 and
                            // >1 at runtime immediately stops or
                            // restarts the animated board on the
                            // next game, rather than carrying a
                            // spawn-time captured flag forever.
                            let liveDisplay = isWorker0 && countBox.count == 1

                            if liveDisplay {
                                gameWatcher.resetCurrentGame()
                                gameWatcher.markPlaying(true)
                            }

                            let machine = ChessMachine()
                            if liveDisplay {
                                machine.delegate = gameWatcher
                            }

                            let gameStart = CFAbsoluteTimeGetCurrent()
                            let result: GameResult
                            do {
                                let task = try machine.beginNewGame(white: white, black: black)
                                result = await task.value
                            } catch {
                                if liveDisplay {
                                    gameWatcher.markPlaying(false)
                                }
                                break
                            }
                            let gameDurationMs = (CFAbsoluteTimeGetCurrent() - gameStart) * 1000

                            // Every worker contributes to the
                            // aggregate session stats identically —
                            // no delegate indirection, no worker
                            // specialness. Move count = total plies
                            // across both players' recorded-moves
                            // counters, read before the next
                            // `beginNewGame` resets them.
                            let positions = white.recordedPliesCount + black.recordedPliesCount
                            pStatsBox.recordCompletedGame(
                                moves: positions,
                                durationMs: gameDurationMs,
                                result: result
                            )
                        }
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
                    [trainer, buffer, box, pStatsBox, trainingGate, triggerBox, ratioController] in
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
                        let timing: TrainStepTiming
                        do {
                            timing = try trainer.trainStep(batch: batch)
                        } catch {
                            box.recordError(error.localizedDescription)
                            return
                        }

                        box.recordStep(timing)
                        pStatsBox.recordTrainingStep()

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

                        // Post-step pause. The replay-ratio controller
                        // records the step, computes the 1-minute rolling
                        // production/consumption ratio, and when auto-
                        // adjust is enabled returns a computed delay that
                        // brings the ratio toward the target. When auto-
                        // adjust is off, the controller returns the manual
                        // delay. Skip the sleep entirely at 0 ms.
                        let stepDelayMs = ratioController.recordStepAndGetDelay(
                            currentBufferTotal: buffer.totalPositionsAdded
                        )
                        if stepDelayMs > 0 {
                            try? await Task.sleep(for: .milliseconds(stepDelayMs))
                        }
                    }
                }

                // Arena coordinator: polls the trigger inbox and runs
                // a tournament whenever one is pending. Blocks its
                // own loop (not the worker tasks) during arena
                // execution. Both the 30-minute auto-fire and the
                // Run Arena button enter here via `triggerBox.trigger()`.
                group.addTask(priority: .userInitiated) {
                    [trainer, network, tBox, selfPlayGate, secondarySelfPlayGates, trainingGate, arenaFlag, triggerBox, overrideBox,
                     candidateInference, arenaChampion, secondaries] in
                    while !Task.isCancelled {
                        if triggerBox.consume() {
                            await self.runArenaParallel(
                                trainer: trainer,
                                champion: network,
                                candidateInference: candidateInference,
                                arenaChampion: arenaChampion,
                                secondarySelfPlayNetworks: secondaries,
                                secondarySelfPlayGates: secondarySelfPlayGates,
                                tBox: tBox,
                                selfPlayGate: selfPlayGate,
                                trainingGate: trainingGate,
                                arenaFlag: arenaFlag,
                                overrideBox: overrideBox
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
                arenaOverrideBox = nil
                parallelWorkerStatsBox = nil
                parallelStats = nil
                workerCountBox = nil
                trainingStepDelayBox = nil
                activeSelfPlayGate = nil
                activeTrainingGate = nil
                currentSessionID = nil
                currentSessionStart = nil
                replayRatioController = nil
                replayRatioSnapshot = nil
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

    private func trainingStatsText() -> (header: String, body: String) {
        let dash = "--"

        // Sweep results trump the per-step display. Once a sweep starts or
        // completes, the table is what the user came here for. The sweep
        // formatter produces its own header line (the "Batch Size Sweep"
        // title) as its first line, so we split it off here so callers can
        // render the split-header layout uniformly across modes.
        if sweepRunning || !sweepResults.isEmpty {
            let sweepText = sweepStatsText()
            let newlineIdx = sweepText.firstIndex(of: "\n") ?? sweepText.endIndex
            let header = String(sweepText[..<newlineIdx])
            let body = newlineIdx == sweepText.endIndex
                ? ""
                : String(sweepText[sweepText.index(after: newlineIdx)...])
            return (header: header, body: body)
        }

        let isSelfPlay = realTraining || replayBuffer != nil
        var lines: [String] = []
        // Header is labelled with the trainer's model ID — the
        // moving SGD copy that arena promotion turns into a
        // champion. The separate Trainer ID / Champion ID rows
        // are dropped: the trainer ID is in the header, and the
        // champion ID is already shown as the Self Play column
        // header.
        let trainerIDStr = trainer?.identifier?.description ?? dash
        let header = "Training [\(trainerIDStr)]"
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
            let bufRamMB = Double(bufCap * ReplayBuffer.bytesPerPosition) / (1024.0 * 1024.0)
            let bufStr = String(format: "%6d / %d  (%.0f MB)", bufCount, bufCap, bufRamMB)
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
            // 1-minute rolling rates from the replay-ratio controller
            if let snap = replayRatioSnapshot {
                let prodStr = snap.productionRate > 0
                    ? String(format: "%.0f", snap.productionRate)
                    : dash
                let consStr = snap.consumptionRate > 0
                    ? String(format: "%.0f", snap.consumptionRate)
                    : dash
                let ratioStr = snap.currentRatio > 0
                    ? String(format: "%.2f", snap.currentRatio)
                    : dash
                lines.append("  1m gen rate: \(prodStr) pos/s")
                lines.append("  1m trn rate: \(consStr) pos/s")
                lines.append("  1m ratio:    \(ratioStr)  (target \(String(format: "%.1f", snap.targetRatio)))")
            }
        }
        lines.append("")

        if let last = lastTrainStep {
            lines.append("Last Step")
            lines.append(String(format: "  Total:       %.2f ms", last.totalMs))
            lines.append(String(format: "  Entropy:     %.6f", last.policyEntropy))
            lines.append("")
        }

        if let stats = trainingStats {
            let stepsStr = stats.steps.formatted(.number.grouping(.automatic))
            // Train time is the sum of per-step wall times — wall time
            // the trainer actually spent inside `trainStep`, exclusive
            // of buffer warmup, gate pauses, and any other idle gaps.
            // Useful as "cumulative GPU training cost"; intentionally
            // not the rate denominator below.
            let trainTimeStr = stats.steps > 0
                ? String(format: "%.2f s", stats.trainingSeconds)
                : dash
            let avgTotal = stats.steps > 0 ? String(format: "%7.2f ms", stats.avgStepMs) : dash

            // Rate denominator: prefer session wall clock from the
            // parallel-worker stats box when one is present (Play
            // and Train mode), so Steps/sec and Moves/sec are
            // directly comparable to the self-play moves/sec figures
            // shown elsewhere — both use "now - sessionStart". In
            // pure training modes (Train Once / Train Continuous)
            // there is no sessionStart, so we fall back to
            // `trainingSeconds`, which in those modes IS the session
            // time anyway because the trainer is the only worker.
            let rateDenomSec: Double
            if let ps = parallelStats {
                rateDenomSec = max(0.1, Date().timeIntervalSince(ps.sessionStart))
            } else {
                rateDenomSec = max(0.1, stats.trainingSeconds)
            }

            let stepsPerSec: Double = stats.steps > 0
                ? Double(stats.steps) / rateDenomSec
                : 0
            let movesPerSec: Double = stats.steps > 0
                ? Double(stats.steps * Self.trainingBatchSize) / rateDenomSec
                : 0

            let rateStr = stats.steps > 0 ? String(format: "%.2f", stepsPerSec) : dash
            let movesSecStr: String
            let movesHrStr: String
            if stats.steps > 0 {
                movesSecStr = Int(movesPerSec.rounded())
                    .formatted(.number.grouping(.automatic))
                movesHrStr = Int((movesPerSec * 3600).rounded())
                    .formatted(.number.grouping(.automatic))
            } else {
                movesSecStr = dash
                movesHrStr = dash
            }

            lines.append("Run Totals")
            lines.append("  Steps done:  \(stepsStr)")
            lines.append("  Train time:  \(trainTimeStr)")
            lines.append("  Avg total:   \(avgTotal)")
            lines.append("  Steps/sec:   \(rateStr)")
            // Moves/sec and moves/hr match the game side's session-
            // stats format (which shows Games/hr and Moves/hr) so the
            // two tables speak the same language. "Moves" here means
            // the same thing as "positions consumed": each training
            // sample is one position, which is equivalent to one
            // move played in the original game.
            lines.append("  Moves/sec:   \(movesSecStr)")
            lines.append("  Moves/hr:    \(movesHrStr)")
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
                // Promoted rows append the ID of the new champion
                // so the stats panel shows the same "which model
                // just took over" information as the session log's
                // [ARENA] lines.
                let marker: String
                if record.promoted, let pid = record.promotedID {
                    marker = "PROMOTED=\(pid.description)"
                } else if record.promoted {
                    marker = "PROMOTED"
                } else {
                    marker = "kept"
                }
                let durStr = Self.formatElapsed(record.durationSec)
                lines.append(String(
                    format: "  #%@ @ %@ steps  %d-%d-%d  score %@  %@  (%@)",
                    number, stepStr,
                    record.candidateWins, record.championWins, record.draws,
                    scoreStr, marker, durStr
                ))
            }
        }

        return (header: header, body: lines.joined(separator: "\n"))
    }

    /// Play and Train self-play stats text. Built from the aggregate
    /// `ParallelWorkerStatsBox` snapshot so all N workers contribute
    /// identically, plus the live `GameWatcher` snapshot used only
    /// when `selfPlayWorkerCount == 1` to render the current-game
    /// Status line. Session rates are computed against wall clock
    /// since `sessionStart` (not the old `GameWatcher` stopwatch,
    /// which was worker-0-only and had an async-dispatch race); a
    /// second column shows the same rates restricted to the rolling
    /// 10-minute window for short-term throughput visibility.
    private func playAndTrainStatsText(
        game: GameWatcher.Snapshot,
        session: ParallelWorkerStatsBox.Snapshot
    ) -> (header: String, body: String) {
        let dash = "--"
        var lines: [String] = []

        // Status line — only meaningful with a single live-driven
        // game. Under N>1 GameWatcher is still fed by worker 0 (so
        // the live board can re-appear instantly when the user
        // drops back to N=1) but the Status line is hidden because
        // it would only describe one of N concurrent games.
        if selfPlayWorkerCount == 1 {
            let status: String
            if game.isPlaying {
                let turn = game.state.currentPlayer == .white ? "White" : "Black"
                let check = MoveGenerator.isInCheck(game.state, color: game.state.currentPlayer) ? " CHECK" : ""
                status = "\(turn) to move (move \(game.moveCount + 1))\(check)"
            } else if let result = game.result {
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
            lines.append("Status: \(status)")
            lines.append("")
        }

        // Section header — labelled with the champion model ID
        // (the network worker 0 plays on; secondaries are
        // weight-mirror copies of it). The lifetime "Time" field
        // used to live here too but moved to the top busy row
        // alongside memory stats — see `busyLabel` for that. The
        // Concurrency row used to live as the first body line but
        // is now rendered outside this string as a SwiftUI HStack
        // with an inline Stepper so the user can adjust N without
        // leaving the stats panel.
        let championIDStr = network?.identifier?.description ?? "no id"
        let header = "Self Play [\(championIDStr)]"

        let games = session.selfPlayGames
        let moves = session.selfPlayPositions
        let elapsed = max(0.1, Date().timeIntervalSince(session.sessionStart))

        let sGames = games > 0 ? games.formatted(.number.grouping(.automatic)) : dash
        let sMoves = moves > 0 ? moves.formatted(.number.grouping(.automatic)) : dash
        // `Time` left this panel on the layout refactor — it now
        // lives in the top busy row next to memory stats. The
        // formatHMS helper still drives that display, just not
        // from here.

        // Wall-clock-derived rate denominator. Rate fields show "--"
        // for the first few seconds of a session so the first game
        // (with elapsed near zero) doesn't flash an absurd
        // millions-of-moves/hr value.
        let ratesValid = elapsed >= Self.statsWarmupSeconds && games > 0

        // System-level averages: every metric measures the
        // collective rate the N workers produce, not the per-worker
        // average. With N workers, "Avg move" is wall-clock seconds
        // divided by total moves (N times faster than per-worker
        // move time), and "Avg game" is wall-clock seconds divided
        // by total games. This matches the user's natural reading:
        // "the system pops out a move every X ms" / "a game every
        // Y ms," which is what the busy label's positions/sec also
        // reports. Per-worker averages are not displayed.
        let elapsedMs = elapsed * 1000
        let lifetimeAvgMoveMs = moves > 0 ? elapsedMs / Double(moves) : 0
        let lifetimeAvgGameMs = games > 0 ? elapsedMs / Double(games) : 0
        let lifetimeMovesPerHour = Double(moves) / elapsed * 3600
        let lifetimeGamesPerHour = Double(games) / elapsed * 3600

        // Rolling-window aggregates. The right denominator for "rate
        // over the last 10 minutes" is `min(recentWindow, elapsed)`,
        // *not* the gap between the oldest stored entry and now —
        // the gap form collapses to zero on the first game and
        // understates the window in steady state. With min(window,
        // elapsed): during the first 10 minutes of a session the
        // rolling values equal the lifetime values (the window
        // covers everything since sessionStart); after 10 minutes
        // the rolling window is exactly 10 minutes wide.
        let recentGames = session.recentGames
        let recentMoves = session.recentMoves
        let recentDenom = min(ParallelWorkerStatsBox.recentWindow, elapsed)
        let recentDenomMs = recentDenom * 1000

        let recentAvgMoveMs = recentMoves > 0 ? recentDenomMs / Double(recentMoves) : 0
        let recentAvgGameMs = recentGames > 0 ? recentDenomMs / Double(recentGames) : 0
        let recentMovesPerHour = recentDenom > 0 ? Double(recentMoves) / recentDenom * 3600 : 0
        let recentGamesPerHour = recentDenom > 0 ? Double(recentGames) / recentDenom * 3600 : 0

        let sAvgMove = ratesValid && moves > 0
            ? String(format: "%.2f ms", lifetimeAvgMoveMs)
            : dash
        let sAvgGame = ratesValid && games > 0
            ? String(format: "%.1f ms", lifetimeAvgGameMs)
            : dash
        let sMovesHr = ratesValid
            ? Int(lifetimeMovesPerHour.rounded()).formatted(.number.grouping(.automatic))
            : dash
        let sGamesHr = ratesValid
            ? Int(lifetimeGamesPerHour.rounded()).formatted(.number.grouping(.automatic))
            : dash

        let sAvgMoveR = ratesValid && recentGames > 0
            ? String(format: "%.2f ms", recentAvgMoveMs)
            : dash
        let sAvgGameR = ratesValid && recentGames > 0
            ? String(format: "%.1f ms", recentAvgGameMs)
            : dash
        let sMovesHrR = ratesValid && recentGames > 0
            ? Int(recentMovesPerHour.rounded()).formatted(.number.grouping(.automatic))
            : dash
        let sGamesHrR = ratesValid && recentGames > 0
            ? Int(recentGamesPerHour.rounded()).formatted(.number.grouping(.automatic))
            : dash

        // Column-aligned output. First rate column is right-padded
        // to 12 chars so the 10-min column starts at a consistent
        // offset regardless of first-column width; second column
        // renders its value directly (no padding needed — it's the
        // last thing on the line).
        func rjust(_ value: String, _ width: Int) -> String {
            guard value.count < width else { return value }
            return String(repeating: " ", count: width - value.count) + value
        }

        lines.append("  Games:     \(rjust(sGames, 12))")
        lines.append("  Moves:     \(rjust(sMoves, 12))")
        lines.append("                             (last 10m)")
        lines.append("  Avg move:  \(rjust(sAvgMove, 12))  \(rjust(sAvgMoveR, 12))")
        lines.append("  Avg game:  \(rjust(sAvgGame, 12))  \(rjust(sAvgGameR, 12))")
        lines.append("  Moves/hr:  \(rjust(sMovesHr, 12))  \(rjust(sMovesHrR, 12))")
        lines.append("  Games/hr:  \(rjust(sGamesHr, 12))  \(rjust(sGamesHrR, 12))")
        lines.append("")

        // Results — per-outcome counters from the aggregate box,
        // formatted exactly like the old GameWatcher rendering so
        // the display layout is unchanged.
        let totalCheckmates = session.whiteCheckmates + session.blackCheckmates
        func pct(_ count: Int) -> String {
            guard games > 0 else { return "" }
            return String(format: "  (%.1f%%)", Double(count) / Double(games) * 100)
        }

        lines.append("Results")
        lines.append("  Checkmate:      \(totalCheckmates)\(pct(totalCheckmates))")
        lines.append("    White wins:     \(session.whiteCheckmates)\(pct(session.whiteCheckmates))")
        lines.append("    Black wins:     \(session.blackCheckmates)\(pct(session.blackCheckmates))")
        lines.append("  Stalemate:      \(session.stalemates)\(pct(session.stalemates))")
        lines.append("  50-move draw:   \(session.fiftyMoveDraws)\(pct(session.fiftyMoveDraws))")
        lines.append("  Threefold rep:  \(session.threefoldRepetitionDraws)\(pct(session.threefoldRepetitionDraws))")
        lines.append("  Insufficient:   \(session.insufficientMaterialDraws)\(pct(session.insufficientMaterialDraws))")

        return (header: header, body: lines.joined(separator: "\n"))
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
