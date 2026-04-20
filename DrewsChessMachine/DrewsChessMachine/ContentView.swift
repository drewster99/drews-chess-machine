import AppKit
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

struct TrainingAlarm: Identifiable, Equatable, Sendable {
    enum Severity: String, Sendable {
        case warning
        case critical
    }

    let id: UUID
    let severity: Severity
    let title: String
    let detail: String
    let raisedAt: Date
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
    private let queue = DispatchQueue(label: "drewschess.cancelbox.serial")
    private var _cancelled = false
    private var _progress: SweepProgress?
    private var _completedRows: [SweepRow] = []
    private var _rowPeakBytes: UInt64 = 0

    var isCancelled: Bool {
        queue.sync { _cancelled }
    }

    func cancel() {
        queue.async { [weak self] in self?._cancelled = true }
    }

    func updateProgress(_ p: SweepProgress) {
        queue.async { [weak self] in self?._progress = p }
    }

    var latestProgress: SweepProgress? {
        queue.sync { _progress }
    }

    func appendRow(_ r: SweepRow) {
        queue.async { [weak self] in self?._completedRows.append(r) }
    }

    var completedRows: [SweepRow] {
        queue.sync { _completedRows }
    }

    /// Update the per-row peak with a new sample. The sweep's worker
    /// thread reads and resets this between rows via `takeRowPeak()`.
    /// Called from both the UI heartbeat (every ~100ms) and from the
    /// trainer at row boundaries — whichever produces the higher value
    /// wins for that row.
    func recordPeakSample(_ bytes: UInt64) {
        queue.async { [weak self] in
            guard let self else { return }
            if bytes > self._rowPeakBytes { self._rowPeakBytes = bytes }
        }
    }

    /// Read the peak observed during the just-finished row and reset the
    /// accumulator for the next one.
    func takeRowPeak() -> UInt64 {
        queue.sync {
            let peak = _rowPeakBytes
            _rowPeakBytes = 0
            return peak
        }
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

    private let queue = DispatchQueue(label: "drewschess.gamewatcher.serial")
    private var s = Snapshot()

    func snapshot() -> Snapshot {
        queue.sync { s }
    }

    func resetCurrentGame() {
        queue.async { [weak self] in
            guard let self else { return }
            self.s.state = .starting
            self.s.result = nil
            self.s.moveCount = 0
            // Keep lastGameStats — show previous game until the next one ends
        }
    }

    func resetAll() {
        queue.async { [weak self] in
            self?.s = Snapshot()
        }
    }

    func markPlaying(_ playing: Bool) {
        let now = CFAbsoluteTimeGetCurrent()
        queue.async { [weak self] in
            self?.setPlayingOnQueue(playing, now: now)
        }
    }

    /// Toggle isPlaying and update the active-play stopwatch. Caller
    /// must already be executing on `queue` and must pass a `now`
    /// value captured at the original call site — the async hop onto
    /// `queue` may land microseconds later, and the stopwatch needs
    /// to reflect the moment the caller decided to start/stop play,
    /// not when the queue got to the closure. Idempotent: calling
    /// with the same value twice is a no-op for the stopwatch.
    private func setPlayingOnQueue(_ playing: Bool, now: CFAbsoluteTime) {
        if playing {
            if s.currentPlayStartTime == nil {
                s.currentPlayStartTime = now
            }
        } else if let start = s.currentPlayStartTime {
            s.activePlaySeconds += max(0, now - start)
            s.currentPlayStartTime = nil
        }
        s.isPlaying = playing
    }

    // MARK: - Delegate (called on ChessMachine.delegateQueue, never main)

    func chessMachine(_ machine: ChessMachine, didApplyMove move: ChessMove, newState: GameState) {
        queue.async { [weak self] in
            guard let self else { return }
            self.s.state = newState
            self.s.moveCount += 1
        }
    }

    func chessMachine(
        _ machine: ChessMachine,
        gameEndedWith result: GameResult,
        finalState: GameState,
        stats: GameStats
    ) {
        let now = CFAbsoluteTimeGetCurrent()
        queue.async { [weak self] in
            guard let self else { return }
            self.s.result = result
            self.s.state = finalState
            self.s.lastGameStats = stats
            self.setPlayingOnQueue(false, now: now)

            self.s.totalGames += 1
            self.s.totalMoves += stats.totalMoves
            self.s.totalGameTimeMs += stats.totalGameTimeMs
            self.s.totalWhiteThinkMs += stats.whiteThinkingTimeMs
            self.s.totalBlackThinkMs += stats.blackThinkingTimeMs
            // Move counting handed off to totalMoves; zero the per-game counter
            // atomically so display helpers using `totalMoves + moveCount` don't
            // double-count between gameEnded and the next resetCurrentGame call.
            self.s.moveCount = 0

            switch result {
            case .checkmate(let winner):
                if winner == .white {
                    self.s.whiteCheckmates += 1
                } else {
                    self.s.blackCheckmates += 1
                }
            case .stalemate:
                self.s.stalemates += 1
            case .drawByFiftyMoveRule:
                self.s.fiftyMoveDraws += 1
            case .drawByInsufficientMaterial:
                self.s.insufficientMaterialDraws += 1
            case .drawByThreefoldRepetition:
                self.s.threefoldRepetitionDraws += 1
            }
        }
    }

    func chessMachine(_ machine: ChessMachine, playerErrored player: any ChessPlayer, error: any Error) {
        let now = CFAbsoluteTimeGetCurrent()
        queue.async { [weak self] in
            self?.setPlayingOnQueue(false, now: now)
        }
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

/// Which network the Candidate test probe evaluates. `.candidate` is
/// the default — the trainer's current in-flight weights get synced
/// into the candidate inference network and probed. `.champion`
/// bypasses the sync and probes the actual champion network directly,
/// so the user can compare the two at the same position. The champion
/// is frozen between promotions, so its output should be stable over
/// the session; diffing its value-head output against the candidate's
/// at a fixed position is the cheapest way to tell whether training
/// is actually moving the value head (or whether it's saturated at
/// the same spot the random init put it).
enum ProbeNetworkTarget: Sendable, Hashable {
    case candidate
    case champion
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
/// How a tournament ended in terms of the promotion decision.
/// `nil` on a `TournamentRecord` means the champion was kept (no
/// promotion); otherwise this distinguishes an automatic promotion
/// (score met the configured threshold and a full tournament was
/// played) from a manual one (user clicked the Promote button,
/// which forces promotion regardless of score or completion).
enum PromotionKind: String, Sendable, Codable {
    case automatic
    case manual
}

struct TournamentRecord: Sendable, Identifiable {
    let id = UUID()
    /// `trainingStats.steps` at the moment the tournament finished.
    let finishedAtStep: Int
    /// Number of arena games that actually completed before the
    /// tournament ended. May be less than `Self.tournamentGames` if
    /// the user clicked Abort or Promote mid-tournament, or if the
    /// session was stopped while the arena was in flight.
    let gamesPlayed: Int
    let candidateWins: Int
    let championWins: Int
    let draws: Int
    let score: Double
    let promoted: Bool
    /// Whether the promotion (when `promoted == true`) was triggered
    /// by the configured score threshold (`.automatic`) or by the
    /// user's Promote button (`.manual`). `nil` when `promoted` is
    /// false.
    let promotionKind: PromotionKind?
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

/// Serial-queue-protected holder for live tournament progress, shared
/// between the driver task (writer, one update per finished game) and
/// the UI heartbeat (reader, polling at 60 Hz). Same pattern as
/// `TrainingLiveStatsBox` and `CancelBox` — a private serial
/// `DispatchQueue` serializes all state access, so the class is
/// safely `@unchecked Sendable`.
final class TournamentLiveBox: @unchecked Sendable {
    private let queue = DispatchQueue(label: "drewschess.tournamentlivebox.serial")
    private var _progress: TournamentProgress?

    func update(_ progress: TournamentProgress) {
        queue.async { [weak self] in self?._progress = progress }
    }

    func snapshot() -> TournamentProgress? {
        queue.sync { _progress }
    }

    func clear() {
        queue.async { [weak self] in self?._progress = nil }
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
    private let queue = DispatchQueue(label: "drewschess.workerpausegate.serial")
    private var _requested = false
    private var _isWaiting = false

    /// Polled by the worker at each iteration boundary.
    var isRequestedToPause: Bool {
        queue.sync { _requested }
    }

    /// Called by the worker when it enters its spin-wait state, so
    /// the coordinator knows it's safe to start the protected work.
    /// Uses `queue.sync` to publish the flag before the worker starts
    /// polling — coordinators spin on `readIsWaiting()` and must see
    /// the acknowledgement as soon as the worker returns from this
    /// method.
    func markWaiting() {
        queue.sync { _isWaiting = true }
    }

    /// Called by the worker when it leaves its spin-wait state and
    /// resumes normal iteration.
    func markRunning() {
        queue.sync { _isWaiting = false }
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

    /// Synchronous helpers so `pauseAndWait()` doesn't hold a lock
    /// across an `await` — the `queue.sync` hop is a bounded,
    /// contention-free critical section that returns immediately.
    private func setRequested(_ value: Bool) {
        queue.sync { _requested = value }
    }

    private func readIsWaiting() -> Bool {
        queue.sync { _isWaiting }
    }

    /// Coordinator: release the worker. Clears the request flag so
    /// the worker's next spin-wait iteration sees it and resumes.
    func resume() {
        queue.sync { _requested = false }
    }
}

/// Lock-protected holder for the current self-play and arena
/// `SamplingSchedule` objects. Written by the SwiftUI edit fields
/// (on the main actor) and read by the `BatchedSelfPlayDriver`'s
/// slot tasks when constructing each new `MPSChessPlayer` pair, and
/// by the arena setup code when building arena players. Because
/// `MPSChessPlayer` captures its `SamplingSchedule` at init and the
/// player is reused across games within a slot, edits take effect
/// on newly-constructed players — i.e. at the next game-boundary
/// within the driver's slotLoop, not mid-game.
final class SamplingScheduleBox: @unchecked Sendable {
    private let queue = DispatchQueue(label: "drewschess.samplingschedulebox.serial")
    private var _selfPlay: SamplingSchedule
    private var _arena: SamplingSchedule

    init(selfPlay: SamplingSchedule, arena: SamplingSchedule) {
        self._selfPlay = selfPlay
        self._arena = arena
    }

    var selfPlay: SamplingSchedule {
        queue.sync { _selfPlay }
    }

    var arena: SamplingSchedule {
        queue.sync { _arena }
    }

    func setSelfPlay(_ s: SamplingSchedule) {
        queue.sync { _selfPlay = s }
    }

    func setArena(_ s: SamplingSchedule) {
        queue.sync { _arena = s }
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
    private let queue = DispatchQueue(label: "drewschess.workercountbox.serial")
    private var _count: Int

    init(initial: Int) {
        precondition(initial >= 1, "WorkerCountBox initial count must be >= 1")
        _count = initial
    }

    var count: Int {
        queue.sync { _count }
    }

    /// Set the active worker count. Clamped at the bottom to 1 so a
    /// stuck Stepper or a sloppy caller can never zero out self-play
    /// (the upper bound is enforced by the Stepper and the spawn
    /// loop's `absoluteMaxSelfPlayWorkers` constant, not here).
    func set(_ value: Int) {
        queue.async { [weak self] in self?._count = max(1, value) }
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
    private let queue = DispatchQueue(label: "drewschess.trainingstepdelaybox.serial")
    private var _ms: Int

    init(initial: Int) {
        _ms = max(0, initial)
    }

    var milliseconds: Int {
        queue.sync { _ms }
    }

    func set(_ value: Int) {
        queue.async { [weak self] in self?._ms = max(0, value) }
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
    private let queue = DispatchQueue(label: "drewschess.arenatriggerbox.serial")
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
        queue.sync {
            if _pending { return false }
            return Date().timeIntervalSince(_lastArenaTime) >= interval
        }
    }

    /// Set the pending flag. The arena coordinator's next poll will
    /// consume it and start an arena.
    func trigger() {
        queue.async { [weak self] in self?._pending = true }
    }

    /// Poll the trigger. Returns true and clears the pending flag if
    /// a trigger was waiting; returns false otherwise.
    func consume() -> Bool {
        queue.sync {
            if _pending {
                _pending = false
                return true
            }
            return false
        }
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
        let now = Date()
        queue.async { [weak self] in
            guard let self else { return }
            self._lastArenaTime = now
            self._pending = false
        }
    }

    /// True if a trigger is currently pending (used for UI
    /// disable-while-queued semantics).
    var isPending: Bool {
        queue.sync { _pending }
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

    private let queue = DispatchQueue(label: "drewschess.arenaoverridebox.serial")
    private var _decision: Decision?

    /// Request abort: end the current tournament early with no
    /// promotion. No-op if a decision (abort or promote) is already
    /// set for this tournament.
    func abort() {
        queue.async { [weak self] in
            guard let self else { return }
            if self._decision == nil {
                self._decision = .abort
            }
        }
    }

    /// Request forced promotion: end the current tournament early
    /// and promote the candidate unconditionally. No-op if a decision
    /// (abort or promote) is already set for this tournament.
    func promote() {
        queue.async { [weak self] in
            guard let self else { return }
            if self._decision == nil {
                self._decision = .promote
            }
        }
    }

    /// True once either `abort()` or `promote()` has been called,
    /// until `consume()` resets the box. Polled by the tournament
    /// driver's `isCancelled` closure so the game loop breaks out
    /// between games the moment the user clicks one of the buttons.
    var isActive: Bool {
        queue.sync { _decision != nil }
    }

    /// Read-and-clear the decision. Returns `nil` if no override
    /// was set (normal tournament completion), or the decision the
    /// user made. Called once by `runArenaParallel` after the
    /// driver returns, both to branch on the decision and to reset
    /// the box for the next tournament.
    func consume() -> Decision? {
        queue.sync {
            let d = _decision
            _decision = nil
            return d
        }
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
    private let queue = DispatchQueue(label: "drewschess.arenaactiveflag.serial")
    private var _active = false

    var isActive: Bool {
        queue.sync { _active }
    }

    func set() {
        queue.async { [weak self] in self?._active = true }
    }

    func clear() {
        queue.async { [weak self] in self?._active = false }
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

    private let queue = DispatchQueue(label: "drewschess.parallelworkerstatsbox.serial")
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
    /// Fixed-capacity ring of recent game lengths (plies), used to
    /// compute p50/p95 in `Snapshot`. Sized for a few hundred games
    /// — plenty for a meaningful percentile on the 10-minute
    /// cadence without carrying the rolling-window memory back to
    /// a per-game scan. Lengths-only (no timestamps) because
    /// percentile semantics don't age the samples out by time.
    private var _recentGameLengths: [Int] = []
    private var _recentGameLengthsHead: Int = 0
    private static let gameLengthRingCapacity: Int = 512
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

    /// Seeded init for session resume. All counters pick up where
    /// the saved session left off so the UI shows continuity.
    init(
        sessionStart: Date,
        totalGames: Int,
        totalMoves: Int,
        totalGameWallMs: Double,
        whiteCheckmates: Int,
        blackCheckmates: Int,
        stalemates: Int,
        fiftyMoveDraws: Int,
        threefoldRepetitionDraws: Int,
        insufficientMaterialDraws: Int,
        trainingSteps: Int
    ) {
        self._sessionStart = sessionStart
        self._totalGames = totalGames
        self._totalMoves = totalMoves
        self._totalGameWallMs = totalGameWallMs
        self._whiteCheckmates = whiteCheckmates
        self._blackCheckmates = blackCheckmates
        self._stalemates = stalemates
        self._fiftyMoveDraws = fiftyMoveDraws
        self._threefoldRepetitionDraws = threefoldRepetitionDraws
        self._insufficientMaterialDraws = insufficientMaterialDraws
        self._trainingSteps = trainingSteps
    }

    /// Advance `sessionStart` to `Date()`. Called once from inside
    /// the Play-and-Train task, immediately before the worker group
    /// is spawned, so that rate denominators only cover the window
    /// in which workers are actually running. Captures `Date()` at
    /// the call site (not inside the closure) so the timestamp
    /// reflects the moment the caller invoked this method — the
    /// async dispatch may land microseconds later once any queued
    /// stats writes drain.
    func markWorkersStarted() {
        let now = Date()
        queue.async { [weak self] in
            self?._sessionStart = now
        }
    }

    /// Reset game-play counters so post-promotion stats reflect
    /// only the newly-promoted champion's self-play performance.
    /// Training step count and sessionStart are NOT reset so
    /// training-rate display stays continuous.
    func resetGameStats() {
        queue.async { [weak self] in
            guard let self else { return }
            self._totalGames = 0
            self._totalMoves = 0
            self._totalGameWallMs = 0
            self._whiteCheckmates = 0
            self._blackCheckmates = 0
            self._stalemates = 0
            self._fiftyMoveDraws = 0
            self._threefoldRepetitionDraws = 0
            self._insufficientMaterialDraws = 0
            self._recentGames.removeAll()
            self._recentGameLengths.removeAll(keepingCapacity: true)
            self._recentGameLengthsHead = 0
        }
    }

    /// Record one completed self-play game. Called from every worker
    /// at game-end with the game's total moves, wall-clock duration,
    /// and final result. Bumps lifetime totals, the per-outcome
    /// counters, and the rolling 10-minute window. Thread-safe via
    /// the box's serial queue. `Date()` is captured at the call site
    /// — the game-end timestamp feeds the rolling-window rate stats,
    /// so it must reflect when the game actually finished rather
    /// than when the queue got around to processing the write.
    func recordCompletedGame(moves: Int, durationMs: Double, result: GameResult) {
        let now = Date()
        queue.async { [weak self] in
            guard let self else { return }
            self._totalGames += 1
            self._totalMoves += moves
            self._totalGameWallMs += durationMs

            switch result {
            case .checkmate(let winner):
                if winner == .white {
                    self._whiteCheckmates += 1
                } else {
                    self._blackCheckmates += 1
                }
            case .stalemate:
                self._stalemates += 1
            case .drawByFiftyMoveRule:
                self._fiftyMoveDraws += 1
            case .drawByInsufficientMaterial:
                self._insufficientMaterialDraws += 1
            case .drawByThreefoldRepetition:
                self._threefoldRepetitionDraws += 1
            }

            self._recentGames.append(GameRecord(timestamp: now, moves: moves, durationMs: durationMs))
            self.pruneRecentOnQueue(now: now)

            // Percentile ring. Pre-sized lazily; FIFO overwrite once
            // full. `moves` is plies (not half-moves pairs), matching
            // every other game-length counter the app exposes.
            if self._recentGameLengths.count < Self.gameLengthRingCapacity {
                self._recentGameLengths.append(moves)
            } else {
                self._recentGameLengths[self._recentGameLengthsHead] = moves
                self._recentGameLengthsHead = (self._recentGameLengthsHead + 1) % Self.gameLengthRingCapacity
            }
        }
    }

    func recordTrainingStep() {
        queue.async { [weak self] in
            self?._trainingSteps += 1
        }
    }

    /// Drop rolling-window entries older than `now - recentWindow`.
    /// Caller must already be executing on `queue`. Records are
    /// appended in monotonic timestamp order so this is a prefix
    /// removal — O(k) where k is the expired count.
    private func pruneRecentOnQueue(now: Date) {
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
        /// Percentiles of recent game lengths (plies) from the
        /// fixed-capacity length ring. `nil` while the ring is empty.
        /// Pairs with `avgLen` in the `[STATS]` line to expose the
        /// draw/mate bimodality that a mean alone hides.
        let gameLenP50: Int?
        let gameLenP95: Int?
    }

    func snapshot() -> Snapshot {
        queue.sync {
            let now = Date()
            pruneRecentOnQueue(now: now)

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

            let (p50, p95): (Int?, Int?) = {
                guard !_recentGameLengths.isEmpty else { return (nil, nil) }
                var sorted = _recentGameLengths
                sorted.sort()
                let n = sorted.count
                func idx(_ p: Double) -> Int {
                    max(0, min(n - 1, Int((p * Double(n - 1)).rounded())))
                }
                return (sorted[idx(0.50)], sorted[idx(0.95)])
            }()

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
                recentWindowSeconds: recentWindowSec,
                gameLenP50: p50,
                gameLenP95: p95
            )
        }
    }
}

// MARK: - Window Accessor

/// Captures the hosting `NSWindow` of a SwiftUI view into a
/// `@State` binding. Needed because the global
/// `NSWindow.willCloseNotification` fires for every NSWindow the
/// app ever puts on screen — including the Log Analysis auxiliary
/// window (`LogAnalysisWindowController`) and any NSOpenPanel /
/// NSSavePanel the user raises via the File menu. Comparing the
/// notification's `object` against this captured pointer lets the
/// teardown hook ignore anything that isn't the main ContentView
/// window.
///
/// The dispatch to main is required: `NSView.window` is nil during
/// `makeNSView` because the view hasn't been inserted into a
/// window yet. Deferring by one runloop tick lets the parent
/// finish attaching and gives us a valid window reference.
private struct WindowAccessor: NSViewRepresentable {
    @Binding var window: NSWindow?
    func makeNSView(context: Context) -> NSView {
        let v = NSView()
        DispatchQueue.main.async {
            self.window = v.window
        }
        return v
    }
    func updateNSView(_ nsView: NSView, context: Context) {}
}

// MARK: - Content View

struct ContentView: View {
    /// Menu-bar command hub. Assigned by `DrewsChessMachineApp`; the
    /// view wires its action functions into the hub's closure slots
    /// and keeps the hub's mirrored state flags synced so the
    /// `.commands` DSL can enable/disable menu items correctly.
    let commandHub: AppCommandHub

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
    // Legacy `secondarySelfPlayNetworks` removed — all self-play
    // workers now share the champion network (`network`) through a
    // `BatchedMoveEvaluationSource` barrier batcher, so N per-worker
    // inference networks are no longer needed.
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
    /// Rolling-window game diversity tracker for self-play. Fed by
    /// every self-play worker at game end; snapshot polled by the UI
    /// heartbeat for display and by the stats logger for [STATS] lines.
    @State private var selfPlayDiversityTracker: GameDiversityTracker?

    /// Latest divergence-ply histogram bars mirrored from
    /// `selfPlayDiversityTracker` by the UI heartbeat, at the same
    /// throttled cadence as `parallelStats`. The chart-grid view reads
    /// these directly — pushing through @State (rather than reading
    /// the tracker at render time) keeps SwiftUI's dependency graph
    /// correct so the bar chart actually redraws as counts shift.
    @State private var currentDiversityHistogramBars: [DiversityHistogramBar] = []

    /// Completed arenas this session, tagged with their start/end
    /// elapsed-second positions on the chart grid's X axis. Appended
    /// to every time `runArenaParallel` finishes; reset on session
    /// start/stop. Drives the "Arena activity" chart tile.
    @State private var arenaChartEvents: [ArenaChartEvent] = []
    /// Elapsed-second mark when the *current* arena started, `nil`
    /// when no arena is in progress. Set at the top of
    /// `runArenaParallel`, cleared when the final completed
    /// `ArenaChartEvent` is appended. The chart tile uses this to
    /// render a "live" band from this start up to the latest chart
    /// sample's elapsed time — so an active arena is visible on the
    /// chart the moment it starts, rather than only appearing at
    /// end-of-arena when the completed band lands.
    @State private var activeArenaStartElapsed: Double?
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
    /// Which network the Candidate test probe runs against. Defaults to
    /// `.candidate` (the historical behavior — probe the trainer's
    /// latest weights by syncing them into the candidate inference
    /// network). `.champion` probes the champion network directly,
    /// giving a frozen reference point the candidate can be diffed
    /// against at any position — useful for confirming whether the
    /// value head is actually moving or is stuck at init saturation.
    @State private var probeNetworkTarget: ProbeNetworkTarget = .candidate
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
    nonisolated static let trainerLearningRateDefault: Float = 5e-5
    nonisolated static let entropyRegularizationCoeffDefault: Float = 1e-1
    nonisolated static let drawPenaltyDefault: Float = 0.1
    nonisolated static let trainingBatchSize = 4096

    /// Policy-entropy floor below which the periodic stats ticker
    /// emits an `[ALARM]` log line.
    ///
    /// Theoretical maximum is `log(policySize) ≈ 8.49` nats for the
    /// current 4864-logit head. **However, init entropy in v2
    /// architecture sits much lower than that** — empirically ~6.5 at
    /// step 0 — because the new policy head is a bare 1×1 conv with
    /// no preceding BN/ReLU, so logits have std ~2-2.5 at init
    /// (versus std ~1 in v1 where the policy head's BN normalized the
    /// input). Compressed softmax entropy by ~σ²/2 ≈ 2-3 nats from
    /// uniform. The entropy regularization term (`entropy_reg = 1e-3`)
    /// pulls entropy back up over training as the network learns; we
    /// observe gradual rise from 6.48 → 6.51 in early steps.
    ///
    /// Threshold of 5.0 leaves a ~1.5-nat margin below the v2 init
    /// baseline — wide enough to avoid false alarms during normal
    /// training, narrow enough to flag genuine collapse (entropy
    /// dropping toward 1-2 nats means the policy has concentrated on
    /// just a handful of moves, the classic "policy collapse"
    /// failure mode).
    nonisolated static let policyEntropyAlarmThreshold: Double = 5.0
    /// Number of training steps at the start of a Play-and-Train
    /// session for which the `[STATS]` line fires on every step.
    /// After this many steps the STATS ticker switches to a 60 s
    /// time-based cadence. 500 picked so the bootstrap window covers
    /// the first few minutes of training — long enough to see the
    /// initial loss curve shape without flooding the log once
    /// training settles.
    nonisolated static let bootstrapStatsStepCount: Int = 500
    nonisolated static let divergenceAlarmGradNormWarningThreshold: Double = 50.0
    nonisolated static let divergenceAlarmGradNormCriticalThreshold: Double = 500.0
    nonisolated static let divergenceAlarmEntropyCriticalThreshold: Double = 3.0
    nonisolated static let divergenceAlarmConsecutiveWarningSamples: Int = 3
    nonisolated static let divergenceAlarmConsecutiveCriticalSamples: Int = 2
    nonisolated static let divergenceAlarmRecoverySamples: Int = 10

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
    nonisolated static let initialSelfPlayWorkerCount: Int = 24
    /// Hard ceiling on how many self-play slots can run
    /// concurrently in a single session. Since all slots share one
    /// `ChessMPSNetwork` (the champion) through the barrier
    /// batcher, raising this no longer costs per-slot network
    /// memory — the limit is now the batcher's per-batch-size feed
    /// cache footprint (one `[N, inputPlanes, 8, 8]` float32 MPSNDArray
    /// per distinct N, so ~5.1 KB per slot) plus the per-batch
    /// `graph.run` latency. Must be ≥ `initialSelfPlayWorkerCount`.
    nonisolated static let absoluteMaxSelfPlayWorkers: Int = 64
    /// Current active self-play worker count for the running
    /// session. The Stepper writes through `workerCountBinding`
    /// which updates this value and `workerCountBox` atomically;
    /// workers poll the box at the top of each iteration to
    /// decide whether to play another game or sit in their idle
    /// wait state. Persisted to UserDefaults via `@AppStorage` so
    /// the user's last chosen concurrency level survives app
    /// restart. Bounded at runtime by `absoluteMaxSelfPlayWorkers`.
    @AppStorage("selfPlayWorkerCount") private var selfPlayWorkerCount: Int = Self.initialSelfPlayWorkerCount
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
    @State private var trainingChartSamples: [TrainingChartSample] = []
    @State private var trainingChartNextId: Int = 0
    /// Previous totalGpuMs reading for computing per-second GPU busy %.
    @State private var prevChartTotalGpuMs: Double = 0
    @State private var showingInfoPopover: Bool = false
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
    /// Currently-hovered elapsed-second on the large progress-rate
    /// chart, `nil` when the mouse isn't over the chart. Drives the
    /// crosshair + overlay readout.
    @State private var bigProgressChartHoveredSec: Double?
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
    nonisolated static let progressRateVisibleDomainSec: Double = 1800
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

    // MARK: - Training Segments (cumulative wall-time tracking)

    /// All Play-and-Train runs that have completed (Stop, Save, or
    /// session restart) since this session was opened. On load, this
    /// is hydrated from `SessionCheckpointState.trainingSegments`. On
    /// save, the current run (if any) is closed and appended before
    /// the snapshot is written. Cumulative status-bar metrics sum
    /// across this array plus the in-flight current run.
    @State private var completedTrainingSegments: [SessionCheckpointState.TrainingSegment] = []

    /// Per-run start info captured when Play-and-Train begins. Held
    /// in-memory only — closed and appended into `completedTrainingSegments`
    /// when training stops, save fires, or the session ends.
    private struct ActiveSegmentStart {
        let startUnix: Int64
        let startDate: Date
        let startingTrainingStep: Int
        let startingTotalPositions: Int
        let startingSelfPlayGames: Int
        let buildNumber: Int?
        let buildGitHash: String?
        let buildGitDirty: Bool?
    }
    @State private var activeSegmentStart: ActiveSegmentStart?

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

    /// Live sampling schedules shared between UI edit fields and the
    /// self-play / arena players. Constructed at session start from the
    /// persisted `@AppStorage` values; `onChange` handlers on the
    /// tau fields call `setSelfPlay` / `setArena` with freshly
    /// constructed `SamplingSchedule` objects so edits take effect at
    /// each slot's next game boundary. Cleared when a session ends.
    @State private var samplingScheduleBox: SamplingScheduleBox?

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
    @AppStorage("entropyRegularizationCoeff")
    private var entropyRegularizationCoeff: Double = Double(entropyRegularizationCoeffDefault)
    /// Bootstrap-phase draw penalty. Drawn games normally contribute
    /// z=0 to the REINFORCE policy loss, which gives no gradient —
    /// when 82 %+ of self-play games are threefold-repetition draws,
    /// that's most of the replay buffer contributing nothing. With
    /// this set to a small positive value, draws enter training as
    /// `z = -drawPenalty`, giving a mild negative signal so the
    /// policy has a reason to avoid shuffling sequences during the
    /// bootstrap phase. See `ChessTrainer.drawPenalty`.
    @AppStorage("drawPenalty")
    private var drawPenalty: Double = Double(drawPenaltyDefault)
    /// L2 weight-decay coefficient (decoupled / AdamW-style). Default
    /// matches `ChessTrainer.weightDecayCDefault` (1e-4); live edits
    /// flow through `ensureTrainer()` into the trainer's per-step feed
    /// so new values take effect on the next SGD step without graph
    /// rebuild.
    @AppStorage("weightDecayC")
    private var weightDecayC: Double = Double(ChessTrainer.weightDecayCDefault)
    /// Global L2-norm gradient clip. Default matches
    /// `ChessTrainer.gradClipMaxNormDefault` (5.0); edits are
    /// hot-applied per step.
    @AppStorage("gradClipMaxNorm")
    private var gradClipMaxNorm: Double = Double(ChessTrainer.gradClipMaxNormDefault)
    /// Policy-loss coefficient K. Scales the policy term in the
    /// total loss so its gradient is comparable to the value term's.
    /// Default matches `ChessTrainer.policyScaleKDefault` (5).
    @AppStorage("policyScaleK")
    private var policyScaleK: Double = Double(ChessTrainer.policyScaleKDefault)

    // Self-play sampling schedule (live-tunable). Backed by
    // `SamplingSchedule.selfPlay` defaults. New schedules take effect
    // on each newly-constructed player at the next game start within
    // a slot; currently-playing games keep their schedule to avoid
    // per-ply reads of shared mutable state.
    @AppStorage("spStartTau")
    private var spStartTau: Double = Double(SamplingSchedule.selfPlay.startTau)
    @AppStorage("spFloorTau")
    private var spFloorTau: Double = Double(SamplingSchedule.selfPlay.floorTau)
    @AppStorage("spDecayPerPly")
    private var spDecayPerPly: Double = Double(SamplingSchedule.selfPlay.decayPerPly)
    @AppStorage("arStartTau")
    private var arStartTau: Double = Double(SamplingSchedule.arena.startTau)
    @AppStorage("arFloorTau")
    private var arFloorTau: Double = Double(SamplingSchedule.arena.floorTau)
    @AppStorage("arDecayPerPly")
    private var arDecayPerPly: Double = Double(SamplingSchedule.arena.decayPerPly)
    /// Last auto-computed step delay, persisted so the next session
    /// starts from where the auto-adjuster left off instead of
    /// falling back to the manual default.
    @AppStorage("lastAutoComputedDelayMs") private var lastAutoComputedDelayMs: Int = 50
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
    @State private var activeTrainingAlarm: TrainingAlarm?
    @State private var trainingAlarmSilenced = false
    @State private var divergenceWarningStreak = 0
    @State private var divergenceCriticalStreak = 0
    @State private var divergenceRecoveryStreak = 0
    @State private var alarmSoundTask: Task<Void, Never>?

    /// Weak-captured reference to the NSWindow hosting this view.
    /// Set by `WindowAccessor` on first appearance; used by the
    /// `NSWindow.willCloseNotification` filter so teardown only
    /// fires when THIS window closes, not an auxiliary one.
    @State private var contentWindow: NSWindow?

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

    // MARK: - Control side-effects
    //
    // The Play-and-Train Board picker, Probe picker, Concurrency
    // Stepper, Replay-Ratio target Stepper, and Auto toggle all used
    // to route their writes through `Binding(get:set:)` computed
    // properties on this view. Those allocated a fresh `Binding`
    // struct on every `body` invocation, which the SwiftUI → AppKit
    // bridge treats as "binding changed" and pays to re-wire the
    // underlying `NSSegmentedControl` / `NSStepper` / `NSButton`
    // (~18 ms each for the segmented control, per Instruments) on
    // every 100 ms heartbeat-driven body render. They're now bound
    // directly to their stored `@State` / `@AppStorage` projected
    // values (stable identity), with side effects hoisted into the
    // `controlSideEffectsProbe` helper below. `.onChange` fires only
    // on real value changes, so the "if newValue != current" guards
    // in the old setters are inherently unnecessary here.
    //
    // The 5 `.onChange` modifiers live on a zero-sized hidden view
    // attached via `.background(controlSideEffectsProbe)` rather
    // than being chained onto `body`'s tail, because tacking them
    // onto the already-long modifier chain tipped the Swift
    // type-checker past its "reasonable time" threshold. Attaching
    // them to a minimal child view keeps each chain short enough
    // for the checker while preserving the same observation
    // semantics.
    //
    // Two computed bindings remain below: `trainingStepDelayBinding`
    // (snaps raw Stepper deltas onto a discrete ladder — the set
    // logic depends on the *direction* of change relative to the
    // current ladder index and can't be expressed as a pure `.onChange`
    // side effect without reintroducing ping-pong) and
    // `sideToMoveBinding` (only visible outside Play-and-Train, so
    // isn't in the heartbeat render path that motivated this refactor).
    @ViewBuilder
    private var controlSideEffectsProbe: some View {
        Color.clear
            .frame(width: 0, height: 0)
            .onChange(of: playAndTrainBoardMode) { _, newValue in
                // Flipping to Candidate-test marks the probe dirty so
                // the driver fires an immediate forward pass on the
                // next gap — otherwise the user would wait up to 15s
                // for the interval probe to trigger on the new mode.
                if newValue == .candidateTest {
                    candidateProbeDirty = true
                }
            }
            .onChange(of: probeNetworkTarget) { _, _ in
                // Flipping the probe target marks the probe dirty so
                // the next driver gap fires a fresh evaluation against
                // the newly-selected network instead of showing stale
                // results.
                candidateProbeDirty = true
            }
            .onChange(of: selfPlayWorkerCount) { oldValue, newValue in
                // Stepper's `in:` range clamps; `.onChange` only fires
                // on real change, so log + box update happen only when
                // N actually shifts. `workerCountBox` is nil between
                // sessions, so out-of-session writes just update the
                // @State and take effect on the next session start.
                SessionLogger.shared.log("[PARAM] selfPlayWorkers: \(oldValue) -> \(newValue)")
                workerCountBox?.set(newValue)
            }
            .onChange(of: replayRatioTarget) { oldValue, newValue in
                SessionLogger.shared.log(
                    String(format: "[PARAM] replayRatioTarget: %.2f -> %.2f", oldValue, newValue)
                )
                replayRatioController?.targetRatio = newValue
            }
            .onChange(of: replayRatioAutoAdjust) { oldValue, newValue in
                SessionLogger.shared.log("[PARAM] replayRatioAutoAdjust: \(oldValue) -> \(newValue)")
                replayRatioController?.autoAdjust = newValue
                if newValue {
                    // Auto ON: seed the computed delay from the current
                    // manual delay so the display doesn't jump.
                    replayRatioController?.computedDelayMs = trainingStepDelayMs
                } else {
                    // Auto OFF: inherit the last auto-computed delay
                    // as the new manual value, snapped to the nearest
                    // ladder rung so the Stepper binding doesn't crash
                    // on an off-ladder value.
                    let lastAuto = replayRatioController?.computedDelayMs ?? trainingStepDelayMs
                    let ladder = Self.stepDelayLadder
                    let snapped = ladder.min(by: { abs($0 - lastAuto) < abs($1 - lastAuto) }) ?? lastAuto
                    trainingStepDelayMs = snapped
                    replayRatioController?.manualDelayMs = snapped
                }
            }
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
                // Find the current value in the ladder. If it's not
                // an exact rung (e.g. inherited from the auto-adjuster
                // which computes arbitrary ms values), snap to the
                // nearest rung before stepping up or down.
                let currentIdx: Int
                if let exact = ladder.firstIndex(of: current) {
                    currentIdx = exact
                } else {
                    currentIdx = ladder.enumerated().min(by: {
                        abs($0.element - current) < abs($1.element - current)
                    })?.offset ?? 0
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
                if snapped != current {
                    SessionLogger.shared.log("[PARAM] stepDelayMs (manual): \(current) -> \(snapped)")
                }
                trainingStepDelayMs = snapped
                trainingStepDelayBox?.set(snapped)
                replayRatioController?.manualDelayMs = snapped
            }
        )
    }

    /// Scratch string for the learning rate text field. Seeded from
    /// the trainer's current LR when Play-and-Train starts; the user
    /// edits freely without the binding reformatting mid-keystroke.
    /// The value is parsed and applied only on Enter (via `.onSubmit`
    /// on the TextField). Invalid input reverts to the current LR.
    @State private var learningRateEditText: String = ""
    @State private var entropyRegularizationEditText: String = ""
    @State private var drawPenaltyEditText: String = ""
    @State private var weightDecayEditText: String = ""
    @State private var gradClipMaxNormEditText: String = ""
    @State private var policyScaleKEditText: String = ""
    @State private var spStartTauEditText: String = ""
    @State private var spFloorTauEditText: String = ""
    @State private var spDecayPerPlyEditText: String = ""
    @State private var arStartTauEditText: String = ""
    @State private var arFloorTauEditText: String = ""
    @State private var arDecayPerPlyEditText: String = ""

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
        VStack(alignment: .leading, spacing: 8) {
            // Title bar
            HStack(spacing: 8) {
                Text("Drew's Chess Machine")
                    .font(.title2)
                    .fontWeight(.semibold)
                // Build banner — bumped from .caption/.tertiary to
                // .callout/.secondary so the build number, git hash,
                // and date are readable at a glance instead of squinting.
                Text(BuildInfo.summary)
                    .font(.callout)
                    .foregroundStyle(.secondary)
                Button(action: { showingInfoPopover.toggle() }) {
                    Image(systemName: "info.circle")
                        .font(.title3)
                }
                .buttonStyle(.plain)
                .popover(isPresented: $showingInfoPopover) {
                    VStack(alignment: .leading, spacing: 12) {
                        Text("About Drew's Chess Machine")
                            .font(.headline)
                        Text("Forward pass through a ~2.4M parameter convolutional network using MPSGraph on the GPU. Weights are randomly initialized (He initialization) — no training has occurred.")
                            .font(.callout)
                        Divider()
                        Text("Architecture: 20×8×8 input → stem(128) → 8 res+SE blocks → policy(4864) + value(1)")
                            .font(.system(.callout, design: .monospaced))
                        Text("Parameters: ~2,400,000 (~2.4M)")
                            .font(.system(.callout, design: .monospaced))
                        if let net = network {
                            Text("Network ID: \(net.identifier?.description ?? "–")")
                                .font(.system(.callout, design: .monospaced))
                            Text("Build time: \(String(format: "%.1f ms", net.buildTimeMs))")
                                .font(.system(.callout, design: .monospaced))
                        }
                    }
                    .padding(16)
                    .frame(width: 500)
                }
                Spacer()
                // Right-side ID + network status — bumped from .caption to
                // .callout so they're readable at glance distance. Contrast
                // (.secondary) was already fine; only the size changes.
                if let net = network {
                    Text("ID: \(net.identifier?.description ?? "–")")
                        .font(.callout)
                        .foregroundStyle(.secondary)
                }
                Text(networkStatus.isEmpty ? "" : networkStatus.components(separatedBy: "\n").first ?? "")
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }

            if let alarm = activeTrainingAlarm {
                HStack(spacing: 12) {
                    VStack(alignment: .leading, spacing: 2) {
                        // Title forced black against the yellow background
                        // for legibility — default `.headline` color in
                        // dark-mode SwiftUI is white, which washes out
                        // against `.yellow.opacity(0.8)`.
                        Text(alarm.title)
                            .font(.headline)
                            .foregroundStyle(Color.black)
                        // Detail text uses a darker red + medium weight so
                        // numeric values in the alarm (entropy, gNorm) read
                        // clearly against the yellow background instead of
                        // washing out as default `.red`.
                        Text(alarm.detail)
                            .font(.callout.weight(.medium))
                            .foregroundStyle(Color(red: 0.55, green: 0.0, blue: 0.0))
                    }
                    Spacer()
                    HStack(spacing: 8) {
                        if trainingAlarmSilenced {
                            Text("Silenced")
                                .font(.caption)
                                .foregroundStyle(Color.black)
                        } else {
                            Button("Silence") {
                                silenceTrainingAlarm()
                            }
                        }
                        // Dismiss clears the banner AND resets the
                        // streak counters, so an alarm only re-raises
                        // if the condition deteriorates fresh from a
                        // healthy baseline. Silence keeps visibility
                        // (banner stays) but quiets the sound; Dismiss
                        // is "I've seen it, start over."
                        Button("Dismiss") {
                            dismissTrainingAlarm()
                        }
                    }
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .background(Color.yellow.opacity(0.8))
                .clipShape(RoundedRectangle(cornerRadius: 8))
            }

            cumulativeStatusBar

            // Status row — the action controls previously lived here
            // but have all moved to the File / Train / Debug menus at
            // the top of the screen. This row now just shows the busy
            // progress indicator and any ephemeral checkpoint status
            // message. Kept as a view so the `.fileImporter` modifiers
            // below (driven by `showingLoadSessionImporter` /
            // `showingLoadModelImporter`, which the menu Load items
            // toggle) have a stable parent to attach to.
            HStack(spacing: 8) {
                if isBusy {
                    ProgressView().controlSize(.small)
                    busyLabelView
                }
                if let msg = checkpointStatusMessage {
                    Text(msg)
                        .font(.callout)
                        .foregroundStyle(checkpointStatusIsError ? .red : .secondary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                }
                Spacer(minLength: 0)
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
                        Picker("Board", selection: $playAndTrainBoardMode) {
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

                    if isCandidateTestActive {
                        Picker("Probe", selection: $probeNetworkTarget) {
                            Text("Candidate").tag(ProbeNetworkTarget.candidate)
                            Text("Champion").tag(ProbeNetworkTarget.champion)
                        }
                        .pickerStyle(.segmented)
                        .labelsHidden()
                        .frame(maxWidth: 220)
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

                            let rightDisabled = inferenceResult == nil || selectedOverlay == ChessNetwork.inputPlanes || !showForwardPassUI
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
                                                value: $selfPlayWorkerCount,
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
                                                value: $replayRatioTarget,
                                                in: 0.1...5.0,
                                                step: 0.1
                                            )
                                            .labelsHidden()
                                            Toggle("Auto", isOn: $replayRatioAutoAdjust)
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
                                                        let prior = trainer?.learningRate ?? Float(Self.trainerLearningRateDefault)
                                                        if abs(parsed - prior) > Float.ulpOfOne {
                                                            SessionLogger.shared.log(
                                                                String(format: "[PARAM] learningRate: %.1e -> %.1e", prior, parsed)
                                                            )
                                                        }
                                                        trainer?.learningRate = parsed
                                                        trainerLearningRate = Double(parsed)
                                                    }
                                                    learningRateEditText = String(
                                                        format: "%.1e",
                                                        trainer?.learningRate ?? Self.trainerLearningRateDefault
                                                    )
                                                }
                                        }
                                        HStack(spacing: 6) {
                                            Text("  Entropy Reg:")
                                            TextField("Entropy Reg", text: $entropyRegularizationEditText)
                                            .monospacedDigit()
                                            .frame(width: 80)
                                            .textFieldStyle(.roundedBorder)
                                            .onSubmit {
                                                if let parsed = Double(entropyRegularizationEditText),
                                                   parsed >= 0, parsed.isFinite {
                                                    let prior = entropyRegularizationCoeff
                                                    if abs(parsed - prior) > Double.ulpOfOne {
                                                        SessionLogger.shared.log(
                                                            String(format: "[PARAM] entropyReg: %.2e -> %.2e", prior, parsed)
                                                        )
                                                    }
                                                    entropyRegularizationCoeff = parsed
                                                    trainer?.entropyRegularizationCoeff = Float(parsed)
                                                }
                                                entropyRegularizationEditText = String(
                                                    format: "%.2e",
                                                    entropyRegularizationCoeff
                                                )
                                            }
                                            Text("clip")
                                                .foregroundStyle(.secondary)
                                            TextField("Clip", text: $gradClipMaxNormEditText)
                                                .monospacedDigit()
                                                .frame(width: 70)
                                                .textFieldStyle(.roundedBorder)
                                                .onSubmit {
                                                    if let parsed = Double(gradClipMaxNormEditText),
                                                       parsed > 0, parsed.isFinite {
                                                        let prior = gradClipMaxNorm
                                                        if abs(parsed - prior) > Double.ulpOfOne {
                                                            SessionLogger.shared.log(
                                                                String(format: "[PARAM] gradClipMaxNorm: %.2f -> %.2f", prior, parsed)
                                                            )
                                                        }
                                                        gradClipMaxNorm = parsed
                                                        trainer?.gradClipMaxNorm = Float(parsed)
                                                    }
                                                    gradClipMaxNormEditText = String(
                                                        format: "%.2f",
                                                        gradClipMaxNorm
                                                    )
                                                }
                                            Text("decay")
                                                .foregroundStyle(.secondary)
                                            TextField("Decay", text: $weightDecayEditText)
                                                .monospacedDigit()
                                                .frame(width: 80)
                                                .textFieldStyle(.roundedBorder)
                                                .onSubmit {
                                                    if let parsed = Double(weightDecayEditText),
                                                       parsed >= 0, parsed.isFinite {
                                                        let prior = weightDecayC
                                                        if abs(parsed - prior) > Double.ulpOfOne {
                                                            SessionLogger.shared.log(
                                                                String(format: "[PARAM] weightDecayC: %.2e -> %.2e", prior, parsed)
                                                            )
                                                        }
                                                        weightDecayC = parsed
                                                        trainer?.weightDecayC = Float(parsed)
                                                    }
                                                    weightDecayEditText = String(
                                                        format: "%.2e",
                                                        weightDecayC
                                                    )
                                                }
                                            Text("K")
                                                .foregroundStyle(.secondary)
                                            TextField("K", text: $policyScaleKEditText)
                                                .monospacedDigit()
                                                .frame(width: 60)
                                                .textFieldStyle(.roundedBorder)
                                                .onSubmit {
                                                    if let parsed = Double(policyScaleKEditText),
                                                       parsed > 0, parsed.isFinite {
                                                        let prior = policyScaleK
                                                        if abs(parsed - prior) > Double.ulpOfOne {
                                                            SessionLogger.shared.log(
                                                                String(format: "[PARAM] policyScaleK: %.2f -> %.2f", prior, parsed)
                                                            )
                                                        }
                                                        policyScaleK = parsed
                                                        trainer?.policyScaleK = Float(parsed)
                                                    }
                                                    policyScaleKEditText = String(
                                                        format: "%.2f",
                                                        policyScaleK
                                                    )
                                                }
                                        }
                                        HStack(spacing: 6) {
                                            Text("  SP tau:")
                                            TextField("SP start", text: $spStartTauEditText)
                                                .monospacedDigit()
                                                .frame(width: 60)
                                                .textFieldStyle(.roundedBorder)
                                                .onSubmit { applySpTauEdit() }
                                            Text("→")
                                                .foregroundStyle(.secondary)
                                            TextField("SP floor", text: $spFloorTauEditText)
                                                .monospacedDigit()
                                                .frame(width: 60)
                                                .textFieldStyle(.roundedBorder)
                                                .onSubmit { applySpTauEdit() }
                                            Text("decay")
                                                .foregroundStyle(.secondary)
                                            TextField("SP decay", text: $spDecayPerPlyEditText)
                                                .monospacedDigit()
                                                .frame(width: 70)
                                                .textFieldStyle(.roundedBorder)
                                                .onSubmit { applySpTauEdit() }
                                            Text("/ply")
                                                .foregroundStyle(.secondary)
                                        }
                                        HStack(spacing: 6) {
                                            Text("  Arena tau:")
                                            TextField("Ar start", text: $arStartTauEditText)
                                                .monospacedDigit()
                                                .frame(width: 60)
                                                .textFieldStyle(.roundedBorder)
                                                .onSubmit { applyArTauEdit() }
                                            Text("→")
                                                .foregroundStyle(.secondary)
                                            TextField("Ar floor", text: $arFloorTauEditText)
                                                .monospacedDigit()
                                                .frame(width: 60)
                                                .textFieldStyle(.roundedBorder)
                                                .onSubmit { applyArTauEdit() }
                                            Text("decay")
                                                .foregroundStyle(.secondary)
                                            TextField("Ar decay", text: $arDecayPerPlyEditText)
                                                .monospacedDigit()
                                                .frame(width: 70)
                                                .textFieldStyle(.roundedBorder)
                                                .onSubmit { applyArTauEdit() }
                                            Text("/ply")
                                                .foregroundStyle(.secondary)
                                        }
                                        HStack(spacing: 6) {
                                            Text("  Draw Penalty:")
                                            TextField("Draw Penalty", text: $drawPenaltyEditText)
                                            .monospacedDigit()
                                            .frame(width: 80)
                                            .textFieldStyle(.roundedBorder)
                                            .onSubmit {
                                                if let parsed = Double(drawPenaltyEditText),
                                                   parsed >= 0, parsed.isFinite {
                                                    let prior = drawPenalty
                                                    if abs(parsed - prior) > Double.ulpOfOne {
                                                        SessionLogger.shared.log(
                                                            String(format: "[PARAM] drawPenalty: %.3f -> %.3f", prior, parsed)
                                                        )
                                                    }
                                                    drawPenalty = parsed
                                                    trainer?.drawPenalty = Float(parsed)
                                                }
                                                drawPenaltyEditText = String(
                                                    format: "%.3f",
                                                    drawPenalty
                                                )
                                            }
                                            Text("(draws → z = −penalty; 0 disables)")
                                                .foregroundStyle(.secondary)
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
                    ForEach(0..<ChessNetwork.inputPlanes, id: \.self) { channel in
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

            // Chart grid — always visible during Play and Train,
            // showing training metrics over time.
            if realTraining {
                Divider()
                TrainingChartGridView(
                    progressRateSamples: progressRateSamples,
                    trainingChartSamples: trainingChartSamples,
                    diversityHistogram: currentDiversityHistogramBars,
                    arenaEvents: arenaChartEvents,
                    activeArenaStartElapsed: activeArenaStartElapsed,
                    promoteThreshold: Self.tournamentPromoteThreshold,
                    appMemoryTotalGB: memoryStatsSnap.map { Double($0.gpuTotalBytes) / (1024 * 1024 * 1024) },
                    gpuMemoryTotalGB: memoryStatsSnap.map { Double($0.gpuTotalBytes) / (1024 * 1024 * 1024) },
                    visibleDomainSec: Self.progressRateVisibleDomainSec,
                    scrollX: $progressRateScrollX
                )
            }
        }
        .padding(16)
        .frame(minWidth: 1400, minHeight: 780)
        .focusable()
        .focusEffectDisabled()
        .onKeyPress(.leftArrow) { navigateOverlay(-1); return .handled }
        .onKeyPress(.rightArrow) { navigateOverlay(1); return .handled }
        .background(WindowAccessor(window: $contentWindow))
        .onAppear {
            wireMenuCommandHub()
            syncMenuCommandHubState()
            if learningRateEditText.isEmpty {
                learningRateEditText = String(format: "%.1e", trainerLearningRate)
            }
            if entropyRegularizationEditText.isEmpty {
                entropyRegularizationEditText = String(format: "%.2e", entropyRegularizationCoeff)
            }
            if drawPenaltyEditText.isEmpty {
                drawPenaltyEditText = String(format: "%.3f", drawPenalty)
            }
            if weightDecayEditText.isEmpty {
                weightDecayEditText = String(format: "%.2e", weightDecayC)
            }
            if gradClipMaxNormEditText.isEmpty {
                gradClipMaxNormEditText = String(format: "%.2f", gradClipMaxNorm)
            }
            if policyScaleKEditText.isEmpty {
                policyScaleKEditText = String(format: "%.2f", policyScaleK)
            }
            if spStartTauEditText.isEmpty {
                spStartTauEditText = String(format: "%.2f", spStartTau)
            }
            if spFloorTauEditText.isEmpty {
                spFloorTauEditText = String(format: "%.2f", spFloorTau)
            }
            if spDecayPerPlyEditText.isEmpty {
                spDecayPerPlyEditText = String(format: "%.3f", spDecayPerPly)
            }
            if arStartTauEditText.isEmpty {
                arStartTauEditText = String(format: "%.2f", arStartTau)
            }
            if arFloorTauEditText.isEmpty {
                arFloorTauEditText = String(format: "%.2f", arFloorTau)
            }
            if arDecayPerPlyEditText.isEmpty {
                arDecayPerPlyEditText = String(format: "%.3f", arDecayPerPly)
            }
        }
        .onReceive(NotificationCenter.default.publisher(for: NSWindow.willCloseNotification)) { note in
            // Teardown on actual main-window close only — NOT on minimize
            // (which fires .onDisappear but not willClose), and NOT on
            // auxiliary windows (Log Analysis, NSOpenPanel/NSSavePanel,
            // etc.) which all post the same notification. Users who
            // deliberately minimize a training session still expect it
            // to keep running; and raising a Save panel must not abort
            // the run mid-save. The object check narrows us to exactly
            // this view's hosting NSWindow, captured above by
            // WindowAccessor.
            guard let closing = note.object as? NSWindow,
                  let ours = contentWindow,
                  closing === ours else {
                return
            }
            stopAnyContinuous()
            clearTrainingAlarm()
        }
        .onChange(of: isBuilding) { _, _ in syncMenuCommandHubState() }
        .onChange(of: continuousPlay) { _, _ in syncMenuCommandHubState() }
        .onChange(of: continuousTraining) { _, _ in syncMenuCommandHubState() }
        .onChange(of: sweepRunning) { _, _ in syncMenuCommandHubState() }
        .onChange(of: realTraining) { _, _ in syncMenuCommandHubState() }
        .onChange(of: isArenaRunning) { _, _ in syncMenuCommandHubState() }
        .onChange(of: checkpointSaveInFlight) { _, _ in syncMenuCommandHubState() }
        .onChange(of: isTrainingOnce) { _, _ in syncMenuCommandHubState() }
        .onChange(of: isEvaluating) { _, _ in syncMenuCommandHubState() }
        .onChange(of: gameSnapshot.isPlaying) { _, _ in syncMenuCommandHubState() }
        .onChange(of: network != nil) { _, _ in syncMenuCommandHubState() }
        .onChange(of: pendingLoadedSession != nil) { _, _ in syncMenuCommandHubState() }
        .background(controlSideEffectsProbe)
        .onChange(of: progressRateScrollX) { _, newValue in
            // Flip off follow-latest when the user scrolls backward.
            // Auto-follow writes `progressRateScrollX` to
            // `latestScrollX`, leaving follow on. A user-initiated
            // backward scroll lands far from latestScrollX and turns
            // follow off so the 1 Hz sampler stops dragging the chart
            // back to the right edge.
            //
            // Lives here (on a persistent view parent) instead of a
            // custom `Binding(get:set:)` because that binding was
            // getting recreated on every body render, handing Swift
            // Charts a fresh scroll-config each time and tripping
            // `onChange(of: ChartScrollPositionConfiguration) action
            // tried to update multiple times per frame` warnings as
            // the histogram state churned.
            Task { @MainActor in
                let latest = progressRateSamples.last?.elapsedSec ?? 0
                let latestScrollX = max(0, latest - Self.progressRateVisibleDomainSec)
                let shouldFollow = abs(newValue - latestScrollX) < 1.0
                if progressRateFollowLatest != shouldFollow {
                    progressRateFollowLatest = shouldFollow
                }
            }
        }
        .onReceive(snapshotTimer) { _ in
            // Defer every @State mutation driven by the 100 ms
            // heartbeat to the next main-actor runloop tick. The
            // timer publisher fires on the main thread and SwiftUI
            // flags "update multiple times per frame" warnings (and
            // measurable hangs) when onReceive synchronously pushes
            // several dozen state-change notifications inline. A
            // `Task { @MainActor in }` wrap coalesces the work into a
            // single render pass.
            Task { @MainActor in
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
                // Replay-ratio snapshot for the UI. Persist the auto-
                // computed delay so the next session starts from where
                // the adjuster left off.
                if let rc = replayRatioController {
                    let snap = rc.snapshot()
                    replayRatioSnapshot = snap
                    if snap.autoAdjust {
                        lastAutoComputedDelayMs = snap.computedDelayMs
                    }
                }
                // Diversity-histogram mirror. Read once per heartbeat off
                // the tracker's thread-safe snapshot. Only push into
                // @State when the bucket totals actually change (or the
                // bar array is currently empty) so SwiftUI doesn't
                // invalidate the chart every tick for a stable reading.
                if let tracker = selfPlayDiversityTracker {
                    let divSnap = tracker.snapshot()
                    let labels = GameDiversityTracker.histogramLabels
                    var newBars: [DiversityHistogramBar] = []
                    newBars.reserveCapacity(divSnap.divergenceHistogram.count)
                    for (idx, count) in divSnap.divergenceHistogram.enumerated()
                    where idx < labels.count {
                        newBars.append(DiversityHistogramBar(
                            id: idx,
                            label: labels[idx],
                            count: count
                        ))
                    }
                    let changed = newBars.count != currentDiversityHistogramBars.count
                    || zip(newBars, currentDiversityHistogramBars)
                        .contains { $0.0.count != $0.1.count }
                    if changed {
                        currentDiversityHistogramBars = newBars
                    }
                }
            }  // Task @MainActor
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
    /// Sample training metrics at the same 1Hz cadence as the
    /// progress rate sampler. Appends a `TrainingChartSample`
    /// with rolling loss, entropy, ratio, and non-neg count.
    /// Append a training chart sample. Called from inside
    /// `refreshProgressRateIfNeeded` at the same 1Hz cadence.
    private func refreshTrainingChartIfNeeded() {
        let now = Date()
        // Use the parallel-worker stats box's `sessionStart` (fresh
        // `Date()` at Play-and-Train start, including after a resume)
        // rather than `currentSessionStart`, which is back-dated on
        // resume by the loaded session's `elapsedTrainingSec` so
        // persistence can accumulate elapsed time across save/resume
        // cycles. Using the back-dated anchor for chart samples puts
        // their `elapsedSec` thousands of seconds ahead of the
        // progress-rate samples (which use the fresh anchor), which
        // drives the shared `scrollX` binding to the progress-rate
        // coordinate space and parks every training chart's data
        // outside the visible window on resumed sessions.
        let sessionStart = parallelStats?.sessionStart ?? currentSessionStart ?? now
        let elapsed = max(0, now.timeIntervalSince(sessionStart))
        let trainingSnap = trainingBox?.snapshot()
        let ratioSnap = replayRatioSnapshot

        let appMemMB = memoryStatsSnap.map { Double($0.appFootprintBytes) / (1024 * 1024) }
        let gpuMemMB = memoryStatsSnap.map { Double($0.gpuAllocatedBytes) / (1024 * 1024) }
        // GPU busy %: fraction of the last 1-second sample interval
        // that the GPU was actively running training steps. Computed
        // from the delta of cumulative GPU ms in TrainingRunStats.
        let currentGpuMs = trainingSnap?.stats.totalGpuMs ?? 0
        let gpuDeltaMs = max(0, currentGpuMs - prevChartTotalGpuMs)
        let gpuBusy = gpuDeltaMs / 10.0 // delta ms / 1000ms * 100%
        prevChartTotalGpuMs = currentGpuMs
        // Power + thermal state read straight from ProcessInfo at
        // sample time. Both are cheap property reads, and both can
        // change between samples without any polling overhead on our
        // part (the OS tracks them). Captured here so the chart
        // tile can render a continuous step trace rather than
        // sampling on hover.
        let pi = ProcessInfo.processInfo
        let sample = TrainingChartSample(
            id: trainingChartNextId,
            elapsedSec: elapsed,
            rollingPolicyLoss: trainingSnap?.rollingPolicyLoss,
            rollingValueLoss: trainingSnap?.rollingValueLoss,
            rollingPolicyEntropy: trainingSnap?.rollingPolicyEntropy,
            rollingPolicyNonNegCount: trainingSnap?.rollingPolicyNonNegCount,
            rollingGradNorm: trainingSnap?.rollingGradGlobalNorm,
            replayRatio: ratioSnap?.currentRatio,
            cpuPercent: cpuPercent,
            gpuBusyPercent: trainingSnap != nil ? gpuBusy : nil,
            gpuMemoryMB: gpuMemMB,
            appMemoryMB: appMemMB,
            lowPowerMode: pi.isLowPowerModeEnabled,
            thermalState: pi.thermalState
        )
        trainingChartSamples.append(sample)
        trainingChartNextId += 1
        evaluateTrainingAlarm(from: sample)
    }

    private func evaluateTrainingAlarm(from sample: TrainingChartSample) {
        let entropy = sample.rollingPolicyEntropy
        let gradNorm = sample.rollingGradNorm
        let warningOutOfLine =
            (entropy.map { $0 < Self.policyEntropyAlarmThreshold } ?? false)
            && (gradNorm.map { $0 > Self.divergenceAlarmGradNormWarningThreshold } ?? false)
        let criticalOutOfLine =
            (entropy.map { $0 < Self.divergenceAlarmEntropyCriticalThreshold } ?? false)
            || (gradNorm.map { $0 > Self.divergenceAlarmGradNormCriticalThreshold } ?? false)

        if criticalOutOfLine {
            divergenceCriticalStreak += 1
            divergenceWarningStreak = 0
            divergenceRecoveryStreak = 0
        } else if warningOutOfLine {
            divergenceWarningStreak += 1
            divergenceCriticalStreak = 0
            divergenceRecoveryStreak = 0
        } else {
            divergenceCriticalStreak = 0
            divergenceWarningStreak = 0
            divergenceRecoveryStreak += 1
        }

        if divergenceCriticalStreak >= Self.divergenceAlarmConsecutiveCriticalSamples {
            raiseTrainingAlarm(
                severity: .critical,
                title: "Critical Training Divergence",
                detail: alarmDetail(entropy: entropy, gradNorm: gradNorm)
            )
        } else if divergenceWarningStreak >= Self.divergenceAlarmConsecutiveWarningSamples {
            raiseTrainingAlarm(
                severity: .warning,
                title: "Training Divergence Warning",
                detail: alarmDetail(entropy: entropy, gradNorm: gradNorm)
            )
        } else if divergenceRecoveryStreak >= Self.divergenceAlarmRecoverySamples {
            clearTrainingAlarm()
        }
    }

    private func alarmDetail(entropy: Double?, gradNorm: Double?) -> String {
        let entropyStr = entropy.map { String(format: "%.4f", $0) } ?? "--"
        let gradStr = gradNorm.map { String(format: "%.3f", $0) } ?? "--"
        return "policy entropy=\(entropyStr), gNorm=\(gradStr)"
    }

    private func raiseTrainingAlarm(
        severity: TrainingAlarm.Severity,
        title: String,
        detail: String
    ) {
        let next = TrainingAlarm(
            id: UUID(),
            severity: severity,
            title: title,
            detail: detail,
            raisedAt: Date()
        )
        let isNewAlarm = activeTrainingAlarm == nil
        let titleOrSeverityChanged = activeTrainingAlarm?.title != next.title
            || activeTrainingAlarm?.severity != next.severity
        activeTrainingAlarm = next
        // Log on first raise OR on title/severity change so the session
        // log captures every banner state the user could see. Periodic
        // re-raises with identical title+severity (and just updated
        // numeric detail) don't relog — those are already covered by
        // the periodic [STATS] / threshold-alarm log lines.
        if isNewAlarm || titleOrSeverityChanged {
            SessionLogger.shared.log("[ALARM] \(title): \(detail)")
        }
        startAlarmSoundLoopIfNeeded()
    }

    private func clearTrainingAlarm() {
        if let prior = activeTrainingAlarm {
            SessionLogger.shared.log("[ALARM] cleared: \(prior.title)")
        }
        activeTrainingAlarm = nil
        trainingAlarmSilenced = false
        alarmSoundTask?.cancel()
        alarmSoundTask = nil
    }

    private func silenceTrainingAlarm() {
        if let active = activeTrainingAlarm {
            SessionLogger.shared.log("[ALARM] silenced: \(active.title)")
        }
        trainingAlarmSilenced = true
        alarmSoundTask?.cancel()
        alarmSoundTask = nil
    }

    /// Clear the banner AND reset the divergence streak counters so
    /// the alarm only re-raises on a *fresh* deterioration from a
    /// healthy baseline. Different from `clearTrainingAlarm()`, which
    /// is the auto-clear path triggered by the recovery streak (and
    /// leaves the warning/critical streaks alone). User-initiated
    /// "I've seen it, move on" gesture.
    private func dismissTrainingAlarm() {
        if let active = activeTrainingAlarm {
            SessionLogger.shared.log("[ALARM] dismissed: \(active.title)")
        }
        activeTrainingAlarm = nil
        trainingAlarmSilenced = false
        alarmSoundTask?.cancel()
        alarmSoundTask = nil
        divergenceWarningStreak = 0
        divergenceCriticalStreak = 0
        divergenceRecoveryStreak = 0
    }

    private func startAlarmSoundLoopIfNeeded() {
        guard activeTrainingAlarm != nil, !trainingAlarmSilenced, alarmSoundTask == nil else { return }
        alarmSoundTask = Task {
            while !Task.isCancelled {
                await playAlarmBuzzBurst()
                do {
                    try await Task.sleep(for: .seconds(300))
                } catch {
                    return
                }
            }
        }
    }

    @MainActor
    private func playAlarmBuzzBurst() async {
        for _ in 0..<3 {
            if Task.isCancelled || activeTrainingAlarm == nil || trainingAlarmSilenced { return }
            NSSound.beep()
            do {
                try await Task.sleep(for: .seconds(1.2))
            } catch {
                return
            }
        }
    }

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
        // Append a training chart sample at the same 1Hz cadence.
        refreshTrainingChartIfNeeded()
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
        // Hover readout: when the user moves the cursor over the
        // chart, display the elapsed-time + all three series values
        // at that time in an overlaid label. Nearest-sample lookup
        // is gated by TrainingChartGridView.hoverMatchToleranceSec
        // so a hover past the last sample (or before the first one)
        // is reported as "no data" rather than silently snapping to
        // the nearest boundary sample and misleading the reader.
        enum BigProgressReadout {
            case hoveringNoData(hoveredTime: Double)
            case hoveringWithData(time: Double, combined: Double, selfPlay: Double, training: Double)
        }
        let hoverReadout: BigProgressReadout? = {
            guard let t = bigProgressChartHoveredSec else { return nil }
            guard !progressRateSamples.isEmpty else {
                return .hoveringNoData(hoveredTime: t)
            }
            var best = progressRateSamples[0]
            var bestDist = Swift.abs(best.elapsedSec - t)
            for s in progressRateSamples.dropFirst() {
                let d = Swift.abs(s.elapsedSec - t)
                if d < bestDist { best = s; bestDist = d }
            }
            if bestDist > TrainingChartGridView.hoverMatchToleranceSec {
                return .hoveringNoData(hoveredTime: t)
            }
            return .hoveringWithData(
                time: best.elapsedSec,
                combined: best.combinedMovesPerHour,
                selfPlay: best.selfPlayMovesPerHour,
                training: best.trainingMovesPerHour
            )
        }()

        // One ForEach per series — SwiftUI Charts only connects
        // LineMarks that share a single enclosing ForEach AND a
        // single Y value. Packing all three series into ONE
        // ForEach made Charts emit spurious thin lines near y=0
        // because it couldn't disambiguate which LineMarks
        // belonged to which logical series within the shared
        // iteration. Splitting per series restores the canonical
        // multi-line rendering.
        return Chart {
            ForEach(progressRateSamples) { sample in
                LineMark(
                    x: .value("Elapsed", sample.elapsedSec),
                    y: .value("Moves/hr", sample.combinedMovesPerHour)
                )
                .foregroundStyle(by: .value("Series", "Combined"))
            }
            ForEach(progressRateSamples) { sample in
                LineMark(
                    x: .value("Elapsed", sample.elapsedSec),
                    y: .value("Moves/hr", sample.selfPlayMovesPerHour)
                )
                .foregroundStyle(by: .value("Series", "Self-play"))
            }
            ForEach(progressRateSamples) { sample in
                LineMark(
                    x: .value("Elapsed", sample.elapsedSec),
                    y: .value("Moves/hr", sample.trainingMovesPerHour)
                )
                .foregroundStyle(by: .value("Series", "Training"))
            }
            if let t = bigProgressChartHoveredSec {
                RuleMark(x: .value("Elapsed", t))
                    .foregroundStyle(Color.gray.opacity(0.5))
                    .lineStyle(StrokeStyle(lineWidth: 1))
            }
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
        .chartScrollPosition(x: $progressRateScrollX)
        .chartOverlay { proxy in
            // Transparent hover-capture rectangle over the plot
            // area — same pattern as `TrainingChartGridView`'s
            // `hoverOverlay` helper but inline here because this
            // chart lives in ContentView's body.
            GeometryReader { geo in
                ZStack(alignment: .topLeading) {
                    Rectangle()
                        .fill(Color.clear)
                        .contentShape(Rectangle())
                        .onContinuousHover { phase in
                            switch phase {
                            case .active(let point):
                                let origin = (proxy.plotFrame.map { geo[$0].origin } ?? .zero)
                                let xInPlot = point.x - origin.x
                                if let sec: Double = proxy.value(atX: xInPlot) {
                                    if sec < 0 {
                                        if bigProgressChartHoveredSec != nil {
                                            bigProgressChartHoveredSec = nil
                                        }
                                        return
                                    }
                                    if bigProgressChartHoveredSec != sec {
                                        bigProgressChartHoveredSec = sec
                                    }
                                }
                            case .ended:
                                if bigProgressChartHoveredSec != nil {
                                    bigProgressChartHoveredSec = nil
                                }
                            }
                        }
                    Group {
                        switch hoverReadout ?? .hoveringNoData(hoveredTime: 0) {
                        case .hoveringWithData(let time, let combined, let selfPlay, let training):
                            VStack(alignment: .leading, spacing: 2) {
                                Text("t=\(Self.formatElapsedAxis(time))")
                                    .font(.caption2)
                                    .monospacedDigit()
                                Text("Combined: \(Int(combined))/hr")
                                    .font(.caption2)
                                    .monospacedDigit()
                                    .foregroundStyle(Color.green)
                                Text("Self-play: \(Int(selfPlay))/hr")
                                    .font(.caption2)
                                    .monospacedDigit()
                                    .foregroundStyle(Color.blue)
                                Text("Training:  \(Int(training))/hr")
                                    .font(.caption2)
                                    .monospacedDigit()
                                    .foregroundStyle(Color.orange)
                            }
                        case .hoveringNoData(let t):
                            VStack(alignment: .leading, spacing: 2) {
                                Text("t=\(Self.formatElapsedAxis(t))")
                                    .font(.caption2)
                                    .monospacedDigit()
                                Text("no data")
                                    .font(.caption2)
                                    .foregroundStyle(.secondary)
                            }
                        }
                    }
                    .padding(6)
                    .background(
                        RoundedRectangle(cornerRadius: 4)
                            .fill(Color(nsColor: .windowBackgroundColor).opacity(0.92))
                    )
                    .overlay(
                        RoundedRectangle(cornerRadius: 4)
                            .stroke(Color.gray.opacity(0.3), lineWidth: 0.5)
                    )
                    .padding(8)
                    .allowsHitTesting(false)
                    .opacity(hoverReadout == nil ? 0 : 1)
                }
            }
        }
        .frame(height: 320)
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
                // App memory lives in the chart grid now (App memory
                // tile), so it's been dropped from the header line
                // here to reduce duplication.
                let gpuGB = Self.bytesToGB(mem.gpuAllocatedBytes)
                let gpuMaxGB = Self.bytesToGB(mem.gpuMaxTargetBytes)
                let gpuTotalGB = Self.bytesToGB(mem.gpuTotalBytes)
                let gpuPct = mem.gpuMaxTargetBytes > 0
                ? Int((Double(mem.gpuAllocatedBytes) / Double(mem.gpuMaxTargetBytes) * 100).rounded())
                : 0
                memLine = String(
                    format: "%@  ·  GPU RAM: %.2f / %.2f GB (%d%%)  ·  Total: %.1f GB",
                    timeStr, gpuGB, gpuMaxGB, gpuPct, gpuTotalGB
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
        if next >= 0, next <= ChessNetwork.inputPlanes { selectedOverlay = next }
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
        // Always echo errors to the session log so a transient on-screen
        // error message that auto-clears in 12 seconds is still
        // recoverable from the persistent log file. (Some callsites
        // also log their own more-detailed [CHECKPOINT] line — minor
        // duplication is fine; visibility is the priority.)
        if isError {
            SessionLogger.shared.log("[CHECKPOINT-ERR] \(message)")
        }
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
        // Close the active segment at save time so the on-disk
        // session captures up-to-date cumulative wall-time totals.
        // If training is still in progress (the user saved without
        // stopping), immediately re-open a fresh segment so the
        // post-save training time continues to accumulate. Without
        // this re-open, every minute after the save would silently
        // disappear from cumulative wall-time totals — the
        // "2 hours today + 1 hour tomorrow = 3 hours" arithmetic
        // would only hold if the user never saved mid-training.
        let wasTraining = realTraining
        closeActiveTrainingSegment(reason: "save")
        if wasTraining && activeSegmentStart == nil {
            beginActiveTrainingSegment()
        }
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
                durationSec: record.durationSec,
                gamesPlayed: record.gamesPlayed,
                promotionKind: record.promotionKind?.rawValue
            )
        }
        let lr = trainer?.learningRate ?? Self.trainerLearningRateDefault
        let entropyCoeff = trainer?.entropyRegularizationCoeff ?? Self.entropyRegularizationCoeffDefault
        let drawPen = trainer?.drawPenalty ?? Float(drawPenalty)
        let bufferSnap = replayBuffer?.stateSnapshot()
        let segments: [SessionCheckpointState.TrainingSegment]? = completedTrainingSegments.isEmpty
            ? nil
            : completedTrainingSegments
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
            entropyRegularizationCoeff: entropyCoeff,
            drawPenalty: drawPen,
            promoteThreshold: Self.tournamentPromoteThreshold,
            arenaGames: Self.tournamentGames,
            selfPlayTau: TauConfigCodable(samplingScheduleBox?.selfPlay ?? buildSelfPlaySchedule()),
            arenaTau: TauConfigCodable(samplingScheduleBox?.arena ?? buildArenaSchedule()),
            selfPlayWorkerCount: selfPlayWorkerCount,
            gradClipMaxNorm: Float(gradClipMaxNorm),
            weightDecayCoeff: Float(weightDecayC),
            replayRatioTarget: replayRatioTarget,
            replayRatioAutoAdjust: replayRatioAutoAdjust,
            stepDelayMs: trainingStepDelayMs,
            lastAutoComputedDelayMs: lastAutoComputedDelayMs,
            whiteCheckmates: snap?.whiteCheckmates,
            blackCheckmates: snap?.blackCheckmates,
            stalemates: snap?.stalemates,
            fiftyMoveDraws: snap?.fiftyMoveDraws,
            threefoldRepetitionDraws: snap?.threefoldRepetitionDraws,
            insufficientMaterialDraws: snap?.insufficientMaterialDraws,
            totalGameWallMs: snap?.totalGameWallMs,
            buildNumber: BuildInfo.buildNumber,
            buildGitHash: BuildInfo.gitHash,
            buildGitBranch: BuildInfo.gitBranch,
            buildDate: BuildInfo.buildDate,
            buildTimestamp: BuildInfo.buildTimestamp,
            buildGitDirty: BuildInfo.gitDirty,
            hasReplayBuffer: bufferSnap != nil,
            replayBufferStoredCount: bufferSnap?.storedCount,
            replayBufferCapacity: bufferSnap?.capacity,
            replayBufferTotalPositionsAdded: bufferSnap?.totalPositionsAdded,
            championID: championID,
            trainerID: trainerID,
            arenaHistory: history
        ).withTrainingSegments(segments)
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
                    try await champion.exportWeights()
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
                    let url = try await CheckpointManager.saveModel(
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
        // jumping to detached work. Capture the replay buffer handle
        // here too so the detached write path can serialize it
        // alongside the two network files — `ReplayBuffer` is
        // `@unchecked Sendable` and serializes access via its own
        // lock, so the buffer can be written from a background task
        // while self-play workers (which only append) are paused.
        let sessionState = buildCurrentSessionState(
            championID: championID,
            trainerID: trainerID
        )
        let trainingStep = trainingStats?.steps ?? 0
        let bufferForSave = replayBuffer

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
                    try await champion.exportWeights()
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
                    try await trainer.network.exportWeights()
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
                [bufferForSave] in
                do {
                    let url = try await CheckpointManager.saveSession(
                        championWeights: championWeights,
                        championID: championID,
                        championMetadata: championMetadata,
                        championCreatedAtUnix: now,
                        trainerWeights: trainerWeights,
                        trainerID: trainerID,
                        trainerMetadata: trainerMetadata,
                        trainerCreatedAtUnix: now,
                        state: sessionState,
                        replayBuffer: bufferForSave,
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
                let bufStr: String
                if let snap = bufferForSave?.stateSnapshot() {
                    bufStr = " replay=\(snap.storedCount)/\(snap.capacity)"
                } else {
                    bufStr = ""
                }
                SessionLogger.shared.log("[CHECKPOINT] Saved session: \(url.lastPathComponent) build=\(BuildInfo.buildNumber) git=\(BuildInfo.gitHash)\(bufStr)")
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
                    try await champion.loadWeights(file.weights)
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
                    try await champion.loadWeights(loaded.championFile.weights)
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
                let savedBuild = loaded.state.buildNumber.map(String.init) ?? "?"
                let savedGit = loaded.state.buildGitHash ?? "?"
                let bufStr: String
                if let stored = loaded.state.replayBufferStoredCount,
                   let cap = loaded.state.replayBufferCapacity {
                    bufStr = " replay=\(stored)/\(cap)"
                } else {
                    bufStr = " replay=none"
                }
                SessionLogger.shared.log("[CHECKPOINT] Loaded session: \(url.lastPathComponent) savedBuild=\(savedBuild) savedGit=\(savedGit)\(bufStr)")
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

    // MARK: - Menu command hub wiring

    /// Assign each menu-bar command to its corresponding action
    /// function. Called once from `.onAppear` so the closures stick
    /// for the lifetime of the view and point at the live view's
    /// `@State`-backed functions (capturing `self` here is safe
    /// because the `@State` storage is keyed by view identity, not
    /// by the struct value).
    private func wireMenuCommandHub() {
        commandHub.buildNetwork = { buildNetwork() }
        commandHub.runForwardPass = { runForwardPass() }
        commandHub.playSingleGame = { playSingleGame() }
        commandHub.startContinuousPlay = { startContinuousPlay() }
        commandHub.trainOnce = { trainOnce() }
        commandHub.startContinuousTraining = { startContinuousTraining() }
        commandHub.startRealTraining = { startRealTraining() }
        commandHub.startSweep = { startSweep() }
        commandHub.stopAnyContinuous = { stopAnyContinuous() }
        commandHub.runArena = {
            SessionLogger.shared.log("[BUTTON] Run Arena")
            guard !isArenaRunning else { return }
            arenaTriggerBox?.trigger()
        }
        commandHub.runEngineDiagnostics = { runEngineDiagnostics() }
        commandHub.abortArena = {
            SessionLogger.shared.log("[BUTTON] Abort Arena")
            arenaOverrideBox?.abort()
        }
        commandHub.promoteCandidate = {
            SessionLogger.shared.log("[BUTTON] Promote")
            arenaOverrideBox?.promote()
        }
        commandHub.saveSession = {
            SessionLogger.shared.log("[BUTTON] Save Session")
            handleSaveSessionManual()
        }
        commandHub.saveChampion = {
            SessionLogger.shared.log("[BUTTON] Save Champion")
            handleSaveChampionAsModel()
        }
        commandHub.loadSession = {
            SessionLogger.shared.log("[BUTTON] Load Session")
            showingLoadSessionImporter = true
        }
        commandHub.loadModel = {
            SessionLogger.shared.log("[BUTTON] Load Model")
            showingLoadModelImporter = true
        }
        commandHub.revealSaves = { handleRevealSaves() }
    }

    /// Push the subset of view state that governs menu enable/disable
    /// into the hub. Called from `.onAppear` and on every relevant
    /// state change so the menu items reflect live conditions
    /// (Build Network greys out after the first build, Save Session
    /// enables once Play-and-Train starts, etc.).
    private func syncMenuCommandHubState() {
        commandHub.networkReady = networkReady
        commandHub.isBusy = isBusy
        commandHub.isBuilding = isBuilding
        commandHub.gameIsPlaying = gameSnapshot.isPlaying
        commandHub.continuousPlay = continuousPlay
        commandHub.continuousTraining = continuousTraining
        commandHub.sweepRunning = sweepRunning
        commandHub.realTraining = realTraining
        commandHub.isArenaRunning = isArenaRunning
        commandHub.checkpointSaveInFlight = checkpointSaveInFlight
        commandHub.pendingLoadedSessionExists = pendingLoadedSession != nil
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
                    Parameters: ~2,400,000 (~2.4M)
                    Architecture: 20x8x8 -> stem(128)
                      -> 8 res+SE blocks -> policy(4864) + value(1)
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
            let championRunner = runner,
            arenaActiveFlag?.isActive != true
        else { return }
        let now = Date()
        let dirty = candidateProbeDirty
        let intervalElapsed = now.timeIntervalSince(lastCandidateProbeTime)
        >= Self.candidateProbeIntervalSec
        guard dirty || intervalElapsed else { return }

        let state = editableState
        let target = probeNetworkTarget
        let result: EvaluationResult
        do {
            switch target {
            case .candidate:
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
                    let weights = try await trainer.network.exportWeights()
                    try await candidateInference.loadWeights(weights)
                    return await Self.performInference(with: candidateRunner, state: state)
                }.value
                // Probe is a transient read-only snapshot, not a checkpoint —
                // candidateInference inherits the trainer's current ID rather
                // than minting a fresh one. (Arena snapshots, by contrast,
                // do mint — see runArenaParallel.)
                candidateInference.identifier = trainer.identifier
            case .champion:
                // Probe the champion directly — no sync. The champion is
                // frozen between promotions, so reading from it through
                // its own runner is the same path Run Forward Pass uses
                // and is safe to call concurrently with self-play workers
                // (they all read through a batcher; direct runner reads
                // just add another fair-share consumer). Skipped while an
                // arena is running because the promotion step briefly
                // writes into the champion under a self-play pause.
                result = await Task.detached(priority: .userInitiated) {
                    await Self.performInference(with: championRunner, state: state)
                }.value
            }
        } catch {
            // Leave probe state unchanged so the previous result stays
            // on screen; the next gap-point call will retry. The error
            // lands in trainingBox via the driver loop's existing
            // plumbing if something structural broke.
            return
        }
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
        // Snapshot losses/entropy at arena start so the log shows
        // the trainer's state entering the arena — especially
        // useful for diagnosing whether divergence was already
        // underway before the arena ran.
        if let snap = trainingBox?.snapshot() {
            let pStr = snap.rollingPolicyLoss.map { String(format: "%+.4f", $0) } ?? "--"
            let vStr = snap.rollingValueLoss.map { String(format: "%+.4f", $0) } ?? "--"
            let eStr = snap.rollingPolicyEntropy.map { String(format: "%.4f", $0) } ?? "--"
            let gStr = snap.rollingGradGlobalNorm.map { String(format: "%.3f", $0) } ?? "--"
            let vmStr = snap.rollingValueMean.map { String(format: "%+.4f", $0) } ?? "--"
            let vaStr = snap.rollingValueAbsMean.map { String(format: "%.4f", $0) } ?? "--"
            let bufCount = replayBuffer?.count ?? 0
            let bufCap = replayBuffer?.capacity ?? Self.replayBufferCapacity
            SessionLogger.shared.log(
                "[STATS] arena-start  steps=\(steps) buffer=\(bufCount)/\(bufCap) pLoss=\(pStr) vLoss=\(vStr) pEnt=\(eStr) gNorm=\(gStr) vMean=\(vmStr) vAbs=\(vaStr) trainer=\(trainerIDStart) champion=\(championIDStart)"
            )
        }

        // Mark arena active and seed live progress. Arena-active
        // suppresses the candidate test probe for the duration so
        // probe and arena don't race on the candidate inference
        // network. isArenaRunning is @State mirror the UI reads to
        // disable the Run Arena button and adjust the busy label.
        arenaFlag.set()
        isArenaRunning = true
        // Record the arena's start elapsed-second position so the
        // chart grid's arena activity tile can render a live band
        // as the arena progresses, rather than only showing the
        // arena post-hoc when the completed ArenaChartEvent lands.
        // Uses `parallelStats.sessionStart` (fresh at Play-and-Train
        // start) so the arena's x-position lands on the same axis
        // the training + progress-rate charts render against. Using
        // the back-dated `currentSessionStart` would park the arena
        // band ~hours off the chart on resumed sessions.
        if let sessionStart = parallelStats?.sessionStart ?? currentSessionStart {
            activeArenaStartElapsed = max(0, Date().timeIntervalSince(sessionStart))
        }

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
                let weights = try await trainer.network.exportWeights()
                try await candidateInference.loadWeights(weights)
                return weights
            }.value
        } catch {
            trainingBox?.recordError("Arena candidate sync failed: \(error.localizedDescription)")
            trainingGate.resume()
            cleanupArenaState(arenaFlag: arenaFlag, tBox: tBox)
            return
        }
        trainingGate.resume()

        // Arena candidate inherits the trainer's current generation
        // ID. If it gets promoted, the promoted champion should keep
        // that exact identifier and the live trainer will roll forward
        // to the next generation after being rewound to the promoted
        // weights.
        candidateInference.identifier = trainer.identifier

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
                let weights = try await champion.exportWeights()
                try await arenaChampion.loadWeights(weights)
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
        let arenaDiversity = GameDiversityTracker(windowSize: totalGames)
        // Snapshot the current arena schedule once at tournament start
        // so every game in this arena uses the same tau settings, even
        // if the user edits the fields mid-tournament.
        let arenaScheduleSnapshot = samplingScheduleBox?.arena ?? buildArenaSchedule()
        let stats = await withTaskCancellationHandler {
            await Task.detached(priority: .userInitiated) {
                [arenaChampion, candidateInference, tBox, cancelBox, overrideBox, arenaDiversity, arenaScheduleSnapshot] in
                let driver = TournamentDriver()
                driver.delegate = nil
                return await driver.run(
                    playerA: {
                        MPSChessPlayer(
                            name: "Candidate",
                            source: DirectMoveEvaluationSource(network: candidateInference),
                            schedule: arenaScheduleSnapshot
                        )
                    },
                    playerB: {
                        MPSChessPlayer(
                            name: "Champion",
                            source: DirectMoveEvaluationSource(network: arenaChampion),
                            schedule: arenaScheduleSnapshot
                        )
                    },
                    games: totalGames,
                    diversityTracker: arenaDiversity,
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
        // `promotionKind` tracks WHY the promotion is happening, so
        // the history / logs can distinguish a user-forced promotion
        // (Promote button) from a score-threshold one. Only read if
        // `promoted` ends up true.
        let promotionKind: PromotionKind?
        switch overrideDecision {
        case .abort:
            shouldPromote = false
            promotionKind = nil
        case .promote:
            shouldPromote = true
            promotionKind = .manual
        case .none:
            shouldPromote = playedGames >= totalGames && score >= Self.tournamentPromoteThreshold
            promotionKind = shouldPromote ? .automatic : nil
        }
        // Holds the new champion weights if promotion succeeds,
        // so we can hand them to a detached autosave task at the
        // end without needing to re-read them from the live
        // network (which would race against self-play again).
        var promotedChampionWeights: [[Float]] = []
        if shouldPromote {
            // Pause both self-play and training, then copy the
            // promoted candidate into both the champion and the live
            // trainer. This keeps self-play and SGD aligned on the
            // exact promoted weights rather than letting training
            // continue from a later, unvalidated post-arena state.
            await selfPlayGate.pauseAndWait()
            await trainingGate.pauseAndWait()
            if !Task.isCancelled {
                do {
                    promotedChampionWeights = try await Task.detached(priority: .userInitiated) {
                        [candidateInference, champion, trainer] in
                        let weights = try await candidateInference.exportWeights()
                        try await champion.loadWeights(weights)
                        try await trainer.network.loadWeights(weights)
                        return weights
                    }.value
                    // Promoted: champion now holds the arena candidate's
                    // exact weights, so it inherits that snapshot ID,
                    // while the rewound live trainer rolls forward to
                    // the next mutable generation in the same lineage.
                    champion.identifier = candidateInference.identifier
                    trainer.identifier = ModelIDMinter.mintTrainerGeneration(
                        from: champion.identifier ?? candidateInference.identifier ?? ModelIDMinter.mint()
                    )
                    promoted = true
                    promotedID = candidateInference.identifier
                    trainingBox?.resetRollingWindows()
                    divergenceWarningStreak = 0
                    divergenceCriticalStreak = 0
                    divergenceRecoveryStreak = 0
                    clearTrainingAlarm()
                } catch {
                    trainingBox?.recordError("Promotion copy failed: \(error.localizedDescription)")
                }
            }
            trainingGate.resume()
            selfPlayGate.resume()
        }

        // Append to history and clear arena state.
        let durationSec = Date().timeIntervalSince(startTime)
        let record = TournamentRecord(
            finishedAtStep: steps,
            gamesPlayed: playedGames,
            candidateWins: stats.playerAWins,
            championWins: stats.playerBWins,
            draws: stats.draws,
            score: score,
            promoted: promoted,
            promotionKind: promoted ? promotionKind : nil,
            promotedID: promotedID,
            durationSec: durationSec
        )
        tournamentHistory.append(record)
        // Mirror into the chart-tile event stream. Compute the
        // elapsed-second start/end against the session-start anchor
        // so the band lands on the same X axis as the time-series
        // charts. Uses `parallelStats.sessionStart` (fresh at
        // Play-and-Train start) to match the training/progress-rate
        // chart axes — the back-dated `currentSessionStart` would
        // push the completed arena band ~hours off the chart on
        // resumed sessions. Guarded by sessionStart existing —
        // a stale arena tick with no session shouldn't happen
        // (arenas only run during Play-and-Train) but we'd rather
        // silently skip than dereference a nil anchor.
        if let sessionStart = parallelStats?.sessionStart ?? currentSessionStart {
            let endElapsed = max(0, Date().timeIntervalSince(sessionStart))
            // Prefer the live start mark captured at arena begin —
            // it avoids a ~5-second drift from backward-inferring
            // startElapsed out of (end - durationSec) after the
            // promotion work ran. Fall back to the durationSec math
            // only if the live mark is somehow nil.
            let startElapsed = activeArenaStartElapsed
            ?? max(0, endElapsed - durationSec)
            arenaChartEvents.append(ArenaChartEvent(
                id: arenaChartEvents.count,
                startElapsedSec: startElapsed,
                endElapsedSec: endElapsed,
                score: score,
                promoted: promoted
            ))
        }
        // Arena no longer active — clear the live band trigger so
        // the chart drops back to just the completed events.
        activeArenaStartElapsed = nil
        logArenaResult(
            record: record,
            index: tournamentHistory.count,
            trainer: trainer,
            candidate: candidateInference,
            championSide: arenaChampion,
            diversity: arenaDiversity.snapshot()
        )
        cleanupArenaState(arenaFlag: arenaFlag, tBox: tBox)

        // On promotion: reset game-play stats so the display
        // reflects only the new champion's self-play performance,
        // and emit a STATS log line so the post-promotion state
        // is visible in the session log (the fixed STATS ticker
        // may not fire for up to an hour at this point in the
        // schedule).
        if promoted {
            parallelWorkerStatsBox?.resetGameStats()
            let trainerIDStr = trainer.identifier?.description ?? "?"
            let championIDStr = champion.identifier?.description ?? "?"
            SessionLogger.shared.log(
                "[STATS] post-promote  steps=\(trainingStats?.steps ?? 0) champion=\(championIDStr) trainer=\(trainerIDStr)"
            )
        }

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
            let bufferForAutosave = replayBuffer

            // Fire-and-forget detached task. The closure captures
            // only Sendable value types (weight arrays, metadata
            // structs, the session state snapshot, and a few
            // strings) — explicitly NOT `self` — so we can safely
            // run past a session end without touching any View
            // @State. Outcome is audited through SessionLogger,
            // which is serial-queue-guarded and thread-safe. The
            // user sees success via "Reveal Saves" finding the
            // timestamped folder; failures show up in the
            // session log.
            Task.detached(priority: .utility) {
                [bufferForAutosave] in
                do {
                    let url = try await CheckpointManager.saveSession(
                        championWeights: championWeightsSnapshot,
                        championID: championID,
                        championMetadata: championMetadata,
                        championCreatedAtUnix: createdAtUnix,
                        trainerWeights: trainerWeightsSnapshot,
                        trainerID: trainerID,
                        trainerMetadata: trainerMetadata,
                        trainerCreatedAtUnix: createdAtUnix,
                        state: sessionState,
                        replayBuffer: bufferForAutosave,
                        trigger: "promote"
                    )
                    let bufStr: String
                    if let snap = bufferForAutosave?.stateSnapshot() {
                        bufStr = " replay=\(snap.storedCount)/\(snap.capacity)"
                    } else {
                        bufStr = ""
                    }
                    SessionLogger.shared.log("[CHECKPOINT] Autosaved session: \(url.lastPathComponent) build=\(BuildInfo.buildNumber)\(bufStr)")
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
        championSide: ChessMPSNetwork,
        diversity: GameDiversityTracker.Snapshot
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
        let kindSuffix: String
        switch record.promotionKind {
        case .automatic:
            kindSuffix = " (auto)"
        case .manual:
            kindSuffix = " (manual)"
        case .none:
            kindSuffix = ""
        }
        if record.promoted, let pid = record.promotedID {
            statusStr = "PROMOTED\(kindSuffix)=\(pid.description)"
        } else if record.promoted {
            statusStr = "PROMOTED\(kindSuffix)"
        } else {
            statusStr = "kept"
        }
        let sp = samplingScheduleBox?.selfPlay ?? buildSelfPlaySchedule()
        let ar = samplingScheduleBox?.arena ?? buildArenaSchedule()
        let lrStr = String(format: "%.1e", trainer.learningRate)
        let scoreStr = String(format: "%.3f", record.score)
        let threshStr = String(format: "%.2f", Self.tournamentPromoteThreshold)
        let spTauStr = String(
            format: "%.2f/%.2f/%.3f",
            Double(sp.startTau),
            Double(sp.floorTau),
            Double(sp.decayPerPly)
        )
        let arTauStr = String(
            format: "%.2f/%.2f/%.3f",
            Double(ar.startTau),
            Double(ar.floorTau),
            Double(ar.decayPerPly)
        )
        let candidateIDStr = candidate.identifier?.description ?? "?"
        let championIDStr = championSide.identifier?.description ?? "?"
        let trainerIDStr = trainer.identifier?.description ?? "?"

        let divStr = String(format: "unique=%d/%d(%.0f%%) avgDiverge=%.1f",
                            diversity.uniqueGames, diversity.gamesInWindow,
                            diversity.uniquePercent, diversity.avgDivergencePly)

        let wld = "\(record.candidateWins)-\(record.championWins)-\(record.draws)"
        let gamesStr = "\(record.gamesPlayed)/\(Self.tournamentGames)"
        let header = "[ARENA] #\(index) @ step \(record.finishedAtStep)  games=\(gamesStr)  W/L/D=\(wld)  score=\(scoreStr)  \(statusStr)  dur=\(durationStr)"
        let params = "        batch=\(Self.trainingBatchSize) lr=\(lrStr) promote>=\(threshStr) games=\(Self.tournamentGames) sp.tau=\(spTauStr) ar.tau=\(arTauStr) workers=\(selfPlayWorkerCount) build=\(BuildInfo.buildNumber)"
        let ids = "        candidate=\(candidateIDStr)  champion=\(championIDStr)  trainer=\(trainerIDStr)"
        let div = "        diversity: \(divStr)"

        print(header)
        print(params)
        print(ids)
        print(div)

        // Mirror the same four lines into the session log so the
        // on-disk file carries the full arena history even when the
        // Xcode console isn't being captured.
        SessionLogger.shared.log(header)
        SessionLogger.shared.log(params)
        SessionLogger.shared.log(ids)
        SessionLogger.shared.log(div)
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
        // Early-exit cleanup — make sure the live arena band on
        // the chart grid isn't left in an "arena active" state
        // after cancellation / error paths that skipped the
        // normal append-then-clear sequence. A no-op on the happy
        // path (the append site already cleared this).
        activeArenaStartElapsed = nil
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
                await Self.performInference(with: runner, state: state)
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
            let source = DirectMoveEvaluationSource(network: network)
            let white = MPSChessPlayer(name: "White", source: source)
            let black = MPSChessPlayer(name: "Black", source: source)
            do {
                _ = try await machine.beginNewGame(white: white, black: black)
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
                let source = DirectMoveEvaluationSource(network: network)
                let white = MPSChessPlayer(name: "White", source: source)
                let black = MPSChessPlayer(name: "Black", source: source)
                do {
                    _ = try await machine.beginNewGame(white: white, black: black)
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
        if let trainer {
            trainer.learningRate = Float(trainerLearningRate)
            trainer.entropyRegularizationCoeff = Float(entropyRegularizationCoeff)
            trainer.drawPenalty = Float(drawPenalty)
            trainer.weightDecayC = Float(weightDecayC)
            trainer.gradClipMaxNorm = Float(gradClipMaxNorm)
            trainer.policyScaleK = Float(policyScaleK)
            return trainer
        }
        do {
            let t = try ChessTrainer(
                learningRate: Float(trainerLearningRate),
                entropyRegularizationCoeff: Float(entropyRegularizationCoeff),
                drawPenalty: Float(drawPenalty),
                weightDecayC: Float(weightDecayC),
                gradClipMaxNorm: Float(gradClipMaxNorm),
                policyScaleK: Float(policyScaleK)
            )
            trainer = t
            return t
        } catch {
            trainingError = "Trainer init failed: \(error.localizedDescription)"
            return nil
        }
    }

    /// Build a `SamplingSchedule` for self-play from the live
    /// `@AppStorage` tau values. Dirichlet noise matches the default
    /// `.selfPlay` preset (AlphaZero noise) — not exposed in the UI,
    /// only the temperature schedule is editable.
    private func buildSelfPlaySchedule() -> SamplingSchedule {
        SamplingSchedule(
            startTau: Float(max(0.01, spStartTau)),
            decayPerPly: Float(max(0.0, spDecayPerPly)),
            floorTau: Float(max(0.01, spFloorTau)),
            dirichletNoise: SamplingSchedule.selfPlay.dirichletNoise
        )
    }

    /// Build a `SamplingSchedule` for arena play from the live
    /// `@AppStorage` tau values. Arena never applies Dirichlet noise
    /// (pure strength measurement).
    private func buildArenaSchedule() -> SamplingSchedule {
        SamplingSchedule(
            startTau: Float(max(0.01, arStartTau)),
            decayPerPly: Float(max(0.0, arDecayPerPly)),
            floorTau: Float(max(0.01, arFloorTau))
        )
    }

    /// Commit edits to the self-play tau fields. Parses each of the
    /// three `@State` strings, clamps to reasonable ranges, writes back
    /// to `@AppStorage`, and pushes the new schedule into the live
    /// `samplingScheduleBox` so the next game played on each self-play
    /// slot picks it up. Invalid entries revert to the persisted value.
    private func applySpTauEdit() {
        var changed = false
        if let v = Double(spStartTauEditText), v > 0, v.isFinite, v <= 10 {
            if abs(v - spStartTau) > Double.ulpOfOne {
                SessionLogger.shared.log(String(format: "[PARAM] sp.startTau: %.3f -> %.3f", spStartTau, v))
                spStartTau = v
                changed = true
            }
        }
        if let v = Double(spFloorTauEditText), v > 0, v.isFinite, v <= 10 {
            if abs(v - spFloorTau) > Double.ulpOfOne {
                SessionLogger.shared.log(String(format: "[PARAM] sp.floorTau: %.3f -> %.3f", spFloorTau, v))
                spFloorTau = v
                changed = true
            }
        }
        if let v = Double(spDecayPerPlyEditText), v >= 0, v.isFinite, v <= 1 {
            if abs(v - spDecayPerPly) > Double.ulpOfOne {
                SessionLogger.shared.log(String(format: "[PARAM] sp.decayPerPly: %.4f -> %.4f", spDecayPerPly, v))
                spDecayPerPly = v
                changed = true
            }
        }
        spStartTauEditText = String(format: "%.2f", spStartTau)
        spFloorTauEditText = String(format: "%.2f", spFloorTau)
        spDecayPerPlyEditText = String(format: "%.3f", spDecayPerPly)
        if changed {
            samplingScheduleBox?.setSelfPlay(buildSelfPlaySchedule())
        }
    }

    /// Commit edits to the arena tau fields. Same contract as
    /// `applySpTauEdit` but writes into the arena slot of the box;
    /// in-flight arenas keep the snapshot they were created with and
    /// only the next arena picks up the new schedule.
    private func applyArTauEdit() {
        var changed = false
        if let v = Double(arStartTauEditText), v > 0, v.isFinite, v <= 10 {
            if abs(v - arStartTau) > Double.ulpOfOne {
                SessionLogger.shared.log(String(format: "[PARAM] ar.startTau: %.3f -> %.3f", arStartTau, v))
                arStartTau = v
                changed = true
            }
        }
        if let v = Double(arFloorTauEditText), v > 0, v.isFinite, v <= 10 {
            if abs(v - arFloorTau) > Double.ulpOfOne {
                SessionLogger.shared.log(String(format: "[PARAM] ar.floorTau: %.3f -> %.3f", arFloorTau, v))
                arFloorTau = v
                changed = true
            }
        }
        if let v = Double(arDecayPerPlyEditText), v >= 0, v.isFinite, v <= 1 {
            if abs(v - arDecayPerPly) > Double.ulpOfOne {
                SessionLogger.shared.log(String(format: "[PARAM] ar.decayPerPly: %.4f -> %.4f", arDecayPerPly, v))
                arDecayPerPly = v
                changed = true
            }
        }
        arStartTauEditText = String(format: "%.2f", arStartTau)
        arFloorTauEditText = String(format: "%.2f", arFloorTau)
        arDecayPerPlyEditText = String(format: "%.3f", arDecayPerPly)
        if changed {
            samplingScheduleBox?.setArena(buildArenaSchedule())
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
            let result = await Self.runOneTrainStep(trainer: trainer)
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
                let result = await Self.runOneTrainStep(trainer: trainer)
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

    nonisolated private static func runOneTrainStep(trainer: ChessTrainer) async -> Result<TrainStepTiming, Error> {
        do {
            return .success(try await trainer.trainStep(batchSize: trainingBatchSize))
        } catch {
            return .failure(error)
        }
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
        // Begin a new training segment for cumulative wall-time
        // tracking. Closed via `closeActiveTrainingSegment` on Stop or
        // at save time. Don't try to open one if the previous Stop
        // didn't actually close (defensive — `closeActiveTrainingSegment`
        // is idempotent on nil but we want the log line to be clean).
        if activeSegmentStart != nil {
            closeActiveTrainingSegment(reason: "restart-without-stop")
        }
        beginActiveTrainingSegment()
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
        clearTrainingAlarm()
        divergenceWarningStreak = 0
        divergenceCriticalStreak = 0
        divergenceRecoveryStreak = 0

        let buffer = ReplayBuffer(capacity: Self.replayBufferCapacity)
        replayBuffer = buffer
        let box = TrainingLiveStatsBox(rollingWindow: Self.rollingLossWindow)
        if let rs = pendingLoadedSession?.state {
            var seeded = TrainingRunStats()
            seeded.steps = rs.trainingSteps
            box.seed(seeded)
        }
        trainingBox = box
        realRollingPolicyLoss = nil
        realRollingValueLoss = nil
        // Seed counters from the loaded session if resuming, or
        // start fresh. This covers the stats box (game/move/result
        // counters), training step count, tournament history, and
        // worker count.
        let resumeState = pendingLoadedSession?.state
        if let rs = resumeState {
            trainer.learningRate = rs.learningRate
            trainerLearningRate = Double(rs.learningRate)
            if let entropyCoeff = rs.entropyRegularizationCoeff {
                trainer.entropyRegularizationCoeff = entropyCoeff
                entropyRegularizationCoeff = Double(entropyCoeff)
            } else {
                trainer.entropyRegularizationCoeff = Float(entropyRegularizationCoeff)
            }
            if let dp = rs.drawPenalty {
                trainer.drawPenalty = dp
                drawPenalty = Double(dp)
            } else {
                trainer.drawPenalty = Float(drawPenalty)
            }
        } else {
            trainer.learningRate = Float(trainerLearningRate)
            trainer.entropyRegularizationCoeff = Float(entropyRegularizationCoeff)
            trainer.drawPenalty = Float(drawPenalty)
        }
        var initialTrainingStats = TrainingRunStats()
        if let rs = resumeState {
            initialTrainingStats.steps = rs.trainingSteps
        }
        trainingStats = initialTrainingStats
        playAndTrainBoardMode = .gameRun
        probeNetworkTarget = .candidate
        candidateProbeDirty = false
        lastCandidateProbeTime = .distantPast
        candidateProbeCount = 0
        learningRateEditText = String(format: "%.1e", trainer.learningRate)
        entropyRegularizationEditText = String(format: "%.2e", trainer.entropyRegularizationCoeff)
        drawPenaltyEditText = String(format: "%.3f", Double(trainer.drawPenalty))
        weightDecayEditText = String(format: "%.2e", trainer.weightDecayC)
        gradClipMaxNormEditText = String(format: "%.2f", trainer.gradClipMaxNorm)
        policyScaleKEditText = String(format: "%.2f", trainer.policyScaleK)
        spStartTauEditText = String(format: "%.2f", spStartTau)
        spFloorTauEditText = String(format: "%.2f", spFloorTau)
        spDecayPerPlyEditText = String(format: "%.3f", spDecayPerPly)
        arStartTauEditText = String(format: "%.2f", arStartTau)
        arFloorTauEditText = String(format: "%.2f", arFloorTau)
        arDecayPerPlyEditText = String(format: "%.3f", arDecayPerPly)
        if let rs = resumeState {
            // Hydrate prior training segments so cumulative wall-time
            // and run-count metrics carry across save/load. Missing
            // (older session files) → empty array, which means this
            // becomes the first segment in the session's history.
            completedTrainingSegments = rs.trainingSegments ?? []
            tournamentHistory = rs.arenaHistory.map { entry in
                // Legacy session files don't store `gamesPlayed` —
                // reconstruct it from the W/L/D totals (same identity
                // the driver uses when building the live record).
                // Legacy files also don't store `promotionKind` — the
                // manual Promote button didn't exist then, so treat
                // any recorded promotion as automatic.
                let gp = entry.gamesPlayed
                ?? (entry.candidateWins + entry.championWins + entry.draws)
                let kind: PromotionKind?
                if entry.promoted {
                    if let raw = entry.promotionKind,
                       let parsed = PromotionKind(rawValue: raw) {
                        kind = parsed
                    } else {
                        kind = .automatic
                    }
                } else {
                    kind = nil
                }
                return TournamentRecord(
                    finishedAtStep: entry.finishedAtStep,
                    gamesPlayed: gp,
                    candidateWins: entry.candidateWins,
                    championWins: entry.championWins,
                    draws: entry.draws,
                    score: entry.score,
                    promoted: entry.promoted,
                    promotionKind: kind,
                    promotedID: entry.promotedID.map { ModelID(value: $0) },
                    durationSec: entry.durationSec
                )
            }
        } else {
            tournamentHistory = []
        }
        tournamentProgress = nil
        let tBox = TournamentLiveBox()
        tournamentBox = tBox
        let pStatsBox: ParallelWorkerStatsBox
        if let rs = resumeState {
            pStatsBox = ParallelWorkerStatsBox(
                sessionStart: Date(),
                totalGames: rs.selfPlayGames,
                totalMoves: rs.selfPlayMoves,
                totalGameWallMs: rs.totalGameWallMs ?? 0,
                whiteCheckmates: rs.whiteCheckmates ?? 0,
                blackCheckmates: rs.blackCheckmates ?? 0,
                stalemates: rs.stalemates ?? 0,
                fiftyMoveDraws: rs.fiftyMoveDraws ?? 0,
                threefoldRepetitionDraws: rs.threefoldRepetitionDraws ?? 0,
                insufficientMaterialDraws: rs.insufficientMaterialDraws ?? 0,
                trainingSteps: rs.trainingSteps
            )
            if let workerCount = resumeState?.selfPlayWorkerCount {
                selfPlayWorkerCount = max(1, min(Self.absoluteMaxSelfPlayWorkers, workerCount))
            }
            if let delay = rs.stepDelayMs {
                trainingStepDelayMs = delay
            }
            if let autoDelay = rs.lastAutoComputedDelayMs {
                lastAutoComputedDelayMs = autoDelay
            }
        } else {
            pStatsBox = ParallelWorkerStatsBox(sessionStart: Date())
        }
        parallelWorkerStatsBox = pStatsBox
        parallelStats = pStatsBox.snapshot()
        let spDiversityTracker = GameDiversityTracker(windowSize: 200)
        selfPlayDiversityTracker = spDiversityTracker
        currentDiversityHistogramBars = []
        arenaChartEvents = []
        // Reset progress-rate sampler state so the new session's
        // chart starts fresh at t=0. Leaving old samples in place
        // would show up as a visible "step" from the previous
        // session's trailing values to the new session's zero
        // reading.
        progressRateSamples = []
        trainingChartSamples = []
        trainingChartNextId = 0
        prevChartTotalGpuMs = 0
        progressRateLastFetch = .distantPast
        progressRateNextId = 0
        progressRateScrollX = 0
        progressRateFollowLatest = true
        // Single self-play gate. All self-play workers now share one
        // `BatchedMoveEvaluationSource` on the champion network, driven
        // by `BatchedSelfPlayDriver`, so there is exactly one consumer
        // for the arena coordinator to pause. The previous per-secondary
        // gate array is gone along with the secondary networks.
        let selfPlayGate = WorkerPauseGate()
        // Shared current-N holder. Workers poll this to decide
        // whether to play another game or sit in their idle wait.
        // The Stepper writes through it (and to `@State
        // selfPlayWorkerCount` simultaneously). Exposed via @State
        // so the UI can disable the buttons when the box is gone
        // (between sessions).
        let countBox = WorkerCountBox(initial: initialWorkerCount)
        workerCountBox = countBox
        // Live schedule box, seeded from the current @AppStorage tau
        // values. Edits in the tau text fields push new schedules into
        // this box; the self-play driver's slots read at the top of
        // each new game, so the next game played uses the edited
        // values.
        let spSchedule = buildSelfPlaySchedule()
        let arSchedule = buildArenaSchedule()
        let scheduleBox = SamplingScheduleBox(
            selfPlay: spSchedule,
            arena: arSchedule
        )
        samplingScheduleBox = scheduleBox
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
            initialDelayMs: replayRatioAutoAdjust
            ? lastAutoComputedDelayMs
            : trainingStepDelayMs,
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
            trainerLearningRate = Double(resumed.state.learningRate)
            if let entropyCoeff = resumed.state.entropyRegularizationCoeff {
                entropyRegularizationCoeff = Double(entropyCoeff)
            }
        } else {
            currentSessionID = ModelIDMinter.mint().value
            currentSessionStart = Date()
        }

        realTrainingTask = Task(priority: .userInitiated) {
            [trainer, network, buffer, box, tBox, pStatsBox, spDiversityTracker,
             selfPlayGate, trainingGate, arenaFlag, triggerBox, overrideBox, countBox,
             gameWatcher, ratioController] in

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

            // Reset the trainer's graph AND initialize its weights.
            // Two paths:
            //
            // (1) Normal start: fork the trainer off the champion so
            //     arena-at-step-0 is a fair tie by construction and
            //     the trainer's starting point is a true fork of the
            //     champion.
            //
            // (2) Resume from a loaded `.dcmsession`: load the trainer
            //     from the session's `trainer.dcmmodel` payload so its
            //     mid-training divergence from the champion is
            //     preserved. Without this branch, resume would throw
            //     away the trainer's in-flight SGD progress every
            //     time. The champion has already been loaded from
            //     disk into `network` at file-load time, so the
            //     batcher (which wraps `network`) picks up the
            //     restored weights automatically.
            let resumedTrainerWeights: [[Float]]? = await MainActor.run {
                pendingLoadedSession?.trainerFile.weights
            }
            let resumedBufferURL: URL? = await MainActor.run {
                pendingLoadedSession?.replayBufferURL
            }
            do {
                try await trainer.resetNetwork()
                try await Task.detached(priority: .userInitiated) {
                    [resumedTrainerWeights] in
                    if let trainerWeights = resumedTrainerWeights {
                        try await trainer.network.loadWeights(trainerWeights)
                    } else {
                        let championWeights = try await network.exportWeights()
                        try await trainer.network.loadWeights(championWeights)
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

            // Restore replay buffer before any self-play worker or
            // the training worker starts — training samples from the
            // buffer, and any worker appends after restore resets
            // would either be clobbered or (worse) race with the
            // restore's counter reset. The restore runs on a
            // detached I/O task so ~GB-scale reads don't block the
            // cooperative hop cadence.
            if let bufferURL = resumedBufferURL {
                do {
                    try await Task.detached(priority: .userInitiated) {
                        [buffer, bufferURL] in
                        try buffer.restore(from: bufferURL)
                    }.value
                    let snap = buffer.stateSnapshot()
                    SessionLogger.shared.log(
                        "[CHECKPOINT] Restored replay buffer: stored=\(snap.storedCount)/\(snap.capacity) totalAdded=\(snap.totalPositionsAdded) writeIndex=\(snap.writeIndex)"
                    )
                } catch {
                    box.recordError("Replay buffer restore failed: \(error.localizedDescription)")
                    SessionLogger.shared.log(
                        "[CHECKPOINT] Replay buffer restore failed: \(error.localizedDescription) — continuing with empty buffer"
                    )
                }
            }

            // Trainer ID: on a fresh start, mint a new one. On a
            // resume, inherit the trainer ID from the loaded session
            // so the audit trail stays continuous.
            await MainActor.run {
                if let resumed = pendingLoadedSession {
                    trainer.identifier = ModelID(value: resumed.trainerFile.modelID)
                } else {
                    trainer.identifier = ModelIDMinter.mintTrainerGeneration(from: network.identifier ?? ModelIDMinter.mint())
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

            // Build the shared self-play batcher and driver. All slots
            // play against one `ChessMPSNetwork` (the champion, via
            // `network`) through a `BatchedMoveEvaluationSource` actor
            // that coalesces N per-ply forward passes into one batched
            // `graph.run`. The driver owns the slot lifecycle: it
            // spawns up to `countBox.count` child tasks, responds to
            // Stepper-driven count changes, and pauses the whole
            // self-play subsystem through the shared `selfPlayGate`
            // when arena requests it (replacing the previous
            // per-worker gate array).
            let selfPlayBatcher = BatchedMoveEvaluationSource(network: network)
            let selfPlayDriver = BatchedSelfPlayDriver(
                batcher: selfPlayBatcher,
                buffer: buffer,
                statsBox: pStatsBox,
                diversityTracker: spDiversityTracker,
                countBox: countBox,
                pauseGate: selfPlayGate,
                gameWatcher: gameWatcher,
                scheduleBox: scheduleBox
            )

            await withTaskGroup(of: Void.self) { group in
                // Self-play driver: manages N concurrent
                // `ChessMachine` game loops against the shared
                // batcher. Slot count tracks `countBox.count` live;
                // pause requests on `selfPlayGate` flow through the
                // driver and stop every slot at its next game
                // boundary. Each slot streams positions into the
                // shared replay buffer via its white/black
                // `MPSChessPlayer` pair, records stats through
                // `pStatsBox`, and feeds `spDiversityTracker`.
                //
                // Slot 0 drives the live-display `GameWatcher` when
                // there is exactly one active slot; all slots
                // contribute identically to `pStatsBox` /
                // `spDiversityTracker`.
                group.addTask(priority: .userInitiated) {
                    [selfPlayDriver] in
                    await selfPlayDriver.run()
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

                        // The enclosing worker already runs at
                        // `.userInitiated` so there's nothing to escape
                        // to — run the SGD step inline and skip the
                        // per-step detached-task + continuation
                        // allocation pair. Sampling happens inside the
                        // trainer on its serial queue so replay-buffer
                        // rows are copied directly into trainer-owned
                        // staging buffers.
                        let timing: TrainStepTiming
                        do {
                            guard let sampledTiming = try await trainer.trainStep(
                                replayBuffer: buffer,
                                batchSize: Self.trainingBatchSize
                            ) else {
                                try? await Task.sleep(for: .milliseconds(100))
                                continue
                            }
                            timing = sampledTiming
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
                            currentBufferTotal: buffer.totalPositionsAdded,
                            stepTimeMs: timing.totalMs
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
                    [trainer, network, tBox, selfPlayGate, trainingGate, arenaFlag, triggerBox, overrideBox,
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
                                arenaFlag: arenaFlag,
                                overrideBox: overrideBox
                            )
                            triggerBox.recordArenaCompleted()
                        } else {
                            try? await Task.sleep(for: .milliseconds(500))
                        }
                    }
                }

                // Periodic session-log ticker. Emits one [STATS] line
                // per training step for the first 500 steps (every step
                // matters during bootstrap — you want to see the curve
                // shape of the first few hundred updates) then drops to
                // one line per 60 seconds for the rest of the session.
                //
                // Each wake-up snapshots the thread-safe stats boxes,
                // optionally refreshes `legalMass` via a sampled
                // forward pass (cadence-gated so the CPU work doesn't
                // pile up during the per-step bootstrap window), and
                // writes one `[STATS]` line. Identifiers are pulled
                // through a brief MainActor hop since they live on
                // classes whose var mutation is otherwise main-actor-
                // driven.
                group.addTask(priority: .utility) {
                    [trainer, network, box, pStatsBox, buffer, spDiversityTracker, ratioController, countBox, scheduleBox] in
                    let sessionStart = Date()
                    // Bootstrap-phase step threshold for the per-step
                    // emit. `Self.bootstrapStatsStepCount` is tunable
                    // on the view; at default 500 steps this covers
                    // roughly the first 1-3 minutes of real-data
                    // training at typical throughput.
                    let bootstrapSteps = Self.bootstrapStatsStepCount
                    // Time between STATS emits after the bootstrap
                    // window closes. 60 s chosen so a session's
                    // steady-state log file grows at a manageable rate
                    // (~60 lines/hr) while still capturing drift
                    // inside the typical 30-minute arena cadence.
                    let steadyInterval: TimeInterval = 60
                    // Cadence for refreshing legalMass during the
                    // per-step bootstrap window — refreshing every
                    // step would double per-step CPU cost for little
                    // additional signal. Every 25 steps roughly
                    // matches the 60-second cadence used afterwards
                    // at typical throughput.
                    let legalMassBootstrapStride = 25
                    let legalMassSampleSize = 128

                    func logOne(elapsedTarget: TimeInterval, legalMassOverride: ChessTrainer.LegalMassSnapshot?) async {
                        let trainingSnap = box.snapshot()
                        let parallelSnap = pStatsBox.snapshot()
                        let bufCount = buffer.count
                        let bufCap = buffer.capacity
                        let ratioSnap = ratioController.snapshot()
                        let workerN = countBox.count
                        let spSched = scheduleBox.selfPlay
                        let arSched = scheduleBox.arena
                        let (trainerID, championID, lr, entropyCoeff, drawPen, weightDec, gradClip, kScale) = await MainActor.run {
                            (
                                trainer.identifier?.description ?? "?",
                                network.identifier?.description ?? "?",
                                trainer.learningRate,
                                trainer.entropyRegularizationCoeff,
                                trainer.drawPenalty,
                                trainer.weightDecayC,
                                trainer.gradClipMaxNorm,
                                trainer.policyScaleK
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
                        let gradNormStr: String
                        if let g = trainingSnap.rollingGradGlobalNorm {
                            gradNormStr = String(format: "%.3f", g)
                        } else {
                            gradNormStr = "--"
                        }
                        let vMeanStr: String
                        if let vm = trainingSnap.rollingValueMean {
                            vMeanStr = String(format: "%+.4f", vm)
                        } else {
                            vMeanStr = "--"
                        }
                        let vAbsStr: String
                        if let va = trainingSnap.rollingValueAbsMean {
                            vAbsStr = String(format: "%.4f", va)
                        } else {
                            vAbsStr = "--"
                        }
                        // vBaseDelta = mean abs delta between trainer's
                        // current v(s) and the play-time-frozen vBaseline
                        // from the random-init champion. Higher = trainer
                        // is genuinely diverging from the champion.
                        let vBaseDeltaStr: String
                        if let vbd = trainingSnap.rollingVBaselineDelta {
                            vBaseDeltaStr = String(format: "%.4f", vbd)
                        } else {
                            vBaseDeltaStr = "--"
                        }
                        let h = Int(elapsedTarget) / 3600
                        let m = (Int(elapsedTarget) % 3600) / 60
                        let s = Int(elapsedTarget) % 60
                        let elapsedStr = String(format: "%d:%02d:%02d", h, m, s)
                        let spTau = String(format: "%.2f/%.2f/%.3f", spSched.startTau, spSched.floorTau, spSched.decayPerPly)
                        let arTau = String(format: "%.2f/%.2f/%.3f", arSched.startTau, arSched.floorTau, arSched.decayPerPly)
                        let pwNormStr: String
                        if let pwn = trainingSnap.rollingPolicyHeadWeightNorm {
                            pwNormStr = String(format: "%.3f", pwn)
                        } else {
                            pwNormStr = "--"
                        }
                        let divSnap = spDiversityTracker.snapshot()
                        let divStr = divSnap.gamesInWindow > 0
                        ? String(format: "unique=%d/%d(%.0f%%) diverge=%.1f", divSnap.uniqueGames, divSnap.gamesInWindow, divSnap.uniquePercent, divSnap.avgDivergencePly)
                        : "n/a"
                        let lrStr = String(format: "%.1e", lr)
                        let ratioStr = String(format: "target=%.2f cur=%.2f prod=%.1f cons=%.1f auto=%@ delay=%dms",
                                              ratioSnap.targetRatio, ratioSnap.currentRatio,
                                              ratioSnap.productionRate, ratioSnap.consumptionRate,
                                              ratioSnap.autoAdjust ? "on" : "off", ratioSnap.computedDelayMs)
                        let outcomeStr = String(format: "wMate=%d bMate=%d stale=%d 50mv=%d 3fold=%d insuf=%d",
                                                parallelSnap.whiteCheckmates, parallelSnap.blackCheckmates,
                                                parallelSnap.stalemates, parallelSnap.fiftyMoveDraws,
                                                parallelSnap.threefoldRepetitionDraws, parallelSnap.insufficientMaterialDraws)
                        let cfgStr = "batch=\(Self.trainingBatchSize) lr=\(lrStr) promote>=\(String(format: "%.2f", Self.tournamentPromoteThreshold)) arenaGames=\(Self.tournamentGames) workers=\(workerN)"
                        let regStr = String(
                            format: "clip=%.1f decay=%.0e ent=%.1e drawPen=%.3f K=%.2f",
                            gradClip,
                            weightDec,
                            entropyCoeff,
                            drawPen,
                            kScale
                        )
                        // Average game length: lifetime and 10-min
                        // rolling window. `selfPlayPositions` counts
                        // every ply played, so dividing by the number
                        // of completed games gives the mean plies-per-
                        // game. Rolling avg tracks recent behavior;
                        // lifetime avg catches longer-term drift.
                        let lifetimeAvgLen: Double = parallelSnap.selfPlayGames > 0
                        ? Double(parallelSnap.selfPlayPositions) / Double(parallelSnap.selfPlayGames)
                        : 0
                        let rollingAvgLen: Double = parallelSnap.recentGames > 0
                        ? Double(parallelSnap.recentMoves) / Double(parallelSnap.recentGames)
                        : 0
                        let p50Str: String
                        let p95Str: String
                        if let p50 = parallelSnap.gameLenP50 {
                            p50Str = String(p50)
                        } else {
                            p50Str = "--"
                        }
                        if let p95 = parallelSnap.gameLenP95 {
                            p95Str = String(p95)
                        } else {
                            p95Str = "--"
                        }
                        let gameLenStr = String(format: "avgLen=%.1f rollingAvgLen=%.1f p50=\(p50Str) p95=\(p95Str)", lifetimeAvgLen, rollingAvgLen)
                        // New encoding/gradient-health signals.
                        let playedProbStr: String
                        if let pm = trainingSnap.rollingPlayedMoveProb {
                            playedProbStr = String(format: "%.4f", pm)
                        } else {
                            playedProbStr = "--"
                        }
                        let pLogitMaxStr: String
                        if let pm = trainingSnap.rollingPolicyLogitAbsMax {
                            pLogitMaxStr = String(format: "%.3f", pm)
                        } else {
                            pLogitMaxStr = "--"
                        }
                        // Advantage distribution summary. Lots of
                        // fields but they go in one parenthesized
                        // block in the line so grep for
                        // "adv=(" when analyzing.
                        func advFmt(_ d: Double?) -> String {
                            guard let d else { return "--" }
                            return String(format: "%+.4f", d)
                        }
                        func advFracFmt(_ d: Double?) -> String {
                            guard let d else { return "--" }
                            return String(format: "%.2f", d)
                        }
                        let advStr = "mean=\(advFmt(trainingSnap.rollingAdvMean)) std=\(advFmt(trainingSnap.rollingAdvStd)) min=\(advFmt(trainingSnap.rollingAdvMin)) max=\(advFmt(trainingSnap.rollingAdvMax)) frac+=\(advFracFmt(trainingSnap.rollingAdvFracPositive)) fracSmall=\(advFracFmt(trainingSnap.rollingAdvFracSmall)) p05=\(advFmt(trainingSnap.advantageP05)) p50=\(advFmt(trainingSnap.advantageP50)) p95=\(advFmt(trainingSnap.advantageP95))"
                        let legalMassStr: String
                        let top1LegalStr: String
                        if let lm = legalMassOverride {
                            legalMassStr = String(format: "%.4f", lm.legalMass)
                            top1LegalStr = String(format: "%.2f", lm.top1LegalFraction)
                        } else {
                            legalMassStr = "--"
                            top1LegalStr = "--"
                        }
                        let line = "[STATS] elapsed=\(elapsedStr) steps=\(trainingSnap.stats.steps) spGames=\(parallelSnap.selfPlayGames) spMoves=\(parallelSnap.selfPlayPositions) \(gameLenStr) buffer=\(bufCount)/\(bufCap) pLoss=\(policyStr) vLoss=\(valueStr) pEnt=\(entropyStr) gNorm=\(gradNormStr) pwNorm=\(pwNormStr) pLogitAbsMax=\(pLogitMaxStr) playedMoveProb=\(playedProbStr) legalMass=\(legalMassStr) top1Legal=\(top1LegalStr) vMean=\(vMeanStr) vAbs=\(vAbsStr) vBaseDelta=\(vBaseDeltaStr) adv=(\(advStr)) sp.tau=\(spTau) ar.tau=\(arTau) diversity=\(divStr) ratio=(\(ratioStr)) outcomes=(\(outcomeStr)) \(cfgStr) reg=(\(regStr)) build=\(BuildInfo.buildNumber) trainer=\(trainerID) champion=\(championID)"
                        SessionLogger.shared.log(line)

                        // Policy-entropy alarm: fires whenever the
                        // rolling entropy (computed over the training
                        // stats window, same as logged above) is below
                        // the threshold. Co-located with the [STATS]
                        // emit so the cadence matches — the log
                        // adjacent lines always tell a consistent
                        // story. Skipped if entropy isn't yet
                        // available (training hasn't started).
                        if let entropy = trainingSnap.rollingPolicyEntropy,
                           entropy < Self.policyEntropyAlarmThreshold {
                            SessionLogger.shared.log(
                                "[ALARM] policy entropy \(String(format: "%.4f", entropy)) < \(String(format: "%.2f", Self.policyEntropyAlarmThreshold)) — policy may be collapsing (steps=\(trainingSnap.stats.steps))"
                            )
                        }
                    }

                    // Cache the most recent legalMass probe result so
                    // we can include it in back-to-back per-step emits
                    // without paying the ~5-20 ms forward-pass cost on
                    // every single step. Refreshed every
                    // `legalMassBootstrapStride` steps during bootstrap
                    // and every time-based emit afterward.
                    var lastLegalMass: ChessTrainer.LegalMassSnapshot? = nil
                    var lastEmittedStep: Int = -1
                    var bootstrapDone = false

                    // Bootstrap phase: poll at short interval, emit one
                    // line per new training step until
                    // bootstrapSteps steps have been logged.
                    while !Task.isCancelled && !bootstrapDone {
                        let trainingSnap = box.snapshot()
                        let steps = trainingSnap.stats.steps
                        if steps > lastEmittedStep && steps > 0 {
                            // Refresh legalMass on a stride so
                            // back-to-back per-step emits share the
                            // most recent probe.
                            if steps == 1 || (steps - max(0, lastEmittedStep)) >= legalMassBootstrapStride {
                                if buffer.count >= legalMassSampleSize {
                                    lastLegalMass = (try? await trainer.legalMassSnapshot(
                                        replayBuffer: buffer,
                                        sampleSize: legalMassSampleSize
                                    )) ?? lastLegalMass
                                }
                            }
                            let elapsed = Date().timeIntervalSince(sessionStart)
                            await logOne(elapsedTarget: elapsed, legalMassOverride: lastLegalMass)
                            lastEmittedStep = steps
                            if steps >= bootstrapSteps {
                                bootstrapDone = true
                                break
                            }
                        }
                        // Short poll — a training step at typical
                        // throughput completes in 50-200 ms, and we
                        // want the per-step cadence to track closely.
                        do {
                            try await Task.sleep(for: .milliseconds(50))
                        } catch {
                            return
                        }
                    }
                    if Task.isCancelled { return }

                    // Steady-state: one emit every `steadyInterval`
                    // seconds. Each emit refreshes the legalMass probe
                    // too.
                    while !Task.isCancelled {
                        do {
                            try await Task.sleep(for: .seconds(steadyInterval))
                        } catch {
                            return
                        }
                        if Task.isCancelled { return }
                        if buffer.count >= legalMassSampleSize {
                            lastLegalMass = (try? await trainer.legalMassSnapshot(
                                replayBuffer: buffer,
                                sampleSize: legalMassSampleSize
                            )) ?? lastLegalMass
                        }
                        let elapsed = Date().timeIntervalSince(sessionStart)
                        await logOne(elapsedTarget: elapsed, legalMassOverride: lastLegalMass)
                    }
                }

                // Wait for all four tasks to complete (only happens
                // on cancellation since each loops forever).
                for await _ in group { }
            }

            await MainActor.run {
                clearTrainingAlarm()
                realTraining = false
                realTrainingTask = nil
                isArenaRunning = false
                arenaActiveFlag = nil
                arenaTriggerBox = nil
                arenaOverrideBox = nil
                parallelWorkerStatsBox = nil
                parallelStats = nil
                currentDiversityHistogramBars = []
                arenaChartEvents = []
                activeArenaStartElapsed = nil
                workerCountBox = nil
                trainingStepDelayBox = nil
                samplingScheduleBox = nil
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
        clearTrainingAlarm()
        // Close the in-progress training segment so cumulative wall-time
        // totals exclude post-Stop idle. If saving immediately after,
        // buildCurrentSessionState will see the segment already closed
        // and won't double-count.
        closeActiveTrainingSegment(reason: "stop")
    }

    /// Begin a new training segment when Play-and-Train starts.
    /// Captures starting counter snapshots and the active build/git
    /// metadata so the resulting segment can be attributed to a
    /// specific code version after-the-fact.
    private func beginActiveTrainingSegment() {
        let now = Date()
        let bufferAdded = replayBuffer?.totalPositionsAdded ?? 0
        let snap = parallelStats
        activeSegmentStart = ActiveSegmentStart(
            startUnix: Int64(now.timeIntervalSince1970),
            startDate: now,
            startingTrainingStep: trainingStats?.steps ?? 0,
            startingTotalPositions: bufferAdded,
            startingSelfPlayGames: snap?.selfPlayGames ?? 0,
            buildNumber: BuildInfo.buildNumber,
            buildGitHash: BuildInfo.gitHash,
            buildGitDirty: BuildInfo.gitDirty
        )
        SessionLogger.shared.log(
            "[SEGMENT] start (segment #\(completedTrainingSegments.count + 1)) "
            + "step=\(activeSegmentStart?.startingTrainingStep ?? 0) "
            + "build=\(BuildInfo.buildNumber)"
        )
    }

    /// Close the in-progress segment with current end-of-segment
    /// counters and append it to `completedTrainingSegments`. Idempotent
    /// — if no segment is active, returns silently. Called from Stop,
    /// from the save path, and from session-end. `reason` is only used
    /// for the log line; the segment data itself is reason-agnostic.
    private func closeActiveTrainingSegment(reason: String) {
        guard let start = activeSegmentStart else { return }
        let now = Date()
        let endUnix = Int64(now.timeIntervalSince1970)
        let durationSec = max(0, now.timeIntervalSince(start.startDate))
        let snap = parallelStats
        let trainingSnap = trainingStats
        let liveSnap = trainingBox?.snapshot()
        let bufferAdded = replayBuffer?.totalPositionsAdded ?? start.startingTotalPositions
        let endLoss: Double? = {
            guard let p = liveSnap?.rollingPolicyLoss,
                  let v = liveSnap?.rollingValueLoss else { return nil }
            return p + v
        }()
        let segment = SessionCheckpointState.TrainingSegment(
            startUnix: start.startUnix,
            endUnix: endUnix,
            durationSec: durationSec,
            startingTrainingStep: start.startingTrainingStep,
            endingTrainingStep: trainingSnap?.steps ?? start.startingTrainingStep,
            startingTotalPositions: start.startingTotalPositions,
            endingTotalPositions: bufferAdded,
            startingSelfPlayGames: start.startingSelfPlayGames,
            endingSelfPlayGames: snap?.selfPlayGames ?? start.startingSelfPlayGames,
            buildNumber: start.buildNumber,
            buildGitHash: start.buildGitHash,
            buildGitDirty: start.buildGitDirty,
            endPolicyEntropy: liveSnap?.rollingPolicyEntropy,
            endLossTotal: endLoss,
            endGradNorm: liveSnap?.rollingGradGlobalNorm
        )
        completedTrainingSegments.append(segment)
        activeSegmentStart = nil
        SessionLogger.shared.log(
            String(format: "[SEGMENT] close (%@) duration=%.1fs steps=%d -> %d positions=%d -> %d",
                   reason,
                   durationSec,
                   segment.startingTrainingStep,
                   segment.endingTrainingStep,
                   segment.startingTotalPositions,
                   segment.endingTotalPositions)
        )
    }

    /// Total active training wall-time across all segments, including
    /// the currently-running one if any. Excludes any time when
    /// training was stopped — sum of segment durations only.
    private var cumulativeActiveTrainingSec: Double {
        let completed = completedTrainingSegments.reduce(0.0) { $0 + $1.durationSec }
        let active = activeSegmentStart.map { Date().timeIntervalSince($0.startDate) } ?? 0
        return completed + max(0, active)
    }

    /// Total run count: segments closed + 1 if a run is currently
    /// active. Useful for "this session has had N runs."
    private var cumulativeRunCount: Int {
        completedTrainingSegments.count + (activeSegmentStart != nil ? 1 : 0)
    }

    /// Cumulative status bar — sums across all completed
    /// Play-and-Train segments + the in-flight one. Visible
    /// whenever this session has had any training (current run
    /// or a hydrated history from a loaded session). Hidden on
    /// a fresh session that has never trained, since all values
    /// would be zero.
    ///
    /// Always includes a small `Run Arena` button when an arena
    /// is not in progress and a network exists, so the user can
    /// kick off an arena from a glanceable spot without hunting
    /// the menu. Hidden during arena runs to avoid double-fire.
    ///
    /// Pulled out into its own computed view to keep the main `body`
    /// under SwiftUI's type-checker complexity threshold.
    @ViewBuilder
    private var cumulativeStatusBar: some View {
        let totalSteps = trainingStats?.steps ?? 0
        let hasHistory = cumulativeRunCount > 0 || totalSteps > 0
        let canRunArena = !isArenaRunning && network != nil && trainer != nil
        if hasHistory || canRunArena {
            let totalPositions = totalSteps * Self.trainingBatchSize
            let activeSec = cumulativeActiveTrainingSec
            HStack(spacing: 16) {
                if hasHistory {
                    statusBarItem(
                        label: "Active training time",
                        value: GameWatcher.Snapshot.formatHMS(seconds: activeSec)
                    )
                    statusBarItem(
                        label: "Training steps",
                        value: Int(totalSteps).formatted()
                    )
                    statusBarItem(
                        label: "Positions trained",
                        value: Self.formatCompactCount(totalPositions)
                    )
                    statusBarItem(
                        label: "Runs",
                        value: "\(cumulativeRunCount)"
                    )
                }
                Spacer()
                if canRunArena {
                    Button {
                        // Defensive guard mirroring the menu's runArena
                        // closure (line ~4125) — the parent view hides
                        // this button when isArenaRunning=true, but a
                        // rapid double-click could still slip through
                        // before SwiftUI re-evaluates. Guard makes
                        // double-fire a no-op.
                        guard !isArenaRunning else { return }
                        SessionLogger.shared.log("[BUTTON] Run Arena (status-bar)")
                        arenaTriggerBox?.trigger()
                    } label: {
                        Label("Run Arena Now", systemImage: "flag.checkered")
                            .font(.callout)
                    }
                    .controlSize(.small)
                }
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
            .background(Color.secondary.opacity(0.10))
            .clipShape(RoundedRectangle(cornerRadius: 6))
        }
    }

    /// Build a single status-bar cell — small caption label above a
    /// monospaced numeric value. Pulled out so the four cells in the
    /// status bar share consistent typography.
    @ViewBuilder
    private func statusBarItem(label: String, value: String) -> some View {
        VStack(alignment: .leading, spacing: 1) {
            Text(label)
                .font(.caption2)
                .foregroundStyle(.secondary)
            Text(value)
                .font(.system(.callout, design: .monospaced).weight(.semibold))
                .monospacedDigit()
        }
    }

    // MARK: - Engine Diagnostics

    /// Run a one-shot battery of correctness probes and log results
    /// with `[DIAG]` prefix. Designed to be triggered on demand after
    /// significant code changes (architecture refactors, encoder
    /// changes) so the user can confirm the engine still passes basic
    /// invariants without waiting for a full training session to
    /// surface any regression.
    ///
    /// Probes:
    ///   1. PolicyEncoding round-trip across all legal moves at the
    ///      starting position.
    ///   2. PolicyEncoding round-trip in a position with promotions.
    ///   3. PolicyEncoding distinct-index check (no two legal moves
    ///      share an index).
    ///   4. ChessGameEngine 3-fold detection on knight shuffle.
    ///   5. BoardEncoder produces correct tensor length.
    ///   6. Network forward pass shape check (if a network exists).
    ///
    /// Designed to complete in well under a second so the user sees
    /// immediate pass/fail feedback. Results go to the session log,
    /// not to a dialog — the log is the canonical record.
    private func runEngineDiagnostics() {
        SessionLogger.shared.log("[BUTTON] Engine Diagnostics")
        // Wrap in a Task so we can `await` the network's async
        // evaluate cleanly. Pure-logic probes run synchronously
        // inside; only the network probe needs the await. Failures
        // are reported via the [DIAG] log lines, not via UI alerts.
        let networkRef = network
        Task {
            await runEngineDiagnosticsAsync(net: networkRef)
        }
    }

    private func runEngineDiagnosticsAsync(net: ChessMPSNetwork?) async {
        SessionLogger.shared.log("[DIAG] === Engine diagnostics begin ===")
        var failed = 0
        var ran = 0

        func check(_ name: String, _ predicate: () -> Bool) {
            ran += 1
            if predicate() {
                SessionLogger.shared.log("[DIAG] PASS  \(name)")
            } else {
                SessionLogger.shared.log("[DIAG] FAIL  \(name)")
                failed += 1
            }
        }

        // 1. PolicyEncoding round-trip on starting position.
        check("PolicyEncoding round-trip at starting position") {
            let state = GameState.starting
            let legals = MoveGenerator.legalMoves(for: state)
            for move in legals {
                let (chan, r, c) = PolicyEncoding.encode(move, currentPlayer: state.currentPlayer)
                guard let decoded = PolicyEncoding.decode(channel: chan, row: r, col: c, state: state),
                      decoded == move else { return false }
            }
            return !legals.isEmpty
        }

        // 2. Round-trip with promotions on the board.
        check("PolicyEncoding round-trip with promotions") {
            var board: [Piece?] = Array(repeating: nil, count: 64)
            board[7 * 8 + 0] = Piece(type: .king, color: .white)
            board[0 * 8 + 7] = Piece(type: .king, color: .black)
            for col in 1..<7 { board[1 * 8 + col] = Piece(type: .pawn, color: .white) }
            let state = GameState(
                board: board, currentPlayer: .white,
                whiteKingsideCastle: false, whiteQueensideCastle: false,
                blackKingsideCastle: false, blackQueensideCastle: false,
                enPassantSquare: nil, halfmoveClock: 0
            )
            let legals = MoveGenerator.legalMoves(for: state)
            for move in legals {
                let (chan, r, c) = PolicyEncoding.encode(move, currentPlayer: state.currentPlayer)
                guard let decoded = PolicyEncoding.decode(channel: chan, row: r, col: c, state: state),
                      decoded == move else { return false }
            }
            // Verify all 4 promotion variants are distinct
            let promos = legals.filter { $0.promotion != nil && $0.fromCol == 1 && $0.toCol == 1 }
            let promoIndices = Set(promos.map {
                PolicyEncoding.policyIndex($0, currentPlayer: state.currentPlayer)
            })
            return promos.count == 4 && promoIndices.count == 4
        }

        // 3. Distinct policy indices for all legal moves.
        check("PolicyEncoding produces distinct indices for legal moves") {
            let legals = MoveGenerator.legalMoves(for: .starting)
            let indices = legals.map { PolicyEncoding.policyIndex($0, currentPlayer: .white) }
            return Set(indices).count == indices.count
        }

        // 4. 3-fold detection via knight shuffle.
        check("ChessGameEngine detects 3-fold via knight shuffle") {
            let engine = ChessGameEngine()
            let nf3 = ChessMove(fromRow: 7, fromCol: 6, toRow: 5, toCol: 5, promotion: nil)
            let nc6 = ChessMove(fromRow: 0, fromCol: 1, toRow: 2, toCol: 2, promotion: nil)
            let ng1 = ChessMove(fromRow: 5, fromCol: 5, toRow: 7, toCol: 6, promotion: nil)
            let nb8 = ChessMove(fromRow: 2, fromCol: 2, toRow: 0, toCol: 1, promotion: nil)
            for _ in 0..<2 {
                _ = try? engine.applyMoveAndAdvance(nf3)
                _ = try? engine.applyMoveAndAdvance(nc6)
                _ = try? engine.applyMoveAndAdvance(ng1)
                _ = try? engine.applyMoveAndAdvance(nb8)
            }
            if case .drawByThreefoldRepetition = engine.result { return true }
            return false
        }

        // 5. BoardEncoder shape check.
        check("BoardEncoder produces tensorLength floats (= \(BoardEncoder.tensorLength))") {
            let tensor = BoardEncoder.encode(.starting)
            return tensor.count == BoardEncoder.tensorLength
        }

        // 6. Network forward-pass shape (only if a network is built).
        if let net = net {
            ran += 1
            do {
                let board = BoardEncoder.encode(.starting)
                let (policy, _) = try await net.evaluate(board: board)
                if policy.count == ChessNetwork.policySize {
                    SessionLogger.shared.log(
                        "[DIAG] PASS  Network forward-pass produces \(ChessNetwork.policySize) logits"
                    )
                } else {
                    SessionLogger.shared.log(
                        "[DIAG] FAIL  Network forward-pass: expected \(ChessNetwork.policySize) logits, got \(policy.count)"
                    )
                    failed += 1
                }
            } catch {
                SessionLogger.shared.log(
                    "[DIAG] FAIL  Network forward-pass error: \(error.localizedDescription)"
                )
                failed += 1
            }
        } else {
            SessionLogger.shared.log("[DIAG] SKIP  Network forward-pass shape (no network built yet)")
        }

        SessionLogger.shared.log("[DIAG] === Engine diagnostics done: \(ran - failed)/\(ran) passed ===")
    }

    /// Compact human-readable count: 12,345 → "12.3K", 1,234,567 →
    /// "1.23M", 1,234,567,890 → "1.23B". Status-bar version of the
    /// "Positions trained" cell — keeps the label narrow regardless
    /// of order-of-magnitude.
    static func formatCompactCount(_ value: Int) -> String {
        let abs = Swift.abs(value)
        switch abs {
        case 0..<1_000:
            return "\(value)"
        case 1_000..<1_000_000:
            return String(format: "%.1fK", Double(value) / 1_000)
        case 1_000_000..<1_000_000_000:
            return String(format: "%.2fM", Double(value) / 1_000_000)
        default:
            return String(format: "%.2fB", Double(value) / 1_000_000_000)
        }
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
                try await trainer.resetNetwork()
            } catch {
                await MainActor.run {
                    trainingError = "Reset failed: \(error.localizedDescription)"
                    sweepRunning = false
                    sweepCancelBox = nil
                }
                return
            }

            let result = await Self.runSweep(
                trainer: trainer,
                sizes: sizes,
                secondsPerSize: secondsPerSize,
                cancelBox: cancelBox
            )

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
    ) async -> Result<[SweepRow], Error> {
        do {
            return .success(try await trainer.runSweep(
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
            ))
        } catch {
            return .failure(error)
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
        // SP tau / Arena tau / clip / decay are now surfaced as editable
        // text fields above the body, so they are not duplicated here.
        // Learning rate likewise lives in the interactive text field.

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
            // Value-head diagnostics: signed mean and absolute mean of v
            // across the trainer's rolling-window batches. vMean drifting
            // strongly negative indicates the value head is over-predicting
            // losses (often a draw-penalty interaction); vAbs near 1.0
            // signals tanh saturation and impending gradient vanishing.
            if let snap = trainingBox?.snapshot() {
                let vMeanStr = snap.rollingValueMean
                    .map { String(format: "%+.4f", $0) } ?? dash
                let vAbsStr = snap.rollingValueAbsMean
                    .map { String(format: "%.4f", $0) } ?? dash
                lines.append("    v mean:        \(vMeanStr)")
                lines.append("    v abs:         \(vAbsStr)")
            }
            if let gNorm = trainingBox?.snapshot().rollingGradGlobalNorm {
                lines.append(String(format: "  Grad norm:   %.3f", gNorm))
            }
            if let pwNorm = trainingBox?.snapshot().rollingPolicyHeadWeightNorm {
                lines.append(String(format: "  pWeight ||₂:  %.3f", pwNorm))
            }
            // Ent reg / Grad clip / Weight dec / Draw pen previously
            // listed here are duplicates of the editable fields shown
            // above the loss section. Removed to avoid redundancy.
            // Candidate-test probe counter + time-since-last, so the user
            // can distinguish "probes firing but imperceptible" from "probes
            // stuck". Shown in both Game run and Candidate test modes so
            // the count is visible while Play and Train is running; the
            // count only advances when Candidate test is active and a
            // gap check actually fires a probe.
            // Probes removed from display (internal timing only).
            // 1-minute rolling rates from the replay-ratio controller
            if let snap = replayRatioSnapshot {
                // Display per-second alongside per-hour so the user can
                // map directly to the [STATS] line's `Moves/hr` figure
                // without doing the ×3600 in their head.
                let prodStr: String
                if snap.productionRate > 0 {
                    let perSec = snap.productionRate
                    let perHr = Int(perSec * 3600).formatted()
                    prodStr = String(format: "%.0f pos/s   (\(perHr)/hr)", perSec)
                } else {
                    prodStr = dash
                }
                let consStr: String
                if snap.consumptionRate > 0 {
                    let perSec = snap.consumptionRate
                    let perHr = Int(perSec * 3600).formatted()
                    consStr = String(format: "%.0f pos/s   (\(perHr)/hr)", perSec)
                } else {
                    consStr = dash
                }
                lines.append("  1m gen rate: \(prodStr)")
                lines.append("  1m trn rate: \(consStr)")
            }
            if let divSnap = selfPlayDiversityTracker?.snapshot(),
               divSnap.gamesInWindow > 0 {
                let pctStr = String(format: "%.0f%%", divSnap.uniquePercent)
                let divStr = String(format: "%.1f", divSnap.avgDivergencePly)
                lines.append("  Diversity:   \(divSnap.uniqueGames)/\(divSnap.gamesInWindow) unique (\(pctStr))  avg diverge ply \(divStr)")
            }
        }
        lines.append("")

        if let last = lastTrainStep {
            lines.append("Last Step")
            lines.append(String(format: "  Total:       %.2f ms", last.totalMs))
            lines.append(String(format: "  Entropy:     %.6f", last.policyEntropy))
            lines.append(String(format: "  Grad norm:   %.3f", last.gradGlobalNorm))
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
                // and the kind of promotion (auto vs manual) so a
                // quick scan of the stats panel tells the same
                // story as the session log's [ARENA] lines.
                let marker: String
                let kindSuffix: String
                switch record.promotionKind {
                case .automatic:
                    kindSuffix = " (auto)"
                case .manual:
                    kindSuffix = " (manual)"
                case .none:
                    kindSuffix = ""
                }
                if record.promoted, let pid = record.promotedID {
                    marker = "PROMOTED\(kindSuffix)=\(pid.description)"
                } else if record.promoted {
                    marker = "PROMOTED\(kindSuffix)"
                } else {
                    marker = "kept"
                }
                let durStr = Self.formatElapsed(record.durationSec)
                // Games played vs configured total — e.g. "12/200"
                // when the user aborted at game 12.
                let gamesStr = "\(record.gamesPlayed)/\(Self.tournamentGames)"
                lines.append(String(
                    format: "  #%@ @ %@ steps  games %@  %d-%d-%d  score %@  %@  (%@)",
                    number, stepStr, gamesStr,
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
        // (the network all self-play slots share through the
        // barrier batcher). The lifetime "Time" field
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
    ) async -> EvaluationResult {
        var lines: [String] = []
        var topMoves: [MoveVisualization] = []
        let board = BoardEncoder.encode(state)

        do {
            let inference = try await runner.evaluate(board: board, state: state, pieces: state.board)
            topMoves = inference.topMoves

            lines.append(String(format: "Forward pass: %.2f ms", inference.inferenceTimeMs))
            lines.append("")
            lines.append("Value Head")
            lines.append(String(format: "  Output: %+.6f", inference.value))
            // Removed the (v+1)/2 → "X% win / Y% loss" line. With a single
            // tanh scalar (no WDL output) and a non-zero draw penalty in
            // training, that mapping was misleading. Just show the raw
            // value; readers familiar with the engine can interpret the
            // sign and magnitude themselves.
            lines.append("")
            lines.append("Policy Head (Top 4 raw — includes illegal)")
            // The list deliberately includes illegal candidates so we
            // can see whether the network has learned move-validity.
            // After enough training, illegal cells should fall out of
            // the top-K; if they keep appearing, the policy hasn't
            // learned legality conditioning on the current position.
            for (rank, move) in inference.topMoves.enumerated() {
                let fromName = BoardEncoder.squareName(move.fromRow * 8 + move.fromCol)
                let toName = BoardEncoder.squareName(move.toRow * 8 + move.toCol)
                let rankCol = String(rank + 1).padding(toLength: 4, withPad: " ", startingAt: 0)
                let moveCol = "\(fromName)-\(toName)".padding(toLength: 8, withPad: " ", startingAt: 0)
                let legalMark = move.isLegal ? "" : "  (illegal)"
                lines.append("  \(rankCol)\(moveCol)\(String(format: "%.6f%%", move.probability * 100))\(legalMark)")
            }
            // Sum of the top-100 move probabilities. With a freshly-
            // initialized network this sits near 100/policySize ≈ 2.06%; as the
            // policy head learns to concentrate mass on promising moves,
            // this number climbs — a cheap scalar that changes visibly
            // between candidate-test probes even when the top-4 move
            // ordering stays stable.
            let top100Sum = inference.policy.sorted(by: >).prefix(100).reduce(0, +)
            lines.append(String(format: "  Top 100 sum: %.6f%%", top100Sum * 100))
            lines.append("")
            lines.append("Policy Stats")
            lines.append(String(format: "  Sum: %.8f", inference.policy.reduce(0, +)))
            // Legality-aware "above-uniform" count for THIS specific
            // position. Counts how many of the legal moves the network
            // gives mass above `1 / N_legal` (i.e., above what a
            // perfectly-uniform-over-legal policy would produce).
            // Direct, interpretable signal: at the starting position
            // there are 20 legal moves and uniform-over-legal threshold
            // is 5%; "8 / 20" means the network rates 8 of those 20
            // moves above uniform. Replaces the old "NonNegligible:
            // X / 4864" metric, which was confusing because most of
            // the 4864 cells in the new 76-channel encoding correspond
            // to physically-impossible moves and so always sit far
            // below the 1/4864 baseline regardless of training.
            let legalMoves = MoveGenerator.legalMoves(for: state)
            let nLegal = max(1, legalMoves.count)
            let legalUniformThreshold = 1.0 / Float(nLegal)
            let abovePerLegalCount = legalMoves
                .map { PolicyEncoding.policyIndex($0, currentPlayer: state.currentPlayer) }
                .filter { idx in
                    idx >= 0 && idx < inference.policy.count
                        && inference.policy[idx] > legalUniformThreshold
                }
                .count
            lines.append(String(format: "  Above uniform: %d / %d legal  (threshold = 1/%d = %.3f%%)",
                                abovePerLegalCount, nLegal, nLegal,
                                Double(legalUniformThreshold) * 100))
            // Total mass on legal moves vs illegal — at training
            // convergence, mass-on-illegal should approach zero since
            // illegal cells never appear as one-hot training targets.
            let legalMassSum = legalMoves
                .map { PolicyEncoding.policyIndex($0, currentPlayer: state.currentPlayer) }
                .reduce(Float(0)) { acc, idx in
                    (idx >= 0 && idx < inference.policy.count) ? acc + inference.policy[idx] : acc
                }
            lines.append(String(format: "  Legal mass sum: %.6f%%   (illegal = %.6f%%)",
                                Double(legalMassSum) * 100,
                                Double(1 - legalMassSum) * 100))
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
    ContentView(commandHub: AppCommandHub())
}
