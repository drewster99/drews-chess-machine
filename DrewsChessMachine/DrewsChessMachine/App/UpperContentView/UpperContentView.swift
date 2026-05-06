import AppKit
import Charts
import Metal
import SwiftUI
import UniformTypeIdentifiers
import OSLog

private let logger = Logger(subsystem: "Foo", category: "Bar")
// MARK: - Upper Content View

struct UpperContentView: View {
    /// Menu-bar command hub. Assigned by `DrewsChessMachineApp`; the
    /// view wires its action functions into the hub's closure slots
    /// and keeps the hub's mirrored state flags synced so the
    /// `.commands` DSL can enable/disable menu items correctly.
    let commandHub: AppCommandHub

    /// True iff the process was launched with `--train` on the
    /// command line. When set, the first `.onAppear` suppresses
    /// the Resume-from-Autosave sheet entirely and chains Build
    /// Network → Play-and-Train → switch the board picker to
    /// Candidate Test as an automated headless-friendly sequence.
    /// Plumbed in from `DrewsChessMachineApp` so the flag lives
    /// alongside the other process-scope wiring rather than
    /// being discovered via `CommandLine.arguments` at view time.
    let autoTrainOnLaunch: Bool

    /// Parsed `--parameters <file>` JSON. Applied to the relevant
    /// `@AppStorage` / `@State` fields right before
    /// `buildNetwork()` fires inside the auto-train sequence, so
    /// every downstream path (`ensureTrainer`, the sampling
    /// schedule builders, the worker-count binding) picks the
    /// new values up through its normal channels. Nil outside
    /// CLI-driven runs.
    let cliConfig: CliTrainingConfig?

    /// Destination URL for the `--output <file>` JSON snapshot.
    /// Presence of this URL is what switches the runtime into
    /// "headless CLI" mode: a `CliTrainingRecorder` is allocated,
    /// arena/stats/probe events are captured into it, and the
    /// `training_time_limit` deadline task writes + exits. Nil
    /// when the flag wasn't passed.
    let cliOutputURL: URL?

    /// Idempotency guard for the auto-train launch sequence.
    /// `.onAppear` can fire more than once over a view's lifetime
    /// (e.g. on window re-parenting), and the auto-train chain
    /// is a one-shot launch behavior — firing it twice would
    /// either refuse (network already built) or thrash the
    /// session. Flipped `true` the first time the sequence
    /// starts and never cleared.
    @State private var autoTrainFired: Bool = false

    /// Live recorder for `--output` runs. Allocated at the start
    /// of `startRealTraining` when `cliOutputURL` is set, and
    /// appended to by the stats/arena/probe code paths while the
    /// session is active. Nil in normal interactive runs — each
    /// capture site guards on `cliRecorder != nil` so the
    /// recording paths are zero-cost when the feature isn't
    /// engaged.
    @State private var cliRecorder: CliTrainingRecorder?

    // MARK: - CLI-overridable effective values
    //
    // These start at the compile-time defaults (the matching
    // `Self.X` static constants) and remain unchanged throughout
    // a normal interactive run. When `--parameters` specifies an
    // override, `applyCliConfigOverrides` writes the new value
    // into the matching field here. Every runtime read site that
    // formerly read `Self.X` now reads `effectiveX` instead, so
    // the override flows to self-play, training, the arena, and
    // the stats/arena log lines uniformly.
    //
    // The corresponding `Self.X` constants stay as the SSOT
    // defaults — these @State fields exist only so a runtime
    // value can shadow them. That split lets the compile-time
    // value still be referenced from anywhere that does not
    // have a `ContentView` instance in scope (e.g. type-level
    // code, tests), while a live session reads the effective
    // value.
    // Migrated to TrainingParameters.shared (see below). Properties
    // formerly stored here as @State / @AppStorage are now accessed
    // via `trainingParams.<name>`. Defaults and persistence are
    // centralized in TrainingParameters.swift.

    // Network
    @State private var network: ChessMPSNetwork?
    @State private var runner: ChessRunner?
    @State private var networkStatus = ""
    @State private var isBuilding = false
    /// Separate inference-mode network used during arena as the
    /// "candidate side" player. Distinct from `network` (the
    /// "champion" used by self-play / Play Game / Run Forward Pass)
    /// because we explicitly want the champion to stay frozen at
    /// whatever weights it was built with until a future arena-based
    /// promotion step decides otherwise. The trainer's current SGD
    /// state is copied into this network at arena start so the
    /// candidate plays its tournament games from a coherent, stable
    /// snapshot rather than drifting mid-match as training continues.
    ///
    /// Cached across Play and Train sessions so the ~100 ms
    /// MPSGraph-build cost only happens once per app launch, not on
    /// every start.
    @State private var candidateInferenceNetwork: ChessMPSNetwork?
    /// Fifth network — dedicated exclusively to the candidate-test
    /// probe. An earlier design had the probe share
    /// `candidateInferenceNetwork` with the arena, which forced the
    /// probe to pause for the entire duration of every arena (the
    /// arena's games read the network while the probe would want to
    /// overwrite it with a fresh trainer snapshot). Pausing produced
    /// a visible discontinuity in `candidate_tests[]` across every
    /// arena boundary — before the gap the probe tracked the trainer
    /// smoothly, after the gap it jumped to reflect ~one-arena-
    /// duration of accumulated SGD. Giving the probe its own network
    /// removes that conflict, so the probe can fire through arenas
    /// and the trajectory stays continuous.
    ///
    /// Built lazily on the first Play and Train session and cached
    /// for the life of the app, same as the other inference
    /// networks.
    @State private var probeInferenceNetwork: ChessMPSNetwork?
    /// ChessRunner wrapping `probeInferenceNetwork`. Used by
    /// `fireCandidateProbeIfNeeded` via the same `performInference`
    /// code path as the pure forward-pass mode.
    @State private var probeRunner: ChessRunner?
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
    // Diversity histogram, completed arena events, and the live
    // arena-start marker live on `chartCoordinator` (see
    // `ChartCoordinator.swift`). Heartbeat updates them via the
    // coordinator's `setDiversityHistogramBars(_:)`,
    // `recordArenaCompleted(_:)`, and `recordArenaStarted(elapsedSec:)`
    // helpers; `LowerContentView` reads them directly from the
    // coordinator.
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

    /// View-menu toggle: when on, the 76-channel policy panel is
    /// rendered to the right of the chess board, sourced from
    /// whatever inference result is currently driving the on-board
    /// Top Moves overlay (Forward Pass / Candidate Test).
    @AppStorage("showPolicyChannelsPanel") private var showPolicyChannelsPanel: Bool = false

    // Inference
    @State private var inferenceResult: EvaluationResult?
    @State private var isEvaluating = false
    /// Mini-board view mode index. `-1` = plain chess board, no
    /// overlay and no channel-strip selection (the new default — just
    /// show the board). `0` = Top Moves overlay (policy arrows). `1...
    /// inputPlanes` = the matching channel of the input tensor.
    /// Right-paging walks left → right through that ordering.
    @State private var selectedOverlay = -1
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
    /// Status-bar "Score" cell display mode toggle. `false` =
    /// percentage view (e.g. `"51.2%"`), `true` = Elo-with-CI view
    /// (e.g. `"+28 [+13, +34]"`). Click on the cell flips it. Session-
    /// local only — not persisted.
    @State private var scoreStatusShowElo: Bool = false

    // MARK: - Chart zoom state
    //
    // Discrete-stop zoom on the training chart grid's horizontal
    // time window. See `ChartZoom` for the list of stops (15m..30d)
    // and the one-hour auto-re-enable rule.

    // Chart-zoom state (`chartCoordinator.chartZoomIdx`, `chartCoordinator.chartZoomAuto`,
    // `chartCoordinator.lastManualChartZoomAt`) lives on `chartCoordinator`. Manual
    // zoom commands route through `chartCoordinator.zoomIn()` /
    // `zoomOut()` / `setAutoZoom(_:)`; the heartbeat zoom tick
    // calls `chartCoordinator.refreshZoomTick()`. The auto-
    // re-engage timeout constant is also on the coordinator
    // (`ChartCoordinator.chartZoomAutoReengageSec`).

    /// Composite version stamp combining both zoom-state scalars
    /// so a single `.onChange` handler can react to either. Pulled
    /// out of two separate `.onChange(of:)` modifiers to keep the
    /// body modifier chain short enough for SwiftUI's type-checker
    /// to handle without hitting the "expression too complex"
    /// ceiling.
    private var chartZoomStateVersion: Int {
        chartCoordinator.chartZoomIdx * 2 + (chartCoordinator.chartZoomAuto ? 1 : 0)
    }
    /// Total number of games a single arena plays. 200 gives us enough
    /// decisive games (~26 at the current ~13% decisive rate with
    /// random networks) for the 0.55 score threshold to be meaningful.
    /// Colors alternate every game, so candidate and champion each get
    /// 100 games as white and 100 as black.
    nonisolated static let tournamentGames = 200
    /// Default number of arena games run concurrently per tournament.
    /// Each game uses two shared `BatchedMoveEvaluationSource` batchers
    /// (one per network) so the GPU sees K-position batches instead of
    /// K serial single-position calls. 32 keeps the GPU well-saturated
    /// without monopolizing it against the concurrent training worker
    /// — the per-network barrier fires K=32 batches in the early game,
    /// and the 5 ms coalescing window catches the desynchronized
    /// steady-state at ~K/2. The value is user-overridable via the
    /// `trainingParams.arenaConcurrency` runtime setting (UI Stepper /
    /// session save+load / parameters.json key `arena_concurrency`).
    nonisolated static let arenaConcurrencyDefault = 200
    /// Hard ceiling on `trainingParams.arenaConcurrency`. Mirrors the
    /// self-play `absoluteMaxSelfPlayWorkers` pattern: bounds the
    /// UI Stepper AND clamps values loaded from `parameters.json` /
    /// `session.json` so a stale or hand-edited file can't push K
    /// past what the GPU can usefully batch in one fire. 256 is
    /// well past the per-batch GPU throughput knee on Apple
    /// Silicon for this network — raising it further wouldn't help
    /// arena throughput because batches that large stall on memory
    /// bandwidth before they finish.
    nonisolated static let absoluteMaxArenaConcurrency: Int = 1024
    /// Coalescing-window upper bound (ms) for the arena's per-network
    /// batchers. The barrier fires on either count-met OR window-
    /// elapsed, whichever happens first; the window only kicks in when
    /// games desynchronize and the count barrier alone wouldn't fire.
    /// 100 ms gives close-to-K-sized batches plenty of room to
    /// assemble in the desynchronized steady-state without making any
    /// single slot pay too much wall-clock for the coalesce.
    /// Hardcoded for now; promote to a UI / persisted setting only
    /// if profiling motivates it.
    nonisolated static let arenaBatchWaitMs: Double = 100.0
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
    /// with no main-actor hop; the 10 Hz `snapshotTimer` poller mirrors the
    /// latest values into `trainingStats` / `lastTrainStep` /
    /// `realRollingPolicyLoss` / `realRollingValueLoss` only when the step
    /// count has actually advanced.
    @State private var trainingBox: TrainingLiveStatsBox?

    /// Mirror of the trainer's warmup-relevant state, refreshed by the
    /// snapshot timer. Captures the data the top-bar Training Status
    /// chip and the LR Warm-up status cell need without touching the
    /// trainer from inside `body`. `effectiveLR` is the same value the
    /// optimizer is being fed this step (warmup × sqrt-batch × base).
    private struct TrainerWarmupSnapshot: Equatable {
        var completedSteps: Int
        var warmupSteps: Int
        var effectiveLR: Float
        var inWarmup: Bool { warmupSteps > 0 && completedSteps < warmupSteps }
    }
    @State private var trainerWarmupSnap: TrainerWarmupSnapshot?

    nonisolated static let trainerLearningRateDefault: Float = 5e-5
    nonisolated static let entropyRegularizationCoeffDefault: Float = 1e-3
    nonisolated static let drawPenaltyDefault: Float = 0.1
    nonisolated static let trainingBatchSize = 4096

    /// Policy-entropy floor below which the periodic stats ticker
    /// emits an `[ALARM]` log line.
    ///
    /// Since legal-move masking (acc5340), the training graph computes
    /// entropy over the post-mask softmax — only legal moves. The
    /// theoretical max is now `log(avg_legal_moves)`, roughly
    /// ln(30) ≈ 3.4 nats in typical midgame. A fresh network starts
    /// at pEnt ≈ 1.9 nats empirically.
    ///
    /// Threshold of 1.0 flags genuine collapse (≈ 2.7 effective legal
    /// moves), leaving a ~0.9-nat margin below fresh-init baseline.
    /// Normal training concentrates the policy over time, so moderate
    /// decline from 1.9 is expected and healthy.
    nonisolated static let policyEntropyAlarmThreshold: Double = 1.0
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
    /// Post-mask entropy critical floor: ≈ 1.6 effective legal moves.
    nonisolated static let divergenceAlarmEntropyCriticalThreshold: Double = 0.5
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
    /// (bounded MSE) components. Mirrored from `trainingBox` by the 10 Hz
    /// poller. The windows themselves live inside the box — these are just
    /// the most recent display values. Split so we can tell whether an
    /// oscillating total-loss plot is the bounded value term going unstable
    /// (training problem) or the unbounded policy term bouncing around
    /// (usually just metric noise).
    @State private var realRollingPolicyLoss: Double?
    @State private var realRollingValueLoss: Double?
    /// Latest legal-mass snapshot the [STATS] logger computed. Cached
    /// here so the chart-sample heartbeat (which fires more often
    /// than the [STATS] tick) can render the legal-masked entropy
    /// trace without recomputing the snapshot itself.
    @State private var realLastLegalMassSnapshot: ChessTrainer.LegalMassSnapshot?
    nonisolated static let replayBufferCapacity = 1_000_000
    /// Default number of active self-play workers when a new
    /// Play and Train session starts. The Stepper and
    /// `trainingParams.selfPlayWorkers` (formerly `@State`) defaults to this
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
    // selfPlayWorkerCount migrated to `trainingParams.selfPlayWorkers`.
    /// Upper bound on the adjustable training-step delay. 500 ms
    /// already turns a ~60 steps/s training worker into roughly
    /// 2 steps/s, which is as slow as anyone reasonably wants to
    /// crawl the learning rate while still making progress.
    nonisolated static let stepDelayMaxMs: Int = 3000
    /// Upper bound on the self-play-side per-game delay the replay-
    /// ratio auto-adjuster may impose. This is the reverse lever
    /// that kicks in when GPU training overhead alone exceeds the
    /// target cycle and training can't be slowed down any further
    /// (its delay is already 0). Per-slot, per-game — with N
    /// workers, a 500 ms rung removes roughly N × 2 games/sec of
    /// aggregate production, which is usually more than enough to
    /// bring the ratio back. 2000 ms is the ceiling so a runaway
    /// auto-adjust can't stall the session outright.
    nonisolated static let selfPlayDelayMaxMs: Int = 3000
    /// Discrete ladder of valid training-step delay values in
    /// milliseconds. Fine-grained 5 ms increments at the low end
    /// where small delays matter most, then 25 ms increments all
    /// the way up to `stepDelayMaxMs`. The Stepper's +/- clicks
    /// walk this ladder one rung at a time via
    /// `trainingStepDelayBinding`.
    nonisolated static let stepDelayLadder: [Int] =
    [0, 5, 10, 15, 20] + Array(stride(from: 25, through: Self.stepDelayMaxMs, by: 25))

    /// Snap a Stepper write onto `stepDelayLadder`. The Stepper is
    /// configured with `step: 1` over the full integer range, but
    /// only the discrete ladder rungs are valid; this helper translates
    /// the raw `current ± 1` write into "advance / retreat one rung".
    /// If `current` isn't already a ladder rung (e.g. inherited from
    /// the auto-adjuster which produces arbitrary ms values), it's
    /// snapped to the nearest rung first and then walked from there.
    /// Shared by `trainingStepDelayBinding` and
    /// `selfPlayStepDelayBinding` since both speak the same ladder.
    nonisolated static func snappedNextDelayRung(current: Int, requested: Int) -> Int {
        let ladder = stepDelayLadder
        let currentIdx: Int
        if let exact = ladder.firstIndex(of: current) {
            currentIdx = exact
        } else {
            currentIdx = ladder.enumerated().min(by: {
                abs($0.element - current) < abs($1.element - current)
            })?.offset ?? 0
        }
        let nextIdx: Int
        if requested > current {
            nextIdx = min(currentIdx + 1, ladder.count - 1)
        } else if requested < current {
            nextIdx = max(currentIdx - 1, 0)
        } else {
            nextIdx = currentIdx
        }
        return ladder[nextIdx]
    }
    // trainingStepDelayMs migrated to `trainingParams.trainingStepDelayMs`;
    // the training worker reads the live delay from
    // `replayRatioController.recordTrainingBatchAndGetDelay(...)` each
    // step, so no separate lock-protected mirror is needed.
    /// Shared lock-protected mirror of `trainingParams.selfPlayWorkers` that
    /// the self-play worker tasks read between games. The Stepper
    /// updates `trainingParams.selfPlayWorkers` AND this box atomically (via
    /// the binding side-effect); workers poll the box at the top
    /// of each iteration to decide whether to play another game
    /// or stay in their idle wait state. Allocated at session
    /// start, cleared on session end.
    @State private var workerCountBox: WorkerCountBox?
    /// Cached memory-stats line shown in the top busy row during
    /// Play and Train. Refreshed at most every
    /// `memoryStatsRefreshSec` seconds via
    /// `refreshMemoryStatsIfNeeded()` (called from the snapshot
    /// timer) so the displayed numbers don't churn at 10 Hz.
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
    /// Shared chart-layer state and decimation pipeline. Owned by
    /// `ContentView` (the composer) and forwarded to both
    /// `UpperContentView` (heartbeat append path) and
    /// `LowerContentView` (chart-grid render path). All
    /// chart-related `@State` that used to live here — the rings,
    /// decimated frame, scroll position, hover position, zoom
    /// state, arena events, diversity bars — moved onto the
    /// coordinator. See `ChartCoordinator.swift`.
    let chartCoordinator: ChartCoordinator
    @State private var showingInfoPopover: Bool = false
    /// Drives the top-bar Arena countdown chip's popover. Toggled by
    /// the chip's tap action; the popover anchors below the chip on
    /// `.top` arrow edge so it reads as a menu-style overlay.
    @State private var showArenaPopover: Bool = false
    /// Edit-text mirrors used by the Arena popover form. Seeded from
    /// `trainingParams` when the popover opens (via the form's
    /// `.onAppear`) and validated on Save before being written back to
    /// `trainingParams`. Kept here on the parent view rather than
    /// inside the popover view so a parse error doesn't cause SwiftUI
    /// to throw away in-progress text on a re-render.
    @State private var arenaPopoverGamesText: String = ""
    @State private var arenaPopoverConcurrencyText: String = ""
    @State private var arenaPopoverIntervalText: String = ""
    @State private var arenaPopoverGamesError: Bool = false
    @State private var arenaPopoverConcurrencyError: Bool = false
    @State private var arenaPopoverIntervalError: Bool = false
    @State private var arenaPopoverTauStartError: Bool = false
    @State private var arenaPopoverTauDecayError: Bool = false
    @State private var arenaPopoverTauFloorError: Bool = false
    /// Lifetime training step count at the moment the current
    /// Play-and-Train segment started (the same instant
    /// `parallelWorkerStatsBox.sessionStart` is captured). On a
    /// resumed session the trainer's seeded `_stats.steps` carries
    /// the cumulative pre-resume count, so the Run Totals rate
    /// numerator must subtract this baseline to express
    /// this-segment-only steps over this-segment-only wall time.
    /// `Avg total` divisor uses the same baseline so it's also
    /// per-segment and not under-reported by lifetime steps.
    @State private var trainingStepsAtSegmentStart: Int = 0
    /// Drives the Arena History sheet. Set true when the user clicks
    /// "History" in the Arena popover; the popover dismisses itself
    /// before flipping this flag so the sheet doesn't anchor to a
    /// dying popover.
    @State private var showArenaHistorySheet: Bool = false
    /// Set true while a one-shot log-scan recovery pass is
    /// running. Disables the "Recover from logs" button so the
    /// user can't trigger overlapping scans, and shows a small
    /// spinner in the sheet header.
    @State private var arenaRecoveryInProgress: Bool = false
    // Sampling cadence (`chartCoordinator.progressRateLastFetch`,
    // `chartCoordinator.progressRateNextId`, `chartCoordinator.trainingChartNextId`,
    // `chartCoordinator.prevChartTotalGpuMs`), chart navigation (`chartCoordinator.scrollX`,
    // `chartCoordinator.followLatest`), and the shared cross-chart hover
    // position (`chartCoordinator.hoveredSec`) all live on
    // `chartCoordinator`. Reads / writes route through the
    // coordinator's `appendProgressRate(_:)`,
    // `appendTrainingChart(_:totalGpuMs:)`,
    // `handleScrollChange(_:)`, and `hoveredSec` properties.
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

    /// Kind of the currently-shown checkpoint status message.
    /// Drives both the icon (checkmark / error glyph) and the text
    /// color, and the auto-clear lifetime. Success messages linger
    /// noticeably longer than in-progress messages so the user
    /// gets a durable "this actually saved" confirmation rather
    /// than seeing the "Saving…" line silently vanish.
    @State private var checkpointStatusKind: CheckpointStatusKind = .progress

    /// Drives the Load Model file importer sheet.
    @State private var showingLoadModelImporter: Bool = false

    /// Drives the Load Session file importer sheet.
    @State private var showingLoadSessionImporter: Bool = false

    /// Drives the Load Parameters file importer sheet (File menu →
    /// Load Parameters…). Loads a parameters JSON file with the same
    /// shape as the CLI `--parameters` flag and applies every named
    /// field as an override on top of the currently-effective values.
    @State private var showingLoadParametersImporter: Bool = false

    /// Drives the Save Parameters file exporter sheet (File menu →
    /// Save Parameters…). Set to `true` after `parametersDocumentForExport`
    /// has been populated with a freshly-encoded snapshot of the
    /// current configuration.
    @State private var showingSaveParametersExporter: Bool = false

    /// Pre-encoded JSON document handed to the Save Parameters file
    /// exporter. Built on the main actor at the moment the user
    /// invokes the menu item, so the encoded values reflect the
    /// session's state at that instant rather than at file-save time
    /// (which can be seconds later if the user takes a while to pick
    /// a destination).
    @State private var parametersDocumentForExport: CliParametersDocument?

    /// Watchdog for in-flight checkpoint saves. Started when a save
    /// begins; if the save hasn't completed within 5 s, the watchdog
    /// promotes the status row to `.slowProgress` and emits a
    /// `[CHECKPOINT-WARN]` log line. Every save-completion path
    /// (success, error, timeout) cancels the task so a fast save's
    /// watchdog never runs its body.
    @State private var slowSaveWatchdogTask: Task<Void, Never>?

    /// Message text for the "can't do that right now" alert surfaced
    /// by in-function guards. Non-nil means show the alert. The same
    /// alert also covers the post-Stop three-way start dialog's
    /// course-correction messages (e.g., "champion was loaded since
    /// last training — trainer was not").
    @State private var menuActionError: String?

    /// Drives the post-Stop three-way confirmation dialog. Shown only
    /// when the user presses Play-and-Train after a prior training
    /// session was stopped in this launch AND there is no pending
    /// loaded session (a disk load has its own resume path).
    @State private var showStartTrainingDialog: Bool = false

    /// Distinguishes how Play-and-Train should treat in-memory state
    /// when the user presses Start. Consumed inside
    /// `startRealTraining(mode:)` to branch replay-buffer, counter,
    /// and trainer-weight handling.
    private enum TrainingStartMode {
        /// Default path. First launch, or resuming from a
        /// `pendingLoadedSession`. Existing behavior: fresh
        /// replay buffer + counters, trainer weights come from
        /// either the loaded session's `trainer.dcmmodel` or a
        /// fresh fork of champion weights.
        case freshOrFromLoadedSession
        /// User stopped training earlier this launch and wants to
        /// pick up exactly where they left off. Reuse the existing
        /// replay buffer, parallel stats box, training stats box,
        /// session ID, tournament history, chart samples, and
        /// active trainer weights. Open a new training segment.
        case continueAfterStop
        /// User stopped training and wants a fresh session on the
        /// existing trainer's weights. Fresh replay buffer +
        /// counters + session ID, but do NOT overwrite trainer
        /// weights with the champion's.
        case newSessionKeepTrainer
        /// User stopped training and wants a fresh session starting
        /// from the champion — copy champion weights into the
        /// trainer, mint a fresh trainer generation ID, fresh
        /// replay buffer + counters + session ID.
        case newSessionResetTrainerFromChampion
    }

    /// Whether an autosave is currently in flight, so repeated
    /// promotions or rapid manual saves don't overlap. Advisory only
    /// — the save path is idempotent except for the "already exists"
    /// check which throws cleanly.
    @State private var checkpointSaveInFlight: Bool = false

    /// Scheduler for the 4-hour periodic autosave. Created on
    /// Play-and-Train start and torn down on Stop; `nil` while no
    /// session is active. Polled on the main heartbeat (throttled
    /// to once per second — a 4-hour deadline doesn't need 10 Hz
    /// resolution). See `PeriodicSaveController` for the
    /// arena-deferral and save-reset invariants.
    @State private var periodicSaveController: PeriodicSaveController?

    /// Last wall-clock time the heartbeat polled
    /// `periodicSaveController.decide(now:)`. Throttles the poll to
    /// roughly 1 Hz regardless of the 10 Hz heartbeat cadence,
    /// which is more than sufficient resolution for a multi-hour
    /// deadline and keeps the hot path cheap.
    @State private var periodicSaveLastPollAt: Date?

    /// `true` while a periodic autosave's write is in flight, so
    /// the heartbeat doesn't schedule another one on the very next
    /// tick. Set at fire time, cleared on success or failure.
    /// Separate from `checkpointSaveInFlight` (which guards the
    /// user-visible menu buttons) because a periodic save must be
    /// able to run even while the menu items remain enabled.
    @State private var periodicSaveInFlight: Bool = false

    /// Interval between scheduled periodic saves while a
    /// Play-and-Train session is active. 4 hours per the
    /// product spec — long enough to keep disk churn low, short
    /// enough that a crash never forfeits more than half a
    /// working day of training.
    nonisolated static let periodicSaveIntervalSec: TimeInterval = 4 * 60 * 60

    // MARK: - Auto-resume sheet state
    //
    // Shown at app launch if a `LastSessionPointer` still names
    // an on-disk `.dcmsession`. Offers the user a 30-second
    // window to resume (after which the resume fires
    // automatically) or dismiss. On dismiss the File menu item
    // "Resume training from autosave" covers the same flow for
    // the rest of the launch.

    /// Drives the sheet presentation. Set to `true` once during
    /// the first `.onAppear` pass when a valid pointer is found,
    /// flipped back to `false` on either button press or the
    /// countdown firing the auto-resume action.
    @State private var autoResumeSheetShowing: Bool = false

    /// Pointer the sheet is offering to resume. Captured when the
    /// sheet is presented so the resume action uses the exact
    /// pointer the user saw (not whatever the UserDefaults key
    /// might hold if the next save raced in between).
    @State private var autoResumePointer: LastSessionPointer?

    /// Lightweight peek of the target session's `session.json`,
    /// loaded synchronously when the sheet is presented. Powers
    /// the rich body lines (started-when, training counters, build
    /// version mismatch). `nil` either when the sheet isn't up or
    /// when the peek failed — in the latter case the sheet falls
    /// back to a minimal pointer-only layout so a corrupt session
    /// still gets the resume prompt rendered.
    @State private var autoResumeSummary: SessionResumeSummary?

    /// Seconds remaining on the countdown. Ticks down once per
    /// second from `autoResumeCountdownStartSec` while the sheet
    /// is showing; displayed in the Resume button's label.
    @State private var autoResumeCountdownRemaining: Int = 0

    /// Handle to the countdown task. Cancelled when the user
    /// dismisses the sheet (either via the Resume button or Not
    /// Now) so the timer doesn't fire a load after the user has
    /// explicitly opted out.
    @State private var autoResumeCountdownTask: Task<Void, Never>?

    /// True while a resume load is in flight, so the File menu
    /// item stays disabled during the load and the Resume button
    /// cannot be pressed twice. Cleared after the load-and-start
    /// chain completes (or errors).
    @State private var autoResumeInFlight: Bool = false

    /// Initial value of the auto-resume countdown in seconds.
    /// 30 s per the product spec — long enough for the user to
    /// notice the dialog and read the session details, short
    /// enough that an unattended app resumes quickly.
    nonisolated static let autoResumeCountdownStartSec: Int = 30

    /// True if a last-saved-session pointer exists and still
    /// names an on-disk directory. Used by the File menu item
    /// and the resume-in-flight guard. Computed live (cheap FS
    /// stat) rather than mirrored into @State so it tracks
    /// external deletions without us having to poll.
    private var canResumeFromAutosave: Bool {
        guard !realTraining,
              !autoResumeInFlight,
              !autoResumeSheetShowing else {
            return false
        }
        guard let pointer = LastSessionPointer.read() else {
            return false
        }
        return pointer.directoryExists
    }

    /// Composite scalar version of the three auto-resume gating
    /// flags so a single `.onChange` handler on the view body can
    /// drive `syncMenuCommandHubState()` when any of them flips.
    /// Modelled on the existing `chartZoomStateVersion` pattern
    /// for the same reason — adding N separate `.onChange` modifiers
    /// near a body this large blows the type-checker's time budget.
    private var autoResumeStateVersion: Int {
        (autoResumeInFlight ? 1 : 0)
        | (autoResumeSheetShowing ? 2 : 0)
    }

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

    /// Outer integral compensator for the replay-ratio controller's
    /// per-tick overhead-subtraction bias. The inner controller's
    /// barrier-tick overhead estimate (`D × P / G`) is dimensionally
    /// internally consistent but observably mis-scaled relative to
    /// the actual barrier-wall inflation under the batched
    /// shared-evaluator architecture (workers serialize through one
    /// barrier rather than running fully in parallel, so the per-game
    /// sleep does not divide cleanly across the pool). Empirically the
    /// inner controller's equilibrium settles at a `cons/prod` ratio
    /// well below the user-configured `replayRatioTarget` (observed
    /// ~0.78 vs target 1.10 in the failing autotrain runs). Rather
    /// than altering the inner formula, we wrap the controller with a
    /// slow integral compensator: each heartbeat we observe the gap
    /// between the user's desired target (`trainingParams
    /// .replayRatioTarget`) and the controller's reported
    /// `currentRatio`, then nudge the controller's INTERNAL target
    /// (`controller.targetRatio`) in the direction that closes the
    /// gap. The user-facing parameter does not move; only the
    /// internal control set-point drifts. Reset to the user value on
    /// session start, on user edit of `replayRatioTarget`, and on
    /// auto-adjust toggle.
    ///
    /// This is the controller's effective set-point (`T_eff` in the
    /// derivation comments) — `nil` when no session is active so
    /// teardown produces a clean re-seed on the next start.
    @State private var effectiveReplayRatioTarget: Double?
    /// Wall-clock stamp of the previous compensator tick. Drives the
    /// dt for the integral update so the gain is expressed in
    /// `target-units per second` rather than `per-heartbeat-tick`,
    /// which keeps the compensator's behavior independent of any
    /// future change to the heartbeat cadence.
    @State private var lastReplayRatioCompensatorAt: Date?
    // The training-parameter properties formerly stored here as
    // @AppStorage / @State are now exposed by `TrainingParameters.shared`
    // and accessed via `trainingParams.<name>`. See TrainingParameters.swift
    // for the canonical definitions, defaults, and persistence.

    /// Last auto-computed step delay, persisted so the next session
    /// starts from where the auto-adjuster left off instead of
    /// falling back to the manual default. Note: this is auto-controller
    /// state, not a training parameter — it is intentionally NOT migrated
    /// to TrainingParameters.shared.
    @AppStorage("lastAutoComputedDelayMs") private var lastAutoComputedDelayMs: Int = 50

    /// Singleton container for all training parameters that were
    /// previously stored as @AppStorage / @State on this view.
    /// Reads (`trainingParams.<name>`) participate in the view's
    /// dependency tracking via `@Observable`. The body shadows this
    /// with `@Bindable var trainingParams = trainingParams` so
    /// `$trainingParams.<name>` projects a real `Binding` for
    /// Steppers/Toggles inside the body. Helper methods use
    /// the bare reference for reads/writes.
    private let trainingParams = TrainingParameters.shared

    /// 100 ms heartbeat that pulls the latest snapshot from `gameWatcher`
    /// into `gameSnapshot`. Standard SwiftUI Combine timer pattern — the
    /// publisher is created when the view struct is initialized and SwiftUI
    /// manages the subscription lifecycle via `.onReceive` below.
    private let snapshotTimer = Timer.publish(
        every: 0.500, on: .main, in: .common
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
        guard realTraining else { return false }
        // Game-run mode with N > 1 self-play workers is a placeholder
        // ("N concurrent games / Live board hidden"), and the
        // Game-run / Candidate-test picker is hidden in that case
        // anyway, so the user has no way to switch out. Treat the
        // multi-worker case as Candidate-test active regardless of
        // what the persisted mode setting happens to be — that's the
        // only mode that produces useful left-side output here.
        if trainingParams.selfPlayWorkers > 1 { return true }
        return playAndTrainBoardMode == .candidateTest
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
    private var menuHubSignature: MenuHubSignature {
        MenuHubSignature(
            isBuilding: isBuilding,
            continuousPlay: continuousPlay,
            continuousTraining: continuousTraining,
            sweepRunning: sweepRunning,
            realTraining: realTraining,
            isArenaRunning: isArenaRunning,
            checkpointSaveInFlight: checkpointSaveInFlight,
            isTrainingOnce: isTrainingOnce,
            isEvaluating: isEvaluating,
            gameIsPlaying: gameSnapshot.isPlaying,
            hasNetwork: network != nil,
            hasPendingLoadedSession: pendingLoadedSession != nil,
            autoResumeStateVersion: autoResumeStateVersion,
            arenaRecoveryInProgress: arenaRecoveryInProgress
        )
    }

    /// Scratch string for the learning rate text field. Seeded from
    /// the trainer's current LR when Play-and-Train starts; the user
    /// edits freely without the binding reformatting mid-keystroke.
    /// The value is parsed and applied only on Enter (via `.onSubmit`
    /// on the TextField). Invalid input reverts to the current LR.
    @State private var learningRateEditText: String = ""
    @State private var lrWarmupStepsEditText: String = ""
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
    @State private var replayBufferCapacityEditText: String = ""
    @State private var replayBufferMinPositionsBeforeTrainingEditText: String = ""
    /// Edit-text + transactional checkbox state backing the new
    /// `TrainingSettingsPopover`. The momentum and √batch fields
    /// have no inline UI today, so this is where their first edit-
    /// surfaces live. Error flags pair with each text field; they
    /// drive the red-overlay on parse failure during Save.
    @State private var momentumCoeffEditText: String = ""
    @State private var sqrtBatchScalingEditValue: Bool = true
    @State private var trainingPopoverLRError: Bool = false
    @State private var trainingPopoverWarmupError: Bool = false
    @State private var trainingPopoverMomentumError: Bool = false
    @State private var trainingPopoverEntropyError: Bool = false
    @State private var trainingPopoverGradClipError: Bool = false
    @State private var trainingPopoverWeightDecayError: Bool = false
    @State private var trainingPopoverPolicyKError: Bool = false
    @State private var trainingPopoverDrawPenaltyError: Bool = false
    /// Edit-text + transactional state for the new tabs (Self Play,
    /// Replay) on `TrainingSettingsPopover`. Same pattern as the
    /// existing optimizer fields above.
    @State private var trainingBatchSizeEditText: String = ""
    @State private var selfPlayWorkersEditText: String = ""
    @State private var replayRatioTargetEditText: String = ""
    @State private var replaySelfPlayDelayEditText: String = ""
    @State private var replayTrainingStepDelayEditText: String = ""
    @State private var replayRatioAutoAdjustEditValue: Bool = true
    /// Stash of pre-edit values for the four replay-ratio control
    /// fields. The Replay tab live-propagates changes to those
    /// fields to `trainingParams` immediately so the user can watch
    /// the live ratio respond — but if they Cancel, we restore
    /// these stash values without warning. Captured in
    /// `trainingPopoverSeedFromParams()` and consumed by
    /// `trainingPopoverCancel()`.
    @State private var originalReplayRatioTarget: Double = 1.0
    @State private var originalReplaySelfPlayDelayMs: Int = 0
    @State private var originalReplayTrainingStepDelayMs: Int = 0
    @State private var originalReplayRatioAutoAdjust: Bool = true
    @State private var trainingPopoverTrainingBatchSizeError: Bool = false
    @State private var trainingPopoverSelfPlayWorkersError: Bool = false
    @State private var trainingPopoverSelfPlayStartTauError: Bool = false
    @State private var trainingPopoverSelfPlayDecayPerPlyError: Bool = false
    @State private var trainingPopoverSelfPlayFloorTauError: Bool = false
    @State private var trainingPopoverReplayBufferCapacityError: Bool = false
    @State private var trainingPopoverReplayBufferMinPositionsError: Bool = false
    @State private var trainingPopoverReplayRatioTargetError: Bool = false
    @State private var trainingPopoverReplaySelfPlayDelayError: Bool = false
    @State private var trainingPopoverReplayTrainingStepDelayError: Bool = false
    @State private var showTrainingPopover: Bool = false

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
        if selectedOverlay < 0 { return "" }
        if selectedOverlay == 0 { return "Top Moves" }
        return "Channel \(selectedOverlay - 1): \(TensorChannelNames.names[selectedOverlay - 1])"
    }

    private var currentOverlay: ChessBoardView.Overlay {
        // Outside of forward-pass / candidate-test contexts (e.g. live
        // game-run) we have no inference data to overlay; render bare.
        if !showForwardPassUI { return .none }
        guard let result = inferenceResult else { return .none }
        // Plain-board mode (-1) and Top Moves (0) both draw the
        // top-moves arrows. The only difference is presentational:
        // -1 hides the "Top Moves" header label and the input tensor
        // strip below; 0 shows both. Channel modes (>=1) overlay a
        // single input plane instead of arrows.
        if selectedOverlay <= 0 {
            return .topMoves(result.topMoves)
        }
        let start = (selectedOverlay - 1) * 64
        return .channel(Array(result.inputTensor[start..<start + 64]))
    }

    var body: some View {
        // Body-local @Bindable shadow of the TrainingParameters singleton
        // so `$trainingParams.<name>` projects a real `Binding<T>` for
        // Steppers/Toggles inside the body.
        @Bindable var trainingParams = self.trainingParams
        return VStack(alignment: .leading, spacing: 8) {
            // Title bar
            HStack(spacing: 8) {
                Text(BuildInfo.summary)
                    .font(.callout)
                    .foregroundStyle(.secondary)
                Button(action: { showingInfoPopover.toggle() }) {
                    Image(systemName: "info.circle")
                        .font(.title3)
                }
                .buttonStyle(.plain)
                .popover(isPresented: $showingInfoPopover) {
                    AboutPopoverContent(network: network)
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
                TrainingAlarmBanner(
                    alarm: alarm,
                    isSilenced: trainingAlarmSilenced,
                    onSilence: { silenceTrainingAlarm() },
                    onDismiss: { dismissTrainingAlarm() }
                )
            }

            cumulativeStatusBar

            // Status row — only renders when there's actually something
            // to show (a non-realTraining busy state, an in-flight
            // tournament, or a checkpoint status message). Wrapped in a
            // Group so the `.fileImporter` / `.alert` /
            // `.confirmationDialog` modifiers below have a stable
            // anchor even when the inner HStack collapses; an empty
            // Group contributes no height or VStack spacing, freeing
            // up the vertical band that used to sit empty between the
            // top status bar and the board.
            let showsBusyContent: Bool = {
                if let _ = checkpointStatusMessage { return true }
                guard isBusy else { return false }
                if !realTraining { return true }
                return tournamentProgress != nil
            }()
            Group {
                if showsBusyContent {
                    HStack(spacing: 8) {
                        if isBusy {
                            if !realTraining {
                                ProgressView().controlSize(.small)
                                busyLabelView
                            } else if tournamentProgress != nil {
                                busyLabelView
                            }
                        }
                        if let msg = checkpointStatusMessage {
                            CheckpointStatusLineView(kind: checkpointStatusKind, message: msg)
                        }
                        Spacer(minLength: 0)
                    }
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
            .fileImporter(
                isPresented: $showingLoadParametersImporter,
                allowedContentTypes: [.json],
                allowsMultipleSelection: false,
                onCompletion: { result in
                    handleLoadParametersPickResult(result)
                }
            )
            .fileExporter(
                isPresented: $showingSaveParametersExporter,
                document: parametersDocumentForExport,
                contentType: .json,
                defaultFilename: "parameters",
                onCompletion: { result in
                    handleSaveParametersExportResult(result)
                }
            )
            .fileDialogDefaultDirectory(
                showingLoadModelImporter
                ? CheckpointPaths.modelsDir
                : CheckpointPaths.sessionsDir
            )
            .alert(
                "Can't do that right now",
                isPresented: Binding(
                    get: { menuActionError != nil },
                    set: { newVal in
                        if !newVal { menuActionError = nil }
                    }
                ),
                actions: {
                    Button("OK", role: .cancel) { menuActionError = nil }
                },
                message: {
                    Text(menuActionError ?? "")
                }
            )
            .confirmationDialog(
                "Start training",
                isPresented: $showStartTrainingDialog,
                titleVisibility: .visible,
                actions: {
                    Button("Continue training") {
                        startRealTraining(mode: .continueAfterStop)
                    }
                    Button("New session with current trainer") {
                        startRealTraining(mode: .newSessionKeepTrainer)
                    }
                    Button("New session — trainer reset from champion") {
                        startRealTraining(mode: .newSessionResetTrainerFromChampion)
                    }
                    Button("Cancel", role: .cancel) { }
                },
                message: {
                    Text(startTrainingDialogMessage())
                }
            )

            // Board + text side by side
            HStack(alignment: .top, spacing: 24) {
                BoardSideView(
                    playAndTrainBoardMode: $playAndTrainBoardMode,
                    sideToMoveBinding: sideToMoveBinding,
                    probeNetworkTarget: $probeNetworkTarget,
                    realTraining: realTraining,
                    workerCount: trainingParams.selfPlayWorkers,
                    inferenceResultPresent: inferenceResult != nil,
                    showForwardPassUI: showForwardPassUI,
                    forwardPassEditable: forwardPassEditable,
                    isCandidateTestActive: isCandidateTestActive,
                    overlayLabel: overlayLabel,
                    board: LiveBoardWithNavigationView(
                        pieces: displayedPieces,
                        overlay: currentOverlay,
                        selectedOverlay: selectedOverlay,
                        inferenceResultPresent: inferenceResult != nil,
                        forwardPassEditable: forwardPassEditable,
                        realTraining: realTraining,
                        isCandidateTestActive: isCandidateTestActive,
                        workerCount: trainingParams.selfPlayWorkers,
                        onNavigate: { navigateOverlay($0) },
                        onApplyFreePlacementDrag: { from, to in
                            applyFreePlacementDrag(from: from, to: to)
                        },
                        squareIndex: { point, size in
                            Self.squareIndex(at: point, boardSize: size)
                        }
                    )
                )

                MainTextPanel(
                    isGameMode: isGameMode,
                    isTrainingMode: isTrainingMode,
                    isCandidateTestActive: isCandidateTestActive,
                    inferenceResultText: inferenceResult?.textOutput,
                    trainingError: trainingError,
                    selfPlayColumn: { selfPlayStatsColumn },
                    trainingColumn: { trainingStatsColumn }
                )

                // Right-of-board policy-channel decomposition. Toggled
                // via View > Show Policy Channels Panel. Driven off
                // the same `inferenceResult` that already feeds the
                // on-board Top Moves overlay, so it auto-updates with
                // the Candidate Test re-eval loop as the trainer
                // learns. Hidden entirely when the toggle is off so
                // the existing two-column layout is unchanged for
                // users who don't opt in. Also gated on
                // `showForwardPassUI` so it never displays a stale
                // result against an unrelated board (Game Run / Game
                // Mode / pure training paths) — same condition the
                // on-board overlay uses for the channel-strip.
                if showPolicyChannelsPanel && showForwardPassUI {
                    PolicyChannelsPanel(
                        pieces: displayedPieces,
                        currentPlayer: editableState.currentPlayer,
                        policyLogits: inferenceResult?.rawInference?.policy
                    )
                }
            }
            .layoutPriority(1)

            // Input tensor channel strip — hidden in plain-board mode
            // (`selectedOverlay < 0`, the new -1 default). Right-paging
            // off the plain board reveals the strip; right-arrow moves
            // selection through it. Divider is gated on the same
            // condition so collapsing the strip also drops its top
            // separator.
            if let result = inferenceResult, showForwardPassUI, selectedOverlay >= 0 {
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

            // The chart layer (zoom-control row + chart grid) is no
            // longer rendered here — it lives in `LowerContentView`,
            // which `ContentView` mounts as a sibling of
            // `UpperContentView` when the chart coordinator's
            // `isActive` flag is true. UpperContentView mirrors
            // `realTraining` into `chartCoordinator.isActive` via
            // the `.onChange` modifier below so ContentView can
            // make that decision without observing the upper view's
            // private state.
        }
        .padding(16)
        .focusable()
        .focusEffectDisabled()
        .onKeyPress(.leftArrow) { navigateOverlay(-1); return .handled }
        .onKeyPress(.rightArrow) { navigateOverlay(1); return .handled }
        .background(WindowAccessor(window: $contentWindow, onAttached: handleWindowAttached))
        .onAppear { handleBodyOnAppear() }
        .sheet(isPresented: $autoResumeSheetShowing) {
            autoResumeSheetContentView()
        }
        .sheet(isPresented: $showArenaHistorySheet) {
            ArenaHistoryView(
                history: tournamentHistory,
                configuredGamesPerTournament: trainingParams.arenaGamesPerTournament,
                promoteThreshold: trainingParams.arenaPromoteThreshold,
                onClose: { showArenaHistorySheet = false }
            )
        }
        .onReceive(NotificationCenter.default.publisher(for: NSWindow.willCloseNotification)) { note in
            handleWindowWillClose(note: note)
        }
        .background(MenuHubSyncProbe(
            signature: menuHubSignature,
            onSignatureChanged: { syncMenuCommandHubState() }
        ))
        .background(ControlSideEffectsProbe(
            playAndTrainBoardMode: $playAndTrainBoardMode,
            probeNetworkTarget: $probeNetworkTarget,
            candidateProbeDirty: $candidateProbeDirty,
            selectedOverlay: $selectedOverlay,
            lrWarmupStepsEditText: $lrWarmupStepsEditText,
            effectiveReplayRatioTarget: $effectiveReplayRatioTarget,
            lastReplayRatioCompensatorAt: $lastReplayRatioCompensatorAt,
            trainingParams: trainingParams,
            workerCountBox: workerCountBox,
            trainer: trainer,
            replayRatioController: replayRatioController,
            snapDelayToLadder: { delay in
                Self.stepDelayLadder.min(by: { abs($0 - delay) < abs($1 - delay) }) ?? delay
            }
        ))
        .onReceive(snapshotTimer) { _ in
            // Defer every @State mutation driven by the heartbeat
            // to a fresh main-actor runloop tick. The timer publisher
            // fires on the main thread, and SwiftUI flags "update
            // multiple times per frame" warnings (and measurable
            // hangs) when onReceive synchronously pushes several
            // dozen state-change notifications inline. The Task wrap
            // coalesces the work into a single render pass. Detached
            // so the surrounding SwiftUI tick context isn't inherited
            // (priority, task locals); the awaited call hops onto
            // the main actor where the tick body actually runs.
            //
            // Capture timestamp at dispatch so the tick body can
            // measure how long the main actor took to begin executing
            // it. A growing gap between dispatch and execution means
            // the main actor is being starved by other work — the
            // primary mechanism behind UI stalls during long sessions.
            let dispatchedAt = CFAbsoluteTimeGetCurrent()
            Task.detached {
                await processSnapshotTimerTick(dispatchedAt: dispatchedAt)
            }
        }
        .onChange(of: realTraining) { _, newValue in
            // Mirror `realTraining` into the coordinator so
            // `ContentView` can decide whether to mount
            // `LowerContentView` as a sibling without observing any
            // of `UpperContentView`'s private @State.
            chartCoordinator.isActive = newValue
        }
    }

    /// Fires when the SwiftUI view tree is first materialized into
    /// AppKit (via `WindowAccessor`'s `onAttached`), regardless of
    /// whether the window has become key / front / visible. This is
    /// the right hook for headless-mode startup work that doesn't
    /// require a visible window: under `--train`, `exec`-launched
    /// processes don't get auto-foregrounded by macOS, so the
    /// formerly-used `.onAppear` hook can stall indefinitely
    /// waiting for a user click on the dock icon. Driving from
    /// `WindowAccessor` decouples headless-mode startup from
    /// window visibility.
    ///
    /// Wires the menu command hub and syncs its mirrored state
    /// here too so `runAutoTrainLaunchSequence`'s prerequisites
    /// (e.g. `buildNetwork`'s enable-check state, which depends on
    /// the hub being live) are met. `handleBodyOnAppear` re-runs
    /// these calls when the window eventually appears; both are
    /// idempotent.
    @MainActor
    private func handleWindowAttached() {
        wireMenuCommandHub()
        syncMenuCommandHubState()
        // `--train` headless launch sequence. Fired here rather
        // than from `.onAppear` so it doesn't wait for the window
        // to become visible. Manual launches without `--train`
        // never enter this branch — their resume-sheet UX still
        // runs from `.onAppear` because that's correctly gated on
        // the window being visible to the user.
        if autoTrainOnLaunch && !autoTrainFired {
            autoTrainFired = true
            runAutoTrainLaunchSequence()
        }
    }

    /// Initial setup that fires once on body's `.onAppear` (so it
    /// is gated on the main window becoming visible/key). Seeds
    /// every TextField mirror state from `trainingParams` so the
    /// inputs read the live values rather than staying empty until
    /// the user touches them, and presents the launch-time
    /// auto-resume sheet for non-headless launches.
    ///
    /// Re-runs `wireMenuCommandHub` / `syncMenuCommandHubState`
    /// (idempotent) in case `WindowAccessor`'s `onAttached` did
    /// not fire before this hook for any reason. The headless
    /// `--train` trigger lives in `handleWindowAttached`, not
    /// here, because `.onAppear` is window-visibility-gated and
    /// would stall a backgrounded `exec`-launched process.
    @MainActor
    private func handleBodyOnAppear() {
        wireMenuCommandHub()
        syncMenuCommandHubState()
        if learningRateEditText.isEmpty {
            learningRateEditText = String(format: "%.1e", trainingParams.learningRate)
        }
        if lrWarmupStepsEditText.isEmpty {
            lrWarmupStepsEditText = String(trainingParams.lrWarmupSteps)
        }
        if entropyRegularizationEditText.isEmpty {
            entropyRegularizationEditText = String(format: "%.2e", trainingParams.entropyBonus)
        }
        if drawPenaltyEditText.isEmpty {
            drawPenaltyEditText = String(format: "%.3f", trainingParams.drawPenalty)
        }
        if weightDecayEditText.isEmpty {
            weightDecayEditText = String(format: "%.2e", trainingParams.weightDecay)
        }
        if gradClipMaxNormEditText.isEmpty {
            gradClipMaxNormEditText = String(format: "%.2f", trainingParams.gradClipMaxNorm)
        }
        if policyScaleKEditText.isEmpty {
            policyScaleKEditText = String(format: "%.2f", trainingParams.policyScaleK)
        }
        if spStartTauEditText.isEmpty {
            spStartTauEditText = String(format: "%.2f", trainingParams.selfPlayStartTau)
        }
        if spFloorTauEditText.isEmpty {
            spFloorTauEditText = String(format: "%.2f", trainingParams.selfPlayTargetTau)
        }
        if spDecayPerPlyEditText.isEmpty {
            spDecayPerPlyEditText = String(format: "%.3f", trainingParams.selfPlayTauDecayPerPly)
        }
        if arStartTauEditText.isEmpty {
            arStartTauEditText = String(format: "%.2f", trainingParams.arenaStartTau)
        }
        if arFloorTauEditText.isEmpty {
            arFloorTauEditText = String(format: "%.2f", trainingParams.arenaTargetTau)
        }
        if arDecayPerPlyEditText.isEmpty {
            arDecayPerPlyEditText = String(format: "%.3f", trainingParams.arenaTauDecayPerPly)
        }
        if replayBufferCapacityEditText.isEmpty {
            replayBufferCapacityEditText = String(trainingParams.replayBufferCapacity)
        }
        if replayBufferMinPositionsBeforeTrainingEditText.isEmpty {
            replayBufferMinPositionsBeforeTrainingEditText = String(trainingParams.replayBufferMinPositionsBeforeTraining)
        }
        // Resume-sheet UX is correctly gated on the window being
        // visible — surfacing a sheet on a hidden window would do
        // nothing useful. Skipped under `--train` because the
        // headless launch path (`handleWindowAttached`) has
        // already kicked off training and the sheet would be
        // confusing on top of an active session.
        if !autoTrainOnLaunch {
            maybePresentAutoResumeSheet()
        }
    }

    private func processSnapshotTimerTick(dispatchedAt: CFAbsoluteTime) async {
        let mainActorEnqueueWaitMs = (CFAbsoluteTimeGetCurrent() - dispatchedAt) * 1000
        let start = CFAbsoluteTimeGetCurrent()
        await __processSnapshotTimerTick()
        let elapsedMs = (CFAbsoluteTimeGetCurrent() - start) * 1000

        // Throttled session-log emit so a slow tick (or growing main-
        // actor wait) is visible in the session log without spamming
        // it with one line per 100 ms heartbeat. Emits at most once
        // per `Self.snapshotTickLogIntervalSec` seconds and ALWAYS on
        // any tick whose tick body or main-actor enqueue wait exceeds
        // the alarm threshold — those are the events worth seeing
        // even if a recent throttled emit just landed.
        let now = Date()
        let shouldEmitPeriodic = snapshotTickLastLogAt == nil
            || now.timeIntervalSince(snapshotTickLastLogAt!) >= Self.snapshotTickLogIntervalSec
        let isAnomalous = elapsedMs >= Self.snapshotTickAlarmMs
            || mainActorEnqueueWaitMs >= Self.snapshotTickAlarmMs
        if shouldEmitPeriodic || isAnomalous {
            let prefix = isAnomalous && !shouldEmitPeriodic ? "[TICK-SLOW]" : "[TICK]"
            SessionLogger.shared.log(String(
                format: "%@ tickMs=%.2f mainActorEnqueueWaitMs=%.2f",
                prefix, elapsedMs, mainActorEnqueueWaitMs
            ))
            snapshotTickLastLogAt = now
        }
    }

    /// Throttling clock for the periodic `[TICK]` emit. Reset every
    /// time a periodic or anomalous emit fires.
    @State private var snapshotTickLastLogAt: Date? = nil

    /// Cadence for the periodic snapshot-tick log emit. 30 s gives a
    /// readable log file at steady state while the anomaly threshold
    /// (`snapshotTickAlarmMs`) makes sure any genuine stall surfaces
    /// promptly even between the periodic emits.
    nonisolated static let snapshotTickLogIntervalSec: TimeInterval = 30

    /// Threshold above which a snapshot-tick wall or its main-actor
    /// enqueue wait is logged immediately (out-of-band) rather than
    /// waiting for the next periodic emit. 50 ms is half the
    /// heartbeat period — anything past that is a meaningful blip.
    nonisolated static let snapshotTickAlarmMs: Double = 50
    /// Body of the 100 ms heartbeat. Pulls every cross-thread state
    /// box (game, sweep, training, arena, parallel-worker counters,
    /// replay-ratio controller, diversity tracker) into the matching
    /// `@State`, throttled internally where each consumer cares about
    /// avoiding redundant invalidations. Extracted out of the inline
    /// `.onReceive(snapshotTimer)` closure so `body`'s expression
    /// type-check stays cheap — the closure used to be ~140 lines and
    /// dragged the whole modifier chain past the
    /// `-warn-long-expression-type-checking` budget.
    private func __processSnapshotTimerTick() async {
        // Pull the latest game state into @State at most every 100ms.
        // Cheap (single locked struct copy) and bounds UI work even
        // when the game loop is doing hundreds of moves per second.
        //
        // `elap(_:)` is the per-stage trace probe that was used to
        // attribute a UI stall to a specific block of this tick.
        // Left as a stub — call sites stay so the next investigation
        // can re-enable it by uncommenting the body — but disabled by
        // default so the snapshot tick stays as cheap as possible
        // (Date() per call + an os_log emit at every site adds up
        // when we're at 10 Hz and chasing main-actor latency in this
        // very tick).
        func elap(_ s: String) {
            // logger.log(">> \(s): \(Date().timeIntervalSince(start))")
        }
        let start = Date()
        _ = start
        elap("start")
        gameSnapshot = await gameWatcher.asyncSnapshot()
        elap("after gameWatcher")
        // Same heartbeat pulls the sweep's worker-thread progress and
        // any newly-completed rows into @State so the table grows live.
        if sweepRunning, let box = sweepCancelBox {
            sweepProgress = await box.asyncLatestProgress()
            // Sample process resident memory and feed it into the
            // sweep's per-row peak. The trainer also samples at row
            // boundaries — we just contribute extra samples while a
            // row is in flight so we don't miss mid-step spikes.
            let phys = await ChessTrainer.asyncCurrentPhysFootprintBytes()
            await box.asyncRecordPeakSample(phys)
            let rows = await box.asyncCompletedRows()
            if rows.count != sweepResults.count {
                sweepResults = rows
            }
            elap("after sweepsom")
        }
        // Same heartbeat pulls live training stats out of the
        // lock-protected box the background training task is writing
        // into. Guarded on step count so a mid-run session that
        // hasn't advanced since the last tick doesn't trigger a
        // useless redraw — and an idle box (after Stop or before
        // first step) stays silent.
        if let box = trainingBox {
            let snap = await box.asyncSnapshot()
            if snap.stats.steps != (trainingStats?.steps ?? -1) {
                trainingStats = snap.stats
                lastTrainStep = snap.lastTiming
                realRollingPolicyLoss = snap.rollingPolicyLoss
                realRollingValueLoss = snap.rollingValueLoss
            }
            if let err = snap.error, trainingError == nil {
                trainingError = err
            }
            elap("after 3")
        }
        // Mirror the trainer's warmup state into @State so the status
        // chip and the LR Warm-up status cell read from a snapshot
        // rather than touching the trainer's executionQueue from inside
        // `body`. Only published while a trainer exists; a missing
        // trainer yields nil so the Idle chip path doesn't accidentally
        // surface stale warmup numbers from a prior session.
        if let trainer {
            let completedTrainSteps = await trainer.asyncCompletedTrainSteps()
            elap("after 3.1")
            // Pass the locally-snapshotted step count so the LR uses the
            // same observation rather than re-acquiring the SyncBox; the
            // count and LR in the published snapshot are then guaranteed
            // consistent (they were previously two independent reads with
            // a one-step disagreement window).
            let effectiveLR = await trainer.asyncEffectiveLearningRate(
                forBatchSize: trainingParams.trainingBatchSize,
                completedSteps: completedTrainSteps
            )
            elap("after 3.2")
            let next = TrainerWarmupSnapshot(
                completedSteps: completedTrainSteps,
                warmupSteps: trainer.lrWarmupSteps,
                effectiveLR: effectiveLR
            )
            elap("after 3.3")
            if next != trainerWarmupSnap { trainerWarmupSnap = next }
        } else if trainerWarmupSnap != nil {
            trainerWarmupSnap = nil
        }
        elap("after 4")
        // Arena progress mirror — cheap lock read, only updates the
        // @State when the game index has advanced (or transitioned
        // between non-running and running), so no redundant view
        // invalidations between tournament games.
        if let tBox = tournamentBox {
            let snap = await tBox.asyncSnapshot()
            if snap?.currentGame != tournamentProgress?.currentGame
                || (snap == nil) != (tournamentProgress == nil) {
                tournamentProgress = snap
            }
        }
        elap("after 5")
        // Parallel worker counters mirror — only updates @State when
        // totals have actually advanced so the body isn't re-evaluated
        // when nothing's changed. The sessionStart timestamp is
        // embedded in the snapshot so the Session panel and busy label
        // can compute wall-clock rates on every render. Dirty check
        // compares the fields that advance on self-play and training
        // events; if either has changed (or the rolling-window count
        // has shifted because an entry aged out), push a new snapshot.
        if let pBox = parallelWorkerStatsBox {
            let snap = await pBox.asyncSnapshot()
            let prev = parallelStats
            // `sessionStart` is included in the dirty check so the
            // one-time shift performed by `markWorkersStarted()` lands
            // in @State immediately, even if no game or training step
            // has recorded yet.
            let changed = snap.selfPlayGames != (prev?.selfPlayGames ?? -1)
            || snap.trainingSteps != (prev?.trainingSteps ?? -1)
            || snap.recentGames != (prev?.recentGames ?? -1)
            || snap.sessionStart != prev?.sessionStart
            if changed {
                parallelStats = snap
            }
        }
        elap("after 6")
        // Memory stats refresh. Throttled internally to
        // `memoryStatsRefreshSec` so this is a cheap timestamp compare
        // on most heartbeats.
        await refreshMemoryStatsIfNeeded()
        elap("after 7")
        // Process %CPU / %GPU refresh — separate (5 s) cadence from
        // memory stats (10 s) so the utilisation line updates twice as
        // often without dragging the heavier Metal property reads
        // along with it.
        await refreshUsagePercentsIfNeeded()
        elap("after 8")
        // Progress-rate chart sampler. 1 Hz during Play and Train;
        // each sample carries the moves/hr averaged over the last 3
        // minutes of work. No-op outside of realTraining.
        await refreshProgressRateIfNeeded()
        elap("after 9")
        // Replay-ratio snapshot for the UI. Persist the auto-computed
        // delay so the next session starts from where the adjuster
        // left off.
        if let rc = replayRatioController {
            let snap = await rc.asyncSnapshot()
            replayRatioSnapshot = snap
            if snap.autoAdjust {
                lastAutoComputedDelayMs = snap.computedDelayMs
            }
            // Outer integral compensator. See the doc comment on
            // `effectiveReplayRatioTarget` for full rationale; in
            // short, the inner controller's per-tick overhead
            // subtraction is mis-scaled for the batched-evaluator
            // architecture and equilibrates at a `cons/prod` ratio
            // below the user's requested target. We close the loop
            // here without modifying the controller: every heartbeat,
            // observe the gap between the user's target and the
            // reported `currentRatio`, then adjust the controller's
            // internal `targetRatio` in the direction that moves
            // observed-ratio toward user-target. Slow gain (no faster
            // than the inner controller's SMA cadence — see
            // `gainPerSecond` and `ReplayRatioController.historyWindowSec`)
            // so the two loops don't fight; bounded so a long-tail
            // noise spike can't drift the controller into a
            // degenerate set-point.
            updateReplayRatioCompensator(snap: snap)
        }
        elap("after 10")
        // Diversity-histogram mirror. Read once per heartbeat off the
        // tracker's thread-safe snapshot. Only push into @State when
        // the bucket totals actually change (or the bar array is
        // currently empty) so SwiftUI doesn't invalidate the chart
        // every tick for a stable reading.
        if let tracker = selfPlayDiversityTracker {
            let divSnap = await tracker.asyncSnapshot()
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
            let changed = newBars.count != chartCoordinator.currentDiversityHistogramBars.count
            || zip(newBars, chartCoordinator.currentDiversityHistogramBars)
                .contains { $0.0.count != $0.1.count }
            if changed {
                chartCoordinator.setDiversityHistogramBars(newBars)
            }
        }
        elap("after 11")
        refreshChartZoomTick()
        elap("after 12")
        periodicSaveTick()
        elap("after LAST")
    }

    /// Per-heartbeat tick that asks the periodic-save scheduler
    /// whether to fire. Throttled to ~1 Hz — a 4-hour deadline does
    /// not benefit from 10 Hz polling and the throttle keeps the
    /// heartbeat hot path from paying a decision cost per frame. A
    /// no-op when no session is active, and an immediate no-op
    /// when a periodic save is already in flight (the fire flag is
    /// cleared when the write task resolves).
    @MainActor
    private func periodicSaveTick() {
        guard let controller = periodicSaveController else {
            periodicSaveLastPollAt = nil
            return
        }
        let now = Date()
        if let lastPoll = periodicSaveLastPollAt,
           now.timeIntervalSince(lastPoll) < 1.0 {
            return
        }
        periodicSaveLastPollAt = now
        if periodicSaveInFlight {
            return
        }
        switch controller.decide(now: now) {
        case .idle:
            return
        case .fire:
            SessionLogger.shared.log(
                "[CHECKPOINT] Periodic save tick — firing (interval=\(Int(Self.periodicSaveIntervalSec))s)"
            )
            handleSaveSessionPeriodic()
        }
    }

    /// Drive the chart-zoom state machine once per heartbeat. The
    /// auto-snap, manual-clamp, and re-engage logic lives on the
    /// coordinator (`refreshZoomTick()`). This wrapper resyncs the
    /// menu command hub when the coordinator reports a state change
    /// so the menu's enabled / disabled flags stay in step.
    private func refreshChartZoomTick() {
        if chartCoordinator.refreshZoomTick() {
            syncMenuCommandHubState()
        }
    }

    // ⌘= / ⌘- / Auto-button actions. Thin wrappers that route to
    // the coordinator's zoom helpers so the keyboard shortcut, the
    // menu item, and `LowerContentView`'s inline controls all share
    // the same code. Each wrapper also calls `syncMenuCommandHubState`
    // so the menu's enabled / disabled flags update immediately
    // (the menu hub doesn't observe the coordinator directly).

    @MainActor
    private func chartZoomIn() {
        chartCoordinator.zoomIn()
        syncMenuCommandHubState()
    }

    @MainActor
    private func chartZoomOut() {
        chartCoordinator.zoomOut()
        syncMenuCommandHubState()
    }

    @MainActor
    private func chartZoomEnableAuto() {
        chartCoordinator.enableAutoZoom()
        syncMenuCommandHubState()
    }

    /// Can the user zoom in further? Used to gate the View menu
    /// item's disabled state and the Auto-row's ⌘= hint styling.
    var canZoomChartIn: Bool { chartCoordinator.canZoomIn }

    /// Can the user zoom out further given the current data span?
    var canZoomChartOut: Bool { chartCoordinator.canZoomOut }

    /// Sample app and GPU memory at most every
    /// `memoryStatsRefreshSec` seconds, caching the result in
    /// `memoryStatsSnap` for the busy label to read. Cheap on a
    /// no-op tick (a single timestamp diff) so it's fine to call
    /// from the 10 Hz heartbeat. The actual sampling reads
    /// `task_info` and a couple of `MTLDevice` properties via
    /// the trainer's existing helpers.
    private func refreshMemoryStatsIfNeeded() async {
        let now = Date()
        if now.timeIntervalSince(memoryStatsLastFetch) < Self.memoryStatsRefreshSec {
            return
        }
        let app = await ChessTrainer.asyncCurrentPhysFootprintBytes()
        let caps: DeviceMemoryCaps?
        if let trainer {
            caps = await trainer.asyncDeviceMemoryCaps()
        } else {
            caps = nil
        }
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
    private func refreshTrainingChartIfNeeded() async {
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
        let trainingSnap: TrainingLiveStatsBox.Snapshot?
        if let trainingBox {
            trainingSnap = await trainingBox.asyncSnapshot()
        } else {
            trainingSnap = nil
        }
        let ratioSnap = replayRatioSnapshot

        let appMemMB = memoryStatsSnap.map { Double($0.appFootprintBytes) / (1024 * 1024) }
        let gpuMemMB = memoryStatsSnap.map { Double($0.gpuAllocatedBytes) / (1024 * 1024) }
        // GPU busy %: fraction of the last 1-second sample interval
        // that the GPU was actively running training steps. Computed
        // from the delta of cumulative GPU ms in TrainingRunStats.
        let currentGpuMs = trainingSnap?.stats.totalGpuMs ?? 0
        let gpuDeltaMs = max(0, currentGpuMs - chartCoordinator.prevChartTotalGpuMs)
        let gpuBusy = gpuDeltaMs / 10.0 // delta ms / 1000ms * 100%
        // The coordinator's `appendTrainingChart(_:totalGpuMs:)` call
        // below updates `prevChartTotalGpuMs` to `currentGpuMs` for
        // the next tick — no need to write it here.
        // Power + thermal state read straight from ProcessInfo at
        // sample time. Both are cheap property reads, and both can
        // change between samples without any polling overhead on our
        // part (the OS tracks them). Captured here so the chart
        // tile can render a continuous step trace rather than
        // sampling on hover.
        let pi = ProcessInfo.processInfo
        let sample = TrainingChartSample(
            id: chartCoordinator.trainingChartNextId,
            elapsedSec: elapsed,
            rollingPolicyLoss: trainingSnap?.rollingPolicyLoss,
            rollingValueLoss: trainingSnap?.rollingValueLoss,
            rollingPolicyEntropy: trainingSnap?.rollingPolicyEntropy,
            rollingPolicyNonNegCount: trainingSnap?.rollingPolicyNonNegCount,
            rollingPolicyNonNegIllegalCount: trainingSnap?.rollingPolicyNonNegIllegalCount,
            rollingGradNorm: trainingSnap?.rollingGradGlobalNorm,
            rollingVelocityNorm: trainingSnap?.rollingVelocityNorm,
            rollingPolicyHeadWeightNorm: trainingSnap?.rollingPolicyHeadWeightNorm,
            replayRatio: ratioSnap?.currentRatio,
            rollingPolicyLossWin: trainingSnap?.rollingPolicyLossWin,
            rollingPolicyLossLoss: trainingSnap?.rollingPolicyLossLoss,
            rollingLegalEntropy: realLastLegalMassSnapshot.map { Double($0.legalEntropy) },
            rollingLegalMass: realLastLegalMassSnapshot.map { Double($0.legalMass) },
            cpuPercent: cpuPercent,
            gpuBusyPercent: trainingSnap != nil ? gpuBusy : nil,
            gpuMemoryMB: gpuMemMB,
            appMemoryMB: appMemMB,
            lowPowerMode: pi.isLowPowerModeEnabled,
            thermalState: pi.thermalState
        )
        // Hand the freshly-built sample to the coordinator. It
        // owns the ring append, the id-counter bump, the GPU-ms
        // baseline update, and the decimated-frame recompute — see
        // `ChartCoordinator.appendTrainingChart(_:totalGpuMs:)`.
        // Pre-evaluation of the training alarm stays here on
        // `UpperContentView` because it mutates streak counters and
        // surfaces the banner / sound-loop state, both of which
        // are upper-layer concerns.
        chartCoordinator.appendTrainingChart(sample, totalGpuMs: currentGpuMs)
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
                title: Self.divergenceCriticalAlarmTitle,
                detail: alarmDetail(entropy: entropy, gradNorm: gradNorm)
            )
        } else if divergenceWarningStreak >= Self.divergenceAlarmConsecutiveWarningSamples {
            raiseTrainingAlarm(
                severity: .warning,
                title: Self.divergenceWarningAlarmTitle,
                detail: alarmDetail(entropy: entropy, gradNorm: gradNorm)
            )
        } else if divergenceRecoveryStreak >= Self.divergenceAlarmRecoverySamples {
            // Scope the auto-clear to alarms this evaluator actually
            // raised. Without this check, a healthy entropy / gNorm
            // reading would wipe OUT the banner from any other
            // detector (e.g. the legal-mass collapse detector that
            // runs on a separate 15 s cadence) — the user would see
            // unrelated alarms appear and disappear as the heartbeat
            // races the 15 s probe cycle. Titles are the de-facto
            // ownership marker because every raise in this file uses
            // a distinct title string.
            let activeTitle = activeTrainingAlarm?.title
            let isOurs = activeTitle == Self.divergenceCriticalAlarmTitle
                || activeTitle == Self.divergenceWarningAlarmTitle
            if isOurs {
                clearTrainingAlarm()
            }
        }
    }

    /// Alarm titles owned by `evaluateTrainingAlarm`. Anchored as
    /// named constants so the raise path and the ownership-scoped
    /// auto-clear check can't drift apart.
    nonisolated static let divergenceCriticalAlarmTitle = "Critical Training Divergence"
    nonisolated static let divergenceWarningAlarmTitle = "Training Divergence Warning"

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

    private func refreshProgressRateIfNeeded() async {
        guard realTraining else { return }
        let now = Date()
        if now.timeIntervalSince(chartCoordinator.progressRateLastFetch) < Self.progressRateRefreshSec {
            return
        }

        guard let session = parallelStats else { return }
        let elapsed = max(0, now.timeIntervalSince(session.sessionStart))
        let curSp = session.selfPlayPositions
        let curTr = (trainingStats?.steps ?? 0) * trainingParams.trainingBatchSize

        // Walk newest → oldest through the coordinator's ring,
        // recording the last sample we see that still falls inside
        // the 3-minute window. Breaks out as soon as we hit a
        // sample older than the cutoff — the ring is timestamp-
        // sorted, so anything older is also out of window. Bounded
        // at ~180 iterations per call in steady state regardless of
        // total session length.
        let cutoff = now.addingTimeInterval(-Self.progressRateWindowSec)
        var windowStart: ProgressRateSample?
        var i = chartCoordinator.progressRateRing.count - 1
        while i >= 0 {
            let sample = chartCoordinator.progressRateRing[i]
            if sample.timestamp >= cutoff {
                windowStart = sample
            } else {
                break
            }
            i -= 1
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
            id: chartCoordinator.progressRateNextId,
            timestamp: now,
            elapsedSec: elapsed,
            selfPlayCumulativeMoves: curSp,
            trainingCumulativeMoves: curTr,
            selfPlayMovesPerHour: spRate,
            trainingMovesPerHour: trRate
        )
        // Coordinator's `appendProgressRate(_:)` does the ring
        // append, the id-counter bump, the
        // `progressRateLastFetch` timestamp update, and the
        // auto-follow scroll-position adjustment in one shot.
        chartCoordinator.appendProgressRate(sample)
        // Append a training chart sample at the same 1Hz cadence.
        await refreshTrainingChartIfNeeded()
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
    private func refreshUsagePercentsIfNeeded() async {
        let now = Date()
        if now.timeIntervalSince(usageStatsLastFetch) < Self.usageStatsRefreshSec {
            return
        }
        usageStatsLastFetch = now
        guard let sample = await ChessTrainer.asyncSampleCurrentProcessUsage() else {
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
    /// `trainingParams.arenaPromoteThreshold` (the CLI-overridable mirror of
    /// `Self.tournamentPromoteThreshold`) so flipping it in one
    /// place re-colors the UI automatically.
    private var busyLabelView: Text {
        if let tp = tournamentProgress {
            let elapsed = Date().timeIntervalSince(tp.startTime)
            let scorePercent = tp.candidateScore * 100
            let thresholdPercent = trainingParams.arenaPromoteThreshold * 100
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

            let concurrencyTail: Text
            if tp.concurrency > 1 {
                concurrencyTail = Text("  (×\(tp.concurrency) concurrent)")
                    .foregroundStyle(Color.blue)
            } else {
                concurrencyTail = Text("")
            }

            let tail = Text("  " + Self.formatElapsed(elapsed))
                .foregroundStyle(Color.blue)

            return (head + score + concurrencyTail + tail).monospacedDigit()
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
            // The session time / GPU RAM / CPU / GPU block that used to
            // live here moved into the top-bar status chip + chart-grid
            // tiles (App memory, GPU, CPU) once those existed. Real
            // training intentionally returns empty here so the busy row
            // collapses to nothing during a session — the chip carries
            // the "what is happening?" information now.
            return ""
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
        // -1 = plain board (always available, even when no inference
        // result is available — it's just the live chess board with
        // no overlay). 0 = Top Moves; 1..inputPlanes = channel views,
        // both of which require an inferenceResult to render.
        let next = selectedOverlay + direction
        if next < -1 || next > ChessNetwork.inputPlanes { return }
        if next >= 0 && inferenceResult == nil { return }
        selectedOverlay = next
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
    ///
    /// `kind` drives the visual treatment (icon + color) and the
    /// auto-clear lifetime. Success messages linger 20 s — long
    /// enough for the user to glance up and confirm the save
    /// actually landed — versus 6 s for progress lines and 12 s
    /// for errors.
    private func setCheckpointStatus(_ message: String, kind: CheckpointStatusKind) {
        checkpointStatusMessage = message
        checkpointStatusKind = kind
        // Always echo errors to the session log so a transient on-screen
        // error message that auto-clears in 12 seconds is still
        // recoverable from the persistent log file. (Some callsites
        // also log their own more-detailed [CHECKPOINT] line — minor
        // duplication is fine; visibility is the priority.)
        if kind == .error {
            SessionLogger.shared.log("[CHECKPOINT-ERR] \(message)")
        }
        // Auto-clear after a kind-dependent lifetime. Grabs the
        // current message at schedule time so a later message isn't
        // wiped out by an earlier one's timer.
        let snapshotMessage = message
        let lifetimeSeconds: Int
        switch kind {
        case .progress: lifetimeSeconds = 6
        // Slow-save status persists noticeably longer than a normal
        // progress line — the user is presumably waiting on it, and a
        // 6-second auto-clear in the middle of a stuck save would just
        // leave them confused about whether anything is still happening.
        case .slowProgress: lifetimeSeconds = 120
        case .success: lifetimeSeconds = 20
        case .error: lifetimeSeconds = 12
        }
        Task { @MainActor in
            try? await Task.sleep(for: .seconds(lifetimeSeconds))
            if checkpointStatusMessage == snapshotMessage {
                checkpointStatusMessage = nil
                checkpointStatusKind = .progress
            }
        }
    }

    /// Slow-save watchdog deadline. If a save has not completed
    /// within this many seconds of starting, the status row flips to
    /// `.slowProgress` and a `[CHECKPOINT-WARN]` line is logged
    /// exactly once per save (no progressive warnings — completion
    /// will eventually flip the row to success/error and restore
    /// normal styling). Calibrated to the typical save cost: a
    /// healthy session save (two ~10 MB `.dcmmodel` files plus a 35
    /// MB replay buffer at 500k positions) takes well under a second
    /// on SSD; 10 s leaves headroom for the post-promotion path's
    /// `.utility`-priority detached task to be scheduled under load
    /// without firing false-positive warnings, while still surfacing
    /// genuinely stuck saves promptly.
    nonisolated static let slowSaveWatchdogSeconds: Int = 10

    /// Start a watchdog that warns the user if the save tagged
    /// `label` has not completed within `slowSaveWatchdogSeconds`.
    /// The returned task is stored in `slowSaveWatchdogTask`; every
    /// save path's completion branch must cancel it so a fast save's
    /// watchdog body never runs. Calling this while a previous
    /// watchdog is still pending cancels the previous one — only one
    /// save can be in flight at a time, and the most recent label is
    /// what should appear if it stalls.
    @MainActor
    private func startSlowSaveWatchdog(label: String) {
        slowSaveWatchdogTask?.cancel()
        let deadline = Self.slowSaveWatchdogSeconds
        slowSaveWatchdogTask = Task { @MainActor in
            do {
                try await Task.sleep(for: .seconds(deadline))
            } catch {
                // The save completed before the deadline — its
                // completion path called `cancelSlowSaveWatchdog()`,
                // which cancelled this Task. The `Task.sleep` throws
                // `CancellationError`. Exit silently; the fast-save
                // case is the common one.
                return
            }
            if Task.isCancelled { return }
            // If the save already finished and emitted a final
            // success/error status, don't clobber it. We only flip to
            // .slowProgress if the row still shows the original
            // "Saving…" line.
            guard checkpointSaveInFlight else { return }
            SessionLogger.shared.log(
                "[CHECKPOINT-WARN] \(label) still running after \(deadline)s — disk busy or replay buffer large?"
            )
            setCheckpointStatus(
                "Saving \(label)… (still running, \(deadline)s+)",
                kind: .slowProgress
            )
        }
    }

    /// Cancel the slow-save watchdog if any. Safe to call on any
    /// completion path — including success, error, and timeout
    /// branches that don't involve `slowSaveWatchdogTask` directly.
    @MainActor
    private func cancelSlowSaveWatchdog() {
        slowSaveWatchdogTask?.cancel()
        slowSaveWatchdogTask = nil
    }

    /// File menu > Load Parameters… handler. Decodes the picked JSON
    /// file as a `CliTrainingConfig` and applies every named field on
    /// top of the currently-effective configuration. Mirrors the
    /// launch-time `--parameters` flag's behavior exactly, so the
    /// `[APP] --parameters override: …` log lines emitted by
    /// `applyCliConfigOverrides` show up in the session log identically
    /// whether the file was loaded at launch or via this menu item.
    @MainActor
    private func handleLoadParametersPickResult(_ result: Result<[URL], Error>) {
        switch result {
        case .failure(let error):
            setCheckpointStatus(
                "Load Parameters cancelled: \(error.localizedDescription)",
                kind: .error
            )
        case .success(let urls):
            guard let url = urls.first else { return }
            let needsAccess = url.startAccessingSecurityScopedResource()
            defer {
                if needsAccess { url.stopAccessingSecurityScopedResource() }
            }
            do {
                let cfg = try CliTrainingConfig.load(from: url)
                SessionLogger.shared.log(
                    "[BUTTON] Load Parameters from \(url.lastPathComponent): \(cfg.summaryString())"
                )
                let changes = applyCliConfigOverridesFromMenu(cfg: cfg)
                // Surface both the count and the field labels in the
                // status row. `applyCliConfigOverrides` already logs
                // a per-field `[APP] --parameters override: …` line
                // for each entry plus a summary; this row is the
                // user-visible mirror of that summary so they don't
                // have to grep the session log to know what landed.
                if changes.isEmpty {
                    setCheckpointStatus(
                        "Loaded \(url.lastPathComponent): no parameters changed",
                        kind: .success
                    )
                } else {
                    let labels = changes.map(\.label).joined(separator: ", ")
                    setCheckpointStatus(
                        "Loaded \(url.lastPathComponent): \(changes.count) parameter\(changes.count == 1 ? "" : "s") changed (\(labels))",
                        kind: .success
                    )
                }
            } catch {
                setCheckpointStatus(
                    "Load Parameters failed: \(error.localizedDescription)",
                    kind: .error
                )
                SessionLogger.shared.log(
                    "[CHECKPOINT-ERR] Load Parameters from \(url.lastPathComponent) failed: \(error.localizedDescription)"
                )
            }
        }
    }

    /// File menu > Save Parameters… handler. Builds a fully-populated
    /// `CliTrainingConfig` from the current `@AppStorage` / `@State`
    /// values, encodes it to JSON, stashes the bytes in
    /// `parametersDocumentForExport`, and triggers the file exporter.
    /// The exporter UI handles destination selection; on completion,
    /// `handleSaveParametersExportResult` logs success/failure.
    @MainActor
    private func handleSaveParametersMenuAction() {
        do {
            let snap = trainingParams.snapshot().rawValueMap()
            var dict: [String: Any] = [:]
            for (id, raw) in snap {
                switch raw {
                case .bool(let x): dict[id] = x
                case .int(let x): dict[id] = x
                case .double(let x): dict[id] = x
                }
            }
            let data = try JSONSerialization.data(
                withJSONObject: dict,
                options: [.prettyPrinted, .sortedKeys]
            )
            parametersDocumentForExport = CliParametersDocument(data: data)
            showingSaveParametersExporter = true
            SessionLogger.shared.log("[BUTTON] Save Parameters")
        } catch {
            setCheckpointStatus(
                "Save Parameters failed (encode): \(error.localizedDescription)",
                kind: .error
            )
        }
    }

    /// Completion handler for the Save Parameters file exporter.
    /// Logs success or failure to the session log; user-visible
    /// status appears in the checkpoint status row.
    @MainActor
    private func handleSaveParametersExportResult(_ result: Result<URL, Error>) {
        parametersDocumentForExport = nil
        switch result {
        case .success(let url):
            setCheckpointStatus(
                "Saved parameters to \(url.lastPathComponent)",
                kind: .success
            )
            SessionLogger.shared.log(
                "[CHECKPOINT] Saved parameters: \(url.lastPathComponent)"
            )
        case .failure(let error):
            // SwiftUI's file exporter surfaces user-cancellation as
            // a failure with `.userCancelled` — don't treat that as
            // an error in the UI. Only real I/O failures get the
            // red status.
            if let cocoa = error as? CocoaError, cocoa.code == .userCancelled {
                return
            }
            setCheckpointStatus(
                "Save Parameters failed: \(error.localizedDescription)",
                kind: .error
            )
            SessionLogger.shared.log(
                "[CHECKPOINT-ERR] Save Parameters failed: \(error.localizedDescription)"
            )
        }
    }

    // currentParametersConfig() removed in the TrainingParameters rewrite —
    // handleSaveParametersMenuAction now serializes directly from
    // `trainingParams.snapshot().rawValueMap()`.

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
                promotionKind: record.promotionKind?.rawValue,
                candidateWinsAsWhite: record.candidateWinsAsWhite,
                candidateWinsAsBlack: record.candidateWinsAsBlack,
                candidateLossesAsWhite: record.candidateLossesAsWhite,
                candidateLossesAsBlack: record.candidateLossesAsBlack,
                candidateDrawsAsWhite: record.candidateDrawsAsWhite,
                candidateDrawsAsBlack: record.candidateDrawsAsBlack,
                finishedAtUnix: record.finishedAt.map { Int64($0.timeIntervalSince1970) },
                candidateID: record.candidateID?.description,
                championID: record.championID?.description
            )
        }
        let lr = trainer?.learningRate ?? Self.trainerLearningRateDefault
        let entropyCoeff = trainer?.entropyRegularizationCoeff ?? Self.entropyRegularizationCoeffDefault
        let drawPen = trainer?.drawPenalty ?? Float(trainingParams.drawPenalty)
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
            trainingPositionsSeen: (trainingSnap?.steps ?? 0) * trainingParams.trainingBatchSize,
            batchSize: trainingParams.trainingBatchSize,
            learningRate: lr,
            entropyRegularizationCoeff: entropyCoeff,
            drawPenalty: drawPen,
            promoteThreshold: trainingParams.arenaPromoteThreshold,
            arenaGames: trainingParams.arenaGamesPerTournament,
            arenaConcurrency: trainingParams.arenaConcurrency,
            selfPlayTau: TauConfigCodable(samplingScheduleBox?.selfPlay ?? buildSelfPlaySchedule()),
            arenaTau: TauConfigCodable(samplingScheduleBox?.arena ?? buildArenaSchedule()),
            selfPlayWorkerCount: trainingParams.selfPlayWorkers,
            gradClipMaxNorm: Float(trainingParams.gradClipMaxNorm),
            weightDecayCoeff: Float(trainingParams.weightDecay),
            policyScaleK: Float(trainingParams.policyScaleK),
            momentumCoeff: Float(trainingParams.momentumCoeff),
            replayRatioTarget: trainingParams.replayRatioTarget,
            replayRatioAutoAdjust: trainingParams.replayRatioAutoAdjust,
            stepDelayMs: trainingParams.trainingStepDelayMs,
            lastAutoComputedDelayMs: lastAutoComputedDelayMs,
            // Schema-expansion fields (added to close the autotrain
            // reproducibility gap — these previously lived only in
            // @AppStorage / @State and so silently picked up the
            // user's current global preference on resume rather than
            // the session's saved value). All Optional in the schema
            // for back-compat with older session.json files.
            lrWarmupSteps: trainingParams.lrWarmupSteps,
            sqrtBatchScalingForLR: trainingParams.sqrtBatchScalingLR,
            replayBufferMinPositionsBeforeTraining: trainingParams.replayBufferMinPositionsBeforeTraining,
            arenaAutoIntervalSec: trainingParams.arenaAutoIntervalSec,
            candidateProbeIntervalSec: trainingParams.candidateProbeIntervalSec,
            legalMassCollapseThreshold: trainingParams.legalMassCollapseThreshold,
            legalMassCollapseGraceSeconds: trainingParams.legalMassCollapseGraceSeconds,
            legalMassCollapseNoImprovementProbes: trainingParams.legalMassCollapseNoImprovementProbes,
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
        // Belt-and-suspenders guards — menu disable is the primary
        // gate but these cover keyboard-shortcut / URL-scheme
        // invocations under a race.
        if checkpointSaveInFlight {
            refuseMenuAction("A save is already in progress. Wait for it to finish.")
            return
        }
        if isArenaRunning {
            refuseMenuAction("Can't save the champion while the arena is running. Wait for it to finish.")
            return
        }
        if isBusy && !realTraining {
            refuseMenuAction("Another operation is in progress. Wait for it to finish, then try again.")
            return
        }
        guard let champion = network else {
            refuseMenuAction("Build or load a model first.")
            return
        }
        let championID = champion.identifier?.description ?? "unknown"
        // Snapshot the active self-play gate up front. If there
        // is no active session, we can safely export directly —
        // nobody is racing against us.
        let gate = activeSelfPlayGate
        checkpointSaveInFlight = true
        setCheckpointStatus("Saving champion…", kind: .progress)
        startSlowSaveWatchdog(label: "champion save")

        Task {
            // Pause worker 0 if a session is running. Bail with a
            // user-visible error on timeout (indicates the session
            // has already ended or the worker is stuck — either way
            // we shouldn't spin forever).
            if let gate {
                let acquired = await gate.pauseAndWait(timeoutMs: Self.saveGateTimeoutMs)
                if !acquired {
                    cancelSlowSaveWatchdog()
                    checkpointSaveInFlight = false
                    setCheckpointStatus("Save aborted: could not pause self-play (timeout)", kind: .error)
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
                cancelSlowSaveWatchdog()
                checkpointSaveInFlight = false
                setCheckpointStatus("Save failed (export): \(exportError.localizedDescription)", kind: .error)
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
            cancelSlowSaveWatchdog()
            checkpointSaveInFlight = false
            switch outcome {
            case .success(let url):
                setCheckpointStatus("Saved \(url.lastPathComponent)", kind: .success)
                SessionLogger.shared.log("[CHECKPOINT] Saved champion: \(url.lastPathComponent)")
            case .failure(let error):
                setCheckpointStatus("Save failed: \(error.localizedDescription)", kind: .error)
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
        // Belt-and-suspenders guards — menu disable is the primary
        // gate but these cover keyboard-shortcut / URL-scheme
        // invocations under a race.
        if checkpointSaveInFlight {
            refuseMenuAction("A save is already in progress. Wait for it to finish.")
            return
        }
        if isArenaRunning {
            refuseMenuAction("Can't save the session while the arena is running. Wait for it to finish.")
            return
        }
        guard realTraining,
              let champion = network,
              let trainer,
              let selfPlayGate = activeSelfPlayGate,
              let trainingGate = activeTrainingGate else {
            refuseMenuAction("No active training session to save. Start Play and Train first.")
            return
        }
        saveSessionInternal(
            champion: champion,
            trainer: trainer,
            selfPlayGate: selfPlayGate,
            trainingGate: trainingGate,
            trigger: .manual
        )
    }

    /// Fired by `PeriodicSaveController` when its 4-hour deadline
    /// elapses (after any arena-deferral has resolved). Behaves
    /// exactly like the manual save path but tagged `.periodic` so
    /// the filename, status-line, and log line distinguish the two
    /// triggers. The controller has already decided we should fire;
    /// any remaining guard failures here (no session, save already
    /// in flight) just make the periodic attempt a no-op — the next
    /// tick of the controller will re-fire since `noteSuccessfulSave`
    /// is never called.
    @MainActor
    private func handleSaveSessionPeriodic() {
        // Guard against an arena starting in the tiny race window
        // between the controller's decide() and this call.
        if isArenaRunning {
            return
        }
        if checkpointSaveInFlight {
            SessionLogger.shared.log("[CHECKPOINT] Periodic save skipped — another save is in flight")
            return
        }
        guard realTraining,
              let champion = network,
              let trainer,
              let selfPlayGate = activeSelfPlayGate,
              let trainingGate = activeTrainingGate else {
            // Should not happen if the controller is armed correctly,
            // but disarm ourselves and bail to be safe.
            periodicSaveController?.disarm()
            return
        }
        periodicSaveInFlight = true
        saveSessionInternal(
            champion: champion,
            trainer: trainer,
            selfPlayGate: selfPlayGate,
            trainingGate: trainingGate,
            trigger: .periodic
        )
    }

    /// Shared save-session internal used by both the manual save
    /// button and the periodic autosave. Handles the gate dance,
    /// exports both networks, builds the session state on the main
    /// actor, and fires off the actual write to a detached task.
    /// The post-promotion autosave uses its own inline code path
    /// (in the arena coordinator) because it re-uses weights already
    /// snapshotted under the arena's own pause and so does not need
    /// to dance the gates again here.
    private func saveSessionInternal(
        champion: ChessMPSNetwork,
        trainer: ChessTrainer,
        selfPlayGate: WorkerPauseGate,
        trainingGate: WorkerPauseGate,
        trigger: SessionSaveTrigger
    ) {
        let championID = champion.identifier?.description ?? "unknown"
        let trainerID = trainer.identifier?.description ?? "unknown"
        let diskTag = trigger.diskTag
        let uiSuffix = trigger.uiSuffix
        checkpointSaveInFlight = true
        setCheckpointStatus("Saving session\(uiSuffix)…", kind: .progress)
        startSlowSaveWatchdog(label: "session save\(uiSuffix)")

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
            // Helper to clear both in-flight flags consistently on
            // every early-return path below. The periodic flag is
            // only meaningful when `trigger == .periodic`, but it's
            // cheap to always clear so we don't have to repeat the
            // branch on every error exit. Cancels the slow-save
            // watchdog too so a fast-failure path doesn't leave a
            // stale "Saving… (still running)" amber line behind.
            @MainActor func clearInFlight() {
                cancelSlowSaveWatchdog()
                checkpointSaveInFlight = false
                periodicSaveInFlight = false
            }

            // Pause self-play briefly so the champion export is
            // race-free, snapshot weights, then resume. Uses the
            // bounded variant so a session end mid-save doesn't
            // spin forever waiting for workers that have exited.
            let selfPlayAcquired = await selfPlayGate.pauseAndWait(timeoutMs: Self.saveGateTimeoutMs)
            guard selfPlayAcquired else {
                clearInFlight()
                setCheckpointStatus("Save aborted: could not pause self-play (timeout)", kind: .error)
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
                clearInFlight()
                setCheckpointStatus("Save failed (champion export): \(championError.localizedDescription)", kind: .error)
                SessionLogger.shared.log("[CHECKPOINT] Save session failed at champion export: \(championError.localizedDescription)")
                return
            }

            // Pause training briefly to snapshot trainer weights.
            let trainingAcquired = await trainingGate.pauseAndWait(timeoutMs: Self.saveGateTimeoutMs)
            guard trainingAcquired else {
                clearInFlight()
                setCheckpointStatus("Save aborted: could not pause training (timeout)", kind: .error)
                SessionLogger.shared.log("[CHECKPOINT] Save session aborted at training pause timeout")
                return
            }
            var trainerWeights: [[Float]] = []
            var trainerError: Error?
            do {
                // exportTrainerWeights bundles trainables + bn +
                // momentum velocity. Caller is responsible for
                // pausing both gates, which we did above.
                trainerWeights = try await Task.detached(priority: .userInitiated) {
                    try await trainer.exportTrainerWeights()
                }.value
            } catch {
                trainerError = error
            }
            trainingGate.resume()

            if let trainerError {
                clearInFlight()
                setCheckpointStatus("Save failed (trainer export): \(trainerError.localizedDescription)", kind: .error)
                SessionLogger.shared.log("[CHECKPOINT] Save session failed at trainer export: \(trainerError.localizedDescription)")
                return
            }

            // Final write + verify on a detached task so UI stays
            // responsive during the ~150 ms scratch-network build.
            let championMetadata = ModelCheckpointMetadata(
                creator: diskTag,
                trainingStep: trainingStep,
                parentModelID: "",
                notes: "Session checkpoint (\(diskTag))"
            )
            let trainerMetadata = ModelCheckpointMetadata(
                creator: diskTag,
                trainingStep: trainingStep,
                parentModelID: championID,
                notes: "Trainer lineage at session checkpoint (\(diskTag))"
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
                        trigger: diskTag
                    )
                    return .success(url)
                } catch {
                    return .failure(error)
                }
            }.value

            clearInFlight()
            switch outcome {
            case .success(let url):
                setCheckpointStatus("Saved \(url.lastPathComponent)\(uiSuffix)", kind: .success)
                let bufStr: String
                if let snap = bufferForSave?.stateSnapshot() {
                    bufStr = " replay=\(snap.storedCount)/\(snap.capacity)"
                } else {
                    bufStr = ""
                }
                SessionLogger.shared.log(
                    "[CHECKPOINT] Saved session (\(diskTag)): \(url.lastPathComponent) build=\(BuildInfo.buildNumber) git=\(BuildInfo.gitHash)\(bufStr)"
                )
                recordLastSessionPointer(
                    directoryURL: url,
                    sessionID: sessionState.sessionID,
                    trigger: diskTag
                )
                periodicSaveController?.noteSuccessfulSave(at: Date())
            case .failure(let error):
                setCheckpointStatus("Save failed: \(error.localizedDescription)", kind: .error)
                SessionLogger.shared.log("[CHECKPOINT] Save session (\(diskTag)) failed: \(error.localizedDescription)")
            }
        }
    }

    /// Persist a pointer to the session directory that was just
    /// saved so the next app launch can offer to auto-resume it.
    /// Called from every successful session-save path (manual,
    /// post-promotion, periodic) so the pointer always names the
    /// freshest on-disk session regardless of which trigger wrote
    /// it.
    @MainActor
    private func recordLastSessionPointer(
        directoryURL: URL,
        sessionID: String,
        trigger: String
    ) {
        let pointer = LastSessionPointer(
            sessionID: sessionID,
            directoryPath: directoryURL.path,
            savedAtUnix: Int64(Date().timeIntervalSince1970),
            trigger: trigger
        )
        pointer.write()
        SessionLogger.shared.log(
            "[CHECKPOINT] resume-pointer set → \(directoryURL.lastPathComponent) (\(trigger))"
        )
    }

    /// Load a standalone `.dcmmodel` into the current champion
    /// network. Triggered from the Load Model file importer. The
    /// network must exist (loading into a built network preserves
    /// the existing graph compilation; we don't rebuild).
    private func handleLoadModelPickResult(_ result: Result<[URL], Error>) {
        switch result {
        case .failure(let error):
            setCheckpointStatus("Load cancelled: \(error.localizedDescription)", kind: .error)
        case .success(let urls):
            guard let url = urls.first else { return }
            loadModelFrom(url: url)
        }
    }

    private func loadModelFrom(url: URL) {
        // In-function guards (belt-and-suspenders with menu disable).
        if isBuildingOrBusy() {
            refuseMenuAction(busyReasonMessage())
            return
        }

        checkpointSaveInFlight = true
        setCheckpointStatus("Loading \(url.lastPathComponent)…", kind: .progress)

        Task {
            // Auto-build the champion shell if it doesn't exist yet.
            // The weights are about to be overwritten, so the random
            // init is only satisfying graph compilation — no reason
            // to require the user to press Build first.
            let championResult = await ensureChampionBuilt()
            switch championResult {
            case .failure(let error):
                checkpointSaveInFlight = false
                setCheckpointStatus("Build failed: \(error.localizedDescription)", kind: .error)
                SessionLogger.shared.log("[CHECKPOINT] Load model auto-build failed: \(error.localizedDescription)")
                return
            case .success(let champion):
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
                    setCheckpointStatus("Loaded \(file.modelID)", kind: .success)
                    SessionLogger.shared.log("[CHECKPOINT] Loaded model: \(url.lastPathComponent) → \(file.modelID)")
                    inferenceResult = nil
                    // Flag champion-replaced for the post-Stop Start
                    // dialog's "Continue" annotation. Cleared as
                    // soon as a new training segment starts.
                    if replayBuffer != nil {
                        championLoadedSinceLastTrainingSegment = true
                    }
                case .failure(let error):
                    setCheckpointStatus("Load failed: \(error.localizedDescription)", kind: .error)
                    SessionLogger.shared.log("[CHECKPOINT] Load model failed: \(error.localizedDescription)")
                }
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
            setCheckpointStatus("Load cancelled: \(error.localizedDescription)", kind: .error)
        case .success(let urls):
            guard let url = urls.first else { return }
            loadSessionFrom(url: url)
        }
    }

    private func loadSessionFrom(url: URL, startAfterLoad: Bool = false) {
        // In-function guards (belt-and-suspenders with menu disable).
        if isBuildingOrBusy() {
            refuseMenuAction(busyReasonMessage())
            return
        }

        checkpointSaveInFlight = true
        setCheckpointStatus("Loading session \(url.lastPathComponent)…", kind: .progress)

        Task {
            // Auto-build the champion shell if it doesn't exist yet.
            // The weights are about to be overwritten, so the random
            // init is only satisfying graph compilation — no reason
            // to require the user to press Build first.
            let championResult = await ensureChampionBuilt()
            guard case .success(let champion) = championResult else {
                checkpointSaveInFlight = false
                if case .failure(let error) = championResult {
                    setCheckpointStatus("Build failed: \(error.localizedDescription)", kind: .error)
                    SessionLogger.shared.log("[CHECKPOINT] Load session auto-build failed: \(error.localizedDescription)")
                }
                return
            }
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
                setCheckpointStatus("Loaded session \(loaded.state.sessionID) — click Play and Train to resume", kind: .success)
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
                if startAfterLoad {
                    // Auto-resume path (from the launch-time sheet or
                    // the File menu "Resume training from autosave"
                    // command). Chain straight into Play-and-Train so
                    // the user's single click results in the session
                    // both loaded AND running.
                    SessionLogger.shared.log("[CHECKPOINT] Auto-resume: starting Play-and-Train on loaded session")
                    startRealTraining()
                }
            case .failure(let error):
                setCheckpointStatus("Load failed: \(error.localizedDescription)", kind: .error)
                SessionLogger.shared.log("[CHECKPOINT] Load session failed: \(error.localizedDescription)")
            }
            if startAfterLoad {
                autoResumeInFlight = false
            }
        }
    }

    // MARK: - Auto-resume flow

    /// Present the auto-resume sheet if a valid last-session
    /// pointer is on disk. Called from the view's first `.onAppear`
    /// pass. A no-op if no pointer exists, the target has been
    /// deleted externally, or the app is already mid-training
    /// (the latter should be impossible at launch, but the guard
    /// is cheap and keeps this callable from anywhere).
    @MainActor
    private func maybePresentAutoResumeSheet() {
        // Suppress auto-resume entirely when running under XCTest.
        // Same env-var signal `DrewsChessMachineApp` already uses to
        // bypass CLI-flag parsing — set unconditionally by the
        // xctest runner. Without this guard, a future test that
        // instantiates `ContentView` would either auto-fire a real
        // training run 30 s into the test or race the countdown
        // Task against test teardown.
        if ProcessInfo.processInfo.environment["XCTestConfigurationFilePath"] != nil {
            SessionLogger.shared.log("[RESUME] Skipping auto-resume sheet — running under XCTest")
            return
        }
        guard !realTraining, !autoResumeSheetShowing else { return }
        guard let pointer = LastSessionPointer.read() else {
            SessionLogger.shared.log("[RESUME] No last-session pointer found — skipping auto-resume prompt")
            return
        }
        guard pointer.directoryExists else {
            SessionLogger.shared.log(
                "[RESUME] Last-session pointer names missing folder \(pointer.directoryPath) — clearing stale pointer"
            )
            LastSessionPointer.clear()
            return
        }
        autoResumePointer = pointer
        // Best-effort peek of the session's metadata file so the
        // sheet can render counters / build info up front. A peek
        // failure leaves `autoResumeSummary` nil and the sheet
        // gracefully falls back to the minimal layout — we never
        // suppress the prompt over a peek failure, since the
        // primary purpose (offering Resume) is independent of the
        // metadata.
        do {
            autoResumeSummary = try CheckpointManager.peekSessionMetadata(at: pointer.directoryURL)
        } catch {
            autoResumeSummary = nil
            SessionLogger.shared.log(
                "[RESUME] session.json peek failed for \(pointer.sessionID): "
                + "\(error.localizedDescription) — sheet will use minimal layout"
            )
        }
        autoResumeCountdownRemaining = Self.autoResumeCountdownStartSec
        autoResumeSheetShowing = true
        let savedAgo = max(0, Int(Date().timeIntervalSince1970) - Int(pointer.savedAtUnix))
        SessionLogger.shared.log(
            "[RESUME] Presenting auto-resume sheet for \(pointer.sessionID) (\(pointer.trigger), saved \(savedAgo)s ago)"
        )
        startAutoResumeCountdownTask()
    }

    /// Kick off the 1 Hz countdown timer that ticks
    /// `autoResumeCountdownRemaining` down to zero, at which point
    /// it fires the resume action as if the user had clicked the
    /// Resume button.
    @MainActor
    private func startAutoResumeCountdownTask() {
        autoResumeCountdownTask?.cancel()
        autoResumeCountdownTask = Task { @MainActor in
            while autoResumeSheetShowing && autoResumeCountdownRemaining > 0 {
                do {
                    try await Task.sleep(for: .seconds(1))
                } catch {
                    // Cancelled (user dismissed). Nothing to do.
                    return
                }
                if Task.isCancelled { return }
                autoResumeCountdownRemaining -= 1
            }
            // Only fire if the sheet is still up — a dismiss path
            // may have zeroed the counter on its way out.
            if autoResumeSheetShowing {
                performAutoResume()
            }
        }
    }

    /// Dismiss the sheet without resuming. The File menu item
    /// "Resume training from autosave" stays available so the
    /// user can still trigger the resume later during this
    /// launch.
    @MainActor
    private func dismissAutoResumeSheet() {
        autoResumeCountdownTask?.cancel()
        autoResumeCountdownTask = nil
        autoResumeSheetShowing = false
        autoResumeSummary = nil
        SessionLogger.shared.log("[RESUME] User dismissed auto-resume sheet")
    }

    /// Drive the `--train` CLI launch sequence: build a fresh
    /// network, wait for it to finish, start Play-and-Train, and
    /// flip the board picker to Candidate Test. Called once from
    /// the first `.onAppear` when `autoTrainOnLaunch` is set, after
    /// the menu command hub has been wired and its mirrored state
    /// has been synced (so `buildNetwork`'s enable-check state is
    /// live).
    ///
    /// The build completes asynchronously inside `buildNetwork` via
    /// a detached `Task`, so we can't call `startRealTraining`
    /// synchronously after it — `networkReady` would still be false.
    /// A short polling loop on the MainActor bridges the two: it
    /// awaits `isBuilding` returning to false (at which point either
    /// `network` is populated on success or is nil on failure),
    /// then either proceeds or bails with a log line. 100 ms is
    /// fine because the build takes at least multiple seconds of
    /// MPSGraph compile/graph-setup; we don't care about the exact
    /// sub-second at which the poll observes completion.
    ///
    /// Setting `playAndTrainBoardMode = .candidateTest` *after*
    /// `startRealTraining` is intentional. `startRealTraining` sets
    /// `realTraining = true` synchronously, which is what gates the
    /// Board picker's visibility — so the picker is already showing
    /// by the time the mode flip lands, and the probe flip to
    /// `.candidateTest` also marks `candidateProbeDirty` via the
    /// existing `onChange` side-effect probe so the driver fires an
    /// immediate forward-pass probe at the first gap point.
    @MainActor
    private func runAutoTrainLaunchSequence() {
        SessionLogger.shared.log("[APP] --train: starting auto-train launch sequence")
        // Defensive state checks. These should always pass on a
        // fresh launch (no network built yet, nothing running),
        // but if somehow another flow already kicked things off
        // we prefer a visible log line over silently colliding.
        if networkReady {
            SessionLogger.shared.log("[APP] --train: network already ready; skipping auto-train")
            return
        }
        if isBusy {
            SessionLogger.shared.log("[APP] --train: app is busy; skipping auto-train")
            return
        }
        // Apply any `--parameters` overrides BEFORE buildNetwork /
        // startRealTraining so the trainer and sampling-schedule
        // constructors read the overridden values out of their
        // normal @AppStorage / @State sources. The no-arg overload
        // pulls from the launch-time `cliConfig` field; a no-op when
        // `--parameters` was not supplied. The same underlying
        // `applyCliConfigOverrides(cfg:)` is also reachable from the
        // File menu's `Load Parameters…` item via
        // `applyCliConfigOverridesFromMenu(cfg:)`, so the override
        // logic is shared rather than CLI-only.
        applyCliConfigOverrides()
        buildNetwork()
        Task { @MainActor in
            while isBuilding {
                do {
                    try await Task.sleep(for: .milliseconds(100))
                } catch {
                    // Task was cancelled (e.g. window closed during
                    // launch). Leave state alone — the build Task
                    // inside buildNetwork runs to completion
                    // independently and will settle `isBuilding`
                    // and `network` on its own.
                    SessionLogger.shared.log("[APP] --train: auto-train poll cancelled before build completion")
                    return
                }
            }
            guard networkReady else {
                SessionLogger.shared.log("[APP] --train: network build failed; aborting auto-train")
                return
            }
            SessionLogger.shared.log("[APP] --train: network built; starting Play-and-Train")
            startRealTraining(mode: .freshOrFromLoadedSession)
            playAndTrainBoardMode = .candidateTest
            SessionLogger.shared.log("[APP] --train: switched to Candidate Test view")
        }
    }

    /// Write the `--parameters` JSON overrides into the relevant
    /// `@AppStorage` / `@State` fields. Only runs when `cliConfig`
    /// is non-nil (i.e. `--parameters` was supplied); each inner
    /// field is independently optional so a partial file only
    /// overrides the keys it names, leaving everything else at
    /// whatever the UI default or persisted value already is.
    ///
    /// Values flow from here into the trainer and the self-play
    /// / arena schedules through `ensureTrainer`, `buildSelfPlaySchedule`,
    /// and `buildArenaSchedule` — the same paths an interactive
    /// user triggers when they edit the same fields in the UI.
    /// `trainingParams.replayRatioAutoAdjust` is left alone because no CLI knob
    /// maps to it; a caller who wants a static training delay can
    /// just set `trainingParams.trainingStepDelayMs` to their desired value.
    /// One row in the change list returned by
    /// `applyCliConfigOverrides(cfg:)`. The label is the JSON key
    /// (e.g. `"learning_rate"`); `before` and `after` are formatted
    /// as strings so heterogeneous parameter types (Int, Double,
    /// Bool) can share a single tuple shape for logging.
    typealias ParameterOverrideChange = (label: String, before: String, after: String)

    /// Apply CLI-style overrides from `cliConfig` (the `--parameters`
    /// flag's payload) onto the live `@AppStorage` / `@State` values.
    /// No-op when `cliConfig` is nil — i.e., the user did not supply
    /// `--parameters` at launch. Returns the list of fields that
    /// actually changed. Used at launch and indirectly by the File
    /// menu's `Load Parameters…` item (which routes through
    /// `applyCliConfigOverridesFromMenu`).
    @MainActor
    @discardableResult
    private func applyCliConfigOverrides() -> [ParameterOverrideChange] {
        guard let cfg = cliConfig else { return [] }
        return applyCliConfigOverrides(cfg: cfg)
    }

    /// Variant that takes the config explicitly. Used by the
    /// File menu's Load Parameters… handler, which loads a
    /// `CliTrainingConfig` from disk at any time during the run
    /// (not just at launch). Returns the list of changed fields
    /// so the menu handler can surface a count and per-field
    /// summary in the UI status row.
    @MainActor
    @discardableResult
    private func applyCliConfigOverridesFromMenu(cfg: CliTrainingConfig) -> [ParameterOverrideChange] {
        applyCliConfigOverrides(cfg: cfg)
    }

    @MainActor
    @discardableResult
    private func applyCliConfigOverrides(cfg: CliTrainingConfig) -> [ParameterOverrideChange] {
        // Pre-process the value map: apply UI-specific clamps that the
        // macro's range alone can't express (e.g. snapping
        // training_step_delay_ms onto the Stepper's ladder, or
        // narrowing self_play_workers to a per-build absolute cap that
        // is tighter than the macro's permissive 1...256).
        var values = cfg.trainingParameters
        if case .int(let v) = values[SelfPlayWorkers.id] {
            let clamped = max(1, min(Self.absoluteMaxSelfPlayWorkers, v))
            values[SelfPlayWorkers.id] = .int(clamped)
        }
        if case .int(let v) = values[TrainingStepDelayMs.id] {
            let clamped = max(0, min(Self.stepDelayMaxMs, v))
            let ladder = Self.stepDelayLadder
            let nearest = ladder.min(by: { abs($0 - clamped) < abs($1 - clamped) }) ?? clamped
            values[TrainingStepDelayMs.id] = .int(nearest)
        }
        if case .int(let v) = values[ArenaConcurrency.id] {
            let clamped = max(1, min(Self.absoluteMaxArenaConcurrency, v))
            values[ArenaConcurrency.id] = .int(clamped)
        }

        // Capture before-snapshot for diffing.
        let beforeSnap = trainingParams.snapshot().rawValueMap()

        // Apply through TrainingParameters — definition.validate runs on each
        // typed setter; out-of-range values throw and are surfaced via the
        // catch below rather than silently dropped.
        do {
            try trainingParams.apply(values)
        } catch {
            SessionLogger.shared.log("[APP] --parameters: apply failed: \(error)")
            return []
        }

        // Build the per-field change log by comparing before vs. after on
        // every id present in the input map.
        let afterSnap = trainingParams.snapshot().rawValueMap()
        var changes: [(String, String, String)] = []
        for id in values.keys.sorted() {
            let before = beforeSnap[id].map(formatParameterValue) ?? "?"
            let after = afterSnap[id].map(formatParameterValue) ?? "?"
            if before != after {
                changes.append((id, before, after))
            }
        }
        for (label, before, after) in changes {
            SessionLogger.shared.log("[APP] --parameters override: \(label): \(before) -> \(after)")
        }
        if changes.isEmpty {
            SessionLogger.shared.log("[APP] --parameters: no overrides applied (empty or all-nil config)")
        } else {
            SessionLogger.shared.log("[APP] --parameters: applied \(changes.count) override(s) to live state")
        }
        return changes.map { (label: $0.0, before: $0.1, after: $0.2) }
    }

    private func formatParameterValue(_ v: ParameterValue) -> String {
        switch v {
        case .bool(let x): "\(x)"
        case .int(let x): "\(x)"
        case .double(let x): "\(x)"
        }
    }

    /// Trigger the actual resume: tear down the sheet, then chain
    /// through `loadSessionFrom(url:startAfterLoad:)` which handles
    /// the network-build, weight-load, and Play-and-Train start.
    /// Errors surface via `setCheckpointStatus(.error)` and the
    /// session log — we deliberately do NOT delete the pointer or
    /// the target folder on failure per the spec ("If the session
    /// can't be loaded, throw up an error and stop. Don't delete
    /// anything.").
    @MainActor
    private func performAutoResume() {
        guard let pointer = autoResumePointer else {
            dismissAutoResumeSheet()
            return
        }
        autoResumeCountdownTask?.cancel()
        autoResumeCountdownTask = nil
        autoResumeSheetShowing = false
        autoResumeSummary = nil
        autoResumeInFlight = true
        SessionLogger.shared.log(
            "[RESUME] Starting auto-resume of \(pointer.sessionID) from \(pointer.directoryURL.lastPathComponent)"
        )
        loadSessionFrom(url: pointer.directoryURL, startAfterLoad: true)
    }

    /// SwiftUI content for the auto-resume sheet. Returns `AnyView`
    /// (rather than `some View` via `@ViewBuilder`) so the call
    /// site in the main body doesn't contribute to the already-huge
    /// body's type-inference cost — the `.sheet { ... }` modifier
    /// only has to prove the closure returns a concrete `View`,
    /// not figure out which of two opaque branches it is.
    private func autoResumeSheetContentView() -> AnyView {
        guard let pointer = autoResumePointer else {
            return AnyView(EmptyView())
        }
        let savedAt = Date(timeIntervalSince1970: TimeInterval(pointer.savedAtUnix))
        let formatter = DateFormatter()
        formatter.dateStyle = .medium
        formatter.timeStyle = .short
        let savedAtString = formatter.string(from: savedAt)
        let agoString = AutoResumeFormat.relativeAgo(savedAtUnix: pointer.savedAtUnix)
        let remaining = autoResumeCountdownRemaining
        let plural = (remaining == 1 ? "" : "s")
        let sessionLine = "Session: \(pointer.sessionID)"
        let savedLine = "Saved \(agoString) (\(savedAtString))"
        let folderLine = pointer.directoryURL.lastPathComponent
        let countdownLine = "Training will automatically resume in \(remaining) second\(plural)."
        let resumeLabel = "Resume Training (\(remaining))"

        let content = VStack(alignment: .leading, spacing: 14) {
            HStack(alignment: .firstTextBaseline) {
                Text("Resume last training session?")
                    .font(.title2)
                    .fontWeight(.semibold)
                Spacer()
                AutoResumeTriggerBadgeView(trigger: pointer.trigger)
            }
            VStack(alignment: .leading, spacing: 4) {
                Text(sessionLine)
                if let summary = autoResumeSummary {
                    Text(AutoResumeFormat.startedLine(sessionStartUnix: summary.sessionStartUnix))
                        .font(.callout)
                        .foregroundStyle(.secondary)
                }
                Text(savedLine)
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }
            if let summary = autoResumeSummary {
                AutoResumeProgressBlockView(summary: summary)
                AutoResumeBuildBlockView(summary: summary)
            }
            Button(action: {
                CheckpointManager.revealInFinder(pointer.directoryURL)
            }) {
                Text(folderLine)
                    .font(.system(.callout, design: .monospaced))
                    .foregroundStyle(.secondary)
                    .underline()
                    .lineLimit(1)
                    .truncationMode(.middle)
            }
            .buttonStyle(.plain)
            .pointerStyle(.link)
            .help("Reveal in Finder")
            Text(countdownLine)
                .font(.callout)
                .foregroundStyle(.secondary)
            HStack(spacing: 12) {
                Spacer()
                Button("Not Now") {
                    dismissAutoResumeSheet()
                }
                .keyboardShortcut(.cancelAction)
                Button(resumeLabel) {
                    performAutoResume()
                }
                .keyboardShortcut(.defaultAction)
            }
        }
        .padding(20)
        .frame(minWidth: 520)
        return AnyView(content)
    }


    /// File-menu entry point for "Resume training from autosave".
    /// Shares `performAutoResume`'s implementation but re-reads
    /// the pointer from UserDefaults first, since the user may
    /// have performed another save (which updated the pointer)
    /// between the launch sheet and the menu click.
    @MainActor
    private func resumeFromAutosaveMenuAction() {
        SessionLogger.shared.log("[BUTTON] Resume Training from Autosave")
        guard !realTraining else {
            refuseMenuAction("Stop the current training session before resuming from autosave.")
            return
        }
        guard !autoResumeInFlight else { return }
        guard let pointer = LastSessionPointer.read() else {
            refuseMenuAction("No saved session available to resume.")
            return
        }
        guard pointer.directoryExists else {
            // Pointer names a folder the user has since deleted.
            // Clear it so subsequent menu-sync ticks disable this
            // item and the next launch's auto-resume prompt does
            // not reappear for a session that will never load.
            SessionLogger.shared.log(
                "[RESUME] Menu resume target missing on disk (\(pointer.directoryPath)) — clearing stale pointer"
            )
            LastSessionPointer.clear()
            refuseMenuAction("Saved session no longer on disk.")
            syncMenuCommandHubState()
            return
        }
        autoResumePointer = pointer
        performAutoResume()
    }

    /// Open Finder pointed at the checkpoint root so the user can
    /// browse saved sessions and models even though Application
    /// Support is hidden by default. Creates the folder if it
    /// doesn't exist yet so the button always works.
    private func handleRevealSaves() {
        do {
            try CheckpointPaths.ensureDirectories()
        } catch {
            setCheckpointStatus("Could not create save folder: \(error.localizedDescription)", kind: .error)
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
        commandHub.startRealTraining = { startTrainingFromMenu() }
        commandHub.startSweep = { startSweep() }
        commandHub.stopAnyContinuous = { stopAnyContinuous() }
        commandHub.runArena = {
            SessionLogger.shared.log("[BUTTON] Run Arena")
            guard !isArenaRunning else { return }
            arenaTriggerBox?.trigger()
        }
        commandHub.runEngineDiagnostics = { runEngineDiagnostics() }
        commandHub.runPolicyConditioningDiagnostic = { runPolicyConditioningDiagnostic() }
        commandHub.recoverArenaHistoryFromLogs = { runArenaHistoryRecovery() }
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
        commandHub.loadParameters = {
            SessionLogger.shared.log("[BUTTON] Load Parameters")
            showingLoadParametersImporter = true
        }
        commandHub.saveParameters = {
            handleSaveParametersMenuAction()
        }
        commandHub.resumeFromAutosave = {
            resumeFromAutosaveMenuAction()
        }
        commandHub.revealSaves = { handleRevealSaves() }
        commandHub.chartZoomIn = { chartZoomIn() }
        commandHub.chartZoomOut = { chartZoomOut() }
        commandHub.chartZoomEnableAuto = { chartZoomEnableAuto() }
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
        commandHub.canResumeFromAutosave = canResumeFromAutosave
        commandHub.arenaRecoveryInProgress = arenaRecoveryInProgress
        // Zoom: the chart grid only renders during Play-and-Train, so
        // the View menu items follow the same gate plus their
        // individual extremum checks.
        commandHub.chartZoomInAvailable = realTraining && canZoomChartIn
        commandHub.chartZoomOutAvailable = realTraining && canZoomChartOut
        commandHub.chartZoomAutoAvailable = realTraining && !chartCoordinator.chartZoomAuto
    }

    /// True iff any operation is currently running that conflicts
    /// with menu-driven Build / Load / Save actions. Used by the
    /// in-function guards in those functions — the menu items are
    /// already disabled for the same conditions, so this fires
    /// only when invoked via keyboard shortcut / URL scheme under
    /// a race, or if the menu state is stale.
    private func isBuildingOrBusy() -> Bool {
        return realTraining
            || continuousPlay
            || continuousTraining
            || sweepRunning
            || gameSnapshot.isPlaying
            || isBuilding
            || checkpointSaveInFlight
    }

    /// Human-readable explanation of *why* `isBuildingOrBusy()` is
    /// true, for the refusal alert.
    private func busyReasonMessage() -> String {
        if realTraining {
            return "Training is running. Stop it before loading or building a model."
        }
        if continuousPlay || continuousTraining {
            return "A continuous task is running. Stop it first."
        }
        if sweepRunning {
            return "A sweep is running. Stop it first."
        }
        if gameSnapshot.isPlaying {
            return "A game is in progress. Wait for it to finish."
        }
        if isBuilding {
            return "The network is still being built. Wait for it to finish."
        }
        if checkpointSaveInFlight {
            return "A save/load is already in progress. Wait for it to finish."
        }
        return "Another operation is in progress."
    }

    /// Ensure `network` exists. If it's already built, returns it.
    /// Otherwise runs the same detached `performBuild` path the
    /// menu's Build Network button uses and wires the result into
    /// the view's state. Returns the champion network on success,
    /// or the build error on failure.
    private func ensureChampionBuilt() async -> Result<ChessMPSNetwork, Error> {
        if let champion = network {
            return .success(champion)
        }
        isBuilding = true
        networkStatus = ""
        trainer = nil
        clearTrainingDisplay()
        SessionLogger.shared.log("[BUILD] Auto-build before load")
        let result = await Task.detached(priority: .userInitiated) {
            Self.performBuild()
        }.value
        switch result {
        case .success(let net):
            net.identifier = ModelIDMinter.mint()
            network = net
            runner = ChessRunner(network: net)
            isBuilding = false
            return .success(net)
        case .failure(let error):
            network = nil
            runner = nil
            networkStatus = "Build failed: \(error.localizedDescription)"
            isBuilding = false
            return .failure(error)
        }
    }

    private func buildNetwork() {
        SessionLogger.shared.log("[BUTTON] Build Network")
        // In-function guards (belt-and-suspenders with menu disable).
        if isBusy {
            refuseMenuAction(busyReasonMessage())
            return
        }
        if networkReady {
            refuseMenuAction("A network is already built. Load Model or Load Session to replace its weights.")
            return
        }
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
        // Guards: Candidate test active, probe runner and network
        // built, trainer available for the trainer → probe sync. All
        // of these are normally true during Play and Train — the
        // early-return cases cover "just-started race before the
        // probe network was built" or "trainer wasn't initialized."
        // Arena-active is NOT a blanket skip any more: the candidate
        // branch uses a dedicated probe network that the arena never
        // touches, so it can fire freely through arenas. The champion
        // branch still skips during arenas (see its case below) —
        // the promotion step briefly writes into the champion under
        // a self-play pause and we don't want to race that write.
        guard
            isCandidateTestActive,
            let trainer,
            let probeRunner,
            let probeInference = probeInferenceNetwork,
            let championRunner = runner
        else { return }
        let now = Date()
        let dirty = candidateProbeDirty
        let intervalElapsed = now.timeIntervalSince(lastCandidateProbeTime)
        >= trainingParams.candidateProbeIntervalSec
        guard dirty || intervalElapsed else { return }

        let state = editableState
        let target = probeNetworkTarget
        let result: EvaluationResult
        do {
            switch target {
            case .candidate:
                // Snapshot the trainer's current state into the probe
                // inference network, then immediately run the probe. Doing
                // the copy here — rather than after every training block —
                // means the ~11.6 MB trainer → probe transfer happens only
                // when the probe is actually about to fire (every 15 s or
                // on drag/side-to-move/mode-flip), not at the ~per-second
                // cadence of training blocks.
                //
                // The probe network is dedicated — no one else reads or
                // writes it — so the probe can run concurrently with an
                // active arena (which reads `candidateInferenceNetwork`,
                // a different object) without racing. The only potential
                // concurrent operation is `trainer.network.exportWeights`
                // during an arena's own trainer-snapshot step; both are
                // reads under the network's internal lock and are safe.
                //
                // Both the sync and the forward pass run on a detached task
                // so we don't stall MainActor while they execute.
                result = try await Task.detached(priority: .userInitiated) {
                    let weights = try await trainer.network.exportWeights()
                    try await probeInference.loadWeights(weights)
                    return await Self.performInference(with: probeRunner, state: state)
                }.value
                // Probe is a transient read-only snapshot, not a checkpoint —
                // probeInference inherits the trainer's current ID rather
                // than minting a fresh one. (Arena snapshots, by contrast,
                // do mint — see runArenaParallel.)
                probeInference.identifier = trainer.identifier
            case .champion:
                // Skip champion probes while an arena is running — the
                // promotion step briefly writes into the champion under
                // a self-play pause and the probe would race that write.
                // (The candidate branch above has no such constraint: it
                // uses a dedicated probe network.)
                if arenaActiveFlag?.isActive == true { return }
                // Probe the champion directly — no sync. The champion is
                // frozen between promotions, so reading from it through
                // its own runner is the same path Run Forward Pass uses
                // and is safe to call concurrently with self-play workers
                // (they all read through a batcher; direct runner reads
                // just add another fair-share consumer).
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
        // CLI-mode capture: if an output JSON is configured, append
        // this probe's diagnostics alongside the arena and stats
        // streams. No-op in normal interactive runs — the recorder
        // is only allocated when `cliOutputURL` is non-nil. Skipped
        // when the forward pass failed (rawInference == nil) so a
        // failed probe doesn't show up as a zeroed entry in the
        // output.
        if let recorder = cliRecorder,
           let inf = result.rawInference,
           let sessionStart = currentSessionStart {
            let elapsed = Date().timeIntervalSince(sessionStart)
            let event = buildCliCandidateTestEvent(
                elapsedSec: elapsed,
                probeIndex: candidateProbeCount,
                target: target,
                state: state,
                inference: inf
            )
            recorder.appendCandidateTest(event)
        }
    }

    /// Build a `CliTrainingRecorder.CandidateTest` from a finished
    /// forward-pass result. Computes the same on-screen policy
    /// diagnostics the text output shows (top-100 sum, above-uniform
    /// count, legal-mass sum, min/max) plus a structured top-10
    /// list — mirroring `performInference` so the JSON and the UI
    /// stay in sync. Factored out so the probe fire path is short
    /// and the capture logic can be exercised independently in
    /// the future if we ever add a headless test for it.
    nonisolated private func buildCliCandidateTestEvent(
        elapsedSec: Double,
        probeIndex: Int,
        target: ProbeNetworkTarget,
        state: GameState,
        inference: ChessRunner.InferenceResult
    ) -> CliTrainingRecorder.CandidateTest {
        let policy = inference.policy
        let sum = Double(policy.reduce(0, +))
        let top100Sum = Double(policy.sorted(by: >).prefix(100).reduce(0, +))
        let minP = Double(policy.min() ?? 0)
        let maxP = Double(policy.max() ?? 0)
        let legalMoves = MoveGenerator.legalMoves(for: state)
        let nLegal = max(1, legalMoves.count)
        let legalUniformThreshold = 1.0 / Double(nLegal)
        let legalIndices = legalMoves
            .map { PolicyEncoding.policyIndex($0, currentPlayer: state.currentPlayer) }
        let abovePerLegalCount = legalIndices.filter { idx in
            idx >= 0 && idx < policy.count
                && Double(policy[idx]) > legalUniformThreshold
        }.count
        let legalMassSum: Double = legalIndices.reduce(0.0) { acc, idx in
            (idx >= 0 && idx < policy.count) ? acc + Double(policy[idx]) : acc
        }
        // Top-10 raw — same extraction path the UI uses for top-4
        // (legality-agnostic, geometrically decoded), just with a
        // larger K.
        let top10 = ChessRunner.extractTopMoves(
            from: policy,
            state: state,
            pieces: state.board,
            count: 10
        )
        let topMovesOut: [CliTrainingRecorder.CandidateTest.TopMove] = top10.enumerated().map { (rank, mv) in
            CliTrainingRecorder.CandidateTest.TopMove(
                rank: rank + 1,
                from: BoardEncoder.squareName(mv.fromRow * 8 + mv.fromCol),
                to: BoardEncoder.squareName(mv.toRow * 8 + mv.toCol),
                fromRow: mv.fromRow,
                fromCol: mv.fromCol,
                toRow: mv.toRow,
                toCol: mv.toCol,
                probability: Double(mv.probability),
                isLegal: mv.isLegal
            )
        }
        let stats = CliTrainingRecorder.CandidateTest.PolicyStats(
            sum: sum,
            top100Sum: top100Sum,
            aboveUniformCount: abovePerLegalCount,
            legalMoveCount: legalMoves.count,
            legalUniformThreshold: legalUniformThreshold,
            legalMassSum: legalMassSum,
            illegalMassSum: max(0.0, 1.0 - legalMassSum),
            min: minP,
            max: maxP
        )
        let targetStr: String
        switch target {
        case .candidate: targetStr = "candidate"
        case .champion: targetStr = "champion"
        }
        return CliTrainingRecorder.CandidateTest(
            elapsedSec: elapsedSec,
            probeIndex: probeIndex,
            probeNetworkTarget: targetStr,
            inferenceTimeMs: inference.inferenceTimeMs,
            valueHead: CliTrainingRecorder.CandidateTest.ValueHead(output: Double(inference.value)),
            policyHead: CliTrainingRecorder.CandidateTest.PolicyHead(
                policyStats: stats,
                topRaw: topMovesOut
            )
        )
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
            let bufCap = replayBuffer?.capacity ?? trainingParams.replayBufferCapacity
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
        // Let the periodic-save scheduler know an arena is in
        // progress. A 4-hour deadline that crosses while an arena
        // runs will be held as a pending fire and only dispatched
        // once the arena ends — unless a post-promotion autosave
        // lands in the meantime, in which case the pending fire is
        // swallowed (see `PeriodicSaveController`).
        periodicSaveController?.noteArenaBegan()
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
            chartCoordinator.recordArenaStarted(elapsedSec: Date().timeIntervalSince(sessionStart))
        }

        let totalGames = trainingParams.arenaGamesPerTournament
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
        //
        // We also snapshot the optimizer velocity. If the candidate
        // wins the arena, we'll restore THIS snapshot when we copy
        // the candidate weights back into the trainer — the velocity
        // that built the validated candidate is the right velocity
        // for the candidate's weight surface. (Earlier behavior
        // zeroed velocity on promotion, throwing away accumulated
        // gradient signal.)
        var trainerSnapshotWeights: [[Float]] = []
        var trainerSnapshotVelocity: [[Float]] = []
        do {
            let snapshot: ([[Float]], [[Float]]) = try await Task.detached(priority: .userInitiated) {
                let weights = try await trainer.network.exportWeights()
                let velocity = try await trainer.exportVelocitySnapshot()
                try await candidateInference.loadWeights(weights)
                return (weights, velocity)
            }.value
            trainerSnapshotWeights = snapshot.0
            trainerSnapshotVelocity = snapshot.1
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

        // --- Concurrent arena: build per-network batchers ---
        //
        // K parallel arena games share two `BatchedMoveEvaluationSource`
        // batchers, one per network. Each game alternates candidate ↔
        // champion moves, so at any instant ~K/2 are pending on each
        // side; a strict count-only barrier (the self-play model) would
        // never fire either batcher in steady state. The coalescing-
        // window timer (`maxBatchWaitMs`) makes the barrier fire on
        // whichever happens first — count met OR window expired —
        // which captures the early-game synchronized phase as a full
        // K-batch and the desynchronized steady state as ~K/2 partial
        // batches.
        //
        // Live count tracking: `expectedSlotCount` starts at `liveK`
        // (clamped to game count) and stays there while replacements
        // keep the pool full. The `onSlotRetired` callback below
        // fires only when a slot leaves the pool (no replacement
        // spawned) — i.e. exactly when each of the final K games
        // finishes — so each batcher's `expectedSlotCount` decrements
        // 1:1 with live game count during the tail. Decrementing on
        // every completion (the prior approach) drove the count to
        // 0 partway through the tournament and forced the rest of
        // the run through drain mode = single-position batches.
        let liveK = max(1, min(trainingParams.arenaConcurrency, totalGames))
        // Joint GPU-busy-wall accumulator shared across both batchers.
        // Each fire reports start/end around its `network.evaluate`
        // await; the timer accumulates wall time during which AT
        // LEAST ONE side was on the GPU. Read once at the end of
        // arena to emit a single `[ARENA] timing joint` line that's
        // directly comparable to arena wall.
        let arenaGpuTimer = ArenaGpuTimer()
        let candidateBatcher = BatchedMoveEvaluationSource(
            network: candidateInference,
            maxBatchWaitMs: Self.arenaBatchWaitMs,
            name: "candidate",
            logBatchTimings: true,
            gpuTimer: arenaGpuTimer
        )
        let championBatcher = BatchedMoveEvaluationSource(
            network: arenaChampion,
            maxBatchWaitMs: Self.arenaBatchWaitMs,
            name: "champion",
            logBatchTimings: true,
            gpuTimer: arenaGpuTimer
        )
        await candidateBatcher.setExpectedSlotCount(liveK)
        await championBatcher.setExpectedSlotCount(liveK)

        // Update the live progress box with the chosen concurrency so
        // the arena's busy-label suffix can render "(×K concurrent)"
        // for the duration of this run.
        tBox.update(TournamentProgress(
            currentGame: 0,
            totalGames: totalGames,
            candidateWins: 0,
            championWins: 0,
            draws: 0,
            startTime: startTime,
            concurrency: liveK
        ))
        tournamentProgress = tBox.snapshot()

        // Records collector. The driver fires `onGameRecorded` from
        // its parent harvest loop — already serial in that task — so
        // this Box's append happens on a single task and needs no
        // synchronization. We lift it to a `final class` only because
        // the closure must be `@Sendable`; the lock is a defensive
        // belt for that contract, not for actual contention.
        let recordsBox = TournamentRecordsBox()
        let stats: TournamentStats
        do {
            stats = try await withTaskCancellationHandler {
                try await Task.detached(priority: .userInitiated) {
                    [candidateBatcher, championBatcher, tBox, cancelBox, overrideBox, arenaDiversity, arenaScheduleSnapshot, liveK, recordsBox] in
                    let driver = TournamentDriver()
                    return try await driver.run(
                        playerA: {
                            MPSChessPlayer(
                                name: "Candidate",
                                source: candidateBatcher,
                                schedule: arenaScheduleSnapshot
                            )
                        },
                        playerB: {
                            MPSChessPlayer(
                                name: "Champion",
                                source: championBatcher,
                                schedule: arenaScheduleSnapshot
                            )
                        },
                        games: totalGames,
                        concurrency: liveK,
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
                                startTime: startTime,
                                concurrency: liveK
                            ))
                        },
                        onSlotRetired: {
                            // Fires only when a slot leaves the
                            // live pool (no replacement spawned).
                            // Decrementing on every completion would
                            // peg the count at 0 mid-tournament and
                            // collapse all remaining batches to size 1.
                            await candidateBatcher.decrementExpectedSlotCount()
                            await championBatcher.decrementExpectedSlotCount()
                        },
                        onGameRecorded: { record in
                            recordsBox.append(record)
                        }
                    )
                }.value
            } onCancel: {
                cancelBox.cancel()
            }
        } catch {
            // Error path. Drain both batchers so any parked
            // continuations resume, then emit whatever telemetry we
            // can over the partial records captured before the throw
            // (so the user can still see how much concurrency we got
            // up to the failure and confirm whether the captured
            // games were internally consistent before bailing).
            // After that, surface the error and tear down arena
            // state.
            await candidateBatcher.setExpectedSlotCount(0)
            await championBatcher.setExpectedSlotCount(0)
            await emitArenaPostRunTelemetry(
                candidateBatcher: candidateBatcher,
                championBatcher: championBatcher,
                gpuTimer: arenaGpuTimer,
                recordsBox: recordsBox,
                wasCancelled: cancelBox.isCancelled || overrideBox.isActive,
                context: "after error",
                arenaStartTime: startTime,
                trainingBox: trainingBox
            )
            trainingBox?.recordError("Arena tournament failed: \(error.localizedDescription)")
            cleanupArenaState(arenaFlag: arenaFlag, tBox: tBox)
            return
        }

        // Drain the batchers explicitly. Drain mode fires any
        // straggler pendings immediately so no continuation is left
        // parked, even on cancellation paths where the driver may
        // have stopped harvesting before all in-flight slots returned.
        await candidateBatcher.setExpectedSlotCount(0)
        await championBatcher.setExpectedSlotCount(0)

        await emitArenaPostRunTelemetry(
            candidateBatcher: candidateBatcher,
            championBatcher: championBatcher,
            gpuTimer: arenaGpuTimer,
            recordsBox: recordsBox,
            wasCancelled: cancelBox.isCancelled || overrideBox.isActive,
            context: nil,
            arenaStartTime: startTime,
            trainingBox: trainingBox
        )

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
            shouldPromote = playedGames >= totalGames && score >= trainingParams.arenaPromoteThreshold
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
                        [candidateInference, champion, trainer, trainerSnapshotVelocity] in
                        let weights = try await candidateInference.exportWeights()
                        try await champion.loadWeights(weights)
                        try await trainer.network.loadWeights(weights)
                        // The trainer's CURRENT velocity was built up
                        // against the post-arena weight surface (which
                        // we just discarded by overwriting with the
                        // candidate weights). Restore the velocity we
                        // snapshotted at arena-start instead — that
                        // velocity is the EMA of gradients that built
                        // the validated candidate, so it's the right
                        // accumulator for the candidate's weight
                        // surface. Both gates are paused at this point,
                        // so the trainer's velocity I/O is safe to
                        // drive directly on network.graph.
                        try await trainer.loadVelocitySnapshot(trainerSnapshotVelocity)
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
            finishedAt: Date(),
            candidateID: candidateInference.identifier,
            championID: arenaChampion.identifier,
            gamesPlayed: playedGames,
            candidateWins: stats.playerAWins,
            championWins: stats.playerBWins,
            draws: stats.draws,
            score: score,
            promoted: promoted,
            promotionKind: promoted ? promotionKind : nil,
            promotedID: promotedID,
            durationSec: durationSec,
            candidateWinsAsWhite: stats.playerAWinsAsWhite,
            candidateWinsAsBlack: stats.playerAWinsAsBlack,
            candidateLossesAsWhite: stats.playerALossesAsWhite,
            candidateLossesAsBlack: stats.playerALossesAsBlack,
            candidateDrawsAsWhite: stats.playerADrawsAsWhite,
            candidateDrawsAsBlack: stats.playerADrawsAsBlack
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
            let startElapsed = chartCoordinator.activeArenaStartElapsed
            ?? max(0, endElapsed - durationSec)
            // `recordArenaCompleted` appends the event AND clears
            // the live-band marker, so the chart drops back to just
            // the completed events on the next render.
            chartCoordinator.recordArenaCompleted(ArenaChartEvent(
                id: chartCoordinator.arenaChartEvents.count,
                startElapsedSec: startElapsed,
                endElapsedSec: endElapsed,
                score: score,
                promoted: promoted
            ))
        } else {
            // No session anchor (shouldn't happen mid-arena) — at
            // least cancel the live band so it doesn't dangle.
            chartCoordinator.cancelActiveArena()
        }
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
        //
        // Status row: we publish the "Saving… / Saved" progression
        // via the same `setCheckpointStatus` channel the manual
        // save uses, tagged "(post-promotion)" so users can tell
        // at a glance which autosave trigger produced the line.
        // `periodicSaveController` sees the success callback and
        // resets its 4-hour deadline, so a promotion that happens
        // shortly before a scheduled periodic save implicitly
        // defers the periodic one — which matches the spec: if a
        // post-promotion save already covered the window, the
        // next periodic tick runs a full 4 hours later from now.
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

            setCheckpointStatus("Saving session (post-promotion)…", kind: .progress)
            checkpointSaveInFlight = true
            startSlowSaveWatchdog(label: "session save (post-promotion)")
            // Fire-and-forget detached task. The closure captures
            // only Sendable value types (weight arrays, metadata
            // structs, the session state snapshot, and a few
            // strings) — explicitly NOT `self` — so we can safely
            // run past a session end without touching any View
            // @State. The completion handler hops back to the
            // MainActor to surface the outcome in the status row
            // and to reset `periodicSaveController`'s deadline;
            // that hop is safe because the handler only touches
            // the View's @State (no network or gate reads).
            Task.detached(priority: .utility) {
                [bufferForAutosave] in
                let outcome: Result<(URL, String), Error>
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
                    outcome = .success((url, bufStr))
                } catch {
                    outcome = .failure(error)
                }
                await MainActor.run {
                    switch outcome {
                    case .success(let (url, bufStr)):
                        setCheckpointStatus(
                            "Saved \(url.lastPathComponent) (post-promotion)",
                            kind: .success
                        )
                        SessionLogger.shared.log(
                            "[CHECKPOINT] Saved session (post-promotion): \(url.lastPathComponent) build=\(BuildInfo.buildNumber) git=\(BuildInfo.gitHash)\(bufStr)"
                        )
                        recordLastSessionPointer(
                            directoryURL: url,
                            sessionID: sessionState.sessionID,
                            trigger: "post-promotion"
                        )
                        periodicSaveController?.noteSuccessfulSave(at: Date())
                    case .failure(let error):
                        setCheckpointStatus(
                            "Autosave failed (post-promotion): \(error.localizedDescription)",
                            kind: .error
                        )
                        SessionLogger.shared.log(
                            "[CHECKPOINT] Saved session (post-promotion) failed: \(error.localizedDescription)"
                        )
                    }
                    cancelSlowSaveWatchdog()
                    checkpointSaveInFlight = false
                }
            }
        }
    }

    /// Emit an expanded multi-line summary of a just-finished arena
    /// tournament to stdout and the session log. Format:
    ///
    ///   [ARENA] #N Candidate vs Champion @ step S
    ///   [ARENA]     Games: G       (played/total)
    ///   [ARENA]     Result: W wins / D draws / L losses (candidate perspective)
    ///   [ARENA]     Score:  51.2% [48.1%, 54.3%]
    ///   [ARENA]     Elo diff: +28 [+13, +34]
    ///   [ARENA]     Draw rate: 17.5%
    ///   [ARENA]     By side:
    ///   [ARENA]       Candidate as white: 54.1%  (W/D/L n/n/n)
    ///   [ARENA]       Candidate as black: 48.3%  (W/D/L n/n/n)
    ///   [ARENA]     batch=… lr=… promote>=… sp.tau=… ar.tau=… workers=… build=…
    ///   [ARENA]     candidate=…  champion=…  trainer=…
    ///   [ARENA]     diversity: unique=…/… (…%) avgDiverge=…
    ///   [ARENA]     Verdict: PROMOTED(auto)=ID | kept    dur=m:ss
    ///   [ARENA] #N kv step=… games=… w=… d=… l=… score=… elo=… elo_lo=… elo_hi=… draw_rate=… cand_white_score=… cand_black_score=… promoted=… kind=… dur_sec=… build=… candidate=… champion=… trainer=…
    ///
    /// The trailing `kv` line stays a single key=value line so
    /// grep-based tooling can pull every arena outcome without
    /// reassembling the multi-line block.
    @MainActor
    private func logArenaResult(
        record: TournamentRecord,
        index: Int,
        trainer: ChessTrainer,
        candidate: ChessMPSNetwork,
        championSide: ChessMPSNetwork,
        diversity: GameDiversityTracker.Snapshot
    ) {
        let sp = samplingScheduleBox?.selfPlay ?? buildSelfPlaySchedule()
        let ar = samplingScheduleBox?.arena ?? buildArenaSchedule()
        let candidateIDStr = candidate.identifier?.description ?? "?"
        let championIDStr = championSide.identifier?.description ?? "?"
        let trainerIDStr = trainer.identifier?.description ?? "?"

        let parameters = ArenaLogFormatter.Parameters(
            batchSize: trainingParams.trainingBatchSize,
            learningRate: trainer.learningRate,
            promoteThreshold: trainingParams.arenaPromoteThreshold,
            tournamentGames: trainingParams.arenaGamesPerTournament,
            spStartTau: sp.startTau,
            spFloorTau: sp.floorTau,
            spDecayPerPly: sp.decayPerPly,
            arStartTau: ar.startTau,
            arFloorTau: ar.floorTau,
            arDecayPerPly: ar.decayPerPly,
            workerCount: trainingParams.selfPlayWorkers,
            buildNumber: BuildInfo.buildNumber
        )
        let diversityCtx = ArenaLogFormatter.Diversity(
            uniqueGames: diversity.uniqueGames,
            gamesInWindow: diversity.gamesInWindow,
            uniquePercent: diversity.uniquePercent,
            avgDivergencePly: diversity.avgDivergencePly
        )

        let lines = ArenaLogFormatter.formatHumanReadable(
            record: record,
            index: index,
            candidateID: candidateIDStr,
            championID: championIDStr,
            trainerID: trainerIDStr,
            parameters: parameters,
            diversity: diversityCtx
        )
        let kv = ArenaLogFormatter.formatKVLine(
            record: record,
            index: index,
            candidateID: candidateIDStr,
            championID: championIDStr,
            trainerID: trainerIDStr,
            buildNumber: BuildInfo.buildNumber
        )

        for line in lines {
            print(line)
            SessionLogger.shared.log(line)
        }
        print(kv)
        SessionLogger.shared.log(kv)

        // CLI-mode capture: mirror the arena block's parse-target
        // fields (KV line) plus the extras from the human-readable
        // block that don't appear in the KV but are useful for
        // downstream tooling (tau schedules, diversity snapshot,
        // per-worker count). The recorder serializes these into
        // the arena_results[] array of the `--output` JSON.
        if let recorder = cliRecorder {
            let elo = record.eloSummary
            let entry = CliTrainingRecorder.Arena(
                index: index,
                finishedAtStep: record.finishedAtStep,
                gamesPlayed: record.gamesPlayed,
                tournamentGames: trainingParams.arenaGamesPerTournament,
                candidateWins: record.candidateWins,
                championWins: record.championWins,
                draws: record.draws,
                score: record.score,
                drawRate: ArenaLogFormatter.drawRateFraction(record: record),
                elo: elo.elo,
                eloLo: elo.eloLo,
                eloHi: elo.eloHi,
                scoreLo: elo.scoreLo,
                scoreHi: elo.scoreHi,
                candidateWinsAsWhite: record.candidateWinsAsWhite,
                candidateDrawsAsWhite: record.candidateDrawsAsWhite,
                candidateLossesAsWhite: record.candidateLossesAsWhite,
                candidateWinsAsBlack: record.candidateWinsAsBlack,
                candidateDrawsAsBlack: record.candidateDrawsAsBlack,
                candidateLossesAsBlack: record.candidateLossesAsBlack,
                candidateScoreAsWhite: record.candidateScoreAsWhite,
                candidateScoreAsBlack: record.candidateScoreAsBlack,
                promoted: record.promoted,
                promotionKind: record.promotionKind?.rawValue,
                promotedID: record.promotedID?.description,
                durationSec: record.durationSec,
                candidateID: candidateIDStr,
                championID: championIDStr,
                trainerID: trainerIDStr,
                learningRate: Double(trainer.learningRate),
                promoteThreshold: trainingParams.arenaPromoteThreshold,
                batchSize: trainingParams.trainingBatchSize,
                workerCount: trainingParams.selfPlayWorkers,
                spStartTau: Double(sp.startTau),
                spFloorTau: Double(sp.floorTau),
                spDecayPerPly: Double(sp.decayPerPly),
                arStartTau: Double(ar.startTau),
                arFloorTau: Double(ar.floorTau),
                arDecayPerPly: Double(ar.decayPerPly),
                diversityUniqueGames: diversity.uniqueGames,
                diversityGamesInWindow: diversity.gamesInWindow,
                diversityUniquePercent: diversity.uniquePercent,
                diversityAvgDivergencePly: diversity.avgDivergencePly,
                buildNumber: BuildInfo.buildNumber
            )
            recorder.appendArena(entry)
        }
    }

    /// Drain the per-batcher batch-size histograms and run the
    /// post-arena game-validity sweep, emitting one log line per
    /// batcher and one for the validation outcome. Called on every
    /// `runArenaParallel` exit leg — success, cancellation, AND
    /// thrown errors — so a mid-tournament throw still surfaces
    /// "how much concurrency did we get before it died" and
    /// "were the captured games internally consistent" rather
    /// than silently losing that diagnostic.
    ///
    /// The validity sweep is skipped under cancellation because
    /// partial records may be incomplete (a slot mid-game when
    /// cancelled didn't append a final move record); under errors
    /// we still run it because partial-but-completed games are
    /// well-formed and worth checking — if a slot threw mid-game,
    /// only its own record is missing, not the others'.
    ///
    /// `context` annotates the log line for non-success paths
    /// (e.g. "after error") so a log reader can tell at a glance
    /// the run didn't complete normally.
    private func emitArenaPostRunTelemetry(
        candidateBatcher: BatchedMoveEvaluationSource,
        championBatcher: BatchedMoveEvaluationSource,
        gpuTimer: ArenaGpuTimer,
        recordsBox: TournamentRecordsBox,
        wasCancelled: Bool,
        context: String?,
        arenaStartTime: Date,
        trainingBox: TrainingLiveStatsBox?
    ) async {
        let suffix = context.map { " (\($0))" } ?? ""
        let candidateBatchStats = await candidateBatcher.snapshotBatchSizeStats()
        let championBatchStats = await championBatcher.snapshotBatchSizeStats()
        let candidateTimingStats = await candidateBatcher.snapshotBatchTimingStats()
        let championTimingStats = await championBatcher.snapshotBatchTimingStats()
        SessionLogger.shared.log(
            "[ARENA] batch sizes  candidate\(suffix): \(candidateBatchStats.formatLogLine())"
        )
        SessionLogger.shared.log(
            "[ARENA] batch sizes  champion \(suffix): \(championBatchStats.formatLogLine())"
        )

        // Joint GPU-wall timing: a single number for "wall time during
        // which AT LEAST one side was on the GPU." Emitted before the
        // per-side timing lines because it's the easier-to-read summary
        // — `nonGpu = wall - gpu` is exactly the time the GPU was idle
        // (CPU per-ply work, scheduler gaps, continuation-resume
        // backpressure), with no per-side double-counting. The
        // per-side lines that follow remain useful for spotting
        // candidate-vs-champion balance (run-time difference, wait
        // asymmetry).
        let arenaTotalSec = max(0, Date().timeIntervalSince(arenaStartTime))
        let gpuBusySec = gpuTimer.totalBusyMs() / 1000.0
        let nonGpuSec = max(0, arenaTotalSec - gpuBusySec)
        let gpuUtilPct = arenaTotalSec > 0 ? (gpuBusySec / arenaTotalSec) * 100.0 : 0.0
        SessionLogger.shared.log(String(
            format: "[ARENA] timing joint%@: wall=%.1fs gpu=%.1fs nonGpu=%.1fs (gpu_util=%.1f%%)",
            suffix,
            arenaTotalSec,
            gpuBusySec,
            nonGpuSec,
            gpuUtilPct
        ))

        // Per-side wall-clock breakdown: wait/run vs the total arena
        // duration. `other` is the residual that's neither wait nor
        // run — time this batcher had nothing to do (the OTHER batcher
        // was busy, or slots were doing CPU-side work between
        // submissions). Each batcher is independent so the two `other`
        // numbers don't sum to anything meaningful; they're each "how
        // much of arena wall time was this batcher idle". Use the
        // joint line above for true GPU saturation.
        emitArenaTimingLine(label: "candidate", suffix: suffix, totalSec: arenaTotalSec, stats: candidateTimingStats)
        emitArenaTimingLine(label: "champion ", suffix: suffix, totalSec: arenaTotalSec, stats: championTimingStats)

        // Fire-reason histogram: answers "is the coalescing-window
        // timer actually firing the barrier, or are we just hitting
        // count-met every time?" without forcing a reader to infer
        // it from the size histogram. Always emits all five reasons
        // so the cand/champ lines line up visually for asymmetry
        // checks. Healthy steady-state: `full` dominates, `timer`
        // small, `drain` small (only on tournament drain), `threshold`
        // and `refill` small or zero.
        SessionLogger.shared.log(
            "[ARENA] fire reasons candidate\(suffix): \(candidateBatchStats.formatFireReasonsLine())"
        )
        SessionLogger.shared.log(
            "[ARENA] fire reasons champion \(suffix): \(championBatchStats.formatFireReasonsLine())"
        )

        // Pre/post `expectedSlotCount` drift counters: how many fires
        // saw the slot-count change during the GPU await, plus the
        // largest such delta. Steady-state non-zero is healthy (games
        // ending mid-fire is the dominant cause). Asymmetry across
        // sides or a sustained `maxDelta > ~5` would point at the
        // coordination paths between the harvest loop and the batcher
        // actor.
        SessionLogger.shared.log(
            "[ARENA] expected-drift candidate\(suffix): \(candidateBatchStats.formatExpectedDriftLine())"
        )
        SessionLogger.shared.log(
            "[ARENA] expected-drift champion \(suffix): \(championBatchStats.formatExpectedDriftLine())"
        )

        if wasCancelled {
            // Partial records under cancellation can be
            // structurally incomplete — skip rather than emit
            // misleading "validation FAILED" lines.
            SessionLogger.shared.log(
                "[ARENA] validation skipped\(suffix): tournament was cancelled mid-run"
            )
            emitArenaPostStatsLine(trainingBox: trainingBox, suffix: suffix)
            return
        }

        let report = validateTournamentRecords(recordsBox.snapshot())
        if report.passed {
            SessionLogger.shared.log(
                "[ARENA] validation passed\(suffix): \(report.gamesChecked) games, "
                + "\(report.totalMovesChecked) moves all legal in their position contexts"
            )
        } else {
            let detail = report.failureDescription ?? "(no detail)"
            SessionLogger.shared.log(
                "[ARENA] validation FAILED\(suffix): \(detail)"
            )
            trainingBox?.recordError("Arena validation failed: \(detail)")
        }

        emitArenaPostStatsLine(trainingBox: trainingBox, suffix: suffix)
    }

    /// One-line stats snapshot taken at the moment an arena ends.
    /// Captures rolling per-step trainer timing means + RSS + VM
    /// region count, so a reader scanning the session log can see
    /// whether each arena boundary corresponds to a step-up in
    /// per-step `gpu`/`step` time, RSS, or IOAccelerator-tagged
    /// VM region count. Emitted from every `emitArenaPostRunTelemetry`
    /// exit path (success, cancel, error) so a slowdown investigation
    /// has a per-arena trace independent of the 60 s [STATS] cadence.
    private func emitArenaPostStatsLine(
        trainingBox: TrainingLiveStatsBox?,
        suffix: String
    ) {
        let trainingSnap = trainingBox?.snapshot()
        let timingStr: String
        if let snap = trainingSnap, let stepMs = snap.recentStepMs {
            timingStr = String(
                format: "step=%.1f gpu=%.1f prep=%.2f read=%.2f wait=%.2f n=%d",
                stepMs,
                snap.recentGpuRunMs ?? 0,
                snap.recentDataPrepMs ?? 0,
                snap.recentReadbackMs ?? 0,
                snap.recentQueueWaitMs ?? 0,
                snap.recentTimingSamples
            )
        } else {
            timingStr = "n=0"
        }
        let rssBytes = DiagSampler.currentResidentBytes()
        let memStr = String(
            format: "rss=%.2fGB",
            Double(rssBytes) / 1024.0 / 1024.0 / 1024.0
        )
        let vm = DiagSampler.currentVMRegionCount()
        let vmStr = String(
            format: "total=%u ioAccel=%u",
            vm.total, vm.ioAccelerator
        )
        SessionLogger.shared.log(
            "[STATS-ARENA-END]\(suffix) timing=(\(timingStr)) mem=(\(memStr)) vm=(\(vmStr))"
        )
    }

    /// Emit one `[ARENA] timing` line for a single batcher. Wall /
    /// wait / run / other render in seconds for readability (arenas
    /// are typically tens of seconds, ms-formatting buries the
    /// signal under trailing zeros). Per-batch means stay in ms
    /// because they're sub-second by nature.
    private func emitArenaTimingLine(
        label: String,
        suffix: String,
        totalSec: Double,
        stats: BatchTimingStats
    ) {
        let waitSec = stats.totalWaitMs / 1000.0
        let runSec = stats.totalRunMs / 1000.0
        let otherSec = max(0, totalSec - waitSec - runSec)
        SessionLogger.shared.log(String(
            format: "[ARENA] timing %@%@: total=%.1fs wait=%.1fs run=%.1fs other=%.1fs (batches=%d meanWait=%.2fms meanRun=%.2fms)",
            label,
            suffix,
            totalSec,
            waitSec,
            runSec,
            otherSec,
            stats.totalBatches,
            stats.meanWaitMs,
            stats.meanRunMs
        ))
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
        chartCoordinator.cancelActiveArena()
        // Release any pending periodic-save fire that was held
        // back during the tournament. The controller decides on
        // the next `decide(now:)` call whether a (post-promotion)
        // successful save landed during the arena window (swallow
        // the pending fire) or not (dispatch the save a little
        // late).
        periodicSaveController?.noteArenaEnded()
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
                // `probeRunner` → the dedicated probe inference network.
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
            trainer.learningRate = Float(trainingParams.learningRate)
            trainer.entropyRegularizationCoeff = Float(trainingParams.entropyBonus)
            trainer.drawPenalty = Float(trainingParams.drawPenalty)
            trainer.weightDecayC = Float(trainingParams.weightDecay)
            trainer.gradClipMaxNorm = Float(trainingParams.gradClipMaxNorm)
            trainer.policyScaleK = Float(trainingParams.policyScaleK)
            trainer.momentumCoeff = Float(trainingParams.momentumCoeff)
            trainer.sqrtBatchScalingForLR = trainingParams.sqrtBatchScalingLR
            trainer.lrWarmupSteps = trainingParams.lrWarmupSteps
            trainer.batchStatsInterval = trainingParams.batchStatsInterval
            return trainer
        }
        do {
            let t = try ChessTrainer(
                learningRate: Float(trainingParams.learningRate),
                entropyRegularizationCoeff: Float(trainingParams.entropyBonus),
                drawPenalty: Float(trainingParams.drawPenalty),
                weightDecayC: Float(trainingParams.weightDecay),
                gradClipMaxNorm: Float(trainingParams.gradClipMaxNorm),
                policyScaleK: Float(trainingParams.policyScaleK),
                momentumCoeff: Float(trainingParams.momentumCoeff),
                sqrtBatchScalingForLR: trainingParams.sqrtBatchScalingLR,
                lrWarmupSteps: trainingParams.lrWarmupSteps
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
            startTau: Float(max(0.01, trainingParams.selfPlayStartTau)),
            decayPerPly: Float(max(0.0, trainingParams.selfPlayTauDecayPerPly)),
            floorTau: Float(max(0.01, trainingParams.selfPlayTargetTau)),
            dirichletNoise: SamplingSchedule.selfPlay.dirichletNoise
        )
    }

    /// Build a `SamplingSchedule` for arena play from the live
    /// `@AppStorage` tau values. Arena never applies Dirichlet noise
    /// (pure strength measurement).
    private func buildArenaSchedule() -> SamplingSchedule {
        SamplingSchedule(
            startTau: Float(max(0.01, trainingParams.arenaStartTau)),
            decayPerPly: Float(max(0.0, trainingParams.arenaTauDecayPerPly)),
            floorTau: Float(max(0.01, trainingParams.arenaTargetTau))
        )
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

    /// Surface a refusal to the user via the menu-action alert,
    /// and log it. Used by in-function guards so pressing a menu
    /// item whose disable state is stale (or invoked via keyboard
    /// shortcut under a race) produces an explanation rather than
    /// silent no-op. Also used for end-of-operation warnings like
    /// "champion was loaded since last training — trainer still
    /// has its prior weights".
    private func refuseMenuAction(_ reason: String) {
        SessionLogger.shared.log("[GUARD] refused: \(reason)")
        menuActionError = reason
    }

    /// Whether there is resumable in-memory state from a prior
    /// Stop in this launch. True iff a replay buffer exists AND
    /// there is no `pendingLoadedSession` queued (a disk load
    /// takes precedence and has its own resume semantics).
    private var hasResumableInMemorySession: Bool {
        pendingLoadedSession == nil && replayBuffer != nil
    }

    /// Message for the three-way Start dialog. Flags trainer/champion
    /// divergence if the user loaded a model (champion) since the
    /// last training session — "Continue" will keep the old trainer
    /// weights playing against the new champion, which is valid but
    /// worth calling out.
    private func startTrainingDialogMessage() -> String {
        var lines: [String] = []
        lines.append("Continue picks up where you stopped, keeping the replay buffer and counters.")
        lines.append("New session resets counters and clears the replay buffer.")
        if championLoadedSinceLastTrainingSegment {
            lines.append("")
            lines.append("Note: the champion was replaced (Load Model) since the last training run. " +
                         "If you pick \"Continue\", the trainer still has its pre-load weights.")
        }
        return lines.joined(separator: "\n")
    }

    /// True iff a Load Model has happened since the most recent
    /// training segment closed. Used only to annotate the
    /// three-way Start dialog.
    @State private var championLoadedSinceLastTrainingSegment: Bool = false

    /// Entry point for the "Play and Train" / "Continue Training"
    /// menu button. Performs in-function guards, then either
    /// starts training directly or raises the three-way
    /// confirmation dialog when in-memory state from a prior Stop
    /// is available.
    private func startTrainingFromMenu() {
        // In-function guards — belt-and-suspenders with the menu-item
        // disable. Any of these being wrong means the menu got out
        // of sync with view state; still better to refuse visibly
        // than to silently no-op or corrupt state.
        if isBusy && !realTraining {
            refuseMenuAction("Another operation is in progress. Wait for it to finish, then try again.")
            return
        }
        if realTraining {
            refuseMenuAction("Training is already running.")
            return
        }
        if continuousPlay || continuousTraining || sweepRunning {
            refuseMenuAction("Another continuous task is running. Stop it first.")
            return
        }
        if isBuilding {
            refuseMenuAction("The network is still being built. Wait for it to finish.")
            return
        }
        guard networkReady, network != nil else {
            refuseMenuAction("Build or load a model first.")
            return
        }
        if hasResumableInMemorySession {
            showStartTrainingDialog = true
        } else {
            startRealTraining(mode: .freshOrFromLoadedSession)
        }
    }

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
    ///
    /// `mode` picks how in-memory state from a prior Stop is handled:
    /// see `TrainingStartMode` for the four cases.
    private func startRealTraining(mode: TrainingStartMode = .freshOrFromLoadedSession) {
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
        // for user input but `trainingParams.selfPlayWorkers` is centrally managed
        // so the value could in principle be edited elsewhere.
        let initialWorkerCount = max(1, min(Self.absoluteMaxSelfPlayWorkers, trainingParams.selfPlayWorkers))
        if initialWorkerCount != trainingParams.selfPlayWorkers {
            trainingParams.selfPlayWorkers = initialWorkerCount
        }
        guard let trainer = ensureTrainer(), let network else { return }
        inferenceResult = nil
        gameWatcher.resetAll()
        gameSnapshot = gameWatcher.snapshot()
        clearTrainingAlarm()
        divergenceWarningStreak = 0
        divergenceCriticalStreak = 0
        divergenceRecoveryStreak = 0

        // `continueMode` controls whether we preserve in-memory state
        // from a prior Stop. When true: reuse replay buffer, stats
        // boxes, session ID, tournament history, chart samples, and
        // trainer weights. The only fresh objects are the transient
        // per-task gates and live schedule boxes (those get cancelled
        // with the task on Stop, so they don't survive).
        let continueMode = (mode == .continueAfterStop)
        if !continueMode {
            clearTrainingDisplay()
        }

        let buffer: ReplayBuffer
        if continueMode, let existing = replayBuffer {
            buffer = existing
        } else {
            buffer = ReplayBuffer(capacity: trainingParams.replayBufferCapacity)
            replayBuffer = buffer
        }
        let box: TrainingLiveStatsBox
        if continueMode, let existing = trainingBox {
            box = existing
        } else {
            let fresh = TrainingLiveStatsBox(rollingWindow: Self.rollingLossWindow)
            if let rs = pendingLoadedSession?.state {
                var seeded = TrainingRunStats()
                seeded.steps = rs.trainingSteps
                fresh.seed(seeded)
            }
            trainingBox = fresh
            realRollingPolicyLoss = nil
            realRollingValueLoss = nil
            box = fresh
        }
        // Seed counters from the loaded session if resuming, or
        // start fresh. This covers the stats box (game/move/result
        // counters), training step count, tournament history, and
        // worker count. In `.continueAfterStop` the trainer's
        // hyperparameters and `trainingStats` stay untouched — the
        // user wants to pick up exactly where they left off.
        let resumeState = pendingLoadedSession?.state
        if !continueMode {
            if let rs = resumeState {
                SessionLogger.shared.log(
                    "[RESUME-PARAM] learning_rate: \(trainingParams.learningRate) -> \(rs.learningRate) (from session)"
                )
                trainer.learningRate = rs.learningRate
                trainingParams.learningRate = Double(rs.learningRate)
                if let entropyCoeff = rs.entropyRegularizationCoeff {
                    SessionLogger.shared.log(
                        "[RESUME-PARAM] entropy_bonus: \(trainingParams.entropyBonus) -> \(entropyCoeff) (from session)"
                    )
                    trainer.entropyRegularizationCoeff = entropyCoeff
                    trainingParams.entropyBonus = Double(entropyCoeff)
                } else {
                    trainer.entropyRegularizationCoeff = Float(trainingParams.entropyBonus)
                    SessionLogger.shared.log(
                        "[RESUME-PARAM] entropy_bonus: saved=nil applied=\(trainingParams.entropyBonus) (defaulted)"
                    )
                }
                if let dp = rs.drawPenalty {
                    SessionLogger.shared.log(
                        "[RESUME-PARAM] draw_penalty: \(trainingParams.drawPenalty) -> \(dp) (from session)"
                    )
                    trainer.drawPenalty = dp
                    trainingParams.drawPenalty = Double(dp)
                } else {
                    trainer.drawPenalty = Float(trainingParams.drawPenalty)
                    SessionLogger.shared.log(
                        "[RESUME-PARAM] draw_penalty: saved=nil applied=\(trainingParams.drawPenalty) (defaulted)"
                    )
                }
                // Regularization knobs that became editable post-v1 session
                // files: hydrate when present, otherwise leave the current
                // @AppStorage-backed value alone but log the fallthrough
                // so a session saved before this field existed never
                // resumes silently under different defaults.
                if let wd = rs.weightDecayCoeff {
                    SessionLogger.shared.log(
                        "[RESUME-PARAM] weight_decay: \(trainingParams.weightDecay) -> \(wd) (from session)"
                    )
                    trainer.weightDecayC = wd
                    trainingParams.weightDecay = Double(wd)
                } else {
                    trainer.weightDecayC = Float(trainingParams.weightDecay)
                    SessionLogger.shared.log(
                        "[RESUME-PARAM] weight_decay: saved=nil applied=\(trainingParams.weightDecay) (defaulted)"
                    )
                }
                if let clip = rs.gradClipMaxNorm {
                    SessionLogger.shared.log(
                        "[RESUME-PARAM] grad_clip_max_norm: \(trainingParams.gradClipMaxNorm) -> \(clip) (from session)"
                    )
                    trainer.gradClipMaxNorm = clip
                    trainingParams.gradClipMaxNorm = Double(clip)
                } else {
                    trainer.gradClipMaxNorm = Float(trainingParams.gradClipMaxNorm)
                    SessionLogger.shared.log(
                        "[RESUME-PARAM] grad_clip_max_norm: saved=nil applied=\(trainingParams.gradClipMaxNorm) (defaulted)"
                    )
                }
                if let k = rs.policyScaleK {
                    SessionLogger.shared.log(
                        "[RESUME-PARAM] K: \(trainingParams.policyScaleK) -> \(k) (from session)"
                    )
                    trainer.policyScaleK = k
                    trainingParams.policyScaleK = Double(k)
                } else {
                    trainer.policyScaleK = Float(trainingParams.policyScaleK)
                    SessionLogger.shared.log(
                        "[RESUME-PARAM] K: saved=nil applied=\(trainingParams.policyScaleK) (defaulted)"
                    )
                }
                if let mu = rs.momentumCoeff {
                    SessionLogger.shared.log(
                        "[RESUME-PARAM] momentum_coeff: \(trainingParams.momentumCoeff) -> \(mu) (from session)"
                    )
                    trainer.momentumCoeff = mu
                    trainingParams.momentumCoeff = Double(mu)
                } else {
                    trainer.momentumCoeff = Float(trainingParams.momentumCoeff)
                    SessionLogger.shared.log(
                        "[RESUME-PARAM] momentum_coeff: saved=nil applied=\(trainingParams.momentumCoeff) (defaulted)"
                    )
                }
                // LR warmup length and sqrt-batch LR scaling are now
                // part of the session schema (Optional, for back-compat
                // with older `.dcmsession` files that pre-date the
                // expansion). When the saved value is present, we
                // restore it onto both the trainer and the @AppStorage
                // mirror so the UI shows what the session was running
                // with — not whatever the user's current global
                // preference happens to be. When absent (older session),
                // we fall through to the @AppStorage value as before.
                // The trainer's internal completed-step counter is
                // seeded from `trainingSteps` so warmup scaling resumes
                // mid-session instead of restarting from zero.
                //
                // Every applied parameter emits a `[RESUME-PARAM]`
                // log line — both the "from session" and the
                // "saved=nil applied=… (defaulted)" branches — so
                // resuming an older session under post-schema-
                // expansion defaults can never silently drift the
                // applied value away from what the session was
                // trained under.
                if let savedSqrt = rs.sqrtBatchScalingForLR {
                    SessionLogger.shared.log(
                        "[RESUME-PARAM] sqrt_batch_scaling_lr: \(trainingParams.sqrtBatchScalingLR) -> \(savedSqrt) (from session)"
                    )
                    trainer.sqrtBatchScalingForLR = savedSqrt
                    trainingParams.sqrtBatchScalingLR = savedSqrt
                } else {
                    trainer.sqrtBatchScalingForLR = trainingParams.sqrtBatchScalingLR
                    SessionLogger.shared.log(
                        "[RESUME-PARAM] sqrt_batch_scaling_lr: saved=nil applied=\(trainingParams.sqrtBatchScalingLR) (defaulted)"
                    )
                }
                if let savedWarmup = rs.lrWarmupSteps, savedWarmup >= 0 {
                    SessionLogger.shared.log(
                        "[RESUME-PARAM] lr_warmup_steps: \(trainingParams.lrWarmupSteps) -> \(savedWarmup) (from session)"
                    )
                    trainer.lrWarmupSteps = savedWarmup
                    trainingParams.lrWarmupSteps = savedWarmup
                } else {
                    trainer.lrWarmupSteps = trainingParams.lrWarmupSteps
                    SessionLogger.shared.log(
                        "[RESUME-PARAM] lr_warmup_steps: saved=nil applied=\(trainingParams.lrWarmupSteps) (defaulted)"
                    )
                }
                trainer.completedTrainSteps = rs.trainingSteps
                // Run-management knobs that previously lived only in
                // @AppStorage. Same Optional-with-fallback pattern but
                // logged on both branches: a saved=nil line when an
                // older session is resumed under post-`74839ee`
                // defaults makes the silent-fallback regression that
                // motivated this audit impossible.
                if let v = rs.replayBufferMinPositionsBeforeTraining, v >= 0 {
                    SessionLogger.shared.log(
                        "[RESUME-PARAM] replay_buffer_min_positions_before_training: \(trainingParams.replayBufferMinPositionsBeforeTraining) -> \(v) (from session)"
                    )
                    trainingParams.replayBufferMinPositionsBeforeTraining = v
                } else {
                    SessionLogger.shared.log(
                        "[RESUME-PARAM] replay_buffer_min_positions_before_training: saved=nil applied=\(trainingParams.replayBufferMinPositionsBeforeTraining) (defaulted)"
                    )
                }
                if let v = rs.arenaAutoIntervalSec, v > 0 {
                    SessionLogger.shared.log(
                        "[RESUME-PARAM] arena_auto_interval_sec: \(trainingParams.arenaAutoIntervalSec) -> \(v) (from session)"
                    )
                    trainingParams.arenaAutoIntervalSec = v
                } else {
                    SessionLogger.shared.log(
                        "[RESUME-PARAM] arena_auto_interval_sec: saved=nil applied=\(trainingParams.arenaAutoIntervalSec) (defaulted)"
                    )
                }
                if let v = rs.arenaConcurrency, v >= 1 {
                    let clamped = min(Self.absoluteMaxArenaConcurrency, v)
                    SessionLogger.shared.log(
                        "[RESUME-PARAM] arena_concurrency: \(trainingParams.arenaConcurrency) -> \(clamped) (from session)"
                    )
                    trainingParams.arenaConcurrency = clamped
                } else {
                    SessionLogger.shared.log(
                        "[RESUME-PARAM] arena_concurrency: saved=nil applied=\(trainingParams.arenaConcurrency) (defaulted)"
                    )
                }
                if let v = rs.candidateProbeIntervalSec, v > 0 {
                    SessionLogger.shared.log(
                        "[RESUME-PARAM] candidate_probe_interval_sec: \(trainingParams.candidateProbeIntervalSec) -> \(v) (from session)"
                    )
                    trainingParams.candidateProbeIntervalSec = v
                } else {
                    SessionLogger.shared.log(
                        "[RESUME-PARAM] candidate_probe_interval_sec: saved=nil applied=\(trainingParams.candidateProbeIntervalSec) (defaulted)"
                    )
                }
                if let v = rs.legalMassCollapseThreshold, v > 0, v < 1 {
                    SessionLogger.shared.log(
                        "[RESUME-PARAM] legal_mass_collapse_threshold: \(trainingParams.legalMassCollapseThreshold) -> \(v) (from session)"
                    )
                    trainingParams.legalMassCollapseThreshold = v
                } else {
                    SessionLogger.shared.log(
                        "[RESUME-PARAM] legal_mass_collapse_threshold: saved=nil applied=\(trainingParams.legalMassCollapseThreshold) (defaulted)"
                    )
                }
                if let v = rs.legalMassCollapseGraceSeconds, v >= 0 {
                    SessionLogger.shared.log(
                        "[RESUME-PARAM] legal_mass_collapse_grace_seconds: \(trainingParams.legalMassCollapseGraceSeconds) -> \(v) (from session)"
                    )
                    trainingParams.legalMassCollapseGraceSeconds = v
                } else {
                    SessionLogger.shared.log(
                        "[RESUME-PARAM] legal_mass_collapse_grace_seconds: saved=nil applied=\(trainingParams.legalMassCollapseGraceSeconds) (defaulted)"
                    )
                }
                if let v = rs.legalMassCollapseNoImprovementProbes, v >= 1 {
                    SessionLogger.shared.log(
                        "[RESUME-PARAM] legal_mass_collapse_no_improvement_probes: \(trainingParams.legalMassCollapseNoImprovementProbes) -> \(v) (from session)"
                    )
                    trainingParams.legalMassCollapseNoImprovementProbes = v
                } else {
                    SessionLogger.shared.log(
                        "[RESUME-PARAM] legal_mass_collapse_no_improvement_probes: saved=nil applied=\(trainingParams.legalMassCollapseNoImprovementProbes) (defaulted)"
                    )
                }
                // Sampling schedule — TauConfigCodable is non-Optional on
                // the session schema (added in v1), so no fallback branch.
                // Writes to @AppStorage propagate through
                // `buildSelfPlaySchedule` / `buildArenaSchedule` the next
                // time the schedule box is built below.
                trainingParams.selfPlayStartTau = Double(rs.selfPlayTau.startTau)
                trainingParams.selfPlayTargetTau = Double(rs.selfPlayTau.floorTau)
                trainingParams.selfPlayTauDecayPerPly = Double(rs.selfPlayTau.decayPerPly)
                trainingParams.arenaStartTau = Double(rs.arenaTau.startTau)
                trainingParams.arenaTargetTau = Double(rs.arenaTau.floorTau)
                trainingParams.arenaTauDecayPerPly = Double(rs.arenaTau.decayPerPly)
            } else {
                trainer.learningRate = Float(trainingParams.learningRate)
                trainer.entropyRegularizationCoeff = Float(trainingParams.entropyBonus)
                trainer.drawPenalty = Float(trainingParams.drawPenalty)
                trainer.weightDecayC = Float(trainingParams.weightDecay)
                trainer.gradClipMaxNorm = Float(trainingParams.gradClipMaxNorm)
                trainer.policyScaleK = Float(trainingParams.policyScaleK)
                trainer.momentumCoeff = Float(trainingParams.momentumCoeff)
                trainer.sqrtBatchScalingForLR = trainingParams.sqrtBatchScalingLR
                trainer.lrWarmupSteps = trainingParams.lrWarmupSteps
            }
            var initialTrainingStats = TrainingRunStats()
            if let rs = resumeState {
                initialTrainingStats.steps = rs.trainingSteps
            }
            trainingStats = initialTrainingStats
        }
        // Game-run mode is meaningless with N > 1 self-play workers
        // (the live board hides itself behind a "N concurrent games"
        // overlay) and the Game-run / Candidate-test picker is hidden
        // in that case anyway. Default the mode to Candidate test in
        // multi-worker sessions so the user gets a usable left-side
        // panel out of the gate; single-worker sessions keep the
        // historical Game-run default.
        playAndTrainBoardMode = trainingParams.selfPlayWorkers > 1 ? .candidateTest : .gameRun
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
        spStartTauEditText = String(format: "%.2f", trainingParams.selfPlayStartTau)
        spFloorTauEditText = String(format: "%.2f", trainingParams.selfPlayTargetTau)
        spDecayPerPlyEditText = String(format: "%.3f", trainingParams.selfPlayTauDecayPerPly)
        arStartTauEditText = String(format: "%.2f", trainingParams.arenaStartTau)
        arFloorTauEditText = String(format: "%.2f", trainingParams.arenaTargetTau)
        arDecayPerPlyEditText = String(format: "%.3f", trainingParams.arenaTauDecayPerPly)
        if continueMode {
            // Preserve `completedTrainingSegments` and
            // `tournamentHistory` as they stood at Stop. The new
            // training segment has already been opened by
            // `beginActiveTrainingSegment` above.
        } else if let rs = resumeState {
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
                    finishedAt: entry.finishedAtUnix.map {
                        Date(timeIntervalSince1970: TimeInterval($0))
                    },
                    candidateID: entry.candidateID.map { ModelID(value: $0) },
                    championID: entry.championID.map { ModelID(value: $0) },
                    gamesPlayed: gp,
                    candidateWins: entry.candidateWins,
                    championWins: entry.championWins,
                    draws: entry.draws,
                    score: entry.score,
                    promoted: entry.promoted,
                    promotionKind: kind,
                    promotedID: entry.promotedID.map { ModelID(value: $0) },
                    durationSec: entry.durationSec,
                    candidateWinsAsWhite: entry.candidateWinsAsWhite ?? 0,
                    candidateWinsAsBlack: entry.candidateWinsAsBlack ?? 0,
                    candidateLossesAsWhite: entry.candidateLossesAsWhite ?? 0,
                    candidateLossesAsBlack: entry.candidateLossesAsBlack ?? 0,
                    candidateDrawsAsWhite: entry.candidateDrawsAsWhite ?? 0,
                    candidateDrawsAsBlack: entry.candidateDrawsAsBlack ?? 0
                )
            }
        } else {
            // Fresh session (first Start of the launch, or one of
            // the new-session modes from the Start dialog). Clear
            // both the arena history and the training-segment
            // wall-time history — "new session" means fresh
            // counters. Pre-existing first-launch behavior left
            // `completedTrainingSegments` alone (benign, since it
            // was already empty at first launch), but after a
            // Stop→"New session" pick the old value would bleed
            // through, so normalize to "always fresh" here.
            tournamentHistory = []
            completedTrainingSegments = []
        }
        tournamentProgress = nil
        let tBox = TournamentLiveBox()
        tournamentBox = tBox
        let pStatsBox: ParallelWorkerStatsBox
        if continueMode, let existing = parallelWorkerStatsBox {
            pStatsBox = existing
            // continueMode preserves the prior box's sessionStart, so
            // the segment baseline must stay where it was — don't
            // reset it here, otherwise the very next Stop/Start cycle
            // would re-zero a still-open segment's rate.
        } else if let rs = resumeState {
            // Resumed session: trainer was seeded with `rs.trainingSteps`
            // (see `liveStatsBox.seed(seeded)` in `runRealTraining`),
            // so the rate numerator must subtract that as the
            // segment-start baseline to express this-segment steps
            // over this-segment wall time.
            trainingStepsAtSegmentStart = rs.trainingSteps
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
                trainingParams.selfPlayWorkers = max(1, min(Self.absoluteMaxSelfPlayWorkers, workerCount))
            }
            if let delay = rs.stepDelayMs {
                trainingParams.trainingStepDelayMs = delay
            }
            if let autoDelay = rs.lastAutoComputedDelayMs {
                lastAutoComputedDelayMs = autoDelay
            }
        } else {
            // Fresh session — no resumed steps to subtract.
            trainingStepsAtSegmentStart = 0
            pStatsBox = ParallelWorkerStatsBox(sessionStart: Date())
        }
        parallelWorkerStatsBox = pStatsBox
        parallelStats = pStatsBox.snapshot()
        let spDiversityTracker: GameDiversityTracker
        if continueMode, let existing = selfPlayDiversityTracker {
            spDiversityTracker = existing
        } else {
            spDiversityTracker = GameDiversityTracker(windowSize: 200)
            selfPlayDiversityTracker = spDiversityTracker
            chartCoordinator.setDiversityHistogramBars([])
        }
        if !continueMode {
            // Fresh session — wipe every chart-layer field back to a
            // zero state so the new session's chart starts at t=0
            // and doesn't show a visible "step" from the previous
            // session's trailing values.
            chartCoordinator.reset()
        }
        // Single self-play gate. All self-play workers now share one
        // `BatchedMoveEvaluationSource` on the champion network, driven
        // by `BatchedSelfPlayDriver`, so there is exactly one consumer
        // for the arena coordinator to pause. The previous per-secondary
        // gate array is gone along with the secondary networks.
        let selfPlayGate = WorkerPauseGate()
        // Shared current-N holder. Workers poll this to decide
        // whether to play another game or sit in their idle wait.
        // The Stepper writes through it (and to `@State
        // trainingParams.selfPlayWorkers simultaneously). Exposed via
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
        let ratioController = ReplayRatioController(
            batchSize: trainingParams.trainingBatchSize,
            targetRatio: trainingParams.replayRatioTarget,
            autoAdjust: trainingParams.replayRatioAutoAdjust,
            initialDelayMs: trainingParams.replayRatioAutoAdjust
            ? lastAutoComputedDelayMs
            : trainingParams.trainingStepDelayMs,
            maxTrainingStepDelayMs: Self.stepDelayMaxMs,
            maxSelfPlayDelayMs: Self.selfPlayDelayMaxMs
        )
        // Seed the controller's manual SP-delay slot from the
        // persisted training parameter so a session that starts in
        // manual mode inherits whatever the user last left in the
        // SP-delay stepper, instead of falling back to 0.
        ratioController.manualSelfPlayDelayMs = trainingParams.selfPlayDelayMs
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
        // Arm the 4-hour periodic-save scheduler. Always construct
        // a fresh controller on each start — a previous stop will
        // have nil'd it out, and `.continueAfterStop` intentionally
        // resets the next-save deadline to a full interval from
        // now rather than inheriting whatever time was left on the
        // prior arm, since the whole point of the scheduler is
        // "saved within the last 4 hours" and the session was not
        // being saved while stopped.
        let controller = PeriodicSaveController(interval: Self.periodicSaveIntervalSec)
        controller.arm(now: Date())
        periodicSaveController = controller
        periodicSaveLastPollAt = Date()
        periodicSaveInFlight = false

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
        if continueMode {
            // Preserve `currentSessionID`, `currentSessionStart`,
            // `replayRatioTarget`, `replayRatioAutoAdjust`, and any
            // trainer-hyperparameter edits the user made between
            // Stop and Start. Nothing to do here.
        } else if let resumed = pendingLoadedSession {
            currentSessionID = resumed.state.sessionID
            currentSessionStart = Date().addingTimeInterval(-resumed.state.elapsedTrainingSec)
            trainingParams.replayRatioTarget = resumed.state.replayRatioTarget ?? 1.0
            trainingParams.replayRatioAutoAdjust = resumed.state.replayRatioAutoAdjust ?? true
            trainingParams.learningRate = Double(resumed.state.learningRate)
            if let entropyCoeff = resumed.state.entropyRegularizationCoeff {
                trainingParams.entropyBonus = Double(entropyCoeff)
            }
        } else {
            currentSessionID = ModelIDMinter.mint().value
            currentSessionStart = Date()
        }

        // CLI capture mode: allocate the recorder before the
        // training Task starts so every downstream capture site
        // (stats logger, arena, candidate probe) can append to it.
        // Allocation fires when `--train` is on (headless CLI run,
        // with a potential `training_time_limit` that needs a JSON
        // artifact at expiry) OR when `--output <file>` was
        // supplied (user wants artifact regardless of train mode).
        // The recorder is `final class @unchecked Sendable` so we
        // can capture the reference into each child task without
        // wrestling MainActor isolation at each append. Cleared in
        // the teardown block along with the other per-session state.
        let recorder: CliTrainingRecorder?
        if cliOutputURL != nil || autoTrainOnLaunch {
            let r = CliTrainingRecorder()
            r.setSessionID(currentSessionID)
            recorder = r
            cliRecorder = r
        } else {
            recorder = nil
        }
        let outputURL = cliOutputURL
        let cliTrainingTimeLimitSec = cliConfig?.trainingTimeLimitSec
        let isAutoTrainRun = autoTrainOnLaunch
        let runStart = Date()

        // Register the early-stop flush handler so SIGUSR1 / SIGHUP /
        // applicationShouldTerminate can write `result.json` cleanly
        // before exiting. Cleared in the teardown block. The closure
        // captures the recorder, outputURL, and runStart so the
        // coordinator doesn't need to know about ContentView's state
        // shape — it just calls the closure with the termination reason.
        if let recorder {
            EarlyStopCoordinator.shared.earlyStopHandler = { reason in
                let elapsed = Date().timeIntervalSince(runStart)
                let destDescription = outputURL?.path ?? "<stdout>"
                SessionLogger.shared.log(
                    "[APP] --train: early-stop on \(reason.rawValue) at elapsed=\(String(format: "%.1f", elapsed))s; writing snapshot to \(destDescription)"
                )
                recorder.setTerminationReason(reason)
                let counts = recorder.countsSnapshot()
                do {
                    if let url = outputURL {
                        try recorder.writeJSON(to: url, totalTrainingSeconds: elapsed)
                    } else {
                        try recorder.writeJSONToStdout(totalTrainingSeconds: elapsed)
                    }
                    SessionLogger.shared.log(
                        "[APP] --train: wrote snapshot to \(destDescription) (arenas=\(counts.arenas), stats=\(counts.stats), probes=\(counts.probes))"
                    )
                } catch {
                    SessionLogger.shared.log(
                        "[APP] --train: early-stop snapshot write FAILED for \(destDescription): \(error.localizedDescription)"
                    )
                }
                SessionLogger.shared.log("[APP] --train: exiting process after early-stop snapshot")
                Darwin._exit(0)
            }
        }

        // Snapshot the CLI-overridable effective values into plain
        // `let`s so the detached Task bodies below can read them
        // without a MainActor hop. These values don't change after
        // session start in the current model, so a one-time capture
        // is safe and keeps the Task-side code MainActor-free.
        let sessionTrainingBatchSize = trainingParams.trainingBatchSize
        let sessionMinBufferBeforeTraining = trainingParams.replayBufferMinPositionsBeforeTraining
        let sessionTournamentGames = trainingParams.arenaGamesPerTournament
        let sessionPromoteThreshold = trainingParams.arenaPromoteThreshold

        realTrainingTask = Task(priority: .high) {
            [trainer, network, buffer, box, tBox, pStatsBox, spDiversityTracker,
             selfPlayGate, trainingGate, arenaFlag, triggerBox, overrideBox, countBox,
             gameWatcher, ratioController, recorder, outputURL, cliTrainingTimeLimitSec,
             isAutoTrainRun,
             sessionTrainingBatchSize, sessionMinBufferBeforeTraining,
             sessionTournamentGames, sessionPromoteThreshold] in

            // --- Setup: build any missing networks, reset the trainer ---

            let needsCandidateBuild = await MainActor.run { candidateInferenceNetwork == nil }
            if needsCandidateBuild {
                do {
                    let built = try await Task.detached(priority: .userInitiated) {
                        try ChessMPSNetwork(.randomWeights)
                    }.value
                    built.network.commandQueue.label = "startrealTraining candidate Inference"
                    await MainActor.run {
                        candidateInferenceNetwork = built
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

            let needsProbeBuild = await MainActor.run { probeInferenceNetwork == nil }
            if needsProbeBuild {
                do {
                    let built = try await Task.detached(priority: .userInitiated) {
                        try ChessMPSNetwork(.randomWeights)
                    }.value
                    built.network.commandQueue.label = "startrealTraining probe Inference"
                    await MainActor.run {
                        probeInferenceNetwork = built
                        probeRunner = ChessRunner(network: built)
                    }
                } catch {
                    box.recordError("Probe network init failed: \(error.localizedDescription)")
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
                    built.network.commandQueue.label = "startrealTraining arena champion"
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

            // Reset the trainer's graph AND initialize its weights,
            // branched by `mode`:
            //
            // - `.freshOrFromLoadedSession`: rebuild graph, then
            //   load trainer weights from the loaded session's
            //   `trainer.dcmmodel` if present, or fork them from
            //   the champion. This is the default first-launch and
            //   disk-resume path.
            // - `.newSessionResetTrainerFromChampion`: rebuild
            //   graph, fork trainer weights from the live champion.
            //   User explicitly asked to throw away accumulated
            //   trainer weights and start fresh from champion.
            // - `.continueAfterStop` and `.newSessionKeepTrainer`:
            //   leave the trainer's graph and weights untouched —
            //   the trainer's MPSGraph stayed alive across Stop,
            //   and preserving its weights is the whole point of
            //   both modes.
            //
            // In the fork-from-champion branch, champion weights
            // loaded from disk at file-load time are already live
            // in `network`, so the batcher (which wraps `network`)
            // picks up the restored weights automatically.
            let resumedTrainerWeights: [[Float]]? = await MainActor.run {
                pendingLoadedSession?.trainerFile.weights
            }
            let resumedBufferURL: URL? = await MainActor.run {
                pendingLoadedSession?.replayBufferURL
            }
            do {
                switch mode {
                case .continueAfterStop, .newSessionKeepTrainer:
                    break
                case .freshOrFromLoadedSession:
                    try await trainer.resetNetwork()
                    try await Task.detached(priority: .userInitiated) {
                        [resumedTrainerWeights] in
                        if let trainerWeights = resumedTrainerWeights {
                            // loadTrainerWeights detects v1 vs v2 layout
                            // by count: v2 includes velocity tensors,
                            // v1 (legacy) is trainables+bn only and
                            // leaves velocity at zero-init.
                            try await trainer.loadTrainerWeights(trainerWeights)
                        } else {
                            // No prior trainer file (fresh session or
                            // pre-existing session without trainer.dcmmodel):
                            // fork from champion. Champion has no
                            // velocities, so loadTrainerWeights takes the
                            // v1 branch and velocity stays at zero-init.
                            let championWeights = try await network.exportWeights()
                            try await trainer.loadTrainerWeights(championWeights)
                        }
                    }.value
                case .newSessionResetTrainerFromChampion:
                    try await trainer.resetNetwork()
                    try await Task.detached(priority: .userInitiated) {
                        // User explicitly asked to discard trainer state
                        // and re-fork from champion. Velocity goes back
                        // to zero (via the v1-count branch in
                        // loadTrainerWeights), which is the correct
                        // semantics for a fresh fork.
                        let championWeights = try await network.exportWeights()
                        try await trainer.loadTrainerWeights(championWeights)
                    }.value
                }
            } catch {
                box.recordError("Reset failed: \(error.localizedDescription)")
                await MainActor.run {
                    realTraining = false
                    realTrainingTask = nil
                }
                return
            }

            // Dump every live MTLCommandQueue's (label, address) pair
            // so subsequent Metal-trace captures can be matched back
            // to the network they came from. The address printed here
            // is what Xcode's Metal frame-capture UI shows under
            // "Command Queues" on the trace selection sheet — pick by
            // label, then verify by address. Emitted once per session
            // start; queues persist for the app lifetime so this is
            // accurate for every later capture in the same launch.
            do {
                let candidateQ = await MainActor.run {
                    candidateInferenceNetwork?.network.commandQueue
                }
                let probeQ = await MainActor.run {
                    probeInferenceNetwork?.network.commandQueue
                }
                let arenaChampionQ = await MainActor.run {
                    arenaChampionNetwork?.network.commandQueue
                }
                let championQ = network.network.commandQueue
                let trainerQ = trainer.network.commandQueue
                func desc(_ label: String, _ q: MTLCommandQueue?) -> String {
                    guard let q else { return "\(label)=<not built>" }
                    // Pointer that matches the address Xcode's Metal
                    // frame-capture UI shows in the queue selector.
                    let opaque = Unmanaged.passUnretained(q as AnyObject).toOpaque()
                    let addr = String(format: "0x%016lx", UInt(bitPattern: opaque))
                    return "\(label)=\(q.label ?? "<no-label>") @ \(addr)"
                }
                SessionLogger.shared.log(
                    "[QUEUES] champion: \(desc("champion", championQ)); "
                    + "trainer: \(desc("trainer", trainerQ)); "
                    + "candidate: \(desc("candidate", candidateQ)); "
                    + "probe: \(desc("probe", probeQ)); "
                    + "arenaChampion: \(desc("arenaChampion", arenaChampionQ))"
                )
            }

            // Restore replay buffer before any self-play worker or
            // the training worker starts — training samples from the
            // buffer, and any worker appends after restore resets
            // would either be clobbered or (worse) race with the
            // restore's counter reset. The restore runs on a
            // detached I/O task so ~GB-scale reads don't block the
            // cooperative hop cadence.
            if let bufferURL = resumedBufferURL {
                let resumedState: SessionCheckpointState? = await MainActor.run {
                    pendingLoadedSession?.state
                }
                do {
                    try await Task.detached(priority: .userInitiated) {
                        [buffer, bufferURL] in
                        try buffer.restore(from: bufferURL)
                    }.value
                    // Cross-check lifetime counter against session.json.
                    // Mismatch here indicates file-pairing error or
                    // residual corruption that happened to SHA-match.
                    if let resumedState {
                        try CheckpointManager.verifyReplayBufferMatchesSession(
                            buffer: buffer,
                            state: resumedState
                        )
                    }
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

            // Trainer ID — branched by `mode`, matching the
            // trainer-weights logic above:
            //
            // - `.freshOrFromLoadedSession`: inherit from the
            //   loaded session's trainer file if present, else
            //   mint a new generation off the champion.
            // - `.newSessionResetTrainerFromChampion`: mint a new
            //   generation off the current champion (trainer
            //   weights were just forked from champion).
            // - `.continueAfterStop` and `.newSessionKeepTrainer`:
            //   keep the trainer's existing ID — its weights
            //   weren't touched, so the lineage is continuous.
            await MainActor.run {
                switch mode {
                case .continueAfterStop, .newSessionKeepTrainer:
                    break
                case .newSessionResetTrainerFromChampion:
                    trainer.identifier = ModelIDMinter.mintTrainerGeneration(
                        from: network.identifier ?? ModelIDMinter.mint()
                    )
                case .freshOrFromLoadedSession:
                    if let resumed = pendingLoadedSession {
                        trainer.identifier = ModelID(value: resumed.trainerFile.modelID)
                    } else {
                        trainer.identifier = ModelIDMinter.mintTrainerGeneration(
                            from: network.identifier ?? ModelIDMinter.mint()
                        )
                    }
                }
                // Consume the pending load — from here on, the
                // running session owns the restored state.
                pendingLoadedSession = nil
                // A training segment has started, so clear the
                // "champion replaced since last training" flag
                // (the Start dialog's annotation is resolved).
                championLoadedSinceLastTrainingSegment = false
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
            // Wire the ratio controller into the batcher so every
            // barrier fire reports an aggregate (positions, elapsed)
            // measurement. `setReplayRatioController` is actor-
            // isolated, so it's dispatched via a `Task`.
            Task { [ratioController] in
                await selfPlayBatcher.setReplayRatioController(ratioController)
            }
            let selfPlayDriver = BatchedSelfPlayDriver(
                batcher: selfPlayBatcher,
                buffer: buffer,
                statsBox: pStatsBox,
                diversityTracker: spDiversityTracker,
                countBox: countBox,
                pauseGate: selfPlayGate,
                gameWatcher: gameWatcher,
                scheduleBox: scheduleBox,
                replayRatioController: ratioController
            )

            // Pin the probe inference network into a local the child
            // tasks can capture. It's a @State on this view, so
            // reading it requires a MainActor hop that child tasks
            // would otherwise repeat. Built lazily earlier in this
            // function (lines ~7694-7703); should be non-nil here but
            // we tolerate nil and skip probes in that case rather
            // than fall back to the pollution-prone trainer-network
            // probe path.
            let probeInferenceForProbes: ChessMPSNetwork? = await MainActor.run {
                probeInferenceNetwork
            }

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
                group.addTask(priority: .high) {
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
                group.addTask(priority: .high) {
                    [trainer, buffer, box, pStatsBox, trainingGate, triggerBox, ratioController,
                     sessionTrainingBatchSize, sessionMinBufferBeforeTraining] in
                    // Track the previous step's applied delay so the
                    // next `recordTrainingBatchAndGetDelay` can report
                    // it as the current per-batch training-side delay
                    // setting. The controller owns the inter-call
                    // wall-clock measurement directly; the caller
                    // just reports its current configured delay.
                    var lastTrainingDelaySettingMs: Int = 0
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
                        if buffer.count < sessionMinBufferBeforeTraining {
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
                                batchSize: sessionTrainingBatchSize
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

                        // Auto-trigger the arena on the configured
                        // cadence. Fires the trigger inbox; the arena
                        // coordinator task picks it up and runs the
                        // tournament. Reads `arenaAutoIntervalSec` live
                        // (the parameter is `liveTunable: true`) so the
                        // status-bar Arena popover's Save edits take
                        // effect on the next poll without restarting
                        // the session.
                        let liveInterval = await TrainingParameters.shared.snapshot().arenaAutoIntervalSec
                        if triggerBox.shouldAutoTrigger(interval: liveInterval) {
                            triggerBox.trigger()
                        }

                        // Post-step pause. Symmetric API with the
                        // self-play barrier tick: caller reports its
                        // currently-applied per-batch delay setting,
                        // controller owns the wall-clock and does
                        // the subtraction. The "current setting"
                        // we report is whatever sleep we just
                        // applied on the previous loop pass — that
                        // IS the configured delay during the
                        // controller-owned inter-call period that
                        // ends right now.
                        let stepDelayMs = ratioController.recordTrainingBatchAndGetDelay(
                            currentDelaySettingMs: Double(lastTrainingDelaySettingMs)
                        )
                        lastTrainingDelaySettingMs = stepDelayMs
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
                group.addTask(priority: .utility) {
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
                    [trainer, network, box, pStatsBox, buffer, spDiversityTracker, ratioController, countBox, scheduleBox, recorder,
                     sessionTrainingBatchSize, sessionTournamentGames, sessionPromoteThreshold, probeInferenceForProbes] in
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

                    // First-observed pwNorm becomes the session baseline
                    // so each [STATS] line can report the absolute value
                    // alongside the drift since session start. Captured
                    // lazily on the first non-nil reading rather than at
                    // task launch — when training has just kicked off
                    // the trainer's rolling weight-norm window may not
                    // be populated yet. Mutated from inside the nested
                    // `logOne` closure; safe because every call site
                    // runs serially on this task.
                    var pwNormBaseline: Double? = nil

                    // Carried across [STATS] emits so each line can
                    // report the delta since the previous tick. nil
                    // until the first sample lands; the very first emit
                    // shows `drss=+0` etc., which is fine — the deltas
                    // are only meaningful from the second tick onward.
                    var prevRssBytes: UInt64 = 0
                    var prevVmTotal: UInt32 = 0
                    var prevVmIoAccel: UInt32 = 0

                    func logOne(elapsedTarget: TimeInterval, legalMassOverride: ChessTrainer.LegalMassSnapshot?) async {
                        let trainingSnap = box.snapshot()
                        let parallelSnap = pStatsBox.snapshot()
                        let bufCount = buffer.count
                        let bufCap = buffer.capacity
                        let ratioSnap = ratioController.snapshot()
                        let workerN = countBox.count
                        let spSched = scheduleBox.selfPlay
                        let arSched = scheduleBox.arena
                        let (trainerID, championID, lr, entropyCoeff, drawPen, weightDec, gradClip, kScale, momentum, sqrtLR, warmupSteps, completedSteps) = await MainActor.run {
                            (
                                trainer.identifier?.description ?? "?",
                                network.identifier?.description ?? "?",
                                trainer.learningRate,
                                trainer.entropyRegularizationCoeff,
                                trainer.drawPenalty,
                                trainer.weightDecayC,
                                trainer.gradClipMaxNorm,
                                trainer.policyScaleK,
                                trainer.momentumCoeff,
                                trainer.sqrtBatchScalingForLR,
                                trainer.lrWarmupSteps,
                                trainer.completedTrainSteps
                            )
                        }
                        let policyStr: String
                        if let p = trainingSnap.rollingPolicyLoss {
                            policyStr = String(format: "%+.4f", p)
                        } else {
                            policyStr = "--"
                        }
                        let pLossWinStr: String
                        if let p = trainingSnap.rollingPolicyLossWin {
                            pLossWinStr = String(format: "%+.4f", p)
                        } else {
                            pLossWinStr = "--"
                        }
                        let pLossLossStr: String
                        if let p = trainingSnap.rollingPolicyLossLoss {
                            pLossLossStr = String(format: "%+.4f", p)
                        } else {
                            pLossLossStr = "--"
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
                        let vNormStr: String
                        if let vn = trainingSnap.rollingVelocityNorm {
                            vNormStr = String(format: "%.3f", vn)
                        } else {
                            vNormStr = "--"
                        }
                        let muStr = String(format: "%.3f", momentum)
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
                            if pwNormBaseline == nil {
                                pwNormBaseline = pwn
                            }
                            let delta = pwn - (pwNormBaseline ?? pwn)
                            pwNormStr = String(format: "%.5f(Δ%+.5f)", pwn, delta)
                        } else {
                            pwNormStr = "--"
                        }
                        let divSnap = spDiversityTracker.snapshot()
                        let divStr = divSnap.gamesInWindow > 0
                        ? String(format: "unique=%d/%d(%.0f%%) diverge=%.1f", divSnap.uniqueGames, divSnap.gamesInWindow, divSnap.uniquePercent, divSnap.avgDivergencePly)
                        : "n/a"
                        // Append a `·√b` marker when sqrt-batch scaling is on
                        // and a `·warmup(i/N)` marker while warmup is still
                        // active, so the reader can tell at a glance what
                        // per-step multipliers the optimizer is actually
                        // seeing. The displayed LR is always the base value.
                        var lrStr = String(format: "%.1e", lr) + (sqrtLR ? "·√b" : "")
                        if warmupSteps > 0 && completedSteps < warmupSteps {
                            lrStr += "·warmup(\(completedSteps)/\(warmupSteps))"
                        }
                        let spMsStr: String
                        if let sp = ratioSnap.selfPlayMsPerMove {
                            spMsStr = String(format: "%.2f", sp)
                        } else {
                            spMsStr = "--"
                        }
                        let trMsStr: String
                        if let tr = ratioSnap.trainingMsPerMove {
                            trMsStr = String(format: "%.2f", tr)
                        } else {
                            trMsStr = "--"
                        }
                        // moves/hour companions to prod/cons (which are
                        // pos/sec). Same rolling 60-s window — these are
                        // a pure unit conversion, exposed here so the
                        // [STATS] line and result.json carry the rate in
                        // the unit a human asks for ("how many moves per
                        // hour are we training on?").
                        let spMovesPerHour = ratioSnap.productionRate * 3600.0
                        let trainMovesPerHour = ratioSnap.consumptionRate * 3600.0
                        let ratioStr = String(format: "target=%.2f cur=%.2f prod=%.1f cons=%.1f spRate=%.0f/hr trainRate=%.0f/hr auto=%@ delay=%dms spDelay=%dms spMs=%@ trMs=%@ workers=%d",
                                              ratioSnap.targetRatio, ratioSnap.currentRatio,
                                              ratioSnap.productionRate, ratioSnap.consumptionRate,
                                              spMovesPerHour, trainMovesPerHour,
                                              ratioSnap.autoAdjust ? "on" : "off",
                                              ratioSnap.computedDelayMs, ratioSnap.computedSelfPlayDelayMs,
                                              spMsStr, trMsStr, ratioSnap.workerCount)
                        let outcomeStr = String(format: "wMate=%d bMate=%d stale=%d 50mv=%d 3fold=%d insuf=%d",
                                                parallelSnap.whiteCheckmates, parallelSnap.blackCheckmates,
                                                parallelSnap.stalemates, parallelSnap.fiftyMoveDraws,
                                                parallelSnap.threefoldRepetitionDraws, parallelSnap.insufficientMaterialDraws)
                        let cfgStr = "batch=\(sessionTrainingBatchSize) lr=\(lrStr) promote>=\(String(format: "%.2f", sessionPromoteThreshold)) arenaGames=\(sessionTournamentGames) workers=\(workerN)"
                        let regStr = String(
                            format: "clip=%.1f decay=%.0e ent=%.1e drawPen=%.3f K=%.2f μ=%.2f",
                            gradClip,
                            weightDec,
                            entropyCoeff,
                            drawPen,
                            kScale,
                            momentum
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
                        // Advantage-sign-conditional played-move
                        // probabilities. Under a working training loop
                        // `posAdv` rises above `1/policySize` and
                        // `negAdv` falls — divergence between the two
                        // is the real action-index/direction-of-
                        // learning signal (the unconditional
                        // `playedMoveProb` is ambiguous under adv-
                        // normalized loss). The `skip=K/N` suffix
                        // reports how many steps in the rolling window
                        // dropped out because their batch had no
                        // positions on that side of A — that's the
                        // effective sample count behind the mean.
                        let condWindowSize = trainingSnap.rollingPlayedMoveCondWindowSize
                        let playedProbPosStr: String
                        if let pm = trainingSnap.rollingPlayedMoveProbPosAdv {
                            playedProbPosStr = String(
                                format: "%.4f(skip=%d/%d)",
                                pm,
                                trainingSnap.rollingPlayedMoveProbPosAdvSkipped,
                                condWindowSize
                            )
                        } else {
                            playedProbPosStr = String(
                                format: "--(skip=%d/%d)",
                                trainingSnap.rollingPlayedMoveProbPosAdvSkipped,
                                condWindowSize
                            )
                        }
                        let playedProbNegStr: String
                        if let pm = trainingSnap.rollingPlayedMoveProbNegAdv {
                            playedProbNegStr = String(
                                format: "%.4f(skip=%d/%d)",
                                pm,
                                trainingSnap.rollingPlayedMoveProbNegAdvSkipped,
                                condWindowSize
                            )
                        } else {
                            playedProbNegStr = String(
                                format: "--(skip=%d/%d)",
                                trainingSnap.rollingPlayedMoveProbNegAdvSkipped,
                                condWindowSize
                            )
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
                        let pEntLegalStr: String
                        if let lm = legalMassOverride {
                            legalMassStr = String(format: "%.4f", lm.legalMass)
                            top1LegalStr = String(format: "%.2f", lm.top1LegalFraction)
                            pEntLegalStr = String(format: "%.4f", lm.legalEntropy)
                        } else {
                            legalMassStr = "--"
                            top1LegalStr = "--"
                            pEntLegalStr = "--"
                        }
                        // Surface the most-recent batch unique-position
                        // ratio so it scrolls past the eye in the same
                        // line as pEnt / gNorm. Reads NaN until the
                        // first stats-collection batch lands.
                        let bufUniqStr: String
                        let bufUniqPct = trainer.lastBatchStatsUniquePct
                        if !bufUniqPct.isNaN {
                            bufUniqStr = String(format: "%.4f", bufUniqPct)
                        } else {
                            bufUniqStr = "--"
                        }
                        // Per-step timing means over the rolling timing
                        // window. Splits `recentStepMs` into prep/gpu/
                        // read/queueWait/step so a slowdown can be
                        // attributed to one component (CPU prep, GPU
                        // compute, readback marshalling, executor
                        // queue backlog, or unaccounted overhead).
                        let timingStr: String
                        if let stepMs = trainingSnap.recentStepMs {
                            let prep = trainingSnap.recentDataPrepMs ?? 0
                            let gpu = trainingSnap.recentGpuRunMs ?? 0
                            let read = trainingSnap.recentReadbackMs ?? 0
                            let wait = trainingSnap.recentQueueWaitMs ?? 0
                            timingStr = String(
                                format: "step=%.1f gpu=%.1f prep=%.2f read=%.2f wait=%.2f n=%d",
                                stepMs, gpu, prep, read, wait, trainingSnap.recentTimingSamples
                            )
                        } else {
                            timingStr = "n=0"
                        }

                        // Process resident memory + delta since previous
                        // [STATS] tick. The delta lets a reader see
                        // sustained leak rate (e.g. "+8 MB / minute")
                        // at a glance rather than having to diff two
                        // log lines mentally.
                        let rssBytes = DiagSampler.currentResidentBytes()
                        let drss: Int64 = prevRssBytes == 0
                            ? 0
                            : Int64(rssBytes) - Int64(prevRssBytes)
                        prevRssBytes = rssBytes
                        let memStr = String(
                            format: "rss=%.2fGB drss=%+.1fMB",
                            Double(rssBytes) / 1024.0 / 1024.0 / 1024.0,
                            Double(drss) / 1024.0 / 1024.0
                        )

                        // VM region count + IOAccelerator-tagged
                        // subset. Cost ~1-3 ms per call; safe at this
                        // 60 s cadence. Each AGX-mapped GPU buffer
                        // shows up as one IOAccelerator region, so
                        // `vmAccel` growing in step with `trMs` is
                        // direct evidence that the per-`commit` cost
                        // is climbing in the kernel's residency walk.
                        let vm = DiagSampler.currentVMRegionCount()
                        let dvmTotal = prevVmTotal == 0
                            ? 0
                            : Int64(vm.total) - Int64(prevVmTotal)
                        let dvmAccel = prevVmIoAccel == 0
                            ? 0
                            : Int64(vm.ioAccelerator) - Int64(prevVmIoAccel)
                        prevVmTotal = vm.total
                        prevVmIoAccel = vm.ioAccelerator
                        let vmStr = String(
                            format: "total=%u dtotal=%+d ioAccel=%u dioAccel=%+d",
                            vm.total, dvmTotal, vm.ioAccelerator, dvmAccel
                        )

                        // Trainer feed-cache size. Stable at 1 means
                        // the trainer is calling MPSGraph with one
                        // feed shape (so any pipeline accumulation in
                        // MPSGraph is not because we're varying our
                        // inputs).
                        let shapesStr = "feedCache=\(trainer.feedCacheCount)"

                        let line = "[STATS] elapsed=\(elapsedStr) steps=\(trainingSnap.stats.steps) spGames=\(parallelSnap.selfPlayGames) spMoves=\(parallelSnap.selfPlayPositions) \(gameLenStr) buffer=\(bufCount)/\(bufCap) pLoss=\(policyStr) pLossWin=\(pLossWinStr) pLossLoss=\(pLossLossStr) vLoss=\(valueStr) pEnt=\(entropyStr) gNorm=\(gradNormStr) vNorm=\(vNormStr) μ=\(muStr) pwNorm=\(pwNormStr) pLogitAbsMax=\(pLogitMaxStr) playedMoveProb=\(playedProbStr) playedMoveProbPosAdv=\(playedProbPosStr) playedMoveProbNegAdv=\(playedProbNegStr) legalMass=\(legalMassStr) top1Legal=\(top1LegalStr) pEntLegal=\(pEntLegalStr) vMean=\(vMeanStr) vAbs=\(vAbsStr) vBaseDelta=\(vBaseDeltaStr) adv=(\(advStr)) sp.tau=\(spTau) ar.tau=\(arTau) diversity=\(divStr) ratio=(\(ratioStr)) outcomes=(\(outcomeStr)) bufUniq=\(bufUniqStr) \(cfgStr) reg=(\(regStr)) timing=(\(timingStr)) mem=(\(memStr)) vm=(\(vmStr)) shapes=(\(shapesStr)) build=\(BuildInfo.buildNumber) trainer=\(trainerID) champion=\(championID)"
                        SessionLogger.shared.log(line)

                        // CLI `--output` capture: one StatsLine per
                        // `[STATS]` log emit. All values come from the
                        // same thread-safe snapshots the log line just
                        // used, so the JSON and the log stay perfectly
                        // consistent. No-op when `recorder == nil`.
                        if let recorder {
                            let entry = CliTrainingRecorder.StatsLine(
                                elapsedSec: elapsedTarget,
                                steps: trainingSnap.stats.steps,
                                selfPlayGames: parallelSnap.selfPlayGames,
                                positionsTrained: parallelSnap.selfPlayPositions,
                                avgLen: lifetimeAvgLen,
                                rollingAvgLen: rollingAvgLen,
                                gameLenP50: parallelSnap.gameLenP50,
                                gameLenP95: parallelSnap.gameLenP95,
                                bufferCount: bufCount,
                                bufferCapacity: bufCap,
                                policyLoss: trainingSnap.rollingPolicyLoss,
                                valueLoss: trainingSnap.rollingValueLoss,
                                policyEntropy: trainingSnap.rollingPolicyEntropy,
                                gradGlobalNorm: trainingSnap.rollingGradGlobalNorm,
                                policyHeadWeightNorm: trainingSnap.rollingPolicyHeadWeightNorm,
                                policyLogitAbsMax: trainingSnap.rollingPolicyLogitAbsMax,
                                playedMoveProb: trainingSnap.rollingPlayedMoveProb,
                                playedMoveProbPosAdv: trainingSnap.rollingPlayedMoveProbPosAdv,
                                playedMoveProbPosAdvSkipped: trainingSnap.rollingPlayedMoveProbPosAdvSkipped,
                                playedMoveProbNegAdv: trainingSnap.rollingPlayedMoveProbNegAdv,
                                playedMoveProbNegAdvSkipped: trainingSnap.rollingPlayedMoveProbNegAdvSkipped,
                                playedMoveCondWindowSize: trainingSnap.rollingPlayedMoveCondWindowSize,
                                legalMass: legalMassOverride.map { Double($0.legalMass) },
                                top1LegalFraction: legalMassOverride.map { Double($0.top1LegalFraction) },
                                legalEntropy: legalMassOverride.map { Double($0.legalEntropy) },
                                policyLossWin: trainingSnap.rollingPolicyLossWin,
                                policyLossLoss: trainingSnap.rollingPolicyLossLoss,
                                batchStats: trainer.lastBatchStatsSummary.map { s in
                                    CliTrainingRecorder.BatchStatsSnapshot(
                                        step: s.step,
                                        batchSize: s.batchSize,
                                        uniqueCount: s.uniqueCount,
                                        uniquePct: s.uniquePct,
                                        dupMax: s.dupMax,
                                        dupDistribution: Dictionary(uniqueKeysWithValues:
                                            s.dupDistribution.map { (String($0.key), $0.value) }
                                        ),
                                        phaseByPlyHistogram: s.phaseByPlyHistogram,
                                        phaseByMaterialHistogram: s.phaseByMaterialHistogram,
                                        gameLengthHistogram: s.gameLengthHistogram,
                                        samplingTauHistogram: s.samplingTauHistogram,
                                        workerIdHistogram: s.workerIdHistogram,
                                        outcomeHistogram: s.outcomeHistogram,
                                        phaseByPlyXOutcomeHistogram: s.phaseByPlyXOutcomeHistogram,
                                        bufferUniquePositions: s.bufferUniquePositions,
                                        bufferStoredCount: s.bufferStoredCount
                                    )
                                },
                                valueMean: trainingSnap.rollingValueMean,
                                valueAbsMean: trainingSnap.rollingValueAbsMean,
                                vBaselineDelta: trainingSnap.rollingVBaselineDelta,
                                advMean: trainingSnap.rollingAdvMean,
                                advStd: trainingSnap.rollingAdvStd,
                                advMin: trainingSnap.rollingAdvMin,
                                advMax: trainingSnap.rollingAdvMax,
                                advFracPositive: trainingSnap.rollingAdvFracPositive,
                                advFracSmall: trainingSnap.rollingAdvFracSmall,
                                advP05: trainingSnap.advantageP05,
                                advP50: trainingSnap.advantageP50,
                                advP95: trainingSnap.advantageP95,
                                spStartTau: Double(spSched.startTau),
                                spFloorTau: Double(spSched.floorTau),
                                spDecayPerPly: Double(spSched.decayPerPly),
                                arStartTau: Double(arSched.startTau),
                                arFloorTau: Double(arSched.floorTau),
                                arDecayPerPly: Double(arSched.decayPerPly),
                                diversityUniqueGames: divSnap.uniqueGames,
                                diversityGamesInWindow: divSnap.gamesInWindow,
                                diversityUniquePercent: divSnap.uniquePercent,
                                diversityAvgDivergencePly: divSnap.avgDivergencePly,
                                ratioTarget: ratioSnap.targetRatio,
                                ratioCurrent: ratioSnap.currentRatio,
                                ratioProductionRate: ratioSnap.productionRate,
                                ratioConsumptionRate: ratioSnap.consumptionRate,
                                selfPlayMovesPerHour: spMovesPerHour,
                                trainingMovesPerHour: trainMovesPerHour,
                                ratioAutoAdjust: ratioSnap.autoAdjust,
                                ratioComputedDelayMs: ratioSnap.computedDelayMs,
                                whiteCheckmates: parallelSnap.whiteCheckmates,
                                blackCheckmates: parallelSnap.blackCheckmates,
                                stalemates: parallelSnap.stalemates,
                                fiftyMoveDraws: parallelSnap.fiftyMoveDraws,
                                threefoldRepetitionDraws: parallelSnap.threefoldRepetitionDraws,
                                insufficientMaterialDraws: parallelSnap.insufficientMaterialDraws,
                                batchSize: sessionTrainingBatchSize,
                                learningRate: Double(lr),
                                promoteThreshold: sessionPromoteThreshold,
                                arenaGames: sessionTournamentGames,
                                workerCount: workerN,
                                gradClipMaxNorm: Double(gradClip),
                                weightDecayC: Double(weightDec),
                                entropyRegularizationCoeff: Double(entropyCoeff),
                                drawPenalty: Double(drawPen),
                                policyScaleK: Double(kScale),
                                buildNumber: BuildInfo.buildNumber,
                                trainerID: trainerID,
                                championID: championID
                            )
                            recorder.appendStats(entry)
                        }

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
                                if buffer.count >= legalMassSampleSize, let probeNet = probeInferenceForProbes {
                                    do {
                                        lastLegalMass = try await trainer.legalMassSnapshot(
                                            replayBuffer: buffer,
                                            sampleSize: legalMassSampleSize,
                                            inferenceNetwork: probeNet
                                        )
                                        // Mirror to @State so the
                                        // chart-sample heartbeat can
                                        // render the legal-entropy
                                        // trace at its own cadence.
                                        let snap = lastLegalMass
                                        await MainActor.run {
                                            realLastLegalMassSnapshot = snap
                                        }
                                    } catch {
                                        lastLegalMass = nil
                                        SessionLogger.shared.log(
                                            "[STATS-ERR] legalMassSnapshot failed at step \(steps): \(error.localizedDescription)"
                                        )
                                    }
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
                        if buffer.count >= legalMassSampleSize, let probeNet = probeInferenceForProbes {
                            let trainingSnap = box.snapshot()
                            let steps = trainingSnap.stats.steps
                            do {
                                lastLegalMass = try await trainer.legalMassSnapshot(
                                    replayBuffer: buffer,
                                    sampleSize: legalMassSampleSize,
                                    inferenceNetwork: probeNet
                                )
                                // Mirror to @State so the chart-sample
                                // heartbeat sees a fresh legal-entropy
                                // value at the steady-state cadence.
                                let snap = lastLegalMass
                                await MainActor.run {
                                    realLastLegalMassSnapshot = snap
                                }
                            } catch {
                                lastLegalMass = nil
                                SessionLogger.shared.log(
                                    "[STATS-ERR] legalMassSnapshot failed at step \(steps): \(error.localizedDescription)"
                                )
                            }
                        }
                        let elapsed = Date().timeIntervalSince(sessionStart)
                        await logOne(elapsedTarget: elapsed, legalMassOverride: lastLegalMass)
                    }
                }

                // Headless CLI deadline. Armed whenever the run
                // started with `--train` AND `training_time_limit`
                // is set — the time limit is the primary driver of
                // output generation and exit, regardless of
                // whether `--output <file>` was supplied. When
                // `--output` is present the JSON snapshot is
                // written to that file (overwriting); when absent
                // the snapshot goes to stdout so the user can
                // redirect or pipe it. Outside of `--train` mode
                // (interactive use) the deadline is ignored — the
                // user is driving the UI and an unexpected
                // termination would be hostile.
                //
                // On wake-up: log the deadline event, emit the
                // recorder's snapshot to the configured sink, then
                // call `Darwin._exit(0)`. The crucial detail is
                // `_exit` vs `exit`: `exit` runs C++ atexit
                // handlers (including CoreAnalytics' exit barrier)
                // while the MPS self-play worker is still mid-
                // `graph.run` on its dedicated serial dispatch
                // queue — the handler tears down global state
                // that `MPSGraphOSLog` still reads, producing
                // EXC_BAD_ACCESS at 0x8 inside the MPSGraph
                // executable run path. `_exit` terminates the
                // process immediately without running atexit or
                // stdio cleanup, which sidesteps the race. It's
                // safe because:
                //   - snapshot file writes use `Data.write(atomic:)`
                //     which fsyncs before rename,
                //   - stdout writes go through `FileHandle`
                //     which is an unbuffered syscall (no libc
                //     stdio buffer to flush),
                //   - session log writes have already been
                //     flushed by SessionLogger before this point.
                if isAutoTrainRun, let recorder, let deadlineSec = cliTrainingTimeLimitSec, deadlineSec > 0 {
                    group.addTask(priority: .utility) {
                        do {
                            try await Task.sleep(for: .seconds(deadlineSec))
                        } catch {
                            // Cancelled before the deadline hit —
                            // session ended for another reason.
                            return
                        }
                        if Task.isCancelled { return }
                        let elapsed = Date().timeIntervalSince(runStart)
                        let destDescription = outputURL?.path ?? "<stdout>"
                        SessionLogger.shared.log(
                            "[APP] --train: training_time_limit=\(deadlineSec)s reached at elapsed=\(String(format: "%.1f", elapsed))s; writing snapshot to \(destDescription)"
                        )
                        recorder.setTerminationReason(.timerExpired)
                        let counts = recorder.countsSnapshot()
                        do {
                            if let outputURL {
                                try recorder.writeJSON(to: outputURL, totalTrainingSeconds: elapsed)
                            } else {
                                try recorder.writeJSONToStdout(totalTrainingSeconds: elapsed)
                            }
                            SessionLogger.shared.log(
                                "[APP] --train: wrote snapshot to \(destDescription) (arenas=\(counts.arenas), stats=\(counts.stats), probes=\(counts.probes))"
                            )
                        } catch {
                            SessionLogger.shared.log(
                                "[APP] --train: snapshot write FAILED for \(destDescription): \(error.localizedDescription)"
                            )
                        }
                        SessionLogger.shared.log("[APP] --train: exiting process after snapshot")
                        Darwin._exit(0)
                    }
                }

                // Legal-mass collapse detector. Runs in every session
                // (interactive or --train), probes at 60 s cadence
                // with a 120 s grace period measured from the first
                // observed SGD step (i.e. earliest possible detection
                // is training_start + 120 s), flags the collapse
                // banner once the network has shown
                // `illegalMass > 0.99` for `collapseProbesToAbort`
                // consecutive probes. In interactive mode we stop at
                // the banner — the user sees it and decides what to
                // do. In --train mode we ALSO write the snapshot with
                // `termination_reason = legal_mass_collapse` and exit
                // the process (status 0 — a clean early abort, not a
                // crash; scripting distinguishes via the JSON field
                // rather than exit code per the user's spec).
                // Local "session start" for the collapse detector.
                // The --train timer task uses its own `runStart`;
                // both are set at roughly the same moment in the
                // task-group setup (drift < 1 ms) and the two are
                // only used for "elapsed since session start"
                // display in logs/alarms, so a separate declaration
                // is intentional — it keeps the detector
                // self-contained and doesn't require plumbing the
                // timer task's runStart into this closure.
                let collapseRunStart = Date()
                let collapseRecorder: CliTrainingRecorder? = (isAutoTrainRun ? recorder : nil)
                let collapseOutputURL: URL? = outputURL
                // Configuration snapshot taken at task start. All three
                // are user-tunable via parameters JSON / @AppStorage; we
                // capture once so the running detector has stable behavior
                // even if the user edits the values mid-session (changes
                // take effect on the next training run).
                let collapseIllegalMassThreshold = trainingParams.legalMassCollapseThreshold
                let collapseGracePeriodSec = trainingParams.legalMassCollapseGraceSeconds
                let collapseNoImprovementProbeCount = max(1, trainingParams.legalMassCollapseNoImprovementProbes)
                group.addTask(priority: .utility) {
                    [trainer, buffer, box, probeInferenceForProbes] in
                    let probeIntervalSec: UInt64 = 60
                    let sampleSize = 128
                    let gracePeriodSec: TimeInterval = collapseGracePeriodSec
                    let illegalMassThreshold: Double = collapseIllegalMassThreshold
                    let noImprovementProbeCount = collapseNoImprovementProbeCount
                    // Sliding window of recent legal-mass readings,
                    // newest at end. Trip condition: window is full,
                    // every reading's illegalMass is above threshold,
                    // AND the newest legal_mass is no better than the
                    // oldest (no upward improvement across the window).
                    // This catches *stuck* collapse — slow climbs out
                    // of near-uniform won't fire even if absolute
                    // legal mass is still below threshold for a while.
                    var legalMassWindow: [Double] = []
                    var aborted = false
                    // Anchor for the grace countdown. Lazily set the
                    // first time we observe at least one completed SGD
                    // step. The replay buffer has to fill enough for
                    // training to begin (minutes in practice), and
                    // before that first step lands, every probe on the
                    // fresh random-init network will naturally see
                    // illegalMass ≈ 0.994 — firing the alarm during
                    // that window is a false positive. Grace is 120 s
                    // measured from TRAINING start, not session start.
                    var trainingStartAt: Date? = nil
                    while !Task.isCancelled && !aborted {
                        do {
                            try await Task.sleep(for: .seconds(probeIntervalSec))
                        } catch {
                            return
                        }
                        if Task.isCancelled { return }
                        // Training-start gate. If no SGD steps have
                        // landed yet, reset the anchor and skip — the
                        // network is still at random init so no
                        // learning-progress signal exists. Once steps
                        // > 0, stamp the anchor on the first qualifying
                        // iteration and measure grace from there.
                        let trainingSteps = box.snapshot().stats.steps
                        guard trainingSteps > 0 else {
                            trainingStartAt = nil
                            continue
                        }
                        if trainingStartAt == nil {
                            trainingStartAt = Date()
                        }
                        guard let startAt = trainingStartAt else { continue }
                        let trainingElapsed = Date().timeIntervalSince(startAt)
                        if trainingElapsed < gracePeriodSec { continue }
                        // Session-elapsed for snapshot-write / log
                        // consistency with the timer task, which also
                        // reports `totalTrainingSeconds` as
                        // session-elapsed rather than training-elapsed.
                        let elapsed = Date().timeIntervalSince(collapseRunStart)
                        // Skip the probe if the inference network isn't
                        // available — the pollution-prone trainer-network
                        // fallback would silently corrupt BN running stats
                        // and defeat the reason we switched detectors off
                        // the trainer network in the first place.
                        guard let probeNet = probeInferenceForProbes else { continue }

                        let snap: ChessTrainer.LegalMassSnapshot?
                        do {
                            snap = try await trainer.legalMassSnapshot(
                                replayBuffer: buffer,
                                sampleSize: sampleSize,
                                inferenceNetwork: probeNet
                            )
                        } catch {
                            SessionLogger.shared.log(
                                "[STATS-ERR] collapse-detector legalMassSnapshot failed: \(error.localizedDescription)"
                            )
                            continue
                        }
                        guard let snap else { continue }
                        let illegalMass = 1.0 - snap.legalMass
                        legalMassWindow.append(snap.legalMass)
                        if legalMassWindow.count > noImprovementProbeCount {
                            legalMassWindow.removeFirst()
                        }
                        // Window-full tripwire: we need
                        // `noImprovementProbeCount` samples before
                        // declaring "no improvement"; until then we
                        // log the probe and continue.
                        let windowFull = legalMassWindow.count >= noImprovementProbeCount
                        let allAboveThreshold: Bool = {
                            guard windowFull else { return false }
                            for legalMass in legalMassWindow where (1.0 - legalMass) <= illegalMassThreshold {
                                return false
                            }
                            return true
                        }()
                        let noImprovement: Bool = {
                            guard windowFull,
                                  let oldest = legalMassWindow.first,
                                  let newest = legalMassWindow.last else {
                                return false
                            }
                            return newest <= oldest
                        }()
                        let probeMatches = allAboveThreshold && noImprovement
                        if probeMatches {
                            SessionLogger.shared.log(
                                String(format: "[ALARM] legal-mass collapse window: %d/%d probes illegalMass>%.4f, newest legalMass=%.5f ≤ oldest=%.5f",
                                    legalMassWindow.count, noImprovementProbeCount,
                                    illegalMassThreshold,
                                    legalMassWindow.last ?? 0, legalMassWindow.first ?? 0)
                            )
                            await MainActor.run {
                                self.raiseTrainingAlarm(
                                    severity: .critical,
                                    title: "Policy Collapse (legal mass)",
                                    detail: String(format: "illegalMass>%.4f for %d probes with no improvement (legal_mass %.5f→%.5f)",
                                        illegalMassThreshold, legalMassWindow.count,
                                        legalMassWindow.first ?? 0, legalMassWindow.last ?? 0)
                                )
                            }
                        } else {
                            SessionLogger.shared.log(
                                String(format: "[ALARM] legal-mass probe ok: legalMass=%.5f illegalMass=%.4f window=%d/%d",
                                    snap.legalMass, illegalMass,
                                    legalMassWindow.count, noImprovementProbeCount)
                            )
                        }

                        if probeMatches {
                            aborted = true
                            SessionLogger.shared.log(
                                String(format: "[ALARM] legal-mass collapse confirmed: %d probes above %.4f with no improvement (legal_mass %.5f→%.5f) — aborting",
                                    legalMassWindow.count, illegalMassThreshold,
                                    legalMassWindow.first ?? 0, legalMassWindow.last ?? 0)
                            )
                            if let rec = collapseRecorder {
                                let destDescription = collapseOutputURL?.path ?? "<stdout>"
                                SessionLogger.shared.log(
                                    "[APP] --train: legal-mass collapse abort at elapsed=\(String(format: "%.1f", elapsed))s; writing snapshot to \(destDescription)"
                                )
                                rec.setTerminationReason(.legalMassCollapse)
                                let counts = rec.countsSnapshot()
                                do {
                                    if let url = collapseOutputURL {
                                        try rec.writeJSON(to: url, totalTrainingSeconds: elapsed)
                                    } else {
                                        try rec.writeJSONToStdout(totalTrainingSeconds: elapsed)
                                    }
                                    SessionLogger.shared.log(
                                        "[APP] --train: wrote snapshot to \(destDescription) (arenas=\(counts.arenas), stats=\(counts.stats), probes=\(counts.probes))"
                                    )
                                } catch {
                                    SessionLogger.shared.log(
                                        "[APP] --train: snapshot write FAILED for \(destDescription): \(error.localizedDescription)"
                                    )
                                }
                                SessionLogger.shared.log("[APP] --train: exiting process after collapse-abort snapshot")
                                Darwin._exit(0)
                            }
                            // Interactive session: alarm already
                            // raised, loop stays exited so we stop
                            // probing (no value continuing once a
                            // confirmed collapse has been flagged).
                        }
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
                chartCoordinator.setDiversityHistogramBars([])
                chartCoordinator.arenaChartEvents = []
                chartCoordinator.cancelActiveArena()
                workerCountBox = nil
                samplingScheduleBox = nil
                activeSelfPlayGate = nil
                activeTrainingGate = nil
                currentSessionID = nil
                currentSessionStart = nil
                replayRatioController = nil
                replayRatioSnapshot = nil
                effectiveReplayRatioTarget = nil
                lastReplayRatioCompensatorAt = nil
                cliRecorder = nil
                EarlyStopCoordinator.shared.earlyStopHandler = nil
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
        // Disarm the periodic-save scheduler so a Stop-then-Start
        // doesn't fire an immediate save on Start. The next Start
        // constructs a fresh controller with a fresh 4-hour
        // deadline.
        periodicSaveController?.disarm()
        periodicSaveController = nil
        periodicSaveLastPollAt = nil
        periodicSaveInFlight = false
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

    /// One of four high-level session states surfaced as a colored
    /// chip in the top status bar. Order is the natural progression of
    /// a training session: Idle → SelfPlayPrefill (workers running but
    /// the trainer is still waiting on
    /// `trainingParams.replayBufferMinPositionsBeforeTraining`) →
    /// TrainingWarmup (trainer is taking steps but the LR-warmup
    /// multiplier is still <1) → Training (warmup complete). Outside
    /// of a Play-and-Train session we always read Idle, regardless of
    /// transient single-shot operations like Build or Forward Pass —
    /// those have their own busy indicators and don't merit a "what
    /// is this session doing?" chip.
    /// Derives the current chip state from session state. Order matters:
    /// the prefill check must come before the warmup check, since the
    /// trainer is waiting on the buffer in that window and has not
    /// taken a step yet (so `completedTrainSteps == 0`, which would
    /// otherwise look like the very first warmup step). Outside of a
    /// Play-and-Train session this always returns `.idle`.
    private var sessionStatusChip: SessionStatusChipView.Kind {
        guard realTraining else { return .idle }
        let bufferCount = replayBuffer?.count ?? 0
        if bufferCount < trainingParams.replayBufferMinPositionsBeforeTraining {
            return .selfPlayPrefill
        }
        if let snap = trainerWarmupSnap, snap.inWarmup {
            return .trainingWarmup
        }
        return .training
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
    /// Teardown on actual main-window close only — NOT on minimize
    /// (which fires .onDisappear but not willClose), and NOT on
    /// auxiliary windows (Log Analysis, NSOpenPanel/NSSavePanel,
    /// etc.) which all post the same notification. The object check
    /// narrows us to exactly this view's hosting NSWindow, captured
    /// by `WindowAccessor`.
    private func handleWindowWillClose(note: Notification) {
        guard let closing = note.object as? NSWindow,
              let ours = contentWindow,
              closing === ours else {
            return
        }
        stopAnyContinuous()
        clearTrainingAlarm()
    }

    private var cumulativeStatusBar: some View {
        let totalSteps = trainingStats?.steps ?? 0
        let hasHistory = cumulativeRunCount > 0 || totalSteps > 0
        let canRunArena = !isArenaRunning && network != nil && trainer != nil
        let totalPositions = totalSteps * trainingParams.trainingBatchSize
        let activeSec = cumulativeActiveTrainingSec
        return CumulativeStatusBar(
            hasHistory: hasHistory,
            isVisible: hasHistory || canRunArena,
            historyCells: {
                StatusBarCell(
                    label: "Active training time",
                    value: GameWatcher.Snapshot.formatHMS(seconds: activeSec)
                )
                    // LR Warm-up cell — only visible while the trainer is
                if let snap = trainerWarmupSnap, snap.inWarmup {
                    StatusBarCell(
                        label: "LR effective",
                        value: String(format: "%.2e", snap.effectiveLR)
                    )
                }
                StatusBarCell(
                    label: "Training steps",
                    value: Int(totalSteps).formatted()
                )
                StatusBarCell(
                    label: "Positions trained",
                    value: Self.formatCompactCount(totalPositions)
                )
                StatusBarCell(
                    label: "Training rate",
                    value: trainingRateStatusValue
                )
                StatusBarCell(
                    label: "Legal mass",
                    value: realLastLegalMassSnapshot.map {
                        String(format: "%.4f%%", Double($0.legalMass) * 100)
                    } ?? "--"
                )
                StatusBarCell(
                    label: "Runs",
                    value: "\(cumulativeRunCount)"
                )
                StatusBarCell(
                    label: "Arenas",
                    value: "\(tournamentHistory.count)"
                )
                StatusBarCell(
                    label: "Promotions",
                    value: "\(tournamentHistory.lazy.filter { $0.promoted }.count)"
                )
                scoreStatusBarCell
            },
            rightChips: {
                SessionStatusChipView(
                    kind: sessionStatusChip,
                    warmupCompletedSteps: trainerWarmupSnap?.completedSteps,
                    warmupTotalSteps: trainerWarmupSnap?.warmupSteps
                )
                TrainingSettingsChip(showPopover: $showTrainingPopover) {
                    TrainingSettingsPopover(
                        modelID: trainer?.identifier?.description ?? "—",
                        sessionStart: currentSessionStart ?? Date(),
                        lrText: $learningRateEditText,
                        warmupText: $lrWarmupStepsEditText,
                        momentumText: $momentumCoeffEditText,
                        sqrtBatchScalingValue: $sqrtBatchScalingEditValue,
                        entropyText: $entropyRegularizationEditText,
                        gradClipText: $gradClipMaxNormEditText,
                        weightDecayText: $weightDecayEditText,
                        policyKText: $policyScaleKEditText,
                        drawPenaltyText: $drawPenaltyEditText,
                        trainingBatchSizeText: $trainingBatchSizeEditText,
                        lrError: trainingPopoverLRError,
                        warmupError: trainingPopoverWarmupError,
                        momentumError: trainingPopoverMomentumError,
                        entropyError: trainingPopoverEntropyError,
                        gradClipError: trainingPopoverGradClipError,
                        weightDecayError: trainingPopoverWeightDecayError,
                        policyKError: trainingPopoverPolicyKError,
                        drawPenaltyError: trainingPopoverDrawPenaltyError,
                        trainingBatchSizeError: trainingPopoverTrainingBatchSizeError,
                        selfPlayWorkersText: $selfPlayWorkersEditText,
                        selfPlayStartTauText: $spStartTauEditText,
                        selfPlayDecayPerPlyText: $spDecayPerPlyEditText,
                        selfPlayFloorTauText: $spFloorTauEditText,
                        selfPlayWorkersError: trainingPopoverSelfPlayWorkersError,
                        selfPlayStartTauError: trainingPopoverSelfPlayStartTauError,
                        selfPlayDecayPerPlyError: trainingPopoverSelfPlayDecayPerPlyError,
                        selfPlayFloorTauError: trainingPopoverSelfPlayFloorTauError,
                        replayBufferCapacityText: $replayBufferCapacityEditText,
                        replayBufferMinPositionsText: $replayBufferMinPositionsBeforeTrainingEditText,
                        replayRatioTargetText: $replayRatioTargetEditText,
                        replaySelfPlayDelayText: $replaySelfPlayDelayEditText,
                        replayTrainingStepDelayText: $replayTrainingStepDelayEditText,
                        replayRatioAutoAdjust: $replayRatioAutoAdjustEditValue,
                        replayBufferCapacityError: trainingPopoverReplayBufferCapacityError,
                        replayBufferMinPositionsError: trainingPopoverReplayBufferMinPositionsError,
                        replayRatioTargetError: trainingPopoverReplayRatioTargetError,
                        replaySelfPlayDelayError: trainingPopoverReplaySelfPlayDelayError,
                        replayTrainingStepDelayError: trainingPopoverReplayTrainingStepDelayError,
                        replayRatioCurrent: replayRatioSnapshot?.currentRatio,
                        replayRatioComputedDelayMs: replayRatioSnapshot?.computedDelayMs,
                        replayRatioComputedSelfPlayDelayMs: replayRatioSnapshot?.computedSelfPlayDelayMs,
                        bytesPerPosition: ReplayBuffer.bytesPerPosition,
                        onLiveReplayRatioTargetChange: { newValue in
                            trainingPopoverApplyLiveReplayRatioTarget(newValue)
                        },
                        onLiveSelfPlayDelayChange: { newValue in
                            trainingPopoverApplyLiveSelfPlayDelay(newValue)
                        },
                        onLiveTrainingStepDelayChange: { newValue in
                            trainingPopoverApplyLiveTrainingStepDelay(newValue)
                        },
                        onLiveReplayRatioAutoAdjustChange: { newValue in
                            trainingPopoverApplyLiveReplayRatioAutoAdjust(newValue)
                        },
                        onCancel: { trainingPopoverCancel() },
                        onSave: { trainingPopoverSave() },
                        onAppearSeed: { trainingPopoverSeedFromParams() }
                    )
                }
                ArenaCountdownChip(
                    isArenaRunning: isArenaRunning,
                    countdownText: { now in arenaCountdownText(at: now) },
                    showPopover: $showArenaPopover
                ) {
                    ArenaSettingsPopover(
                        nextArenaDate: arenaTriggerBox.map {
                            $0.lastArenaTime.addingTimeInterval(trainingParams.arenaAutoIntervalSec)
                        },
                        lastArena: tournamentHistory.last,
                        isArenaRunning: isArenaRunning,
                        realTraining: realTraining,
                        gamesText: $arenaPopoverGamesText,
                        concurrencyText: $arenaPopoverConcurrencyText,
                        intervalText: $arenaPopoverIntervalText,
                        tauStartText: $arStartTauEditText,
                        tauDecayText: $arDecayPerPlyEditText,
                        tauFloorText: $arFloorTauEditText,
                        gamesError: arenaPopoverGamesError,
                        concurrencyError: arenaPopoverConcurrencyError,
                        intervalError: arenaPopoverIntervalError,
                        tauStartError: arenaPopoverTauStartError,
                        tauDecayError: arenaPopoverTauDecayError,
                        tauFloorError: arenaPopoverTauFloorError,
                        onRunNow: {
                            SessionLogger.shared.log("[BUTTON] Run Arena (popover)")
                            arenaTriggerBox?.trigger()
                            showArenaPopover = false
                        },
                        onShowHistory: {
                            SessionLogger.shared.log("[BUTTON] Open Arena History")
                            showArenaPopover = false
                            showArenaHistorySheet = true
                        },
                        onCancel: { showArenaPopover = false },
                        onSave: { arenaPopoverSave() },
                        onAppearSeed: { arenaPopoverSeedFromParams() }
                    )
                }
            }
        )
    }

    /// Outer integral compensator step. Called every heartbeat tick
    /// from `processSnapshotTimerTick` while a session is live.
    ///
    /// Wraps the replay-ratio controller without modifying it. The
    /// inner controller's per-tick overhead estimate
    /// (`currentDelaySettingMs × positionsProduced / G`) is
    /// dimensionally consistent on its own terms, but in the batched-
    /// shared-evaluator architecture the per-game self-play sleep does
    /// not parallelize cleanly across `N` workers — they serialize
    /// through one batcher barrier — so the wall inflation per tick
    /// is observably mis-scaled relative to what the controller
    /// expects to subtract. The mismatch lets the controller's signed
    /// equilibrium settle at a `cons/prod` ratio meaningfully below
    /// (or above, depending on workload) the user-configured
    /// `replayRatioTarget`. Empirically, with `replayRatioTarget=1.10`
    /// and `selfPlayWorkers=48`, the inner controller settles at
    /// `currentRatio ≈ 0.78` and stays there — well outside the R1
    /// monitoring band `[0.90, 1.25]`.
    ///
    /// Mechanism: each tick, observe `gap = userTarget −
    /// snap.currentRatio`. Move the controller's INTERNAL set-point
    /// `T_eff` in the same sign as `gap` (positive gap means observed
    /// is too low → push the controller to demand more SP throttling
    /// → raise `T_eff`). Update is `T_eff += k × gap × dt` with `k`
    /// (`gainPerSecond` below) tuned so the outer loop is no faster
    /// than the inner controller's SMA bandwidth
    /// (`1 / ReplayRatioController.historyWindowSec`); the dead-band
    /// and bounded `T_eff` clamp keep the two loops from fighting
    /// even when the bandwidths are comparable. Convergence on
    /// typical observed gaps lands inside the autotrain warm-up
    /// window before R1/R2 monitoring begins.
    ///
    /// Bounds: `T_eff ∈ [0.5, 5.0] × userTarget`. Lower bound prevents
    /// the compensator from disabling SP throttling entirely; upper
    /// bound prevents a transient noise spike from drifting the
    /// controller into a degenerate set-point that would take many
    /// minutes to recover. Both bounds are far outside any healthy
    /// equilibrium so they only trip on pathological inputs.
    ///
    /// Skips:
    ///   • `autoAdjust == false` — the controller's SP delay is
    ///     pinned at 0 in this mode, so there is no SP throttle to
    ///     compensate. We also reset the saved `T_eff` so the next
    ///     auto-on transition starts fresh.
    ///   • `currentRatio <= 0` — the controller hasn't accumulated
    ///     enough samples for a meaningful ratio yet. Holds the
    ///     existing `T_eff` and waits for real data.
    ///   • `realTraining == false` — out of an abundance of caution
    ///     even though `replayRatioController` is itself nil outside
    ///     a session. Cheap belt-and-braces.
    @MainActor
    private func updateReplayRatioCompensator(
        snap: ReplayRatioController.RatioSnapshot
    ) {
        guard realTraining,
              let rc = replayRatioController else {
            // No active session — clear state so a fresh start
            // re-seeds. Also handles the brief window between
            // `replayRatioController = nil` (in the session-stop
            // teardown) and the next session's start.
            if effectiveReplayRatioTarget != nil {
                effectiveReplayRatioTarget = nil
                lastReplayRatioCompensatorAt = nil
            }
            return
        }
        let userTarget = trainingParams.replayRatioTarget
        guard userTarget > 0 else { return }
        // Auto-adjust off: SP throttle is pinned at 0, the inner
        // controller is in manual training-delay mode, and the outer
        // compensator has nothing to compensate. Drop saved state so
        // a future flip back to auto starts from `userTarget`.
        if !snap.autoAdjust {
            if effectiveReplayRatioTarget != nil {
                effectiveReplayRatioTarget = nil
                lastReplayRatioCompensatorAt = nil
            }
            return
        }
        // First post-(re)start tick: seed `T_eff` to the user value
        // and stamp the clock. No update this tick — the next tick's
        // `dt` will be one heartbeat (~100 ms), which is long enough
        // to integrate against. This matches the inner controller's
        // first-call convention (stamp only, no measurement).
        let now = Date()
        guard let prevTeff = effectiveReplayRatioTarget,
              let prevAt = lastReplayRatioCompensatorAt else {
            effectiveReplayRatioTarget = userTarget
            lastReplayRatioCompensatorAt = now
            // Make sure the controller's internal target matches the
            // user value at the start of compensation, in case any
            // prior session left it drifted. The inner controller's
            // own onChange handler also writes this value, but doing
            // it here is robust to ordering on the very first tick
            // after session start.
            rc.targetRatio = userTarget
            return
        }
        // No meaningful ratio yet (insufficient samples in the
        // 60 s rolling window). Hold state and wait. The controller
        // returns 0 from `rollingRates` until at least one second of
        // span is in the window, which produces `currentRatio == 0`.
        guard snap.currentRatio > 0 else {
            lastReplayRatioCompensatorAt = now
            return
        }
        let dt = now.timeIntervalSince(prevAt)
        // Defensive: dt should be ~0.1s on the 10 Hz heartbeat. Bound
        // it so a long pause (e.g. waking from sleep, a UI hang, an
        // arena-pause that still left this tick running) doesn't
        // produce a giant integration step. 1 s is far longer than
        // any normal heartbeat gap and still small enough that the
        // gain math doesn't blow up.
        let dtClamped = min(max(dt, 0.0), 1.0)
        // Integral gain in `target-units per (ratio-unit × second)`.
        // Tuned to be no faster than the inner controller's SMA
        // bandwidth (`1 / ReplayRatioController.historyWindowSec`);
        // the dead-band below and the bounded `T_eff` clamp keep
        // the two loops from fighting even when the bandwidths are
        // comparable. Adjust both knobs together if `historyWindowSec`
        // changes meaningfully — otherwise the outer loop can outrun
        // the inner SMA and the controller starts hunting.
        let gainPerSecond = 0.05
        let gap = userTarget - snap.currentRatio
        var nextTeff = prevTeff + gainPerSecond * gap * dtClamped
        let lo = 0.5 * userTarget
        let hi = 5.0 * userTarget
        if nextTeff < lo { nextTeff = lo }
        if nextTeff > hi { nextTeff = hi }
        // Only push to the controller when the change is meaningful.
        // A ~0.001 dead-band keeps us off the controller's serial
        // queue when the loop is at equilibrium. The previous value
        // we hold is the LAST PUSHED value, not the most recent
        // unconverged compute, so this dead-band doesn't accumulate
        // un-pushed drift.
        if abs(nextTeff - prevTeff) > 0.001 {
            rc.targetRatio = nextTeff
            effectiveReplayRatioTarget = nextTeff
        }
        lastReplayRatioCompensatorAt = now
    }


    /// Format the Score status-bar cell's value. `nil` lastArena
    /// renders a dimmed em-dash. Otherwise the cell toggles between a
    /// percentage view and an Elo-with-CI view controlled by
    /// `scoreStatusShowElo`.
    private func scoreStatusCellValue(lastArena: TournamentRecord?) -> String {
        guard let r = lastArena else { return "—" }
        if scoreStatusShowElo {
            return ArenaEloStats.formatEloWithCI(r.eloSummary)
        }
        return String(format: "%.1f%%", r.score * 100)
    }

    /// The Score / Elo cell for the status bar. Broken out of the
    /// main `cumulativeStatusBar` HStack because the ternary-produced
    /// optional closure and optional color were pushing SwiftUI's
    /// type-checker past its complexity budget when inlined alongside
    /// the other five cells.
    private var scoreStatusBarCell: StatusBarCell {
        let lastArena = tournamentHistory.last
        let label = scoreStatusShowElo ? "Elo" : "Score"
        let value = scoreStatusCellValue(lastArena: lastArena)
        let hasArena = lastArena != nil
        let action: (() -> Void)?
        if hasArena {
            action = { scoreStatusShowElo.toggle() }
        } else {
            action = nil
        }
        let color: Color = hasArena ? .primary : .secondary
        return StatusBarCell(
            label: label,
            value: value,
            action: action,
            valueColor: color
        )
    }

    /// Apply attribute-based color highlighting to the multi-line
    /// body text of a stats panel. Wraps `AttributedMetricColor`
    /// with the live grad-clip ceiling and the project's
    /// `policyEntropyAlarmThreshold` so the entropy/gNorm bands stay
    /// calibrated against whatever values are currently in use.
    private func colorizedPanelBody(_ body: String) -> AttributedString {
        let thresholds = AttributedMetricColor.Thresholds.default(
            entropyCollapseBelow: Self.policyEntropyAlarmThreshold,
            gradClipMaxNorm: trainingParams.gradClipMaxNorm
        )
        return AttributedMetricColor.colorize(body: body, thresholds: thresholds)
    }

    /// The bold zoom-level indicator row that sits above the upper-
    /// left chart of the grid. Shows the current window length (e.g.
    /// "30m"), the ⌘= / ⌘- keyboard hints, and an Auto toggle. The
    /// indicator text is heavier than the chart tile titles
    /// (`.title3.weight(.bold)` vs `.caption2` per tile) so the eye
    /// picks it up as the row's left anchor.

    // MARK: - Arena countdown chip

    /// Seconds remaining until the next auto-arena fires, evaluated at
    /// `now`. Returns nil when there is no active session or no
    /// trigger box yet — the chip then renders `--:--:--`. Returns 0
    /// when the configured interval has already elapsed (the box will
    /// fire on its next poll and the countdown will reset).
    ///
    /// `now` is taken as a parameter rather than read from `Date()`
    /// inside so the `TimelineView` schedule below can drive updates
    /// off `context.date` deterministically.
    private func secondsUntilNextArena(at now: Date) -> Double? {
        guard let box = arenaTriggerBox, realTraining else { return nil }
        let elapsed = now.timeIntervalSince(box.lastArenaTime)
        return max(0, trainingParams.arenaAutoIntervalSec - elapsed)
    }

    /// Renderable label for the arena countdown chip. `--:--:--` when
    /// no session is active or the trigger box hasn't been built yet.
    private func arenaCountdownText(at now: Date) -> String {
        guard let secs = secondsUntilNextArena(at: now) else { return "--:--:--" }
        return GameWatcher.Snapshot.formatHMS(seconds: secs)
    }

    /// Parse a duration spec of the form `<number>[s|m|h|d]`. Whitespace
    /// trimmed, case-insensitive on the suffix. An unqualified number
    /// is taken as seconds. Returns nil on any parse failure or
    /// non-positive result; consumers surface the parse error via the
    /// red-overlay-on-the-field pattern in the Arena popover form.
    static func parseDurationSpec(_ raw: String) -> Double? {
        let s = raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !s.isEmpty else { return nil }
        let suffixes: [(suffix: Character, mul: Double)] = [
            ("s", 1), ("m", 60), ("h", 3600), ("d", 86400)
        ]
        if let last = s.last, last.isLetter {
            guard let entry = suffixes.first(where: { $0.suffix == last }) else { return nil }
            let numericPart = String(s.dropLast())
            guard let n = Double(numericPart), n > 0, n.isFinite else { return nil }
            return n * entry.mul
        } else {
            guard let n = Double(s), n > 0, n.isFinite else { return nil }
            return n
        }
    }

    /// Format `seconds` back to the largest unit that divides cleanly,
    /// so a `1800`-second interval reads back as `30m` rather than
    /// `1800s` when the popover seeds its Interval text field.
    static func formatDurationSpec(_ seconds: Double) -> String {
        let s = max(0, seconds)
        if s >= 86400, s.truncatingRemainder(dividingBy: 86400) == 0 {
            return "\(Int(s / 86400))d"
        }
        if s >= 3600, s.truncatingRemainder(dividingBy: 3600) == 0 {
            return "\(Int(s / 3600))h"
        }
        if s >= 60, s.truncatingRemainder(dividingBy: 60) == 0 {
            return "\(Int(s / 60))m"
        }
        return "\(Int(s))s"
    }


    /// Seed the popover edit-text mirrors from the current
    /// `trainingParams` values. Called from the form's `.onAppear`,
    /// so opening the popover always reflects the live state, even if
    /// the user edited a parameter elsewhere (CLI, params file) since
    /// the last open.
    private func arenaPopoverSeedFromParams() {
        arenaPopoverGamesText = String(trainingParams.arenaGamesPerTournament)
        arenaPopoverConcurrencyText = String(trainingParams.arenaConcurrency)
        arenaPopoverIntervalText = Self.formatDurationSpec(trainingParams.arenaAutoIntervalSec)
        // Reuse the same edit-text @State that backs the inline
        // stats-panel tau row (now removed) so a single edit
        // location keeps `trainingParams` and `@State` in sync.
        arStartTauEditText = String(format: "%.2f", trainingParams.arenaStartTau)
        arDecayPerPlyEditText = String(format: "%.3f", trainingParams.arenaTauDecayPerPly)
        arFloorTauEditText = String(format: "%.2f", trainingParams.arenaTargetTau)
        arenaPopoverGamesError = false
        arenaPopoverConcurrencyError = false
        arenaPopoverIntervalError = false
        arenaPopoverTauStartError = false
        arenaPopoverTauDecayError = false
        arenaPopoverTauFloorError = false
    }

    /// Validate all three popover fields against their parameter
    /// ranges and write valid values back to `trainingParams`. On any
    /// parse failure the field's red-overlay flag is set and the
    /// popover stays open. On full success the popover dismisses.
    private func arenaPopoverSave() {
        var anyError = false

        let parsedGames = Int(arenaPopoverGamesText.trimmingCharacters(in: .whitespaces))
        if let g = parsedGames, g >= 4, g <= 10000 {
            arenaPopoverGamesError = false
            if g != trainingParams.arenaGamesPerTournament {
                trainingParams.arenaGamesPerTournament = g
            }
        } else {
            arenaPopoverGamesError = true
            anyError = true
        }

        let parsedConcurrency = Int(arenaPopoverConcurrencyText.trimmingCharacters(in: .whitespaces))
        if let c = parsedConcurrency, c >= 1, c <= Self.absoluteMaxArenaConcurrency {
            arenaPopoverConcurrencyError = false
            if c != trainingParams.arenaConcurrency {
                trainingParams.arenaConcurrency = c
            }
        } else {
            arenaPopoverConcurrencyError = true
            anyError = true
        }

        if let secs = Self.parseDurationSpec(arenaPopoverIntervalText), secs >= 60, secs <= 86400 {
            arenaPopoverIntervalError = false
            if secs != trainingParams.arenaAutoIntervalSec {
                trainingParams.arenaAutoIntervalSec = secs
            }
        } else {
            arenaPopoverIntervalError = true
            anyError = true
        }

        // tau Start — same range as the inline stats-panel
        // editor it replaced: (0, 10].
        if let v = Double(arStartTauEditText.trimmingCharacters(in: .whitespaces)),
           v > 0, v.isFinite, v <= 10 {
            arenaPopoverTauStartError = false
            if abs(v - trainingParams.arenaStartTau) > Double.ulpOfOne {
                SessionLogger.shared.log(
                    String(format: "[PARAM] ar.startTau: %.3f -> %.3f", trainingParams.arenaStartTau, v)
                )
                trainingParams.arenaStartTau = v
            }
        } else {
            arenaPopoverTauStartError = true
            anyError = true
        }

        // tau Decay — [0, 1].
        if let v = Double(arDecayPerPlyEditText.trimmingCharacters(in: .whitespaces)),
           v >= 0, v.isFinite, v <= 1 {
            arenaPopoverTauDecayError = false
            if abs(v - trainingParams.arenaTauDecayPerPly) > Double.ulpOfOne {
                SessionLogger.shared.log(
                    String(format: "[PARAM] ar.decayPerPly: %.4f -> %.4f", trainingParams.arenaTauDecayPerPly, v)
                )
                trainingParams.arenaTauDecayPerPly = v
            }
        } else {
            arenaPopoverTauDecayError = true
            anyError = true
        }

        // tau Floor — same range as Start: (0, 10].
        if let v = Double(arFloorTauEditText.trimmingCharacters(in: .whitespaces)),
           v > 0, v.isFinite, v <= 10 {
            arenaPopoverTauFloorError = false
            if abs(v - trainingParams.arenaTargetTau) > Double.ulpOfOne {
                SessionLogger.shared.log(
                    String(format: "[PARAM] ar.floorTau: %.3f -> %.3f", trainingParams.arenaTargetTau, v)
                )
                trainingParams.arenaTargetTau = v
            }
        } else {
            arenaPopoverTauFloorError = true
            anyError = true
        }
        // Push the freshly-edited arena schedule into the live
        // `samplingScheduleBox` so the next arena tournament picks
        // up the new tau curve. Without this push the box keeps
        // its session-start snapshot and updated `trainingParams`
        // values don't take effect until the next Play-and-Train
        // restart. (Pre-existing latent bug — the arena popover
        // never wired this through. Cleaned up alongside the
        // matching push from the new training popover.)
        samplingScheduleBox?.setArena(buildArenaSchedule())

        if !anyError { showArenaPopover = false }
    }

    /// Seed the `TrainingSettingsPopover`'s edit-text bindings from
    /// the live `trainingParams` snapshot. Called from
    /// `onAppearSeed` so the user always sees the current values
    /// when the popover opens, even if a CLI / parameters-file
    /// override edited them since the last open.
    private func trainingPopoverSeedFromParams() {
        // --- Optimizer tab ---
        learningRateEditText = String(format: "%.2e", trainingParams.learningRate)
        lrWarmupStepsEditText = String(trainingParams.lrWarmupSteps)
        momentumCoeffEditText = String(format: "%.3f", trainingParams.momentumCoeff)
        sqrtBatchScalingEditValue = trainingParams.sqrtBatchScalingLR
        entropyRegularizationEditText = String(format: "%.2e", trainingParams.entropyBonus)
        gradClipMaxNormEditText = String(format: "%.1f", trainingParams.gradClipMaxNorm)
        weightDecayEditText = String(format: "%.2e", trainingParams.weightDecay)
        policyScaleKEditText = String(format: "%.2f", trainingParams.policyScaleK)
        drawPenaltyEditText = String(format: "%.3f", trainingParams.drawPenalty)
        trainingBatchSizeEditText = String(trainingParams.trainingBatchSize)
        // --- Self Play tab ---
        selfPlayWorkersEditText = String(trainingParams.selfPlayWorkers)
        spStartTauEditText = String(format: "%.2f", trainingParams.selfPlayStartTau)
        spDecayPerPlyEditText = String(format: "%.3f", trainingParams.selfPlayTauDecayPerPly)
        spFloorTauEditText = String(format: "%.2f", trainingParams.selfPlayTargetTau)
        // --- Replay tab ---
        replayBufferCapacityEditText = String(trainingParams.replayBufferCapacity)
        replayBufferMinPositionsBeforeTrainingEditText = String(
            trainingParams.replayBufferMinPositionsBeforeTraining
        )
        replayRatioTargetEditText = String(format: "%.2f", trainingParams.replayRatioTarget)
        replaySelfPlayDelayEditText = String(trainingParams.selfPlayDelayMs)
        replayTrainingStepDelayEditText = String(trainingParams.trainingStepDelayMs)
        replayRatioAutoAdjustEditValue = trainingParams.replayRatioAutoAdjust
        // Stash pre-edit values for the four replay-ratio control
        // fields. The Replay tab live-propagates changes to those
        // fields; if the user hits Cancel we restore from this
        // stash, matching the standard "Cancel discards" mental
        // model from the user's POV even though the live writes
        // already reached `trainingParams`.
        originalReplayRatioTarget = trainingParams.replayRatioTarget
        originalReplaySelfPlayDelayMs = trainingParams.selfPlayDelayMs
        originalReplayTrainingStepDelayMs = trainingParams.trainingStepDelayMs
        originalReplayRatioAutoAdjust = trainingParams.replayRatioAutoAdjust
        // Reset every error flag — a fresh open should never carry
        // red overlays from a previously-cancelled bad input.
        trainingPopoverLRError = false
        trainingPopoverWarmupError = false
        trainingPopoverMomentumError = false
        trainingPopoverEntropyError = false
        trainingPopoverGradClipError = false
        trainingPopoverWeightDecayError = false
        trainingPopoverPolicyKError = false
        trainingPopoverDrawPenaltyError = false
        trainingPopoverTrainingBatchSizeError = false
        trainingPopoverSelfPlayWorkersError = false
        trainingPopoverSelfPlayStartTauError = false
        trainingPopoverSelfPlayDecayPerPlyError = false
        trainingPopoverSelfPlayFloorTauError = false
        trainingPopoverReplayBufferCapacityError = false
        trainingPopoverReplayBufferMinPositionsError = false
        trainingPopoverReplayRatioTargetError = false
        trainingPopoverReplaySelfPlayDelayError = false
        trainingPopoverReplayTrainingStepDelayError = false
    }

    /// Cancel handler for `TrainingSettingsPopover`. Restores the
    /// three live-propagated replay-ratio control fields from the
    /// stash captured in `trainingPopoverSeedFromParams()`, then
    /// dismisses the popover. Matches the user-facing "Cancel
    /// discards changes" pattern even though the underlying
    /// `trainingParams` writes already happened during the edit
    /// session — the revert here puts everything back without a
    /// confirmation prompt, by design. No `[PARAM]` log on revert
    /// (the original live-update writes were not logged either —
    /// see the Save path's commit-time logging for the source of
    /// truth).
    private func trainingPopoverCancel() {
        if abs(trainingParams.replayRatioTarget - originalReplayRatioTarget) > Double.ulpOfOne {
            trainingParams.replayRatioTarget = originalReplayRatioTarget
            // The `ControlSideEffectsProbe` watches
            // `trainingParams.replayRatioTarget` and pushes the new
            // value into the live `ReplayRatioController`, so this
            // single write is sufficient — no direct controller
            // call needed here.
        }
        if trainingParams.selfPlayDelayMs != originalReplaySelfPlayDelayMs {
            trainingParams.selfPlayDelayMs = originalReplaySelfPlayDelayMs
            replayRatioController?.manualSelfPlayDelayMs = originalReplaySelfPlayDelayMs
        }
        if trainingParams.trainingStepDelayMs != originalReplayTrainingStepDelayMs {
            trainingParams.trainingStepDelayMs = originalReplayTrainingStepDelayMs
            replayRatioController?.manualDelayMs = originalReplayTrainingStepDelayMs
        }
        if trainingParams.replayRatioAutoAdjust != originalReplayRatioAutoAdjust {
            trainingParams.replayRatioAutoAdjust = originalReplayRatioAutoAdjust
        }
        showTrainingPopover = false
    }

    /// Live-propagate the user's replay-ratio-target edit straight
    /// to `trainingParams.replayRatioTarget`. The
    /// `ControlSideEffectsProbe` watches that property and forwards
    /// the new value into the live `ReplayRatioController`'s
    /// `targetRatio`, so this single write is sufficient. Snapped
    /// to the parameter's `[0.1, 5.0]` range to match the slider
    /// validation in `trainingPopoverSave`.
    private func trainingPopoverApplyLiveReplayRatioTarget(_ newValue: Double) {
        guard newValue.isFinite else { return }
        let snapped = max(0.1, min(5.0, newValue))
        if abs(trainingParams.replayRatioTarget - snapped) > Double.ulpOfOne {
            trainingParams.replayRatioTarget = snapped
        }
    }

    /// Live-propagate the user's self-play-delay edit straight to
    /// `trainingParams.selfPlayDelayMs` and the live
    /// `ReplayRatioController`. Fired on every text-field commit
    /// or stepper press while the popover is open. On Cancel the
    /// parent restores the stashed pre-open value via
    /// `trainingPopoverCancel()`.
    private func trainingPopoverApplyLiveSelfPlayDelay(_ newValue: Int) {
        let snapped = max(0, min(Self.selfPlayDelayMaxMs, newValue))
        if trainingParams.selfPlayDelayMs != snapped {
            trainingParams.selfPlayDelayMs = snapped
            replayRatioController?.manualSelfPlayDelayMs = snapped
        }
    }

    /// Live-propagate the user's train-step-delay edit. Same
    /// rationale as `trainingPopoverApplyLiveSelfPlayDelay`. Also
    /// writes through to `replayRatioController.manualDelayMs`
    /// because that's what `recordTrainingBatchAndGetDelay` reads
    /// each training step — without this push the controller would
    /// keep returning the old delay despite `trainingParams` having
    /// the new one.
    private func trainingPopoverApplyLiveTrainingStepDelay(_ newValue: Int) {
        let snapped = max(0, min(Self.stepDelayMaxMs, newValue))
        if trainingParams.trainingStepDelayMs != snapped {
            trainingParams.trainingStepDelayMs = snapped
            replayRatioController?.manualDelayMs = snapped
        }
    }

    /// Live-propagate the auto-control checkbox toggle. The
    /// `ControlSideEffectsProbe` watches `replayRatioAutoAdjust`
    /// and on the OFF transition writes inherited last-auto values
    /// into `trainingParams.trainingStepDelayMs` / `selfPlayDelayMs`
    /// — that runs after this setter returns. We defer a re-seed
    /// of the popover's two delay text bindings to the next main-
    /// actor tick so the editable PopoverRow that appears when
    /// auto goes OFF shows the inherited values rather than the
    /// pre-toggle stash. Without the deferral the user sees the
    /// stale text and has to tap the Stepper before the probe's
    /// inherit becomes visible.
    private func trainingPopoverApplyLiveReplayRatioAutoAdjust(_ newValue: Bool) {
        if trainingParams.replayRatioAutoAdjust != newValue {
            trainingParams.replayRatioAutoAdjust = newValue
            if !newValue {
                Task { @MainActor in
                    replaySelfPlayDelayEditText = String(trainingParams.selfPlayDelayMs)
                    replayTrainingStepDelayEditText = String(trainingParams.trainingStepDelayMs)
                }
            }
        }
    }

    /// Validate every `TrainingSettingsPopover` field against its
    /// parameter range and write valid values back to
    /// `trainingParams` (and mirror to the live `trainer` for the
    /// optimizer-touching params). On any parse failure the
    /// affected field's red-overlay flag is set and the popover
    /// stays open. On full success the popover dismisses.
    ///
    /// Mirrors `arenaPopoverSave()`: `[PARAM] name: old -> new`
    /// log line on every actual change, no log when value is
    /// unchanged. Trainer side-write uses `Float(parsed)` for the
    /// optimizer floats, matching the inline-row pattern that this
    /// popover replaces.
    private func trainingPopoverSave() {
        var anyError = false

        // LR — Double in [1e-7, 1.0].
        if let v = Double(learningRateEditText.trimmingCharacters(in: .whitespaces)),
           v >= 1e-7, v <= 1.0, v.isFinite {
            trainingPopoverLRError = false
            if abs(v - trainingParams.learningRate) > Double.ulpOfOne {
                SessionLogger.shared.log(
                    String(format: "[PARAM] learningRate: %.3e -> %.3e", trainingParams.learningRate, v)
                )
                trainingParams.learningRate = v
                trainer?.learningRate = Float(v)
            }
        } else {
            trainingPopoverLRError = true
            anyError = true
        }

        // LR Warmup steps — Int in [0, 100_000].
        if let n = Int(lrWarmupStepsEditText.trimmingCharacters(in: .whitespaces)),
           n >= 0, n <= 100_000 {
            trainingPopoverWarmupError = false
            if n != trainingParams.lrWarmupSteps {
                SessionLogger.shared.log(
                    "[PARAM] lrWarmupSteps: \(trainingParams.lrWarmupSteps) -> \(n)"
                )
                trainingParams.lrWarmupSteps = n
                trainer?.lrWarmupSteps = n
            }
        } else {
            trainingPopoverWarmupError = true
            anyError = true
        }

        // Momentum — Double in [0, 0.99]. No inline UI before this
        // popover, so this is the only edit surface.
        if let v = Double(momentumCoeffEditText.trimmingCharacters(in: .whitespaces)),
           v >= 0.0, v <= 0.99, v.isFinite {
            trainingPopoverMomentumError = false
            if abs(v - trainingParams.momentumCoeff) > Double.ulpOfOne {
                SessionLogger.shared.log(
                    String(format: "[PARAM] momentumCoeff: %.3f -> %.3f", trainingParams.momentumCoeff, v)
                )
                trainingParams.momentumCoeff = v
                trainer?.momentumCoeff = Float(v)
            }
        } else {
            trainingPopoverMomentumError = true
            anyError = true
        }

        // √batch scaling toggle — Bool, cannot fail to parse.
        if sqrtBatchScalingEditValue != trainingParams.sqrtBatchScalingLR {
            SessionLogger.shared.log(
                "[PARAM] sqrtBatchScalingLR: \(trainingParams.sqrtBatchScalingLR) -> \(sqrtBatchScalingEditValue)"
            )
            trainingParams.sqrtBatchScalingLR = sqrtBatchScalingEditValue
        }

        // Entropy regularization — Double in [0, 0.1].
        if let v = Double(entropyRegularizationEditText.trimmingCharacters(in: .whitespaces)),
           v >= 0.0, v <= 0.1, v.isFinite {
            trainingPopoverEntropyError = false
            if abs(v - trainingParams.entropyBonus) > Double.ulpOfOne {
                SessionLogger.shared.log(
                    String(format: "[PARAM] entropyBonus: %.3e -> %.3e", trainingParams.entropyBonus, v)
                )
                trainingParams.entropyBonus = v
                trainer?.entropyRegularizationCoeff = Float(v)
            }
        } else {
            trainingPopoverEntropyError = true
            anyError = true
        }

        // Grad clip — Double in [0.1, 1000].
        if let v = Double(gradClipMaxNormEditText.trimmingCharacters(in: .whitespaces)),
           v >= 0.1, v <= 1000.0, v.isFinite {
            trainingPopoverGradClipError = false
            if abs(v - trainingParams.gradClipMaxNorm) > Double.ulpOfOne {
                SessionLogger.shared.log(
                    String(format: "[PARAM] gradClipMaxNorm: %.2f -> %.2f", trainingParams.gradClipMaxNorm, v)
                )
                trainingParams.gradClipMaxNorm = v
                trainer?.gradClipMaxNorm = Float(v)
            }
        } else {
            trainingPopoverGradClipError = true
            anyError = true
        }

        // Weight decay — Double in [0, 0.1].
        if let v = Double(weightDecayEditText.trimmingCharacters(in: .whitespaces)),
           v >= 0.0, v <= 0.1, v.isFinite {
            trainingPopoverWeightDecayError = false
            if abs(v - trainingParams.weightDecay) > Double.ulpOfOne {
                SessionLogger.shared.log(
                    String(format: "[PARAM] weightDecay: %.3e -> %.3e", trainingParams.weightDecay, v)
                )
                trainingParams.weightDecay = v
                trainer?.weightDecayC = Float(v)
            }
        } else {
            trainingPopoverWeightDecayError = true
            anyError = true
        }

        // Policy scale K — Double in [0.1, 20].
        if let v = Double(policyScaleKEditText.trimmingCharacters(in: .whitespaces)),
           v >= 0.1, v <= 20.0, v.isFinite {
            trainingPopoverPolicyKError = false
            if abs(v - trainingParams.policyScaleK) > Double.ulpOfOne {
                SessionLogger.shared.log(
                    String(format: "[PARAM] policyScaleK: %.2f -> %.2f", trainingParams.policyScaleK, v)
                )
                trainingParams.policyScaleK = v
                trainer?.policyScaleK = Float(v)
            }
        } else {
            trainingPopoverPolicyKError = true
            anyError = true
        }

        // Draw penalty — Double in [-1, 1].
        if let v = Double(drawPenaltyEditText.trimmingCharacters(in: .whitespaces)),
           v >= -1.0, v <= 1.0, v.isFinite {
            trainingPopoverDrawPenaltyError = false
            if abs(v - trainingParams.drawPenalty) > Double.ulpOfOne {
                SessionLogger.shared.log(
                    String(format: "[PARAM] drawPenalty: %.3f -> %.3f", trainingParams.drawPenalty, v)
                )
                trainingParams.drawPenalty = v
                trainer?.drawPenalty = Float(v)
            }
        } else {
            trainingPopoverDrawPenaltyError = true
            anyError = true
        }

        // Training batch size — Int in [32, 32_768]. Snapshot-only;
        // the live trainer rebuilds its feed cache lazily on the
        // next batch shape it sees, so the change takes effect at
        // the next training step without an explicit trainer write.
        if let n = Int(trainingBatchSizeEditText.trimmingCharacters(in: .whitespaces)),
           n >= 32, n <= 32_768 {
            trainingPopoverTrainingBatchSizeError = false
            if n != trainingParams.trainingBatchSize {
                SessionLogger.shared.log(
                    "[PARAM] trainingBatchSize: \(trainingParams.trainingBatchSize) -> \(n)"
                )
                trainingParams.trainingBatchSize = n
            }
        } else {
            trainingPopoverTrainingBatchSizeError = true
            anyError = true
        }

        // Self-play workers — Int in [1, absoluteMaxSelfPlayWorkers].
        // Live-tunable: the BatchedSelfPlayDriver reconcile loop
        // picks up the new count on its next reconcile tick.
        if let n = Int(selfPlayWorkersEditText.trimmingCharacters(in: .whitespaces)),
           n >= 1, n <= Self.absoluteMaxSelfPlayWorkers {
            trainingPopoverSelfPlayWorkersError = false
            if n != trainingParams.selfPlayWorkers {
                SessionLogger.shared.log(
                    "[PARAM] selfPlayWorkers: \(trainingParams.selfPlayWorkers) -> \(n)"
                )
                trainingParams.selfPlayWorkers = n
            }
        } else {
            trainingPopoverSelfPlayWorkersError = true
            anyError = true
        }

        // Self-play tau schedule — Doubles in [0.01, 5.0] (start /
        // floor) and [0.0, 1.0] (decay). The schedule is rebuilt
        // by `buildSelfPlaySchedule()` next time the schedule box
        // is constructed; mid-session changes don't retroactively
        // alter games already in progress.
        if let v = Double(spStartTauEditText.trimmingCharacters(in: .whitespaces)),
           v >= 0.01, v <= 5.0, v.isFinite {
            trainingPopoverSelfPlayStartTauError = false
            if abs(v - trainingParams.selfPlayStartTau) > Double.ulpOfOne {
                SessionLogger.shared.log(
                    String(format: "[PARAM] selfPlayStartTau: %.2f -> %.2f", trainingParams.selfPlayStartTau, v)
                )
                trainingParams.selfPlayStartTau = v
            }
        } else {
            trainingPopoverSelfPlayStartTauError = true
            anyError = true
        }
        if let v = Double(spDecayPerPlyEditText.trimmingCharacters(in: .whitespaces)),
           v >= 0.0, v <= 1.0, v.isFinite {
            trainingPopoverSelfPlayDecayPerPlyError = false
            if abs(v - trainingParams.selfPlayTauDecayPerPly) > Double.ulpOfOne {
                SessionLogger.shared.log(
                    String(format: "[PARAM] selfPlayTauDecayPerPly: %.3f -> %.3f", trainingParams.selfPlayTauDecayPerPly, v)
                )
                trainingParams.selfPlayTauDecayPerPly = v
            }
        } else {
            trainingPopoverSelfPlayDecayPerPlyError = true
            anyError = true
        }
        if let v = Double(spFloorTauEditText.trimmingCharacters(in: .whitespaces)),
           v >= 0.01, v <= 5.0, v.isFinite {
            trainingPopoverSelfPlayFloorTauError = false
            if abs(v - trainingParams.selfPlayTargetTau) > Double.ulpOfOne {
                SessionLogger.shared.log(
                    String(format: "[PARAM] selfPlayTargetTau: %.2f -> %.2f", trainingParams.selfPlayTargetTau, v)
                )
                trainingParams.selfPlayTargetTau = v
            }
        } else {
            trainingPopoverSelfPlayFloorTauError = true
            anyError = true
        }
        // Push the freshly-edited self-play schedule into the live
        // `samplingScheduleBox` so the next self-play game on each
        // worker slot picks up the new tau curve. Without this push
        // the box would keep its session-start snapshot and the
        // updated `trainingParams` values wouldn't take effect
        // until the next Play-and-Train start. (The box's own
        // `setSelfPlay` is a no-op when `samplingScheduleBox` is
        // nil — i.e. before the first session — so the call is
        // safe to make unconditionally.)
        samplingScheduleBox?.setSelfPlay(buildSelfPlaySchedule())

        // Replay buffer capacity — Int in [1024, 100_000_000].
        // Snapshot-only: the live ReplayBuffer ring cannot resize
        // mid-session, so this value takes effect at the next
        // session start.
        if let n = Int(replayBufferCapacityEditText.trimmingCharacters(in: .whitespaces)),
           n >= 1024, n <= 100_000_000 {
            trainingPopoverReplayBufferCapacityError = false
            if n != trainingParams.replayBufferCapacity {
                SessionLogger.shared.log(
                    "[PARAM] replayBufferCapacity: \(trainingParams.replayBufferCapacity) -> \(n)"
                )
                trainingParams.replayBufferCapacity = n
            }
        } else {
            trainingPopoverReplayBufferCapacityError = true
            anyError = true
        }

        // Pre-train fill threshold — Int in [0, 100_000_000]. Live-
        // tunable; the gate that holds the trainer until the
        // buffer fills past this many positions reads from
        // trainingParams every check.
        if let n = Int(replayBufferMinPositionsBeforeTrainingEditText.trimmingCharacters(in: .whitespaces)),
           n >= 0, n <= 100_000_000 {
            trainingPopoverReplayBufferMinPositionsError = false
            if n != trainingParams.replayBufferMinPositionsBeforeTraining {
                SessionLogger.shared.log(
                    "[PARAM] replayBufferMinPositionsBeforeTraining: \(trainingParams.replayBufferMinPositionsBeforeTraining) -> \(n)"
                )
                trainingParams.replayBufferMinPositionsBeforeTraining = n
            }
        } else {
            trainingPopoverReplayBufferMinPositionsError = true
            anyError = true
        }

        // Replay-ratio control fields (replayRatioTarget,
        // selfPlayDelayMs, trainingStepDelayMs, replayRatioAutoAdjust)
        // are live-propagated during edits via
        // `trainingPopoverApplyLive…` — the writes already reached
        // `trainingParams`. Save validates the current text values
        // for red-overlay display only; no parameter writes here.
        if let v = Double(replayRatioTargetEditText.trimmingCharacters(in: .whitespaces)),
           v >= 0.1, v <= 5.0, v.isFinite {
            trainingPopoverReplayRatioTargetError = false
        } else {
            trainingPopoverReplayRatioTargetError = true
            anyError = true
        }
        if let n = Int(replaySelfPlayDelayEditText.trimmingCharacters(in: .whitespaces)),
           n >= 0, n <= Self.selfPlayDelayMaxMs {
            trainingPopoverReplaySelfPlayDelayError = false
        } else {
            trainingPopoverReplaySelfPlayDelayError = true
            anyError = true
        }
        if let n = Int(replayTrainingStepDelayEditText.trimmingCharacters(in: .whitespaces)),
           n >= 0, n <= 10_000 {
            trainingPopoverReplayTrainingStepDelayError = false
        } else {
            trainingPopoverReplayTrainingStepDelayError = true
            anyError = true
        }

        if !anyError {
            // Commit-time [PARAM] log lines for the four live-
            // propagated replay-ratio fields. The live writes
            // during the edit session are intentionally silent
            // (a log per keystroke would be noise); this is the
            // single authoritative log line per Save, mirroring
            // the rest of the trainingPopoverSave fields.
            if abs(trainingParams.replayRatioTarget - originalReplayRatioTarget) > Double.ulpOfOne {
                SessionLogger.shared.log(
                    String(
                        format: "[PARAM] replayRatioTarget: %.2f -> %.2f",
                        originalReplayRatioTarget,
                        trainingParams.replayRatioTarget
                    )
                )
            }
            if trainingParams.selfPlayDelayMs != originalReplaySelfPlayDelayMs {
                SessionLogger.shared.log(
                    "[PARAM] selfPlayDelayMs: \(originalReplaySelfPlayDelayMs) -> \(trainingParams.selfPlayDelayMs)"
                )
            }
            if trainingParams.trainingStepDelayMs != originalReplayTrainingStepDelayMs {
                SessionLogger.shared.log(
                    "[PARAM] trainingStepDelayMs: \(originalReplayTrainingStepDelayMs) -> \(trainingParams.trainingStepDelayMs)"
                )
            }
            if trainingParams.replayRatioAutoAdjust != originalReplayRatioAutoAdjust {
                SessionLogger.shared.log(
                    "[PARAM] replayRatioAutoAdjust: \(originalReplayRatioAutoAdjust) -> \(trainingParams.replayRatioAutoAdjust)"
                )
            }
            // On successful save, the stash that backs Cancel
            // becomes the new "pre-edit" baseline — closing the
            // popover with Save commits the live ratio writes.
            originalReplayRatioTarget = trainingParams.replayRatioTarget
            originalReplaySelfPlayDelayMs = trainingParams.selfPlayDelayMs
            originalReplayTrainingStepDelayMs = trainingParams.trainingStepDelayMs
            originalReplayRatioAutoAdjust = trainingParams.replayRatioAutoAdjust
            showTrainingPopover = false
        }
    }

    /// Backfill `finishedAt` / `candidateID` / `championID` on
    /// any `tournamentHistory` entries that lack them, by scanning
    /// the per-launch session logs at
    /// `~/Library/Logs/DrewsChessMachine/`. Triggered by the
    /// "Recover from logs" button in `ArenaHistoryView`.
    ///
    /// Match strategy: each `[ARENA] #N kv step=…` log line is
    /// keyed by training-step. We accept a recovery if and only
    /// if the log line's W/D/L counts match the saved record's
    /// — same step but different counts means the saved record
    /// belongs to a different arena run (e.g. a re-run after a
    /// crash that didn't survive the next save), and silently
    /// merging would corrupt history.
    ///
    /// The scan runs on a detached background task. After a
    /// successful merge the in-memory `tournamentHistory` is
    /// replaced with the recovered version and a save-button log
    /// line is emitted; the user can hit Save Session (or wait
    /// for the periodic autosave) to persist. We deliberately
    /// don't auto-save here so the recovery is reversible from
    /// the user's perspective until they confirm.
    @MainActor
    private func runArenaHistoryRecovery() {
        guard !arenaRecoveryInProgress else { return }
        arenaRecoveryInProgress = true
        SessionLogger.shared.log("[BUTTON] Recover Arena History from logs (start)")

        Task.detached(priority: .userInitiated) { [tournamentHistory] in
            let logsDir: URL
            do {
                logsDir = try ArenaLogRecovery.defaultLogsDirectory()
            } catch {
                await MainActor.run {
                    SessionLogger.shared.log(
                        "[RECOVER] failed: cannot resolve logs directory: \(error.localizedDescription)"
                    )
                    self.arenaRecoveryInProgress = false
                }
                return
            }
            let recovered = ArenaLogRecovery.scan(logsDirectory: logsDir)

            // Merge in-place on a copy; only adopt the new array
            // if at least one record actually changed.
            var updated = tournamentHistory
            var changedCount = 0
            for (i, record) in tournamentHistory.enumerated() {
                guard let hit = recovered[record.finishedAtStep] else { continue }
                guard hit.candidateWins == record.candidateWins,
                      hit.draws == record.draws,
                      hit.championWins == record.championWins else {
                    continue
                }
                let newFinishedAt = record.finishedAt ?? hit.finishedAt
                let newCandidateID = record.candidateID
                    ?? hit.candidateID.map { ModelID(value: $0) }
                let newChampionID = record.championID
                    ?? hit.championID.map { ModelID(value: $0) }
                if newFinishedAt != record.finishedAt
                    || newCandidateID != record.candidateID
                    || newChampionID != record.championID {
                    updated[i] = TournamentRecord(
                        finishedAtStep: record.finishedAtStep,
                        finishedAt: newFinishedAt,
                        candidateID: newCandidateID,
                        championID: newChampionID,
                        gamesPlayed: record.gamesPlayed,
                        candidateWins: record.candidateWins,
                        championWins: record.championWins,
                        draws: record.draws,
                        score: record.score,
                        promoted: record.promoted,
                        promotionKind: record.promotionKind,
                        promotedID: record.promotedID,
                        durationSec: record.durationSec,
                        candidateWinsAsWhite: record.candidateWinsAsWhite,
                        candidateWinsAsBlack: record.candidateWinsAsBlack,
                        candidateLossesAsWhite: record.candidateLossesAsWhite,
                        candidateLossesAsBlack: record.candidateLossesAsBlack,
                        candidateDrawsAsWhite: record.candidateDrawsAsWhite,
                        candidateDrawsAsBlack: record.candidateDrawsAsBlack
                    )
                    changedCount += 1
                }
            }

            await MainActor.run {
                if changedCount > 0 {
                    self.tournamentHistory = updated
                }
                SessionLogger.shared.log(
                    "[RECOVER] Arena history scan: \(recovered.count) kv lines mapped, \(changedCount) records updated (of \(tournamentHistory.count) total). Save the session to persist."
                )
                self.arenaRecoveryInProgress = false
            }
        }
    }

    // The chart-zoom control row moved to `LowerContentView`
    // alongside the chart grid it controls. Keyboard shortcuts and
    // menu items still route through this view's `chartZoomIn()` /
    // `chartZoomOut()` / `chartZoomEnableAuto()` wrappers above so
    // the menu hub plumbing stays in one place.

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

        func check(_ name: String, _ predicate: () throws -> Bool) {
            ran += 1
            do {
                if try predicate() {
                    SessionLogger.shared.log("[DIAG] PASS  \(name)")
                } else {
                    SessionLogger.shared.log("[DIAG] FAIL  \(name)")
                    failed += 1
                }
            } catch {
                SessionLogger.shared.log("[DIAG] FAIL  \(name): \(error.localizedDescription)")
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
                _ = try engine.applyMoveAndAdvance(nf3)
                _ = try engine.applyMoveAndAdvance(nc6)
                _ = try engine.applyMoveAndAdvance(ng1)
                _ = try engine.applyMoveAndAdvance(nb8)
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

    /// Run a one-shot "is the policy head producing position-conditional
    /// output?" probe. Two very different positions go through the live
    /// champion network in inference mode; the policy outputs are
    /// compared for L1 distance, max single-cell |Δ|, and value-head Δ.
    /// If the policy outputs are essentially identical (avg per-cell
    /// |Δ| < 1e-4) the policy head has collapsed to a position-agnostic
    /// constant — that's the symptom we've been chasing in the masked
    /// CE / entropy debugging. Healthy networks emit meaningfully
    /// different policies for unrelated boards.
    private func runPolicyConditioningDiagnostic() {
        SessionLogger.shared.log("[BUTTON] Policy Conditioning Probe")
        let networkRef = network
        Task {
            await runPolicyConditioningDiagnosticAsync(net: networkRef)
        }
    }

    private func runPolicyConditioningDiagnosticAsync(net: ChessMPSNetwork?) async {
        SessionLogger.shared.log("[DIAG] === Policy-conditioning probe begin ===")
        guard let net else {
            SessionLogger.shared.log("[DIAG] SKIP  No network built yet")
            SessionLogger.shared.log("[DIAG] === Policy-conditioning probe done ===")
            return
        }

        // Position 1: white-to-move starting position. Plain, common,
        // policy is well-defined.
        let pos1 = GameState.starting

        // Position 2: a midgame-ish black-to-move position with very
        // different piece layout — different side to move, different
        // material, different square occupancies — so every input plane
        // looks different from pos1.
        var midboard: [Piece?] = Array(repeating: nil, count: 64)
        midboard[0 * 8 + 4] = Piece(type: .king, color: .black)
        midboard[7 * 8 + 4] = Piece(type: .king, color: .white)
        midboard[3 * 8 + 3] = Piece(type: .queen, color: .black)
        midboard[4 * 8 + 5] = Piece(type: .knight, color: .white)
        midboard[2 * 8 + 6] = Piece(type: .rook, color: .white)
        midboard[5 * 8 + 1] = Piece(type: .bishop, color: .black)
        let pos2 = GameState(
            board: midboard, currentPlayer: .black,
            whiteKingsideCastle: false, whiteQueensideCastle: false,
            blackKingsideCastle: false, blackQueensideCastle: false,
            enPassantSquare: nil, halfmoveClock: 0
        )

        do {
            let board1 = BoardEncoder.encode(pos1)
            let board2 = BoardEncoder.encode(pos2)
            let (policy1, value1) = try await net.evaluate(board: board1)
            let (policy2, value2) = try await net.evaluate(board: board2)

            guard policy1.count == policy2.count else {
                SessionLogger.shared.log(
                    "[DIAG] FAIL  policy length mismatch: \(policy1.count) vs \(policy2.count)"
                )
                SessionLogger.shared.log("[DIAG] === Policy-conditioning probe done ===")
                return
            }

            // Per-position summary stats.
            let mean1 = policy1.reduce(Float(0), +) / Float(policy1.count)
            let mean2 = policy2.reduce(Float(0), +) / Float(policy2.count)
            var var1: Float = 0
            var var2: Float = 0
            for v in policy1 { var1 += (v - mean1) * (v - mean1) }
            for v in policy2 { var2 += (v - mean2) * (v - mean2) }
            let std1 = (var1 / Float(policy1.count)).squareRoot()
            let std2 = (var2 / Float(policy2.count)).squareRoot()

            // L1 distance + max single-cell |Δ| + per-cell average.
            var l1: Double = 0
            var maxAbsDiff: Double = 0
            var maxAbsIdx: Int = 0
            for i in 0..<policy1.count {
                let d = Double(abs(policy1[i] - policy2[i]))
                l1 += d
                if d > maxAbsDiff {
                    maxAbsDiff = d
                    maxAbsIdx = i
                }
            }
            let avgPerCellDiff = l1 / Double(policy1.count)

            SessionLogger.shared.log(
                String(
                    format: "[DIAG]   pos1: mean=%+0.4f std=%.4f, pos2: mean=%+0.4f std=%.4f",
                    mean1, std1, mean2, std2
                )
            )
            SessionLogger.shared.log(
                String(
                    format: "[DIAG]   policy Δ: L1=%.3f, maxAbs=%.4f at idx=%d, avg per-cell |Δ|=%.6f",
                    l1, maxAbsDiff, maxAbsIdx, avgPerCellDiff
                )
            )
            SessionLogger.shared.log(
                String(
                    format: "[DIAG]   value Δ: pos1=%+0.4f, pos2=%+0.4f, Δ=%+0.4f",
                    value1, value2, value1 - value2
                )
            )

            // Pass criterion: avg per-cell |Δ| above the noise floor
            // (1e-4 is generous — a randomly-initialized network with
            // logit std ~2.5 should easily produce avg |Δ| ≈ 1.0+ on
            // unrelated inputs). Below 1e-4 means the policy head's
            // output is effectively independent of the input — exactly
            // the failure mode we've been hypothesizing.
            let policyConditional = avgPerCellDiff >= 1e-4
            if policyConditional {
                SessionLogger.shared.log(
                    String(
                        format: "[DIAG] PASS  Policy head is position-conditional (avg per-cell |Δ| = %.6f, threshold ≥ 1e-4)",
                        avgPerCellDiff
                    )
                )
            } else {
                SessionLogger.shared.log(
                    String(
                        format: "[DIAG] FAIL  Policy head appears position-AGNOSTIC (avg per-cell |Δ| = %.6f, < 1e-4 threshold) — outputs are effectively the same regardless of input",
                        avgPerCellDiff
                    )
                )
            }

            // Also probe the value head: a healthy value head should
            // give meaningfully different scalars for two unrelated
            // positions. A pinned-to-zero value head would give |Δ|≈0.
            let valueDelta = abs(Double(value1 - value2))
            if valueDelta < 1e-4 {
                SessionLogger.shared.log(
                    String(
                        format: "[DIAG] WARN  Value head also looks position-agnostic (|Δ|=%.6f)",
                        valueDelta
                    )
                )
            }
        } catch {
            SessionLogger.shared.log(
                "[DIAG] FAIL  Policy probe error: \(error.localizedDescription)"
            )
        }

        SessionLogger.shared.log("[DIAG] === Policy-conditioning probe done ===")
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

    /// Format a moves/hour rate with an SI-ish suffix and the units
    /// baked in, e.g. `9.26M moves/hr`, `523.4K moves/hr`,
    /// `842 moves/hr`. Used by the "Training rate" status-bar cell
    /// so the cell's value string is self-describing.
    static func formatMovesPerHour(_ value: Double) -> String {
        let abs = Swift.abs(value)
        if abs >= 1_000_000 {
            return String(format: "%.2fM moves/hr", value / 1_000_000)
        } else if abs >= 1_000 {
            return String(format: "%.1fK moves/hr", value / 1_000)
        } else {
            return String(format: "%.0f moves/hr", value)
        }
    }

    /// Value for the status bar's "Training rate" cell: the most
    /// recent 3-minute-rolling trainer moves/hour, or `"—"` before
    /// any sample has been recorded this session. `chartCoordinator.progressRateRing`
    /// is populated at 1 Hz while `realTraining == true`; after Stop
    /// the last value persists (until the next fresh Play-and-Train
    /// start clears the buffer), which matches the rest of the row's
    /// "last-known value" semantics.
    private var trainingRateStatusValue: String {
        guard let rate = chartCoordinator.progressRateRing.last?.trainingMovesPerHour else {
            return "—"
        }
        return Self.formatMovesPerHour(rate)
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
        lines.append("  Batch size:  \(trainingParams.trainingBatchSize)")
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
            let bufCap = replayBuffer?.capacity ?? trainingParams.replayBufferCapacity
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
            // Take a single snapshot for all diagnostic lines below so
            // they're all reading the same moment rather than racing
            // against independent box reads.
            let diagSnap = trainingBox?.snapshot()
            if let gNorm = diagSnap?.rollingGradGlobalNorm {
                lines.append(String(format: "  Grad norm:   %.3f", gNorm))
            }
            if let pwNorm = diagSnap?.rollingPolicyHeadWeightNorm {
                lines.append(String(format: "  pWeight ||₂:  %.3f", pwNorm))
            }
            if let pLogitMax = diagSnap?.rollingPolicyLogitAbsMax {
                lines.append(String(format: "  pLogit |max|: %.3f", pLogitMax))
            }
            if let playedProb = diagSnap?.rollingPlayedMoveProb {
                lines.append(String(format: "  p(played):   %.4f", playedProb))
            }
            if let posProb = diagSnap?.rollingPlayedMoveProbPosAdv,
               let negProb = diagSnap?.rollingPlayedMoveProbNegAdv,
               let diag = diagSnap {
                lines.append(String(
                    format: "  p(played|A): +%.4f / -%.4f  skip=%d/%d,%d/%d",
                    posProb, negProb,
                    diag.rollingPlayedMoveProbPosAdvSkipped,
                    diag.rollingPlayedMoveCondWindowSize,
                    diag.rollingPlayedMoveProbNegAdvSkipped,
                    diag.rollingPlayedMoveCondWindowSize
                ))
            }
            // Advantage distribution — per-batch mean / std / signed
            // fractions show whether the fresh-baseline is centering
            // correctly. p05/p50/p95 are percentiles of raw advantage
            // values over the rolling window.
            if let advMean = diagSnap?.rollingAdvMean,
               let advStd = diagSnap?.rollingAdvStd {
                let posStr = diagSnap?.rollingAdvFracPositive
                    .map { String(format: "%.2f", $0) } ?? dash
                let smallStr = diagSnap?.rollingAdvFracSmall
                    .map { String(format: "%.2f", $0) } ?? dash
                lines.append(String(
                    format: "  Adv μ/σ:      %+.4f / %.4f  frac+=%@ fracSmall=%@",
                    advMean, advStd, posStr, smallStr
                ))
                if let advMin = diagSnap?.rollingAdvMin,
                   let advMax = diagSnap?.rollingAdvMax {
                    lines.append(String(
                        format: "  Adv [min,max]: [%+.3f, %+.3f]", advMin, advMax
                    ))
                }
                if let p05 = diagSnap?.advantageP05,
                   let p50 = diagSnap?.advantageP50,
                   let p95 = diagSnap?.advantageP95 {
                    lines.append(String(
                        format: "  Adv pct (05/50/95): %+.3f / %+.3f / %+.3f",
                        p05, p50, p95
                    ))
                }
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
            // Steps actually completed in *this* segment — the trainer
            // is seeded with the resumed lifetime count so `stats.steps`
            // alone is lifetime, but `totalStepMs` resets each segment.
            // Subtract the segment-start baseline so Avg total and
            // Steps/sec describe the live segment, not a phantom
            // ratio of this-segment-ms over lifetime-steps.
            let segmentSteps = max(0, stats.steps - trainingStepsAtSegmentStart)
            // Train time is the sum of per-step wall times — wall time
            // the trainer actually spent inside `trainStep`, exclusive
            // of buffer warmup, gate pauses, and any other idle gaps.
            // Useful as "cumulative GPU training cost"; intentionally
            // not the rate denominator below.
            let trainTimeStr = segmentSteps > 0
            ? String(format: "%.2f s", stats.trainingSeconds)
            : dash
            // Avg total is computed against `segmentSteps` rather than
            // `stats.steps` so a resumed session displays the live
            // per-step wall time, not totalStepMs/lifetime-steps.
            let avgTotal: String
            if segmentSteps > 0 {
                let avgMs = stats.totalStepMs / Double(segmentSteps)
                avgTotal = String(format: "%7.2f ms", avgMs)
            } else {
                avgTotal = dash
            }

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

            let stepsPerSec: Double = segmentSteps > 0
            ? Double(segmentSteps) / rateDenomSec
            : 0
            let movesPerSec: Double = segmentSteps > 0
            ? Double(segmentSteps * trainingParams.trainingBatchSize) / rateDenomSec
            : 0

            let rateStr = segmentSteps > 0 ? String(format: "%.2f", stepsPerSec) : dash
            let movesSecStr: String
            let movesHrStr: String
            if segmentSteps > 0 {
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

        // Arena history used to be appended here as a multi-line
        // block. It now lives in the Arena History sheet (opened
        // via the History button in the Arena settings popover) so
        // the stats text panel stays focused on live training
        // counters and doesn't grow unboundedly with every
        // tournament.

        return (header: header, body: lines.joined(separator: "\n"))
    }

    /// Play and Train self-play stats text. Built from the aggregate
    /// `ParallelWorkerStatsBox` snapshot so all N workers contribute
    /// identically, plus the live `GameWatcher` snapshot used only
    /// when `trainingParams.selfPlayWorkers == 1` to render the current-game
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
        if trainingParams.selfPlayWorkers == 1 {
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
        var rawInference: ChessRunner.InferenceResult? = nil
        let board = BoardEncoder.encode(state)

        do {
            let inference = try await runner.evaluate(board: board, state: state, pieces: state.board)
            topMoves = inference.topMoves
            rawInference = inference

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
            lines.append(String(
                format: "  Legal moves above uniform (%.3f%%): %d / %d  (threshold = 1/legalCount = 1/%d)",
                Double(legalUniformThreshold) * 100,
                abovePerLegalCount, nLegal, nLegal
            ))
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

        return EvaluationResult(
            topMoves: topMoves,
            textOutput: lines.joined(separator: "\n"),
            inputTensor: board,
            rawInference: rawInference
        )
    }
}

// MARK: - Body subviews
//
// Extracted out of `body` so each chunk type-checks independently. Before
// these existed, `body` was ~1020 lines of nested generics and clocked in
// at ~16 seconds in the type-checker (`-warn-long-function-bodies=100`
// flagged it on every clean build). Each extracted property is its own
// type-check unit, so the compiler solves them independently and `body`
// shrinks to a flat composition of named pieces.
//
// Properties that need `$trainingParams.<name>` projections take a local
// `@Bindable` shadow; everything else stays a plain `@ViewBuilder`. None
// of these accept parameters — they read directly from the surrounding
// `UpperContentView`'s state, mirroring how the original inline code worked.
extension UpperContentView {

    var selfPlayStatsColumn: some View {
        let column: (header: String, body: String)?
        if realTraining, let session = parallelStats {
            column = playAndTrainStatsText(game: gameSnapshot, session: session)
        } else {
            column = nil
        }
        return SelfPlayStatsColumn(
            realTrainingColumn: column,
            fallbackText: gameSnapshot.statsText(
                continuousPlay: continuousPlay || realTraining
            ),
            colorize: { colorizedPanelBody($0) }
        )
    }

    var trainingStatsColumn: some View {
        @Bindable var trainingParams = self.trainingParams
        let column = trainingStatsText()
        return TrainingStatsColumn(
            header: column.header,
            bodyText: colorizedPanelBody(column.body),
            realTraining: realTraining
        ) {
            // Read-only replay-ratio display. Editable controls
            // (Step Delay / SP Delay / Auto toggle / Target Ratio
            // stepper) moved to the Training Settings popover's
            // Replay tab. The user explicitly asked to keep this
            // view-only readout on the main screen so the live
            // ratio remains glanceable without opening the popover.
            //
            // SP tau, Buffer Cap, Prefill rows previously here also
            // moved to the popover (Self Play and Replay tabs).
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
                Text(String(format: "%.2f", trainingParams.replayRatioTarget))
                    .monospacedDigit()
                    .frame(minWidth: 32, alignment: .trailing)
                if trainingParams.replayRatioAutoAdjust {
                    Text("(auto)")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
        }
    }

}
