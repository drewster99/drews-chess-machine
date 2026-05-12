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

    /// Live recorder for `--output` runs. Moved to SessionController in
    /// Stage 4h — forwarding proxy. (Allocated at the start of startRealTraining
    /// when cliOutputURL is set, appended to by the stats/arena/probe paths
    /// while the session is active, nil in normal interactive runs — each
    /// capture site guards on `!= nil`.)
    private var cliRecorder: CliTrainingRecorder? {
        get { session.cliRecorder } nonmutating set { session.cliRecorder = newValue }
    }

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

    /// Session-lifecycle controller. Owns the champion `network` / `runner` /
    /// `networkStatus` / `isBuilding`, the build flow (`buildNetwork()` /
    /// `ensureChampionBuilt()`), the three life-of-app inference networks
    /// (`candidateInferenceNetwork` / `arenaChampionNetwork` /
    /// `probeInferenceNetwork` + `probeRunner`), the parallel-worker stats and
    /// arena coordination boxes. The training + arena orchestration migrate
    /// onto it in later Stage 4 slices.
    @State private var session = SessionController()

    // Network — these forward to `session` so the ~150 existing `network` /
    // `runner` / `networkStatus` / `isBuilding` references keep working
    // unchanged while the source of truth lives on `SessionController`. They
    // can be inlined to `session.…` in a later cleanup pass.
    private var network: ChessMPSNetwork? {
        get { session.network }
        nonmutating set { session.network = newValue }
    }
    private var runner: ChessRunner? {
        get { session.runner }
        nonmutating set { session.runner = newValue }
    }
    private var networkStatus: String {
        get { session.networkStatus }
        nonmutating set { session.networkStatus = newValue }
    }
    private var isBuilding: Bool {
        get { session.isBuilding }
        nonmutating set { session.isBuilding = newValue }
    }
    // Legacy `secondarySelfPlayNetworks` removed — all self-play
    // workers now share the champion network (`network`) through a
    // `BatchedMoveEvaluationSource` barrier batcher, so N per-worker
    // inference networks are no longer needed.
    // The parallel-worker stats (parallelStats / parallelWorkerStatsBox), the
    // self-play diversity tracker (selfPlayDiversityTracker), and the arena
    // coordination boxes (arenaActiveFlag / arenaTriggerBox / arenaOverrideBox
    // / isArenaRunning) moved to SessionController in Stage 4b — accessed via
    // `session.…`. (Diversity histogram, completed arena events, and the live
    // arena-start marker still live on `chartCoordinator` — see
    // ChartCoordinator.swift — and LowerContentView reads them from there.)

    /// View-menu toggle: when on, the 76-channel policy panel is
    /// rendered to the right of the chess board, sourced from
    /// whatever inference result is currently driving the on-board
    /// Top Moves overlay (Forward Pass / Candidate Test).
    @AppStorage("showPolicyChannelsPanel") private var showPolicyChannelsPanel: Bool = false

    /// Square (0..<64, row*8+col) the cursor is currently hovering
    /// on the main chess board. Drives the top-3-channels-at-this-
    /// square overlay that replaces the right-hand `MainTextPanel`
    /// during hover. nil = cursor is off the board.
    @State private var hoveredBoardSquare: Int? = nil

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
    // The candidate-test probe state (playAndTrainBoardMode / probeNetworkTarget
    // / candidateProbeDirty / lastCandidateProbeTime / candidateProbeCount) and
    // the `cliRecorder` handle moved to SessionController in Stage 4h —
    // forwarding proxies below. (playAndTrainBoardMode: which board the
    // Play-and-Train view shows — .gameRun = the live self-play game,
    // .candidateTest = the editable forward-pass board the user watches evolve.
    // probeNetworkTarget: .candidate syncs trainer weights into the probe net,
    // .champion probes the frozen champion directly. candidateProbeDirty: set
    // on board edit; the driver fires a probe at the next gap point.
    // lastCandidateProbeTime / candidateProbeCount: cadence + visible counter.)
    private var playAndTrainBoardMode: PlayAndTrainBoardMode {
        get { session.playAndTrainBoardMode } nonmutating set { session.playAndTrainBoardMode = newValue }
    }
    private var probeNetworkTarget: ProbeNetworkTarget {
        get { session.probeNetworkTarget } nonmutating set { session.probeNetworkTarget = newValue }
    }
    private var candidateProbeDirty: Bool {
        get { session.candidateProbeDirty } nonmutating set { session.candidateProbeDirty = newValue }
    }
    private var lastCandidateProbeTime: Date {
        get { session.lastCandidateProbeTime } nonmutating set { session.lastCandidateProbeTime = newValue }
    }
    private var candidateProbeCount: Int {
        get { session.candidateProbeCount } nonmutating set { session.candidateProbeCount = newValue }
    }

    // MARK: - Arena Tournament State
    //
    // Arena tournaments run inside the Play and Train driver task every
    // `stepsPerTournament` SGD steps. They play N games candidate vs
    // champion, alternating colors, pause self-play + training for the
    // duration, and either promote the candidate into the champion
    // (AlphaZero-style 0.55 score threshold) or leave the champion
    // alone. History is appended to `tournamentHistory` for display.

    // Tournament/arena display state (tournamentProgress / tournamentBox /
    // tournamentHistory / scoreStatusShowElo) moved to SessionController in
    // Stage 4j — forwarding proxies below. (tournamentProgress: live progress
    // mirrored from tournamentBox by the heartbeat. tournamentBox: lock-protected
    // box the arena driver writes per-game. tournamentHistory: all completed
    // tournaments this session, appended after each arena, in-memory only.
    // scoreStatusShowElo: status-bar "Score" cell percentage↔Elo toggle.)
    private var tournamentProgress: TournamentProgress? {
        get { session.tournamentProgress } nonmutating set { session.tournamentProgress = newValue }
    }
    private var tournamentBox: TournamentLiveBox? {
        get { session.tournamentBox } nonmutating set { session.tournamentBox = newValue }
    }
    private var tournamentHistory: [TournamentRecord] {
        get { session.tournamentHistory } nonmutating set { session.tournamentHistory = newValue }
    }
    private var scoreStatusShowElo: Bool {
        get { session.scoreStatusShowElo } nonmutating set { session.scoreStatusShowElo = newValue }
    }

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

    // Training — the trainer + training-run state moved to SessionController
    // in Stage 4d. These forward to `session` so the existing references
    // (heartbeat mirroring, status text, the giant `startRealTraining` /
    // `runArenaParallel`, etc.) keep working unchanged while the source of
    // truth lives on `SessionController`; they get inlined to `session.…` in
    // a later cleanup pass. (The trainer owns its own training-mode
    // `ChessNetwork` internally — not shared with `network` — so its weight
    // updates do not flow into inference; only one training mode runs at a
    // time so there's no cross-mode concurrency on the trainer.)
    private var trainer: ChessTrainer? {
        get { session.trainer } nonmutating set { session.trainer = newValue }
    }
    private var isTrainingOnce: Bool {
        get { session.isTrainingOnce } nonmutating set { session.isTrainingOnce = newValue }
    }
    private var continuousTraining: Bool {
        get { session.continuousTraining } nonmutating set { session.continuousTraining = newValue }
    }
    private var trainingTask: Task<Void, Never>? {
        get { session.trainingTask } nonmutating set { session.trainingTask = newValue }
    }
    private var lastTrainStep: TrainStepTiming? {
        get { session.lastTrainStep } nonmutating set { session.lastTrainStep = newValue }
    }
    private var trainingStats: TrainingRunStats? {
        get { session.trainingStats } nonmutating set { session.trainingStats = newValue }
    }
    private var trainingError: String? {
        get { session.trainingError } nonmutating set { session.trainingError = newValue }
    }
    private var trainingBox: TrainingLiveStatsBox? {
        get { session.trainingBox } nonmutating set { session.trainingBox = newValue }
    }

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

    // `trainerLearningRateDefault` / `entropyRegularizationCoeffDefault` moved
    // to `SessionController` in Stage 4m (used only by `buildCurrentSessionState`).
    nonisolated static let drawPenaltyDefault: Float = 0.1
    // `trainingBatchSize` (demo-training batch size) moved to
    // `SessionController` in Stage 4g — `SessionController.trainingBatchSize`.

    /// Number of training steps at the start of a Play-and-Train
    /// session for which the `[STATS]` line fires on every step.
    /// After this many steps the STATS ticker switches to a 60 s
    /// time-based cadence. 500 picked so the bootstrap window covers
    /// the first few minutes of training — long enough to see the
    /// initial loss curve shape without flooding the log once
    /// training settles.
    nonisolated static let bootstrapStatsStepCount: Int = 500

    // The training-alarm thresholds, divergence-streak detector, and the
    // banner / beep state moved to `TrainingAlarmController` (held below as
    // `@State private var trainingAlarm`). `policyEntropyAlarmThreshold` lives
    // there now too — referenced as `TrainingAlarmController.policyEntropyAlarmThreshold`.

    // Real (self-play) training run state moved to SessionController in
    // Stage 4d — forwarding proxies below. (Self-play generates games, labels
    // positions from the final outcome, pushes them through the shared
    // `trainer`; only one training mode runs at a time so no cross-mode
    // concurrency on the trainer.)
    private var realTraining: Bool {
        get { session.realTraining } nonmutating set { session.realTraining = newValue }
    }
    private var realTrainingTask: Task<Void, Never>? {
        get { session.realTrainingTask } nonmutating set { session.realTrainingTask = newValue }
    }
    private var replayBuffer: ReplayBuffer? {
        get { session.replayBuffer } nonmutating set { session.replayBuffer = newValue }
    }
    private var realRollingPolicyLoss: Double? {
        get { session.realRollingPolicyLoss } nonmutating set { session.realRollingPolicyLoss = newValue }
    }
    private var realRollingValueLoss: Double? {
        get { session.realRollingValueLoss } nonmutating set { session.realRollingValueLoss = newValue }
    }
    /// Latest legal-mass snapshot the [STATS] logger computed. Cached
    /// here so the chart-sample heartbeat (which fires more often
    /// than the [STATS] tick) can render the legal-masked entropy
    /// trace without recomputing the snapshot itself.
    // realLastLegalMassSnapshot moved to SessionController in Stage 4o — proxy.
    private var realLastLegalMassSnapshot: ChessTrainer.LegalMassSnapshot? {
        get { session.realLastLegalMassSnapshot } nonmutating set { session.realLastLegalMassSnapshot = newValue }
    }
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
    /// Discrete set of valid delay values in milliseconds used by
    /// both manual stepper edits and auto-computed delay handoff.
    /// Fine-grained 5 ms increments at the low end where small
    /// delays matter most, then 25 ms increments up to
    /// `stepDelayMaxMs`.
    nonisolated static let validDelayRungsMs: [Int] =
    [0, 5, 10, 15, 20] + Array(stride(from: 25, through: Self.stepDelayMaxMs, by: 25))
    // trainingStepDelayMs migrated to `trainingParams.trainingStepDelayMs`;
    // the training worker reads the live delay from
    // `replayRatioController.recordTrainingBatchAndGetDelay(...)` each
    // step, so no separate lock-protected mirror is needed.
    /// Shared lock-protected mirror of `trainingParams.selfPlayWorkers` the
    /// self-play worker tasks read between games. Moved to SessionController in
    /// Stage 4d — forwarding proxy. (The Stepper updates
    /// `trainingParams.selfPlayWorkers` AND this box atomically via the
    /// binding side-effect; workers poll the box at the top of each iteration.
    /// Allocated at session start, cleared on session end.)
    private var workerCountBox: WorkerCountBox? {
        get { session.workerCountBox } nonmutating set { session.workerCountBox = newValue }
    }
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
    /// All transactional state for the top-bar Arena countdown chip's
    /// popover (presentation flag, edit-text mirrors, per-field error
    /// flags, validation + write-back). Lives on its own `@Observable`
    /// model rather than as a forest of `@State` here. Wired with an
    /// `onAfterSave` closure (in `handleBodyOnAppear`) that pushes the
    /// freshly-edited arena τ-schedule into the live `samplingScheduleBox`.
    @State private var arenaSettingsPopover = ArenaSettingsPopoverModel(
        maxConcurrency: UpperContentView.absoluteMaxArenaConcurrency,
        formatDurationSpec: UpperContentView.formatDurationSpec,
        parseDurationSpec: UpperContentView.parseDurationSpec
    )
    /// Drives the Arena History sheet. Set true when the user clicks
    /// "History" in the Arena popover; the popover dismisses itself
    /// before flipping this flag so the sheet doesn't anchor to a
    /// dying popover.
    @State private var showArenaHistorySheet: Bool = false
    /// True while a one-shot log-scan recovery pass is running. Moved to
    /// SessionController in Stage 4r (along with `runArenaHistoryRecovery`) —
    /// forwarding proxy. (Disables the "Recover from logs" button against
    /// overlapping scans; drives a spinner in the arena-history sheet header.)
    private var arenaRecoveryInProgress: Bool {
        get { session.arenaRecoveryInProgress } nonmutating set { session.arenaRecoveryInProgress = newValue }
    }
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

    // `rollingLossWindow` (live-loss rolling-window size) moved to
    // `SessionController` in Stage 4g — `SessionController.rollingLossWindow`.

    // Batch-size sweep state + the startSweep/stopSweep/runSweep/sweepStatsText
    // methods moved to SessionController in Stage 4p — forwarding proxies below.
    private var sweepRunning: Bool {
        get { session.sweepRunning } nonmutating set { session.sweepRunning = newValue }
    }
    private var sweepResults: [SweepRow] {
        get { session.sweepResults } nonmutating set { session.sweepResults = newValue }
    }
    private var sweepProgress: SweepProgress? {
        get { session.sweepProgress } nonmutating set { session.sweepProgress = newValue }
    }
    private var sweepDeviceCaps: MetalDeviceMemoryLimits? {
        get { session.sweepDeviceCaps } nonmutating set { session.sweepDeviceCaps = newValue }
    }

    // MARK: - Checkpoint state (save / load models and sessions)
    //
    // Session-identity, checkpoint.lastSavedAt, session-start clock, the completed-
    // segments list, the in-flight segment record, the segment-relative
    // training-step anchor, the ActiveSegmentStart nested struct, the begin/
    // close-segment methods, and the cumulative wall-time / run-count
    // computed properties have moved to CheckpointController in Stage 3c
    // part 2b. External readers/writers reach them through `checkpoint.…`.

    /// A parsed session loaded from disk but not yet applied (the next
    /// `startRealTraining` consumes it and seeds the trainer / counters / IDs,
    /// then clears it). Moved to SessionController in Stage 4j — forwarding
    /// proxy.
    private var pendingLoadedSession: LoadedSession? {
        get { session.pendingLoadedSession } nonmutating set { session.pendingLoadedSession = newValue }
    }

    /// The checkpoint subsystem (status display, slow-save watchdog, segment
    /// tracking, parameter + load-model + load-session file importers + the
    /// last-session pointer record — the save/load engines and periodic-save
    /// will join in Stage 3c parts 2d–2e). Read by the body's status row
    /// (`CheckpointStatusLineView`); written by the save/load methods still on
    /// `UpperContentView` via `checkpoint.setCheckpointStatus(_:kind:)` /
    /// `.startSlowSaveWatchdog(label:)` / `.cancelSlowSaveWatchdog()` and the
    /// `.checkpointSaveInFlight` flag.
    @State private var checkpoint = CheckpointController()

    // checkpoint.showingLoadModelImporter / checkpoint.showingLoadSessionImporter / checkpoint.pendingLoadedModel
    // moved to CheckpointController in Stage 3c part 2c; the `.fileImporter`
    // modifiers in `body` now bind to `$checkpoint.checkpoint.showingLoadModelImporter`
    // etc.

    // The Load Parameters / Save Parameters importer/exporter state and the
    // three Parameter-handler methods moved to `CheckpointController` in
    // Stage 3c part 2a. `.fileImporter` / `.fileExporter` modifiers bind to
    // `$checkpoint.showingLoadParametersImporter` etc.

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
    // TrainingStartMode moved to SessionController in Stage 4o.

    // The periodic-autosave scheduler state (periodicSaveController /
    // periodicSaveLastPollAt / periodicSaveInFlight) moved to SessionController
    // in Stage 4l — forwarding proxies below. (periodicSaveController: the
    // 4-hour scheduler, created on Play-and-Train start, torn down on Stop,
    // polled by the heartbeat ~1 Hz; see PeriodicSaveController for the
    // arena-deferral / save-reset invariants. periodicSaveLastPollAt: throttles
    // the heartbeat poll to ~1 Hz. periodicSaveInFlight: guards against
    // double-firing while a periodic write is in flight — separate from
    // checkpoint.checkpointSaveInFlight because a periodic save runs even while
    // the menu items remain enabled.)
    private var periodicSaveController: PeriodicSaveController? {
        get { session.periodicSaveController } nonmutating set { session.periodicSaveController = newValue }
    }
    private var periodicSaveLastPollAt: Date? {
        get { session.periodicSaveLastPollAt } nonmutating set { session.periodicSaveLastPollAt = newValue }
    }
    private var periodicSaveInFlight: Bool {
        get { session.periodicSaveInFlight } nonmutating set { session.periodicSaveInFlight = newValue }
    }

    /// Interval between scheduled periodic saves while a
    /// Play-and-Train session is active. 4 hours per the
    /// product spec — long enough to keep disk churn low, short
    /// enough that a crash never forfeits more than half a
    /// working day of training.
    nonisolated static let periodicSaveIntervalSec: TimeInterval = 4 * 60 * 60

    // MARK: - Auto-resume sheet
    //
    // The launch-time "Resume last training session?" flow (sheet, 30-second
    // countdown, File-menu fallback, in-flight guard) lives on its own
    // `@MainActor @Observable` controller. The actual load-and-start chain
    // stays here (`loadSessionFrom(url:startAfterLoad:)`); `autoResume.onResume`
    // is wired to it in `handleBodyOnAppear`.
    @State private var autoResume = AutoResumeController()

    /// True if the File-menu "Resume training from autosave" item should be
    /// enabled: no live training run AND the controller has a resumable pointer
    /// (and isn't already mid-resume / showing the sheet).
    private var canResumeFromAutosave: Bool {
        !realTraining && autoResume.canResume
    }

    /// Composite scalar of the auto-resume gating flags, fed to `menuHubSignature`
    /// so a single `.onChange` on the body drives `syncMenuCommandHubState()`.
    private var autoResumeStateVersion: Int {
        autoResume.stateVersion
    }

    /// Live sampling schedules shared between UI edit fields and the
    /// self-play / arena players. Constructed at session start from the
    /// persisted `@AppStorage` values; `onChange` handlers on the
    /// tau fields call `setSelfPlay` / `setArena` with freshly
    /// constructed `SamplingSchedule` objects so edits take effect at
    /// each slot's next game boundary. Cleared when a session ends.
    // Session-runtime boxes (samplingScheduleBox / activeSelfPlayGate /
    // activeTrainingGate / replayRatioController / replayRatioSnapshot) and the
    // replay-ratio compensator state (effectiveReplayRatioTarget /
    // lastReplayRatioCompensatorAt) moved to SessionController in Stage 4f —
    // forwarding proxies below. (Set at session start by startRealTraining,
    // polled by the heartbeat, used by the checkpoint save path, cleared at
    // session end.)
    private var samplingScheduleBox: SamplingScheduleBox? {
        get { session.samplingScheduleBox } nonmutating set { session.samplingScheduleBox = newValue }
    }
    private var activeSelfPlayGate: WorkerPauseGate? {
        get { session.activeSelfPlayGate } nonmutating set { session.activeSelfPlayGate = newValue }
    }
    private var activeTrainingGate: WorkerPauseGate? {
        get { session.activeTrainingGate } nonmutating set { session.activeTrainingGate = newValue }
    }
    private var replayRatioController: ReplayRatioController? {
        get { session.replayRatioController } nonmutating set { session.replayRatioController = newValue }
    }
    private var replayRatioSnapshot: ReplayRatioController.RatioSnapshot? {
        get { session.replayRatioSnapshot } nonmutating set { session.replayRatioSnapshot = newValue }
    }

    // The replay-ratio compensator's state (`effectiveReplayRatioTarget` =
    // T_eff, `lastReplayRatioCompensatorAt`) and the `updateReplayRatioCompensator`
    // method moved to SessionController in Stage 4f — forwarding proxies below;
    // the heartbeat calls `session.updateReplayRatioCompensator(snap:)`.
    private var effectiveReplayRatioTarget: Double? {
        get { session.effectiveReplayRatioTarget } nonmutating set { session.effectiveReplayRatioTarget = newValue }
    }
    private var lastReplayRatioCompensatorAt: Date? {
        get { session.lastReplayRatioCompensatorAt } nonmutating set { session.lastReplayRatioCompensatorAt = newValue }
    }
    // The training-parameter properties formerly stored here as
    // @AppStorage / @State are now exposed by `TrainingParameters.shared`
    // and accessed via `trainingParams.<name>`. See TrainingParameters.swift
    // for the canonical definitions, defaults, and persistence.

    /// Last auto-computed step delay (auto-controller state, persisted across
    /// sessions, intentionally NOT a training parameter). Moved to
    /// SessionController in Stage 4l — forwarding proxy; the source of truth is
    /// `SessionController.lastAutoComputedDelayMs`, a `UserDefaults`-backed
    /// computed property (was `@AppStorage("lastAutoComputedDelayMs")` here).
    private var lastAutoComputedDelayMs: Int {
        get { session.lastAutoComputedDelayMs } nonmutating set { session.lastAutoComputedDelayMs = newValue }
    }

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
    /// The training-alarm subsystem (divergence detector, banner state, beep
    /// loop). Lives on its own `@MainActor @Observable` controller rather than
    /// as a fistful of `@State` here. Fed each heartbeat via
    /// `trainingAlarm.evaluate(from:)`; the legal-mass-collapse probe in
    /// `startRealTraining` calls `trainingAlarm.raise(...)` directly.
    @State private var trainingAlarm = TrainingAlarmController()

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
    // `isCandidateTestActive` moved to SessionController in Stage 4h — proxy.
    private var isCandidateTestActive: Bool { session.isCandidateTestActive }

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
            isArenaRunning: session.isArenaRunning,
            checkpointSaveInFlight: checkpoint.checkpointSaveInFlight,
            isTrainingOnce: isTrainingOnce,
            isEvaluating: isEvaluating,
            gameIsPlaying: gameSnapshot.isPlaying,
            hasNetwork: network != nil,
            hasPendingLoadedSession: pendingLoadedSession != nil,
            autoResumeStateVersion: autoResumeStateVersion,
            arenaRecoveryInProgress: arenaRecoveryInProgress
        )
    }

    /// All transactional state for the Training Settings popover (presentation
    /// flag, ~22 edit-text/checkbox fields, ~21 per-field error flags, the
    /// cancel stash for the live-propagated replay-ratio fields, validation +
    /// write-back, and the `applyLive…` propagation). Lives on its own
    /// `@Observable` model rather than as a forest of `@State` here. Its
    /// `trainerProvider` / `replayRatioControllerProvider` / `pushSelfPlaySchedule`
    /// hooks are wired in `handleBodyOnAppear`.
    @State private var trainingSettingsPopover = TrainingSettingsPopoverModel(
        selfPlayDelayMaxMs: UpperContentView.selfPlayDelayMaxMs,
        stepDelayMaxMs: UpperContentView.stepDelayMaxMs,
        maxSelfPlayWorkers: UpperContentView.absoluteMaxSelfPlayWorkers
    )

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

    /// "Last saved: 5/6/26 at 4:34 PM" or "Last saved: Never" — the
    /// "Never" case covers both a fresh Build Network with no save
    /// yet and a session that was resumed from disk but hasn't been
    /// re-saved in this app run. The on-disk `.dcmsession`'s age is
    /// deliberately ignored; this label tracks save activity *in
    /// this app session* so the user can see at a glance whether
    /// the trainer's current state has been written anywhere.
    private var lastSavedDisplayString: String {
        guard let when = checkpoint.lastSavedAt else { return "Last saved: Never" }
        return "Last saved: \(when.formatted(date: .abbreviated, time: .shortened))"
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
        // TEMP perf instrumentation (heartbeat-tick stall investigation):
        // time how long a single `body` evaluation (tree construction)
        // takes and how often it fires, on the same stdout stream as the
        // `>> after N:` / `[DISPATCH-LATENCY]` / `[MEMORY-DEBUG]` probes.
        // The `defer` runs when this getter's scope exits — i.e. after the
        // whole `VStack { … }` ViewBuilder below has finished — so `ms`
        // covers all the subview construction and `@ViewBuilder` helper
        // accesses. 2 ms floor keeps idle no-op renders out of the log.
        let __bodyT0 = CFAbsoluteTimeGetCurrent()
        defer {
            let ms = (CFAbsoluteTimeGetCurrent() - __bodyT0) * 1000
            if ms > 2 { print(String(format: "[BODY-EVAL] %.1f ms", ms)) }
        }
        #if DEBUG
        // Logs which tracked dependency triggered this invalidation.
        Self._printChanges()
        #endif
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
                    Text("Self play ID: \(net.identifier?.description ?? "–")")
                        .font(.callout)
                        .foregroundStyle(.secondary)
                }
                Text(networkStatus.isEmpty ? "" : networkStatus.components(separatedBy: "\n").first ?? "")
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                HStack(spacing: 4) {
                    if checkpoint.lastSavedAt != nil {
                        Image(systemName: "checkmark.circle.fill")
                            .foregroundStyle(.green)
                    }
                    Text(lastSavedDisplayString)
                        .font(.callout)
                        .foregroundStyle(checkpoint.lastSavedAt == nil ? AnyShapeStyle(.secondary) : AnyShapeStyle(Color.green))
                        .lineLimit(1)
                }
            }

            if let alarm = trainingAlarm.active {
                TrainingAlarmBanner(
                    alarm: alarm,
                    isSilenced: trainingAlarm.silenced,
                    onSilence: { trainingAlarm.silence() },
                    onDismiss: { trainingAlarm.dismiss() }
                )
            }

            cumulativeStatusBar

            // Status row — only renders when there's actually something
            // to show (a non-realTraining busy state, an in-flight
            // tournament, or a checkpoint status message). Keep the
            // row itself conditional so the layout stays compact when
            // idle; dialog / importer presentation hosts live on the
            // always-mounted root VStack below.
            let showsBusyContent: Bool = {
                if let _ = checkpoint.checkpointStatusMessage { return true }
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
                        if let msg = checkpoint.checkpointStatusMessage {
                            CheckpointStatusLineView(kind: checkpoint.checkpointStatusKind, message: msg)
                        }
                        Spacer(minLength: 0)
                    }
                }
            }
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
                        session.startRealTraining(mode: .continueAfterStop)
                    }
                    Button("New session with current trainer") {
                        session.startRealTraining(mode: .newSessionKeepTrainer)
                    }
                    Button("New session — trainer reset from champion") {
                        session.startRealTraining(mode: .newSessionResetTrainerFromChampion)
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
                    playAndTrainBoardMode: $session.playAndTrainBoardMode,
                    sideToMoveBinding: sideToMoveBinding,
                    probeNetworkTarget: $session.probeNetworkTarget,
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
                        },
                        onHoverSquare: { sq in
                            hoveredBoardSquare = sq
                        }
                    )
                )

                // Hover-driven top-3 channels overlay. When the
                // cursor is over a square AND we have an inference
                // result with raw logits, show a horizontal row of
                // 3 mini-board tiles displaying the top-3 channels
                // (by per-channel logit at that from-square) — each
                // tile draws the channel's geometric move as an
                // arrow from the hovered square. Replaces the
                // MainTextPanel during hover; restores on un-hover.
                if let sq = hoveredBoardSquare,
                   let logits = inferenceResult?.rawInference?.logits,
                   logits.count == ChessNetwork.policySize {
                    HoverPolicyOverlay(
                        hoveredRow: sq / 8,
                        hoveredCol: sq % 8,
                        currentPlayer: editableState.currentPlayer,
                        pieces: displayedPieces,
                        policyLogits: logits,
                        policyProbs: inferenceResult?.rawInference?.policy
                    )
                } else {
                    MainTextPanel(
                        isGameMode: isGameMode,
                        isTrainingMode: isTrainingMode,
                        isCandidateTestActive: isCandidateTestActive,
                        inferenceResultText: inferenceResult?.textOutput,
                        trainingError: trainingError,
                        selfPlayColumn: { selfPlayStatsColumn },
                        trainingColumn: { trainingStatsColumnView }
                    )
                }
            }
            .layoutPriority(1)

            // Full-area policy-channel decomposition. Toggled via
            // View > Show Policy Channels Panel. When on, takes over
            // the entire region freed up by the (now-hidden) chart
            // pane — `ContentView` drops `LowerContentView` for the
            // same toggle. When off, layout is unchanged. Gated on
            // `showForwardPassUI` so it never renders against a
            // stale result on an unrelated board (Game Run / Game
            // Mode / pure-training paths) — passes nil logits in
            // those cases so the panel shows its own placeholder.
            if showPolicyChannelsPanel {
                Divider()
                PolicyChannelsPanel(
                    pieces: displayedPieces,
                    currentPlayer: editableState.currentPlayer,
                    policyLogits: showForwardPassUI ? inferenceResult?.rawInference?.logits : nil,
                    policyProbs: showForwardPassUI ? inferenceResult?.rawInference?.policy : nil
                )
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .layoutPriority(1)
            }

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
        .background {
            // Each importer/exporter stays on its own host view:
            // stacking multiple file-presentation modifiers on one
            // SwiftUI host became unreliable on newer macOS builds.
            // The important part is that the host itself must also be
            // always mounted; File-menu actions can fire while the
            // status row above is absent.
            Color.clear
                .fileImporter(
                    isPresented: $checkpoint.showingLoadModelImporter,
                    allowedContentTypes: [.data, .item],
                    allowsMultipleSelection: false,
                    onCompletion: { result in
                        handleLoadModelPickResult(result)
                    }
                )
                .fileDialogDefaultDirectory(
                    checkpoint.showingLoadModelImporter
                    ? CheckpointPaths.modelsDir
                    : CheckpointPaths.sessionsDir
                )

            Color.clear
                .fileImporter(
                    isPresented: $checkpoint.showingLoadSessionImporter,
                    allowedContentTypes: [.folder],
                    allowsMultipleSelection: false,
                    onCompletion: { result in
                        handleLoadSessionPickResult(result)
                    }
                )

            Color.clear
                .fileImporter(
                    isPresented: $checkpoint.showingLoadParametersImporter,
                    allowedContentTypes: [.json],
                    allowsMultipleSelection: false,
                    onCompletion: { result in
                        checkpoint.handleLoadParametersPickResult(result)
                    }
                )

            Color.clear
                .fileExporter(
                    isPresented: $checkpoint.showingSaveParametersExporter,
                    document: checkpoint.parametersDocumentForExport,
                    contentType: .json,
                    defaultFilename: "parameters",
                    onCompletion: { result in
                        checkpoint.handleSaveParametersExportResult(result)
                    }
                )
        }
        .background(WindowAccessor(window: $contentWindow, onAttached: handleWindowAttached))
        .onAppear { handleBodyOnAppear() }
        .sheet(isPresented: $autoResume.sheetShowing) {
            if let pointer = autoResume.pointer {
                AutoResumeSheetView(
                    pointer: pointer,
                    summary: autoResume.summary,
                    countdownRemaining: autoResume.countdownRemaining,
                    onDismiss: { autoResume.dismiss() },
                    onResume: { autoResume.performResume() }
                )
            }
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
            playAndTrainBoardMode: $session.playAndTrainBoardMode,
            probeNetworkTarget: $session.probeNetworkTarget,
            candidateProbeDirty: $session.candidateProbeDirty,
            selectedOverlay: $selectedOverlay,
            resyncLrWarmupText: trainingSettingsPopover.resyncLrWarmupText,
            effectiveReplayRatioTarget: $session.effectiveReplayRatioTarget,
            lastReplayRatioCompensatorAt: $session.lastReplayRatioCompensatorAt,
            trainingParams: trainingParams,
            workerCountBox: workerCountBox,
            trainer: trainer,
            replayRatioController: replayRatioController,
            snapDelayToNearestValidRung: { delay in
                Self.validDelayRungsMs.min(by: { abs($0 - delay) < abs($1 - delay) }) ?? delay
            }
        ))
        .onReceive(snapshotTimer) { _ in
            // Capture timestamp at dispatch so the tick body can
            // measure how long the main actor took to begin executing
            // it. A growing gap between dispatch and execution means
            // the main actor is being starved by other work — the
            // primary mechanism behind UI stalls during long sessions.
            let dispatchedAt = CFAbsoluteTimeGetCurrent()

            guard !snapshotTickInFlight else {
                return
            }

            snapshotTickInFlight = true
            Task.detached {
                await processSnapshotTimerTick(dispatchedAt: dispatchedAt)
                await MainActor.run { snapshotTickInFlight = false }
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
        // Wire the popover models' side-effect hooks. The Arena popover pushes
        // the new τ schedule into the live `samplingScheduleBox` after a Save;
        // the Training popover does the same for its self-play schedule and
        // needs live references to the trainer / replay-ratio controller so its
        // optimizer-mirror and live-delay writes reach them. (`samplingScheduleBox`,
        // `trainer`, `replayRatioController` are `@State` — reference-backed — so
        // these closures see the current values even when re-evaluated.)
        arenaSettingsPopover.onAfterSave = {
            samplingScheduleBox?.setArena(session.buildArenaSchedule())
        }
        trainingSettingsPopover.trainerProvider = { trainer }
        trainingSettingsPopover.replayRatioControllerProvider = { replayRatioController }
        trainingSettingsPopover.pushSelfPlaySchedule = {
            samplingScheduleBox?.setSelfPlay(session.buildSelfPlaySchedule())
        }
        // The auto-resume controller chains into the load-and-start path here.
        autoResume.onResume = { pointer in
            loadSessionFrom(url: pointer.directoryURL, startAfterLoad: true)
        }
        // The Load-Parameters file-import path needs to apply the picked
        // CliTrainingConfig over `trainingParams` and return the list of
        // fields that changed (so the controller can surface them in the
        // status row). `applyCliConfigOverridesFromMenu(cfg:)` does both and
        // already returns a `[ParameterOverrideChange]` shaped to fit.
        checkpoint.onApplyOverrides = { applyCliConfigOverridesFromMenu(cfg: $0) }
        // Wire the segment-tracking providers. CheckpointController calls
        // these at begin/close-segment time to capture the live counter
        // snapshots without holding direct references to the @State that
        // back them.
        checkpoint.trainingStepsProvider = { trainingStats?.steps }
        checkpoint.totalPositionsAddedProvider = { replayBuffer?.totalPositionsAdded }
        checkpoint.selfPlayGamesProvider = { session.parallelStats?.selfPlayGames }
        checkpoint.trainingBoxSnapshotProvider = { trainingBox?.snapshot() }
        // Wire the build flow's view-facing hooks. SessionController owns the
        // build path now but still reaches the busy gate / refuse alert /
        // clear-training-display / trainer-drop / last-saved-at marker through
        // these until that state migrates too.
        session.isBusyProvider = { isBusy }
        session.busyReasonProvider = { busyReasonMessage() }
        session.onRefuseMenuAction = { refuseMenuAction($0) }
        session.onClearTrainingDisplay = { clearTrainingDisplay() }
        session.onDropTrainer = { trainer = nil }
        session.onResetBoardDisplay = {
            inferenceResult = nil
            gameWatcher.resetAll()
            gameSnapshot = gameWatcher.snapshot()
        }
        session.editableStateProvider = { editableState }
        session.onInferenceResult = { inferenceResult = $0 }
        session.gameWatcherProvider = { gameWatcher }
        session.onClearChampionLoadedFlag = { championLoadedSinceLastTrainingSegment = false }
        session.cliConfig = cliConfig
        session.cliOutputURL = cliOutputURL
        session.autoTrainOnLaunch = autoTrainOnLaunch
        session.checkpoint = checkpoint
        session.chartCoordinator = chartCoordinator
        session.trainingAlarm = trainingAlarm
        // Resume-sheet UX is correctly gated on the window being
        // visible — surfacing a sheet on a hidden window would do
        // nothing useful. Skipped under `--train` because the
        // headless launch path (`handleWindowAttached`) has
        // already kicked off training and the sheet would be
        // confusing on top of an active session.
        if !autoTrainOnLaunch {
            autoResume.maybePresentSheet(isTrainingActive: realTraining)
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
    @State private var snapshotTickInFlight = false
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
        // `elap(_:)` is the per-stage trace probe used to attribute a
        // UI stall to a specific block of this tick. Cheap enough to
        // leave on at the 2 Hz heartbeat — `os_log` is hundreds of
        // nanoseconds per emit and we're producing ~13 lines/sec,
        // negligible against the work the tick body itself does.
        func elap(_ s: String) {
            logger.log(">> \(s): \(Date().timeIntervalSince(start))")
        }
        let start = Date()
        _ = start
        elap("start")
        gameSnapshot = await gameWatcher.asyncSnapshot()
        elap("after gameWatcher")
        // Same heartbeat pulls the sweep's worker-thread progress and
        // any newly-completed rows into @State so the table grows live.
        if sweepRunning, let box = session.sweepCancelBox {
            sweepProgress = await box.asyncLatestProgress()
            // Sample process resident memory and feed it into the
            // sweep's per-row peak. The trainer also samples at row
            // boundaries — we just contribute extra samples while a
            // row is in flight so we don't miss mid-step spikes.
            let phys = await ChessTrainer.getAppMemoryFootprintBytes()
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
            let snap = await box.snapshot()
            if snap.stats.steps != (trainingStats?.steps ?? -1) {
                trainingStats = snap.stats
                lastTrainStep = snap.lastTiming
                realRollingPolicyLoss = snap.rollingPolicyLoss
                realRollingValueLoss = snap.rollingValueLoss
                
                // Periodic memory log to stdout to correlate with performance degradation
                if snap.stats.steps % 100 == 0 {
                    let appMB = Double(memoryStatsSnap?.appFootprintBytes ?? 0) / 1024 / 1024
                    let gpuMB = Double(memoryStatsSnap?.gpuAllocatedBytes ?? 0) / 1024 / 1024
                    print(String(format: "[MEMORY-DEBUG] step=%d app=%.1fMB gpu=%.1fMB", snap.stats.steps, appMB, gpuMB))
                }
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
        if let pBox = session.parallelWorkerStatsBox {
            let snap = await pBox.asyncSnapshot()
            let prev = session.parallelStats
            // `sessionStart` is included in the dirty check so the
            // one-time shift performed by `markWorkersStarted()` lands
            // in @State immediately, even if no game or training step
            // has recorded yet.
            let changed = snap.selfPlayGames != (prev?.selfPlayGames ?? -1)
            || snap.trainingSteps != (prev?.trainingSteps ?? -1)
            || snap.recentGames != (prev?.recentGames ?? -1)
            || snap.sessionStart != prev?.sessionStart
            if changed {
                session.parallelStats = snap
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
            session.updateReplayRatioCompensator(snap: snap)
        }
        elap("after 10")
        // Diversity-histogram mirror. Read once per heartbeat off the
        // tracker's thread-safe snapshot. Only push into @State when
        // the bucket totals actually change (or the bar array is
        // currently empty) so SwiftUI doesn't invalidate the chart
        // every tick for a stable reading.
        if let tracker = session.selfPlayDiversityTracker {
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
        let app = await ChessTrainer.getAppMemoryFootprintBytes()
        let caps: MetalDeviceMemoryLimits?
        if let trainer {
            caps = await trainer.currentMetalMemoryLimits()
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
        // Chart sample elapsed-second axis comes off
        // `chartCoordinator.chartElapsedAnchor`, NOT
        // `session.parallelStats.sessionStart` or `checkpoint.currentSessionStart`.
        // The chart anchor is back-dated on session resume so a
        // restored chart trajectory and post-resume samples share
        // one continuous elapsed-sec axis (no visible gap, no
        // overlap). Every chart-axis call site — this one, the
        // progress-rate sampler, and both arena-event sites — must
        // use the same anchor or the elapsedSec values across
        // sources land in different coordinate spaces and the
        // shared `scrollX` binding parks some sources off-screen
        // (the same bug class an earlier mismatch between
        // `session.parallelStats.sessionStart` and the back-dated
        // `checkpoint.currentSessionStart` originally introduced).
        let elapsed = max(0, now.timeIntervalSince(chartCoordinator.chartElapsedAnchor))
        let trainingSnap: TrainingLiveStatsBox.Snapshot?
        if let trainingBox {
            trainingSnap = await trainingBox.snapshot()
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
            rollingValueAbsMean: trainingSnap?.rollingValueAbsMean,
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
        // The training-alarm evaluation (divergence streaks → banner / beep)
        // lives on `TrainingAlarmController` — see `trainingAlarm`.
        await chartCoordinator.appendTrainingChart(sample, totalGpuMs: currentGpuMs)
        trainingAlarm.evaluate(from: sample)
    }

    private func refreshProgressRateIfNeeded() async {
        guard realTraining else { return }
        let now = Date()
        if now.timeIntervalSince(chartCoordinator.progressRateLastFetch) < Self.progressRateRefreshSec {
            return
        }

        guard let session = session.parallelStats else { return }
        // ElapsedSec on chart axis comes off the chart-coordinator's
        // anchor, NOT `session.sessionStart`. See the matching block
        // in `refreshTrainingChartIfNeeded` for full reasoning —
        // both samplers must share an anchor or `scrollX` ends up
        // straddling two coordinate spaces. `session.sessionStart`
        // remains the correct anchor for everything else this
        // function does (cumulative-counter deltas use 3-minute
        // window timestamps, not elapsedSec).
        let elapsed = max(0, now.timeIntervalSince(chartCoordinator.chartElapsedAnchor))
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

    // `setCheckpointStatus`, `startSlowSaveWatchdog`, `cancelSlowSaveWatchdog`,
    // and the `slowSaveWatchdogSeconds` constant moved to
    // `App/UpperContentView/CheckpointController.swift` (Stage 3c part 1).
    // The save / load methods below still call them through `checkpoint.…`.

    // handleLoadParametersPickResult / handleSaveParametersMenuAction /
    // handleSaveParametersExportResult moved to CheckpointController in
    // Stage 3c part 2a — call them via `checkpoint.handleX(...)`. The CLI-config
    // override path is reached via `checkpoint.onApplyOverrides`, wired in
    // `handleBodyOnAppear` to `applyCliConfigOverridesFromMenu(cfg:)`.

    // buildCurrentSessionState(championID:trainerID:) moved to SessionController
    // in Stage 4m — the save paths call session.buildCurrentSessionState(...).

    // seedChartCoordinatorFromLoadedSession(chartURLs:) moved to
    // SessionController in Stage 4k — the resume branch calls
    // session.seedChartCoordinatorFromLoadedSession(chartURLs:).

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
        if checkpoint.checkpointSaveInFlight {
            refuseMenuAction("A save is already in progress. Wait for it to finish.")
            return
        }
        if session.isArenaRunning {
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
        checkpoint.checkpointSaveInFlight = true
        checkpoint.setCheckpointStatus("Saving champion…", kind: .progress)
        checkpoint.startSlowSaveWatchdog(label: "champion save")

        Task {
            // Pause worker 0 if a session is running. Bail with a
            // user-visible error on timeout (indicates the session
            // has already ended or the worker is stuck — either way
            // we shouldn't spin forever).
            if let gate {
                let acquired = await gate.pauseAndWait(timeoutMs: Self.saveGateTimeoutMs)
                if !acquired {
                    checkpoint.cancelSlowSaveWatchdog()
                    checkpoint.checkpointSaveInFlight = false
                    checkpoint.setCheckpointStatus("Save aborted: could not pause self-play (timeout)", kind: .error)
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
                checkpoint.cancelSlowSaveWatchdog()
                checkpoint.checkpointSaveInFlight = false
                checkpoint.setCheckpointStatus("Save failed (export): \(exportError.localizedDescription)", kind: .error)
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
            checkpoint.cancelSlowSaveWatchdog()
            checkpoint.checkpointSaveInFlight = false
            switch outcome {
            case .success(let url):
                checkpoint.setCheckpointStatus("Saved \(url.lastPathComponent)", kind: .success)
                SessionLogger.shared.log("[CHECKPOINT] Saved champion: \(url.lastPathComponent)")
            case .failure(let error):
                checkpoint.setCheckpointStatus("Save failed: \(error.localizedDescription)", kind: .error)
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
        if checkpoint.checkpointSaveInFlight {
            refuseMenuAction("A save is already in progress. Wait for it to finish.")
            return
        }
        if session.isArenaRunning {
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
        if session.isArenaRunning {
            return
        }
        if checkpoint.checkpointSaveInFlight {
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
        checkpoint.checkpointSaveInFlight = true
        checkpoint.setCheckpointStatus("Saving session\(uiSuffix)…", kind: .progress)
        checkpoint.startSlowSaveWatchdog(label: "session save\(uiSuffix)")

        // Build the state snapshot on the main actor before
        // jumping to detached work. Capture the replay buffer handle
        // here too so the detached write path can serialize it
        // alongside the two network files — `ReplayBuffer` is
        // `@unchecked Sendable` and serializes access via its own
        // lock, so the buffer can be written from a background task
        // while self-play workers (which only append) are paused.
        let sessionState = session.buildCurrentSessionState(
            championID: championID,
            trainerID: trainerID
        )
        let trainingStep = trainingStats?.steps ?? 0
        let bufferForSave = replayBuffer
        // Snapshot the chart-coordinator state on the main actor
        // BEFORE jumping to detached work — the rings are
        // `@MainActor`-isolated so the array copies have to happen
        // here. `buildSnapshot()` returns nil when collection is off
        // or both rings are empty, in which case the save path skips
        // writing chart-companion files entirely (matching the
        // existing `bufferForSave == nil` skip).
        let chartSnapshotForSave = chartCoordinator.buildSnapshot()

        Task {
            // Helper to clear both in-flight flags consistently on
            // every early-return path below. The periodic flag is
            // only meaningful when `trigger == .periodic`, but it's
            // cheap to always clear so we don't have to repeat the
            // branch on every error exit. Cancels the slow-save
            // watchdog too so a fast-failure path doesn't leave a
            // stale "Saving… (still running)" amber line behind.
            @MainActor func clearInFlight() {
                checkpoint.cancelSlowSaveWatchdog()
                checkpoint.checkpointSaveInFlight = false
                periodicSaveInFlight = false
            }

            // Pause self-play briefly so the champion export is
            // race-free, snapshot weights, then resume. Uses the
            // bounded variant so a session end mid-save doesn't
            // spin forever waiting for workers that have exited.
            let selfPlayAcquired = await selfPlayGate.pauseAndWait(timeoutMs: Self.saveGateTimeoutMs)
            guard selfPlayAcquired else {
                clearInFlight()
                checkpoint.setCheckpointStatus("Save aborted: could not pause self-play (timeout)", kind: .error)
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
                checkpoint.setCheckpointStatus("Save failed (champion export): \(championError.localizedDescription)", kind: .error)
                SessionLogger.shared.log("[CHECKPOINT] Save session failed at champion export: \(championError.localizedDescription)")
                return
            }

            // Pause training briefly to snapshot trainer weights.
            let trainingAcquired = await trainingGate.pauseAndWait(timeoutMs: Self.saveGateTimeoutMs)
            guard trainingAcquired else {
                clearInFlight()
                checkpoint.setCheckpointStatus("Save aborted: could not pause training (timeout)", kind: .error)
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
                checkpoint.setCheckpointStatus("Save failed (trainer export): \(trainerError.localizedDescription)", kind: .error)
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
                [bufferForSave, chartSnapshotForSave] in
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
                        chartSnapshot: chartSnapshotForSave,
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
                checkpoint.setCheckpointStatus("Saved \(url.lastPathComponent)\(uiSuffix)", kind: .success)
                let bufStr: String
                if let snap = bufferForSave?.stateSnapshot() {
                    bufStr = " replay=\(snap.storedCount)/\(snap.capacity)"
                } else {
                    bufStr = ""
                }
                SessionLogger.shared.log(
                    "[CHECKPOINT] Saved session (\(diskTag)): \(url.lastPathComponent) build=\(BuildInfo.buildNumber) git=\(BuildInfo.gitHash)\(bufStr)"
                )
                checkpoint.recordLastSessionPointer(
                    directoryURL: url,
                    sessionID: sessionState.sessionID,
                    trigger: diskTag
                )
                periodicSaveController?.noteSuccessfulSave(at: Date())
                checkpoint.lastSavedAt = Date()
            case .failure(let error):
                checkpoint.setCheckpointStatus("Save failed: \(error.localizedDescription)", kind: .error)
                SessionLogger.shared.log("[CHECKPOINT] Save session (\(diskTag)) failed: \(error.localizedDescription)")
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
            checkpoint.setCheckpointStatus("Load cancelled: \(error.localizedDescription)", kind: .error)
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

        checkpoint.checkpointSaveInFlight = true
        checkpoint.setCheckpointStatus("Loading \(url.lastPathComponent)…", kind: .progress)

        Task {
            // Auto-build the champion shell if it doesn't exist yet.
            // The weights are about to be overwritten, so the random
            // init is only satisfying graph compilation — no reason
            // to require the user to press Build first.
            let championResult = await session.ensureChampionBuilt()
            switch championResult {
            case .failure(let error):
                checkpoint.checkpointSaveInFlight = false
                checkpoint.setCheckpointStatus("Build failed: \(error.localizedDescription)", kind: .error)
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
                checkpoint.checkpointSaveInFlight = false
                switch outcome {
                case .success(let file):
                    champion.identifier = ModelID(value: file.modelID)
                    networkStatus = "Loaded model \(file.modelID)\nFrom: \(url.lastPathComponent)"
                    checkpoint.setCheckpointStatus("Loaded \(file.modelID)", kind: .success)
                    SessionLogger.shared.log("[CHECKPOINT] Loaded model: \(url.lastPathComponent) → \(file.modelID)")
                    inferenceResult = nil
                    // Flag champion-replaced for the post-Stop Start
                    // dialog's "Continue" annotation. Cleared as
                    // soon as a new training segment starts.
                    if replayBuffer != nil {
                        championLoadedSinceLastTrainingSegment = true
                    }
                case .failure(let error):
                    checkpoint.setCheckpointStatus("Load failed: \(error.localizedDescription)", kind: .error)
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
            checkpoint.setCheckpointStatus("Load cancelled: \(error.localizedDescription)", kind: .error)
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

        checkpoint.checkpointSaveInFlight = true
        checkpoint.setCheckpointStatus("Loading session \(url.lastPathComponent)…", kind: .progress)

        Task {
            // Auto-build the champion shell if it doesn't exist yet.
            // The weights are about to be overwritten, so the random
            // init is only satisfying graph compilation — no reason
            // to require the user to press Build first.
            let championResult = await session.ensureChampionBuilt()
            guard case .success(let champion) = championResult else {
                checkpoint.checkpointSaveInFlight = false
                if case .failure(let error) = championResult {
                    checkpoint.setCheckpointStatus("Build failed: \(error.localizedDescription)", kind: .error)
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
            checkpoint.checkpointSaveInFlight = false
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
                checkpoint.lastSavedAt = nil
                checkpoint.setCheckpointStatus("Loaded session \(loaded.state.sessionID) — click Play and Train to resume", kind: .success)
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
                    session.startRealTraining()
                }
            case .failure(let error):
                checkpoint.setCheckpointStatus("Load failed: \(error.localizedDescription)", kind: .error)
                SessionLogger.shared.log("[CHECKPOINT] Load session failed: \(error.localizedDescription)")
            }
            if startAfterLoad {
                autoResume.markResumeFinished()
            }
        }
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
        session.buildNetwork()
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
            session.startRealTraining(mode: .freshOrFromLoadedSession)
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
            let ladder = Self.validDelayRungsMs
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

    /// File-menu entry point for "Resume training from autosave". Re-reads the
    /// pointer from UserDefaults first (the user may have saved again — which
    /// updated the pointer — since the launch sheet), guards on no live training
    /// run / no in-flight resume / pointer-still-on-disk, then hands off to the
    /// `AutoResumeController`. (The launch-sheet path goes through
    /// `autoResume.performResume()` directly.)
    @MainActor
    private func resumeFromAutosaveMenuAction() {
        SessionLogger.shared.log("[BUTTON] Resume Training from Autosave")
        guard !realTraining else {
            refuseMenuAction("Stop the current training session before resuming from autosave.")
            return
        }
        guard !autoResume.inFlight else { return }
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
        autoResume.resumeFromPointer(pointer)
    }

    /// Open Finder pointed at the checkpoint root so the user can
    /// browse saved sessions and models even though Application
    /// Support is hidden by default. Creates the folder if it
    /// doesn't exist yet so the button always works.
    private func handleRevealSaves() {
        do {
            try CheckpointPaths.ensureDirectories()
        } catch {
            checkpoint.setCheckpointStatus("Could not create save folder: \(error.localizedDescription)", kind: .error)
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
        commandHub.buildNetwork = { session.buildNetwork() }
        commandHub.runForwardPass = { runForwardPass() }
        commandHub.playSingleGame = { playSingleGame() }
        commandHub.startContinuousPlay = { startContinuousPlay() }
        commandHub.trainOnce = { session.trainOnce() }
        commandHub.startContinuousTraining = { session.startContinuousTraining() }
        commandHub.startRealTraining = { startTrainingFromMenu() }
        commandHub.startSweep = { Task.detached { await session.startSweep() } }
        commandHub.stopAnyContinuous = { stopAnyContinuous() }
        commandHub.runArena = {
            SessionLogger.shared.log("[BUTTON] Run Arena")
            guard !session.isArenaRunning else { return }
            session.arenaTriggerBox?.trigger()
        }
        commandHub.runEngineDiagnostics = { session.runEngineDiagnostics() }
        commandHub.runPolicyConditioningDiagnostic = { session.runPolicyConditioningDiagnostic() }
        commandHub.recoverArenaHistoryFromLogs = { session.runArenaHistoryRecovery() }
        commandHub.abortArena = {
            SessionLogger.shared.log("[BUTTON] Abort Arena")
            session.arenaOverrideBox?.abort()
        }
        commandHub.promoteCandidate = {
            SessionLogger.shared.log("[BUTTON] Promote")
            session.arenaOverrideBox?.promote()
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
            checkpoint.showingLoadSessionImporter = true
        }
        commandHub.loadModel = {
            SessionLogger.shared.log("[BUTTON] Load Model")
            checkpoint.showingLoadModelImporter = true
        }
        commandHub.loadParameters = {
            SessionLogger.shared.log("[BUTTON] Load Parameters")
            checkpoint.showingLoadParametersImporter = true
        }
        commandHub.saveParameters = {
            checkpoint.handleSaveParametersMenuAction()
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
        commandHub.isArenaRunning = session.isArenaRunning
        commandHub.checkpointSaveInFlight = checkpoint.checkpointSaveInFlight
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
            || checkpoint.checkpointSaveInFlight
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
        if checkpoint.checkpointSaveInFlight {
            return "A save/load is already in progress. Wait for it to finish."
        }
        return "Another operation is in progress."
    }

    // `ensureChampionBuilt()` and `buildNetwork()` moved to `SessionController`
    // in Stage 4c — call them as `session.ensureChampionBuilt()` /
    // `session.buildNetwork()`. The view-facing bits they touch (the busy
    // gate, the menu-refuse alert, `clearTrainingDisplay()`, dropping the
    // trainer, `checkpoint.lastSavedAt`) are wired into `session` via closures
    // / weak reference in `handleBodyOnAppear`.

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
    // fireCandidateProbeIfNeeded() / buildCliCandidateTestEvent() moved to
    // SessionController in Stage 4i — the Play-and-Train driver loop calls
    // session.fireCandidateProbeIfNeeded(). It reads editableState via
    // session.editableStateProvider and publishes the result via
    // session.onInferenceResult (both wired in handleBodyOnAppear).

    // runArenaParallel(...) / logArenaResult(...) / cleanupArenaState(...) moved to
    // SessionController in Stage 4n — startRealTraining's arena-coordinator task
    // calls await session.runArenaParallel(...).

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
                // `session.probeRunner` → the dedicated probe inference network.
                await SessionController.performInference(with: runner, state: state)
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
        if continuousTraining { session.stopContinuousTraining() }
        if sweepRunning { session.stopSweep() }
        if realTraining { session.stopRealTraining() }
    }

    // MARK: - Training Actions

    /// Build (or reuse) the trainer. The trainer manages its own
    /// training-mode network internally — it doesn't share weights with
    /// the inference network used by Play / Forward Pass — so the inference
    /// network can keep its frozen-stats BN for fast play while the trainer
    /// measures realistic training-step costs through batch-stats BN.
    // `session.ensureTrainer()`, `session.buildSelfPlaySchedule()`, `session.buildArenaSchedule()`
    // moved to `SessionController` in Stage 4e — call them as
    // `session.ensureTrainer()` / `session.buildSelfPlaySchedule()` /
    // `session.buildArenaSchedule()`. They only depend on
    // `TrainingParameters.shared` plus the (now session-owned) `trainer` /
    // `trainingError`, so the move is closure-free.

    // `trainOnce()` / `startContinuousTraining()` / `stopContinuousTraining()`
    // / `runOneTrainStep` and the `trainingBatchSize` / `rollingLossWindow`
    // statics moved to `SessionController` in Stage 4g — call them as
    // `session.trainOnce()` etc. The on-board display reset they do (clearing
    // `inferenceResult` / resetting `gameWatcher`) goes through
    // `session.onResetBoardDisplay`, wired in `handleBodyOnAppear`.

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
            session.startRealTraining(mode: .freshOrFromLoadedSession)
        }
    }

    // startRealTraining(mode:) and stopRealTraining() moved to SessionController
    // in Stage 4o — callers use session.startRealTraining(mode:) /
    // session.stopRealTraining().

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
        trainingAlarm.clear()
    }

    fileprivate var cumulativeStatusBar: UpperCumulativeStatusBar<some View> {
        let totalSteps = trainingStats?.steps ?? 0
        let hasHistory = checkpoint.cumulativeRunCount > 0 || totalSteps > 0
        let canRunArena = !session.isArenaRunning && network != nil && trainer != nil
        let totalPositions = totalSteps * trainingParams.trainingBatchSize
        let warmupLR: String? = {
            if let snap = trainerWarmupSnap, snap.inWarmup {
                return String(format: "%.2e", snap.effectiveLR)
            }
            return nil
        }()
        let legalMassStr = realLastLegalMassSnapshot.map {
            String(format: "%.4f%%", Double($0.legalMass) * 100)
        } ?? "--"
        return UpperCumulativeStatusBar(
            hasHistory: hasHistory,
            canRunArena: canRunArena,
            activeTrainingTime: GameWatcher.Snapshot.formatHMS(seconds: checkpoint.cumulativeActiveTrainingSec),
            warmupLREffective: warmupLR,
            trainingSteps: Int(totalSteps).formatted(),
            positionsTrained: Self.formatCompactCount(totalPositions),
            trainingRate: trainingRateStatusValue,
            legalMass: legalMassStr,
            runs: "\(checkpoint.cumulativeRunCount)",
            arenas: "\(tournamentHistory.count)",
            promotions: "\(tournamentHistory.lazy.filter { $0.promoted }.count)",
            scoreCell: scoreStatusBarCell,
            // Right-side chips. Built each parent render. The
            // popovers' bindings / error flags / callbacks remain
            // captured here exactly as before. The chip-side
            // state graph can be tightened in a follow-up by
            // extracting each chip+popover into its own wrapper
            // struct.
            rightChips: {
                SessionStatusChipView(
                    kind: sessionStatusChip,
                    warmupCompletedSteps: trainerWarmupSnap?.completedSteps,
                    warmupTotalSteps: trainerWarmupSnap?.warmupSteps
                )
                trainingSettingsChip
                ArenaCountdownChip(
                    isArenaRunning: session.isArenaRunning,
                    countdownText: { now in arenaCountdownText(at: now) },
                    showPopover: $arenaSettingsPopover.isPresented
                ) {
                    ArenaSettingsPopover(
                        model: arenaSettingsPopover,
                        nextArenaDate: session.arenaTriggerBox.map {
                            $0.lastArenaTime.addingTimeInterval(trainingParams.arenaAutoIntervalSec)
                        },
                        lastArena: tournamentHistory.last,
                        isArenaRunning: session.isArenaRunning,
                        realTraining: realTraining,
                        onRunNow: {
                            SessionLogger.shared.log("[BUTTON] Run Arena (popover)")
                            session.arenaTriggerBox?.trigger()
                            arenaSettingsPopover.isPresented = false
                        },
                        onShowHistory: {
                            SessionLogger.shared.log("[BUTTON] Open Arena History")
                            arenaSettingsPopover.isPresented = false
                            showArenaHistorySheet = true
                        }
                    )
                }
            }
        )
    }

    // updateReplayRatioCompensator(snap:) — the outer integral compensator
    // for the replay-ratio controller's per-tick overhead-subtraction bias —
    // moved to SessionController in Stage 4f. The heartbeat calls
    // session.updateReplayRatioCompensator(snap:).

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
            entropyCollapseBelow: TrainingAlarmController.policyEntropyAlarmThreshold,
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
        guard let box = session.arenaTriggerBox, realTraining else { return nil }
        let elapsed = now.timeIntervalSince(box.lastArenaTime)
        let interval = trainingParams.arenaAutoIntervalSec

        // Freeze countdown at the full interval while the model
        // is still in warmup. The training worker keeps the anchor
        // fresh during this phase; the explicit check here avoids
        // a 2-second 'jitter' (e.g. 30:00 -> 29:58 -> 30:00) on the
        // heartbeat as the worker resets the anchor.
        //
        // Source-of-truth: read `trainer.completedTrainSteps` — the
        // same counter the worker's gating loop reads — so the UI
        // and the worker can never disagree on whether warmup has
        // elapsed. `trainingStats.steps` is updated off the heartbeat
        // and lags behind the trainer by up to one tick, which would
        // briefly show "warmup" to the user after the worker had
        // already started ticking the anchor forward.
        let stepsForGate = trainer?.completedTrainSteps ?? 0
        let isWarmup = stepsForGate < trainingParams.lrWarmupSteps
        if isWarmup {
            return interval
        }

        return max(0, interval - elapsed)
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
    nonisolated static func parseDurationSpec(_ raw: String) -> Double? {
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
    nonisolated static func formatDurationSpec(_ seconds: Double) -> String {
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
    // runArenaHistoryRecovery() moved to SessionController in Stage 4r —
    // commandHub.recoverArenaHistoryFromLogs calls session.runArenaHistoryRecovery().

    // The chart-zoom control row moved to `LowerContentView`
    // alongside the chart grid it controls. Keyboard shortcuts and
    // menu items still route through this view's `chartZoomIn()` /
    // `chartZoomOut()` / `chartZoomEnableAuto()` wrappers above so
    // the menu hub plumbing stays in one place.

    // Engine diagnostics (runEngineDiagnostics / runPolicyConditioningDiagnostic +
    // their async runners) moved to SessionController in Stage 4q — the commandHub
    // wiring calls session.runEngineDiagnostics() / session.runPolicyConditioningDiagnostic().

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

    // Sweep methods (startSweep/stopSweep/runSweep/sweepStatsText) moved to
    // SessionController in Stage 4p.


    // MARK: - Training Stats Display

    private func trainingStatsText() -> (header: String, body: String) {
        let dash = "--"

        // Sweep results trump the per-step display. Once a sweep starts or
        // completes, the table is what the user came here for. The sweep
        // formatter produces its own header line (the "Batch Size Sweep"
        // title) as its first line, so we split it off here so callers can
        // render the split-header layout uniformly across modes.
        if sweepRunning || !sweepResults.isEmpty {
            let sweepText = session.sweepStatsText()
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
            if let divSnap = session.selfPlayDiversityTracker?.snapshot(),
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
            let segmentSteps = max(0, stats.steps - checkpoint.trainingStepsAtSegmentStart)
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
            if let ps = session.parallelStats {
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

    // bytesToGB(_:) moved to SessionController in Stage 4p (used by sweepStatsText).

    // MARK: - Background Work

    // `performBuild()` moved to `SessionController` in Stage 4a — call it as
    // `SessionController.performBuild()` from the detached build tasks.
    // performInference(with:state:) moved to SessionController in Stage 4i —
    // call it as SessionController.performInference(with:state:). (runForwardPass
    // uses it for the Run Forward Pass demo; the candidate probe uses it inside
    // session.fireCandidateProbeIfNeeded.)
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

    var selfPlayStatsColumn: SelfPlayStatsColumn {
        let column: (header: String, body: String)?
        if realTraining, let session = session.parallelStats {
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

    fileprivate var trainingStatsColumnView: UpperTrainingStatsColumn {
        let column = trainingStatsText()
        return UpperTrainingStatsColumn(
            header: column.header,
            bodyText: colorizedPanelBody(column.body),
            realTraining: realTraining,
            replayRatioSnapshot: replayRatioSnapshot,
            replayRatioTarget: trainingParams.replayRatioTarget,
            replayRatioAutoAdjust: trainingParams.replayRatioAutoAdjust
        )
    }

    /// The Training Settings chip + popover. All transactional state and
    /// per-field validation now lives on `trainingSettingsPopover`
    /// (`TrainingSettingsPopoverModel`) — editing a field clears its own
    /// error via the model's `didSet`, so this is a single trivial wrapper:
    /// no `.onChange` chain, no `AnyView`. (The old form had to split a
    /// 19-deep `.onChange` chain into five `AnyView` chunks to stay under
    /// the type-checker's per-expression budget; that's all gone.)
    var trainingSettingsChip: some View {
        TrainingSettingsChip(showPopover: $trainingSettingsPopover.isPresented) {
            TrainingSettingsPopover(
                model: trainingSettingsPopover,
                modelID: trainer?.identifier?.description ?? "—",
                sessionStart: checkpoint.currentSessionStart ?? Date(),
                replayRatioCurrent: replayRatioSnapshot?.currentRatio,
                replayRatioComputedDelayMs: replayRatioSnapshot?.computedDelayMs,
                replayRatioComputedSelfPlayDelayMs: replayRatioSnapshot?.computedSelfPlayDelayMs,
                bytesPerPosition: ReplayBuffer.bytesPerPosition
            )
        }
    }

}

// `UpperCumulativeStatusBar` and `UpperTrainingStatsColumn` moved to their own
// files (`App/UpperContentView/UpperCumulativeStatusBar.swift` and
// `…/UpperTrainingStatsColumn.swift`) — one View struct per file.
