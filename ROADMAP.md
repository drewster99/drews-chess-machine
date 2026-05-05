# Roadmap

Long-term goals, deferred work, and notes on decisions.

**Validated against implementation on 2026-05-05.** This roadmap was reconciled
against the actual Swift code under `DrewsChessMachine/DrewsChessMachine/` rather
than against comments or old roadmap assumptions. Items that were implemented
have been moved out of Future Improvements. Items that were rejected or made
obsolete are kept with context in Decisions Not Pursued / Historical Notes so the
original rationale is not lost.

## Future improvements (validated open)

- **`BatchFeedsInput` struct for `ChessTrainer.buildFeeds`.** Still open.
  Current implementation evidence: `ChessTrainer.buildFeeds(batchSize:boards:moves:zs:vBaselines:legalMasks:)`
  in `Training/ChessTrainer.swift` still takes six positional arguments:
  `batchSize`, `UnsafePointer<Float>` boards, `UnsafePointer<Int32>` moves,
  `UnsafePointer<Float>` z outcomes, `UnsafePointer<Float>` value baselines,
  and `UnsafePointer<Float>` legal masks. The same call sites still pass nested
  `withUnsafeBufferPointer` base addresses from random-data sweep code and real
  replay-buffer sample code. The original safety concern is still valid: three
  same-typed `UnsafePointer<Float>` inputs can be silently swapped by a future
  refactor and still produce a shaped batch.

  Planned shape remains unchanged: wrap the inputs in a small `BatchFeedsInput`
  struct with named fields so the compiler binds by name rather than position.
  No behavioral change; pure call-site safety. Re-check `runPreparedStep` at the
  same time if it grows beyond its current `feeds`, `prepMs`, `queueWaitMs`, and
  `totalStart` argument list.

- **Autosave retention pruning.** Still open, with corrected current-state
  details. The old roadmap text was directionally right that autosaves are kept
  indefinitely, but it understated the current persistence payload: modern
  `.dcmsession` directories can include `replay_buffer.bin`, and the current
  replay-buffer file format is v6, so disk growth can be larger than the older
  "model + session.json only" session plan implied.

  Current implementation evidence:
  - `UpperContentView.periodicSaveIntervalSec` is `4 * 60 * 60`, and
    `PeriodicSaveController` schedules 4-hour saves while Play-and-Train is
    armed. The controller defers during arenas and resets its deadline after any
    successful manual, post-promotion, or periodic save.
  - Post-promotion autosave is enabled by `UpperContentView.autosaveSessionsOnPromote = true`.
  - `CheckpointPaths.makeSessionDirectoryName` generates unique names of the
    form `YYYYMMDD-HHMMSS-<sessionID>-<trigger>.dcmsession`; `CheckpointManager`
    refuses to overwrite existing targets.
  - No code path or menu item named "Manage Autosaves", "Trim to last N", or
    equivalent pruning UI was found. The File menu currently exposes "Resume
    Training from Autosave" and "Open Data Folder in Finder", not autosave
    retention management.

  Desired policy remains: manual saves are always preserved; post-promotion and
  periodic autosaves may be pruned beyond the last N (configurable, default on
  the order of 20); pruning should run lazily after successful saves so there is
  no dedicated sweep racing save/load; optional UI can show total disk footprint,
  counts per trigger, and a "Trim to last N" action. Deferred until disk
  footprint is a demonstrated problem; the "never overwrite" invariant remains
  in force until retention is explicitly implemented.

- **Human-vs-model play.** Still open, but the current UI state needed
  correction. The existing File/View command "Play Game" is not human-vs-model:
  `UpperContentView.playSingleGame()` constructs one `ChessMachine`, creates a
  `DirectMoveEvaluationSource(network: network)`, then creates both White and
  Black as `MPSChessPlayer` instances pointing at the same champion network.
  `startContinuousPlay()` does the same in a loop. A search found no `HumanPlayer`,
  user-move player, slot picker, side-to-play picker, or UI path that lets a
  human supply moves.

  The original design goal remains valid: let a human play against either the
  champion or a trainer/candidate snapshot from the UI, for sanity-checking play
  quality and comparing a mid-training trainer against its parent champion.
  Implementation sketch, corrected against current code:
  - Engine side can reuse `ChessMachine`, `ChessGameEngine`, `ChessPlayer`,
    `MPSChessPlayer`, and `DirectMoveEvaluationSource`; a new human-controlled
    `ChessPlayer` implementation or a UI-driven move bridge is still needed.
  - Extend the current Play Game command/path with a model slot picker
    (champion / candidate inference network / frozen trainer snapshot) and a
    side-to-play picker.
  - For trainer snapshots, copy trainer weights into an inference network using
    the existing `exportWeights` → `loadWeights` path. The arena already uses
    this pattern for candidate/champion snapshots, and `ChessNetwork.loadWeights`
    validates tensor count and shape before assignment.
  - Do not block Play-and-Train: human-vs-model should use
    `candidateInferenceNetwork` or a dedicated inference slot, never call
    `trainer.network` directly while SGD continues.
  - If the user wants to preserve a mid-training opponent, expose a named
    snapshot/freeze action rather than tying the game to continually-mutating
    trainer weights.

- **Adaptive learning-rate schedule.** Still open, with important corrections.
  Current implementation is not a full schedule. `TrainingParameters.LearningRate`
  defaults to `5e-5` and is live-tunable/persisted. `ChessTrainer.buildFeeds`
  feeds the effective LR each step after applying two local multipliers: optional
  `sqrtBatchScalingForLR` and linear warmup over `lrWarmupSteps`. The UI lets the
  user edit learning rate and warmup; `[PARAM] learningRate` and
  `[PARAM] lrWarmupSteps` logs are emitted on manual changes. Session restore
  persists/restores `learningRate`, `lrWarmupSteps`, and `sqrtBatchScalingForLR`.

  Not implemented: no `lr_init`, positions-per-decay `τ`, exponential decay,
  promotion multiplier, LR floor, auto/off schedule toggle, schedule read-only
  slider mode, or schedule persistence fields were found. `trainingPositionsSeen`
  exists in `SessionCheckpointState`/logs but is not used to compute LR.

  ### Candidate trigger families (surveyed)

  - **Step-based decay.** LR multiplied by γ every N training steps
    (e.g., γ=0.5 every 100K steps). Predictable, tunable, but blind
    to actual training health.
  - **Plateau detection.** Watch a smoothed loss; when it stops
    decreasing for N consecutive measurements, multiply LR by 0.5.
    Standard "ReduceLROnPlateau" pattern.
  - **Promotion-driven.** Drop LR by a factor on every successful
    arena promotion. The intuition: promotion proves the current
    policy has shifted meaningfully, so subsequent updates should be
    gentler to lock in that progress before the next promotion
    window. Rejected as an upward-bump mechanism: arena failure more
    likely means the candidate diverged in a worse direction (LR too
    hot, or replay overfit) than "stuck in a flat region that a
    bigger step would escape." Raising LR on failure would make those
    cases worse. Promotion-driven *downward* nudge still useful as a
    secondary overlay.
  - **Cosine annealing with restarts (SGDR).** Smoothly decays LR
    across an "epoch" then restarts to the high value. Often
    empirically strong in supervised vision but adds another tunable
    (epoch length) and has no natural "epoch" concept in self-play.
  - **Replay-ratio aware.** Tie LR to the cons/prod ratio so that
    undertraining (cons < prod) doesn't get amplified by a too-hot
    LR.

  ### What AZ-family engines actually use (survey, 2026-04)

  Researched the five canonical systems' published LR schedules,
  normalized to positions-seen so batch-size differences don't
  confuse the comparison:

  - **AlphaZero** (Silver et al., 1712.01815 v1 + Science 2018). SGD
    + momentum 0.9. Start 0.2, step decay 10× per drop at 100K /
    300K / 500K training steps → 0.0002 floor. Batch 4,096 positions.
    Normalized: first drop at ~410M positions, final at ~2.05B, total
    ~2.87B positions across 700K steps. (Go variant starts at 0.02
    rather than 0.2.)
  - **AlphaGo Zero** (Silver et al., Nature 2017). SGD + momentum
    0.9, L2 = 1e-4. Start 1e-2, step decay 10× → 1e-3 → 1e-4.
    Extended Data Table 3 milestones not confirmable from primary
    source in this pass; ELF OpenGo (1902.04522), which explicitly
    reproduces AGZ faithfully, used 500K / 1M / 1.5M minibatches.
    Batch 2,048 positions. Normalized (via ELF): ~1.0B / ~2.0B /
    ~3.0B positions.
  - **Leela Chess Zero** (`lczero-training/tf/configs/example.yaml`).
    `lr_values=[0.02, 0.002, 0.0005]`, `lr_boundaries=[100000,
    130000]` steps, 250-step warmup, 140K total. Batch 2,048.
    Normalized: drops at 205M / 266M positions, total 287M. Caveat:
    that's the documented *example* — T60/T70/T78/T80/BT production
    runs don't publish per-run YAML values anywhere I could find.
    The blog/wiki describe "LR starts high and is occasionally
    reduced" without numbers.
  - **KataGo** (Wu, 1902.10565, and the live `python/train.py`).
    Parameterized in *samples* (= positions), not gradient steps.
    Paper "g170" run: base per-sample 6e-5, batch 256 →
    effective per-batch 0.01536. Warmup: first 5M samples at ⅓ LR.
    For the b20c256 run, one 10× drop late (after ~17.5 days). Not
    really step-decay in the AZ sense — effectively constant LR with
    warmup + one late drop, combined with SWA / EMA of snapshots
    (every ~250K samples snapshot; every ~1M samples a new EMA
    candidate with decay 0.75) doing work the others get from
    discrete drops. Current `train.py` has drifted to base per-sample
    3e-5 with a 9-stage warmup over first 2M samples and a piecewise
    `lr_scale_auto2` multiplier ramping 12× down to 0.05× over ~600M
    samples — much more elaborate than the paper.
  - **MuZero** (Schrittwieser et al., Nature 2020, board-games
    config in the paper's pseudocode). SGD + momentum 0.9, WD 1e-4.
    Start 0.1 (chess/shogi) or 0.01 (Go). **Exponential** decay, not
    step decay:
    `lr = lr_init · 0.1^(training_step / 400_000)`. Batch 2,048; 1M
    total steps. Normalized: LR falls 10× every ~819M positions. At
    1M steps (~2.05B positions) LR has decayed ~316× (chess final
    ≈ 3.2e-4).

  **Synthesis.** Common pattern is step decay at 10× per drop, ~3
  drops total, first drop somewhere between ~200M and ~1B positions
  seen. Nobody in this family uses cosine, cyclic, or warm restarts.
  Batches cluster at 2,048 (AGZ, lc0, MuZero), with AZ doubling to
  4,096 and KataGo going small at 256. Outliers: KataGo in *shape*
  (mostly flat + late drop + EMA), MuZero in *mechanism* (continuous
  exponential vs. discrete drops). At current `learningRate = 5e-5`,
  the project is roughly at KataGo's paper per-sample scale before any
  future schedule overlay.

  ### Chosen design (not yet implemented)

  MuZero-style continuous exponential decay keyed to
  `trainingPositionsSeen` (an invariant under live batch-size
  changes), with a promotion-driven secondary overlay:

  - **Primary schedule**: `lr = lr_init · γ_e^(positions / τ)` with
    `γ_e = 0.1` (10× per τ, matching the family) and default
    `τ = 500M` positions per 10× — puts us in the AZ / MuZero zone.
    Both `lr_init` and `τ` live-tunable in the UI.
  - **Promotion overlay**: on every successful arena promotion,
    additionally multiply LR by ~0.9 (consolidation nudge). Monotonic
    non-increasing — no upward bumps on arena failure. Per the
    rejection above, failure-upward is more likely to hurt than help.
  - **Floor**: 1e-7 so long runs don't collapse to zero.
  - **Warmup interaction**: linear warmup already exists today via
    `lrWarmupSteps` and should compose multiplicatively with the scheduled
    LR if/when the schedule is added.
  - **Manual UI override wins**: "Schedule: auto / off" toggle; with
    off, the slider is authoritative and the schedule is paused
    (preserves the current auto value so re-enabling doesn't snap).
    With on, the slider shows the current scheduled value read-only.
  - **Logging**: every schedule-driven change logged as
    `[PARAM] learningRate <old>→<new> reason=<decay|promotion|warmup>`
    so change history lives alongside `[STATS]` in the session log.
  - **Persistence**: `lr_init`, `τ`, `γ_promotion`, schedule on/off,
    and the last-computed LR should live in `session.json` so a reload
    resumes at the same scheduled value rather than jumping.

  Deliberately *not* doing: step-milestone decay (operates on
  training steps, which live-tunable batch size makes non-invariant),
  plateau-on-loss (`pLoss` is outcome-weighted and unbounded — unreliable
  plateau signal), replay-ratio aware (the `ReplayRatioController` already
  handles cons/prod; doubling up couples two control loops with no clear
  benefit), cosine/SGDR (no natural epoch in self-play; adds a tunable
  without a principled way to set it).

- **Compiled `MPSGraphExecutable`.** Still open. `ChessNetwork.evaluate`,
  `ChessNetwork.evaluate(batchBoards:count:)`, export/load helpers, BN stats,
  and `ChessTrainer.runPreparedStep` still call `graph.run(...)` directly.
  `ChessMPSNetwork.NetworkInitMode.package(URL)` still throws
  `ChessMPSNetworkError.packageLoadingNotImplemented`, so package loading is not
  implemented. Pre-compiling via `graph.compile(...)` may still remove per-call
  executable-cache lookup and provide a reusable serialized executable for the
  package path, but this should be revisited only after measuring current
  steady-state `graph.run` overhead.

## Completed / corrected from older Future entries

- **Model and session save/load — implemented, with scope expanded beyond the
  original future plan.** The old Future entry said "Today nothing persists
  across app launches — quit mid-training and you lose the champion, the trainer,
  every accumulated counter, and the replay buffer." That statement is now
  historical, not current. The original design context is preserved below, with
  corrections against current code.

  **Single model — `.dcmmodel` (flat binary file), implemented.** The original
  plan was: wrap one network's weights plus identity and metadata in a fixed
  binary header, then the tensors from `ChessMPSNetwork.exportWeights()` /
  `ChessNetwork.exportWeights()` in declared order, then a trailing 32-byte
  SHA-256 over all preceding bytes for integrity. Header carries magic
  `"DCMMODEL"`, format version, `archHash` (hash of filters / blocks / input
  channels / policy dim — hard-refuses to load on mismatch, no migration),
  `numTensors` sanity-check, creation wall-clock time, `ModelID`, parent
  `ModelID` at time of save, and a JSON metadata blob (arena stats at mint,
  training-step count, creator tag). Loadable into any training- or
  inference-mode `ChessNetwork` via the existing `loadWeights` path — this is
  the unit for "take any model at any point and use for inference."

  Current code evidence: `ModelCheckpointFile.swift` implements `.dcmmodel`
  encode/decode with magic/version/arch hash, tensor count, metadata, weights,
  and SHA-256 validation. `ChessNetwork.exportWeights()` returns the current
  trainable variables plus BN running stats in declared order; current tensor
  count is validated dynamically by `loadWeights` rather than being hard-coded in
  ROADMAP. `ChessNetwork.loadWeights(_:)` checks tensor count and each tensor's
  element count before assigning through prebuilt load placeholders.

  **Training session — `.dcmsession` (directory), implemented and expanded.**
  The original plan was a directory holding `champion.dcmmodel`,
  `trainer.dcmmodel`, and `session.json`, rather than a custom bundle, so (a)
  extraction is free — Finder-copy any model out of a session — and (b) only one
  binary model format needs debugging. `session.json` was planned as a Codable
  blob with stable `sessionID`, format version, save and session-start
  wall-clock timestamps, accumulated training time, STATS-line counters
  (`trainingSteps`, `selfPlayGames`, `selfPlayMoves`, `trainingPositionsSeen`),
  hyperparameters appearing in the arena footer (batch, lr, promote threshold,
  arena games, self-play/arena tau configs, self-play worker count), both
  network IDs duplicated from `.dcmmodel` headers for fast index reads, and light
  arena history (W/L/D + kept/promoted + step-at-run for each arena so far).

  Current code evidence: `SessionCheckpointState` now contains all of the above
  and more: game-result breakdown, replay-ratio settings, step delay / last
  auto-computed delay, LR warmup, sqrt-batch LR scaling, replay-buffer min
  positions, arena auto interval, candidate probe interval, legal-mass collapse
  thresholds, build metadata, replay-buffer presence/counters, training segments,
  arena concurrency, and expanded arena side-breakdown fields. `CheckpointManager`
  writes `champion.dcmmodel`, `trainer.dcmmodel`, `session.json`, and, when
  requested, `replay_buffer.bin`.

  **Important correction to the original v1 exclusions.** The original plan
  excluded the 500k-position replay buffer (~2.3 GB / later noted as 4.6 GB)
  because resume warmup/refill was considered acceptable, excluded the candidate
  network because it only exists mid-arena, and excluded in-flight self-play
  games because workers abandon on save like Stop. Current code no longer
  excludes replay-buffer contents when `state.hasReplayBuffer == true` and a
  `ReplayBuffer` is passed: `CheckpointManager.saveSession` writes
  `replay_buffer.bin`, updates `session.json` replay counters from the exact
  `ReplayBuffer.write(to:)` snapshot, and verifies by restoring into a scratch
  buffer. Candidate network and in-flight games remain excluded from session
  state.

  **Save triggers, implemented with naming updates.** Original plan: Menu items
  Save Session, Save Champion as Model, Load Session, Load Model; autosave on
  arena promotion defaults on; Save Session disabled mid-arena; Load Session and
  Load Model require Play-and-Train to be stopped. Current File menu implements
  Save Session, Save Champion, Load Session, Load Model, Load Parameters, Save
  Parameters, Resume Training from Autosave, and Open Data Folder in Finder.
  Save Session is disabled unless real training is active and no arena/save is
  running. Load Session/Model are disabled during real training, continuous play,
  continuous training, sweep, game play, build, or save-in-flight. Post-promotion
  autosave is enabled by `autosaveSessionsOnPromote = true`, and periodic
  4-hour autosave is also implemented.

  **File locations, implemented with corrected session naming.** Original plan:
  all saves — manual and auto — land under fixed Library paths:
  `~/Library/Application Support/DrewsChessMachine/Sessions/` for sessions and
  `~/Library/Application Support/DrewsChessMachine/Models/` for single models.
  Every save keeps the old file; nothing is overwritten; users prune manually.
  The planned naming scheme was `<YYYYMMDD-HHMMSS>-<modelID>-<trigger>.<ext>`
  where trigger is `manual` or `promote`; wall-clock prefix gives natural Finder
  sort order; a reveal/open button makes the hidden `Application Support`
  location discoverable; load uses a file importer so files can be loaded from
  anywhere.

  Current code evidence: `CheckpointPaths.rootURL`, `sessionsDir`, and
  `modelsDir` implement the Library paths. `CheckpointPaths.makeFilename` uses
  `<timestamp>-<modelID>-<trigger>.<ext>` for standalone models.
  `CheckpointPaths.makeSessionDirectoryName` uses
  `<timestamp>-<sessionID>-<trigger>.dcmsession` for sessions, so multiple
  autosaves for the same run cluster by stable session ID rather than by a fresh
  model ID. `CheckpointManager` refuses overwrites with target-exists guards.
  The UI command is currently named "Open Data Folder in Finder" rather than the
  originally proposed "Reveal Saves in Finder".

  **Every save is self-verified before it is marked successful — implemented and
  hardened.** Original plan: after atomic writing (tmp + fsync + rename), re-read
  the file(s), bit-compare tensors against exported `[[Float]]`, load weights
  into a throwaway `ChessMPSNetwork`, run forward pass on canonical test
  positions (starting position + one fixed mid-game FEN), and compare policy and
  value outputs bit-exactly to the source network. Any mismatch deletes the fresh
  `.tmp`, leaves prior saves untouched, and surfaces a user-visible error.

  Current code evidence: `CheckpointManager.saveModel` and `saveSession` perform
  model verification via `verifyModelFile`, session JSON decode round-trip,
  replay-buffer scratch restore/counter comparison when a replay buffer is
  present, `F_FULLFSYNC` on files/directories, tmp staging, atomic final rename,
  parent-directory sync, and launch-time orphan cleanup of interrupted `.tmp`
  artifacts. See the existing Completed entry "Session durability hardening —
  saved means golden" for the full durability pipeline.

  **Original validation checklist, status after implementation.**
  (1) Build succeeds — covered by current project/test workflow, not rerun by
  this documentation-only roadmap edit. (2) Round-trip a single model: Save
  Champion → quit → relaunch → Load Model → run Forward Pass on a fixed FEN →
  policy/value bit-exact to pre-save — supported by model encode/decode,
  `loadWeights`, and save verification. (3) Round-trip a session:
  Play-and-Train → Save Session → quit → relaunch → Load Session → counters and
  ModelIDs match → champion/trainer forward outputs bit-exact → Play-and-Train
  resumes and later arena can promote — supported by session state restore and
  model verification; replay buffer now can restore rather than always refill.
  (4) Arch-mismatch file refuses to load with a clear error — implemented via
  `.dcmmodel` arch hash. (5) Truncated `.dcmmodel` refuses to load. (6) SHA
  mismatch in `.dcmmodel` refuses to load. (7) Save-mid-arena is disabled or
  errors clearly — Save Session menu is disabled while `isArenaRunning`. (8)
  Save atomicity under `SIGKILL` while writing `.tmp` leaves prior saves intact
  and launch cleanup removes orphans — implemented via tmp staging, no-overwrite
  final rename, fsyncs, and `CheckpointPaths.cleanupOrphans()`. (9) Existing
  tests should still pass — not rerun for this roadmap-only task.

  **Session restore coverage — original table corrected against current code.**

  | Field | Save | Restore / current status |
  |---|---|---|
  | Champion + trainer weights | `.dcmmodel` files | loaded into networks |
  | Champion + trainer model IDs | `session.json` | restored to identifiers |
  | Session ID | `session.json` | inherited for continuity |
  | Elapsed training time | `session.json` | back-dated `sessionStart` anchor / training segments now add more context |
  | Training step count | `session.json` | seeded into stats/trainer state |
  | Self-play games / moves | `session.json` | seeded into `ParallelWorkerStatsBox` / display state |
  | Game results (W/B checkmates, stalemate, 50-move, 3-fold, insuff. material) | `session.json` Optional fields | restored when present, back-compatible when absent |
  | Learning rate | `session.json` | restored to `TrainingParameters` + trainer |
  | LR warmup + sqrt-batch LR scaling | `session.json` Optional fields | restored when present; this is newer than original table |
  | Replay ratio target + auto-adjust toggle | `session.json` Optional fields | restored to live parameters/controller state when present |
  | Step delay + last auto-computed delay | `session.json` Optional fields | restored when present |
  | Self-play worker count | `session.json` | restored/clamped to runtime worker bounds |
  | Arena concurrency | `session.json` Optional field | restored/clamped; newer than original table |
  | Arena/candidate/legal-mass tuning fields | `session.json` Optional fields | restored when present; newer than original table |
  | Build metadata | `session.json` Optional fields | displayed/used for forensic context; newer than original table |
  | Training segments | `session.json` Optional array | restored/summed for active-training-time history; newer than original table |
  | Arena history (W/L/D, score, promoted flag per arena) | `session.json` | rebuilt into `tournamentHistory`; side breakdown fields are optional/back-compatible |
  | Replay buffer contents | `replay_buffer.bin` when `hasReplayBuffer == true` | restored via `ReplayBuffer.restore(from:)` and cross-checked against `session.json`; older sessions without a buffer still resume by refilling |
  | Progress rate chart samples | not saved as the original table said | rebuilds from new data |
  | Rolling loss windows | not saved as the original table said | rebuilds from new steps |

- **Legal-move masking in the training policy loss — implemented for training,
  not for inference.** The old Future item "Fuse legal-move masking into the
  policy head" said the graph emitted a full policy and the CPU masked illegal
  moves. That is no longer fully accurate. `ChessTrainer.buildTrainingOps` now
  creates a `legal_move_mask` placeholder, builds `masked_logits =
  network.policyOutput + (1 - legalMask) * -1e9`, and feeds `maskedLogits` to
  `graph.softMaxCrossEntropy(...)`. `ChessTrainer.buildFeeds` writes the
  `legalMasks` pointer into a cached `legalMaskND` and includes
  `legalMaskPlaceholder` in the feeds dictionary. Thus the training loss's
  softmax is graph-masked.

  Inference remains intentionally unmasked at the network boundary:
  `ChessNetwork.evaluate` returns raw 4864 logits, `ChessRunner` softmaxes for
  the Forward Pass demo, and `MPSChessPlayer` samples over legal moves using the
  move list it is given. Keeping raw logits visible preserves diagnostics such
  as illegal-mass/top-cell collapse; do not describe current training as using a
  CPU-renormalized policy loss.

## Decisions not pursued / historical notes

- **Inference-side graph legal-mask softmax is not currently being pursued.**
  Because training now masks logits in-graph and inference diagnostics benefit
  from seeing illegal raw logits, the remaining version of this idea is only an
  inference-path optimization/design change: add a legal-mask feed to inference
  and return already-normalized legal probabilities. That would hide illegal-mass
  telemetry unless a raw-logit path were retained. Keep the current raw-logit
  inference contract unless a measured hot path needs graph-side inference
  masking.

- **Partial heap / quickselect for top-k policy moves is not worth changing now.**
  The original text cited a 4096-entry policy vector. Current architecture v2
  has `ChessNetwork.policySize = 76 * 8 * 8 = 4864`, and
  `ChessRunner.extractTopMoves` full-sorts the policy indices. That full sort is
  intentional after the catastrophic-collapse fix: sorting the whole vector
  guarantees enough on-board decoded moves even when the top cells are off-board.
  The path is the Forward Pass / Candidate Test UI path, not the self-play hot
  path. Revisit a heap/quickselect only if top-k extraction moves into a per-ply
  search/hot loop, and preserve the full-vector/off-board robustness.

- **Old per-worker-network self-play topology is historical.** The existing
  Completed section preserves the original N-worker design in detail, but current
  runtime uses a shared `BatchedMoveEvaluationSource` rather than
  `secondarySelfPlayNetworks`. Treat that section as context, not current
  architecture.

## Tech debt / migrations to remove

- **Drop v1 trainer.dcmmodel zero-pad migration** *(added 2026-05-04;
  remove after 2026-06-04)*. Trainer state persistence (Polyak momentum
  velocity) was added with `ModelCheckpointFile` format version 2,
  bumping from v1 (trainables + bn) to v2 (trainables + bn + velocity).
  The decoder accepts both versions; the trainer's
  `loadTrainerWeights(_:)` count-detects v1 files and leaves velocity
  at zero-init. After 2026-06-04, any in-flight v1 trainer.dcmmodel
  files should have been re-saved as v2 (a single Save Session
  re-emits with the new format), so the v1 acceptance branch can be
  removed:
  - Tighten `ModelCheckpointFile.supportedReadVersions` to `[2]` only.
  - Remove the `weights.count == v1Count` branch in
    `ChessTrainer.loadTrainerWeights(_:)`.
  - Remove the `// TODO(persist-velocity, after 2026-06-04)` marker
    comments in both files.

## Findings

- **Batch-size sweep is reliable at 1 s per batch size.** The Batch Size
  Sweep panel runs a training-mode copy of the network through real SGD
  steps at each batch size and reports steady-state throughput. We tried
  longer per-size windows (15 s, 5 s, 3 s, 1.5 s) and found 1 s gives
  essentially the same shape and the same winner — the fast-warming MPSGraph
  caches mean each row converges within a handful of steps and the tail just
  accumulates redundant samples. Keeping it at 1 s makes the whole sweep
  cheap enough to run any time on a new machine to pick the most efficient
  batch size for *that* hardware, rather than baking a single number in.

- **Sweep memory guard is empirical, not architectural.** The sweep refuses
  to run a batch size whose predicted resident footprint exceeds 75 % of
  `min(recommendedMaxWorkingSetSize, maxBufferLength)`, or whose largest
  single buffer would exceed `maxBufferLength`. The prediction comes from
  a least-squares linear fit over the (batch, peak `phys_footprint`) pairs
  already observed during the same sweep — no per-architecture fudge
  factor. Peak `phys_footprint` is sampled by the UI heartbeat (~10 Hz)
  plus once at the start and end of each row, so we catch transient spikes
  during a step rather than relying on `MTLDevice.currentAllocatedSize`,
  which is post-step and undercounts. Skipped rows still appear in the
  table with the prediction and the reason they were skipped, so the
  sweep walks the full ladder and makes its limits visible.

- **First decisive arena promotion under the autotrain loop
  (2026-04-30, `experiments/20260430-170725/`, accepted as commit
  `42c35c9`).** A 2400 s Play-and-Train run produced one promotion at
  arena #3 of 4. Worth preserving in detail because it's the first
  arena result during automated parameter tuning where the candidate
  was clearly stronger than the champion rather than a coin-flip
  hovering around 0.50. Build 403, champion `20260430-53-gNPD`,
  candidate `20260430-53-gNPD-1` (promoted), trainer
  `20260430-53-gNPD-2`.

  Training/arena parameters in effect for this run:

  | Parameter                                          | Value     |
  |----------------------------------------------------|-----------|
  | `learning_rate`                                    | 5e-05     |
  | `lr_warmup_steps`                                  | 30        |
  | `K` (policy loss scale)                            | 5         |
  | `entropy_bonus`                                    | 0.016     |
  | `weight_decay`                                     | 2e-04     |
  | `grad_clip_max_norm`                               | 25        |
  | `draw_penalty`                                     | 0.1       |
  | `training_batch_size`                              | 4096      |
  | `self_play_workers`                                | 48        |
  | `replay_ratio_target` (auto-adjust on)             | 1.1       |
  | `replay_buffer_capacity`                           | 500 000   |
  | `replay_buffer_min_positions_before_training`      | 75 000    |
  | `self_play_start_tau` → `target_tau` / decay/ply   | 2.0 → 0.8 / 0.03 |
  | `arena_start_tau` → `target_tau` / decay/ply       | 2.0 → 0.5 / 0.01 |
  | `arena_promote_threshold`                          | 0.55      |
  | `arena_games_per_tournament`                       | 100       |
  | `arena_auto_interval_sec`                          | 300       |
  | `candidate_probe_interval_sec`                     | 15        |
  | `legal_mass_collapse_threshold` / grace / probes   | 0.999 / 600 s / 8 |
  | `training_time_limit` (this run window)            | 2400 s, 1427 trainer steps |

  Per-arena results (each tournament = 100 games, 50 as White +
  50 as Black; "W-D-L" is candidate-relative):

  | # | Finished @ step | W-D-L (cand) | White (W-D-L) | Black (W-D-L) | Score | Score CI95     | Elo | Elo CI95     | Promoted |
  |---|-----------------|--------------|---------------|---------------|-------|----------------|-----|--------------|----------|
  | 1 | 179             | 7-85-8       | 5-39-6        | 2-46-2        | 0.495 | [0.457, 0.533] | −3  | [−30, +23]   |          |
  | 2 | 528             | 10-83-7      | 6-38-6        | 4-45-1        | 0.515 | [0.475, 0.555] | +10 | [−18, +39]   |          |
  | 3 | 866             | 19-76-5      | 8-40-2        | 11-36-3       | 0.570 | [0.524, 0.616] | +49 | [+17, +82]   | ✅       |
  | 4 | 1175            | 6-85-9       | 3-43-4        | 3-42-5        | 0.485 | [0.447, 0.523] | −10 | [−37, +16]   |          |

  Score / Elo confidence intervals are the Wald 95% CI computed in
  `ArenaEloStats.summary` from per-game outcomes in {1, 0.5, 0}; Elo
  CI is the score CI mapped through `400·log10(p/(1−p))`. Promotion is
  gated on the point estimate vs `arena_promote_threshold`, not on the
  CI.

  Why arena #3 is decisive rather than borderline:

  - 19 wins vs 5 losses (24 decisive games; candidate took 79 % of them).
  - Score 0.570 with CI95 [0.524, 0.616] — the entire CI sits above
    0.50; the lower bound dips just under the 0.55 promote line but the
    point estimate clears it cleanly.
  - Elo +49 with CI95 [+17, +82] — even the lower bound is +17 Elo, so
    "candidate is genuinely stronger" is well-supported, not noise.
  - Balanced across colors (8 wins as White, 11 as Black) rather than
    one-sided color luck.

  The surrounding arenas (#1, #2, #4) all sit inside [0.485, 0.515] with
  CIs straddling 0.50 by a wide margin — typical noise-floor draws when
  two near-equivalent networks face off (draw rates 76–85 %). Arena #3
  is cleanly separated from that floor. Useful as a reference point for
  what a real training-driven promotion looks like in this engine, vs
  the borderline 0.50–0.53 promotions seen earlier in the project's
  history (e.g. the 5-arena run at scores `[0.51, 0.525, 0.515, 0.52,
  0.5075]` from the 2026-04-21 BN-warmup CHANGELOG entry, which the
  team correctly diagnosed as a stuck network rather than real
  progress).

  Followup pure-window-extension run (2700 s,
  `experiments/20260430-184042/`, accepted as commit `be9d2d3`)
  produced 0 promotions across 5 arenas (scores 0.51 / 0.535 / 0.53 /
  0.47 / 0.485) but dramatically healthier end-of-run policy state
  (max prob 0.150 vs baseline 0.998, illegal_mass 0.678 vs 1.000,
  pEnt 6.44 well above the 5.0 alarm threshold) — the autotrain goal
  axis ("longer training without full collapse") favored the longer
  window despite no promotion, on the principle that a healthy
  policy-head distribution is a prerequisite for future promotions.

## Completed

- **Full parameter coverage in session save + Load/Save Parameters
  menu items + slow-save watchdog (2026-04-30).** Three coupled
  changes that close the parameter-reproducibility gap and add a
  save observability backstop. Original design captured in this
  ROADMAP under Future improvements, then implemented in the same
  session.

  **What landed:**
  - Eight new Optional fields on `SessionCheckpointState`
    (`SessionCheckpointFile.swift`): `lrWarmupSteps`,
    `sqrtBatchScalingForLR`, `replayBufferMinPositionsBeforeTraining`,
    `arenaAutoIntervalSec`, `candidateProbeIntervalSec`,
    `legalMassCollapseThreshold`, `legalMassCollapseGraceSeconds`,
    `legalMassCollapseNoImprovementProbes`. All Optional →
    older `.dcmsession` files decode unchanged with new fields nil.
    `buildCurrentSessionState` populates them; `startRealTraining`
    resume code reads them with `if let v = rs.foo { … = v } else { … = currentAppStorageValue }`
    fallback. Each restored field also writes back to its
    `@AppStorage` mirror so the UI shows what the session was
    actually running with, not what the user's current global
    preference happens to be.
  - `[RESUME-PARAM]` log lines added for every restored field
    (both the eight new ones and the existing pre-expansion ones
    `learning_rate`, `entropy_bonus`, `draw_penalty`,
    `weight_decay`, `grad_clip_max_norm`, `K`). Lines fire only
    when the saved value is present and valid — older sessions
    falling through to `@AppStorage` stay silent. Format:
    `[RESUME-PARAM] <field>: <before> -> <after> (from session)`.
  - `CliTrainingConfig` promoted from `Decodable` to `Codable`
    (`CliTrainingConfig.swift`) with a new `encodeJSON()` helper
    using `.prettyPrinted, .sortedKeys` for stable, diffable output.
    Optional fields with nil values omit cleanly via Swift's
    synthesized `encodeIfPresent`.
  - Two new File menu items wired through `AppCommandHub`:
    `Load Parameters…` (file picker → decode `CliTrainingConfig`
    → call `applyCliConfigOverridesFromMenu(cfg:)` which routes
    through the same `applyCliConfigOverrides(cfg:)` the launch
    `--parameters` flag uses) and `Save Parameters…` (file
    exporter → build a fully-populated `CliTrainingConfig` via
    `currentParametersConfig()` → encode JSON via `CliParametersDocument`
    `FileDocument` adapter). Load Parameters is disabled during
    realTraining / continuousPlay / continuousTraining / sweep /
    game-in-progress / building / save-in-flight, matching Load
    Session / Load Model. Save Parameters is always enabled (no
    destructive effects).
  - `applyCliConfigOverrides` refactored: no-arg overload reads
    `cliConfig` (launch path); new `applyCliConfigOverrides(cfg:)`
    parameterized variant takes a config directly; new
    `applyCliConfigOverridesFromMenu` is the menu's named entry
    point. All three return `[ParameterOverrideChange]` — a
    typealias for `(label: String, before: String, after: String)`
    — used by the menu handler to surface count and field labels
    in the status row: `Loaded <file>: N parameters changed
    (label1, label2, …)`. Per-field `[APP] --parameters override:`
    log lines were already there.
  - New `.slowProgress` case on `CheckpointStatusKind` (orange
    text + `clock.badge.exclamationmark.fill` icon, 120-second
    auto-clear). `slowSaveWatchdogSeconds = 10` constant.
    `startSlowSaveWatchdog(label:)` and `cancelSlowSaveWatchdog()`
    helpers. Wired into all four save sites: manual + periodic
    (`saveSessionInternal` via the shared `clearInFlight` helper),
    post-promotion (inline arena-coordinator task), and
    `Save Champion as Model` (`handleSaveChampionAsModel`). Each
    save's completion path (success, failure, timeout, export
    error) cancels the watchdog so a fast save's body never runs.
    A slow save logs `[CHECKPOINT-WARN] <label> still running
    after 10s — disk busy or replay buffer large?` and updates
    the status row to amber with a `(still running, 10s+)` suffix.
    Fires exactly once per save — no progressive warnings —
    because completion will eventually flip the row to
    success/error and restore normal styling.
  - Watchdog deadline tuned from 5 s (initial spec) to 10 s
    (final shipped value) after considering that the
    post-promotion save runs at `.utility` priority and could be
    delayed under load. 10 s leaves headroom against
    false-positive warnings while still surfacing genuinely stuck
    saves promptly.

  **Tests added:**
  - `CliTrainingConfigTests.testEncodeDecodeRoundTripPreservesEveryField`:
    every field round-trips through `encodeJSON()` → decode.
  - `CliTrainingConfigTests.testEncodeJSONUsesSortedKeys`: pins
    the sorted-keys output so the UI-saved file diffs cleanly
    against an autotrain-saved file with the same values.
  - `CliTrainingConfigTests.testEncodeJSONOmitsNilFields`: pins
    `encodeIfPresent` semantics — partial configs produce
    partial files.
  - `SessionCheckpointSchemaExpansionTests.testRoundTripPreservesAllExpansionFields`:
    8 new schema fields encode → decode cleanly.
  - `SessionCheckpointSchemaExpansionTests.testLegacySessionWithoutExpansionFieldsDecodes`:
    older `.dcmsession` files without the new keys still decode,
    with new fields nil — back-compat pin.
  - `SessionCheckpointSchemaExpansionTests.testCrossFormatKeysAreIndependent`:
    snake_case in `parameters.json` and camelCase in
    `session.json` decode independently.
  - Existing `testAllFieldsDecode` and
    `testPartialJsonLeavesMissingFieldsNil` extended to cover the
    new fields.

  **Cross-format invariant achieved:** an autotrain `parameters.json`
  is directly loadable in the UI; a UI-saved parameters file is
  directly usable as `--parameters` input to the CLI. Same Codable
  shape, same field names, same units.

  **Deviations from the original plan:** none of substance. The
  watchdog deadline went from 5 → 10 s during implementation per
  user direction. The `Save Parameters…` menu item is always
  enabled (no `networkReady` gate) — minor concession noted in
  review, since a defaults-dump can be useful as a starting
  template even before any model is built.

- **Engine-level legal-move validation (2026-04-20).** Previously
  `ChessGameEngine.applyMoveAndAdvance` trusted the caller to supply a
  legal move, and `MoveGenerator.applyMove` would trap on a force-unwrap
  for moves whose from-square was empty. The argument for the
  performance shortcut was that the game loop in `ChessMachine` already
  generates the legal-move list per ply for player choice and
  end-detection, so re-deriving it inside apply would be wasted work.
  That trust held in practice — the only caller was `ChessMachine`,
  which always sampled from `MoveGenerator.legalMoves(for:)` — but it
  left the engine unsafe for any future caller (UI drag-drop, loaded
  PGN, network input, a buggy player) to invoke directly.

  The fix preserves the one-`legalMoves`-call-per-ply invariant by
  having the engine *own* the legal-move list rather than duplicating
  it on the caller side. `ChessGameEngine` gained a
  `private(set) var currentLegalMoves: [ChessMove]` that is computed
  once at init (seeded from the starting state) and refreshed inside
  `applyMoveAndAdvance` after each successful move — using the same
  `nextMoves = MoveGenerator.legalMoves(for: state)` call that already
  powered end-of-game detection. `applyMoveAndAdvance` now guards
  `currentLegalMoves.contains(move)` before apply and throws the new
  `ChessGameError.illegalMove(ChessMove)` if the guard fails
  (`ChessMove` is already `Equatable`). No extra `legalMoves` calls on
  the hot path.

  `ChessMachine.runGameLoop` dropped its local `var currentLegalMoves`
  and now reads `engine.currentLegalMoves` both when calling
  `player.onChooseNextMove(...)` and implicitly through the engine's
  self-refresh. Illegal moves from a buggy player flow through the
  existing `playerErrored` + break path in the game loop, identical to
  any other thrown error — partial-game positions up to the failure
  point still flush to the replay buffer as before. `applyMoveAndAdvance`
  retains `@discardableResult` on its `[ChessMove]` return (kept for
  callers who prefer the inline value, e.g. tests), so the change is
  API-additive rather than breaking. Callers that used `try?` to
  discard errors (the ContentView sanity-check knight-shuffle probe,
  `RepetitionTrackingTests` paths) continue to work: their moves are
  legal so validation passes, and any genuine illegal move is now
  surfaced through the same error-swallowing behavior that already
  existed for `gameAlreadyOver`.

  Build green. This closes the roadmap item of the same name.

- **Session durability hardening — "saved means golden" (2026-04-20).**
  Closes TODO_NEXT.md #3. The save pipeline now guarantees that either
  a fully-verified, fsync'd `.dcmsession` bundle appears on disk under
  its final name, or nothing appears with that name. Restored
  sessions are bit-identical to what was saved, or loading fails with
  a specific error describing which check tripped.

  **Principle.** If any piece of a session save cannot be fully
  verified end-to-end, the whole save fails, all partials are removed,
  and no final-named artifact appears on disk. Built as a coordinated
  change across `ReplayBuffer.swift`, `CheckpointManager.swift`,
  `DrewsChessMachineApp.swift`, and `ContentView.swift`, plus docs in
  `replay_buffer_file_format.md` (new) and `dcmmodel_file_format.md`
  (new), and five new XCTest cases in `ReplayBufferTests.swift`.

  ### ReplayBuffer format v3 → v4

  `ReplayBuffer.fileVersion` bumped 3 → 4. Readers reject v1/v2/v3
  cleanly with `PersistenceError.unsupportedVersion`. No migration
  path — matches the project's delete-and-retrain stance (the user's
  existing v3 `.replay_buffer.bin` files will not load; the replay
  buffer is recovered by resumed self-play).

  - **SHA-256 trailer.** 32-byte digest appended over header + all
    four body sections, verified before any header field is trusted
    at load time. Mirrors the `.dcmmodel` integrity-trailer
    convention. Computed streaming during write (CryptoKit
    `SHA256.update(data:)` per chunk), so no extra hashing pass — the
    bytes fed to `handle.write(...)` are also fed to the hasher.
  - **Strict file-size equality check.** Restore computes
    `expectedBytes = headerSize(56) + storedCount × (floatsPerBoard ×
    4 + 12) + trailerSize(32)` and requires `actualFileSize ==
    expectedBytes`. Uses `==`, not `>=`, because the format is fully
    deterministic — any deviation is corruption. New
    `PersistenceError.sizeMismatch(expected, got)`.
  - **Upper-bound sanity caps.** Applied before any allocation or
    seek arithmetic so a corrupted header can't drive a
    multi-terabyte allocation or overflow the size computation.
    Caps: `floatsPerBoard ≤ 8_192`, `capacity ≤ 10_000_000`,
    `storedCount ≤ 10_000_000`. New
    `PersistenceError.upperBoundExceeded(field, value, max)`.
  - **`handle.synchronize()` before close.** In `_writeLocked`,
    forces APFS to flush dirty pages to the device before the file
    handle closes. On top of this, `CheckpointManager.saveSession`
    adds `fcntl(F_FULLFSYNC)` via `fullSyncPath` for drive-cache-bypass
    durability.
  - **Atomic write-and-snapshot.** `ReplayBuffer.write(to:)` now
    returns the `StateSnapshot` reflecting exactly the state
    serialized into the file, captured under the same `queue.sync`
    lock that serializes the write. Post-save verification compares
    against this value — a subsequent `stateSnapshot()` call would
    diverge because concurrent self-play workers resume appending
    after the save-gate releases the trainer pause. Annotated
    `@discardableResult` so existing callers (tests, etc.) compile
    unchanged. **This was a recheck-catch** — the first pass
    compared against a freshly-called `live.stateSnapshot()` and
    would have spuriously failed the counter comparison every time
    saves happened during active training.

  ### `CheckpointManager` durability pipeline

  `saveSession` now runs a full fsync pipeline on top of the existing
  tmp-dir-then-rename atomicity:

  1. Write all four files (two `.dcmmodel`, `session.json`,
     `.replay_buffer.bin`) into a `.tmp` staging directory.
  2. `fullSyncPath` each of the four files — issues `fcntl(fd,
     F_FULLFSYNC)` on Apple filesystems (falls back to `fsync` if
     unsupported). Bypasses the drive's write cache, not just the VFS
     page cache.
  3. Verify:
     - Both `.dcmmodel`: existing bit-exact + forward-pass
       round-trip (unchanged).
     - `session.json`: existing decode round-trip (unchanged).
     - **NEW** — `.replay_buffer.bin`: re-load into a scratch
       `ReplayBuffer` via `restore(from:)`. The scratch restore runs
       the full v4 verification stack (SHA, size, caps). Scratch is
       allocated at `max(1, writtenSnap.storedCount)` — sized to the
       actual data, not to the live ring's full capacity — so a
       half-full 1 M-slot ring does not double peak memory during
       verify. Then compare the restored `storedCount` and
       `totalPositionsAdded` against the `writtenSnap` captured
       atomically from the write. Drift here implies a write-path
       regression that produced a valid-SHA file with wrong bytes
       (the SHA alone cannot catch this class of bug if the write is
       internally consistent but semantically wrong).
  4. `fullSyncPath(tmpDir)` — flush directory-entry metadata.
  5. `fm.moveItem(tmpDir, finalDir)` — atomic rename.
     `FileManager.moveItem` aborts if the destination already
     exists (unlike POSIX `mv`); plus the existing
     `fileExists(finalDirURL)` guard. Two independent guards.
  6. `fullSyncPath(CheckpointPaths.sessionsDir)` — flush the parent
     so the rename itself is durable. A failure here leaves the
     session visible (rename already committed) and logs a warning
     that the parent-directory flush wasn't guaranteed — we do not
     remove the session, as it's already the "best we've got" on
     disk.

  Any failure in steps 1–4 triggers `cleanupTmp()` and throws. No
  final-named artifact appears on disk.

  `saveModel` gets the same treatment: `fullSyncPath` on the tmp file
  before verify-and-rename, and `fullSyncPath` on
  `CheckpointPaths.modelsDir` after rename.

  **New `CheckpointManagerError` cases:** `fsyncFailed(URL, Error)`,
  `replayVerificationFailed(String)`, `sessionReplayMismatch(detail:
  String)`.

  ### Load-time cross-check

  New helper `CheckpointManager.verifyReplayBufferMatchesSession(buffer:
  state:)` runs after `ReplayBuffer.restore(from:)` at session load
  time. Compares `buffer.stateSnapshot().totalPositionsAdded` against
  `state.replayBufferTotalPositionsAdded` from `session.json`.
  Mismatch throws `CheckpointManagerError.sessionReplayMismatch` and
  surfaces in the load UI.

  Only the lifetime counter is cross-checked, not `storedCount` or
  `capacity` — those two intentionally diverge when loading a larger
  saved ring into a smaller live one (existing restore
  `skip = fileStored - target` logic). `totalPositionsAdded` survives
  that logic verbatim and is effectively unique across sessions, so a
  mismatch strongly implies a file-pairing error (wrong replay paired
  with wrong session.json) or SHA-collision-scale corruption.
  `replayBufferTotalPositionsAdded` is Optional in
  `SessionCheckpointState` (back-compat) — missing → check is skipped
  rather than forced to mismatch.

  Wired into `ContentView.loadSessionFrom`'s post-restore path.

  ### Launch-time orphan sweep

  New `CheckpointPaths.cleanupOrphans()` runs from
  `DrewsChessMachineApp.init` after `SessionLogger.start`. Removes:

  - `Sessions/<name>.tmp/` directories — `saveSession`'s staging
    directory (matches the `.tmp` suffix appended to the target
    session dir name).
  - `Models/<name>.dcmmodel.tmp` files — `saveModel`'s staging file
    (matches the `.tmp` extension appended to the final `.dcmmodel`
    filename).

  Each removal is logged `[CLEANUP] Removed orphan <name>`; failures
  log `[CLEANUP-ERR]` and do not abort the sweep (a stuck orphan
  should not prevent the app from starting). Runs once at launch,
  before any save/load UI activates.

  ### Documentation

  - **NEW** `dcmmodel_file_format.md` — full byte-level spec for the
    `.dcmmodel` format, with expanded FNV-1a documentation
    (constants `0x811C9DC5` offset basis and `0x01000193` prime,
    algorithm pseudocode, Swift reference implementation, byte-order
    rationale, worked example, comparison vs. CRC32/xxHash/SHA-256),
    SHA-256 trailer spec, decode protocol, error taxonomy, explicit
    non-goals.
  - **NEW** `replay_buffer_file_format.md` — v3 (historical) + v4
    (current) sections. v4 section includes full layout, decode
    protocol ordering, write protocol, durability pipeline in
    session saves, cross-check semantics, launch-time orphan sweep,
    and full error taxonomy.
  - `CHANGELOG.md` entry at top (short form — points here for full
    detail).
  - `TODO_NEXT.md` §3 removed (was detailed there during planning;
    now done).

  ### Tests

  `DrewsChessMachineTests/ReplayBufferTests.swift` — 5 new tests, all
  green:

  - `testV3FileRejectedWithUnsupportedVersion` — synthesized v3
    header (no SHA trailer) rejects with `unsupportedVersion(3)`.
  - `testV4SHAMismatchRejected` — valid v4 file with one byte flipped
    at offset 56 rejects with `hashMismatch`.
  - `testV4SizeMismatchOnTruncation` — valid v4 file truncated by
    one byte rejects with `sizeMismatch`.
  - `testV4SizeMismatchOnTrailingGarbage` — valid v4 file with an
    extra byte appended rejects with `sizeMismatch`.
  - `testV4UpperBoundRejectedOnCapacity` — header with
    `capacity = Int64.max` rejects with `upperBoundExceeded(field:
    "capacity", ...)`, not an allocation crash.

  Existing tests (`testEmptyBufferWriteRead`,
  `testSinglePositionWriteRead`, `testV2FileRejectedWithUnsupportedVersion`,
  `testBadMagicRejected`, `testTruncatedHeaderRejected`) pass
  unchanged against v4 via the public API. Full test suite: 55/55
  green.

  ### Scope limits explicitly not taken

  - No per-record hashing (file-level SHA-256 is sufficient).
  - No compression (writes stay raw-float).
  - No cross-architecture or cross-version migration.
  - No `session.json` schema change (cross-check reads existing
    Optional fields).

  ### Parameter reference

  New constants (private statics on `ReplayBuffer`):
  - `fileVersion: UInt32 = 4` (was 3)
  - `trailerSize: Int = 32`
  - `maxReasonableCapacity: Int64 = 10_000_000`
  - `maxReasonableStoredCount: Int64 = 10_000_000`
  - `maxReasonableFloatsPerBoard: Int64 = 8_192`

  No existing hyperparameter (LR, batch size, clip, tau, etc.) was
  changed.

- **N-worker concurrent self-play in Play and Train.**

  **Superseded, 2026-04 onwards:** the per-worker-network topology
  described below was replaced by a single shared `BatchedMoveEvaluationSource`
  on the champion network — see the "Batched self-play evaluator"
  entry in the Completed section. The original design is preserved
  verbatim here (per the ROADMAP convention) for historical context
  and rationale. **Do not use this section as a description of
  current runtime behavior.** In particular:
    - `secondarySelfPlayNetworks` no longer exists (ContentView.swift
      has an in-code note to that effect where the field used to
      live).
    - On promotion, candidate weights are now copied into **both**
      the champion (`network`) and the trainer (`trainer.network`);
      there is no per-worker network to mirror.
    - The "topology is asymmetric — worker 0 reuses champion,
      workers 1..N-1 use secondaries" split is obsolete; all
      workers share the champion through the batcher.

  **Original design (as shipped, now historical):** Play and Train
  previously ran a single self-play worker, which at ~357 moves/sec against
  a 3,012 moves/sec training consumer meant every replay-buffer position
  was sampled ~8.4× on average before eviction — far above the 2–4×
  replay ratio common for off-policy RL, and the buffer also covered only
  ~625 games of play diversity. The fix is to spawn `N` concurrent
  self-play workers at session start, each with its own dedicated
  `ChessMPSNetwork` instance so no two concurrent `evaluate` calls share
  MPSGraph state. `ContentView.initialSelfPlayWorkerCount` (currently
  `6`) sets the default active count when a session begins;
  `ContentView.absoluteMaxSelfPlayWorkers` (currently `16`) is the hard
  ceiling — we pre-build that many inference networks and spawn that
  many worker tasks so the user can live-tune N inside
  `[1, absoluteMaxSelfPlayWorkers]` via a Stepper next to Run Arena
  without restarting the session. Topology is asymmetric: worker 0
  reuses the existing `network` (the champion, also the arena snapshot
  source), and workers `1..N-1` use new `secondarySelfPlayNetworks`
  mirrored from the champion at session start and at every arena
  promotion. Each worker owns its own `WorkerPauseGate`, so the
  arena-champion snapshot path (which only reads `network`) still
  pauses only worker 0, and only the promotion branch pauses every
  worker to `loadWeights` into every self-play network. Players
  (`MPSChessPlayer` white/black) are now allocated once per worker and
  reused across games — `ChessMachine.beginNewGame` already calls
  `onNewGame` on each, which resets per-game scratches while keeping
  backing storage alive. Under N=1 (checked live per game via
  `countBox.count == 1`, not captured at spawn), worker 0 wires
  `GameWatcher` as its `ChessMachine` delegate for the animated board;
  under N>1 no worker does, and a placeholder overlay "N = X concurrent
  games" hides the static board slot so the Candidate test picker
  remains usable. Aggregate self-play rates accumulate through the
  thread-safe `ParallelWorkerStatsBox`, which every worker calls
  identically via `recordCompletedGame(moves:durationMs:result:)` —
  no worker-0 specialness in the stats path. Setting N to 1
  reproduces the pre-change behavior (modulo the per-game player
  reuse cleanup). Memory cost is ~12 MB per additional inference
  network, trivial on unified memory.

  **Idle workers stay allocated deliberately.** When the user drops N
  from 6 to 3 via the Stepper, workers 3–5 finish their current game,
  then on their next iteration evaluate `countBox.count > workerIndex`,
  see false, and enter `WorkerPauseGate.markWaiting()` — a 50 ms
  sleep-poll loop that costs near-zero CPU. Their `ChessMPSNetwork`
  instances, `MPSChessPlayer` scratches, `WorkerPauseGate` state, and
  Swift tasks **all stay alive for the life of the session.** Only GPU
  cycles, CPU cycles for move generation / encoding / sampling, and
  replay-buffer lock contention are freed. Networks are only actually
  deallocated when Play and Train stops — and even then
  `secondarySelfPlayNetworks` persists in `@State` across sessions so
  re-entering Play and Train doesn't re-pay the MPSGraph build cost
  (~100 ms + per-network kernel JIT).

  This is a deliberate memory-vs-latency trade. The alternative design
  would cancel tasks and release networks on Stepper-down, then rebuild
  on Stepper-up — saving ~12 MB per idled worker but costing ~100–300 ms
  per + click for MPSGraph construction, first-run kernel JIT, and
  weight sync from the champion. Keeping everything pre-spawned means +
  and − clicks are effectively instant (≤50 ms, bounded by the idle
  poll interval) with no visible latency on the UI. At
  `absoluteMaxSelfPlayWorkers = 16` the steady-state memory cost is
  ~180 MB of idle network state plus ~74 MB of `MPSChessPlayer` scratch
  buffers, which is fine on Apple Silicon unified-memory systems. If
  that footprint ever becomes a problem on tighter hardware, the
  release-on-shrink design is the fallback; for now the latency win on
  live tuning is worth the static allocation.

- **Bundled architecture refresh (v2), 2026-04-19/20.** A coordinated
  redesign of the policy encoding, input planes, policy head, and
  value-baseline semantics, delivered as one bundle because the moves
  are coupled (the policy-encoding bijection and the fully-conv head
  both change the meaning of the 4864 logits; a staged rollout would
  have required throwaway migration code). Full design and phase
  breakdown live in `dcm_architecture_v2.md`. Summary of what shipped:

    - **AlphaZero-shape policy encoding (Phase 2).** The old flat 4096
      = 64×64 from-to encoding was replaced with 4864 = 76 × 64
      channel-square logits: 56 queen-style directions × distances, 8
      knight, 9 underpromotion (N/R/B × 3 directions, channels 64–72),
      3 queen-promotion (channel 73–75). Dedicated underpromotion
      channels fix the prior silent-collapse where all four promotion
      pieces mapped to the same index. The bijection lives in
      `PolicyEncoding.policyIndex(_:currentPlayer:)` — `ChessMove.policyIndex`
      was deleted so callers must think about the side-to-move frame flip.
    - **Fully-convolutional 1×1 policy head (Phase 4.3).** The old FC
      head (`128 × 64 → 4096`, ~528K params) was replaced with a 1×1
      conv `128 → 76` (~9.8K params, ~50× smaller). Translation
      equivariance is preserved end-to-end, which matches modern lc0
      practice and was the motivation cited in the ML review.
    - **20-plane input (Phase 3).** Planes 18 (≥1× before) and 19 (≥2×
      before) feed the network threefold-repetition context.
      Implementation reuses the engine's existing `positionCounts`
      table — no Zobrist machinery was needed (see `dcm_architecture_v2.md`
      Phase 1 for why we deviated from the originally-planned Zobrist
      path).
    - **Fresh-baseline value targets (Post-impl Addendum A).** The old
      `vBaseline` was a frozen self-play-time value; the trainer now
      runs an extra forward-only pass on its *current* network before
      each training step to recompute per-position `v(s)`, then
      overwrites the play-time staging before feeding the training
      graph. `MPSGraphGradientSemanticsTests` verified the
      placeholder-boundary stop-gradient semantics (MPSGraph has no
      `stop_gradient` op, and `with`-array exclusion does not prune
      backward-pass paths — the placeholder feed is the only correct
      way). Cost is ~33% extra forward FLOPs per training step;
      diagnostic `vBaselineDelta` now appears in `[STATS]`.
    - **`maxTensorElementCount` now computed from live arch
      (Phase 7.2).** `ModelCheckpointFile.maxTensorElementCount` is
      derived from `ChessNetwork.channels`, `inputPlanes`,
      `policyChannels`, and `seReductionRatio` (covers stem conv,
      residual conv, policy conv, SE FC). Acts as a defense-in-depth
      sanity cap on per-tensor element counts during `.dcmmodel`
      load — rejects implausible sizes before allocation, even when
      the SHA-256 trailer happens to match. Auto-tracks any future
      architecture change, no manual bump needed.

  The v2 bundle also introduced a `ReplayBuffer` format v3 (old
  buffers rejected cleanly on load), arch-hash bump on `.dcmmodel`
  (old checkpoints rejected with `.archMismatch`), and the first
  XCTest target. `CHANGELOG.md` has the commit-level breakdown;
  `dcm_architecture_v2.md` has the phase-by-phase design and a
  consolidated "Current state (as-built)" section that includes the
  post-impl follow-ups (sampling tau bump to 2.0, Candidate Test RAW-
  cell top-K display, entropy-alarm threshold) *and* the four post-v2
  commits listed below, plus a full parameter-defaults table.

  **Post-v2 follow-ups shipped during the first v2 run** (commits
  `9298273` → `068f805` → `7757418` → `cf1cc24`, all 2026-04-20):

    - **Advantage standardization + K dropped 50 → 5.** The policy-
      gradient weight is now `A_norm = (A − mean(A)) / sqrt(var(A) + 1e-6)`
      computed per batch inside the graph, autograd-safe because `A`
      depends only on the `z` and `vBaseline` placeholders. Removes the
      systematic bias when the value head has a global offset (e.g.
      `E[v] ≈ 0.45` once draws dominated self-play). With `A_norm`
      already at unit stdev, the pre-standardization `policyScaleK = 50`
      was pinning `gradClipMaxNorm` almost every step; dropped to `5.0`.
    - **Live-editable hyperparameters.** `weightDecayC`, `gradClipMaxNorm`,
      `policyScaleK`, `learnRate`, `entropyRegCoeff`, `drawPenalty`, and
      both sampling schedules are now fed to the training graph per step
      via scalar placeholders, so UI edits commit immediately without
      rebuilding the graph. Values persist in `@AppStorage` where
      applicable and are restored on session load. Every commit writes
      a `[PARAM] name: old -> new` line. `SessionCheckpointState` gained
      `policyScaleK: Float?` (Optional for back-compat) and the full
      `wd / clip / K / sp+ar tau` set on load.
    - **Diagnostics expansion.** New `TrainStepTiming` fields:
      `playedMoveProb`, `policyLogitAbsMax`, `policyHeadWeightNorm`,
      advantage distribution (`mean / std / min / max / fracPos / fracSmall`),
      plus `p05 / p50 / p95` from a rolling raw-advantage ring.
      Separate `legalMassSnapshot` probe (legal-mass + top1-legal) via
      `BoardEncoder.decodeSynthetic`, refreshed every 25 steps during
      bootstrap. `ParallelWorkerStatsBox` gained a 512-entry game-length
      ring (`p50 / p95 / avgLen`). `MPSChessPlayer` now counts
      "randomish" plies where post-temperature max probability is below
      `1.5 / N_legal` (policy-collapse signal independent of tau).
      `[STATS]` emitter restructured into bootstrap (per-step, first 500
      steps) + steady-state (60 s) phases.
    - **Full-sort top-K for catastrophic collapse.**
      `ChessRunner.extractTopMoves` now sorts the full 4864-cell policy
      vector rather than capping at `count × 4`, so a collapsed policy
      whose top cells are all off-board still produces `count` legal
      visualizations instead of an empty Candidate Test panel.
    - **MPSGraph reshape layout + sign-consistency tests.** New
      `MPSGraphReshapeLayoutTests` empirically verifies the policy head's
      `[B, 76, 8, 8] → [B, 4864]` reshape is NCHW row-major under
      `c·64 + r·8 + col` (plus end-to-end through `oneHot` +
      `softMaxCrossEntropy`). New `SignConsistencyTests` covers encoder
      symmetry, policy-index symmetry for mirrored moves, outcome-sign
      truth table, advantage-formula sign convention, geometric-decode
      round-trip, and bit-identical network output for bit-identical
      inputs.
    - **Advantage raw ring capped at 32K.** The `_advRawRing`
      in `TrainingLiveStatsBox` was originally sized `rollingWindow ×
      batchSize = 512 × 4096 ≈ 2 M Float`. `snapshot()` sorts the full
      filled portion for percentile extraction, and the 10 Hz UI
      heartbeat's `Task { @MainActor }` calls `snapshot()` via
      `queue.sync` — once the ring filled (~step 500) each sort cost
      ~150 ms on main, saturating the main actor. Because
      `fireCandidateProbeIfNeeded` is `@MainActor` and awaited after
      every training step, training throughput collapsed from
      ~2300 moves/sec to ~300 moves/sec and the UI went non-responsive.
      `advRawRingMaxCapacity = 32_768` drops sort cost from ~150 ms to
      ~1 ms while keeping percentile error below 0.5 % for a
      log-eyeballed diagnostic.

  **Default-parameter drift to record** (relative to this ROADMAP's
  earlier "N-worker concurrent self-play" entry):
  `initialSelfPlayWorkerCount` is now `24` (was `6`),
  `absoluteMaxSelfPlayWorkers` is now `64` (was `16`),
  `trainingBatchSize` is `4096`, `replayBufferCapacity` is `1_000_000`,
  `tournamentGames` is `200`, `tournamentPromoteThreshold` is `0.55`.
  The memory-vs-latency analysis in that older entry still applies —
  the numbers scaled up, the trade-off didn't change. The full current
  parameter-default table lives in `dcm_architecture_v2.md` under
  "Current parameter defaults".

  **Still open after v2.** `TODO_NEXT.md` #3 (ReplayBuffer durability —
  `fsync` + length invariant + reordered atomic save) remains
  unaddressed; adaptive LR schedule remains a design-only entry.
  One runtime regression observed but not yet root-caused: an
  `Unsupported MPS operation mps.placeholder` assertion during
  training after the live-hyperparameters change in `7757418`.
