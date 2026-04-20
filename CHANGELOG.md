# Changelog

All notable changes to Drew's Chess Machine are recorded here, newest first.
Each entry is timestamped with the date and time the change was committed
(CDT, −05:00, matching the recorded git author time). Entries corresponding
to a specific commit are tagged with the short hash; design/plan entries
that precede implementation are tagged `(DESIGN)`.

---

## 2026-04-20 (later still) — Session durability hardening (ReplayBuffer v4, fsync pipeline)

**Files:** `ReplayBuffer.swift`, `CheckpointManager.swift`, `DrewsChessMachineApp.swift`, `ContentView.swift`, `DrewsChessMachineTests/ReplayBufferTests.swift`, `replay_buffer_file_format.md`, `dcmmodel_file_format.md` (new), `ROADMAP.md`, `TODO_NEXT.md`.

Closes `TODO_NEXT.md` #3. Session saves now guarantee bit-identical restore-or-fail-with-precise-error. Full detail in `ROADMAP.md` → Completed.

- **ReplayBuffer format v3 → v4**: SHA-256 trailer (32 bytes), strict file-size equality check, upper-bound caps on header counters, `handle.synchronize()` before close. v1/v2/v3 files reject cleanly with `unsupportedVersion`. No migration path.
- **`CheckpointManager`**: new `fullSyncPath(_:)` helper using `fcntl(F_FULLFSYNC)` (fallback: `fsync`). Applied to each staged file, the tmp directory, and the parent `Sessions/` / `Models/` directories. `saveSession` now re-loads the replay buffer into a scratch `ReplayBuffer` post-write and compares counters against the `StateSnapshot` returned by `write(to:)` (captured atomically under the write lock — avoids a race where concurrent self-play appends advance the live ring between write and verify).
- **`ReplayBuffer.write(to:)` now returns `@discardableResult StateSnapshot`** — the snapshot that was actually serialized. Existing callers (tests, etc.) compile unchanged.
- **Load-time cross-check**: `verifyReplayBufferMatchesSession(buffer:state:)` compares `totalPositionsAdded` between restored buffer and `session.json`. Surfaced in `ContentView` session-resume path.
- **Launch-time orphan sweep**: `CheckpointPaths.cleanupOrphans()` removes `Sessions/*.tmp/` and `Models/*.dcmmodel.tmp` left behind by crashed-mid-save runs. Invoked from `DrewsChessMachineApp.init`. Logs `[CLEANUP]` / `[CLEANUP-ERR]`.
- **New `CheckpointManagerError` cases**: `fsyncFailed(URL, Error)`, `replayVerificationFailed(String)`, `sessionReplayMismatch(detail: String)`.
- **New `ReplayBuffer.PersistenceError` cases**: `hashMismatch`, `sizeMismatch(expected: Int64, got: Int64)`, `upperBoundExceeded(field: String, value: Int64, max: Int64)`.

### Parameter reference

New private statics on `ReplayBuffer`:
- `fileVersion: UInt32 = 4` (was 3)
- `trailerSize: Int = 32`
- `maxReasonableCapacity: Int64 = 10_000_000`
- `maxReasonableStoredCount: Int64 = 10_000_000`
- `maxReasonableFloatsPerBoard: Int64 = 8_192`

No existing hyperparameter (LR, batch size, clip, tau, draw penalty, entropy reg, policy scale K, etc.) was changed.

### Tests

5 new XCTests in `ReplayBufferTests.swift` covering v3 rejection, SHA-tamper rejection, size-mismatch on truncation and trailing garbage, upper-bound cap rejection on `capacity = Int64.max`. Full suite: 55/55 green.

### Docs

New `dcmmodel_file_format.md` (full byte spec, FNV-1a detail) and `replay_buffer_file_format.md` (v3 historical + v4 current) created. `ROADMAP.md` Completed section has the full writeup.

---

## 2026-04-20 (later) — Advantage standardization, live hyperparameters, stats expansion, session format

**Files:** `ChessTrainer.swift`, `ChessNetwork.swift`, `BatchedSelfPlayDriver.swift`, `MPSChessPlayer.swift`, `ContentView.swift`, `SessionCheckpointFile.swift`.

Follow-up to the ML review of the 80 %-draw training plateau. The review highlighted three root causes: (1) uncentered advantages biasing the policy gradient into the trunk in one direction, (2) gradient clip sitting 160× below the natural gNorm so the effective LR was driven by clip, not by `lr`, and (3) no visibility into the logit-scale / played-move-probability / advantage-distribution signals that would diagnose policy collapse early.

### Per-batch advantage standardization

Before this change the policy gradient used raw `A = z − vBaseline`. With a value head stuck at a global bias (e.g. `E[v] ≈ 0.45` under the current draw-heavy regime), raw advantages were systematically asymmetric — wins ≈ +0.55, draws ≈ −0.75 after `drawPenalty`, losses ≈ −1.45 — pushing the shared trunk in a biased direction regardless of position content.

`weightedCE = advantage_normalized * negLogProb`, where `advantage_normalized = (A − E[A]) / √(Var[A] + 1e-6)`. `E[A]` and `Var[A]` are computed over the current batch; ε = 1e-6 floors the denominator.

Correctness: MPSGraph has no `stopGradient`, but this is a non-issue because `advantage` is a pure function of the two placeholders `z` and `vBaseline`. The gradient of the total loss w.r.t. trainable variables flows through `negLogProb` only, with `advantage_normalized` acting as a batch-constant coefficient. The standardization adjusts the forward REINFORCE weight and never touches the autograd path.

Raw-advantage diagnostics (`advantageMean/Std/Min/Max/FracPos/FracSmall/Raw`) are unchanged — they still measure the unnormalized distribution so the baseline's fit is visible in the log.

### Policy scale K lowered from 50 to 5

`policyWeight × policyLoss` was amplifying the policy-gradient by 50×, which is what drove gNorm to ~800 and pinned the clip scale at `5/800 ≈ 0.6 %`. Combined with the normalized advantages (which already have unit scale), K = 5 is enough to keep both heads comparable without fighting the clip.

### Live-editable hyperparameters (LR already existed)

The following now flow through the training graph as per-step scalar placeholders (`graph.placeholder(shape: [1], …)`), fed every step via `MPSNDArray.writeBytes` into a pre-allocated feed:

- `weightDecayC` (default 1e-4)
- `gradClipMaxNorm` (default 5.0)
- `policyScaleK` (default 5.0)

UI: editable text fields on the Entropy Reg row (`clip` / `decay` / `K`). Persisted via `@AppStorage`. Trainer init takes the current values; edits take effect at the next SGD step without graph rebuild.

Sampling schedule (self-play + arena `startTau` / `floorTau` / `decayPerPly`) is also now live-editable via a new `SamplingScheduleBox` threaded through `BatchedSelfPlayDriver`. Driver reads the schedule at each game boundary and assigns `player.schedule` on the (reused) `MPSChessPlayer`. `MPSChessPlayer.schedule` changed from `let` to `var`; writes are guaranteed to happen between games by the slot loop structure, so no data race with mid-game `sampleMove` reads. Arena snapshots the schedule at tournament start for stability.

### New training-health diagnostics

All read via graph `targetTensors` (GPU-computed, zero extra round-trips beyond the readback) and surfaced on both the `[STATS]` log line and the in-app Training panel:

- **`policyHeadWeightNorm`** — L2 norm of the policy head's final 1×1 conv weights. Direct measure of the tensor whose growth drives logit-scale runaway. Requires exposing `ChessNetwork.policyHeadFinalWeights` (wrapped with `graph.read()` before the reshape to avoid an MPSGraph lowering edge case with variable-fed reshapes).
- **`policyLogitAbsMax`** — batch mean of `max_i |logits[i]|`. Pre-saturation early warning; grows before entropy collapses.
- **`playedMoveProb`** — batch mean of `softmax[movePlayed]`. Action-index sanity check: under a working loop it rises from `~1/policySize` toward 1.0 over training; a flat plateau despite pLoss moving = action-index mismatch.
- **`advantageMean` / `advantageStd` / `advantageMin` / `advantageMax` / `advantageFracPositive` / `advantageFracSmall`** — scalar reductions over the raw advantage tensor, plus p05/p50/p95 percentiles over a rolling ring of raw batch-concatenated advantages maintained in `TrainingLiveStatsBox`.
- **`legalMass` / `top1Legal`** — batch mean of softmax mass on legal moves, and fraction of positions where the full-policy argmax is a legal move. Computed via a separate forward-only probe (`ChessTrainer.legalMassSnapshot`) on `sampleSize` replay-buffer positions, refreshed every ~25 steps during the bootstrap STATS window and every 60 s thereafter.

The `[STATS]` emitter was restructured into a bootstrap phase (one line per new step until step ≥ `bootstrapStatsStepCount`) + steady-state phase (one line every 60 s).

### Session checkpoint format

Added `policyScaleK: Float?` to `SessionCheckpointState`. Optional for back-compat with session files written before the field became editable.

### Cleanup

Removed dead `policySoftmaxTensor` field (stored but never referenced as a target tensor or by the separate `legalMassSnapshot` path, which runs its own inference-graph pass).

### What this does NOT fix

The `Unsupported MPS operation mps.placeholder` runtime assertion observed during training is **not yet resolved**. The scalar placeholders for `weightDecay`/`clip`/`K` use the exact same construction pattern as the already-working `lr` and `entropyCoeff` placeholders; the `graph.read(...)` addition for the policy-weight-norm path is defensive but speculative. Root cause still under investigation.

---

## 2026-04-20 (later) — Fresh-baseline forward pass + tau=2.0 bump

**Files:** `ChessTrainer.swift`, `MPSChessPlayer.swift`, `ContentView.swift`.
New tests: `MPSGraphGradientSemanticsTests.swift`.

Two correctness/exploration fixes after deeper analysis of why the
trainer wasn't winning arenas. Both are runtime hyperparameter / data-
flow changes; no architecture or schema impact.

### Fresh-baseline forward pass (resolves ML Review #4)

The `vBaseline` values stored in the replay buffer at play time come
from the champion's value head — and since no promotion has happened,
the champion is still the random-init network. So every advantage
`(z - vBaseline)` was being computed against essentially-noise
baselines that varied by random seed. With our particular run's seed
landing vBaseline at ~-0.48, draws (z=-0.1) got advantage = +0.38 —
small POSITIVE, meaning the policy gradient was REINFORCING shuffle
moves. Combined with 84% drawn games in self-play, the trainer was
~2.7× more strongly pushed toward shuffles than toward winning moves,
locking in a "draw plateau" failure mode.

Fix: trainer now does a forward-only pass on its CURRENT network for
every training batch, before the actual training step runs. The fresh
v(s) values overwrite the staging `vBaselines` before they feed the
training graph. The `vBaseline` placeholder boundary already provides
stop-gradient semantics; only the source of values changed.

Required because empirical tests (`MPSGraphGradientSemanticsTests`)
confirmed:
- MPSGraph has NO `stop_gradient` op — only one autodiff method,
  `gradients(of:with:name:)`.
- The `with` parameter selects which gradients to RETURN; it does not
  prune backward-pass paths (verified with two distilled scenarios).

So the fresh-baseline-via-placeholder-feed pattern is the only
correct way to get the trainer's current v(s) into the policy
advantage. Cost: ~33% extra forward FLOPs per training step. Worth
it — eliminates seed-dependent training trajectories and removes the
shuffle-reinforcement bias.

`TrainStepTiming` gained `vBaselineDelta: Float?` (mean abs delta
between fresh and stale baseline values — divergence diagnostic) and
`freshBaselineMs: Double?`. `totalMs` now includes the fresh-baseline
phase so the replay-ratio controller throttles correctly.
`TrainingLiveStatsBox` gained a rolling `vBaselineDelta` window.
`[STATS]` line gained `vBaseDelta=0.XXXX`.

### Sampling tau bumped to 2.0 (both schedules)

Both `SamplingSchedule.selfPlay.startTau` and `.arena.startTau`
raised from 1.0 / 0.7 → **2.0**. Decay rates and floors unchanged:
- self-play: 2.0 → 0.4, decay 0.03/ply, floor at ply 54
- arena: 2.0 → 0.2, decay 0.04/ply, floor at ply 45

Tau=2.0 halves the legal-move logits before softmax, flattening the
early-game distribution. With the bootstrap-phase policy concentrated
on a small set of cells (entropy ~6.5 vs uniform 8.49), this pulls
mass back toward broader move selection. Goals: more decisive games
in self-play (currently ~16%), shorter average game length (currently
~342 plies), more candidate-vs-champion divergence in arenas.

### Misc UI cleanup

Removed duplicate `Draw pen:` row from the Training column display —
already shown in the editable hyperparameter section above.

---

## 2026-04-20 — Post-v2 polish: UI, diagnostics, segments, parameter logging, tests

**Files:** `ContentView.swift`, `ChessRunner.swift`, `ChessBoardView.swift`,
`ChessMove.swift`, `PolicyEncoding.swift`, `SessionCheckpointFile.swift`,
`ModelCheckpointFile.swift`, `TrainingChartGridView.swift`,
`AppCommandHub.swift`, `DrewsChessMachineApp.swift`,
`TensorCarouselView.swift`, `ReplayBuffer.swift`. New tests:
`PolicyEncodingTests.swift`, `BoardEncoderTests.swift`,
`RepetitionTrackingTests.swift`, `ReplayBufferTests.swift`. ROADMAP entry
for adaptive learn-rate schedule.

Follow-up pass after the v2 architecture refresh shipped — covers
runtime polish surfaced by the user's first training sessions, plus
the deferred test target and engine diagnostics work.

### UI / display

- **Cumulative status bar** at top of window: `Active training time`
  (HH:MM:SS), `Training steps`, `Positions trained` (compact format
  like `52.6M`), `Runs` (segment count). Computed across all completed
  Play-and-Train segments + the in-flight one. Hidden until the
  session has had any training. Includes a `Run Arena Now` button
  (right-aligned) when an arena isn't already running.
- **Training-segment tracking**: new `SessionCheckpointState.TrainingSegment`
  struct captures one Play-and-Train run's start/end times, training-
  step + position-counter snapshots, and the build/git context active
  at the time. Segments accumulate across save/load so cumulative
  wall-time excludes idle gaps. Save closes the in-flight segment AND
  reopens a fresh one if training continues — without the reopen,
  post-save training time would silently disappear from totals.
- **Title-bar contrast**: build-banner text bumped from
  `.caption + .tertiary` → `.callout + .secondary` (bigger AND
  higher-contrast); right-side ID + status text bumped from `.caption`
  → `.callout` (just bigger). Both legibility complaints from the
  user.
- **Alarm banner**: title now `Color.black` (was washing out as white
  on yellow); detail text now medium-weight + dark red `(0.55, 0, 0)`
  for legibility against the yellow background; "Silenced" label also
  black. New `Dismiss` button alongside `Silence` — Dismiss clears
  the banner AND resets the divergence streak counters so the alarm
  only re-raises on a fresh deterioration from a healthy baseline.
- **Training column** display: `vMean` and `vAbs` rows added under
  `Loss value:` (read from the trainer's rolling-window snapshot,
  same source as the `[STATS]` log line). Removed duplicate `Ent reg`,
  `Grad clip`, and `Weight dec` rows (those values are already shown
  in the editable hyperparameter section above).
- **Hourly rate** added next to per-second rates in the gen/trn rate
  display: `1m gen rate: 3500 pos/s   (12,600,000/hr)`. Comma grouping
  via `Int.formatted()`.
- **Chart label** "Non-negligible policy count" → "Above-uniform policy
  count". Underlying metric unchanged; the new label avoids the
  misleading implication that the X / 4864 ratio is a "fraction of
  useful cells" (most of the 4864 cells are physically-impossible
  moves anywhere, so they always sit below the 1/4864 baseline).

### Diagnostic & logging

- **Engine Diagnostics** menu item (Debug → Run Engine Diagnostics):
  one-shot battery of probes — PolicyEncoding round-trips, no-shared-
  index check, ChessGameEngine 3-fold detection, BoardEncoder shape,
  network forward-pass shape if a network exists. Output goes to
  the session log with `[DIAG]` prefix; designed to complete in well
  under a second for immediate pass/fail feedback.
- **Build banner** at session start now includes
  `arch_hash=0xXXXXXXXX inputPlanes=20 policySize=4864` so log
  forensics can immediately tell which architecture variant produced
  a given session.
- **Parameter-change logging**: every UI commit on Learn Rate, Entropy
  Reg, Draw Penalty, Self-Play Workers, Step Delay (manual), Replay
  Ratio Target, and Replay Ratio Auto-Adjust writes a `[PARAM] name:
  old -> new` line to the session log. No-op edits are suppressed.
- **Alarm banner activity** always logged: every raise (new or
  changed-severity) writes `[ALARM] <title>: <detail>`; clear writes
  `[ALARM] cleared: <title>`; silence and dismiss similarly.
- **Save errors** auto-log: every `setCheckpointStatus(isError: true)`
  call also writes `[CHECKPOINT-ERR] <message>` to the session log so
  the on-screen 12-second-auto-clear message is permanently
  recoverable from the log.
- **Policy entropy alarm threshold**: 7.0 → 7.2 → 5.0 (multiple steps
  as we calibrated against the v2 architecture's actual init entropy).
  The new policy head has no BN before its 1×1 conv, so init logits
  have larger σ than v1 — entropy starts ~6.5 instead of ~8.3. The
  5.0 threshold leaves ~1.5 nat margin below init for healthy
  training while still flagging genuine collapse.

### Candidate Test view

- Removed the misleading `(v+1)/2 → "X% win / Y% loss"` line. Without
  WDL output, that mapping was dishonest about what the scalar value
  meant. Now shows just the raw value scalar.
- Replaced the misleading `NonNegligible: X / 4864` metric with two
  legality-aware metrics: `Above uniform: X / N legal (threshold =
  1/N)` and `Legal mass sum: Y%`. Both compute over the legal-move
  subset for the current position.
- **Top-K display now shows raw policy cells, not legal-only.** My
  earlier refactor of `extractTopMoves` to iterate legal moves only
  lost the diagnostic for "is the network learning what's a valid
  move?" Restored: top-4 raw cells with `(illegal)` flag on cells
  that decode to illegal moves in this position. Board ghost arrows
  show the same set (illegal-but-on-board candidates draw normally;
  off-board cells skipped). Required:
  - New `PolicyEncoding.geometricDecode(channel:row:col:currentPlayer:)`
    method that returns a `ChessMove?` based on geometry only (off-
    board → nil), no legality filter.
  - `PolicyEncoding.decode(...)` refactored to call `geometricDecode`
    and add the legal-moves filter on top.
  - `MoveVisualization` gained `isLegal: Bool` field (defaults to
    true for legacy callers).
  - `ChessMove` now conforms to `Hashable` (synthesized) — used to
    build a fast `Set` for legality lookup in `extractTopMoves`.

### Tests (new XCTest target — DrewsChessMachineTests)

- **`PolicyEncodingTests`**: round-trips for starting position, after-
  1.e4, all 6 promotion files × 4 piece variants, knight-corner
  positions, castling positions. Bijection check across many
  positions (no two legal moves share an index). Off-board guard
  rejection. Out-of-range channel/row/col rejection. Castling
  encoded as queen-style E direction distance 2 (channel 15).
  Symmetric pawn pushes (white e2-e4 ↔ black e7-e5) hit the same
  channel.
- **`BoardEncoderTests`**: tensor length matches `inputPlanes × 64`.
  Starting position has correct piece counts per plane (8 pawns, 2
  knights, etc.). All four castling planes all-1 at start. EP plane
  exactly one cell after 1.e4. Halfmove plane saturates at /99 and
  doesn't overflow above 99. Repetition planes 18/19 follow always-
  fill semantics with proper saturation at count=2. Black-to-move
  perspective flip puts black king at encoder row 7.
- **`RepetitionTrackingTests`**: starting position has rep count 0.
  First move resets to 0 (novel position). Knight-shuffle
  Nf3/Nc6/Ng1/Nb8 × 2 produces correct count escalation (0, 1, 2)
  and triggers `drawByThreefoldRepetition`. Pawn move clears the
  history table.
- **`ReplayBufferTests`**: empty buffer round-trips. Single position
  round-trips (board floats, move index, outcome, vBaseline all
  exact). Synthetic v2 file (with the older 1152-float board stride)
  rejected with `unsupportedVersion(2)`. Bad magic and truncated
  header also reject cleanly.
- All 38 tests pass via `mcp__xcode-mcp-server__run_project_tests`.

### Misc

- ROADMAP entry added for adaptive learn-rate schedule, with five
  candidate trigger families (step decay, plateau detection,
  promotion-driven, cosine annealing, replay-ratio aware) and notes
  on how each interacts with our pre-MCTS bootstrap regime.
- Stale `1152` doc references in `ReplayBuffer.swift` cleaned up
  (now describes only the current v3 format).

---

## 2026-04-19 (DESIGN+IMPL) — Architecture v2: 76-channel policy head, SE blocks, 20-plane input

**Files:** `ChessNetwork.swift`, `BoardEncoder.swift`, `PolicyEncoding.swift` (new),
`ChessGameEngine.swift`, `ChessTrainer.swift`, `MPSChessPlayer.swift`,
`BatchedMoveEvaluationSource.swift`, `ChessRunner.swift`,
`GameDiversityTracker.swift`, `TrainingChartGridView.swift`,
`ContentView.swift`, `ChessMove.swift`, `ReplayBuffer.swift`,
`ModelCheckpointFile.swift`, `TensorCarouselView.swift`, `MoveEvaluationSource.swift`,
`ChessMPSNetwork.swift`. Plan: `dcm_architecture_v2.md`.

Bundled architectural refresh — three independent improvements landed
together to amortize one forced retrain. **All existing checkpoints
become incompatible** (arch hash bumps automatically; ReplayBuffer v2
files reject as `unsupportedVersion`). User explicitly accepted the
clean break.

- **Policy head: FC bottleneck → fully convolutional 1×1 conv 128→76.**
  Old head was `1×1 conv 128→2 → BN → ReLU → flatten → FC 128→4096`,
  collapsing all spatial structure through a 128-float bottleneck.
  New head is a single 1×1 conv emitting `[B, 76, 8, 8]` reshaped to
  `[B, 4864]` logits. 76 = 56 queen-style + 8 knight + 9 underpromotion
  + 3 queen-promotion (AlphaZero-shape with dedicated queen-promotion
  channels for symmetry). Move ↔ logit-index bijection lives in
  `PolicyEncoding.swift`. Translation equivariance preserved end-to-
  end. Underpromotion gets dedicated channels (fixes the prior
  policy-collision bug where Q/R/B/N for the same (from, to) all hit
  one logit). ~50× fewer head parameters (~9.8K vs ~528K).

- **SE blocks added to every residual block.** Squeeze-and-Excitation
  module sits between BN2 and the skip-add: global avg pool → FC 128→32
  + ReLU → FC 32→128 + sigmoid → channel-wise scale. Provides per-
  position dynamic channel reweighting. Matches modern lc0 design;
  reduction ratio 4. ~67K added params.

- **Input planes 18 → 20: threefold-repetition signals.** Plane 18 is
  1.0 if current position has occurred ≥1 time before this game; plane
  19 is 1.0 if ≥2× before. Computed via the existing
  `ChessGameEngine.positionCounts` table (no new Zobrist machinery —
  the existing `PositionKey`-based tracking already exists for
  threefold-detection). Saturated at 2 (only the rule-relevant
  thresholds matter). `GameState` gains a `repetitionCount` field with
  default 0 and a `withRepetitionCount(_:)` helper; the engine layers
  the count onto each new state after `applyMove`.

Side effects rolled in:
- `ChessMove.policyIndex` deleted (forces every callsite to use
  `PolicyEncoding.policyIndex(_:currentPlayer:)` which knows about the
  encoder-frame perspective flip).
- `ReplayBuffer` format bump to v3; v1/v2 cleanly rejected.
- `ModelCheckpointFile.maxTensorElementCount` made computed (auto-
  tracks arch changes instead of needing a manual bump).
- `MPSChessPlayer.networkPolicyIndex` removed (replaced by
  `PolicyEncoding.policyIndex` everywhere).
- `ChessRunner.extractTopMoves` rewritten to iterate legal moves
  through `PolicyEncoding.decode` instead of decoding the flat policy
  vector geometrically.
- `policyEntropyAlarmThreshold` 7.0 → 7.2 (preserves the same ~1.3-nat
  margin below the new uniform-init entropy `log(4864) ≈ 8.49`).
- `TensorCarouselView` plane labels extended for the two new planes.
- `TrainingChartGridView` non-negligible-count Y-axis derived from
  `policySize` instead of hardcoded 4096.
- BatchedMoveEvaluationSource's hardcoded `policySize = 4096` /
  `boardFloats = 18*8*8` constants now derived from
  `ChessNetwork.policySize` / `BoardEncoder.tensorLength`.

Total network parameter count: ~2.92M → ~2.4M (-17.7%). Forward FLOPs
essentially unchanged (+0.15%). Replay buffer storage per slot grows
~11% (board floats: 1152 → 1280).

---

## 2026-04-18 23:30 CDT — Bootstrap hyperparameter retune: arena tau, draw penalty, LR

**Files:** `MPSChessPlayer.swift`, `ChessTrainer.swift`, `ContentView.swift`,
`sampling-parameters.md`.

Three defaults changed together as a retuning pass for the current
REINFORCE bootstrap phase:

- **Arena startTau: 1.0 → 0.7** (`SamplingSchedule.arena`). With
  `decayPerPly=0.04` and `floorTau=0.2` held constant, the floor is
  now reached at ply 13 instead of ply 20. Arena opening play now
  sits closer to each network's actual preferences; enough opening
  diversity still comes from color-alternating pairings plus the
  residual 0.7 tau to keep the 200-game tournament from collapsing
  into a handful of deterministic lines. Self-play schedule
  (`startTau=1.0`, decay 0.03, floor 0.4) is unchanged — its higher
  tau remains the right setting for replay-buffer coverage.
  `sampling-parameters.md` updated (table row + rationale bullet).

- **Draw penalty default: 0.0 → 0.1** (`drawPenaltyDefault` in
  `ContentView.swift`, matching init default in `ChessTrainer`).
  The 2026-04-18 22:46 commit introduced the knob with a no-op
  default; this commit turns it on by default so fresh sessions
  and new installs start with the REINFORCE draw-stasis break-out
  already active. Existing sessions that persisted a value via
  `@AppStorage("drawPenalty")` keep their previous setting.

- **Trainer learning rate default: 1e-4 → 5e-5**
  (`trainerLearningRateDefault` + `ChessTrainer.init` default).
  Halving the LR as a conservative default alongside the draw
  penalty — the penalty introduces a new gradient signal on
  ~80 %+ of positions (since draws dominate the replay buffer),
  and a gentler step size reduces the risk of the policy
  overshooting into a degenerate low-entropy attractor before
  the value head catches up and cancels most of the signal.

No graph rebuilds required — all three values are live-tunable
through the existing UI fields; the defaults just seed new
sessions and reset the baseline for anyone without a persisted
override.

---

## 2026-04-18 22:46 CDT — Configurable draw penalty for REINFORCE bootstrap

**Files:** `ChessTrainer.swift`, `ContentView.swift`,
`SessionCheckpointFile.swift`.

Adds a single tunable, `drawPenalty` (Float, default 0.0), that
rewrites drawn-game `z` values from `0.0` to `-drawPenalty` before
they reach the graph. Motivation: during the current bootstrap
phase, 82 %+ of self-play games end in threefold-repetition draws,
each contributing `z=0` to the REINFORCE policy loss — i.e. no
gradient. With the penalty set to a small positive value (e.g.
`0.1`), drawn positions contribute a mild negative signal so the
policy has a reason to avoid shuffling sequences. The value is
entered as a positive magnitude; the code negates it internally.

Mechanically: CPU-side transform in `ChessTrainer.trainStep`
between `replayBuffer.sample()` and `buildFeeds()`. Mutates the
private `replayBatchZs` staging buffer in place; the next sample
call fully overwrites it. The graph path is unchanged —
`advantage = z − v.detach()` just sees rewritten z values. Since
v(s) eventually learns to predict `-drawPenalty` for draw-prone
positions, the signal is self-limiting: strong while v is
uninformed, fading as v converges.

Applies uniformly to all four draw types (stalemate, 50-move,
threefold, insufficient material — all of which set `z = 0.0`
exactly in `MPSChessPlayer.onGameEnded`). Per-draw-type
distinction would require per-position draw-reason tracking in
the replay buffer; deferred.

**UI and plumbing:**
- `@AppStorage("drawPenalty")` persists across launches.
- New `Draw Penalty` `TextField` underneath `Entropy Reg` in the
  training controls, formatted `%.3f`, rejects negatives, hint
  "(draws → z = −penalty; 0 disables)". Live-tunable: the trainer
  reads the property on every batch, no graph rebuild needed.
- `ensureTrainer()` and the Play-and-Train start path propagate
  the value to the trainer.
- Session-checkpoint audit field `drawPenalty: Float?` added to
  `SessionCheckpointState`, Optional for back-compat. Resumes
  restore the value from the session file if present.
- `[STATS]` log line gained a `drawPen=X.XXX` token alongside
  `clip=... decay=... ent=...`, and the live stats panel shows
  `Draw pen: X.XXX` next to the other reg hyperparameters.

Default 0.0 means existing training is unaffected until the user
sets a value.

---

## 2026-04-18 22:13 CDT — Design doc: honest provenance of K=50 and lever analysis `(DESIGN)`

**File:** `chess-engine-design.md`

Revision note appended under the existing item 6 of the
"Stability Enhancements" section. Three changes, no code touched:

1. **K=50 provenance corrected.** The original item 6 framed
   K=50 as "working as intended," which overstated the principle
   behind the specific value. K=50 was empirically chosen during
   stability tuning after the 2026-04-15 collapse and the
   2026-04-16 stasis, and kept because it produced no
   catastrophic instability — not because 50 (vs 10, 20, 100) was
   derived from any policy-vs-value ratio argument. The doc now
   says so.

2. **Lever analysis under chronic clipping.** After extended
   training on build 156, `gNorm` sits at ~28–29 against
   `clip=5`, giving a ~0.17 clip scale on every step. The doc
   now records which levers move what in this regime:
   - K sets the policy-vs-value *ratio* (preserved post-clip)
     but not per-step magnitude.
   - `LR · clip_value` sets per-step magnitude when clipping is
     active. K does not enter this product.
   - Raising K pushes gNorm up without moving weights further.
   - Lowering K reduces gNorm; once gNorm falls below
     `clip_value` the effective step becomes `LR · raw_gNorm`,
     which is smaller than `LR · clip`.
   - LR scales effective step uniformly while staying bounded
     by `LR · clip_value` per step. Safer than raising clip.

3. **Revisit criterion rewritten.** The original
   "revisit after MCTS visit-count targets" condition was
   vacuous given MCTS is now an explicit non-goal (per
   `CLAUDE.md`). New criterion: if the system can be unfrozen
   (raise LR) without triggering the 2026-04-15 failure mode,
   re-examine whether K still needs the full 50× amplification.

No implementation in this commit — the operational step that
follows (raising `learningRate` from `1e-4` toward `3e-4`
through the live-fed UI field) is a runtime tuning change, not
a source change.

---

## 2026-04-18 21:32 CDT — Give the claude subprocess a usable PATH (`7b551d7`)

**File:** `LogAnalysisWindow.swift`

macOS apps launched via LaunchServices inherit an almost-empty
`PATH` (typically `/usr/bin:/bin`). The Node-based `claude` CLI
spawns `node`, `git`, and other helpers via `PATH` lookup, which
silently fails under that minimal env and makes `-p` return a
non-zero exit before producing any output.

The subprocess now starts from the inherited environment (so
`HOME` and shell-specific vars stay intact) and prepends the
common install dirs where `claude` and its deps live —
`~/.local/bin`, Homebrew paths, `/usr/local/bin`, and the
standard system dirs.

---

## 2026-04-18 21:24 CDT — Fix three correctness issues in the Log Analysis window (`2bf0c0d`)

**File:** `LogAnalysisWindow.swift`

All three surfaced during post-commit recheck of the 21:09 Analyze
Log feature.

1. **`NSInvalidArgumentException` on cancel-before-launch.** The
   view model published `activeProcess = claudeProc` to the main
   actor BEFORE calling `claudeProc.run()`, so a window close
   during the ~microsecond window between publish-hop and launch
   would call `terminate()` on an un-launched `Process` — which
   raises. Publish now lands after both `run()` calls succeed.
2. **Hung subprocess when claude fails to launch.** If
   `catProc.run()` succeeded but `claudeProc.run()` threw, nothing
   cleaned up cat — it would write into a pipe with no reader,
   fill the 64 KB buffer, then block forever on its next write,
   and the surrounding `Task` would never return. The
   claude-failure catch now terminates cat and waits for it before
   surfacing the error.
3. **`try?` on `AttributedString(markdown:)`** (violates the
   project-wide rule against `try?` without explicit
   justification). Now uses `do`/`catch` with a plain-text
   fallback so parser rejections render the raw string rather
   than dropping it.

Also dropped a now-unreachable outer `do`/`catch` in `runAnalysis`
that the compiler warned about after the inner cleanup paths
started handling every throw site themselves.

---

## 2026-04-18 21:09 CDT — Debug > Analyze Log: pipe session log through claude CLI (`005bf7b`)

**Files:** `LogAnalysisWindow.swift` (new), `DrewsChessMachineApp.swift`.

New menu item opens a split window: raw session log in a
monospaced top pane, `claude -p` response as markdown-rendered
text in the bottom pane. Launcher checks `~/.local/bin/claude`
exists and is executable first, then reads the current
`SessionLogger` file, then runs

```
/bin/cat <log> | ~/.local/bin/claude -p "Analyze this log. ..."
```

on a GCD background queue so the main actor and Swift's
cooperative executor stay free during the multi-second claude
invocation. Window close terminates the live subprocess so
closing mid-analysis doesn't leak a long-running claude process.

`LogAnalysisWindow.swift` contains a minimal block-level markdown
renderer (headings, bullet/numbered lists, fenced code blocks,
inline bold/italic/code/links via `AttributedString`) — rich
enough for typical `claude -p` output without pulling in a
third-party package.

---

## 2026-04-18 20:58 CDT — Debug > Open Session Log menu item (`5c85160`)

**File:** `DrewsChessMachineApp.swift`

Opens the active `SessionLogger` file in the default text editor
via `NSWorkspace`. Disabled when the logger hasn't started
(pre-launch) or `start()` failed to open a file. Faster than
navigating to `~/Library/Logs/DrewsChessMachine` when debugging a
running session.

---

## 2026-04-18 20:55 CDT — Fix chart x-axis mismatch on resumed sessions (`57467d5`)

**File:** `ContentView.swift`

On session resume the training chart tiles rendered blank because
their samples landed thousands of seconds past the visible window.
`refreshTrainingChartIfNeeded` anchored `elapsedSec` at
`currentSessionStart` — intentionally back-dated by the loaded
session's `elapsedTrainingSec` so persisted session state keeps
accumulating elapsed time across save/resume cycles. But
`refreshProgressRateIfNeeded` uses `parallelStats.sessionStart`,
which is a fresh `Date()` at Play-and-Train start, so its samples
start at `elapsedSec = 0`.

The two chart groups share one `scrollX` binding driven by the
progress-rate coordinate space. With the training-chart samples
offset ~hours forward of the progress-rate samples, every training
tile had its data parked outside the visible window.

**Fix:** use `parallelStats.sessionStart` (same anchor the
progress rate already uses) for `refreshTrainingChartIfNeeded`
and the two arena-activity elapsed-time calculations.
`currentSessionStart` stays back-dated — only
`buildCurrentSessionState` reads it now, which is exactly the
persistence path that needs the cumulative wall clock.

---

## 2026-04-18 19:56 CDT — Capture timestamps at call site, not inside `queue.async` closure (`3c05b10`)

**File:** `ContentView.swift`

Recheck of the `NSLock` → `DispatchQueue` conversion caught four
spots where `Date()` / `CFAbsoluteTimeGetCurrent()` was evaluated
inside the async closure instead of at the moment the caller
invoked the method. When the serial queue has any backlog the
timestamp skews forward, which matters most for
`ParallelWorkerStatsBox.recordCompletedGame` — that `Date()` feeds
the rolling-window rate stats the UI displays, so "when did this
game end" has to be the actual game-end instant, not when the
stats queue got around to the write.

Affected methods:
- `ParallelWorkerStatsBox.markWorkersStarted`
- `ParallelWorkerStatsBox.recordCompletedGame`
- `GameWatcher.markPlaying` / `ChessMachine` delegate methods
  (threaded through `setPlayingOnQueue(_:now:)`)
- `ArenaTriggerBox.recordArenaCompleted`

---

## 2026-04-18 19:51 CDT — Replace every NSLock with a private serial DispatchQueue (`d2e3d31`)

**Files:** `SessionLogger.swift`, `ReplayBuffer.swift`,
`ReplayRatioController.swift`, `GameDiversityTracker.swift`,
`ChessTrainer.swift`, `ContentView.swift`.

Converts all 15 lock-protected state holders in the app to serial
`DispatchQueue`-backed synchronization. Writers dispatch
asynchronously where safe (workers never wait on UI pollers and
vice versa); readers use `queue.sync` so the FIFO ordering of the
serial queue still gives callers a consistent atomic view.
`ReplayBuffer.append` / `sample` / `write` / `restore` stay
`queue.sync` because the caller owns input or output pointers
that must be fully processed before return.
`WorkerPauseGate.markWaiting` / `markRunning` / `resume` stay
`queue.sync` because the coordinator's spin-wait depends on the
ack being visible the instant the worker returns.

Classes converted: `SessionLogger` (log is now fire-and-forget,
no more disk-fsync blocking on hot training paths),
`ReplayRatioController`, `GameDiversityTracker`, `ReplayBuffer`,
`TrainingLiveStatsBox`, `CancelBox`, `GameWatcher`,
`TournamentLiveBox`, `WorkerPauseGate`, `WorkerCountBox`,
`TrainingStepDelayBox`, `ArenaTriggerBox`, `ArenaOverrideBox`,
`ArenaActiveFlag`, `ParallelWorkerStatsBox`.

Former `*_locked` helpers were renamed to `*OnQueue` to reflect
the new precondition (must already be executing on the owning
queue).

---

## 2026-04-18 18:10 CDT — Move action buttons into File / Train / Debug menu bar menus (`0958401`)

**Files:** `AppCommandHub.swift` (new), `DrewsChessMachineApp.swift`,
`ContentView.swift`, `TrainingChartGridView.swift`.

Introduces `AppCommandHub` (an `@Observable @MainActor` class
owned by `DrewsChessMachineApp`) as a bridge between the top-level
`.commands { ... }` menu DSL and `ContentView`'s state + action
functions. `ContentView` wires each action closure into the hub
on `.onAppear` and keeps a small set of mirrored state flags in
sync via `.onChange` handlers so menu items can `.disabled(...)`
correctly.

**Menu layout** (order: File Edit View Train Debug Window Help —
SwiftUI places `CommandMenu` entries before Window, so Debug lands
adjacent to Train rather than between Window and Help):

- **File** (via `CommandGroup` after `.newItem`): Save Session,
  Save Champion, Load Session…, Load Model…, Open Data Folder in
  Finder (formerly "Reveal Saves" inline).
- **Train:** Build Network, Play and Train / Continue Training,
  Stop (⎋), Run Arena, Abort Arena, Promote Trainee.
- **Debug:** Run Forward Pass (↩), Play Game, Play Continuous,
  Train Once, Train Continuous, Sweep Batch Sizes.

The two inline button rows in `ContentView` collapse to a single
status row carrying the `ProgressView` + busy label + checkpoint
status message. The row is kept as a stable parent so the
`.fileImporter` modifiers (driven by the menu Load items) still
have a view to attach to. Silence (alarm banner) and the
board-overlay chevrons stay inline as requested.

**Chart grid renames and formatting:**
- Loss total → "Loss (pLoss + vLoss)"
- Loss policy → "pLoss (policy loss)"
- Loss value → "vLoss (value loss)"
- Grad norm → "gNorm (gradient L2 norm)"
- Progress rate → "Progress rate (self play + train)"
- Diversity histogram → "Longest move prefix"
- Non-negligible policy count header adds `(P%)` after `N / 4096`
  to mirror the policy-entropy tile.
- App memory → "App memory (RAM)"; GPU RAM → "GPU memory (RAM)".
  Both memory tiles switch to a new `memoryChart` helper that
  formats the header as `used GB / total GB (pct%)` against
  `ProcessInfo.physicalMemory`. The unified-memory total is
  plumbed through from `ContentView.memoryStatsSnap.gpuTotalBytes`
  as new `appMemoryTotalGB` / `gpuMemoryTotalGB` view params.

---

## 2026-04-18 17:39 CDT — Entropy regularization: build differentiable entropy in trainer (`0d70a51`)

**Files:** `ChessTrainer.swift`, `ContentView.swift`, `ModelID.swift`,
`SessionCheckpointFile.swift`, `TrainingChartGridView.swift`,
`DrewsChessMachine.xcscheme`.

The entropy-regularization feature feeds `policyEntropy` into
`totalLoss` as `-coeff * H(p)`. The diagnostic path it reused was
built from a manual max-subtracted log-softmax, which MPSGraph
cannot autodiff: `reductionMaximum` has no registered gradient,
and `graph.gradients(of: totalLoss, ...)` hit

```
MPSGraphOperation.mm:217: 'Op gradient not implemented...'
```

on every Play-and-Train start.

**Fix:** rebuild the entropy path inside
`ChessTrainer.buildTrainingOps` from `graph.softMax(with:axis:)`
(fused, has autograd) plus `log(softmax + 1e-10)`. Mathematically
equivalent to the old diagnostic up to an ~1e-7 epsilon bias;
numerically stable because `softMax` handles the max-subtraction
internally; fully differentiable end-to-end.

Surrounding feature wiring for the entropy coefficient (field on
`ChessTrainer`, placeholder + ND-array feed, UI text field in
`ContentView`, `@AppStorage` persistence, `SessionCheckpointFile`
round-trip, `[STATS]` log line) lands in the same commit since
the fix only makes sense alongside it.

Also in this commit:
- `ModelIDMinter.mintTrainerGeneration(from:)` for forking a
  lineage root into generation-suffixed trainer IDs.
- Rolling grad-norm mini-chart in `TrainingChartGridView`.
- `TrainingLiveStatsBox.resetRollingWindows()` so post-promotion
  charts start from the new regime.
- Scheme: drop `enableGPUValidationMode` from the Run action.

---

## 2026-04-18 15:58 CDT — Move MPSGraph work off cooperative executor (`1f67d5e`)

**Files:** `BatchedMoveEvaluationSource.swift`,
`CheckpointManager.swift`, `ChessMPSNetwork.swift`,
`ChessNetwork.swift`, `ChessRunner.swift`, `ChessTrainer.swift`,
`ContentView.swift`, `MoveEvaluationSource.swift`,
`ReplayBuffer.swift`.

Swift's cooperative executor requires every task to make progress;
a long synchronous `MPSGraph.run` call parked on a concurrency
thread starves every other task sharing that thread. Every
GPU-dispatch path is now wrapped in `withCheckedContinuation` +
a dedicated `DispatchQueue` so the concurrency thread returns
immediately and the MPSGraph call runs on a queue worker, with
`continuation.resume` bouncing the async caller back onto the
cooperative executor once the graph finishes.

Touches every GPU-adjacent call site: batched and direct move
evaluation, training step, checkpoint save/load, and replay-buffer
persistence. `ReplayBuffer` loses a large chunk of its
lock-centric plumbing because the restore path no longer needs to
coordinate with a concurrent task blocking on GPU work.

---

## 2026-04-17 23:21 CDT — Diversity histogram chart

**Files:** `GameDiversityTracker.swift`, `TrainingChartGridView.swift`,
`ContentView.swift`.

Adds a 6-bucket histogram of divergence plies across the 200-game
rolling window to surface "are we producing near-identical games?"
as a direct visual in the chart grid. The existing
`avgDivergencePly` metric buries the signal in a mean (stuck near 2.0
in steady state even when some games share long opening lines because
it averages 200 frozen per-game values); the histogram exposes the
tail directly.

**Buckets** (`GameDiversityTracker.histogramBounds`):
`[0-2]`, `[3-5]`, `[6-10]`, `[11-20]`, `[21-40]`, `[41+]`. First four
are the healthy-diversity range; last two are the collapse warning
zone. Bucket colors ramp green → mint → yellow → orange → red →
dark-red for visual severity.

**How it surfaces:**
- `GameDiversityTracker.Snapshot.divergenceHistogram: [Int]` (6
  entries, sums to `gamesInWindow`). Computed inside the single
  existing O(stored) pass in `snapshot()` — one extra bucket-assign
  per game, no extra lock acquisitions.
- `TrainingChartGridView` gains a `diversityHistogram:
  [DiversityHistogramBar]` input and a new `diversityHistogramChart`
  tile placed next to "Loss value" in row 3 of the grid.
- `ContentView.currentDiversityHistogramBars: @State` mirrored from
  the tracker on the UI heartbeat. Diff-checked before pushing into
  state so stable readings don't invalidate the chart every tick.

**Interpretation at steady-state diverse self-play:**
Most games sit in `[0-2]` (they branch within the opening); a handful
in `[3-5]`. If `[11-20]` or beyond start filling in, the policy is
locking onto deeper shared lines. If `[41+]` has non-zero count, deep
middlegame/endgame play is being replayed — the user's "thousands of
near-identical games" scenario.

---

## 2026-04-17 23:00 CDT — Policy-entropy alarm + avg-game-length in [STATS]

**File:** `ContentView.swift`

- **Policy-entropy alarm.** New constant
  `policyEntropyAlarmThreshold: Double = 7.0` (nats). The periodic
  `[STATS]` ticker, immediately after emitting its line, checks
  `trainingSnap.rollingPolicyEntropy` and emits
  `[ALARM] policy entropy X.XXXX < 7.00 — policy may be collapsing
  (steps=N)` whenever the rolling mean is below threshold. Random
  init sits at `log(4096) ≈ 8.318`; the `7.0` floor corresponds to
  roughly `exp(7.0) ≈ 1100` effective equiprobable moves, which
  leaves plenty of room for healthy sharpening while catching true
  collapse. Fires on every ticker wake-up below threshold (so the
  `[ALARM]` cadence matches the ramp-up + 15-min stats interval
  rather than spamming on every training step).
- **Average game length in `[STATS]`.**
  `avgLen=<lifetime> rollingAvgLen=<10-min window>` added to every
  periodic `[STATS]` line. Data already existed in
  `ParallelWorkerStatsBox` — `selfPlayPositions / selfPlayGames` for
  lifetime and `recentMoves / recentGames` for the rolling window —
  we just weren't surfacing the derived ratio.

Neither change affects any training behavior; both are pure
observability. Already-audited items that the user asked about but
were already shipped:

- Gradual temperature decay: `8ca529b` (21:24 CDT). Linear-decay
  `SamplingSchedule` replacing the two-phase schedule.
- L2 regularization: `458c321` (22:02 CDT),
  `ChessTrainer.weightDecayC = 1e-4` (decoupled, applied to every
  trainable variable).
- Duplicate / near-duplicate game detection: `8ca529b` (21:24 CDT).
  `GameDiversityTracker` tracks full move-sequence FNV-1a hashes +
  per-game divergence ply (longest shared prefix with any stored
  game) over a rolling 200-game window. Surfaced in `[STATS]` as
  `diversity=unique=X/Y(pct%) diverge=Z.Z` and in each `[ARENA]`
  result's trailing `div` line.

---

## 2026-04-17 22:46 CDT — Advantage baseline: store v at play time, feed as placeholder

**Files:** `ReplayBuffer.swift`, `ChessTrainer.swift`, `MPSChessPlayer.swift`

Completes plan item #3 from the 17:23 CDT design entry. MPSGraph has
no `stopGradient` / `detach` op (confirmed by the 22:35 CDT
`[EXP-DETACH]` experiment — `variableFromTensor` + `read` does *not*
block autodiff), so we get detach semantics by feeding the baseline
through a placeholder. Since every feed is a leaf in the gradient
graph, the policy loss can't walk backward from `(z − vBaseline)`
into the value head.

**How it works:**
- `MPSChessPlayer` captures the scalar `v` returned by the same
  `network.evaluate(board:)` call that already runs to pick each move
  (zero extra forward passes). Stored per-ply in `gameValueScalars`
  and bulk-flushed into the replay buffer at game end alongside the
  boards, policy indices, and outcome.
- `ReplayBuffer` gains a fourth ring-storage array
  `vBaselineStorage: [capacity]` of `Float`. `append(...)` takes a
  `vBaselines:` pointer; `sample(count:)` returns them via the new
  `TrainingBatch.vBaselines` field.
- `ChessTrainer.buildTrainingOps` adds a new `vBaseline` placeholder
  of shape `[-1, 1]` and builds `advantage = z − vBaseline`; the
  policy loss is now `mean(advantage · negLogProb)`. Everything else
  (value loss, K=50 scaling, weight decay, gradient clipping) stays
  intact.
- `BatchFeeds` cache grows to include the vBaseline ND array + tensor
  data wrapper. `buildFeeds(...)` writes the new column on every
  step. `trainStep(batchSize:)` random-data path feeds all-zero
  vBaselines, which degrades the advantage formulation back to
  `z · negLogProb` — keeps the sweep's numerical results comparable
  to prior runs.

**Replay-buffer file format:** `DCMRPBUF` header bumped from v1 → v2
with the addition of a fourth column after outcomes. Reader accepts
both versions: v1 files (saved before this commit) are restored with
`vBaseline = 0` for every slot, which degrades gracefully to
`z · negLogProb` until those positions age out of the ring and are
replaced by fresh v2 self-play data. New writes always produce v2.

**Trade-offs (documented earlier in-conversation):**
- Baseline staleness — the stored `v` is from the self-play inference
  network (= champion, or a secondary mirrored from champion) at the
  time the position was played, not the trainer's current `v`. Any
  state-dependent baseline gives unbiased gradients, so this produces
  a weaker variance reduction than an "ideal" current-step baseline
  but can't make the update worse.
- Effective gradient magnitude drops because `|z − v|` < `|z|` once
  the value head starts tracking outcomes. May need to revisit K=50
  or LR if `gNorm` collapses post-baseline. Watch `gNorm` in `[STATS]`.
- Warm-up dead zone: at the start of training, `v ≈ 0`, so
  `(z − v) ≈ z` — the baseline does nothing until the value head has
  learned something. Expected.

**Observed effect:** TBD — next Play-and-Train session will populate
v2 replay buffer and the advantage formulation will take effect once
the value head starts producing meaningful predictions.

---

## 2026-04-17 22:35 CDT — Gradient-stop experiment: `variableFromTensor` does NOT detach

Added a one-shot launch-time experiment in `ExperimentStopGradient.swift`
to answer the question "does `MPSGraph.variableFromTensor(_:name:)` +
`read(_:name:)` act as a gradient stop?" Test graph:
`w = variable(3.0)`, `x = 2*w`, `L_direct = x*x`,
`xVar = variableFromTensor(x)`, `xRead = read(xVar)`,
`L_via_var = xRead*xRead`.

Expected if detach works: `grad_via_var[w] = 0`.
Observed: `grad_via_var[w] = 24.0` — identical to `grad_direct[w]`.
Forward values agree (36.0 on both paths), so the computation ran
correctly; MPSGraph autodiff simply walks transparently through the
variable resource. Full log lines:

```
[EXP-DETACH] L_direct   = 36.0 (expected 36.0)
[EXP-DETACH] L_via_var  = 36.0 (expected 36.0 — forward should match)
[EXP-DETACH] grad_direct[w]  = 24.0 (expected 24.0)
[EXP-DETACH] grad_via_var[w] = 24.0 (0 ⇒ DETACH works; 24 ⇒ does NOT detach)
```

**Decision:** Advantage baseline (plan item #3) will be implemented
via the **store-v-at-play-time** replay-buffer schema change rather
than a two-run training step or a `variableFromTensor` detach. The
self-play inference already computes `v(position)` to pick moves; we
capture it into a new per-position field in `ReplayBuffer`, feed it as
`vBaseline` alongside `z` at train time, and use
`(z − vBaseline) * −log p(a*)` as the policy loss. Zero runtime cost,
mild baseline staleness, `DCMRPBUF` file format bumps v1 → v2.

Experiment file removed after the verdict was recorded. Next entry
will be the advantage-baseline implementation itself.

---

## 2026-04-17 22:02 CDT — Gradient clipping, weight decay, batch 4096 + lr 1e-3

**File:** `DrewsChessMachine/DrewsChessMachine/ChessTrainer.swift`,
`DrewsChessMachine/DrewsChessMachine/ContentView.swift`

Implements items #1, #2, #4, and #5 from the 17:23 CDT plan. Item #3
(advantage baseline) is **deferred at this commit** — MPSGraph as of
macOS 15 exposes no `stopGradient` / `detach` op in its public
headers, so implementing `(z − v.detached()) * −log p(a*)` cleanly
would require either a second autodiff pass to compute and subtract
the unwanted gradient contribution, or a two-run training step that
feeds v back as a placeholder (≈2× forward cost). Noted inline at
the policy-loss build site; will revisit once we pick an approach.

> **Status update — resolved 2026-04-17 22:46 CDT (`d745cfe`).** The
> deferral was lifted within the same evening. The 22:35 CDT
> `[EXP-DETACH]` experiment (`181f600`) confirmed `variableFromTensor`
> + `read` does *not* block autodiff, so the chosen approach is to
> capture `v(position)` during self-play inference (zero extra
> forward passes), persist it in the replay buffer, and feed it as
> an external `vBaseline` placeholder at train time — every
> placeholder is a leaf, which gives detach semantics for free. See
> the 22:46 CDT entry above for the full implementation.

**Implemented:**
- **Gradient clipping** (`ChessTrainer.gradClipMaxNorm = 5.0`). After
  autodiff, the global L2 norm of the flattened gradient vector across
  every trainable variable is computed inside the graph. Every
  per-variable gradient is then multiplied by
  `clipScale = maxNorm / max(globalNorm, maxNorm)`. Norms at or below
  5.0 are no-ops; spikes are capped to L2 = 5.0 exactly. The pre-clip
  global norm is added to `TrainStepTiming` and the
  `TrainingLiveStatsBox` rolling window as `gradGlobalNorm`, surfaced
  in `[STATS]` lines as `gNorm=…` so we can see whether clip events
  are occurring.
- **Weight decay** (`ChessTrainer.weightDecayC = 1e-4`). Decoupled
  (AdamW-style) L2 applied to every trainable variable including
  biases and BN params. The SGD update is
  `v_new = v − lr · (clipped_grad + c · v)`.
- **Batch size 1024 → 4096 + learning rate default 0.1 → 1e-3.**
  The replay-buffer sampler's pre-training guard already requires at
  least 200 k positions (20 % of 1 M capacity), which is ≫ 4096, so
  the ratio of fill-before-train stays the same. `@AppStorage`
  persists any user-overridden LR across launches — only the default
  moves.

**Log surface additions:**
- `[STATS]` periodic lines gain `gNorm=<rolling mean>` and
  `reg=(clip=5.0 decay=1e-4)`.
- `[STATS] arena-start` lines gain `gNorm=<rolling mean>`.

**Observed effect:** TBD — next Play-and-Train session will log the
rolling gNorm values plus the loss trajectory. If gNorm never
approaches 5.0, clipping is dormant (safe). If it occasionally hits
the ceiling during early training, the circuit breaker is doing its
job. Steady >5.0 would mean lr is too high for the current loss
landscape.

---

## 2026-04-17 21:47 CDT — Per-build counter, replay buffer persistence, richer log output (`165f1cf`)

**Files:** `DrewsChessMachine.xcodeproj/project.pbxproj`,
`generate-build-info.sh`, `BuildInfo.swift`,
`DrewsChessMachineApp.swift`, `ReplayBuffer.swift`,
`SessionCheckpointFile.swift`, `CheckpointManager.swift`,
`ContentView.swift`, `build_counter.txt` (new).

**Build counter:**
- New `PBXShellScriptBuildPhase` "Generate BuildInfo.swift" inserted
  before Sources in the target's build phases, `alwaysOutOfDate = 1`
  so it runs on every build (including incremental). Calls
  `generate-build-info.sh` via `${PROJECT_DIR}`.
- `generate-build-info.sh` increments `build_counter.txt` (seeded
  from `git rev-list --count HEAD + 1` on first run), emits
  `BuildInfo.swift` with `buildNumber`, `buildDate`, `buildTimestamp`,
  `gitHash`, `gitBranch`, `gitDirty`, and a `summary` string
  (e.g. `build 76 (d745cfe*) 2026-04-17` — `*` indicates dirty tree).
- `ENABLE_USER_SCRIPT_SANDBOXING = NO` on both Debug and Release so
  the script can read `.git/` from `${PROJECT_DIR}/..`.
- Title bar uses `BuildInfo.summary`.

**Replay buffer persistence:**
- New `write(to:)` / `restore(from:)` methods on `ReplayBuffer` with
  binary format `DCMRPBUF` v1. Header: 8-byte magic, `UInt32` version,
  pad, then `Int64` fields for `floatsPerBoard`, `capacity`,
  `storedCount`, `writeIndex`, `totalPositionsAdded`. Payload is
  oldest-first (size scales with `storedCount`, not `capacity`) and
  handles capacity changes across save/restore by keeping the newest
  `min(stored, capacity)` entries. Chunked I/O at 32 MB bounds peak
  memory on 1 M-position rings. *(v2 schema added later at 22:46 CDT;
  see that entry.)*
- New `stateSnapshot()` returns storedCount/capacity/writeIndex/
  totalPositionsAdded under the lock for consistent saving.
- New `bytesPerPosition` constant so the UI can compute buffer RAM.

**Session checkpoint additions:**
- `SessionCheckpointLayout.replayBufferFilename = "replay_buffer.bin"`
  in the `.dcmsession` directory.
- `SessionCheckpointState` gains optional fields for build metadata
  (`buildNumber`, `buildGitHash`, `buildGitBranch`, `buildDate`,
  `buildTimestamp`, `buildGitDirty`) plus replay-buffer summary
  (`hasReplayBuffer`, `replayBufferStoredCount`,
  `replayBufferCapacity`, `replayBufferTotalPositionsAdded`). All
  Optional for back-compat with pre-v1 session.json files.
- `LoadedSession` gains `replayBufferURL: URL?`, populated when the
  state flags `hasReplayBuffer == true` AND the file exists.
- `CheckpointManager.saveSession(...)` accepts an optional
  `ReplayBuffer`; writes it into the staging dir before atomic
  rename if the state says to. `loadSession(...)` populates
  `replayBufferURL` when appropriate.
- `ContentView.buildCurrentSessionState` populates build metadata +
  replay-buffer metadata. Manual save and post-promote autosave both
  pass the live `replayBuffer` through.
- `startRealTraining` resume path restores the buffer on a detached
  task *before* any worker starts, then logs
  `[CHECKPOINT] Restored replay buffer: stored=N/cap totalAdded=M
  writeIndex=K`.

**Log surface additions:**
- `[APP] launched build=… git=…[*] branch=… date=… timestamp=…`
- `[APP] session log: <path>` now also emitted to the log itself.
- Periodic `[STATS]` gains
  `ratio=(target/cur/prod/cons/auto/delay)`,
  `outcomes=(wMate/bMate/stale/50mv/3fold/insuf)`,
  `batch=`, `lr=`, `promote>=`, `arenaGames=`, `workers=`,
  `build=`, `trainer=`, `champion=`.
- `[ARENA]` params line gains `workers` and `build`.
- `[CHECKPOINT] Saved session` / `Autosaved session` include
  `build=… git=… replay=N/cap`.
- `[CHECKPOINT] Loaded session` includes saved-session `savedBuild=`
  and `replay=N/cap` (or `replay=none`).

Verified working on second build: `build_counter.txt` increments,
title bar updates, new log columns present.

---

## 2026-04-17 17:23 CDT — Planned stability + learning-speed upgrade (DESIGN, not yet implemented)

Reasoning captured in `chess-engine-design.md` → "Stability Enhancements and Learning-Rate Upgrades". Summary grid:

| # | Change | Value | Helps with | Phase |
|---|--------|-------|------------|-------|
| 1 | Gradient clipping (global L2 norm) | `max_norm = 5.0` | Caps per-step parameter change. Prevents 2026-04-15-style single-step blowup. Silent on healthy batches. | Safety |
| 2 | Weight decay (L2 on all params) | `c = 1e-4` | Persistent pressure against slow weight growth. Generalization. Prevents the conditions that prime runaway logits. AlphaZero / ResNet standard. | Safety |
| 3 | Advantage baseline (`z − v.detached()`) | replace raw z in policy loss | 5–20× reduction in policy-gradient variance. Moves in obvious wins/losses get near-zero gradient; surprise outcomes get strong gradient. Biggest single learning-speed lever. | Speed |
| 4 | Batch size | `1024 → 4096` | 2× additional gradient-variance reduction at zero throughput cost (self-play is the bottleneck). Supports higher lr. Peak RAM 8.7 GB → 17.4 GB, within 37 GB budget. | Speed |
| 5 | Learning rate | `5e-4 → 1e-3` | Square-root scaling with 4× batch growth; conservative vs Lc0's linear rule which would say 2e-3. | Speed |
| 6 | K (policy-loss coefficient) | keep `50` | Already proven to produce signal post-fix `1ec8a13`; clipping removes need for K-warmup. | — |
| — | Logit L2 regularization | **skipped** | Redundant with weight decay; weight decay has more side benefits. | — |
| — | Advantage clamp | **skipped** | `v ∈ [−1, +1]` via tanh, so advantage is already bounded in [−2, +2]. Clamping would suppress most informative surprise cases. | — |
| — | Buffer pre-fill | **keep 20%** | Only affects time-to-first-step, not steady state. | — |
| — | K warmup | **skipped** | Gradient clipping handles the same "initial gradient too big" problem. | — |

**Implementation order:** #1 → #2 → #3 → (#4 + #5 together). Each its own commit and CHANGELOG entry with observed effect.

---

## 2026-04-17 21:24 CDT — Linear-decay sampling schedule + game diversity tracker (`8ca529b`)

| Area | Before | After |
|---|---|---|
| `SamplingSchedule` shape | two-phase (`openingPlies`, `openingTau`, `mainTau`) | linear decay (`startTau`, `decayPerPly`, `floorTau`) |
| Self-play tau | `1.0` first 25 plies/player → `0.25` | `1.0 → 0.4`, linear at `0.03/ply` |
| Arena tau | `1.0` first 15 plies/player → `0.10` | `1.0 → 0.2`, linear at `0.04/ply` |
| `SessionCheckpointFile.TauConfigCodable` | two-phase fields | new schedule shape |
| `GameDiversityTracker` | — | rolling-window tracker of move-sequence hashes + divergence plies; shared across self-play workers, one per tournament |
| `CHANGELOG.md`, `chess-engine-design.md`, `sampling-parameters.md` | — | updated to describe new schedule + diversity design; CHANGELOG seeded |

The abrupt cliff between exploratory and sharp tau in the two-phase schedule made it hard to disentangle exploration-vs-exploitation effects from the step-function itself; a linear ramp gives a continuous exploration→sharpening trajectory, and the tracker lets us actually *measure* game diversity instead of inferring it from draw rate.

---

## 2026-04-17 14:13 CDT — Lowercase second word in chart labels (`0f812d2`)

Sentence-case instead of title-case in the chart grid headers.

---

## 2026-04-17 14:06 CDT — Reorganize chart grid layout, fix memory units, fix non-neg chart (`9114ec4`)

| Area | Change |
|---|---|
| Grid column order | Col 1 Loss Total / Policy / Value · Col 2 Policy Entropy / Non-Negligible Policy Count · Col 3 Progress Rate / Replay Ratio · Col 4 CPU % / GPU % · Col 5 App Memory (GB) / GPU RAM (GB) |
| Memory display | was "K MB" → **GB** |
| Non-Negligible Policy Count | fixed Y-axis `0…4096` via `.chartYScale(domain: 0...4096)` so the scale is always meaningful |

---

## 2026-04-17 13:43 CDT — Entropy shows 3 digits + percentage, GPU% from training step time (`8e2a5d7`)

| Chart | Before | After |
|---|---|---|
| Policy Entropy header | `8.30` | `8.301 (99.8%)` — 3 decimals plus `%` of max entropy `log(4096)` |
| GPU % | `task_gpu_utilisation` (returns 0 on Apple Silicon) | Δ cumulative training GPU ms per 1-second sample — fraction of wall time GPU was running training |
| Loss Policy / Loss Value labels | "CE weighted" / "MSE" | removed (CE-weighted value wasn't actually the displayed number; MSE was jargon) |
| Chart grid | — | now 5×2 = 10 charts: Progress Rate · Entropy · Loss Total/Policy/Value · Replay Ratio · Non-Neg Count · CPU% · GPU% · App Memory · GPU RAM |

---

## 2026-04-17 13:34 CDT — Chart labels: descriptive units, GPU RAM replaces GPU %, aligned non-neg threshold (`8900a3c`)

| Chart | Label / Unit |
|---|---|
| Progress Rate | "moves/hour" |
| Policy Entropy | "0=focused 8.3=uniform" |
| Loss Total / Policy / Value | "policy+value" / "CE weighted" / "MSE" |
| Replay Ratio | "train/move" |
| Non-Neg Count | whole number, "/ 4096" |
| GPU chart | **replaces broken GPU %** with GPU RAM (`MTLDevice.currentAllocatedSize`) — `task_gpu_utilisation` returns 0 on Apple Silicon |
| CPU chart | "%" |
| App / GPU Memory | "MB" |
| Forward-pass Non-Negligible count | threshold aligned to training diagnostic: `1/4096` (was `>1e-10`, misleadingly showing 4096/4096) |

---

## 2026-04-17 13:28 CDT — Auto-incrementing build number from git commit count (`48b6ab2`)

`generate-build-info.sh` writes `BuildInfo.swift` with `git rev-list --count HEAD`, current date, and short git hash. `ContentView` title bar reads from `BuildInfo` instead of `Bundle`. Optionally wired as an Xcode Run Script phase before Compile Sources.

---

## 2026-04-17 13:20 CDT — Fix CPU % to match Activity Monitor (`809c39c`)

| | Before | After |
|---|---|---|
| CPU sampling API | `proc_pid_rusage` `ri_user_time` | `task_info` `TASK_THREAD_TIMES_INFO` (user + system across all threads via `time_value_t`) |
| Observed | ~14% vs Activity Monitor ~560% on macOS 26 | matches Activity Monitor |

---

## 2026-04-17 13:16 CDT — Chart grid: visual separation, latest values, CPU/GPU/memory charts (`0003164`)

| Area | Change |
|---|---|
| Chart styling | card-style backgrounds, 1 px separator grid via background-color gap |
| Chart headers | now show latest value with units (e.g. "Policy Entropy 8.30 nats", "CPU 12.3%") |
| New charts | CPU %, GPU %, App Memory (MB) |
| Layout | 5 cols × 2 rows, chart height 75 pt |
| Data sources | existing `@State cpuPercent`, `gpuPercent`, `memoryStatsSnap` (heartbeat-polled) |

---

## 2026-04-17 13:09 CDT — Fix policy-loss scaling: drop `(w+1)` normalizer, set K=50 (`1ec8a13`)

**File:** `DrewsChessMachine/DrewsChessMachine/ChessTrainer.swift`

| | Before | After |
|---|---|---|
| Total loss | `(1000·pLoss + vLoss) / 1001` | `50·pLoss + vLoss` |
| Effective policy coef | ≈ 1 (division cancelled the 1000× boost) | **50** |
| Effective value coef | ≈ 1/1001 | **1** |
| Recommended LR pairing | — | drop `lr` from `1e-2` to `5e-4` so the shared trunk doesn't diverge under the stronger combined gradient |

The prior `(1000·pLoss + vLoss)/1001` divided both terms, so the effective coefficients were policy ≈ 0.999 and value ≈ 0.001 — the *opposite* of the intended boost. The policy head saw no amplification (entropy stuck near max `log(4096)` for 10k+ steps in multi-hour runs); the value head still learned because `(z−v)²` gradients are naturally large even at 1/1000 weight.

**Observed effect (1 h post-change, lr=5e-4, batch=1024):**
- `vLoss` dropping much faster than prior runs: 0.83 → 0.11 in 1 h (prior run's floor was ~0.29 after 19 h).
- `pEnt` shows first measurable directional movement: 8.3046 → 8.3034 → 8.3024 → 8.2986 → 8.2973 over 45 min.
- Arena #1 score 0.507, arena #2 score 0.495 — candidate now tracking near parity; monitoring for regression.

---

## 2026-04-17 13:08 CDT — Fix empty charts: call training chart sampler from progress rate 1 Hz tick (`9e7c3c3`)

The 1 Hz tick was only feeding the progress-rate chart. All other grid charts looked empty until the sampler was also invoked from that tick.

---

## 2026-04-17 13:05 CDT — Remove network status text block from main screen (`6c6dd96`)

Architecture/parameters info now only lives in the (i) popover.

---

## 2026-04-17 13:01 CDT — UI redesign: chart grid, compact title, deduplicated displays (`3d9fe76`)

| Area | Change |
|---|---|
| Title bar | replaces always-visible description with compact title + (i) info popover (architecture / parameters on demand) |
| Chart grid | 7-chart grid below main content during Play and Train: progress rate, entropy, loss total/policy/value, replay ratio, non-negligible count |
| Scroll | charts share synchronized horizontal scroll via existing `progressRateScrollBinding` |
| Dedup | removed duplicate learn-rate and ratio displays from training stats text; removed probes display |
| Window | min raised to 1400×780 for chart grid; padding reduced to 16 pt |
| **New diagnostic — Non-Negligible Policy Count** | GPU-side count of softmax entries above `1/4096` (uniform prob), averaged across batch; added to `TrainStepTiming`, `TrainingLiveStatsBox` rolling window; diagnostic only, no gradient impact; starts ~2048 with random init, drops as policy concentrates |
| New file | `TrainingChartGridView.swift` — `TrainingChartSample` struct sampled at 1 Hz from heartbeat, reusable mini-chart components, 4-column `LazyVGrid` |

---

## 2026-04-16 17:34 CDT — Document session restore coverage table in ROADMAP (`d2e6b43`)

Added the 14-field save/restore coverage matrix to ROADMAP so future audits can verify nothing is silently discarded again.

---

## 2026-04-16 17:08 CDT — Full session restore: counters, arena history, worker count, delays (`361c452`)

Audit found **10 of 14 session state fields were saved but silently discarded on resume**. Now all are restored.

| Layer | Fields added / wired |
|---|---|
| `SessionCheckpointState` (save, all Optional for back-compat) | Game result breakdown: `whiteCheckmates`, `blackCheckmates`, `stalemates`, `fiftyMoveDraws`, `threefoldRepetitionDraws`, `insufficientMaterialDraws`, `totalGameWallMs`. Step delay: `stepDelayMs`, `lastAutoComputedDelayMs` |
| Restore in `startRealTraining` | `ParallelWorkerStatsBox` seeded init (all counters + training steps). `TrainingLiveStatsBox.seed()` sets step counter so heartbeat doesn't overwrite. `tournamentHistory` rebuilt from `arenaHistory`. `selfPlayWorkerCount` restored. `trainingStepDelayMs` + `lastAutoComputedDelayMs` restored |
| UI affordance | Button label flips to "**Continue Training**" when a session is loaded |
| Verified (from recheck) | Loss normalization `(1000·pLoss + vLoss) / 1001` is consistent on save and restore; logged `policyLoss` is pre-normalization |

---

## 2026-04-16 16:57 CDT — Fix crash: snap off-ladder delay values to nearest rung (`f011203`)

The auto-adjuster produces arbitrary ms values (e.g. 258 ms) outside the step ladder `[0,5,10,15,20,25,50,…,2000]`. Toggling auto off and clicking the Stepper tried `firstIndex(of:)` and hit `preconditionFailure`. Fixed in two places: the auto-toggle-off path snaps to the nearest rung, and the Stepper binding itself gracefully snaps off-ladder values.

---

## 2026-04-16 16:52 CDT — Rewrite auto-delay: overhead from measured consumption, damped convergence (`6b8f043`)

Previous approaches all failed for the same reason: they couldn't accurately estimate per-step overhead (GPU + buffer locks + task scheduling + gate checks ≈ 560 ms total vs 277 ms GPU-only).

| Step | Formula / action |
|---|---|
| 1. Measure overhead | `overhead = batchSize / consumptionRate − currentDelay` — captures **all** overhead automatically via the 1-minute consumption window |
| 2. Floor overhead | floor at EMA of GPU step time so it can't collapse to 0 when delay exceeds stale cycle during a transition |
| 3. Target delay | `targetDelay = desiredCycle − overhead` |
| 4. Damping | 10 % per-step toward target; at ~3 steps/s converges in ~15 s without oscillating against the 60 s measurement window |

Simulated with live numbers (production = 1826/s, consumption = 1251/s, delay = 258 ms, emaGpu = 277 ms): converges delay to 0 ms in ~15 s, ratio reaches 1.0 after the 60 s window refreshes. Reverse direction (production drops) shows mild overshoot floored by EMA, corrects after one window. No oscillation in either case.

---

## 2026-04-16 12:16 CDT — Revert per-position CE clipping to isolate normalization change (`8e9abb3`)

Pulled the per-position clip from `cc9400b` so the effect of the `/1001` normalization (`6c67953`) could be measured in isolation.

---

## 2026-04-16 12:15 CDT — Normalize total loss so policy weight doesn't inflate effective LR (`6c67953`)

| | Before | After |
|---|---|---|
| Total loss | `valueLoss + 1000·policyLoss` | `(valueLoss + 1000·policyLoss) / 1001` |

Intent: preserve the 1000 : 1 gradient ratio between heads while keeping total gradient magnitude the same as the original unweighted sum. Without normalization, the ×1000 also multiplied the effective learning rate for the shared trunk by ~1000×, so `lr=1e-4` behaved like `lr=0.1` — cause of the `lr=0.1` session (20260416-121926) where `pLoss` went to `−3.9 × 10⁸` and `pEnt` collapsed from 8.30 to 0.53 within two hours.

*Note (corrected next day):* this normalizer turned out to be too heavy a hammer — see `1ec8a13` (2026-04-17 13:09 CDT) for the fix that drops the `/1001` and sets `K = 50`.

---

## 2026-04-16 12:05 CDT — Stats every 15 min after ramp-up, STATS line at arena start (`b6d8ac0`)

| | Before | After |
|---|---|---|
| Periodic STATS cadence | 30 s · 1 m · 2 m · 5 m · 15 m · 30 m · 1 h · then **hourly** | 30 s · 1 m · 2 m · 5 m · 15 m · then **every 15 min forever** |
| Arena start | (no dedicated log line) | `[STATS] arena-start` with losses/entropy/buffer, so trainer's state entering an arena is visible regardless of the fixed schedule |

Motivation: the hourly cadence left 60-minute gaps that hid divergence onset (exactly the failure mode of the `lr=0.1` blowup).

---

## 2026-04-16 11:48 CDT — Clip per-position policy CE, reset game stats on promotion, log post-promote (`cc9400b`)

| Change | Detail |
|---|---|
| Per-position CE clip | clip to `[0, log(4096)]` before `z`-weighting. Caps the gradient contribution from any single low-probability move at "maximally surprising". Prevents the unbounded-loss catastrophe that caused NaN weights after promotion (trainer at `pEnt=7.46`, new champion at `pEnt=8.28` produced diverse moves the trainer assigned near-zero probability to; ×1000 amplification pushed gradients to infinity) |
| Reset on promote | `ParallelWorkerStatsBox` game counters reset on promotion — panel reflects only the newly-promoted champion. Training step count + session anchor preserved |
| `[STATS]` post-promote | emitted immediately after promotion so the session log captures the post-promotion state without waiting up to an hour for the fixed ticker |

*(Clip was reverted in `8e9abb3` the next commit to isolate the normalization change.)*

---

## 2026-04-16 10:01 CDT — Rewrite delay auto-adjuster to eliminate oscillation (`1646ee4`)

The delta-accumulation approach (`newDelay = currentDelay + error`) oscillated wildly because adjustments were applied every step but the 60 s measurement window took minutes to reflect the new delay — classic high-gain + long-delay instability.

Replaced with direct computation from two independent, smooth inputs:

```
desiredCycle = batchSize / (targetRatio × productionRate)
delay        = max(0, desiredCycle − emaGpuTime)
```

- `productionRate`: 1-minute rolling window, smooth by construction.
- `emaGpuTime`: exponential moving average of per-step GPU duration (α = 0.05, ~20-step half-life). **Measured independently of the delay — no feedback loop through the measurement window.**

No accumulation, no delta, no oscillation. *(Superseded two commits later by `6b8f043`, which derives overhead empirically from end-to-end consumption rather than just GPU EMA.)*

---

## 2026-04-16 09:46 CDT — Smooth auto/manual delay transitions on toggle (`96e2f6b`)

| Toggle direction | Behaviour |
|---|---|
| auto **ON** | seed computed delay from current manual value so display doesn't jump to a stale value; adjuster moves it gradually once warmup window fills |
| auto **OFF** | inherit last auto-computed delay as the new manual Stepper value so training pace doesn't jump when user takes manual control |

Added public `computedDelayMs` setter on `ReplayRatioController` to support the ON-toggle sync.

---

## 2026-04-16 09:42 CDT — Persist auto-computed delay so next session starts where adjuster left off (`237c027`)

Added `@AppStorage lastAutoComputedDelayMs`. When auto-adjust is on, the controller seeds from this instead of the manual Stepper value; the heartbeat writes the latest computed delay back each tick. Prevents the delay from resetting to 50 ms on every session start when the adjuster had converged to a different value.

---

## 2026-04-16 09:40 CDT — Fix recheck issues: LR restore, delay oscillation, LR ND array reset (`19a6cbb`)

| # | Fix |
|---|---|
| 1 | Restore learning rate from saved session on resume (previously resumed sessions used global `@AppStorage` LR instead of session's saved value) |
| 2 | Fix circular delay computation in `ReplayRatioController`: old approach subtracted previous iteration's delay to estimate GPU time → feedback loop / oscillation. Replaced with `newDelay = currentDelay + (desiredCycle − measuredCycle)` delta controller. *(Superseded by `1646ee4`, `6b8f043`.)* |
| 3 | Recreate `lrNDArray` and `lrTensorData` in `resetNetwork()` so the new graph's LR placeholder maps to a fresh tensor-data wrapper instead of reusing the old graph's allocation |

---

## 2026-04-16 09:31 CDT — Raise training step delay cap from 500 ms to 2000 ms (`c4016b9`)

| | Before | After |
|---|---|---|
| `stepDelayLadder` max rung | 500 ms | 2000 ms |

---

## 2026-04-16 09:31 CDT — Persist step delay, replay ratio target, and learning rate across launches (`8711594`)

| Field | `@State` → `@AppStorage` |
|---|---|
| `trainingStepDelayMs` | ✓ |
| `replayRatioTarget` | ✓ |
| `trainerLearningRate` | ✓ (also written on LR text field `onSubmit`) |

---

## 2026-04-16 09:26 CDT — Fix LR text field: only apply on Enter, don't reformat mid-type (`89d9222`)

Replaced two-way `Binding` (reformatted on every keystroke) with plain `@State string + .onSubmit`. Invalid input reverts to current LR. Seeded at Play-and-Train start.

---

## 2026-04-16 09:24 CDT — Use 1-minute average step time for delay auto-adjustment (`e268bb2`)

| | Before | After |
|---|---|---|
| GPU step time source | per-step `stepTimeMs` parameter to `recordStepAndGetDelay` | derived from 1-minute consumption window: `total wall time / steps in window − current delay` |

Smooths out per-step variance instead of jittering on a single fast or slow step.

---

## 2026-04-16 09:16 CDT — Make learning rate adjustable while training is running (`beb9474`)

| | Before | After |
|---|---|---|
| LR representation | MPSGraph **constant** baked into the graph | MPSGraph **placeholder** fed each step via a pre-allocated scalar ND array |
| `trainer.learningRate` | `let` | `var` — writes take effect on the next step with no graph rebuild |
| UI | — | text field in Play-and-Train training panel; parses & applies on commit |

---

## 2026-04-16 07:31 CDT — Learning rate 0.01 → 0.1 (`5ca881b`)

One-line change. Subsequently proved catastrophic at the prevailing (unnormalized) `1000·pLoss + vLoss` total-loss formulation — session `20260416-121926` drove `pLoss` to `−3.9 × 10⁸` and `pEnt` from 8.30 to 0.53 over ~2 h. Root cause (effective shared-trunk LR ~1000× the nominal value) was diagnosed in `6c67953` (normalize by /1001) and properly fixed in `1ec8a13` (use `K=50`, drop normalizer).

---

## 2026-04-15 23:20 CDT — Fix auto-adjust warmup guard: use half-window threshold (`51b6f8c`)

| | Before | After |
|---|---|---|
| Warmup guard | `dt >= windowSeconds` (impossible — pruning keeps samples strictly younger than the window) | `dt >= windowSeconds * 0.5` (30 s) — enough data for a meaningful rate estimate while still skipping the initial buffer-fill period |

---

## 2026-04-15 23:11 CDT — Show ratio rates after 3 s, keep 60 s guard for auto-adjust only (`e6adf92`)

Split the two gates: display shows rates after 3 s of data; the auto-adjuster still waits for a 60 s baseline before touching the delay. Prior behaviour showed dashes for several minutes (buffer fill + 60 s window).

---

## 2026-04-15 22:43 CDT — Fix session load: make ratio fields truly Optional for old files (`200bfc5`)

Swift's synthesized `Codable` `init(from:)` calls `decode` (not `decodeIfPresent`) for non-optional `var` properties **even with defaults**. Changed `replayRatioTarget` and `replayRatioAutoAdjust` to `Double?` / `Bool?`; use sites unwrap with `?? 1.0` / `?? true`. Also improved `invalidJSON` error to include underlying `DecodingError` description and the first 2000 bytes of the file.

---

## 2026-04-15 22:40 CDT — Default replay-ratio fields in session.json for backward compat (`fe95771`)

Initial (incorrect) attempt at backward-compat: changed `let` → `var` with defaults. Did not work; see `200bfc5` above.

---

## 2026-04-15 22:20 CDT — Default file importer to the canonical save directories (`545e388`)

Load Model opens to `~/Library/Application Support/DrewsChessMachine/Models/`, Load Session to `.../Sessions/`, via `fileDialogDefaultDirectory` with a ternary on which importer is active.

---

## 2026-04-15 21:12 CDT — Normalize directory URL in session load to fix file-not-found (`628d80e`)

macOS file importer can return file-reference or bookmark URLs whose `appendingPathComponent` doesn't resolve to the expected child path. Reconstruct the directory URL via `URL(fileURLWithPath:isDirectory:true)` before building `champion`/`trainer`/`session.json` children, stripping metadata that breaks child resolution.

---

## 2026-04-15 20:59 CDT — Wait for full 60 s window before ratio auto-adjustment (`d8bfa12`)

During the first minute self-play is filling the buffer while training hasn't started (or just started), so the production/consumption ratio is meaninglessly skewed. Changed guard from `dt > 1s` to `dt >= windowSeconds` (60 s) in both `recordStepAndGetDelay` and `snapshot`. *(Relaxed to half-window in `51b6f8c`.)*

---

## 2026-04-15 20:53 CDT — Save/load checkpoints, replay ratio controller, and tuning changes (`4929d93`)

Large commit. Three substantive systems plus a hyperparameter bump.

### Checkpoint system

| File | Role |
|---|---|
| `ModelCheckpointFile.swift` | `.dcmmodel` binary format: trailing SHA-256, arch-hash validation, length-prefixed metadata JSON |
| `SessionCheckpointFile.swift` | `.dcmsession` directory: `champion.dcmmodel` + `trainer.dcmmodel` + `session.json` (counters, hyperparams, arena history) |
| `CheckpointManager.swift` | save/load orchestration; post-save verification (bit-exact weight round-trip + forward-pass round-trip on a throwaway scratch network); atomic writes via tmp+rename; save dir `~/Library/Application Support/DrewsChessMachine/` with timestamped never-overwrite filenames |
| UI | Save Session · Save Champion · Load Session · Load Model · Reveal Saves; autosave on arena promotion (default on) using pre-captured weights with zero post-return gate interaction to avoid deadlock |

### Replay ratio controller

| File / UI | Role |
|---|---|
| `ReplayRatioController.swift` | tracks 1-min rolling production (self-play) and consumption (training) rates; with auto-adjust on, computes training step delay that drives the ratio toward target (default 1.0). Prevents the training-outpaces-self-play divergence observed during device sleep |
| `ReplayBuffer.swift` | added `totalPositionsAdded` monotonic counter and `bytesPerPosition` for RAM display |
| UI | current ratio · target stepper · auto-adjust toggle; Step Delay shows "(auto)" and disables manual Stepper when auto is on; buffer line shows estimated RAM; 1-min generation and consumption rates shown |

### Threading / safety (from recheck)

- `WorkerPauseGate.pauseAndWait(timeoutMs:)` bounded variant prevents deadlock when workers exit before acknowledging a pause.
- Save Champion pauses worker 0 with timeout; disabled during arena and non-Play-and-Train busy modes.
- Chart scroll binding guards against multi-update-per-frame warning.

### Hyperparameters

| Parameter | Before | After |
|---|---|---|
| Learning rate | `1e-4` | **`0.01`** |
| Batch size | `256` | **`1024`** |
| Replay buffer | `500 000` | **`1 000 000`** |

Design-doc update: added Lc0 LR/batch-size scaling reference table link.

---

## 2026-04-15 15:42 CDT — Plan: model and session save/load with post-save verification (`46d4472`) (DESIGN)

Details v1 design for two file formats (`.dcmmodel` flat binary, `.dcmsession` directory), fixed Library save locations with never-overwrite history, autosave-on-promotion default-on, and per-save bit-exact round-trip verification (re-read + byte-compare + forward-pass compare) that runs automatically on every manual and auto save. No code changes — plan only, per the rule that features must be fully planned before implementation.

---

## 2026-04-15 07:34 CDT — Document bootstrap policy loss weighting and fused CE op (`1b4e9ec`)

Captures in `chess-engine-design.md`:

| Detail | Reason |
|---|---|
| 1000× policy-loss weight during pre-MCTS bootstrap | REINFORCE on a 4096-way softmax has per-logit gradient ~`1/(N·batch)`, three orders of magnitude below the value head |
| Use `MPSGraph.softMaxCrossEntropy` for policy loss | manual softMax→log NaNs under the 1000× gradient, and the stable-log-softmax fix using `reductionMaximum` crashes in `gradientForPrimaryTensor` because MPSGraph's autodiff has no gradient implementation for that op |
| Policy entropy keeps its own stable log-softmax | safe only because it's not reachable from `totalLoss` |

---

## 2026-04-15 07:25 CDT — Use fused `softMaxCrossEntropy` for policy loss (`745aa27`)

Manual stable log-softmax used `reductionMaximum`, which MPSGraph's autodiff has no gradient implementation for — `buildTrainingOps` crashed inside `gradientForPrimaryTensor` with "Op gradient not implemented" as soon as Play and Train created the trainer.

| | Before | After |
|---|---|---|
| Policy loss | manual `x - max(x) - log(sum(exp(x - max(x))))` + gather + multiply | `graph.softMaxCrossEntropy(.none)` + reshape + outcome-weighted multiply |
| Policy entropy diagnostic | manual stable log-softmax | unchanged (not in `totalLoss`, autodiff never walks it) |

---

## 2026-04-15 07:15 CDT — Scale policy loss 1000× and compute log-softmax stably (`b651e41`)

| | Before | After |
|---|---|---|
| Policy loss weight in `totalLoss` | `1 × pLoss` | **`1000 × pLoss`** |
| Log-softmax form | naive `softMax → log` (underflowed to `log(0) = -inf` under strong gradients, contaminating both `pLoss` and `pEnt` with NaN) | stable `logSoftmax = x − max(x) − log(sum(exp(x − max(x))))`; recover softmax for entropy as `exp(logSoftmax)` |

Motivation: bootstrap policy loss is REINFORCE on the played move over a 4096-way softmax, so per-logit gradient was ~1000× weaker than value's `(z−v)²`. `pEnt` sat at ~8.297 (uniform) for hours while `vLoss` converged normally.

---

## 2026-04-15 02:38 CDT — Reorder Progress rate chart layers: combined → self-play → training (`2c8b318`)

Draws the combined series first (bottom of Z-stack), then self-play, then training on top. Colors unchanged (combined green, self-play blue, training orange).

---

## 2026-04-15 02:28 CDT — Use native Swift Charts scrolling for Progress rate chart (`3c8bf75`)

| | Before | After |
|---|---|---|
| Zoom/pan | custom `MagnificationGesture + DragGesture` overlay | `chartScrollableAxes + chartXVisibleDomain + chartScrollPosition` (Swift Charts native: trackpad, mouse-wheel, keyboard) |
| Pinch zoom | available | **dropped** — fixed 10-minute window matches the "last 10 m" stats column |
| Follow-latest | implicit | explicit flag pauses 1 Hz auto-advance when user scrolls back, resumes within one sampler tick of the right edge |
| Net | — | 226 lines deleted, 74 added |

---

## 2026-04-15 02:26 CDT — Add Abort and Promote arena buttons, surface promoted model ID in logs (`2cc0127`)

| Button | Effect |
|---|---|
| Abort | ends tournament with **no promotion** regardless of score |
| Promote | ends tournament **early** and forcibly promotes the candidate regardless of score |

Decision is set-once via a new `ArenaOverrideBox` polled by `isCancelled` between games, so conflicting rapid clicks can't produce contradictory state. `TournamentRecord` now carries `promotedID`; stdout `[ARENA]` header, session log, and on-screen arena history all surface `PROMOTED=<id>` instead of bare `PROMOTED`.

---

## 2026-04-15 02:19 CDT — Exclude Play-and-Train setup delay from session rate denominators (`e4500b1`)

`ParallelWorkerStatsBox.sessionStart` was stamped at button-press time, so network builds, trainer reset, and weight copies baked a multi-second setup tax into every session average for the life of the session. Made `sessionStart` a lock-protected `var` and advanced once, right before the worker task group spawns.

---

## 2026-04-15 02:18 CDT — Add Progress rate chart and live %CPU/%GPU utilisation (`f8792a0`)

| Area | Change |
|---|---|
| New tab | third Play-and-Train board tab: Swift Charts line chart of rolling moves/hr for self-play, training, and combined; 1 Hz sampling, 3-min trailing window; pinch-zoom + drag-pan with reset |
| `ProcessUsageSample` | reads `proc_pid_rusage` (CPU ns) and `task_info(TASK_POWER_INFO_V2)` (GPU ns), sampled every 5 s from heartbeat; busy label shows live %CPU and %GPU alongside memory stats |

---

## 2026-04-15 01:55 CDT — Move Concurrency and Step Delay into their stats sections (`a81d6bd`)

The Workers and Step Delay Steppers lived in the top button row, detached from the numbers they affect. Both now render inside the stats panels: Concurrency Stepper = first row of Self Play column; Step Delay Stepper = first row of Training column. `playAndTrainStatsText` / `trainingStatsText` now return `(header, body)` splits so SwiftUI can inject the control `HStack`s between header and monospaced body.

| Default | Before | After |
|---|---|---|
| `initialSelfPlayWorkerCount` | 6 | **5** |
| `trainingStepDelayMs` | 0 ms | **50 ms** (so a fresh session doesn't let training starve the N self-play workers of GPU time) |

---

## 2026-04-15 01:45 CDT — Wire training-step delay into the worker loop (`103add5`)

Training worker now reads `stepDelayBox.milliseconds` at the bottom of each iteration and sleeps that long (skipping sleep at 0 ms), so Stepper clicks take effect on the very next step. Tightened `trainingStepDelayBinding` to crash on off-ladder current value instead of silently snapping (later relaxed in `f011203` after the auto-adjuster started producing off-ladder values). Cleared `trainingStepDelayBox` in session-end cleanup.

---

## 2026-04-15 01:40 CDT — Add `TrainingStepDelayBox` scaffolding for adjustable per-step pause (`33b6e36`)

Data plane only: `TrainingStepDelayBox`, `@State trainingStepDelayMs` / `trainingStepDelayBox`, and `stepDelayLadder` constant (fine 5 ms rungs at the low end, 25 ms rungs up to 500 ms) that the forthcoming Stepper will walk.

---

## 2026-04-15 01:39 CDT — Document idle-worker memory-vs-latency trade-off in ROADMAP (`9beafd0`)

Expands the N-worker entry to cover the current `initialSelfPlayWorkerCount` / `absoluteMaxSelfPlayWorkers` split, the live Stepper, the runtime `countBox.count==1` check for `GameWatcher` wiring, and — most importantly — **why idle workers stay allocated instead of being torn down**:

- Keeping all tasks and networks alive buys ≤ 50 ms live-tuning latency at the cost of ~180 MB idle network state plus ~74 MB player scratch at the ceiling.
- Alternative (release on shrink, rebuild on grow) would cost ~100–300 ms per `+` click for MPSGraph construction, first-run kernel JIT, and weight sync.

---

## 2026-04-15 01:33 CDT — Mark N-worker concurrent self-play as completed in ROADMAP (`6903d85`)

First entry under a new Completed section. Notes the replay-ratio motivation (~8.4× down to ~1–3× at the default N) and the ~12 MB per extra network memory cost.

---

## 2026-04-15 01:33 CDT — Spawn N concurrent self-play workers with live tuning and aggregate stats (`ae40697`)

| Area | Change |
|---|---|
| Self-play parallelism | up to `absoluteMaxSelfPlayWorkers = 16` concurrent workers; Stepper next to Run Arena for live N adjustment in `[1, 16]` |
| Worker 0 | uses existing champion network |
| Workers 1..N−1 | use dedicated secondary inference networks mirrored from the champion at session start and every arena promotion |
| Per-worker `WorkerPauseGate` | arena coordinator pauses exactly the workers whose networks a given sync point touches |
| Player reuse | each worker owns its own pair of reusable `MPSChessPlayer` instances surviving across games via `ChessMachine.beginNewGame`'s `onNewGame` calls |
| `WorkerCountBox` | self-play tasks read current N between games without hopping to the main actor |
| Live display | runtime decision `liveDisplay = isWorker0 && countBox.count == 1`; toggling N between 1 and >1 re-enables/suppresses the animated board on the next game |
| `ParallelWorkerStatsBox` | per-outcome counters, total game wall time, rolling 10-minute window — single source of truth for aggregate self-play stats |
| UI | new `playAndTrainStatsText` with Concurrency row, lifetime totals, 10-min column beside Avg move / Avg game / Moves/hr / Games/hr; column headers carry model IDs ("Self Play [id]" / "Training [id]"); dropped old Trainer ID / Champion ID rows |
| Top busy row | replaces rate line with total session time + memory-stats line (app footprint, GPU allocated/target, total unified RAM), refreshed out-of-band every 10 s |
| Training Run Totals | rates now computed against `Date().timeIntervalSince(parallelStats.sessionStart)` so moves/sec is directly comparable to the self-play column; Last Step trimmed to Total + Entropy; removed Avg GPU, Min step, Max step, Proj 250× |
| Board slot | "N = X concurrent games" overlay when N>1 and not in Candidate test mode |

---

## 2026-04-15 01:33 CDT — Disable MainThreadChecker and performance antipattern checker in scheme (`0b50476`)

Empty body. Scheme-only change to silence false positives from the heavy parallel worker refactor.

---

## 2026-04-14 22:03 CDT — Scale replay buffer to 500 k positions with a proportional warmup gate (`d3510c4`)

| | Before | After |
|---|---|---|
| Replay buffer capacity | 50 000 | **500 000** |
| Warmup threshold | fixed `16 × batch` (4 096 positions, ~8 % of old ring) | `max(25 000, capacity / 5)` — 20 % fractional gate for large rings, meaningful absolute floor for small ones. At 500 k this holds off training until 100 k positions have landed |

Gives the trainer a substantially more diverse / decorrelated warmup cohort and reduces the window where a tiny initial batch can dominate early gradient updates.

---

## 2026-04-14 22:02 CDT — Clear pending arena trigger on arena completion (`66fd879`)

Training worker runs in parallel with the arena and polls `shouldAutoTrigger` against the pre-arena `_lastArenaTime`. If the interval elapsed mid-arena, it stamped `_pending` while the arena was still running, and without clearing the flag on completion the coordinator would fire a back-to-back arena the instant it looped back. `recordArenaCompleted` now resets `_pending` alongside the last-arena timestamp.

---

## 2026-04-14 22:02 CDT — Widen self-play and arena opening sampling windows (`eb0294b`)

| Schedule | Opening plies/player | Opening τ | Main τ |
|---|---|---|---|
| Self-play before | 8 | 1.0 | 0.5 |
| **Self-play after** | **25** | **1.0** | **0.25** |
| Arena before | 4 | 1.0 | 0.1 |
| **Arena after** | **15** | **1.0** | **0.1** (kept) |

Longer exploratory window gives the replay buffer broader opening + early-middlegame coverage; sharper main τ produces fewer drawn technical endings and more non-zero `z` labels. Arena widening prevents color-alternating tournaments from collapsing into identical deterministic lines before the scoring phase.

---

## 2026-04-14 22:02 CDT — Add policy entropy as a training diagnostic (`b29e6db`)

Wires a policy-entropy tensor through the trainer graph alongside the existing policy and value loss outputs, tracks its rolling mean in the live stats box, and surfaces it in the session `STATS` line and the training snapshot display. **Not part of `totalLoss`** — diagnostic for spotting policy collapse (entropy → 0) or stuck-at-uniform learning failure (entropy pinned near `log(4096) ≈ 8.318`).

---

## 2026-04-14 19:58 CDT — Sampling schedules, model IDs, session logging, hot-path cleanup (`5c9a567`)

Multi-area commit.

### Sampling schedules (`sampling-parameters.md`)

Two-phase τ applied in `MPSChessPlayer.sampleMove`:

| Schedule | Opening plies/player | Opening τ | Main τ |
|---|---|---|---|
| Self-play | 8 | 1.0 | **0.5** |
| Arena | 4 | 1.0 | **0.1** |
| Play Game / Forward Pass | — | — | flat 1.0 via `.uniform` preset (legacy behaviour unchanged) |

Fixes the arena-pinned-at-0.5 stall: τ = 1 sampling was drowning candidate policy preferences in noise, and the high-draw rate left no decisive-game signal to measure improvement.

### Model IDs (`ModelID.swift`)

| Property | Detail |
|---|---|
| Shape | `yyyymmdd-N-XXXX` (per-date counter in `UserDefaults`, 4-char base62 random suffix for cross-machine dedup) |
| Mint at | Build Network, Play-and-Train start, arena snapshots |
| Inherit by | probe copies, champion→arenaChampion snapshots, promotions |
| Trainer | now forks from champion's weights at Play-and-Train start (previously re-randomized → arena at step 0 was two unrelated random inits) |
| Surfaced in | training status text, `[ARENA]` stdout log (trainer + champion + arena-candidate IDs) |

### Session logging (`SessionLogger.swift`)

Thread-safe file logger at `~/Library/Logs/DrewsChessMachine/dcm_log_yyyymmdd-HHMMSS.txt`, fsync'd per line. Hooks: `APP launched`, `BUTTON` taps for every main-row button, `ARENA` start + 3-line result, plus a **STATS** ticker at 30 s / 1 m / 2 m / 5 m / 15 m / 30 m / 60 m then hourly with steps / games / buffer fill / rolling losses / IDs.

### Hot-path allocation cleanup

| Layer | Change |
|---|---|
| `ReplayBuffer` | rewritten around flat `UnsafeMutablePointer` rings (boards / moves / outcomes) with pre-allocated reusable sample batches; `TrainingBatch` now non-owning views |
| `BoardEncoder.encode(_:into:)` | overload writes into caller-owned buffer so `MPSChessPlayer` keeps per-game tensor scratch alive (no fresh `[Float](1152)` per ply) |
| `ChessNetwork` | caches feeds dict and target tensor list at init; exposes raw-pointer readback scratches so inference returns `UnsafeBufferPointer` with zero per-call allocation |
| `TournamentDriver` | no longer collects `TrainingPositions`; players push directly into their attached `ReplayBuffer` at game end, saving ~184 MB per arena evaluation |
| Xcode scheme | LaunchAction → Release build, no debugger attached — `Cmd+R` runs optimized code |

---

## 2026-04-13 17:39 CDT — Reuse MPSNDArray pools across inference and training hot paths (`bfc9662`)

Pre-allocate `MPSNDArray` + `MPSGraphTensorData` wrappers once and write new values in place each call, so per-move inference and per-step training no longer allocate `MPSGraphTensorData`, `Data`, or `NSNumber` shape arrays on the hot path.

| Layer | Detail |
|---|---|
| `ChessNetwork` | one `[1,18,8,8]` inference-input pool reused by `evaluate()`; one zero-filled dummy shared by `exportWeights`/`loadWeights`; one pool entry per persistent variable for `loadWeights` to write through; new `boardSizeMismatch` error and `writeInferenceInput` / `writeFloats` helpers |
| `ChessTrainer` | per-batch-size `BatchFeeds` cache that lazily builds the three placeholder ND arrays on first use and reuses them forever (or until `resetNetwork` clears the cache); warmup step at each new batch size pays allocation; timed loop runs allocation-free |
| Float16 paths | `fatalError` loud until a reused half-scratch is added |

---

## 2026-04-13 17:17 CDT — Run Play and Train self-play and training as concurrent workers with a four-network arena (`00c3d10`)

Replaces the sequential Play-and-Train driver (alternating one game → 10 training steps) with **three concurrent tasks inside a `withTaskGroup`**:

| Worker | Role |
|---|---|
| Self-play | tight loop, one game at a time on the champion; streams positions into replay buffer; records per-game counts on shared `ParallelWorkerStatsBox`; pauses at `selfPlayGate` for the two brief champion-write moments each arena |
| Training | tight-loop single-step SGD on the trainer (no more blocks of 10); samples batches each step; records timings + parallel counters; calls candidate test probe between steps; auto-triggers arena every 30 wall-clock min via shared `ArenaTriggerBox`; pauses at `trainingGate` for the one brief trainer-read moment at each arena start |
| Arena coordinator | polls the trigger; on fire runs `runArenaParallel` on a **fourth network** ("arenaChampionNetwork", built lazily, cached across sessions) copied from the real champion once at arena start so self-play's champion stays free throughout the ~80 s tournament; candidate probe and arena mutually excluded via `ArenaActiveFlag` |

Coordination types:

| Type | Purpose |
|---|---|
| `WorkerPauseGate` | request/ack gate for brief per-worker pauses with cancellation-safe spin-waits |
| `ArenaTriggerBox` | trigger inbox + last-arena timestamp for both 30-min auto and Run Arena button |
| `ArenaActiveFlag` | probe/arena mutex |
| `ParallelWorkerStatsBox` | shared rolling counters for live pos/sec rate display |

| | Before | After |
|---|---|---|
| Arena cadence | every **5000 SGD steps** | **wall-clock, every 30 min** (`secondsPerTournament`); `stepsPerTournament` removed |
| Run Arena button | — | visible only during Play and Train, disabled while arena in flight |
| Busy label (parallel) | — | "Self-play: N games, M moves/s · Train: K steps, L moves/s · Buf: B" |
| Training Run Totals | Steps/sec only | Moves/sec + Moves/hr alongside Steps/sec |
| Stop latency | unbounded during arena | ≤ one in-flight self-play game (~400 ms) or one arena game (~400 ms) via `CancelBox` in `withTaskCancellationHandler` wrapping the detached tournament driver |

---

## 2026-04-13 15:45 CDT — Show live elapsed time in arena busy label and history (`f3bb253`)

`TournamentProgress` gains a `startTime` carried through every per-game update; busy label computes elapsed wall clock on each render via `Date().timeIntervalSince(startTime)`. `TournamentRecord` gains `durationSec` displayed in Arena History. New `formatElapsed` renders "12.3s" under a minute and "m:ss" from one minute on.

---

## 2026-04-13 15:41 CDT — Add arena tournament every 5000 SGD steps with 0.55 promotion threshold (`998f0d5`)

| Parameter | Value |
|---|---|
| Arena cadence | every **5000 individual SGD steps** |
| Arena games | **200**, alternating colors |
| Promotion threshold | **candidate score ≥ 0.55** (AlphaZero paper) — draws = 0.5 win |
| Candidate source | trainer snapshot synced into dedicated candidate inference network at top of arena |
| Champion source | self-play network |

`TournamentDriver` gains three optional parameters: `collectTrainingPositions` (to skip ~184 MB position accumulation arena callers don't need), `isCancelled` (cooperative cancellation from outer tasks that can't propagate `Task.isCancelled`), and `onGameCompleted` (live per-game progress). All defaulted. New types in `ContentView`: `TournamentProgress`, `TournamentRecord`, `TournamentLiveBox` (`NSLock`-protected). `runArenaTournament` is an `@MainActor async` helper called at gap point #2 when `trainingStats.steps − lastTournamentAtStep >= 5000`. Clicking Stop during an arena aborts at the next per-game boundary (worst ~400 ms). **Promotion gate requires full 200 games AND score ≥ 0.55**; partial arenas from cancellation cannot promote.

---

## 2026-04-13 14:58 CDT — Add Play and Train Candidate test probe with proper BN running stats and weight transfer (`f847bc2`)

Builds out the Play and Train mode into **three distinct networks**:

| Network | Role |
|---|---|
| Champion (`self.network`) | untouched by training; drives self-play, Play Game, Play Continuous, Run Forward Pass. Reserved for future arena-based promotion |
| Trainer (`trainer.network`) | SGD-updated with training-mode BN that EMA-tracks running mean/variance alongside weight updates (**momentum 0.99**). Each `trainStep` runs SGD assigns + BN running-stat assigns in the same graph execution |
| Candidate inference | new cached `ChessMPSNetwork`, built lazily on first Play-and-Train session. Inference-mode BN so outputs are calibrated like a deployed network. Used only by Candidate test probe — no self-play through it |

Weight transfer (conv/FC weights + BN γ/β + running stats) trainer → candidate inference happens inside `fireCandidateProbeIfNeeded` right before each probe's forward pass. In Game run mode zero transfers fire; in Candidate test mode one ~11.6 MB copy every 15 s (or immediately on drag / side-to-move / Board picker flip). `ChessNetwork` gains `exportWeights` and `loadWeights` sharing a per-variable placeholder + assign op pair built once at init time (load is a single atomic graph execution).

UI: button rename "Train on Self-Play" → **"Play and Train"**; Board segmented picker (Game run | Candidate test) visible only during Play and Train; probe counter line in training stats; more decimal places across inference text output so drift is readable at early-training magnitudes.

---

## 2026-04-13 11:47 CDT — Add interactive forward-pass editor, split training loss, wire self-play replay buffer (`dfc6166`)

| Area | Change |
|---|---|
| Forward-pass board | free-placement editable: drag to move, drop off-board to delete, side-to-move picker flips perspective. Auto-reruns inference on every edit; persists across Build Network and mode switches. `ChessRunner.evaluate` now takes the display board + flip flag so arrows land on correct squares regardless of side-to-move |
| Training loss | **split into policy and value** components. Separate rolling windows on `TrainingLiveStatsBox`; UI shows both in self-play mode and Last Step block |
| Default LR | **`1e-3` → `1e-4`** as a stability-diagnostic baseline — the bounded value MSE (`[0,4]` via tanh + `{−1,0,1}` outcomes) makes value oscillation a genuine instability signal, distinct from policy-term metric noise |
| Replay buffer wiring | `MPSChessPlayer` optionally pushes labeled game positions into a shared `ReplayBuffer` at game end. Non-training paths (Play Game, Play Continuous) default to `nil` and are unchanged |

---

## 2026-04-13 00:18 CDT — Add training-mode batch-size sweep with empirical memory guard (`7334110`)

`ChessTrainer` builds a training-mode copy of the network, runs SGD steps on synthetic inputs, and times steady-state training throughput across a ladder of batch sizes. The sweep walks the full ladder but refuses to run any batch size whose predicted resident footprint exceeds **75 %** of `min(recommendedMaxWorkingSetSize, maxBufferLength)`, or whose largest single buffer would exceed `maxBufferLength`. Prediction comes from a least-squares linear fit over `(batch, peak phys_footprint)` pairs already observed during the same sweep — no per-architecture fudge factors.

Process-wide `phys_footprint` is sampled by the UI heartbeat (~10 Hz) plus once at the start and end of each row so transient spikes don't slip past us the way `MTLDevice.currentAllocatedSize` would. Table header reports device caps; skipped rows show predicted RAM + largest buffer + reason.

---

## 2026-04-12 08:59 CDT — Hold ChessMachine strongly during delegate dispatch (`8980f65`)

`DelegateBox` held the machine via a weak reference → race when the final `gameEnded` event was dispatched. In continuous play this manifested as a stuck spinner after Stop. Fix: hold machine strongly in the box (each box keeps the machine alive only until its own queued event is delivered); delegate stays weak so events become no-ops when the UI owning it goes away.

---

## 2026-04-12 08:46 CDT — Detect threefold repetition draws (`e06e423`)

FIDE Article 9.2. `PositionKey` = piece placement + side-to-move + all four castling rights + en passant target. `ChessGameEngine` maintains a `[PositionKey: Int]` tally seeded with the starting position. Each `applyMoveAndAdvance` clears the table on `halfmoveClock == 0` (pawn move / capture → no prior position can recur after an irreversible move) then inserts/increments. Count of 3 triggers `.drawByThreefoldRepetition`. Wired through every `GameResult` switch. En passant included verbatim (not FIDE's strict "only if a capture is actually playable" exception — can miss a small number of legitimate draws but never declares a wrong one).

---

## 2026-04-12 02:20 CDT — Use active play time as the basis for session rates (`a9c5aed`)

Games/hr was visibly drifting downward during an in-progress game because the denominator was wall-clock since session start — even though `totalGames` only changed at game-end, the displayed rate fell throughout each game. Replaced with active-play stopwatch: `Snapshot` tracks `activePlaySeconds` (cumulative) + `currentPlayStartTime`; `setPlayingLocked(_:)` toggles `isPlaying` and starts/stops the stopwatch atomically; Games/hr is now discrete `totalGames / (totalGameTimeMs / 3.6e6)` updated only inside `gameEnded`. Moves/hr stays live but uses `activeSeconds()` as denominator. Session "Time" also shows `activeSeconds()`.

---

## 2026-04-12 02:09 CDT — Smooth live stats display and broaden insufficient-material draws (`6773c0a`)

| Area | Change |
|---|---|
| Moves/hr | now counts in-progress moves so rate updates smoothly (was sagging during game, snapping at game-end) |
| Session Moves | shows live count |
| Per-game `moveCount` reset | atomic inside `gameEndedWith` |
| Last Game section | hidden during continuous play |
| Session Time | wall-clock elapsed since session start, not sum of completed game durations |
| Formatting | comma-group Games, Moves, Moves/hr, Games/hr; indent White/Black wins under Checkmate |
| Insufficient material | now catches **K+B(s) vs K+B(s)** when every bishop sits on a single color complex (FIDE 5.2.2). Still correctly does **not** flag K+N+N vs K (FIDE only requires forced draws, KNN-K mate is technically reachable with cooperation) |

---

## 2026-04-12 01:04 CDT — Tighten game loop, decouple UI from inference throughput, add ROADMAP (`8ccf59d`)

Perf + threading + code-review commit.

| Area | Change |
|---|---|
| Legal-move generation | **once per ply** (was 3×) via threading through player call + `ChessGameEngine.applyMoveAndAdvance` |
| `GameState.board` | nested `[[Piece?]]` → flat `[Piece?]` of 64 — single CoW copy per ply |
| `ChessMachine.beginNewGame` | throws if game in progress (was silent cancel); stop-after-current documented on `cancelGame` |
| Delegate callbacks | serial `userInteractive DispatchQueue` (`drewschess.delegate`); game loop fires-and-forgets |
| `GameWatcher` | no longer `@Observable`; lock-protected state exposed via `snapshot()`; `ContentView` polls via 100 ms `Timer.publish + onReceive` heartbeat — **decouples UI redraw rate from game throughput** |
| `gameWatcher` | `@State` (was `let` — would reconstruct across view rebuilds, breaking weak delegate refs) |
| `playSingleGame` | refreshes snapshot synchronously after `markPlaying(true)` so the button disables before a fast double-click |
| Policy softmax | moved **out of MPSGraph** (network emits logits); `MPSChessPlayer.sampleMove` does numerically-stable softmax over only legal-move logits |
| `ChessNetwork` | data type switchable: `makeWeightData` / `readFloats` branch on `dataType`, converting via Accelerate vImage for float16 |
| He init | vectorized with `vDSP`/`vForce` (Box-Muller via `vvlogf`, `vvsqrtf`, `vvcosf`, `vDSP_vmul`) |
| `ChessRunner` | takes network in `init`; `ContentView` no longer builds twice; build errors surface via `Result` |
| `MoveGenerator.applyMove` | no longer silently no-ops on missing source square; callers pass legal moves |
| Canvas | redraws asynchronously |
| Dead code | removed `ChessGameEngine.playGame`, `isInCheck`, `ChessGameError.illegalMove`, `ChessRunner.networkBuildTimeMs` |
| ROADMAP | added with deferred items: bitboard repr, engine-level legality validation, compiled `MPSGraphExecutable`, fused mask+softmax in graph, partial heap / quickselect for top-k |

---

## 2026-04-10 22:13 CDT — Add macOS chess engine app with MPSGraph neural network (`bab8654`)

First running engine. Complete forward-pass and self-play scaffolding.

| Component | Detail |
|---|---|
| Neural network | 18×8×8 input, 128-channel stem, 8 residual blocks, policy head (4096 moves) + value head, ~2.9 M parameters via MPSGraph |
| Board encoder | general `GameState → tensor` with perspective flip for black, He weight initialization |
| Legal move generator | full chess rules: castling, en passant, promotion, pin detection, check/checkmate/stalemate |
| Game engine | `ChessGameEngine` (rules), `ChessMachine` (orchestration) with async game loop, delegate callbacks, per-move timing |
| Players | `MPSChessPlayer` (policy sampling + training data recording), `RandomPlayer`, `NullPlayer` |
| Tournament driver | multi-game series, alternating colors, stats, training position aggregation |
| UI | SwiftUI with board visualization (SVG piece assets), move arrows with gradient opacity, tensor channel overlay browser, live game display, comprehensive stats (timing, win/loss/draw breakdown by type, games/hr, moves/hr), continuous play mode |

---

## 2026-04-10 17:36 CDT — Fix MPSGraphTensorData initializer to use correct API (`1f42d3a`)

`MPSGraphTensorData` requires `(device:data:shape:dataType:)`, not `[Float]` directly. Added `makeTensorData` helper that converts `[Float]` to `Data` via `withUnsafeBytes`. Fixed training step and inference function to use correct `MPSGraphDevice` and `Data` types.

---

## 2026-04-10 14:53 CDT — Update MPSGraph primitives doc to match design doc (`50ea141`) (DOC)

| Fix | Detail |
|---|---|
| Input planes | `19 → 18` throughout (placeholder, weights, shapes, inventory) |
| Parameter count | `~2.6 M → ~2.9 M` |
| MCTS | all references removed; one-hot move targets |
| FC biases | added to policy head, value head FC1, value head FC2 |
| API fixes (3) | `MPSGraphAdamOptimizer class → graph.adam()` method · `graph.crossEntropyLoss → graph.softMaxCrossEntropy` (with Apple's `reuctionType` typo) · `graph.meanSquaredError → manual subtract/multiply/mean` |
| Batch norm | compute mean/variance from batch via `graph.mean` / `graph.variance` |
| Illegal-move masking | moved from graph ops to CPU-side Swift |
| Stem conv | now shows both `3x3x18` (stem) and `3x3x128` (tower) |
| Pseudocode | replaced with real implementations (`readBytes`, `guard let`s, `withUnsafeMutableBytes`) |
| API verification | all 14 MPSGraph APIs verified against framework headers |

---

## 2026-04-10 14:16 CDT — Overhaul chess engine design doc (`f5b03eb`) (DOC)

| Area | Change |
|---|---|
| Input tensor | `19 → 18` planes (removed dead opponent en-passant plane) |
| Plane list | converted to table format |
| Parameter counts | fixed with per-section breakdowns (total ~2.9 M, not ~2.6 M) |
| MCTS | removed from initial design, moved to Future Improvements with full reference material |
| Self-play loop | updated for pure network play (no tree search) |
| New explanations | convolution math, batch norm with symbols, skip connections, ReLU placement, padding mechanics, width-vs-depth trade-off |
| Structure | architecture flow (stem/tower/heads) first; math detail after; internal anchor links |
| Pipeline | PyTorch → **Swift + MPSGraph** implementation stack |

---

## 2026-04-04 12:23 CDT — add `__PUBLIC_REPO` (`f31a598`)

Marker file for repository visibility.

---

## 2026-04-04 12:13 CDT — Initial commit (`1a8f51e`)

Empty repository bootstrap.
